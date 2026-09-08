"""Route J: V-JEPA 2 as a WORLD MODEL for time-series videos (latent prediction + pixel readout).

Same videos, masks, data stream and evaluation as Route B (pretrain_route_b.py). The difference
is the objective: instead of a pixel decoder attending over mask tokens, the ENCODER encodes the
visible (context) tubelets, the PREDICTOR predicts the REPRESENTATIONS of the hidden future
tubelets, and a per-token READOUT head maps those predicted representations to raw pixels.

  loss = L_jepa  (smooth-L1 between predicted and EMA-target-encoder representations of the
                  future tokens; targets are layer-normalised, no gradient)
       + L_pix   (MSE between readout(pred) and the raw pixels of the future tokens)

Arms (identical data / masks / optimiser / steps):
  vjepa2_wm       facebook/vjepa2-vitl-fpc64-256 encoder + predictor init, EMA targets
  vjepa2_wm_rand  same architecture, random init                         (condition i)
  vjepa2_pixonly  pretrained init, lambda_jepa = 0 (no latent objective)  (does the world-model
                                                                            objective matter?)
The future frames enter ONLY the EMA target encoder (targets) and the pixel loss; the online
encoder never sees them (context_mask), so the forecaster is causal by construction.
"""
import argparse, copy, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pretrain_route_b as B
import videomae_patch_utils as U

VJEPA_CKPT = "facebook/vjepa2-vitl-fpc64-256"
TOK_PER_TUBELET = B.GH * B.GH


def token_index_sets(mask_row):
    """bool [N_TOK] (True = hidden) -> (context idx, target idx) as int64 tensors.
    Context = visible tokens; target = hidden tokens of the FUTURE tubelets only (the hidden
    LEAD tubelets, if any, are in neither set: they are just absent)."""
    m = mask_row.reshape(B.TT, TOK_PER_TUBELET)
    hidden_t = m.all(1)
    ctx = torch.nonzero(~mask_row.reshape(-1)).flatten()
    # future tubelets = trailing run of hidden tubelets
    t = B.TT - 1
    fut = []
    while t >= 0 and hidden_t[t]:
        fut.append(t); t -= 1
    fut = sorted(fut)
    tgt = torch.cat([torch.arange(f * TOK_PER_TUBELET, (f + 1) * TOK_PER_TUBELET) for f in fut]) if fut else torch.zeros(0, dtype=torch.long)
    return ctx, tgt, len(fut)


class WorldModel(nn.Module):
    def __init__(self, init="pretrained", lam_jepa=1.0, ema=0.998, grad_ckpt=False):
        super().__init__()
        from transformers import VJEPA2Model, VJEPA2Config
        # crop_size sets the RoPE grid (tokens per frame row); our videos are 224 -> 14x14 patches
        cfg = VJEPA2Config.from_pretrained(VJEPA_CKPT, crop_size=B.IMG, frames_per_clip=B.NF)
        if init == "pretrained":
            self.model = VJEPA2Model.from_pretrained(VJEPA_CKPT, config=cfg)
        else:
            torch.manual_seed(0)
            self.model = VJEPA2Model(cfg)
        self.cfg = cfg
        D = cfg.hidden_size
        self.readout = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(),
                                     nn.Linear(D, B.TS * B.PS * B.PS * 3))
        self.target_encoder = copy.deepcopy(self.model.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.lam_jepa, self.ema = lam_jepa, ema
        if grad_ckpt and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

    @torch.no_grad()
    def ema_update(self, tau=None):
        tau = self.ema if tau is None else tau
        for pt, po in zip(self.target_encoder.parameters(), self.model.encoder.parameters()):
            pt.mul_(tau).add_(po.detach(), alpha=1 - tau)

    def encode_context(self, vid, ctx_idx):
        """run the ONLINE encoder on the context tokens only (HF's VJEPA2Model.forward encodes the
        whole clip and masks afterwards, which would leak the future into the context tokens).
        RoPE position ids are the true global token ids, as in V-JEPA's masked pretraining."""
        enc = self.model.encoder
        h = enc.embeddings(vid)                                              # [B, N, D]
        Bn, D = h.shape[0], h.shape[-1]
        pos = ctx_idx[None].expand(Bn, -1).to(h.device)
        h = torch.gather(h, 1, pos[..., None].expand(-1, -1, D))
        for layer in enc.layer:
            h = layer(h, pos, None, False)[0]
        return enc.layernorm(h), pos

    def predict_repr(self, vid, ctx_idx, tgt_idx):
        """vid [B,T,3,H,W] normalised -> predicted representations of tgt tokens [B, n_tgt, D].
        The predictor gathers context rows by global index, so the context encoding is scattered
        into a full-length buffer (non-context rows are zeros and never gathered)."""
        h_ctx, pos = self.encode_context(vid, ctx_idx)
        Bn, D = h_ctx.shape[0], h_ctx.shape[-1]
        full = torch.zeros(Bn, B.N_TOK, D, device=h_ctx.device, dtype=h_ctx.dtype)
        full = full.scatter(1, pos[..., None].expand(-1, -1, D), h_ctx)
        out = self.model.predictor(encoder_hidden_states=full, context_mask=[pos],
                                   target_mask=[tgt_idx[None].expand(Bn, -1).to(vid.device)])
        return out.last_hidden_state

    @torch.no_grad()
    def target_repr(self, vid, tgt_idx):
        h = self.target_encoder(pixel_values_videos=vid).last_hidden_state          # [B, N, D]
        h = h[:, tgt_idx.to(vid.device)]
        return F.layer_norm(h, (h.shape[-1],))

    def forward_loss(self, vid, mask_row, hp):
        ctx_idx, tgt_idx, n_fut = token_index_sets(mask_row)
        pred = self.predict_repr(vid, ctx_idx, tgt_idx)                                 # [B, n_tgt, D]
        pix = self.readout(pred).float()                                               # [B, n_tgt, 1536]
        labels = U.patchify(U.unnormalize(vid)).reshape(vid.shape[0], B.N_TOK, -1)[:, tgt_idx.to(vid.device)].float()
        l_pix = F.mse_loss(pix, labels)
        if self.lam_jepa > 0:
            tgt = self.target_repr(vid, tgt_idx)
            l_jepa = F.smooth_l1_loss(pred.float(), tgt.float())
        else:
            l_jepa = torch.zeros((), device=vid.device)
        return l_pix + self.lam_jepa * l_jepa, l_pix.detach(), l_jepa.detach()

    @torch.no_grad()
    def forecast_pixels(self, vid, mask_row, hp):
        """-> predicted future frames [B, hp, 3, IMG, IMG] in [0,1] (only the future tubelets)."""
        ctx_idx, tgt_idx, n_fut = token_index_sets(mask_row)
        pred = self.predict_repr(vid, ctx_idx, tgt_idx)
        pix = self.readout(pred).float()
        Bn = vid.shape[0]
        c = pix.view(Bn, n_fut, B.GH, B.GH, B.TS, B.PS, B.PS, 3)
        return c.permute(0, 1, 4, 7, 2, 5, 3, 6).reshape(Bn, n_fut * B.TS, 3, B.IMG, B.IMG)


def save_ckpt(wm, path, extra):
    os.makedirs(path, exist_ok=True)
    torch.save({"model": wm.model.state_dict(), "readout": wm.readout.state_dict(),
                "target_encoder": wm.target_encoder.state_dict()}, os.path.join(path, "world_model.pt"))
    json.dump(extra, open(os.path.join(path, "route_j.json"), "w"), indent=1)


def load_ckpt(path, device="cuda"):
    meta = json.load(open(os.path.join(path, "route_j.json")))
    wm = WorldModel(init="random", lam_jepa=meta.get("lam_jepa", 1.0), grad_ckpt=False)
    sd = torch.load(os.path.join(path, "world_model.pt"), map_location="cpu")
    wm.model.load_state_dict(sd["model"]); wm.readout.load_state_dict(sd["readout"])
    wm.target_encoder.load_state_dict(sd["target_encoder"])
    return wm.to(device).eval(), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["vjepa2_wm", "vjepa2_wm_rand", "vjepa2_pixonly"], required=True)
    ap.add_argument("--corpus-dir", default="/nyx-storage1/hanliu/wm4ts/route_b/corpus")
    ap.add_argument("--out", default="/nyx-storage1/hanliu/wm4ts/route_j")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro", type=int, default=8, help="micro-batch; grad accumulation to --batch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--save-every", type=int, default=2500)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--tag", default="")
    ap.add_argument("--grad-ckpt", action="store_true", help="activation checkpointing (same maths, less memory)")
    args = ap.parse_args()

    dev = f"cuda:{args.gpu}"; torch.cuda.set_device(args.gpu)
    init = "random" if args.arm == "vjepa2_wm_rand" else "pretrained"
    lam = 0.0 if args.arm == "vjepa2_pixonly" else 1.0
    wm = WorldModel(init=init, lam_jepa=lam, ema=args.ema, grad_ckpt=args.grad_ckpt).to(dev).train()
    out_dir = os.path.join(args.out, args.arm + args.tag); os.makedirs(out_dir, exist_ok=True)
    info = {**vars(args), "init": init, "lam_jepa": lam,
            "params_M": sum(p.numel() for p in wm.model.parameters()) / 1e6,
            "readout_M": sum(p.numel() for p in wm.readout.parameters()) / 1e6}
    json.dump(info, open(os.path.join(out_dir, "run_config.json"), "w"), indent=1)
    print(f"[{args.arm}] {info}", flush=True)

    params = list(wm.model.parameters()) + list(wm.readout.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    def lr_at(s):
        if s < args.warmup: return args.lr * s / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * p))

    dl = DataLoader(B.Stream(args.corpus_dir, args.batch, seed=args.seed, split="train"), batch_size=None,
                    num_workers=args.workers, prefetch_factor=4, persistent_workers=True)
    vstream = iter(B.Stream(args.corpus_dir, args.batch, seed=777, p_forecast=1.0, split="holdout"))
    val = [next(vstream) for _ in range(args.val_batches)]
    log = open(os.path.join(out_dir, "train_log.jsonl"), "a")
    it = iter(dl); t0 = time.time()
    for step in range(1, args.steps + 1):
        rows, mask, hp, lead = next(it)
        if hp == 0:                                    # tube batches: no future block -> use future=last 2 frames
            hp = 2; lead = 0
            mask = B.forecast_mask(2).unsqueeze(0).expand(rows.shape[0], -1).clone()
        for pg in opt.param_groups: pg["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        tot = lp = lj = 0.0
        n_micro = max(1, rows.shape[0] // args.micro)
        for s in range(0, rows.shape[0], args.micro):
            vid = B.rows_to_video(rows[s:s + args.micro], dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, l_pix, l_jepa = wm.forward_loss(vid, mask[0], hp)
            (loss / n_micro).backward()
            tot += float(loss) / n_micro; lp += float(l_pix) / n_micro; lj += float(l_jepa) / n_micro
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if lam > 0:
            wm.ema_update()
        if step % 50 == 0:
            rec = {"step": step, "loss": tot, "l_pix": lp, "l_jepa": lj, "hp": int(hp), "lead": int(lead),
                   "lr": lr_at(step), "gn": float(gn), "t": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 200 == 0: print(json.dumps(rec), flush=True)
        if step % args.val_every == 0 or step == args.steps:
            wm.eval(); vl = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for vrows, vmask, vhp, vlead in val:
                    for s in range(0, vrows.shape[0], args.micro):
                        _, l_pix, _ = wm.forward_loss(B.rows_to_video(vrows[s:s + args.micro], dev), vmask[0], vhp)
                        vl.append(float(l_pix))
            wm.train()
            rec = {"step": step, "val_pixel_loss": float(np.mean(vl))}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_ckpt(wm, os.path.join(out_dir, f"step_{step}"), {"arm": args.arm, "lam_jepa": lam, "step": step})
            print(f"[ckpt] {out_dir}/step_{step}", flush=True)
    log.close()


if __name__ == "__main__":
    main()
