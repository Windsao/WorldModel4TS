"""
Phase 3: VisionTS-paper-level protocol, upgraded for the video model.

LC-VMAE layout: each frame is a VisionTS-style 2D grid of 14 periods (1 period =
1 patch column of 16px), 8 content steps x 2 duplicated frames = 16 frames =
112 periods total visual field. Forecast = masked right columns of the LAST
tubelet only, so the horizon is the standard 96 steps (hourly) while the context
is (112 - HP) periods (~2592 steps for hourly -- VisionTS paper scale).

Arms per dataset (equal context budget everywhere):
  naive / snaive / smean         : baselines at matched long context
  visionts_zs / visionts_ft      : official VisionTS, context=(112-HP)*P, LN-FT 1 epoch
  lcvmae_zs / lcvmae_zero        : long-context VideoMAE zero-shot + zero-pred control
  lcvmae_ft                      : LN-FT 1 epoch (end-to-end, full train set w/ cap)

Standard eval: horizon = HP*P steps, test stride configurable (1 for ETT).
Env: wm4ts (transformers<5).
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_pilot as rp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG, PS, COLS, STEPS, DUP = 224, 16, 14, 8, 2
NP = COLS * STEPS                      # 112 periods total visual field
IMN_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMN_STD = torch.tensor([0.229, 0.224, 0.225])


# ---------------------------------------------------------------- data
def get_split_windows(data, name, context, horizon, split, stride, cap=None):
    cfg = rp.DATASETS[name]
    if cfg["kind"] == "ett":
        b_train, b_val, b_test = cfg["borders"]
    else:
        T = len(data)
        b_train, b_val, b_test = int(0.7 * T), int(0.8 * T), T
    lo, hi = {"train": (context, b_train - horizon),
              "test": (b_val, b_test - horizon)}[split]
    ts = np.arange(lo, hi + 1, stride)
    C = data.shape[1]
    n_total = len(ts) * C
    # sample BEFORE materializing — long contexts otherwise allocate 10s of GB
    if cap and n_total > cap:
        picks = np.sort(np.random.default_rng(0).choice(n_total, cap,
                                                        replace=False))
        print(f"[info] {split} windows capped {cap}/{n_total}", flush=True)
    else:
        picks = np.arange(n_total)
    X = np.empty((len(picks), context), dtype=np.float32)
    Y = np.empty((len(picks), horizon), dtype=np.float32)
    for k, p in enumerate(picks):
        t, c = ts[p // C], p % C
        X[k] = data[t - context:t, c]
        Y[k] = data[t:t + horizon, c]
    return X, Y


# ---------------------------------------------------------------- LC-VMAE
class LCVMAE(nn.Module):
    def __init__(self, P, hp, pretrained=True):
        super().__init__()
        import transformers
        assert transformers.__version__ < "5"
        from transformers import VideoMAEForPreTraining, VideoMAEConfig
        name = os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base")
        if pretrained:
            self.m = VideoMAEForPreTraining.from_pretrained(name)
        else:
            self.m = VideoMAEForPreTraining(VideoMAEConfig.from_pretrained(name))
        # continued-pretrained ckpts predict raw (ImageNet-normalized) pixels
        self.raw_pixel = not getattr(self.m.config, "norm_pix_loss", True)
        self.P, self.hp = P, hp
        self.cp = NP - hp                                  # context periods
        gh = IMG // PS                                     # 14
        assert gh == COLS
        mask3d = torch.zeros(STEPS, gh, gh, dtype=torch.bool)
        mask3d[-1, :, COLS - hp:] = True                   # right cols, last tubelet
        self.register_buffer("bool_masked", mask3d.flatten())
        self.n_masked = int(mask3d.sum())                  # 14*hp
        self.register_buffer("imn_mean", IMN_MEAN.view(1, 1, 3, 1, 1))
        self.register_buffer("imn_std", IMN_STD.view(1, 1, 3, 1, 1))

    def render(self, x):
        """x [B, cp*P] -> vid [B, 16, 3, 224, 224], (mu, sd).

        RENDER_AUG=1: channels carry distinct views instead of replicated gray —
        R = raw values, G = first difference (local dynamics),
        B = expanding per-phase mean (level anchor). Forecast is decoded from
        channel R only.
        """
        B = x.shape[0]
        mu = x.mean(1, keepdim=True)
        sd = x.std(1, keepdim=True) + 1e-8
        z = ((x - mu) / (3 * sd)).clamp(-1, 1)
        g = (z + 1) / 2
        chans = [g]
        if os.environ.get("RENDER_AUG") == "1":
            d = torch.diff(x, dim=1, prepend=x[:, :1])
            sdd = d.std(1, keepdim=True) + 1e-8
            chans.append(((d / (3 * sdd)).clamp(-1, 1) + 1) / 2)
            per = g.view(B, self.cp, self.P)
            csum = per.cumsum(1) / torch.arange(1, self.cp + 1,
                                                device=x.device).view(1, -1, 1)
            chans.append(csum.reshape(B, -1))
        else:
            chans += [g, g]
        while len(chans) < 3:
            chans.append(chans[0])
        vids = []
        for c in chans:
            grid = torch.full((B, NP, self.P), 0.5)
            grid[:, :self.cp] = c.view(B, self.cp, self.P)
            grid = grid.view(B, STEPS, COLS, self.P).permute(0, 1, 3, 2)
            # column-aligned: phase-axis interpolation only, 1 period = 1 patch
            # column, no cross-column mixing (P0-1 in the known-issues doc)
            img = F.interpolate(grid.reshape(B * STEPS, 1, self.P, COLS),
                                size=(IMG, COLS), mode="bilinear",
                                align_corners=False).repeat_interleave(PS, dim=-1)
            vids.append(img.view(B, STEPS, IMG, IMG))
        vid = torch.stack(vids, dim=2)                     # [B, STEPS, 3, H, W]
        vid = vid.unsqueeze(2).expand(B, STEPS, DUP, 3, IMG, IMG)
        vid = vid.reshape(B, STEPS * DUP, 3, IMG, IMG).to(x.device)
        return (vid - self.imn_mean.to(x.device)) / self.imn_std.to(x.device), mu, sd

    def patchify(self, vid):
        """[B, 16, 3, H, W] -> [B, STEPS, gh, gh, DUP*PS*PS, 3] (HF token order)."""
        B = vid.shape[0]
        gh = IMG // PS
        v = vid.view(B, STEPS, DUP, 3, gh, PS, gh, PS)
        return v.permute(0, 1, 4, 6, 2, 5, 7, 3).reshape(B, STEPS, gh, gh,
                                                         DUP * PS * PS, 3)

    def forward(self, x, zero_pred=False):
        B = x.shape[0]
        gh = IMG // PS
        vid, mu, sd = self.render(x)
        with torch.no_grad():
            tok = self.patchify(vid)                       # [B,8,14,14,512,3]
            vis_mask = ~self.bool_masked.view(STEPS, gh, gh)
            vt = tok[:, vis_mask]                          # [B, n_vis, 512, 3]
            vh = vis_mask.nonzero()[:, 1]                  # h index of visible tokens
            mu_h = torch.zeros(B, gh, 1, 3, device=x.device)
            sd_h = torch.zeros(B, gh, 1, 3, device=x.device)
            for h in range(gh):                            # per-phase-band stats
                sel = vt[:, vh == h]
                mu_h[:, h, 0] = sel.mean(dim=(1, 2))
                sd_h[:, h, 0] = (sel.var(dim=2, unbiased=True) + 1e-6).sqrt().mean(1)
        if zero_pred:
            pred = torch.zeros(B, self.n_masked, DUP * PS * PS * 3, device=x.device)
        else:
            out = self.m(pixel_values=vid,
                         bool_masked_pos=self.bool_masked.unsqueeze(0).expand(B, -1))
            pred = out.logits                              # [B, 14*hp, 1536]
        # masked tokens ordered h-major then w (t fixed = last)
        pred = pred.view(B, gh, self.hp, DUP * PS * PS, 3)
        if self.raw_pixel:
            rec = pred                                     # already pixel space
        else:
            rec = pred * sd_h.unsqueeze(2) + mu_h.unsqueeze(2)  # phase prior
        # -> pixel columns: [B, gh(h), hp(w), DUP, PS, PS, 3]
        rec = rec.view(B, gh, self.hp, DUP, PS, PS, 3).mean(3)  # avg dup frames
        rec = rec * IMN_STD.to(x.device) + IMN_MEAN.to(x.device)  # per-channel denorm
        if os.environ.get("RENDER_AUG") == "1":
            rec = rec[..., 0]                              # R channel = raw values
        else:
            rec = rec.mean(-1)                             # grayscale [B,gh,hp,PS,PS]
        # assemble each masked period column: stack h -> [B, hp, gh*PS(=224), PS]
        col = rec.permute(0, 2, 1, 3, 4).reshape(B, self.hp, IMG, PS)
        vals = F.adaptive_avg_pool2d(col.mean(-1, keepdim=True), (self.P, 1))
        zhat = 2 * vals.view(B, self.hp * self.P) - 1
        return zhat.clamp(-1, 1) * 3 * sd + mu


# ---------------------------------------------------------------- helpers
def predict(model, X, batch=48, **kw):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).float().to(DEVICE)
            out.append(model(xb, **kw).float().cpu().numpy())
    return np.concatenate(out)


def ln_finetune(model, Xtr, Ytr, lr=1e-4, batch=32, regime="ln", epochs=1):
    import math
    n_tr = 0
    for n, p in model.named_parameters():
        p.requires_grad = (regime == "full") or "norm" in n.lower() \
            or "head" in n.lower()
        n_tr += p.requires_grad * p.numel()
    print(f"[info] {regime}-FT trainable {n_tr/1e6:.2f}M epochs={epochs}",
          flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    n_steps = epochs * ((len(Xtr) + batch - 1) // batch)
    si = 0
    model.train()
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), batch):
            for pg in opt.param_groups:                    # cosine schedule
                pg["lr"] = lr * 0.5 * (1 + math.cos(math.pi * si / n_steps))
            si += 1
            idx = perm[i:i + batch]
            xb = torch.from_numpy(Xtr[idx]).float().to(DEVICE)
            yb = torch.from_numpy(Ytr[idx]).float().to(DEVICE)
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item() * len(idx)
            if (i // batch) % 300 == 0:
                print(f"[info] ep{ep} step {i//batch} loss {loss.item():.4f}",
                      flush=True)
        print(f"[info] epoch {ep} train MSE {tot/len(Xtr):.4f}", flush=True)
    return model


class HankelHead(LCVMAE):
    """Delay-embedding video + regression head (doc section 4.2/4.5).

    Frame f = Hankel matrix of a sliding context window: H[i, j] = w[j + i*d]
    (rows = lags, columns = time), so temporal evolution flows diagonally and
    recurring motifs become textures — the closest TS analog of natural video
    motion. Context-only rendering (no future pixels exist), causal by
    construction; forecast comes from the numeric head, no decode needed.
    """

    W, D, NLAG = 512, 4, 64

    def __init__(self, P, hp, pretrained=True):
        super().__init__(P, hp, pretrained)
        d = self.m.config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hp * P))
        nj = self.W - (self.NLAG - 1) * self.D
        idx = (torch.arange(self.NLAG)[:, None] * self.D +
               torch.arange(nj)[None, :])
        self.register_buffer("hidx", idx)                  # [64, nj]

    def forward(self, x, zero_pred=False):
        B, L = x.shape
        mu = x.mean(1, keepdim=True)
        sd = x.std(1, keepdim=True) + 1e-8
        g = (((x - mu) / (3 * sd)).clamp(-1, 1) + 1) / 2
        s = max(1, (L - self.W) // 15)
        frames = []
        for f in range(16):
            st = min(f * s, L - self.W)
            w = g[:, st:st + self.W]
            H = w[:, self.hidx]                            # [B, 64, nj]
            frames.append(H)
        fr = torch.stack(frames, 1)                        # [B, 16, 64, nj]
        img = F.interpolate(fr.reshape(B * 16, 1, *fr.shape[2:]),
                            size=(IMG, IMG), mode="bilinear",
                            align_corners=False)
        vid = img.view(B, 16, 1, IMG, IMG).repeat(1, 1, 3, 1, 1)
        vid = (vid - self.imn_mean.to(x.device)) / self.imn_std.to(x.device)
        out = self.m.videomae(pixel_values=vid)            # no mask: full video
        z = self.head(out.last_hidden_state.mean(1))
        return z * sd + mu


class NLinearL(nn.Module):
    def __init__(self, context, horizon):
        super().__init__()
        self.lin = nn.Linear(context, horizon)

    def forward(self, x):
        last = x[:, -1:]
        return self.lin(x - last) + last


class LCVMAEHead(LCVMAE):
    """Encoder + direct regression head — bypasses the pixel-reconstruction
    decode path entirely (the suspected zero-shot bottleneck)."""

    def __init__(self, P, hp, pretrained=True):
        super().__init__(P, hp, pretrained)
        d = self.m.config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hp * P))

    def forward(self, x, zero_pred=False):
        B = x.shape[0]
        vid, mu, sd = self.render(x)
        out = self.m.videomae(pixel_values=vid,
                              bool_masked_pos=self.bool_masked.unsqueeze(0)
                              .expand(B, -1))
        z = self.head(out.last_hidden_state.mean(1))       # [B, horizon]
        return z * sd + mu


class VisionTSWrap(nn.Module):
    def __init__(self, P, context, horizon):
        super().__init__()
        from visionts import VisionTS
        self.m = VisionTS(arch="mae_base", finetune_type="ln", load_ckpt=True,
                          ckpt_dir=os.environ.get("VISIONTS_CKPT", "./ckpt"))
        self.m.update_config(context_len=context, pred_len=horizon, periodicity=P)

    def forward(self, x):
        y = self.m(x.unsqueeze(-1))
        if isinstance(y, (tuple, list)):
            y = y[0]
        return y.squeeze(-1)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(rp.DATASETS))
    ap.add_argument("--hp", type=int, default=4, help="horizon in periods (<=14)")
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--out-dir", default="pilot/results_longctx")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ft-cap", type=int, default=100000)
    ap.add_argument("--methods", default="all")
    args = ap.parse_args()
    assert 1 <= args.hp <= COLS

    data, _ = rp.load_dataset(args.dataset, args.data_dir)
    P = rp.P
    context, horizon = (NP - args.hp) * P, args.hp * P
    Xte, Yte = get_split_windows(data, args.dataset, context, horizon, "test",
                                 args.stride)
    Xtr, Ytr = get_split_windows(data, args.dataset, context, horizon, "train",
                                 1, cap=args.ft_cap)
    print(f"dataset={args.dataset} P={P} context={context} horizon={horizon} "
          f"test={len(Xte)} train={len(Xtr)} stride={args.stride}", flush=True)

    cp = NP - args.hp

    def m_smean(X):
        return np.tile(X.reshape(-1, cp, P).mean(1), (1, args.hp))

    METHODS = {
        "naive": lambda: np.repeat(Xte[:, -1:], horizon, axis=1),
        "snaive": lambda: np.tile(Xte[:, -P:], (1, args.hp)),
        "smean": lambda: m_smean(Xte),
        "visionts_zs": lambda: predict(VisionTSWrap(P, context, horizon).to(DEVICE),
                                       Xte, batch=128),
        "visionts_ft": lambda: predict(
            ln_finetune(VisionTSWrap(P, context, horizon).to(DEVICE), Xtr, Ytr),
            Xte, batch=128),
        "lcvmae_zs": lambda: predict(LCVMAE(P, args.hp).to(DEVICE), Xte),
        "lcvmae_zero": lambda: predict(LCVMAE(P, args.hp).to(DEVICE), Xte,
                                       zero_pred=True),
        "lcvmae_ft": lambda: predict(
            ln_finetune(LCVMAE(P, args.hp).to(DEVICE), Xtr, Ytr), Xte),
        "lcvmae_fullft": lambda: predict(
            ln_finetune(LCVMAE(P, args.hp).to(DEVICE), Xtr, Ytr,
                        lr=2e-5, regime="full"), Xte),
        "visionts_fullft": lambda: predict(
            ln_finetune(VisionTSWrap(P, context, horizon).to(DEVICE), Xtr, Ytr,
                        lr=2e-5, regime="full"), Xte, batch=128),
        # FT-focused arms: 3-epoch cosine recipe (use VMAE_CKPT=v2 for CPT init)
        "lcvmae_ft3": lambda: predict(
            ln_finetune(LCVMAE(P, args.hp).to(DEVICE), Xtr, Ytr,
                        lr=5e-5, regime="full", epochs=3), Xte),
        "lcvmae_head3": lambda: predict(
            ln_finetune(LCVMAEHead(P, args.hp).to(DEVICE), Xtr, Ytr,
                        lr=5e-5, regime="full", epochs=3), Xte),
        "nlinear": lambda: predict(
            ln_finetune(NLinearL(context, horizon).to(DEVICE), Xtr, Ytr,
                        lr=1e-3, regime="full", epochs=10, batch=256),
            Xte, batch=512),
        "hankel_head3": lambda: predict(
            ln_finetune(HankelHead(P, args.hp).to(DEVICE), Xtr, Ytr,
                        lr=5e-5, regime="full", epochs=3), Xte),
    }
    names = list(METHODS) if args.methods == "all" else args.methods.split(",")

    results = {}
    for name in names:
        try:
            pred = METHODS[name]()
            mse = float(np.mean((pred - Yte) ** 2))
            mae = float(np.mean(np.abs(pred - Yte)))
            results[name] = {"MSE": round(mse, 4), "MAE": round(mae, 4)}
            print(f"[done] {name:14s} MSE={mse:.4f}  MAE={mae:.4f}", flush=True)
            torch.cuda.empty_cache()
        except Exception:
            import traceback
            print(f"[fail] {name}", flush=True)
            traceback.print_exc()
            results[name] = {"error": True}

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.dataset}_h{horizon}"
    with open(os.path.join(args.out_dir, f"longctx_{tag}.json"), "w") as f:
        json.dump({"config": {"dataset": args.dataset, "P": P, "context": context,
                              "horizon": horizon, "stride": args.stride,
                              "n_test": len(Xte), "n_train": len(Xtr)},
                   "results": results}, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
