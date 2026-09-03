"""Stage E1: label-free patch-statistic lens (spec section 8).

Regime LA. The Stage-C audit located the bottleneck precisely: VideoMAE's normalized-cube
logits carry signal (oracle inversion works) but the per-cube raw mean/std cannot be
recovered causally (3-10x worse than oracle). `norm_pix_loss` discarded exactly those
statistics, so this trains a small lens to put them back.

Label-free by construction: the lens is trained ONLY on pseudo-futures cut from observed
history of the two SELECTION datasets (ETTh2, ETTm2); no forecast target from any held-out
dataset touches training or selection. It predicts each masked cube's raw mean and
log-std from the decoder logits, the cube position, and the nearest visible boundary
statistics -- never from hidden pixels.

Required paired control (section 8 E2): identical lens, identical init, identical
pseudo-windows and steps, on a RANDOM-init frozen backbone. Without it a gain could come
entirely from the lens rather than from video pretraining.
"""

import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "third_party", "VisionTS"))
import run_field as RF
import videomae_patch_utils as VP
import video_visionts_renderers as VR
from run_visionts_reference import HORIZON, build_manifest, gather, git_info

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID, PATCH, NF, TUB = 14, 16, 16, 2
SEL = ("ETTh2", "ETTm2")
DS6 = ("ETTh1", "ETTh2", "ETTm2", "electricity", "traffic", "solar")


class StatLens(nn.Module):
    """Predict raw per-cube (mean, log_std) per channel. Under 500k parameters."""

    def __init__(self, d_logit=1536, hidden=96, n_pos=8 * GRID * GRID, C=3):
        super().__init__()
        self.proj = nn.Linear(d_logit, hidden)                 # 1536*96 = 147,456
        self.pos = nn.Embedding(n_pos, hidden)                 # 1568*96 = 150,528
        self.bnd = nn.Linear(2 * C, hidden)                    # tiny
        self.body = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Linear(hidden, 2 * C)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, logits, pos_idx, bnd_mean, bnd_std):
        h = self.proj(logits) + self.pos(pos_idx) + self.bnd(
            torch.cat([bnd_mean, torch.log(bnd_std.clamp_min(1e-6))], -1))
        h = self.body(h)
        o = self.head(h)
        C = bnd_mean.shape[-1]
        # residual on the causal boundary estimate: zero-init head == plain causal rule
        mean = bnd_mean + o[..., :C]
        log_std = torch.log(bnd_std.clamp_min(1e-6)) + o[..., C:]
        return mean, log_std


def render_batch(xb, P, H, L, cols, blank_future=True):
    ctx = xb[:, :L]
    fut = torch.zeros(xb.shape[0], H, device=xb.device, dtype=xb.dtype) if blank_future else xb[:, L:L + H]
    vi, vf, mu, sd = VR.dense_static(ctx, fut, P, cols)
    return vi, vf, mu, sd


def lens_inputs(cubes_in, bm, logits):
    """Position index and causal boundary statistics for every masked cube."""
    B, N = bm.shape
    cm, cs = VP.causal_cube_stats(cubes_in, bm, rule="nearest_visible_left")
    pos = torch.arange(N, device=bm.device).unsqueeze(0).expand(B, N)[bm].view(B, -1)
    return pos, cm.squeeze(2), cs.squeeze(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default="pilot/results_field/video_visionts")
    ap.add_argument("--ckpt", default=os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base"))
    ap.add_argument("--vts-ckpt-dir", default="./ckpt/")
    ap.add_argument("--mask", default="right_10")
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--train-per-ds", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--random-backbone", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cols = VP.masked_columns(args.mask)

    from transformers import VideoMAEForPreTraining, VideoMAEConfig
    if args.random_backbone:
        torch.manual_seed(args.seed)
        model = VideoMAEForPreTraining(VideoMAEConfig.from_pretrained(args.ckpt))
    else:
        model = VideoMAEForPreTraining.from_pretrained(args.ckpt)
    model = model.to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # lens seeded AFTER backbone construction so both arms get identical initial weights
    torch.manual_seed(args.seed)
    lens = StatLens().to(DEV)
    n_par = sum(p.numel() for p in lens.parameters())
    assert n_par < 500_000, f"lens too large: {n_par}"
    print(f"lens params {n_par} (<500k), backbone={'random' if args.random_backbone else 'pretrained'}", flush=True)

    # ---- label-free training pool: pseudo-futures from TRAIN-split history of ETTh2/ETTm2
    pool = []
    for ds in SEL:
        data, borders = RF.load_mv(ds, args.data_dir, args.max_ch)
        P = RF.P; L = NF * P; H = HORIZON[ds]
        Xh, _ = RF.windows(data, borders, "train", L + H, 1, 32, args.train_per_ds * 3 + 500)
        Xu = Xh.transpose(0, 2, 1).reshape(-1, L + H, 1)[:, :, 0]
        k = np.random.default_rng(args.seed).choice(len(Xu), min(args.train_per_ds, len(Xu)), False)
        pool.append((ds, P, L, H, Xu[np.sort(k)].astype(np.float32)))
    print("train pool:", [(d, len(x)) for d, _, _, _, x in pool], flush=True)

    opt = torch.optim.AdamW(lens.parameters(), lr=args.lr, weight_decay=1e-2)
    rng = np.random.default_rng(args.seed)
    lens.train()
    for step in range(args.steps):
        ds, P, L, H, X = pool[step % len(pool)]
        idx = rng.choice(len(X), args.batch, False)
        xb = torch.from_numpy(X[idx]).to(DEV)
        with torch.no_grad():
            vi, vf, mu, sd = render_batch(xb, P, H, L, cols, blank_future=True)
            bm = VP.make_mask(args.mask, xb.shape[0], DEV)
            logits = model(pixel_values=vi, bool_masked_pos=bm).logits
            cubes_in = VP.patchify(VP.unnormalize(vi))
            cubes_true = VP.patchify(VP.unnormalize(vf))       # HISTORICAL pseudo-future
            tmean = cubes_true.mean(-2)[bm].view(xb.shape[0], -1, 3)
            tstd = cubes_true.var(-2, unbiased=True).sqrt()[bm].view(xb.shape[0], -1, 3) + 1e-6
            pos, bm_, bs_ = lens_inputs(cubes_in, bm, logits)
        pm, pls = lens(logits, pos, bm_, bs_)
        loss = F.mse_loss(pm, tmean) + F.mse_loss(pls, torch.log(tstd))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"[info] step {step} loss {loss.item():.5f}", flush=True)
    lens.eval()

    # ---- evaluate on all six datasets (genuine future, metrics only)
    from visionts import VisionTS
    results = {}
    for ds in DS6:
        data, borders, P, L, H, Xte, Yte, pairs, man = build_manifest(
            ds, args.data_dir, args.max_ch, args.stride, args.n_eval, args.seed)
        Xu, Yu = gather(Xte, Yte, pairs)
        preds = {"lens": [], "causal": []}
        for i in range(0, len(Xu), args.batch * 2):
            xb = torch.from_numpy(Xu[i:i + args.batch * 2]).to(DEV)[:, :, 0]
            B = xb.shape[0]
            with torch.no_grad():
                blank = torch.zeros(B, H, device=DEV)
                vi, _, mu, sd = VR.dense_static(xb[:, :L], blank, P, cols)
                bm = VP.make_mask(args.mask, B, DEV)
                logits = model(pixel_values=vi, bool_masked_pos=bm).logits
                cubes_in = VP.patchify(VP.unnormalize(vi))
                pos, bmn, bsd = lens_inputs(cubes_in, bm, logits)
                pm, pls = lens(logits, pos, bmn, bsd)
                for tag, (m_, s_) in (("lens", (pm.unsqueeze(2), pls.exp().unsqueeze(2))),
                                      ("causal", (bmn.unsqueeze(2), bsd.unsqueeze(2)))):
                    filled = cubes_in.clone()
                    filled[bm] = VP.denormalize_cubes(logits, m_, s_).view(-1, 512, 3)
                    img = VP.unpatchify(filled)[:, NF - 1].mean(1).clamp(0, 1) * 2 - 1
                    vis_w = (GRID - cols) * PATCH
                    preds[tag].append(VR.invert_dense(img[:, :, vis_w:], P, H, mu, sd).cpu().numpy())
        met = {k: round(float(np.mean((np.concatenate(v) - Yu) ** 2)), 6) for k, v in preds.items()}
        vref = json.load(open(os.path.join(args.root, "visionts_reference", f"visionts_{ds}.json")))
        met["visionts"] = vref["metrics"]["visionts_pretrained"]
        met["smean"] = vref["metrics"]["smean"]; met["snaive"] = vref["metrics"]["snaive"]
        results[ds] = met
        print(f"[done] {ds} lens={met['lens']:.4f} causal={met['causal']:.4f} "
              f"visionts={met['visionts']:.4f}", flush=True)

    import transformers
    commit, dirty = git_info()
    out = {"status": "complete", "stage": "E1", "regime": "LA",
           "backbone": "random" if args.random_backbone else "pretrained",
           "mask": args.mask, "renderer": "dense_static", "lens_params": n_par,
           "train_datasets": list(SEL), "train_steps": args.steps, "batch": args.batch,
           "lr": args.lr, "seed": args.seed, "checkpoint": args.ckpt,
           "git_commit": commit, "git_dirty": dirty,
           "torch": torch.__version__, "transformers": transformers.__version__,
           "results": results, "wall_clock_s": round(time.time() - t0, 1),
           "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEV == "cuda" else None}
    d = os.path.join(args.root, "label_free_adaptation"); os.makedirs(d, exist_ok=True)
    tag = "rand" if args.random_backbone else "pt"
    json.dump(out, open(os.path.join(d, f"E1_{tag}_s{args.seed}.json"), "w"), indent=2)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
