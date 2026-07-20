"""
Field-video design for VideoMAE on time series.

Core idea (differs from all prior arms): put the PREDICTION on the frame axis
and use the WITHIN-frame 2D image for structure the per-frame encoder can read.
Forecast comes from a regression head on pooled spatiotemporal tokens -- NOT
from pixel reconstruction (A8 showed the pixel-decode path is the bottleneck).

Two rendering modes:

  field   : MULTIVARIATE-JOINT. Each frame f = the field at period f, an image
            with rows = variables (ordered by train-split correlation), cols =
            phase within the period, pixel = value. 16 context periods -> 16
            frames; consecutive frames show the field evolving period-to-period
            = genuine motion. One forward predicts ALL channels jointly. This is
            the regime a video model should win (electricity/traffic = fields).

  uni     : channel-independent control (one variable). Each frame = a single
            variable's period rendered as rows=phase (VisionTS-like) tiled to a
            square; frame = period index. Lets us compare joint-field vs
            independent on the same backbone.

No future frames are ever rendered (head predicts), so there is no mask geometry
and no causal-leakage surface. Context-only normalization. Nearest-neighbor
pixel expansion (patch-aligned, no cross-cell bilinear mixing).

Env: wm4ts (transformers<5). VMAE_CKPT selects the backbone checkpoint.
"""

import argparse
import json
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_pilot as rp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG, PS, NF = 224, 16, 16
GH = IMG // PS            # 14
IMN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- data
def load_mv(name, data_dir, max_ch):
    cfg = rp.DATASETS[name]
    rp.configure(cfg["P"])
    path = os.path.join(data_dir, cfg["file"])
    if cfg["kind"] == "ett":
        data = __import__("pandas").read_csv(path).iloc[:, 1:].values.astype(np.float32)
        b_train, b_val, b_test = cfg["borders"]
    else:
        data = np.loadtxt(path, delimiter=",").astype(np.float32)
        T = len(data)
        b_train, b_val, b_test = int(0.7 * T), int(0.8 * T), T
    mean, std = data[:b_train].mean(0), data[:b_train].std(0) + 1e-8
    data = (data - mean) / std
    M = data.shape[1]
    if M > max_ch:
        keep = np.sort(np.random.default_rng(0).choice(M, max_ch, replace=False))
        data = data[:, keep]
    # order variables by train-split correlation (greedy nearest-neighbor chain)
    tr = data[:b_train]
    C = np.corrcoef(tr.T)
    C = np.nan_to_num(C)
    order, used = [0], {0}
    for _ in range(data.shape[1] - 1):
        last = order[-1]
        cand = [(abs(C[last, j]), j) for j in range(data.shape[1]) if j not in used]
        j = max(cand)[1]
        order.append(j); used.add(j)
    data = data[:, order]
    return data, (b_train, b_val, b_test)


def windows(data, borders, split, context, horizon, stride, cap=None):
    b_train, b_val, b_test = borders
    lo, hi = {"train": (context, b_train - horizon),
              "test": (b_val, b_test - horizon)}[split]
    ts = np.arange(lo, hi + 1, stride)
    if cap and len(ts) > cap:
        ts = np.sort(np.random.default_rng(0).choice(ts, cap, replace=False))
    X = np.stack([data[t - context:t] for t in ts])      # [N, context, M]
    Y = np.stack([data[t:t + horizon] for t in ts])      # [N, horizon, M]
    return X.astype(np.float32), Y.astype(np.float32)


# ---------------------------------------------------------------- model
class FieldVMAE(nn.Module):
    def __init__(self, M, P, horizon, mode="field", pretrained=True,
                 backbone="video"):
        super().__init__()
        import transformers
        assert transformers.__version__ < "5"
        self.backbone = backbone
        if backbone == "video":
            from transformers import VideoMAEModel, VideoMAEConfig
            name = os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base")
            self.enc = (VideoMAEModel.from_pretrained(name) if pretrained
                        else VideoMAEModel(VideoMAEConfig.from_pretrained(name)))
        else:  # image control: ViT-MAE encoder applied per frame (same input)
            from transformers import ViTMAEModel, ViTMAEConfig
            name = "facebook/vit-mae-base"
            cfg = ViTMAEConfig.from_pretrained(name)
            cfg.mask_ratio = 0.0
            self.enc = (ViTMAEModel.from_pretrained(name, config=cfg) if pretrained
                        else ViTMAEModel(cfg))
        self.M, self.P, self.horizon, self.mode = M, P, horizon, mode
        d = self.enc.config.hidden_size
        out_dim = (M * horizon) if mode == "field" else horizon
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d),
                                  nn.GELU(), nn.Linear(d, out_dim))
        self.register_buffer("imn_mean", IMN_MEAN)
        self.register_buffer("imn_std", IMN_STD)

    def render(self, x):
        """x [B, NF*P, M] -> vid [B, NF, 3, 224, 224], (mu, sd) per (B,M).

        field: frame f = field at period f; rows=variables, cols=phase.
        uni:   x is [B, NF*P, 1]; frame f = period f, rows=phase tiled.
        """
        B, L, M = x.shape
        mu = x.mean(1, keepdim=True)                       # [B,1,M] context stats
        sd = x.std(1, keepdim=True) + 1e-8
        g = (((x - mu) / (3 * sd)).clamp(-1, 1) + 1) / 2   # [B, L, M] in [0,1]
        g = g.view(B, NF, self.P, M)                       # [B, NF, P, M]
        if self.mode == "field":
            # frame image: rows = M variables, cols = P phase
            fr = g.permute(0, 1, 3, 2)                     # [B, NF, M, P]
            rr, cc = M, self.P
        else:
            # uni: rows = P phase (single variable), tile to square
            fr = g.permute(0, 1, 3, 2)                     # [B, NF, 1, P]
            fr = fr.expand(B, NF, self.P, self.P)          # [B,NF,P,P] broadcast rows
            rr, cc = self.P, self.P
        img = F.interpolate(fr.reshape(B * NF, 1, rr, cc), size=(IMG, IMG),
                            mode="nearest")                # patch-aligned, no mix
        vid = img.view(B, NF, 1, IMG, IMG).expand(B, NF, 3, IMG, IMG)
        vid = (vid - self.imn_mean.unsqueeze(1)) / self.imn_std.unsqueeze(1)
        return vid.contiguous(), mu, sd

    def forward(self, x):
        B = x.shape[0]
        vid, mu, sd = self.render(x)                       # [B, NF, 3, H, W]
        if self.backbone == "video":
            h = self.enc(pixel_values=vid).last_hidden_state.mean(1)   # [B, d]
        else:  # image: encode each frame, mean-pool patch tokens, then over frames
            f = self.enc(pixel_values=vid.reshape(B * NF, 3, IMG, IMG))
            h = f.last_hidden_state.mean(1).view(B, NF, -1).mean(1)    # [B, d]
        out = self.head(h)
        if self.mode == "field":
            z = out.view(B, self.horizon, self.M)
            return z * sd + mu
        else:
            z = out.view(B, self.horizon, 1)
            return z * sd + mu


# ---------------------------------------------------------------- train / eval
def run(model, Xtr, Ytr, Xte, Yte, epochs, lr, batch, mode):
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    n_steps = epochs * ((len(Xtr) + batch - 1) // batch)
    si = 0
    model.train()
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), batch):
            for pg in opt.param_groups:
                pg["lr"] = lr * 0.5 * (1 + math.cos(math.pi * si / n_steps))
            si += 1
            idx = perm[i:i + batch]
            xb = torch.from_numpy(Xtr[idx]).to(DEVICE)
            yb = torch.from_numpy(Ytr[idx]).to(DEVICE)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            tot += loss.item() * len(idx)
            if (i // batch) % 200 == 0:
                print(f"[info] ep{ep} step {i//batch} loss {loss.item():.4f}",
                      flush=True)
        print(f"[info] epoch {ep} train MSE {tot/len(Xtr):.4f}", flush=True)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), batch):
            xb = torch.from_numpy(Xte[i:i + batch]).to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(rp.DATASETS))
    ap.add_argument("--horizon-p", type=int, default=4)
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--out-dir", default="pilot/results_field")
    ap.add_argument("--mode", choices=["field", "uni"], default="field")
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ft-cap", type=int, default=40000)
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--backbone", choices=["video", "image"], default="video")
    args = ap.parse_args()

    data, borders = load_mv(args.dataset, args.data_dir, args.max_ch)
    P = rp.P
    context, horizon = NF * P, args.horizon_p * P
    M = data.shape[1]
    Xtr, Ytr = windows(data, borders, "train", context, horizon, 1, args.ft_cap)
    Xte, Yte = windows(data, borders, "test", context, horizon, args.stride)
    print(f"dataset={args.dataset} mode={args.mode} M={M} P={P} context={context} "
          f"horizon={horizon} train={len(Xtr)} test={len(Xte)}", flush=True)

    results = {}
    # baselines (on same windows)
    def sm(X):  # seasonal mean over context periods
        return np.tile(X.reshape(len(X), NF, P, M).mean(1), (1, args.horizon_p, 1))
    for nm, fn in [("snaive", lambda X: np.tile(X[:, -P:], (1, args.horizon_p, 1))),
                   ("smean", sm)]:
        p = fn(Xte)
        results[nm] = {"MSE": round(float(np.mean((p - Yte) ** 2)), 4),
                       "MAE": round(float(np.mean(np.abs(p - Yte))), 4)}
        print(f"[done] {nm:10s} {results[nm]}", flush=True)

    if args.mode == "uni":
        # flatten to per-channel univariate samples
        Xtr = Xtr.transpose(0, 2, 1).reshape(-1, context, 1)
        Ytr = Ytr.transpose(0, 2, 1).reshape(-1, horizon, 1)
        Xte_u = Xte.transpose(0, 2, 1).reshape(-1, context, 1)
        Yte_u = Yte.transpose(0, 2, 1).reshape(-1, horizon, 1)
        if len(Xtr) > args.ft_cap:
            k = np.random.default_rng(0).choice(len(Xtr), args.ft_cap, False)
            Xtr, Ytr = Xtr[k], Ytr[k]
        model = FieldVMAE(1, P, horizon, "uni", bool(args.pretrained), args.backbone).to(DEVICE)
        pred = run(model, Xtr, Ytr, Xte_u, Yte_u, args.epochs, args.lr,
                   args.batch, "uni")
        mse = float(np.mean((pred - Yte_u) ** 2)); mae = float(np.mean(np.abs(pred - Yte_u)))
    else:
        model = FieldVMAE(M, P, horizon, "field", bool(args.pretrained), args.backbone).to(DEVICE)
        pred = run(model, Xtr, Ytr, Xte, Yte, args.epochs, args.lr,
                   args.batch, "field")
        mse = float(np.mean((pred - Yte) ** 2)); mae = float(np.mean(np.abs(pred - Yte)))
    tag = f"{args.backbone}_{args.mode}" + ("" if args.pretrained else "_rand")
    results[tag] = {"MSE": round(mse, 4), "MAE": round(mae, 4)}
    print(f"[done] {tag:14s} MSE={mse:.4f} MAE={mae:.4f}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir,
                           f"field_{args.dataset}_{args.mode}.json"), "w") as f:
        json.dump({"config": vars(args) | {"M": M, "P": P, "context": context,
                                           "horizon": horizon}, "results": results},
                  f, indent=2)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
