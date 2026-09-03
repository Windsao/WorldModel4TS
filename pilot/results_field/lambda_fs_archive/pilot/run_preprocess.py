"""Frozen-backbone paired forecasting for the input-alignment experiment.

Answers: can a better TS->video renderer make a PRETRAINED VideoMAE beat an identically
configured RANDOM one? Absolute MSE is not the success metric -- the pretrained-minus-
random gap is (PREPROCESSING_EXPERIMENTS.md section 0 and 13).

Kept isolated from pilot/run_field.py, which remains the established entry point; only
its data/window/baseline utilities are reused so the protocol is identical.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_field as RF
from preprocess_renderers import RENDERERS, NORMS, input_stats, contact_sheet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NF = 16
EVAL_BATCH = 64          # memory-safe; 256 OOMed when jobs shared a GPU


class PreprocModel(nn.Module):
    """Frozen VideoMAE + one of three readouts. No raw-series branch by design:
    an NLinear side path would carry the forecast and hide the video contribution."""

    def __init__(self, P, horizon, renderer, readout, pretrained, ckpt, norm, seed=0):
        super().__init__()
        import transformers
        assert transformers.__version__ < "5"
        from transformers import VideoMAEModel, VideoMAEConfig
        cfg = VideoMAEConfig.from_pretrained(ckpt)
        self.enc = VideoMAEModel.from_pretrained(ckpt) if pretrained else VideoMAEModel(cfg)
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.enc.eval()
        d = self.enc.config.hidden_size
        self.P, self.horizon, self.norm = P, horizon, norm
        self.renderer_name, self.readout_name = renderer, readout
        self.render_fn = RENDERERS[renderer]
        self.n_tub = NF // self.enc.config.tubelet_size          # 8
        # seed HERE so the readout is bit-identical for the pretrained/random pair,
        # regardless of how much RNG the encoder construction consumed
        torch.manual_seed(seed)
        if readout == "global":
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                      nn.Linear(d, horizon))
        elif readout == "temporal":
            self.head = nn.Sequential(nn.LayerNorm(self.n_tub * d),
                                      nn.Linear(self.n_tub * d, horizon))
        elif readout == "xattn":
            self.q = nn.Parameter(torch.randn(8, d) * 0.02)
            self.xattn = nn.MultiheadAttention(d, 8, batch_first=True)
            self.xnorm = nn.LayerNorm(d)
            self.head = nn.Linear(8 * d, horizon)
        else:
            raise ValueError(readout)

    def readout_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x):
        vid, mu, sd = self.render_fn(x, self.P, self.norm)
        with torch.no_grad():                      # encoder is frozen: no graph needed
            tok = self.enc(pixel_values=vid).last_hidden_state      # [B, 1568, d]
        B, N, d = tok.shape
        if self.readout_name == "global":
            z = self.head(tok.mean(1))
        elif self.readout_name == "temporal":
            gh = int(round(math.sqrt(N / self.n_tub)))
            z = self.head(tok.view(B, self.n_tub, gh * gh, d).mean(2).reshape(B, -1))
        else:
            q = self.q.unsqueeze(0).expand(B, -1, -1)
            r, _ = self.xattn(q, tok, tok)
            z = self.head(self.xnorm(r).reshape(B, -1))
        return z.unsqueeze(-1) * sd + mu


def train_eval(model, Xtr, Ytr, Xte, Yte, epochs, lr, batch, seed):
    params = model.readout_params()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    n_steps = max(1, epochs * ((len(Xtr) + batch - 1) // batch))
    si = 0
    for ep in range(epochs):
        model.train(); model.enc.eval()
        perm = np.random.default_rng(1000 * seed + ep).permutation(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), batch):
            for pg in opt.param_groups:
                pg["lr"] = lr * 0.5 * (1 + math.cos(math.pi * si / n_steps))
            si += 1
            idx = perm[i:i + batch]
            xb = torch.from_numpy(Xtr[idx]).to(DEVICE)
            yb = torch.from_numpy(Ytr[idx]).to(DEVICE)
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            tot += loss.item() * len(idx)
            if (i // batch) % 100 == 0:
                print(f"[info] ep{ep} step {i//batch} loss {loss.item():.4f}", flush=True)
        print(f"[info] epoch {ep} train MSE {tot/len(Xtr):.4f}", flush=True)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), EVAL_BATCH):
            preds.append(model(torch.from_numpy(Xte[i:i + EVAL_BATCH]).to(DEVICE)).cpu().numpy())
    return np.concatenate(preds)


def git_info():
    try:
        c = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        d = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return c, d
    except Exception:
        return "unknown", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(RF.DATASETS))
    ap.add_argument("--renderer", required=True, choices=list(RENDERERS))
    ap.add_argument("--readout", default="xattn", choices=["global", "temporal", "xattn"])
    ap.add_argument("--norm", default="std_clip_3", choices=list(NORMS))
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--ckpt", default=os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base"))
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--out-dir", default="pilot/results_field/preprocessing/screen")
    ap.add_argument("--horizon-steps", type=int, default=96)
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--ft-cap", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true", help="save a contact sheet and exit")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    t0 = time.time()
    data, borders = RF.load_mv(args.dataset, args.data_dir, args.max_ch)
    P = RF.P
    context = NF * P
    horizon = args.horizon_steps
    M = data.shape[1]
    Xtr, Ytr = RF.windows(data, borders, "train", context, horizon, 1, args.ft_cap)
    Xte, Yte = RF.windows(data, borders, "test", context, horizon, args.stride)
    print(f"dataset={args.dataset} M={M} P={P} ctx={context} h={horizon} "
          f"train={len(Xtr)} test={len(Xte)}", flush=True)

    reps = (horizon + P - 1) // P
    G = context // P
    results = {}
    for nm, fn in [("snaive", lambda X: np.tile(X[:, -P:], (1, reps, 1))[:, :horizon]),
                   ("smean", lambda X: np.tile(X.reshape(len(X), G, P, M).mean(1),
                                               (1, reps, 1))[:, :horizon])]:
        p = fn(Xte)
        results[nm] = {"MSE": round(float(np.mean((p - Yte) ** 2)), 4),
                       "MAE": round(float(np.mean(np.abs(p - Yte))), 4)}
        print(f"[done] {nm:8s} {results[nm]}", flush=True)

    # univariate flatten -- sample indices are cached so the pretrained/random pair
    # sees byte-identical data (section 6 C1, test 7)
    Xtr = Xtr.transpose(0, 2, 1).reshape(-1, context, 1)
    Ytr = Ytr.transpose(0, 2, 1).reshape(-1, horizon, 1)
    Xte = Xte.transpose(0, 2, 1).reshape(-1, context, 1)
    Yte = Yte.transpose(0, 2, 1).reshape(-1, horizon, 1)
    if len(Xtr) > args.ft_cap:
        k = np.random.default_rng(0).choice(len(Xtr), args.ft_cap, False)
        k.sort()
        Xtr, Ytr = Xtr[k], Ytr[k]
        sample_idx_hash = int(k.sum())
    else:
        sample_idx_hash = -1

    model = PreprocModel(P, horizon, args.renderer, args.readout, bool(args.pretrained),
                         args.ckpt, args.norm, seed=args.seed).to(DEVICE)

    if args.preview:
        vid, _, _ = model.render_fn(torch.from_numpy(Xtr[:1]).to(DEVICE), P, args.norm)
        d = os.path.join(os.path.dirname(args.out_dir), "previews")
        pth = contact_sheet(vid[0], os.path.join(d, f"{args.renderer}_{args.dataset}.png"))
        print(json.dumps({"preview": pth, "stats": input_stats(vid)}, indent=2))
        return

    with torch.no_grad():
        vid, _, _ = model.render_fn(torch.from_numpy(Xtr[:min(128, len(Xtr))]).to(DEVICE),
                                    P, args.norm)
        diag = input_stats(vid)
        del vid
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"[info] renderer={args.renderer} readout={args.readout} "
          f"pretrained={args.pretrained} trainable={n_tr}/{n_tot}", flush=True)

    pred = train_eval(model, Xtr, Ytr, Xte, Yte, args.epochs, args.lr, args.batch, args.seed)
    mse = float(np.mean((pred - Yte) ** 2)); mae = float(np.mean(np.abs(pred - Yte)))
    tag = f"{'pt' if args.pretrained else 'rand'}"
    results[f"model_{tag}"] = {"MSE": round(mse, 4), "MAE": round(mae, 4)}
    print(f"[done] model_{tag:5s} MSE={mse:.4f} MAE={mae:.4f}  "
          f"(smean {results['smean']['MSE']:.4f})", flush=True)

    import transformers
    commit, dirty = git_info()
    meta = {"git_commit": commit, "git_dirty": dirty, "dataset": args.dataset,
            "data_dir": os.path.abspath(args.data_dir), "renderer": args.renderer,
            "readout": args.readout, "norm": args.norm, "pretrained": args.pretrained,
            "ckpt": args.ckpt, "transformers": transformers.__version__,
            "torch": torch.__version__, "context": context, "horizon": horizon,
            "M": M, "P": P, "n_train": len(Xtr), "n_test": len(Xte),
            "stride": args.stride, "ft_cap": args.ft_cap, "epochs": args.epochs,
            "batch": args.batch, "lr": args.lr, "seed": args.seed,
            "eval_batch": EVAL_BATCH, "sample_idx_hash": sample_idx_hash,
            "trainable_params": n_tr, "total_params": n_tot,
            "input_diagnostics": diag,
            "wall_clock_s": round(time.time() - t0, 1),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if torch.cuda.is_available() else None}
    os.makedirs(args.out_dir, exist_ok=True)
    fn = (f"{args.dataset}_{args.renderer}_{args.readout}_{args.norm}_"
          f"{tag}_s{args.seed}.json")
    with open(os.path.join(args.out_dir, fn), "w") as f:
        json.dump({"config": meta, "results": results}, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
