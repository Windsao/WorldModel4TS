"""
Mechanism B: continued pretraining of VideoMAE on synthetic TS-rendered videos
(the VisionTS++ move, one modality up — "VideoMAE-TS").

Fixes the three diagnosed failure modes:
  1. level pathway  : norm_pix_loss=False -> model learns to predict RAW pixels
  2. content OOD    : training distribution = TS renderings, not natural video
  3. layout coverage: trains on BOTH layouts (static LC grid + scrolling), with
                      50% forecast-shaped masks / 50% native random tube masks

Data: infinite RealTS-style synthetic series (harmonic seasonality + slow
components + trends + AR noise + level shifts + spikes). Purely synthetic ->
zero benchmark leakage; zero-shot evaluation on real datasets stays clean.

Run (2 GPUs):
  torchrun --nproc_per_node=2 pilot/pretrain_vmae_ts.py --steps 30000 \
      --out /nyx-storage1/hanliu/wm4ts/ckpt_vmae_ts
"""

import argparse
import math
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

IMG, PS, COLS = 224, 16, 14
LC_STEPS, LC_DUP = 8, 2            # static grid: 112 periods
NF, SCROLL = 16, 2                 # scroll: 44 periods
GH = IMG // PS
TT = 8                             # tubelets


# ---------------------------------------------------------------- synthetic TS
def synth_series(rng, n_periods, P):
    n = n_periods * P
    t = np.arange(n, dtype=np.float64)
    y = np.zeros(n)
    for _ in range(rng.integers(1, 4)):                    # seasonal harmonics
        f = P / rng.choice([1, 2, 3, 4])
        y += rng.uniform(0.2, 1.5) * np.sin(2 * np.pi * t / f +
                                            rng.uniform(0, 2 * np.pi))
    if rng.random() < 0.5:                                 # slow component
        f = P * rng.uniform(3, max(4, n_periods / 2))
        y += rng.uniform(0.3, 1.2) * np.sin(2 * np.pi * t / f +
                                            rng.uniform(0, 2 * np.pi))
    if rng.random() < 0.7:                                 # trend
        y += rng.uniform(-2, 2) * t / n
    e = rng.normal(0, rng.uniform(0.05, 0.4), n)           # AR(1) noise
    a = rng.uniform(0, 0.95)
    for i in range(1, n):
        e[i] += a * e[i - 1]
    y += e
    if rng.random() < 0.3:                                 # level shift
        y[rng.integers(n):] += rng.uniform(-1.5, 1.5)
    if rng.random() < 0.3:                                 # spikes
        for _ in range(rng.integers(1, 5)):
            y[rng.integers(n)] += rng.normal(0, 2)
    return y.astype(np.float32)


# ---------------------------------------------------------------- rendering
def to_gray(y, ctx_len=None):
    """P0-2 fix: normalization stats from CONTEXT ONLY (never from the target),
    so visible pixel intensities cannot carry future information."""
    ref = y[:ctx_len] if ctx_len else y
    mu, sd = ref.mean(), ref.std() + 1e-8
    return np.clip((y - mu) / (3 * sd), -1, 1) * 0.5 + 0.5


def _col_aligned(grid_tp, P):
    """P0-1 fix: interpolate ONLY the phase axis; each logical period column
    maps to exactly one 16px patch column with zero cross-column mixing."""
    img = F.interpolate(grid_tp.unsqueeze(1), size=(IMG, COLS), mode="bilinear",
                        align_corners=False)               # [T,1,224,14]
    return img.repeat_interleave(PS, dim=-1)               # [T,1,224,224]


def render_lc(g, P):                                       # g: [112*P] in [0,1]
    grid = torch.from_numpy(g).view(LC_STEPS, COLS, P).permute(0, 2, 1)
    img = _col_aligned(grid, P)                            # [8,1,224,224]
    vid = img.unsqueeze(1).expand(LC_STEPS, LC_DUP, 1, IMG, IMG)
    return vid.reshape(NF, 1, IMG, IMG).expand(NF, 3, IMG, IMG).contiguous()


def render_scroll(g, P):                                   # g: [44*P]
    grid = torch.from_numpy(g).view((NF - 1) * SCROLL + COLS, P)
    frames = torch.stack([grid[f * SCROLL:f * SCROLL + COLS] for f in range(NF)])
    img = _col_aligned(frames.permute(0, 2, 1), P)
    return img.expand(NF, 3, IMG, IMG).contiguous()


def forecast_mask_lc(hp):
    m = torch.zeros(TT, GH, GH, dtype=torch.bool)
    m[-1, :, COLS - hp:] = True
    return m.flatten()


def forecast_mask_scroll(hp):
    w0 = ((NF - 1) * SCROLL + COLS - hp) - (NF - 1) * SCROLL
    m = torch.zeros(TT, GH, GH, dtype=torch.bool)
    m[-1, :, w0:] = True
    return m.flatten()


def tube_mask(rng, ratio=0.75):
    n = GH * GH
    k = int(n * ratio)
    cols = torch.from_numpy(rng.choice(n, k, replace=False))
    m = torch.zeros(TT, GH * GH, dtype=torch.bool)
    m[:, cols] = True
    return m.flatten()


IMN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMN_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


REAL_DATASETS = {  # file, period, train fraction/border (train split ONLY)
    "ETTh1": ("ETTh1.csv", 24, 8640), "ETTh2": ("ETTh2.csv", 24, 8640),
    "ETTm1": ("ETTm1.csv", 96, 34560), "ETTm2": ("ETTm2.csv", 96, 34560),
    "electricity": ("electricity.txt", 24, None),
    "traffic": ("traffic.txt", 24, None), "solar": ("solar_AL.txt", 144, None),
}


def load_real_corpus(data_dir):
    import pandas as pd
    corpus = []
    for name, (fn, P, b) in REAL_DATASETS.items():
        path = os.path.join(data_dir, fn)
        if fn.endswith(".csv"):
            arr = pd.read_csv(path).iloc[:, 1:].values.astype(np.float32)
        else:
            arr = np.loadtxt(path, delimiter=",").astype(np.float32)
        b = b or int(0.7 * len(arr))
        tr = arr[:b]
        tr = (tr - tr.mean(0)) / (tr.std(0) + 1e-8)        # train-only stats
        corpus.append((tr, P))                             # train region only
    return corpus


class TSVideos(IterableDataset):
    """Yields whole batches (same mask count within a batch)."""

    def __init__(self, batch, seed, data_mode="synth", data_dir=None,
                 real_frac=0.8, forecast_only=False):
        self.batch, self.seed = batch, seed
        self.data_mode, self.data_dir = data_mode, data_dir
        self.real_frac, self.forecast_only = real_frac, forecast_only

    def sample_series(self, rng, n_p):
        if self.corpus and rng.random() < self.real_frac:
            arr, P = self.corpus[rng.integers(len(self.corpus))]
            need = n_p * P
            if len(arr) > need + 1:
                t = int(rng.integers(need, len(arr)))
                c = int(rng.integers(arr.shape[1]))
                return arr[t - need:t, c].copy(), P
        P = int(rng.integers(16, 169))
        return synth_series(rng, n_p, P), P

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0) * 9973)
        self.corpus = (load_real_corpus(self.data_dir)
                       if self.data_mode == "real" else None)
        while True:
            scroll = rng.random() < 0.5
            n_p = (NF - 1) * SCROLL + COLS if scroll else LC_STEPS * COLS
            hp = 0
            if self.forecast_only or rng.random() < 0.5:   # forecast mask
                # scroll: future must stay inside the last tubelet -> hp <= 4
                hp = int(rng.integers(1, 5 if scroll else 9))
                mask = (forecast_mask_scroll(hp) if scroll
                        else forecast_mask_lc(hp))
                # NOTE: do NOT blank the future region. Masked tokens are never
                # fed to the encoder; their pixels are only the TARGET.
            else:                                          # native tube mask
                mask = tube_mask(rng)
            vids = []
            for _ in range(self.batch):
                y, P = self.sample_series(rng, n_p)
                # P0-2: context-only stats for forecast masks; temporal-prefix
                # stats for tube masks (no whole-sample statistic).
                ctx_len = (n_p - hp) * P if hp else (n_p // 2) * P
                g = to_gray(y, ctx_len=ctx_len)
                vid = (render_scroll(g, P) if scroll
                       else render_lc(g, P))
                vids.append((vid - IMN_MEAN) / IMN_STD)
            yield torch.stack(vids), mask.unsqueeze(0).expand(self.batch, -1)


# ---------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", choices=["synth", "real"], default="synth")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--real-frac", type=float, default=0.8)
    ap.add_argument("--forecast-only", action="store_true")
    ap.add_argument("--init", default="MCG-NJU/videomae-base")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"

    from transformers import VideoMAEForPreTraining
    model = VideoMAEForPreTraining.from_pretrained(args.init)
    model.config.norm_pix_loss = False                     # raw-pixel targets
    model = model.to(dev)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * s / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * p))

    dl = DataLoader(TSVideos(args.batch, seed=1234 + rank, data_mode=args.data,
                             data_dir=args.data_dir, real_frac=args.real_frac,
                             forecast_only=args.forecast_only),
                    batch_size=None, num_workers=6, prefetch_factor=4,
                    persistent_workers=True)
    model.train()
    it = iter(dl)
    for step in range(1, args.steps + 1):
        vids, mask = next(it)
        vids, mask = vids.to(dev, non_blocking=True), mask.to(dev)
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(pixel_values=vids, bool_masked_pos=mask)
            loss = out.loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if rank == 0 and step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} lr {lr_at(step):.2e}",
                  flush=True)
        if rank == 0 and (step % args.save_every == 0 or step == args.steps):
            d = os.path.join(args.out, f"step_{step}")
            model.module.save_pretrained(d)
            print(f"[ckpt] {d}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
