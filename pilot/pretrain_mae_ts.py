"""
A8 modality control (PREPROCESSING_AND_KNOWN_ISSUES.md section 6): continued
pretraining of IMAGE MAE (facebook/vit-mae-base, ImageNet) on the same TS data,
objective (raw-pixel reconstruction, forecast-shaped masks) and step budget as
the VideoMAE run — isolating "video vs image pretraining" as the only modality
variable at CPT time.

Rendering: one column-aligned 14-period grid per image (1 period = 1 patch
column, phase-axis-only interpolation), context-only normalization — identical
invariants to the fixed video renderer, causality test applies.

Note: an image sample sees 14-hp context periods vs 104+ for LC-video; this is
inherent to the modality (video's supposed advantage IS the longer view) and is
reported, not hidden.

Run: torchrun --nproc_per_node=2 pilot/pretrain_mae_ts.py --steps 12000 \
       --data real --data-dir ... --out ...
"""

import argparse
import math
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

from pretrain_vmae_ts import (synth_series, to_gray, _col_aligned,
                              load_real_corpus, IMN_MEAN, IMN_STD,
                              IMG, PS, COLS, GH)


def render_image(g, P):
    """g: [14*P] in [0,1] -> [3, 224, 224] (column-aligned)."""
    grid = torch.from_numpy(g).view(COLS, P).permute(1, 0)     # [P, 14]
    img = _col_aligned(grid.unsqueeze(0), P)                   # [1,1,224,224]
    return img[0].expand(3, IMG, IMG).contiguous()


class TSImages(IterableDataset):
    """Yields (pixel_values, noise) batches; forecast mask via noise ordering."""

    def __init__(self, batch, seed, data_mode="synth", data_dir=None,
                 real_frac=0.8):
        self.batch, self.seed = batch, seed
        self.data_mode, self.data_dir, self.real_frac = data_mode, data_dir, real_frac

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0) * 9973)
        corpus = (load_real_corpus(self.data_dir)
                  if self.data_mode == "real" else None)
        while True:
            hp = int(rng.integers(1, 9))
            # noise ranks: masked (right hp columns) get the largest values
            base = torch.rand(GH, GH) * 0.5
            base[:, COLS - hp:] += 1.0
            noise = base.flatten()
            imgs = []
            for _ in range(self.batch):
                if corpus is not None and rng.random() < self.real_frac:
                    arr, P = corpus[rng.integers(len(corpus))]
                    need = COLS * P
                    t = int(rng.integers(need, len(arr)))
                    c = int(rng.integers(arr.shape[1]))
                    y = arr[t - need:t, c].copy()
                else:
                    P = int(rng.integers(16, 169))
                    y = synth_series(rng, COLS, P)
                g = to_gray(y, ctx_len=(COLS - hp) * P)        # context-only
                img = render_image(g, P)
                imgs.append((img - IMN_MEAN.view(3, 1, 1)) / IMN_STD.view(3, 1, 1))
            yield torch.stack(imgs), noise.unsqueeze(0).expand(self.batch, -1), hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", choices=["synth", "real"], default="real")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--real-frac", type=float, default=0.8)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"

    from transformers import ViTMAEForPreTraining
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    model.config.norm_pix_loss = False                     # raw-pixel targets
    model = model.to(dev)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    opt = torch.optim.AdamW(ddp.parameters(), lr=args.lr, weight_decay=0.05)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * s / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * p))

    dl = DataLoader(TSImages(args.batch, seed=99 + rank, data_mode=args.data,
                             data_dir=args.data_dir, real_frac=args.real_frac),
                    batch_size=None, num_workers=6, prefetch_factor=4,
                    persistent_workers=True)
    ddp.train()
    it = iter(dl)
    for step in range(1, args.steps + 1):
        imgs, noise, hp = next(it)
        imgs, noise = imgs.to(dev, non_blocking=True), noise.to(dev)
        model.config.mask_ratio = (GH * hp) / (GH * GH)    # exact masked count
        model.vit.embeddings.config.mask_ratio = model.config.mask_ratio
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = ddp(pixel_values=imgs, noise=noise)
            loss = out.loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        opt.step()
        if rank == 0 and step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} lr {lr_at(step):.2e}",
                  flush=True)
        if rank == 0 and (step % args.save_every == 0 or step == args.steps):
            d = os.path.join(args.out, f"step_{step}")
            model.save_pretrained(d)
            print(f"[ckpt] {d}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
