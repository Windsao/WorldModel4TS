"""
Evaluate the A8 control: image-MAE continued-pretrained on TS (pretrain_mae_ts).

Renders each context window as ONE column-aligned 14-period grid image
(context = 14-hp periods -- inherently shorter than the video model's 108;
reported, not hidden). Forecast = raw-pixel reconstruction of the right hp
patch columns, controlled via the ViTMAE noise argument.

  MAE_TS_CKPT=<dir> python pilot/run_maets.py --dataset ETTh1 --hp 4 --stride 4
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F

import run_pilot as rp
from run_longctx import get_split_windows
from pretrain_mae_ts import render_image
from pretrain_vmae_ts import to_gray, IMN_MEAN, IMN_STD, IMG, PS, COLS, GH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MAETS(torch.nn.Module):
    def __init__(self, hp):
        super().__init__()
        from transformers import ViTMAEForPreTraining
        ckpt = os.environ.get("MAE_TS_CKPT", "facebook/vit-mae-base")
        self.m = ViTMAEForPreTraining.from_pretrained(ckpt)
        self.raw_pixel = not getattr(self.m.config, "norm_pix_loss", True)
        self.hp = hp
        self.m.config.mask_ratio = (GH * hp) / (GH * GH)
        self.m.vit.embeddings.config.mask_ratio = self.m.config.mask_ratio
        base = torch.rand(GH, GH) * 0.5
        base[:, COLS - hp:] += 1.0
        self.register_buffer("noise", base.flatten())

    def forward(self, x, zero_pred=False):
        B, P = x.shape[0], rp.P
        cp = COLS - self.hp
        mu = x.mean(1, keepdim=True)
        sd = x.std(1, keepdim=True) + 1e-8
        imgs = []
        for i in range(B):
            y = np.concatenate([x[i].cpu().numpy(),
                                np.zeros(self.hp * P, np.float32)])
            g = to_gray(y, ctx_len=cp * P)
            g[cp * P:] = 0.5
            imgs.append((render_image(g, P) - IMN_MEAN.view(3, 1, 1))
                        / IMN_STD.view(3, 1, 1))
        pix = torch.stack(imgs).to(x.device)
        if zero_pred:
            logits = torch.zeros(B, GH * GH, PS * PS * 3, device=x.device)
        else:
            out = self.m(pixel_values=pix,
                         noise=self.noise.unsqueeze(0).expand(B, -1))
            logits = out.logits                            # [B, 196, 768]
        # decode masked columns (right hp): ViTMAE patchify order is row-major,
        # patch content (PS, PS, 3)
        cols = []
        for w in range(COLS - self.hp, COLS):
            idx = torch.arange(GH, device=x.device) * GH + w
            rec = logits[:, idx].view(B, GH, PS, PS, 3)
            rec = (rec * IMN_STD.reshape(3).to(x.device)
                   + IMN_MEAN.reshape(3).to(x.device))
            rec = rec.mean(-1)                             # [B, GH, PS, PS]
            col = rec.reshape(B, IMG, PS).mean(-1, keepdim=True)  # [B,224,1]
            vals = F.adaptive_avg_pool2d(col.unsqueeze(1), (P, 1))
            cols.append(vals.view(B, P))
        zhat = 2 * torch.cat(cols, dim=1) - 1              # [B, hp*P]
        return zhat.clamp(-1, 1) * 3 * sd + mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(rp.DATASETS))
    ap.add_argument("--hp", type=int, default=4)
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--out-dir", default="pilot/results_maets")
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    data, _ = rp.load_dataset(args.dataset, args.data_dir)
    P = rp.P
    context, horizon = (COLS - args.hp) * P, args.hp * P
    Xte, Yte = get_split_windows(data, args.dataset, context, horizon, "test",
                                 args.stride)
    print(f"dataset={args.dataset} P={P} context={context} horizon={horizon} "
          f"test={len(Xte)}", flush=True)

    model = MAETS(args.hp).to(DEVICE).eval()
    results = {}
    for name, kw in [("maets_zs", {}), ("maets_zero", {"zero_pred": True})]:
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xte), 128):
                xb = torch.from_numpy(Xte[i:i + 128]).float().to(DEVICE)
                preds.append(model(xb, **kw).cpu().numpy())
        p = np.concatenate(preds)
        results[name] = {"MSE": round(float(np.mean((p - Yte) ** 2)), 4),
                         "MAE": round(float(np.mean(np.abs(p - Yte))), 4)}
        print(f"[done] {name:12s} {results[name]}", flush=True)
    for name, fn in [("naive", lambda X: np.repeat(X[:, -1:], horizon, 1)),
                     ("snaive", lambda X: np.tile(X[:, -P:], (1, args.hp)))]:
        p = fn(Xte)
        results[name] = {"MSE": round(float(np.mean((p - Yte) ** 2)), 4),
                         "MAE": round(float(np.mean(np.abs(p - Yte))), 4)}
        print(f"[done] {name:12s} {results[name]}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir,
                           f"maets_{args.dataset}_h{horizon}.json"), "w") as f:
        json.dump({"config": vars(args) | {"P": P, "context": context,
                                           "horizon": horizon},
                   "results": results}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
