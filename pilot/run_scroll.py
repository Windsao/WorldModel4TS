"""
Mechanism A: motion-native "scrolling waveform" rendering for VideoMAE.

Hypothesis: the static duplicated-frame grid layout is OOD for a Kinetics prior
built on motion. Here the series scrolls across frames like a camera pan:
frame f shows periods [f*s, f*s + 14) as a 14-column grid (1 period = 1 patch
column), s = SCROLL periods/frame, 16 frames, no duplication. The forecast is
the new content revealed at the right edge of the final frames.

Layout (s=2): total span NP = 15*2+14 = 44 periods; context = NP - HP periods.
Masking: all tokens of the last tubelet whose columns can contain future periods
(cols >= 14 - HP - s covers both frames of the tubelet).
Decode: forecast = right HP columns of the LAST frame (p0=1 slice of tubelet 7).

Compare: scroll_zs / scroll_zero / scroll_ft vs the static-layout lcvmae at the
SAME context budget, plus baselines. Env: wm4ts (transformers<5).
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_pilot as rp
from run_longctx import (IMG, PS, COLS, IMN_MEAN, IMN_STD,
                         get_split_windows, ln_finetune, predict, VisionTSWrap)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NF = 16            # frames, no duplication
SCROLL = 2         # periods advanced per frame
NP = (NF - 1) * SCROLL + COLS          # 44 periods total span


class ScrollVMAE(nn.Module):
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
        self.raw_pixel = not getattr(self.m.config, "norm_pix_loss", True)
        # future must be confined to the last tubelet (frames 14-15), else it
        # would appear unmasked in frame NF-3: hp <= NP - ((NF-3)*SCROLL + COLS)
        assert hp <= NP - ((NF - 3) * SCROLL + COLS)
        self.P, self.hp = P, hp
        self.cp = NP - hp
        gh = IMG // PS
        tt = NF // 2
        # last tubelet: frame 14 covers [28,42), frame 15 covers [30,44).
        # future periods >= cp appear at cols >= cp-30 in frame 15 and >= cp-28
        # in frame 14; mask the UNION (earliest column across both frames),
        # otherwise frame-15 cols [cp-30, cp-28) would leak future content:
        w0 = self.cp - (NF - 1) * SCROLL
        mask3d = torch.zeros(tt, gh, gh, dtype=torch.bool)
        mask3d[-1, :, w0:] = True
        self.register_buffer("bool_masked", mask3d.flatten())
        self.w0 = w0
        self.n_masked = int(mask3d.sum())
        self.register_buffer("imn_mean", IMN_MEAN.view(1, 1, 3, 1, 1))
        self.register_buffer("imn_std", IMN_STD.view(1, 1, 3, 1, 1))

    def render(self, x):
        B = x.shape[0]
        mu = x.mean(1, keepdim=True)
        sd = x.std(1, keepdim=True) + 1e-8
        z = ((x - mu) / (3 * sd)).clamp(-1, 1)
        g = (z + 1) / 2
        grid = torch.full((B, NP, self.P), 0.5)
        grid[:, :self.cp] = g.view(B, self.cp, self.P)
        # sliding windows: frame f = periods [f*s, f*s+14)
        frames = torch.stack([grid[:, f * SCROLL:f * SCROLL + COLS]
                              for f in range(NF)], dim=1)   # [B, NF, 14, P]
        frames = frames.permute(0, 1, 3, 2)                 # [B, NF, P, 14]
        img = F.interpolate(frames.reshape(B * NF, 1, self.P, COLS),
                            size=(IMG, IMG), mode="bilinear", align_corners=False)
        vid = img.view(B, NF, 1, IMG, IMG).repeat(1, 1, 3, 1, 1).to(x.device)
        return (vid - self.imn_mean.to(x.device)) / self.imn_std.to(x.device), mu, sd

    def forward(self, x, zero_pred=False):
        B = x.shape[0]
        gh = IMG // PS
        tt = NF // 2
        vid, mu, sd = self.render(x)
        with torch.no_grad():
            v = vid.view(B, tt, 2, 3, gh, PS, gh, PS)
            tok = v.permute(0, 1, 4, 6, 2, 5, 7, 3).reshape(B, tt, gh, gh,
                                                            2 * PS * PS, 3)
            vis_mask = ~self.bool_masked.view(tt, gh, gh)
            vt = tok[:, vis_mask]
            vh = vis_mask.nonzero()[:, 1]
            mu_h = torch.zeros(B, gh, 3, device=x.device)
            sd_h = torch.zeros(B, gh, 3, device=x.device)
            for h in range(gh):
                sel = vt[:, vh == h]
                mu_h[:, h] = sel.mean(dim=(1, 2))
                sd_h[:, h] = (sel.var(dim=2, unbiased=True) + 1e-6).sqrt().mean(1)
        if zero_pred:
            pred = torch.zeros(B, self.n_masked, 2 * PS * PS * 3, device=x.device)
        else:
            out = self.m(pixel_values=vid,
                         bool_masked_pos=self.bool_masked.unsqueeze(0).expand(B, -1))
            pred = out.logits
        nw = gh - self.w0                                   # masked cols count
        pred = pred.view(B, gh, nw, 2 * PS * PS, 3)
        if self.raw_pixel:
            rec = pred                                      # already pixel space
        else:
            rec = pred * sd_h.view(B, gh, 1, 1, 3) + mu_h.view(B, gh, 1, 1, 3)
        rec = rec.view(B, gh, nw, 2, PS, PS, 3)[:, :, :, 1]  # frame 15 slice
        rec = rec * IMN_STD.to(x.device) + IMN_MEAN.to(x.device)
        rec = rec.mean(-1)                                  # [B, gh, nw, PS, PS]
        # forecast periods = frame-15 cols [14-hp, 14) = last hp of masked cols
        rec = rec[:, :, nw - self.hp:]
        col = rec.permute(0, 2, 1, 3, 4).reshape(B, self.hp, IMG, PS)
        vals = F.adaptive_avg_pool2d(col.mean(-1, keepdim=True), (self.P, 1))
        zhat = 2 * vals.view(B, self.hp * self.P) - 1
        return zhat.clamp(-1, 1) * 3 * sd + mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(rp.DATASETS))
    ap.add_argument("--hp", type=int, default=4)
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--out-dir", default="pilot/results_scroll")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ft-cap", type=int, default=100000)
    ap.add_argument("--methods", default="all")
    args = ap.parse_args()

    data, _ = rp.load_dataset(args.dataset, args.data_dir)
    P = rp.P
    context, horizon = (NP - args.hp) * P, args.hp * P
    Xte, Yte = get_split_windows(data, args.dataset, context, horizon, "test",
                                 args.stride)
    Xtr, Ytr = get_split_windows(data, args.dataset, context, horizon, "train", 1)
    if len(Xtr) > args.ft_cap:
        idx = np.random.default_rng(0).choice(len(Xtr), args.ft_cap, replace=False)
        Xtr, Ytr = Xtr[idx], Ytr[idx]
    print(f"dataset={args.dataset} P={P} context={context} horizon={horizon} "
          f"test={len(Xte)} train={len(Xtr)} scroll={SCROLL}", flush=True)

    cp = NP - args.hp
    METHODS = {
        "naive": lambda: np.repeat(Xte[:, -1:], horizon, axis=1),
        "snaive": lambda: np.tile(Xte[:, -P:], (1, args.hp)),
        "smean": lambda: np.tile(Xte.reshape(-1, cp, P).mean(1), (1, args.hp)),
        "visionts_zs": lambda: predict(VisionTSWrap(P, context, horizon).to(DEVICE),
                                       Xte, batch=128),
        "scroll_zs": lambda: predict(ScrollVMAE(P, args.hp).to(DEVICE), Xte),
        "scroll_zero": lambda: predict(ScrollVMAE(P, args.hp).to(DEVICE), Xte,
                                       zero_pred=True),
        "scroll_ft": lambda: predict(
            ln_finetune(ScrollVMAE(P, args.hp).to(DEVICE), Xtr, Ytr), Xte),
        "scroll_fullft": lambda: predict(
            ln_finetune(ScrollVMAE(P, args.hp).to(DEVICE), Xtr, Ytr,
                        lr=2e-5, regime="full"), Xte),
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
    with open(os.path.join(args.out_dir,
                           f"scroll_{args.dataset}_h{horizon}.json"), "w") as f:
        json.dump({"config": {"dataset": args.dataset, "P": P, "context": context,
                              "horizon": horizon, "scroll": SCROLL,
                              "stride": args.stride, "n_test": len(Xte)},
                   "results": results}, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
