"""
Fine-tuning experiment: is a video-pretrained model (VideoMAE/Kinetics) a better
foundation for time series forecasting than an image-pretrained model (MAE/ImageNet)
or TS baselines, when fine-tuned per dataset (standard protocol)?

Per dataset, equal-budget fine-tuning (TRAIN_CAP windows x FT_EPOCHS epoch):
  nlinear        : NLinear supervised from scratch (strong linear baseline)
  chronos        : Chronos-bolt-base zero-shot (TS foundation model reference)
  visionts_ln    : VisionTS (ImageNet MAE), LayerNorm-only fine-tuning
  visionts_full  : VisionTS, full fine-tuning
  visionts_rand  : VisionTS random init, full fine-tuning (pretraining ablation)
  videomae_ln    : VideoMAE (Kinetics), LayerNorm-only fine-tuning
  videomae_full  : VideoMAE, full fine-tuning
  videomae_rand  : VideoMAE random init, full fine-tuning (pretraining ablation)

VideoMAE is trained end-to-end: render 12 context periods as frames -> masked
reconstruction of 4 future frames -> differentiable decode to values -> MSE on the
forecast. This removes the zero-shot handicaps (norm_pix_loss level-blindness and
mask-ratio mismatch) because the model learns through the fixed decode transform.

Same data protocol as run_pilot.py (shared module): standardized by train stats,
context 12 periods, horizon 4 periods, test windows at stride = 1 period.
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
TRAIN_CAP = 40000     # max training windows per dataset (seeded subsample; logged)
FT_EPOCHS = 1
LR_LN, LR_FULL, LR_LINEAR = 1e-4, 2e-5, 1e-3
IMG = 224


# ---------------------------------------------------------------- train windows
def get_train_windows(data, name):
    cfg = rp.DATASETS[name]
    if cfg["kind"] == "ett":
        b_train = cfg["borders"][0]
    else:
        b_train = int(0.7 * len(data))
    L, H = rp.CONTEXT, rp.HORIZON
    ts = np.arange(L, b_train - H + 1)
    C = data.shape[1]
    n_total = len(ts) * C
    rng = np.random.default_rng(0)
    if n_total > TRAIN_CAP:
        picks = rng.choice(n_total, TRAIN_CAP, replace=False)
        print(f"[info] train windows capped {TRAIN_CAP}/{n_total} (seed 0)", flush=True)
    else:
        picks = np.arange(n_total)
    X = np.empty((len(picks), L), dtype=np.float32)
    Y = np.empty((len(picks), H), dtype=np.float32)
    for j, p in enumerate(picks):
        t, c = ts[p // C], p % C
        X[j] = data[t - L:t, c]
        Y[j] = data[t:t + H, c]
    return X, Y


# ---------------------------------------------------------------- models
class NLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(rp.CONTEXT, rp.HORIZON)

    def forward(self, x):                      # x [B, L]
        last = x[:, -1:]
        return self.lin(x - last) + last


class VisionTSForecaster(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from visionts import VisionTS
        ckpt_dir = os.environ.get("VISIONTS_CKPT", "./ckpt")
        self.m = VisionTS(arch="mae_base", finetune_type="ln",
                          load_ckpt=pretrained, ckpt_dir=ckpt_dir)
        self.m.update_config(context_len=rp.CONTEXT, pred_len=rp.HORIZON,
                             periodicity=rp.P)

    def forward(self, x):                      # x [B, L]
        y = self.m(x.unsqueeze(-1))
        if isinstance(y, (tuple, list)):
            y = y[0]
        return y.squeeze(-1)


class VideoMAEForecaster(nn.Module):
    def __init__(self, pretrained=True, model_name="MCG-NJU/videomae-base"):
        super().__init__()
        from transformers import VideoMAEForPreTraining, VideoMAEConfig
        if pretrained:
            self.m = VideoMAEForPreTraining.from_pretrained(model_name)
        else:
            cfg = VideoMAEConfig.from_pretrained(model_name)
            self.m = VideoMAEForPreTraining(cfg)
        c = self.m.config
        self.ts, self.ps, self.nf = c.tubelet_size, c.patch_size, c.num_frames
        self.gh, self.tt = IMG // self.ps, self.nf // self.ts
        assert rp.CTX_P + rp.PRED_P == self.nf
        self.vis_t = rp.CTX_P // self.ts
        self.mt = self.tt - self.vis_t
        mask3d = torch.zeros(self.tt, self.gh, self.gh, dtype=torch.bool)
        mask3d[self.vis_t:] = True
        self.register_buffer("bool_masked", mask3d.flatten())
        self.register_buffer("imn_mean", torch.tensor(rp.IMN_MEAN).view(1, 1, 3, 1, 1))
        self.register_buffer("imn_std", torch.tensor(rp.IMN_STD).view(1, 1, 3, 1, 1))

    def forward(self, x):                      # x [B, L] standardized values
        B, P = x.shape[0], rp.P
        ts, ps, nf, gh, tt = self.ts, self.ps, self.nf, self.gh, self.tt
        mu = x.mean(1, keepdim=True)
        sd = x.std(1, keepdim=True) + 1e-8
        z = ((x - mu) / (3 * sd)).clamp(-1, 1)
        g = (z + 1) / 2
        frames = torch.full((B, nf, P), 0.5, device=x.device)
        frames[:, :rp.CTX_P] = g.view(B, rp.CTX_P, P)
        img = F.interpolate(frames.reshape(B * nf, 1, P, 1), size=(IMG, IMG),
                            mode="bilinear", align_corners=False)
        vid = img.view(B, nf, 1, IMG, IMG).repeat(1, 1, 3, 1, 1)
        vid = (vid - self.imn_mean) / self.imn_std

        with torch.no_grad():                  # decode stats from (constant) input
            v = vid.view(B, tt, ts, 3, gh, ps, gh, ps)
            v = v.permute(0, 1, 4, 6, 2, 5, 7, 3).reshape(B, tt, gh, gh, ts * ps * ps, 3)
            vis = v[:, :self.vis_t]
            mu_p = vis.mean(dim=4).mean(dim=1)
            sd_p = (vis.var(dim=4, unbiased=True) + 1e-6).sqrt().mean(dim=1)

        out = self.m(pixel_values=vid,
                     bool_masked_pos=self.bool_masked.unsqueeze(0).expand(B, -1))
        pred = out.logits.view(B, self.mt, gh, gh, ts * ps * ps, 3)
        rec = pred * sd_p.view(B, 1, gh, gh, 1, 3) + mu_p.view(B, 1, gh, gh, 1, 3)
        rec = rec.view(B, self.mt, gh, gh, ts, ps, ps, 3)
        rec = rec.permute(0, 1, 4, 7, 2, 5, 3, 6).reshape(B, self.mt * ts, 3, IMG, IMG)
        rec = rec * self.imn_std.view(1, 1, 3, 1, 1) + self.imn_mean.view(1, 1, 3, 1, 1)
        gray = rec.mean(2)
        vals = F.adaptive_avg_pool2d(gray, (P, 1)).squeeze(-1)
        zhat = 2 * vals.reshape(B, rp.HORIZON) - 1
        return zhat * 3 * sd + mu


# ---------------------------------------------------------------- train / eval
def set_trainable(model, regime):
    n_train = 0
    for n, p in model.named_parameters():
        p.requires_grad = (regime == "full") or ("norm" in n.lower())
        n_train += p.requires_grad * p.numel()
    total = sum(p.numel() for p in model.parameters())
    print(f"[info] trainable params: {n_train/1e6:.2f}M / {total/1e6:.1f}M "
          f"({regime})", flush=True)


def train(model, Xtr, Ytr, lr, epochs=FT_EPOCHS, batch=32):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    n = len(Xtr)
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = torch.from_numpy(Xtr[idx]).float().to(DEVICE)
            yb = torch.from_numpy(Ytr[idx]).float().to(DEVICE)
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        print(f"[info] epoch {ep}: train MSE {tot/n:.4f}", flush=True)


def predict(model, X, batch=128):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).float().to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def run_ft(cls, regime, pretrained, lr, Xtr, Ytr, Xte, batch=32):
    model = cls(pretrained=pretrained).to(DEVICE)
    set_trainable(model, regime)
    train(model, Xtr, Ytr, lr, batch=batch)
    out = predict(model, Xte)
    del model
    torch.cuda.empty_cache()
    return out


def m_chronos(X, batch=256):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base",
                                               device_map=DEVICE,
                                               torch_dtype=torch.float32)
    preds = []
    for i in range(0, len(X), batch):
        ctx = torch.from_numpy(X[i:i + batch]).float()
        _, mean = pipe.predict_quantiles(ctx, prediction_length=rp.HORIZON,
                                         quantile_levels=[0.5])
        preds.append(mean.float().cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(rp.DATASETS))
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--out-dir", default="pilot/results")
    args = ap.parse_args()

    data, test_span = rp.load_dataset(args.dataset, args.data_dir)
    Xte, Yte = rp.get_windows(data, test_span)
    Xtr, Ytr = get_train_windows(data, args.dataset)
    print(f"dataset={args.dataset}  device={DEVICE}  train={len(Xtr)}  "
          f"test={len(Xte)}  P={rp.P}  context={rp.CONTEXT}  horizon={rp.HORIZON}",
          flush=True)

    METHODS = {
        "nlinear": lambda: (lambda m: (set_trainable(m, "full"),
                                       train(m, Xtr, Ytr, LR_LINEAR, epochs=10, batch=256),
                                       predict(m, Xte))[-1])(NLinear().to(DEVICE)),
        "chronos": lambda: m_chronos(Xte),
        "visionts_ln": lambda: run_ft(VisionTSForecaster, "ln", True, LR_LN, Xtr, Ytr, Xte),
        "visionts_full": lambda: run_ft(VisionTSForecaster, "full", True, LR_FULL, Xtr, Ytr, Xte),
        "visionts_rand": lambda: run_ft(VisionTSForecaster, "full", False, LR_FULL, Xtr, Ytr, Xte),
        "videomae_ln": lambda: run_ft(VideoMAEForecaster, "ln", True, LR_LN, Xtr, Ytr, Xte),
        "videomae_full": lambda: run_ft(VideoMAEForecaster, "full", True, LR_FULL, Xtr, Ytr, Xte),
        "videomae_rand": lambda: run_ft(VideoMAEForecaster, "full", False, LR_FULL, Xtr, Ytr, Xte),
        "videomae_l_ln": lambda: run_ft(
            lambda pretrained: VideoMAEForecaster(pretrained, "MCG-NJU/videomae-large"),
            "ln", True, LR_LN, Xtr, Ytr, Xte, batch=16),
        "videomae_l_full": lambda: run_ft(
            lambda pretrained: VideoMAEForecaster(pretrained, "MCG-NJU/videomae-large"),
            "full", True, LR_FULL, Xtr, Ytr, Xte, batch=16),
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
        except Exception:
            import traceback
            print(f"[fail] {name}", flush=True)
            traceback.print_exc()
            results[name] = {"error": True}

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"finetune_{args.dataset}.json")
    with open(out, "w") as f:
        json.dump({"config": {"dataset": args.dataset, "P": rp.P,
                              "context": rp.CONTEXT, "horizon": rp.HORIZON,
                              "train_windows": len(Xtr), "epochs": FT_EPOCHS,
                              "train_cap": TRAIN_CAP},
                   "results": results}, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
