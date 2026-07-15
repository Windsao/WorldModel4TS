"""
Pilot experiment: does a video-pretrained model (VideoMAE, Kinetics-400) transfer to
numeric time series forecasting, compared to an image-pretrained model (VisionTS / MAE)?

Protocol
--------
Channel-independent, values standardized by train-split mean/std (metrics in that
space). Context = 12 periods, horizon = 4 periods for every dataset (fixed period
budget so the video mapping is uniform: 12 visible frames + 4 masked frames).
Test windows at stride = one period. NOTE: horizons are therefore dataset-dependent
(96 steps for hourly, 384 for 15-min, 576 for 10-min) and NOT directly comparable
to standard literature numbers -- this is a controlled internal comparison.

Requires transformers<5 (v5 renames VideoMAE q_bias/v_bias keys and silently
re-initializes all attention biases).

Methods
-------
naive          : repeat last value
snaive         : repeat last period
smean          : per-phase mean over context periods (seasonal average)
visionts       : official VisionTS package, ImageNet MAE, zero-shot
videomae       : VideoMAE zero-shot. Render each period as one frame
                 (phase -> image rows, grayscale intensity = value), 12 context
                 frames + 4 masked future frames, forecast = reconstruction of
                 masked tubelets. Since the checkpoint was trained with
                 norm_pix_loss, masked-patch predictions are de-normalized with
                 per-(h,w) statistics estimated from the visible tubelets
                 (a seasonal prior over levels; the model contributes the shape).
videomae_zero  : same decode pipeline with model predictions zeroed out --
                 isolates the de-normalization prior from the model's signal.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CTX_P = 12      # context periods
PRED_P = 4      # forecast periods
IMG = 224
IMN_MEAN = [0.485, 0.456, 0.406]
IMN_STD = [0.229, 0.224, 0.225]

# set by configure()
P = CONTEXT = HORIZON = None

DATASETS = {
    "ETTh1":       dict(kind="ett",    file="ETTh1.csv",       P=24,  borders=(8640, 11520, 14400)),
    "ETTh2":       dict(kind="ett",    file="ETTh2.csv",       P=24,  borders=(8640, 11520, 14400)),
    "ETTm1":       dict(kind="ett",    file="ETTm1.csv",       P=96,  borders=(34560, 46080, 57600)),
    "ETTm2":       dict(kind="ett",    file="ETTm2.csv",       P=96,  borders=(34560, 46080, 57600)),
    "electricity": dict(kind="lstnet", file="electricity.txt", P=24,  max_ch=64),
    "traffic":     dict(kind="lstnet", file="traffic.txt",     P=24,  max_ch=64),
    "solar":       dict(kind="lstnet", file="solar_AL.txt",    P=144, max_ch=64),
}


def configure(period):
    global P, CONTEXT, HORIZON
    P = period
    CONTEXT = P * CTX_P
    HORIZON = P * PRED_P


# ---------------------------------------------------------------- data
def load_dataset(name, data_dir):
    cfg = DATASETS[name]
    configure(cfg["P"])
    path = os.path.join(data_dir, cfg["file"])
    if cfg["kind"] == "ett":
        data = pd.read_csv(path).iloc[:, 1:].values.astype(np.float32)
        b_train, b_val, b_test = cfg["borders"]
    else:
        data = np.loadtxt(path, delimiter=",").astype(np.float32)
        T = len(data)
        b_train, b_val, b_test = int(0.7 * T), int(0.8 * T), T
        max_ch = cfg.get("max_ch")
        if max_ch and data.shape[1] > max_ch:
            orig_ch = data.shape[1]
            rng = np.random.default_rng(0)
            keep = np.sort(rng.choice(orig_ch, max_ch, replace=False))
            data = data[:, keep]
            print(f"[info] {name}: subsampled {max_ch}/{orig_ch} channels (seed 0)",
                  flush=True)
    train = data[:b_train]
    mean, std = train.mean(0), train.std(0) + 1e-8
    data = (data - mean) / std
    return data, (b_val, b_test)


def get_windows(data, test_span):
    test_start, test_end = test_span
    X, Y = [], []
    for t in range(test_start, test_end - HORIZON + 1, P):
        if t - CONTEXT < 0:
            continue
        X.append(data[t - CONTEXT:t])
        Y.append(data[t:t + HORIZON])
    X, Y = np.stack(X), np.stack(Y)                       # [N, L, C], [N, H, C]
    X = X.transpose(0, 2, 1).reshape(-1, CONTEXT)         # [N*C, L] channel-independent
    Y = Y.transpose(0, 2, 1).reshape(-1, HORIZON)
    return X, Y


# ---------------------------------------------------------------- baselines
def m_naive(X):
    return np.repeat(X[:, -1:], HORIZON, axis=1)


def m_snaive(X):
    return np.tile(X[:, -P:], (1, PRED_P))


def m_smean(X):
    return np.tile(X.reshape(-1, CTX_P, P).mean(1), (1, PRED_P))


# ---------------------------------------------------------------- VisionTS (image prior)
def m_visionts(X, batch=256):
    from visionts import VisionTS

    ckpt_dir = os.environ.get("VISIONTS_CKPT", "./ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    model = VisionTS(arch="mae_base", finetune_type="ln", load_ckpt=True,
                     ckpt_dir=ckpt_dir).to(DEVICE).eval()
    model.update_config(context_len=CONTEXT, pred_len=HORIZON, periodicity=P)

    preds = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().unsqueeze(-1).to(DEVICE)  # [B, L, 1]
        with torch.no_grad():
            y = model(xb)
        if isinstance(y, (tuple, list)):
            y = y[0]
        preds.append(y.squeeze(-1).float().cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------- VideoMAE (video prior)
def m_videomae(X, zero_pred=False, batch=64):
    import transformers
    assert transformers.__version__ < "5", \
        "transformers>=5 breaks VideoMAE checkpoint loading (q_bias/v_bias rename)"
    from transformers import VideoMAEForPreTraining

    model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base").to(DEVICE).eval()
    cfg = model.config
    ts, ps, nf = cfg.tubelet_size, cfg.patch_size, cfg.num_frames  # 2, 16, 16
    gh, tt = IMG // ps, nf // ts                                   # 14, 8
    assert CTX_P + PRED_P == nf, "context+pred periods must equal num_frames"
    vis_t = CTX_P // ts                                            # 6 visible tubelets

    mask3d = torch.zeros(tt, gh, gh, dtype=torch.bool)
    mask3d[vis_t:] = True
    bool_masked = mask3d.flatten()                                 # [tt*gh*gh]
    n_masked = int(bool_masked.sum())                              # 392
    mt = tt - vis_t                                                # 2 masked tubelets

    imn_mean = torch.tensor(IMN_MEAN, device=DEVICE).view(1, 1, 3, 1, 1)
    imn_std = torch.tensor(IMN_STD, device=DEVICE).view(1, 1, 3, 1, 1)

    preds = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float()
        B = xb.shape[0]
        mu = xb.mean(1, keepdim=True)
        sd = xb.std(1, keepdim=True) + 1e-8
        z = ((xb - mu) / (3 * sd)).clamp(-1, 1)
        g = (z + 1) / 2                                            # [0,1] grayscale
        frames = torch.full((B, nf, P), 0.5)
        frames[:, :CTX_P] = g.view(B, CTX_P, P)
        # each frame: P phase values -> vertical bands in a 224x224 image
        img = F.interpolate(frames.view(B * nf, 1, P, 1), size=(IMG, IMG),
                            mode="bilinear", align_corners=False)
        vid = img.view(B, nf, 1, IMG, IMG).repeat(1, 1, 3, 1, 1).to(DEVICE)
        vid = (vid - imn_mean) / imn_std

        # patchify our own video exactly as HF does:
        # "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c"
        v = vid.view(B, tt, ts, 3, gh, ps, gh, ps)
        v = v.permute(0, 1, 4, 6, 2, 5, 7, 3).reshape(B, tt, gh, gh, ts * ps * ps, 3)
        vis = v[:, :vis_t]                                         # [B, 6, 14, 14, 512, 3]
        mu_p = vis.mean(dim=4).mean(dim=1)                         # [B, 14, 14, 3]
        sd_p = (vis.var(dim=4, unbiased=True) + 1e-6).sqrt().mean(dim=1)

        if zero_pred:
            pred = torch.zeros(B, n_masked, ts * ps * ps * 3, device=DEVICE)
        else:
            with torch.no_grad():
                out = model(pixel_values=vid,
                            bool_masked_pos=bool_masked.unsqueeze(0).expand(B, -1).to(DEVICE))
            pred = out.logits                                      # [B, 392, 1536]

        # de-normalize with per-(h,w) stats from visible tubelets (seasonal-phase prior)
        pred = pred.view(B, mt, gh, gh, ts * ps * ps, 3)
        rec = pred * sd_p.view(B, 1, gh, gh, 1, 3) + mu_p.view(B, 1, gh, gh, 1, 3)

        # unpatchify masked tubelets -> frames [B, mt*ts, 3, 224, 224]
        rec = rec.view(B, mt, gh, gh, ts, ps, ps, 3)
        rec = rec.permute(0, 1, 4, 7, 2, 5, 3, 6).reshape(B, mt * ts, 3, IMG, IMG)

        # undo ImageNet norm, average channels -> grayscale [0,1]
        rec = rec * imn_std.view(1, 1, 3, 1, 1) + imn_mean.view(1, 1, 3, 1, 1)
        gray = rec.mean(2)                                         # [B, 4, 224, 224]
        vals = F.adaptive_avg_pool2d(gray, (P, 1)).squeeze(-1)     # [B, 4, P]
        zhat = 2 * vals.reshape(B, HORIZON) - 1
        yhat = zhat.cpu() * 3 * sd + mu
        preds.append(yhat.numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------- main
METHODS = {
    "naive": m_naive,
    "snaive": m_snaive,
    "smean": m_smean,
    "visionts": m_visionts,
    "videomae": m_videomae,
    "videomae_zero": lambda X: m_videomae(X, zero_pred=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--out-dir", default="pilot/results")
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--subsample-seed", type=int, default=123)
    args = ap.parse_args()

    data, test_span = load_dataset(args.dataset, args.data_dir)
    X, Y = get_windows(data, test_span)
    if args.subsample and args.subsample < len(X):
        idx = np.sort(np.random.default_rng(args.subsample_seed)
                      .choice(len(X), args.subsample, replace=False))
        X, Y = X[idx], Y[idx]
    print(f"dataset={args.dataset}  device={DEVICE}  series={X.shape[0]}  "
          f"P={P}  context={CONTEXT}  horizon={HORIZON}", flush=True)

    names = list(METHODS) if args.methods == "all" else args.methods.split(",")
    results = {}
    for name in names:
        try:
            pred = METHODS[name](X)
            mse = float(np.mean((pred - Y) ** 2))
            mae = float(np.mean(np.abs(pred - Y)))
            results[name] = {"MSE": round(mse, 4), "MAE": round(mae, 4)}
            print(f"[done] {name:14s} MSE={mse:.4f}  MAE={mae:.4f}", flush=True)
        except Exception:
            import traceback
            print(f"[fail] {name}", flush=True)
            traceback.print_exc()
            results[name] = {"error": True}

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f"_sub{args.subsample}" if args.subsample else ""
    out = os.path.join(args.out_dir, f"pilot_{args.dataset}{suffix}.json")
    with open(out, "w") as f:
        json.dump({"config": {"dataset": args.dataset, "P": P, "context": CONTEXT,
                              "horizon": HORIZON, "n_series": int(X.shape[0])},
                   "results": results}, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
