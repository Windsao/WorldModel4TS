"""VACE-TS: use the VACE (Wan) video *editing* model as a forecaster via masked
temporal inpainting. Render TS context as a bands video; append gray, MASKED future
frames; VACE inpaints the future; decode the generated future frames back to values.

This is the trained-inpainting analog of jiale's run_wan (which hand-RePainted vanilla
Wan and was a "copy machine"). Runs on a few ETTh1 test windows; reports MSE/MAE vs
smean on the same windows. Env: open-sora (flash_attn+diffusers) + patched VACE clone.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

VACE_DIR = os.environ.get("VACE_DIR", "/home/mzh1800/WorldModel4TS/VACE")
sys.path.insert(0, VACE_DIR)

P, CTX_P, PRED_P = 24, 12, 4          # period, context periods, forecast periods
PROMPT = ("abstract grayscale pattern of horizontal bands, each band slowly and smoothly "
          "changing its brightness over time, minimal clean texture, no objects, no camera motion")


def load_etth1(path):
    import pandas as pd
    d = pd.read_csv(path).iloc[:, 1:].values.astype(np.float32)   # [T, 7]
    b_train = 8640
    mean, std = d[:b_train].mean(0), d[:b_train].std(0) + 1e-8
    return (d - mean) / std, (b_train, 11520, 14400)


def windows(data, borders, n, seed=123):
    ctx, hor = CTX_P * P, PRED_P * P
    lo, hi = borders[1], borders[2] - hor
    ts = np.sort(np.random.default_rng(seed).choice(np.arange(lo + ctx, hi), n, False))
    X = np.stack([data[t - ctx:t] for t in ts])
    Y = np.stack([data[t:t + hor] for t in ts])
    return X.astype(np.float32), Y.astype(np.float32)


def render_pair(x, fpp, H, W):
    """x [ctx] one channel -> (src_video [3,T,H,W] in [-1,1], src_mask [1,T,H,W] 0/1, mu, sd).
    context frames = real bands; future frames = gray 0.5 and MASKED (mask=1)."""
    mu, sd = x.mean(), x.std() + 1e-8
    def gray(v): return ((torch.tensor(v) - mu) / (3 * sd)).clamp(-1, 1).add(1).div(2)  # [0,1]
    per = gray(x).view(CTX_P, P)                              # [CTX_P, P]
    n_ctx = CTX_P * fpp + 1
    n_tot = n_ctx + PRED_P * fpp
    frames = torch.full((n_tot, P), 0.5)
    frames[0] = per[0]
    for j in range(CTX_P):
        frames[1 + j * fpp:1 + (j + 1) * fpp] = per[j]
    img = F.interpolate(frames.view(n_tot, 1, P, 1), size=(H, W), mode="bilinear", align_corners=False)
    vid = img.repeat(1, 3, 1, 1).permute(1, 0, 2, 3) * 2 - 1  # [3, T, H, W] in [-1,1]
    mask = torch.zeros(1, n_tot, H, W)
    mask[:, n_ctx:] = 1.0                                     # future = inpaint
    return vid.float(), mask.float(), float(mu), float(sd), n_ctx, n_tot


def decode(video, n_ctx, mu, sd, fpp):
    """video [3,T,H,W] in [-1,1] -> forecast [PRED_P*P]."""
    g = ((video.float() + 1) / 2).clamp(0, 1).mean(0)        # [T,H,W]
    fut = g[n_ctx:]                                          # [PRED_P*fpp, H, W]
    fut = fut.view(PRED_P, fpp, *fut.shape[1:]).mean(1)      # [PRED_P, H, W]
    vals = F.adaptive_avg_pool2d(fut, (P, 1)).squeeze(-1)    # [PRED_P, P]
    return (2 * vals.reshape(-1) - 1).cpu().numpy() * 3 * sd + mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/nyx-storage1/hanliu/wm4ts/data")
    ap.add_argument("--ckpt-dir", default=None, help="VACE checkpoint dir (1.3B or 14B)")
    ap.add_argument("--model-name", default="vace-1.3B")
    ap.add_argument("--out-dir", default="pilot/results_field/vace")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--fpp", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guide", type=float, default=1.0)
    ap.add_argument("--context-scale", type=float, default=1.0)
    ap.add_argument("--size", default="480p")
    args = ap.parse_args()

    data, borders = load_etth1(os.path.join(args.data_dir, "ETTh1.csv"))
    X, Y = windows(data, borders, args.n)
    hor = PRED_P * P
    # smean baseline on the same windows
    base = X.reshape(len(X), CTX_P, P, X.shape[-1]).mean(1)
    smean = np.tile(base, (1, PRED_P, 1))
    sm_mse = float(np.mean((smean - Y) ** 2)); sm_mae = float(np.mean(np.abs(smean - Y)))
    print(f"[cfg] ETTh1 n={len(X)} ctx={CTX_P*P} horizon={hor} model={args.model_name} steps={args.steps} cs={args.context_scale}", flush=True)
    print(f"[done] smean MSE={sm_mse:.4f} MAE={sm_mae:.4f}", flush=True)

    from vace.models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS
    from vace.models.wan.wan_vace import WanVace
    cfg = WAN_CONFIGS[args.model_name]
    H, W = SIZE_CONFIGS[args.size][1], SIZE_CONFIGS[args.size][0]
    ckpt = args.ckpt_dir or os.path.expanduser("~/.cache/huggingface")
    print(f"[info] loading WanVace {args.model_name} from {ckpt} ({W}x{H})", flush=True)
    wan = WanVace(config=cfg, checkpoint_dir=ckpt, device_id=0, rank=0,
                  t5_fsdp=False, dit_fsdp=False, use_usp=False)

    preds = np.zeros((len(X), hor, X.shape[-1]), dtype=np.float32)
    for ci in range(X.shape[-1]):                            # per channel (univariate video)
        for i in range(len(X)):
            vid, mask, mu, sd, n_ctx, n_tot = render_pair(X[i, :, ci], args.fpp, H, W)
            video = wan.generate(PROMPT, [vid.to("cuda")], [mask.to("cuda")], [None],
                                 size=SIZE_CONFIGS[args.size], frame_num=n_tot,
                                 context_scale=args.context_scale, shift=3.0,
                                 sample_solver="unipc", sampling_steps=args.steps,
                                 guide_scale=args.guide, n_prompt="", seed=0, offload_model=True)
            preds[i, :, ci] = decode(video, n_ctx, mu, sd, args.fpp)
        print(f"[info] channel {ci+1}/{X.shape[-1]} done", flush=True)

    mse = float(np.mean((preds - Y) ** 2)); mae = float(np.mean(np.abs(preds - Y)))
    res = {"smean": {"MSE": round(sm_mse, 4), "MAE": round(sm_mae, 4)},
           "vace": {"MSE": round(mse, 4), "MAE": round(mae, 4)}}
    print(f"[done] vace MSE={mse:.4f} MAE={mae:.4f}  (smean {sm_mse:.4f})", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"vace_ETTh1_{args.model_name}_cs{args.context_scale}.json"), "w") as f:
        json.dump({"config": vars(args), "results": res}, f, indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
