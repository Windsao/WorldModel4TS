"""LTX-TS: VACE's LTX-Video backend as a forecaster (other-architecture generative test).
Same render/decode bridge as run_vace_ts but using LTXVace (LTX-Video 2B) instead of Wan.
Render TS context -> bands video, mask future frames, LTXVace inpaints, decode back.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

VACE_DIR = os.environ.get("VACE_DIR", "/home/mzh1800/WorldModel4TS/VACE")
sys.path.insert(0, VACE_DIR); sys.path.insert(0, os.path.join(VACE_DIR, "vace"))

P, CTX_P, PRED_P = 24, 12, 4
PROMPT = ("abstract grayscale pattern of horizontal bands, each band slowly and smoothly "
          "changing its brightness over time, minimal clean texture, no objects, no camera motion")


def load_etth1(path):
    import pandas as pd
    d = pd.read_csv(path).iloc[:, 1:].values.astype(np.float32)
    b_train = 8640
    mean, std = d[:b_train].mean(0), d[:b_train].std(0) + 1e-8
    return (d - mean) / std, (b_train, 11520, 14400)


def windows(data, borders, n, seed=123):
    ctx, hor = CTX_P * P, PRED_P * P
    lo, hi = borders[1], borders[2] - hor
    ts = np.sort(np.random.default_rng(seed).choice(np.arange(lo + ctx, hi), n, False))
    X = np.stack([data[t - ctx:t] for t in ts]); Y = np.stack([data[t:t + hor] for t in ts])
    return X.astype(np.float32), Y.astype(np.float32)


def render_pair(x, fpp, H, W):
    mu, sd = x.mean(), x.std() + 1e-8
    g = ((torch.tensor(x) - mu) / (3 * sd)).clamp(-1, 1).add(1).div(2).view(CTX_P, P)
    n_ctx = CTX_P * fpp + 1
    n_tot = n_ctx + PRED_P * fpp
    frames = torch.full((n_tot, P), 0.5)
    frames[0] = g[0]
    for j in range(CTX_P):
        frames[1 + j * fpp:1 + (j + 1) * fpp] = g[j]
    img = F.interpolate(frames.view(n_tot, 1, P, 1), size=(H, W), mode="bilinear", align_corners=False)
    vid = img.repeat(1, 3, 1, 1).permute(1, 0, 2, 3) * 2 - 1     # [3,T,H,W] [-1,1]
    mask = torch.zeros(1, n_tot, H, W); mask[:, n_ctx:] = 1.0
    return vid.float(), mask.float(), float(mu), float(sd), n_ctx, n_tot


def to_video_tensor(out):
    """LTXVace.generate output -> [3,T,H,W] in [-1,1]."""
    if isinstance(out, dict):
        out = out.get("video", out.get("frames", list(out.values())[0]))
    v = out[0] if isinstance(out, (list, tuple)) else out
    v = torch.as_tensor(v).float().squeeze()
    if v.dim() == 4 and v.shape[-1] == 3: v = v.permute(3, 0, 1, 2)   # THWC->CTHW
    if v.max() > 1.5: v = v / 127.5 - 1                                # 0..255 -> [-1,1]
    elif v.min() >= 0: v = v * 2 - 1                                   # [0,1] -> [-1,1]
    return v


def decode(video, n_ctx, mu, sd, fpp):
    g = ((video.float() + 1) / 2).clamp(0, 1).mean(0)
    fut = g[n_ctx:].view(PRED_P, fpp, *g.shape[1:]).mean(1)
    vals = F.adaptive_avg_pool2d(fut, (P, 1)).squeeze(-1)
    return (2 * vals.reshape(-1) - 1).cpu().numpy() * 3 * sd + mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/nyx-storage1/hanliu/wm4ts/data")
    ap.add_argument("--ckpt-dir", required=True, help="VACE-LTX snapshot dir")
    ap.add_argument("--out-dir", default="pilot/results_field/vace")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--fpp", type=int, default=6)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--context-scale", type=float, default=1.0)
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=768)
    args = ap.parse_args()

    data, borders = load_etth1(os.path.join(args.data_dir, "ETTh1.csv"))
    X, Y = windows(data, borders, args.n)
    base = X.reshape(len(X), CTX_P, P, X.shape[-1]).mean(1)
    smean = np.tile(base, (1, PRED_P, 1))
    sm_mse = float(np.mean((smean - Y) ** 2)); sm_mae = float(np.mean(np.abs(smean - Y)))
    print(f"[cfg] LTX ETTh1 n={len(X)} steps={args.steps} cs={args.context_scale} {args.W}x{args.H}", flush=True)
    print(f"[done] smean MSE={sm_mse:.4f} MAE={sm_mae:.4f}", flush=True)

    from models.ltx.ltx_vace import LTXVace
    ckpt = os.path.join(args.ckpt_dir, "ltx-video-2b-v0.9.safetensors")
    tenc = args.ckpt_dir
    print(f"[info] loading LTXVace from {ckpt}", flush=True)
    ltx = LTXVace(ckpt_path=ckpt, text_encoder_path=tenc, precision="bfloat16", offload_to_cpu=True)

    preds = np.zeros((len(X), PRED_P * P, X.shape[-1]), dtype=np.float32)
    for ci in range(X.shape[-1]):
        for i in range(len(X)):
            vid, mask, mu, sd, n_ctx, n_tot = render_pair(X[i, :, ci], args.fpp, args.H, args.W)
            out = ltx.generate(src_video=[vid.to("cuda")], src_mask=[mask.to("cuda")], src_ref_images=[],
                               prompt=PROMPT, negative_prompt="", seed=0,
                               num_inference_steps=args.steps, context_scale=args.context_scale,
                               output_height=args.H, output_width=args.W, num_frames=n_tot)
            preds[i, :, ci] = decode(to_video_tensor(out), n_ctx, mu, sd, args.fpp)
        print(f"[info] channel {ci+1}/{X.shape[-1]} done", flush=True)

    mse = float(np.mean((preds - Y) ** 2)); mae = float(np.mean(np.abs(preds - Y)))
    res = {"smean": {"MSE": round(sm_mse, 4), "MAE": round(sm_mae, 4)},
           "ltx": {"MSE": round(mse, 4), "MAE": round(mae, 4)}}
    print(f"[done] ltx MSE={mse:.4f} MAE={mae:.4f}  (smean {sm_mse:.4f})", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"ltx_ETTh1_cs{args.context_scale}.json"), "w") as f:
        json.dump({"config": vars(args), "results": res}, f, indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
