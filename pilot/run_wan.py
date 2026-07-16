"""
Zero-shot time series forecasting with Wan2.1-T2V-1.3B via temporal outpainting.

Rendering: each period -> FPP identical frames (phase -> horizontal bands, grayscale
= value), 12 context periods + lead frame + 4 forecast periods = 4k+1 frames.
The context video is VAE-encoded; during flow-matching sampling (Euler), after every
step the KNOWN latent frames (context) are replaced by freshly-noised ground truth at
the current sigma (RePaint-style), so the DiT only generates the future frames.
Causal VAE (4x temporal compression) => context/future boundary is exact at latent
frame 1 + 12*FPP/4.

Runs on a seeded subsample of test windows (diffusion sampling is expensive) and
reports naive/snaive/smean on the SAME subsample for comparison. Use
run_pilot.py --subsample with the same seed to get VisionTS/VideoMAE subset numbers.

Env: prograph conda env (diffusers 0.37, transformers 5.x is fine here - no VideoMAE).
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F

import run_pilot as rp

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
PROMPT = ("abstract grayscale pattern of horizontal bands, each band slowly and "
          "smoothly changing its brightness over time, minimal clean texture, "
          "no objects, no camera motion")
NEG = "sudden changes, flicker, objects, text, camera motion, colorful"


def render_video(x, fpp, H, W, y_est=None):
    """x [B, CONTEXT] -> video [B, 3, T, H, W] in [-1,1], plus (mu, sd) and T info.

    If y_est [B, HORIZON] is given, future frames render it (SDEdit init);
    otherwise future frames are neutral gray.
    """
    B = x.shape[0]
    mu = x.mean(1, keepdim=True)
    sd = x.std(1, keepdim=True) + 1e-8

    def to_gray(v):
        z = ((v - mu) / (3 * sd)).clamp(-1, 1)
        return (z + 1) / 2                                 # [0,1]

    per = to_gray(x).view(B, rp.CTX_P, rp.P)               # [B, 12, P]
    n_ctx_f = rp.CTX_P * fpp + 1                           # lead frame + context
    n_tot_f = n_ctx_f + rp.PRED_P * fpp
    frames = torch.full((B, n_tot_f, rp.P), 0.5)
    frames[:, 0] = per[:, 0]                               # lead = first period
    for j in range(rp.CTX_P):
        frames[:, 1 + j * fpp:1 + (j + 1) * fpp] = per[:, j:j + 1]
    if y_est is not None:
        fper = to_gray(y_est).view(B, rp.PRED_P, rp.P)
        for j in range(rp.PRED_P):
            frames[:, n_ctx_f + j * fpp:n_ctx_f + (j + 1) * fpp] = fper[:, j:j + 1]
    img = F.interpolate(frames.view(B * n_tot_f, 1, rp.P, 1), size=(H, W),
                        mode="bilinear", align_corners=False)
    vid = img.view(B, n_tot_f, 1, H, W).repeat(1, 1, 3, 1, 1)
    vid = vid.permute(0, 2, 1, 3, 4) * 2 - 1               # [B, 3, T, H, W]
    return vid, mu, sd, n_ctx_f, n_tot_f


def decode_forecast(video, fpp, mu, sd, n_ctx_f):
    """video [B, 3, T, H, W] in [-1,1] -> forecast [B, HORIZON]."""
    B = video.shape[0]
    gray = ((video.float() + 1) / 2).clamp(0, 1).mean(1)   # [B, T, H, W]
    pred = gray[:, n_ctx_f:]                               # [B, PRED_P*fpp, H, W]
    pred = pred.view(B, rp.PRED_P, fpp, *pred.shape[2:]).mean(2)   # avg frames/period
    vals = F.adaptive_avg_pool2d(pred, (rp.P, 1)).squeeze(-1)      # [B, PRED_P, P]
    zhat = 2 * vals.reshape(B, rp.HORIZON) - 1
    return zhat.cpu() * 3 * sd + mu


@torch.no_grad()
def wan_forecast(X, args):
    from diffusers import WanPipeline
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=DTYPE)
    if os.environ.get("WAN_LORA"):
        from peft import PeftModel
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer, os.environ["WAN_LORA"],
            torch_dtype=DTYPE).merge_and_unload()
        print(f"[info] merged LoRA from {os.environ['WAN_LORA']}", flush=True)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=args.shift)
    pipe.to(DEVICE)

    prompt = "" if args.prompt == "none" else PROMPT
    pe, npe = pipe.encode_prompt(prompt=prompt, negative_prompt=NEG,
                                 do_classifier_free_guidance=args.guidance > 1,
                                 device=DEVICE)
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    vae, tf, sch = pipe.vae, pipe.transformer, pipe.scheduler
    lm = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(DEVICE, DTYPE)
    ls = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(DEVICE, DTYPE)

    preds = []
    g = torch.Generator(device=DEVICE).manual_seed(0)
    for i in range(0, len(X), args.batch):
        xb = torch.from_numpy(X[i:i + args.batch]).float()
        B = xb.shape[0]
        y0 = None
        if args.sdedit > 0:                                # init future with baseline
            init_fn = {"smean": rp.m_smean, "snaive": rp.m_snaive}[args.sdedit_init]
            y0 = torch.from_numpy(init_fn(X[i:i + args.batch])).float()
        vid, mu, sd, n_ctx_f, n_tot_f = render_video(xb, args.fpp, args.height,
                                                     args.width, y_est=y0)
        lat_gt = vae.encode(vid.to(DEVICE, DTYPE)).latent_dist.mode()
        lat_gt = ((lat_gt - lm) / ls).float()              # normalized latents
        n_known = 1 + (n_ctx_f - 1) // 4                   # known latent frames

        sch.set_timesteps(args.steps, device=DEVICE)
        sigmas = sch.sigmas                                # [steps+1], 1 -> 0
        noise = torch.randn(lat_gt.shape, generator=g, device=DEVICE)
        si0 = 0
        if args.sdedit > 0:                                # start mid-schedule
            si0 = int(next(i for i, s in enumerate(sigmas[:-1]) if s <= args.sdedit))
            lat = (1 - sigmas[si0]) * lat_gt + sigmas[si0] * noise
        else:
            lat = noise.clone()                            # sigma=1 start
        for si, t in list(enumerate(sch.timesteps))[si0:]:
            # RePaint: clamp known region to noised GT at current sigma
            s = sigmas[si]
            eps = torch.randn(lat_gt.shape, generator=g, device=DEVICE)
            known = (1 - s) * lat_gt + s * eps
            lat[:, :, :n_known] = known[:, :, :n_known]

            lat_in = lat.to(DTYPE)
            ts = t.expand(B)
            out = tf(hidden_states=lat_in, timestep=ts,
                     encoder_hidden_states=pe, return_dict=False)[0]
            if args.guidance > 1:
                out_u = tf(hidden_states=lat_in, timestep=ts,
                           encoder_hidden_states=npe, return_dict=False)[0]
                out = out_u + args.guidance * (out - out_u)
            lat = sch.step(out.float(), t, lat, return_dict=False)[0]
        # final: known region = exact GT
        lat[:, :, :n_known] = lat_gt[:, :, :n_known]

        dec = (lat.to(DTYPE) * ls + lm)
        video = vae.decode(dec, return_dict=False)[0]      # [B, 3, T, H, W] [-1,1]
        preds.append(decode_forecast(video, args.fpp, mu, sd, n_ctx_f).numpy())

        if i == 0:
            gi = ((vid[0].float() + 1) / 2).clamp(0, 1).mean(0)
            go = ((video[0].float().cpu() + 1) / 2).clamp(0, 1).mean(0)
            ctx_mse = float(((gi[1:n_ctx_f] - go[1:n_ctx_f]) ** 2).mean())
            print(f"[info] context recon MSE (grayscale): {ctx_mse:.5f}", flush=True)
            if args.save_probe:
                try:
                    save_probe(vid, video, args.save_probe, n_ctx_f)
                except Exception as e:
                    print(f"[warn] probe save failed: {e}", flush=True)
        print(f"[info] {min(i + args.batch, len(X))}/{len(X)} windows", flush=True)
    return np.concatenate(preds)


def save_probe(vid_in, vid_out, path, n_ctx_f):
    """Save frame strips spanning the context/future boundary as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gi = ((vid_in[0].float() + 1) / 2).clamp(0, 1).mean(0)     # [T, H, W]
    go = ((vid_out[0].float().cpu() + 1) / 2).clamp(0, 1).mean(0)
    T = gi.shape[0]
    frames_to_show = [n_ctx_f - 8, n_ctx_f - 4, n_ctx_f - 1] + \
        [min(n_ctx_f + k, T - 1) for k in (0, 3, 7, 11, 15)]
    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    for k, fr in enumerate(frames_to_show):
        axes[0, k].imshow(gi[fr], cmap="gray", vmin=0, vmax=1)
        axes[0, k].set_title(f"in f{fr}", fontsize=8)
        axes[1, k].imshow(go[fr], cmap="gray", vmin=0, vmax=1)
        axes[1, k].set_title(f"gen f{fr}", fontsize=8)
        for ax in (axes[0, k], axes[1, k]):
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close()
    print(f"[info] probe image -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(rp.DATASETS))
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--out-dir", default="pilot/results_wan")
    ap.add_argument("--subsample", type=int, default=96)
    ap.add_argument("--subsample-seed", type=int, default=123)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=416)
    ap.add_argument("--fpp", type=int, default=4, help="frames per period")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sdedit", type=float, default=0.0,
                    help="if >0, init future with a baseline and start at this sigma")
    ap.add_argument("--sdedit-init", choices=["smean", "snaive"], default="smean")
    ap.add_argument("--prompt", choices=["bands", "none"], default="bands")
    ap.add_argument("--save-probe", default="")
    args = ap.parse_args()
    assert (rp.CTX_P * args.fpp) % 4 == 0 and (rp.PRED_P * args.fpp) % 4 == 0
    os.makedirs(args.out_dir, exist_ok=True)

    data, test_span = rp.load_dataset(args.dataset, args.data_dir)
    X, Y = rp.get_windows(data, test_span)
    if args.subsample and args.subsample < len(X):
        idx = np.sort(np.random.default_rng(args.subsample_seed)
                      .choice(len(X), args.subsample, replace=False))
        X, Y = X[idx], Y[idx]
    print(f"dataset={args.dataset}  n={len(X)}  P={rp.P}  context={rp.CONTEXT}  "
          f"horizon={rp.HORIZON}  res={args.height}x{args.width}  fpp={args.fpp}  "
          f"steps={args.steps}  cfg={args.guidance}", flush=True)

    results = {}
    for name, fn in [("naive", rp.m_naive), ("snaive", rp.m_snaive),
                     ("smean", rp.m_smean)]:
        p = fn(X)
        results[name] = {"MSE": round(float(np.mean((p - Y) ** 2)), 4),
                         "MAE": round(float(np.mean(np.abs(p - Y))), 4)}
    try:
        p = wan_forecast(X, args)
        results["wan21_1.3b"] = {"MSE": round(float(np.mean((p - Y) ** 2)), 4),
                                 "MAE": round(float(np.mean(np.abs(p - Y))), 4)}
    except Exception:
        import traceback
        traceback.print_exc()
        results["wan21_1.3b"] = {"error": True}

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"wan_{args.dataset}.json"), "w") as f:
        json.dump({"config": vars(args) | {"P": rp.P, "context": rp.CONTEXT,
                                           "horizon": rp.HORIZON, "n": len(X)},
                   "results": results}, f, indent=2, default=str)
    for k, v in results.items():
        print(f"[done] {k:12s} {v}", flush=True)


if __name__ == "__main__":
    main()
