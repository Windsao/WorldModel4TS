"""Stages B/C: VideoMAE historical-reconstruction factorization (spec section 6).

Regime HA. Everything reconstructed lies INSIDE observed history, so true pixels and true
values are known without touching a real forecast target. This separates the four candidate
explanations for strict-zero-shot failure:

  D1 decoder-domain   -> I1 oracle_stats works but I2 causal_stats does not
  D2 mask mismatch    -> random_tube_* works but right_* does not
  D3 representation   -> dense_* works but line_rolling_legacy does not
  D4 task mismatch    -> historical right-block works but the genuine future (Stage D) does not

A forecasting MSE alone cannot distinguish these, so the primary metric here is
`native_ratio` = MSE(logits, native target) / MSE(zeros, native target), measured against
the exact normalized-cube target the checkpoint was actually trained on.
"""

import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_field as RF
import videomae_patch_utils as VP
import video_visionts_renderers as VR
import run_reconstruct as RC

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID, PATCH, NF, TUB = 14, 16, 16, 2
HORIZON = {"ETTh2": 96, "ETTm2": 96, "ETTh1": 96, "electricity": 96, "traffic": 96, "solar": 144}
MASKS = ["random_tube_90", "random_tube_75", "right_13", "right_10", "right_3"]


def build_full_cubes(vid_full):
    frames = VP.unnormalize(vid_full)
    return VP.patchify(frames)                                  # [B,N,512,3]


def cubes_to_image(cubes, frame_idx=NF - 1):
    """[B,N,512,3] -> gray image of one frame, [B,224,224]."""
    vid = VP.unpatchify(cubes)                                  # [B,16,3,224,224]
    return vid[:, frame_idx].mean(1).clamp(0, 1) * 2 - 1        # to [-1,1] gray


def audit_one(model, vid_in, vid_full, bm, cols, P, H, mu, sd, y_true, want_ts):
    """One (renderer, mask) cell. Returns a metrics dict."""
    B = vid_in.shape[0]
    with torch.no_grad():
        logits = model(pixel_values=vid_in, bool_masked_pos=bm).logits    # [B,n,1536]
    tgt, cmean, cstd = VP.native_targets(vid_full)
    labels = tgt[bm].view(B, -1, 1536)
    native_model = float(F.mse_loss(logits, labels))
    native_zero = float(F.mse_loss(torch.zeros_like(logits), labels))
    out = {"native_model": round(native_model, 6), "native_zero": round(native_zero, 6),
           "native_ratio": round(native_model / (native_zero + 1e-12), 6)}
    out["native_gain"] = round(1 - out["native_ratio"], 6)

    cubes_true = build_full_cubes(vid_full)                     # [B,N,512,3]
    C = cubes_true.shape[-1]
    sel_mean = cmean[bm].view(B, -1, 1, C)
    sel_std = cstd[bm].view(B, -1, 1, C)
    raw_true = cubes_true[bm].view(B, -1, 512, C)

    # I1 oracle: use the TRUE per-cube stats (diagnostic upper bound, not a forecast)
    raw_oracle = VP.denormalize_cubes(logits, sel_mean, sel_std)
    out["raw_oracle_mse"] = round(float(F.mse_loss(raw_oracle, raw_true)), 6)
    raw_zero_oracle = VP.denormalize_cubes(torch.zeros_like(logits), sel_mean, sel_std)
    out["raw_oracle_zero_mse"] = round(float(F.mse_loss(raw_zero_oracle, raw_true)), 6)

    # I2 causal: estimate stats without hidden values
    causal = {}
    for rule in ("nearest_visible_left", "context_global"):
        cm, cs = VP.causal_cube_stats(cubes_true, bm, rule=rule)
        raw_c = VP.denormalize_cubes(logits, cm, cs)
        causal[rule] = round(float(F.mse_loss(raw_c, raw_true)), 6)
    out["raw_causal_mse"] = causal

    if want_ts and cols is not None:
        # place predictions back and invert to a time series (right-block masks only)
        for tag, lg in (("model", logits), ("zeroed", torch.zeros_like(logits))):
            for stat, (m_, s_) in (("oracle", (sel_mean, sel_std)),
                                   ("causal", VP.causal_cube_stats(cubes_true, bm,
                                                                   "nearest_visible_left"))):
                filled = cubes_true.clone()
                filled[bm] = VP.denormalize_cubes(lg, m_, s_).view(-1, 512, C)
                img = cubes_to_image(filled)
                vis_w = (GRID - cols) * PATCH
                yh = VR.invert_dense(img[:, :, vis_w:], P, H, mu, sd)
                e = (yh.cpu().numpy() - y_true)
                out[f"ts_{tag}_{stat}_mse"] = round(float(np.mean(e ** 2)), 6)
                out[f"ts_{tag}_{stat}_mae"] = round(float(np.mean(np.abs(e))), 6)
        # patch-boundary seam error on the reconstruction
        filled = cubes_true.clone()
        filled[bm] = VP.denormalize_cubes(logits, sel_mean, sel_std).view(-1, 512, C)
        img = cubes_to_image(filled)
        vis_w = (GRID - cols) * PATCH
        seam = float((img[:, :, vis_w - 1] - img[:, :, vis_w]).abs().mean())
        out["seam_error"] = round(seam, 6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HORIZON))
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default="pilot/results_field/video_visionts")
    ap.add_argument("--ckpt", default=os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base"))
    ap.add_argument("--renderer", required=True, choices=list(VR.RENDERERS))
    ap.add_argument("--mask", required=True, choices=MASKS)
    ap.add_argument("--mask-seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--random-init", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    data, borders = RF.load_mv(args.dataset, args.data_dir, args.max_ch)
    P = RF.P; L = NF * P; H = HORIZON[args.dataset]
    # HISTORICAL AUDIT: the "hidden" block is known history, so we need L+H (+15P rolling)
    extra = (NF - 1) * P if args.renderer == "dense_rolling" else 0
    need = L + extra + H
    Xh, _ = RF.windows(data, borders, "train", need, 1, args.stride, args.n * 4 + 500)
    M = data.shape[1]
    Xu = Xh.transpose(0, 2, 1).reshape(-1, need, 1)[:, :, 0]
    if len(Xu) > args.n:
        idx = np.random.default_rng(args.seed).choice(len(Xu), args.n, False); idx.sort()
        Xu = Xu[idx]
    print(f"{args.dataset} {args.renderer} {args.mask} P={P} L={L} H={H} n={len(Xu)}", flush=True)

    from transformers import VideoMAEForPreTraining, VideoMAEConfig
    if args.random_init:
        model = VideoMAEForPreTraining(VideoMAEConfig.from_pretrained(args.ckpt))
    else:
        model = VideoMAEForPreTraining.from_pretrained(args.ckpt)
    model = model.to(DEV).eval()
    assert model.config.norm_pix_loss is True

    cols = VP.masked_columns(args.mask)
    want_ts = cols is not None and args.renderer != "line_rolling_legacy"
    accum, nb = {}, 0
    for i in range(0, len(Xu), args.batch):
        xb = torch.from_numpy(Xu[i:i + args.batch].astype(np.float32)).to(DEV)
        B = xb.shape[0]
        if args.renderer == "dense_static":
            ctx, fut = xb[:, :L], xb[:, L:L + H]
            vi, vf, mu, sd = VR.dense_static(ctx, fut, P, cols or 3)
            y_true = fut.unsqueeze(-1).cpu().numpy()
        elif args.renderer == "dense_rolling":
            vi, vf, mu, sd = VR.dense_rolling(xb, P, H, cols or 3, L)
            y_true = xb[:, L + (NF - 1) * P:L + (NF - 1) * P + H].unsqueeze(-1).cpu().numpy()
        else:
            cL = int(RC.CTX_PX * H / RC.FUT_PX) + NF * 4 + P
            ctx = xb[:, :cL].unsqueeze(-1)
            vi, mu, sd = RC.render_rolling(ctx, H)
            vf = vi                       # legacy control never renders the hidden block
            y_true = xb[:, cL:cL + H].unsqueeze(-1).cpu().numpy()
        bm = VP.make_mask(args.mask, B, DEV, seed=args.mask_seed)
        m = audit_one(model, vi, vf, bm, cols, P, H, mu, sd, y_true, want_ts)
        for k, v in m.items():
            if isinstance(v, dict):
                accum.setdefault(k, {})
                for kk, vv in v.items():
                    accum[k][kk] = accum[k].get(kk, 0.0) + vv
            else:
                accum[k] = accum.get(k, 0.0) + v
        nb += 1
    met = {}
    for k, v in accum.items():
        met[k] = {kk: round(vv / nb, 6) for kk, vv in v.items()} if isinstance(v, dict) \
            else round(v / nb, 6)
    met["native_ratio"] = round(met["native_model"] / (met["native_zero"] + 1e-12), 6)
    met["native_gain"] = round(1 - met["native_ratio"], 6)

    # baselines on the same pseudo-target
    reps = (H + P - 1) // P; G = L // P
    off = (NF - 1) * P if args.renderer == "dense_rolling" else 0
    ctxn = Xu[:, off:off + L]
    base = ctxn.reshape(len(Xu), G, P).mean(1)
    yt = Xu[:, off + L:off + L + H]
    met["smean"] = round(float(np.mean((np.tile(base, (1, reps))[:, :H] - yt) ** 2)), 6)
    met["snaive"] = round(float(np.mean((np.tile(ctxn[:, -P:], (1, reps))[:, :H] - yt) ** 2)), 6)

    import transformers
    out = {"status": "complete", "stage": "C", "regime": "HA", "dataset": args.dataset,
           "renderer": args.renderer, "mask": args.mask,
           "mask_spatial_count": VP.MASK_SPATIAL_COUNT[args.mask],
           "mask_ratio": round(VP.MASK_SPATIAL_COUNT[args.mask] / 196, 4),
           "mask_seed": args.mask_seed, "masked_cols": cols,
           "control": "random" if args.random_init else "pretrained",
           "checkpoint": args.ckpt, "norm_pix_loss": True,
           "git_commit": os.environ.get("WM4TS_GIT_COMMIT", "unknown"),
           "git_dirty": os.environ.get("WM4TS_GIT_DIRTY") == "1",
           "torch": torch.__version__, "transformers": transformers.__version__,
           "P": P, "context": L, "horizon": H, "M": M, "n_eval": int(len(Xu)),
           "stride": args.stride, "seed": args.seed, "batch": args.batch,
           "metrics": met, "wall_clock_s": round(time.time() - t0, 1),
           "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEV == "cuda" else None}
    d = os.path.join(args.root, "historical_reconstruction"); os.makedirs(d, exist_ok=True)
    ctl = "rand" if args.random_init else "pt"
    fn = f"{args.dataset}_{args.renderer}_{args.mask}_ms{args.mask_seed}_{ctl}.json"
    json.dump(out, open(os.path.join(d, fn), "w"), indent=2)
    print(json.dumps(met, indent=2), flush=True)


if __name__ == "__main__":
    main()
