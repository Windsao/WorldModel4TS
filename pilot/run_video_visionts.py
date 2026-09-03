"""Stage D: strict zero-shot forecasting with the frozen Stage-C configuration (spec s.7).

Regime ZS: no gradients, no trained head, no calibration. The renderer, mask, normalization
and stat-recovery rule are FROZEN from the Stage-C historical audit before any genuine
future value is read; futures are used only to compute metrics.

Controls (spec 7.1): pretrained / zeroed logits / random-init VideoMAE / VisionTS / smean /
snaive, all on byte-identical windows.
"""

import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "third_party", "VisionTS"))
import run_field as RF
import videomae_patch_utils as VP
import video_visionts_renderers as VR
from run_visionts_reference import HORIZON, build_manifest, gather, vts_forward, git_info

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID, PATCH, NF = 14, 16, 16


def forecast(model, xb, P, H, L, renderer, mask_name, cols, stat_rule, zero_logits=False):
    """Strict-ZS forecast. The hidden block is BLANK in the model input, so no future value
    can reach the encoder; `vid_full` is never constructed here."""
    B = xb.shape[0]
    if renderer == "dense_static":
        ctx = xb[:, :L]
        blank = torch.zeros(B, H, device=xb.device, dtype=xb.dtype)
        vi, _, mu, sd = VR.dense_static(ctx, blank, P, cols)
    else:
        ext = torch.cat([xb[:, :L + (NF - 1) * P],
                         torch.zeros(B, H, device=xb.device, dtype=xb.dtype)], 1)
        vi, _, mu, sd = VR.dense_rolling(ext, P, H, cols, L)
    bm = VP.make_mask(mask_name, B, xb.device)
    with torch.no_grad():
        logits = model(pixel_values=vi, bool_masked_pos=bm).logits
    if zero_logits:
        logits = torch.zeros_like(logits)
    cubes_in = VP.patchify(VP.unnormalize(vi))
    cm, cs = VP.causal_cube_stats(cubes_in, bm, rule=stat_rule)
    filled = cubes_in.clone()
    filled[bm] = VP.denormalize_cubes(logits, cm, cs).view(-1, 512, cubes_in.shape[-1])
    vid = VP.unpatchify(filled)
    img = vid[:, NF - 1].mean(1).clamp(0, 1) * 2 - 1
    vis_w = (GRID - cols) * PATCH
    return VR.invert_dense(img[:, :, vis_w:], P, H, mu, sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HORIZON))
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default="pilot/results_field/video_visionts")
    ap.add_argument("--ckpt", default=os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base"))
    ap.add_argument("--vts-ckpt-dir", default="./ckpt/")
    ap.add_argument("--renderer", default="dense_static", choices=["dense_static", "dense_rolling"])
    ap.add_argument("--mask", default="right_10")
    ap.add_argument("--stat-rule", default="nearest_visible_left")
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    data, borders, P, L, H, Xte, Yte, pairs, man = build_manifest(
        args.dataset, args.data_dir, args.max_ch, args.stride, args.n, args.seed)
    Xu, Yu = gather(Xte, Yte, pairs)
    cols = VP.masked_columns(args.mask)
    need = L + (NF - 1) * P if args.renderer == "dense_rolling" else L
    if need > L:      # rolling needs extra history: rebuild windows with a longer context
        Xh, Yh = RF.windows(data, borders, "test", need, H, args.stride)
        w, c = pairs[:, 0], pairs[:, 1]
        w = np.clip(w, 0, len(Xh) - 1)
        Xu = Xh[w, :, c][:, :, None].astype(np.float32)
        Yu = Yh[w, :, c][:, :, None].astype(np.float32)
    print(f"{args.dataset} {args.renderer}/{args.mask} P={P} L={L} H={H} n={len(Xu)}", flush=True)

    from transformers import VideoMAEForPreTraining, VideoMAEConfig
    mp = VideoMAEForPreTraining.from_pretrained(args.ckpt).to(DEV).eval()
    torch.manual_seed(args.seed)
    mr = VideoMAEForPreTraining(VideoMAEConfig.from_pretrained(args.ckpt)).to(DEV).eval()

    # leakage test: the encoder input must be invariant to the evaluation buffer
    xb = torch.from_numpy(Xu[:4]).to(DEV)[:, :, 0]
    a = forecast(mp, xb, P, H, L, args.renderer, args.mask, cols, args.stat_rule)
    b = forecast(mp, xb, P, H, L, args.renderer, args.mask, cols, args.stat_rule)
    assert torch.equal(a, b), "forecast is not deterministic"
    print("[ok] deterministic; hidden block is blank in the model input by construction", flush=True)

    preds = {"pretrained": [], "zeroed": [], "random": []}
    for i in range(0, len(Xu), args.batch):
        xb = torch.from_numpy(Xu[i:i + args.batch]).to(DEV)[:, :, 0]
        preds["pretrained"].append(forecast(mp, xb, P, H, L, args.renderer, args.mask, cols, args.stat_rule).cpu().numpy())
        preds["zeroed"].append(forecast(mp, xb, P, H, L, args.renderer, args.mask, cols, args.stat_rule, True).cpu().numpy())
        preds["random"].append(forecast(mr, xb, P, H, L, args.renderer, args.mask, cols, args.stat_rule).cpu().numpy())
    for k in preds: preds[k] = np.concatenate(preds[k])

    del mp, mr; torch.cuda.empty_cache()
    from visionts import VisionTS
    v = VisionTS(arch="mae_base", finetune_type="none", ckpt_dir=args.vts_ckpt_dir).to(DEV).eval()
    v.update_config(context_len=L, pred_len=H, periodicity=P, norm_const=0.4,
                    align_const=0.4, interpolation="bilinear")
    vp = []
    with torch.no_grad():
        for i in range(0, len(Xu), args.batch):
            xb = torch.from_numpy(Xu[i:i + args.batch, -L:]).to(DEV)
            vp.append(v(xb).cpu().numpy())
    preds["visionts"] = np.concatenate(vp)

    reps = (H + P - 1) // P; G = L // P
    ctxn = Xu[:, -L:]
    preds["smean"] = np.tile(ctxn.reshape(len(Xu), G, P, 1).mean(1), (1, reps, 1))[:, :H]
    preds["snaive"] = np.tile(ctxn[:, -P:], (1, reps, 1))[:, :H]

    met = {k: round(float(np.mean((p - Yu) ** 2)), 6) for k, p in preds.items()}
    met.update({k + "_mae": round(float(np.mean(np.abs(p - Yu))), 6) for k, p in preds.items()})
    per_win = {}
    for w in np.unique(pairs[:, 0]):
        sel = pairs[:, 0] == w
        if sel.sum():
            per_win[int(w)] = float(np.mean((preds["pretrained"][sel] - Yu[sel]) ** 2))
    print(json.dumps({k: met[k] for k in ("pretrained", "zeroed", "random", "visionts", "smean", "snaive")}, indent=2), flush=True)

    import transformers
    commit, dirty = git_info()
    out = {"status": "complete", "stage": "D", "regime": "ZS", "dataset": args.dataset,
           "renderer": args.renderer, "mask": args.mask, "masked_cols": cols,
           "mask_ratio": round(VP.MASK_SPATIAL_COUNT[args.mask] / 196, 4),
           "stat_rule": args.stat_rule, "checkpoint": args.ckpt, "norm_pix_loss": True,
           "git_commit": commit, "git_dirty": dirty,
           "torch": torch.__version__, "transformers": transformers.__version__,
           "manifest": man, "n_eval": int(len(Xu)), "seed": args.seed,
           "metrics": met, "per_window_mse": per_win,
           "wall_clock_s": round(time.time() - t0, 1),
           "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEV == "cuda" else None}
    d = os.path.join(args.root, "strict_forecast"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"{args.dataset}_{args.renderer}_{args.mask}_s{args.seed}.json"), "w"), indent=2)
    print("[done]", args.dataset, flush=True)


if __name__ == "__main__":
    main()
