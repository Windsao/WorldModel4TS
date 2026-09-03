"""Stage R5: frozen genuine-future evaluation (spec section 8).

Primary configuration frozen from the historical audit BEFORE any genuine future is read:
  renderer   dense_static (unchanged)
  width rule pre-registered VisionTS allocation -> right_10 / right_9 per dataset shape
  layout     G0  (G1 and G2 both failed the R3 promotion gate)
  stat rule  nearest_visible_physical
  lens       L0, i.e. none (L1/L2/L3 all failed the R4 historical gate)

Controls on identical manifests: zero logits, random-init VideoMAE, existing Stage-D
right_10 result, official VisionTS, smean, snaive. Per-origin squared errors are saved for
EVERY method so the paired bootstrap can be computed for all pairs.
"""

import argparse, hashlib, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "third_party", "VisionTS"))
import run_field as RF
import videomae_patch_utils as VP
import video_visionts_renderers as VR
import video_visionts_layouts as VL
from run_visionts_reference import HORIZON, build_manifest, gather, git_info

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID, PATCH, NF = 14, 16, 16


def method_forecast(model, xb, P, L, H, vc, canon_of_phys, phys_mask, zero_logits=False):
    """Strict ZS: the hidden block is blank in the encoder input by construction."""
    B = xb.shape[0]
    cols = GRID - vc
    blank = torch.zeros(B, H, device=xb.device, dtype=xb.dtype)
    vi, _, mu, sd = VR.dense_static(xb[:, :L], blank, P, cols)
    g = VR.video_to_gray(vi)[:, 0]
    if canon_of_phys is not None:
        g = VL.permute_image(g, canon_of_phys)
    im = ((g + 1) / 2).clamp(0, 1).unsqueeze(1).unsqueeze(1).expand(B, NF, 3, 224, 224)
    vip = ((im - VR.IMN_MEAN.to(g.device).unsqueeze(1))
           / VR.IMN_STD.to(g.device).unsqueeze(1)).contiguous()
    bm = VL.tube_mask_from_spatial(phys_mask, B, DEV)
    with torch.no_grad():
        logits = model(pixel_values=vip, bool_masked_pos=bm).logits
        if zero_logits:
            logits = torch.zeros_like(logits)
        cubes_in = VP.patchify(VP.unnormalize(vip))
        m_, s_ = VL.nearest_visible_physical(cubes_in, bm)
        filled = cubes_in.clone()
        filled[bm] = VP.denormalize_cubes(logits, m_, s_).view(-1, 512, cubes_in.shape[-1])
        img = VP.unpatchify(filled)[:, NF - 1].mean(1).clamp(0, 1) * 2 - 1
        if canon_of_phys is not None:
            img = VL.unpermute_image(img, canon_of_phys)
    return VR.invert_dense(img[:, :, vc * PATCH:], P, H, mu, sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HORIZON))
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default="pilot/results_field/video_visionts_recovery")
    ap.add_argument("--old-root", default="pilot/results_field/video_visionts")
    ap.add_argument("--ckpt", default=os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base"))
    ap.add_argument("--vts-ckpt-dir", default="./ckpt/")
    ap.add_argument("--layout", default="G0")
    ap.add_argument("--layout-seed", type=int, default=0)
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
    vc, mc = VL.width_rule(L, H)
    cp, _, pmask = VL.build_layout(args.layout, vc, args.layout_seed)
    cp = None if args.layout == "G0" else cp
    print(f"{args.dataset} layout={args.layout} vc={vc} mc={mc} n={len(Xu)}", flush=True)

    from transformers import VideoMAEForPreTraining, VideoMAEConfig
    mp = VideoMAEForPreTraining.from_pretrained(args.ckpt).to(DEV).eval()
    torch.manual_seed(args.seed)
    mr = VideoMAEForPreTraining(VideoMAEConfig.from_pretrained(args.ckpt)).to(DEV).eval()

    preds = {"method": [], "zeroed": [], "random": []}
    for i in range(0, len(Xu), args.batch):
        xb = torch.from_numpy(Xu[i:i + args.batch]).to(DEV)[:, :, 0]
        preds["method"].append(method_forecast(mp, xb, P, L, H, vc, cp, pmask).cpu().numpy())
        preds["zeroed"].append(method_forecast(mp, xb, P, L, H, vc, cp, pmask, True).cpu().numpy())
        preds["random"].append(method_forecast(mr, xb, P, L, H, vc, cp, pmask).cpu().numpy())
    for k in preds: preds[k] = np.concatenate(preds[k])
    del mp, mr; torch.cuda.empty_cache()

    from visionts import VisionTS
    v = VisionTS(arch="mae_base", finetune_type="none", ckpt_dir=args.vts_ckpt_dir).to(DEV).eval()
    v.update_config(context_len=L, pred_len=H, periodicity=P, norm_const=0.4,
                    align_const=0.4, interpolation="bilinear")
    vp = []
    with torch.no_grad():
        for i in range(0, len(Xu), args.batch):
            vp.append(v(torch.from_numpy(Xu[i:i + args.batch]).to(DEV)).cpu().numpy())
    preds["visionts"] = np.concatenate(vp)

    reps = (H + P - 1) // P; G = L // P
    preds["smean"] = np.tile(Xu.reshape(len(Xu), G, P, 1).mean(1), (1, reps, 1))[:, :H]
    preds["snaive"] = np.tile(Xu[:, -P:], (1, reps, 1))[:, :H]

    met = {k: round(float(np.mean((p - Yu) ** 2)), 6) for k, p in preds.items()}
    met.update({k + "_mae": round(float(np.mean(np.abs(p - Yu))), 6) for k, p in preds.items()})
    old = os.path.join(args.old_root, "strict_forecast",
                       f"{args.dataset}_dense_static_right_10_s0.json")
    if os.path.exists(old):
        met["stage_d"] = json.load(open(old))["metrics"]["pretrained"]
    # per-origin squared error for EVERY method (spec section 8)
    per_origin = {k: {} for k in preds}
    for w in np.unique(pairs[:, 0]):
        sel = pairs[:, 0] == w
        for k, p in preds.items():
            per_origin[k][int(w)] = float(np.mean((p[sel] - Yu[sel]) ** 2))
    print(json.dumps({k: met[k] for k in ("method", "zeroed", "random", "visionts", "smean", "snaive")}, indent=2), flush=True)

    import transformers
    commit, dirty = git_info()
    out = {"status": "complete", "stage": "R5", "regime": "ZS", "dataset": args.dataset,
           "renderer": "dense_static", "layout": args.layout, "layout_seed": args.layout_seed,
           "variant": "L0", "stat_rule": "nearest_visible_physical", "backbone": "pretrained",
           "width_rule": "visionts_align0.4", "visible_cols": vc, "masked_cols": mc,
           "mask_hash": hashlib.sha1(pmask.tobytes()).hexdigest()[:16],
           "checkpoint": args.ckpt, "git_commit": commit, "git_dirty": dirty,
           "torch": torch.__version__, "transformers": transformers.__version__,
           "manifest": man, "n_eval": int(len(Xu)), "seed": args.seed,
           "block_len": int(math.ceil(H / args.stride)),
           "metrics": met, "per_origin_mse": per_origin,
           "wall_clock_s": round(time.time() - t0, 1)}
    d = os.path.join(args.root, "strict_forecast"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"{args.dataset}_{args.layout}_L0_s{args.seed}.json"), "w"), indent=2)
    print("[done]", args.dataset, flush=True)


if __name__ == "__main__":
    main()
