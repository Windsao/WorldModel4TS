"""Stages M0/M1: matched-mask historical factorization (spec sections 3-4).

Geometry is FIXED at 4 context / 10 future columns for every mask. Each batch is rendered
ONCE and the identical `video_input` / `video_full` tensors are reused for every mask, with
hash assertions -- this is what the previous audit failed to do.

Region membership always comes from the canonical patch map, never from the mask name.
Subset metrics are accumulated as elementwise sums and counts so batches recombine exactly.
"""

import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import videomae_patch_utils as VP
import video_visionts_renderers as VR
import video_visionts_mask_factorization as MF
from run_video_visionts_recovery_audit import (make_historical_pair, build_partitions,
                                               gather_hist, HORIZON)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NF, PATCH = 16, 16
COLS = MF.FUT_COLS                       # fixed 10 future columns -- never derived from the mask


def mask_plan():
    """(config_name, spatial_mask, seed) for the whole factorization."""
    plan = []
    for name in ("right_future_100", "future_random_140",
                 "future_block_2", "future_block_5", "future_block_8", "future_block_10"):
        plan.append((name, MF.make_matched_mask(name, 0), 0))
    for name in sorted(MF.RANDOM_MASKS):
        for s in (0, 1, 2):
            plan.append((name, MF.make_matched_mask(name, s), s))
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ETTh2", "ETTm2"])
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default="pilot/results_field/video_visionts_matched_mask")
    ap.add_argument("--ckpt", default=os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base"))
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    data, P, L, H, parts, man = build_partitions(args.dataset, args.data_dir, args.max_ch,
                                                 n_audit=args.n, seed=args.seed)
    X = gather_hist(data, parts["audit"], L, H)
    print(f"{args.dataset} P={P} L={L} H={H} audit_n={len(X)} geometry=4ctx/10fut", flush=True)

    from transformers import VideoMAEForPreTraining, VideoMAEConfig
    mp = VideoMAEForPreTraining.from_pretrained(args.ckpt).to(DEV).eval()
    torch.manual_seed(args.seed)
    mr = VideoMAEForPreTraining(VideoMAEConfig.from_pretrained(args.ckpt)).to(DEV).eval()

    plan = mask_plan()
    acc = {(n, s): {"pt": {}, "rand": {}, "raw": {}} for n, _, s in plan}
    cohort_acc = {}                                  # M1.5 right-100 cohort scoring
    seen_hash, std_hist = None, []

    for i in range(0, len(X), args.batch):
        xb = torch.from_numpy(X[i:i + args.batch]).to(DEV)
        B = xb.shape[0]
        # ---- render ONCE, fixed geometry
        vi, vf, mu, sd = make_historical_pair(xb, P, H, L, COLS)
        h_in, h_full = MF.tensor_hash(vi), MF.tensor_hash(vf)
        if seen_hash is None:
            seen_hash = (h_in, h_full)
        tgt_all, cmean, cstd = VP.native_targets(vf)          # mask-independent
        cubes_true = VP.patchify(VP.unnormalize(vf))
        cubes_in = VP.patchify(VP.unnormalize(vi))
        C = cubes_true.shape[-1]
        std_hist.append(cstd.reshape(-1).detach().cpu().numpy()[::37])

        right_logits, right_bm = None, None
        for name, sp, s in plan:
            bm = MF.to_tube(sp, B, DEV)
            assert (MF.tensor_hash(vi), MF.tensor_hash(vf)) == (h_in, h_full), "render changed across masks"
            labels = tgt_all[bm].view(B, -1, 1536)
            is_fut, col = MF.token_region_labels(sp, DEV)
            std_tok = cstd[bm].view(B, -1, C).mean(-1)[0] > 1e-5     # per-token non-degenerate
            with torch.no_grad():
                lg_p = mp(pixel_values=vi, bool_masked_pos=bm).logits
                lg_r = mr(pixel_values=vi, bool_masked_pos=bm).logits
            if name == "right_future_100":
                right_logits, right_bm, right_sp = lg_p, bm, sp
            for tag, lg in (("pt", lg_p), ("rand", lg_r)):
                MF.merge_sums(acc[(name, s)][tag],
                              MF.subset_sums((lg - labels) ** 2, labels ** 2, is_fut, col, std_tok))
            # raw-pixel oracle reconstruction for the same subsets
            sm, ss = cmean[bm].view(B, -1, 1, C), cstd[bm].view(B, -1, 1, C)
            raw_true = cubes_true[bm].view(B, -1, 512, C)
            raw_p = VP.denormalize_cubes(lg_p, sm, ss)
            raw_z = VP.denormalize_cubes(torch.zeros_like(lg_p), sm, ss)
            MF.merge_sums(acc[(name, s)]["raw"],
                          MF.subset_sums(((raw_p - raw_true) ** 2).flatten(2),
                                         ((raw_z - raw_true) ** 2).flatten(2), is_fut, col, std_tok))
        # ---- M1.5: score the SINGLE right-100 forward on other cohorts
        if right_logits is not None:
            r_fut, r_col = MF.token_region_labels(right_sp, DEV)
            r_lab = tgt_all[right_bm].view(B, -1, 1536)
            r_std = cstd[right_bm].view(B, -1, C).mean(-1)[0] > 1e-5
            fut_idx = np.where(MF.canonical_is_future())[0]
            for cname in ("future_random_35", "future_random_70", "future_random_105",
                          "future_block_2", "future_block_5", "future_block_8"):
                csp = MF.make_matched_mask(cname, 0)
                sel_np = np.isin(fut_idx, np.where(csp)[0])
                sel = torch.as_tensor(np.tile(sel_np, 8), device=DEV)
                key = f"right100_on_{cname}"
                cohort_acc.setdefault(key, {})
                MF.merge_sums(cohort_acc[key],
                              MF.subset_sums(((right_logits - r_lab) ** 2)[:, sel],
                                             (r_lab ** 2)[:, sel], r_fut[sel], r_col[sel], r_std[sel]))

    res = {}
    for (name, s), d in acc.items():
        res[f"{name}|seed{s}"] = {"native_pt": MF.finalize(d["pt"]),
                                  "native_rand": MF.finalize(d["rand"]),
                                  "raw_oracle": MF.finalize(d["raw"]),
                                  "mask_hash": MF.mask_hash(MF.make_matched_mask(name, s)),
                                  "masked_context": int((MF.make_matched_mask(name, s) & ~MF.canonical_is_future()).sum()),
                                  "masked_future": int((MF.make_matched_mask(name, s) & MF.canonical_is_future()).sum())}
    cohorts = {k: MF.finalize(v) for k, v in cohort_acc.items()}
    allstd = np.concatenate(std_hist)
    tgt_dist = {"frac_std_le_1e-5": float((allstd <= 1e-5).mean()),
                "std_p05": float(np.percentile(allstd, 5)),
                "std_median": float(np.median(allstd)),
                "std_p95": float(np.percentile(allstd, 95))}

    # bookkeeping check: context + future sums must reproduce the all-masked sum
    chk = []
    for k, v in res.items():
        p = v["native_pt"]
        if "context" in p and "future" in p and "all" in p:
            lhs = p["context"]["mse_pt"] * p["context"]["n"] + p["future"]["mse_pt"] * p["future"]["n"]
            rhs = p["all"]["mse_pt"] * p["all"]["n"]
            chk.append(abs(lhs - rhs) / max(abs(rhs), 1e-9))
    print(f"[check] max relative subset-recombination error = {max(chk) if chk else 0:.2e}", flush=True)
    for k in ("right_future_100|seed0", "global_random_147|seed0", "context_random_42|seed0"):
        if k in res:
            p = res[k]["native_pt"]
            print(f"  {k:28s} all={p.get('all',{}).get('native_ratio_zero',float('nan')):.4f} "
                  f"ctx={p.get('context',{}).get('native_ratio_zero',float('nan')):.4f} "
                  f"fut={p.get('future',{}).get('native_ratio_zero',float('nan')):.4f}", flush=True)

    import transformers
    out = {"status": "complete", "stage": "M1", "regime": "HA", "dataset": args.dataset,
           "geometry": {"renderer": "dense_static", "context_cols": MF.CTX_COLS,
                        "future_cols": MF.FUT_COLS, "grid": MF.GRID},
           "video_input_hash": seen_hash[0], "video_full_hash": seen_hash[1],
           "checkpoint": args.ckpt, "P": P, "context": L, "horizon": H,
           "n_eval": int(len(X)), "batch": args.batch, "seed": args.seed,
           "manifest": man, "target_dist": tgt_dist,
           "subset_recombination_max_rel_err": float(max(chk) if chk else 0),
           "masks": res, "cohort_scoring": cohorts,
           "git_commit": os.environ.get("WM4TS_GIT_COMMIT", "unknown"),
           "torch": torch.__version__, "transformers": transformers.__version__,
           "wall_clock_s": round(time.time() - t0, 1),
           "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEV == "cuda" else None}
    d = os.path.join(args.root, "matched_audit"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"M1_{args.dataset}_s{args.seed}.json"), "w"), indent=2)
    print("[done]", args.dataset, flush=True)


if __name__ == "__main__":
    main()
