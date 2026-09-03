"""Stage M5: carrier historical audit (spec section 8.3).

Triggered by Outcome A. Only carriers that pass the section-8.2 numerical checks may run;
C1 (soft edge position) failed its decode-accuracy check (0.0511 > 0.02) and is excluded
before any model forward, which is what that gate exists for.

Decoding reads the NORMALIZED logits directly, so no unknown raw cube mean/std is ever
estimated -- this is the whole point of a patch-normalization-invariant carrier.
"""

import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import videomae_patch_utils as VP
import video_visionts_renderers as VR
import video_visionts_mask_factorization as MF
import video_visionts_carriers as CA
import video_visionts_layouts as VL
from run_video_visionts_recovery_audit import (make_historical_pair, build_partitions,
                                               gather_hist, HORIZON)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NF, PATCH, GRID = 16, 16, 14


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ETTh2", "ETTm2"])
    ap.add_argument("--carrier", default="C2", choices=["C1", "C2"])
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
    print(f"{args.dataset} carrier={args.carrier} n={len(X)} P={P} L={L} H={H}", flush=True)

    from transformers import VideoMAEForPreTraining, VideoMAEConfig
    mp = VideoMAEForPreTraining.from_pretrained(args.ckpt).to(DEV).eval()
    torch.manual_seed(args.seed)
    mr = VideoMAEForPreTraining(VideoMAEConfig.from_pretrained(args.ckpt)).to(DEV).eval()

    sp = MF.make_matched_mask("right_future_100", 0)
    fut_mask = MF.canonical_is_future()
    acc = {k: {"err": 0.0, "tgt": 0.0, "n": 0} for k in ("pt", "rand")}
    ts = {k: 0.0 for k in ("pt", "rand", "zero", "i0")}
    cnt = 0
    for i in range(0, len(X), args.batch):
        xb = torch.from_numpy(X[i:i + args.batch]).to(DEV)
        B = xb.shape[0]
        ctx, fut = xb[:, :L], xb[:, L:L + H]
        vi, vf, zgrid, mu, sd = CA.render_carrier(ctx, fut, P, args.carrier)
        bm = MF.to_tube(sp, B, DEV)
        tgt_all, cmean, cstd = VP.native_targets(vf)
        labels = tgt_all[bm].view(B, -1, 1536)
        with torch.no_grad():
            lp = mp(pixel_values=vi, bool_masked_pos=bm).logits
            lr = mr(pixel_values=vi, bool_masked_pos=bm).logits
        for tag, lg in (("pt", lp), ("rand", lr)):
            acc[tag]["err"] += float(((lg - labels).double() ** 2).sum())
            acc[tag]["tgt"] += float((labels.double() ** 2).sum())
            acc[tag]["n"] += labels.numel()
        # decode straight from normalized logits -> future scalar grid -> series
        yt = fut.unsqueeze(-1)
        for tag, lg in (("pt", lp), ("rand", lr), ("zero", torch.zeros_like(lp))):
            zf = CA.decode_future(lg, args.carrier)                       # [B, n_masked]
            zf = zf.view(B, 8, -1)[:, 0].view(B, GRID, -1)                # first tubelet, [B,14,10]
            yh = CA.z_to_series(zf, P, H, mu, sd)
            ts[tag] += float(((yh - yt) ** 2).sum())
        # I0 reference: incumbent dense_static + nearest-visible-physical on the same windows
        vi0, vf0, mu0, sd0 = make_historical_pair(xb, P, H, L, MF.FUT_COLS)
        bm0 = VL.tube_mask_from_spatial(sp, B, DEV)
        with torch.no_grad():
            l0 = mp(pixel_values=vi0, bool_masked_pos=bm0).logits
            cin = VP.patchify(VP.unnormalize(vi0))
            m_, s_ = VL.nearest_visible_physical(cin, bm0)
            filled = cin.clone()
            filled[bm0] = VP.denormalize_cubes(l0, m_, s_).view(-1, 512, cin.shape[-1])
            img = VP.unpatchify(filled)[:, NF - 1].mean(1).clamp(0, 1) * 2 - 1
            y0 = VR.invert_dense(img[:, :, MF.CTX_COLS * PATCH:], P, H, mu0, sd0)
        ts["i0"] += float(((y0 - yt) ** 2).sum())
        cnt += B * H
    res = {}
    for tag in ("pt", "rand"):
        res[f"native_{tag}"] = {"mse": acc[tag]["err"] / acc[tag]["n"],
                                "mse_zero": acc[tag]["tgt"] / acc[tag]["n"],
                                "ratio_zero": acc[tag]["err"] / acc[tag]["tgt"]}
    res["native_pt_over_rand"] = acc["pt"]["err"] / acc["rand"]["err"]
    for tag in ("pt", "rand", "zero", "i0"):
        res[f"ts_{tag}_mse"] = ts[tag] / cnt
    res["ts_pt_over_zero"] = ts["pt"] / max(ts["zero"], 1e-12)
    res["ts_pt_over_i0"] = ts["pt"] / max(ts["i0"], 1e-12)
    res["ts_pt_over_rand"] = ts["pt"] / max(ts["rand"], 1e-12)
    print(json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in res.items()}, indent=2), flush=True)

    import transformers
    out = {"status": "complete", "stage": "M5", "regime": "HA", "dataset": args.dataset,
           "carrier": args.carrier, "mask": "right_future_100",
           "checkpoint": args.ckpt, "P": P, "context": L, "horizon": H,
           "n_eval": int(len(X)), "seed": args.seed, "manifest": man, "metrics": res,
           "git_commit": os.environ.get("WM4TS_GIT_COMMIT", "unknown"),
           "torch": torch.__version__, "transformers": transformers.__version__,
           "wall_clock_s": round(time.time() - t0, 1)}
    d = os.path.join(args.root, "carrier_historical"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"M5_{args.dataset}_{args.carrier}_s{args.seed}.json"), "w"), indent=2)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
