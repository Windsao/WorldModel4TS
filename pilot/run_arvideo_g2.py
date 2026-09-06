"""Route A gate G2: historical audit of MAGI-1 video continuation as a forecaster on real series.

Datasets ETTh2 and ETTm2, HISTORICAL audit windows only (ZL.historical_manifest split='audit';
every window lies inside the train range, no benchmark future is read). For each origin:
  context  = the last 8 periods of the L-step context window, normalised by their own mean/std
  prefix   = 32-frame smooth-silhouette video of those 8 periods (+ clock strip)
  MAGI v2v = 48 generated frames -> periods 8.. decoded at absolute frames 4k+3 (offset from clock)
  forecast = first H decoded values, de-normalised
Scored against: snaive, smean (all L), copy-of-last-period (== snaive), ZL-051 blend on the full
context, and the model-vs-copy diagnostic. Paired moving-block bootstrap vs snaive and vs blend.
Gate (plan section 3): TS MSE / snaive <= 0.95 and / blend <= 1.0 on both datasets.
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arvideo_render as R
import run_arvideo_g1 as G1
import zeroshot_loop as ZL

PRIOR_FAMILY = ["smean", "snaive", "last", "recent_mean"]


def one_origin(args, ds, i, ctx_i, P, H, gpu, root):
    d = os.path.join(root, f"{ds}_o{i:04d}")
    os.makedirs(d, exist_ok=True)
    res_path = os.path.join(d, "result.json")
    if os.path.exists(res_path) and not args.rescore:
        return json.load(open(res_path))
    G_ctx = R.PREFIX_FRAMES // R.M
    tail = ctx_i[-G_ctx * P:]
    mu, sd = float(tail.mean()), float(tail.std() + 1e-6)
    Z_ctx = ((tail - mu) / sd).reshape(G_ctx, P)
    np.save(os.path.join(d, "Z_ctx.npy"), Z_ctx)
    prefix_mp4 = os.path.join(d, "prefix.mp4")
    out_mp4 = os.path.join(d, "out.mp4")
    if not (args.rescore and os.path.exists(out_mp4)):
        G1.write_mp4(R.render_prefix(Z_ctx), prefix_mp4)
        cfg = os.path.join(d, "cfg.json")
        G1.make_cfg(args.cfg, args.seed, args.num_frames, cfg)
        rc, secs = G1.run_magi(gpu, cfg, prefix_mp4, out_mp4, os.path.join(d, "magi.log"))
    else:
        rc, secs = 0, None
    res = {"dataset": ds, "origin": i, "rc": rc, "secs": secs, "mu": mu, "sd": sd}
    if rc != 0 or not os.path.exists(out_mp4):
        res["error"] = "magi failed"; json.dump(res, open(res_path, "w")); return res
    out = G1.resize_frames(G1.read_mp4(out_mp4), R.S)
    off_c, rate, drift, frac, ts = G1.align_by_clock(out)
    off = off_c if off_c is not None else R.PREFIX_FRAMES
    dec = R.decode_video(out, P)
    np.save(os.path.join(d, "decoded_all.npy"), dec)
    n_fut = int(np.ceil(H / P))
    zs = []
    for k in range(G_ctx, G_ctx + n_fut):
        j = R.period_frame(k) - off
        if j >= out.shape[0]:
            break
        zs.append(dec[j])
    if len(zs) < n_fut:
        res["error"] = "not enough generated frames"; json.dump(res, open(res_path, "w")); return res
    z = np.concatenate(zs)[:H]
    res.update({"offset": int(off), "clock_rate": rate, "clock_final_drift": drift, "clock_readable_frac": frac,
                "forecast": (z * sd + mu).tolist()})
    json.dump(res, open(res_path, "w"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/nyx-storage1/hanliu/wm4ts/arvideo/g2")
    ap.add_argument("--cfg", default=os.path.join(HERE, "magi_4p5B_480_distill.json"))
    ap.add_argument("--data-dir", default="/nyx-storage1/hanliu/wm4ts/data")
    ap.add_argument("--datasets", default="ETTh2,ETTm2")
    ap.add_argument("--n", type=int, default=48, help="origins per dataset (distinct windows)")
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--num-frames", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",")]
    summary = {}
    for ds in args.datasets.split(","):
        X, pairs, man = ZL.historical_manifest(ds, args.data_dir, args.max_ch, stride=args.stride,
                                               n_max=4000, seed=args.seed, split="audit")
        P, L, H = man["P"], man["L"], man["H"]
        # one origin per distinct window (channel chosen by the manifest order), spread over time
        uniq_w, first_idx = np.unique(pairs[:, 0], return_index=True)
        sel = first_idx[np.linspace(0, len(first_idx) - 1, min(args.n, len(first_idx))).round().astype(int)]
        ctx, fut = X[sel, :L], X[sel, L:L + H]
        print(f"[G2] {ds} n={len(sel)} P={P} L={L} H={H} windows={len(uniq_w)}", flush=True)
        root = os.path.join(args.out, ds)
        q = Queue()
        for i in range(len(sel)): q.put(i)
        results = [None] * len(sel)
        def worker(gpu):
            while True:
                try: i = q.get_nowait()
                except Exception: return
                results[i] = one_origin(args, ds, i, ctx[i], P, H, gpu, root)
                r = results[i]
                print(json.dumps({k: r.get(k) for k in ("dataset", "origin", "rc", "secs", "offset", "clock_rate", "error")}), flush=True)
        with ThreadPoolExecutor(len(gpus)) as ex:
            list(ex.map(worker, gpus))
        ok = [i for i, r in enumerate(results) if r and "forecast" in r]
        Y = np.array([results[i]["forecast"] for i in ok]); F = fut[ok]; C = ctx[ok]
        allp = ZL.priors(C, P, H)
        pr = {k: allp[k] for k in PRIOR_FAMILY}
        _, blend, _, _ = ZL.pseudo_origin_blend(C, P, H, pr, rule="pow_n", dense=True, min_ctx_periods=8)
        err = {"magi": ((Y - F) ** 2).mean(1), "snaive": ((allp["snaive"] - F) ** 2).mean(1),
               "smean": ((allp["smean"] - F) ** 2).mean(1), "blend": ((blend - F) ** 2).mean(1)}
        blk = int(np.ceil((L + H) / args.stride))
        origins = pairs[sel][ok, 0]
        s = {"n": len(ok), "P": P, "H": H, "mse": {k: float(v.mean()) for k, v in err.items()},
             "ratio_vs_snaive": float(err["magi"].mean() / err["snaive"].mean()),
             "ratio_vs_blend": float(err["magi"].mean() / err["blend"].mean()),
             "boot_vs_snaive": ZL.paired_bootstrap(err["magi"], err["snaive"], origins, blk),
             "boot_vs_blend": ZL.paired_bootstrap(err["magi"], err["blend"], origins, blk),
             "magi_vs_copy_mse": float(((Y - allp["snaive"]) ** 2).mean()),
             "clock_rate_mean": float(np.nanmean([results[i].get("clock_rate", np.nan) for i in ok])),
             "offsets": sorted(set(int(results[i]["offset"]) for i in ok))}
        s["gate_snaive<=0.95"] = s["ratio_vs_snaive"] <= 0.95
        s["gate_blend<=1.0"] = s["ratio_vs_blend"] <= 1.0
        summary[ds] = s
        np.savez_compressed(os.path.join(root, "per_origin.npz"), Y=Y, F=F, origins=origins, **err)
        print(json.dumps({ds: {k: v for k, v in s.items() if k not in ("boot_vs_snaive", "boot_vs_blend")}}, indent=1), flush=True)
        json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1, default=float)
    summary["G2_pass"] = all(summary[d]["gate_snaive<=0.95"] and summary[d]["gate_blend<=1.0"] for d in summary)
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1, default=float)
    print("G2_pass", summary["G2_pass"])


if __name__ == "__main__":
    main()
