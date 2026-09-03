"""DEV -- select prior family + combiner on HISTORICAL audit windows (benchmark untouched)."""
import argparse, json, math, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zeroshot_loop as ZL

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="pilot/data")
ap.add_argument("--datasets", nargs="+",
                default=["ETTh1", "ETTh2", "ETTm2", "electricity", "traffic", "solar"])
a = ap.parse_args()

FAM = {"base4": ["smean", "snaive", "last", "recent_mean"],
       "seasonal": ["smean", "smean2", "smean4", "smean8", "snaive", "smean_drift",
                    "smean4_drift", "recent_mean", "last"]}
res, ind = {}, {}
for d in a.datasets:
    X, pairs, man = ZL.historical_manifest(d, a.data_dir, 112, stride=8, n_max=2000,
                                           seed=0, split="audit")
    P, L, H = man["P"], man["L"], man["H"]
    ctx, fut = X[:, :L], X[:, L:L + H]
    allp = ZL.priors(ctx, P, H)
    ind[d] = {k: float(np.mean((v - fut) ** 2)) for k, v in allp.items()}
    ref = ind[d]["smean"]
    for fam, keys in FAM.items():
        cand = {k: allp[k] for k in keys if k in allp}
        _, sel, _ = ZL.pseudo_origin_select(ctx, P, H, cand)
        res.setdefault(f"{fam}/argmin", {})[d] = float(np.mean((sel - fut) ** 2)) / ref
        for rule in ("softmin", "invloss", "uniform"):
            _, bl, _, _ = ZL.pseudo_origin_blend(ctx, P, H, cand, rule=rule)
            res.setdefault(f"{fam}/{rule}", {})[d] = float(np.mean((bl - fut) ** 2)) / ref
        for rule in ("softmin", "invloss", "softmin_n", "pow_n"):
            for mc in (4, 8):
                _, bl, _, u = ZL.pseudo_origin_blend(ctx, P, H, cand, rule=rule, dense=True,
                                                     min_ctx_periods=mc)
                res.setdefault(f"{fam}/{rule}-dense{mc}", {})[d] = float(np.mean((bl-fut)**2))/ref
                res.setdefault("_n_origins", {})[f"{d}/mc{mc}"] = u
    res.setdefault("fixed/smean", {})[d] = 1.0

print("=== individual priors, MSE relative to smean (historical) ===")
ks = sorted({k for v in ind.values() for k in v})
print(f"{'dataset':12s} " + " ".join(f"{k[:11]:>11s}" for k in ks))
for d in a.datasets:
    print(f"{d:12s} " + " ".join(f"{ind[d][k]/ind[d]['smean']:11.4f}" for k in ks))

print("\n=== combiners, MSE relative to smean; last column = geometric mean ===")
print(f"{'combiner':22s} " + " ".join(f"{d[:11]:>11s}" for d in a.datasets) + f"{'GEO':>10s}")
rank = []
for k, v in res.items():
    if k.startswith('_'):
        continue
    g = math.exp(sum(math.log(v[d]) for d in a.datasets) / len(a.datasets))
    rank.append((g, k, v))
for g, k, v in sorted(rank):
    print(f"{k:22s} " + " ".join(f"{v[d]:11.4f}" for d in a.datasets) + f"{g:10.4f}")
json.dump({"individual": ind, "combiners": {k: v for _, k, v in rank}},
          open(os.path.join(os.path.dirname(a.data_dir.rstrip('/')), "dev_combiner.json")
               if False else "dev_combiner_result.json", "w"), indent=2)
