"""Aggregate Route B evaluation JSONs (eval_route_b.py) into the tables of the results report.

usage: python pilot/summarize_route_b.py --dir <eval root> --step 20000 [--stage F]
Prints per-dataset MSE for every method, equal-dataset geometric-mean ratios (Q) against the
blend and VisionTS, condition (i)/(ii) verdicts with paired bootstrap CIs, and writes
<dir>/summary_<stage>_<step>.json.
"""
import argparse, glob, json, os
import numpy as np

DS = ["ETTh1", "ETTh2", "ETTm2", "electricity", "traffic", "solar"]
ARMS = ["vmae_full", "vmae_enc", "imae_enc", "random", "imae_full", "imae_enc_d8", "vmae_enc_d8", "random_d8"]


def geo(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x) and x > 0]
    return float(np.exp(np.mean(np.log(xs)))) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/nyx-storage1/hanliu/wm4ts/route_b/eval")
    ap.add_argument("--step", type=int, default=20000)
    ap.add_argument("--stage", default="F")
    args = ap.parse_args()
    R = {}
    for ds in DS:
        p = os.path.join(args.dir, f"stage{args.stage}", f"{ds}_step{args.step}.json")
        if os.path.exists(p):
            R[ds] = json.load(open(p))["results"]
    if not R:
        print("no results"); return
    methods = ["smean", "snaive", "blend"] + (["visionts"] if any("visionts_mse" in r for r in R.values()) else [])
    present = [a for a in ARMS if any(f"video_{a}_mse" in r for r in R.values())]
    methods += [f"video_{a}" for a in present] + [f"blend+video_{a}" for a in present]
    print(f"\n### MSE per dataset (stage {args.stage}, step {args.step})\n")
    print("| method | " + " | ".join(R) + " |")
    print("|---|" + "---:|" * len(R))
    for m in methods:
        row = [R[ds].get(f"{m}_mse") for ds in R]
        if all(v is None for v in row): continue
        print(f"| {m} | " + " | ".join("—" if v is None else f"{v:.4f}" for v in row) + " |")
    print("\n### Equal-dataset geometric means\n")
    print("| method | Q vs blend | Q vs VisionTS | Q vs smean | Q vs video_random |")
    print("|---|---:|---:|---:|---:|")
    summ = {"per_dataset": R, "Q": {}}
    for m in methods:
        if m == "blend": continue
        qb = geo([R[ds][f"{m}_mse"] / R[ds]["blend_mse"] for ds in R if f"{m}_mse" in R[ds]])
        qv = geo([R[ds][f"{m}_mse"] / R[ds]["visionts_mse"] for ds in R if f"{m}_mse" in R[ds] and "visionts_mse" in R[ds]])
        qs = geo([R[ds][f"{m}_mse"] / R[ds]["smean_mse"] for ds in R if f"{m}_mse" in R[ds]])
        qr = geo([R[ds][f"{m}_mse"] / R[ds]["video_random_mse"] for ds in R if f"{m}_mse" in R[ds] and "video_random_mse" in R[ds]])
        summ["Q"][m] = {"vs_blend": qb, "vs_visionts": qv, "vs_smean": qs, "vs_video_random": qr}
        f = lambda v: "—" if v is None else f"{v:.4f}"
        print(f"| {m} | {f(qb)} | {f(qv)} | {f(qs)} | {f(qr)} |")
    print("\n### Paired bootstrap (ratio, 95% CI, P(better)) per dataset\n")
    keys = set()
    for r in R.values(): keys |= set(r.get("bootstrap", {}).keys())
    for k in sorted(keys):
        print(f"\n{k}")
        for ds in R:
            b = R[ds].get("bootstrap", {}).get(k)
            if not b: continue
            if isinstance(b, dict):
                ratio = b.get("ratio", b.get("mean"))
                lo, hi = b.get("ci", [b.get("lo"), b.get("hi")])[:2] if isinstance(b.get("ci"), list) else (b.get("lo"), b.get("hi"))
                pw = b.get("p_better", b.get("wins"))
                print(f"  {ds:12s} ratio {ratio if ratio is None else round(float(ratio), 4)}  CI [{lo}, {hi}]  P(better) {pw}")
            else:
                print(f"  {ds:12s} {b}")
    # verdicts
    print("\n### Verdicts\n")
    for a in present:
        m = f"video_{a}"
        qi = summ["Q"].get(m, {}).get("vs_video_random")
        qii = summ["Q"].get(f"blend+{m}", {}).get("vs_blend")
        print(f"{a:10s} condition (i) Q vs random = {qi if qi is None else round(qi, 4)}   condition (ii) Q(blend+arm / blend) = {qii if qii is None else round(qii, 4)}")
    out = os.path.join(args.dir, f"summary_{args.stage}_{args.step}.json")
    json.dump(summ, open(out, "w"), indent=1, default=float)
    print("\nwritten", out)


if __name__ == "__main__":
    main()
