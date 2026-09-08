"""Aggregate pilot/eval_lsf.py outputs into the paper-style LSF table and compare with published
zero-shot numbers (VisionTS paper Table 9; averages from VisionTS / VisionTS++ tables).

usage: python pilot/summarize_lsf.py --dir <lsf out dir> --tag vmf_auto [--tag2 visionts]
"""
import argparse, glob, json, os
import numpy as np

DS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "weather"]
HS = [96, 192, 336, 720]
# VisionTS paper, Table 9 (zero-shot, MSE / MAE per horizon)
VTS_PAPER = {
    "ETTh1": {96: (0.353, 0.383), 192: (0.392, 0.410), 336: (0.407, 0.423), 720: (0.406, 0.441)},
    "ETTh2": {96: (0.271, 0.328), 192: (0.328, 0.367), 336: (0.345, 0.381), 720: (0.388, 0.422)},
    "ETTm1": {96: (0.341, 0.347), 192: (0.360, 0.360), 336: (0.377, 0.374), 720: (0.416, 0.405)},
    "ETTm2": {96: (0.228, 0.282), 192: (0.262, 0.305), 336: (0.293, 0.328), 720: (0.343, 0.370)},
    "electricity": {96: (0.177, 0.266), 192: (0.188, 0.277), 336: (0.207, 0.296), 720: (0.256, 0.337)},
    "weather": {96: (0.220, 0.257), 192: (0.244, 0.275), 336: (0.280, 0.299), 720: (0.330, 0.337)},
}
# average-over-horizon MSE from the VisionTS paper (zero-shot table)
AVG_MSE_PUBLISHED = {
    "VisionTS (paper)": {"ETTh1": 0.390, "ETTh2": 0.333, "ETTm1": 0.374, "ETTm2": 0.282, "electricity": 0.207, "weather": 0.269},
    "Moirai-small": {"ETTh1": 0.400, "ETTh2": 0.341, "ETTm1": 0.448, "ETTm2": 0.300, "electricity": 0.233, "weather": 0.242},
    "Moirai-base": {"ETTh1": 0.434, "ETTh2": 0.346, "ETTm1": 0.382, "ETTm2": 0.272, "electricity": 0.188, "weather": 0.238},
    "Moirai-large": {"ETTh1": 0.510, "ETTh2": 0.354, "ETTm1": 0.390, "ETTm2": 0.276, "electricity": 0.188, "weather": 0.260},
    "TimeLLM": {"ETTh1": 0.556, "ETTh2": 0.370, "ETTm1": 0.404, "ETTm2": 0.277, "electricity": 0.175, "weather": 0.234},
    "GPT4TS": {"ETTh1": 0.590, "ETTh2": 0.397, "ETTm1": 0.464, "ETTm2": 0.293, "electricity": 0.176, "weather": 0.238},
}


def load(d, tag):
    """merge partial (origin-range) runs of the same (dataset, H) by their sums."""
    parts = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        r = json.load(open(f))
        if r.get("model") != tag:
            continue
        parts.setdefault((r["dataset"], r["pred_len"]), []).append(r)
    out = {}
    for k, rs in parts.items():
        if len(rs) == 1 and not rs[0].get("origin_range"):
            out[k] = rs[0]; continue
        se = sum(r["se"] for r in rs); ae = sum(r["ae"] for r in rs); cnt = sum(r["cnt"] for r in rs)
        n = sum(r["n_origins"] for r in rs)
        r0 = dict(rs[0]); r0.update({"mse": se / cnt, "mae": ae / cnt, "n_origins": n, "merged_parts": len(rs)})
        if n != rs[0].get("n_origins_all", n):
            r0["INCOMPLETE"] = f"{n}/{rs[0].get('n_origins_all')}"
        out[k] = r0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="pilot/results_field/lsf")
    ap.add_argument("--tag", default="vmf_auto")
    args = ap.parse_args()
    R = load(args.dir, args.tag)
    print(f"\n### {args.tag} vs VisionTS (paper), MSE / MAE per horizon\n")
    print("| dataset | H | ours MSE | ours MAE | VisionTS MSE | VisionTS MAE | ours/VTS |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    avg = {}
    for ds in DS:
        ms, ma = [], []
        for H in HS:
            r = R.get((ds, H))
            v = VTS_PAPER[ds][H]
            if r is None:
                print(f"| {ds} | {H} | — | — | {v[0]:.3f} | {v[1]:.3f} | — |"); continue
            ms.append(r["mse"]); ma.append(r["mae"])
            flag = " (partial " + r["INCOMPLETE"] + ")" if "INCOMPLETE" in r else ""
            print(f"| {ds} | {H} | {r['mse']:.3f}{flag} | {r['mae']:.3f} | {v[0]:.3f} | {v[1]:.3f} | {r['mse'] / v[0]:.3f} |")
        if len(ms) == 4:
            avg[ds] = (float(np.mean(ms)), float(np.mean(ma)))
    print(f"\n### average over horizons (MSE), published zero-shot baselines vs {args.tag}\n")
    models = list(AVG_MSE_PUBLISHED) + [args.tag]
    print("| model | " + " | ".join(DS) + " | mean |")
    print("|---|" + "---:|" * (len(DS) + 1))
    for m in models:
        row = []
        for ds in DS:
            v = AVG_MSE_PUBLISHED[m].get(ds) if m in AVG_MSE_PUBLISHED else (avg.get(ds, (None,))[0])
            row.append(v)
        vals = [v for v in row if v is not None]
        mean = np.mean(vals) if len(vals) == len(DS) else None
        print(f"| {m} | " + " | ".join("—" if v is None else f"{v:.3f}" for v in row) + f" | {'—' if mean is None else f'{mean:.3f}'} |")
    json.dump({"per_horizon": {f"{k[0]}_{k[1]}": {"mse": v["mse"], "mae": v["mae"]} for k, v in R.items()}, "avg": avg},
              open(os.path.join(args.dir, f"summary_{args.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
