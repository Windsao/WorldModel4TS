"""Matched-mask aggregation, subset tables and Stage-M2 decision gates."""
import argparse, glob, json, math, os
import numpy as np

SEL = ("ETTh2", "ETTm2")


def group_key(j):
    return (j.get("dataset"), j.get("mask"), j.get("seed", 0), j.get("schedule"), j.get("carrier"))


def geo_ratio(method, reference):
    ds = sorted(set(method) & set(reference))
    return math.exp(sum(math.log(method[d] / reference[d]) for d in ds) / len(ds)) if ds else float("nan")


def loo(method, reference):
    ds = sorted(set(method) & set(reference))
    return {d: math.exp(sum(math.log(method[x] / reference[x]) for x in ds if x != d) / (len(ds) - 1))
            for d in ds} if len(ds) > 1 else {}


def block_bootstrap(pm, pr, blocks, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ds = sorted(set(pm) & set(pr)); qs = []
    for _ in range(n_boot):
        logs = []
        for d in ds:
            a, b = np.asarray(pm[d]), np.asarray(pr[d]); n, blk = len(a), max(1, blocks.get(d, 1))
            st = rng.integers(0, max(1, n - blk + 1), size=max(1, int(np.ceil(n / blk))))
            idx = np.concatenate([np.arange(s, min(s + blk, n)) for s in st])[:n]
            logs.append(math.log(max(a[idx].mean(), 1e-12) / max(b[idx].mean(), 1e-12)))
        qs.append(math.exp(sum(logs) / len(logs)))
    q = np.sort(np.array(qs))
    return float(np.percentile(q, 2.5)), float(np.percentile(q, 97.5))


def agg(per_ds, key):
    v = [per_ds[d][key] for d in per_ds if key in per_ds[d] and per_ds[d][key] > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="pilot/results_field/video_visionts_matched_mask")
    args = ap.parse_args()
    files = glob.glob(os.path.join(args.root, "matched_audit", "*.json"))
    data = {}
    for f in files:
        j = json.load(open(f))
        if j.get("status") == "complete":
            data[j["dataset"]] = j
    L = ["# Matched-mask factorization summary", ""]
    if not data:
        L.append("no completed results")
    else:
        L += ["Geometry fixed at 4 context / 10 future columns for every mask; the image is",
              "rendered once per batch and reused (hash-checked).", "",
              "| mask | seed | ctx masked | fut masked | ctx ratio | fut ratio | fut PT/RAND | fut nondeg ratio |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        keys = sorted({k for d in data.values() for k in d["masks"]})
        for k in keys:
            row = {}
            for ds in data:
                if k not in data[ds]["masks"]: continue
                m = data[ds]["masks"][k]
                p, r = m["native_pt"], m["native_rand"]
                row[ds] = {"ctx": p.get("context", {}).get("native_ratio_zero", float("nan")),
                           "fut": p.get("future", {}).get("native_ratio_zero", float("nan")),
                           "fnd": p.get("future_nondegenerate", {}).get("native_ratio_zero", float("nan")),
                           "pr": (p.get("future", {}).get("mse_pt", float("nan")) /
                                  max(r.get("future", {}).get("mse_pt", float("nan")), 1e-12))}
            if not row: continue
            any_ds = data[list(data)[0]]["masks"].get(k, {})
            L.append(f"| {k.split('|')[0]} | {k.split('seed')[-1]} | {any_ds.get('masked_context','-')} | "
                     f"{any_ds.get('masked_future','-')} | {agg(row,'ctx'):.4f} | {agg(row,'fut'):.4f} | "
                     f"{agg(row,'pr'):.4f} | {agg(row,'fnd'):.4f} |")
    os.makedirs(args.root, exist_ok=True)
    open(os.path.join(args.root, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
