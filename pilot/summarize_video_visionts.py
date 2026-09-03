"""Aggregation, gates and decision report for VIDEO_VISIONTS_EXPERIMENTS.md (section 3.4/13).

Never averages raw MSE across datasets: scales differ by an order of magnitude. Uses the
equal-dataset geometric mean of paired ratios q_d = MSE(method,d) / MSE(reference,d).
"""

import argparse
import glob
import json
import math
import os

import numpy as np

HELDOUT = ("ETTh1", "electricity", "traffic", "solar")
SELECTION = ("ETTh2", "ETTm2")
ALL6 = ("ETTh1", "ETTh2", "ETTm2", "electricity", "traffic", "solar")


def geo_ratio(method, reference):
    """Q = exp(mean_d log(method_d / reference_d)) over the common datasets."""
    ds = sorted(set(method) & set(reference))
    if not ds:
        return float("nan")
    return math.exp(sum(math.log(method[d] / reference[d]) for d in ds) / len(ds))


def gain(method, reference):
    return 1.0 - geo_ratio(method, reference)


def loo(method, reference):
    """Leave-one-dataset-out sensitivity of Q."""
    ds = sorted(set(method) & set(reference))
    out = {}
    for d in ds:
        sub = [x for x in ds if x != d]
        out[d] = math.exp(sum(math.log(method[x] / reference[x]) for x in sub) / len(sub))
    return out


def block_bootstrap_q(per_origin_method, per_origin_ref, block, n_boot=2000, seed=0):
    """Paired moving-block bootstrap over forecast origins -> 95% CI for Q.

    per_origin_* : {dataset: 1-D array of per-origin squared error means}
    Overlapping test windows make i.i.d. resampling invalid, so blocks of at least
    ceil(H/stride) consecutive origins are resampled together.
    """
    rng = np.random.default_rng(seed)
    ds = sorted(set(per_origin_method) & set(per_origin_ref))
    qs = []
    for _ in range(n_boot):
        logs = []
        for d in ds:
            a, b = per_origin_method[d], per_origin_ref[d]
            n = len(a)
            nb = max(1, int(np.ceil(n / block)))
            starts = rng.integers(0, max(1, n - block + 1), size=nb)
            idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
            logs.append(math.log(a[idx].mean() / b[idx].mean()))
        qs.append(math.exp(sum(logs) / len(logs)))
    qs = np.sort(np.array(qs))
    return float(np.percentile(qs, 2.5)), float(np.percentile(qs, 97.5))


def load(dirs):
    rows = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.json")):
            try:
                j = json.load(open(f))
            except Exception:
                continue
            if j.get("status") != "complete":
                continue
            rows.append(j)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="pilot/results_field/video_visionts")
    args = ap.parse_args()
    dirs = [os.path.join(args.root, x) for x in
            ("visionts_reference", "decoder_audit", "historical_reconstruction",
             "strict_forecast", "label_free_adaptation")]
    rows = load(dirs)
    lines = ["# Video-VisionTS diagnostic summary", ""]
    if not rows:
        lines.append("No completed result JSONs found yet.")
        os.makedirs(args.root, exist_ok=True)
        open(os.path.join(args.root, "summary.md"), "w").write("\n".join(lines) + "\n")
        print("\n".join(lines)); return

    # ---- Stage A reference table
    ref = {r["dataset"]: r for r in rows if r.get("stage") == "A"}
    if ref:
        lines += ["## Stage A — VisionTS reference", "",
                  "| dataset | visionts | zeroed | random | smean | snaive |",
                  "|---|---:|---:|---:|---:|---:|"]
        for d in ALL6:
            if d not in ref: continue
            m = ref[d]["metrics"]
            lines.append(f"| {d} | {m['visionts_pretrained']:.4f} | {m['visionts_zeroed']:.4f} | "
                         f"{m['visionts_random']:.4f} | {m['smean']:.4f} | {m['snaive']:.4f} |")
        V = {d: ref[d]["metrics"]["visionts_pretrained"] for d in ref}
        Z = {d: ref[d]["metrics"]["visionts_zeroed"] for d in ref}
        R = {d: ref[d]["metrics"]["visionts_random"] for d in ref}
        q_z, q_r = geo_ratio(V, Z), geo_ratio(V, R)
        lines += ["", f"- Q(VisionTS / zeroed)  = **{q_z:.4f}**  (gate: < 1)",
                  f"- Q(VisionTS / random)  = **{q_r:.4f}**  (gate: < 1)",
                  f"- **Stage A gate: {'PASS' if (q_z < 1 and q_r < 1) else 'FAIL'}**", ""]

    # ---- Stage C native reconstruction table
    hc = [r for r in rows if r.get("stage") == "C"]
    if hc:
        lines += ["## Stage C — historical reconstruction (regime HA)", "",
                  "| dataset | renderer | mask | native model | native zero | native_ratio | oracle TS MSE | causal TS MSE |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
        for r in sorted(hc, key=lambda x: (x["dataset"], x["renderer"], x["mask"])):
            m = r["metrics"]
            lines.append(f"| {r['dataset']} | {r['renderer']} | {r['mask']} | "
                         f"{m.get('native_model', float('nan')):.4f} | {m.get('native_zero', float('nan')):.4f} | "
                         f"**{m.get('native_ratio', float('nan')):.4f}** | "
                         f"{m.get('ts_oracle_mse', float('nan')):.4f} | {m.get('ts_causal_mse', float('nan')):.4f} |")
        lines.append("")

    # ---- Stage D strict forecast
    sd = [r for r in rows if r.get("stage") == "D"]
    if sd:
        lines += ["## Stage D — strict zero-shot forecast (regime ZS)", "",
                  "| dataset | pretrained | random | zeroed | visionts | smean | snaive | q vs VisionTS |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        M, RR, ZZ, VV = {}, {}, {}, {}
        for r in sorted(sd, key=lambda x: x["dataset"]):
            m = r["metrics"]; d = r["dataset"]
            M[d] = m["pretrained"]; RR[d] = m.get("random", float("nan"))
            ZZ[d] = m.get("zeroed", float("nan")); VV[d] = m.get("visionts", float("nan"))
            q = m["pretrained"] / m["visionts"] if m.get("visionts") else float("nan")
            lines.append(f"| {d} | {m['pretrained']:.4f} | {RR[d]:.4f} | {ZZ[d]:.4f} | "
                         f"{VV[d]:.4f} | {m['smean']:.4f} | {m['snaive']:.4f} | {q:.3f} |")
        Vd = {d: v for d, v in VV.items() if v == v}
        q_all = geo_ratio(M, Vd)
        q_ho = geo_ratio({d: M[d] for d in M if d in HELDOUT},
                         {d: Vd[d] for d in Vd if d in HELDOUT})
        lines += ["", f"- **Q_all = {q_all:.4f}** (gain vs VisionTS {100*(1-q_all):+.1f}%)",
                  f"- **Q_heldout = {q_ho:.4f}** (gain {100*(1-q_ho):+.1f}%)",
                  f"- Q(pretrained/random)  = {geo_ratio(M, RR):.4f}",
                  f"- Q(pretrained/zeroed)  = {geo_ratio(M, ZZ):.4f}",
                  f"- leave-one-out Q_all: " + ", ".join(f"{k}:{v:.3f}" for k, v in loo(M, Vd).items()),
                  "",
                  f"- **Primary forecasting claim (Q_all < 1): {'MET' if q_all < 1 else 'NOT MET'}**",
                  f"- **Attribution (pretrained beats random and zeroed): "
                  f"{'MET' if geo_ratio(M, RR) < 1 and geo_ratio(M, ZZ) < 1 else 'NOT MET'}**", ""]

    os.makedirs(args.root, exist_ok=True)
    open(os.path.join(args.root, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
