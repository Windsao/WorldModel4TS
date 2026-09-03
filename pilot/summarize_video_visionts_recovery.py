"""Recovery-study aggregation: full-configuration grouping, gates, CIs (spec s.8.1, s.13).

Groups by the COMPLETE configuration identity (dataset, layout, layout seed, lens variant,
backbone, stat rule) so G0/G1/G2 and different lens variants can never overwrite one another.
"""

import argparse, glob, json, math, os
import numpy as np

HELDOUT = ("ETTh1", "electricity", "traffic", "solar")
ALL6 = ("ETTh1", "ETTh2", "ETTm2", "electricity", "traffic", "solar")


def group_key(j):
    return (j.get("layout", "G0"), j.get("layout_seed", 0), j.get("variant", "L0"),
            j.get("stat_rule", "nearest_visible_physical"), j.get("backbone", "pretrained"))


def geo_ratio(method, reference):
    ds = sorted(set(method) & set(reference))
    if not ds:
        return float("nan")
    return math.exp(sum(math.log(method[d] / reference[d]) for d in ds) / len(ds))


def loo(method, reference):
    ds = sorted(set(method) & set(reference))
    return {d: math.exp(sum(math.log(method[x] / reference[x]) for x in ds if x != d) / (len(ds) - 1))
            for d in ds} if len(ds) > 1 else {}


def block_bootstrap(per_origin_m, per_origin_r, blocks, n_boot=2000, seed=0):
    """Paired moving-block bootstrap over forecast origins -> 95% CI for Q."""
    rng = np.random.default_rng(seed)
    ds = sorted(set(per_origin_m) & set(per_origin_r))
    qs = []
    for _ in range(n_boot):
        logs = []
        for d in ds:
            a, b = np.asarray(per_origin_m[d]), np.asarray(per_origin_r[d])
            n, blk = len(a), max(1, blocks.get(d, 1))
            nb = max(1, int(np.ceil(n / blk)))
            st = rng.integers(0, max(1, n - blk + 1), size=nb)
            idx = np.concatenate([np.arange(s, min(s + blk, n)) for s in st])[:n]
            logs.append(math.log(max(a[idx].mean(), 1e-12) / max(b[idx].mean(), 1e-12)))
        qs.append(math.exp(sum(logs) / len(logs)))
    q = np.sort(np.array(qs))
    return float(np.percentile(q, 2.5)), float(np.percentile(q, 97.5))


def load(root, sub):
    out = []
    for f in glob.glob(os.path.join(root, sub, "*.json")):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        if j.get("status") == "complete":
            out.append(j)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="pilot/results_field/video_visionts_recovery")
    args = ap.parse_args()
    L = ["# Video-VisionTS recovery summary", "",
         "> The previous Stage E1 outputs in `video_visionts/label_free_adaptation/` are",
         "> **`invalid_target_bug`**: they were trained against a target rendered from a zero",
         "> future. They are retained for provenance and excluded from every table here.", ""]
    au = load(args.root, "layout_audit")
    if au:
        L += ["## R3 — historical layout audit (regime HA)", "",
              "| dataset | layout | seed | ctl | native ratio | oracle raw | causal(phys) raw | TS model causal | TS zero causal |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|"]
        for j in sorted(au, key=lambda x: (x["layout"], x["dataset"], x["layout_seed"])):
            m = j["metrics"]
            L.append(f"| {j['dataset']} | {j['layout']} | {j['layout_seed']} | {j['control'][:4]} | "
                     f"**{m['native_ratio']:.4f}** | {m.get('raw_oracle', float('nan')):.5f} | "
                     f"{m.get('raw_nearest_visible_physical', float('nan')):.5f} | "
                     f"{m.get('ts_model_nearest_visible_physical_mse', float('nan')):.4f} | "
                     f"{m.get('ts_zero_nearest_visible_physical_mse', float('nan')):.4f} |")
        L.append("")
    he = load(args.root, "historical_eval")
    if he:
        L += ["## R4 — corrected lens historical evaluation (regime LA)", "",
              "| variant | layout | backbone | seed | aggregate ratio vs L0 | best R_val | mean-clamp | logstd-sat |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
        for j in sorted(he, key=lambda x: (x["variant"], x["layout"], x["backbone"], x["seed"])):
            cs = j.get("clamp_stats", {})
            mc = np.mean([v["mean_clamp_rate"] for v in cs.values()]) if cs else float("nan")
            sa = np.mean([v["logstd_saturation_rate"] for v in cs.values()]) if cs else float("nan")
            L.append(f"| {j['variant']} | {j['layout']} | {j['backbone']} | {j['seed']} | "
                     f"**{j['aggregate_ratio_vs_L0']:.4f}** | {j['best_R_val']:.4f} | {mc:.3f} | {sa:.3f} |")
        L.append("")
    sf = load(args.root, "strict_forecast")
    if sf:
        groups = {}
        for j in sf:
            groups.setdefault(group_key(j), {})[j["dataset"]] = j
        for k, rows in sorted(groups.items()):
            L += [f"## R5 — strict forecast  layout={k[0]} seed={k[1]} lens={k[2]} stat={k[3]} backbone={k[4]}", "",
                  "| dataset | method | zeroed | random | VisionTS | Stage-D right_10 | smean | snaive | q vs VTS |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
            M, V = {}, {}
            for d in ALL6:
                if d not in rows: continue
                m = rows[d]["metrics"]; M[d] = m["method"]; V[d] = m["visionts"]
                L.append(f"| {d} | {m['method']:.4f} | {m.get('zeroed', float('nan')):.4f} | "
                         f"{m.get('random', float('nan')):.4f} | {m['visionts']:.4f} | "
                         f"{m.get('stage_d', float('nan')):.4f} | {m['smean']:.4f} | {m['snaive']:.4f} | "
                         f"{m['method']/m['visionts']:.3f} |")
            qa, qh = geo_ratio(M, V), geo_ratio({d: M[d] for d in M if d in HELDOUT},
                                                {d: V[d] for d in V if d in HELDOUT})
            wins = sum(1 for d in M if M[d] < V[d])
            L += ["", f"- **Q_all = {qa:.4f}**  ({'MET' if qa < 1 else 'NOT MET'}), "
                      f"**Q_heldout = {qh:.4f}**", f"- win/loss vs VisionTS: {wins}/{len(M)-wins}",
                  f"- leave-one-out Q_all: " + ", ".join(f"{a}:{b:.3f}" for a, b in loo(M, V).items()), ""]
    os.makedirs(args.root, exist_ok=True)
    open(os.path.join(args.root, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
