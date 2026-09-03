"""Aggregation, sequential gates and Markdown summary for the carrier-signal study."""
import argparse, glob, json, math, os
import numpy as np

DS2 = ("ETTh2", "ETTm2")
IDENTITY = ("dataset", "split", "manifest_seed", "checkpoint", "carrier", "frequency",
            "norm_const", "geometry", "mask", "input_arm", "backbone_arm",
            "random_init_seed", "shuffle_seed", "decoder", "calibration", "git_commit")


def identity_of(j):
    return tuple(str(j.get(k) if k in j else j.get("config", {}).get(k)) for k in IDENTITY)


def load_group(paths):
    """Load JSONs, raising on duplicate full identities instead of overwriting by dataset."""
    out, seen = [], {}
    for p in paths:
        j = json.load(open(p))
        if j.get("status") != "complete":
            continue
        k = identity_of(j)
        if k in seen:
            raise ValueError(f"duplicate full identity {k}: {seen[k]} and {p}")
        seen[k] = p
        out.append(j)
    return out


def assert_paired(a, b):
    ha = a.get("manifest", {}).get("pairs_sha1")
    hb = b.get("manifest", {}).get("pairs_sha1")
    if ha != hb:
        raise ValueError(f"mixed pair hashes in a paired comparison: {ha} vs {hb}")
    return True


def q2(num_by_ds, den_by_ds):
    ds = sorted(set(num_by_ds) & set(den_by_ds))
    if not ds:
        return float("nan")
    return math.exp(sum(math.log(num_by_ds[d] / den_by_ds[d]) for d in ds) / len(ds))


def gate_codec(codec):
    """Section 4.3."""
    o0 = max(c["codec_oracles"]["o0_max_abs_err"] for c in codec.values())
    diff = max(c["codec_oracles"]["o2_o3_max_abs_diff"] for c in codec.values())
    r_zero = {d: c["codec_oracles"]["o3_over_ctxmean"] for d, c in codec.items()}
    r_i0 = {d: c["codec_oracles"]["o3_over_i0"] for d, c in codec.items()}
    checks = [("O0 max abs err <= 1e-5", o0, 1e-5, o0 <= 1e-5),
              ("O2/O3 max abs diff <= 2e-4", diff, 2e-4, diff <= 2e-4),
              ("O3/ctx-mean <= 0.50 per dataset", max(r_zero.values()), 0.50,
               all(v <= 0.50 for v in r_zero.values())),
              ("geo O3/I0 <= 0.90", q2(r_i0, {d: 1.0 for d in r_i0}), 0.90,
               q2(r_i0, {d: 1.0 for d in r_i0}) <= 0.90),
              ("no dataset with O3/I0 > 1.00", max(r_i0.values()), 1.00,
               all(v <= 1.00 for v in r_i0.values()))]
    return {"checks": [{"name": n, "value": float(v), "threshold": float(t), "pass": bool(p)}
                       for n, v, t, p in checks],
            "pass": all(p for *_, p in checks)}


def gate_conditional(z):
    """Section 10.2. z[ds][arm] = z-MSE."""
    r_b1 = q2({d: z[d]["pt_true"] for d in z}, {d: z[d]["b1_neutral"] for d in z})
    r_xs = q2({d: z[d]["pt_true"] for d in z}, {d: z[d]["pt_shuffle"] for d in z})
    r_ys = q2({d: z[d]["pt_true"] for d in z}, {d: z[d]["y_shuffle"] for d in z})
    worse = any(z[d]["pt_true"] > z[d]["pt_shuffle"] for d in z)
    corr_ok = all(z[d]["pearson"] > 0 for d in z)
    checks = [("Q2(PT true / B1 neutral) <= 0.95", r_b1, 0.95, r_b1 <= 0.95),
              ("Q2(PT true / X_shuffle) <= 0.95", r_xs, 0.95, r_xs <= 0.95),
              ("Q2(PT true / Y_shuffle) <= 0.95", r_ys, 0.95, r_ys <= 0.95),
              ("no dataset worse than its shuffled control", float(worse), 0.0, not worse),
              ("Pearson z corr > 0 on both", min(z[d]["pearson"] for d in z), 0.0, corr_ok)]
    return {"checks": [{"name": n, "value": float(v), "threshold": float(t), "pass": bool(p)}
                       for n, v, t, p in checks], "pass": all(p for *_, p in checks)}


def gate_pretrain(z, nat):
    r = q2({d: z[d]["pt_true"] for d in z}, {d: z[d]["rand_true"] for d in z})
    both = all(z[d]["pt_true"] < z[d]["rand_true"] for d in z)
    rn = q2({d: nat[d]["pt_true"] for d in nat}, {d: nat[d]["b1_neutral"] for d in nat})
    checks = [("Q2(z-MSE PT/RAND) <= 0.95", r, 0.95, r <= 0.95),
              ("pretrained better on both datasets", float(both), 1.0, both),
              ("Q2(native PT / B1) <= 0.95", rn, 0.95, rn <= 0.95)]
    return {"checks": [{"name": n, "value": float(v), "threshold": float(t), "pass": bool(p)}
                       for n, v, t, p in checks], "pass": all(p for *_, p in checks)}


def gate_forecast(ts):
    r_cm = q2({d: ts[d]["d2"] for d in ts}, {d: ts[d]["ctx_mean"] for d in ts})
    r_i0 = q2({d: ts[d]["d2"] for d in ts}, {d: ts[d]["i0"] for d in ts})
    worst = max(ts[d]["d2"] / ts[d]["i0"] for d in ts)
    checks = [("Q2(TS D2 / context-mean) <= 0.95", r_cm, 0.95, r_cm <= 0.95),
              ("Q2(TS D2 / I0) <= 0.97", r_i0, 0.97, r_i0 <= 0.97),
              ("no dataset > 5% worse than I0", worst, 1.05, worst <= 1.05)]
    return {"checks": [{"name": n, "value": float(v), "threshold": float(t), "pass": bool(p)}
                       for n, v, t, p in checks], "pass": all(p for *_, p in checks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="pilot/results_field/video_visionts_carrier_signal")
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()
    L = ["# Carrier-signal validation summary", ""]
    gd = os.path.join(args.root, "gates"); os.makedirs(gd, exist_ok=True)

    codec = {j["dataset"]: j for j in load_group(glob.glob(os.path.join(args.root, "codec", "*.json")))}
    if codec:
        g = gate_codec(codec)
        json.dump(g, open(os.path.join(gd, "codec_gate.json"), "w"), indent=2)
        L += ["## Table A — codec ceiling", "",
              "| dataset | clip frac | O1 MSE | O2 MSE | O3 MSE | O3/ctx-mean | O3/I0 | O2-O3 max diff |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for d in DS2:
            if d not in codec: continue
            o = codec[d]["codec_oracles"]
            L.append(f"| {d} | {o['clip_frac']:.4f} | {o['o1_mse']:.5f} | {o['o2_mse']:.5f} | "
                     f"{o['o3_mse']:.5f} | {o['o3_over_ctxmean']:.4f} | {o['o3_over_i0']:.4f} | "
                     f"{o['o2_o3_max_abs_diff']:.2e} |")
        L += ["", "### G_codec", "", "| check | value | threshold | status |", "|---|---:|---:|---|"]
        for c in g["checks"]:
            L.append(f"| {c['name']} | {c['value']:.6g} | {c['threshold']:.6g} | "
                     f"{'PASS' if c['pass'] else '**FAIL**'} |")
        L += ["", f"**G_codec: {'PASS' if g['pass'] else 'FAIL'}**", ""]

    sig = load_group(glob.glob(os.path.join(args.root, "signal", "*.json")))
    if sig:
        S = {j["dataset"]: j for j in sig}
        z = {d: S[d]["phase_metrics"]["z_mse"] for d in S}
        for d in S:
            z[d]["pearson"] = S[d]["phase_metrics"]["pearson_pt_true"]
        nat = {d: S[d]["native_metrics"] for d in S}
        ts = {d: S[d]["ts_metrics"] for d in S}
        gc = gate_conditional(z); gp = gate_pretrain(z, nat); gf = gate_forecast(ts)
        json.dump({"conditional": gc, "pretrain": gp, "forecast": gf},
                  open(os.path.join(gd, "conditional_gate.json"), "w"), indent=2)
        L += ["## Table B/C/D — signal", "",
              "| dataset | arm | native MSE | /B0 zero | /B1 neutral | z MSE | phase cos |",
              "|---|---|---:|---:|---:|---:|---:|"]
        for d in DS2:
            if d not in S: continue
            n = S[d]["native_metrics"]; ph = S[d]["phase_metrics"]
            for arm in ("b0_zero", "b1_neutral", "b3_smean", "b4_snaive",
                        "pt_true", "pt_shuffle", "pt_neutral", "rand_true"):
                if arm not in n: continue
                L.append(f"| {d} | {arm} | {n[arm]:.5f} | {n[arm]/n['b0_zero']:.4f} | "
                         f"{n[arm]/n['b1_neutral']:.4f} | "
                         f"{ph['z_mse'].get(arm, float('nan')):.5f} | "
                         f"{ph['phase_cos'].get(arm, float('nan')):.4f} |")
        L += ["", "| dataset | decoder | z MSE | rho median | TS MSE | /ctx-mean | /smean | /I0 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
        for d in DS2:
            if d not in S: continue
            t = S[d]["ts_metrics"]
            for dec in ("d0", "d1", "d2", "d3"):
                if dec not in t: continue
                L.append(f"| {d} | {dec.upper()} | {S[d]['phase_metrics']['z_mse'].get(dec, float('nan')):.5f} | "
                         f"{S[d]['phase_metrics'].get('rho_median', float('nan')):.4f} | {t[dec]:.5f} | "
                         f"{t[dec]/t['ctx_mean']:.4f} | {t[dec]/t['smean']:.4f} | {t[dec]/t['i0']:.4f} |")
        for nm, g in (("G_conditional", gc), ("G_pretrain", gp), ("G_forecast", gf)):
            L += ["", f"### {nm}", "", "| check | value | threshold | status |", "|---|---:|---:|---|"]
            for c in g["checks"]:
                L.append(f"| {c['name']} | {c['value']:.6g} | {c['threshold']:.6g} | "
                         f"{'PASS' if c['pass'] else '**FAIL**'} |")
            L.append(f"\n**{nm}: {'PASS' if g['pass'] else 'FAIL'}**")
    os.makedirs(args.root, exist_ok=True)
    open(os.path.join(args.root, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
