"""Assemble the final paper-protocol table: validation-selected variants (tag vmf_val / vmf_val_ens)
where they exist, otherwise the rule-based recipe (vmf_auto / vmf_auto_ens). Prints markdown and
writes pilot/results_field/lsf/final_table.json."""
import glob, json, os
import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_field", "lsf")
DS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "weather"]
HS = [96, 192, 336, 720]
VTS = {"ETTh1": {96: 0.353, 192: 0.392, 336: 0.407, 720: 0.406}, "ETTh2": {96: 0.271, 192: 0.328, 336: 0.345, 720: 0.388},
       "ETTm1": {96: 0.341, 192: 0.360, 336: 0.377, 720: 0.416}, "ETTm2": {96: 0.228, 192: 0.262, 336: 0.293, 720: 0.343},
       "electricity": {96: 0.177, 192: 0.188, 336: 0.207, 720: 0.256}, "weather": {96: 0.220, 192: 0.244, 336: 0.280, 720: 0.330}}
VTS_MAE = {"ETTh1": {96: 0.383, 192: 0.410, 336: 0.423, 720: 0.441}, "ETTh2": {96: 0.328, 192: 0.367, 336: 0.381, 720: 0.422},
           "ETTm1": {96: 0.347, 192: 0.360, 336: 0.374, 720: 0.405}, "ETTm2": {96: 0.282, 192: 0.305, 336: 0.328, 720: 0.370},
           "electricity": {96: 0.266, 192: 0.277, 336: 0.296, 720: 0.337}, "weather": {96: 0.257, 192: 0.275, 336: 0.299, 720: 0.337}}


def load(tag):
    parts = {}
    for f in glob.glob(os.path.join(D, "*.json")):
        r = json.load(open(f))
        if r.get("model") != tag or r.get("split", "test") != "test":
            continue
        parts.setdefault((r["dataset"], r["pred_len"]), []).append(r)
    out = {}
    for k, rs in parts.items():
        se = sum(r["se"] for r in rs) if all("se" in r for r in rs) else None
        if len(rs) == 1 or se is None:
            r = rs[0]
            out[k] = {"mse": r["mse"], "mae": r["mae"], "ens": r.get("ensemble_mse"), "ens_mae": r.get("ensemble_mae"),
                      "vts_own": (r.get("extra_models") or {}).get("visionts"), "stride": r["stride"], "mode": r["mode"], "scales": r["scales"]}
        else:
            ae = sum(r["ae"] for r in rs); cnt = sum(r["cnt"] for r in rs)
            out[k] = {"mse": se / cnt, "mae": ae / cnt, "ens": None, "ens_mae": None, "vts_own": None, "stride": rs[0]["stride"], "mode": rs[0]["mode"], "scales": rs[0]["scales"], "parts": len(rs)}
    return out


def main():
    auto, val, auto_e, val_e = load("vmf_auto"), load("vmf_val"), load("vmf_auto_ens"), load("vmf_val_ens")
    final, src = {}, {}
    for ds in DS:
        for H in HS:
            k = (ds, H)
            r = val.get(k) or auto.get(k)
            e = val_e.get(k) if k in val else auto_e.get(k)
            if r is None:
                continue
            final[k] = {"mse": r["mse"], "mae": r["mae"], "mode": r["mode"], "scales": r["scales"],
                        "ens": e["ens"] if e else None, "ens_mae": e["ens_mae"] if e else None,
                        "ens_stride": e["stride"] if e else None, "selected_on": "validation" if k in val else "rule"}
    print("| dataset | H | ours MSE | ours MAE | VisionTS MSE | VisionTS MAE | ours/VTS | ours+VTS MSE | recipe (selected on) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    avg = {}
    for ds in DS:
        ms, ma, es = [], [], []
        for H in HS:
            f = final.get((ds, H))
            if not f:
                print(f"| {ds} | {H} | — | — | {VTS[ds][H]:.3f} | {VTS_MAE[ds][H]:.3f} | — | — | — |"); continue
            ms.append(f["mse"]); ma.append(f["mae"])
            if f["ens"] is not None: es.append(f["ens"])
            ens = "—" if f["ens"] is None else f"{f['ens']:.3f}" + ("" if f["ens_stride"] == 1 else f" (stride {f['ens_stride']})")
            print(f"| {ds} | {H} | {f['mse']:.3f} | {f['mae']:.3f} | {VTS[ds][H]:.3f} | {VTS_MAE[ds][H]:.3f} | {f['mse'] / VTS[ds][H]:.2f} | {ens} | {f['mode']} sc{f['scales'].replace(',', '')} ({f['selected_on']}) |")
        if len(ms) == 4:
            avg[ds] = {"mse": float(np.mean(ms)), "mae": float(np.mean(ma)), "ens": float(np.mean(es)) if len(es) == 4 else None,
                       "vts": float(np.mean(list(VTS[ds].values())))}
    print("\n| avg MSE | " + " | ".join(DS) + " | mean |")
    print("|---|" + "---:|" * 7)
    for name, key in (("VisionTS (paper)", "vts"), ("ours", "mse"), ("ours + VisionTS", "ens")):
        row = [avg[ds][key] if ds in avg else None for ds in DS]
        ok = [v for v in row if v is not None]
        print(f"| {name} | " + " | ".join("—" if v is None else f"{v:.3f}" for v in row) + f" | {np.mean(ok):.3f} |" if len(ok) == 6 else f"| {name} | " + " | ".join("—" if v is None else f"{v:.3f}" for v in row) + " | — |")
    json.dump({"cells": {f"{k[0]}_{k[1]}": v for k, v in final.items()}, "avg": avg}, open(os.path.join(D, "final_table.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
