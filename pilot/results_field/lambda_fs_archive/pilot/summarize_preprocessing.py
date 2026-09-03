"""Aggregate paired pretrained/random preprocessing runs into summary.md / summary.json.

absolute_gain = (smean - pretrained) / smean      does preprocessing forecast better at all
transfer_gain = (random  - pretrained) / random   does PRETRAINING contribute (the real test)

A renderer that lowers absolute MSE but not transfer_gain has improved the trainable
head, not demonstrated video-pretraining transfer (PREPROCESSING_EXPERIMENTS.md s.13).
"""

import argparse
import glob
import json
import os

SCREEN = ("ETTh2", "electricity")
HELDOUT = ("ETTh1", "ETTm2", "traffic", "solar")


def load(dirs):
    rows = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.json")):
            j = json.load(open(f))
            c, r = j["config"], j["results"]
            key = (c["dataset"], c["renderer"], c["readout"], c["norm"], c["seed"])
            e = rows.setdefault(key, {"smean": r["smean"]["MSE"], "snaive": r["snaive"]["MSE"]})
            if "model_pt" in r:
                e["pt"] = r["model_pt"]["MSE"]
            if "model_rand" in r:
                e["rand"] = r["model_rand"]["MSE"]
            e["diag"] = c.get("input_diagnostics", {})
    return rows


def gains(e):
    ag = (e["smean"] - e["pt"]) / e["smean"] if "pt" in e else None
    tg = (e["rand"] - e["pt"]) / e["rand"] if ("pt" in e and "rand" in e) else None
    return ag, tg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+",
                    default=["pilot/results_field/preprocessing/screen",
                             "pilot/results_field/preprocessing/ablation",
                             "pilot/results_field/preprocessing/validation"])
    ap.add_argument("--out", default="pilot/results_field/preprocessing")
    args = ap.parse_args()
    rows = load(args.dirs)
    if not rows:
        print("no result JSONs found in", args.dirs)
        return

    lines = ["# Preprocessing experiment summary", "",
             "`transfer_gain` is the decisive column: pretrained vs the SAME architecture",
             "randomly initialized. Absolute gain over `smean` only shows the readout learned",
             "something.", "", "## Renderer screen", "",
             "| Dataset | Renderer | Readout | Norm | PT MSE | Rand MSE | smean | "
             "Transfer gain | Beats random? | Beats smean? |",
             "|---|---|---|---|---:|---:|---:|---:|---|---|"]
    out = {}
    for k in sorted(rows):
        ds, rend, ro, nm, seed = k
        e = rows[k]
        ag, tg = gains(e)
        out["|".join(map(str, k))] = {**{x: e.get(x) for x in ("pt", "rand", "smean", "snaive")},
                                      "absolute_gain": ag, "transfer_gain": tg}
        lines.append(
            f"| {ds} | {rend} | {ro} | {nm} | "
            f"{e.get('pt', float('nan')):.4f} | "
            f"{e.get('rand', float('nan')):.4f} | {e['smean']:.4f} | "
            f"{(tg * 100 if tg is not None else float('nan')):+.1f}% | "
            f"{'YES' if (tg or 0) > 0 else 'no'} | "
            f"{'YES' if (ag or 0) > 0 else 'no'} |")

    ho = [k for k in rows if k[0] in HELDOUT and "pt" in rows[k] and "rand" in rows[k]]
    if ho:
        tgs = [gains(rows[k])[1] for k in ho]
        ags = [gains(rows[k])[0] for k in ho]
        n_beat_smean = sum(a > 0 for a in ags)
        n_beat_rand = sum(t > 0 for t in tgs)
        mean_tg = sum(tgs) / len(tgs)
        lines += ["", "## Held-out verdict (section 13 criteria)", "",
                  f"- held-out pairs: {len(ho)}",
                  f"- (1) beats smean on {n_beat_smean}/{len(ho)} (need >=3 of 4)",
                  f"- (2) mean transfer gain {mean_tg*100:+.2f}% (need >= +3%)",
                  f"- (3) transfer gain positive on {n_beat_rand}/{len(ho)} (need >=3 of 4)",
                  "",
                  f"**VERDICT: {'POSITIVE' if (n_beat_smean>=3 and mean_tg>=0.03 and n_beat_rand>=3) else 'NOT a positive result'}**"]
        out["_heldout"] = {"n": len(ho), "beats_smean": n_beat_smean,
                           "beats_random": n_beat_rand, "mean_transfer_gain": mean_tg}
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, "summary.md"), "w").write("\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
