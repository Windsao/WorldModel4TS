"""Final comparison table under the VisionTS LSF protocol: our checkpoint(s) vs published zero-shot baselines.

usage: python pilot/final_bigtable.py <tag> [<tag> ...] [--names "Ours (32f)" ...] [--dir /nyx-storage1/hanliu/wm4ts/lsf]
Each tag = the --tag used by eval_lsf.py; electricity halves (origin-range) are merged by summed squared / absolute
errors. Prints a markdown table (4-horizon average MSE/MAE per dataset, six-dataset mean) with the best value per
column in bold, plus a per-horizon table for the first tag. Baseline numbers: VisionTS++ paper Table 4 (base-size
models only; VisionTS++ large omitted because our backbone is ViT-Base)."""
import argparse, glob, json

DS = ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "electricity", "weather"]
HS = [96, 192, 336, 720]
BASELINES = [  # name -> {dataset: (mse, mae)}
    ("VisionTS++ base", {"ETTm1": (0.360, 0.372), "ETTm2": (0.244, 0.298), "ETTh1": (0.402, 0.416), "ETTh2": (0.333, 0.370), "electricity": (0.184, 0.265), "weather": (0.222, 0.241)}),
    ("VisionTS", {"ETTm1": (0.374, 0.372), "ETTm2": (0.318, 0.366), "ETTh1": (0.390, 0.414), "ETTh2": (0.333, 0.375), "electricity": (0.207, 0.294), "weather": (0.269, 0.292)}),
    ("Moirai small", {"ETTm1": (0.448, 0.410), "ETTm2": (0.272, 0.321), "ETTh1": (0.400, 0.424), "ETTh2": (0.341, 0.379), "electricity": (0.233, 0.320), "weather": (0.242, 0.267)}),
    ("Moirai base", {"ETTm1": (0.382, 0.388), "ETTm2": (0.276, 0.320), "ETTh1": (0.434, 0.439), "ETTh2": (0.346, 0.382), "electricity": (0.188, 0.274), "weather": (0.238, 0.261)}),
    ("Moirai large", {"ETTm1": (0.390, 0.389), "ETTm2": (0.317, 0.366), "ETTh1": (0.510, 0.469), "ETTh2": (0.354, 0.377), "electricity": (0.188, 0.273), "weather": (0.260, 0.275)}),
    ("Chronos small", {"ETTm1": (0.640, 0.500), "ETTm2": (0.310, 0.350), "ETTh1": (0.545, 0.472), "ETTh2": (0.424, 0.430), "electricity": (0.220, 0.284), "weather": (0.300, 0.318)}),
    ("Chronos base", {"ETTm1": (0.646, 0.500), "ETTm2": (0.295, 0.338), "ETTh1": (0.591, 0.468), "ETTh2": (0.406, 0.411), "electricity": (0.215, 0.279), "weather": (0.293, 0.315)}),
    ("Chronos large", {"ETTm1": (0.556, 0.465), "ETTm2": (0.300, 0.341), "ETTh1": (0.589, 0.466), "ETTh2": (0.455, 0.427), "electricity": (0.204, 0.274), "weather": (0.279, 0.306)}),
    ("Time-MoE small", {"ETTm1": (0.394, 0.416), "ETTm2": (0.316, 0.361), "ETTh1": (0.400, 0.424), "ETTh2": (0.367, 0.404), "weather": (0.266, 0.297)}),
    ("Time-MoE base", {"ETTm1": (0.376, 0.406), "ETTm2": (0.349, 0.380), "ETTh1": (0.394, 0.420), "ETTh2": (0.405, 0.415), "weather": (0.270, 0.300)}),
    ("MOMENT", {"ETTm1": (0.670, 0.537), "ETTm2": (0.316, 0.371), "ETTh1": (0.684, 0.566), "ETTh2": (0.362, 0.410), "electricity": (0.765, 0.687), "weather": (0.294, 0.326)}),
    ("Timer 28B", {"ETTm1": (0.487, 0.457), "ETTm2": (0.328, 0.347), "ETTh1": (0.444, 0.457), "ETTh2": (0.358, 0.407), "weather": (0.304, 0.331)}),
    ("TimesFM", {"ETTm1": (0.433, 0.419), "ETTh1": (0.473, 0.444), "ETTh2": (0.392, 0.406)}),
]


def load_cells(d, tag):
    """-> {(dataset, H): (mse, mae, complete)}; origin-range parts are merged by se/ae sums."""
    parts = {}
    for f in glob.glob(d + "/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if r.get("model") != tag or r.get("split", "test") != "test" or "dataset" not in r or r.get("mse") != r.get("mse"):
            continue
        parts.setdefault((r["dataset"], r["pred_len"]), []).append(r)
    out = {}
    for k, rs in parts.items():
        full = [r for r in rs if not r.get("origin_range")]
        if full:
            r = full[0]; out[k] = (r["mse"], r["mae"], True); continue
        rngs = sorted(r["origin_range"] for r in rs)
        se = sum(r["se"] for r in rs); ae = sum(r["ae"] for r in rs); cnt = sum(r["cnt"] for r in rs)
        cover = rngs[0].startswith("0:") and all(rngs[i].split(":")[1] == rngs[i + 1].split(":")[0] for i in range(len(rngs) - 1))
        out[k] = (se / cnt, ae / cnt, cover and len(rngs) >= 2)
    return out


def dataset_avgs(cells):
    """-> {dataset: (mse, mae, complete)} over the 4 horizons (None when a horizon is missing)."""
    out = {}
    for ds in DS:
        v = [cells.get((ds, h)) for h in HS]
        if all(v):
            out[ds] = (sum(x[0] for x in v) / 4, sum(x[1] for x in v) / 4, all(x[2] for x in v))
        else:
            out[ds] = None
    return out


def fmt(v, best, partial=False):
    s = f"{v:.3f}"
    if best: s = f"**{s}**"
    return s + ("†" if partial else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--dir", default="/nyx-storage1/hanliu/wm4ts/lsf")
    a = ap.parse_args()
    names = a.names or [f"Ours ({t})" for t in a.tags]
    ours = [(n, dataset_avgs(load_cells(a.dir, t))) for n, t in zip(names, a.tags)]

    rows = []  # (name, {ds: (mse, mae, partial)}, mean_mse, mean_mae)
    for n, av in ours:
        cells = {ds: (v[0], v[1], not v[2]) for ds, v in av.items() if v}
        rows.append((n, cells, True))
    for n, b in BASELINES:
        rows.append((n, {ds: (v[0], v[1], False) for ds, v in b.items()}, False))
    # best per column (over rows that have the value)
    best_mse = {ds: min(r[1][ds][0] for r in rows if ds in r[1]) for ds in DS}
    best_mae = {ds: min(r[1][ds][1] for r in rows if ds in r[1]) for ds in DS}
    means = {}
    for n, cells, _ in rows:
        if all(ds in cells for ds in DS):
            means[n] = (sum(cells[ds][0] for ds in DS) / 6, sum(cells[ds][1] for ds in DS) / 6)
    bm = min(v[0] for v in means.values()); ba = min(v[1] for v in means.values())

    print("| Model | " + " | ".join(DS) + " | Avg |")
    print("|---|" + "---:|" * 7)
    for n, cells, is_ours in rows:
        cs = []
        for ds in DS:
            if ds not in cells:
                cs.append("–"); continue
            m, e, p = cells[ds]
            cs.append(f"{fmt(m, abs(m - best_mse[ds]) < 5e-4, p)}/{fmt(e, abs(e - best_mae[ds]) < 5e-4, p)}")
        mean = means.get(n)
        ms = "–" if mean is None else f"{fmt(mean[0], abs(mean[0] - bm) < 5e-4)}/{fmt(mean[1], abs(mean[1] - ba) < 5e-4)}"
        label = f"**{n}**" if is_ours else n
        print(f"| {label} | " + " | ".join(cs) + f" | {ms} |")
    print("\n† = electricity origin ranges not yet complete for that cell.\n")

    # per-horizon detail for the first tag
    cells = load_cells(a.dir, a.tags[0])
    print(f"Per-horizon MSE/MAE for {names[0]}:\n\n| dataset | " + " | ".join(f"H{h}" for h in HS) + " | avg |")
    print("|---|" + "---:|" * 5)
    for ds in DS:
        v = [cells.get((ds, h)) for h in HS]
        avg = f"{sum(x[0] for x in v) / 4:.3f}/{sum(x[1] for x in v) / 4:.3f}" if all(v) else "–"
        print(f"| {ds} | " + " | ".join("–" if x is None else f"{x[0]:.3f}/{x[1]:.3f}" + ("" if x[2] else "†") for x in v) + f" | {avg} |")


if __name__ == "__main__":
    main()
