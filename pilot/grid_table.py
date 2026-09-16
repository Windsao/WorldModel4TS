"""5-dataset x 3-horizon comparison table with origin-range parts merged (for pilot/par_grid.sh outputs).

usage: python pilot/grid_table.py "name=tag" ["name=tag" ...] [--dir /nyx-storage1/hanliu/wm4ts/lsf]
Prints, per horizon, per-dataset MSE and the 5-dataset mean; a cell shows '†' while its parts are incomplete."""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_bigtable import load_cells

DS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="name=tag")
    ap.add_argument("--dir", default="/nyx-storage1/hanliu/wm4ts/lsf")
    ap.add_argument("--horizons", default="96,336,720")
    a = ap.parse_args()
    arms = [x.split("=", 1) for x in a.arms]
    cells = {name: load_cells(a.dir, tag) for name, tag in arms}
    for H in [int(h) for h in a.horizons.split(",")]:
        print(f"\n**H={H}** | " + " | ".join(DS) + " | mean |")
        print("|---|" + "---:|" * (len(DS) + 1))
        for name, _ in arms:
            v = [cells[name].get((ds, H)) for ds in DS]
            txt = ["—" if x is None else f"{x[0]:.3f}" + ("" if x[2] else "†") for x in v]
            mean = f"{sum(x[0] for x in v) / len(v):.3f}" if all(v) else "—"
            print(f"| {name} | " + " | ".join(txt) + f" | {mean} |")


if __name__ == "__main__":
    main()
