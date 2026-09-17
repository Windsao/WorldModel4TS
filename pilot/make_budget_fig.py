"""Figure 1 of the paper: the budget curve under the two objectives.

Left panel: VideoMAE's native masked-pixel objective on rendered series. Its own held-out loss falls
monotonically while downstream forecast error rises monotonically over the same range.
Right panel: the same backbone and corpus under the forecast-aligned value-space objective; the
curve is monotonically improving at every horizon.

usage: python pilot/make_budget_fig.py [out.pdf]
Numbers: VIDEO_TS_RESCUE_RESULTS.md 1.10-1.13 (see PAPER_MATERIAL.md 7-8).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# pixel objective, 16-frame clips
PIX_STEPS = [5_000, 20_000, 60_000, 176_000]
PIX_LOSS = [0.0366, 0.0333, 0.0329, 0.0312]          # held-out pixel loss
DOWN_STEPS = [5_000, 20_000, 176_000]                 # no 60k downstream eval
DOWN_MSE = [0.339, 0.344, 0.381]                      # 5-dataset mean over 3 horizons

# forecast-aligned objective, 32-frame clips: 5-dataset mean MSE per horizon
AL_STEPS = [5_000, 20_000, 60_000]
ALIGNED = {96: [0.250, 0.245, 0.242], 336: [0.337, 0.330, 0.329], 720: [0.384, 0.379, 0.377]}

BLUE, RED = "#1f4e79", "#b03a2e"


def main(out="paper/figures/budget_curve.pdf"):
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.35))

    a.plot(PIX_STEPS, PIX_LOSS, "o-", color=BLUE, ms=3.5, lw=1.2, label="held-out pixel loss")
    a.set_xscale("log")
    a.set_xlabel("continual-pretraining steps")
    a.set_ylabel("held-out pixel loss", color=BLUE)
    a.tick_params(axis="y", labelcolor=BLUE)
    a.set_title("(a) native masked-pixel objective", fontsize=8)
    a2 = a.twinx()
    a2.plot(DOWN_STEPS, DOWN_MSE, "s--", color=RED, ms=3.5, lw=1.2, label="forecast MSE")
    a2.set_ylabel("forecast MSE", color=RED)
    a2.tick_params(axis="y", labelcolor=RED)
    a.annotate("objective improves", (20_000, 0.0333), xytext=(24_000, 0.0355),
               color=BLUE, fontsize=7)
    a2.annotate("forecasting degrades", (176_000, 0.381), xytext=(9_000, 0.374),
                color=RED, fontsize=7)

    for h, style in zip([96, 336, 720], ["o-", "s-", "^-"]):
        b.plot(AL_STEPS, ALIGNED[h], style, ms=3.5, lw=1.2, label=f"$H$={h}")
    b.set_xscale("log")
    b.set_xlabel("continual-pretraining steps")
    b.set_ylabel("5-dataset mean MSE")
    b.set_title("(b) forecast-aligned value-space objective", fontsize=8)
    b.legend(frameon=False, fontsize=7, loc="center right")

    for ax in (a, a2, b):
        ax.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
