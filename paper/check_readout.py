"""Numerical check that the paper's Equations 1-2 invert each other exactly.

Renders every representable column height with the paper's drawing rule (boundary row included in
the fill, grid lines on the background only) and decodes it with the paper's read-out. Reports the
maximum error in z units. No GPU, no data, runs in a second.

usage: python paper/check_readout.py
"""
import numpy as np

IMG, FILL, BG, GRID, GRID_EVERY = 224, 0.12, 0.92, 0.80, 28
ZMAX, BAND = 3.0, 0.40


def render(h):
    """height in [0.1,0.9] -> one rendered column of gray levels, top row first."""
    col = np.full(IMG, BG)
    rows = np.arange(IMG)
    b = int(round((1.0 - h) * (IMG - 1)))
    fill = rows >= b                                   # boundary row included
    col[(rows % GRID_EVERY == 0) & ~fill] = GRID       # grid on background only
    col[fill] = FILL
    return col


def decode(col):
    soft = np.clip((GRID - col) / (GRID - FILL), 0.0, 1.0)
    return 1.0 - (IMG - soft.sum()) / (IMG - 1)


def main():
    zs = np.linspace(-ZMAX, ZMAX, 2001)
    hs = 0.5 + BAND * zs / ZMAX
    err_z = np.array([(decode(render(h)) - h) * ZMAX / BAND for h in hs])
    print(f"max |z error|  {np.abs(err_z).max():.6f}")
    print(f"mean z error   {err_z.mean():+.6f}   (bias)")
    print(f"quantisation bound  {0.5 / (IMG - 1) * ZMAX / BAND:.6f}")
    # what a wrong fill convention would cost, for the record
    def render_excl(h):
        col = np.full(IMG, BG)
        rows = np.arange(IMG)
        b = int(round((1.0 - h) * (IMG - 1)))
        fill = rows > b
        col[(rows % GRID_EVERY == 0) & ~fill] = GRID
        col[fill] = FILL
        return col
    bias = np.array([(decode(render_excl(h)) - h) * ZMAX / BAND for h in hs]).mean()
    print(f"bias if the boundary row were excluded from the fill: {bias:+.4f} z")


if __name__ == "__main__":
    main()
