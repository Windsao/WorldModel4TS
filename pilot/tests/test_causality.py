"""
Causal-invariance test (PREPROCESSING_AND_KNOWN_ISSUES.md section 2.3).

For a forecast-shaped mask, rendering the same context with two different
futures must produce bitwise-identical VISIBLE tokens; otherwise future
information leaks into the encoder input. Also verifies the neutral-gray
variant used at evaluation time.

Run: python pilot/tests/test_causality.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from pretrain_vmae_ts import (to_gray, render_lc, render_scroll,
                              forecast_mask_lc, forecast_mask_scroll,
                              LC_STEPS, COLS, NF, SCROLL, IMG, PS, TT, GH)


def patchify(vid):
    """[16, 3, 224, 224] -> tokens [TT*GH*GH, ...] in HF (t h w) order."""
    v = vid.view(TT, 2, 3, GH, PS, GH, PS)
    v = v.permute(0, 3, 5, 1, 4, 6, 2).reshape(TT * GH * GH, -1)
    return v


def check(layout, hp, P, seed):
    rng = np.random.default_rng(seed)
    n_p = (NF - 1) * SCROLL + COLS if layout == "scroll" else LC_STEPS * COLS
    cp = n_p - hp
    ctx = rng.normal(0, 1, cp * P).astype(np.float32)
    fut_a = rng.normal(0, 1, hp * P).astype(np.float32)
    fut_b = rng.normal(5, 3, hp * P).astype(np.float32)   # wildly different
    fut_g = None                                          # gray placeholder

    render = render_scroll if layout == "scroll" else render_lc
    mask = (forecast_mask_scroll(hp) if layout == "scroll"
            else forecast_mask_lc(hp))

    toks = []
    for fut in (fut_a, fut_b, fut_g):
        if fut is None:
            g = to_gray(np.concatenate([ctx, np.zeros(hp * P, np.float32)]),
                        ctx_len=cp * P)
            g[cp * P:] = 0.5                              # eval-style gray
        else:
            g = to_gray(np.concatenate([ctx, fut]), ctx_len=cp * P)
        toks.append(patchify(render(g, P))[~mask])

    d_ab = (toks[0] - toks[1]).abs().max().item()
    d_ag = (toks[0] - toks[2]).abs().max().item()
    status = "OK " if d_ab == 0 and d_ag == 0 else "LEAK"
    print(f"[{status}] layout={layout:6s} hp={hp} P={P:3d} "
          f"|visible(A)-visible(B)|={d_ab:.2e}  |visible(A)-visible(gray)|={d_ag:.2e}")
    return d_ab == 0 and d_ag == 0


if __name__ == "__main__":
    ok = True
    for layout, hps in (("lc", (1, 4, 8)), ("scroll", (1, 4))):
        for hp in hps:
            for P in (24, 96, 144):
                ok &= check(layout, hp, P, seed=hp * 1000 + P)
    print("\nCAUSALITY TEST:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
