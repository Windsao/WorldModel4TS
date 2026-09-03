"""Matched-mask factorization: canonical region labels, matched masks, subset bookkeeping.

Fixes the confound documented in VIDEO_VISIONTS_MATCHED_MASK_EXPERIMENTS.md section 1.1.
The previous audit derived the RENDERER width from the mask name (`cols or 3`), so random
masks were scored on an 11-context/3-future image while `right_10` used 4-context/10-future.
78.6% of the reported random-mask metric was interpolation inside observed history.

Here the geometry is FIXED at 4 context / 10 future columns for every mask, the image is
rendered once per batch, and only the mask changes. Region membership is always read from the
canonical patch map, never inferred from the mask name.
"""

import hashlib

import numpy as np
import torch

GRID, PATCH, NTUB = 14, 16, 8
CTX_COLS, FUT_COLS = 4, 10
N_SPATIAL = GRID * GRID                       # 196
N_CTX = CTX_COLS * GRID                       # 56
N_FUT = FUT_COLS * GRID                       # 140


def canonical_is_future(grid=GRID, ctx_cols=CTX_COLS):
    """[196] bool. Canonical column 0..3 = context, 4..13 = future."""
    m = np.zeros((grid, grid), dtype=bool)
    m[:, ctx_cols:] = True
    return m.reshape(-1)


def canonical_column(grid=GRID):
    """[196] int: canonical patch index -> its column."""
    return np.tile(np.arange(grid), (grid, 1)).reshape(-1)


MATCHED_MASKS = (
    ["right_future_100", "global_random_140", "global_random_147", "context_random_42"]
    + [f"future_random_{k}" for k in (35, 70, 105, 140)]
    + [f"future_block_{c}" for c in (2, 5, 8, 10)]
)
RANDOM_MASKS = {"global_random_140", "global_random_147", "context_random_42",
                "future_random_35", "future_random_70", "future_random_105"}


def make_matched_mask(name, seed=0, grid=GRID):
    """Spatial mask [196] bool under the FIXED 4/10 geometry."""
    fut = canonical_is_future(grid)
    col = canonical_column(grid)
    sp = np.zeros(N_SPATIAL, dtype=bool)
    rng = np.random.default_rng(seed)
    if name == "right_future_100":
        sp[fut] = True
    elif name.startswith("global_random_"):
        k = int(name.rsplit("_", 1)[1])
        sp[np.sort(rng.choice(N_SPATIAL, k, replace=False))] = True
    elif name.startswith("future_random_"):
        k = int(name.rsplit("_", 1)[1])
        idx = np.where(fut)[0]
        sp[np.sort(rng.choice(idx, k, replace=False))] = True
    elif name.startswith("future_block_"):
        c = int(name.rsplit("_", 1)[1])
        sp[fut & (col < CTX_COLS + c)] = True         # c columns nearest the context boundary
    elif name == "context_random_42":
        idx = np.where(~fut)[0]
        sp[np.sort(rng.choice(idx, 42, replace=False))] = True
    else:
        raise ValueError(name)
    return sp


def expected_counts(name):
    """(masked_context, masked_future) that a mask must produce."""
    if name == "right_future_100":
        return 0, N_FUT
    if name.startswith("future_random_"):
        return 0, int(name.rsplit("_", 1)[1])
    if name.startswith("future_block_"):
        return 0, GRID * int(name.rsplit("_", 1)[1])
    if name == "context_random_42":
        return 42, 0
    return None, None                                  # global_random: split is stochastic


def to_tube(sp, B, device, n_tubelets=NTUB):
    t = torch.as_tensor(sp, device=device, dtype=torch.bool).view(1, -1).expand(B, -1)
    return t.unsqueeze(1).expand(B, n_tubelets, -1).reshape(B, -1).contiguous()


def mask_hash(sp):
    return hashlib.sha1(np.asarray(sp).tobytes()).hexdigest()[:16]


def tensor_hash(t):
    return hashlib.sha1(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def token_region_labels(sp, device, n_tubelets=NTUB):
    """For the MASKED tokens (in bool_masked_pos order): is_future, canonical column."""
    fut = torch.as_tensor(canonical_is_future(), device=device)
    col = torch.as_tensor(canonical_column(), device=device)
    m = torch.as_tensor(sp, device=device, dtype=torch.bool)
    f = fut.unsqueeze(0).expand(n_tubelets, -1).reshape(-1)
    c = col.unsqueeze(0).expand(n_tubelets, -1).reshape(-1)
    mm = m.unsqueeze(0).expand(n_tubelets, -1).reshape(-1)
    return f[mm], c[mm]


def subset_sums(err_sq, tgt_sq, is_future, col, std_ok):
    """Elementwise sums/counts per subset (spec section 3.1).

    err_sq / tgt_sq : [B, n_masked, D] squared error and squared native target.
    Sums and counts are returned so batches recombine exactly; never average rounded MSEs.

    Sums are accumulated in float64. A float32 reduction over the ~5e6 elements of one batch
    loses about 3e-8 relative precision, which is enough to break the exact subset-
    recombination check the spec requires (section 3.1).
    """
    err_sq = err_sq.double()
    tgt_sq = tgt_sq.double()
    out = {}

    def add(key, sel):
        if sel is None:
            e, t, n = err_sq.sum(), tgt_sq.sum(), err_sq.numel()
        else:
            if sel.sum() == 0:
                out[key] = {"err_sum": 0.0, "tgt_sum": 0.0, "n": 0}
                return
            e, t = err_sq[:, sel].sum(), tgt_sq[:, sel].sum()
            n = int(sel.sum()) * err_sq.shape[0] * err_sq.shape[2]
        out[key] = {"err_sum": float(e), "tgt_sum": float(t), "n": int(n)}

    add("all", None)
    add("context", ~is_future)
    add("future", is_future)
    add("future_nondegenerate", is_future & std_ok)
    add("future_degenerate", is_future & (~std_ok))
    for c in range(CTX_COLS, GRID):
        add(f"future_col_{c}", is_future & (col == c))
    return out


def merge_sums(a, b):
    for k, v in b.items():
        if k not in a:
            a[k] = {"err_sum": 0.0, "tgt_sum": 0.0, "n": 0}
        for f in ("err_sum", "tgt_sum", "n"):
            a[k][f] += v[f]
    return a


def finalize(sums):
    """sums -> MSE_PT, MSE_ZERO and native_ratio_zero per subset."""
    out = {}
    for k, v in sums.items():
        if v["n"] == 0:
            continue
        mse_pt = v["err_sum"] / v["n"]
        mse_zero = v["tgt_sum"] / v["n"]
        out[k] = {"n": v["n"], "mse_pt": round(mse_pt, 8), "mse_zero": round(mse_zero, 8),
                  "native_ratio_zero": round(mse_pt / (mse_zero + 1e-12), 6)}
    return out
