"""Shared infrastructure for the autonomous zero-shot loop (ZEROSHOT_LOOP_ENGINEERING_PROMPT).

Exact-pair evaluator, equal-dataset Q, context-only priors, the bounded safe-residual
wrapper, and a paired moving-block bootstrap. Everything here is strict zero-shot: no
function may read a value at or after the forecast origin except to compute a metric.
"""

import hashlib, json, math, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_field as RF

DS6 = ("ETTh1", "ETTh2", "ETTm2", "electricity", "traffic", "solar")
HORIZON = {"ETTh1": 96, "ETTh2": 96, "ETTm2": 96, "electricity": 96, "traffic": 96, "solar": 144}
ROOT = "pilot/results_field/zeroshot_loop"


# ---------------------------------------------------------------- manifests
def historical_manifest(dataset, data_dir, max_ch=112, stride=32, n_max=512, seed=0,
                        split="audit"):
    """Chronological train-range split with an L+H purge; returns (X [n, L+H], pairs, meta).

    Every window lies inside the observed TRAIN range, so its pseudo-future is historical.
    """
    data, borders = RF.load_mv(dataset, data_dir, max_ch)
    P = RF.P
    L, H = 16 * P, HORIZON[dataset]
    span = L + H
    lo, hi = span, borders[0]
    n = hi - lo
    c1, c2 = lo + int(0.6 * n), lo + int(0.8 * n)
    rng = {"train": (lo, c1 - span), "val": (c1 + span, c2 - span), "audit": (c2 + span, hi)}[split]
    origins = np.arange(rng[0], rng[1], stride)
    M = data.shape[1]
    pairs = np.array([(o, c) for o in origins for c in range(M)], dtype=np.int64)
    if len(pairs) > n_max:
        k = np.random.default_rng(seed).choice(len(pairs), n_max, replace=False)
        k.sort(); pairs = pairs[k]
    idx = pairs[:, 0][:, None] + np.arange(-L, H)[None, :]
    X = data[idx, pairs[:, 1][:, None]].astype(np.float32)
    meta = {"dataset": dataset, "split": split, "P": P, "L": L, "H": H, "M": M,
            "stride": stride, "seed": seed, "n": int(len(pairs)),
            "n_origins": int(len(np.unique(pairs[:, 0]))),
            "pairs_sha1": hashlib.sha1(pairs.tobytes()).hexdigest()[:16],
            "range": [int(rng[0]), int(rng[1])]}
    return X, pairs, meta


# ---------------------------------------------------------------- context-only priors
def priors(ctx, P, H):
    """All deterministic, context-only. ctx [n, L] numpy."""
    n, L = ctx.shape
    G = L // P
    reps = (H + P - 1) // P
    smean = np.tile(ctx[:, L - G * P:].reshape(n, G, P).mean(1), (1, reps))[:, :H]
    snaive = np.tile(ctx[:, -P:], (1, reps))[:, :H]
    last = np.repeat(ctx[:, -1:], H, axis=1)
    recent = np.repeat(ctx[:, -P:].mean(1, keepdims=True), H, axis=1)
    out = {"smean": smean, "snaive": snaive, "last": last, "recent_mean": recent}
    # --- richer seasonal family (added after Stage E v2). On electricity/traffic/solar the
    # seasonal priors dominate the level priors by ~6x, so the blend needs seasonal VARIANTS to
    # spread weight over, not just level fallbacks it has to down-weight.
    for k in (2, 4, 8):
        if G >= k:
            out[f"smean{k}"] = np.tile(ctx[:, L - k * P:].reshape(n, k, P).mean(1), (1, reps))[:, :H]
    # level-adjusted seasonal: seasonal SHAPE re-based on the most recent level. Handles the
    # very common case where the shape is stable but the series has drifted.
    sh = smean - smean.mean(1, keepdims=True)
    out["smean_drift"] = sh + ctx[:, -P:].mean(1, keepdims=True)
    sh2 = out.get("smean4", smean)
    out["smean4_drift"] = (sh2 - sh2.mean(1, keepdims=True)) + ctx[:, -P:].mean(1, keepdims=True)
    return out


def _fallback(nm, pp):
    """Longest-window sibling of `nm` that exists in this (shorter) pseudo-origin context."""
    for k in ("smean8", "smean4", "smean2", "smean"):
        if nm.startswith(k[:5]) and k in pp:
            return k
    return "smean" if "smean" in pp else sorted(pp)[0]


def pseudo_origin_select(ctx, P, H, cand, n_pseudo=3, gap=None):
    """Pick the best prior per sample using ONLY rolling pseudo-origins inside the context.

    The k-th pseudo-origin ends at L - k*(H+gap); its pseudo-future is still inside ctx.
    Returns (name_per_sample, chosen [n,H]).
    """
    n, L = ctx.shape
    gap = P if gap is None else gap
    names = sorted(cand)
    err = np.zeros((n, len(names)))
    used = 0
    for k in range(1, n_pseudo + 1):
        end = L - k * (H + gap)
        # a pseudo-origin lives INSIDE the context, so its own context is necessarily
        # shorter than L. Require only enough whole periods for the seasonal priors.
        if end < 4 * P:
            break
        used += 1
        sub = ctx[:, :end]
        tgt = ctx[:, end:end + H]
        pp = priors(sub, P, H)
        for j, nm in enumerate(names):
            q = pp.get(nm)
            if q is None:
                q = pp[_fallback(nm, pp)]
            err[:, j] += ((q - tgt) ** 2).mean(1)
    if used == 0:
        pick = np.zeros(n, dtype=int)
    else:
        pick = err.argmin(1)
    full = priors(ctx, P, H)
    out = np.stack([full[names[p]][i] for i, p in enumerate(pick)])
    return np.array([names[p] for p in pick]), out, used


# ---------------------------------------------------------------- safe residual
def safe_residual(base, cand, ctx, P, H, alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
                  clip_k=1.0, n_pseudo=3, gap=None, cand_pseudo=None):
    """y = base + alpha * clip(cand - base).

    alpha is chosen per sample from internal pseudo-forecast errors only. `cand_pseudo`
    supplies the candidate's predictions at the same pseudo-origins; when it is None the
    method degenerates to alpha=0 (a safe fall-back to the prior), which is the declared
    behaviour rather than a silent failure.
    """
    n, L = ctx.shape
    scale = np.maximum(np.abs(np.diff(ctx, axis=1)).mean(1, keepdims=True), 1e-6)
    r = np.clip(cand - base, -clip_k * scale * math.sqrt(H), clip_k * scale * math.sqrt(H))
    if cand_pseudo is None:
        a = np.zeros((n, 1))
        return base + a * r, a[:, 0]
    err = np.zeros((n, len(alphas)))
    for (sub_base, sub_cand, tgt) in cand_pseudo:
        sc = np.maximum(np.abs(np.diff(ctx, axis=1)).mean(1, keepdims=True), 1e-6)
        rr = np.clip(sub_cand - sub_base, -clip_k * sc * math.sqrt(H), clip_k * sc * math.sqrt(H))
        for j, al in enumerate(alphas):
            err[:, j] += (((sub_base + al * rr) - tgt) ** 2).mean(1)
    pick = err.argmin(1)
    a = np.array([alphas[p] for p in pick])[:, None]
    return base + a * r, a[:, 0]


# ---------------------------------------------------------------- metrics
def per_pair_mse(pred, true):
    return ((pred - true) ** 2).mean(1)


def q_equal(num, den):
    ds = sorted(set(num) & set(den))
    return math.exp(sum(math.log(num[d] / den[d]) for d in ds) / len(ds))


def paired_bootstrap(err_a, err_b, origins, block, n_boot=5000, seed=20260902):
    rng = np.random.default_rng(seed)
    uo = np.unique(origins)
    pa = np.array([err_a[origins == o].mean() for o in uo])
    pb = np.array([err_b[origins == o].mean() for o in uo])
    n = len(uo); blk = max(1, min(block, n))
    out, wins = [], 0
    for _ in range(n_boot):
        st = rng.integers(0, n, size=int(np.ceil(n / blk)))
        idx = np.concatenate([(np.arange(s, s + blk) % n) for s in st])[:n]
        ra, rb = pa[idx].mean(), pb[idx].mean()
        out.append(ra / max(rb, 1e-12)); wins += int(ra < rb)
    q = np.sort(np.array(out))
    return {"lo": float(np.percentile(q, 2.5)), "hi": float(np.percentile(q, 97.5)),
            "p_better": wins / n_boot, "n_origins": int(n), "block": int(blk), "seed": seed}


def cfg_hash(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def ledger_append(rec, path=os.path.join(ROOT, "ledger.jsonl")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


# ------------------------------------------------------- pseudo-origin BLENDING
def pseudo_origin_blend(ctx, P, H, cand, n_pseudo=3, gap=None, rule="softmin",
                        dense=False, min_ctx_periods=4):
    """Model AVERAGING over priors instead of model SELECTION.

    Stage E showed argmin selection is WORSE than always using smean on 5 of 6 datasets:
    the pseudo-origin loss is a noisy estimate, and argmin is its highest-variance functional.
    A weighted average of the same priors, with the same information, is the standard fix.

    Weight rules, all parameter-free (no temperature to tune):
      softmin : w_p propto exp(-(L_p/L_min - 1))   -- scale-free; argmin is its zero-temp limit
      invloss : w_p propto L_min / L_p
      uniform : w_p = 1/|priors|                   -- control, ignores the pseudo-origins
    Returns (weights [n,n_priors], blended [n,H], names, n_pseudo_used).
    """
    n, L = ctx.shape
    gap = P if gap is None else gap
    names = sorted(cand)
    err = np.zeros((n, len(names)))
    used = 0
    # DENSE pseudo-origins: the sparse schedule (2-3 origins, spaced H+P apart) gives a loss
    # estimate so noisy that on electricity/traffic/solar the resulting weights were WORSE than
    # a fixed smean. Rolling the origin by one period instead yields ~5-12 estimates per sample.
    if dense:
        ends = [e for e in range(L - H, min_ctx_periods * P - 1, -P)]
    else:
        ends = []
        for k in range(1, n_pseudo + 1):
            e = L - k * (H + gap)
            if e < min_ctx_periods * P:
                break
            ends.append(e)
    for end in ends:
        used += 1
        pp = priors(ctx[:, :end], P, H)
        tgt = ctx[:, end:end + H]
        for j, nm in enumerate(names):
            # a pseudo-origin context is shorter, so long-window priors (e.g. smean8) may not
            # exist there. Score such a prior by its shortest available sibling rather than
            # dropping it -- dropping would silently change the candidate set per dataset.
            q = pp.get(nm)
            if q is None:
                q = pp[_fallback(nm, pp)]
            err[:, j] += ((q - tgt) ** 2).mean(1)
    full = priors(ctx, P, H)
    stack = np.stack([full[nm] for nm in names], 1)          # [n, n_priors, H]
    if used == 0 or rule == "uniform":
        w = np.full((n, len(names)), 1.0 / len(names))
    else:
        lo = err.min(1, keepdims=True) + 1e-12
        r = err / lo
        if rule == "softmin":
            w = np.exp(-(r - 1.0))
        elif rule == "invloss":
            w = 1.0 / (r + 1e-12)
        elif rule == "softmin_n":
            # evidence-scaled: the pseudo-origin loss is an average over `used` origins, so the
            # weight of evidence grows with `used`. Unit-temperature softmin is too flat when
            # one prior dominates (electricity/traffic/solar); this sharpens toward argmin as
            # evidence accumulates, and is still parameter-free.
            w = np.exp(-used * (r - 1.0))
        elif rule == "pow_n":
            w = np.power(np.maximum(r, 1e-12), -float(used))
        else:
            raise ValueError(rule)
        w /= w.sum(1, keepdims=True)
    return w, np.einsum("np,nph->nh", w, stack), names, used
