"""Carrier-signal validation utilities (VIDEO_VISIONTS_CARRIER_SIGNAL_EXPERIMENTS.md).

Why this exists: a C2 phase-carrier native target is ALWAYS a nonzero sinusoid, so beating
zero logits (B0) only proves the decoder emitted a generic sinusoid. The carrier-aware
baseline is B1 -- the exact native target of a NEUTRAL (z=0) carrier -- which already has the
correct frequency and amplitude and differs from the truth only in phase. Any claim about
future-conditioned signal must be measured against B1 and against shuffled-context controls.

Also fixes two artifacts of the previous runner: the time-series decoder used only tubelet 0
(discarding 7/8 of the evidence), and the degeneracy selector used sample 0's mask for the
whole batch.
"""

import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F

import videomae_patch_utils as VP
import video_visionts_carriers as CA

GRID, PATCH, NF, NTUB = 14, 16, 16, 8
CTX_COLS, FUT_COLS = 4, 10
FREQ = 2
NORM_CONST = CA.NORM_CONST


# ---------------------------------------------------------------- token bookkeeping
def future_token_view(x):
    """[B, n_masked, ...] under right_future_100 -> [B, 8, 14, 10, ...].

    bool_masked_pos is flattened [tubelet, row, col] row-major, so selecting w>=CTX_COLS
    yields tubelet-major then row-major then future-column order.
    """
    B, n = x.shape[0], x.shape[1]
    assert n == NTUB * GRID * FUT_COLS, f"unexpected masked count {n}"
    return x.view(B, NTUB, GRID, FUT_COLS, *x.shape[2:])


def tile_from_logits(logits):
    """[B,n,1536] native cubes -> [B,n,16,16] after averaging the tubelet's 2 frames and RGB."""
    B, n, D = logits.shape
    return logits.view(B, n, 2, PATCH, PATCH, 3).mean(2).mean(-1)


def fourier_ab(tile, freq=FREQ):
    """[...,16,16] -> (a, b) correlation with the fixed sine/cosine bases along the x axis."""
    P_ = tile.shape[-1]
    x = torch.arange(P_, device=tile.device, dtype=tile.dtype)
    s = torch.sin(2 * math.pi * freq * x / P_).view(*([1] * (tile.dim() - 1)), P_)
    c = torch.cos(2 * math.pi * freq * x / P_).view(*([1] * (tile.dim() - 1)), P_)
    return (tile * s).sum((-1, -2)), (tile * c).sum((-1, -2))


# ---------------------------------------------------------------- decoders D0-D3
def decode(a, b, kind="D2"):
    """a, b: [B, 8, 14, 10]. Returns (z [B,14,10], rho [B,14,10])."""
    c = torch.complex(a.double(), b.double())
    r = c.abs()
    if kind == "D0":
        phi = torch.atan2(b[:, 0].double(), a[:, 0].double())
        z = (phi / (math.pi / 2)).clamp(-1, 1)
        return z.float(), torch.ones_like(z).float()
    csum = c.sum(1)
    rho = csum.abs() / (r.sum(1) + 1e-8)
    phi_raw = torch.atan2(csum.imag, csum.real)
    z_raw = (phi_raw / (math.pi / 2)).clamp(-1, 1)
    if kind == "D1":
        return z_raw.float(), rho.float()
    if kind == "D2":
        return (rho * z_raw).float(), rho.float()
    if kind == "D3":
        cu = c / (r + 1e-8)
        s = cu.sum(1)
        phi = torch.atan2(s.imag, s.real)
        return (phi / (math.pi / 2)).clamp(-1, 1).float(), rho.float()
    raise ValueError(kind)


# ---------------------------------------------------------------- codec oracles
def context_stats(ctx):
    mu = ctx.mean(1, keepdim=True)
    sd = ctx.std(1, keepdim=True, unbiased=False) + 1e-5
    return mu, sd


def znorm(x, mu, sd):
    return NORM_CONST * (x - mu) / sd


def future_grid(zf, P, clip=True):
    """[B,H] normalized future -> [B,14,10] scalar grid."""
    B, H = zf.shape
    n = H // P
    m = zf[:, :n * P].view(B, n, P).permute(0, 2, 1).unsqueeze(1)
    g = F.interpolate(m, size=(GRID, FUT_COLS), mode="bilinear", align_corners=False).squeeze(1)
    return g.clamp(-1, 1) if clip else g


def grid_to_series(g, P, H, mu, sd):
    B = g.shape[0]
    n = max(1, H // P)
    s = F.interpolate(g.unsqueeze(1), size=(P, n), mode="bilinear", align_corners=False).squeeze(1)
    flat = s.permute(0, 2, 1).reshape(B, -1)[:, :H]
    return (flat / NORM_CONST * sd + mu).unsqueeze(-1)


def codec_oracles(ctx, fut, P, H):
    """O0 identity, O1 resample-only, O2 resample+clip, O3 native round trip."""
    mu, sd = context_stats(ctx)
    zf = znorm(fut, mu, sd)
    o0 = (zf / NORM_CONST * sd + mu).unsqueeze(-1)
    g1 = future_grid(zf, P, clip=False)
    g2 = future_grid(zf, P, clip=True)
    o1 = grid_to_series(g1, P, H, mu, sd)
    o2 = grid_to_series(g2, P, H, mu, sd)
    clip_frac = float(((zf.abs() > 1).float().mean()))
    return dict(mu=mu, sd=sd, zf=zf, g2=g2, o0=o0, o1=o1, o2=o2, clip_frac=clip_frac)


def carrier_video(zgrid_full):
    """[B,14,14] scalar grid -> ImageNet-normalized 16-frame video."""
    img = CA.c2_tiles(zgrid_full)
    B = img.shape[0]
    im = img.clamp(0, 1).unsqueeze(1).unsqueeze(1).expand(B, NF, 3, 224, 224)
    return ((im - CA.IMN_MEAN.to(img.device).unsqueeze(1))
            / CA.IMN_STD.to(img.device).unsqueeze(1)).contiguous()


def native_future_targets(vid_full, bm):
    """-> (targets [B,n,1536], cube_std [B,n,3])."""
    tgt, cmean, cstd = VP.native_targets(vid_full)
    B = vid_full.shape[0]
    return tgt[bm].view(B, -1, 1536), cstd[bm].view(B, -1, 3)


def o3_from_targets(tgt, P, H, mu, sd):
    """Native round trip: exact targets -> all-tubelet phase -> series."""
    a, b = fourier_ab(tile_from_logits(tgt))
    a, b = future_token_view(a), future_token_view(b)
    z, _ = decode(a, b, "D1")
    return grid_to_series(z, P, H, mu, sd), z


# ---------------------------------------------------------------- baselines
def neutral_grid(zc_grid):
    """Context columns kept, future columns set to the declared neutral z=0."""
    g = zc_grid.clone()
    g[:, :, CTX_COLS:] = 0.0
    return g


def const_phase_grid(zc_grid, zc):
    g = zc_grid.clone()
    g[:, :, CTX_COLS:] = zc
    return g


def context_grid(ctx, P, mu, sd):
    B = ctx.shape[0]
    zc = znorm(ctx, mu, sd)
    n = zc.shape[1] // P
    m = zc[:, :n * P].view(B, n, P).permute(0, 2, 1).unsqueeze(1)
    return F.interpolate(m, size=(GRID, CTX_COLS), mode="bilinear",
                         align_corners=False).squeeze(1).clamp(-1, 1)


def full_grid(ctx_g, fut_g):
    return torch.cat([ctx_g, fut_g], -1)


# ---------------------------------------------------------------- metrics
def sums(err, n=None):
    e = err.double()
    return {"sum": float(e.sum()), "n": int(e.numel() if n is None else n)}


def mse_from(s):
    return s["sum"] / max(s["n"], 1)


def pearson(x, y):
    x = x.flatten().double(); y = y.flatten().double()
    xs, ys = x - x.mean(), y - y.mean()
    d = float((xs.norm() * ys.norm()))
    return 0.0 if d < 1e-12 else float((xs * ys).sum() / d)


def spearman(x, y):
    def rank(v):
        idx = torch.argsort(v.flatten().double())
        r = torch.empty_like(idx, dtype=torch.float64)
        r[idx] = torch.arange(len(idx), dtype=torch.float64, device=v.device)
        return r
    return pearson(rank(x), rank(y))


def phase_cosine(z_pred, z_true):
    return float(torch.cos((z_pred.double() - z_true.double()) * (math.pi / 2)).mean())


def cfg_hash(d):
    return hashlib.sha1(repr(sorted(d.items())).encode()).hexdigest()[:16]


def block_bootstrap_paired(err_a, err_b, origins, block, n_boot=5000, seed=20260902):
    """Paired moving-block bootstrap over unique forecast origins -> CI for mean(a)/mean(b)."""
    rng = np.random.default_rng(seed)
    uo = np.unique(origins)
    per_a = np.array([err_a[origins == o].mean() for o in uo])
    per_b = np.array([err_b[origins == o].mean() for o in uo])
    n = len(uo); blk = max(1, min(block, n))
    rat, imp = [], 0
    for _ in range(n_boot):
        nb = int(np.ceil(n / blk))
        st = rng.integers(0, n, size=nb)
        idx = np.concatenate([(np.arange(s, s + blk) % n) for s in st])[:n]
        ra, rb = per_a[idx].mean(), per_b[idx].mean()
        rat.append(ra / max(rb, 1e-12))
        imp += int(ra < rb)
    q = np.sort(np.array(rat))
    return {"ratio_lo": float(np.percentile(q, 2.5)), "ratio_hi": float(np.percentile(q, 97.5)),
            "p_improve": imp / n_boot, "n_origins": int(n), "block": int(blk),
            "seed": seed, "n_boot": n_boot}
