"""Stage M5: patch-normalization-invariant carriers (spec section 8).

`norm_pix_loss=True` subtracts each cube's mean and divides by its std, so an intensity
renderer can have its entire numeric level erased -- 14-21% of future cube/channel targets in
the audit have std near 1e-6. These carriers encode the scalar in patch-local GEOMETRY, which
survives that normalization, and decode straight from the normalized logits so no unknown raw
mean/std ever has to be estimated.

Coarse canonical grid: the normalized period matrix is resampled to 14x14 scalars
(4 context columns, 10 future columns, 14 phase rows) so each cell is exactly one 16x16 patch.
Context and future grids are resampled SEPARATELY so no future value enters a context cell.
"""

import numpy as np
import torch
import torch.nn.functional as F

GRID, PATCH, IMG, NF = 14, 16, 224, 16
CTX_COLS, FUT_COLS = 4, 10
IMN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
NORM_CONST = 0.4


def scalar_grid(ctx, fut, P):
    """ctx [B,L], fut [B,H] normalized -> scalar grid [B,14,14] (cols 0..3 ctx, 4..13 fut)."""
    B = ctx.shape[0]
    def mat(x):
        n = x.shape[1] // P
        return x[:, :n * P].view(B, n, P).permute(0, 2, 1).unsqueeze(1)     # [B,1,P,n]
    c = F.interpolate(mat(ctx), size=(GRID, CTX_COLS), mode="bilinear", align_corners=False)
    f = F.interpolate(mat(fut), size=(GRID, FUT_COLS), mode="bilinear", align_corners=False)
    return torch.cat([c, f], -1).squeeze(1).clamp(-1, 1)                     # [B,14,14]


def c1_tiles(z):
    """C1 soft edge-position carrier. z [B,14,14] in [-1,1] -> image [B,224,224]."""
    B = z.shape[0]
    y = torch.arange(PATCH, device=z.device, dtype=z.dtype).view(1, 1, 1, PATCH, 1)
    h = 1 + 14 * (z + 1) / 2                                                 # [B,14,14]
    t = torch.sigmoid((h.unsqueeze(-1).unsqueeze(-1) - y) / 0.75)            # [B,14,14,16,1]
    t = t.expand(B, GRID, GRID, PATCH, PATCH)
    return t.permute(0, 1, 3, 2, 4).reshape(B, IMG, IMG)


def c1_decode(tile_norm):
    """[B,n,16,16] patch-normalized predictions -> z in [-1,1].

    Patch normalization is affine (subtract mean, divide std), so the raw zero crossing is
    NOT affine-invariant and degenerates where the sigmoid saturates. Instead locate the
    MIDPOINT crossing of the row profile, which is invariant to any affine rescaling, and
    fall back to the declared neutral z=0 for a flat profile (e.g. zero logits).
    """
    prof = tile_norm.mean(-1)                                                # [B,n,16]
    B, n, Ph = prof.shape
    lo = prof.min(-1, keepdim=True).values
    hi = prof.max(-1, keepdim=True).values
    flat = (hi - lo) < 1e-6
    mid = (hi + lo) / 2
    d = prof - mid                                                           # decreasing in y
    sign_pos = d > 0
    idx = sign_pos.float().sum(-1).clamp(1, Ph - 1)
    i0 = (idx - 1).long()
    a = torch.gather(d, -1, i0.unsqueeze(-1)).squeeze(-1)
    b = torch.gather(d, -1, (i0 + 1).clamp(max=Ph - 1).unsqueeze(-1)).squeeze(-1)
    frac = (a / (a - b + 1e-9)).clamp(0, 1)
    hpos = i0.to(prof.dtype) + frac
    z = (2 * (hpos - 1) / 14 - 1).clamp(-1, 1)
    return torch.where(flat.squeeze(-1), torch.zeros_like(z), z)


def c2_tiles(z, freq=2):
    """C2 phase carrier. phi = (pi/2) z, no wrap ambiguity over z in [-1,1]."""
    B = z.shape[0]
    x = torch.arange(PATCH, device=z.device, dtype=z.dtype).view(1, 1, 1, 1, PATCH)
    phi = (math_pi() / 2) * z
    t = 0.5 + 0.45 * torch.sin(2 * math_pi() * freq * x / PATCH + phi.unsqueeze(-1).unsqueeze(-1))
    t = t.expand(B, GRID, GRID, PATCH, PATCH)
    return t.permute(0, 1, 3, 2, 4).reshape(B, IMG, IMG)


def math_pi():
    return float(np.pi)


def c2_decode(tile_norm, freq=2):
    """Correlate with fixed sine/cosine bases on the normalized logits."""
    P_ = tile_norm.shape[-1]
    x = torch.arange(P_, device=tile_norm.device, dtype=tile_norm.dtype)
    s = torch.sin(2 * math_pi() * freq * x / P_).view(1, 1, 1, P_)
    c = torch.cos(2 * math_pi() * freq * x / P_).view(1, 1, 1, P_)
    a = (tile_norm * s).sum((-1, -2))
    b = (tile_norm * c).sum((-1, -2))
    phi = torch.atan2(b, a)
    return (phi / (math_pi() / 2)).clamp(-1, 1)


CARRIERS = {"C1": (c1_tiles, c1_decode), "C2": (c2_tiles, c2_decode)}


def render_carrier(ctx, fut, P, carrier, blank_future=True):
    """Returns (video_input with the future region blanked, video_full, scalar grid)."""
    mu = ctx.mean(1, keepdim=True)
    sd = ctx.std(1, keepdim=True, unbiased=False) + 1e-5
    zc = (NORM_CONST * (ctx - mu) / sd)
    zf = (NORM_CONST * (fut - mu) / sd)
    z = scalar_grid(zc, zf, P)
    tiles = CARRIERS[carrier][0]
    full = tiles(z)
    zb = z.clone(); zb[:, :, CTX_COLS:] = 0.0                                # neutral blank
    inp = tiles(zb)
    def to_vid(g):
        B = g.shape[0]
        im = g.clamp(0, 1).unsqueeze(1).unsqueeze(1).expand(B, NF, 3, IMG, IMG)
        return ((im - IMN_MEAN.to(g.device).unsqueeze(1))
                / IMN_STD.to(g.device).unsqueeze(1)).contiguous()
    return to_vid(inp), to_vid(full), z, mu, sd


def decode_future(logits_norm, carrier, n_masked_per_sample=None):
    """[B,n,1536] normalized logits over the masked (future) tokens -> z [B,n]."""
    B, n, D = logits_norm.shape
    t = logits_norm.view(B, n, 2, PATCH, PATCH, 3).mean(2).mean(-1)          # tubelet+RGB mean
    return CARRIERS[carrier][1](t)


def z_to_series(zf, P, H, mu, sd):
    """Future scalar columns [B,14,10] -> time series [B,H,1] (period-major)."""
    B = zf.shape[0]
    n = max(1, H // P)
    g = F.interpolate(zf.unsqueeze(1), size=(P, n), mode="bilinear", align_corners=False).squeeze(1)
    flat = g.permute(0, 2, 1).reshape(B, -1)[:, :H]
    return (flat / NORM_CONST * sd + mu).unsqueeze(-1)
