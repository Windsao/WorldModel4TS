"""Time-series -> video renderers for the VideoMAE input-alignment experiment.

Implements R0-R5 from PREPROCESSING_EXPERIMENTS.md. Every renderer:
  * consumes CONTEXT ONLY (never future values),
  * takes x [B, L, 1] and returns vid [B, 16, 3, 224, 224] plus (mu, sd),
  * applies ONE context-wide normalization shared by all 16 frames,
  * applies ImageNet channel normalization only after rendering,
  * is deterministic.

The original barcode renderer (R0) is reproduced here as the negative control; the
implementation in pilot/run_field.py:161 is left untouched and is the reference that
tests/test_preprocess_renderers.py checks R0 against numerically.
"""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F

IMG, NF = 224, 16
IMN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

NORMS = ("std_clip_3", "std_clip_2", "robust", "visionts_r04")


# ---------------------------------------------------------------- normalization
def normalize_context(x, norm="std_clip_3"):
    """x [B, L, 1] -> (z in [-1,1] [B,L], mu [B,1,1], sd [B,1,1]).

    One statistic per window, shared by every frame. mu/sd are always the plain
    mean/std so the output de-normalization path (pred * sd + mu) is identical
    across normalization variants, per PREPROCESSING_EXPERIMENTS.md section 4.
    """
    assert x.dim() == 3 and x.shape[-1] == 1, f"expected [B,L,1], got {tuple(x.shape)}"
    mu = x.mean(1, keepdim=True)
    sd = x.std(1, keepdim=True) + 1e-8
    if norm == "std_clip_3":
        z = ((x - mu) / (3 * sd)).clamp(-1, 1)
    elif norm == "std_clip_2":
        z = ((x - mu) / (2 * sd)).clamp(-1, 1)
    elif norm == "robust":
        med = x.median(1, keepdim=True).values
        q = torch.quantile(x, torch.tensor([0.25, 0.75], dtype=x.dtype, device=x.device), dim=1)
        iqr = (q[1] - q[0]).unsqueeze(1) + 1e-8
        z = ((x - med) / (3 * iqr)).clamp(-1, 1)
    elif norm == "visionts_r04":
        z = ((x - mu) / sd * 0.4).clamp(-1, 1)
    else:
        raise ValueError(f"unknown norm {norm}")
    return z[..., 0], mu, sd


def _to_video(gray):
    """gray [B, NF, H, W] in [0,1] -> ImageNet-normalized [B, NF, 3, 224, 224]."""
    B, T, H, W = gray.shape
    assert (H, W) == (IMG, IMG), f"renderer must emit {IMG}x{IMG}, got {H}x{W}"
    vid = gray.unsqueeze(2).expand(B, T, 3, IMG, IMG)
    vid = (vid - IMN_MEAN.to(gray.device)) / IMN_STD.to(gray.device)
    return vid.contiguous()


def _periods(z, P):
    """z [B, L] -> [B, G, P] using the last G*P samples (context only)."""
    B, L = z.shape
    G = L // P
    return z[:, L - G * P:].reshape(B, G, P), G


def _resample_frames(g, mode):
    """g [B, G, P] -> [B, NF, P] on the frame axis when G != NF."""
    B, G, P = g.shape
    if G == NF:
        return g
    return F.interpolate(g.permute(0, 2, 1), size=NF, mode=mode,
                         align_corners=False if mode == "linear" else None).permute(0, 2, 1)


# ---------------------------------------------------------------- R0 / R1 barcode
def _barcode(x, P, norm, interp):
    z, mu, sd = normalize_context(x, norm)
    g, G = _periods(z, P)
    g = _resample_frames(g, "linear") if G != NF else g
    gray01 = (g + 1) / 2                                   # [B,NF,P] in [0,1]
    fr = gray01.unsqueeze(2).expand(-1, -1, P, -1)         # rows = tiled phase
    B = fr.shape[0]
    kw = {} if interp == "nearest" else {"align_corners": False}
    img = F.interpolate(fr.reshape(B * NF, 1, P, P), size=(IMG, IMG), mode=interp, **kw)
    return _to_video(img.view(B, NF, IMG, IMG)), mu, sd


def barcode_nearest(x, P, norm="std_clip_3"):
    """R0: exact reproduction of the current univariate period renderer."""
    return _barcode(x, P, norm, "nearest")


def barcode_bilinear(x, P, norm="std_clip_3"):
    """R1: R0 with bilinear resizing -- isolates interpolation alone."""
    return _barcode(x, P, norm, "bilinear")


# ---------------------------------------------------------------- R2 period matrix
def period_matrix(x, P, norm="std_clip_3", rows=8):
    """R2: each frame is a rolling heatmap of the most recent `rows` periods.

    Row 0 = oldest, row -1 = newest. Early frames pad with the earliest available
    period, so the matrix slides smoothly as one period enters and one leaves.
    """
    z, mu, sd = normalize_context(x, norm)
    g, G = _periods(z, P)
    g = _resample_frames(g, "linear") if G != NF else g          # [B,NF,P]
    B = g.shape[0]
    idx = torch.arange(NF).unsqueeze(1) - torch.arange(rows - 1, -1, -1).unsqueeze(0)
    idx = idx.clamp(min=0).to(g.device)                          # [NF, rows] pad with period 0
    frames = g[:, idx.reshape(-1), :].reshape(B, NF, rows, P)    # [B,NF,rows,P]
    gray01 = (frames + 1) / 2
    img = F.interpolate(gray01.reshape(B * NF, 1, rows, P), size=(IMG, IMG),
                        mode="bilinear", align_corners=False)
    return _to_video(img.view(B, NF, IMG, IMG)), mu, sd


# ---------------------------------------------------------------- line drawing
def _draw_curves(curves, weights, margin=12, half_w=1.75):
    """Antialiased multi-curve line raster.

    curves  [B, K, P] values in [-1, 1]; curve 0 is drawn brightest.
    weights K intensities.
    Returns [B, IMG, IMG] in [0,1] on a neutral background.

    For each output column the curve spans the y-interval between its values at the
    column edges, so steep segments stay connected instead of breaking into dots.
    Coverage falls off linearly with distance from that interval, which antialiases
    the edges and keeps the value recoverable from CONTOUR POSITION rather than mean
    brightness (hypothesis H2).
    """
    B, K, P = curves.shape
    dev = curves.device
    usable = IMG - 2 * margin
    # sample each curve at the two edges of every output column
    pos = torch.linspace(-0.5, IMG - 0.5, IMG + 1, device=dev)          # column edges
    src = (pos / max(IMG - 1, 1) * (P - 1)).clamp(0, P - 1)
    lo_i = src.floor().long().clamp(0, P - 1)
    hi_i = src.ceil().long().clamp(0, P - 1)
    frac = (src - lo_i.to(src.dtype)).view(1, 1, -1)
    v = curves[:, :, lo_i] * (1 - frac) + curves[:, :, hi_i] * frac      # [B,K,IMG+1]
    # value -> row (y grows downward; +1 maps to the top margin)
    y = margin + (1 - v) / 2 * usable                                    # [B,K,IMG+1]
    y0, y1 = y[:, :, :-1], y[:, :, 1:]
    ylo = torch.minimum(y0, y1).unsqueeze(2)                             # [B,K,1,IMG]
    yhi = torch.maximum(y0, y1).unsqueeze(2)
    rows = torch.arange(IMG, device=dev).view(1, 1, IMG, 1).to(curves.dtype)
    dist = torch.clamp(ylo - rows, min=0) + torch.clamp(rows - yhi, min=0)
    cov = (half_w - dist).clamp(0, 1)                                    # [B,K,IMG,IMG]
    w = torch.as_tensor(weights, dtype=curves.dtype, device=dev).view(1, K, 1, 1)
    return (cov * w).amax(dim=1).clamp(0, 1)


def period_line(x, P, norm="std_clip_3"):
    """R3: each frame is an antialiased line plot of one period (primary candidate)."""
    z, mu, sd = normalize_context(x, norm)
    g, G = _periods(z, P)
    g = _resample_frames(g, "linear") if G != NF else g                 # [B,NF,P]
    B = g.shape[0]
    ink = _draw_curves(g.reshape(B * NF, 1, P), [1.0])                  # [B*NF,IMG,IMG]
    gray = 0.15 + 0.75 * ink                                            # neutral bg, bright line
    return _to_video(gray.view(B, NF, IMG, IMG)), mu, sd


def period_trails(x, P, norm="std_clip_3", trail=(1.0, 0.65, 0.40, 0.25)):
    """R4: current period plus the previous three, fading -- tests contour persistence.

    Frame f draws periods f, f-1, f-2, f-3 (clamped at 0), so no future period is ever
    visible in a frame.
    """
    z, mu, sd = normalize_context(x, norm)
    g, G = _periods(z, P)
    g = _resample_frames(g, "linear") if G != NF else g
    B, _, Pp = g.shape
    K = len(trail)
    idx = (torch.arange(NF).unsqueeze(1) - torch.arange(K).unsqueeze(0)).clamp(min=0).to(g.device)
    curves = g[:, idx.reshape(-1), :].reshape(B * NF, K, Pp)
    ink = _draw_curves(curves, list(trail))
    gray = 0.15 + 0.75 * ink
    return _to_video(gray.view(B, NF, IMG, IMG)), mu, sd


# ---------------------------------------------------------------- R5 recurrence
def recurrence(x, P, norm="std_clip_3", seg_periods=4, res=64, tau=0.25):
    """R5: soft recurrence plot R[i,j] = exp(-|z_i - z_j| / tau) on a rolling segment.

    2-D texture control. Recurrence is symmetric, so it partly obscures the direction
    of time -- included as a control, not an expected winner.
    """
    z, mu, sd = normalize_context(x, norm)
    g, G = _periods(z, P)
    g = _resample_frames(g, "linear") if G != NF else g                 # [B,NF,P]
    B = g.shape[0]
    idx = (torch.arange(NF).unsqueeze(1) - torch.arange(seg_periods - 1, -1, -1).unsqueeze(0))
    idx = idx.clamp(min=0).to(g.device)                                 # [NF, seg]
    seg = g[:, idx.reshape(-1), :].reshape(B * NF, seg_periods * g.shape[-1])
    seg = F.interpolate(seg.unsqueeze(1), size=res, mode="linear", align_corners=False)
    s = seg.squeeze(1).clamp(-1, 1)                                     # [B*NF, res]
    R = torch.exp(-(s.unsqueeze(2) - s.unsqueeze(1)).abs() / tau)       # [B*NF,res,res]
    img = F.interpolate(R.unsqueeze(1), size=(IMG, IMG), mode="bilinear", align_corners=False)
    return _to_video(img.view(B, NF, IMG, IMG)), mu, sd


RENDERERS = {
    "barcode_nearest": barcode_nearest,
    "barcode_bilinear": barcode_bilinear,
    "period_matrix": period_matrix,
    "period_line": period_line,
    "period_trails": period_trails,
    "recurrence": recurrence,
}


# ---------------------------------------------------------------- diagnostics
def input_stats(vid, tub_t=2, tub_s=16, const_eps=1e-3):
    """Input-alignment diagnostics from section 8. vid is ImageNet-normalized.

    These are descriptive only -- a renderer must never be selected on them alone.
    """
    v = (vid * IMN_STD.to(vid.device).unsqueeze(1) + IMN_MEAN.to(vid.device).unsqueeze(1))
    g = v[:, :, 0]                                                       # [B,T,H,W] back to [0,1]
    B, T, H, W = g.shape
    tb = g.reshape(B, T // tub_t, tub_t, H // tub_s, tub_s, W // tub_s, tub_s)
    tb = tb.permute(0, 1, 3, 5, 2, 4, 6).reshape(B, -1, tub_t * tub_s * tub_s)
    tstd = tb.std(-1)
    dx = (g[..., 1:] - g[..., :-1]).pow(2).mean()
    dy = (g[:, :, 1:] - g[:, :, :-1]).pow(2).mean()
    dt = (g[:, 1:] - g[:, :-1]).pow(2).mean()
    spatial = float(((dx + dy) / 2).sqrt())
    temporal = float(dt.sqrt())
    return {
        "const_tubelet_frac": round(float((tstd < const_eps).float().mean()), 4),
        "tubelet_pixel_std": round(float(tstd.mean()), 4),
        "spatial_grad_rms": round(spatial, 4),
        "temporal_diff_rms": round(temporal, 4),
        "temporal_over_spatial": round(temporal / (spatial + 1e-8), 4),
    }


def contact_sheet(vid_one, path, cols=4):
    """Save a 4x4 contact sheet of the 16 frames of ONE sample for visual audit."""
    v = vid_one * IMN_STD.to(vid_one.device) + IMN_MEAN.to(vid_one.device)
    g = v[:, 0].clamp(0, 1).detach().cpu().numpy()                      # [T,H,W]
    T = g.shape[0]
    rows = (T + cols - 1) // cols
    sheet = np.ones((rows * (IMG + 4), cols * (IMG + 4)), dtype=np.float32)
    for i in range(T):
        r, c = divmod(i, cols)
        sheet[r * (IMG + 4):r * (IMG + 4) + IMG, c * (IMG + 4):c * (IMG + 4) + IMG] = g[i]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray((sheet * 255).astype(np.uint8)).save(path)
    except ImportError:
        np.save(path.replace(".png", ".npy"), sheet)
    return path
