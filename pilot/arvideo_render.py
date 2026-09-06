"""Route A of VIDEO_TS_LITERATURE_AND_RESCUE_PLAN.md: render a series as a slowly moving
"skyline" (filled area chart) video, and decode such frames back to values.

Design rules (from the plan):
  * value -> POSITION (the height of the filled area), never intensity;
  * frame axis = period axis: period g is shown exactly at frame T_g = g*m + (m-1), and
    frames in between linearly interpolate the heights, so motion is slow and continuous;
  * the encoder side never sees the future: `render_prefix` renders periods 0..G_ctx-1 only,
    and the last prefix frame is exactly period G_ctx-1 (no partial interpolation towards
    period G_ctx, which would leak);
  * decoding is per column: the first foreground row from the top gives the height.

A CLOCK strip runs along the top of every frame: a small red square that slides right by
CLOCK_PX pixels per frame. Reading its position in a generated frame gives that frame's absolute
time index, so the continuation is aligned to the period axis without trusting any assumption
about how many prefix frames the video pipeline re-emits. It also measures whether the model
keeps a constant frame rate.

MAGI-1 reads exactly PREFIX_FRAMES=32 prefix frames at 24 fps, so with m=4 the prefix holds
8 periods; period k (k >= 8) of the continuation sits at absolute frame 4k+3.
"""
import numpy as np

S = 480                       # canvas side (pixels); latent 60x60 for MAGI's 8x VAE
M = 4                         # frames per period step
PREFIX_FRAMES = 32            # MAGI-1 hard-codes prefix_frame=32
ZMAX = 3.0                    # z range mapped onto the usable height band
H_LO, H_HI = 0.10, 0.90       # usable height band as a fraction of the chart area
BG = np.array([236, 236, 236], np.uint8)
FG = np.array([28, 58, 138], np.uint8)     # dark blue fill: R channel separates FG/BG cleanly
GRID_GRAY = 214
GRID_EVERY = S // 8
CLOCK_H = 32                  # clock strip height (rows 0..CLOCK_H-1); chart occupies rows below
CLOCK_SQ = 10                 # marker size
CLOCK_X0 = 6                  # marker x at t = 0
CLOCK_PX = 3                  # pixels per frame
CLOCK_RGB = np.array([215, 40, 40], np.uint8)
CHART_TOP = CLOCK_H           # first chart row


def period_frame(g, m=M):
    """absolute frame index at which period g is displayed exactly."""
    return g * m + (m - 1)


def z_to_height(z):
    """z (any shape) -> height fraction in [H_LO, H_HI]; 0 -> 0.5, +-ZMAX -> band edges."""
    return 0.5 + (np.clip(z, -ZMAX, ZMAX) / ZMAX) * (H_HI - 0.5)


def height_to_z(h):
    return (h - 0.5) / (H_HI - 0.5) * ZMAX


def height_to_row(h, s=S):
    """height fraction (of the chart area) -> boundary row (absolute). Rows >= boundary are filled."""
    ch = s - CHART_TOP
    return CHART_TOP + np.rint((1.0 - h) * (ch - 1)).astype(np.int64)


def row_to_height(row, s=S):
    ch = s - CHART_TOP
    return 1.0 - (row - CHART_TOP) / (ch - 1)


def phase_bins(P, s=S):
    """column ranges for the P phases: phase p owns columns [edges[p], edges[p+1])."""
    return np.rint(np.linspace(0, s, P + 1)).astype(np.int64)


def clock_x(t):
    return CLOCK_X0 + CLOCK_PX * int(t)


def _col_z(z_cols, s=S):
    """per-column z: linear interpolation between phase CENTRES (flat beyond the end centres),
    so the silhouette is a smooth curve rather than a staircase of building-like blocks."""
    P = len(z_cols)
    edges = phase_bins(P, s)
    centres = phase_centres(P, s)
    return np.interp(np.arange(s), centres, np.asarray(z_cols, np.float64))


def phase_centres(P, s=S):
    """integer centre column of every phase bin: the curve takes exactly the phase value there,
    so decoding samples that column (a smoothing window would mix neighbouring phases)."""
    edges = phase_bins(P, s)
    return edges[:-1] + (edges[1:] - edges[:-1]) // 2


def render_frame(z_cols, t, s=S):
    """z_cols [P], absolute frame index t -> uint8 image [s, s, 3]."""
    img = np.empty((s, s, 3), np.uint8)
    img[:] = BG
    img[CHART_TOP::GRID_EVERY, :, :] = GRID_GRAY          # faint horizontal grid lines (texture)
    img[CHART_TOP - 1, :, :] = GRID_GRAY                   # strip separator
    rows = height_to_row(z_to_height(_col_z(z_cols, s)), s)      # [s] boundary row per column
    rgrid = np.arange(s)[:, None]
    fill = rgrid >= rows[None, :]
    img[fill] = FG
    x = clock_x(t)
    if x + CLOCK_SQ <= s:
        y0 = (CLOCK_H - CLOCK_SQ) // 2
        img[y0:y0 + CLOCK_SQ, x:x + CLOCK_SQ, :] = CLOCK_RGB
    return img


def _interp_heights(Z, n_frames, m=M):
    """Z [G, P] -> per-frame z [n_frames, P] with period g exact at period_frame(g)."""
    G, P = Z.shape
    out = np.empty((n_frames, P), np.float64)
    for t in range(n_frames):
        if t <= period_frame(0, m):
            out[t] = Z[0]
            continue
        g = (t - (m - 1)) // m           # last period at or before t
        frac = ((t - (m - 1)) % m) / m
        out[t] = Z[G - 1] if g >= G - 1 else (1 - frac) * Z[g] + frac * Z[g + 1]
    return out


def render_video(Z, n_frames=None, m=M, s=S, t0=0):
    """Z [G, P] -> frames uint8 [n_frames, s, s, 3]; frame i carries clock time t0+i."""
    G = Z.shape[0]
    if n_frames is None:
        n_frames = period_frame(G - 1, m) + 1
    zf = _interp_heights(np.asarray(Z, np.float64), n_frames, m)
    return np.stack([render_frame(zf[t], t0 + t, s) for t in range(n_frames)])


def render_prefix(Z_ctx, m=M, s=S, prefix_frames=PREFIX_FRAMES):
    """Exactly `prefix_frames` frames covering periods 0..G_ctx-1 with G_ctx = prefix_frames//m.
    Raises if Z_ctx has a different number of periods (a silent mismatch would misalign time)."""
    G_ctx = prefix_frames // m
    if Z_ctx.shape[0] != G_ctx:
        raise ValueError(f"prefix needs {G_ctx} periods for {prefix_frames} frames at m={m}, got {Z_ctx.shape[0]}")
    vid = render_video(Z_ctx, n_frames=prefix_frames, m=m, s=s)
    assert vid.shape[0] == prefix_frames
    return vid


# ------------------------------------------------------------------ decoding
def decode_frame(img, P, soft=True, **_):
    """uint8 [s, s, 3] -> z [P]. Per column (chart rows only) COUNT the rows whose R value is
    below the FG/BG midpoint; the boundary row is bottom - count. A symmetric vertical blur does
    not change that count, isolated noise pixels barely do, grid lines (214) never cross the
    threshold, and a global brightness drift of +-60 is tolerated. The value of phase p is read
    at its integer centre column (see phase_centres)."""
    s = img.shape[0]
    r = img[CHART_TOP:, :, 0].astype(np.float32)
    thresh = (float(FG[0]) + float(BG[0])) / 2.0             # 132: robust to +-60 brightness drift
    n_dark = (r < thresh).sum(axis=0).astype(np.float64)      # [s] filled rows per column
    boundary = (s - n_dark)                                   # absolute boundary row (float)
    centres = phase_centres(P, s)
    return height_to_z(row_to_height(boundary[centres], s))


def decode_video(frames, P, **kw):
    return np.stack([decode_frame(f, P, **kw) for f in frames])


def decode_clock(img):
    """absolute time index read from the red marker in the clock strip (nan if not found).
    Red = high R, low G/B; the chart fill is blue and the background gray, so this is unambiguous."""
    strip = img[:CLOCK_H].astype(np.int32)
    red = (strip[..., 0] > 150) & (strip[..., 1] < 110) & (strip[..., 2] < 110)
    cols = red.sum(axis=0)
    if cols.max() < CLOCK_SQ // 2:
        return np.nan
    xs = np.nonzero(cols >= max(2, cols.max() // 2))[0]
    x_left = float(xs.min()) + 0.0
    x_center = float((xs.min() + xs.max()) / 2.0) - (CLOCK_SQ - 1) / 2.0     # -> left edge estimate
    x = 0.5 * (x_left + x_center)
    return (x - CLOCK_X0) / CLOCK_PX


def decode_clocks(frames):
    return np.array([decode_clock(f) for f in frames])


# ------------------------------------------------------------------ synthetic signals (G1)
def base_shape(P, rng, n_harm=2):
    t = np.arange(P) / P
    y = np.zeros(P)
    for k in range(1, n_harm + 1):
        y += rng.uniform(0.4, 1.0) / k * np.sin(2 * np.pi * k * t + rng.uniform(0, 2 * np.pi))
    return y / (np.abs(y).max() + 1e-9)


def synth_signal(kind, P, G, rng, G_ctx=PREFIX_FRAMES // M):
    """Z [G, P] for the four G1 probes, normalised so the CONTEXT periods have unit std.
      const      : same shape every period                     (drift test)
      ramp       : level rises linearly, +0.25 std per period  (linear extrapolation)
      sine_level : level follows sin with period 8 periods     (oscillation extrapolation)
      travel     : shape rotates by one eighth of a period per period (translational motion)
    """
    shape = base_shape(P, rng)
    Z = np.empty((G, P))
    for g in range(G):
        if kind == "const":
            Z[g] = shape
        elif kind == "ramp":
            Z[g] = shape + 0.25 * g
        elif kind == "sine_level":
            Z[g] = shape + 1.0 * np.sin(2 * np.pi * g / 8.0)
        elif kind == "travel":
            Z[g] = np.roll(shape, int(round(g * P / 8.0)))
        else:
            raise ValueError(kind)
    ctx = Z[:G_ctx]
    mu, sd = ctx.mean(), ctx.std() + 1e-9
    return (Z - mu) / sd


def baselines(Z_ctx, H):
    """copy-last (snaive analogue) and linear extrapolation of the last two periods."""
    last = Z_ctx[-1]
    copy = np.repeat(last[None], H, 0)
    slope = Z_ctx[-1] - Z_ctx[-2]
    lin = np.stack([last + (k + 1) * slope for k in range(H)])
    return {"copy_last": copy, "lin_extrap": lin}


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))
