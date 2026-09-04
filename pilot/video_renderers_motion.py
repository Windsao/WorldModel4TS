"""Renderers that put real MOTION on the frame axis.

Every feature-route experiment in this repository so far fed VideoMAE a STATIC image repeated
over all 16 frames. That wastes the only thing the backbone was pretrained for. The frame-order
audit measured an 11% cost for permuting frames (CI excluding 1), so the model does read the
frame axis -- it was simply never given a meaningful one.

Each renderer maps a context of G periods to G frames, so frame f is period f and motion across
frames is how the seasonal shape evolves. Values are encoded as POSITION (where a mark sits in
the frame) rather than as INTENSITY wherever possible: position is what Kinetics pretraining
preserves, fine amplitude texture is what it discards.

All renderers return float tensors in [0,1] of shape [B, T, 3, S, S].
"""
import torch
import torch.nn.functional as F

S = 224


def _z(z):
    """z [B,G,P] already in [-1,1] -> row index in [0,S) with 0 at the top."""
    return ((1.0 - (z.clamp(-1, 1) + 1.0) / 2.0) * (S - 1)).round().long()


def _blank(B, T, dev, dtype):
    return torch.zeros(B, T, S, S, device=dev, dtype=dtype)


def _cols(P, dev):
    """map P samples across the full frame width."""
    return (torch.arange(P, device=dev).float() * (S - 1) / max(P - 1, 1)).round().long()


def r_static_matrix(z):
    """CONTROL -- the current renderer: one period x phase image, repeated over frames."""
    B, G, P = z.shape
    img = F.interpolate(((z + 1) / 2).unsqueeze(1), size=(S, S), mode="bilinear",
                        align_corners=False)
    return img.expand(B, 16, S, S).unsqueeze(2).expand(B, 16, 3, S, S).contiguous()


def r_period_dot(z, thick=2):
    """frame f = period f drawn as sparse high-contrast marks at height proportional to value."""
    B, G, P = z.shape
    dev = z.device
    f = _blank(B, G, dev, z.dtype)
    r = _z(z)
    c = _cols(P, dev).view(1, 1, P).expand(B, G, P)
    b = torch.arange(B, device=dev).view(B, 1, 1).expand(B, G, P)
    t = torch.arange(G, device=dev).view(1, G, 1).expand(B, G, P)
    for dr in range(-thick, thick + 1):
        for dc in range(-thick, thick + 1):
            f[b, t, (r + dr).clamp(0, S - 1), (c + dc).clamp(0, S - 1)] = 1.0
    return f.unsqueeze(2).expand(B, G, 3, S, S).contiguous()


def r_period_line(z):
    """frame f = period f as a connected polyline: the mark MOVES continuously between frames."""
    B, G, P = z.shape
    dev = z.device
    f = _blank(B, G, dev, z.dtype)
    up = F.interpolate(z.reshape(B * G, 1, P), size=S, mode="linear", align_corners=True)
    r = _z(up.reshape(B, G, S))
    c = torch.arange(S, device=dev).view(1, 1, S).expand(B, G, S)
    b = torch.arange(B, device=dev).view(B, 1, 1).expand(B, G, S)
    t = torch.arange(G, device=dev).view(1, G, 1).expand(B, G, S)
    for dr in (-1, 0, 1):
        f[b, t, (r + dr).clamp(0, S - 1), c] = 1.0
    return f.unsqueeze(2).expand(B, G, 3, S, S).contiguous()


def r_period_bar(z):
    """frame f = period f as filled columns: value is an AREA that grows and shrinks over time."""
    B, G, P = z.shape
    dev = z.device
    up = F.interpolate(z.reshape(B * G, 1, P), size=S, mode="nearest").reshape(B, G, S)
    r = _z(up)
    rows = torch.arange(S, device=dev).view(1, 1, S, 1)
    f = (rows >= r.unsqueeze(2)).to(z.dtype)
    return f.unsqueeze(2).expand(B, G, 3, S, S).contiguous()


def r_scroll(z, span=2):
    """frame f = a SLIDING window of `span` periods, so consecutive frames overlap and the
    trace translates smoothly -- the closest thing to camera motion over a static scene."""
    B, G, P = z.shape
    dev = z.device
    flat = z.reshape(B, G * P)
    T = G
    out = _blank(B, T, dev, z.dtype)
    w = span * P
    for t in range(T):
        s = min(t * P, max(0, G * P - w))
        seg = flat[:, s:s + w]
        if seg.shape[1] < w:
            seg = F.pad(seg, (0, w - seg.shape[1]), mode="replicate")
        up = F.interpolate(seg.unsqueeze(1), size=S, mode="linear", align_corners=True)
        r = _z(up.squeeze(1))
        c = torch.arange(S, device=dev).view(1, S).expand(B, S)
        b = torch.arange(B, device=dev).view(B, 1).expand(B, S)
        for dr in (-1, 0, 1):
            out[b, t, (r + dr).clamp(0, S - 1), c] = 1.0
    return out.unsqueeze(2).expand(B, T, 3, S, S).contiguous()


def r_growing_matrix(z):
    """frame f = the period x phase image built from periods 0..f only, zero elsewhere.
    Time is CAUSAL ACCUMULATION: the picture literally fills in as the context advances."""
    B, G, P = z.shape
    img = ((z + 1) / 2)
    out = []
    for t in range(G):
        m = torch.zeros_like(img)
        m[:, :t + 1] = img[:, :t + 1]
        out.append(F.interpolate(m.unsqueeze(1), size=(S, S), mode="bilinear",
                                 align_corners=False))
    v = torch.cat(out, 1)
    return v.unsqueeze(2).expand(B, G, 3, S, S).contiguous()


RENDERERS = {"static_matrix": r_static_matrix, "period_dot": r_period_dot,
             "period_line": r_period_line, "period_bar": r_period_bar,
             "scroll": r_scroll, "growing_matrix": r_growing_matrix}


# ---------------------------------------------------------------------------
# frames = CHANNELS, not time.
#
# Every renderer above puts time on the frame axis, which asks the backbone to do temporal
# extrapolation -- the thing six families of experiments say it cannot convert into values.
# This one puts a DIFFERENT CHANNEL in each frame, all at the same origin. The model's
# cross-frame attention then operates on cross-channel structure, which is (a) information no
# univariate prior has, and (b) the only place any backbone in this repository showed an
# advantage (electricity and traffic, both with hundreds of channels).
#
# Nothing is asked of the model that it was not pretrained to do: relate co-occurring frames.

def r_channel_frames(zq, zc):
    """zq [B,G,P] query channel, zc [B,T-1,G,P] co-observed channels -> [B,T,3,S,S].

    Frame 0 is the query channel's period matrix; frames 1..T-1 are other channels observed at
    the SAME origin. All values are at or before the origin, so the zero-shot contract holds.
    """
    B, G, P = zq.shape
    T = zc.shape[1] + 1
    allz = torch.cat([zq.unsqueeze(1), zc], 1).reshape(B * T, 1, G, P)
    img = F.interpolate((allz + 1) / 2, size=(S, S), mode="bilinear", align_corners=False)
    return img.reshape(B, T, S, S).unsqueeze(2).expand(B, T, 3, S, S).contiguous()
