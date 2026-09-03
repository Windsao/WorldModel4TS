"""Dense period-matrix renderers for the Video-VisionTS diagnostic (spec section 6.1).

V0 dense_static        VisionTS-style dense period matrix, repeated over 16 frames
V1 dense_rolling       same matrix, temporally ordered rolling video
V2 line_rolling_legacy the existing pilot/run_reconstruct.py line renderer (regression control)

Every renderer returns BOTH
  vid_input : the tensor fed to the model, with the hidden block BLANKED, and
  vid_full  : the same image with true hidden pixels, used ONLY to build targets/metrics,
so a leakage bug cannot silently feed the answer to the encoder.

Normalization (spec 3.3): mu/sd from the visible context only, z = 0.4*(x-mu)/sd, shared by
all 16 frames. Visible and hidden matrices are resized SEPARATELY and concatenated on a
patch boundary so bilinear interpolation cannot mix a target pixel into the visible side.
"""

import torch
import torch.nn.functional as F

IMG, NF, PATCH, GRID = 224, 16, 16, 14
IMN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
NORM_CONST = 0.4


def context_stats(ctx):
    """ctx [B,L] visible only -> mu,sd [B,1]."""
    mu = ctx.mean(1, keepdim=True)
    sd = ctx.std(1, keepdim=True, unbiased=False) + 1e-5
    return mu, sd


def normalize(x, mu, sd):
    return NORM_CONST * (x - mu) / sd


def _matrix(z, P):
    """z [B, n*P] -> dense period matrix [B,1,P,n]: rows = phase, cols = period index."""
    B, L = z.shape
    n = L // P
    return z[:, :n * P].view(B, n, P).permute(0, 2, 1).unsqueeze(1).contiguous()


def _gray_to_video(gray):
    """gray [B,H,W] in roughly [-1,1] -> ImageNet-normalized [B,16,3,224,224]."""
    B = gray.shape[0]
    img = ((gray + 1) / 2).clamp(0, 1).unsqueeze(1)                    # [B,1,H,W] to [0,1]
    vid = img.unsqueeze(1).expand(B, NF, 3, IMG, IMG)
    return ((vid - IMN_MEAN.to(gray.device).unsqueeze(1))
            / IMN_STD.to(gray.device).unsqueeze(1)).contiguous()


def _compose(vis_mat, hid_mat, cols):
    """Resize visible and hidden matrices separately, concatenate on a patch boundary."""
    vis_w = (GRID - cols) * PATCH
    hid_w = cols * PATCH
    vis = F.interpolate(vis_mat, size=(IMG, vis_w), mode="bilinear", align_corners=False)
    hid = F.interpolate(hid_mat, size=(IMG, hid_w), mode="bilinear", align_corners=False)
    return vis.squeeze(1), hid.squeeze(1)                              # [B,IMG,w]


def dense_static(ctx, fut, P, cols):
    """V0. ctx [B,L], fut [B,H] (true hidden block). Static image repeated 16x."""
    mu, sd = context_stats(ctx)
    zc, zf = normalize(ctx, mu, sd), normalize(fut, mu, sd)
    vis, hid = _compose(_matrix(zc, P), _matrix(zf, P), cols)
    B = ctx.shape[0]
    blank = torch.zeros_like(hid)
    full = torch.cat([vis, hid], -1)
    inp = torch.cat([vis, blank], -1)
    return _gray_to_video(inp), _gray_to_video(full), mu, sd


def dense_rolling(ext, P, H, cols, L):
    """V1. ext [B, L + 15P + H] history buffer; frame 15's hidden block is ext[-H:].

    Frame f has origin at L + f*P inside `ext`; its visible window is the L steps ending
    there and its hidden block the H steps after. Only frame 15's hidden block is the
    genuine future. Normalization comes from frame 15's visible context for ALL frames.
    """
    B = ext.shape[0]
    origin15 = L + (NF - 1) * P
    mu, sd = context_stats(ext[:, origin15 - L:origin15])
    vis_l, hid_l, full_l = [], [], []
    for f in range(NF):
        o = L + f * P
        zc = normalize(ext[:, o - L:o], mu, sd)
        zf = normalize(ext[:, o:o + H], mu, sd)
        v, h = _compose(_matrix(zc, P), _matrix(zf, P), cols)
        vis_l.append(v); hid_l.append(h)
        full_l.append(torch.cat([v, h], -1))
    inp = torch.stack([torch.cat([v, torch.zeros_like(h)], -1)
                       for v, h in zip(vis_l, hid_l)], 1)               # [B,16,IMG,IMG]
    full = torch.stack(full_l, 1)
    def to_vid(g):
        im = ((g + 1) / 2).clamp(0, 1).unsqueeze(2).expand(B, NF, 3, IMG, IMG)
        return ((im - IMN_MEAN.to(g.device).unsqueeze(1))
                / IMN_STD.to(g.device).unsqueeze(1)).contiguous()
    return to_vid(inp), to_vid(full), mu, sd


def invert_dense(hid_img, P, H, mu, sd):
    """Inverse of the hidden block: [B,IMG,cols*PATCH] -> forecast [B,H,1]."""
    n = max(1, H // P)
    seg = F.interpolate(hid_img.unsqueeze(1), size=(P, n), mode="bilinear",
                        align_corners=False).squeeze(1)                 # [B,P,n]
    flat = seg.permute(0, 2, 1).reshape(hid_img.shape[0], -1)[:, :H]    # period-major
    return (flat / NORM_CONST * sd + mu).unsqueeze(-1)


def video_to_gray(vid):
    """ImageNet-normalized video -> gray in [-1,1], frame-wise."""
    v = vid * IMN_STD.to(vid.device).unsqueeze(1) + IMN_MEAN.to(vid.device).unsqueeze(1)
    return (v.mean(2).clamp(0, 1) * 2 - 1)


RENDERERS = ("dense_static", "dense_rolling", "line_rolling_legacy")
