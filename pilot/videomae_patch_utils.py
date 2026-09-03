"""Exact VideoMAE cube packing, native targets, masks and unpatchify.

Reproduces the target construction inside `VideoMAEForPreTraining.forward` for the
installed Transformers version, so that decoder logits can be compared against the
*native* normalized-cube target rather than a guessed one.

The packing is NOT inferred from shapes -- it mirrors the installed source exactly
(permute (0,1,4,6,2,5,7,3), per-cube per-channel normalization over the 2*16*16 pixel
positions, `var(unbiased=True).sqrt() + 1e-6`), and
tests/test_video_visionts.py verifies it numerically against `model(...).loss`.

Token order is N = (t' * H' + h') * W' + w' with T'=8, H'=W'=14 -> 1568 tokens.
Within one token the 1536 values are ordered [ts, ps_h, ps_w, C] with channel fastest.
"""

import hashlib

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def unnormalize(pixel_values):
    """[B,T,3,H,W] ImageNet-normalized -> [0,1] frames, matching the modeling code."""
    dev, dt = pixel_values.device, pixel_values.dtype
    mean = torch.as_tensor(IMAGENET_MEAN, device=dev, dtype=dt)[None, None, :, None, None]
    std = torch.as_tensor(IMAGENET_STD, device=dev, dtype=dt)[None, None, :, None, None]
    return pixel_values * std + mean


def patchify(frames, tubelet_size=2, patch_size=16):
    """[B,T,C,H,W] in [0,1] -> cubes [B, N, ts*ps*ps, C] (step 1-3 of the modeling code)."""
    B, T, C, H, W = frames.shape
    f = frames.view(B, T // tubelet_size, tubelet_size, C,
                    H // patch_size, patch_size, W // patch_size, patch_size)
    f = f.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
    return f.view(B, (T // tubelet_size) * (H // patch_size) * (W // patch_size),
                  tubelet_size * patch_size * patch_size, C)


def unpatchify(cubes, T=16, H=224, W=224, tubelet_size=2, patch_size=16):
    """Exact inverse of `patchify`: [B,N,ts*ps*ps,C] -> [B,T,C,H,W]."""
    B = cubes.shape[0]
    C = cubes.shape[-1]
    Tp, Hp, Wp = T // tubelet_size, H // patch_size, W // patch_size
    f = cubes.view(B, Tp, Hp, Wp, tubelet_size, patch_size, patch_size, C)
    f = f.permute(0, 1, 4, 7, 2, 5, 3, 6).contiguous()      # inverse permutation
    return f.view(B, T, C, H, W)


def native_targets(pixel_values, tubelet_size=2, patch_size=16, return_stats=True):
    """Normalized-cube targets exactly as VideoMAEForPreTraining builds them.

    Returns (targets [B,N,ts*ps*ps*C], mean [B,N,1,C], std [B,N,1,C]).
    The mean/std are the per-cube per-channel statistics needed for oracle inversion
    (interpretation I1) -- they are the very quantities `norm_pix_loss` discards.
    """
    frames = unnormalize(pixel_values)
    cubes = patchify(frames, tubelet_size, patch_size)                 # [B,N,P,C]
    mean = cubes.mean(dim=-2, keepdim=True)
    std = cubes.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
    norm = (cubes - mean) / std
    B, N, P, C = cubes.shape
    return norm.view(B, N, P * C), mean, std


def denormalize_cubes(pred_flat, mean, std, tubelet_size=2, patch_size=16, C=3):
    """[B,n,ts*ps*ps*C] normalized predictions + per-cube stats -> raw-pixel cubes."""
    B, n, D = pred_flat.shape
    P = D // C
    return pred_flat.view(B, n, P, C) * std + mean


# ---------------------------------------------------------------- masks
def _spatial_to_tube(spatial, n_tubelets=8):
    """[B,196] spatial map -> [B,1568] repeated identically over every tubelet."""
    B = spatial.shape[0]
    return spatial.unsqueeze(1).expand(B, n_tubelets, -1).reshape(B, -1).contiguous()


def make_mask(name, B, device, seed=0, grid=14, n_tubelets=8):
    """Build a bool_masked_pos [B, 1568]. Every variant is a TUBE mask: the same
    spatial pattern is repeated across all 8 tubelets (spec section 6.2)."""
    S = grid * grid                                    # 196
    if name.startswith("random_tube_"):
        k = {"random_tube_90": 176, "random_tube_75": 147}[name]
        g = torch.Generator(device="cpu").manual_seed(seed)
        sp = torch.zeros(B, S, dtype=torch.bool)
        for b in range(B):                             # per-sample pattern, fixed by seed
            idx = torch.randperm(S, generator=g)[:k]
            sp[b, idx] = True
        sp = sp.to(device)
    elif name.startswith("right_"):
        cols = {"right_13": 13, "right_10": 10, "right_9": 9, "right_3": 3}[name]
        m = torch.zeros(grid, grid, dtype=torch.bool, device=device)
        m[:, grid - cols:] = True
        sp = m.reshape(1, S).expand(B, S).contiguous()
    else:
        raise ValueError(f"unknown mask {name}")
    return _spatial_to_tube(sp, n_tubelets)


MASK_SPATIAL_COUNT = {"random_tube_90": 176, "random_tube_75": 147,
                      "right_13": 182, "right_10": 140, "right_9": 126, "right_3": 42}


def mask_hash(mask):
    return hashlib.sha1(mask.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def masked_columns(name, grid=14):
    """Number of masked patch columns for right-block masks (None for random)."""
    return {"right_13": 13, "right_10": 10, "right_9": 9, "right_3": 3}.get(name)


# ---------------------------------------------------------------- causal stats (I2)
def causal_cube_stats(cubes, bool_masked_pos, rule="nearest_visible_left",
                      grid=14, n_tubelets=8):
    """Estimate masked-cube mean/std WITHOUT using hidden values (interpretation I2).

    cubes [B,N,P,C] are the true cubes; only entries where bool_masked_pos is False may
    be read. Returns (mean, std) shaped [B, n_masked, 1, C] in masked-token order.

    rule:
      nearest_visible_left -- copy from the nearest visible cube in the same row and
                              tubelet, searching leftwards (the only causal direction
                              for a right block);
      context_global       -- mean/std over all visible cubes of the sample.
    """
    B, N, P, C = cubes.shape
    m = bool_masked_pos.view(B, n_tubelets, grid, grid)
    cu = cubes.view(B, n_tubelets, grid, grid, P, C)
    cmean = cu.mean(dim=-2)                                       # [B,t,h,w,C]
    cstd = cu.var(dim=-2, unbiased=True).sqrt() + 1e-6
    if rule == "context_global":
        vis = (~m).unsqueeze(-1).float()                           # [B,t,h,w,1]
        denom = vis.sum(dim=(1, 2, 3)).clamp(min=1.0)              # [B,1]
        gm = (cmean * vis).sum(dim=(1, 2, 3)) / denom              # [B,C]
        gs = (cstd * vis).sum(dim=(1, 2, 3)) / denom
        em = gm[:, None, None, None, :].expand_as(cmean)
        es = gs[:, None, None, None, :].expand_as(cstd)
    elif rule == "nearest_visible_left":
        em = cmean.clone(); es = cstd.clone()
        for w in range(1, grid):                                   # sweep left -> right
            need = m[:, :, :, w]                                   # [B,t,h]
            em[:, :, :, w] = torch.where(need.unsqueeze(-1), em[:, :, :, w - 1], em[:, :, :, w])
            es[:, :, :, w] = torch.where(need.unsqueeze(-1), es[:, :, :, w - 1], es[:, :, :, w])
        # any column-0 masked cube falls back to the global visible mean
        vis = (~m).unsqueeze(-1).float()
        denom = vis.sum(dim=(1, 2, 3)).clamp(min=1.0)
        gm = (cmean * vis).sum(dim=(1, 2, 3)) / denom
        gs = (cstd * vis).sum(dim=(1, 2, 3)) / denom
        col0 = m[:, :, :, 0].unsqueeze(-1)
        em[:, :, :, 0] = torch.where(col0, gm[:, None, None, :].expand_as(em[:, :, :, 0]), em[:, :, :, 0])
        es[:, :, :, 0] = torch.where(col0, gs[:, None, None, :].expand_as(es[:, :, :, 0]), es[:, :, :, 0])
    else:
        raise ValueError(rule)
    em = em.view(B, N, 1, C); es = es.view(B, N, 1, C)
    sel = bool_masked_pos
    return (em[sel].view(B, -1, 1, C), es[sel].view(B, -1, 1, C))
