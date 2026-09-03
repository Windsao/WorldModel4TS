"""Patch layouts for the Video-VisionTS recovery study (VIDEO_VISIONTS_RECOVERY_EXPERIMENTS s.5).

Everything upstream (time-series normalization, dense period matrix, ImageNet
normalization, backbone, cube packing) is held fixed. Only the PATCH LAYOUT changes, so any
difference isolates the mask-topology hypothesis: VideoMAE reconstructs scattered tube masks
far better than one contiguous right block (native ratio 0.585 vs 0.929 in the audit).

A layout is a permutation `canon_of_phys[p] = c`: the canonical patch `c` is displayed at
physical grid position `p`. The mask marks exactly those physical positions whose canonical
source is a future patch, so no future content can ever land in a visible slot.
"""

import numpy as np
import torch

GRID, PATCH, IMG, NF = 14, 16, 224, 16
LAYOUTS = ("G0", "G1", "G2")


def width_rule(L, H, grid=GRID, align_const=0.4):
    """Pre-registered VisionTS allocation (spec R2.0). Mechanical, never dataset-tuned."""
    visible = max(1, int((L / (L + H)) * grid * align_const))
    return visible, grid - visible


def _fps_positions(k, grid=GRID, seed=0):
    """Deterministic farthest-point sampling of k positions on the grid (blue noise)."""
    coords = np.array([(r, c) for r in range(grid) for c in range(grid)], dtype=np.float64)
    start = int(np.random.default_rng(seed).integers(0, len(coords)))
    chosen = [start]
    d = np.abs(coords - coords[start]).sum(1).astype(np.float64)
    for _ in range(k - 1):
        nxt = int(np.lexsort((np.arange(len(coords)), -d))[0])   # max dist, ties -> lowest index
        chosen.append(nxt)
        d = np.minimum(d, np.abs(coords - coords[nxt]).sum(1))
        d[nxt] = -1
    return np.sort(np.array(chosen))


def canonical_is_future(vc, grid=GRID):
    """[196] bool: canonical patch index -> is it part of the future block?"""
    m = np.zeros((grid, grid), dtype=bool)
    m[:, vc:] = True
    return m.reshape(-1)


def build_layout(name, vc, seed=0, grid=GRID):
    """Return (canon_of_phys [196] int64, phys_of_canon [196] int64, phys_mask [196] bool)."""
    S = grid * grid
    fut_canon = canonical_is_future(vc, grid)
    if name == "G0":
        canon_of_phys = np.arange(S)
    elif name == "G1":
        # visible canonical COLUMNS spread evenly over physical columns, order preserved
        phys_vis_cols = np.unique(np.round(np.linspace(0, grid - 1, vc)).astype(int))
        while len(phys_vis_cols) < vc:                              # guard against collisions
            for c in range(grid):
                if c not in phys_vis_cols:
                    phys_vis_cols = np.sort(np.append(phys_vis_cols, c)); break
        phys_fut_cols = np.array([c for c in range(grid) if c not in phys_vis_cols])
        col_map = np.empty(grid, dtype=int)                          # physical col -> canonical col
        col_map[phys_vis_cols] = np.arange(vc)
        col_map[phys_fut_cols] = np.arange(vc, grid)
        canon_of_phys = (np.arange(grid)[:, None] * grid + col_map[None, :]).reshape(-1)
    elif name == "G2":
        vis_pos = _fps_positions(vc * grid, grid, seed)
        fut_pos = np.array([p for p in range(S) if p not in set(vis_pos.tolist())])
        vis_canon = np.where(~fut_canon)[0]                          # row-major canonical order
        fut_canon_idx = np.where(fut_canon)[0]
        canon_of_phys = np.empty(S, dtype=int)
        canon_of_phys[vis_pos] = vis_canon
        canon_of_phys[fut_pos] = fut_canon_idx
    else:
        raise ValueError(name)
    phys_of_canon = np.argsort(canon_of_phys)
    phys_mask = fut_canon[canon_of_phys]                             # physical slot holds a future patch
    assert phys_mask.sum() == fut_canon.sum(), "mask/content mapping mismatch"
    return canon_of_phys.astype(np.int64), phys_of_canon.astype(np.int64), phys_mask


def _to_patches(img, grid=GRID, ps=PATCH):
    B = img.shape[0]
    return img.view(B, grid, ps, grid, ps).permute(0, 1, 3, 2, 4).reshape(B, grid * grid, ps, ps)


def _from_patches(p, grid=GRID, ps=PATCH):
    B = p.shape[0]
    return p.view(B, grid, grid, ps, ps).permute(0, 1, 3, 2, 4).reshape(B, grid * ps, grid * ps)


def permute_image(img, canon_of_phys):
    """Canonical image [B,224,224] -> physical layout."""
    idx = torch.as_tensor(canon_of_phys, device=img.device, dtype=torch.long)
    return _from_patches(_to_patches(img)[:, idx])


def unpermute_image(img, canon_of_phys):
    """Physical image -> canonical. Exact inverse of `permute_image`."""
    inv = torch.as_tensor(np.argsort(canon_of_phys), device=img.device, dtype=torch.long)
    return _from_patches(_to_patches(img)[:, inv])


def tube_mask_from_spatial(phys_mask, B, device, n_tubelets=8):
    """[196] bool -> VideoMAE bool_masked_pos [B,1568], identical over all tubelets."""
    sp = torch.as_tensor(phys_mask, device=device, dtype=torch.bool).view(1, -1).expand(B, -1)
    return sp.unsqueeze(1).expand(B, n_tubelets, -1).reshape(B, -1).contiguous()


def nearest_visible_physical(cubes, bm, grid=GRID, n_tubelets=8):
    """Copy mean/std from the closest VISIBLE physical patch (Manhattan, ties -> lowest
    row-major token index). Reads only positions where bool_masked_pos is False."""
    B, N, P, C = cubes.shape
    m = bm.view(B, n_tubelets, grid, grid)
    cu = cubes.view(B, n_tubelets, grid, grid, P, C)
    cmean = cu.mean(dim=-2)
    cstd = cu.var(dim=-2, unbiased=True).sqrt() + 1e-6
    coords = np.array([(r, c) for r in range(grid) for c in range(grid)])
    sp = m[:, 0].reshape(B, -1).detach().cpu().numpy()               # tube mask: same each tubelet
    em, es = cmean.clone(), cstd.clone()
    for b in range(B):
        vis = np.where(~sp[b])[0]
        if len(vis) == 0:
            continue
        d = np.abs(coords[:, None, :] - coords[None, vis, :]).sum(-1)   # [196, n_vis]
        nearest = vis[np.lexsort((np.tile(vis, (len(coords), 1)).T, d.T), axis=0)[0]]
        src = torch.as_tensor(nearest, device=cubes.device, dtype=torch.long)
        need = torch.as_tensor(sp[b], device=cubes.device, dtype=torch.bool)
        flat_m = cmean[b].reshape(n_tubelets, grid * grid, C)
        flat_s = cstd[b].reshape(n_tubelets, grid * grid, C)
        nm = flat_m[:, src]; ns = flat_s[:, src]
        em[b] = torch.where(need.view(1, -1, 1), nm, flat_m).view(n_tubelets, grid, grid, C)
        es[b] = torch.where(need.view(1, -1, 1), ns, flat_s).view(n_tubelets, grid, grid, C)
    em = em.reshape(B, N, 1, C); es = es.reshape(B, N, 1, C)
    return em[bm].view(B, -1, 1, C), es[bm].view(B, -1, 1, C)
