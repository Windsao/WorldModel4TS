"""Qualitative figure: rendered context frames -> predicted future frames -> decoded values.

Runs one forecast window through a route_b checkpoint and writes
  <out>/qual_context.png   last few context frames the model sees
  <out>/qual_future.png    two rows: ground-truth future frames (top), predicted (bottom)
  <out>/qual_values.csv    t, truth, pred  (context tail + horizon, standardised units)

usage: python pilot/make_qual_fig.py <ckpt_dir> [--dataset ETTh1] [--pred-len 96] [--out paper/figures]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pretrain_route_b as B
from eval_lsf import PERIOD, VTS_CONTEXT, RouteBForecaster, load_lsf


def gray_frames(rows, device):
    """rows [B,NF,224] int -> gray [B,NF,224,224] in [0,1] (same drawing as B.rows_to_video)."""
    rows = rows.to(device).long()
    rgrid = torch.arange(B.IMG, device=device).view(1, 1, B.IMG, 1)
    fill = rgrid >= rows.unsqueeze(2)
    img = torch.where(fill, torch.tensor(B.DARK, device=device), torch.tensor(B.LIGHT, device=device))
    gl = (torch.arange(B.IMG, device=device) % B.GRID_EVERY == 0).view(1, 1, B.IMG, 1) & ~fill
    return torch.where(gl, torch.tensor(B.GRID, device=device), img)


def save_tiled(frames, path, cols, pad=6, scale=2):
    """frames [n,224,224] in [0,1] -> a tiled PNG, white gaps, nearest-neighbour downscale by `scale`."""
    n = frames.shape[0]
    rows_n = int(np.ceil(n / cols))
    f = frames[:, ::scale, ::scale]
    s = f.shape[-1]
    canvas = np.ones((rows_n * s + (rows_n - 1) * pad, cols * s + (cols - 1) * pad), np.float32)
    for i in range(n):
        r, c = divmod(i, cols)
        canvas[r * (s + pad):r * (s + pad) + s, c * (s + pad):c * (s + pad) + s] = f[i]
    arr = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except ImportError:                                     # no Pillow: torchvision writes PNGs too
        import torchvision
        torchvision.io.write_png(torch.from_numpy(arr)[None], path)
    print("wrote", path, arr.shape)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--pred-len", type=int, default=96)
    ap.add_argument("--channel", type=int, default=-1)       # OT
    ap.add_argument("--origin", type=int, default=0)         # offset into the test split
    ap.add_argument("--data-dir", default="/nyx-storage1/hanliu/wm4ts/data")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--n-context", type=int, default=6)      # context frames to show
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    data, test_start, _ = load_lsf(a.dataset, a.data_dir)
    P, L, H = PERIOD[a.dataset], VTS_CONTEXT[a.dataset], a.pred_len
    fc = RouteBForecaster(a.ckpt)
    dev = fc.device

    o = test_start + a.origin
    ctx_np = data[o - L:o, a.channel][None].astype(np.float32)
    truth = data[o:o + H, a.channel].astype(np.float32)

    Pe, hp, k, G_use, lead = fc.plan(P, H, L, "multiperiod")
    ctx = torch.from_numpy(ctx_np[:, -G_use * Pe:]).to(dev)
    mu = ctx.mean(1, keepdim=True)
    sd = ctx.std(1, unbiased=False, keepdim=True) + 1e-6
    z = ((ctx - mu) / sd).clamp(-B.ZMAX, B.ZMAX).view(1, G_use, Pe)
    h = 0.5 + B.BAND * z / B.ZMAX
    c2p = torch.as_tensor(B.col2phase(Pe), device=dev)
    rows_ctx = torch.round((1.0 - h) * (B.IMG - 1)).to(torch.int16)[:, :, c2p]
    rows = torch.full((1, B.NF, B.IMG), int(round(0.5 * (B.IMG - 1))), dtype=torch.int16, device=dev)
    rows[:, lead * B.TS: lead * B.TS + G_use] = rows_ctx

    mask = B.forecast_mask(hp, lead)
    vid = B.rows_to_video(rows, dev)
    B.set_attn_bias(fc.model, mask)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
        logits = fc.model(pixel_values=vid, bool_masked_pos=mask[None].to(dev)).logits
    nf = B.n_future_tokens(hp)
    pred_frames = B.future_frames_from_tokens(logits.float()[:, -nf:], hp).mean(2)[0]      # [hp,224,224]
    zf = B.decode_frames_gray(pred_frames[None], Pe).reshape(1, hp * Pe)
    pred = (torch.nan_to_num(zf, nan=0.0) * sd + mu)[0, :H].cpu().numpy()

    # ground-truth future rendered with the SAME context statistics, for a like-for-like comparison
    zt = ((torch.from_numpy(truth).to(dev)[None] - mu) / sd).clamp(-B.ZMAX, B.ZMAX)
    pad = hp * Pe - H
    if pad > 0:
        zt = torch.cat([zt, zt[:, -1:].expand(-1, pad)], 1)
    ht = (0.5 + B.BAND * zt / B.ZMAX).view(1, hp, Pe)
    rows_t = torch.round((1.0 - ht) * (B.IMG - 1)).to(torch.int16)[:, :, c2p]
    true_frames = gray_frames(rows_t, dev)[0]

    ctx_gray = gray_frames(rows, dev)[0][lead * B.TS: lead * B.TS + G_use]
    show = ctx_gray[-a.n_context:].cpu().numpy()
    save_tiled(show, os.path.join(a.out, "qual_context.png"), cols=a.n_context)
    both = torch.cat([true_frames, pred_frames.clamp(0, 1)], 0).cpu().numpy()
    save_tiled(both, os.path.join(a.out, "qual_future.png"), cols=hp)

    tail = 2 * P
    csv = os.path.join(a.out, "qual_values.csv")
    with open(csv, "w") as f:
        f.write("t,truth,pred\n")
        for i in range(-tail, 0):
            f.write(f"{i},{ctx_np[0, i]:.5f},nan\n")
        for i in range(H):
            f.write(f"{i},{truth[i]:.5f},{pred[i]:.5f}\n")
    mse = float(np.mean((pred - truth) ** 2))
    print(f"wrote {csv}  window MSE {mse:.4f}  dataset {a.dataset} H={H} P={P} hp={hp} k={k}")


if __name__ == "__main__":
    main()
