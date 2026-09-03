"""Stage F: strict zero-shot forecasting through VideoMAE's PRETRAINED reconstruction head.

PREPROCESSING_EXPERIMENTS.md section 14. Instead of bolting a freshly initialized
regression head onto frozen features, this uses the interface VideoMAE was actually
pretrained on -- masked patch reconstruction via VideoMAEForPreTraining -- which is the
closest analogue of what makes VisionTS work.

Design (section 14 items 1-6):
  1. each of the 16 frames is an antialiased rolling line chart;
  2. context occupies the LEFT spatial region, the requested future a RIGHT strip;
  3. the same right-hand strip is masked in every frame -> a SPATIAL TUBE mask, which is
     the masking geometry VideoMAE was pretrained with (not whole-future-frame masking);
  4. frame t is rolled back in time, so every VISIBLE pixel is strictly older than the
     forecast origin;
  5. the predicted line geometry is read out of the masked patches and the shared context
     normalization is inverted;
  6. tests/test_reconstruct.py asserts that overwriting the true future cannot change the
     prediction.

CRITICAL CONTROL: `zeroed` repeats the identical decode with the model output replaced by
zeros. pilot/PILOT_RESULTS.md Part 1 found videomae ~= videomae_zero because norm_pix_loss
de-normalization embeds a seasonal prior that does the work. Any Stage F number without
this control is uninterpretable.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_field as RF
from preprocess_renderers import IMG, NF, IMN_MEAN, IMN_STD, _draw_curves, _to_video

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH, TUBELET = 16, 2
GRID = IMG // PATCH                      # 14 x 14
FUT_COLS = 3                             # right strip = 3 patch columns = 48 px
FUT_PX = FUT_COLS * PATCH
CTX_PX = IMG - FUT_PX                    # 176


def render_rolling(x, horizon, shift_px=4):
    """x [B, L, 1] CONTEXT ONLY -> (vid [B,16,3,224,224], mu, sd).

    The x-axis spans CTX_PX context pixels + FUT_PX future pixels. Frame t is shifted
    back by (15-t)*shift_px pixels, so the newest visible sample in every frame is
    strictly older than the forecast origin -- no future value is ever drawn.
    """
    B, L, _ = x.shape
    mu = x.mean(1, keepdim=True)
    sd = x.std(1, keepdim=True) + 1e-8
    z = ((x - mu) / (3 * sd)).clamp(-1, 1)[..., 0]              # [B, L]
    steps_per_px = horizon / FUT_PX                              # time steps per pixel
    ctx_steps = CTX_PX * steps_per_px
    frames = []
    for t in range(NF):
        back = (NF - 1 - t) * shift_px * steps_per_px
        end = L - back                                           # exclusive, <= L always
        start = end - ctx_steps
        pos = torch.linspace(start, end - 1, CTX_PX, device=x.device).clamp(0, L - 1)
        lo = pos.floor().long(); hi = pos.ceil().long().clamp(max=L - 1)
        fr = (pos - lo).unsqueeze(0)
        seg = z[:, lo] * (1 - fr) + z[:, hi] * fr                # [B, CTX_PX]
        pad = torch.full((B, FUT_PX), float("nan"), device=x.device)
        frames.append(torch.cat([seg, pad], 1))
    curves = torch.stack(frames, 1)                              # [B,16,224]
    ink = _draw_curves(curves.nan_to_num(0.0).reshape(B * NF, 1, IMG), [1.0])
    ink = ink.view(B, NF, IMG, IMG)
    ink[..., CTX_PX:] = 0.0                                      # future strip is blank
    gray = 0.15 + 0.75 * ink
    return _to_video(gray), mu, sd


def tube_mask(B, device):
    """Boolean mask [B, 1568] marking the right strip in every frame (spatial tube)."""
    m = torch.zeros(NF // TUBELET, GRID, GRID, dtype=torch.bool, device=device)
    m[:, :, GRID - FUT_COLS:] = True
    return m.reshape(-1).unsqueeze(0).expand(B, -1).contiguous()


def decode_geometry(recon_img, horizon, mu, sd):
    """recon_img [B, IMG, IMG] (last frame) -> forecast [B, horizon, 1].

    Reads the LINE POSITION per column, not brightness: VideoMAE's norm_pix_loss makes
    absolute patch brightness meaningless, but within-patch structure -- hence contour
    position -- survives. Each column is standardized before a soft-argmax over rows.
    """
    B = recon_img.shape[0]
    strip = recon_img[:, :, CTX_PX:]                             # [B, IMG, FUT_PX]
    col = strip - strip.mean(1, keepdim=True)
    col = col / (col.std(1, keepdim=True) + 1e-6)
    w = torch.softmax(col * 4.0, dim=1)                          # soft-argmax over rows
    rows = torch.arange(IMG, device=recon_img.device, dtype=w.dtype).view(1, IMG, 1)
    y = (w * rows).sum(1)                                        # [B, FUT_PX]
    margin, usable = 12, IMG - 24
    v = (1 - 2 * (y - margin) / usable).clamp(-1, 1)             # invert the y mapping
    v = F.interpolate(v.unsqueeze(1), size=horizon, mode="linear",
                      align_corners=False).squeeze(1)            # [B, horizon]
    return (v.unsqueeze(-1) * 3 * sd + mu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ETTh2", choices=list(RF.DATASETS))
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--out-dir", default="pilot/results_field/preprocessing/reconstruct")
    ap.add_argument("--ckpt", default=os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base"))
    ap.add_argument("--horizon-steps", type=int, default=96)
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()

    data, borders = RF.load_mv(args.dataset, args.data_dir, args.max_ch)
    P = RF.P
    horizon = args.horizon_steps
    context = int(np.ceil(CTX_PX * horizon / FUT_PX)) + NF * 4 + P
    Xte, Yte = RF.windows(data, borders, "test", context, horizon, args.stride)
    M = data.shape[1]
    reps = (horizon + P - 1) // P
    G = context // P
    res = {}
    for nm, fn in [("snaive", lambda X: np.tile(X[:, -P:], (1, reps, 1))[:, :horizon]),
                   ("smean", lambda X: np.tile(X[:, context - G * P:].reshape(len(X), G, P, M).mean(1),
                                               (1, reps, 1))[:, :horizon])]:
        p = fn(Xte)
        res[nm] = {"MSE": round(float(np.mean((p - Yte) ** 2)), 4),
                   "MAE": round(float(np.mean(np.abs(p - Yte))), 4)}
        print(f"[done] {nm:8s} {res[nm]}", flush=True)

    Xu = Xte.transpose(0, 2, 1).reshape(-1, context, 1)
    Yu = Yte.transpose(0, 2, 1).reshape(-1, horizon, 1)
    if len(Xu) > args.n:
        k = np.random.default_rng(0).choice(len(Xu), args.n, False); k.sort()
        Xu, Yu = Xu[k], Yu[k]
    print(f"dataset={args.dataset} ctx={context} h={horizon} samples={len(Xu)}", flush=True)

    import transformers
    assert transformers.__version__ < "5"
    from transformers import VideoMAEForPreTraining
    model = VideoMAEForPreTraining.from_pretrained(args.ckpt).to(DEVICE).eval()

    preds = {"model": [], "zeroed": []}
    with torch.no_grad():
        for i in range(0, len(Xu), args.batch):
            xb = torch.from_numpy(Xu[i:i + args.batch]).to(DEVICE)
            vid, mu, sd = render_rolling(xb, horizon)
            B = xb.shape[0]
            bm = tube_mask(B, DEVICE)
            out = model(pixel_values=vid, bool_masked_pos=bm)
            logits = out.logits                                  # [B, n_masked, 2*16*16*3]
            for tag, lg in (("model", logits), ("zeroed", torch.zeros_like(logits))):
                img = torch.zeros(B, IMG, IMG, device=DEVICE)
                # masked patches for the LAST tubelet, second sub-frame = frame 15
                pt = lg.view(B, NF // TUBELET, GRID, FUT_COLS, TUBELET, PATCH, PATCH, 3)
                last = pt[:, -1, :, :, -1].mean(-1)              # [B,GRID,FUT_COLS,16,16]
                strip = last.permute(0, 1, 3, 2, 4).reshape(B, GRID * PATCH, FUT_COLS * PATCH)
                img[:, :, CTX_PX:] = strip
                preds[tag].append(decode_geometry(img, horizon, mu, sd).cpu().numpy())
            if i % (args.batch * 20) == 0:
                print(f"[info] {i}/{len(Xu)}", flush=True)

    for tag in ("model", "zeroed"):
        p = np.concatenate(preds[tag])
        res[f"recon_{tag}"] = {"MSE": round(float(np.mean((p - Yu) ** 2)), 4),
                               "MAE": round(float(np.mean(np.abs(p - Yu))), 4)}
        print(f"[done] recon_{tag:7s} {res[f'recon_{tag}']}", flush=True)

    meta = {"dataset": args.dataset, "ckpt": args.ckpt, "context": context,
            "horizon": horizon, "n": len(Xu), "stride": args.stride, "seed": args.seed,
            "mask": "spatial tube, right %d patch cols (%.1f%%)" % (FUT_COLS, 100 * FUT_COLS / GRID),
            "transformers": transformers.__version__, "torch": torch.__version__,
            "wall_clock_s": round(time.time() - t0, 1)}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"recon_{args.dataset}_h{horizon}_s{args.seed}.json"), "w") as f:
        json.dump({"config": meta, "results": res}, f, indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
