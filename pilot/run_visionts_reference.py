"""Stage A: official VisionTS reference on the six shared datasets (spec section 4).

Required positive control. Nothing about VideoMAE may be diagnosed until this reproduces.

Uses the vendored official implementation in third_party/VisionTS unmodified. The
pretrained path is run through `VisionTS.forward` itself; a mirrored forward is used only
for the `zeroed` control (masked patch predictions replaced by zeros before unpatchify),
and it is validated numerically against the official forward before being trusted.
"""

import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np
import torch
import einops
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "third_party", "VisionTS"))
import run_field as RF

HORIZON = {"ETTh1": 96, "ETTh2": 96, "ETTm2": 96, "electricity": 96, "traffic": 96, "solar": 144}


def file_hash(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def git_info():
    """Provenance. On the compute host the working copy is not a git repo, so the
    launcher passes the commit through WM4TS_GIT_COMMIT / WM4TS_GIT_DIRTY."""
    if os.environ.get("WM4TS_GIT_COMMIT"):
        return os.environ["WM4TS_GIT_COMMIT"], os.environ.get("WM4TS_GIT_DIRTY") == "1"
    try:
        c = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        d = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return c, d
    except Exception:
        return "unknown", None


def visionts_commit(path):
    if os.environ.get("VISIONTS_COMMIT"):
        return os.environ["VISIONTS_COMMIT"]
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_manifest(dataset, data_dir, max_ch, stride, n_max, seed=0):
    """Shared window/channel selection, cached by hash so every model sees identical data."""
    data, borders = RF.load_mv(dataset, data_dir, max_ch)
    P = RF.P
    L, H = 16 * P, HORIZON[dataset]
    Xte, Yte = RF.windows(data, borders, "test", L, H, stride)
    M = data.shape[1]
    n_win = len(Xte)
    pairs = np.array([(w, c) for w in range(n_win) for c in range(M)], dtype=np.int64)
    if len(pairs) > n_max:
        idx = np.random.default_rng(seed).choice(len(pairs), n_max, replace=False)
        idx.sort()
        pairs = pairs[idx]
    fpath = os.path.join(data_dir, RF.DATASETS[dataset]["file"])
    man = {"dataset": dataset, "data_path": os.path.abspath(fpath),
           "data_sha1": file_hash(fpath), "P": P, "context": L, "horizon": H,
           "borders": list(map(int, borders)), "max_ch": max_ch, "M": M,
           "stride": stride, "n_windows": n_win, "n_pairs": int(len(pairs)),
           "pairs_sha1": hashlib.sha1(pairs.tobytes()).hexdigest()[:16], "seed": seed}
    return data, borders, P, L, H, Xte, Yte, pairs, man


def gather(X, Y, pairs):
    """[n_win,L,M] -> univariate [n,L,1] using the cached (window, channel) pairs."""
    w, c = pairs[:, 0], pairs[:, 1]
    return X[w, :, c][:, :, None].astype(np.float32), Y[w, :, c][:, :, None].astype(np.float32)


def vts_forward(v, x, zero_pred=False):
    """Mirror of VisionTS.forward, with the option to zero the predicted patches.

    Kept byte-faithful to the vendored implementation; `test_matches_official` below
    asserts it reproduces v(x) exactly when zero_pred=False.
    """
    means = x.mean(1, keepdim=True).detach()
    x_enc = x - means
    stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    stdev /= v.norm_const
    x_enc = x_enc / stdev
    x_enc = einops.rearrange(x_enc, 'b s n -> b n s')
    x_pad = F.pad(x_enc, (v.pad_left, 0), mode='replicate')
    x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=v.periodicity)
    x_resize = v.input_resize(x_2d)
    masked = torch.zeros((x_2d.shape[0], 1, v.image_size, v.num_patch_output * v.patch_size),
                         device=x_2d.device, dtype=x_2d.dtype)
    image_input = einops.repeat(torch.cat([x_resize, masked], dim=-1), 'b 1 h w -> b c h w', c=3)
    _, y, mask = v.vision_model(image_input, mask_ratio=v.mask_ratio,
                                noise=einops.repeat(v.mask, '1 l -> n l', n=image_input.shape[0]))
    if zero_pred:
        y = torch.zeros_like(y)
    rec = v.vision_model.unpatchify(y)
    y_grey = torch.mean(rec, 1, keepdim=True)
    y_seg = v.output_resize(y_grey)
    y_flat = einops.rearrange(y_seg, '(b n) 1 f p -> b (p f) n', b=x.shape[0], f=v.periodicity)
    out = y_flat[:, v.pad_left + v.context_len: v.pad_left + v.context_len + v.pred_len, :]
    return out * stdev.repeat(1, v.pred_len, 1) + means.repeat(1, v.pred_len, 1), image_input, rec, mask


def run_model(v, Xu, batch, device, zero_pred=False, keep_images=0):
    preds, imgs = [], []
    with torch.no_grad():
        for i in range(0, len(Xu), batch):
            xb = torch.from_numpy(Xu[i:i + batch]).to(device)
            p, inp, rec, mk = vts_forward(v, xb, zero_pred)
            preds.append(p.cpu().numpy())
            if keep_images and len(imgs) < keep_images:
                imgs.append((inp[:4].cpu(), rec[:4].cpu()))
    return np.concatenate(preds), imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HORIZON))
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default="pilot/results_field/video_visionts")
    ap.add_argument("--ckpt-dir", default="./ckpt/")
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    data, borders, P, L, H, Xte, Yte, pairs, man = build_manifest(
        args.dataset, args.data_dir, args.max_ch, args.stride, args.n, args.seed)
    Xu, Yu = gather(Xte, Yte, pairs)
    print(f"{args.dataset}: P={P} L={L} H={H} windows={man['n_windows']} pairs={len(pairs)}", flush=True)

    from visionts import VisionTS
    v = VisionTS(arch="mae_base", finetune_type="none", ckpt_dir=args.ckpt_dir).to(dev).eval()
    assert v.vision_model.norm_pix_loss is False, "must use the raw-pixel visualize checkpoint"
    assert v.vision_model.patch_embed.patch_size[0] == 16
    assert v.vision_model.patch_embed.img_size[0] == 224
    v.update_config(context_len=L, pred_len=H, periodicity=P,
                    norm_const=0.4, align_const=0.4, interpolation="bilinear")

    # fidelity check: mirrored forward must equal the official forward
    xb = torch.from_numpy(Xu[:8]).to(dev)
    with torch.no_grad():
        off = v(xb)
        mine, _, _, _ = vts_forward(v, xb)
    fid = float((off - mine).abs().max())
    assert fid < 1e-4, f"mirrored forward diverges from official ({fid:.2e})"
    print(f"[ok] mirrored forward matches official (max|d|={fid:.2e})", flush=True)

    pred_pt, imgs = run_model(v, Xu, args.batch, dev, False, keep_images=4)
    pred_z, _ = run_model(v, Xu, args.batch, dev, True)

    torch.manual_seed(args.seed)
    vr = VisionTS(arch="mae_base", finetune_type="none", ckpt_dir=args.ckpt_dir,
                  load_ckpt=False).to(dev).eval()
    vr.update_config(context_len=L, pred_len=H, periodicity=P,
                     norm_const=0.4, align_const=0.4, interpolation="bilinear")
    pred_r, _ = run_model(vr, Xu, args.batch, dev)

    reps = (H + P - 1) // P
    G = L // P
    base = Xu.reshape(len(Xu), G, P, 1).mean(1)
    smean = np.tile(base, (1, reps, 1))[:, :H]
    snaive = np.tile(Xu[:, -P:], (1, reps, 1))[:, :H]

    def mse(p): return float(np.mean((p - Yu) ** 2))
    def mae(p): return float(np.mean(np.abs(p - Yu)))
    metrics = {"visionts_pretrained": mse(pred_pt), "visionts_zeroed": mse(pred_z),
               "visionts_random": mse(pred_r), "smean": mse(smean), "snaive": mse(snaive),
               "visionts_pretrained_mae": mae(pred_pt), "smean_mae": mae(smean)}
    # per-origin squared errors for the paired moving-block bootstrap
    per_win = {}
    for w in np.unique(pairs[:, 0]):
        sel = pairs[:, 0] == w
        per_win[int(w)] = float(np.mean((pred_pt[sel] - Yu[sel]) ** 2))
    print(json.dumps(metrics, indent=2), flush=True)

    import transformers
    commit, dirty = git_info()
    out = {"status": "complete", "stage": "A", "regime": "ZS", "dataset": args.dataset,
           "git_commit": commit, "git_dirty": dirty,
           "visionts_commit": visionts_commit(os.path.join(HERE, "..", "third_party", "VisionTS")),
           "checkpoint": "mae_visualize_vit_base.pth", "arch": "mae_base",
           "norm_pix_loss": False, "finetune_type": "none",
           "norm_const": 0.4, "align_const": 0.4, "interpolation": "bilinear",
           "torch": torch.__version__, "transformers": transformers.__version__,
           "manifest": man, "n_eval": int(len(Xu)),
           "metrics": metrics, "per_window_mse": per_win,
           "mirrored_forward_max_abs_diff": fid,
           "wall_clock_s": round(time.time() - t0, 1),
           "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None}
    d = os.path.join(args.root, "visionts_reference"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"visionts_{args.dataset}.json"), "w"), indent=2)
    md = os.path.join(args.root, "manifests"); os.makedirs(md, exist_ok=True)
    json.dump(man, open(os.path.join(md, f"manifest_{args.dataset}.json"), "w"), indent=2)

    # save >=16 input/reconstruction examples
    try:
        from PIL import Image
        pd = os.path.join(args.root, "previews"); os.makedirs(pd, exist_ok=True)
        tiles = []
        for inp, rec in imgs:
            for k in range(inp.shape[0]):
                tiles.append((inp[k, 0].numpy(), rec[k, 0].numpy()))
        tiles = tiles[:16]
        if tiles:
            h, w = tiles[0][0].shape
            sheet = np.ones((len(tiles) * (h + 4), 2 * w + 4), dtype=np.float32) * 0.5
            for i, (a, b) in enumerate(tiles):
                r = i * (h + 4)
                sheet[r:r + h, :w] = np.clip((a + 1) / 2, 0, 1)
                sheet[r:r + h, w + 4:2 * w + 4] = np.clip((b + 1) / 2, 0, 1)
            Image.fromarray((sheet * 255).astype(np.uint8)).save(
                os.path.join(pd, f"visionts_{args.dataset}.png"))
    except Exception as e:
        print("[warn] preview failed:", e, flush=True)
    print("[done]", args.dataset, flush=True)


if __name__ == "__main__":
    main()
