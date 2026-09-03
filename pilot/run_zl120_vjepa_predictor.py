"""ZL-120 -- V-JEPA 2's PREDICTOR rolled forward in representation space.

This is the experiment the project is named for. Every candidate so far used a video model as a
frozen feature extractor and then asked a retrieval rule or a ridge to turn features into values.
V-JEPA 2 ships the missing piece: a predictor trained to produce the REPRESENTATION of masked
spatio-temporal regions from the representation of the visible ones. That is a latent-space world
model rollout, and it is exactly what a masked pixel autoencoder cannot do -- VideoMAE-base was
trained with norm_pix_loss=True and its decoder structurally cannot emit absolute values.

Mechanism. Frame f is period f. The trailing `reps` frames -- the forecast horizon -- are the
TARGET; everything before is the CONTEXT. The predictor is asked for the target tokens'
representations from the context tokens alone. A ridge fit on pseudo-origins strictly inside the
context maps those predicted representations to values.

Zero-shot contract: the target frames are never shown to the model, and the ridge's training
pairs all end before the origin. Controls: the same predictor with random-init weights, the
encoder-only descriptor of the context (no rollout), and the video-free prior blend.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TFNEW = os.environ.get("TFNEW", "/nyx-storage1/hanliu/shang/vvts/tfnew")
if os.path.isdir(TFNEW):
    sys.path.insert(0, TFNEW)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "third_party", "VisionTS"))
import zeroshot_loop as ZL
from video_renderers_motion import RENDERERS
from run_zl070_incontext_ridge import ridge_fit_predict
from run_visionts_reference import build_manifest, HORIZON

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAMILY = ["smean", "snaive", "last", "recent_mean"]
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def render(win, P, renderer, size):
    import torch.nn.functional as F
    N, L = win.shape
    G = L // P
    C = torch.from_numpy(win[:, L - G * P:]).float().to(DEV)
    mu, sd = C.mean(1, keepdim=True), C.std(1, keepdim=True) + 1e-6
    z = ((C - mu) / (3 * sd)).clamp(-1, 1).view(N, G, P)
    v = RENDERERS[renderer](z)
    if size != v.shape[-1]:
        b, T = v.shape[:2]
        v = F.interpolate(v.reshape(b * T, 3, v.shape[-2], v.shape[-1]), size=(size, size),
                          mode="bilinear", align_corners=False).reshape(b, T, 3, size, size)
    return ((v - IM_MEAN.to(DEV)) / IM_STD.to(DEV)), mu.cpu().numpy(), sd.cpu().numpy()


def rollout_feats(model, win, P, renderer, size, reps, batch=4, mode="predictor"):
    """-> [N, D] float64 descriptor of the PREDICTED future-token representations."""
    N = win.shape[0]
    G = win.shape[1] // P
    g = size // 16
    n_tub = G // 2
    tok_per_tub = g * g
    n_tgt_tub = max(1, (reps + 1) // 2)
    tgt = torch.arange((n_tub - n_tgt_tub) * tok_per_tub, n_tub * tok_per_tub, device=DEV)
    ctxt = torch.arange(0, (n_tub - n_tgt_tub) * tok_per_tub, device=DEV)
    out = []
    for i in range(0, N, batch):
        v, _, _ = render(win[i:i + batch], P, renderer, size)
        b = v.shape[0]
        if mode == "predictor":
            o = model(pixel_values_videos=v,
                      context_mask=[ctxt[None].expand(b, -1)],
                      target_mask=[tgt[None].expand(b, -1)])
            h = o.predictor_output.last_hidden_state
        else:
            h = model(pixel_values_videos=v, skip_predictor=True).last_hidden_state
            h = h[:, ctxt]
        t = max(1, h.shape[1] // tok_per_tub)
        e = h[:, :t * tok_per_tub].view(b, t, tok_per_tub, h.shape[-1]).mean(2)
        out.append(torch.cat([e.mean(1), e[:, -1], e[:, 0]], 1).double().cpu())
        del v, h, e
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HORIZON))
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default=ZL.ROOT)
    ap.add_argument("--ckpt", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--renderer", default="period_line")
    ap.add_argument("--max-ch", type=int, default=48)
    ap.add_argument("--n-origins", type=int, default=10)
    ap.add_argument("--n-pseudo", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    import transformers
    from transformers import VJEPA2Model, VJEPA2Config
    print(f"[ZL-120] transformers {transformers.__version__} ckpt={a.ckpt}", flush=True)
    data, borders, P, L, H, _, _, _, man = build_manifest(a.dataset, a.data_dir, a.max_ch,
                                                          8, 2000, a.seed)
    series = data.T
    M = min(a.max_ch, series.shape[0])
    reps = max(1, H // P)
    lo = borders[1] + L + a.n_pseudo * P + H
    hi = borders[2] - H
    step = max(1, (hi - lo) // a.n_origins)
    origins = list(range(lo, hi, step))[:a.n_origins]
    ctx = np.stack([series[c, t - L:t] for t in origins for c in range(M)])
    fut = np.stack([series[c, t:t + H] for t in origins for c in range(M)])
    pr = {k_: v for k_, v in ZL.priors(ctx, P, H).items() if k_ in FAMILY}
    _, blend, _, _ = ZL.pseudo_origin_blend(ctx, P, H, pr, rule="pow_n", dense=True,
                                            min_ctx_periods=8)
    mb = float(np.mean((blend - fut) ** 2))
    print(f"[ZL-120] {a.dataset} M={M} G={L//P} reps={reps} origins={len(origins)} blend={mb:.4f}",
          flush=True)
    mp = VJEPA2Model.from_pretrained(a.ckpt).to(DEV).eval()
    torch.manual_seed(a.seed)
    mr = VJEPA2Model(VJEPA2Config.from_pretrained(a.ckpt)).to(DEV).eval()
    res = {"blend_mse": mb, "renderer": a.renderer, "n_origins": len(origins), "M": M}
    err, ori = {"blend": ((blend - fut) ** 2).mean(1)}, np.repeat(np.array(origins), M)
    blk = max(1, int(np.ceil((L + H) / max(1, step))))
    for nm, model, mode in (("predictor_pt", mp, "predictor"), ("predictor_rand", mr, "predictor"),
                            ("encoder_pt", mp, "encoder")):
        preds = []
        for t in origins:
            tw, ty = [], []
            for j in range(1, a.n_pseudo + 1):
                e = t - j * P
                if e - L < 0 or e + H > t:
                    continue
                tw.append(np.stack([series[c, e - L:e] for c in range(M)]))
                ty.append(np.stack([series[c, e:e + H] for c in range(M)]))
            W, Y = np.concatenate(tw), np.concatenate(ty)
            Wm = W[:, -(L // P) * P:].mean(1, keepdims=True)
            Ws = W[:, -(L // P) * P:].std(1, keepdims=True) + 1e-6
            with torch.no_grad():
                Xtr = rollout_feats(model, W, P, a.renderer, a.size, reps, mode=mode)
                Q = np.stack([series[c, t - L:t] for c in range(M)])
                Xte = rollout_feats(model, Q, P, a.renderer, a.size, reps, mode=mode)
            Qm = Q.mean(1, keepdims=True); Qs = Q.std(1, keepdims=True) + 1e-6
            zp, _ = ridge_fit_predict(Xtr, (Y - Wm) / (3 * Ws), Xte, seed=a.seed)
            preds.append(zp * 3 * Qs + Qm)
        y = np.concatenate(preds)
        res[f"{nm}_mse"] = float(np.mean((y - fut) ** 2))
        err[nm] = ((y - fut) ** 2).mean(1)
        res[f"{nm}_half_blend"] = float(np.mean((0.5 * y + 0.5 * blend - fut) ** 2))
        print(f"  {nm:16s} = {res[f'{nm}_mse']:.4f}  (/blend {res[f'{nm}_mse']/mb:.4f}, "
              f"half-blend {res[f'{nm}_half_blend']:.4f})", flush=True)
    res["pt_over_rand"] = res["predictor_pt_mse"] / res["predictor_rand_mse"]
    res["predictor_over_encoder"] = res["predictor_pt_mse"] / res["encoder_pt_mse"]
    res["pt_over_blend"] = res["predictor_pt_mse"] / mb
    res["boot_pt_vs_rand"] = ZL.paired_bootstrap(err["predictor_pt"], err["predictor_rand"], ori, blk)
    res["boot_pt_vs_blend"] = ZL.paired_bootstrap(err["predictor_pt"], err["blend"], ori, blk)
    for k_ in ("pt_over_rand", "predictor_over_encoder", "pt_over_blend"):
        print(f"  {k_:24s} = {res[k_]:.4f}", flush=True)
    print(f"  boot pt vs rand  CI=[{res['boot_pt_vs_rand']['lo']:.4f},"
          f"{res['boot_pt_vs_rand']['hi']:.4f}]")
    print(f"  boot pt vs blend CI=[{res['boot_pt_vs_blend']['lo']:.4f},"
          f"{res['boot_pt_vs_blend']['hi']:.4f}]", flush=True)
    d = os.path.join(a.root, "zl120"); os.makedirs(d, exist_ok=True)
    json.dump({"status": "complete", "candidate_id": "ZL-120", "dataset": a.dataset,
               "manifest": man, "results": res, "wall_clock_s": round(time.time() - t0, 1)},
              open(os.path.join(d, f"{a.dataset}.json"), "w"), indent=2, default=float)
    print("[done]", a.dataset, flush=True)


if __name__ == "__main__":
    main()
