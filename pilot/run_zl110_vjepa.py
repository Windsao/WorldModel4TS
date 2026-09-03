"""ZL-110 -- V-JEPA 2 as the backbone instead of VideoMAE.

Why a different backbone is the right next move. VideoMAE-base is a masked PIXEL autoencoder
trained with norm_pix_loss=True, so its decoder structurally cannot emit absolute values -- a
blocker documented in this repository and the reason the whole generative family had to be
scored on normalised reconstructions. V-JEPA 2 predicts in REPRESENTATION space instead: it is
trained to predict the embedding of masked spatio-temporal regions, which is exactly the
"world model" framing this project is named for, and it has no norm_pix_loss to work around.

Protocol is deliberately identical to ZL-080 so the two backbones are directly comparable: the
same renderers, the same in-context ridge readout, the same paired bootstrap, and the same
decision statistic (pretrained / random-init of the identical architecture).
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


def feats(model, win, P, renderer, size, batch=4):
    """win [N,L] -> (X [N,3d] float64, mu, sd) using per-temporal-position token means."""
    import torch.nn.functional as F
    N, L = win.shape
    G = L // P
    C = torch.from_numpy(win[:, L - G * P:]).float().to(DEV)
    mu, sd = C.mean(1, keepdim=True), C.std(1, keepdim=True) + 1e-6
    z = ((C - mu) / (3 * sd)).clamp(-1, 1).view(N, G, P)
    fn = RENDERERS[renderer]
    m_, s_ = IM_MEAN.to(DEV), IM_STD.to(DEV)
    out = []
    for i in range(0, N, batch):
        v = fn(z[i:i + batch])                                  # [b,T,3,224,224] in [0,1]
        if size != v.shape[-1]:
            b, T = v.shape[:2]
            v = F.interpolate(v.reshape(b * T, 3, v.shape[-2], v.shape[-1]),
                              size=(size, size), mode="bilinear",
                              align_corners=False).reshape(b, T, 3, size, size)
        v = (v - m_) / s_
        h = model(pixel_values_videos=v).last_hidden_state       # [b, n_tokens, d]
        b, n, d = h.shape
        g = size // 16
        t = max(1, n // (g * g))
        e = h[:, :t * g * g].view(b, t, g * g, d).mean(2)
        out.append(torch.cat([e.mean(1), e[:, -1], e[:, -2] if t > 1 else e[:, -1]], 1)
                   .double().cpu())
        del v, h, e
    return torch.cat(out).numpy(), mu.cpu().numpy(), sd.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HORIZON))
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default=ZL.ROOT)
    ap.add_argument("--ckpt", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--max-ch", type=int, default=48)
    ap.add_argument("--n-origins", type=int, default=8)
    ap.add_argument("--n-pseudo", type=int, default=6)
    ap.add_argument("--renderers", nargs="+", default=["static_matrix", "period_line"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    import transformers
    print(f"[ZL-110] transformers {transformers.__version__} ckpt={a.ckpt}", flush=True)
    from transformers import VJEPA2Model, VJEPA2Config
    data, borders, P, L, H, _, _, _, man = build_manifest(a.dataset, a.data_dir, a.max_ch,
                                                          8, 2000, a.seed)
    series = data.T
    M = min(a.max_ch, series.shape[0])
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
    print(f"[ZL-110] {a.dataset} M={M} origins={len(origins)} pairs/origin={M*a.n_pseudo} "
          f"blend={mb:.4f}", flush=True)
    mp = VJEPA2Model.from_pretrained(a.ckpt).to(DEV).eval()
    torch.manual_seed(a.seed)
    mr = VJEPA2Model(VJEPA2Config.from_pretrained(a.ckpt)).to(DEV).eval()
    res = {"blend_mse": mb, "backbone": a.ckpt, "n_origins": len(origins), "M": M,
           "renderers": {}}
    ori = np.repeat(np.array(origins), M)
    blk = max(1, int(np.ceil((L + H) / max(1, step))))
    for rn in a.renderers:
        row, perr = {}, {}
        for nm, model in (("pt", mp), ("rand", mr)):
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
                with torch.no_grad():
                    Xtr, mtr, str_ = feats(model, W, P, rn, a.size)
                    Q = np.stack([series[c, t - L:t] for c in range(M)])
                    Xte, mte, ste = feats(model, Q, P, rn, a.size)
                zp, _ = ridge_fit_predict(Xtr, (Y - mtr) / (3 * str_), Xte, seed=a.seed)
                preds.append(zp * 3 * ste + mte)
            y = np.concatenate(preds)
            row[f"{nm}_mse"] = float(np.mean((y - fut) ** 2))
            row[f"{nm}_half_blend"] = float(np.mean((0.5 * y + 0.5 * blend - fut) ** 2))
            perr[nm] = ((y - fut) ** 2).mean(1)
        row["pt_over_rand"] = row["pt_mse"] / row["rand_mse"]
        row["pt_over_blend"] = row["pt_mse"] / mb
        row["boot_pt_vs_rand"] = ZL.paired_bootstrap(perr["pt"], perr["rand"], ori, blk)
        res["renderers"][rn] = row
        ci = row["boot_pt_vs_rand"]
        print(f"  {rn:16s} pt={row['pt_mse']:.4f} rand={row['rand_mse']:.4f} "
              f"pt/rand={row['pt_over_rand']:.4f} pt/blend={row['pt_over_blend']:.4f} "
              f"CI=[{ci['lo']:.3f},{ci['hi']:.3f}]{'*' if ci['hi'] < 1 else ''}", flush=True)
    d = os.path.join(a.root, "zl110"); os.makedirs(d, exist_ok=True)
    json.dump({"status": "complete", "candidate_id": "ZL-110", "dataset": a.dataset,
               "manifest": man, "results": res, "wall_clock_s": round(time.time() - t0, 1)},
              open(os.path.join(d, f"{a.dataset}.json"), "w"), indent=2, default=float)
    print("[done]", a.dataset, flush=True)


if __name__ == "__main__":
    main()
