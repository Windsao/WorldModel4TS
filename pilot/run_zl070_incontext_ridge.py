"""ZL-070 -- IN-CONTEXT RIDGE READOUT on frozen video features.

Every value-producing route tested so far was cosine RETRIEVAL, which can only average
neighbours. That is a severe restriction: it cannot express any linear map from feature space to
value space. The measured signature of families F1-F4 is precisely "pretrained ranks neighbours
better but the ranking never becomes values" -- which is exactly what a retrieval-only decoder
would produce if the useful information were linearly decodable but not metric-aligned.

Mechanism. At each origin t, fit a ridge from frozen features to the next H values using ONLY
pairs whose targets are fully observed before t: for every channel and every pseudo-origin e < t,
(features of the window ending at e) -> (the observed continuation [e, e+H)). Pooling across
CHANNELS at the same origin gives hundreds of training pairs. Then apply it to the real window.

Zero-shot contract: for origin t only data strictly before t is touched. The ridge is refit from
scratch at every origin and nothing is carried across origins, so no future value can enter.

Controls: random-init backbone (identical architecture and code path), raw period-vector
features, and the ZL-051 blend. The decisive number is pretrained / random-init.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "third_party", "VisionTS"))
import zeroshot_loop as ZL
from run_zl020_token_kernel import column_tokens
from run_visionts_reference import build_manifest, HORIZON

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAMILY = ["smean", "snaive", "last", "recent_mean"]


def feats(model, win, P, raw=False, batch=64):
    """win [N,L] -> (X [N,D] float64 features, mu [N,1], sd [N,1]).

    Column tokens are NEVER globally mean-pooled into a single vector alone: the ZL-021 audit
    measured pt_col / pt_meanpool = 0.5979, i.e. global pooling destroys most of the signal.
    The descriptor keeps the whole-context mean AND the trailing columns separately.
    """
    N, L = win.shape
    G = L // P
    C = torch.from_numpy(win[:, L - G * P:]).float().to(DEV)
    mu = C.mean(1, keepdim=True)
    sd = C.std(1, keepdim=True) + 1e-6
    z = ((C - mu) / (3 * sd)).clamp(-1, 1).view(N, G, P)
    if raw:
        X = torch.cat([z.reshape(N, -1), z[:, -2:].reshape(N, -1)], 1)
    else:
        E = column_tokens(model, z, G, batch=batch).to(DEV)             # [N,G,d]
        X = torch.cat([E.mean(1), E[:, -1], E[:, -2]], 1)               # [N,3d]
    return X.double().cpu().numpy(), mu.cpu().numpy(), sd.cpu().numpy()


def ridge_fit_predict(Xtr, Ytr, Xte, lams=(1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5), val=0.25, seed=0):
    """Dual-form ridge in float64 with lambda picked on a held-out split of the TRAINING pairs.

    float64 through an SVD, not a fixed lambda in float32: probe_frozen.py originally did the
    latter and returned skill ratios of order 1e152 on electricity.
    """
    n = len(Xtr)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    nv = max(1, int(n * val))
    vi, ti = perm[:nv], perm[nv:]
    mu = Xtr[ti].mean(0, keepdims=True)
    ym = Ytr[ti].mean(0, keepdims=True)
    A = Xtr[ti] - mu
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    UtY = U.T @ (Ytr[ti] - ym)
    best, bl = None, None
    for lam in lams:
        W = Vt.T @ ((S[:, None] / (S[:, None] ** 2 + lam)) * UtY)
        e = float(np.mean(((Xtr[vi] - mu) @ W + ym - Ytr[vi]) ** 2))
        if best is None or e < best:
            best, bl = e, lam
    mu = Xtr.mean(0, keepdims=True)
    ym = Ytr.mean(0, keepdims=True)
    A = Xtr - mu
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    W = Vt.T @ ((S[:, None] / (S[:, None] ** 2 + bl)) * (U.T @ (Ytr - ym)))
    return (Xte - mu) @ W + ym, bl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HORIZON))
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default=ZL.ROOT)
    ap.add_argument("--ckpt", default="MCG-NJU/videomae-base")
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--n-origins", type=int, default=40)
    ap.add_argument("--n-pseudo", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    data, borders, P, L, H, _, _, _, man = build_manifest(a.dataset, a.data_dir, a.max_ch,
                                                          8, 2000, a.seed)
    series = data.T
    M = min(a.max_ch, series.shape[0])
    lo = borders[1] + L + a.n_pseudo * P + H
    hi = borders[2] - H
    step = max(1, (hi - lo) // a.n_origins)
    origins = list(range(lo, hi, step))[:a.n_origins]
    print(f"[ZL-070] {a.dataset} M={M} P={P} L={L} H={H} origins={len(origins)} "
          f"pseudo/origin={a.n_pseudo} train_pairs/origin={M*a.n_pseudo}", flush=True)

    from transformers import VideoMAEModel, VideoMAEConfig
    mp = VideoMAEModel.from_pretrained(a.ckpt).to(DEV).eval()
    torch.manual_seed(a.seed)
    mr = VideoMAEModel(VideoMAEConfig.from_pretrained(a.ckpt)).to(DEV).eval()

    ctx = np.stack([series[c, t - L:t] for t in origins for c in range(M)])
    fut = np.stack([series[c, t:t + H] for t in origins for c in range(M)])
    pr = {k_: v for k_, v in ZL.priors(ctx, P, H).items() if k_ in FAMILY}
    _, blend, _, _ = ZL.pseudo_origin_blend(ctx, P, H, pr, rule="pow_n", dense=True,
                                            min_ctx_periods=8)
    res = {"blend_mse": float(np.mean((blend - fut) ** 2)),
           "smean_mse": float(np.mean((ZL.priors(ctx, P, H)["smean"] - fut) ** 2)),
           "n_origins": len(origins), "M": M}
    err = {"blend": ((blend - fut) ** 2).mean(1)}

    for nm, model, raw in (("pt", mp, False), ("rand", mr, False), ("raw", None, True)):
        preds, lams = [], []
        for t in origins:
            tw, ty = [], []
            for j in range(1, a.n_pseudo + 1):
                e = t - j * P
                if e - L < 0 or e + H > t:
                    continue
                tw.append(np.stack([series[c, e - L:e] for c in range(M)]))
                ty.append(np.stack([series[c, e:e + H] for c in range(M)]))
            W = np.concatenate(tw)
            Y = np.concatenate(ty)
            with torch.no_grad():
                Xtr, mtr, str_ = feats(model, W, P, raw=raw)
                Q = np.stack([series[c, t - L:t] for c in range(M)])
                Xte, mte, ste = feats(model, Q, P, raw=raw)
            Ytr = (Y - mtr) / (3 * str_)
            zp, bl = ridge_fit_predict(Xtr, Ytr, Xte, seed=a.seed)
            preds.append(zp * 3 * ste + mte)
            lams.append(bl)
        y = np.concatenate(preds)
        res[f"{nm}_mse"] = float(np.mean((y - fut) ** 2))
        res[f"{nm}_lambda_median"] = float(np.median(lams))
        err[nm] = ((y - fut) ** 2).mean(1)
        yb = 0.5 * y + 0.5 * blend
        res[f"{nm}_half_blend_mse"] = float(np.mean((yb - fut) ** 2))
        err[f"{nm}_half"] = ((yb - fut) ** 2).mean(1)
        print(f"  {nm:5s} ridge={res[f'{nm}_mse']:.4f} half-blend={res[f'{nm}_half_blend_mse']:.4f} "
              f"lam~{res[f'{nm}_lambda_median']:.0f}", flush=True)

    res["pt_over_rand"] = res["pt_mse"] / res["rand_mse"]
    res["pt_over_raw"] = res["pt_mse"] / res["raw_mse"]
    res["pt_over_blend"] = res["pt_mse"] / res["blend_mse"]
    res["pthalf_over_blend"] = res["pt_half_blend_mse"] / res["blend_mse"]
    ori = np.repeat(np.array(origins), M)
    blk = max(1, int(np.ceil((L + H) / max(1, step))))
    res["boot_pt_vs_rand"] = ZL.paired_bootstrap(err["pt"], err["rand"], ori, blk)
    res["boot_pt_vs_blend"] = ZL.paired_bootstrap(err["pt"], err["blend"], ori, blk)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("boot")},
                     indent=2, default=float), flush=True)
    print("pt/rand CI:", json.dumps(res["boot_pt_vs_rand"], default=float), flush=True)
    d = os.path.join(a.root, "zl070"); os.makedirs(d, exist_ok=True)
    json.dump({"status": "complete", "candidate_id": "ZL-070", "dataset": a.dataset,
               "manifest": man, "results": res, "wall_clock_s": round(time.time() - t0, 1)},
              open(os.path.join(d, f"{a.dataset}.json"), "w"), indent=2, default=float)
    np.savez_compressed(os.path.join(d, f"{a.dataset}_per_pair.npz"), origins=ori, **err)
    print("[done]", a.dataset, flush=True)


if __name__ == "__main__":
    main()
