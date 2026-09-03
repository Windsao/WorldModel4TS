"""Frozen-feature diagnostic: is the frozen-backbone failure a numerical/input-
structure problem or a representational one?

For a frozen (Kinetics) VideoMAE we extract pooled features for three renderings
of the SAME univariate series, plus the multivariate 'field' rendering, and measure
  (1) participation ratio (effective dimensionality) of the feature cloud, and
  (2) a closed-form ridge linear-probe skill = probe_MSE / constant_MSE
      (lower = features carry more of the target; 1.0 = useless).

Renderings:
  uni  : our period-per-frame barcode (one period tiled down all rows)   [degenerate]
  vts  : VisionTS-style single 2D image [n_periods x phase], static clip  [2D, non-degenerate]
  field: multivariate 2D per frame (rows=variables, cols=phase)           [2D, joint]

If `vts`/`field` features are much higher-rank and lower-skill-ratio than `uni`,
the frozen failure is driven by input-structure OOD (fixable), not by video
features being fundamentally unusable.
"""
import os, sys, json, argparse
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_field as R
from run_field import load_mv, windows, FieldVMAE, NF, IMG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def feats(enc, vid, imn_mean, imn_std, bs=64):
    """vid: [B, NF, 3, IMG, IMG] already ImageNet-normed -> pooled feats [B, d]."""
    out = []
    enc.eval()
    with torch.no_grad():
        for i in range(0, len(vid), bs):
            v = vid[i:i + bs].to(DEVICE)
            h = enc(pixel_values=v).last_hidden_state.mean(1)
            out.append(h.cpu().numpy())
    return np.concatenate(out)


def imnet(img_bnf):  # img_bnf: [B, NF, 1, IMG, IMG] in [0,1] -> normed 3ch
    imn_mean = R.IMN_MEAN.view(1, 1, 3, 1, 1)
    imn_std = R.IMN_STD.view(1, 1, 3, 1, 1)
    vid = img_bnf.expand(-1, -1, 3, -1, -1)
    return (vid - imn_mean) / imn_std


def render_uni(x, P):  # x [B, L, 1] -> barcode video [B,NF,1,IMG,IMG] in [0,1]
    B, L, _ = x.shape
    mu = x.mean(1, keepdim=True); sd = x.std(1, keepdim=True) + 1e-8
    g = (((x - mu) / (3 * sd)).clamp(-1, 1) + 1) / 2      # [B,L,1]
    G = L // P
    g = g[:, :G * P].view(B, G, P, 1)
    if G != NF:
        g = g.permute(0, 2, 3, 1).reshape(B, P, G)
        g = F.interpolate(g, size=NF, mode="linear", align_corners=False)
        g = g.reshape(B, P, 1, NF).permute(0, 3, 1, 2)    # [B,NF,P,1]
    fr = g.permute(0, 1, 3, 2).expand(B, NF, P, P)        # tile row -> barcode
    img = F.interpolate(fr.reshape(B * NF, 1, P, P), size=(IMG, IMG), mode="nearest")
    return imnet(img.view(B, NF, 1, IMG, IMG))


def render_vts(x, P):  # VisionTS-style: whole lookback -> ONE 2D image, static clip
    B, L, _ = x.shape
    mu = x.mean(1, keepdim=True); sd = x.std(1, keepdim=True) + 1e-8
    g = (((x - mu) / (3 * sd)).clamp(-1, 1) + 1) / 2      # [B,L,1]
    G = L // P
    g = g[:, :G * P].view(B, G, P)                        # [B, n_periods, phase]  <- 2D signal
    img1 = F.interpolate(g.unsqueeze(1), size=(IMG, IMG), mode="nearest")  # [B,1,IMG,IMG]
    vid = img1.unsqueeze(1).expand(B, NF, 1, IMG, IMG)    # static clip (same frame x16)
    return imnet(vid)


def render_field(x, P):  # x [B, L, M] -> [B,NF,1,IMG,IMG]; rows=vars, cols=phase
    B, L, M = x.shape
    mu = x.mean(1, keepdim=True); sd = x.std(1, keepdim=True) + 1e-8
    g = (((x - mu) / (3 * sd)).clamp(-1, 1) + 1) / 2
    G = L // P
    g = g[:, :G * P].view(B, G, P, M)
    if G != NF:
        g = g.permute(0, 2, 3, 1).reshape(B, P * M, G)
        g = F.interpolate(g, size=NF, mode="linear", align_corners=False)
        g = g.reshape(B, P, M, NF).permute(0, 3, 1, 2)
    fr = g.permute(0, 1, 3, 2)                            # [B,NF,M,P]
    img = F.interpolate(fr.reshape(B * NF, 1, M, P), size=(IMG, IMG), mode="nearest")
    return imnet(img.view(B, NF, 1, IMG, IMG))


def participation_ratio(Fm):
    Fc = Fm - Fm.mean(0, keepdims=True)
    s = np.linalg.svd(Fc, compute_uv=False)
    ev = s ** 2
    return float((ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-12))   # effective #dims


def ridge_skill(Fm, Y, lams=(1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6),
                split=0.7, val=0.15):
    """Numerically stable ridge linear probe.

    The frozen features are strongly collinear (participation ratio << feat_dim), so the
    768x768 Gram matrix is near-singular. Solving it in float32 with a fixed lam=10 --
    the original implementation -- returns garbage: observed skill_ratio ~1e152 on
    electricity. Two fixes: (a) solve in float64 through an SVD instead of forming and
    inverting the Gram matrix, (b) select lam on a held-out validation split rather than
    fixing it, since the right scale depends on how collapsed the features are.
    Returns (probe_mse, const_mse, skill_ratio, chosen_lam).
    """
    Fm = np.asarray(Fm, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = len(Fm)
    k, kv = int(n * split), int(n * (split + val))
    Xtr, Xva, Xte = Fm[:k], Fm[k:kv], Fm[kv:]
    Ytr, Yva, Yte = Y[:k], Y[k:kv], Y[kv:]
    mx = Xtr.mean(0, keepdims=True)
    sx = Xtr.std(0, keepdims=True) + 1e-8
    Xtr, Xva, Xte = (Xtr - mx) / sx, (Xva - mx) / sx, (Xte - mx) / sx
    my = Ytr.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
    UtY = U.T @ (Ytr - my)
    best_v, best_lam, best_W = np.inf, None, None
    for lam in lams:
        W = Vt.T @ ((S / (S ** 2 + lam))[:, None] * UtY)
        v = float(np.mean((Xva @ W + my - Yva) ** 2))
        if np.isfinite(v) and v < best_v:
            best_v, best_lam, best_W = v, lam, W
    pred = Xte @ best_W + my
    probe_mse = float(np.mean((pred - Yte) ** 2))
    const_mse = float(np.mean((my - Yte) ** 2))
    return probe_mse, const_mse, probe_mse / (const_mse + 1e-12), best_lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="electricity")
    ap.add_argument("--data-dir", default="/nyx-storage1/hanliu/wm4ts/data")
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--n", type=int, default=1500, help="samples for the probe")
    ap.add_argument("--horizon-p", type=int, default=4)
    args = ap.parse_args()

    data, borders = load_mv(args.dataset, args.data_dir, args.max_ch)
    P = R.P; context = NF * P; horizon = args.horizon_p * P; M = data.shape[1]
    # cap windows to what the probe needs (field needs args.n; avoids OOM on long-context solar)
    Xtr, Ytr = windows(data, borders, "train", context, horizon, 1, args.n * 3 + 1000)
    print(f"dataset={args.dataset} M={M} P={P} ctx={context} h={horizon} windows={len(Xtr)}", flush=True)

    # univariate samples (shared by uni & vts)
    Xu = Xtr.transpose(0, 2, 1).reshape(-1, context, 1)
    Yu = Ytr.transpose(0, 2, 1).reshape(-1, horizon)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xu), min(args.n, len(Xu)), replace=False)
    Xu, Yu = Xu[idx], Yu[idx]
    # field samples (multivariate)
    idf = rng.choice(len(Xtr), min(args.n, len(Xtr)), replace=False)
    Xf = Xtr[idf]; Yf = Ytr[idf].reshape(len(idf), -1)

    enc = FieldVMAE(1, P, horizon, "uni", pretrained=True).enc.to(DEVICE)

    results = {}
    for name, X, Y, render in [
        ("uni_barcode", Xu, Yu, render_uni),
        ("vts_2d",      Xu, Yu, render_vts),
        ("field_2d",    Xf, Yf, render_field),
    ]:
        xb = torch.from_numpy(X.astype("float32"))
        vid = render(xb, P)
        Fm = feats(enc, vid, None, None)
        pr = participation_ratio(Fm)
        pmse, cmse, skill, lam = ridge_skill(Fm, Y.astype("float32"))
        results[name] = {"partic_ratio": round(pr, 2), "feat_dim": Fm.shape[1],
                         "probe_mse": round(pmse, 4), "const_mse": round(cmse, 4),
                         "skill_ratio": round(skill, 4), "ridge_lam": lam}
        print(f"[{name:12s}] PR={pr:7.2f}/{Fm.shape[1]}  probe={pmse:.4f} "
              f"const={cmse:.4f}  skill={skill:.4f}  lam={lam:g}", flush=True)

    os.makedirs("pilot/results_field/probe", exist_ok=True)
    with open(f"pilot/results_field/probe/probe_{args.dataset}.json", "w") as f:
        json.dump({"dataset": args.dataset, "M": M, "P": P, "n": args.n,
                   "results": results}, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
