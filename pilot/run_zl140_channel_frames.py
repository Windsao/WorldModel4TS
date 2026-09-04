"""ZL-140 -- frames = CHANNELS. The backbone's cross-frame attention applied to cross-channel
structure, evaluated against VisionTS on the published benchmark.

Every previous rendering put TIME on the frame axis, asking the backbone for temporal
extrapolation -- the operation six families of experiments say it cannot turn into values. Here
frame 0 is the query channel's period matrix and frames 1..15 are OTHER channels observed at the
same origin. Cross-frame attention then operates on cross-channel structure, which is
(a) information no univariate prior has and (b) the only setting where any backbone here showed
an advantage (electricity and traffic, both with hundreds of channels).

Every value is at or before the origin, so the zero-shot contract holds. The forecast comes from
an in-context ridge fit only on pseudo-origins inside the context. Controls: random-init backbone,
a raw-feature ridge given the identical training budget, and the video-free prior blend. Scored
against the stored VisionTS per-window MSE on its own manifest (pairs_sha1 asserted equal).
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
TFNEW = os.environ.get("TFNEW", "")
if TFNEW and os.path.isdir(TFNEW):
    sys.path.insert(0, TFNEW)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "third_party", "VisionTS"))
import zeroshot_loop as ZL
from video_renderers_motion import r_channel_frames
from run_zl070_incontext_ridge import ridge_fit_predict
from run_visionts_reference import build_manifest, HORIZON

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAMILY = ["smean", "snaive", "last", "recent_mean"]
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
T = 16


def norm(win, P):
    G = win.shape[1] // P
    C = torch.from_numpy(win[:, win.shape[1] - G * P:]).float().to(DEV)
    mu, sd = C.mean(1, keepdim=True), C.std(1, keepdim=True) + 1e-6
    return ((C - mu) / (3 * sd)).clamp(-1, 1).view(-1, G, P), mu.cpu().numpy(), sd.cpu().numpy()


def feats(model, win, P, rng, batch=4):
    """win [M,L] all channels at ONE origin -> X [M, 3d]. Each channel is a query in turn; the
    companion frames are a fixed random sample of the OTHER channels at the same origin."""
    z, mu, sd = norm(win, P)
    M = z.shape[0]
    comp = np.stack([rng.choice(np.delete(np.arange(M), i), size=min(T - 1, M - 1),
                                replace=(M - 1) < (T - 1)) for i in range(M)])
    out = []
    m_, s_ = IM_MEAN.to(DEV), IM_STD.to(DEV)
    for i in range(0, M, batch):
        idx = comp[i:i + batch]
        zc = z[torch.from_numpy(idx).to(DEV)]
        v = (r_channel_frames(z[i:i + batch], zc) - m_) / s_
        h = model(pixel_values=v).last_hidden_state
        b, n, d = h.shape
        t = max(1, n // (14 * 14))
        e = h.view(b, t, 14 * 14, d).mean(2)
        out.append(torch.cat([e.mean(1), e[:, 0], e[:, -1]], 1).double().cpu())
        del v, h, e
    return torch.cat(out).numpy(), mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HORIZON))
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default=ZL.ROOT)
    ap.add_argument("--ckpt", default="MCG-NJU/videomae-base")
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--max-origins", type=int, default=60)
    ap.add_argument("--n-pseudo", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    data, borders, P, L, H, Xte, Yte, pairs, man = build_manifest(
        a.dataset, a.data_dir, a.max_ch, 8, 2000, a.seed)
    # walk UP from --root: the reference lives beside the results tree, and --root may point
    # several levels deeper (e.g. .../zeroshot_loop/v2), which a single dirname() misses.
    vp = None
    cands = ["pilot/results_field/video_visionts"]
    d_ = os.path.abspath(a.root)
    for _ in range(5):
        d_ = os.path.dirname(d_)
        cands.append(os.path.join(d_, "video_visionts"))
    for r_ in cands:
        c = os.path.join(r_, "visionts_reference", f"visionts_{a.dataset}.json")
        if os.path.exists(c):
            vp = c
            break
    assert vp is not None, f"VisionTS reference for {a.dataset} not found; searched {cands}"
    vj = json.load(open(vp))
    assert vj["manifest"]["pairs_sha1"] == man["pairs_sha1"], "manifest mismatch vs VisionTS"
    pw = vj["per_window_mse"]
    series = data.T
    M = min(a.max_ch, series.shape[0])
    stride = man["stride"]
    uo = np.unique(pairs[:, 0])
    keep = [w for w in uo if borders[1] + int(w) * stride >= a.n_pseudo * P]
    # SPAN the whole test range. keep[::step][:max_origins] silently truncates to the FIRST
    # max_origins*step windows -- on electricity that dropped the last 70 of 620 origins and made
    # the subset systematically harder (blend/VisionTS 1.31 here vs 0.98 on the full manifest).
    if len(keep) > a.max_origins:
        sel = np.linspace(0, len(keep) - 1, a.max_origins).round().astype(int)
        wins = [keep[i] for i in sorted(set(sel.tolist()))]
    else:
        wins = list(keep)
    step = max(1, len(keep) // max(1, len(wins)))
    print(f"[ZL-140] {a.dataset} M={M} P={P} L={L} H={H} origins {len(wins)}/{len(uo)} "
          f"pairs/origin={M*a.n_pseudo}", flush=True)
    from transformers import VideoMAEModel, VideoMAEConfig
    mp = VideoMAEModel.from_pretrained(a.ckpt).to(DEV).eval()
    torch.manual_seed(a.seed)
    mr = VideoMAEModel(VideoMAEConfig.from_pretrained(a.ckpt)).to(DEV).eval()
    CTX, FUT, VTS, ORI = [], [], [], []
    for w in wins:
        t = borders[1] + int(w) * stride + L
        if t + H > series.shape[1]:
            continue
        CTX.append(np.stack([series[c, t - L:t] for c in range(M)]))
        FUT.append(np.stack([series[c, t:t + H] for c in range(M)]))
        VTS.append(float(pw[str(int(w))])); ORI.append(int(w))
    ctx, fut, vts = np.concatenate(CTX), np.concatenate(FUT), np.array(VTS)
    pr = {k_: v for k_, v in ZL.priors(ctx, P, H).items() if k_ in FAMILY}
    _, blend, _, _ = ZL.pseudo_origin_blend(ctx, P, H, pr, rule="pow_n", dense=True,
                                            min_ctx_periods=8)
    Y = {"blend": blend}
    for nm in ("pt", "rand", "raw"):
        preds = []
        for w in ORI:
            t = borders[1] + int(w) * stride + L
            rng = np.random.default_rng(a.seed + int(w))
            tw, ty = [], []
            for j in range(1, a.n_pseudo + 1):
                e = t - j * P
                if e - L < 0 or e + H > t:
                    continue
                tw.append(np.stack([series[c, e - L:e] for c in range(M)]))
                ty.append(np.stack([series[c, e:e + H] for c in range(M)]))
            W, Yt = np.concatenate(tw), np.concatenate(ty)
            Q = np.stack([series[c, t - L:t] for c in range(M)])
            if nm == "raw":
                def rf(X):
                    m_ = X.mean(1, keepdims=True); s_ = X.std(1, keepdims=True) + 1e-6
                    Z = ((X - m_) / (3 * s_)).clip(-1, 1)
                    return np.concatenate([Z, Z[:, -2 * P:]], 1).astype(np.float64), m_, s_
                Xtr, mtr, str_ = rf(W); Xte, mte, ste = rf(Q)
            else:
                model = mp if nm == "pt" else mr
                with torch.no_grad():
                    Xl, ml, sl = [], [], []
                    for k in range(0, len(W), M):
                        x_, m_, s_ = feats(model, W[k:k + M], P,
                                           np.random.default_rng(a.seed + int(w) + k))
                        Xl.append(x_); ml.append(m_); sl.append(s_)
                    Xtr = np.concatenate(Xl); mtr = np.concatenate(ml); str_ = np.concatenate(sl)
                    Xte, mte, ste = feats(model, Q, P, rng)
            zp, _ = ridge_fit_predict(Xtr, (Yt - mtr) / (3 * str_), Xte, seed=a.seed)
            preds.append(zp * 3 * ste + mte)
        Y[nm] = np.concatenate(preds)
        e = ((Y[nm] - fut) ** 2).mean(1).reshape(len(ORI), M).mean(1)
        print(f"  {nm:5s} {e.mean():.4f}  /VisionTS {e.mean()/vts.mean():.4f}", flush=True)
    res = {"n_origins": len(ORI), "M": M, "visionts_mse": float(vts.mean()),
           "pairs_sha1": man["pairs_sha1"], "renderer": "channel_frames"}
    arms = {"blend": Y["blend"], "raw": Y["raw"], "pt": Y["pt"], "rand": Y["rand"],
            "A_no_video": (Y["blend"] + Y["raw"]) / 2,
            "B_plus_pt": (Y["blend"] + Y["raw"] + Y["pt"]) / 3,
            "C_plus_rand": (Y["blend"] + Y["raw"] + Y["rand"]) / 3}
    rng2 = np.random.default_rng(20260902)
    blk = min(max(1, int(np.ceil((L + H) / stride / max(1, step)))), max(1, len(ORI) // 4))
    per = {}
    for nm, y in arms.items():
        e = ((y - fut) ** 2).mean(1).reshape(len(ORI), M).mean(1); per[nm] = e
        res[f"{nm}_mse"] = float(e.mean()); res[f"{nm}_over_visionts"] = float(e.mean()/vts.mean())
        nb = int(np.ceil(len(ORI) / blk)); rs = []
        for _ in range(2000):
            s_ = rng2.integers(0, max(1, len(ORI) - blk + 1), nb)
            idx = np.concatenate([np.arange(s, s + blk) for s in s_])[:len(ORI)]
            rs.append(e[idx].mean() / vts[idx].mean())
        rs = np.array(rs)
        res[f"{nm}_ci"] = [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))]
        v = "WINS" if res[f"{nm}_ci"][1] < 1 else ("loses" if res[f"{nm}_ci"][0] > 1 else "tie")
        print(f"  {nm:12s} {e.mean():.4f} /VisionTS {res[f'{nm}_over_visionts']:.4f} "
              f"CI [{res[f'{nm}_ci'][0]:.4f},{res[f'{nm}_ci'][1]:.4f}] {v}", flush=True)
    res["B_over_A"] = res["B_plus_pt_mse"] / res["A_no_video_mse"]
    res["B_over_C"] = res["B_plus_pt_mse"] / res["C_plus_rand_mse"]
    res["pt_over_rand"] = res["pt_mse"] / res["rand_mse"]
    res["boot_B_vs_A"] = ZL.paired_bootstrap(per["B_plus_pt"], per["A_no_video"],
                                             np.array(ORI), blk)
    print(f"  pt/rand={res['pt_over_rand']:.4f}  B/A={res['B_over_A']:.4f} "
          f"CI [{res['boot_B_vs_A']['lo']:.4f},{res['boot_B_vs_A']['hi']:.4f}]  "
          f"B/C={res['B_over_C']:.4f}", flush=True)
    d = os.path.join(a.root, "zl140"); os.makedirs(d, exist_ok=True)
    json.dump({"status": "complete", "candidate_id": "ZL-140", "dataset": a.dataset,
               "manifest": man, "results": res, "wall_clock_s": round(time.time() - t0, 1)},
              open(os.path.join(d, f"{a.dataset}.json"), "w"), indent=2, default=float)
    print("[done]", a.dataset, flush=True)


if __name__ == "__main__":
    main()
