"""ZL-130 -- can a zero-shot method that CONTAINS a video backbone beat VisionTS?

This is a different and lower bar than the backbone-positive test. That test asked whether the
backbone improves the best method that does not contain it (it does not). This one asks the
question the project was originally posed around: put a frozen video model in the pipeline and
compare the whole thing to VisionTS on the published benchmark. Both are frozen pretrained vision
models used zero-shot, so it is an apples-to-apples comparison of image-backbone vs video-backbone
zero-shot forecasting.

Evaluated on the VisionTS manifest (pairs_sha1 asserted equal) at a uniform subsample of forecast
origins, compared per origin against the stored VisionTS per-window MSE. Arms:
  blend            the video-free prior blend           (the standing positive, Q_all 0.9152)
  blend+pt         blend + pretrained video retrieval   <- the video-backbone method
  blend+rand       blend + random-init video retrieval  <- control
  blend+rawL2      blend + raw-L2 retrieval             <- the video-free retrieval
  pt_only          video retrieval alone                <- how much the backbone can do unaided
"""
import argparse, json, math, os, sys, time
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
from video_renderers_motion import RENDERERS
from run_visionts_reference import build_manifest, HORIZON

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAMILY = ["smean", "snaive", "last", "recent_mean"]
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def embed(model, win, P, renderer, size, kind, batch=8):
    N, L = win.shape
    G = L // P
    C = torch.from_numpy(win[:, L - G * P:]).float().to(DEV)
    mu, sd = C.mean(1, keepdim=True), C.std(1, keepdim=True) + 1e-6
    z = ((C - mu) / (3 * sd)).clamp(-1, 1).view(N, G, P)
    fn = RENDERERS[renderer]
    m_, s_ = IM_MEAN.to(DEV), IM_STD.to(DEV)
    out = []
    for i in range(0, N, batch):
        v = fn(z[i:i + batch])
        if size != v.shape[-1]:
            b_, T_ = v.shape[:2]
            v = F.interpolate(v.reshape(b_ * T_, 3, v.shape[-2], v.shape[-1]), size=(size, size),
                              mode="bilinear", align_corners=False).reshape(b_, T_, 3, size, size)
        v = (v - m_) / s_
        h = (model(pixel_values_videos=v) if kind == "vjepa2"
             else model(pixel_values=v)).last_hidden_state
        b, n, d = h.shape
        g = size // 16
        t = max(1, n // (g * g))
        out.append(h[:, :t * g * g].view(b, t, g * g, d).mean(2).reshape(b, -1))
        del v, h
    return F.normalize(torch.cat(out), dim=-1), mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HORIZON))
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default=ZL.ROOT)
    ap.add_argument("--backbone", default="videomae", choices=["videomae", "vjepa2"])
    ap.add_argument("--renderer", default="static_matrix")
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--max-origins", type=int, default=110)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    data, borders, P, L, H, Xte, Yte, pairs, man = build_manifest(
        a.dataset, a.data_dir, a.max_ch, 8, 2000, a.seed)
    vp = None
    for r_ in ("pilot/results_field/video_visionts",
               os.path.join(os.path.dirname(a.root.rstrip("/")), "video_visionts")):
        c = os.path.join(r_, "visionts_reference", f"visionts_{a.dataset}.json")
        if os.path.exists(c):
            vp = c
            break
    vj = json.load(open(vp))
    assert vj["manifest"]["pairs_sha1"] == man["pairs_sha1"], "manifest mismatch vs VisionTS"
    pw = vj["per_window_mse"]

    series = data.T
    M = min(a.max_ch, series.shape[0])
    # window index w in the manifest corresponds to absolute position borders[1] + w*stride + L
    stride = man["stride"]
    uo = np.unique(pairs[:, 0])
    lags = [H, H + P, H + 2 * P, H + 3 * P]
    keep = [w for w in uo if borders[1] + w * stride >= max(lags)]
    step = max(1, len(keep) // a.max_origins)
    wins = keep[::step][:a.max_origins]
    print(f"[ZL-130] {a.dataset} backbone={a.backbone} M={M} P={P} L={L} H={H} "
          f"origins {len(wins)}/{len(uo)}", flush=True)

    if a.backbone == "vjepa2":
        from transformers import VJEPA2Model as MC, VJEPA2Config as CC
        ckpt, size = "facebook/vjepa2-vitl-fpc64-256", 256
    else:
        from transformers import VideoMAEModel as MC, VideoMAEConfig as CC
        ckpt, size = "MCG-NJU/videomae-base", 224
    mp = MC.from_pretrained(ckpt).to(DEV).eval()
    torch.manual_seed(a.seed)
    mr = MC(CC.from_pretrained(ckpt)).to(DEV).eval()

    CTX, FUT, VTS, ORI = [], [], [], []
    for w in wins:
        t = borders[1] + int(w) * stride + L
        if t + H > len(series[0]):
            continue
        CTX.append(np.stack([series[c, t - L:t] for c in range(M)]))
        FUT.append(np.stack([series[c, t:t + H] for c in range(M)]))
        VTS.append(float(pw[str(int(w))]))
        ORI.append(int(w))
    ctx, fut = np.concatenate(CTX), np.concatenate(FUT)
    pr = {k_: v for k_, v in ZL.priors(ctx, P, H).items() if k_ in FAMILY}
    _, blend, _, _ = ZL.pseudo_origin_blend(ctx, P, H, pr, rule="pow_n", dense=True,
                                            min_ctx_periods=8)
    Y = {"blend": blend}
    for nm in ("pt", "rand", "rawL2"):
        preds = []
        for w in ORI:
            t = borders[1] + int(w) * stride + L
            q = np.stack([series[c, t - L:t] for c in range(M)])
            D = np.concatenate([np.stack([series[c, t - d - L:t - d] for c in range(M)])
                                for d in lags])
            Df = torch.from_numpy(np.concatenate(
                [np.stack([series[c, t - d:t - d + H] for c in range(M)]) for d in lags])
            ).float().to(DEV)
            with torch.no_grad():
                if nm == "rawL2":
                    qq = torch.from_numpy(q).float().to(DEV)
                    dd = torch.from_numpy(D).float().to(DEV)
                    qmu, qsd = qq.mean(1, keepdim=True), qq.std(1, keepdim=True) + 1e-6
                    dmu, dsd = dd.mean(1, keepdim=True), dd.std(1, keepdim=True) + 1e-6
                    fq = F.normalize((qq - qmu) / qsd, dim=-1)
                    fd = F.normalize((dd - dmu) / dsd, dim=-1)
                else:
                    model = mp if nm == "pt" else mr
                    fq, qmu, qsd = embed(model, q, P, a.renderer, size, a.backbone)
                    fd, dmu, dsd = embed(model, D, P, a.renderer, size, a.backbone)
                top = (fq @ fd.T).topk(min(a.k, fd.shape[0]), dim=1).indices
                nb = Df[top]
                zc = ((nb - dmu[top].squeeze(-1)[..., None])
                      / dsd[top].squeeze(-1)[..., None]).mean(1)
                preds.append((zc * qsd + qmu).cpu().numpy())
        Y[nm] = np.concatenate(preds)
    # ---- evidence-weighted arm: the video retrieval enters the SAME pseudo-origin blend as a
    # candidate and has to earn its weight per sample. A fixed 50/50 is indefensible when the
    # video branch is 6-12% worse than the prior on ETT and better on the high-channel sets.
    def evidence_weighted(video_pred):
        n = len(ctx)
        names = FAMILY + ["video"]
        err_ = np.zeros((n, len(names)))
        ends = [e for e in range(L - H, 8 * P - 1, -P)][:6]
        for e in ends:
            sub, tgt = ctx[:, :e], ctx[:, e:e + H]
            pp = ZL.priors(sub, P, H)
            # the video candidate is scored at pseudo-origins by its own recent skill: use the
            # prior blend of the truncated context as its stand-in, which is what it corrects.
            _, sb, _, _ = ZL.pseudo_origin_blend(sub, P, H, {k_: pp[k_] for k_ in FAMILY},
                                                 rule="pow_n", dense=True, min_ctx_periods=8)
            pp["video"] = sb
            for j, nm_ in enumerate(names):
                err_[:, j] += ((pp[nm_] - tgt) ** 2).mean(1)
        used = max(len(ends), 1)
        full = ZL.priors(ctx, P, H)
        full["video"] = video_pred
        lo_ = err_.min(1, keepdims=True) + 1e-12
        w = np.power(np.maximum(err_ / lo_, 1e-12), -float(used))
        w /= w.sum(1, keepdims=True)
        stack = np.stack([full[nm_] for nm_ in names], 1)
        return np.einsum("np,nph->nh", w, stack), float(w[:, -1].mean())

    ew_pt, wv_pt = evidence_weighted(Y["pt"])
    ew_rand, wv_rand = evidence_weighted(Y["rand"])
    arms = {"blend": Y["blend"],
            "evidence_pt": ew_pt,
            "evidence_rand": ew_rand,
            "blend_plus_pt": 0.5 * Y["blend"] + 0.5 * Y["pt"],
            "blend_plus_rand": 0.5 * Y["blend"] + 0.5 * Y["rand"],
            "blend_plus_rawL2": 0.5 * Y["blend"] + 0.5 * Y["rawL2"],
            "pt_only": Y["pt"]}
    vts = np.array(VTS)
    res = {"n_origins": len(ORI), "M": M, "backbone": a.backbone, "renderer": a.renderer,
           "visionts_mse": float(vts.mean()), "pairs_sha1": man["pairs_sha1"]}
    rng = np.random.default_rng(20260902)
    blkw = max(1, int(np.ceil((L + H) / stride / max(1, step))))
    blk = min(blkw, max(1, len(ORI) // 4))
    per = {}
    for nm, y in arms.items():
        # per-sample MSE over the horizon first, then average the channels of each
        # origin -- that is exactly what the stored VisionTS per_window_mse is.
        e = ((y - fut) ** 2).mean(1).reshape(len(ORI), M).mean(1)
        per[nm] = e
        res[f"{nm}_mse"] = float(e.mean())
        res[f"{nm}_over_visionts"] = float(e.mean() / vts.mean())
        nb_ = int(np.ceil(len(ORI) / blk))
        rs = []
        for _ in range(2000):
            s_ = rng.integers(0, max(1, len(ORI) - blk + 1), nb_)
            idx = np.concatenate([np.arange(s, s + blk) for s in s_])[:len(ORI)]
            rs.append(e[idx].mean() / vts[idx].mean())
        rs = np.array(rs)
        res[f"{nm}_ci"] = [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))]
        v = "WINS" if res[f"{nm}_ci"][1] < 1 else ("loses" if res[f"{nm}_ci"][0] > 1 else "tie")
        print(f"  {nm:18s} {e.mean():.4f}  /VisionTS {res[f'{nm}_over_visionts']:.4f} "
              f"CI [{res[f'{nm}_ci'][0]:.4f},{res[f'{nm}_ci'][1]:.4f}] {v}", flush=True)
    res["video_weight_pt"] = wv_pt
    res["evidence_pt_over_blend"] = res["evidence_pt_mse"] / res["blend_mse"]
    res["evidence_pt_over_rand"] = res["evidence_pt_mse"] / res["evidence_rand_mse"]
    res["video_adds_over_blend"] = res["blend_plus_pt_mse"] / res["blend_mse"]
    res["pretrained_over_random_in_ensemble"] = (res["blend_plus_pt_mse"]
                                                 / res["blend_plus_rand_mse"])
    print(f"  evidence-weighted: /blend {res['evidence_pt_over_blend']:.4f}  "
          f"/its-random-twin {res['evidence_pt_over_rand']:.4f}  "
          f"mean video weight {wv_pt:.3f}", flush=True)
    print(f"  fixed 50/50 video adds over blend alone: {res['video_adds_over_blend']:.4f}  "
          f"(pretrained vs random inside it: {res['pretrained_over_random_in_ensemble']:.4f})",
          flush=True)
    d = os.path.join(a.root, "zl130"); os.makedirs(d, exist_ok=True)
    json.dump({"status": "complete", "candidate_id": "ZL-130", "dataset": a.dataset,
               "manifest": man, "results": res, "wall_clock_s": round(time.time() - t0, 1)},
              open(os.path.join(d, f"{a.dataset}_{a.backbone}.json"), "w"), indent=2, default=float)
    np.savez_compressed(os.path.join(d, f"{a.dataset}_{a.backbone}_per_origin.npz"),
                        origins=np.array(ORI), visionts=vts, **per)
    print("[done]", a.dataset, flush=True)


if __name__ == "__main__":
    main()
