"""ZL-100 -- the two positive signals combined: CROSS-CHANNEL retrieval on MOTION renderings.

Two independent findings point the same way:
  ZL-060  cross-channel donor retrieval: pretrained/random = 0.59 on electricity, and the
          pretrained retrieval blended with the prior beat the prior alone (0.1810 vs 0.1878).
  ZL-080  the frame axis was dead. Every earlier feature route fed VideoMAE a static image
          repeated 16 times (measured inter-frame difference exactly 0.0000). With real motion
          -- frame f = period f -- pretrained/random flips from 1.1396 to 0.7276.

So the retrieval in ZL-060 was computed on exactly the degenerate rendering that ZL-080 shows
suppresses the pretrained advantage. This candidate re-runs it with motion.

Preregistered decision rule. The video backbone is credited only if BOTH hold, on both datasets:
  (i)  pretrained beats its random-init control, paired CI excluding 1;
  (ii) adding the pretrained retrieval to the best video-free method (prior blend + raw-L2
       retrieval) improves it, paired CI excluding 1, AND beats the same ensemble built with
       the random-init backbone.
Condition (ii) is what ZL-060 could not satisfy: raw-L2 retrieval alone beat the pretrained one.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "third_party", "VisionTS"))
import zeroshot_loop as ZL
from video_renderers_motion import RENDERERS
from run_visionts_reference import build_manifest, HORIZON

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAMILY = ["smean", "snaive", "last", "recent_mean"]
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def motion_embed(model, win, P, renderer, batch=8):
    """win [N,L] -> (F [N,D] L2-normalised descriptor, mu, sd), per-temporal-position tokens."""
    N, L = win.shape
    G = L // P
    C = torch.from_numpy(win[:, L - G * P:]).float().to(DEV)
    mu, sd = C.mean(1, keepdim=True), C.std(1, keepdim=True) + 1e-6
    z = ((C - mu) / (3 * sd)).clamp(-1, 1).view(N, G, P)
    fn = RENDERERS[renderer]
    m_, s_ = IM_MEAN.to(DEV), IM_STD.to(DEV)
    out = []
    for i in range(0, N, batch):
        v = (fn(z[i:i + batch]) - m_) / s_
        h = model(pixel_values=v).last_hidden_state
        b, n, d = h.shape
        t = n // (14 * 14)
        out.append(h.view(b, t, 14 * 14, d).mean(2).reshape(b, -1))
        del v, h
    return F.normalize(torch.cat(out), dim=-1), mu, sd


def retrieve(fq, fd, Df, dmu, dsd, qmu, qsd, k):
    top = (fq @ fd.T).topk(min(k, fd.shape[0]), dim=1).indices
    nb = Df[top]
    zc = ((nb - dmu[top].squeeze(-1)[..., None]) / dsd[top].squeeze(-1)[..., None]).mean(1)
    return (zc * qsd + qmu).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HORIZON))
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default=ZL.ROOT)
    ap.add_argument("--ckpt", default="MCG-NJU/videomae-base")
    ap.add_argument("--renderer", default="period_line")
    ap.add_argument("--max-ch", type=int, default=96)
    ap.add_argument("--n-origins", type=int, default=120)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    data, borders, P, L, H, _, _, _, man = build_manifest(a.dataset, a.data_dir, a.max_ch,
                                                          8, 2000, a.seed)
    series = data.T
    M = min(a.max_ch, series.shape[0])
    lags = [H, H + P, H + 2 * P, H + 3 * P]
    lo = borders[1] + L + max(lags)
    hi = borders[2] - H
    step = max(1, (hi - lo) // a.n_origins)
    origins = list(range(lo, hi, step))[:a.n_origins]
    ctx = np.stack([series[c, t - L:t] for t in origins for c in range(M)])
    fut = np.stack([series[c, t:t + H] for t in origins for c in range(M)])
    pr = {k_: v for k_, v in ZL.priors(ctx, P, H).items() if k_ in FAMILY}
    _, blend, _, _ = ZL.pseudo_origin_blend(ctx, P, H, pr, rule="pow_n", dense=True,
                                            min_ctx_periods=8)
    print(f"[ZL-100] {a.dataset} renderer={a.renderer} M={M} origins={len(origins)} "
          f"donors/origin={M*len(lags)} blend={np.mean((blend-fut)**2):.4f}", flush=True)
    from transformers import VideoMAEModel, VideoMAEConfig
    mp = VideoMAEModel.from_pretrained(a.ckpt).to(DEV).eval()
    torch.manual_seed(a.seed)
    mr = VideoMAEModel(VideoMAEConfig.from_pretrained(a.ckpt)).to(DEV).eval()
    Y = {"blend": blend}
    for nm in ("pt", "rand", "rawL2"):
        preds = []
        for t in origins:
            q = np.stack([series[c, t - L:t] for c in range(M)])
            dw = [np.stack([series[c, t - d - L:t - d] for c in range(M)]) for d in lags]
            df = [np.stack([series[c, t - d:t - d + H] for c in range(M)]) for d in lags]
            D, Df = np.concatenate(dw), torch.from_numpy(np.concatenate(df)).float().to(DEV)
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
                    fq, qmu, qsd = motion_embed(model, q, P, a.renderer)
                    fd, dmu, dsd = motion_embed(model, D, P, a.renderer)
                preds.append(retrieve(fq, fd, Df, dmu, dsd, qmu, qsd, a.k))
        Y[nm] = np.concatenate(preds)
        print(f"  {nm:6s} alone={np.mean((Y[nm]-fut)**2):.4f} "
              f"half-blend={np.mean((0.5*Y[nm]+0.5*blend-fut)**2):.4f}", flush=True)
    ens = {"A_no_video": (Y["blend"] + Y["rawL2"]) / 2,
           "B_plus_pretrained": (Y["blend"] + Y["rawL2"] + Y["pt"]) / 3,
           "C_plus_random": (Y["blend"] + Y["rawL2"] + Y["rand"]) / 3}
    res = {"renderer": a.renderer, "n_origins": len(origins), "M": M}
    err = {}
    for k_, v_ in list(Y.items()) + list(ens.items()):
        res[f"{k_}_mse"] = float(np.mean((v_ - fut) ** 2))
        err[k_] = ((v_ - fut) ** 2).mean(1)
    res["pt_over_rand"] = res["pt_mse"] / res["rand_mse"]
    res["pt_over_rawL2"] = res["pt_mse"] / res["rawL2_mse"]
    res["B_over_A"] = res["B_plus_pretrained_mse"] / res["A_no_video_mse"]
    res["B_over_C"] = res["B_plus_pretrained_mse"] / res["C_plus_random_mse"]
    ori = np.repeat(np.array(origins), M)
    blk = max(1, int(np.ceil((L + H) / max(1, step))))
    res["boot_pt_vs_rand"] = ZL.paired_bootstrap(err["pt"], err["rand"], ori, blk)
    res["boot_B_vs_A"] = ZL.paired_bootstrap(err["B_plus_pretrained"], err["A_no_video"], ori, blk)
    res["boot_B_vs_C"] = ZL.paired_bootstrap(err["B_plus_pretrained"], err["C_plus_random"], ori, blk)
    print(f"\n  A(no video)={res['A_no_video_mse']:.4f}  B(+pretrained)={res['B_plus_pretrained_mse']:.4f}"
          f"  C(+random)={res['C_plus_random_mse']:.4f}")
    print(f"  pt/rand={res['pt_over_rand']:.4f} CI={res['boot_pt_vs_rand']['lo']:.4f},"
          f"{res['boot_pt_vs_rand']['hi']:.4f}")
    print(f"  B/A    ={res['B_over_A']:.4f} CI={res['boot_B_vs_A']['lo']:.4f},"
          f"{res['boot_B_vs_A']['hi']:.4f}")
    print(f"  B/C    ={res['B_over_C']:.4f} CI={res['boot_B_vs_C']['lo']:.4f},"
          f"{res['boot_B_vs_C']['hi']:.4f}", flush=True)
    crit = (res["boot_pt_vs_rand"]["hi"] < 1 and res["boot_B_vs_A"]["hi"] < 1
            and res["boot_B_vs_C"]["hi"] < 1)
    res["backbone_positive"] = bool(crit)
    print(f"\n  BACKBONE-POSITIVE (both preregistered conditions): {crit}", flush=True)
    d = os.path.join(a.root, "zl100"); os.makedirs(d, exist_ok=True)
    json.dump({"status": "complete", "candidate_id": "ZL-100", "dataset": a.dataset,
               "manifest": man, "results": res, "wall_clock_s": round(time.time() - t0, 1)},
              open(os.path.join(d, f"{a.dataset}_{a.renderer}.json"), "w"), indent=2, default=float)
    np.savez_compressed(os.path.join(d, f"{a.dataset}_{a.renderer}_per_pair.npz"),
                        origins=ori, **err)
    print("[done]", a.dataset, flush=True)


if __name__ == "__main__":
    main()
