"""ZL-080 -- renderer sweep x in-context ridge: does ANY preprocessing make pretraining pay?

Diagnosis this tests. Every feature-route experiment so far fed VideoMAE a static image repeated
over 16 frames -- measured inter-frame difference exactly 0.0000. The backbone's only pretrained
advantage is temporal, and the frame-permutation audit showed it does read the frame axis
(11% cost, CI excluding 1). So the null result may be a rendering artefact, not a fact about
the representation.

Each renderer maps G periods to G frames so that frame f is period f and motion across frames is
the evolution of the seasonal shape. Value is encoded as POSITION where possible, because
position is what Kinetics pretraining preserves and fine amplitude texture is what it discards.

Decision statistic: pretrained / random-init under an identical in-context ridge readout. A
renderer for which this drops clearly below 1 is the first evidence of transferable pretrained
structure and gets promoted to a full run.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "third_party", "VisionTS"))
import zeroshot_loop as ZL
from video_renderers_motion import RENDERERS
from run_zl070_incontext_ridge import ridge_fit_predict
from run_visionts_reference import build_manifest, HORIZON

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAMILY = ["smean", "snaive", "last", "recent_mean"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def video_feats(model, win, P, renderer, batch=8):
    """win [N,L] -> (X [N,3d] float64, mu, sd) via per-TEMPORAL-POSITION token means.

    Aggregation is per temporal position, never a single global mean: the ZL-021 audit measured
    pt_col / pt_meanpool = 0.5979, i.e. global pooling destroys most of the signal.
    """
    N, L = win.shape
    G = L // P
    C = torch.from_numpy(win[:, L - G * P:]).float().to(DEV)
    mu = C.mean(1, keepdim=True)
    sd = C.std(1, keepdim=True) + 1e-6
    z = ((C - mu) / (3 * sd)).clamp(-1, 1).view(N, G, P)
    fn = RENDERERS[renderer]
    out = []
    m_, s_ = MEAN.to(DEV), STD.to(DEV)
    for i in range(0, N, batch):
        v = fn(z[i:i + batch])                                   # [b,T,3,224,224] in [0,1]
        v = (v - m_) / s_
        h = model(pixel_values=v).last_hidden_state               # [b, T/2*14*14, d]
        b, n, d = h.shape
        t = n // (14 * 14)
        e = h.view(b, t, 14 * 14, d).mean(2)                      # [b, t, d] per temporal pos
        out.append(torch.cat([e.mean(1), e[:, -1], e[:, -2]], 1).double().cpu())
        del v, h, e
    return torch.cat(out).numpy(), mu.cpu().numpy(), sd.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HORIZON))
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default=ZL.ROOT)
    ap.add_argument("--ckpt", default="MCG-NJU/videomae-base")
    ap.add_argument("--max-ch", type=int, default=64)
    ap.add_argument("--n-origins", type=int, default=12)
    ap.add_argument("--n-pseudo", type=int, default=6)
    ap.add_argument("--renderers", nargs="+", default=list(RENDERERS))
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
    ctx = np.stack([series[c, t - L:t] for t in origins for c in range(M)])
    fut = np.stack([series[c, t:t + H] for t in origins for c in range(M)])
    pr = {k_: v for k_, v in ZL.priors(ctx, P, H).items() if k_ in FAMILY}
    _, blend, _, _ = ZL.pseudo_origin_blend(ctx, P, H, pr, rule="pow_n", dense=True,
                                            min_ctx_periods=8)
    mb = float(np.mean((blend - fut) ** 2))
    print(f"[ZL-080] {a.dataset} M={M} P={P} G={L//P} origins={len(origins)} "
          f"pairs/origin={M*a.n_pseudo} blend={mb:.4f}", flush=True)

    from transformers import VideoMAEModel, VideoMAEConfig
    mp = VideoMAEModel.from_pretrained(a.ckpt).to(DEV).eval()
    torch.manual_seed(a.seed)
    mr = VideoMAEModel(VideoMAEConfig.from_pretrained(a.ckpt)).to(DEV).eval()
    res = {"blend_mse": mb, "n_origins": len(origins), "M": M, "renderers": {}}
    ori = np.repeat(np.array(origins), M)
    blk = max(1, int(np.ceil((L + H) / max(1, step))))
    perr = {}

    for rname in a.renderers:
        row = {}
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
                    Xtr, mtr, str_ = video_feats(model, W, P, rname)
                    Q = np.stack([series[c, t - L:t] for c in range(M)])
                    Xte, mte, ste = video_feats(model, Q, P, rname)
                zp, _ = ridge_fit_predict(Xtr, (Y - mtr) / (3 * str_), Xte, seed=a.seed)
                preds.append(zp * 3 * ste + mte)
            y = np.concatenate(preds)
            row[f"{nm}_mse"] = float(np.mean((y - fut) ** 2))
            row[f"{nm}_half_blend"] = float(np.mean((0.5 * y + 0.5 * blend - fut) ** 2))
            perr[nm] = ((y - fut) ** 2).mean(1)
        row["pt_over_rand"] = row["pt_mse"] / row["rand_mse"]
        row["boot_pt_vs_rand"] = ZL.paired_bootstrap(perr["pt"], perr["rand"], ori, blk)
        row["pt_over_blend"] = row["pt_mse"] / mb
        row["pthalf_over_blend"] = row["pt_half_blend"] / mb
        res["renderers"][rname] = row
        print(f"  {rname:16s} pt={row['pt_mse']:.4f} rand={row['rand_mse']:.4f} "
              f"pt/rand={row['pt_over_rand']:.4f} pt/blend={row['pt_over_blend']:.4f} "
              f"half/blend={row['pthalf_over_blend']:.4f} "
              f"CI=[{row['boot_pt_vs_rand']['lo']:.3f},{row['boot_pt_vs_rand']['hi']:.3f}]"
              f"{'*' if row['boot_pt_vs_rand']['hi'] < 1 else ''}", flush=True)

    best = min(res["renderers"], key=lambda k: res["renderers"][k]["pt_over_rand"])
    res["best_renderer_by_pt_over_rand"] = best
    res["best_pt_over_rand"] = res["renderers"][best]["pt_over_rand"]
    print(f"\n  BEST pt/rand: {best} = {res['best_pt_over_rand']:.4f}", flush=True)
    d = os.path.join(a.root, "zl080"); os.makedirs(d, exist_ok=True)
    json.dump({"status": "complete", "candidate_id": "ZL-080", "dataset": a.dataset,
               "manifest": man, "results": res, "wall_clock_s": round(time.time() - t0, 1)},
              open(os.path.join(d, f"{a.dataset}.json"), "w"), indent=2, default=float)
    print("[done]", a.dataset, flush=True)


if __name__ == "__main__":
    main()
