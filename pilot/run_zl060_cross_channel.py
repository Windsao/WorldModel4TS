"""ZL-060 (hypothesis H-A) -- CROSS-CHANNEL analog retrieval at a single origin.

Every earlier candidate was channel-independent: neighbours were sought across TIME inside one
channel. The unifying failure signature of families F1-F4 is specifically about temporal
continuation and says nothing about cross-channel structure. electricity and traffic observe
hundreds of series at the SAME origin.

Mechanism. For query channel c at origin t, a donor is a pair (channel j, lag d) with d >= H, so
the donor's window [t-d-L, t-d) AND its continuation [t-d, t-d+H) are both fully observed at t.
Rank donors by similarity of their period-column token embeddings to the query's, then transfer
the top-k continuations, z-scored by each donor's own context and rescaled to the query's level.

Strictly zero-shot: every value used lies at or before the origin. Controls: random-init
backbone (identical architecture and code path), raw period-vector L2 ranking, same-channel-only
donors, and the ZL-051 blend.
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


def embed(model, win, P, batch=64):
    """win [N,L] -> [N,G,d] period-column token embeddings."""
    N, L = win.shape
    G = L // P
    C = torch.from_numpy(win[:, L - G * P:]).float().to(DEV)
    mu = C.mean(1, keepdim=True)
    sd = C.std(1, keepdim=True) + 1e-6
    z = ((C - mu) / (3 * sd)).clamp(-1, 1).view(N, G, P)
    return column_tokens(model, z, G, batch=batch).to(DEV), mu, sd


def run(model, series, origins, chans, P, L, H, lags, k=8, raw=False):
    """Returns prediction [n_pairs, H] for every (origin, query-channel) pair."""
    preds = []
    for t in origins:
        q = np.stack([series[c, t - L:t] for c in chans])              # [M,L] queries
        dw, df = [], []
        for d in lags:
            if t - d - L < 0:
                continue
            dw.append(np.stack([series[c, t - d - L:t - d] for c in chans]))
            df.append(np.stack([series[c, t - d:t - d + H] for c in chans]))
        if not dw:
            preds.append(np.zeros((len(chans), H)))
            continue
        D = np.concatenate(dw)                                          # [M*nlag, L]
        Df = np.concatenate(df)                                         # [M*nlag, H]
        with torch.no_grad():
            if raw:
                qq = torch.from_numpy(q).float().to(DEV)
                dd = torch.from_numpy(D).float().to(DEV)
                qmu, qsd = qq.mean(1, keepdim=True), qq.std(1, keepdim=True) + 1e-6
                dmu, dsd = dd.mean(1, keepdim=True), dd.std(1, keepdim=True) + 1e-6
                fq = F.normalize((qq - qmu) / qsd, dim=-1)
                fd = F.normalize((dd - dmu) / dsd, dim=-1)
            else:
                Eq, qmu, qsd = embed(model, q, P)
                Ed, dmu, dsd = embed(model, D, P)
                fq = F.normalize(Eq.reshape(len(q), -1), dim=-1)
                fd = F.normalize(Ed.reshape(len(D), -1), dim=-1)
            sim = fq @ fd.T                                             # [M, M*nlag]
            # a donor may not be the query itself at lag 0; lags are all >= H so this cannot
            # happen, but guard anyway against a donor window identical to the query window.
            top = sim.topk(min(k, fd.shape[0]), dim=1).indices
            fut = torch.from_numpy(Df).float().to(DEV)
            nb = fut[top]                                               # [M,k,H]
            nmu = dmu[top].squeeze(-1)[..., None]
            nsd = dsd[top].squeeze(-1)[..., None]
            zc = ((nb - nmu) / nsd).mean(1)                             # donor-normalised mean
            preds.append((zc * qsd + qmu).cpu().numpy())
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HORIZON))
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default=ZL.ROOT)
    ap.add_argument("--ckpt", default="MCG-NJU/videomae-base")
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--n-origins", type=int, default=60)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    data, borders, P, L, H, Xte, Yte, pairs, man = build_manifest(
        a.dataset, a.data_dir, a.max_ch, 8, 2000, a.seed)
    series = data.T                                                     # [M, T]
    M = min(a.max_ch, series.shape[0])
    chans = list(range(M))
    lags = [d for d in (H, H + P, H + 2 * P, H + 3 * P)]
    lo = borders[1] + L + max(lags)
    hi = borders[2] - H
    step = max(1, (hi - lo) // a.n_origins)
    origins = list(range(lo, hi, step))[:a.n_origins]
    print(f"[ZL-060] {a.dataset} M={M} P={P} L={L} H={H} lags={lags} "
          f"origins={len(origins)} donors/origin={M*len(lags)}", flush=True)
    ctx = np.stack([series[c, t - L:t] for t in origins for c in chans])
    fut = np.stack([series[c, t:t + H] for t in origins for c in chans])
    pr = {k_: v for k_, v in ZL.priors(ctx, P, H).items() if k_ in FAMILY}
    _, blend, _, _ = ZL.pseudo_origin_blend(ctx, P, H, pr, rule="pow_n", dense=True,
                                            min_ctx_periods=8)
    res = {"blend_mse": float(np.mean((blend - fut) ** 2)),
           "smean_mse": float(np.mean((ZL.priors(ctx, P, H)["smean"] - fut) ** 2)),
           "n_origins": len(origins), "n_donors_per_origin": M * len(lags), "lags": lags}
    from transformers import VideoMAEModel, VideoMAEConfig
    mp = VideoMAEModel.from_pretrained(a.ckpt).to(DEV).eval()
    torch.manual_seed(a.seed)
    mr = VideoMAEModel(VideoMAEConfig.from_pretrained(a.ckpt)).to(DEV).eval()
    err = {"blend": ((blend - fut) ** 2).mean(1)}
    Y = {"blend": blend}
    for nm, model, raw in (("pt", mp, False), ("rand", mr, False), ("rawL2", None, True)):
        y = run(model, series, origins, chans, P, L, H, lags, k=a.k, raw=raw)
        Y[nm] = y
        res[f"{nm}_mse"] = float(np.mean((y - fut) ** 2))
        err[nm] = ((y - fut) ** 2).mean(1)
        yb = 0.5 * y + 0.5 * blend
        res[f"{nm}_half_blend_mse"] = float(np.mean((yb - fut) ** 2))
        err[f"{nm}_half"] = ((yb - fut) ** 2).mean(1)
        print(f"  {nm:6s} alone={res[f'{nm}_mse']:.4f} half-blend={res[f'{nm}_half_blend_mse']:.4f}",
              flush=True)

    # ---- THE DECISIVE TEST for the goal "does the video backbone add anything?"
    # Equal-weight ensembles (parameter-free). A is the best method containing NO video model.
    # B adds the pretrained backbone to it; C adds an identical untrained one.
    ens = {"A_blend_rawL2": (Y["blend"] + Y["rawL2"]) / 2,
           "B_plus_pretrained": (Y["blend"] + Y["rawL2"] + Y["pt"]) / 3,
           "C_plus_random": (Y["blend"] + Y["rawL2"] + Y["rand"]) / 3}
    for k_, v_ in ens.items():
        res[f"{k_}_mse"] = float(np.mean((v_ - fut) ** 2))
        err[k_] = ((v_ - fut) ** 2).mean(1)
    res["B_over_A"] = res["B_plus_pretrained_mse"] / res["A_blend_rawL2_mse"]
    res["B_over_C"] = res["B_plus_pretrained_mse"] / res["C_plus_random_mse"]
    print(f"  ENSEMBLE  A(no video)={res['A_blend_rawL2_mse']:.4f} "
          f"B(+pretrained)={res['B_plus_pretrained_mse']:.4f} "
          f"C(+random)={res['C_plus_random_mse']:.4f} "
          f"| B/A={res['B_over_A']:.4f} B/C={res['B_over_C']:.4f}", flush=True)
    res["pt_over_rand"] = res["pt_mse"] / res["rand_mse"]
    res["pt_over_rawL2"] = res["pt_mse"] / res["rawL2_mse"]
    res["pt_half_over_blend"] = res["pt_half_blend_mse"] / res["blend_mse"]
    ori = np.repeat(np.array(origins), M)
    blk = int(np.ceil((L + H) / max(1, step)))
    res["boot_pt_vs_rand"] = ZL.paired_bootstrap(err["pt"], err["rand"], ori, max(1, blk))
    res["boot_pthalf_vs_blend"] = ZL.paired_bootstrap(err["pt_half"], err["blend"], ori, max(1, blk))
    res["boot_B_vs_A"] = ZL.paired_bootstrap(err["B_plus_pretrained"], err["A_blend_rawL2"],
                                             ori, max(1, blk))
    res["boot_B_vs_C"] = ZL.paired_bootstrap(err["B_plus_pretrained"], err["C_plus_random"],
                                             ori, max(1, blk))
    print("B vs A CI:", json.dumps(res["boot_B_vs_A"], default=float), flush=True)
    print("B vs C CI:", json.dumps(res["boot_B_vs_C"], default=float), flush=True)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("boot")},
                     indent=2, default=float), flush=True)
    print("pt/rand CI:", res["boot_pt_vs_rand"], flush=True)
    out = {"status": "complete", "candidate_id": "ZL-060", "hypothesis": "H-A cross-channel",
           "dataset": a.dataset, "manifest": man, "results": res,
           "wall_clock_s": round(time.time() - t0, 1)}
    d = os.path.join(a.root, "zl060"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"{a.dataset}.json"), "w"), indent=2, default=float)
    np.savez_compressed(os.path.join(d, f"{a.dataset}_per_pair.npz"), origins=ori, **err)
    print("[done]", a.dataset, flush=True)


if __name__ == "__main__":
    main()
