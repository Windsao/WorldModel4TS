"""Locked ZL-024 method — Stage E (historical) and Stage F (benchmark) evaluation.

Strict zero-shot: frozen backbone, zero trained parameters, every statistic from the observed
context of each origin. The configuration was fixed on ETTh2/ETTm2 historical audit windows and
is serialized in locked_final_config.json before any six-dataset run.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "third_party", "VisionTS"))
import zeroshot_loop as ZL
import run_field as RF
from run_zl020_token_kernel import period_matrix_video, column_tokens
from run_visionts_reference import build_manifest, gather, HORIZON

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID = 14


def locked_method(E, z, ctx, fut, P, H, G, reps, m, k, base, mu, sd):
    """E [N,G,d] column-token embeddings -> (prediction, rho). Deterministic, no parameters."""
    N = z.shape[0]
    starts = [s for s in range(0, G - m - reps + 1)]
    q_start = G - m
    vals = torch.stack([z[:, s + m:s + m + reps] for s in starts], 1)
    q = F.normalize(E[:, q_start:q_start + m].mean(1), dim=-1)
    kk = F.normalize(torch.stack([E[:, s:s + m].mean(1) for s in starts], 1), dim=-1)
    sim = torch.einsum("nd,nkd->nk", q, kk)
    top = sim.topk(min(k, len(starts)), dim=1).indices
    nb = torch.gather(vals, 1, top[:, :, None, None].expand(-1, -1, reps, P))
    pick = nb.mean(1)
    disp = nb.var(dim=1, unbiased=False).mean(dim=(1, 2))
    tot = nb.mean(1).var(dim=(1, 2), unbiased=False) + disp + 1e-8
    rho = (1.0 - disp / tot).clamp(0, 1).cpu().numpy()[:, None]
    cand = (pick.reshape(N, reps * P)[:, :H] * 3 * sd + mu).cpu().numpy()
    cap = 3 * np.abs(np.diff(ctx, axis=1)).mean(1, keepdims=True) * math.sqrt(H)
    return base + rho * np.clip(cand - base, -cap, cap), rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HORIZON))
    ap.add_argument("--stage", required=True, choices=["E", "F"])
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default=ZL.ROOT)
    ap.add_argument("--ckpt", default=os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base"))
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    if args.stage == "F":
        data, borders, P, L, H, Xte, Yte, pairs, man = build_manifest(
            args.dataset, args.data_dir, args.max_ch, args.stride, args.n, args.seed)
        Xu, Yu = gather(Xte, Yte, pairs)
        ctx, fut = Xu[:, :, 0], Yu[:, :, 0]
    else:
        X, pairs, man = ZL.historical_manifest(args.dataset, args.data_dir, args.max_ch,
                                               stride=args.stride, n_max=args.n,
                                               seed=args.seed, split="audit")
        P, L, H = man["P"], man["L"], man["H"]
        ctx, fut = X[:, :L], X[:, L:L + H]
    G, reps, N = L // P, H // P, len(ctx)
    print(f"[{args.stage}] {args.dataset} n={N} P={P} L={L} H={H} G={G} reps={reps}", flush=True)

    Ct = torch.from_numpy(ctx).to(DEV)
    mu = Ct.mean(1, keepdim=True); sd = Ct.std(1, keepdim=True) + 1e-6
    z = ((Ct - mu) / (3 * sd)).clamp(-1, 1).view(N, G, P)
    allp = ZL.priors(ctx, P, H)
    FAMILY = ["smean", "snaive", "last", "recent_mean"]
    pr = {k: allp[k] for k in FAMILY}
    # v3 combiner, locked on HISTORICAL audit windows of all six datasets (benchmark untouched):
    #   - model AVERAGING, not argmin selection (argmin lost to a fixed smean on 5 of 6)
    #   - DENSE pseudo-origins rolled one period at a time, >= 8 periods of context each
    #   - EVIDENCE-SCALED weights w_p propto (L_min/L_p)^n_origins, which sharpen toward argmin
    #     as evidence accumulates. Unit-temperature rules were too flat where one prior
    #     dominates by ~6x (electricity/traffic/solar) and lost to a fixed smean there.
    w, base, pnames, n_po = ZL.pseudo_origin_blend(ctx, P, H, pr, rule="pow_n", dense=True,
                                                   min_ctx_periods=8)
    names, sel_base, _ = ZL.pseudo_origin_select(ctx, P, H, pr)

    from transformers import VideoMAEModel, VideoMAEConfig
    mp = VideoMAEModel.from_pretrained(args.ckpt).to(DEV).eval()
    torch.manual_seed(args.seed)
    mr = VideoMAEModel(VideoMAEConfig.from_pretrained(args.ckpt)).to(DEV).eval()
    res, err = {}, {}
    for nm, model in (("pt", mp), ("rand", mr)):
        E = column_tokens(model, z, G, pool="column").to(DEV)
        y, rho = locked_method(E, z, ctx, fut, P, H, G, reps, args.m, args.k, base, mu, sd)
        res[f"{nm}_mse"] = float(np.mean((y - fut) ** 2))
        res[f"{nm}_mae"] = float(np.mean(np.abs(y - fut)))
        err[nm] = ((y - fut) ** 2).mean(1)
        res[f"{nm}_rho_mean"] = float(rho.mean())
    res["base_mse"] = float(np.mean((base - fut) ** 2)); err["base"] = ((base - fut) ** 2).mean(1)
    for kk_, v in allp.items():
        res[f"prior_{kk_}_mse"] = float(np.mean((v - fut) ** 2))
    res["prior_pick_counts"] = {kk_: int((names == kk_).sum()) for kk_ in set(names.tolist())}
    res["argmin_base_mse"] = float(np.mean((sel_base - fut) ** 2))
    res["blend_weight_mean"] = {n_: float(w[:, j].mean()) for j, n_ in enumerate(pnames)}
    res["n_pseudo_origins"] = int(n_po)
    res["smean_mse"] = float(np.mean((allp["smean"] - fut) ** 2))
    err["blend"] = ((base - fut) ** 2).mean(1)
    if args.stage == "F":
        cands = [os.path.join(r_, "visionts_reference", f"visionts_{args.dataset}.json")
                 for r_ in ("pilot/results_field/video_visionts",
                            os.path.join(os.path.dirname(args.root.rstrip("/")), "video_visionts"))]
        vp = next(c for c in cands if os.path.exists(c))
        vj = json.load(open(vp))
        assert vj["manifest"]["pairs_sha1"] == man["pairs_sha1"], "pair manifest mismatch vs VisionTS"
        res["visionts_mse"] = vj["metrics"]["visionts_pretrained"]
        res["visionts_pairs_sha1"] = vj["manifest"]["pairs_sha1"]
    blk = int(np.ceil((L + H) / args.stride))
    for b in ("rand", "base"):
        res[f"boot_pt_vs_{b}"] = ZL.paired_bootstrap(err["pt"], err[b], pairs[:, 0], blk)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, np.ndarray)},
                     indent=2, default=float), flush=True)
    out = {"status": "complete", "candidate_id": "ZL-050", "stage": args.stage,
           "dataset": args.dataset, "manifest": man, "checkpoint": args.ckpt,
           "m": args.m, "k": args.k, "combiner": "pow_n-dense8-base4", "results": res,
           "git_commit": os.environ.get("WM4TS_GIT_COMMIT", "unknown"),
           "wall_clock_s": round(time.time() - t0, 1)}
    d = os.path.join(args.root, f"stage{args.stage}"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"{args.dataset}.json"), "w"), indent=2, default=float)
    np.savez_compressed(os.path.join(d, f"{args.dataset}_per_pair.npz"), pairs=pairs, **err)
    print("[done]", args.dataset, flush=True)


if __name__ == "__main__":
    main()
