"""ZL-030: deterministic token kernel in PRIOR-RESIDUAL space.

The kernel retrieves what the prior GOT WRONG on historical windows instead of retrieving raw
successors, so the retrieved quantity is already a correction and no safety shrinkage is needed.
Strict zero-shot: every key, value and prior is computed from periods at or before the origin.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zeroshot_loop as ZL
from run_zl020_token_kernel import period_matrix_video, column_tokens

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID = 14


def prior_at(z, e, reps, kind):
    """Prior forecast of periods [e, e+reps) using ONLY periods [0, e). z [N,G,P]."""
    if kind == "smean":
        p = z[:, :e].mean(1)
    elif kind == "snaive":
        p = z[:, e - 1]
    elif kind == "recent_mean":
        p = z[:, e - 1].mean(-1, keepdim=True).expand(-1, z.shape[-1])
    else:
        p = z[:, e - 1, -1:].expand(-1, z.shape[-1])          # last value
    return p.unsqueeze(1).expand(-1, reps, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default=ZL.ROOT)
    ap.add_argument("--ckpt", default=os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base"))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    X, pairs, man = ZL.historical_manifest(args.dataset, args.data_dir, n_max=args.n,
                                           stride=args.stride, seed=args.seed, split="audit")
    P, L, H = man["P"], man["L"], man["H"]
    G, reps, N = L // P, H // P, len(X)
    ctx, fut = X[:, :L], X[:, L:L + H]
    Ct = torch.from_numpy(ctx).to(DEV)
    mu = Ct.mean(1, keepdim=True); sd = Ct.std(1, keepdim=True) + 1e-6
    z = ((Ct - mu) / (3 * sd)).clamp(-1, 1).view(N, G, P)
    futz = ((torch.from_numpy(fut).to(DEV) - mu) / (3 * sd)).clamp(-1, 1).view(N, reps, P)

    starts = [s for s in range(0, G - args.m - reps + 1)]
    q_start = G - args.m
    print(f"{args.dataset} n={N} P={P} G={G} reps={reps} keys={len(starts)} "
          f"origins={man['n_origins']}", flush=True)

    from transformers import VideoMAEModel, VideoMAEConfig
    mp = VideoMAEModel.from_pretrained(args.ckpt).to(DEV).eval()
    torch.manual_seed(args.seed)
    mr = VideoMAEModel(VideoMAEConfig.from_pretrained(args.ckpt)).to(DEV).eval()
    emb = {"pt_col": column_tokens(mp, z, G, pool="column").to(DEV),
           "rand_col": column_tokens(mr, z, G, pool="column").to(DEV),
           "raw_period": F.normalize(z, dim=-1)}

    # per-sample prior family selected from internal pseudo-origins (context-only)
    pr = ZL.priors(ctx, P, H)
    names, base, _ = ZL.pseudo_origin_select(ctx, P, H, pr)
    kinds = sorted(set(names.tolist()))
    sel = {k: torch.from_numpy(names == k).to(DEV) for k in kinds}

    def prior_mix(e):
        out = torch.zeros(N, reps, P, device=DEV)
        for k in kinds:
            out[sel[k]] = prior_at(z, e, reps, k)[sel[k]]
        return out

    # residual values: what the prior got WRONG on each historical window
    resid = torch.stack([z[:, s + args.m:s + args.m + reps] - prior_mix(s + args.m)
                         for s in starts], 1)                       # [N,K,reps,P]
    pq = prior_mix(G)                                               # prior for the real future
    res = {"prior_pick_counts": {k: int((names == k).sum()) for k in kinds}}
    for nm, E in emb.items():
        q_ = F.normalize(E[:, q_start:q_start + args.m].mean(1), dim=-1)
        kk = F.normalize(torch.stack([E[:, s:s + args.m].mean(1) for s in starts], 1), dim=-1)
        sim = torch.einsum("nd,nkd->nk", q_, kk)
        top = sim.topk(min(args.k, len(starts)), dim=1).indices
        corr = torch.gather(resid, 1, top[:, :, None, None].expand(-1, -1, reps, P)).mean(1)
        predz = (pq + corr).clamp(-1, 1)
        pred = (predz.reshape(N, H) * 3 * sd + mu).cpu().numpy()
        res[f"{nm}_ts_mse"] = float(np.mean((pred - fut) ** 2))
        res[f"{nm}_percase"] = ((pred - fut) ** 2).mean(1)
        res[f"{nm}_resid_norm_mse"] = float(((corr - (futz - pq)) ** 2).mean())
    base_pred = (pq.reshape(N, H) * 3 * sd + mu).cpu().numpy()
    res["prior_only_ts_mse"] = float(np.mean((base_pred - fut) ** 2))
    res["prior_only_percase"] = ((base_pred - fut) ** 2).mean(1)
    res["base_selected_ts_mse"] = float(np.mean((base - fut) ** 2))
    res["base_selected_percase"] = ((base - fut) ** 2).mean(1)
    blk = int(np.ceil((L + H) / man["stride"]))
    for b in ("rand_col", "raw_period", "prior_only", "base_selected"):
        res[f"boot_pt_vs_{b}"] = ZL.paired_bootstrap(res["pt_col_percase"], res[f"{b}_percase"],
                                                     pairs[:, 0], blk)
    show = {k: v for k, v in res.items() if not k.endswith("_percase")}
    print(json.dumps(show, indent=2, default=float), flush=True)
    out = {"status": "complete", "candidate_id": "ZL-030", "dataset": args.dataset,
           "split": "audit", "manifest": man, "checkpoint": args.ckpt, "m": args.m, "k": args.k,
           "n_keys": len(starts), "config_hash": ZL.cfg_hash(vars(args)), "results": show,
           "git_commit": os.environ.get("WM4TS_GIT_COMMIT", "unknown"),
           "wall_clock_s": round(time.time() - t0, 1)}
    d = os.path.join(args.root, "candidates", "ZL-030"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"metrics_{args.dataset}.json"), "w"), indent=2, default=float)
    np.savez_compressed(os.path.join(d, f"per_case_{args.dataset}.npz"), pairs=pairs,
                        **{k: v for k, v in res.items() if k.endswith("_percase")})
    print("[done]", args.dataset, flush=True)


if __name__ == "__main__":
    main()
