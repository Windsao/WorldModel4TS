"""ZL-020: deterministic token kernel over the period matrix (Family 3).

Queries = frozen tokens of the most recent periods; keys = tokens of earlier periods; values =
their already-observed successor periods. Similarity is a fixed cosine; nothing is trained and
no learned projection exists. Every key and every value lies inside the observed context.
"""
import argparse, json, os, sys, time
import math
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zeroshot_loop as ZL
from preprocess_renderers import IMN_MEAN, IMN_STD

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMG, NF, GRID = 224, 16, 14


def period_matrix_video(z):
    """z [N, G, P] normalized periods -> [N,16,3,224,224] period-matrix video.

    rows = phase, cols = period; the image is repeated over frames, matching the render that
    made the supervised cross-attention readout work.
    """
    N, G, P = z.shape
    m = z.permute(0, 2, 1).unsqueeze(1)                       # [N,1,P,G] rows=phase cols=period
    img = F.interpolate(m, size=(IMG, IMG), mode="bilinear", align_corners=False)
    g = ((img.squeeze(1) + 1) / 2).clamp(0, 1)
    vid = g.unsqueeze(1).unsqueeze(1).expand(N, NF, 3, IMG, IMG)
    return ((vid - IMN_MEAN.to(g.device).unsqueeze(1)) / IMN_STD.to(g.device).unsqueeze(1)).contiguous()


def column_tokens(model, z, G, batch=16, pool="column"):
    """-> [N, G, d]: token embeddings aggregated per PERIOD column (not globally pooled).

    Renders in CHUNKS: materializing 3000 videos at once needs ~29 GB and OOMs.
    """
    outs = []
    with torch.no_grad():
        for i in range(0, len(z), batch):
            vids = period_matrix_video(z[i:i + batch])
            h = model(pixel_values=vids).last_hidden_state                 # [b,1568,d]
            del vids
            b, n, d = h.shape
            t = h.view(b, NF // 2, GRID, GRID, d).mean(1)                  # collapse tubelets
            if pool == "global":
                e = t.mean((1, 2)).unsqueeze(1).expand(b, G, d)
            else:
                col = t.mean(1)                                            # [b,14,d] per token column
                idx = (torch.arange(G, device=h.device).float() * (GRID - 1) / max(G - 1, 1)).round().long()
                e = col[:, idx]                                            # [b,G,d] map period -> column
            outs.append(F.normalize(e, dim=-1).cpu())
    return torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default=ZL.ROOT)
    ap.add_argument("--ckpt", default=os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base"))
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--m", type=int, default=2, help="query length in periods")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    X, pairs, man = ZL.historical_manifest(args.dataset, args.data_dir, n_max=args.n,
                                           stride=args.stride, seed=args.seed, split="audit")
    P, L, H = man["P"], man["L"], man["H"]
    G = L // P
    reps = H // P
    assert reps >= 1
    ctx, fut = X[:, :L], X[:, L:L + H]
    N = len(X)
    Ct = torch.from_numpy(ctx).to(DEV)
    mu = Ct.mean(1, keepdim=True); sd = Ct.std(1, keepdim=True) + 1e-6
    z = ((Ct - mu) / (3 * sd)).clamp(-1, 1).view(N, G, P)
    futz = ((torch.from_numpy(fut).to(DEV) - mu) / (3 * sd)).clamp(-1, 1).view(N, reps, P)

    # key windows: m consecutive periods whose SUCCESSOR reps periods are also observed
    starts = [s for s in range(0, G - args.m - reps + 1)]
    q_start = G - args.m                                   # query = the most recent m periods
    assert starts and max(starts) + args.m + reps <= q_start + args.m
    print(f"{args.dataset} n={N} P={P} G={G} reps={reps} m={args.m} keys={len(starts)}", flush=True)

    from transformers import VideoMAEModel, VideoMAEConfig
    mp = VideoMAEModel.from_pretrained(args.ckpt).to(DEV).eval()
    torch.manual_seed(args.seed)
    mr = VideoMAEModel(VideoMAEConfig.from_pretrained(args.ckpt)).to(DEV).eval()

    emb = {}
    for nm, model, pool in (("pt_col", mp, "column"), ("rand_col", mr, "column"),
                            ("pt_meanpool", mp, "global")):
        emb[nm] = column_tokens(model, z, G, pool=pool).to(DEV)
    # raw period-vector kernel
    emb["raw_period"] = F.normalize(z, dim=-1)

    vals = torch.stack([z[:, s + args.m:s + args.m + reps] for s in starts], 1)   # [N,K,reps,P]
    tgt = futz                                                                     # [N,reps,P]
    res = {}
    for nm, E in emb.items():
        q = E[:, q_start:q_start + args.m].mean(1)                                 # [N,d]
        kk = torch.stack([E[:, s:s + args.m].mean(1) for s in starts], 1)          # [N,K,d]
        sim = torch.einsum("nd,nkd->nk", F.normalize(q, dim=-1), F.normalize(kk, dim=-1))
        top = sim.topk(min(args.k, len(starts)), dim=1).indices
        pick = torch.gather(vals, 1, top[:, :, None, None].expand(-1, -1, reps, P)).mean(1)
        res[f"{nm}_norm_succ_mse"] = float(((pick - tgt) ** 2).mean())
        res[f"{nm}_percase"] = ((pick - tgt) ** 2).mean(dim=(1, 2)).cpu().numpy()
        pred = (pick.view(N, H) * 3 * sd + mu).cpu().numpy()
        res[f"{nm}_ts_mse"] = float(np.mean((pred - fut) ** 2))
    best = ((vals - tgt.unsqueeze(1)) ** 2).mean(dim=(2, 3)).min(1).values
    res["oracle_best_key_norm_mse"] = float(best.mean())
    pr = ZL.priors(ctx, P, H)

    # ---- ZL-021: safe residual on the best context-only prior.
    # alpha is chosen per sample at an INTERNAL pseudo-origin one step back: query and keys are
    # shifted one successor-block earlier, so its target is already observed. The embeddings are
    # reused, so this costs no extra forward pass and reads nothing at or after the origin.
    names, base, used = ZL.pseudo_origin_select(ctx, P, H, pr)
    res["prior_pick_counts"] = {k: int((names == k).sum()) for k in set(names.tolist())}
    ps = q_start - reps                                   # pseudo query start (m periods)
    pkeys = [t for t in starts if t + args.m + reps <= ps]
    alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
    if pkeys:
        pv = torch.stack([z[:, t + args.m:t + args.m + reps] for t in pkeys], 1)
        ptgt = z[:, ps + args.m:ps + args.m + reps]
        sub = ctx[:, :L - H]
        pbase_n = ZL.priors(sub, P, H)["smean"]
        pbase = torch.from_numpy(pbase_n).to(DEV).view(N, reps, P)
        pbase = ((pbase - mu.view(N, 1, 1)) / (3 * sd.view(N, 1, 1))).clamp(-1, 1)
    for nm, E in emb.items():
        cand_n = None
        if pkeys:
            pq = E[:, ps:ps + args.m].mean(1)
            pk = torch.stack([E[:, t:t + args.m].mean(1) for t in pkeys], 1)
            psim = torch.einsum("nd,nkd->nk", F.normalize(pq, dim=-1), F.normalize(pk, dim=-1))
            ptop = psim.topk(min(args.k, len(pkeys)), dim=1).indices
            ppick = torch.gather(pv, 1, ptop[:, :, None, None].expand(-1, -1, reps, P)).mean(1)
            perr = np.stack([(((pbase + a * (ppick - pbase)) - ptgt) ** 2).mean(dim=(1, 2)).cpu().numpy()
                             for a in alphas], 1)
            pick_a = perr.argmin(1)
        else:
            pick_a = np.zeros(N, dtype=int)
        a_sel = np.array([alphas[i] for i in pick_a])[:, None]
        q_ = E[:, q_start:q_start + args.m].mean(1)
        kk_ = torch.stack([E[:, s:s + args.m].mean(1) for s in starts], 1)
        sim_ = torch.einsum("nd,nkd->nk", F.normalize(q_, dim=-1), F.normalize(kk_, dim=-1))
        top_ = sim_.topk(min(args.k, len(starts)), dim=1).indices
        pick_ = torch.gather(vals, 1, top_[:, :, None, None].expand(-1, -1, reps, P)).mean(1)
        # ---- ZL-023: parameter-free neighbour-agreement shrinkage.
        # The pseudo-origin alpha is noisy when few keys exist (ETTh2 has 5). Agreement among the
        # retrieved successors is a deterministic, context-only confidence that needs no selection:
        # when the k neighbours disagree, shrink toward the prior.
        nb = torch.gather(vals, 1, top_[:, :, None, None].expand(-1, -1, reps, P))     # [N,k,reps,P]
        disp = nb.var(dim=1, unbiased=False).mean(dim=(1, 2))                          # spread
        tot = nb.mean(1).var(dim=(1, 2), unbiased=False) + disp + 1e-8
        rho = (1.0 - disp / tot).clamp(0, 1).cpu().numpy()[:, None]
        res[f"{nm}_rho_mean"] = float(rho.mean())
        cand_ts = (pick_.view(N, H) * 3 * sd + mu).cpu().numpy()
        res[f"{nm}_cand_ts"] = cand_ts
        cap = 3 * np.abs(np.diff(ctx, axis=1)).mean(1, keepdims=True) * math.sqrt(H)
        y_rho = base + rho * np.clip(cand_ts - base, -cap, cap)
        res[f"{nm}_rho_ts_mse"] = float(np.mean((y_rho - fut) ** 2))
        res[f"{nm}_rho_percase"] = ((y_rho - fut) ** 2).mean(1)
        y = base + a_sel * np.clip(cand_ts - base, -3 * np.abs(np.diff(ctx, axis=1)).mean(1, keepdims=True) * math.sqrt(H),
                                    3 * np.abs(np.diff(ctx, axis=1)).mean(1, keepdims=True) * math.sqrt(H))
        res[f"{nm}_wrapped_ts_mse"] = float(np.mean((y - fut) ** 2))
        res[f"{nm}_wrapped_percase"] = ((y - fut) ** 2).mean(1)
        res[f"{nm}_alpha_nonzero"] = float((a_sel > 0).mean())
    # ---- ZL-031 ORACLE CEILING (diagnostic only, uses the target; never a forecast).
    # Upper bound on what ANY safe per-origin blend of prior and kernel could achieve, and
    # therefore an upper bound on attributable gain.
    grid = np.linspace(0.0, 1.0, 21)
    for nm in ("pt_col", "rand_col", "raw_period"):
        cand_o = res[f"{nm}_cand_ts"]
        cap_o = 3 * np.abs(np.diff(ctx, axis=1)).mean(1, keepdims=True) * math.sqrt(H)
        r_o = np.clip(cand_o - base, -cap_o, cap_o)
        errs = np.stack([(((base + a * r_o) - fut) ** 2).mean(1) for a in grid], 1)
        res[f"{nm}_oracle_ts_mse"] = float(errs.min(1).mean())
        res[f"{nm}_oracle_percase"] = errs.min(1)
        res[f"{nm}_oracle_alpha_mean"] = float(grid[errs.argmin(1)].mean())
    res["base_ts_mse"] = float(np.mean((base - fut) ** 2))
    res["base_percase"] = ((base - fut) ** 2).mean(1)
    for k, v in pr.items():
        res[f"prior_{k}_ts_mse"] = float(np.mean((v - fut) ** 2))
    blk = int(np.ceil((L + H) / man["stride"]))
    for b in ("rand_col", "raw_period", "pt_meanpool"):
        res[f"boot_pt_col_vs_{b}"] = ZL.paired_bootstrap(res["pt_col_percase"], res[f"{b}_percase"],
                                                         pairs[:, 0], blk)
    for b in ("rand_col_rho", "base_rho"):
        tgt_e = res["rand_col_rho_percase"] if b == "rand_col_rho" else res["base_percase"]
        res[f"boot_rho_pt_vs_{b}"] = ZL.paired_bootstrap(res["pt_col_rho_percase"], tgt_e, pairs[:, 0], blk)
    for b in ("rand_col_wrapped", "base"):
        res[f"boot_wrapped_pt_vs_{b}"] = ZL.paired_bootstrap(
            res["pt_col_wrapped_percase"], res[f"{b}_percase" if b == "base" else "rand_col_wrapped_percase"],
            pairs[:, 0], blk)
    for k in [k for k in list(res) if k.endswith("_cand_ts")]:
        res.pop(k)
    show = {k: v for k, v in res.items()
            if not k.endswith("_percase") and not isinstance(v, np.ndarray)}
    print(json.dumps(show, indent=2, default=float), flush=True)
    out = {"status": "complete", "candidate_id": "ZL-020", "dataset": args.dataset,
           "split": "audit", "manifest": man, "checkpoint": args.ckpt, "m": args.m, "k": args.k,
           "n_keys": len(starts), "config_hash": ZL.cfg_hash(vars(args)), "results": show,
           "git_commit": os.environ.get("WM4TS_GIT_COMMIT", "unknown"),
           "wall_clock_s": round(time.time() - t0, 1)}
    d = os.path.join(args.root, "candidates", "ZL-020"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"metrics_{args.dataset}.json"), "w"), indent=2, default=float)
    np.savez_compressed(os.path.join(d, f"per_case_{args.dataset}.npz"), pairs=pairs,
                        **{k: v for k, v in res.items() if k.endswith("_percase")})
    print("[done]", args.dataset, flush=True)


if __name__ == "__main__":
    main()
