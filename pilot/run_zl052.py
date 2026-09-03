"""ZL-052 -- the video kernel enters the evidence-scaled blend as a FIFTH CANDIDATE.

Stage E v3 showed the kernel helps on electricity/traffic/ETTm2 and hurts on solar/ETTh1, so
bolting it on unconditionally is a net loss. Here it is scored by exactly the same dense
pseudo-origin machinery as the four deterministic priors and receives a per-sample weight.

Leak discipline: the kernel forecast at a pseudo-origin is computed by RE-EMBEDDING the
truncated context. Reusing the full-context embedding would let the pseudo-future leak into the
statistic that decides the kernel's weight.
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
from run_visionts_reference import build_manifest, gather, HORIZON

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAMILY = ["smean", "snaive", "last", "recent_mean"]


def kernel_forecast(model, ctx, P, H, base, m=2, k=3):
    """Deterministic period-column-token kernel. Returns [n,H]. No trained parameters."""
    n, L = ctx.shape
    G, reps = L // P, max(1, H // P)
    if G - m - reps + 1 < 1:
        return base.copy()
    C = torch.from_numpy(ctx[:, L - G * P:]).float().to(DEV)
    mu = C.mean(1, keepdim=True); sd = C.std(1, keepdim=True) + 1e-6
    z = ((C - mu) / (3 * sd)).clamp(-1, 1).view(n, G, P)
    E = column_tokens(model, z, G, pool="column").to(DEV)
    starts = list(range(0, G - m - reps + 1))
    vals = torch.stack([z[:, s + m:s + m + reps] for s in starts], 1)
    q = F.normalize(E[:, G - m:G].mean(1), dim=-1)
    kk = F.normalize(torch.stack([E[:, s:s + m].mean(1) for s in starts], 1), dim=-1)
    top = torch.einsum("nd,nkd->nk", q, kk).topk(min(k, len(starts)), dim=1).indices
    nb = torch.gather(vals, 1, top[:, :, None, None].expand(-1, -1, reps, P))
    cand = (nb.mean(1).reshape(n, reps * P)[:, :H] * 3 * sd + mu).cpu().numpy()
    if cand.shape[1] < H:
        cand = np.pad(cand, ((0, 0), (0, H - cand.shape[1])), mode="edge")
    cap = 3 * np.abs(np.diff(ctx, axis=1)).mean(1, keepdims=True) * math.sqrt(H)
    return base + np.clip(cand[:, :H] - base, -cap, cap)


def blend_with_kernel(model, ctx, P, H, min_ctx_periods=8, max_po=6):
    """Evidence-scaled blend over the 4 priors PLUS the kernel, all scored on dense
    pseudo-origins with honest re-embedding. Returns (pred, weights, names, n_origins)."""
    n, L = ctx.shape
    names = FAMILY + ["kernel"]
    err = np.zeros((n, len(names)))
    ends = [e for e in range(L - H, min_ctx_periods * P - 1, -P)][:max_po]
    for e in ends:
        sub, tgt = ctx[:, :e], ctx[:, e:e + H]
        pp = ZL.priors(sub, P, H)
        _, sub_base, _, _ = ZL.pseudo_origin_blend(sub, P, H, {k_: pp[k_] for k_ in FAMILY},
                                                   rule="pow_n", dense=True,
                                                   min_ctx_periods=min_ctx_periods)
        pp["kernel"] = kernel_forecast(model, sub, P, H, sub_base)
        for j, nm in enumerate(names):
            err[:, j] += ((pp[nm] - tgt) ** 2).mean(1)
    used = max(len(ends), 1)
    full = ZL.priors(ctx, P, H)
    _, base, _, _ = ZL.pseudo_origin_blend(ctx, P, H, {k_: full[k_] for k_ in FAMILY},
                                           rule="pow_n", dense=True,
                                           min_ctx_periods=min_ctx_periods)
    full["kernel"] = kernel_forecast(model, ctx, P, H, base)
    lo = err.min(1, keepdims=True) + 1e-12
    w = np.power(np.maximum(err / lo, 1e-12), -float(used))
    w /= w.sum(1, keepdims=True)
    stack = np.stack([full[nm] for nm in names], 1)
    return np.einsum("np,nph->nh", w, stack), w, names, used, base, full["kernel"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HORIZON))
    ap.add_argument("--stage", required=True, choices=["E", "F"])
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default=ZL.ROOT)
    ap.add_argument("--ckpt", default="MCG-NJU/videomae-base")
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    if a.stage == "F":
        _, _, P, L, H, Xte, Yte, pairs, man = build_manifest(a.dataset, a.data_dir, a.max_ch,
                                                             a.stride, a.n, a.seed)
        Xu, Yu = gather(Xte, Yte, pairs)
        ctx, fut = Xu[:, :, 0], Yu[:, :, 0]
    else:
        X, pairs, man = ZL.historical_manifest(a.dataset, a.data_dir, a.max_ch, stride=a.stride,
                                               n_max=a.n, seed=a.seed, split="audit")
        P, L, H = man["P"], man["L"], man["H"]
        ctx, fut = X[:, :L], X[:, L:L + H]
    print(f"[{a.stage}] {a.dataset} n={len(ctx)} P={P} L={L} H={H}", flush=True)
    from transformers import VideoMAEModel, VideoMAEConfig
    mp = VideoMAEModel.from_pretrained(a.ckpt).to(DEV).eval()
    torch.manual_seed(a.seed)
    mr = VideoMAEModel(VideoMAEConfig.from_pretrained(a.ckpt)).to(DEV).eval()
    allp = ZL.priors(ctx, P, H)
    res, err = {"smean_mse": float(np.mean((allp["smean"] - fut) ** 2))}, {}
    for nm, model in (("pt", mp), ("rand", mr)):
        with torch.no_grad():
            y, w, names, used, base, kraw = blend_with_kernel(model, ctx, P, H)
        res[f"{nm}_mse"] = float(np.mean((y - fut) ** 2))
        res[f"{nm}_kernel_only_mse"] = float(np.mean((kraw - fut) ** 2))
        res[f"{nm}_weight_mean"] = {n_: float(w[:, j].mean()) for j, n_ in enumerate(names)}
        err[nm] = ((y - fut) ** 2).mean(1)
        res["blend_mse"] = float(np.mean((base - fut) ** 2)); err["blend"] = ((base - fut) ** 2).mean(1)
        res["n_pseudo_origins"] = int(used)
    if a.stage == "F":
        vj = json.load(open(os.path.join("pilot/results_field/video_visionts/visionts_reference",
                                         f"visionts_{a.dataset}.json")))
        assert vj["manifest"]["pairs_sha1"] == man["pairs_sha1"], "manifest mismatch vs VisionTS"
        res["visionts_mse"] = vj["metrics"]["visionts_pretrained"]
    blk = int(np.ceil((L + H) / a.stride))
    for b in ("rand", "blend"):
        res[f"boot_pt_vs_{b}"] = ZL.paired_bootstrap(err["pt"], err[b], pairs[:, 0], blk)
    print(json.dumps(res, indent=2, default=float), flush=True)
    out = {"status": "complete", "candidate_id": "ZL-052", "stage": a.stage,
           "dataset": a.dataset, "manifest": man, "results": res,
           "wall_clock_s": round(time.time() - t0, 1)}
    d = os.path.join(a.root, f"zl052_stage{a.stage}"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"{a.dataset}.json"), "w"), indent=2, default=float)
    np.savez_compressed(os.path.join(d, f"{a.dataset}_per_pair.npz"), pairs=pairs, **err)
    print("[done]", a.dataset, flush=True)


if __name__ == "__main__":
    main()
