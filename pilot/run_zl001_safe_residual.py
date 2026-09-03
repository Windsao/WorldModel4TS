"""ZL-001: bounded context-calibrated safe residual around the best strict VideoMAE route.

Strict zero-shot: alpha is chosen per sample from rolling pseudo-origins that lie entirely
inside the observed context. No future target, no training, no benchmark test value.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zeroshot_loop as ZL
import videomae_patch_utils as VP
import video_visionts_renderers as VR
import video_visionts_layouts as VL
import video_visionts_mask_factorization as MF

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NF, PATCH, GRID = 16, 16, 14


def vmae_forecast(model, ctx_t, P, L, H, zero_logits=False):
    """The incumbent strict route: dense_static + right_10 + nearest-visible-physical."""
    B = ctx_t.shape[0]
    vi, _, mu, sd = VR.dense_static(ctx_t, torch.zeros(B, H, device=ctx_t.device), P, 10)
    sp = MF.make_matched_mask("right_future_100", 0)
    bm = VL.tube_mask_from_spatial(sp, B, DEV)
    with torch.no_grad():
        lg = model(pixel_values=vi, bool_masked_pos=bm).logits
        if zero_logits:
            lg = torch.zeros_like(lg)
        cin = VP.patchify(VP.unnormalize(vi))
        m_, s_ = VL.nearest_visible_physical(cin, bm)
        f = cin.clone()
        f[bm] = VP.denormalize_cubes(lg, m_, s_).view(-1, 512, 3)
        img = VP.unpatchify(f)[:, NF - 1].mean(1).clamp(0, 1) * 2 - 1
    return VR.invert_dense(img[:, :, 4 * PATCH:], P, H, mu, sd)[:, :, 0].cpu().numpy()


def run_arm(model, X, P, L, H, batch, zero=False):
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch, :L]).to(DEV)
        out.append(vmae_forecast(model, xb, P, L, H, zero))
    return np.concatenate(out)


def pseudo_cand(model, X, P, L, H, batch, n_pseudo, gap, zero=False):
    """Candidate + base predictions at internal pseudo-origins (targets inside context)."""
    outs = []
    for k in range(1, n_pseudo + 1):
        end = L - k * (H + gap)
        if end < 4 * P:
            break
        sub = X[:, :end]
        tgt = X[:, end:end + H]
        # the pseudo-context must still fill 16 period-frames: pad by repeating its own head
        need = 16 * P
        subc = sub if sub.shape[1] >= need else np.concatenate(
            [np.tile(sub[:, :1], (1, need - sub.shape[1])), sub], 1)
        cand = run_arm(model, np.concatenate([subc, np.zeros((len(X), H), np.float32)], 1),
                       P, need, H, batch, zero)
        base = ZL.priors(subc, P, H)["smean"]
        outs.append((base, cand, tgt))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default=ZL.ROOT)
    ap.add_argument("--ckpt", default=os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base"))
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-pseudo", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    X, pairs, man = ZL.historical_manifest(args.dataset, args.data_dir, n_max=args.n,
                                           seed=args.seed, split="audit")
    P, L, H = man["P"], man["L"], man["H"]
    ctx, fut = X[:, :L], X[:, L:L + H]
    print(f"{args.dataset} n={len(X)} origins={man['n_origins']} P={P} L={L} H={H}", flush=True)

    from transformers import VideoMAEForPreTraining, VideoMAEConfig
    mp = VideoMAEForPreTraining.from_pretrained(args.ckpt).to(DEV).eval()
    torch.manual_seed(args.seed)
    mr = VideoMAEForPreTraining(VideoMAEConfig.from_pretrained(args.ckpt)).to(DEV).eval()

    pr = ZL.priors(ctx, P, H)
    names, base, used = ZL.pseudo_origin_select(ctx, P, H, pr)
    gap = P
    res = {"prior_pick_counts": {k: int((names == k).sum()) for k in set(names.tolist())},
           "pseudo_origins_used": used}
    arms = {}
    for tag, model, zero in (("pt", mp, False), ("rand", mr, False), ("neutral", mp, True)):
        cand = run_arm(model, X, P, L, H, args.batch, zero)
        cp = pseudo_cand(model, X, P, L, H, args.batch, args.n_pseudo, gap, zero)
        y, alpha = ZL.safe_residual(base, cand, ctx, P, H, cand_pseudo=cp)
        arms[tag] = {"raw": cand, "wrapped": y, "alpha": alpha}
        res[f"alpha_nonzero_frac_{tag}"] = float((alpha > 0).mean())
        res[f"alpha_mean_{tag}"] = float(alpha.mean())
    e = {"base": ZL.per_pair_mse(base, fut), "smean": ZL.per_pair_mse(pr["smean"], fut),
         "snaive": ZL.per_pair_mse(pr["snaive"], fut)}
    for tag in arms:
        e[f"{tag}_raw"] = ZL.per_pair_mse(arms[tag]["raw"], fut)
        e[f"{tag}_wrapped"] = ZL.per_pair_mse(arms[tag]["wrapped"], fut)
    res["mse"] = {k: float(v.mean()) for k, v in e.items()}
    # does internal selection predict held-out gain?
    gain = e["base"] - e["pt_wrapped"]
    res["corr_alpha_vs_gain"] = float(np.corrcoef(arms["pt"]["alpha"], gain)[0, 1]) \
        if arms["pt"]["alpha"].std() > 0 else 0.0
    blk = int(np.ceil((L + H) / man["stride"]))
    res["boot_pt_vs_rand"] = ZL.paired_bootstrap(e["pt_wrapped"], e["rand_wrapped"], pairs[:, 0], blk)
    res["boot_pt_vs_base"] = ZL.paired_bootstrap(e["pt_wrapped"], e["base"], pairs[:, 0], blk)
    print(json.dumps({k: v for k, v in res.items() if k != "mse"}, indent=2, default=float), flush=True)
    print(json.dumps(res["mse"], indent=2), flush=True)
    out = {"status": "complete", "candidate_id": "ZL-001", "dataset": args.dataset,
           "split": "audit", "manifest": man, "checkpoint": args.ckpt,
           "config_hash": ZL.cfg_hash(vars(args)), "results": res,
           "git_commit": os.environ.get("WM4TS_GIT_COMMIT", "unknown"),
           "wall_clock_s": round(time.time() - t0, 1)}
    d = os.path.join(args.root, "candidates", "ZL-001"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"metrics_{args.dataset}.json"), "w"), indent=2, default=float)
    np.savez_compressed(os.path.join(d, f"per_pair_{args.dataset}.npz"),
                        pairs=pairs, alpha=arms["pt"]["alpha"], **{f"err_{k}": v for k, v in e.items()})
    print("[done]", args.dataset, flush=True)


if __name__ == "__main__":
    main()
