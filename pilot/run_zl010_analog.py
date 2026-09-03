"""ZL-010: video-native context-only analog retrieval (Family 2).

The backbone is used ONLY as a similarity kernel. Query and candidate motifs, and every
candidate's continuation, lie entirely inside the observed context of that origin, so no
future value can influence retrieval. Nothing is trained.

The decisive measurement happens BEFORE any decoder: how good are the retrieved neighbours'
normalized continuations? Arms: pretrained scrolling video, identical random-init, the same
pretrained backbone on frame-PERMUTED video, on a STATIC repeated frame, and raw time-domain L2.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zeroshot_loop as ZL
from preprocess_renderers import _draw_curves, IMN_MEAN, IMN_STD

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMG, NF = 224, 16


def scroll_video(motifs, win_frac=0.5, permute=False, static=False, gen=None):
    """[N, W] level-free motifs -> [N,16,3,224,224]. Frame f shows a sliding window, so
    visual motion corresponds to the series advancing in time."""
    N, W = motifs.shape
    win = max(8, int(W * win_frac))
    starts = torch.linspace(0, W - win, NF, device=motifs.device)
    if static:
        starts = torch.full_like(starts, float(W - win))
    order = torch.arange(NF, device=motifs.device)
    if permute:
        order = torch.as_tensor(gen.permutation(NF), device=motifs.device)
    frames = []
    for f in range(NF):
        s = starts[order[f]]
        pos = s + torch.linspace(0, win - 1, IMG, device=motifs.device)
        lo = pos.floor().long().clamp(0, W - 1); hi = pos.ceil().long().clamp(0, W - 1)
        fr = (pos - lo).unsqueeze(0)
        frames.append(motifs[:, lo] * (1 - fr) + motifs[:, hi] * fr)
    cur = torch.stack(frames, 1)                                   # [N,16,224]
    ink = _draw_curves(cur.reshape(N * NF, 1, IMG).clamp(-1, 1), [1.0]).view(N, NF, IMG, IMG)
    g = 0.15 + 0.75 * ink
    vid = g.unsqueeze(2).expand(N, NF, 3, IMG, IMG)
    return ((vid - IMN_MEAN.to(g.device).unsqueeze(1)) / IMN_STD.to(g.device).unsqueeze(1)).contiguous()


def embed_motifs(model, motifs, batch=24, **kw):
    """Render and embed in CHUNKS. Materializing every candidate video at once needs ~41 GB
    for a 192-origin x 15-candidate pilot, which is what OOMed the first attempt."""
    out = []
    with torch.no_grad():
        for i in range(0, len(motifs), batch):
            v = scroll_video(motifs[i:i + batch], **kw)
            h = model(pixel_values=v).last_hidden_state
            out.append(F.normalize(h.mean(1), dim=-1))
            del v
    return torch.cat(out)


def znorm(x, floor=None):
    """Level-free normalization with a scale FLOOR.

    A nearly constant motif has std ~ 0; dividing by it sends the transferred continuation to
    ~1e11 and destroys the comparison. The floor is tied to the origin's own context scale, so
    it stays context-only and deterministic.
    """
    m = x.mean(-1, keepdim=True)
    s = x.std(-1, keepdim=True)
    if floor is not None:
        s = torch.maximum(s, floor)
    return (x - m) / (s + 1e-8), m, s + 1e-8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default=ZL.ROOT)
    ap.add_argument("--ckpt", default=os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base"))
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--motif-periods", type=int, default=2)
    ap.add_argument("--cand-stride", type=int, default=12)
    ap.add_argument("--gap-periods", type=int, default=1)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    X, pairs, man = ZL.historical_manifest(args.dataset, args.data_dir, n_max=args.n,
                                           seed=args.seed, split="audit")
    P, L, H = man["P"], man["L"], man["H"]
    W = args.motif_periods * P
    stride = args.cand_stride
    gap = args.gap_periods * P
    ctx, fut = X[:, :L], X[:, L:L + H]
    # candidate motif starts: motif [s, s+W) and its continuation [s+W, s+W+H) inside context,
    # with an exclusion gap from the query motif which is the last W steps of the context.
    q_start = L - W
    starts = [s for s in range(0, L - W - H + 1, stride) if s + W + H <= q_start - gap]
    assert starts, "no admissible candidate motifs"
    print(f"{args.dataset} n={len(X)} P={P} L={L} H={H} W={W} candidates={len(starts)} "
          f"origins={man['n_origins']}", flush=True)
    overlap = 0.0
    dist = min(q_start - (s + W + H) for s in starts)

    from transformers import VideoMAEModel, VideoMAEConfig
    mp = VideoMAEModel.from_pretrained(args.ckpt).to(DEV).eval()
    torch.manual_seed(args.seed)
    mr = VideoMAEModel(VideoMAEConfig.from_pretrained(args.ckpt)).to(DEV).eval()
    gen = np.random.default_rng(args.seed)

    Ct = torch.from_numpy(ctx).to(DEV)
    q_raw = Ct[:, q_start:q_start + W]
    ctx_scale = Ct.std(-1, keepdim=True)                                    # context-only
    floor = 0.05 * ctx_scale
    qz, qm, qs = znorm(q_raw, floor)
    cand = torch.stack([Ct[:, s:s + W] for s in starts], 1)                 # [N,C,W]
    cont = torch.stack([Ct[:, s + W:s + W + H] for s in starts], 1)         # [N,C,H]
    cz, cm, cs = znorm(cand, floor.unsqueeze(1))
    contz = ((cont - cm) / cs).clamp(-8, 8)          # level-free, bounded against tail blowups
    N, C = cz.shape[0], cz.shape[1]
    futz = ((torch.from_numpy(fut).to(DEV) - qm) / qs).clamp(-8, 8)

    arms = {}
    for name in ("pt_scroll", "rand_scroll", "pt_permuted", "pt_static"):
        model = mr if name == "rand_scroll" else mp
        perm = name == "pt_permuted"; stat = name == "pt_static"
        gen = np.random.default_rng(args.seed)          # same permutation for query and candidates
        eq = embed_motifs(model, qz, permute=perm, static=stat, gen=gen)
        gen = np.random.default_rng(args.seed)
        ec = embed_motifs(model, cz.reshape(N * C, W), permute=perm, static=stat, gen=gen).view(N, C, -1)
        sim = torch.einsum("nd,ncd->nc", eq, ec)
        arms[name] = sim
    arms["raw_l2"] = -((qz.unsqueeze(1) - cz) ** 2).mean(-1)

    res = {}
    for name, sim in arms.items():
        top = sim.topk(min(args.k, C), dim=1).indices
        pick = torch.gather(contz, 1, top.unsqueeze(-1).expand(-1, -1, H)).mean(1)
        res[f"{name}_norm_cont_mse"] = float(((pick - futz) ** 2).mean())
        pred = pick * qs + qm
        res[f"{name}_ts_mse"] = float(((pred - torch.from_numpy(fut).to(DEV)) ** 2).mean())
        res[f"{name}_percase"] = ((pick - futz) ** 2).mean(-1).cpu().numpy()
    # references on the same pairs
    pr = ZL.priors(ctx, P, H)
    for k, v in pr.items():
        res[f"prior_{k}_ts_mse"] = float(np.mean((v - fut) ** 2))
    # oracle ceiling: best possible candidate per origin
    best = ((contz.unsqueeze(1) - futz.unsqueeze(1).unsqueeze(1)) ** 2).mean(-1).squeeze(1).min(1).values
    res["oracle_best_candidate_norm_mse"] = float(best.mean())
    blk = int(np.ceil((L + H) / man["stride"]))
    for a, b in (("pt_scroll", "rand_scroll"), ("pt_scroll", "pt_permuted"),
                 ("pt_scroll", "pt_static"), ("pt_scroll", "raw_l2")):
        res[f"boot_{a}_vs_{b}"] = ZL.paired_bootstrap(res[f"{a}_percase"], res[f"{b}_percase"],
                                                      pairs[:, 0], blk)
    show = {k: v for k, v in res.items() if not k.endswith("_percase")}
    print(json.dumps(show, indent=2, default=float), flush=True)
    out = {"status": "complete", "candidate_id": "ZL-010", "dataset": args.dataset,
           "split": "audit", "manifest": man, "checkpoint": args.ckpt,
           "motif_W": W, "n_candidates": len(starts), "k": args.k,
           "query_candidate_min_gap_steps": int(dist), "candidate_overlap_with_future": overlap,
           "config_hash": ZL.cfg_hash(vars(args)), "results": show,
           "git_commit": os.environ.get("WM4TS_GIT_COMMIT", "unknown"),
           "wall_clock_s": round(time.time() - t0, 1)}
    d = os.path.join(args.root, "candidates", "ZL-010"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"metrics_{args.dataset}.json"), "w"), indent=2, default=float)
    np.savez_compressed(os.path.join(d, f"per_case_{args.dataset}.npz"), pairs=pairs,
                        **{k: v for k, v in res.items() if k.endswith("_percase")})
    print("[done]", args.dataset, flush=True)


if __name__ == "__main__":
    main()
