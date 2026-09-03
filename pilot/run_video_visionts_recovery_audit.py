"""Stages R0-R3: repaired target builder, leakage-safe manifests, historical layout audit.

R0 fixes the bug that invalidated the previous Stage E1: the old code rendered the target
video with a ZERO future (`blank_future=True`) and then used it as the historical
pseudo-future target, so the lens learned the statistics of a constant block. Here the target
builder has no switch -- it always receives the real historical pseudo-future, and
`dense_static` blanks it in the encoder input only.
"""

import argparse, hashlib, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_field as RF
import videomae_patch_utils as VP
import video_visionts_renderers as VR
import video_visionts_layouts as VL

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID, PATCH, NF = 14, 16, 16
HORIZON = {"ETTh1": 96, "ETTh2": 96, "ETTm2": 96, "electricity": 96, "traffic": 96, "solar": 144}
SEL = ("ETTh2", "ETTm2")


# ---------------------------------------------------------------- R0 target builder
def make_historical_pair(xb, P, H, L, cols):
    """xb [B, L+H] history. Returns (video_input, video_full, mu, sd).

    No `blank_future` switch by design. `dense_static` blanks the hidden block inside
    `video_input`, so passing the REAL pseudo-future cannot leak it to the encoder while
    still giving `video_full` the correct reconstruction target.
    """
    ctx = xb[:, :L]
    pseudo_future = xb[:, L:L + H]
    return VR.dense_static(ctx, pseudo_future, P, cols)


def masked_target_stats(video_full, bm):
    """Use the exact native-target utility rather than duplicating its statistics logic."""
    _, cube_mean, cube_std = VP.native_targets(video_full)
    B = video_full.shape[0]
    C = cube_mean.shape[-1]
    tm = cube_mean[bm].view(B, -1, C)
    ts = cube_std[bm].view(B, -1, C)
    return tm, torch.log(ts.clamp_min(1e-6))


def target_distribution(tm, ts_log):
    ts = ts_log.exp()
    f = lambda t, q: float(torch.quantile(t.flatten().float(), q))
    return {"frac_std_le_1e-5": round(float((ts <= 1e-5).float().mean()), 6),
            "mean_median": round(f(tm, .5), 6), "mean_p05": round(f(tm, .05), 6),
            "mean_p95": round(f(tm, .95), 6), "std_median": round(f(ts, .5), 6),
            "std_p05": round(f(ts, .05), 6), "std_p95": round(f(ts, .95), 6)}


# ---------------------------------------------------------------- R1 manifests
def build_partitions(dataset, data_dir, max_ch, n_train=1500, n_val=512, n_audit=512,
                     stride=32, seed=0):
    """Chronological 60/20/20 split of the TRAIN range with an L+H purge between partitions."""
    data, borders = RF.load_mv(dataset, data_dir, max_ch)
    P = RF.P; L = NF * P; H = HORIZON[dataset]
    b_train = borders[0]
    span = L + H
    lo, hi = span, b_train                       # forecast origins usable inside train range
    n = hi - lo
    c1, c2 = lo + int(0.6 * n), lo + int(0.8 * n)
    parts = {"train": (lo, c1 - span), "val": (c1 + span, c2 - span), "audit": (c2 + span, hi)}
    M = data.shape[1]
    out, rng = {}, np.random.default_rng(seed)
    for name, (a, b) in parts.items():
        origins = np.arange(a, b, stride)
        assert len(origins) > 0, f"{dataset}/{name} empty"
        pairs = np.array([(o, c) for o in origins for c in range(M)], dtype=np.int64)
        cap = {"train": n_train, "val": n_val, "audit": n_audit}[name]
        if len(pairs) > cap:
            k = rng.choice(len(pairs), cap, replace=False); k.sort(); pairs = pairs[k]
        out[name] = pairs
    # disjointness including context and pseudo-future support
    sup = {k: (int(v[:, 0].min()) - L, int(v[:, 0].max()) + H) for k, v in out.items()}
    assert sup["train"][1] < sup["val"][0] and sup["val"][1] < sup["audit"][0], \
        f"partition supports overlap: {sup}"
    man = {"dataset": dataset, "P": P, "context": L, "horizon": H, "M": M,
           "borders": list(map(int, borders)), "purge": span, "stride": stride,
           "splits": {k: [int(x) for x in v] for k, v in parts.items()},
           "support": {k: list(v) for k, v in sup.items()},
           "counts": {k: int(len(v)) for k, v in out.items()},
           "pair_sha1": {k: hashlib.sha1(v.tobytes()).hexdigest()[:16] for k, v in out.items()},
           "seed": seed}
    return data, P, L, H, out, man


def gather_hist(data, pairs, L, H):
    """(origin, channel) -> [n, L+H] raw history windows."""
    o, c = pairs[:, 0], pairs[:, 1]
    idx = o[:, None] + np.arange(-L, H)[None, :]
    return data[idx, c[:, None]].astype(np.float32)


# ---------------------------------------------------------------- R3 layout audit
def audit_layout(model, X, P, L, H, vc, layout, seed, batch, stat_rules):
    cols = GRID - vc
    canon_of_phys, phys_of_canon, phys_mask = VL.build_layout(layout, vc, seed)
    acc, tot = {}, 0
    dist = None
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(DEV)
        B = xb.shape[0]
        vi, vf, mu, sd = make_historical_pair(xb, P, H, L, cols)
        gi = VR.video_to_gray(vi)[:, 0]; gf = VR.video_to_gray(vf)[:, 0]
        if layout != "G0":
            gi = VL.permute_image(gi, canon_of_phys); gf = VL.permute_image(gf, canon_of_phys)
        def to_vid(g):
            im = ((g + 1) / 2).clamp(0, 1).unsqueeze(1).unsqueeze(1).expand(B, NF, 3, 224, 224)
            return ((im - VR.IMN_MEAN.to(g.device).unsqueeze(1))
                    / VR.IMN_STD.to(g.device).unsqueeze(1)).contiguous()
        vip, vfp = to_vid(gi), to_vid(gf)
        bm = VL.tube_mask_from_spatial(phys_mask, B, DEV)
        with torch.no_grad():
            logits = model(pixel_values=vip, bool_masked_pos=bm).logits
        tgt, cmean, cstd = VP.native_targets(vfp)
        labels = tgt[bm].view(B, -1, 1536)
        n_el = labels.numel()
        add = lambda k, v: acc.__setitem__(k, acc.get(k, 0.0) + v * n_el)
        add("native_model", float(F.mse_loss(logits, labels)))
        add("native_zero", float(F.mse_loss(torch.zeros_like(logits), labels)))
        cubes_true = VP.patchify(VP.unnormalize(vfp))
        cubes_in = VP.patchify(VP.unnormalize(vip))
        C = cubes_true.shape[-1]
        raw_true = cubes_true[bm].view(B, -1, 512, C)
        sm, ss = cmean[bm].view(B, -1, 1, C), cstd[bm].view(B, -1, 1, C)
        add("raw_oracle", float(F.mse_loss(VP.denormalize_cubes(logits, sm, ss), raw_true)))
        add("raw_oracle_zero", float(F.mse_loss(VP.denormalize_cubes(torch.zeros_like(logits), sm, ss), raw_true)))
        y_true = xb[:, L:L + H].unsqueeze(-1).cpu().numpy()
        for rule in stat_rules:
            if rule == "oracle":
                m_, s_ = sm, ss
            elif rule == "nearest_visible_physical":
                m_, s_ = VL.nearest_visible_physical(cubes_in, bm)
            else:
                m_, s_ = VP.causal_cube_stats(cubes_in, bm, rule=rule)
            add(f"raw_{rule}", float(F.mse_loss(VP.denormalize_cubes(logits, m_, s_), raw_true)))
            for tag, lg in (("model", logits), ("zero", torch.zeros_like(logits))):
                filled = cubes_in.clone()
                filled[bm] = VP.denormalize_cubes(lg, m_, s_).view(-1, 512, C)
                img = VP.unpatchify(filled)[:, NF - 1].mean(1).clamp(0, 1) * 2 - 1
                if layout != "G0":
                    img = VL.unpermute_image(img, canon_of_phys)
                vis_w = vc * PATCH
                yh = VR.invert_dense(img[:, :, vis_w:], P, H, mu, sd).cpu().numpy()
                acc[f"ts_{tag}_{rule}_mse"] = acc.get(f"ts_{tag}_{rule}_mse", 0.0) + float(np.mean((yh - y_true) ** 2)) * n_el
                acc[f"ts_{tag}_{rule}_mae"] = acc.get(f"ts_{tag}_{rule}_mae", 0.0) + float(np.mean(np.abs(yh - y_true))) * n_el
        if dist is None:
            tm, tls = masked_target_stats(vfp, bm)
            dist = target_distribution(tm, tls)
        tot += n_el
    met = {k: round(v / tot, 6) for k, v in acc.items()}
    met["native_ratio"] = round(met["native_model"] / (met["native_zero"] + 1e-12), 6)
    met["target_dist"] = dist
    met["layout_hash"] = hashlib.sha1(canon_of_phys.tobytes()).hexdigest()[:16]
    met["mask_hash"] = hashlib.sha1(phys_mask.tobytes()).hexdigest()[:16]
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HORIZON))
    ap.add_argument("--layout", required=True, choices=list(VL.LAYOUTS))
    ap.add_argument("--layout-seed", type=int, default=0)
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default="pilot/results_field/video_visionts_recovery")
    ap.add_argument("--ckpt", default=os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base"))
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--random-init", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    data, P, L, H, parts, man = build_partitions(args.dataset, args.data_dir, args.max_ch,
                                                 n_audit=args.n, seed=args.seed)
    vc, mc = VL.width_rule(L, H)
    X = gather_hist(data, parts["audit"], L, H)
    print(f"{args.dataset} {args.layout} vc={vc} mc={mc} audit_n={len(X)}", flush=True)

    from transformers import VideoMAEForPreTraining, VideoMAEConfig
    if args.random_init:
        torch.manual_seed(args.seed)
        model = VideoMAEForPreTraining(VideoMAEConfig.from_pretrained(args.ckpt))
    else:
        model = VideoMAEForPreTraining.from_pretrained(args.ckpt)
    model = model.to(DEV).eval()

    rules = ["oracle", "context_global", "nearest_visible_left", "nearest_visible_physical"]
    met = audit_layout(model, X, P, L, H, vc, args.layout, args.layout_seed, args.batch, rules)
    print(json.dumps({k: v for k, v in met.items() if not isinstance(v, dict)}, indent=2)[:900], flush=True)

    import transformers
    out = {"status": "complete", "stage": "R3", "regime": "HA", "dataset": args.dataset,
           "layout": args.layout, "layout_seed": args.layout_seed,
           "visible_cols": vc, "masked_cols": mc,
           "control": "random" if args.random_init else "pretrained",
           "checkpoint": args.ckpt, "renderer": "dense_static", "width_rule": "visionts_align0.4",
           "git_commit": os.environ.get("WM4TS_GIT_COMMIT", "unknown"),
           "torch": torch.__version__, "transformers": transformers.__version__,
           "manifest": man, "n_eval": int(len(X)), "seed": args.seed,
           "metrics": met, "wall_clock_s": round(time.time() - t0, 1)}
    d = os.path.join(args.root, "layout_audit"); os.makedirs(d, exist_ok=True)
    ctl = "rand" if args.random_init else "pt"
    fn = f"{args.dataset}_{args.layout}_ls{args.layout_seed}_{ctl}.json"
    json.dump(out, open(os.path.join(d, fn), "w"), indent=2)
    md = os.path.join(args.root, "manifests"); os.makedirs(md, exist_ok=True)
    json.dump(man, open(os.path.join(md, f"manifest_{args.dataset}.json"), "w"), indent=2)
    print("[done]", args.dataset, args.layout, flush=True)


if __name__ == "__main__":
    main()
