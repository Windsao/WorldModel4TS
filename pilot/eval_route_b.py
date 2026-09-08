"""Route B zero-shot evaluation on the six benchmark datasets, identical origins to VisionTS /
ZL-051 (pairs_sha1 asserted), with the two preregistered conditions:

  (i)  arm beats the random-init arm trained identically          (video_arm / video_random)
  (ii) adding the arm as a FIFTH candidate of the ZL-051 evidence-scaled prior blend improves
       the blend                                                    (blend+arm / blend)

plus the literal comparisons (arm alone vs VisionTS, vs blend, vs smean). Everything is strict
zero-shot: the checkpoints never saw these datasets, all statistics come from each origin's
context, and the blend weights are computed on dense pseudo-origins inside the context.

Stage F = benchmark test windows (VisionTS manifest).  Stage E = historical audit windows.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zeroshot_loop as ZL
import pretrain_route_b as B
from run_visionts_reference import build_manifest, gather, HORIZON

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PRIOR_FAMILY = ["smean", "snaive", "last", "recent_mean"]
MID_ROW = int(round(0.5 * (B.IMG - 1)))


def hp_for(P, H):
    k = int(math.ceil(H / P))
    return k + (k % 2)                         # tubelet-aligned number of future frames


class VideoForecaster:
    """frozen Route-B checkpoint -> forecast from the last periods of a context window."""

    def __init__(self, ckpt, device=DEV, batch=64):
        from transformers import VideoMAEForPreTraining
        self.model = VideoMAEForPreTraining.from_pretrained(ckpt).to(device).eval()
        self.device, self.batch = device, batch
        rc = os.path.join(os.path.dirname(ckpt.rstrip("/")), "run_config.json")
        self.enc_attn = json.load(open(rc)).get("enc_attn", "full") if os.path.exists(rc) else "full"
        if self.enc_attn == "spatial":
            B.apply_spatial_attention(self.model)
            print(f"[spatial encoder attention] {ckpt}", flush=True)

    @torch.no_grad()
    def __call__(self, ctx, P, H):
        """ctx [n, L] numpy -> [n, H] numpy. Uses up to 16-hp trailing periods of ctx; shorter
        contexts hide the leading tubelets (the model was trained for 8..12 visible periods)."""
        n, L = ctx.shape
        hp = hp_for(P, H)
        G_ctx_max = B.NF - hp
        G_avail = L // P
        G_use = min(G_ctx_max, G_avail)
        if (G_ctx_max - G_use) % 2:                # lead must be whole tubelets
            G_use -= 1
        lead = (G_ctx_max - G_use) // 2
        assert G_use >= 8, (P, H, L, G_use)
        tail = ctx[:, L - G_use * P:]                              # [n, G_use*P]
        mu = tail.mean(1, keepdims=True); sd = tail.std(1, keepdims=True) + 1e-6
        z = np.clip((tail - mu) / sd, -B.ZMAX, B.ZMAX).reshape(n, G_use, P)
        h = 0.5 + B.BAND * z / B.ZMAX
        rows_ctx = np.rint((1.0 - h) * (B.IMG - 1)).astype(np.int16)[:, :, B.col2phase(P)]   # [n,G_use,224]
        rows = np.full((n, B.NF, B.IMG), MID_ROW, np.int16)
        rows[:, lead * B.TS: lead * B.TS + G_use] = rows_ctx
        mask = B.forecast_mask(hp, lead)
        out = np.empty((n, H), np.float32)
        nf = B.n_future_tokens(hp)
        for s in range(0, n, self.batch):
            r = torch.from_numpy(rows[s:s + self.batch])
            m = mask[None].expand(r.shape[0], -1)
            vid = B.rows_to_video(r, self.device)
            B.set_attn_bias(self.model, mask)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
                logits = self.model(pixel_values=vid, bool_masked_pos=m.to(self.device)).logits
            logits = logits.float()[:, -nf:]
            c = logits.view(r.shape[0], hp // B.TS, B.GH, B.GH, B.TS, B.PS, B.PS, 3)
            frames = c.permute(0, 1, 4, 7, 2, 5, 3, 6).reshape(r.shape[0], hp, 3, B.IMG, B.IMG)
            zf = B.decode_frames_gray(frames.mean(2), P)                     # [b, hp, P]
            zf = zf.reshape(r.shape[0], hp * P)[:, :H].cpu().numpy()
            out[s:s + self.batch] = zf * sd[s:s + self.batch] + mu[s:s + self.batch]
        return out


def blend_with_extra(ctx, P, H, extra=None, min_ctx_periods=8):
    """ZL-051 combiner (pow_n, dense pseudo-origins, >= 8 context periods) over the four priors
    plus optional extra candidates {name: f(ctx_sub, P, H) -> [n, H]}. Returns dict of blends:
    'blend' (priors only, must equal ZL.pseudo_origin_blend) and 'blend+<name>' for each extra,
    plus the per-candidate pseudo-origin losses."""
    extra = extra or {}
    n, L = ctx.shape
    names = sorted(PRIOR_FAMILY)
    ends = [e for e in range(L - H, min_ctx_periods * P - 1, -P)]
    err = {nm: np.zeros(n) for nm in names + list(extra)}
    for end in ends:
        sub = ctx[:, :end]; tgt = ctx[:, end:end + H]
        pp = ZL.priors(sub, P, H)
        for nm in names:
            err[nm] += ((pp[nm] - tgt) ** 2).mean(1)
        for nm, f in extra.items():
            err[nm] += ((f(sub, P, H) - tgt) ** 2).mean(1)
    full = ZL.priors(ctx, P, H)
    cand = {nm: full[nm] for nm in names}
    for nm, f in extra.items():
        cand[nm] = f(ctx, P, H)
    used = len(ends)

    def combine(cnames):
        E = np.stack([err[c] for c in cnames], 1)
        S = np.stack([cand[c] for c in cnames], 1)
        if used == 0:
            w = np.full((n, len(cnames)), 1.0 / len(cnames))
        else:
            r = E / (E.min(1, keepdims=True) + 1e-12)
            w = np.power(np.maximum(r, 1e-12), -float(used)); w /= w.sum(1, keepdims=True)
        return np.einsum("np,nph->nh", w, S), w

    out = {"n_pseudo_origins": used, "cand": cand, "err": err}
    out["blend"], out["w_blend"] = combine(names)
    for nm in extra:
        out[f"blend+{nm}"], out[f"w_blend+{nm}"] = combine(names + [nm])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HORIZON))
    ap.add_argument("--stage", default="F", choices=["E", "F"])
    ap.add_argument("--data-dir", default="/nyx-storage1/hanliu/wm4ts/data")
    ap.add_argument("--ckpt-root", default="/nyx-storage1/hanliu/wm4ts/route_b")
    ap.add_argument("--arms", default=",".join(B.ARMS))
    ap.add_argument("--step", type=int, default=20000)
    ap.add_argument("--vts-ref-dir", default=os.path.join(HERE, "results_field", "video_visionts", "visionts_reference"))
    ap.add_argument("--out", default=os.path.join(HERE, "results_field", "route_b"))
    ap.add_argument("--n", type=int, default=None, help="override n pairs (Stage E)")
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--max-ch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    vts = None
    if args.stage == "F":
        vp = os.path.join(args.vts_ref_dir, f"visionts_{args.dataset}.json")
        vts = json.load(open(vp))
        man_v = vts["manifest"]
        max_ch = args.max_ch or man_v["max_ch"]; stride = args.stride or man_v["stride"]
        n_max = args.n or man_v["n_pairs"]
        data, borders, P, L, H, Xte, Yte, pairs, man = build_manifest(
            args.dataset, args.data_dir, max_ch, stride, n_max, man_v.get("seed", 0))
        assert man["pairs_sha1"] == man_v["pairs_sha1"], ("pair manifest mismatch vs VisionTS", man["pairs_sha1"], man_v["pairs_sha1"])
        Xu, Yu = gather(Xte, Yte, pairs)
        ctx, fut = Xu[:, :, 0], Yu[:, :, 0]
    else:
        X, pairs, man = ZL.historical_manifest(args.dataset, args.data_dir, args.max_ch or 112,
                                               stride=args.stride or 8, n_max=args.n or 2000,
                                               seed=args.seed, split="audit")
        P, L, H = man["P"], man["L"], man["H"]
        ctx, fut = X[:, :L], X[:, L:L + H]
    n = len(ctx)
    print(f"[{args.stage}] {args.dataset} n={n} P={P} L={L} H={H} hp={hp_for(P, H)}", flush=True)

    # sanity: our prior-only blend must reproduce ZL-051's combiner exactly
    pr = {k: v for k, v in ZL.priors(ctx, P, H).items() if k in PRIOR_FAMILY}
    _, base_zl, _, n_po = ZL.pseudo_origin_blend(ctx, P, H, pr, rule="pow_n", dense=True, min_ctx_periods=8)

    arms = [a for a in args.arms.split(",") if a]
    fcs = {}
    for a in arms:
        ck = os.path.join(args.ckpt_root, a, f"step_{args.step}")
        if not os.path.isdir(ck):
            print(f"[skip] {ck} missing", flush=True); continue
        fcs[f"video_{a}"] = VideoForecaster(ck)
    bl = blend_with_extra(ctx, P, H, extra=fcs)
    assert np.allclose(bl["blend"], base_zl, atol=1e-5), "prior-only blend != ZL-051 combiner"
    assert bl["n_pseudo_origins"] == n_po

    err, res = {}, {"n": n, "P": P, "L": L, "H": H, "hp": hp_for(P, H), "n_pseudo_origins": n_po}
    def add(name, y):
        e = ((y - fut) ** 2).mean(1); err[name] = e
        res[f"{name}_mse"] = float(e.mean()); res[f"{name}_mae"] = float(np.abs(y - fut).mean())
    allp = ZL.priors(ctx, P, H)
    add("smean", allp["smean"]); add("snaive", allp["snaive"])
    add("blend", bl["blend"])
    for nm in fcs:
        add(nm, bl["cand"][nm])
        add(f"blend+{nm}", bl[f"blend+{nm}"])
        res[f"w_{nm}_in_blend_mean"] = float(bl[f"w_blend+{nm}"][:, -1].mean())
    if vts is not None:
        res["visionts_mse"] = vts["metrics"]["visionts_pretrained"]
        res["visionts_pairs_sha1"] = vts["manifest"]["pairs_sha1"]

    blk = int(np.ceil((L + H) / man["stride"])) if "stride" in man else int(np.ceil((L + H) / 8))
    boot = {}
    for nm in fcs:
        boot[f"{nm}_vs_blend"] = ZL.paired_bootstrap(err[nm], err["blend"], pairs[:, 0], blk)
        boot[f"blend+{nm}_vs_blend"] = ZL.paired_bootstrap(err[f"blend+{nm}"], err["blend"], pairs[:, 0], blk)
        boot[f"{nm}_vs_smean"] = ZL.paired_bootstrap(err[nm], err["smean"], pairs[:, 0], blk)
        if "video_random" in fcs and nm != "video_random":
            boot[f"{nm}_vs_video_random"] = ZL.paired_bootstrap(err[nm], err["video_random"], pairs[:, 0], blk)
    res["bootstrap"] = boot
    for k in ("smean_mse", "snaive_mse", "blend_mse", "visionts_mse"):
        if k in res: print(f"  {k:28s} {res[k]:.4f}")
    for nm in fcs:
        print(f"  {nm:28s} {res[nm + '_mse']:.4f}   blend+{nm}: {res['blend+' + nm + '_mse']:.4f}"
              f"   w={res[f'w_{nm}_in_blend_mean']:.3f}", flush=True)
    out = {"status": "complete", "stage": args.stage, "dataset": args.dataset, "step": args.step,
           "manifest": man, "results": res, "wall_clock_s": round(time.time() - t0, 1)}
    d = os.path.join(args.out, f"stage{args.stage}"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"{args.dataset}_step{args.step}.json"), "w"), indent=2, default=float)
    np.savez_compressed(os.path.join(d, f"{args.dataset}_step{args.step}_per_pair.npz"), pairs=pairs, **err)
    print("[done]", args.dataset, round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
