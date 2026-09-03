"""Stage R4: corrected patch-stat lenses L1 / L2 / L3 with the paired random-backbone control.

The previous lens (pilot/run_video_visionts_adapt.py) trained against a target rendered from a
ZERO future, so its result is invalid. Here the target comes from `make_historical_pair`,
which always receives the real historical pseudo-future.

L1  corrected reproduction of the original architecture (isolates the effect of the repair)
L2  bounded residual lens -- cannot catastrophically leave the causal starting point
L3  bounded tube-shared lens -- exploits that dense_static repeats one image over 8 tubelets,
    so the true raw statistics of a spatial patch are identical in every tubelet
"""

import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_field as RF
import videomae_patch_utils as VP
import video_visionts_renderers as VR
import video_visionts_layouts as VL
from run_video_visionts_recovery_audit import (make_historical_pair, masked_target_stats,
                                               build_partitions, gather_hist, HORIZON, SEL)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID, PATCH, NF, TUB = 14, 16, 16, 2
NTUB = NF // TUB


class LensV2(nn.Module):
    """variant in {L1, L2, L3}. Final layer zero-init, so step 0 == the causal rule (L0)."""

    def __init__(self, variant="L2", d_logit=1536, hidden=96, C=3):
        super().__init__()
        self.variant = variant
        n_pos = GRID * GRID if variant == "L3" else NTUB * GRID * GRID
        self.proj = nn.Linear(d_logit, hidden)
        self.pos = nn.Embedding(n_pos, hidden)
        self.bnd = nn.Linear(2 * C, hidden)
        self.body = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Linear(hidden, 2 * C)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, logits, pos_idx, bnd_mean, bnd_std, group=None, n_group=None):
        C = bnd_mean.shape[-1]
        h = self.proj(logits)
        if self.variant == "L3":
            # average the projected representation over the 8 tubelets of one spatial patch
            B, n, d = h.shape
            g = group.unsqueeze(-1).expand(-1, -1, d)
            summed = torch.zeros(B, n_group, d, device=h.device, dtype=h.dtype).scatter_add_(1, g, h)
            cnt = torch.zeros(B, n_group, 1, device=h.device, dtype=h.dtype).scatter_add_(
                1, group.unsqueeze(-1), torch.ones_like(h[..., :1]))
            h = (summed / cnt.clamp_min(1)).gather(1, g)
        h = h + self.pos(pos_idx) + self.bnd(
            torch.cat([bnd_mean, torch.log(bnd_std.clamp_min(1e-6))], -1))
        o = self.head(self.body(h))
        dm, dls = o[..., :C], o[..., C:]
        if self.variant == "L1":
            return bnd_mean + dm, torch.log(bnd_std.clamp_min(1e-6)) + dls
        # L2/L3 bounded residual: mean stays in [0,1]; std moves by at most exp(+-2)
        mean = torch.clamp(bnd_mean + 0.25 * torch.tanh(dm), 0.0, 1.0)
        log_std = torch.log(bnd_std.clamp_min(1e-6)) + 2.0 * torch.tanh(dls)
        return mean, log_std


def spatial_group(bm):
    """For masked tokens, the index of their spatial patch and the tubelet-collapsed count."""
    B, N = bm.shape
    tok = torch.arange(N, device=bm.device).unsqueeze(0).expand(B, N)[bm].view(B, -1)
    return tok % (GRID * GRID), GRID * GRID, tok


def lens_batch(model, xb, P, L, H, vc, canon_of_phys, phys_mask, need_target=True):
    cols = GRID - vc
    B = xb.shape[0]
    vi, vf, mu, sd = make_historical_pair(xb, P, H, L, cols)
    gi, gf = VR.video_to_gray(vi)[:, 0], VR.video_to_gray(vf)[:, 0]
    if canon_of_phys is not None:
        gi = VL.permute_image(gi, canon_of_phys); gf = VL.permute_image(gf, canon_of_phys)
    def to_vid(g):
        im = ((g + 1) / 2).clamp(0, 1).unsqueeze(1).unsqueeze(1).expand(B, NF, 3, 224, 224)
        return ((im - VR.IMN_MEAN.to(g.device).unsqueeze(1))
                / VR.IMN_STD.to(g.device).unsqueeze(1)).contiguous()
    vip, vfp = to_vid(gi), to_vid(gf)
    bm = VL.tube_mask_from_spatial(phys_mask, B, DEV)
    with torch.no_grad():
        logits = model(pixel_values=vip, bool_masked_pos=bm).logits
        cubes_in = VP.patchify(VP.unnormalize(vip))
        bmn, bsd = VL.nearest_visible_physical(cubes_in, bm)
        tm = tls = None
        if need_target:
            tm, tls = masked_target_stats(vfp, bm)
    return dict(logits=logits, bm=bm, cubes_in=cubes_in, mu=mu, sd=sd,
                bmn=bmn.squeeze(2), bsd=bsd.squeeze(2), tm=tm, tls=tls, vip=vip)


def invert(cubes_in, bm, logits, mean, std, canon_of_phys, vc, P, H, mu, sd):
    C = cubes_in.shape[-1]
    filled = cubes_in.clone()
    filled[bm] = VP.denormalize_cubes(logits, mean.unsqueeze(2), std.unsqueeze(2)).view(-1, 512, C)
    img = VP.unpatchify(filled)[:, NF - 1].mean(1).clamp(0, 1) * 2 - 1
    if canon_of_phys is not None:
        img = VL.unpermute_image(img, canon_of_phys)
    return VR.invert_dense(img[:, :, vc * PATCH:], P, H, mu, sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="L2", choices=["L1", "L2", "L3"])
    ap.add_argument("--layout", default="G0", choices=list(VL.LAYOUTS))
    ap.add_argument("--layout-seed", type=int, default=0)
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default="pilot/results_field/video_visionts_recovery")
    ap.add_argument("--ckpt", default=os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base"))
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--random-backbone", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.variant == "L1":
        args.lr, args.steps = 1e-3, 600
    t0 = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    from transformers import VideoMAEForPreTraining, VideoMAEConfig
    if args.random_backbone:
        torch.manual_seed(args.seed)
        model = VideoMAEForPreTraining(VideoMAEConfig.from_pretrained(args.ckpt))
    else:
        model = VideoMAEForPreTraining.from_pretrained(args.ckpt)
    model = model.to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(args.seed)                       # identical lens init in both arms
    lens = LensV2(args.variant).to(DEV)
    n_par = sum(p.numel() for p in lens.parameters())
    assert n_par < 500_000, n_par
    print(f"{args.variant} layout={args.layout} params={n_par} "
          f"backbone={'random' if args.random_backbone else 'pretrained'}", flush=True)

    pool = {}
    for ds in SEL:
        data, P, L, H, parts, man = build_partitions(ds, args.data_dir, args.max_ch, seed=args.seed)
        vc, _ = VL.width_rule(L, H)
        cp, _, pm = VL.build_layout(args.layout, vc, args.layout_seed)
        cp = None if args.layout == "G0" else cp
        pool[ds] = dict(P=P, L=L, H=H, vc=vc, cp=cp, pm=pm,
                        tr=gather_hist(data, parts["train"], L, H),
                        va=gather_hist(data, parts["val"], L, H))
    print("pool:", {d: (len(v["tr"]), len(v["va"])) for d, v in pool.items()}, flush=True)

    opt = torch.optim.AdamW(lens.parameters(), lr=args.lr, weight_decay=1e-2)
    rng = np.random.default_rng(args.seed)

    def val_ratio():
        """R_val = geo-mean over datasets of pseudo-TS MSE(candidate) / pseudo-TS MSE(L0)."""
        lens.eval(); logs = []
        with torch.no_grad():
            for ds, v in pool.items():
                num = den = 0.0
                for i in range(0, min(len(v["va"]), 128), args.batch):
                    xb = torch.from_numpy(v["va"][i:i + args.batch]).to(DEV)
                    d = lens_batch(model, xb, v["P"], v["L"], v["H"], v["vc"], v["cp"], v["pm"], False)
                    grp, ng, tok = spatial_group(d["bm"])
                    pos = grp if args.variant == "L3" else tok
                    pm_, pls_ = lens(d["logits"], pos, d["bmn"], d["bsd"], grp, ng)
                    yl = invert(d["cubes_in"], d["bm"], d["logits"], pm_, pls_.exp(), v["cp"], v["vc"], v["P"], v["H"], d["mu"], d["sd"])
                    y0 = invert(d["cubes_in"], d["bm"], d["logits"], d["bmn"], d["bsd"], v["cp"], v["vc"], v["P"], v["H"], d["mu"], d["sd"])
                    yt = torch.from_numpy(v["va"][i:i + args.batch, v["L"]:v["L"] + v["H"]]).to(DEV).unsqueeze(-1)
                    num += float(((yl - yt) ** 2).mean()); den += float(((y0 - yt) ** 2).mean())
                logs.append(math.log(max(num, 1e-12) / max(den, 1e-12)))
        lens.train()
        return math.exp(sum(logs) / len(logs))

    best, best_state, bad, hist = float("inf"), None, 0, []
    lens.train()
    for step in range(args.steps):
        ds = list(pool)[step % len(pool)]
        v = pool[ds]
        idx = rng.choice(len(v["tr"]), args.batch, False)
        xb = torch.from_numpy(v["tr"][idx]).to(DEV)
        d = lens_batch(model, xb, v["P"], v["L"], v["H"], v["vc"], v["cp"], v["pm"], True)
        grp, ng, tok = spatial_group(d["bm"])
        pos = grp if args.variant == "L3" else tok
        pm_, pls_ = lens(d["logits"], pos, d["bmn"], d["bsd"], grp, ng)
        loss = F.smooth_l1_loss(pm_, d["tm"]) + F.smooth_l1_loss(pls_, d["tls"])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % args.val_every == 0:
            r = val_ratio(); hist.append((step + 1, round(r, 5)))
            if r < best - 1e-4:
                best, bad = r, 0
                best_state = {k: x.detach().clone() for k, x in lens.state_dict().items()}
            else:
                bad += 1
            print(f"[info] step {step+1} loss {loss.item():.5f} R_val {r:.4f} best {best:.4f}", flush=True)
            if bad >= args.patience:
                print("[info] early stop", flush=True); break
    if best_state is not None:
        lens.load_state_dict(best_state)
    lens.eval()

    # frozen historical-audit evaluation (never the genuine future)
    res, clamp_stats = {}, {}
    with torch.no_grad():
        for ds in SEL:
            data, P, L, H, parts, man = build_partitions(ds, args.data_dir, args.max_ch, seed=args.seed)
            v = pool[ds]; Xa = gather_hist(data, parts["audit"], L, H)
            num = den = 0.0; hits = tot = 0; sat = 0
            for i in range(0, len(Xa), args.batch):
                xb = torch.from_numpy(Xa[i:i + args.batch]).to(DEV)
                d = lens_batch(model, xb, P, L, H, v["vc"], v["cp"], v["pm"], False)
                grp, ng, tok = spatial_group(d["bm"])
                pos = grp if args.variant == "L3" else tok
                pm_, pls_ = lens(d["logits"], pos, d["bmn"], d["bsd"], grp, ng)
                hits += int(((pm_ <= 1e-6) | (pm_ >= 1 - 1e-6)).sum()); tot += pm_.numel()
                sat += int((( (pls_ - torch.log(d["bsd"].clamp_min(1e-6))).abs() / 2.0) > 0.95).sum())
                yl = invert(d["cubes_in"], d["bm"], d["logits"], pm_, pls_.exp(), v["cp"], v["vc"], P, H, d["mu"], d["sd"])
                y0 = invert(d["cubes_in"], d["bm"], d["logits"], d["bmn"], d["bsd"], v["cp"], v["vc"], P, H, d["mu"], d["sd"])
                yt = torch.from_numpy(Xa[i:i + args.batch, L:L + H]).to(DEV).unsqueeze(-1)
                num += float(((yl - yt) ** 2).sum()); den += float(((y0 - yt) ** 2).sum())
            res[ds] = {"lens_mse": round(num / max(1, len(Xa) * H), 6),
                       "L0_mse": round(den / max(1, len(Xa) * H), 6),
                       "ratio": round(num / max(den, 1e-12), 6)}
            clamp_stats[ds] = {"mean_clamp_rate": round(hits / max(tot, 1), 6),
                               "logstd_saturation_rate": round(sat / max(tot, 1), 6)}
            print(f"[audit] {ds} {res[ds]}", flush=True)
    agg = math.exp(sum(math.log(res[d]["ratio"]) for d in res) / len(res))
    print(f"[audit] aggregate ratio vs L0 = {agg:.4f}", flush=True)

    import transformers
    out = {"status": "complete", "stage": "R4", "regime": "LA", "variant": args.variant,
           "layout": args.layout, "layout_seed": args.layout_seed,
           "backbone": "random" if args.random_backbone else "pretrained",
           "lens_params": n_par, "steps": args.steps, "lr": args.lr, "batch": args.batch,
           "seed": args.seed, "checkpoint": args.ckpt, "val_history": hist,
           "best_R_val": round(best, 6), "audit": res, "aggregate_ratio_vs_L0": round(agg, 6),
           "clamp_stats": clamp_stats,
           "git_commit": os.environ.get("WM4TS_GIT_COMMIT", "unknown"),
           "torch": torch.__version__, "transformers": transformers.__version__,
           "wall_clock_s": round(time.time() - t0, 1)}
    d_ = os.path.join(args.root, "historical_eval"); os.makedirs(d_, exist_ok=True)
    tag = "rand" if args.random_backbone else "pt"
    json.dump(out, open(os.path.join(d_, f"{args.variant}_{args.layout}_ls{args.layout_seed}_{tag}_s{args.seed}.json"), "w"), indent=2)
    ck = os.path.join(args.root, "lens_checkpoints"); os.makedirs(ck, exist_ok=True)
    torch.save(lens.state_dict(), os.path.join(ck, f"{args.variant}_{args.layout}_{tag}_s{args.seed}.pt"))
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
