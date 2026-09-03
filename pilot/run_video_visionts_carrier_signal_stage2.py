"""Stage C2/C3/C4: carrier-aware baselines, conditionality controls, all-tubelet decoding."""

import hashlib, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import videomae_patch_utils as VP
import video_visionts_mask_factorization as MF
import video_visionts_carrier_signal as CS
from run_video_visionts_recovery_audit import build_partitions, gather_hist

DEV = "cuda" if torch.cuda.is_available() else "cpu"
G, FC, NT, CC = CS.GRID, CS.FUT_COLS, CS.NTUB, CS.CTX_COLS


def derangement_within_channel(pairs, shift=1):
    """Cyclic shift of origins within each channel: no fixed points, stays in distribution."""
    perm = np.arange(len(pairs))
    for c in np.unique(pairs[:, 1]):
        i = np.where(pairs[:, 1] == c)[0]
        if len(i) > 1:
            perm[i] = np.roll(i, shift)
    return perm


def stage_signal(args):
    from transformers import VideoMAEForPreTraining, VideoMAEConfig
    split_key = {"audit": "audit", "validation": "val"}[args.split]
    data, P, L, H, parts, man = build_partitions(args.dataset, args.data_dir, args.max_ch,
                                                 seed=args.manifest_seed)
    pairs = parts[split_key]
    X = gather_hist(data, pairs, L, H)
    n = len(X)
    perm_in = derangement_within_channel(pairs, 1)
    perm_tg = derangement_within_channel(pairs, 2)
    print(f"{args.dataset}/{args.split} n={n} P={P} L={L} H={H}", flush=True)

    mp = VideoMAEForPreTraining.from_pretrained(args.ckpt).to(DEV).eval()
    torch.manual_seed(args.random_init_seed)
    mr = VideoMAEForPreTraining(VideoMAEConfig.from_pretrained(args.ckpt)).to(DEV).eval()
    sp = MF.make_matched_mask("right_future_100", 0)

    ARMS = ["b0_zero", "b1_neutral", "b3_smean", "b4_snaive",
            "pt_true", "pt_shuffle", "pt_neutral", "rand_true"]
    nat = {a: {"sum": 0.0, "n": 0} for a in ARMS}
    AB = {a: np.zeros((n, NT, G, FC, 2), dtype=np.float32) for a in
          ("pt_true", "pt_shuffle", "pt_neutral", "rand_true")}
    Zt = np.zeros((n, G, FC), dtype=np.float32)
    TSerr = {}
    zerr = {}
    ptrue_z, ttrue_z = [], []
    rho_all = []

    ctx_all = torch.from_numpy(X[:, :L])
    mu_all, sd_all = CS.context_stats(ctx_all)
    ctxg_all = CS.context_grid(ctx_all, P, mu_all, sd_all)          # [n,14,4] on CPU

    for i in range(0, n, args.batch):
        sl = slice(i, min(i + args.batch, n))
        xb = torch.from_numpy(X[sl]).to(DEV)
        B = xb.shape[0]
        ctx, fut = xb[:, :L], xb[:, L:L + H]
        mu, sd = CS.context_stats(ctx)
        zf = CS.znorm(fut, mu, sd)
        gfut = CS.future_grid(zf, P)                                # target future grid
        Zt[sl] = gfut.cpu().numpy()
        gctx = ctxg_all[sl].to(DEV)
        gdonor = ctxg_all[perm_in[sl]].to(DEV)
        bm = MF.to_tube(sp, B, DEV)
        # ---- targets (from the TRUE future) -- never enter any input
        vf = CS.carrier_video(CS.full_grid(gctx, gfut))
        tgt, _ = CS.native_future_targets(vf, bm)
        # ---- inputs
        inputs = {"pt_true": CS.carrier_video(CS.full_grid(gctx, torch.zeros(B, G, FC, device=DEV))),
                  "pt_shuffle": CS.carrier_video(CS.full_grid(gdonor, torch.zeros(B, G, FC, device=DEV))),
                  "pt_neutral": CS.carrier_video(CS.full_grid(torch.zeros(B, G, CC, device=DEV),
                                                              torch.zeros(B, G, FC, device=DEV)))}
        inputs["rand_true"] = inputs["pt_true"]
        with torch.no_grad():
            preds = {a: (mr if a == "rand_true" else mp)(pixel_values=inputs[a],
                                                         bool_masked_pos=bm).logits
                     for a in inputs}
            # ---- carrier-aware baselines, as native targets of a constructed carrier
            preds["b0_zero"] = torch.zeros_like(tgt)
            preds["b1_neutral"] = CS.native_future_targets(
                CS.carrier_video(CS.full_grid(gctx, torch.zeros(B, G, FC, device=DEV))), bm)[0]
            reps = (H + P - 1) // P; Gp = L // P
            smn = torch.from_numpy(np.tile(ctx.reshape(B, Gp, P).mean(1).cpu().numpy(),
                                           (1, reps))[:, :H]).to(DEV)
            snv = torch.from_numpy(np.tile(ctx[:, -P:].cpu().numpy(), (1, reps))[:, :H]).to(DEV)
            for nm, series in (("b3_smean", smn), ("b4_snaive", snv)):
                gg = CS.future_grid(CS.znorm(series, mu, sd), P)
                preds[nm] = CS.native_future_targets(
                    CS.carrier_video(CS.full_grid(gctx, gg)), bm)[0]
        for a in ARMS:
            e = ((preds[a] - tgt) ** 2).double()
            nat[a]["sum"] += float(e.sum()); nat[a]["n"] += e.numel()
        # ---- Fourier coefficients + decoders
        yt = fut.unsqueeze(-1)
        for a in ARMS:
            aa, bb = CS.fourier_ab(CS.tile_from_logits(preds[a]))
            aa, bb = CS.future_token_view(aa), CS.future_token_view(bb)
            if a in AB:
                AB[a][sl, ..., 0] = aa.float().cpu().numpy()
                AB[a][sl, ..., 1] = bb.float().cpu().numpy()
            for dec in ("D0", "D1", "D2", "D3"):
                z, rho = CS.decode(aa, bb, dec)
                zerr.setdefault((a, dec), 0.0)
                zerr[(a, dec)] += float((((z - gfut) ** 2).double()).sum())
                if a in ("pt_true", "rand_true", "b1_neutral") or dec == "D2":
                    ts = CS.grid_to_series(z, P, H, mu, sd)
                    TSerr.setdefault((a, dec), []).append(
                        ((ts - yt) ** 2).mean(dim=(1, 2)).double().cpu().numpy())
                if a == "pt_true" and dec == "D2":
                    ptrue_z.append(z.cpu()); ttrue_z.append(gfut.cpu()); rho_all.append(rho.cpu())
    # ---- post-hoc target shuffle (no new forward)
    aa = torch.from_numpy(AB["pt_true"][..., 0]); bb = torch.from_numpy(AB["pt_true"][..., 1])
    z_pt, _ = CS.decode(aa, bb, "D2")
    Ztt = torch.from_numpy(Zt)
    zerr[("y_shuffle", "D2")] = float((((z_pt - Ztt[perm_tg]) ** 2).double()).sum())
    zn = float(Ztt.numel())
    zmse = {f"{a}": zerr[(a, "D2")] / zn for a, d in zerr if d == "D2"}
    for dec in ("D0", "D1", "D2", "D3"):
        zmse[dec] = zerr[("pt_true", dec)] / zn
    pz = torch.cat(ptrue_z); tz = torch.cat(ttrue_z); rh = torch.cat(rho_all)
    ts_mse = {f"{a}_{d}".lower(): float(np.concatenate(v).mean()) for (a, d), v in TSerr.items()}
    ts_out = {"d0": ts_mse.get("pt_true_d0"), "d1": ts_mse.get("pt_true_d1"),
              "d2": ts_mse.get("pt_true_d2"), "d3": ts_mse.get("pt_true_d3"),
              "rand_d2": ts_mse.get("rand_true_d2"), "b1_d2": ts_mse.get("b1_neutral_d2")}
    codec_p = os.path.join(args.root, "codec", f"codec_{args.dataset}_s{args.manifest_seed}.json")
    cj = json.load(open(codec_p))["codec_oracles"]
    ts_out.update({"ctx_mean": cj["ctxmean_mse"], "smean": cj["smean_mse"],
                   "snaive": cj["snaive_mse"], "i0": cj["i0_mse"], "o3": cj["o3_mse"]})
    origins = pairs[:, 0]
    boot = {}
    if ("pt_true", "D2") in TSerr and ("pt_shuffle", "D2") in TSerr:
        boot["d2_true_vs_shuffle_ts"] = CS.block_bootstrap_paired(
            np.concatenate(TSerr[("pt_true", "D2")]), np.concatenate(TSerr[("pt_shuffle", "D2")]),
            origins, block=int(np.ceil((L + H) / 32)))
    rec = {"status": "complete", "stage": "C3", "dataset": args.dataset, "split": args.split,
           "regime": "historical_audit", "manifest_seed": args.manifest_seed,
           "checkpoint": args.ckpt, "carrier": "C2", "frequency": CS.FREQ,
           "norm_const": CS.NORM_CONST, "geometry": "4/10", "mask": "right_future_100",
           "input_arm": "matrix", "backbone_arm": "matrix",
           "random_init_seed": args.random_init_seed, "shuffle_seed": args.shuffle_seed,
           "decoder": "D0-D3", "calibration": None,
           "git_commit": os.environ.get("WM4TS_GIT_COMMIT", "unknown"),
           "manifest": {"manifest_seed": args.manifest_seed,
                        "pairs_sha1": man["pair_sha1"][split_key], "n_pairs": int(n),
                        "n_origins": int(len(np.unique(origins)))},
           "native_metrics": {a: nat[a]["sum"] / nat[a]["n"] for a in ARMS},
           "phase_metrics": {"z_mse": zmse,
                             "pearson_pt_true": CS.pearson(pz, tz),
                             "spearman_pt_true": CS.spearman(pz, tz),
                             "phase_cos": {a: CS.phase_cosine(
                                 CS.decode(torch.from_numpy(AB[a][..., 0]),
                                           torch.from_numpy(AB[a][..., 1]), "D2")[0], Ztt)
                                 for a in AB},
                             "rho_median": float(rh.median()), "rho_mean": float(rh.mean())},
           "ts_metrics": ts_out, "bootstrap": boot,
           "provenance": {"torch": torch.__version__}}
    d = os.path.join(args.root, "signal"); os.makedirs(d, exist_ok=True)
    json.dump(rec, open(os.path.join(d, f"C3_{args.dataset}_s{args.manifest_seed}.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(d, f"C3_{args.dataset}_s{args.manifest_seed}_arrays.npz"),
                        target_z=Zt, pairs=pairs, perm_in=perm_in, perm_tg=perm_tg,
                        **{f"ab_{a}": AB[a] for a in AB})
    print(json.dumps({"native": rec["native_metrics"], "z_mse": zmse,
                      "ts": ts_out, "pearson": rec["phase_metrics"]["pearson_pt_true"]},
                     indent=2, default=float), flush=True)
    print("[done] signal", args.dataset, flush=True)
