"""C0-C4 runner: codec oracles, carrier-aware baselines, conditionality controls,
all-tubelet phase decoding. ETTh2 and ETTm2 only; genuine test futures are forbidden here."""

import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_field as RF
import videomae_patch_utils as VP
import video_visionts_renderers as VR
import video_visionts_layouts as VL
import video_visionts_mask_factorization as MF
import video_visionts_carrier_signal as CS
from run_video_visionts_recovery_audit import build_partitions, gather_hist, HORIZON

DEV = "cuda" if torch.cuda.is_available() else "cpu"
G, FC, NT, PATCH, NF = CS.GRID, CS.FUT_COLS, CS.NTUB, 16, 16


def git_info():
    if os.environ.get("WM4TS_GIT_COMMIT"):
        return os.environ["WM4TS_GIT_COMMIT"], os.environ.get("WM4TS_GIT_DIRTY") == "1"
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), None
    except Exception:
        return "unknown", None


def load_split(dataset, data_dir, max_ch, split, manifest_seed):
    data, P, L, H, parts, man = build_partitions(dataset, data_dir, max_ch, seed=manifest_seed)
    pairs = parts[split]
    X = gather_hist(data, pairs, L, H)
    return X, pairs, P, L, H, man


def i0_forecast(model, xb, P, L, H):
    """Incumbent: dense_static + nearest-visible-physical inversion, same windows."""
    B = xb.shape[0]
    vi, _, mu, sd = VR.dense_static(xb[:, :L], torch.zeros(B, H, device=xb.device), P, FC)
    sp = MF.make_matched_mask("right_future_100", 0)
    bm = VL.tube_mask_from_spatial(sp, B, DEV)
    with torch.no_grad():
        lg = model(pixel_values=vi, bool_masked_pos=bm).logits
        cin = VP.patchify(VP.unnormalize(vi))
        m_, s_ = VL.nearest_visible_physical(cin, bm)
        f = cin.clone()
        f[bm] = VP.denormalize_cubes(lg, m_.unsqueeze(2) if m_.dim() == 3 else m_,
                                     s_.unsqueeze(2) if s_.dim() == 3 else s_).view(-1, 512, 3)
        img = VP.unpatchify(f)[:, NF - 1].mean(1).clamp(0, 1) * 2 - 1
    return VR.invert_dense(img[:, :, CS.CTX_COLS * PATCH:], P, H, mu, sd)


def stage_codec(args):
    from transformers import VideoMAEForPreTraining
    mp = VideoMAEForPreTraining.from_pretrained(args.ckpt).to(DEV).eval()
    out = {}
    for split in ("validation", "audit"):
        X, pairs, P, L, H, man = load_split(args.dataset, args.data_dir, args.max_ch,
                                            {"validation": "val", "audit": "audit"}[split],
                                            args.manifest_seed)
        acc = {k: 0.0 for k in ("o0", "o1", "o2", "o3", "i0", "ctxmean", "smean", "snaive")}
        maxo0 = maxdiff = 0.0
        clip, n = [], 0
        zq = []
        for i in range(0, len(X), args.batch):
            xb = torch.from_numpy(X[i:i + args.batch]).to(DEV)
            B = xb.shape[0]
            ctx, fut = xb[:, :L], xb[:, L:L + H]
            o = CS.codec_oracles(ctx, fut, P, H)
            clip.append(o["clip_frac"]); zq.append(o["zf"].flatten().cpu().numpy()[::13])
            yt = fut.unsqueeze(-1)
            gfull = torch.zeros(B, G, G, device=DEV)
            gfull[:, :, CS.CTX_COLS:] = o["g2"]
            bm = MF.to_tube(MF.make_matched_mask("right_future_100", 0), B, DEV)
            tgt, _ = CS.native_future_targets(CS.carrier_video(gfull), bm)
            o3, _ = CS.o3_from_targets(tgt, P, H, o["mu"], o["sd"])
            maxo0 = max(maxo0, float((o["o0"] - yt).abs().max()))
            maxdiff = max(maxdiff, float((o["o2"] - o3).abs().max()))
            reps = (H + P - 1) // P; Gp = L // P
            sm = torch.from_numpy(np.tile(ctx.reshape(B, Gp, P).mean(1).cpu().numpy(),
                                          (1, reps))[:, :H]).to(DEV).unsqueeze(-1)
            sn = torch.from_numpy(np.tile(ctx[:, -P:].cpu().numpy(), (1, reps))[:, :H]).to(DEV).unsqueeze(-1)
            cm = o["mu"].unsqueeze(-1).expand(B, H, 1)
            i0 = i0_forecast(mp, xb, P, L, H)
            for k, v in (("o0", o["o0"]), ("o1", o["o1"]), ("o2", o["o2"]), ("o3", o3),
                         ("i0", i0), ("ctxmean", cm), ("smean", sm), ("snaive", sn)):
                acc[k] += float((((v - yt) ** 2).double()).sum())
            n += B * H
        z = np.concatenate(zq)
        mse = {k: acc[k] / n for k in acc}
        out[split] = {"clip_frac": float(np.mean(clip)),
                      "z_p01": float(np.percentile(z, 1)), "z_p05": float(np.percentile(z, 5)),
                      "z_median": float(np.median(z)), "z_p95": float(np.percentile(z, 95)),
                      "z_p99": float(np.percentile(z, 99)),
                      "o0_max_abs_err": maxo0, "o2_o3_max_abs_diff": maxdiff,
                      "o0_mse": mse["o0"], "o1_mse": mse["o1"], "o2_mse": mse["o2"],
                      "o3_mse": mse["o3"], "i0_mse": mse["i0"], "ctxmean_mse": mse["ctxmean"],
                      "smean_mse": mse["smean"], "snaive_mse": mse["snaive"],
                      "o3_over_ctxmean": mse["o3"] / mse["ctxmean"],
                      "o3_over_i0": mse["o3"] / mse["i0"],
                      "o1_over_ctxmean": mse["o1"] / mse["ctxmean"],
                      "n_pairs": int(len(X)), "pairs_sha1": man["pair_sha1"][
                          {"validation": "val", "audit": "audit"}[split]]}
        print(f"[{split}] " + json.dumps({k: round(v, 6) for k, v in out[split].items()
                                          if isinstance(v, float)}), flush=True)
    commit, dirty = git_info()
    rec = {"status": "complete", "stage": "codec", "dataset": args.dataset, "split": "both",
           "regime": "historical_audit", "manifest_seed": args.manifest_seed,
           "checkpoint": args.ckpt, "carrier": "C2", "frequency": CS.FREQ,
           "norm_const": CS.NORM_CONST, "geometry": "4/10", "mask": "right_future_100",
           "input_arm": "oracle", "backbone_arm": "n/a", "random_init_seed": None,
           "shuffle_seed": None, "decoder": "D1", "calibration": None, "git_commit": commit,
           "manifest": {"manifest_seed": args.manifest_seed,
                        "pairs_sha1": out["audit"]["pairs_sha1"],
                        "n_pairs": out["audit"]["n_pairs"]},
           "codec_oracles": out["audit"], "codec_validation": out["validation"],
           "provenance": {"torch": torch.__version__, "git_dirty": dirty}}
    d = os.path.join(args.root, "codec"); os.makedirs(d, exist_ok=True)
    json.dump(rec, open(os.path.join(d, f"codec_{args.dataset}_s{args.manifest_seed}.json"), "w"), indent=2)
    print("[done] codec", args.dataset, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["codec", "signal"])
    ap.add_argument("--dataset", required=True, choices=["ETTh2", "ETTm2"])
    ap.add_argument("--split", default="audit", choices=["audit", "validation"])
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default="pilot/results_field/video_visionts_carrier_signal")
    ap.add_argument("--ckpt", default=os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base"))
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--manifest-seed", type=int, default=0)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--random-init-seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    t0 = time.time()
    if args.stage == "codec":
        stage_codec(args)
    else:
        from run_video_visionts_carrier_signal_stage2 import stage_signal
        stage_signal(args)
    print(f"[time] {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
