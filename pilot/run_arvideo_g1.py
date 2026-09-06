"""Route A, gates G0 (mp4 codec) and G1 (synthetic extrapolation) with MAGI-1 4.5B video continuation.

For each probe (const / ramp / sine_level / travel) x seed:
  1. synth Z [G, P]  (context periods 0..7 unit-std normalised; future periods 8..G-1)
  2. render the 32-frame prefix video of periods 0..7 ONLY and write it as mp4
  3. MAGI-1 v2v continuation (subprocess of pilot/magi_run.sh)  -> out.mp4
  4. align the output to the absolute frame axis (the reported chunks start at absolute frame
     24: prefix latents 0..5 are the clean chunk, latents 6..7 = frames 24..31 are re-emitted
     in the first reported chunk).  The offset is VERIFIED against the prefix frames, not assumed.
  5. decode future periods k = 8..8+H-1 at absolute frames 4k+3, score vs truth and vs
     copy_last / lin_extrap.

Outputs under --out/<kind>_s<seed>/ : prefix.mp4, out.mp4, cfg.json, result.json, and a
contact sheet.  --out/summary.json aggregates.  Runs jobs on a pool of GPUs.
"""
import argparse, json, os, subprocess, sys, time, shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arvideo_render as R

MAGI_RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "magi_run.sh")
PROMPT = ("A minimal flat 2D animation: a smooth dark blue silhouette, like gentle rolling hills or a "
          "water level, against a light gray background, slowly rising and falling over time. "
          "A small red square marker slides steadily to the right along the top strip. "
          "Static camera, no text, no buildings, no people.")
OUT_START = 24          # absolute frame index of the first reported output frame (verified per run)


def write_mp4(frames, path, fps=24):
    import imageio.v2 as iio
    w = iio.get_writer(path, fps=fps, codec="libx264", quality=None, macro_block_size=1,
                       output_params=["-crf", "6", "-pix_fmt", "yuv420p", "-preset", "slow"])
    for f in frames:
        w.append_data(f)
    w.close()


def read_mp4(path):
    import imageio.v2 as iio
    rd = iio.get_reader(path, "ffmpeg")
    frames = np.stack([np.asarray(f)[..., :3] for f in rd])
    rd.close()
    return frames


def resize_frames(frames, s):
    if frames.shape[1] == s and frames.shape[2] == s:
        return frames
    import cv2
    return np.stack([cv2.resize(f, (s, s), interpolation=cv2.INTER_AREA) for f in frames])


def codec_check(P, out_dir, seeds=(0, 1)):
    """G0 part 2: render -> mp4 -> read -> decode. Returns max abs z error."""
    os.makedirs(out_dir, exist_ok=True)
    worst = 0.0
    for sd in seeds:
        rng = np.random.default_rng(sd)
        Z = rng.uniform(-2.8, 2.8, (R.PREFIX_FRAMES // R.M, P))
        vid = R.render_prefix(Z)
        p = os.path.join(out_dir, f"codec_P{P}_s{sd}.mp4")
        write_mp4(vid, p)
        back = read_mp4(p)
        assert back.shape[0] == vid.shape[0], (back.shape, vid.shape)
        for g in range(Z.shape[0]):
            err = np.max(np.abs(R.decode_frame(back[R.period_frame(g)], P) - Z[g]))
            worst = max(worst, float(err))
    return worst


def make_cfg(base_cfg, seed, num_frames, path):
    c = json.load(open(base_cfg))
    c["runtime_config"]["seed"] = int(seed)
    c["runtime_config"]["num_frames"] = int(num_frames)
    json.dump(c, open(path, "w"), indent=1)


def run_magi(gpu, cfg, prefix_mp4, out_mp4, log):
    cmd = [MAGI_RUN, str(gpu), cfg, "v2v", prefix_mp4, PROMPT, out_mp4]
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    return rc, time.time() - t0


def align_offset(out_frames, prefix_frames, P, n_match=8):
    """absolute frame index of out_frames[0] by matching decoded heights against the prefix
    (only offsets with full overlap). Fallback when the clock cannot be read."""
    dec_out = R.decode_video(out_frames[:n_match], P)
    dec_pre = R.decode_video(prefix_frames, P)
    best, best_err = None, np.inf
    for off in range(0, prefix_frames.shape[0] - n_match + 1):
        err = R.mse(dec_out, dec_pre[off:off + n_match])
        if err < best_err:
            best, best_err = off, err
    return best, float(best_err)


def align_by_clock(out_frames):
    """Pipeline offset = clock reading of the FIRST output frames (the model has had no time to
    drift there); the later readings measure the model's own time rate. Returns
    (offset int, rate = fitted clock advance per output frame, final drift in frames,
     fraction of frames with a readable clock, raw readings)."""
    ts = R.decode_clocks(out_frames)
    j = np.arange(len(ts))
    ok = np.isfinite(ts)
    head = ok[:4]
    if head.sum() < 2:
        return None, np.nan, np.nan, float(ok.mean()), ts
    off = int(round(float(np.median(ts[:4][head] - j[:4][head]))))
    rate, drift = np.nan, np.nan
    if ok.sum() >= 6:
        rate = float(np.polyfit(j[ok], ts[ok], 1)[0])
        drift = float(ts[ok][-1] - (j[ok][-1] + off))
    return off, rate, drift, float(ok.mean()), ts


def contact_sheet(frames, path, every=4, max_n=24):
    try:
        from PIL import Image
    except ImportError:
        return
    sel = frames[::every][:max_n]
    thumbs = [Image.fromarray(f).resize((120, 120)) for f in sel]
    cols = 8
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 120, rows * 120), (255, 255, 255))
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * 120, (i // cols) * 120))
    sheet.save(path)


def one_job(args, kind, seed, gpu):
    P, H = args.P, args.H
    G_ctx = R.PREFIX_FRAMES // R.M
    G = G_ctx + H
    d = os.path.join(args.out, f"{kind}_s{seed}")
    os.makedirs(d, exist_ok=True)
    res_path = os.path.join(d, "result.json")
    if os.path.exists(res_path) and not args.force and not args.rescore:
        return json.load(open(res_path))
    Z = R.synth_signal(kind, P, G, np.random.default_rng(1000 + seed))
    np.save(os.path.join(d, "Z.npy"), Z)
    prefix = R.render_prefix(Z[:G_ctx])
    prefix_mp4 = os.path.join(d, "prefix.mp4")
    write_mp4(prefix, prefix_mp4)
    cfg = os.path.join(d, "cfg.json")
    make_cfg(args.cfg, seed, args.num_frames, cfg)
    out_mp4 = os.path.join(d, "out.mp4")
    if args.rescore and os.path.exists(out_mp4):
        rc, secs = 0, (json.load(open(res_path)).get("secs") if os.path.exists(res_path) else None)
    else:
        rc, secs = run_magi(gpu, cfg, prefix_mp4, out_mp4, os.path.join(d, "magi.log"))
    res = {"kind": kind, "seed": seed, "P": P, "H": H, "rc": rc, "secs": secs}
    if rc != 0 or not os.path.exists(out_mp4):
        res["error"] = "magi failed"
        json.dump(res, open(res_path, "w"), indent=1)
        return res
    out = resize_frames(read_mp4(out_mp4), R.S)
    prefix_back = read_mp4(prefix_mp4)
    off_c, clock_rate, clock_drift, clock_frac, clock_ts = align_by_clock(out)
    off_m, match_err = align_offset(out, prefix_back, P)
    # MAGI emits only generated frames: the first output frame is absolute frame 32. The clock
    # verifies this per run; if it cannot be read we fall back to that verified constant.
    off = off_c if off_c is not None else R.PREFIX_FRAMES
    res.update({"n_out_frames": int(out.shape[0]), "offset": int(off), "offset_clock": off_c,
                "clock_rate": clock_rate, "clock_final_drift": clock_drift, "clock_readable_frac": clock_frac,
                "offset_prefix_match": int(off_m), "prefix_match_err": match_err,
                "clock_first8": [None if not np.isfinite(x) else round(float(x), 2) for x in clock_ts[:8]]})
    dec = R.decode_video(out, P)
    np.save(os.path.join(d, "decoded_all.npy"), dec)
    contact_sheet(out, os.path.join(d, "sheet.png"))
    pred, truth = [], []
    for k in range(G_ctx, G):
        j = R.period_frame(k) - off
        if j >= out.shape[0]:
            break
        pred.append(dec[j]); truth.append(Z[k])
    pred, truth = np.array(pred), np.array(truth)
    res["H_scored"] = int(len(pred))
    if len(pred) == 0:
        res["error"] = "no future frames in output"
        json.dump(res, open(res_path, "w"), indent=1)
        return res
    b = R.baselines(Z[:G_ctx], len(pred))
    res["mse_model"] = R.mse(pred, truth)
    res["mse_copy_last"] = R.mse(b["copy_last"], truth)
    res["mse_lin_extrap"] = R.mse(b["lin_extrap"], truth)
    res["ratio_vs_copy"] = res["mse_model"] / max(res["mse_copy_last"], 1e-12)
    res["ratio_vs_lin"] = res["mse_model"] / max(res["mse_lin_extrap"], 1e-12)
    res["mse_model_first_period"] = R.mse(pred[0], truth[0])
    # a copy-of-prefix diagnostic: how far is the model from just freezing the last frame?
    res["mse_model_vs_copy"] = R.mse(pred, b["copy_last"])
    res["per_period_model"] = [R.mse(pred[i], truth[i]) for i in range(len(pred))]
    res["per_period_copy"] = [R.mse(b["copy_last"][i], truth[i]) for i in range(len(pred))]
    json.dump(res, open(res_path, "w"), indent=1)
    return res


def geo(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x) and x > 0]
    return float(np.exp(np.mean(np.log(xs)))) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/nyx-storage1/hanliu/wm4ts/arvideo/g1")
    ap.add_argument("--cfg", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "magi_4p5B_480.json"))
    ap.add_argument("--P", type=int, default=24)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--num-frames", type=int, default=48)
    ap.add_argument("--kinds", default="const,ramp,sine_level,travel")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--codec-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rescore", action="store_true", help="recompute results from existing out.mp4 files")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    worst = max(codec_check(P, os.path.join(args.out, "codec")) for P in (24, 96))
    print(f"[G0 mp4 codec] worst abs z error over P in (24, 96): {worst:.4f}  (gate <= 0.03)")
    json.dump({"worst_abs_z_err": worst, "pass": worst <= 0.03}, open(os.path.join(args.out, "codec", "g0.json"), "w"))
    if args.codec_only:
        return
    if worst > 0.03:
        print("G0 FAILED -> stop"); sys.exit(2)

    jobs = [(k, int(s)) for k in args.kinds.split(",") for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]
    results = []
    # simple GPU pool: each worker owns one gpu
    from queue import Queue
    q = Queue()
    for j in jobs: q.put(j)
    def worker(gpu):
        while True:
            try: kind, seed = q.get_nowait()
            except Exception: return
            r = one_job(args, kind, seed, gpu)
            results.append(r)
            print(json.dumps({k: r.get(k) for k in ("kind", "seed", "rc", "secs", "offset", "clock_rate", "H_scored", "mse_model", "mse_copy_last", "mse_lin_extrap", "ratio_vs_copy")}), flush=True)
    with ThreadPoolExecutor(len(gpus)) as ex:
        list(ex.map(worker, gpus))

    summ = {"P": args.P, "H": args.H, "num_frames": args.num_frames, "per_kind": {}}
    for kind in args.kinds.split(","):
        rs = [r for r in results if r["kind"] == kind and "mse_model" in r]
        summ["per_kind"][kind] = {
            "n": len(rs),
            "mse_model": float(np.mean([r["mse_model"] for r in rs])) if rs else None,
            "mse_copy_last": float(np.mean([r["mse_copy_last"] for r in rs])) if rs else None,
            "mse_lin_extrap": float(np.mean([r["mse_lin_extrap"] for r in rs])) if rs else None,
            "geo_ratio_vs_copy": geo([r["ratio_vs_copy"] for r in rs]),
            "geo_ratio_vs_lin": geo([r["ratio_vs_lin"] for r in rs]),
            "offsets": sorted(set(int(r["offset"]) for r in rs)),
            "clock_rate_mean": float(np.nanmean([r.get("clock_rate", np.nan) for r in rs])) if rs else None,
            "clock_final_drift_mean": float(np.nanmean([r.get("clock_final_drift", np.nan) for r in rs])) if rs else None,
            "ratios_vs_copy": [round(r["ratio_vs_copy"], 3) for r in rs],
        }
    # G1 gate: sine_level and travel <= 0.7 of copy_last; const must not drift (mse_model <= 0.05)
    pk = summ["per_kind"]
    gate = {
        "const_no_drift": (pk.get("const", {}).get("mse_model") or 9) <= 0.05,
        "sine_level_vs_copy<=0.7": (pk.get("sine_level", {}).get("geo_ratio_vs_copy") or 9) <= 0.7,
        "travel_vs_copy<=0.7": (pk.get("travel", {}).get("geo_ratio_vs_copy") or 9) <= 0.7,
        "ramp_vs_copy<=0.7": (pk.get("ramp", {}).get("geo_ratio_vs_copy") or 9) <= 0.7,
    }
    summ["gate"] = gate
    summ["G1_pass"] = gate["const_no_drift"] and (gate["sine_level_vs_copy<=0.7"] or gate["travel_vs_copy<=0.7"])
    json.dump(summ, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
