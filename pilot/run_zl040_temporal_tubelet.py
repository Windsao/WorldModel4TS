"""ZL-040: genuine temporal tubelet continuation (Family 4).

frame f = period f. The masked region is the FINAL frames -- future TIME occupying the video's
temporal axis -- not a spatial carrier strip. This is the one masked-reconstruction variant this
repository has never run: every prior test repeated a static image and masked space.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zeroshot_loop as ZL
import videomae_patch_utils as VP

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMG, NF, GRID, PATCH, NTUB = 224, 16, 14, 16, 8
IMN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def period_frames_video(zf):
    """zf [N, 16, P] one period per FRAME -> [N,16,3,224,224]. Frame f renders period f as a
    phase profile tiled across the image, so consecutive frames are consecutive time."""
    N, T, P = zf.shape
    g = ((zf + 1) / 2).clamp(0, 1)
    img = F.interpolate(g.reshape(N * T, 1, 1, P), size=(IMG, IMG), mode="bilinear",
                        align_corners=False).view(N, T, IMG, IMG)
    vid = img.unsqueeze(2).expand(N, T, 3, IMG, IMG)
    return ((vid - IMN_MEAN.to(g.device).unsqueeze(1)) / IMN_STD.to(g.device).unsqueeze(1)).contiguous()


def temporal_tail_mask(n_future_frames, B, device):
    """Mask the FINAL frames: whole temporal tubelets, so future TIME is what is hidden."""
    n_tub = max(1, n_future_frames // 2)
    m = torch.zeros(NTUB, GRID, GRID, dtype=torch.bool, device=device)
    m[NTUB - n_tub:] = True
    return m.reshape(1, -1).expand(B, -1).contiguous(), n_tub


def decode_frames(cubes, n_tub, P):
    """Reconstructed cubes -> values of the masked future frames [N, n_tub*2, P]."""
    vid = VP.unpatchify(cubes)                                   # [N,16,3,224,224]
    tail = vid[:, NF - n_tub * 2:].mean(2).clamp(0, 1)           # [N,f,224,224]
    prof = tail.mean(2)                                          # collapse the tiled axis
    v = F.interpolate(prof.unsqueeze(1), size=(prof.shape[1], P), mode="bilinear",
                      align_corners=False).squeeze(1)
    return v * 2 - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default=ZL.ROOT)
    ap.add_argument("--ckpt", default=os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base"))
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    X, pairs, man = ZL.historical_manifest(args.dataset, args.data_dir, n_max=args.n,
                                           stride=args.stride, seed=args.seed, split="audit")
    P, L, H = man["P"], man["L"], man["H"]
    reps, N = H // P, len(X)
    nfut = max(2, ((reps + 1) // 2) * 2)                          # future frames, even (tubelets)
    nctx = NF - nfut
    print(f"{args.dataset} n={N} P={P} reps={reps} ctx_frames={nctx} future_frames={nfut}", flush=True)

    ctx, fut = X[:, :L], X[:, L:L + H]
    Ct = torch.from_numpy(ctx).to(DEV)
    mu = Ct.mean(1, keepdim=True); sd = Ct.std(1, keepdim=True) + 1e-6
    zc = ((Ct - mu) / (3 * sd)).clamp(-1, 1).view(N, L // P, P)[:, -nctx:]
    zf = ((torch.from_numpy(fut).to(DEV) - mu) / (3 * sd)).clamp(-1, 1).view(N, reps, P)
    zf_pad = torch.cat([zf, zf[:, -1:].expand(-1, nfut - reps, -1)], 1) if nfut > reps else zf

    from transformers import VideoMAEForPreTraining, VideoMAEConfig
    mp = VideoMAEForPreTraining.from_pretrained(args.ckpt).to(DEV).eval()
    torch.manual_seed(args.seed)
    mr = VideoMAEForPreTraining(VideoMAEConfig.from_pretrained(args.ckpt)).to(DEV).eval()

    acc = {k: {"e": 0.0, "n": 0} for k in ("pt", "rand", "zero")}
    ts = {k: [] for k in ("pt", "rand", "zero")}
    for i in range(0, N, args.batch):
        s = slice(i, min(i + args.batch, N))
        B = zc[s].shape[0]
        blank = torch.zeros(B, nfut, P, device=DEV)
        vin = period_frames_video(torch.cat([zc[s], blank], 1))     # future frames BLANK
        vfull = period_frames_video(torch.cat([zc[s], zf_pad[s]], 1))
        bm, n_tub = temporal_tail_mask(nfut, B, DEV)
        tgt, cmean, cstd = VP.native_targets(vfull)
        lab = tgt[bm].view(B, -1, 1536)
        with torch.no_grad():
            lp = mp(pixel_values=vin, bool_masked_pos=bm).logits
            lr = mr(pixel_values=vin, bool_masked_pos=bm).logits
        for k, lg in (("pt", lp), ("rand", lr), ("zero", torch.zeros_like(lp))):
            e = ((lg - lab) ** 2).double()
            acc[k]["e"] += float(e.sum()); acc[k]["n"] += e.numel()
            cin = VP.patchify(VP.unnormalize(vin))
            sm, ss = cmean[bm].view(B, -1, 1, 3), cstd[bm].view(B, -1, 1, 3)
            filled = cin.clone()
            filled[bm] = VP.denormalize_cubes(lg, sm, ss).view(-1, 512, 3)
            vals = decode_frames(filled, n_tub, P)[:, :reps]
            pred = (vals.reshape(B, -1)[:, :H] * 3 * sd[s] + mu[s]).cpu().numpy()
            ts[k].append(((pred - fut[s]) ** 2).mean(1))
    res = {f"native_{k}": acc[k]["e"] / acc[k]["n"] for k in acc}
    res["native_pt_over_zero"] = res["native_pt"] / res["native_zero"]
    res["native_pt_over_rand"] = res["native_pt"] / res["native_rand"]
    for k in ts:
        v = np.concatenate(ts[k]); res[f"ts_{k}_mse"] = float(v.mean()); res[f"{k}_percase"] = v
    pr = ZL.priors(ctx, P, H)
    _, base, _ = ZL.pseudo_origin_select(ctx, P, H, pr)
    res["base_ts_mse"] = float(np.mean((base - fut) ** 2))
    res["base_percase"] = ((base - fut) ** 2).mean(1)
    blk = int(np.ceil((L + H) / man["stride"]))
    for b in ("rand", "zero", "base"):
        res[f"boot_pt_vs_{b}"] = ZL.paired_bootstrap(res["pt_percase"], res[f"{b}_percase"],
                                                     pairs[:, 0], blk)
    show = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
    print(json.dumps(show, indent=2, default=float), flush=True)
    out = {"status": "complete", "candidate_id": "ZL-040", "dataset": args.dataset,
           "split": "audit", "manifest": man, "checkpoint": args.ckpt,
           "ctx_frames": nctx, "future_frames": nfut, "masked_tubelets": nfut // 2,
           "config_hash": ZL.cfg_hash(vars(args)), "results": show,
           "wall_clock_s": round(time.time() - t0, 1)}
    d = os.path.join(args.root, "candidates", "ZL-040"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"metrics_{args.dataset}.json"), "w"), indent=2, default=float)
    print("[done]", args.dataset, flush=True)


if __name__ == "__main__":
    main()
