"""ZL-090 -- masked temporal inpainting with CAUSAL de-normalisation.

Why this is not ZL-040 again. VideoMAE-base was pretrained with norm_pix_loss=True, so its
decoder predicts per-patch NORMALISED pixels and structurally cannot emit absolute values. ZL-040
worked around that by de-normalising with statistics taken from a video that contained the true
future -- oracle-statistic leakage, caught by the zero-logit sanity check (zero logits "beat" the
prior 2.3x).

The honest version of the same decomposition: supply the per-patch mean and std of every masked
future patch from the CONTEXT-ONLY prior (the ZL-051 blend), and let the backbone supply only the
normalised SHAPE inside each patch. That is exactly the division of labour where a video model
could plausibly add something: it is not asked for the level or the scale, only for the local
pattern. No future value is read anywhere.

Falsification: reconstruct with (a) the pretrained decoder, (b) a random-init decoder, and
(c) ZERO logits, which reduces exactly to the prior. If the pretrained arm does not beat both,
the generative route is dead in its last leak-free form.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "third_party", "VisionTS"))
import zeroshot_loop as ZL
import videomae_patch_utils as VP
from run_visionts_reference import build_manifest, HORIZON

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAMILY = ["smean", "snaive", "last", "recent_mean"]
S, T = 224, 16
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def render(z):
    """[B,G,P] in [-1,1] -> [B,16,3,224,224]: frame f is period f (real motion on the frame axis)."""
    import torch.nn.functional as F
    B, G, P = z.shape
    img = F.interpolate(((z + 1) / 2).reshape(B * G, 1, 1, P), size=(S, S), mode="bilinear",
                        align_corners=False).reshape(B, G, S, S)
    return img.unsqueeze(2).expand(B, G, 3, S, S).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HORIZON))
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default=ZL.ROOT)
    ap.add_argument("--ckpt", default="MCG-NJU/videomae-base")
    ap.add_argument("--max-ch", type=int, default=64)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    data, borders, P, L, H, Xte, Yte, pairs, man = build_manifest(a.dataset, a.data_dir,
                                                                  a.max_ch, 8, a.n, a.seed)
    from run_visionts_reference import gather
    Xu, Yu = gather(Xte, Yte, pairs)
    ctx, fut = Xu[:, :, 0], Yu[:, :, 0]
    G, reps = L // P, max(1, H // P)
    pr = {k_: v for k_, v in ZL.priors(ctx, P, H).items() if k_ in FAMILY}
    _, blend, _, _ = ZL.pseudo_origin_blend(ctx, P, H, pr, rule="pow_n", dense=True,
                                            min_ctx_periods=8)
    N = len(ctx)
    print(f"[ZL-090] {a.dataset} n={N} P={P} G={G} reps={reps} blend={np.mean((blend-fut)**2):.4f}",
          flush=True)
    # the model sees context periods AND the PRIOR's forecast periods; the future tubelets are
    # masked, so their content is predicted, but their per-patch statistics come from the prior.
    full = np.concatenate([ctx, blend], 1)[:, -(G + reps) * P:]
    Gf = full.shape[1] // P
    C = torch.from_numpy(full).float()
    mu, sd = C.mean(1, keepdim=True), C.std(1, keepdim=True) + 1e-6
    zf = ((C - mu) / (3 * sd)).clamp(-1, 1).view(N, Gf, P)
    if Gf != T:
        idx = (torch.arange(T).float() * (Gf - 1) / (T - 1)).round().long()
        zf = zf[:, idx]
        fut_frames = (idx >= (Gf - reps)).nonzero().flatten().tolist()
    else:
        fut_frames = list(range(Gf - reps, Gf))
    from transformers import VideoMAEForPreTraining, VideoMAEConfig
    mp = VideoMAEForPreTraining.from_pretrained(a.ckpt).to(DEV).eval()
    torch.manual_seed(a.seed)
    mr = VideoMAEForPreTraining(VideoMAEConfig.from_pretrained(a.ckpt)).to(DEV).eval()
    tub = sorted({f // 2 for f in fut_frames})
    mask = torch.zeros(T // 2, 14, 14, dtype=torch.bool)
    mask[tub] = True
    bm = mask.flatten()
    print(f"  masked tubelets={tub} -> {int(bm.sum())}/{bm.numel()} tokens", flush=True)
    res = {"blend_mse": float(np.mean((blend - fut) ** 2)),
           "smean_mse": float(np.mean((ZL.priors(ctx, P, H)["smean"] - fut) ** 2))}
    for nm in ("pt", "rand", "zero"):
        model = mr if nm == "rand" else mp
        outs = []
        for i in range(0, N, a.batch):
            zb = zf[i:i + a.batch].to(DEV)
            v01 = render(zb)                                   # [b,16,3,224,224] in [0,1]
            pv = (v01 - IM_MEAN.to(DEV)) / IM_STD.to(DEV)      # ImageNet-normalised input
            b = pv.shape[0]
            with torch.no_grad():
                # CAUSAL statistics: `pv` was built with the PRIOR occupying the future frames,
                # so every per-cube mean/std here is a function of the context only. This is the
                # exact place where ZL-040 leaked, and the exact fix.
                _, cmean, cstd = VP.native_targets(pv)
                cubes = VP.patchify(VP.unnormalize(pv))        # [b,N,P,C] raw pixels
                if nm == "zero":
                    lg = torch.zeros(b, int(bm.sum()), cubes.shape[-2] * cubes.shape[-1],
                                     device=DEV)
                else:
                    lg = model(pixel_values=pv,
                               bool_masked_pos=bm[None].expand(b, -1).to(DEV)).logits
                filled = cubes.clone()
                filled[:, bm] = VP.denormalize_cubes(lg, cmean[:, bm], cstd[:, bm])
                frames = VP.unpatchify(filled)                 # [b,16,3,224,224]
                # frame f is period f; read the series back as the per-frame row profile
                prof = frames.mean(2).mean(1)                  # [b,224] mean over C and rows
                seg = torch.nn.functional.interpolate(prof[:, None, None, :], size=(1, P),
                                                      mode="bilinear", align_corners=False)
                rows = frames.mean(2)                          # [b,T,224,224]
                colprof = rows.mean(2)                         # [b,T,224] column profile/frame
                per = torch.nn.functional.interpolate(colprof.unsqueeze(1), size=(T, P),
                                                      mode="bilinear",
                                                      align_corners=False).squeeze(1)
                outs.append((per * 2 - 1).cpu().numpy())       # [b,T,P] back to [-1,1]
        rec = np.concatenate(outs)                             # [N,T,P]
        flat = rec.reshape(N, T * P)
        y = flat[:, -H:] * 3 * sd.numpy() + mu.numpy()
        res[f"{nm}_mse"] = float(np.mean((y - fut) ** 2))
        print(f"  {nm:5s} = {res[f'{nm}_mse']:.4f}", flush=True)
    res["pt_over_zero"] = res["pt_mse"] / res["zero_mse"]
    res["pt_over_rand"] = res["pt_mse"] / res["rand_mse"]
    res["pt_over_blend"] = res["pt_mse"] / res["blend_mse"]
    print(json.dumps(res, indent=2, default=float), flush=True)
    d = os.path.join(a.root, "zl090"); os.makedirs(d, exist_ok=True)
    json.dump({"status": "complete", "candidate_id": "ZL-090", "dataset": a.dataset,
               "manifest": man, "results": res, "wall_clock_s": round(time.time() - t0, 1)},
              open(os.path.join(d, f"{a.dataset}.json"), "w"), indent=2, default=float)
    print("[done]", a.dataset, flush=True)


if __name__ == "__main__":
    main()
