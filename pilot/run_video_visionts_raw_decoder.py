"""Stage R6 (conditional): raw-pixel decoder adaptation (spec section 9).

Gate that triggered this stage: native/oracle historical reconstruction is still useful
(G0 native ratio 0.9687 pretrained vs 1.1905 random-init) while no corrected lens passed the
R4 historical gate (L1 1.0347, L2 1.0204, L3 1.0216).

Freeze the patch embedding and encoder. Train ONLY the decoder and prediction projection to
predict raw [0,1] cube pixels instead of per-cube normalized targets -- this removes the
`norm_pix_loss` bottleneck at its source rather than trying to re-estimate the discarded
statistics. Trained solely on historical pseudo-mask tasks.

Required pair: pretrained frozen encoder vs random frozen encoder, identical decoder init,
manifests, batches, steps and stopping rule.
"""

import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import videomae_patch_utils as VP
import video_visionts_renderers as VR
import video_visionts_layouts as VL
from run_video_visionts_recovery_audit import (make_historical_pair, build_partitions,
                                               gather_hist, HORIZON, SEL)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID, PATCH, NF = 14, 16, 16


def prep(model, xb, P, L, H, vc, pmask, need_target=True):
    B = xb.shape[0]
    cols = GRID - vc
    vi, vf, mu, sd = make_historical_pair(xb, P, H, L, cols)
    bm = VL.tube_mask_from_spatial(pmask, B, DEV)
    cubes_true = VP.patchify(VP.unnormalize(vf)) if need_target else None
    tgt = cubes_true[bm].view(B, -1, 512 * 3) if need_target else None
    return vi, bm, mu, sd, tgt, VP.patchify(VP.unnormalize(vi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--root", default="pilot/results_field/video_visionts_recovery")
    ap.add_argument("--ckpt", default=os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base"))
    ap.add_argument("--layout", default="G0")
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--random-backbone", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    from transformers import VideoMAEForPreTraining, VideoMAEConfig
    cfg = VideoMAEConfig.from_pretrained(args.ckpt)
    if args.random_backbone:
        torch.manual_seed(args.seed)
        model = VideoMAEForPreTraining(cfg)
    else:
        model = VideoMAEForPreTraining.from_pretrained(args.ckpt)
    model = model.to(DEV)
    # freeze patch embedding + encoder; adapt decoder + prediction projection only
    for p in model.videomae.parameters():
        p.requires_grad_(False)
    model.videomae.eval()
    torch.manual_seed(args.seed)               # identical decoder init in both arms
    for m in model.decoder.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    train_params = [p for p in model.decoder.parameters()]
    for p in train_params:
        p.requires_grad_(True)
    n_par = sum(p.numel() for p in train_params)
    print(f"R6 decoder params {n_par/1e6:.2f}M  backbone="
          f"{'random' if args.random_backbone else 'pretrained'}", flush=True)

    pool = {}
    for ds in SEL:
        data, P, L, H, parts, man = build_partitions(ds, args.data_dir, args.max_ch, seed=args.seed)
        vc, _ = VL.width_rule(L, H)
        _, _, pm = VL.build_layout(args.layout, vc, 0)
        pool[ds] = dict(P=P, L=L, H=H, vc=vc, pm=pm,
                        tr=gather_hist(data, parts["train"], L, H),
                        va=gather_hist(data, parts["val"], L, H),
                        au=gather_hist(data, parts["audit"], L, H))

    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.05)
    rng = np.random.default_rng(args.seed)

    def evaluate(split, cap=128):
        model.decoder.eval()
        out = {}
        with torch.no_grad():
            for ds, v in pool.items():
                X = v[split][:cap]
                num = den = 0.0
                for i in range(0, len(X), args.batch):
                    xb = torch.from_numpy(X[i:i + args.batch]).to(DEV)
                    vi, bm, mu, sd, tgt, cubes_in = prep(model, xb, v["P"], v["L"], v["H"], v["vc"], v["pm"])
                    logits = model(pixel_values=vi, bool_masked_pos=bm).logits
                    raw = logits.clamp(0, 1)
                    filled = cubes_in.clone()
                    filled[bm] = raw.view(-1, 512, 3)
                    img = VP.unpatchify(filled)[:, NF - 1].mean(1).clamp(0, 1) * 2 - 1
                    yh = VR.invert_dense(img[:, :, v["vc"] * PATCH:], v["P"], v["H"], mu, sd)
                    # L0 reference through the identical inverse path
                    m_, s_ = VL.nearest_visible_physical(cubes_in, bm)
                    f0 = cubes_in.clone()
                    f0[bm] = VP.denormalize_cubes(torch.zeros_like(logits), m_, s_).view(-1, 512, 3)
                    img0 = VP.unpatchify(f0)[:, NF - 1].mean(1).clamp(0, 1) * 2 - 1
                    y0 = VR.invert_dense(img0[:, :, v["vc"] * PATCH:], v["P"], v["H"], mu, sd)
                    yt = torch.from_numpy(X[i:i + args.batch, v["L"]:v["L"] + v["H"]]).to(DEV).unsqueeze(-1)
                    num += float(((yh - yt) ** 2).sum()); den += float(((y0 - yt) ** 2).sum())
                out[ds] = {"r6_mse": round(num / max(1, len(X) * v["H"]), 6),
                           "L0_mse": round(den / max(1, len(X) * v["H"]), 6),
                           "ratio": round(num / max(den, 1e-12), 6)}
        model.decoder.train()
        return out, math.exp(sum(math.log(out[d]["ratio"]) for d in out) / len(out))

    best, best_state, bad, hist = float("inf"), None, 0, []
    model.decoder.train()
    for step in range(args.steps):
        ds = list(pool)[step % len(pool)]
        v = pool[ds]
        idx = rng.choice(len(v["tr"]), args.batch, False)
        xb = torch.from_numpy(v["tr"][idx]).to(DEV)
        vi, bm, mu, sd, tgt, _ = prep(model, xb, v["P"], v["L"], v["H"], v["vc"], v["pm"])
        logits = model(pixel_values=vi, bool_masked_pos=bm).logits
        loss = F.mse_loss(logits, tgt)                       # RAW [0,1] cube pixels
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % args.val_every == 0:
            _, r = evaluate("va")
            hist.append((step + 1, round(r, 5)))
            if r < best - 1e-4:
                best, bad = r, 0
                best_state = {k: x.detach().clone() for k, x in model.decoder.state_dict().items()}
            else:
                bad += 1
            print(f"[info] step {step+1} loss {loss.item():.5f} R_val {r:.4f} best {best:.4f}", flush=True)
            if bad >= args.patience:
                print("[info] early stop", flush=True); break
    if best_state is not None:
        model.decoder.load_state_dict(best_state)
    audit, agg = evaluate("au", cap=512)
    print(f"[audit] {json.dumps(audit)}  aggregate={agg:.4f}", flush=True)

    import transformers
    out = {"status": "complete", "stage": "R6", "regime": "LA",
           "backbone": "random" if args.random_backbone else "pretrained",
           "layout": args.layout, "decoder_params": n_par, "steps": args.steps,
           "lr": args.lr, "batch": args.batch, "seed": args.seed, "val_history": hist,
           "best_R_val": round(best, 6), "audit": audit, "aggregate_ratio_vs_L0": round(agg, 6),
           "git_commit": os.environ.get("WM4TS_GIT_COMMIT", "unknown"),
           "torch": torch.__version__, "transformers": transformers.__version__,
           "wall_clock_s": round(time.time() - t0, 1)}
    d = os.path.join(args.root, "raw_decoder"); os.makedirs(d, exist_ok=True)
    tag = "rand" if args.random_backbone else "pt"
    json.dump(out, open(os.path.join(d, f"R6_{tag}_s{args.seed}.json"), "w"), indent=2)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
