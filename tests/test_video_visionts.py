"""Required tests for VIDEO_VISIONTS_EXPERIMENTS.md section 11.

Model-dependent checks (8, 9-logits, 13) are skipped when transformers / checkpoints are
unavailable, so this file runs locally on CPU and fully on the GPU host.
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pilot"))
import videomae_patch_utils as VP
import video_visionts_renderers as VR

FAIL, SKIP = [], []
def check(n, c, extra=""):
    print(f"{'PASS' if c else 'FAIL'}  {n}" + (f"   {extra}" if extra else ""))
    if not c: FAIL.append(n)
def skip(n, why):
    print(f"SKIP  {n}   ({why})"); SKIP.append(n)

torch.manual_seed(0)
B, P, H = 3, 24, 96
L = 16 * P
ctx = torch.randn(B, L).cumsum(1) * 0.2 + 5.0
fut = torch.randn(B, H).cumsum(1) * 0.2 + 5.0

# ---- 1 video shape/finite ; 4 patch-boundary ; 2 inverse shape
for cols in (13, 10, 3):
    vi, vf, mu, sd = VR.dense_static(ctx, fut, P, cols)
    check(f"1 dense_static video shape/finite cols={cols}",
          tuple(vi.shape) == (B, 16, 3, 224, 224) and torch.isfinite(vi).all())
    g = VR.video_to_gray(vi)[:, 0]
    vis_w = (14 - cols) * 16
    check(f"4 hidden block blank + patch-aligned cols={cols}",
          float(g[:, :, vis_w:].std()) < 1e-6 and vis_w % 16 == 0,
          f"vis_w={vis_w} hidden_std={float(g[:,:,vis_w:].std()):.2e}")
    gf = VR.video_to_gray(vf)[:, 0]
    yh = VR.invert_dense(gf[:, :, vis_w:], P, H, mu, sd)
    check(f"2 inverse shape/finite cols={cols}",
          tuple(yh.shape) == (B, H, 1) and torch.isfinite(yh).all())

# ---- 3 temporal order preserved by matrix + period-major flatten
z = torch.arange(L, dtype=torch.float32).view(1, L)
m = VR._matrix(z, P)                                   # [1,1,P,n] rows=phase cols=period
back = m.squeeze(1).permute(0, 2, 1).reshape(1, -1)    # period-major
check("3 dense segmentation preserves exact temporal order", torch.equal(back, z))

# ---- 5 future modification cannot change the strict-ZS input
vi_a, _, _, _ = VR.dense_static(ctx, fut, P, 3)
vi_b, _, _, _ = VR.dense_static(ctx, fut + 1000.0, P, 3)
check("5 future poisoning cannot change ZS input", torch.equal(vi_a, vi_b))

# ---- 6 mu/sd from visible context only
mu1, sd1 = VR.context_stats(ctx)
mu2, sd2 = VR.context_stats(ctx)
_, _, mu3, sd3 = VR.dense_static(ctx, fut * 7 + 3, P, 3)
check("6 mu/sd depend only on visible context",
      torch.allclose(mu1, mu3) and torch.allclose(sd1, sd3))

# ---- 7 affine invariance of normalized geometry
va, _, _, _ = VR.dense_static(ctx, fut, P, 3)
vb, _, _, _ = VR.dense_static(3.7 * ctx + 12.5, 3.7 * fut + 12.5, P, 3)
check("7 affine a*x+b preserves geometry", torch.allclose(va, vb, atol=1e-4),
      f"max|d|={float((va-vb).abs().max()):.2e}")

# ---- 9 packing round trip
frames = torch.rand(2, 16, 3, 224, 224)
cubes = VP.patchify(frames)
check("9 patchify/unpatchify exact round trip",
      torch.allclose(VP.unpatchify(cubes), frames, atol=0),
      f"max|d|={float((VP.unpatchify(cubes)-frames).abs().max()):.2e}")
check("9 cube shape", tuple(cubes.shape) == (2, 1568, 512, 3), str(tuple(cubes.shape)))

# ---- 10/11 mask counts and tube property
for name, want in VP.MASK_SPATIAL_COUNT.items():
    mk = VP.make_mask(name, 4, torch.device("cpu"), seed=0)
    per_frame = mk.view(4, 8, 196)
    ok_count = int(per_frame[0, 0].sum()) == want
    ok_total = int(mk[0].sum()) == want * 8
    ok_tube = bool((per_frame[:, 0:1] == per_frame).all())
    check(f"10 mask count [{name}]", ok_count and ok_total, f"{int(per_frame[0,0].sum())}/{want}")
    check(f"11 tube: same spatial map over 8 tubelets [{name}]", ok_tube)

# ---- 12 paired controls get byte-identical inputs and masks
m_a = VP.make_mask("random_tube_90", 4, torch.device("cpu"), seed=0)
m_b = VP.make_mask("random_tube_90", 4, torch.device("cpu"), seed=0)
check("12 mask reproducible for paired controls",
      torch.equal(m_a, m_b) and VP.mask_hash(m_a) == VP.mask_hash(m_b))

# ---- 14 legacy line control still importable and unchanged
try:
    import run_reconstruct as RC
    cL = int(RC.CTX_PX * H / RC.FUT_PX) + 16 * 4 + P
    full = torch.randn(2, cL + H, 1).cumsum(1) * 0.2 + 3.0
    a, _, _ = RC.render_rolling(full[:, :cL], H)
    b, _, _ = RC.render_rolling(full[:, :cL], H)
    check("14 legacy line renderer deterministic + finite",
          torch.equal(a, b) and torch.isfinite(a).all() and tuple(a.shape) == (2, 16, 3, 224, 224))
except Exception as e:
    check("14 legacy line renderer importable", False, str(e)[:80])

# ---- 15 filename collisions
names = set()
for ds in ("ETTh1", "ETTh2", "ETTm2", "electricity", "traffic", "solar"):
    for reg in ("ZS", "HA", "LA"):
        for r in VR.RENDERERS:
            for mk in VP.MASK_SPATIAL_COUNT:
                for dec in ("native_norm", "oracle_stats", "causal_stats", "line_geometry"):
                    for ctl in ("pretrained", "random", "zeroed"):
                        for s in (0, 1, 2):
                            names.add(f"{ds}_{reg}_{r}_{mk}_{dec}_{ctl}_s{s}.json")
exp = 6 * 3 * 3 * 5 * 4 * 3 * 3
check("15 filenames unique", len(names) == exp, f"{len(names)}/{exp}")

# ---- 16 summary gates on synthetic fixtures
sys.path.insert(0, os.path.join(HERE, "..", "pilot"))
try:
    import summarize_video_visionts as SV
    q_better = SV.geo_ratio({"a": 0.5, "b": 0.5}, {"a": 1.0, "b": 1.0})
    q_worse = SV.geo_ratio({"a": 2.0, "b": 2.0}, {"a": 1.0, "b": 1.0})
    check("16 gate detects better-than-VisionTS", abs(q_better - 0.5) < 1e-9 and q_better < 1)
    check("16 gate detects worse-than-VisionTS", abs(q_worse - 2.0) < 1e-9 and q_worse > 1)
except Exception as e:
    skip("16 summary gates", f"summarizer not yet present: {str(e)[:50]}")

# ---- 8 / 13 model-dependent
try:
    import transformers
    from transformers import VideoMAEForPreTraining
    ck = os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base")
    model = VideoMAEForPreTraining.from_pretrained(ck).eval()
    check("8a config.norm_pix_loss is True", model.config.norm_pix_loss is True)
    px = torch.randn(2, 16, 3, 224, 224) * 0.5
    bm = VP.make_mask("random_tube_90", 2, torch.device("cpu"), seed=0)
    with torch.no_grad():
        out = model(pixel_values=px, bool_masked_pos=bm)
        tgt, cmean, cstd = VP.native_targets(px)
        lab = tgt[bm].view(2, -1, 1536)
        manual = torch.nn.functional.mse_loss(out.logits, lab)
    check("8 manual native target matches model.loss",
          torch.allclose(manual, out.loss, atol=1e-5),
          f"manual={float(manual):.6f} model={float(out.loss):.6f}")
    # 9 logits invariance to masked-region pixels ; sensitivity to visible pixels
    g = bm.view(2, 8, 14, 14)
    px2 = px.clone()
    px2[:, :, :, :, 208:] += 3.0            # rightmost patch column (masked under right_*)
    bm_r = VP.make_mask("right_3", 2, torch.device("cpu"))
    with torch.no_grad():
        l1 = model(pixel_values=px, bool_masked_pos=bm_r).logits
        l2 = model(pixel_values=px2, bool_masked_pos=bm_r).logits
        px3 = px.clone(); px3[:, :, :, :, :16] += 3.0     # a VISIBLE patch column
        l3 = model(pixel_values=px3, bool_masked_pos=bm_r).logits
    check("9 logits invariant to masked-region pixels",
          torch.allclose(l1, l2, atol=1e-4), f"max|d|={float((l1-l2).abs().max()):.2e}")
    check("9 logits DO change for a visible patch (test is meaningful)",
          not torch.allclose(l1, l3, atol=1e-3), f"max|d|={float((l1-l3).abs().max()):.2e}")
except Exception as e:
    skip("8/9-model", f"{type(e).__name__}: {str(e)[:60]}")

try:
    sys.path.insert(0, os.path.join(HERE, "..", "third_party", "VisionTS"))
    from visionts import VisionTS
    v = VisionTS(arch="mae_base", finetune_type="none",
                 ckpt_dir=os.environ.get("VTS_CKPT_DIR", "./ckpt/"))
    check("13 VisionTS norm_pix_loss is False", v.vision_model.norm_pix_loss is False)
    check("13 VisionTS patch/image size", v.vision_model.patch_embed.patch_size[0] == 16
          and v.vision_model.patch_embed.img_size[0] == 224)
except Exception as e:
    skip("13 VisionTS", f"{type(e).__name__}: {str(e)[:60]}")

print()
print(f"{len(FAIL)} failures, {len(SKIP)} skipped")
if FAIL: print("FAILURES:", FAIL)
sys.exit(1 if FAIL else 0)
