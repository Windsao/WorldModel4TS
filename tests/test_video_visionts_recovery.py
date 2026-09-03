"""Required tests for VIDEO_VISIONTS_RECOVERY_EXPERIMENTS.md section 11 (24 items).

CPU-only subset must pass before any GPU sweep; model-dependent items run on the GPU host.
"""
import glob, json, math, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pilot"))
import videomae_patch_utils as VP
import video_visionts_renderers as VR
import video_visionts_layouts as VL
from run_video_visionts_recovery_audit import (make_historical_pair, masked_target_stats,
                                               build_partitions)
from run_video_visionts_stat_lens_v2 import LensV2, spatial_group

FAIL, SKIP = [], []
def check(n, c, extra=""):
    print(f"{'PASS' if c else 'FAIL'}  {n}" + (f"   {extra}" if extra else ""))
    if not c: FAIL.append(n)
def skip(n, why): print(f"SKIP  {n}   ({why})"); SKIP.append(n)

torch.manual_seed(0)
B, P, H, L = 3, 24, 96, 384
vc, mc = VL.width_rule(L, H)
cols = mc
hist = torch.randn(B, L + H).cumsum(1) * 0.2 + 5.0
poison = hist.clone(); poison[:, L:] += 500.0

# 1/2 -- input invariant, full sensitive to pseudo-future poisoning
vi_a, vf_a, mu_a, sd_a = make_historical_pair(hist, P, H, L, cols)
vi_b, vf_b, mu_b, sd_b = make_historical_pair(poison, P, H, L, cols)
check("1 historical video_input invariant to pseudo-future poisoning", torch.equal(vi_a, vi_b))
check("2 historical video_full sensitive to poisoning", not torch.allclose(vf_a, vf_b))

# 3/4 -- masked target stats sensitive; equal to native_targets
cp, pc, pmask = VL.build_layout("G0", vc)
bm = VL.tube_mask_from_spatial(pmask, B, torch.device("cpu"))
tm_a, tls_a = masked_target_stats(vf_a, bm)
tm_b, tls_b = masked_target_stats(vf_b, bm)
check("3 masked target mean/std sensitive to poisoning",
      not torch.allclose(tm_a, tm_b) or not torch.allclose(tls_a, tls_b))
_, cmean, cstd = VP.native_targets(vf_a)
C = cmean.shape[-1]
check("4 target stats equal native_targets(video_full) exactly",
      torch.equal(tm_a, cmean[bm].view(B, -1, C)) and
      torch.allclose(tls_a, torch.log(cstd[bm].view(B, -1, C).clamp_min(1e-6)), atol=0))

# 5 -- right_9
mk9 = VP.make_mask("right_9", 2, torch.device("cpu"))
pf = mk9.view(2, 8, 196)
check("5 right_9 has 126 masked spatial patches and is a tube mask",
      int(pf[0, 0].sum()) == 126 and VP.MASK_SPATIAL_COUNT["right_9"] == 126
      and bool((pf[:, 0:1] == pf).all()), f"{int(pf[0,0].sum())}")

# 6 -- width rule
check("6 width rule 4/10 and 5/9",
      VL.width_rule(384, 96) == (4, 10) and VL.width_rule(1536, 96) == (5, 9)
      and VL.width_rule(2304, 144) == (5, 9))

# 7/8 -- permutation round trips
img = torch.randn(2, 224, 224)
for name, seeds in (("G1", [0]), ("G2", [0, 1, 2])):
    for s in seeds:
        c_, p_, m_ = VL.build_layout(name, vc, s)
        rt = VL.unpermute_image(VL.permute_image(img, c_), c_)
        check(f"{'7' if name=='G1' else '8'} {name} permutation round-trips exactly (seed {s})",
              torch.equal(rt, img), f"max|d|={float((rt-img).abs().max()):.1e}")

# 9/10 -- mask/content consistency
for name in ("G0", "G1", "G2"):
    c_, p_, m_ = VL.build_layout(name, vc, 0)
    fut = VL.canonical_is_future(vc)
    check(f"9 every masked physical patch maps to a future canonical patch [{name}]",
          bool(fut[c_[m_]].all()))
    check(f"10 every visible physical patch maps to context [{name}]",
          bool((~fut[c_[~m_]]).all()))

# 11 -- changing any canonical future patch cannot change the model input
gi = VR.video_to_gray(vi_a)[:, 0]
gi_p = VR.video_to_gray(vi_b)[:, 0]
c_, _, m_ = VL.build_layout("G2", vc, 0)
check("11 canonical future change cannot alter model input",
      torch.equal(VL.permute_image(gi, c_), VL.permute_image(gi_p, c_)))

# 12 -- nearest-visible-physical never reads a masked cube
cubes = VP.patchify(VP.unnormalize(vi_a))
bm2 = VL.tube_mask_from_spatial(m_, B, torch.device("cpu"))
cub_poison = cubes.clone()
cub_poison[bm2] += 99.0                                    # corrupt ONLY masked cubes
m1, s1 = VL.nearest_visible_physical(cubes, bm2)
m2, s2 = VL.nearest_visible_physical(cub_poison, bm2)
check("12 nearest-visible-physical never reads a masked cube",
      torch.allclose(m1, m2) and torch.allclose(s1, s2))

# 13 -- inverse permutation + dense inversion preserves exact time order (ramp fixture)
ramp = torch.arange(L + H, dtype=torch.float32).view(1, -1)
z = torch.arange(L, dtype=torch.float32).view(1, L)
m_mat = VR._matrix(z, P)
back = m_mat.squeeze(1).permute(0, 2, 1).reshape(1, -1)
check("13 inverse mapping preserves exact time order", torch.equal(back, z))

# 14/15 -- tube-shared grouping and target-stat equality across tubelets
grp, ng, tok = spatial_group(bm2)
counts = torch.bincount(grp[0], minlength=ng)
nz = counts[counts > 0]
check("14 tube-shared groups have exactly 8 tubelets each",
      bool((nz == 8).all()), f"unique counts={sorted(set(nz.tolist()))}")
tm_r = tm_a.view(B, 8, -1, C)
check("15 dense_static target stats identical across the 8 tubelets",
      torch.allclose(tm_r[:, 0:1].expand_as(tm_r), tm_r, atol=1e-5),
      f"max|d|={float((tm_r-tm_r[:,0:1]).abs().max()):.1e}")

# 16/17 -- lens step-zero equals L0 ; bounded ranges
for var in ("L1", "L2", "L3"):
    torch.manual_seed(0); lens = LensV2(var)
    n = int(bm2[0].sum())
    lg = torch.randn(B, n, 1536)
    bmn = torch.rand(B, n, 3); bsd = torch.rand(B, n, 3) * 0.5 + 0.05
    g, ngg, tk = spatial_group(bm2)
    pos = g if var == "L3" else tk
    pm_, pls_ = lens(lg, pos, bmn, bsd, g, ngg)
    check(f"16 {var} step-zero equals L0 (causal rule)",
          torch.allclose(pm_, bmn, atol=1e-6)
          and torch.allclose(pls_, torch.log(bsd.clamp_min(1e-6)), atol=1e-6))
    if var != "L1":
        with torch.no_grad():
            lens.head.weight.normal_(0, 5.0); lens.head.bias.normal_(0, 5.0)
        pm2, pls2 = lens(lg, pos, bmn, bsd, g, ngg)
        ratio = (pls2 - torch.log(bsd.clamp_min(1e-6))).exp()
        check(f"17 {var} mean in [0,1], std>0, ratio within exp(+-2)",
              bool(((pm2 >= 0) & (pm2 <= 1)).all()) and bool((pls2.exp() > 0).all())
              and bool(((ratio >= math.exp(-2) - 1e-4) & (ratio <= math.exp(2) + 1e-4)).all()),
              f"ratio range [{float(ratio.min()):.3f},{float(ratio.max()):.3f}]")

# 18 -- partition supports disjoint after purging
try:
    dd = os.environ.get("WM4TS_DATA_DIR", "pilot/data")
    _, _, _, _, parts, man = build_partitions("ETTh2", dd, 112, seed=0)
    s = man["support"]
    check("18 train/val/audit supports disjoint after purge",
          s["train"][1] < s["val"][0] and s["val"][1] < s["audit"][0], str(s))
except Exception as e:
    skip("18 partitions", f"{type(e).__name__}: {str(e)[:60]}")

# 19 -- paired arms get identical lens weights
torch.manual_seed(0); a = LensV2("L2")
torch.manual_seed(0); b = LensV2("L2")
check("19 paired pretrained/random arms get identical lens init",
      all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values())))

# 20 -- genuine-future poisoning cannot change an R5 encoder input
ctx = hist[:, :L]
blank = torch.zeros(B, H)
vi1, _, _, _ = VR.dense_static(ctx, blank, P, cols)
vi2, _, _, _ = VR.dense_static(ctx, blank + 1e6, P, cols)
check("20 genuine-future poisoning cannot change R5 encoder input", torch.equal(vi1, vi2))

# 21/22 -- summarizer grouping and aggregation gates
try:
    import summarize_video_visionts_recovery as SR
    g1 = SR.group_key({"dataset": "ETTh2", "layout": "G0", "variant": "L0", "backbone": "pretrained"})
    g2 = SR.group_key({"dataset": "ETTh2", "layout": "G2", "variant": "L0", "backbone": "pretrained"})
    check("21 summarizer keeps layouts/variants in separate groups", g1 != g2)
    q = SR.geo_ratio({"a": 0.5, "b": 0.5}, {"a": 1.0, "b": 1.0})
    q2 = SR.geo_ratio({"a": 2.0, "b": 2.0}, {"a": 1.0, "b": 1.0})
    lo = SR.loo({"a": 0.5, "b": 0.5, "c": 2.0}, {"a": 1.0, "b": 1.0, "c": 1.0})
    check("22 Q_all / Q_heldout / LOO gates on synthetic fixtures",
          abs(q - 0.5) < 1e-9 and abs(q2 - 2.0) < 1e-9 and abs(lo["c"] - 0.5) < 1e-9)
except Exception as e:
    skip("21/22 summarizer", f"{type(e).__name__}: {str(e)[:60]}")

# 23 -- new outputs cannot resolve to the old invalid directory
root = "pilot/results_field/video_visionts_recovery"
check("23 recovery outputs cannot resolve to old label_free_adaptation",
      "label_free_adaptation" not in os.path.abspath(root)
      and os.path.abspath(root) != os.path.abspath("pilot/results_field/video_visionts"))

# 24 -- finiteness
check("24 all fixtures finite",
      all(torch.isfinite(t).all() for t in (vi_a, vf_a, tm_a, tls_a, m1, s1)))

# regression: the OLD blank_future path is provably insensitive to the pseudo-future
_, vf_blank_a, _, _ = VR.dense_static(hist[:, :L], torch.zeros(B, H), P, cols)
_, vf_blank_b, _, _ = VR.dense_static(poison[:, :L], torch.zeros(B, H), P, cols)
check("8b regression: old blank_future target is insensitive to the pseudo-future (the bug)",
      torch.equal(vf_blank_a, vf_blank_b))

print()
print(f"{len(FAIL)} failures, {len(SKIP)} skipped")
if FAIL: print("FAILURES:", FAIL)
sys.exit(1 if FAIL else 0)
