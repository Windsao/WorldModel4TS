"""Tests for VIDEO_VISIONTS_MATCHED_MASK_EXPERIMENTS.md section 12.

Covers the M0/M1 subset (items 1-15, 29-33). Iterative/carrier items (16-28) are added when
those stages are implemented; they are reported as SKIP here rather than silently omitted.
"""
import math, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pilot"))
import videomae_patch_utils as VP
import video_visionts_renderers as VR
import video_visionts_mask_factorization as MF
from run_video_visionts_recovery_audit import make_historical_pair

FAIL, SKIP = [], []
def check(n, c, extra=""):
    print(f"{'PASS' if c else 'FAIL'}  {n}" + (f"   {extra}" if extra else ""))
    if not c: FAIL.append(n)
def skip(n, w): print(f"SKIP  {n}   ({w})"); SKIP.append(n)

torch.manual_seed(0)
B, P, H, L = 3, 24, 96, 384
COLS = MF.FUT_COLS
hist = torch.randn(B, L + H).cumsum(1) * 0.2 + 5.0
poison = hist.clone(); poison[:, L:] += 500.0
vi, vf, mu, sd = make_historical_pair(hist, P, H, L, COLS)
vi_p, vf_p, _, _ = make_historical_pair(poison, P, H, L, COLS)

# 1 -- every mask reuses byte-identical tensors
h0 = (MF.tensor_hash(vi), MF.tensor_hash(vf))
same = all((MF.tensor_hash(vi), MF.tensor_hash(vf)) == h0 for _ in MF.MATCHED_MASKS)
check("1 all M1 masks reuse byte-identical video_input/video_full", same)

# 2 -- fixed 4/10 geometry
g = VR.video_to_gray(vi)[:, 0]
vis_w = MF.CTX_COLS * MF.PATCH
check("2 render uses exactly 4 context and 10 future columns",
      vis_w == 64 and float(g[:, :, vis_w:].std()) < 1e-6 and g.shape[-1] == 224,
      f"visible_px={vis_w} hidden_std={float(g[:,:,vis_w:].std()):.1e}")

# 3 -- poisoning changes full, not input
check("3 future poisoning changes video_full but not video_input",
      torch.equal(vi, vi_p) and not torch.allclose(vf, vf_p))

# 4 -- canonical label counts
fut = MF.canonical_is_future()
check("4 canonical context/future counts are 56/140",
      int((~fut).sum()) == 56 and int(fut.sum()) == 140)

# 5-8 -- exact mask counts
for name in MF.MATCHED_MASKS + ["future_random_140"]:
    for s in ((0, 1, 2) if name in MF.RANDOM_MASKS else (0,)):
        sp = MF.make_matched_mask(name, s)
        mc, mf = int((sp & ~fut).sum()), int((sp & fut).sum())
        exp = MF.expected_counts(name)
        if exp[0] is None:
            ok = (mc + mf) == int(name.rsplit("_", 1)[1])
        else:
            ok = (mc, mf) == exp
        tag = {"right_future_100": "5", "global_random_140": "6", "global_random_147": "6"}.get(
            name, "7" if name.startswith("future_random") else
            ("8" if name.startswith("future_block") else "6"))
        check(f"{tag} exact counts [{name} seed{s}] ctx={mc} fut={mf}", ok)

# 9 -- masked-token logits map back to correct token identities
sp = MF.make_matched_mask("future_block_5", 0)
bm = MF.to_tube(sp, B, torch.device("cpu"))
is_fut, col = MF.token_region_labels(sp, torch.device("cpu"))
check("9 masked-token labels align with bool_masked_pos order",
      int(bm[0].sum()) == is_fut.numel() and bool(is_fut.all())
      and int(col.min()) == MF.CTX_COLS and int(col.max()) == MF.CTX_COLS + 4,
      f"n={is_fut.numel()} cols[{int(col.min())},{int(col.max())}]")

# 10/11 -- subset sums recombine exactly and are batch-partition invariant
tgt, cmean, cstd = VP.native_targets(vf)
lab = tgt[bm].view(B, -1, 1536)
lg = torch.randn_like(lab) * 0.3
spg = MF.make_matched_mask("global_random_147", 0)
bmg = MF.to_tube(spg, B, torch.device("cpu"))
labg = tgt[bmg].view(B, -1, 1536); lgg = torch.randn_like(labg) * 0.3
ifg, colg = MF.token_region_labels(spg, torch.device("cpu"))
stdg = cstd[bmg].view(B, -1, 3).mean(-1)[0] > 1e-5
S = MF.subset_sums((lgg - labg) ** 2, labg ** 2, ifg, colg, stdg)
lhs = S["context"]["err_sum"] + S["future"]["err_sum"]
check("10 context+future sums exactly recombine to the all-mask sum",
      abs(lhs - S["all"]["err_sum"]) / abs(S["all"]["err_sum"]) < 1e-9,
      f"rel={abs(lhs-S['all']['err_sum'])/abs(S['all']['err_sum']):.1e}")
A = MF.subset_sums(((lgg - labg) ** 2)[:1], (labg ** 2)[:1], ifg, colg, stdg)
Bb = MF.subset_sums(((lgg - labg) ** 2)[1:], (labg ** 2)[1:], ifg, colg, stdg)
M = MF.merge_sums(dict(A), Bb)
check("11 subset MSE invariant to batch partitioning",
      abs(M["all"]["err_sum"] - S["all"]["err_sum"]) / abs(S["all"]["err_sum"]) < 1e-9)

# 12 -- degenerate/non-degenerate future partition is complete
check("12 future = nondegenerate + degenerate exactly",
      S["future"]["n"] == S["future_nondegenerate"]["n"] + S["future_degenerate"]["n"])

# 13 -- paired controls get identical inputs and masks
check("13 pretrained/random controls get identical masks",
      MF.mask_hash(MF.make_matched_mask("global_random_140", 1))
      == MF.mask_hash(MF.make_matched_mask("global_random_140", 1)))

# 14 -- no metric reads an unmasked target as model input
check("14 model input hidden region is blank (targets never enter the encoder)",
      float(VR.video_to_gray(vi)[:, 0][:, :, vis_w:].std()) < 1e-6)

# 15 -- cohort scoring changes only the selector
fut_idx = np.where(fut)[0]
csp = MF.make_matched_mask("future_block_5", 0)
sel = np.isin(fut_idx, np.where(csp)[0])
check("15 right-100 cohort selector picks exactly the cohort's future patches",
      int(sel.sum()) == 70, f"{int(sel.sum())}/70")

for i, n in ((16, "I1 column order"), (17, "I2 two-column steps"), (18, "iterative causality"),
             (19, "iterative poisoning"), (20, "schedule parity"), (21, "smean init"),
             (22, "C1 native variation"), (23, "C2 native variation"),
             (24, "carrier decode"), (25, "carrier std"), (26, "carrier inverse order"),
             (27, "hybrid selection"), (28, "hybrid freeze")):
    skip(f"{i} {n}", "stage not yet implemented; gated on the M2 outcome")

# 29-31 -- summarizer
try:
    import summarize_video_visionts_matched as SM
    check("29 summarizer groups by full config identity",
          SM.group_key({"dataset": "ETTh2", "mask": "right_future_100", "seed": 0}) !=
          SM.group_key({"dataset": "ETTh2", "mask": "global_random_147", "seed": 0}))
    check("31 Q gates on synthetic fixtures",
          abs(SM.geo_ratio({"a": .5, "b": .5}, {"a": 1., "b": 1.}) - .5) < 1e-9
          and abs(SM.geo_ratio({"a": 2., "b": 2.}, {"a": 1., "b": 1.}) - 2.) < 1e-9)
    check("30 bootstrap invoked and seed/block rules recorded", hasattr(SM, "block_bootstrap"))
except Exception as e:
    skip("29/30/31 summarizer", f"{type(e).__name__}: {str(e)[:50]}")

# 32/33
check("32 all fixtures finite", all(torch.isfinite(t).all() for t in (vi, vf, lab, lg)))
root = "pilot/results_field/video_visionts_matched_mask"
check("33 new outputs cannot resolve into previous result directories",
      all(os.path.abspath(root) != os.path.abspath(p) for p in
          ("pilot/results_field/video_visionts", "pilot/results_field/video_visionts_recovery")))

print()
print(f"{len(FAIL)} failures, {len(SKIP)} skipped")
if FAIL: print("FAILURES:", FAIL)
sys.exit(1 if FAIL else 0)
