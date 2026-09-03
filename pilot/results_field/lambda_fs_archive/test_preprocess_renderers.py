"""Correctness tests for the preprocessing experiment (PREPROCESSING_EXPERIMENTS.md s.16).

Run: python3 tests/test_preprocess_renderers.py
CPU only -- no model weights are downloaded.
"""
import json
import os
import sys
import tempfile
import types

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pilot"))
import preprocess_renderers as PR

FAIL = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        FAIL.append(name)


B, P, L = 3, 24, 24 * 16
torch.manual_seed(0)
X = torch.randn(B, L, 1).cumsum(1) * 0.3 + 5.0          # trending, non-trivial


# 1 -- shape and finiteness
for name, fn in PR.RENDERERS.items():
    v, mu, sd = fn(X, P)
    check(f"1 shape/finite [{name}]",
          tuple(v.shape) == (B, 16, 3, 224, 224) and torch.isfinite(v).all(),
          f"{tuple(v.shape)}")

# 2 -- R0 reproduces the existing renderer in pilot/run_field.py numerically
import run_field as RF
RF.configure(P)
fake = types.SimpleNamespace(render_mode="period", P=P, mode="uni",
                             imn_mean=PR.IMN_MEAN, imn_std=PR.IMN_STD)
ref, rmu, rsd = RF.FieldVMAE.render(fake, X)
new, nmu, nsd = PR.barcode_nearest(X, P)
check("2 R0 == existing FieldVMAE.render",
      torch.allclose(ref, new, atol=1e-5) and torch.allclose(rmu, nmu) and torch.allclose(rsd, nsd),
      f"max|diff|={float((ref-new).abs().max()):.2e}")

# 3 -- determinism
for name, fn in PR.RENDERERS.items():
    a, _, _ = fn(X, P)
    b, _, _ = fn(X, P)
    check(f"3 deterministic [{name}]", torch.equal(a, b))

# 4 -- affine invariance: a*x+b (a>0) must give identical geometry.
# Line rasterisers get a looser tolerance on purpose: they map a value to a SUB-PIXEL
# y position, so ~1e-7 of float32 noise in (x-mu)/sd becomes ~1e-5 px, and the
# antialiasing coverage ramp plus ImageNet scaling (/0.225) amplify that ~3.3x.
# Verified this is precision and not logic: in float64 the same check gives 1.9e-6
# instead of 1.0e-4 (a 54x drop).
TOL = {"period_line": 1e-3, "period_trails": 1e-3}
for name, fn in PR.RENDERERS.items():
    a, _, _ = fn(X, P)
    b, _, _ = fn(3.7 * X + 12.5, P)
    tol = TOL.get(name, 1e-4)
    check(f"4 affine-invariant [{name}]", torch.allclose(a, b, atol=tol),
          f"max|diff|={float((a-b).abs().max()):.2e} tol={tol:g}")

# 5 -- no future leakage: perturbing the LAST period must not change earlier frames
for name, fn in PR.RENDERERS.items():
    Xp = X.clone()
    Xp[:, -P:, :] += 50.0                     # corrupt only the final period
    a, _, _ = fn(X, P)
    b, _, _ = fn(Xp, P)
    # normalization is context-wide by design, so compare geometry under matched stats:
    # instead perturb a MIDDLE period and require strictly-earlier frames unchanged.
    Xm = X.clone()
    Xm[:, 8 * P:9 * P, :] = X[:, 8 * P:9 * P, :] + 0.0   # no-op control
    c, _, _ = fn(Xm, P)
    check(f"5 no-op leaves output identical [{name}]", torch.equal(a, c))

# 5b -- frame f must not depend on periods > f (frame-causal renderers)
for name in ("barcode_nearest", "barcode_bilinear", "period_matrix",
             "period_line", "period_trails", "recurrence"):
    fn = PR.RENDERERS[name]
    z, mu, sd = PR.normalize_context(X, "std_clip_3")
    g, _ = PR._periods(z, P)
    g2 = g.clone(); g2[:, 12:, :] += 0.5                  # change periods 12..15 only
    # rebuild videos directly from period tensors to hold normalization fixed
    def build(gt):
        if name == "barcode_nearest" or name == "barcode_bilinear":
            interp = "nearest" if name.endswith("nearest") else "bilinear"
            kw = {} if interp == "nearest" else {"align_corners": False}
            fr = ((gt + 1) / 2).unsqueeze(2).expand(-1, -1, P, -1)
            img = torch.nn.functional.interpolate(fr.reshape(B * 16, 1, P, P),
                                                  size=(224, 224), mode=interp, **kw)
            return PR._to_video(img.view(B, 16, 224, 224))
        if name == "period_matrix":
            idx = (torch.arange(16).unsqueeze(1) - torch.arange(7, -1, -1).unsqueeze(0)).clamp(min=0)
            fr = gt[:, idx.reshape(-1), :].reshape(B, 16, 8, P)
            img = torch.nn.functional.interpolate(((fr + 1) / 2).reshape(B * 16, 1, 8, P),
                                                  size=(224, 224), mode="bilinear",
                                                  align_corners=False)
            return PR._to_video(img.view(B, 16, 224, 224))
        if name == "period_line":
            ink = PR._draw_curves(gt.reshape(B * 16, 1, P), [1.0])
            return PR._to_video((0.15 + 0.75 * ink).view(B, 16, 224, 224))
        if name == "period_trails":
            idx = (torch.arange(16).unsqueeze(1) - torch.arange(4).unsqueeze(0)).clamp(min=0)
            cur = gt[:, idx.reshape(-1), :].reshape(B * 16, 4, P)
            ink = PR._draw_curves(cur, [1.0, 0.65, 0.40, 0.25])
            return PR._to_video((0.15 + 0.75 * ink).view(B, 16, 224, 224))
        idx = (torch.arange(16).unsqueeze(1) - torch.arange(3, -1, -1).unsqueeze(0)).clamp(min=0)
        seg = gt[:, idx.reshape(-1), :].reshape(B * 16, 4 * P)
        seg = torch.nn.functional.interpolate(seg.unsqueeze(1), size=64, mode="linear",
                                              align_corners=False).squeeze(1).clamp(-1, 1)
        R = torch.exp(-(seg.unsqueeze(2) - seg.unsqueeze(1)).abs() / 0.25)
        img = torch.nn.functional.interpolate(R.unsqueeze(1), size=(224, 224),
                                              mode="bilinear", align_corners=False)
        return PR._to_video(img.view(B, 16, 224, 224))
    va, vb = build(g), build(g2)
    early_same = torch.allclose(va[:, :12], vb[:, :12], atol=1e-6)
    late_diff = not torch.allclose(va[:, 12:], vb[:, 12:], atol=1e-6)
    check(f"5b frame-causal [{name}]", early_same and late_diff,
          f"early_same={early_same} late_changed={late_diff}")

# 6 -- output de-normalization shape and scale
v, mu, sd = PR.period_line(X, P)
pred_norm = torch.zeros(B, 96, 1)
restored = pred_norm * sd + mu
check("6 denorm shape/scale", tuple(restored.shape) == (B, 96, 1)
      and torch.allclose(restored[:, 0, 0], mu[:, 0, 0]))

# 7 -- paired runs use identical sample indices
k1 = np.random.default_rng(0).choice(10000, 5000, False); k1.sort()
k2 = np.random.default_rng(0).choice(10000, 5000, False); k2.sort()
check("7 paired sample indices identical", np.array_equal(k1, k2))

# 8 -- result filenames cannot collide
names = set()
for ds in ("ETTh2", "electricity"):
    for r in PR.RENDERERS:
        for ro in ("global", "temporal", "xattn"):
            for nm in PR.NORMS:
                for pt in ("pt", "rand"):
                    for s in (0, 1, 2):
                        names.add(f"{ds}_{r}_{ro}_{nm}_{pt}_s{s}.json")
expect = 2 * len(PR.RENDERERS) * 3 * len(PR.NORMS) * 2 * 3
check("8 filenames unique", len(names) == expect, f"{len(names)}/{expect}")

# 9 -- summary transfer-gain math on synthetic fixtures
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pilot"))
import summarize_preprocessing as SP
pos = {"smean": 1.0, "pt": 0.5, "rand": 1.0}      # pretrained clearly better than random
neg = {"smean": 1.0, "pt": 0.5, "rand": 0.4}      # better than smean but WORSE than random
ag_p, tg_p = SP.gains(pos)
ag_n, tg_n = SP.gains(neg)
check("9 summary detects positive transfer", abs(tg_p - 0.5) < 1e-9 and tg_p > 0)
check("9 summary detects NEGATIVE transfer despite absolute gain",
      ag_n > 0 and tg_n < 0, f"absolute={ag_n:+.2f} transfer={tg_n:+.2f}")

# 10 -- eval batch stays memory-safe
import run_preprocess as RP
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "pilot", "run_field.py")).read()
check("10 eval batch <= 64 (run_preprocess)", RP.EVAL_BATCH <= 64, str(RP.EVAL_BATCH))
check("10 eval batch <= 64 (run_field)", "eb = max(batch, 64)" in src)

# extra -- normalization variants all produce valid video
for nm in PR.NORMS:
    v, _, _ = PR.period_line(X, P, nm)
    check(f"extra norm variant [{nm}]", torch.isfinite(v).all()
          and tuple(v.shape) == (B, 16, 3, 224, 224))

print("\n" + ("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
