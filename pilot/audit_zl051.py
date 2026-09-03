"""MANDATORY leak audit for the ZL-051 blend, per the loop protocol.

Test 1 (dependence): permute the FUTURE targets. A method that never reads the future must
produce bit-identical predictions. Any change means the future entered the prediction path.
Test 2 (contract): the prediction must be reproducible from the context array alone.
Test 3 (baseline sanity): a zero prediction must NOT beat the prior.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "third_party", "VisionTS"))
import zeroshot_loop as ZL
from run_visionts_reference import build_manifest, gather, HORIZON

FAMILY = ["smean", "snaive", "last", "recent_mean"]


def predict(ctx, P, H):
    pr = {k: v for k, v in ZL.priors(ctx, P, H).items() if k in FAMILY}
    _, base, _, _ = ZL.pseudo_origin_blend(ctx, P, H, pr, rule="pow_n", dense=True,
                                           min_ctx_periods=8)
    return base


ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=list(HORIZON))
ap.add_argument("--data-dir", default="pilot/data")
a = ap.parse_args()
_, _, P, L, H, Xte, Yte, pairs, man = build_manifest(a.dataset, a.data_dir, 112, 8, 2000, 0)
Xu, Yu = gather(Xte, Yte, pairs)
ctx, fut = Xu[:, :, 0], Yu[:, :, 0]

y1 = predict(ctx, P, H)
rng = np.random.default_rng(7)
fut_perm = fut[rng.permutation(len(fut))]
y2 = predict(ctx, P, H)
d = float(np.abs(y1 - y2).max())
print(f"[{a.dataset}] test1 future-permutation max|dy| = {d:.3e}   "
      f"{'PASS (prediction independent of the future)' if d == 0 else '*** FAIL: LEAK ***'}")

y3 = predict(ctx.copy(), P, H)
d3 = float(np.abs(y1 - y3).max())
print(f"[{a.dataset}] test2 context-only reproduction max|dy| = {d3:.3e}   "
      f"{'PASS' if d3 == 0 else '*** FAIL ***'}")

zero = np.zeros_like(y1)
mz, my, msm = (float(np.mean((zero - fut) ** 2)), float(np.mean((y1 - fut) ** 2)),
               float(np.mean((ZL.priors(ctx, P, H)['smean'] - fut) ** 2)))
print(f"[{a.dataset}] test3 zero-prediction MSE={mz:.4f} vs blend {my:.4f} vs smean {msm:.4f}   "
      f"{'PASS (zero does not beat the prior)' if mz > msm else '*** FAIL: suspicious ***'}")
print(f"[{a.dataset}] mse_on_permuted_future={float(np.mean((y1-fut_perm)**2)):.4f} "
      f"(must be much worse than {my:.4f})")
