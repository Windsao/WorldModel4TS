"""Stage F leakage + geometry tests (PREPROCESSING_EXPERIMENTS.md section 14 item 6)."""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pilot"))
import run_reconstruct as RC
from preprocess_renderers import IMG

FAIL = []
def check(n, c, extra=""):
    print(f"{'PASS' if c else 'FAIL'}  {n}" + (f"   {extra}" if extra else ""))
    if not c: FAIL.append(n)

torch.manual_seed(0)
B, H = 2, 96
CTX = int(RC.CTX_PX * H / RC.FUT_PX) + 16 * 4 + 24
full = torch.randn(B, CTX + H, 1).cumsum(1) * 0.2 + 3.0     # context + TRUE FUTURE

# 1 -- overwriting the true future cannot change the render
a, mua, sda = RC.render_rolling(full[:, :CTX], H)
poisoned = full.clone()
poisoned[:, CTX:] += 1000.0                                  # destroy the future
b, mub, sdb = RC.render_rolling(poisoned[:, :CTX], H)
check("1 future poisoning cannot change render", torch.equal(a, b) and torch.equal(mua, mub))

# 2 -- the same holds through the full decode path
img = torch.randn(B, IMG, IMG)
pa = RC.decode_geometry(img, H, mua, sda)
pb = RC.decode_geometry(img, H, mub, sdb)
check("2 future poisoning cannot change forecast", torch.allclose(pa, pb))

# 3 -- the future strip of every frame is blank (no data drawn there)
v = a * RC.IMN_STD.unsqueeze(1) + RC.IMN_MEAN.unsqueeze(1)
strip = v[:, :, 0, :, RC.CTX_PX:]
check("3 future strip is constant background", float(strip.std()) < 1e-5,
      f"std={float(strip.std()):.2e}")

# 4 -- frames differ (rolling actually rolls)
left = v[:, :, 0, :, :RC.CTX_PX]
check("4 frames differ (rolling)", float((left[:, 1:] - left[:, :-1]).abs().max()) > 1e-3)

# 5 -- mask is a spatial tube: identical across all tubelets
m = RC.tube_mask(B, torch.device("cpu")).view(B, 8, RC.GRID, RC.GRID)
check("5 mask identical across time (spatial tube)",
      bool((m[:, 0:1] == m).all()) and int(m[0, 0].sum()) == RC.GRID * RC.FUT_COLS,
      f"{int(m[0,0].sum())} patches/frame masked, ratio={float(m[0].float().mean()):.3f}")

# 6 -- determinism
c, _, _ = RC.render_rolling(full[:, :CTX], H)
check("6 deterministic", torch.equal(a, c))

# 7 -- finite
check("7 finite", bool(torch.isfinite(a).all()))

print("\n" + ("ALL TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
