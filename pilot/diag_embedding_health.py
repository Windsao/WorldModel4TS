"""Is random-init a FAIR control, or a degenerate one?

ZL-060 measured pretrained/random = 0.59 on cross-channel retrieval -- by far the largest
pretrained advantage in this repository -- while the trivial raw-L2 baseline beat pretrained.
That pattern is exactly what a DEGENERATE random network would produce: if random-init
embeddings barely vary across inputs, their cosine ranking is near-arbitrary and "pretrained
beats random" says nothing about transferable structure.

Diagnostics per arm, on the same inputs:
  spread        std across samples / mean L2 norm  -- collapse indicator
  eff_rank      exp(entropy of the singular-value spectrum) -- how many directions are used
  knn_agree     fraction of top-8 neighbours shared with the raw-L2 ranking
  dist_corr     Spearman correlation between the arm's distances and raw-L2 distances
A healthy control has spread and eff_rank of the same order as pretrained.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "third_party", "VisionTS"))
from run_zl060_cross_channel import embed
from run_visionts_reference import build_manifest, HORIZON

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def eff_rank(X):
    s = np.linalg.svd(X - X.mean(0, keepdims=True), compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="electricity")
ap.add_argument("--data-dir", default="pilot/data")
ap.add_argument("--max-ch", type=int, default=112)
a = ap.parse_args()
data, borders, P, L, H, _, _, _, _ = build_manifest(a.dataset, a.data_dir, a.max_ch, 8, 2000, 0)
series = data.T
M = min(a.max_ch, series.shape[0])
t = borders[1] + L + 200
win = np.stack([series[c, t - L:t] for c in range(M)])

from transformers import VideoMAEModel, VideoMAEConfig
mp = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(DEV).eval()
torch.manual_seed(0)
mr = VideoMAEModel(VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base")).to(DEV).eval()

zt = torch.from_numpy(win).float().to(DEV)
zm, zs = zt.mean(1, keepdim=True), zt.std(1, keepdim=True) + 1e-6
raw = F.normalize((zt - zm) / zs, dim=-1)
Draw = (1 - raw @ raw.T).cpu().numpy()
kraw = (raw @ raw.T).topk(9, dim=1).indices[:, 1:].cpu().numpy()

print(f"{a.dataset}: M={M} windows, L={L}\n")
print(f"{'arm':10s} {'spread':>9s} {'eff_rank':>9s} {'knn_agree':>10s} {'dist_corr':>10s} {'meannorm':>10s}")
out = {}
for nm, model in (("pretrained", mp), ("random", mr)):
    with torch.no_grad():
        E, _, _ = embed(model, win, P)
    X = E.reshape(M, -1)
    Xn = X.cpu().numpy().astype(np.float64)
    nrm = np.linalg.norm(Xn, axis=1)
    spread = float(Xn.std(0).mean() / (nrm.mean() / np.sqrt(Xn.shape[1])))
    er = eff_rank(Xn)
    f = F.normalize(X, dim=-1)
    kk = (f @ f.T).topk(9, dim=1).indices[:, 1:].cpu().numpy()
    agree = float(np.mean([len(set(kk[i]) & set(kraw[i])) / 8 for i in range(M)]))
    D = (1 - f @ f.T).cpu().numpy()
    iu = np.triu_indices(M, 1)
    dc = spearman(D[iu], Draw[iu])
    out[nm] = {"spread": spread, "eff_rank": er, "knn_agree_with_rawL2": agree,
               "dist_spearman_vs_rawL2": dc, "mean_norm": float(nrm.mean())}
    print(f"{nm:10s} {spread:9.4f} {er:9.2f} {agree:10.4f} {dc:10.4f} {nrm.mean():10.2f}")
r = out["pretrained"]["eff_rank"] / max(out["random"]["eff_rank"], 1e-9)
print(f"\n  eff_rank(pretrained)/eff_rank(random) = {r:.3f}")
print(f"  spread ratio                          = "
      f"{out['pretrained']['spread']/max(out['random']['spread'],1e-12):.3f}")
print("\n  VERDICT:", "random-init is DEGENERATE -- the pt/rand comparison is not meaningful"
      if (out["random"]["eff_rank"] < 0.25 * out["pretrained"]["eff_rank"]
          or out["random"]["spread"] < 0.25 * out["pretrained"]["spread"])
      else "random-init is a fair, well-conditioned control")
json.dump(out, open(f"embedding_health_{a.dataset}.json", "w"), indent=2)
