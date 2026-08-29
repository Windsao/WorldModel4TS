"""A1 — spatiotemporal video: does a frozen video prior earn its keep on data that
is genuinely a MOVING FIELD (traffic sensor network), unlike univariate benchmarks?

METR-LA: 207 speed sensors with lat/lon. Each timestep -> a 2D spatial heat-map frame
(sensor values scattered to a grid by coordinates); 16 consecutive frames = a video in
which congestion PROPAGATES across space = real motion, on-distribution for VideoMAE.

Model (same residual-decomposition idea as the adapter): per-sensor NLinear on the
normalized context (level/trend) + frozen VideoMAE reading the spatial video -> per-sensor
correction. --no-vbranch ablates the video branch (pure NLinear) to isolate its share.
Missing values (0 in METR-LA) are masked in loss and metrics.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG, NF = 224, 16
IMN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def load_metrla(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "METR-LA.csv"), index_col=0)
    data = df.values.astype(np.float32)                       # [T, N] speeds (0 = missing)
    loc = pd.read_csv(os.path.join(data_dir, "sensor_loc.csv"))
    sid2ll = {str(r.sensor_id): (r.latitude, r.longitude) for r in loc.itertuples()}
    coords = np.array([sid2ll[str(c)] for c in df.columns], dtype=np.float32)  # [N,2] lat,lon
    return data, coords


def grid_index(coords, gh):
    """Precompute, for each of gh*gh grid cells, its nearest sensor (Voronoi selection)."""
    xy = (coords - coords.min(0)) / (np.ptp(coords, 0) + 1e-9)    # [N,2] in [0,1]
    gy, gx = np.mgrid[0:1:complex(gh), 0:1:complex(gh)]
    cells = np.stack([gy.ravel(), gx.ravel()], 1)             # [gh*gh, 2]
    d = ((cells[:, None, :] - xy[None, :, :]) ** 2).sum(-1)   # [gh*gh, N]
    return d.argmin(1)                                        # [gh*gh] nearest sensor per cell


def windows(data, mask, lo, hi, ctx, horizon, stride, cap=None):
    ts = np.arange(lo + ctx, hi - horizon, stride)
    if cap and len(ts) > cap:
        ts = np.sort(np.random.default_rng(0).choice(ts, cap, False))
    X = np.stack([data[t - ctx:t] for t in ts])              # [B, ctx, N]
    Y = np.stack([data[t:t + horizon] for t in ts])          # [B, horizon, N]
    My = np.stack([mask[t:t + horizon] for t in ts])         # [B, horizon, N]
    return X.astype(np.float32), Y.astype(np.float32), My.astype(np.float32)


class STVid(nn.Module):
    def __init__(self, N, ctx, horizon, gidx, no_vbranch=False):
        super().__init__()
        import transformers
        assert transformers.__version__ < "5"
        from transformers import VideoMAEModel
        self.enc = VideoMAEModel.from_pretrained(os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base"))
        d = self.enc.config.hidden_size
        self.N, self.ctx, self.horizon, self.no_vbranch = N, ctx, horizon, no_vbranch
        self.register_buffer("gidx", torch.tensor(gidx, dtype=torch.long))
        self.gh = int(len(gidx) ** 0.5)
        self.lin = nn.Linear(ctx, horizon)                    # per-sensor NLinear
        self.vhead = nn.Linear(d, horizon * N)                # video -> per-sensor correction
        nn.init.zeros_(self.vhead.weight); nn.init.zeros_(self.vhead.bias)  # warm start = pure NLinear
        self.register_buffer("imn_mean", IMN_MEAN); self.register_buffer("imn_std", IMN_STD)

    def render(self, z):                                      # z [B, ctx, N] normalized -> video
        B = z.shape[0]
        g = ((z / 6).clamp(-1, 1) + 1) / 2                    # [B,ctx,N] -> [0,1] (speeds already z-scored)
        frames = g[:, :, self.gidx].view(B, self.ctx, self.gh, self.gh)   # [B,ctx,gh,gh] spatial
        img = F.interpolate(frames.reshape(B * self.ctx, 1, self.gh, self.gh),
                            size=(IMG, IMG), mode="nearest")
        vid = img.view(B, self.ctx, 1, IMG, IMG).expand(B, self.ctx, 3, IMG, IMG)
        return ((vid - self.imn_mean) / self.imn_std).contiguous()

    def forward(self, x):                                     # x [B, ctx, N] (already z-scored)
        B = x.shape[0]
        mu = x.mean(1, keepdim=True); sd = x.std(1, keepdim=True) + 1e-8   # RevIN on the window
        z = (x - mu) / sd
        lin_out = self.lin(z.permute(0, 2, 1)).permute(0, 2, 1)           # [B,horizon,N]
        if self.no_vbranch:
            return lin_out * sd + mu
        h = self.enc(pixel_values=self.render(z)).last_hidden_state.mean(1)  # [B,d] frozen
        vid_out = self.vhead(h).view(B, self.horizon, self.N)
        return (lin_out + vid_out) * sd + mu


def masked_metrics(pred, y, m):
    m = m > 0
    e = (pred - y)[m]
    return float((e ** 2).mean()), float(np.abs(e).mean())


def run(model, Xtr, Ytr, Mtr, Xte, Yte, Mte, epochs, lr, batch):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    for ep in range(epochs):
        model.train(); perm = np.random.default_rng(ep).permutation(len(Xtr)); tot = 0
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            xb = torch.from_numpy(Xtr[idx]).to(DEVICE)
            yb = torch.from_numpy(Ytr[idx]).to(DEVICE); mb = torch.from_numpy(Mtr[idx]).to(DEVICE)
            pred = model(xb)
            loss = (((pred - yb) ** 2) * mb).sum() / mb.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); tot += loss.item() * len(idx)
            if (i // batch) % 200 == 0:
                print(f"[info] ep{ep} step {i//batch} loss {loss.item():.4f}", flush=True)
        print(f"[info] epoch {ep} train {tot/len(Xtr):.4f}", flush=True)
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), max(batch, 128)):
            preds.append(model(torch.from_numpy(Xte[i:i + max(batch, 128)]).to(DEVICE)).cpu().numpy())
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/nyx-storage1/hanliu/wm4ts/data/stgcn")
    ap.add_argument("--out-dir", default="pilot/results_field/a1")
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--ctx", type=int, default=16)
    ap.add_argument("--gh", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--ft-cap", type=int, default=40000)
    ap.add_argument("--no-vbranch", action="store_true")
    args = ap.parse_args()

    data, coords = load_metrla(args.data_dir)                 # [T,N], [N,2]
    T, N = data.shape
    mask = (data != 0).astype(np.float32)
    b_tr, b_val = int(0.7 * T), int(0.8 * T)
    mean = (data[:b_tr] * mask[:b_tr]).sum(0) / mask[:b_tr].sum(0).clip(1)
    std = np.sqrt((((data[:b_tr] - mean) ** 2) * mask[:b_tr]).sum(0) / mask[:b_tr].sum(0).clip(1)) + 1e-3
    dz = (data - mean) / std                                  # z-scored per sensor
    gidx = grid_index(coords, args.gh)
    Xtr, Ytr, Mtr = windows(dz, mask, 0, b_tr, args.ctx, args.horizon, 1, args.ft_cap)
    Xte, Yte, Mte = windows(dz, mask, b_val, T, args.ctx, args.horizon, args.stride)
    # de-standardize targets back to speed space for metrics
    Ytr_s = Ytr * std + mean; Yte_s = Yte * std + mean
    print(f"METR-LA N={N} ctx={args.ctx} horizon={args.horizon} gh={args.gh} "
          f"train={len(Xtr)} test={len(Xte)} vbranch={not args.no_vbranch}", flush=True)

    # baselines in speed space
    def denorm(z): return z * std + mean
    res = {}
    # seasonal-naive: repeat last ctx value
    snaive = np.repeat(denorm(Xte[:, -1:]), args.horizon, 1)
    res["snaive"] = dict(zip(("MSE", "MAE"), masked_metrics(snaive, Yte_s, Mte)))
    print(f"[done] snaive {res['snaive']}", flush=True)

    model = STVid(N, args.ctx, args.horizon, gidx, args.no_vbranch).to(DEVICE)
    if not args.no_vbranch:
        for p in model.enc.parameters(): p.requires_grad_(False)   # FROZEN backbone
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[info] trainable {n} frozen_backbone={not args.no_vbranch}", flush=True)
    pred_z = run(model, Xtr, Ytr, Mtr, Xte, Yte, Mte, args.epochs, args.lr, args.batch)
    pred_s = denorm(pred_z)
    mse, mae = masked_metrics(pred_s, Yte_s, Mte)
    tag = "nlin" if args.no_vbranch else "stvid"
    res[tag] = {"MSE": round(mse, 4), "MAE": round(mae, 4)}
    print(f"[done] {tag} MSE={mse:.4f} MAE={mae:.4f}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"metrla_{tag}_h{args.horizon}.json"), "w") as f:
        json.dump({"config": vars(args) | {"N": N}, "results": res}, f, indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
