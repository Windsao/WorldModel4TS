"""
Field-video design for VideoMAE on time series.

Core idea (differs from all prior arms): put the PREDICTION on the frame axis
and use the WITHIN-frame 2D image for structure the per-frame encoder can read.
Forecast comes from a regression head on pooled spatiotemporal tokens -- NOT
from pixel reconstruction (A8 showed the pixel-decode path is the bottleneck).

Two rendering modes:

  field   : MULTIVARIATE-JOINT. Each frame f = the field at period f, an image
            with rows = variables (ordered by train-split correlation), cols =
            phase within the period, pixel = value. 16 context periods -> 16
            frames; consecutive frames show the field evolving period-to-period
            = genuine motion. One forward predicts ALL channels jointly. This is
            the regime a video model should win (electricity/traffic = fields).

  uni     : channel-independent control (one variable). Each frame = a single
            variable's period rendered as rows=phase (VisionTS-like) tiled to a
            square; frame = period index. Lets us compare joint-field vs
            independent on the same backbone.

No future frames are ever rendered (head predicts), so there is no mask geometry
and no causal-leakage surface. Context-only normalization. Nearest-neighbor
pixel expansion (patch-aligned, no cross-cell bilinear mixing).

Env: wm4ts (transformers<5). VMAE_CKPT selects the backbone checkpoint.
"""

import argparse
import json
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- dataset config
# (inlined so this file is standalone — the positive-result design)
DATASETS = {
    "ETTh1":       dict(kind="ett",    file="ETTh1.csv",       P=24,  borders=(8640, 11520, 14400)),
    "ETTh2":       dict(kind="ett",    file="ETTh2.csv",       P=24,  borders=(8640, 11520, 14400)),
    "ETTm1":       dict(kind="ett",    file="ETTm1.csv",       P=96,  borders=(34560, 46080, 57600)),
    "ETTm2":       dict(kind="ett",    file="ETTm2.csv",       P=96,  borders=(34560, 46080, 57600)),
    "electricity": dict(kind="lstnet", file="electricity.txt", P=24),
    "traffic":     dict(kind="lstnet", file="traffic.txt",     P=24),
    "solar":       dict(kind="lstnet", file="solar_AL.txt",    P=144),
}
P = None


def configure(period):
    global P
    P = period


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG, PS, NF = 224, 16, 16
GH = IMG // PS            # 14
IMN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- data
def load_mv(name, data_dir, max_ch):
    cfg = DATASETS[name]
    configure(cfg["P"])
    path = os.path.join(data_dir, cfg["file"])
    if cfg["kind"] == "ett":
        data = __import__("pandas").read_csv(path).iloc[:, 1:].values.astype(np.float32)
        b_train, b_val, b_test = cfg["borders"]
    else:
        data = np.loadtxt(path, delimiter=",").astype(np.float32)
        T = len(data)
        b_train, b_val, b_test = int(0.7 * T), int(0.8 * T), T
    mean, std = data[:b_train].mean(0), data[:b_train].std(0) + 1e-8
    data = (data - mean) / std
    M = data.shape[1]
    if M > max_ch:
        keep = np.sort(np.random.default_rng(0).choice(M, max_ch, replace=False))
        data = data[:, keep]
    # order variables by train-split correlation (greedy nearest-neighbor chain)
    tr = data[:b_train]
    C = np.corrcoef(tr.T)
    C = np.nan_to_num(C)
    order, used = [0], {0}
    for _ in range(data.shape[1] - 1):
        last = order[-1]
        cand = [(abs(C[last, j]), j) for j in range(data.shape[1]) if j not in used]
        j = max(cand)[1]
        order.append(j); used.add(j)
    data = data[:, order]
    return data, (b_train, b_val, b_test)


def windows(data, borders, split, context, horizon, stride, cap=None):
    b_train, b_val, b_test = borders
    lo, hi = {"train": (context, b_train - horizon),
              "test": (b_val, b_test - horizon)}[split]
    ts = np.arange(lo, hi + 1, stride)
    if cap and len(ts) > cap:
        ts = np.sort(np.random.default_rng(0).choice(ts, cap, replace=False))
    X = np.stack([data[t - context:t] for t in ts])      # [N, context, M]
    Y = np.stack([data[t:t + horizon] for t in ts])      # [N, horizon, M]
    return X.astype(np.float32), Y.astype(np.float32)


# ---------------------------------------------------------------- model
class FieldVMAE(nn.Module):
    def __init__(self, M, P, horizon, mode="field", pretrained=True,
                 backbone="video", render_mode="period", adapter=False, context=None,
                 fusion="add"):
        super().__init__()
        import transformers
        assert transformers.__version__ < "5"
        self.backbone = backbone
        self.render_mode = render_mode   # "period" (per-frame) or "vts" (single 2D static clip)
        self.adapter = adapter           # frozen-backbone: linear residual + cross-attn readout
        if backbone == "video":
            from transformers import VideoMAEModel, VideoMAEConfig
            name = os.environ.get("VMAE_CKPT", "MCG-NJU/videomae-base")
            self.enc = (VideoMAEModel.from_pretrained(name) if pretrained
                        else VideoMAEModel(VideoMAEConfig.from_pretrained(name)))
        else:  # image control: ViT-MAE encoder applied per frame (same input)
            from transformers import ViTMAEModel, ViTMAEConfig
            name = "facebook/vit-mae-base"
            cfg = ViTMAEConfig.from_pretrained(name)
            cfg.mask_ratio = 0.0
            self.enc = (ViTMAEModel.from_pretrained(name, config=cfg) if pretrained
                        else ViTMAEModel(cfg))
        self.M, self.P, self.horizon, self.mode = M, P, horizon, mode
        d = self.enc.config.hidden_size
        out_dim = (M * horizon) if mode == "field" else horizon
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d),
                                  nn.GELU(), nn.Linear(d, out_dim))
        if adapter:
            # residual decomposition (frozen backbone): a per-channel NLinear pathway on the
            # normalized context carries level/trend/seasonal; the frozen VideoMAE tokens add
            # the residual structure. uni: cross-attn readout. field (A2): the video sees the
            # MULTIVARIATE field video (rows=vars, cols=phase, frames=time = genuine cross-channel
            # motion) and adds a per-channel correction NLinear (channel-independent) cannot.
            ctx = context if context else NF * P
            self.lin = nn.Linear(ctx, horizon)                       # per-channel NLinear (uni & field)
            self.fusion = fusion
            if mode == "uni":
                self.q = nn.Parameter(torch.randn(8, d) * 0.02)      # learned readout queries
                self.xattn = nn.MultiheadAttention(d, 8, batch_first=True)
                self.xnorm = nn.LayerNorm(d)
                self.xhead = nn.Linear(8 * d, horizon)
                nn.init.zeros_(self.xhead.weight); nn.init.zeros_(self.xhead.bias)  # warm start = pure linear
                if fusion == "film":
                    self.film_g = nn.Linear(8 * d, horizon); nn.init.zeros_(self.film_g.weight); nn.init.zeros_(self.film_g.bias)
                    self.film_b = nn.Linear(8 * d, horizon); nn.init.zeros_(self.film_b.weight); nn.init.zeros_(self.film_b.bias)
            else:  # field adapter (A2): pooled field tokens -> per-channel correction
                self.vhead_field = nn.Linear(d, M * horizon)
                nn.init.zeros_(self.vhead_field.weight); nn.init.zeros_(self.vhead_field.bias)  # warm start = pure NLinear
        self.register_buffer("imn_mean", IMN_MEAN)
        self.register_buffer("imn_std", IMN_STD)

    def render(self, x):
        """x [B, NF*P, M] -> vid [B, NF, 3, 224, 224], (mu, sd) per (B,M).

        field: frame f = field at period f; rows=variables, cols=phase.
        uni:   x is [B, NF*P, 1]; frame f = period f, rows=phase tiled.
        """
        B, L, M = x.shape
        mu = x.mean(1, keepdim=True)                       # [B,1,M] context stats
        sd = x.std(1, keepdim=True) + 1e-8
        g = (((x - mu) / (3 * sd)).clamp(-1, 1) + 1) / 2   # [B, L, M] in [0,1]
        G = L // self.P
        if self.render_mode == "vts":
            # VisionTS-style: whole lookback -> ONE 2D image [n_periods x phase],
            # replicated to a static 16-frame clip. uni task only (M==1). Both spatial
            # axes carry signal (non-degenerate), unlike the period-per-frame barcode.
            assert self.mode == "uni" and M == 1, "vts render is uni-only"
            g2 = g[:, :G * self.P].view(B, G, self.P)              # [B, n_periods, phase]
            img1 = F.interpolate(g2.unsqueeze(1), size=(IMG, IMG), mode="nearest")
            vid = img1.unsqueeze(1).expand(B, NF, 1, IMG, IMG)     # static clip x16
            vid = vid.expand(B, NF, 3, IMG, IMG)
            vid = (vid - self.imn_mean.unsqueeze(1)) / self.imn_std.unsqueeze(1)
            return vid.contiguous(), mu, sd
        g = g[:, :G * self.P].view(B, G, self.P, M)        # [B, G, P, M]
        if G != NF:
            # temporal resize: resample the G period-frames to NF=16 (VisionTS-style,
            # applied on the frame axis) so context is decoupled from the 16-frame req.
            g = g.permute(0, 2, 3, 1).reshape(B, self.P * M, G)   # [B, P*M, G]
            g = F.interpolate(g, size=NF, mode="linear", align_corners=False)
            g = g.reshape(B, self.P, M, NF).permute(0, 3, 1, 2)   # [B, NF, P, M]
        if self.mode == "field":
            # frame image: rows = M variables, cols = P phase
            fr = g.permute(0, 1, 3, 2)                     # [B, NF, M, P]
            rr, cc = M, self.P
        else:
            # uni: rows = P phase (single variable), tile to square
            fr = g.permute(0, 1, 3, 2)                     # [B, NF, 1, P]
            fr = fr.expand(B, NF, self.P, self.P)          # [B,NF,P,P] broadcast rows
            rr, cc = self.P, self.P
        img = F.interpolate(fr.reshape(B * NF, 1, rr, cc), size=(IMG, IMG),
                            mode="nearest")                # patch-aligned, no mix
        vid = img.view(B, NF, 1, IMG, IMG).expand(B, NF, 3, IMG, IMG)
        vid = (vid - self.imn_mean.unsqueeze(1)) / self.imn_std.unsqueeze(1)
        return vid.contiguous(), mu, sd

    def _forward_adapter_field(self, x):
        """A2: per-channel NLinear + frozen VideoMAE on the multivariate field video."""
        B, L, M = x.shape
        mu = x.mean(1, keepdim=True); sd = x.std(1, keepdim=True) + 1e-8
        z = (x - mu) / sd                                          # [B, L, M]
        lin_out = self.lin(z.permute(0, 2, 1)).permute(0, 2, 1)   # per-channel NLinear -> [B, horizon, M]
        if getattr(self, "no_vbranch", False):
            return lin_out * sd + mu
        vid, _, _ = self.render(x)                                # field render (evolving multivariate field)
        h = self.enc(pixel_values=vid).last_hidden_state.mean(1)  # [B, d] frozen
        vid_out = self.vhead_field(h).view(B, self.horizon, M)    # zero-init: per-channel correction
        return (lin_out + vid_out) * sd + mu

    def _forward_adapter(self, x):
        """frozen backbone: forecast = NLinear(norm context) + xattn-readout(frozen tokens).
        no_vbranch=True ablates the video branch -> pure NLinear+RevIN (isolates its share)."""
        if self.mode == "field":
            return self._forward_adapter_field(x)
        B, L, _ = x.shape
        mu = x.mean(1, keepdim=True); sd = x.std(1, keepdim=True) + 1e-8
        z = (x - mu) / sd                                       # [B, L, 1] normalized
        lin_out = self.lin(z[..., 0])                          # [B, horizon] NLinear
        if getattr(self, "no_vbranch", False):
            return lin_out.unsqueeze(-1) * sd + mu             # NLinear-only ablation
        vid, _, _ = self.render(x)                             # frozen-backbone input
        tokens = self.enc(pixel_values=vid).last_hidden_state  # [B, N, d] (frozen)
        q = self.q.unsqueeze(0).expand(B, -1, -1)              # [B, 8, d]
        r, _ = self.xattn(q, tokens, tokens)                   # [B, 8, d]
        r = self.xnorm(r).reshape(B, -1)                       # [B, 8*d]
        if getattr(self, "fusion", "add") == "film":
            gamma = 1 + self.film_g(r)                         # [B, horizon] video scales NLinear
            beta = self.film_b(r)                              # [B, horizon] video shifts NLinear
            z_hat = gamma * lin_out + beta                     # video in the main path
        else:
            vmae_out = self.xhead(r)                            # [B, horizon] normalized residual
            z_hat = lin_out + vmae_out
        return z_hat.unsqueeze(-1) * sd + mu                    # [B, horizon, 1]

    def forward(self, x):
        if self.adapter:
            return self._forward_adapter(x)
        B = x.shape[0]
        vid, mu, sd = self.render(x)                       # [B, NF, 3, H, W]
        if self.backbone == "video":
            h = self.enc(pixel_values=vid).last_hidden_state.mean(1)   # [B, d]
        else:  # image: encode each frame, mean-pool patch tokens, then over frames
            f = self.enc(pixel_values=vid.reshape(B * NF, 3, IMG, IMG))
            h = f.last_hidden_state.mean(1).view(B, NF, -1).mean(1)    # [B, d]
        out = self.head(h)
        if self.mode == "field":
            z = out.view(B, self.horizon, self.M)
            return z * sd + mu
        else:
            z = out.view(B, self.horizon, 1)
            return z * sd + mu


# ---------------------------------------------------------------- train / eval
class LoRAWeight(nn.Module):
    """LoRA as a *weight parametrization*: W_eff = W0 + (alpha/r) * B @ A.

    Registered via torch.nn.utils.parametrize so it applies whether the caller does
    `self.query(x)` or `F.linear(x, self.query.weight)` -- VideoMAESelfAttention does
    the latter (it splices in q_bias/v_bias by hand), so wrapping the nn.Linear module
    would be silently bypassed. B is zero-init => W_eff == W0 at step 0.
    """

    def __init__(self, w, r, alpha):
        super().__init__()
        out_f, in_f = w.shape
        self.A = nn.Parameter(torch.empty(r, in_f, device=w.device, dtype=w.dtype))
        self.B = nn.Parameter(torch.zeros(out_f, r, device=w.device, dtype=w.dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, W):
        return W + (self.B @ self.A) * self.scale


def inject_lora(enc, r, alpha, targets=("query", "value")):
    """Attach LoRA to every attention query/value projection in the frozen encoder."""
    from torch.nn.utils import parametrize
    n = 0
    for m in list(enc.modules()):
        for name in targets:
            sub = getattr(m, name, None)
            if isinstance(sub, nn.Linear):
                parametrize.register_parametrization(sub, "weight", LoRAWeight(sub.weight, r, alpha))
                n += 1
    return n


def apply_tune(model, tune, lora_r=8, lora_alpha=16):
    """full: everything trainable (default). frozen: head only.
    ln: head + encoder LayerNorm affine params only.
    lora: head + rank-r LoRA on every attention query/value projection.
    adapter: head unused; NLinear + cross-attn readout on the frozen backbone."""
    if tune == "full":
        return
    for p in model.enc.parameters():
        p.requires_grad_(False)
    if tune == "ln":
        for m in model.enc.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad_(True)
    elif tune == "lora":
        n = inject_lora(model.enc, lora_r, lora_alpha)
        print(f"[info] lora r={lora_r} alpha={lora_alpha} injected into {n} projections", flush=True)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    print(f"[info] tune={tune} trainable={n}/{tot} ({100*n/tot:.2f}%)", flush=True)


def run(model, Xtr, Ytr, Xte, Yte, epochs, lr, batch, mode, seed=0):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    n_steps = epochs * ((len(Xtr) + batch - 1) // batch)
    si = 0
    model.train()
    for ep in range(epochs):
        perm = np.random.default_rng(1000 * seed + ep).permutation(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), batch):
            for pg in opt.param_groups:
                pg["lr"] = lr * 0.5 * (1 + math.cos(math.pi * si / n_steps))
            si += 1
            idx = perm[i:i + batch]
            xb = torch.from_numpy(Xtr[idx]).to(DEVICE)
            yb = torch.from_numpy(Ytr[idx]).to(DEVICE)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            tot += loss.item() * len(idx)
            if (i // batch) % 200 == 0:
                print(f"[info] ep{ep} step {i//batch} loss {loss.item():.4f}",
                      flush=True)
        print(f"[info] epoch {ep} train MSE {tot/len(Xtr):.4f}", flush=True)
    model.eval()
    preds = []
    eb = max(batch, 256)   # eval throughput: uni mode iterates millions of series
    with torch.no_grad():
        for i in range(0, len(Xte), eb):
            xb = torch.from_numpy(Xte[i:i + eb]).to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--horizon-p", type=int, default=4)
    ap.add_argument("--horizon-steps", type=int, default=0, help="if >0, horizon in raw steps (overrides horizon-p); head can output any length")
    ap.add_argument("--context-steps", type=int, default=0, help="if >0, lookback in raw steps (must be a multiple of P); the G=context/P period-frames are temporally resampled to 16 frames (VisionTS-style resize). default 0 => 16*P (one period per frame)")
    ap.add_argument("--data-dir", default="pilot/data")
    ap.add_argument("--out-dir", default="pilot/results_field")
    ap.add_argument("--mode", choices=["field", "uni"], default="field")
    ap.add_argument("--max-ch", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ft-cap", type=int, default=40000)
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--backbone", choices=["video", "image"], default="video")
    ap.add_argument("--lora-r", type=int, default=8, help="LoRA rank (--tune lora)")
    ap.add_argument("--lora-alpha", type=int, default=16, help="LoRA scaling alpha (--tune lora)")
    ap.add_argument("--tune", choices=["full", "frozen", "ln", "lora", "adapter"], default="full",
                    help="full: fine-tune all; frozen: head only; ln: head + encoder LayerNorm affines only")
    ap.add_argument("--no-vbranch", action="store_true", help="adapter ablation: disable the video branch (pure NLinear+RevIN) to isolate the frozen-backbone contribution")
    ap.add_argument("--fusion", choices=["add", "film"], default="add", help="adapter fusion: add=video residual (default); film=video modulates NLinear (gamma*lin+beta), putting video in the main path")
    ap.add_argument("--render", choices=["period", "vts"], default="period",
                    help="period: period-per-frame (barcode for uni); vts: VisionTS-style single 2D static clip (uni only)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    data, borders = load_mv(args.dataset, args.data_dir, args.max_ch)
    if args.context_steps > 0:
        assert args.context_steps % P == 0, f"--context-steps must be a multiple of P={P}"
        context = args.context_steps
    else:
        context = NF * P
    G = context // P                                   # periods in context (resampled to NF=16)
    horizon = args.horizon_steps if args.horizon_steps > 0 else args.horizon_p * P
    M = data.shape[1]
    Xtr, Ytr = windows(data, borders, "train", context, horizon, 1, args.ft_cap)
    Xte, Yte = windows(data, borders, "test", context, horizon, args.stride)
    print(f"dataset={args.dataset} mode={args.mode} M={M} P={P} context={context} "
          f"horizon={horizon} train={len(Xtr)} test={len(Xte)}", flush=True)

    results = {}
    reps = (horizon + P - 1) // P    # periods needed to cover horizon (then trim)
    # baselines (on same windows); work for arbitrary horizon in steps
    def sm(X):  # seasonal mean over context periods
        base = X.reshape(len(X), G, P, M).mean(1)          # [N, P, M]
        return np.tile(base, (1, reps, 1))[:, :horizon]
    for nm, fn in [("snaive", lambda X: np.tile(X[:, -P:], (1, reps, 1))[:, :horizon]),
                   ("smean", sm)]:
        p = fn(Xte)
        results[nm] = {"MSE": round(float(np.mean((p - Yte) ** 2)), 4),
                       "MAE": round(float(np.mean(np.abs(p - Yte))), 4)}
        print(f"[done] {nm:10s} {results[nm]}", flush=True)

    if args.mode == "uni":
        # flatten to per-channel univariate samples
        Xtr = Xtr.transpose(0, 2, 1).reshape(-1, context, 1)
        Ytr = Ytr.transpose(0, 2, 1).reshape(-1, horizon, 1)
        Xte_u = Xte.transpose(0, 2, 1).reshape(-1, context, 1)
        Yte_u = Yte.transpose(0, 2, 1).reshape(-1, horizon, 1)
        if len(Xtr) > args.ft_cap:
            k = np.random.default_rng(0).choice(len(Xtr), args.ft_cap, False)
            Xtr, Ytr = Xtr[k], Ytr[k]
        model = FieldVMAE(1, P, horizon, "uni", bool(args.pretrained), args.backbone, args.render,
                          adapter=(args.tune == "adapter"), context=context,
                          fusion=args.fusion).to(DEVICE)
        model.no_vbranch = args.no_vbranch
        apply_tune(model, args.tune, args.lora_r, args.lora_alpha)
        pred = run(model, Xtr, Ytr, Xte_u, Yte_u, args.epochs, args.lr,
                   args.batch, "uni", args.seed)
        mse = float(np.mean((pred - Yte_u) ** 2)); mae = float(np.mean(np.abs(pred - Yte_u)))
    else:
        model = FieldVMAE(M, P, horizon, "field", bool(args.pretrained), args.backbone, args.render,
                          adapter=(args.tune == "adapter"), context=context, fusion=args.fusion).to(DEVICE)
        model.no_vbranch = args.no_vbranch
        apply_tune(model, args.tune, args.lora_r, args.lora_alpha)
        pred = run(model, Xtr, Ytr, Xte, Yte, args.epochs, args.lr,
                   args.batch, "field", args.seed)
        mse = float(np.mean((pred - Yte) ** 2)); mae = float(np.mean(np.abs(pred - Yte)))
    tag = f"{args.backbone}_{args.mode}" + ("" if args.pretrained else "_rand") + \
          ("_nlin" if args.no_vbranch else "") + \
          ("_film" if args.fusion=="film" else "") + \
          ("" if args.tune == "full" else f"_{args.tune}") + \
          (f"r{args.lora_r}" if args.tune == "lora" else "") + \
          ("" if args.render == "period" else f"_{args.render}") + f"_s{args.seed}"
    results[tag] = {"MSE": round(mse, 4), "MAE": round(mae, 4)}
    print(f"[done] {tag:14s} MSE={mse:.4f} MAE={mae:.4f}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir,
                           f"field_{args.dataset}_{args.mode}_{args.backbone}"
                           + ("" if args.pretrained else "_rand")
                           + ("" if args.tune == "full" else f"_{args.tune}")
                           + (f"r{args.lora_r}" if args.tune == "lora" else "")
                           + ("_nlin" if args.no_vbranch else "")
                           + ("_film" if args.fusion=="film" else "")
                           + ("" if args.render == "period" else f"_{args.render}")
                           + f"_L{context}_h{horizon}_s{args.seed}.json"), "w") as f:
        json.dump({"config": vars(args) | {"M": M, "P": P, "context": context,
                                           "horizon": horizon}, "results": results},
                  f, indent=2)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
