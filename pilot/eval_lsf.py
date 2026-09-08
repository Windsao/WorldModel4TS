"""Standard long-term-forecasting (LSF) zero-shot protocol, as in VisionTS / VisionTS++ / TSFM papers
(Time-Series-Library convention):

  * ETTh*: train 12 months / val 4 / test 4 (hourly); ETTm*: same in 15-min steps;
    custom (electricity, weather, traffic): 70% / 10% / 20%.
  * StandardScaler fitted on the TRAIN split; MSE / MAE computed on standardised values.
  * Every channel, every test origin (stride 1): origins t = test_start .. N - H.
  * Horizons 96 / 192 / 336 / 720. Context length is the method's own choice (VisionTS tunes it
    on validation: ETTh1 2880, ETTh2 1728, ETTm1 2304, ETTm2 4032, electricity 2880, weather 4032).

Methods:
  --model visionts                 the official VisionTS checkpoint (protocol check against the paper)
  --model <route-B checkpoint dir> our continually-pretrained VideoMAE forecaster; the model sees its
                                   last 12 (or 14) periods and predicts 4 (or 2) frames = periods per
                                   pass; longer horizons by autoregressive ROLLOUT (predictions are
                                   appended to the context and the window slides by hp periods).
  --blend                          also report the ZL-051 prior blend on the full context, and
                                   model+blend (blend with the model as fifth candidate).
Outputs <out>/<dataset>_H<H>_<tag>.json with mse / mae / n_origins / n_channels / timing.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "third_party", "VisionTS"))
import pretrain_route_b as B

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VTS_CONTEXT = {"ETTh1": 2880, "ETTh2": 1728, "ETTm1": 2304, "ETTm2": 4032, "electricity": 2880,
               "weather": 4032, "traffic": 2880}
PERIOD = {"ETTh1": 24, "ETTh2": 24, "ETTm1": 96, "ETTm2": 96, "electricity": 24, "weather": 144, "traffic": 24}
FILES = {"ETTh1": "ETTh1.csv", "ETTh2": "ETTh2.csv", "ETTm1": "ETTm1.csv", "ETTm2": "ETTm2.csv",
         "electricity": "electricity.txt", "weather": "weather.csv", "traffic": "traffic.txt"}


# ------------------------------------------------------------------ data (TSL convention)
def load_lsf(dataset, data_dir):
    """-> (data float32 [N, C] standardised with TRAIN stats, test_start index, raw shape)."""
    import pandas as pd
    path = os.path.join(data_dir, FILES[dataset])
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        cols = [c for c in df.columns if c != "date"]
        raw = df[cols].values.astype(np.float64)
    else:
        raw = np.loadtxt(path, delimiter=",").astype(np.float64)
    N = len(raw)
    if dataset in ("ETTh1", "ETTh2"):
        n_train, n_val_end = 12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24
        test_end = 12 * 30 * 24 + 8 * 30 * 24
    elif dataset in ("ETTm1", "ETTm2"):
        n_train, n_val_end = 12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4
        test_end = 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4
    else:
        n_train = int(N * 0.7); n_test = int(N * 0.2)
        n_val_end = N - n_test; test_end = N
    mean = raw[:n_train].mean(0); std = raw[:n_train].std(0)          # sklearn StandardScaler (ddof=0)
    std = np.where(std == 0, 1.0, std)
    data = ((raw - mean) / std).astype(np.float32)
    return data[:test_end], n_val_end, raw.shape


# ------------------------------------------------------------------ forecasters
class RouteBForecaster:
    """Route-B VideoMAE checkpoint, torch-side rendering, autoregressive rollout."""

    def __init__(self, ckpt, device=DEV):
        from transformers import VideoMAEForPreTraining
        self.model = VideoMAEForPreTraining.from_pretrained(ckpt).to(device).eval()
        rc = os.path.join(os.path.dirname(ckpt.rstrip("/")), "run_config.json")
        self.enc_attn = json.load(open(rc)).get("enc_attn", "full") if os.path.exists(rc) else "full"
        if self.enc_attn == "spatial":
            B.apply_spatial_attention(self.model)
        self.device = device
        self.c2p = {}

    def _col2phase(self, P):
        if P not in self.c2p:
            self.c2p[P] = torch.as_tensor(B.col2phase(P), device=self.device)
        return self.c2p[P]

    @torch.no_grad()
    def _one_pass(self, ctx, P, hp, lead):
        """ctx [n, G_use*P] torch (float32, device) -> z-forecast [n, hp*P] in ctx-normalised units,
        plus (mu, sd) [n,1]."""
        n = ctx.shape[0]
        G_use = ctx.shape[1] // P
        mu = ctx.mean(1, keepdim=True); sd = ctx.std(1, unbiased=False, keepdim=True) + 1e-6
        z = ((ctx - mu) / sd).clamp(-B.ZMAX, B.ZMAX).view(n, G_use, P)
        h = 0.5 + B.BAND * z / B.ZMAX
        rows_ctx = torch.round((1.0 - h) * (B.IMG - 1)).to(torch.int16)[:, :, self._col2phase(P)]   # [n,G_use,224]
        rows = torch.full((n, B.NF, B.IMG), int(round(0.5 * (B.IMG - 1))), dtype=torch.int16, device=self.device)
        rows[:, lead * B.TS: lead * B.TS + G_use] = rows_ctx
        mask = B.forecast_mask(hp, lead)
        m = mask[None].expand(n, -1).to(self.device)
        vid = B.rows_to_video(rows, self.device)
        B.set_attn_bias(self.model, mask)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            logits = self.model(pixel_values=vid, bool_masked_pos=m).logits
        nf = B.n_future_tokens(hp)
        logits = logits.float()[:, -nf:]
        c = logits.view(n, hp // B.TS, B.GH, B.GH, B.TS, B.PS, B.PS, 3)
        frames = c.permute(0, 1, 4, 7, 2, 5, 3, 6).reshape(n, hp, 3, B.IMG, B.IMG)
        zf = B.decode_frames_gray(frames.mean(2), P).reshape(n, hp * P)
        zf = torch.nan_to_num(zf, nan=0.0, posinf=B.ZMAX, neginf=-B.ZMAX)
        return zf * sd + mu

    @staticmethod
    def plan(P, H, L, mode):
        """-> (P_eff, hp, k, G_use, lead). multiperiod: a frame holds k true periods so that ONE pass
        (hp frames) covers H and the context spans 12 or 14 x k periods; rollout: k = 1."""
        if mode.startswith("rollout"):
            passes = int(mode[len("rollout"):] or 1)               # rollout, rollout2, rollout4 ...
            hp = 2 if H <= P else 4
            k = max(1, int(math.ceil(H / (hp * P * passes))))
        else:
            if H <= P:            hp, k = 2, 1
            elif H <= 2 * P:      hp, k = 2, 1
            elif H <= 4 * P:      hp, k = 4, 1
            else:                 hp, k = 4, int(math.ceil(H / (4 * P)))
        Pe = k * P
        G_max = B.NF - hp
        G_use = min(G_max, L // Pe)
        if (G_max - G_use) % 2:
            G_use -= 1
        lead = (G_max - G_use) // 2
        assert G_use >= 8, (P, H, L, mode, G_use)
        return Pe, hp, k, G_use, lead

    @torch.no_grad()
    def forecast(self, ctx_np, P, H, batch=192, mode="multiperiod", scales=(1,)):
        """ctx_np [n, L] -> [n, H]. scales: multipliers of the planned k (multi-scale ensemble:
        each scale renders k*s true periods per frame, so the context spans 12*k*s periods; the
        forecasts are averaged)."""
        if len(scales) > 1:
            ys = [self.forecast(ctx_np, P, H, batch, mode, scales=(s_,)) for s_ in scales]
            return np.mean(ys, 0)
        n, L = ctx_np.shape
        Pe, hp, k, G_use, lead = self.plan(P, H, L, mode)
        if scales[0] != 1:
            k = k * scales[0]; Pe = k * P
            G_max = B.NF - hp
            G_use = min(G_max, L // Pe)
            if (G_max - G_use) % 2: G_use -= 1
            lead = (G_max - G_use) // 2
            if G_use < 8:
                return self.forecast(ctx_np, P, H, batch, mode, scales=(1,))    # scale does not fit
        out = np.empty((n, H), np.float32)
        for s in range(0, n, batch):
            ctx = torch.from_numpy(ctx_np[s:s + batch]).to(self.device)
            if mode.startswith("rollout"):
                pred, got = [], 0
                while got < H:
                    y = self._one_pass(ctx[:, -G_use * Pe:], Pe, hp, lead)
                    pred.append(y); got += hp * Pe
                    ctx = torch.cat([ctx, y], 1)
                out[s:s + batch] = torch.cat(pred, 1)[:, :H].cpu().numpy()
            else:
                y = self._one_pass(ctx[:, -G_use * Pe:], Pe, hp, lead)     # [b, hp*Pe] >= H
                out[s:s + batch] = y[:, :H].cpu().numpy()
        return out


class VisionTSForecaster:
    def __init__(self, dataset, H, device=DEV):
        from visionts import VisionTS
        ckpt_dir = os.environ.get("VISIONTS_CKPT", "/nyx-storage1/hanliu/wm4ts/ckpt")
        self.vm = VisionTS(arch="mae_base", finetune_type="none", ckpt_dir=ckpt_dir, load_ckpt=True).to(device).eval()
        self.vm.update_config(context_len=VTS_CONTEXT[dataset], pred_len=H, periodicity=PERIOD[dataset],
                             norm_const=0.4, align_const=0.4)
        self.device = device

    @torch.no_grad()
    def forecast(self, ctx_np, P, H, batch=64):
        n = ctx_np.shape[0]
        out = np.empty((n, H), np.float32)
        for s in range(0, n, batch):
            x = torch.from_numpy(ctx_np[s:s + batch]).to(self.device)[:, :, None]
            y = self.vm(x, fp64=False)
            out[s:s + batch] = y[:, :, 0].float().cpu().numpy()
        return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(FILES))
    ap.add_argument("--pred-len", type=int, required=True)
    ap.add_argument("--model", required=True, help="'visionts' or a Route-B checkpoint dir")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--context", type=int, default=None, help="context length fed to the method (default: VisionTS's)")
    ap.add_argument("--data-dir", default="/nyx-storage1/hanliu/wm4ts/data")
    ap.add_argument("--out", default="/nyx-storage1/hanliu/wm4ts/lsf")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--origin-batch", type=int, default=16, help="origins per outer batch (x all channels)")
    ap.add_argument("--blend", action="store_true")
    ap.add_argument("--mode", default="multiperiod", help="multiperiod | rollout | rollout<n> (n passes with k periods/frame)")
    ap.add_argument("--scales", default="1", help="comma list of k multipliers to ensemble, e.g. 1,2,4")
    ap.add_argument("--extra-models", default="", help="comma list of extra members ('visionts' or Route-B ckpt dirs) averaged with --model; each member's own MSE is also recorded")
    ap.add_argument("--limit", type=int, default=None, help="debug: first k origins only")
    ap.add_argument("--origin-range", default=None, help="a:b slice of the origin list (for splitting a dataset over GPUs); partial sums are saved and merged by summarize_lsf.py")
    args = ap.parse_args()
    t0 = time.time()
    data, test_start, shape = load_lsf(args.dataset, args.data_dir)
    N, C = data.shape
    H, P = args.pred_len, PERIOD[args.dataset]
    if args.mode == "auto":
        # recipe fixed from the ETTh1/ETTh2/ETTm2/weather development runs (2026-09-08):
        #   phase resolution matters: keep k*P <= 96 samples per frame;
        #   H <= 4P  -> one pass, ensemble over scales with k*s*P <= 96
        #   H >  4P  -> rollout with k = min(ceil(H/4P), 96//P) periods per frame
        k_max = max(1, 96 // P)
        if H <= 4 * P:
            args.mode = "multiperiod"
            args.scales = ",".join(str(s_) for s_ in (1, 2, 4) if s_ * P <= 96)
        else:
            k_direct = int(math.ceil(H / (4 * P)))
            # hourly data (P=24): 2-pass rollout with 4 periods/frame beat 8 periods/frame direct
            # (ETTh1 H720 0.477 vs 0.521); 15-min data (P=96): direct 2 periods/frame beat the
            # 2-pass rollout (ETTm2 H720 0.371 vs 0.409). Rule: rollout only when P < 96.
            passes = int(math.ceil(k_direct / k_max)) if P < 96 else 1
            args.mode = "multiperiod" if passes == 1 else f"rollout{passes}"
            args.scales = "1"
        print(f"[auto] mode={args.mode} scales={args.scales}", flush=True)
    L = args.context or VTS_CONTEXT[args.dataset]
    origins = np.arange(test_start, N - H + 1, args.stride)
    if args.limit: origins = origins[:args.limit]
    n_all = len(origins)
    if args.origin_range:
        a, b = (int(x) for x in args.origin_range.split(":")); origins = origins[a:b]
    tag = args.tag or ("visionts" if args.model == "visionts" else os.path.basename(os.path.dirname(args.model.rstrip("/"))) + "_" + os.path.basename(args.model.rstrip("/")))
    print(f"[LSF] {args.dataset} H={H} P={P} C={C} origins={len(origins)} context={L} model={tag}", flush=True)
    fc = VisionTSForecaster(args.dataset, H) if args.model == "visionts" else RouteBForecaster(args.model)
    extras = []
    for m in [x for x in args.extra_models.split(",") if x]:
        extras.append((m, VisionTSForecaster(args.dataset, H) if m == "visionts" else RouteBForecaster(m)))
    se_x = {m: 0.0 for m, _ in extras}; se_ens = 0.0; ae_ens = 0.0
    if args.blend:
        import zeroshot_loop as ZL
    se = ae = 0.0; se_bl = ae_bl = 0.0; se_mb = ae_mb = 0.0; cnt = 0
    for s in range(0, len(origins), args.origin_batch):
        ob = origins[s:s + args.origin_batch]
        # [n_o, C, L] contexts and [n_o, C, H] targets, flattened over (origin, channel)
        ctx = np.stack([data[t - L:t].T for t in ob])              # [n_o, C, L]
        tgt = np.stack([data[t:t + H].T for t in ob])              # [n_o, C, H]
        ctx = np.nan_to_num(ctx.reshape(-1, L)); tgt = tgt.reshape(-1, H)
        scales = tuple(int(x) for x in args.scales.split(","))
        y = fc.forecast(ctx, P, H, mode=args.mode, scales=scales) if isinstance(fc, RouteBForecaster) else fc.forecast(ctx, P, H)
        se += float(((y - tgt) ** 2).sum()); ae += float(np.abs(y - tgt).sum()); cnt += y.size
        if extras:
            ys = [y]
            for m, f_ in extras:
                y2 = f_.forecast(ctx, P, H, mode=args.mode, scales=scales) if isinstance(f_, RouteBForecaster) else f_.forecast(ctx, P, H)
                se_x[m] += float(((y2 - tgt) ** 2).sum()); ys.append(y2)
            ye = np.mean(ys, 0)
            se_ens += float(((ye - tgt) ** 2).sum()); ae_ens += float(np.abs(ye - tgt).sum())
        if args.blend:
            pr = {k: v for k, v in ZL.priors(ctx, P, H).items() if k in ("smean", "snaive", "last", "recent_mean")}
            _, base, _, _ = ZL.pseudo_origin_blend(ctx, P, H, pr, rule="pow_n", dense=True, min_ctx_periods=8)
            se_bl += float(((base - tgt) ** 2).sum()); ae_bl += float(np.abs(base - tgt).sum())
            mb = 0.5 * (y + base)                                  # parameter-free ensemble
            se_mb += float(((mb - tgt) ** 2).sum()); ae_mb += float(np.abs(mb - tgt).sum())
        if (s // args.origin_batch) % 20 == 0:
            print(f"  {s + len(ob)}/{len(origins)} origins  mse so far {se / cnt:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    res = {"dataset": args.dataset, "pred_len": H, "period": P, "context": L, "model": tag, "stride": args.stride, "mode": args.mode, "scales": args.scales,
           "plan": list(RouteBForecaster.plan(P, H, L, args.mode)) if args.model != "visionts" else None,
           "n_origins": int(len(origins)), "n_origins_all": int(n_all), "origin_range": args.origin_range,
           "se": se, "ae": ae, "cnt": cnt, "n_channels": int(C), "mse": se / cnt, "mae": ae / cnt,
           "wall_s": round(time.time() - t0, 1), "raw_shape": list(shape), "test_start": int(test_start)}
    if args.blend:
        res.update({"blend_mse": se_bl / cnt, "blend_mae": ae_bl / cnt, "avg_model_blend_mse": se_mb / cnt, "avg_model_blend_mae": ae_mb / cnt})
    if extras:
        res["extra_models"] = {m: se_x[m] / cnt for m, _ in extras}
        res["ensemble_mse"] = se_ens / cnt; res["ensemble_mae"] = ae_ens / cnt
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, f"{args.dataset}_H{H}_{tag}{'_' + args.mode if args.model != 'visionts' else ''}{'_sc' + args.scales.replace(',', '') if args.scales != '1' else ''}{'_ens' + str(len(extras)) if extras else ''}{'_s' + str(args.stride) if args.stride != 1 else ''}{'_r' + args.origin_range.replace(':', '-') if args.origin_range else ''}.json")
    json.dump(res, open(p, "w"), indent=1)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)


if __name__ == "__main__":
    main()
