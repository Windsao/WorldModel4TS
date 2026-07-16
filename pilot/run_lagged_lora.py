"""
Fast VideoMAE transfer experiment: frozen checkpoint vs query/value LoRA.

Representation
--------------
The input is 12 visible frames followed by 4 fully masked future frames.  A
visible frame summarizes a rolling history window as three lag matrices:

  R: consecutive lag-window rows (stride 1 raw step)
  G: quarter-period lag-window rows (stride P/4)
  B: one-period lag-window rows (stride P)

Every lag row is a length-P window.  The newest row is repeated over the bottom
16-pixel patch band so it can be decoded cleanly; all older lag rows are pooled
into the other 13 patch bands.  Consecutive video frames advance by one period.
The four masked frames therefore represent the next four periods, and only the
newest (bottom) row of each reconstructed frame is scored.

Training
--------
The Kinetics VideoMAE checkpoint stays frozen.  The LoRA arm adds low-rank
updates to every query/value projection in the encoder and decoder; these are
the only trainable parameters.  Weight parametrization is used instead of a
Linear wrapper because transformers 4.46 VideoMAE reads query/value `.weight`
directly through functional.linear.

Example (quick ETTh1 run):
  python pilot/run_lagged_lora.py --dataset ETTh1 --data-dir pilot/data
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize

import run_pilot as rp


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 224
VISIBLE_FRAMES = 12
PRED_FRAMES = 4
NUM_FRAMES = VISIBLE_FRAMES + PRED_FRAMES
IMN_MEAN = torch.tensor(rp.IMN_MEAN)
IMN_STD = torch.tensor(rp.IMN_STD)


# ---------------------------------------------------------------------- data
def split_bounds(data, name, split):
    cfg = rp.DATASETS[name]
    if cfg["kind"] == "ett":
        b_train, b_val, b_test = cfg["borders"]
    else:
        n = len(data)
        b_train, b_val, b_test = int(0.7 * n), int(0.8 * n), n
    if split == "train":
        return 0, b_train
    if split == "test":
        return b_val, b_test
    raise ValueError(f"unknown split: {split}")


def get_windows(data, name, context, horizon, split, stride, cap, seed):
    """Build seeded channel-independent windows without materializing all candidates."""
    lo, hi = split_bounds(data, name, split)
    first = context if split == "train" else lo
    times = np.arange(first, hi - horizon + 1, stride, dtype=np.int64)
    channels = data.shape[1]
    total = len(times) * channels
    if total == 0:
        raise ValueError(
            f"no {split} windows: context={context}, horizon={horizon}, "
            f"bounds=({lo}, {hi})"
        )
    if cap and total > cap:
        picks = np.sort(np.random.default_rng(seed).choice(total, cap, replace=False))
    else:
        picks = np.arange(total)
    x = np.empty((len(picks), context), dtype=np.float32)
    y = np.empty((len(picks), horizon), dtype=np.float32)
    for j, pick in enumerate(picks):
        t = times[pick // channels]
        c = pick % channels
        x[j] = data[t - context:t, c]
        y[j] = data[t:t + horizon, c]
    return x, y, total


def resolve_channel_strides(spec, period):
    if spec == "auto":
        return (1, max(1, period // 4), period)
    values = tuple(int(v.strip()) for v in spec.split(","))
    if len(values) != 3 or any(v <= 0 for v in values):
        raise ValueError("--channel-strides must be 'auto' or three positive integers")
    return values


# ---------------------------------------------------------------------- LoRA
class AdditiveLoRA(nn.Module):
    """Parametrize W as W + (alpha/rank) * B @ A, with a zero initial delta."""

    def __init__(self, weight, rank, alpha):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.A = nn.Parameter(weight.new_empty(rank, weight.shape[1]))
        self.B = nn.Parameter(weight.new_zeros(weight.shape[0], rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, base_weight):
        return base_weight + (self.B @ self.A) * self.scaling


def inject_qv_lora(model, rank, alpha):
    """Freeze the checkpoint and add LoRA only to attention query/value weights."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in {"query", "value"}
    ]
    if not targets:
        raise RuntimeError("no VideoMAE query/value Linear modules found for LoRA")
    for _, module in targets:
        parametrize.register_parametrization(
            module, "weight", AdditiveLoRA(module.weight, rank, alpha)
        )
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("parametrizations.weight.0" not in n for n, _ in trainable):
        raise RuntimeError("LoRA invariant failed: a non-LoRA parameter is trainable")
    return [name for name, _ in targets], sum(p.numel() for _, p in trainable)


def adapter_state_dict(model):
    marker = ".parametrizations.weight.0."
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if marker in name
    }


# --------------------------------------------------------------------- model
class LaggedStrideVMAE(nn.Module):
    def __init__(self, period, history_periods, channel_strides, model_name):
        super().__init__()
        import transformers

        major = int(transformers.__version__.split(".", 1)[0])
        assert major < 5, "transformers>=5 breaks this repository's VideoMAE checkpoint"
        from transformers import VideoMAEForPreTraining

        self.m = VideoMAEForPreTraining.from_pretrained(model_name)
        cfg = self.m.config
        self.period = period
        self.history_periods = history_periods
        self.history_steps = history_periods * period
        self.channel_strides = tuple(channel_strides)
        self.context = (history_periods + VISIBLE_FRAMES - 1) * period
        self.horizon = PRED_FRAMES * period
        self.ts = cfg.tubelet_size
        self.ps = cfg.patch_size
        self.nf = cfg.num_frames
        self.gh = IMG // self.ps
        self.tt = self.nf // self.ts
        if self.nf != NUM_FRAMES or self.ts != 2 or self.ps != 16:
            raise ValueError(
                "this layout requires VideoMAE with 16 frames, tubelet 2, patch 16; "
                f"got frames={self.nf}, tubelet={self.ts}, patch={self.ps}"
            )
        max_lag = self.history_steps - period
        if any(stride > max_lag for stride in self.channel_strides):
            raise ValueError(
                f"channel strides {self.channel_strides} exceed available lag {max_lag}"
            )
        visible_tubelets = VISIBLE_FRAMES // self.ts
        mask = torch.zeros(self.tt, self.gh, self.gh, dtype=torch.bool)
        mask[visible_tubelets:] = True
        self.register_buffer("bool_masked", mask.flatten())
        self.masked_tubelets = self.tt - visible_tubelets
        self.n_masked = int(mask.sum())
        self.register_buffer("imn_mean", IMN_MEAN.view(1, 1, 3, 1, 1))
        self.register_buffer("imn_std", IMN_STD.view(1, 1, 3, 1, 1))

    def _lag_image(self, history, stride):
        """history [B,H] -> [B,224,224], oldest lag rows at top."""
        batch, length = history.shape
        max_start = length - self.period
        starts = torch.arange(max_start, -1, -stride, device=history.device).flip(0)
        offsets = torch.arange(self.period, device=history.device)
        windows = history[:, starts[:, None] + offsets[None, :]]

        # Resize each length-P lag window horizontally.  Reserve the final
        # 16-pixel patch band for an exact copy of the newest lag row; pool all
        # older rows into the remaining 13 patch bands so all history contributes.
        rows = F.interpolate(
            windows.unsqueeze(1),
            size=(len(starts), IMG),
            mode="bilinear",
            align_corners=False,
        )
        current = rows[:, :, -1:].expand(batch, 1, self.ps, IMG)
        past = rows[:, :, :-1]
        past_height = IMG - self.ps
        if past.shape[-2] > past_height:
            past = F.adaptive_avg_pool2d(past, (past_height, IMG))
        elif past.shape[-2] < past_height:
            past = F.interpolate(
                past, size=(past_height, IMG), mode="bilinear", align_corners=False
            )
        return torch.cat((past, current), dim=2).squeeze(1)

    def render(self, x):
        if x.shape[1] != self.context:
            raise ValueError(f"expected context {self.context}, got {x.shape[1]}")
        batch = x.shape[0]
        mu = x.mean(1, keepdim=True)
        sd = x.std(1, keepdim=True) + 1e-8
        gray = (((x - mu) / (3 * sd)).clamp(-1, 1) + 1) / 2

        frames = []
        for frame_idx in range(VISIBLE_FRAMES):
            start = frame_idx * self.period
            history = gray[:, start:start + self.history_steps]
            channels = [
                self._lag_image(history, stride) for stride in self.channel_strides
            ]
            frames.append(torch.stack(channels, dim=1))
        neutral = torch.full(
            (batch, 3, IMG, IMG), 0.5, dtype=x.dtype, device=x.device
        )
        frames.extend([neutral] * PRED_FRAMES)
        video = torch.stack(frames, dim=1)
        return (video - self.imn_mean) / self.imn_std, mu, sd

    def _visible_patch_stats(self, video):
        batch = video.shape[0]
        tokens = video.view(
            batch, self.tt, self.ts, 3, self.gh, self.ps, self.gh, self.ps
        )
        tokens = tokens.permute(0, 1, 4, 6, 2, 5, 7, 3).reshape(
            batch, self.tt, self.gh, self.gh, self.ts * self.ps * self.ps, 3
        )
        visible = tokens[:, :VISIBLE_FRAMES // self.ts]
        mean = visible.mean(dim=4).mean(dim=1)
        std = (visible.var(dim=4, unbiased=True) + 1e-6).sqrt().mean(dim=1)
        return mean, std

    def forward(self, x, clip_output=False):
        batch = x.shape[0]
        video, mu, sd = self.render(x)
        with torch.no_grad():
            patch_mean, patch_std = self._visible_patch_stats(video)
        output = self.m(
            pixel_values=video,
            bool_masked_pos=self.bool_masked.unsqueeze(0).expand(batch, -1),
        )
        pred = output.logits.view(
            batch,
            self.masked_tubelets,
            self.gh,
            self.gh,
            self.ts * self.ps * self.ps,
            3,
        )
        rec = pred * patch_std[:, None, :, :, None, :] + patch_mean[:, None, :, :, None, :]
        rec = rec.view(
            batch,
            self.masked_tubelets,
            self.gh,
            self.gh,
            self.ts,
            self.ps,
            self.ps,
            3,
        )
        rec = rec.permute(0, 1, 4, 7, 2, 5, 3, 6).reshape(
            batch, PRED_FRAMES, 3, IMG, IMG
        )
        rec = rec * self.imn_std + self.imn_mean

        # The bottom patch band is the newest lag window in all three channels.
        # Decode R, the fine-scale channel, and invert the window normalization.
        newest = rec[:, :, 0, -self.ps:, :].mean(dim=2)
        values = F.interpolate(
            newest.reshape(batch * PRED_FRAMES, 1, IMG),
            size=self.period,
            mode="linear",
            align_corners=False,
        ).reshape(batch, self.horizon)
        zhat = 2 * values - 1
        if clip_output:
            zhat = zhat.clamp(-1, 1)
        return zhat * 3 * sd + mu


# --------------------------------------------------------------- train/eval
def predict(model, x, batch_size):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(DEVICE)
            with torch.autocast(
                device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"
            ):
                pred = model(xb, clip_output=True)
            outputs.append(pred.float().cpu().numpy())
    return np.concatenate(outputs)


def train_lora(model, x, y, epochs, batch_size, lr, log_every):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("LoRA training requested with no trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    model.train()
    for epoch in range(epochs):
        permutation = np.random.default_rng(epoch).permutation(len(x))
        total_loss = 0.0
        for step, start in enumerate(range(0, len(x), batch_size)):
            idx = permutation[start:start + batch_size]
            xb = torch.from_numpy(x[idx]).to(DEVICE)
            yb = torch.from_numpy(y[idx]).to(DEVICE)
            with torch.autocast(
                device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"
            ):
                loss = F.mse_loss(model(xb, clip_output=False).float(), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)
            if step % log_every == 0:
                print(
                    f"[train] epoch={epoch + 1} step={step} loss={loss.item():.4f}",
                    flush=True,
                )
        print(
            f"[train] epoch={epoch + 1} mean_MSE={total_loss / len(x):.4f}",
            flush=True,
        )


def metrics(pred, target):
    return {
        "MSE": round(float(np.mean((pred - target) ** 2)), 4),
        "MAE": round(float(np.mean(np.abs(pred - target))), 4),
    }


# ---------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(rp.DATASETS))
    parser.add_argument("--data-dir", default="pilot/data")
    parser.add_argument("--out-dir", default="pilot/results_lagged_lora")
    parser.add_argument("--model", default="MCG-NJU/videomae-base")
    parser.add_argument("--methods", default="frozen,lora")
    parser.add_argument("--history-periods", type=int, default=14)
    parser.add_argument(
        "--channel-strides",
        default="auto",
        help="auto = (1, P/4, P) raw steps; otherwise comma-separated integers",
    )
    parser.add_argument("--train-cap", type=int, default=2048)
    parser.add_argument("--test-cap", type=int, default=1024)
    parser.add_argument(
        "--test-stride", type=int, default=0, help="raw steps; 0 means one period"
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch", type=int, default=8)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if args.history_periods < 2:
        parser.error("--history-periods must be at least 2")
    if args.lora_rank <= 0 or args.lora_alpha <= 0:
        parser.error("LoRA rank and alpha must be positive")
    methods = [name.strip() for name in args.methods.split(",") if name.strip()]
    unknown = set(methods) - {"frozen", "lora"}
    if unknown:
        parser.error(f"unknown methods: {sorted(unknown)}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data, _ = rp.load_dataset(args.dataset, args.data_dir)
    period = rp.P
    strides = resolve_channel_strides(args.channel_strides, period)
    context = (args.history_periods + VISIBLE_FRAMES - 1) * period
    horizon = PRED_FRAMES * period
    test_stride = args.test_stride or period
    x_train, y_train, train_total = get_windows(
        data,
        args.dataset,
        context,
        horizon,
        "train",
        1,
        args.train_cap,
        args.seed,
    )
    x_test, y_test, test_total = get_windows(
        data,
        args.dataset,
        context,
        horizon,
        "test",
        test_stride,
        args.test_cap,
        args.seed + 1,
    )
    print(
        f"dataset={args.dataset} device={DEVICE} P={period} context={context} "
        f"horizon={horizon} strides={strides} train={len(x_train)}/{train_total} "
        f"test={len(x_test)}/{test_total}",
        flush=True,
    )

    total_periods = context // period
    results = {
        "naive": metrics(np.repeat(x_test[:, -1:], horizon, axis=1), y_test),
        "snaive": metrics(np.tile(x_test[:, -period:], (1, PRED_FRAMES)), y_test),
        "smean": metrics(
            np.tile(x_test.reshape(-1, total_periods, period).mean(1),
                    (1, PRED_FRAMES)),
            y_test,
        ),
    }

    adapter_path = None
    lora_targets = []
    trainable_params = 0
    if methods:
        model = LaggedStrideVMAE(
            period, args.history_periods, strides, args.model
        ).to(DEVICE)
        for parameter in model.parameters():
            parameter.requires_grad = False

        if "frozen" in methods:
            pred = predict(model, x_test, args.eval_batch)
            results["frozen"] = metrics(pred, y_test)
            print(f"[done] frozen {results['frozen']}", flush=True)

        if "lora" in methods:
            lora_targets, trainable_params = inject_qv_lora(
                model, args.lora_rank, args.lora_alpha
            )
            total_params = sum(p.numel() for p in model.parameters())
            print(
                f"[info] LoRA targets={len(lora_targets)} trainable="
                f"{trainable_params:,}/{total_params:,}",
                flush=True,
            )
            train_lora(
                model,
                x_train,
                y_train,
                args.epochs,
                args.train_batch,
                args.lr,
                args.log_every,
            )
            pred = predict(model, x_test, args.eval_batch)
            results["lora"] = metrics(pred, y_test)
            print(f"[done] lora {results['lora']}", flush=True)

            os.makedirs(args.out_dir, exist_ok=True)
            adapter_path = os.path.join(
                args.out_dir, f"lagged_lora_{args.dataset}_r{args.lora_rank}.pt"
            )
            torch.save(
                {
                    "state_dict": adapter_state_dict(model),
                    "model": args.model,
                    "period": period,
                    "history_periods": args.history_periods,
                    "channel_strides": strides,
                    "rank": args.lora_rank,
                    "alpha": args.lora_alpha,
                    "targets": lora_targets,
                },
                adapter_path,
            )
            print(f"[adapter] {adapter_path}", flush=True)

    config = {
        "dataset": args.dataset,
        "model": args.model,
        "period": period,
        "context": context,
        "horizon": horizon,
        "history_periods": args.history_periods,
        "visible_frames": VISIBLE_FRAMES,
        "pred_frames": PRED_FRAMES,
        "channel_strides": strides,
        "train_windows": len(x_train),
        "train_candidates": train_total,
        "test_windows": len(x_test),
        "test_candidates": test_total,
        "test_stride": test_stride,
        "epochs": args.epochs,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_targets": len(lora_targets),
        "trainable_params": trainable_params,
        "adapter": adapter_path,
        "seed": args.seed,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    output_path = os.path.join(args.out_dir, f"lagged_lora_{args.dataset}.json")
    with open(output_path, "w") as handle:
        json.dump({"config": config, "results": results}, handle, indent=2)
    print(json.dumps(results, indent=2), flush=True)
    print(f"[results] {output_path}", flush=True)


if __name__ == "__main__":
    main()
