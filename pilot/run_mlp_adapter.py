"""Train input/output MLPs around a completely frozen VideoMAE encoder.

Path:
    numeric context -> ingest MLP -> frozen VideoMAE -> forecast MLP -> horizon

The input is arranged on VideoMAE's [8 tubelets, 14, 14] token volume.  Each
token receives the two frames' RGB values (six scalars).  The ingest MLP is
initialized to exactly reproduce the checkpoint's 3D patch convolution for a
constant 16x16 patch, then learns a nonlinear residual.  The forecast MLP sees
only the final output of all frozen encoder blocks.  There is no raw-series or
patch-embedding skip to the output head.
"""

import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, TensorDataset

import run_pilot as rp


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID = 14
FRAMES = 16
TUBELETS = 8
IMN_MEAN = torch.tensor(rp.IMN_MEAN).view(1, 1, 1, 1, 3)
IMN_STD = torch.tensor(rp.IMN_STD).view(1, 1, 1, 1, 3)


def dataset_borders(data, name):
    config = rp.DATASETS[name]
    if config["kind"] == "ett":
        return config["borders"]
    total = len(data)
    return int(0.7 * total), int(0.8 * total), total


def get_windows(data, name, context, horizon, split, stride, cap, seed):
    train_end, validation_end, test_end = dataset_borders(data, name)
    bounds = {
        "train": (context, train_end - horizon),
        "val": (train_end, validation_end - horizon),
        "test": (validation_end, test_end - horizon),
    }
    first, last = bounds[split]
    times = np.arange(first, last + 1, stride, dtype=np.int64)
    channels = data.shape[1]
    total = len(times) * channels
    if cap and total > cap:
        picks = np.sort(np.random.default_rng(seed).choice(total, cap, replace=False))
    else:
        picks = np.arange(total)
    x = np.empty((len(picks), context), dtype=np.float32)
    y = np.empty((len(picks), horizon), dtype=np.float32)
    for row, pick in enumerate(picks):
        time = times[pick // channels]
        channel = pick % channels
        x[row] = data[time - context:time, channel]
        y[row] = data[time:time + horizon, channel]
    return x, y, total


class PhaseTubeletTokenizer(nn.Module):
    """Create six phase-grid features for every VideoMAE tubelet token.

    The original hourly layout is preserved exactly.  For other cadences, each
    half-period is bilinearly aligned to at most 14 patch rows; when fewer than
    14 complete periods fit in the context, the period axis is aligned to 14
    columns.  These are geometry-driven resizes, not dataset-specific lags.
    """

    def __init__(self, context, period):
        super().__init__()
        if period < 2:
            raise ValueError("period must contain at least two steps")
        self.context = context
        self.period = period
        self.pad_left = (-context) % period
        self.periods = (context + self.pad_left) // period
        if self.periods >= GRID:
            starts = torch.linspace(
                0, self.periods - GRID, TUBELETS
            ).round().long()
        else:
            starts = torch.zeros(TUBELETS, dtype=torch.long)
        self.register_buffer("starts", starts)

    @staticmethod
    def display_scale(value):
        center = value.mean(1, keepdim=True)
        scale = value.std(1, keepdim=True, unbiased=False) + 1e-6
        return (((value - center) / (3 * scale)).clamp(-1, 1) + 1) / 2

    def forward(self, x):
        batch = x.shape[0]
        mean = x.mean(1, keepdim=True)
        scale = x.std(1, keepdim=True, unbiased=False) + 1e-6
        first = torch.diff(x, dim=1, prepend=x[:, :1])
        seasonal = x - torch.cat((x[:, :self.period], x[:, :-self.period]), 1)
        views = torch.stack(
            (
                self.display_scale(x),
                self.display_scale(first),
                self.display_scale(seasonal),
            ),
            dim=-1,
        )
        if self.pad_left:
            views = torch.cat(
                (views[:, :1].expand(-1, self.pad_left, -1), views), dim=1
            )
        views = views.view(batch, self.periods, self.period, 3)

        frames = torch.full(
            (batch, FRAMES, GRID, GRID, 3),
            0.5,
            dtype=x.dtype,
            device=x.device,
        )
        phase_bounds = (0, self.period // 2, self.period)
        for tubelet, start in enumerate(self.starts.tolist()):
            period_count = min(GRID, self.periods)
            for half, (low, high) in enumerate(zip(phase_bounds, phase_bounds[1:])):
                frame = 2 * tubelet + half
                grid = views[
                    :, start:start + period_count, low:high
                ].permute(0, 3, 2, 1)
                rows = min(GRID, high - low)
                if grid.shape[-2:] != (rows, GRID):
                    grid = F.interpolate(
                        grid, size=(rows, GRID), mode="bilinear",
                        align_corners=False,
                    )
                frames[:, frame, :rows] = grid.permute(0, 2, 3, 1)
        frames = (frames - IMN_MEAN.to(x.device)) / IMN_STD.to(x.device)
        # HF VideoMAE token order is tubelet, patch row, patch column.  Six
        # features are temporal-0 RGB followed by temporal-1 RGB.
        token_features = frames.view(batch, TUBELETS, 2, GRID, GRID, 3)
        token_features = token_features.permute(0, 1, 3, 4, 2, 5)
        return token_features.reshape(batch, TUBELETS * GRID * GRID, 6), mean, scale


class IngestMLP(nn.Module):
    def __init__(self, patch_projection, width):
        super().__init__()
        hidden = patch_projection.out_channels
        self.base = nn.Linear(6, hidden)
        self.residual = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, width),
            nn.GELU(),
            nn.Linear(width, hidden),
        )
        weight = patch_projection.weight.detach()
        directions = []
        for temporal in range(2):
            for channel in range(3):
                directions.append(weight[:, channel, temporal].sum(dim=(-1, -2)))
        with torch.no_grad():
            self.base.weight.copy_(torch.stack(directions, dim=1))
            if patch_projection.bias is None:
                self.base.bias.zero_()
            else:
                self.base.bias.copy_(patch_projection.bias)
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features):
        return self.base(features) + self.residual(features)


class ForecastMLP(nn.Module):
    def __init__(self, hidden, tokens, horizon, token_width, head_width):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, token_width),
            nn.GELU(),
        )
        self.readout = nn.Sequential(
            nn.Linear(tokens * token_width, head_width),
            nn.GELU(),
            nn.Linear(head_width, horizon),
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, hidden):
        return self.readout(self.token_mlp(hidden).flatten(1))


class FrozenVideoMAEAdapter(nn.Module):
    def __init__(
        self, model_name, context, period, horizon, ingest_width,
        token_width, head_width, use_checkpoint=True,
    ):
        super().__init__()
        import transformers

        if int(transformers.__version__.split(".", 1)[0]) >= 5:
            raise RuntimeError("transformers<5 is required for this checkpoint")
        from transformers import VideoMAEModel

        self.backbone = VideoMAEModel.from_pretrained(model_name)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()
        config = self.backbone.config
        if (
            config.num_frames != FRAMES
            or config.tubelet_size != 2
            or config.image_size // config.patch_size != GRID
        ):
            raise ValueError("this runner expects VideoMAE's native 16x224 geometry")
        self.tokenizer = PhaseTubeletTokenizer(context, period)
        self.ingest_mlp = IngestMLP(
            self.backbone.embeddings.patch_embeddings.projection, ingest_width
        )
        self.forecast_mlp = ForecastMLP(
            config.hidden_size,
            TUBELETS * GRID * GRID,
            horizon,
            token_width,
            head_width,
        )
        self.use_checkpoint = use_checkpoint

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x, return_normalized=False):
        features, mean, scale = self.tokenizer(x)
        hidden = self.ingest_mlp(features)
        position = self.backbone.embeddings.position_embeddings
        hidden = hidden + position.type_as(hidden).to(hidden.device)
        for block in self.backbone.encoder.layer:
            if self.training and self.use_checkpoint:
                hidden = checkpoint(
                    lambda value, module=block: module(
                        value, head_mask=None, output_attentions=False
                    )[0],
                    hidden,
                    use_reentrant=False,
                )
            else:
                hidden = block(hidden, head_mask=None, output_attentions=False)[0]
        if self.backbone.layernorm is not None:
            hidden = self.backbone.layernorm(hidden)
        normalized_prediction = self.forecast_mlp(hidden)
        if return_normalized:
            return normalized_prediction
        return normalized_prediction * scale + mean


def evaluate(model, bundle, batch_size):
    x, y = bundle
    model.eval()
    squared = absolute = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(DEVICE)
            target = torch.from_numpy(y[start:start + batch_size]).to(DEVICE)
            with torch.autocast(
                device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"
            ):
                prediction = model(xb)
            difference = prediction.float() - target
            squared += difference.square().sum().item()
            absolute += difference.abs().sum().item()
            count += target.numel()
    return {"MSE": squared / count, "MAE": absolute / count}


def train_model(
    model, train_bundle, val_bundle, epochs, batch_size, input_lr, head_lr,
    weight_decay, patience,
):
    x, y = train_bundle
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        (
            {"params": model.ingest_mlp.parameters(), "lr": input_lr},
            {"params": model.forecast_mlp.parameters(), "lr": head_lr},
        ),
        weight_decay=weight_decay,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    best_state = None
    best_val = float("inf")
    stale = 0
    for epoch in range(epochs):
        model.train()
        total = 0.0
        examples = 0
        for step, (xb, target) in enumerate(loader):
            xb = xb.to(DEVICE)
            target = target.to(DEVICE)
            mean = xb.mean(1, keepdim=True)
            scale = xb.std(1, keepdim=True, unbiased=False) + 1e-6
            normalized_target = (target - mean) / scale
            with torch.autocast(
                device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"
            ):
                normalized_prediction = model(xb, return_normalized=True)
                loss = F.mse_loss(normalized_prediction.float(), normalized_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total += loss.item() * len(xb)
            examples += len(xb)
            if step % 100 == 0:
                print(
                    f"[train] epoch={epoch + 1} step={step}/{len(loader)} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )
        validation = evaluate(model, val_bundle, max(batch_size, 16))
        print(
            f"[epoch] {epoch + 1} train_MSE={total/examples:.4f} "
            f"val_MSE={validation['MSE']:.4f} val_MAE={validation['MAE']:.4f}",
            flush=True,
        )
        if validation["MSE"] < best_val - 1e-5:
            best_val = validation["MSE"]
            best_state = {
                "ingest_mlp": copy.deepcopy(model.ingest_mlp.state_dict()),
                "forecast_mlp": copy.deepcopy(model.forecast_mlp.state_dict()),
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.ingest_mlp.load_state_dict(best_state["ingest_mlp"])
    model.forecast_mlp.load_state_dict(best_state["forecast_mlp"])
    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1", choices=list(rp.DATASETS))
    parser.add_argument("--data-dir", default="pilot/data")
    parser.add_argument("--out-dir", default="pilot/results_mlp_adapter")
    parser.add_argument("--model", default="MCG-NJU/videomae-base")
    parser.add_argument("--context", type=int, default=600)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--train-cap", type=int, default=2048)
    parser.add_argument("--val-cap", type=int, default=512)
    parser.add_argument("--test-cap", type=int, default=0)
    parser.add_argument("--eval-stride", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--ingest-width", type=int, default=64)
    parser.add_argument("--token-width", type=int, default=2)
    parser.add_argument("--head-width", type=int, default=256)
    parser.add_argument("--input-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    data, _ = rp.load_dataset(args.dataset, args.data_dir)
    period = rp.P
    bundles = {}
    totals = {}
    for split, stride, cap, seed in (
        ("train", 1, args.train_cap, args.seed),
        ("val", args.eval_stride, args.val_cap, args.seed + 1),
        ("test", args.eval_stride, args.test_cap, args.seed + 2),
    ):
        x, y, total = get_windows(
            data, args.dataset, args.context, args.horizon,
            split, stride, cap, seed,
        )
        bundles[split] = (x, y)
        totals[split] = total
    print(
        f"dataset={args.dataset} context={args.context} horizon={args.horizon} "
        f"train={len(bundles['train'][0])}/{totals['train']} "
        f"val={len(bundles['val'][0])}/{totals['val']} "
        f"test={len(bundles['test'][0])}/{totals['test']}",
        flush=True,
    )

    model = FrozenVideoMAEAdapter(
        args.model, args.context, period, args.horizon,
        args.ingest_width, args.token_width, args.head_width,
        not args.no_checkpoint,
    ).to(DEVICE)
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or any(
        not (name.startswith("ingest_mlp") or name.startswith("forecast_mlp"))
        for name in trainable_names
    ):
        raise RuntimeError(f"unexpected trainable parameters: {trainable_names}")
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )
    print(f"trainable={trainable:,} frozen={frozen:,}", flush=True)
    best_val = train_model(
        model, bundles["train"], bundles["val"], args.epochs,
        args.batch_size, args.input_lr, args.head_lr,
        args.weight_decay, args.patience,
    )
    result = evaluate(model, bundles["test"], max(args.batch_size, 16))
    rounded = {key: round(value, 4) for key, value in result.items()}
    print(f"[done] mlp_adapter {rounded}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    output_path = os.path.join(
        args.out_dir,
        f"mlp_adapter_{args.dataset}_h{args.horizon}_n{len(bundles['train'][0])}.json",
    )
    payload = {
        "config": {
            **vars(args),
            "period": period,
            "train_windows": len(bundles["train"][0]),
            "val_windows": len(bundles["val"][0]),
            "test_windows": len(bundles["test"][0]),
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "best_validation_mse": best_val,
            "trainable_modules": ["ingest_mlp", "forecast_mlp"],
        },
        "results": {"mlp_adapter": rounded},
    }
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[results] {output_path}", flush=True)


if __name__ == "__main__":
    main()
