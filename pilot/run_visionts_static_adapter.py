"""Train MLP adapters around frozen VideoMAE using VisionTS rendering.

Path:
    context -> VisionTS image -> pixel ingest MLP -> frozen VideoMAE
            -> forecast MLP -> horizon

VisionTS produces one 224x224 grayscale image.  VideoMAE-base has a fixed
two-frame tubelet convolution, so the image is duplicated once to make the
smallest clip the checkpoint can consume: one static two-frame tubelet.  Only
the ingest and forecast MLPs are trainable.
"""

import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, TensorDataset

import run_mlp_adapter as baseline
import run_pilot as rp


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
PATCH_SIZE = 16
GRID = IMAGE_SIZE // PATCH_SIZE
NORM_CONST = 0.4


class VisionTSRenderer(nn.Module):
    """Exact VisionTS 1.0 segmentation, normalization, and image alignment."""

    def __init__(
        self, context, horizon, periodicity, norm_const=NORM_CONST,
        align_const=0.4,
    ):
        super().__init__()
        from visionts.util import safe_resize

        self.context = context
        self.horizon = horizon
        self.periodicity = periodicity
        self.norm_const = norm_const
        self.pad_left = (-context) % periodicity
        self.pad_right = (-horizon) % periodicity
        ratio = (self.pad_left + context) / (
            self.pad_left + context + self.pad_right + horizon
        )
        self.num_patch_input = max(int(ratio * GRID * align_const), 1)
        self.num_patch_output = GRID - self.num_patch_input
        observed_width = self.num_patch_input * PATCH_SIZE
        self.input_resize = safe_resize(
            (IMAGE_SIZE, observed_width), interpolation=Image.BILINEAR
        )

    def statistics(self, x):
        mean = x.mean(1, keepdim=True).detach()
        scale = torch.sqrt(
            torch.var(x - mean, dim=1, keepdim=True, unbiased=False) + 1e-5
        ) / self.norm_const
        return mean, scale

    def render_normalized(self, normalized):
        """Render a pre-normalized, observed-only sequence."""
        if normalized.ndim != 2 or normalized.shape[1] != self.context:
            raise ValueError(
                f"expected [batch, {self.context}], got {tuple(normalized.shape)}"
            )
        if self.pad_left:
            normalized = F.pad(normalized, (self.pad_left, 0), mode="replicate")
        periods = normalized.shape[1] // self.periodicity
        image = normalized.reshape(-1, periods, self.periodicity)
        image = image.permute(0, 2, 1).unsqueeze(1)
        image = self.input_resize(image)
        masked = torch.zeros(
            image.shape[0], 1, IMAGE_SIZE,
            self.num_patch_output * PATCH_SIZE,
            dtype=image.dtype,
            device=image.device,
        )
        return torch.cat((image, masked), dim=-1)

    def render_with_statistics(self, x, mean, scale):
        """Render ``x`` using explicitly supplied observed-only statistics."""
        return self.render_normalized((x - mean) / scale)

    def forward(self, x):
        mean, scale = self.statistics(x)
        image = self.render_with_statistics(x, mean, scale)
        return image, mean, scale


class PixelIngestMLP(nn.Module):
    """Map each VisionTS grayscale value into VideoMAE's RGB input space."""

    def __init__(self, width):
        super().__init__()
        self.base = nn.Linear(1, 3)
        self.residual = nn.Sequential(
            nn.Linear(1, width),
            nn.GELU(),
            nn.Linear(width, 3),
        )
        with torch.no_grad():
            self.base.weight.fill_(1.0)
            self.base.bias.zero_()
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def forward(self, image):
        values = image.permute(0, 2, 3, 1)
        rgb = self.base(values) + self.residual(values)
        return rgb.permute(0, 3, 1, 2)


class StaticImageVideoMAEAdapter(nn.Module):
    def __init__(
        self, model_name, context, periodicity, horizon, ingest_width,
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
            config.tubelet_size != 2
            or config.image_size != IMAGE_SIZE
            or config.patch_size != PATCH_SIZE
        ):
            raise ValueError("expected VideoMAE-base's 2x16x16 patch geometry")
        self.renderer = VisionTSRenderer(context, horizon, periodicity)
        self.ingest_mlp = PixelIngestMLP(ingest_width)
        self.forecast_mlp = baseline.ForecastMLP(
            config.hidden_size, GRID * GRID, horizon, token_width, head_width
        )
        temporal_tokens = config.num_frames // config.tubelet_size
        position = self.backbone.embeddings.position_embeddings.detach()
        position = position.reshape(1, temporal_tokens, GRID * GRID, -1).mean(1)
        self.register_buffer("static_position", position, persistent=False)
        self.use_checkpoint = use_checkpoint

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def target_statistics(self, x):
        return self.renderer.statistics(x)

    def forward(self, x, return_normalized=False):
        image, mean, scale = self.renderer(x)
        rgb = self.ingest_mlp(image)
        # A literal one-frame tensor cannot pass a tubelet_size=2 convolution.
        # Two identical frames encode one static image as exactly one tubelet.
        clip = rgb.unsqueeze(1).expand(-1, 2, -1, -1, -1)
        hidden = self.backbone.embeddings.patch_embeddings(clip)
        hidden = hidden + self.static_position.type_as(hidden).to(hidden.device)
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
            mean, scale = model.target_statistics(xb)
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
    parser.add_argument("--out-dir", default="pilot/results_static_image")
    parser.add_argument("--model", default="MCG-NJU/videomae-base")
    parser.add_argument("--context", type=int, default=600)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--train-cap", type=int, default=4096)
    parser.add_argument("--val-cap", type=int, default=0)
    parser.add_argument("--test-cap", type=int, default=0)
    parser.add_argument("--eval-stride", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--ingest-width", type=int, default=32)
    parser.add_argument("--token-width", type=int, default=2)
    parser.add_argument("--head-width", type=int, default=256)
    parser.add_argument("--input-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--window-seed", type=int, default=None,
        help="window-sampling seed; defaults to --seed",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    data, _ = rp.load_dataset(args.dataset, args.data_dir)
    window_seed = args.seed if args.window_seed is None else args.window_seed
    bundles = {}
    totals = {}
    for split, stride, cap, seed in (
        ("train", 1, args.train_cap, window_seed),
        ("val", args.eval_stride, args.val_cap, window_seed + 1),
        ("test", args.eval_stride, args.test_cap, window_seed + 2),
    ):
        x, y, total = baseline.get_windows(
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

    model = StaticImageVideoMAEAdapter(
        args.model, args.context, rp.P, args.horizon, args.ingest_width,
        args.token_width, args.head_width, not args.no_checkpoint,
    ).to(DEVICE)
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or any(
        not (name.startswith("ingest_mlp") or name.startswith("forecast_mlp"))
        for name in trainable_names
    ):
        raise RuntimeError(f"unexpected trainable parameters: {trainable_names}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(
        f"visionts_input_patches={model.renderer.num_patch_input} "
        f"video_tubelets=1 trainable={trainable:,} frozen={frozen:,}",
        flush=True,
    )
    best_val = train_model(
        model, bundles["train"], bundles["val"], args.epochs,
        args.batch_size, args.input_lr, args.head_lr,
        args.weight_decay, args.patience,
    )
    result = evaluate(model, bundles["test"], max(args.batch_size, 16))
    rounded = {key: round(value, 4) for key, value in result.items()}
    print(f"[done] visionts_static_videomae {rounded}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    seed_suffix = (
        "" if args.seed == 0 and window_seed == 0
        else f"_s{args.seed}_ws{window_seed}"
    )
    output_path = os.path.join(
        args.out_dir,
        f"visionts_static_{args.dataset}_h{args.horizon}_n"
        f"{len(bundles['train'][0])}{seed_suffix}.json",
    )
    payload = {
        "config": {
            **vars(args),
            "period": rp.P,
            "visionts_norm_const": NORM_CONST,
            "visionts_align_const": 0.4,
            "visionts_input_patches": model.renderer.num_patch_input,
            "video_frames": 2,
            "video_tubelets": 1,
            "static_duplicate_frames": True,
            "train_windows": len(bundles["train"][0]),
            "val_windows": len(bundles["val"][0]),
            "test_windows": len(bundles["test"][0]),
            "model_seed": args.seed,
            "effective_window_seed": window_seed,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "best_validation_mse": best_val,
            "trainable_modules": ["ingest_mlp", "forecast_mlp"],
        },
        "results": {"visionts_static_videomae": rounded},
    }
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[results] {output_path}", flush=True)


if __name__ == "__main__":
    main()
