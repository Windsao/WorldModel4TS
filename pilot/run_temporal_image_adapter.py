"""Causal temporal augmentations of VisionTS images for frozen VideoMAE.

The following dataset-independent views are implemented:

``prefix_video``
    Uniformly spaced causal prefixes of the observed context become a short
    video.  No lag length or dataset-specific offset is selected.

``dyadic_rgb``
    The full context is sampled at strides 1, 2, and 4.  Their VisionTS images
    become RGB channels of one static image (duplicated only to satisfy
    VideoMAE's tubelet-size-two convolution).

``dyadic_rgb_residual``
    The same scale-space, but the ingest MLP is initialized to repeat only the
    full-resolution channel.  It therefore starts exactly at the static-image
    representation and learns coarse-scale contributions as a residual.

``dyadic_prefix_video``
    Uniform causal prefixes are true video frames, and each frame uses the
    same dyadic RGB scale-space.  With two frames this remains one tubelet and
    has the same forecast-head capacity as ``dyadic_rgb``.

``dyadic_scale_video``
    A coarse-to-fine video whose grayscale frames use dyadic sampling strides
    (for four frames: 8, 4, 2, 1).  This exposes temporal scale to VideoMAE's
    temporal convolution without dataset-specific lag choices.

``step_shift_video``
    Full-context frames end at consecutive observed steps.  A two-frame clip is
    exactly previous-step then current-step and uses one tubelet/196 tokens—the
    same capacity as the static model.  Repeat/reverse variants isolate whether
    the genuine one-step arrow of time contributes.

``step_shift_dyadic_rgb_residual``
    The same previous/current ordering, but each frame contains the aligned
    stride-1/2/4 history views in RGB.  Previous views reuse their matching
    current-view statistics so normalization cannot erase temporal change.
    Its repeat control is exactly the residual-dyadic-RGB input.

Only the pixel ingest MLP and final forecast MLP are trainable.
"""

import argparse
import gc
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import run_mlp_adapter as baseline
import run_pilot as rp
import run_visionts_static_adapter as static


DEVICE = static.DEVICE
GRID = static.GRID


class CausalImageViews(nn.Module):
    def __init__(self, context, horizon, periodicity, mode, video_frames):
        super().__init__()
        self.context = context
        self.horizon = horizon
        self.periodicity = periodicity
        self.mode = mode
        if mode == "prefix_video":
            if video_frames < 2 or video_frames % 2:
                raise ValueError("prefix_video requires a positive even frame count")
            endpoints = np.rint(
                np.linspace(context / video_frames, context, video_frames)
            ).astype(int)
            endpoints = np.maximum(endpoints, min(periodicity, context))
            if len(np.unique(endpoints)) != video_frames:
                raise ValueError("context is too short for distinct causal prefixes")
            self.endpoints = tuple(int(value) for value in endpoints)
            self.renderers = nn.ModuleList(
                static.VisionTSRenderer(value, horizon, periodicity)
                for value in self.endpoints
            )
            self.video_frames = video_frames
            self.input_channels = 1
        elif mode in {
            "dyadic_rgb", "dyadic_rgb_residual", "dyadic_prefix_video",
            "step_shift_dyadic_rgb_residual",
            "step_shift_dyadic_rgb_residual_repeat",
            "step_shift_dyadic_rgb_residual_reverse",
        }:
            self.scales = (1, 2, 4)
            if periodicity % self.scales[-1] or horizon % self.scales[-1]:
                raise ValueError("periodicity and horizon must support dyadic scaling")
            if mode in {"dyadic_rgb", "dyadic_rgb_residual"}:
                self.endpoints = (context,)
                self.video_frames = 2
            elif mode.startswith("step_shift_dyadic_rgb_residual"):
                if video_frames != 2:
                    raise ValueError(
                        "step-shift dyadic RGB is defined for exactly two frames"
                    )
                if context < 2:
                    raise ValueError(
                        "context is too short for a previous/current pair"
                    )
                self.endpoints = (context,)
                self.video_frames = 2
            else:
                if video_frames < 2 or video_frames % 2:
                    raise ValueError(
                        "dyadic_prefix_video requires a positive even frame count"
                    )
                endpoints = np.rint(
                    np.linspace(context / video_frames, context, video_frames)
                ).astype(int)
                self.endpoints = tuple(int(value) for value in endpoints)
                self.video_frames = video_frames
            self.renderers = nn.ModuleList()
            for endpoint in self.endpoints:
                group = nn.ModuleList()
                for scale in self.scales:
                    start = (endpoint - 1) % scale
                    length = len(range(start, endpoint, scale))
                    group.append(
                        static.VisionTSRenderer(
                            length, horizon // scale, periodicity // scale
                        )
                    )
                self.renderers.append(group)
            self.input_channels = 3
        elif mode == "dyadic_scale_video":
            if video_frames < 2 or video_frames % 2:
                raise ValueError(
                    "dyadic_scale_video requires a positive even frame count"
                )
            self.scales = tuple(
                2 ** exponent for exponent in range(video_frames - 1, -1, -1)
            )
            if periodicity % self.scales[0] or horizon % self.scales[0]:
                raise ValueError("periodicity and horizon must support all scales")
            self.renderers = nn.ModuleList()
            for stride in self.scales:
                start = (context - 1) % stride
                length = len(range(start, context, stride))
                self.renderers.append(
                    static.VisionTSRenderer(
                        length, horizon // stride, periodicity // stride
                    )
                )
            self.video_frames = video_frames
            self.input_channels = 1
        elif mode in {
            "step_shift_video",
            "step_shift_video_repeat",
            "step_shift_video_reverse",
        }:
            if video_frames != 2:
                raise ValueError(
                    "step-shift video is validated for exactly two frames"
                )
            if context < 2:
                raise ValueError("context is too short for a previous/current pair")
            self.shift_renderer = static.VisionTSRenderer(
                context, horizon, periodicity
            )
            self.video_frames = video_frames
            self.input_channels = 1
        else:
            raise ValueError(f"unknown temporal image mode: {mode}")
        self.full_renderer = static.VisionTSRenderer(
            context, horizon, periodicity
        )

    def statistics(self, x):
        return self.full_renderer.statistics(x)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.context:
            raise ValueError(f"expected [batch, {self.context}], got {tuple(x.shape)}")
        mean, scale = self.statistics(x)
        if self.mode == "prefix_video":
            images = [
                renderer(x[:, :endpoint])[0]
                for endpoint, renderer in zip(self.endpoints, self.renderers)
            ]
            return torch.stack(images, dim=1), mean, scale

        if self.mode == "dyadic_scale_video":
            images = []
            for stride, renderer in zip(self.scales, self.renderers):
                start = (self.context - 1) % stride
                images.append(renderer(x[:, start::stride])[0])
            return torch.stack(images, dim=1), mean, scale

        if self.mode.startswith("step_shift_video"):
            images = []
            for lag in range(self.video_frames - 1, -1, -1):
                if lag:
                    shifted = torch.cat(
                        (x[:, :1].expand(-1, lag), x[:, :-lag]), dim=1
                    )
                else:
                    shifted = x
                images.append(
                    self.shift_renderer.render_with_statistics(
                        shifted, mean, scale
                    )
                )
            if self.mode == "step_shift_video_repeat":
                images = [images[-1]] * self.video_frames
            elif self.mode == "step_shift_video_reverse":
                images = images[::-1]
            return torch.stack(images, dim=1), mean, scale

        if self.mode.startswith("step_shift_dyadic_rgb_residual"):
            shifted = torch.cat((x[:, :1], x[:, :-1]), dim=1)
            previous_channels = []
            current_channels = []
            for stride, renderer in zip(self.scales, self.renderers[0]):
                start = (self.context - 1) % stride
                current = x[:, start::stride]
                previous = shifted[:, start::stride]
                view_mean, view_scale = renderer.statistics(current)
                current_channels.append(
                    renderer.render_with_statistics(
                        current, view_mean, view_scale
                    )
                )
                previous_channels.append(
                    renderer.render_with_statistics(
                        previous, view_mean, view_scale
                    )
                )
            current_frame = torch.cat(current_channels, dim=1)
            previous_frame = torch.cat(previous_channels, dim=1)
            frames = [previous_frame, current_frame]
            if self.mode == "step_shift_dyadic_rgb_residual_repeat":
                frames = [current_frame, current_frame]
            elif self.mode == "step_shift_dyadic_rgb_residual_reverse":
                frames = frames[::-1]
            return torch.stack(frames, dim=1), mean, scale

        frames = []
        for endpoint, group in zip(self.endpoints, self.renderers):
            images = []
            for stride, renderer in zip(self.scales, group):
                # Align each view so it contains the prefix's latest sample.
                start = (endpoint - 1) % stride
                view = x[:, start:endpoint:stride]
                images.append(renderer(view)[0])
            frames.append(torch.cat(images, dim=1))
        clip = torch.stack(frames, dim=1)
        if self.mode in {"dyadic_rgb", "dyadic_rgb_residual"}:
            clip = clip.expand(-1, 2, -1, -1, -1)
        return clip, mean, scale


class ChannelIngestMLP(nn.Module):
    def __init__(self, input_channels, width, repeat_first=False):
        super().__init__()
        self.input_channels = input_channels
        self.base = nn.Linear(input_channels, 3)
        self.residual = nn.Sequential(
            nn.Linear(input_channels, width),
            nn.GELU(),
            nn.Linear(width, 3),
        )
        with torch.no_grad():
            self.base.weight.zero_()
            if input_channels == 1:
                self.base.weight.fill_(1.0)
            elif input_channels == 3 and repeat_first:
                self.base.weight[:, 0] = 1.0
            elif input_channels == 3:
                self.base.weight.copy_(torch.eye(3))
            self.base.bias.zero_()
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def forward(self, frames):
        batch, time, channels, height, width = frames.shape
        values = frames.permute(0, 1, 3, 4, 2)
        rgb = self.base(values) + self.residual(values)
        return rgb.permute(0, 1, 4, 2, 3).reshape(
            batch, time, 3, height, width
        )


class TemporalImageVideoMAEAdapter(nn.Module):
    def __init__(
        self, model_name, context, periodicity, horizon, mode, video_frames,
        ingest_width, token_width, head_width, use_checkpoint=True,
    ):
        super().__init__()
        from transformers import VideoMAEModel

        self.backbone = VideoMAEModel.from_pretrained(model_name)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()
        config = self.backbone.config
        self.views = CausalImageViews(
            context, horizon, periodicity, mode, video_frames
        )
        if self.views.video_frames > config.num_frames:
            raise ValueError("temporal view has more frames than the checkpoint")
        self.ingest_mlp = ChannelIngestMLP(
            self.views.input_channels, ingest_width,
            repeat_first=(
                mode == "dyadic_rgb_residual"
                or mode.startswith("step_shift_dyadic_rgb_residual")
            ),
        )
        tubelets = self.views.video_frames // config.tubelet_size
        self.forecast_mlp = baseline.ForecastMLP(
            config.hidden_size, tubelets * GRID * GRID, horizon,
            token_width, head_width,
        )
        all_position = self.backbone.embeddings.position_embeddings.detach()
        all_position = all_position.reshape(
            1, config.num_frames // config.tubelet_size, GRID * GRID, -1
        )
        if tubelets == 1:
            position = all_position.mean(1)
        else:
            indices = torch.linspace(
                0, all_position.shape[1] - 1, tubelets
            ).round().long()
            position = all_position[:, indices].reshape(
                1, tubelets * GRID * GRID, -1
            )
        self.register_buffer("view_position", position, persistent=False)
        self.use_checkpoint = use_checkpoint

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def target_statistics(self, x):
        return self.views.statistics(x)

    def forward(self, x, return_normalized=False):
        frames, mean, scale = self.views(x)
        clip = self.ingest_mlp(frames)
        hidden = self.backbone.embeddings.patch_embeddings(clip)
        hidden = hidden + self.view_position.type_as(hidden).to(hidden.device)
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
                hidden = block(
                    hidden, head_mask=None, output_attentions=False
                )[0]
        if self.backbone.layernorm is not None:
            hidden = self.backbone.layernorm(hidden)
        prediction = self.forecast_mlp(hidden)
        if return_normalized:
            return prediction
        return prediction * scale + mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1", choices=list(rp.DATASETS))
    parser.add_argument("--data-dir", default="pilot/data")
    parser.add_argument("--out-dir", default="pilot/results_temporal_image")
    parser.add_argument("--model", default="MCG-NJU/videomae-base")
    parser.add_argument(
        "--mode", required=True,
        choices=(
            "prefix_video", "dyadic_rgb", "dyadic_rgb_residual",
            "dyadic_prefix_video",
            "dyadic_scale_video",
            "step_shift_video", "step_shift_video_repeat",
            "step_shift_video_reverse",
            "step_shift_dyadic_rgb_residual",
            "step_shift_dyadic_rgb_residual_repeat",
            "step_shift_dyadic_rgb_residual_reverse",
        )
    )
    parser.add_argument("--video-frames", type=int, default=4)
    parser.add_argument("--context", type=int, default=600)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--train-cap", type=int, default=4096)
    parser.add_argument("--val-cap", type=int, default=1024)
    parser.add_argument("--test-cap", type=int, default=1024)
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
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--eval-controls", action="store_true",
        help="evaluate the selected checkpoint on repeated/reversed validation views",
    )
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
        f"dataset={args.dataset} mode={args.mode} context={args.context} "
        f"horizon={args.horizon} train={len(bundles['train'][0])}/{totals['train']} "
        f"val={len(bundles['val'][0])}/{totals['val']} "
        f"test={'skipped' if args.skip_test else len(bundles['test'][0])}",
        flush=True,
    )

    model = TemporalImageVideoMAEAdapter(
        args.model, args.context, rp.P, args.horizon, args.mode,
        args.video_frames, args.ingest_width, args.token_width,
        args.head_width, not args.no_checkpoint,
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
    with torch.no_grad():
        diagnostic_x = torch.from_numpy(
            bundles["train"][0][:min(16, len(bundles["train"][0]))]
        ).to(DEVICE)
        diagnostic_frames, _, _ = model.views(diagnostic_x)
        temporal_delta_mse = float(
            (diagnostic_frames[:, 1:] - diagnostic_frames[:, :-1])
            .float().square().mean().item()
        )
    print(
        f"video_frames={model.views.video_frames} "
        f"tubelets={model.views.video_frames // 2} "
        f"temporal_delta_MSE={temporal_delta_mse:.6f} "
        f"trainable={trainable:,} frozen={frozen:,}",
        flush=True,
    )
    best_val = static.train_model(
        model, bundles["train"], bundles["val"], args.epochs,
        args.batch_size, args.input_lr, args.head_lr,
        args.weight_decay, args.patience,
    )
    validation_controls = {}
    control_modes = {
        "step_shift_video": (
            "step_shift_video_repeat", "step_shift_video_reverse"
        ),
        "step_shift_dyadic_rgb_residual": (
            "step_shift_dyadic_rgb_residual_repeat",
            "step_shift_dyadic_rgb_residual_reverse",
        ),
    }
    if args.eval_controls and args.mode in control_modes:
        original_mode = model.views.mode
        for control_mode in control_modes[args.mode]:
            model.views.mode = control_mode
            score = static.evaluate(
                model, bundles["val"], max(args.batch_size, 16)
            )
            validation_controls[control_mode] = {
                key: round(value, 4) for key, value in score.items()
            }
            print(
                f"[control:validation] {control_mode} "
                f"{validation_controls[control_mode]}",
                flush=True,
            )
        model.views.mode = original_mode
    result = None
    if not args.skip_test:
        result = static.evaluate(model, bundles["test"], max(args.batch_size, 16))
        result = {key: round(value, 4) for key, value in result.items()}
        print(f"[done] {args.mode} {result}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    seed_suffix = (
        "" if args.seed == 0 and window_seed == 0
        else f"_s{args.seed}_ws{window_seed}"
    )
    path = os.path.join(
        args.out_dir,
        f"{args.mode}_{args.dataset}_h{args.horizon}_n"
        f"{len(bundles['train'][0])}{seed_suffix}.json",
    )
    payload = {
        "config": {
            **vars(args),
            "periodicity": rp.P,
            "actual_video_frames": model.views.video_frames,
            "train_windows": len(bundles["train"][0]),
            "val_windows": len(bundles["val"][0]),
            "test_windows": 0 if args.skip_test else len(bundles["test"][0]),
            "best_validation_mse": best_val,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "test_used_for_selection": False,
            "input_temporal_delta_mse": temporal_delta_mse,
            "model_seed": args.seed,
            "effective_window_seed": window_seed,
        },
        "results": {} if result is None else {args.mode: result},
        "validation_controls": validation_controls,
    }
    if args.mode in {"prefix_video", "dyadic_prefix_video"}:
        payload["config"]["prefix_endpoints"] = model.views.endpoints
    if args.mode in {
        "dyadic_rgb", "dyadic_rgb_residual", "dyadic_prefix_video"
    } or args.mode.startswith("step_shift_dyadic_rgb_residual"):
        payload["config"]["dyadic_scales"] = model.views.scales
    if args.mode == "dyadic_scale_video":
        payload["config"]["dyadic_scales"] = model.views.scales
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[results] {path}", flush=True)
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
