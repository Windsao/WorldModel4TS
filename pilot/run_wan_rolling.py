"""Zero-shot forecasting by causal rolling-chart video outpainting with Wan 2.1.

The mapping is deliberately invertible and aligned with video time:

* every frame advances ``horizon / 16`` numeric samples;
* the visible chart scrolls left by exactly that many samples;
* the newly entered right-edge stripes are the samples to decode;
* 48 observed frames and 16 future frames form a 65-frame Wan clip after one
  duplicated lead frame required by the causal 4x temporal VAE.

Wan refines a context-only analytic initializer with SDEdit.  There is no MLP,
parameter fitting, or benchmark-label access.  The unchanged initializer is
always reported as the video-off control.
"""

import argparse
import gc
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import run_mlp_adapter as windowing
import run_pilot as rp
from run_reference_baselines import visionts_predict


MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
PROMPTS = {
    "stripes": (
        "fixed camera close-up of a clean grayscale strip chart, vertical gray "
        "stripes translating steadily from right to left as new stripes enter "
        "at the right edge, smooth continuous motion, no text"
    ),
    "area": (
        "fixed camera close-up of a clean monochrome filled area graph scrolling "
        "steadily from right to left as the graph continues at the right edge, "
        "smooth continuous boundary motion, no axes, no text"
    ),
    "cycle_bands": (
        "fixed camera abstract grayscale pattern of horizontal bands, each band "
        "smoothly changing its brightness over time, minimal clean texture, no "
        "objects, no text, no camera motion"
    ),
    "cycle_area": (
        "fixed camera clean monochrome filled waveform, the same waveform "
        "smoothly changing shape over time, no axes, no text, no camera motion"
    ),
}
NEGATIVE_PROMPT = (
    "camera motion, cuts, objects, people, text, color, flicker, deformation"
)


def metrics(prediction, target):
    difference = prediction - target
    return {
        "MSE": float(np.mean(difference ** 2)),
        "MAE": float(np.mean(np.abs(difference))),
    }


def level_anchored_seasonal(x, period, horizon, offset_weight=0.6):
    """Phase mean plus a validation-locked fraction of the latest level offset."""
    complete_cycles = x.shape[1] // period
    if complete_cycles < 1:
        raise ValueError("context must contain at least one complete period")
    cycles = x[:, -complete_cycles * period:].reshape(
        len(x), complete_cycles, period
    )
    phase_mean = cycles.mean(1)
    level_offset = (cycles[:, -1] - phase_mean).mean(1, keepdims=True)
    template = phase_mean + offset_weight * level_offset
    repeats = (horizon + period - 1) // period
    return np.tile(template, (1, repeats))[:, :horizon]


def rolling_values(
    x,
    future_initializer,
    period,
    context_frames=48,
    future_frames=16,
    display_window=None,
):
    """Return causal rolling windows ``[B, 65, display_window]``.

    The final ``future_frames`` windows contain initializer values only.  The
    known portion is therefore invariant to any replacement of that initializer.
    """
    if future_initializer.shape[1] % future_frames:
        raise ValueError("horizon must be divisible by the number of future frames")
    step = future_initializer.shape[1] // future_frames
    display_window = display_window or max(period, 4 * step)
    if display_window < step:
        raise ValueError("display window must hold at least one generated chunk")

    combined = torch.cat((x, future_initializer), dim=1)
    context_length = x.shape[1]
    endpoints = [
        context_length - (context_frames - 1 - index) * step
        for index in range(context_frames)
    ] + [
        context_length + (index + 1) * step for index in range(future_frames)
    ]
    left_padding = max(0, display_window - min(endpoints))
    padded = torch.cat(
        (x[:, :1].expand(-1, left_padding), combined), dim=1
    )
    windows = []
    for endpoint in endpoints:
        adjusted = endpoint + left_padding
        windows.append(padded[:, adjusted - display_window:adjusted])
    windows = torch.stack(windows, dim=1)
    # Wan's causal VAE maps 4k+1 pixels frames to k+1 latent frames.  The lead
    # duplicate makes the context/future boundary land exactly between latents.
    windows = torch.cat((windows[:, :1], windows), dim=1)
    return windows, step, display_window, 1 + context_frames


def render_stripe_video(
    x,
    future_initializer,
    period,
    height,
    width,
    context_frames=48,
    future_frames=16,
    display_window=None,
    value_limit=4.0,
    render_mode="stripes",
):
    """Render rolling values as stripes or a filled area in ``[-1, 1]``."""
    if render_mode not in PROMPTS:
        raise ValueError(f"unknown render mode: {render_mode}")
    x = x.float()
    future_initializer = future_initializer.float()
    mean = x.mean(1, keepdim=True)
    scale = x.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
    windows, step, display_window, known_frames = rolling_values(
        x,
        future_initializer,
        period,
        context_frames,
        future_frames,
        display_window,
    )
    normalized = ((windows - mean[:, None]) / (value_limit * scale[:, None]))
    gray = ((normalized.clamp(-1, 1) + 1) / 2)
    if render_mode == "stripes":
        gray = gray.reshape(-1, 1, 1, display_window)
        # Nearest expansion keeps each sample distinct and exactly decodable.
        frames = F.interpolate(gray, size=(height, width), mode="nearest-exact")
    else:
        boundary = F.interpolate(
            gray.reshape(-1, 1, 1, display_window),
            size=(1, width),
            mode="nearest-exact",
        )
        rows = torch.linspace(
            0, 1, height, device=x.device, dtype=x.dtype
        ).reshape(1, 1, height, 1)
        # Column area equals the encoded value while the boundary is a visible
        # object trajectory for the video model to continue.
        frames = torch.sigmoid((boundary - (1 - rows)) * height / 1.5)
    frames = frames.reshape(len(x), windows.shape[1], 1, height, width)
    video = frames.repeat(1, 1, 3, 1, 1).permute(0, 2, 1, 3, 4)
    return video * 2 - 1, mean, scale, step, display_window, known_frames


def decode_stripe_video(
    video,
    mean,
    scale,
    step,
    display_window,
    known_frames,
    value_limit=4.0,
):
    """Decode the newly entered right-edge samples from each future frame."""
    gray = ((video.float() + 1) / 2).clamp(0, 1).mean(1).mean(2)
    stripes = F.adaptive_avg_pool1d(gray, display_window)
    chunks = stripes[:, known_frames:, -step:]
    normalized = 2 * chunks.reshape(len(video), -1) - 1
    return normalized.cpu() * value_limit * scale + mean


def render_cycle_video(
    x,
    future_initializer,
    period,
    height,
    width,
    context_frames=48,
    future_frames=16,
    value_limit=4.0,
    render_mode="cycle_bands",
):
    """Render each frame as one seasonal-cycle state evolving in video time."""
    if render_mode not in {"cycle_bands", "cycle_area"}:
        raise ValueError(f"unknown cycle render mode: {render_mode}")
    x = x.float()
    future_initializer = future_initializer.float()
    mean = x.mean(1, keepdim=True)
    scale = x.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
    complete_cycles = x.shape[1] // period
    context_cycles = x[:, -complete_cycles * period:].reshape(
        len(x), complete_cycles, period
    )
    context_states = F.interpolate(
        context_cycles.permute(0, 2, 1),
        size=context_frames,
        mode="nearest-exact",
    ).permute(0, 2, 1)

    future_cycle_count = (future_initializer.shape[1] + period - 1) // period
    padded_length = future_cycle_count * period
    if padded_length > future_initializer.shape[1]:
        padding = padded_length - future_initializer.shape[1]
        future_initializer = torch.cat(
            (future_initializer, future_initializer[:, :padding]), dim=1
        )
    future_cycles = future_initializer.reshape(
        len(x), future_cycle_count, period
    )
    if future_cycle_count <= future_frames:
        future_states = F.interpolate(
            future_cycles.permute(0, 2, 1),
            size=future_frames,
            mode="nearest-exact",
        ).permute(0, 2, 1)
    else:
        future_states = F.interpolate(
            future_cycles.permute(0, 2, 1),
            size=future_frames,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)
    states = torch.cat(
        (context_states[:, :1], context_states, future_states), dim=1
    )
    normalized = ((states - mean[:, None]) / (value_limit * scale[:, None]))
    gray = (normalized.clamp(-1, 1) + 1) / 2
    if render_mode == "cycle_bands":
        frames = F.interpolate(
            gray.reshape(-1, 1, period, 1),
            size=(height, width),
            mode="nearest-exact",
        )
    else:
        boundary = F.interpolate(
            gray.reshape(-1, 1, 1, period),
            size=(1, width),
            mode="nearest-exact",
        )
        rows = torch.linspace(
            0, 1, height, device=x.device, dtype=x.dtype
        ).reshape(1, 1, height, 1)
        frames = torch.sigmoid((boundary - (1 - rows)) * height / 1.5)
    frames = frames.reshape(len(x), states.shape[1], 1, height, width)
    video = frames.repeat(1, 1, 3, 1, 1).permute(0, 2, 1, 3, 4)
    return video * 2 - 1, mean, scale, 1 + context_frames, future_cycle_count


def decode_cycle_video(
    video,
    mean,
    scale,
    period,
    horizon,
    known_frames,
    future_cycle_count,
    value_limit=4.0,
    render_mode="cycle_bands",
):
    """Invert future cycle states and resample them to the numeric horizon."""
    gray = ((video.float() + 1) / 2).clamp(0, 1).mean(1)
    if render_mode == "cycle_bands":
        phase = F.adaptive_avg_pool1d(gray.mean(-1), period)
    elif render_mode == "cycle_area":
        phase = F.adaptive_avg_pool1d(gray.mean(-2), period)
    else:
        raise ValueError(f"unknown cycle render mode: {render_mode}")
    future_states = phase[:, known_frames:]
    if future_cycle_count <= future_states.shape[1]:
        cycles = F.interpolate(
            future_states.permute(0, 2, 1),
            size=future_cycle_count,
            mode="nearest-exact",
        ).permute(0, 2, 1)
    else:
        cycles = F.interpolate(
            future_states.permute(0, 2, 1),
            size=future_cycle_count,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)
    normalized = 2 * cycles.reshape(len(video), -1)[:, :horizon] - 1
    return normalized.cpu() * value_limit * scale + mean


@torch.no_grad()
def forecast_with_wan(x, initializer, period, args):
    from diffusers import WanPipeline
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    if not torch.cuda.is_available():
        raise RuntimeError("Wan rolling forecast requires CUDA")
    torch.cuda.set_per_process_memory_fraction(args.max_gpu_fraction)
    torch.cuda.reset_peak_memory_stats()

    pipeline = WanPipeline.from_pretrained(
        MODEL,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler(shift=args.shift)
    # Never place the large text encoder, video transformer, and VAE on the GPU
    # together.  This staged transfer is the main guard against the prior OOM.
    pipeline.text_encoder.to("cuda")
    prompt, negative = pipeline.encode_prompt(
        prompt=PROMPTS[args.render_mode],
        negative_prompt=NEGATIVE_PROMPT,
        do_classifier_free_guidance=args.guidance > 1,
        device="cuda",
    )
    pipeline.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    pipeline.transformer.to("cuda")
    pipeline.vae.to("cuda")

    vae = pipeline.vae
    transformer = pipeline.transformer
    scheduler = pipeline.scheduler
    latent_mean = torch.tensor(vae.config.latents_mean).view(
        1, vae.config.z_dim, 1, 1, 1
    ).to("cuda", torch.bfloat16)
    latent_std = torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1
    ).to("cuda", torch.bfloat16)

    corrected_outputs = []
    raw_outputs = []
    roundtrip_outputs = []
    context_reconstruction = []
    vae_roundtrip_error = []
    generator = torch.Generator(device="cuda").manual_seed(args.generation_seed)
    for start in range(0, len(x), args.batch_size):
        xb = torch.from_numpy(x[start:start + args.batch_size]).float()
        init = torch.from_numpy(
            initializer[start:start + args.batch_size]
        ).float()
        if args.render_mode.startswith("cycle_"):
            rendered = render_cycle_video(
                xb,
                init,
                period,
                args.height,
                args.width,
                args.context_frames,
                args.future_frames,
                args.value_limit,
                args.render_mode,
            )
            video, mean, scale, known_frames, future_cycle_count = rendered

            def decode(rendered_video):
                return decode_cycle_video(
                    rendered_video,
                    mean,
                    scale,
                    period,
                    init.shape[1],
                    known_frames,
                    future_cycle_count,
                    args.value_limit,
                    args.render_mode,
                )
        else:
            rendered = render_stripe_video(
                xb,
                init,
                period,
                args.height,
                args.width,
                args.context_frames,
                args.future_frames,
                args.display_window or None,
                args.value_limit,
                args.render_mode,
            )
            video, mean, scale, step, display_window, known_frames = rendered

            def decode(rendered_video):
                return decode_stripe_video(
                    rendered_video,
                    mean,
                    scale,
                    step,
                    display_window,
                    known_frames,
                    args.value_limit,
                )
        ground_latent = vae.encode(
            video.to("cuda", torch.bfloat16)
        ).latent_dist.mode()
        ground_latent = ((ground_latent - latent_mean) / latent_std).float()
        known_latents = 1 + (known_frames - 1) // 4

        # Decode the untouched initializer through the same lossy VAE.  Wan's
        # useful contribution is the diffusion change relative to this exact
        # round trip, not the VAE's brightness/smoothing bias.
        roundtrip_video = vae.decode(
            ground_latent.to(torch.bfloat16) * latent_std + latent_mean,
            return_dict=False,
        )[0]
        roundtrip_forecast = decode(roundtrip_video)
        roundtrip_outputs.append(roundtrip_forecast.numpy())
        vae_roundtrip_error.append(
            float(F.mse_loss(roundtrip_forecast, init).item())
        )
        del roundtrip_video

        scheduler.set_timesteps(args.steps, device="cuda")
        sigmas = scheduler.sigmas
        noise = torch.randn(
            ground_latent.shape,
            generator=generator,
            device="cuda",
            dtype=ground_latent.dtype,
        )
        start_index = next(
            (
                index
                for index, sigma in enumerate(sigmas[:-1])
                if float(sigma) <= args.sdedit
            ),
            len(sigmas) - 2,
        )
        latent = (
            (1 - sigmas[start_index]) * ground_latent
            + sigmas[start_index] * noise
        )
        for index, timestep in list(enumerate(scheduler.timesteps))[start_index:]:
            sigma = sigmas[index]
            known_noise = torch.randn(
                ground_latent.shape,
                generator=generator,
                device="cuda",
                dtype=ground_latent.dtype,
            )
            known = (1 - sigma) * ground_latent + sigma * known_noise
            latent[:, :, :known_latents] = known[:, :, :known_latents]
            prediction = transformer(
                hidden_states=latent.to(torch.bfloat16),
                timestep=timestep.expand(len(xb)),
                encoder_hidden_states=prompt,
                return_dict=False,
            )[0]
            if args.guidance > 1:
                unconditional = transformer(
                    hidden_states=latent.to(torch.bfloat16),
                    timestep=timestep.expand(len(xb)),
                    encoder_hidden_states=negative,
                    return_dict=False,
                )[0]
                prediction = unconditional + args.guidance * (
                    prediction - unconditional
                )
            latent = scheduler.step(
                prediction.float(), timestep, latent, return_dict=False
            )[0]
        latent[:, :, :known_latents] = ground_latent[:, :, :known_latents]
        decoded = vae.decode(
            latent.to(torch.bfloat16) * latent_std + latent_mean,
            return_dict=False,
        )[0]
        raw_forecast = decode(decoded)
        raw_outputs.append(raw_forecast.numpy())
        corrected_outputs.append(
            (init + raw_forecast - roundtrip_forecast).numpy()
        )
        input_gray = video[:, :, :known_frames].mean(1)
        output_gray = decoded[:, :, :known_frames].float().cpu().mean(1)
        context_reconstruction.append(
            float(F.mse_loss(output_gray, input_gray).item())
        )
        print(f"[wan] {min(start + len(xb), len(x))}/{len(x)}", flush=True)

    peak = float(torch.cuda.max_memory_allocated() / 2**30)
    pipeline.transformer.to("cpu")
    pipeline.vae.to("cpu")
    del pipeline, vae, transformer
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "bias_corrected": np.concatenate(corrected_outputs),
        "raw_decoded": np.concatenate(raw_outputs),
        "vae_roundtrip": np.concatenate(roundtrip_outputs),
    }, {
        "max_cuda_gib": peak,
        "context_reconstruction_mse": float(np.mean(context_reconstruction)),
        "future_vae_roundtrip_mse": float(np.mean(vae_roundtrip_error)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(rp.DATASETS))
    parser.add_argument("--data-dir", default="pilot/data")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--context", type=int, default=600)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--cap", type=int, default=8)
    parser.add_argument("--window-seed", type=int, default=125)
    parser.add_argument("--offset-weight", type=float, default=0.6)
    parser.add_argument("--video-weight", type=float, default=1.0)
    parser.add_argument(
        "--report-video-weights", default="0.1,0.25,0.5,1.0",
        help="comma-separated fixed blends evaluated from the same Wan output",
    )
    parser.add_argument("--context-frames", type=int, default=48)
    parser.add_argument("--future-frames", type=int, default=16)
    parser.add_argument("--display-window", type=int, default=0)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--value-limit", type=float, default=4.0)
    parser.add_argument(
        "--render-mode", choices=tuple(PROMPTS), default="area"
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--sdedit", type=float, default=0.6)
    parser.add_argument("--generation-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-gpu-fraction", type=float, default=0.55)
    parser.add_argument("--compare-visionts", action="store_true")
    parser.add_argument("--out-dir", default="pilot/results_wan_rolling")
    args = parser.parse_args()

    if args.horizon % args.future_frames:
        raise ValueError("horizon must be divisible by future-frames")
    if not 1 <= args.batch_size <= 4:
        raise ValueError("batch-size must be between 1 and 4 (8 OOMs the 24 GiB GPU)")
    if not 0 < args.max_gpu_fraction <= 0.8:
        raise ValueError("max-gpu-fraction must be in (0, 0.8]")
    data, _ = rp.load_dataset(args.dataset, args.data_dir)
    x, y, total = windowing.get_windows(
        data,
        args.dataset,
        args.context,
        args.horizon,
        args.split,
        args.stride,
        args.cap,
        args.window_seed,
    )
    initializer = level_anchored_seasonal(
        x, rp.P, args.horizon, args.offset_weight
    )
    results = {
        "video_off_initializer": metrics(initializer, y),
    }
    predictions, diagnostic = forecast_with_wan(x, initializer, rp.P, args)
    corrected_prediction = predictions["bias_corrected"]
    weights = sorted(
        set(
            [args.video_weight]
            + [
                float(value)
                for value in args.report_video_weights.split(",")
                if value.strip()
            ]
        )
    )
    for weight in weights:
        prediction = initializer + weight * (
            corrected_prediction - initializer
        )
        results[f"wan_rolling_weight_{weight:g}"] = metrics(prediction, y)
        raw_blend = initializer + weight * (
            predictions["raw_decoded"] - initializer
        )
        results[f"wan_raw_weight_{weight:g}"] = metrics(raw_blend, y)
        vae_blend = initializer + weight * (
            predictions["vae_roundtrip"] - initializer
        )
        results[f"video_vae_weight_{weight:g}"] = metrics(vae_blend, y)
    results["video_vae_roundtrip"] = metrics(
        predictions["vae_roundtrip"], y
    )
    results["wan_raw_decoded"] = metrics(predictions["raw_decoded"], y)
    results["wan_rolling_zero_shot"] = results[
        f"wan_rolling_weight_{args.video_weight:g}"
    ]
    if args.compare_visionts:
        reference = visionts_predict(
            x, args.context, args.horizon, rp.P, batch_size=128
        )
        results["visionts_zero_shot"] = metrics(reference, y)
    payload = {
        "config": {
            **vars(args),
            "periodicity": rp.P,
            "windows": len(x),
            "candidate_windows": total,
            "trainable_parameters": 0,
            "test_used_for_selection": False,
        },
        "results": results,
        "diagnostic": diagnostic,
    }
    print(json.dumps(payload, indent=2), flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(
        args.out_dir,
        f"rolling_{args.dataset}_{args.split}_c{args.context}_h{args.horizon}_n{len(x)}.json",
    )
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[results] {path}", flush=True)


if __name__ == "__main__":
    main()
