"""Evaluate causal statistical, VisionTS, and Chronos reference baselines."""

import argparse
import gc
import json
import math
import os

import numpy as np
import torch

import run_mlp_adapter as windows
import run_pilot as rp


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def metrics(prediction, target):
    difference = prediction - target
    return {
        "MSE": round(float(np.mean(difference ** 2)), 4),
        "MAE": round(float(np.mean(np.abs(difference))), 4),
    }


def visionts_predict(x, context, horizon, periodicity, batch_size):
    from visionts import VisionTS

    checkpoint = os.environ.get("VISIONTS_CKPT", "./ckpt")
    model = VisionTS(
        arch="mae_base", finetune_type="none", load_ckpt=True,
        ckpt_dir=checkpoint,
    ).to(DEVICE).eval()
    model.update_config(
        context_len=context, pred_len=horizon, periodicity=periodicity
    )
    output = []
    for start in range(0, len(x), batch_size):
        value = torch.from_numpy(x[start:start + batch_size]).to(DEVICE)
        with torch.no_grad():
            prediction = model(value.unsqueeze(-1))
        output.append(prediction.squeeze(-1).float().cpu().numpy())
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(output)


def chronos_predict(x, horizon, batch_size):
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base", device_map=DEVICE,
        torch_dtype=torch.float32,
    )
    output = []
    for start in range(0, len(x), batch_size):
        value = torch.from_numpy(x[start:start + batch_size]).float()
        _, prediction = pipeline.predict_quantiles(
            value, prediction_length=horizon, quantile_levels=[0.5]
        )
        output.append(prediction.float().cpu().numpy())
    del pipeline
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(rp.DATASETS))
    parser.add_argument("--data-dir", default="pilot/data")
    parser.add_argument("--out-dir", default="pilot/results_reference")
    parser.add_argument("--context", type=int, default=600)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--test-cap", type=int, default=1024)
    parser.add_argument("--eval-stride", type=int, default=24)
    parser.add_argument("--visionts-batch", type=int, default=128)
    parser.add_argument("--chronos-batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data, _ = rp.load_dataset(args.dataset, args.data_dir)
    periodicity = rp.P
    x, y, total = windows.get_windows(
        data, args.dataset, args.context, args.horizon, "test",
        args.eval_stride, args.test_cap, args.seed + 2,
    )
    repeats = math.ceil(args.horizon / periodicity)
    results = {
        "naive": metrics(np.repeat(x[:, -1:], args.horizon, axis=1), y),
        "seasonal_naive": metrics(
            np.tile(x[:, -periodicity:], (1, repeats))[:, :args.horizon], y
        ),
    }
    print(
        f"dataset={args.dataset} context={args.context} horizon={args.horizon} "
        f"test={len(x)}/{total} periodicity={periodicity}",
        flush=True,
    )
    prediction = visionts_predict(
        x, args.context, args.horizon, periodicity, args.visionts_batch
    )
    results["visionts_zero_shot"] = metrics(prediction, y)
    print(f"[done] visionts_zero_shot {results['visionts_zero_shot']}", flush=True)
    prediction = chronos_predict(x, args.horizon, args.chronos_batch)
    results["chronos_bolt_base_zero_shot"] = metrics(prediction, y)
    print(
        f"[done] chronos_bolt_base_zero_shot "
        f"{results['chronos_bolt_base_zero_shot']}",
        flush=True,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(
        args.out_dir, f"reference_{args.dataset}_h{args.horizon}_n{len(x)}.json"
    )
    with open(path, "w") as handle:
        json.dump(
            {
                "config": {
                    **vars(args),
                    "periodicity": periodicity,
                    "test_windows": len(x),
                    "test_candidates": total,
                    "normalization": "training-split statistics only",
                    "test_used_for_selection": False,
                },
                "results": results,
            },
            handle,
            indent=2,
        )
    print(f"[results] {path}", flush=True)


if __name__ == "__main__":
    main()
