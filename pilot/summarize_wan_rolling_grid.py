"""Consolidate and verify the locked Wan rolling-video benchmark grid."""

import argparse
import glob
import json
import os


DATASETS = ("ETTh1", "ETTm1", "weather")
HORIZONS = (96, 192, 336, 720)


def summarize(result_dir):
    rows = {}
    for path in glob.glob(os.path.join(result_dir, "*.json")):
        with open(path) as handle:
            payload = json.load(handle)
        if "config" not in payload:
            continue
        config = payload["config"]
        key = (config["dataset"], int(config["horizon"]))
        if key not in {(dataset, horizon) for dataset in DATASETS for horizon in HORIZONS}:
            continue
        if key in rows:
            raise ValueError(f"duplicate grid cell: {key}")
        if config["split"] != "test" or config["test_used_for_selection"]:
            raise ValueError(f"invalid test protocol in {path}")
        if config["render_mode"] != "cycle_bands":
            raise ValueError(f"unlocked render mode in {path}")
        if config["offset_weight"] != 0.6 or config["video_weight"] != 1.0:
            raise ValueError(f"unlocked forecast weights in {path}")
        if config["windows"] != 32:
            raise ValueError(f"unexpected sample count in {path}")
        results = payload["results"]
        rows[key] = {
            "wan": results["wan_rolling_zero_shot"],
            "visionts": results["visionts_zero_shot"],
            "video_off_initializer": results["video_off_initializer"],
            "max_cuda_gib": payload["diagnostic"]["max_cuda_gib"],
            "source": path,
        }

    expected = {(dataset, horizon) for dataset in DATASETS for horizon in HORIZONS}
    missing = expected - set(rows)
    if missing:
        raise ValueError(f"missing grid cells: {sorted(missing)}")

    wins = sum(row["wan"]["MSE"] < row["visionts"]["MSE"] for row in rows.values())
    video_improvements = sum(
        row["wan"]["MSE"] < row["video_off_initializer"]["MSE"]
        for row in rows.values()
    )
    mean_wan = sum(row["wan"]["MSE"] for row in rows.values()) / len(rows)
    mean_visionts = sum(row["visionts"]["MSE"] for row in rows.values()) / len(rows)
    mean_initializer = sum(
        row["video_off_initializer"]["MSE"] for row in rows.values()
    ) / len(rows)
    table = {
        dataset: {
            str(horizon): rows[(dataset, horizon)] for horizon in HORIZONS
        }
        for dataset in DATASETS
    }
    return {
        "protocol": {
            "datasets": list(DATASETS),
            "horizons": list(HORIZONS),
            "context": 600,
            "windows_per_cell": 32,
            "primary_metric": "MSE",
            "selection": "mapping and global weights locked on validation",
            "trainable_parameters": 0,
        },
        "summary": {
            "wins": wins,
            "cells": len(rows),
            "win_rate": wins / len(rows),
            "mean_wan_mse": mean_wan,
            "mean_visionts_mse": mean_visionts,
            "relative_mean_mse_reduction": 1 - mean_wan / mean_visionts,
            "video_improves_initializer_cells": video_improvements,
            "mean_initializer_mse": mean_initializer,
            "max_cuda_gib": max(row["max_cuda_gib"] for row in rows.values()),
        },
        "results": table,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir", default="pilot/results_wan_rolling/test_locked_n32"
    )
    parser.add_argument(
        "--output",
        default="pilot/results_wan_rolling/test_locked_n32/grid_summary.json",
    )
    args = parser.parse_args()
    payload = summarize(args.result_dir)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["summary"], indent=2))
    print(f"[results] {args.output}")


if __name__ == "__main__":
    main()
