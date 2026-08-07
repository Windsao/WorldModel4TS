#!/usr/bin/env python3
"""Reproduce the reported Wan2.1 SDEdit ETTh1/H96 run.

This is a thin launcher around ``pilot/run_wan.py``.  It fixes the inference
configuration and checkpoint revision while leaving only data/output paths
configurable.  The future initialization is the seasonal mean (``smean``),
and Wan performs the SDEdit/RePaint update beginning at sigma 0.6.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"


def parse_args() -> argparse.Namespace:
    default_data = os.environ.get(
        "WORLDMODEL4TS_DATA_DIR", str(REPO_ROOT / "pilot/data/ETT-small")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(default_data),
        help="directory containing ETTh1.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "pilot/reproduction_results/wan_sdedit_etth1",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the locked configuration without loading the model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    probe_path = out_dir / "wan_sdedit_ETTh1_probe.png"

    locked_args = [
        "--dataset", "ETTh1",
        "--data-dir", str(data_dir),
        "--out-dir", str(out_dir),
        "--subsample", "96",
        "--subsample-seed", "123",
        "--steps", "20",
        "--guidance", "1.0",
        "--shift", "3.0",
        "--height", "480",
        "--width", "832",
        "--fpp", "4",
        "--batch", "2",
        "--sdedit", "0.6",
        "--sdedit-init", "smean",
        "--prompt", "bands",
        "--save-probe", str(probe_path),
    ]
    manifest = {
        "model": MODEL_REPO,
        "revision": MODEL_REVISION,
        "entrypoint": str(REPO_ROOT / "pilot/run_wan.py"),
        "arguments": locked_args,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return

    data_file = data_dir / "ETTh1.csv"
    if not data_file.is_file():
        raise SystemExit(f"ETTh1.csv not found at {data_file}")

    from huggingface_hub import snapshot_download
    import run_wan

    model_path = snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION)
    run_wan.MODEL = model_path
    original_argv = sys.argv
    try:
        sys.argv = [str(Path(run_wan.__file__).resolve()), *locked_args]
        run_wan.main()
    finally:
        sys.argv = original_argv

    result_path = out_dir / "wan_ETTh1.json"
    result = json.loads(result_path.read_text())
    if result.get("results", {}).get("wan21_1.3b", {}).get("error"):
        raise RuntimeError(f"Wan inference failed; inspect {result_path}")


if __name__ == "__main__":
    main()
