"""Regression checks for forecasting split and rendering causality.

Run from the repository root:
    python pilot/test_no_lookahead.py
"""

import os
import tempfile

import numpy as np
import pandas as pd
import torch

import pretrain_vmae_ts as pretrain
import run_finetune as finetune
import run_lagged_lora as lagged
import run_longctx as longctx
import run_mlp_adapter as adapter
import run_pilot as pilot


def assert_window_bundle(x, y, context, horizon):
    assert x.shape[1] == context
    assert y.shape[1] == horizon
    # Synthetic source below is value[t, channel] = 10*t + channel.
    np.testing.assert_allclose(y[:, 0] - x[:, -1], 10.0)


def test_train_only_dataset_normalization():
    with tempfile.TemporaryDirectory() as directory:
        length = pilot.DATASETS["ETTh1"]["borders"][2]
        train_end = pilot.DATASETS["ETTh1"]["borders"][0]
        value = np.arange(length, dtype=np.float32) % 17
        value[train_end:] += 10000
        frame = pd.DataFrame({"date": np.arange(length), "value": value})
        frame.to_csv(os.path.join(directory, "ETTh1.csv"), index=False)
        normalized, _ = pilot.load_dataset("ETTh1", directory)
        assert abs(float(normalized[:train_end].mean())) < 1e-5
        assert float(normalized[train_end:].mean()) > 100


def test_custom_csv_uses_train_only_normalization():
    with tempfile.TemporaryDirectory() as directory:
        total = 100
        train_end = int(0.7 * total)
        value = np.arange(total, dtype=np.float32)
        value[train_end:] += 10000
        frame = pd.DataFrame({"date": np.arange(total), "OT": value})
        frame.to_csv(os.path.join(directory, "weather.csv"), index=False)
        normalized, borders = pilot.load_dataset("weather", directory)
        assert borders == (int(0.8 * total), total)
        assert abs(float(normalized[:train_end].mean())) < 1e-5
        assert float(normalized[train_end:].mean()) > 100


def test_window_boundaries():
    train_end, validation_end, test_end = pilot.DATASETS["ETTh1"]["borders"]
    time = np.arange(test_end, dtype=np.float32)[:, None]
    data = 10 * time + np.arange(2, dtype=np.float32)[None, :]
    context, horizon = 120, 48

    for split, expected_start, expected_end in (
        ("train", context, train_end),
        ("val", train_end, validation_end),
        ("test", validation_end, test_end),
    ):
        x, y, _ = adapter.get_windows(
            data, "ETTh1", context, horizon, split, 37, 0, 0
        )
        assert_window_bundle(x, y, context, horizon)
        target_times = y[:, 0] // 10
        assert target_times.min() >= expected_start
        assert (target_times + horizon).max() <= expected_end

    for split, expected_start, expected_end in (
        ("train", context, train_end),
        ("test", validation_end, test_end),
    ):
        x, y, _ = lagged.get_windows(
            data, "ETTh1", context, horizon, split, 37, 0, 0
        )
        assert_window_bundle(x, y, context, horizon)
        target_times = y[:, 0] // 10
        assert target_times.min() >= expected_start
        assert (target_times + horizon).max() <= expected_end

        x, y = longctx.get_split_windows(
            data, "ETTh1", context, horizon, split, 37
        )
        assert_window_bundle(x, y, context, horizon)
        target_times = y[:, 0] // 10
        assert target_times.min() >= expected_start
        assert (target_times + horizon).max() <= expected_end

    pilot.configure(24)
    x, y = finetune.get_train_windows(data, "ETTh1")
    assert_window_bundle(x, y, pilot.CONTEXT, pilot.HORIZON)
    target_times = y[:, 0] // 10
    assert target_times.min() >= pilot.CONTEXT
    assert (target_times + pilot.HORIZON).max() <= train_end

    x, y = pilot.get_windows(data, (validation_end, test_end))
    assert_window_bundle(x, y, pilot.CONTEXT, pilot.HORIZON)
    target_times = y[:, 0] // 10
    assert target_times.min() >= validation_end
    assert (target_times + pilot.HORIZON).max() <= test_end


def test_custom_dataset_window_boundaries():
    total = 1000
    train_end, validation_end, test_end = 700, 800, 1000
    time = np.arange(total, dtype=np.float32)[:, None]
    data = 10 * time + np.arange(2, dtype=np.float32)[None, :]
    context, horizon = 120, 48
    for split, expected_start, expected_end in (
        ("train", context, train_end),
        ("val", train_end, validation_end),
        ("test", validation_end, test_end),
    ):
        x, y, _ = adapter.get_windows(
            data, "weather", context, horizon, split, 17, 0, 0
        )
        assert_window_bundle(x, y, context, horizon)
        target_times = y[:, 0] // 10
        assert target_times.min() >= expected_start
        assert (target_times + horizon).max() <= expected_end


def test_forecast_scaling_is_future_invariant():
    rng = np.random.default_rng(7)
    context = rng.normal(size=400).astype(np.float32)
    future_a = rng.normal(size=80).astype(np.float32)
    future_b = (1000 + 100 * rng.normal(size=80)).astype(np.float32)
    series_a = np.concatenate((context, future_a))
    series_b = np.concatenate((context, future_b))
    gray_a = pretrain.to_gray(series_a, context)
    gray_b = pretrain.to_gray(series_b, context)
    np.testing.assert_array_equal(gray_a[:len(context)], gray_b[:len(context)])


def test_general_phase_tokenizer_is_context_only():
    torch.manual_seed(13)
    for period in (24, 96, 144):
        tokenizer = adapter.PhaseTubeletTokenizer(600, period)
        context = torch.randn(2, 600)
        tokens_a, mean_a, scale_a = tokenizer(context)
        # A target tensor is intentionally constructed and mutated, but the
        # tokenizer receives only the identical observed slice.
        joined_a = torch.cat((context, torch.randn(2, 192)), dim=1)
        joined_b = joined_a.clone()
        joined_b[:, 600:] += 10000
        tokens_b, mean_b, scale_b = tokenizer(joined_b[:, :600])
        torch.testing.assert_close(tokens_a, tokens_b, rtol=0, atol=0)
        torch.testing.assert_close(mean_a, mean_b, rtol=0, atol=0)
        torch.testing.assert_close(scale_a, scale_b, rtol=0, atol=0)
        assert tokens_a.shape == (2, 8 * 14 * 14, 6)


def test_forecast_masks_cover_every_future_token():
    for hp in range(1, 9):
        mask = pretrain.forecast_mask_lc(hp).view(
            pretrain.TT, pretrain.GH, pretrain.GH
        )
        context_periods = pretrain.LC_STEPS * pretrain.COLS - hp
        for tubelet in range(pretrain.TT):
            for column in range(pretrain.GH):
                source_period = tubelet * pretrain.COLS + column
                if source_period >= context_periods:
                    assert bool(mask[tubelet, :, column].all())

    total_periods = (pretrain.NF - 1) * pretrain.SCROLL + pretrain.COLS
    for hp in range(1, 5):
        mask = pretrain.forecast_mask_scroll(hp).view(
            pretrain.TT, pretrain.GH, pretrain.GH
        )
        context_periods = total_periods - hp
        for tubelet in range(pretrain.TT):
            for column in range(pretrain.GH):
                frames = (2 * tubelet, 2 * tubelet + 1)
                sources = [frame * pretrain.SCROLL + column for frame in frames]
                if any(source >= context_periods for source in sources):
                    assert bool(mask[tubelet, :, column].all())


def main():
    tests = (
        test_train_only_dataset_normalization,
        test_custom_csv_uses_train_only_normalization,
        test_window_boundaries,
        test_custom_dataset_window_boundaries,
        test_forecast_scaling_is_future_invariant,
        test_general_phase_tokenizer_is_context_only,
        test_forecast_masks_cover_every_future_token,
    )
    for test in tests:
        test()
        print(f"[pass] {test.__name__}")
    print(f"[pass] {len(tests)} no-look-ahead checks")


if __name__ == "__main__":
    main()
