"""Causality and geometry checks for temporal VisionTS image views."""

import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, "pilot")

from run_temporal_image_adapter import CausalImageViews, ChannelIngestMLP


def test_prefix_video_is_causal_and_latest_frame_is_full_context():
    torch.manual_seed(17)
    context = torch.randn(2, 600)
    future_a = torch.randn(2, 192)
    future_b = future_a + 10000
    for periodicity in (24, 96, 144):
        views = CausalImageViews(600, 192, periodicity, "prefix_video", 4)
        clip_a, mean_a, scale_a = views(
            torch.cat((context, future_a), 1)[:, :600]
        )
        clip_b, mean_b, scale_b = views(
            torch.cat((context, future_b), 1)[:, :600]
        )
        torch.testing.assert_close(clip_a, clip_b, rtol=0, atol=0)
        torch.testing.assert_close(mean_a, mean_b, rtol=0, atol=0)
        torch.testing.assert_close(scale_a, scale_b, rtol=0, atol=0)
        full, _, _ = views.full_renderer(context)
        torch.testing.assert_close(clip_a[:, -1], full, rtol=0, atol=0)
        assert clip_a.shape == (2, 4, 1, 224, 224)


def test_dyadic_rgb_is_causal_and_aligned():
    torch.manual_seed(19)
    context = torch.randn(2, 600)
    for periodicity in (24, 96, 144):
        views = CausalImageViews(600, 96, periodicity, "dyadic_rgb", 4)
        clip, _, _ = views(context)
        assert clip.shape == (2, 2, 3, 224, 224)
        torch.testing.assert_close(clip[:, 0], clip[:, 1], rtol=0, atol=0)
        for channel, (stride, renderer) in enumerate(
            zip(views.scales, views.renderers[0])
        ):
            expected, _, _ = renderer(context[:, stride - 1::stride])
            torch.testing.assert_close(
                clip[:, 0, channel:channel + 1], expected, rtol=0, atol=0
            )


def test_dyadic_prefix_video_has_distinct_causal_frames():
    torch.manual_seed(23)
    context = torch.randn(2, 600)
    views = CausalImageViews(
        600, 192, 96, "dyadic_prefix_video", 2
    )
    clip, _, _ = views(context)
    assert views.endpoints == (300, 600)
    assert clip.shape == (2, 2, 3, 224, 224)
    assert not torch.equal(clip[:, 0], clip[:, 1])
    for frame, (endpoint, group) in enumerate(
        zip(views.endpoints, views.renderers)
    ):
        for channel, (stride, renderer) in enumerate(zip(views.scales, group)):
            start = (endpoint - 1) % stride
            expected, _, _ = renderer(context[:, start:endpoint:stride])
            torch.testing.assert_close(
                clip[:, frame, channel:channel + 1], expected,
                rtol=0, atol=0,
            )


def test_dyadic_scale_video_is_coarse_to_fine_and_causal():
    torch.manual_seed(29)
    context = torch.randn(2, 600)
    views = CausalImageViews(600, 192, 144, "dyadic_scale_video", 4)
    clip, _, _ = views(context)
    assert views.scales == (8, 4, 2, 1)
    assert clip.shape == (2, 4, 1, 224, 224)
    for frame, (stride, renderer) in enumerate(zip(views.scales, views.renderers)):
        start = (599 % stride)
        expected, _, _ = renderer(context[:, start::stride])
        torch.testing.assert_close(clip[:, frame], expected, rtol=0, atol=0)


def test_residual_dyadic_ingest_starts_as_static_grayscale():
    torch.manual_seed(31)
    image = torch.randn(2, 2, 3, 12, 13)
    ingest = ChannelIngestMLP(3, 32, repeat_first=True)
    actual = ingest(image)
    expected = image[:, :, :1].expand(-1, -1, 3, -1, -1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_step_shift_video_is_exact_previous_then_current():
    torch.manual_seed(47)
    context = torch.randn(2, 600)
    future_a = torch.randn(2, 192)
    future_b = future_a + 10000
    forward = CausalImageViews(600, 192, 24, "step_shift_video", 2)
    repeat = CausalImageViews(
        600, 192, 24, "step_shift_video_repeat", 2
    )
    reverse = CausalImageViews(
        600, 192, 24, "step_shift_video_reverse", 2
    )
    clip, mean, scale = forward(context)
    clip_a, _, _ = forward(torch.cat((context, future_a), 1)[:, :600])
    clip_b, _, _ = forward(torch.cat((context, future_b), 1)[:, :600])
    torch.testing.assert_close(clip, clip_a, rtol=0, atol=0)
    torch.testing.assert_close(clip, clip_b, rtol=0, atol=0)
    assert clip.shape == (2, 2, 1, 224, 224)
    previous = torch.cat((context[:, :1], context[:, :-1]), dim=1)
    expected_previous = forward.shift_renderer.render_with_statistics(
        previous, mean, scale
    )
    expected_current = forward.shift_renderer.render_with_statistics(
        context, mean, scale
    )
    torch.testing.assert_close(clip[:, 0], expected_previous, rtol=0, atol=0)
    torch.testing.assert_close(clip[:, 1], expected_current, rtol=0, atol=0)
    assert torch.mean((clip[:, 1] - clip[:, 0]) ** 2) > 0
    repeated, _, _ = repeat(context)
    reversed_clip, _, _ = reverse(context)
    torch.testing.assert_close(
        repeated, expected_current[:, None].expand_as(clip), rtol=0, atol=0
    )
    torch.testing.assert_close(reversed_clip, clip.flip(1), rtol=0, atol=0)


def test_locked_step_shift_results_are_internally_consistent():
    result_path = Path(__file__).parent / (
        "results_temporal_axis/step_shift_multiseed.json"
    )
    baseline_path = Path(__file__).parent / (
        "results_benchmark_grid/benchmark_grid.json"
    )
    payload = json.loads(result_path.read_text())
    baselines = json.loads(baseline_path.read_text())["results"]
    assert all(
        value > 0
        for horizons in payload["input_non_degeneracy"].values()
        if isinstance(horizons, dict)
        for value in horizons.values()
    )
    static_wins = 0
    residual_wins = 0

    for dataset, horizons in payload["results"].items():
        for horizon, result in horizons.items():
            test = result["test"]
            assert round(statistics.mean(test["MSE_seeds"]), 4) == test["MSE_mean"]
            assert round(statistics.stdev(test["MSE_seeds"]), 4) == test["MSE_std"]
            assert round(statistics.mean(test["MAE_seeds"]), 4) == test["MAE_mean"]
            assert round(statistics.stdev(test["MAE_seeds"]), 4) == test["MAE_std"]

            control = result["validation_counterfactual"]
            forward = control["forward_MSE_seeds"]
            repeat = control["repeat_MSE_seeds"]
            reverse = control["reverse_MSE_seeds"]
            repeat_delta = statistics.mean(
                changed - base for base, changed in zip(forward, repeat)
            )
            reverse_delta = statistics.mean(
                changed - base for base, changed in zip(forward, reverse)
            )
            assert round(repeat_delta, 4) == control["repeat_minus_forward_mean"]
            assert round(reverse_delta, 4) == control["reverse_minus_forward_mean"]
            assert repeat_delta > 0
            assert reverse_delta > 0
            assert sum(changed > base for base, changed in zip(forward, repeat)) == (
                control["forward_better_than_repeat_seeds"]
            )
            assert sum(changed > base for base, changed in zip(forward, reverse)) == (
                control["forward_better_than_reverse_seeds"]
            )

            comparison = result["comparison_MSE_mean"]
            baseline = baselines[dataset][horizon]
            assert comparison["static_image"] == baseline["static_image"]["MSE_mean"]
            assert comparison["residual_dyadic_rgb"] == (
                baseline["residual_dyadic_rgb"]["MSE_mean"]
            )
            static_wins += test["MSE_mean"] < comparison["static_image"]
            residual_wins += test["MSE_mean"] < comparison["residual_dyadic_rgb"]

    summary = payload["summary"]
    assert static_wins == summary["test_MSE_cells_better_than_static_image"] == 5
    assert residual_wins == (
        summary["test_MSE_cells_better_than_residual_dyadic_rgb"]
    ) == 1


if __name__ == "__main__":
    tests = (
        test_prefix_video_is_causal_and_latest_frame_is_full_context,
        test_dyadic_rgb_is_causal_and_aligned,
        test_dyadic_prefix_video_has_distinct_causal_frames,
        test_dyadic_scale_video_is_coarse_to_fine_and_causal,
        test_residual_dyadic_ingest_starts_as_static_grayscale,
        test_step_shift_video_is_exact_previous_then_current,
        test_locked_step_shift_results_are_internally_consistent,
    )
    for test in tests:
        test()
        print(f"[pass] {test.__name__}")
    print(f"[pass] {len(tests)} temporal-image checks")
