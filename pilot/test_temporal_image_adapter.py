"""Causality and geometry checks for temporal VisionTS image views."""

import sys

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


if __name__ == "__main__":
    tests = (
        test_prefix_video_is_causal_and_latest_frame_is_full_context,
        test_dyadic_rgb_is_causal_and_aligned,
        test_dyadic_prefix_video_has_distinct_causal_frames,
        test_dyadic_scale_video_is_coarse_to_fine_and_causal,
        test_residual_dyadic_ingest_starts_as_static_grayscale,
    )
    for test in tests:
        test()
        print(f"[pass] {test.__name__}")
    print(f"[pass] {len(tests)} temporal-image checks")
