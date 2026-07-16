"""Regression checks for VisionTS rendering and static VideoMAE geometry."""

import sys

import torch

sys.path.insert(0, "pilot")

from run_visionts_static_adapter import VisionTSRenderer


class CapturedInput(RuntimeError):
    pass


def authoritative_visionts_image(x, horizon, periodicity):
    from visionts import VisionTS

    reference = VisionTS(load_ckpt=False, finetune_type="none")
    reference.update_config(
        context_len=x.shape[1], pred_len=horizon, periodicity=periodicity
    )
    captured = {}

    def stop_at_patch_embed(module, args):
        captured["image"] = args[0].detach().clone()
        raise CapturedInput

    handle = reference.vision_model.patch_embed.register_forward_pre_hook(
        stop_at_patch_embed
    )
    try:
        reference(x.unsqueeze(-1))
    except CapturedInput:
        pass
    finally:
        handle.remove()
    return captured["image"]


def test_renderer_matches_visionts():
    torch.manual_seed(7)
    x = torch.randn(2, 600)
    for periodicity in (24, 96, 144):
        for horizon in (96, 192):
            renderer = VisionTSRenderer(600, horizon, periodicity)
            actual, _, _ = renderer(x)
            expected = authoritative_visionts_image(x, horizon, periodicity)
            expected = expected[:, :1]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert renderer.num_patch_input == 4


def test_future_values_cannot_change_image_or_statistics():
    torch.manual_seed(11)
    context = torch.randn(3, 600)
    future_a = torch.randn(3, 192)
    future_b = future_a + 1000 * torch.randn_like(future_a)
    renderer = VisionTSRenderer(600, 192, 144)
    image_a, mean_a, scale_a = renderer(torch.cat((context, future_a), 1)[:, :600])
    image_b, mean_b, scale_b = renderer(torch.cat((context, future_b), 1)[:, :600])
    torch.testing.assert_close(image_a, image_b, rtol=0, atol=0)
    torch.testing.assert_close(mean_a, mean_b, rtol=0, atol=0)
    torch.testing.assert_close(scale_a, scale_b, rtol=0, atol=0)


if __name__ == "__main__":
    tests = (
        test_renderer_matches_visionts,
        test_future_values_cannot_change_image_or_statistics,
    )
    for test in tests:
        test()
        print(f"[pass] {test.__name__}")
    print(f"[pass] {len(tests)} VisionTS static-adapter checks")
