"""G0 codec gate for Route A (VIDEO_TS_LITERATURE_AND_RESCUE_PLAN.md section 3).

Pure numpy; runs on CPU. The mp4 round trip (x264 compression) is checked separately on the
cluster by pilot/run_arvideo_g1.py --codec-only because it needs ffmpeg.
"""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pilot"))
import arvideo_render as R


@pytest.mark.parametrize("P", [12, 24, 48, 96])
def test_roundtrip_dense_z(P):
    rng = np.random.default_rng(0)
    z = rng.uniform(-2.8, 2.8, P)
    img = R.render_frame(z, 5)
    zr = R.decode_frame(img, P)
    # one pixel at S=480 over the 0.8*S band is 3.0/(0.4*480)=0.0156 z; require <= 0.02
    assert np.max(np.abs(zr - z)) <= 0.02, np.max(np.abs(zr - z))


def test_clipping_is_the_only_loss():
    z = np.array([-5.0, -3.0, 0.0, 3.0, 5.0])
    zr = R.decode_frame(R.render_frame(z, 0), 5)
    assert np.allclose(zr, np.clip(z, -3, 3), atol=0.02)


def test_grid_lines_do_not_fool_decoder():
    # a frame with everything at the bottom of the band: grid lines lie above the fill
    z = np.full(24, -3.0)
    zr = R.decode_frame(R.render_frame(z, 0), 24)
    assert np.allclose(zr, -3.0, atol=0.02)


def test_blur_and_noise_robustness():
    """simulate codec/generation degradation: gaussian blur + pixel noise + slight colour shift."""
    rng = np.random.default_rng(1)
    z = rng.uniform(-2.5, 2.5, 24)
    img = R.render_frame(z, 7).astype(np.float32)
    k = np.array([1, 4, 6, 4, 1], np.float32); k = np.outer(k, k); k /= k.sum()
    pad = 2
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    blurred = np.zeros_like(img)
    for i in range(5):
        for j in range(5):
            blurred += k[i, j] * padded[i:i + img.shape[0], j:j + img.shape[1]]
    noisy = np.clip(blurred + rng.normal(0, 6, img.shape) + 8, 0, 255).astype(np.uint8)
    zr = R.decode_frame(noisy, 24)
    # random per-phase values make sharp apexes that a horizontal blur necessarily lowers
    assert np.max(np.abs(zr - z)) <= 0.2, np.max(np.abs(zr - z))
    assert abs(R.decode_clock(noisy) - 7) <= 0.5
    # a smooth (realistic) within-period shape is decoded to well under 0.05 under the same blur
    zs = 2.0 * np.sin(2 * np.pi * np.arange(24) / 24) + 0.5 * np.cos(4 * np.pi * np.arange(24) / 24)
    img = R.render_frame(zs, 7).astype(np.float32)
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    blurred = np.zeros_like(img)
    for i in range(5):
        for j in range(5):
            blurred += k[i, j] * padded[i:i + img.shape[0], j:j + img.shape[1]]
    noisy = np.clip(blurred + rng.normal(0, 6, img.shape) + 8, 0, 255).astype(np.uint8)
    assert np.max(np.abs(R.decode_frame(noisy, 24) - zs)) <= 0.05


def test_prefix_alignment_and_no_leak():
    """last prefix frame == period G_ctx-1 exactly; no frame depends on period G_ctx."""
    rng = np.random.default_rng(2)
    P, G = 24, 12
    Z = rng.uniform(-2, 2, (G, P))
    G_ctx = R.PREFIX_FRAMES // R.M
    vid = R.render_prefix(Z[:G_ctx])
    assert vid.shape == (R.PREFIX_FRAMES, R.S, R.S, 3)
    assert np.allclose(R.decode_frame(vid[-1], P), Z[G_ctx - 1], atol=0.02)
    for g in range(G_ctx):
        assert np.allclose(R.decode_frame(vid[R.period_frame(g)], P), Z[g], atol=0.02)
    # changing the future must not change any prefix pixel
    Z2 = Z.copy(); Z2[G_ctx:] = rng.uniform(-2, 2, (G - G_ctx, P))
    vid2 = R.render_prefix(Z2[:G_ctx])
    assert np.array_equal(vid, vid2)
    with pytest.raises(ValueError):
        R.render_prefix(Z[:G_ctx + 1])


def test_full_video_period_frames_and_interpolation():
    rng = np.random.default_rng(3)
    P, G = 24, 16
    Z = rng.uniform(-2, 2, (G, P))
    vid = R.render_video(Z)
    assert vid.shape[0] == R.period_frame(G - 1) + 1
    dec = R.decode_video(vid, P)
    for g in range(G):
        assert np.allclose(dec[R.period_frame(g)], Z[g], atol=0.02)
    # midway frame between period 3 and 4 is the average (m=4 -> frac 0.5 at +2 frames)
    mid = R.period_frame(3) + 2
    assert np.allclose(dec[mid], 0.5 * (Z[3] + Z[4]), atol=0.03)


def test_synthetic_signals_context_normalised_and_distinct():
    rng = np.random.default_rng(4)
    P, G = 24, 16
    G_ctx = R.PREFIX_FRAMES // R.M
    out = {}
    for kind in ["const", "ramp", "sine_level", "travel"]:
        Z = R.synth_signal(kind, P, G, np.random.default_rng(4))
        assert Z.shape == (G, P)
        assert abs(Z[:G_ctx].std() - 1.0) < 1e-6
        out[kind] = Z
    b = R.baselines(out["const"][:G_ctx], G - G_ctx)
    assert R.mse(b["copy_last"], out["const"][G_ctx:]) < 1e-12       # const: copy is perfect
    b = R.baselines(out["ramp"][:G_ctx], G - G_ctx)
    assert R.mse(b["lin_extrap"], out["ramp"][G_ctx:]) < 1e-12       # ramp: linear is perfect
    assert R.mse(b["copy_last"], out["ramp"][G_ctx:]) > 0.1
    b = R.baselines(out["sine_level"][:G_ctx], G - G_ctx)
    assert R.mse(b["copy_last"], out["sine_level"][G_ctx:]) > 0.1     # neither trivial baseline is perfect
    assert R.mse(b["lin_extrap"], out["sine_level"][G_ctx:]) > 0.1


def test_clock_roundtrip_and_video_time_axis():
    rng = np.random.default_rng(9)
    z = rng.uniform(-2, 2, 24)
    for t in [0, 1, 31, 32, 63, 100, 150]:
        assert abs(R.decode_clock(R.render_frame(z, t)) - t) <= 0.34, t
    Z = rng.uniform(-2, 2, (16, 24))
    vid = R.render_video(Z)
    ts = R.decode_clocks(vid)
    assert np.allclose(ts, np.arange(vid.shape[0]), atol=0.34)
    # clock strip never contaminates the chart decode
    assert np.allclose(R.decode_frame(vid[3], 24), Z[0], atol=0.02)
