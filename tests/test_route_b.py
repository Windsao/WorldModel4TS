"""Correctness checks for pilot/pretrain_route_b.py (Route B matched continual pretraining).

CPU-only except the weight-mapping test, which needs the VisionTS MAE checkpoint and is skipped
when it is absent.
"""
import os, sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pilot"))
import pretrain_route_b as B


def test_freq_to_period():
    assert B.freq_to_period("H") == 24
    assert B.freq_to_period("30T") == 48
    assert B.freq_to_period("15T") == 96
    assert B.freq_to_period("10T") == 144
    assert B.freq_to_period("D") == 7
    assert B.freq_to_period("W-SUN") == 52
    assert B.freq_to_period("M") == 12 and B.freq_to_period("MS") == 12
    assert B.freq_to_period("Q-DEC") == 4
    assert B.freq_to_period("A-DEC") is None and B.freq_to_period("Y") is None
    assert B.freq_to_period("") is None


@pytest.mark.parametrize("P", [7, 12, 24, 52, 96, 144])
def test_render_decode_roundtrip(P):
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, B.NF * P).astype(np.float32)
    rows, mu, sd = B.z_rows(y, P, ctx_periods=12)
    assert rows.shape == (B.NF, B.IMG)
    vid = B.rows_to_video(torch.from_numpy(rows)[None], "cpu")            # [1,16,3,224,224]
    gray = (vid * B.IMN_STD + B.IMN_MEAN).mean(2)[0]                        # [16,224,224] in [0,1]
    z = B.decode_frames_gray(gray, P).numpy()
    z_true = np.clip((y - mu) / sd, -3, 3).reshape(B.NF, P)
    # one pixel row is 3/(0.4*223) = 0.034 z; rounding to a row costs at most half of that
    assert np.max(np.abs(z - z_true)) < 0.03, np.max(np.abs(z - z_true))


def test_lead_masked_context_stats_use_visible_periods_only():
    rng = np.random.default_rng(5)
    P = 24
    y = rng.normal(0, 1, B.NF * P).astype(np.float32)
    y2 = y.copy(); y2[:4 * P] += 50.0                                    # change only the hidden lead frames
    r1, mu1, sd1 = B.z_rows(y, P, ctx_periods=8, ctx_start=4)
    r2, mu2, sd2 = B.z_rows(y2, P, ctx_periods=8, ctx_start=4)
    assert mu1 == mu2 and sd1 == sd2 and np.array_equal(r1[4:], r2[4:])


def test_context_only_normalisation_no_future_leak():
    rng = np.random.default_rng(1)
    P = 24
    y = rng.normal(0, 1, B.NF * P).astype(np.float32)
    y2 = y.copy(); y2[-4 * P:] += 50.0                                   # change only the last 4 periods
    r1, mu1, sd1 = B.z_rows(y, P, ctx_periods=12)
    r2, mu2, sd2 = B.z_rows(y2, P, ctx_periods=12)
    assert mu1 == mu2 and sd1 == sd2
    assert np.array_equal(r1[:12], r2[:12])                              # context frames identical
    assert not np.array_equal(r1[12:], r2[12:])


def test_forecast_mask_covers_exactly_last_frames():
    for hp in (2, 4):
        m = B.forecast_mask(hp).view(B.TT, B.GH * B.GH)
        assert m[B.TT - hp // 2:].all() and not m[: B.TT - hp // 2].any()
        assert int(m.sum()) == (hp // 2) * B.GH * B.GH
    m = B.forecast_mask(4, lead_tubelets=2).view(B.TT, B.GH * B.GH)
    assert m[:2].all() and m[6:].all() and not m[2:6].any()
    t = B.tube_mask(np.random.default_rng(0))
    tm = t.view(B.TT, B.GH * B.GH)
    assert int(tm[0].sum()) == int(0.75 * 196) and torch.equal(tm[0], tm[7])


def test_hf_target_packing_matches_decode():
    """HF VideoMAE (norm_pix_loss=False) targets are raw [0,1] pixels packed tubelet-major; the
    forecast_z unpacking must invert that packing. Check with a tiny fake 'logits' built from
    the ground-truth pixels."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pilot"))
    import videomae_patch_utils as U
    rng = np.random.default_rng(2)
    P = 24
    y = rng.normal(0, 1, B.NF * P).astype(np.float32)
    rows, mu, sd = B.z_rows(y, P, ctx_periods=12)
    vid = B.rows_to_video(torch.from_numpy(rows)[None], "cpu")
    cubes = U.patchify(U.unnormalize(vid))                                # [1, 1568, 1536, ...]
    flat = cubes.reshape(1, B.N_TOK, -1)
    hp = 4
    n_fut = (hp // B.TS) * B.GH * B.GH
    logits = flat[:, -n_fut:]
    c = logits.view(1, hp // B.TS, B.GH, B.GH, B.TS, B.PS, B.PS, 3)
    frames = c.permute(0, 1, 4, 7, 2, 5, 3, 6).reshape(1, hp, 3, B.IMG, B.IMG)
    gray = frames.mean(2)[0]
    z = B.decode_frames_gray(gray, P).numpy()
    z_true = np.clip((y - mu) / sd, -3, 3).reshape(B.NF, P)[-hp:]
    assert np.max(np.abs(z - z_true)) < 0.03


def _hf_ok():
    try:
        import transformers  # noqa
        from transformers import VideoMAEForPreTraining  # noqa
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _hf_ok(), reason="transformers/VideoMAE not importable here")
def test_fresh_models_share_decoder_init():
    a = B.fresh_model(0).state_dict(); b = B.fresh_model(0).state_dict()
    for k in a:
        assert torch.equal(a[k], b[k])


@pytest.mark.skipif(not os.path.exists("/nyx-storage1/hanliu/wm4ts/ckpt/mae_visualize_vit_base.pth") or not _hf_ok(),
                    reason="MAE checkpoint only on the cluster")
def test_image_mae_mapping_is_complete_and_loads():
    model, info = B.build_arm("imae_enc", 0, "/nyx-storage1/hanliu/wm4ts/ckpt/mae_visualize_vit_base.pth")
    assert info["encoder_loaded"] > 0
    rnd = B.fresh_model(0).state_dict()
    sd = model.state_dict()
    enc_keys = [k for k in sd if k.startswith("videomae.")]
    changed = sum(not torch.equal(sd[k], rnd[k]) for k in enc_keys)
    assert changed == len(enc_keys), (changed, len(enc_keys))
    dec_keys = [k for k in sd if not k.startswith("videomae.")]
    assert all(torch.equal(sd[k], rnd[k]) for k in dec_keys)              # decoder untouched
    # forward runs and is finite
    rows = torch.zeros(1, B.NF, B.IMG, dtype=torch.int16) + 100
    out = model(pixel_values=B.rows_to_video(rows, "cpu"), bool_masked_pos=B.forecast_mask(2)[None])
    assert torch.isfinite(out.loss)


@pytest.mark.skipif(not _hf_ok(), reason="transformers/VideoMAE not importable here")
def test_masked_loss_uses_future_tokens_only():
    """with a lead mask, changing the hidden lead frames must not change the forecast loss."""
    model = B.fresh_model(0).eval()
    rng = np.random.default_rng(6)
    P = 24
    y = rng.normal(0, 1, B.NF * P).astype(np.float32)
    rows, _, _ = B.z_rows(y, P, ctx_periods=8, ctx_start=4)
    rows2 = rows.copy(); rows2[:4] = 200                                 # different hidden lead content
    mask = B.forecast_mask(4, lead_tubelets=2)[None]
    with torch.no_grad():
        l1 = B.masked_loss(model, B.rows_to_video(torch.from_numpy(rows)[None], "cpu"), mask, 4)
        l2 = B.masked_loss(model, B.rows_to_video(torch.from_numpy(rows2)[None], "cpu"), mask, 4)
    assert torch.isfinite(l1) and abs(float(l1) - float(l2)) < 1e-6
    # and it does depend on the future frames
    rows3 = rows.copy(); rows3[-4:] = 200
    with torch.no_grad():
        l3 = B.masked_loss(model, B.rows_to_video(torch.from_numpy(rows3)[None], "cpu"), mask, 4)
    assert abs(float(l1) - float(l3)) > 1e-4


@pytest.mark.skipif(not os.path.exists("/nyx-storage1/hanliu/wm4ts/ckpt/mae_visualize_vit_base.pth") or not _hf_ok(),
                    reason="MAE checkpoint only on the cluster")
def test_image_mae_full_arm_loads_decoder_and_inflates_head():
    ck = "/nyx-storage1/hanliu/wm4ts/ckpt/mae_visualize_vit_base.pth"
    model, info = B.build_arm("imae_full", 0, ck)
    rnd = B.fresh_model(0, decoder="mae").state_dict()
    sd = model.state_dict()
    dec_keys = [k for k in sd if not k.startswith("videomae.")]
    changed = sum(not torch.equal(sd[k], rnd[k]) for k in dec_keys)
    assert changed == len(dec_keys), (changed, len(dec_keys))
    mae = torch.load(ck, map_location="cpu"); mae = mae.get("model", mae)
    W = mae["decoder_pred.weight"]
    assert torch.equal(sd["decoder.head.weight"][:768], W) and torch.equal(sd["decoder.head.weight"][768:], W)
    assert sd["decoder.head.weight"].shape == (1536, 512) and sd["encoder_to_decoder.weight"].shape == (512, 768)
    # the three 8x512 controls share the decoder init bit-for-bit
    a = B.build_arm("vmae_enc_d8", 0, ck)[0].state_dict(); b = B.build_arm("random_d8", 0, ck)[0].state_dict()
    c = B.build_arm("imae_enc_d8", 0, ck)[0].state_dict()
    for k in dec_keys:
        assert torch.equal(a[k], b[k]) and torch.equal(a[k], c[k])
    rows = torch.zeros(1, B.NF, B.IMG, dtype=torch.int16) + 100
    out = model(pixel_values=B.rows_to_video(rows, "cpu"), bool_masked_pos=B.forecast_mask(2)[None])
    assert torch.isfinite(out.loss)
