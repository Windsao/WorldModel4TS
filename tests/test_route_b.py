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


@pytest.mark.skipif(not _hf_ok(), reason="transformers/VideoMAE not importable here")
def test_spatial_attention_equals_per_tubelet_encoding():
    """with the block bias, an encoder layer on the full visible sequence must equal running the
    same layer on each tubelet's tokens separately; with no bias it must equal the stock forward."""
    model = B.fresh_model(0).eval()
    mask = B.forecast_mask(4, lead_tubelets=1)                            # tubelets 1..5 visible
    rows = torch.zeros(1, B.NF, B.IMG, dtype=torch.int16) + 100
    vid = B.rows_to_video(rows, "cpu")
    with torch.no_grad():
        ref = model(pixel_values=vid, bool_masked_pos=mask[None]).logits.clone()
        B.apply_spatial_attention(model)
        same = model(pixel_values=vid, bool_masked_pos=mask[None]).logits    # bias None -> stock
        assert torch.allclose(ref, same, atol=1e-5)
        B.set_attn_bias(model, mask)
        emb = model.videomae.embeddings(vid, mask[None])                    # [1, n_vis, 768]
        layer = model.videomae.encoder.layer[0]
        full = layer(emb)[0]
        idx = torch.nonzero(~mask).flatten() // (B.GH * B.GH)
        layer.attention.attention._attn_bias = None
        parts = torch.cat([layer(emb[:, idx == t])[0] for t in idx.unique()], 1)
        assert torch.allclose(full, parts, atol=1e-4), (full - parts).abs().max()


# ------------------------------------------------------------------ novel pretraining designs (2026-09-10)
def test_value_loss_zero_when_prediction_equals_label_and_recovers_z():
    """the value-space loss is 0 for a perfect prediction and its column heights decode the rendered z."""
    rng = np.random.default_rng(11)
    P, hp = 24, 4
    y = rng.normal(0, 1, B.NF * P).astype(np.float32)
    rows, mu, sd = B.z_rows(y, P, ctx_periods=B.NF - hp)
    vid = B.rows_to_video(torch.from_numpy(rows)[None], "cpu")
    mask = B.forecast_mask(hp)[None]
    labels = B.U.patchify(B.U.unnormalize(vid)).reshape(1, B.N_TOK, -1)[mask].view(1, -1, B.TS * B.PS * B.PS * 3)
    nf = B.n_future_tokens(hp)
    fut = labels[:, -nf:]
    assert float(B.value_loss(fut, fut, hp)) < 1e-8
    h = B.column_heights(B.future_frames_from_tokens(fut, hp))              # [1, hp, 224]
    z = ((h - 0.5) / B.BAND * B.ZMAX)[0].numpy()
    z_true = np.clip((y - mu) / sd, -3, 3).reshape(B.NF, P)[-hp:][:, B.col2phase(P)]
    assert np.max(np.abs(z - z_true)) < 0.03
    # a wrong prediction (future frames shifted up by 20 rows) costs about (20/223 * 3/0.4)^2 in z units
    rows2 = rows.copy(); rows2[-hp:] = np.clip(rows2[-hp:] - 20, 0, B.IMG - 1)
    vid2 = B.rows_to_video(torch.from_numpy(rows2)[None], "cpu")
    lab2 = B.U.patchify(B.U.unnormalize(vid2)).reshape(1, B.N_TOK, -1)[mask].view(1, -1, B.TS * B.PS * B.PS * 3)
    v = float(B.value_loss(lab2[:, -nf:], fut, hp))
    expect = (20 / (B.IMG - 1) * B.ZMAX / B.BAND) ** 2
    assert abs(v - expect) / expect < 0.15, (v, expect)


def test_value_loss_is_differentiable_and_masked_loss_parts_add_up():
    model = B.fresh_model(0).eval()
    rng = np.random.default_rng(12)
    P, hp = 12, 4
    y = rng.normal(0, 1, B.NF * P).astype(np.float32)
    rows, _, _ = B.z_rows(y, P, ctx_periods=B.NF - hp)
    vid = B.rows_to_video(torch.from_numpy(rows)[None], "cpu")
    mask = B.forecast_mask(hp)[None]
    total, pix, val = B.masked_loss(model, vid, mask, hp, value_w=0.1, return_parts=True)
    assert abs(float(total) - (float(pix) + 0.1 * float(val))) < 1e-6 and float(val) > 0
    total.backward()
    g = model.decoder.head.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    # value_w = 0 reproduces the old scalar behaviour
    with torch.no_grad():
        old = B.masked_loss(model, vid, mask, hp)
    assert abs(float(old) - float(pix)) < 1e-6


def test_scale_aug_frame_is_k_periods():
    """with frame = k periods the rendered rows still have shape [16, 224] and phase k*P maps to 224 columns;
    the k-period frame of a P-periodic signal shows k full cycles."""
    P, k = 24, 4
    t = np.arange(B.NF * k * P)
    y = np.sin(2 * np.pi * t / P).astype(np.float32)
    rows, _, _ = B.z_rows(y, k * P, ctx_periods=12)
    assert rows.shape == (B.NF, B.IMG)
    c2p = B.col2phase(k * P)
    assert c2p.max() == k * P - 1 and len(np.unique(c2p)) == k * P
    r0 = rows[0].astype(float)
    # k cycles per frame -> the row profile repeats every 224/k columns
    seg = B.IMG // k
    assert np.abs(r0[:seg] - r0[seg:2 * seg]).mean() < 2.0


def test_stream_scale_aug_and_hp_max(tmp_path):
    """the stream honours hp_max (masks up to hp_max/2 tubelets) and scale_aug (multi-period frames) using synthetic data only."""
    st = B.Stream(str(tmp_path), batch=4, seed=3, synth_frac=1.0, p_forecast=1.0, p_lead=0.0, scale_aug=True, hp_max=8)
    corpus_stub = type("C", (), {"sample": lambda self, rng, n_p=B.NF: (None, None)})()
    B.Corpus = lambda *a, **k: corpus_stub                                # no corpus on disk
    it = iter(st)
    hps = set()
    for _ in range(12):
        rows, mask, hp, lead = next(it)
        assert rows.shape == (4, B.NF, B.IMG) and mask.shape == (4, B.N_TOK)
        assert hp in (2, 4, 6, 8) and lead == 0
        assert mask[0].reshape(B.TT, -1)[-hp // B.TS:].all() and not mask[0].reshape(B.TT, -1)[:B.TT - hp // B.TS].any()
        hps.add(hp)
    assert len(hps) >= 3


def test_forecast_mask_hp8_and_n_future_tokens():
    m = B.forecast_mask(8).reshape(B.TT, -1)
    assert m[-4:].all() and not m[:-4].any()
    assert B.n_future_tokens(8) == 4 * B.GH * B.GH


def test_prior_anchor_pulls_toward_initial_weights():
    """decoupled decay toward theta0: p <- p - lr*l2sp*(p - p0) moves every parameter toward its initial value."""
    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 8)
    theta0 = [p.detach().clone() for p in lin.parameters()]
    with torch.no_grad():
        for p in lin.parameters():
            p.add_(torch.randn_like(p))
    d_before = sum(float((p - p0).norm() ** 2) for p, p0 in zip(lin.parameters(), theta0))
    lr, l2sp = 1e-1, 0.5
    with torch.no_grad():
        for p, p0 in zip(lin.parameters(), theta0):
            p.sub_(lr * l2sp * (p - p0))
    d_after = sum(float((p - p0).norm() ** 2) for p, p0 in zip(lin.parameters(), theta0))
    assert abs(d_after - d_before * (1 - lr * l2sp) ** 2) < 1e-5 * d_before


# ------------------------------------------------------------------ 32-frame clips (2026-09-10)
@pytest.fixture
def frames32():
    B.set_frames(32)
    yield
    B.set_frames(16)


def test_set_frames_updates_masks_and_tokens(frames32):
    assert (B.NF, B.TT, B.N_TOK) == (32, 16, 16 * 196)
    m = B.forecast_mask(4, lead_tubelets=2)
    assert m.numel() == B.N_TOK and m.reshape(B.TT, -1)[-2:].all() and m.reshape(B.TT, -1)[:2].all() and not m.reshape(B.TT, -1)[2:-2].any()
    assert B.n_future_tokens(4) == 2 * 196
    rows, _, _ = B.z_rows(np.random.default_rng(0).normal(size=32 * 24).astype(np.float32), 24, ctx_periods=28)
    assert rows.shape == (32, 224)


def test_fresh_model_32_frames_forward_and_value_loss(frames32):
    model = B.fresh_model(0).eval()
    assert model.config.num_frames == 32
    rng = np.random.default_rng(5); P, hp = 24, 4
    rows, _, _ = B.z_rows(rng.normal(size=B.NF * P).astype(np.float32), P, ctx_periods=B.NF - hp)
    vid = B.rows_to_video(torch.from_numpy(rows)[None], "cpu")
    assert vid.shape == (1, 32, 3, 224, 224)
    mask = B.forecast_mask(hp)[None]
    with torch.no_grad():
        total, pix, val = B.masked_loss(model, vid, mask, hp, value_w=0.1, return_parts=True)
    assert torch.isfinite(total) and float(val) > 0


def test_set_frames_restores_16():
    assert (B.NF, B.TT, B.N_TOK) == (16, 8, 1568)
