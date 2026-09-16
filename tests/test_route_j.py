"""Route J (V-JEPA 2 world model) correctness checks. GPU/cluster-only parts skip elsewhere."""
import os, sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pilot"))
import pretrain_route_b as B


def test_token_index_sets_split_context_future_and_drop_lead():
    import pretrain_route_j as J
    m = B.forecast_mask(4, lead_tubelets=1)
    ctx, tgt, n_fut = J.token_index_sets(m)
    assert n_fut == 2 and len(tgt) == 2 * 196 and len(ctx) == 5 * 196
    assert set(ctx.tolist()).isdisjoint(set(tgt.tolist()))
    assert tgt.min() == 6 * 196 and tgt.max() == 8 * 196 - 1          # last two tubelets
    assert ctx.min() == 196                                            # lead tubelet 0 absent
    m2 = B.forecast_mask(2)
    ctx2, tgt2, n2 = J.token_index_sets(m2)
    assert n2 == 1 and len(ctx2) == 7 * 196 and tgt2.min() == 7 * 196


def _jepa_ok():
    try:
        from transformers import VJEPA2Model  # noqa
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _jepa_ok() or not torch.cuda.is_available(), reason="needs V-JEPA 2 + GPU")
def test_world_model_is_causal_and_readout_packing():
    import pretrain_route_j as J
    wm = J.WorldModel(init="random", grad_ckpt=False).cuda().eval()
    rng = np.random.default_rng(0); P = 24
    y = rng.normal(0, 1, B.NF * P).astype(np.float32)
    rows, mu, sd = B.z_rows(y, P, ctx_periods=12)
    rows2 = rows.copy(); rows2[-4:] = 200                              # change only the future frames
    mask = B.forecast_mask(4)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        v1 = B.rows_to_video(torch.from_numpy(rows)[None], "cuda"); v2 = B.rows_to_video(torch.from_numpy(rows2)[None], "cuda")
        ctx, tgt, _ = J.token_index_sets(mask)
        p1 = wm.predict_repr(v1, ctx, tgt); p2 = wm.predict_repr(v2, ctx, tgt)
        assert torch.allclose(p1, p2, atol=1e-3)                      # predictor never sees the future
        f = wm.forecast_pixels(v1, mask, 4)
        assert f.shape == (1, 4, 3, B.IMG, B.IMG)
        loss, lp, lj = wm.forward_loss(v1, mask, 4)
        assert torch.isfinite(loss) and lj > 0
    # readout packing: feed the TRUE pixels through the same unpack and decode them back
    labels = B.U.patchify(B.U.unnormalize(v1.float())).reshape(1, B.N_TOK, -1)[:, tgt]
    c = labels.view(1, 2, B.GH, B.GH, B.TS, B.PS, B.PS, 3)
    frames = c.permute(0, 1, 4, 7, 2, 5, 3, 6).reshape(1, 4, 3, B.IMG, B.IMG)
    z = B.decode_frames_gray(frames.mean(2)[0].cpu(), P).numpy()
    z_true = np.clip((y - mu) / sd, -3, 3).reshape(B.NF, P)[-4:]
    assert np.max(np.abs(z - z_true)) < 0.03


@pytest.mark.skipif(not _jepa_ok() or not torch.cuda.is_available(), reason="needs V-JEPA 2 + GPU")
def test_ema_update_moves_target_towards_online():
    import pretrain_route_j as J
    wm = J.WorldModel(init="random", grad_ckpt=False, ema=0.5).cuda()
    with torch.no_grad():
        for p in wm.model.encoder.parameters(): p.add_(1.0)
    p_on = next(wm.model.encoder.parameters()).clone(); p_t0 = next(wm.target_encoder.parameters()).clone()
    wm.ema_update()
    p_t1 = next(wm.target_encoder.parameters())
    assert torch.allclose(p_t1, 0.5 * p_t0 + 0.5 * p_on, atol=1e-5)


# ------------------------------------------------------------------ full recipe on Route J (2026-09-12)
@pytest.mark.skipif(not _jepa_ok() or not torch.cuda.is_available(), reason="needs V-JEPA 2 + GPU")
def test_route_j_value_loss_and_32_frames():
    """value-space term is added when value_w > 0 (and only then); 32-frame clips forward and forecast."""
    import pretrain_route_j as J
    B.set_frames(32)
    try:
        wm = J.WorldModel(init="random", grad_ckpt=False).cuda().eval()
        rng = np.random.default_rng(1); P, hp = 24, 4
        rows, _, _ = B.z_rows(rng.normal(0, 1, B.NF * P).astype(np.float32), P, ctx_periods=B.NF - hp)
        v = B.rows_to_video(torch.from_numpy(rows)[None], "cuda")
        assert v.shape[1] == 32
        mask = B.forecast_mask(hp)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            wm.value_w = 0.0; l0, lp0, _ = wm.forward_loss(v, mask, hp)
            wm.value_w = 0.1; l1, lp1, _ = wm.forward_loss(v, mask, hp)
            f = wm.forecast_pixels(v, mask, hp)
        assert abs(float(lp0) - float(lp1)) < 1e-6 and float(l1) > float(l0)
        assert f.shape == (1, hp, 3, B.IMG, B.IMG) and torch.isfinite(f).all()
    finally:
        B.set_frames(16)


# ------------------------------------------------------------------ V-JEPA 2.1 ViT-B world model (2026-09-15)
def _v21_ok():
    import os as _os
    ck = _os.environ.get("VJEPA21_CKPT", "/nyx-storage1/hanliu/wm4ts/ckpt/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt")
    repo = _os.environ.get("VJEPA21_REPO", "/nyx-storage1/hanliu/wm4ts/vjepa2_repo")
    return _os.path.exists(ck) and _os.path.isdir(repo + "/app/vjepa_2_1")


def _v21_teacher_ok():
    import os as _os
    t = _os.environ.get("VJEPA21_TEACHER", "/nyx-storage1/hanliu/wm4ts/ckpt/vjepa2_1/vjepa2_1_vitG_384_teacher_bf16.pt")
    return _v21_ok() and _os.path.exists(t)


@pytest.mark.skipif(not _v21_ok() or not torch.cuda.is_available(), reason="needs the V-JEPA 2.1 repo + ckpt + GPU")
def test_vjepa21_vitb_loads_is_causal_and_forecasts():
    import pretrain_route_j as J
    B.set_frames(32)
    try:
        wm = J.WorldModel21(init="pretrained", grad_ckpt=False).cuda().eval()
        assert wm.load_report["encoder"] == "strict"
        assert wm.load_report["predictor_missing"] == ["predictor_proj.bias", "predictor_proj.weight"]
        n_enc = sum(p.numel() for p in wm.model.encoder.parameters()) / 1e6
        assert 80 < n_enc < 95, n_enc
        rng = np.random.default_rng(0); P, hp = 24, 4
        y = rng.normal(0, 1, B.NF * P).astype(np.float32)
        rows, _, _ = B.z_rows(y, P, ctx_periods=B.NF - hp)
        rows2 = rows.copy(); rows2[-hp:] = 200                               # change ONLY the future frames
        mask = B.forecast_mask(hp)
        ctx, tgt, n_fut = J.token_index_sets(mask)
        v1 = B.rows_to_video(torch.from_numpy(rows)[None], "cuda")
        v2 = B.rows_to_video(torch.from_numpy(rows2)[None], "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h1, _ = wm.encode_context(v1, ctx); h2, _ = wm.encode_context(v2, ctx)
            p1 = wm.predict_repr(v1, ctx, tgt); p2 = wm.predict_repr(v2, ctx, tgt)
            assert torch.allclose(h1.float(), h2.float(), atol=1e-3)         # context encoding never sees the future
            assert torch.allclose(p1.float(), p2.float(), atol=1e-3)         # nor does the predictor
            assert p1.shape == (1, tgt.numel(), 768)
            t1 = wm.target_repr(v1, tgt); t2 = wm.target_repr(v2, tgt)
            assert not torch.allclose(t1.float(), t2.float(), atol=1e-3)     # targets DO depend on the future
            f = wm.forecast_pixels(v1, mask, hp)
            assert f.shape == (1, hp, 3, B.IMG, B.IMG) and torch.isfinite(f).all()
            wm.value_w = 0.1
            loss, lp, lj = wm.forward_loss(v1, mask, hp)
            assert torch.isfinite(loss) and float(lj) > 0
    finally:
        B.set_frames(16)


@pytest.mark.skipif(not _v21_ok() or not torch.cuda.is_available(), reason="needs the V-JEPA 2.1 repo + ckpt + GPU")
def test_vjepa21_checkpoint_roundtrip(tmp_path):
    import pretrain_route_j as J
    B.set_frames(32)
    try:
        wm = J.WorldModel21(init="pretrained").cuda().eval()
        J.save_ckpt(wm, str(tmp_path / "s"), {"arm": "vjepa21b_wm", "lam_jepa": 1.0, "step": 0, "frames": 32,
                                               "backbone": "vjepa2_1_vitb"})
        wm2, meta = J.load_ckpt(str(tmp_path / "s"), "cuda")
        assert isinstance(wm2, J.WorldModel21) and meta["backbone"] == "vjepa2_1_vitb"
        rows, _, _ = B.z_rows(np.random.default_rng(1).normal(size=B.NF * 24).astype(np.float32), 24, ctx_periods=28)
        v = B.rows_to_video(torch.from_numpy(rows)[None], "cuda"); mask = B.forecast_mask(4)
        with torch.no_grad():
            a = wm.forecast_pixels(v, mask, 4); b = wm2.forecast_pixels(v, mask, 4)
        assert torch.allclose(a, b, atol=1e-5)
    finally:
        B.set_frames(16)


@pytest.mark.skipif(not _v21_teacher_ok() or not torch.cuda.is_available(), reason="needs V-JEPA 2.1 ViT-G teacher + GPU")
def test_vjepa21_vitG_teacher_keeps_head_and_is_causal(tmp_path):
    """teacher='vitG': the full pretrained predictor (head included) loads strictly, predictions and targets are 1664-d,
    the student never sees the future, and the saved checkpoint excludes the 2B teacher yet reloads identically."""
    import os, pretrain_route_j as J
    B.set_frames(32)
    try:
        wm = J.WorldModel21(init="pretrained", teacher="vitG").cuda().eval()
        assert wm.load_report["predictor"] == "strict (head kept)"
        assert wm.model.predictor.predictor_proj.weight.shape == (1664, 384)
        rng = np.random.default_rng(3); P, hp = 24, 4
        rows, _, _ = B.z_rows(rng.normal(0, 1, B.NF * P).astype(np.float32), P, ctx_periods=B.NF - hp)
        rows2 = rows.copy(); rows2[-hp:] = 200
        mask = B.forecast_mask(hp); ctx, tgt, _ = J.token_index_sets(mask)
        v1 = B.rows_to_video(torch.from_numpy(rows)[None], "cuda"); v2 = B.rows_to_video(torch.from_numpy(rows2)[None], "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p1 = wm.predict_repr(v1, ctx, tgt); p2 = wm.predict_repr(v2, ctx, tgt)
            t1 = wm.target_repr(v1, tgt); t2 = wm.target_repr(v2, tgt)
            assert p1.shape == (1, tgt.numel(), 1664) and t1.shape == (1, tgt.numel(), 1664)
            assert torch.allclose(p1.float(), p2.float(), atol=1e-3)
            assert not torch.allclose(t1.float(), t2.float(), atol=1e-3)
            loss, lp, lj = wm.forward_loss(v1, mask, hp)
            assert torch.isfinite(loss) and float(lj) > 0
            cos = torch.nn.functional.cosine_similarity(p1.float(), t1.float(), dim=-1).mean()
        print("pretrained head vs ViT-G target cosine at step 0:", float(cos))
        J.save_ckpt(wm, str(tmp_path / "g"), {"arm": "vjepa21b_wm", "lam_jepa": 1.0, "step": 0, "frames": 32,
                                               "backbone": "vjepa2_1_vitb", "teacher": "vitG"})
        assert "target_encoder" not in torch.load(str(tmp_path / "g" / "world_model.pt"), map_location="cpu")
        wm2, _ = J.load_ckpt(str(tmp_path / "g"), "cuda")
        assert wm2.target_encoder is None
        with torch.no_grad():
            assert torch.allclose(wm.forecast_pixels(v1, mask, hp), wm2.forecast_pixels(v1, mask, hp), atol=1e-5)
    finally:
        B.set_frames(16)
