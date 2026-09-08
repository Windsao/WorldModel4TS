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
