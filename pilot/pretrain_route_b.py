"""Route B (VIDEO_TS_LITERATURE_AND_RESCUE_PLAN.md section 4): matched continual pretraining of the
VideoMAE-B architecture on real time series from FOUR encoder initialisations, identical in
everything else (data stream, masks, decoder init, optimiser, steps, seed).

  arm vmae_full : Kinetics-400 VideoMAE encoder + its pretrained decoder      (VisionTS++-style)
  arm vmae_enc  : Kinetics-400 VideoMAE encoder, FRESH decoder (seeded)
  arm imae_enc  : ImageNet image-MAE encoder (the VisionTS checkpoint) inflated to 3D, FRESH decoder
  arm random    : fresh encoder + fresh decoder (same seed)

vmae_enc / imae_enc / random share a bit-identical decoder initialisation, so the only thing
that differs between them is the encoder's starting weights. That is the manipulated variable.

Rendering: PERIOD-FRAME area chart. Frame f = period f of a 16-period window, drawn as a filled
silhouette whose height at phase p is the (context-normalised) value -- value is POSITION, not
intensity. Forecasting = masking the last 1 or 2 tubelets (2 or 4 whole future frames) and
predicting their raw pixels (norm_pix_loss=False). Native 75% tube masks are mixed in so the
model keeps its generic inpainting skill. Normalisation statistics come from the CONTEXT frames
only, so no visible pixel carries future information.

Data: a LOTSA subset with the energy / solar / road-traffic domains EXCLUDED (see
--lotsa-dir; built by --build-corpus into one float32 memmap), plus 20% synthetic series.
The six benchmark datasets are never read here.

Usage:
  python pilot/pretrain_route_b.py --build-corpus --lotsa-dir ... --corpus-dir ...
  python pilot/pretrain_route_b.py --arm imae_enc --gpu 2 --corpus-dir ... --out ... --steps 20000
"""
import argparse, glob, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "july_cpt"))
sys.path.insert(0, HERE)
import videomae_patch_utils as U

IMG, PS, NF, TS = 224, 16, 16, 2
GH = IMG // PS                    # 14
TT = NF // TS                     # 8 tubelets
N_TOK = TT * GH * GH              # 1568
DARK, LIGHT, GRID = 0.12, 0.92, 0.80
GRID_EVERY = 28
ZMAX, BAND = 3.0, 0.40            # z in [-3,3] -> height 0.5 +- 0.4
IMN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
EXCLUDE_SUBSTR = ("electricity", "elec", "solar", "wind", "pems", "traffic", "loop", "energy",
                  "load", "power", "smart_meter", "lcl", "bdg", "buildings", "residential")


# ------------------------------------------------------------------ frequency -> period
def freq_to_period(freq):
    f = (freq or "").upper()
    if not f:
        return None
    import re
    m = re.match(r"^(\d*)\s*([A-Z]+)", f)
    if not m:
        return None
    n = int(m.group(1)) if m.group(1) else 1
    u = m.group(2)
    if u in ("T", "MIN"):
        return 1440 // n if 1440 % n == 0 and 1440 // n <= 288 and 1440 // n >= 4 else None
    if u == "H":
        return 24 // n if 24 % n == 0 and 24 // n >= 4 else None
    if u in ("D",):
        return 7
    if u in ("B",):
        return 5
    if u.startswith("W"):
        return 52
    if u.startswith("M"):
        return 12
    if u.startswith("Q"):
        return 4
    return None                    # yearly and unknown: skip


# ------------------------------------------------------------------ corpus
def build_corpus(lotsa_dir, corpus_dir, min_periods=18, max_series_per_dataset=20000):
    import pyarrow as pa
    os.makedirs(corpus_dir, exist_ok=True)
    chunks, offs, lens, Ps, dids, names = [], [], [], [], [], []
    total = 0
    for d in sorted(os.listdir(lotsa_dir)):
        if any(s in d.lower() for s in EXCLUDE_SUBSTR):
            print(f"[corpus] EXCLUDED by domain rule: {d}"); continue
        files = sorted(glob.glob(os.path.join(lotsa_dir, d, "*.arrow")))
        if not files:
            continue
        n_ser, n_obs = 0, 0
        did = len(names); names.append(d)
        for fp in files:
            try:
                t = pa.ipc.open_stream(open(fp, "rb")).read_all()
            except Exception:
                t = pa.ipc.open_file(open(fp, "rb")).read_all()
            cols = t.column_names
            for i in range(t.num_rows):
                if n_ser >= max_series_per_dataset:
                    break
                freq = t.column("freq")[i].as_py() if "freq" in cols else None
                P = freq_to_period(freq)
                if P is None:
                    continue
                tg = t.column("target")[i].as_py()
                series = tg if (len(tg) and isinstance(tg[0], (list, tuple))) else [tg]
                for s in series:
                    a = np.asarray(s, dtype=np.float32)
                    if a.ndim != 1 or len(a) < min_periods * P:
                        continue
                    if np.isnan(a).mean() > 0.2:
                        continue
                    chunks.append(a); offs.append(total); lens.append(len(a)); Ps.append(P); dids.append(did)
                    total += len(a); n_ser += 1; n_obs += len(a)
        print(f"[corpus] {d}: {n_ser} series, {n_obs/1e6:.2f}M obs", flush=True)
    mm = np.memmap(os.path.join(corpus_dir, "corpus.f32"), dtype=np.float32, mode="w+", shape=(total,))
    for a, o in zip(chunks, offs):
        mm[o:o + len(a)] = a
    mm.flush()
    np.savez(os.path.join(corpus_dir, "index.npz"), offs=np.array(offs, np.int64), lens=np.array(lens, np.int64),
             P=np.array(Ps, np.int32), did=np.array(dids, np.int32))
    json.dump({"datasets": names, "total_obs": int(total), "n_series": len(offs)},
              open(os.path.join(corpus_dir, "meta.json"), "w"), indent=1)
    print(f"[corpus] total {len(offs)} series, {total/1e6:.1f}M observations, {len(names)} datasets")


class Corpus:
    def __init__(self, corpus_dir, split="train", holdout_frac=0.05, seed=0):
        idx = np.load(os.path.join(corpus_dir, "index.npz"))
        self.offs, self.lens, self.P, self.did = idx["offs"], idx["lens"], idx["P"], idx["did"]
        meta = json.load(open(os.path.join(corpus_dir, "meta.json")))
        self.mm = np.memmap(os.path.join(corpus_dir, "corpus.f32"), dtype=np.float32, mode="r", shape=(meta["total_obs"],))
        rng = np.random.default_rng(seed)
        hold = rng.random(len(self.offs)) < holdout_frac
        sel = ~hold if split == "train" else hold
        self.ids = np.nonzero(sel)[0]
        # dataset weights ~ sqrt(observations), series uniform within dataset
        nd = len(meta["datasets"])
        obs = np.zeros(nd);
        for i in self.ids: obs[self.did[i]] += self.lens[i]
        w = np.sqrt(obs); self.dw = w / w.sum()
        self.by_d = [self.ids[self.did[self.ids] == d] for d in range(nd)]

    def sample(self, rng, n_p=NF, tries=10):
        for _ in range(tries):
            d = rng.choice(len(self.dw), p=self.dw)
            if len(self.by_d[d]) == 0: continue
            i = self.by_d[d][rng.integers(len(self.by_d[d]))]
            P, L = int(self.P[i]), int(self.lens[i])
            need = n_p * P
            if L < need: continue
            t = int(rng.integers(0, L - need + 1))
            y = np.array(self.mm[self.offs[i] + t: self.offs[i] + t + need], dtype=np.float32)
            if np.isnan(y).any(): continue
            if y.std() < 1e-6: continue
            return y, P
        return None, None


# ------------------------------------------------------------------ rendering
def col2phase(P):
    edges = np.rint(np.linspace(0, IMG, P + 1)).astype(np.int64)
    c2p = np.empty(IMG, np.int64)
    for p in range(P):
        c2p[edges[p]:edges[p + 1]] = p
    return c2p


def z_rows(y, P, ctx_periods, ctx_start=0):
    """y [16*P] -> (boundary rows [16, 224] int16, mu, sd). Stats from periods
    [ctx_start, ctx_start+ctx_periods) only -- the frames the encoder will actually see."""
    ctx = y[ctx_start * P:(ctx_start + ctx_periods) * P]
    mu, sd = float(ctx.mean()), float(ctx.std() + 1e-6)
    z = np.clip((y - mu) / sd, -ZMAX, ZMAX).reshape(NF, P)
    h = 0.5 + BAND * z / ZMAX
    rows = np.rint((1.0 - h) * (IMG - 1)).astype(np.int16)         # [16, P]
    return rows[:, col2phase(P)], mu, sd                           # [16, 224]


def rows_to_video(rows, device):
    """rows [B,16,224] (int) -> ImageNet-normalised video [B,16,3,224,224] float."""
    rows = rows.to(device).long()
    rgrid = torch.arange(IMG, device=device).view(1, 1, IMG, 1)
    fill = rgrid >= rows.unsqueeze(2)                              # [B,16,224,224]
    img = torch.where(fill, torch.tensor(DARK, device=device), torch.tensor(LIGHT, device=device))
    gl = (torch.arange(IMG, device=device) % GRID_EVERY == 0).view(1, 1, IMG, 1) & ~fill
    img = torch.where(gl, torch.tensor(GRID, device=device), img)
    vid = img.unsqueeze(2).expand(-1, -1, 3, -1, -1)
    return (vid - IMN_MEAN.to(device)) / IMN_STD.to(device)


def decode_frames_gray(gray, P):
    """gray [..., 224, 224] in [0,1] (LIGHT background, DARK fill, GRID lines in the background)
    -> z [..., P]. Soft row count: a pixel counts as filled in proportion to how far it lies
    below the GRID level, so grid lines (0.80) and background (0.92) count 0 and the fill (0.12)
    counts 1; blurred boundary pixels count fractionally. Exact inverse of z_rows up to rounding."""
    soft = ((GRID - gray) / (GRID - DARK)).clamp(0, 1)
    n_dark = soft.sum(-2)                                                  # [..., 224] filled rows
    rows = IMG - n_dark                                                    # boundary row
    h = 1.0 - rows / (IMG - 1)
    c2p = torch.as_tensor(col2phase(P), device=gray.device)
    out = torch.zeros(*h.shape[:-1], P, device=gray.device, dtype=h.dtype)
    cnt = torch.zeros(P, device=gray.device, dtype=h.dtype)
    out.index_add_(-1, c2p, h); cnt.index_add_(0, c2p, torch.ones(IMG, device=gray.device, dtype=h.dtype))
    return (out / cnt - 0.5) / BAND * ZMAX


# ------------------------------------------------------------------ masks
def forecast_mask(hp_frames, lead_tubelets=0):
    """mask the last hp_frames frames (the future) and, optionally, the first lead_tubelets
    tubelets (a SHORTER context: those frames are hidden and excluded from the loss)."""
    m = torch.zeros(TT, GH * GH, dtype=torch.bool)
    m[TT - hp_frames // TS:, :] = True
    if lead_tubelets:
        m[:lead_tubelets, :] = True
    return m.flatten()


def n_future_tokens(hp_frames):
    return (hp_frames // TS) * GH * GH


def masked_loss(model, vid, mask, hp):
    """raw-pixel MSE on the FUTURE tokens only for forecast batches (masked tokens are ordered by
    token index, so the future tubelets are the last n_future tokens); all masked tokens for tube
    batches. Labels use the exact HF packing via videomae_patch_utils."""
    set_attn_bias(model, mask[0])
    out = model(pixel_values=vid, bool_masked_pos=mask)
    logits = out.logits.float()
    B = vid.shape[0]
    labels = U.patchify(U.unnormalize(vid)).reshape(B, N_TOK, -1)[mask].view(B, logits.shape[1], -1).float()
    if hp:
        nf = n_future_tokens(hp)
        return F.mse_loss(logits[:, -nf:], labels[:, -nf:])
    return F.mse_loss(logits, labels)


def tube_mask(rng, ratio=0.75):
    n = GH * GH
    cols = torch.from_numpy(rng.choice(n, int(n * ratio), replace=False))
    m = torch.zeros(TT, n, dtype=torch.bool)
    m[:, cols] = True
    return m.flatten()


# ------------------------------------------------------------------ encoder attention constraint
def _spatial_attn_forward(self, hidden_states, head_mask=None, output_attentions=False):
    """HF VideoMAE self-attention (transformers 4.46.3 default = SDPA) plus an additive attention
    bias `self._attn_bias` [n_vis, n_vis] (0 within a tubelet, -1e4 across tubelets), so the
    ENCODER sees each tubelet as an independent image while the decoder keeps full attention.
    Uses scaled_dot_product_attention so no [B, heads, n, n] probability tensor is materialised."""
    k_bias = torch.zeros_like(self.v_bias, requires_grad=False) if self.q_bias is not None else None
    keys = F.linear(input=hidden_states, weight=self.key.weight, bias=k_bias)
    values = F.linear(input=hidden_states, weight=self.value.weight, bias=self.v_bias)
    queries = F.linear(input=hidden_states, weight=self.query.weight, bias=self.q_bias)
    key_layer = self.transpose_for_scores(keys)
    value_layer = self.transpose_for_scores(values)
    query_layer = self.transpose_for_scores(queries)
    bias = getattr(self, "_attn_bias", None)
    if bias is not None:
        bias = bias.to(query_layer.dtype)
    context_layer = F.scaled_dot_product_attention(
        query_layer, key_layer, value_layer, attn_mask=bias,
        dropout_p=self.dropout.p if self.training else 0.0)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    context_layer = context_layer.view(context_layer.size()[:-2] + (self.all_head_size,))
    return (context_layer, None) if output_attentions else (context_layer,)


def apply_spatial_attention(model):
    """patch every ENCODER self-attention module (decoder untouched)."""
    import types
    n = 0
    for layer in model.videomae.encoder.layer:
        att = layer.attention.attention
        att.forward = types.MethodType(_spatial_attn_forward, att)
        att._attn_bias = None
        n += 1
    model._spatial_attention = True
    return n


def set_attn_bias(model, mask_row):
    """mask_row: bool [N_TOK] (same for the batch). Visible tokens keep their tubelet-major order
    inside HF VideoMAE, so visible token i belongs to tubelet (global index // 196)."""
    if not getattr(model, "_spatial_attention", False):
        return
    idx = torch.nonzero(~mask_row.reshape(-1)).flatten()
    tid = idx // (GH * GH)
    same = tid[:, None] == tid[None, :]
    bias = torch.where(same, torch.zeros((), dtype=torch.float32), torch.full((), -1e4, dtype=torch.float32))
    dev = next(model.parameters()).device
    for layer in model.videomae.encoder.layer:
        layer.attention.attention._attn_bias = bias.to(dev)


# ------------------------------------------------------------------ data stream
class Stream(IterableDataset):
    """Yields (rows int16 [B,16,224], mask bool [B,1568], hp int, lead int). Same seed -> same stream.
    With probability p_lead a forecast batch also hides the first `lead` tubelets, so the model
    learns to forecast from 8..12 context periods (needed at pseudo-origins in evaluation)."""

    def __init__(self, corpus_dir, batch, seed, synth_frac=0.2, p_forecast=0.7, p_lead=0.3, split="train"):
        self.corpus_dir, self.batch, self.seed = corpus_dir, batch, seed
        self.synth_frac, self.p_forecast, self.p_lead, self.split = synth_frac, p_forecast, p_lead, split

    def __iter__(self):
        from pretrain_vmae_ts import synth_series
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi else 0
        rng = np.random.default_rng(self.seed * 1000 + wid)
        corpus = Corpus(self.corpus_dir, split=self.split)
        while True:
            u = rng.random()
            if u < self.p_forecast / 2: hp = 2
            elif u < self.p_forecast: hp = 4
            else: hp = 0
            lead = 0
            if hp and rng.random() < self.p_lead:
                kmax = (NF - hp - 8) // TS                # keep >= 8 visible context periods
                lead = int(rng.integers(1, kmax + 1))
            mask = forecast_mask(hp, lead) if hp else tube_mask(rng)
            ctx_start = lead * TS
            ctx_periods = (NF - hp - ctx_start) if hp else NF - 4   # tube batches: first 12 periods
            rows = np.empty((self.batch, NF, IMG), np.int16)
            for b in range(self.batch):
                y, P = (None, None) if rng.random() < self.synth_frac else corpus.sample(rng)
                if y is None:
                    P = int(rng.choice([7, 12, 24, 48, 52, 96, 144]))
                    y = synth_series(rng, NF, P)
                rows[b], _, _ = z_rows(y, P, ctx_periods, ctx_start)
            yield torch.from_numpy(rows), mask.unsqueeze(0).expand(self.batch, -1).clone(), hp, lead


# ------------------------------------------------------------------ models / arms
MAE_DECODER = dict(decoder_num_hidden_layers=8, decoder_hidden_size=512,
                   decoder_num_attention_heads=16, decoder_intermediate_size=2048)


def fresh_model(seed, decoder="vmae"):
    """decoder='vmae': VideoMAE-B's 4x384 decoder; 'mae': the image MAE's 8x512 decoder shape
    (needed to host the MAE pretrained decoder weights and as its capacity-matched controls)."""
    from transformers import VideoMAEForPreTraining, VideoMAEConfig
    cfg = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base")
    cfg.norm_pix_loss = False
    if decoder == "mae":
        for k, v in MAE_DECODER.items():
            setattr(cfg, k, v)
    torch.manual_seed(seed)
    return VideoMAEForPreTraining(cfg)


def map_image_mae_to_videomae(mae_sd):
    """official MAE ViT-B keys -> HF VideoMAE encoder keys (3D-inflated patch embedding).
    The key bias of qkv is dropped: a per-query constant added to every logit cancels in softmax."""
    out = {}
    w = mae_sd["patch_embed.proj.weight"]                            # [768,3,16,16]
    out["videomae.embeddings.patch_embeddings.projection.weight"] = w.unsqueeze(2).repeat(1, 1, TS, 1, 1) / TS
    out["videomae.embeddings.patch_embeddings.projection.bias"] = mae_sd["patch_embed.proj.bias"]
    for i in range(12):
        s, d = f"blocks.{i}.", f"videomae.encoder.layer.{i}."
        qkv_w, qkv_b = mae_sd[s + "attn.qkv.weight"], mae_sd[s + "attn.qkv.bias"]
        q_w, k_w, v_w = qkv_w.chunk(3, 0)
        q_b, _, v_b = qkv_b.chunk(3, 0)
        out[d + "attention.attention.query.weight"] = q_w
        out[d + "attention.attention.key.weight"] = k_w
        out[d + "attention.attention.value.weight"] = v_w
        out[d + "attention.attention.q_bias"] = q_b
        out[d + "attention.attention.v_bias"] = v_b
        out[d + "attention.output.dense.weight"] = mae_sd[s + "attn.proj.weight"]
        out[d + "attention.output.dense.bias"] = mae_sd[s + "attn.proj.bias"]
        out[d + "layernorm_before.weight"] = mae_sd[s + "norm1.weight"]
        out[d + "layernorm_before.bias"] = mae_sd[s + "norm1.bias"]
        out[d + "layernorm_after.weight"] = mae_sd[s + "norm2.weight"]
        out[d + "layernorm_after.bias"] = mae_sd[s + "norm2.bias"]
        out[d + "intermediate.dense.weight"] = mae_sd[s + "mlp.fc1.weight"]
        out[d + "intermediate.dense.bias"] = mae_sd[s + "mlp.fc1.bias"]
        out[d + "output.dense.weight"] = mae_sd[s + "mlp.fc2.weight"]
        out[d + "output.dense.bias"] = mae_sd[s + "mlp.fc2.bias"]
    out["videomae.layernorm.weight"] = mae_sd["norm.weight"]
    out["videomae.layernorm.bias"] = mae_sd["norm.bias"]
    return out


def map_image_mae_decoder_to_videomae(mae_sd):
    """official MAE ViT-B DECODER keys -> HF VideoMAE decoder keys (8x512 shape). The pixel head
    (768 = 16*16*3 per image patch) is inflated to a tubelet head (1536 = 2*16*16*3) by stacking
    it twice, so at init both frames of a tubelet get the image decoder's prediction; the
    decoder_embed bias is dropped (HF's encoder_to_decoder has no bias); position embeddings are
    fixed 3D sincos in HF and are not mapped."""
    out = {"mask_token": mae_sd["mask_token"],
           "encoder_to_decoder.weight": mae_sd["decoder_embed.weight"]}
    for i in range(8):
        s, d = f"decoder_blocks.{i}.", f"decoder.decoder_layers.{i}."
        q_w, k_w, v_w = mae_sd[s + "attn.qkv.weight"].chunk(3, 0)
        q_b, _, v_b = mae_sd[s + "attn.qkv.bias"].chunk(3, 0)
        out[d + "attention.attention.query.weight"] = q_w
        out[d + "attention.attention.key.weight"] = k_w
        out[d + "attention.attention.value.weight"] = v_w
        out[d + "attention.attention.q_bias"] = q_b
        out[d + "attention.attention.v_bias"] = v_b
        out[d + "attention.output.dense.weight"] = mae_sd[s + "attn.proj.weight"]
        out[d + "attention.output.dense.bias"] = mae_sd[s + "attn.proj.bias"]
        out[d + "layernorm_before.weight"] = mae_sd[s + "norm1.weight"]
        out[d + "layernorm_before.bias"] = mae_sd[s + "norm1.bias"]
        out[d + "layernorm_after.weight"] = mae_sd[s + "norm2.weight"]
        out[d + "layernorm_after.bias"] = mae_sd[s + "norm2.bias"]
        out[d + "intermediate.dense.weight"] = mae_sd[s + "mlp.fc1.weight"]
        out[d + "intermediate.dense.bias"] = mae_sd[s + "mlp.fc1.bias"]
        out[d + "output.dense.weight"] = mae_sd[s + "mlp.fc2.weight"]
        out[d + "output.dense.bias"] = mae_sd[s + "mlp.fc2.bias"]
    out["decoder.norm.weight"] = mae_sd["decoder_norm.weight"]
    out["decoder.norm.bias"] = mae_sd["decoder_norm.bias"]
    out["decoder.head.weight"] = torch.cat([mae_sd["decoder_pred.weight"]] * TS, 0)
    out["decoder.head.bias"] = torch.cat([mae_sd["decoder_pred.bias"]] * TS, 0)
    return out


ARMS = ["vmae_full", "vmae_enc", "imae_enc", "random",           # 4x384 decoder (round 1)
        "imae_full", "imae_enc_d8", "vmae_enc_d8", "random_d8"]  # 8x512 decoder (round 2)


def _load_mae(mae_ckpt):
    mae_sd = torch.load(mae_ckpt, map_location="cpu")
    return mae_sd.get("model", mae_sd)


def _overwrite(base, part):
    sd = base.state_dict()
    for k, v in part.items():
        assert k in sd and sd[k].shape == v.shape, (k, sd[k].shape if k in sd else None, v.shape)
    sd.update(part)
    base.load_state_dict(sd)
    return len(part)


def build_arm(arm, seed, mae_ckpt):
    from transformers import VideoMAEForPreTraining
    if arm.endswith("_d8") or arm == "imae_full":
        base = fresh_model(seed, decoder="mae")    # shared fresh 8x512 decoder for the 3 controls
        if arm == "random_d8":
            return base, {"encoder_loaded": 0, "decoder": "fresh_8x512"}
        if arm == "vmae_enc_d8":
            pt = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")
            n = _overwrite(base, {k: v for k, v in pt.state_dict().items() if k.startswith("videomae.")})
            return base, {"encoder_loaded": n, "decoder": "fresh_8x512"}
        mae_sd = _load_mae(mae_ckpt)
        enc = map_image_mae_to_videomae(mae_sd)
        missing = [k for k in base.state_dict() if k.startswith("videomae.") and k not in enc]
        assert not missing, f"unmapped encoder keys: {missing[:5]}"
        n = _overwrite(base, enc)
        if arm == "imae_enc_d8":
            return base, {"encoder_loaded": n, "decoder": "fresh_8x512"}
        dec = map_image_mae_decoder_to_videomae(mae_sd)
        missing = [k for k in base.state_dict() if not k.startswith("videomae.") and k not in dec]
        assert not missing, f"unmapped decoder keys: {missing[:5]}"
        m = _overwrite(base, dec)
        return base, {"encoder_loaded": n, "decoder": f"pretrained_mae_{m}_tensors"}
    base = fresh_model(seed)                       # the shared fresh init (decoder for 3 arms)
    if arm == "random":
        return base, {"encoder_loaded": 0}
    if arm in ("vmae_full", "vmae_enc"):
        pt = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")
        pt.config.norm_pix_loss = False
        if arm == "vmae_full":
            return pt, {"encoder_loaded": 1, "decoder": "pretrained"}
        sd = base.state_dict()
        enc = {k: v for k, v in pt.state_dict().items() if k.startswith("videomae.")}
        sd.update(enc)
        base.load_state_dict(sd)
        return base, {"encoder_loaded": len(enc), "decoder": "fresh"}
    if arm == "imae_enc":
        mae_sd = torch.load(mae_ckpt, map_location="cpu")
        mae_sd = mae_sd.get("model", mae_sd)
        enc = map_image_mae_to_videomae(mae_sd)
        sd = base.state_dict()
        missing = [k for k in sd if k.startswith("videomae.") and k not in enc]
        assert not missing, f"unmapped encoder keys: {missing[:5]}"
        for k, v in enc.items():
            assert sd[k].shape == v.shape, (k, sd[k].shape, v.shape)
        sd.update(enc)
        base.load_state_dict(sd)
        return base, {"encoder_loaded": len(enc), "decoder": "fresh"}
    raise ValueError(arm)


# ------------------------------------------------------------------ eval helper (pixel -> z)
@torch.no_grad()
def forecast_z(model, rows, mask, hp, P_list, device):
    """rows [B,16,224], forecast mask with hp future frames -> decoded z of the future frames
    [B, hp, P] per sample (list, since P varies) plus pixel loss. Uses the raw-pixel logits."""
    vid = rows_to_video(rows, device)
    out = model(pixel_values=vid, bool_masked_pos=mask.to(device))
    logits = out.logits.float()                                        # [B, n_masked, 1536]
    B = rows.shape[0]
    n_fut_tok = (hp // TS) * GH * GH
    cubes = logits[:, -n_fut_tok:].view(B, hp // TS, GH, GH, TS, PS, PS, 3)   # tubelet-major as HF packs
    frames = cubes.permute(0, 1, 4, 7, 2, 5, 3, 6).reshape(B, hp, 3, IMG, IMG)
    gray = frames.mean(2)                                              # [B, hp, 224, 224] in ~[0,1]
    zs = [decode_frames_gray(gray[b], int(P_list[b])) for b in range(B)]
    return zs, float(out.loss)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-corpus", action="store_true")
    ap.add_argument("--lotsa-dir", default="/nyx-storage1/hanliu/wm4ts/lotsa")
    ap.add_argument("--corpus-dir", default="/nyx-storage1/hanliu/wm4ts/route_b/corpus")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--mae-ckpt", default="/nyx-storage1/hanliu/wm4ts/ckpt/mae_visualize_vit_base.pth")
    ap.add_argument("--out", default="/nyx-storage1/hanliu/wm4ts/route_b")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--save-every", type=int, default=2500)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--enc-attn", choices=["full", "spatial"], default="full",
                    help="spatial: encoder attends only within each tubelet (decoder unchanged)")
    ap.add_argument("--tag", default="", help="suffix for the output dir, e.g. _spatial or _60k")
    args = ap.parse_args()

    if args.build_corpus:
        build_corpus(args.lotsa_dir, args.corpus_dir)
        return

    dev = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    model, info = build_arm(args.arm, args.seed, args.mae_ckpt)
    if args.enc_attn == "spatial":
        info["spatial_attention_layers"] = apply_spatial_attention(model)
    model = model.to(dev).train()
    out_dir = os.path.join(args.out, args.arm + args.tag)
    os.makedirs(out_dir, exist_ok=True)
    json.dump({**vars(args), **info}, open(os.path.join(out_dir, "run_config.json"), "w"), indent=1)
    print(f"[{args.arm}] {info}  params {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))

    def lr_at(s):
        if s < args.warmup:
            return args.lr * s / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * p))

    dl = DataLoader(Stream(args.corpus_dir, args.batch, seed=args.seed, split="train"), batch_size=None,
                    num_workers=args.workers, prefetch_factor=4, persistent_workers=True)
    # fixed validation batches (held-out series, forecast masks only), identical across arms
    vstream = iter(Stream(args.corpus_dir, args.batch, seed=777, p_forecast=1.0, split="holdout"))
    val = [next(vstream) for _ in range(args.val_batches)]
    print(f"[val] {len(val)} held-out batches, hp/lead = {[(v[2], v[3]) for v in val]}", flush=True)

    log = open(os.path.join(out_dir, "train_log.jsonl"), "a")
    it = iter(dl)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        rows, mask, hp, lead = next(it)
        vid = rows_to_video(rows, dev)
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = masked_loss(model, vid, mask.to(dev), hp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            rec = {"step": step, "loss": float(loss.detach()), "hp": int(hp), "lead": int(lead),
                   "lr": lr_at(step), "gn": float(gn), "t": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 200 == 0:
                print(json.dumps(rec), flush=True)
        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            vl = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for vrows, vmask, vhp, vlead in val:
                    vl.append(float(masked_loss(model, rows_to_video(vrows, dev), vmask.to(dev), vhp)))
            model.train()
            rec = {"step": step, "val_pixel_loss": float(np.mean(vl))}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            d = os.path.join(out_dir, f"step_{step}")
            model.save_pretrained(d)
            print(f"[ckpt] {d}", flush=True)
    log.close()


if __name__ == "__main__":
    main()
