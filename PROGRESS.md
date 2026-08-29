# WorldModel4TS — Progress Report

**Question:** do video foundation models (VideoMAE, Wan/VACE) transfer to time-series forecasting?

**Answer: no.** Across 101 runs covering discriminative and generative backbones, four
fine-tuning regimes, seven standard benchmarks, one genuinely spatiotemporal benchmark, and
two model scales, the pretrained video prior contributes **at most ~2% MSE** over a
random-initialized backbone, only on high-channel data, and is **actively harmful** on the
small ETT datasets. The generative video-editing models are **1.7–2.2x worse than a trivial
seasonal-mean baseline**.

Branch: `field-video-clean`. Last commit: `48791c3`.
Date of this report: 2026-08-28.

---

## 1. Code map

Everything lives in `pilot/`. Five standalone scripts, no shared package.

| File | What it is | LOC |
|---|---|---|
| `pilot/run_field.py` | **Main driver.** VideoMAE/ViT-MAE forecaster; all four tuning regimes; all ablations. | 22.6 KB |
| `pilot/probe_frozen.py` | No-training diagnostic: frozen-feature participation ratio + ridge linear-probe skill. | 7.2 KB |
| `pilot/run_stvid.py` | **A1**: METR-LA spatiotemporal traffic (207 sensors) rendered as a spatial heatmap video. | 9.1 KB |
| `pilot/run_vace_ts.py` | **Generative**: Wan2.1-VACE (1.3B / 14B) as a forecaster via masked video inpainting. | 6.4 KB |
| `pilot/run_ltx_ts.py` | Generative, LTX-Video 2B backend. **Never ran** — dependency conflict, see §8. | 5.7 KB |

Supporting: `VACE/` (patched clone of ali-vilab/VACE, see §9), `third_party/`,
`pilot/results_field/` (all 100 result JSONs), `README.md`.

### 1.1 `pilot/run_field.py` — key implementation points

| Concept | Location |
|---|---|
| Dataset registry (period `P`, split borders) | `run_field.py:41-49` |
| Data load, z-score on train split, channel reorder by correlation | `load_mv`, `run_field.py:66` |
| Sliding windows, train/test split | `windows`, `run_field.py:97` |
| Model | `class FieldVMAE`, `run_field.py:110` |
| **Rendering** TS -> video | `FieldVMAE.render`, `run_field.py:161` |
| — `period` mode: frame f = period f (rows=phase, cols=phase) | `run_field.py:183-203` |
| — `vts` mode: VisionTS-style single 2D image, replicated to a static 16-frame clip | `run_field.py:172-182` |
| — `field` mode: rows = variables, cols = phase (true multivariate field) | `run_field.py:190-193` |
| — context/16-frame decoupling (temporal resample of G period-frames to NF=16) | `run_field.py:184-189` |
| **Full-FT forward** (pooled tokens -> MLP head) | `FieldVMAE.forward`, `run_field.py:243` |
| **Adapter forward** (uni) | `_forward_adapter`, `run_field.py:218` |
| **Adapter forward** (field / A2) | `_forward_adapter_field`, `run_field.py:205` |
| — zero-init readout head => warm start is exactly NLinear+RevIN | `run_field.py:151`, `:157` |
| — `--no-vbranch` ablation (pure NLinear+RevIN) | `run_field.py:227-228`, `:211-212` |
| — FiLM fusion (video modulates NLinear instead of adding to it) | `run_field.py:152-154`, `:234-237` |
| **Freezing logic** for `frozen` / `ln` / `adapter` | `apply_tune`, `run_field.py:263` |
| Train / eval loop | `run`, `run_field.py:280` |
| CLI | `main`, `run_field.py:315-338` |
| Baselines (seasonal-naive, seasonal-mean) | `run_field.py:359` |

### 1.2 The adapter architecture (`run_field.py:218`)

This is **not LoRA.** `apply_tune(model, "adapter")` sets `requires_grad_(False)` on every
parameter of `model.enc` and re-enables nothing inside it. The VideoMAE forward pass is
bit-for-bit the original pretrained model; all trainable parameters sit *outside* the backbone.

```
        x  [B, L, 1]  raw context
              |
    mu, sd = x.mean(1), x.std(1)          <- RevIN-style instance norm
    z = (x - mu) / sd
              |
      +-------+---------------------------------+
      |                                         |
  MAIN PATH                              SIDE PATH: render(x)
  self.lin                                       |
  Linear(L -> horizon)                  +--------v---------+
  = the NLinear baseline                |    VideoMAE      |  FROZEN
  37K params                            |     (86M)        |
      |                                 +--------+---------+
      |                              tokens [B, 1568, 768]
      |                                          |
      |                        8 learned queries + cross-attention
      |                        -> xhead(6144 -> horizon)  <- ZERO-INIT
      |                                          |
      +-------------> lin_out + vmae_out <-------+
                           |
                      * sd + mu  -> forecast
```

Contrast with LoRA:

| | LoRA | this adapter |
|---|---|---|
| Insertion point | inside backbone, `W + BA` on q/v projections | entirely outside the backbone |
| Backbone forward | modified | unchanged (features are cacheable) |
| Bypass path | none — all information flows through the backbone | **yes** — NLinear maps input to output without touching VideoMAE |
| Goal | adapt the backbone to the task | read out from *frozen* features |

**Trainable parameter budget** (electricity, L=384, h=96, d=768):

| Component | Params | Share of trainable |
|---|---|---|
| `self.lin` (NLinear) | 36,960 | 1.2% |
| `self.q` (8 queries) | 6,144 | 0.2% |
| `self.xattn` (MultiheadAttention) | 2,362,368 | 78.9% |
| `self.xnorm` | 1,536 | 0.05% |
| `self.xhead` | 589,920 | 19.7% |
| **Effective total** | **2,996,928** | (~3.3% of the 89.9M model) |

> **Implementation wart:** `self.head` (665,952 params, `run_field.py:135`) is constructed
> unconditionally but is never called in adapter mode. It receives no gradient yet is counted by
> `apply_tune`'s trainable-parameter print, which is why logs report **~4%** rather than the
> effective **~3.3%**. Harmless, but the reported percentage is inflated.

### 1.3 `pilot/run_stvid.py` — A1 spatiotemporal

| Concept | Location |
|---|---|
| METR-LA load (207 sensors + lat/lon + validity mask) | `load_metrla`, `run_stvid.py:26` |
| Voronoi-style sensor -> 32x32 grid assignment | `grid_index`, `run_stvid.py:35` |
| Model (same adapter recipe: NLinear + frozen VideoMAE, zero-init `vhead`) | `class STVid`, `run_stvid.py:54` |
| Spatial heatmap video render (frame t = sensor field at t) | `STVid.render`, `run_stvid.py:70` |
| Forward | `run_stvid.py:79` |
| Masked MSE/MAE (METR-LA has missing sensors) | `masked_metrics`, `run_stvid.py:91` |

### 1.4 `pilot/run_vace_ts.py` — generative bridge

| Concept | Location |
|---|---|
| TS -> video: context frames as gray bands, future frames blanked to 0.5 + binary mask | `render_pair`, `run_vace_ts.py:39` |
| Inference: `WanVace.generate(prompt, [src_video], [src_mask], [None], ...)` | `run_vace_ts.py:~110` |
| Video -> TS: average generated future frames back to values | `decode`, `run_vace_ts.py:58` |

The bridge is deliberately loss-free in the forward direction: `decode(render_pair(x))` on the
*context* frames reconstructs the input to numerical precision, so any error is attributable to
the generative model, not the encoding.

---

## 2. How to reproduce

Environment: conda env `wm4ts`, `transformers==4.46.3` (the code asserts `<5`).
Data: `ETTh1/h2/m1/m2.csv`, `electricity.txt`, `traffic.txt`, `solar_AL.txt` in `--data-dir`.

```bash
# 1. Full fine-tune (the original positive result)
python pilot/run_field.py --dataset electricity --mode uni --tune full \
  --context-steps 384 --horizon-steps 96 --max-ch 112 --stride 8 --epochs 3 \
  --ft-cap 40000 --out-dir pilot/results_field/ctx

# 2. Frozen backbone, linear head only
python pilot/run_field.py --dataset electricity --mode uni --tune frozen ...

# 3. LayerNorm-only
python pilot/run_field.py --dataset electricity --mode uni --tune ln ...

# 4. Adapter (frozen backbone + NLinear + cross-attn readout)
python pilot/run_field.py --dataset electricity --mode uni --tune adapter \
  --horizon-steps 96 --max-ch 112 --stride 8 --epochs 3 --ft-cap 40000 \
  --out-dir pilot/results_field/adapter

# 5. THE DECISIVE ABLATION: same as 4, video branch deleted
python pilot/run_field.py ... --tune adapter --no-vbranch \
  --out-dir pilot/results_field/ablation

# 6. Random-init backbone (isolates what Kinetics pretraining is worth)
python pilot/run_field.py ... --tune adapter --pretrained 0

# 7. VisionTS-style 2D static render instead of a real video
python pilot/run_field.py ... --tune adapter --render vts

# 8. FiLM fusion (video in the main path, not a residual)
python pilot/run_field.py ... --tune adapter --fusion film

# 9. Frozen-feature diagnostic, no training
python pilot/probe_frozen.py --dataset electricity --n 1500

# 10. A1 spatiotemporal
python pilot/run_stvid.py --horizon 12 --ctx 16 --gh 32 --epochs 3
python pilot/run_stvid.py --horizon 12 --no-vbranch          # NLinear control

# 11. Generative VACE
python pilot/run_vace_ts.py --ckpt-dir <Wan2.1-VACE-1.3B snapshot> --model-name vace-1.3B \
  --n 6 --fpp 4 --steps 15 --size 480p
```

---

## 3. Experiment inventory (101 runs)

| Directory | Experiment | Runs |
|---|---|---|
| `results_field/matched/` | Full fine-tune, 7 datasets, h=96 (solar h=144) | 7 |
| `results_field/hsweep/` | Full fine-tune x horizon sweep | 20 |
| `results_field/expand/` | 4-seed repeats + ViT-MAE image control | 12 |
| `results_field/ctx/` | `--context-steps` L=96 vs L=384 | 2 |
| `results_field/frozen/` | Frozen backbone, linear head only | 7 |
| `results_field/ln/` | LayerNorm-affine-only tuning | 4 |
| `results_field/adapter/` | Frozen backbone + adapter, full grid | 24 |
| `results_field/ablation/` | `--no-vbranch` / `--pretrained 0` / `--render vts` / `--fusion film` | 11 |
| `results_field/a2/` | Field-mode adapter (multivariate field video) | 6 |
| `results_field/a1/` | METR-LA spatiotemporal | 4 |
| `results_field/probe/` | Frozen-feature diagnostics (no training) | 2 |
| `results_field/vace/` | Generative VACE 1.3B (+ 14B, cluster-only) | 1 (+1) |

All metrics are MSE / MAE on z-scored data (train-split statistics). Two baselines are computed
in every run: `snaive` (seasonal-naive: repeat the last period) and `smean` (seasonal-mean:
average over context periods) — `run_field.py:359`.

---

## 4. Main results

### 4.1 Full fine-tune vs baseline (h=96, solar h=144, `matched/`)

| Dataset | smean | full-FT | |
|---|---|---|---|
| ETTh1 | 0.4019 | 0.4616 | worse, +14.9% |
| ETTh2 | 0.3496 | 0.3513 | tie |
| ETTm1 | 0.3761 | **0.3345** | better, -11.1% |
| ETTm2 | 0.3224 | **0.2003** | better, -37.9% |
| electricity | 0.2073 | **0.1407** | better, -32.1% |
| solar | 0.2009 | 0.2240 | worse, +11.5% |
| traffic | 0.6439 | **0.3863** | better, -40.0% |

The wins are all on high-channel datasets (electricity 321, traffic 862). ETT (7 channels)
loses or ties.

### 4.2 Fine-tuning ladder (h=96)

| Dataset | smean | frozen | LN-only | adapter (3.3%) | full-FT (100%) |
|---|---|---|---|---|---|
| electricity | 0.2071 | 0.3009 | 0.2508 | 0.1438 | **0.1275**\* |
| ETTh1 | 0.4022 | 0.4960 | 0.4383 | **0.3935** | 0.4616 |
| traffic | 0.5175 | 0.6074 | 0.5303 | 0.3661 | **0.3233**\* |
| solar (h=144) | 0.2005 | 0.2260 | 0.2125 | — | 0.2240 |

\* see §7 for the config-matching caveat; these are the *comparable* full-FT numbers, not the
`matched/` ones.

**Frozen and LN-only cannot beat even the seasonal-mean baseline.** Zero-shot transfer of
VideoMAE features to forecasting fails outright.

### 4.3 Adapter grid (24 runs, MSE, adapter / smean)

| | h=96 | h=192 | h=336 | h=720 |
|---|---|---|---|---|
| ETTh1 | **0.394** / 0.402 | 0.435 / 0.417 | 0.482 / 0.416 | — |
| ETTh2 | **0.299** / 0.350 | 0.401 / 0.363 | 0.427 / 0.357 | 0.530 / 0.399 |
| ETTm1 | **0.319** / 0.376 | **0.349** / 0.385 | **0.374** / 0.396 | 0.432 / 0.413 |
| ETTm2 | **0.167** / 0.322 | — | **0.287** / 0.346 | 0.382 / 0.363 |
| electricity | **0.144** / 0.207 | **0.155** / 0.211 | **0.170** / 0.223 | **0.206** / 0.254 |
| solar | **0.172** / 0.202 | — | **0.200** / 0.205 | — |
| traffic | **0.366** / 0.518 | **0.375** / 0.512 | **0.382** / 0.519 | **0.406** / 0.540 |

High-channel datasets win at every horizon. ETT wins only at the shortest horizon and loses to
"predict the seasonal mean" beyond that.

### 4.4 Seed variance (4 seeds, full-FT, `expand/`)

| Dataset | Runs | Mean | Std | Relative |
|---|---|---|---|---|
| electricity h=96 | 0.1417, 0.1379, 0.1374, 0.1386 | 0.1389 | 0.0017 | **1.2%** |
| traffic h=96 | 0.3233, 0.3191, 0.3197, 0.3225 | 0.3212 | 0.0018 | **0.6%** |

**This sets the noise floor: differences below ~1.2% are not real.** It is the yardstick for §5.

---

## 5. The decisive ablations

Every comparison in this section is **exactly config-matched** — same dataset, channels,
context, horizon, stride, epochs, cap, seed. Only the named flag differs.

### 5.1 How much does the video branch contribute? (`--no-vbranch`)

| | adapter | pure NLinear+RevIN | video contribution |
|---|---|---|---|
| electricity h=96 | 0.1438 | 0.1544 | -6.9% |
| traffic h=96 | 0.3661 | 0.3896 | -6.0% |
| ETTh1 h=96 | 0.3935 | 0.4026 | -2.3% |
| ETTh1 h=336 | 0.4823 | 0.4499 | **+7.2% WORSE** |
| ETTh2 h=96 | 0.2988 | 0.2918 | **+2.4% WORSE** |
| ETTh2 h=336 | 0.4270 | 0.3733 | **+14.4% WORSE** |

98.8% of the trainable parameters are attached to the video branch. They buy 6.9% at best, and
lose 14.4% at worst. The 1.2% of parameters in the linear layer do everything else.

### 5.2 What is Kinetics pretraining actually worth? (`--pretrained 0`)

electricity, h=96, identical config:

```
pure NLinear + RevIN                    0.1544
  + randomly-initialized VideoMAE       0.1470     (-4.8%)
  + Kinetics-400 pretrained weights     0.1438     (-2.2%)
```

**Only 30% of the video branch's gain comes from pretraining.** The other 70% is the generic
capacity effect of bolting on 3M extra parameters — a randomly-initialized backbone gets most
of it. The entire value of "a video foundation model pretrained on Kinetics-400" is **2.2% MSE
on one dataset**, barely above the 1.2% seed noise floor.

### 5.3 Does the temporal structure of the video matter? (`--render vts`)

| | period-per-frame video | VisionTS-style static 2D image |
|---|---|---|
| electricity h=96 | 0.1438 | **0.1421** (better) |
| traffic h=96 | 0.3661 | 0.3657 (tie) |

Replacing the video with a **single static image replicated 16 times** is as good or better.
The motion prior — the entire reason to prefer a video model over an image model — contributes
nothing.

### 5.4 Fusion style (`--fusion film`)

Putting the video in the main path (`gamma * NLinear + beta`) instead of as an additive residual:

| | additive | FiLM | NLinear-only |
|---|---|---|---|
| electricity h=96 | 0.1438 | 0.1409 | 0.1544 |
| ETTh1 h=96 | 0.3935 | 0.3824 | 0.4026 |

FiLM is marginally better (2.0% / 2.8%) but does not change the picture: still within ~2x the
noise floor, still ~5% total over the linear baseline.

### 5.5 A2 — field-mode adapter (multivariate field video)

Rendering rows = variables, cols = phase, so consecutive frames show a genuine *cross-channel*
field evolving. This is the regime where a video model should have a structural advantage that
a channel-independent linear model provably cannot match.

| | field adapter | field, `--no-vbranch` | contribution |
|---|---|---|---|
| electricity | 0.1637 | 0.1643 | 0.4% |
| solar | 0.1795 | 0.1800 | 0.3% |
| traffic | 0.4566 | 0.4580 | 0.3% |

**All three are below the seed-noise floor: the video branch contributes exactly nothing.**
And field mode is *worse overall* than uni mode (electricity +13.8%, solar +4.5%, traffic
+24.7%) — spreading the channels across image rows destroys more signal than the cross-channel
view recovers.

---

## 6. Rescue attempts

### 6.1 A1 — genuinely spatiotemporal data (METR-LA, 207 sensors)

The hypothesis: video fails on ETT/electricity because those are not spatiotemporal. METR-LA
sensors have real coordinates, so a heatmap video has real spatial structure and real motion
(traffic waves propagating along roads). Masked MSE/MAE over valid sensors.

| Horizon | snaive | NLinear | STVid (video) | vs NLinear |
|---|---|---|---|---|
| h=3 | **53.44** / **3.504** | 59.71 / 3.876 | 55.73 / 3.858 | MSE -6.7%, MAE -0.5% |
| h=12 | 123.14 / **5.082** | 125.82 / 5.568 | **111.23** / 5.319 | MSE -11.6%, MAE -4.5% |

This is the **largest video contribution anywhere in the project**. Caveats that sink it:

- STVid **loses to seasonal-naive on MAE at both horizons** (3.858 vs 3.504; 5.319 vs 5.082).
- It beats snaive on MSE only at h=12. The MSE gain far exceeds the MAE gain, meaning the video
  branch helps only on rare large-error events, not typical error.
- The absolute level is nowhere near METR-LA state of the art (DCRNN: MAE 2.77 @ h=3, 3.15 @
  h=12, vs our 3.86 / 5.32). Our setup is weak overall, so a relative gain over our own weak
  linear control is not strong evidence.

### 6.2 Generative video-editing models

Render the context as a video, blank the future frames, let the model inpaint, decode back.

| Model | MSE | MAE | smean MSE | Ratio |
|---|---|---|---|---|
| Wan2.1 RePaint | — | — | — | degenerates to a copy machine |
| **Wan2.1-VACE-1.3B** (ETTh1) | 0.5480 | 0.5016 | 0.3233 | **1.70x worse** |
| **Wan2.1-VACE-14B** (ETTh1) | 0.6337 | 0.5458 | 0.2899 | **2.19x worse** |

Both are far worse than a trivial baseline, and **the 10x larger model is worse**, so this is
not a scale problem. VACE-14B result is on the cluster only; the JSON was not pulled into
`results_field/vace/`.

---

## 7. Config-matching caveats — READ BEFORE CITING ANY NUMBER

The result directories were produced at different times with different sweep settings. Not all
cross-directory comparisons are valid.

| Directory | `max_ch` | `ft_cap` | `stride` | `epochs` |
|---|---|---|---|---|
| `matched/` | 1000 (all channels) | 40000 | **1** (ETT) / 8 | 3 |
| `hsweep/` | 1000 (all channels) | **25000** | 8 | **2** |
| `expand/` | 112 | **20000** | 8 | 3 |
| `ctx/` | 112 | 40000 | 8 | 3 |
| `frozen/` | 112 | 40000 | 8 | **5** |
| `ln/`, `adapter/`, `ablation/`, `a2/` | 112 | 40000 | 8 | 3 |

**Consequences:**

1. **The `--no-vbranch`, `--pretrained 0`, `--render vts`, `--fusion film`, and A2 ablations
   (§5) are all perfectly matched** — `adapter/` and `ablation/`/`a2/` share every setting.
   Every conclusion in §5 is sound.

2. **The claim "the adapter matches or beats full fine-tuning" is only half true.** It was
   originally read off `adapter/` vs `matched/`, which differ in channel count on the
   high-channel datasets (electricity 112 vs 321, traffic 112 vs 862, solar 112 vs 137) — those
   are different test sets. At genuinely matched config:

   | Dataset | adapter | comparable full-FT | verdict |
   |---|---|---|---|
   | electricity h=96 | 0.1438 | **0.1275** (`ctx/`, identical config) | **full-FT wins by 11.4%** |
   | traffic h=96 | 0.3661 | **0.3233** (`expand/`, smaller cap — favours adapter) | **full-FT wins by 11.7%** |
   | ETTh1 h=96 | **0.3935** | 0.4616 (`matched/`, stride 1 = ~8x more data) | adapter wins by 14.8% |
   | ETTh2 h=96 | **0.2988** | 0.3513 (same caveat) | adapter wins by 14.9% |
   | ETTm1 h=96 | **0.3187** | 0.3345 (same caveat) | adapter wins by 4.7% |
   | ETTm2 h=96 | **0.1670** | 0.2003 (same caveat) | adapter wins by 16.6% |
   | solar | 0.1718 (h=96) | 0.2240 (h=144, M=137) | **no valid comparison** |

   Corrected statement: **the adapter beats full fine-tuning on the small 7-channel ETT
   datasets (where full-FT overfits 86M parameters on a few thousand windows) and loses to it
   on the high-channel datasets.** For ETT the full-FT runs even had ~8x more training windows
   and still lost, so that half of the claim is if anything understated.

   This does not affect the headline conclusion, which rests entirely on §5.

3. `hsweep/` (epochs 2, cap 25000) is not comparable to `adapter/` (epochs 3, cap 40000). The
   adapter-vs-full-FT horizon sweep is confounded on three axes and should not be tabulated.

4. `frozen/` used 5 epochs vs 3 elsewhere — i.e. the frozen arm got *more* training and still
   came last. The ladder in §4.2 is directionally safe.

---

## 8. Supporting evidence

### 8.1 Frozen-feature diagnostic (`probe_frozen.py`, no training)

Participation ratio = effective number of active feature dimensions out of 768.
Skill ratio = ridge-probe MSE / constant-predictor MSE (lower is better).

| Dataset | Render | Participation ratio | Skill ratio |
|---|---|---|---|
| electricity | `uni_barcode` | 4.88 / 768 | 0.621 |
| electricity | `vts_2d` | 5.86 / 768 | 0.518 |
| electricity | `field_2d` | 13.07 / 768 | 0.214 |
| solar | `uni_barcode` | 7.49 / 768 | 0.259 |
| solar | `vts_2d` | 7.36 / 768 | 0.222 |
| solar | `field_2d` | 5.23 / 768 | 0.174 |

**Only 5–13 of 768 dimensions carry any variance.** Feeding VideoMAE a rendered time series
collapses its representation almost completely — a direct mechanistic explanation for why the
frozen arm (§4.2) cannot beat a seasonal mean.

### 8.2 Context length matters far more than the backbone

electricity h=96, full-FT, identical config, `--context-steps`:

| Lookback | MSE | smean |
|---|---|---|
| L = 96 | 0.1679 | 0.2362 |
| L = 384 | **0.1275** | 0.2071 |

**-24.1% from a 4x longer lookback**, vs -6.9% from the entire video branch and -2.2% from
Kinetics pretraining. This is the single most important design variable, and it has nothing to
do with video.

### 8.3 Image beats video

solar h=144, identical config, ViT-MAE (`facebook/vit-mae-base`) vs VideoMAE:

| Backbone | MSE |
|---|---|
| ViT-MAE (image) | **0.2079** |
| VideoMAE (video) | 0.2176 / 0.2240 |

Consistent with §5.3: the temporal axis is not merely useless, it is mildly harmful.

---

## 9. Infrastructure notes

Northwestern CS slurm cluster (`erebus` / `hemera` / `nyx`, 4x A40 48GB each), conda env
`wm4ts`, `transformers==4.46.3`.

Issues resolved along the way, recorded so they are not rediscovered:

- **`salloc` defaults to 1 CPU**, which OOM-kills model loading. Always pass `-c 32 --mem=96G`.
- **NFS caching served stale bytecode** after edits (md5 matched on disk but Python read the old
  file). Workaround: write to a fresh filename rather than editing in place.
- **NumPy 2.0**: `ndarray.ptp()` and `ndarray.abs()` were removed — use `np.ptp(x, 0)`,
  `np.abs(x)`.
- **VACE setup**: needs `conda install -c conda-forge libglvnd libgl` for `libGL.so.1`; comment
  out `from . import annotators` in `vace/__init__.py` to avoid the pycocotools/onnxruntime
  dependency cascade; `pip install --no-deps wan@git+https://github.com/Wan-Video/Wan2.1`.
- **VACE-14B** ships as 7 shards but `WanVace` expects a single file. Fixed by re-merging to one
  bf16 safetensors and patching `wan_vace.py:87,546` with
  `VaceWanModel.from_pretrained(ckpt, use_safetensors=True, torch_dtype=torch.bfloat16)`.
  fp32 needs 63GB and OOMs; bf16 is 34.7GB and fits on one A40.
- **LTX-Video was abandoned** (`run_ltx_ts.py` never produced a result). The full HF repo
  download filled the shared disk to 100%; after cleanup and a targeted re-download of
  `ali-vilab/VACE-LTX-Video-0.9` only, it hit an unresolvable dependency conflict —
  `EncoderDecoderCache` requires `transformers>=4.38` but the shared `open-sora` env pins
  4.37.1. Fixing it would have broken the working `WanVace` setup for other users of that env.
- `accelerate` was upgraded in the shared `open-sora` env while chasing LTX and has not been
  reverted.

---

## 10. Limitations / what was not run

- **No LoRA arm.** A reviewer will ask. It would land somewhere between LN-only (0.2508) and
  full-FT (0.1275) on electricity. It does not bear on the conclusion, because §5.1 shows the
  frozen features carry almost no usable signal regardless of how the backbone is adapted — but
  it closes an obvious hole. Roughly half a day of compute.
- **Single seed for every adapter and ablation run.** The 4-seed variance study (§4.4) was run
  only on full-FT. The §5 effects (6.9%, 2.2%) are 2–6x the measured noise floor, but multi-seed
  ablations would make them airtight.
- **No published-baseline reproduction.** VisionTS / PatchTST / iTransformer numbers were taken
  from their papers, not re-run here, and their protocols differ from ours (channel subsampling,
  window caps). Cross-paper comparison in `README.md` should be treated as indicative only.
- **A2 and A1 were each run at one or two horizons**, not a full grid.
- **VACE-14B JSON is not in the repo** — the number lives only in the cluster log.
- `hsweep/` is under-powered (2 epochs, 25000 cap) relative to the other sweeps.

---

## 11. Conclusion

Video foundation models do not usefully transfer to time-series forecasting. The evidence:

1. **Zero-shot / frozen transfer fails outright** — worse than a seasonal mean (§4.2), because
   the representation collapses to ~5–13 of 768 dimensions (§8.1).
2. **Where an adapter does work, the video branch is not what makes it work.** Deleting it costs
   at most 6.9% and often *helps* (§5.1). 98.8% of the trainable parameters live in that branch.
3. **Kinetics-400 pretraining is worth 2.2%** over a randomly-initialized backbone of the same
   shape — barely above the 1.2% seed-noise floor (§5.2).
4. **The temporal axis is dead weight.** A static image replicated 16 times matches or beats the
   video render (§5.3), and an image backbone beats the video backbone outright (§8.3).
5. **Giving video a fair fight does not help.** Multivariate field rendering contributes 0.3%,
   below noise (§5.5). Genuinely spatiotemporal data (METR-LA) gives the largest gain in the
   project but still loses to seasonal-naive on MAE and sits far from that benchmark's SOTA
   (§6.1).
6. **Generative video models are worse than trivial baselines** — 1.7x at 1.3B, 2.2x at 14B, so
   scaling does not rescue it (§6.2).
7. **Context length matters 3.5x more than the entire video pipeline** (§8.2).

The apparent competitiveness of the adapter is a strong linear baseline (NLinear + RevIN) with
a generic parameter-count bump on top. This is a well-supported negative result in the spirit of
*"Are Transformers Effective for Time Series Forecasting?"* (Zeng et al., AAAI 2023).
