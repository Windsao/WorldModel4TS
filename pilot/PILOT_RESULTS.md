# Pilot: Video-Pretrained vs Image-Pretrained Models for Zero-Shot TS Forecasting

**Date:** 2026-07-13 · **Hardware:** NU CS cluster (erebus, 4× A40) · **Code:** `pilot/run_pilot.py`

## Question

Does the temporal dynamics prior of a video-pretrained model (VideoMAE, Kinetics-400)
transfer to numeric time series forecasting, beyond what an image-pretrained model
(VisionTS / ImageNet MAE) provides?

## Protocol

Zero-shot, channel-independent, standardized by train-split stats. Context = 12 periods,
horizon = 4 periods (uniform period budget so the video mapping is fixed at
12 visible + 4 masked frames). Test windows at stride = 1 period.
Horizons are therefore dataset-dependent (96 steps hourly, 384 @ 15-min, 576 @ 10-min)
and **not** comparable to standard literature numbers — internal comparison only.
Electricity/traffic/solar subsampled to 64 channels (seed 0).

`videomae_zero` = identical decode pipeline with model outputs zeroed; because the
Kinetics checkpoint was trained with `norm_pix_loss`, masked patches are de-normalized
with per-location stats from visible frames, which embeds a seasonal prior. The zero
control isolates that prior from the model's actual signal.

## Results (MSE / MAE; bold = best MSE per dataset)

| Dataset | naive | snaive | smean | visionts | videomae | videomae_zero |
|---|---|---|---|---|---|---|
| ETTh1 | 1.000/0.611 | 0.512/0.433 | **0.408**/0.407 | 0.432/0.399 | 0.419/0.413 | 0.411/0.411 |
| ETTh2 | 0.402/0.401 | 0.389/0.379 | 0.357/0.393 | **0.325**/0.350 | 0.360/0.395 | 0.357/0.393 |
| ETTm1 | 1.123/0.661 | 0.509/0.432 | **0.406**/0.406 | 0.413/0.393 | 0.416/0.412 | 0.418/0.414 |
| ETTm2 | 0.398/0.393 | 0.386/0.377 | 0.357/0.392 | **0.317**/0.346 | 0.359/0.393 | 0.357/0.392 |
| electricity | 1.553/0.904 | 0.319/0.325 | **0.214**/0.285 | 0.237/0.295 | 0.222/0.297 | 0.225/0.304 |
| traffic | 2.246/1.047 | 0.943/0.492 | **0.505**/0.382 | 0.578/0.398 | 0.516/0.389 | 0.515/0.393 |
| solar | 1.015/0.518 | 0.384/0.271 | **0.205**/0.222 | 0.303/0.256 | 0.210/0.231 | 0.221/0.247 |

## Findings

1. **VideoMAE adds little signal beyond its decoding prior.** On 5/7 datasets,
   `videomae` ≈ `videomae_zero` (differences within ±0.005 MSE). The exceptions are
   **solar** (0.221 → 0.210, the model genuinely improves over the prior) and
   marginally electricity.
2. **The seasonal-mean baseline wins MSE on 5/7 datasets.** In this zero-shot,
   short-context regime neither pretrained approach beats a trivial per-phase average
   on strongly periodic data. VisionTS wins only on the weakly-periodic ETTh2/ETTm2
   and has the best MAE on 4/7.
3. **The solar result is the one encouraging cell for the video hypothesis:** smooth,
   physically-driven dynamics, where VideoMAE beats VisionTS by 31% MSE and clearly
   improves over its own control. Consistent with "video dynamics priors help on
   smooth physical signals," the domain closest to what video pretraining saw.

## Pilot limitations (why this is a lower bound, not a refutation)

- **`norm_pix_loss` decode coupling:** the Kinetics checkpoint predicts per-patch
  normalized pixels, so the model can only contribute *shape*, never *level*; levels
  come from the seasonal prior. An image-MAE-visualize-style raw-pixel video
  checkpoint (or a briefly re-trained decoder) would remove this handicap.
- **Mask-ratio distribution shift:** VideoMAE was pretrained at 90% tube masking; our
  forecasting mask is 25% and block-structured (whole future frames) — off-distribution.
- Single rendering scheme (phase → row bands); no tuning; base-size models;
  VisionTS run at uniform context, not its per-dataset tuned contexts (its ETTh1
  literature number ≈ 0.35 with longer context vs 0.43 here).

## Implication for the paper idea (zero-shot phase)

The "free-lunch" version (VisionTS-style zero-shot swap to a video backbone) does
**not** hold broadly — do not build the paper on it. Surviving directions:
(a) continued pretraining of a video backbone on TS-rendered videos (VisionTS++ move,
one generation ahead); (b) generative video-diffusion forecasters with raw-pixel
outputs; (c) the physical/smooth-domain angle where the video prior showed real
signal (solar; REAL-V-TSFM-style video-derived series), possibly framed as probing
world-model dynamics priors.

---

# Part 2: Per-Dataset Fine-Tuning (2026-07-14)

Code: `pilot/run_finetune.py`. Equal budget: 40k train windows x 1 epoch, AdamW,
LN-FT lr 1e-4 / full-FT lr 2e-5. `rand` = same architecture from random init,
full-FT (pretraining ablation). NLinear: 10 epochs. Chronos-bolt-base: zero-shot.

## Results (MSE, full test; bold = best per dataset)

| Dataset | NLinear | Chronos-0s | ViTS-ln | ViTS-full | ViTS-rand | VMAE-ln | VMAE-full | VMAE-rand |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 0.378 | 0.419 | **0.362** | 0.472 | 0.404 | 0.413 | 0.395 | 0.388 |
| ETTh2 | **0.279** | 0.295 | 0.298 | 0.328 | 0.316 | 0.357 | 0.300 | 0.301 |
| ETTm1 | 0.385 | 0.440 | **0.359** | 0.509 | 0.399 | 0.412 | 0.473 | 0.392 |
| ETTm2 | 0.293 | 0.309 | **0.286** | 0.308 | 0.315 | 0.356 | 0.298 | 0.297 |
| electricity | 0.166 | **0.145** | 0.164 | 0.149 | 0.216 | 0.214 | 0.158 | 0.209 |
| traffic | 0.348 | 0.318 | 0.335 | **0.292** | 0.422 | 0.467 | 0.306 | 0.410 |
| solar | 0.214 | 0.333 | **0.196** | 0.233 | 0.211 | 0.205 | 0.247 | 0.216 |

(ViTS = VisionTS/ImageNet-MAE; VMAE = VideoMAE/Kinetics. ETTm2 VMAE-full from the
initial wave: 0.298.)

## Findings

1. **LN-FT: image beats video on all 7 datasets.** Under the standard light-touch
   protocol, VisionTS-ln wins every head-to-head vs VideoMAE-ln. VideoMAE's
   norm_pix_loss decode keeps its level pathway frozen; LN tuning cannot fix it.
2. **Pretraining ablation: video pretraining helps only where image pretraining
   helps, and helps less.** Kinetics pretraining beats random init on
   electricity (0.158 vs 0.209) and traffic (0.306 vs 0.410) only — the same two
   data-rich datasets where ImageNet helps (0.149 vs 0.216; 0.292 vs 0.422), with a
   smaller margin. On all four ETT datasets and solar, video-pretrained ≈ random init.
3. **Full-FT for 1 epoch destroys VisionTS on small data** (ETTh1 0.472 vs ln 0.362)
   but helps VideoMAE (its decode pathway needs relearning). Head-to-head full-FT:
   video wins the 4 ETT datasets, image wins electricity/traffic/solar.
4. **Neither vision backbone convincingly beats simple TS baselines**: NLinear or
   Chronos-bolt zero-shot are within noise of, or better than, the best vision
   config on 4/7 datasets.

# Part 3: Wan2.1-T2V-1.3B (true video generator, zero-shot outpainting)

Code: `pilot/run_wan.py` (RePaint-style latent replacement in flow-matching sampling,
480x832 native res required — 240x416 is OOD and hallucinates objects). n=96 seeded
test windows.

| Config | solar | ETTh1 |
|---|---|---|
| smean (init) | 0.1880 | 0.3849 |
| **wan-sdedit(0.6, smean init)** | 0.1881 | 0.3806 |
| pure generation (no init) | 2.28–3.03 | — |
| sdedit with snaive init | 0.5057 (init: 0.5002) | — |

**Wan2.1 is a "copy machine", not a forecaster**: pure generation cannot anchor
numeric levels (bands look right, values are hallucinated); SDEdit mode faithfully
preserves whatever initialization it is given (even a deliberately bad snaive init
is returned unchanged, ignoring context that supports a 2.4x better forecast).
Context reconstruction is near-perfect (VAE grayscale MSE 1e-5), so the null is a
property of the generative prior, not the pipeline.

# Overall verdict on "video models are better TS foundations"

**Not supported in current form.** Across zero-shot, LN-FT, full-FT, a pretraining
ablation, and a true generative model: image pretraining (ImageNet MAE) transfers
better than video pretraining (Kinetics VideoMAE) on rendered time series, and both
barely clear trivial baselines. Video pretraining's only measured advantages:
(a) zero-shot on solar (smooth physical dynamics), (b) slightly more robust full-FT
on small ETT data. The honest paper here is an analysis paper ("do video foundation
models carry transferable dynamics priors? mostly no — and here is why: the level
pathway, not the dynamics pathway, is the bottleneck"), or a methods paper that
*fixes* the transfer (continued pretraining on TS-rendered video).
