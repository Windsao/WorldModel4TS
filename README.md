# Video Foundation Models for Time Series Forecasting

VideoMAE (a Kinetics-pretrained **video** model) becomes a competitive time-series
forecaster when the series is fed as a **period-per-frame video** and read out with a
**regression head** — beating seasonal baselines and an image-backbone control on
high-channel benchmarks.

> This branch contains only the working design (`pilot/run_field.py`) and its
> results. The exploratory phases that failed (zero-shot, pixel-reconstruction,
> Wan diffusion, continued pretraining) live on `master`.

## The design

1. **Frame = period.** A window of 16 periods → 16 frames; each frame renders one
   period's within-period waveform. Prediction lives on the **frame (temporal)
   axis** — the video model's actual competence — not on within-frame masked
   columns.
2. **Regression head, not pixel reconstruction.** The pooled spatiotemporal tokens
   go through an MLP head that outputs the forecast directly. No future frames are
   rendered → no causal-leakage surface. Context-only normalization; nearest-neighbor
   patch-aligned rendering (one period → one 16px patch column, no cross-cell mixing).
3. **Two modes:** `uni` (channel-independent — stronger) and `field`
   (multivariate-joint — rows = variables, cols = phase).

## Results (test MSE, VideoMAE-base, full fine-tune, 3 epochs)

| Dataset (ch) | smean | **VideoMAE (uni)** | ViT-MAE (image) | VideoMAE random-init |
|---|---|---|---|---|
| electricity (321) | 0.207 | **0.138 ± 0.0006** (3 seeds) | 0.214 | 0.160 |
| traffic (862) | 0.517 | **0.320 ± 0.0018** (3 seeds) | 0.500 | 0.358 |
| electricity h=192 | 0.211 | **0.157** | — | — |
| solar (137) | 0.200 | 0.218 (uni) / **0.177** (field) | 0.208 | — |
| ETTm1 (7) | 0.399 | 0.436 | — | — |

**Three pieces of evidence the result is real:**
1. **Beats baselines** on all high-channel datasets (electricity −33%, traffic −38%),
   multi-seed σ ≈ 0.001. electricity 0.138 is in specialized-SOTA range.
2. **The Kinetics prior contributes**: pretrained beats random-init 16–21% on the
   fields, ~0% on 7-channel ETTm/ETTh (the benefit scales with field structure).
3. **Video beats image on identical input**: same period-frames, same head, only the
   backbone differs — VideoMAE beats ViT-MAE 35% (electricity 0.139 vs 0.214,
   traffic 0.321 vs 0.500). Since ViT-MAE encodes the 16 frames independently while
   VideoMAE attends across them, **cross-frame temporal attention is the mechanism.**

**Boundaries (honest):** the win is clearest on hourly high-channel data; on solar
(10-min, P=144) only the `field` mode beats the baseline and video ≈ image; small
non-field ETT loses. Continued pretraining does not help once the wiring is correct.

## Run

```bash
pip install "transformers==4.46.3" torch torchvision pandas numpy   # transformers<5 required
export HF_HOME=<hf-cache>

# channel-independent (recommended), electricity, horizon 96
python pilot/run_field.py --dataset electricity --mode uni --horizon-p 4 \
    --stride 8 --max-ch 112 --epochs 3 --data-dir <data> --out-dir results

# multivariate-joint field mode
python pilot/run_field.py --dataset solar --mode field --horizon-p 1 --data-dir <data>

# image-backbone control (same input)
python pilot/run_field.py --dataset electricity --mode uni --backbone image --data-dir <data>

# arbitrary horizon in raw steps (any dataset): --horizon-steps 336
```

Key flags: `--mode {uni,field}` · `--backbone {video,image}` · `--horizon-p N` (periods)
or `--horizon-steps N` · `--max-ch N` (channel cap; use large for full channels) ·
`--stride N` (test window stride) · `--seed N` · `VMAE_CKPT=<dir>` env to swap backbone.

Data: ETT csvs from `zhouhaoyi/ETDataset`; electricity/traffic/solar from
`laiguokun/multivariate-time-series-data` (gunzip into the data dir).

## Constraint

VideoMAE requires exactly 16 frames, and we map one period per frame, so **context is
fixed at 16 periods** (dataset-dependent lookback: 384 steps hourly, 1536 for 15-min,
2304 for 10-min — comparable to VisionTS's tuned 1728–4032). The horizon is free (the
head outputs any length via `--horizon-steps`).

⚠️ **transformers < 5 required** — v5 silently re-initializes VideoMAE's attention
biases (`q_bias`/`v_bias` rename), producing garbage. The code asserts this.
