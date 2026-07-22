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

## How to run (step by step)

### Requirements
- Python ≥ 3.9, one GPU (≈16 GB is enough for `uni` mode; ≈32 GB for the `image`
  backbone).
- **`transformers` must be < 5** (v5 silently breaks VideoMAE — see the warning at
  the bottom). The code asserts this and will stop otherwise.

### 1. Install
```bash
pip install "transformers==4.46.3" torch torchvision pandas numpy einops requests
```

### 2. Get the data
Download the 7 benchmarks into one folder, e.g. `./data`:
```bash
mkdir -p data && cd data
# ETT (4 files)
for f in ETTh1 ETTh2 ETTm1 ETTm2; do
  curl -sLO https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/$f.csv
done
# electricity / traffic / solar (LSTNet versions)
base=https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master
curl -sL $base/electricity/electricity.txt.gz | gunzip > electricity.txt
curl -sL $base/traffic/traffic.txt.gz          | gunzip > traffic.txt
curl -sL $base/solar-energy/solar_AL.txt.gz    | gunzip > solar_AL.txt
cd ..
```
The `--data-dir` you pass to the script is this `./data` folder. File names must match
the table in `pilot/run_field.py` (`ETTh1.csv`, `electricity.txt`, `solar_AL.txt`, …).

### 3. Set the HuggingFace cache (so the VideoMAE checkpoint downloads somewhere sane)
```bash
export HF_HOME=$PWD/hf_cache        # first run downloads MCG-NJU/videomae-base (~350 MB)
export CUDA_VISIBLE_DEVICES=0
```

### 4. Run — the main positive result (electricity, channel-independent, horizon 96)
```bash
python pilot/run_field.py \
    --dataset electricity \
    --mode uni \
    --horizon-p 4 \          # 4 periods × 24 h = horizon 96
    --stride 8 \
    --max-ch 112 \           # subsample channels (use a big number for ALL channels)
    --epochs 3 \
    --data-dir ./data \
    --out-dir ./results
```

### 5. What you should see
The script prints the config, per-epoch train MSE, then the final metrics, and writes
a JSON to `--out-dir`:
```
dataset=electricity mode=uni M=112 P=24 context=384 horizon=96 train=40000 test=...
[done] snaive     {'MSE': 0.34, ...}
[done] smean      {'MSE': 0.207, ...}
[info] epoch 0 train MSE 0.19
...
[done] video_uni_s0   MSE=0.1379 MAE=0.2425     ← our method (≈0.13–0.14, beats smean 0.207)
```
`results/field_electricity_uni_video_h96_s0.json` holds the numbers.

### 6. Other configurations
```bash
# multivariate-joint "field" mode (rows=variables, cols=phase) — better on solar
python pilot/run_field.py --dataset solar --mode field --horizon-p 1 --data-dir ./data

# image-backbone control (same input, ViT-MAE instead of VideoMAE) — should be ~35% worse
python pilot/run_field.py --dataset electricity --mode uni --backbone image --data-dir ./data

# random-init ablation (is the Kinetics prior helping?)
python pilot/run_field.py --dataset electricity --mode uni --pretrained 0 --data-dir ./data

# arbitrary horizon in raw steps, any dataset (head outputs any length)
python pilot/run_field.py --dataset ETTm1 --mode uni --horizon-steps 336 --data-dir ./data

# full-channel, matched-literature protocol (stride 1, all channels)
python pilot/run_field.py --dataset electricity --mode uni --horizon-p 4 \
    --stride 1 --max-ch 1000 --data-dir ./data
```

### All flags
| flag | meaning | default |
|---|---|---|
| `--dataset` | ETTh1/ETTh2/ETTm1/ETTm2/electricity/traffic/solar | required |
| `--mode` | `uni` (channel-independent, recommended) or `field` (joint) | `field` |
| `--backbone` | `video` (VideoMAE) or `image` (ViT-MAE control) | `video` |
| `--horizon-p` | horizon in periods (`4`→96 for hourly) | 4 |
| `--horizon-steps` | horizon in raw steps; overrides `--horizon-p` if > 0 | 0 |
| `--max-ch` | channel cap (large ⇒ all channels) | 112 |
| `--stride` | test-window stride (1 = matched-literature) | 1 |
| `--epochs` / `--lr` / `--batch` | full fine-tune schedule | 3 / 5e-5 / 16 |
| `--seed` | run seed (for error bars) | 0 |
| `--ft-cap` | max training windows | 40000 |
| `VMAE_CKPT=<dir>` | env var to load a different VideoMAE checkpoint | HF default |

Each run writes `field_<dataset>_<mode>_<backbone>_h<horizon>_s<seed>.json` with the
model metric plus `snaive`/`smean` baselines on the same windows.

## Constraint

VideoMAE requires exactly 16 frames, and we map one period per frame, so **context is
fixed at 16 periods** (dataset-dependent lookback: 384 steps hourly, 1536 for 15-min,
2304 for 10-min — comparable to VisionTS's tuned 1728–4032). The horizon is free (the
head outputs any length via `--horizon-steps`).

⚠️ **transformers < 5 required** — v5 silently re-initializes VideoMAE's attention
biases (`q_bias`/`v_bias` rename), producing garbage. The code asserts this.
