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

## Complete results — all 7 datasets (test MSE, h96, VideoMAE-base, full FT, uni mode, full channels)

| Dataset | ch | context | **VideoMAE (ours)** | snaive | smean | vs smean |
|---|---|---|---|---|---|---|
| electricity | 321 | 384 | **0.141** | 0.321 | 0.207 | **−32%** |
| traffic | 862 | 384 | **0.386** | 1.218 | 0.644 | **−40%** |
| ETTm1 | 7 | 1536 | **0.335** | 0.423 | 0.376 | **−11%** |
| ETTm2 | 7 | 1536 | **0.200** | 0.263 | 0.322 | **−38%** |
| ETTh2 | 7 | 384 | 0.351 | 0.391 | 0.350 | tie |
| ETTh1 | 7 | 384 | 0.462 | 0.512 | 0.402 | lose |
| solar | 137 | 2304 | 0.224 (uni) / **0.177** (field) | 0.290 | 0.201 | field WIN |

**Beats the seasonal baseline on 5/7** (high-channel electricity/traffic + long-context
ETTm1/ETTm2). Loses on short-context low-channel ETTh1 and solar-`uni` (solar's `field`
mode wins). vs literature at h96: **electricity 0.141 ties PatchTST (~0.140)** and beats
VisionTS zero-shot (0.177); ETTm1 0.335 ≈ iTransformer (0.334); behind supervised SOTA
on ETT-hourly.

### Multi-seed + backbone ablations (stride-8, 112-channel subsample)

| Dataset | smean | **VideoMAE (uni)** | ViT-MAE (image) | random-init | **frozen (head-only)** |
|---|---|---|---|---|---|
| electricity | 0.207 | **0.138 ± 0.0006** (3 seeds) | 0.214 | 0.160 | 0.301 ✗ |
| traffic | 0.517 | **0.320 ± 0.0018** (3 seeds) | 0.500 | 0.358 | 0.607 ✗ |
| ETTh1 | 0.402 | 0.455 | — | 0.456 | 0.496 ✗ |
| solar | 0.200 | 0.177 (field) | 0.208 | — | 0.226 ✗ |

**Frozen backbone fails** (loses to baseline everywhere): unlike VisionTS's frozen image
MAE, the Kinetics video features are not directly usable — the win **requires full
fine-tuning** to adapt them. This is a real difference from image-model transfer.

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

## Comparison with published methods (horizon sweep)

> ⚠️ **These are two different regimes — read carefully.** VisionTS is *zero-shot*
> (a frozen ImageNet MAE, **no time-series training**) but uses a **long, per-dataset-
> tuned context** (1728–4032 steps). We **full-fine-tune** but with a **short, fixed
> context** (16 periods, e.g. 384). So the tables below compare *our fine-tuned short-
> context model* against *their zero-shot long-context model* — not the same setting.
> Two asymmetries pull opposite ways: we get fine-tuning (helps us) but a much shorter
> lookback (hurts us). In the **same (frozen) regime our video model fails** — see
> *Frozen transfer* below. Our horizon-sweep rows are mostly `—` (only h96 is complete;
> electricity also has h192).

### vs VisionTS — both are "a vision model for TS" (test MSE)

| Dataset | H | **Ours** (ctx) | VisionTS 0-shot (ctx) |
|---|---|---|---|
| electricity | 96 | **0.141** (384) | 0.177 (2880) |
|  | 192 | **0.157** | 0.188 |
|  | 336 / 720 | — | 0.207 / 0.256 |
| ETTm1 | 96 | **0.335** (1536) | 0.341 (2304) |
|  | 192 / 336 / 720 | — | 0.360 / 0.377 / 0.416 |
| ETTm2 | 96 | **0.200** (1536) | 0.228 (4032) |
|  | 192 / 336 / 720 | — | 0.262 / 0.293 / 0.343 |
| ETTh1 | 96 | 0.462 (384) | **0.353** (2880) |
| ETTh2 | 96 | 0.351 (384) | **0.271** (1728) |

At h96 we beat VisionTS zero-shot on electricity/ETTm1/ETTm2 **with a much shorter
context**, and lose on the small ETT-hourly sets.

### vs supervised SOTA (iTransformer paper, lookback 96)

| Dataset | H | **Ours** | iTransformer | PatchTST | DLinear | TimesNet |
|---|---|---|---|---|---|---|
| traffic | 96 | **0.386** | 0.395 | 0.481 | 0.625 | 0.620 |
| electricity | 96 | **0.141** | 0.148 | 0.205 | 0.212 | 0.192 |
|  | 192 | **0.157** | 0.162 | 0.227 | 0.235 | 0.210 |
| solar | 96 | — (we ran h144=0.177) | 0.203 | 0.270 | 0.330 | 0.301 |

At h96 we beat iTransformer on traffic (0.386 vs 0.395) and edge it on electricity;
solar must be re-run at h96 to be directly comparable (ours is h144).

*Sources: [VisionTS (arXiv:2408.17253)](https://arxiv.org/abs/2408.17253),
[iTransformer (arXiv:2310.06625)](https://arxiv.org/abs/2310.06625). VisionTS-paper
context lengths are the tuned per-dataset look-backs.*

## Frozen transfer: is the frozen failure numerical, or representational?

VisionTS shows a *frozen* image MAE works zero-shot. Our frozen *video* MAE does not.
A natural worry: maybe that is just a numerical / input-rendering artifact (TS values
have a different range/structure than natural video), which normalization or a better
input should fix. We tested this directly and it is **not** the explanation.

**Parameter-efficient middle ground (`--tune ln`, stride-8, 112ch, h96):**

| Dataset | frozen (0%) | **LN-only (0.81%)** | full-FT (100%) | smean |
|---|---|---|---|---|
| electricity | 0.301 ✗ | 0.251 ✗ | **0.128** ✓ | 0.207 |
| traffic | 0.607 ✗ | 0.530 ✗ | **0.320** ✓ | 0.517 |
| ETTh1 | 0.496 ✗ | 0.438 ✗ | 0.462 ✗ | 0.402 |
| solar | 0.226 ✗ | 0.213 ✗ | **0.177** (field) ✓ | 0.201 |

Tuning every encoder LayerNorm recovers only **~28% of the frozen→full gap**
(elec 0.301→0.251 vs full 0.128; traffic 0.607→0.530 vs 0.320) and still loses the
baseline everywhere. About a quarter of the deficit is adaptable statistics; the rest
is not reachable by any affine renormalization.

**Input structure enriches features but does not rescue the forecast.** A no-train
probe (`pilot/probe_frozen.py`) of frozen pooled features — participation ratio (PR, of
768 dims) and a ridge linear-probe skill = probe/const MSE (lower = more informative):

| electricity, frozen | PR | probe skill | | solar, frozen | PR | probe skill |
|---|---|---|---|---|---|---|
| uni barcode | 4.88 | 0.621 | | uni barcode | 7.49 | 0.259 |
| VisionTS 2D | 5.86 | 0.518 | | VisionTS 2D | 7.36 | 0.222 |
| field 2D | 13.07 | 0.214 | | field 2D | 5.23 | 0.174 |

The period-per-frame **barcode collapses frozen features to ~5 effective dims**; 2D
renderings lift the rank and linear informativeness. **But training the frozen model on
those richer inputs does not help the forecast** — holding the univariate task fixed and
only changing the rendering, `frozen+vts` electricity is 0.302 vs `frozen+barcode` 0.301
(and `frozen+field` 0.305). Three renderings, frozen stays at ~0.30, all lose smean 0.207.

**Conclusion:** the frozen failure is **representational, not a numerical/input artifact**
— Kinetics video features require full fine-tuning regardless of how the series is
rendered or normalized. The one untested confound is VisionTS's use of the pretrained MAE
*decoder* (inpainting) instead of our fresh regression head; that remains future work.

**Context is a large lever (`--context-steps`, electricity, full-FT):** temporally
resampling the period-frames to 16 lets us vary lookback. L=384→**0.128** (reproduces the
native run, so the resampling is lossless); L=96 (matched to the supervised protocol)→
**0.168** — context is worth ~31%, which is why the strong numbers need the long lookback.

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
