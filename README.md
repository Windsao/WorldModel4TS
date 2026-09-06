# Video Foundation Models for Time Series Forecasting — a negative result

**Question:** does the temporal-dynamics prior of a video foundation model (VideoMAE /
Kinetics-400, Wan2.1-VACE) transfer to time-series forecasting?

**Answer: no.** Across 101 runs spanning discriminative and generative backbones, four
fine-tuning regimes, seven standard benchmarks, a genuinely spatiotemporal benchmark, and two
model scales, the pretrained *video* prior contributes at most **~2% MSE** over an
identically-shaped randomly-initialized backbone — high-channel datasets only, and it is
**actively harmful** on the small ETT sets. Generative video-editing models are **1.7–2.2x
worse than a trivial seasonal-mean baseline**.

> **Read [`PROGRESS.md`](PROGRESS.md) for the full record** — all 101 runs, per-file code
> pointers, reproduction commands, and the config-matching audit.

> **2026-09-06 update — the negative above is regime-specific.** The answer "no" holds for
> *frozen, zero-shot* use of the backbone. In the *continual-pretraining* regime it flips:
> after identical time-series continual pretraining (LOTSA subset, 20k steps, four arms that
> differ only in encoder initialisation), the Kinetics-VideoMAE-initialised forecaster beats
> VisionTS (Q = 0.787) and improves the strongest video-free method, the ZL-051 prior blend
> (Q = 0.858, CI excluding 1 on 4/6 datasets), with the ordering video init > image init >
> random. An autoregressive video model (MAGI-1) extrapolates rendered motion zero-shot but
> still loses to the blend on real data. Report: [`VIDEO_TS_RESCUE_RESULTS.md`](VIDEO_TS_RESCUE_RESULTS.md);
> plan and literature: [`VIDEO_TS_LITERATURE_AND_RESCUE_PLAN.md`](VIDEO_TS_LITERATURE_AND_RESCUE_PLAN.md).

### What we thought we had, and what it actually was

An earlier version of this README claimed VideoMAE "becomes a competitive time-series
forecaster." That claim did not survive its own ablations. The corrections:

| Earlier claim | Status |
|---|---|
| "Beats seasonal baselines on 5/7 datasets" | **Holds** for the raw numbers, but see below for what is doing the work. |
| "The Kinetics prior contributes 16–21%" | **Regime-dependent.** True under 1-epoch full-FT; only **2.2%** under a frozen backbone, vs a 1.2% seed-noise floor. And image pretraining helps *more* on the same datasets. |
| "Video beats image by 35%; cross-frame temporal attention is the mechanism" | **Withdrawn.** The image control ran at `--stride 16` vs video's `--stride 8` — half the training windows. On the one matched comparison we have (solar), **image beats video** (0.208 vs 0.218). `master`'s own controlled study reached the same verdict: *"LN-FT: image beats video on all 7 datasets."* |
| "The win requires full fine-tuning to adapt the features" | **Holds**, and is itself the problem — see §Frozen transfer. |
| "A frozen-backbone adapter matches/beats full-FT at 4% params" (commit `48791c3`) | **Half true.** It beats full-FT on the 7-channel ETT sets and **loses** to it on the high-channel sets. The original comparison used different channel counts. See `PROGRESS.md` §7. |

**The decisive ablation:** delete the video branch from the adapter (`--no-vbranch`), leaving a
per-channel NLinear on RevIN-normalized context. It costs at most **6.9%** — and on ETT it
*improves* the forecast by up to 14.4%. 98.8% of the trainable parameters live in the video
branch. The linear layer is doing the work.

## The design

1. **Frame = period.** A window of 16 periods -> 16 frames; each frame renders one period's
   within-period waveform. Prediction lives on the frame (temporal) axis, not on within-frame
   masked columns.
2. **Regression head, not pixel reconstruction.** Pooled spatiotemporal tokens go through an
   MLP head that outputs the forecast directly. No future frames are rendered, so there is no
   causal-leakage surface. Context-only normalization; nearest-neighbour patch-aligned
   rendering (one period -> one 16px patch column, no cross-cell mixing).
3. **Two modes:** `uni` (channel-independent) and `field` (multivariate-joint — rows =
   variables, cols = phase).
4. **Four tuning regimes** (`--tune`): `full`, `frozen` (head only), `ln` (LayerNorm affine),
   `lora` (rank-r on attention q/v), `adapter` (frozen backbone + NLinear + cross-attn readout).

## Headline table — all 7 datasets (test MSE, h96, VideoMAE-base, **full FT**, uni, all channels)

| Dataset | ch | context | VideoMAE | snaive | smean | vs smean |
|---|---|---|---|---|---|---|
| electricity | 321 | 384 | **0.141** | 0.321 | 0.207 | −32% |
| traffic | 862 | 384 | **0.386** | 1.218 | 0.644 | −40% |
| ETTm1 | 7 | 1536 | **0.335** | 0.423 | 0.376 | −11% |
| ETTm2 | 7 | 1536 | **0.200** | 0.263 | 0.322 | −38% |
| ETTh2 | 7 | 384 | 0.351 | 0.391 | 0.350 | tie |
| ETTh1 | 7 | 384 | 0.462 | 0.512 | 0.402 | lose |
| solar | 137 | 2304 | 0.224 | 0.290 | 0.201 | lose |

> ⚠️ These numbers are real, but they are **not evidence that video pretraining works**. At a
> matched 112-channel config, a pure NLinear+RevIN control with **no video model at all** gets
> electricity 0.154 and traffic 0.390 — i.e. the entire video pipeline buys ~7% over a linear
> layer, of which ~2% is attributable to Kinetics pretraining.

## Ablations that decide the question

All comparisons below are **exactly config-matched** — same dataset, channels, context,
horizon, stride, epochs, cap, seed. Only the named flag differs. (h=96, 112-channel,
stride 8, cap 40000, 3 epochs.)

### 1. Is the video branch doing anything? (`--no-vbranch`)

| | with video | pure NLinear+RevIN | video contribution |
|---|---|---|---|
| electricity | 0.1438 | 0.1544 | −6.9% |
| traffic | 0.3661 | 0.3896 | −6.0% |
| ETTh1 h96 | 0.3935 | 0.4026 | −2.3% |
| ETTh1 h336 | 0.4823 | 0.4499 | **+7.2% worse** |
| ETTh2 h96 | 0.2988 | 0.2918 | **+2.4% worse** |
| ETTh2 h336 | 0.4270 | 0.3733 | **+14.4% worse** |

### 2. What is Kinetics pretraining worth? (`--pretrained 0`)

electricity, h96, identical config:

```
pure NLinear + RevIN                   0.1544
  + randomly-initialised VideoMAE      0.1470   (−4.8%)
  + Kinetics-400 pretrained weights    0.1438   (−2.2%)
```

**Only 30% of the video branch's gain comes from pretraining**; the rest is the generic
capacity effect of adding 3M parameters. Seed-noise floor is 1.2% (4 seeds).

### 3. Does the temporal axis matter? (`--render vts`)

| | period-per-frame video | one static 2D image x16 |
|---|---|---|
| electricity | 0.1438 | **0.1421** |
| traffic | 0.3661 | 0.3657 |

Replacing the video with a **single static image replicated 16 times** is as good or better.
The motion prior — the only reason to prefer a video model over an image model — contributes
nothing. Consistent with the solar image-vs-video control (ViT-MAE 0.208 beats VideoMAE 0.218)
and with `master`'s LN-FT study (image wins 7/7).

### 4. Giving video a fair fight does not help

- **Multivariate field render** (rows = variables, so frames show a real cross-channel field
  evolving — structure a channel-independent linear model provably cannot capture): video
  contributes **0.3–0.4%**, below the noise floor, on electricity / solar / traffic.
- **Genuinely spatiotemporal data** (METR-LA, 207 sensors with real coordinates): the largest
  video contribution in the project (MSE −11.6% vs NLinear at h=12) — but it **loses to
  seasonal-naive on MAE at both horizons** and sits far from that benchmark's SOTA
  (DCRNN MAE 2.77/3.15 vs our 3.86/5.32).
- **Generative video-editing models**, rendering context and inpainting the future:

  | Model | MSE | smean | |
  |---|---|---|---|
  | Wan2.1 RePaint | — | — | degenerates to a copy machine |
  | Wan2.1-VACE-1.3B | 0.548 | 0.323 | **1.70x worse** |
  | Wan2.1-VACE-14B | 0.634 | 0.290 | **2.19x worse** |

  A 10x larger model is *worse*, so this is not a scale problem.

### 5. Context length matters far more than the backbone

electricity h96, full-FT, identical config, varying `--context-steps`:

| Lookback | MSE |
|---|---|
| L = 96 | 0.1679 |
| L = 384 | **0.1275** |

**−24% from a 4x longer lookback**, vs −6.9% from the entire video branch and −2.2% from
Kinetics pretraining. The single most important design variable has nothing to do with video.

## Frozen transfer: representational, not a numerical artifact

VisionTS shows a *frozen* image MAE works zero-shot. Our frozen *video* MAE does not.

| Dataset (h96, 112ch) | frozen (0%) | LN-only (0.81%) | adapter (3.3%) | full-FT (100%) | smean |
|---|---|---|---|---|---|
| electricity | 0.301 ✗ | 0.251 ✗ | 0.144 | **0.128** | 0.207 |
| traffic | 0.607 ✗ | 0.530 ✗ | 0.366 | **0.323** | 0.517 |
| ETTh1 | 0.496 ✗ | 0.438 ✗ | **0.394** | 0.462 | 0.402 |

Frozen and LN-only **lose to a seasonal mean everywhere**. A no-train probe
(`pilot/probe_frozen.py`) of the frozen pooled features shows why — participation ratio (of 768
dims) and ridge linear-probe skill (probe/const MSE, lower = more informative):

| electricity | PR | skill | | solar | PR | skill |
|---|---|---|---|---|---|---|
| uni barcode | 4.88 | 0.621 | | uni barcode | 7.49 | 0.259 |
| VisionTS 2D | 5.86 | 0.518 | | VisionTS 2D | 7.36 | 0.222 |
| field 2D | 13.07 | 0.214 | | field 2D | 5.23 | 0.174 |

**Only 5–13 of 768 dimensions carry variance.** Richer 2D renderings lift the rank but do not
rescue the forecast: holding the task fixed and changing only the rendering, frozen electricity
is 0.302 (`vts`) / 0.301 (barcode) / 0.305 (field) — three renderings, all stuck at ~0.30, all
losing to smean 0.207. The failure is representational, not a normalization artifact.

## Comparison with published methods

> ⚠️ **Not a like-for-like comparison — do not cite these as wins.** Three mismatches:
> (1) VisionTS is *zero-shot* with a long tuned context (1728–4032); we full-fine-tune with a
> short fixed context (384–2304). (2) iTransformer/PatchTST numbers are at **lookback 96**;
> ours use 384–1536, and §5 above shows lookback alone is worth 24%. (3) We subsample channels
> and use stride-8 test windows in most runs. We did **not** re-run any baseline ourselves.

### vs VisionTS (test MSE)

| Dataset | H | Ours (ctx) | VisionTS 0-shot (ctx) |
|---|---|---|---|
| electricity | 96 | 0.141 (384) | 0.177 (2880) |
|  | 192 | 0.157 | 0.188 |
| ETTm1 | 96 | 0.335 (1536) | 0.341 (2304) |
| ETTm2 | 96 | 0.200 (1536) | 0.228 (4032) |
| ETTh1 | 96 | 0.462 (384) | **0.353** (2880) |
| ETTh2 | 96 | 0.351 (384) | **0.271** (1728) |

### vs supervised methods (iTransformer paper, **their** lookback 96)

| Dataset | H | Ours (ctx 384) | iTransformer | PatchTST | DLinear | TimesNet |
|---|---|---|---|---|---|---|
| traffic | 96 | 0.386 | 0.395 | 0.481 | 0.625 | 0.620 |
| electricity | 96 | 0.141 | 0.148 | 0.205 | 0.212 | 0.192 |
|  | 192 | 0.157 | 0.162 | 0.227 | 0.235 | 0.210 |

*Sources: [VisionTS (arXiv:2408.17253)](https://arxiv.org/abs/2408.17253),
[iTransformer (arXiv:2310.06625)](https://arxiv.org/abs/2310.06625).*

## How to run

### Requirements
- Python >= 3.9, one GPU (~16 GB for `uni` mode; ~32 GB for the `image` backbone).
- **`transformers` must be < 5** — v5 silently re-initializes VideoMAE's attention biases
  (`q_bias`/`v_bias` rename), producing garbage. The code asserts this.

```bash
pip install "transformers==4.46.3" torch torchvision pandas numpy einops requests
```

### Data
```bash
mkdir -p data && cd data
for f in ETTh1 ETTh2 ETTm1 ETTm2; do
  curl -sLO https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/$f.csv
done
base=https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master
curl -sL $base/electricity/electricity.txt.gz | gunzip > electricity.txt
curl -sL $base/traffic/traffic.txt.gz          | gunzip > traffic.txt
curl -sL $base/solar-energy/solar_AL.txt.gz    | gunzip > solar_AL.txt
cd ..
export HF_HOME=$PWD/hf_cache        # first run pulls MCG-NJU/videomae-base (~350 MB)
export CUDA_VISIBLE_DEVICES=0
```

### The main run
```bash
python pilot/run_field.py --dataset electricity --mode uni \
    --horizon-steps 96 --stride 8 --max-ch 112 --epochs 3 \
    --data-dir ./data --out-dir ./results
```
Prints the config, per-epoch train MSE, then final metrics, and writes a JSON to `--out-dir`
containing the model metric plus `snaive`/`smean` baselines on the same windows.

### The ablations that matter
```bash
# frozen-backbone adapter
python pilot/run_field.py --dataset electricity --mode uni --tune adapter ...
# ... minus the video branch  <- the decisive control
python pilot/run_field.py --dataset electricity --mode uni --tune adapter --no-vbranch ...
# is the Kinetics prior helping?
python pilot/run_field.py --dataset electricity --mode uni --tune adapter --pretrained 0 ...
# does the temporal axis matter? (static 2D image x16)
python pilot/run_field.py --dataset electricity --mode uni --tune adapter --render vts ...
# LoRA on attention q/v
python pilot/run_field.py --dataset electricity --mode uni --tune lora --lora-r 8 --lr 5e-4 ...
# image-backbone control (use the SAME --stride as the video run)
python pilot/run_field.py --dataset electricity --mode uni --backbone image --stride 8 ...
# no-train frozen-feature diagnostic
python pilot/probe_frozen.py --dataset electricity --n 1500
```

### All flags
| flag | meaning | default |
|---|---|---|
| `--dataset` | ETTh1/ETTh2/ETTm1/ETTm2/electricity/traffic/solar | required |
| `--mode` | `uni` (channel-independent) or `field` (joint) | `field` |
| `--backbone` | `video` (VideoMAE) or `image` (ViT-MAE control) | `video` |
| `--tune` | `full` / `frozen` / `ln` / `lora` / `adapter` | `full` |
| `--lora-r`, `--lora-alpha` | LoRA rank / scaling (`--tune lora`) | 8 / 16 |
| `--no-vbranch` | adapter ablation: delete the video branch | off |
| `--fusion` | `add` (residual) or `film` (video modulates NLinear) | `add` |
| `--render` | `period` (video) or `vts` (static 2D image x16) | `period` |
| `--pretrained` | 0 = random-init backbone | 1 |
| `--horizon-p` / `--horizon-steps` | horizon in periods / raw steps | 4 / 0 |
| `--context-steps` | lookback in raw steps (multiple of P); resampled to 16 frames | 0 (= 16*P) |
| `--max-ch` | channel cap (large => all channels) | 112 |
| `--stride` | test-window stride (1 = matched-literature) | 1 |
| `--epochs` / `--lr` / `--batch` | schedule | 3 / 5e-5 / 16 |
| `--ft-cap` | max training windows | 40000 |
| `--seed` | run seed | 0 |
| `VMAE_CKPT=<dir>` | env var for a different VideoMAE checkpoint | HF default |

Other entry points: `pilot/run_stvid.py` (METR-LA spatiotemporal),
`pilot/run_vace_ts.py` (generative Wan-VACE), `pilot/probe_frozen.py` (frozen-feature probe).

## Constraint

VideoMAE requires exactly 16 frames. By default we map one period per frame, fixing context at
16 periods; `--context-steps` decouples this by temporally resampling G period-frames to 16.

⚠️ **transformers < 5 required** (see above). Anyone adding LoRA should note that
`VideoMAESelfAttention.forward` calls `F.linear(x, self.query.weight)` directly rather than
`self.query(x)` — wrapping the `nn.Linear` module is **silently bypassed**. We attach LoRA as a
weight *parametrization* instead (`run_field.py:263`).

## Verdict

The apparent competitiveness is a strong linear baseline (NLinear + RevIN) plus a generic
parameter-count bump. Video pretraining, video architecture, video-scale generative models, and
even genuinely spatiotemporal data all fail to add meaningful signal. This is a negative result
in the spirit of *"Are Transformers Effective for Time Series Forecasting?"* (Zeng et al., AAAI
2023). Full record and caveats: [`PROGRESS.md`](PROGRESS.md).
