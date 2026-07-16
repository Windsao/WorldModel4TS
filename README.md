# WorldModel4TS: Do Video Foundation Models Transfer to Time Series?

A systematic empirical study of whether **video-pretrained models** (VideoMAE,
Wan2.1) are better foundations for **numeric time series forecasting** than
image-pretrained models (VisionTS / ImageNet MAE) or TS baselines — plus ongoing
work on *making* the transfer work via continued pretraining on synthetic
TS-rendered videos.

**TL;DR (as of 2026-07-15):** naive transfer of video models to time series
**fails** under every protocol we tested — zero-shot, LayerNorm fine-tuning, full
fine-tuning, pretraining ablations, literature-level long-context protocols, and
settings deliberately favorable to video priors. The image-pretrained VisionTS
recipe dominates its video counterpart on all 7 datasets. We traced the failure
to three mechanisms (level-pathway blindness, layout/content OOD, static-frame
rendering) and show the first positive signal from **continued pretraining of
VideoMAE on synthetic TS-rendered videos** ("VideoMAE-TS"), which is training as
this README is written.

## Setup

- 7 datasets: ETTh1/h2, ETTm1/m2 (standard borders), electricity, traffic,
  solar (LSTNet versions, 64-channel seeded subsample for the big ones)
- Channel-independent, standardized by train-split stats; MSE/MAE on test split
- Models: VisionTS (`mae_base`, ImageNet), VideoMAE (`MCG-NJU/videomae-base`,
  Kinetics-400), Wan2.1-T2V-1.3B (Diffusers), NLinear, Chronos-bolt-base
- Hardware: 4x NVIDIA A40

> ⚠️ **transformers < 5 required for VideoMAE.** transformers 5.x renames
> `q_bias`/`v_bias` and silently re-initializes all attention biases of
> VideoMAE checkpoints (encoder *and* decoder) — models load and run but
> produce garbage. Pinned: `transformers==4.46.3`. See the assert in
> `pilot/run_pilot.py`.

## Phase 1 — Zero-shot, short context (12 periods -> 4 periods)

Render each period as one frame (phase -> rows); forecast = reconstruction of 4
masked future frames. `videomae_zero` = same decode with model output zeroed
(isolates the seasonal prior embedded in the norm_pix_loss de-normalization).

| MSE | naive | snaive | smean | VisionTS | VideoMAE | VideoMAE-zero |
|---|---|---|---|---|---|---|
| ETTh1 | 1.000 | 0.512 | **0.408** | 0.432 | 0.419 | 0.411 |
| ETTh2 | 0.402 | 0.389 | 0.357 | **0.325** | 0.360 | 0.357 |
| ETTm1 | 1.123 | 0.509 | **0.406** | 0.413 | 0.416 | 0.418 |
| ETTm2 | 0.398 | 0.386 | 0.357 | **0.317** | 0.359 | 0.357 |
| electricity | 1.553 | 0.319 | **0.214** | 0.237 | 0.222 | 0.225 |
| traffic | 2.246 | 0.943 | **0.505** | 0.578 | 0.516 | 0.515 |
| solar | 1.015 | 0.384 | **0.205** | 0.303 | **0.210** | 0.221 |

**Finding:** VideoMAE ≈ its own zero-control on 5/7 datasets — the Kinetics
prior contributes almost nothing. The solar "win" over VisionTS later turned out
to be a short-context artifact (Phase 4).

## Phase 2 — Per-dataset fine-tuning (equal budget: 40k windows x 1 epoch)

`rand` = same architecture from random init (pretraining ablation).

| MSE | NLinear | Chronos-0s | ViTS-ln | ViTS-full | ViTS-rand | VMAE-ln | VMAE-full | VMAE-rand |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 0.378 | 0.419 | **0.362** | 0.472 | 0.404 | 0.413 | 0.395 | 0.388 |
| ETTh2 | **0.279** | 0.295 | 0.298 | 0.328 | 0.316 | 0.357 | 0.300 | 0.301 |
| ETTm1 | 0.385 | 0.440 | **0.359** | 0.509 | 0.399 | 0.412 | 0.473 | 0.392 |
| ETTm2 | 0.293 | 0.309 | **0.286** | 0.308 | 0.315 | 0.356 | 0.298 | 0.297 |
| electricity | 0.166 | **0.145** | 0.164 | 0.149 | 0.216 | 0.214 | 0.158 | 0.209 |
| traffic | 0.348 | 0.318 | 0.335 | **0.292** | 0.422 | 0.467 | 0.306 | 0.410 |
| solar | 0.214 | 0.333 | **0.196** | 0.233 | 0.211 | 0.205 | 0.247 | 0.216 |

**Findings:** (1) LN-FT: image beats video on **all 7** datasets. (2) Kinetics
pretraining helps only on electricity/traffic (+25% vs random init) — the same
data-rich datasets where ImageNet helps *more* (+31%). On ETT x4 + solar,
video-pretrained ≈ random init. (3) 1-epoch full-FT destroys VisionTS on small
data but helps VideoMAE (its decode pathway must be relearned).

## Phase 3 — Wan2.1-T2V-1.3B, a true video generator (zero-shot outpainting)

RePaint-style latent replacement inside flow-matching sampling
(`pilot/run_wan.py`); native 480x832 required (240x416 is OOD — the model
hallucinates objects instead of band patterns). n=96 seeded test windows.

| MSE | solar | ETTh1 |
|---|---|---|
| smean (init) | 0.1880 | 0.3849 |
| **Wan SDEdit(σ=0.6, smean init)** | 0.1881 | 0.3806 |
| Wan pure generation | 2.28–3.03 | — |
| Wan SDEdit with *snaive* init | 0.5057 (init: 0.5002) | — |

**Finding:** Wan2.1 is a **copy machine, not a forecaster**: pure generation
cannot anchor numeric levels; SDEdit faithfully returns whatever initialization
it is given — even a deliberately bad one, ignoring context that supports a
2.4x better forecast. Context reconstruction is near-perfect (grayscale MSE
1e-5), so this is a property of the generative prior, not the pipeline.

## Phase 4 — Literature-level protocol (VisionTS paper settings)

Long context (108 periods ≈ VisionTS's tuned 2880 for hourly), horizon 96/336,
test stride 1. LC-VMAE: each frame = a VisionTS-style 14-period grid
(1 period = 1 patch column), 8 content steps x 2 frames; forecast = masked right
columns of the last tubelet. This *reproduces the published VisionTS zero-shot
numbers* (ETTh1-96: 0.355 vs ≈0.35 in the paper), so the comparison lives on the
same footing as the accepted literature.

| MSE (stride-1) | ETTh1 h96 | ETTh1 h336 | ETTm2 h96 | solar h144 | electricity h96 |
|---|---|---|---|---|---|
| snaive / smean | 0.512 / 0.565 | 0.650 / 0.567 | 0.263 / 0.664 | 0.289 / 0.231 | 0.318 / 0.378 |
| **VisionTS zero-shot** | **0.355** | **0.403** | 0.192 | 0.191 | 0.188 |
| VisionTS LN-FT | 0.379 | 0.403 | **0.187** | 0.195 | **0.152** |
| VisionTS full-FT | 0.462 | 0.493 | — | — | — |
| LC-VMAE zero-shot | 0.566 | 0.571 | 0.666 | 0.232 | 0.380 |
| LC-VMAE zero-control | 0.572 | 0.573 | 0.668 | 0.245 | 0.390 |
| LC-VMAE LN-FT | 0.561 | 0.567 | 0.636 | 0.229 | — |
| LC-VMAE full-FT | 0.454 | 0.519 | — | — | — |

(ETTm2 stride 2, solar/electricity stride 8; "—" = run in progress/killed by a
node policy change, values to be filled.)

**Findings:** (1) The long-context protocol unlocks VisionTS (0.432 -> 0.355 on
ETTh1) but does **not** unlock VideoMAE — zero-shot still equals its
zero-control everywhere. (2) The horizon-336 arm — deliberately favorable to
temporal priors — goes to the image model by an even larger margin. (3) The
Phase-1 solar "advantage" disappears once VisionTS gets long context.

## Phase 5 — Motion-native "scrolling" rendering (ETTh1 probe, stride 16)

Frames slide over the series like a camera pan (2 periods/frame); forecast =
newly revealed right-edge content of the final frames.

| scroll_zs | scroll_zero | scroll LN-FT | scroll full-FT | VisionTS zs (same ctx) |
|---|---|---|---|---|
| 0.493 | 0.436 | 0.401 | 0.403 | **0.375** |

**Finding:** motion alone does not unlock zero-shot transfer either (model still
underperforms its own control zero-shot).

## Phase 6 (ongoing) — VideoMAE-TS: continued pretraining on synthetic TS videos

The VisionTS++ move, one modality up. Continue VideoMAE pretraining on
**infinite synthetic series** (harmonic seasonality + slow components + trends +
AR noise + level shifts + spikes) rendered as videos in both layouts, with 50%
forecast-shaped masks / 50% native tube masks, and **`norm_pix_loss=False`** so
the model learns to predict raw pixels — directly repairing the diagnosed
level-pathway blindness. Purely synthetic -> zero benchmark overlap.
(`pilot/pretrain_vmae_ts.py`, 20k steps x batch 32, 2x A40)

> **Leakage audit (2026-07-16):** the original synthetic renderer normalized
> each clip using statistics from the complete clip, including masked forecast
> values. Visible pixels therefore contained future level/variance information.
> The code now uses visible-context statistics for forecast masks and a separate
> non-rendered burn-in prefix for random masks. All checkpoints and numbers from
> before this fix must be discarded and retrained.

The now-invalid checkpoint @ step 2500 had reported:

| | model | blank-control | Δ |
|---|---|---|---|
| LC layout | 0.740 | 1.083 | **model adds +31% signal** |
| scroll layout | 0.675 | 0.868 | **model adds +22% signal** |

These values are retained only as an audit record and are not evidence of
zero-shot forecasting performance.

Also tested: channel augmentation (R=raw, G=first difference, B=expanding mean)
— no zero-shot gain with the Kinetics checkpoint (0.569 vs 0.566 grayscale).

## Fast experimental arm — lag matrices + frozen/LoRA VideoMAE

`pilot/run_lagged_lora.py` is a deliberately cheap alternative to continued
pretraining. It keeps the Kinetics checkpoint frozen and compares it with
query/value LoRA (rank 4 by default; no learned input or forecast head). Each of
12 visible frames is a rolling lag matrix whose newest length-P window occupies
the bottom patch row. RGB carries three history scales: raw-step stride 1,
quarter-period stride P/4, and full-period stride P. Four future frames are fully
masked and their reconstructed bottom rows give the next four periods.

The default quick protocol uses 2,048 seeded training windows, 1,024 seeded test
windows, and one LoRA epoch. It does not use a VideoMAE-TS checkpoint and is not
included in the result tables above yet.

## Strict two-MLP frozen-backbone adapter

`pilot/run_mlp_adapter.py` implements the constrained architecture directly:

```
numeric tubelet features -> ingest MLP -> frozen VideoMAE encoder -> forecast MLP
```

The ingest MLP is initialized to the checkpoint's native constant-patch 3D
convolution and learns a nonlinear residual. Gradients pass through all 12
encoder blocks, but all 86.2M VideoMAE parameters remain frozen. The forecast
MLP receives only the final encoder tokens; there is no raw-history,
patch-embedding, inverse-projection, or ridge skip.

Leakage-audited ETTh1 pilots (context 600, 4,096 train windows, full
validation/test at stride 24) score **0.4188 / 0.4300** MSE/MAE at horizon 96
and **0.4728 / 0.4654** at horizon 192. The results are stored in
`pilot/results_mlp_adapter/`.

## VisionTS image -> static VideoMAE -> MLP forecast

`pilot/run_visionts_static_adapter.py` replaces the hand-built phase tubelets
with the installed VisionTS 1.0.1 preprocessing exactly: context-only
normalization, periodic folding, bilinear resizing to four observed patch
columns, and ten zero forecast columns. Its output is one 224x224 grayscale
image. VideoMAE's native temporal convolution has tubelet size 2, so a literal
one-frame tensor is invalid; the runner duplicates the image into the smallest
legal static clip, two identical frames and therefore exactly one tubelet.

The trainable pixel MLP maps each grayscale value to VideoMAE's RGB input. The
frozen patch projection and all 12 frozen encoder blocks produce 196 final
tokens, which are the only input to the forecast MLP. There is no raw-series or
embedding skip. The renderer is bit-exact against VisionTS and has a regression
test proving that changing future values cannot alter its image or statistics.

Initial seed-0 ETTh1 results (MSE / MAE; superseded by the multi-seed table below):

| horizon | strict phase MLP | static-image VideoMAE | VisionTS zero-shot | Chronos-Bolt-base |
|---:|---:|---:|---:|---:|
| 96 | 0.4188 / 0.4300 | **0.3875** / 0.4195 | 0.3916 / 0.3810 | 0.3899 / **0.3809** |
| 192 | 0.4728 / 0.4654 | **0.4225** / 0.4455 | 0.4266 / **0.4091** | 0.4381 / 0.4102 |

The static-image adapter is competitive on MSE and is best in both rows, but it
does not win MAE. Full configs and results are in `pilot/results_static_image/`.

## Residual dyadic RGB time augmentation

The expanded benchmark adds ETTm1 (15-minute electricity-transformer data) and
Weather (10-minute meteorology), both from VisionTS's official six-dataset
zero-shot suite. `pilot/download_benchmark_data.py` fetches the exact TSLib
files. Every runner now supports periods 24, 96, and 144, and the original
ETTh1 phase tokenization remains bit-exact.

`pilot/run_temporal_image_adapter.py` contains six causal, dataset-independent
ways to expose time to VideoMAE. The selected method is **residual dyadic RGB**:

1. subsample the complete observed context at strides 1, 2, and 4;
2. render each scale with the exact VisionTS transform;
3. place the three images in RGB;
4. initialize the ingest MLP to repeat only stride 1 as grayscale, exactly
   reproducing the static input, while zero-initialized residual weights learn
   whether strides 2/4 help;
5. duplicate the image only because VideoMAE requires two frames per tubelet.

The strides are a fixed dyadic scale-space, not dataset-specific seasonal lags.
The residual version has only 70 more parameters than the static ingest MLP and
still trains only the ingest/forecast MLPs. Actual short-video alternatives
(four causal prefixes, a two-frame dyadic-prefix pair, and a four-frame
coarse-to-fine scale video) were selected on validation only and did not win.

Due-diligence results use every stride-24 validation/test window and three model
seeds with a fixed set of 4,096 training windows. Trainable-method values below
are mean ± sample standard deviation; zero-shot references are deterministic:

| dataset / H | static image MSE | residual dyadic MSE | VisionTS MSE | Chronos MSE |
|---|---:|---:|---:|---:|
| ETTh1 / 96 | 0.4026 ± 0.0155 | 0.3946 ± 0.0270 | 0.3916 | **0.3899** |
| ETTh1 / 192 | 0.4369 ± 0.0200 | **0.4145 ± 0.0043** | 0.4266 | 0.4381 |
| ETTm1 / 96 | 0.3332 ± 0.0085 | 0.3302 ± 0.0094 | 0.3687 | **0.3187** |
| ETTm1 / 192 | **0.3542 ± 0.0065** | 0.3605 ± 0.0089 | 0.3838 | 0.3768 |
| Weather / 96 | 0.1721 ± 0.0030 | **0.1661 ± 0.0048** | 0.2942 | 0.1746 |
| Weather / 192 | 0.2140 ± 0.0078 | **0.2084 ± 0.0034** | 0.3053 | 0.2185 |

The globally selected residual method is best on mean MSE in **3/6 cells
(50%)**, but seed variance is material—especially ETTh1/96—and it wins no MAE
cells. Three seeds quantify optimization variance but do not establish
statistical significance. This full-split table supersedes the earlier
single-seed/capped-subset table even though the aggregate 3/6 count happens to
remain unchanged. Protocol metadata, individual seed scores, statistical
controls, and validation-only ablations are in `pilot/results_benchmark_grid/`.

## Meaningful two-frame time axis

The static and residual-RGB inputs above still duplicate one image to satisfy
VideoMAE's size-two tubelet. `step_shift_video` instead makes the two frames
carry consecutive observed states:

```text
frame 0 = VisionTS([x0, x0, ..., x(L-2)])
frame 1 = VisionTS([x0, x1, ..., x(L-1)])
```

Both frames use statistics from the observed current context only. This remains
one tubelet, 196 tokens, and the same ingest/forecast-MLP capacity as the static
adapter. A repeat control is exactly the duplicated static input; a reverse
control swaps the genuine frames.

The control test evaluates the **same selected checkpoint** on forward, repeat,
and reverse validation inputs. Separately training each convention would let
the forecast head adapt and would not test whether the learned model uses time.
Across the same full-split, three-seed protocol, repeat-minus-forward and
reverse-minus-forward validation MSE are positive on average in all 6 cells.
The input is therefore nondegenerate and its order is used.

| dataset / H | previous/current MSE | static MSE | residual dyadic MSE | repeat Δ | reverse Δ |
|---|---:|---:|---:|---:|---:|
| ETTh1 / 96 | **0.3873 ± 0.0065** | 0.4026 ± 0.0155 | 0.3946 ± 0.0270 | +0.0721 | +0.1434 |
| ETTh1 / 192 | 0.4215 ± 0.0114 | 0.4369 ± 0.0200 | **0.4145 ± 0.0043** | +0.0455 | +0.1384 |
| ETTm1 / 96 | 0.3317 ± 0.0052 | 0.3332 ± 0.0085 | **0.3302 ± 0.0094** | +0.0164 | +0.0104 |
| ETTm1 / 192 | 0.3643 ± 0.0049 | **0.3542 ± 0.0065** | 0.3605 ± 0.0089 | +0.0182 | +0.0076 |
| Weather / 96 | 0.1709 ± 0.0095 | 0.1721 ± 0.0030 | **0.1661 ± 0.0048** | +0.0059 | +0.0085 |
| Weather / 192 | 0.2119 ± 0.0071 | 0.2140 ± 0.0078 | **0.2084 ± 0.0034** | +0.0004 | +0.0043 |

It improves mean MSE over the static adapter in 5/6 cells, but over residual
dyadic RGB in only 1/6. This supports a meaningful and competitive temporal
axis, not a broad accuracy or state-of-the-art claim. Full construction,
limitations, seed scores, and reproduction instructions are in
`pilot/PREVIOUS_CURRENT_VIDEO.md` and `pilot/results_temporal_axis/`.

The tested **multiscale temporal hybrid** puts stride-1/2/4 views in RGB in
both previous and current frames. It beats residual RGB in 4/6 cells and the
grayscale previous/current model in 4/6, but its six-cell macro MSE is 0.3129
versus 0.3124 for residual RGB. Same-checkpoint controls confirm temporal use
on ETTh1/ETTm1, while Weather/96 prefers the repeated residual-RGB input. The
combination works, but scale and temporal benefits are not uniformly additive.

## Diagnosed failure modes (why naive transfer fails)

1. **Level-pathway blindness.** VideoMAE's `norm_pix_loss` training predicts
   per-patch *normalized* pixels: the model can output shape but never absolute
   level. VisionTS works partly because it uses a raw-pixel MAE checkpoint.
2. **Content/layout OOD.** TS renderings (band grids) are far from Kinetics;
   at 90%-tube-mask pretraining vs forecasting-shaped masks the task is also
   off-distribution. Wan2.1 needs its native resolution to even produce bands.
3. **No level anchoring in generative sampling.** Diffusion outpainting
   preserves or hallucinates levels; it does not infer them from context.

## Repository layout

```
pilot/
  run_pilot.py          # Phase 1: zero-shot, short context (+ --subsample)
  run_finetune.py       # Phase 2: per-dataset FT (8 methods incl. ablations)
  run_wan.py            # Phase 3: Wan2.1 temporal outpainting (RePaint/SDEdit)
  run_longctx.py        # Phase 4: literature protocol, LC-VMAE layout
  run_scroll.py         # Phase 5: scrolling-window rendering
  pretrain_vmae_ts.py   # Phase 6: continued pretraining (torchrun, DDP)
  run_lagged_lora.py    # fast lag-matrix frozen-vs-LoRA experiment
  run_mlp_adapter.py    # ingest MLP -> frozen VideoMAE -> forecast MLP
  run_visionts_static_adapter.py # exact VisionTS image -> static VideoMAE
  run_temporal_image_adapter.py # causal prefix/dyadic/previous-current views
  run_reference_baselines.py # naive, seasonal, VisionTS, and Chronos
  download_benchmark_data.py # official ETTm1 and Weather files
  test_no_lookahead.py  # split, scaling, and forecast-mask regression checks
  test_visionts_static_adapter.py # renderer equivalence and causality checks
  test_temporal_image_adapter.py # temporal view causality/geometry checks
  PREVIOUS_CURRENT_VIDEO.md # validated nondegenerate two-frame construction
  PILOT_RESULTS.md      # detailed phase 1-3 write-up
  results*/             # result JSONs (+ Wan probe images)
```

## Reproducing

```bash
pip install "transformers==4.46.3" torch torchvision einops pandas requests \
            visionts chronos-forecasting          # wan runs need diffusers>=0.33
# data: ETT csvs from zhouhaoyi/ETDataset; electricity/traffic/solar from
#       laiguokun/multivariate-time-series-data (gunzip to pilot/data/)
python pilot/run_pilot.py    --dataset ETTh1 --data-dir pilot/data
python pilot/run_finetune.py --dataset ETTh1 --data-dir pilot/data
python pilot/run_longctx.py  --dataset ETTh1 --hp 4 --stride 1
VMAE_CKPT=<ckpt_dir> python pilot/run_longctx.py ...   # eval a continued-pretrained ckpt
torchrun --nproc_per_node=2 pilot/pretrain_vmae_ts.py --steps 20000 --out <dir>
python pilot/run_lagged_lora.py --dataset ETTh1 --data-dir pilot/data
python pilot/test_no_lookahead.py
python pilot/run_mlp_adapter.py --dataset ETTh1 --data-dir pilot/data \
  --horizon 96 --train-cap 4096 --val-cap 0 --batch-size 32 --no-checkpoint
python pilot/run_visionts_static_adapter.py --dataset ETTh1 \
  --data-dir pilot/data --horizon 96 --train-cap 4096 --val-cap 0 \
  --batch-size 32 --no-checkpoint
python pilot/test_visionts_static_adapter.py
python pilot/download_benchmark_data.py --data-dir pilot/data
python pilot/run_reference_baselines.py --dataset ETTm1 --data-dir pilot/data \
  --horizon 96
python pilot/run_temporal_image_adapter.py --dataset ETTm1 \
  --data-dir pilot/data --mode dyadic_rgb_residual --horizon 96 \
  --train-cap 4096 --val-cap 1024 --test-cap 1024 --batch-size 32 \
  --no-checkpoint
python pilot/run_temporal_image_adapter.py --dataset ETTm1 \
  --data-dir pilot/data --mode step_shift_video --video-frames 2 \
  --horizon 96 --train-cap 4096 --val-cap 0 --test-cap 0 \
  --eval-stride 24 --window-seed 0 --seed 0 --no-checkpoint --eval-controls
python pilot/test_temporal_image_adapter.py
```

## Roadmap

- [ ] VideoMAE-TS learning curve (2.5k -> 20k) on all 7 datasets, both layouts
- [ ] Continued-pretraining ablations: synthetic-only vs +real (LOTSA subset);
      forecast-mask ratio; layout mix; equal-budget image-MAE continued
      pretraining as the critical *video-vs-image* control
- [ ] LN-FT / full-FT on top of VideoMAE-TS
- [ ] Evaluate the implemented lag/Hankel LoRA arm on all 7 datasets
- [ ] Wan2.1 with fine-tuned temporal-inpainting adapter (LoRA)

## Closest prior work

[VisionTS](https://arxiv.org/abs/2408.17253) (ICML'25) ·
[VisionTS++](https://arxiv.org/abs/2508.04379) ·
[ViTime](https://arxiv.org/abs/2407.07311) ·
[Time-VLM](https://arxiv.org/abs/2502.04395) ·
[Harnessing Vision Models for TS: Survey](https://arxiv.org/abs/2502.08869) (IJCAI'25) ·
[Deep Video Prediction for TSF](https://arxiv.org/abs/2102.12061) ·
No published work uses a video-pretrained backbone for general TS tasks as of
2026-07 — this study fills that gap (and documents why it is hard).
