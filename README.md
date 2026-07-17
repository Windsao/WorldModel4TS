# WorldModel4TS: Do Video Foundation Models Transfer to Time Series?

A systematic empirical study of whether **video-pretrained models** (VideoMAE,
Wan2.1) are better foundations for **numeric time series forecasting** than
image-pretrained models (VisionTS / ImageNet MAE) or TS baselines — plus ongoing
work on *making* the transfer work via continued pretraining on synthetic
TS-rendered videos.

> **Preprocessing status:** Phase 6 currently has two causal-leakage risks in the
> synthetic renderer. See [Known Issues and Data-Preprocessing Research Plan](PREPROCESSING_AND_KNOWN_ISSUES.md)
> before interpreting or extending continued-pretraining results.

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
level-pathway blindness. Purely synthetic -> zero benchmark leakage.
(`pilot/pretrain_vmae_ts.py`, 20k steps x batch 32, 2x A40)

**First signals (checkpoint @ step 2500 of 20k, ETTh1 stride 32, zero-shot):**

| | model | blank-control | Δ |
|---|---|---|---|
| LC layout | 0.740 | 1.083 | **model adds +31% signal** |
| scroll layout | 0.675 | 0.868 | **model adds +22% signal** |

The **first configuration in the entire study where a video model demonstrably
predicts the future of a numeric series zero-shot** (Kinetics checkpoints never
beat their controls). Absolute quality at 12.5% of training is still behind
VisionTS (0.355); the learning curve across checkpoints (2.5k/5k/.../20k) will
tell where it lands.

Also tested: channel augmentation (R=raw, G=first difference, B=expanding mean)
— no zero-shot gain with the Kinetics checkpoint (0.569 vs 0.566 grayscale).

## Phase 7 — Fine-tuning campaign: how close can video get? (2026-07-16)

Goal: drop zero-shot, push per-dataset FT to the best achievable numbers.
Recipe upgrades tested (all VMAE-TS-v2-init unless noted): 3-epoch cosine
full-FT (`ft3`), encoder+regression-head bypass (`head3`), Hankel
delay-embedding rendering, R/G/B channel augmentation. Long-context protocol.

| MSE | NLinear | VisionTS best | video best (config) |
|---|---|---|---|
| ETTh1 | 0.406 | **0.355** (zs) | 0.454 (Kinetics full-FT 1ep; ft3 0.470, head3 0.511, hankel 0.641, aug 0.496) |
| ETTh2 | **0.297** | 0.325 (zs, short-ctx) | 0.350 (ft3) |
| ETTm2 | 0.189 | **0.187** (LN-FT) | 0.273 (ft3 — best video number yet, prev 0.298) |
| solar | 0.215 | **0.195** (LN-FT) | 0.205 (Phase-2 LN-FT) |
| electricity | 0.166 | 0.149 (full-FT) | **0.158** (full-FT) — beats NLinear |
| traffic | 0.348 | **0.292** (full-FT) | 0.306 (full-FT) — beats NLinear |

**Verdict on "can video FT reach good results":** domain-split. On data-rich
electricity/traffic, video FT beats the linear baseline and sits 5-6% behind the
image model — a genuine positive cell. On ETT/solar, twelve-plus video variants
(2 inits x 2 decode paths x 4 renderings x 2 recipes) all fail to beat NLinear;
more epochs overfit (0.454 -> 0.470), head/Hankel/channel variants lose to the
reconstruction path. Under matched full-FT, video beats image on all four ETT
sets — image's edge lives in its LN-FT efficiency and zero-shot, consistent
with the A8 architectural diagnosis.

Wan2.1 LoRA (4k synthetic samples): level anchoring improves 4x (pure-gen MSE
2.3-3.0 -> 0.67); the copy-machine pathology reverses direction (snaive-init
0.377 -> 0.371 improved, vs 0.506 degraded zero-shot). Direction cured, dosage
insufficient — scaling LoRA data is the obvious next lever.

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
```

## Roadmap

- [ ] VideoMAE-TS learning curve (2.5k -> 20k) on all 7 datasets, both layouts
- [ ] Continued-pretraining ablations: synthetic-only vs +real (LOTSA subset);
      forecast-mask ratio; layout mix; equal-budget image-MAE continued
      pretraining as the critical *video-vs-image* control
- [ ] LN-FT / full-FT on top of VideoMAE-TS
- [ ] Hankel (delay-embedding) rendering — motion structure closest to video
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
