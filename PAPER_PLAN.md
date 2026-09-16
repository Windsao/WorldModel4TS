# Paper plan: forecast-aligned continual pretraining of video/world models for time series

Status 2026-09-14. Single source of truth for the story, the contributions, every number we can
cite today, and what is still missing. Numbers come from `VIDEO_TS_RESCUE_RESULTS.md` §1.10–1.13
(raw JSON in `/nyx-storage1/hanliu/wm4ts/lsf/`, assembler `pilot/final_bigtable.py`).
Placeholders are marked **[TODO]** with the experiment that fills them.

---

## 1. Story

**A time-series world model: make a visual backbone predict the future of a rendered series, and what
transfers is the dynamics prior, not the pretraining objective.**

### Terminology (state this explicitly in the paper; reviewers will test it)

- **Time-series world model** = the model *we train*: it observes the past frames of a rendered series and
  predicts the future ones. This is the standard world-model setting (predict future states from past),
  and it is what our forecast masking plus value-space loss optimises. Every arm in this paper, whatever
  its backbone, is trained as such a world model.
- **Dynamics prior** = what the backbone brought from its own pretraining, in three grades:
  *image model* (ImageNet MAE: no temporal dimension at all), *implicit dynamics* (VideoMAE: masked
  spatiotemporal tube reconstruction, bidirectional, so it models motion but is interpolative rather than
  predictive), *explicit predictive dynamics* (V-JEPA 2/2.1: predicts representations of unseen
  spatiotemporal blocks from a visible context).
- We do **not** claim VideoMAE is itself a world model. Calling a bidirectional masked autoencoder a world
  model is exactly the claim RoMAE and OccamVTS push back on, and our own §3.2 evidence agrees: trained with
  its native pixel objective it gets monotonically *worse* at forecasting. It becomes a world model only
  after the objective is translated.

### The four findings

1. Out of the box a video backbone does not forecast, and continuing its native masked-pixel pretraining on
   rendered series makes forecasting monotonically *worse* (5k → 176k steps), while the pretraining loss
   keeps improving. The objective, not the backbone, is the problem.
2. Translating the objective makes the same backbone a world model: decode the predicted future frames back
   to values and train on the value-space error, with period-aligned rendering and a 32-frame window. The
   budget curve flips from monotonically down to monotonically up (5k → 20k → 60k).
3. What the backbone must bring is a dynamics prior, not a particular pretraining objective. At matched size
   and budget, both video backbones beat the image backbone at every horizon (3–5%) and are indistinguishable
   from each other; a randomly initialised backbone of the same architecture is the worst arm of all.
4. The result is state of the art on the standard zero-shot LSF benchmark (six-dataset mean MSE 0.285 vs
   VisionTS++ base 0.291 and large 0.289, Moirai-base 0.311, VisionTS 0.315), at roughly 1/13 of
   VisionTS++'s continual-pretraining steps and 0.3% of its data.

**Budget is measured in pretraining samples (steps x batch), not FLOPs.** The compute-fair pair is
VideoMAE-B vs ImageNet MAE-B (same ViT-B): the video prior is worth 5.3% at 5k steps and the two converge by
20k, i.e. the prior buys sample efficiency rather than a higher ceiling. V-JEPA 2-L is ViT-L and costs ~9x
more per step, so its extra 2% is not a compute-efficiency claim.

---

## 2. Novelty and contributions

**C1. A diagnosis, with a full budget curve.** Pixel-reconstruction continual pretraining and
forecasting decouple: held-out pixel loss falls monotonically (0.0366 → 0.0312) while downstream
error rises (0.339 → 0.381 three-horizon mean) from 5k to 176k steps. To our knowledge no prior work
reports this; VisionTS++ reports only that more pretraining helps under *its* objective.

**C2. Forecast-aligned pretraining objective (the "translator").** The predicted future frames are
decoded back to values with a differentiable soft row-count, and the loss is the MSE in
standardised value space, added to the pixel loss. Column-wise, so it is independent of the period.
This single component carries the entire gain (ablation: 5–8%, growing with horizon).

**C3. Temporal-window scaling for rendered series.** Clip length 16 → 32 frames doubles the usable
context and is what lets the 15-minute datasets reach VisionTS-level context. VideoMAE's sinusoidal
position table is rebuilt; pretrained weights load unchanged.

**C4. The dynamics prior is what transfers; the pretraining objective is not.** At matched size (ViT-B) and
matched budget (5k steps, identical recipe and corpus), both video world models beat the image model at every
horizon, while the two video models are indistinguishable from each other:

| backbone (ViT-B, 5k steps) | H=96 | H=336 | H=720 |
|---|---:|---:|---:|
| ImageNet MAE-B (no dynamics) | 0.264 | 0.350 | 0.395 |
| VideoMAE-B (implicit dynamics, reconstructive) | **0.250** | 0.337 | 0.384 |
| V-JEPA 2.1-B (explicit dynamics, predictive) | 0.252 | **0.335** | **0.382** |

A randomly initialised V-JEPA 2-L is the worst arm of all (0.280 / 0.369 / 0.418), below even the 3.5x smaller
ImageNet MAE-B, so the gain is the pretrained prior rather than capacity. Scale still helps on top: V-JEPA 2-L
reaches 0.245 / 0.329 / 0.379.

**C5. SOTA at 1/40 budget.** See §3.1.

Not claimed (and explicitly discussed): compute efficiency for V-JEPA 2 (it is ViT-L); that a
reconstructive video model beats an image model at large budget (they converge by 20k steps, the
prior buys sample efficiency); that our gains come from more data (they do not, §3.5).

---

## 3. Results we can cite today

### 3.1 Main table (VisionTS LSF protocol, 4-horizon average MSE/MAE, six datasets)

Final model: VideoMAE-B (Kinetics-400) + full recipe, 32-frame clips, corpus v2, 60k steps,
batch 32, lr 1e-4 cosine, wd 0. One checkpoint for all 24 cells, `--mode auto` inference, no
per-dataset switching, no ensembling, no blending.

| Model | ETTm1 | ETTm2 | ETTh1 | ETTh2 | electricity | weather | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Ours (60k)** | **0.352**/**0.369** | **0.241**/0.302 | 0.404/0.416 | 0.335/0.376 | **0.163**/**0.254** | **0.215**/0.252 | **0.285**/0.328 |
| Ours (20k, 6.3 h on 1 A40) | 0.360/0.371 | 0.242/0.304 | 0.400/0.417 | 0.342/0.381 | 0.169/0.261 | 0.217/0.253 | 0.288/0.331 |
| Ours (20k, 2-seed prediction ensemble) | 0.351 | 0.242 | 0.401 | 0.338 | 0.168 | 0.215 | 0.286 |
| VisionTS++ large (ViT-L) | 0.354/0.369 | 0.244/**0.298** | 0.403/0.418 | **0.327**/**0.365** | 0.181/0.264 | 0.226/0.243 | 0.289/**0.326** |
| VisionTS++ base | 0.360/0.372 | 0.244/**0.298** | 0.402/0.416 | 0.333/0.370 | 0.184/0.265 | 0.222/**0.241** | 0.291/0.327 |
| VisionTS | 0.374/0.372 | 0.318/0.366 | **0.390**/**0.414** | **0.333**/0.375 | 0.207/0.294 | 0.269/0.292 | 0.315/0.352 |
| Moirai small / base / large | 0.448 / 0.382 / 0.390 | 0.272 / 0.276 / 0.317 | 0.400 / 0.434 / 0.510 | 0.341 / 0.346 / 0.354 | 0.233 / 0.188 / 0.188 | 0.242 / 0.238 / 0.260 | 0.323 / 0.311 / 0.337 |
| Chronos small / base / large | 0.640 / 0.646 / 0.556 | 0.310 / 0.295 / 0.300 | 0.545 / 0.591 / 0.589 | 0.424 / 0.406 / 0.455 | 0.220 / 0.215 / 0.204 | 0.300 / 0.293 / 0.279 | 0.407 / 0.408 / 0.397 |
| Time-MoE small / base | 0.394 / 0.376 | 0.316 / 0.349 | 0.400 / 0.394 | 0.367 / 0.405 | – | 0.266 / 0.270 | – |
| MOMENT | 0.670/0.537 | 0.316/0.371 | 0.684/0.566 | 0.362/0.410 | 0.765/0.687 | 0.294/0.326 | 0.515/0.483 |
| Timer 28B | 0.487/0.457 | 0.328/0.347 | 0.444/0.457 | 0.358/0.407 | – | 0.304/0.331 | – |
| TimesFM | 0.433/0.419 | – | 0.473/0.444 | 0.392/0.406 | – | – | – |

Baselines from the VisionTS++ paper, Table 4. VisionTS++ large (ViT-L) is included: our ViT-B model beats it on
mean MSE (0.285 vs 0.289), wins ETTm1, ETTm2, electricity (−10%) and weather (−4.9%), ties ETTh1, and loses ETTh2
(+2.4%); on MAE it is 0.328 vs 0.326.
Per-dataset vs VisionTS++: win electricity (−11%), ETTm1 (−2.2%), weather (−3.2%), ETTm2 (−1.2%);
tie ETTh1 (+0.5%); lose ETTh2 (+0.6%).

Budget: 60k steps × batch 32 on a 654M-observation corpus ≈ 1/13 of VisionTS++'s steps and 0.3% of
its data; the 20k row is 1/40 of the steps at 6.3 GPU-hours.

### 3.2 The diagnosis (motivates the method)

Held-out pixel loss vs downstream error, pixel objective, 16-frame clips:

| steps | 5k | 20k | 60k | 176k |
|---|---:|---:|---:|---:|
| held-out pixel loss | 0.0366 | 0.0333 | 0.0329 | 0.0312 |
| downstream mean MSE (3 horizons) | 0.339 | 0.344 | — | 0.381 |

The 176k run cost $26 on a rented A100 (12.5 h) and is 5–60% worse than the 20k baseline per cell,
worst at long horizons (ETTh2-720 0.429 → 0.617).

### 3.3 Ablations on the final architecture (32 frames, 5-dataset mean MSE)

| configuration | H=96 | H=336 | H=720 |
|---|---:|---:|---:|
| full recipe, 20k | 0.245 | 0.330 | 0.379 |
| pixel objective only | 0.257 | 0.347 | 0.411 |
| without prior anchoring | 0.244 | 0.330 | 0.378 |
| 5k steps | 0.250 | 0.337 | 0.384 |
| 60k steps | 0.242 | 0.329 | 0.377 |

Readings: the forecast-aligned objective is worth 5 / 5 / 8% and the gap grows with horizon; prior
anchoring is unnecessary at 32 frames (it was essential at 16); the budget curve is now monotone
*upward*, the exact opposite of §3.2.

### 3.4 Backbone study (matched 5k steps, 32 frames, identical recipe and corpus)

| backbone | H=96 | H=336 | H=720 |
|---|---:|---:|---:|
| V-JEPA 2 (predictive-latent world model, ViT-L) | **0.245** | **0.329** | **0.379** |
| VideoMAE-B (reconstructive video) | 0.250 | 0.337 | 0.384 |
| ImageNet MAE-B (reconstructive image) | 0.264 | 0.350 | 0.395 |

V-JEPA wins 14 of 15 dataset cells and at 5k steps equals VideoMAE at 20k.

Condition-(i) control, same ViT-L architecture with random init (complete, 5-dataset mean MSE):

| backbone (5k steps, 32 frames) | H=96 | H=336 | H=720 |
|---|---:|---:|---:|
| V-JEPA 2 pretrained | **0.245** | **0.329** | **0.379** |
| VideoMAE-B pretrained | 0.250 | 0.337 | 0.384 |
| ImageNet MAE-B pretrained | 0.264 | 0.350 | 0.395 |
| V-JEPA 2 random init (same ViT-L) | 0.280 | 0.369 | 0.418 |

World-model pretraining is worth 12.5% / 10.8% / 9.3% at identical architecture and budget. The
randomly initialised ViT-L is the worst row at every horizon, below even ImageNet MAE-B despite
~3.5x the parameters, so the V-JEPA advantage is the pretrained prior, not model size. The largest
per-dataset gap is ETTh1 at long horizons (H=336 0.420 vs 0.534, H=720 0.454 vs 0.568, 21-25%).

Initialization at two clip lengths (full recipe, two seeds each, 5-dataset mean):

| clip length | video init | image init |
|---|---:|---:|
| 16 frames | 0.257 / 0.350 / 0.383 | 0.268 / 0.359 / 0.400 |
| 32 frames | 0.245 / 0.330 / 0.379 | 0.244 / 0.329 / 0.373 |

The video prior is worth 3–4% when the window is short and nothing when it is long: it supplies
long-range temporal context that a longer window supplies directly.

### 3.5 What did not work (report as negative results, prevents reviewer "why not try X")

| attempt | result |
|---|---|
| 9x pretraining budget (176k steps, pixel objective) | 5–60% worse |
| larger batch (128, 256 via accumulation) | equal or worse at every horizon |
| 4x more in-domain data (corpus v3: buildings_900k, LargeST, Q-Traffic, Azure) | ties corpus v2 |
| longer masked future (hp = 8 of 16 frames) | slightly worse |
| value-loss weight 0.3 instead of 0.1 | worse |
| lr 2e-4 | worse; 5e-5 ties 1e-4 |
| 3-checkpoint ensemble | no gain |
| averaging with the zero-parameter prior blend | validation selection does not transfer, net 0% |
| cross-corpus weight soup (v1 + v2) | worse than both parents |
| same-corpus 2-seed weight soup | better on ETT, 16% worse on electricity, net worse |

---

## 4. Experiments still to run

| # | experiment | why | cost | status |
|---|---|---|---|---|
| E1 | random-init V-JEPA full grid | isolates world-model pretraining from ViT-L size; needed for C4 | done | **done 2026-09-15** (pretraining worth 9-13%) |
| E2 | third seed of the main 60k model + variance | 0.285 vs 0.291 is a 2% gap; reviewers will ask | 20 h + 12 h eval | **[TODO]** |
| E3 | second benchmark (GIFT-Eval or Monash subset) | answers "tuned on six datasets"; VisionTS++ reports GIFT-Eval | ~2 days incl. data loader | **[TODO]** |
| E4 | our objective inside the official VisionTS codebase | strongest evidence the design transfers beyond our implementation | ~1 day | **[TODO]** |
| E5 | inference-cost table (latency/memory vs VisionTS++, 16 vs 32 frames) | 32-frame electricity is ~8x slower; must be disclosed | ~2 h | **[TODO]** |
| E6 | V-JEPA 2 at 20k steps | would have made the ViT-L world model the main-table model | ~54 h | **cancelled 2026-09-15** (V-JEPA ties VideoMAE at matched size, so a bigger V-JEPA row is not needed) |
| E7 | qualitative figure: rendered frames, prediction, decoded values | reviewers want to see the representation | ~2 h | **[TODO]** |
| E8 | size-matched control | removes the ViT-L vs ViT-B confound in C4 | done | **done 2026-09-15** via V-JEPA 2.1 ViT-B (official Meta release) instead of two ViT-L runs |
| E9 | budget curves per backbone: image and VideoMAE at 10k and 60k, V-JEPA at 10k (from E6 checkpoint) | turns "4x fewer samples" into a curve figure | ~2 days | **[TODO]** |

---

## 5. Paper skeleton

1. **Introduction.** Visual foundation models for TS (VisionTS line); the gap: nobody asks whether
   the *objective* survives the modality transfer. Contributions C1–C5.
2. **Related work.** VisionTS / VisionTS++ (image MAE + LOTSA), Moirai / Chronos / Time-MoE / Timer,
   SVTime and OccamVTS (what visual pretraining actually provides), RoMAE (bidirectional MAE is
   interpolative), V-JEPA 2 as a world model.
3. **Method.** Rendering (period-aligned frames), the forecast-aligned value loss with the
   differentiable decoder, temporal scale augmentation, 32-frame clips, inference recipe.
4. **Experiments.** Setup and protocol; main table (§3.1); ablations (§3.3); budget curves
   (§3.2 vs §3.3); backbone study (§3.4); negative results (§3.5).
5. **Analysis.** Why the pixel objective fails (error grows with horizon); what the video prior
   buys and when; cost. **[TODO: E5 numbers]**
6. **Limitations.** ETTh2 remains behind VisionTS++; six-dataset protocol only until E3;
   32-frame inference cost; single backbone family for the main result.

**Venue.** ICLR fits (method + negative-result-driven design + full ablations + SOTA at a fraction
of the budget). CVPR is a poor fit: TS forecasting is peripheral there and our strongest claims are
about the objective and efficiency, not vision.

---

## 6. Reproducibility pointers

- Trainer `pilot/pretrain_route_b.py` (flags `--frames`, `--scale-aug`, `--value-loss`, `--l2sp`,
  `--hp-max`, `--accum`, `--no-domain-exclude`); world-model trainer `pilot/pretrain_route_j.py`.
- Evaluation `pilot/eval_lsf.py` (`--mode auto`), table assembler `pilot/final_bigtable.py`.
- Tests: `tests/test_route_b.py` (25), `tests/test_route_j.py` (4).
- Corpus v2: 98 LOTSA subsets, 184,013 series, 654.2M observations, at
  `/nyx-storage1/hanliu/wm4ts/route_b/corpus_v2`.
- Main checkpoint: `/nyx-storage1/hanliu/wm4ts/route_b/vmae_full_f32_savl_l2sp_v2_60k/step_60000`.
- Gotcha that cost us results once: any launcher taking a variant override must put that override in
  the **result tag**, not only in the checkpoint path, or two runs overwrite each other's JSON.
