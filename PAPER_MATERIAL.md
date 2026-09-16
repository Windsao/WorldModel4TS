# Paper material (self-contained): a time-series world model from a video backbone

Everything a writer needs to draft the paper: story, claims that the data supports and claims it does
not, method details, every number, related work, limitations, and open items. Numbers are final unless
marked *pending*. Status 2026-09-16.

---

## 0. One-paragraph summary

Visual foundation models for time-series forecasting (VisionTS, VisionTS++) render a series as an image
and reuse an ImageNet masked autoencoder. We ask what a *video* backbone adds and find that its native
objective is the obstacle: continuing masked-pixel pretraining on rendered series lowers the pretraining
loss while making forecasting monotonically worse, up to 9× the budget. We convert the setup into a
time-series **world model** — the model observes past frames and predicts future frames — by decoding
the predicted frames back to values with a differentiable read-out and training on the value-space
error, together with period-aligned rendering, frame-scale augmentation and a 32-frame clip. The budget
curve flips from monotonically decreasing to monotonically increasing, and the resulting zero-shot
forecaster attains the lowest six-dataset mean MSE in the published zero-shot table (0.285 vs
VisionTS++ large 0.289 and base 0.291) using ~1/13 of VisionTS++'s continual-pretraining steps and
0.3% of its data. A matched-size backbone study shows that what transfers is the *dynamics prior*: at
ViT-B, both video backbones (VideoMAE, V-JEPA 2.1) beat the image backbone by 3–5% at every horizon and
are indistinguishable from each other, while a randomly initialised backbone of the same architecture is
the worst arm of all.

## 1. Title candidates

1. Time-Series World Models from Video Backbones: the Objective, not the Architecture
2. Forecast-Aligned Continual Pretraining Turns a Video Backbone into a State-of-the-Art Zero-Shot Forecaster
3. What Transfers from Video to Time Series is the Dynamics Prior
4. Predicting the Future of a Rendered Series: World-Model Pretraining for Zero-Shot Forecasting

## 2. Claims

**Supported (write these):**
- C1 Diagnosis: masked-pixel continual pretraining on rendered series decouples from forecasting —
  held-out pixel loss falls monotonically (0.0366 → 0.0312) while forecast error rises (0.339 → 0.381)
  from 5k to 176k steps.
- C2 Method: a forecast-aligned value-space loss (plus scale augmentation and a 32-frame window) turns
  the same backbone into a world model and flips the budget curve to monotonically improving.
- C3 State of the art on the standard zero-shot LSF protocol at a fraction of the budget.
- C4 The dynamics prior is what transfers; the specific video pretraining objective is not, and capacity
  is not the explanation (random-init control).
- C5 Every ablation and control uses one identical recipe, corpus, and inference rule.

**Not supported (do not write):**
- "Video backbones beat image backbones in general" — true at matched size and *small* budget (5k steps,
  3–5%); the gap closes by 20k steps under the aligned objective.
- "Predictive world models (V-JEPA) beat reconstructive video models (VideoMAE)" — they tie at ViT-B
  (0.252 vs 0.250 at H=96). The V-JEPA 2-L advantage (0.245) is capacity.
- "Our method is more compute-efficient than VisionTS++" — we measure budget in *pretraining samples*
  (steps × batch) and data, not FLOPs. V-JEPA 2-L costs ~9× more per step than a ViT-B.
- "VideoMAE is a world model" — it is a bidirectional masked autoencoder; it *becomes* a world model
  only when trained with our forecast objective. Call its prior "implicit dynamics".
- Any claim on traffic: the corpus contains Monash `traffic_hourly`, which *is* the LSF traffic set.

## 3. Method (write §3 from this)

**Rendering.** A univariate series of length `16k·P` (`P` = period from the sampling frequency; `k` =
periods per frame) is standardised with the statistics of the *visible* context only, clipped to
z ∈ [−3, 3], mapped to a bar height `h = 0.5 + 0.4·z/3`, and drawn as an area chart in a 224×224 frame:
row `round((1−h)·223)` is the boundary, below it is filled (0.12), above it background (0.92), with grid
lines every 28 rows (0.80). Frame *f* holds periods `[f·k, (f+1)·k)`, so vertical position encodes value
and horizontal position encodes phase. Clip length is 32 frames (tubelet 2 → 16 temporal tokens, 14×14
spatial tokens, 3136 tokens per clip).

**Masking.** The last `hp ∈ {2,4}` frames are masked (the future); with probability 0.3 the first `lead`
tubelets are also hidden so the model also trains on shorter contexts. 30% of batches instead use tube
masking. The loss is computed on future tokens only.

**Forecast-aligned objective (the contribution).** Let `x̂` be the predicted future frames. A
differentiable soft row-count decodes each column back to a height:
`soft = clamp((0.80 − gray)/(0.80 − 0.12), 0, 1)`, `h = 1 − (224 − Σ_rows soft)/223`, and the loss is
`L = MSE_pixel(x̂, x) + λ·(3/0.4)² · MSE(h(x̂), h(x))` with λ = 0.1. The second term is an MSE in
standardised value units, computed column-wise so it is independent of `P`.

**Frame-scale augmentation.** During training `k ∈ {1,2,4}` with probability .5/.3/.2, matching the
multi-scale inference rule.

**Prior anchoring (optional).** Weight decay toward the initial weights (`p ← p − lr·λ_sp·(p − θ₀)`,
λ_sp = 0.05, wd = 0) instead of toward zero. Essential at 16 frames, *unnecessary at 32* (ablation).

**Inference (`--mode auto`).** Given period `P` and horizon `H`: if `H ≤ 4P`, one pass with
`hp = 4` and an ensemble over `k·s·P ≤ 96`; otherwise `k = ceil(H/4P)` periods per frame, with
`ceil(k / (96/P))` rollout passes when `P < 96` and a single direct pass when `P ≥ 96`. One checkpoint,
one rule, all 24 cells; no per-dataset tuning, no ensembling of checkpoints, no blending.

**Backbone and optimisation.** VideoMAE-B (Kinetics-400), `norm_pix_loss=False`, AdamW (β = 0.9/0.95,
lr 1e-4 cosine, warm-up 500), batch 32 (micro-batch 8 × 4 accumulation), bf16, 60k steps ≈ 20 h on one
A40. Position embeddings are sinusoidal and rebuilt for 32 frames, so pretrained weights load unchanged.

**Corpus.** 98 LOTSA subsets, 184,013 series, 654.2M observations, 20% synthetic series mixed in.
Datasets are sampled ∝ √(observations). None of the six evaluation datasets is included.

## 4. Experimental setup

VisionTS-paper LSF protocol, reproduced to three decimals against the official VisionTS checkpoint
(ETTh1-96 0.3527/0.3833 vs paper 0.353/0.383): Time-Series-Library splits (ETT 12/4/4 months, others
70/10/20), `StandardScaler` fitted on train, all channels, stride-1 test origins, horizons 96/192/336/720,
MSE/MAE on standardised values. Contexts per dataset follow VisionTS (ETTh1 2880, ETTh2 1728, ETTm1 2304,
ETTm2 4032, electricity 2880, weather 4032); periods 24/24/96/96/24/144. Baselines quoted from the
VisionTS++ paper Table 4 and the VisionTS paper.

## 5. Main table — four-horizon averages, MSE/MAE

| Model | ETTm1 | ETTm2 | ETTh1 | ETTh2 | electricity | weather | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Ours (VideoMAE-B, 32f, 60k)** | **0.352**/**0.369** | **0.241**/0.302 | 0.404/0.416 | 0.335/0.376 | **0.163**/**0.254** | **0.215**/0.252 | **0.285**/0.328 |
| Ours (20k steps, 6.3 GPU-h) | 0.360/0.371 | 0.242/0.304 | 0.400/0.417 | 0.342/0.381 | 0.169/0.261 | 0.217/0.253 | 0.288/0.331 |
| Ours (20k, two-seed prediction ensemble) | 0.351 | 0.242 | 0.401 | 0.338 | 0.168 | 0.215 | 0.286 |
| VisionTS++ large (ViT-L) | 0.354/0.369 | 0.244/**0.298** | 0.403/0.418 | **0.327**/**0.365** | 0.181/0.264 | 0.226/0.243 | 0.289/**0.326** |
| VisionTS++ base (ViT-B) | 0.360/0.372 | 0.244/**0.298** | 0.402/0.416 | 0.333/0.370 | 0.184/0.265 | 0.222/**0.241** | 0.291/0.327 |
| VisionTS | 0.374/0.372 | 0.318/0.366 | **0.390**/**0.414** | 0.333/0.375 | 0.207/0.294 | 0.269/0.292 | 0.315/0.352 |
| Moirai small | 0.448/0.410 | 0.272/0.321 | 0.400/0.424 | 0.341/0.379 | 0.233/0.320 | 0.242/0.267 | 0.323/0.353 |
| Moirai base | 0.382/0.388 | 0.276/0.320 | 0.434/0.439 | 0.346/0.382 | 0.188/0.274 | 0.238/0.261 | 0.311/0.344 |
| Moirai large | 0.390/0.389 | 0.317/0.366 | 0.510/0.469 | 0.354/0.377 | 0.188/0.273 | 0.260/0.275 | 0.337/0.358 |
| Chronos small | 0.640/0.500 | 0.310/0.350 | 0.545/0.472 | 0.424/0.430 | 0.220/0.284 | 0.300/0.318 | 0.407/0.392 |
| Chronos base | 0.646/0.500 | 0.295/0.338 | 0.591/0.468 | 0.406/0.411 | 0.215/0.279 | 0.293/0.315 | 0.408/0.385 |
| Chronos large | 0.556/0.465 | 0.300/0.341 | 0.589/0.466 | 0.455/0.427 | 0.204/0.274 | 0.279/0.306 | 0.397/0.380 |
| Time-MoE small | 0.394/0.416 | 0.316/0.361 | 0.400/0.424 | 0.367/0.404 | – | 0.266/0.297 | – |
| Time-MoE base | 0.376/0.406 | 0.349/0.380 | 0.394/0.420 | 0.405/0.415 | – | 0.270/0.300 | – |
| Timer 28B | 0.487/0.457 | 0.328/0.347 | 0.444/0.457 | 0.358/0.407 | – | 0.304/0.331 | – |
| TimesFM | 0.433/0.419 | – | 0.473/0.444 | 0.392/0.406 | – | – | – |
| MOMENT | 0.670/0.537 | 0.316/0.371 | 0.684/0.566 | 0.362/0.410 | 0.765/0.687 | 0.294/0.326 | 0.515/0.483 |

Ours wins four datasets outright, loses ETTh1 to VisionTS (0.404 vs 0.390) and ETTh2 to VisionTS++ large
(0.335 vs 0.327). Mean MAE is 0.328 vs VisionTS++ large 0.326.

## 6. Per-horizon table — our model, MSE/MAE

| dataset | H=96 | H=192 | H=336 | H=720 | avg |
|---|---|---|---|---|---|
| ETTm1 | 0.292/0.332 | 0.332/0.357 | 0.363/0.377 | 0.423/0.409 | 0.352/0.369 |
| ETTm2 | 0.160/0.245 | 0.214/0.283 | 0.264/0.318 | 0.328/0.362 | 0.241/0.302 |
| ETTh1 | 0.352/0.380 | 0.396/0.408 | 0.424/0.425 | 0.446/0.449 | 0.404/0.416 |
| ETTh2 | 0.266/0.324 | 0.333/0.369 | 0.361/0.395 | 0.379/0.415 | 0.335/0.376 |
| electricity | 0.128/0.219 | 0.148/0.239 | 0.170/0.262 | 0.207/0.294 | 0.163/0.254 |
| weather | 0.139/0.183 | 0.181/0.226 | 0.232/0.266 | 0.308/0.331 | 0.215/0.252 |

Per-cell against VisionTS (the only baseline with published per-horizon numbers) ours wins **18 of 24**.
The six losses are ETTh1-192/336/720 (+1.1/+4.1/+9.8%), ETTh2-192/336 (+1.7/+4.7%), ETTm1-720 (+1.7%).
Largest wins: weather-96 −36.9%, ETTm2-96 −29.8%, electricity −18 to −28% at every horizon.

## 7. Ablations (5-dataset mean MSE, ETT×4 + weather, 32-frame clips, identical corpus)

| configuration | H=96 | H=336 | H=720 |
|---|---:|---:|---:|
| full recipe, 20k steps | 0.245 | 0.330 | 0.379 |
| − value-space loss (native pixel objective) | 0.257 | 0.347 | 0.411 |
| − prior anchoring | 0.244 | 0.330 | 0.378 |
| 5k steps | 0.250 | 0.337 | 0.384 |
| 60k steps | **0.242** | **0.329** | **0.377** |

The value-space loss accounts for the entire gain and the gap grows with horizon (5/5/8%). Anchoring is
inert at 32 frames. The budget curve is monotonically improving, which is the opposite of the pixel
objective (§8).

## 8. The diagnosis (16-frame clips, pixel objective) — motivating result

| steps | 5k | 20k | 60k | 176k |
|---|---:|---:|---:|---:|
| held-out pixel loss | 0.0366 | 0.0333 | 0.0329 | 0.0312 |
| downstream mean MSE (3 horizons) | 0.339 | 0.344 | — | 0.381 |

Per-cell the 176k model is 5–60% worse than the 20k one, worst at long horizons (ETTh2-720
0.429 → 0.617). Ruled out as explanations: checkpoint key sets, transformers version, attention config,
loader warnings; train loss is flat (0.0092 → 0.0089), so it is not pixel-space overfitting. All weight
norms shrink ~30% (AdamW weight decay over 9× steps).

*If the paper must be 32-frame-only:* the same curve at 32 frames needs two more runs (pixel objective at
5k and 60k); the 20k point already exists (0.257/0.347/0.411).

## 9. Backbone study (matched budget 5k steps, 32 frames, identical recipe and corpus)

| backbone | pretraining | params | H=96 | H=336 | H=720 |
|---|---|---:|---:|---:|---:|
| ImageNet MAE-B | images, no dynamics | 87M | 0.264 | 0.350 | 0.395 |
| VideoMAE-B | video, implicit dynamics | 87M | **0.250** | 0.337 | 0.384 |
| V-JEPA 2.1-B | video, explicit predictive dynamics | 87M (+23M predictor) | 0.252 | **0.335** | **0.382** |
| V-JEPA 2-L | same, larger | 300M | 0.245 | 0.329 | 0.379 |
| V-JEPA 2-L, random init | none | 300M | 0.280 | 0.369 | 0.418 |

Random-init per dataset at H=96: 0.401/0.282/0.369/0.182/0.166 vs pretrained 0.357/0.270/0.294/0.162/0.141
(12.5% overall); at H=336 the gap is 10.8%, at H=720 9.3%. The randomly initialised ViT-L is worse than
the 3.5× smaller ImageNet MAE-B, so capacity is not the explanation.

Caveats to state: V-JEPA 2.1-B is distilled from a ViT-G teacher, its predictor output head was
re-initialised (the pretrained head maps to the 1664-d teacher space, our targets are the 768-d ViT-B EMA),
and it was pretrained at 384 px while we render 224.

### Initialization at two clip lengths (two seeds each, 5-dataset mean)

| clip length | video init | image init |
|---|---|---|
| 16 frames | 0.257 / 0.350 / 0.383 and 0.256 / 0.348 / 0.384 | 0.268 / 0.359 / 0.400 and 0.265 / 0.353 / 0.400 |
| 32 frames | 0.245 / 0.330 / 0.379 | 0.244 / 0.329 / 0.373 and 0.245 / 0.330 / 0.373 |

At 16 frames the video prior is worth 3–4%; at 32 frames the two tie. Reading: part of what video
pretraining supplies is long-range temporal context, which a longer window supplies directly. (Omit this
subsection if the paper is 32-frame-only; then §9's matched-size comparison at 5k carries the claim.)

## 10. Seed variance

Two seeds of the 20k model (five datasets, 3 horizons): 0.245/0.330/0.379 and 0.243/0.330/0.378 —
differences ≤ 0.002. On the full 6×4 table, seed 0 = 0.288 and seed 1 matches on five of six datasets
(ETTm1 0.351 vs 0.360, ETTh1 0.405 vs 0.400, ETTh2 0.336 vs 0.342, ETTm2 0.245 vs 0.242, weather 0.216 vs
0.217; electricity *pending*). A second seed of the 60k main model is training (*pending*, ~3 h left).

## 11. Negative results worth a paragraph each

| attempt | outcome |
|---|---|
| 9× pretraining budget under the pixel objective (176k steps, $26 on a rented A100) | 5–60% worse per cell |
| larger batch (128, 256 via accumulation) | equal or worse at every horizon |
| 4× more in-domain data (corpus v3: buildings_900k, LargeST, Q-Traffic, Azure) | ties corpus v2 |
| longer masked future (8 of 16 frames) | slightly worse |
| value-loss weight 0.3 instead of 0.1 | worse; lr 2e-4 worse, 5e-5 ties 1e-4 |
| three-checkpoint ensemble | no gain |
| averaging with a zero-parameter statistical prior blend | validation selection does not transfer (net 0%) |
| weight-space soup of two seeds | better on ETT, 16% worse on electricity, net worse (0.291) |
| cross-corpus weight soup (v1+v2) | worse than both parents |

Prediction-space averaging of two seeds *does* help (+0.6%, never worse): 0.286 on the full table.

## 12. Efficiency

| | continual-pretraining steps | corpus | wall clock |
|---|---:|---|---|
| Ours (main) | 60k × batch 32 = 1.9M windows | 654M observations | ~20 h, 1× A40 |
| Ours (20k row) | 20k × 32 = 0.64M windows | same | 6.3 h, 1× A40 |
| VisionTS++ | ~100k steps, large batch | full LOTSA, 231B observations | not reported |

Ratio: ~1/13 of the steps (1/40 for the 20k row) and 0.3% of the data. Inference cost is *pending*
(32-frame clips are ~8× slower than 16-frame on electricity; the table must disclose this).

## 13. Related work (one line each, with the angle to take)

- **VisionTS / VisionTS++** — image MAE reused for forecasting; VisionTS++ shows full-parameter continual
  pretraining on LOTSA is what makes it SOTA. We share the "visual backbone" premise but change the
  objective, and we are the first to report that the native objective *hurts* with scale.
- **Moirai / Chronos / Time-MoE / Timer / TimesFM / MOMENT** — native time-series foundation models; our
  baselines in the main table.
- **V-JEPA 2 / 2.1** — predictive latent world models for video; we use them as backbones and as the
  "explicit dynamics" end of the prior spectrum.
- **VideoMAE** — masked spatiotemporal reconstruction; implicit dynamics, bidirectional, interpolative.
- **RoMAE**, **OccamVTS** — evidence that bidirectional MAEs interpolate rather than extrapolate and that
  only low-level visual features matter for TS; consistent with our diagnosis.
- **SVTime** — a 215k-parameter model with three inductive biases matches VisionTS, i.e. the transferable
  visual prior is small; motivates asking what *else* a video prior buys.

## 14. Limitations (write them, reviewers will find them)

1. ETTh1 and ETTh2 mid-to-long horizons remain behind VisionTS/VisionTS++; our context is capped at
   32 frames × ≤4 periods.
2. Six-dataset protocol only; no GIFT-Eval or Monash results yet.
3. Mean MAE ties rather than beats VisionTS++ (0.328 vs 0.326).
4. Budget is measured in samples and data, not FLOPs.
5. One backbone family for the main result (VideoMAE-B); V-JEPA arms are 5k-step controls only.
6. The corpus contains `traffic_hourly`, so the traffic benchmark cannot be added without rebuilding it.

## 15. Open items (state as future work or finish before submission)

| item | status |
|---|---|
| second seed of the 60k main model | training, ~3 h left |
| electricity cell of the 20k seed-1 table | evaluating |
| GIFT-Eval or Monash second benchmark | not started (~2 days) |
| inference-cost table | not started (~2 h) |
| pixel-objective budget curve at 32 frames (5k, 60k) | not started (~1 day) |
| our objective ported into the official VisionTS codebase | not started (~1 day) |
| qualitative figure (rendered frames → prediction → decoded values) | not started (~2 h) |

## 16. Reproducibility pointers

- Trainer `pilot/pretrain_route_b.py` (`--frames`, `--scale-aug`, `--value-loss`, `--l2sp`, `--hp-max`,
  `--accum`, `--no-domain-exclude`); world-model arms `pilot/pretrain_route_j.py` (`WorldModel`,
  `WorldModel21`, `--teacher ema_b|vitG`).
- Evaluation `pilot/eval_lsf.py --mode auto`; tables `pilot/final_bigtable.py`, `pilot/grid_table.py`.
- Parallel evaluation: `pilot/par_grid.sh` (CS cluster), `pilot/quest_cell.sh` + `pilot/wait_quest.sh` +
  `pilot/pull_quest.sh` (Quest H100s).
- Tests: `tests/test_route_b.py` (25), `tests/test_route_j.py` (6), including causality tests that the
  context encoding and the predictor never see the future.
- Branch `world-model-ts` at https://github.com/Windsao/WorldModel4TS.
- Full experimental record: `VIDEO_TS_RESCUE_RESULTS.md` §1.10–1.13; plan `PAPER_PLAN.md`.

## 17. Writing guidance

- Lead the introduction with the diagnosis (§8), not with "video models are good at dynamics".
- Define "time-series world model" explicitly as the trained model (past frames → future frames) and
  "dynamics prior" as what the backbone brings; do not call VideoMAE a world model.
- Report MSE and MAE together everywhere; never report only the metric we win.
- Every table must state: one checkpoint, one inference rule, all cells, zero-shot.
- When quoting the efficiency ratio, say "steps and pretraining samples", never "compute".
