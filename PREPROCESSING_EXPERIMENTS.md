# VideoMAE Input-Alignment Experiments

## 0. Objective

Design, implement, and run a controlled experiment to answer:

> Can a better time-series-to-video preprocessing method make a pretrained VideoMAE backbone contribute useful forecasting signal?

The goal is **not** merely to obtain a lower forecasting MSE with a larger trainable head. A preprocessing method counts as successful only when the pretrained backbone performs measurably better than an identically configured randomly initialized backbone.

This document is an implementation and execution specification for Claude Code. Work from the repository root on the current `field-video-clean` branch. Preserve all existing user changes and results.

## 1. Current evidence and known confounds

### 1.1 Completed frozen-backbone results

The recently completed runs all lose to the seasonal-mean baseline:

| Dataset | Frozen VideoMAE MSE | Seasonal-mean MSE | Relative gap |
|---|---:|---:|---:|
| ETTh1 | 0.4960 | 0.4022 | +23.3% |
| ETTh2 | 0.3721 | 0.3496 | +6.4% |
| electricity | 0.3009 | 0.2071 | +45.3% |

The traffic run reached the end of training but failed during evaluation because several jobs shared one GPU. The local user change in `pilot/run_field.py` reduces the evaluation batch from 256 to 64; preserve that change.

Backbone-scale and source-dataset experiments also show no positive transfer under the current renderer: Kinetics, Something-Something V2, UCF101, base/large checkpoints all give similar adapter results, while the random backbone is sometimes slightly better.

### 1.2 Why the existing renderer is poorly aligned

The current univariate renderer creates a grayscale “barcode”:

- each frame contains one period;
- the value is encoded mainly as column brightness;
- the single row is tiled vertically;
- resizing uses nearest-neighbor interpolation;
- the final backbone tokens are globally mean-pooled.

This conflicts with VideoMAE pretraining in several ways:

1. VideoMAE was trained on 16-frame natural videos containing 2-D contours and spatially localized, coherent motion.
2. Its patch embedding uses `2 x 16 x 16` tubelets. Large regions in the barcode are constant or nearly constant within a tubelet.
3. The `MCG-NJU/videomae-base` checkpoint has `norm_pix_loss=true`: the reconstruction target is normalized within each tubelet. Encoding a number primarily as uniform patch brightness therefore does not match what the pretrained objective encouraged the model to preserve.
4. Global mean pooling removes the `8 x 14 x 14` token layout before forecasting, so temporal order and spatial location can be lost even if the backbone encodes them.

### 1.3 The existing frozen probe is not yet a valid final diagnostic

Preserve the user's numerical-stability fix in `pilot/probe_frozen.py`, but do not use its current `skill_ratio` as the final representation test without correcting the target.

The renderer instance-normalizes each context window, while the probe currently fits the rendered features directly to the original globally standardized future target. The forecasting model instead predicts normalized future values and then applies `prediction * context_sd + context_mu`.

For a fair representation probe, use:

```text
Y_norm = (Y - context_mu) / context_sd
```

The new run also reports participation ratios around 114–129 rather than the old 5–13. The reason for this discrepancy must be recorded with checkpoint ID, `transformers` version, and code commit before making another feature-collapse claim.

## 2. Core hypotheses

Test the following hypotheses separately.

### H1 — Smooth interpolation

Nearest-neighbor resizing creates block boundaries and unnatural high-frequency artifacts. Bilinear resizing may better match visual pretraining.

### H2 — Geometry instead of brightness

Encoding a value as the vertical location of an antialiased line should preserve the signal through patch-level normalization better than encoding it as uniform grayscale intensity.

### H3 — Coherent motion

A sequence of smoothly evolving or scrolling plots should be closer to natural-video temporal statistics than independent barcode frames, a static image repeated 16 times, or global frame flicker.

### H4 — Two-dimensional local structure

A rolling period matrix or recurrence plot may provide non-degenerate 2-D texture and local neighborhoods that VideoMAE can process.

### H5 — Structured token readout

An aligned renderer may still fail if all spatiotemporal tokens are averaged. Preserving token structure should improve the pretrained-minus-random gap.

### H6 — Pretraining-task alignment

If all encoder-plus-head variants fail, forecasting should be reformulated as pretrained masked video reconstruction using `VideoMAEForPreTraining`, rather than adding an unrelated regression head.

## 3. Preprocessing candidates

Implement all renderers in a new file such as `pilot/preprocess_renderers.py`. Do not replace the existing `FieldVMAE.render` implementation; the original renderer is a required control.

Every renderer must:

- consume context only, never future values;
- accept univariate input shaped `[B, L, 1]`;
- output `[B, 16, 3, 224, 224]`;
- apply one context-wide normalization and return its `mu` and `sd`;
- use the same normalization across all 16 frames;
- apply the checkpoint's ImageNet channel normalization only after rendering;
- be deterministic for a fixed input;
- save a 4-by-4 contact-sheet preview for visual inspection.

Use six candidates:

### R0 — `barcode_nearest`

Exact reproduction of the current univariate period renderer. This is the negative control.

```text
frame = one period
x-axis = phase
pixel brightness = normalized value
row is tiled vertically
resize = nearest neighbor
```

Verify numerically that this renderer matches the current implementation on the same tensor.

### R1 — `barcode_bilinear`

Identical to R0 except for bilinear spatial resizing. This isolates interpolation from every other design choice.

### R2 — `period_matrix`

Each frame is a rolling heatmap of the most recent eight periods:

```text
height = period index, oldest to newest
width = phase within a period
pixel brightness = normalized value
resize = bilinear
```

At early frames, pad missing history with the earliest available period. Across frames the matrix should move smoothly as one period enters and one leaves.

### R3 — `period_line`

Recommended primary candidate. Each frame is an antialiased line plot of one period:

```text
x-axis = phase
y-axis = normalized value
16 periods = 16 frames
```

Implementation constraints:

- use one robust scale shared by the full context;
- clip standardized values at approximately `[-3, 3]`;
- reserve about 10–16 pixels of vertical margin;
- use an antialiased line approximately 3–4 pixels wide;
- use a plain neutral background;
- do not draw axes, text, ticks, legends, or grids;
- start with grayscale replicated into RGB;
- render directly at 224-by-224 or use smooth interpolation only.

The value must be recoverable primarily from contour position, not mean brightness.

### R4 — `period_trails`

Draw the current period plus the previous three period curves in every frame. Use decreasing intensity for older curves, for example `1.0, 0.65, 0.40, 0.25`.

This tests whether object persistence and coherent contour motion are more useful than a single independently changing line.

Do not use future periods when constructing a frame.

### R5 — `recurrence`

Each frame is a soft recurrence plot computed from a rolling context segment:

```text
R[i,j] = exp(-abs(z[i] - z[j]) / tau)
```

Recommended initial settings:

- rolling segment: four periods;
- resample the segment to length 64 before forming the pairwise matrix;
- `tau = 0.25` after clipping the standardized input to `[-1, 1]`;
- bilinear resize from 64-by-64 to 224-by-224.

This is a 2-D texture control. Do not assume it will win: recurrence encoding may obscure the direction of time and the inverse mapping needed for forecasting.

## 4. Normalization ablation

Do not sweep many normalization choices initially. Use the following default for every renderer:

```text
z = (x - context_mean) / context_std
z = clip(z, -3, 3) / 3
```

After selecting the best renderer, perform a small normalization ablation on the two screening datasets only:

| Variant | Definition |
|---|---|
| `std_clip_2` | standardize, clip to `[-2,2]` |
| `std_clip_3` | standardize, clip to `[-3,3]` — default |
| `robust` | subtract median and divide by IQR, then clip |
| `visionts_r04` | scale normalized values to standard deviation approximately 0.4 before rendering |

Keep context de-normalization identical in the output path. Do not normalize each frame independently.

## 5. Readout controls

Implement a new experiment model in `pilot/run_preprocess.py`. Reuse dataset loading, window construction, baselines, and training utilities from `pilot/run_field.py`, but keep this experiment isolated from the established entry point.

Support three frozen-backbone readouts:

### P0 — `global`

Exact current control:

```text
tokens -> mean over all 1568 tokens -> existing MLP head
```

### P1 — `temporal`

Preserve temporal order without greatly increasing head capacity:

```text
tokens [B, 1568, 768]
-> reshape [B, 8 tubelets, 14, 14, 768]
-> mean over the 14-by-14 spatial grid
-> flatten [B, 8*768]
-> LayerNorm + Linear to horizon
```

### P2 — `xattn`

Use eight learned forecast queries with cross-attention over all 1568 frozen tokens, followed by a regression head. Do not add NLinear or another raw-series branch; otherwise the video contribution cannot be identified.

For every readout:

- freeze every encoder parameter;
- train only the specified readout;
- apply the same `mu/sd` output de-normalization;
- initialize the readout identically for the pretrained and random-backbone pair;
- report total and trainable parameter counts;
- use exactly the same train/test windows for paired comparisons.

Primary screening should use `xattn`, because it is least likely to discard renderer information. After selecting the top renderer, run all three readouts to determine whether preprocessing or pooling caused the gain.

## 6. Required controls

Every claimed result needs the following paired runs:

### C0 — Pretrained backbone

```text
MCG-NJU/videomae-base
pretrained weights loaded
encoder frozen
```

### C1 — Random backbone

```text
same VideoMAE config and parameter count
randomly initialized
encoder frozen
same renderer, readout, data, seed, and optimizer
```

### C2 — Seasonal mean

Compute `smean` on exactly the same test windows in every JSON.

### C3 — Current renderer

Always include `barcode_nearest`; do not compare a new renderer only against old result files produced by a different code path.

Optional only after a candidate succeeds: compare Kinetics-pretrained with Something-Something-V2-pretrained. Do not start another broad checkpoint sweep before input alignment shows a positive pretrained-minus-random gap.

## 7. Dataset selection and selection protocol

Use six of the seven datasets:

```text
ETTh1, ETTh2, ETTm2, electricity, traffic, solar
```

Exclude ETTm1 because it is the most redundant with ETTm2. This set retains:

- small, 7-channel hourly datasets: ETTh1 and ETTh2;
- a small, high-frequency dataset: ETTm2;
- high-channel datasets: electricity, traffic, and solar;
- periods 24, 96, and 144.

Do not use all six datasets to select the renderer.

### Screening datasets

```text
ETTh2 + electricity
```

They represent low-channel and high-channel regimes and have different failure margins.

### Held-out validation datasets

```text
ETTh1 + ETTm2 + traffic + solar
```

The preprocessing choice must be frozen before these four results are inspected. A positive research claim should be based primarily on these held-out datasets, not the two used to select the method.

## 8. Stage A — Renderer audit without forecasting claims

Before training, save one contact sheet per renderer and dataset. Manually verify:

- no future leakage;
- time runs in the intended direction;
- all 16 frames differ when expected;
- line plots are antialiased and not clipped excessively;
- no frame-wise normalization removes inter-period changes;
- no renderer produces NaN/Inf.

Compute and save the following input statistics on at least 128 samples:

```text
fraction of nearly constant 2x16x16 tubelets
mean tubelet pixel standard deviation
spatial gradient RMS
temporal difference RMS
temporal/spatial energy ratio
```

These are diagnostics, not success metrics. Do not select the final renderer solely because its pixel statistics look more “natural.”

## 9. Stage B — Corrected feature probe

Implement `pilot/probe_preprocessing.py` or extend the existing probe without deleting its old output schema.

For each renderer on ETTh2 and electricity:

1. extract frozen features from pretrained and random backbones;
2. evaluate layers 3, 6, 9, and 12;
3. test both global pooled features and a temporal-order-preserving summary;
4. predict normalized future target `Y_norm`;
5. select ridge regularization on a validation split using float64/SVD;
6. report test `probe_mse / constant_mse`;
7. record checkpoint name/hash, library versions, sample indices, and seed.

Use the exact same samples for every renderer and backbone. Cache sample indices, not merely the RNG seed.

The feature probe is a screening signal. Do not call it positive transfer unless pretrained features beat random features under the same probe.

## 10. Stage C — Fast supervised screen

Run all six renderers on ETTh2 and electricity with:

```text
readout       = xattn
backbones     = pretrained and random
context       = default 16 periods
horizon       = 96
max_ch        = 112
epochs        = 2
ft_cap        = 5000
stride        = 32
batch         = 16
learning rate = 1e-4
seed          = 0
```

This is 24 runs:

```text
6 renderers x 2 datasets x 2 backbone initializations
```

Run jobs sequentially on one GPU unless GPU memory measurements prove that concurrency is safe. The previous traffic failure was caused by concurrent jobs and an evaluation batch of 256.

Rank renderers by both:

```text
absolute_gain = (smean_mse - pretrained_mse) / smean_mse
transfer_gain = (random_mse - pretrained_mse) / random_mse
```

Advance at most the top two renderers. A renderer that lowers absolute MSE but does not improve over the random backbone has not demonstrated use of pretrained video capability.

## 11. Stage D — Readout and normalization ablation

For the top one or two renderers, still on ETTh2 and electricity:

1. compare `global`, `temporal`, and `xattn` readouts;
2. compare the four normalization variants in Section 4;
3. run pretrained and random pairs for every configuration;
4. keep all other hyperparameters fixed.

Choose exactly one renderer/readout/normalization configuration before proceeding to held-out validation.

Do not choose separate preprocessing hyperparameters for each dataset. The purpose is to find a transferable input bridge.

## 12. Stage E — Six-dataset validation

Run the frozen configuration selected in Stage D on:

```text
ETTh1, ETTh2, ETTm2, electricity, traffic, solar
```

Use:

```text
horizon       = 96 for all except solar
solar horizon = 144
context       = default 16 periods
max_ch        = 112
epochs        = 5
ft_cap        = 40000
stride        = 8
batch         = 16
learning rate = 1e-4
seed          = 0
```

Run both pretrained and random backbones. The four held-out datasets are the primary validation set; ETTh2 and electricity are reported but marked as renderer-selection datasets.

If seed 0 passes the positive criteria, repeat pretrained and random runs with seeds 1 and 2. Do not spend three-seed compute on configurations that fail seed 0.

## 13. Positive-result criteria

A preprocessing configuration is considered a credible positive result only if all conditions hold:

1. **Absolute utility:** pretrained VideoMAE beats `smean` on at least three of the four held-out datasets.
2. **Pretraining utility:** pretrained VideoMAE beats the matched random backbone by at least 3% mean MSE on the held-out datasets.
3. **Consistency:** pretrained-minus-random improvement has the same sign on at least three of four held-out datasets.
4. **Multi-seed confirmation:** after seeds 0, 1, and 2, the mean transfer gain remains larger than the observed run-to-run noise.
5. **No capacity confound:** pretrained and random runs use identical trainable readouts and parameter counts.
6. **No selection leakage:** no renderer or hyperparameter is changed after looking at held-out results.

Report partial success honestly. For example, “better than smean but no better than random” means the preprocessing helps the trainable head, not that VideoMAE pretraining transfers.

## 14. Stage F — Strict zero-shot reconstruction, only if encoder transfer fails

If no renderer produces positive pretrained-minus-random transfer, implement a separate experiment with `VideoMAEForPreTraining` and its pretrained decoder.

Do not mask the future only as complete future frames without an ablation: VideoMAE was pretrained primarily with spatial tube masks, not causal future-frame masks.

Recommended design:

1. Each of 16 frames contains an antialiased rolling line chart.
2. Context occupies the left spatial region; the requested future occupies a right spatial strip.
3. Use the same right-side mask across frames so it is a spatial tube mask.
4. Earlier rolling frames contain only historical values available before the forecast origin.
5. Decode the predicted line geometry from masked patches and invert the shared context normalization.
6. Confirm by unit test that changing true future values in the input buffer cannot change the prediction.

This is a distinct experiment and should write to a separate results directory. It is the closest analogue of VisionTS because it uses the pretrained reconstruction interface rather than a newly trained forecasting head.

## 15. Files to add

Prefer the following isolated structure:

```text
pilot/preprocess_renderers.py       renderer implementations and image diagnostics
pilot/run_preprocess.py             frozen-backbone paired forecasting entry point
pilot/probe_preprocessing.py        corrected layerwise representation probe
pilot/summarize_preprocessing.py    aggregate pretrained/random/smean results
pilot/run_preprocess_sweep.sh       sequential Stage C/D/E launcher
tests/test_preprocess_renderers.py  renderer shape, determinism, leakage, and invariance tests
```

Do not rewrite or delete existing result JSONs. Use:

```text
pilot/results_field/preprocessing/
  previews/
  probe/
  screen/
  ablation/
  validation/
  logs/
  summary.md
  summary.json
```

Every result JSON must include:

```text
git commit and dirty-worktree flag
dataset and exact data path
renderer and every renderer hyperparameter
readout
pretrained/random flag and checkpoint
transformers and torch versions
context, horizon, channels, windows, stride, ft_cap
optimizer parameters and seed
trainable/total parameter counts
input-alignment diagnostics
snaive, smean, and model MSE/MAE
wall-clock time and peak GPU memory
```

## 16. Required tests

Before GPU runs, add and execute tests for:

1. every renderer returns `[B,16,3,224,224]` and finite values;
2. R0 matches the existing renderer numerically;
3. repeated calls are deterministic;
4. affine changes `a*x+b`, for positive `a`, produce the same normalized rendered geometry up to tolerance;
5. each frame uses context values only;
6. output de-normalization has correct shape and restores scale;
7. pretrained and random paired runs use identical sample indices;
8. result filenames cannot collide across renderer/readout/backbone/seed;
9. summary code correctly computes positive and negative transfer on synthetic JSON fixtures;
10. the evaluation batch remains at the user's memory-safe value of 64 or lower.

Also run one small CPU/GPU smoke experiment with `ft_cap<=64`, `epochs=1`, and a single batch before launching the matrix. Smoke metrics are not research results and must use a separate directory.

## 17. Reporting format

The generated `summary.md` should contain at least these tables.

### Renderer screen

| Dataset | Renderer | Pretrained MSE | Random MSE | smean | Transfer gain | Beats random? | Beats smean? |
|---|---|---:|---:|---:|---:|---|---|

### Held-out validation

| Dataset | PT mean +/- std | Random mean +/- std | smean | Transfer gain | Absolute gain |
|---|---:|---:|---:|---:|---:|

### Final verdict

Explicitly answer:

```text
Did preprocessing improve absolute forecasting?
Did visual pretraining improve over the same random architecture?
Was the gain larger than seed noise?
Did it generalize beyond the renderer-selection datasets?
Which hypothesis was supported or rejected?
```

## 18. Stop conditions

Stop expanding the sweep when any of the following occurs:

- no candidate has positive pretrained-minus-random transfer on either screening dataset;
- a candidate improves only through a larger readout and random performs equally well;
- the result depends on per-dataset preprocessing tuning;
- input statistics look more natural but forecasting and normalized probes do not improve;
- compute failures leave unmatched pretrained/random pairs.

In those cases, preserve the negative result and proceed only to the strict reconstruction experiment in Section 14 or to a learned modality lens. A learned lens trained on time-series data must be described as modality adaptation, not strict zero-shot.

