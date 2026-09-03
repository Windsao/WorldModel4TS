# Video-VisionTS Recovery Experiments

## 0. Purpose

This document specifies the next experiments after `VIDEO_VISIONTS_RESULTS.md`.
It is an implementation-and-execution contract for Claude Code, not a discussion draft.

The goal is **not** to win on every dataset. The primary goal remains:

```text
Q_all = geometric_mean_d(MSE_method,d / MSE_VisionTS,d) < 1
```

The existing best strict result, `dense_static/right_10`, has `Q_all = 1.0391`.
It is therefore a near miss: the new method needs roughly a 4% aggregate improvement, not a
complete redesign justified by a decisive negative result.

This recovery study has two complementary tracks:

1. repair and re-evaluate the label-free patch-stat lens, whose previous training target was
   incorrect;
2. change the **patch layout** so that future-only masked patches look more like VideoMAE's
   random tube-mask pretraining distribution than a contiguous right block.

Do not edit, delete, or overwrite the old JSONs. The old Stage E1 outputs must remain available
for provenance but must be labelled `invalid_target_bug` in the new report.

## 1. Facts already established

Treat the following as fixed evidence, not as hyperparameters to re-select:

```text
VisionTS positive control:                         PASS
VideoMAE packing and native-target reproduction:  PASS
dense_static/right_10 strict Q_all:                1.0391
dense_static/right_10 strict Q_heldout:            1.0386
pretrained / zeroed aggregate:                     0.9601
pretrained / random aggregate:                     0.9553
strict wins versus VisionTS:                       ETTh1, solar
strict losses versus VisionTS:                     ETTh2, ETTm2, electricity, traffic
```

The strongest diagnostic observation is:

```text
dense_static random_tube_75 native ratio:  0.585 / 0.640  (ETTh2 / ETTm2)
dense_static right_10 native ratio:         0.929 / 0.927
```

The pretrained decoder contains useful signal, but most of that signal is lost when the mask
becomes a one-sided contiguous block. In addition, VideoMAE's `norm_pix_loss=True` target
removes each hidden cube's raw mean and standard deviation, which are unavailable at forecast
time.

### 1.1 The previous E1 result is invalid

The old training path called:

```python
render_batch(xb, ..., blank_future=True)
```

and then treated the returned `vf` as a historical pseudo-future target. With
`blank_future=True`, however, `vf` was constructed from a zero time series. The lens therefore
learned the statistics of a nearly constant rendered block, not the statistics of
`xb[:, L:L+H]`.

The following claims in the previous result report must not be reused:

```text
"E1 fails decisively"
"the statistics do not transfer across datasets"
"the label-free bridge does not remove the bottleneck"
```

They have not yet been tested by a valid experiment.

## 2. Non-negotiable experimental protocol

### 2.1 Regimes

Keep the regimes separate:

```text
HA  historical audit: known blocks cut entirely from observed training history
LA  label-free adaptation: train only on historical pseudo-futures
ZS  strict forecast: genuine test future is used only for final metrics
```

No ZS target may be used for renderer selection, layout selection, hyperparameter selection,
early stopping, checkpoint selection, clipping-range selection, or debugging.

### 2.2 Dataset roles

Use:

```text
selection datasets: ETTh2, ETTm2
held-out datasets:  ETTh1, electricity, traffic, solar
final table:         all six datasets
```

All layout and lens choices must be frozen using only historical pseudo-targets from ETTh2
and ETTm2. Once frozen, run the genuine-future evaluation on all six. Per-dataset wins are
reported but are not gates.

### 2.3 Preserve the old baseline

Do not modify `pilot/run_video_visionts_adapt.py` in place and then write into its old output
directory. Implement a versioned recovery path and use a new output root:

```text
pilot/results_field/video_visionts_recovery/
```

Copy or import stable utilities where appropriate, but make every new JSON self-describing.

## 3. Stage R0 — repair the pseudo-future target first

Implement a target builder with no ambiguous `blank_future` switch:

```python
def make_historical_pair(xb, P, H, L, cols):
    ctx = xb[:, :L]
    pseudo_future = xb[:, L:L + H]
    video_input, video_full, mu, sd = dense_static(
        ctx, pseudo_future, P, cols
    )
    return video_input, video_full, mu, sd
```

`dense_static` already blanks the hidden block in `video_input`; passing the real historical
pseudo-future does not leak it into the encoder input. It is required so that `video_full`
contains the correct target.

Use the exact native-target utility rather than duplicating its statistics logic:

```python
_, cube_mean, cube_std = native_targets(video_full)
target_mean = cube_mean[bool_masked_pos]
target_log_std = log(cube_std[bool_masked_pos])
```

The target must use the same `unbiased=True` variance and `+1e-6` convention as the installed
VideoMAE implementation.

### 3.1 Required R0 tests

These tests must pass before launching a GPU training job:

1. changing `pseudo_future` by a large amount leaves `video_input` bit-identical;
2. changing `pseudo_future` changes `video_full` in the hidden region;
3. changing `pseudo_future` changes masked `target_mean` or `target_log_std`;
4. visible pixels of `video_full` equal visible pixels of `video_input`;
5. masked pixels of `video_input` are exactly the configured blank value;
6. target stats from the target builder exactly match `native_targets(video_full)`;
7. at least one deterministic nonconstant fixture has nonzero masked-cube target variance;
8. the old `blank_future=True` path is reproduced in a regression test and explicitly shown
   to be insensitive to the pseudo-future;
9. step-zero lens predictions exactly equal the causal-stat baseline;
10. no test or training function accepts a genuine evaluation future tensor.

For each real historical audit, also report:

```text
fraction of target cube/channel std <= 1e-5
median and 5/95 percentiles of target mean
median and 5/95 percentiles of target std
```

Stop if more than 50% of the target cube/channel standard deviations collapse below `1e-5`.
That indicates another degenerate renderer/target rather than a learnable stat problem.

## 4. Stage R1 — leakage-safe historical manifests

For each selection dataset, split the official training range chronologically into:

```text
first 60%:  lens training
next 20%:   validation and early stopping
last 20%:   frozen historical audit
```

Purge at least `L + H` raw time steps between adjacent partitions so overlapping windows
cannot cross a boundary. Assign windows to a partition by forecast origin before flattening
channels.

Use fixed manifests shared by all candidates and controls:

```text
train: up to 1500 window/channel pairs per dataset
validation: 512 pairs per dataset
historical audit: 512 pairs per dataset
stride: 32 unless fewer than 512 pairs are available
manifest seeds: 0, 1, 2
```

Store raw origin indices, selected channel indices, data hash, pair hash, split boundaries,
purge width, period `P`, context `L`, and horizon `H`. Assert that the three origin sets do not
overlap after accounting for context and pseudo-future support.

## 5. Stage R2 — preprocessing/layout candidates

Keep the time-series normalization, dense period matrix, ImageNet normalization, backbone,
and cube packing fixed. Change only patch allocation/layout. This isolates the mask-topology
hypothesis.

### R2.0 Width rule

Replace the globally hard-coded `right_10` choice with the pre-registered VisionTS allocation
rule:

```python
visible_cols = max(1, int((L / (L + H)) * 14 * 0.4))
masked_cols = 14 - visible_cols
```

This yields:

```text
ETTh1, ETTh2, electricity, traffic: visible 4, masked 10
ETTm2, solar:                       visible 5, masked 9
```

Add `right_9` support and tests. This rule is dataset-shape-dependent but not dataset-tuned;
it must be applied mechanically to every dataset.

### R2.1 Layout G0 — contiguous baseline

Canonical dense period matrix with the mechanically selected right block. This reproduces
the valid Stage-D baseline, except that ETTm2 and solar use `right_9`.

### R2.2 Layout G1 — interleaved columns

Construct the canonical full image first, divide it into the `14 x 14` spatial patch grid,
then apply a fixed reversible column permutation:

1. preserve the left-to-right order of canonical visible columns;
2. place those visible columns at approximately evenly spaced physical columns;
3. preserve the order of canonical future columns in the remaining physical columns;
4. mask exactly the physical destinations occupied by future columns;
5. apply the same permutation to every frame and channel;
6. invert the permutation before converting the reconstructed future back to a time series.

For four visible columns, use a deterministic evenly spaced placement generated by code,
not a hand-tuned dataset-specific list. For five visible columns, use the identical rule.

This candidate keeps vertical period-phase structure intact while breaking the single large
right-side hole into narrower masked stripes.

### R2.3 Layout G2 — blue-noise patch scatter

Construct the same canonical patch grid, then use a fixed reversible patch permutation:

1. choose exactly `visible_cols * 14` physical positions by deterministic farthest-point
   sampling on the `14 x 14` grid;
2. map canonical visible patches, in row-major order, to those physical positions;
3. map canonical future patches, in row-major order, to all remaining positions;
4. use the future destinations as the tube mask, repeated over all eight tubelets;
5. use layout seeds 0, 1, and 2 to test topology robustness;
6. inverse-permute the reconstructed cubes before canonical dense-matrix inversion.

The mask must never be sampled independently of the content mapping: every masked physical
patch must correspond to a future canonical patch, and every visible physical patch must
correspond to context.

This is the closest forecast-valid analogue of the successful `random_tube_75` audit. It
tests whether VideoMAE can use its inpainting prior when the future is encoded into scattered
slots rather than a contiguous extrapolation block.

### R2.4 Statistics rules for scattered layouts

Evaluate all layouts with:

```text
oracle_stats               diagnostic only
context_global             strict causal
nearest_visible_physical   strict causal
```

`nearest_visible_physical` copies mean/std from the closest visible spatial patch in the same
tubelet using Manhattan distance. Resolve distance ties deterministically by smallest
row-major token index. It may read only entries where `bool_masked_pos=False`.

After inverse permutation, also evaluate a canonical-boundary rule that copies from the last
canonical visible column. This separates benefit from mask topology from benefit from the new
physical-neighbor stat estimate.

## 6. Stage R3 — historical layout audit

Run G0, G1, and G2 on historical pseudo-futures from ETTh2 and ETTm2 before training a lens.

Required controls:

```text
pretrained VideoMAE logits
zero logits
randomly initialized VideoMAE for G0 and any promoted layout
random_tube_75 audit-only upper anchor from the existing results
```

Required metrics, accumulated by total element count rather than unweighted batch means:

```text
native normalized-cube MSE and ratio to zero logits
raw-pixel MSE with oracle stats
raw-pixel MSE with every causal stat rule
pseudo-future time-series MSE and MAE with oracle stats
pseudo-future time-series MSE and MAE with every causal stat rule
zero-logit versions through the identical inverse path
seam error after inverse permutation
target-stat distribution diagnostics from Stage R0
```

Save visual sheets in both physical and inverse-permuted canonical coordinates.

### 6.1 Layout promotion gate

Aggregate ETTh2 and ETTm2 with an equal-dataset geometric mean. Promote at most two layouts.
A layout is eligible when all of the following hold on the frozen historical-audit partition:

1. aggregate native ratio is below `0.85`, or improves at least 10% relative to G0;
2. pretrained oracle time-series MSE is at least 5% below zero-logit oracle MSE;
3. predictions and inverse transforms are finite and time ordering tests pass;
4. the conclusion is not carried by only one layout seed;
5. no hidden canonical patch is visible after permutation.

Rank eligible layouts by aggregate causal pseudo-future MSE. If neither G1 nor G2 passes,
retain G0 only and proceed to the corrected-lens test. Do not inspect genuine test futures to
rescue a failed layout.

## 7. Stage R4 — corrected patch-stat lenses

Run these variants on G0 and each promoted layout.

### L0 — no learned lens

The best strict causal-stat rule from Stage R3. This is the zero-trainable-parameter baseline.

### L1 — corrected reproduction

The existing `StatLens` architecture with only the target bug fixed. Keep the original
settings to isolate the effect of the repair:

```text
parameters < 500k
AdamW, lr = 1e-3, weight_decay = 1e-2
600 steps, batch 8
zero-initialized final head
```

### L2 — bounded residual lens

Use the same inputs and parameter budget, but prevent a learned bridge from catastrophically
destroying the causal starting point:

```python
pred_mean = clamp(boundary_mean + 0.25 * tanh(delta_mean), 0.0, 1.0)
pred_std = boundary_std * exp(2.0 * tanh(delta_log_std))
```

The final layer remains zero-initialized, so step zero must equal L0 up to floating-point
tolerance. The standard deviation stays positive and can change by at most a factor of
`exp(2)` in either direction from the causal starting point; it is not hard-clamped away from
that starting point. Use Smooth-L1 loss on raw mean and log-standard-deviation. Use `lr=3e-4`,
at most 1500 steps, validate every 50 steps, and stop after six validation checks without
improvement.

### L3 — bounded tube-shared lens

Dense-static videos repeat the same image through time, and the true raw statistics for a
fixed spatial patch are identical across all eight tubelets. Exploit that known symmetry:

1. map masked logits back to `(tubelet, spatial_patch)` identity;
2. project logits, then average the projected representation over the eight tubelets for the
   same spatial patch;
3. use spatial position only, not temporal position, for the stat prediction;
4. predict one bounded mean/std pair per spatial patch and broadcast it to all tubelets;
5. compute the stat loss once per unique spatial patch rather than eight times.

Use the L2 optimizer and early-stopping settings. Add an ordering test proving that each group
contains exactly the eight tubelets of one spatial location.

### 7.1 Training and selection rules

Use only the R1 training and validation manifests. Checkpoint selection minimizes:

```text
R_val = geometric_mean_d(
    pseudo_TS_MSE(candidate, d) / pseudo_TS_MSE(L0, d)
)
```

Do not select on stat loss alone. A lower mean/log-std loss that does not improve reconstructed
time-series MSE is not useful.

For the promoted lens/layout combinations, run seeds 0, 1, and 2. Then train the identical
lens on a frozen random-init VideoMAE using:

```text
identical lens initialization
identical historical manifests and batch order
identical optimizer and stopping rule
identical layout and mask
```

### 7.2 Lens promotion gate

Select one primary combination and optionally one ablation using only the frozen historical
audit partition. The primary combination must satisfy:

1. aggregate pseudo-future MSE versus L0 is `<= 0.97`;
2. neither selection dataset is more than 10% worse than L0;
3. all three seeds improve the aggregate over their own step-zero checkpoint;
4. pretrained-backbone aggregate is `<= 0.97` relative to the paired random backbone, or the
   result is explicitly labelled `bridge-only` rather than video-pretraining attribution;
5. no predicted stat is NaN/Inf, mean-clamp hit rate is below 25%, and the fractions of
   `|tanh(delta_log_std)| > 0.95` are reported.

If no learned lens passes, the best non-learned layout may still be promoted if it improves
historical causal reconstruction. Do not run a failed lens on genuine futures merely because
training completed.

## 8. Stage R5 — frozen genuine-future evaluation

Before starting R5, save a frozen configuration JSON containing:

```text
renderer and normalization
width rule
layout name and seed rule
patch permutation hashes
mask hashes
causal-stat rule
lens architecture and checkpoint hashes, if any
training manifests and hyperparameters
selection metrics used to choose it
```

Once this file is written, do not replace the primary configuration after seeing an R5
result. Run the primary on:

```text
ETTh1, ETTh2, ETTm2, electricity, traffic, solar
```

Use the same test manifests as the existing Stage D so metrics are paired. Run an optional
ablation only if it was also frozen before R5.

Required controls on the identical inputs and manifests:

```text
pretrained VideoMAE complete method
zero decoder logits through the same layout/stat inverse path
random-init VideoMAE with the identical bridge, if a bridge is used
existing dense_static/right_10 result
official VisionTS
smean
snaive
```

For every method, save per-origin squared error, not just for the pretrained arm. Bootstrap
paired origins with a dataset-specific block length:

```python
block_d = ceil(H_d / stride_d)
```

The summarizer must group results by the full configuration identity, not only by dataset, so
multiple layouts cannot overwrite one another.

### 8.1 Primary success criteria

The main claim is met when:

```text
Q_all < 1
```

No 6/6 win requirement exists. Report:

```text
all six MSE and MAE values
all six ratios versus VisionTS
win/tie/loss count
Q_all and Q_heldout
95% paired moving-block bootstrap intervals
leave-one-dataset-out Q_all
ratios versus zero logits and random backbone
ratios versus smean and snaive
```

The stronger transfer claim requires:

```text
Q_heldout <= 0.97
upper endpoint of the 95% CI for Q_heldout < 1
pretrained / paired-random held-out ratio <= 0.97
```

The current deficits on electricity and traffic are the largest. As a diagnostic only, if all
other datasets remain unchanged, reducing both of those MSEs by about 11% is sufficient to
move the current six-dataset geometric mean below 1. This is not permission to tune on their
test futures.

## 9. Conditional Stage R6 — raw-pixel decoder adaptation

Run R6 only if:

```text
native/oracle historical reconstruction remains useful,
but no corrected lens passes the R4 historical gate.
```

Freeze the VideoMAE patch embedding and encoder. Train the decoder and prediction projection
on historical pseudo-mask tasks to predict raw `[0,1]` cube pixels directly, rather than
per-cube normalized targets. Use the best frozen layout from R3.

Required pair:

```text
pretrained frozen encoder + raw-pixel decoder adaptation
random frozen encoder     + identically initialized raw-pixel decoder adaptation
```

Use identical manifests, batches, steps, and early stopping. Evaluate R6 first on the R1
historical partitions and apply the same 3% promotion and attribution gates. Only a promoted
R6 model may enter R5-style genuine-future evaluation.

Do not infer that R6 will fail from the old E1 outputs; those outputs used the wrong target.

## 10. Files to implement

Prefer new versioned files so the invalid run remains reproducible:

```text
pilot/video_visionts_layouts.py
    patch permutations, dynamic width rule, masks, inverse permutations

pilot/run_video_visionts_recovery_audit.py
    R0-R3 historical target validation and layout audit

pilot/run_video_visionts_stat_lens_v2.py
    corrected L1, bounded L2, tube-shared L3, paired random control

pilot/run_video_visionts_recovery_forecast.py
    frozen R5 strict forecast

pilot/run_video_visionts_raw_decoder.py
    conditional R6 only

pilot/summarize_video_visionts_recovery.py
    full-config grouping, tables, CIs, gates, invalid-old-run annotation

pilot/run_video_visionts_recovery.sh
    sequential resumable launcher with stage gates

tests/test_video_visionts_recovery.py
    all R0/layout/leakage/aggregation tests
```

Write outputs under:

```text
pilot/results_field/video_visionts_recovery/
  manifests/
  target_audit/
  layout_audit/
  lens_checkpoints/
  historical_eval/
  frozen_configs/
  strict_forecast/
  raw_decoder/
  previews/
  logs/
  summary.json
  summary.md
```

The final human-readable report should be written separately as:

```text
VIDEO_VISIONTS_RECOVERY_RESULTS.md
```

## 11. Required implementation tests

At minimum, implement all of the following:

1. historical `video_input` is invariant to pseudo-future poisoning;
2. historical `video_full` is sensitive to pseudo-future poisoning;
3. masked target mean/std are sensitive to pseudo-future poisoning;
4. target stats equal `native_targets(video_full)` exactly;
5. `right_9` has 126 masked spatial patches and is a valid tube mask;
6. dynamic width rule returns the expected 4/10 and 5/9 allocations;
7. G1 patch/column permutation round-trips exactly;
8. G2 patch permutation round-trips exactly for seeds 0, 1, and 2;
9. every physical masked patch maps to a canonical future patch;
10. every physical visible patch maps to canonical context;
11. changing any canonical future patch cannot change the model input;
12. nearest-visible physical stats never read a masked cube;
13. inverse permutation plus dense inversion preserves exact time order on a ramp fixture;
14. tube-shared groups contain one spatial patch across exactly eight tubelets;
15. dense-static target stats are equal across those eight tubelets;
16. L1/L2/L3 step-zero output equals L0;
17. L2/L3 means stay in `[0,1]`, standard deviations stay positive, and their ratio to the
    causal standard deviation stays within the declared `exp(-2)` to `exp(2)` interval;
18. train/validation/audit raw support intervals are disjoint after purging;
19. paired pretrained/random runs use identical lens weights and pair manifests;
20. genuine-future poisoning cannot change any R5 encoder input;
21. summarizer keeps G0/G1/G2 and different lens variants in separate groups;
22. synthetic fixtures verify `Q_all`, `Q_heldout`, leave-one-out, and CI gates;
23. new outputs cannot resolve to the old `label_free_adaptation` paths;
24. all tensors, metrics, and saved predictions are finite.

No GPU sweep is valid until the CPU-only correctness subset passes. Model-dependent packing
and leakage tests must then pass on the GPU environment before the first training job.

## 12. Execution order and stop rules

Run in this exact order:

```text
R0 tests and target-distribution audit
  fail -> stop and fix correctness

R1 manifest construction and disjointness tests
  fail -> stop and fix leakage

R2/R3 historical layout audit on ETTh2 + ETTm2
  -> retain at most two layouts

R4 seed-0 lens screen
  -> retain only historical improvements

R4 seeds 0/1/2 + paired random-backbone control
  -> freeze one primary and at most one ablation

R5 all-six strict forecast
  -> report regardless of whether Q_all crosses 1

R6 only if the explicit conditional gate is met
```

Do not run all six datasets for every preprocessing/lens candidate. The only all-six jobs are
the frozen primary and optional pre-registered ablation.

## 13. Required final interpretation

The recovery report must choose one of these bounded conclusions:

```text
A. Corrected E1 works and beats VisionTS in aggregate, with pretrained attribution.
B. A scattered future layout works and beats VisionTS, with or without a learned stat lens.
C. The complete method beats VisionTS, but the paired-random control shows the gain is bridge-only.
D. Historical gates pass but genuine-future transfer remains below VisionTS.
E. Corrected lenses fail historical validation; the previous E1 result was invalid, but the
   corrected result is now a valid negative.
F. No layout preserves the random-mask advantage when masked patches are constrained to be
   future-only; mask/task mismatch is the remaining bottleneck.
G. Results are inconclusive because a correctness, leakage, provenance, or robustness gate failed.
```

Do not use number of per-dataset wins as the primary conclusion. A method may lose on some
datasets and still satisfy the project objective when `Q_all < 1`.
