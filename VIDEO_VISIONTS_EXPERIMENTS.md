# Video-VisionTS Zero-Shot Diagnostic Experiments

## 0. Objective

Implement and run a staged experiment that answers the following question:

> Can a pretrained VideoMAE be turned into a useful zero-shot time-series forecaster by
> giving it the same task interface that makes VisionTS work?

The immediate goal is **not** another broad renderer sweep. The first goal is to identify
which of the following currently prevents strict zero-shot forecasting:

1. **D1 — decoder-domain mismatch:** `MCG-NJU/videomae-base` predicts independently
   normalized cube pixels, not raw pixels that can be directly unpatchified into values;
2. **D2 — mask mismatch:** the current experiment masks a contiguous 21.4% right strip,
   while VideoMAE was pretrained primarily with approximately 90% random spatial tube masks;
3. **D3 — representation mismatch:** the successful VisionTS-style dense period matrix was
   not used by the current strict reconstruction experiment;
4. **D4 — task mismatch:** masked reconstruction may work on known historical regions but
   still fail when the right block represents genuinely unseen future values.

The experiment must separate these explanations. A forecasting MSE alone cannot do that.

This document is an implementation and execution specification for Claude Code. Work from
the repository root on the current branch. Preserve all existing user changes and result
files. Do not rewrite `pilot/run_reconstruct.py`; it is a required legacy control.

## 1. Current evidence and what it does not prove

Read `PREPROCESSING_RESULTS.md` before implementation.

- The supervised frozen-encoder experiment found a tentative signal from
  `period_matrix + xattn`: pretrained beats the matched random encoder on 4/4 held-out
  datasets by a mean of 4.47%. It is single-seed and trains about 2.96M readout parameters,
  so it is **not zero-shot**.
- The strict reconstruction experiment fails on 6/6 datasets: model decoder output is
  worse than replacing the logits with zeros.
- That strict experiment uses a rolling line renderer, three masked patch columns
  (`3/14 = 21.4%`), and a geometry heuristic over normalized VideoMAE decoder logits.
- It therefore does **not** establish that dense period matrices cannot work through a
  video backbone, and it does **not** cleanly distinguish model failure from invalid
  decoder inversion.

Do not describe the old Stage F mask as fully “native VideoMAE masking.” It matches the
temporal tube property because the same spatial positions are masked in every tubelet, but
its spatial topology and masking ratio are far from VideoMAE pretraining.

## 2. Experimental regimes and terminology

Keep the following regimes separate in filenames, tables, and conclusions.

### ZS — strict zero-shot

- no gradients on any time-series data;
- no trained forecasting head;
- no decoder or calibration training;
- period, normalization, renderer, mask, and decoding rules are fixed before test results;
- observed context may be used to compute per-window statistics;
- future values are used only for final metric computation.

### HA — historical-audit reconstruction

- no gradients;
- a known region inside the observed history is hidden and reconstructed;
- its ground truth may be used for diagnostics, including oracle analyses;
- this is **not** a forecast result.

### LA — label-free time-series adaptation

- only observed historical values and synthetic historical masks may be used for training;
- no true forecast target may enter training or model selection;
- report this as self-supervised or label-free adaptation, never as strict zero-shot.

### ST — supervised transfer

The existing trained `xattn` experiments belong here. Do not add another supervised
forecasting readout in this study.

## 3. Shared data protocol

### 3.1 Datasets

Use these six datasets for reference and final validation:

```text
ETTh1, ETTh2, ETTm2, electricity, traffic, solar
```

Use only these two datasets for diagnostic selection:

```text
ETTh2, ETTm2
```

Rationale:

- ETTh2 gives a clear but not catastrophic current reconstruction failure;
- ETTm2 is the closest current model-versus-zeroed result and tests a different frequency;
- both are inexpensive enough for the mask/renderer factorization;
- the remaining four datasets stay untouched until all choices are frozen.

### 3.2 Forecast protocol

Reuse `load_mv`, split borders, `windows`, `snaive`, and `smean` from
`pilot/run_field.py`.

Use:

```text
context L     = 16 * dataset period P
horizon H     = 96 for ETTh1, ETTh2, ETTm2, electricity, and traffic
horizon H     = 144 for solar
max_ch         = 112
test stride    = 8 for final results
audit stride   = 32
audit samples  = at most 256 univariate windows per dataset
final samples  = at most 2000 univariate windows per dataset
seed           = 0 unless a mask-seed ablation is explicitly requested
```

Flatten multivariate datasets channel-independently exactly as in the existing zero-shot
experiment. Cache the selected `(window_index, channel_index)` pairs once and reuse them
byte-for-byte across all models, renderers, masks, and controls.

Before Stage A, write a manifest containing:

```text
dataset
data path and file hash
period P, context L, horizon H
split borders
selected window/channel pairs and their hash
torch/numpy seed
```

### 3.3 Normalization and leakage rule

All normalization statistics must be computed from the visible context only:

```text
mu = mean(visible context)
sd = std(visible context) + eps
z  = 0.4 * (x - mu) / sd
```

Use the same `mu` and `sd` for all 16 frames of one video. Never normalize a frame
independently. When constructing a known pseudo-future target, its values may be transformed
with context-derived `mu/sd`, but may not contribute to those statistics.

The visible/masked boundary must be patch-aligned. Resize the visible matrix and target
matrix separately before concatenation so bilinear interpolation cannot leak a target pixel
into the visible side.

### 3.4 Primary comparison metric

The final forecasting target is **aggregate performance against VisionTS**, not winning every
individual dataset. Because raw MSE scales differ substantially by dataset, never average raw
MSE values across datasets.

For every dataset `d`, compute the paired ratio:

```text
q_d = MSE(VideoMAE method, d) / MSE(VisionTS, d)
```

Aggregate with an equal-dataset geometric mean:

```text
Q(S)       = exp(mean_{d in S}(log(q_d)))
gain_vs_V  = 1 - Q(S)
```

Report both:

```text
Q_all      over all six datasets
Q_heldout  over ETTh1, electricity, traffic, and solar
```

Interpretation:

- `Q < 1` means the method is better than VisionTS in aggregate;
- `Q = 1` is a tie;
- `Q > 1` means VisionTS is better;
- per-dataset win count is descriptive only and is **not** a hard success requirement.

Also compute a paired 95% confidence interval using errors from common original test windows.
Keep all channels and horizon steps belonging to one original window together. Because test
windows overlap, use a moving-block bootstrap over forecast origins with block length at
least `ceil(H / stride)`. Report leave-one-dataset-out `Q` values as a sensitivity analysis.

`smean`, `snaive`, zero logits, and random backbones remain required controls, but the final
forecasting method does not need to beat each one on every dataset. They answer different
questions:

- VisionTS determines whether the proposed method is competitively better;
- pretrained versus random determines whether video pretraining contributes;
- pretrained versus zero logits determines whether the decoder contributes;
- `smean`/`snaive` reveal whether both visual methods are below a simple baseline.

## 4. Stage A — reproduce the VisionTS reference on all six datasets

This is the required positive control. Do not diagnose VideoMAE until this stage is valid.

### 4.1 Official implementation

Use the official VisionTS repository and record its resolved git commit. Keep the vendor
checkout isolated under `third_party/VisionTS`; do not edit its source.

Run the original VisionTS model with:

```text
architecture     = mae_base
checkpoint       = mae_visualize_vit_base.pth
finetune_type    = none
periodicity      = dataset period P
norm_const       = 0.4
align_const      = 0.4
interpolation    = bilinear
context/horizon  = the shared windows in Section 3
```

Explicitly assert and record:

```text
vision_model.norm_pix_loss == False
patch size == 16
image size == 224
```

Do not replace the visualization checkpoint with a representation-learning MAE checkpoint
trained using normalized pixel targets.

### 4.2 Required VisionTS controls

On the exact same images and masks, compute:

1. `visionts_pretrained`: official pretrained reconstruction;
2. `visionts_zeroed`: replace predicted masked patches with zeros before unpatchify;
3. `visionts_random`: same architecture with randomly initialized weights, seed 0;
4. `smean` and `snaive`: existing time-series baselines.

Save at least 16 input/mask/reconstruction examples for every dataset. Confirm visually that
the right masked block reconstructs raw grayscale values and that the inverse reshape follows
the correct temporal order.

### 4.3 Stage A gate

Proceed only when all of these hold:

1. predictions are finite and deterministic;
2. its aggregate equal-dataset ratio against `visionts_zeroed` is below 1;
3. its aggregate ratio against the random ImageMAE control is below 1;
4. no single dataset or invalid output dominates those aggregate ratios;
5. exported images confirm correct segmentation, mask location, and inverse mapping.

VisionTS does not need to beat `smean`, `snaive`, or the zeroed control on every dataset. This
stage validates a functioning reference implementation, not universal per-dataset dominance.

If this gate fails, first debug data windows, time ordering, normalization, checkpoint, and
inverse rendering. Do not interpret a failed reference implementation as evidence against
VideoMAE.

## 5. Stage B — validate VideoMAE target packing and decoder semantics

Use:

```text
checkpoint = MCG-NJU/videomae-base
model      = VideoMAEForPreTraining
frames     = 16
image      = 224 x 224
tubelet    = 2
patch      = 16 x 16
tokens     = 8 x 14 x 14 = 1568
```

Assert that `model.config.norm_pix_loss is True`.

### 5.1 Implement the exact native target transform

Add a utility that independently reproduces the target construction used by the installed
Transformers version:

1. undo ImageNet input normalization to `[0,1]` RGB frames;
2. reshape into `[tubelet, patch_h, patch_w, channel]` cubes;
3. compute mean and standard deviation independently for every cube and channel over its
   `2*16*16` pixel positions;
4. normalize each cube;
5. flatten in exactly the decoder-logit order;
6. select the entries indicated by `bool_masked_pos`.

Do not infer ordering from shape alone. Verify it numerically against `model.loss`.

### 5.2 Required packing tests

Before dataset runs, pass all of these tests:

1. patchify then unpatchify is an exact round trip for raw video tensors;
2. manual native-target MSE equals `VideoMAEForPreTraining(...).loss` within tolerance;
3. each masked logit is assigned to the correct tubelet, row, column, subframe, and RGB
   channel;
4. changing only masked input pixels changes the external diagnostic target but leaves model
   logits unchanged within numerical tolerance;
5. changing one visible patch changes logits, proving the invariance test is meaningful;
6. all mask variants have the exact intended number of masked tokens;
7. the same spatial mask is repeated over all eight tubelets for every tube-mask variant.

### 5.3 Do not perform this invalid ablation

Do **not** merely set `model.config.norm_pix_loss=False` at inference. That changes label
construction, not the meaning learned by the pretrained decoder weights. It cannot turn a
normalized-cube checkpoint into a raw-pixel checkpoint.

## 6. Stage C — historical reconstruction factorization

Run this stage only on ETTh2 and ETTm2, initially with at most 256 samples each. Everything
being reconstructed is inside observed history, so the true pixels and time-series values are
known without using a real forecast target.

### 6.1 Representations

Implement three isolated representations.

#### V0 — `dense_static`

Create the exact VisionTS-style dense period matrix:

```text
height = phase within period
width  = consecutive periods/time
pixel  = normalized value
```

Allocate the visible and hidden portions according to the mask configuration, resize each
portion separately, concatenate them on a patch boundary, and repeat the resulting image for
all 16 frames. This tests whether VideoMAE can process the successful image representation
without relying on synthetic motion.

#### V1 — `dense_rolling`

Use the same dense period matrix, but create a temporally ordered rolling video.

- frame 15 has the latest origin;
- each preceding frame moves its origin backward by one dataset period `P`;
- every frame contains an `L`-step visible context and an `H`-step hidden right block;
- obtain the required extra historical prefix explicitly (`L + 15P` total history);
- use one normalization computed from the latest frame's visible context for all frames;
- the same spatial right block is masked in every frame;
- use only frame 15 for the forecast-style inverse output.

It is acceptable and expected that a target hidden in an earlier frame later appears in a
visible part of a newer frame. That is normal temporal correspondence. The final frame's
hidden future must never appear anywhere in the input.

#### V2 — `line_rolling_legacy`

Reproduce the existing `pilot/run_reconstruct.py` rolling line representation and decode as
closely as possible. This is a regression control, not the recommended primary candidate.
First confirm that the new code reproduces the existing six JSON results within normal
floating-point tolerance on the old windows.

### 6.2 Mask configurations

Test these five masks:

| ID | Spatial mask | Effective ratio | Purpose |
|---|---|---:|---|
| `random_tube_90` | randomly mask 176 of 196 spatial locations, repeat over time | 89.8% | closest pretraining control |
| `random_tube_75` | randomly mask 147 of 196 locations, repeat over time | 75.0% | mask-ratio control |
| `right_13` | mask rightmost 13 of 14 columns | 92.9% | high-ratio future block |
| `right_10` | mask rightmost 10 of 14 columns | 71.4% | VisionTS-like alignment |
| `right_3` | mask rightmost 3 of 14 columns | 21.4% | current Stage F control |

For random masks, use mask seeds 0, 1, and 2 and average the result. The spatial mask must be
identical across the temporal axis but may differ by sample. Persist every mask or its exact
hash so the run is reproducible.

Random masks are evaluated only as reconstruction diagnostics. Do not decode them as a
contiguous future forecast.

### 6.3 Four decoder interpretations

For every representation/mask pair, report all applicable interpretations.

#### I0 — `native_norm`

Compare decoder logits directly with the exact normalized-cube target. This is the most
faithful measure of whether VideoMAE recognizes and reconstructs the rendered data.

```text
native_ratio = MSE(model_logits, normalized_target) /
               MSE(zero_logits, normalized_target)
native_gain  = 1 - native_ratio
```

Lower `native_ratio` is better; `<1` means the pretrained decoder adds signal over zeros.

#### I1 — `oracle_stats`

For diagnosis only, reconstruct raw pixels using the true mean and standard deviation of each
masked cube:

```text
raw_prediction = normalized_prediction * true_cube_sd + true_cube_mean
```

This is not a valid forecast because it uses hidden-target statistics. It is an upper bound
that answers whether useful geometry/value information exists in the normalized logits.

#### I2 — `causal_stats`

Estimate each masked cube's mean/std without hidden values. Implement at least:

1. nearest visible cube in the same row and tubelet;
2. context-wide visible pixel mean/std.

For a right block, propagate only from the visible left side. Select one rule on historical
ETTh2/ETTm2 and freeze it before real-future evaluation.

#### I3 — `line_geometry`

For `line_rolling_legacy` only, retain the current contour-position decoder. Report it beside
the native and oracle metrics; do not assume that patch-normalized logits are globally
comparable merely because local line position survives.

### 6.4 Metrics and artifacts

Save for each cell:

```text
native model MSE
native zero-logit MSE
native gain
raw-pixel MSE with oracle stats
raw-pixel MSE with each causal-stat rule
time-series MSE/MAE after inverse rendering where applicable
zero-logit time-series MSE/MAE
smean and snaive MSE/MAE on the same pseudo-target
patch-boundary seam error
wall time and peak GPU memory
```

Save contact sheets containing:

```text
ground truth full video/image
visible input plus mask
normalized decoder output
oracle-stat raw reconstruction
causal-stat raw reconstruction
```

### 6.5 Stage C interpretation gates

Use these gates as a decision tree, not as success criteria to optimize repeatedly.

```text
Does pretrained VideoMAE beat zero logits on native_norm under random_tube_90?
  no  -> packing bug, renderer is severely OOD, or checkpoint lacks reconstruction ability
  yes -> Does it also work for a contiguous right mask?
           no  -> mask topology/extrapolation boundary is the bottleneck
           yes -> Does oracle_stats recover the historical time series?
                    no  -> decoder logits do not preserve the needed numeric geometry
                    yes -> Do causal_stats recover it?
                             no  -> per-cube mean/std is the bottleneck
                             yes -> proceed to genuine future forecasting
```

Specific stop conditions:

- If every representation has `native_ratio >= 1` under `random_tube_90` on both datasets,
  stop strict forecasting runs and audit packing with a natural-video positive control.
- If random masks work but all right masks fail, do not spend six-dataset compute; report
  block-mask distribution shift.
- If oracle reconstruction works but causal reconstruction fails, move to Stage E rather
  than claiming the backbone cannot model the signal.

Select at most two representation/mask/decoder configurations using only historical-audit
results from ETTh2 and ETTm2.

## 7. Stage D — strict zero-shot genuine forecasting

Freeze every choice from Stage C before reading any real-future result.

### 7.1 Diagnostic forecast

First run the selected one or two configurations on ETTh2 and ETTm2 using true future values
only for evaluation.

Required controls:

1. pretrained VideoMAE reconstruction;
2. zero decoder logits through the identical inverse path;
3. randomly initialized VideoMAE with the same architecture and mask;
4. official `visionts_pretrained` from Stage A;
5. `smean` and `snaive`.

The encoder input must be invariant to arbitrary modifications of the future evaluation
buffer. Add and run this leakage test before metrics are accepted.

Treat this two-dataset run as an execution sanity check, not as the final success gate and not
as another opportunity to select hyperparameters. Continue when predictions are finite,
leakage tests pass, and reconstructions map back to the correct time indices. Do not require a
win on both datasets: VisionTS itself is not uniformly best on every dataset.

Pre-register one primary configuration from Stage C and run it on all six datasets regardless
of the ETTh2/ETTm2 ranking. A second Stage-C configuration may be reported as an ablation, but
must not replace the primary after genuine-future results are seen.

### 7.2 Held-out six-dataset run

Run exactly one frozen configuration on:

```text
ETTh1, ETTh2, ETTm2, electricity, traffic, solar
```

ETTh2 and ETTm2 are selection datasets. ETTh1, electricity, traffic, and solar are held out.
Do not tune a separate mask ratio, renderer, normalization, decoding temperature, context, or
stat-recovery rule per dataset.

Use two explicitly separate claims.

#### Primary forecasting claim — better than VisionTS

The basic competitive result is established when:

1. `Q_all < 1` on the complete six-dataset table;
2. all inputs and inverse transforms pass leakage and ordering tests.

This is the direct answer to “is the method better than VisionTS on this benchmark set?” A
stronger cross-dataset generalization result additionally requires `Q_heldout < 1` on ETTh1,
electricity, traffic, and solar.

No per-dataset sweep or universal win count is required. Report the six individual ratios,
win/tie/loss count, paired bootstrap interval, and leave-one-dataset-out sensitivity so a
single unusually easy dataset cannot be hidden by the aggregate.

For a stronger, publication-level transfer claim, require:

```text
gain_vs_V on held-out datasets >= 3%
upper endpoint of the paired 95% CI for Q_heldout < 1
```

#### Attribution claim — the gain uses pretrained video capability

Separately compute aggregate ratios against the identical zero-logit and random-VideoMAE
controls. A pretrained-video attribution is supported when both held-out aggregate ratios are
below 1, preferably by at least 3%. This does not have to hold on every individual dataset.

`smean` and `snaive` are mandatory sanity baselines but are not hard per-dataset gates. If the
method beats VisionTS while both visual methods lose to a simple baseline, report exactly that
limited conclusion rather than calling the method state of the art.

## 8. Stage E — optional label-free bridge if strict zero-shot fails

Run this stage only when Stage C shows that native or oracle VideoMAE reconstruction contains
signal but strict causal inversion/forecasting fails.

Train using pseudo-futures taken entirely from observed history. Freeze configuration choices
using ETTh2 and ETTm2, then evaluate transfer to:

```text
ETTh1, electricity, traffic, solar
```

Do not use any forecast target from those four held-out datasets for training or selection.

### E1 — patch-stat lens

Freeze all VideoMAE parameters. Train a small shared head that predicts each masked cube's
raw-pixel mean and log-standard-deviation from:

```text
decoder token or normalized logits
cube position embedding
nearest visible boundary statistics
```

Keep the lens below 500k trainable parameters. It may only undo the normalized-target
bottleneck; it must not consume raw future values or a supervised forecast label.

### E2 — raw-pixel decoder adaptation

Freeze the VideoMAE encoder. Adapt only its decoder and prediction projection on historical
pseudo-mask tasks using raw cube pixels as targets. Use the same masks and renderer selected
in Stage C.

Required paired control:

```text
pretrained frozen encoder + adapted decoder
random frozen encoder     + identically adapted decoder
```

Initialize the trainable decoder/lens identically, use identical pseudo-windows, and match
optimizer steps and parameter counts. This comparison is required to determine whether the
pretrained video backbone contributes anything beyond the learned bridge.

Evaluate label-free adaptation with the same equal-dataset ratios from Section 3.4. Keep two
claims separate:

1. **forecasting:** the complete adapted method has `Q_heldout < 1` against VisionTS;
2. **attribution:** its held-out aggregate MSE is at least 3% lower than the identically
   adapted random-encoder control;
3. **robustness:** the aggregate conclusions survive seeds 0, 1, and 2 and a paired bootstrap;
4. **sanity:** report `smean` and `snaive` without requiring a win on every dataset.

Per-dataset wins, ties, and losses remain visible in the table but are not hard gates.

If per-dataset self-supervised adaptation is additionally tested, report it separately as
transductive adaptation. Do not mix it with the cross-dataset result above.

## 9. Files to add

Prefer this isolated structure:

```text
pilot/video_visionts_renderers.py       dense static/rolling renderers and inverse transforms
pilot/videomae_patch_utils.py           exact cube packing, targets, masks, and unpatchify
pilot/run_visionts_reference.py         Stage A official VisionTS control
pilot/run_videomae_decoder_audit.py     Stages B/C historical reconstruction
pilot/run_video_visionts.py             Stage D strict forecast
pilot/run_video_visionts_adapt.py       optional Stage E label-free bridge
pilot/summarize_video_visionts.py       tables, gates, and decision report
pilot/run_video_visionts_sweep.sh       sequential launcher with resume/skip behavior
tests/test_video_visionts.py            leakage, packing, mask, and inverse tests
```

Reuse utilities from existing files where safe, but do not change existing result semantics
or overwrite old JSONs.

Write outputs under:

```text
pilot/results_field/video_visionts/
  manifests/
  visionts_reference/
  decoder_audit/
  historical_reconstruction/
  strict_forecast/
  label_free_adaptation/
  previews/
  logs/
  summary.json
  summary.md
```

## 10. Result schema

Every JSON must include:

```text
regime: ZS, HA, or LA
git commit and dirty-worktree flag
dataset path/hash and sample-manifest hash
dataset, period, context, horizon, channel count
checkpoint ID/revision and model config
VisionTS repository commit when applicable
torch and transformers versions
renderer and every renderer parameter
normalization source and constants
mask name, exact count, ratio, topology, seed, and hash
decoder interpretation and stat-recovery rule
pretrained/random/zeroed control identity
all native, raw-pixel, time-series, and baseline metrics
number of evaluated samples
wall-clock time and peak GPU memory
```

Never silently reuse a result produced with a different window manifest, library version,
mask seed, or checkpoint revision.

## 11. Required tests before GPU sweeps

At minimum, implement and run tests for:

1. all videos have shape `[B,16,3,224,224]` and finite values;
2. all inverse forecasts have shape `[B,H,1]` and finite values;
3. dense matrix segmentation and inverse segmentation preserve exact temporal order;
4. visible and hidden rendering are separated at a patch boundary;
5. future modifications cannot change strict-zero-shot inputs or predictions;
6. only visible context contributes to `mu/sd`;
7. affine transformations `a*x+b`, `a>0`, preserve normalized geometry;
8. manual VideoMAE native loss matches model loss;
9. cube packing/unpacking and masked-logit placement are numerically correct;
10. mask counts are 176, 147, 182, 140, and 42 spatial-temporal tube choices as
    appropriate after temporal repetition;
11. every tube mask repeats the same spatial map over all eight tubelets;
12. pretrained and random controls use byte-identical inputs and masks;
13. VisionTS uses the raw-pixel visualization checkpoint and `norm_pix_loss=False`;
14. the legacy line control reproduces the old reconstruction path;
15. filenames cannot collide across dataset/regime/renderer/mask/decoder/control/seed;
16. summary gates are tested using synthetic positive and negative JSON fixtures.

Clarification for mask-count test 10: the canonical spatial counts are `176/196`,
`147/196`, `182/196` (`13*14`), `140/196` (`10*14`), and `42/196` (`3*14`). The flattened
VideoMAE boolean mask contains eight repetitions of the selected spatial pattern.

Run one smoke test with at most eight windows per configuration before the 256-sample audit.
Smoke outputs must go to a separate directory and must never enter research tables.

## 12. Execution order and compute gates

Execute strictly in this order:

```text
0. unit tests and tiny smoke runs
1. Stage A VisionTS reference on all six datasets
2. Stage B packing/loss validation
3. Stage C 3 representations x 5 masks x 2 datasets, n <= 256
4. rerun at most two selected cells with n <= 2000
5. Stage D diagnostic future forecast on ETTh2 and ETTm2
6. Stage D six-dataset run for the pre-registered primary configuration
7. Stage E only if native/oracle signal justifies adaptation
```

Run inference jobs sequentially unless measured memory permits safe concurrency. Use an
evaluation batch no larger than 32 initially and lower it on OOM. A failed process must not
leave a result JSON that looks complete. The sweep script must skip only JSONs containing an
explicit `status: complete` field and a matching configuration hash.

Do not run multi-seed adaptation until seed 0 passes its gate. Strict deterministic right-mask
inference does not need repeated seeds; random-mask audit results use mask seeds 0, 1, and 2.

## 13. Required summary and final conclusions

The generated `summary.md` must contain:

1. VisionTS reference table for all six datasets;
2. native normalized-cube reconstruction table by renderer and mask;
3. oracle-versus-causal inversion table;
4. historical reconstruction versus genuine future table;
5. final strict-zero-shot table with pretrained, random, zeroed, VisionTS, `smean`, and
   `snaive`;
6. `Q_all`, `Q_heldout`, gain versus VisionTS, paired confidence intervals, win/tie/loss
   counts, and leave-one-dataset-out sensitivity;
7. optional label-free pretrained-versus-random table;
8. saved preview links and failed-run inventory;
9. a completed decision tree identifying the first failed gate.

Use one of these bounded conclusions:

```text
A. implementation/protocol invalid — VisionTS positive control did not reproduce
B. renderer OOD — VideoMAE fails even native random-tube historical reconstruction
C. mask mismatch — random tubes work, contiguous future blocks do not
D. decoder-domain mismatch — native/oracle work, causal de-normalization does not
E. forecasting mismatch — historical right-block reconstruction works, genuine future fails
F. competitive positive — VideoMAE method beats VisionTS in aggregate, attribution not proven
G. strict zero-shot video-transfer positive — beats VisionTS and pretrained beats random/zero
H. label-free bridge positive — adapted method beats VisionTS and pretrained beats random
```

Do not collapse B–E into “VideoMAE does not work.” They imply different next research moves.

## 14. Primary-source references

- [VisionTS paper](https://arxiv.org/html/2408.17253v4)
- [VisionTS official implementation](https://github.com/Keytoyze/VisionTS)
- [MAE official pretraining notes](https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md)
- [VideoMAE paper](https://arxiv.org/html/2203.12602)
- [VideoMAE official implementation](https://github.com/MCG-NJU/VideoMAE)
- [`MCG-NJU/videomae-base` config](https://huggingface.co/MCG-NJU/videomae-base/blob/main/config.json)
