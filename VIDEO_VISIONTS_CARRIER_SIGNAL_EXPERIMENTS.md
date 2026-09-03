# Video-VisionTS Carrier Signal Validation Experiments

## 0. Purpose

This document specifies the next experiments for Claude Code to implement and run. It follows
`VIDEO_VISIONTS_MATCHED_MASK_RESULTS.md` and does not overwrite any previous code, JSON, log,
summary, or conclusion.

The matched-mask result established one useful fact:

> The old random-mask gain was dominated by reconstruction of masked historical context.

The C2 result did **not** yet establish that VideoMAE predicted future values. A normalized
phase-carrier target is always a nonzero sinusoidal texture, while the reported native
baseline was zero logits. A pretrained decoder can beat that baseline by producing a generic
sinusoid without predicting the target-dependent phase. In addition, the current time-series
decoder uses only tubelet 0, and the end-to-end oracle loss from clipping and
`P -> 14 -> P` resampling was never measured.

This experiment must answer, in order:

1. Can the current C2 codec represent the target series accurately enough even with perfect
   native targets?
2. Does pretrained VideoMAE predict the **future-dependent phase**, or only the carrier
   texture/frequency?
3. Does aggregating all eight tubelets and using phase confidence convert any conditional
   signal into lower time-series error?

Do not run a six-dataset sweep here. The complete scope is ETTh2 and ETTm2. A larger run is
allowed only if the gates in Section 10 pass.

## 1. Claims and hypotheses

### 1.1 Claims that may be retained

The following claims are inputs to this study and do not need to be retested:

```text
fixed 4-context / 10-future geometry removes the old renderer confound
global-random context reconstruction is strong
global-random future reconstruction is approximately equal to zero logits
the current C2 end-to-end forecast is worse than zero and I0
```

### 1.2 Claims that remain unproven

Do not state any of the following before this experiment:

```text
C2 recovers future-conditioned representation signal
native_ratio_zero = 0.85 proves useful phase prediction
numeric inversion is the only remaining bottleneck
blank-visible future patches are the unique cause of the partial-mask result
the phase-carrier route is closed
```

### 1.3 Pre-registered hypotheses

`H_codec`: a perfect C2 native target survives clipping, grid resampling, native packing,
phase decoding, and time-series inversion with enough headroom to beat the existing I0
forecast.

`H_conditional`: with identical targets and masks, true context gives lower phase/z error
than shuffled or neutral context.

`H_pretrain`: pretrained VideoMAE gives lower phase/z error than a random-init VideoMAE
under the same input and decoder.

`H_forecast`: the fixed all-tubelet decoder beats zero and I0 on historical audit windows.

## 2. Fixed protocol

### 2.1 Datasets and splits

Use only:

```text
ETTh2
ETTm2
```

Reuse `build_partitions` from `pilot/run_video_visionts_recovery_audit.py` with the existing
chronological train/validation/audit boundaries, purge, stride 32, maximum 112 channels, and
the existing context/horizon definitions:

```text
context = 16 * P
horizon = 96
```

Roles are strict:

```text
validation: select any calibrated decoder coefficient
audit:      report all primary results and gates
genuine test / six datasets: forbidden in this document
```

Use manifest seed 0 for the primary run and save the exact `(origin, channel)` pair arrays,
not only their hashes. All arms must use byte-identical pair arrays. Do not average sampling,
shuffle, and random-initialization seeds under one generic field named `seed`.

Use separate fields:

```json
{
  "manifest_seed": 0,
  "shuffle_seed": 0,
  "random_init_seed": 0
}
```

Sampling-manifest seeds 1 and 2 are a gated robustness extension only.

### 2.2 Model, renderer, and mask

Fix:

```text
checkpoint: MCG-NJU/videomae-base
carrier: C2 phase carrier only
carrier frequency: 2
phase range: phi = (pi / 2) * z
normalization: context-only mu/sd
norm_const: 0.4
spatial grid: 14 x 14
context columns: 4
future columns: 10
video frames: 16
mask: right_future_100
```

Run pretrained and random-init models in `eval()` with no gradients. Do not tune the
backbone, decoder head, frequency, norm constant, geometry, or mask.

The current renderer repeats a static image over 16 frames. This experiment validates that
specific route; it must not be described as a test of learned video motion extrapolation.

### 2.3 Leakage rules

For every original sample:

```text
input may read the original context only
target may read the original pseudo-future
mu/sd must be computed from the original context only
no oracle target, audit target, or target-derived coefficient may affect a model input
```

Poisoning the original pseudo-future must leave all true-context, shuffled-context, and
neutral-context model inputs byte-identical.

### 2.4 New output namespace

Write only to:

```text
pilot/results_field/video_visionts_carrier_signal/
```

Never modify or delete:

```text
pilot/results_field/video_visionts/
pilot/results_field/video_visionts_recovery/
pilot/results_field/video_visionts_matched_mask/
```

## 3. Stage C0 — correctness and artifact repair

Implement the missing carrier tests before a GPU model run.

### 3.1 Full-range C2 numerical fixture

Use a dense scalar fixture covering the complete declared domain `[-1, 1]`, including both
endpoints. Execute the real path:

```text
z grid
-> C2 tiles
-> 16-frame RGB video
-> ImageNet normalization
-> native_targets
-> masked-token packing
-> phase decoder
```

Required assertions:

```text
max absolute z decode error <= 2e-5
zero logits decode to exactly z=0 within 1e-7
minimum raw cube std > 1e-5
token order is tubelet-major, then row-major spatial order
all 8 decoded tubelets recover the same fixture
```

C1 remains excluded. Do not spend GPU time tuning it.

### 3.2 Fix sample-specific degeneracy bookkeeping

The old matched runner uses the first sample's std mask for the entire batch. New code must
keep:

```text
std_ok shape = [B, n_masked]
```

Subset reductions must accept a sample-specific selector. Add a fixture in which the same
token is degenerate for sample 0 and non-degenerate for sample 1 and verify exact counts.

This repair applies to the new namespace only; do not rewrite old JSON.

### 3.3 Store replayable compact outputs

Do not save full `[B, 1120, 1536]` logits. For every masked token, save the sufficient
Fourier statistics after averaging the two frames inside its tubelet and the RGB channels:

```text
a = sum(tile * sine_basis)
b = sum(tile * cosine_basis)
r = sqrt(a^2 + b^2)
```

Save `a` and `b` as float32 arrays with shape:

```text
[n_sample, 8, 14, 10]
```

These arrays are small enough to replay every tubelet aggregation and confidence decoder
without another VideoMAE forward.

Also save:

```text
target z grid
mu and sd
true pseudo-future
smean and snaive forecasts
per-sample native error sums and element counts
exact pair array and exact mask indices
input/full-target hash for every batch
shuffle permutation
configuration hash
git commit and dirty flag
package versions and checkpoint identity
```

Do not round stored sums or per-sample errors. Round only Markdown display values.

## 4. Stage C1 — codec oracle and information ceiling

Run this stage before any new **carrier** VideoMAE forward. It measures whether the current
representation itself has enough time-series fidelity. For the I0 comparison only, an old
aggregate may be reused after regenerating the exact pair array and asserting its hash; if no
exact match exists, compute I0 separately before evaluating the gate.

### 4.1 Oracle paths

For each validation and audit sample compute four paths.

The `--stage codec` command must compute both splits in one deterministic run: choose B2
using validation, freeze it, and only then score audit. It must not choose any constant,
threshold, or decoder from audit.

#### O0 — scalar identity

```text
future
-> context normalization
-> inverse context normalization
```

This must reproduce the future to numerical precision and detects normalization/order errors.

#### O1 — resampling only

Build the current `14 x 10` future scalar grid without clipping, then run
`z_to_series`. This isolates loss from `P -> 14 -> P` and horizontal resampling.

#### O2 — current grid oracle

Build the current scalar grid with `clamp(-1, 1)`, then run `z_to_series`. This measures
the combined loss from resampling and clipping.

#### O3 — native-target round trip

```text
current full C2 carrier video
-> native_targets
-> select all right_future_100 targets
-> compute a/b for every tubelet
-> all-tubelet phase aggregation
-> z_to_series
```

O3 must agree with O2. It detects packing, order, phase, or tubelet bugs that O2 bypasses.

### 4.2 Required codec diagnostics

Report per dataset:

```text
clip fraction before clamp
z p01 / p05 / median / p95 / p99 before clamp
O0/O1/O2/O3 MSE and MAE
O1/O2/O3 ratio to context-mean forecast
O1/O2/O3 ratio to smean, snaive, and I0
max absolute difference between O2 and O3 predictions
per-horizon and per-phase-row oracle MSE
```

`I0` must be recomputed on the exact same pair manifest or imported only after asserting an
exact pair-array hash match.

### 4.3 Codec gate

`G_codec` passes only when all are true:

```text
O0 max absolute reconstruction error <= 1e-5
O2/O3 prediction max absolute difference <= 2e-4
O3 MSE / context-mean MSE <= 0.50 on each dataset
equal-dataset geo MSE(O3 / I0) <= 0.90
neither dataset has MSE(O3 / I0) > 1.00
```

If this gate fails, stop. The `14 x 10` carrier lacks sufficient oracle headroom; do not
interpret later model errors as a VideoMAE failure and do not run another phase frequency
sweep.

## 5. Stage C2 — carrier-aware baselines

The native zero-logit baseline is retained as a sanity check, but it is not a sufficient
baseline for a structured carrier.

### 5.1 Baseline definitions

Compute every baseline against the same native target and target z grid.

#### B0 — zero logits

```text
prediction = all-zero normalized cube
```

This reproduces the old native ratio. Label it `zero_logits_texture_absent`.

#### B1 — neutral carrier

Construct the exact normalized native C2 target for `z=0` at every future patch and
tubelet. This predicts the correct carrier frequency and amplitude but no future-dependent
phase.

Label it:

```text
neutral_phase_texture_present
```

This is the primary carrier-aware native baseline.

#### B2 — validation constant phase

On validation only, select one constant `z_const` from:

```text
[-1.00, -0.95, ..., 0.95, 1.00]
```

that minimizes validation z-grid MSE. Freeze it before audit. This is a
historical-calibrated diagnostic baseline, not strict zero-shot. Save the complete validation
curve and the selected value.

#### B3 — seasonal-mean carrier

Compute `smean` from context only, normalize it using the original context mu/sd, pass it
through the same scalar-grid and C2 carrier construction, and use its native targets as the
prediction. This tests whether a simple time-series prior explains the carrier-space score.

#### B4 — seasonal-naive carrier

Repeat the last observed period, then use the same normalization, grid, and carrier target
construction. This is `snaive` represented in carrier space.

### 5.2 Required baseline metrics

For B0–B4 report:

```text
native cube MSE
native ratio to B0
native ratio to B1
z-grid MSE and MAE
phase cosine similarity: mean cos(phi_pred - phi_target)
decoded time-series MSE and MAE
```

The report must not call a model result positive merely because it beats B0. It must beat B1
and the appropriate time-series prior.

## 6. Stage C3 — conditionality controls

This stage determines whether any pretrained advantage depends on the correct context-target
pairing.

### 6.1 Inputs

Use three input variants with the original target held fixed.

#### X_true — correct context

The existing leakage-safe C2 input built from the sample's true context.

#### X_shuffle — wrong but in-distribution context

Create a deterministic derangement of contexts within each dataset and channel. Prefer a
cyclic shift of origins within channel so there are no fixed points. Use the donor sample's
already-normalized context scalar grid as visible carrier patches, while retaining:

```text
the recipient's original target
the recipient's target z grid
the recipient's mu/sd for final target-space reporting
the identical future mask
```

Do not shuffle individual pixels or patches. That would create a new low-level distribution
shift instead of testing context-target pairing.

#### X_neutral — no context information

Set every visible context scalar cell to `z=0`, render the valid C2 neutral texture, and keep
the original target and future mask.

### 6.2 Backbones

The primary pilot matrix is:

| backbone | X_true | X_shuffle | X_neutral |
|---|---:|---:|---:|
| pretrained | run | run | run |
| random-init seed 0 | run | no | no |

All four forwards must use the same audit samples, target tensors, and mask. Load each model
once per dataset. Render a batch once per input arm and reuse it for every applicable
backbone.

If the primary conditionality gate passes, add:

```text
shuffle seeds 1 and 2 for pretrained
random-init seeds 1 and 2 on X_true
manifest seeds 1 and 2 using frozen decisions
```

Do not run these robustness jobs before the primary gate.

### 6.3 Post-hoc target-shuffle control

Without another forward, cyclically permute target z grids within channel and rescore the
`X_true` predictions. This measures how much performance can be explained by the
unconditional spatial/phase distribution.

Call this `Y_shuffle`. Store the exact permutation separately from the input shuffle.

### 6.4 Fair pairing

For every comparison, accumulate paired per-sample sums:

```text
PT true versus PT shuffled-context
PT true versus PT neutral-context
PT true versus PT target-shuffled
PT true versus random-init true
PT true versus B1 neutral carrier
```

Never compare arms built from different manifests.

## 7. Stage C4 — all-tubelet phase decoding

The target video is static, so all eight tubelets encode the same scalar grid. The current
implementation discards seven tubelets. Replace that only in the new runner.

### 7.1 Per-tubelet coefficients

For tubelet `t`, spatial cell `j`, and its decoded tile `L[t,j]`, compute:

```text
a[t,j] = sum(L[t,j] * sine_basis)
b[t,j] = sum(L[t,j] * cosine_basis)
c[t,j] = a[t,j] + i*b[t,j]
r[t,j] = abs(c[t,j])
```

The two frames inside a tubelet and RGB are averaged before the spatial Fourier
coefficients, matching the static target construction.

### 7.2 Fixed decoders

Report all of these, but pre-register D2 as the primary strict decoder.

#### D0 — first tubelet

Reproduce the old implementation exactly:

```text
phi = atan2(b[0], a[0])
```

This is a regression control only.

#### D1 — raw complex mean

```text
c_sum = sum_t c[t]
phi_raw = atan2(imag(c_sum), real(c_sum))
z_raw = clip(2 * phi_raw / pi, -1, 1)
```

This is equivalent to averaging tubelet logits before phase correlation.

#### D2 — coherence-shrunk complex mean

Define cross-tubelet coherence:

```text
rho = abs(sum_t c[t]) / (sum_t abs(c[t]) + 1e-8)
z_D2 = rho * z_raw
```

`rho` lies in `[0,1]`. Low-agreement predictions shrink toward the declared neutral
forecast without fitted parameters. D2 is the **primary strict decoder**.

#### D3 — unit-phasor circular mean

```text
c_unit[t] = c[t] / (abs(c[t]) + 1e-8)
phi = angle(sum_t c_unit[t])
z_D3 = clip(2 * phi / pi, -1, 1)
```

D3 is diagnostic. It tests whether a high-amplitude outlier dominates D1.

### 7.3 Phase and confidence metrics

For each decoder/input/backbone report:

```text
z MSE and MAE
Pearson correlation(predicted z, target z)
Spearman correlation
mean cos(phi_pred - phi_target)
mean and quantiles of rho
error by rho quintile
error by future column
error by tubelet for D0-style per-tubelet decoding
decoded TS MSE and MAE
```

Correlations must be computed globally and per sample. Constant predictions must return a
declared zero correlation rather than NaN.

### 7.4 Optional historical calibration

Run this stage only if `G_conditional` passes but the fixed forecast gate fails.

Using validation only, select:

```text
prior in {neutral, smean}
alpha in {0.00, 0.10, ..., 1.00}
```

with:

```text
z_calibrated = z_prior + alpha * rho * (z_raw - z_prior)
```

Select by validation time-series MSE, freeze `prior` and `alpha`, then evaluate once on
audit. Save every validation candidate, not only the winner.

This result must be labeled:

```text
historical-calibrated / transductive
```

It cannot be used to claim strict cross-dataset zero-shot performance. Always report the
`alpha=0` prior beside it so the incremental contribution from VideoMAE is visible.

## 8. Stage C5 — time-series references

Every reference must use the exact same historical audit pair array and pseudo-future.

Required references:

```text
context mean: z=0 after context normalization
smean: mean phase profile over all complete context periods
snaive: repeat the last context period
I0: dense_static + pretrained VideoMAE + nearest-visible-physical inversion
```

Do not copy a metric from a previous JSON unless its complete pair array and manifest hash
match exactly. Per-window predictions/errors are required for paired intervals; if an old
artifact contains only an aggregate, recompute it.

### 8.1 Gated VisionTS reference

Do not load or run VisionTS during C0–C4. Run the official VisionTS implementation on the same
historical validation/audit windows only if D2 passes `G_forecast`.

Use the existing official path and unchanged configuration:

```text
arch: mae_base
finetune_type: none
norm_const: 0.4
align_const: 0.4
interpolation: bilinear
periodicity: dataset P
```

The existing `pilot/results_field/video_visionts/visionts_reference/*.json` files use genuine
test windows and therefore are not numerically interchangeable with this historical audit.

This two-dataset historical VisionTS comparison is diagnostic. It does not replace a final
six-dataset genuine-future evaluation.

## 9. Metrics, aggregation, and uncertainty

### 9.1 Primary units

Store elementwise sums and counts, then derive MSE. Do not average rounded batch MSEs.

For time-series metrics, save both:

```text
per-pair error: one origin/channel pair
per-origin error: mean across available channels at that origin
```

The paired bootstrap unit is the forecast origin, not an individual scalar timestep.

### 9.2 Dataset aggregation

For method `m` and reference `r`:

```text
Q_2(m/r) = exp(mean_d(log(MSE_m,d / MSE_r,d)))
d in {ETTh2, ETTm2}
```

Give ETTh2 and ETTm2 equal weight. Never pool their scalar errors before taking the ratio.
Report MSE and MAE, but use MSE for gates.

### 9.3 Paired moving-block bootstrap

Use 5,000 paired replicates and bootstrap seed 20260902. Sort unique origins
chronologically. Use circular or moving blocks with:

```text
block length in origins = ceil((context + horizon) / stride)
```

This gives 15 origins for ETTh2 and 51 for ETTm2 under the fixed protocol. If a dataset has
fewer unique origins than twice the block length, report that limitation and also provide a
paired origin-level permutation interval; do not silently switch the sampling unit.

Save:

```text
bootstrap seed and block length
all gate-relevant 95% confidence intervals
number of unique origins and pairs
paired improvement probability
```

The summarizer must actually call the bootstrap function. A test for the function's existence
does not count.

## 10. Gates and decisions

Apply gates sequentially. Later stages must not run after an earlier terminal failure.

### 10.1 G_codec

Use the exact gate in Section 4.3.

Failure means:

> The current coarse C2 codec has insufficient oracle fidelity.

Stop the C2 model audit. The next representation would need more scalar capacity, likely
using the temporal axis instead of repeating one static image.

### 10.2 G_conditional

Evaluate with pretrained X_true and the fixed D2 decoder. Pass only if all are true:

```text
Q_2(z-MSE PT_true / B1_neutral) <= 0.95
Q_2(z-MSE PT_true / PT_X_shuffle) <= 0.95
Q_2(z-MSE PT_true / PT_Y_shuffle) <= 0.95
neither dataset is worse than its shuffled-context control
Pearson z correlation > 0 on both datasets
paired 95% CI for true-vs-shuffled z-MSE improvement excludes zero in at least one dataset
and the pooled equal-dataset direction is positive
```

Failure means:

> The old native advantage is adequately explained by a carrier/decoder prior; useful
> future-conditioned phase signal has not been demonstrated.

Stop. Do not run calibration, VisionTS, extra carrier frequencies, or more datasets.

### 10.3 G_pretrain

Pass only if:

```text
Q_2(z-MSE pretrained_true / random-init_true) <= 0.95
pretrained is better on both ETTh2 and ETTm2
Q_2(native MSE pretrained_true / B1_neutral) <= 0.95
```

Failure with `G_conditional` passing means context contains signal but pretrained weights
are not responsible for extracting it.

### 10.4 G_forecast

Use only the fixed, parameter-free D2 decoder. Pass only if:

```text
Q_2(TS MSE D2 / context-mean) <= 0.95
Q_2(TS MSE D2 / I0) <= 0.97
neither dataset is more than 5% worse than I0
```

If this fails while both `G_conditional` and `G_pretrain` pass, run the one optional
validation-calibration stage in Section 7.4. Do not open a general decoder sweep.

### 10.5 G_VisionTS

Only after `G_forecast`, run the matched historical VisionTS reference. Promotion to a
larger genuine-future experiment requires:

```text
Q_2(TS MSE D2 / VisionTS) < 1.00
pretrained D2 beats VisionTS on at least one of the two datasets
Q_2(TS MSE pretrained D2 / random-init D2) <= 0.95
```

A win on every dataset is not required.

### 10.6 Decision table

| outcome | interpretation | next action |
|---|---|---|
| G_codec fails | representation loses too much information | stop C2; design higher-capacity/temporal codec |
| G_codec passes, G_conditional fails | generic carrier texture prior | close current C2 route |
| conditional passes, pretrain fails | signal is not attributable to pretrained weights | close zero-shot-pretraining claim |
| conditional + pretrain pass, forecast fails | phase signal exists but fixed decoder is weak | run exactly one validation-calibrated shrinkage stage |
| fixed forecast passes, VisionTS fails | useful forecast, not yet better than VisionTS | analyze paired failures before scaling |
| all gates pass | credible pilot positive | write a separate 5–6 dataset genuine-future plan |
| only calibrated route passes | practical historical-calibrated result | report separately as transductive |

## 11. Implementation contract

### 11.1 Files

Claude Code should create:

```text
pilot/video_visionts_carrier_signal.py
    codec oracles, carrier-aware baselines, Fourier coefficients,
    tubelet aggregation, coherence, and compact artifact utilities

pilot/run_video_visionts_carrier_signal.py
    C0-C4 dataset/split runner

pilot/run_video_visionts_carrier_signal_visionts.py
    gated matched historical VisionTS reference

pilot/summarize_video_visionts_carrier_signal.py
    aggregation, bootstrap, gates, and Markdown summary

pilot/run_video_visionts_carrier_signal.sh
    staged launcher that stops at failed gates

tests/test_video_visionts_carrier_signal.py
    all correctness and aggregation tests in Section 12
```

Existing carrier/patch utilities may be imported or extended, but old result JSON must remain
immutable.

### 11.2 Result tree

```text
pilot/results_field/video_visionts_carrier_signal/
  manifests/
    ETTh2_seed0_pairs.npz
    ETTm2_seed0_pairs.npz
  codec/
    codec_ETTh2_s0.json
    codec_ETTm2_s0.json
  signal/
    C3_ETTh2_s0.json
    C3_ETTh2_s0_arrays.npz
    C3_ETTm2_s0.json
    C3_ETTm2_s0_arrays.npz
  calibration/
  visionts/
  gates/
    codec_gate.json
    conditional_gate.json
    forecast_gate.json
  logs/
  summary.md
```

Use a full configuration identity and hash in each filename or JSON. The summarizer must group
by:

```text
dataset
split
manifest_seed
checkpoint
carrier and frequency
norm_const
geometry
mask
input arm
backbone arm
random_init_seed
shuffle_seed
decoder
calibration identity
git commit
```

It must raise an error on duplicate identities instead of silently overwriting by dataset.

### 11.3 Minimum JSON schema

```json
{
  "status": "complete",
  "stage": "C3",
  "dataset": "ETTh2",
  "split": "audit",
  "regime": "historical_audit",
  "config_id": "...",
  "manifest": {
    "manifest_seed": 0,
    "pairs_sha1": "...",
    "pairs_file": "...",
    "n_pairs": 0,
    "n_origins": 0
  },
  "carrier": {
    "name": "C2",
    "frequency": 2,
    "norm_const": 0.4,
    "context_cols": 4,
    "future_cols": 10
  },
  "controls": {},
  "codec_oracles": {},
  "native_metrics": {},
  "phase_metrics": {},
  "ts_metrics": {},
  "per_origin_errors": {},
  "bootstrap": {},
  "artifacts": {},
  "provenance": {}
}
```

Do not emit JSON `NaN` or `Infinity`. Use `null` plus a reason field for an undefined
metric.

## 12. Mandatory tests

All tests must run before the first GPU job. No carrier test may be marked SKIP after the
carrier runner exists.

1. C2 normalized native targets differ for `z=-0.5` and `z=+0.5`.
2. Full `[-1,1]` C2 fixture decodes within `2e-5`, including endpoints.
3. Zero native logits decode to neutral `z=0`.
4. C2 raw cube std stays above `1e-5`.
5. Future poisoning changes the target but not any model input arm.
6. Visible context construction never reads the recipient pseudo-future.
7. Native masked-token order maps exactly to `[tubelet,row,future_column]`.
8. Every future spatial patch appears exactly once per tubelet.
9. All eight native-oracle tubelets recover the same target z grid.
10. O0 preserves period-major time order and values.
11. O2 and O3 agree within the declared tolerance for both `P=24` and `P=96`.
12. A ramp fixture detects phase-row or period-column transposition.
13. Sample-specific degeneracy selectors handle different samples independently.
14. D0 exactly reproduces the old first-tubelet implementation.
15. D1 equals Fourier decoding after explicitly averaging logits across tubelets.
16. D2 coherence is 1 for identical phasors and near 0 for cancelling phasors.
17. D2 returns finite neutral output when every coefficient has zero amplitude.
18. Input shuffle is a derangement, stays within channel, and leaves targets unchanged.
19. Target shuffle is distinct from input shuffle and requires no new forward.
20. B1 is the exact native target for a neutral C2 carrier, not zero logits.
21. B3/B4 depend only on context.
22. Validation-selected alpha is frozen before audit is opened.
23. All compared arms have identical pair and mask hashes.
24. Elementwise and batch-partitioned sums produce identical MSE.
25. Per-origin aggregation correctly combines available channels.
26. The bootstrap function is invoked, uses the stored seed/block length, and writes intervals.
27. Synthetic positive, null, and negative fixtures trigger the correct sequential gates.
28. The summarizer rejects duplicate full identities.
29. The summarizer rejects mixed pair hashes in a paired comparison.
30. Saved JSON and NPZ arrays contain only finite values.
31. New output paths cannot resolve into an old result directory.

The test command must exit nonzero on any failure:

```bash
python tests/test_video_visionts_carrier_signal.py
```

## 13. Execution order and compute budget

### Phase A — CPU correctness and codec

```bash
python tests/test_video_visionts_carrier_signal.py

python pilot/run_video_visionts_carrier_signal.py \
  --stage codec --dataset ETTh2 --manifest-seed 0

python pilot/run_video_visionts_carrier_signal.py \
  --stage codec --dataset ETTm2 --manifest-seed 0

python pilot/summarize_video_visionts_carrier_signal.py --stage codec
```

Stop automatically if `G_codec` fails.

### Phase B — primary audit signal

```bash
python pilot/run_video_visionts_carrier_signal.py \
  --stage signal --split audit --dataset ETTh2 \
  --manifest-seed 0 --shuffle-seed 0 --random-init-seed 0

python pilot/run_video_visionts_carrier_signal.py \
  --stage signal --split audit --dataset ETTm2 \
  --manifest-seed 0 --shuffle-seed 0 --random-init-seed 0

python pilot/summarize_video_visionts_carrier_signal.py --stage signal
```

Each dataset needs four carrier forwards per batch:

```text
pretrained X_true
pretrained X_shuffle
pretrained X_neutral
random-init X_true
```

I0 is one additional forward only if no exact matched replay artifact exists. Every decoder
variant must reuse saved a/b arrays and must not trigger a new model forward.

Stop automatically if `G_conditional` or `G_pretrain` fails.

### Phase C — optional calibration

Run only if conditional/pretrain gates pass and the fixed D2 forecast gate fails:

```bash
python pilot/run_video_visionts_carrier_signal.py \
  --stage calibration --split validation --dataset ETTh2 --manifest-seed 0

python pilot/run_video_visionts_carrier_signal.py \
  --stage calibration --split validation --dataset ETTm2 --manifest-seed 0

python pilot/summarize_video_visionts_carrier_signal.py --stage calibration
```

Apply the frozen choices to the saved audit coefficients; do not rerun audit forwards merely
to decode them.

### Phase D — gated VisionTS

Run only if the fixed D2 forecast gate passes:

```bash
python pilot/run_video_visionts_carrier_signal_visionts.py \
  --dataset ETTh2 --manifest-seed 0

python pilot/run_video_visionts_carrier_signal_visionts.py \
  --dataset ETTm2 --manifest-seed 0

python pilot/summarize_video_visionts_carrier_signal.py --stage final
```

### Phase E — gated robustness

Only after a positive Phase D:

```text
two additional shuffled-context permutations
two additional random-init backbones
two additional audit sampling manifests with all choices frozen
```

Do not run other carriers, frequencies, mask geometries, or datasets in this experiment.

## 14. Required summary

Generate:

```text
pilot/results_field/video_visionts_carrier_signal/summary.md
```

The summary must contain, at minimum:

### Table A — codec ceiling

| dataset | clip frac | O1 MSE | O2 MSE | O3 MSE | O3/zero | O3/I0 | O2–O3 max diff |
|---|---:|---:|---:|---:|---:|---:|---:|

### Table B — carrier-aware native baselines

| dataset | arm | native MSE | / zero logits | / neutral carrier | z MSE | phase cosine |
|---|---|---:|---:|---:|---:|---:|

### Table C — conditionality

| dataset | decoder | PT true z MSE | PT shuffled | PT neutral | Y shuffled | random true | z corr |
|---|---|---:|---:|---:|---:|---:|---:|

### Table D — tubelet decoding

| dataset | decoder | z MSE | rho median | TS MSE | / zero | / smean | / I0 |
|---|---|---:|---:|---:|---:|---:|---:|

### Table E — gates

| gate | value | threshold | status |
|---|---:|---:|---|

Every table must state:

```text
split
manifest seed/hash
whether a value is strict, diagnostic oracle, or historical-calibrated
number of pairs and unique origins
```

Show all paired intervals near the metric they support. Do not report only an aggregate gate
when ETTh2 and ETTm2 move in opposite directions.

After the run, write a separate human-readable:

```text
VIDEO_VISIONTS_CARRIER_SIGNAL_RESULTS.md
```

That report must be generated from saved artifacts and must include failed gates and stopped
stages, not only successful arms.

## 15. Interpretation rules

Use the narrowest conclusion supported by the first failed gate.

Allowed examples:

```text
the coarse carrier lacks oracle capacity
pretrained VideoMAE reconstructs carrier texture but not target-dependent phase
correct context improves phase prediction relative to paired shuffled controls
all-tubelet coherence reduces decoder noise
a historical-calibrated residual helps, but strict zero-shot still fails
```

Forbidden examples:

```text
native MSE below zero alone proves future signal
PT/RAND alone proves forecasting capability
an oracle path is a deployable forecast
a validation-selected alpha is strict zero-shot
two historical datasets prove a universal result
failure of static C2 proves video backbones cannot transfer to time series
```

The final objective remains to beat VisionTS in equal-dataset aggregate, not to win every
dataset. This document only determines whether the current C2 carrier deserves that larger
evaluation.
