# Video-VisionTS Matched-Mask Experiments

## 0. Purpose

This document specifies the next Video-VisionTS experiments for Claude Code to implement and
run. It supersedes the **interpretation** of the random-mask comparison in
`VIDEO_VISIONTS_RESULTS.md` and `VIDEO_VISIONTS_RECOVERY_RESULTS.md`; it does not overwrite or
delete their code or artifacts.

The immediate question is:

> When renderer geometry, samples, target composition, and score region are held fixed, does
> pretrained VideoMAE contain useful reconstruction signal for **future patches themselves**?

Only after answering that question should we spend compute on another strict six-dataset run.

The project objective remains:

```text
Q_all = geometric_mean_d(MSE_method,d / MSE_VisionTS,d) < 1
```

A 6/6 per-dataset win is not required.

## 1. Why another factorization is required

### 1.1 The old random/right comparison was not geometry-matched

The old audit contains:

```python
cols = masked_columns(mask)
video_input, video_full, ... = dense_static(ctx, fut, P, cols or 3)
```

For `random_tube_75` and `random_tube_90`, `masked_columns(mask)` is `None`, so the renderer
silently uses `cols=3`. The resulting canonical image has:

```text
11 context patch columns
 3 future patch columns
```

For `right_10`, the renderer has:

```text
 4 context patch columns
10 future patch columns
```

The mask comparison therefore changed all of the following simultaneously:

```text
mask topology
context/future pixel allocation
temporal resize scale
fraction of scored targets belonging to context
fraction of scored targets belonging to future
```

With seed 0 and 256 samples, a `random_tube_75` mask contains, on average:

```text
147.0 masked spatial patches total
115.5 masked context patches
 31.5 masked future patches
```

Approximately 78.5% of the reported random-mask native metric therefore measured interpolation
inside observed history. In contrast, all 140 patches scored by `right_10` were future patches.

The numerical results remain valid for their individual tasks, but `0.585 versus 0.929` does
not isolate mask topology and cannot support a general claim about future-only masks.

### 1.2 What the recovery study did establish

Keep these bounded conclusions:

```text
the old E1 target was invalid and is now fixed
L1/L2/L3 did not improve historical pseudo-forecast MSE on seed 0
direct column interleaving G1 failed
direct patch-jigsaw scattering G2 failed
the frozen R5 configuration did not beat VisionTS
pretrained weights still improve over zero/random controls by about 4%
```

Do not retain these broader claims without the matched experiment below:

```text
random-mask gain is purely a topology effect
random-mask gain is purely a locality effect
no future-only layout can preserve the gain
mask topology is closed as an actionable direction
```

## 2. Non-negotiable protocol

### 2.1 Dataset roles

Use only historical pseudo-futures for development:

```text
selection/audit: ETTh2, ETTm2
strict final:    ETTh1, ETTh2, ETTm2, electricity, traffic, solar
```

Reuse the chronological train/validation/audit partitions and purge rules from the recovery
study. Every candidate and control must use the same pair manifests.

### 2.2 Fixed primary geometry

Use the current project incumbent geometry for the entire primary factorization:

```text
renderer: dense_static
visible context columns: 4
future columns: 10
mask grid: 14 x 14
context length: 16 * P
horizon: existing dataset horizon
normalization: context-only mu/sd, norm_const=0.4
```

Use `cols=10` for both ETTh2 and ETTm2. Do not call `cols or 3`, do not derive renderer width
from the mask name, and do not rerender when the mask changes.

`right_9` may be reported later as a secondary geometry ablation, but it must not be mixed into
the primary matched comparison.

### 2.3 Render once, mask many times

For each historical batch:

```python
video_input, video_full, mu, sd = make_historical_pair(
    xb, P=P, H=H, L=L, cols=10
)
```

Compute and store hashes of `video_input` and `video_full`. Reuse those exact tensors for every
mask in the factorization. Assert that the hashes are identical across mask configurations.

The true pseudo-future may appear only in `video_full`, native targets, and metrics. Poisoning
it must not change `video_input`.

### 2.4 New output namespace

Do not overwrite previous experiments. Write to:

```text
pilot/results_field/video_visionts_matched_mask/
```

## 3. Stage M0 — exact target-region bookkeeping

Define canonical spatial patch labels for the fixed 4/10 geometry:

```text
context patch: canonical column 0..3
future patch:  canonical column 4..13
```

Expand these labels identically over all eight tubelets. For every masked token, save:

```text
tubelet index
spatial row and column
canonical context/future label
target cube std
whether target std <= 1e-5
```

All native metrics must be available for:

```text
all masked tokens
masked context tokens only
masked future tokens only
non-degenerate masked future tokens only
degenerate masked future tokens only
each canonical future column separately
```

Never infer future/context membership from the physical mask name. Use the canonical patch map.

### 3.1 Subset metric definition

For a subset `S` of masked tokens, compute from elementwise sums:

```text
MSE_PT(S)   = sum squared error of pretrained logits over S / number of scalar elements in S
MSE_ZERO(S) = sum squared native target over S / number of scalar elements in S
MSE_RAND(S) = identical error for a random-init backbone

native_ratio_zero(S) = MSE_PT(S) / MSE_ZERO(S)
native_ratio_rand(S) = MSE_PT(S) / MSE_RAND(S)
```

Do not average already-rounded batch MSE values. Store sums and counts in JSON.

Verify that the weighted combination of context-only and future-only sums exactly reproduces
the all-masked metric.

## 4. Stage M1 — matched mask factorization

Run the following masks on the exact same rendered tensors and manifests.

### M1.0 `right_future_100`

Mask all 140 future spatial patches and no context patches. This is the existing strict
right-block task and the primary reference.

### M1.1 `global_random_140`

Mask exactly 140 of the 196 spatial patches uniformly at random, repeated over tubelets. This
matches the total masked-patch count of `right_future_100` while changing topology.

Report context-only and future-only errors separately. The aggregate is secondary.

### M1.2 `global_random_147`

Mask 147 of 196 patches to reproduce the count of `random_tube_75`, but retain the fixed
4-context/10-future renderer. Again report context and future subsets separately.

### M1.3 `future_random_k`

Keep every context patch visible and randomly mask exactly `k` of the 140 future patches:

```text
k = 35, 70, 105, 140
```

These correspond to 25%, 50%, 75%, and 100% of future patches. Future patches that are not
selected remain blank but visible to the encoder. Label this input policy explicitly as:

```text
blank_visible_future
```

It is diagnostic rather than a final forecast method.

### M1.4 `future_block_c`

Mask the first `c` canonical future columns nearest the context boundary:

```text
c = 2, 5, 8, 10
```

Keep the remaining future columns blank-visible. This separates random topology from mask
size at approximately matched future counts.

### M1.5 `right_future_100`, cohort scoring

Run the full right mask once, then score its logits on the same token cohorts used by
`future_random_k` and `future_block_c`. This does not change the encoder mask. It answers
whether an apparent subset gain comes from easier target patches or from changing the mask.

### M1.6 Context-only interpolation control

Mask 42 random context patches, corresponding to 75% of the 56 context patches, while every
future patch stays blank-visible. Score only context targets.

This directly measures the task that dominated the old random-mask result.

### 4.1 Seeds and controls

Run mask seeds 0, 1, and 2 for every genuinely random mask. `future_random_140` is identical
to the deterministic full-future mask and needs only seed 0. Deterministic block masks also
need only seed 0.

For every cell run:

```text
pretrained VideoMAE
zero logits
random-init VideoMAE with identical architecture
```

The random model must see byte-identical inputs and masks.

For efficiency, load each backbone once per dataset, render each batch once, and loop over all
masks while the tensors remain in memory. Zero logits require no additional model forward.

### 4.2 Required M1 outputs

For every dataset/mask/seed/control, save:

```text
input and full-target hashes
mask hash and exact spatial indices
masked context/future counts
native sums, counts, MSEs, and ratios for every M0 subset
raw-pixel oracle-stat sums/counts for every subset
target std distribution over the complete audit manifest
per-future-column native ratios
wall time and peak memory
```

No time-series inversion is required for random masks at this stage. This stage exists to
locate the pretrained signal, not to claim forecasting quality.

## 5. Stage M2 — decision gates after factorization

Aggregate ETTh2 and ETTm2 using an equal-dataset geometric mean. Use these mutually exclusive
interpretations.

### Outcome A — old gain is historical interpolation

Choose A when:

```text
global-random context native_ratio_zero <= 0.80
global-random future native_ratio_zero >= 0.95
```

Interpretation:

> The old random-mask positive control primarily measured inpainting of observed history and
> did not demonstrate future-patch reconstruction ability.

Do not perform another patch-scatter sweep. Proceed to the carrier experiment in Stage M5 or
switch away from the pretrained reconstruction head.

### Outcome B — partial future masking contains signal

Choose B when at least one `future_random_k` or `future_block_c` cell satisfies all of:

```text
future native_ratio_zero <= 0.90
future MSE_PT / MSE_RAND <= 0.95
at least 10% relative improvement over the matched right-100 cohort
same direction on ETTh2 and ETTm2
same direction on at least two of three mask seeds where applicable
```

Proceed to iterative reconstruction in Stage M3.

### Outcome C — only normalized logits improve

Choose C when native future ratios pass Outcome B but raw oracle reconstruction does not beat
zero logits by at least 5%.

The logits contain normalized-shape signal but not useful numeric reconstruction. Run the
carrier experiment before iterative time-series evaluation.

### Outcome D — future signal exists even at 100%

Choose D when `right_future_100` itself has:

```text
future native_ratio_zero <= 0.90
future MSE_PT / MSE_RAND <= 0.95
raw oracle MSE_PT / MSE_ZERO <= 0.95
```

The primary bottleneck is then causal numeric inversion rather than masking. Do not rerun the
failed L1/L2/L3 lenses unchanged; test carrier decoding or a correctly paired raw decoder.

## 6. Stage M3 — causal iterative reconstruction

Run M3 only for Outcome B. Select the mask size/order using historical results only.

The key rule is:

```text
At every iteration, model input may contain context and earlier model/prior predictions,
but never a true pseudo-future or genuine future pixel.
```

### I0 — one-shot reference

The existing `right_future_100` forecast with nearest-visible causal statistics.

### I1 — boundary-to-right, one column per iteration

Initialize the entire future region blank. For iteration `j=0..9`:

1. mark all not-yet-accepted future columns as masked;
2. run VideoMAE;
3. take only the predicted cubes for the nearest remaining future column;
4. denormalize using statistics computed exclusively from currently visible context or earlier
   predicted cubes;
5. insert that predicted column into the model input and mark it visible;
6. continue until all ten future columns have been accepted.

Although the model predicts every remaining masked token, discard all but the accepted column
at each iteration.

### I2 — boundary-to-right, two columns per iteration

Identical to I1, but accept the nearest two remaining future columns per pass. This requires
five forward passes and measures the accuracy/compute trade-off.

### I3 — seasonal-prior masked refinement

Construct an initial future using `smean`, computed from context only. Render this synthetic
future into the canonical future region. It is an initialization, not a label.

Use two deterministic complementary future cohorts of 70 patches each:

```text
pass A: mask cohort A, keep current cohort B visible
pass B: mask cohort B, keep updated cohort A visible
```

Run one and two full A/B sweeps. For each masked patch, compute raw statistics from the current
visible synthetic/predicted image, never from the true target. Replace rather than average in
the primary version; a fixed 0.5 residual blend may be reported as one pre-registered ablation.

Required baselines:

```text
the unrefined smean initialization
the identical schedule with zero logits
the identical schedule with a random-init backbone
one-shot I0
VisionTS on the same historical pseudo-targets
```

### 6.1 Historical M3 metrics

On the frozen ETTh2/ETTm2 historical audit partitions, report:

```text
time-series MSE and MAE
MSE by forecast horizon quartile
MSE by accepted future column
error amplification from one iteration to the next
ratio to I0
ratio to the initialization baseline
ratio to paired zero/random schedules
number of VideoMAE forward passes
```

### 6.2 M3 promotion gate

Freeze at most one iterative schedule for strict evaluation. It must satisfy:

```text
geo MSE(schedule / I0) <= 0.97
geo MSE(pretrained schedule / random schedule) <= 0.97
neither ETTh2 nor ETTm2 more than 5% worse than I0
finite output and no monotonic error explosion across columns
```

If none passes, do not run M3 on six datasets.

## 7. Stage M4 — strict evaluation of a promoted schedule

Write a frozen configuration JSON before reading any genuine future. Use the existing Stage-D
test pair manifests and run:

```text
ETTh1, ETTh2, ETTm2, electricity, traffic, solar
```

Required methods:

```text
promoted pretrained iterative method
paired zero-logit iterative schedule
paired random-backbone iterative schedule
I0 one-shot right_10 incumbent
VisionTS
smean
snaive
```

Store predictions and per-origin errors for every arm. Report:

```text
six MSE/MAE values and ratios
Q_all and Q_heldout
win/tie/loss count
paired moving-block bootstrap intervals
leave-one-dataset-out Q
runtime and number of forward passes
```

Primary success is `Q_all < 1`; there is no all-dataset-win gate.

## 8. Stage M5 — patch-normalization-invariant carriers

Run M5 for Outcome A or C, or when M3 fails its historical gate.

The current intensity renderer is structurally vulnerable to `norm_pix_loss=True`: 19–25% of
future cube/channel targets in the recovery audit have standard deviation near `1e-6`, and
per-cube normalization can erase their entire numeric level. M5 encodes value in patch-local
geometry that survives subtraction of patch mean and division by patch std.

### 8.1 Coarse canonical scalar grid

Resample the normalized period matrix directly to a `14 x 14` scalar grid before rendering:

```text
4 context scalar columns
10 future scalar columns
14 phase rows
```

Each scalar grid cell becomes exactly one `16 x 16` spatial patch. Visible and future grids
must be resampled separately so no future value enters a context cell.

### C1 — soft edge-position carrier

For normalized scalar `z` clipped to `[-1,1]`, map its value to a vertical boundary position
inside the patch:

```python
h = 1 + 14 * (z + 1) / 2
tile[y, x] = sigmoid((h - y) / 0.75)
```

Repeat across `x` and RGB, then repeat the static image over 16 frames. Patch normalization
removes brightness level but retains edge position. Decode `z` from the predicted normalized
edge profile; do not estimate unknown raw cube mean/std.

### C2 — phase carrier

Encode `z` as the phase of a fixed low-frequency grating:

```python
phi = (pi / 2) * z
tile[y, x] = 0.5 + 0.45 * sin(2*pi*2*x/16 + phi)
```

Use a phase range without wrap ambiguity. Decode via correlation with fixed sine/cosine bases
on the native normalized logits. Again, no raw-pixel mean/std prediction is allowed.

### 8.2 Carrier checks

Before a model run, prove numerically that:

```text
native_targets(carrier(z1)) != native_targets(carrier(z2)) for separated values
decoder(carrier native target) recovers z with max error <= 0.02 on a dense z grid
target cube std never collapses below 1e-5 on the fixture grid
zero native logits decode to the declared neutral forecast z=0
```

### 8.3 Carrier historical audit

Run only ETTh2 and ETTm2 initially, with `right_future_100`, pretrained/random/zero controls,
and seeds 0/1/2 for sampling manifests.

Report native future ratios and directly decoded time-series errors. Promote at most one carrier
when:

```text
future native_ratio_zero <= 0.90
future MSE_PT / MSE_RAND <= 0.95
decoded TS MSE_PT / MSE_ZERO <= 0.95
geo decoded TS MSE / I0 <= 0.97
```

Only a promoted carrier may enter the six-dataset strict evaluation from Stage M4.

## 9. Optional Track H — historical-calibrated hybrid

This track answers a different, practical question:

> Can the available VideoMAE forecast and simple seasonal forecast be combined using only
> already-observed history so the complete method beats VisionTS?

It is **not strict cross-dataset zero-shot**. Report it separately as historical-calibrated or
transductive label-free forecasting.

For each dataset, use a purged historical validation block before the test range to select one
of the following fixed formulas:

```python
blend_alpha = (1-alpha) * smean + alpha * vmae
alpha in {0.00, 0.25, 0.50, 0.75, 1.00}

residual_beta = smean + beta * (vmae - zero_logits)
beta in {0.00, 0.50, 1.00, 1.50}
```

Use one scalar coefficient per dataset, shared across all channels and forecast origins. Select
by historical MSE only, freeze it, then evaluate the genuine future once.

The post-hoc oracle choice between the current VideoMAE and `smean` values would have
`Q_all ≈ 0.956`; this is motivation and an upper-bound diagnostic, **not a valid result**.
The experiment is successful only if the historical selector reproduces an aggregate win
without test-target selection.

Required attribution controls:

```text
freeze the coefficient selected for the pretrained method
replace pretrained forecast/residual with the paired random forecast
replace pretrained forecast/residual with the zero-logit forecast
report both fixed-coefficient controls
```

Primary Track-H criterion:

```text
Q_all < 1 versus VisionTS
```

Pretrained attribution additionally requires the complete selected method to beat both fixed-
coefficient random and zero controls in aggregate.

## 10. Do not rerun R6 unchanged

The previous raw-decoder pair did not isolate encoder pretraining:

```text
model.decoder was reinitialized and trained
encoder_to_decoder remained different between pretrained/random arms
mask_token remained different between pretrained/random arms
```

Its reported `L0` denominator also used zero logits rather than the actual pretrained L0
forecast. The large R6 failure remains informative, but the claimed 22.8% encoder attribution
is not a clean paired result.

If raw-decoder adaptation is revisited later:

1. make the frozen `videomae` encoder the only component allowed to differ between arms;
2. identically reinitialize and train `encoder_to_decoder`, `mask_token`, decoder blocks, and
   prediction head;
3. compare against the actual one-shot pretrained L0 as well as zero logits;
4. save actual executed steps, not only the requested maximum.

This is lower priority than M1 and must not run before the matched-mask factorization.

## 11. Files to implement

Prefer isolated versioned files:

```text
pilot/video_visionts_mask_factorization.py
    canonical region labels, matched masks, subset sums/counts

pilot/run_video_visionts_matched_audit.py
    M0/M1 historical factorization

pilot/video_visionts_iterative.py
    I1/I2/I3 causal rollout and refinement

pilot/run_video_visionts_iterative.py
    M3 historical and promoted M4 strict evaluation

pilot/video_visionts_carriers.py
    C1/C2 renderers and direct native-logit decoders

pilot/run_video_visionts_carrier.py
    M5 audit and optional strict evaluation

pilot/run_video_visionts_hybrid.py
    optional Track-H historical selection and frozen test

pilot/summarize_video_visionts_matched.py
    subset tables, gates, complete-config grouping, paired CIs

pilot/run_video_visionts_matched.sh
    resumable staged launcher with stop gates

tests/test_video_visionts_matched.py
    correctness, leakage, token-region, rollout, carrier, aggregation tests
```

Write results under:

```text
pilot/results_field/video_visionts_matched_mask/
  manifests/
  matched_audit/
  iterative_historical/
  carrier_historical/
  frozen_configs/
  strict_forecast/
  hybrid/
  previews/
  logs/
  summary.json
  summary.md
```

Write the final human report to:

```text
VIDEO_VISIONTS_MATCHED_MASK_RESULTS.md
```

## 12. Required tests

All relevant tests must pass before GPU sweeps:

1. all M1 masks reuse byte-identical `video_input` and `video_full`;
2. every primary M1 render uses exactly 4 context and 10 future columns;
3. future poisoning changes `video_full` but not `video_input`;
4. canonical context/future token labels have counts 56/140 per tubelet;
5. `right_future_100` masks exactly all 140 future and zero context patches;
6. `global_random_140/147` have exact counts for every sample and seed;
7. `future_random_k` masks exactly `k` future and zero context patches;
8. `future_block_c` masks exactly `14*c` future and zero context patches;
9. masked-token logits map back to correct full token identities;
10. context/future subset sums exactly recombine to the all-mask sum;
11. native subset MSE is invariant to batch partitioning;
12. target-distribution statistics accumulate over every audit batch, not only the first;
13. pretrained/random controls receive identical inputs and masks;
14. no M0/M1 metric reads an unmasked target as a model input;
15. right-100 cohort scoring reuses one forward pass and changes only the score selector;
16. I1 accepts every future column exactly once in left-to-right order;
17. I2 accepts every future column exactly once in five two-column steps;
18. iterative input at step `j` contains only context and predictions from steps `<j`;
19. poisoning any unaccepted pseudo-future never changes iterative predictions;
20. zero/random/pretrained iterative schedules use identical accepted-column logic;
21. seasonal-prior initialization depends only on context;
22. C1 native targets vary with edge position after patch normalization;
23. C2 native targets vary with phase after patch normalization;
24. perfect C1/C2 native targets decode the scalar fixture within tolerance;
25. carrier target std does not collapse on the fixture grid;
26. carrier inverse preserves period-major time order after resampling;
27. hybrid coefficient selection cannot access test origins or targets;
28. hybrid coefficients are frozen before genuine-future evaluation;
29. summarizer groups by full renderer/mask/schedule/carrier/config identity;
30. bootstrap is actually invoked and its seed/block rules are saved in JSON;
31. synthetic fixtures validate Q, held-out Q, LOO, and CI gates;
32. all tensors, saved predictions, and metrics are finite;
33. no new output path resolves into previous result directories.

The inverse-order test must actually execute render, patch permutation if applicable, native
packing/unpacking, carrier or dense inverse, and final time-series reconstruction. Testing only
`_matrix` followed by its direct reshape is insufficient.

## 13. Execution order

Run in this exact order:

```text
CPU correctness tests
  fail -> stop

GPU model-dependent packing/leakage tests
  fail -> stop

M1 ETTh2 + ETTm2 matched factorization, all controls
  -> choose Outcome A/B/C/D

Outcome B only: M3 iterative historical audit
  fail promotion -> stop iterative route

Outcome A/C or failed M3: M5 carrier historical audit
  fail promotion -> stop reconstruction-head preprocessing route

Only a promoted M3 or M5 candidate: M4 all-six strict evaluation

Optional Track H runs and reports separately from strict M4
```

Do not run all six datasets for every mask, schedule, or carrier. The only all-six jobs are one
historically frozen strict candidate and, if desired, the separately labelled Track-H method.

## 14. Required final conclusions

The final report must choose the narrowest supported statement:

```text
A. Old random-mask gain was dominated by masked historical context; it was not a future positive control.
B. Future patches contain pretrained signal under partial masks, and iterative causal filling recovers it.
C. Future normalized logits contain signal, but numeric inversion remains the bottleneck.
D. A patch-normalization-invariant carrier converts pretrained reconstruction into useful forecasts.
E. Matched future-only masks contain no useful pretrained signal, including at low future-mask fractions.
F. A strict method beats VisionTS in aggregate, with paired pretrained attribution.
G. A historical-calibrated hybrid beats VisionTS, but the result is transductive rather than strict zero-shot.
H. The result is inconclusive because a correctness, pairing, leakage, or robustness gate failed.
```

Do not infer a universal impossibility result from failure of G1/G2, one carrier family, or one
seed. Do not use per-dataset win count as a substitute for `Q_all`.
