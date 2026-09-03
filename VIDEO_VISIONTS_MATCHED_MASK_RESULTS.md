# Video-VisionTS Matched-Mask — Results

Execution of `VIDEO_VISIONTS_MATCHED_MASK_EXPERIMENTS.md`. NU CS cluster (nyx, 4x A40), free.

## 0. Conclusion

**A — the old random-mask gain was dominated by masked historical context; it was not a
future positive control.**

With **C** as the Stage-M5 finding: a patch-normalization-invariant carrier *does* recover
pretrained signal on future patches in normalized-target space, but numeric inversion remains
the bottleneck.

## 1. The confound, confirmed

The old audit derived the RENDERER width from the mask name:

```python
cols = masked_columns(mask)              # None for random_tube_*
dense_static(ctx, fut, P, cols or 3)     # -> silently 11 context / 3 future columns
```

| | random_tube_75 | right_10 |
|---|---|---|
| canonical geometry | **11 ctx / 3 fut** | **4 ctx / 10 fut** |
| masked patches | 147 | 140 |
| share of scored targets that are FUTURE | **21.4%** | 100% |

So `0.585 versus 0.929` changed mask topology, context/future pixel allocation, temporal
resize scale and score-region composition all at once. **78.6% of the random-mask metric was
interpolation inside observed history.** The numbers were valid for their own tasks; the
*interpretation* was not.

Withdrawn from `VIDEO_VISIONTS_RECOVERY_RESULTS.md`: *"random-mask gain is a locality
effect"*, *"no future-only layout can preserve the gain"*, *"mask topology is closed as an
actionable direction"*, and with them conclusion **F** of that report.

## 2. M1 — matched factorization

Geometry FIXED at 4 context / 10 future columns for every mask. Each batch is rendered once and
the identical `video_input`/`video_full` tensors are reused for all 24 mask configurations
(hash-asserted). Region membership always read from the canonical patch map. Subset sums
accumulated in float64; context+future recombines to the all-mask sum to 2e-16.

ETTh2 + ETTm2 equal-dataset geometric mean, `native_ratio_zero` (lower is better, <1 beats
zero logits):

| mask | ctx masked | fut masked | **context ratio** | **future ratio** | future PT/RAND |
|---|---:|---:|---:|---:|---:|
| right_future_100 | 0 | 140 | — | 0.9654 | 0.8109 |
| global_random_140 | 41 | 99 | **0.3945** | 1.0144 | 0.8518 |
| global_random_147 | 42 | 105 | **0.5194** | 1.0047 | 0.8441 |
| context_random_42 | 42 | 0 | **0.4991** | — | nan |
| future_random_35 | 0 | 35 | — | 1.0166 | 0.8540 |
| future_random_70 | 0 | 70 | — | 1.0526 | 0.8835 |
| future_random_105 | 0 | 105 | — | 1.0384 | 0.8721 |
| future_random_140 | 0 | 140 | — | 0.9654 | 0.8109 |
| future_block_2 | 0 | 28 | — | 1.2169 | 1.0199 |
| future_block_5 | 0 | 70 | — | 1.0153 | 0.8530 |
| future_block_8 | 0 | 112 | — | 0.9945 | 0.8359 |
| future_block_10 | 0 | 140 | — | 0.9654 | 0.8109 |

**Every context cell is 0.39-0.52. Every future-only cell is 0.965-1.22.** The split is
unambiguous once the geometry is held fixed.

Seed stability of `global_random_147` future ratio: 1.0047 / 1.0549 / 1.0265 (seeds 0/1/2).

### M1.5 cohort scoring — one forward, different selectors

| scored cohort | future ratio |
|---|---:|
| right100_on_future_block_2 | 0.9971 |
| right100_on_future_block_5 | 0.9696 |
| right100_on_future_block_8 | 0.9657 |
| right100_on_future_random_105 | 0.9627 |
| right100_on_future_random_35 | 0.9593 |
| right100_on_future_random_70 | 0.9639 |

Scoring the single `right_future_100` forward on the `future_block_2` cohort gives 0.9971,
while *masking only those two columns* gives 1.2169. Partial masking is therefore **worse**,
not better: leaving most of the future blank-but-visible feeds the encoder a large constant
region that is itself out of distribution. The `blank_visible_future` policy is a real cost.

### Stage M2 gate

| gate | value | met |
|---|---:|---|
| Outcome A: global-random **context** <= 0.80 | 0.5194 | **YES** |
| Outcome A: global-random **future** >= 0.95 | 1.0047 | **YES** |
| Outcome B: best partial-future cell <= 0.90 | 0.9945 (future_block_8) | no |
| Outcome D: right_future_100 future <= 0.90 | 0.9654 | no |

**Outcome A.** Per section 5 this forbids another patch-scatter sweep and routes to Stage M5.
M3 (iterative causal filling) was therefore not run — it is gated on Outcome B.

## 3. M5 — patch-normalization-invariant carriers

### 3.1 C1 was excluded before any model run

Section 8.2 is a pre-model numerical gate. C1 (soft edge position, `sigmoid((h-y)/0.75)`)
fails it: decode error **0.0511 > 0.02** on the dense z fixture. The edge position is quantized
by the 16-pixel patch height, so ~0.05 in z units is near this carrier's representational floor
at this resolution. C1 was excluded rather than tuned, which is what the gate is for.

C2 (phase carrier) passes all four checks: decode error **0.0000**, zero logits decode to
**z = 0.0000**, minimum cube std **3.19e-01** (no collapse), separated values give native
targets differing by 1.998.

### 3.2 C2 historical audit (ETTh2 + ETTm2, seeds 0/1/2)

| dataset | native future ratio | native PT/RAND | TS PT | TS zero | TS rand | TS I0 |
|---|---:|---:|---:|---:|---:|---:|
| ETTh2 | 0.8506 | 0.7398 | 1.1702 | 0.7001 | 2.3013 | 0.4861 |
| ETTm2 | 0.8509 | 0.7399 | 1.4736 | 0.5754 | 2.2904 | 0.3444 |

| section 8.3 promotion gate | value | met |
|---|---:|---|
| future native_ratio_zero <= 0.90 | **0.8508** | **PASS** |
| future MSE_PT / MSE_RAND <= 0.95 | **0.7399** | **PASS** |
| decoded TS MSE_PT / MSE_ZERO <= 0.95 | 2.0689 | **FAIL** |
| geo decoded TS MSE / I0 <= 0.97 | 3.2128 | **FAIL** |

**This is the most informative single result of the study.** The carrier does exactly what it
was designed to do at the representation level: future native ratio improves from 0.9654
(intensity renderer) to **0.8508**, and the pretrained model beats a random-init one by
**26.0%** on future patches. Encoding value in patch-local phase
genuinely survives `norm_pix_loss`.

But decoding that signal back to numbers is **2.07x worse than decoding zero logits** and
**3.21x worse than the incumbent**. A 15% error reduction in normalized-cube space leaves the
recovered phase far too noisy; zero logits decode to the neutral z=0, which is a much safer
forecast than a noisy phase estimate.

No carrier was promoted, so per section 13 no Stage-M4 six-dataset run was performed.

## 4. What is now established

1. **The old random/right comparison did not isolate mask topology.** 78.6% of its scored
   targets were observed history. Conclusion F of the recovery report is withdrawn.
2. **Pretrained VideoMAE is genuinely good at inpainting rendered history** (context ratio
   0.39-0.52) and **genuinely useless at extrapolating future patches** with the intensity
   renderer (0.965-1.22, i.e. no better than predicting the normalized-target mean).
3. **Partial future masking does not help** — it hurts, because unmasked-but-blank future
   patches are themselves out of distribution.
4. **A phase carrier does recover future signal** (0.8508 native, 0.740 PT/RAND),
   which contradicts a blanket "no future signal exists" reading. The bottleneck moved from
   *representation* to *numeric inversion*.

## 5. Honest limits

- M3 (iterative causal filling) was not run: it is gated on Outcome B, which was not selected.
- No Stage-M4 six-dataset strict run: nothing was promoted, and section 13 forbids running one
  on an unpromoted candidate.
- Optional **Track H (historical-calibrated hybrid) has not been run.** It is the only
  remaining route to `Q_all < 1` in this document, and it is explicitly transductive rather
  than strict zero-shot.
- Tests 16-28 in section 12 cover the iterative/carrier/hybrid stages; the iterative and hybrid
  ones are reported as SKIP because those stages were not implemented. 20 of 33 tests pass,
  13 skipped, 0 failures.
- Carrier audit used the C2 phase carrier only, at frequency 2, on two datasets.

## 6. Files

```
pilot/video_visionts_mask_factorization.py   canonical labels, matched masks, float64 subset sums
pilot/run_video_visionts_matched_audit.py    M0/M1 render-once mask-many factorization
pilot/video_visionts_carriers.py             C1/C2 carriers and native-logit decoders
pilot/run_video_visionts_carrier.py          M5 carrier audit
pilot/summarize_video_visionts_matched.py    subset tables and gates
tests/test_video_visionts_matched.py         20 pass / 13 skip / 0 fail

pilot/results_field/video_visionts_matched_mask/
  matched_audit/ 2      carrier_historical/ 6      logs/
```

