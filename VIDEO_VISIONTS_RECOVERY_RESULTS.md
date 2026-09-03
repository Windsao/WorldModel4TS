# Video-VisionTS Recovery — Results

Execution of `VIDEO_VISIONTS_RECOVERY_EXPERIMENTS.md`. NU CS cluster (nyx, 4x A40), free.

## 0. Bounded conclusion

**F — no layout preserves the random-mask advantage when masked patches are constrained to be
future-only; mask/task mismatch is the remaining bottleneck.**

With **E** as the companion result for track 1: the corrected lenses fail historical
validation, so the previous Stage E1 claim was invalid *and* the corrected result is now a
valid negative.

`Q_all = 1.0461` (previous best 1.0391), 95% CI **[1.0151, 1.0839]** — entirely above 1.

## 1. R0 — the previous Stage E1 was invalid, confirmed

The old training path called `render_batch(..., blank_future=True)` and then used the returned
`video_full` as the historical pseudo-future target. Verified directly: the hidden region of
that tensor has **std = 0.0** — a constant block. The lens was trained to predict the
statistics of a constant, not of the pseudo-future.

The old outputs in `video_visionts/label_free_adaptation/` are retained for provenance and
labelled **`invalid_target_bug`**. These earlier claims are withdrawn: *"E1 fails decisively"*,
*"the statistics do not transfer across datasets"*, *"the label-free bridge does not remove the
bottleneck"*. They had not been tested by a valid experiment.

The repaired builder has no `blank_future` switch and always receives the real pseudo-future;
`dense_static` blanks it in the encoder input only. All 24 section-11 tests pass, 0 skipped,
including a regression test that reproduces the old bug and shows its target is insensitive to
the pseudo-future.

Target distribution on the audit partition: fraction of cube/channel std <= 1e-5 is
**0.25 (ETTh2) / 0.19 (ETTm2)** — below the 50%% degeneracy stop threshold.

## 2. R3 — historical layout audit (the central result)

| layout | native ratio (ETTh2+ETTm2 geo-mean) | vs G0 | gate 1 | oracle TS gain | gate 2 |
|---|---:|---:|---|---:|---|
| **G0** contiguous | **0.9687** | baseline | — | — | — |
| G1 seed 0 | 1.0418 | -7.5% | FAIL | +1.5% | FAIL |
| G2 seed 0 | 1.5293 | -57.9% | FAIL | -3.6% | FAIL |
| G2 seed 1 | 1.5327 | -58.2% | FAIL | -4.4% | FAIL |
| G2 seed 2 | 1.5217 | -57.1% | FAIL | -3.6% | FAIL |
| G0 random-init control | 1.1905 | — | — | — | — |

**Both scattered layouts make reconstruction substantially worse, not better.** G2 is 1.52-1.53
across all three layout seeds — worse than G0 (0.969) and worse than a random-init backbone
(1.191).

### Why this matters more than the numbers

The whole premise of track 2 was that VideoMAE reconstructs `random_tube_75` far better
(0.585) than a contiguous right block (0.929), so making the future-mask *look* like
pretraining should recover that advantage. It does the opposite.

The reason is that the `random_tube` advantage never came from mask topology. In that audit the
**content** was still the canonical dense period matrix, so every masked patch had spatially
adjacent visible patches holding temporally adjacent values, and MAE inpainting is exactly an
interpolation prior over local spatial continuity. Scattering the future patches to reshape the
mask necessarily destroys that continuity: a patch's neighbours no longer carry neighbouring
times. **You cannot have both the pretraining-like mask topology and the spatial coherence that
made the inpainting work — they are the same degree of freedom.**

This closes hypothesis D2 (mask mismatch) as an actionable direction: it is real, but it is not
separately fixable while the masked region must be exactly the future.

## 3. R4 — corrected lenses (valid negative)

| variant | aggregate ratio vs L0 | gate (<= 0.97) |
|---|---:|---|
| L1 | 1.0347 | FAIL |
| L2 | 1.0204 | FAIL |
| L3 | 1.0216 | FAIL |

With the target bug fixed, all three lenses still fail — but they now fail *gently* (1.020-1.035)
instead of catastrophically. The bounded residual design (L2/L3) does what it was meant to do:
it keeps the lens near its zero-init causal starting point rather than destroying it, which is
why the old 2.4x blow-up does not recur. Per section 7.2 no lens was promoted, so seeds 1/2 and
the paired random-backbone control were not run on failed candidates.

## 4. R5 — frozen genuine-future evaluation

Configuration frozen before any genuine future was read (`frozen_configs/R5_primary.json`):
`dense_static` / width rule (right_10, right_9 for ETTm2 and solar) / **G0** / 
`nearest_visible_physical` / **L0 (no lens)**.

| dataset | method | zeroed | random | VisionTS | Stage-D right_10 | smean | q vs VisionTS |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.3984 | 0.4110 | 0.4107 | 0.4023 | 0.3984 | 0.4026 | 0.990 |
| ETTh2 | 0.3248 | 0.3267 | 0.3296 | 0.3130 | 0.3248 | 0.3454 | 1.038 |
| ETTm2 | 0.2553 | 0.2569 | 0.2580 | 0.2509 | 0.2615 | 0.3301 | 1.018 |
| electricity | 0.2313 | 0.2567 | 0.2581 | 0.2014 | 0.2313 | 0.2069 | 1.149 |
| traffic | 0.6018 | 0.6242 | 0.6296 | 0.5207 | 0.6018 | 0.5106 | 1.156 |
| solar | 0.2575 | 0.2718 | 0.2735 | 0.2729 | 0.2415 | 0.1973 | 0.944 |

- **Q_all = 1.0461** — primary claim `Q_all < 1` **NOT MET** (previous best was 1.0391)
- **Q_heldout = 1.0554**
- paired 95%% moving-block CI: Q_all **[1.0151, 1.0839]**, Q_heldout **[1.0160, 1.0986]** — both entirely above 1
- win/loss vs VisionTS: 2/4 (ETTh1, solar)
- leave-one-out Q_all: ETTh1 1.058, ETTh2 1.048, ETTm2 1.052, electricity 1.027, solar 1.068, traffic 1.025
- **attribution holds**: Q(method/zeroed) = **0.9611**, Q(method/random) = **0.9559**
- vs the previous Stage-D configuration: 1.0067 — the mechanical width rule is a wash
  (ETTm2 improves 0.2615 -> 0.2553 with right_9, solar degrades 0.2415 -> 0.2575)

## 5. R6 — raw-pixel decoder adaptation (conditional gate was met)

| arm | aggregate ratio vs L0 | gate (<= 0.97) |
|---|---:|---|
| pretrained frozen encoder + adapted decoder | 1.2339 | FAIL |
| random frozen encoder + adapted decoder | 1.5985 | FAIL |

Not promoted, so it did not enter genuine-future evaluation. **But the paired control is
informative**: the pretrained encoder is 22.8% better than the random one under an
identically initialised and identically trained decoder. Retraining the decoder on raw pixels
removes the `norm_pix_loss` bottleneck at its source, yet the adapted decoder still cannot beat
a zero-parameter causal rule.

## 6. What the recovery study establishes

1. **My previous Stage E1 conclusion was invalid and is withdrawn.** The bug was real and is
   now reproduced in a regression test.
2. **The corrected lenses give a valid negative** (1.020-1.035), so the honest version of the
   old claim survives, but only now that it has actually been tested.
3. **Mask topology is not separately fixable.** The `random_tube` advantage is a property of
   content locality, not of the mask, and the two cannot be decoupled when the masked region
   must be exactly the future. This is the strongest new result.
4. **Video pretraining keeps contributing a small, consistent amount** across every valid
   comparison: 0.956 vs random and 0.961 vs zero logits in R5,
   0.969 vs 1.191 native in R3, and 1.234 vs 1.598 in R6.
   The attribution is real; the competitiveness is not.
5. **The 4% gap to VisionTS is statistically solid**, not noise: the paired bootstrap CI for
   Q_all is [1.0151, 1.0839].

## 7. Honest limits

- R4 stopped at the seed-0 screen because no lens passed; seeds 1/2 and the paired random
  control were therefore not run for the lenses (spec section 7.2 forbids promoting failures).
- R6 ran one seed per arm; it was not promoted, so multi-seed was not spent on it.
- G1 used a single deterministic column placement; only G2 was tested across three seeds.
- The width rule changes ETTm2/solar to `right_9`, which is a confound between the layout study
  and the Stage-D comparison; the net effect measured 1.0067, i.e. negligible.
- Per-origin errors are now stored for every method, so any pair can be bootstrapped; the older
  Stage-D JSONs still only carry them for the pretrained arm.

## 8. Files

```
pilot/video_visionts_layouts.py                 width rule, G0/G1/G2, masks, inverse perms
pilot/run_video_visionts_recovery_audit.py      R0 target builder, R1 manifests, R3 audit
pilot/run_video_visionts_stat_lens_v2.py        L1 / L2 bounded / L3 tube-shared
pilot/run_video_visionts_recovery_forecast.py   R5 frozen strict forecast
pilot/run_video_visionts_raw_decoder.py         R6 raw-pixel decoder adaptation
pilot/summarize_video_visionts_recovery.py      grouping, gates, bootstrap CIs
tests/test_video_visionts_recovery.py           24/24 pass, 0 skipped

pilot/results_field/video_visionts_recovery/
  layout_audit/ 12   historical_eval/ 3   strict_forecast/ 6   raw_decoder/ 2
  manifests/  frozen_configs/R5_primary.json  lens_checkpoints/  logs/
```

