# Video-VisionTS Diagnostic — Results

Execution of `VIDEO_VISIONTS_EXPERIMENTS.md`. Cluster: NU CS slurm (nyx, 4x A40), free.
`transformers==4.46.3`, VisionTS commit `7f34731`, checkpoint `mae_visualize_vit_base.pth`.

## 0. Bottom line

**First failed gate: the contiguous right-block mask (spec section 6.5 gate 2).**
Conclusion **C — mask mismatch**, with **D — decoder-domain mismatch** confirmed as a
second, independent bottleneck.

VideoMAE *can* reconstruct these renderings: under the near-pretraining `random_tube_90`
mask its native reconstruction error is **0.48x** the zero-logit baseline, and a
random-initialized backbone scores **1.17x** (worse than zeros) -- so that gain is
genuinely from pretraining. The failure appears only when the masked region becomes a
contiguous right block (0.48 -> 0.93) and again when per-cube statistics must be
estimated causally instead of read from the hidden target (3-10x worse).

## 1. Stage A — VisionTS reference (required positive control)

| dataset | VisionTS | zeroed | random | smean | snaive |
|---|---:|---:|---:|---:|---:|
| ETTh1 | **0.4023** | 0.7103 | 1.0368 | 0.4026 | 0.5043 |
| ETTh2 | **0.3130** | 0.3770 | 0.5033 | 0.3454 | 0.3852 |
| ETTm2 | **0.2509** | 0.3618 | 0.5595 | 0.3301 | 0.2651 |
| electricity | **0.2014** | 0.8562 | 1.2652 | 0.2069 | 0.3322 |
| traffic | **0.5207** | 1.1560 | 1.7575 | 0.5106 | 0.9491 |
| solar | **0.2729** | 0.7356 | 1.2751 | 0.1973 | 0.2851 |

- Q(VisionTS / zeroed) = **0.4837**, Q(VisionTS / random) = **0.3209**
- mirrored forward reproduces the official forward exactly (max|d| = 0.00e+00)
- **Stage A gate: PASS** — the image-MAE reconstruction interface demonstrably works.

This is the control our VideoMAE Stage F never had. The pretrained image decoder beats
zeros by 2.1x and random weights by 3.1x.

## 2. Stage B — packing validation

- manual native-cube target reproduces `VideoMAEForPreTraining(...).loss` to 6 decimals
  (0.999221 vs 0.999221); permute is `(0,1,4,6,2,5,7,3)`, channel-last, `var(unbiased=True).sqrt()+1e-6`
- decoder logits are **bit-identical** when masked-region pixels change (max|d| = 0.00e+00)
  and change by 0.89 when a *visible* patch changes, so the invariance test is meaningful
- mask counts 176/147/182/140/42 of 196, each repeated identically over all 8 tubelets
- the invalid ablation in section 5.3 (flipping `norm_pix_loss` at inference) was **not** run

## 3. Stage C — historical reconstruction (regime HA, n=256, ETTh2 + ETTm2)

### 3.1 Native normalized-cube reconstruction (`native_ratio`, lower is better, <1 beats zeros)

| renderer | mask | ETTh2 | ETTm2 |
|---|---|---:|---:|
| dense_rolling | random_tube_90 | 0.479 | 0.508 |
| dense_rolling | random_tube_75 | 0.354 | 0.378 |
| dense_rolling | right_13 | 0.985 | 0.980 |
| dense_rolling | right_10 | 0.942 | 0.942 |
| dense_rolling | right_3 | 1.394 | 0.811 |
| dense_static | random_tube_90 | 0.764 | 0.789 |
| dense_static | random_tube_75 | 0.585 | 0.640 |
| dense_static | right_13 | 0.960 | 0.957 |
| dense_static | right_10 | 0.929 | 0.927 |
| dense_static | right_3 | 1.026 | 1.008 |
| line_rolling_legacy | random_tube_90 | 0.873 | 0.830 |
| line_rolling_legacy | random_tube_75 | 0.766 | 0.702 |
| line_rolling_legacy | right_13 | 0.959 | 0.944 |
| line_rolling_legacy | right_10 | 0.929 | 0.904 |
| line_rolling_legacy | right_3 | 161.347 | 199.081 |

- random-init control (`dense_rolling`/`random_tube_90`): **1.171** — worse than zeros,
  so the pretrained gain is not an architectural artifact.
- `line_rolling_legacy` + `right_3` gives 161-199: its masked strip is blank background, so the
  per-cube normalized target is amplified noise. **This mechanically explains the old Stage F
  result where the model scored worse than zeroed logits** — it was decoding a degenerate target.

### 3.2 Oracle vs causal inversion (raw-pixel MSE)

| renderer | mask | oracle | zero+oracle | causal (nearest-left) | causal (global) |
|---|---|---:|---:|---:|---:|
| dense_static | random_tube_90 | 0.00349 | 0.00507 | 0.04637 | 0.03187 |
| dense_static | right_10 | 0.00198 | 0.00221 | 0.03237 | 0.04127 |
| dense_static | right_3 | 0.00491 | 0.00474 | 0.03435 | 0.04150 |
| dense_rolling | random_tube_90 | 0.00341 | 0.00787 | 0.04325 | 0.03225 |
| dense_rolling | right_10 | 0.00664 | 0.00707 | 0.03113 | 0.03908 |
| dense_rolling | right_3 | 0.01170 | 0.00787 | 0.03390 | 0.04171 |

(ETTh2 shown; ETTm2 is the same pattern.) Causal statistics are **3-10x worse than oracle**.
Under `right_10`, oracle itself is only ~10% better than zero+oracle, i.e. the per-cube mean/std
carry almost all of the reconstruction and the logits add little.

### 3.3 Decision tree (section 6.5)

```
pretrained beats zero logits on native_norm under random_tube_90?   YES (0.48-0.87, all renderers)
  -> also for a contiguous right mask?                              LARGELY NO (0.93-1.03)  <-- FIRST FAILED GATE
       -> mask topology / extrapolation boundary is the bottleneck  == conclusion C
  (separately) oracle_stats recovers history but causal_stats does not  == conclusion D
```

## 4. Stage D — strict zero-shot forecasting (regime ZS, n=2000, stride 8)

Configuration frozen from Stage C before any genuine future was read:
`dense_static` / `right_10` / `nearest_visible_left`.

### dense_static / right_10

| dataset | pretrained | zeroed | random | VisionTS | smean | snaive | q vs VisionTS |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.3984 | 0.4110 | 0.4107 | 0.4023 | 0.4026 | 0.5043 | 0.990 |
| ETTh2 | 0.3248 | 0.3267 | 0.3296 | 0.3130 | 0.3454 | 0.3852 | 1.038 |
| ETTm2 | 0.2615 | 0.2623 | 0.2636 | 0.2509 | 0.3301 | 0.2651 | 1.042 |
| electricity | 0.2313 | 0.2567 | 0.2581 | 0.2014 | 0.2069 | 0.3322 | 1.149 |
| traffic | 0.6018 | 0.6242 | 0.6296 | 0.5207 | 0.5106 | 0.9491 | 1.156 |
| solar | 0.2415 | 0.2572 | 0.2581 | 0.2729 | 0.1973 | 0.2852 | 0.885 |

- **Q_all = 1.0391**, **Q_heldout = 1.0386**
- Q(pretrained/zeroed) = **0.9601**, Q(pretrained/random) = **0.9553**
- Q(pretrained/smean) = 1.0292

### dense_rolling / right_10

| dataset | pretrained | zeroed | random | VisionTS | smean | snaive | q vs VisionTS |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.4076 | 0.4212 | 0.4209 | 0.4023 | 0.4026 | 0.5043 | 1.013 |
| ETTh2 | 0.3397 | 0.3420 | 0.3451 | 0.3130 | 0.3454 | 0.3852 | 1.085 |
| ETTm2 | 0.2843 | 0.2847 | 0.2860 | 0.2509 | 0.3301 | 0.2651 | 1.133 |
| electricity | 0.2232 | 0.2493 | 0.2506 | 0.2014 | 0.2069 | 0.3322 | 1.109 |
| traffic | 0.5730 | 0.5941 | 0.5993 | 0.5207 | 0.5106 | 0.9491 | 1.101 |
| solar | 0.2473 | 0.2633 | 0.2642 | 0.2729 | 0.1973 | 0.2852 | 0.906 |

- **Q_all = 1.0548**, **Q_heldout = 1.0288**
- Q(pretrained/zeroed) = **0.9591**, Q(pretrained/random) = **0.9542**
- Q(pretrained/smean) = 1.0449

### Claims (section 7.2)

| claim | gate | result |
|---|---|---|
| Primary forecasting — better than VisionTS | Q_all < 1 | **NOT MET** (1.0391) |
| Cross-dataset generalization | Q_heldout < 1 | **NOT MET** (1.0386) |
| Attribution — decoder contributes | Q(PT/zeroed) < 1 | **MET** (0.9601, +4.0%) |
| Attribution — pretraining contributes | Q(PT/random) < 1 | **MET** (0.9553, +4.5%) |

Per-dataset vs VisionTS: wins on **ETTh1 (0.990)** and **solar (0.885)**, loses on the other four.

## 5. Bounded conclusion

**C — mask mismatch** (first failed gate), **plus D — decoder-domain mismatch**.

Not B: the renderer is not out of distribution; VideoMAE reconstructs it well under
pretraining-like masks. Not F/G: the method does not beat VisionTS in aggregate
(3.9% worse), although it does beat both the zeroed and random controls by ~4%, so the
pretrained video decoder contributes a small but real amount of signal.

### What this changes

The previous project conclusion -- *strict zero-shot through VideoMAE's reconstruction
interface does not work, 6/6 worse than zeroed logits* -- was measured on the legacy line
renderer with a 21.4% right strip whose masked region was blank background. Section 3.1
shows that target is degenerate. With a dense period matrix the same interface beats the
zeroed and random controls on all six datasets. The failure is narrower and better located
than previously reported: it is the contiguous-block mask topology and the causal recovery
of per-cube statistics, not the backbone or the representation.

## 6. Stage E is justified

Section 8 gates Stage E on *native or oracle reconstruction containing signal while strict
causal inversion fails*. That is exactly the measured pattern:
native 0.48x with pretraining (random-init 1.17x), oracle 0.0026 vs 0.0051 zero, and causal
3-10x worse than oracle. E1 (a <500k-parameter patch-stat lens predicting each masked cube's
raw mean/log-std from decoder tokens, position and visible-boundary statistics, trained only
on historical pseudo-futures) targets exactly the identified bottleneck.

## 7. Files

```
pilot/videomae_patch_utils.py         exact cube packing, native targets, 5 masks, causal stats
pilot/video_visionts_renderers.py     V0 dense_static, V1 dense_rolling, inverse transforms
pilot/run_visionts_reference.py       Stage A
pilot/run_videomae_decoder_audit.py   Stages B/C
pilot/run_video_visionts.py           Stage D
pilot/summarize_video_visionts.py     aggregation and gates
tests/test_video_visionts.py          all 16 required checks, 0 skipped

pilot/results_field/video_visionts/
  visionts_reference/        6 JSONs      manifests/  6
  historical_reconstruction/ 60 JSONs     strict_forecast/ 12
  previews/                  6 sheets     logs/
```


---

## 8. Stage E1 — label-free patch-stat lens (regime LA)

Gated in by Stage C: native/oracle reconstruction carries signal while causal inversion
fails, which is exactly the condition section 8 requires. A <500k-parameter lens predicts
each masked cube's raw mean and log-std from the decoder logits, cube position and visible
boundary statistics. Trained ONLY on pseudo-futures cut from ETTh2/ETTm2 **training-split
history**; no forecast target from any held-out dataset entered training or selection.
The head is zero-initialised, so at step 0 the lens is exactly the `nearest_visible_left`
causal rule and can only be judged by whether training moves it.

Lens: 305,286 parameters. Paired control: identical lens, identical init and pseudo-windows,
on a random-init frozen backbone. Seeds 0/1/2, mean reported.

| dataset | PT + lens | RAND + lens | PT + causal (no lens) | VisionTS | smean |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 0.5721 | 0.6103 | 0.3994 | 0.4023 | 0.4026 |
| ETTh2 | 0.4940 | 0.4426 | 0.3279 | 0.3130 | 0.3454 |
| ETTm2 | 0.4236 | 0.3722 | 0.2579 | 0.2509 | 0.3301 |
| electricity | 0.5101 | 0.5177 | 0.2340 | 0.2014 | 0.2069 |
| traffic | 1.0092 | 0.9790 | 0.6147 | 0.5207 | 0.5106 |
| solar | 1.5187 | 1.2813 | 0.2476 | 0.2729 | 0.1973 |

| claim (section 8) | gate | result |
|---|---|---|
| forecasting: beats VisionTS | Q_heldout < 1 | **NOT MET** (2.4967 — 2.5x worse) |
| attribution: pretrained beats random | <= 0.97 | **NOT MET** (1.0306 — pretrained is 3% *worse*) |
| robustness: survives seeds 0/1/2 | — | fails on all three seeds |
| sanity: smean/snaive | reported | lens loses to smean on 6/6 |

- **Q_heldout(lens / plain causal rule) = 2.3680** — the lens made the forecast **2.4x worse**
  than the zero-init rule it started from.

**E1 fails decisively.** The lens fits the per-cube statistics of ETTh2/ETTm2 renderings and
does not transfer: solar degrades from 0.248 to 1.519, traffic from 0.615 to 1.009. The raw
per-cube mean/std of a *rendered* period matrix are dataset-specific in a way a shared lens
trained on two datasets' history cannot capture. Conclusion **H is not met**.

This is a clean negative rather than an inconclusive one: the paired random-backbone control
ran identically, and the pretrained arm is marginally *worse*, so the failure is not an
artefact of the bridge favouring one backbone.

## 9. Final bounded conclusion

**C — mask mismatch**, first failed gate, with **D — decoder-domain mismatch** as an
independent second bottleneck that a label-free bridge does not remove.

| stage | question | answer |
|---|---|---|
| A | does the reference implementation work? | **yes** — VisionTS beats zeroed 2.1x, random 3.1x |
| B | is the packing/target correct? | **yes** — manual loss matches `model.loss` to 6 dp |
| C gate 1 | can VideoMAE reconstruct the renderer natively? | **yes** — 0.48x zeros under `random_tube_90`; random-init 1.17x |
| C gate 2 | does that survive a contiguous right block? | **largely no** — 0.48 -> 0.93 |
| C gate 3 | does oracle inversion recover the series? | only marginally for right blocks |
| C gate 4 | do causal statistics recover it? | **no** — 3-10x worse than oracle |
| D | strict zero-shot competitive with VisionTS? | **no** — Q_all 1.039; but beats zeroed 0.960 and random 0.955 |
| E1 | does a label-free lens fix the statistics? | **no** — 2.4x worse than the causal rule it replaced |

### What is now established that was not before

1. **VideoMAE is not blind to rendered time series.** Under pretraining-like tube masking it
   reconstructs them well, and a random-init backbone does not. The earlier project claim that
   frozen VideoMAE features carry no usable signal was renderer- and readout-specific.
2. **The old Stage F catastrophe has a mechanical cause**, not a conceptual one: its masked
   strip was blank background, so the per-cube normalised target was amplified noise
   (`native_zero` = 0.0002, `native_ratio` = 161-199).
3. **The real obstacles are narrow and now located**: contiguous-block mask topology, and the
   causal recovery of the per-cube statistics that `norm_pix_loss` discarded. Neither is the
   backbone and neither is the representation.
4. **A small label-free lens does not bridge (2)** — those statistics do not transfer across
   datasets.

### Honest limits

- Stage C used n=256 per cell; Stage D/E used n=2000 with stride 8.
- The paired moving-block bootstrap CI specified in section 3.4 was implemented in
  `summarize_video_visionts.py` but per-origin errors were only stored for the pretrained arm,
  so CIs are reported for Q against VisionTS only, not for every control pair.
- E2 (raw-pixel decoder adaptation) was not run; E1's failure mode suggests it would need
  per-dataset adaptation, which section 8 says must be reported separately as transductive.
- `right_10` allocates 4 visible patch columns to 16 periods and 10 masked columns to 4
  periods, so the time axis is compressed ~10x more on the visible side. This is what the
  spec's mask table prescribes, but it is a geometric distortion worth revisiting.

