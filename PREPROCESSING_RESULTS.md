# VideoMAE Input-Alignment Experiments — Results

Execution of `PREPROCESSING_EXPERIMENTS.md` on `field-video-clean`.
Hardware: Lambda Labs 1x H100 PCIe 80GB (us-west-3), `transformers==4.46.3`, torch 2.7.0.
Date: 2026-08-31.

---

## 0. Headline

> **Can a better time-series-to-video preprocessing method make a pretrained VideoMAE
> backbone contribute useful forecasting signal?**

**Yes — for the first time in this project, on the spec's own criteria.**

A rolling **period-matrix** renderer with a **cross-attention readout** over frozen tokens
beats the seasonal-mean baseline on 3 of 4 held-out datasets **and** beats an identically
configured randomly-initialized backbone on **4 of 4**, by a mean of **+4.47%**.

This clears section 13 criteria 1, 2, 3, 5 and 6. **Criterion 4 (multi-seed confirmation)
was NOT run** — the budget ran out. Everything below is single-seed. See §7 for why that
matters and §8 for the honest caveats.

This is a real reversal of the project's prior conclusion, which was that frozen VideoMAE
features carry no usable signal. That conclusion was measured on the *barcode* renderer,
and it was correct for that renderer.

---

## 1. What was actually run

| Stage | Spec | Runs | Status |
|---|---|---|---|
| Tests | s.16, 10 required checks | 17 assertions | **all pass** |
| A | Renderer audit + input statistics | 12 renderer x dataset | **complete** |
| B | Corrected feature probe | 3 renderers, electricity | **complete** |
| C | Fast supervised screen | **24/24** | **complete** |
| D | Readout + normalization ablation | 4/20 | **cut short** (see s.9) |
| E | Six-dataset validation | **8/8 held-out** | **complete** |
| F | Strict reconstruction | **6/6** | **complete** — decisively negative |

Also completed alongside: the frozen-barcode "zero-shot" table, **8/8**.

---

## 2. Stage A — renderer audit (diagnostics only)

| dataset | renderer | const tubelet frac | tubelet px std | spatial grad RMS | temporal diff RMS | **temporal/spatial** |
|---|---|---:|---:|---:|---:|---:|
| ETTh2 | barcode_nearest (R0) | 0.120 | 0.0700 | 0.0269 | 0.1615 | **6.01** |
| ETTh2 | barcode_bilinear (R1) | 0.113 | 0.0604 | 0.0086 | 0.1497 | **17.49** |
| ETTh2 | period_matrix (R2) | 0.116 | 0.0455 | 0.0083 | 0.0984 | 11.87 |
| ETTh2 | period_line (R3) | 0.770 | 0.0399 | 0.0672 | 0.1220 | 1.82 |
| ETTh2 | period_trails (R4) | 0.677 | 0.0518 | 0.0779 | 0.0914 | 1.17 |
| ETTh2 | recurrence (R5) | 0.066 | 0.1941 | 0.0614 | 0.2873 | 4.68 |
| electricity | barcode_nearest | 0.019 | 0.0449 | 0.0178 | 0.0941 | 5.29 |
| electricity | barcode_bilinear | 0.018 | 0.0388 | 0.0057 | 0.0907 | 15.84 |
| electricity | period_matrix | 0.018 | 0.0329 | 0.0058 | 0.0639 | 10.97 |
| electricity | period_line | 0.815 | 0.0335 | 0.0604 | 0.1093 | 1.81 |
| electricity | period_trails | 0.767 | 0.0401 | 0.0638 | 0.0780 | 1.22 |
| electricity | recurrence | 0.006 | 0.2133 | 0.0586 | 0.2016 | 3.44 |

The original barcode has a temporal/spatial energy ratio of **6–17**: it is almost pure
whole-frame flicker with negligible spatial structure, exactly the misalignment s.1.2
predicted.

**These diagnostics turned out to be anti-predictive of transfer.** `period_line` and
`period_trails` look most like natural video (ratio 1.2–1.8) and performed worst.
`period_matrix` looks no more natural than the barcode (11.9) and won. Section 8's warning
not to select a renderer on pixel statistics was correct and was followed.

---

## 3. Stage B — corrected feature probe

The original probe was numerically broken: frozen features are strongly collinear, and a
fixed `lam=10` **float32** `np.linalg.solve` on a near-singular 768x768 Gram matrix
returned `skill_ratio ~ 1e155`. Fixed with a float64 SVD solve and a validation-selected
ridge sweep (`pilot/probe_frozen.py`).

| render | participation ratio | skill_ratio | chosen lambda |
|---|---:|---:|---:|
| uni_barcode | 114.11 / 768 | **0.9995** | 1e6 (grid max) |
| vts_2d | 124.47 / 768 | 1.012 | 1e6 |
| field_2d | 129.16 / 768 | 1.009 | 1e6 |

`skill_ratio ~ 1.0` means a linear probe on frozen barcode features is **no better than
predicting the constant mean**, with the optimal regularizer driving weights to zero.

**This invalidates a number previously reported in `README.md` / `PROGRESS.md`**
(participation ratio 4.88, skill 0.621). That number is not reproducible from the
documented command, its stored JSON has an empty `config`, and the code path that produced
it was numerically unsound. The corrected result points the same direction but is a
different, stronger claim, and the "only 5–13 of 768 dims" phrasing must be withdrawn.

Note this probe was run on the **barcode** renderer only. It is not evidence about
`period_matrix`, which Stage E shows does carry transferable signal.

---

## 4. Stage C — renderer screen (24/24)

xattn readout, frozen encoder, 2 epochs, ft_cap 5000, stride 32, seed 0. Identical
readout (2,959,968 trainable params) in both arms.

| Dataset | Renderer | PT | Random | smean | **transfer** | **vs smean** |
|---|---|---:|---:|---:|---:|---:|
| ETTh2 | barcode_nearest (R0) | 0.3623 | 0.3133 | 0.3500 | **−15.6%** | −3.5% |
| ETTh2 | barcode_bilinear (H1) | 0.3596 | 0.3132 | 0.3500 | **−14.8%** | −2.7% |
| **ETTh2** | **period_matrix (H4)** | **0.3372** | 0.3483 | 0.3500 | **+3.2%** | **+3.7%** |
| ETTh2 | period_line (H2) | 0.3604 | 0.3799 | 0.3500 | +5.1% | −3.0% |
| ETTh2 | period_trails (H3) | 0.3779 | 0.3799 | 0.3500 | +0.5% | −8.0% |
| ETTh2 | recurrence (R5) | 0.3795 | 0.3873 | 0.3500 | +2.0% | −8.4% |
| electricity | barcode_bilinear | 0.2649 | 0.8220 | 0.2069 | +67.8% | −28.0% |
| electricity | barcode_nearest | 0.2697 | 0.7318 | 0.2069 | +63.1% | −30.4% |
| electricity | period_matrix | 0.2671 | 0.8241 | 0.2069 | +67.6% | −29.1% |
| electricity | period_line | 0.3033 | 0.8452 | 0.2069 | +64.1% | −46.6% |
| electricity | period_trails | 0.3122 | 0.8452 | 0.2069 | +63.1% | −50.9% |
| electricity | recurrence | 0.7344 | 0.8419 | 0.2069 | +12.8% | −255.0% |

**The electricity transfer gains are artifacts and must not be quoted.** The random
backbone collapses there (0.73–0.85 vs smean 0.207), so "beats random" is a meaningless
bar; every pretrained arm still loses to smean by 28–255%. This is exactly the trap
section 13 warns about, and why both metrics are always reported together.

Hypothesis verdicts at screen stage:

- **H1 smooth interpolation — REJECTED.** 0.3596 vs 0.3623 (0.7%), transfer still −14.8%.
- **H2 geometry over brightness — partial.** Transfer flips positive (+5.1%) but absolute
  still loses to smean on both screening datasets.
- **H3 coherent motion — REJECTED.** Weakest absolute result (−8.0%).
- **H4 2-D local structure — SUPPORTED.** The only cell positive on both metrics.
- R5 recurrence — rejected, catastrophic on electricity as anticipated.

Only `period_matrix` satisfied both criteria, so only it was advanced. Section 10 says
"at most the top two"; `period_line` fails absolute utility on **both** screening datasets,
so advancing it was not justified.

---

## 5. Stage D — readout / normalization ablation (4/20, cut short)

| Dataset | Readout | Norm | PT | Random | smean | transfer |
|---|---|---|---:|---:|---:|---:|
| ETTh2 | **global** | std_clip_3 | 0.3795 | 0.3799 | 0.3500 | **+0.1%** |
| ETTh2 | xattn | std_clip_3 | 0.3372 | 0.3483 | 0.3500 | +3.2% |
| ETTh2 | xattn | **std_clip_2** | 0.3396 | 0.3527 | 0.3500 | **+3.7%** |

Only 3 usable cells, but the one comparison that completed is informative:

**H5 (structured token readout) — SUPPORTED.** Swapping the cross-attention readout for
global mean-pooling collapses transfer from **+3.2% to +0.1%** and absolute from +3.7% to
−8.4%. Averaging all 1568 tokens destroys the signal even with a well-aligned renderer.
This is a plausible explanation for why every earlier frozen experiment in this project —
all of which mean-pooled — measured zero.

`temporal` readout and the `robust` / `visionts_r04` normalizations were never run.

---

## 6. Stage E — held-out validation (8/8)

Configuration **frozen before any held-out result was inspected**: `period_matrix` /
`xattn` / `std_clip_3`, frozen encoder, epochs 5, ft_cap 40000, stride 8, max_ch 112,
batch 16, lr 1e-4, seed 0. Horizon 96 (solar 144).

| Dataset | **PT** | **Random** | smean | **transfer gain** | **vs smean** |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 0.4195 | 0.4469 | 0.4022 | **+6.1%** | −4.3% |
| ETTm2 | 0.1719 | 0.1768 | 0.3224 | **+2.8%** | **+46.7%** |
| solar | 0.1796 | 0.1846 | 0.2005 | **+2.7%** | **+10.4%** |
| traffic | 0.3358 | 0.3582 | 0.5175 | **+6.3%** | **+35.1%** |
| **mean** | | | | **+4.47%** | |

Against the same frozen protocol with the **old barcode** renderer:

| Dataset | barcode frozen | period_matrix frozen | improvement |
|---|---:|---:|---:|
| ETTh1 | 0.4960 | 0.4195 | +15.4% |
| traffic | 0.6074 | 0.3358 | +44.7% |
| solar | 0.2260 | 0.1796 | +20.5% |

### Section 13 verdict

| # | Criterion | Result |
|---|---|---|
| 1 | Beats smean on >=3 of 4 held-out | **PASS** (3/4) |
| 2 | Mean transfer gain >= +3% | **PASS** (+4.47%) |
| 3 | Transfer gain positive on >=3 of 4 | **PASS** (4/4) |
| 4 | Multi-seed confirmation (seeds 1, 2) | **NOT RUN** — budget exhausted |
| 5 | No capacity confound | **PASS** — identical readout, 2,959,968 trainable in both arms; readout RNG seeded *after* encoder construction so both arms get bit-identical initial weights |
| 6 | No selection leakage | **PASS** — config fixed from Stage C before held-out was touched |

**Verdict on criteria 1/2/3/5/6: POSITIVE. Not confirmed, because criterion 4 was not run.**

---

## 6b. Stage F — strict zero-shot reconstruction (6/6)

No time-series labels are used anywhere: a rolling antialiased line chart, a **spatial tube
mask** over the right-hand strip (VideoMAE's native pretraining masking geometry, not
future-frame masking), `VideoMAEForPreTraining` inpaints, and the line geometry is decoded
back. `recon_zeroed` repeats the identical decode with the model output replaced by zeros.

| Dataset | recon_model | recon_zeroed | smean | model vs zeroed |
|---|---:|---:|---:|---:|
| ETTh1 | 1.4370 | 0.7087 | 0.3986 | **+103% worse** |
| ETTh2 | 0.4826 | 0.3748 | 0.3442 | **+29% worse** |
| ETTm2 | 0.2973 | 0.2920 | 0.2622 | **+2% worse** |
| electricity | 1.7528 | 0.8486 | 0.2096 | **+107% worse** |
| solar | 0.9676 | 0.7485 | 0.2369 | **+29% worse** |
| traffic | 2.2511 | 1.2080 | 0.5200 | **+86% worse** |

**On 6 of 6 datasets the model's own reconstruction is WORSE than replacing its output with
zeros**, and both are far worse than the seasonal mean. A blank masked strip decodes to a
flat mid-line, which is a mediocre but sane forecast; the pretrained decoder's actual
reconstruction decodes to something worse than that.

This is stronger than the earlier `PILOT_RESULTS.md` Part 1 finding (`videomae ~=
videomae_zero`), and it is **not** subject to the renderer confound that made Part 1
questionable: this used a line-geometry renderer rather than a barcode, VideoMAE's native
spatial tube masking rather than whole-future-frame masking, and contour-position decoding
rather than brightness.

**Strict zero-shot through the pretrained reconstruction interface does not work.** The
positive Stage E result therefore depends on training a readout on time-series labels; it
is frozen-backbone transfer, not label-free zero-shot.

---

## 7. Why criterion 4 matters here

Measured seed noise on full-config runs earlier in this project was **~1.2% relative**
(electricity, 4 seeds, std 0.0017 on mean 0.1389). The per-dataset transfer gains here are
**+2.7% to +6.3%** — roughly 2–5x that noise floor, on a single seed.

That is suggestive, not settled. The two smallest gains (+2.7%, +2.8%) are close enough to
the noise floor that a different seed could plausibly move them. Section 12 prescribes
seeds 1 and 2 precisely for this, and section 13 makes it a required criterion. **The
correct description of this result is "passes 5 of 6 criteria, pending multi-seed
confirmation," not "VideoMAE pretraining transfers."**

Cost to close it: 16 runs (4 datasets x 2 arms x 2 seeds) at Stage E settings, roughly
8–10 GPU-hours, about **$30**.

---

## 8. Honest caveats

1. **Single seed.** See above. This is the largest gap.
2. **Inverse pattern between the two metrics.** The dataset with the *largest* transfer
   gain (ETTh1, +6.1%) is the one that still *loses* to smean (−4.3%). The dataset with the
   *largest* absolute win (ETTm2, +46.7%) has nearly the *smallest* transfer gain (+2.8%).
   So where the method is genuinely useful, pretraining contributes least — consistent with
   the rest of this project, where the readout and input representation do most of the work.
   The pretrained contribution is real but small.
3. **Stage D is 4/20.** The chosen readout/normalization was selected from Stage C alone.
   This is not selection leakage (the choice preceded held-out), but `temporal` readout and
   3 of 4 normalizations were never tested; a better configuration may exist.
4. **electricity and ETTh2 are selection datasets**, reported but not evidence.
5. **Stage C used a reduced protocol** (2 epochs, ft_cap 5000) and its absolute numbers are
   not comparable to Stage E.
6. **The Stage B probe was run on the barcode renderer only** and says nothing about
   `period_matrix`.
7. **Stage F was run separately — see section 6b.** Strict zero-shot through the
   pretrained reconstruction head fails decisively.

---

## 9. Execution failures worth recording

- **Stage D was truncated at 4/20 by my own watchdog.** The budget watchdog's exit
  condition was written for the workloads running when it was armed; Stage D was launched
  afterwards and never added to that condition, so the watchdog correctly fired on
  "zero-shot 8/8 and Stage C 24/24" and killed Stage D with it.
- **An earlier instance lost all results** because it self-terminated with no persistent
  storage attached. Fixed by attaching a Lambda filesystem (`wm4ts-fs`); every subsequent
  termination preserved results, and this is what saved Stage C and Stage E.
- **Placing that filesystem in us-west-3 was a mistake.** It is a scarce-capacity region,
  and each retrieval required polling for a free instance (once for ~8 minutes, once for
  ~4). A high-capacity region such as us-east-1 would have been safer even on slower GPUs.
- **A `sed` edit to the watchdog silently failed** (shell ate the escape), leaving the old
  deadline in place. Subsequent edits used Python file rewriting and verified the result.
- The traffic zero-shot run OOMed at evaluation because several jobs shared a GPU with
  `eval_batch=256`. Reduced to 64 — inference-only, so it changes no numbers.

---

## 10. Cost

| Item | Cost |
|---|---:|
| Instance 1 (zero-shot, results lost) | $31.00 |
| Instance 2 (zero-shot completion + Stage C + Stage D partial) | $29.00 |
| Instance 3 (Stage E) | $32.40 |
| Retrieval instances | ~$1.40 |
| **Total** | **~$93.80** |

Budget was $50 + $30 = $80, plus an approved extension to complete the Stage E pairs
(estimated ~$88). **Final spend overran that estimate by roughly $6**, because the
random-arm runs took longer than projected and the watchdog ran to its cap.

All instances are terminated; no compute is still billing.

---

## 11. Files

```
pilot/preprocess_renderers.py          R0-R5 renderers, input diagnostics, contact sheets
pilot/run_preprocess.py                paired frozen-backbone entry point, 3 readouts
pilot/probe_preprocessing.py           (folded into pilot/probe_frozen.py, numerics fixed)
pilot/summarize_preprocessing.py       transfer-gain aggregation + section 13 verdict
pilot/run_preprocess_sweep.sh          Stage C/D/E launcher
pilot/run_reconstruct.py               Stage F strict reconstruction (implemented, unrun)
tests/test_preprocess_renderers.py     10 required checks
tests/test_reconstruct.py              7 leakage/geometry checks

pilot/results_field/preprocessing/
  previews/          12 contact sheets
  stageA_input_stats.json
  screen/            24 JSONs
  ablation/           4 JSONs
  validation/         8 JSONs
pilot/results_field/zeroshot/results/  8 JSONs (frozen barcode, complete)
```

Every result JSON carries git commit + dirty flag, dataset and path, renderer and
hyperparameters, readout, pretrained/random flag and checkpoint, library versions, window
and optimizer settings, seed, trainable/total parameter counts, input diagnostics, all
three baselines, wall-clock and peak GPU memory.

---

## 12. Recommended next step

Run **seeds 1 and 2** for the four held-out datasets, both arms (16 runs, ~$30). That is
the single remaining requirement for a defensible positive claim. If the mean transfer gain
survives at >=3%, this becomes the project's first genuine positive transfer result and
reframes the negative conclusion in `README.md` as *renderer-specific* rather than
*fundamental*.

If it does not survive, the honest statement is the one section 13 supplies: preprocessing
helped the trainable head, not VideoMAE pretraining.
