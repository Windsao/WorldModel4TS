# Video-VisionTS Carrier Signal — Results

Execution of `VIDEO_VISIONTS_CARRIER_SIGNAL_EXPERIMENTS.md`. NU CS cluster (nyx, A40), free.
Scope: ETTh2 + ETTm2, historical audit split only. No genuine test future was read.

## 0. Conclusion

**Pretrained VideoMAE reconstructs the carrier texture but not the target-dependent phase.**

First failed gate: `G_conditional`. Per section 10.2 the study stops here — no calibration,
no VisionTS reference, no additional frequencies or datasets were run.

## 1. My previous M5 claim was an artifact, and is withdrawn

`VIDEO_VISIONTS_MATCHED_MASK_RESULTS.md` reported `native_ratio_zero = 0.8508` and read it as
*"the carrier recovers pretrained signal on future patches"*. That comparison used B0 (zero
logits) as the baseline. A C2 native target is **always** a nonzero sinusoid, so emitting a
generic sinusoid with arbitrary phase already beats zero. Measured against the carrier-aware
baseline B1 — the exact native target of a neutral z=0 carrier, which has the correct
frequency and amplitude and differs only in phase:

| native cube MSE | ETTh2 | ETTm2 | aggregate ratio |
|---|---:|---:|---:|
| B0 zero logits | 0.99804 | 0.99804 | — |
| **B1 neutral carrier** | **0.40645** | **0.35219** | — |
| pretrained X_true | 0.84932 | 0.84849 | **0.8506 vs B0** / **2.2437 vs B1** |
| random-init X_true | 1.15163 | 1.15274 | — |

The 0.85 reproduces exactly. Against B1 the same predictions are **2.24x worse**. The
pretrained decoder is not beating a phase-free prior; it is losing to one.

## 2. G_codec — PASS, with large margin

| dataset | clip frac | O1 MSE | O2 MSE | O3 MSE | O3/ctx-mean | O3/I0 | O2-O3 max diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh2 | 0.0721 | 0.02454 | 0.16201 | 0.16201 | 0.2314 | 0.3333 | 7.2e-07 |
| ETTm2 | 0.0443 | 0.03545 | 0.08819 | 0.08819 | 0.1537 | 0.2661 | 9.5e-07 |

| G_codec check | value | threshold | status |
|---|---:|---:|---|
| O0 max abs err | 5.66e-07 | 1e-5 | PASS |
| O2/O3 max abs diff | 9.54e-07 | 2e-4 | PASS |
| O3/ctx-mean, both datasets | 0.2314 | 0.50 | PASS |
| geo O3/I0 | 0.2978 | 0.90 | PASS |
| max O3/I0 | 0.3333 | 1.00 | PASS |

**The representation is not the problem.** With perfect native targets the 14x10 C2 codec
reconstructs the pseudo-future **3.0-3.8x better than the incumbent I0**. Clipping is the main
loss (O1 0.025/0.035 -> O2 0.162/0.088 at only 4-7% clipped, because the z tails reach 2.6/1.3),
but even so there is ample headroom. Any later failure is a model failure, not a codec ceiling.

## 3. G_conditional — FAIL (the terminal gate)

All numbers are the fixed parameter-free D2 coherence-shrunk decoder, z-grid MSE, audit split.

| arm | ETTh2 | ETTm2 |
|---|---:|---:|
| B0 zero logits | 0.1874 | 0.1598 |
| **B1 neutral carrier** | 0.1874 | 0.1598 |
| B3 smean carrier | 0.1744 | 0.1465 |
| B4 snaive carrier | 0.1860 | 0.1238 |
| **pretrained X_true** | 0.3593 | 0.3249 |
| pretrained X_shuffle | 0.3726 | 0.3347 |
| pretrained X_neutral | 0.3557 | 0.3100 |
| Y_shuffle (target permuted) | 0.3429 | 0.3170 |
| random-init X_true | 0.7531 | 0.7768 |

| G_conditional check | value | threshold | status |
|---|---:|---:|---|
| Q2(PT true / B1 neutral) | 1.9747 | <= 0.95 | **FAIL** |
| Q2(PT true / X_shuffle) | 0.9676 | <= 0.95 | **FAIL** |
| Q2(PT true / Y_shuffle) | 1.0364 | <= 0.95 | **FAIL** |
| no dataset worse than its shuffled control | none | — | PASS |
| Pearson z correlation > 0 on both | 0.0763 | > 0 | PASS |

Three observations make this unambiguous:

1. **Predicting the neutral phase is twice as good as the model.** PT true 0.359/0.325 vs B1
   0.187/0.160.
2. **Scoring against a *different* sample's target is better than the true pairing.**
   Y_shuffle 0.343/0.317 beats PT true 0.359/0.325. The output is closer to the unconditional
   phase distribution than to its own target.
3. **Removing the context barely changes anything.** X_neutral 0.356/0.310 is as good as or
   better than X_true 0.359/0.325.

### The one real conditional effect, and why it does not rescue the gate

Paired moving-block bootstrap on time-series error, true vs shuffled context
(5,000 replicates, seed 20260902):

| dataset | ratio 95% CI | P(true better) | origins | block |
|---|---|---:|---:|---:|
| ETTh2 | [0.9582, 0.9929] | 0.999 | 36 | 15 |
| ETTm2 | [0.9970, 1.0178] | 0.064 | 153 | 51 |

ETTh2 shows a genuine but tiny conditional effect (CI excludes 1, P=0.999); ETTm2 does not
(CI spans 1, P=0.064). The aggregate is 0.968 against a 0.95 threshold. So correct context is
worth roughly 3% — real on one dataset, absent on the other, and an order of magnitude too
small to matter.

## 4. Pretrained vs random, and why it is not evidence

Random-init z-MSE is 0.753/0.777 versus
pretrained 0.359/0.325 — pretrained is about **2.3x better**.
In the previous reports that ratio was treated as attribution evidence. It is not sufficient
here: **both arms lose to B1**, a parameter-free neutral texture with no model at all. The
pretrained weights produce a cleaner sinusoid than random weights; neither produces the right
phase. `G_pretrain` was not formally evaluated because `G_conditional` is sequentially prior
and terminal.

## 5. Time-series consequences (D0-D3, all-tubelet decoding)

| TS MSE | ETTh2 | ETTm2 |
|---|---:|---:|
| D2 pretrained (primary) | 1.0932 | 1.2605 |
| B1 neutral carrier | 0.7001 | 0.5738 |
| D2 random-init | 2.2067 | 2.3269 |
| context mean | 0.7001 | 0.5738 |
| smean | 0.6760 | 0.5517 |
| snaive | 0.5289 | 0.3084 |
| I0 incumbent | 0.4861 | 0.3313 |
| O3 codec oracle | 0.1620 | 0.0882 |

Aggregating all eight tubelets did help the decoder — D0 (first tubelet only, the previous
implementation) 1.1702/1.4592 improves to D2
1.0932/1.2605, about 6-14%. But the result is still worse than
predicting the context mean, so the aggregation fixes decoder noise, not the missing signal.

## 6. Claims now retired

| claim | status |
|---|---|
| `C2 recovers future-conditioned representation signal` | **refuted** (2.24x worse than B1) |
| `native_ratio_zero = 0.85 proves useful phase prediction` | **refuted** — it proves a sinusoid was emitted |
| `numeric inversion is the only remaining bottleneck` | **refuted** — the codec has 3x headroom; phase prediction is the bottleneck |
| `the phase-carrier route is closed` | not claimed; only *this* static C2 configuration is closed |

Retained from earlier work: the fixed 4/10 geometry removes the old renderer confound,
global-random context reconstruction is strong, and global-random future reconstruction is
approximately equal to zero logits.

## 7. Honest limits

- Two datasets, one manifest seed, one shuffle seed, one random-init seed. Section 6.2
  robustness extensions were not run because they are gated on the primary gate passing.
- Historical audit split only; no genuine future was touched.
- This tests the **static** C2 carrier repeated over 16 frames. It is not a test of learned
  video motion extrapolation, and failure here says nothing about video backbones in general.
- ETTh2 has only 36 unique origins with block length 15, so its bootstrap has limited
  resolution; that is reported rather than worked around.
- C1 remains excluded by its section-8.2 decode-accuracy check (0.0511 > 0.02).

## 8. Files

```
pilot/video_visionts_carrier_signal.py            codec oracles, B0-B4, Fourier, D0-D3, bootstrap
pilot/run_video_visionts_carrier_signal.py        C0-C4 runner (codec stage)
pilot/run_video_visionts_carrier_signal_stage2.py signal stage
pilot/summarize_video_visionts_carrier_signal.py  sequential gates and summary
tests/test_video_visionts_carrier_signal.py       31/31 pass, 0 skipped, 0 failures

pilot/results_field/video_visionts_carrier_signal/
  codec/ 2    signal/ 2 (+ compact a/b npz)    gates/    summary.md
```

