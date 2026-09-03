# Does a video foundation model help zero-shot forecasting? Two backbones, four ways of using them.

**No — and the shape of the failure is now precise.** Across two architectures with different
pretraining objectives, four ways of using them, and six datasets, every controlled test lands in
the same place:

> The pretrained weights beat an identical untrained network decisively — **every single time,
> with a paired CI excluding 1**. And in no configuration do they add value to the best method
> that contains no video model.

The representation is real. It is not the representation forecasting needs.

---

## 1. The decisive test

Preregistered, two conditions, both required:

- **(i)** pretrained beats random-init of the identical architecture, paired CI excluding 1
- **(ii)** adding it to the best video-free method **improves** that method, paired CI excluding 1

Condition (ii) is the one that matters. A backbone that only beats its own untrained twin has not
been shown to be useful — it has been shown to be non-random.

| | VideoMAE-base (ZL-100) | V-JEPA 2 ViT-L (ZL-111) | V-JEPA 2 predictor (ZL-120) |
|---|---|---|---|
| how the backbone is used | cross-channel retrieval | cross-channel retrieval | **latent-space rollout** |
| pretrained / random-init | **0.6572** [0.6249, 0.6871] | **0.6995** [0.5508, 0.9113] | **0.6657** [0.6105, 0.7243] |
| condition (i) | PASSES | PASSES | PASSES |
| pretrained / trivial baseline | 1.2985 | 1.0355 | 1.2648 |
| **B / A** | **1.0479** [1.0401, 1.0554] | **1.0078** [0.9702, 1.0442] | **1.0895** [1.0373, 1.1435] |
| condition (ii) | FAILS (worse) | FAILS (tie) | FAILS (worse) |
| **BACKBONE-POSITIVE** | **False** | **False** | **False** |

A = the best video-free method · B = A + the pretrained backbone · C = A + the same backbone
untrained. On every row, B beats C — the pretrained weights are the better of the two additions.
They are simply both worse than not adding anything.

## 2. Why V-JEPA 2, and why its predictor

Neither was a shot in the dark.

**V-JEPA 2** was chosen because VideoMAE-base is a masked *pixel* autoencoder trained with
`norm_pix_loss=True`: its decoder structurally cannot emit absolute values, which is the blocker
documented for this repository's entire generative family. V-JEPA 2 predicts in *representation*
space and has no such blocker. It is a genuine improvement — its B/A confidence interval no longer
excludes 1 in the wrong direction — and it still clears neither condition. **The null is not
specific to the masked-pixel objective.**

**The predictor** was chosen because every candidate before it treated the backbone as a frozen
feature extractor. V-JEPA 2 ships a module trained to produce the representation of masked
spatio-temporal regions from the visible ones: an actual latent-space rollout, the thing this
project is named for. Frame *f* is period *f*, the trailing frames are the target and are never
shown, and an in-context ridge maps the predicted representations to values.

Result: **predictor / encoder-only = 0.9952**. Using the world model as a world
model is worth half a percent over just reading its features.

> **A correction.** A 2-origin, 7-channel smoke test put this ratio at 0.8662 and I reported a
> 13.4% gain from the rollout. At 16 origins and 24 channels it is 0.9952. The smoke-test figure
> was noise; the claim is withdrawn. Both runs are kept so the withdrawal is checkable:
> `pilot/results_field/zeroshot_loop/smoke/zl120/ETTm2.json` (the smoke test) against
> `pilot/results_field/zeroshot_loop/zl120/electricity.json` (at scale).

## 3. Preprocessing does not rescue it

Every earlier feature-route experiment fed VideoMAE **a static image repeated over 16 frames** —
measured inter-frame difference exactly 0.0000 — while the frame-order audit had shown the model
does read the frame axis. Five motion renderings were tested, encoding the value as a position
rather than an intensity. Geometric mean of pretrained/random over six datasets:

| `static_matrix` | `scroll` | `period_line` | `period_dot` |
|---|---|---|---|
| **0.9938** | 0.9956 | 1.0099 | 1.0187 |

All within 2% of parity, with the static control the *second best*. The per-dataset spread is
large and pulls both ways with CIs excluding 1 on either side (electricity 0.77, ETTh2 1.47) —
dataset heterogeneity, not a property of the rendering. Both the motion hypothesis and the weaker
fallback that the static rendering had been masking a real advantage are dead.

## 4. Was random-init a fair control?

Checked rather than assumed (`pilot/diag_embedding_health.py`): random-init is **not** degenerate
— higher spread (0.177 vs 0.136) and identical norm — it simply uses **2.4x fewer effective
directions** (effective rank 7.07 vs 16.98). That richness is exactly what condition (i) measures,
and exactly why it does not survive contact with a normalised-L2 baseline: more directions is not
more forecasting-relevant directions.

## 5. The standing positive result contains no video model

`ZEROSHOT_POSITIVE_RESULT.md`: four context-only priors combined by evidence-scaled weights over
dense pseudo-origins, zero trained parameters, **Q_all = 0.9152 against VisionTS** on benchmark
test windows at identical origins — 4 wins with paired CIs excluding 1, 2 ties, 0 losses.

That is the honest state of this project: the best zero-shot forecaster we can build under a
strict contract beats the pretrained vision forecaster it was measured against, and contains no
video model at all.

---

Ledger: `pilot/results_field/zeroshot_loop/ledger.jsonl` (20 preregistered candidates).
Code: `pilot/run_zl100_motion_crosschannel.py`, `pilot/run_zl110_vjepa.py`,
`pilot/run_zl120_vjepa_predictor.py`, `pilot/run_zl080_renderer_sweep.py`,
`pilot/video_renderers_motion.py`, `pilot/diag_embedding_health.py`.
