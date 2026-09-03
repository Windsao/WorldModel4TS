# Zero-shot loop — summary

Loop is **RESUMABLE, not complete**. No candidate has reached metric-positive.

| candidate | mechanism | key result | verdict |
|---|---|---|---|
| ZL-000 | infrastructure + evidence audit | all six VisionTS anchors verified against artifacts; best strict Q_all = 1.0391 confirmed | done |
| ZL-001 | bounded context-calibrated safe residual | wrapper recovers a lot (0.855x vs its own raw route, 0.960x vs the prior) but pretrained/random = **0.9949**, pretrained/neutral = 0.9970, corr(alpha,gain) ~ 0 | **KILL** (preregistered) |
| ZL-010 | video-native analog retrieval | both datasets: PT beats frame-permuted 0.931 (ETTm2 CI [0.765,0.897] excludes 1) but NOT random-init 0.997; raw L2 far better 1.193 | **KILL** |

## What the loop has actually learned
1. **The incumbent strict VideoMAE route is harmful relative to a good context-only prior.**
   Wrapping it as a bounded residual improves it by 14.5%, and the improvement survives with
   random and with zero logits — so the gain comes from the deterministic inversion path.
2. **The temporal axis is genuinely used — this is a new, statistically solid finding.** Frame
   permutation costs 6.9% of retrieval quality in aggregate and 11% on ETTm2, where the paired
   bootstrap CI [0.765, 0.897] excludes 1 at p=1.000. VideoMAE really does read frame order.
   But the pretrained weights are no better than random for this similarity task (0.997).
3. **Raw time-domain L2 retrieval beats the video embedding.** Mean pooling over a scrolling
   video destroys neighbour geometry. This is the single most actionable finding: the next
   experiment must change the representation, not the blending.

## Next
ZL-011 — the one permitted F2 refinement: temporal tokens instead of global mean pooling, middle
layer. If pretrained still loses to raw L2 on both datasets, KILL F2 and move to F3 (deterministic
token kernel), which is the top-ranked open hypothesis because the supervised cross-attention
readout already showed that structured tokens carry information mean pooling discards.

**NO QUALIFIED POSITIVE RESULT YET — LOOP STATE SAVED FOR RESUMPTION**

## Update — Family 3 (token kernel) is the first mechanism to work

| candidate | change | candidate/base | pt/random | verdict |
|---|---|---:|---:|---|
| ZL-020 | period-matrix column tokens, deterministic cosine kernel | — | **0.958** (pre-decoder) | PROMOTE |
| ZL-021 | + safe residual (pseudo-origin alpha) | 0.944 | 0.989 | REFINE |
| ZL-022 | query m=1 | 0.956 | 1.017 | KILL, revert to m=2 |
| ZL-023 | + agreement shrinkage (parameter-free) | 0.918 | 0.990 | REFINE |
| ZL-024 | + dense origins (144/620) | **0.9242** | 0.992 | REFINE |

### The central discovery
`pt_col / pt_meanpool = 0.5979`. **Global mean pooling was destroying the signal.** Every
previous zero-shot attempt in this project pooled, which is why they all measured zero while the
supervised cross-attention readout — which keeps token structure — worked. Keeping per-period
column tokens recovers a 40% better neighbour geometry.

### Where it stands
- The method beats the strongest context-only prior by **7.6%** with paired CIs excluding 1 on
  both datasets. That is a real, strict, zero-shot forecasting gain.
- Attribution is solid at the representation level (pt/random 0.973, pt/raw-kernel 0.942).
- It is **not backbone-positive**: at forecast level pt/random is 0.992, short of 0.98, because
  the safety shrinkage (rho ~ 0.47) halves the video contribution.
- Nothing has been run on the six-dataset benchmark.

**NO QUALIFIED POSITIVE RESULT YET — LOOP STATE SAVED FOR RESUMPTION**

## Update — all four families closed

| candidate | mechanism | verdict | decisive number |
|---|---|---|---|
| ZL-030 | kernel in prior-residual space (no shrinkage) | KILL | 12-16% WORSE than the prior; pt/rand 0.9985 |
| ZL-031 | oracle ceiling of a perfect blend | KILL | **oracle pt/rand = 1.0006** — the pretrained weights are worth nothing even under an oracle decoder |
| ZL-040 | genuine temporal tubelet masking | KILL | native pt/zero = **1.086** — worse than emitting the neutral cube |

### The unifying failure signature
VideoMAE features carry structure that correlates with time-series **similarity** — better
neighbour ranking than random, statistically solid (pre-decoder pt/random 0.973, CI excludes 1;
frame permutation costs 11%) — but not with time-series **continuation**. Every family died at
the same joint: the ranking never converts into values. ZL-031 is the sharpest statement of it:
an ORACLE per-origin blend, allowed to see the target, extracts **−0.06%** from the pretrained
weights relative to an identical random network, while the retrieval structure itself is worth
20% over the prior.

### Integrity note
ZL-040 initially showed a 2-3.5x gain over the prior. The mandatory suspicious-gain audit found
that its decode path de-normalized masked cubes using statistics from a video containing the true
future. Zero logits — containing no model output at all — also "beat" the prior 2.3x, which is
the signature. Those forecast numbers were voided; only the leak-free native metrics were kept.

**NO QUALIFIED POSITIVE RESULT YET — LOOP STATE SAVED FOR RESUMPTION**

## POSITIVE RESULT — ZL-051 (metric-positive, strict zero-shot)

**Q_all = 0.9152 against VisionTS** on benchmark test windows at identical origins; 4 wins with
paired CI excluding 1, 2 ties, 0 losses; **zero trained parameters**; leak audit clean.

| | ETTh1 | ETTh2 | ETTm2 | electricity | traffic | solar |
|---|---|---|---|---|---|---|
| blend / VisionTS | 1.0075 | 0.9651 | 0.7923 | 0.9906 | 0.9669 | 0.7962 |
| verdict | tie | wins | wins | tie | wins | wins |

The method: four context-only priors combined by evidence-scaled weights
 over dense pseudo-origins. No temperature, no tuned constant.

### The three corrections that produced it
1. argmin prior selection lost to a fixed smean on 5 of 6 datasets — selection is the
   highest-variance functional of a noisy estimate.
2. plain averaging fixed ETT and broke electricity/traffic/solar — unit-temperature weights are
   far too flat where one prior dominates by ~6x.
3. densifying the pseudo-origins and scaling weight sharpness by the amount of evidence repaired
   both regimes at once (0.7542 of smean on historical windows).

### The video branch, measured against this stronger base
- bolted on: Q = 1.0156 (worse)
- as a scored fifth candidate with honest re-embedding: 1.0246 / 1.0339 on ETTh1 / ETTh2
- pretrained / random-init on the benchmark: 1.0288

It earns 15-27% of the blend weight from pseudo-origin evidence and is 24-31% worse at the real
origin — the same **ranking-does-not-become-values** signature that closed families F1-F4.
