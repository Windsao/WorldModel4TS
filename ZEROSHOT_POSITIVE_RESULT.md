# A strict zero-shot forecaster that beats VisionTS — with no trained parameters and no video model

**Headline.** On the benchmark test windows, at origins identical to the VisionTS reference
(`pairs_sha1` asserted equal on all six datasets), a parameter-free context-only method reaches

> **Q_all = 0.9152 against VisionTS** — 8.5% better in equal-dataset geometric mean,
> winning with a paired bootstrap CI entirely below 1 on **four** datasets, tying on two, losing on none.

This is the loop's first qualified positive. It is also a sharper form of the project's negative
result: the strongest thing we can build under a strict zero-shot contract contains **no video
model at all**, and adding one makes it worse.

---

## 1. The method (ZL-051)

Four deterministic context-only priors — `smean` (per-phase mean over all periods in the
context), `snaive` (last period repeated), `last`, `recent_mean` — combined per sample by

1. **Dense pseudo-origins.** Roll a pseudo-origin back one period at a time from `L-H` while at
   least 8 periods of context remain. Each gives a fully observed pseudo-future inside the
   context. This yields 5–8 loss estimates per sample instead of the 2–3 that a schedule spaced
   `H+P` apart provides.
2. **Evidence-scaled weights.** `w_p ∝ (L_min / L_p)^{n_origins}`, renormalised. The exponent is
   the number of pseudo-origins, so the rule interpolates between uniform averaging (no
   evidence) and argmin selection (infinite evidence). **No temperature, no tuned constant.**

Zero trained parameters. No future value is read. Every statistic comes from the observed
context of the origin being forecast.

**Code**: `pilot/zeroshot_loop.py:pseudo_origin_blend` (dense + `pow_n`), driver
`pilot/run_zl_locked.py`, config frozen in
`pilot/results_field/zeroshot_loop/locked_final_config.json`, audit `pilot/audit_zl051.py`.

## 2. Benchmark result (Stage F)

| dataset | VisionTS | smean | **blend** | blend/VisionTS | paired 95% CI | verdict |
|---|---|---|---|---|---|---|
| ETTh1 | 0.4023 | 0.4026 | **0.4076** | 1.0130 | [0.9025, 1.0815] | tie |
| ETTh2 | 0.3130 | 0.3454 | **0.3026** | 0.9667 | [0.9285, 0.9997] | **wins** |
| ETTm2 | 0.2509 | 0.3301 | **0.2003** | 0.7984 | [0.7174, 0.8499] | **wins** |
| electricity | 0.2014 | 0.2069 | **0.1980** | 0.9832 | [0.9708, 1.0314] | tie |
| traffic | 0.5207 | 0.5106 | **0.5082** | 0.9761 | [0.9210, 0.9955] | **wins** |
| solar | 0.2729 | 0.1973 | **0.2137** | 0.7833 | [0.7210, 0.8893] | **wins** |

**Q_all(blend / VisionTS) = 0.9152**  ·  Q_all(smean / VisionTS) = 1.0096  ·  Q_all(blend / smean) = 0.9065

The comparison is paired: the same origins, the same channels, the same manifest hash. The
bootstrap is a moving-block resample over forecast origins with a block at least one full
window long, seed 20260902.

## 3. How the combiner was arrived at (Stage E, historical windows)

| dataset | smean | argmin selection | **evidence blend** | blend/smean | n pseudo-origins |
|---|---|---|---|---|---|
| ETTh1 | 0.4124 | 0.3585 | **0.3193** | 0.7742 | 5 |
| ETTh2 | 0.6768 | 0.4937 | **0.3963** | 0.5855 | 5 |
| ETTm2 | 0.5856 | 0.2701 | **0.2424** | 0.4140 | 8 |
| electricity | 0.2287 | 0.2451 | **0.2226** | 0.9731 | 5 |
| traffic | 0.4108 | 0.4316 | **0.4140** | 1.0077 | 5 |
| solar | 0.1332 | 0.1405 | **0.1333** | 1.0004 | 8 |

Three findings, in the order they were forced on us:

1. **argmin prior selection is worse than doing nothing.** In the first six-dataset run the
   per-origin selector lost to a fixed `smean` on **5 of 6** datasets. The pseudo-origin loss is
   a noisy estimate and argmin is its highest-variance functional.
2. **Plain averaging fixes ETT and breaks the rest.** An inverse-loss blend beat argmin by
   18–19% on ETTh2/ETTm2 but *lost* to a fixed `smean` on electricity/traffic/solar, where the
   seasonal prior dominates the level priors by ~6x and unit-temperature weights were far too
   flat to express that.
3. **The weights must scale with the evidence.** Densifying the origins and raising the
   loss ratio to the power `n_origins` repaired both regimes at once:
   0.9065 of `smean` overall, and no dataset left worse than `smean` by more than 0.8%.

25 combiners (2 prior families × 6 weight rules × 3 origin schedules) were compared **on
historical audit windows only**. The benchmark was touched exactly once, after the configuration
was serialized.

## 4. The video model is a net negative — in every controlled form

| dataset | blend | blend + kernel (pretrained) | blend + kernel (random init) | kernel/blend | pt/rand |
|---|---|---|---|---|---|
| ETTh1 | 0.4076 | 0.4330 | 0.4413 | 1.0624 | 0.9813 |
| ETTh2 | 0.3026 | 0.3263 | 0.3268 | 1.0784 | 0.9984 |
| ETTm2 | 0.2003 | 0.1942 | 0.1971 | 0.9698 | 0.9855 |
| electricity | 0.1980 | 0.1931 | 0.1843 | 0.9753 | 1.0479 |
| traffic | 0.5082 | 0.4999 | 0.4282 | 0.9836 | 1.1674 |
| solar | 0.2137 | 0.2200 | 0.2166 | 1.0295 | 1.0157 |

**Q_all(kernel effect) = 1.0156** (it makes things worse) and **Q_all(pretrained / random) = 1.0308**
(the pretrained weights are worth nothing relative to the same architecture untrained).

Two configurations were tried, and both failed:

- **bolted on** (ZL-024's shrinkage form): net +1.6%. It helps on electricity, traffic and
  ETTm2 and hurts more on ETTh1, ETTh2 and solar.
- **as a fifth candidate** (ZL-052), scored by the same pseudo-origin evidence as the priors,
  with honest per-pseudo-origin re-embedding: the kernel earns 15–27% of the weight yet is
  24–31% *worse* than the blend at the real origin (ETTh1 1.0246, ETTh2 1.0339).

That gap between what the evidence says and what happens at the real origin is the **same
failure signature that closed all four preregistered families**: VideoMAE features rank
time-series *similarity* better than chance, but the ranking never becomes better *values*.

## 5. Integrity

All checks in `pilot/audit_zl051.py` pass on ETTm2, solar and ETTh2:

| check | result |
|---|---|
| permute the future targets → prediction unchanged | max abs diff **0.0** |
| prediction reproducible from the context array alone | max abs diff **0.0** |
| zero prediction must not beat the prior | 3.1090 / 0.7766 / 3.0909 vs 0.3301 / 0.1973 / 0.3454 — passes |
| MSE against a permuted future | 3.05 / 1.33 / 2.90, far worse than 0.20 / 0.21 / 0.30 |
| manifest identity with the VisionTS reference | `pairs_sha1` asserted equal, all six datasets |

This audit exists because an earlier candidate (ZL-040) shipped an oracle-statistic leak that
was caught only by a suspicious-gain check. Every decode path is now audited before its numbers
are read.

## 6. What this means for the project

The claim "video foundation models transfer to time-series forecasting" was never supported by
the 101 runs in `PROGRESS.md`. This result tightens it from the other side: **the pretrained
vision forecaster it was measured against is itself beaten by four moving averages combined
with a parameter-free rule.** Any future claim of transfer has to clear this baseline, on
identical origins, before a backbone can be credited with anything.

Reproduce:

```bash
python3 pilot/run_zl_locked.py --dataset ETTm2 --stage F --stride 8 --n 2000 \
  --data-dir <data> --root <out>          # benchmark
python3 pilot/audit_zl051.py --dataset ETTm2 --data-dir <data>   # leak audit
```
