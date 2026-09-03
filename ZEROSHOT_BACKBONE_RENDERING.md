# Can any preprocessing, or any use of the backbone, make VideoMAE pay off? (ZL-060 / ZL-080 / ZL-100)

**Answer: no, and the reason is now precise.** The pretrained weights do something real — they
beat an identical untrained network by 34% with a paired CI excluding 1 — and they are still
strictly dominated by a two-line normalised-L2 similarity, so adding them to the best video-free
method makes it *worse*. Preprocessing does not change this: across six datasets, every rendering
tested lands within ±2% of parity with random init.

---

## 1. The omission this started from

Every feature-route experiment in this repository — families F1–F4 and the whole ZL-020 series —
fed VideoMAE **a static image repeated over all 16 frames**. Measured inter-frame difference:
**exactly 0.0000**. The backbone's only pretrained advantage is temporal, and the frame-order
audit had shown it *does* read the frame axis (permuting frames costs 11%, CI excluding 1). So
the null might have been a rendering artefact. `pilot/video_renderers_motion.py` adds five
renderings in which frame *f* is period *f*, scored by an identical in-context ridge readout,
with **pretrained / random-init** of the same architecture as the decision statistic.

## 2. Result: the rendering does not matter in aggregate

| renderer | value coded as | inter-frame motion | electricity | traffic | ETTm2 | ETTh1 | ETTh2 | solar | geo mean |
|---|---|---|---|---|---|---|---|---|---|
| `static_matrix` | intensity | 0.0000 | 1.1396 | 1.0983 ! | 1.0041 | 0.8394 \* | 0.9106 | 1.0032 | **0.9938** |
| `growing_matrix` | intensity | 0.0319 | 1.1570 | – | 0.5197 | – | – | – | (2ds) |
| `period_bar` | area | 0.3937 | 1.0078 | – | 0.5997 | – | – | – | (2ds) |
| `period_dot` | position | 0.0193 | 0.7721 | 0.8744 \* | 0.8450 | 1.1723 | 1.4742 ! | 1.1334 | **1.0187** |
| `period_line` | position | 0.0259 | 0.7276 | 0.8953 \* | 1.0273 | 1.1100 ! | 1.2374 ! | 1.1542 | **1.0099** |
| `scroll` | position | 0.0242 | 0.7443 | 0.8799 \* | 0.7880 | 1.0417 | 1.6136 ! | 1.1229 | **0.9956** |

\* paired CI entirely below 1 (pretrained better) · ! CI entirely above 1 (pretrained worse)

All four fully-replicated renderings sit between **0.9938 and 1.0187** — and the static control is
the *second best* of them. The per-dataset spread is large and pulls in both directions with CIs
excluding 1 on either side (electricity and traffic favour pretrained; ETTh1, ETTh2 and solar
reverse it). That spread is dataset heterogeneity, not a property of the rendering.

**Two hypotheses died here, and the second was mine.** The motion hypothesis died: encoding the
value as a position rather than an intensity changes nothing in aggregate. And the weaker fallback
I briefly kept — "the static rendering was suppressing the pretrained advantage, so earlier
verdicts understated it" — died with it, because the static control scores 0.9938 against 1.0187
for the best motion rendering. There is no confound to correct for; the earlier verdicts were
measured on a rendering that is, in aggregate, no worse than any alternative.

On top of that, the ridge readout is **1.2–2.4x worse than the video-free prior blend** under
every renderer, so a better ratio here would not have bought a better forecaster anyway.

## 3. The decisive test: does the backbone add anything at all? (ZL-100)

Preregistered, on electricity, `period_line`, 120 origins, paired moving-block bootstrap:

| quantity | value | 95% CI | reading |
|---|---|---|---|
| pretrained / random-init | **0.6572** | [0.6249, 0.6871] | **pretraining does something real** |
| pretrained / raw-L2 retrieval | 1.2985 | – | a trivial baseline is better |
| **B / A** — adding the backbone to the best video-free method | **1.0479** | [1.0401, 1.0554] | **it makes that method worse** |
| B / C — pretrained vs random inside that ensemble | 0.8922 | [0.8736, 0.9093] | pretrained is the better of the two additions |

A = prior blend + raw-L2 retrieval (0.1826) · B = A + pretrained retrieval
(0.1913) · C = A + random-init retrieval (0.2145).

**BACKBONE-POSITIVE = False.** Condition (i) passes decisively; condition (ii) fails in the wrong
direction with a tight CI. Repeated on solar with the static rendering: pt/rand = 1.0476,
**B/A = 1.0356** — same direction.

## 4. Was random-init a fair control?

Checked rather than assumed, because "pretrained beats random" is worthless if random is
degenerate (`pilot/diag_embedding_health.py`, electricity, 112 windows):

| arm | spread | effective rank | mean norm |
|---|---|---|---|
| pretrained | 0.1359 | **16.98** | 4.00 |
| random-init | 0.1767 | **7.07** | 4.00 |

Random-init is **not** collapsed — higher spread, identical norm. It simply uses **2.4x fewer
effective directions**. That richness is what the pretrained advantage over random reflects, and
also why it does not survive contact with raw-L2: more directions is not more
forecasting-relevant directions.

## 5. What this leaves

The failure is **not** that VideoMAE features are unstructured. They are more structured than
random by a wide, significant margin. The structure is not the structure forecasting needs, no
rendering tested moves that, and a two-line normalised-L2 similarity captures more of it.

Code: `pilot/video_renderers_motion.py`, `pilot/run_zl080_renderer_sweep.py`,
`pilot/run_zl100_motion_crosschannel.py`, `pilot/diag_embedding_health.py`.
Results: `pilot/results_field/zeroshot_loop/{zl060,zl080,rep/zl080,zl100}/`.
