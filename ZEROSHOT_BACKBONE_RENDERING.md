# Can any preprocessing, or any use of the backbone, make VideoMAE pay off? (ZL-060 / ZL-080 / ZL-100)

**Answer: no.** The pretrained weights are demonstrably doing something real — they beat an
identical untrained network by 34% with a paired CI excluding 1 — and they are still strictly
dominated by a normalised-L2 similarity, so adding them to the best video-free method makes it
worse. Both facts are established below on the same windows.

---

## 1. The omission this started from

Every feature-route experiment in this repository — families F1–F4 and the whole ZL-020 series —
fed VideoMAE **a static image repeated over all 16 frames**. Measured inter-frame difference:
**exactly 0.0000**. The backbone's only pretrained advantage is temporal, and the frame-order
audit had already shown it *does* read the frame axis (permuting frames costs 11%, CI excluding
1). So the null might have been a rendering artefact rather than a fact about the representation.

`pilot/video_renderers_motion.py` adds five renderings in which frame *f* is period *f*. Each is
scored by an identical in-context ridge readout, and the decision statistic is
**pretrained / random-init** of the same architecture through the same code path.

## 2. The renderer effect is real per dataset and does not replicate

| renderer | value coded as | inter-frame motion | electricity | traffic | ETTm2 | ETTh2 | solar | geo mean |
|---|---|---|---|---|---|---|---|---|
| `static_matrix` | intensity | 0.0000 | 1.1396 | 1.0983 ! | 1.0041 | 0.9106 | – | **1.0343** (4ds) |
| `growing_matrix` | intensity | 0.0319 | 1.1570 | – | 0.5197 | – | – | **0.7754** (2ds) |
| `period_bar` | area | 0.3937 | 1.0078 | – | 0.5997 | – | – | **0.7774** (2ds) |
| `period_dot` | **position** | 0.0193 | 0.7721 | 0.8744 \* | 0.8450 | 1.4742 ! | – | **0.9576** (4ds) |
| `period_line` | **position** | 0.0259 | 0.7276 | 0.8953 \* | 1.0273 | 1.2374 ! | – | **0.9539** (4ds) |
| `scroll` | **position** | 0.0242 | 0.7443 | 0.8799 \* | 0.7880 | 1.6136 ! | – | **0.9553** (4ds) |

\* paired CI entirely below 1 (pretrained better) · ! CI entirely above 1 (pretrained worse)

What separates the two groups is **not the amount of motion**: `period_bar` has by far the
largest inter-frame difference (0.3937) and shows nothing, while the renderings that do show an
effect encode the value as a **position** — where a mark sits in the frame — rather than as an
intensity or an area.

But it **does not hold up across datasets**. electricity and traffic favour pretrained with CIs
excluding 1; ETTh2 and solar reverse it, ETTh2 significantly. The five-dataset geometric mean for
`period_dot` is **0.990** against **1.034** for the static control — a difference far too small to
call a mechanism. On top of that the ridge readout is **1.2–2.4x worse than the video-free prior
blend** under every renderer, so a better ratio here would not have bought a better forecaster.

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

**Preregistered verdict: BACKBONE-POSITIVE = False.** Condition (i) passes decisively; condition
(ii) fails in the wrong direction with a tight CI. Repeated on solar with the static rendering:
pt/rand = 1.0476, **B/A = 1.0356** — same direction.

## 4. Was random-init a fair control?

Checked, because "pretrained beats random" is worthless if random is degenerate
(`pilot/diag_embedding_health.py`, electricity, 112 windows):

| arm | spread | effective rank | mean norm |
|---|---|---|---|
| pretrained | 0.1359 | **16.98** | 4.00 |
| random-init | 0.1767 | **7.07** | 4.00 |

Random-init is **not** collapsed — its spread is higher and its norm identical. It simply uses
**2.4x fewer effective directions**. That richness is exactly what the pretrained advantage over
random reflects, and it is also why that advantage does not survive contact with raw-L2: more
directions is not the same as more forecasting-relevant directions.

## 5. What survives

- A confound worth recording: every earlier "the pretrained weights are worth nothing" verdict
  was measured on a rendering that suppresses the pretrained advantage. Those margins were much
  larger than the confound, so the negative survives — but "we used the period-matrix image" is
  no longer an adequate description of the input.
- A sharper statement of the negative: the failure is not that VideoMAE features are
  unstructured. They are *more* structured than random by a wide, significant margin. The failure
  is that the structure is not the structure forecasting needs, and a two-line L2 baseline
  captures more of it.

Code: `pilot/video_renderers_motion.py`, `pilot/run_zl080_renderer_sweep.py`,
`pilot/run_zl100_motion_crosschannel.py`, `pilot/diag_embedding_health.py`.
Results: `pilot/results_field/zeroshot_loop/{zl060,zl080,rep/zl080,zl100}/`.
