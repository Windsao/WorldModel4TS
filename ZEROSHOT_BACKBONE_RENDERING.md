# Does any preprocessing make the video backbone pay off? (ZL-080 / ZL-081)

## The omission this started from

Every feature-route experiment in this repository — all of families F1–F4 and the ZL-020 series —
fed VideoMAE **a static image repeated over all 16 frames**. Measured inter-frame difference:
**exactly 0.0000**. The backbone's only pretrained advantage is temporal, and the frame-order
audit had already shown it *does* read the frame axis (permuting frames costs 11%, CI excluding
1). So the null result might have been a rendering artefact rather than a fact about the
representation.

`pilot/video_renderers_motion.py` builds six renderings in which frame *f* is period *f*, so
motion across frames is the evolution of the seasonal shape. Each is scored by an identical
in-context ridge readout (`pilot/run_zl080_renderer_sweep.py`), and the decision statistic is
**pretrained / random-init** — same architecture, same code path, same readout.

## Result: the rendering changes the sign of the pretrained effect

| renderer | value coded as | inter-frame motion | electricity | traffic | ETTm2 | ETTh2 | geo mean |
|---|---|---|---|---|---|---|---|
| `static_matrix` | intensity | 0.0000 | 1.1396 | 1.0983! | 1.0041 | 0.9106 | **1.0343** |
| `growing_matrix` | intensity | 0.0319 | 1.1570 | – | 0.5197 | – | n/a |
| `period_bar` | area | 0.3937 | 1.0078 | – | 0.5997 | – | n/a |
| `period_dot` | **position** | 0.0193 | 0.7721 | 0.8744\* | 0.8450 | 1.4742! | **0.9576** |
| `period_line` | **position** | 0.0259 | 0.7276 | 0.8953\* | 1.0273 | 1.2374! | **0.9539** |
| `scroll` | **position** | 0.0242 | 0.7443 | 0.8799\* | 0.7880 | 1.6136! | **0.9553** |

\* paired bootstrap CI entirely below 1 (pretrained better) · ! CI entirely above 1 (pretrained worse)

**What separates the two groups is not the amount of motion.** `period_bar` has by far the largest
inter-frame difference (0.3937) and shows no pretrained advantage; the three renderings that do
show one have *less* motion but encode the value as a **position** — where a mark sits in the
frame — rather than as an intensity or an area. Position is what Kinetics pretraining preserves;
fine amplitude texture is what it discards.

Aggregated over the four datasets, moving from the static intensity rendering to position-coded
motion moves pretrained/random from **1.034** to about **0.956** — a ~7.5% relative shift that is
attributable purely to preprocessing.

## What this does NOT establish

Three honest limits, stated because the first version of this finding (electricity only) looked
much stronger than it is:

1. **It does not replicate on ETTh2.** There all three position renderings make pretrained
   *significantly worse* than random (1.24–1.61, CIs excluding 1). The effect is real on
   electricity and traffic (CIs excluding 1 in the favourable direction) and present on ETTm2,
   and reversed on ETTh2. It is not a universal law.
2. **The readout is still far worse than doing no learning at all.** `pt/blend` is 1.43–2.15
   everywhere: the in-context ridge on video features loses to the four-prior evidence blend of
   `ZEROSHOT_POSITIVE_RESULT.md` by a wide margin, whichever renderer is used. A better
   pretrained/random ratio on a weak method is not a useful forecaster.
3. **Therefore this is not a backbone-positive result.** The preregistered bar requires the
   pretrained backbone to improve a method that is competitive, not merely to beat its own
   untrained twin.

## Why it still matters

It identifies a **confound that invalidates the strength — though not the direction — of the
earlier negative results**. Every one of the 14 candidates that concluded "the pretrained weights
are worth nothing" measured that on a rendering which suppresses the pretrained advantage by
about 7.5% relative. The negative conclusion survives (the margins were much larger than 7.5%),
but any future claim about video→TS transfer has to control the rendering, and "we used the
period-matrix image" is no longer an adequate description of the input.

Code: `pilot/video_renderers_motion.py`, `pilot/run_zl080_renderer_sweep.py`.
Results: `pilot/results_field/zeroshot_loop/zl080/`, `.../rep/zl080/`.
