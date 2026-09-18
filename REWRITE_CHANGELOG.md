# Rewrite changelog

Rewrite of the ICLR submission in `paper/` following `CLAUDE_ICLR_WORLD_MODEL_REWRITE.md`.
Previous version: commit `e2d0ac4` (title "World Models Are Better Time-Series Learners:
Forecast-Aligned Pretraining of a Video Backbone"). Status 2026-09-17.

## 1. What the paper now argues

Before: a diagnosis paper. The pixel objective decouples from forecasting; we add a value-space
loss; it fixes the curve; we are SOTA. The contribution list led with the negative result and the
title asserted "the objective, not the architecture".

After: a transfer paper. Video pretraining supplies a prior worth having under a small time-series
adaptation budget; realising it needs a framework built around the forecasting task (period-aligned
rendering, future-block prediction, differentiable numeric read-out, scale alignment); the result is
a competitive zero-shot forecaster trained on 654M observations with 60k optimizer updates. The
negative-transfer experiment is the reason the framework is needed, and now sits in
§\ref{sec:diagnosis} as the fourth experimental question rather than as the first contribution.

## 2. Section mapping

| Old | New | Why |
|---|---|---|
| Abstract (diagnosis-led, 2 sentences of pixel-loss numbers) | Abstract (problem → hypothesis → framework → result with audited budget → bounded conclusion) | The brief caps the failure diagnosis at one sentence. |
| §1 Introduction, 4 paragraphs of experiment narrative | §1, 5 paragraphs per the brief: problem, image-transfer precedent + testable hypothesis, why swapping the backbone is not enough, the framework, results + bounded initialisation claim | The old version opened with the failure and read as a lab notebook. |
| §2 Related work | §2, reordered: TSFM first (they set the resource bar we undercut), then visual backbones, then video/world models, then transfer regularisation | Positions the paper against the models it actually competes with; corrects the VisionTS++ objective. |
| §3 Method, a flat list of components | §3 with an Overview paragraph and three named modules (A representation, B prediction and supervision, C scale alignment), plus a stages subsection separating video pretraining / TS continual pretraining / zero-shot evaluation | The brief asks for a framework, not a parameter list, and for the three stages to be named. |
| §4 Experiments ordered by what we tried | §4 ordered by research question: accuracy + budget, initialisation contribution, design contribution, why naive transfer fails, robustness | Removes the "experiment adventure" ordering. |
| §5 Discussion "The objective, not the architecture" | §5 "Two factors, separately supported" + "Where the initialisation advantage stops" | The old framing contradicted the paper's own title and over-attributed. |
| §6 Conclusion | Rewritten, states both bounds explicitly | — |
| Appendix A implementation, B init at two clip lengths, C failed alternatives | A implementation, B per-cell head-to-head, C init at two clip lengths, D inference cost per cell, E alternatives that did not work | Adds the per-cell and inference-cost tables the brief asks for. |

## 3. Claims corrected or narrowed

| Old wording | New wording | Basis |
|---|---|---|
| "1/13 of VisionTS++'s continual-pretraining steps" | corpus size, optimizer updates and sample presentations reported as separate columns; ~1/27 sample presentations, ~0.3% of the LOTSA archive | `CLAIM_EVIDENCE_AUDIT.md` §1. The old ratio mixed step counts with batch sizes. |
| "steps are counted in windows seen" | removed | Ambiguous between micro-batches and optimizer updates. |
| "the lowest six-dataset mean MSE in the published zero-shot table" | "the lowest mean MSE among the models we compare on this protocol" | No exhaustive survey was done. |
| "mean MAE ties" | "mean MAE is 0.328 against 0.326: we are behind on that metric" | 0.328 > 0.326 and no uncertainty estimate exists. |
| "only VisionTS publishes per-horizon numbers on this protocol" | removed / corrected | VisionTS++ v3 reports per-horizon results; see audit §4. |
| "VisionTS++ leaves the read-out outside the loss" (implied) | explicit statement that VisionTS++ trains a multi-quantile (pinball) objective on the forecast, and that our mismatch result is about the VideoMAE configuration we tested | VisionTS++ arXiv:2508.04379v3 §3.4. |
| "This is the whole of the method's gain" | "in this matched comparison the value-space term accounts for the measured gain" | Scope. |
| "the same architecture, only the initialisation changes" (V-JEPA arms) | size-matched, with an extra predictor, EMA targets and a re-initialised head | audit §2. |
| "18 of 24 cells, all six losses on hourly ETT" | recounted; the losing cells are named individually | The old claim contradicted its own appendix table (ETTm1 H=720). |
| "one checkpoint, one inference rule, single pass" | single checkpoint and single rule, but multi-scale averaging and rollout mean several forward passes on some cells; per-cell counts in the appendix | audit §2. |
| "a prediction-space ensemble appears as a secondary row in Table 1" | removed; the ensemble is discussed in §4.7 and is not part of the reported system | No such row existed. |
| seed variance presented next to the 60k headline | explicitly labelled as 20k seed stability, with no error bar on the 60k number | Both seeds are 20k runs. |

## 4. Figures and tables

- **New Figure 1**: method overview (pure TikZ), showing series → context-only normalisation →
  period-aligned frames → visible past / masked future → backbone → predicted frames → read-out,
  with the training and inference paths distinguished. The brief asked for this to precede the
  diagnosis figure; it now opens the method.
- **Figure 2** (was Figure 1): the objective-mismatch curves, moved into §4.6 and re-captioned to
  state that the two panels differ in clip length as well as objective and are therefore not a
  controlled comparison.
- **Table 1** regenerated by `paper/make_main_table.py`: datasets as column groups with MSE/MAE
  subcolumns, bold best and underlined second best computed from the numbers, model families
  separated, first-place count per model. Our 20k and two-seed rows moved into the text.
- **New Table 3** (budget): corpus size, optimizer updates, effective batch, sample presentations,
  measured wall-clock, one column per system, "not reported" where unknown.
- **New Table 5** (`initbudget_table.tex`): video vs image initialisation at 5k and 20k updates, so
  the catch-up appears next to the advantage rather than in an appendix.
- **New appendix table**: forward passes per evaluation cell.

## 5. Files changed

```
paper/iclr2027_conference.tex          title, packages, section order
paper/sections/1_abstract.tex          rewritten
paper/sections/2_introduction.tex      rewritten
paper/sections/3_related_work.tex      rewritten
paper/sections/4_method.tex            rewritten as a framework with three modules
paper/sections/5_experiments.tex       reorganised by research question
paper/sections/6_analysis.tex          rewritten
paper/sections/7_conclusion.tex        rewritten
paper/sections/appendix.tex            reorganised, two new sections
paper/figures/overview.tex             new
paper/figures/budget_table.tex         new
paper/figures/backbone_table.tex       new
paper/figures/initbudget_table.tex     new
paper/figures/ablation_table.tex       new
paper/figures/main_table.tex           regenerated
paper/make_main_table.py               generator for Table 1
CLAIM_EVIDENCE_AUDIT.md                new
REWRITE_CHANGELOG.md                   new
REMAINING_EVIDENCE_GAPS.md             new
```

No file under `/nyx-storage1` (raw results, checkpoints, logs) was modified, and no training or
evaluation job was started for this rewrite.

---

## 6. Second pass: evidence-first and confident (addendum)

Applied `CLAUDE_ICLR_EVIDENCE_FIRST_ADDENDUM.md` on top of the rewrite above. The first pass had
corrected the paper's factual problems but had over-corrected the prose: several verified results
were stated so defensively that a reader could not tell what was actually established. This pass
changes the register without changing any number.

The reference the addendum names (arXiv:2608.08975) does not resolve, so nothing from it is used or
cited; the addendum is self-contained and was applied on its own terms.

**Claims stated directly instead of hedged.** "A plausible reading is that...", "consistent with this
and does not establish it", "we have no uncertainty estimate that would let us call the difference
immaterial" and similar constructions were removed where they were attached to directly measured
results. Observation, interpretation and generalisation are now separated explicitly: the measured
facts are stated flatly, the drift reading of the objective mismatch is labelled an interpretation
once, and nothing is broadened past the tested budget.

**The headline is the trade-off, not an MSE value.** The abstract and introduction now lead with
"leading mean MSE band at 0.3% of the corpus, 1/27 of the sample presentations and 18.9 GPU-hours"
rather than with a list of isolated numbers, and both say in the same breath that the MSE margin is
small and that mean MAE is behind. The efficiency claim is the substantive one and is labelled as
such.

**Novelty made specific.** Section 2 gains a paragraph that, for each of the three components, names
the nearest existing practice, what changes, and which experiment supports it — including the two
places where we have no isolating experiment (rendering vs frame-based prediction) and where the
component is a recipe choice rather than an ablated contribution (scale alignment).

**Headings describe results.** Empirical subsections are now named for what they show ("Lowest mean
MSE of the models compared, behind on mean MAE"; "18.9 GPU-hours on 0.3% of the archive"; "The
native objective and forecasting accuracy move in opposite directions") rather than for the topic.

**Captions are self-contained.** Each main table caption now states the comparison, the aggregation,
the conditions held fixed, and a one-line takeaway; the initialisation table's caption says in the
caption itself that the two panels are not comparable to each other.

**Limitations split by function.** The three limitations that bound the central claim (MSE-only lead,
the 20k catch-up, the budget units) stay in the discussion beside that claim. The caveats that scope
the work — hourly-ETT losses, the single seed, how the inference rule's thresholds were fixed, corpus
composition and the leakage screen — move to a new Appendix F so they are complete without being
scattered through the argument.

**Blind comprehension audit.** Two fresh-context model readers were given only the compiled PDF, with
no description of the intended story, and asked to extract the research question, the contribution,
the difference from the nearest baseline, the best-supported result with its budget, the
system-versus-initialisation distinction, and the main evidence boundary, each with a page or table
citation. These are model readers, not independent human reviewers, and the exercise is labelled as a
self-audit. Their findings and the resulting fixes are recorded in `CLAIM_EVIDENCE_AUDIT.md` §6.
