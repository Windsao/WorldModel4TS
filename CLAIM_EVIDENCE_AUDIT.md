# Claim-to-evidence audit

Audit of every load-bearing claim in the rewritten ICLR submission (`paper/`). Status 2026-09-17.
Three independent audits were run against the code, the raw result JSONs and the baseline papers'
arXiv HTML (budget/corpus, method implementation, tables/baselines). Their findings are folded in
below and several changed the paper's numbers. Column *Support* is one of: **VERIFIED** (checked
against code, raw results, or the cited paper),
**REPORTED** (taken from our own experiment records, not independently re-derived in this pass),
**NARROWED** (the previous draft claimed more than the evidence carries; the paper now claims less),
**OPEN** (stated conservatively in the paper and listed in `REMAINING_EVIDENCE_GAPS.md`).

Raw results live on the NU CS cluster under `/nyx-storage1/hanliu/wm4ts/` (result JSONs in `lsf/`,
run directories in `route_b/`). Nothing there was modified for this rewrite and no job was launched.

---

## 1. Budget and corpus

| # | Claim in the paper | Where | Support | Evidence and reasoning |
|---|---|---|---|---|
| 1.1 | 60,000 optimizer updates at effective batch 32 | §4.3, Table 3 | VERIFIED | `pretrain_route_b.py:632-653`: the counter increments once per `opt.step()`, after an inner `for _ in range(args.accum)` loop; `--batch 8 --accum 4` = 32 windows per update; single A40, no DDP anywhere in the file (confirmed by `sacct` showing `gres/gpu=1`). The paper no longer says "steps are counted in windows seen". |
| 1.2 | 1.92M sample presentations (60k), 0.64M (20k) | Table 3 | VERIFIED (arithmetic) | $60{,}000\times32$ and $20{,}000\times32$. Stated as *windows shown*, not as compute. |
| 1.3 | VisionTS++ trains 100k steps at batch 512 → 51.2M sample presentations | Table 3 | VERIFIED (source) | VisionTS++ arXiv:2508.04379v3 §4.1 as quoted in the rewrite brief. Their definition of a step and of batch (devices, accumulation) is not restated in the source, so the paper presents the two rows side by side rather than as a single ratio. |
| 1.4 | "≈1/27 of VisionTS++'s sample presentations" | §1, Table 3 caption | NARROWED | Was "1/13 of the steps", which compared our optimizer updates to their steps while ignoring the 16× batch difference. $1.92/51.2 = 0.0375 \approx 1/26.7$. The paper never converts this into compute, FLOPs or speed-up. |
| 1.5 | Corpus of 654.2M observations (621.9M reachable), ≈0.3% of the LOTSA archive | §3.6, Table 3 | VERIFIED | `corpus_v2/meta.json`: 98 subset dirs scanned, 74 contributing, 184,013 series, 654,167,390 observations; a 5% holdout leaves 621.9M reachable, which the paper now reports separately. $654.2\text{M}/231\text{B}=0.283\%$. Sampling is with replacement, so the 60k run draws 1.92M windows ≈ 6.8 passes and the 5k run ≈ 0.57 of a pass. |
| 1.6 | Synthetic series mixed in at probability 0.2 | §3.6 | VERIFIED | `pretrain_route_b.py:370-380`: the draw is per batch element, and synthetic series are generated on the fly and never written to the corpus file, so they sit **on top of** the 654.2M. The paper now states this. |
| 1.7 | Measured training time 18.9 h (60k) and 6.3 h (20k) on one A40 | Table 3 | VERIFIED | From the `t` field of each run's `train_log.jsonl` (67850 s / 22716 s), cross-checked against `sacct`. The 23.5 h Slurm elapsed for the 60k job bundles a downstream evaluation grid and is *not* what we report. |
| 1.8 | No ETT data is in the corpus; Monash `weather` and `traffic_hourly` are | §3.6, §5 | VERIFIED, and materially revised | The corpus actually used (`corpus_v2`) was built with `--no-domain-exclude`, so the domain filter described in the module docstring was **not** applied: it contains Monash `weather` (w=0.048; Australian daily, a different source from the 10-minute Jena LSF benchmark), `traffic_hourly` (w=0.028; the same SF freeway source as the LSF traffic benchmark), `PEMS_*`, `LOOP_SEATTLE`, `lcl` and `london_smart_meters` (the largest single weight). No ETT data of any kind is present. The paper now states all of this in §3.6 and §5 instead of claiming domain exclusion. |

## 2. Main results

| # | Claim | Where | Support | Evidence |
|---|---|---|---|---|
| 2.1 | Six-dataset four-horizon mean MSE 0.285 / MAE 0.328 | Table 1 | VERIFIED | Recomputed from the raw JSONs of the 60k seed-0 run (tag `st_f32_savl_l2sp_v2_60k`) by summing `se`/`ae`/`cnt` across origin-range parts: 0.2852/0.3279. All 24 cells are present and complete (origin counts check out against `cnt/(channels×H)`). Two printed values round differently (ETTh2 H192 0.3335, ETTh1 avg 0.4045); the paper now prints 0.334 and 0.405. |
| 2.2 | "The lowest mean MSE among the models we compare on this protocol" | §1, §4.2 | NARROWED | Was "the lowest in the published zero-shot table" and "state of the art". No exhaustive survey of the literature was performed; the claim is now explicitly scoped to the models in Table 1. |
| 2.3 | Mean MAE 0.328 vs VisionTS++ large 0.326 | §1, §4.2, §5 | NARROWED | Was "ties". $0.328 > 0.326$; we have no uncertainty estimate, so the paper says we are behind on that metric. |
| 2.4 | Baseline numbers | Table 1 | **CORRECTED — this changed the table** | The previous draft's entire ETTm2 column was mis-assigned. VisionTS++ Table 4's ETTm2 `avg` row is typeset one column to the left from the VisionTS column onward (13 MSE/MAE pairs for 14 method columns, trailing blank), so every value we copied belonged to the *next* method: our "VisionTS 0.318/0.366" was in fact Time-MoE-small's. Correct VisionTS ETTm2 is 0.282/0.321 (VisionTS arXiv 2408.17253v4 Table 9), and its six-dataset mean is 0.309/0.345, not 0.315/0.352. Every affected baseline's ETTm2 and Avg was recomputed and cross-checked against each source's own Average row; the fix is applied in `paper/make_main_table.py` and in `pilot/final_bigtable.py` so regenerating cannot reintroduce it. No ranking changes: our margin over VisionTS is 7.8%, not 9.5%. |
| 2.5 | "18 of 24 cells against VisionTS, all six losses on hourly ETT" | §4.2, App. B | **CORRECTED** | The count 18/24 is right (and it is also 18/24 on MAE), but five of the six losses are hourly-ETT cells and the sixth is ETTm1 at $H{=}720$; the old wording contradicted the paper's own table. Both are now stated per cell. |
| 2.5b | Comparison against VisionTS++ per cell | §4.2, App. B | **ADDED** | VisionTS++ v3 publishes per-horizon results, so the head-to-head is now reported: we win 17 of 24 on MSE but only 11 of 24 on MAE, with the MAE losses concentrated on weather. This is unfavourable and is stated in the main text and in the limitations. |
| 2.6 | A prediction-space two-seed ensemble appears as a secondary row of Table 1 | previously App. E | REMOVED | No such row existed in the table. The ensemble is now discussed in §4.7 and in the appendix and is explicitly not part of the reported system. |
| 2.7 | Seed variance | §4.7, App. F | **RESOLVED** | Both seeds of the 60k model now have all 24 cells: 0.2852/0.3279 (seed 0) and 0.2838/0.3272 (seed 1), a spread of 0.0014 MSE. Table 1 reports seed 0, the weaker of the two, so the headline is the conservative number; the paper states both and does not average them, so that every reported cell comes from one checkpoint. |

## 3. Method and implementation

| # | Claim | Where | Support | Evidence |
|---|---|---|---|---|
| 3.1 | A clip of $F$ frames covers $FkP$ observations | §3.3 | VERIFIED | The previous draft wrote $L = 16kP$, which was correct only for the 16-frame configuration; the main model uses $F=32$. The sampler draws $F\cdot k$ periods and renders $k$ per frame. |
| 3.2 | The read-out is differentiable and inside the training loss | §3.4 | VERIFIED | The value term is computed from the model's predicted tokens inside the loss function, on the same graph as the pixel term; it is not a post-hoc decode. |
| 3.3 | The value term is "an MSE in exactly the units the benchmark reports" | previously §3.3 | NARROWED | Each window is standardised by its own visible-context statistics, while the benchmark standardises with a scaler fitted on the target dataset's training split. The two differ by a per-window ratio of standard deviations. The paper now says the term is an error on values rather than on pixels, and explicitly not numerically identical to the benchmark metric. |
| 3.4 | Grid lines and read-out invertibility | §3.4 | **VERIFIED, reason corrected** | Grid lines contribute *exactly zero*, not a cancelling constant offset: they are drawn at the clamp's upper knee, so `soft` is 0 there, identically to background. A numerical check over all 224 boundary rows gives max decode error 0.0 on exact renders. The number of visible grid rows does vary with column height, so an "offset cancels" argument would have been wrong. The paper says the thresholds make grid and background contribute zero. Caveat now worth knowing: on the model's blurry predictions a grid line rendered at 0.75 instead of 0.80 induces a height-dependent bias up to ~0.02 z at the bottom of the frame. |
| 3.4b | Gradient actually flows through the read-out | §3.4 | VERIFIED | `column_heights(future_frames_from_tokens(pred_tokens, hp))` runs on the predicted logits with grad enabled, targets under `no_grad`. The `clamp(0,1)` means gradient reaches only pixels whose predicted grey lies strictly between the fill and grid levels, i.e. the boundary band of each column. |
| 3.5 | "Removing the value-space loss recovers the native pixel objective" | previously §4.3 | **CORRECTED** | It does not. With the value weight at zero the objective is still raw-pixel (`norm_pix_loss=False`, un-normalised targets) MSE under a 70/30 future-block / 75%-tube mixture with the forecast loss restricted to future tokens; native VideoMAE uses per-cube normalised targets and 90% tube masking. The ablation isolates the value-space term only, which is what the paper now says. |
| 3.5b | Masking description | §3.4 | **CORRECTED** | `hp` counts frames (1–2 tubelets), not tubelets; tube-masked batches take the pixel loss on *all* masked tokens and carry no value term; the lead mask hides 1–10 tubelets drawn uniformly, not "the leading tubelets". |
| 3.6 | "One checkpoint, one inference rule" | §3.7, §4.1 | VERIFIED but qualified | The rule is a closed-form function of $(P,H)$ containing no dataset identity, and the result filenames confirm all 24 main-table cells were produced by it, including electricity. Two qualifications are now in the paper: it is not single-pass (6 of 24 cells use 2–3 forward passes; Appendix D lists them), and the rule's two thresholds were chosen on development runs over four of the six datasets, so it was not selected blind to the benchmark. |
| 3.7 | "We change only the initialisation" (backbone study) | §4.4 | **CORRECTED — table restructured** | Across the VideoMAE and V-JEPA families this was false, not merely imprecise: the V-JEPA arms use a different trainer (added predictor, EMA target encoder, an extra latent smooth-L1 term, no tube-masked batches), a larger backbone, and a freshly initialised read-out head. Table 5 is now split into two panels: the upper panel (ImageNet MAE-B vs VideoMAE-B) is a genuine initialisation-only contrast, and so is pretrained vs random V-JEPA 2-L within the lower panel; across panels the comparison is system-level and the paper says so. |
| 3.8 | VideoMAE is a world model | previously title/§1 | REMOVED | VideoMAE is a masked video autoencoder. The paper now says our *training stage* is world-model-style in one specific sense (observed past → predicted future) and disclaims action conditioning, planning and physical reasoning, none of which we test. |

## 4. Positioning against baselines

| # | Claim | Where | Support | Evidence |
|---|---|---|---|---|
| 4.1 | VisionTS++ optimises pixel reconstruction | previously §2/§5 | CORRECTED | VisionTS++ arXiv:2508.04379v3 §3.4 reports a multi-quantile (pinball) training objective on the forecast. §2 now states this explicitly and confines our objective-mismatch finding to the VideoMAE configuration we tested. |
| 4.2 | "Only VisionTS publishes per-horizon numbers on this protocol" | previously §4.3 | REMOVED | VisionTS++ v3 reports per-horizon results (Appendix C.2 / Table 4 per the rewrite brief). The paper no longer gives this as a reason. Adding a per-horizon comparison against VisionTS++ requires transcribing their table and confirming protocol identity: gap G1. |
| 4.3 | The image-initialised 5k arm stands in for VisionTS++ | previously implied | REMOVED | Stated in §4.4 and §5 as an initialisation control only. |
| 4.4 | Video initialisation beats image initialisation | §1, §4.4, §5 | NARROWED | True at a 5k-update budget (3–5% at every horizon). At 20k the image initialisation is level or slightly ahead at $H{=}720$. Both budgets are now in the same table, and the claim is adaptation efficiency, not a higher ceiling. |

## 5. The objective-mismatch result

| # | Claim | Where | Support | Evidence |
|---|---|---|---|---|
| 5.1 | Pixel loss falls 0.0366 → 0.0312 while forecast error rises 0.339 → 0.381 | §4.6, Fig. 2 | REPORTED | Four checkpoints evaluated on held-out pixel loss; three evaluated downstream. The 60k downstream point was never run and is shown as missing. |
| 5.2 | "Monotonic degradation with scale" | §4.6 | NARROWED | Described as a trend over the checkpoints we measured, not as a scaling law. |
| 5.3 | Figure 2(a) vs 2(b) is an objective comparison | Fig. 2 caption, §4.6 | NARROWED | Panel (a) is 16-frame, panel (b) is 32-frame, so they differ in two ways. The caption and the text both say so and point at Table 4 for the matched comparison. |
| 5.4 | Weight-norm shrinkage explains the degradation | App. E | NARROWED | Reported as the one quantity that moves monotonically, explicitly not isolated as the cause. |

## 6. Blind comprehension audit

Two fresh-context model readers were given only the compiled PDF, with no description of the intended
story, and asked to extract the research question, contribution, difference from the nearest
baseline, best-supported result with its budget, the system-versus-initialisation distinction, and
the main evidence boundary, each with a page or table citation. They are model readers, not
independent human reviewers; this is a self-audit and is labelled as one. Their findings, classified:

**Factual inconsistencies in the manuscript (fixed).**
1. *The paper violated its own panel rule.* Table 5's caption forbids comparing across its two
   panels, and then §4.4 and §1 both did exactly that, using the random-init V-JEPA arm against the
   ImageNet ViT-B arm to argue "the effect is not capacity" — the single piece of evidence for that
   claim. Now the random-init control is stated within its own panel (a $14\%$ gap from the weights
   alone, same trainer and architecture), and the cross-panel observation is marked as suggestive
   rather than controlled.
2. *Comparator switching inside one sentence.* §4.2 quoted $0.181$ (VisionTS++ large) and $0.222$
   (VisionTS++ base) as if from one baseline row. Both rows are now named.
3. *Terminology.* The title says "world-model transfer" while §2 disclaimed every strong sense of the
   term, so the title asserted what the body retracted. The operational definition now appears in the
   abstract at first use, and §2 and §3.4 scope that definition instead of retracting it.

**A genuine evidence gap (documented, not invented away).**
The initialisation study runs at 5k and 20k updates; the reported system is 60k, past the crossover
where the image initialisation caught up, and no 60k image-initialised counterpart was trained. The
paper now says in the introduction, the experiments and the discussion that it does not claim the
headline number requires a video initialisation. Recorded as G6 in `REMAINING_EVIDENCE_GAPS.md` with
the minimal experiment that would close it.

**Already disclosed, reader confirmed it was findable.** The inference rule's thresholds were chosen
on development runs over four of the six evaluation datasets; the leakage screen covered an earlier
build of the corpus. Both were already in the paper; the first has been promoted from the appendix
into the main limitations because it qualifies the zero-shot framing.

**Stale reading, no action.** One reader reported that the headline came from a single seed with no
error bar. That was true of the build they read; the second seed's evaluation completed during the
rewrite and both seeds are now reported.

### 6b. Method-specification findings from the second blind reader

The second reader attempted to reimplement Section 3 from the paper alone and could not. Every gap
below was checked against the code and the result files before being fixed; one of them was a factual
error in the paper's own appendix table.

1. **Appendix D's inference table was wrong for electricity.** It listed all four electricity cells as
   single-pass `multiperiod`. The result files show electricity follows the same $P{=}24$ pattern as
   the ETTh sets: `sc124`/`sc12`/`multiperiod`/`rollout2`. The table is corrected and the main-text
   count changes from "6 of 24 cells use more than one pass" to **9 of 24**. The reader found this by
   noticing that the table contradicted the rule as the paper had described it.
2. **The inference rule was described imprecisely, which made it look self-contradictory.** The paper
   stated a "$kP \le 96$" threshold as if it constrained every frame; in the implementation it
   constrains only which additional scales enter the multi-scale average, while the base
   $k = \lceil H/4P \rceil$ may exceed it. Section 3.5 now gives the rule explicitly: $h_p$, $k$, the
   visible-context frame count, the scale-admissibility condition and the rollout condition, with
   $h_p$ shown to stay inside the trained set $\{2,4\}$.
3. **The rule depends on the context length $L$, not only on $(P,H)$.** Stated.
4. **The rendering description was wrong for $k>1$.** "The horizontal axis is phase inside the period"
   holds only at one period per frame; with $k$ periods the frame carries $kP$ consecutive samples
   laid side by side. Corrected, with the column-to-sample partition stated so the mapping is visibly
   invertible.
5. **Clip length contradicted itself.** Section 3.2 said $FkP$ observations per clip, Appendix A said
   $16kP$ — a leftover from the 16-frame configuration. Appendix A now says $FkP$.
6. **The masking schedule was incomplete.** The lead-mask count, the tube-mask ratio, and which loss
   applies to which batch type are now given in both Section 3.3 and Appendix A.
7. **Two omissions that blocked reimplementation** are now stated: Equation 2 is applied to the mean
   of the three replicated channels, and the inverse transform back to the dataset scale is written
   out.

8. **Symbols and conventions that blocked reimplementation.** $\mathrm{IMG}$, $g_{\mathrm{fill}}$,
   $g_{\mathrm{grid}}$, the row-index origin and $L$ were used before being defined; the per-dataset
   context lengths had been dropped from the paper entirely when Section 4.1 was compressed. All are
   now in Section 3.2 and Appendix A.
9. **The fill convention was stated wrongly, with a measurable consequence.** The paper said the
   column is filled "below the boundary row"; the renderer includes the boundary row. The reader
   computed that the excluded reading would bias every decoded value by $-0.034$ in $z$ — twice the
   rounding error the paper acknowledged. `paper/check_readout.py` confirms it exactly:
   $-0.0336$ under the excluded convention, versus a mean error of $-8\times10^{-6}$ and a maximum of
   $0.0168$ (the $\pm0.5$-row quantisation bound) under the implemented one. The wording is corrected
   and the measured numbers are now in the paper.
10. **One finding did not survive checking.** The reader reasoned that if grid lines were painted over
    the fill, decoded values would carry an error growing to $0.22\sigma$ at the extremes. The
    renderer draws grid lines on the background only (`(row % 28 == 0) & ~fill`), so the effect does
    not occur. The paper now states the draw order in the method section rather than only in the
    appendix, since that omission is what made the worst case readable into the text, and it reports
    the counterfactual magnitude so the reader can see why the order matters.
11. **Loss support.** Both terms are taken over the masked future tokens only; this was not stated and
    now is, along with the tube-batch schedule and ratio.
