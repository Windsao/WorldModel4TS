# Claim-to-evidence audit

Audit of every load-bearing claim in the rewritten ICLR submission (`paper/`). Status 2026-09-17.
Column *Support* is one of: **VERIFIED** (checked against code, raw results, or the cited paper),
**REPORTED** (taken from our own experiment records, not independently re-derived in this pass),
**NARROWED** (the previous draft claimed more than the evidence carries; the paper now claims less),
**OPEN** (stated conservatively in the paper and listed in `REMAINING_EVIDENCE_GAPS.md`).

Raw results live on the NU CS cluster under `/nyx-storage1/hanliu/wm4ts/` (result JSONs in `lsf/`,
run directories in `route_b/`). Nothing there was modified for this rewrite and no job was launched.

---

## 1. Budget and corpus

| # | Claim in the paper | Where | Support | Evidence and reasoning |
|---|---|---|---|---|
| 1.1 | 60,000 optimizer updates at effective batch 32 | §4.3, Table 3 | REPORTED | `--batch 8 --accum 4` on one A40; the step counter advances once per optimizer step. The paper no longer says "steps are counted in windows seen". |
| 1.2 | 1.92M sample presentations (60k), 0.64M (20k) | Table 3 | VERIFIED (arithmetic) | $60{,}000\times32$ and $20{,}000\times32$. Stated as *windows shown*, not as compute. |
| 1.3 | VisionTS++ trains 100k steps at batch 512 → 51.2M sample presentations | Table 3 | VERIFIED (source) | VisionTS++ arXiv:2508.04379v3 §4.1 as quoted in the rewrite brief. Their definition of a step and of batch (devices, accumulation) is not restated in the source, so the paper presents the two rows side by side rather than as a single ratio. |
| 1.4 | "≈1/27 of VisionTS++'s sample presentations" | §1, Table 3 caption | NARROWED | Was "1/13 of the steps", which compared our optimizer updates to their steps while ignoring the 16× batch difference. $1.92/51.2 = 0.0375 \approx 1/26.7$. The paper never converts this into compute, FLOPs or speed-up. |
| 1.5 | Corpus of 654.2M observations, ≈0.3% of the LOTSA archive | §3.6, Table 3 | REPORTED / VERIFIED (arithmetic) | $654.2\text{M}/231\text{B} = 0.283\%$. The paper says "of the LOTSA archive", not "of the data VisionTS++ consumed": VisionTS++ filters LOTSA, and we did not verify their post-filter volume. |
| 1.6 | 20% synthetic series mixed in | §3.6, App. A | OPEN | Our records describe a 20% synthetic mixture; whether that fraction is over series, windows or per-draw probability, and whether the synthetic data is inside or on top of the 654.2M figure, is not resolved in this pass. The paper therefore says "with synthetic series mixed in during sampling" and gives no synthetic-inclusive total. |
| 1.7 | Measured wall clock ≈20 h (60k) and 6.3 h (20k) on one A40 | Table 3 | REPORTED | From our run records. Reported only for our own runs; the baseline column says "not reported". |
| 1.8 | No evaluation dataset is in the corpus; `traffic_hourly` is | §3.6, §5 | VERIFIED | Corpus subset list checked against the six evaluation datasets in an earlier audit; Monash `traffic_hourly` is the same data as the LSF traffic benchmark, which is why traffic is excluded from all of our tables. |

## 2. Main results

| # | Claim | Where | Support | Evidence |
|---|---|---|---|---|
| 2.1 | Six-dataset four-horizon mean MSE 0.285 / MAE 0.328 | Table 1 | REPORTED | Our 60k seed-0 checkpoint under `--mode auto`, tag `st_f32_savl_l2sp_v2_60k`, merged over origin-range parts. |
| 2.2 | "The lowest mean MSE among the models we compare on this protocol" | §1, §4.2 | NARROWED | Was "the lowest in the published zero-shot table" and "state of the art". No exhaustive survey of the literature was performed; the claim is now explicitly scoped to the models in Table 1. |
| 2.3 | Mean MAE 0.328 vs VisionTS++ large 0.326 | §1, §4.2, §5 | NARROWED | Was "ties". $0.328 > 0.326$; we have no uncertainty estimate, so the paper says we are behind on that metric. |
| 2.4 | Baseline numbers quoted from VisionTS++ Table 4 and VisionTS | Table 1 | REPORTED | Not re-derived. See gap G3: the VisionTS ETTm2 average in our table (0.318) is inconsistent with the per-horizon VisionTS values we use in the appendix (mean 0.2815). Until that is resolved the appendix table and the main table may not be citing the same source revision. |
| 2.5 | "18 of 24 cells against VisionTS, all six losses on hourly ETT" | previously §4.3 | NARROWED / REMOVED | The old claim contradicted its own appendix table, which lists ETTm1 at $H{=}720$ as a loss. The main text no longer summarises the count; the appendix table shows every cell and bolds the winner, so the reader counts from the data. |
| 2.6 | A prediction-space two-seed ensemble appears as a secondary row of Table 1 | previously App. E | REMOVED | No such row existed in the table. The ensemble is now discussed in §4.7 and in the appendix and is explicitly not part of the reported system. |
| 2.7 | Seed variance | §4.7 | NARROWED | Both seeds are 20k runs. The paper labels this as seed stability of the 20k configuration and states that the 60k headline number carries no error bar. |

## 3. Method and implementation

| # | Claim | Where | Support | Evidence |
|---|---|---|---|---|
| 3.1 | A clip of $F$ frames covers $FkP$ observations | §3.3 | VERIFIED | The previous draft wrote $L = 16kP$, which was correct only for the 16-frame configuration; the main model uses $F=32$. The sampler draws $F\cdot k$ periods and renders $k$ per frame. |
| 3.2 | The read-out is differentiable and inside the training loss | §3.4 | VERIFIED | The value term is computed from the model's predicted tokens inside the loss function, on the same graph as the pixel term; it is not a post-hoc decode. |
| 3.3 | The value term is "an MSE in exactly the units the benchmark reports" | previously §3.3 | NARROWED | Each window is standardised by its own visible-context statistics, while the benchmark standardises with a scaler fitted on the target dataset's training split. The two differ by a per-window ratio of standard deviations. The paper now says the term is an error on values rather than on pixels, and explicitly not numerically identical to the benchmark metric. |
| 3.4 | Grid lines contribute a bounded offset to the read-out | §3.4 | OPEN | The thresholds make background and grid pixels contribute zero *by construction of the soft occupancy*, but we did not verify algebraically that the offset is independent of the column height for every height. The paper claims only that the read-out inverts the drawing rule "up to rounding". |
| 3.5 | "Removing the value-space loss recovers the native pixel objective" | previously §4.3 | NARROWED | The ablation arm keeps our masking, rendering, corpus and schedule and changes only the supervision term, so it isolates the objective, not the whole native VideoMAE configuration. The paper now says "pixel-only supervision" and describes what is held fixed. |
| 3.6 | "One checkpoint, one inference rule" | §3.6, §4.1 | VERIFIED but qualified | True, and it remains the paper's claim. It is *not* single-pass: multi-scale averaging and rollout cost extra forward passes. Appendix D now lists the configuration and pass count for each of the 24 cells. |
| 3.7 | "We change only the initialisation" (backbone study) | §4.4 | NARROWED | The V-JEPA arms add a predictor, use EMA targets, and re-initialise the predictor output head because the pretrained head maps into a 1664-d teacher space our 768-d targets do not use; V-JEPA 2.1-B was also pretrained at 384px while we render at 224. The table caption and the text now say "size-matched, not architecturally identical". |
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
