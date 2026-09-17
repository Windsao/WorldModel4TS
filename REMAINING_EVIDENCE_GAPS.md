# Remaining evidence gaps

Only items that affect a conclusion in the paper. Each says whether it needs a *source or code
check* (cheap, no compute) or a *new experiment* (and if so, the minimal version). Nothing here has
been started; the paper states each of these conservatively today.

---

## Needs a source or code check only

**G1. Per-horizon comparison against VisionTS++.**
The previous draft justified comparing per-horizon only against VisionTS by claiming nobody else
publishes per-horizon numbers on this protocol. VisionTS++ v3 does (Appendix C.2 / Table 4). The
paper no longer makes the false claim, but it also does not yet contain the stronger comparison.
*To close:* transcribe their per-horizon MSE/MAE for the six datasets, confirm their splits, context
lengths and channel handling match ours, and extend Appendix B to three methods. No compute.
*Affects:* the strength of Appendix B, not any headline number.

**G2. VisionTS baseline numbers are internally inconsistent.**
Our main table lists VisionTS ETTm2 four-horizon average MSE as 0.318; the per-horizon VisionTS
values used in Appendix B are 0.228/0.262/0.293/0.343, whose mean is 0.2815. One of the two is from
a different source revision or a different protocol variant. We have not resolved which, and we did
not silently replace either: both are still the numbers the corresponding source gave us.
*To close:* locate each number in the VisionTS and VisionTS++ papers, record the version and table,
and use one consistent source. *Affects:* the size of our margin on ETTm2 in Table 1, and possibly
the first-place count. This is the most important open item in the paper.

**G3. Synthetic-series share of the corpus.**
We report 654.2M observations over 98 LOTSA subsets and say synthetic series are mixed in during
sampling. Whether the 20% figure in our records is a fraction of series, of drawn windows or of
per-sample probability, and whether the synthetic data is counted inside 654.2M, is unresolved.
*To close:* read the corpus builder and the sampler. *Affects:* the precise corpus figure, not the
order of magnitude, and therefore not the ≈0.3% statement.

**G4. Step semantics on both sides.**
Our step counter is an optimizer update with gradient accumulation folded in (batch 8 × accum 4 = 32
windows per update), on a single device. VisionTS++ reports 100k steps at batch 512 without stating
whether that batch spans devices or accumulation. The paper therefore reports the two as separate
rows and never divides one by the other except as "sample presentations", with the caveat in the
caption. *To close:* their released config, if any. *Affects:* the wording of the efficiency claim,
which is already hedged.

**G5. Grid-line offset in the read-out.**
The soft occupancy is constructed so that grid and background pixels contribute zero, but we have
not shown algebraically that the resulting offset is independent of where the column boundary sits.
*To close:* a short numerical check over the height range, comparing `decode(encode(h))` against `h`.
A five-line script, no GPU. *Affects:* the strength of the "inverts the drawing rule" statement in
§3.4, which is currently hedged with "up to rounding".

**G6. Page limit.**
The main text through the conclusion fits in 9 pages with the current ICLR template; we did not
verify what the 2027 call actually allows, since it is not published. No font, margin or table
compression was used to fit.

## Needs a new experiment

**G7. Seed replication of the 60k main model.**
Both existing seeds are 20k runs, so the headline 0.285 has no error bar and the paper says so.
*Minimal version:* one additional 60k run with a different seed, evaluated on the six-dataset grid
(≈20 GPU-hours training plus the evaluation grid). A seed-1 60k run and its evaluation are in
progress in the project but were not complete at the time of writing, so nothing from it is used.

**G8. Pixel-objective budget curve at 32 frames.**
Figure 2 compares a 16-frame pixel-objective curve with a 32-frame value-objective curve, so the two
panels differ in two respects. The controlled objective comparison exists only at 20k updates
(Table 4). *Minimal version:* two pixel-objective 32-frame runs, at 5k and 60k updates, reusing the
existing 20k point, then the five-dataset three-horizon evaluation. About one day of one GPU.
*Affects:* whether Figure 2 can be presented as a single controlled comparison instead of as two
curves with a caveat.

**G9. The read-out on an image backbone.**
The paper claims the framework makes a video initialisation usable, and separately that the image
initialisation catches up by 20k updates. It does not test whether the value-space read-out would
also improve an image-backbone pipeline such as VisionTS, which is the natural question a reviewer
will ask about how specific the design is to video. *Minimal version:* our objective ported into the
VisionTS codebase, evaluated on the same six-dataset grid. About one day of work plus one training
run. *Affects:* the scope of the method claim, not its correctness.

**G10. A second benchmark.**
All results come from one six-dataset protocol. GIFT-Eval or Monash would test whether the
low-budget result generalises. *Minimal version:* GIFT-Eval zero-shot with the existing checkpoint,
no retraining, roughly two days of evaluation. *Affects:* external validity, stated as a limitation.
