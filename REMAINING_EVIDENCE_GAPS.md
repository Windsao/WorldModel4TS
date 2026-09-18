# Remaining evidence gaps

Only items that affect a conclusion in the paper. Each says whether it needs a *source or code
check* (cheap, no compute) or a *new experiment* (and if so, the minimal version). Nothing here has been
started; the paper states each of these conservatively today. Closed since the first pass and no
longer listed: the second 60k seed, whose evaluation finished at all 24 cells (0.2838/0.3272 against
seed 0's 0.2852/0.3279 — the paper reports the weaker seed), and three items resolved by the audits: the VisionTS ETTm2 inconsistency (resolved
and corrected in the tables), the availability of VisionTS++ per-horizon numbers (now used), and the
synthetic-share and step-semantics questions about our own runs (both settled from the code).

---

## Needs a source or code check only

**G1. Leakage screen against the corpus actually used. (Most important open item.)**
The exact-match anchor screen (`pilot/audit_paper_corpus_overlap.py`) reports zero benchmark matches,
but it was run against a corpus of 278.2M observations — the earlier, domain-*excluded* build — while
every model in this paper was trained on `corpus_v2` (654.2M), which was built with
`--no-domain-exclude` and re-admits Monash `weather`, `traffic_hourly`, `PEMS_*`, `LOOP_SEATTLE`,
`lcl` and `london_smart_meters`. No ETT data is present in either build, so the ETT results are not
at risk; the exposure is weather (a different source, but the same name) and anything traffic-derived
(which is why traffic appears nowhere in this paper).
*To close:* rerun the existing script with `--corpus /nyx-storage1/hanliu/wm4ts/route_b/corpus_v2`.
This is an evaluation job on existing data, not a new experiment; it was not run here because the
audit was read-only. The script's own caveat still applies: it cannot detect rescaled, aggregated or
interpolated copies. *Affects:* the strength of the "no evaluation dataset is in the corpus"
statement, which the paper currently makes only for ETT while disclosing the rest.

**G2. Inference-rule thresholds were set with benchmark knowledge.**
The rule is a closed-form function of period and horizon with no dataset identity in it, and all 24
main-table cells were produced by it. But its two thresholds (keep $kP \le 96$; roll out only when
$P<96$) were chosen by comparing configurations on development runs over four of the six datasets,
and the inline justification quotes H=720 numbers. This is disclosed in §3.7 and §5.
*To close:* re-derive the thresholds on validation splits only and confirm the rule is unchanged, or
report the comparison that fixed them. No new training. *Affects:* how strong "no per-dataset tuning"
can be stated.

**G3. Step semantics on the baseline side.**
Ours is settled: the counter is an optimizer update, effective batch 32, single GPU. VisionTS++
reports 100k steps at batch 512 without saying whether that batch spans devices or accumulation, so
the sample-presentation ratio assumes their number is a global batch. *To close:* their released
config, if any. *Affects:* the wording of an already-hedged claim.

**G4. Seeding bug in the V-JEPA 2.1 random-init arm.**
`pilot/pretrain_route_j.py` calls `torch.manual_seed(0)` *after* constructing the encoder and
predictor in `WorldModel21`, so a ViT-B random-init control from that class is not seed-controlled.
The random-init row the paper reports comes from `WorldModel` (V-JEPA 2-L), which seeds correctly, so
no reported number is affected. *To close:* move the seed call before construction if a ViT-B
random-init control is ever run. *Affects:* nothing in the current paper; recorded so it is not
rediscovered later.

**G5. Page limit.**
Everything through the discussion fits in 9 pages with the current ICLR template; the conclusion
runs about eleven lines onto page 10. The 2027 call is not published, so we did not verify what it
allows, and we did not shrink fonts, margins or tables to fit. If a 9-page limit applies, the three
cheapest cuts are: fold the per-horizon summary sentence in §4.2 into the table caption, move the
"what is new here, component by component" paragraph of §2 to the appendix, and drop the second
paragraph of §4.4 to a footnote — together about fifteen lines, none of them load-bearing.

## Needs a new experiment

**G7. Pixel-objective budget curve at 32 frames.**
Figure 2 compares a 16-frame pixel-objective curve with a 32-frame value-objective curve, so the two
panels differ in two respects. The controlled objective comparison exists only at 20k updates
(Table 4). *Minimal version:* two pixel-objective 32-frame runs, at 5k and 60k updates, reusing the
existing 20k point, then the five-dataset three-horizon evaluation. About one day of one GPU.
*Affects:* whether Figure 2 can be presented as a single controlled comparison instead of as two
curves with a caveat.

**G8. The read-out on an image backbone.**
The paper claims the framework makes a video initialisation usable, and separately that the image
initialisation catches up by 20k updates. It does not test whether the value-space read-out would
also improve an image-backbone pipeline such as VisionTS, which is the natural question a reviewer
will ask about how specific the design is to video. *Minimal version:* our objective ported into the
VisionTS codebase, evaluated on the same six-dataset grid. About one day of work plus one training
run. *Affects:* the scope of the method claim, not its correctness.

**G9. A second benchmark.**
All results come from one six-dataset protocol. GIFT-Eval or Monash would test whether the
low-budget result generalises. *Minimal version:* GIFT-Eval zero-shot with the existing checkpoint,
no retraining, roughly two days of evaluation. *Affects:* external validity, stated as a limitation.
