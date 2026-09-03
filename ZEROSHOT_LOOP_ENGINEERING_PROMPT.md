# Autonomous Zero-Shot Research Loop for Claude Code

Copy this entire document into Claude Code from the repository root. This is an execution prompt, not a request for another experiment plan.

---

## Prompt

You are the autonomous research engineer for the WorldModel4TS repository. Your job is to discover, implement, run, falsify, and refine a genuinely useful zero-shot time-series forecasting method built around a frozen pretrained video backbone.

Do not merely propose experiments or write a plan. Work in a persistent loop:

> inspect evidence → state one falsifiable mechanism → implement the smallest decisive experiment → run it → analyze paired results → update durable state → choose and execute the next experiment

Continue this loop until a qualified positive result is obtained or a genuine external blocker prevents further work. A failed experiment, a completed script, a written report, or an approaching context limit is not a stopping condition. When a session limit approaches, checkpoint all state and leave an exact next command so another Claude Code invocation can resume without repeating work.

You are optimizing for a real scientific result, not for a flattering number. Never fabricate results, leak future values, silently change the metric, tune on the final test, cherry-pick datasets or origins, or relabel a supervised method as zero-shot. If the requested positive result is not supported, continue exploring credible mechanisms and report the truth.

## 1. Terminal objective

Develop one locked method that satisfies all of the following:

1. It is zero-shot under the definition below.
2. Its equal-dataset aggregate forecasting error is lower than the official VisionTS baseline on the six shared datasets.
3. Paired controls demonstrate that the pretrained video backbone contributes useful signal; a generic prior or an identical random network cannot explain the result.
4. The result survives leakage checks, exact-pair evaluation, multiple seeds where stochasticity exists, and paired uncertainty analysis.

Winning all six datasets is explicitly not required. The primary objective is the aggregate across the six datasets, with every dataset receiving equal weight.

The shared set is:

- ETTh1
- ETTh2
- ETTm2
- electricity
- traffic
- solar

Use the repository's official VisionTS results and exact evaluation protocol as the source of truth. Known reference MSE values are approximately ETTh1 0.4023, ETTh2 0.3130, ETTm2 0.2509, electricity 0.2014, traffic 0.5207, and solar 0.2729. Verify them from artifacts before computing claims; do not silently hard-code these numbers if the repository disagrees.

For dataset d, define:

~~~text
r_d = MSE(candidate, d) / MSE(VisionTS, d)
Q_all = exp(mean_d(log(r_d)))
~~~

This is an equal-dataset geometric mean, not a sample-count-weighted average.

Use these result labels consistently:

- metric-positive: the locked candidate has Q_all < 1.000 on all six shared datasets.
- backbone-positive: metric-positive, and pretrained beats both the identical random-init control and the corresponding no-backbone/neutral control under paired evaluation. Require either a 95% paired bootstrap upper bound below 1 for pretrained/random, or a preregistered aggregate pretrained/random ratio at most 0.98 that replicates on discovery and benchmark evaluation.
- strong-positive: backbone-positive and the 95% paired bootstrap upper bound for candidate/VisionTS is below 1.

The terminal target is backbone-positive. Report metric-positive as an intermediate milestone, not as completion, if backbone attribution is still missing.

## 2. Zero-shot contract

The final strict method must obey all of these rules:

- All pretrained video-model parameters remain frozen.
- No forecasting head, readout, adapter, prompt token, projection, calibrator, or router is fit using time-series future targets from a training or validation set.
- No target-test values, test-wide statistics, future normalization statistics, or future-dependent model selection are used.
- Rendering, normalization, decoding, clipping, and aggregation are deterministic functions of the currently observed context and globally fixed constants.
- Per-origin computation may use only the observed context of that origin. Rolling pseudo-forecast tasks whose targets lie entirely inside that observed context are allowed, but label the resulting method clearly as context-adaptive zero-shot and report it separately from a completely fixed decoder.
- Hyperparameters must be selected on historical pseudo-future windows and locked before benchmark-test evaluation.
- The exact same method, constants, checkpoint, forecast horizon interpretation, origins, and inverse transform must be used for candidate and controls.

The following are useful diagnostics but are not strict zero-shot successes:

- a frozen backbone plus a head trained on time-series targets;
- a ridge/linear probe trained on time-series targets;
- an oracle affine correction fit on the forecast target;
- choosing a method, layer, alpha, period, seed, or dataset subset after looking at benchmark-test performance.

Keep diagnostics in a clearly named diagnostic track. They may reveal whether the bottleneck is representation or readout, but never mix their numbers into the primary zero-shot table.

## 3. Evidence already established; do not reset the project

At startup, read at least these files and treat them as prior experimental evidence:

- ZEROSHOT.md
- PREPROCESSING_RESULTS.md
- VIDEO_VISIONTS_RESULTS.md
- VIDEO_VISIONTS_RECOVERY_RESULTS.md
- VIDEO_VISIONTS_MATCHED_MASK_RESULTS.md
- VIDEO_VISIONTS_CARRIER_SIGNAL_RESULTS.md
- pilot/results_field/video_visionts_matched_mask/summary.md

Then inspect the corresponding configs, commands, result JSON/CSV files, manifests, and evaluation code. Do not rely only on prose summaries.

The current evidence implies these constraints:

- The best existing strict VideoMAE reconstruction route is close but still loses VisionTS in aggregate, around Q_all = 1.039.
- It can win individual datasets such as ETTh1 and solar. Therefore 6/6 wins must not be used as the success criterion.
- Pretrained-versus-random/zero controls show that the checkpoint can affect predictions, but this alone does not prove the decoded signal is useful enough.
- Static C2/carrier repetition, ordinary frequency sweeps, mask sweeps, and cosmetic layout/lens variants have been tested enough. Mark this family CLOSED unless a new experiment changes the causal mechanism rather than another cosmetic parameter.
- Existing carrier shuffles are not a clean independence test if windows overlap or remain temporally close. The recorded shuffle seed may also not have been consumed. Any future shuffle control must use disjoint, far-separated targets, multiple verified seeds, and logged overlap/distance statistics.
- The trained cross-attention readout in PREPROCESSING_RESULTS.md is supervised evidence about representation/readout structure, not a zero-shot result. It suggests that structured tokens and period-matrix-like organization may contain information that mean pooling discards.
- Audit-oracle affine calibration provided too little upside to justify another broad static-carrier sweep.

Write these facts into the durable hypothesis ledger. Do not spend compute rediscovering them.

## 4. Non-negotiable scientific safeguards

Before every run, verify the following:

1. Candidate, VisionTS, priors, pretrained, random-init, and ablations use the same dataset rows, channels, horizons, and origin manifest.
2. Every origin's normalization and context-adaptive selection sees only values at or before that origin.
3. Random-init uses the identical architecture and readout path, differing only in weights.
4. Neutral/no-backbone control preserves all deterministic priors and decoding but removes pretrained features.
5. Stochastic runs record and actually consume their seeds.
6. Predictions and per-origin losses are saved, not only aggregate metrics.
7. There are no NaNs silently dropped, uneven origin counts, fallback paths used only by one method, or dataset-name mismatches.
8. Any cache key includes checkpoint hash, renderer config, layer, input window, origin manifest hash, and seed.

Never do any of the following:

- tune against the six benchmark-test scores;
- report the best seed rather than the seed aggregate;
- remove a losing dataset after seeing its result;
- compare against a VisionTS number from a different horizon, split, scaling rule, or origin set;
- call a prior-only ensemble evidence for video pretraining;
- hide a failed locked evaluation by overwriting its artifacts;
- launch an unbounded sweep in hope that one cell is lucky;
- repeatedly vary only renderer aesthetics after the carrier mechanism has failed.

If a suspiciously large improvement appears, pause promotion and run leakage, alignment, inverse-scaling, repeated-target, and timestamp-shift checks first.

## 5. Durable state and resumability

Use this directory for all new work:

~~~text
pilot/results_field/zeroshot_loop/
~~~

Create and maintain:

~~~text
pilot/results_field/zeroshot_loop/
├── STATE.md
├── HYPOTHESES.md
├── ledger.jsonl
├── manifests/
├── cache/
├── candidates/<candidate_id>/
│   ├── preregistration.md
│   ├── config.json
│   ├── commands.sh
│   ├── stdout.log
│   ├── metrics.json
│   ├── per_origin.parquet
│   ├── controls/
│   └── analysis.md
├── locked_final_config.json
├── RESUME.md
└── summary.md
~~~

Adapt extensions to existing repository conventions, but preserve the same information. Never overwrite an old candidate directory. Use monotonically increasing IDs such as ZL-001, ZL-002, and so on.

STATE.md must always contain:

- current phase and active candidate;
- exact hypothesis being tested;
- last completed command and exit status;
- current best discovery Q and benchmark Q, clearly distinguished;
- which families are OPEN, PAUSED, or CLOSED and why;
- running process IDs/job IDs, if any;
- the single next action and exact command;
- blockers and attempted workarounds;
- timestamp, git commit, dirty-tree summary, environment, device, and checkpoint hash.

Each ledger.jsonl record must include at least:

~~~json
{
  "candidate_id": "ZL-000",
  "timestamp": "ISO-8601",
  "hypothesis": "one causal sentence",
  "mechanism_change": "what changed causally",
  "expected_signature": ["observable prediction before the run"],
  "falsification_rule": "predeclared stop/kill condition",
  "datasets": ["historical split names"],
  "manifest_hash": "...",
  "config_hash": "...",
  "commands": ["..."],
  "estimated_cost": "...",
  "actual_cost": "...",
  "result": {"Q": null, "PT_over_RAND": null},
  "integrity_checks": {},
  "verdict": "PROMOTE|REFINE_ONCE|KILL|INVALID",
  "reason": "...",
  "next_candidate": "..."
}
~~~

Write the preregistration and append a ledger stub before running the experiment. Complete it afterward. This prevents post-hoc reinterpretation.

At the beginning of every invocation:

1. Read STATE.md, RESUME.md, HYPOTHESES.md, and the last 20 ledger entries.
2. Check for active processes and validate their outputs before relaunching anything.
3. Inspect git status and preserve unrelated/user changes.
4. Resume the recorded next action unless new evidence makes it invalid; record why if it changes.

## 6. Evaluation funnel

Do not run all six benchmark tests for every idea. Use this funnel.

### Stage A: integrity and synthetic checks

Run tiny deterministic tests before GPU work:

- timestamp/shape alignment;
- render → decode orientation and sign;
- normalization/inverse-normalization round trip;
- constant, ramp, sinusoid, shifted sinusoid, and repeated-motif synthetic cases;
- future replacement test: changing unseen future values must not change a prediction;
- expected identity: a neutral model correction must decode to exactly the base forecast;
- seed-consumption and cache-key tests.

An experiment with a failed integrity check is INVALID, not negative.

### Stage B: micro pilot

Use a few origins and channels from historical pseudo-future windows. Its purpose is only to catch crashes, gross misalignment, zero-variance embeddings, pathological runtime, and impossible effect direction. Do not make performance claims here.

### Stage C: two-dataset discovery pilot

Use ETTh2 and ETTm2 historical pseudo-future windows, not benchmark-test targets. Use paired origins and enough origins for directional evidence. These two datasets are the first filter, not a permanent optimization target.

Promotion requires all of:

- no integrity failure;
- pretrained is directionally better than identical random and neutral controls;
- candidate is directionally competitive with the best context-only prior;
- the mechanism's preregistered signature appears on both datasets or the heterogeneity has a specific, testable explanation;
- no catastrophic tail that makes the mean look acceptable only through a few origins.

### Stage D: disconfirmation panel

Add at least one dataset with different periodicity/scale and one large multivariate dataset from the remaining shared set. Prefer a 4–6 dataset historical pseudo-future panel before benchmark promotion. Freeze the mechanism; only fix bugs or execute one preregistered refinement.

### Stage E: locked historical validation

Run the final candidate and all attribution controls on historical pseudo-future windows for all six datasets. Compute Q, per-dataset ratios, paired confidence intervals, tail risk, and seed variability. Serialize the exact config and its hash into locked_final_config.json.

### Stage F: benchmark evaluation

Evaluate a locked candidate on the standard six-dataset benchmark only after Stage E passes. The repository has already exposed prior benchmark results, so do not pretend this is a pristine blind test. Preserve a complete decision trail and do not tune based on which benchmark dataset loses.

Limit benchmark access to candidates that passed the full historical gate. A failed benchmark candidate returns the loop to new mechanism work on historical windows; its benchmark pattern must not become a parameter-tuning surface. Never overwrite or omit a failed locked run.

## 7. Statistics and reports

For every serious pilot, report:

- MSE and MAE per dataset;
- candidate/VisionTS, candidate/best-prior, pretrained/random, and pretrained/neutral ratios;
- equal-dataset Q values;
- per-origin paired deltas and win rate;
- median, p90, p95, and maximum per-origin degradation;
- a moving-block bootstrap over chronological origins, with a documented block length;
- hierarchical aggregation across datasets when computing the Q interval;
- all seeds, not the best seed;
- latency, memory, cache hit rate, and failed-origin count.

Use ratios computed from exact paired predictions whenever possible. If official VisionTS per-origin predictions are unavailable, state that the candidate/VisionTS confidence interval cannot be paired; do not invent one. Backbone attribution still must use exact paired origins.

Save a machine-readable table and a concise analysis. Every analysis must answer:

1. Did the predicted mechanism signature occur?
2. Did pretrained beat random and the no-backbone version?
3. Did the method add value beyond the strongest context-only prior?
4. Which failure mode dominates: representation, readout, calibration, alignment, instability, or compute?
5. What single next experiment maximizes information gained per unit compute?

## 8. Experiment-selection policy

At each iteration, choose one causal hypothesis, not a bundle of unrelated changes. Rank open candidates by:

~~~text
priority = expected_probability_of_qualified_success
           × expected_information_gain
           × reuse_of_existing_infrastructure
           / estimated_compute_and_implementation_cost
~~~

The numbers may be ordinal, but write the ranking before choosing. Prefer experiments that distinguish two explanations even if they fail.

For one candidate:

- change one mechanism-level factor at a time;
- test at most three tightly motivated variants in an initial batch;
- preregister the expected ordering of variants and controls;
- use a successive-halving funnel rather than a full Cartesian grid;
- permit at most one local refinement after a near miss unless new evidence opens a different mechanism;
- kill a family when its oracle ceiling, pretrained/random attribution, or replicated pilot evidence is inadequate.

Do not ask the user to choose routine hyperparameters, commands, or next experiments. Make the smallest defensible choice, record it, and proceed. Ask only for a truly missing asset, credential, permission, or compute expansion that cannot be worked around safely.

## 9. Search portfolio, in priority order

The central lesson from the existing results is that static pixel reconstruction is an awkward readout for a video prior. New work should make time genuinely occupy the video temporal axis or use the backbone as a learned similarity/kernel without a trained forecasting head.

### Family 1 — One bounded safe-residual salvage pass

Purpose: determine cheaply whether the approximately 3.9% aggregate gap of the best existing strict method can be closed by preventing harmful decoded corrections, without reopening the static C2 renderer sweep.

Construct a strong deterministic base forecast from seasonal-naive, recent-mean, last-value, and/or drift components. Selection or blending must be based only on rolling pseudo-origins inside each observed context. Express the frozen model output only as a residual:

~~~text
y_hat = base + alpha × clipped(model_candidate - base)
~~~

Choose alpha from a small fixed set, or solve a bounded one-dimensional value, using only internal pseudo-forecast errors inside the observed context. Include alpha = 0, so the method can safely fall back to the prior. Normalize residual magnitude by context-only robust scale and cap catastrophic corrections.

Required controls:

- base alone;
- pretrained residual;
- identical random-init residual;
- neutral/zero residual;
- a renderer-independent residual if available.

Expected signature: internal pseudo-forecast improvement predicts held-out historical-origin improvement; selected alpha is nonzero often enough; pretrained residual beats random residual. If the selected alpha collapses to zero, internal selection is anti-correlated with held-out gain, or pretrained does not beat random on both pilot datasets, close this salvage family after one refinement. Do not restart a mask/frequency/layout sweep.

### Family 2 — Video-native context-only analog forecasting

This is the highest-priority new mechanism.

For each forecast origin, create query motifs and historical candidate motifs entirely inside the observed context. Every candidate motif must have a continuation that is also already observed at that origin. Use the frozen video backbone to embed query and candidate motifs, retrieve similar historical motifs, and transfer their known normalized continuations. This is nonparametric forecasting; no forecasting head is trained.

Make the representation genuinely temporal:

- create 16 chronological frames from a rolling or progressively revealed segment;
- let visual motion correspond to changes in the time series rather than repeating a static carrier;
- keep geometry, axes, line thickness, colors, and normalization fixed;
- consider level-free shapes, first differences, and relative-to-last-value coordinates;
- render enough history for periodic structure while ensuring candidates and their continuations remain inside context;
- cache embeddings with complete cache keys.

Start with a very small, mechanistically motivated menu:

- a scrolling line/trajectory video;
- a progressively revealed period-matrix video;
- at most two backbone layers suggested by existing trained-readout diagnostics;
- cosine similarity after deterministic token normalization;
- k in a small set such as 1, 3, and 5, selected only through context-internal pseudo-origins.

Decode by transferring candidate continuations in normalized delta space, restoring the query's context-only level and scale. Combine with a base prior through the same safe-residual wrapper.

Required controls:

- raw time-domain nearest neighbors on exactly the same motifs;
- a simple correlation/DTW-style similarity if already available cheaply;
- identical random-init VideoMAE embeddings;
- pretrained embeddings with temporal frame order permuted;
- pretrained embeddings with frames duplicated into a static video;
- base prior with no retrieval.

Expected signature: pretrained temporal embeddings rank continuations whose normalized futures are more similar than those selected by random/static embeddings; that retrieval advantage survives conversion to forecast MSE and is not explained by overlapping windows.

Enforce an exclusion gap and log candidate/query overlap. Run a far-candidate version. If natural temporal embeddings do not improve neighbor continuation quality before decoding, fix representation/token extraction rather than tuning forecast blending.

### Family 3 — Frozen token-kernel continuation

Use the positive supervised cross-attention result only as a structural clue. Replace a trained readout with deterministic attention/kernel regression over observed tokens.

Build keys from frozen tokens for historical patches/motifs, queries from the current ending motif, and values from the already observed continuations associated with historical keys. Compute similarities with a fixed cosine or temperature-free rank kernel, then form a weighted continuation. No learned projection is allowed in the strict track.

Investigate, in a bounded way:

- temporal tokens versus global mean pooling;
- middle versus final frozen layer;
- within-channel and cross-channel retrieval;
- relative/difference targets versus absolute targets.

Layer/readout choice must be fixed from historical pseudo-future evidence or selected inside each context through pseudo-origins. Compare every option to random tokens and to raw-series kernels.

Expected signature: pretrained token similarity improves the continuation-quality ranking before any base blend. If an oracle linear probe works but deterministic token kernels do not, record a readout bottleneck; do not call the probe zero-shot.

### Family 4 — Genuine temporal masked continuation

Only enter this family after verifying that the checkpoint includes a compatible pretrained reconstruction decoder or that the repository's current decoder path faithfully uses pretrained temporal structure.

Represent chronology across video frames and mask future temporal tubelets, rather than repeating an image and masking an arbitrary spatial carrier. Make a neutral reconstruction decode exactly to the deterministic base forecast and let the backbone provide a bounded residual. Verify tubelet order and checkpoint-key coverage explicitly.

Candidate renderers must be invertible enough that a synthetic ramp/sinusoid reconstruction reveals orientation and phase errors. Use no more than two temporal renderers before requiring a pretrained/random separation. Kill the family if the oracle decode ceiling is inadequate or if pretrained and random are indistinguishable.

### Family 5 — Context-only router and multi-scale ensemble

Enter this family only when at least two preceding experts have complementary, replicated historical strengths. Use rolling pseudo-origins inside each current context to choose or weight experts. Experts may include the strongest prior, video-analog retrieval, token-kernel continuation, and temporal masked continuation.

Use a deterministic low-capacity rule such as inverse internal error with shrinkage toward the prior. Do not train a router across dataset targets. Require that internal expert ranking predicts held-out-origin ranking; otherwise kill the router.

The video expert must receive nonzero weight on a meaningful fraction of origins and pretrained must beat random after routing. A prior-only gain is not backbone-positive.

## 10. Diagnostic branching rules

Use these rules immediately after each result:

| Observation | Interpretation | Next action |
|---|---|---|
| Pretrained ≈ random before decoding | Representation is not carrying useful task similarity | Change temporal construction or token/layer extraction; do not tune alpha |
| Pretrained beats random before decoding but not in MSE | Readout/calibration bottleneck | Move to normalized continuation transfer, safe residual, or token kernel |
| Raw-series retrieval beats pretrained retrieval | Video representation is hurting neighbor geometry | Inspect invariances and temporal ordering; require a new representation mechanism before continuing |
| Oracle decoder ceiling cannot beat the prior | Family cannot reach the target | Kill the family |
| Internal pseudo-origin gain fails to predict held-out gain | Context selector is unreliable | Increase pseudo-origin separation or remove adaptation; do not tune on benchmark |
| Mean improves but p95/max collapses | Unsafe correction | Add preregistered context-only abstention/clipping, then allow one retest |
| Only one dataset improves | Possible regime-specific skill | Identify a context-observable regime signature and test it historically; never hard-code dataset names |
| Temporal order permutation has no effect | The video temporal axis is not being used | Audit tensor ordering/checkpoint loading or kill the claimed temporal mechanism |
| Pretrained wins but no-backbone prior is still better | Backbone signal exists but is not useful enough | Preserve evidence, change readout once, then move families |
| Large sudden gain | Possible leakage or alignment error | Run future-replacement, timestamp-shift, duplicate-window, scale, and manifest audits before promotion |

Any new hypothesis must name the observation that motivated it and predict an ablation ordering. “Try more preprocessing” is not a valid hypothesis.

## 11. Initial execution sequence

Begin immediately with the following sequence, adapting command names only after inspecting the repository:

### ZL-000 — Infrastructure and evidence audit

- Build or verify one common exact-pair evaluator and Q calculator.
- Materialize immutable historical pseudo-future manifests and hashes.
- Confirm checkpoint coverage, tensor order, normalization, official VisionTS anchors, and random/zero control semantics.
- Parse existing result artifacts into the new ledger rather than rerunning expensive jobs.
- Add synthetic and future-replacement tests.

Do not let this become an open-ended refactor. Once metrics and manifests are trustworthy, proceed.

### ZL-001 — Bounded context-calibrated safe residual

- Wrap the best existing strict candidate around the strongest context-only base.
- Select bounded alpha using only separated rolling pseudo-origins within context.
- Run a micro pilot, then ETTh2 and ETTm2 historical discovery if valid.
- Execute pretrained, random, neutral, and base controls on exact pairs.
- Kill or promote using the preregistered rule. Do not reopen static renderer tuning.

### ZL-010 — Video-native analog retrieval

- Implement the smallest scrolling/progressive temporal renderer and embedding cache.
- First test neighbor-continuation quality independently of the final decoder.
- Compare pretrained, random, temporally permuted, static-frame, and raw-series retrieval.
- Only if pretrained retrieval has a real paired advantage, attach the continuation decoder and safe base blend.

Then follow evidence into Family 3, 4, or 5. Do not execute every family blindly; the ledger and branching rules select the next one.

## 12. Resource discipline

Inspect available CPU/GPU memory and existing job conventions before launching runs. Estimate cost in the preregistration. Use cached frozen features, batched inference, mixed precision only after equivalence checks, small origin counts for smoke tests, and resumable outputs.

Never launch a broad unattended grid. A single experiment may contain at most three preregistered variants before successive halving. Keep logs visible, detect stalled/failed jobs, and resume partial deterministic work instead of starting over.

If the next decisive experiment requires materially more compute than is currently available, first try a lower-cost falsification and profiling run. Ask the user only when no meaningful lower-cost path remains.

## 13. Coding rules

- Reuse the repository's dataset loaders, split logic, evaluation manifests, and metric implementations.
- Add isolated modules/configs rather than rewriting working baselines.
- Preserve unrelated dirty-worktree changes.
- Provide CLI help, deterministic seed handling, and a dry-run mode.
- Add focused tests for causality, indices, shapes, decoding, and cache invalidation.
- Save exact commands in each candidate directory.
- Check return codes and inspect artifacts; the existence of a file is not proof a run succeeded.
- Do not commit, push, delete, or overwrite user work unless explicitly authorized.

## 14. Loop pseudocode

Follow this control flow literally:

~~~text
BOOT:
    read durable state and existing evidence
    recover or validate any interrupted run
    if infrastructure is invalid: fix the smallest blocking issue and test it

WHILE terminal target is not satisfied:
    summarize the strongest current evidence in <= 10 lines
    rank open hypotheses by information-adjusted cost
    choose one hypothesis
    write preregistration + ledger stub before seeing new results
    implement the minimum mechanism-changing patch
    run unit/synthetic/leakage tests
    if invalid:
        mark INVALID, repair once if it is a bug, and rerun
    else:
        run micro pilot
        if mechanism signature is absent:
            mark KILL or REFINE_ONCE using the preregistered rule
        else:
            run paired discovery pilot and required controls
            compute uncertainty and failure diagnostics
            PROMOTE, REFINE_ONCE, or KILL
    update STATE, HYPOTHESES, ledger, summary, and RESUME
    immediately start the next selected action

    if a candidate passes all historical gates:
        lock config and hash
        run one benchmark evaluation with controls
        audit suspicious gains
        if backbone-positive:
            write final reproducible report and stop successfully
        else:
            preserve the failed locked run
            return to historical mechanism discovery without benchmark tuning

    if context/session capacity is approaching:
        checkpoint exact state and next command
        report that the loop is resumable, not complete
~~~

## 15. Legitimate stopping conditions

Stop with SUCCESS only when the backbone-positive definition is met and all reproducibility artifacts exist.

Stop with BLOCKED only if an essential checkpoint/data file/permission/device is unavailable, the same blocker survives at least three concrete workarounds, and no lower-cost informative experiment can proceed. Record the exact missing item and resume command.

Do not stop merely because:

- the first several ideas fail;
- a script or Markdown report was completed;
- the best result is close to VisionTS;
- one or more datasets lose;
- an experiment is slow but still progressing;
- the current family is falsified.

If all listed families are genuinely falsified, use their failure signatures to formulate at least two new mechanism-level hypotheses before declaring the search frontier exhausted. A new hypothesis must preserve strict zero-shot validity and explain why it evades the observed failure. If no scientifically credible route remains, report the best reproducible negative result honestly; never manufacture a positive result.

## 16. Final deliverable on success

Produce one concise final report containing:

- exact zero-shot definition and any context adaptation used;
- locked config hash and one-command reproduction;
- six-dataset table for candidate, VisionTS, best prior, random-init, and neutral/no-backbone;
- Q_all and uncertainty;
- per-origin paired attribution statistics;
- seed stability and tail risk;
- leakage/alignment audit results;
- compute cost;
- all failed candidate IDs and why they were rejected;
- limitations, including prior exposure to benchmark results.

The final sentence must state exactly one of:

~~~text
QUALIFIED BACKBONE-POSITIVE ZERO-SHOT RESULT ACHIEVED
~~~

or

~~~text
NO QUALIFIED POSITIVE RESULT YET — LOOP STATE SAVED FOR RESUMPTION
~~~

Now inspect the repository, create the durable loop state, and execute ZL-000. Do not respond with another plan.

