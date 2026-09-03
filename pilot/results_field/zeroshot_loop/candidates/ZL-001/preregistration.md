# ZL-001 — Bounded context-calibrated safe residual

## Hypothesis (causal, falsifiable)
The ~3.9% aggregate gap of the best strict VideoMAE route (Q_all=1.0391) is caused by
*harmful* decoded corrections on a subset of origins, not by an absent signal. If so, wrapping
the same frozen candidate as a bounded residual on top of the strongest context-only prior,
with alpha chosen from rolling pseudo-origins INSIDE the observed context, will recover most
of the gap without touching the renderer.

## Mechanism change
Nothing about the renderer, mask, or backbone changes. Only the OUTPUT path changes:
`y = base + alpha * clip(candidate - base)`, alpha in {0,0.25,0.5,0.75,1.0} selected per
sample by internal pseudo-forecast error; alpha=0 is always available as a safe fallback.

## Expected signature (declared before running)
1. Internal pseudo-forecast improvement correlates positively with held-out historical gain.
2. Selected alpha is nonzero on a meaningful fraction of origins (> 25%).
3. Pretrained residual beats the identical random-init residual on BOTH ETTh2 and ETTm2.
4. The wrapped method beats the base prior alone.

## Falsification rule (predeclared KILL)
KILL the salvage family after at most one refinement if ANY of:
- selected alpha collapses to 0 on > 75% of origins, or
- internal selection is anti-correlated with held-out gain, or
- pretrained residual does not beat random residual on both pilot datasets.
Do NOT reopen mask/frequency/layout sweeps under any outcome.

## Controls (exact same pairs)
base alone / pretrained residual / random-init residual / neutral(zero-logit) residual.

## Split
ETTh2 + ETTm2 historical AUDIT windows. No benchmark test target is read.
