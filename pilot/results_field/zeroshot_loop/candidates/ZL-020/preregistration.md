# ZL-020 — Deterministic token kernel over the period matrix (Family 3)

## Motivating observations (both from this loop's own data)
1. ZL-010: global mean-pooled video embeddings LOSE to raw time-domain L2 (agg 1.193). Pooling
   destroys neighbour geometry.
2. ZL-010: frame permutation costs 6.9% aggregate / 11% on ETTm2 with CI [0.765,0.897]
   excluding 1 -- the backbone genuinely reads temporal order.
3. PREPROCESSING_RESULTS: a SUPERVISED cross-attention readout over the 1568 frozen tokens of a
   period-matrix render beat random-init on 4/4 held-out datasets (+4.47%), where mean pooling
   measured zero. The information survives in tokens; the readout is what extracts it.

## Hypothesis (causal, falsifiable)
If the useful structure lives in per-token (period, phase) geometry rather than in a pooled
vector, then a DETERMINISTIC cosine kernel over frozen period-matrix tokens -- queries from the
most recent periods, keys from earlier periods, values from their already-observed successor
periods -- will rank continuations better than the identical random-init tokens and better than
the same kernel on raw period vectors.

## Mechanism change vs ZL-010
Representation and granularity, not blending:
- period-matrix render (the representation that made the supervised readout work) instead of a
  scrolling line video;
- per-token column embeddings instead of a single mean-pooled vector;
- period-level analogs instead of whole-motif analogs.
No training, no learned projection, no pixel reconstruction.

## Preregistered expected ordering (pre-decoder, normalized successor MSE)
tokens_pretrained < tokens_random   AND   tokens_pretrained <= raw_period_kernel

## Falsification rule (predeclared)
KILL Family 3 if pretrained tokens fail to beat BOTH random-init tokens AND the raw period
kernel on both ETTh2 and ETTm2. One refinement (layer choice only) if pretrained beats random
but loses to the raw kernel. If F3 is killed, the token-structure route is exhausted and the
loop must move to a mechanism that does not rely on VideoMAE feature geometry at all.

## Controls
pretrained tokens / random-init tokens / raw period-vector kernel / mean-pooled tokens
(regression control) / best context-only prior.

## Split
ETTh2 + ETTm2 historical AUDIT windows. No benchmark test target is read.
