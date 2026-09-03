# Hypothesis ledger (durable — do not rediscover)

## Established facts, do not re-run
- VisionTS anchors verified against artifacts: ETTh1 0.4023, ETTh2 0.3130, ETTm2 0.2509,
  electricity 0.2014, traffic 0.5207, solar 0.2729; stride 8, n=2000, pair hashes recorded in
  manifests/visionts_anchors.json.
- Best strict VideoMAE route Q_all = 1.0391. It wins ETTh1 and solar, loses the other four.
- Static C2 carrier: pretrained is 2.24x WORSE than a neutral carrier in phase space; the
  earlier 0.85 "native ratio" only proved a generic sinusoid was emitted.
- Matched-mask factorization: masked-CONTEXT reconstruction 0.39-0.52, masked-FUTURE 0.965-1.22.
  The backbone inpaints observed history well and does not extrapolate futures.
- Scattered future layouts (G1/G2) are worse than the contiguous block; content locality, not
  mask topology, drove the old random-mask advantage.
- Supervised cross-attention readout works (held-out 3/4 beat smean, 4/4 beat random) but is
  NOT zero-shot; it is evidence that structured tokens carry information mean pooling discards.

## Open hypotheses, ranked
1. **H-tokens (F3)** — the supervised readout worked with cross-attention over 1568 tokens while
   mean pooling failed. A deterministic token kernel over frozen tokens may preserve that
   structure without training. Highest expected information per unit compute.
2. **H-temporal-repr (F2 refinement)** — raw L2 beat the video embedding, so global mean pooling
   over a scrolling video destroys neighbour geometry. Temporal tokens / middle layer may fix it.
3. **H-temporal-mask (F4)** — mask future temporal TUBELETS instead of a spatial carrier, so the
   pretrained temporal prior is the thing being queried. Requires verifying decoder key coverage.

## Refuted
- Bounded safe residual recovers the gap (ZL-001): the gain is from the deterministic inversion
  path, not the pretrained weights.
- Analog retrieval with mean-pooled scrolling embeddings beats raw L2 (ZL-010 ETTh2): it does not.

## Families F1-F4 are now all CLOSED — failure signatures

| family | closed by | signature |
|---|---|---|
| F1 safe residual | ZL-001 | gain came from the deterministic inversion path; pt/rand 0.9949 |
| F2 analog retrieval | ZL-010 | frame order IS read (permutation costs 11%, CI excludes 1) but pretrained is no better than random for motif similarity and loses to raw L2 |
| F3 token kernel | ZL-031 | 2.7% pre-decoder ranking advantage is ORTHOGONAL to forecast quality: an ORACLE per-origin blend gives pretrained -0.06% over random |
| F4 temporal tubelet | ZL-040 | pretrained is 8.6% WORSE than zero logits at reconstructing masked future frames |

**The unifying signature**: VideoMAE features carry structure that correlates with time-series
*similarity* (better neighbour ranking, better than random) but not with time-series
*continuation* (never better than emitting a neutral value, never attributable under an oracle
decoder). Every family died at the same joint: ranking -> values.

## Two new mechanism-level hypotheses (required by section 15)

### H-A — cross-channel transfer at a single origin
Every experiment so far was channel-independent: neighbours were sought across TIME within one
channel. The unifying failure signature is specifically about temporal continuation, and says
nothing about cross-channel structure. electricity (321 channels) and traffic (862) observe
hundreds of series at the SAME origin. Render a multivariate field video and use the backbone to
rank which OTHER channels' already-observed continuations transfer to the query channel. This is
strictly zero-shot (all values are at or before the origin), uses information no univariate prior
has, and evades the F2/F3 failure because the retrieval axis is channels, not time.

### H-B — regime detection instead of value prediction
F3 produced a real, statistically solid signal: pretrained tokens rank neighbours 2.7% better
than random (CI excludes 1). It died only when asked to produce VALUES. A mechanism that consumes
only the ranking — using the backbone to decide WHICH deterministic prior to trust at this origin
(e.g. seasonal vs recent-mean regime) — demands orders of magnitude less from the representation.
The prior-selection oracle is worth 20% (ZL-031), and the internal pseudo-origin selector is
demonstrably unreliable (corr(alpha,gain) ~ 0 in ZL-001), so there is real headroom for a better
selector that is not required to output numbers.
