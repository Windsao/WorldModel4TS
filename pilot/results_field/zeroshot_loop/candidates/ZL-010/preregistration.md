# ZL-010 — Video-native context-only analog retrieval (Family 2)

## Motivating observation
ZL-001 showed the deterministic inversion path, not the pretrained weights, produces the gain
(pt/rand 0.9949, pt/neutral 0.9970). Every previous family fed the backbone a STATIC image and
read pixels back. The temporal axis of the video model has never actually carried time.

## Hypothesis (causal, falsifiable)
If a frozen video backbone encodes temporal dynamics, then embedding motifs as genuinely
SCROLLING videos will rank historical motifs by continuation similarity better than (a) an
identical random-init backbone, (b) the same backbone on frame-permuted video, (c) the same
backbone on a static repeated frame, and (d) raw time-domain L2 retrieval.

## Mechanism change
No pixel reconstruction and no decoder. The backbone is used only as a similarity kernel over
motifs; forecasts are the observed continuations of retrieved historical motifs, transferred in
level-free normalized space. Every motif AND its continuation lie inside the observed context.

## Expected signature (declared before running, ordering preregistered)
Normalized continuation MSE, lower is better:
  pretrained_scroll  <  {random_scroll, permuted_scroll, static} and <= raw_L2
If the video temporal axis is used at all, `permuted_scroll` must be measurably worse than
`pretrained_scroll`; if permutation changes nothing, the temporal axis is inert.

## Falsification rule (predeclared)
KILL Family 2 if pretrained_scroll fails to beat BOTH random_scroll and permuted_scroll on both
ETTh2 and ETTm2 pre-decoder. One refinement allowed (layer/token choice) only if pretrained beats
random but loses to raw_L2.

## Leakage controls
Exclusion gap between query and candidate motifs; every candidate continuation ends at or before
the forecast origin; overlap/distance statistics logged.
