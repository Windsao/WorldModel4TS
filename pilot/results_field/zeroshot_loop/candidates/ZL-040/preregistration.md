# ZL-040 — Genuine temporal tubelet continuation (Family 4)

## Why this family is still open after F1/F2/F3 closed
Every reconstruction experiment in this repository fed VideoMAE a STATIC image repeated 16 times
and masked a SPATIAL carrier region. The temporal axis therefore never carried time in any
masked-reconstruction test. Two facts make the untested variant worth one bounded attempt:
- ZL-010 proved the backbone reads frame order (permutation costs 11% on ETTm2, CI [0.765,0.897]).
- The matched-mask study showed masked-CONTEXT reconstruction is strong (0.39-0.52) while masked
  future is not; but "future" there meant future COLUMNS of a static image, never future FRAMES.

## Hypothesis (causal, falsifiable)
If VideoMAE's pretrained prior is temporal, then rendering one period per frame and masking the
final temporal TUBELETS (i.e. future time, not a spatial strip) will let the pretrained decoder
beat both zero logits and an identical random-init model on the masked-future reconstruction,
and that advantage will exceed the ~0 attributable ceiling measured for Family 3.

## Mechanism change
- render: frame f = period f (16 frames = 12 context periods + reps future periods)
- mask: the last reps frames, i.e. whole temporal tubelets -- NOT a spatial carrier
- decode: unpatchify the reconstructed tubelets back to period values, invert context norm
- neutral identity: zero logits must decode to the deterministic base forecast

## Preregistered expected ordering
native masked-future MSE: pretrained < zero-logits AND pretrained < random-init,
with the pretrained/random ratio <= 0.95.

## Falsification rule (predeclared, terminal for the reconstruction route)
If pretrained does not beat BOTH zero logits and random-init on both datasets, Family 4 is
CLOSED and, with F1/F2/F3 already closed, the loop must report a reproducible negative and
formulate new mechanism-level hypotheses rather than continue tuning.
