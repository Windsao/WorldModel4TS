# ZL-030 — Token kernel in PRIOR-RESIDUAL space (Family 3, mechanism change)

## Motivating observation (this loop's own data)
ZL-021/023/024 all land at forecast-level pt/random ~ 0.99 while pre-decoder pt/random is 0.973
and pt/raw-kernel is 0.942. Every wrapper multiplies the video contribution by a safety factor
(alpha ~ 0.5, rho ~ 0.47), so a 3-6% representation advantage arrives as ~1%. The dilution is
structural, not a tuning problem.

## Hypothesis (causal, falsifiable)
If the kernel retrieves the PRIOR'S ERROR rather than the raw successor, the retrieved quantity
is already the correction the prior needs, so no shrinkage is required to keep it safe and the
representation-level advantage should survive to forecast level.

## Mechanism change
keys  : unchanged frozen period-matrix column tokens of m-period windows
values: (observed successor) - (same-family prior forecast computed from periods [0,e) only)
output: prediction = prior_at(G) + weighted mean of retrieved residuals
No shrinkage factor, no alpha, no learned parameter. Everything read is inside the context.

## Preregistered expected ordering
forecast-level pt/random must improve on 0.9921 (the ZL-024 value) and reach <= 0.98,
while candidate/base stays <= 1.00.

## Falsification rule (predeclared)
If ZL-030 also lands at pt/random > 0.98, the readout is NOT the bottleneck. Stop trying
decoders and instead measure the ORACLE ceiling of a perfect kernel-vs-prior combination; if
that ceiling cannot deliver >= 2% attributable gain, Family 3 is closed and the loop must find a
mechanism that does not depend on VideoMAE feature geometry.

## Controls
pretrained tokens / random-init tokens / raw period kernel / prior alone. Identical pairs.
