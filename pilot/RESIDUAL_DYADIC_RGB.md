# Residual Dyadic RGB (RDR)

## Construction

For an observed univariate context `x[0:L]`, construct three causal views that
all end at `x[L-1]`:

```
x_1 = x[0:L:1]
x_2 = x[1:L:2]
x_4 = x[3:L:4]
```

Each view is rendered independently with the exact VisionTS transform after
dividing its context length, horizon, and periodicity by the same stride. This
keeps the observed/forecast geometry aligned. The resulting stride-1/2/4 images
are placed in R/G/B.

The ingest MLP is initialized as

```
RDR(R, G, B) = (R, R, R) + residual(R, G, B)
```

where the residual's last layer starts at zero. The frozen VideoMAE therefore
sees exactly the proven static grayscale input at initialization; coarse scales
can only enter through a learned residual. The image is duplicated into two
frames solely because VideoMAE's tubelet convolution has temporal kernel size
two. All VideoMAE weights remain frozen.

## Why this is general

- Dyadic scales are fixed and do not depend on dataset identity or a tuned lag.
- The same code runs at periods 24, 96, and 144.
- No calendar fields, channel labels, or dataset metadata enter the model.
- All normalization uses either the training split or the observed context.
- The target/test split is never used for rendering, epoch selection, or
  representation selection.

## Position relative to prior work

- [VisionTS](https://arxiv.org/abs/2408.17253) renders one normalized time
  series as one grayscale image.
- [VisionTS++](https://arxiv.org/abs/2508.04379) introduces RGB colorization for
  multiple variables, not dyadic temporal views of one variable.
- [Multi-resolution Time-Series Transformer](https://proceedings.mlr.press/v238/zhang24l.html)
  uses temporal subsampling inside a purpose-built time-series Transformer, not
  as aligned RGB images for a frozen video backbone.
- [VideoMAE](https://arxiv.org/abs/2203.12602) supplies the frozen tubelet
  backbone but does not address numeric time-series rendering.

The targeted literature search found multi-scale time-series models, RGB
multivariate rendering, and video forecasting separately, but not this exact
combination of aligned dyadic VisionTS images, RGB packing, a static identity
path, and a frozen VideoMAE backbone. This is evidence that RDR is a distinct
design; it is not a formal exhaustive novelty proof.

## Result boundary

Across full stride-24 evaluation splits and three fixed-window optimization
seeds, the one globally selected RDR configuration is best on mean MSE in 3/6
benchmark cells. It wins no MAE cells. Seed variance is material and `n=3` is
not a significance study, so claims must remain explicitly limited to the
observed mean-MSE result.
