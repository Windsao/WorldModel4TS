---
license: cc-by-nc-4.0
tags:
  - time-series-forecasting
  - zero-shot
  - transfer-learning
pipeline_tag: time-series-forecasting
---

# Video-pretrained backbones adapted for zero-shot time-series forecasting

Checkpoints from the paper *World-Model Transfer for Data-Efficient Time-Series Forecasting*.
A video backbone is adapted to forecasting by rendering a series one period per video frame,
predicting a masked block of future frames from the visible past, and decoding those frames back
into values with a differentiable read-out that carries the training loss.

## What is in this repository

| Directory | Backbone | Role in the paper | Size |
|---|---|---|---|
| `videomae_b_60k/` | VideoMAE-B (Kinetics-400 init) | **Main model.** Every number in Table 1 and Table 2 | 360 MB |
| `vjepa2_l_5k/` | V-JEPA 2 ViT-L | Initialisation study, Table 5 lower panel | 2.5 GB |
| `vjepa2_1_b_5k/` | V-JEPA 2.1 ViT-B | Initialisation study, Table 5 lower panel | 790 MB |

The main model is a single checkpoint evaluated under a single inference rule on all 24 cells of the
six-dataset long-sequence benchmark: mean MSE 0.285, mean MAE 0.328. It was trained with 60,000
optimizer updates at an effective batch of 32 over a 654.2M-observation corpus, 18.9 hours on one
A40.

## Loading

`videomae_b_60k/` is a standard `transformers` checkpoint:

```python
from transformers import VideoMAEForPreTraining
model = VideoMAEForPreTraining.from_pretrained("videomae_b_60k")
```

The two V-JEPA directories are **not** `from_pretrained`-loadable. Each holds a `world_model.pt`
state dict plus a `route_j.json` describing the arm, and needs the repository's own loader
(`pilot/pretrain_route_j.py`, function `load_ckpt`); the V-JEPA 2.1 arm additionally needs Meta's
V-JEPA 2.1 source on the path. The forecasting pipeline, the renderer and the inference rule live in
the code repository, not here.

## Intended use and limits

These are research checkpoints for zero-shot univariate forecasting of series with a known dominant
period. Points worth knowing before using them:

- The pretraining corpus is built from LOTSA without domain filtering. It contains no ETT data, but
  it does contain Monash `weather` (a different source from the LSF weather benchmark) and Monash
  `traffic_hourly`, which is the same source as the LSF traffic benchmark. **Do not evaluate these
  checkpoints on the traffic benchmark and call it zero-shot.**
- An exact-match leakage screen over the six evaluation datasets returned no matches, but it was run
  against an earlier, domain-excluded build of the corpus rather than the one used here.
- The model emits point forecasts only. It has no quantile or distributional output, so it cannot be
  scored on CRPS or WQL benchmarks without an added head.
- Accuracy is reported on one six-dataset protocol. The V-JEPA checkpoints are 5k-update controls for
  an initialisation study, not tuned models.

## Citation

Paper under review; citation to be added.
