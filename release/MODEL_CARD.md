---
license: cc-by-nc-4.0
tags:
  - time-series-forecasting
  - zero-shot
  - transfer-learning
pipeline_tag: time-series-forecasting
---

# A video-pretrained backbone adapted for zero-shot time-series forecasting

The main checkpoint from *World-Model Transfer for Data-Efficient Time-Series Forecasting*. A
VideoMAE-B backbone is adapted to forecasting by rendering a series one period per video frame,
predicting a masked block of future frames from the visible past, and decoding those frames back
into values with a differentiable read-out that carries the training loss.

This single checkpoint, under a single inference rule, produces every number in the paper's main
table: **mean MSE 0.285 and mean MAE 0.328** across six standard long-sequence datasets (ETTh1,
ETTh2, ETTm1, ETTm2, electricity, weather) and four horizons (96, 192, 336, 720), evaluated
zero-shot.

It was trained with 60,000 optimizer updates at an effective batch of 32 over a 654.2M-observation
corpus built from LOTSA: 18.9 hours on one A40, about 0.3% of the archive the strongest visual
baseline trains on.

## Loading

```python
from transformers import VideoMAEForPreTraining
model = VideoMAEForPreTraining.from_pretrained("Windsao/wm4ts-checkpoints", subfolder="videomae_b_60k")
```

The weights alone are not a forecaster. The renderer, the differentiable read-out and the inference
rule live in the code repository; `run_config.json` records the exact training configuration this
checkpoint came from.

## Intended use and limits

Research use for zero-shot univariate forecasting of series with a known dominant period. Before
using it, know that:

- The pretraining corpus was built from LOTSA without domain filtering. It contains no ETT data, but
  it does contain Monash `weather` (a different source from the LSF weather benchmark) and Monash
  `traffic_hourly`, which is the same source as the LSF traffic benchmark. **Do not evaluate this
  checkpoint on the traffic benchmark and call the result zero-shot.**
- An exact-match leakage screen over the six evaluation datasets returned no matches, but it was run
  against an earlier, domain-excluded build of the corpus rather than the one used here.
- The model emits point forecasts only. With no quantile or distributional head it cannot be scored
  on CRPS or WQL benchmarks.
- Accuracy is reported on one six-dataset protocol. Mean MAE is behind the strongest baseline
  (0.328 against 0.326), and on per-cell MAE against a same-size baseline the model wins 11 of 24.

## Citation

To be added.
