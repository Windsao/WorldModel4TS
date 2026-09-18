---
license: cc-by-nc-4.0
tags:
  - time-series-forecasting
  - zero-shot
  - transfer-learning
pipeline_tag: time-series-forecasting
---

# Video-pretrained backbones adapted for zero-shot time-series forecasting

Checkpoints from *World-Model Transfer for Data-Efficient Time-Series Forecasting*. A video backbone
is adapted to forecasting by rendering a series one period per video frame, predicting a masked block
of future frames from the visible past, and decoding those frames back into values with a
differentiable read-out that carries the training loss.

| Directory | Backbone | Role | Size |
|---|---|---|---|
| `videomae_b_60k/` | VideoMAE-B, Kinetics-400 init | **Main model.** Every number in the main table | 377 MB |
| `vjepa2_l_5k/` | V-JEPA 2 ViT-L | Initialisation study, 5k updates | 2.5 GB |
| `vjepa2_1_b_5k/` | V-JEPA 2.1 ViT-B | Initialisation study, 5k updates | 790 MB |

All three were trained here on the same corpus with the same recipe; the V-JEPA pair are the
5k-update arms of the initialisation study, not tuned models. Five-dataset mean MSE at
H=96/336/720: V-JEPA 2-L 0.245/0.329/0.379, V-JEPA 2.1-B 0.252/0.335/0.382, against VideoMAE-B
0.250/0.337/0.384 at the same 5k budget.

## The main model

`videomae_b_60k/` under a single inference rule produces every number in the paper's main table: **mean MSE 0.285 and mean MAE 0.328** across six standard long-sequence datasets (ETTh1,
ETTh2, ETTm1, ETTm2, electricity, weather) and four horizons (96, 192, 336, 720), evaluated
zero-shot.

It was trained with 60,000 optimizer updates at an effective batch of 32 over a 654.2M-observation
corpus built from LOTSA: 18.9 hours on one A40, about 0.3% of the archive the strongest visual
baseline trains on.

## Loading

The main model is a standard `transformers` checkpoint:

```python
from transformers import VideoMAEForPreTraining
model = VideoMAEForPreTraining.from_pretrained("Windsao/wm4ts-checkpoints", subfolder="videomae_b_60k")
```

The two V-JEPA directories are **not** `from_pretrained`-loadable. Each holds a `world_model.pt`
state dict, which bundles the encoder, the predictor, the EMA target copy and the read-out head, plus
a `route_j.json` naming the arm. They need the code repository's own loader
(`pilot/pretrain_route_j.py`, `load_ckpt`), and the V-JEPA 2.1 arm additionally needs Meta's V-JEPA
2.1 source on the path.

The weights alone are not a forecaster in any case: the renderer, the differentiable read-out and the
inference rule live in the code repository. Each directory's `run_config.json` or `route_j.json`
records the configuration it came from.

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
- The V-JEPA arms ran 5k updates as initialisation controls and were never trained to the main
  model's budget. They are published for reproducing the initialisation study, not as forecasters to
  deploy.

## Citation

To be added.
