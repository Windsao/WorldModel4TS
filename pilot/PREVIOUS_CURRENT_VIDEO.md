# Previous/Current Video Axis

## Construction

Let the observed context be `x[0:L]`. The smallest legal VideoMAE clip has two
frames because the checkpoint uses tubelets of size two. Build those frames as:

```text
previous = [x[0], x[0], x[1], ..., x[L-2]]
current  = [x[0], x[1], x[2], ..., x[L-1]]
```

Both length-`L` sequences use the mean and scale of the observed current
context, then pass through the exact VisionTS renderer. The left-edge replicate
keeps the construction fixed-length without introducing a learned value or any
future sample. The clip is ordered `previous -> current`.

This retains one tubelet and 196 encoder tokens, exactly the same forecast-head
capacity as the duplicated static image. Only the pixel ingest MLP and forecast
MLP train; all 86,227,200 VideoMAE backbone parameters remain frozen.

## Non-degeneracy and causality checks

Three properties are regression tested:

1. the two frames are exact renderings of the shifted and current observed
   contexts and differ whenever the input changes;
2. changing any forecast value leaves both frames and their statistics
   bit-exact;
3. the repeat control is two exact copies of the current VisionTS image, while
   the reverse control swaps the two genuine frames.

The observed mean squared pixel difference between the two input frames is
strictly positive in every locked cell (0.000316–0.006738).

The temporal-use diagnostic does not train a separate control model. After a
forward checkpoint is selected on validation MSE, the same checkpoint is
evaluated on repeated and reversed validation clips. This prevents a new head
from learning around the control and isolates dependence on frame content and
order. Controls never participate in checkpoint selection or test evaluation.

## Locked result

The protocol uses a fixed set of 4,096 training windows, every stride-24
validation/test window, and model seeds 0/1/2. Values are mean ± sample standard
deviation.

| dataset / H | previous/current MSE | static MSE | residual dyadic MSE | repeat−forward validation MSE | reverse−forward validation MSE |
|---|---:|---:|---:|---:|---:|
| ETTh1 / 96 | **0.3873 ± 0.0065** | 0.4026 ± 0.0155 | 0.3946 ± 0.0270 | +0.0721 | +0.1434 |
| ETTh1 / 192 | 0.4215 ± 0.0114 | 0.4369 ± 0.0200 | **0.4145 ± 0.0043** | +0.0455 | +0.1384 |
| ETTm1 / 96 | 0.3317 ± 0.0052 | 0.3332 ± 0.0085 | **0.3302 ± 0.0094** | +0.0164 | +0.0104 |
| ETTm1 / 192 | 0.3643 ± 0.0049 | **0.3542 ± 0.0065** | 0.3605 ± 0.0089 | +0.0182 | +0.0076 |
| Weather / 96 | 0.1709 ± 0.0095 | 0.1721 ± 0.0030 | **0.1661 ± 0.0048** | +0.0059 | +0.0085 |
| Weather / 192 | 0.2119 ± 0.0071 | 0.2140 ± 0.0078 | **0.2084 ± 0.0034** | +0.0004 | +0.0043 |

The forward clip beats both same-checkpoint controls on average in all six
cells, so the temporal axis is genuinely used. It beats the static adapter on
mean test MSE in 5/6 cells, but beats residual dyadic RGB in only 1/6. Two
individual repeat controls are negative (ETTh1/96 seed 1 and Weather/192 seed
1), and the Weather/192 aggregate gap is tiny. The supported claim is therefore
that this is a meaningful, competitive temporal axis—not that it is a broad
accuracy improvement or a new state of the art.

Machine-readable seed scores and protocol metadata are in
`results_temporal_axis/step_shift_multiseed.json`.

## Reproduce

```bash
python pilot/run_temporal_image_adapter.py \
  --dataset ETTh1 --data-dir pilot/data --mode step_shift_video \
  --video-frames 2 --horizon 96 --train-cap 4096 \
  --val-cap 0 --test-cap 0 --eval-stride 24 --batch-size 32 \
  --epochs 8 --patience 3 --window-seed 0 --seed 0 \
  --no-checkpoint --eval-controls
```

Repeat with seeds 1 and 2 and with each dataset/horizon pair.
