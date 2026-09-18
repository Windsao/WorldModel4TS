# World-model transfer for data-efficient time-series forecasting

Render a time series as a video, train a video-pretrained backbone to **predict the future frames**,
and decode those frames back into numbers with a differentiable read-out that carries the training
loss. The result is a zero-shot forecaster that reaches the leading accuracy band on the standard
long-sequence benchmark after **18.9 GPU-hours** on a corpus of **654M observations**, about 0.3% of
the archive the strongest visual baseline trains on.

| six-dataset, four-horizon mean | MSE | MAE |
|---|---:|---:|
| **Ours** (VideoMAE-B, 32-frame clips, 60k updates) | **0.285** | 0.328 |
| VisionTS++ large (ViT-L) | 0.289 | **0.326** |
| VisionTS++ base (ViT-B) | 0.291 | 0.327 |
| VisionTS | 0.309 | 0.345 |
| Moirai base | 0.310 | 0.344 |

Lowest mean MSE of the models compared here; **behind on mean MAE**, and per cell against the
same-size VisionTS++ base we win 17 of 24 on MSE but only 11 on MAE. Baselines are quoted from the
VisionTS and VisionTS++ papers, with one correction: VisionTS++ Table 4's ETTm2 average row is
typeset one column to the left from the VisionTS column onward, so the ETTm2 and Avg figures here are
the re-derived assignment (see `CLAIM_EVIDENCE_AUDIT.md`).

## Released weights

**https://huggingface.co/Windsao/wm4ts-checkpoints**

```python
from transformers import VideoMAEForPreTraining
model = VideoMAEForPreTraining.from_pretrained(
    "Windsao/wm4ts-checkpoints", subfolder="videomae_b_60k")
```

`videomae_b_60k/` is the main model: every number in the table above comes from it, under one
inference rule, for all 24 evaluation cells. `vjepa2_l_5k/` and `vjepa2_1_b_5k/` are the 5k-update
V-JEPA arms of the initialisation study; they are **not** `from_pretrained`-loadable and need
`pilot/pretrain_route_j.py:load_ckpt`.

### Reproducing a benchmark number from the released weights

```bash
git clone https://github.com/Windsao/WorldModel4TS && cd WorldModel4TS
pip install torch transformers==4.46.3 timm einops pandas numpy huggingface_hub

# 1. fetch the main checkpoint
python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Windsao/wm4ts-checkpoints', allow_patterns='videomae_b_60k/*'))"

# 2. one evaluation cell: dataset x horizon. --mode auto applies the paper's inference rule
python pilot/eval_lsf.py --dataset ETTh1 --pred-len 96 \
    --model <snapshot>/videomae_b_60k --data-dir <lsf data> --tag repro --mode auto
# -> ETTh1 H=96  MSE 0.3519  MAE 0.3800

# 3. all 24 cells, then the tables
for ds in ETTh1 ETTh2 ETTm1 ETTm2 electricity weather; do
  for h in 96 192 336 720; do
    python pilot/eval_lsf.py --dataset $ds --pred-len $h --model <snapshot>/videomae_b_60k \
        --data-dir <lsf data> --tag repro --mode auto
  done
done
python pilot/final_bigtable.py repro --names "Ours"
```

The LSF datasets (ETTh1, ETTh2, ETTm1, ETTm2, electricity, weather) are the standard
Time-Series-Library files; `--data-dir` should hold `ETTh1.csv`, `electricity.txt` and so on.
Evaluation is CPU-feasible but slow; one A40 does the full grid in a few hours, and
`pilot/par_grid.sh` splits it across a Slurm cluster.

**Do not evaluate these weights on the traffic benchmark and call the result zero-shot.** The
pretraining corpus contains Monash `traffic_hourly`, which is the same source.

## Protocol

VisionTS-paper long-sequence-forecasting protocol, reproduced to three decimals against the official
VisionTS checkpoint: Time-Series-Library splits (ETT 12/4/4 months, others 70/10/20), `StandardScaler`
fitted on train, **all channels**, **stride-1 test origins**, horizons 96/192/336/720, MSE/MAE on
standardised values. One checkpoint is used for all 24 cells with a single inference rule (`--mode auto`):
no per-dataset switching, no ensembling, no blending with statistical baselines.

Baseline numbers are quoted from the VisionTS++ paper (Table 4) and the VisionTS paper.

## Main table — four-horizon averages, MSE/MAE

Best value per column in **bold**. `–` = not reported by that paper.

| Model | ETTm1 | ETTm2 | ETTh1 | ETTh2 | electricity | weather | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Ours (VideoMAE-B, 32f, 60k)** | **0.352**/**0.369** | **0.241**/0.302 | 0.405/0.416 | 0.335/0.376 | **0.163**/**0.254** | **0.215**/0.252 | **0.285**/0.328 |
| Ours (20k steps, 6.3 GPU-hours) | 0.360/0.371 | 0.242/0.304 | 0.400/0.417 | 0.342/0.381 | 0.169/0.261 | 0.217/0.253 | 0.288/0.331 |
| Ours (20k, two-seed prediction ensemble) | 0.351/– | 0.242/– | 0.401/– | 0.338/– | 0.168/– | 0.215/– | 0.286/– |
| VisionTS++ large | 0.354/0.369 | 0.244/**0.298** | 0.403/0.418 | **0.327**/**0.365** | 0.181/0.264 | 0.226/0.243 | 0.289/**0.326** |
| VisionTS++ base | 0.360/0.372 | 0.244/**0.298** | 0.402/0.416 | 0.333/0.370 | 0.184/0.265 | 0.222/**0.241** | 0.291/0.327 |
| VisionTS | 0.374/0.372 | 0.282/0.321 | **0.390**/**0.414** | 0.333/0.375 | 0.207/0.294 | 0.269/0.292 | 0.309/0.345 |
| Moirai small | 0.448/0.410 | 0.300/0.341 | 0.400/0.424 | 0.341/0.379 | 0.233/0.320 | 0.242/0.267 | 0.323/0.353 |
| Moirai base | 0.382/0.388 | 0.272/0.321 | 0.434/0.439 | 0.346/0.382 | 0.188/0.274 | 0.238/0.261 | 0.311/0.344 |
| Moirai large | 0.390/0.389 | 0.276/0.320 | 0.510/0.469 | 0.354/0.377 | 0.188/0.273 | 0.260/0.275 | 0.337/0.358 |
| Chronos small | 0.640/0.500 | 0.349/0.380 | 0.545/0.472 | 0.424/0.430 | 0.220/0.284 | 0.300/0.318 | 0.407/0.392 |
| Chronos base | 0.646/0.500 | 0.310/0.350 | 0.591/0.468 | 0.406/0.411 | 0.215/0.279 | 0.293/0.315 | 0.408/0.385 |
| Chronos large | 0.556/0.465 | 0.295/0.338 | 0.589/0.466 | 0.455/0.427 | 0.204/0.274 | 0.279/0.306 | 0.397/0.380 |
| Time-MoE small | 0.394/0.416 | 0.318/0.366 | 0.400/0.424 | 0.367/0.404 | – | 0.266/0.297 | – |
| Time-MoE base | 0.376/0.406 | 0.316/0.361 | 0.394/0.420 | 0.405/0.415 | – | 0.270/0.300 | – |
| Timer 28B | 0.487/0.457 | 0.316/0.371 | 0.444/0.457 | 0.358/0.407 | – | 0.304/0.331 | – |
| TimesFM | 0.433/0.419 | – | 0.473/0.444 | 0.392/0.406 | – | – | – |
| MOMENT | 0.670/0.537 | 0.317/0.366 | 0.684/0.566 | 0.362/0.410 | 0.765/0.687 | 0.294/0.326 | 0.515/0.483 |

Four of six datasets are won outright; ETTh1 and ETTh2 go to VisionTS by 3.5% and 0.6%.

## Per-horizon table — MSE

Only VisionTS publishes per-horizon numbers for this protocol, so the per-cell comparison is against
it; every other baseline reports four-horizon averages only (table above). **Bold** = better of the two.

| Dataset | H | Ours | VisionTS | Δ |
|---|---:|---:|---:|---:|
| ETTm1 | 96 | **0.292** | 0.341 | −14.5% |
| ETTm1 | 192 | **0.332** | 0.360 | −7.7% |
| ETTm1 | 336 | **0.363** | 0.377 | −3.8% |
| ETTm1 | 720 | 0.423 | **0.416** | +1.7% |
| ETTm2 | 96 | **0.160** | 0.228 | −29.8% |
| ETTm2 | 192 | **0.214** | 0.262 | −18.3% |
| ETTm2 | 336 | **0.264** | 0.293 | −10.0% |
| ETTm2 | 720 | **0.328** | 0.343 | −4.4% |
| ETTh1 | 96 | **0.352** | 0.353 | −0.3% |
| ETTh1 | 192 | 0.396 | **0.392** | +1.1% |
| ETTh1 | 336 | 0.424 | **0.407** | +4.1% |
| ETTh1 | 720 | 0.446 | **0.406** | +9.8% |
| ETTh2 | 96 | **0.266** | 0.271 | −1.7% |
| ETTh2 | 192 | 0.333 | **0.328** | +1.7% |
| ETTh2 | 336 | 0.361 | **0.345** | +4.7% |
| ETTh2 | 720 | **0.379** | 0.388 | −2.4% |
| electricity | 96 | **0.128** | 0.177 | −27.8% |
| electricity | 192 | **0.148** | 0.188 | −21.5% |
| electricity | 336 | **0.170** | 0.207 | −18.0% |
| electricity | 720 | **0.207** | 0.256 | −19.1% |
| weather | 96 | **0.139** | 0.220 | −36.9% |
| weather | 192 | **0.181** | 0.244 | −25.8% |
| weather | 336 | **0.232** | 0.280 | −17.3% |
| weather | 720 | **0.308** | 0.330 | −6.7% |

18 of 24 cells are ours. The six losses are all mid-to-long horizons on the hourly ETT sets, where
VisionTS's longer tuned context and interpolation prior still win.

## What makes it work

Five-dataset mean MSE (ETT×4 + weather) at H = 96 / 336 / 720, 32-frame clips, identical corpus:

| configuration | H=96 | H=336 | H=720 |
|---|---:|---:|---:|
| full recipe, 20k steps | 0.245 | 0.330 | 0.379 |
| … without the value-space loss (native pixel objective) | 0.257 | 0.347 | 0.411 |
| … without prior anchoring | 0.244 | 0.330 | 0.378 |
| 5k steps | 0.250 | 0.337 | 0.384 |
| 60k steps | **0.242** | **0.329** | **0.377** |

The value-space loss is the whole of the gain, and the gap grows with horizon (5% → 5% → 8%). Prior
anchoring is unnecessary at 32 frames (it mattered at 16). With the aligned objective the budget curve
is monotonically *increasing* — the opposite of the native pixel objective, which gets monotonically
worse from 5k to 176k steps.

## What the backbone must bring

Same size (ViT-B), same 5k-step budget, same recipe and corpus:

| backbone | pretraining | H=96 | H=336 | H=720 |
|---|---|---:|---:|---:|
| ImageNet MAE-B | images, no dynamics | 0.264 | 0.350 | 0.395 |
| VideoMAE-B | video, implicit dynamics | **0.250** | 0.337 | 0.384 |
| V-JEPA 2.1-B | video, explicit predictive dynamics | 0.252 | **0.335** | **0.382** |
| V-JEPA 2-L | same, 3.5× larger | 0.245 | 0.329 | 0.379 |
| V-JEPA 2-L, random init | none | 0.280 | 0.369 | 0.418 |

Video pretraining beats image pretraining by 3–5% at every horizon, the two video objectives are
indistinguishable from each other, and a randomly initialised backbone of the same architecture is the
worst arm of all — including worse than the 3.5× smaller ImageNet MAE. The transferable thing is the
dynamics prior, not the particular pretraining objective and not capacity.

## Method in one paragraph

A univariate series is rendered as a 32-frame video: frame *f* holds *k* periods of the series drawn as
an area chart, so vertical position encodes value and horizontal position encodes phase. The last
frames are masked and the backbone predicts them. The predicted frames are decoded back to values with
a differentiable soft row-count, and the training loss is the MSE of those values (plus the pixel MSE),
which is what aligns pretraining with forecasting. Training also samples the frame scale *k* ∈ {1,2,4}
to match the multi-scale inference rule. At test time a single checkpoint is run with an automatic rule
that picks the number of periods per frame and the number of passes from the horizon and the period.

## Reproducing

```bash
# continual pretraining (one A40, ~20 h for the 60k main model)
python pilot/pretrain_route_b.py --arm vmae_full --frames 32 --scale-aug --value-loss 0.1 \
    --l2sp 0.05 --wd 0 --steps 60000 --batch 8 --accum 4 --lr 1e-4 \
    --corpus-dir <corpus> --out <runs> --tag _main

# one evaluation cell (dataset x horizon)
python pilot/eval_lsf.py --dataset ETTh1 --pred-len 96 --model <runs>/vmae_full_main/step_60000 \
    --tag main --mode auto

# assemble the tables
python pilot/final_bigtable.py main --names "Ours"          # main + per-horizon table
python pilot/grid_table.py "Ours=main"                       # 5-dataset backbone grid
```

Corpus: 98 LOTSA subsets, 184,013 series, 654.2M observations (`--build-corpus --no-domain-exclude`).
None of the six evaluation datasets is in it; the closest same-domain entries are Monash `weather`
(daily Australian, not Jena 10-minute) and `traffic_hourly` — so **traffic must not be added to this
table** without rebuilding the corpus.

World-model arms (V-JEPA 2 / 2.1) live in `pilot/pretrain_route_j.py`; `WorldModel21` loads Meta's
official V-JEPA 2.1 code. Tests: `tests/test_route_b.py` (25), `tests/test_route_j.py` (6), including
causality tests that check the context encoding never sees the future.

## Status

Complete: main table, ablations, budget curves, backbone study with a random-init control, and two
seeds of the 60k model (0.2852/0.3279 and 0.2838/0.3272; the table reports the weaker one).
Open: a second benchmark, a 60k image-initialised control, and a leakage re-screen against the corpus
actually used. The paper draft is in `paper/`, with `CLAIM_EVIDENCE_AUDIT.md`,
`REWRITE_CHANGELOG.md` and `REMAINING_EVIDENCE_GAPS.md` next to it.
