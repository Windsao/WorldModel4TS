# Known Issues and Data-Preprocessing Research Plan

This document records the current preprocessing pipeline, implementation issues
that can affect the conclusions, and the next representation experiments for
transferring image/video foundation models to time-series forecasting.

The most important distinction is:

- **Confirmed implementation issue:** can leak future information or invalidate a
  diagnostic. Fix before interpreting new checkpoints.
- **Protocol limitation:** does not necessarily make a result wrong, but weakens
  model-to-model comparisons or the strength of a claim.
- **Research hypothesis:** a preprocessing direction that still requires an
  ablation against simple time-series baselines.

> **Current status (2026-07-15):** Phase 1--5 results remain useful as evidence
> about the tested inference/fine-tuning pipelines. Phase 6 continued-pretraining
> checkpoints should be treated as preliminary and retrained after fixing the two
> P0 preprocessing issues below.

## 1. What the code currently does

### 1.1 Dataset-level preprocessing

1. Split each dataset chronologically.
2. Compute mean and standard deviation from the **training split only**, per
   variable.
3. Standardize train/validation/test with those training statistics.
4. Create forecasting windows without shuffling time.
5. Flatten variables into independent univariate samples (`[window, channel]`).

The train-only standardization is correct and avoids dataset-level test leakage.
The channel-independent conversion is a deliberate baseline, but it discards
cross-variable structure; therefore the current experiments do not yet test
whether a 2D/3D visual layout helps model variable interactions.

### 1.2 Value-to-pixel mapping

For most evaluation renderers, each context window is normalized again:

```text
mu = mean(context)
sigma = std(context)
z = clip((x - mu) / (3 sigma), -1, 1)
gray = (z + 1) / 2
```

The inverse transform uses the same context `mu` and `sigma`. The grayscale image
is replicated to RGB and then ImageNet-normalized before entering MAE/VideoMAE.

### 1.3 Current 1D -> 2D/3D renderers

| Renderer | Mapping | Intended prior | Main limitation |
|---|---|---|---|
| VisionTS | reshape by period into a 2D phase-by-period grid | image texture and periodic structure | no explicit motion; channel-independent |
| Short VideoMAE | one period per frame; phase becomes vertical bands | temporal video encoder | only 12 context frames and strongly OOD content |
| LC-VideoMAE | 14 periods per grid, 8 content steps duplicated to 16 frames | long context plus VideoMAE tubelets | duplicated frames contain little real motion |
| Scroll-VideoMAE | a 14-period window shifts by 2 periods per frame | camera-like motion and local continuity | repeated observations and boundary/mask complexity |
| Wan2.1 | each period is repeated for several frames as horizontal bands | generative video prior | not trained for numeric temporal inpainting |
| VideoMAE-TS CPT | 50/50 LC and scroll layouts; 50/50 forecast and tube masks | adapt Kinetics VideoMAE to TS pixels | currently affected by P0 leakage below |

### 1.4 Current synthetic continued-pretraining data

`pilot/pretrain_vmae_ts.py` generates infinite univariate series containing:

- seasonal harmonics;
- slow sinusoidal components;
- linear trends;
- AR(1) noise;
- occasional level shifts;
- occasional spikes.

The model is trained with `norm_pix_loss=False`, so it predicts raw
ImageNet-normalized pixels instead of per-patch-normalized pixels. This is a good
direction because absolute level is essential for forecasting, but the renderer
must first be made causal and patch-aligned.

## 2. P0: issues to fix before retraining Phase 6

### 2.1 Horizontal bilinear interpolation leaks a future period into visible patches

Affected functions:

- `render_lc` in `pilot/pretrain_vmae_ts.py`
- `render_scroll` in `pilot/pretrain_vmae_ts.py`

The logical grid contains 14 period columns and is resized to 224 pixels with
bilinear interpolation. One period should correspond exactly to one 16-pixel
VideoMAE patch column, but bilinear resizing blends adjacent logical columns.
With scale factor 16 and `align_corners=False`, the right half of a visible patch
blends with the next logical period. On average, about 12.5% of that patch's
horizontal pixel mass comes from the next column.

During continued pretraining the full series, including the true future, is
rendered before the forecast mask is applied. Therefore a nominally visible token
can contain pixels derived from a masked future period. This does not affect the
same way at evaluation, where future cells are filled with neutral gray before
rendering, so it also creates a train/evaluation mismatch.

Recommended fix: interpolate only the phase/vertical dimension, then expand each
logical period into exactly 16 pixels without mixing columns.

```python
# grid: [T, P, 14]
img = F.interpolate(
    grid.unsqueeze(1), size=(IMG, COLS), mode="bilinear", align_corners=False
)
img = img.repeat_interleave(PS, dim=-1)  # [T, 1, 224, 224]
```

`nearest` interpolation in the horizontal dimension is also acceptable. The
invariant is more important than the exact implementation: **one logical period
must map to one patch column with no cross-column mixing**.

### 2.2 Synthetic normalization uses statistics from the future target

Affected function: `to_gray` in `pilot/pretrain_vmae_ts.py`.

`to_gray(y)` currently computes mean and standard deviation from the complete
synthetic sequence. For a forecast-shaped mask, the visible context pixels thus
depend on future values through the sequence-level mean and standard deviation.
Even if future tokens are removed from the encoder, future information is still
encoded globally in the visible token intensities.

Recommended behavior:

- **Forecast masks:** compute normalization statistics from the context only and
  use them for both context and target.
- **Random tube masks:** prefer a fixed generator scale, generator metadata, or a
  dataset-level transform independent of the hidden values. Computing statistics
  from the whole sample again makes every visible value depend on masked values.

Example for a forecast mask:

```python
context_end = (n_periods - hp) * P
mu = y[:context_end].mean()
sd = y[:context_end].std() + 1e-8
g = np.clip((y - mu) / (3 * sd), -1, 1) * 0.5 + 0.5
```

A cleaner synthetic option is to generate in a known standardized coordinate
system and use a fixed clipping scale. Then no target-dependent sample statistic
is needed.

### 2.3 Required causal-invariance test

Add a unit test that renders the same context with two different futures:

```text
render(context + future_A) -> patchify -> visible tokens
render(context + future_B) -> patchify -> visible tokens
```

For a forecast mask, the visible token tensors must be bitwise equal (or equal
within a strict floating-point tolerance). Run this test for every renderer,
horizon, period length, and augmentation channel. A second version should replace
the future with neutral gray and verify the same invariant.

## 3. P1: protocol and diagnostic limitations

### 3.1 The current raw-pixel blank control is not enough

For a continued-pretrained checkpoint with `norm_pix_loss=False`, setting predicted
patches to zero produces a constant value in ImageNet-normalized pixel space. It
does not answer whether the model uses the temporal context.

Add the following controls:

- blank context;
- shuffled context periods;
- reversed context;
- context from another sample in the batch;
- frozen random-init model with the same renderer and decoder;
- seasonal-mean and seasonal-naive numeric baselines.

A useful model should beat simple TS baselines and degrade materially when its
context is destroyed. `model < blank-control` alone is only a first signal.

### 3.2 Image-vs-video comparisons change more than pretraining modality

VisionTS and the current VideoMAE paths differ in renderer, architecture,
tokenization, mask geometry, temporal context, and decoder. A performance gap
cannot be attributed only to ImageNet versus Kinetics pretraining.

Critical controls:

1. Use the same patch-aligned grid and numeric decoder for both models.
2. Match visible-token count, context length, mask ratio, training windows, and
   optimization budget.
3. Compare random initialization, ImageNet initialization, Kinetics
   initialization, and TS continued pretraining within the closest possible
   architecture.
4. Include an image MAE continued-pretrained on the same synthetic TS budget.

### 3.3 Wan context reconstruction is not evidence of context use

`pilot/run_wan.py` hard-replaces known latent frames with ground-truth latents at
the end, so near-perfect context reconstruction is expected. In addition, fresh
noise is sampled for the known region at every denoising step; a fixed
`known_noise` would define a consistent forward noising path.

Wan2.1-T2V is not an inpainting checkpoint. The current result supports the narrow
claim that this RePaint/SDEdit wrapper does not forecast well; it does not yet show
that Wan cannot be adapted for forecasting. Also align the default resolution
with the native-resolution experiment before drawing architecture-level claims.

### 3.4 Numeric decoder has an unmeasured reconstruction floor

The pipeline applies rendering, resizing, patchification, reconstruction,
channel averaging, and adaptive pooling. Some forecasting error may come from
this codec even when the model predicts perfect pixels.

Add an oracle round-trip test: pass the ground-truth masked patches directly into
the decoder and report numeric MSE/MAE. This should be close to zero for a useful
representation. Report this error separately from model error.

### 3.5 Reproducibility and claim strength

Current experiments often use one seed and a short fine-tuning schedule. Some
methods also receive different epoch or sample budgets. Before making strong
claims, record and/or add:

- at least 3 seeds for fine-tuned results;
- validation-based learning rate, epoch, and early-stopping selection;
- identical sample/token/optimizer-step budgets for paired comparisons;
- full-channel evaluation or explicit labeling of 64-channel subsampling;
- git commit, model revision, package versions, seed, renderer version, and mask
  version in every result JSON;
- unique run IDs so result files cannot silently overwrite one another.

## 4. Preprocessing directions to test

The goal is not to make time series look visually rich. The representation should
preserve causality, be approximately invertible, align with the backbone's patch
geometry, and expose structure that its pretraining can use.

### 4.1 Patch-aligned periodic grid (highest priority)

This is the cleanest VisionTS-style baseline for both image and video models.

- reshape `x` into `[phase within period, period index]`;
- map one period to exactly one patch column;
- resize only the phase axis;
- fill future logical columns before rendering;
- keep the same renderer and decoder across MAE and VideoMAE.

Try several period choices: known calendar period, FFT/autocorrelation-selected
period, and a small multi-period bank. Wrong period selection can destroy the
useful 2D locality, so period choice must be included in the ablation.

### 4.2 Hankel / delay-embedding images

Create a matrix

```text
H[i, j] = x[i + j * delay]
```

so diagonals encode temporal evolution and repeated motifs become textures. A
sequence of nearby Hankel windows can form a video with real, structured motion.

Advantages:

- does not require a known seasonal period;
- exposes local dynamics and recurring trajectories;
- closer to natural motion than duplicated static grids.

Risks:

- the same value appears in multiple pixels, so mask design must prevent future
  values from appearing in an unmasked diagonal location;
- decoding requires overlap averaging and a carefully defined forecast region.

Start with a causal Hankel construction and verify the causal-invariance test
before training.

### 4.3 Multi-scale periodic views

Render several period grids at different scales, for example daily, weekly, and
an automatically detected period. Possible mappings are:

- one scale per RGB channel;
- one scale per frame;
- vertically stacked subfigures;
- separate encoder views followed by feature fusion.

This may help when a dataset contains multiple seasonalities, but mixing scales
inside RGB can be hard for a natural-image model to interpret. Compare against a
simple concatenated numeric seasonal baseline.

### 4.4 Semantically defined RGB channels

The current experiment uses:

- R: normalized raw value;
- G: first difference;
- B: expanding phase mean.

This is reasonable, but each channel must be causal and independently scaled
using context-only or fixed statistics. Additional candidates include trend,
seasonal residual, rolling volatility, and missingness mask.

Run a channel ablation rather than assuming RGB is always better:

```text
gray(raw) -> RGB(raw/raw/raw) -> raw/diff/trend -> raw/seasonal/residual
```

The decoder should reconstruct the raw-value channel only; auxiliary channels are
conditioning features, not forecast targets unless explicitly designed so.

### 4.5 Motion-native video without duplicated information

The scroll renderer is a useful direction, but motion should be compared at equal
information and token budgets. Candidate videos include:

- a patch-aligned sliding periodic grid;
- progressively revealed context, where each frame adds new observations;
- moving causal Hankel windows;
- coarse-to-fine frames: trend, seasonal component, residual, and raw signal.

Avoid letting the same future value appear in an earlier visible frame. Tubelet
masking must be derived from the union of all pixel locations containing a future
observation.

### 4.6 Multivariate layouts

The current channel-independent setup cannot exploit spatial relations between
variables. Test layouts such as:

- variables as rows and time/periods as columns;
- groups of correlated variables as vertically stacked subfigures;
- one variable group per frame;
- learned variable ordering based only on training-split correlations.

Preserve variable identity with a fixed ordering or an explicit embedding. Never
compute clustering/order from validation or test data. Compare with a
multivariate TS baseline so gains are not incorrectly attributed to vision.

### 4.7 Frequency-domain views

STFT or wavelet images expose time-localized frequencies and may fit image priors.
They are most useful as an auxiliary view because magnitude-only spectrograms lose
phase and are not directly invertible. If used as the target representation,
retain phase or use a numeric reconstruction head and report oracle round-trip
error.

Recurrence plots and Gramian angular fields can be used for representation
learning, but they are a lower priority for direct forecasting because their
inverse mapping and causal masking are less clean.

## 5. Safe train-time augmentations

Prefer transformations with a clear time-series meaning and a known inverse:

| Augmentation | Purpose | Constraint |
|---|---|---|
| amplitude scale and offset | robustness to units and level | apply consistently to context and target; retain inverse parameters |
| phase/time shift | reduce dependence on a fixed grid origin | shift context and target together |
| context cropping | support variable context lengths | never crop across the forecast boundary |
| small observation noise | robustness | calibrate to training-split scale |
| span dropout / missingness | missing-data robustness | provide a missingness mask; do not treat missing as true zero |
| trend/seasonal mixing | broaden synthetic dynamics | keep a known target and avoid mixing train/test samples |
| period jitter or multi-period sampling | robustness to imperfect periodicity | maintain patch and mask alignment |
| temporal resampling | speed variation | use carefully; update horizon and period metadata |

Avoid generic image augmentations unless their TS semantics are explicit:

- horizontal flip reverses time;
- vertical flip negates/reorders values;
- random crop can delete the forecast boundary;
- color jitter changes numeric values;
- arbitrary rotation destroys the time/value axes.

These transformations may be useful only as intentional, invertible TS
augmentations—not as default computer-vision augmentation.

## 6. Minimal ablation plan

Fix P0 issues first, discard affected Phase 6 checkpoints for final reporting,
and then change one preprocessing factor at a time.

| ID | Renderer / preprocessing | Purpose |
|---|---|---|
| A0 | patch-aligned periodic grid, grayscale, context-only scale | leakage-free reference |
| A1 | A0 with fixed synthetic scale | context scale vs fixed scale |
| A2 | static frames vs scroll, matched context/tokens | test value of motion |
| A3 | grayscale vs raw/diff/trend RGB | test feature-plane augmentation |
| A4 | known period vs detected period vs multi-period | test period sensitivity |
| A5 | periodic grid vs causal Hankel | test representation family |
| A6 | channel-independent vs multivariate layout | test cross-variable benefit |
| A7 | ImageNet MAE vs Kinetics VideoMAE vs random init | isolate pretraining source |
| A8 | image MAE-CPT vs VideoMAE-CPT, same synthetic data/steps | critical modality control |

For every arm report:

- forecasting MSE and MAE;
- improvement over seasonal-naive and seasonal-mean;
- oracle renderer/decoder round-trip error;
- blank, shuffle, reverse, and cross-sample context controls;
- number of context values, visible tokens, training samples, optimizer steps, and
  GPU-hours;
- mean and standard deviation over seeds for trained models.

## 7. Suggested implementation structure

Move representation logic into a shared module so training and evaluation cannot
silently diverge:

```text
pilot/preprocessing/
  config.py          # renderer/mask/normalization versioned configs
  normalize.py       # context-only and fixed-scale transforms
  periodic_grid.py   # patch-aligned static renderer
  scroll.py          # patch-aligned motion renderer
  hankel.py          # causal delay embedding
  channels.py        # raw/diff/trend and other causal feature planes
  masks.py           # masks derived from observation-to-pixel provenance
  decode.py          # shared pixel-to-number decoder
  tests/
    test_causality.py
    test_mask_coverage.py
    test_roundtrip.py
```

Each rendered pixel should ideally have provenance metadata indicating which
original time index produced it. A mask can then be generated from provenance:
if a patch contains any value at or after the forecast boundary, mask that entire
patch. This makes leakage checks systematic for periodic, scroll, and Hankel
layouts.

## 8. Interpretation policy until the fixes land

- Phase 6 checkpoint improvements are promising engineering signals, not yet
  evidence of leakage-free zero-shot forecasting.
- Wan results describe the tested outpainting wrapper, not an impossibility result
  for video diffusion models.
- A video model should be called useful only when it beats meaningful numeric
  baselines and context-destruction controls under a causal, matched protocol.
- The key scientific question is not whether TS can be converted to pixels; it is
  whether video pretraining contributes predictive information beyond an image
  prior, a random model, and strong time-series baselines.
