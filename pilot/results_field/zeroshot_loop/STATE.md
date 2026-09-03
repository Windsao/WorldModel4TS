# Zero-shot loop — STATE

- **timestamp**: 2026-09-03T03:39:40Z
- **git**: 86ece3af7191c2b610f281695289b7dba8cbb3fe (dirty worktree; many untracked experiment files preserved)
- **env**: NU CS slurm, node nyx, 4x A40, conda `wm4ts`, transformers 4.46.3, torch 2.13.0
- **checkpoint**: MCG-NJU/videomae-base
- **workdir on cluster**: /nyx-storage1/hanliu/shang/vvts

## Phase
Family 2 (video-native analog retrieval) discovery pilot. ZL-000 and ZL-001 complete.

## Best numbers so far
- **discovery (ETTh2+ETTm2 historical audit, stride 8, 144/620 origins)** — candidate ZL-024:
  - candidate / best context prior = **0.9242** (ETTh2 CI [0.906,0.963], ETTm2 CI [0.895,0.924], both exclude 1)
  - pre-decoder attribution: pt/random = **0.9729**, pt/raw-period-kernel = **0.9416**
  - forecast-level attribution: pt/random = **0.9921** -> NOT backbone-positive (needs <= 0.98)
- **benchmark**: untouched by this loop. Best strict route from prior work remains Q_all = 1.0391.
- Status: **metric-positive not yet claimed** (no benchmark run); **backbone-positive not reached**.

## Families
| family | status | why |
|---|---|---|
| static carrier / mask / layout / lens sweeps | **CLOSED** | prior reports; carrier signal study showed PT is 2.24x worse than a neutral carrier |
| F1 safe residual (ZL-001) | **CLOSED** | preregistered KILL: pt/rand 0.9949 > 0.98, ETTm2 CI includes 1, corr(alpha,gain)~0 |
| F2 video-native analog retrieval (ZL-010) | **CLOSED** | both datasets in: PT beats frame-permuted (0.931 agg, ETTm2 CI excludes 1) but NOT random-init (0.997 agg), and raw L2 is far better (1.193). Refinement clause not triggered. |
| F3 frozen token-kernel | **CLOSED** | ZL-031 oracle ceiling: even a PERFECT per-origin blend gives pretrained a -0.06% advantage over random (1.0006). The retrieval structure is worth 20% over the prior; the pretrained weights are worth nothing. |
| F4 temporal masked continuation | **CLOSED** | ZL-040: pretrained is 8.6% WORSE than zero logits at reconstructing masked future frames (beats random 0.919 but loses to neutral) |
| F5 context-only router | BLOCKED | needs two complementary experts first |

## Last completed command / status
`python3 pilot/run_zl010_analog.py --dataset ETTh2 ... --n 192` -> exit 0, artifacts written.

## Running jobs
None. ZL-010 completed on both datasets (exit 0, artifacts validated).

## TERMINAL OBJECTIVE — METRIC-POSITIVE REACHED (ZL-051)

**Q_all = 0.9152 against VisionTS** on the benchmark test windows at identical origins
(`pairs_sha1` verified on all six datasets). Wins with a paired CI excluding 1 on ETTh2, ETTm2,
traffic and solar; ties on ETTh1 and electricity; loses on none. Zero trained parameters.
Full report: `ZEROSHOT_POSITIVE_RESULT.md`. Leak audit: all checks pass.

**Backbone-positive is NOT reached and the evidence says it is not reachable by this route.**
The video kernel is a net negative in both controlled configurations, and pretrained/random =
1.0288 on the benchmark. The positive result contains no video model.

## Single next action
Nothing is blocking. The remaining open question is H-A (cross-channel transfer at a single
origin), which is the only mechanism-level hypothesis never tested. It now has to clear a much
higher bar: the ZL-051 blend, not VisionTS.

## MANDATORY for any new decode path
ZL-040 shipped an oracle-statistic leak: it de-normalized masked cubes with statistics taken from
a video containing the true future. It was caught only by the suspicious-gain audit (zero logits
beat the prior 2.3x). Every future decode MUST use causal statistics and MUST run the zero-logit
sanity check before its numbers are read.

## Blockers
None external. Compute is free on nyx.
