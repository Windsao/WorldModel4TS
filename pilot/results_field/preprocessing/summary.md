# Preprocessing experiment summary

`transfer_gain` is the decisive column: pretrained vs the SAME architecture
randomly initialized. Absolute gain over `smean` only shows the readout learned
something.

## Renderer screen

| Dataset | Renderer | Readout | Norm | PT MSE | Rand MSE | smean | Transfer gain | Beats random? | Beats smean? |
|---|---|---|---|---:|---:|---:|---:|---|---|
| ETTh1 | period_matrix | xattn | std_clip_3 | 0.4195 | 0.4469 | 0.4022 | +6.1% | YES | no |
| ETTm2 | period_matrix | xattn | std_clip_3 | 0.1719 | 0.1768 | 0.3224 | +2.8% | YES | YES |
| solar | period_matrix | xattn | std_clip_3 | 0.1796 | 0.1846 | 0.2005 | +2.7% | YES | YES |
| traffic | period_matrix | xattn | std_clip_3 | 0.3358 | 0.3582 | 0.5175 | +6.3% | YES | YES |

## Held-out verdict (section 13 criteria)

- held-out pairs: 4
- (1) beats smean on 3/4 (need >=3 of 4)
- (2) mean transfer gain +4.47% (need >= +3%)
- (3) transfer gain positive on 4/4 (need >=3 of 4)

**VERDICT: POSITIVE**
