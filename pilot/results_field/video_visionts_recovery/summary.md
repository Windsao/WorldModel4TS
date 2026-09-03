# Video-VisionTS recovery summary

> The previous Stage E1 outputs in `video_visionts/label_free_adaptation/` are
> **`invalid_target_bug`**: they were trained against a target rendered from a zero
> future. They are retained for provenance and excluded from every table here.

## R3 — historical layout audit (regime HA)

| dataset | layout | seed | ctl | native ratio | oracle raw | causal(phys) raw | TS model causal | TS zero causal |
|---|---|---:|---|---:|---:|---:|---:|---:|
| ETTh2 | G0 | 0 | rand | **1.1807** | 0.00409 | 0.04123 | 0.4930 | 0.4888 |
| ETTh2 | G0 | 0 | pret | **0.9671** | 0.00340 | 0.03959 | 0.4861 | 0.4888 |
| ETTm2 | G0 | 0 | rand | **1.2003** | 0.00652 | 0.03034 | 0.3207 | 0.3193 |
| ETTm2 | G0 | 0 | pret | **0.9704** | 0.00549 | 0.02898 | 0.3180 | 0.3193 |
| ETTh2 | G1 | 0 | pret | **1.0121** | 0.00341 | 0.05699 | 0.9162 | 0.9193 |
| ETTm2 | G1 | 0 | pret | **1.0723** | 0.00598 | 0.05116 | 0.7760 | 0.7679 |
| ETTh2 | G2 | 0 | pret | **1.5643** | 0.00532 | 0.06344 | 0.9293 | 0.8699 |
| ETTh2 | G2 | 1 | pret | **1.5601** | 0.00536 | 0.06386 | 0.9683 | 0.9299 |
| ETTh2 | G2 | 2 | pret | **1.5504** | 0.00535 | 0.06282 | 0.9535 | 0.8962 |
| ETTm2 | G2 | 0 | pret | **1.4951** | 0.00785 | 0.05511 | 0.7894 | 0.7421 |
| ETTm2 | G2 | 1 | pret | **1.5057** | 0.00786 | 0.05628 | 1.0301 | 0.9750 |
| ETTm2 | G2 | 2 | pret | **1.4936** | 0.00782 | 0.05513 | 0.8526 | 0.8243 |

## R4 — corrected lens historical evaluation (regime LA)

| variant | layout | backbone | seed | aggregate ratio vs L0 | best R_val | mean-clamp | logstd-sat |
|---|---|---|---:|---:|---:|---:|---:|
| L1 | G0 | pretrained | 0 | **1.0347** | 0.9992 | 0.000 | 0.098 |
| L2 | G0 | pretrained | 0 | **1.0204** | 0.9985 | 0.000 | 0.000 |
| L3 | G0 | pretrained | 0 | **1.0216** | 0.9989 | 0.000 | 0.000 |

## R5 — strict forecast  layout=G0 seed=0 lens=L0 stat=nearest_visible_physical backbone=pretrained

| dataset | method | zeroed | random | VisionTS | Stage-D right_10 | smean | snaive | q vs VTS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.3984 | 0.4110 | 0.4107 | 0.4023 | 0.3984 | 0.4026 | 0.5043 | 0.990 |
| ETTh2 | 0.3248 | 0.3267 | 0.3296 | 0.3130 | 0.3248 | 0.3454 | 0.3852 | 1.038 |
| ETTm2 | 0.2553 | 0.2569 | 0.2580 | 0.2509 | 0.2615 | 0.3301 | 0.2651 | 1.018 |
| electricity | 0.2313 | 0.2567 | 0.2581 | 0.2014 | 0.2313 | 0.2069 | 0.3322 | 1.149 |
| traffic | 0.6018 | 0.6242 | 0.6296 | 0.5207 | 0.6018 | 0.5106 | 0.9491 | 1.156 |
| solar | 0.2575 | 0.2718 | 0.2735 | 0.2729 | 0.2415 | 0.1973 | 0.2852 | 0.944 |

- **Q_all = 1.0461**  (NOT MET), **Q_heldout = 1.0554**
- win/loss vs VisionTS: 2/4
- leave-one-out Q_all: ETTh1:1.058, ETTh2:1.048, ETTm2:1.052, electricity:1.027, solar:1.068, traffic:1.025

