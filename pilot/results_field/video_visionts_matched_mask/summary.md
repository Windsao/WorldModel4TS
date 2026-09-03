# Matched-mask factorization summary

Geometry fixed at 4 context / 10 future columns for every mask; the image is
rendered once per batch and reused (hash-checked).

| mask | seed | ctx masked | fut masked | ctx ratio | fut ratio | fut PT/RAND | fut nondeg ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| context_random_42 | 0 | 42 | 0 | 0.4991 | nan | nan | nan |
| context_random_42 | 1 | 42 | 0 | 0.4113 | nan | nan | nan |
| context_random_42 | 2 | 42 | 0 | 0.4545 | nan | nan | nan |
| future_block_10 | 0 | 0 | 140 | nan | 0.9654 | 0.8109 | 0.9633 |
| future_block_2 | 0 | 0 | 28 | nan | 1.2169 | 1.0199 | 1.2009 |
| future_block_5 | 0 | 0 | 70 | nan | 1.0153 | 0.8530 | 1.0121 |
| future_block_8 | 0 | 0 | 112 | nan | 0.9945 | 0.8359 | 0.9924 |
| future_random_105 | 0 | 0 | 105 | nan | 1.0384 | 0.8721 | 1.0360 |
| future_random_105 | 1 | 0 | 105 | nan | 1.0620 | 0.8919 | 1.0578 |
| future_random_105 | 2 | 0 | 105 | nan | 1.0692 | 0.8980 | 1.0642 |
| future_random_140 | 0 | 0 | 140 | nan | 0.9654 | 0.8109 | 0.9633 |
| future_random_35 | 0 | 0 | 35 | nan | 1.0166 | 0.8540 | 1.0159 |
| future_random_35 | 1 | 0 | 35 | nan | 1.0634 | 0.8919 | 1.0591 |
| future_random_35 | 2 | 0 | 35 | nan | 1.0216 | 0.8572 | 1.0203 |
| future_random_70 | 0 | 0 | 70 | nan | 1.0526 | 0.8835 | 1.0490 |
| future_random_70 | 1 | 0 | 70 | nan | 1.0699 | 0.8983 | 1.0651 |
| future_random_70 | 2 | 0 | 70 | nan | 1.0248 | 0.8607 | 1.0240 |
| global_random_140 | 0 | 41 | 99 | 0.3945 | 1.0144 | 0.8518 | 1.0135 |
| global_random_140 | 1 | 44 | 96 | 0.5245 | 1.0131 | 0.8509 | 1.0123 |
| global_random_140 | 2 | 35 | 105 | 0.3614 | 1.0137 | 0.8517 | 1.0130 |
| global_random_147 | 0 | 42 | 105 | 0.5194 | 1.0047 | 0.8441 | 1.0045 |
| global_random_147 | 1 | 40 | 107 | 0.3890 | 1.0549 | 0.8861 | 1.0511 |
| global_random_147 | 2 | 42 | 105 | 0.4966 | 1.0265 | 0.8618 | 1.0246 |
| right_future_100 | 0 | 0 | 140 | nan | 0.9654 | 0.8109 | 0.9633 |
