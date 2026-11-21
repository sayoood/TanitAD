n = 282 windows / 141 episode clusters; box = 22 designed candidates (+ seeds/proposal not used here); goal dim [64, 1024]; REAL `refa_v1._goal_term` float32 (Thor)

| metric | goal(cv) median [CI] | goal(cv) max | exactly 0 / exactly 1 | term median over box | box spread | L/R decision as % of term [CI] | iter-0 EXCLUDED @median / @max window (shipped w) |
|---|---|---|---|---|---|---|---|
| `cos` | 1.19e-07 [5.96e-08, 1.79e-07] | 0.000558 | 95 / 0 | 3.85e-05 | 0.00242 | **31.30 %** [30.51, 38.36] | **100.00 % / 100.00 %** |
| `chord` | 0.000363 [9.47e-08, 0.00055] | 0.0334 | 0 / 0 | 0.00878 | 0.0684 | **8.48 %** [0.79, 15.56] | **100.00 % / 100.00 %** |
| `ccos` | 1 [1, 1] | 1.18 | 0 / 133 | 1 | 1.95 | **189.39 %** [0.00, 191.19] | **1.67 % / 0.33 %** |

| form vs `cos` | value leverage (median cell ratio) | decision leverage (median box-spread ratio) [CI] | L/R-gap ratio | compensated triple (decision) | iter-0 excluded @median, compensated decision / value |
|---|---|---|---|---|---|
| `chord` | 178.8x | **28.16x** [26.5, 29.9] | 107.9x | (0.5632, 1.408, 2.816) | 100.00 % / 100.00 % |
| `ccos` | 5,165.6x | **128.63x** [0.0, 318.2] | 123,650.4x | (2.573, 6.432, 12.86) | 100.00 % / 100.00 % |

Controls (real function, real fields): zero-model ptp max = `cos` 0, `chord` 0, `ccos` 0; identity max|.| = `cos` 2.38e-07, `chord` 0, `ccos` 1; ccos(cv) scored alone vs z_ref: exactly 1.0 on 133/282 (min 0.777159, max 1.176566); float64 cross-check max|ccos32-ccos64| = 2.49e-07; ||g-z_ref||/||z_ref|| median 0.000364, < 1e-6 on 133/282 windows (the HOLD stratum).
Second probe (independently written): matched 282 windows; max|Δ| `cos` 1.79e-07, `chord` 2.24e-07, `ccos` 0.0314.
