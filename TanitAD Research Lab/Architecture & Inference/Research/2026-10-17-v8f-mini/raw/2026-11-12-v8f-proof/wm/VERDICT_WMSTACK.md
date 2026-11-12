# VERDICT_WMSTACK: PREREG_V8F_PROOF_ADD2 (WMSTACK and JUMPCURVE)

MEASURED (wmstack.py; RESULT_WMSTACK.json), tier T0: a world-model diagnostic, never driving performance. eval-141, 1,976 windows, 141 episodes, 5 episode folds, episode-cluster bootstrap B 2,000.
STRATEGIC family: not applicable to a world-model diagnostic.

**WMSTACK: STACK-ADDS** (horizons where the stack adds, per seed: {'M2b_s0': 4, 'M2b_s1': 4}).  **JUMPCURVE: JUMP-MONOTONE.**  Controls (k1 shuffled change <= 0.005; k2 fixed-coefficient stack = learned exactly): ALL OK.

## M2b_s0
| horizon | learned | linear (z, a) | stack | stack - best [95% CI] | adds | k1 shuffled | k2 diff |
|---|---|---|---|---|---|---|---|
| 0.2 s (L1/k1) | +0.0010 | +0.0222 | +0.0402 | +0.0181 [+0.0130, +0.0228] | True | +0.0020 | 0.0e+00 |
| 0.4 s (L2/k1) | +0.0295 | +0.0272 | +0.0619 | +0.0325 [+0.0233, +0.0432] | True | +0.0008 | 0.0e+00 |
| 0.8 s (L3/k1) | +0.1316 | +0.0752 | +0.1531 | +0.0215 [+0.0123, +0.0322] | True | +0.0015 | 0.0e+00 |
| 2.0 s (L1/k10) | -0.1500 | +0.1225 | +0.1710 | +0.0485 [+0.0357, +0.0602] | True | +0.0027 | 0.0e+00 |

| k (s) | ratio(k) = eps_k/(k eps_1) pooled | composed/direct pooled | ratio per fold | composed/direct per fold |
|---|---|---|---|---|
| 1 (0.2) | 1.005 | 0.995 | 1.008, 1.005, 1.002, 1.004, 1.006 | 0.992, 0.995, 0.998, 0.996, 0.994 |
| 2 (0.4) | 0.743 | 0.994 | 0.728, 0.722, 0.720, 0.780, 0.764 | 0.997, 0.996, 0.990, 0.997, 0.990 |
| 4 (0.8) | 0.573 | 1.013 | 0.545, 0.531, 0.565, 0.587, 0.636 | 1.024, 1.024, 1.006, 1.025, 0.992 |
| 5 (1.0) | 0.521 | 1.019 | 0.488, 0.485, 0.519, 0.541, 0.571 | 1.035, 1.038, 1.003, 1.035, 0.991 |
| 10 (2.0) | 0.416 | 1.067 | 0.395, 0.401, 0.399, 0.426, 0.457 | 1.111, 1.090, 1.045, 1.102, 1.002 |
monotone (ratio < 1 for k >= 2 and non-increasing over k = 2, 4, 5, 10): True

## M2b_s1
| horizon | learned | linear (z, a) | stack | stack - best [95% CI] | adds | k1 shuffled | k2 diff |
|---|---|---|---|---|---|---|---|
| 0.2 s (L1/k1) | -0.0010 | +0.0215 | +0.0400 | +0.0185 [+0.0130, +0.0236] | True | +0.0017 | 0.0e+00 |
| 0.4 s (L2/k1) | +0.0264 | +0.0265 | +0.0585 | +0.0320 [+0.0236, +0.0410] | True | +0.0010 | 0.0e+00 |
| 0.8 s (L3/k1) | +0.1351 | +0.0752 | +0.1561 | +0.0210 [+0.0105, +0.0330] | True | +0.0004 | 0.0e+00 |
| 2.0 s (L1/k10) | -0.1724 | +0.1292 | +0.1641 | +0.0349 [+0.0239, +0.0463] | True | +0.0023 | 0.0e+00 |

| k (s) | ratio(k) = eps_k/(k eps_1) pooled | composed/direct pooled | ratio per fold | composed/direct per fold |
|---|---|---|---|---|
| 1 (0.2) | 1.006 | 0.994 | 1.004, 1.006, 1.005, 1.007, 1.007 | 0.996, 0.994, 0.995, 0.993, 0.993 |
| 2 (0.4) | 0.745 | 0.993 | 0.745, 0.722, 0.712, 0.790, 0.759 | 0.996, 0.997, 0.989, 0.994, 0.990 |
| 4 (0.8) | 0.586 | 1.011 | 0.564, 0.537, 0.576, 0.613, 0.637 | 1.019, 1.026, 1.007, 1.017, 0.992 |
| 5 (1.0) | 0.539 | 1.016 | 0.508, 0.495, 0.538, 0.576, 0.578 | 1.027, 1.041, 1.005, 1.020, 0.992 |
| 10 (2.0) | 0.433 | 1.059 | 0.420, 0.412, 0.416, 0.447, 0.467 | 1.076, 1.089, 1.050, 1.081, 1.010 |
monotone (ratio < 1 for k >= 2 and non-increasing over k = 2, 4, 5, 10): True

Choices (card silent): linear = ridge on (z_t, the WM action inputs) as in WMCEIL; better single = the higher point skill of learned / linear; monotonicity read over k >= 2 (ratio(1) is shown, near 1 by construction); lambda from an inner 4-fold episode split; k1 shuffles the change across windows (same stack inputs); the stack at 2.0 s takes the composed ten-step rollout as its learned input.
