# VERDICT_WMCEIL: H-V8F-PROOF-3 (is the learned world model below the closed-form linear predictor?)

MEASURED, T0 (a world-model diagnostic, never driving performance). Raw: RESULT_WMCEIL.json. eval-141, 1976 windows, 141 episodes, 5 folds by episode (seed 0); episode-cluster bootstrap B 2,000.
Smoke run: False

**Control (c): learned skill by the banked rule vs RESULT_WMACT.json, max abs diff over the four headline cells = 0.000000; OK (<= 0.01).**
**Outcome: WM-BELOW-LINEAR**   JUMP-DOMINATES: True
Next lever (card): write the ridge solution into the predictor as its base (persistence + linear + learned residual) under its own card

## M2b_s0
| horizon | learned | SKILL_lin (z, a) | SKILL_lin (z) | learned - lin (z, a) | CI separated | label |
|---|---|---|---|---|---|---|
| 0.2 s (L1/k1) | +0.0010 | +0.0222 | +0.0218 | -0.0212 | True | BELOW |
| 0.4 s (L2/k1) | +0.0295 | +0.0272 | +0.0246 | +0.0022 | False | AT-CEILING |
| 0.8 s (L3/k1) | +0.1316 | +0.0752 | +0.0545 | +0.0564 | True | ABOVE |
| 2.0 s (L1/k10) | -0.1500 | +0.1225 | +0.0894 | -0.2725 | True | BELOW |

rho of the target latents (per-dimension; mean / median / p10 / p90) and the AR(1) ceiling (1 - rho)/2:
| cell | rho mean | median | p10 | p90 | ceiling of mean rho | measured SKILL_lin (z) |
|---|---|---|---|---|---|---|
| L1/k1 (0.2 s) | 0.975 | 0.976 | 0.965 | 0.983 | +0.0127 | +0.0218 |
| L1/k5 (1.0 s) | 0.931 | 0.934 | 0.903 | 0.955 | +0.0345 | +0.0548 |
| L1/k10 (2.0 s) | 0.883 | 0.887 | 0.836 | 0.927 | +0.0583 | +0.0894 |
| L2/k1 (0.4 s) | 0.952 | 0.955 | 0.935 | 0.964 | +0.0240 | +0.0246 |
| L2/k3 (1.2 s) | 0.893 | 0.896 | 0.853 | 0.922 | +0.0534 | +0.0569 |
| L3/k1 (0.8 s) | 0.914 | 0.919 | 0.882 | 0.938 | +0.0430 | +0.0545 |
| L3/k3 (2.4 s) | 0.793 | 0.802 | 0.727 | 0.846 | +0.1033 | +0.1084 |

(d) direct 2 s ridge vs 10-fold composed 0.2 s ridge (level 1), ratio = eps_k / (k eps_1), per fold:
* inputs za: pooled ratio 0.416 (boot 95% [0.389, 0.449]); per fold 0.395, 0.401, 0.399, 0.426, 0.457; composed above direct on every fold: True; JUMP-DOMINATES True
* inputs z: pooled ratio 0.431 (boot 95% [0.401, 0.466]); per fold 0.414, 0.416, 0.418, 0.437, 0.469; composed above direct on every fold: False; JUMP-DOMINATES False

## M2b_s1
| horizon | learned | SKILL_lin (z, a) | SKILL_lin (z) | learned - lin (z, a) | CI separated | label |
|---|---|---|---|---|---|---|
| 0.2 s (L1/k1) | -0.0010 | +0.0215 | +0.0211 | -0.0225 | True | BELOW |
| 0.4 s (L2/k1) | +0.0264 | +0.0265 | +0.0236 | -0.0001 | False | AT-CEILING |
| 0.8 s (L3/k1) | +0.1351 | +0.0752 | +0.0578 | +0.0599 | True | ABOVE |
| 2.0 s (L1/k10) | -0.1724 | +0.1292 | +0.0915 | -0.3016 | True | BELOW |

rho of the target latents (per-dimension; mean / median / p10 / p90) and the AR(1) ceiling (1 - rho)/2:
| cell | rho mean | median | p10 | p90 | ceiling of mean rho | measured SKILL_lin (z) |
|---|---|---|---|---|---|---|
| L1/k1 (0.2 s) | 0.976 | 0.976 | 0.966 | 0.985 | +0.0121 | +0.0211 |
| L1/k5 (1.0 s) | 0.932 | 0.935 | 0.908 | 0.957 | +0.0340 | +0.0570 |
| L1/k10 (2.0 s) | 0.885 | 0.886 | 0.841 | 0.929 | +0.0576 | +0.0915 |
| L2/k1 (0.4 s) | 0.952 | 0.953 | 0.939 | 0.966 | +0.0239 | +0.0236 |
| L2/k3 (1.2 s) | 0.892 | 0.894 | 0.863 | 0.925 | +0.0541 | +0.0679 |
| L3/k1 (0.8 s) | 0.911 | 0.913 | 0.884 | 0.936 | +0.0447 | +0.0578 |
| L3/k3 (2.4 s) | 0.781 | 0.788 | 0.723 | 0.839 | +0.1097 | +0.1189 |

(d) direct 2 s ridge vs 10-fold composed 0.2 s ridge (level 1), ratio = eps_k / (k eps_1), per fold:
* inputs za: pooled ratio 0.433 (boot 95% [0.403, 0.467]); per fold 0.420, 0.412, 0.416, 0.447, 0.467; composed above direct on every fold: True; JUMP-DOMINATES True
* inputs z: pooled ratio 0.451 (boot 95% [0.418, 0.490]); per fold 0.442, 0.426, 0.437, 0.461, 0.489; composed above direct on every fold: False; JUMP-DOMINATES False

Evidence class: MEASURED (this script; files above). Tier T0. STRATEGIC family: PAUSED (PI item 25), not part of this read.
Implementation choices (card silent): cells = the banked cells (level-native steps 0.2 / 0.4 / 0.8 s; 2.0 s = L1 k10); start frame = window frame 0, windows with 16 real frames (as the banked reader); action inputs = WM-scaled raw actions at L1 and the stack's macro-action embeddings at L2 / L3 (as the WM gets them); lambda from an inner 4-fold episode split of the training folds; the one-step ridge in (d) uses all adjacent pairs of the training folds; AT-CEILING read at two or more horizons like BELOW.
