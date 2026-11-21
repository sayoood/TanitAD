# RESULT tables: EP-RELIAB-PICKS (generated from `raw/results/result.json` by `code/gen_tables.py`)

prereg sha256 `e54067d4b16f598c1710064a7f01eafe825d19c01eadbe7124b86fa67e9a8e48` + Amendment 1; 15808 sets, 117 logs; paired log-cluster bootstrap B = 10000, seed 20260927. Tier: SELECTION, SEEN scenes, P2 IN-SAMPLE; label-lookup PDMS; no GPU.

**FINAL LABEL: EP-OVERPREDICTED**  (deciding form: MASS; the other form reads EP-OVERPREDICTED; form-sensitive: False; dominant PDMS-loss term: DAC).

## 1. Scale gate and the two decision statistics

| seed | n_changed | within-set slope of EP label on sigma (safe proposals) | on-scale | Delta = predicted - realised gain | class | D_M = M_p - M_e | class |
|---|---|---|---|---|---|---|---|
| 0 | 12314 | 0.309 [0.297, 0.321] | False | +0.1346 [+0.1132, +0.1596] | OVER | +0.1349 [+0.1114, +0.1640] | OVER |
| 1 | 12292 | 0.311 [0.299, 0.322] | False | +0.1332 [+0.1119, +0.1579] | OVER | +0.1331 [+0.1101, +0.1618] | OVER |

Scale gate: ON-SCALE iff the slope CI lies inside [0.9, 1.1] on BOTH seeds. Result: OFF-SCALE (the rank-free mass form D_M decides; no mapping fitted).

## 2. The gain, predicted and realised (EP on the sigmoid scale = label scale)

| seed | predicted gain (composite's EP row) | realised gain (labels6 EP) | Delta | M_e (obs/pred EP mass at composite picks) | M_p (at P2 picks) | predicted gain from P2's own EP row |
|---|---|---|---|---|---|---|
| 0 | +0.1240 [+0.0991, +0.1537] | -0.0106 [-0.0156, -0.0042] | +0.1346 [+0.1132, +0.1596] | 0.845 [0.834, 0.855] | 0.980 [0.962, 1.002] | -0.0071 [-0.0097, -0.0042] |
| 1 | +0.1219 [+0.0972, +0.1513] | -0.0113 [-0.0165, -0.0048] | +0.1332 [+0.1119, +0.1579] | 0.844 [0.833, 0.854] | 0.977 [0.959, 0.999] | -0.0076 [-0.0102, -0.0046] |

Changed sets only (descriptive, no authority):

| seed | n_changed | predicted gain | realised gain | Delta |
|---|---|---|---|---|
| 0 | 12314 | +0.1592 [+0.1272, +0.1972] | -0.0136 [-0.0200, -0.0054] | +0.1728 [+0.1454, +0.2048] |
| 1 | 12292 | +0.1568 [+0.1248, +0.1945] | -0.0145 [-0.0211, -0.0062] | +0.1713 [+0.1439, +0.2029] |

## 3. Exact attribution of dPDMS (composite - P2), x100, over all 15,808 sets

Sequential substitution in the fixed order NC, DAC, EP, TTC, C (the EP, TTC and C terms are valued at the composite's NC and DAC multipliers). Weights (5, 5, 2) / 12: `pickrule_lib.V1_W` = `refe/planner.py:520`.

| seed | term | sequential contribution | share of the loss | Shapley (descriptive) |
|---|---|---|---|---|
| 0 | NC | -0.3886 [-0.5977, -0.2285] | 0.339 [0.214, 0.493] | -0.1536 [-0.2697, -0.0615] |
| 0 | DAC | -0.7209 [-1.0205, -0.4713] | 0.629 [0.418, 0.920] | -0.4784 [-0.7090, -0.2859] |
| 0 | EP | +0.0758 [-0.2211, +0.4269] | -0.066 [-0.405, 0.195] | -0.2933 [-0.5282, -0.0090] |
| 0 | TTC | -0.1160 [-0.2731, +0.0365] | 0.101 [-0.038, 0.208] | -0.2245 [-0.3996, -0.0632] |
| 0 | C | +0.0042 [+0.0000, +0.0100] | -0.004 [-0.010, 0.000] | +0.0042 [+0.0000, +0.0100] |
| 0 | **total dPDMS** | -1.1455 [-1.4730, -0.8398] | 1 | -1.1455 [-1.4730, -0.8398] |
| 1 | NC | -0.3924 [-0.5989, -0.2322] | 0.334 [0.212, 0.486] | -0.1602 [-0.2731, -0.0689] |
| 1 | DAC | -0.6994 [-0.9901, -0.4565] | 0.596 [0.395, 0.877] | -0.4557 [-0.6790, -0.2673] |
| 1 | EP | +0.0442 [-0.2555, +0.3997] | -0.038 [-0.376, 0.221] | -0.3250 [-0.5642, -0.0345] |
| 1 | TTC | -0.1292 [-0.2875, +0.0267] | 0.110 [-0.028, 0.216] | -0.2359 [-0.4127, -0.0696] |
| 1 | C | +0.0032 [+0.0000, +0.0071] | -0.003 [-0.007, 0.000] | +0.0032 [+0.0000, +0.0071] |
| 1 | **total dPDMS** | -1.1736 [-1.5035, -0.8579] | 1 | -1.1736 [-1.5035, -0.8579] |

Dominant term (largest share of the loss), per seed, and how often it is the largest in the bootstrap draws:

| seed | dominant | bootstrap frequency by term |
|---|---|---|
| 0 | DAC | C 0.000, DAC 0.988, EP 0.000, NC 0.012, NO-LOSS 0.000, TTC 0.000 |
| 1 | DAC | C 0.000, DAC 0.983, EP 0.001, NC 0.016, NO-LOSS 0.000, TTC 0.000 |

Arm-level dominant term: **DAC** (by seed: ['DAC', 'DAC']).

## 4. Reliability at the picks (observed / predicted mass; rows of the composite; descriptive, no authority)

Primary form: deficit (risk) `sum(1-y)/sum(1-sigma)` for NC, DAC, TTC; positive `sum(y)/sum(sigma)` for EP.

| seed | head | primary form | at composite's picks | at P2's picks | composite - P2 (paired) |
|---|---|---|---|---|---|
| 0 | NC | deficit | 0.895 [0.664, 1.168] | 0.782 [0.584, 1.027] | +0.1136 [-0.0353, +0.2696] |
| 0 | DAC | deficit | 0.947 [0.837, 1.084] | 0.933 [0.787, 1.125] | +0.0142 [-0.0764, +0.0829] |
| 0 | EP | positive | 0.845 [0.834, 0.855] | 0.980 [0.962, 1.002] | -0.1349 [-0.1640, -0.1114] |
| 0 | TTC | deficit | 0.925 [0.762, 1.089] | 0.885 [0.708, 1.070] | +0.0400 [-0.0507, +0.1376] |
| 1 | NC | deficit | 0.910 [0.683, 1.177] | 0.782 [0.584, 1.027] | +0.1282 [-0.0216, +0.2828] |
| 1 | DAC | deficit | 0.941 [0.830, 1.078] | 0.933 [0.787, 1.125] | +0.0083 [-0.0826, +0.0769] |
| 1 | EP | positive | 0.844 [0.833, 0.854] | 0.977 [0.959, 0.999] | -0.1331 [-0.1618, -0.1101] |
| 1 | TTC | deficit | 0.936 [0.770, 1.101] | 0.885 [0.708, 1.070] | +0.0506 [-0.0423, +0.1506] |

Observed and predicted unsafe masses behind the safety rows at the picks (seed 0; the NC, DAC, TTC rows are P2's):

| head | picks | observed unsafe mass | predicted unsafe mass | observed per set |
|---|---|---|---|---|
| NC | composite_picks | 175.0 | 195.4 | 0.01107 |
| NC | P2_picks | 123.5 | 158.0 | 0.00781 |
| DAC | composite_picks | 474.0 | 500.6 | 0.02998 |
| DAC | P2_picks | 361.0 | 387.0 | 0.02284 |
| TTC | composite_picks | 682.0 | 737.0 | 0.04314 |
| TTC | P2_picks | 568.0 | 641.5 | 0.03593 |

EP row of P2 itself at P2's picks (what P2 believed): positive form 0.989 [0.981, 0.997].

## 5. Controls

| control | pass |
|---|---|
| C-ATTR[seed0] | True |
| C-ATTR[seed1] | True |
| C-ID | True |
| C-ID-LANDED[seed0] | True |
| C-ID-LANDED[seed1] | True |
| C-ID-R53[seed0] | True |
| C-ID-R53[seed1] | True |
| C-ID-RULE | True |
| C-PDMS | True |
| C-POWER | True |
| C-REACH | True |
| C-REACH-DECISION | True |
| C-SCALE | True |

C-NULL (picks shuffled within token, 5 replicates, deciding form MASS): fired on **1 of 5** (VOID iff >= 2): void = False.

| rep | perm seed | outcome (deciding form) | fires | D_M point seed 0 / seed 1 |
|---|---|---|---|---|
| 0 | 20261019 | EP-WITHIN-NOISE | False | -0.0021 / -0.0002 |
| 1 | 20268938 | EP-WITHIN-NOISE | False | +0.0028 / +0.0026 |
| 2 | 20276857 | EP-WITHIN-NOISE | False | -0.0048 / -0.0049 |
| 3 | 20284776 | EP-WITHIN-NOISE | False | +0.0084 / +0.0074 |
| 4 | 20292695 | EP-UNDERPREDICTED | True | -0.0069 / -0.0075 |

E-NULL (descriptive, 200 more replicates): empirical any-fire rate of the registered rule on the real structure **0.040** (other form 0.040); P(>= 2 of 5 fire) at that rate 0.0148.

| seed | null point mean | null point sd | null 2.5 / 97.5 % | real point | real in null-sd units | empirical two-sided p (+1) |
|---|---|---|---|---|---|---|
| 0 | -0.00015 | 0.00376 | -0.0075 / +0.0074 | +0.1349 | 35.8 | 0.0050 |
| 1 | -0.00015 | 0.00374 | -0.0074 / +0.0077 | +0.1331 | 35.6 | 0.0050 |

Independent re-derivation (`code/independent_check.py`, no shared code, own picks and loops): max |independent - library| over predicted gain, realised gain, Delta, M_e, M_p, D_M, slope, dPDMS and the five sequential terms = seed 0: 1.78e-15, seed 1: 7.22e-16.

Minimal detectable effect (descriptive): half-width of the Delta interval seed 0: 0.0232, seed 1: 0.0230; of the D_M interval seed 0: 0.0263, seed 1: 0.0259.
