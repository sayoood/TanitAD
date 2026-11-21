# NC-FASTCAL: result tables (generated from `raw/results/result.json` and `posthoc_diag.json` by `code/gen_tables.py`)

PREREG sha256 `4db61ce178c16e1ba862e4087ff55712fb12c03ed6111a886979aeccfef8cc4f`; 15808 sets, 117 logs, 8922 tokens; paired log-cluster bootstrap B = 10000, seed 20260927.

## 1. The registered decision (PRIMARY arm C_s: the EP-led composite with the OOF per-stratum Platt NC)

| seed | dNC vs P2 (NC-unsafe rate), point [lo, hi] | (i) upper < +0.001 | dPDMS x100 vs P2, point [lo, hi] | (ii) lower >= 0 | seed meets both |
|---|---|---|---|---|---|
| 0 | +0.0029 [+0.0009, +0.0054] | False | -1.205 [-1.542, -0.882] | False | False |
| 1 | +0.0030 [+0.0009, +0.0055] | False | -1.212 [-1.549, -0.890] | False | False |

**RECOVERS = False** (both seeds must meet both conditions). C-NULL: RECOVERS fired on 0 of 5; VOID = False. **Final: NO-RECOVER.**

## 2. Every arm against P2's rule (rates over all sets; d = arm minus P2's unvetoed pick, paired, same draws)

| arm | NC-unsafe rate | dNC | DAC-unsafe rate | dDAC (no authority) | pick-level PDMS x100 | dPDMS x100 | picks changed vs P2 |
|---|---|---|---|---|---|---|---|
| P2 (rule on P2's rows) | 0.0075 [0.0052, 0.0102] | | 0.0228 | | 90.365 | | |
| U uncalibrated EP-led composite (beside), seed 0 | 0.0104 | +0.0029 [+0.0009, +0.0054] | 0.0300 | +0.0071 [+0.0044, +0.0104] | 89.219 | -1.145 [-1.473, -0.840] | 12314 |
| **C calibrated composite (PRIMARY)**, seed 0 | 0.0104 | +0.0029 [+0.0009, +0.0054] | 0.0305 | +0.0077 [+0.0049, +0.0109] | 89.160 | -1.205 [-1.542, -0.882] | 12430 |
| G global Platt (descriptive), seed 0 | 0.0103 | +0.0028 [+0.0008, +0.0054] | 0.0300 | +0.0072 [+0.0046, +0.0105] | 89.213 | -1.152 [-1.482, -0.843] | 12275 |
| U uncalibrated EP-led composite (beside), seed 1 | 0.0105 | +0.0030 [+0.0010, +0.0054] | 0.0297 | +0.0069 [+0.0043, +0.0101] | 89.191 | -1.174 [-1.503, -0.858] | 12292 |
| **C calibrated composite (PRIMARY)**, seed 1 | 0.0104 | +0.0030 [+0.0009, +0.0055] | 0.0300 | +0.0072 [+0.0045, +0.0104] | 89.152 | -1.212 [-1.549, -0.890] | 12439 |
| G global Platt (descriptive), seed 1 | 0.0103 | +0.0028 [+0.0008, +0.0054] | 0.0297 | +0.0069 [+0.0042, +0.0101] | 89.207 | -1.158 [-1.491, -0.848] | 12277 |
| PC: P2's own EP with the calibrated NC (descriptive) | 0.0081 | +0.0006 [+0.0002, +0.0011] | 0.0226 | -0.0003 [-0.0006, +0.0001] | 90.434 | +0.070 [-0.016, +0.149] | 2446 |

Picks changed by the calibration inside the composite (C vs U): seed 0 993, seed 1 981 of 15808 sets.

## 3. The calibration: P2's NC head is already near-calibrated inside every speed stratum

Fitted (a, b) by fold (5 folds), `z' = a z + b`; held-out BCE pooled over the five held-out folds, before (raw logit) and after (calibrated):

| stratum | rows | a by fold | b by fold | held-out BCE before | after | improvement (nats) |
|---|---|---|---|---|---|---|
| S | 12480 | 1.164, 1.072, 1.010, 1.047, 1.122 | +0.110, +0.210, -0.385, +0.088, +0.151 | 0.23348 | 0.25246 | -0.01898 |
| Q1 | 249808 | 1.135, 1.056, 1.051, 1.075, 1.068 | -0.128, -0.166, -0.211, -0.124, -0.154 | 0.05643 | 0.05705 | -0.00062 |
| Q2 | 249808 | 1.060, 0.991, 1.018, 1.022, 1.000 | +0.031, +0.100, -0.071, +0.018, +0.033 | 0.06084 | 0.06122 | -0.00038 |
| Q3 | 249808 | 1.094, 1.099, 1.097, 1.096, 1.035 | -0.092, -0.117, -0.137, -0.106, +0.012 | 0.11646 | 0.11626 | +0.00020 |
| Q4 | 249808 | 1.065, 1.097, 1.065, 1.079, 1.066 | -0.043, -0.052, -0.026, -0.047, -0.043 | 0.32834 | 0.32797 | +0.00037 |

Global Platt (one stratum), held-out BCE 0.14166 -> 0.14157 (+0.00009). Fit failures: 0 of 25 (stratum, fold) fits; max Newton iterations 4, max |gradient| 8.84e-11.

## 4. C-NULL: strata codes permuted within token (5 replicates; a replicate FIRES iff it reads RECOVERS)

| rep | perm seed | RECOVERS | (i) alone on both seeds | seed 0: dNC upper / dPDMS lower | seed 1: dNC upper / dPDMS lower | mean a by stratum S, Q1..Q4 |
|---|---|---|---|---|---|---|
| 0 | 20261019 | False | False | +0.0053 / -1.483 | +0.0053 / -1.488 | 1.045, 1.045, 1.031, 1.045, 1.085 |
| 1 | 20268938 | False | False | +0.0052 / -1.467 | +0.0053 / -1.484 | 1.046, 1.033, 1.028, 1.070, 1.079 |
| 2 | 20276857 | False | False | +0.0050 / -1.474 | +0.0052 / -1.493 | 1.051, 1.038, 1.039, 1.051, 1.081 |
| 3 | 20284776 | False | False | +0.0055 / -1.501 | +0.0056 / -1.506 | 1.047, 1.040, 1.029, 1.060, 1.080 |
| 4 | 20292695 | False | False | +0.0053 / -1.488 | +0.0054 / -1.531 | 1.044, 1.047, 1.036, 1.052, 1.078 |

Fires: **0 of 5**; condition (i) alone on both seeds: 0 of 5; VOID iff fires >= 2: **False**.

## 5. Q1-style numbers for the calibrated composite (descriptive, no authority)

| seed | n sets where C is NC-unsafe and P2 is not | of which own calibrated NC sigmoid < 0.5 | median calibrated sigmoid there (q25, q75) | the uncalibrated composite: n, over, median (q25, q75) |
|---|---|---|---|---|
| 0 | 84 | 0 | 0.965 (0.898, 0.984) | 82, 0, 0.960 (0.888, 0.979) |
| 1 | 85 | 0 | 0.965 (0.899, 0.984) | 83, 0, 0.959 (0.890, 0.979) |

## 6. POST-HOC, DESCRIPTIVE (written after `result.json` existed; `posthoc_diag.py`; no authority): predicted vs observed unsafe mass at the SELECTED picks

Predicted unsafe mass = sum over picks of (1 - NC sigmoid); observed = sum of (1 - NC label), labels 0 / 0.5 / 1 read as probabilities of safe. Ratio observed / predicted with the paired log-cluster bootstrap interval (same draws). Below 1 = the row is not overconfident at its picks.

| picks of | n | label-0 picks | predicted mass | observed mass | observed / predicted |
|---|---|---|---|---|---|
| C_s0 | 15808 | 164 | 172.4 | 175.5 | 1.018 [0.747, 1.333] |
| C_s1 | 15808 | 165 | 171.5 | 176.5 | 1.029 [0.759, 1.346] |
| P2 | 15808 | 118 | 158.0 | 123.5 | 0.782 [0.584, 1.027] |
| U_s0 | 15808 | 164 | 195.4 | 175.0 | 0.895 [0.664, 1.168] |
| U_s1 | 15808 | 166 | 194.5 | 177.0 | 0.910 [0.683, 1.177] |

By fixed bins of the RAW NC sigmoid at the pick (n, observed unsafe mass, predicted mass, label-0 picks):

| picks of | [0,0.5) | [0.5,0.9) | [0.9,0.97) | [0.97,0.99) | [0.99,1) |
|---|---|---|---|---|---|
| C_s0 | 2, 0.0, 1.0, 0 | 183, 37.5, 27.0, 31 | 1486, 54.0, 72.4, 51 | 3348, 59.0, 58.2, 57 | 10789, 25.0, 39.0, 25 |
| C_s1 | 2, 0.0, 1.0, 0 | 182, 37.0, 27.0, 31 | 1478, 55.5, 71.7, 52 | 3328, 59.0, 57.8, 57 | 10818, 25.0, 39.0, 25 |
| P2 | 0, 0.0, 0.0, 0 | 62, 8.0, 8.6, 7 | 1039, 30.5, 48.0, 28 | 3334, 49.0, 56.2, 47 | 11373, 36.0, 45.2, 36 |
| U_s0 | 2, 0.0, 1.0, 0 | 180, 37.0, 26.4, 31 | 1461, 55.0, 71.0, 52 | 3325, 57.0, 57.8, 55 | 10840, 26.0, 39.2, 26 |
| U_s1 | 2, 0.0, 1.0, 0 | 177, 35.5, 26.2, 30 | 1448, 56.5, 70.5, 53 | 3314, 58.0, 57.6, 56 | 10867, 27.0, 39.2, 27 |
