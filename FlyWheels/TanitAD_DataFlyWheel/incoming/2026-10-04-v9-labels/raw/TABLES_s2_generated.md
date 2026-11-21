## E1 — trivial planner floor (constant v0, arc through the checkpoint)

### EVAL-DIAG (route package grid; bank GT; windows where all four variants are valid) — n: {'turn': 107, 'turnL': 40, 'turnR': 67, 'straight': 574, 'gentle': 104, 'all_classified': 785}

| arm | turn ADE m | turn dir-correct | turn heading ≤15° | straight ADE m | straight heading ≤15° | all ADE m |
|---|---|---|---|---|---|---|
| CV_straight | 9.640 [8.215, 11.051] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 2.599 | 1.000 | 3.917 |
| TP_A30 | 4.182 [3.564, 4.876] | 0.888 [0.827, 0.945] | 0.477 [0.351, 0.598] | 2.420 | 0.963 | 2.800 |
| TP_A30_gtspeed | 1.871 [1.515, 2.228] | 0.888 [0.829, 0.939] | 0.439 [0.327, 0.556] | 0.353 | 0.981 | 0.660 |
| TP_A30_noised | 4.313 [3.728, 4.979] | 0.860 [0.783, 0.926] | 0.486 [0.364, 0.607] | 3.502 | 0.902 | 3.706 |
| CTRL_TP_A30_deranged | 10.180 [8.151, 12.226] | 0.206 [0.101, 0.310] | 0.028 [0.000, 0.062] | 7.688 | 0.728 | 8.033 |
| TP_A50 | 4.588 [3.850, 5.454] | 0.916 [0.848, 0.970] | 0.654 [0.531, 0.763] | 2.410 | 0.965 | 2.837 |
| TP_A50_gtspeed | 2.214 [1.762, 2.737] | 0.944 [0.891, 0.988] | 0.626 [0.521, 0.731] | 0.316 | 0.976 | 0.661 |
| TP_A50_noised | 4.656 [3.934, 5.507] | 0.925 [0.846, 0.983] | 0.579 [0.467, 0.685] | 2.733 | 0.960 | 3.094 |
| CTRL_TP_A50_deranged | 10.638 [8.491, 12.771] | 0.178 [0.074, 0.287] | 0.000 [0.000, 0.000] | 7.713 | 0.730 | 8.071 |
| TP_A80 | 5.503 [4.560, 6.567] | 0.953 [0.872, 1.000] | 0.542 [0.400, 0.667] | 2.501 | 0.941 | 3.059 |
| TP_A80_gtspeed | 3.251 [2.543, 4.048] | 0.953 [0.899, 0.992] | 0.617 [0.491, 0.723] | 0.455 | 0.951 | 0.951 |
| TP_A80_noised | 5.544 [4.596, 6.610] | 0.944 [0.861, 1.000] | 0.551 [0.415, 0.673] | 2.577 | 0.943 | 3.127 |
| CTRL_TP_A80_deranged | 10.642 [8.432, 12.830] | 0.150 [0.057, 0.250] | 0.019 [0.000, 0.049] | 7.719 | 0.723 | 8.064 |
| TP_B | 4.773 [4.007, 5.629] | 0.972 [0.926, 1.000] | 0.626 [0.504, 0.736] | 2.434 | 0.946 | 2.887 |
| TP_B_gtspeed | 2.480 [1.877, 3.178] | 0.963 [0.913, 1.000] | 0.636 [0.511, 0.742] | 0.395 | 0.958 | 0.784 |
| TP_B_noised | 4.792 [4.019, 5.647] | 0.944 [0.867, 1.000] | 0.579 [0.447, 0.688] | 2.527 | 0.943 | 2.959 |
| CTRL_TP_B_deranged | 10.572 [8.356, 12.770] | 0.159 [0.060, 0.264] | 0.009 [0.000, 0.033] | 8.023 | 0.721 | 8.306 |
| REFCV7_E9_pick | 3.186 [2.712, 3.701] | 0.841 [0.750, 0.918] | 0.514 [0.427, 0.598] | 1.602 | 0.981 | 1.966 |

### all eval139 windows (log GT at 6 s, all variants valid) — n: {'turn': 2795, 'turnL': 1072, 'turnR': 1723, 'straight': 16029, 'gentle': 3017, 'all_classified': 21841}

| arm | turn ADE m | turn dir-correct | turn heading ≤15° | straight ADE m | straight heading ≤15° | all ADE m |
|---|---|---|---|---|---|---|
| CV_straight | 9.197 [7.804, 10.492] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 2.424 | 1.000 | 3.728 |
| TP_A30 | 4.058 [3.558, 4.600] | 0.901 [0.855, 0.939] | 0.449 [0.360, 0.542] | 2.239 | 0.965 | 2.671 |
| TP_A30_gtspeed | 1.816 [1.557, 2.087] | 0.912 [0.867, 0.950] | 0.464 [0.373, 0.556] | 0.341 | 0.975 | 0.652 |
| TP_A30_noised | 4.230 [3.775, 4.743] | 0.896 [0.843, 0.938] | 0.438 [0.356, 0.517] | 3.494 | 0.882 | 3.683 |
| CTRL_TP_A30_deranged | 9.869 [8.044, 11.696] | 0.177 [0.086, 0.275] | 0.025 [0.006, 0.051] | 7.976 | 0.726 | 8.197 |
| TP_A50 | 4.322 [3.711, 5.022] | 0.937 [0.888, 0.978] | 0.640 [0.536, 0.731] | 2.219 | 0.964 | 2.685 |
| TP_A50_gtspeed | 1.971 [1.647, 2.342] | 0.956 [0.910, 0.990] | 0.610 [0.518, 0.697] | 0.296 | 0.970 | 0.616 |
| TP_A50_noised | 4.369 [3.761, 5.053] | 0.934 [0.879, 0.979] | 0.635 [0.532, 0.726] | 2.573 | 0.962 | 2.964 |
| CTRL_TP_A50_deranged | 10.161 [8.324, 11.977] | 0.152 [0.066, 0.245] | 0.013 [0.002, 0.026] | 7.978 | 0.734 | 8.176 |
| TP_A80 | 5.033 [4.286, 5.877] | 0.925 [0.854, 0.975] | 0.547 [0.416, 0.659] | 2.319 | 0.945 | 2.890 |
| TP_A80_gtspeed | 2.889 [2.351, 3.493] | 0.955 [0.911, 0.990] | 0.600 [0.474, 0.705] | 0.439 | 0.952 | 0.896 |
| TP_A80_noised | 5.062 [4.315, 5.908] | 0.923 [0.852, 0.976] | 0.545 [0.414, 0.655] | 2.417 | 0.945 | 2.969 |
| CTRL_TP_A80_deranged | 10.223 [8.353, 12.121] | 0.125 [0.052, 0.207] | 0.013 [0.002, 0.027] | 8.014 | 0.720 | 8.232 |
| TP_B | 4.397 [3.781, 5.033] | 0.941 [0.889, 0.982] | 0.625 [0.521, 0.715] | 2.252 | 0.953 | 2.728 |
| TP_B_gtspeed | 2.077 [1.626, 2.572] | 0.955 [0.907, 0.992] | 0.672 [0.570, 0.765] | 0.373 | 0.965 | 0.722 |
| TP_B_noised | 4.436 [3.810, 5.079] | 0.937 [0.883, 0.982] | 0.615 [0.509, 0.708] | 2.354 | 0.953 | 2.812 |
| CTRL_TP_B_deranged | 10.182 [8.285, 12.041] | 0.127 [0.052, 0.212] | 0.013 [0.002, 0.028] | 8.303 | 0.722 | 8.467 |

GT cross-check (log vs bank, EVAL-DIAG): {"n_slots": 8339, "median_m": 0.1302, "p95_m": 0.653, "max_m": 1.4373, "v0_abs_diff_median": 0.0015}

## E2 — speed leak: recovered fraction ρ of the future-speed information v0 lacks (mean over τ = 1..6 s)

| population · instrument | certified | ρ NAV | RC A30 / NAV+RC−NAV | RC A50 / NAV+RC−NAV | RC A80 / NAV+RC−NAV | RC B / NAV+RC−NAV | ρ deranged | ρ O1 oracle | ρ target |
|---|---|---|---|---|---|---|---|---|---|
| eval139_ols (n 21348, 132 clips) | True | +0.027 | +0.005 / +0.007 PASS | +0.031 / +0.011 PASS | +0.041 / +0.008 PASS | +0.015 / -0.006 PASS | +0.001 | +0.282 | +1.000 |
| eval139_hgb1 (n 21348, 132 clips) | False | -0.059 | +0.084 / +0.126 FAIL | +0.012 / +0.025 FAIL | -0.048 / -0.011 PASS | -0.015 / +0.000 FAIL | -0.213 | +0.336 | +0.938 |
| eval139_hgb2 (n 21348, 132 clips) | False | +0.007 | +0.079 / +0.069 FAIL | +0.019 / +0.005 PASS | -0.047 / -0.025 PASS | +0.001 / -0.026 PASS | -0.134 | +0.300 | +0.912 |
| trainS_ols (n 92579, 576 clips) | True | +0.040 | +0.005 / +0.010 PASS | +0.007 / +0.006 PASS | +0.016 / +0.009 PASS | +0.001 / +0.010 PASS | -0.003 | +0.329 | +1.000 |
| trainS_hgb1 (n 92579, 576 clips) | False | +0.149 | +0.211 / +0.131 FAIL | +0.166 / +0.089 FAIL | +0.147 / +0.070 FAIL | +0.169 / +0.058 FAIL | -0.059 | +0.381 | +0.997 |
| trainS_hgb2 (n 92579, 576 clips) | False | +0.171 | +0.211 / +0.119 FAIL | +0.167 / +0.078 FAIL | +0.149 / +0.059 FAIL | +0.174 / +0.052 FAIL | -0.044 | +0.382 | +0.997 |

### E2 levers (SPEC E4 order) — NAV+RC−NAV, trainS, both tree instruments (the decisive population)

| arm | A30 hgb1 / hgb2 | A50 hgb1 / hgb2 | A80 hgb1 / hgb2 | B hgb1 / hgb2 |
|---|---|---|---|---|
| REG_s8 | +0.131 / +0.119 | +0.089 / +0.078 | +0.070 / +0.059 | +0.058 / +0.052 |
| L1_s15 | +0.106 / +0.090 | +0.083 / +0.071 | +0.069 / +0.059 | +0.062 / +0.052 |
| L2_s8_noheading | +0.117 / +0.105 | +0.083 / +0.072 | +0.067 / +0.059 | +0.048 / +0.043 |
| L3_s8_noised | +0.043 / +0.036 | +0.050 / +0.043 | +0.050 / +0.045 | +0.040 / +0.036 |
| DIAG_L1L2_s15_noheading | +0.099 / +0.088 | +0.077 / +0.066 | +0.067 / +0.057 | +0.047 / +0.043 |
| DIAG heavy σ25 (A50 only) | — | +0.074 / +0.063 | — | — |

## E3 — lateral leak |δ| of the checkpoint from the σ = 25 m route at the same arc (m)

| population | variant | all p90 (registered bar ≤ 1.0) | straight-support p90 (A1 bar ≤ 1.0) / share | curved-support p90 | straight: LC-scale vs none, median | dev6 ΔR² (input over heavy route) |
|---|---|---|---|---|---|---|
| eval139 | A30 | 3.856 FAIL | 0.332 PASS / 0.49 | 5.512 | 0.141 vs 0.051 | 0.0529 |
| eval139 | A50 | 3.633 FAIL | 0.340 PASS / 0.46 | 5.519 | 0.157 vs 0.051 | 0.0193 |
| eval139 | A80 | 3.020 FAIL | 0.310 PASS / 0.42 | 4.576 | 0.142 vs 0.047 | -0.0362 |
| eval139 | B | 5.611 FAIL | 0.328 PASS / 0.42 | 6.893 | 0.147 vs 0.047 | 0.0351 |
| trainS | A30 | 4.342 FAIL | 0.334 PASS / 0.44 | 6.018 | 0.143 vs 0.050 | 0.0049 |
| trainS | A50 | 3.772 FAIL | 0.302 PASS / 0.41 | 5.744 | 0.127 vs 0.047 | 0.0161 |
| trainS | A80 | 3.195 FAIL | 0.256 PASS / 0.37 | 4.842 | 0.106 vs 0.043 | 0.0130 |
| trainS | B | 5.502 FAIL | 0.274 PASS / 0.37 | 6.451 | 0.111 vs 0.044 | 0.0270 |

## E4 — decision (rule fixed in SPEC §9.E + addendum S2-A1)

```
{
 "per_variant": {
  "A30": {
   "E2_eval139": false,
   "E2_trainS": false,
   "E2_used": false,
   "E3_as_registered": false,
   "E3_A1": true,
   "E3_failure_localised_on_curved_support": true,
   "E3_used": true,
   "TP_turn_h15_allwindows": 0.44865831842576026,
   "eligible": false
  },
  "A50": {
   "E2_eval139": false,
   "E2_trainS": false,
   "E2_used": false,
   "E3_as_registered": false,
   "E3_A1": true,
   "E3_failure_localised_on_curved_support": true,
   "E3_used": true,
   "TP_turn_h15_allwindows": 0.6400715563506261,
   "eligible": false
  },
  "A80": {
   "E2_eval139": false,
   "E2_trainS": false,
   "E2_used": false,
   "E3_as_registered": false,
   "E3_A1": true,
   "E3_failure_localised_on_curved_support": true,
   "E3_used": true,
   "TP_turn_h15_allwindows": 0.5470483005366726,
   "eligible": false
  },
  "B": {
   "E2_eval139": false,
   "E2_trainS": false,
   "E2_used": false,
   "E3_as_registered": false,
   "E3_A1": true,
   "E3_failure_localised_on_curved_support": true,
   "E3_used": true,
   "TP_turn_h15_allwindows": 0.625402504472272,
   "eligible": false
  }
 },
 "eligible": [],
 "recommended": null,
 "rule": "among variants passing E2 (eval139, both models; trainS replicate) and E3 (as registered, or E3-A1 when the failure is localised on curved-support windows), the highest TP heading-within-15 deg on GT-turn windows (all eval139 windows); ties -> larger L"
}
```
