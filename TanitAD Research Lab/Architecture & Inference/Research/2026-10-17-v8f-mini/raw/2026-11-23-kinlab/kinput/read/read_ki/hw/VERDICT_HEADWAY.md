# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: F:/TanitAD-scratch/strat1/read/inputs/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 28, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.01779359430604982, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0C10XHb20:I10B0_R0C10XHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0C10XHb20:I10KI_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0G50Hb20:I10KI_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0C10Hb20:I10KI_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0C10XHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0G50Hb20:I10B0_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0C10Hb20:I10B0_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0C10XHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0Hb20:I10B0_R0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0C10Hb20:I10B0_R0C10Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0G50Hb20:I10B0_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0G0Hb20:I10B0_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0KHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0Hb20:I10RH_R0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0C10XHb20:I10RH_R0C10XHb20", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10KI_R0Hb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10KI_R0C10XHb20 minus I10B0_R0C10XHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0C10XHb20 minus I10KI_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0G50Hb20 minus I10KI_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0C10Hb20 minus I10KI_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0C10XHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0G50Hb20 minus I10B0_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0C10Hb20 minus I10B0_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0C10XHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0Hb20 minus I10B0_R0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0C10Hb20 minus I10B0_R0C10Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0G50Hb20 minus I10B0_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0G0Hb20 minus I10B0_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0KHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0Hb20 minus I10RH_R0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0C10XHb20 minus I10RH_R0C10XHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10KI R0C10XHb20 minus I10B0 R0C10XHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0053 / +0.0071 | +0.0062 | [-0.0034, +0.0175] | 0.0178 (0.0178) | OPEN |
| C contact rate | +0.0031 / -0.0031 | +0.0000 | [-0.0054, +0.0064] | 0.0016 (0.0016) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0140 / +0.0016 | -0.0062 | [-0.0411, +0.0215] | 0.0374 (0.0374) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0044 / -0.0161 | -0.0058 | [-0.0227, +0.0105] | 0.0208 (0.0208) | OPEN |
| gap bias 4.0 s (m) | +0.1281 / -0.0252 | +0.0514 | [-0.0960, +0.1876] | 0.1384 (0.1384) | OPEN |
| gap bias 6.0 s (m) | -0.1330 / -0.1300 | -0.1315 | [-0.8287, +0.4406] | 0.9844 (0.9844) | OPEN |
| gap error 2.0 s (m) | +0.0028 / +0.0025 | +0.0027 | [-0.0026, +0.0080] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.1205 / +0.0325 | +0.0765 | [-0.0293, +0.1704] | 0.0003 (0.0003) | OPEN |
| gap error 6.0 s (m) | -0.0400 / -0.0069 | -0.0235 | [-0.5598, +0.4397] | 0.7417 (0.7417) | OPEN |
| time-gap error 2.0 s (s) | +0.0976 / +0.0024 | +0.0500 | [-0.0247, +0.1724] | 0.1336 (0.1336) | OPEN |
| time-gap error over the scored steps (s) | +0.6764 / +0.5599 | +0.6181 | [+0.3570, +0.8888] | 0.2617 (0.2617) | WORSE |

## I10KI R0C10XHb20 minus I10KI R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0463 / -0.0676 | -0.0569 | [-0.0889, -0.0294] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0327 / -0.0421 | -0.0374 | [-0.0615, -0.0175] | 0.0016 (0.0016) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1355 / +0.1682 | +0.1519 | [+0.1102, +0.1966] | 0.0374 (0.0374) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0208 (0.0208) | OPEN |
| gap bias 4.0 s (m) | +1.4039 / +1.5792 | +1.4916 | [+1.2689, +1.7399] | 0.1384 (0.1384) | WORSE |
| gap bias 6.0 s (m) | +4.8875 / +6.0029 | +5.4452 | [+4.4764, +6.5704] | 0.9844 (0.9844) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.6198 / +0.6334 | +0.6266 | [+0.4073, +0.8632] | 0.0003 (0.0003) | WORSE |
| gap error 6.0 s (m) | +2.0625 / +2.7223 | +2.3924 | [+1.4910, +3.3458] | 0.7417 (0.7417) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.1336 (0.1336) | OPEN |
| time-gap error over the scored steps (s) | +2.1588 / +2.1802 | +2.1695 | [+1.4788, +2.9154] | 0.2617 (0.2617) | WORSE |

## I10KI R0G50Hb20 minus I10KI R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0267 / -0.0374 | -0.0320 | [-0.0516, -0.0147] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0187 / -0.0249 | -0.0218 | [-0.0368, -0.0091] | 0.0016 (0.0016) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0701 / +0.0779 | +0.0740 | [+0.0509, +0.0987] | 0.0374 (0.0374) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0208 (0.0208) | OPEN |
| gap bias 4.0 s (m) | +0.7395 / +0.7891 | +0.7643 | [+0.6589, +0.8741] | 0.1384 (0.1384) | WORSE |
| gap bias 6.0 s (m) | +2.6080 / +3.2412 | +2.9246 | [+2.3680, +3.5554] | 0.9844 (0.9844) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.2647 / +0.2255 | +0.2451 | [+0.1201, +0.3755] | 0.0003 (0.0003) | WORSE |
| gap error 6.0 s (m) | +0.9826 / +1.0705 | +1.0266 | [+0.4959, +1.5725] | 0.7417 (0.7417) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.1336 (0.1336) | OPEN |
| time-gap error over the scored steps (s) | +1.1533 / +0.9940 | +1.0736 | [+0.6041, +1.5871] | 0.2617 (0.2617) | WORSE |

## I10KI R0C10Hb20 minus I10KI R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0196 / -0.0338 | -0.0267 | [-0.0458, -0.0122] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0156 / -0.0202 | -0.0179 | [-0.0339, -0.0060] | 0.0016 (0.0016) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0639 / +0.0981 | +0.0810 | [+0.0492, +0.1155] | 0.0374 (0.0374) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0208 (0.0208) | OPEN |
| gap bias 4.0 s (m) | +0.7169 / +0.8039 | +0.7604 | [+0.6481, +0.8891] | 0.1384 (0.1384) | WORSE |
| gap bias 6.0 s (m) | +2.5004 / +3.2690 | +2.8847 | [+2.2698, +3.5841] | 0.9844 (0.9844) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.3785 / +0.4243 | +0.4014 | [+0.2936, +0.5165] | 0.0003 (0.0003) | WORSE |
| gap error 6.0 s (m) | +1.2111 / +2.0220 | +1.6165 | [+0.9946, +2.2937] | 0.7417 (0.7417) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.1336 (0.1336) | OPEN |
| time-gap error over the scored steps (s) | +0.8096 / +1.1954 | +1.0025 | [+0.6410, +1.4582] | 0.2617 (0.2617) | WORSE |

## I10B0 R0C10XHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0516 / -0.0747 | -0.0632 | [-0.0977, -0.0308] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0405 / -0.0374 | -0.0389 | [-0.0614, -0.0194] | 0.0016 (0.0016) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1511 / +0.1791 | +0.1651 | [+0.1215, +0.2139] | 0.0374 (0.0374) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0208 (0.0208) | OPEN |
| gap bias 4.0 s (m) | +1.3409 / +1.5946 | +1.4678 | [+1.2207, +1.7382] | 0.1384 (0.1384) | WORSE |
| gap bias 6.0 s (m) | +5.1910 / +6.2410 | +5.7160 | [+4.6498, +6.9298] | 0.9844 (0.9844) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.5314 / +0.6764 | +0.6039 | [+0.3704, +0.8402] | 0.0003 (0.0003) | WORSE |
| gap error 6.0 s (m) | +2.2039 / +2.9716 | +2.5877 | [+1.5457, +3.7505] | 0.7417 (0.7417) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.1336 (0.1336) | OPEN |
| time-gap error over the scored steps (s) | +1.5846 / +1.7639 | +1.6743 | [+1.1307, +2.2340] | 0.2617 (0.2617) | WORSE |

## I10B0 R0G50Hb20 minus I10B0 R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0285 / -0.0391 | -0.0338 | [-0.0540, -0.0149] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0202 / -0.0202 | -0.0202 | [-0.0352, -0.0082] | 0.0016 (0.0016) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0561 / +0.0888 | +0.0724 | [+0.0441, +0.1092] | 0.0374 (0.0374) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0208 (0.0208) | OPEN |
| gap bias 4.0 s (m) | +0.6755 / +0.8065 | +0.7410 | [+0.6149, +0.8796] | 0.1384 (0.1384) | WORSE |
| gap bias 6.0 s (m) | +2.5213 / +3.3152 | +2.9182 | [+2.2914, +3.6283] | 0.9844 (0.9844) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.2169 / +0.2442 | +0.2305 | [+0.1141, +0.3530] | 0.0003 (0.0003) | WORSE |
| gap error 6.0 s (m) | +0.8243 / +1.1466 | +0.9854 | [+0.4232, +1.5805] | 0.7417 (0.7417) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.1336 (0.1336) | OPEN |
| time-gap error over the scored steps (s) | +0.8436 / +1.0007 | +0.9222 | [+0.5077, +1.3596] | 0.2617 (0.2617) | WORSE |

## I10B0 R0C10Hb20 minus I10B0 R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0249 / -0.0302 | -0.0276 | [-0.0440, -0.0132] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0218 / -0.0171 | -0.0195 | [-0.0334, -0.0077] | 0.0016 (0.0016) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0950 / +0.1012 | +0.0981 | [+0.0698, +0.1293] | 0.0374 (0.0374) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0208 (0.0208) | OPEN |
| gap bias 4.0 s (m) | +0.6707 / +0.7715 | +0.7211 | [+0.6054, +0.8476] | 0.1384 (0.1384) | WORSE |
| gap bias 6.0 s (m) | +2.7210 / +3.1187 | +2.9198 | [+2.3181, +3.5892] | 0.9844 (0.9844) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.2994 / +0.3979 | +0.3486 | [+0.2279, +0.4805] | 0.0003 (0.0003) | WORSE |
| gap error 6.0 s (m) | +1.4038 / +2.0636 | +1.7337 | [+1.0930, +2.4937] | 0.7417 (0.7417) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.1336 (0.1336) | OPEN |
| time-gap error over the scored steps (s) | +0.6962 / +0.6219 | +0.6590 | [+0.3831, +0.9571] | 0.2617 (0.2617) | WORSE |

## I10KI R0C10XHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0463 / -0.0676 | -0.0569 | [-0.0886, -0.0287] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0374 / -0.0405 | -0.0389 | [-0.0621, -0.0194] | 0.0016 (0.0016) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1371 / +0.1807 | +0.1589 | [+0.1122, +0.2087] | 0.0374 (0.0374) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0044 / -0.0161 | -0.0058 | [-0.0227, +0.0105] | 0.0208 (0.0208) | OPEN |
| gap bias 4.0 s (m) | +1.4690 / +1.5694 | +1.5192 | [+1.3058, +1.7450] | 0.1384 (0.1384) | WORSE |
| gap bias 6.0 s (m) | +5.0580 / +6.1110 | +5.5845 | [+4.6275, +6.6682] | 0.9844 (0.9844) | WORSE |
| gap error 2.0 s (m) | +0.0028 / +0.0025 | +0.0027 | [-0.0026, +0.0080] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.6519 / +0.7088 | +0.6803 | [+0.4508, +0.9203] | 0.0003 (0.0003) | WORSE |
| gap error 6.0 s (m) | +2.1639 / +2.9647 | +2.5643 | [+1.5739, +3.5823] | 0.7417 (0.7417) | WORSE |
| time-gap error 2.0 s (s) | +0.0976 / +0.0024 | +0.0500 | [-0.0247, +0.1724] | 0.1336 (0.1336) | OPEN |
| time-gap error over the scored steps (s) | +2.2610 / +2.3238 | +2.2924 | [+1.5672, +3.0670] | 0.2617 (0.2617) | WORSE |

## I10KI R0Hb20 minus I10B0 R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0018 / +0.0036 | +0.0027 | [-0.0086, +0.0137] | 0.0178 (0.0178) | OPEN |
| C contact rate | -0.0031 / +0.0078 | +0.0023 | [-0.0071, +0.0128] | 0.0016 (0.0016) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0140 / +0.0125 | +0.0132 | [-0.0008, +0.0255] | 0.0374 (0.0374) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0044 / -0.0161 | -0.0058 | [-0.0227, +0.0105] | 0.0208 (0.0208) | OPEN |
| gap bias 4.0 s (m) | +0.1614 / +0.0267 | +0.0941 | [-0.0216, +0.2132] | 0.1384 (0.1384) | OPEN |
| gap bias 6.0 s (m) | +0.2640 / -0.0528 | +0.1056 | [-0.4424, +0.5453] | 0.9844 (0.9844) | OPEN |
| gap error 2.0 s (m) | +0.0028 / +0.0025 | +0.0027 | [-0.0026, +0.0080] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.0978 / +0.0751 | +0.0865 | [+0.0258, +0.1482] | 0.0003 (0.0003) | WORSE |
| gap error 6.0 s (m) | +0.4333 / +0.3029 | +0.3681 | [-0.0050, +0.6681] | 0.7417 (0.7417) | OPEN |
| time-gap error 2.0 s (s) | +0.0976 / +0.0024 | +0.0500 | [-0.0247, +0.1724] | 0.1336 (0.1336) | OPEN |
| time-gap error over the scored steps (s) | +0.3010 / +0.2901 | +0.2955 | [+0.0985, +0.4800] | 0.2617 (0.2617) | WORSE |

## I10KI R0C10Hb20 minus I10B0 R0C10Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0018 / -0.0036 | -0.0009 | [-0.0097, +0.0070] | 0.0178 (0.0178) | OPEN |
| C contact rate | +0.0031 / -0.0031 | +0.0000 | [-0.0069, +0.0083] | 0.0016 (0.0016) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0140 / +0.0016 | -0.0062 | [-0.0333, +0.0159] | 0.0374 (0.0374) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0044 / -0.0161 | -0.0058 | [-0.0227, +0.0105] | 0.0208 (0.0208) | OPEN |
| gap bias 4.0 s (m) | +0.1811 / +0.0191 | +0.1001 | [-0.0516, +0.2491] | 0.1384 (0.1384) | OPEN |
| gap bias 6.0 s (m) | +0.0164 / +0.2404 | +0.1284 | [-0.4950, +0.7019] | 0.9844 (0.9844) | OPEN |
| gap error 2.0 s (m) | +0.0028 / +0.0025 | +0.0027 | [-0.0026, +0.0080] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.1575 / +0.0721 | +0.1148 | [+0.0075, +0.2166] | 0.0003 (0.0003) | WORSE |
| gap error 6.0 s (m) | +0.0074 / +0.0768 | +0.0421 | [-0.5064, +0.5336] | 0.7417 (0.7417) | OPEN |
| time-gap error 2.0 s (s) | +0.0976 / +0.0024 | +0.0500 | [-0.0247, +0.1724] | 0.1336 (0.1336) | OPEN |
| time-gap error over the scored steps (s) | +0.5354 / +0.7011 | +0.6183 | [+0.3699, +0.8827] | 0.2617 (0.2617) | WORSE |

## I10KI R0G50Hb20 minus I10B0 R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0036 / +0.0000 | -0.0018 | [-0.0090, +0.0057] | 0.0178 (0.0178) | OPEN |
| C contact rate | -0.0031 / +0.0000 | -0.0016 | [-0.0054, +0.0023] | 0.0016 (0.0016) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0171 / +0.0047 | +0.0109 | [-0.0061, +0.0266] | 0.0374 (0.0374) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0044 / -0.0161 | -0.0058 | [-0.0227, +0.0105] | 0.0208 (0.0208) | OPEN |
| gap bias 4.0 s (m) | +0.1349 / -0.0133 | +0.0608 | [-0.0628, +0.1718] | 0.1384 (0.1384) | OPEN |
| gap bias 6.0 s (m) | +0.2371 / +0.0901 | +0.1636 | [-0.1095, +0.4122] | 0.9844 (0.9844) | OPEN |
| gap error 2.0 s (m) | +0.0028 / +0.0025 | +0.0027 | [-0.0026, +0.0080] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.0784 / +0.0458 | +0.0621 | [+0.0052, +0.1172] | 0.0003 (0.0003) | WORSE |
| gap error 6.0 s (m) | +0.2002 / +0.1185 | +0.1593 | [-0.0603, +0.3643] | 0.7417 (0.7417) | OPEN |
| time-gap error 2.0 s (s) | +0.0976 / +0.0024 | +0.0500 | [-0.0247, +0.1724] | 0.1336 (0.1336) | OPEN |
| time-gap error over the scored steps (s) | +0.4219 / +0.1276 | +0.2748 | [+0.0398, +0.5594] | 0.2617 (0.2617) | WORSE |

## I10KI R0G0Hb20 minus I10B0 R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0053 / -0.0018 | -0.0036 | [-0.0087, +0.0015] | 0.0178 (0.0178) | OPEN |
| C contact rate | -0.0047 / +0.0047 | +0.0000 | [-0.0070, +0.0085] | 0.0016 (0.0016) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0031 / +0.0156 | +0.0093 | [+0.0003, +0.0218] | 0.0374 (0.0374) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0044 / -0.0161 | -0.0058 | [-0.0227, +0.0105] | 0.0208 (0.0208) | OPEN |
| gap bias 4.0 s (m) | +0.0710 / +0.0042 | +0.0376 | [-0.0332, +0.1011] | 0.1384 (0.1384) | OPEN |
| gap bias 6.0 s (m) | +0.1503 / +0.1641 | +0.1572 | [+0.0250, +0.2872] | 0.9844 (0.9844) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0028 / +0.0025 | +0.0027 | [-0.0026, +0.0080] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.0306 / +0.0644 | +0.0475 | [+0.0056, +0.0866] | 0.0003 (0.0003) | WORSE |
| gap error 6.0 s (m) | +0.0418 / +0.1946 | +0.1182 | [-0.0181, +0.2614] | 0.7417 (0.7417) | OPEN |
| time-gap error 2.0 s (s) | +0.0976 / +0.0024 | +0.0500 | [-0.0247, +0.1724] | 0.1336 (0.1336) | OPEN |
| time-gap error over the scored steps (s) | +0.1123 / +0.1343 | +0.1233 | [+0.0420, +0.2382] | 0.2617 (0.2617) | INSIDE-SPREAD |

## I10KI R0KHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0000 | +0.0000 | [-0.0043, +0.0043] | 0.0178 (0.0178) | OPEN |
| C contact rate | -0.0047 / +0.0016 | -0.0016 | [-0.0070, +0.0048] | 0.0016 (0.0016) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0016 / +0.0125 | +0.0070 | [-0.0011, +0.0188] | 0.0374 (0.0374) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0044 / -0.0161 | -0.0058 | [-0.0227, +0.0105] | 0.0208 (0.0208) | OPEN |
| gap bias 4.0 s (m) | +0.0651 / -0.0098 | +0.0276 | [-0.0465, +0.0930] | 0.1384 (0.1384) | OPEN |
| gap bias 6.0 s (m) | +0.1705 / +0.1081 | +0.1393 | [-0.0098, +0.2771] | 0.9844 (0.9844) | OPEN |
| gap error 2.0 s (m) | +0.0028 / +0.0025 | +0.0027 | [-0.0026, +0.0080] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.0321 / +0.0754 | +0.0538 | [+0.0096, +0.0952] | 0.0003 (0.0003) | WORSE |
| gap error 6.0 s (m) | +0.1013 / +0.2424 | +0.1719 | [+0.0365, +0.3053] | 0.7417 (0.7417) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0976 / +0.0024 | +0.0500 | [-0.0247, +0.1724] | 0.1336 (0.1336) | OPEN |
| time-gap error over the scored steps (s) | +0.1022 / +0.1436 | +0.1229 | [+0.0396, +0.2513] | 0.2617 (0.2617) | INSIDE-SPREAD |

## I10KI R0Hb20 minus I10RH R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0196 / -0.0160 | -0.0178 | [-0.0323, -0.0058] | 0.0178 (0.0178) | INSIDE-SPREAD |
| C contact rate | -0.0202 / -0.0062 | -0.0132 | [-0.0306, -0.0006] | 0.0016 (0.0016) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0483 / +0.0576 | +0.0530 | [+0.0303, +0.0764] | 0.0374 (0.0374) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0348 / +0.0235 | +0.0291 | [+0.0100, +0.0525] | 0.0208 (0.0208) | WORSE |
| gap bias 4.0 s (m) | +0.3915 / +0.3915 | +0.3915 | [+0.2593, +0.5322] | 0.1384 (0.1384) | WORSE |
| gap bias 6.0 s (m) | +1.7102 / +1.7573 | +1.7337 | [+1.2608, +2.2224] | 0.9844 (0.9844) | WORSE |
| gap error 2.0 s (m) | -0.0288 / -0.0194 | -0.0241 | [-0.0422, -0.0088] | 0.0136 (0.0136) | BETTER |
| gap error 4.0 s (m) | -0.0201 / +0.1156 | +0.0477 | [-0.1150, +0.1828] | 0.0003 (0.0003) | OPEN |
| gap error 6.0 s (m) | +0.6011 / +0.9893 | +0.7952 | [+0.2643, +1.2363] | 0.7417 (0.7417) | WORSE |
| time-gap error 2.0 s (s) | +0.0704 / -0.0452 | +0.0126 | [-0.0764, +0.1707] | 0.1336 (0.1336) | OPEN |
| time-gap error over the scored steps (s) | +0.8359 / +0.8543 | +0.8451 | [+0.5191, +1.1912] | 0.2617 (0.2617) | WORSE |

## I10KI R0C10XHb20 minus I10RH R0C10XHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0125 / -0.0107 | -0.0116 | [-0.0235, -0.0018] | 0.0178 (0.0178) | INSIDE-SPREAD |
| C contact rate | -0.0140 / -0.0156 | -0.0148 | [-0.0268, -0.0047] | 0.0016 (0.0016) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0545 / +0.0654 | +0.0600 | [+0.0360, +0.0868] | 0.0374 (0.0374) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0348 / +0.0235 | +0.0291 | [+0.0100, +0.0525] | 0.0208 (0.0208) | WORSE |
| gap bias 4.0 s (m) | +0.4262 / +0.4753 | +0.4508 | [+0.3193, +0.5857] | 0.1384 (0.1384) | WORSE |
| gap bias 6.0 s (m) | +1.6947 / +2.0251 | +1.8599 | [+1.3447, +2.3960] | 0.9844 (0.9844) | WORSE |
| gap error 2.0 s (m) | -0.0288 / -0.0194 | -0.0241 | [-0.0422, -0.0088] | 0.0136 (0.0136) | BETTER |
| gap error 4.0 s (m) | +0.0494 / +0.1941 | +0.1218 | [-0.0412, +0.2585] | 0.0003 (0.0003) | OPEN |
| gap error 6.0 s (m) | +0.8386 / +1.3673 | +1.1029 | [+0.5053, +1.6112] | 0.7417 (0.7417) | WORSE |
| time-gap error 2.0 s (s) | +0.0704 / -0.0452 | +0.0126 | [-0.0764, +0.1707] | 0.1336 (0.1336) | OPEN |
| time-gap error over the scored steps (s) | +1.3774 / +1.2732 | +1.3253 | [+0.8728, +1.8504] | 0.2617 (0.2617) | WORSE |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10B0_R0C10Hb20 (0,1) | 0.0329 [0.0136, 0.0571] | 0.0195 [0.0060, 0.0360] | 0.2882 [0.2316, 0.3454] | 0.0891 [0.0465, 0.1353] | 2.1520 [1.6960, 2.6073] | 7.6474 [6.1154, 9.2291] | 0.4317 [0.3899, 0.4802] | 2.9594 [2.5948, 3.3607] | 8.7568 [7.5255, 10.0816] | 0.8992 [0.6263, 1.2159] | 3.7619 [2.7922, 4.7955] |
| I10B0_R0C10XHb20 (0,1) | 0.0294 [0.0104, 0.0534] | 0.0210 [0.0073, 0.0374] | 0.2796 [0.2236, 0.3374] | 0.0891 [0.0465, 0.1353] | 2.1882 [1.7296, 2.6381] | 7.4618 [6.0483, 8.9499] | 0.4317 [0.3899, 0.4802] | 2.9698 [2.6130, 3.3670] | 8.6467 [7.5117, 9.8733] | 0.8992 [0.6263, 1.2159] | 3.8517 [2.8854, 4.8691] |
| I10B0_R0G0Hb20 (0,1) | 0.0943 [0.0578, 0.1338] | 0.0592 [0.0296, 0.0938] | 0.1176 [0.0805, 0.1614] | 0.0891 [0.0465, 0.1353] | 0.6899 [0.3114, 1.0864] | 1.8093 [0.6138, 3.0721] | 0.4317 [0.3899, 0.4802] | 2.3802 [2.0930, 2.7083] | 6.0377 [5.2686, 6.8989] | 0.8992 [0.6263, 1.2159] | 2.1807 [1.5120, 2.9252] |
| I10B0_R0G50Hb20 (0,1) | 0.0605 [0.0351, 0.0901] | 0.0389 [0.0150, 0.0669] | 0.1900 [0.1424, 0.2401] | 0.0891 [0.0465, 0.1353] | 1.4309 [1.0198, 1.8362] | 4.7275 [3.3100, 6.1037] | 0.4317 [0.3899, 0.4802] | 2.6107 [2.2874, 2.9683] | 7.0231 [6.0954, 8.0773] | 0.8992 [0.6263, 1.2159] | 3.1028 [2.2323, 4.0349] |
| I10B0_R0Hb20 (0,1) | 0.0614 [0.0289, 0.0956] | 0.0280 [0.0116, 0.0479] | 0.2033 [0.1503, 0.2595] | 0.0891 [0.0465, 0.1353] | 1.4575 [0.9911, 1.9165] | 5.2877 [3.8033, 6.8244] | 0.4317 [0.3899, 0.4802] | 2.6569 [2.3239, 3.0194] | 7.3768 [6.2706, 8.5258] | 0.8992 [0.6263, 1.2159] | 3.1310 [2.2928, 4.0200] |
| I10B0_R0KHb20 (0,1) | 0.0925 [0.0563, 0.1324] | 0.0600 [0.0297, 0.0927] | 0.1145 [0.0783, 0.1567] | 0.0891 [0.0465, 0.1353] | 0.7204 [0.3444, 1.1110] | 1.7458 [0.5279, 3.0407] | 0.4317 [0.3899, 0.4802] | 2.3660 [2.0834, 2.6885] | 6.0590 [5.2991, 6.9059] | 0.8992 [0.6263, 1.2159] | 2.1774 [1.5055, 2.9188] |
| I10KI_R0C10Hb20 (0,1) | 0.0320 [0.0132, 0.0579] | 0.0195 [0.0041, 0.0415] | 0.2819 [0.2214, 0.3421] | 0.0833 [0.0361, 0.1338] | 2.2522 [1.8063, 2.7194] | 7.7758 [6.1621, 9.3846] | 0.4343 [0.3918, 0.4817] | 3.0742 [2.6784, 3.5216] | 8.7989 [7.5324, 10.0933] | 0.9492 [0.6485, 1.2997] | 4.3801 [3.3070, 5.5242] |
| I10KI_R0C10XHb20 (0,1) | 0.0356 [0.0156, 0.0608] | 0.0210 [0.0063, 0.0413] | 0.2734 [0.2156, 0.3317] | 0.0833 [0.0361, 0.1338] | 2.2396 [1.7865, 2.6922] | 7.3303 [5.8246, 8.7962] | 0.4343 [0.3918, 0.4817] | 3.0463 [2.6618, 3.4830] | 8.6233 [7.4635, 9.8049] | 0.9492 [0.6485, 1.2997] | 4.4698 [3.3622, 5.6562] |
| I10KI_R0G0Hb20 (0,1) | 0.0907 [0.0552, 0.1285] | 0.0592 [0.0302, 0.0915] | 0.1269 [0.0885, 0.1692] | 0.0833 [0.0361, 0.1338] | 0.7274 [0.3440, 1.1466] | 1.9665 [0.7243, 3.2585] | 0.4343 [0.3918, 0.4817] | 2.4277 [2.1352, 2.7546] | 6.1559 [5.3724, 7.0421] | 0.9492 [0.6485, 1.2997] | 2.3040 [1.5890, 3.0837] |
| I10KI_R0G50Hb20 (0,1) | 0.0587 [0.0343, 0.0865] | 0.0374 [0.0147, 0.0642] | 0.2009 [0.1511, 0.2512] | 0.0833 [0.0361, 0.1338] | 1.4917 [1.0833, 1.9227] | 4.8911 [3.4684, 6.2157] | 0.4343 [0.3918, 0.4817] | 2.6728 [2.3384, 3.0537] | 7.1824 [6.2269, 8.2536] | 0.9492 [0.6485, 1.2997] | 3.3776 [2.4233, 4.4078] |
| I10KI_R0Hb20 (0,1) | 0.0641 [0.0315, 0.1026] | 0.0304 [0.0118, 0.0509] | 0.2165 [0.1601, 0.2747] | 0.0833 [0.0361, 0.1338] | 1.5516 [1.0992, 2.0166] | 5.3933 [3.8756, 6.9573] | 0.4343 [0.3918, 0.4817] | 2.7433 [2.3861, 3.1287] | 7.7449 [6.6315, 8.9354] | 0.9492 [0.6485, 1.2997] | 3.4265 [2.5654, 4.3292] |
| I10KI_R0KHb20 (0,1) | 0.0925 [0.0558, 0.1320] | 0.0584 [0.0289, 0.0918] | 0.1215 [0.0845, 0.1623] | 0.0833 [0.0361, 0.1338] | 0.7480 [0.3665, 1.1635] | 1.8851 [0.6342, 3.2028] | 0.4343 [0.3918, 0.4817] | 2.4197 [2.1308, 2.7388] | 6.2309 [5.4606, 7.0819] | 0.9492 [0.6485, 1.2997] | 2.3003 [1.5883, 3.0829] |
| I10RH_R0C10XHb20 (0,1) | 0.0472 [0.0221, 0.0795] | 0.0358 [0.0137, 0.0613] | 0.2134 [0.1621, 0.2687] | 0.0541 [-0.0003, 0.1137] | 1.7888 [1.3557, 2.2578] | 5.4704 [4.1477, 6.8065] | 0.4584 [0.4066, 0.5113] | 2.9245 [2.5330, 3.3539] | 7.5203 [6.6391, 8.5576] | 0.9366 [0.6450, 1.2757] | 3.1445 [2.3022, 4.0131] |
| I10RH_R0Hb20 (0,1) | 0.0819 [0.0438, 0.1239] | 0.0436 [0.0206, 0.0728] | 0.1636 [0.1163, 0.2142] | 0.0541 [-0.0003, 0.1137] | 1.1601 [0.7080, 1.6234] | 3.6596 [2.2695, 5.1805] | 0.4584 [0.4066, 0.5113] | 2.6956 [2.3561, 3.0709] | 6.9497 [6.0649, 7.9524] | 0.9366 [0.6450, 1.2757] | 2.5814 [1.8594, 3.3407] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
