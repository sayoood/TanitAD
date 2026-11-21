# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: /workspace/v8proof/podread/data/join_eval141.npz. Alignment reference (ego speed): /workspace/v8proof/podread/data/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 36, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.03202846975088969, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0M5Hb20:I10R0M5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcM5Hb20:I10RLcM5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50M5Hb20:I10RQ50M5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0M5:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcM5:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50M5:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0M5Hb20:I10R0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0KM5Hb20:I10R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcKM5Hb20:I10RLcKF5Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50KM5Hb20:I10RQ50KF5Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcM5Hb20:I10RLcHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50M5Hb20:I10RQ50Hb20", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10R0M5Hb20 minus I10R0M5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcM5Hb20 minus I10RLcM5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50M5Hb20 minus I10RQ50M5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0M5 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcM5 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50M5 minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0M5Hb20 minus I10R0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0KM5Hb20 minus I10R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcKM5Hb20 minus I10RLcKF5Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50KM5Hb20 minus I10RQ50KF5Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcM5Hb20 minus I10RLcHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50M5Hb20 minus I10RQ50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10R0M5Hb20 minus I10R0M5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0445 / -0.0409 | -0.0427 | [-0.0729, -0.0172] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0062 / -0.0187 | -0.0125 | [-0.0283, -0.0001] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0343 / -0.0358 | -0.0350 | [-0.0594, -0.0132] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0751 / -0.1074 | -0.0912 | [-0.1436, -0.0483] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -1.6245 / -1.3083 | -1.4664 | [-2.6681, -0.4794] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -4.9572 / -0.3584 | -2.6578 | [-8.1122, +0.9417] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | -0.0944 / -0.1176 | -0.1060 | [-0.1611, -0.0613] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -2.0739 / -1.8404 | -1.9572 | [-3.1120, -0.9851] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -5.8110 / -1.4751 | -3.6431 | [-8.1915, -0.5856] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.6364 / -0.5476 | -0.5920 | [-1.2116, -0.1447] | 0.0330 (0.0330) | BETTER |
| time-gap error over the scored steps (s) | -0.1222 / -0.1921 | -0.1572 | [-0.6761, +0.2122] | 0.5140 (0.5140) | OPEN |

## I10RLcM5Hb20 minus I10RLcM5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0836 / -0.1139 | -0.0988 | [-0.1298, -0.0675] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0078 / -0.0078 | -0.0078 | [-0.0164, +0.0007] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0265 / -0.0389 | -0.0327 | [-0.0511, -0.0140] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0281 / -0.0415 | -0.0348 | [-0.0598, -0.0129] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.3239 / -0.2282 | -0.2760 | [-0.4943, -0.0885] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -1.3290 / -1.6176 | -1.4733 | [-2.2352, -0.6669] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0458 / -0.0530 | -0.0494 | [-0.0758, -0.0284] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -0.4932 / -0.2595 | -0.3763 | [-0.6009, -0.1531] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -1.2440 / -1.3482 | -1.2961 | [-2.0283, -0.6132] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.5111 / -0.3883 | -0.4497 | [-1.0931, +0.0004] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.0798 / +0.0421 | +0.0609 | [-0.4719, +0.4670] | 0.5140 (0.5140) | OPEN |

## I10RQ50M5Hb20 minus I10RQ50M5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.1228 / -0.0907 | -0.1068 | [-0.1473, -0.0670] | 0.0320 (0.0320) | BETTER |
| C contact rate | +0.0016 / -0.0109 | -0.0047 | [-0.0184, +0.0111] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0296 / -0.0140 | -0.0218 | [-0.0428, +0.0005] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0355 / -0.0626 | -0.0491 | [-0.0881, -0.0185] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -0.9329 / -0.1982 | -0.5655 | [-1.2165, -0.0618] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -2.6530 / -1.3760 | -2.0145 | [-3.7070, -0.7606] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0451 / -0.0703 | -0.0577 | [-0.0982, -0.0274] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -1.0408 / -0.2096 | -0.6252 | [-1.2947, -0.0953] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -2.6376 / -1.5680 | -2.1028 | [-3.5189, -0.9462] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.5501 / -0.4596 | -0.5048 | [-1.1495, -0.0531] | 0.0330 (0.0330) | BETTER |
| time-gap error over the scored steps (s) | +0.6278 / +0.1789 | +0.4034 | [-0.2523, +1.0627] | 0.5140 (0.5140) | OPEN |

## I10R0M5 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.7046 / +0.7829 | +0.7438 | [+0.6709, +0.8132] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.5732 / +0.6900 | +0.6316 | [+0.5353, +0.7204] | 0.0343 (0.0343) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0249 / -0.1246 | -0.0748 | [-0.1589, +0.0028] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.8234 / -0.9369 | -0.8801 | [-1.0195, -0.7523] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -13.2057 / -19.2321 | -16.2189 | [-20.8368, -11.7505] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -39.4510 / -62.4200 | -50.9355 | [-66.4438, -34.7254] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.8326 / +0.9574 | +0.8950 | [+0.7815, +1.0248] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +14.9033 / +19.1277 | +17.0155 | [+13.7623, +20.2508] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +47.0642 / +54.1366 | +50.6004 | [+44.5907, +57.1429] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.8021 / +0.8116 | +0.8069 | [+0.3390, +1.4221] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.8451 / +0.4830 | +0.6641 | [+0.1286, +1.1720] | 0.5140 (0.5140) | WORSE |

## I10RLcM5 minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.4235 / +0.4733 | +0.4484 | [+0.3789, +0.5189] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0171 / +0.0171 | +0.0171 | [+0.0090, +0.0256] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0467 / -0.0312 | +0.0078 | [-0.0595, +0.0746] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.4889 / -0.5777 | -0.5333 | [-0.7072, -0.3547] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -3.1207 / -5.2605 | -4.1906 | [-6.1070, -2.3717] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -2.2666 / -6.2674 | -4.2670 | [-8.3785, -0.2598] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.8363 / +0.8715 | +0.8539 | [+0.7454, +0.9661] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +5.2758 / +5.0161 | +5.1459 | [+4.2130, +6.1488] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +8.7809 / +6.5379 | +7.6594 | [+4.9695, +10.2758] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.6972 / +0.6875 | +0.6923 | [+0.2248, +1.3180] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.3717 / -0.1302 | +0.1208 | [-0.4905, +0.6696] | 0.5140 (0.5140) | OPEN |

## I10RQ50M5 minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.4573 / +0.5338 | +0.4956 | [+0.4216, +0.5718] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0763 / +0.1044 | +0.0903 | [+0.0526, +0.1306] | 0.0343 (0.0343) | WORSE |
| F falls behind by more than 10 m at the last scored step | +0.0530 / -0.0763 | -0.0117 | [-0.0997, +0.0776] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1930 / -0.5428 | -0.3679 | [-0.6877, -0.0567] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -1.2127 / -5.4782 | -3.3455 | [-6.3421, -0.1691] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -1.8290 / -8.6594 | -5.2442 | [-10.8447, +0.8908] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +1.0036 / +0.9677 | +0.9857 | [+0.8467, +1.1392] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +6.3923 / +4.6402 | +5.5163 | [+4.0215, +7.2018] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +13.9457 / +7.4040 | +10.6749 | [+5.4352, +16.4090] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +1.1376 / +0.9296 | +1.0336 | [+0.5162, +1.6926] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.2014 / -0.2547 | -0.0266 | [-0.6336, +0.5396] | 0.5140 (0.5140) | OPEN |

## I10R0M5Hb20 minus I10R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.6940 / +0.7847 | +0.7393 | [+0.6606, +0.8143] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.5888 / +0.6776 | +0.6332 | [+0.5517, +0.7072] | 0.0343 (0.0343) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0779 / -0.1573 | -0.1176 | [-0.1901, -0.0507] | 0.0903 (0.0903) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.8985 / -1.0442 | -0.9714 | [-1.1142, -0.8423] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -14.8049 / -20.3759 | -17.5904 | [-21.7232, -13.7427] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -45.4159 / -62.9756 | -54.1957 | [-67.0453, -42.4149] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.7382 / +0.8398 | +0.7890 | [+0.6843, +0.9018] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +12.8450 / +17.4132 | +15.1291 | [+11.7556, +18.6351] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +41.3320 / +52.9797 | +47.1558 | [+39.0866, +56.1916] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.1656 / +0.2641 | +0.2149 | [+0.0547, +0.3715] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.5011 / +0.0548 | +0.2779 | [-0.2518, +0.7513] | 0.5140 (0.5140) | OPEN |

## I10R0KM5Hb20 minus I10R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.1228 / +0.1192 | +0.1210 | [+0.0769, +0.1680] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0623 / +0.0623 | +0.0623 | [+0.0384, +0.0883] | 0.0343 (0.0343) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0607 / -0.0810 | -0.0709 | [-0.1068, -0.0400] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1091 / -0.1386 | -0.1239 | [-0.1818, -0.0678] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -2.0156 / -2.4907 | -2.2531 | [-2.7635, -1.7848] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -5.1717 / -6.5814 | -5.8766 | [-7.2627, -4.6072] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.1001 / +0.1133 | +0.1067 | [+0.0599, +0.1544] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.9430 / +0.9280 | +0.9355 | [+0.5793, +1.3265] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +1.9607 / +2.0352 | +1.9979 | [+0.9148, +3.0677] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.0224 / -0.0369 | -0.0072 | [-0.2687, +0.3154] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.2860 / -0.6304 | -0.4582 | [-1.0252, +0.0578] | 0.5140 (0.5140) | OPEN |

## I10RLcKM5Hb20 minus I10RLcKF5Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0463 / +0.0463 | +0.0463 | [+0.0248, +0.0715] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0016 / +0.0000 | +0.0008 | [+0.0000, +0.0031] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0530 / -0.0794 | -0.0662 | [-0.1053, -0.0313] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0982 / +0.0625 | +0.0803 | [-0.0622, +0.2503] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.9464 / -1.3413 | -1.1438 | [-1.7204, -0.5726] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -2.7103 / -3.8323 | -3.2713 | [-4.6157, -2.0679] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.2341 / +0.2449 | +0.2395 | [+0.1109, +0.4078] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.7308 / +0.5941 | +0.6624 | [+0.2537, +1.0986] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +0.6722 / +0.3383 | +0.5053 | [-0.5085, +1.4675] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | -0.0094 / -0.0684 | -0.0389 | [-0.3010, +0.2855] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.3358 / -0.7114 | -0.5236 | [-1.0894, -0.0177] | 0.5140 (0.5140) | BETTER |

## I10RQ50KM5Hb20 minus I10RQ50KF5Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0587 / +0.0641 | +0.0614 | [+0.0325, +0.0911] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0265 / +0.0389 | +0.0327 | [+0.0145, +0.0563] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0545 / -0.0748 | -0.0646 | [-0.1015, -0.0315] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0001 / -0.0615 | -0.0308 | [-0.1240, +0.0633] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -1.0406 / -1.6008 | -1.3207 | [-1.8874, -0.7935] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -2.4475 / -3.8070 | -3.1273 | [-4.4012, -1.9180] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.1882 / +0.2008 | +0.1945 | [+0.1291, +0.2708] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.7116 / +0.6198 | +0.6657 | [+0.3317, +0.9868] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +0.5675 / +0.1406 | +0.3541 | [-0.5543, +1.2157] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.1362 / +0.0410 | +0.0886 | [-0.1910, +0.4420] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.0643 / -0.5197 | -0.2920 | [-0.8627, +0.2654] | 0.5140 (0.5140) | OPEN |

## I10RLcM5Hb20 minus I10RLcHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.3434 / +0.3683 | +0.3559 | [+0.3020, +0.4099] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0093 / +0.0093 | +0.0093 | [+0.0025, +0.0167] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0078 / -0.0701 | -0.0312 | [-0.1002, +0.0349] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.5171 / -0.6192 | -0.5681 | [-0.7473, -0.3898] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -3.4098 / -5.3684 | -4.3891 | [-6.2566, -2.7090] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -3.9517 / -7.7841 | -5.8679 | [-9.7653, -2.1054] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.7905 / +0.8185 | +0.8045 | [+0.7001, +0.9168] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +4.8072 / +4.8627 | +4.8349 | [+3.9201, +5.8629] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +7.1991 / +5.3017 | +6.2504 | [+3.7355, +8.7365] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.1861 / +0.2992 | +0.2426 | [+0.0779, +0.4087] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.1926 / -0.3457 | -0.0765 | [-0.6331, +0.4225] | 0.5140 (0.5140) | OPEN |

## I10RQ50M5Hb20 minus I10RQ50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.3363 / +0.4448 | +0.3906 | [+0.3045, +0.4815] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0732 / +0.0950 | +0.0841 | [+0.0488, +0.1206] | 0.0343 (0.0343) | WORSE |
| F falls behind by more than 10 m at the last scored step | +0.0125 / -0.0794 | -0.0335 | [-0.1063, +0.0350] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.2286 / -0.6054 | -0.4170 | [-0.7543, -0.0961] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -2.0910 / -5.5211 | -3.8060 | [-6.3710, -1.1329] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -4.6820 / -9.8936 | -7.2878 | [-12.2791, -2.1933] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.9585 / +0.8975 | +0.9280 | [+0.7779, +1.0886] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +5.3784 / +4.5393 | +4.9588 | [+3.8094, +6.2322] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +11.0663 / +5.9828 | +8.5245 | [+4.0976, +13.4307] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.5875 / +0.4701 | +0.5288 | [+0.2893, +0.8025] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.6139 / -0.3415 | +0.1362 | [-0.6554, +0.9206] | 0.5140 (0.5140) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10R0Hb20 (0,1) | 0.0916 [0.0453, 0.1445] | 0.0452 [0.0224, 0.0734] | 0.1480 [0.0872, 0.2156] | -0.2236 [-0.2757, -0.1682] | 0.3534 [-0.3798, 1.0966] | 3.3965 [1.2253, 5.6225] | 0.4729 [0.4299, 0.5219] | 2.4213 [2.1439, 2.7423] | 6.6793 [5.5074, 7.8860] | 1.0435 [0.7059, 1.4212] | 2.2708 [1.6210, 2.9961] |
| I10R0KHb20 (0,1) | 0.0970 [0.0608, 0.1366] | 0.0537 [0.0264, 0.0848] | 0.1067 [0.0705, 0.1471] | -0.2236 [-0.2757, -0.1682] | 0.3222 [-0.1594, 0.7965] | 1.3463 [-0.1516, 2.8420] | 0.4729 [0.4299, 0.5219] | 2.2113 [1.9446, 2.4997] | 5.8179 [5.0804, 6.6424] | 1.0435 [0.7059, 1.4212] | 2.2124 [1.5547, 2.9468] |
| I10R0KM5Hb20 (0,1) | 0.2180 [0.1517, 0.2861] | 0.1160 [0.0722, 0.1636] | 0.0358 [0.0175, 0.0569] | -0.3474 [-0.4245, -0.2696] | -1.9310 [-2.4434, -1.4270] | -4.5303 [-5.9549, -3.0792] | 0.5796 [0.5123, 0.6551] | 3.1468 [2.7295, 3.6294] | 7.8158 [6.5886, 9.1288] | 1.0362 [0.6536, 1.5140] | 1.7541 [1.1362, 2.5375] |
| I10R0M5 (0,1) | 0.8737 [0.8176, 0.9191] | 0.6908 [0.6102, 0.7630] | 0.0654 [0.0378, 0.0987] | -1.1037 [-1.2468, -0.9695] | -15.7706 [-19.8054, -11.7181] | -48.1414 [-61.7632, -33.8481] | 1.3679 [1.2375, 1.5208] | 19.5076 [16.1343, 22.9679] | 57.4782 [50.7509, 64.8401] | 1.8503 [1.2289, 2.6007] | 2.7059 [1.8928, 3.7647] |
| I10R0M5Hb20 (0,1) | 0.8310 [0.7751, 0.8820] | 0.6783 [0.6054, 0.7445] | 0.0304 [0.0143, 0.0509] | -1.1949 [-1.3375, -1.0715] | -17.2370 [-20.8771, -13.7647] | -50.7992 [-61.9295, -40.4198] | 1.2619 [1.1431, 1.4015] | 17.5504 [14.0964, 21.1796] | 53.8351 [45.0863, 63.7791] | 1.2583 [0.9081, 1.7117] | 2.5487 [1.9618, 3.2341] |
| I10RLc (0,1) | 0.0383 [0.0190, 0.0628] | 0.0016 [0.0000, 0.0051] | 0.2181 [0.1459, 0.2897] | 0.2229 [-0.0327, 0.5472] | 2.0938 [1.2319, 3.0706] | 6.7474 [4.5816, 8.9276] | 0.7885 [0.5579, 1.0945] | 3.2710 [2.6177, 4.1001] | 8.0958 [6.4712, 9.8010] | 1.0876 [0.7460, 1.4643] | 2.1535 [1.4770, 2.9189] |
| I10RLcHb20 (0,1) | 0.0320 [0.0127, 0.0567] | 0.0016 [0.0000, 0.0051] | 0.2243 [0.1570, 0.2909] | 0.2229 [-0.0327, 0.5472] | 2.0162 [1.1856, 2.9752] | 6.8750 [4.8011, 8.9005] | 0.7885 [0.5579, 1.0945] | 3.2056 [2.5510, 4.0451] | 8.2087 [6.6620, 9.8507] | 1.0876 [0.7460, 1.4643] | 2.4117 [1.7601, 3.1589] |
| I10RLcKF5Hb20 (0,1) | 0.0365 [0.0166, 0.0590] | 0.0000 [0.0000, 0.0000] | 0.1550 [0.1023, 0.2140] | 0.2660 [-0.0005, 0.6041] | 1.5564 [0.8610, 2.3908] | 3.6380 [2.1931, 5.1343] | 0.7970 [0.5589, 1.1110] | 2.6382 [2.1182, 3.3187] | 5.6866 [4.7784, 6.7972] | 1.1020 [0.7679, 1.4781] | 2.2406 [1.5686, 2.9970] |
| I10RLcKM5Hb20 (0,1) | 0.0827 [0.0490, 0.1219] | 0.0008 [0.0000, 0.0031] | 0.0888 [0.0447, 0.1453] | 0.3463 [-0.0527, 0.8434] | 0.4125 [-0.5553, 1.5802] | 0.3667 [-1.3052, 2.2586] | 1.0366 [0.6740, 1.5108] | 3.3007 [2.4931, 4.3299] | 6.1919 [4.9448, 7.7167] | 1.0631 [0.6845, 1.5333] | 1.7170 [1.1048, 2.5101] |
| I10RLcM5 (0,1) | 0.4867 [0.4184, 0.5555] | 0.0187 [0.0108, 0.0273] | 0.2259 [0.1722, 0.2837] | -0.3104 [-0.7041, 0.1461] | -2.0968 [-4.2249, 0.0735] | 2.4804 [-1.7053, 6.8513] | 1.6424 [1.3486, 2.0128] | 8.4169 [7.2754, 9.7230] | 15.7552 [13.1288, 18.7193] | 1.7800 [1.1768, 2.5370] | 2.2743 [1.5065, 3.3193] |
| I10RLcM5Hb20 (0,1) | 0.3879 [0.3341, 0.4412] | 0.0109 [0.0048, 0.0182] | 0.1931 [0.1371, 0.2518] | -0.3452 [-0.7420, 0.1148] | -2.3728 [-4.4622, -0.3023] | 1.0071 [-3.0962, 5.2117] | 1.5930 [1.3050, 1.9671] | 8.0406 [6.8921, 9.3294] | 14.4591 [11.8611, 17.2581] | 1.3302 [0.9800, 1.7867] | 2.3352 [1.7666, 2.9842] |
| I10RQ50 (0,1) | 0.0801 [0.0480, 0.1174] | 0.0265 [0.0102, 0.0465] | 0.2593 [0.1862, 0.3304] | 0.2697 [0.0335, 0.5486] | 2.4645 [1.6073, 3.4225] | 7.6520 [5.6761, 9.6557] | 0.8241 [0.5966, 1.0699] | 3.7859 [3.0818, 4.6228] | 9.4150 [7.8273, 11.0990] | 1.1816 [0.8261, 1.5783] | 2.5241 [1.8524, 3.2957] |
| I10RQ50Hb20 (0,1) | 0.0783 [0.0451, 0.1153] | 0.0280 [0.0100, 0.0533] | 0.2593 [0.1942, 0.3218] | 0.2697 [0.0335, 0.5486] | 2.3596 [1.5024, 3.3067] | 7.6812 [5.7500, 9.6407] | 0.8241 [0.5966, 1.0699] | 3.7182 [3.0177, 4.5458] | 9.4625 [7.9226, 11.1337] | 1.1816 [0.8261, 1.5783] | 2.7647 [2.0805, 3.5577] |
| I10RQ50KF5Hb20 (0,1) | 0.0738 [0.0427, 0.1106] | 0.0296 [0.0105, 0.0525] | 0.1682 [0.1245, 0.2169] | 0.3589 [0.0841, 0.6833] | 1.6336 [1.0778, 2.2703] | 3.6992 [2.5487, 4.9307] | 0.8726 [0.6109, 1.1655] | 2.8769 [2.4025, 3.4140] | 6.2598 [5.4336, 7.1778] | 1.1992 [0.8358, 1.6032] | 2.4250 [1.7255, 3.2065] |
| I10RQ50KM5Hb20 (0,1) | 0.1352 [0.0872, 0.1885] | 0.0623 [0.0323, 0.0931] | 0.1036 [0.0621, 0.1490] | 0.3282 [-0.0244, 0.7239] | 0.3129 [-0.6442, 1.3199] | 0.5719 [-1.0766, 2.3923] | 1.0671 [0.7699, 1.4091] | 3.5426 [2.9234, 4.2613] | 6.6138 [5.5278, 7.8355] | 1.2878 [0.8252, 1.8105] | 2.1331 [1.3729, 2.9996] |
| I10RQ50M5 (0,1) | 0.5756 [0.5018, 0.6486] | 0.1168 [0.0813, 0.1546] | 0.2477 [0.1845, 0.3185] | -0.0982 [-0.6460, 0.4743] | -0.8809 [-4.0130, 2.4903] | 2.4079 [-3.5213, 9.0091] | 1.8098 [1.4772, 2.1781] | 9.3022 [7.6778, 11.2168] | 20.0898 [15.1278, 25.9184] | 2.2152 [1.5551, 3.0144] | 2.4975 [1.7350, 3.5154] |
| I10RQ50M5Hb20 (0,1) | 0.4689 [0.3839, 0.5582] | 0.1121 [0.0797, 0.1476] | 0.2259 [0.1656, 0.2939] | -0.1473 [-0.6983, 0.4339] | -1.4464 [-4.2435, 1.4345] | 0.3934 [-5.0705, 6.1044] | 1.7521 [1.4013, 2.1282] | 8.6770 [7.3684, 10.2105] | 17.9870 [13.6279, 23.1139] | 1.7104 [1.2623, 2.2092] | 2.9009 [2.1605, 3.7046] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
