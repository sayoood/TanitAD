# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: /workspace/v8proof/podread/data/join_eval141.npz. Alignment reference (ego speed): /workspace/v8proof/podread/data/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 32, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.03202846975088969, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0C10Hb20:I10R0C10", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0C10XHb20:I10R0C10X", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcHb20:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50Hb20:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0C10X:I10R0C10", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0C10XHb20:I10R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0KHb20:I10R0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcKF5Hb20:I10RLcHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50KF5Hb20:I10RQ50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0Hb05:I10R0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0Hb10:I10R0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0Hb40:I10R0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0C10XHb20:I10R0Hb20", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10R0C10Hb20 minus I10R0C10 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0C10XHb20 minus I10R0C10X over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcHb20 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50Hb20 minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0C10X minus I10R0C10 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0C10XHb20 minus I10R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0KHb20 minus I10R0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcKF5Hb20 minus I10RLcHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50KF5Hb20 minus I10RQ50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0Hb05 minus I10R0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0Hb10 minus I10R0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0Hb40 minus I10R0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0C10XHb20 minus I10R0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10R0C10Hb20 minus I10R0C10

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0730 / -0.0569 | -0.0649 | [-0.1013, -0.0321] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0249 / -0.0452 | -0.0350 | [-0.0598, -0.0136] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0421 / +0.0125 | +0.0273 | [+0.0007, +0.0560] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.3382 / +0.3211 | +0.3297 | [+0.1859, +0.4915] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.0788 / +3.5791 | +2.8289 | [+1.5910, +4.3897] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.1662 / +0.0058 | -0.0802 | [-0.2455, +0.1025] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +0.0162 / -2.1469 | -1.0654 | [-2.7961, +0.3714] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +1.1987 / +0.9971 | +1.0979 | [+0.5438, +1.6745] | 0.5140 (0.5140) | WORSE |

## I10R0C10XHb20 minus I10R0C10X

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0623 / -0.0498 | -0.0560 | [-0.0871, -0.0280] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0452 / -0.0280 | -0.0366 | [-0.0591, -0.0170] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0389 / +0.0312 | +0.0350 | [+0.0195, +0.0532] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.1864 / +0.0891 | +0.1377 | [+0.0430, +0.2561] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.1584 / +1.3947 | +1.7766 | [+1.0737, +2.7169] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.0777 / +0.0060 | -0.0358 | [-0.1524, +0.0519] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +0.3682 / +0.5130 | +0.4406 | [+0.0585, +0.8023] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +1.5070 / +1.3265 | +1.4168 | [+0.8278, +2.0231] | 0.5140 (0.5140) | WORSE |

## I10RLcHb20 minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0036 / -0.0089 | -0.0062 | [-0.0133, -0.0010] | 0.0320 (0.0320) | INSIDE-SPREAD |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0125 / +0.0000 | +0.0062 | [-0.0060, +0.0194] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.0348 / -0.1204 | -0.0776 | [-0.1534, -0.0169] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.3561 / -0.1008 | +0.1276 | [-0.1953, +0.5167] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.0246 / -0.1061 | -0.0653 | [-0.1369, -0.0128] | 0.2417 (0.2417) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.3378 / -0.1120 | +0.1129 | [-0.1945, +0.4853] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.2589 / +0.2576 | +0.2583 | [-0.1516, +0.6222] | 0.5140 (0.5140) | OPEN |

## I10RQ50Hb20 minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0018 / -0.0018 | -0.0018 | [-0.0080, +0.0044] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0047 / -0.0016 | +0.0016 | [-0.0030, +0.0079] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0109 / -0.0109 | +0.0000 | [-0.0190, +0.0185] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.0545 / -0.1553 | -0.1049 | [-0.1896, -0.0372] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.2001 / -0.1417 | +0.0292 | [-0.2728, +0.3189] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.0268 / -0.1087 | -0.0678 | [-0.1358, -0.0163] | 0.2417 (0.2417) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.2418 / -0.1468 | +0.0475 | [-0.2625, +0.3485] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.2153 / +0.2658 | +0.2405 | [-0.1647, +0.5825] | 0.5140 (0.5140) | OPEN |

## I10R0C10X minus I10R0C10

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0302 / -0.0409 | -0.0356 | [-0.0580, -0.0116] | 0.0320 (0.0320) | BETTER |
| C contact rate | +0.0047 / -0.0249 | -0.0101 | [-0.0338, +0.0154] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0171 / -0.0093 | +0.0039 | [-0.0183, +0.0320] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.3727 / +0.3380 | +0.3553 | [+0.2105, +0.5214] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.6447 / +2.4992 | +1.5720 | [+0.0434, +3.2967] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.0178 / +0.0871 | +0.0524 | [-0.0650, +0.1686] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -0.0803 / -2.4386 | -1.2595 | [-3.1131, +0.2792] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.5949 / +0.1315 | +0.3632 | [+0.0621, +0.7454] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10R0C10XHb20 minus I10R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0427 / -0.0267 | -0.0347 | [-0.0628, -0.0079] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0312 / -0.0202 | -0.0257 | [-0.0461, -0.0105] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.2679 / +0.3006 | +0.2843 | [+0.2223, +0.3453] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +2.5929 / +2.7918 | +2.6923 | [+2.3199, +3.0685] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +8.7275 / +9.2757 | +9.0016 | [+7.1535, +10.9087] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +1.1973 / +1.5396 | +1.3684 | [+0.9575, +1.7762] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +5.1949 / +6.6522 | +5.9235 | [+4.1827, +7.9604] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +2.8681 / +2.4213 | +2.6447 | [+1.9468, +3.4309] | 0.5140 (0.5140) | WORSE |

## I10R0KHb20 minus I10R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0089 / +0.0196 | +0.0053 | [-0.0279, +0.0445] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0016 / +0.0156 | +0.0086 | [-0.0110, +0.0269] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0187 / -0.0639 | -0.0413 | [-0.0852, -0.0031] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.1937 / -0.2562 | -0.0312 | [-0.4456, +0.3365] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | -1.2858 / -2.8148 | -2.0503 | [-3.4663, -0.7508] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.1782 / -0.2419 | -0.2100 | [-0.3961, -0.0344] | 0.2417 (0.2417) | INSIDE-SPREAD |
| gap error 6.0 s (m) | -0.2488 / -1.4740 | -0.8614 | [-2.0017, +0.1085] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.0861 / -0.2030 | -0.0584 | [-0.4290, +0.2341] | 0.5140 (0.5140) | OPEN |

## I10RLcKF5Hb20 minus I10RLcHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0036 / +0.0125 | +0.0044 | [-0.0143, +0.0253] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0000 / -0.0031 | -0.0016 | [-0.0051, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0561 / -0.0826 | -0.0693 | [-0.1145, -0.0289] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0409 / +0.0453 | +0.0431 | [+0.0249, +0.0651] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.4513 / -0.4684 | -0.4599 | [-0.7678, -0.1683] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -2.9821 / -3.4919 | -3.2370 | [-4.3918, -2.0364] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0081 / +0.0090 | +0.0086 | [-0.0071, +0.0254] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.6472 / -0.4876 | -0.5674 | [-0.8737, -0.3122] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -2.4192 / -2.6248 | -2.5220 | [-3.6210, -1.4513] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.0160 / +0.0127 | +0.0143 | [+0.0029, +0.0288] | 0.0330 (0.0330) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.0572 / -0.2850 | -0.1711 | [-0.5098, +0.1150] | 0.5140 (0.5140) | OPEN |

## I10RQ50KF5Hb20 minus I10RQ50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0107 / +0.0018 | -0.0044 | [-0.0352, +0.0303] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0047 / +0.0078 | +0.0016 | [-0.0113, +0.0165] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0639 / -0.1184 | -0.0911 | [-0.1465, -0.0387] | 0.0903 (0.0903) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1121 / +0.0663 | +0.0892 | [+0.0462, +0.1400] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | -0.5269 / -0.9251 | -0.7260 | [-1.2282, -0.2430] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -3.5330 / -4.4309 | -3.9820 | [-5.3472, -2.6442] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.0685 / +0.0283 | +0.0484 | [+0.0086, +0.0952] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | -0.7261 / -0.9565 | -0.8413 | [-1.2793, -0.4483] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -2.9056 / -3.4999 | -3.2027 | [-4.5213, -1.9407] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.0281 / +0.0071 | +0.0176 | [+0.0036, +0.0441] | 0.0330 (0.0330) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.1548 / -0.5245 | -0.3396 | [-0.7091, -0.0077] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10R0Hb05 minus I10R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0089 / -0.0036 | -0.0062 | [-0.0145, -0.0009] | 0.0320 (0.0320) | INSIDE-SPREAD |
| C contact rate | -0.0062 / +0.0000 | -0.0031 | [-0.0097, +0.0018] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0093 / +0.0078 | +0.0086 | [-0.0021, +0.0202] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.0635 / +0.0517 | +0.0576 | [+0.0297, +0.0871] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.3166 / +0.1884 | +0.2525 | [+0.0808, +0.4635] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.0181 / +0.0291 | +0.0236 | [-0.0005, +0.0502] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +0.1181 / +0.1758 | +0.1469 | [+0.0132, +0.2985] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.1091 / +0.3495 | +0.2293 | [+0.0305, +0.5444] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10R0Hb10 minus I10R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0089 / +0.0000 | -0.0044 | [-0.0143, +0.0000] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0031 / +0.0016 | -0.0008 | [-0.0070, +0.0043] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0031 / +0.0016 | +0.0023 | [-0.0066, +0.0105] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.0030 / +0.0111 | +0.0070 | [-0.0087, +0.0243] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +0.0257 / +0.0831 | +0.0544 | [-0.0443, +0.1560] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.0031 / +0.0078 | +0.0055 | [-0.0065, +0.0195] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -0.0250 / +0.0698 | +0.0224 | [-0.0646, +0.1209] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.0372 / +0.0856 | +0.0242 | [-0.0872, +0.1980] | 0.5140 (0.5140) | OPEN |

## I10R0Hb40 minus I10R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0036 / +0.0036 | +0.0036 | [-0.0009, +0.0088] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0000 / -0.0062 | -0.0031 | [-0.0099, +0.0021] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0031 / +0.0078 | +0.0023 | [-0.0081, +0.0112] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.0069 / +0.0414 | +0.0241 | [-0.0009, +0.0556] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | -0.0565 / +0.0204 | -0.0181 | [-0.1421, +0.0867] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.0001 / +0.0093 | +0.0046 | [-0.0087, +0.0221] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -0.0057 / +0.0767 | +0.0355 | [-0.0627, +0.1250] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.0662 / +0.0140 | -0.0261 | [-0.1279, +0.0780] | 0.5140 (0.5140) | OPEN |

## I10R0C10XHb20 minus I10R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0516 / -0.0071 | -0.0294 | [-0.0784, +0.0198] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0296 / -0.0047 | -0.0171 | [-0.0428, +0.0039] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.2492 / +0.2368 | +0.2430 | [+0.1919, +0.2994] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +2.7866 / +2.5356 | +2.6611 | [+2.3018, +3.0610] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +7.4418 / +6.4610 | +6.9514 | [+5.2515, +8.7004] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +1.0191 / +1.2977 | +1.1584 | [+0.8129, +1.5097] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +4.9461 / +5.1781 | +5.0621 | [+3.7851, +6.5470] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +2.9542 / +2.2183 | +2.5863 | [+1.8484, +3.4733] | 0.5140 (0.5140) | WORSE |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10R0C10 (0,1) | 0.1539 [0.1079, 0.2022] | 0.0748 [0.0464, 0.1078] | 0.3520 [0.2687, 0.4350] | -0.2236 [-0.2757, -0.1682] | 2.5214 [1.8035, 3.2494] | 6.9994 [4.6887, 9.4838] | 0.4729 [0.4299, 0.5219] | 3.5631 [3.1336, 4.0115] | 12.5603 [9.7445, 15.6164] | 1.0435 [0.7059, 1.4212] | 3.0771 [2.3359, 3.9956] |
| I10R0C10Hb20 (0,1) | 0.0890 [0.0563, 0.1262] | 0.0397 [0.0207, 0.0612] | 0.3793 [0.3075, 0.4481] | -0.2236 [-0.2757, -0.1682] | 2.8511 [2.1548, 3.5435] | 9.8283 [7.6454, 12.2833] | 0.4729 [0.4299, 0.5219] | 3.4829 [2.9868, 4.0090] | 11.4949 [9.7368, 13.5424] | 1.0435 [0.7059, 1.4212] | 4.1750 [3.2460, 5.1506] |
| I10R0C10X (0,1) | 0.1183 [0.0771, 0.1598] | 0.0646 [0.0389, 0.0920] | 0.3559 [0.2850, 0.4274] | -0.2236 [-0.2757, -0.1682] | 2.8768 [2.1777, 3.5833] | 8.5713 [6.1493, 11.2370] | 0.4729 [0.4299, 0.5219] | 3.6155 [3.1480, 4.1029] | 11.3008 [9.6782, 13.2106] | 1.0435 [0.7059, 1.4212] | 3.4403 [2.7090, 4.2487] |
| I10R0C10XHb20 (0,1) | 0.0623 [0.0388, 0.0910] | 0.0280 [0.0123, 0.0462] | 0.3910 [0.3191, 0.4592] | -0.2236 [-0.2757, -0.1682] | 3.0145 [2.3455, 3.6779] | 10.3479 [8.3469, 12.6032] | 0.4729 [0.4299, 0.5219] | 3.5797 [3.0625, 4.1116] | 11.7414 [10.0248, 13.7045] | 1.0435 [0.7059, 1.4212] | 4.8571 [3.8136, 5.9361] |
| I10R0Hb05 (0,1) | 0.0854 [0.0413, 0.1355] | 0.0421 [0.0209, 0.0671] | 0.1565 [0.0946, 0.2258] | -0.2236 [-0.2757, -0.1682] | 0.4110 [-0.3220, 1.1508] | 3.6490 [1.5249, 5.8404] | 0.4729 [0.4299, 0.5219] | 2.4449 [2.1592, 2.7748] | 6.8262 [5.5990, 8.1059] | 1.0435 [0.7059, 1.4212] | 2.5001 [1.7416, 3.4232] |
| I10R0Hb10 (0,1) | 0.0872 [0.0443, 0.1357] | 0.0444 [0.0218, 0.0708] | 0.1503 [0.0889, 0.2162] | -0.2236 [-0.2757, -0.1682] | 0.3604 [-0.3766, 1.1083] | 3.4509 [1.2590, 5.7091] | 0.4729 [0.4299, 0.5219] | 2.4267 [2.1469, 2.7492] | 6.7017 [5.4777, 7.9735] | 1.0435 [0.7059, 1.4212] | 2.2950 [1.6025, 3.0907] |
| I10R0Hb20 (0,1) | 0.0916 [0.0453, 0.1445] | 0.0452 [0.0224, 0.0734] | 0.1480 [0.0872, 0.2156] | -0.2236 [-0.2757, -0.1682] | 0.3534 [-0.3798, 1.0966] | 3.3965 [1.2253, 5.6225] | 0.4729 [0.4299, 0.5219] | 2.4213 [2.1439, 2.7423] | 6.6793 [5.5074, 7.8860] | 1.0435 [0.7059, 1.4212] | 2.2708 [1.6210, 2.9961] |
| I10R0Hb40 (0,1) | 0.0952 [0.0488, 0.1469] | 0.0421 [0.0184, 0.0732] | 0.1503 [0.0849, 0.2240] | -0.2236 [-0.2757, -0.1682] | 0.3775 [-0.3726, 1.1351] | 3.3785 [1.1583, 5.6301] | 0.4729 [0.4299, 0.5219] | 2.4259 [2.1477, 2.7507] | 6.7148 [5.4989, 8.0053] | 1.0435 [0.7059, 1.4212] | 2.2447 [1.5641, 3.0055] |
| I10R0KHb20 (0,1) | 0.0970 [0.0608, 0.1366] | 0.0537 [0.0264, 0.0848] | 0.1067 [0.0705, 0.1471] | -0.2236 [-0.2757, -0.1682] | 0.3222 [-0.1594, 0.7965] | 1.3463 [-0.1516, 2.8420] | 0.4729 [0.4299, 0.5219] | 2.2113 [1.9446, 2.4997] | 5.8179 [5.0804, 6.6424] | 1.0435 [0.7059, 1.4212] | 2.2124 [1.5547, 2.9468] |
| I10RLc (0,1) | 0.0383 [0.0190, 0.0628] | 0.0016 [0.0000, 0.0051] | 0.2181 [0.1459, 0.2897] | 0.2229 [-0.0327, 0.5472] | 2.0938 [1.2319, 3.0706] | 6.7474 [4.5816, 8.9276] | 0.7885 [0.5579, 1.0945] | 3.2710 [2.6177, 4.1001] | 8.0958 [6.4712, 9.8010] | 1.0876 [0.7460, 1.4643] | 2.1535 [1.4770, 2.9189] |
| I10RLcHb20 (0,1) | 0.0320 [0.0127, 0.0567] | 0.0016 [0.0000, 0.0051] | 0.2243 [0.1570, 0.2909] | 0.2229 [-0.0327, 0.5472] | 2.0162 [1.1856, 2.9752] | 6.8750 [4.8011, 8.9005] | 0.7885 [0.5579, 1.0945] | 3.2056 [2.5510, 4.0451] | 8.2087 [6.6620, 9.8507] | 1.0876 [0.7460, 1.4643] | 2.4117 [1.7601, 3.1589] |
| I10RLcKF5Hb20 (0,1) | 0.0365 [0.0166, 0.0590] | 0.0000 [0.0000, 0.0000] | 0.1550 [0.1023, 0.2140] | 0.2660 [-0.0005, 0.6041] | 1.5564 [0.8610, 2.3908] | 3.6380 [2.1931, 5.1343] | 0.7970 [0.5589, 1.1110] | 2.6382 [2.1182, 3.3187] | 5.6866 [4.7784, 6.7972] | 1.1020 [0.7679, 1.4781] | 2.2406 [1.5686, 2.9970] |
| I10RQ50 (0,1) | 0.0801 [0.0480, 0.1174] | 0.0265 [0.0102, 0.0465] | 0.2593 [0.1862, 0.3304] | 0.2697 [0.0335, 0.5486] | 2.4645 [1.6073, 3.4225] | 7.6520 [5.6761, 9.6557] | 0.8241 [0.5966, 1.0699] | 3.7859 [3.0818, 4.6228] | 9.4150 [7.8273, 11.0990] | 1.1816 [0.8261, 1.5783] | 2.5241 [1.8524, 3.2957] |
| I10RQ50Hb20 (0,1) | 0.0783 [0.0451, 0.1153] | 0.0280 [0.0100, 0.0533] | 0.2593 [0.1942, 0.3218] | 0.2697 [0.0335, 0.5486] | 2.3596 [1.5024, 3.3067] | 7.6812 [5.7500, 9.6407] | 0.8241 [0.5966, 1.0699] | 3.7182 [3.0177, 4.5458] | 9.4625 [7.9226, 11.1337] | 1.1816 [0.8261, 1.5783] | 2.7647 [2.0805, 3.5577] |
| I10RQ50KF5Hb20 (0,1) | 0.0738 [0.0427, 0.1106] | 0.0296 [0.0105, 0.0525] | 0.1682 [0.1245, 0.2169] | 0.3589 [0.0841, 0.6833] | 1.6336 [1.0778, 2.2703] | 3.6992 [2.5487, 4.9307] | 0.8726 [0.6109, 1.1655] | 2.8769 [2.4025, 3.4140] | 6.2598 [5.4336, 7.1778] | 1.1992 [0.8358, 1.6032] | 2.4250 [1.7255, 3.2065] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
