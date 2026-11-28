# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: /workspace/v8proof/podread/data/join_eval141.npz. Alignment reference (ego speed): /workspace/v8proof/podread/data/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 20, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.03202846975088969, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcK:I10R0G0X", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50K:I10R0G0X", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcKF5:I10RLcK", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50KF5:I10RQ50K", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcKF5:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50KF5:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcK:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50K:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcKF5:I10RLcF5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50KF5:I10RQ50F5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50KF5:I10RLcKF5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcKF5:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50KF5:I10R0", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10RLcK minus I10R0G0X over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50K minus I10R0G0X over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcKF5 minus I10RLcK over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50KF5 minus I10RQ50K over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcKF5 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50KF5 minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcK minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50K minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcKF5 minus I10RLcF5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50KF5 minus I10RQ50F5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50KF5 minus I10RLcKF5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcKF5 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50KF5 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10RLcK minus I10R0G0X

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0569 / -0.0445 | -0.0507 | [-0.0739, -0.0310] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0561 / -0.0514 | -0.0537 | [-0.0823, -0.0281] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0436 / +0.0483 | +0.0460 | [+0.0124, +0.0904] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4463 / +0.4467 | +0.4465 | [+0.1858, +0.7734] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +1.1028 / +1.0172 | +1.0600 | [+0.4499, +1.7923] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.4442 / +2.2280 | +2.3361 | [+1.3611, +3.4843] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3151 / +0.3162 | +0.3156 | [+0.0869, +0.6111] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.3437 / +0.3959 | +0.3698 | [-0.1068, +0.9771] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -0.3190 / +0.0739 | -0.1226 | [-0.9170, +0.7930] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0426 / +0.0456 | +0.0441 | [+0.0199, +0.0728] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | -0.0210 / +0.0389 | +0.0089 | [-0.0620, +0.0870] | 0.5140 (0.5140) | OPEN |

## I10RQ50K minus I10R0G0X

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0285 / -0.0107 | -0.0196 | [-0.0409, -0.0055] | 0.0320 (0.0320) | INSIDE-SPREAD |
| C contact rate | -0.0187 / -0.0156 | -0.0171 | [-0.0280, -0.0084] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0717 / +0.0436 | +0.0576 | [+0.0255, +0.0947] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6020 / +0.3845 | +0.4933 | [+0.2312, +0.7833] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +1.3466 / +0.7400 | +1.0433 | [+0.4616, +1.7244] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.8039 / +1.5120 | +2.1579 | [+1.1027, +3.3827] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.4406 / +0.2619 | +0.3513 | [+0.1357, +0.5963] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.7692 / +0.4071 | +0.5881 | [+0.1963, +1.0229] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +0.7304 / +0.2792 | +0.5048 | [-0.0428, +1.1026] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.1783 / +0.0979 | +0.1381 | [+0.0529, +0.2535] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.4047 / +0.2486 | +0.3266 | [+0.1352, +0.5697] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RLcKF5 minus I10RLcK

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0160 / -0.0107 | -0.0133 | [-0.0235, -0.0049] | 0.0320 (0.0320) | INSIDE-SPREAD |
| C contact rate | -0.0047 / -0.0031 | -0.0039 | [-0.0090, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0047 / +0.0062 | +0.0055 | [+0.0008, +0.0111] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0409 / +0.0453 | +0.0431 | [+0.0249, +0.0651] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.1136 / +0.1191 | +0.1163 | [+0.0668, +0.1740] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.1760 / +0.1341 | +0.1550 | [+0.0685, +0.2538] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0081 / +0.0090 | +0.0086 | [-0.0071, +0.0254] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.0203 / +0.0289 | +0.0246 | [-0.0176, +0.0720] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +0.0177 / +0.0270 | +0.0224 | [-0.0512, +0.1087] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0160 / +0.0127 | +0.0143 | [+0.0029, +0.0288] | 0.0330 (0.0330) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0133 / +0.0128 | +0.0131 | [+0.0034, +0.0251] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50KF5 minus I10RQ50K

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0071 / -0.0071 | -0.0071 | [-0.0128, -0.0020] | 0.0320 (0.0320) | INSIDE-SPREAD |
| C contact rate | -0.0109 / -0.0109 | -0.0109 | [-0.0192, -0.0040] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0125 / +0.0062 | +0.0093 | [+0.0014, +0.0173] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1121 / +0.0663 | +0.0892 | [+0.0462, +0.1400] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +0.2927 / +0.1661 | +0.2294 | [+0.1136, +0.3609] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.5615 / +0.3844 | +0.4729 | [+0.2713, +0.6894] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0685 / +0.0283 | +0.0484 | [+0.0086, +0.0952] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.0947 / +0.0141 | +0.0544 | [-0.0480, +0.1744] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +0.1044 / -0.0262 | +0.0391 | [-0.1693, +0.2630] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0281 / +0.0071 | +0.0176 | [+0.0036, +0.0441] | 0.0330 (0.0330) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.0183 / +0.0047 | -0.0068 | [-0.0455, +0.0217] | 0.5140 (0.5140) | OPEN |

## I10RLcKF5 minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0071 / +0.0036 | -0.0018 | [-0.0183, +0.0169] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0000 / -0.0031 | -0.0016 | [-0.0051, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0530 / -0.0826 | -0.0678 | [-0.1149, -0.0249] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0409 / +0.0453 | +0.0431 | [+0.0249, +0.0651] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.4357 / -0.4734 | -0.4546 | [-0.7588, -0.1589] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -2.7508 / -3.5421 | -3.1465 | [-4.3765, -1.9094] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0081 / +0.0090 | +0.0086 | [-0.0071, +0.0254] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.6312 / -0.4930 | -0.5621 | [-0.8673, -0.3052] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -2.2000 / -2.6786 | -2.4393 | [-3.5292, -1.3628] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.0160 / +0.0127 | +0.0143 | [+0.0029, +0.0288] | 0.0330 (0.0330) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.1886 / -0.3457 | -0.2672 | [-0.5236, -0.0328] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50KF5 minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0107 / -0.0018 | -0.0062 | [-0.0372, +0.0269] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0000 / +0.0062 | +0.0031 | [-0.0083, +0.0167] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0607 / -0.1262 | -0.0935 | [-0.1539, -0.0354] | 0.0903 (0.0903) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1121 / +0.0663 | +0.0892 | [+0.0462, +0.1400] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | -0.5164 / -0.9414 | -0.7289 | [-1.2397, -0.2255] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -3.3405 / -4.4822 | -3.9114 | [-5.3231, -2.4952] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.0685 / +0.0283 | +0.0484 | [+0.0086, +0.0952] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | -0.7043 / -0.9535 | -0.8289 | [-1.2823, -0.4232] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -2.6638 / -3.5650 | -3.1144 | [-4.4606, -1.7702] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.0281 / +0.0071 | +0.0176 | [+0.0036, +0.0441] | 0.0330 (0.0330) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.2209 / -0.4590 | -0.3400 | [-0.6474, -0.0633] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RLcK minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0089 / +0.0142 | +0.0116 | [-0.0049, +0.0310] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0047 / +0.0000 | +0.0023 | [-0.0021, +0.0079] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0576 / -0.0888 | -0.0732 | [-0.1192, -0.0303] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.5493 / -0.5925 | -0.5709 | [-0.8985, -0.2591] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -2.9268 / -3.6762 | -3.3015 | [-4.5405, -2.0293] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.6515 / -0.5219 | -0.5867 | [-0.9063, -0.3201] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -2.2177 / -2.7056 | -2.4616 | [-3.5740, -1.3529] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.2020 / -0.3585 | -0.2802 | [-0.5409, -0.0414] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50K minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0036 / +0.0053 | +0.0009 | [-0.0301, +0.0334] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0109 / +0.0171 | +0.0140 | [+0.0018, +0.0307] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0732 / -0.1324 | -0.1028 | [-0.1639, -0.0465] | 0.0903 (0.0903) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.8091 / -1.1075 | -0.9583 | [-1.4894, -0.4776] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -3.9020 / -4.8666 | -4.3843 | [-5.8589, -2.9519] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.7990 / -0.9676 | -0.8833 | [-1.3780, -0.4296] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -2.7682 / -3.5388 | -3.1535 | [-4.6139, -1.7785] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.2027 / -0.4637 | -0.3332 | [-0.6479, -0.0520] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RLcKF5 minus I10RLcF5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0053 / +0.0089 | +0.0071 | [-0.0082, +0.0248] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0545 / -0.0872 | -0.0709 | [-0.1199, -0.0276] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.5010 / -0.4826 | -0.4918 | [-0.8016, -0.2043] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -2.7653 / -3.6706 | -3.2180 | [-4.4933, -1.9156] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.6287 / -0.4662 | -0.5475 | [-0.8725, -0.3006] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -2.1521 / -2.7958 | -2.4739 | [-3.6347, -1.3214] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.2055 / -0.3662 | -0.2859 | [-0.5486, -0.0532] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50KF5 minus I10RQ50F5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0018 / +0.0036 | +0.0027 | [-0.0257, +0.0322] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0156 / +0.0109 | +0.0132 | [+0.0002, +0.0285] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0592 / -0.1293 | -0.0942 | [-0.1577, -0.0345] | 0.0903 (0.0903) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.8963 / -1.1944 | -1.0454 | [-1.5526, -0.5942] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -3.5249 / -4.8900 | -4.2075 | [-5.7667, -2.7131] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.9168 / -1.0707 | -0.9937 | [-1.4912, -0.5530] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -2.6835 / -3.7337 | -3.2086 | [-4.6523, -1.8699] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.2626 / -0.5371 | -0.3998 | [-0.7139, -0.1167] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50KF5 minus I10RLcKF5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0374 / +0.0374 | +0.0374 | [+0.0210, +0.0564] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0312 / +0.0280 | +0.0296 | [+0.0105, +0.0525] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0358 / -0.0047 | +0.0156 | [-0.0242, +0.0541] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2269 / -0.0411 | +0.0929 | [-0.1062, +0.3382] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.4229 / -0.2301 | +0.0964 | [-0.4148, +0.6714] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +0.7451 / -0.4657 | +0.1397 | [-0.7526, +1.1462] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.1860 / -0.0349 | +0.0755 | [-0.0960, +0.2893] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.4998 / -0.0037 | +0.2481 | [-0.1907, +0.7251] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +1.1360 / +0.1522 | +0.6441 | [-0.1136, +1.5006] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.1478 / +0.0466 | +0.0972 | [-0.0058, +0.2344] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.3941 / +0.2015 | +0.2978 | [+0.0885, +0.5453] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RLcKF5 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.1068 / -0.0801 | -0.0934 | [-0.1440, -0.0463] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0763 / -0.0421 | -0.0592 | [-0.0966, -0.0288] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0358 / -0.0156 | +0.0101 | [-0.0566, +0.0740] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4872 / +0.4919 | +0.4896 | [+0.2142, +0.8279] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +1.4739 / +0.9080 | +1.1910 | [+0.3324, +2.1135] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +1.9865 / -0.3728 | +0.8069 | [-1.4425, +3.0285] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.3231 / +0.3252 | +0.3242 | [+0.0855, +0.6310] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.2280 / +0.2058 | +0.2169 | [-0.2910, +0.8313] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -0.7451 / -1.6975 | -1.2213 | [-2.3199, -0.2282] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0586 / +0.0584 | +0.0585 | [+0.0316, +0.0900] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | -0.0852 / -0.2258 | -0.1555 | [-0.4247, +0.0707] | 0.5140 (0.5140) | OPEN |

## I10RQ50KF5 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0694 / -0.0427 | -0.0560 | [-0.1064, -0.0098] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0452 / -0.0140 | -0.0296 | [-0.0601, -0.0050] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0717 / -0.0202 | +0.0257 | [-0.0539, +0.0975] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7142 / +0.4508 | +0.5825 | [+0.2805, +0.9190] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +1.8968 / +0.6779 | +1.2874 | [+0.2839, +2.3705] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +2.7316 / -0.8385 | +0.9466 | [-1.7805, +3.6377] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.5091 / +0.2902 | +0.3997 | [+0.1473, +0.6846] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.7278 / +0.2021 | +0.4649 | [-0.0381, +0.9837] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +0.3909 / -1.5454 | -0.5772 | [-2.0889, +0.8096] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.2064 / +0.1050 | +0.1557 | [+0.0593, +0.2930] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.3089 / -0.0243 | +0.1423 | [-0.2658, +0.5464] | 0.5140 (0.5140) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10R0G0X (0,1) | 0.1005 [0.0646, 0.1408] | 0.0576 [0.0295, 0.0883] | 0.0989 [0.0615, 0.1416] | -0.2236 [-0.2757, -0.1682] | 0.4629 [-0.0807, 1.0171] | 1.1098 [-0.4325, 2.6695] | 0.4729 [0.4299, 0.5219] | 2.3145 [2.0317, 2.6434] | 5.7567 [5.0133, 6.5885] | 1.0435 [0.7059, 1.4212] | 1.8643 [1.2610, 2.5995] |
| I10RLc (0,1) | 0.0383 [0.0190, 0.0628] | 0.0016 [0.0000, 0.0051] | 0.2181 [0.1459, 0.2897] | 0.2229 [-0.0327, 0.5472] | 2.0938 [1.2319, 3.0706] | 6.7474 [4.5816, 8.9276] | 0.7885 [0.5579, 1.0945] | 3.2710 [2.6177, 4.1001] | 8.0958 [6.4712, 9.8010] | 1.0876 [0.7460, 1.4643] | 2.1535 [1.4770, 2.9189] |
| I10RLcF5 (0,1) | 0.0294 [0.0161, 0.0468] | 0.0000 [0.0000, 0.0000] | 0.2212 [0.1473, 0.2949] | 0.2660 [-0.0005, 0.6041] | 2.1311 [1.3035, 3.1056] | 6.8189 [4.6041, 9.0795] | 0.7970 [0.5589, 1.1110] | 3.2564 [2.6256, 4.0653] | 8.1304 [6.4624, 9.9422] | 1.1020 [0.7679, 1.4781] | 2.1722 [1.4940, 2.9385] |
| I10RLcK (0,1) | 0.0498 [0.0240, 0.0801] | 0.0039 [0.0000, 0.0090] | 0.1449 [0.0925, 0.2034] | 0.2229 [-0.0327, 0.5472] | 1.5229 [0.8322, 2.3450] | 3.4459 [1.9386, 4.9996] | 0.7885 [0.5579, 1.0945] | 2.6843 [2.1843, 3.3389] | 5.6341 [4.7058, 6.7572] | 1.0876 [0.7460, 1.4643] | 1.8733 [1.2520, 2.6364] |
| I10RLcKF5 (0,1) | 0.0365 [0.0166, 0.0590] | 0.0000 [0.0000, 0.0000] | 0.1503 [0.0957, 0.2124] | 0.2660 [-0.0005, 0.6041] | 1.6392 [0.9187, 2.4925] | 3.6010 [2.0988, 5.1610] | 0.7970 [0.5589, 1.1110] | 2.7089 [2.1777, 3.4010] | 5.6565 [4.6924, 6.8198] | 1.1020 [0.7679, 1.4781] | 1.8863 [1.2658, 2.6506] |
| I10RQ50 (0,1) | 0.0801 [0.0480, 0.1174] | 0.0265 [0.0102, 0.0465] | 0.2593 [0.1862, 0.3304] | 0.2697 [0.0335, 0.5486] | 2.4645 [1.6073, 3.4225] | 7.6520 [5.6761, 9.6557] | 0.8241 [0.5966, 1.0699] | 3.7859 [3.0818, 4.6228] | 9.4150 [7.8273, 11.0990] | 1.1816 [0.8261, 1.5783] | 2.5241 [1.8524, 3.2957] |
| I10RQ50F5 (0,1) | 0.0712 [0.0449, 0.1013] | 0.0164 [0.0049, 0.0307] | 0.2601 [0.1836, 0.3338] | 0.3589 [0.0841, 0.6833] | 2.7810 [1.9098, 3.7616] | 7.9481 [5.9245, 10.0877] | 0.8726 [0.6109, 1.1655] | 3.9507 [3.2072, 4.8369] | 9.5092 [7.9092, 11.2367] | 1.1992 [0.8358, 1.6032] | 2.5840 [1.8978, 3.3631] |
| I10RQ50K (0,1) | 0.0810 [0.0492, 0.1184] | 0.0405 [0.0173, 0.0668] | 0.1565 [0.1142, 0.2009] | 0.2697 [0.0335, 0.5486] | 1.5062 [0.9842, 2.0822] | 3.2677 [2.0400, 4.5163] | 0.8241 [0.5966, 1.0699] | 2.9026 [2.4956, 3.3433] | 6.2615 [5.4974, 7.0703] | 1.1816 [0.8261, 1.5783] | 2.1909 [1.5369, 2.9299] |
| I10RQ50KF5 (0,1) | 0.0738 [0.0431, 0.1106] | 0.0296 [0.0105, 0.0525] | 0.1659 [0.1223, 0.2129] | 0.3589 [0.0841, 0.6833] | 1.7356 [1.1796, 2.3753] | 3.7406 [2.5599, 4.9933] | 0.8726 [0.6109, 1.1655] | 2.9570 [2.4927, 3.4815] | 6.3006 [5.4689, 7.2014] | 1.1992 [0.8358, 1.6032] | 2.1841 [1.5283, 2.9270] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
