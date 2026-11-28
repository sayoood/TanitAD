# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: /workspace/v8proof/podread/data/join_eval141.npz. Alignment reference (ego speed): /workspace/v8proof/podread/data/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 12, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.03202846975088969, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0M5:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcM5:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50M5:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcM5:I10R0M5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50M5:I10R0M5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLc:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50:I10R0", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10R0M5 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcM5 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50M5 minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcM5 minus I10R0M5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50M5 minus I10R0M5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLc minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

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

## I10RLcM5 minus I10R0M5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.3808 / -0.3932 | -0.3870 | [-0.4477, -0.3273] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.6324 / -0.7118 | -0.6721 | [-0.7414, -0.5945] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1604 / +0.1604 | +0.1604 | [+0.0951, +0.2265] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7808 / +0.8059 | +0.7933 | [+0.3927, +1.2658] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +11.9947 / +15.3529 | +13.6738 | [+9.6686, +17.6433] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +41.9216 / +59.3220 | +50.6218 | [+36.7735, +63.9954] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.3187 / +0.2303 | +0.2745 | [-0.0226, +0.6324] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -8.7684 / -13.4129 | -11.0907 | [-14.9208, -7.4184] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -36.8284 / -46.6176 | -41.7230 | [-49.9999, -33.3820] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.0622 / -0.0785 | -0.0704 | [-0.1669, +0.0250] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.3699 / -0.4934 | -0.4316 | [-0.5669, -0.2962] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50M5 minus I10R0M5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.3060 / -0.2900 | -0.2980 | [-0.3568, -0.2437] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.5421 / -0.6059 | -0.5740 | [-0.6446, -0.4965] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.2103 / +0.1542 | +0.1822 | [+0.1127, +0.2589] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.2324 / +0.7786 | +1.0055 | [+0.4933, +1.5609] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +14.4062 / +15.3732 | +14.8897 | [+11.5389, +18.2151] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +43.6942 / +57.4043 | +50.5492 | [+37.6958, +62.2189] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.6116 / +0.2722 | +0.4419 | [+0.0705, +0.8538] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | -7.0789 / -13.3319 | -10.2054 | [-14.7939, -5.7383] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -30.0637 / -44.7130 | -37.3883 | [-48.3098, -25.8079] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.5138 / +0.2159 | +0.3649 | [+0.1020, +0.7798] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | -0.1138 / -0.3030 | -0.2084 | [-0.3721, -0.0057] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RLc minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0996 / -0.0836 | -0.0916 | [-0.1372, -0.0509] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0763 / -0.0389 | -0.0576 | [-0.0966, -0.0270] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0888 / +0.0670 | +0.0779 | [+0.0369, +0.1246] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4463 / +0.4467 | +0.4465 | [+0.1858, +0.7734] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +1.9097 / +1.3814 | +1.6455 | [+0.7390, +2.7288] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +4.7373 / +3.1694 | +3.9533 | [+2.0177, +6.2407] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.3151 / +0.3162 | +0.3156 | [+0.0869, +0.6111] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.8592 / +0.6987 | +0.7789 | [+0.1714, +1.5388] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +1.4548 / +0.9811 | +1.2180 | [+0.1267, +2.5314] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0426 / +0.0456 | +0.0441 | [+0.0199, +0.0728] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.1035 / +0.1198 | +0.1116 | [+0.0165, +0.2114] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0587 / -0.0409 | -0.0498 | [-0.0868, -0.0174] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0452 / -0.0202 | -0.0327 | [-0.0577, -0.0128] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1324 / +0.1059 | +0.1192 | [+0.0710, +0.1710] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6020 / +0.3845 | +0.4933 | [+0.2312, +0.7833] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.4132 / +1.6194 | +2.0163 | [+1.0353, +3.1714] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +6.0722 / +3.6437 | +4.8579 | [+2.5270, +7.5458] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.4406 / +0.2619 | +0.3513 | [+0.1357, +0.5963] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +1.4321 / +1.1556 | +1.2939 | [+0.6447, +2.0721] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +3.0547 / +2.0196 | +2.5372 | [+1.1229, +4.2045] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.1783 / +0.0979 | +0.1381 | [+0.0529, +0.2535] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.5299 / +0.4347 | +0.4823 | [+0.2771, +0.7058] | 0.5140 (0.5140) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10R0M5 (0,1) | 0.8737 [0.8176, 0.9191] | 0.6908 [0.6102, 0.7630] | 0.0654 [0.0378, 0.0987] | -1.1037 [-1.2468, -0.9695] | -15.7706 [-19.8054, -11.7181] | -48.1414 [-61.7632, -33.8481] | 1.3679 [1.2375, 1.5208] | 19.5076 [16.1343, 22.9679] | 57.4782 [50.7509, 64.8401] | 1.8503 [1.2289, 2.6007] | 2.7059 [1.8928, 3.7647] |
| I10RLc (0,1) | 0.0383 [0.0190, 0.0628] | 0.0016 [0.0000, 0.0051] | 0.2181 [0.1459, 0.2897] | 0.2229 [-0.0327, 0.5472] | 2.0938 [1.2319, 3.0706] | 6.7474 [4.5816, 8.9276] | 0.7885 [0.5579, 1.0945] | 3.2710 [2.6177, 4.1001] | 8.0958 [6.4712, 9.8010] | 1.0876 [0.7460, 1.4643] | 2.1535 [1.4770, 2.9189] |
| I10RLcM5 (0,1) | 0.4867 [0.4184, 0.5555] | 0.0187 [0.0108, 0.0273] | 0.2259 [0.1722, 0.2837] | -0.3104 [-0.7041, 0.1461] | -2.0968 [-4.2249, 0.0735] | 2.4804 [-1.7053, 6.8513] | 1.6424 [1.3486, 2.0128] | 8.4169 [7.2754, 9.7230] | 15.7552 [13.1288, 18.7193] | 1.7800 [1.1768, 2.5370] | 2.2743 [1.5065, 3.3193] |
| I10RQ50 (0,1) | 0.0801 [0.0480, 0.1174] | 0.0265 [0.0102, 0.0465] | 0.2593 [0.1862, 0.3304] | 0.2697 [0.0335, 0.5486] | 2.4645 [1.6073, 3.4225] | 7.6520 [5.6761, 9.6557] | 0.8241 [0.5966, 1.0699] | 3.7859 [3.0818, 4.6228] | 9.4150 [7.8273, 11.0990] | 1.1816 [0.8261, 1.5783] | 2.5241 [1.8524, 3.2957] |
| I10RQ50M5 (0,1) | 0.5756 [0.5018, 0.6486] | 0.1168 [0.0813, 0.1546] | 0.2477 [0.1845, 0.3185] | -0.0982 [-0.6460, 0.4743] | -0.8809 [-4.0130, 2.4903] | 2.4079 [-3.5213, 9.0091] | 1.8098 [1.4772, 2.1781] | 9.3022 [7.6778, 11.2168] | 20.0898 [15.1278, 25.9184] | 2.2152 [1.5551, 3.0144] | 2.4975 [1.7350, 3.5154] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
