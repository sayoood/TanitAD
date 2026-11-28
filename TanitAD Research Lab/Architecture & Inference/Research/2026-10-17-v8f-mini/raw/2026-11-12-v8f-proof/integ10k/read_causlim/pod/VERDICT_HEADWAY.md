# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: /workspace/v8proof/podread/data/join_eval141.npz. Alignment reference (ego speed): /workspace/v8proof/podread/data/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 14, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.03202846975088969, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0CE5:I10R0M5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0CE5:I10R0C5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0CE5:I10R0E5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0CE5:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0CE10:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0CE10:I10R0E10", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10R0CE5 minus I10R0M5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0CE5 minus I10R0C5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0CE5 minus I10R0E5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0CE5 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0CE10 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0CE10 minus I10R0E10 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10R0CE5 minus I10R0M5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.3808 / -0.4947 | -0.4377 | [-0.5504, -0.3377] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.4813 / -0.5592 | -0.5202 | [-0.5962, -0.4450] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0078 / -0.0187 | -0.0055 | [-0.0289, +0.0165] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.5036 / +0.5999 | +0.5518 | [+0.4335, +0.6879] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +8.4121 / +13.6160 | +11.0140 | [+7.1054, +15.0248] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +27.5703 / +47.0785 | +37.3244 | [+24.0685, +50.1689] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | -0.4768 / -0.5728 | -0.5248 | [-0.6401, -0.4224] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -11.4436 / -16.2149 | -13.8292 | [-17.4281, -10.1953] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -37.7726 / -48.1377 | -42.9551 | [-50.7996, -35.3297] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.5855 / -0.2882 | -0.4369 | [-0.7910, -0.1300] | 0.0330 (0.0330) | BETTER |
| time-gap error over the scored steps (s) | -0.7260 / -0.7743 | -0.7501 | [-0.9321, -0.5935] | 0.5140 (0.5140) | BETTER |

## I10R0CE5 minus I10R0C5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.3434 / -0.4733 | -0.4084 | [-0.5230, -0.3036] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.3972 / -0.5218 | -0.4595 | [-0.5594, -0.3584] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0000 / -0.0140 | -0.0070 | [-0.0259, +0.0115] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2327 / +0.3593 | +0.2960 | [+0.2010, +0.4157] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +7.4832 / +11.7305 | +9.6069 | [+6.4462, +12.6732] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +20.5735 / +40.7184 | +30.6459 | [+18.4525, +43.5581] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | -0.1860 / -0.3485 | -0.2672 | [-0.3897, -0.1613] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -7.1154 / -11.5860 | -9.3507 | [-12.5071, -6.0980] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -23.1973 / -42.3047 | -32.7510 | [-44.8457, -21.5378] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.2063 / -0.2304 | -0.2183 | [-0.2922, -0.1454] | 0.0330 (0.0330) | BETTER |
| time-gap error over the scored steps (s) | -0.5519 / -0.6331 | -0.5925 | [-0.7364, -0.4659] | 0.5140 (0.5140) | BETTER |

## I10R0CE5 minus I10R0E5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.1281 / -0.1726 | -0.1504 | [-0.2119, -0.1027] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0935 / -0.0841 | -0.0888 | [-0.1244, -0.0544] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0265 / +0.0000 | +0.0132 | [-0.0098, +0.0426] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2670 / +0.2444 | +0.2557 | [+0.2074, +0.3105] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +1.5409 / +1.5155 | +1.5282 | [+1.2550, +1.8092] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +5.9108 / +4.6539 | +5.2823 | [+4.2488, +6.5756] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | -0.2300 / -0.1963 | -0.2131 | [-0.2621, -0.1697] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -1.5882 / -1.3967 | -1.4925 | [-1.7935, -1.2165] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -4.7428 / -4.2306 | -4.4867 | [-5.4615, -3.7013] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.3416 / +0.0262 | -0.1577 | [-0.5003, +0.2175] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.2097 / -0.1379 | -0.1738 | [-0.2643, -0.0809] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10R0CE5 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.3238 / +0.2883 | +0.3060 | [+0.2173, +0.3876] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0919 / +0.1308 | +0.1114 | [+0.0594, +0.1653] | 0.0343 (0.0343) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0171 / -0.1433 | -0.0802 | [-0.1759, +0.0112] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.3197 / -0.3370 | -0.3284 | [-0.4093, -0.2440] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -4.7936 / -5.6161 | -5.2048 | [-6.0536, -4.3574] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -11.8807 / -15.3415 | -13.6111 | [-16.7142, -10.1315] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.3558 / +0.3847 | +0.3702 | [+0.3115, +0.4312] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +3.4597 / +2.9128 | +3.1863 | [+2.5236, +3.8469] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +9.2916 / +5.9989 | +7.6452 | [+4.7899, +10.3503] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.2165 / +0.5235 | +0.3700 | [-0.0417, +0.9822] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.1191 / -0.2913 | -0.0861 | [-0.6068, +0.3303] | 0.5140 (0.5140) | OPEN |

## I10R0CE10 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0872 / -0.0765 | -0.0819 | [-0.1261, -0.0416] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0374 / -0.0265 | -0.0319 | [-0.0530, -0.0134] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.2617 / +0.2726 | +0.2671 | [+0.2160, +0.3186] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0444 / +0.0600 | +0.0522 | [+0.0333, +0.0784] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.7113 / +2.4598 | +2.5856 | [+2.2217, +2.9876] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +8.4129 / +7.5749 | +7.9939 | [+6.5918, +9.5515] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | -0.0161 / -0.0189 | -0.0175 | [-0.0298, -0.0087] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | +1.0363 / +1.2795 | +1.1579 | [+0.8248, +1.4999] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +4.7979 / +5.1042 | +4.9510 | [+3.7276, +6.3499] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.0056 / +0.0081 | +0.0068 | [+0.0035, +0.0117] | 0.0330 (0.0330) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.7318 / +1.6971 | +1.7145 | [+1.3163, +2.1643] | 0.5140 (0.5140) | WORSE |

## I10R0CE10 minus I10R0E10

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0445 / -0.0320 | -0.0383 | [-0.0750, -0.0065] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0171 / -0.0140 | -0.0156 | [-0.0324, -0.0005] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.2399 / +0.2383 | +0.2391 | [+0.1895, +0.2907] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +2.4602 / +2.2294 | +2.3448 | [+2.0183, +2.7058] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +7.4766 / +6.5754 | +7.0260 | [+5.4828, +8.7403] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.9994 / +1.2266 | +1.1130 | [+0.7979, +1.4345] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +4.6176 / +4.7045 | +4.6610 | [+3.4976, +6.0487] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +1.6523 / +1.2380 | +1.4452 | [+0.9220, +1.9790] | 0.5140 (0.5140) | WORSE |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10R0C5 (0,1) | 0.8443 [0.7805, 0.8992] | 0.6301 [0.5222, 0.7273] | 0.0670 [0.0363, 0.1056] | -0.8479 [-0.9911, -0.7285] | -14.3635 [-17.4989, -11.2362] | -41.4629 [-55.0771, -28.6468] | 1.1104 [0.9710, 1.2809] | 15.0290 [11.8692, 18.1955] | 47.2741 [36.8435, 58.7339] | 1.6318 [1.0542, 2.4072] | 2.5482 [1.7862, 3.5246] |
| I10R0CE10 (0,1) | 0.0480 [0.0230, 0.0761] | 0.0273 [0.0088, 0.0507] | 0.4073 [0.3175, 0.4935] | -0.1713 [-0.2304, -0.1092] | 3.0338 [2.2990, 3.7660] | 10.7880 [8.3591, 13.2688] | 0.4554 [0.4151, 0.5001] | 3.6499 [3.1138, 4.2105] | 11.8289 [9.9867, 13.8021] | 1.0503 [0.7100, 1.4289] | 3.7563 [2.9500, 4.6622] |
| I10R0CE5 (0,1) | 0.4359 [0.3368, 0.5249] | 0.1706 [0.1156, 0.2288] | 0.0600 [0.0281, 0.1022] | -0.5519 [-0.6532, -0.4436] | -4.7566 [-5.4835, -3.9704] | -10.8170 [-13.1702, -8.2361] | 0.8431 [0.7740, 0.9183] | 5.6783 [5.1873, 6.2124] | 14.5231 [12.7769, 16.2127] | 1.4135 [0.8849, 2.1790] | 1.9557 [1.2803, 2.8387] |
| I10R0E10 (0,1) | 0.0863 [0.0460, 0.1315] | 0.0428 [0.0185, 0.0740] | 0.1682 [0.0878, 0.2535] | -0.1713 [-0.2304, -0.1092] | 0.6890 [-0.0968, 1.4984] | 3.7620 [1.1637, 6.4667] | 0.4554 [0.4151, 0.5001] | 2.5369 [2.2161, 2.9066] | 7.1678 [5.7232, 8.7185] | 1.0503 [0.7100, 1.4289] | 2.3111 [1.5041, 3.2560] |
| I10R0E5 (0,1) | 0.5863 [0.5080, 0.6549] | 0.2593 [0.1913, 0.3263] | 0.0467 [0.0238, 0.0776] | -0.8076 [-0.9135, -0.6925] | -6.2848 [-7.0702, -5.4625] | -16.0993 [-18.7639, -13.2846] | 1.0563 [0.9822, 1.1325] | 7.1708 [6.5711, 7.8093] | 19.0097 [16.8729, 21.2217] | 1.5711 [1.0318, 2.2311] | 2.1296 [1.4322, 3.0257] |
| I10R0M5 (0,1) | 0.8737 [0.8176, 0.9191] | 0.6908 [0.6102, 0.7630] | 0.0654 [0.0378, 0.0987] | -1.1037 [-1.2468, -0.9695] | -15.7706 [-19.8054, -11.7181] | -48.1414 [-61.7632, -33.8481] | 1.3679 [1.2375, 1.5208] | 19.5076 [16.1343, 22.9679] | 57.4782 [50.7509, 64.8401] | 1.8503 [1.2289, 2.6007] | 2.7059 [1.8928, 3.7647] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
