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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcB8:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50B8:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcB10:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcB10:I10RLcB8", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcB8:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50B8:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50B8:I10RLcB8", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10RLcB8 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50B8 minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcB10 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcB10 minus I10RLcB8 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcB8 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50B8 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50B8 minus I10RLcB8 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10RLcB8 minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0071 / +0.0125 | +0.0098 | [+0.0011, +0.0208] | 0.0320 (0.0320) | INSIDE-SPREAD |
| C contact rate | +0.0000 / -0.0031 | -0.0016 | [-0.0051, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0016 / +0.0078 | +0.0047 | [-0.0046, +0.0159] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0217 / +0.0222 | +0.0220 | [+0.0076, +0.0417] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.3945 / +0.5540 | +0.4743 | [+0.0650, +0.9683] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.0802 / +1.1417 | +1.1110 | [+0.1218, +2.3660] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0175 / +0.0145 | +0.0160 | [+0.0019, +0.0349] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.3864 / +0.4927 | +0.4396 | [+0.0421, +0.9182] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +1.1380 / +1.1654 | +1.1517 | [+0.1640, +2.4150] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0027 / +0.0002 | +0.0014 | [-0.0027, +0.0056] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.0457 / +0.0692 | +0.0574 | [-0.0005, +0.1345] | 0.5140 (0.5140) | OPEN |

## I10RQ50B8 minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0036 / +0.0053 | +0.0009 | [-0.0084, +0.0136] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0031 / +0.0000 | +0.0016 | [-0.0124, +0.0155] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0257 / +0.0215 | +0.0236 | [+0.0115, +0.0373] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.6303 / +0.2767 | +0.4535 | [+0.0763, +0.9228] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.8677 / +0.1273 | +0.4975 | [-0.2598, +1.3499] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0213 / +0.0170 | +0.0191 | [+0.0073, +0.0327] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.7186 / +0.3408 | +0.5297 | [+0.1468, +1.0107] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +0.8471 / +0.0414 | +0.4442 | [-0.2952, +1.3004] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | -0.0136 / +0.0052 | -0.0042 | [-0.0272, +0.0075] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.2286 / +0.0264 | +0.1275 | [-0.0037, +0.2879] | 0.5140 (0.5140) | OPEN |

## I10RLcB10 minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0089 / +0.0142 | +0.0116 | [+0.0015, +0.0245] | 0.0320 (0.0320) | INSIDE-SPREAD |
| C contact rate | +0.0016 / -0.0016 | +0.0000 | [-0.0029, +0.0030] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0016 / +0.0062 | +0.0023 | [-0.0089, +0.0179] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0524 / +0.0505 | +0.0515 | [+0.0240, +0.0862] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +0.2103 / +0.3384 | +0.2743 | [-0.1421, +0.7462] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +0.0433 / +0.9081 | +0.4757 | [-0.2528, +1.6283] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0420 / +0.0401 | +0.0410 | [+0.0131, +0.0771] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.2113 / +0.2775 | +0.2444 | [-0.1437, +0.6810] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +0.0997 / +0.9521 | +0.5259 | [-0.1832, +1.7063] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0085 / +0.0066 | +0.0076 | [+0.0022, +0.0135] | 0.0330 (0.0330) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0828 / +0.0596 | +0.0712 | [-0.0030, +0.1600] | 0.5140 (0.5140) | OPEN |

## I10RLcB10 minus I10RLcB8

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0018 / +0.0018 | +0.0018 | [-0.0052, +0.0090] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0016 / +0.0016 | +0.0016 | [+0.0000, +0.0040] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0031 / -0.0016 | -0.0023 | [-0.0130, +0.0075] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0307 / +0.0283 | +0.0295 | [+0.0129, +0.0498] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.1843 / -0.2156 | -0.1999 | [-0.6569, +0.1657] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | -1.0369 / -0.2336 | -0.6353 | [-1.6487, -0.0008] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0244 / +0.0256 | +0.0250 | [+0.0083, +0.0454] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | -0.1752 / -0.2152 | -0.1952 | [-0.6371, +0.1630] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -1.0383 / -0.2133 | -0.6258 | [-1.6143, +0.0087] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0058 / +0.0064 | +0.0061 | [+0.0025, +0.0108] | 0.0330 (0.0330) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0371 / -0.0095 | +0.0138 | [-0.0321, +0.0730] | 0.5140 (0.5140) | OPEN |

## I10RLcB8 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0925 / -0.0712 | -0.0819 | [-0.1257, -0.0442] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0763 / -0.0421 | -0.0592 | [-0.0966, -0.0288] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0903 / +0.0748 | +0.0826 | [+0.0378, +0.1321] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4681 / +0.4689 | +0.4685 | [+0.1962, +0.8057] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.3042 / +1.9354 | +2.1198 | [+0.9001, +3.5894] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +5.8175 / +4.3111 | +5.0643 | [+2.2843, +8.3433] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.3326 / +0.3307 | +0.3316 | [+0.0929, +0.6378] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +1.2456 / +1.1914 | +1.2185 | [+0.3077, +2.4184] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +2.5928 / +2.1464 | +2.3696 | [+0.4556, +4.6431] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.0453 / +0.0458 | +0.0456 | [+0.0212, +0.0741] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.1491 / +0.1890 | +0.1691 | [+0.0489, +0.3122] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50B8 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0623 / -0.0356 | -0.0489 | [-0.0886, -0.0130] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0452 / -0.0202 | -0.0327 | [-0.0577, -0.0128] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1355 / +0.1059 | +0.1207 | [+0.0723, +0.1744] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6277 / +0.4060 | +0.5169 | [+0.2475, +0.8183] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +3.0435 / +1.8960 | +2.4698 | [+1.1650, +3.9725] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +6.9398 / +3.7710 | +5.3554 | [+2.5761, +8.5211] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.4619 / +0.2789 | +0.3704 | [+0.1470, +0.6248] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +2.1507 / +1.4964 | +1.8236 | [+0.8574, +2.9418] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +3.9018 / +2.0610 | +2.9814 | [+1.1310, +5.1917] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.1648 / +0.1031 | +0.1339 | [+0.0562, +0.2311] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.7585 / +0.4611 | +0.6098 | [+0.3109, +0.9224] | 0.5140 (0.5140) | WORSE |

## I10RQ50B8 minus I10RLcB8

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0302 / +0.0356 | +0.0329 | [+0.0134, +0.0554] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0312 / +0.0218 | +0.0265 | [+0.0102, +0.0465] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0452 / +0.0312 | +0.0382 | [+0.0104, +0.0701] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1596 / -0.0629 | +0.0484 | [-0.1396, +0.2723] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.7393 / -0.0394 | +0.3500 | [-0.4127, +1.3985] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +1.1223 / -0.5401 | +0.2911 | [-1.3700, +2.3558] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.1293 / -0.0517 | +0.0388 | [-0.1285, +0.2311] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.9051 / +0.3050 | +0.6050 | [-0.0351, +1.5412] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +1.3090 / -0.0854 | +0.6118 | [-0.9567, +2.4246] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.1194 / +0.0573 | +0.0883 | [+0.0027, +0.1858] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.6093 / +0.2721 | +0.4407 | [+0.1293, +0.7675] | 0.5140 (0.5140) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10RLc (0,1) | 0.0383 [0.0190, 0.0628] | 0.0016 [0.0000, 0.0051] | 0.2181 [0.1459, 0.2897] | 0.2229 [-0.0327, 0.5472] | 2.0938 [1.2319, 3.0706] | 6.7474 [4.5816, 8.9276] | 0.7885 [0.5579, 1.0945] | 3.2710 [2.6177, 4.1001] | 8.0958 [6.4712, 9.8010] | 1.0876 [0.7460, 1.4643] | 2.1535 [1.4770, 2.9189] |
| I10RLcB10 (0,1) | 0.0498 [0.0283, 0.0748] | 0.0016 [0.0000, 0.0040] | 0.2204 [0.1455, 0.2981] | 0.2744 [-0.0071, 0.6299] | 2.3681 [1.3325, 3.6750] | 7.2231 [4.5463, 10.2612] | 0.8295 [0.5734, 1.1592] | 3.5154 [2.6852, 4.6037] | 8.6216 [6.5145, 11.1694] | 1.0952 [0.7520, 1.4698] | 2.2247 [1.5620, 2.9776] |
| I10RLcB8 (0,1) | 0.0480 [0.0276, 0.0734] | 0.0000 [0.0000, 0.0000] | 0.2227 [0.1466, 0.3006] | 0.2449 [-0.0187, 0.5829] | 2.5681 [1.3924, 4.0111] | 7.8584 [5.0352, 10.9713] | 0.8045 [0.5655, 1.1170] | 3.7106 [2.7508, 4.9920] | 9.2474 [6.9986, 11.9273] | 1.0891 [0.7474, 1.4649] | 2.2109 [1.5334, 2.9740] |
| I10RQ50 (0,1) | 0.0801 [0.0480, 0.1174] | 0.0265 [0.0102, 0.0465] | 0.2593 [0.1862, 0.3304] | 0.2697 [0.0335, 0.5486] | 2.4645 [1.6073, 3.4225] | 7.6520 [5.6761, 9.6557] | 0.8241 [0.5966, 1.0699] | 3.7859 [3.0818, 4.6228] | 9.4150 [7.8273, 11.0990] | 1.1816 [0.8261, 1.5783] | 2.5241 [1.8524, 3.2957] |
| I10RQ50B8 (0,1) | 0.0810 [0.0517, 0.1141] | 0.0265 [0.0102, 0.0465] | 0.2609 [0.1880, 0.3297] | 0.2933 [0.0479, 0.5794] | 2.9180 [1.8805, 4.1289] | 8.1495 [6.0906, 10.2668] | 0.8433 [0.6066, 1.0992] | 4.3156 [3.3460, 5.4621] | 9.8592 [8.1103, 11.8058] | 1.1774 [0.8275, 1.5682] | 2.6516 [2.0074, 3.3486] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
