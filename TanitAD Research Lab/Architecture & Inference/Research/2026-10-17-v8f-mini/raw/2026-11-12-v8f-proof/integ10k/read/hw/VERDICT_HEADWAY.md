# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 12, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.03202846975088969, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RR:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RR:I10RLb", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLb:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RR0:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RRid:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RRx:I10R0", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10RR minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RR minus I10RLb over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLb minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RR0 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RRid minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RRx minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10RR minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0623 / -0.0623 | -0.0623 | [-0.1010, -0.0271] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0483 / -0.0249 | -0.0366 | [-0.0621, -0.0154] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.2212 / +0.1760 | +0.1986 | [+0.1350, +0.2639] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.0474 / +0.8906 | +0.9690 | [+0.6343, +1.3430] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +4.3956 / +3.0563 | +3.7259 | [+2.2628, +5.4301] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +9.6311 / +6.5692 | +8.1001 | [+4.8657, +11.7081] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.8042 / +0.6633 | +0.7338 | [+0.4323, +1.0681] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +3.1437 / +2.3726 | +2.7581 | [+1.6742, +3.9674] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +6.2526 / +4.5268 | +5.3897 | [+3.0308, +8.1049] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.2225 / +0.1916 | +0.2070 | [+0.1240, +0.3078] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.9573 / +0.8565 | +0.9069 | [+0.6102, +1.2294] | 0.5140 (0.5140) | WORSE |

## I10RR minus I10RLb

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0053 / +0.0000 | -0.0027 | [-0.0138, +0.0072] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0047 / +0.0016 | -0.0016 | [-0.0103, +0.0050] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.1184 / +0.0966 | +0.1075 | [+0.0665, +0.1551] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.5997 / +0.4385 | +0.5191 | [+0.3166, +0.7762] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.2472 / +1.3121 | +1.7796 | [+0.8947, +3.0539] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +5.0171 / +2.9660 | +3.9916 | [+2.0027, +6.6453] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.4594 / +0.3182 | +0.3888 | [+0.2088, +0.6152] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +1.7983 / +1.2054 | +1.5018 | [+0.7717, +2.4935] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +4.4667 / +2.8089 | +3.6378 | [+1.8356, +5.9944] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.1576 / +0.1240 | +0.1408 | [+0.0733, +0.2280] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.6860 / +0.5149 | +0.6005 | [+0.3220, +0.8985] | 0.5140 (0.5140) | WORSE |

## I10RLb minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0569 / -0.0623 | -0.0596 | [-0.0933, -0.0288] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0436 / -0.0265 | -0.0350 | [-0.0578, -0.0171] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1028 / +0.0794 | +0.0911 | [+0.0455, +0.1418] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4477 / +0.4520 | +0.4499 | [+0.1834, +0.7692] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.1484 / +1.7442 | +1.9463 | [+0.8825, +3.2618] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +4.6139 / +3.6032 | +4.1086 | [+2.1176, +6.4346] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.3449 / +0.3451 | +0.3450 | [+0.1170, +0.6319] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +1.3454 / +1.1672 | +1.2563 | [+0.4256, +2.2917] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +1.7860 / +1.7179 | +1.7519 | [+0.5367, +3.2583] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.0649 / +0.0675 | +0.0662 | [+0.0262, +0.1131] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.2714 / +0.3416 | +0.3065 | [+0.1541, +0.4923] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RR0 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.1228 / -0.0925 | -0.1077 | [-0.1598, -0.0587] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0717 / -0.0389 | -0.0553 | [-0.0925, -0.0245] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.5452 / +0.4891 | +0.5171 | [+0.4415, +0.5853] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +5.0910 / +5.5462 | +5.3186 | [+4.6118, +6.0630] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +14.4373 / +14.1027 | +14.2700 | [+11.5979, +16.9885] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +24.1256 / +22.2910 | +23.2083 | [+18.3238, +28.2654] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +4.5090 / +4.9788 | +4.7439 | [+4.0562, +5.4848] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +12.8937 / +13.4388 | +13.1662 | [+10.9110, +15.6460] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +20.1144 / +20.0790 | +20.0967 | [+15.6529, +24.7514] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +1.8681 / +2.3248 | +2.0964 | [+1.6728, +2.5766] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +6.7545 / +7.0672 | +6.9109 | [+5.8460, +8.0361] | 0.5140 (0.5140) | WORSE |

## I10RRid minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.5140 (0.5140) | OPEN |

## I10RRx minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0552 / -0.0587 | -0.0569 | [-0.0960, -0.0244] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0483 / -0.0218 | -0.0350 | [-0.0615, -0.0138] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.2118 / +0.1869 | +0.1994 | [+0.1382, +0.2599] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.8996 / +1.0368 | +0.9682 | [+0.6169, +1.3603] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +3.5685 / +3.8235 | +3.6960 | [+2.4382, +5.0720] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +8.1419 / +7.5332 | +7.8376 | [+5.1887, +10.7022] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.6627 / +0.8098 | +0.7363 | [+0.4262, +1.0923] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +2.4841 / +2.9977 | +2.7409 | [+1.7394, +3.8880] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +4.8487 / +5.4004 | +5.1245 | [+3.0944, +7.5044] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.2098 / +0.2136 | +0.2117 | [+0.1241, +0.3195] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.7502 / +0.9725 | +0.8614 | [+0.5747, +1.2030] | 0.5140 (0.5140) | WORSE |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10RLb (0,1) | 0.0703 [0.0382, 0.1113] | 0.0241 [0.0065, 0.0484] | 0.2313 [0.1606, 0.3039] | 0.2263 [-0.0383, 0.5548] | 2.3946 [1.2957, 3.6479] | 6.9027 [4.4318, 9.5247] | 0.8178 [0.5897, 1.1089] | 3.7483 [2.8924, 4.8297] | 8.6297 [6.8435, 10.6964] | 1.1097 [0.7592, 1.4936] | 2.3483 [1.6463, 3.1694] |
| I10RR (0,1) | 0.0676 [0.0393, 0.1055] | 0.0226 [0.0069, 0.0418] | 0.3388 [0.2669, 0.4094] | 0.7454 [0.4244, 1.1084] | 4.1742 [3.0928, 5.4141] | 10.8942 [8.3762, 13.4805] | 1.2067 [0.9032, 1.5434] | 5.2502 [4.1900, 6.4470] | 12.2675 [10.1056, 14.6726] | 1.2505 [0.8831, 1.6481] | 2.9488 [2.2494, 3.7525] |
| I10RR0 (0,1) | 0.0222 [0.0102, 0.0358] | 0.0039 [0.0000, 0.0092] | 0.6573 [0.5916, 0.7158] | 5.0951 [4.3844, 5.8503] | 14.7183 [12.3033, 17.2723] | 26.0024 [21.6001, 30.9471] | 5.2168 [4.5316, 5.9519] | 15.6583 [13.3573, 18.1467] | 26.9745 [22.7206, 31.7590] | 3.1399 [2.4972, 3.9219] | 8.9527 [7.7639, 10.2767] |
| I10RRid (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10RRx (0,1) | 0.0730 [0.0425, 0.1099] | 0.0241 [0.0088, 0.0427] | 0.3396 [0.2622, 0.4148] | 0.7446 [0.3969, 1.1386] | 4.1443 [2.7557, 5.6258] | 10.6317 [7.6413, 14.0691] | 1.2092 [0.9062, 1.5701] | 5.2329 [4.1375, 6.4936] | 12.0024 [9.5654, 14.9552] | 1.2552 [0.8880, 1.6598] | 2.9032 [2.1051, 3.8087] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
