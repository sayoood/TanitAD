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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0V5:I10R0M5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0V5:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0V10:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0V5:I10R0W5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0V5:I10R0Z5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0V10:I10R0W10", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10R0V5 minus I10R0M5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0V5 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0V10 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0V5 minus I10R0W5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0V5 minus I10R0Z5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0V10 minus I10R0W10 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10R0V5 minus I10R0M5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.4733 / -0.5231 | -0.4982 | [-0.5715, -0.4236] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.5031 / -0.5981 | -0.5506 | [-0.6298, -0.4678] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | -0.0047 / +0.0109 | +0.0031 | [-0.0132, +0.0235] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7479 / +0.8554 | +0.8017 | [+0.7010, +0.9142] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +10.6438 / +15.7823 | +13.2130 | [+9.2422, +17.0545] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +31.6784 / +51.4745 | +41.5765 | [+27.5433, +54.3391] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | -0.6628 / -0.7610 | -0.7119 | [-0.8142, -0.6193] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -13.5325 / -17.7643 | -15.6484 | [-18.8899, -12.4363] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -42.3953 / -49.7640 | -46.0797 | [-53.1623, -40.0776] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.4761 / -0.4271 | -0.4516 | [-0.6075, -0.2998] | 0.0330 (0.0330) | BETTER |
| time-gap error over the scored steps (s) | -0.7835 / -0.9363 | -0.8599 | [-1.0284, -0.7132] | 0.5140 (0.5140) | BETTER |

## I10R0V5 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.2313 / +0.2598 | +0.2456 | [+0.1840, +0.3066] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0701 / +0.0919 | +0.0810 | [+0.0383, +0.1237] | 0.0343 (0.0343) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0296 / -0.1137 | -0.0717 | [-0.1503, -0.0024] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0754 / -0.0814 | -0.0784 | [-0.1332, -0.0186] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -2.5619 / -3.4498 | -3.0058 | [-3.8467, -2.2470] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -7.7726 / -10.9454 | -9.3590 | [-12.3946, -6.3809] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.1699 / +0.1964 | +0.1832 | [+0.1333, +0.2373] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +1.3708 / +1.3634 | +1.3671 | [+0.9890, +1.7635] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +4.6688 / +4.3726 | +4.5207 | [+2.9726, +6.0994] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.3259 / +0.3845 | +0.3552 | [-0.1273, +0.9972] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.0615 / -0.4533 | -0.1959 | [-0.7630, +0.2865] | 0.5140 (0.5140) | OPEN |

## I10R0V10 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0356 / -0.0285 | -0.0320 | [-0.0736, +0.0057] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0234 / +0.0031 | -0.0101 | [-0.0390, +0.0146] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0234 / -0.0327 | -0.0047 | [-0.0529, +0.0419] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.3897 / +0.0645 | +0.2271 | [-0.0830, +0.5308] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +0.9658 / -0.8673 | +0.0493 | [-1.5151, +1.4765] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.0732 / -0.1476 | -0.1104 | [-0.2822, +0.0550] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -0.1407 / -0.9337 | -0.5372 | [-1.4003, +0.2225] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.2288 / +0.2091 | +0.2189 | [+0.0147, +0.4285] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10R0V5 minus I10R0W5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0516 / -0.0907 | -0.0712 | [-0.1179, -0.0378] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0171 / -0.0327 | -0.0249 | [-0.0483, -0.0051] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0000 / +0.0016 | +0.0008 | [-0.0131, +0.0127] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0416 / +0.0399 | +0.0407 | [+0.0301, +0.0515] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.6945 / +0.4526 | +0.5736 | [+0.3599, +0.8030] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.9767 / +1.2944 | +1.1356 | [-0.1170, +2.2133] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | -0.0102 / -0.0159 | -0.0131 | [-0.0226, -0.0029] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -0.5030 / -0.7147 | -0.6089 | [-0.8257, -0.4197] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -2.1863 / -2.7258 | -2.4560 | [-3.4425, -1.5110] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.0184 / +0.0122 | +0.0153 | [-0.0362, +0.0781] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.0354 / -0.0958 | -0.0656 | [-0.1350, -0.0178] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10R0V5 minus I10R0Z5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.1566 / +0.1619 | +0.1593 | [+0.1137, +0.2042] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0249 / +0.0202 | +0.0226 | [+0.0053, +0.0436] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0171 / +0.0109 | +0.0140 | [-0.0043, +0.0323] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0585 / -0.0623 | -0.0604 | [-0.0998, -0.0240] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -0.8364 / -0.8248 | -0.8306 | [-1.1393, -0.5337] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -2.4791 / -1.9980 | -2.2385 | [-3.3270, -1.2301] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0143 / -0.0292 | -0.0218 | [-0.0468, +0.0052] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.4002 / +0.4631 | +0.4316 | [+0.1741, +0.6850] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +2.6440 / +3.6145 | +3.1293 | [+1.9268, +4.4272] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.1020 / +0.1011 | +0.1016 | [-0.1577, +0.4430] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.2247 / -0.4204 | -0.3225 | [-0.6871, -0.0175] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10R0V10 minus I10R0W10

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0142 / -0.0178 | -0.0160 | [-0.0371, +0.0046] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0031 / -0.0093 | -0.0031 | [-0.0228, +0.0123] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0343 / -0.0701 | -0.0522 | [-0.0841, -0.0218] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.5515 / -0.5853 | -0.5684 | [-0.7106, -0.4300] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -1.1291 / -1.0343 | -1.0817 | [-1.7419, -0.3531] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.3801 / -0.3658 | -0.3730 | [-0.5018, -0.2554] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -0.7254 / -2.0219 | -1.3736 | [-2.3440, -0.4972] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.3171 / -0.5235 | -0.4203 | [-0.6923, -0.2001] | 0.5140 (0.5140) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10R0M5 (0,1) | 0.8737 [0.8176, 0.9191] | 0.6908 [0.6102, 0.7630] | 0.0654 [0.0378, 0.0987] | -1.1037 [-1.2468, -0.9695] | -15.7706 [-19.8054, -11.7181] | -48.1414 [-61.7632, -33.8481] | 1.3679 [1.2375, 1.5208] | 19.5076 [16.1343, 22.9679] | 57.4782 [50.7509, 64.8401] | 1.8503 [1.2289, 2.6007] | 2.7059 [1.8928, 3.7647] |
| I10R0V10 (0,1) | 0.0979 [0.0626, 0.1399] | 0.0491 [0.0249, 0.0746] | 0.1355 [0.0944, 0.1812] | -0.2236 [-0.2757, -0.1682] | 0.6753 [0.0650, 1.2894] | 2.8433 [1.2605, 4.4989] | 0.4729 [0.4299, 0.5219] | 2.3816 [2.0806, 2.7230] | 6.3406 [5.4119, 7.3573] | 1.0435 [0.7059, 1.4212] | 2.2608 [1.5270, 3.1108] |
| I10R0V5 (0,1) | 0.3754 [0.2975, 0.4505] | 0.1402 [0.0935, 0.1904] | 0.0685 [0.0399, 0.1034] | -0.3020 [-0.3789, -0.2119] | -2.5576 [-3.0131, -2.0404] | -6.5649 [-8.0924, -4.7935] | 0.6560 [0.5926, 0.7286] | 3.8592 [3.4723, 4.3291] | 11.3985 [9.8645, 13.2040] | 1.3987 [0.7923, 2.1812] | 1.8460 [1.1332, 2.7911] |
| I10R0W10 (0,1) | 0.1139 [0.0736, 0.1574] | 0.0522 [0.0276, 0.0784] | 0.1877 [0.1297, 0.2538] | -0.2236 [-0.2757, -0.1682] | 1.2437 [0.5671, 1.9071] | 3.9250 [2.3093, 5.7152] | 0.4729 [0.4299, 0.5219] | 2.7546 [2.4040, 3.1458] | 7.7143 [6.1726, 9.3531] | 1.0435 [0.7059, 1.4212] | 2.6811 [1.8659, 3.6560] |
| I10R0W5 (0,1) | 0.4466 [0.3615, 0.5284] | 0.1651 [0.1142, 0.2192] | 0.0678 [0.0374, 0.1030] | -0.3427 [-0.4178, -0.2570] | -3.1311 [-3.6690, -2.5103] | -7.7005 [-9.6377, -5.3580] | 0.6691 [0.6049, 0.7438] | 4.4680 [3.9562, 5.1133] | 13.8545 [11.8407, 16.1871] | 1.3834 [0.8009, 2.1162] | 1.9116 [1.1864, 2.8892] |
| I10R0Z5 (0,1) | 0.2162 [0.1489, 0.2851] | 0.1176 [0.0714, 0.1683] | 0.0545 [0.0323, 0.0815] | -0.2416 [-0.3451, -0.1223] | -1.7270 [-2.3181, -1.0712] | -4.3264 [-5.9363, -2.5736] | 0.6778 [0.6023, 0.7674] | 3.4275 [2.9799, 3.9379] | 8.2692 [7.0430, 9.5783] | 1.2971 [0.8284, 1.8486] | 2.1685 [1.3787, 3.1127] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
