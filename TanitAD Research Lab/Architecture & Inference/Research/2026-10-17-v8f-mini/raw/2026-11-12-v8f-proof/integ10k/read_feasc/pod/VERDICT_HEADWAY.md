# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: /workspace/v8proof/podread/data/join_eval141.npz. Alignment reference (ego speed): /workspace/v8proof/podread/data/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 22, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.03202846975088969, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0E5:I10R0M5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcE5:I10RLcM5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50E5:I10RQ50M5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0E5:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcE5:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50E5:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0E10:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcE5:I10R0E5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50E5:I10R0E5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0E5L:I10R0E5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0E5L:I10R0M5", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10R0E5 minus I10R0M5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcE5 minus I10RLcM5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50E5 minus I10RQ50M5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0E5 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcE5 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50E5 minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0E10 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcE5 minus I10R0E5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50E5 minus I10R0E5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0E5L minus I10R0E5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0E5L minus I10R0M5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10R0E5 minus I10R0M5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.2527 / -0.3221 | -0.2874 | [-0.3697, -0.2154] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.3879 / -0.4751 | -0.4315 | [-0.5164, -0.3522] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | -0.0187 / -0.0187 | -0.0187 | [-0.0373, -0.0022] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2366 / +0.3555 | +0.2960 | [+0.2022, +0.4162] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +6.8711 / +12.1005 | +9.4858 | [+5.6426, +13.4560] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +21.6595 / +42.4247 | +32.0421 | [+18.3127, +45.2460] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | -0.2469 / -0.3764 | -0.3116 | [-0.4273, -0.2168] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -9.8553 / -14.8182 | -12.3368 | [-15.9643, -8.6495] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -33.0298 / -43.9071 | -38.4685 | [-46.4755, -30.6052] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.2440 / -0.3144 | -0.2792 | [-0.3974, -0.1761] | 0.0330 (0.0330) | BETTER |
| time-gap error over the scored steps (s) | -0.5162 / -0.6364 | -0.5763 | [-0.7521, -0.4347] | 0.5140 (0.5140) | BETTER |

## I10RLcE5 minus I10RLcM5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.3292 / -0.3683 | -0.3488 | [-0.4011, -0.2921] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0171 / -0.0202 | -0.0187 | [-0.0273, -0.0108] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0312 / -0.0576 | -0.0444 | [-0.0751, -0.0152] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2273 / +0.2883 | +0.2578 | [+0.1769, +0.3410] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +0.3318 / +1.6117 | +0.9718 | [-0.0582, +2.0718] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | -1.2638 / -1.2629 | -1.2634 | [-2.7522, +0.2245] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | -0.1636 / -0.2335 | -0.1985 | [-0.2935, -0.1072] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -1.8568 / -2.2308 | -2.0438 | [-2.7125, -1.5253] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -2.5496 / -3.0553 | -2.8024 | [-4.0275, -1.6249] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.2388 / -0.2852 | -0.2620 | [-0.3787, -0.1627] | 0.0330 (0.0330) | BETTER |
| time-gap error over the scored steps (s) | -0.1211 / -0.1702 | -0.1456 | [-0.3095, -0.0074] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50E5 minus I10RQ50M5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.3167 / -0.3541 | -0.3354 | [-0.3966, -0.2775] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0607 / -0.0717 | -0.0662 | [-0.0989, -0.0384] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | -0.0171 / -0.0249 | -0.0210 | [-0.0430, -0.0022] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1063 / +0.2208 | +0.1635 | [+0.0777, +0.2667] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | -0.1984 / +1.8264 | +0.8140 | [-0.6565, +2.3837] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +1.5083 / +2.5387 | +2.0235 | [-0.2097, +4.5418] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | -0.2251 / -0.3180 | -0.2715 | [-0.3611, -0.1900] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -2.4169 / -2.1377 | -2.2773 | [-2.9921, -1.5290] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -6.2912 / -4.9000 | -5.5956 | [-8.1216, -3.3007] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.2834 / -0.3717 | -0.3276 | [-0.4756, -0.1956] | 0.0330 (0.0330) | BETTER |
| time-gap error over the scored steps (s) | +0.0804 / -0.0503 | +0.0150 | [-0.1977, +0.2163] | 0.5140 (0.5140) | OPEN |

## I10R0E5 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.4520 / +0.4609 | +0.4564 | [+0.3786, +0.5304] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.1854 / +0.2150 | +0.2002 | [+0.1342, +0.2644] | 0.0343 (0.0343) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0436 / -0.1433 | -0.0935 | [-0.1791, -0.0182] | 0.0903 (0.0903) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.5868 / -0.5814 | -0.5841 | [-0.6663, -0.4957] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -6.3345 / -7.1316 | -6.7331 | [-7.6224, -5.7992] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -17.7915 / -19.9953 | -18.8934 | [-21.8516, -15.6217] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.5858 / +0.5810 | +0.5834 | [+0.5250, +0.6417] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +5.0480 / +4.3095 | +4.6787 | [+3.9272, +5.4370] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +14.0344 / +10.2295 | +12.1319 | [+8.9747, +15.1798] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.5581 / +0.4972 | +0.5276 | [+0.0836, +1.0830] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.3288 / -0.1534 | +0.0877 | [-0.4594, +0.5440] | 0.5140 (0.5140) | OPEN |

## I10RLcE5 minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0943 / +0.1050 | +0.0996 | [+0.0643, +0.1403] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0000 / -0.0031 | -0.0016 | [-0.0051, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0156 / -0.0888 | -0.0366 | [-0.1237, +0.0401] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.2617 / -0.2893 | -0.2755 | [-0.4414, -0.0862] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -2.7889 / -3.6488 | -3.2188 | [-4.3465, -2.1273] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -3.5305 / -7.5303 | -5.5304 | [-9.0815, -2.1497] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.6727 / +0.6381 | +0.6554 | [+0.5386, +0.7949] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +3.4189 / +2.7853 | +3.1021 | [+2.3729, +3.8432] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +6.2313 / +3.4826 | +4.8569 | [+2.4137, +7.2674] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.4584 / +0.4022 | +0.4303 | [-0.0089, +0.9928] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.2506 / -0.3003 | -0.0249 | [-0.5977, +0.4635] | 0.5140 (0.5140) | OPEN |

## I10RQ50E5 minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.1406 / +0.1797 | +0.1601 | [+0.1087, +0.2145] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0156 / +0.0327 | +0.0241 | [+0.0029, +0.0457] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0358 / -0.1012 | -0.0327 | [-0.1321, +0.0568] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0867 / -0.3220 | -0.2044 | [-0.4345, +0.0368] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -1.4111 / -3.6519 | -2.5315 | [-4.2152, -0.7089] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -0.3207 / -6.1207 | -3.2207 | [-7.4193, +1.4465] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.7786 / +0.6498 | +0.7142 | [+0.5773, +0.8617] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +3.9754 / +2.5024 | +3.2389 | [+2.0911, +4.4701] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +7.6545 / +2.5040 | +5.0793 | [+1.4790, +8.9877] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.8541 / +0.5579 | +0.7060 | [+0.1943, +1.3262] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.2818 / -0.3051 | -0.0116 | [-0.6103, +0.5125] | 0.5140 (0.5140) | OPEN |

## I10R0E10 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0427 / -0.0445 | -0.0436 | [-0.0680, -0.0226] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0202 / -0.0125 | -0.0164 | [-0.0287, -0.0070] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0218 / +0.0343 | +0.0280 | [+0.0142, +0.0437] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0444 / +0.0600 | +0.0522 | [+0.0333, +0.0784] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +0.2511 / +0.2304 | +0.2408 | [+0.1765, +0.3137] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.9363 / +0.9995 | +0.9679 | [+0.5439, +1.4059] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0161 / -0.0189 | -0.0175 | [-0.0298, -0.0087] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | +0.0368 / +0.0529 | +0.0449 | [-0.0263, +0.1187] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +0.1803 / +0.3997 | +0.2900 | [+0.0302, +0.5607] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0056 / +0.0081 | +0.0068 | [+0.0035, +0.0117] | 0.0330 (0.0330) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0795 / +0.4591 | +0.2693 | [-0.0050, +0.6015] | 0.5140 (0.5140) | OPEN |

## I10RLcE5 minus I10R0E5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.4573 / -0.4395 | -0.4484 | [-0.5097, -0.3845] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.2617 / -0.2570 | -0.2593 | [-0.3263, -0.1913] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1480 / +0.1215 | +0.1347 | [+0.0727, +0.1972] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7714 / +0.7387 | +0.7551 | [+0.3486, +1.2406] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +5.4553 / +4.8642 | +5.1598 | [+3.5145, +6.8830] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +18.9983 / +15.6344 | +17.3164 | [+12.8848, +21.5951] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.4020 / +0.3732 | +0.3876 | [+0.0454, +0.8013] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | -0.7699 / -0.8254 | -0.7977 | [-1.8935, +0.4071] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -6.3482 / -5.7658 | -6.0570 | [-8.9763, -3.1383] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.0571 / -0.0493 | -0.0532 | [-0.1451, +0.0314] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.0253 / -0.0272 | -0.0009 | [-0.1305, +0.1307] | 0.5140 (0.5140) | OPEN |

## I10RQ50E5 minus I10R0E5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.3701 / -0.3221 | -0.3461 | [-0.4210, -0.2752] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.2150 / -0.2025 | -0.2087 | [-0.2676, -0.1502] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.2118 / +0.1480 | +0.1799 | [+0.1056, +0.2530] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.1021 / +0.6439 | +0.8730 | [+0.4051, +1.3833] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +7.3366 / +5.0991 | +6.2178 | [+4.0874, +8.4744] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +23.5430 / +17.5184 | +20.5307 | [+15.2322, +26.2192] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.6334 / +0.3307 | +0.4820 | [+0.1379, +0.8737] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.3595 / -0.6514 | -0.1459 | [-1.4324, +1.3274] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -3.3251 / -5.7058 | -4.5155 | [-7.8365, -0.8057] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.4744 / +0.1586 | +0.3165 | [+0.0586, +0.7395] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.4829 / +0.2830 | +0.3830 | [+0.1554, +0.6728] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10R0E5L minus I10R0E5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.1815 / -0.2153 | -0.1984 | [-0.2494, -0.1509] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0966 / -0.0981 | -0.0974 | [-0.1364, -0.0621] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | -0.0140 / -0.0156 | -0.0148 | [-0.0331, -0.0002] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.3447 / +0.3617 | +0.3532 | [+0.2961, +0.4132] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.2976 / +2.2299 | +2.2638 | [+1.9683, +2.5902] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +6.0419 / +5.2334 | +5.6377 | [+4.6124, +6.7502] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | -0.2880 / -0.2949 | -0.2914 | [-0.3419, -0.2441] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -2.4006 / -2.1127 | -2.2566 | [-2.6136, -1.9357] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -6.6759 / -5.3613 | -6.0186 | [-7.1790, -4.9550] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.2899 / -0.2471 | -0.2685 | [-0.3988, -0.1725] | 0.0330 (0.0330) | BETTER |
| time-gap error over the scored steps (s) | -0.2428 / -0.2450 | -0.2439 | [-0.3310, -0.1624] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10R0E5L minus I10R0M5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.4342 / -0.5374 | -0.4858 | [-0.5881, -0.3910] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.4844 / -0.5732 | -0.5288 | [-0.6093, -0.4474] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | -0.0327 / -0.0343 | -0.0335 | [-0.0551, -0.0143] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.5813 / +0.7171 | +0.6492 | [+0.5115, +0.8151] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +9.1688 / +14.3304 | +11.7496 | [+7.8139, +15.8041] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +27.7014 / +47.6581 | +37.6797 | [+24.0273, +50.7605] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | -0.5348 / -0.6713 | -0.6031 | [-0.7421, -0.4790] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -12.2559 / -16.9309 | -14.5934 | [-18.1407, -10.9806] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -39.7057 / -49.2685 | -44.4871 | [-52.0317, -37.1203] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.5339 / -0.5616 | -0.5478 | [-0.7574, -0.3627] | 0.0330 (0.0330) | BETTER |
| time-gap error over the scored steps (s) | -0.7590 / -0.8814 | -0.8202 | [-1.0428, -0.6288] | 0.5140 (0.5140) | BETTER |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10R0E10 (0,1) | 0.0863 [0.0460, 0.1315] | 0.0428 [0.0185, 0.0740] | 0.1682 [0.0878, 0.2535] | -0.1713 [-0.2304, -0.1092] | 0.6890 [-0.0968, 1.4984] | 3.7620 [1.1637, 6.4667] | 0.4554 [0.4151, 0.5001] | 2.5369 [2.2161, 2.9066] | 7.1678 [5.7232, 8.7185] | 1.0503 [0.7100, 1.4289] | 2.3111 [1.5041, 3.2560] |
| I10R0E5 (0,1) | 0.5863 [0.5080, 0.6549] | 0.2593 [0.1913, 0.3263] | 0.0467 [0.0238, 0.0776] | -0.8076 [-0.9135, -0.6925] | -6.2848 [-7.0702, -5.4625] | -16.0993 [-18.7639, -13.2846] | 1.0563 [0.9822, 1.1325] | 7.1708 [6.5711, 7.8093] | 19.0097 [16.8729, 21.2217] | 1.5711 [1.0318, 2.2311] | 2.1296 [1.4322, 3.0257] |
| I10R0E5L (0,1) | 0.3879 [0.2966, 0.4736] | 0.1620 [0.1060, 0.2214] | 0.0319 [0.0155, 0.0500] | -0.4545 [-0.5627, -0.3379] | -4.0210 [-4.6688, -3.3055] | -10.4616 [-12.4381, -8.2674] | 0.7648 [0.7046, 0.8258] | 4.9142 [4.4530, 5.4077] | 12.9911 [11.5633, 14.5342] | 1.3026 [0.8158, 1.9154] | 1.8857 [1.2444, 2.7526] |
| I10R0M5 (0,1) | 0.8737 [0.8176, 0.9191] | 0.6908 [0.6102, 0.7630] | 0.0654 [0.0378, 0.0987] | -1.1037 [-1.2468, -0.9695] | -15.7706 [-19.8054, -11.7181] | -48.1414 [-61.7632, -33.8481] | 1.3679 [1.2375, 1.5208] | 19.5076 [16.1343, 22.9679] | 57.4782 [50.7509, 64.8401] | 1.8503 [1.2289, 2.6007] | 2.7059 [1.8928, 3.7647] |
| I10RLc (0,1) | 0.0383 [0.0190, 0.0628] | 0.0016 [0.0000, 0.0051] | 0.2181 [0.1459, 0.2897] | 0.2229 [-0.0327, 0.5472] | 2.0938 [1.2319, 3.0706] | 6.7474 [4.5816, 8.9276] | 0.7885 [0.5579, 1.0945] | 3.2710 [2.6177, 4.1001] | 8.0958 [6.4712, 9.8010] | 1.0876 [0.7460, 1.4643] | 2.1535 [1.4770, 2.9189] |
| I10RLcE5 (0,1) | 0.1379 [0.0999, 0.1757] | 0.0000 [0.0000, 0.0000] | 0.1815 [0.1235, 0.2428] | -0.0526 [-0.4630, 0.4413] | -1.1250 [-2.6280, 0.5545] | 1.2170 [-2.3316, 5.0531] | 1.4439 [1.1124, 1.8698] | 6.3731 [5.3979, 7.5841] | 12.9527 [10.7397, 15.5616] | 1.5179 [0.9991, 2.1678] | 2.1286 [1.4532, 3.0336] |
| I10RLcM5 (0,1) | 0.4867 [0.4184, 0.5555] | 0.0187 [0.0108, 0.0273] | 0.2259 [0.1722, 0.2837] | -0.3104 [-0.7041, 0.1461] | -2.0968 [-4.2249, 0.0735] | 2.4804 [-1.7053, 6.8513] | 1.6424 [1.3486, 2.0128] | 8.4169 [7.2754, 9.7230] | 15.7552 [13.1288, 18.7193] | 1.7800 [1.1768, 2.5370] | 2.2743 [1.5065, 3.3193] |
| I10RQ50 (0,1) | 0.0801 [0.0480, 0.1174] | 0.0265 [0.0102, 0.0465] | 0.2593 [0.1862, 0.3304] | 0.2697 [0.0335, 0.5486] | 2.4645 [1.6073, 3.4225] | 7.6520 [5.6761, 9.6557] | 0.8241 [0.5966, 1.0699] | 3.7859 [3.0818, 4.6228] | 9.4150 [7.8273, 11.0990] | 1.1816 [0.8261, 1.5783] | 2.5241 [1.8524, 3.2957] |
| I10RQ50E5 (0,1) | 0.2402 [0.1786, 0.2996] | 0.0506 [0.0271, 0.0755] | 0.2266 [0.1566, 0.2986] | 0.0653 [-0.3947, 0.5569] | -0.0670 [-2.0326, 2.0724] | 4.4313 [0.0583, 9.3768] | 1.5383 [1.1917, 1.9141] | 7.0248 [5.6762, 8.5511] | 14.4943 [10.9267, 18.6457] | 1.8876 [1.2602, 2.6299] | 2.5125 [1.7861, 3.3841] |
| I10RQ50M5 (0,1) | 0.5756 [0.5018, 0.6486] | 0.1168 [0.0813, 0.1546] | 0.2477 [0.1845, 0.3185] | -0.0982 [-0.6460, 0.4743] | -0.8809 [-4.0130, 2.4903] | 2.4079 [-3.5213, 9.0091] | 1.8098 [1.4772, 2.1781] | 9.3022 [7.6778, 11.2168] | 20.0898 [15.1278, 25.9184] | 2.2152 [1.5551, 3.0144] | 2.4975 [1.7350, 3.5154] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
