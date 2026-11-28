# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: F:/TanitAD-scratch/strat1/read/inputs/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 14, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.0752218477307355, "A_hi": 0.19026440814805612, "spread_max": 0.003558718861209967, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0C10XHb20:I10B0_R0C10XHb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0C10XHb20:I10RH_R0KHb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0G50Hb20:I10RH_R0G0Hb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0C10Hb20:I10RH_R0G50Hb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0C10XHb20:I10B0_R0KHb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0G50Hb20:I10B0_R0G0Hb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0C10Hb20:I10B0_R0G50Hb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0C10XHb20:I10B0_R0KHb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0Hb20:I10B0_R0Hb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0C10Hb20:I10B0_R0C10Hb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0G50Hb20:I10B0_R0G50Hb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0G0Hb20:I10B0_R0G0Hb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0KHb20:I10B0_R0KHb20", "seeds": [2]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10UEA_R0C10XHb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10RH_R0C10XHb20 minus I10B0_R0C10XHb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0C10XHb20 minus I10RH_R0KHb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0G50Hb20 minus I10RH_R0G0Hb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0C10Hb20 minus I10RH_R0G50Hb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0C10XHb20 minus I10B0_R0KHb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0G50Hb20 minus I10B0_R0G0Hb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0C10Hb20 minus I10B0_R0G50Hb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0C10XHb20 minus I10B0_R0KHb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0Hb20 minus I10B0_R0Hb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0C10Hb20 minus I10B0_R0C10Hb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0G50Hb20 minus I10B0_R0G50Hb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0G0Hb20 minus I10B0_R0G0Hb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0KHb20 minus I10B0_R0KHb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10RH R0C10XHb20 minus I10B0 R0C10XHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0125 | +0.0125 | [+0.0000, +0.0267] | 0.0036 (0.0036) | OPEN |
| C contact rate | +0.0078 | +0.0078 | [-0.0020, +0.0233] | 0.0062 (0.0062) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0343 | -0.0343 | [-0.0639, -0.0042] | 0.0561 (0.0561) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0289 | -0.0289 | [-0.0625, -0.0014] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.2610 | -0.2610 | [-0.4340, -0.1066] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -1.3479 | -1.3479 | [-2.0525, -0.7786] | 1.8616 (1.8616) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0232 | +0.0232 | [+0.0061, +0.0443] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.0519 | +0.0519 | [-0.0704, +0.1950] | 0.4100 (0.4100) | OPEN |
| gap error 6.0 s (m) | -0.5772 | -0.5772 | [-1.1459, +0.0022] | 1.5504 (1.5504) | OPEN |
| time-gap error 2.0 s (s) | +0.0869 | +0.0869 | [+0.0084, +0.1970] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | -0.6156 | -0.6156 | [-1.1200, -0.1122] | 0.2075 (0.2075) | BETTER |

## I10RH R0C10XHb20 minus I10RH R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0391 | -0.0391 | [-0.0703, -0.0123] | 0.0036 (0.0036) | BETTER |
| C contact rate | -0.0171 | -0.0171 | [-0.0345, -0.0041] | 0.0062 (0.0062) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0794 | +0.0794 | [+0.0406, +0.1266] | 0.0561 (0.0561) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.9256 | +0.9256 | [+0.7343, +1.1584] | 0.6885 (0.6885) | WORSE |
| gap bias 6.0 s (m) | +3.4542 | +3.4542 | [+2.7870, +4.2103] | 1.8616 (1.8616) | WORSE |
| gap error 2.0 s (m) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.2962 | +0.2962 | [+0.0773, +0.5527] | 0.4100 (0.4100) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.1017 | +1.1017 | [+0.2464, +1.9988] | 1.5504 (1.5504) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.6655 | +0.6655 | [+0.2623, +1.1222] | 0.2075 (0.2075) | WORSE |

## I10RH R0G50Hb20 minus I10RH R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0267 | -0.0267 | [-0.0514, -0.0068] | 0.0036 (0.0036) | BETTER |
| C contact rate | -0.0078 | -0.0078 | [-0.0147, -0.0018] | 0.0062 (0.0062) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0530 | +0.0530 | [+0.0255, +0.0851] | 0.0561 (0.0561) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.5060 | +0.5060 | [+0.4033, +0.6235] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.9527 | +1.9527 | [+1.5615, +2.3927] | 1.8616 (1.8616) | WORSE |
| gap error 2.0 s (m) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.1158 | +0.1158 | [-0.0084, +0.2517] | 0.4100 (0.4100) | OPEN |
| gap error 6.0 s (m) | +0.4916 | +0.4916 | [-0.0447, +1.0534] | 1.5504 (1.5504) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.3495 | +0.3495 | [+0.0625, +0.6554] | 0.2075 (0.2075) | WORSE |

## I10RH R0C10Hb20 minus I10RH R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0071 | -0.0071 | [-0.0161, +0.0016] | 0.0036 (0.0036) | OPEN |
| C contact rate | -0.0047 | -0.0047 | [-0.0159, +0.0050] | 0.0062 (0.0062) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0265 | +0.0265 | [+0.0056, +0.0523] | 0.0561 (0.0561) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.3659 | +0.3659 | [+0.2837, +0.4585] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.2933 | +1.2933 | [+0.9764, +1.6473] | 1.8616 (1.8616) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.1488 | +0.1488 | [+0.0598, +0.2520] | 0.4100 (0.4100) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.4891 | +0.4891 | [+0.1963, +0.8242] | 1.5504 (1.5504) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.2832 | +0.2832 | [+0.1340, +0.4717] | 0.2075 (0.2075) | WORSE |

## I10B0 R0C10XHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0445 | -0.0445 | [-0.0782, -0.0164] | 0.0036 (0.0036) | BETTER |
| C contact rate | -0.0218 | -0.0218 | [-0.0429, -0.0051] | 0.0062 (0.0062) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1075 | +0.1075 | [+0.0631, +0.1551] | 0.0561 (0.0561) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +1.0949 | +1.0949 | [+0.8510, +1.3635] | 0.6885 (0.6885) | WORSE |
| gap bias 6.0 s (m) | +4.5436 | +4.5436 | [+3.5576, +5.6036] | 1.8616 (1.8616) | WORSE |
| gap error 2.0 s (m) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.3339 | +0.3339 | [+0.0805, +0.6097] | 0.4100 (0.4100) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.7135 | +1.7135 | [+0.5879, +2.8222] | 1.5504 (1.5504) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +1.2253 | +1.2253 | [+0.4938, +2.0222] | 0.2075 (0.2075) | WORSE |

## I10B0 R0G50Hb20 minus I10B0 R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0285 | -0.0285 | [-0.0541, -0.0074] | 0.0036 (0.0036) | BETTER |
| C contact rate | -0.0125 | -0.0125 | [-0.0265, -0.0020] | 0.0062 (0.0062) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0498 | +0.0498 | [+0.0247, +0.0785] | 0.0561 (0.0561) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.5412 | +0.5412 | [+0.4207, +0.6778] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.3292 | +2.3292 | [+1.8292, +2.8515] | 1.8616 (1.8616) | WORSE |
| gap error 2.0 s (m) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.1254 | +0.1254 | [-0.0112, +0.2669] | 0.4100 (0.4100) | OPEN |
| gap error 6.0 s (m) | +0.6172 | +0.6172 | [+0.0440, +1.2298] | 1.5504 (1.5504) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.4824 | +0.4824 | [+0.0894, +0.8710] | 0.2075 (0.2075) | WORSE |

## I10B0 R0C10Hb20 minus I10B0 R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0160 | -0.0160 | [-0.0293, -0.0055] | 0.0036 (0.0036) | BETTER |
| C contact rate | -0.0156 | -0.0156 | [-0.0316, -0.0040] | 0.0062 (0.0062) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0685 | +0.0685 | [+0.0354, +0.1083] | 0.0561 (0.0561) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.5364 | +0.5364 | [+0.4143, +0.6700] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.4517 | +2.4517 | [+1.7986, +3.1468] | 1.8616 (1.8616) | WORSE |
| gap error 2.0 s (m) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.1903 | +0.1903 | [+0.0668, +0.3211] | 0.4100 (0.4100) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.3217 | +1.3217 | [+0.5969, +2.0973] | 1.5504 (1.5504) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.6795 | +0.6795 | [+0.3401, +1.0815] | 0.2075 (0.2075) | WORSE |

## I10RH R0C10XHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0320 | -0.0320 | [-0.0562, -0.0097] | 0.0036 (0.0036) | BETTER |
| C contact rate | -0.0140 | -0.0140 | [-0.0316, +0.0000] | 0.0062 (0.0062) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0732 | +0.0732 | [+0.0340, +0.1197] | 0.0561 (0.0561) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0289 | -0.0289 | [-0.0625, -0.0014] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.8338 | +0.8338 | [+0.6296, +1.0797] | 0.6885 (0.6885) | WORSE |
| gap bias 6.0 s (m) | +3.1957 | +3.1957 | [+2.4945, +3.9988] | 1.8616 (1.8616) | WORSE |
| gap error 2.0 s (m) | +0.0232 | +0.0232 | [+0.0061, +0.0443] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.3857 | +0.3857 | [+0.1737, +0.6540] | 0.4100 (0.4100) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.1363 | +1.1363 | [+0.3133, +2.0477] | 1.5504 (1.5504) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0869 | +0.0869 | [+0.0084, +0.1970] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.6097 | +0.6097 | [+0.2030, +1.0626] | 0.2075 (0.2075) | WORSE |

## I10RH R0Hb20 minus I10B0 R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0089 | +0.0089 | [-0.0038, +0.0233] | 0.0036 (0.0036) | OPEN |
| C contact rate | +0.0047 | +0.0047 | [-0.0052, +0.0158] | 0.0062 (0.0062) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0234 | -0.0234 | [-0.0467, +0.0000] | 0.0561 (0.0561) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0289 | -0.0289 | [-0.0625, -0.0014] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.1246 | -0.1246 | [-0.2765, +0.0069] | 0.6885 (0.6885) | OPEN |
| gap bias 6.0 s (m) | -0.7223 | -0.7223 | [-1.2264, -0.2328] | 1.8616 (1.8616) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0232 | +0.0232 | [+0.0061, +0.0443] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.0231 | +0.0231 | [-0.1018, +0.1588] | 0.4100 (0.4100) | OPEN |
| gap error 6.0 s (m) | -0.1572 | -0.1572 | [-0.5940, +0.2928] | 1.5504 (1.5504) | OPEN |
| time-gap error 2.0 s (s) | +0.0869 | +0.0869 | [+0.0084, +0.1970] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | -0.3781 | -0.3781 | [-0.7966, +0.0518] | 0.2075 (0.2075) | OPEN |

## I10RH R0C10Hb20 minus I10B0 R0C10Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0178 | +0.0178 | [+0.0000, +0.0370] | 0.0036 (0.0036) | OPEN |
| C contact rate | +0.0156 | +0.0156 | [+0.0016, +0.0348] | 0.0062 (0.0062) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0561 | -0.0561 | [-0.0932, -0.0233] | 0.0561 (0.0561) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0289 | -0.0289 | [-0.0625, -0.0014] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.3079 | -0.3079 | [-0.4861, -0.1479] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -1.9019 | -1.9019 | [-2.8845, -1.0872] | 1.8616 (1.8616) | BETTER |
| gap error 2.0 s (m) | +0.0232 | +0.0232 | [+0.0061, +0.0443] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.0258 | +0.0258 | [-0.0992, +0.1717] | 0.4100 (0.4100) | OPEN |
| gap error 6.0 s (m) | -0.9261 | -0.9261 | [-1.7303, -0.1130] | 1.5504 (1.5504) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0869 | +0.0869 | [+0.0084, +0.1970] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | -0.6265 | -0.6265 | [-1.1494, -0.1007] | 0.2075 (0.2075) | BETTER |

## I10RH R0G50Hb20 minus I10B0 R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0089 | +0.0089 | [-0.0072, +0.0272] | 0.0036 (0.0036) | OPEN |
| C contact rate | +0.0047 | +0.0047 | [-0.0032, +0.0143] | 0.0062 (0.0062) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0140 | -0.0140 | [-0.0312, +0.0019] | 0.0561 (0.0561) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0289 | -0.0289 | [-0.0625, -0.0014] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.1374 | -0.1374 | [-0.2630, -0.0254] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.7436 | -0.7436 | [-1.1848, -0.3561] | 1.8616 (1.8616) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0232 | +0.0232 | [+0.0061, +0.0443] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.0674 | +0.0674 | [-0.0297, +0.1806] | 0.4100 (0.4100) | OPEN |
| gap error 6.0 s (m) | -0.0935 | -0.0935 | [-0.3865, +0.2093] | 1.5504 (1.5504) | OPEN |
| time-gap error 2.0 s (s) | +0.0869 | +0.0869 | [+0.0084, +0.1970] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | -0.2302 | -0.2302 | [-0.5547, +0.0958] | 0.2075 (0.2075) | OPEN |

## I10RH R0G0Hb20 minus I10B0 R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0071 | +0.0071 | [-0.0063, +0.0212] | 0.0036 (0.0036) | OPEN |
| C contact rate | +0.0000 | +0.0000 | [-0.0062, +0.0062] | 0.0062 (0.0062) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0171 | -0.0171 | [-0.0369, -0.0019] | 0.0561 (0.0561) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0289 | -0.0289 | [-0.0625, -0.0014] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.1023 | -0.1023 | [-0.2071, -0.0110] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.3670 | -0.3670 | [-0.5882, -0.1477] | 1.8616 (1.8616) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0232 | +0.0232 | [+0.0061, +0.0443] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.0770 | +0.0770 | [+0.0029, +0.1619] | 0.4100 (0.4100) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.0322 | +0.0322 | [-0.1692, +0.2416] | 1.5504 (1.5504) | OPEN |
| time-gap error 2.0 s (s) | +0.0869 | +0.0869 | [+0.0084, +0.1970] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | -0.0972 | -0.0972 | [-0.3020, +0.0993] | 0.2075 (0.2075) | OPEN |

## I10RH R0KHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0071 | +0.0071 | [+0.0000, +0.0161] | 0.0036 (0.0036) | OPEN |
| C contact rate | +0.0031 | +0.0031 | [+0.0000, +0.0079] | 0.0062 (0.0062) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0062 | -0.0062 | [-0.0180, +0.0039] | 0.0561 (0.0561) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0289 | -0.0289 | [-0.0625, -0.0014] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.0917 | -0.0917 | [-0.1929, -0.0052] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.2585 | -0.2585 | [-0.4475, -0.0862] | 1.8616 (1.8616) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0232 | +0.0232 | [+0.0061, +0.0443] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.0895 | +0.0895 | [+0.0173, +0.1754] | 0.4100 (0.4100) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.0347 | +0.0347 | [-0.1319, +0.2019] | 1.5504 (1.5504) | OPEN |
| time-gap error 2.0 s (s) | +0.0869 | +0.0869 | [+0.0084, +0.1970] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | -0.0558 | -0.0558 | [-0.2500, +0.1421] | 0.2075 (0.2075) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10B0_R0C10Hb20 (2) | 0.0445 [0.0166, 0.0858] | 0.0374 [0.0071, 0.0767] | 0.2134 [0.1410, 0.2908] | 0.0693 [0.0038, 0.1357] | 1.4888 [0.9542, 2.0571] | 6.0142 [4.3254, 7.7101] | 0.4401 [0.3809, 0.5025] | 2.5896 [2.1552, 3.1118] | 7.7168 [6.3830, 9.2579] | 0.8098 [0.4803, 1.2385] | 3.2844 [2.1531, 4.5684] |
| I10B0_R0C10XHb20 (2) | 0.0427 [0.0154, 0.0840] | 0.0421 [0.0096, 0.0846] | 0.2009 [0.1327, 0.2746] | 0.0693 [0.0038, 0.1357] | 1.5595 [1.0401, 2.1278] | 5.7329 [4.1033, 7.4876] | 0.4401 [0.3809, 0.5025] | 2.5862 [2.1562, 3.1027] | 7.5098 [6.2460, 8.9867] | 0.8098 [0.4803, 1.2385] | 3.3196 [2.1717, 4.6138] |
| I10B0_R0G0Hb20 (2) | 0.0890 [0.0436, 0.1456] | 0.0654 [0.0221, 0.1176] | 0.0950 [0.0517, 0.1437] | 0.0693 [0.0038, 0.1357] | 0.4112 [-0.1021, 0.9209] | 1.2333 [-0.4078, 2.8483] | 0.4401 [0.3809, 0.5025] | 2.2739 [1.8872, 2.6757] | 5.7778 [4.7153, 6.9391] | 0.8098 [0.4803, 1.2385] | 2.1225 [1.1735, 3.2254] |
| I10B0_R0G50Hb20 (2) | 0.0605 [0.0284, 0.1032] | 0.0530 [0.0160, 0.1003] | 0.1449 [0.0862, 0.2097] | 0.0693 [0.0038, 0.1357] | 0.9524 [0.4339, 1.4810] | 3.5625 [1.9880, 5.2675] | 0.4401 [0.3809, 0.5025] | 2.3993 [2.0000, 2.8484] | 6.3950 [5.2589, 7.6935] | 0.8098 [0.4803, 1.2385] | 2.6049 [1.6319, 3.7265] |
| I10B0_R0Hb20 (2) | 0.0836 [0.0396, 0.1376] | 0.0530 [0.0162, 0.0958] | 0.1542 [0.0926, 0.2250] | 0.0693 [0.0038, 0.1357] | 0.8123 [0.2410, 1.4303] | 3.3374 [1.6215, 5.1913] | 0.4401 [0.3809, 0.5025] | 2.5359 [2.1626, 2.9715] | 6.6418 [5.5197, 7.9649] | 0.8098 [0.4803, 1.2385] | 2.6050 [1.6488, 3.7716] |
| I10B0_R0KHb20 (2) | 0.0872 [0.0427, 0.1441] | 0.0639 [0.0217, 0.1159] | 0.0935 [0.0528, 0.1415] | 0.0693 [0.0038, 0.1357] | 0.4646 [-0.0515, 0.9747] | 1.1893 [-0.4969, 2.7960] | 0.4401 [0.3809, 0.5025] | 2.2523 [1.8666, 2.6603] | 5.7963 [4.7441, 6.9550] | 0.8098 [0.4803, 1.2385] | 2.0944 [1.1339, 3.2235] |
| I10RH_R0C10Hb20 (2) | 0.0623 [0.0260, 0.1117] | 0.0530 [0.0144, 0.0995] | 0.1573 [0.0983, 0.2284] | 0.0404 [-0.0444, 0.1203] | 1.1809 [0.5989, 1.7811] | 4.1123 [2.4746, 5.8342] | 0.4632 [0.3951, 0.5348] | 2.6154 [2.1790, 3.1215] | 6.7907 [5.6032, 8.1732] | 0.8967 [0.5148, 1.3922] | 2.6579 [1.6179, 3.7803] |
| I10RH_R0C10XHb20 (2) | 0.0552 [0.0215, 0.1048] | 0.0498 [0.0118, 0.0971] | 0.1667 [0.1066, 0.2376] | 0.0404 [-0.0444, 0.1203] | 1.2984 [0.7186, 1.9070] | 4.3850 [2.7552, 6.1241] | 0.4632 [0.3951, 0.5348] | 2.6381 [2.1938, 3.1656] | 6.9327 [5.7398, 8.3397] | 0.8967 [0.5148, 1.3922] | 2.7040 [1.6612, 3.8044] |
| I10RH_R0G0Hb20 (2) | 0.0961 [0.0446, 0.1597] | 0.0654 [0.0223, 0.1179] | 0.0779 [0.0386, 0.1226] | 0.0404 [-0.0444, 0.1203] | 0.3090 [-0.2683, 0.8573] | 0.8663 [-0.8543, 2.4847] | 0.4632 [0.3951, 0.5348] | 2.3509 [1.9397, 2.7879] | 5.8100 [4.7566, 6.9728] | 0.8967 [0.5148, 1.3922] | 2.0252 [1.1014, 3.1056] |
| I10RH_R0G50Hb20 (2) | 0.0694 [0.0304, 0.1236] | 0.0576 [0.0166, 0.1075] | 0.1308 [0.0775, 0.1934] | 0.0404 [-0.0444, 0.1203] | 0.8150 [0.2526, 1.3967] | 2.8190 [1.1537, 4.5702] | 0.4632 [0.3951, 0.5348] | 2.4667 [2.0511, 2.9326] | 6.3016 [5.1564, 7.6033] | 0.8967 [0.5148, 1.3922] | 2.3747 [1.4199, 3.3986] |
| I10RH_R0Hb20 (2) | 0.0925 [0.0429, 0.1505] | 0.0576 [0.0174, 0.1078] | 0.1308 [0.0743, 0.1997] | 0.0404 [-0.0444, 0.1203] | 0.6877 [0.0752, 1.3215] | 2.6151 [0.8977, 4.4329] | 0.4632 [0.3951, 0.5348] | 2.5589 [2.1529, 3.0128] | 6.4845 [5.3939, 7.7404] | 0.8967 [0.5148, 1.3922] | 2.2269 [1.3214, 3.2800] |
| I10RH_R0KHb20 (2) | 0.0943 [0.0444, 0.1568] | 0.0670 [0.0222, 0.1220] | 0.0872 [0.0467, 0.1318] | 0.0404 [-0.0444, 0.1203] | 0.3729 [-0.2053, 0.9203] | 0.9308 [-0.7732, 2.5775] | 0.4632 [0.3951, 0.5348] | 2.3418 [1.9335, 2.7794] | 5.8310 [4.7737, 6.9965] | 0.8967 [0.5148, 1.3922] | 2.0385 [1.1075, 3.1296] |
| I10UEA_R0C10XHb20 (0,1) | 0.0623 [0.0388, 0.0910] | 0.0280 [0.0123, 0.0462] | 0.3910 [0.3191, 0.4592] | -0.2236 [-0.2757, -0.1682] | 3.0145 [2.3455, 3.6779] | 10.3479 [8.3469, 12.6031] | 0.4729 [0.4299, 0.5219] | 3.5797 [3.0625, 4.1116] | 11.7414 [10.0248, 13.7045] | 1.0435 [0.7059, 1.4212] | 4.8571 [3.8136, 5.9361] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
