# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 26, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSXP:LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSXG:LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSXC:LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSXZ:LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDXP:LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDXG:LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDXC:LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDXZ:LSD", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, LSXP minus LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSXG minus LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSXC minus LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSXZ minus LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDXP minus LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDXG minus LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDXC minus LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDXZ minus LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## LSXP minus LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0730 / -0.0783 | -0.0756 | [-0.1080, -0.0479] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0312 / -0.0561 | -0.0436 | [-0.0742, -0.0189] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0701 / +0.0452 | +0.0576 | [+0.0210, +0.1019] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.3938 / +0.3991 | +0.3964 | [+0.1595, +0.6970] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.1878 / +1.1040 | +1.1459 | [+0.5568, +1.9019] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.8119 / +2.7920 | +2.8020 | [+1.5813, +4.3397] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.2927 / +0.2730 | +0.2828 | [+0.0759, +0.5455] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.5524 / +0.5616 | +0.5570 | [+0.0913, +1.1679] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.6323 / +0.5043 | +1.0683 | [-0.3059, +2.5154] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0387 / +0.0398 | +0.0393 | [+0.0184, +0.0642] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.1411 / +0.0736 | +0.1074 | [+0.0176, +0.2091] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSXG minus LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0409 / -0.0534 | -0.0472 | [-0.0802, -0.0222] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0171 / -0.0405 | -0.0288 | [-0.0552, -0.0114] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0623 / +0.1028 | +0.0826 | [+0.0389, +0.1354] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.3584 / +0.6327 | +0.4955 | [+0.1939, +0.8719] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.2478 / +1.9177 | +1.5828 | [+0.6550, +2.6045] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.4507 / +4.3368 | +3.3938 | [+1.3490, +5.6111] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.2736 / +0.4561 | +0.3649 | [+0.1231, +0.6593] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.6995 / +1.5293 | +1.1144 | [+0.3129, +2.1599] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +1.6051 / +2.3020 | +1.9536 | [+0.5790, +3.6951] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0245 / +0.2275 | +0.1260 | [+0.0149, +0.3192] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0372 / +0.6686 | +0.3529 | [-0.0035, +0.8489] | 1.7849 (0.8089) | OPEN |

## LSXC minus LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0641 / -0.0480 | -0.0560 | [-0.0902, -0.0232] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0171 / -0.0280 | -0.0226 | [-0.0385, -0.0106] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0857 / +0.0872 | +0.0864 | [+0.0459, +0.1329] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4943 / +0.5031 | +0.4987 | [+0.2145, +0.8447] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.6249 / +1.8627 | +1.7438 | [+0.8968, +2.7362] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +3.3567 / +4.2899 | +3.8233 | [+1.9851, +5.9487] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3873 / +0.3664 | +0.3769 | [+0.1308, +0.6749] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.0042 / +1.4092 | +1.2067 | [+0.4820, +2.1143] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +2.4458 / +2.3566 | +2.4012 | [+0.9610, +4.1074] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0359 / +0.0838 | +0.0598 | [+0.0214, +0.1262] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.2752 / +0.3157 | +0.2955 | [+0.1317, +0.4796] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSXZ minus LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0552 / -0.0552 | -0.0552 | [-0.0867, -0.0278] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0171 / -0.0234 | -0.0202 | [-0.0329, -0.0095] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0857 / +0.0701 | +0.0779 | [+0.0417, +0.1238] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4462 / +0.4487 | +0.4475 | [+0.1850, +0.7670] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.7047 / +1.4506 | +1.5777 | [+0.7244, +2.6591] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +3.6705 / +3.4400 | +3.5553 | [+1.8666, +5.7495] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3467 / +0.3329 | +0.3398 | [+0.1131, +0.6181] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.1505 / +1.0994 | +1.1250 | [+0.3918, +2.0461] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +2.6855 / +1.5672 | +2.1264 | [+0.4799, +4.1573] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0664 / +0.0580 | +0.0622 | [+0.0270, +0.1045] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.2496 / +0.2876 | +0.2686 | [+0.1405, +0.4144] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDXP minus LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0836 / -0.0712 | -0.0774 | [-0.1138, -0.0447] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0187 / -0.0717 | -0.0452 | [-0.0906, -0.0109] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0732 / +0.0467 | +0.0600 | [+0.0309, +0.1012] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4028 / +0.3829 | +0.3928 | [+0.1535, +0.6873] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.2391 / +1.1741 | +1.2066 | [+0.5779, +2.0073] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.3716 / +3.2662 | +2.8189 | [+1.5011, +4.3958] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.2899 / +0.2473 | +0.2686 | [+0.0773, +0.5135] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.7015 / +0.6531 | +0.6773 | [+0.1066, +1.3884] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.5073 / +0.8365 | +1.1719 | [-0.0344, +2.5524] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0422 / +0.0370 | +0.0396 | [+0.0173, +0.0654] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.1320 / +0.0608 | +0.0964 | [+0.0007, +0.1973] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDXG minus LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0267 / -0.0356 | -0.0311 | [-0.0478, -0.0159] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0093 / -0.0312 | -0.0202 | [-0.0414, -0.0044] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0717 / +0.0327 | +0.0522 | [+0.0223, +0.0937] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.3978 / +0.3818 | +0.3898 | [+0.1506, +0.6815] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.2331 / +0.9717 | +1.1024 | [+0.3912, +1.9702] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.2502 / +2.5135 | +2.3818 | [+1.2739, +3.8110] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.2813 / +0.2468 | +0.2640 | [+0.0744, +0.4949] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.8662 / +0.7427 | +0.8045 | [+0.1885, +1.5803] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.6538 / +0.6283 | +1.1411 | [-0.2418, +2.5982] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0228 / +0.0203 | +0.0216 | [+0.0073, +0.0391] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0668 / +0.1091 | +0.0879 | [+0.0146, +0.2016] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDXC minus LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0569 / -0.0569 | -0.0569 | [-0.0891, -0.0283] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0093 / -0.0467 | -0.0280 | [-0.0581, -0.0054] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0779 / +0.0670 | +0.0724 | [+0.0393, +0.1130] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4637 / +0.5101 | +0.4869 | [+0.2003, +0.8484] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.5503 / +1.4642 | +1.5073 | [+0.7291, +2.4900] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.9927 / +3.5079 | +3.2503 | [+1.8688, +5.0175] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3566 / +0.3593 | +0.3580 | [+0.1257, +0.6536] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.0991 / +1.0841 | +1.0916 | [+0.3631, +1.9848] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +2.2330 / +1.4605 | +1.8467 | [+0.3867, +3.5011] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0333 / +0.0284 | +0.0309 | [+0.0134, +0.0512] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.1816 / +0.1272 | +0.1544 | [+0.0476, +0.2739] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDXZ minus LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0765 / -0.0694 | -0.0730 | [-0.1109, -0.0392] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0125 / -0.0389 | -0.0257 | [-0.0502, -0.0067] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0841 / +0.0732 | +0.0787 | [+0.0467, +0.1174] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4465 / +0.4504 | +0.4485 | [+0.1827, +0.7673] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.4237 / +1.3651 | +1.3944 | [+0.6795, +2.2961] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.6602 / +3.1693 | +2.9147 | [+1.6978, +4.3666] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3484 / +0.3041 | +0.3263 | [+0.1102, +0.5958] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.9697 / +0.9601 | +0.9649 | [+0.3166, +1.7784] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.8827 / +1.1086 | +1.4957 | [+0.3181, +2.8628] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0568 / +0.0510 | +0.0539 | [+0.0246, +0.0874] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.2648 / +0.2268 | +0.2458 | [+0.1193, +0.3921] | 1.7849 (0.8089) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LS (0,1) | 0.1246 [0.0823, 0.1714] | 0.0444 [0.0193, 0.0766] | 0.2111 [0.1246, 0.3163] | -0.2350 [-0.3195, -0.1538] | 1.1867 [0.5392, 1.9640] | 5.4393 [3.1026, 8.0310] | 0.5163 [0.4689, 0.5676] | 2.7856 [2.2778, 3.4051] | 8.3298 [6.8681, 10.0377] | 1.2365 [0.7221, 1.7936] | 2.4602 [1.4184, 3.6642] |
| LSD (0,1) | 0.1343 [0.0873, 0.1879] | 0.0452 [0.0109, 0.0906] | 0.2531 [0.1958, 0.3138] | -0.2031 [-0.2748, -0.1325] | 1.4817 [0.8172, 2.2957] | 6.7598 [4.6651, 8.6599] | 0.5100 [0.4630, 0.5586] | 2.8968 [2.3956, 3.5375] | 9.3372 [8.1968, 10.6886] | 1.4640 [0.8170, 2.2047] | 2.5391 [1.6003, 3.5518] |
| LSDXC (0,1) | 0.0774 [0.0473, 0.1158] | 0.0171 [0.0042, 0.0385] | 0.3255 [0.2582, 0.3932] | 0.2838 [0.0126, 0.6204] | 2.9890 [1.8870, 4.1690] | 10.0100 [7.6793, 12.3139] | 0.8679 [0.6258, 1.1778] | 3.9884 [3.0501, 5.0176] | 11.1839 [9.2636, 13.2363] | 1.4949 [0.8513, 2.2420] | 2.6935 [1.7438, 3.7231] |
| LSDXG (0,1) | 0.1032 [0.0640, 0.1468] | 0.0249 [0.0045, 0.0571] | 0.3053 [0.2281, 0.3835] | 0.1866 [-0.0366, 0.4597] | 2.5841 [1.4691, 3.7323] | 9.1416 [6.8977, 11.2569] | 0.7740 [0.5701, 1.0218] | 3.7012 [2.8265, 4.6442] | 10.4782 [8.6535, 12.2760] | 1.4856 [0.8401, 2.2283] | 2.6270 [1.7133, 3.6222] |
| LSDXP (0,1) | 0.0569 [0.0269, 0.0990] | 0.0000 [0.0000, 0.0000] | 0.3131 [0.2423, 0.3844] | 0.1897 [-0.0408, 0.4770] | 2.6883 [1.6867, 3.6943] | 9.5787 [7.5537, 11.5574] | 0.7786 [0.5764, 1.0350] | 3.5741 [2.7238, 4.4611] | 10.5090 [8.7594, 12.3479] | 1.5036 [0.8550, 2.2437] | 2.6355 [1.6770, 3.6968] |
| LSDXZ (0,1) | 0.0614 [0.0323, 0.1030] | 0.0195 [0.0032, 0.0467] | 0.3318 [0.2650, 0.3969] | 0.2453 [-0.0049, 0.5551] | 2.8761 [1.8262, 4.0054] | 9.6745 [7.5776, 11.6781] | 0.8362 [0.6095, 1.1196] | 3.8617 [2.9711, 4.8228] | 10.8328 [9.1522, 12.5669] | 1.5179 [0.8692, 2.2734] | 2.7849 [1.8421, 3.8356] |
| LSXC (0,1) | 0.0685 [0.0363, 0.1122] | 0.0218 [0.0072, 0.0433] | 0.2975 [0.2024, 0.3978] | 0.2637 [-0.0212, 0.6042] | 2.9305 [1.9758, 3.9639] | 9.2626 [6.7534, 11.7512] | 0.8931 [0.6464, 1.2029] | 3.9922 [3.2155, 4.8727] | 10.7310 [8.6213, 12.9089] | 1.2963 [0.7856, 1.8377] | 2.7557 [1.7348, 3.9155] |
| LSXG (0,1) | 0.0774 [0.0481, 0.1123] | 0.0156 [0.0053, 0.0288] | 0.2936 [0.2166, 0.3781] | 0.2605 [-0.0090, 0.5814] | 2.7694 [1.9052, 3.7896] | 8.8331 [6.7092, 10.9397] | 0.8811 [0.6359, 1.1904] | 3.9000 [3.1509, 4.7730] | 10.2834 [8.4180, 12.1821] | 1.3624 [0.9040, 1.8612] | 2.8131 [1.9603, 3.7515] |
| LSXP (0,1) | 0.0489 [0.0257, 0.0783] | 0.0008 [0.0000, 0.0032] | 0.2687 [0.1630, 0.3844] | 0.1615 [-0.0849, 0.4606] | 2.3326 [1.4548, 3.2861] | 8.2413 [5.7126, 10.8140] | 0.7991 [0.5922, 1.0687] | 3.3426 [2.5848, 4.1317] | 9.3981 [7.1138, 11.8388] | 1.2757 [0.7665, 1.8341] | 2.5676 [1.4979, 3.8216] |
| LSXZ (0,1) | 0.0694 [0.0419, 0.1063] | 0.0241 [0.0072, 0.0500] | 0.2889 [0.1850, 0.3985] | 0.2125 [-0.0567, 0.5367] | 2.7644 [1.6694, 3.9942] | 8.9946 [6.2010, 12.0337] | 0.8560 [0.6312, 1.1440] | 3.9105 [2.9618, 4.9515] | 10.4562 [8.0463, 13.2088] | 1.2986 [0.7759, 1.8737] | 2.7288 [1.6905, 3.9145] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
