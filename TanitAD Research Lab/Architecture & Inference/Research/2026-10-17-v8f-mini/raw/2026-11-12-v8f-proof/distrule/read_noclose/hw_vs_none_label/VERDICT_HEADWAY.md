# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 22, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSRZ:LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDRZ:LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQZ:LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDQZ:LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSRZ:LSRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDRZ:LSDRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQZ:LSRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDQZ:LSDRL", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, LSRZ minus LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDRZ minus LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSQZ minus LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDQZ minus LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSRZ minus LSRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDRZ minus LSDRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSQZ minus LSRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDQZ minus LSDRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## LSRZ minus LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0391 / -0.0374 | -0.0383 | [-0.0651, -0.0160] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0125 / -0.0249 | -0.0187 | [-0.0337, -0.0075] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0685 / +0.1075 | +0.0880 | [+0.0429, +0.1399] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.3458 / +0.6681 | +0.5070 | [+0.1891, +0.8876] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.3267 / +2.2346 | +1.7807 | [+0.6954, +2.9710] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +2.9093 / +4.3470 | +3.6282 | [+1.7035, +5.7337] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.2820 / +0.5046 | +0.3933 | [+0.1422, +0.6912] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.8663 / +1.9800 | +1.4232 | [+0.4327, +2.6050] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +2.0935 / +2.7737 | +2.4336 | [+1.0028, +4.1739] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0196 / +0.3520 | +0.1858 | [+0.0113, +0.4711] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0901 / +0.8789 | +0.4845 | [+0.0395, +1.0831] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDRZ minus LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0214 / -0.0196 | -0.0205 | [-0.0370, -0.0047] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0062 / -0.0280 | -0.0171 | [-0.0379, -0.0038] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0888 / +0.0483 | +0.0685 | [+0.0308, +0.1152] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.3912 / +0.3964 | +0.3938 | [+0.1664, +0.6729] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.5012 / +1.1968 | +1.3490 | [+0.5780, +2.2843] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.8728 / +2.7592 | +2.8160 | [+1.5724, +4.4155] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3030 / +0.2506 | +0.2768 | [+0.0923, +0.5038] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.1816 / +1.1444 | +1.1630 | [+0.4993, +2.0057] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +2.3327 / +1.1906 | +1.7617 | [+0.3010, +3.4342] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0247 / +0.0513 | +0.0380 | [+0.0128, +0.0844] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.1690 / +0.2656 | +0.2173 | [+0.0784, +0.4537] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSQZ minus LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0623 / -0.0552 | -0.0587 | [-0.0914, -0.0285] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0171 / -0.0280 | -0.0226 | [-0.0377, -0.0108] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1511 / +0.1573 | +0.1542 | [+0.1038, +0.2081] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.8555 / +1.1444 | +1.0000 | [+0.5938, +1.4843] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.9195 / +3.5121 | +3.2158 | [+1.9453, +4.6577] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +5.5215 / +7.0241 | +6.2728 | [+3.7774, +8.9169] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.6590 / +0.8810 | +0.7700 | [+0.4116, +1.1963] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.1998 / +3.1412 | +2.6705 | [+1.4610, +4.0778] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +4.4762 / +5.0192 | +4.7477 | [+2.7708, +7.0380] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.2527 / +0.4134 | +0.3330 | [+0.1734, +0.5511] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.8422 / +1.2154 | +1.0288 | [+0.6558, +1.4939] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDQZ minus LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0747 / -0.0712 | -0.0730 | [-0.1093, -0.0396] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0093 / -0.0421 | -0.0257 | [-0.0534, -0.0037] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1776 / +0.1231 | +0.1503 | [+0.0962, +0.2126] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.1063 / +1.0719 | +1.0891 | [+0.7045, +1.5278] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.6968 / +3.1513 | +3.4240 | [+2.2331, +4.8193] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.3361 / +5.3406 | +5.8383 | [+3.9153, +8.0041] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.8669 / +0.8163 | +0.8416 | [+0.5175, +1.2199] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +3.0901 / +2.7670 | +2.9286 | [+1.8140, +4.1698] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +5.4860 / +3.4332 | +4.4596 | [+2.1960, +6.8914] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.1730 / +0.2207 | +0.1968 | [+0.1328, +0.2835] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.8865 / +0.7244 | +0.8055 | [+0.5159, +1.1573] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSRZ minus LSRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0356 / +0.0463 | +0.0409 | [+0.0229, +0.0628] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0187 / +0.0327 | +0.0257 | [+0.0089, +0.0493] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0047 / +0.0545 | +0.0249 | [-0.0178, +0.0717] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0901 / +0.2214 | +0.0656 | [-0.1232, +0.3189] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.0221 / +0.9072 | +0.4425 | [-0.1863, +1.2668] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.1772 / +0.9120 | +0.3674 | [-0.5938, +1.6682] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0443 / +0.2112 | +0.0835 | [-0.0796, +0.3035] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.1565 / +1.2628 | +0.7096 | [+0.0006, +1.5584] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.1843 / +1.7383 | +0.9613 | [-0.1539, +2.4307] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0216 / +0.3123 | +0.1453 | [-0.0298, +0.4338] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0480 / +0.7887 | +0.3703 | [-0.1031, +1.0041] | 1.7849 (0.8089) | OPEN |

## LSDRZ minus LSDRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0676 / +0.0658 | +0.0667 | [+0.0406, +0.0956] | 0.0463 (0.0214) | WORSE |
| C contact rate | +0.0125 / +0.0436 | +0.0280 | [+0.0065, +0.0608] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0093 / -0.0047 | +0.0023 | [-0.0205, +0.0262] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0454 / -0.0462 | -0.0458 | [-0.1491, +0.0405] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.1487 / -0.2226 | -0.0370 | [-0.4165, +0.3220] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.1742 / -0.8432 | -0.3345 | [-1.2115, +0.5526] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0117 / -0.0241 | -0.0179 | [-0.1008, +0.0463] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.3979 / +0.3063 | +0.3521 | [+0.0894, +0.6321] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.4984 / +0.0794 | +0.2889 | [-0.2910, +0.8571] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0193 / +0.0118 | -0.0037 | [-0.0361, +0.0468] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0033 / +0.1888 | +0.0927 | [-0.0422, +0.3503] | 1.7849 (0.8089) | OPEN |

## LSQZ minus LSRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0125 / +0.0285 | +0.0205 | [+0.0041, +0.0400] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0140 / +0.0296 | +0.0218 | [+0.0067, +0.0447] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0779 / +0.1044 | +0.0911 | [+0.0529, +0.1285] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4196 / +0.6977 | +0.5586 | [+0.3365, +0.8634] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.5707 / +2.1846 | +1.8776 | [+1.1084, +2.8243] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +2.4350 / +3.5891 | +3.0121 | [+1.5913, +4.6524] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3328 / +0.5875 | +0.4602 | [+0.2479, +0.7361] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.4899 / +2.4239 | +1.9569 | [+1.1153, +2.9915] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +2.5669 / +3.9838 | +3.2754 | [+1.8342, +4.9645] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.2115 / +0.3736 | +0.2926 | [+0.1352, +0.5141] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.7042 / +1.1252 | +0.9147 | [+0.5545, +1.3877] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDQZ minus LSDRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0142 / +0.0142 | +0.0142 | [+0.0013, +0.0278] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0093 / +0.0296 | +0.0195 | [+0.0048, +0.0418] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0981 / +0.0701 | +0.0841 | [+0.0497, +0.1211] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6698 / +0.6293 | +0.6496 | [+0.4482, +0.8765] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.3442 / +1.7319 | +2.0381 | [+1.1936, +2.9902] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +3.6374 / +1.7382 | +2.6878 | [+1.0404, +4.5881] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.5522 / +0.5416 | +0.5469 | [+0.3698, +0.7496] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.3064 / +1.9290 | +2.1177 | [+1.3563, +2.9987] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.6517 / +2.3220 | +2.9868 | [+1.5995, +4.6255] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.1289 / +0.1812 | +0.1551 | [+0.0940, +0.2400] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.7141 / +0.6476 | +0.6809 | [+0.4368, +0.9801] | 1.7849 (0.8089) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LS (0,1) | 0.1246 [0.0823, 0.1714] | 0.0444 [0.0193, 0.0766] | 0.2111 [0.1246, 0.3163] | -0.2350 [-0.3195, -0.1538] | 1.1867 [0.5392, 1.9640] | 5.4393 [3.1026, 8.0310] | 0.5163 [0.4689, 0.5676] | 2.7856 [2.2778, 3.4051] | 8.3298 [6.8681, 10.0377] | 1.2365 [0.7221, 1.7936] | 2.4602 [1.4184, 3.6642] |
| LSD (0,1) | 0.1343 [0.0873, 0.1879] | 0.0452 [0.0109, 0.0906] | 0.2531 [0.1958, 0.3138] | -0.2031 [-0.2748, -0.1325] | 1.4817 [0.8172, 2.2957] | 6.7598 [4.6651, 8.6599] | 0.5100 [0.4630, 0.5586] | 2.8968 [2.3956, 3.5375] | 9.3372 [8.1968, 10.6886] | 1.4640 [0.8170, 2.2047] | 2.5391 [1.6003, 3.5518] |
| LSDQZ (0,1) | 0.0614 [0.0318, 0.1007] | 0.0195 [0.0048, 0.0418] | 0.4034 [0.3173, 0.4888] | 0.8860 [0.5179, 1.2950] | 4.9057 [3.3476, 6.5154] | 12.5981 [9.3383, 15.5273] | 1.3516 [1.0134, 1.7377] | 5.8253 [4.4966, 7.2675] | 13.7967 [11.1344, 16.3311] | 1.6608 [1.0197, 2.3732] | 3.3446 [2.3064, 4.4521] |
| LSDRL (0,1) | 0.0472 [0.0227, 0.0829] | 0.0000 [0.0000, 0.0000] | 0.3193 [0.2469, 0.3910] | 0.2364 [-0.0076, 0.5420] | 2.8677 [1.8890, 3.9087] | 9.9103 [7.7746, 11.9708] | 0.8047 [0.5833, 1.0843] | 3.7077 [2.8459, 4.6228] | 10.8099 [8.9499, 12.7615] | 1.5058 [0.8571, 2.2451] | 2.6637 [1.6935, 3.7219] |
| LSDRZ (0,1) | 0.1139 [0.0709, 0.1612] | 0.0280 [0.0065, 0.0608] | 0.3217 [0.2441, 0.4007] | 0.1906 [-0.0228, 0.4479] | 2.8307 [1.6862, 4.0351] | 9.5758 [7.1212, 11.8803] | 0.7868 [0.5898, 1.0280] | 4.0598 [3.1581, 5.0082] | 11.0988 [9.1744, 13.0523] | 1.5020 [0.8594, 2.2291] | 2.7564 [1.8403, 3.7299] |
| LSQZ (0,1) | 0.0658 [0.0378, 0.1040] | 0.0218 [0.0067, 0.0447] | 0.3653 [0.2687, 0.4650] | 0.7650 [0.3845, 1.2240] | 4.4025 [3.1204, 5.8402] | 11.7121 [8.9593, 14.4143] | 1.2863 [0.9192, 1.7256] | 5.4560 [4.3283, 6.7354] | 13.0775 [10.7193, 15.4900] | 1.5695 [1.0548, 2.1342] | 3.4890 [2.5209, 4.5788] |
| LSRL (0,1) | 0.0454 [0.0238, 0.0728] | 0.0000 [0.0000, 0.0000] | 0.2741 [0.1700, 0.3895] | 0.2064 [-0.0545, 0.5267] | 2.5248 [1.6354, 3.5112] | 8.7001 [6.1813, 11.2690] | 0.8261 [0.5991, 1.1154] | 3.4991 [2.7134, 4.3601] | 9.8022 [7.5079, 12.2263] | 1.2769 [0.7638, 1.8367] | 2.5743 [1.5172, 3.8082] |
| LSRZ (0,1) | 0.0863 [0.0532, 0.1267] | 0.0257 [0.0089, 0.0493] | 0.2991 [0.2188, 0.3845] | 0.2720 [-0.0115, 0.6056] | 2.9674 [2.0333, 4.0448] | 9.0675 [6.8195, 11.3895] | 0.9095 [0.6568, 1.2228] | 4.2088 [3.3643, 5.1750] | 10.7634 [8.9089, 12.6988] | 1.4222 [0.9760, 1.8918] | 2.9447 [2.1333, 3.8503] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
