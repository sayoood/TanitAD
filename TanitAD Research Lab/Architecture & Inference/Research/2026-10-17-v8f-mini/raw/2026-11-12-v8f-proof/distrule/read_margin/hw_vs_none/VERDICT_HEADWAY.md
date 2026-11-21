# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 30, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSMC:LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSMT:LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSMG:LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSML:LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSMH:LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDMC:LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDMT:LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDMG:LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDML:LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDMH:LSD", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, LSMC minus LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSMT minus LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSMG minus LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSML minus LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSMH minus LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDMC minus LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDMT minus LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDMG minus LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDML minus LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDMH minus LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## LSMC minus LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0587 / -0.0836 | -0.0712 | [-0.1118, -0.0326] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0280 / -0.0467 | -0.0374 | [-0.0648, -0.0171] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1573 / +0.2321 | +0.1947 | [+0.1151, +0.2737] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.0306 / +1.7832 | +1.4069 | [+0.7767, +2.1215] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.6610 / +5.1906 | +4.4258 | [+2.6052, +6.2048] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.5148 / +9.6660 | +8.0904 | [+4.7954, +11.5638] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.8018 / +1.4437 | +1.1228 | [+0.5709, +1.7422] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.8884 / +4.5098 | +3.6991 | [+2.0289, +5.4643] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +5.3678 / +7.2889 | +6.3284 | [+3.7031, +9.3319] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.2389 / +0.8534 | +0.5461 | [+0.1987, +1.0425] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.1376 / +2.2821 | +1.7099 | [+0.9608, +2.6210] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSMT minus LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0409 / -0.0391 | -0.0400 | [-0.0686, -0.0127] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0125 / -0.0296 | -0.0210 | [-0.0399, -0.0076] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0872 / +0.1106 | +0.0989 | [+0.0593, +0.1430] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.3695 / +0.7136 | +0.5416 | [+0.2014, +0.9478] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.6511 / +2.3763 | +2.0137 | [+0.9853, +3.0785] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +3.4065 / +4.7307 | +4.0686 | [+2.1824, +6.0292] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3025 / +0.5486 | +0.4256 | [+0.1556, +0.7587] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.1005 / +2.1219 | +1.6112 | [+0.6495, +2.7441] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +2.5099 / +3.0634 | +2.7867 | [+1.3510, +4.4314] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0220 / +0.3682 | +0.1951 | [+0.0132, +0.5129] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.1371 / +0.8915 | +0.5143 | [+0.0767, +1.1245] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSMG minus LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0587 / -0.0836 | -0.0712 | [-0.1118, -0.0326] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0280 / -0.0467 | -0.0374 | [-0.0648, -0.0171] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1558 / +0.2321 | +0.1939 | [+0.1148, +0.2733] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.0306 / +1.7832 | +1.4069 | [+0.7767, +2.1215] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.6610 / +5.1906 | +4.4258 | [+2.6052, +6.2048] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.4734 / +9.6660 | +8.0697 | [+4.7704, +11.5236] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.8018 / +1.4437 | +1.1228 | [+0.5709, +1.7422] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.8884 / +4.5098 | +3.6991 | [+2.0289, +5.4643] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +5.3329 / +7.2889 | +6.3109 | [+3.6757, +9.3319] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.2389 / +0.8534 | +0.5461 | [+0.1987, +1.0425] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.1382 / +2.2821 | +1.7102 | [+0.9615, +2.6210] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSML minus LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0463 / -0.0498 | -0.0480 | [-0.0793, -0.0209] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0156 / -0.0312 | -0.0234 | [-0.0426, -0.0099] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1199 / +0.1215 | +0.1207 | [+0.0769, +0.1661] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.5429 / +0.8251 | +0.6840 | [+0.3320, +1.1051] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.3596 / +2.5089 | +2.4343 | [+1.4240, +3.5716] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +4.7446 / +4.8301 | +4.7874 | [+2.9650, +6.8093] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.4316 / +0.6374 | +0.5345 | [+0.2329, +0.8948] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.7553 / +2.1826 | +1.9690 | [+1.0704, +3.0350] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.8462 / +3.0642 | +3.4552 | [+1.8204, +5.3998] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.0414 / +0.3950 | +0.2182 | [+0.0296, +0.5279] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.2510 / +0.9993 | +0.6251 | [+0.1623, +1.2223] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSMH minus LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0961 / -0.1103 | -0.1032 | [-0.1477, -0.0632] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0312 / -0.0545 | -0.0428 | [-0.0734, -0.0187] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.2773 / +0.3645 | +0.3209 | [+0.2303, +0.4062] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +2.4780 / +3.2420 | +2.8600 | [+2.1485, +3.6475] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +7.2656 / +8.9743 | +8.1199 | [+5.9595, +10.3642] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +12.4780 / +14.3610 | +13.4195 | [+9.9681, +16.9499] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +2.0481 / +2.7256 | +2.3868 | [+1.7224, +3.0953] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +6.1373 / +7.9803 | +7.0588 | [+5.0287, +9.3364] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +10.7188 / +11.6397 | +11.1792 | [+7.9310, +14.5256] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.7832 / +1.2069 | +0.9951 | [+0.6755, +1.4008] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +2.7974 / +3.9679 | +3.3827 | [+2.4925, +4.3924] | 1.7849 (0.8089) | WORSE |

## LSDMC minus LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0765 / -0.0890 | -0.0827 | [-0.1202, -0.0483] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0171 / -0.0545 | -0.0358 | [-0.0692, -0.0095] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1682 / +0.1449 | +0.1565 | [+0.1106, +0.2085] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.3475 / +1.3145 | +1.3310 | [+0.9120, +1.8017] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.9992 / +4.0078 | +4.0035 | [+2.6427, +5.6029] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.4212 / +7.3399 | +6.8805 | [+4.6875, +9.4846] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.0872 / +1.0556 | +1.0714 | [+0.7146, +1.4976] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +3.4647 / +3.4971 | +3.4809 | [+2.2100, +4.9949] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +5.5657 / +5.0698 | +5.3178 | [+3.0329, +7.8247] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.3724 / +0.3789 | +0.3756 | [+0.2442, +0.5482] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.1555 / +1.1123 | +1.1339 | [+0.7824, +1.5397] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDMT minus LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0178 / -0.0178 | -0.0178 | [-0.0326, -0.0040] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0047 / -0.0187 | -0.0117 | [-0.0246, -0.0024] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0919 / +0.0561 | +0.0740 | [+0.0361, +0.1202] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4117 / +0.4433 | +0.4275 | [+0.1858, +0.7365] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.6064 / +1.6501 | +1.6282 | [+0.8336, +2.6051] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.9981 / +3.3782 | +3.1881 | [+1.9395, +4.6613] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3119 / +0.2991 | +0.3055 | [+0.1123, +0.5530] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.2842 / +1.5236 | +1.4039 | [+0.6690, +2.3494] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +2.4413 / +1.7454 | +2.0933 | [+0.7471, +3.5690] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0265 / +0.0447 | +0.0356 | [+0.0138, +0.0701] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.1776 / +0.2866 | +0.2321 | [+0.0829, +0.4659] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDMG minus LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0747 / -0.0819 | -0.0783 | [-0.1160, -0.0443] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0171 / -0.0545 | -0.0358 | [-0.0692, -0.0095] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1682 / +0.1433 | +0.1558 | [+0.1100, +0.2072] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.3300 / +1.3018 | +1.3159 | [+0.8980, +1.7912] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.9935 / +3.9339 | +3.9637 | [+2.6027, +5.4979] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.4252 / +7.2600 | +6.8426 | [+4.6783, +9.4551] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.0729 / +1.0429 | +1.0579 | [+0.6982, +1.4825] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +3.4624 / +3.4921 | +3.4773 | [+2.2130, +4.9820] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +5.5786 / +5.0751 | +5.3268 | [+3.0544, +7.8313] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.3709 / +0.3782 | +0.3745 | [+0.2437, +0.5464] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.1542 / +1.1077 | +1.1310 | [+0.7758, +1.5391] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDML minus LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0534 / -0.0534 | -0.0534 | [-0.0818, -0.0291] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0093 / -0.0389 | -0.0241 | [-0.0500, -0.0050] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1215 / +0.0748 | +0.0981 | [+0.0544, +0.1516] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6976 / +0.6827 | +0.6902 | [+0.3758, +1.0665] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.4192 / +2.0535 | +2.2363 | [+1.2885, +3.3306] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +4.2221 / +4.4371 | +4.3296 | [+2.8107, +6.0852] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.5147 / +0.5005 | +0.5076 | [+0.2494, +0.8285] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.0115 / +1.8134 | +1.9124 | [+1.0177, +2.9266] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.5076 / +2.5306 | +3.0191 | [+1.4028, +4.8685] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0768 / +0.1177 | +0.0972 | [+0.0491, +0.1716] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.5429 / +0.4205 | +0.4817 | [+0.2355, +0.7702] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDMH minus LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0996 / -0.1210 | -0.1103 | [-0.1599, -0.0672] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0187 / -0.0685 | -0.0436 | [-0.0870, -0.0109] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.2928 / +0.2991 | +0.2960 | [+0.2389, +0.3477] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +2.7308 / +2.7757 | +2.7533 | [+2.2086, +3.3368] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +7.2882 / +7.8908 | +7.5895 | [+5.8389, +9.4559] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +11.2218 / +11.7966 | +11.5092 | [+8.9931, +14.0364] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +2.3112 / +2.2785 | +2.2949 | [+1.7912, +2.8127] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +6.4969 / +6.8816 | +6.6892 | [+5.0554, +8.3896] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +10.2392 / +8.9338 | +9.5865 | [+6.7866, +12.2759] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.9475 / +0.8096 | +0.8786 | [+0.6495, +1.1422] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +3.0429 / +2.5699 | +2.8064 | [+2.1950, +3.5143] | 1.7849 (0.8089) | WORSE |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LS (0,1) | 0.1246 [0.0823, 0.1714] | 0.0444 [0.0193, 0.0766] | 0.2111 [0.1246, 0.3163] | -0.2350 [-0.3195, -0.1538] | 1.1867 [0.5392, 1.9640] | 5.4393 [3.1026, 8.0310] | 0.5163 [0.4689, 0.5676] | 2.7856 [2.2778, 3.4051] | 8.3298 [6.8681, 10.0377] | 1.2365 [0.7221, 1.7936] | 2.4602 [1.4184, 3.6642] |
| LSD (0,1) | 0.1343 [0.0873, 0.1879] | 0.0452 [0.0109, 0.0906] | 0.2531 [0.1958, 0.3138] | -0.2031 [-0.2748, -0.1325] | 1.4817 [0.8172, 2.2957] | 6.7598 [4.6651, 8.6599] | 0.5100 [0.4630, 0.5586] | 2.8968 [2.3956, 3.5375] | 9.3372 [8.1968, 10.6886] | 1.4640 [0.8170, 2.2047] | 2.5391 [1.6003, 3.5518] |
| LSDMC (0,1) | 0.0516 [0.0277, 0.0823] | 0.0093 [0.0000, 0.0261] | 0.4097 [0.3379, 0.4816] | 1.1278 [0.7191, 1.5873] | 5.4852 [3.9211, 7.1340] | 13.6403 [10.7057, 16.7188] | 1.5814 [1.2109, 2.0053] | 6.3777 [4.9718, 7.8509] | 14.6549 [12.0072, 17.5735] | 1.8397 [1.1472, 2.5956] | 3.6730 [2.6322, 4.7572] |
| LSDMG (0,1) | 0.0560 [0.0301, 0.0912] | 0.0093 [0.0000, 0.0261] | 0.4089 [0.3371, 0.4816] | 1.1127 [0.7039, 1.5713] | 5.4454 [3.8589, 7.0812] | 13.6024 [10.6442, 16.6801] | 1.5679 [1.1919, 1.9927] | 6.3741 [4.9779, 7.8531] | 14.6640 [12.0121, 17.5946] | 1.8386 [1.1472, 2.5930] | 3.6701 [2.6248, 4.7552] |
| LSDMH (0,1) | 0.0240 [0.0113, 0.0380] | 0.0016 [0.0000, 0.0051] | 0.5491 [0.4748, 0.6103] | 2.5501 [2.0222, 3.1061] | 9.0712 [7.3524, 10.7473] | 18.2690 [14.9898, 21.3254] | 2.8048 [2.2958, 3.3284] | 9.5860 [7.9233, 11.2412] | 18.9236 [15.7737, 21.8890] | 2.3426 [1.5567, 3.2238] | 5.3455 [4.1002, 6.7545] |
| LSDML (0,1) | 0.0810 [0.0470, 0.1248] | 0.0210 [0.0042, 0.0477] | 0.3512 [0.2702, 0.4347] | 0.4870 [0.1888, 0.8437] | 3.7180 [2.3627, 5.0573] | 11.0894 [8.5720, 13.5416] | 1.0176 [0.7520, 1.3406] | 4.8092 [3.7034, 5.9425] | 12.3563 [10.2458, 14.5408] | 1.5613 [0.9175, 2.2785] | 3.0208 [1.9841, 4.1141] |
| LSDMT (0,1) | 0.1165 [0.0746, 0.1633] | 0.0335 [0.0072, 0.0708] | 0.3271 [0.2509, 0.4032] | 0.2244 [-0.0011, 0.5060] | 3.1099 [2.0245, 4.2360] | 9.9479 [7.5280, 12.1979] | 0.8155 [0.6068, 1.0734] | 4.3006 [3.4383, 5.2380] | 11.4305 [9.5443, 13.5194] | 1.4996 [0.8583, 2.2313] | 2.7712 [1.8700, 3.7206] |
| LSMC (0,1) | 0.0534 [0.0312, 0.0772] | 0.0070 [0.0000, 0.0177] | 0.4058 [0.3276, 0.4795] | 1.1719 [0.5946, 1.8411] | 5.6125 [4.1334, 7.2170] | 13.5297 [10.7494, 16.4204] | 1.6390 [1.0908, 2.2664] | 6.4847 [5.0902, 8.0342] | 14.6582 [12.0491, 17.4352] | 1.7826 [1.3123, 2.2538] | 4.1701 [3.3443, 5.0752] |
| LSMG (0,1) | 0.0534 [0.0312, 0.0772] | 0.0070 [0.0000, 0.0177] | 0.4050 [0.3275, 0.4776] | 1.1719 [0.5946, 1.8411] | 5.6125 [4.1334, 7.2170] | 13.5090 [10.7492, 16.4188] | 1.6390 [1.0908, 2.2664] | 6.4847 [5.0902, 8.0342] | 14.6407 [12.0428, 17.4009] | 1.7826 [1.3123, 2.2538] | 4.1704 [3.3447, 5.0753] |
| LSMH (0,1) | 0.0214 [0.0121, 0.0331] | 0.0016 [0.0000, 0.0053] | 0.5319 [0.4595, 0.5982] | 2.6250 [1.9439, 3.3623] | 9.3066 [7.4985, 11.3251] | 18.8588 [15.4101, 22.0929] | 2.9031 [2.2414, 3.6215] | 9.8444 [8.0697, 11.8615] | 19.5090 [16.2516, 22.6837] | 2.2315 [1.7206, 2.7620] | 5.8429 [4.9456, 6.7637] |
| LSML (0,1) | 0.0765 [0.0471, 0.1121] | 0.0210 [0.0074, 0.0395] | 0.3318 [0.2356, 0.4329] | 0.4491 [0.1264, 0.8341] | 3.6209 [2.4838, 4.8864] | 10.2267 [7.5575, 13.1692] | 1.0507 [0.7526, 1.4125] | 4.7545 [3.8012, 5.8417] | 11.7850 [9.4929, 14.3969] | 1.4547 [1.0080, 1.9167] | 3.0853 [2.2624, 3.9878] |
| LSMT (0,1) | 0.0845 [0.0527, 0.1232] | 0.0234 [0.0086, 0.0426] | 0.3100 [0.2259, 0.4009] | 0.3066 [0.0047, 0.6708] | 3.2004 [2.2391, 4.2372] | 9.5079 [7.0923, 11.8941] | 0.9418 [0.6696, 1.2884] | 4.3968 [3.5541, 5.3752] | 11.1165 [9.1719, 13.2105] | 1.4316 [0.9801, 1.9035] | 2.9745 [2.1468, 3.8918] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
