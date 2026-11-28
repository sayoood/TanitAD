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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQC:LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQC:LSRH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQC:LSRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQC:LSMC", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDQC:LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDQC:LSDRH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDQC:LSDRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDQC:LSDMC", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, LSQC minus LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSQC minus LSRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSQC minus LSRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSQC minus LSMC over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDQC minus LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDQC minus LSDRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDQC minus LSDRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDQC minus LSDMC over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## LSQC minus LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0587 / -0.0605 | -0.0596 | [-0.0942, -0.0272] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0156 / -0.0343 | -0.0249 | [-0.0466, -0.0083] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1589 / +0.1745 | +0.1667 | [+0.1137, +0.2222] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.9025 / +1.2399 | +1.0712 | [+0.6314, +1.5848] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.3007 / +3.9329 | +3.6168 | [+2.2835, +5.1769] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.6327 / +7.9253 | +7.2790 | [+4.7442, +10.0376] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.6957 / +0.9533 | +0.8245 | [+0.4364, +1.2709] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.6164 / +3.5265 | +3.0714 | [+1.8319, +4.5685] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +5.6212 / +5.8979 | +5.7596 | [+3.4895, +8.2391] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.0926 / +0.4928 | +0.2927 | [+0.0744, +0.6436] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.5736 / +1.3257 | +0.9496 | [+0.4556, +1.5993] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSQC minus LSRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0178 / -0.0214 | -0.0196 | [-0.0372, -0.0042] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0031 / -0.0047 | -0.0039 | [-0.0097, +0.0016] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0717 / +0.0639 | +0.0678 | [+0.0402, +0.0978] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.5329 / +0.5262 | +0.5296 | [+0.3705, +0.6999] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.6496 / +1.5566 | +1.6031 | [+1.0277, +2.1986] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +3.2262 / +3.1945 | +3.2104 | [+2.0781, +4.4443] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3932 / +0.4046 | +0.3989 | [+0.2476, +0.5622] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.5159 / +1.4046 | +1.4602 | [+0.9129, +2.0388] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.1113 / +2.8346 | +2.9729 | [+1.8069, +4.1700] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0705 / +0.1246 | +0.0975 | [+0.0551, +0.1704] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.4365 / +0.4341 | +0.4353 | [+0.2784, +0.5919] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSQC minus LSRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0160 / +0.0231 | +0.0196 | [+0.0012, +0.0391] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0156 / +0.0234 | +0.0195 | [+0.0078, +0.0364] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0857 / +0.1215 | +0.1036 | [+0.0594, +0.1472] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4666 / +0.7931 | +0.6299 | [+0.3692, +0.9818] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.9519 / +2.6054 | +2.2787 | [+1.4312, +3.3531] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +3.5462 / +4.4902 | +4.0182 | [+2.4646, +5.8650] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3695 / +0.6598 | +0.5147 | [+0.2651, +0.8237] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.9065 / +2.8092 | +2.3579 | [+1.4552, +3.5050] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.7119 / +4.8625 | +4.2872 | [+2.7161, +6.1422] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.0514 / +0.4531 | +0.2522 | [+0.0357, +0.6071] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.4355 / +1.2354 | +0.8355 | [+0.3268, +1.5085] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSQC minus LSMC

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0231 | +0.0116 | [-0.0078, +0.0337] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0125 / +0.0125 | +0.0125 | [+0.0049, +0.0213] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0016 / -0.0576 | -0.0280 | [-0.0739, +0.0142] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1281 / -0.5433 | -0.3357 | [-0.6403, -0.0938] | 0.0689 (0.0315) | BETTER |
| gap bias 4.0 s (m) | -0.3602 / -1.2577 | -0.8090 | [-1.6413, -0.1107] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.1179 / -1.7407 | -0.8114 | [-2.5437, +0.4623] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.1061 / -0.4905 | -0.2983 | [-0.5816, -0.0756] | 0.0426 (0.0189) | BETTER |
| gap error 4.0 s (m) | -0.2720 / -0.9833 | -0.6277 | [-1.3647, -0.0171] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.2534 / -1.3909 | -0.5688 | [-2.2081, +0.6463] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.1463 / -0.3606 | -0.2534 | [-0.4322, -0.1125] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.5640 / -0.9565 | -0.7603 | [-1.1610, -0.4217] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDQC minus LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0587 / -0.0783 | -0.0685 | [-0.1071, -0.0355] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0078 / -0.0467 | -0.0273 | [-0.0588, -0.0045] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1838 / +0.1308 | +0.1573 | [+0.1036, +0.2188] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.1480 / +1.1177 | +1.1328 | [+0.7205, +1.5882] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.7792 / +3.7935 | +3.7863 | [+2.4387, +5.2668] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.0942 / +6.9371 | +6.5157 | [+4.3578, +9.0470] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.8861 / +0.8666 | +0.8763 | [+0.5211, +1.2808] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +3.2725 / +3.3566 | +3.3146 | [+2.0404, +4.7545] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +5.3446 / +4.8597 | +5.1021 | [+2.8196, +7.5165] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.1592 / +0.1976 | +0.1784 | [+0.1083, +0.2806] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.8557 / +0.7287 | +0.7922 | [+0.4881, +1.1533] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDQC minus LSDRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0391 / -0.0641 | -0.0516 | [-0.0846, -0.0245] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0031 / -0.0280 | -0.0156 | [-0.0379, -0.0013] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0935 / +0.0763 | +0.0849 | [+0.0565, +0.1150] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7406 / +0.6809 | +0.7108 | [+0.4978, +0.9306] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.1763 / +2.1837 | +2.1800 | [+1.4813, +2.9096] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +3.1110 / +3.5619 | +3.3364 | [+2.1767, +4.6403] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.5775 / +0.5741 | +0.5758 | [+0.3802, +0.7902] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.9949 / +1.8561 | +1.9255 | [+1.2499, +2.6162] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +2.9556 / +3.1173 | +3.0364 | [+1.9209, +4.2478] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.1330 / +0.1534 | +0.1432 | [+0.0858, +0.2331] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.6790 / +0.4446 | +0.5618 | [+0.3414, +0.8741] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDQC minus LSDRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0302 / +0.0071 | +0.0187 | [-0.0010, +0.0423] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0109 / +0.0249 | +0.0179 | [+0.0047, +0.0379] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1044 / +0.0779 | +0.0911 | [+0.0572, +0.1287] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7114 / +0.6751 | +0.6933 | [+0.4649, +0.9572] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.4267 / +2.3741 | +2.4004 | [+1.5592, +3.3809] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +3.3956 / +3.3347 | +3.3651 | [+1.9808, +4.8325] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.5714 / +0.5919 | +0.5816 | [+0.3814, +0.8251] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.4888 / +2.5185 | +2.5037 | [+1.6837, +3.4573] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.5102 / +3.7485 | +3.6294 | [+2.2434, +5.1072] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.1151 / +0.1581 | +0.1366 | [+0.0761, +0.2273] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.6834 / +0.6519 | +0.6676 | [+0.3998, +0.9803] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDQC minus LSDMC

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0178 / +0.0107 | +0.0142 | [-0.0018, +0.0305] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0093 / +0.0078 | +0.0086 | [+0.0018, +0.0174] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0156 / -0.0140 | +0.0008 | [-0.0239, +0.0272] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1995 / -0.1968 | -0.1981 | [-0.2963, -0.1001] | 0.0689 (0.0315) | BETTER |
| gap bias 4.0 s (m) | -0.2200 / -0.2143 | -0.2172 | [-0.6325, +0.2218] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.3269 / -0.4028 | -0.3649 | [-1.1997, +0.4563] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.2011 / -0.1890 | -0.1951 | [-0.2800, -0.1084] | 0.0426 (0.0189) | BETTER |
| gap error 4.0 s (m) | -0.1922 / -0.1406 | -0.1664 | [-0.5333, +0.2358] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.2212 / -0.2101 | -0.2156 | [-0.9537, +0.5498] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.2132 / -0.1813 | -0.1972 | [-0.2973, -0.1145] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.2997 / -0.3836 | -0.3417 | [-0.5690, -0.1179] | 1.7849 (0.8089) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LS (0,1) | 0.1246 [0.0823, 0.1714] | 0.0444 [0.0193, 0.0766] | 0.2111 [0.1246, 0.3163] | -0.2350 [-0.3195, -0.1538] | 1.1867 [0.5392, 1.9640] | 5.4393 [3.1026, 8.0310] | 0.5163 [0.4689, 0.5676] | 2.7856 [2.2778, 3.4051] | 8.3298 [6.8681, 10.0377] | 1.2365 [0.7221, 1.7936] | 2.4602 [1.4184, 3.6642] |
| LSD (0,1) | 0.1343 [0.0873, 0.1879] | 0.0452 [0.0109, 0.0906] | 0.2531 [0.1958, 0.3138] | -0.2031 [-0.2748, -0.1325] | 1.4817 [0.8172, 2.2957] | 6.7598 [4.6651, 8.6599] | 0.5100 [0.4630, 0.5586] | 2.8968 [2.3956, 3.5375] | 9.3372 [8.1968, 10.6886] | 1.4640 [0.8170, 2.2047] | 2.5391 [1.6003, 3.5518] |
| LSDMC (0,1) | 0.0516 [0.0277, 0.0823] | 0.0093 [0.0000, 0.0261] | 0.4097 [0.3379, 0.4816] | 1.1278 [0.7191, 1.5873] | 5.4852 [3.9211, 7.1340] | 13.6403 [10.7057, 16.7188] | 1.5814 [1.2109, 2.0053] | 6.3777 [4.9718, 7.8509] | 14.6549 [12.0072, 17.5735] | 1.8397 [1.1472, 2.5956] | 3.6730 [2.6322, 4.7572] |
| LSDQC (0,1) | 0.0658 [0.0389, 0.0988] | 0.0179 [0.0047, 0.0379] | 0.4104 [0.3222, 0.4975] | 0.9297 [0.5356, 1.3644] | 5.2680 [3.7314, 6.8447] | 13.2754 [10.2952, 16.3070] | 1.3863 [1.0227, 1.7990] | 6.2113 [4.8507, 7.6690] | 14.4393 [11.7734, 17.3270] | 1.6424 [0.9985, 2.3604] | 3.3313 [2.3127, 4.4132] |
| LSDRH (0,1) | 0.1174 [0.0734, 0.1670] | 0.0335 [0.0072, 0.0708] | 0.3255 [0.2493, 0.4022] | 0.2189 [-0.0074, 0.5021] | 3.0880 [1.9867, 4.2275] | 9.9390 [7.5229, 12.1905] | 0.8106 [0.6022, 1.0655] | 4.2859 [3.4176, 5.2197] | 11.4028 [9.5253, 13.4774] | 1.4992 [0.8581, 2.2308] | 2.7695 [1.8682, 3.7196] |
| LSDRL (0,1) | 0.0472 [0.0227, 0.0829] | 0.0000 [0.0000, 0.0000] | 0.3193 [0.2469, 0.3910] | 0.2364 [-0.0076, 0.5420] | 2.8677 [1.8890, 3.9087] | 9.9103 [7.7746, 11.9708] | 0.8047 [0.5833, 1.0843] | 3.7077 [2.8459, 4.6228] | 10.8099 [8.9499, 12.7615] | 1.5058 [0.8571, 2.2451] | 2.6637 [1.6935, 3.7219] |
| LSMC (0,1) | 0.0534 [0.0312, 0.0772] | 0.0070 [0.0000, 0.0177] | 0.4058 [0.3276, 0.4795] | 1.1719 [0.5946, 1.8411] | 5.6125 [4.1334, 7.2170] | 13.5297 [10.7494, 16.4204] | 1.6390 [1.0908, 2.2664] | 6.4847 [5.0902, 8.0342] | 14.6582 [12.0491, 17.4352] | 1.7826 [1.3123, 2.2538] | 4.1701 [3.3443, 5.0752] |
| LSQC (0,1) | 0.0649 [0.0384, 0.0985] | 0.0195 [0.0078, 0.0364] | 0.3777 [0.2840, 0.4724] | 0.8362 [0.4309, 1.3170] | 4.8035 [3.4585, 6.2442] | 12.7183 [9.6831, 15.6759] | 1.3408 [0.9562, 1.7937] | 5.8570 [4.6629, 7.2501] | 14.0894 [11.3615, 16.9036] | 1.5291 [1.0824, 1.9879] | 3.4098 [2.5781, 4.2949] |
| LSRH (0,1) | 0.0845 [0.0527, 0.1232] | 0.0234 [0.0086, 0.0426] | 0.3100 [0.2259, 0.4009] | 0.3066 [0.0047, 0.6708] | 3.2004 [2.2391, 4.2372] | 9.5079 [7.0923, 11.8941] | 0.9418 [0.6696, 1.2884] | 4.3968 [3.5541, 5.3752] | 11.1165 [9.1719, 13.2105] | 1.4316 [0.9801, 1.9035] | 2.9745 [2.1468, 3.8918] |
| LSRL (0,1) | 0.0454 [0.0238, 0.0728] | 0.0000 [0.0000, 0.0000] | 0.2741 [0.1700, 0.3895] | 0.2064 [-0.0545, 0.5267] | 2.5248 [1.6354, 3.5112] | 8.7001 [6.1813, 11.2690] | 0.8261 [0.5991, 1.1154] | 3.4991 [2.7134, 4.3601] | 9.8022 [7.5079, 12.2263] | 1.2769 [0.7638, 1.8367] | 2.5743 [1.5172, 3.8082] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
