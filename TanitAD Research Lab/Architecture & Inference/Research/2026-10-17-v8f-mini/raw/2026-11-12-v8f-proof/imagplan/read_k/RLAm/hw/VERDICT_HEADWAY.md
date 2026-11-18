# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 18, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPlx_RLAm:IPox_RLAm", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLAm:IPox_RLAm", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLAm:IPcx_RLAm", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLAm:IPlx_RLAm", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcx_RLAm:IPox_RLAm", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPksx_RLAm:IPkx_RLAm", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLAm:IPix_RLAm", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPix_RLAm:IPox_RLAm", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, IPlx_RLAm minus IPox_RLAm over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLAm minus IPox_RLAm over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLAm minus IPcx_RLAm over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLAm minus IPlx_RLAm over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcx_RLAm minus IPox_RLAm over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPksx_RLAm minus IPkx_RLAm over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLAm minus IPix_RLAm over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPix_RLAm minus IPox_RLAm over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## IPlx RLAm minus IPox RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0890 / -0.0996 | -0.0943 | [-0.1307, -0.0612] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0467 / -0.0358 | -0.0413 | [-0.0678, -0.0194] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0950 / +0.0779 | +0.0864 | [+0.0359, +0.1364] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.8836 / +0.8864 | +0.8850 | [+0.3617, +1.5550] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.4885 / +3.4990 | +3.4937 | [+1.4719, +6.0818] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.2675 / +5.5460 | +5.9068 | [+2.4203, +10.3126] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.7618 / +0.7646 | +0.7632 | [+0.2643, +1.3910] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.7349 / +2.3751 | +2.5550 | [+0.6967, +4.9669] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +4.6477 / +3.9711 | +4.3094 | [+1.1022, +8.4136] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.3105 / +0.2775 | +0.2940 | [+0.1265, +0.5188] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.3058 / +0.7856 | +1.0457 | [+0.3362, +2.1925] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPkx RLAm minus IPox RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0249 / -0.0605 | -0.0427 | [-0.0828, -0.0143] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0171 / -0.0062 | -0.0117 | [-0.0231, -0.0032] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0903 / +0.1090 | +0.0997 | [+0.0481, +0.1635] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7526 / +0.9954 | +0.8740 | [+0.3514, +1.5178] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.8397 / +4.0483 | +3.4440 | [+1.3351, +6.0609] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +4.8492 / +6.8658 | +5.8575 | [+2.3219, +10.6112] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.6831 / +0.8976 | +0.7903 | [+0.3079, +1.3951] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.5922 / +3.1876 | +2.8899 | [+1.1213, +5.0678] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +4.0697 / +5.8419 | +4.9558 | [+1.5359, +9.2239] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.1707 / +0.2482 | +0.2095 | [+0.0919, +0.3451] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.9708 / +1.1822 | +1.0765 | [+0.3512, +1.8745] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPkx RLAm minus IPcx RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0178 / +0.0107 | +0.0142 | [+0.0013, +0.0263] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0062 / +0.0078 | +0.0070 | [+0.0008, +0.0150] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0031 / -0.0156 | -0.0062 | [-0.0274, +0.0159] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0463 / -0.0692 | -0.0114 | [-0.1297, +0.0862] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.1562 / -0.1646 | -0.0042 | [-0.4170, +0.3615] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.1744 / -0.5443 | -0.3594 | [-1.1505, +0.3967] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0721 / -0.0658 | +0.0031 | [-0.1230, +0.1103] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.2605 / -0.1311 | +0.0647 | [-0.3593, +0.4098] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.1738 / -0.4148 | -0.1205 | [-0.9752, +0.6150] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.1757 / -0.2968 | -0.2362 | [-0.4660, -0.0739] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.1714 / -0.3571 | -0.2642 | [-0.5737, -0.0236] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPkx RLAm minus IPlx RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0641 / +0.0391 | +0.0516 | [+0.0256, +0.0822] | 0.0463 (0.0214) | WORSE |
| C contact rate | +0.0296 / +0.0296 | +0.0296 | [+0.0128, +0.0497] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0047 / +0.0312 | +0.0132 | [-0.0247, +0.0489] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1310 / +0.1090 | -0.0110 | [-0.2086, +0.2835] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.6488 / +0.5493 | -0.0498 | [-0.9393, +1.2212] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -1.4183 / +1.3198 | -0.0492 | [-2.0009, +2.6859] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0788 / +0.1330 | +0.0271 | [-0.1592, +0.3049] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.1427 / +0.8125 | +0.3349 | [-0.4462, +1.5180] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.5779 / +1.8708 | +0.6464 | [-1.1055, +3.1615] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.1398 / -0.0293 | -0.0845 | [-0.2842, +0.0855] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.3350 / +0.3966 | +0.0308 | [-0.5532, +0.7272] | 1.7849 (0.8089) | OPEN |

## IPcx RLAm minus IPox RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0427 / -0.0712 | -0.0569 | [-0.0940, -0.0274] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0234 / -0.0140 | -0.0187 | [-0.0317, -0.0081] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0872 / +0.1246 | +0.1059 | [+0.0539, +0.1702] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7063 / +1.0646 | +0.8854 | [+0.3438, +1.5475] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.6835 / +4.2128 | +3.4482 | [+1.3399, +6.1049] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +5.0237 / +7.4102 | +6.2169 | [+2.7107, +11.0232] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.6110 / +0.9634 | +0.7872 | [+0.2809, +1.4126] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.3317 / +3.3188 | +2.8252 | [+1.0155, +5.0794] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.8960 / +6.2567 | +5.0764 | [+1.6925, +9.4821] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.3464 / +0.5450 | +0.4457 | [+0.2091, +0.7346] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.1422 / +1.5393 | +1.3408 | [+0.6275, +2.1210] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPksx RLAm minus IPkx RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0801 / -0.0907 | -0.0854 | [-0.1224, -0.0528] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0296 / -0.0280 | -0.0288 | [-0.0488, -0.0115] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.4579 / +0.4393 | +0.4486 | [+0.3846, +0.5077] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +5.2501 / +5.4015 | +5.3258 | [+4.6118, +6.0210] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +14.3574 / +15.1544 | +14.7559 | [+12.5580, +16.9644] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +22.0010 / +23.3705 | +22.6858 | [+18.9327, +26.5682] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +4.7986 / +4.9708 | +4.8847 | [+4.1917, +5.5654] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +13.3586 / +14.0088 | +13.6837 | [+11.5551, +15.8179] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +20.3844 / +21.7131 | +21.0488 | [+17.3293, +24.9775] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +20.0749 / +18.2665 | +19.1707 | [+15.4901, +22.9270] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +21.5635 / +21.4961 | +21.5298 | [+18.4834, +24.6364] | 1.7849 (0.8089) | WORSE |

## IPkx RLAm minus IPix RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0036 / +0.0000 | +0.0018 | [-0.0025, +0.0062] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0016 / +0.0016 | +0.0016 | [+0.0000, +0.0041] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0031 / -0.0031 | -0.0031 | [-0.0083, +0.0024] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0140 / -0.0043 | +0.0049 | [-0.0304, +0.0415] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.0318 / -0.0047 | +0.0136 | [-0.1153, +0.1559] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.1057 / +0.0198 | -0.0429 | [-0.2778, +0.2873] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0140 / -0.0043 | +0.0049 | [-0.0304, +0.0415] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0318 / -0.0016 | +0.0151 | [-0.1096, +0.1550] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.0411 / +0.0212 | -0.0100 | [-0.2456, +0.2994] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0067 / -0.0325 | -0.0196 | [-0.0548, +0.0024] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0356 / -0.1183 | -0.0769 | [-0.1743, -0.0097] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPix RLAm minus IPox RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0285 / -0.0605 | -0.0445 | [-0.0813, -0.0165] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0187 / -0.0078 | -0.0132 | [-0.0252, -0.0037] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0935 / +0.1121 | +0.1028 | [+0.0500, +0.1661] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7386 / +0.9996 | +0.8691 | [+0.3573, +1.5081] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.8078 / +4.0530 | +3.4304 | [+1.3660, +6.0538] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +4.9550 / +6.8460 | +5.9005 | [+2.4657, +10.5513] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.6691 / +0.9019 | +0.7855 | [+0.3133, +1.3789] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.5604 / +3.1892 | +2.8748 | [+1.1462, +5.0178] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +4.1108 / +5.8207 | +4.9658 | [+1.6481, +9.2220] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.1774 / +0.2807 | +0.2290 | [+0.1019, +0.3808] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.0064 / +1.3005 | +1.1535 | [+0.4108, +1.9643] | 1.7849 (0.8089) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| IPcx_RLAm (0,1) | 0.0747 [0.0436, 0.1115] | 0.0226 [0.0076, 0.0406] | 0.2874 [0.2251, 0.3505] | 0.6636 [0.1228, 1.3352] | 4.1846 [2.3303, 6.5665] | 11.1824 [7.7597, 15.4987] | 1.2759 [0.7782, 1.8924] | 5.4838 [3.6453, 7.8141] | 12.7629 [9.5392, 16.9530] | 1.5347 [1.0849, 2.0300] | 3.3574 [2.4873, 4.2625] |
| IPix_RLAm (0,1) | 0.0872 [0.0533, 0.1237] | 0.0280 [0.0118, 0.0474] | 0.2843 [0.2243, 0.3453] | 0.6473 [0.1367, 1.2834] | 4.1668 [2.3094, 6.4503] | 10.8660 [7.4086, 15.1870] | 1.2742 [0.8067, 1.8587] | 5.5333 [3.7504, 7.7475] | 12.6523 [9.3874, 16.7873] | 1.3181 [0.9419, 1.7461] | 3.1701 [2.3302, 4.0852] |
| IPksx_RLAm (0,1) | 0.0036 [0.0000, 0.0107] | 0.0008 [0.0000, 0.0033] | 0.7298 [0.6684, 0.7799] | 5.9780 [5.1123, 6.7988] | 18.9363 [16.2013, 21.8347] | 33.5088 [28.3673, 39.0245] | 6.1637 [5.3499, 6.9495] | 19.2322 [16.5784, 22.0787] | 33.6911 [28.6456, 39.2336] | 20.4692 [16.7787, 24.1949] | 24.6229 [21.5594, 27.6636] |
| IPkx_RLAm (0,1) | 0.0890 [0.0548, 0.1258] | 0.0296 [0.0128, 0.0497] | 0.2812 [0.2207, 0.3417] | 0.6522 [0.1304, 1.2941] | 4.1804 [2.3212, 6.4983] | 10.8230 [7.2931, 15.2669] | 1.2790 [0.7971, 1.8645] | 5.5485 [3.7238, 7.7843] | 12.6423 [9.3529, 16.8415] | 1.2985 [0.9297, 1.7179] | 3.0931 [2.2574, 4.0042] |
| IPlx_RLAm (0,1) | 0.0374 [0.0137, 0.0723] | 0.0000 [0.0000, 0.0000] | 0.2679 [0.2076, 0.3312] | 0.6632 [0.1405, 1.3390] | 4.2301 [2.2104, 6.7198] | 10.8723 [7.1930, 15.2635] | 1.2519 [0.7576, 1.8853] | 5.2136 [3.3490, 7.6136] | 11.9959 [8.5764, 16.1214] | 1.3830 [1.0016, 1.8203] | 3.0623 [2.2093, 4.2410] |
| IPox_RLAm (0,1) | 0.1317 [0.0831, 0.1892] | 0.0413 [0.0194, 0.0678] | 0.1815 [0.1372, 0.2287] | -0.2218 [-0.2701, -0.1740] | 0.7364 [0.0292, 1.3799] | 4.9655 [3.4659, 6.5884] | 0.4887 [0.4391, 0.5413] | 2.6585 [2.3764, 2.9619] | 7.6865 [6.6276, 8.9726] | 1.0890 [0.7424, 1.4765] | 2.0166 [1.4838, 2.6347] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
