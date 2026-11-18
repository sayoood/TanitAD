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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPlx_RLFT:IPox_RLFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLFT:IPox_RLFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLFT:IPcx_RLFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLFT:IPlx_RLFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcx_RLFT:IPox_RLFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPksx_RLFT:IPkx_RLFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLFT:IPix_RLFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPix_RLFT:IPox_RLFT", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, IPlx_RLFT minus IPox_RLFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLFT minus IPox_RLFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLFT minus IPcx_RLFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLFT minus IPlx_RLFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcx_RLFT minus IPox_RLFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPksx_RLFT minus IPkx_RLFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLFT minus IPix_RLFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPix_RLFT minus IPox_RLFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## IPlx RLFT minus IPox RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0694 / -0.1068 | -0.0881 | [-0.1350, -0.0517] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0421 / -0.0421 | -0.0421 | [-0.0660, -0.0206] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0919 / +0.0997 | +0.0958 | [+0.0521, +0.1476] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.8777 / +0.8873 | +0.8825 | [+0.3597, +1.5585] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.4035 / +3.5187 | +3.4611 | [+1.4460, +6.0877] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +5.7205 / +5.9319 | +5.8262 | [+2.5239, +10.3388] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.7531 / +0.7649 | +0.7590 | [+0.2641, +1.3908] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.7589 / +2.3898 | +2.5744 | [+0.7291, +4.9637] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +4.4360 / +4.4319 | +4.4339 | [+1.3711, +8.6026] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.3193 / +0.2827 | +0.3010 | [+0.1262, +0.5316] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.0867 / +0.8058 | +0.9462 | [+0.3441, +1.7911] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPkx RLFT minus IPox RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0231 / -0.0623 | -0.0427 | [-0.0823, -0.0132] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0125 / -0.0109 | -0.0117 | [-0.0207, -0.0043] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0841 / +0.1184 | +0.1012 | [+0.0495, +0.1625] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7402 / +0.9881 | +0.8641 | [+0.3540, +1.4996] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.7394 / +3.9265 | +3.3330 | [+1.3089, +5.9093] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +4.4964 / +6.8334 | +5.6649 | [+2.2533, +10.4846] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.6640 / +0.9020 | +0.7830 | [+0.3087, +1.3731] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.4898 / +3.1203 | +2.8051 | [+1.0701, +4.9448] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.9481 / +5.8209 | +4.8845 | [+1.6645, +9.0404] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.2004 / +0.2442 | +0.2223 | [+0.1002, +0.3597] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.8252 / +1.1935 | +1.0093 | [+0.3772, +1.7214] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPkx RLFT minus IPcx RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0053 / +0.0125 | +0.0089 | [-0.0008, +0.0229] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0078 / +0.0062 | +0.0070 | [+0.0008, +0.0145] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0016 / -0.0187 | -0.0086 | [-0.0287, +0.0098] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0237 / -0.0583 | -0.0173 | [-0.1106, +0.0705] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.0547 / -0.2661 | -0.1057 | [-0.4763, +0.1890] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.2090 / -0.7946 | -0.5018 | [-1.2130, +0.0786] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0504 / -0.0476 | +0.0014 | [-0.0951, +0.0908] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.1166 / -0.1565 | -0.0199 | [-0.3534, +0.2454] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.0460 / -0.5978 | -0.2759 | [-1.0008, +0.2910] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.1318 / -0.2702 | -0.2010 | [-0.4204, -0.0535] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.1881 / -0.3045 | -0.2463 | [-0.5135, -0.0302] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPkx RLFT minus IPlx RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0463 / +0.0445 | +0.0454 | [+0.0242, +0.0670] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0296 / +0.0312 | +0.0304 | [+0.0133, +0.0496] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0078 / +0.0187 | +0.0055 | [-0.0249, +0.0368] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1375 / +0.1008 | -0.0184 | [-0.2249, +0.2675] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.6641 / +0.4078 | -0.1282 | [-0.9760, +1.0767] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -1.2241 / +0.9015 | -0.1613 | [-1.8227, +2.2107] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0891 / +0.1371 | +0.0240 | [-0.1753, +0.3033] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.2691 / +0.7305 | +0.2307 | [-0.5794, +1.3958] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.4880 / +1.3891 | +0.4506 | [-1.0494, +2.5784] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.1189 / -0.0386 | -0.0787 | [-0.2947, +0.1069] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.2615 / +0.3877 | +0.0631 | [-0.4979, +0.7618] | 1.7849 (0.8089) | OPEN |

## IPcx RLFT minus IPox RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0285 / -0.0747 | -0.0516 | [-0.0959, -0.0165] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0202 / -0.0171 | -0.0187 | [-0.0305, -0.0090] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0826 / +0.1371 | +0.1098 | [+0.0540, +0.1772] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7165 / +1.0464 | +0.8814 | [+0.3550, +1.5375] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.6847 / +4.1925 | +3.4386 | [+1.3436, +6.1067] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +4.7054 / +7.6280 | +6.1667 | [+2.6489, +11.1296] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.6136 / +0.9496 | +0.7816 | [+0.2824, +1.4022] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.3732 / +3.2768 | +2.8250 | [+1.0119, +5.0642] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.9021 / +6.4187 | +5.1604 | [+1.8441, +9.5575] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.3322 / +0.5144 | +0.4233 | [+0.1897, +0.6991] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.0132 / +1.4980 | +1.2556 | [+0.6127, +1.9755] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPksx RLFT minus IPkx RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0730 / -0.0854 | -0.0792 | [-0.1150, -0.0479] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0296 / -0.0312 | -0.0304 | [-0.0496, -0.0133] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.4626 / +0.4486 | +0.4556 | [+0.3926, +0.5135] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +5.2075 / +5.4799 | +5.3437 | [+4.6407, +6.0403] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +14.4719 / +15.3599 | +14.9159 | [+12.7738, +17.0904] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +22.5530 / +24.1429 | +23.3479 | [+19.5373, +27.2731] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +4.7732 / +5.0437 | +4.9084 | [+4.2208, +5.5910] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +13.4949 / +14.1288 | +13.8118 | [+11.7667, +15.9123] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +20.9163 / +22.2433 | +21.5798 | [+17.7858, +25.5167] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +19.6705 / +18.8409 | +19.2557 | [+15.9036, +22.7904] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +22.3326 / +21.6438 | +21.9882 | [+18.9437, +25.1489] | 1.7849 (0.8089) | WORSE |

## IPkx RLFT minus IPix RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0071 | +0.0036 | [+0.0000, +0.0116] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0047 / +0.0047 | +0.0047 | [+0.0000, +0.0111] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0140 / -0.0047 | -0.0093 | [-0.0178, -0.0022] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.2720 / -0.2640 | -0.2680 | [-0.4086, -0.1504] | 0.0689 (0.0315) | BETTER |
| gap bias 4.0 s (m) | -1.0547 / -0.9921 | -1.0234 | [-1.5737, -0.5831] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -1.7048 / -1.7383 | -1.7216 | [-2.7466, -0.8871] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.2675 / -0.2640 | -0.2658 | [-0.4048, -0.1485] | 0.0426 (0.0189) | BETTER |
| gap error 4.0 s (m) | -1.0076 / -0.9651 | -0.9864 | [-1.5326, -0.5445] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | -1.5908 / -1.6676 | -1.6292 | [-2.6644, -0.7887] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.3014 / -0.3188 | -0.3101 | [-0.5764, -0.1058] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.7456 / -0.6782 | -0.7119 | [-1.2047, -0.3301] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPix RLFT minus IPox RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0231 / -0.0694 | -0.0463 | [-0.0892, -0.0135] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0171 / -0.0156 | -0.0164 | [-0.0314, -0.0045] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0981 / +0.1231 | +0.1106 | [+0.0605, +0.1697] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.0122 / +1.2521 | +1.1322 | [+0.5678, +1.8126] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.7942 / +4.9185 | +4.3564 | [+2.1788, +7.1997] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.2012 / +8.5717 | +7.3865 | [+3.5408, +12.5691] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.9315 / +1.1660 | +1.0488 | [+0.5218, +1.7005] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +3.4975 / +4.0854 | +3.7914 | [+1.8494, +6.2022] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +5.5388 / +7.4885 | +6.5137 | [+2.8699, +11.1061] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.5018 / +0.5630 | +0.5324 | [+0.2728, +0.8437] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.5708 / +1.8717 | +1.7212 | [+0.9188, +2.6223] | 1.7849 (0.8089) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| IPcx_RLFT (0,1) | 0.0730 [0.0423, 0.1079] | 0.0234 [0.0079, 0.0412] | 0.2928 [0.2287, 0.3555] | 0.6693 [0.1476, 1.3225] | 4.1947 [2.3313, 6.5850] | 11.0846 [7.7242, 15.4944] | 1.2653 [0.7738, 1.8846] | 5.4832 [3.6221, 7.8247] | 12.7389 [9.5319, 16.9624] | 1.4788 [1.0188, 1.9716] | 3.3044 [2.3931, 4.3012] |
| IPix_RLFT (0,1) | 0.0783 [0.0482, 0.1104] | 0.0257 [0.0101, 0.0426] | 0.2936 [0.2320, 0.3546] | 0.9201 [0.3513, 1.6001] | 5.1125 [3.0767, 7.5526] | 12.3044 [8.5146, 17.0955] | 1.5325 [1.0034, 2.1736] | 6.4496 [4.4226, 8.9650] | 14.0922 [10.4770, 18.6212] | 1.5879 [1.1626, 2.0576] | 3.7700 [2.8027, 4.8766] |
| IPksx_RLFT (0,1) | 0.0027 [0.0000, 0.0073] | 0.0000 [0.0000, 0.0000] | 0.7399 [0.6803, 0.7888] | 5.9958 [5.1372, 6.8429] | 19.0050 [16.3058, 21.8124] | 33.9307 [28.8605, 39.4405] | 6.1752 [5.3547, 6.9960] | 19.2751 [16.6103, 22.0766] | 34.0428 [29.0322, 39.5625] | 20.5335 [17.2190, 24.0212] | 25.0463 [22.0467, 28.0178] |
| IPkx_RLFT (0,1) | 0.0819 [0.0495, 0.1171] | 0.0304 [0.0133, 0.0496] | 0.2843 [0.2237, 0.3456] | 0.6521 [0.1492, 1.2777] | 4.0891 [2.2634, 6.3437] | 10.5828 [7.2580, 14.9629] | 1.2667 [0.7974, 1.8475] | 5.4633 [3.6421, 7.7213] | 12.4630 [9.2878, 16.6017] | 1.2778 [0.9077, 1.6891] | 3.0581 [2.1896, 3.9990] |
| IPlx_RLFT (0,1) | 0.0365 [0.0184, 0.0599] | 0.0000 [0.0000, 0.0000] | 0.2788 [0.2192, 0.3428] | 0.6704 [0.1507, 1.3392] | 4.2172 [2.2274, 6.7107] | 10.7441 [7.2134, 15.0229] | 1.2427 [0.7510, 1.8789] | 5.2326 [3.3773, 7.6725] | 12.0125 [8.7763, 16.1163] | 1.3565 [0.9790, 1.7857] | 2.9950 [2.2129, 3.9053] |
| IPox_RLFT (0,1) | 0.1246 [0.0752, 0.1873] | 0.0421 [0.0206, 0.0660] | 0.1830 [0.1376, 0.2315] | -0.2121 [-0.2594, -0.1651] | 0.7561 [0.0790, 1.3428] | 4.9179 [3.3324, 6.5198] | 0.4837 [0.4329, 0.5361] | 2.6582 [2.3576, 2.9983] | 7.5785 [6.6350, 8.6730] | 1.0555 [0.7232, 1.4345] | 2.0488 [1.4877, 2.6843] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
