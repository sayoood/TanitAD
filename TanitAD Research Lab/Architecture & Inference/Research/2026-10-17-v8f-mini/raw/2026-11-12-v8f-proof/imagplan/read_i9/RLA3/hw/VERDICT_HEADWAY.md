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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPlx_RLA3:IPox_RLA3", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPix_RLA3:IPox_RLA3", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPix_RLA3:IPcx_RLA3", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPix_RLA3:IPlx_RLA3", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcx_RLA3:IPox_RLA3", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPsx_RLA3:IPix_RLA3", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLA3:IPix_RLA3", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLA3:IPox_RLA3", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLA3:IPcx_RLA3", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, IPlx_RLA3 minus IPox_RLA3 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPix_RLA3 minus IPox_RLA3 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPix_RLA3 minus IPcx_RLA3 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPix_RLA3 minus IPlx_RLA3 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcx_RLA3 minus IPox_RLA3 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPsx_RLA3 minus IPix_RLA3 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLA3 minus IPix_RLA3 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLA3 minus IPox_RLA3 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLA3 minus IPcx_RLA3 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## IPlx RLA3 minus IPox RLA3

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0872 / -0.0961 | -0.0916 | [-0.1276, -0.0587] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0530 / -0.0436 | -0.0483 | [-0.0761, -0.0245] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0857 / +0.0997 | +0.0927 | [+0.0508, +0.1430] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.8842 / +0.8857 | +0.8850 | [+0.3608, +1.5569] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.4983 / +3.5107 | +3.5045 | [+1.4932, +6.0924] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +5.7166 / +6.0175 | +5.8671 | [+2.7062, +10.2418] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.7572 / +0.7662 | +0.7617 | [+0.2656, +1.3923] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.7747 / +2.4197 | +2.5972 | [+0.7584, +4.9855] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +4.1414 / +4.5865 | +4.3639 | [+1.4483, +8.5427] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.2964 / +0.2920 | +0.2942 | [+0.1295, +0.5060] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.8565 / +0.8483 | +0.8524 | [+0.3612, +1.4830] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPix RLA3 minus IPox RLA3

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0160 / -0.0890 | -0.0525 | [-0.1131, -0.0084] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0109 / -0.0265 | -0.0187 | [-0.0421, -0.0060] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0358 / +0.2134 | +0.1246 | [+0.0217, +0.2603] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2055 / +1.4746 | +0.8400 | [+0.1099, +1.9020] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +0.8939 / +6.0131 | +3.4535 | [+0.4862, +7.8048] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +1.7509 / +11.4652 | +6.6080 | [+1.0456, +15.1509] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.1663 / +1.3078 | +0.7370 | [+0.0822, +1.7153] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.6958 / +4.7967 | +2.7463 | [+0.3398, +6.3691] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +1.3666 / +9.8606 | +5.6136 | [+0.6236, +13.1957] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.0537 / +0.6317 | +0.3427 | [+0.0287, +0.7871] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.1918 / +1.9793 | +1.0855 | [+0.1186, +2.4122] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPix RLA3 minus IPcx RLA3

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0320 / -0.0285 | +0.0018 | [-0.0461, +0.0483] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0218 / -0.0062 | +0.0078 | [-0.0113, +0.0292] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0436 / +0.0701 | +0.0132 | [-0.0574, +0.0930] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.4832 / +0.4298 | -0.0267 | [-0.6806, +0.5987] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -1.7450 / +1.8425 | +0.0487 | [-2.4759, +2.5441] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -2.8017 / +3.6786 | +0.4384 | [-3.8354, +5.0792] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.4233 / +0.3580 | -0.0326 | [-0.6112, +0.5152] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -1.5815 / +1.4886 | -0.0464 | [-2.2967, +2.1575] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -2.0585 / +3.2251 | +0.5833 | [-3.0681, +4.6147] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.2697 / +0.1084 | -0.0806 | [-0.3862, +0.1990] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.6432 / +0.4380 | -0.1026 | [-0.8603, +0.6172] | 1.7849 (0.8089) | OPEN |

## IPix RLA3 minus IPlx RLA3

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0712 / +0.0071 | +0.0391 | [-0.0036, +0.0899] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0421 / +0.0171 | +0.0296 | [+0.0096, +0.0582] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0498 / +0.1137 | +0.0319 | [-0.0649, +0.1472] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.6787 / +0.5888 | -0.0449 | [-0.9422, +0.8533] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -2.6043 / +2.5024 | -0.0510 | [-3.5836, +3.6475] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -3.9658 / +5.4477 | +0.7410 | [-5.2589, +7.8901] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.5910 / +0.5416 | -0.0247 | [-0.8400, +0.7886] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -2.0789 / +2.3770 | +0.1491 | [-3.0168, +3.3701] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -2.7748 / +5.2741 | +1.2497 | [-4.1144, +7.6118] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.2427 / +0.3397 | +0.0485 | [-0.3476, +0.4749] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.6647 / +1.1310 | +0.2331 | [-0.9911, +1.4966] | 1.7849 (0.8089) | OPEN |

## IPcx RLA3 minus IPox RLA3

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0480 / -0.0605 | -0.0543 | [-0.0831, -0.0287] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0327 / -0.0202 | -0.0265 | [-0.0436, -0.0120] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0794 / +0.1433 | +0.1114 | [+0.0539, +0.1829] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6887 / +1.0448 | +0.8667 | [+0.3402, +1.5236] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.6389 / +4.1706 | +3.4048 | [+1.3426, +6.0396] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +4.5526 / +7.7866 | +6.1696 | [+2.7527, +11.1033] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.5896 / +0.9498 | +0.7697 | [+0.2721, +1.3933] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.2773 / +3.3081 | +2.7927 | [+1.0000, +5.0554] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.4251 / +6.6355 | +5.0303 | [+1.5642, +9.5465] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.3234 / +0.5233 | +0.4234 | [+0.1917, +0.6974] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.8349 / +1.5413 | +1.1881 | [+0.5701, +1.9365] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPsx RLA3 minus IPix RLA3

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0872 / -0.0445 | -0.0658 | [-0.1122, -0.0312] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0421 / -0.0171 | -0.0296 | [-0.0582, -0.0096] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.4704 / +0.4393 | +0.4548 | [+0.3929, +0.5115] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +4.8216 / +5.8096 | +5.3156 | [+4.3659, +6.1961] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +14.1438 / +15.8712 | +15.0075 | [+12.5983, +17.1881] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +21.5840 / +23.5884 | +22.5862 | [+18.9925, +26.2395] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +4.3747 / +5.4153 | +4.8950 | [+3.9414, +5.7881] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +13.0729 / +15.0006 | +14.0367 | [+11.5862, +16.2521] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +19.5119 / +22.4527 | +20.9823 | [+16.9556, +24.9450] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +16.1208 / +22.6701 | +19.3954 | [+13.9367, +24.8358] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +18.6471 / +26.5309 | +22.5890 | [+16.6806, +28.4020] | 1.7849 (0.8089) | WORSE |

## IPkx RLA3 minus IPix RLA3

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0053 | +0.0027 | [+0.0000, +0.0082] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / +0.0016 | +0.0008 | [+0.0000, +0.0032] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0031 / -0.0031 | +0.0000 | [-0.0060, +0.0052] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0137 / +0.0509 | +0.0186 | [-0.0383, +0.0850] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.0725 / +0.2090 | +0.0683 | [-0.1751, +0.3486] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.1180 / +0.3943 | +0.1382 | [-0.2397, +0.7355] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0137 / +0.0559 | +0.0211 | [-0.0378, +0.0889] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.0725 / +0.2308 | +0.0791 | [-0.1734, +0.3678] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.0781 / +0.4036 | +0.1628 | [-0.2008, +0.7480] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0030 / +0.0292 | +0.0131 | [-0.0071, +0.0537] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0170 / +0.0514 | +0.0172 | [-0.0468, +0.1248] | 1.7849 (0.8089) | OPEN |

## IPkx RLA3 minus IPox RLA3

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0160 / -0.0836 | -0.0498 | [-0.1065, -0.0084] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0109 / -0.0249 | -0.0179 | [-0.0404, -0.0059] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0389 / +0.2103 | +0.1246 | [+0.0243, +0.2560] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1918 / +1.5255 | +0.8587 | [+0.1063, +1.9676] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +0.8214 / +6.2221 | +3.5218 | [+0.4854, +8.0771] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +1.6329 / +11.8595 | +6.7462 | [+0.9980, +15.6087] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.1526 / +1.3637 | +0.7581 | [+0.0813, +1.7891] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.6233 / +5.0275 | +2.8254 | [+0.3304, +6.6216] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +1.2885 / +10.2642 | +5.7764 | [+0.6423, +13.6728] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.0507 / +0.6609 | +0.3558 | [+0.0269, +0.8324] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.1748 / +2.0307 | +1.1027 | [+0.1113, +2.4770] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPkx RLA3 minus IPcx RLA3

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0320 / -0.0231 | +0.0044 | [-0.0404, +0.0483] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0218 / -0.0047 | +0.0086 | [-0.0094, +0.0292] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0405 / +0.0670 | +0.0132 | [-0.0541, +0.0890] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.4969 / +0.4807 | -0.0081 | [-0.7104, +0.6750] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -1.8175 / +2.0515 | +0.1170 | [-2.6104, +2.8230] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -2.9197 / +4.0729 | +0.5766 | [-4.0487, +5.6778] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.4370 / +0.4139 | -0.0115 | [-0.6393, +0.6049] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -1.6540 / +1.7194 | +0.0327 | [-2.4033, +2.4944] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -2.1366 / +3.6287 | +0.7460 | [-3.2777, +5.3163] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.2727 / +0.1376 | -0.0676 | [-0.3917, +0.2494] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.6602 / +0.4894 | -0.0854 | [-0.8811, +0.7037] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| IPcx_RLA3 (0,1) | 0.0721 [0.0413, 0.1070] | 0.0218 [0.0071, 0.0380] | 0.2819 [0.2245, 0.3401] | 0.6479 [0.1147, 1.3064] | 4.1313 [2.2647, 6.5346] | 11.0026 [7.6951, 15.2736] | 1.2568 [0.7673, 1.8743] | 5.4322 [3.6022, 7.7582] | 12.6083 [9.5250, 16.7116] | 1.5133 [1.0516, 2.0277] | 3.2076 [2.4214, 4.1050] |
| IPix_RLA3 (0,1) | 0.0738 [0.0343, 0.1256] | 0.0296 [0.0096, 0.0582] | 0.2952 [0.1968, 0.4012] | 0.6212 [-0.1246, 1.6799] | 4.1800 [1.4698, 7.9610] | 11.4410 [6.1729, 18.7380] | 1.2242 [0.5808, 2.1999] | 5.3858 [2.8833, 8.9933] | 13.1915 [8.6061, 19.9961] | 1.4327 [0.8887, 2.0757] | 3.1050 [1.9103, 4.5302] |
| IPkx_RLA3 (0,1) | 0.0765 [0.0385, 0.1256] | 0.0304 [0.0106, 0.0583] | 0.2952 [0.2011, 0.3968] | 0.6398 [-0.1286, 1.7455] | 4.2483 [1.4652, 8.2283] | 11.5792 [6.1074, 19.4395] | 1.2453 [0.5742, 2.2656] | 5.4649 [2.8549, 9.2924] | 13.3543 [8.5728, 20.6044] | 1.4457 [0.8874, 2.1062] | 3.1221 [1.8990, 4.6038] |
| IPlx_RLA3 (0,1) | 0.0347 [0.0176, 0.0554] | 0.0000 [0.0000, 0.0000] | 0.2632 [0.2005, 0.3276] | 0.6661 [0.1433, 1.3417] | 4.2310 [2.2404, 6.7037] | 10.7001 [7.2197, 14.9164] | 1.2489 [0.7531, 1.8844] | 5.2367 [3.3868, 7.6277] | 11.9419 [8.6871, 15.9191] | 1.3841 [0.9958, 1.8257] | 2.8718 [2.1728, 3.6635] |
| IPox_RLA3 (0,1) | 0.1263 [0.0838, 0.1732] | 0.0483 [0.0245, 0.0761] | 0.1706 [0.1201, 0.2288] | -0.2188 [-0.2691, -0.1701] | 0.7265 [0.0899, 1.2902] | 4.8330 [3.2531, 6.4849] | 0.4872 [0.4370, 0.5396] | 2.6395 [2.3568, 2.9477] | 7.5779 [6.4087, 9.0271] | 1.0899 [0.7426, 1.4986] | 2.0194 [1.4759, 2.6579] |
| IPsx_RLA3 (0,1) | 0.0080 [0.0000, 0.0181] | 0.0000 [0.0000, 0.0000] | 0.7500 [0.6667, 0.8279] | 5.9368 [4.3136, 7.5587] | 19.1876 [14.4195, 24.0104] | 34.0272 [26.1391, 42.6776] | 6.1192 [4.5740, 7.6679] | 19.4225 [14.7287, 24.1851] | 34.1738 [26.4197, 42.7842] | 20.8281 [15.1105, 26.5250] | 25.6939 [18.9113, 32.2498] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
