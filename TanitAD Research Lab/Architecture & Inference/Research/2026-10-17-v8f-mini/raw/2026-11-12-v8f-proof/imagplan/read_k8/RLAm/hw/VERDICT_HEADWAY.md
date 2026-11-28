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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPlx_RLAm:IPox_RLAm", "seeds": [2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLAm:IPox_RLAm", "seeds": [2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLAm:IPcx_RLAm", "seeds": [2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLAm:IPlx_RLAm", "seeds": [2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcx_RLAm:IPox_RLAm", "seeds": [2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPksx_RLAm:IPkx_RLAm", "seeds": [2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPkx_RLAm:IPix_RLAm", "seeds": [2, 3]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPix_RLAm:IPox_RLAm", "seeds": [2, 3]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, IPlx_RLAm minus IPox_RLAm over seeds [2, 3] of [2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLAm minus IPox_RLAm over seeds [2, 3] of [2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLAm minus IPcx_RLAm over seeds [2, 3] of [2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLAm minus IPlx_RLAm over seeds [2, 3] of [2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcx_RLAm minus IPox_RLAm over seeds [2, 3] of [2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPksx_RLAm minus IPkx_RLAm over seeds [2, 3] of [2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPkx_RLAm minus IPix_RLAm over seeds [2, 3] of [2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPix_RLAm minus IPox_RLAm over seeds [2, 3] of [2, 3]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## IPlx RLAm minus IPox RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0979 / -0.0996 | -0.0988 | [-0.1407, -0.0616] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0530 / -0.0561 | -0.0545 | [-0.0811, -0.0303] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1028 / +0.1090 | +0.1059 | [+0.0615, +0.1592] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.8893 / +0.8938 | +0.8916 | [+0.3671, +1.5603] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.4049 / +3.5937 | +3.4993 | [+1.4799, +6.1210] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.2343 / +6.6207 | +6.4275 | [+3.0844, +11.0628] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.7643 / +0.7663 | +0.7653 | [+0.2677, +1.4007] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.5461 / +2.6213 | +2.5837 | [+0.7714, +4.9490] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +4.1734 / +4.6934 | +4.4334 | [+1.4693, +8.5265] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.2673 / +0.3049 | +0.2861 | [+0.1323, +0.4860] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.7055 / +1.0530 | +0.8793 | [+0.3340, +1.6598] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPkx RLAm minus IPox RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0427 / -0.0463 | -0.0445 | [-0.0735, -0.0197] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0202 / -0.0140 | -0.0171 | [-0.0295, -0.0069] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1028 / +0.1184 | +0.1106 | [+0.0606, +0.1667] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6964 / +0.7744 | +0.7354 | [+0.3099, +1.2373] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.6306 / +3.0974 | +2.8640 | [+1.2310, +4.8484] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +4.7628 / +5.2333 | +4.9980 | [+2.4016, +8.3585] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.6123 / +0.7028 | +0.6575 | [+0.2642, +1.1291] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.1640 / +2.4798 | +2.3219 | [+0.9373, +3.9896] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.3177 / +4.2345 | +3.7761 | [+1.4781, +6.6734] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.2915 / +0.1968 | +0.2442 | [+0.1082, +0.4275] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.6724 / +0.6697 | +0.6711 | [+0.3358, +1.0676] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPkx RLAm minus IPcx RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0196 / +0.0089 | +0.0142 | [-0.0004, +0.0337] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0078 / +0.0156 | +0.0117 | [+0.0025, +0.0226] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0156 / -0.0016 | -0.0086 | [-0.0298, +0.0090] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0627 / +0.0593 | -0.0017 | [-0.1187, +0.1419] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.2303 / +0.2694 | +0.0195 | [-0.4321, +0.5813] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.9862 / +0.2786 | -0.3538 | [-1.3998, +0.8867] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0494 / +0.0611 | +0.0058 | [-0.1016, +0.1410] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.0896 / +0.2824 | +0.0964 | [-0.2921, +0.5857] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.6576 / +0.7182 | +0.0303 | [-1.0019, +1.3073] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.2334 / -0.0896 | -0.1615 | [-0.3543, -0.0152] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.3568 / +0.0550 | -0.1509 | [-0.5134, +0.1984] | 1.7849 (0.8089) | OPEN |

## IPkx RLAm minus IPlx RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0552 / +0.0534 | +0.0543 | [+0.0309, +0.0812] | 0.0463 (0.0214) | WORSE |
| C contact rate | +0.0327 / +0.0421 | +0.0374 | [+0.0157, +0.0634] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0000 / +0.0093 | +0.0047 | [-0.0157, +0.0258] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1929 / -0.1194 | -0.1561 | [-0.3949, +0.0733] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.7743 / -0.4964 | -0.6353 | [-1.5196, +0.2406] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -1.4716 / -1.3875 | -1.4295 | [-3.0516, +0.0008] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.1520 / -0.0635 | -0.1078 | [-0.3411, +0.1208] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.3821 / -0.1415 | -0.2618 | [-1.1400, +0.5767] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.8557 / -0.4589 | -0.6573 | [-2.2443, +0.8708] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0242 / -0.1081 | -0.0420 | [-0.2516, +0.1617] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0331 / -0.3833 | -0.2082 | [-0.7853, +0.2671] | 1.7849 (0.8089) | OPEN |

## IPcx RLAm minus IPox RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0623 / -0.0552 | -0.0587 | [-0.0962, -0.0268] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0280 / -0.0296 | -0.0288 | [-0.0431, -0.0158] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1184 / +0.1199 | +0.1192 | [+0.0686, +0.1771] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7591 / +0.7151 | +0.7371 | [+0.3479, +1.1842] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.8609 / +2.8280 | +2.8445 | [+1.3767, +4.6408] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +5.7490 / +4.9546 | +5.3518 | [+2.9549, +8.3878] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.6617 / +0.6417 | +0.6517 | [+0.2927, +1.0745] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.2535 / +2.1975 | +2.2255 | [+0.9627, +3.7121] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.9753 / +3.5163 | +3.7458 | [+1.7318, +6.2400] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.5250 / +0.2864 | +0.4057 | [+0.1505, +0.7702] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.0292 / +0.6147 | +0.8219 | [+0.4055, +1.3709] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPksx RLAm minus IPkx RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0872 / -0.0943 | -0.0907 | [-0.1278, -0.0574] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0327 / -0.0436 | -0.0382 | [-0.0646, -0.0158] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.4626 / +0.4735 | +0.4681 | [+0.4043, +0.5304] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +5.6155 / +5.7971 | +5.7063 | [+4.9964, +6.3526] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +15.2019 / +15.7969 | +15.4994 | [+13.2686, +17.6139] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +23.2542 / +24.4802 | +23.8672 | [+20.1505, +27.7172] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +5.1990 / +5.2724 | +5.2357 | [+4.5697, +5.8402] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +14.1620 / +14.7519 | +14.4570 | [+12.3014, +16.5360] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +21.7272 / +22.7899 | +22.2586 | [+18.4313, +26.1584] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +19.7908 / +22.5238 | +21.1573 | [+17.1044, +25.3927] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +20.3277 / +23.6167 | +21.9722 | [+18.1431, +25.5576] | 1.7849 (0.8089) | WORSE |

## IPkx RLAm minus IPix RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0053 / +0.0018 | +0.0036 | [+0.0000, +0.0099] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0016 / +0.0016 | +0.0016 | [+0.0000, +0.0039] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0062 / -0.0093 | -0.0078 | [-0.0152, -0.0019] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0070 / +0.0013 | +0.0041 | [-0.0254, +0.0369] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.0492 / -0.0100 | +0.0196 | [-0.1103, +0.1444] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.1803 / -0.2200 | -0.2002 | [-0.4151, -0.0096] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0138 / +0.0013 | +0.0075 | [-0.0247, +0.0410] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0514 / -0.0044 | +0.0235 | [-0.1011, +0.1482] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.1258 / -0.1923 | -0.1591 | [-0.3806, +0.0322] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0079 / -0.0999 | -0.0539 | [-0.1936, +0.0023] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0123 / -0.0988 | -0.0555 | [-0.1601, +0.0077] | 1.7849 (0.8089) | OPEN |

## IPix RLAm minus IPox RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0480 / -0.0480 | -0.0480 | [-0.0792, -0.0220] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0218 / -0.0156 | -0.0187 | [-0.0312, -0.0081] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1090 / +0.1277 | +0.1184 | [+0.0684, +0.1763] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6895 / +0.7731 | +0.7313 | [+0.3114, +1.2180] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.5814 / +3.1074 | +2.8444 | [+1.2404, +4.8533] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +4.9431 / +5.4533 | +5.1982 | [+2.6016, +8.5559] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.5985 / +0.7015 | +0.6500 | [+0.2637, +1.1170] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.1125 / +2.4842 | +2.2984 | [+0.9345, +3.9706] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.4435 / +4.4268 | +3.9352 | [+1.6562, +6.7867] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.2994 / +0.2967 | +0.2981 | [+0.1258, +0.5073] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.6847 / +0.7685 | +0.7266 | [+0.3685, +1.1701] | 1.7849 (0.8089) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| IPcx_RLAm (2,3) | 0.0783 [0.0469, 0.1142] | 0.0265 [0.0103, 0.0456] | 0.2718 [0.2073, 0.3353] | 0.5077 [0.1009, 0.9595] | 3.6167 [2.2067, 5.2254] | 9.1381 [6.6758, 11.7639] | 1.1380 [0.7682, 1.5544] | 4.8771 [3.5529, 6.4439] | 10.6712 [8.5558, 13.0676] | 1.6156 [1.1080, 2.2082] | 2.9165 [2.1344, 3.7640] |
| IPix_RLAm (2,3) | 0.0890 [0.0553, 0.1265] | 0.0366 [0.0150, 0.0622] | 0.2710 [0.2053, 0.3354] | 0.5019 [0.0873, 0.9958] | 3.6166 [2.0643, 5.4319] | 8.9844 [6.2420, 12.1468] | 1.1363 [0.7381, 1.6129] | 4.9500 [3.5451, 6.6405] | 10.8605 [8.3942, 13.7499] | 1.5080 [1.0569, 1.9827] | 2.8212 [2.1480, 3.5775] |
| IPksx_RLAm (2,3) | 0.0018 [0.0000, 0.0070] | 0.0000 [0.0000, 0.0000] | 0.7313 [0.6601, 0.7901] | 6.2123 [5.4161, 6.9306] | 19.1356 [16.5301, 21.8786] | 32.6514 [27.7710, 37.9562] | 6.3795 [5.6131, 7.0737] | 19.4305 [16.9177, 22.1109] | 32.9601 [28.1948, 38.1951] | 22.6114 [18.6221, 26.7408] | 24.7378 [20.9120, 28.4562] |
| IPkx_RLAm (2,3) | 0.0925 [0.0576, 0.1295] | 0.0382 [0.0158, 0.0646] | 0.2632 [0.1983, 0.3282] | 0.5060 [0.0867, 1.0062] | 3.6362 [2.0885, 5.4113] | 8.7843 [6.0415, 11.9518] | 1.1438 [0.7423, 1.6210] | 4.9735 [3.5626, 6.6861] | 10.7015 [8.2487, 13.6044] | 1.4541 [1.0248, 1.9465] | 2.7656 [2.0972, 3.4838] |
| IPlx_RLAm (2,3) | 0.0383 [0.0186, 0.0619] | 0.0008 [0.0000, 0.0032] | 0.2586 [0.1958, 0.3218] | 0.6622 [0.1348, 1.3152] | 4.2715 [2.3939, 6.7332] | 10.2138 [6.9570, 14.6301] | 1.2516 [0.7550, 1.9000] | 5.2353 [3.4464, 7.6212] | 11.3588 [8.2832, 15.4707] | 1.4960 [1.0653, 1.9537] | 2.9738 [2.2063, 3.9183] |
| IPox_RLAm (2,3) | 0.1370 [0.0897, 0.1855] | 0.0553 [0.0309, 0.0821] | 0.1526 [0.1067, 0.1995] | -0.2294 [-0.2878, -0.1750] | 0.7722 [0.3178, 1.2553] | 3.7863 [2.4486, 5.1684] | 0.4863 [0.4369, 0.5437] | 2.6516 [2.3632, 2.9619] | 6.9254 [6.0739, 7.8562] | 1.2099 [0.8219, 1.6524] | 2.0946 [1.4948, 2.7981] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
