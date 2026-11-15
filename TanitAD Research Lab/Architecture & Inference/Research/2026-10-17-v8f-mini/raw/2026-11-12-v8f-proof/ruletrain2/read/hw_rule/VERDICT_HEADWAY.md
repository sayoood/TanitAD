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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GFTRL:GFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GOWRL:GOW", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GOW4RL:GOW4", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GOW:GFTRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GOW4:GFTRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GOWRL:GFTRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GOW4RL:GFTRL", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, GFTRL minus GFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GOWRL minus GOW over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GOW4RL minus GOW4 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GOW minus GFTRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GOW4 minus GFTRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GOWRL minus GFTRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GOW4RL minus GFTRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## GFTRL minus GFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0569 / -0.0783 | -0.0676 | [-0.1008, -0.0362] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0358 / -0.0452 | -0.0405 | [-0.0648, -0.0212] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0670 / +0.0685 | +0.0678 | [+0.0281, +0.1134] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4346 / +0.4358 | +0.4352 | [+0.1818, +0.7601] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.5137 / +1.1397 | +1.3267 | [+0.5822, +2.2604] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.8306 / +2.8614 | +2.8460 | [+1.3194, +4.6864] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3354 / +0.3216 | +0.3285 | [+0.0964, +0.6274] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.7415 / +0.5542 | +0.6479 | [+0.0752, +1.3464] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.4310 / +1.1152 | +1.2731 | [+0.1971, +2.6265] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0448 / +0.0424 | +0.0436 | [+0.0203, +0.0708] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0738 / +0.0992 | +0.0865 | [+0.0093, +0.1686] | 1.7849 (0.8089) | INSIDE-SPREAD |

## GOWRL minus GOW

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0409 / -0.0641 | -0.0525 | [-0.0832, -0.0245] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0109 / -0.0109 | -0.0109 | [-0.0196, -0.0034] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0047 / +0.0093 | +0.0070 | [-0.0025, +0.0179] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0392 / +0.0455 | +0.0424 | [+0.0241, +0.0652] | 0.0689 (0.0315) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.1104 / -0.0069 | +0.0518 | [-0.0466, +0.1497] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.0899 / +0.0544 | +0.0721 | [-0.3555, +0.4508] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0248 / +0.0427 | +0.0337 | [+0.0136, +0.0588] | 0.0426 (0.0189) | INSIDE-SPREAD |
| gap error 4.0 s (m) | +0.0152 / -0.0577 | -0.0213 | [-0.0951, +0.0517] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.1355 / -0.4581 | -0.2968 | [-0.7457, +0.0719] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.1600 / +0.0317 | +0.0959 | [+0.0143, +0.2483] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0770 / +0.2129 | +0.1450 | [-0.0669, +0.3623] | 1.7849 (0.8089) | OPEN |

## GOW4RL minus GOW4

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0320 / -0.0320 | -0.0320 | [-0.0514, -0.0150] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0125 / -0.0109 | -0.0117 | [-0.0212, -0.0041] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0047 / -0.0016 | -0.0031 | [-0.0099, +0.0035] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0208 / +0.0180 | +0.0194 | [+0.0087, +0.0321] | 0.0689 (0.0315) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.0591 / +0.1265 | +0.0928 | [+0.0326, +0.1698] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.0951 / +0.1406 | +0.0227 | [-0.2477, +0.3390] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0185 / +0.0173 | +0.0179 | [+0.0075, +0.0301] | 0.0426 (0.0189) | INSIDE-SPREAD |
| gap error 4.0 s (m) | -0.0290 / +0.0064 | -0.0113 | [-0.0526, +0.0477] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.1986 / -0.0523 | -0.1255 | [-0.3545, +0.1259] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0173 / +0.0124 | +0.0148 | [+0.0018, +0.0288] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.3909 / +0.0044 | +0.1977 | [-0.0311, +0.5869] | 1.7849 (0.8089) | OPEN |

## GOW minus GFTRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0285 / +0.0516 | +0.0400 | [+0.0094, +0.0750] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0109 / +0.0109 | +0.0109 | [+0.0034, +0.0196] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.2850 / +0.2679 | +0.2765 | [+0.2164, +0.3354] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.8655 / +2.5620 | +2.2138 | [+1.3887, +3.2648] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +8.8191 / +9.5477 | +9.1834 | [+6.4392, +12.2350] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +17.0103 / +17.0343 | +17.0223 | [+11.9008, +22.2821] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.6265 / +2.2366 | +1.9316 | [+1.1167, +2.9756] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +8.3643 / +8.9496 | +8.6569 | [+5.9702, +11.7191] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +16.8120 / +16.7547 | +16.7833 | [+11.6558, +22.0205] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +2.4811 / +3.1862 | +2.8337 | [+1.9038, +3.9035] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +3.7964 / +3.1795 | +3.4880 | [+2.2609, +4.8267] | 1.7849 (0.8089) | WORSE |

## GOW4 minus GFTRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0214 / +0.0089 | +0.0151 | [-0.0133, +0.0459] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0125 / +0.0109 | +0.0117 | [+0.0041, +0.0212] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.4159 / +0.3193 | +0.3676 | [+0.2814, +0.4537] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +3.7875 / +3.3176 | +3.5526 | [+2.4147, +4.7496] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +14.0790 / +10.0838 | +12.0814 | [+8.0907, +16.2270] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +24.7600 / +17.3846 | +21.0723 | [+14.4553, +28.3342] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +3.4953 / +2.9662 | +3.2307 | [+2.0906, +4.4490] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +13.4559 / +9.5768 | +11.5163 | [+7.6342, +15.6275] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +24.2042 / +16.8260 | +20.5151 | [+13.9147, +27.9154] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +5.6656 / +5.1039 | +5.3848 | [+3.9813, +6.8223] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +8.0893 / +4.4892 | +6.2893 | [+3.7455, +8.9243] | 1.7849 (0.8089) | WORSE |

## GOWRL minus GFTRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0125 / -0.0125 | -0.0125 | [-0.0352, +0.0101] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.2897 / +0.2773 | +0.2835 | [+0.2259, +0.3403] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.9047 / +2.6076 | +2.2562 | [+1.4176, +3.3221] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +8.9296 / +9.5408 | +9.2352 | [+6.5135, +12.2444] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +17.1002 / +17.0887 | +17.0944 | [+12.0222, +22.3570] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.6513 / +2.2793 | +1.9653 | [+1.1435, +3.0195] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +8.3795 / +8.8919 | +8.6357 | [+5.9718, +11.7010] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +16.6765 / +16.2966 | +16.4866 | [+11.3672, +21.6294] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +2.6412 / +3.2180 | +2.9296 | [+1.9616, +4.0056] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +3.8734 / +3.3924 | +3.6329 | [+2.3799, +5.0073] | 1.7849 (0.8089) | WORSE |

## GOW4RL minus GFTRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0107 / -0.0231 | -0.0169 | [-0.0426, +0.0091] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.4112 / +0.3178 | +0.3645 | [+0.2801, +0.4490] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +3.8084 / +3.3356 | +3.5720 | [+2.4311, +4.7632] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +14.1381 / +10.2104 | +12.1743 | [+8.1957, +16.2953] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +24.6649 / +17.5252 | +21.0951 | [+14.6000, +28.2736] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +3.5138 / +2.9835 | +3.2486 | [+2.1049, +4.4632] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +13.4269 / +9.5832 | +11.5050 | [+7.6043, +15.6140] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +24.0056 / +16.7737 | +20.3896 | [+13.8469, +27.6991] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +5.6829 / +5.1162 | +5.3996 | [+3.9953, +6.8322] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +8.4802 / +4.4936 | +6.4869 | [+3.7498, +9.4224] | 1.7849 (0.8089) | WORSE |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| GFT (0,1) | 0.1121 [0.0711, 0.1570] | 0.0405 [0.0212, 0.0648] | 0.1931 [0.1119, 0.2846] | -0.1611 [-0.2359, -0.0881] | 1.0823 [0.4765, 1.7433] | 5.2278 [2.7695, 7.6887] | 0.4680 [0.4239, 0.5183] | 2.7143 [2.3027, 3.2118] | 7.9076 [6.3657, 9.6000] | 1.2414 [0.7961, 1.7429] | 2.3332 [1.5451, 3.2287] |
| GFTRL (0,1) | 0.0445 [0.0255, 0.0664] | 0.0000 [0.0000, 0.0000] | 0.2609 [0.1758, 0.3522] | 0.2741 [0.0113, 0.5893] | 2.4091 [1.4418, 3.4539] | 8.0738 [5.4192, 10.5852] | 0.7965 [0.5692, 1.0945] | 3.3622 [2.6218, 4.1937] | 9.1807 [7.1953, 11.2347] | 1.2850 [0.8478, 1.7875] | 2.4197 [1.6515, 3.2980] |
| GOW (0,1) | 0.0845 [0.0506, 0.1200] | 0.0109 [0.0034, 0.0196] | 0.5374 [0.4218, 0.6388] | 2.4879 [1.4553, 3.7317] | 11.5925 [8.4754, 14.9974] | 25.0961 [18.6553, 31.3656] | 2.7280 [1.7111, 3.9766] | 12.0191 [8.9328, 15.4730] | 25.9640 [19.7118, 32.0403] | 4.1187 [3.1580, 5.1582] | 5.9076 [4.3993, 7.3870] |
| GOW4 (0,1) | 0.0596 [0.0352, 0.0893] | 0.0117 [0.0041, 0.0212] | 0.6285 [0.4821, 0.7630] | 3.8267 [2.4614, 5.3003] | 14.4905 [9.7170, 19.3796] | 29.1462 [20.2390, 38.4956] | 4.0272 [2.6922, 5.4804] | 14.8785 [10.3871, 19.5509] | 29.6958 [21.4025, 38.5684] | 6.6697 [5.0271, 8.2798] | 8.7089 [5.7377, 11.6355] |
| GOW4RL (0,1) | 0.0276 [0.0128, 0.0474] | 0.0000 [0.0000, 0.0000] | 0.6254 [0.4798, 0.7597] | 3.8461 [2.4849, 5.3240] | 14.5833 [9.8622, 19.4513] | 29.1689 [20.3782, 38.3461] | 4.0451 [2.7069, 5.5045] | 14.8672 [10.4063, 19.5250] | 29.5703 [21.2860, 38.3564] | 6.6846 [5.0323, 8.2967] | 8.9066 [5.7508, 12.0984] |
| GOWRL (0,1) | 0.0320 [0.0173, 0.0488] | 0.0000 [0.0000, 0.0000] | 0.5444 [0.4318, 0.6426] | 2.5303 [1.4921, 3.7883] | 11.6442 [8.5262, 15.0459] | 25.1683 [18.8753, 31.4381] | 2.7617 [1.7342, 4.0284] | 11.9979 [8.9068, 15.4544] | 25.6673 [19.5052, 31.7890] | 4.2145 [3.2199, 5.2610] | 6.0526 [4.5892, 7.5127] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
