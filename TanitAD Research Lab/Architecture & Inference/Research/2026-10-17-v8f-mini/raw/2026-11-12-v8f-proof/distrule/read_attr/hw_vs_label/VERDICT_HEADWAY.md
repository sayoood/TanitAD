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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSXP:LSRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSXG:LSRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSXC:LSRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSXZ:LSRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDXP:LSDRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDXG:LSDRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDXC:LSDRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDXZ:LSDRL", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, LSXP minus LSRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSXG minus LSRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSXC minus LSRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSXZ minus LSRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDXP minus LSDRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDXG minus LSDRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDXC minus LSDRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDXZ minus LSDRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## LSXP minus LSRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0018 / +0.0053 | +0.0036 | [+0.0000, +0.0111] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / +0.0016 | +0.0008 | [+0.0000, +0.0032] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0031 / -0.0078 | -0.0055 | [-0.0133, +0.0004] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0421 / -0.0477 | -0.0449 | [-0.0753, -0.0192] | 0.0689 (0.0315) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.1610 / -0.2235 | -0.1922 | [-0.3621, -0.0642] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.2745 / -0.6431 | -0.4588 | [-1.0453, -0.0837] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0335 / -0.0205 | -0.0270 | [-0.0536, -0.0048] | 0.0426 (0.0189) | INSIDE-SPREAD |
| gap error 4.0 s (m) | -0.1574 / -0.1557 | -0.1565 | [-0.2755, -0.0567] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | -0.2770 / -0.5311 | -0.4041 | [-0.8461, -0.0876] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.0024 / +0.0001 | -0.0012 | [-0.0032, +0.0017] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0031 / -0.0166 | -0.0068 | [-0.0270, +0.0214] | 1.7849 (0.8089) | OPEN |

## LSXG minus LSRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0338 / +0.0302 | +0.0320 | [+0.0143, +0.0522] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0140 / +0.0171 | +0.0156 | [+0.0053, +0.0288] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0109 / +0.0498 | +0.0195 | [-0.0180, +0.0641] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0775 / +0.1859 | +0.0542 | [-0.1097, +0.2731] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.1011 / +0.5903 | +0.2446 | [-0.2087, +0.9309] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.6357 / +0.9018 | +0.1330 | [-0.8857, +1.5171] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0526 / +0.1627 | +0.0551 | [-0.0835, +0.2418] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.0104 / +0.8120 | +0.4008 | [-0.1103, +1.1157] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.3041 / +1.2666 | +0.4812 | [-0.5201, +1.8765] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0167 / +0.1877 | +0.0855 | [-0.0255, +0.2797] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.1008 / +0.5784 | +0.2388 | [-0.1561, +0.7716] | 1.7849 (0.8089) | OPEN |

## LSXC minus LSRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0107 / +0.0356 | +0.0231 | [+0.0018, +0.0502] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0140 / +0.0296 | +0.0218 | [+0.0072, +0.0433] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0125 / +0.0343 | +0.0234 | [-0.0022, +0.0491] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0584 / +0.0563 | +0.0574 | [-0.0053, +0.1260] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.2761 / +0.5353 | +0.4057 | [+0.0752, +0.7874] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.2702 / +0.8549 | +0.5625 | [-0.2503, +1.5059] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0611 / +0.0729 | +0.0670 | [+0.0116, +0.1297] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.2943 / +0.6919 | +0.4931 | [+0.1422, +0.9176] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.5365 / +1.3212 | +0.9289 | [+0.0670, +1.8739] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.0053 / +0.0440 | +0.0194 | [-0.0179, +0.0884] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.1372 / +0.2255 | +0.1813 | [+0.0343, +0.3452] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSXZ minus LSRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0196 / +0.0285 | +0.0240 | [+0.0122, +0.0409] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0140 / +0.0343 | +0.0241 | [+0.0072, +0.0500] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0125 / +0.0171 | +0.0148 | [-0.0031, +0.0318] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0103 / +0.0020 | +0.0061 | [-0.0334, +0.0499] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.3559 / +0.1232 | +0.2395 | [-0.0804, +0.5966] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.5840 / +0.0050 | +0.2945 | [-0.4483, +1.2076] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0205 / +0.0394 | +0.0299 | [+0.0013, +0.0660] | 0.0426 (0.0189) | INSIDE-SPREAD |
| gap error 4.0 s (m) | +0.4406 / +0.3821 | +0.4114 | [+0.1533, +0.7014] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.7763 / +0.5318 | +0.6540 | [+0.0383, +1.4268] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0252 / +0.0182 | +0.0217 | [-0.0153, +0.0676] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.1115 / +0.1974 | +0.1544 | [+0.0585, +0.2756] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDXP minus LSDRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0053 / +0.0142 | +0.0098 | [+0.0016, +0.0239] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0062 / -0.0062 | -0.0062 | [-0.0133, -0.0006] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0338 / -0.0596 | -0.0467 | [-0.0833, -0.0176] | 0.0689 (0.0315) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.1134 / -0.2453 | -0.1794 | [-0.3789, -0.0515] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.3270 / -0.3362 | -0.3316 | [-0.6777, -0.0820] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0247 / -0.0274 | -0.0261 | [-0.0502, -0.0066] | 0.0426 (0.0189) | INSIDE-SPREAD |
| gap error 4.0 s (m) | -0.0822 / -0.1850 | -0.1336 | [-0.2841, -0.0202] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | -0.3270 / -0.2747 | -0.3009 | [-0.5765, -0.0812] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.0018 / -0.0025 | -0.0022 | [-0.0037, -0.0009] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.0403 / -0.0160 | -0.0282 | [-0.0572, -0.0079] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDXG minus LSDRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0623 / +0.0498 | +0.0560 | [+0.0286, +0.0864] | 0.0463 (0.0214) | WORSE |
| C contact rate | +0.0093 / +0.0405 | +0.0249 | [+0.0045, +0.0571] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0078 / -0.0202 | -0.0140 | [-0.0362, +0.0090] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0388 / -0.0608 | -0.0498 | [-0.1549, +0.0345] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.1194 / -0.4477 | -0.2836 | [-0.7787, +0.1148] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.4485 / -1.0890 | -0.7687 | [-1.6189, -0.0077] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0334 / -0.0280 | -0.0307 | [-0.1160, +0.0374] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0824 / -0.0953 | -0.0064 | [-0.4235, +0.3184] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.1805 / -0.4829 | -0.3317 | [-1.0411, +0.2619] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0212 / -0.0192 | -0.0202 | [-0.0431, -0.0025] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.1056 / +0.0323 | -0.0366 | [-0.1503, +0.0867] | 1.7849 (0.8089) | OPEN |

## LSDXC minus LSDRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0320 / +0.0285 | +0.0302 | [+0.0173, +0.0439] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0093 / +0.0249 | +0.0171 | [+0.0042, +0.0385] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0016 / +0.0140 | +0.0062 | [-0.0165, +0.0266] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0272 / +0.0676 | +0.0474 | [-0.0025, +0.1092] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.1978 / +0.0448 | +0.1213 | [-0.0975, +0.3445] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.2940 / -0.0946 | +0.0997 | [-0.4201, +0.7500] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0419 / +0.0846 | +0.0633 | [+0.0184, +0.1218] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.3153 / +0.2461 | +0.2807 | [+0.0907, +0.4922] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.3986 / +0.3493 | +0.3740 | [-0.1076, +0.9305] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0108 / -0.0110 | -0.0109 | [-0.0333, +0.0068] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0092 / +0.0504 | +0.0298 | [-0.0422, +0.1033] | 1.7849 (0.8089) | OPEN |

## LSDXZ minus LSDRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0125 / +0.0160 | +0.0142 | [+0.0046, +0.0253] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0062 / +0.0327 | +0.0195 | [+0.0032, +0.0467] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0047 / +0.0202 | +0.0125 | [-0.0054, +0.0278] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0100 / +0.0078 | +0.0089 | [-0.0279, +0.0512] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.0712 / -0.0544 | +0.0084 | [-0.2289, +0.2382] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.0385 / -0.4332 | -0.2358 | [-0.7026, +0.2937] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0337 / +0.0294 | +0.0315 | [+0.0050, +0.0644] | 0.0426 (0.0189) | INSIDE-SPREAD |
| gap error 4.0 s (m) | +0.1860 / +0.1220 | +0.1540 | [-0.0693, +0.3715] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.0484 / -0.0026 | +0.0229 | [-0.3855, +0.4370] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0127 / +0.0116 | +0.0121 | [-0.0211, +0.0487] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0924 / +0.1500 | +0.1212 | [+0.0214, +0.2308] | 1.7849 (0.8089) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LSDRL (0,1) | 0.0472 [0.0227, 0.0829] | 0.0000 [0.0000, 0.0000] | 0.3193 [0.2469, 0.3910] | 0.2364 [-0.0076, 0.5420] | 2.8677 [1.8890, 3.9087] | 9.9103 [7.7746, 11.9708] | 0.8047 [0.5833, 1.0843] | 3.7077 [2.8459, 4.6228] | 10.8099 [8.9499, 12.7615] | 1.5058 [0.8571, 2.2451] | 2.6637 [1.6935, 3.7219] |
| LSDXC (0,1) | 0.0774 [0.0473, 0.1158] | 0.0171 [0.0042, 0.0385] | 0.3255 [0.2582, 0.3932] | 0.2838 [0.0126, 0.6204] | 2.9890 [1.8870, 4.1690] | 10.0100 [7.6793, 12.3139] | 0.8679 [0.6258, 1.1778] | 3.9884 [3.0501, 5.0176] | 11.1839 [9.2636, 13.2363] | 1.4949 [0.8513, 2.2420] | 2.6935 [1.7438, 3.7231] |
| LSDXG (0,1) | 0.1032 [0.0640, 0.1468] | 0.0249 [0.0045, 0.0571] | 0.3053 [0.2281, 0.3835] | 0.1866 [-0.0366, 0.4597] | 2.5841 [1.4691, 3.7323] | 9.1416 [6.8977, 11.2569] | 0.7740 [0.5701, 1.0218] | 3.7012 [2.8265, 4.6442] | 10.4782 [8.6535, 12.2760] | 1.4856 [0.8401, 2.2283] | 2.6270 [1.7133, 3.6222] |
| LSDXP (0,1) | 0.0569 [0.0269, 0.0990] | 0.0000 [0.0000, 0.0000] | 0.3131 [0.2423, 0.3844] | 0.1897 [-0.0408, 0.4770] | 2.6883 [1.6867, 3.6943] | 9.5787 [7.5537, 11.5574] | 0.7786 [0.5764, 1.0350] | 3.5741 [2.7238, 4.4611] | 10.5090 [8.7594, 12.3479] | 1.5036 [0.8550, 2.2437] | 2.6355 [1.6770, 3.6968] |
| LSDXZ (0,1) | 0.0614 [0.0323, 0.1030] | 0.0195 [0.0032, 0.0467] | 0.3318 [0.2650, 0.3969] | 0.2453 [-0.0049, 0.5551] | 2.8761 [1.8262, 4.0054] | 9.6745 [7.5776, 11.6781] | 0.8362 [0.6095, 1.1196] | 3.8617 [2.9711, 4.8228] | 10.8328 [9.1522, 12.5669] | 1.5179 [0.8692, 2.2734] | 2.7849 [1.8421, 3.8356] |
| LSRL (0,1) | 0.0454 [0.0238, 0.0728] | 0.0000 [0.0000, 0.0000] | 0.2741 [0.1700, 0.3895] | 0.2064 [-0.0545, 0.5267] | 2.5248 [1.6354, 3.5112] | 8.7001 [6.1813, 11.2690] | 0.8261 [0.5991, 1.1154] | 3.4991 [2.7134, 4.3601] | 9.8022 [7.5079, 12.2263] | 1.2769 [0.7638, 1.8367] | 2.5743 [1.5172, 3.8082] |
| LSXC (0,1) | 0.0685 [0.0363, 0.1122] | 0.0218 [0.0072, 0.0433] | 0.2975 [0.2024, 0.3978] | 0.2637 [-0.0212, 0.6042] | 2.9305 [1.9758, 3.9639] | 9.2626 [6.7534, 11.7512] | 0.8931 [0.6464, 1.2029] | 3.9922 [3.2155, 4.8727] | 10.7310 [8.6213, 12.9089] | 1.2963 [0.7856, 1.8377] | 2.7557 [1.7348, 3.9155] |
| LSXG (0,1) | 0.0774 [0.0481, 0.1123] | 0.0156 [0.0053, 0.0288] | 0.2936 [0.2166, 0.3781] | 0.2605 [-0.0090, 0.5814] | 2.7694 [1.9052, 3.7896] | 8.8331 [6.7092, 10.9397] | 0.8811 [0.6359, 1.1904] | 3.9000 [3.1509, 4.7730] | 10.2834 [8.4180, 12.1821] | 1.3624 [0.9040, 1.8612] | 2.8131 [1.9603, 3.7515] |
| LSXP (0,1) | 0.0489 [0.0257, 0.0783] | 0.0008 [0.0000, 0.0032] | 0.2687 [0.1630, 0.3844] | 0.1615 [-0.0849, 0.4606] | 2.3326 [1.4548, 3.2861] | 8.2413 [5.7126, 10.8140] | 0.7991 [0.5922, 1.0687] | 3.3426 [2.5848, 4.1317] | 9.3981 [7.1138, 11.8388] | 1.2757 [0.7665, 1.8341] | 2.5676 [1.4979, 3.8216] |
| LSXZ (0,1) | 0.0694 [0.0419, 0.1063] | 0.0241 [0.0072, 0.0500] | 0.2889 [0.1850, 0.3985] | 0.2125 [-0.0567, 0.5367] | 2.7644 [1.6694, 3.9942] | 8.9946 [6.2010, 12.0337] | 0.8560 [0.6312, 1.1440] | 3.9105 [2.9618, 4.9515] | 10.4562 [8.0463, 13.2088] | 1.2986 [0.7759, 1.8737] | 2.7288 [1.6905, 3.9145] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
