# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: F:/TanitAD-scratch/strat1/read/inputs/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 28, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.033807829181494664, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RA_R0C10XHb20:I10B0_R0C10XHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RA_R0C10XHb20:I10RA_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RA_R0G50Hb20:I10RA_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RA_R0C10Hb20:I10RA_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0C10XHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0G50Hb20:I10B0_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0C10Hb20:I10B0_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RA_R0C10XHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RA_R0Hb20:I10B0_R0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RA_R0C10Hb20:I10B0_R0C10Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RA_R0G50Hb20:I10B0_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RA_R0G0Hb20:I10B0_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RA_R0KHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RA_R0Hb20:I10RH_R0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RA_R0C10XHb20:I10RH_R0C10XHb20", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10RA_R0Hb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10RA_R0C10XHb20 minus I10B0_R0C10XHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RA_R0C10XHb20 minus I10RA_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RA_R0G50Hb20 minus I10RA_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RA_R0C10Hb20 minus I10RA_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0C10XHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0G50Hb20 minus I10B0_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0C10Hb20 minus I10B0_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RA_R0C10XHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RA_R0Hb20 minus I10B0_R0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RA_R0C10Hb20 minus I10B0_R0C10Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RA_R0G50Hb20 minus I10B0_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RA_R0G0Hb20 minus I10B0_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RA_R0KHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RA_R0Hb20 minus I10RH_R0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RA_R0C10XHb20 minus I10RH_R0C10XHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10RA R0C10XHb20 minus I10B0 R0C10XHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0214 / +0.0214 | +0.0214 | [+0.0078, +0.0373] | 0.0338 (0.0338) | INSIDE-SPREAD |
| C contact rate | +0.0171 / +0.0078 | +0.0125 | [+0.0033, +0.0258] | 0.0140 (0.0140) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0779 / -0.0576 | -0.0678 | [-0.1073, -0.0382] | 0.0405 (0.0405) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0272 / -0.0279 | -0.0276 | [-0.0468, -0.0102] | 0.0009 (0.0009) | BETTER |
| gap bias 4.0 s (m) | -0.3105 / -0.3933 | -0.3519 | [-0.5033, -0.2101] | 0.2635 (0.2635) | BETTER |
| gap bias 6.0 s (m) | -1.8404 / -1.8722 | -1.8563 | [-2.4988, -1.2824] | 1.2896 (1.2896) | BETTER |
| gap error 2.0 s (m) | +0.0274 / +0.0267 | +0.0270 | [+0.0137, +0.0421] | 0.0141 (0.0141) | WORSE |
| gap error 4.0 s (m) | +0.0114 / -0.0954 | -0.0420 | [-0.1843, +0.0916] | 0.0535 (0.0535) | OPEN |
| gap error 6.0 s (m) | -0.8604 / -1.1247 | -0.9926 | [-1.6167, -0.4576] | 0.4280 (0.4280) | BETTER |
| time-gap error 2.0 s (s) | +0.0261 / +0.0494 | +0.0378 | [-0.0009, +0.0912] | 0.0151 (0.0151) | OPEN |
| time-gap error over the scored steps (s) | -0.7749 / -0.7110 | -0.7430 | [-1.0383, -0.4897] | 0.2327 (0.2327) | BETTER |

## I10RA R0C10XHb20 minus I10RA R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0391 / -0.0534 | -0.0463 | [-0.0701, -0.0220] | 0.0338 (0.0338) | BETTER |
| C contact rate | -0.0265 / -0.0327 | -0.0296 | [-0.0492, -0.0139] | 0.0140 (0.0140) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0841 / +0.1246 | +0.1044 | [+0.0679, +0.1475] | 0.0405 (0.0405) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0009 (0.0009) | OPEN |
| gap bias 4.0 s (m) | +1.0922 / +1.2884 | +1.1903 | [+1.0052, +1.4020] | 0.2635 (0.2635) | WORSE |
| gap bias 6.0 s (m) | +3.5236 / +4.5896 | +4.0566 | [+3.2320, +5.0008] | 1.2896 (1.2896) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0141 (0.0141) | OPEN |
| gap error 4.0 s (m) | +0.4739 / +0.4928 | +0.4833 | [+0.3067, +0.6741] | 0.0535 (0.0535) | WORSE |
| gap error 6.0 s (m) | +1.2998 / +1.7690 | +1.5344 | [+0.8472, +2.2876] | 0.4280 (0.4280) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0151 (0.0151) | OPEN |
| time-gap error over the scored steps (s) | +0.8325 / +1.1563 | +0.9944 | [+0.5844, +1.4699] | 0.2327 (0.2327) | WORSE |

## I10RA R0G50Hb20 minus I10RA R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0160 / -0.0409 | -0.0285 | [-0.0543, -0.0092] | 0.0338 (0.0338) | INSIDE-SPREAD |
| C contact rate | -0.0109 / -0.0140 | -0.0125 | [-0.0218, -0.0052] | 0.0140 (0.0140) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0467 / +0.0685 | +0.0576 | [+0.0353, +0.0838] | 0.0405 (0.0405) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0009 (0.0009) | OPEN |
| gap bias 4.0 s (m) | +0.5898 / +0.6856 | +0.6377 | [+0.5382, +0.7508] | 0.2635 (0.2635) | WORSE |
| gap bias 6.0 s (m) | +1.9291 / +2.6907 | +2.3099 | [+1.7671, +2.9611] | 1.2896 (1.2896) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0141 (0.0141) | OPEN |
| gap error 4.0 s (m) | +0.2183 / +0.2007 | +0.2095 | [+0.1065, +0.3161] | 0.0535 (0.0535) | WORSE |
| gap error 6.0 s (m) | +0.6717 / +0.8315 | +0.7516 | [+0.3233, +1.2256] | 0.4280 (0.4280) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0151 (0.0151) | OPEN |
| time-gap error over the scored steps (s) | +0.4636 / +0.9226 | +0.6931 | [+0.3123, +1.1677] | 0.2327 (0.2327) | WORSE |

## I10RA R0C10Hb20 minus I10RA R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0107 / -0.0142 | -0.0125 | [-0.0242, -0.0032] | 0.0338 (0.0338) | INSIDE-SPREAD |
| C contact rate | -0.0031 / -0.0109 | -0.0070 | [-0.0161, -0.0005] | 0.0140 (0.0140) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0218 / +0.0498 | +0.0358 | [+0.0167, +0.0618] | 0.0405 (0.0405) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0009 (0.0009) | OPEN |
| gap bias 4.0 s (m) | +0.3624 / +0.4809 | +0.4217 | [+0.3258, +0.5285] | 0.2635 (0.2635) | WORSE |
| gap bias 6.0 s (m) | +1.0514 / +1.4651 | +1.2583 | [+0.8895, +1.6492] | 1.2896 (1.2896) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0141 (0.0141) | OPEN |
| gap error 4.0 s (m) | +0.1939 / +0.2256 | +0.2097 | [+0.1395, +0.2896] | 0.0535 (0.0535) | WORSE |
| gap error 6.0 s (m) | +0.3637 / +0.7097 | +0.5367 | [+0.2349, +0.8757] | 0.4280 (0.4280) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0151 (0.0151) | OPEN |
| time-gap error over the scored steps (s) | +0.2170 / +0.0440 | +0.1305 | [-0.0862, +0.3209] | 0.2327 (0.2327) | OPEN |

## I10B0 R0C10XHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0516 / -0.0747 | -0.0632 | [-0.0977, -0.0308] | 0.0338 (0.0338) | BETTER |
| C contact rate | -0.0405 / -0.0374 | -0.0389 | [-0.0614, -0.0194] | 0.0140 (0.0140) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1511 / +0.1791 | +0.1651 | [+0.1215, +0.2139] | 0.0405 (0.0405) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0009 (0.0009) | OPEN |
| gap bias 4.0 s (m) | +1.3409 / +1.5946 | +1.4678 | [+1.2207, +1.7382] | 0.2635 (0.2635) | WORSE |
| gap bias 6.0 s (m) | +5.1910 / +6.2410 | +5.7160 | [+4.6498, +6.9298] | 1.2896 (1.2896) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0141 (0.0141) | OPEN |
| gap error 4.0 s (m) | +0.5314 / +0.6764 | +0.6039 | [+0.3704, +0.8402] | 0.0535 (0.0535) | WORSE |
| gap error 6.0 s (m) | +2.2039 / +2.9716 | +2.5877 | [+1.5457, +3.7505] | 0.4280 (0.4280) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0151 (0.0151) | OPEN |
| time-gap error over the scored steps (s) | +1.5846 / +1.7639 | +1.6743 | [+1.1307, +2.2340] | 0.2327 (0.2327) | WORSE |

## I10B0 R0G50Hb20 minus I10B0 R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0285 / -0.0391 | -0.0338 | [-0.0540, -0.0149] | 0.0338 (0.0338) | INSIDE-SPREAD |
| C contact rate | -0.0202 / -0.0202 | -0.0202 | [-0.0352, -0.0082] | 0.0140 (0.0140) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0561 / +0.0888 | +0.0724 | [+0.0441, +0.1092] | 0.0405 (0.0405) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0009 (0.0009) | OPEN |
| gap bias 4.0 s (m) | +0.6755 / +0.8065 | +0.7410 | [+0.6149, +0.8796] | 0.2635 (0.2635) | WORSE |
| gap bias 6.0 s (m) | +2.5213 / +3.3152 | +2.9182 | [+2.2914, +3.6283] | 1.2896 (1.2896) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0141 (0.0141) | OPEN |
| gap error 4.0 s (m) | +0.2169 / +0.2442 | +0.2305 | [+0.1141, +0.3530] | 0.0535 (0.0535) | WORSE |
| gap error 6.0 s (m) | +0.8243 / +1.1466 | +0.9854 | [+0.4232, +1.5805] | 0.4280 (0.4280) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0151 (0.0151) | OPEN |
| time-gap error over the scored steps (s) | +0.8436 / +1.0007 | +0.9222 | [+0.5077, +1.3596] | 0.2327 (0.2327) | WORSE |

## I10B0 R0C10Hb20 minus I10B0 R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0249 / -0.0302 | -0.0276 | [-0.0440, -0.0132] | 0.0338 (0.0338) | INSIDE-SPREAD |
| C contact rate | -0.0218 / -0.0171 | -0.0195 | [-0.0334, -0.0077] | 0.0140 (0.0140) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0950 / +0.1012 | +0.0981 | [+0.0698, +0.1293] | 0.0405 (0.0405) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0009 (0.0009) | OPEN |
| gap bias 4.0 s (m) | +0.6707 / +0.7715 | +0.7211 | [+0.6054, +0.8476] | 0.2635 (0.2635) | WORSE |
| gap bias 6.0 s (m) | +2.7210 / +3.1187 | +2.9198 | [+2.3181, +3.5892] | 1.2896 (1.2896) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0141 (0.0141) | OPEN |
| gap error 4.0 s (m) | +0.2994 / +0.3979 | +0.3486 | [+0.2279, +0.4805] | 0.0535 (0.0535) | WORSE |
| gap error 6.0 s (m) | +1.4038 / +2.0636 | +1.7337 | [+1.0930, +2.4937] | 0.4280 (0.4280) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0151 (0.0151) | OPEN |
| time-gap error over the scored steps (s) | +0.6962 / +0.6219 | +0.6590 | [+0.3831, +0.9571] | 0.2327 (0.2327) | WORSE |

## I10RA R0C10XHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0302 / -0.0534 | -0.0418 | [-0.0678, -0.0179] | 0.0338 (0.0338) | BETTER |
| C contact rate | -0.0234 / -0.0296 | -0.0265 | [-0.0451, -0.0115] | 0.0140 (0.0140) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0732 / +0.1215 | +0.0974 | [+0.0589, +0.1447] | 0.0405 (0.0405) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0272 / -0.0279 | -0.0276 | [-0.0468, -0.0102] | 0.0009 (0.0009) | BETTER |
| gap bias 4.0 s (m) | +1.0304 / +1.2013 | +1.1158 | [+0.9277, +1.3332] | 0.2635 (0.2635) | WORSE |
| gap bias 6.0 s (m) | +3.3506 / +4.3689 | +3.8597 | [+3.0518, +4.8085] | 1.2896 (1.2896) | WORSE |
| gap error 2.0 s (m) | +0.0274 / +0.0267 | +0.0270 | [+0.0137, +0.0421] | 0.0141 (0.0141) | WORSE |
| gap error 4.0 s (m) | +0.5427 / +0.5810 | +0.5619 | [+0.3830, +0.7616] | 0.0535 (0.0535) | WORSE |
| gap error 6.0 s (m) | +1.3434 / +1.8469 | +1.5951 | [+0.9079, +2.4037] | 0.4280 (0.4280) | WORSE |
| time-gap error 2.0 s (s) | +0.0261 / +0.0494 | +0.0378 | [-0.0009, +0.0912] | 0.0151 (0.0151) | OPEN |
| time-gap error over the scored steps (s) | +0.8097 / +1.0529 | +0.9313 | [+0.5539, +1.3547] | 0.2327 (0.2327) | WORSE |

## I10RA R0Hb20 minus I10B0 R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0302 / +0.0160 | +0.0231 | [+0.0072, +0.0427] | 0.0338 (0.0338) | INSIDE-SPREAD |
| C contact rate | +0.0218 / +0.0171 | +0.0195 | [+0.0058, +0.0360] | 0.0140 (0.0140) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0343 / -0.0327 | -0.0335 | [-0.0532, -0.0162] | 0.0405 (0.0405) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0272 / -0.0279 | -0.0276 | [-0.0468, -0.0102] | 0.0009 (0.0009) | BETTER |
| gap bias 4.0 s (m) | -0.2570 / -0.2666 | -0.2618 | [-0.3754, -0.1513] | 0.2635 (0.2635) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -1.5904 / -1.6021 | -1.5962 | [-2.0572, -1.1709] | 1.2896 (1.2896) | BETTER |
| gap error 2.0 s (m) | +0.0274 / +0.0267 | +0.0270 | [+0.0137, +0.0421] | 0.0141 (0.0141) | WORSE |
| gap error 4.0 s (m) | +0.0751 / -0.0008 | +0.0371 | [-0.0658, +0.1560] | 0.0535 (0.0535) | OPEN |
| gap error 6.0 s (m) | -0.1266 / -0.5708 | -0.3487 | [-0.8323, +0.1524] | 0.4280 (0.4280) | OPEN |
| time-gap error 2.0 s (s) | +0.0261 / +0.0494 | +0.0378 | [-0.0009, +0.0912] | 0.0151 (0.0151) | OPEN |
| time-gap error over the scored steps (s) | -0.5850 / -0.6248 | -0.6049 | [-0.8662, -0.3452] | 0.2327 (0.2327) | BETTER |

## I10RA R0C10Hb20 minus I10B0 R0C10Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0338 / +0.0196 | +0.0267 | [+0.0096, +0.0499] | 0.0338 (0.0338) | INSIDE-SPREAD |
| C contact rate | +0.0296 / +0.0171 | +0.0234 | [+0.0089, +0.0417] | 0.0140 (0.0140) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0919 / -0.0763 | -0.0841 | [-0.1244, -0.0514] | 0.0405 (0.0405) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0272 / -0.0279 | -0.0276 | [-0.0468, -0.0102] | 0.0009 (0.0009) | BETTER |
| gap bias 4.0 s (m) | -0.4595 / -0.5138 | -0.4867 | [-0.6439, -0.3364] | 0.2635 (0.2635) | BETTER |
| gap bias 6.0 s (m) | -2.5340 / -2.5001 | -2.5170 | [-3.2803, -1.8070] | 1.2896 (1.2896) | BETTER |
| gap error 2.0 s (m) | +0.0274 / +0.0267 | +0.0270 | [+0.0137, +0.0421] | 0.0141 (0.0141) | WORSE |
| gap error 4.0 s (m) | -0.0293 / -0.1194 | -0.0744 | [-0.2199, +0.0683] | 0.0535 (0.0535) | OPEN |
| gap error 6.0 s (m) | -1.2054 / -1.5352 | -1.3703 | [-2.1516, -0.6916] | 0.4280 (0.4280) | BETTER |
| time-gap error 2.0 s (s) | +0.0261 / +0.0494 | +0.0378 | [-0.0009, +0.0912] | 0.0151 (0.0151) | OPEN |
| time-gap error over the scored steps (s) | -0.8952 / -0.7097 | -0.8024 | [-1.1718, -0.5060] | 0.2327 (0.2327) | BETTER |

## I10RA R0G50Hb20 minus I10B0 R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0196 / +0.0036 | +0.0116 | [-0.0020, +0.0305] | 0.0338 (0.0338) | OPEN |
| C contact rate | +0.0109 / +0.0109 | +0.0109 | [+0.0023, +0.0214] | 0.0140 (0.0140) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0187 / -0.0249 | -0.0218 | [-0.0403, -0.0054] | 0.0405 (0.0405) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0272 / -0.0279 | -0.0276 | [-0.0468, -0.0102] | 0.0009 (0.0009) | BETTER |
| gap bias 4.0 s (m) | -0.1512 / -0.2232 | -0.1872 | [-0.2977, -0.0776] | 0.2635 (0.2635) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.8643 / -0.8465 | -0.8554 | [-1.1573, -0.5641] | 1.2896 (1.2896) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0274 / +0.0267 | +0.0270 | [+0.0137, +0.0421] | 0.0141 (0.0141) | WORSE |
| gap error 4.0 s (m) | +0.0762 / +0.0528 | +0.0645 | [-0.0158, +0.1523] | 0.0535 (0.0535) | OPEN |
| gap error 6.0 s (m) | -0.1652 / -0.1813 | -0.1733 | [-0.4626, +0.0969] | 0.4280 (0.4280) | OPEN |
| time-gap error 2.0 s (s) | +0.0261 / +0.0494 | +0.0378 | [-0.0009, +0.0912] | 0.0151 (0.0151) | OPEN |
| time-gap error over the scored steps (s) | -0.4161 / -0.1318 | -0.2739 | [-0.5721, -0.0398] | 0.2327 (0.2327) | BETTER |

## I10RA R0G0Hb20 minus I10B0 R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0071 / +0.0053 | +0.0062 | [+0.0000, +0.0139] | 0.0338 (0.0338) | OPEN |
| C contact rate | +0.0016 / +0.0047 | +0.0031 | [-0.0021, +0.0095] | 0.0140 (0.0140) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0093 / -0.0047 | -0.0070 | [-0.0172, +0.0023] | 0.0405 (0.0405) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0272 / -0.0279 | -0.0276 | [-0.0468, -0.0102] | 0.0009 (0.0009) | BETTER |
| gap bias 4.0 s (m) | -0.0655 / -0.1023 | -0.0839 | [-0.1586, -0.0107] | 0.2635 (0.2635) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.2722 / -0.2220 | -0.2471 | [-0.4173, -0.0741] | 1.2896 (1.2896) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0274 / +0.0267 | +0.0270 | [+0.0137, +0.0421] | 0.0141 (0.0141) | WORSE |
| gap error 4.0 s (m) | +0.0747 / +0.0964 | +0.0856 | [+0.0296, +0.1386] | 0.0535 (0.0535) | WORSE |
| gap error 6.0 s (m) | -0.0127 / +0.1338 | +0.0605 | [-0.1182, +0.2307] | 0.4280 (0.4280) | OPEN |
| time-gap error 2.0 s (s) | +0.0261 / +0.0494 | +0.0378 | [-0.0009, +0.0912] | 0.0151 (0.0151) | OPEN |
| time-gap error over the scored steps (s) | -0.0360 / -0.0536 | -0.0448 | [-0.1551, +0.0493] | 0.2327 (0.2327) | OPEN |

## I10RA R0KHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0089 / +0.0000 | +0.0044 | [-0.0024, +0.0128] | 0.0338 (0.0338) | OPEN |
| C contact rate | +0.0031 / +0.0031 | +0.0031 | [-0.0021, +0.0087] | 0.0140 (0.0140) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0109 / -0.0031 | -0.0070 | [-0.0167, +0.0026] | 0.0405 (0.0405) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0272 / -0.0279 | -0.0276 | [-0.0468, -0.0102] | 0.0009 (0.0009) | BETTER |
| gap bias 4.0 s (m) | -0.0618 / -0.0871 | -0.0745 | [-0.1439, -0.0079] | 0.2635 (0.2635) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.1730 / -0.2207 | -0.1969 | [-0.3570, -0.0545] | 1.2896 (1.2896) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0274 / +0.0267 | +0.0270 | [+0.0137, +0.0421] | 0.0141 (0.0141) | WORSE |
| gap error 4.0 s (m) | +0.0689 / +0.0881 | +0.0785 | [+0.0279, +0.1315] | 0.0535 (0.0535) | WORSE |
| gap error 6.0 s (m) | +0.0436 / +0.0778 | +0.0607 | [-0.0605, +0.1865] | 0.4280 (0.4280) | OPEN |
| time-gap error 2.0 s (s) | +0.0261 / +0.0494 | +0.0378 | [-0.0009, +0.0912] | 0.0151 (0.0151) | OPEN |
| time-gap error over the scored steps (s) | -0.0228 / -0.1034 | -0.0631 | [-0.1902, +0.0346] | 0.2327 (0.2327) | OPEN |

## I10RA R0Hb20 minus I10RH R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0089 / -0.0036 | +0.0027 | [-0.0081, +0.0132] | 0.0338 (0.0338) | OPEN |
| C contact rate | +0.0047 / +0.0031 | +0.0039 | [-0.0016, +0.0110] | 0.0140 (0.0140) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0000 / +0.0125 | +0.0062 | [-0.0071, +0.0176] | 0.0405 (0.0405) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0031 / +0.0117 | +0.0074 | [-0.0022, +0.0154] | 0.0009 (0.0009) | OPEN |
| gap bias 4.0 s (m) | -0.0269 / +0.0981 | +0.0356 | [-0.0629, +0.1213] | 0.2635 (0.2635) | OPEN |
| gap bias 6.0 s (m) | -0.1443 / +0.2080 | +0.0319 | [-0.2668, +0.3037] | 1.2896 (1.2896) | OPEN |
| gap error 2.0 s (m) | -0.0042 / +0.0047 | +0.0003 | [-0.0087, +0.0068] | 0.0141 (0.0141) | OPEN |
| gap error 4.0 s (m) | -0.0429 / +0.0397 | -0.0016 | [-0.0713, +0.0633] | 0.0535 (0.0535) | OPEN |
| gap error 6.0 s (m) | +0.0411 / +0.1157 | +0.0784 | [-0.0758, +0.2220] | 0.4280 (0.4280) | OPEN |
| time-gap error 2.0 s (s) | -0.0010 / +0.0018 | +0.0004 | [-0.0158, +0.0182] | 0.0151 (0.0151) | OPEN |
| time-gap error over the scored steps (s) | -0.0500 / -0.0606 | -0.0553 | [-0.1464, +0.0400] | 0.2327 (0.2327) | OPEN |

## I10RA R0C10XHb20 minus I10RH R0C10XHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0036 / +0.0036 | +0.0036 | [-0.0036, +0.0123] | 0.0338 (0.0338) | OPEN |
| C contact rate | +0.0000 / -0.0047 | -0.0023 | [-0.0094, +0.0024] | 0.0140 (0.0140) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0093 / +0.0062 | -0.0016 | [-0.0164, +0.0135] | 0.0405 (0.0405) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0031 / +0.0117 | +0.0074 | [-0.0022, +0.0154] | 0.0009 (0.0009) | OPEN |
| gap bias 4.0 s (m) | -0.0124 / +0.1072 | +0.0474 | [-0.0502, +0.1312] | 0.2635 (0.2635) | OPEN |
| gap bias 6.0 s (m) | -0.0127 / +0.2830 | +0.1351 | [-0.1075, +0.3666] | 1.2896 (1.2896) | OPEN |
| gap error 2.0 s (m) | -0.0042 / +0.0047 | +0.0003 | [-0.0087, +0.0068] | 0.0141 (0.0141) | OPEN |
| gap error 4.0 s (m) | -0.0597 / +0.0663 | +0.0033 | [-0.0892, +0.0892] | 0.0535 (0.0535) | OPEN |
| gap error 6.0 s (m) | +0.0182 / +0.2495 | +0.1338 | [-0.0600, +0.3194] | 0.4280 (0.4280) | OPEN |
| time-gap error 2.0 s (s) | -0.0010 / +0.0018 | +0.0004 | [-0.0158, +0.0182] | 0.0151 (0.0151) | OPEN |
| time-gap error over the scored steps (s) | -0.0739 / +0.0023 | -0.0358 | [-0.1476, +0.0954] | 0.2327 (0.2327) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10B0_R0C10Hb20 (0,1) | 0.0329 [0.0136, 0.0571] | 0.0195 [0.0060, 0.0360] | 0.2882 [0.2316, 0.3454] | 0.0891 [0.0465, 0.1353] | 2.1520 [1.6960, 2.6073] | 7.6474 [6.1154, 9.2291] | 0.4317 [0.3899, 0.4802] | 2.9594 [2.5948, 3.3607] | 8.7568 [7.5255, 10.0816] | 0.8992 [0.6263, 1.2159] | 3.7619 [2.7922, 4.7955] |
| I10B0_R0C10XHb20 (0,1) | 0.0294 [0.0104, 0.0534] | 0.0210 [0.0073, 0.0374] | 0.2796 [0.2236, 0.3374] | 0.0891 [0.0465, 0.1353] | 2.1882 [1.7296, 2.6381] | 7.4618 [6.0483, 8.9499] | 0.4317 [0.3899, 0.4802] | 2.9698 [2.6130, 3.3670] | 8.6467 [7.5117, 9.8733] | 0.8992 [0.6263, 1.2159] | 3.8517 [2.8854, 4.8691] |
| I10B0_R0G0Hb20 (0,1) | 0.0943 [0.0578, 0.1338] | 0.0592 [0.0296, 0.0938] | 0.1176 [0.0805, 0.1614] | 0.0891 [0.0465, 0.1353] | 0.6899 [0.3114, 1.0864] | 1.8093 [0.6138, 3.0721] | 0.4317 [0.3899, 0.4802] | 2.3802 [2.0930, 2.7083] | 6.0377 [5.2686, 6.8989] | 0.8992 [0.6263, 1.2159] | 2.1807 [1.5120, 2.9252] |
| I10B0_R0G50Hb20 (0,1) | 0.0605 [0.0351, 0.0901] | 0.0389 [0.0150, 0.0669] | 0.1900 [0.1424, 0.2401] | 0.0891 [0.0465, 0.1353] | 1.4309 [1.0198, 1.8362] | 4.7275 [3.3100, 6.1037] | 0.4317 [0.3899, 0.4802] | 2.6107 [2.2874, 2.9683] | 7.0231 [6.0954, 8.0773] | 0.8992 [0.6263, 1.2159] | 3.1028 [2.2323, 4.0349] |
| I10B0_R0Hb20 (0,1) | 0.0614 [0.0289, 0.0956] | 0.0280 [0.0116, 0.0479] | 0.2033 [0.1503, 0.2595] | 0.0891 [0.0465, 0.1353] | 1.4575 [0.9911, 1.9165] | 5.2877 [3.8033, 6.8244] | 0.4317 [0.3899, 0.4802] | 2.6569 [2.3239, 3.0194] | 7.3768 [6.2706, 8.5258] | 0.8992 [0.6263, 1.2159] | 3.1310 [2.2928, 4.0200] |
| I10B0_R0KHb20 (0,1) | 0.0925 [0.0563, 0.1324] | 0.0600 [0.0297, 0.0927] | 0.1145 [0.0783, 0.1567] | 0.0891 [0.0465, 0.1353] | 0.7204 [0.3444, 1.1110] | 1.7458 [0.5279, 3.0407] | 0.4317 [0.3899, 0.4802] | 2.3660 [2.0834, 2.6885] | 6.0590 [5.2991, 6.9059] | 0.8992 [0.6263, 1.2159] | 2.1774 [1.5055, 2.9188] |
| I10RA_R0C10Hb20 (0,1) | 0.0596 [0.0282, 0.0998] | 0.0428 [0.0185, 0.0742] | 0.2040 [0.1515, 0.2604] | 0.0615 [0.0090, 0.1192] | 1.6654 [1.1954, 2.1488] | 5.1304 [3.5926, 6.6473] | 0.4587 [0.4092, 0.5114] | 2.8850 [2.5221, 3.2899] | 7.3865 [6.4438, 8.4452] | 0.9370 [0.6490, 1.2744] | 2.9594 [2.1690, 3.7841] |
| I10RA_R0C10XHb20 (0,1) | 0.0507 [0.0244, 0.0842] | 0.0335 [0.0127, 0.0598] | 0.2118 [0.1596, 0.2677] | 0.0615 [0.0090, 0.1192] | 1.8362 [1.3694, 2.3068] | 5.6056 [4.1754, 7.0447] | 0.4587 [0.4092, 0.5114] | 2.9278 [2.5579, 3.3488] | 7.6542 [6.7557, 8.7120] | 0.9370 [0.6490, 1.2744] | 3.1087 [2.2871, 3.9726] |
| I10RA_R0G0Hb20 (0,1) | 0.1005 [0.0609, 0.1438] | 0.0623 [0.0321, 0.0963] | 0.1106 [0.0741, 0.1532] | 0.0615 [0.0090, 0.1192] | 0.6060 [0.2088, 1.0422] | 1.5622 [0.3450, 2.8797] | 0.4587 [0.4092, 0.5114] | 2.4657 [2.1552, 2.7938] | 6.0982 [5.3281, 6.9690] | 0.9370 [0.6490, 1.2744] | 2.1358 [1.4870, 2.8638] |
| I10RA_R0G50Hb20 (0,1) | 0.0721 [0.0394, 0.1136] | 0.0498 [0.0229, 0.0811] | 0.1682 [0.1224, 0.2172] | 0.0615 [0.0090, 0.1192] | 1.2437 [0.8211, 1.6881] | 3.8721 [2.4228, 5.2664] | 0.4587 [0.4092, 0.5114] | 2.6752 [2.3374, 3.0492] | 6.8498 [5.9772, 7.8459] | 0.9370 [0.6490, 1.2744] | 2.8289 [2.0158, 3.7327] |
| I10RA_R0Hb20 (0,1) | 0.0845 [0.0429, 0.1329] | 0.0475 [0.0229, 0.0780] | 0.1698 [0.1168, 0.2269] | 0.0615 [0.0090, 0.1192] | 1.1957 [0.7156, 1.7054] | 3.6915 [2.1146, 5.2792] | 0.4587 [0.4092, 0.5114] | 2.6940 [2.3599, 3.0702] | 7.0281 [6.1222, 8.0628] | 0.9370 [0.6490, 1.2744] | 2.5261 [1.8344, 3.2552] |
| I10RA_R0KHb20 (0,1) | 0.0970 [0.0585, 0.1400] | 0.0631 [0.0320, 0.0978] | 0.1075 [0.0721, 0.1473] | 0.0615 [0.0090, 0.1192] | 0.6459 [0.2501, 1.0764] | 1.5490 [0.3063, 2.8769] | 0.4587 [0.4092, 0.5114] | 2.4445 [2.1370, 2.7731] | 6.1197 [5.3496, 6.9738] | 0.9370 [0.6490, 1.2744] | 2.1143 [1.4721, 2.8375] |
| I10RH_R0C10XHb20 (0,1) | 0.0472 [0.0221, 0.0795] | 0.0358 [0.0137, 0.0613] | 0.2134 [0.1621, 0.2687] | 0.0541 [-0.0003, 0.1137] | 1.7888 [1.3557, 2.2578] | 5.4704 [4.1477, 6.8065] | 0.4584 [0.4066, 0.5113] | 2.9245 [2.5330, 3.3539] | 7.5203 [6.6391, 8.5576] | 0.9366 [0.6450, 1.2757] | 3.1445 [2.3022, 4.0131] |
| I10RH_R0Hb20 (0,1) | 0.0819 [0.0438, 0.1239] | 0.0436 [0.0206, 0.0728] | 0.1636 [0.1163, 0.2142] | 0.0541 [-0.0003, 0.1137] | 1.1601 [0.7080, 1.6234] | 3.6596 [2.2695, 5.1805] | 0.4584 [0.4066, 0.5113] | 2.6956 [2.3561, 3.0709] | 6.9497 [6.0649, 7.9524] | 0.9366 [0.6450, 1.2757] | 2.5814 [1.8594, 3.3407] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
