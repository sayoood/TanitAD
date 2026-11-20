# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: /workspace/v8proof/podread/data/join_eval141.npz. Alignment reference (ego speed): /workspace/v8proof/podread/data/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 22, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.03202846975088969, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0Hb20:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0KHb20:I10R0G0X", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0KHb10:I10R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0KHb20:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcKF5Hb20:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50KF5Hb20:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcKF5Hb20:I10RLcKF5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50KF5Hb20:I10RQ50KF5", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10R0Hb20 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0KHb20 minus I10R0G0X over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0KHb10 minus I10R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0KHb20 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcKF5Hb20 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50KF5Hb20 minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcKF5Hb20 minus I10RLcKF5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50KF5Hb20 minus I10RQ50KF5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10R0Hb20 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0338 / -0.0427 | -0.0383 | [-0.0691, -0.0134] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0218 / -0.0062 | -0.0140 | [-0.0328, -0.0006] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0187 / -0.0031 | +0.0078 | [-0.0095, +0.0264] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.0253 / -0.1644 | -0.0949 | [-0.2026, -0.0060] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.0077 / +0.1972 | +0.6025 | [-0.0310, +1.4349] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.0156 / -0.1259 | -0.0708 | [-0.1588, -0.0064] | 0.2417 (0.2417) | INSIDE-SPREAD |
| gap error 6.0 s (m) | -0.0788 / -0.3182 | -0.1985 | [-0.4362, +0.0093] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.2217 / +0.2362 | +0.2290 | [-0.1780, +0.5708] | 0.5140 (0.5140) | OPEN |

## I10R0KHb20 minus I10R0G0X

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0089 / +0.0018 | -0.0036 | [-0.0140, +0.0060] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0047 / -0.0031 | -0.0039 | [-0.0091, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0125 / +0.0031 | +0.0078 | [-0.0001, +0.0164] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.0891 / -0.1924 | -0.1408 | [-0.2340, -0.0687] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.3556 / +0.1174 | +0.2365 | [+0.0583, +0.4688] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.0577 / -0.1488 | -0.1032 | [-0.1843, -0.0400] | 0.2417 (0.2417) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.1163 / +0.0061 | +0.0612 | [-0.0296, +0.1594] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.3854 / +0.3107 | +0.3480 | [+0.0262, +0.6692] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10R0KHb10 minus I10R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0018 / -0.0018 | -0.0018 | [-0.0045, +0.0000] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0016 / -0.0016 | +0.0000 | [-0.0030, +0.0032] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0047 / +0.0000 | +0.0023 | [-0.0036, +0.0086] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.0136 / +0.0018 | +0.0077 | [-0.0060, +0.0198] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +0.1355 / +0.0585 | +0.0970 | [+0.0268, +0.1719] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.0062 / +0.0099 | +0.0080 | [-0.0002, +0.0183] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +0.0954 / +0.0545 | +0.0750 | [+0.0204, +0.1383] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.1130 / +0.1680 | +0.1405 | [+0.0499, +0.2674] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10R0KHb20 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0427 / -0.0231 | -0.0329 | [-0.0743, +0.0090] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0202 / +0.0093 | -0.0055 | [-0.0353, +0.0223] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0000 / -0.0670 | -0.0335 | [-0.0885, +0.0185] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.1684 / -0.4206 | -0.1261 | [-0.6134, +0.3026] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | -0.2781 / -2.6175 | -1.4478 | [-3.3484, +0.4369] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.1938 / -0.3678 | -0.2808 | [-0.4956, -0.0748] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -0.3276 / -1.7922 | -1.0599 | [-2.3106, +0.0225] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.3079 / +0.0331 | +0.1705 | [-0.3256, +0.6553] | 0.5140 (0.5140) | OPEN |

## I10RLcKF5Hb20 minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0071 / +0.0036 | -0.0018 | [-0.0183, +0.0169] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0000 / -0.0031 | -0.0016 | [-0.0051, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0436 / -0.0826 | -0.0631 | [-0.1138, -0.0171] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0409 / +0.0453 | +0.0431 | [+0.0249, +0.0651] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.4861 / -0.5888 | -0.5374 | [-0.8397, -0.2298] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -2.6261 / -3.5927 | -3.1094 | [-4.3571, -1.7745] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0081 / +0.0090 | +0.0086 | [-0.0071, +0.0254] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.6718 / -0.5937 | -0.6327 | [-0.9198, -0.3861] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -2.0814 / -2.7368 | -2.4091 | [-3.5237, -1.2707] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.0160 / +0.0127 | +0.0143 | [+0.0029, +0.0288] | 0.0330 (0.0330) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.2016 / -0.0274 | +0.0871 | [-0.4076, +0.5703] | 0.5140 (0.5140) | OPEN |

## I10RQ50KF5Hb20 minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0125 / +0.0000 | -0.0062 | [-0.0365, +0.0279] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0000 / +0.0062 | +0.0031 | [-0.0083, +0.0167] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0530 / -0.1293 | -0.0911 | [-0.1565, -0.0274] | 0.0903 (0.0903) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1121 / +0.0663 | +0.0892 | [+0.0462, +0.1400] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | -0.5815 / -1.0803 | -0.8309 | [-1.3736, -0.3136] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -3.3330 / -4.5727 | -3.9528 | [-5.3897, -2.5015] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.0685 / +0.0283 | +0.0484 | [+0.0086, +0.0952] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | -0.7529 / -1.0652 | -0.9091 | [-1.3614, -0.4839] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -2.6638 / -3.6467 | -3.1552 | [-4.5170, -1.7989] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.0281 / +0.0071 | +0.0176 | [+0.0036, +0.0441] | 0.0330 (0.0330) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0606 / -0.2588 | -0.0991 | [-0.5902, +0.3978] | 0.5140 (0.5140) | OPEN |

## I10RLcKF5Hb20 minus I10RLcKF5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0093 / +0.0000 | +0.0047 | [-0.0035, +0.0131] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.0503 / -0.1154 | -0.0829 | [-0.1457, -0.0361] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.1247 / -0.0506 | +0.0371 | [-0.0760, +0.1682] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.0406 / -0.1007 | -0.0707 | [-0.1250, -0.0278] | 0.2417 (0.2417) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.1185 / -0.0582 | +0.0301 | [-0.0856, +0.1619] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.3903 / +0.3183 | +0.3543 | [+0.0363, +0.6669] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50KF5Hb20 minus I10RQ50KF5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0018 / +0.0018 | +0.0000 | [-0.0033, +0.0038] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0078 / -0.0031 | +0.0023 | [-0.0062, +0.0106] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.0651 / -0.1389 | -0.1020 | [-0.1774, -0.0451] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.0076 / -0.0905 | -0.0414 | [-0.1245, +0.0423] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.0486 / -0.1117 | -0.0801 | [-0.1456, -0.0311] | 0.2417 (0.2417) | INSIDE-SPREAD |
| gap error 6.0 s (m) | -0.0000 / -0.0817 | -0.0408 | [-0.1132, +0.0349] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.2815 / +0.2003 | +0.2409 | [-0.0586, +0.5262] | 0.5140 (0.5140) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10R0G0X (0,1) | 0.1005 [0.0646, 0.1408] | 0.0576 [0.0295, 0.0883] | 0.0989 [0.0615, 0.1416] | -0.2236 [-0.2757, -0.1682] | 0.4629 [-0.0807, 1.0171] | 1.1098 [-0.4325, 2.6695] | 0.4729 [0.4299, 0.5219] | 2.3145 [2.0317, 2.6434] | 5.7567 [5.0133, 6.5885] | 1.0435 [0.7059, 1.4212] | 1.8643 [1.2610, 2.5995] |
| I10R0Hb20 (0,1) | 0.0916 [0.0453, 0.1445] | 0.0452 [0.0224, 0.0734] | 0.1480 [0.0872, 0.2156] | -0.2236 [-0.2757, -0.1682] | 0.3534 [-0.3798, 1.0966] | 3.3965 [1.2253, 5.6225] | 0.4729 [0.4299, 0.5219] | 2.4213 [2.1439, 2.7423] | 6.6793 [5.5074, 7.8860] | 1.0435 [0.7059, 1.4212] | 2.2708 [1.6210, 2.9961] |
| I10R0KHb10 (0,1) | 0.0952 [0.0604, 0.1343] | 0.0537 [0.0259, 0.0853] | 0.1090 [0.0740, 0.1479] | -0.2236 [-0.2757, -0.1682] | 0.3298 [-0.1483, 0.8031] | 1.4433 [-0.0485, 2.9450] | 0.4729 [0.4299, 0.5219] | 2.2193 [1.9529, 2.5119] | 5.8929 [5.1383, 6.7389] | 1.0435 [0.7059, 1.4212] | 2.3528 [1.6461, 3.1638] |
| I10R0KHb20 (0,1) | 0.0970 [0.0608, 0.1366] | 0.0537 [0.0264, 0.0848] | 0.1067 [0.0705, 0.1471] | -0.2236 [-0.2757, -0.1682] | 0.3222 [-0.1594, 0.7965] | 1.3463 [-0.1516, 2.8420] | 0.4729 [0.4299, 0.5219] | 2.2113 [1.9446, 2.4997] | 5.8179 [5.0804, 6.6424] | 1.0435 [0.7059, 1.4212] | 2.2124 [1.5547, 2.9468] |
| I10RLc (0,1) | 0.0383 [0.0190, 0.0628] | 0.0016 [0.0000, 0.0051] | 0.2181 [0.1459, 0.2897] | 0.2229 [-0.0327, 0.5472] | 2.0938 [1.2319, 3.0706] | 6.7474 [4.5816, 8.9276] | 0.7885 [0.5579, 1.0945] | 3.2710 [2.6177, 4.1001] | 8.0958 [6.4712, 9.8010] | 1.0876 [0.7460, 1.4643] | 2.1535 [1.4770, 2.9189] |
| I10RLcKF5 (0,1) | 0.0365 [0.0166, 0.0590] | 0.0000 [0.0000, 0.0000] | 0.1503 [0.0957, 0.2124] | 0.2660 [-0.0005, 0.6041] | 1.6392 [0.9187, 2.4925] | 3.6010 [2.0988, 5.1610] | 0.7970 [0.5589, 1.1110] | 2.7089 [2.1777, 3.4010] | 5.6565 [4.6924, 6.8198] | 1.1020 [0.7679, 1.4781] | 1.8863 [1.2658, 2.6506] |
| I10RLcKF5Hb20 (0,1) | 0.0365 [0.0166, 0.0590] | 0.0000 [0.0000, 0.0000] | 0.1550 [0.1023, 0.2140] | 0.2660 [-0.0005, 0.6041] | 1.5564 [0.8610, 2.3908] | 3.6380 [2.1931, 5.1343] | 0.7970 [0.5589, 1.1110] | 2.6382 [2.1182, 3.3187] | 5.6866 [4.7784, 6.7972] | 1.1020 [0.7679, 1.4781] | 2.2406 [1.5686, 2.9970] |
| I10RQ50 (0,1) | 0.0801 [0.0480, 0.1174] | 0.0265 [0.0102, 0.0465] | 0.2593 [0.1862, 0.3304] | 0.2697 [0.0335, 0.5486] | 2.4645 [1.6073, 3.4225] | 7.6520 [5.6761, 9.6557] | 0.8241 [0.5966, 1.0699] | 3.7859 [3.0818, 4.6228] | 9.4150 [7.8273, 11.0990] | 1.1816 [0.8261, 1.5783] | 2.5241 [1.8524, 3.2957] |
| I10RQ50KF5 (0,1) | 0.0738 [0.0431, 0.1106] | 0.0296 [0.0105, 0.0525] | 0.1659 [0.1223, 0.2129] | 0.3589 [0.0841, 0.6833] | 1.7356 [1.1796, 2.3753] | 3.7406 [2.5599, 4.9933] | 0.8726 [0.6109, 1.1655] | 2.9570 [2.4927, 3.4815] | 6.3006 [5.4689, 7.2014] | 1.1992 [0.8358, 1.6032] | 2.1841 [1.5283, 2.9270] |
| I10RQ50KF5Hb20 (0,1) | 0.0738 [0.0427, 0.1106] | 0.0296 [0.0105, 0.0525] | 0.1682 [0.1245, 0.2169] | 0.3589 [0.0841, 0.6833] | 1.6336 [1.0778, 2.2703] | 3.6992 [2.5487, 4.9307] | 0.8726 [0.6109, 1.1655] | 2.8769 [2.4025, 3.4140] | 6.2598 [5.4336, 7.1778] | 1.1992 [0.8358, 1.6032] | 2.4250 [1.7255, 3.2065] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
