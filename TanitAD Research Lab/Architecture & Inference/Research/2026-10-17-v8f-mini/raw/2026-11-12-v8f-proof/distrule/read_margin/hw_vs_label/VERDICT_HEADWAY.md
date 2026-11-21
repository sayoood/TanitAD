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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSMC:LSRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSMT:LSRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSMG:LSRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSML:LSRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSMH:LSRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDMC:LSDRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDMT:LSDRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDMG:LSDRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDML:LSDRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDMH:LSDRL", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, LSMC minus LSRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSMT minus LSRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSMG minus LSRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSML minus LSRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSMH minus LSRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDMC minus LSDRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDMT minus LSDRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDMG minus LSDRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDML minus LSDRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDMH minus LSDRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## LSMC minus LSRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0160 / +0.0000 | +0.0080 | [-0.0107, +0.0289] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0031 / +0.0109 | +0.0070 | [+0.0000, +0.0177] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0841 / +0.1791 | +0.1316 | [+0.0582, +0.2082] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.5947 / +1.3364 | +0.9656 | [+0.4991, +1.5366] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.3121 / +3.8632 | +3.0876 | [+1.7478, +4.5180] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +3.4283 / +6.2310 | +4.8296 | [+2.5310, +7.4107] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.4756 / +1.1503 | +0.8129 | [+0.3789, +1.3295] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.1786 / +3.7925 | +2.9855 | [+1.6527, +4.4712] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.4585 / +6.2535 | +4.8560 | [+2.6208, +7.4445] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.1977 / +0.8136 | +0.5057 | [+0.1622, +1.0009] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.9996 / +2.1919 | +1.5957 | [+0.8273, +2.5176] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSMT minus LSRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0338 / +0.0445 | +0.0391 | [+0.0197, +0.0623] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0187 / +0.0280 | +0.0234 | [+0.0086, +0.0426] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0140 / +0.0576 | +0.0358 | [+0.0029, +0.0770] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0664 / +0.2669 | +0.1003 | [-0.1042, +0.3869] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.3023 / +1.0488 | +0.6755 | [+0.0459, +1.5311] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.3200 / +1.2957 | +0.8078 | [-0.1655, +2.2545] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0237 / +0.2552 | +0.1158 | [-0.0656, +0.3624] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.3906 / +1.4046 | +0.8976 | [+0.1536, +1.8300] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.6006 / +2.0280 | +1.3143 | [+0.2109, +2.9360] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.0191 / +0.3285 | +0.1547 | [-0.0273, +0.4715] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0010 / +0.8013 | +0.4002 | [-0.0627, +1.0380] | 1.7849 (0.8089) | OPEN |

## LSMG minus LSRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0160 / +0.0000 | +0.0080 | [-0.0107, +0.0289] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0031 / +0.0109 | +0.0070 | [+0.0000, +0.0177] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0826 / +0.1791 | +0.1308 | [+0.0568, +0.2082] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.5947 / +1.3364 | +0.9656 | [+0.4991, +1.5366] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.3121 / +3.8632 | +3.0876 | [+1.7478, +4.5180] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +3.3870 / +6.2310 | +4.8090 | [+2.5030, +7.4107] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.4756 / +1.1503 | +0.8129 | [+0.3789, +1.3295] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.1786 / +3.7925 | +2.9855 | [+1.6527, +4.4712] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.4236 / +6.2535 | +4.8385 | [+2.5984, +7.4445] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.1977 / +0.8136 | +0.5057 | [+0.1622, +1.0009] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.0002 / +2.1919 | +1.5960 | [+0.8276, +2.5176] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSML minus LSRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0285 / +0.0338 | +0.0311 | [+0.0146, +0.0497] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0156 / +0.0265 | +0.0210 | [+0.0074, +0.0395] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0467 / +0.0685 | +0.0576 | [+0.0230, +0.0918] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1070 / +0.3784 | +0.2427 | [+0.0605, +0.5069] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.0108 / +1.1814 | +1.0961 | [+0.5536, +1.7075] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.6582 / +1.3951 | +1.5266 | [+0.3928, +2.6509] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.1053 / +0.3439 | +0.2246 | [+0.0580, +0.4637] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.0455 / +1.4654 | +1.2554 | [+0.7124, +1.8951] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +1.9369 / +2.0287 | +1.9828 | [+0.9459, +3.0586] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0002 / +0.3553 | +0.1777 | [-0.0089, +0.4864] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.1129 / +0.9091 | +0.5110 | [+0.0508, +1.1306] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSMH minus LSRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0214 / -0.0267 | -0.0240 | [-0.0485, -0.0038] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0000 / +0.0031 | +0.0016 | [+0.0000, +0.0053] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.2040 / +0.3115 | +0.2578 | [+0.1676, +0.3442] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +2.0421 / +2.7952 | +2.4187 | [+1.8510, +3.0829] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +5.9168 / +7.6469 | +6.7818 | [+5.0882, +8.6356] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +9.3916 / +10.9259 | +10.1587 | [+7.6076, +12.6196] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.7219 / +2.4321 | +2.0770 | [+1.5259, +2.7008] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +5.4274 / +7.2630 | +6.3452 | [+4.6068, +8.2530] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +8.8095 / +10.6043 | +9.7069 | [+7.0858, +12.3313] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.7421 / +1.1671 | +0.9546 | [+0.6366, +1.3679] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +2.6594 / +3.8777 | +3.2685 | [+2.3589, +4.3186] | 1.7849 (0.8089) | WORSE |

## LSDMC minus LSDRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0125 / -0.0036 | +0.0044 | [-0.0156, +0.0223] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0016 / +0.0171 | +0.0093 | [+0.0000, +0.0261] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0888 / +0.0919 | +0.0903 | [+0.0594, +0.1219] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.9109 / +0.8719 | +0.8914 | [+0.6482, +1.1514] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.6467 / +2.5883 | +2.6175 | [+1.7751, +3.5574] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +3.7225 / +3.7375 | +3.7300 | [+2.3417, +5.2264] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.7725 / +0.7809 | +0.7767 | [+0.5651, +1.0132] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.6810 / +2.6591 | +2.6700 | [+1.8432, +3.5994] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.7314 / +3.9586 | +3.8450 | [+2.4838, +5.3457] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.3283 / +0.3394 | +0.3339 | [+0.2117, +0.4976] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.9831 / +1.0355 | +1.0093 | [+0.6853, +1.4121] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDMT minus LSDRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0712 / +0.0676 | +0.0694 | [+0.0412, +0.0997] | 0.0463 (0.0214) | WORSE |
| C contact rate | +0.0140 / +0.0530 | +0.0335 | [+0.0072, +0.0708] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0125 / +0.0031 | +0.0078 | [-0.0166, +0.0345] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0248 / +0.0008 | -0.0120 | [-0.1176, +0.0942] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.2539 / +0.2307 | +0.2423 | [-0.1896, +0.7201] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.2994 / -0.2243 | +0.0376 | [-0.8306, +1.0162] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0028 / +0.0244 | +0.0108 | [-0.0752, +0.0913] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.5004 / +0.6855 | +0.5930 | [+0.2015, +1.0473] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.6069 / +0.6341 | +0.6205 | [-0.1306, +1.4511] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0176 / +0.0052 | -0.0062 | [-0.0346, +0.0329] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0053 / +0.2098 | +0.1075 | [-0.0533, +0.3592] | 1.7849 (0.8089) | OPEN |

## LSDMG minus LSDRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0142 / +0.0036 | +0.0089 | [-0.0104, +0.0263] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0016 / +0.0171 | +0.0093 | [+0.0000, +0.0261] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0888 / +0.0903 | +0.0896 | [+0.0592, +0.1205] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.8934 / +0.8592 | +0.8763 | [+0.6389, +1.1373] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.6410 / +2.5145 | +2.5778 | [+1.7217, +3.5289] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +3.7266 / +3.6575 | +3.6921 | [+2.3099, +5.1562] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.7582 / +0.7682 | +0.7632 | [+0.5548, +0.9981] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.6787 / +2.6540 | +2.6664 | [+1.8394, +3.5946] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.7443 / +3.9639 | +3.8541 | [+2.4879, +5.3430] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.3268 / +0.3387 | +0.3328 | [+0.2110, +0.4965] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.9819 / +1.0310 | +1.0064 | [+0.6834, +1.4091] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDML minus LSDRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0356 / +0.0320 | +0.0338 | [+0.0131, +0.0579] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0093 / +0.0327 | +0.0210 | [+0.0042, +0.0477] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0421 / +0.0218 | +0.0319 | [+0.0052, +0.0647] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2610 / +0.2402 | +0.2506 | [+0.1008, +0.4129] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.0667 / +0.6341 | +0.8504 | [+0.2508, +1.4681] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.5234 / +0.8347 | +1.1790 | [+0.2546, +2.2364] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.2000 / +0.2258 | +0.2129 | [+0.0953, +0.3538] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.2277 / +0.9753 | +1.1015 | [+0.6214, +1.6711] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +1.6733 / +1.4193 | +1.5463 | [+0.7171, +2.4673] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0327 / +0.0782 | +0.0555 | [+0.0076, +0.1313] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.3706 / +0.3437 | +0.3571 | [+0.1422, +0.6173] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDMH minus LSDRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0107 / -0.0356 | -0.0231 | [-0.0569, -0.0008] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0000 / +0.0031 | +0.0016 | [+0.0000, +0.0051] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.2134 / +0.2461 | +0.2298 | [+0.1787, +0.2791] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +2.2943 / +2.3332 | +2.3137 | [+1.9454, +2.7015] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +5.9357 / +6.4714 | +6.2035 | [+4.9509, +7.4966] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +8.5231 / +8.1942 | +8.3587 | [+6.5233, +10.1289] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.9965 / +2.0038 | +2.0002 | [+1.6364, +2.3733] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +5.7132 / +6.0435 | +5.8783 | [+4.6366, +7.1269] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +8.4049 / +7.8225 | +8.1137 | [+6.2423, +9.9437] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.9035 / +0.7701 | +0.8368 | [+0.6127, +1.0850] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +2.8706 / +2.4931 | +2.6818 | [+2.1044, +3.3493] | 1.7849 (0.8089) | WORSE |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LSDMC (0,1) | 0.0516 [0.0277, 0.0823] | 0.0093 [0.0000, 0.0261] | 0.4097 [0.3379, 0.4816] | 1.1278 [0.7191, 1.5873] | 5.4852 [3.9211, 7.1340] | 13.6403 [10.7057, 16.7188] | 1.5814 [1.2109, 2.0053] | 6.3777 [4.9718, 7.8509] | 14.6549 [12.0072, 17.5735] | 1.8397 [1.1472, 2.5956] | 3.6730 [2.6322, 4.7572] |
| LSDMG (0,1) | 0.0560 [0.0301, 0.0912] | 0.0093 [0.0000, 0.0261] | 0.4089 [0.3371, 0.4816] | 1.1127 [0.7039, 1.5713] | 5.4454 [3.8589, 7.0812] | 13.6024 [10.6442, 16.6801] | 1.5679 [1.1919, 1.9927] | 6.3741 [4.9779, 7.8531] | 14.6640 [12.0121, 17.5946] | 1.8386 [1.1472, 2.5930] | 3.6701 [2.6248, 4.7552] |
| LSDMH (0,1) | 0.0240 [0.0113, 0.0380] | 0.0016 [0.0000, 0.0051] | 0.5491 [0.4748, 0.6103] | 2.5501 [2.0222, 3.1061] | 9.0712 [7.3524, 10.7473] | 18.2690 [14.9898, 21.3254] | 2.8048 [2.2958, 3.3284] | 9.5860 [7.9233, 11.2412] | 18.9236 [15.7737, 21.8890] | 2.3426 [1.5567, 3.2238] | 5.3455 [4.1002, 6.7545] |
| LSDML (0,1) | 0.0810 [0.0470, 0.1248] | 0.0210 [0.0042, 0.0477] | 0.3512 [0.2702, 0.4347] | 0.4870 [0.1888, 0.8437] | 3.7180 [2.3627, 5.0573] | 11.0894 [8.5720, 13.5416] | 1.0176 [0.7520, 1.3406] | 4.8092 [3.7034, 5.9425] | 12.3563 [10.2458, 14.5408] | 1.5613 [0.9175, 2.2785] | 3.0208 [1.9841, 4.1141] |
| LSDMT (0,1) | 0.1165 [0.0746, 0.1633] | 0.0335 [0.0072, 0.0708] | 0.3271 [0.2509, 0.4032] | 0.2244 [-0.0011, 0.5060] | 3.1099 [2.0245, 4.2360] | 9.9479 [7.5280, 12.1979] | 0.8155 [0.6068, 1.0734] | 4.3006 [3.4383, 5.2380] | 11.4305 [9.5443, 13.5194] | 1.4996 [0.8583, 2.2313] | 2.7712 [1.8700, 3.7206] |
| LSDRL (0,1) | 0.0472 [0.0227, 0.0829] | 0.0000 [0.0000, 0.0000] | 0.3193 [0.2469, 0.3910] | 0.2364 [-0.0076, 0.5420] | 2.8677 [1.8890, 3.9087] | 9.9103 [7.7746, 11.9708] | 0.8047 [0.5833, 1.0843] | 3.7077 [2.8459, 4.6228] | 10.8099 [8.9499, 12.7615] | 1.5058 [0.8571, 2.2451] | 2.6637 [1.6935, 3.7219] |
| LSMC (0,1) | 0.0534 [0.0312, 0.0772] | 0.0070 [0.0000, 0.0177] | 0.4058 [0.3276, 0.4795] | 1.1719 [0.5946, 1.8411] | 5.6125 [4.1334, 7.2170] | 13.5297 [10.7494, 16.4204] | 1.6390 [1.0908, 2.2664] | 6.4847 [5.0902, 8.0342] | 14.6582 [12.0491, 17.4352] | 1.7826 [1.3123, 2.2538] | 4.1701 [3.3443, 5.0752] |
| LSMG (0,1) | 0.0534 [0.0312, 0.0772] | 0.0070 [0.0000, 0.0177] | 0.4050 [0.3275, 0.4776] | 1.1719 [0.5946, 1.8411] | 5.6125 [4.1334, 7.2170] | 13.5090 [10.7492, 16.4188] | 1.6390 [1.0908, 2.2664] | 6.4847 [5.0902, 8.0342] | 14.6407 [12.0428, 17.4009] | 1.7826 [1.3123, 2.2538] | 4.1704 [3.3447, 5.0753] |
| LSMH (0,1) | 0.0214 [0.0121, 0.0331] | 0.0016 [0.0000, 0.0053] | 0.5319 [0.4595, 0.5982] | 2.6250 [1.9439, 3.3623] | 9.3066 [7.4985, 11.3251] | 18.8588 [15.4101, 22.0929] | 2.9031 [2.2414, 3.6215] | 9.8444 [8.0697, 11.8615] | 19.5090 [16.2516, 22.6837] | 2.2315 [1.7206, 2.7620] | 5.8429 [4.9456, 6.7637] |
| LSML (0,1) | 0.0765 [0.0471, 0.1121] | 0.0210 [0.0074, 0.0395] | 0.3318 [0.2356, 0.4329] | 0.4491 [0.1264, 0.8341] | 3.6209 [2.4838, 4.8864] | 10.2267 [7.5575, 13.1692] | 1.0507 [0.7526, 1.4125] | 4.7545 [3.8012, 5.8417] | 11.7850 [9.4929, 14.3969] | 1.4547 [1.0080, 1.9167] | 3.0853 [2.2624, 3.9878] |
| LSMT (0,1) | 0.0845 [0.0527, 0.1232] | 0.0234 [0.0086, 0.0426] | 0.3100 [0.2259, 0.4009] | 0.3066 [0.0047, 0.6708] | 3.2004 [2.2391, 4.2372] | 9.5079 [7.0923, 11.8941] | 0.9418 [0.6696, 1.2884] | 4.3968 [3.5541, 5.3752] | 11.1165 [9.1719, 13.2105] | 1.4316 [0.9801, 1.9035] | 2.9745 [2.1468, 3.8918] |
| LSRL (0,1) | 0.0454 [0.0238, 0.0728] | 0.0000 [0.0000, 0.0000] | 0.2741 [0.1700, 0.3895] | 0.2064 [-0.0545, 0.5267] | 2.5248 [1.6354, 3.5112] | 8.7001 [6.1813, 11.2690] | 0.8261 [0.5991, 1.1154] | 3.4991 [2.7134, 4.3601] | 9.8022 [7.5079, 12.2263] | 1.2769 [0.7638, 1.8367] | 2.5743 [1.5172, 3.8082] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
