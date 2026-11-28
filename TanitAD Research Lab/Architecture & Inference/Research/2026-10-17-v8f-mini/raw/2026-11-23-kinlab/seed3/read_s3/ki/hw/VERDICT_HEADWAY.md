# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: F:/TanitAD-scratch/strat1/read/inputs/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 16, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.0752218477307355, "A_hi": 0.19026440814805612, "spread_max": 0.003558718861209967, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0C10XHb20:I10B0_R0C10XHb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0C10XHb20:I10KI_R0KHb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0G50Hb20:I10KI_R0G0Hb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0C10Hb20:I10KI_R0G50Hb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0C10XHb20:I10B0_R0KHb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0G50Hb20:I10B0_R0G0Hb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0C10Hb20:I10B0_R0G50Hb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0C10XHb20:I10B0_R0KHb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0Hb20:I10B0_R0Hb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0C10Hb20:I10B0_R0C10Hb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0G50Hb20:I10B0_R0G50Hb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0G0Hb20:I10B0_R0G0Hb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0KHb20:I10B0_R0KHb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0Hb20:I10RH_R0Hb20", "seeds": [2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10KI_R0C10XHb20:I10RH_R0C10XHb20", "seeds": [2]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10UEA_R0C10XHb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10KI_R0C10XHb20 minus I10B0_R0C10XHb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0C10XHb20 minus I10KI_R0KHb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0G50Hb20 minus I10KI_R0G0Hb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0C10Hb20 minus I10KI_R0G50Hb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0C10XHb20 minus I10B0_R0KHb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0G50Hb20 minus I10B0_R0G0Hb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0C10Hb20 minus I10B0_R0G50Hb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0C10XHb20 minus I10B0_R0KHb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0Hb20 minus I10B0_R0Hb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0C10Hb20 minus I10B0_R0C10Hb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0G50Hb20 minus I10B0_R0G50Hb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0G0Hb20 minus I10B0_R0G0Hb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0KHb20 minus I10B0_R0KHb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0Hb20 minus I10RH_R0Hb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10KI_R0C10XHb20 minus I10RH_R0C10XHb20 over seeds [2] of [2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10KI R0C10XHb20 minus I10B0 R0C10XHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0089 | -0.0089 | [-0.0198, +0.0000] | 0.0036 (0.0036) | OPEN |
| C contact rate | -0.0125 | -0.0125 | [-0.0274, -0.0019] | 0.0062 (0.0062) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0031 | +0.0031 | [-0.0243, +0.0291] | 0.0561 (0.0561) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0150 | +0.0150 | [+0.0028, +0.0277] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.2023 | +0.2023 | [+0.0620, +0.3515] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.3750 | +0.3750 | [-0.3558, +1.0005] | 1.8616 (1.8616) | OPEN |
| gap error 2.0 s (m) | -0.0032 | -0.0032 | [-0.0107, +0.0048] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.0913 | +0.0913 | [-0.0128, +0.2119] | 0.4100 (0.4100) | OPEN |
| gap error 6.0 s (m) | +0.0790 | +0.0790 | [-0.5196, +0.6293] | 1.5504 (1.5504) | OPEN |
| time-gap error 2.0 s (s) | +0.0953 | +0.0953 | [+0.0165, +0.2256] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.4041 | +0.4041 | [+0.0783, +0.7598] | 0.2075 (0.2075) | WORSE |

## I10KI R0C10XHb20 minus I10KI R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0534 | -0.0534 | [-0.0880, -0.0221] | 0.0036 (0.0036) | BETTER |
| C contact rate | -0.0280 | -0.0280 | [-0.0503, -0.0092] | 0.0062 (0.0062) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0966 | +0.0966 | [+0.0600, +0.1375] | 0.0561 (0.0561) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +1.1629 | +1.1629 | [+0.9023, +1.4665] | 0.6885 (0.6885) | WORSE |
| gap bias 6.0 s (m) | +4.4933 | +4.4933 | [+3.5700, +5.5202] | 1.8616 (1.8616) | WORSE |
| gap error 2.0 s (m) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.3928 | +0.3928 | [+0.1186, +0.6912] | 0.4100 (0.4100) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.5957 | +1.5957 | [+0.4391, +2.7795] | 1.5504 (1.5504) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +1.4539 | +1.4539 | [+0.6177, +2.3979] | 0.2075 (0.2075) | WORSE |

## I10KI R0G50Hb20 minus I10KI R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0285 | -0.0285 | [-0.0490, -0.0110] | 0.0036 (0.0036) | BETTER |
| C contact rate | -0.0156 | -0.0156 | [-0.0273, -0.0055] | 0.0062 (0.0062) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0498 | +0.0498 | [+0.0250, +0.0790] | 0.0561 (0.0561) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.6168 | +0.6168 | [+0.4751, +0.7808] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.4878 | +2.4878 | [+1.9470, +3.1161] | 1.8616 (1.8616) | WORSE |
| gap error 2.0 s (m) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.1485 | +0.1485 | [-0.0100, +0.3159] | 0.4100 (0.4100) | OPEN |
| gap error 6.0 s (m) | +0.7351 | +0.7351 | [+0.0573, +1.4536] | 1.5504 (1.5504) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.6646 | +0.6646 | [+0.1751, +1.2107] | 0.2075 (0.2075) | WORSE |

## I10KI R0C10Hb20 minus I10KI R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0196 | -0.0196 | [-0.0358, -0.0048] | 0.0036 (0.0036) | BETTER |
| C contact rate | -0.0156 | -0.0156 | [-0.0294, -0.0035] | 0.0062 (0.0062) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0592 | +0.0592 | [+0.0373, +0.0853] | 0.0561 (0.0561) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.5663 | +0.5663 | [+0.4389, +0.7129] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.1316 | +2.1316 | [+1.6826, +2.5872] | 1.8616 (1.8616) | WORSE |
| gap error 2.0 s (m) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.2421 | +0.2421 | [+0.1097, +0.3829] | 0.4100 (0.4100) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.0021 | +1.0021 | [+0.5134, +1.4551] | 1.5504 (1.5504) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.4974 | +0.4974 | [+0.2269, +0.8088] | 0.2075 (0.2075) | WORSE |

## I10B0 R0C10XHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0445 | -0.0445 | [-0.0782, -0.0164] | 0.0036 (0.0036) | BETTER |
| C contact rate | -0.0218 | -0.0218 | [-0.0429, -0.0051] | 0.0062 (0.0062) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1075 | +0.1075 | [+0.0631, +0.1551] | 0.0561 (0.0561) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +1.0949 | +1.0949 | [+0.8510, +1.3635] | 0.6885 (0.6885) | WORSE |
| gap bias 6.0 s (m) | +4.5436 | +4.5436 | [+3.5576, +5.6036] | 1.8616 (1.8616) | WORSE |
| gap error 2.0 s (m) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.3339 | +0.3339 | [+0.0805, +0.6097] | 0.4100 (0.4100) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.7135 | +1.7135 | [+0.5879, +2.8222] | 1.5504 (1.5504) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +1.2253 | +1.2253 | [+0.4938, +2.0222] | 0.2075 (0.2075) | WORSE |

## I10B0 R0G50Hb20 minus I10B0 R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0285 | -0.0285 | [-0.0541, -0.0074] | 0.0036 (0.0036) | BETTER |
| C contact rate | -0.0125 | -0.0125 | [-0.0265, -0.0020] | 0.0062 (0.0062) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0498 | +0.0498 | [+0.0247, +0.0785] | 0.0561 (0.0561) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.5412 | +0.5412 | [+0.4207, +0.6778] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.3292 | +2.3292 | [+1.8292, +2.8515] | 1.8616 (1.8616) | WORSE |
| gap error 2.0 s (m) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.1254 | +0.1254 | [-0.0112, +0.2669] | 0.4100 (0.4100) | OPEN |
| gap error 6.0 s (m) | +0.6172 | +0.6172 | [+0.0440, +1.2298] | 1.5504 (1.5504) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.4824 | +0.4824 | [+0.0894, +0.8710] | 0.2075 (0.2075) | WORSE |

## I10B0 R0C10Hb20 minus I10B0 R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0160 | -0.0160 | [-0.0293, -0.0055] | 0.0036 (0.0036) | BETTER |
| C contact rate | -0.0156 | -0.0156 | [-0.0316, -0.0040] | 0.0062 (0.0062) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0685 | +0.0685 | [+0.0354, +0.1083] | 0.0561 (0.0561) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.5364 | +0.5364 | [+0.4143, +0.6700] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.4517 | +2.4517 | [+1.7986, +3.1468] | 1.8616 (1.8616) | WORSE |
| gap error 2.0 s (m) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.1903 | +0.1903 | [+0.0668, +0.3211] | 0.4100 (0.4100) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.3217 | +1.3217 | [+0.5969, +2.0973] | 1.5504 (1.5504) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.6795 | +0.6795 | [+0.3401, +1.0815] | 0.2075 (0.2075) | WORSE |

## I10KI R0C10XHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0534 | -0.0534 | [-0.0880, -0.0221] | 0.0036 (0.0036) | BETTER |
| C contact rate | -0.0343 | -0.0343 | [-0.0628, -0.0114] | 0.0062 (0.0062) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1106 | +0.1106 | [+0.0688, +0.1591] | 0.0561 (0.0561) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0150 | +0.0150 | [+0.0028, +0.0277] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +1.2972 | +1.2972 | [+1.0081, +1.6337] | 0.6885 (0.6885) | WORSE |
| gap bias 6.0 s (m) | +4.9186 | +4.9186 | [+3.9569, +6.0113] | 1.8616 (1.8616) | WORSE |
| gap error 2.0 s (m) | -0.0032 | -0.0032 | [-0.0107, +0.0048] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.4252 | +0.4252 | [+0.1153, +0.7630] | 0.4100 (0.4100) | WORSE |
| gap error 6.0 s (m) | +1.7925 | +1.7925 | [+0.5110, +3.0984] | 1.5504 (1.5504) | WORSE |
| time-gap error 2.0 s (s) | +0.0953 | +0.0953 | [+0.0165, +0.2256] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +1.6294 | +1.6294 | [+0.7013, +2.6342] | 0.2075 (0.2075) | WORSE |

## I10KI R0Hb20 minus I10B0 R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0053 | -0.0053 | [-0.0185, +0.0095] | 0.0036 (0.0036) | OPEN |
| C contact rate | -0.0218 | -0.0218 | [-0.0437, -0.0050] | 0.0062 (0.0062) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0000 | +0.0000 | [-0.0141, +0.0137] | 0.0561 (0.0561) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0150 | +0.0150 | [+0.0028, +0.0277] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.2178 | +0.2178 | [+0.1058, +0.3402] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.4818 | +0.4818 | [+0.0228, +0.9192] | 1.8616 (1.8616) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0032 | -0.0032 | [-0.0107, +0.0048] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.0356 | -0.0356 | [-0.1362, +0.0729] | 0.4100 (0.4100) | OPEN |
| gap error 6.0 s (m) | +0.1968 | +0.1968 | [-0.1416, +0.5354] | 1.5504 (1.5504) | OPEN |
| time-gap error 2.0 s (s) | +0.0953 | +0.0953 | [+0.0165, +0.2256] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.2802 | +0.2802 | [+0.0537, +0.5388] | 0.2075 (0.2075) | WORSE |

## I10KI R0C10Hb20 minus I10B0 R0C10Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0071 | -0.0071 | [-0.0174, +0.0019] | 0.0036 (0.0036) | OPEN |
| C contact rate | -0.0093 | -0.0093 | [-0.0216, +0.0000] | 0.0062 (0.0062) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0000 | +0.0000 | [-0.0320, +0.0287] | 0.0561 (0.0561) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0150 | +0.0150 | [+0.0028, +0.0277] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.2501 | +0.2501 | [+0.1131, +0.4034] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.2662 | +0.2662 | [-0.5203, +0.9666] | 1.8616 (1.8616) | OPEN |
| gap error 2.0 s (m) | -0.0032 | -0.0032 | [-0.0107, +0.0048] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.1029 | +0.1029 | [-0.0044, +0.2188] | 0.4100 (0.4100) | OPEN |
| gap error 6.0 s (m) | -0.0318 | -0.0318 | [-0.7691, +0.6486] | 1.5504 (1.5504) | OPEN |
| time-gap error 2.0 s (s) | +0.0953 | +0.0953 | [+0.0165, +0.2256] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.1814 | +0.1814 | [-0.1100, +0.4743] | 0.2075 (0.2075) | OPEN |

## I10KI R0G50Hb20 minus I10B0 R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0036 | -0.0036 | [-0.0162, +0.0109] | 0.0036 (0.0036) | OPEN |
| C contact rate | -0.0093 | -0.0093 | [-0.0184, -0.0021] | 0.0062 (0.0062) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0093 | +0.0093 | [-0.0054, +0.0248] | 0.0561 (0.0561) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0150 | +0.0150 | [+0.0028, +0.0277] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.2203 | +0.2203 | [+0.1248, +0.3296] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.5862 | +0.5862 | [+0.2267, +0.9533] | 1.8616 (1.8616) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0032 | -0.0032 | [-0.0107, +0.0048] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.0512 | +0.0512 | [-0.0351, +0.1432] | 0.4100 (0.4100) | OPEN |
| gap error 6.0 s (m) | +0.2878 | +0.2878 | [-0.0158, +0.6068] | 1.5504 (1.5504) | OPEN |
| time-gap error 2.0 s (s) | +0.0953 | +0.0953 | [+0.0165, +0.2256] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.3635 | +0.3635 | [+0.1034, +0.6871] | 0.2075 (0.2075) | WORSE |

## I10KI R0G0Hb20 minus I10B0 R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0036 | -0.0036 | [-0.0128, +0.0039] | 0.0036 (0.0036) | OPEN |
| C contact rate | -0.0062 | -0.0062 | [-0.0142, +0.0000] | 0.0062 (0.0062) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0093 | +0.0093 | [-0.0015, +0.0212] | 0.0561 (0.0561) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0150 | +0.0150 | [+0.0028, +0.0277] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.1446 | +0.1446 | [+0.0781, +0.2192] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.4276 | +0.4276 | [+0.2321, +0.6336] | 1.8616 (1.8616) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0032 | -0.0032 | [-0.0107, +0.0048] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.0281 | +0.0281 | [-0.0273, +0.0881] | 0.4100 (0.4100) | OPEN |
| gap error 6.0 s (m) | +0.1699 | +0.1699 | [-0.0026, +0.3565] | 1.5504 (1.5504) | OPEN |
| time-gap error 2.0 s (s) | +0.0953 | +0.0953 | [+0.0165, +0.2256] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.1813 | +0.1813 | [+0.0490, +0.3269] | 0.2075 (0.2075) | INSIDE-SPREAD |

## I10KI R0KHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0036 (0.0036) | OPEN |
| C contact rate | -0.0062 | -0.0062 | [-0.0142, +0.0000] | 0.0062 (0.0062) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0140 | +0.0140 | [+0.0016, +0.0276] | 0.0561 (0.0561) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0150 | +0.0150 | [+0.0028, +0.0277] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.1343 | +0.1343 | [+0.0667, +0.2085] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.4253 | +0.4253 | [+0.2189, +0.6608] | 1.8616 (1.8616) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0032 | -0.0032 | [-0.0107, +0.0048] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.0324 | +0.0324 | [-0.0237, +0.0937] | 0.4100 (0.4100) | OPEN |
| gap error 6.0 s (m) | +0.1967 | +0.1967 | [+0.0217, +0.4071] | 1.5504 (1.5504) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0953 | +0.0953 | [+0.0165, +0.2256] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.1755 | +0.1755 | [+0.0415, +0.3202] | 0.2075 (0.2075) | INSIDE-SPREAD |

## I10KI R0Hb20 minus I10RH R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0142 | -0.0142 | [-0.0326, +0.0016] | 0.0036 (0.0036) | OPEN |
| C contact rate | -0.0265 | -0.0265 | [-0.0573, -0.0045] | 0.0062 (0.0062) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0234 | +0.0234 | [+0.0017, +0.0450] | 0.0561 (0.0561) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0439 | +0.0439 | [+0.0158, +0.0789] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.3424 | +0.3424 | [+0.1570, +0.5739] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.2040 | +1.2040 | [+0.5923, +1.8736] | 1.8616 (1.8616) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0263 | -0.0263 | [-0.0488, -0.0071] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -0.0586 | -0.0586 | [-0.2356, +0.1134] | 0.4100 (0.4100) | OPEN |
| gap error 6.0 s (m) | +0.3540 | +0.3540 | [-0.2141, +0.9715] | 1.5504 (1.5504) | OPEN |
| time-gap error 2.0 s (s) | +0.0084 | +0.0084 | [-0.1217, +0.1772] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.6583 | +0.6583 | [+0.0808, +1.2389] | 0.2075 (0.2075) | WORSE |

## I10KI R0C10XHb20 minus I10RH R0C10XHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0214 | -0.0214 | [-0.0377, -0.0078] | 0.0036 (0.0036) | BETTER |
| C contact rate | -0.0202 | -0.0202 | [-0.0438, -0.0027] | 0.0062 (0.0062) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0374 | +0.0374 | [+0.0089, +0.0678] | 0.0561 (0.0561) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0439 | +0.0439 | [+0.0158, +0.0789] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.4634 | +0.4634 | [+0.2673, +0.6912] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.7228 | +1.7228 | [+1.1267, +2.4002] | 1.8616 (1.8616) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0263 | -0.0263 | [-0.0488, -0.0071] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | +0.0394 | +0.0394 | [-0.1447, +0.2363] | 0.4100 (0.4100) | OPEN |
| gap error 6.0 s (m) | +0.6561 | +0.6561 | [+0.0225, +1.3522] | 1.5504 (1.5504) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0084 | +0.0084 | [-0.1217, +0.1772] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +1.0197 | +1.0197 | [+0.3181, +1.7828] | 0.2075 (0.2075) | WORSE |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10B0_R0C10Hb20 (2) | 0.0445 [0.0166, 0.0858] | 0.0374 [0.0071, 0.0767] | 0.2134 [0.1410, 0.2908] | 0.0693 [0.0038, 0.1357] | 1.4888 [0.9542, 2.0571] | 6.0142 [4.3254, 7.7101] | 0.4401 [0.3809, 0.5025] | 2.5896 [2.1552, 3.1118] | 7.7168 [6.3830, 9.2579] | 0.8098 [0.4803, 1.2385] | 3.2844 [2.1531, 4.5684] |
| I10B0_R0C10XHb20 (2) | 0.0427 [0.0154, 0.0840] | 0.0421 [0.0096, 0.0846] | 0.2009 [0.1327, 0.2746] | 0.0693 [0.0038, 0.1357] | 1.5595 [1.0401, 2.1278] | 5.7329 [4.1033, 7.4876] | 0.4401 [0.3809, 0.5025] | 2.5862 [2.1562, 3.1027] | 7.5098 [6.2460, 8.9867] | 0.8098 [0.4803, 1.2385] | 3.3196 [2.1717, 4.6138] |
| I10B0_R0G0Hb20 (2) | 0.0890 [0.0436, 0.1456] | 0.0654 [0.0221, 0.1176] | 0.0950 [0.0517, 0.1437] | 0.0693 [0.0038, 0.1357] | 0.4112 [-0.1021, 0.9209] | 1.2333 [-0.4078, 2.8483] | 0.4401 [0.3809, 0.5025] | 2.2739 [1.8872, 2.6757] | 5.7778 [4.7153, 6.9391] | 0.8098 [0.4803, 1.2385] | 2.1225 [1.1735, 3.2254] |
| I10B0_R0G50Hb20 (2) | 0.0605 [0.0284, 0.1032] | 0.0530 [0.0160, 0.1003] | 0.1449 [0.0862, 0.2097] | 0.0693 [0.0038, 0.1357] | 0.9524 [0.4339, 1.4810] | 3.5625 [1.9880, 5.2675] | 0.4401 [0.3809, 0.5025] | 2.3993 [2.0000, 2.8484] | 6.3950 [5.2589, 7.6935] | 0.8098 [0.4803, 1.2385] | 2.6049 [1.6319, 3.7265] |
| I10B0_R0Hb20 (2) | 0.0836 [0.0396, 0.1376] | 0.0530 [0.0162, 0.0958] | 0.1542 [0.0926, 0.2250] | 0.0693 [0.0038, 0.1357] | 0.8123 [0.2410, 1.4303] | 3.3374 [1.6215, 5.1913] | 0.4401 [0.3809, 0.5025] | 2.5359 [2.1626, 2.9715] | 6.6418 [5.5197, 7.9649] | 0.8098 [0.4803, 1.2385] | 2.6050 [1.6488, 3.7716] |
| I10B0_R0KHb20 (2) | 0.0872 [0.0427, 0.1441] | 0.0639 [0.0217, 0.1159] | 0.0935 [0.0528, 0.1415] | 0.0693 [0.0038, 0.1357] | 0.4646 [-0.0515, 0.9747] | 1.1893 [-0.4969, 2.7960] | 0.4401 [0.3809, 0.5025] | 2.2523 [1.8666, 2.6603] | 5.7963 [4.7441, 6.9550] | 0.8098 [0.4803, 1.2385] | 2.0944 [1.1339, 3.2235] |
| I10KI_R0C10Hb20 (2) | 0.0374 [0.0121, 0.0739] | 0.0280 [0.0053, 0.0593] | 0.2134 [0.1446, 0.2921] | 0.0843 [0.0179, 0.1502] | 1.7390 [1.1728, 2.3568] | 6.2804 [4.6288, 8.1245] | 0.4369 [0.3781, 0.4990] | 2.6925 [2.2248, 3.2356] | 7.6849 [6.2993, 9.2476] | 0.9051 [0.5345, 1.3675] | 3.4658 [2.2889, 4.7589] |
| I10KI_R0C10XHb20 (2) | 0.0338 [0.0084, 0.0707] | 0.0296 [0.0063, 0.0619] | 0.2040 [0.1345, 0.2837] | 0.0843 [0.0179, 0.1502] | 1.7618 [1.1912, 2.3835] | 6.1078 [4.3959, 8.0094] | 0.4369 [0.3781, 0.4990] | 2.6775 [2.2106, 3.2112] | 7.5888 [6.2224, 9.2331] | 0.9051 [0.5345, 1.3675] | 3.7237 [2.4338, 5.1009] |
| I10KI_R0G0Hb20 (2) | 0.0854 [0.0429, 0.1387] | 0.0592 [0.0196, 0.1082] | 0.1044 [0.0593, 0.1534] | 0.0843 [0.0179, 0.1502] | 0.5558 [0.0375, 1.0745] | 1.6609 [-0.0082, 3.3688] | 0.4369 [0.3781, 0.4990] | 2.3020 [1.9201, 2.7142] | 5.9478 [4.8364, 7.1222] | 0.9051 [0.5345, 1.3675] | 2.3038 [1.3225, 3.4650] |
| I10KI_R0G50Hb20 (2) | 0.0569 [0.0265, 0.0950] | 0.0436 [0.0111, 0.0849] | 0.1542 [0.0943, 0.2217] | 0.0843 [0.0179, 0.1502] | 1.1727 [0.6417, 1.7273] | 4.1488 [2.4472, 5.9714] | 0.4369 [0.3781, 0.4990] | 2.4504 [2.0412, 2.9045] | 6.6829 [5.4074, 8.0713] | 0.9051 [0.5345, 1.3675] | 2.9684 [1.8677, 4.2030] |
| I10KI_R0Hb20 (2) | 0.0783 [0.0320, 0.1339] | 0.0312 [0.0073, 0.0591] | 0.1542 [0.0928, 0.2246] | 0.0843 [0.0179, 0.1502] | 1.0301 [0.4498, 1.6453] | 3.8192 [1.9973, 5.7880] | 0.4369 [0.3781, 0.4990] | 2.5003 [2.0936, 2.9532] | 6.8386 [5.6234, 8.2418] | 0.9051 [0.5345, 1.3675] | 2.8852 [1.8228, 4.1284] |
| I10KI_R0KHb20 (2) | 0.0872 [0.0427, 0.1441] | 0.0576 [0.0190, 0.1060] | 0.1075 [0.0618, 0.1575] | 0.0843 [0.0179, 0.1502] | 0.5989 [0.0872, 1.1128] | 1.6146 [-0.1042, 3.3511] | 0.4369 [0.3781, 0.4990] | 2.2847 [1.9041, 2.6928] | 5.9931 [4.9143, 7.1372] | 0.9051 [0.5345, 1.3675] | 2.2699 [1.2645, 3.4135] |
| I10RH_R0C10XHb20 (2) | 0.0552 [0.0215, 0.1048] | 0.0498 [0.0118, 0.0971] | 0.1667 [0.1066, 0.2376] | 0.0404 [-0.0444, 0.1203] | 1.2984 [0.7186, 1.9070] | 4.3850 [2.7552, 6.1241] | 0.4632 [0.3951, 0.5348] | 2.6381 [2.1938, 3.1656] | 6.9327 [5.7398, 8.3397] | 0.8967 [0.5148, 1.3922] | 2.7040 [1.6612, 3.8044] |
| I10RH_R0Hb20 (2) | 0.0925 [0.0429, 0.1505] | 0.0576 [0.0174, 0.1078] | 0.1308 [0.0743, 0.1997] | 0.0404 [-0.0444, 0.1203] | 0.6877 [0.0752, 1.3215] | 2.6151 [0.8977, 4.4329] | 0.4632 [0.3951, 0.5348] | 2.5589 [2.1529, 3.0128] | 6.4845 [5.3939, 7.7404] | 0.8967 [0.5148, 1.3922] | 2.2269 [1.3214, 3.2800] |
| I10UEA_R0C10XHb20 (0,1) | 0.0623 [0.0388, 0.0910] | 0.0280 [0.0123, 0.0462] | 0.3910 [0.3191, 0.4592] | -0.2236 [-0.2757, -0.1682] | 3.0145 [2.3455, 3.6779] | 10.3479 [8.3469, 12.6031] | 0.4729 [0.4299, 0.5219] | 3.5797 [3.0625, 4.1116] | 11.7414 [10.0248, 13.7045] | 1.0435 [0.7059, 1.4212] | 4.8571 [3.8136, 5.9361] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
