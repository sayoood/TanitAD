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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQH:LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQH:LSRH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQH:LSRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQH:LSMC", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSQH:LSQC", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDQH:LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDQH:LSDRH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDQH:LSDRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDQH:LSDMC", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDQH:LSDQC", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, LSQH minus LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSQH minus LSRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSQH minus LSRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSQH minus LSMC over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSQH minus LSQC over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDQH minus LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDQH minus LSDRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDQH minus LSDRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDQH minus LSDMC over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDQH minus LSDQC over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## LSQH minus LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0658 / -0.0569 | -0.0614 | [-0.0954, -0.0281] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0156 / -0.0343 | -0.0249 | [-0.0469, -0.0090] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1215 / +0.1371 | +0.1293 | [+0.0808, +0.1826] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.5715 / +0.8616 | +0.7166 | [+0.3667, +1.1210] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.3403 / +3.0552 | +2.6978 | [+1.5875, +3.9272] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +4.6310 / +6.2110 | +5.4210 | [+3.2169, +7.8004] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.4627 / +0.6729 | +0.5678 | [+0.2792, +0.9018] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.6869 / +2.6459 | +2.1664 | [+1.1571, +3.3880] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.6360 / +4.3249 | +3.9804 | [+2.1986, +6.0486] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.0482 / +0.3887 | +0.2184 | [+0.0355, +0.5349] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.3382 / +0.9601 | +0.6492 | [+0.2315, +1.1954] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSQH minus LSRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0249 / -0.0178 | -0.0214 | [-0.0337, -0.0108] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0031 / -0.0047 | -0.0039 | [-0.0089, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0343 / +0.0265 | +0.0304 | [+0.0115, +0.0533] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2020 / +0.1479 | +0.1750 | [+0.0829, +0.2897] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +0.6892 / +0.6789 | +0.6841 | [+0.3199, +1.1577] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.2246 / +1.4802 | +1.3524 | [+0.6344, +2.3420] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.1602 / +0.1242 | +0.1422 | [+0.0598, +0.2469] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.5864 / +0.5240 | +0.5552 | [+0.2258, +0.9828] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.1261 / +1.2615 | +1.1938 | [+0.4807, +2.1401] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0262 / +0.0204 | +0.0233 | [+0.0119, +0.0354] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.2011 / +0.0686 | +0.1348 | [+0.0375, +0.2662] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSQH minus LSRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0089 / +0.0267 | +0.0178 | [+0.0005, +0.0413] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0156 / +0.0234 | +0.0195 | [+0.0070, +0.0346] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0483 / +0.0841 | +0.0662 | [+0.0303, +0.1061] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1356 / +0.4148 | +0.2752 | [+0.0679, +0.5215] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +0.9915 / +1.7278 | +1.3596 | [+0.6495, +2.1752] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.5446 / +2.7759 | +2.1602 | [+0.9114, +3.5884] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.1365 / +0.3794 | +0.2580 | [+0.0748, +0.4746] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.9771 / +1.9286 | +1.4528 | [+0.6593, +2.3555] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +1.7267 / +3.2894 | +2.5081 | [+1.1450, +4.0301] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0071 / +0.3489 | +0.1780 | [-0.0047, +0.4925] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.2001 / +0.8699 | +0.5350 | [+0.0995, +1.1039] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSQH minus LSMC

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0071 / +0.0267 | +0.0098 | [-0.0166, +0.0387] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0125 / +0.0125 | +0.0125 | [+0.0044, +0.0221] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0358 / -0.0950 | -0.0654 | [-0.1156, -0.0192] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.4591 / -0.9216 | -0.6903 | [-1.0550, -0.3806] | 0.0689 (0.0315) | BETTER |
| gap bias 4.0 s (m) | -1.3207 / -2.1354 | -1.7280 | [-2.5048, -0.8773] | 1.6421 (0.6283) | BETTER |
| gap bias 6.0 s (m) | -1.8838 / -3.4551 | -2.6694 | [-4.1038, -1.2045] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.3391 / -0.7709 | -0.5550 | [-0.8969, -0.2577] | 0.0426 (0.0189) | BETTER |
| gap error 4.0 s (m) | -1.2015 / -1.8639 | -1.5327 | [-2.2595, -0.7507] | 1.0764 (0.6913) | BETTER |
| gap error 6.0 s (m) | -1.7318 / -2.9640 | -2.3479 | [-3.6626, -1.0533] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.1907 / -0.4647 | -0.3277 | [-0.5615, -0.1533] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.7994 / -1.3220 | -1.0607 | [-1.5605, -0.6355] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSQH minus LSQC

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0071 / +0.0036 | -0.0018 | [-0.0157, +0.0162] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [-0.0038, +0.0037] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0374 / -0.0374 | -0.0374 | [-0.0588, -0.0163] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.3309 / -0.3783 | -0.3546 | [-0.4987, -0.2224] | 0.0689 (0.0315) | BETTER |
| gap bias 4.0 s (m) | -0.9605 / -0.8777 | -0.9191 | [-1.4692, -0.4274] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -2.0017 / -1.7143 | -1.8580 | [-2.8677, -0.9572] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.2330 / -0.2804 | -0.2567 | [-0.3932, -0.1324] | 0.0426 (0.0189) | BETTER |
| gap error 4.0 s (m) | -0.9295 / -0.8806 | -0.9050 | [-1.4487, -0.4250] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | -1.9852 / -1.5731 | -1.7792 | [-2.7567, -0.8903] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.0444 / -0.1041 | -0.0742 | [-0.1497, -0.0315] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.2354 / -0.3655 | -0.3005 | [-0.4934, -0.1418] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDQH minus LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0552 / -0.0552 | -0.0552 | [-0.0890, -0.0261] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0109 / -0.0436 | -0.0273 | [-0.0548, -0.0060] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1293 / +0.0717 | +0.1005 | [+0.0508, +0.1576] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6523 / +0.6078 | +0.6301 | [+0.3463, +0.9512] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.4521 / +2.1335 | +2.2928 | [+1.3695, +3.3401] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +4.3210 / +4.2051 | +4.2630 | [+2.7972, +5.8508] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.4881 / +0.4350 | +0.4615 | [+0.2272, +0.7224] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.9239 / +1.9424 | +1.9332 | [+1.0806, +2.9254] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.5973 / +2.4786 | +3.0379 | [+1.2959, +4.7199] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0577 / +0.0587 | +0.0582 | [+0.0328, +0.0892] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.2680 / +0.3339 | +0.3009 | [+0.1105, +0.5344] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDQH minus LSDRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0356 / -0.0409 | -0.0383 | [-0.0616, -0.0174] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0062 / -0.0249 | -0.0156 | [-0.0328, -0.0024] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0389 / +0.0171 | +0.0280 | [+0.0101, +0.0545] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2450 / +0.1711 | +0.2080 | [+0.1043, +0.3201] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +0.8493 / +0.5238 | +0.6865 | [+0.3416, +1.1247] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.3378 / +0.8299 | +1.0838 | [+0.5148, +1.9784] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.1795 / +0.1424 | +0.1610 | [+0.0766, +0.2494] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.6463 / +0.4419 | +0.5441 | [+0.2174, +0.8913] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.2084 / +0.7361 | +0.9722 | [+0.3205, +1.8258] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0315 / +0.0144 | +0.0229 | [+0.0092, +0.0412] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0913 / +0.0498 | +0.0706 | [-0.0124, +0.1773] | 1.7849 (0.8089) | OPEN |

## LSDQH minus LSDRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0338 / +0.0302 | +0.0320 | [+0.0166, +0.0482] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0078 / +0.0280 | +0.0179 | [+0.0043, +0.0405] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0498 / +0.0187 | +0.0343 | [+0.0059, +0.0682] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2158 / +0.1653 | +0.1905 | [+0.0634, +0.3271] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.0996 / +0.7141 | +0.9069 | [+0.3398, +1.5303] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.6223 / +0.6027 | +1.1125 | [+0.0048, +2.4297] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.1734 / +0.1603 | +0.1668 | [+0.0683, +0.2734] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.1402 / +1.1043 | +1.1223 | [+0.6281, +1.6777] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +1.7630 / +1.3673 | +1.5652 | [+0.5913, +2.7188] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0136 / +0.0192 | +0.0164 | [-0.0140, +0.0510] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0956 / +0.2571 | +0.1764 | [-0.0101, +0.4236] | 1.7849 (0.8089) | OPEN |

## LSDQH minus LSDMC

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0214 / +0.0338 | +0.0276 | [+0.0118, +0.0475] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0062 / +0.0109 | +0.0086 | [+0.0021, +0.0186] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0389 / -0.0732 | -0.0561 | [-0.0962, -0.0214] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.6951 / -0.7066 | -0.7009 | [-0.9095, -0.5082] | 0.0689 (0.0315) | BETTER |
| gap bias 4.0 s (m) | -1.5471 / -1.8743 | -1.7107 | [-2.3742, -1.0697] | 1.6421 (0.6283) | BETTER |
| gap bias 6.0 s (m) | -2.1002 / -3.1348 | -2.6175 | [-4.0675, -1.2814] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.5991 / -0.6206 | -0.6099 | [-0.8052, -0.4291] | 0.0426 (0.0189) | BETTER |
| gap error 4.0 s (m) | -1.5408 / -1.5547 | -1.5478 | [-2.2123, -0.9598] | 1.0764 (0.6913) | BETTER |
| gap error 6.0 s (m) | -1.9684 / -2.5912 | -2.2798 | [-3.6128, -1.0657] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.3147 / -0.3203 | -0.3175 | [-0.4820, -0.1937] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.8875 / -0.7784 | -0.8330 | [-1.1570, -0.5551] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDQH minus LSDQC

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0036 / +0.0231 | +0.0133 | [-0.0068, +0.0371] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0031 / +0.0031 | +0.0000 | [-0.0065, +0.0080] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0545 / -0.0592 | -0.0569 | [-0.0868, -0.0295] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.4957 / -0.5098 | -0.5027 | [-0.6936, -0.3362] | 0.0689 (0.0315) | BETTER |
| gap bias 4.0 s (m) | -1.3271 / -1.6600 | -1.4935 | [-2.2182, -0.8589] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -1.7732 / -2.7320 | -2.2526 | [-3.7143, -0.9202] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.3980 / -0.4316 | -0.4148 | [-0.6052, -0.2569] | 0.0426 (0.0189) | BETTER |
| gap error 4.0 s (m) | -1.3486 / -1.4142 | -1.3814 | [-2.0401, -0.7774] | 1.0764 (0.6913) | BETTER |
| gap error 6.0 s (m) | -1.7473 / -2.3811 | -2.0642 | [-3.4119, -0.8334] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.1015 / -0.1390 | -0.1202 | [-0.2129, -0.0624] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.5877 / -0.3948 | -0.4913 | [-0.7359, -0.3037] | 1.7849 (0.8089) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LS (0,1) | 0.1246 [0.0823, 0.1714] | 0.0444 [0.0193, 0.0766] | 0.2111 [0.1246, 0.3163] | -0.2350 [-0.3195, -0.1538] | 1.1867 [0.5392, 1.9640] | 5.4393 [3.1026, 8.0310] | 0.5163 [0.4689, 0.5676] | 2.7856 [2.2778, 3.4051] | 8.3298 [6.8681, 10.0377] | 1.2365 [0.7221, 1.7936] | 2.4602 [1.4184, 3.6642] |
| LSD (0,1) | 0.1343 [0.0873, 0.1879] | 0.0452 [0.0109, 0.0906] | 0.2531 [0.1958, 0.3138] | -0.2031 [-0.2748, -0.1325] | 1.4817 [0.8172, 2.2957] | 6.7598 [4.6651, 8.6599] | 0.5100 [0.4630, 0.5586] | 2.8968 [2.3956, 3.5375] | 9.3372 [8.1968, 10.6886] | 1.4640 [0.8170, 2.2047] | 2.5391 [1.6003, 3.5518] |
| LSDMC (0,1) | 0.0516 [0.0277, 0.0823] | 0.0093 [0.0000, 0.0261] | 0.4097 [0.3379, 0.4816] | 1.1278 [0.7191, 1.5873] | 5.4852 [3.9211, 7.1340] | 13.6403 [10.7057, 16.7188] | 1.5814 [1.2109, 2.0053] | 6.3777 [4.9718, 7.8509] | 14.6549 [12.0072, 17.5735] | 1.8397 [1.1472, 2.5956] | 3.6730 [2.6322, 4.7572] |
| LSDQC (0,1) | 0.0658 [0.0389, 0.0988] | 0.0179 [0.0047, 0.0379] | 0.4104 [0.3222, 0.4975] | 0.9297 [0.5356, 1.3644] | 5.2680 [3.7314, 6.8447] | 13.2754 [10.2952, 16.3070] | 1.3863 [1.0227, 1.7990] | 6.2113 [4.8507, 7.6690] | 14.4393 [11.7734, 17.3270] | 1.6424 [0.9985, 2.3604] | 3.3313 [2.3127, 4.4132] |
| LSDQH (0,1) | 0.0792 [0.0482, 0.1169] | 0.0179 [0.0043, 0.0405] | 0.3536 [0.2686, 0.4417] | 0.4269 [0.1446, 0.7327] | 3.7745 [2.4781, 5.0320] | 11.0228 [8.3502, 13.3694] | 0.9715 [0.7268, 1.2465] | 4.8299 [3.7907, 5.8712] | 12.3751 [10.2054, 14.5075] | 1.5222 [0.8726, 2.2667] | 2.8400 [1.9047, 3.8211] |
| LSDRH (0,1) | 0.1174 [0.0734, 0.1670] | 0.0335 [0.0072, 0.0708] | 0.3255 [0.2493, 0.4022] | 0.2189 [-0.0074, 0.5021] | 3.0880 [1.9867, 4.2275] | 9.9390 [7.5229, 12.1905] | 0.8106 [0.6022, 1.0655] | 4.2859 [3.4176, 5.2197] | 11.4028 [9.5253, 13.4774] | 1.4992 [0.8581, 2.2308] | 2.7695 [1.8682, 3.7196] |
| LSDRL (0,1) | 0.0472 [0.0227, 0.0829] | 0.0000 [0.0000, 0.0000] | 0.3193 [0.2469, 0.3910] | 0.2364 [-0.0076, 0.5420] | 2.8677 [1.8890, 3.9087] | 9.9103 [7.7746, 11.9708] | 0.8047 [0.5833, 1.0843] | 3.7077 [2.8459, 4.6228] | 10.8099 [8.9499, 12.7615] | 1.5058 [0.8571, 2.2451] | 2.6637 [1.6935, 3.7219] |
| LSMC (0,1) | 0.0534 [0.0312, 0.0772] | 0.0070 [0.0000, 0.0177] | 0.4058 [0.3276, 0.4795] | 1.1719 [0.5946, 1.8411] | 5.6125 [4.1334, 7.2170] | 13.5297 [10.7494, 16.4204] | 1.6390 [1.0908, 2.2664] | 6.4847 [5.0902, 8.0342] | 14.6582 [12.0491, 17.4352] | 1.7826 [1.3123, 2.2538] | 4.1701 [3.3443, 5.0752] |
| LSQC (0,1) | 0.0649 [0.0384, 0.0985] | 0.0195 [0.0078, 0.0364] | 0.3777 [0.2840, 0.4724] | 0.8362 [0.4309, 1.3170] | 4.8035 [3.4585, 6.2442] | 12.7183 [9.6831, 15.6759] | 1.3408 [0.9562, 1.7937] | 5.8570 [4.6629, 7.2501] | 14.0894 [11.3615, 16.9036] | 1.5291 [1.0824, 1.9879] | 3.4098 [2.5781, 4.2949] |
| LSQH (0,1) | 0.0632 [0.0341, 0.1013] | 0.0195 [0.0070, 0.0346] | 0.3403 [0.2487, 0.4354] | 0.4816 [0.1600, 0.8462] | 3.8844 [2.8386, 4.9846] | 10.8603 [8.1814, 13.2860] | 1.0840 [0.7896, 1.4390] | 4.9520 [4.0309, 5.9978] | 12.3102 [10.0744, 14.5036] | 1.4549 [1.0033, 1.9270] | 3.1094 [2.2488, 4.0395] |
| LSRH (0,1) | 0.0845 [0.0527, 0.1232] | 0.0234 [0.0086, 0.0426] | 0.3100 [0.2259, 0.4009] | 0.3066 [0.0047, 0.6708] | 3.2004 [2.2391, 4.2372] | 9.5079 [7.0923, 11.8941] | 0.9418 [0.6696, 1.2884] | 4.3968 [3.5541, 5.3752] | 11.1165 [9.1719, 13.2105] | 1.4316 [0.9801, 1.9035] | 2.9745 [2.1468, 3.8918] |
| LSRL (0,1) | 0.0454 [0.0238, 0.0728] | 0.0000 [0.0000, 0.0000] | 0.2741 [0.1700, 0.3895] | 0.2064 [-0.0545, 0.5267] | 2.5248 [1.6354, 3.5112] | 8.7001 [6.1813, 11.2690] | 0.8261 [0.5991, 1.1154] | 3.4991 [2.7134, 4.3601] | 9.8022 [7.5079, 12.2263] | 1.2769 [0.7638, 1.8367] | 2.5743 [1.5172, 3.8082] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
