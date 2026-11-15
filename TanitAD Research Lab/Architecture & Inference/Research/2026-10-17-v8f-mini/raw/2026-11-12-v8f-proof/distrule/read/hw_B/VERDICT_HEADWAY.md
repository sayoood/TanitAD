# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 24, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSRH:LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSRL:LS", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSHRH:LSH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSHRL:LSH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDRH:LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDRL:LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSRH:LSRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSHRH:LSHRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDRH:LSDRL", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, LSRH minus LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSRL minus LS over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSHRH minus LSH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSHRL minus LSH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDRH minus LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDRL minus LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSRH minus LSRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSHRH minus LSHRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDRH minus LSDRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## LSRH minus LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0409 / -0.0391 | -0.0400 | [-0.0686, -0.0127] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0125 / -0.0296 | -0.0210 | [-0.0399, -0.0076] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0872 / +0.1106 | +0.0989 | [+0.0593, +0.1430] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.3695 / +0.7136 | +0.5416 | [+0.2014, +0.9478] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.6511 / +2.3763 | +2.0137 | [+0.9853, +3.0785] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +3.4065 / +4.7307 | +4.0686 | [+2.1824, +6.0292] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3025 / +0.5486 | +0.4256 | [+0.1556, +0.7587] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.1005 / +2.1219 | +1.6112 | [+0.6495, +2.7441] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +2.5099 / +3.0634 | +2.7867 | [+1.3510, +4.4314] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0220 / +0.3682 | +0.1951 | [+0.0132, +0.5129] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.1371 / +0.8915 | +0.5143 | [+0.0767, +1.1245] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSRL minus LS

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0747 / -0.0836 | -0.0792 | [-0.1137, -0.0495] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0312 / -0.0576 | -0.0444 | [-0.0766, -0.0193] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0732 / +0.0530 | +0.0631 | [+0.0252, +0.1079] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4359 / +0.4468 | +0.4413 | [+0.1852, +0.7616] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.3488 / +1.3274 | +1.3381 | [+0.6605, +2.1583] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +3.0865 / +3.4351 | +3.2608 | [+1.7422, +5.1192] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3262 / +0.2935 | +0.3098 | [+0.0841, +0.5960] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.7099 / +0.7173 | +0.7136 | [+0.1863, +1.3994] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.9093 / +1.0354 | +1.4723 | [-0.0304, +3.1044] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0412 / +0.0397 | +0.0405 | [+0.0188, +0.0661] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.1381 / +0.0902 | +0.1141 | [+0.0211, +0.2210] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSHRH minus LSH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0071 / -0.0641 | -0.0356 | [-0.0858, -0.0006] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0093 / -0.0312 | -0.0202 | [-0.0412, -0.0050] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0623 / +0.0950 | +0.0787 | [+0.0362, +0.1258] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2342 / +0.4987 | +0.3665 | [+0.1326, +0.7194] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.0510 / +1.6666 | +1.3588 | [+0.5783, +2.4081] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.8849 / +3.8963 | +2.8906 | [+1.1494, +5.2374] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.1597 / +0.3833 | +0.2715 | [+0.0772, +0.5645] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.7357 / +1.1916 | +0.9637 | [+0.3401, +1.8108] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.2856 / +2.3368 | +1.8112 | [+0.6132, +3.5955] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0135 / +0.0539 | +0.0337 | [+0.0079, +0.0667] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0673 / +0.3555 | +0.2114 | [+0.0280, +0.4850] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSHRL minus LSH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0747 / -0.0979 | -0.0863 | [-0.1268, -0.0520] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0358 / -0.0545 | -0.0452 | [-0.0736, -0.0222] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0530 / +0.0670 | +0.0600 | [+0.0230, +0.1021] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4398 / +0.4367 | +0.4382 | [+0.1805, +0.7639] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.2744 / +1.3544 | +1.3144 | [+0.6124, +2.1931] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.4626 / +3.3369 | +2.8997 | [+1.4808, +4.7589] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3254 / +0.3188 | +0.3221 | [+0.0996, +0.6128] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.6770 / +0.6914 | +0.6842 | [+0.1502, +1.3601] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.3695 / +1.3485 | +1.3590 | [+0.2405, +2.7739] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0510 / +0.0404 | +0.0457 | [+0.0210, +0.0746] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0935 / +0.0836 | +0.0886 | [+0.0131, +0.1700] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDRH minus LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0196 / -0.0142 | -0.0169 | [-0.0324, -0.0035] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0047 / -0.0187 | -0.0117 | [-0.0246, -0.0024] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0903 / +0.0545 | +0.0724 | [+0.0351, +0.1186] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4074 / +0.4368 | +0.4221 | [+0.1791, +0.7309] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.6028 / +1.6098 | +1.6063 | [+0.8038, +2.5665] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.9832 / +3.3752 | +3.1792 | [+1.9257, +4.6598] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3086 / +0.2925 | +0.3006 | [+0.1064, +0.5433] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.2777 / +1.5005 | +1.3891 | [+0.6602, +2.3242] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +2.3890 / +1.7424 | +2.0657 | [+0.7336, +3.5364] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0262 / +0.0443 | +0.0352 | [+0.0136, +0.0697] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.1767 / +0.2840 | +0.2304 | [+0.0807, +0.4610] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDRL minus LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0890 / -0.0854 | -0.0872 | [-0.1264, -0.0518] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0187 / -0.0717 | -0.0452 | [-0.0906, -0.0109] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0794 / +0.0530 | +0.0662 | [+0.0346, +0.1098] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4366 / +0.4426 | +0.4396 | [+0.1842, +0.7609] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.3525 / +1.4194 | +1.3860 | [+0.6870, +2.2648] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.6987 / +3.6024 | +3.1505 | [+1.7040, +4.8432] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3147 / +0.2747 | +0.2947 | [+0.0861, +0.5631] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.7837 / +0.8381 | +0.8109 | [+0.1837, +1.6287] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.8343 / +1.1112 | +1.4728 | [+0.1130, +3.0097] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0441 / +0.0395 | +0.0418 | [+0.0189, +0.0684] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.1724 / +0.0768 | +0.1246 | [+0.0149, +0.2388] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSRH minus LSRL

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

## LSHRH minus LSHRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0676 / +0.0338 | +0.0507 | [+0.0227, +0.0848] | 0.0463 (0.0214) | WORSE |
| C contact rate | +0.0265 / +0.0234 | +0.0249 | [+0.0108, +0.0415] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0093 / +0.0280 | +0.0187 | [-0.0063, +0.0440] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.2055 / +0.0621 | -0.0717 | [-0.3073, +0.1159] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.2233 / +0.3123 | +0.0445 | [-0.5037, +0.5731] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.5777 / +0.5594 | -0.0092 | [-0.9799, +1.0375] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.1657 / +0.0645 | -0.0506 | [-0.2581, +0.1155] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0587 / +0.5003 | +0.2795 | [-0.1904, +0.7451] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.0838 / +0.9882 | +0.4522 | [-0.3969, +1.4525] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0374 / +0.0135 | -0.0120 | [-0.0532, +0.0294] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0262 / +0.2719 | +0.1228 | [-0.0699, +0.3826] | 1.7849 (0.8089) | OPEN |

## LSDRH minus LSDRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0694 / +0.0712 | +0.0703 | [+0.0426, +0.0996] | 0.0463 (0.0214) | WORSE |
| C contact rate | +0.0140 / +0.0530 | +0.0335 | [+0.0072, +0.0708] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0109 / +0.0016 | +0.0062 | [-0.0176, +0.0325] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0292 / -0.0058 | -0.0175 | [-0.1237, +0.0887] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.2503 / +0.1903 | +0.2203 | [-0.2160, +0.6976] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.2846 / -0.2272 | +0.0287 | [-0.8321, +1.0019] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0061 / +0.0178 | +0.0059 | [-0.0807, +0.0868] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.4939 / +0.6624 | +0.5782 | [+0.1898, +1.0248] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.5546 / +0.6312 | +0.5929 | [-0.1712, +1.4221] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0179 / +0.0048 | -0.0066 | [-0.0350, +0.0323] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0043 / +0.2072 | +0.1058 | [-0.0539, +0.3559] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LS (0,1) | 0.1246 [0.0823, 0.1714] | 0.0444 [0.0193, 0.0766] | 0.2111 [0.1246, 0.3163] | -0.2350 [-0.3195, -0.1538] | 1.1867 [0.5392, 1.9640] | 5.4393 [3.1026, 8.0310] | 0.5163 [0.4689, 0.5676] | 2.7856 [2.2778, 3.4051] | 8.3298 [6.8681, 10.0377] | 1.2365 [0.7221, 1.7936] | 2.4602 [1.4184, 3.6642] |
| LSD (0,1) | 0.1343 [0.0873, 0.1879] | 0.0452 [0.0109, 0.0906] | 0.2531 [0.1958, 0.3138] | -0.2031 [-0.2748, -0.1325] | 1.4817 [0.8172, 2.2957] | 6.7598 [4.6651, 8.6599] | 0.5100 [0.4630, 0.5586] | 2.8968 [2.3956, 3.5375] | 9.3372 [8.1968, 10.6886] | 1.4640 [0.8170, 2.2047] | 2.5391 [1.6003, 3.5518] |
| LSDRH (0,1) | 0.1174 [0.0734, 0.1670] | 0.0335 [0.0072, 0.0708] | 0.3255 [0.2493, 0.4022] | 0.2189 [-0.0074, 0.5021] | 3.0880 [1.9867, 4.2275] | 9.9390 [7.5229, 12.1905] | 0.8106 [0.6022, 1.0655] | 4.2859 [3.4176, 5.2197] | 11.4028 [9.5253, 13.4774] | 1.4992 [0.8581, 2.2308] | 2.7695 [1.8682, 3.7196] |
| LSDRL (0,1) | 0.0472 [0.0227, 0.0829] | 0.0000 [0.0000, 0.0000] | 0.3193 [0.2469, 0.3910] | 0.2364 [-0.0076, 0.5420] | 2.8677 [1.8890, 3.9087] | 9.9103 [7.7746, 11.9708] | 0.8047 [0.5833, 1.0843] | 3.7077 [2.8459, 4.6228] | 10.8099 [8.9499, 12.7615] | 1.5058 [0.8571, 2.2451] | 2.6637 [1.6935, 3.7219] |
| LSH (0,1) | 0.1335 [0.0883, 0.1849] | 0.0452 [0.0222, 0.0736] | 0.2064 [0.1380, 0.2897] | -0.2322 [-0.2989, -0.1671] | 1.0451 [0.4824, 1.6813] | 5.4365 [3.5098, 7.5196] | 0.4902 [0.4365, 0.5462] | 2.6766 [2.3186, 3.1134] | 7.7832 [6.5384, 9.2696] | 1.0958 [0.7200, 1.4891] | 2.3047 [1.4352, 3.2622] |
| LSHRH (0,1) | 0.0979 [0.0626, 0.1368] | 0.0249 [0.0108, 0.0415] | 0.2850 [0.2207, 0.3558] | 0.1343 [-0.1396, 0.4940] | 2.4039 [1.6030, 3.2892] | 8.3271 [6.6280, 10.0948] | 0.7617 [0.5830, 1.0415] | 3.6403 [2.9694, 4.4345] | 9.5944 [8.1120, 11.2373] | 1.1296 [0.7684, 1.5073] | 2.5160 [1.7604, 3.3670] |
| LSHRL (0,1) | 0.0472 [0.0280, 0.0715] | 0.0000 [0.0000, 0.0000] | 0.2664 [0.1968, 0.3445] | 0.2060 [-0.0561, 0.5324] | 2.3594 [1.5485, 3.2398] | 8.3363 [6.4697, 10.2262] | 0.8123 [0.5805, 1.0997] | 3.3608 [2.7056, 4.1317] | 9.1422 [7.4005, 10.9131] | 1.1415 [0.7659, 1.5363] | 2.3932 [1.5281, 3.3518] |
| LSRH (0,1) | 0.0845 [0.0527, 0.1232] | 0.0234 [0.0086, 0.0426] | 0.3100 [0.2259, 0.4009] | 0.3066 [0.0047, 0.6708] | 3.2004 [2.2391, 4.2372] | 9.5079 [7.0923, 11.8941] | 0.9418 [0.6696, 1.2884] | 4.3968 [3.5541, 5.3752] | 11.1165 [9.1719, 13.2105] | 1.4316 [0.9801, 1.9035] | 2.9745 [2.1468, 3.8918] |
| LSRL (0,1) | 0.0454 [0.0238, 0.0728] | 0.0000 [0.0000, 0.0000] | 0.2741 [0.1700, 0.3895] | 0.2064 [-0.0545, 0.5267] | 2.5248 [1.6354, 3.5112] | 8.7001 [6.1813, 11.2690] | 0.8261 [0.5991, 1.1154] | 3.4991 [2.7134, 4.3601] | 9.8022 [7.5079, 12.2263] | 1.2769 [0.7638, 1.8367] | 2.5743 [1.5172, 3.8082] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
