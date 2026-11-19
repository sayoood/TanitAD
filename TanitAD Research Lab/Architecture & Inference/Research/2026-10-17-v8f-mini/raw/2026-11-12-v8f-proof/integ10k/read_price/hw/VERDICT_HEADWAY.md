# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 12, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.03202846975088969, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50:I10RLb", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50:I10RR", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ35:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ35:I10RLb", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RRaw:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RRaw:I10RLb", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RR:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RR:I10RLb", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10RQ50 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50 minus I10RLb over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50 minus I10RR over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ35 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ35 minus I10RLb over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RRaw minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RRaw minus I10RLb over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RR minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RR minus I10RLb over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10RQ50 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0587 / -0.0409 | -0.0498 | [-0.0868, -0.0174] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0452 / -0.0202 | -0.0327 | [-0.0577, -0.0128] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1324 / +0.1059 | +0.1192 | [+0.0710, +0.1710] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6020 / +0.3845 | +0.4933 | [+0.2312, +0.7833] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.4132 / +1.6194 | +2.0163 | [+1.0353, +3.1714] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +6.0722 / +3.6437 | +4.8579 | [+2.5270, +7.5458] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.4406 / +0.2619 | +0.3513 | [+0.1357, +0.5963] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +1.4321 / +1.1556 | +1.2939 | [+0.6447, +2.0721] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +3.0547 / +2.0196 | +2.5372 | [+1.1229, +4.2045] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.1783 / +0.0979 | +0.1381 | [+0.0529, +0.2535] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.5299 / +0.4347 | +0.4823 | [+0.2771, +0.7058] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50 minus I10RLb

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0018 / +0.0214 | +0.0098 | [-0.0084, +0.0353] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0016 / +0.0062 | +0.0023 | [-0.0056, +0.0094] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0296 / +0.0265 | +0.0280 | [+0.0011, +0.0557] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1543 / -0.0676 | +0.0434 | [-0.1465, +0.2700] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.2648 / -0.1248 | +0.0700 | [-0.4587, +0.7458] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +1.4582 / +0.0405 | +0.7494 | [-0.4212, +2.1483] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0958 / -0.0831 | +0.0063 | [-0.1594, +0.2004] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.0867 / -0.0116 | +0.0376 | [-0.4239, +0.5448] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +1.2687 / +0.3018 | +0.7852 | [-0.1093, +1.8770] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.1135 / +0.0304 | +0.0719 | [-0.0198, +0.1714] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.2585 / +0.0932 | +0.1758 | [-0.0595, +0.3710] | 0.5140 (0.5140) | OPEN |

## I10RQ50 minus I10RR

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0036 / +0.0214 | +0.0125 | [-0.0032, +0.0387] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0031 / +0.0047 | +0.0039 | [+0.0004, +0.0084] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0888 / -0.0701 | -0.0794 | [-0.1187, -0.0468] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.4454 / -0.5061 | -0.4757 | [-0.6491, -0.3206] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -1.9824 / -1.4369 | -1.7097 | [-2.6380, -1.0352] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -3.5589 / -2.9255 | -3.2422 | [-5.0085, -1.8465] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.3636 / -0.4014 | -0.3825 | [-0.5396, -0.2349] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -1.7116 / -1.2169 | -1.4643 | [-2.2769, -0.8570] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -3.1979 / -2.5072 | -2.8526 | [-4.5327, -1.5655] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.0441 / -0.0937 | -0.0689 | [-0.1204, -0.0358] | 0.0330 (0.0330) | BETTER |
| time-gap error over the scored steps (s) | -0.4274 / -0.4218 | -0.4246 | [-0.6048, -0.2722] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ35 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0569 / -0.0534 | -0.0552 | [-0.0913, -0.0225] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0436 / -0.0249 | -0.0343 | [-0.0573, -0.0156] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1667 / +0.1340 | +0.1503 | [+0.0972, +0.2082] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7143 / +0.5732 | +0.6437 | [+0.3801, +0.9409] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +3.3592 / +2.2515 | +2.8053 | [+1.5941, +4.2057] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +7.3860 / +4.8374 | +6.1117 | [+3.4459, +9.3023] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.5278 / +0.4064 | +0.4671 | [+0.2459, +0.7241] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +2.3026 / +1.6555 | +1.9791 | [+1.0808, +2.9794] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +4.4045 / +2.9680 | +3.6862 | [+1.8505, +5.9126] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.1845 / +0.1221 | +0.1533 | [+0.0710, +0.2628] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.6253 / +0.5793 | +0.6023 | [+0.4043, +0.8271] | 0.5140 (0.5140) | WORSE |

## I10RQ35 minus I10RLb

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0089 | +0.0044 | [-0.0066, +0.0159] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0000 / +0.0016 | +0.0008 | [-0.0049, +0.0056] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0639 / +0.0545 | +0.0592 | [+0.0282, +0.0955] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2666 / +0.1211 | +0.1939 | [+0.0400, +0.3961] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +1.2108 / +0.5073 | +0.8590 | [+0.1686, +1.7986] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.7720 / +1.2342 | +2.0031 | [+0.6758, +4.1551] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.1830 / +0.0613 | +0.1221 | [-0.0113, +0.3029] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.9573 / +0.4883 | +0.7228 | [+0.1456, +1.4697] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +2.6185 / +1.2501 | +1.9343 | [+0.7137, +3.8379] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.1196 / +0.0546 | +0.0871 | [+0.0043, +0.1818] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.3539 / +0.2377 | +0.2958 | [+0.0869, +0.4813] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RRaw minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0552 / -0.0409 | -0.0480 | [-0.0840, -0.0168] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0452 / -0.0202 | -0.0327 | [-0.0577, -0.0128] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1215 / +0.0872 | +0.1044 | [+0.0580, +0.1543] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.5497 / +0.3681 | +0.4589 | [+0.2219, +0.7272] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.2489 / +1.4328 | +1.8408 | [+0.8575, +2.9259] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +5.9341 / +3.3846 | +4.6593 | [+2.1983, +7.3373] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.3920 / +0.2480 | +0.3200 | [+0.1221, +0.5455] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +1.3825 / +1.0017 | +1.1921 | [+0.5514, +1.9483] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +2.9417 / +1.7797 | +2.3607 | [+0.9331, +4.0033] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.1755 / +0.0985 | +0.1370 | [+0.0527, +0.2508] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.5108 / +0.3884 | +0.4496 | [+0.2345, +0.6782] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RRaw minus I10RLb

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0018 / +0.0214 | +0.0116 | [-0.0039, +0.0357] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0016 / +0.0062 | +0.0023 | [-0.0056, +0.0094] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0187 / +0.0078 | +0.0132 | [-0.0141, +0.0441] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1020 / -0.0840 | +0.0090 | [-0.1718, +0.2077] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.1005 / -0.3114 | -0.1055 | [-0.6203, +0.5413] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +1.3202 / -0.2187 | +0.5508 | [-0.6131, +2.1681] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0471 / -0.0970 | -0.0249 | [-0.1800, +0.1445] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.0371 / -0.1655 | -0.0642 | [-0.5221, +0.4572] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +1.1558 / +0.0619 | +0.6088 | [-0.2864, +1.8965] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.1106 / +0.0309 | +0.0708 | [-0.0203, +0.1697] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.2395 / +0.0468 | +0.1431 | [-0.1092, +0.3513] | 0.5140 (0.5140) | OPEN |

## I10RR minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0623 / -0.0623 | -0.0623 | [-0.1010, -0.0271] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0483 / -0.0249 | -0.0366 | [-0.0621, -0.0154] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.2212 / +0.1760 | +0.1986 | [+0.1350, +0.2639] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.0474 / +0.8906 | +0.9690 | [+0.6343, +1.3430] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +4.3956 / +3.0563 | +3.7259 | [+2.2628, +5.4301] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +9.6311 / +6.5692 | +8.1001 | [+4.8657, +11.7081] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.8042 / +0.6633 | +0.7338 | [+0.4323, +1.0681] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +3.1437 / +2.3726 | +2.7581 | [+1.6742, +3.9674] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +6.2526 / +4.5268 | +5.3897 | [+3.0308, +8.1049] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.2225 / +0.1916 | +0.2070 | [+0.1240, +0.3078] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.9573 / +0.8565 | +0.9069 | [+0.6102, +1.2294] | 0.5140 (0.5140) | WORSE |

## I10RR minus I10RLb

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0053 / +0.0000 | -0.0027 | [-0.0138, +0.0072] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0047 / +0.0016 | -0.0016 | [-0.0103, +0.0050] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.1184 / +0.0966 | +0.1075 | [+0.0665, +0.1551] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.5997 / +0.4385 | +0.5191 | [+0.3166, +0.7762] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.2472 / +1.3121 | +1.7796 | [+0.8947, +3.0539] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +5.0171 / +2.9660 | +3.9916 | [+2.0027, +6.6453] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.4594 / +0.3182 | +0.3888 | [+0.2088, +0.6152] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +1.7983 / +1.2054 | +1.5018 | [+0.7717, +2.4935] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +4.4667 / +2.8089 | +3.6378 | [+1.8356, +5.9944] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.1576 / +0.1240 | +0.1408 | [+0.0733, +0.2280] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.6860 / +0.5149 | +0.6005 | [+0.3220, +0.8985] | 0.5140 (0.5140) | WORSE |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10RLb (0,1) | 0.0703 [0.0382, 0.1113] | 0.0241 [0.0065, 0.0484] | 0.2313 [0.1606, 0.3039] | 0.2263 [-0.0383, 0.5548] | 2.3946 [1.2957, 3.6479] | 6.9027 [4.4318, 9.5247] | 0.8178 [0.5897, 1.1089] | 3.7483 [2.8924, 4.8297] | 8.6297 [6.8435, 10.6964] | 1.1097 [0.7592, 1.4936] | 2.3483 [1.6463, 3.1694] |
| I10RQ35 (0,1) | 0.0747 [0.0432, 0.1112] | 0.0249 [0.0073, 0.0473] | 0.2905 [0.2196, 0.3608] | 0.4202 [0.1710, 0.7030] | 3.2536 [2.3216, 4.3317] | 8.9058 [6.6511, 11.2610] | 0.9400 [0.7104, 1.2026] | 4.4711 [3.5979, 5.4829] | 10.5641 [8.7368, 12.5698] | 1.1968 [0.8374, 1.5864] | 2.6441 [1.9645, 3.4206] |
| I10RQ50 (0,1) | 0.0801 [0.0480, 0.1174] | 0.0265 [0.0102, 0.0465] | 0.2593 [0.1862, 0.3304] | 0.2697 [0.0335, 0.5486] | 2.4645 [1.6073, 3.4225] | 7.6520 [5.6761, 9.6557] | 0.8241 [0.5966, 1.0699] | 3.7859 [3.0818, 4.6228] | 9.4150 [7.8273, 11.0990] | 1.1816 [0.8261, 1.5783] | 2.5241 [1.8524, 3.2957] |
| I10RR (0,1) | 0.0676 [0.0393, 0.1055] | 0.0226 [0.0069, 0.0418] | 0.3388 [0.2669, 0.4094] | 0.7454 [0.4244, 1.1084] | 4.1742 [3.0928, 5.4141] | 10.8942 [8.3762, 13.4805] | 1.2067 [0.9032, 1.5434] | 5.2502 [4.1900, 6.4470] | 12.2675 [10.1056, 14.6726] | 1.2505 [0.8831, 1.6481] | 2.9488 [2.2494, 3.7525] |
| I10RRaw (0,1) | 0.0819 [0.0487, 0.1198] | 0.0265 [0.0102, 0.0465] | 0.2445 [0.1775, 0.3105] | 0.2353 [0.0184, 0.4896] | 2.2891 [1.4967, 3.2225] | 7.4534 [5.5496, 9.4877] | 0.7929 [0.5869, 1.0179] | 3.6841 [3.0234, 4.4925] | 9.2386 [7.7462, 10.8750] | 1.1804 [0.8245, 1.5759] | 2.4914 [1.8345, 3.2540] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
