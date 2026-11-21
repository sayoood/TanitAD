# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: /workspace/v8proof/podread/data/join_eval141.npz. Alignment reference (ego speed): /workspace/v8proof/podread/data/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 24, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.03202846975088969, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0B5:I10R0Bk10", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcBk5:I10RLcBk10", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50Bk5:I10RQ50Bk10", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcBk5:I10R0B5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50Bk5:I10R0B5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcBk10:I10R0Bk10", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50Bk10:I10R0Bk10", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcBk5:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50Bk5:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0Bk10:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcBk10:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50Bk10:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcZ5:I10R0Z5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50Z5:I10R0Z5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcZ5:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50Z5:I10RQ50", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10R0B5 minus I10R0Bk10 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcBk5 minus I10RLcBk10 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50Bk5 minus I10RQ50Bk10 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcBk5 minus I10R0B5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50Bk5 minus I10R0B5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcBk10 minus I10R0Bk10 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50Bk10 minus I10R0Bk10 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcBk5 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50Bk5 minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0Bk10 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcBk10 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50Bk10 minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcZ5 minus I10R0Z5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50Z5 minus I10R0Z5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcZ5 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50Z5 minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10R0B5 minus I10R0Bk10

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0463 / +0.0480 | +0.0472 | [+0.0221, +0.0746] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0202 / +0.0234 | +0.0218 | [+0.0089, +0.0366] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0202 / +0.0078 | +0.0140 | [-0.0145, +0.0427] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1570 / +0.1713 | +0.1641 | [+0.0975, +0.2406] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | -0.0669 / -0.2213 | -0.1441 | [-0.4606, +0.1784] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | -1.2169 / -1.9485 | -1.5827 | [-2.6613, -0.4710] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0927 / +0.1179 | +0.1053 | [+0.0556, +0.1605] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.6688 / +0.6659 | +0.6674 | [+0.4276, +0.9075] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +1.7175 / +1.2612 | +1.4893 | [+0.7221, +2.2698] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0578 / +0.1009 | +0.0793 | [-0.2168, +0.4256] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.2725 / +0.1456 | +0.2090 | [-0.0216, +0.4353] | 0.5140 (0.5140) | OPEN |

## I10RLcBk5 minus I10RLcBk10

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0196 / +0.0125 | +0.0160 | [+0.0016, +0.0337] | 0.0320 (0.0320) | INSIDE-SPREAD |
| C contact rate | -0.0016 / -0.0031 | -0.0023 | [-0.0064, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0125 / +0.0078 | +0.0101 | [-0.0178, +0.0375] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.3915 / +0.3776 | +0.3845 | [+0.2383, +0.5619] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +0.4880 / +0.4487 | +0.4684 | [+0.1267, +0.8190] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -1.1308 / -1.1325 | -1.1316 | [-2.0294, -0.2508] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.2783 / +0.2793 | +0.2788 | [+0.1344, +0.4564] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.6844 / +0.7542 | +0.7193 | [+0.4366, +1.0460] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | -0.1176 / +0.0875 | -0.0150 | [-0.8679, +0.7749] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0432 / +0.0323 | +0.0377 | [-0.2545, +0.3701] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.1858 / +0.1640 | +0.1749 | [-0.0275, +0.4065] | 0.5140 (0.5140) | OPEN |

## I10RQ50Bk5 minus I10RQ50Bk10

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0267 / +0.0214 | +0.0240 | [+0.0024, +0.0478] | 0.0320 (0.0320) | INSIDE-SPREAD |
| C contact rate | -0.0047 / -0.0031 | -0.0039 | [-0.0135, +0.0059] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0125 / +0.0062 | -0.0031 | [-0.0343, +0.0295] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.3938 / +0.2720 | +0.3329 | [+0.1875, +0.4896] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +0.3430 / +0.1724 | +0.2577 | [-0.0839, +0.5814] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | -1.9403 / -1.2450 | -1.5926 | [-2.6055, -0.6918] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3315 / +0.2469 | +0.2892 | [+0.1657, +0.4301] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.6159 / +0.6158 | +0.6159 | [+0.3479, +0.9064] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | -0.8909 / -0.0617 | -0.4763 | [-1.5326, +0.4226] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.2387 / +0.1591 | +0.1989 | [-0.1412, +0.5901] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.0995 / +0.0038 | -0.0479 | [-0.3024, +0.2422] | 0.5140 (0.5140) | OPEN |

## I10RLcBk5 minus I10R0B5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0854 / -0.0907 | -0.0881 | [-0.1239, -0.0565] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0857 / -0.0857 | -0.0857 | [-0.1265, -0.0480] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0654 / +0.0701 | +0.0678 | [+0.0256, +0.1207] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6808 / +0.6529 | +0.6669 | [+0.2626, +1.1639] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.1521 / +2.0178 | +2.0850 | [+1.0713, +3.2942] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +4.1381 / +3.9783 | +4.0582 | [+2.3727, +5.9767] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.5007 / +0.4776 | +0.4892 | [+0.1094, +0.9652] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.6774 / +0.6381 | +0.6577 | [-0.2235, +1.7331] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -0.2416 / -0.1393 | -0.1904 | [-1.6434, +1.4143] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0280 / -0.0229 | +0.0025 | [-0.0773, +0.0626] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.0805 / -0.0673 | -0.0739 | [-0.1564, +0.0059] | 0.5140 (0.5140) | OPEN |

## I10RQ50Bk5 minus I10R0B5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0445 / -0.0409 | -0.0427 | [-0.0699, -0.0192] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0421 / -0.0343 | -0.0382 | [-0.0585, -0.0210] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1044 / +0.0717 | +0.0880 | [+0.0434, +0.1387] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.8388 / +0.4852 | +0.6620 | [+0.2690, +1.1101] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.5648 / +1.4887 | +2.0267 | [+0.9672, +3.2351] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +5.0883 / +3.1445 | +4.1164 | [+2.3115, +6.2161] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.6794 / +0.3909 | +0.5351 | [+0.1889, +0.9236] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +1.3071 / +0.5992 | +0.9531 | [+0.1105, +1.9194] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +1.1637 / -0.0668 | +0.5484 | [-0.8679, +2.0946] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.3593 / +0.1561 | +0.2577 | [+0.0331, +0.6400] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.2243 / +0.0822 | +0.1533 | [+0.0128, +0.3870] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RLcBk10 minus I10R0Bk10

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0587 / -0.0552 | -0.0569 | [-0.0876, -0.0313] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0639 / -0.0592 | -0.0615 | [-0.0923, -0.0341] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0732 / +0.0701 | +0.0717 | [+0.0314, +0.1202] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4463 / +0.4467 | +0.4465 | [+0.1858, +0.7734] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +1.5972 / +1.3478 | +1.4725 | [+0.6932, +2.4102] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +4.0520 / +3.1624 | +3.6072 | [+1.8457, +5.6618] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3151 / +0.3162 | +0.3156 | [+0.0869, +0.6111] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.6617 / +0.5498 | +0.6058 | [-0.0272, +1.3989] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +1.5934 / +1.0344 | +1.3139 | [-0.1015, +2.9794] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0426 / +0.0456 | +0.0441 | [+0.0199, +0.0728] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.0061 / -0.0858 | -0.0398 | [-0.1553, +0.0544] | 0.5140 (0.5140) | OPEN |

## I10RQ50Bk10 minus I10R0Bk10

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0249 / -0.0142 | -0.0196 | [-0.0400, -0.0061] | 0.0320 (0.0320) | INSIDE-SPREAD |
| C contact rate | -0.0171 / -0.0078 | -0.0125 | [-0.0245, -0.0038] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1371 / +0.0732 | +0.1051 | [+0.0489, +0.1718] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6020 / +0.3845 | +0.4933 | [+0.2312, +0.7833] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.1549 / +1.0949 | +1.6249 | [+0.7196, +2.6981] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +5.8116 / +2.4410 | +4.1263 | [+1.7429, +7.1406] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.4406 / +0.2619 | +0.3513 | [+0.1357, +0.5963] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +1.3599 / +0.6493 | +1.0046 | [+0.3461, +1.7907] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +3.7721 / +1.2561 | +2.5141 | [+0.7017, +4.8663] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.1783 / +0.0979 | +0.1381 | [+0.0529, +0.2535] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.5963 / +0.2240 | +0.4101 | [+0.1408, +0.7893] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RLcBk5 minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0214 / +0.0302 | +0.0258 | [+0.0039, +0.0543] | 0.0320 (0.0320) | INSIDE-SPREAD |
| C contact rate | +0.0000 / -0.0031 | -0.0016 | [-0.0051, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0296 / -0.0841 | -0.0569 | [-0.1239, +0.0035] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.3915 / +0.3776 | +0.3845 | [+0.2383, +0.5619] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +0.0158 / -0.3753 | -0.1797 | [-0.7604, +0.3572] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | -2.1475 / -3.9424 | -3.0449 | [-4.8714, -1.2960] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.2783 / +0.2793 | +0.2788 | [+0.1344, +0.4564] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.3755 / +0.3104 | +0.3429 | [-0.0573, +0.7248] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -0.5977 / -1.6632 | -1.1305 | [-2.5501, +0.2907] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0432 / +0.0323 | +0.0377 | [-0.2545, +0.3701] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.0086 / -0.4501 | -0.2293 | [-0.7454, +0.1950] | 0.5140 (0.5140) | OPEN |

## I10RQ50Bk5 minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0214 / +0.0374 | +0.0294 | [-0.0085, +0.0749] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0125 / +0.0296 | +0.0210 | [-0.0002, +0.0461] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0343 / -0.1215 | -0.0779 | [-0.1592, -0.0023] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.3938 / +0.2720 | +0.3329 | [+0.1875, +0.4896] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | -0.0750 / -1.1424 | -0.6087 | [-1.4628, +0.2122] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | -2.5322 / -5.2505 | -3.8914 | [-6.1719, -1.8043] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.3315 / +0.2469 | +0.2892 | [+0.1657, +0.4301] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.4322 / -0.1854 | +0.1234 | [-0.4589, +0.6933] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -0.7923 / -2.6293 | -1.7108 | [-3.5206, -0.0242] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.2387 / +0.1591 | +0.1989 | [-0.1412, +0.5901] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.1302 / -0.6154 | -0.3728 | [-0.9616, +0.1108] | 0.5140 (0.5140) | OPEN |

## I10R0Bk10 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0391 / -0.0107 | -0.0249 | [-0.0652, +0.0187] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0109 / +0.0202 | +0.0047 | [-0.0295, +0.0342] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0265 / -0.0950 | -0.0607 | [-0.1217, -0.0108] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.1597 / -0.7904 | -0.4751 | [-0.9851, -0.0132] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.3314 / -2.8028 | -1.5671 | [-3.4441, +0.1901] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.1115 / -0.2948 | -0.2032 | [-0.4478, +0.0212] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -0.6187 / -1.8041 | -1.2114 | [-2.3821, -0.2830] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.0971 / -0.4085 | -0.2528 | [-0.6047, +0.0156] | 0.5140 (0.5140) | OPEN |

## I10RLcBk10 minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0018 / +0.0178 | +0.0098 | [-0.0085, +0.0352] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0016 / +0.0000 | +0.0008 | [-0.0022, +0.0037] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0421 / -0.0919 | -0.0670 | [-0.1190, -0.0236] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.4722 / -0.8240 | -0.6481 | [-0.9956, -0.3076] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -1.0167 / -2.8098 | -1.9133 | [-3.2492, -0.5564] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.3090 / -0.4438 | -0.3764 | [-0.6048, -0.1601] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -0.4801 / -1.7508 | -1.1154 | [-2.2104, -0.0778] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.1944 / -0.6141 | -0.4043 | [-0.8048, -0.0883] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50Bk10 minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0053 / +0.0160 | +0.0053 | [-0.0213, +0.0412] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0171 / +0.0327 | +0.0249 | [+0.0075, +0.0476] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0218 / -0.1277 | -0.0748 | [-0.1531, -0.0048] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.4180 / -1.3149 | -0.8664 | [-1.5437, -0.2045] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.5920 / -4.0055 | -2.2987 | [-4.5145, -0.1489] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.1837 / -0.8012 | -0.4925 | [-1.0246, +0.0031] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +0.0986 / -2.5676 | -1.2345 | [-3.1797, +0.5050] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.0307 / -0.6192 | -0.3249 | [-0.8341, +0.1207] | 0.5140 (0.5140) | OPEN |

## I10RLcZ5 minus I10R0Z5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.1317 / -0.1263 | -0.1290 | [-0.1838, -0.0823] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.1199 / -0.1137 | -0.1168 | [-0.1669, -0.0713] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0436 / +0.0452 | +0.0444 | [+0.0039, +0.0971] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6464 / +0.6281 | +0.6373 | [+0.2253, +1.1363] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.1920 / +2.1304 | +2.1612 | [+1.1382, +3.4056] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +4.5687 / +4.7086 | +4.6386 | [+2.7694, +6.7188] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.4191 / +0.3996 | +0.4094 | [+0.0320, +0.8827] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.0961 / -0.0330 | +0.0316 | [-0.8528, +1.1052] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -1.8517 / -1.9480 | -1.8998 | [-3.5053, -0.1518] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.0122 / -0.0484 | -0.0181 | [-0.1018, +0.0464] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.0817 / -0.2139 | -0.1478 | [-0.2971, -0.0353] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50Z5 minus I10R0Z5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0819 / -0.0676 | -0.0747 | [-0.1192, -0.0392] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0592 / -0.0467 | -0.0530 | [-0.0815, -0.0301] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0794 / +0.0296 | +0.0545 | [+0.0106, +0.1044] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7790 / +0.4789 | +0.6290 | [+0.2711, +1.0279] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.4741 / +1.6647 | +2.0694 | [+1.1644, +3.1018] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +5.3693 / +4.2381 | +4.8037 | [+3.2278, +6.5777] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.5644 / +0.3383 | +0.4513 | [+0.1458, +0.7987] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.5687 / -0.0350 | +0.2669 | [-0.4516, +1.0697] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -0.8791 / -2.0715 | -1.4753 | [-2.9341, -0.0043] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.2958 / +0.1307 | +0.2132 | [+0.0303, +0.5206] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.2588 / -0.0940 | +0.0824 | [-0.1543, +0.4580] | 0.5140 (0.5140) | OPEN |

## I10RLcZ5 minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0427 / +0.0552 | +0.0489 | [+0.0195, +0.0830] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0016 / -0.0031 | -0.0008 | [-0.0051, +0.0030] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0919 / -0.1464 | -0.1192 | [-0.1875, -0.0602] | 0.0903 (0.0903) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1831 / +0.1623 | +0.1727 | [+0.0190, +0.3568] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | -1.4432 / -1.8760 | -1.6596 | [-2.3113, -1.0915] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -5.4621 / -7.4083 | -6.4352 | [-8.4670, -4.5219] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.2883 / +0.3091 | +0.2987 | [+0.1613, +0.4769] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.2076 / +0.1686 | +0.1881 | [-0.2413, +0.6074] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -1.2817 / -2.1711 | -1.7264 | [-3.2623, -0.2652] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.1935 / +0.1893 | +0.1914 | [-0.1213, +0.5858] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.1010 / -0.3665 | -0.1327 | [-0.7005, +0.3828] | 0.5140 (0.5140) | OPEN |

## I10RQ50Z5 minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0516 / +0.0712 | +0.0614 | [+0.0240, +0.1059] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0312 / +0.0452 | +0.0382 | [+0.0150, +0.0661] | 0.0343 (0.0343) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0997 / -0.2009 | -0.1503 | [-0.2371, -0.0698] | 0.0903 (0.0903) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1601 / +0.0753 | +0.1177 | [-0.0122, +0.2493] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -1.6645 / -2.5796 | -2.1221 | [-2.9079, -1.3784] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -5.9963 / -8.3531 | -7.1747 | [-9.2309, -5.2329] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.3080 / +0.3020 | +0.3050 | [+0.2149, +0.4132] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.1073 / -0.2903 | -0.0915 | [-0.6405, +0.4615] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -1.9091 / -3.3331 | -2.6211 | [-4.3707, -0.9748] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.3414 / +0.3162 | +0.3288 | [-0.0096, +0.7172] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.0151 / -0.5616 | -0.2732 | [-0.9471, +0.2964] | 0.5140 (0.5140) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10R0B5 (0,1) | 0.1521 [0.1018, 0.2023] | 0.0857 [0.0480, 0.1265] | 0.0935 [0.0593, 0.1300] | -0.0594 [-0.1529, 0.0442] | -0.1709 [-0.7715, 0.4592] | -0.3557 [-1.9097, 1.2465] | 0.5782 [0.5133, 0.6566] | 2.9562 [2.5788, 3.4042] | 7.1557 [6.1615, 8.2528] | 1.1228 [0.7390, 1.6089] | 1.9981 [1.2943, 2.8406] |
| I10R0Bk10 (0,1) | 0.1050 [0.0650, 0.1487] | 0.0639 [0.0352, 0.0961] | 0.0794 [0.0495, 0.1130] | -0.2236 [-0.2757, -0.1682] | -0.0268 [-0.4978, 0.4853] | 1.2270 [-0.0581, 2.5934] | 0.4729 [0.4299, 0.5219] | 2.2888 [2.0418, 2.5813] | 5.6664 [4.9525, 6.4671] | 1.0435 [0.7059, 1.4212] | 1.7891 [1.1950, 2.5049] |
| I10R0Z5 (0,1) | 0.2162 [0.1489, 0.2851] | 0.1176 [0.0714, 0.1683] | 0.0545 [0.0323, 0.0815] | -0.2416 [-0.3451, -0.1223] | -1.7270 [-2.3181, -1.0712] | -4.3264 [-5.9363, -2.5736] | 0.6778 [0.6023, 0.7674] | 3.4275 [2.9799, 3.9379] | 8.2692 [7.0430, 9.5783] | 1.2971 [0.8284, 1.8486] | 2.1685 [1.3787, 3.1127] |
| I10RLc (0,1) | 0.0383 [0.0190, 0.0628] | 0.0016 [0.0000, 0.0051] | 0.2181 [0.1459, 0.2897] | 0.2229 [-0.0327, 0.5472] | 2.0938 [1.2319, 3.0706] | 6.7474 [4.5816, 8.9276] | 0.7885 [0.5579, 1.0945] | 3.2710 [2.6177, 4.1001] | 8.0958 [6.4712, 9.8010] | 1.0876 [0.7460, 1.4643] | 2.1535 [1.4770, 2.9189] |
| I10RLcBk10 (0,1) | 0.0480 [0.0230, 0.0772] | 0.0023 [0.0000, 0.0064] | 0.1511 [0.1047, 0.2049] | 0.2229 [-0.0327, 0.5472] | 1.4457 [0.6861, 2.3719] | 4.8341 [3.1657, 6.7134] | 0.7885 [0.5579, 1.0945] | 2.8946 [2.2792, 3.6900] | 6.9803 [5.6310, 8.6990] | 1.0876 [0.7460, 1.4643] | 1.7492 [1.1990, 2.4270] |
| I10RLcBk5 (0,1) | 0.0641 [0.0355, 0.0978] | 0.0000 [0.0000, 0.0000] | 0.1612 [0.1092, 0.2193] | 0.6075 [0.2183, 1.1054] | 1.9141 [0.9400, 3.0994] | 3.7025 [2.0257, 5.6211] | 1.0673 [0.6998, 1.5375] | 3.6139 [2.7761, 4.6709] | 6.9653 [5.6818, 8.5622] | 1.1254 [0.7552, 1.5883] | 1.9241 [1.2273, 2.7654] |
| I10RLcZ5 (0,1) | 0.0872 [0.0512, 0.1278] | 0.0008 [0.0000, 0.0031] | 0.0989 [0.0535, 0.1555] | 0.3956 [0.0026, 0.8984] | 0.4342 [-0.5565, 1.6366] | 0.3122 [-1.3954, 2.2817] | 1.0872 [0.7281, 1.5643] | 3.4591 [2.6698, 4.4985] | 6.3694 [5.0955, 7.9124] | 1.2791 [0.8328, 1.8223] | 2.0207 [1.2697, 2.9602] |
| I10RQ50 (0,1) | 0.0801 [0.0480, 0.1174] | 0.0265 [0.0102, 0.0465] | 0.2593 [0.1862, 0.3304] | 0.2697 [0.0335, 0.5486] | 2.4645 [1.6073, 3.4225] | 7.6520 [5.6761, 9.6557] | 0.8241 [0.5966, 1.0699] | 3.7859 [3.0818, 4.6228] | 9.4150 [7.8273, 11.0990] | 1.1816 [0.8261, 1.5783] | 2.5241 [1.8524, 3.2957] |
| I10RQ50Bk10 (0,1) | 0.0854 [0.0513, 0.1247] | 0.0514 [0.0253, 0.0799] | 0.1846 [0.1319, 0.2427] | 0.2697 [0.0335, 0.5486] | 1.5981 [0.8130, 2.4906] | 5.3533 [3.3202, 7.8042] | 0.8241 [0.5966, 1.0699] | 3.2934 [2.6370, 4.0491] | 8.1805 [6.5176, 10.3430] | 1.1816 [0.8261, 1.5783] | 2.1992 [1.5604, 2.9262] |
| I10RQ50Bk5 (0,1) | 0.1094 [0.0681, 0.1563] | 0.0475 [0.0220, 0.0764] | 0.1815 [0.1287, 0.2375] | 0.6026 [0.2340, 1.0181] | 1.8558 [0.8376, 2.9639] | 3.7607 [1.8887, 5.8070] | 1.1133 [0.7791, 1.4922] | 3.9093 [3.0987, 4.8175] | 7.7042 [6.2741, 9.2920] | 1.3805 [0.9186, 1.9097] | 2.1513 [1.4241, 2.9855] |
| I10RQ50Z5 (0,1) | 0.1415 [0.0924, 0.1963] | 0.0646 [0.0342, 0.0960] | 0.1090 [0.0653, 0.1551] | 0.3874 [0.0401, 0.7753] | 0.3424 [-0.5730, 1.3338] | 0.4773 [-1.1814, 2.2684] | 1.1292 [0.8409, 1.4651] | 3.6944 [3.0753, 4.3975] | 6.7939 [5.6902, 7.9936] | 1.5104 [1.0094, 2.0834] | 2.2509 [1.4605, 3.1417] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
