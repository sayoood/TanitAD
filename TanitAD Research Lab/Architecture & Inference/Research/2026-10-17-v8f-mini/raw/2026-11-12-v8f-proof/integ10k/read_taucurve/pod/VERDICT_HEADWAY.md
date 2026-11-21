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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcT08:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcT10:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcT15:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcT20:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50T08:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50T10:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50T15:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50T20:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50T15:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50T20:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcT20:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50T20:I10R0", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10RLcT08 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcT10 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcT15 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcT20 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50T08 minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50T10 minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50T15 minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50T20 minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50T15 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50T20 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcT20 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50T20 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10RLcT08 minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0552 / +0.0480 | +0.0516 | [+0.0224, +0.0894] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0000 / -0.0016 | -0.0008 | [-0.0032, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0498 / -0.0374 | -0.0436 | [-0.0710, -0.0206] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.3140 / -0.3113 | -0.3126 | [-0.5305, -0.1352] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -1.1449 / -0.8266 | -0.9857 | [-1.5971, -0.4531] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -1.8824 / -1.2325 | -1.5574 | [-2.7313, -0.6729] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.2333 / -0.2349 | -0.2341 | [-0.4478, -0.0651] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -0.6714 / -0.4925 | -0.5820 | [-1.0491, -0.1902] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -1.4816 / -0.8286 | -1.1551 | [-2.1511, -0.4513] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.0146 / -0.0204 | -0.0175 | [-0.0351, -0.0031] | 0.0330 (0.0330) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.1438 / -0.0939 | -0.1188 | [-0.1834, -0.0606] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RLcT10 minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0463 / +0.0374 | +0.0418 | [+0.0145, +0.0767] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0000 / -0.0016 | -0.0008 | [-0.0032, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0202 / -0.0140 | -0.0171 | [-0.0369, -0.0005] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1624 / -0.1612 | -0.1618 | [-0.2532, -0.0813] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -0.4435 / -0.3112 | -0.3774 | [-0.7793, -0.0710] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.6903 / -0.0245 | -0.3574 | [-1.5408, +0.5301] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | -0.1118 / -0.1156 | -0.1137 | [-0.2098, -0.0355] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -0.1959 / -0.1502 | -0.1731 | [-0.4468, +0.0586] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -0.4022 / +0.1329 | -0.1347 | [-1.0616, +0.6327] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | -0.0059 / -0.0130 | -0.0095 | [-0.0211, +0.0013] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.0813 / -0.0648 | -0.0730 | [-0.1317, -0.0220] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RLcT15 minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0107 / -0.0071 | -0.0089 | [-0.0184, +0.0009] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [-0.0023, +0.0026] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0685 / +0.0654 | +0.0670 | [+0.0381, +0.1001] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.3629 / +0.3711 | +0.3670 | [+0.2220, +0.5192] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.0338 / +1.6853 | +1.8595 | [+1.1068, +2.6635] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +3.0118 / +2.7614 | +2.8866 | [+1.5693, +4.3449] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.2701 / +0.2741 | +0.2721 | [+0.1408, +0.4159] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +1.6817 / +1.4274 | +1.5546 | [+0.8502, +2.2747] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +2.6573 / +2.6935 | +2.6754 | [+1.4526, +4.0496] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.0260 / +0.0294 | +0.0277 | [+0.0182, +0.0383] | 0.0330 (0.0330) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.2359 / +0.2326 | +0.2343 | [+0.1527, +0.3244] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RLcT20 minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0160 / +0.0018 | -0.0071 | [-0.0323, +0.0136] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0000 / -0.0031 | -0.0016 | [-0.0051, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.1651 / +0.1371 | +0.1511 | [+0.0992, +0.2066] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.1683 / +1.1639 | +1.1661 | [+0.7905, +1.5630] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +4.5027 / +3.7917 | +4.1472 | [+2.6645, +5.7431] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +7.5194 / +6.6297 | +7.0746 | [+4.3383, +10.1168] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.9638 / +0.9601 | +0.9619 | [+0.5993, +1.3382] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +3.9380 / +3.3962 | +3.6671 | [+2.2947, +5.1216] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +6.7847 / +6.5110 | +6.6478 | [+4.0789, +9.6010] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.1280 / +0.1250 | +0.1265 | [+0.0901, +0.1632] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.7992 / +0.6758 | +0.7375 | [+0.5154, +0.9674] | 0.5140 (0.5140) | WORSE |

## I10RQ50T08 minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0178 / +0.0000 | +0.0089 | [-0.0050, +0.0300] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0031 / +0.0016 | +0.0023 | [+0.0000, +0.0055] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0810 / -0.0857 | -0.0833 | [-0.1187, -0.0507] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.4290 / -0.3030 | -0.3660 | [-0.5803, -0.1663] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -1.5963 / -1.3012 | -1.4487 | [-2.2314, -0.7695] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -3.2983 / -2.3215 | -2.8099 | [-4.3235, -1.4917] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.3304 / -0.2257 | -0.2781 | [-0.4716, -0.1032] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -1.0699 / -1.0185 | -1.0442 | [-1.7066, -0.4917] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -2.7405 / -1.9015 | -2.3210 | [-3.6429, -1.1406] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.0153 / -0.0172 | -0.0163 | [-0.0333, +0.0063] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.2493 / -0.2370 | -0.2431 | [-0.3504, -0.1465] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50T10 minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0089 / -0.0018 | +0.0036 | [-0.0071, +0.0154] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0016 / +0.0000 | +0.0008 | [+0.0000, +0.0031] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0343 / -0.0576 | -0.0460 | [-0.0752, -0.0210] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.2505 / -0.1656 | -0.2081 | [-0.3276, -0.1001] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -0.6773 / -0.6729 | -0.6751 | [-1.0558, -0.3420] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -1.4987 / -1.3201 | -1.4094 | [-2.1294, -0.7443] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.2100 / -0.1271 | -0.1685 | [-0.2837, -0.0620] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -0.4303 / -0.5533 | -0.4918 | [-0.8591, -0.1795] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -1.1298 / -1.2005 | -1.1652 | [-1.8860, -0.5613] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.0088 / -0.0114 | -0.0101 | [-0.0236, +0.0095] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.1183 / -0.1503 | -0.1343 | [-0.2099, -0.0678] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50T15 minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0036 / -0.0160 | -0.0098 | [-0.0321, +0.0037] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0047 / +0.0031 | -0.0008 | [-0.0070, +0.0052] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0779 / +0.0514 | +0.0646 | [+0.0332, +0.0993] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4326 / +0.4152 | +0.4239 | [+0.2903, +0.5728] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +1.7025 / +0.8903 | +1.2964 | [+0.5976, +2.1133] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +3.1915 / +2.0152 | +2.6034 | [+1.1420, +4.2331] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3523 / +0.3349 | +0.3436 | [+0.2112, +0.4805] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +1.5210 / +0.7536 | +1.1373 | [+0.5225, +1.9007] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +2.9471 / +1.7044 | +2.3257 | [+0.9636, +3.8933] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.0606 / +0.0457 | +0.0531 | [+0.0338, +0.0767] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.4541 / +0.1435 | +0.2988 | [+0.0956, +0.5589] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50T20 minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0089 / -0.0285 | -0.0187 | [-0.0500, +0.0015] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0156 / -0.0047 | -0.0101 | [-0.0237, -0.0021] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1636 / +0.1573 | +0.1604 | [+0.1155, +0.2056] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.2161 / +1.1769 | +1.1965 | [+0.8808, +1.5407] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +4.6650 / +4.1368 | +4.4009 | [+3.0621, +5.8872] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +6.9975 / +6.2265 | +6.6120 | [+4.3615, +9.1254] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +1.0359 / +0.9759 | +1.0059 | [+0.6991, +1.3391] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +4.2191 / +3.7465 | +3.9828 | [+2.7352, +5.3627] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +6.4067 / +5.7615 | +6.0841 | [+4.0285, +8.4503] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.2140 / +0.1421 | +0.1781 | [+0.1202, +0.2453] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +1.1120 / +1.0137 | +1.0629 | [+0.7864, +1.3948] | 0.5140 (0.5140) | WORSE |

## I10RQ50T15 minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0374 / +0.0267 | +0.0320 | [+0.0131, +0.0521] | 0.0320 (0.0320) | INSIDE-SPREAD |
| C contact rate | +0.0265 / +0.0218 | +0.0241 | [+0.0103, +0.0403] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1215 / +0.0903 | +0.1059 | [+0.0638, +0.1538] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.5883 / +0.3530 | +0.4707 | [+0.2700, +0.7399] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.2060 / +1.1283 | +1.6671 | [+0.8137, +2.7151] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +4.5264 / +2.4896 | +3.5080 | [+1.6605, +5.7960] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.4779 / +0.2806 | +0.3792 | [+0.2071, +0.6168] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +2.0940 / +1.2105 | +1.6522 | [+0.9256, +2.5330] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +4.5469 / +2.7429 | +3.6449 | [+1.9546, +5.6986] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.1963 / +0.0980 | +0.1471 | [+0.0477, +0.2800] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.8805 / +0.4584 | +0.6695 | [+0.3316, +1.0402] | 0.5140 (0.5140) | WORSE |

## I10RQ50T20 minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0320 / +0.0142 | +0.0231 | [+0.0038, +0.0443] | 0.0320 (0.0320) | INSIDE-SPREAD |
| C contact rate | +0.0156 / +0.0140 | +0.0148 | [+0.0045, +0.0266] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.2072 / +0.1963 | +0.2017 | [+0.1474, +0.2552] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.3718 / +1.1147 | +1.2432 | [+0.8841, +1.6656] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +5.1685 / +4.3748 | +4.7716 | [+3.3229, +6.4249] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +8.3324 / +6.7008 | +7.5166 | [+4.9180, +10.5332] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +1.1614 / +0.9216 | +1.0415 | [+0.6980, +1.4390] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +4.7920 / +4.2034 | +4.4977 | [+3.1231, +5.9583] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +8.0066 / +6.8001 | +7.4033 | [+4.9513, +10.0485] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.3497 / +0.1944 | +0.2721 | [+0.1420, +0.4385] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +1.5384 / +1.3286 | +1.4335 | [+1.0494, +1.8592] | 0.5140 (0.5140) | WORSE |

## I10RLcT20 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.1157 / -0.0819 | -0.0988 | [-0.1513, -0.0495] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0763 / -0.0421 | -0.0592 | [-0.0966, -0.0288] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.2539 / +0.2040 | +0.2290 | [+0.1532, +0.3045] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.6146 / +1.6106 | +1.6126 | [+1.0482, +2.2101] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +6.4124 / +5.1731 | +5.7928 | [+3.6394, +8.0501] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +12.2567 / +9.7991 | +11.0279 | [+7.0366, +15.2765] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +1.2788 / +1.2762 | +1.2775 | [+0.7521, +1.8450] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +4.7971 / +4.0949 | +4.4460 | [+2.6989, +6.2453] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +8.2395 / +7.4921 | +7.8658 | [+4.4871, +11.3326] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.1706 / +0.1707 | +0.1707 | [+0.1224, +0.2196] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.9026 / +0.7956 | +0.8491 | [+0.5825, +1.1146] | 0.5140 (0.5140) | WORSE |

## I10RQ50T20 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0676 / -0.0694 | -0.0685 | [-0.1094, -0.0325] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0607 / -0.0249 | -0.0428 | [-0.0780, -0.0162] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.2960 / +0.2632 | +0.2796 | [+0.2084, +0.3500] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.8181 / +1.5614 | +1.6897 | [+1.1792, +2.2446] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +7.0782 / +5.7562 | +6.4172 | [+4.3143, +8.5950] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +13.0696 / +9.8702 | +11.4699 | [+7.6070, +15.6495] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +1.4765 / +1.2378 | +1.3571 | [+0.8785, +1.8616] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +5.6512 / +4.9021 | +5.2767 | [+3.5967, +7.0378] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +9.4614 / +7.7812 | +8.6213 | [+5.5469, +11.8839] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.3923 / +0.2401 | +0.3162 | [+0.1900, +0.4845] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +1.6419 / +1.4484 | +1.5452 | [+1.1371, +1.9811] | 0.5140 (0.5140) | WORSE |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10RLc (0,1) | 0.0383 [0.0190, 0.0628] | 0.0016 [0.0000, 0.0051] | 0.2181 [0.1459, 0.2897] | 0.2229 [-0.0327, 0.5472] | 2.0938 [1.2319, 3.0706] | 6.7474 [4.5816, 8.9276] | 0.7885 [0.5579, 1.0945] | 3.2710 [2.6177, 4.1001] | 8.0958 [6.4712, 9.8010] | 1.0876 [0.7460, 1.4643] | 2.1535 [1.4770, 2.9189] |
| I10RLcT08 (0,1) | 0.0899 [0.0498, 0.1361] | 0.0008 [0.0000, 0.0030] | 0.1745 [0.1019, 0.2492] | -0.0897 [-0.1873, 0.0218] | 1.1081 [0.3327, 1.8579] | 5.1900 [2.9621, 7.4127] | 0.5544 [0.4756, 0.6447] | 2.6890 [2.2739, 3.1732] | 6.9407 [5.4017, 8.5446] | 1.0701 [0.7317, 1.4491] | 2.0347 [1.3400, 2.8111] |
| I10RLcT10 (0,1) | 0.0801 [0.0420, 0.1250] | 0.0008 [0.0000, 0.0030] | 0.2009 [0.1269, 0.2754] | 0.0611 [-0.1280, 0.2979] | 1.7164 [0.7483, 2.7488] | 6.3900 [3.7809, 9.1511] | 0.6748 [0.5141, 0.8835] | 3.0979 [2.4252, 3.9513] | 7.9611 [6.0311, 10.1854] | 1.0782 [0.7393, 1.4563] | 2.0804 [1.3994, 2.8391] |
| I10RLcT15 (0,1) | 0.0294 [0.0146, 0.0502] | 0.0016 [0.0000, 0.0054] | 0.2850 [0.2089, 0.3640] | 0.5899 [0.2268, 1.0048] | 3.9533 [2.6351, 5.4096] | 9.6340 [6.9737, 12.5775] | 1.0606 [0.7299, 1.4444] | 4.8255 [3.6263, 6.1992] | 10.7712 [8.4371, 13.3270] | 1.1153 [0.7718, 1.4906] | 2.3877 [1.7231, 3.1540] |
| I10RLcT20 (0,1) | 0.0311 [0.0181, 0.0450] | 0.0000 [0.0000, 0.0000] | 0.3692 [0.2876, 0.4462] | 1.3891 [0.8323, 1.9891] | 6.2410 [4.3734, 8.1025] | 13.8220 [10.1374, 17.5223] | 1.7504 [1.2259, 2.3023] | 6.9381 [5.1936, 8.7485] | 14.7436 [11.3246, 18.2873] | 1.2141 [0.8727, 1.5853] | 2.8910 [2.2364, 3.6202] |
| I10RQ50 (0,1) | 0.0801 [0.0480, 0.1174] | 0.0265 [0.0102, 0.0465] | 0.2593 [0.1862, 0.3304] | 0.2697 [0.0335, 0.5486] | 2.4645 [1.6073, 3.4225] | 7.6520 [5.6761, 9.6557] | 0.8241 [0.5966, 1.0699] | 3.7859 [3.0818, 4.6228] | 9.4150 [7.8273, 11.0990] | 1.1816 [0.8261, 1.5783] | 2.5241 [1.8524, 3.2957] |
| I10RQ50T08 (0,1) | 0.0890 [0.0470, 0.1349] | 0.0288 [0.0114, 0.0510] | 0.1760 [0.1149, 0.2366] | -0.0963 [-0.1781, 0.0078] | 1.0158 [0.3574, 1.6353] | 4.8421 [2.9793, 6.7657] | 0.5461 [0.4681, 0.6492] | 2.7417 [2.3629, 3.1527] | 7.0940 [5.8095, 8.4375] | 1.1653 [0.8047, 1.5765] | 2.2810 [1.6213, 3.0379] |
| I10RQ50T10 (0,1) | 0.0836 [0.0453, 0.1259] | 0.0273 [0.0104, 0.0490] | 0.2134 [0.1537, 0.2713] | 0.0616 [-0.0818, 0.2337] | 1.7894 [1.1076, 2.5350] | 6.2426 [4.3859, 8.1787] | 0.6556 [0.5258, 0.8049] | 3.2941 [2.7865, 3.9396] | 8.2498 [6.9256, 9.7452] | 1.1715 [0.8139, 1.5835] | 2.3898 [1.7439, 3.1484] |
| I10RQ50T15 (0,1) | 0.0703 [0.0410, 0.1051] | 0.0257 [0.0109, 0.0425] | 0.3240 [0.2554, 0.3894] | 0.6936 [0.3539, 1.0715] | 3.7609 [2.7597, 4.9112] | 10.2554 [8.0196, 12.6377] | 1.1677 [0.8527, 1.5209] | 4.9232 [3.9291, 6.0752] | 11.7407 [9.7795, 13.9196] | 1.2348 [0.8745, 1.6273] | 2.8229 [2.1979, 3.5284] |
| I10RQ50T20 (0,1) | 0.0614 [0.0346, 0.0943] | 0.0164 [0.0059, 0.0292] | 0.4198 [0.3392, 0.4961] | 1.4662 [0.9765, 2.0075] | 6.8655 [5.2067, 8.6443] | 14.2640 [11.2858, 17.4112] | 1.8300 [1.3531, 2.3408] | 7.7687 [6.1379, 9.5209] | 15.4991 [12.7270, 18.4762] | 1.3597 [0.9964, 1.7715] | 3.5870 [2.8866, 4.3488] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
