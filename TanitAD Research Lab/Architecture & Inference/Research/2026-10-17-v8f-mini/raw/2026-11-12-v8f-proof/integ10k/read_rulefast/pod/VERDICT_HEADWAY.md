# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: /workspace/v8proof/podread/data/join_eval141.npz. Alignment reference (ego speed): /workspace/v8proof/podread/data/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 12, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.03202846975088969, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcF5:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50F5:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcF2:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcF2:I10RLcF5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcF5:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50F5:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcF2:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50F5:I10RLcF5", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10RLcF5 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50F5 minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcF2 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcF2 minus I10RLcF5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcF5 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50F5 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcF2 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50F5 minus I10RLcF5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10RLcF5 minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0125 / -0.0053 | -0.0089 | [-0.0176, -0.0011] | 0.0320 (0.0320) | INSIDE-SPREAD |
| C contact rate | +0.0000 / -0.0031 | -0.0016 | [-0.0051, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0016 / +0.0047 | +0.0031 | [-0.0023, +0.0098] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0409 / +0.0453 | +0.0431 | [+0.0249, +0.0651] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.0653 / +0.0092 | +0.0373 | [-0.0538, +0.1221] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +0.0146 / +0.1285 | +0.0715 | [-0.1442, +0.2340] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0081 / +0.0090 | +0.0086 | [-0.0071, +0.0254] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.0024 / -0.0268 | -0.0146 | [-0.0899, +0.0537] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -0.0479 / +0.1172 | +0.0346 | [-0.1905, +0.2036] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0160 / +0.0127 | +0.0143 | [+0.0029, +0.0288] | 0.0330 (0.0330) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0168 / +0.0206 | +0.0187 | [-0.0065, +0.0438] | 0.5140 (0.5140) | OPEN |

## I10RQ50F5 minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0125 / -0.0053 | -0.0089 | [-0.0213, +0.0004] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0156 / -0.0047 | -0.0101 | [-0.0235, -0.0022] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0016 / +0.0031 | +0.0008 | [-0.0081, +0.0086] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1121 / +0.0663 | +0.0892 | [+0.0462, +0.1400] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +0.3800 / +0.2530 | +0.3165 | [+0.1656, +0.4963] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.1844 / +0.4078 | +0.2961 | [-0.0334, +0.5886] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0685 / +0.0283 | +0.0484 | [+0.0086, +0.0952] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.2125 / +0.1171 | +0.1648 | [+0.0215, +0.3398] | 0.2417 (0.2417) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.0198 / +0.1687 | +0.0943 | [-0.1790, +0.3372] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0281 / +0.0071 | +0.0176 | [+0.0036, +0.0441] | 0.0330 (0.0330) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0416 / +0.0781 | +0.0598 | [-0.0160, +0.1261] | 0.5140 (0.5140) | OPEN |

## I10RLcF2 minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0071 / -0.0107 | -0.0089 | [-0.0191, +0.0002] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0000 / -0.0031 | -0.0016 | [-0.0051, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0016 / +0.0062 | +0.0039 | [-0.0023, +0.0104] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0586 / +0.0597 | +0.0592 | [+0.0347, +0.0888] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +0.0231 / +0.1296 | +0.0764 | [-0.0672, +0.2110] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +0.2155 / +0.3208 | +0.2682 | [-0.0063, +0.5372] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0246 / +0.0234 | +0.0240 | [+0.0041, +0.0487] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | -0.0620 / +0.0292 | -0.0164 | [-0.1463, +0.0921] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +0.1238 / +0.2999 | +0.2119 | [-0.0763, +0.5003] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0219 / +0.0203 | +0.0211 | [+0.0081, +0.0367] | 0.0330 (0.0330) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0172 / +0.0539 | +0.0356 | [-0.0082, +0.0759] | 0.5140 (0.5140) | OPEN |

## I10RLcF2 minus I10RLcF5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0053 / -0.0053 | +0.0000 | [-0.0090, +0.0096] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0000 / +0.0016 | +0.0008 | [-0.0054, +0.0078] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0177 / +0.0145 | +0.0161 | [+0.0081, +0.0259] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.0422 / +0.1204 | +0.0391 | [-0.1070, +0.2119] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +0.2010 / +0.1924 | +0.1967 | [+0.0693, +0.3465] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0166 / +0.0144 | +0.0155 | [+0.0077, +0.0252] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | -0.0596 / +0.0559 | -0.0018 | [-0.1252, +0.1345] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +0.1717 / +0.1828 | +0.1772 | [+0.0447, +0.3415] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0059 / +0.0076 | +0.0068 | [+0.0030, +0.0119] | 0.0330 (0.0330) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0004 / +0.0333 | +0.0169 | [-0.0139, +0.0484] | 0.5140 (0.5140) | OPEN |

## I10RLcF5 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.1121 / -0.0890 | -0.1005 | [-0.1483, -0.0575] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0763 / -0.0421 | -0.0592 | [-0.0966, -0.0288] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0903 / +0.0717 | +0.0810 | [+0.0384, +0.1276] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4872 / +0.4919 | +0.4896 | [+0.2142, +0.8279] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +1.9750 / +1.3906 | +1.6828 | [+0.7691, +2.7692] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +4.7518 / +3.2978 | +4.0248 | [+2.0349, +6.2892] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.3231 / +0.3252 | +0.3242 | [+0.0855, +0.6310] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.8567 / +0.6720 | +0.7643 | [+0.1845, +1.4740] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +1.4070 / +1.0983 | +1.2526 | [+0.1244, +2.5610] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0586 / +0.0584 | +0.0585 | [+0.0316, +0.0900] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.1203 / +0.1404 | +0.1303 | [+0.0394, +0.2215] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50F5 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0712 / -0.0463 | -0.0587 | [-0.0976, -0.0235] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0607 / -0.0249 | -0.0428 | [-0.0772, -0.0160] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1308 / +0.1090 | +0.1199 | [+0.0724, +0.1717] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7142 / +0.4508 | +0.5825 | [+0.2805, +0.9190] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.7932 / +1.8723 | +2.3327 | [+1.2684, +3.5612] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +6.2565 / +4.0515 | +5.1540 | [+2.8310, +7.6971] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.5091 / +0.2902 | +0.3997 | [+0.1473, +0.6846] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +1.6446 / +1.2728 | +1.4587 | [+0.7386, +2.3095] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +3.0745 / +2.1884 | +2.6314 | [+1.2119, +4.2492] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.2064 / +0.1050 | +0.1557 | [+0.0593, +0.2930] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.5715 / +0.5128 | +0.5421 | [+0.3347, +0.7909] | 0.5140 (0.5140) | WORSE |

## I10RLcF2 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.1068 / -0.0943 | -0.1005 | [-0.1461, -0.0603] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0763 / -0.0421 | -0.0592 | [-0.0966, -0.0288] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0903 / +0.0732 | +0.0818 | [+0.0405, +0.1282] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.5049 / +0.5064 | +0.5057 | [+0.2238, +0.8588] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +1.9328 / +1.5110 | +1.7219 | [+0.8328, +2.7442] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +4.9528 / +3.4902 | +4.2215 | [+2.1543, +6.5397] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.3397 / +0.3396 | +0.3396 | [+0.0953, +0.6490] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.7972 / +0.7279 | +0.7625 | [+0.2053, +1.4494] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +1.5787 / +1.2810 | +1.4298 | [+0.2036, +2.8342] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0646 / +0.0660 | +0.0653 | [+0.0361, +0.0990] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.1207 / +0.1737 | +0.1472 | [+0.0558, +0.2419] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50F5 minus I10RLcF5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0409 / +0.0427 | +0.0418 | [+0.0207, +0.0675] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0156 / +0.0171 | +0.0164 | [+0.0049, +0.0307] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0405 / +0.0374 | +0.0389 | [+0.0091, +0.0665] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2269 / -0.0411 | +0.0929 | [-0.1062, +0.3382] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.8182 / +0.4817 | +0.6499 | [+0.2533, +1.1843] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.5047 / +0.7537 | +1.1292 | [+0.1346, +2.2464] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.1860 / -0.0349 | +0.0755 | [-0.0960, +0.2893] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.7879 / +0.6008 | +0.6943 | [+0.3783, +1.1170] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +1.6675 / +1.0901 | +1.3788 | [+0.5827, +2.3416] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.1478 / +0.0466 | +0.0972 | [-0.0058, +0.2344] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.4512 / +0.3724 | +0.4118 | [+0.2228, +0.6220] | 0.5140 (0.5140) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10RLc (0,1) | 0.0383 [0.0190, 0.0628] | 0.0016 [0.0000, 0.0051] | 0.2181 [0.1459, 0.2897] | 0.2229 [-0.0327, 0.5472] | 2.0938 [1.2319, 3.0706] | 6.7474 [4.5816, 8.9276] | 0.7885 [0.5579, 1.0945] | 3.2710 [2.6177, 4.1001] | 8.0958 [6.4712, 9.8010] | 1.0876 [0.7460, 1.4643] | 2.1535 [1.4770, 2.9189] |
| I10RLcF2 (0,1) | 0.0294 [0.0135, 0.0501] | 0.0000 [0.0000, 0.0000] | 0.2220 [0.1462, 0.2970] | 0.2821 [0.0084, 0.6326] | 2.1702 [1.2928, 3.1485] | 7.0156 [4.7583, 9.3239] | 0.8125 [0.5670, 1.1268] | 3.2546 [2.6258, 4.0388] | 8.3076 [6.5786, 10.1990] | 1.1088 [0.7746, 1.4837] | 2.1890 [1.4984, 2.9680] |
| I10RLcF5 (0,1) | 0.0294 [0.0161, 0.0468] | 0.0000 [0.0000, 0.0000] | 0.2212 [0.1473, 0.2949] | 0.2660 [-0.0005, 0.6041] | 2.1311 [1.3035, 3.1056] | 6.8189 [4.6041, 9.0795] | 0.7970 [0.5589, 1.1110] | 3.2564 [2.6256, 4.0653] | 8.1304 [6.4624, 9.9422] | 1.1020 [0.7679, 1.4781] | 2.1722 [1.4940, 2.9385] |
| I10RQ50 (0,1) | 0.0801 [0.0480, 0.1174] | 0.0265 [0.0102, 0.0465] | 0.2593 [0.1862, 0.3304] | 0.2697 [0.0335, 0.5486] | 2.4645 [1.6073, 3.4225] | 7.6520 [5.6761, 9.6557] | 0.8241 [0.5966, 1.0699] | 3.7859 [3.0818, 4.6228] | 9.4150 [7.8273, 11.0990] | 1.1816 [0.8261, 1.5783] | 2.5241 [1.8524, 3.2957] |
| I10RQ50F5 (0,1) | 0.0712 [0.0449, 0.1013] | 0.0164 [0.0049, 0.0307] | 0.2601 [0.1836, 0.3338] | 0.3589 [0.0841, 0.6833] | 2.7810 [1.9098, 3.7616] | 7.9481 [5.9245, 10.0877] | 0.8726 [0.6109, 1.1655] | 3.9507 [3.2072, 4.8369] | 9.5092 [7.9092, 11.2367] | 1.1992 [0.8358, 1.6032] | 2.5840 [1.8978, 3.3631] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
