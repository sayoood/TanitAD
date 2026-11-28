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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50d:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50x:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50d:I10RQ50c", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50d:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50d:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50:I10R0", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10RQ50d minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50x minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50d minus I10RQ50c over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50d minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50d minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10RQ50d minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0000 | +0.0000 | [-0.0076, +0.0076] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0218 / -0.0031 | -0.0125 | [-0.0324, -0.0014] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0171 / +0.0000 | +0.0086 | [-0.0063, +0.0288] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0677 / +0.0584 | +0.0631 | [+0.0242, +0.1103] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +0.3327 / +0.1711 | +0.2519 | [+0.0504, +0.5218] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.1360 / +0.6486 | +0.8923 | [+0.3756, +1.5297] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0618 / +0.0477 | +0.0548 | [+0.0151, +0.1021] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.2667 / +0.1482 | +0.2075 | [+0.0115, +0.4558] | 0.2417 (0.2417) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.8490 / +0.3787 | +0.6138 | [+0.1506, +1.2252] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.1059 / +0.0215 | +0.0637 | [+0.0023, +0.1907] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.2345 / +0.0388 | +0.1366 | [-0.0203, +0.4078] | 0.5140 (0.5140) | OPEN |

## I10RQ50x minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0178 / -0.0196 | -0.0009 | [-0.0377, +0.0297] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0062 / -0.0016 | +0.0023 | [-0.0039, +0.0090] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0125 / +0.0125 | +0.0000 | [-0.0328, +0.0338] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.2197 / +0.2135 | -0.0031 | [-0.3110, +0.3182] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.4003 / +0.3901 | -0.0051 | [-0.7739, +0.8112] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | -0.9981 / +1.0316 | +0.0167 | [-1.6980, +2.0166] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | -0.1807 / +0.1814 | +0.0003 | [-0.2614, +0.2750] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.1609 / +0.2526 | +0.0458 | [-0.4065, +0.5533] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -0.7040 / +0.6824 | -0.0108 | [-1.3010, +1.4089] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | -0.0772 / +0.0786 | +0.0007 | [-0.1405, +0.1467] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.1320 / +0.2989 | +0.0835 | [-0.2532, +0.4588] | 0.5140 (0.5140) | OPEN |

## I10RQ50d minus I10RQ50c

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0107 / +0.0142 | +0.0125 | [-0.0026, +0.0287] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0062 / +0.0016 | +0.0039 | [-0.0022, +0.0110] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0327 / +0.0358 | +0.0343 | [+0.0158, +0.0541] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0991 / +0.1085 | +0.1038 | [+0.0334, +0.1884] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +0.3514 / +0.6550 | +0.5032 | [+0.1333, +0.9257] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.0887 / +0.8087 | +0.9487 | [+0.1823, +1.7782] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.1073 / +0.1092 | +0.1083 | [+0.0450, +0.1883] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.2596 / +0.7052 | +0.4824 | [+0.0202, +0.9322] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +1.0420 / +0.9594 | +1.0007 | [+0.2476, +1.8134] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.1812 / +0.0771 | +0.1292 | [+0.0342, +0.2775] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.3888 / +0.3142 | +0.3515 | [+0.1518, +0.6236] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50d minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0587 / -0.0409 | -0.0498 | [-0.0855, -0.0188] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0670 / -0.0234 | -0.0452 | [-0.0858, -0.0152] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1495 / +0.1059 | +0.1277 | [+0.0738, +0.1838] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6698 / +0.4429 | +0.5564 | [+0.2869, +0.8640] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.7458 / +1.7904 | +2.2681 | [+1.1900, +3.5319] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +7.2082 / +4.2923 | +5.7502 | [+3.0272, +8.7816] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.5025 / +0.3096 | +0.4060 | [+0.1829, +0.6658] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +1.6988 / +1.3039 | +1.5014 | [+0.7603, +2.3665] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +3.9037 / +2.3984 | +3.1510 | [+1.4354, +5.1916] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.2842 / +0.1194 | +0.2018 | [+0.0774, +0.4305] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.7643 / +0.4735 | +0.6189 | [+0.3208, +1.0266] | 0.5140 (0.5140) | WORSE |

## I10RQ50d minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0409 / +0.0427 | +0.0418 | [+0.0210, +0.0685] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0093 / +0.0156 | +0.0125 | [+0.0033, +0.0258] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0607 / +0.0389 | +0.0498 | [+0.0201, +0.0819] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2235 / -0.0037 | +0.1099 | [-0.0885, +0.3560] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.8362 / +0.4090 | +0.6226 | [+0.1668, +1.3122] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.4709 / +1.1229 | +1.7969 | [+0.5970, +3.1665] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.1874 / -0.0066 | +0.0904 | [-0.0879, +0.3071] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.8397 / +0.6052 | +0.7224 | [+0.3362, +1.3032] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +2.4488 / +1.4173 | +1.9331 | [+0.9514, +3.1007] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.2416 / +0.0737 | +0.1576 | [+0.0267, +0.3880] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.6609 / +0.3537 | +0.5073 | [+0.2150, +0.9002] | 0.5140 (0.5140) | INSIDE-SPREAD |

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

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10RLc (0,1) | 0.0383 [0.0190, 0.0628] | 0.0016 [0.0000, 0.0051] | 0.2181 [0.1459, 0.2897] | 0.2229 [-0.0327, 0.5472] | 2.0938 [1.2319, 3.0706] | 6.7474 [4.5816, 8.9276] | 0.7885 [0.5579, 1.0945] | 3.2710 [2.6177, 4.1001] | 8.0958 [6.4712, 9.8010] | 1.0876 [0.7460, 1.4643] | 2.1535 [1.4770, 2.9189] |
| I10RQ50 (0,1) | 0.0801 [0.0480, 0.1174] | 0.0265 [0.0102, 0.0465] | 0.2593 [0.1862, 0.3304] | 0.2697 [0.0335, 0.5486] | 2.4645 [1.6073, 3.4225] | 7.6520 [5.6761, 9.6557] | 0.8241 [0.5966, 1.0699] | 3.7859 [3.0818, 4.6228] | 9.4150 [7.8273, 11.0990] | 1.1816 [0.8261, 1.5783] | 2.5241 [1.8524, 3.2957] |
| I10RQ50c (0,1) | 0.0676 [0.0425, 0.0969] | 0.0101 [0.0013, 0.0254] | 0.2336 [0.1715, 0.2965] | 0.2290 [-0.0037, 0.4971] | 2.2132 [1.4459, 3.1186] | 7.5956 [5.6614, 9.6262] | 0.7707 [0.5511, 1.0157] | 3.5110 [2.7391, 4.4854] | 9.0281 [7.3886, 10.8566] | 1.1161 [0.7670, 1.4997] | 2.3093 [1.6943, 3.0099] |
| I10RQ50d (0,1) | 0.0801 [0.0483, 0.1148] | 0.0140 [0.0039, 0.0294] | 0.2679 [0.2018, 0.3317] | 0.3328 [0.0823, 0.6228] | 2.7164 [1.8335, 3.6722] | 8.5443 [6.5618, 10.6666] | 0.8789 [0.6414, 1.1500] | 3.9934 [3.2376, 4.8706] | 10.0288 [8.3402, 11.9062] | 1.2453 [0.8570, 1.6832] | 2.6608 [1.9826, 3.3806] |
| I10RQ50x (0,1) | 0.0792 [0.0397, 0.1277] | 0.0288 [0.0100, 0.0537] | 0.2593 [0.1784, 0.3442] | 0.2666 [-0.0153, 0.5867] | 2.4594 [1.3100, 3.6372] | 7.6688 [4.9586, 10.6202] | 0.8245 [0.6086, 1.0819] | 3.8317 [3.0738, 4.7075] | 9.4042 [7.4884, 11.6411] | 1.1823 [0.8066, 1.5937] | 2.6076 [1.7938, 3.5651] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
