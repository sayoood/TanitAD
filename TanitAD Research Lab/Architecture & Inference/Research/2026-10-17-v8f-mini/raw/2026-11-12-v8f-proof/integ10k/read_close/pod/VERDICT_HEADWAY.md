# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: /workspace/v8proof/podread/data/join_eval141.npz. Alignment reference (ego speed): /workspace/v8proof/podread/data/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 10, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.03202846975088969, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLc:I10RLb", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLc:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50c:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50c:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLb:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50:I10R0", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10RLc minus I10RLb over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLc minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50c minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50c minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLb minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10RLc minus I10RLb

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0427 / -0.0214 | -0.0320 | [-0.0573, -0.0107] | 0.0320 (0.0320) | INSIDE-SPREAD |
| C contact rate | -0.0327 / -0.0125 | -0.0226 | [-0.0476, -0.0043] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0140 / -0.0125 | -0.0132 | [-0.0305, +0.0035] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0014 / -0.0054 | -0.0034 | [-0.0476, +0.0354] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.2387 / -0.3628 | -0.3008 | [-0.6741, +0.1112] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +0.1234 / -0.4339 | -0.1553 | [-0.8713, +0.7382] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | -0.0298 / -0.0289 | -0.0293 | [-0.0632, -0.0018] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -0.4862 / -0.4685 | -0.4773 | [-0.8124, -0.1798] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -0.3311 / -0.7368 | -0.5340 | [-1.1566, +0.2690] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | -0.0222 / -0.0219 | -0.0221 | [-0.0734, +0.0212] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.1679 / -0.2218 | -0.1948 | [-0.3348, -0.0807] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RLc minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0996 / -0.0836 | -0.0916 | [-0.1372, -0.0509] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0763 / -0.0389 | -0.0576 | [-0.0966, -0.0270] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0888 / +0.0670 | +0.0779 | [+0.0369, +0.1246] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4463 / +0.4467 | +0.4465 | [+0.1858, +0.7734] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +1.9097 / +1.3814 | +1.6455 | [+0.7390, +2.7288] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +4.7373 / +3.1694 | +3.9533 | [+2.0177, +6.2407] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.3151 / +0.3162 | +0.3156 | [+0.0869, +0.6111] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.8592 / +0.6987 | +0.7789 | [+0.1714, +1.5388] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +1.4548 / +0.9811 | +1.2180 | [+0.1267, +2.5314] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0426 / +0.0456 | +0.0441 | [+0.0199, +0.0728] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.1035 / +0.1198 | +0.1116 | [+0.0165, +0.2114] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50c minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0107 / -0.0142 | -0.0125 | [-0.0307, +0.0021] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0280 / -0.0047 | -0.0164 | [-0.0416, -0.0016] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0156 / -0.0358 | -0.0257 | [-0.0496, -0.0050] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0314 / -0.0501 | -0.0407 | [-0.0927, +0.0095] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.0187 / -0.4839 | -0.2513 | [-0.7079, +0.2955] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +0.0473 / -0.1601 | -0.0564 | [-0.8608, +0.8019] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | -0.0455 / -0.0615 | -0.0535 | [-0.0973, -0.0104] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | +0.0071 / -0.5570 | -0.2749 | [-0.7351, +0.3293] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -0.1930 / -0.5807 | -0.3869 | [-1.1474, +0.4720] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | -0.0754 / -0.0556 | -0.0655 | [-0.1402, -0.0050] | 0.0330 (0.0330) | BETTER |
| time-gap error over the scored steps (s) | -0.1543 / -0.2754 | -0.2149 | [-0.4128, -0.0536] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RQ50c minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0694 / -0.0552 | -0.0623 | [-0.0981, -0.0280] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0732 / -0.0249 | -0.0491 | [-0.0938, -0.0167] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1168 / +0.0701 | +0.0935 | [+0.0437, +0.1497] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.5706 / +0.3344 | +0.4525 | [+0.1934, +0.7428] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.3945 / +1.1355 | +1.7650 | [+0.6339, +3.0976] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +6.1195 / +3.4836 | +4.8015 | [+2.2309, +7.6909] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.3951 / +0.2004 | +0.2978 | [+0.0838, +0.5458] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +1.4392 / +0.5986 | +1.0189 | [+0.2523, +1.9932] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +2.8617 / +1.4390 | +2.1503 | [+0.6142, +4.0917] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.1030 / +0.0423 | +0.0726 | [+0.0190, +0.1596] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.3755 / +0.1593 | +0.2674 | [+0.0994, +0.4729] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RLb minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0569 / -0.0623 | -0.0596 | [-0.0933, -0.0288] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0436 / -0.0265 | -0.0350 | [-0.0578, -0.0171] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1028 / +0.0794 | +0.0911 | [+0.0455, +0.1418] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4477 / +0.4520 | +0.4499 | [+0.1834, +0.7692] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.1484 / +1.7442 | +1.9463 | [+0.8825, +3.2618] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +4.6139 / +3.6032 | +4.1086 | [+2.1176, +6.4346] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.3449 / +0.3451 | +0.3450 | [+0.1170, +0.6319] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +1.3454 / +1.1672 | +1.2563 | [+0.4256, +2.2917] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +1.7860 / +1.7179 | +1.7519 | [+0.5367, +3.2583] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.0649 / +0.0675 | +0.0662 | [+0.0262, +0.1131] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.2714 / +0.3416 | +0.3065 | [+0.1541, +0.4923] | 0.5140 (0.5140) | INSIDE-SPREAD |

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
| I10RLb (0,1) | 0.0703 [0.0382, 0.1113] | 0.0241 [0.0065, 0.0484] | 0.2313 [0.1606, 0.3039] | 0.2263 [-0.0383, 0.5548] | 2.3946 [1.2957, 3.6479] | 6.9027 [4.4318, 9.5247] | 0.8178 [0.5897, 1.1089] | 3.7483 [2.8924, 4.8297] | 8.6297 [6.8435, 10.6964] | 1.1097 [0.7592, 1.4936] | 2.3483 [1.6463, 3.1694] |
| I10RLc (0,1) | 0.0383 [0.0190, 0.0628] | 0.0016 [0.0000, 0.0051] | 0.2181 [0.1459, 0.2897] | 0.2229 [-0.0327, 0.5472] | 2.0938 [1.2319, 3.0706] | 6.7474 [4.5816, 8.9276] | 0.7885 [0.5579, 1.0945] | 3.2710 [2.6177, 4.1001] | 8.0958 [6.4712, 9.8010] | 1.0876 [0.7460, 1.4643] | 2.1535 [1.4770, 2.9189] |
| I10RQ50 (0,1) | 0.0801 [0.0480, 0.1174] | 0.0265 [0.0102, 0.0465] | 0.2593 [0.1862, 0.3304] | 0.2697 [0.0335, 0.5486] | 2.4645 [1.6073, 3.4225] | 7.6520 [5.6761, 9.6557] | 0.8241 [0.5966, 1.0699] | 3.7859 [3.0818, 4.6228] | 9.4150 [7.8273, 11.0990] | 1.1816 [0.8261, 1.5783] | 2.5241 [1.8524, 3.2957] |
| I10RQ50c (0,1) | 0.0676 [0.0425, 0.0969] | 0.0101 [0.0013, 0.0254] | 0.2336 [0.1715, 0.2965] | 0.2290 [-0.0037, 0.4971] | 2.2132 [1.4459, 3.1186] | 7.5956 [5.6614, 9.6262] | 0.7707 [0.5511, 1.0157] | 3.5110 [2.7391, 4.4854] | 9.0281 [7.3886, 10.8566] | 1.1161 [0.7670, 1.4997] | 2.3093 [1.6943, 3.0099] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
