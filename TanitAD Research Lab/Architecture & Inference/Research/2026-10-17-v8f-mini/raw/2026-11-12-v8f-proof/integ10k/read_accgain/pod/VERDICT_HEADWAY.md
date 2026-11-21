# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: /workspace/v8proof/podread/data/join_eval141.npz. Alignment reference (ego speed): /workspace/v8proof/podread/data/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 14, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.03202846975088969, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0G50:I10R0G0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0C10:I10R0G50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0G0:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0G0X:I10R0G0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0Gm25:I10R0G0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0Gm25:I10R0Bk10", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0G0X:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0G0:I10R0C10", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0G50:I10R0C10", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0Gm25:I10R0C10", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0G0X:I10R0C10", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10R0G50 minus I10R0G0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0C10 minus I10R0G50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0G0 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0G0X minus I10R0G0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0Gm25 minus I10R0G0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0Gm25 minus I10R0Bk10 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0G0X minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0G0 minus I10R0C10 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0G50 minus I10R0C10 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0Gm25 minus I10R0C10 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0G0X minus I10R0C10 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10R0G50 minus I10R0G0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0374 / -0.0160 | -0.0267 | [-0.0512, -0.0002] | 0.0320 (0.0320) | INSIDE-SPREAD |
| C contact rate | -0.0296 / -0.0109 | -0.0202 | [-0.0421, +0.0014] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.1059 / +0.1667 | +0.1363 | [+0.0890, +0.1972] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +1.2980 / +1.4913 | +1.3947 | [+1.1661, +1.6475] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +4.7291 / +4.8896 | +4.8094 | [+3.7639, +5.8985] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.3905 / +0.6090 | +0.4997 | [+0.2297, +0.7640] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +1.4341 / +3.2710 | +2.3526 | [+0.9186, +3.9489] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.8822 / +1.3258 | +1.1040 | [+0.6821, +1.6268] | 0.5140 (0.5140) | WORSE |

## I10R0C10 minus I10R0G50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0907 / +0.0836 | +0.0872 | [+0.0535, +0.1235] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0389 / +0.0467 | +0.0428 | [+0.0242, +0.0626] | 0.0343 (0.0343) | WORSE |
| F falls behind by more than 10 m at the last scored step | +0.0981 / +0.0997 | +0.0989 | [+0.0481, +0.1513] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.6870 / +0.7134 | +0.7002 | [+0.4042, +0.9862] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.9451 / -0.0541 | +0.4455 | [-1.7749, +2.5276] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.8243 / +0.6849 | +0.7546 | [+0.5135, +0.9751] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +3.4960 / +4.9983 | +4.2472 | [+2.4613, +6.0793] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.2271 / -0.1950 | +0.0160 | [-0.4968, +0.4410] | 0.5140 (0.5140) | OPEN |

## I10R0G0 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0463 / -0.0267 | -0.0365 | [-0.0772, +0.0035] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0202 / +0.0062 | -0.0070 | [-0.0353, +0.0176] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0078 / -0.0545 | -0.0234 | [-0.0776, +0.0275] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.2173 / -0.2606 | -0.0217 | [-0.4281, +0.3532] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | -0.0279 / -2.0712 | -1.0496 | [-2.8412, +0.6846] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.1513 / -0.2152 | -0.1833 | [-0.3682, +0.0005] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -0.3507 / -1.4838 | -0.9172 | [-2.0544, +0.0437] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.0352 / -0.1343 | -0.0847 | [-0.3554, +0.1549] | 0.5140 (0.5140) | OPEN |

## I10R0G0X minus I10R0G0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0125 / +0.0018 | +0.0071 | [-0.0022, +0.0184] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0047 / +0.0062 | +0.0055 | [-0.0001, +0.0116] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0202 / -0.0156 | -0.0179 | [-0.0313, -0.0044] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.0403 / +0.0324 | +0.0364 | [-0.0152, +0.0947] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | -0.6058 / -0.6637 | -0.6347 | [-0.9543, -0.2917] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.0152 / -0.0038 | +0.0057 | [-0.0379, +0.0465] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -0.0932 / -0.3146 | -0.2039 | [-0.5049, +0.0793] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.0423 / -0.1433 | -0.0928 | [-0.1857, -0.0244] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10R0Gm25 minus I10R0G0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0516 / +0.0463 | +0.0489 | [+0.0258, +0.0750] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0265 / +0.0280 | +0.0273 | [+0.0156, +0.0398] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0514 / -0.0623 | -0.0569 | [-0.0831, -0.0339] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.8305 / -0.9764 | -0.9034 | [-1.0649, -0.7493] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -2.5763 / -3.1260 | -2.8512 | [-3.4197, -2.3105] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.0754 / -0.1317 | -0.0282 | [-0.2260, +0.1543] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +0.0786 / -0.1959 | -0.0587 | [-0.7086, +0.5184] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.2463 / -0.5365 | -0.3914 | [-0.7160, -0.1706] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10R0Gm25 minus I10R0Bk10

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0445 / +0.0302 | +0.0374 | [+0.0149, +0.0636] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0171 / +0.0140 | +0.0156 | [+0.0076, +0.0251] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0171 / -0.0218 | -0.0195 | [-0.0450, +0.0040] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.4535 / -0.4466 | -0.4501 | [-0.5719, -0.3376] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -2.2728 / -2.3944 | -2.3336 | [-3.0971, -1.6423] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.0356 / -0.0521 | -0.0082 | [-0.1324, +0.1076] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +0.3466 / +0.1244 | +0.2355 | [-0.3567, +0.7907] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.1844 / -0.2624 | -0.2234 | [-0.3461, -0.1242] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10R0G0X minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0338 / -0.0249 | -0.0294 | [-0.0668, +0.0064] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0156 / +0.0125 | -0.0016 | [-0.0317, +0.0253] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0125 / -0.0701 | -0.0413 | [-0.0925, +0.0052] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.2576 / -0.2282 | +0.0147 | [-0.4167, +0.3976] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | -0.6337 / -2.7349 | -1.6843 | [-3.4711, +0.0238] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.1361 / -0.2190 | -0.1776 | [-0.3726, +0.0097] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -0.4439 / -1.7984 | -1.1211 | [-2.3220, -0.0751] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.0775 / -0.2776 | -0.1775 | [-0.4588, +0.0463] | 0.5140 (0.5140) | OPEN |

## I10R0G0 minus I10R0C10

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0534 / -0.0676 | -0.0605 | [-0.1021, -0.0193] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0093 / -0.0358 | -0.0226 | [-0.0532, +0.0032] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.2040 / -0.2664 | -0.2352 | [-0.3144, -0.1622] | 0.0903 (0.0903) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -1.9850 / -2.2047 | -2.0948 | [-2.5588, -1.6795] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -5.6742 / -4.8355 | -5.2548 | [-7.9414, -2.3920] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -1.2148 / -1.2939 | -1.2544 | [-1.6182, -0.9159] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -4.9301 / -8.2694 | -6.5997 | [-9.5582, -3.9961] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -1.1093 / -1.1308 | -1.1200 | [-1.5130, -0.7394] | 0.5140 (0.5140) | BETTER |

## I10R0G50 minus I10R0C10

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0907 / -0.0836 | -0.0872 | [-0.1235, -0.0535] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0389 / -0.0467 | -0.0428 | [-0.0626, -0.0242] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | -0.0981 / -0.0997 | -0.0989 | [-0.1513, -0.0481] | 0.0903 (0.0903) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -0.6870 / -0.7134 | -0.7002 | [-0.9862, -0.4042] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.9451 / +0.0541 | -0.4455 | [-2.5276, +1.7749] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -0.8243 / -0.6849 | -0.7546 | [-0.9751, -0.5135] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -3.4960 / -4.9983 | -4.2472 | [-6.0793, -2.4613] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.2271 / +0.1950 | -0.0160 | [-0.4410, +0.4968] | 0.5140 (0.5140) | OPEN |

## I10R0Gm25 minus I10R0C10

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0018 / -0.0214 | -0.0116 | [-0.0596, +0.0349] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0171 / -0.0078 | +0.0047 | [-0.0241, +0.0315] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.2555 / -0.3287 | -0.2921 | [-0.3734, -0.2153] | 0.0903 (0.0903) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -2.8155 / -3.1811 | -2.9983 | [-3.5192, -2.5185] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -8.2505 / -7.9615 | -8.1060 | [-10.7638, -5.4532] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -1.1395 / -1.4256 | -1.2825 | [-1.7201, -0.8584] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -4.8515 / -8.4653 | -6.6584 | [-9.7035, -3.9286] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -1.3556 / -1.6673 | -1.5115 | [-1.9207, -1.1599] | 0.5140 (0.5140) | BETTER |

## I10R0G0X minus I10R0C10

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0409 / -0.0658 | -0.0534 | [-0.0958, -0.0097] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0047 / -0.0296 | -0.0171 | [-0.0469, +0.0085] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.2243 / -0.2819 | -0.2531 | [-0.3275, -0.1838] | 0.0903 (0.0903) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -1.9447 / -2.1723 | -2.0585 | [-2.5166, -1.6385] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -6.2800 / -5.4992 | -5.8896 | [-8.5189, -3.1968] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | -1.1996 / -1.2977 | -1.2486 | [-1.6009, -0.9171] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -5.0232 / -8.5839 | -6.8036 | [-9.8490, -4.1267] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -1.1516 / -1.2740 | -1.2128 | [-1.6280, -0.8594] | 0.5140 (0.5140) | BETTER |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10R0Bk10 (0,1) | 0.1050 [0.0650, 0.1487] | 0.0639 [0.0352, 0.0961] | 0.0794 [0.0495, 0.1130] | -0.2236 [-0.2757, -0.1682] | -0.0268 [-0.4978, 0.4853] | 1.2270 [-0.0581, 2.5934] | 0.4729 [0.4299, 0.5219] | 2.2888 [2.0418, 2.5813] | 5.6664 [4.9525, 6.4671] | 1.0435 [0.7059, 1.4212] | 1.7891 [1.1950, 2.5049] |
| I10R0C10 (0,1) | 0.1539 [0.1079, 0.2022] | 0.0748 [0.0464, 0.1078] | 0.3520 [0.2687, 0.4350] | -0.2236 [-0.2757, -0.1682] | 2.5214 [1.8035, 3.2494] | 6.9994 [4.6887, 9.4838] | 0.4729 [0.4299, 0.5219] | 3.5631 [3.1336, 4.0115] | 12.5603 [9.7445, 15.6164] | 1.0435 [0.7059, 1.4212] | 3.0771 [2.3359, 3.9956] |
| I10R0G0 (0,1) | 0.0934 [0.0613, 0.1296] | 0.0522 [0.0256, 0.0824] | 0.1168 [0.0796, 0.1610] | -0.2236 [-0.2757, -0.1682] | 0.4266 [-0.1114, 0.9818] | 1.7445 [0.1261, 3.3924] | 0.4729 [0.4299, 0.5219] | 2.3088 [2.0178, 2.6523] | 5.9606 [5.1370, 6.9084] | 1.0435 [0.7059, 1.4212] | 1.9571 [1.3168, 2.7608] |
| I10R0G0X (0,1) | 0.1005 [0.0646, 0.1408] | 0.0576 [0.0295, 0.0883] | 0.0989 [0.0615, 0.1416] | -0.2236 [-0.2757, -0.1682] | 0.4629 [-0.0807, 1.0171] | 1.1098 [-0.4325, 2.6695] | 0.4729 [0.4299, 0.5219] | 2.3145 [2.0317, 2.6434] | 5.7567 [5.0133, 6.5885] | 1.0435 [0.7059, 1.4212] | 1.8643 [1.2610, 2.5995] |
| I10R0G50 (0,1) | 0.0667 [0.0393, 0.0968] | 0.0319 [0.0141, 0.0523] | 0.2531 [0.1782, 0.3352] | -0.2236 [-0.2757, -0.1682] | 1.8212 [1.1494, 2.5065] | 6.5539 [4.9229, 8.3082] | 0.4729 [0.4299, 0.5219] | 2.8085 [2.3644, 3.2883] | 8.3131 [6.5492, 10.1462] | 1.0435 [0.7059, 1.4212] | 3.0611 [2.1444, 4.1099] |
| I10R0Gm25 (0,1) | 0.1423 [0.0955, 0.1934] | 0.0794 [0.0462, 0.1156] | 0.0600 [0.0348, 0.0901] | -0.2236 [-0.2757, -0.1682] | -0.4769 [-0.9232, -0.0120] | -1.1066 [-2.4488, 0.2769] | 0.4729 [0.4299, 0.5219] | 2.2806 [2.0211, 2.5767] | 5.9019 [5.0965, 6.8152] | 1.0435 [0.7059, 1.4212] | 1.5657 [1.0107, 2.2504] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
