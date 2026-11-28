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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0A5:I10R0M5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcA5:I10RLcM5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50A5:I10RQ50M5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0A5:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcA5:I10RLc", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50A5:I10RQ50", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0A10:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RLcA5:I10R0A5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RQ50A5:I10R0A5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0A5L:I10R0A5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0A5L:I10R0M5", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10R0A5 minus I10R0M5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcA5 minus I10RLcM5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50A5 minus I10RQ50M5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0A5 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcA5 minus I10RLc over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50A5 minus I10RQ50 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0A10 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RLcA5 minus I10R0A5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RQ50A5 minus I10R0A5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0A5L minus I10R0A5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0A5L minus I10R0M5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10R0A5 minus I10R0M5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.2438 / -0.3238 | -0.2838 | [-0.3694, -0.2075] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.3879 / -0.4766 | -0.4322 | [-0.5181, -0.3505] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | -0.0187 / -0.0062 | -0.0125 | [-0.0363, +0.0049] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2409 / +0.3625 | +0.3017 | [+0.2046, +0.4250] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +6.8985 / +12.1280 | +9.5132 | [+5.6776, +13.4961] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +21.8510 / +42.9241 | +32.3875 | [+18.4211, +45.6573] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | -0.2516 / -0.3802 | -0.3159 | [-0.4316, -0.2209] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -9.8692 / -14.8068 | -12.3380 | [-15.9498, -8.6714] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -33.0311 / -43.5091 | -38.2701 | [-46.1673, -30.6101] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.2097 / -0.2534 | -0.2315 | [-0.3071, -0.1595] | 0.0330 (0.0330) | BETTER |
| time-gap error over the scored steps (s) | -0.5130 / -0.6258 | -0.5694 | [-0.7184, -0.4385] | 0.5140 (0.5140) | BETTER |

## I10RLcA5 minus I10RLcM5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.3238 / -0.3665 | -0.3452 | [-0.3992, -0.2881] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0171 / -0.0202 | -0.0187 | [-0.0273, -0.0108] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0343 / -0.0483 | -0.0413 | [-0.0706, -0.0150] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2507 / +0.3000 | +0.2753 | [+0.1948, +0.3597] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +0.4980 / +1.7085 | +1.1033 | [+0.1047, +2.1882] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | -0.7570 / -0.7921 | -0.7745 | [-2.3228, +0.7632] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | -0.1499 / -0.2331 | -0.1915 | [-0.2927, -0.0940] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -1.7435 / -2.1843 | -1.9639 | [-2.6286, -1.4431] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -2.2538 / -2.6857 | -2.4697 | [-3.7059, -1.2813] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.0877 / -0.2176 | -0.1526 | [-0.2688, +0.0038] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.0875 / -0.1466 | -0.1171 | [-0.2592, +0.0183] | 0.5140 (0.5140) | OPEN |

## I10RQ50A5 minus I10RQ50M5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.3149 / -0.3505 | -0.3327 | [-0.3940, -0.2757] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0623 / -0.0732 | -0.0678 | [-0.0996, -0.0396] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0047 / -0.0062 | -0.0008 | [-0.0272, +0.0210] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1547 / +0.2332 | +0.1940 | [+0.1244, +0.2805] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +0.1800 / +1.9614 | +1.0707 | [-0.2067, +2.5385] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +2.1299 / +3.1791 | +2.6545 | [+0.4621, +5.1950] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.1852 / -0.3157 | -0.2504 | [-0.3566, -0.1527] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -2.1158 / -2.0777 | -2.0968 | [-2.7756, -1.4440] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -5.8118 / -4.3410 | -5.0764 | [-7.6688, -2.7925] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.1157 / -0.3118 | -0.0981 | [-0.3877, +0.3286] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.1400 / +0.0230 | +0.0815 | [-0.1315, +0.2765] | 0.5140 (0.5140) | OPEN |

## I10R0A5 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.4609 / +0.4591 | +0.4600 | [+0.3830, +0.5310] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.1854 / +0.2134 | +0.1994 | [+0.1341, +0.2632] | 0.0343 (0.0343) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0436 / -0.1308 | -0.0872 | [-0.1654, -0.0192] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.5825 / -0.5743 | -0.5784 | [-0.6615, -0.4897] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -6.3071 / -7.1041 | -6.7056 | [-7.6083, -5.7550] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -17.6000 / -19.4959 | -18.5479 | [-21.5792, -15.3005] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.5811 / +0.5772 | +0.5791 | [+0.5204, +0.6362] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +5.0341 / +4.3209 | +4.6775 | [+3.9374, +5.4128] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +14.0331 / +10.6275 | +12.3303 | [+9.4206, +15.1958] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.5924 / +0.5583 | +0.5753 | [+0.1039, +1.1797] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.3321 / -0.1427 | +0.0947 | [-0.4560, +0.5684] | 0.5140 (0.5140) | OPEN |

## I10RLcA5 minus I10RLc

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0996 / +0.1068 | +0.1032 | [+0.0690, +0.1425] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0000 / -0.0031 | -0.0016 | [-0.0051, +0.0000] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0125 / -0.0794 | -0.0335 | [-0.1131, +0.0399] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.2383 / -0.2777 | -0.2580 | [-0.4277, -0.0604] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -2.6227 / -3.5520 | -3.0873 | [-4.2586, -1.9697] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -3.0236 / -7.0595 | -5.0416 | [-8.7481, -1.4409] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.6864 / +0.6384 | +0.6624 | [+0.5407, +0.8026] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +3.5323 / +2.8318 | +3.1820 | [+2.4146, +3.9594] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +6.5271 / +3.8522 | +5.1896 | [+2.7092, +7.5888] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.6095 / +0.4699 | +0.5397 | [+0.0570, +1.1542] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.2842 / -0.2768 | +0.0037 | [-0.5763, +0.5040] | 0.5140 (0.5140) | OPEN |

## I10RQ50A5 minus I10RQ50

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.1423 / +0.1833 | +0.1628 | [+0.1103, +0.2181] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0140 / +0.0312 | +0.0226 | [+0.0014, +0.0442] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0576 / -0.0826 | -0.0125 | [-0.1116, +0.0809] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0383 / -0.3095 | -0.1739 | [-0.4237, +0.0937] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -1.0327 / -3.5168 | -2.2747 | [-4.0876, -0.2961] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | +0.3009 / -5.4802 | -2.5896 | [-6.8022, +2.0741] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.8185 / +0.6520 | +0.7352 | [+0.5823, +0.9089] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +4.2765 / +2.5624 | +3.4195 | [+2.1406, +4.8046] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +8.1340 / +3.0630 | +5.5985 | [+2.0287, +9.5221] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +1.2533 / +0.6178 | +0.9355 | [+0.2506, +1.7189] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.3415 / -0.2317 | +0.0549 | [-0.5526, +0.5826] | 0.5140 (0.5140) | OPEN |

## I10R0A10 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0391 / -0.0445 | -0.0418 | [-0.0657, -0.0218] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0218 / -0.0109 | -0.0164 | [-0.0301, -0.0057] | 0.0343 (0.0343) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0218 / +0.0374 | +0.0296 | [+0.0144, +0.0474] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0444 / +0.0600 | +0.0522 | [+0.0333, +0.0784] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +0.2515 / +0.2289 | +0.2402 | [+0.1752, +0.3138] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.9753 / +0.9869 | +0.9811 | [+0.5663, +1.4398] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0161 / -0.0189 | -0.0175 | [-0.0298, -0.0087] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | +0.0365 / +0.0507 | +0.0436 | [-0.0268, +0.1167] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | +0.1901 / +0.3769 | +0.2835 | [+0.0254, +0.5417] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0056 / +0.0081 | +0.0068 | [+0.0035, +0.0117] | 0.0330 (0.0330) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0952 / +0.4731 | +0.2842 | [+0.0113, +0.6116] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10RLcA5 minus I10R0A5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.4609 / -0.4359 | -0.4484 | [-0.5085, -0.3830] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.2617 / -0.2555 | -0.2586 | [-0.3252, -0.1902] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1449 / +0.1184 | +0.1316 | [+0.0699, +0.1935] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7905 / +0.7433 | +0.7669 | [+0.3567, +1.2527] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +5.5941 / +4.9335 | +5.2638 | [+3.5814, +7.0327] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +19.3137 / +15.6058 | +17.4597 | [+12.7305, +22.0283] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.4204 / +0.3774 | +0.3989 | [+0.0502, +0.8202] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | -0.6427 / -0.7904 | -0.7165 | [-1.8289, +0.5325] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -6.0512 / -5.7942 | -5.9227 | [-8.8413, -2.9291] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.0598 / -0.0428 | +0.0085 | [-0.1165, +0.1663] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.0556 / -0.0142 | +0.0207 | [-0.1087, +0.1564] | 0.5140 (0.5140) | OPEN |

## I10RQ50A5 minus I10R0A5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.3772 / -0.3167 | -0.3470 | [-0.4260, -0.2719] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.2165 / -0.2025 | -0.2095 | [-0.2696, -0.1500] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.2336 / +0.1542 | +0.1939 | [+0.1127, +0.2712] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.1462 / +0.6493 | +0.8977 | [+0.4079, +1.4407] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +7.6876 / +5.2067 | +6.4471 | [+4.2230, +8.8635] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +23.9731 / +17.6594 | +20.8162 | [+15.3893, +26.5726] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.6780 / +0.3367 | +0.5074 | [+0.1419, +0.9294] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.6745 / -0.6028 | +0.0359 | [-1.3412, +1.6948] | 0.2417 (0.2417) | OPEN |
| gap error 6.0 s (m) | -2.8444 / -5.5448 | -4.1946 | [-7.6655, -0.2990] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | +0.8393 / +0.1574 | +0.4984 | [+0.0629, +1.1463] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.5393 / +0.3457 | +0.4425 | [+0.2034, +0.7233] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10R0A5L minus I10R0A5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.1922 / -0.2171 | -0.2046 | [-0.2537, -0.1554] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.0997 / -0.0966 | -0.0981 | [-0.1369, -0.0626] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | -0.0109 / -0.0249 | -0.0179 | [-0.0351, -0.0017] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.3409 / +0.3551 | +0.3480 | [+0.2921, +0.4063] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +2.3132 / +2.2412 | +2.2772 | [+1.9851, +2.5879] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +6.1148 / +4.7301 | +5.4225 | [+4.2172, +6.7325] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | -0.2841 / -0.2924 | -0.2883 | [-0.3367, -0.2411] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -2.3575 / -2.0733 | -2.2154 | [-2.5688, -1.8933] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -6.4690 / -5.6846 | -6.0768 | [-6.9688, -5.2052] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.2786 / -0.2794 | -0.2790 | [-0.3986, -0.1824] | 0.0330 (0.0330) | BETTER |
| time-gap error over the scored steps (s) | -0.2101 / -0.2940 | -0.2521 | [-0.4171, -0.1271] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10R0A5L minus I10R0M5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.4359 / -0.5409 | -0.4884 | [-0.5919, -0.3920] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.4875 / -0.5732 | -0.5304 | [-0.6093, -0.4503] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | -0.0296 / -0.0312 | -0.0304 | [-0.0521, -0.0101] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.5819 / +0.7176 | +0.6497 | [+0.5114, +0.8159] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +9.2118 / +14.3692 | +11.7905 | [+7.8537, +15.8270] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +27.9658 / +47.6542 | +37.8100 | [+24.3219, +50.8354] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | -0.5357 / -0.6727 | -0.6042 | [-0.7436, -0.4791] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -12.2267 / -16.8801 | -14.5534 | [-18.0739, -10.9513] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -39.5000 / -49.1936 | -44.3468 | [-51.9319, -36.8738] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.4883 / -0.5328 | -0.5105 | [-0.6794, -0.3592] | 0.0330 (0.0330) | BETTER |
| time-gap error over the scored steps (s) | -0.7231 / -0.9198 | -0.8215 | [-1.1225, -0.5940] | 0.5140 (0.5140) | BETTER |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10R0A10 (0,1) | 0.0881 [0.0464, 0.1349] | 0.0428 [0.0197, 0.0729] | 0.1698 [0.0878, 0.2563] | -0.1713 [-0.2304, -0.1092] | 0.6885 [-0.0971, 1.4965] | 3.7752 [1.1899, 6.4537] | 0.4554 [0.4151, 0.5001] | 2.5356 [2.2167, 2.9052] | 7.1613 [5.7324, 8.7012] | 1.0503 [0.7100, 1.4289] | 2.3260 [1.5148, 3.2545] |
| I10R0A5 (0,1) | 0.5899 [0.5076, 0.6602] | 0.2586 [0.1902, 0.3252] | 0.0530 [0.0295, 0.0822] | -0.8019 [-0.9107, -0.6863] | -6.2574 [-7.0505, -5.4234] | -15.7538 [-18.5649, -12.7562] | 1.0520 [0.9781, 1.1295] | 7.1695 [6.5716, 7.7984] | 19.2081 [17.2658, 21.1998] | 1.6188 [1.0434, 2.3272] | 2.1365 [1.4293, 3.0631] |
| I10R0A5L (0,1) | 0.3852 [0.2935, 0.4719] | 0.1604 [0.1045, 0.2197] | 0.0350 [0.0179, 0.0550] | -0.4539 [-0.5624, -0.3373] | -3.9801 [-4.6379, -3.2540] | -10.3314 [-12.3119, -8.0901] | 0.7637 [0.7030, 0.8249] | 4.9542 [4.5061, 5.4383] | 13.1313 [11.7196, 14.7034] | 1.3398 [0.8546, 1.9431] | 1.8844 [1.2577, 2.7013] |
| I10R0M5 (0,1) | 0.8737 [0.8176, 0.9191] | 0.6908 [0.6102, 0.7630] | 0.0654 [0.0378, 0.0987] | -1.1037 [-1.2468, -0.9695] | -15.7706 [-19.8054, -11.7181] | -48.1414 [-61.7632, -33.8481] | 1.3679 [1.2375, 1.5208] | 19.5076 [16.1343, 22.9679] | 57.4782 [50.7509, 64.8401] | 1.8503 [1.2289, 2.6007] | 2.7059 [1.8928, 3.7647] |
| I10RLc (0,1) | 0.0383 [0.0190, 0.0628] | 0.0016 [0.0000, 0.0051] | 0.2181 [0.1459, 0.2897] | 0.2229 [-0.0327, 0.5472] | 2.0938 [1.2319, 3.0706] | 6.7474 [4.5816, 8.9276] | 0.7885 [0.5579, 1.0945] | 3.2710 [2.6177, 4.1001] | 8.0958 [6.4712, 9.8010] | 1.0876 [0.7460, 1.4643] | 2.1535 [1.4770, 2.9189] |
| I10RLcA5 (0,1) | 0.1415 [0.1038, 0.1799] | 0.0000 [0.0000, 0.0000] | 0.1846 [0.1295, 0.2453] | -0.0350 [-0.4487, 0.4618] | -0.9935 [-2.5348, 0.7195] | 1.7059 [-1.9969, 5.6295] | 1.4509 [1.1163, 1.8792] | 6.4530 [5.4402, 7.6896] | 13.2854 [10.9111, 16.0284] | 1.6273 [1.0684, 2.3303] | 2.1572 [1.4720, 3.0623] |
| I10RLcM5 (0,1) | 0.4867 [0.4184, 0.5555] | 0.0187 [0.0108, 0.0273] | 0.2259 [0.1722, 0.2837] | -0.3104 [-0.7041, 0.1461] | -2.0968 [-4.2249, 0.0735] | 2.4804 [-1.7053, 6.8513] | 1.6424 [1.3486, 2.0128] | 8.4169 [7.2754, 9.7230] | 15.7552 [13.1288, 18.7193] | 1.7800 [1.1768, 2.5370] | 2.2743 [1.5065, 3.3193] |
| I10RQ50 (0,1) | 0.0801 [0.0480, 0.1174] | 0.0265 [0.0102, 0.0465] | 0.2593 [0.1862, 0.3304] | 0.2697 [0.0335, 0.5486] | 2.4645 [1.6073, 3.4225] | 7.6520 [5.6761, 9.6557] | 0.8241 [0.5966, 1.0699] | 3.7859 [3.0818, 4.6228] | 9.4150 [7.8273, 11.0990] | 1.1816 [0.8261, 1.5783] | 2.5241 [1.8524, 3.2957] |
| I10RQ50A5 (0,1) | 0.2429 [0.1798, 0.3035] | 0.0491 [0.0261, 0.0741] | 0.2469 [0.1723, 0.3228] | 0.0958 [-0.3835, 0.6132] | 0.1898 [-1.9117, 2.4887] | 5.0624 [0.6618, 10.0998] | 1.5594 [1.1928, 1.9615] | 7.2054 [5.7188, 8.8954] | 15.0135 [11.4206, 19.2536] | 2.1172 [1.3347, 2.9988] | 2.5790 [1.8559, 3.4542] |
| I10RQ50M5 (0,1) | 0.5756 [0.5018, 0.6486] | 0.1168 [0.0813, 0.1546] | 0.2477 [0.1845, 0.3185] | -0.0982 [-0.6460, 0.4743] | -0.8809 [-4.0130, 2.4903] | 2.4079 [-3.5213, 9.0091] | 1.8098 [1.4772, 2.1781] | 9.3022 [7.6778, 11.2168] | 20.0898 [15.1278, 25.9184] | 2.2152 [1.5551, 3.0144] | 2.4975 [1.7350, 3.5154] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
