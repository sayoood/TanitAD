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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0C5:I10R0M5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0B5:I10R0M5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0C5:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0B5:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0C10:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0C5:I10R0B5", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10R0C5 minus I10R0M5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0B5 minus I10R0M5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0C5 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0B5 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0C10 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0C5 minus I10R0B5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10R0C5 minus I10R0M5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0374 / -0.0214 | -0.0294 | [-0.0588, -0.0052] | 0.0320 (0.0320) | INSIDE-SPREAD |
| C contact rate | -0.0841 / -0.0374 | -0.0607 | [-0.1066, -0.0220] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0078 / -0.0047 | +0.0016 | [-0.0177, +0.0222] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2709 / +0.2406 | +0.2558 | [+0.2039, +0.3179] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +0.9288 / +1.8855 | +1.4072 | [-0.0775, +2.7740] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | +6.9967 / +6.3601 | +6.6784 | [+2.8818, +10.3515] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | -0.2908 / -0.2243 | -0.2576 | [-0.3274, -0.1977] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -4.3282 / -4.6289 | -4.4786 | [-5.8631, -3.2230] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -14.5752 / -5.8330 | -10.2041 | [-17.1589, -4.4854] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.3793 / -0.0578 | -0.2186 | [-0.5804, +0.0911] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.1741 / -0.1412 | -0.1576 | [-0.2632, -0.0784] | 0.5140 (0.5140) | INSIDE-SPREAD |

## I10R0B5 minus I10R0M5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.6975 / -0.7456 | -0.7215 | [-0.7808, -0.6589] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.5639 / -0.6464 | -0.6051 | [-0.6805, -0.5275] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0187 / +0.0374 | +0.0280 | [-0.0020, +0.0579] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.9804 / +1.1082 | +1.0443 | [+0.9160, +1.1906] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +12.9790 / +18.2203 | +15.5997 | [+11.5811, +19.6563] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +37.9026 / +57.6687 | +47.7857 | [+33.6809, +61.0762] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | -0.7400 / -0.8395 | -0.7898 | [-0.9002, -0.6875] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -14.3461 / -18.7566 | -16.5514 | [-19.8861, -13.1762] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -45.9654 / -54.6795 | -50.3225 | [-57.7788, -43.6864] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.7443 / -0.7107 | -0.7275 | [-1.1035, -0.4362] | 0.0330 (0.0330) | BETTER |
| time-gap error over the scored steps (s) | -0.6697 / -0.7460 | -0.7078 | [-1.0289, -0.4496] | 0.5140 (0.5140) | BETTER |

## I10R0C5 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.6673 / +0.7616 | +0.7144 | [+0.6338, +0.7998] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.4891 / +0.6526 | +0.5709 | [+0.4432, +0.6852] | 0.0343 (0.0343) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0171 / -0.1293 | -0.0732 | [-0.1617, +0.0107] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.5524 / -0.6963 | -0.6244 | [-0.7630, -0.5030] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -12.2768 / -17.3466 | -14.8117 | [-18.4508, -11.1882] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -32.4542 / -56.0598 | -44.2570 | [-59.9364, -29.3446] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.5418 / +0.7331 | +0.6375 | [+0.5059, +0.7902] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +10.5751 / +14.4988 | +12.5370 | [+9.5682, +15.5289] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +32.4889 / +48.3036 | +40.3962 | [+30.6569, +50.9178] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.4228 / +0.7538 | +0.5883 | [+0.1620, +1.2234] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.6710 / +0.3418 | +0.5064 | [-0.0010, +0.9542] | 0.5140 (0.5140) | OPEN |

## I10R0B5 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0071 / +0.0374 | +0.0222 | [-0.0127, +0.0648] | 0.0320 (0.0320) | OPEN |
| C contact rate | +0.0093 / +0.0436 | +0.0265 | [-0.0139, +0.0656] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0062 / -0.0872 | -0.0467 | [-0.1224, +0.0215] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1570 / +0.1713 | +0.1641 | [+0.0975, +0.2406] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | -0.2266 / -1.0117 | -0.6192 | [-1.3892, +0.0815] | 1.0787 (1.0787) | OPEN |
| gap bias 6.0 s (m) | -1.5483 / -4.7513 | -3.1498 | [-5.8960, -0.4548] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0927 / +0.1179 | +0.1053 | [+0.0556, +0.1605] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.5572 / +0.3711 | +0.4642 | [+0.0513, +0.8639] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +1.0987 / -0.5429 | +0.2779 | [-1.4118, +1.8528] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.0578 / +0.1009 | +0.0793 | [-0.2168, +0.4256] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.1754 / -0.2629 | -0.0438 | [-0.5516, +0.3815] | 0.5140 (0.5140) | OPEN |

## I10R0C10 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0071 / +0.0409 | +0.0240 | [-0.0319, +0.0816] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0109 / +0.0421 | +0.0156 | [-0.0286, +0.0599] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.2118 / +0.2118 | +0.2118 | [+0.1582, +0.2686] | 0.0903 (0.0903) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +2.2023 / +1.9441 | +2.0732 | [+1.6377, +2.5324] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +5.6463 / +2.7643 | +4.2053 | [+0.9133, +6.9646] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +1.0635 / +1.0786 | +1.0711 | [+0.7767, +1.3814] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +4.5794 / +6.7856 | +5.6825 | [+3.5397, +7.9539] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +1.0741 / +0.9965 | +1.0353 | [+0.6747, +1.3843] | 0.5140 (0.5140) | WORSE |

## I10R0C5 minus I10R0B5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.6601 / +0.7242 | +0.6922 | [+0.6225, +0.7637] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.4798 / +0.6090 | +0.5444 | [+0.4356, +0.6436] | 0.0343 (0.0343) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0109 / -0.0421 | -0.0265 | [-0.0624, +0.0097] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.7094 / -0.8676 | -0.7885 | [-0.9269, -0.6720] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -12.0502 / -16.3349 | -14.1925 | [-17.3690, -10.9570] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -30.9059 / -51.3086 | -41.1072 | [-54.5037, -28.4569] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.4492 / +0.6152 | +0.5322 | [+0.4203, +0.6580] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +10.0179 / +14.1277 | +12.0728 | [+8.8958, +15.1478] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +31.3902 / +48.8465 | +40.1183 | [+29.8051, +51.6546] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.3650 / +0.6529 | +0.5090 | [+0.2598, +0.9113] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.4956 / +0.6048 | +0.5502 | [+0.3304, +0.8091] | 0.5140 (0.5140) | WORSE |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10R0B5 (0,1) | 0.1521 [0.1018, 0.2023] | 0.0857 [0.0480, 0.1265] | 0.0935 [0.0593, 0.1300] | -0.0594 [-0.1529, 0.0442] | -0.1709 [-0.7715, 0.4592] | -0.3557 [-1.9097, 1.2465] | 0.5782 [0.5133, 0.6566] | 2.9562 [2.5788, 3.4042] | 7.1557 [6.1615, 8.2528] | 1.1228 [0.7390, 1.6089] | 1.9981 [1.2943, 2.8406] |
| I10R0C10 (0,1) | 0.1539 [0.1079, 0.2022] | 0.0748 [0.0464, 0.1078] | 0.3520 [0.2687, 0.4350] | -0.2236 [-0.2757, -0.1682] | 2.5214 [1.8035, 3.2494] | 6.9994 [4.6887, 9.4838] | 0.4729 [0.4299, 0.5219] | 3.5631 [3.1336, 4.0115] | 12.5603 [9.7445, 15.6164] | 1.0435 [0.7059, 1.4212] | 3.0771 [2.3359, 3.9956] |
| I10R0C5 (0,1) | 0.8443 [0.7805, 0.8992] | 0.6301 [0.5222, 0.7273] | 0.0670 [0.0363, 0.1056] | -0.8479 [-0.9911, -0.7285] | -14.3635 [-17.4989, -11.2362] | -41.4629 [-55.0771, -28.6468] | 1.1104 [0.9710, 1.2809] | 15.0290 [11.8692, 18.1955] | 47.2741 [36.8435, 58.7339] | 1.6318 [1.0542, 2.4072] | 2.5482 [1.7862, 3.5246] |
| I10R0M5 (0,1) | 0.8737 [0.8176, 0.9191] | 0.6908 [0.6102, 0.7630] | 0.0654 [0.0378, 0.0987] | -1.1037 [-1.2468, -0.9695] | -15.7706 [-19.8054, -11.7181] | -48.1414 [-61.7632, -33.8481] | 1.3679 [1.2375, 1.5208] | 19.5076 [16.1343, 22.9679] | 57.4782 [50.7509, 64.8401] | 1.8503 [1.2289, 2.6007] | 2.7059 [1.8928, 3.7647] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
