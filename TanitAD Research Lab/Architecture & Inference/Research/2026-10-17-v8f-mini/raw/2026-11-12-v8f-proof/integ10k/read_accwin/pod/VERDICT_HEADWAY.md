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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0W5:I10R0M5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0Z5:I10R0M5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0W5:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0Z5:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0W10:I10R0", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0W5:I10R0Z5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0W5:I10R0C5", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10R0Z5:I10R0C5", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10R0, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10R0W5 minus I10R0M5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0Z5 minus I10R0M5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0W5 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0Z5 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0W10 minus I10R0 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0W5 minus I10R0Z5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0W5 minus I10R0C5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10R0Z5 minus I10R0C5 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10R0W5 minus I10R0M5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.4217 / -0.4324 | -0.4270 | [-0.4990, -0.3555] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.4860 / -0.5654 | -0.5257 | [-0.6038, -0.4489] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | -0.0047 / +0.0093 | +0.0023 | [-0.0141, +0.0237] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.7064 / +0.8156 | +0.7610 | [+0.6593, +0.8749] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +9.9493 / +15.3297 | +12.6395 | [+8.5552, +16.6010] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +30.7016 / +50.1801 | +40.4409 | [+26.8900, +52.9391] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | -0.6525 / -0.7451 | -0.6988 | [-0.7954, -0.6101] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -13.0295 / -17.0496 | -15.0396 | [-18.1320, -11.9374] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -40.2091 / -47.0382 | -43.6236 | [-50.5270, -38.0599] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.4945 / -0.4393 | -0.4669 | [-0.6509, -0.3207] | 0.0330 (0.0330) | BETTER |
| time-gap error over the scored steps (s) | -0.7481 / -0.8405 | -0.7943 | [-0.9208, -0.6726] | 0.5140 (0.5140) | BETTER |

## I10R0Z5 minus I10R0M5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.6299 / -0.6851 | -0.6575 | [-0.7325, -0.5806] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.5280 / -0.6184 | -0.5732 | [-0.6523, -0.4939] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | -0.0218 / +0.0000 | -0.0109 | [-0.0355, +0.0146] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.8064 / +0.9177 | +0.8621 | [+0.7523, +0.9878] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +11.4802 / +16.6071 | +14.0436 | [+10.1292, +17.9577] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +34.1575 / +53.4725 | +43.8150 | [+30.2710, +56.6271] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | -0.6484 / -0.7318 | -0.6901 | [-0.7863, -0.6011] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -13.9327 / -18.2274 | -16.0800 | [-19.3159, -12.7739] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -45.0394 / -53.3785 | -49.2090 | [-56.3865, -42.6705] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.5781 / -0.5283 | -0.5532 | [-0.8697, -0.3134] | 0.0330 (0.0330) | BETTER |
| time-gap error over the scored steps (s) | -0.5589 / -0.5159 | -0.5374 | [-0.8855, -0.2093] | 0.5140 (0.5140) | BETTER |

## I10R0W5 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.2829 / +0.3505 | +0.3167 | [+0.2422, +0.3966] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0872 / +0.1246 | +0.1059 | [+0.0568, +0.1603] | 0.0343 (0.0343) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0296 / -0.1153 | -0.0724 | [-0.1539, -0.0037] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1170 / -0.1213 | -0.1192 | [-0.1697, -0.0648] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -3.2564 / -3.9024 | -3.5794 | [-4.3666, -2.8774] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -8.7493 / -12.2399 | -10.4946 | [-13.8997, -7.4041] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.1801 / +0.2124 | +0.1962 | [+0.1465, +0.2510] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +1.8738 / +2.0782 | +1.9760 | [+1.5092, +2.5139] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +6.8551 / +7.0984 | +6.9767 | [+5.0348, +8.9419] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.3076 / +0.3723 | +0.3400 | [-0.1163, +0.9449] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.0969 / -0.3575 | -0.1303 | [-0.6909, +0.3291] | 0.5140 (0.5140) | OPEN |

## I10R0Z5 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0747 / +0.0979 | +0.0863 | [+0.0244, +0.1500] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0452 / +0.0717 | +0.0584 | [+0.0130, +0.1036] | 0.0343 (0.0343) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0467 / -0.1246 | -0.0857 | [-0.1592, -0.0244] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0169 / -0.0192 | -0.0181 | [-0.0992, +0.0729] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | -1.7255 / -2.6250 | -2.1752 | [-3.0458, -1.3569] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -5.2934 / -8.9475 | -7.1205 | [-10.2105, -4.0210] | 3.6529 (3.6529) | BETTER |
| gap error 2.0 s (m) | +0.1842 / +0.2257 | +0.2049 | [+0.1418, +0.2736] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.9706 / +0.9003 | +0.9355 | [+0.4850, +1.4041] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +2.0248 / +0.7581 | +1.3914 | [-0.3271, +2.9740] | 1.5578 (1.5578) | OPEN |
| time-gap error 2.0 s (s) | +0.2240 / +0.2834 | +0.2537 | [-0.0490, +0.6523] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.2862 / -0.0329 | +0.1267 | [-0.3783, +0.5921] | 0.5140 (0.5140) | OPEN |

## I10R0W10 minus I10R0

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0214 / -0.0107 | -0.0160 | [-0.0655, +0.0316] | 0.0320 (0.0320) | OPEN |
| C contact rate | -0.0265 / +0.0125 | -0.0070 | [-0.0435, +0.0300] | 0.0343 (0.0343) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0576 / +0.0374 | +0.0475 | [+0.0154, +0.0778] | 0.0903 (0.0903) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0490 (0.0490) | OPEN |
| gap bias 4.0 s (m) | +0.9412 / +0.6498 | +0.7955 | [+0.4735, +1.1262] | 1.0787 (1.0787) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.0949 / +0.1670 | +1.1310 | [-0.7576, +2.7199] | 3.6529 (3.6529) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.3069 / +0.2182 | +0.2625 | [+0.0301, +0.4904] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +0.5847 / +1.0882 | +0.8365 | [+0.0950, +1.6991] | 1.5578 (1.5578) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | +0.5459 / +0.7327 | +0.6393 | [+0.3817, +0.9378] | 0.5140 (0.5140) | WORSE |

## I10R0W5 minus I10R0Z5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.2082 / +0.2527 | +0.2304 | [+0.1669, +0.2939] | 0.0320 (0.0320) | WORSE |
| C contact rate | +0.0421 / +0.0530 | +0.0475 | [+0.0254, +0.0739] | 0.0343 (0.0343) | WORSE |
| F falls behind by more than 10 m at the last scored step | +0.0171 / +0.0093 | +0.0132 | [-0.0051, +0.0324] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1001 / -0.1021 | -0.1011 | [-0.1446, -0.0625] | 0.0490 (0.0490) | BETTER |
| gap bias 4.0 s (m) | -1.5309 / -1.2774 | -1.4041 | [-1.8039, -1.0564] | 1.0787 (1.0787) | BETTER |
| gap bias 6.0 s (m) | -3.4559 / -3.2924 | -3.3741 | [-4.5437, -2.1948] | 3.6529 (3.6529) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0041 / -0.0133 | -0.0087 | [-0.0327, +0.0167] | 0.0069 (0.0069) | OPEN |
| gap error 4.0 s (m) | +0.9031 / +1.1778 | +1.0405 | [+0.7022, +1.3699] | 0.2417 (0.2417) | WORSE |
| gap error 6.0 s (m) | +4.8303 / +6.3403 | +5.5853 | [+3.8919, +7.3528] | 1.5578 (1.5578) | WORSE |
| time-gap error 2.0 s (s) | +0.0836 / +0.0890 | +0.0863 | [-0.1544, +0.3877] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.1893 / -0.3246 | -0.2569 | [-0.6080, +0.0476] | 0.5140 (0.5140) | OPEN |

## I10R0W5 minus I10R0C5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.3843 / -0.4110 | -0.3977 | [-0.4703, -0.3267] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.4019 / -0.5280 | -0.4650 | [-0.5673, -0.3618] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | -0.0125 / +0.0140 | +0.0008 | [-0.0229, +0.0227] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4354 / +0.5750 | +0.5052 | [+0.4105, +0.6127] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +9.0204 / +13.4442 | +11.2323 | [+8.0207, +14.3725] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +23.7049 / +43.8200 | +33.7624 | [+21.4483, +46.4293] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | -0.3618 / -0.5207 | -0.4412 | [-0.5540, -0.3430] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -8.7014 / -12.4206 | -10.5610 | [-13.3683, -7.7231] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -25.6338 / -41.2052 | -33.4195 | [-44.5321, -23.8004] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.1152 / -0.3815 | -0.2484 | [-0.4765, +0.0884] | 0.0330 (0.0330) | OPEN |
| time-gap error over the scored steps (s) | -0.5740 / -0.6993 | -0.6367 | [-0.7609, -0.5171] | 0.5140 (0.5140) | BETTER |

## I10R0Z5 minus I10R0C5

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.5925 / -0.6637 | -0.6281 | [-0.7079, -0.5455] | 0.0320 (0.0320) | BETTER |
| C contact rate | -0.4439 / -0.5810 | -0.5125 | [-0.6157, -0.4035] | 0.0343 (0.0343) | BETTER |
| F falls behind by more than 10 m at the last scored step | -0.0296 / +0.0047 | -0.0125 | [-0.0443, +0.0170] | 0.0903 (0.0903) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.5355 / +0.6771 | +0.6063 | [+0.5026, +0.7197] | 0.0490 (0.0490) | WORSE |
| gap bias 4.0 s (m) | +10.5513 / +14.7216 | +12.6365 | [+9.4989, +15.6763] | 1.0787 (1.0787) | WORSE |
| gap bias 6.0 s (m) | +27.1608 / +47.1124 | +37.1366 | [+24.8705, +49.9037] | 3.6529 (3.6529) | WORSE |
| gap error 2.0 s (m) | -0.3576 / -0.5075 | -0.4325 | [-0.5389, -0.3350] | 0.0069 (0.0069) | BETTER |
| gap error 4.0 s (m) | -9.6045 / -13.5984 | -11.6015 | [-14.5836, -8.5756] | 0.2417 (0.2417) | BETTER |
| gap error 6.0 s (m) | -30.4642 / -47.5455 | -39.0048 | [-50.2800, -28.8917] | 1.5578 (1.5578) | BETTER |
| time-gap error 2.0 s (s) | -0.1988 / -0.4705 | -0.3346 | [-0.6452, -0.1141] | 0.0330 (0.0330) | BETTER |
| time-gap error over the scored steps (s) | -0.3848 / -0.3747 | -0.3797 | [-0.6615, -0.0892] | 0.5140 (0.5140) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10R0 (0,1) | 0.1299 [0.0791, 0.1841] | 0.0592 [0.0288, 0.0966] | 0.1402 [0.0712, 0.2174] | -0.2236 [-0.2757, -0.1682] | 0.4483 [-0.3582, 1.2645] | 2.7941 [0.2796, 5.4221] | 0.4729 [0.4299, 0.5219] | 2.4920 [2.1883, 2.8518] | 6.8778 [5.5665, 8.2678] | 1.0435 [0.7059, 1.4212] | 2.0418 [1.3741, 2.7965] |
| I10R0C5 (0,1) | 0.8443 [0.7805, 0.8992] | 0.6301 [0.5222, 0.7273] | 0.0670 [0.0363, 0.1056] | -0.8479 [-0.9911, -0.7285] | -14.3635 [-17.4989, -11.2362] | -41.4629 [-55.0771, -28.6468] | 1.1104 [0.9710, 1.2809] | 15.0290 [11.8692, 18.1955] | 47.2741 [36.8435, 58.7339] | 1.6318 [1.0542, 2.4072] | 2.5482 [1.7862, 3.5246] |
| I10R0M5 (0,1) | 0.8737 [0.8176, 0.9191] | 0.6908 [0.6102, 0.7630] | 0.0654 [0.0378, 0.0987] | -1.1037 [-1.2468, -0.9695] | -15.7706 [-19.8054, -11.7181] | -48.1414 [-61.7632, -33.8481] | 1.3679 [1.2375, 1.5208] | 19.5076 [16.1343, 22.9679] | 57.4782 [50.7509, 64.8401] | 1.8503 [1.2289, 2.6007] | 2.7059 [1.8928, 3.7647] |
| I10R0W10 (0,1) | 0.1139 [0.0736, 0.1574] | 0.0522 [0.0276, 0.0784] | 0.1877 [0.1297, 0.2538] | -0.2236 [-0.2757, -0.1682] | 1.2437 [0.5671, 1.9071] | 3.9250 [2.3093, 5.7152] | 0.4729 [0.4299, 0.5219] | 2.7546 [2.4040, 3.1458] | 7.7143 [6.1726, 9.3531] | 1.0435 [0.7059, 1.4212] | 2.6811 [1.8659, 3.6560] |
| I10R0W5 (0,1) | 0.4466 [0.3615, 0.5284] | 0.1651 [0.1142, 0.2192] | 0.0678 [0.0374, 0.1030] | -0.3427 [-0.4178, -0.2570] | -3.1311 [-3.6690, -2.5103] | -7.7005 [-9.6377, -5.3580] | 0.6691 [0.6049, 0.7438] | 4.4680 [3.9562, 5.1133] | 13.8545 [11.8407, 16.1871] | 1.3834 [0.8009, 2.1162] | 1.9116 [1.1864, 2.8892] |
| I10R0Z5 (0,1) | 0.2162 [0.1489, 0.2851] | 0.1176 [0.0714, 0.1683] | 0.0545 [0.0323, 0.0815] | -0.2416 [-0.3451, -0.1223] | -1.7270 [-2.3181, -1.0712] | -4.3264 [-5.9363, -2.5736] | 0.6778 [0.6023, 0.7674] | 3.4275 [2.9799, 3.9379] | 8.2692 [7.0430, 9.5783] | 1.2971 [0.8284, 1.8486] | 2.1685 [1.3787, 3.1127] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
