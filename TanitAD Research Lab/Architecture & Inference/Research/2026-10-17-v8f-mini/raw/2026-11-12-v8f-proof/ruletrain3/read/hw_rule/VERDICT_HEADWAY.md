# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 18, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GFTRL:GFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GOLRL:GOL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GOL4RL:GOL4", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GOL:GFTRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GOL4:GFTRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GOLRL:GFTRL", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GOL4RL:GFTRL", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, GFTRL minus GFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GOLRL minus GOL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GOL4RL minus GOL4 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GOL minus GFTRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GOL4 minus GFTRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GOLRL minus GFTRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GOL4RL minus GFTRL over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## GFTRL minus GFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0569 / -0.0783 | -0.0676 | [-0.1008, -0.0362] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0358 / -0.0452 | -0.0405 | [-0.0648, -0.0212] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0670 / +0.0685 | +0.0678 | [+0.0281, +0.1134] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.4346 / +0.4358 | +0.4352 | [+0.1818, +0.7601] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +1.5137 / +1.1397 | +1.3267 | [+0.5822, +2.2604] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.8306 / +2.8614 | +2.8460 | [+1.3194, +4.6864] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.3354 / +0.3216 | +0.3285 | [+0.0964, +0.6274] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.7415 / +0.5542 | +0.6479 | [+0.0752, +1.3464] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.4310 / +1.1152 | +1.2731 | [+0.1971, +2.6265] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0448 / +0.0424 | +0.0436 | [+0.0203, +0.0708] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0738 / +0.0992 | +0.0865 | [+0.0093, +0.1686] | 1.7849 (0.8089) | INSIDE-SPREAD |

## GOLRL minus GOL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0552 / -0.0445 | -0.0498 | [-0.0734, -0.0287] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0171 / -0.0078 | -0.0125 | [-0.0240, -0.0040] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0016 / -0.0062 | -0.0039 | [-0.0094, +0.0015] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0579 / +0.0533 | +0.0556 | [+0.0274, +0.0895] | 0.0689 (0.0315) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.1355 / +0.1205 | +0.1280 | [+0.0419, +0.2299] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.1533 / +0.3756 | +0.2644 | [-0.2228, +0.7580] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0354 / +0.0329 | +0.0341 | [+0.0113, +0.0654] | 0.0426 (0.0189) | INSIDE-SPREAD |
| gap error 4.0 s (m) | +0.0296 / +0.0468 | +0.0382 | [-0.0392, +0.1484] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.2582 / +0.1751 | -0.0416 | [-0.5339, +0.4828] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0304 / +0.0399 | +0.0351 | [+0.0109, +0.0668] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0037 / +0.1117 | +0.0577 | [-0.0517, +0.1757] | 1.7849 (0.8089) | OPEN |

## GOL4RL minus GOL4

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0427 / -0.0338 | -0.0383 | [-0.0599, -0.0205] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0016 / -0.0062 | -0.0039 | [-0.0112, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0031 / +0.0062 | +0.0047 | [-0.0017, +0.0120] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0235 / +0.0175 | +0.0205 | [+0.0081, +0.0356] | 0.0689 (0.0315) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.0443 / -0.0292 | +0.0076 | [-0.0840, +0.0671] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.1107 / +0.0298 | -0.0404 | [-0.2971, +0.3070] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0170 / +0.0171 | +0.0170 | [+0.0053, +0.0308] | 0.0426 (0.0189) | INSIDE-SPREAD |
| gap error 4.0 s (m) | +0.0127 / -0.1013 | -0.0443 | [-0.1527, +0.0268] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -0.1481 / -0.1289 | -0.1385 | [-0.3822, +0.1625] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0533 / +0.0106 | +0.0319 | [+0.0024, +0.0916] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.0061 / +0.0621 | +0.0341 | [-0.0762, +0.1322] | 1.7849 (0.8089) | OPEN |

## GOL minus GFTRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0409 / +0.0125 | +0.0267 | [-0.0043, +0.0578] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0171 / +0.0078 | +0.0125 | [+0.0040, +0.0240] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1947 / +0.2383 | +0.2165 | [+0.1587, +0.2744] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.7901 / +2.1620 | +1.9761 | [+1.2057, +2.8986] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +7.6242 / +7.5889 | +7.6066 | [+5.0499, +10.2799] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +13.2741 / +12.2614 | +12.7677 | [+8.5942, +17.1031] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.5901 / +1.8950 | +1.7426 | [+0.9744, +2.6969] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +7.3124 / +7.1737 | +7.2431 | [+4.7050, +9.9315] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +13.4468 / +11.9770 | +12.7119 | [+8.5300, +17.0989] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +2.3648 / +1.9256 | +2.1452 | [+1.3492, +3.0246] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +2.5056 / +2.2211 | +2.3634 | [+1.8024, +2.9943] | 1.7849 (0.8089) | WORSE |

## GOL4 minus GFTRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0178 / +0.0018 | +0.0098 | [-0.0192, +0.0396] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0016 / +0.0062 | +0.0039 | [+0.0000, +0.0112] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.3505 / +0.3769 | +0.3637 | [+0.3054, +0.4195] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +2.9552 / +3.1900 | +3.0726 | [+2.1421, +4.0980] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +11.0545 / +10.9508 | +11.0026 | [+8.3626, +13.8212] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +20.0582 / +19.0369 | +19.5476 | [+15.2108, +24.1147] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +2.6843 / +2.8402 | +2.7623 | [+1.8210, +3.7876] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +10.4697 / +10.3064 | +10.3880 | [+7.7449, +13.1879] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +19.5314 / +18.1097 | +18.8205 | [+14.4552, +23.4152] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +4.9165 / +4.6606 | +4.7885 | [+3.4522, +6.3454] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +5.4641 / +3.9071 | +4.6856 | [+3.4159, +6.0608] | 1.7849 (0.8089) | WORSE |

## GOLRL minus GFTRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0142 / -0.0320 | -0.0231 | [-0.0473, -0.0028] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.1931 / +0.2321 | +0.2126 | [+0.1573, +0.2696] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.8480 / +2.2153 | +2.0317 | [+1.2732, +2.9436] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +7.7598 / +7.7095 | +7.7346 | [+5.1916, +10.4081] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +13.4273 / +12.6370 | +13.0321 | [+8.9289, +17.4289] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.6256 / +1.9279 | +1.7767 | [+1.0107, +2.7315] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +7.3419 / +7.2205 | +7.2812 | [+4.7424, +10.0004] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +13.1885 / +12.1521 | +12.6703 | [+8.5008, +17.0300] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +2.3952 / +1.9655 | +2.1803 | [+1.3796, +3.0846] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +2.5094 / +2.3328 | +2.4211 | [+1.8584, +3.0336] | 1.7849 (0.8089) | WORSE |

## GOL4RL minus GFTRL

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0249 / -0.0320 | -0.0285 | [-0.0520, -0.0071] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.3536 / +0.3832 | +0.3684 | [+0.3094, +0.4251] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +2.9787 / +3.2076 | +3.0932 | [+2.1623, +4.1187] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +11.0987 / +10.9216 | +11.0102 | [+8.3655, +13.8072] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +19.9475 / +19.0668 | +19.5071 | [+15.1471, +24.1233] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +2.7012 / +2.8574 | +2.7793 | [+1.8399, +3.8152] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +10.4824 / +10.2051 | +10.3437 | [+7.6765, +13.1499] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +19.3833 / +17.9808 | +18.6820 | [+14.2716, +23.2710] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +4.9698 / +4.6712 | +4.8205 | [+3.4635, +6.3935] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +5.4702 / +3.9691 | +4.7197 | [+3.4495, +6.0641] | 1.7849 (0.8089) | WORSE |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| GFT (0,1) | 0.1121 [0.0711, 0.1570] | 0.0405 [0.0212, 0.0648] | 0.1931 [0.1119, 0.2846] | -0.1611 [-0.2359, -0.0881] | 1.0823 [0.4765, 1.7433] | 5.2278 [2.7695, 7.6887] | 0.4680 [0.4239, 0.5183] | 2.7143 [2.3027, 3.2118] | 7.9076 [6.3657, 9.6000] | 1.2414 [0.7961, 1.7429] | 2.3332 [1.5451, 3.2287] |
| GFTRL (0,1) | 0.0445 [0.0255, 0.0664] | 0.0000 [0.0000, 0.0000] | 0.2609 [0.1758, 0.3522] | 0.2741 [0.0113, 0.5893] | 2.4091 [1.4418, 3.4539] | 8.0738 [5.4192, 10.5852] | 0.7965 [0.5692, 1.0945] | 3.3622 [2.6218, 4.1937] | 9.1807 [7.1953, 11.2347] | 1.2850 [0.8478, 1.7875] | 2.4197 [1.6515, 3.2980] |
| GOL (0,1) | 0.0712 [0.0422, 0.1067] | 0.0125 [0.0040, 0.0240] | 0.4774 [0.3931, 0.5564] | 2.2502 [1.2646, 3.4589] | 10.0156 [6.9124, 13.3702] | 20.8415 [15.4135, 26.6666] | 2.5390 [1.5762, 3.7429] | 10.6053 [7.6094, 13.9370] | 21.8926 [16.7320, 27.4750] | 3.4302 [2.4293, 4.5261] | 4.7831 [3.7688, 5.8412] |
| GOL4 (0,1) | 0.0543 [0.0311, 0.0824] | 0.0039 [0.0000, 0.0112] | 0.6246 [0.5394, 0.6974] | 3.3468 [2.2200, 4.6308] | 13.4117 [10.1503, 16.7908] | 27.6214 [21.8552, 33.5413] | 3.5587 [2.4493, 4.8387] | 13.7502 [10.5978, 17.0573] | 28.0012 [22.5196, 33.7936] | 6.0735 [4.5921, 7.7298] | 7.1053 [5.3575, 8.9175] |
| GOL4RL (0,1) | 0.0160 [0.0064, 0.0293] | 0.0000 [0.0000, 0.0000] | 0.6293 [0.5445, 0.7012] | 3.3673 [2.2346, 4.6541] | 13.4192 [10.1719, 16.8256] | 27.5810 [21.8889, 33.5489] | 3.5758 [2.4597, 4.8605] | 13.7059 [10.5331, 17.0346] | 27.8627 [22.3505, 33.7033] | 6.1054 [4.6037, 7.8128] | 7.1394 [5.4212, 8.9251] |
| GOLRL (0,1) | 0.0214 [0.0074, 0.0412] | 0.0000 [0.0000, 0.0000] | 0.4735 [0.3874, 0.5546] | 2.3058 [1.3234, 3.5167] | 10.1437 [7.0362, 13.5285] | 21.1060 [15.6786, 27.0111] | 2.5732 [1.6146, 3.7702] | 10.6434 [7.5897, 13.9807] | 21.8510 [16.7638, 27.3840] | 3.4653 [2.4538, 4.5788] | 4.8408 [3.8606, 5.8417] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
