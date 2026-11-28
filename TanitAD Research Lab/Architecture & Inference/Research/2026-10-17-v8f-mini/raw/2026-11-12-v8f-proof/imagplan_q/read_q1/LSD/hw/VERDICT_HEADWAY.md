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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPlq_LSD:IPoq_LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcq_LSD:IPoq_LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcq_LSD:IPlq_LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcqq_LSD:IPcq_LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcqq_LSD:IPlq_LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcqq_LSD:IPoq_LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcq0q_LSD:IPcq_LSD", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "IPcqidq_LSD:IPcq_LSD", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, IPlq_LSD minus IPoq_LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcq_LSD minus IPoq_LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcq_LSD minus IPlq_LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcqq_LSD minus IPcq_LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcqq_LSD minus IPlq_LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcqq_LSD minus IPoq_LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcq0q_LSD minus IPcq_LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, IPcqidq_LSD minus IPcq_LSD over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## IPlq LSD minus IPoq LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0925 / -0.0996 | -0.0961 | [-0.1362, -0.0600] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0187 / -0.0717 | -0.0452 | [-0.0906, -0.0109] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1012 / +0.0717 | +0.0864 | [+0.0506, +0.1323] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.8881 / +0.8930 | +0.8905 | [+0.3627, +1.5689] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.5263 / +3.4851 | +3.5057 | [+1.5232, +6.0679] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.0362 / +6.2800 | +6.1581 | [+3.1569, +10.1054] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.7697 / +0.7328 | +0.7513 | [+0.2711, +1.3617] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.8849 / +2.8125 | +2.8487 | [+0.9177, +5.3348] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +5.1370 / +3.7091 | +4.4230 | [+1.2002, +8.5206] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.3141 / +0.3004 | +0.3073 | [+0.1441, +0.5172] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.0465 / +0.7717 | +0.9091 | [+0.3718, +1.6166] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPcq LSD minus IPoq LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0445 / -0.0302 | -0.0374 | [-0.0591, -0.0194] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0047 / -0.0202 | -0.0125 | [-0.0255, -0.0025] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1199 / +0.0639 | +0.0919 | [+0.0456, +0.1540] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.8279 / +0.7878 | +0.8079 | [+0.3464, +1.3675] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.3681 / +3.0987 | +3.2334 | [+1.4377, +5.4605] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +6.2630 / +5.5387 | +5.9008 | [+3.1140, +9.4179] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.7302 / +0.6510 | +0.6906 | [+0.2809, +1.1892] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.9737 / +2.9331 | +2.9534 | [+1.2542, +5.0653] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +5.6073 / +3.9506 | +4.7789 | [+1.7303, +8.5282] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.1893 / +0.1753 | +0.1823 | [+0.0845, +0.3020] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.0259 / +0.8730 | +0.9495 | [+0.4291, +1.6008] | 1.7849 (0.8089) | INSIDE-SPREAD |

## IPcq LSD minus IPlq LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0480 / +0.0694 | +0.0587 | [+0.0316, +0.0899] | 0.0463 (0.0214) | WORSE |
| C contact rate | +0.0140 / +0.0514 | +0.0327 | [+0.0079, +0.0693] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0187 / -0.0078 | +0.0055 | [-0.0216, +0.0357] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0601 / -0.1052 | -0.0827 | [-0.2790, +0.0901] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.1581 / -0.3863 | -0.2722 | [-1.0245, +0.4124] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.2268 / -0.7413 | -0.2573 | [-1.6109, +1.1080] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0396 / -0.0818 | -0.0607 | [-0.2424, +0.0904] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0888 / +0.1206 | +0.1047 | [-0.6333, +0.7567] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.4704 / +0.2415 | +0.3559 | [-0.7862, +1.5551] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.1248 / -0.1252 | -0.1250 | [-0.2829, -0.0045] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.0206 / +0.1013 | +0.0404 | [-0.3672, +0.3729] | 1.7849 (0.8089) | OPEN |

## IPcqq LSD minus IPcq LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0409 / -0.0534 | -0.0472 | [-0.0720, -0.0251] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0140 / -0.0405 | -0.0273 | [-0.0537, -0.0078] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0779 / +0.1153 | +0.0966 | [+0.0541, +0.1384] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.9675 / +0.9982 | +0.9828 | [+0.6662, +1.3137] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.3197 / +3.9527 | +3.6362 | [+2.3929, +5.0106] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +5.2442 / +7.0874 | +6.1658 | [+3.8205, +8.8103] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.8340 / +0.9061 | +0.8700 | [+0.5731, +1.1940] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +3.0641 / +3.5346 | +3.2994 | [+2.1382, +4.5823] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +4.7984 / +6.2163 | +5.5073 | [+3.2149, +8.1113] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +1.5166 / +1.0054 | +1.2610 | [+0.7231, +1.9392] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +2.1174 / +1.7688 | +1.9431 | [+1.2948, +2.6747] | 1.7849 (0.8089) | WORSE |

## IPcqq LSD minus IPlq LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0071 / +0.0160 | +0.0116 | [-0.0025, +0.0286] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / +0.0109 | +0.0055 | [+0.0000, +0.0176] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0966 / +0.1075 | +0.1020 | [+0.0638, +0.1383] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.9073 / +0.8931 | +0.9002 | [+0.5618, +1.2479] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +3.1616 / +3.5664 | +3.3640 | [+2.0935, +4.7924] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +5.4710 / +6.3461 | +5.9085 | [+3.4649, +8.4809] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +0.7944 / +0.8243 | +0.8094 | [+0.4927, +1.1440] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +3.1529 / +3.6553 | +3.4041 | [+2.1053, +4.8621] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +5.2687 / +6.4578 | +5.8633 | [+3.4197, +8.5314] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +1.3918 / +0.8803 | +1.1360 | [+0.6471, +1.7694] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +2.0968 / +1.8701 | +1.9834 | [+1.3106, +2.7012] | 1.7849 (0.8089) | WORSE |

## IPcqq LSD minus IPoq LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0854 / -0.0836 | -0.0845 | [-0.1198, -0.0514] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0187 / -0.0607 | -0.0397 | [-0.0761, -0.0109] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1978 / +0.1791 | +0.1885 | [+0.1332, +0.2447] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +1.7954 / +1.7861 | +1.7907 | [+1.1589, +2.4699] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +6.6879 / +7.0514 | +6.8696 | [+4.4802, +9.5392] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +11.5072 / +12.6261 | +12.0666 | [+8.1215, +16.4695] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.5641 / +1.5571 | +1.5606 | [+0.9898, +2.1894] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +6.0378 / +6.4678 | +6.2528 | [+3.9903, +8.9053] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +10.4057 / +10.1668 | +10.2863 | [+6.2162, +14.6103] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +1.7059 / +1.1807 | +1.4433 | [+0.8758, +2.1412] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +3.1433 / +2.6418 | +2.8925 | [+1.9321, +3.9129] | 1.7849 (0.8089) | WORSE |

## IPcq0q LSD minus IPcq LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0712 / -0.1085 | -0.0899 | [-0.1376, -0.0510] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0140 / -0.0514 | -0.0327 | [-0.0693, -0.0079] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.4361 / +0.4969 | +0.4665 | [+0.3936, +0.5375] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +7.3343 / +6.6590 | +6.9966 | [+6.1352, +7.8120] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +18.7196 / +18.2746 | +18.4971 | [+16.1882, +20.7351] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +26.4989 / +27.3545 | +26.9267 | [+22.6452, +30.9845] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +6.8615 / +6.2305 | +6.5460 | [+5.7068, +7.3106] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +17.9312 / +17.0566 | +17.4939 | [+15.1540, +19.7328] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +25.5867 / +25.4463 | +25.5165 | [+21.3707, +29.6549] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +32.4007 / +29.2749 | +30.8378 | [+25.6037, +36.1722] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +25.6846 / +24.2853 | +24.9849 | [+21.7028, +28.2071] | 1.7849 (0.8089) | WORSE |

## IPcqidq LSD minus IPcq LSD

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| IPcq0q_LSD (0,1) | 0.0071 [0.0015, 0.0150] | 0.0000 [0.0000, 0.0000] | 0.8115 [0.7524, 0.8582] | 7.6014 [6.6866, 8.3928] | 23.2122 [20.3936, 25.8141] | 39.5873 [34.3111, 44.4994] | 7.7466 [6.8845, 8.4994] | 23.3441 [20.5243, 25.9155] | 39.6326 [34.3804, 44.5186] | 32.4841 [26.9586, 38.0039] | 28.4735 [24.9761, 32.0556] |
| IPcq_LSD (0,1) | 0.0970 [0.0559, 0.1481] | 0.0327 [0.0079, 0.0693] | 0.3450 [0.2562, 0.4338] | 0.6047 [0.1609, 1.1367] | 4.7151 [2.7290, 6.9349] | 12.6606 [8.7548, 16.7799] | 1.2006 [0.7847, 1.7181] | 5.8502 [4.0689, 7.9826] | 14.1161 [10.6950, 17.9822] | 1.6463 [0.9976, 2.3733] | 3.4886 [2.3917, 4.5975] |
| IPcqidq_LSD (0,1) | 0.0970 [0.0559, 0.1481] | 0.0327 [0.0079, 0.0693] | 0.3450 [0.2562, 0.4338] | 0.6047 [0.1609, 1.1367] | 4.7151 [2.7290, 6.9349] | 12.6606 [8.7548, 16.7799] | 1.2006 [0.7847, 1.7181] | 5.8502 [4.0689, 7.9826] | 14.1161 [10.6950, 17.9822] | 1.6463 [0.9976, 2.3733] | 3.4886 [2.3917, 4.5975] |
| IPcqq_LSD (0,1) | 0.0498 [0.0259, 0.0858] | 0.0055 [0.0000, 0.0176] | 0.4416 [0.3612, 0.5160] | 1.5876 [0.9795, 2.2303] | 8.3513 [5.8643, 10.9396] | 18.8264 [14.1186, 23.8072] | 2.0706 [1.4881, 2.6999] | 9.1496 [6.7694, 11.6826] | 19.6234 [15.1267, 24.5975] | 2.9073 [1.8936, 4.0970] | 5.4316 [4.0233, 6.9430] |
| IPlq_LSD (0,1) | 0.0383 [0.0194, 0.0658] | 0.0000 [0.0000, 0.0000] | 0.3396 [0.2628, 0.4140] | 0.6874 [0.1738, 1.3222] | 4.9873 [2.8242, 7.5697] | 12.9179 [9.1835, 17.2267] | 1.2612 [0.7756, 1.8788] | 5.7455 [3.7011, 8.2504] | 13.7602 [10.2035, 18.0364] | 1.7713 [1.1057, 2.5207] | 3.4482 [2.2877, 4.6870] |
| IPoq_LSD (0,1) | 0.1343 [0.0873, 0.1879] | 0.0452 [0.0109, 0.0906] | 0.2531 [0.1958, 0.3138] | -0.2031 [-0.2748, -0.1325] | 1.4817 [0.8172, 2.2957] | 6.7598 [4.6651, 8.6599] | 0.5100 [0.4630, 0.5586] | 2.8968 [2.3956, 3.5375] | 9.3372 [8.1968, 10.6886] | 1.4640 [0.8170, 2.2047] | 2.5391 [1.6003, 3.5518] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
