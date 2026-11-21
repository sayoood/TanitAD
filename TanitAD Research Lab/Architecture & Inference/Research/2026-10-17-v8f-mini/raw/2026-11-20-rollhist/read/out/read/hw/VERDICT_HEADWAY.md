# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: F:/TanitAD-scratch/strat1/read/inputs/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 24, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.021352313167259773, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0C10XHb20:I10B0_R0C10XHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0C10XHb20:I10RH_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0G50Hb20:I10RH_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0C10Hb20:I10RH_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0C10XHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0G50Hb20:I10B0_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0C10Hb20:I10B0_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0C10XHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0Hb20:I10B0_R0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0C10Hb20:I10B0_R0C10Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0G50Hb20:I10B0_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0G0Hb20:I10B0_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0KHb20:I10B0_R0KHb20", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10RH_R0Hb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10RH_R0C10XHb20 minus I10B0_R0C10XHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0C10XHb20 minus I10RH_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0G50Hb20 minus I10RH_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0C10Hb20 minus I10RH_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0C10XHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0G50Hb20 minus I10B0_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0C10Hb20 minus I10B0_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0C10XHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0Hb20 minus I10B0_R0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0C10Hb20 minus I10B0_R0C10Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0G50Hb20 minus I10B0_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0G0Hb20 minus I10B0_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0KHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10RH R0C10XHb20 minus I10B0 R0C10XHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0178 / +0.0178 | +0.0178 | [+0.0064, +0.0313] | 0.0214 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0171 / +0.0125 | +0.0148 | [+0.0040, +0.0276] | 0.0125 (0.0125) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0685 / -0.0639 | -0.0662 | [-0.1025, -0.0380] | 0.0280 (0.0280) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0303 / -0.0396 | -0.0350 | [-0.0569, -0.0141] | 0.0095 (0.0095) | BETTER |
| gap bias 4.0 s (m) | -0.2982 / -0.5005 | -0.3993 | [-0.5954, -0.2054] | 0.1385 (0.1385) | BETTER |
| gap bias 6.0 s (m) | -1.8277 / -2.1552 | -1.9914 | [-2.6824, -1.3897] | 0.9373 (0.9373) | BETTER |
| gap error 2.0 s (m) | +0.0316 / +0.0219 | +0.0268 | [+0.0107, +0.0452] | 0.0230 (0.0230) | WORSE |
| gap error 4.0 s (m) | +0.0711 / -0.1617 | -0.0453 | [-0.2437, +0.1545] | 0.1360 (0.1360) | OPEN |
| gap error 6.0 s (m) | -0.8786 / -1.3742 | -1.1264 | [-1.8312, -0.4986] | 0.3535 (0.3535) | BETTER |
| time-gap error 2.0 s (s) | +0.0272 / +0.0476 | +0.0374 | [-0.0087, +0.0888] | 0.0180 (0.0180) | OPEN |
| time-gap error over the scored steps (s) | -0.7011 / -0.7134 | -0.7072 | [-1.0008, -0.4429] | 0.2433 (0.2433) | BETTER |

## I10RH R0C10XHb20 minus I10RH R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0409 / -0.0552 | -0.0480 | [-0.0745, -0.0227] | 0.0214 (0.0214) | BETTER |
| C contact rate | -0.0218 / -0.0280 | -0.0249 | [-0.0422, -0.0109] | 0.0125 (0.0125) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0935 / +0.1215 | +0.1075 | [+0.0753, +0.1446] | 0.0280 (0.0280) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0095 (0.0095) | OPEN |
| gap bias 4.0 s (m) | +1.0992 / +1.2448 | +1.1720 | [+1.0119, +1.3585] | 0.1385 (0.1385) | WORSE |
| gap bias 6.0 s (m) | +3.4739 / +4.4221 | +3.9480 | [+3.2053, +4.8402] | 0.9373 (0.9373) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0230 (0.0230) | OPEN |
| gap error 4.0 s (m) | +0.5120 / +0.4600 | +0.4860 | [+0.3031, +0.6737] | 0.1360 (0.1360) | WORSE |
| gap error 6.0 s (m) | +1.2874 / +1.5622 | +1.4248 | [+0.7629, +2.1700] | 0.3535 (0.3535) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0180 (0.0180) | OPEN |
| time-gap error over the scored steps (s) | +0.8690 / +1.1113 | +0.9902 | [+0.5804, +1.4751] | 0.2433 (0.2433) | WORSE |

## I10RH R0G50Hb20 minus I10RH R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0196 / -0.0320 | -0.0258 | [-0.0434, -0.0106] | 0.0214 (0.0214) | BETTER |
| C contact rate | -0.0125 / -0.0171 | -0.0148 | [-0.0252, -0.0064] | 0.0125 (0.0125) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0545 / +0.0701 | +0.0623 | [+0.0400, +0.0845] | 0.0280 (0.0280) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0095 (0.0095) | OPEN |
| gap bias 4.0 s (m) | +0.5943 / +0.6647 | +0.6295 | [+0.5414, +0.7286] | 0.1385 (0.1385) | WORSE |
| gap bias 6.0 s (m) | +1.9381 / +2.6118 | +2.2750 | [+1.7696, +2.8666] | 0.9373 (0.9373) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0230 (0.0230) | OPEN |
| gap error 4.0 s (m) | +0.2266 / +0.1886 | +0.2076 | [+0.1000, +0.3123] | 0.1360 (0.1360) | WORSE |
| gap error 6.0 s (m) | +0.6518 / +0.7847 | +0.7183 | [+0.3001, +1.1708] | 0.3535 (0.3535) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0180 (0.0180) | OPEN |
| time-gap error over the scored steps (s) | +0.5161 / +0.8317 | +0.6739 | [+0.3509, +1.0639] | 0.2433 (0.2433) | WORSE |

## I10RH R0C10Hb20 minus I10RH R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0071 / -0.0214 | -0.0142 | [-0.0283, -0.0042] | 0.0214 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0047 / -0.0093 | -0.0070 | [-0.0162, -0.0000] | 0.0125 (0.0125) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0265 / +0.0498 | +0.0382 | [+0.0208, +0.0640] | 0.0280 (0.0280) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0095 (0.0095) | OPEN |
| gap bias 4.0 s (m) | +0.3771 / +0.4885 | +0.4328 | [+0.3494, +0.5399] | 0.1385 (0.1385) | WORSE |
| gap bias 6.0 s (m) | +1.0438 / +1.4899 | +1.2669 | [+0.9075, +1.6678] | 0.9373 (0.9373) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0230 (0.0230) | OPEN |
| gap error 4.0 s (m) | +0.2302 / +0.2276 | +0.2289 | [+0.1604, +0.3018] | 0.1360 (0.1360) | WORSE |
| gap error 6.0 s (m) | +0.4236 / +0.6185 | +0.5211 | [+0.2726, +0.8011] | 0.3535 (0.3535) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0180 (0.0180) | OPEN |
| time-gap error over the scored steps (s) | +0.2390 / +0.2522 | +0.2456 | [+0.0767, +0.4349] | 0.2433 (0.2433) | WORSE |

## I10B0 R0C10XHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0516 / -0.0747 | -0.0632 | [-0.0977, -0.0308] | 0.0214 (0.0214) | BETTER |
| C contact rate | -0.0405 / -0.0374 | -0.0389 | [-0.0614, -0.0194] | 0.0125 (0.0125) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1511 / +0.1791 | +0.1651 | [+0.1215, +0.2139] | 0.0280 (0.0280) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0095 (0.0095) | OPEN |
| gap bias 4.0 s (m) | +1.3409 / +1.5946 | +1.4678 | [+1.2207, +1.7382] | 0.1385 (0.1385) | WORSE |
| gap bias 6.0 s (m) | +5.1910 / +6.2410 | +5.7160 | [+4.6498, +6.9298] | 0.9373 (0.9373) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0230 (0.0230) | OPEN |
| gap error 4.0 s (m) | +0.5314 / +0.6764 | +0.6039 | [+0.3704, +0.8402] | 0.1360 (0.1360) | WORSE |
| gap error 6.0 s (m) | +2.2039 / +2.9716 | +2.5877 | [+1.5457, +3.7505] | 0.3535 (0.3535) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0180 (0.0180) | OPEN |
| time-gap error over the scored steps (s) | +1.5846 / +1.7639 | +1.6743 | [+1.1307, +2.2340] | 0.2433 (0.2433) | WORSE |

## I10B0 R0G50Hb20 minus I10B0 R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0285 / -0.0391 | -0.0338 | [-0.0540, -0.0149] | 0.0214 (0.0214) | BETTER |
| C contact rate | -0.0202 / -0.0202 | -0.0202 | [-0.0352, -0.0082] | 0.0125 (0.0125) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0561 / +0.0888 | +0.0724 | [+0.0441, +0.1092] | 0.0280 (0.0280) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0095 (0.0095) | OPEN |
| gap bias 4.0 s (m) | +0.6755 / +0.8065 | +0.7410 | [+0.6149, +0.8796] | 0.1385 (0.1385) | WORSE |
| gap bias 6.0 s (m) | +2.5213 / +3.3152 | +2.9182 | [+2.2914, +3.6283] | 0.9373 (0.9373) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0230 (0.0230) | OPEN |
| gap error 4.0 s (m) | +0.2169 / +0.2442 | +0.2305 | [+0.1141, +0.3530] | 0.1360 (0.1360) | WORSE |
| gap error 6.0 s (m) | +0.8243 / +1.1466 | +0.9854 | [+0.4232, +1.5805] | 0.3535 (0.3535) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0180 (0.0180) | OPEN |
| time-gap error over the scored steps (s) | +0.8436 / +1.0007 | +0.9222 | [+0.5077, +1.3596] | 0.2433 (0.2433) | WORSE |

## I10B0 R0C10Hb20 minus I10B0 R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0249 / -0.0302 | -0.0276 | [-0.0440, -0.0132] | 0.0214 (0.0214) | BETTER |
| C contact rate | -0.0218 / -0.0171 | -0.0195 | [-0.0334, -0.0077] | 0.0125 (0.0125) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0950 / +0.1012 | +0.0981 | [+0.0698, +0.1293] | 0.0280 (0.0280) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0095 (0.0095) | OPEN |
| gap bias 4.0 s (m) | +0.6707 / +0.7715 | +0.7211 | [+0.6054, +0.8476] | 0.1385 (0.1385) | WORSE |
| gap bias 6.0 s (m) | +2.7210 / +3.1187 | +2.9198 | [+2.3181, +3.5892] | 0.9373 (0.9373) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0230 (0.0230) | OPEN |
| gap error 4.0 s (m) | +0.2994 / +0.3979 | +0.3486 | [+0.2279, +0.4805] | 0.1360 (0.1360) | WORSE |
| gap error 6.0 s (m) | +1.4038 / +2.0636 | +1.7337 | [+1.0930, +2.4937] | 0.3535 (0.3535) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0180 (0.0180) | OPEN |
| time-gap error over the scored steps (s) | +0.6962 / +0.6219 | +0.6590 | [+0.3831, +0.9571] | 0.2433 (0.2433) | WORSE |

## I10RH R0C10XHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0338 / -0.0569 | -0.0454 | [-0.0740, -0.0197] | 0.0214 (0.0214) | BETTER |
| C contact rate | -0.0234 / -0.0249 | -0.0241 | [-0.0414, -0.0098] | 0.0125 (0.0125) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0826 / +0.1153 | +0.0989 | [+0.0659, +0.1384] | 0.0280 (0.0280) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0303 / -0.0396 | -0.0350 | [-0.0569, -0.0141] | 0.0095 (0.0095) | BETTER |
| gap bias 4.0 s (m) | +1.0428 / +1.0941 | +1.0684 | [+0.9141, +1.2443] | 0.1385 (0.1385) | WORSE |
| gap bias 6.0 s (m) | +3.3633 / +4.0859 | +3.7246 | [+3.0555, +4.5192] | 0.9373 (0.9373) | WORSE |
| gap error 2.0 s (m) | +0.0316 / +0.0219 | +0.0268 | [+0.0107, +0.0452] | 0.0230 (0.0230) | WORSE |
| gap error 4.0 s (m) | +0.6025 / +0.5147 | +0.5586 | [+0.3634, +0.7568] | 0.1360 (0.1360) | WORSE |
| gap error 6.0 s (m) | +1.3253 / +1.5974 | +1.4613 | [+0.7943, +2.2286] | 0.3535 (0.3535) | WORSE |
| time-gap error 2.0 s (s) | +0.0272 / +0.0476 | +0.0374 | [-0.0087, +0.0888] | 0.0180 (0.0180) | OPEN |
| time-gap error over the scored steps (s) | +0.8836 / +1.0506 | +0.9671 | [+0.5903, +1.3846] | 0.2433 (0.2433) | WORSE |

## I10RH R0Hb20 minus I10B0 R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0214 / +0.0196 | +0.0205 | [+0.0087, +0.0339] | 0.0214 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0171 / +0.0140 | +0.0156 | [+0.0041, +0.0297] | 0.0125 (0.0125) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0343 / -0.0452 | -0.0397 | [-0.0612, -0.0194] | 0.0280 (0.0280) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0303 / -0.0396 | -0.0350 | [-0.0569, -0.0141] | 0.0095 (0.0095) | BETTER |
| gap bias 4.0 s (m) | -0.2301 / -0.3648 | -0.2974 | [-0.4409, -0.1366] | 0.1385 (0.1385) | BETTER |
| gap bias 6.0 s (m) | -1.4461 / -1.8101 | -1.6281 | [-2.1263, -1.1435] | 0.9373 (0.9373) | BETTER |
| gap error 2.0 s (m) | +0.0316 / +0.0219 | +0.0268 | [+0.0107, +0.0452] | 0.0230 (0.0230) | WORSE |
| gap error 4.0 s (m) | +0.1180 / -0.0405 | +0.0387 | [-0.0969, +0.1964] | 0.1360 (0.1360) | OPEN |
| gap error 6.0 s (m) | -0.1677 / -0.6865 | -0.4271 | [-0.9511, +0.1021] | 0.3535 (0.3535) | OPEN |
| time-gap error 2.0 s (s) | +0.0272 / +0.0476 | +0.0374 | [-0.0087, +0.0888] | 0.0180 (0.0180) | OPEN |
| time-gap error over the scored steps (s) | -0.5350 / -0.5642 | -0.5496 | [-0.8173, -0.2975] | 0.2433 (0.2433) | BETTER |

## I10RH R0C10Hb20 minus I10B0 R0C10Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0285 / +0.0214 | +0.0249 | [+0.0095, +0.0438] | 0.0214 (0.0214) | WORSE |
| C contact rate | +0.0234 / +0.0171 | +0.0202 | [+0.0088, +0.0339] | 0.0125 (0.0125) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0826 / -0.0794 | -0.0810 | [-0.1189, -0.0488] | 0.0280 (0.0280) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0303 / -0.0396 | -0.0350 | [-0.0569, -0.0141] | 0.0095 (0.0095) | BETTER |
| gap bias 4.0 s (m) | -0.4245 / -0.5997 | -0.5121 | [-0.7065, -0.3174] | 0.1385 (0.1385) | BETTER |
| gap bias 6.0 s (m) | -2.4126 / -2.7261 | -2.5693 | [-3.3765, -1.8574] | 0.9373 (0.9373) | BETTER |
| gap error 2.0 s (m) | +0.0316 / +0.0219 | +0.0268 | [+0.0107, +0.0452] | 0.0230 (0.0230) | WORSE |
| gap error 4.0 s (m) | +0.0378 / -0.1583 | -0.0602 | [-0.2443, +0.1437] | 0.1360 (0.1360) | OPEN |
| gap error 6.0 s (m) | -1.1627 / -1.7143 | -1.4385 | [-2.2648, -0.7079] | 0.3535 (0.3535) | BETTER |
| time-gap error 2.0 s (s) | +0.0272 / +0.0476 | +0.0374 | [-0.0087, +0.0888] | 0.0180 (0.0180) | OPEN |
| time-gap error over the scored steps (s) | -0.7785 / -0.5832 | -0.6809 | [-1.0490, -0.3926] | 0.2433 (0.2433) | BETTER |

## I10RH R0G50Hb20 minus I10B0 R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0107 / +0.0125 | +0.0116 | [+0.0022, +0.0231] | 0.0214 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0062 / +0.0093 | +0.0078 | [+0.0014, +0.0158] | 0.0125 (0.0125) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0140 / -0.0280 | -0.0210 | [-0.0432, -0.0031] | 0.0280 (0.0280) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0303 / -0.0396 | -0.0350 | [-0.0569, -0.0141] | 0.0095 (0.0095) | BETTER |
| gap bias 4.0 s (m) | -0.1309 / -0.3166 | -0.2238 | [-0.3821, -0.0635] | 0.1385 (0.1385) | BETTER |
| gap bias 6.0 s (m) | -0.7353 / -1.0974 | -0.9164 | [-1.2998, -0.5455] | 0.9373 (0.9373) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0316 / +0.0219 | +0.0268 | [+0.0107, +0.0452] | 0.0230 (0.0230) | WORSE |
| gap error 4.0 s (m) | +0.1070 / +0.0120 | +0.0595 | [-0.0396, +0.1736] | 0.1360 (0.1360) | OPEN |
| gap error 6.0 s (m) | -0.1825 / -0.2692 | -0.2258 | [-0.4896, +0.0373] | 0.3535 (0.3535) | OPEN |
| time-gap error 2.0 s (s) | +0.0272 / +0.0476 | +0.0374 | [-0.0087, +0.0888] | 0.0180 (0.0180) | OPEN |
| time-gap error over the scored steps (s) | -0.3214 / -0.2136 | -0.2675 | [-0.4924, -0.0806] | 0.2433 (0.2433) | BETTER |

## I10RH R0G0Hb20 minus I10B0 R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0018 / +0.0053 | +0.0036 | [-0.0026, +0.0122] | 0.0214 (0.0214) | OPEN |
| C contact rate | -0.0016 / +0.0062 | +0.0023 | [-0.0046, +0.0098] | 0.0125 (0.0125) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0125 / -0.0093 | -0.0109 | [-0.0222, -0.0004] | 0.0280 (0.0280) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0303 / -0.0396 | -0.0350 | [-0.0569, -0.0141] | 0.0095 (0.0095) | BETTER |
| gap bias 4.0 s (m) | -0.0496 / -0.1748 | -0.1122 | [-0.2207, -0.0018] | 0.1385 (0.1385) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.1521 / -0.3940 | -0.2731 | [-0.5020, -0.0590] | 0.9373 (0.9373) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0316 / +0.0219 | +0.0268 | [+0.0107, +0.0452] | 0.0230 (0.0230) | WORSE |
| gap error 4.0 s (m) | +0.0973 / +0.0676 | +0.0824 | [+0.0289, +0.1439] | 0.1360 (0.1360) | INSIDE-SPREAD |
| gap error 6.0 s (m) | -0.0100 / +0.0927 | +0.0413 | [-0.1033, +0.1860] | 0.3535 (0.3535) | OPEN |
| time-gap error 2.0 s (s) | +0.0272 / +0.0476 | +0.0374 | [-0.0087, +0.0888] | 0.0180 (0.0180) | OPEN |
| time-gap error over the scored steps (s) | +0.0062 / -0.0446 | -0.0192 | [-0.1149, +0.0520] | 0.2433 (0.2433) | OPEN |

## I10RH R0KHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0071 / -0.0018 | +0.0027 | [-0.0035, +0.0106] | 0.0214 (0.0214) | OPEN |
| C contact rate | -0.0016 / +0.0031 | +0.0008 | [-0.0048, +0.0064] | 0.0125 (0.0125) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0109 / -0.0062 | -0.0086 | [-0.0162, -0.0006] | 0.0280 (0.0280) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0303 / -0.0396 | -0.0350 | [-0.0569, -0.0141] | 0.0095 (0.0095) | BETTER |
| gap bias 4.0 s (m) | -0.0565 / -0.1507 | -0.1036 | [-0.2004, -0.0095] | 0.1385 (0.1385) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.1106 / -0.3363 | -0.2234 | [-0.4549, -0.0298] | 0.9373 (0.9373) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0316 / +0.0219 | +0.0268 | [+0.0107, +0.0452] | 0.0230 (0.0230) | WORSE |
| gap error 4.0 s (m) | +0.0904 / +0.0547 | +0.0725 | [+0.0193, +0.1348] | 0.1360 (0.1360) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.0379 / +0.0351 | +0.0365 | [-0.0700, +0.1447] | 0.3535 (0.3535) | OPEN |
| time-gap error 2.0 s (s) | +0.0272 / +0.0476 | +0.0374 | [-0.0087, +0.0888] | 0.0180 (0.0180) | OPEN |
| time-gap error over the scored steps (s) | +0.0146 / -0.0607 | -0.0231 | [-0.1343, +0.0585] | 0.2433 (0.2433) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10B0_R0C10Hb20 (0,1) | 0.0329 [0.0136, 0.0571] | 0.0195 [0.0060, 0.0360] | 0.2882 [0.2316, 0.3454] | 0.0891 [0.0465, 0.1353] | 2.1520 [1.6960, 2.6073] | 7.6474 [6.1154, 9.2291] | 0.4317 [0.3899, 0.4802] | 2.9594 [2.5948, 3.3607] | 8.7568 [7.5255, 10.0816] | 0.8992 [0.6263, 1.2159] | 3.7619 [2.7922, 4.7955] |
| I10B0_R0C10XHb20 (0,1) | 0.0294 [0.0104, 0.0534] | 0.0210 [0.0073, 0.0374] | 0.2796 [0.2236, 0.3374] | 0.0891 [0.0465, 0.1353] | 2.1882 [1.7296, 2.6381] | 7.4618 [6.0483, 8.9499] | 0.4317 [0.3899, 0.4802] | 2.9698 [2.6130, 3.3670] | 8.6467 [7.5117, 9.8733] | 0.8992 [0.6263, 1.2159] | 3.8517 [2.8854, 4.8691] |
| I10B0_R0G0Hb20 (0,1) | 0.0943 [0.0578, 0.1338] | 0.0592 [0.0296, 0.0938] | 0.1176 [0.0805, 0.1614] | 0.0891 [0.0465, 0.1353] | 0.6899 [0.3114, 1.0864] | 1.8093 [0.6138, 3.0721] | 0.4317 [0.3899, 0.4802] | 2.3802 [2.0930, 2.7083] | 6.0377 [5.2686, 6.8989] | 0.8992 [0.6263, 1.2159] | 2.1807 [1.5120, 2.9252] |
| I10B0_R0G50Hb20 (0,1) | 0.0605 [0.0351, 0.0901] | 0.0389 [0.0150, 0.0669] | 0.1900 [0.1424, 0.2401] | 0.0891 [0.0465, 0.1353] | 1.4309 [1.0198, 1.8362] | 4.7275 [3.3100, 6.1037] | 0.4317 [0.3899, 0.4802] | 2.6107 [2.2874, 2.9683] | 7.0231 [6.0954, 8.0773] | 0.8992 [0.6263, 1.2159] | 3.1028 [2.2323, 4.0349] |
| I10B0_R0Hb20 (0,1) | 0.0614 [0.0289, 0.0956] | 0.0280 [0.0116, 0.0479] | 0.2033 [0.1503, 0.2595] | 0.0891 [0.0465, 0.1353] | 1.4575 [0.9911, 1.9165] | 5.2877 [3.8033, 6.8244] | 0.4317 [0.3899, 0.4802] | 2.6569 [2.3239, 3.0194] | 7.3768 [6.2706, 8.5258] | 0.8992 [0.6263, 1.2159] | 3.1310 [2.2928, 4.0200] |
| I10B0_R0KHb20 (0,1) | 0.0925 [0.0563, 0.1324] | 0.0600 [0.0297, 0.0927] | 0.1145 [0.0783, 0.1567] | 0.0891 [0.0465, 0.1353] | 0.7204 [0.3444, 1.1110] | 1.7458 [0.5279, 3.0407] | 0.4317 [0.3899, 0.4802] | 2.3660 [2.0834, 2.6885] | 6.0590 [5.2991, 6.9059] | 0.8992 [0.6263, 1.2159] | 2.1774 [1.5055, 2.9188] |
| I10RH_R0C10Hb20 (0,1) | 0.0578 [0.0301, 0.0937] | 0.0397 [0.0163, 0.0674] | 0.2072 [0.1557, 0.2620] | 0.0541 [-0.0003, 0.1137] | 1.6399 [1.1962, 2.1063] | 5.0781 [3.7244, 6.4696] | 0.4584 [0.4066, 0.5113] | 2.8991 [2.5245, 3.3097] | 7.3183 [6.4004, 8.3502] | 0.9366 [0.6450, 1.2757] | 3.0810 [2.2501, 3.9539] |
| I10RH_R0C10XHb20 (0,1) | 0.0472 [0.0221, 0.0795] | 0.0358 [0.0137, 0.0613] | 0.2134 [0.1621, 0.2687] | 0.0541 [-0.0003, 0.1137] | 1.7888 [1.3557, 2.2578] | 5.4704 [4.1477, 6.8065] | 0.4584 [0.4066, 0.5113] | 2.9245 [2.5330, 3.3539] | 7.5203 [6.6391, 8.5576] | 0.9366 [0.6450, 1.2757] | 3.1445 [2.3022, 4.0131] |
| I10RH_R0G0Hb20 (0,1) | 0.0979 [0.0581, 0.1409] | 0.0615 [0.0315, 0.0942] | 0.1067 [0.0707, 0.1491] | 0.0541 [-0.0003, 0.1137] | 0.5777 [0.1723, 1.0290] | 1.5362 [0.3418, 2.8583] | 0.4584 [0.4066, 0.5113] | 2.4626 [2.1459, 2.8106] | 6.0790 [5.2989, 6.9440] | 0.9366 [0.6450, 1.2757] | 2.1615 [1.5042, 2.9057] |
| I10RH_R0G50Hb20 (0,1) | 0.0721 [0.0421, 0.1055] | 0.0467 [0.0213, 0.0761] | 0.1690 [0.1231, 0.2158] | 0.0541 [-0.0003, 0.1137] | 1.2072 [0.7873, 1.6766] | 3.8112 [2.4776, 5.1570] | 0.4584 [0.4066, 0.5113] | 2.6702 [2.3254, 3.0567] | 6.7973 [5.9519, 7.7804] | 0.9366 [0.6450, 1.2757] | 2.8354 [2.0471, 3.6798] |
| I10RH_R0Hb20 (0,1) | 0.0819 [0.0438, 0.1239] | 0.0436 [0.0206, 0.0728] | 0.1636 [0.1163, 0.2142] | 0.0541 [-0.0003, 0.1137] | 1.1601 [0.7080, 1.6234] | 3.6596 [2.2695, 5.1805] | 0.4584 [0.4066, 0.5113] | 2.6956 [2.3561, 3.0709] | 6.9497 [6.0649, 7.9524] | 0.9366 [0.6450, 1.2757] | 2.5814 [1.8594, 3.3407] |
| I10RH_R0KHb20 (0,1) | 0.0952 [0.0569, 0.1374] | 0.0607 [0.0306, 0.0944] | 0.1059 [0.0714, 0.1459] | 0.0541 [-0.0003, 0.1137] | 0.6168 [0.2197, 1.0551] | 1.5224 [0.2789, 2.8409] | 0.4584 [0.4066, 0.5113] | 2.4385 [2.1253, 2.7781] | 6.0955 [5.3116, 6.9523] | 0.9366 [0.6450, 1.2757] | 2.1543 [1.4999, 2.8825] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
