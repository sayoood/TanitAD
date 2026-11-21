# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: F:/TanitAD-scratch/strat1/read/inputs/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 28, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.01779359430604982, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHK_R0C10XHb20:I10B0_R0C10XHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHK_R0C10XHb20:I10RHK_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHK_R0G50Hb20:I10RHK_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHK_R0C10Hb20:I10RHK_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0C10XHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0G50Hb20:I10B0_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0C10Hb20:I10B0_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHK_R0C10XHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHK_R0Hb20:I10B0_R0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHK_R0C10Hb20:I10B0_R0C10Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHK_R0G50Hb20:I10B0_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHK_R0G0Hb20:I10B0_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHK_R0KHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHK_R0Hb20:I10RH_R0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHK_R0C10XHb20:I10RH_R0C10XHb20", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10RHK_R0Hb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10RHK_R0C10XHb20 minus I10B0_R0C10XHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHK_R0C10XHb20 minus I10RHK_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHK_R0G50Hb20 minus I10RHK_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHK_R0C10Hb20 minus I10RHK_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0C10XHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0G50Hb20 minus I10B0_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0C10Hb20 minus I10B0_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHK_R0C10XHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHK_R0Hb20 minus I10B0_R0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHK_R0C10Hb20 minus I10B0_R0C10Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHK_R0G50Hb20 minus I10B0_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHK_R0G0Hb20 minus I10B0_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHK_R0KHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHK_R0Hb20 minus I10RH_R0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHK_R0C10XHb20 minus I10RH_R0C10XHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10RHK R0C10XHb20 minus I10B0 R0C10XHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0036 / +0.0000 | -0.0018 | [-0.0087, +0.0044] | 0.0178 (0.0178) | OPEN |
| C contact rate | -0.0016 / -0.0047 | -0.0031 | [-0.0074, +0.0000] | 0.0016 (0.0016) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0078 / +0.0125 | +0.0023 | [-0.0180, +0.0208] | 0.0514 (0.0514) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0166 / +0.0117 | +0.0142 | [+0.0025, +0.0248] | 0.0052 (0.0052) | WORSE |
| gap bias 4.0 s (m) | +0.1713 / +0.0937 | +0.1325 | [+0.0306, +0.2257] | 0.2465 (0.2465) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.1786 / +0.2881 | +0.2334 | [-0.0467, +0.5004] | 1.1853 (1.1853) | OPEN |
| gap error 2.0 s (m) | +0.0159 / +0.0156 | +0.0158 | [+0.0087, +0.0228] | 0.0136 (0.0136) | WORSE |
| gap error 4.0 s (m) | +0.1194 / +0.0979 | +0.1087 | [+0.0317, +0.1824] | 0.0299 (0.0299) | WORSE |
| gap error 6.0 s (m) | +0.0675 / +0.1832 | +0.1253 | [-0.1255, +0.3636] | 0.8591 (0.8591) | OPEN |
| time-gap error 2.0 s (s) | +0.0427 / +0.0444 | +0.0436 | [+0.0182, +0.0685] | 0.0367 (0.0367) | WORSE |
| time-gap error over the scored steps (s) | +0.1122 / -0.0314 | +0.0404 | [-0.2008, +0.2509] | 0.1422 (0.1422) | OPEN |

## I10RHK R0C10XHb20 minus I10RHK R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0552 / -0.0730 | -0.0641 | [-0.1009, -0.0298] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0421 / -0.0421 | -0.0421 | [-0.0658, -0.0214] | 0.0016 (0.0016) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1511 / +0.1838 | +0.1674 | [+0.1225, +0.2165] | 0.0514 (0.0514) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0052 (0.0052) | OPEN |
| gap bias 4.0 s (m) | +1.4297 / +1.6354 | +1.5326 | [+1.2947, +1.7894] | 0.2465 (0.2465) | WORSE |
| gap bias 6.0 s (m) | +5.2756 / +6.3938 | +5.8347 | [+4.7577, +7.1093] | 1.1853 (1.1853) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.5936 / +0.6950 | +0.6443 | [+0.4065, +0.8866] | 0.0299 (0.0299) | WORSE |
| gap error 6.0 s (m) | +2.2825 / +3.0822 | +2.6823 | [+1.6743, +3.8745] | 0.8591 (0.8591) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0367 (0.0367) | OPEN |
| time-gap error over the scored steps (s) | +1.7431 / +1.8408 | +1.7919 | [+1.1858, +2.4464] | 0.1422 (0.1422) | WORSE |

## I10RHK R0G50Hb20 minus I10RHK R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0302 / -0.0463 | -0.0383 | [-0.0640, -0.0157] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0249 / -0.0202 | -0.0226 | [-0.0383, -0.0094] | 0.0016 (0.0016) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0748 / +0.1028 | +0.0888 | [+0.0577, +0.1287] | 0.0514 (0.0514) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0052 (0.0052) | OPEN |
| gap bias 4.0 s (m) | +0.7403 / +0.8360 | +0.7881 | [+0.6702, +0.9168] | 0.2465 (0.2465) | WORSE |
| gap bias 6.0 s (m) | +2.7589 / +3.4356 | +3.0973 | [+2.5109, +3.7771] | 1.1853 (1.1853) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.2571 / +0.2685 | +0.2628 | [+0.1322, +0.3928] | 0.0299 (0.0299) | WORSE |
| gap error 6.0 s (m) | +1.0038 / +1.1887 | +1.0962 | [+0.5236, +1.7008] | 0.8591 (0.8591) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0367 (0.0367) | OPEN |
| time-gap error over the scored steps (s) | +0.9587 / +1.1071 | +1.0329 | [+0.5770, +1.5245] | 0.1422 (0.1422) | WORSE |

## I10RHK R0C10Hb20 minus I10RHK R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0231 / -0.0285 | -0.0258 | [-0.0427, -0.0115] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0187 / -0.0187 | -0.0187 | [-0.0327, -0.0069] | 0.0016 (0.0016) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0826 / +0.0826 | +0.0826 | [+0.0583, +0.1111] | 0.0514 (0.0514) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0052 (0.0052) | OPEN |
| gap bias 4.0 s (m) | +0.7001 / +0.8021 | +0.7511 | [+0.6328, +0.8811] | 0.2465 (0.2465) | WORSE |
| gap bias 6.0 s (m) | +2.7037 / +3.1541 | +2.9289 | [+2.3263, +3.6070] | 1.1853 (1.1853) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.3211 / +0.4006 | +0.3608 | [+0.2392, +0.4980] | 0.0299 (0.0299) | WORSE |
| gap error 6.0 s (m) | +1.4334 / +2.0798 | +1.7566 | [+1.1505, +2.5384] | 0.8591 (0.8591) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0367 (0.0367) | OPEN |
| time-gap error over the scored steps (s) | +0.6314 / +0.6075 | +0.6194 | [+0.3376, +0.9244] | 0.1422 (0.1422) | WORSE |

## I10B0 R0C10XHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0516 / -0.0747 | -0.0632 | [-0.0977, -0.0308] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0405 / -0.0374 | -0.0389 | [-0.0614, -0.0194] | 0.0016 (0.0016) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1511 / +0.1791 | +0.1651 | [+0.1215, +0.2139] | 0.0514 (0.0514) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0052 (0.0052) | OPEN |
| gap bias 4.0 s (m) | +1.3409 / +1.5946 | +1.4678 | [+1.2207, +1.7382] | 0.2465 (0.2465) | WORSE |
| gap bias 6.0 s (m) | +5.1910 / +6.2410 | +5.7160 | [+4.6498, +6.9298] | 1.1853 (1.1853) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.5314 / +0.6764 | +0.6039 | [+0.3704, +0.8402] | 0.0299 (0.0299) | WORSE |
| gap error 6.0 s (m) | +2.2039 / +2.9716 | +2.5877 | [+1.5457, +3.7505] | 0.8591 (0.8591) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0367 (0.0367) | OPEN |
| time-gap error over the scored steps (s) | +1.5846 / +1.7639 | +1.6743 | [+1.1307, +2.2340] | 0.1422 (0.1422) | WORSE |

## I10B0 R0G50Hb20 minus I10B0 R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0285 / -0.0391 | -0.0338 | [-0.0540, -0.0149] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0202 / -0.0202 | -0.0202 | [-0.0352, -0.0082] | 0.0016 (0.0016) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0561 / +0.0888 | +0.0724 | [+0.0441, +0.1092] | 0.0514 (0.0514) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0052 (0.0052) | OPEN |
| gap bias 4.0 s (m) | +0.6755 / +0.8065 | +0.7410 | [+0.6149, +0.8796] | 0.2465 (0.2465) | WORSE |
| gap bias 6.0 s (m) | +2.5213 / +3.3152 | +2.9182 | [+2.2914, +3.6283] | 1.1853 (1.1853) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.2169 / +0.2442 | +0.2305 | [+0.1141, +0.3530] | 0.0299 (0.0299) | WORSE |
| gap error 6.0 s (m) | +0.8243 / +1.1466 | +0.9854 | [+0.4232, +1.5805] | 0.8591 (0.8591) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0367 (0.0367) | OPEN |
| time-gap error over the scored steps (s) | +0.8436 / +1.0007 | +0.9222 | [+0.5077, +1.3596] | 0.1422 (0.1422) | WORSE |

## I10B0 R0C10Hb20 minus I10B0 R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0249 / -0.0302 | -0.0276 | [-0.0440, -0.0132] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0218 / -0.0171 | -0.0195 | [-0.0334, -0.0077] | 0.0016 (0.0016) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0950 / +0.1012 | +0.0981 | [+0.0698, +0.1293] | 0.0514 (0.0514) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0052 (0.0052) | OPEN |
| gap bias 4.0 s (m) | +0.6707 / +0.7715 | +0.7211 | [+0.6054, +0.8476] | 0.2465 (0.2465) | WORSE |
| gap bias 6.0 s (m) | +2.7210 / +3.1187 | +2.9198 | [+2.3181, +3.5892] | 1.1853 (1.1853) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.2994 / +0.3979 | +0.3486 | [+0.2279, +0.4805] | 0.0299 (0.0299) | WORSE |
| gap error 6.0 s (m) | +1.4038 / +2.0636 | +1.7337 | [+1.0930, +2.4937] | 0.8591 (0.8591) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0367 (0.0367) | OPEN |
| time-gap error over the scored steps (s) | +0.6962 / +0.6219 | +0.6590 | [+0.3831, +0.9571] | 0.1422 (0.1422) | WORSE |

## I10RHK R0C10XHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0552 / -0.0747 | -0.0649 | [-0.0999, -0.0317] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0421 / -0.0421 | -0.0421 | [-0.0651, -0.0216] | 0.0016 (0.0016) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1433 / +0.1916 | +0.1674 | [+0.1180, +0.2243] | 0.0514 (0.0514) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0166 / +0.0117 | +0.0142 | [+0.0025, +0.0248] | 0.0052 (0.0052) | WORSE |
| gap bias 4.0 s (m) | +1.5122 / +1.6883 | +1.6003 | [+1.3544, +1.8702] | 0.2465 (0.2465) | WORSE |
| gap bias 6.0 s (m) | +5.3696 / +6.5292 | +5.9494 | [+4.8229, +7.2802] | 1.1853 (1.1853) | WORSE |
| gap error 2.0 s (m) | +0.0159 / +0.0156 | +0.0158 | [+0.0087, +0.0228] | 0.0136 (0.0136) | WORSE |
| gap error 4.0 s (m) | +0.6507 / +0.7743 | +0.7125 | [+0.4527, +0.9867] | 0.0299 (0.0299) | WORSE |
| gap error 6.0 s (m) | +2.2713 / +3.1548 | +2.7131 | [+1.6408, +4.0122] | 0.8591 (0.8591) | WORSE |
| time-gap error 2.0 s (s) | +0.0427 / +0.0444 | +0.0436 | [+0.0182, +0.0685] | 0.0367 (0.0367) | WORSE |
| time-gap error over the scored steps (s) | +1.6969 / +1.7325 | +1.7147 | [+1.1194, +2.3521] | 0.1422 (0.1422) | WORSE |

## I10RHK R0Hb20 minus I10B0 R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0018 / +0.0000 | -0.0009 | [-0.0095, +0.0066] | 0.0178 (0.0178) | OPEN |
| C contact rate | -0.0062 / +0.0016 | -0.0023 | [-0.0089, +0.0043] | 0.0016 (0.0016) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0047 / +0.0171 | +0.0109 | [-0.0024, +0.0260] | 0.0514 (0.0514) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0166 / +0.0117 | +0.0142 | [+0.0025, +0.0248] | 0.0052 (0.0052) | WORSE |
| gap bias 4.0 s (m) | +0.1655 / +0.1389 | +0.1522 | [+0.0724, +0.2324] | 0.2465 (0.2465) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.5150 / +0.3991 | +0.4571 | [+0.1924, +0.7409] | 1.1853 (1.1853) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0159 / +0.0156 | +0.0158 | [+0.0087, +0.0228] | 0.0136 (0.0136) | WORSE |
| gap error 4.0 s (m) | +0.1270 / +0.1344 | +0.1307 | [+0.0705, +0.1966] | 0.0299 (0.0299) | WORSE |
| gap error 6.0 s (m) | +0.3494 / +0.3363 | +0.3429 | [+0.1417, +0.5628] | 0.8591 (0.8591) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0427 / +0.0444 | +0.0436 | [+0.0182, +0.0685] | 0.0367 (0.0367) | WORSE |
| time-gap error over the scored steps (s) | +0.1845 / +0.0541 | +0.1193 | [-0.0556, +0.2626] | 0.1422 (0.1422) | OPEN |

## I10RHK R0C10Hb20 minus I10B0 R0C10Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0018 / -0.0018 | -0.0018 | [-0.0071, +0.0031] | 0.0178 (0.0178) | OPEN |
| C contact rate | -0.0031 / -0.0016 | -0.0023 | [-0.0056, +0.0000] | 0.0016 (0.0016) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0000 / +0.0062 | +0.0031 | [-0.0113, +0.0158] | 0.0514 (0.0514) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0166 / +0.0117 | +0.0142 | [+0.0025, +0.0248] | 0.0052 (0.0052) | WORSE |
| gap bias 4.0 s (m) | +0.1813 / +0.1318 | +0.1566 | [+0.0632, +0.2399] | 0.2465 (0.2465) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.3344 / +0.3358 | +0.3351 | [+0.0764, +0.5888] | 1.1853 (1.1853) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0159 / +0.0156 | +0.0158 | [+0.0087, +0.0228] | 0.0136 (0.0136) | WORSE |
| gap error 4.0 s (m) | +0.1382 / +0.1217 | +0.1299 | [+0.0542, +0.2052] | 0.0299 (0.0299) | WORSE |
| gap error 6.0 s (m) | +0.2437 / +0.2408 | +0.2423 | [+0.0276, +0.4597] | 0.8591 (0.8591) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0427 / +0.0444 | +0.0436 | [+0.0182, +0.0685] | 0.0367 (0.0367) | WORSE |
| time-gap error over the scored steps (s) | +0.0439 / +0.0193 | +0.0316 | [-0.1677, +0.2525] | 0.1422 (0.1422) | OPEN |

## I10RHK R0G50Hb20 minus I10B0 R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0036 / -0.0036 | -0.0036 | [-0.0085, +0.0012] | 0.0178 (0.0178) | OPEN |
| C contact rate | -0.0062 / +0.0000 | -0.0031 | [-0.0086, +0.0000] | 0.0016 (0.0016) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0125 / +0.0249 | +0.0187 | [+0.0058, +0.0356] | 0.0514 (0.0514) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0166 / +0.0117 | +0.0142 | [+0.0025, +0.0248] | 0.0052 (0.0052) | WORSE |
| gap bias 4.0 s (m) | +0.1520 / +0.1012 | +0.1266 | [+0.0474, +0.2008] | 0.2465 (0.2465) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.3518 / +0.3003 | +0.3261 | [+0.1414, +0.5162] | 1.1853 (1.1853) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0159 / +0.0156 | +0.0158 | [+0.0087, +0.0228] | 0.0136 (0.0136) | WORSE |
| gap error 4.0 s (m) | +0.1165 / +0.1190 | +0.1177 | [+0.0626, +0.1777] | 0.0299 (0.0299) | WORSE |
| gap error 6.0 s (m) | +0.2142 / +0.2247 | +0.2194 | [+0.0573, +0.3876] | 0.8591 (0.8591) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0427 / +0.0444 | +0.0436 | [+0.0182, +0.0685] | 0.0367 (0.0367) | WORSE |
| time-gap error over the scored steps (s) | +0.1087 / +0.0337 | +0.0712 | [-0.0925, +0.2193] | 0.1422 (0.1422) | OPEN |

## I10RHK R0G0Hb20 minus I10B0 R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0018 / +0.0036 | +0.0009 | [-0.0069, +0.0101] | 0.0178 (0.0178) | OPEN |
| C contact rate | -0.0016 / +0.0000 | -0.0008 | [-0.0046, +0.0027] | 0.0016 (0.0016) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0062 / +0.0109 | +0.0023 | [-0.0102, +0.0196] | 0.0514 (0.0514) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0166 / +0.0117 | +0.0142 | [+0.0025, +0.0248] | 0.0052 (0.0052) | WORSE |
| gap bias 4.0 s (m) | +0.0872 / +0.0718 | +0.0795 | [+0.0228, +0.1336] | 0.2465 (0.2465) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.1142 / +0.1799 | +0.1470 | [-0.0028, +0.3165] | 1.1853 (1.1853) | OPEN |
| gap error 2.0 s (m) | +0.0159 / +0.0156 | +0.0158 | [+0.0087, +0.0228] | 0.0136 (0.0136) | WORSE |
| gap error 4.0 s (m) | +0.0762 / +0.0947 | +0.0854 | [+0.0470, +0.1308] | 0.0299 (0.0299) | WORSE |
| gap error 6.0 s (m) | +0.0346 / +0.1826 | +0.1086 | [-0.0450, +0.3076] | 0.8591 (0.8591) | OPEN |
| time-gap error 2.0 s (s) | +0.0427 / +0.0444 | +0.0436 | [+0.0182, +0.0685] | 0.0367 (0.0367) | WORSE |
| time-gap error over the scored steps (s) | -0.0064 / -0.0727 | -0.0395 | [-0.1746, +0.0550] | 0.1422 (0.1422) | OPEN |

## I10RHK R0KHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / -0.0018 | -0.0009 | [-0.0049, +0.0046] | 0.0178 (0.0178) | OPEN |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [-0.0029, +0.0031] | 0.0016 (0.0016) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0078 / +0.0078 | +0.0000 | [-0.0118, +0.0135] | 0.0514 (0.0514) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0166 / +0.0117 | +0.0142 | [+0.0025, +0.0248] | 0.0052 (0.0052) | WORSE |
| gap bias 4.0 s (m) | +0.0825 / +0.0529 | +0.0677 | [+0.0108, +0.1242] | 0.2465 (0.2465) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.0940 / +0.1354 | +0.1147 | [-0.0288, +0.2698] | 1.1853 (1.1853) | OPEN |
| gap error 2.0 s (m) | +0.0159 / +0.0156 | +0.0158 | [+0.0087, +0.0228] | 0.0136 (0.0136) | WORSE |
| gap error 4.0 s (m) | +0.0572 / +0.0793 | +0.0682 | [+0.0290, +0.1134] | 0.0299 (0.0299) | WORSE |
| gap error 6.0 s (m) | -0.0112 / +0.0726 | +0.0307 | [-0.1045, +0.1798] | 0.8591 (0.8591) | OPEN |
| time-gap error 2.0 s (s) | +0.0427 / +0.0444 | +0.0436 | [+0.0182, +0.0685] | 0.0367 (0.0367) | WORSE |
| time-gap error over the scored steps (s) | -0.0462 / -0.1082 | -0.0772 | [-0.1997, +0.0210] | 0.1422 (0.1422) | OPEN |

## I10RHK R0Hb20 minus I10RH R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0231 / -0.0196 | -0.0214 | [-0.0350, -0.0096] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0234 / -0.0125 | -0.0179 | [-0.0349, -0.0061] | 0.0016 (0.0016) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0389 / +0.0623 | +0.0506 | [+0.0239, +0.0784] | 0.0514 (0.0514) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0470 / +0.0513 | +0.0491 | [+0.0317, +0.0667] | 0.0052 (0.0052) | WORSE |
| gap bias 4.0 s (m) | +0.3956 / +0.5037 | +0.4496 | [+0.2912, +0.5844] | 0.2465 (0.2465) | WORSE |
| gap bias 6.0 s (m) | +1.9612 / +2.2092 | +2.0852 | [+1.5845, +2.6152] | 1.1853 (1.1853) | WORSE |
| gap error 2.0 s (m) | -0.0157 / -0.0063 | -0.0110 | [-0.0282, +0.0049] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.0090 / +0.1749 | +0.0920 | [-0.0921, +0.2425] | 0.0299 (0.0299) | OPEN |
| gap error 6.0 s (m) | +0.5171 / +1.0228 | +0.7700 | [+0.1698, +1.3579] | 0.8591 (0.8591) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0156 / -0.0032 | +0.0062 | [-0.0370, +0.0550] | 0.0367 (0.0367) | OPEN |
| time-gap error over the scored steps (s) | +0.7195 / +0.6184 | +0.6689 | [+0.3997, +1.0000] | 0.1422 (0.1422) | WORSE |

## I10RHK R0C10XHb20 minus I10RH R0C10XHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0214 / -0.0178 | -0.0196 | [-0.0339, -0.0070] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0187 / -0.0171 | -0.0179 | [-0.0320, -0.0059] | 0.0016 (0.0016) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0607 / +0.0763 | +0.0685 | [+0.0379, +0.1006] | 0.0514 (0.0514) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0470 / +0.0513 | +0.0491 | [+0.0317, +0.0667] | 0.0052 (0.0052) | WORSE |
| gap bias 4.0 s (m) | +0.4695 / +0.5942 | +0.5319 | [+0.3723, +0.6899] | 0.2465 (0.2465) | WORSE |
| gap bias 6.0 s (m) | +2.0063 / +2.4433 | +2.2248 | [+1.6099, +2.9525] | 1.1853 (1.1853) | WORSE |
| gap error 2.0 s (m) | -0.0157 / -0.0063 | -0.0110 | [-0.0282, +0.0049] | 0.0136 (0.0136) | OPEN |
| gap error 4.0 s (m) | +0.0483 / +0.2596 | +0.1540 | [-0.0464, +0.3445] | 0.0299 (0.0299) | OPEN |
| gap error 6.0 s (m) | +0.9461 / +1.5574 | +1.2517 | [+0.5890, +2.0403] | 0.8591 (0.8591) | WORSE |
| time-gap error 2.0 s (s) | +0.0156 / -0.0032 | +0.0062 | [-0.0370, +0.0550] | 0.0367 (0.0367) | OPEN |
| time-gap error over the scored steps (s) | +0.8133 / +0.6820 | +0.7476 | [+0.4135, +1.1467] | 0.1422 (0.1422) | WORSE |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10B0_R0C10Hb20 (0,1) | 0.0329 [0.0136, 0.0571] | 0.0195 [0.0060, 0.0360] | 0.2882 [0.2316, 0.3454] | 0.0891 [0.0465, 0.1353] | 2.1520 [1.6960, 2.6073] | 7.6474 [6.1154, 9.2291] | 0.4317 [0.3899, 0.4802] | 2.9594 [2.5948, 3.3607] | 8.7568 [7.5255, 10.0816] | 0.8992 [0.6263, 1.2159] | 3.7619 [2.7922, 4.7955] |
| I10B0_R0C10XHb20 (0,1) | 0.0294 [0.0104, 0.0534] | 0.0210 [0.0073, 0.0374] | 0.2796 [0.2236, 0.3374] | 0.0891 [0.0465, 0.1353] | 2.1882 [1.7296, 2.6381] | 7.4618 [6.0483, 8.9499] | 0.4317 [0.3899, 0.4802] | 2.9698 [2.6130, 3.3670] | 8.6467 [7.5117, 9.8733] | 0.8992 [0.6263, 1.2159] | 3.8517 [2.8854, 4.8691] |
| I10B0_R0G0Hb20 (0,1) | 0.0943 [0.0578, 0.1338] | 0.0592 [0.0296, 0.0938] | 0.1176 [0.0805, 0.1614] | 0.0891 [0.0465, 0.1353] | 0.6899 [0.3114, 1.0864] | 1.8093 [0.6138, 3.0721] | 0.4317 [0.3899, 0.4802] | 2.3802 [2.0930, 2.7083] | 6.0377 [5.2686, 6.8989] | 0.8992 [0.6263, 1.2159] | 2.1807 [1.5120, 2.9252] |
| I10B0_R0G50Hb20 (0,1) | 0.0605 [0.0351, 0.0901] | 0.0389 [0.0150, 0.0669] | 0.1900 [0.1424, 0.2401] | 0.0891 [0.0465, 0.1353] | 1.4309 [1.0198, 1.8362] | 4.7275 [3.3100, 6.1037] | 0.4317 [0.3899, 0.4802] | 2.6107 [2.2874, 2.9683] | 7.0231 [6.0954, 8.0773] | 0.8992 [0.6263, 1.2159] | 3.1028 [2.2323, 4.0349] |
| I10B0_R0Hb20 (0,1) | 0.0614 [0.0289, 0.0956] | 0.0280 [0.0116, 0.0479] | 0.2033 [0.1503, 0.2595] | 0.0891 [0.0465, 0.1353] | 1.4575 [0.9911, 1.9165] | 5.2877 [3.8033, 6.8244] | 0.4317 [0.3899, 0.4802] | 2.6569 [2.3239, 3.0194] | 7.3768 [6.2706, 8.5258] | 0.8992 [0.6263, 1.2159] | 3.1310 [2.2928, 4.0200] |
| I10B0_R0KHb20 (0,1) | 0.0925 [0.0563, 0.1324] | 0.0600 [0.0297, 0.0927] | 0.1145 [0.0783, 0.1567] | 0.0891 [0.0465, 0.1353] | 0.7204 [0.3444, 1.1110] | 1.7458 [0.5279, 3.0407] | 0.4317 [0.3899, 0.4802] | 2.3660 [2.0834, 2.6885] | 6.0590 [5.2991, 6.9059] | 0.8992 [0.6263, 1.2159] | 2.1774 [1.5055, 2.9188] |
| I10RHK_R0C10Hb20 (0,1) | 0.0311 [0.0133, 0.0568] | 0.0171 [0.0043, 0.0326] | 0.2913 [0.2318, 0.3512] | 0.1033 [0.0540, 0.1551] | 2.3086 [1.8306, 2.7923] | 7.9825 [6.4418, 9.6343] | 0.4474 [0.4027, 0.4977] | 3.0893 [2.6873, 3.5139] | 8.9991 [7.7807, 10.4254] | 0.9428 [0.6578, 1.2755] | 3.7935 [2.8352, 4.8508] |
| I10RHK_R0C10XHb20 (0,1) | 0.0276 [0.0098, 0.0517] | 0.0179 [0.0054, 0.0341] | 0.2819 [0.2222, 0.3425] | 0.1033 [0.0540, 0.1551] | 2.3207 [1.8492, 2.7979] | 7.6952 [6.2293, 9.3032] | 0.4474 [0.4027, 0.4977] | 3.0785 [2.6798, 3.5057] | 8.7721 [7.6055, 10.1177] | 0.9428 [0.6578, 1.2755] | 3.8921 [2.9410, 4.9514] |
| I10RHK_R0G0Hb20 (0,1) | 0.0952 [0.0570, 0.1365] | 0.0584 [0.0291, 0.0918] | 0.1199 [0.0824, 0.1623] | 0.1033 [0.0540, 0.1551] | 0.7694 [0.3861, 1.1954] | 1.9563 [0.7422, 3.2604] | 0.4474 [0.4027, 0.4977] | 2.4656 [2.1679, 2.8038] | 6.1463 [5.3789, 7.0190] | 0.9428 [0.6578, 1.2755] | 2.1411 [1.5034, 2.8607] |
| I10RHK_R0G50Hb20 (0,1) | 0.0569 [0.0329, 0.0846] | 0.0358 [0.0135, 0.0624] | 0.2087 [0.1554, 0.2651] | 0.1033 [0.0540, 0.1551] | 1.5575 [1.1288, 1.9984] | 5.0536 [3.6199, 6.5073] | 0.4474 [0.4027, 0.4977] | 2.7285 [2.3811, 3.1023] | 7.2425 [6.2523, 8.3020] | 0.9428 [0.6578, 1.2755] | 3.1741 [2.3101, 4.1070] |
| I10RHK_R0Hb20 (0,1) | 0.0605 [0.0289, 0.0940] | 0.0257 [0.0107, 0.0431] | 0.2142 [0.1537, 0.2769] | 0.1033 [0.0540, 0.1551] | 1.6097 [1.1248, 2.1143] | 5.7448 [4.2621, 7.3256] | 0.4474 [0.4027, 0.4977] | 2.7876 [2.4263, 3.1678] | 7.7197 [6.6050, 8.9137] | 0.9428 [0.6578, 1.2755] | 3.2503 [2.4349, 4.1128] |
| I10RHK_R0KHb20 (0,1) | 0.0916 [0.0549, 0.1327] | 0.0600 [0.0300, 0.0931] | 0.1145 [0.0779, 0.1555] | 0.1033 [0.0540, 0.1551] | 0.7881 [0.4080, 1.2154] | 1.8605 [0.6486, 3.1658] | 0.4474 [0.4027, 0.4977] | 2.4342 [2.1410, 2.7676] | 6.0897 [5.3188, 6.9298] | 0.9428 [0.6578, 1.2755] | 2.1002 [1.4788, 2.7994] |
| I10RH_R0C10XHb20 (0,1) | 0.0472 [0.0221, 0.0795] | 0.0358 [0.0137, 0.0613] | 0.2134 [0.1621, 0.2687] | 0.0541 [-0.0003, 0.1137] | 1.7888 [1.3557, 2.2578] | 5.4704 [4.1477, 6.8065] | 0.4584 [0.4066, 0.5113] | 2.9245 [2.5330, 3.3539] | 7.5203 [6.6391, 8.5576] | 0.9366 [0.6450, 1.2757] | 3.1445 [2.3022, 4.0131] |
| I10RH_R0Hb20 (0,1) | 0.0819 [0.0438, 0.1239] | 0.0436 [0.0206, 0.0728] | 0.1636 [0.1163, 0.2142] | 0.0541 [-0.0003, 0.1137] | 1.1601 [0.7080, 1.6234] | 3.6596 [2.2695, 5.1805] | 0.4584 [0.4066, 0.5113] | 2.6956 [2.3561, 3.0709] | 6.9497 [6.0649, 7.9524] | 0.9366 [0.6450, 1.2757] | 2.5814 [1.8594, 3.3407] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
