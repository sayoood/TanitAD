# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: F:/TanitAD-scratch/strat1/read/inputs/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 28, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.0409252669039146, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHp3_R0C10XHb20:I10B0_R0C10XHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHp3_R0C10XHb20:I10RHp3_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHp3_R0G50Hb20:I10RHp3_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHp3_R0C10Hb20:I10RHp3_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0C10XHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0G50Hb20:I10B0_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0C10Hb20:I10B0_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHp3_R0C10XHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHp3_R0Hb20:I10B0_R0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHp3_R0C10Hb20:I10B0_R0C10Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHp3_R0G50Hb20:I10B0_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHp3_R0G0Hb20:I10B0_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHp3_R0KHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHp3_R0Hb20:I10RH_R0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RHp3_R0C10XHb20:I10RH_R0C10XHb20", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10RHp3_R0Hb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10RHp3_R0C10XHb20 minus I10B0_R0C10XHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHp3_R0C10XHb20 minus I10RHp3_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHp3_R0G50Hb20 minus I10RHp3_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHp3_R0C10Hb20 minus I10RHp3_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0C10XHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0G50Hb20 minus I10B0_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0C10Hb20 minus I10B0_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHp3_R0C10XHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHp3_R0Hb20 minus I10B0_R0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHp3_R0C10Hb20 minus I10B0_R0C10Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHp3_R0G50Hb20 minus I10B0_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHp3_R0G0Hb20 minus I10B0_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHp3_R0KHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHp3_R0Hb20 minus I10RH_R0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RHp3_R0C10XHb20 minus I10RH_R0C10XHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10RHp3 R0C10XHb20 minus I10B0 R0C10XHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0302 / +0.0391 | +0.0347 | [+0.0137, +0.0593] | 0.0409 (0.0409) | INSIDE-SPREAD |
| C contact rate | +0.0327 / +0.0265 | +0.0296 | [+0.0121, +0.0500] | 0.0156 (0.0156) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.1168 / -0.0950 | -0.1059 | [-0.1466, -0.0739] | 0.0389 (0.0389) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1136 / -0.0856 | -0.0996 | [-0.1525, -0.0531] | 0.0277 (0.0277) | BETTER |
| gap bias 4.0 s (m) | -0.8872 / -0.9224 | -0.9048 | [-1.1354, -0.6830] | 0.2144 (0.2144) | BETTER |
| gap bias 6.0 s (m) | -3.3949 / -3.5677 | -3.4813 | [-4.2534, -2.7549] | 0.9742 (0.9742) | BETTER |
| gap error 2.0 s (m) | +0.0908 / +0.0793 | +0.0850 | [+0.0526, +0.1222] | 0.0249 (0.0249) | WORSE |
| gap error 4.0 s (m) | -0.0445 / -0.0529 | -0.0487 | [-0.2278, +0.1410] | 0.0554 (0.0554) | OPEN |
| gap error 6.0 s (m) | -1.5723 / -1.6619 | -1.6171 | [-2.3018, -0.9601] | 0.5019 (0.5019) | BETTER |
| time-gap error 2.0 s (s) | +0.0785 / +0.1612 | +0.1198 | [+0.0248, +0.2747] | 0.0443 (0.0443) | WORSE |
| time-gap error over the scored steps (s) | -1.3093 / -0.9721 | -1.1407 | [-1.6908, -0.6936] | 0.3619 (0.3619) | BETTER |

## I10RHp3 R0C10XHb20 minus I10RHp3 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0302 / -0.0445 | -0.0374 | [-0.0594, -0.0187] | 0.0409 (0.0409) | INSIDE-SPREAD |
| C contact rate | -0.0156 / -0.0156 | -0.0156 | [-0.0274, -0.0058] | 0.0156 (0.0156) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0592 / +0.0826 | +0.0709 | [+0.0462, +0.1013] | 0.0389 (0.0389) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0277 (0.0277) | OPEN |
| gap bias 4.0 s (m) | +0.7471 / +0.9268 | +0.8369 | [+0.6936, +1.0228] | 0.2144 (0.2144) | WORSE |
| gap bias 6.0 s (m) | +2.4026 / +3.2288 | +2.8157 | [+2.2267, +3.5661] | 0.9742 (0.9742) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0249 (0.0249) | OPEN |
| gap error 4.0 s (m) | +0.3003 / +0.3821 | +0.3412 | [+0.1977, +0.5152] | 0.0554 (0.0554) | WORSE |
| gap error 6.0 s (m) | +0.6261 / +1.0618 | +0.8440 | [+0.3107, +1.4829] | 0.5019 (0.5019) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0443 (0.0443) | OPEN |
| time-gap error over the scored steps (s) | +0.4106 / +0.8378 | +0.6242 | [+0.2892, +1.0476] | 0.3619 (0.3619) | WORSE |

## I10RHp3 R0G50Hb20 minus I10RHp3 R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0125 / -0.0196 | -0.0160 | [-0.0252, -0.0072] | 0.0409 (0.0409) | INSIDE-SPREAD |
| C contact rate | -0.0125 / -0.0062 | -0.0093 | [-0.0200, -0.0015] | 0.0156 (0.0156) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0312 / +0.0467 | +0.0389 | [+0.0217, +0.0591] | 0.0389 (0.0389) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0277 (0.0277) | OPEN |
| gap bias 4.0 s (m) | +0.4545 / +0.5246 | +0.4895 | [+0.4104, +0.5853] | 0.2144 (0.2144) | WORSE |
| gap bias 6.0 s (m) | +1.3707 / +1.9057 | +1.6382 | [+1.2402, +2.1195] | 0.9742 (0.9742) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0249 (0.0249) | OPEN |
| gap error 4.0 s (m) | +0.1565 / +0.1665 | +0.1615 | [+0.0758, +0.2523] | 0.0554 (0.0554) | WORSE |
| gap error 6.0 s (m) | +0.3380 / +0.5409 | +0.4395 | [+0.1093, +0.8437] | 0.5019 (0.5019) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0443 (0.0443) | OPEN |
| time-gap error over the scored steps (s) | +0.1755 / +0.6469 | +0.4112 | [+0.0988, +0.8253] | 0.3619 (0.3619) | WORSE |

## I10RHp3 R0C10Hb20 minus I10RHp3 R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0036 / -0.0196 | -0.0116 | [-0.0262, -0.0002] | 0.0409 (0.0409) | INSIDE-SPREAD |
| C contact rate | +0.0000 / -0.0031 | -0.0016 | [-0.0073, +0.0039] | 0.0156 (0.0156) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0047 / +0.0218 | +0.0132 | [-0.0034, +0.0321] | 0.0389 (0.0389) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0277 (0.0277) | OPEN |
| gap bias 4.0 s (m) | +0.0890 / +0.2530 | +0.1710 | [+0.0434, +0.2934] | 0.2144 (0.2144) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.2622 / +0.7199 | +0.4911 | [+0.1149, +0.8607] | 0.9742 (0.9742) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0249 (0.0249) | OPEN |
| gap error 4.0 s (m) | +0.1073 / +0.1753 | +0.1413 | [+0.0689, +0.2158] | 0.0554 (0.0554) | WORSE |
| gap error 6.0 s (m) | +0.1353 / +0.3806 | +0.2580 | [+0.0195, +0.4883] | 0.5019 (0.5019) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0443 (0.0443) | OPEN |
| time-gap error over the scored steps (s) | +0.1974 / +0.0538 | +0.1256 | [-0.0575, +0.2666] | 0.3619 (0.3619) | OPEN |

## I10B0 R0C10XHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0516 / -0.0747 | -0.0632 | [-0.0977, -0.0308] | 0.0409 (0.0409) | BETTER |
| C contact rate | -0.0405 / -0.0374 | -0.0389 | [-0.0614, -0.0194] | 0.0156 (0.0156) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1511 / +0.1791 | +0.1651 | [+0.1215, +0.2139] | 0.0389 (0.0389) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0277 (0.0277) | OPEN |
| gap bias 4.0 s (m) | +1.3409 / +1.5946 | +1.4678 | [+1.2207, +1.7382] | 0.2144 (0.2144) | WORSE |
| gap bias 6.0 s (m) | +5.1910 / +6.2410 | +5.7160 | [+4.6498, +6.9298] | 0.9742 (0.9742) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0249 (0.0249) | OPEN |
| gap error 4.0 s (m) | +0.5314 / +0.6764 | +0.6039 | [+0.3704, +0.8402] | 0.0554 (0.0554) | WORSE |
| gap error 6.0 s (m) | +2.2039 / +2.9716 | +2.5877 | [+1.5457, +3.7505] | 0.5019 (0.5019) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0443 (0.0443) | OPEN |
| time-gap error over the scored steps (s) | +1.5846 / +1.7639 | +1.6743 | [+1.1307, +2.2340] | 0.3619 (0.3619) | WORSE |

## I10B0 R0G50Hb20 minus I10B0 R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0285 / -0.0391 | -0.0338 | [-0.0540, -0.0149] | 0.0409 (0.0409) | INSIDE-SPREAD |
| C contact rate | -0.0202 / -0.0202 | -0.0202 | [-0.0352, -0.0082] | 0.0156 (0.0156) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0561 / +0.0888 | +0.0724 | [+0.0441, +0.1092] | 0.0389 (0.0389) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0277 (0.0277) | OPEN |
| gap bias 4.0 s (m) | +0.6755 / +0.8065 | +0.7410 | [+0.6149, +0.8796] | 0.2144 (0.2144) | WORSE |
| gap bias 6.0 s (m) | +2.5213 / +3.3152 | +2.9182 | [+2.2914, +3.6283] | 0.9742 (0.9742) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0249 (0.0249) | OPEN |
| gap error 4.0 s (m) | +0.2169 / +0.2442 | +0.2305 | [+0.1141, +0.3530] | 0.0554 (0.0554) | WORSE |
| gap error 6.0 s (m) | +0.8243 / +1.1466 | +0.9854 | [+0.4232, +1.5805] | 0.5019 (0.5019) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0443 (0.0443) | OPEN |
| time-gap error over the scored steps (s) | +0.8436 / +1.0007 | +0.9222 | [+0.5077, +1.3596] | 0.3619 (0.3619) | WORSE |

## I10B0 R0C10Hb20 minus I10B0 R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0249 / -0.0302 | -0.0276 | [-0.0440, -0.0132] | 0.0409 (0.0409) | INSIDE-SPREAD |
| C contact rate | -0.0218 / -0.0171 | -0.0195 | [-0.0334, -0.0077] | 0.0156 (0.0156) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0950 / +0.1012 | +0.0981 | [+0.0698, +0.1293] | 0.0389 (0.0389) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0277 (0.0277) | OPEN |
| gap bias 4.0 s (m) | +0.6707 / +0.7715 | +0.7211 | [+0.6054, +0.8476] | 0.2144 (0.2144) | WORSE |
| gap bias 6.0 s (m) | +2.7210 / +3.1187 | +2.9198 | [+2.3181, +3.5892] | 0.9742 (0.9742) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0249 (0.0249) | OPEN |
| gap error 4.0 s (m) | +0.2994 / +0.3979 | +0.3486 | [+0.2279, +0.4805] | 0.0554 (0.0554) | WORSE |
| gap error 6.0 s (m) | +1.4038 / +2.0636 | +1.7337 | [+1.0930, +2.4937] | 0.5019 (0.5019) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0443 (0.0443) | OPEN |
| time-gap error over the scored steps (s) | +0.6962 / +0.6219 | +0.6590 | [+0.3831, +0.9571] | 0.3619 (0.3619) | WORSE |

## I10RHp3 R0C10XHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0214 / -0.0356 | -0.0285 | [-0.0470, -0.0121] | 0.0409 (0.0409) | INSIDE-SPREAD |
| C contact rate | -0.0078 / -0.0109 | -0.0093 | [-0.0182, -0.0009] | 0.0156 (0.0156) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0343 / +0.0841 | +0.0592 | [+0.0214, +0.1074] | 0.0389 (0.0389) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1136 / -0.0856 | -0.0996 | [-0.1525, -0.0531] | 0.0277 (0.0277) | BETTER |
| gap bias 4.0 s (m) | +0.4537 / +0.6722 | +0.5629 | [+0.3442, +0.8429] | 0.2144 (0.2144) | WORSE |
| gap bias 6.0 s (m) | +1.7960 / +2.6733 | +2.2347 | [+1.5253, +3.1596] | 0.9742 (0.9742) | WORSE |
| gap error 2.0 s (m) | +0.0908 / +0.0793 | +0.0850 | [+0.0526, +0.1222] | 0.0249 (0.0249) | WORSE |
| gap error 4.0 s (m) | +0.4869 / +0.6235 | +0.5552 | [+0.3911, +0.7598] | 0.0554 (0.0554) | WORSE |
| gap error 6.0 s (m) | +0.6315 / +1.3097 | +0.9706 | [+0.3508, +1.7747] | 0.5019 (0.5019) | WORSE |
| time-gap error 2.0 s (s) | +0.0785 / +0.1612 | +0.1198 | [+0.0248, +0.2747] | 0.0443 (0.0443) | WORSE |
| time-gap error over the scored steps (s) | +0.2754 / +0.7919 | +0.5336 | [+0.1334, +1.0037] | 0.3619 (0.3619) | WORSE |

## I10RHp3 R0Hb20 minus I10B0 R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0391 / +0.0178 | +0.0285 | [+0.0088, +0.0556] | 0.0409 (0.0409) | INSIDE-SPREAD |
| C contact rate | +0.0296 / +0.0234 | +0.0265 | [+0.0086, +0.0484] | 0.0156 (0.0156) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0654 / -0.0654 | -0.0654 | [-0.0891, -0.0439] | 0.0389 (0.0389) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1136 / -0.0856 | -0.0996 | [-0.1525, -0.0531] | 0.0277 (0.0277) | BETTER |
| gap bias 4.0 s (m) | -0.6172 / -0.6760 | -0.6466 | [-0.8565, -0.4288] | 0.2144 (0.2144) | BETTER |
| gap bias 6.0 s (m) | -2.5817 / -2.9088 | -2.7453 | [-3.4007, -2.0981] | 0.9742 (0.9742) | BETTER |
| gap error 2.0 s (m) | +0.0908 / +0.0793 | +0.0850 | [+0.0526, +0.1222] | 0.0249 (0.0249) | WORSE |
| gap error 4.0 s (m) | +0.1761 / +0.0983 | +0.1372 | [-0.0326, +0.3304] | 0.0554 (0.0554) | OPEN |
| gap error 6.0 s (m) | -0.3748 / -0.7450 | -0.5599 | [-1.1233, +0.0695] | 0.5019 (0.5019) | OPEN |
| time-gap error 2.0 s (s) | +0.0785 / +0.1612 | +0.1198 | [+0.0248, +0.2747] | 0.0443 (0.0443) | WORSE |
| time-gap error over the scored steps (s) | -0.8452 / -0.7558 | -0.8005 | [-1.1510, -0.4928] | 0.3619 (0.3619) | BETTER |

## I10RHp3 R0C10Hb20 minus I10B0 R0C10Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0498 / +0.0427 | +0.0463 | [+0.0176, +0.0816] | 0.0409 (0.0409) | WORSE |
| C contact rate | +0.0405 / +0.0312 | +0.0358 | [+0.0159, +0.0594] | 0.0156 (0.0156) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.1402 / -0.1199 | -0.1301 | [-0.1757, -0.0895] | 0.0389 (0.0389) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1136 / -0.0856 | -0.0996 | [-0.1525, -0.0531] | 0.0277 (0.0277) | BETTER |
| gap bias 4.0 s (m) | -1.1100 / -1.1058 | -1.1079 | [-1.3424, -0.8700] | 0.2144 (0.2144) | BETTER |
| gap bias 6.0 s (m) | -4.4350 / -4.5164 | -4.4757 | [-5.4650, -3.5829] | 0.9742 (0.9742) | BETTER |
| gap error 2.0 s (m) | +0.0908 / +0.0793 | +0.0850 | [+0.0526, +0.1222] | 0.0249 (0.0249) | WORSE |
| gap error 4.0 s (m) | -0.0613 / -0.0421 | -0.0517 | [-0.2625, +0.1546] | 0.0554 (0.0554) | OPEN |
| gap error 6.0 s (m) | -1.8136 / -1.9474 | -1.8805 | [-2.7351, -1.1207] | 0.5019 (0.5019) | BETTER |
| time-gap error 2.0 s (s) | +0.0785 / +0.1612 | +0.1198 | [+0.0248, +0.2747] | 0.0443 (0.0443) | WORSE |
| time-gap error over the scored steps (s) | -1.3267 / -0.9459 | -1.1363 | [-1.7138, -0.6873] | 0.3619 (0.3619) | BETTER |

## I10RHp3 R0G50Hb20 minus I10B0 R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0285 / +0.0320 | +0.0302 | [+0.0076, +0.0581] | 0.0409 (0.0409) | INSIDE-SPREAD |
| C contact rate | +0.0187 / +0.0171 | +0.0179 | [+0.0064, +0.0319] | 0.0156 (0.0156) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0498 / -0.0405 | -0.0452 | [-0.0704, -0.0218] | 0.0389 (0.0389) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1136 / -0.0856 | -0.0996 | [-0.1525, -0.0531] | 0.0277 (0.0277) | BETTER |
| gap bias 4.0 s (m) | -0.5283 / -0.5872 | -0.5578 | [-0.7383, -0.3886] | 0.2144 (0.2144) | BETTER |
| gap bias 6.0 s (m) | -1.9762 / -2.1177 | -2.0469 | [-2.5456, -1.5828] | 0.9742 (0.9742) | BETTER |
| gap error 2.0 s (m) | +0.0908 / +0.0793 | +0.0850 | [+0.0526, +0.1222] | 0.0249 (0.0249) | WORSE |
| gap error 4.0 s (m) | +0.1307 / +0.1805 | +0.1556 | [+0.0081, +0.2955] | 0.0554 (0.0554) | WORSE |
| gap error 6.0 s (m) | -0.5451 / -0.2644 | -0.4047 | [-0.8875, +0.0544] | 0.5019 (0.5019) | OPEN |
| time-gap error 2.0 s (s) | +0.0785 / +0.1612 | +0.1198 | [+0.0248, +0.2747] | 0.0443 (0.0443) | WORSE |
| time-gap error over the scored steps (s) | -0.8279 / -0.3779 | -0.6029 | [-1.0593, -0.2039] | 0.3619 (0.3619) | BETTER |

## I10RHp3 R0G0Hb20 minus I10B0 R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0125 / +0.0125 | +0.0125 | [+0.0008, +0.0274] | 0.0409 (0.0409) | INSIDE-SPREAD |
| C contact rate | +0.0109 / +0.0031 | +0.0070 | [-0.0013, +0.0175] | 0.0156 (0.0156) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0249 / +0.0016 | -0.0117 | [-0.0348, +0.0130] | 0.0389 (0.0389) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1136 / -0.0856 | -0.0996 | [-0.1525, -0.0531] | 0.0277 (0.0277) | BETTER |
| gap bias 4.0 s (m) | -0.3073 / -0.3053 | -0.3063 | [-0.4441, -0.1686] | 0.2144 (0.2144) | BETTER |
| gap bias 6.0 s (m) | -0.8257 / -0.7082 | -0.7669 | [-1.0954, -0.4282] | 0.9742 (0.9742) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0908 / +0.0793 | +0.0850 | [+0.0526, +0.1222] | 0.0249 (0.0249) | WORSE |
| gap error 4.0 s (m) | +0.1911 / +0.2581 | +0.2246 | [+0.1110, +0.3256] | 0.0554 (0.0554) | WORSE |
| gap error 6.0 s (m) | -0.0589 / +0.3413 | +0.1412 | [-0.2209, +0.4999] | 0.5019 (0.5019) | OPEN |
| time-gap error 2.0 s (s) | +0.0785 / +0.1612 | +0.1198 | [+0.0248, +0.2747] | 0.0443 (0.0443) | WORSE |
| time-gap error over the scored steps (s) | -0.1598 / -0.0240 | -0.0919 | [-0.2579, +0.0800] | 0.3619 (0.3619) | OPEN |

## I10RHp3 R0KHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0089 / +0.0089 | +0.0089 | [+0.0024, +0.0164] | 0.0409 (0.0409) | INSIDE-SPREAD |
| C contact rate | +0.0078 / +0.0047 | +0.0062 | [-0.0002, +0.0149] | 0.0156 (0.0156) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0249 / +0.0016 | -0.0117 | [-0.0341, +0.0109] | 0.0389 (0.0389) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1136 / -0.0856 | -0.0996 | [-0.1525, -0.0531] | 0.0277 (0.0277) | BETTER |
| gap bias 4.0 s (m) | -0.2934 / -0.2546 | -0.2740 | [-0.4155, -0.1338] | 0.2144 (0.2144) | BETTER |
| gap bias 6.0 s (m) | -0.6066 / -0.5555 | -0.5810 | [-0.8718, -0.2671] | 0.9742 (0.9742) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0908 / +0.0793 | +0.0850 | [+0.0526, +0.1222] | 0.0249 (0.0249) | WORSE |
| gap error 4.0 s (m) | +0.1866 / +0.2414 | +0.2140 | [+0.1091, +0.3096] | 0.0554 (0.0554) | WORSE |
| gap error 6.0 s (m) | +0.0054 / +0.2479 | +0.1267 | [-0.1267, +0.3937] | 0.5019 (0.5019) | OPEN |
| time-gap error 2.0 s (s) | +0.0785 / +0.1612 | +0.1198 | [+0.0248, +0.2747] | 0.0443 (0.0443) | WORSE |
| time-gap error over the scored steps (s) | -0.1352 / -0.0459 | -0.0906 | [-0.2459, +0.0736] | 0.3619 (0.3619) | OPEN |

## I10RHp3 R0Hb20 minus I10RH R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0178 / -0.0018 | +0.0080 | [-0.0076, +0.0266] | 0.0409 (0.0409) | OPEN |
| C contact rate | +0.0125 / +0.0093 | +0.0109 | [+0.0000, +0.0245] | 0.0156 (0.0156) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0312 / -0.0202 | -0.0257 | [-0.0407, -0.0117] | 0.0389 (0.0389) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0833 / -0.0460 | -0.0646 | [-0.1035, -0.0287] | 0.0277 (0.0277) | BETTER |
| gap bias 4.0 s (m) | -0.3871 / -0.3112 | -0.3492 | [-0.4932, -0.2167] | 0.2144 (0.2144) | BETTER |
| gap bias 6.0 s (m) | -1.1356 / -1.0987 | -1.1171 | [-1.4658, -0.7822] | 0.9742 (0.9742) | BETTER |
| gap error 2.0 s (m) | +0.0592 / +0.0574 | +0.0583 | [+0.0376, +0.0818] | 0.0249 (0.0249) | WORSE |
| gap error 4.0 s (m) | +0.0582 / +0.1388 | +0.0985 | [-0.0264, +0.2094] | 0.0554 (0.0554) | OPEN |
| gap error 6.0 s (m) | -0.2070 / -0.0586 | -0.1328 | [-0.4906, +0.1535] | 0.5019 (0.5019) | OPEN |
| time-gap error 2.0 s (s) | +0.0513 / +0.1136 | +0.0825 | [-0.0072, +0.1958] | 0.0443 (0.0443) | OPEN |
| time-gap error over the scored steps (s) | -0.3102 / -0.1916 | -0.2509 | [-0.4380, -0.0527] | 0.3619 (0.3619) | INSIDE-SPREAD |

## I10RHp3 R0C10XHb20 minus I10RH R0C10XHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0125 / +0.0214 | +0.0169 | [+0.0042, +0.0342] | 0.0409 (0.0409) | INSIDE-SPREAD |
| C contact rate | +0.0156 / +0.0140 | +0.0148 | [+0.0028, +0.0290] | 0.0156 (0.0156) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0483 / -0.0312 | -0.0397 | [-0.0600, -0.0220] | 0.0389 (0.0389) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0833 / -0.0460 | -0.0646 | [-0.1035, -0.0287] | 0.0277 (0.0277) | BETTER |
| gap bias 4.0 s (m) | -0.5890 / -0.4219 | -0.5055 | [-0.6729, -0.3361] | 0.2144 (0.2144) | BETTER |
| gap bias 6.0 s (m) | -1.5673 / -1.4126 | -1.4899 | [-1.8203, -1.1636] | 0.9742 (0.9742) | BETTER |
| gap error 2.0 s (m) | +0.0592 / +0.0574 | +0.0583 | [+0.0376, +0.0818] | 0.0249 (0.0249) | WORSE |
| gap error 4.0 s (m) | -0.1155 / +0.1088 | -0.0034 | [-0.1973, +0.1734] | 0.0554 (0.0554) | OPEN |
| gap error 6.0 s (m) | -0.6937 / -0.2877 | -0.4907 | [-0.9294, -0.0959] | 0.5019 (0.5019) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0513 / +0.1136 | +0.0825 | [-0.0072, +0.1958] | 0.0443 (0.0443) | OPEN |
| time-gap error over the scored steps (s) | -0.6082 / -0.2587 | -0.4335 | [-0.7933, -0.0748] | 0.3619 (0.3619) | BETTER |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10B0_R0C10Hb20 (0,1) | 0.0329 [0.0136, 0.0571] | 0.0195 [0.0060, 0.0360] | 0.2882 [0.2316, 0.3454] | 0.0891 [0.0465, 0.1353] | 2.1520 [1.6960, 2.6073] | 7.6474 [6.1154, 9.2291] | 0.4317 [0.3899, 0.4802] | 2.9594 [2.5948, 3.3607] | 8.7568 [7.5255, 10.0816] | 0.8992 [0.6263, 1.2159] | 3.7619 [2.7922, 4.7955] |
| I10B0_R0C10XHb20 (0,1) | 0.0294 [0.0104, 0.0534] | 0.0210 [0.0073, 0.0374] | 0.2796 [0.2236, 0.3374] | 0.0891 [0.0465, 0.1353] | 2.1882 [1.7296, 2.6381] | 7.4618 [6.0483, 8.9499] | 0.4317 [0.3899, 0.4802] | 2.9698 [2.6130, 3.3670] | 8.6467 [7.5117, 9.8733] | 0.8992 [0.6263, 1.2159] | 3.8517 [2.8854, 4.8691] |
| I10B0_R0G0Hb20 (0,1) | 0.0943 [0.0578, 0.1338] | 0.0592 [0.0296, 0.0938] | 0.1176 [0.0805, 0.1614] | 0.0891 [0.0465, 0.1353] | 0.6899 [0.3114, 1.0864] | 1.8093 [0.6138, 3.0721] | 0.4317 [0.3899, 0.4802] | 2.3802 [2.0930, 2.7083] | 6.0377 [5.2686, 6.8989] | 0.8992 [0.6263, 1.2159] | 2.1807 [1.5120, 2.9252] |
| I10B0_R0G50Hb20 (0,1) | 0.0605 [0.0351, 0.0901] | 0.0389 [0.0150, 0.0669] | 0.1900 [0.1424, 0.2401] | 0.0891 [0.0465, 0.1353] | 1.4309 [1.0198, 1.8362] | 4.7275 [3.3100, 6.1037] | 0.4317 [0.3899, 0.4802] | 2.6107 [2.2874, 2.9683] | 7.0231 [6.0954, 8.0773] | 0.8992 [0.6263, 1.2159] | 3.1028 [2.2323, 4.0349] |
| I10B0_R0Hb20 (0,1) | 0.0614 [0.0289, 0.0956] | 0.0280 [0.0116, 0.0479] | 0.2033 [0.1503, 0.2595] | 0.0891 [0.0465, 0.1353] | 1.4575 [0.9911, 1.9165] | 5.2877 [3.8033, 6.8244] | 0.4317 [0.3899, 0.4802] | 2.6569 [2.3239, 3.0194] | 7.3768 [6.2706, 8.5258] | 0.8992 [0.6263, 1.2159] | 3.1310 [2.2928, 4.0200] |
| I10B0_R0KHb20 (0,1) | 0.0925 [0.0563, 0.1324] | 0.0600 [0.0297, 0.0927] | 0.1145 [0.0783, 0.1567] | 0.0891 [0.0465, 0.1353] | 0.7204 [0.3444, 1.1110] | 1.7458 [0.5279, 3.0407] | 0.4317 [0.3899, 0.4802] | 2.3660 [2.0834, 2.6885] | 6.0590 [5.2991, 6.9059] | 0.8992 [0.6263, 1.2159] | 2.1774 [1.5055, 2.9188] |
| I10RH_R0C10XHb20 (0,1) | 0.0472 [0.0221, 0.0795] | 0.0358 [0.0137, 0.0613] | 0.2134 [0.1621, 0.2687] | 0.0541 [-0.0003, 0.1137] | 1.7888 [1.3557, 2.2578] | 5.4704 [4.1477, 6.8065] | 0.4584 [0.4066, 0.5113] | 2.9245 [2.5330, 3.3539] | 7.5203 [6.6391, 8.5576] | 0.9366 [0.6450, 1.2757] | 3.1445 [2.3022, 4.0131] |
| I10RH_R0Hb20 (0,1) | 0.0819 [0.0438, 0.1239] | 0.0436 [0.0206, 0.0728] | 0.1636 [0.1163, 0.2142] | 0.0541 [-0.0003, 0.1137] | 1.1601 [0.7080, 1.6234] | 3.6596 [2.2695, 5.1805] | 0.4584 [0.4066, 0.5113] | 2.6956 [2.3561, 3.0709] | 6.9497 [6.0649, 7.9524] | 0.9366 [0.6450, 1.2757] | 2.5814 [1.8594, 3.3407] |
| I10RHp3_R0C10Hb20 (0,1) | 0.0792 [0.0407, 0.1257] | 0.0553 [0.0257, 0.0899] | 0.1581 [0.1059, 0.2146] | -0.0105 [-0.0868, 0.0661] | 1.0442 [0.5086, 1.5967] | 3.1717 [1.5802, 4.7991] | 0.5167 [0.4557, 0.5854] | 2.9076 [2.5543, 3.2773] | 6.8763 [5.9305, 7.9014] | 1.0190 [0.6817, 1.4241] | 2.6255 [1.8762, 3.4950] |
| I10RHp3_R0C10XHb20 (0,1) | 0.0641 [0.0343, 0.0980] | 0.0506 [0.0237, 0.0823] | 0.1737 [0.1237, 0.2280] | -0.0105 [-0.0868, 0.0661] | 1.2833 [0.7621, 1.8265] | 3.9805 [2.5096, 5.5008] | 0.5167 [0.4557, 0.5854] | 2.9212 [2.5563, 3.3119] | 7.0296 [6.1087, 8.0671] | 1.0190 [0.6817, 1.4241] | 2.7110 [1.9286, 3.5914] |
| I10RHp3_R0G0Hb20 (0,1) | 0.1068 [0.0645, 0.1547] | 0.0662 [0.0332, 0.1060] | 0.1059 [0.0693, 0.1476] | -0.0105 [-0.0868, 0.0661] | 0.3836 [-0.0708, 0.8590] | 1.0424 [-0.2684, 2.3894] | 0.5167 [0.4557, 0.5854] | 2.6048 [2.2950, 2.9556] | 6.1789 [5.3506, 7.0670] | 1.0190 [0.6817, 1.4241] | 2.0887 [1.4547, 2.7800] |
| I10RHp3_R0G50Hb20 (0,1) | 0.0907 [0.0510, 0.1351] | 0.0569 [0.0272, 0.0915] | 0.1449 [0.0997, 0.1942] | -0.0105 [-0.0868, 0.0661] | 0.8732 [0.3860, 1.3770] | 2.6806 [1.1983, 4.1449] | 0.5167 [0.4557, 0.5854] | 2.7663 [2.4249, 3.1407] | 6.6184 [5.7094, 7.5991] | 1.0190 [0.6817, 1.4241] | 2.4999 [1.7478, 3.3652] |
| I10RHp3_R0Hb20 (0,1) | 0.0899 [0.0442, 0.1417] | 0.0545 [0.0267, 0.0879] | 0.1379 [0.0882, 0.1921] | -0.0105 [-0.0868, 0.0661] | 0.8109 [0.2976, 1.3331] | 2.5425 [1.0226, 4.1473] | 0.5167 [0.4557, 0.5854] | 2.7941 [2.4587, 3.1638] | 6.8169 [5.9208, 7.8418] | 1.0190 [0.6817, 1.4241] | 2.3305 [1.6570, 3.0864] |
| I10RHp3_R0KHb20 (0,1) | 0.1014 [0.0618, 0.1451] | 0.0662 [0.0331, 0.1051] | 0.1028 [0.0669, 0.1425] | -0.0105 [-0.0868, 0.0661] | 0.4464 [0.0032, 0.9175] | 1.1648 [-0.1056, 2.5176] | 0.5167 [0.4557, 0.5854] | 2.5800 [2.2661, 2.9411] | 6.1857 [5.3955, 7.0576] | 1.0190 [0.6817, 1.4241] | 2.0868 [1.4504, 2.7759] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
