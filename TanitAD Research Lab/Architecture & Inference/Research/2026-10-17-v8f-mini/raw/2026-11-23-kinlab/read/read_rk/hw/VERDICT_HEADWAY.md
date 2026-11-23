# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: F:/TanitAD-scratch/strat1/read/inputs/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 30, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.02669039145907473, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RK_R0C10XHb20:I10B0_R0C10XHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RK_R0C10XHb20:I10RK_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RK_R0G50Hb20:I10RK_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RK_R0C10Hb20:I10RK_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0C10XHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0G50Hb20:I10B0_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0C10Hb20:I10B0_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RK_R0C10XHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RK_R0Hb20:I10B0_R0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RK_R0C10Hb20:I10B0_R0C10Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RK_R0G50Hb20:I10B0_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RK_R0G0Hb20:I10B0_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RK_R0KHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RK_R0Hb20:I10RH_R0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RK_R0C10XHb20:I10RH_R0C10XHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RK_R0C10XHb20:I10RL_R0C10XHb20", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10RK_R0Hb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10RK_R0C10XHb20 minus I10B0_R0C10XHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RK_R0C10XHb20 minus I10RK_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RK_R0G50Hb20 minus I10RK_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RK_R0C10Hb20 minus I10RK_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0C10XHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0G50Hb20 minus I10B0_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0C10Hb20 minus I10B0_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RK_R0C10XHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RK_R0Hb20 minus I10B0_R0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RK_R0C10Hb20 minus I10B0_R0C10Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RK_R0G50Hb20 minus I10B0_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RK_R0G0Hb20 minus I10B0_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RK_R0KHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RK_R0Hb20 minus I10RH_R0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RK_R0C10XHb20 minus I10RH_R0C10XHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RK_R0C10XHb20 minus I10RL_R0C10XHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10RK R0C10XHb20 minus I10B0 R0C10XHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0178 / +0.0178 | +0.0178 | [+0.0065, +0.0310] | 0.0267 (0.0267) | INSIDE-SPREAD |
| C contact rate | +0.0187 / +0.0078 | +0.0132 | [+0.0025, +0.0272] | 0.0156 (0.0156) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0685 / -0.0545 | -0.0615 | [-0.0973, -0.0330] | 0.0312 (0.0312) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0274 / -0.0359 | -0.0317 | [-0.0514, -0.0130] | 0.0088 (0.0088) | BETTER |
| gap bias 4.0 s (m) | -0.2716 / -0.4179 | -0.3447 | [-0.5161, -0.1885] | 0.1837 (0.1837) | BETTER |
| gap bias 6.0 s (m) | -1.6694 / -1.8546 | -1.7620 | [-2.4541, -1.1981] | 1.0323 (1.0323) | BETTER |
| gap error 2.0 s (m) | +0.0232 / +0.0206 | +0.0219 | [+0.0089, +0.0362] | 0.0159 (0.0159) | WORSE |
| gap error 4.0 s (m) | +0.0429 / -0.1319 | -0.0445 | [-0.2176, +0.1157] | 0.0840 (0.0840) | OPEN |
| gap error 6.0 s (m) | -0.7153 / -1.1418 | -0.9285 | [-1.6531, -0.3351] | 0.3766 (0.3766) | BETTER |
| time-gap error 2.0 s (s) | +0.0202 / +0.0690 | +0.0446 | [+0.0008, +0.1175] | 0.0104 (0.0104) | WORSE |
| time-gap error over the scored steps (s) | -0.6664 / -0.6335 | -0.6500 | [-0.9183, -0.4071] | 0.2853 (0.2853) | BETTER |

## I10RK R0C10XHb20 minus I10RK R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0409 / -0.0552 | -0.0480 | [-0.0736, -0.0239] | 0.0267 (0.0267) | BETTER |
| C contact rate | -0.0249 / -0.0312 | -0.0280 | [-0.0465, -0.0127] | 0.0156 (0.0156) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1012 / +0.1231 | +0.1121 | [+0.0799, +0.1488] | 0.0312 (0.0312) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0088 (0.0088) | OPEN |
| gap bias 4.0 s (m) | +1.1358 / +1.3031 | +1.2195 | [+1.0437, +1.4195] | 0.1837 (0.1837) | WORSE |
| gap bias 6.0 s (m) | +3.7079 / +4.6785 | +4.1932 | [+3.3922, +5.0908] | 1.0323 (1.0323) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0159 (0.0159) | OPEN |
| gap error 4.0 s (m) | +0.5104 / +0.4723 | +0.4913 | [+0.3098, +0.6814] | 0.0840 (0.0840) | WORSE |
| gap error 6.0 s (m) | +1.4998 / +1.7347 | +1.6172 | [+0.9372, +2.3495] | 0.3766 (0.3766) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0104 (0.0104) | OPEN |
| time-gap error over the scored steps (s) | +0.9552 / +1.1592 | +1.0572 | [+0.6395, +1.5207] | 0.2853 (0.2853) | WORSE |

## I10RK R0G50Hb20 minus I10RK R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0160 / -0.0338 | -0.0249 | [-0.0468, -0.0100] | 0.0267 (0.0267) | INSIDE-SPREAD |
| C contact rate | -0.0187 / -0.0171 | -0.0179 | [-0.0298, -0.0083] | 0.0156 (0.0156) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0623 / +0.0717 | +0.0670 | [+0.0459, +0.0881] | 0.0312 (0.0312) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0088 (0.0088) | OPEN |
| gap bias 4.0 s (m) | +0.6211 / +0.6840 | +0.6526 | [+0.5659, +0.7499] | 0.1837 (0.1837) | WORSE |
| gap bias 6.0 s (m) | +2.0270 / +2.6496 | +2.3383 | [+1.8542, +2.9066] | 1.0323 (1.0323) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0159 (0.0159) | OPEN |
| gap error 4.0 s (m) | +0.2341 / +0.1840 | +0.2091 | [+0.0989, +0.3214] | 0.0840 (0.0840) | WORSE |
| gap error 6.0 s (m) | +0.7654 / +0.7237 | +0.7445 | [+0.3073, +1.1735] | 0.3766 (0.3766) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0104 (0.0104) | OPEN |
| time-gap error over the scored steps (s) | +0.4223 / +0.7787 | +0.6005 | [+0.2803, +0.9725] | 0.2853 (0.2853) | WORSE |

## I10RK R0C10Hb20 minus I10RK R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0160 / -0.0178 | -0.0169 | [-0.0268, -0.0079] | 0.0267 (0.0267) | INSIDE-SPREAD |
| C contact rate | -0.0062 / -0.0125 | -0.0093 | [-0.0197, +0.0000] | 0.0156 (0.0156) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0218 / +0.0623 | +0.0421 | [+0.0152, +0.0764] | 0.0312 (0.0312) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0088 (0.0088) | OPEN |
| gap bias 4.0 s (m) | +0.4162 / +0.5361 | +0.4762 | [+0.3843, +0.5879] | 0.1837 (0.1837) | WORSE |
| gap bias 6.0 s (m) | +1.2592 / +1.8281 | +1.5436 | [+1.1270, +2.0157] | 1.0323 (1.0323) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0159 (0.0159) | OPEN |
| gap error 4.0 s (m) | +0.2180 / +0.2342 | +0.2261 | [+0.1537, +0.3043] | 0.0840 (0.0840) | WORSE |
| gap error 6.0 s (m) | +0.5103 / +0.8673 | +0.6888 | [+0.3785, +1.0330] | 0.3766 (0.3766) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0104 (0.0104) | OPEN |
| time-gap error over the scored steps (s) | +0.3144 / +0.3136 | +0.3140 | [+0.1626, +0.4696] | 0.2853 (0.2853) | WORSE |

## I10B0 R0C10XHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0516 / -0.0747 | -0.0632 | [-0.0977, -0.0308] | 0.0267 (0.0267) | BETTER |
| C contact rate | -0.0405 / -0.0374 | -0.0389 | [-0.0614, -0.0194] | 0.0156 (0.0156) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1511 / +0.1791 | +0.1651 | [+0.1215, +0.2139] | 0.0312 (0.0312) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0088 (0.0088) | OPEN |
| gap bias 4.0 s (m) | +1.3409 / +1.5946 | +1.4678 | [+1.2207, +1.7382] | 0.1837 (0.1837) | WORSE |
| gap bias 6.0 s (m) | +5.1910 / +6.2410 | +5.7160 | [+4.6498, +6.9298] | 1.0323 (1.0323) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0159 (0.0159) | OPEN |
| gap error 4.0 s (m) | +0.5314 / +0.6764 | +0.6039 | [+0.3704, +0.8402] | 0.0840 (0.0840) | WORSE |
| gap error 6.0 s (m) | +2.2039 / +2.9716 | +2.5877 | [+1.5457, +3.7505] | 0.3766 (0.3766) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0104 (0.0104) | OPEN |
| time-gap error over the scored steps (s) | +1.5846 / +1.7639 | +1.6743 | [+1.1307, +2.2340] | 0.2853 (0.2853) | WORSE |

## I10B0 R0G50Hb20 minus I10B0 R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0285 / -0.0391 | -0.0338 | [-0.0540, -0.0149] | 0.0267 (0.0267) | BETTER |
| C contact rate | -0.0202 / -0.0202 | -0.0202 | [-0.0352, -0.0082] | 0.0156 (0.0156) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0561 / +0.0888 | +0.0724 | [+0.0441, +0.1092] | 0.0312 (0.0312) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0088 (0.0088) | OPEN |
| gap bias 4.0 s (m) | +0.6755 / +0.8065 | +0.7410 | [+0.6149, +0.8796] | 0.1837 (0.1837) | WORSE |
| gap bias 6.0 s (m) | +2.5213 / +3.3152 | +2.9182 | [+2.2914, +3.6283] | 1.0323 (1.0323) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0159 (0.0159) | OPEN |
| gap error 4.0 s (m) | +0.2169 / +0.2442 | +0.2305 | [+0.1141, +0.3530] | 0.0840 (0.0840) | WORSE |
| gap error 6.0 s (m) | +0.8243 / +1.1466 | +0.9854 | [+0.4232, +1.5805] | 0.3766 (0.3766) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0104 (0.0104) | OPEN |
| time-gap error over the scored steps (s) | +0.8436 / +1.0007 | +0.9222 | [+0.5077, +1.3596] | 0.2853 (0.2853) | WORSE |

## I10B0 R0C10Hb20 minus I10B0 R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0249 / -0.0302 | -0.0276 | [-0.0440, -0.0132] | 0.0267 (0.0267) | BETTER |
| C contact rate | -0.0218 / -0.0171 | -0.0195 | [-0.0334, -0.0077] | 0.0156 (0.0156) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0950 / +0.1012 | +0.0981 | [+0.0698, +0.1293] | 0.0312 (0.0312) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0088 (0.0088) | OPEN |
| gap bias 4.0 s (m) | +0.6707 / +0.7715 | +0.7211 | [+0.6054, +0.8476] | 0.1837 (0.1837) | WORSE |
| gap bias 6.0 s (m) | +2.7210 / +3.1187 | +2.9198 | [+2.3181, +3.5892] | 1.0323 (1.0323) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0159 (0.0159) | OPEN |
| gap error 4.0 s (m) | +0.2994 / +0.3979 | +0.3486 | [+0.2279, +0.4805] | 0.0840 (0.0840) | WORSE |
| gap error 6.0 s (m) | +1.4038 / +2.0636 | +1.7337 | [+1.0930, +2.4937] | 0.3766 (0.3766) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0104 (0.0104) | OPEN |
| time-gap error over the scored steps (s) | +0.6962 / +0.6219 | +0.6590 | [+0.3831, +0.9571] | 0.2853 (0.2853) | WORSE |

## I10RK R0C10XHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0338 / -0.0569 | -0.0454 | [-0.0746, -0.0207] | 0.0267 (0.0267) | BETTER |
| C contact rate | -0.0218 / -0.0296 | -0.0257 | [-0.0438, -0.0111] | 0.0156 (0.0156) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0826 / +0.1246 | +0.1036 | [+0.0661, +0.1472] | 0.0312 (0.0312) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0274 / -0.0359 | -0.0317 | [-0.0514, -0.0130] | 0.0088 (0.0088) | BETTER |
| gap bias 4.0 s (m) | +1.0693 / +1.1768 | +1.1230 | [+0.9558, +1.3121] | 0.1837 (0.1837) | WORSE |
| gap bias 6.0 s (m) | +3.5215 / +4.3865 | +3.9540 | [+3.1854, +4.8351] | 1.0323 (1.0323) | WORSE |
| gap error 2.0 s (m) | +0.0232 / +0.0206 | +0.0219 | [+0.0089, +0.0362] | 0.0159 (0.0159) | WORSE |
| gap error 4.0 s (m) | +0.5743 / +0.5444 | +0.5594 | [+0.3775, +0.7428] | 0.0840 (0.0840) | WORSE |
| gap error 6.0 s (m) | +1.4886 / +1.8298 | +1.6592 | [+0.9822, +2.4336] | 0.3766 (0.3766) | WORSE |
| time-gap error 2.0 s (s) | +0.0202 / +0.0690 | +0.0446 | [+0.0008, +0.1175] | 0.0104 (0.0104) | WORSE |
| time-gap error over the scored steps (s) | +0.9182 / +1.1304 | +1.0243 | [+0.6012, +1.4546] | 0.2853 (0.2853) | WORSE |

## I10RK R0Hb20 minus I10B0 R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0231 / +0.0160 | +0.0196 | [+0.0078, +0.0358] | 0.0267 (0.0267) | INSIDE-SPREAD |
| C contact rate | +0.0187 / +0.0125 | +0.0156 | [+0.0043, +0.0305] | 0.0156 (0.0156) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0312 / -0.0389 | -0.0350 | [-0.0551, -0.0176] | 0.0312 (0.0312) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0274 / -0.0359 | -0.0317 | [-0.0514, -0.0130] | 0.0088 (0.0088) | BETTER |
| gap bias 4.0 s (m) | -0.2230 / -0.3124 | -0.2677 | [-0.3934, -0.1416] | 0.1837 (0.1837) | BETTER |
| gap bias 6.0 s (m) | -1.3773 / -1.6462 | -1.5118 | [-1.9978, -1.0697] | 1.0323 (1.0323) | BETTER |
| gap error 2.0 s (m) | +0.0232 / +0.0206 | +0.0219 | [+0.0089, +0.0362] | 0.0159 (0.0159) | WORSE |
| gap error 4.0 s (m) | +0.0706 / -0.0359 | +0.0173 | [-0.0930, +0.1430] | 0.0840 (0.0840) | OPEN |
| gap error 6.0 s (m) | -0.0938 / -0.5894 | -0.3416 | [-0.8391, +0.1524] | 0.3766 (0.3766) | OPEN |
| time-gap error 2.0 s (s) | +0.0202 / +0.0690 | +0.0446 | [+0.0008, +0.1175] | 0.0104 (0.0104) | WORSE |
| time-gap error over the scored steps (s) | -0.5681 / -0.5553 | -0.5617 | [-0.8261, -0.3235] | 0.2853 (0.2853) | BETTER |

## I10RK R0C10Hb20 minus I10B0 R0C10Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0231 / +0.0214 | +0.0222 | [+0.0091, +0.0378] | 0.0267 (0.0267) | INSIDE-SPREAD |
| C contact rate | +0.0218 / +0.0125 | +0.0171 | [+0.0064, +0.0309] | 0.0156 (0.0156) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0841 / -0.0623 | -0.0732 | [-0.1135, -0.0403] | 0.0312 (0.0312) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0274 / -0.0359 | -0.0317 | [-0.0514, -0.0130] | 0.0088 (0.0088) | BETTER |
| gap bias 4.0 s (m) | -0.3695 / -0.5082 | -0.4389 | [-0.6183, -0.2812] | 0.1837 (0.1837) | BETTER |
| gap bias 6.0 s (m) | -2.1922 / -2.3069 | -2.2496 | [-3.0149, -1.5930] | 1.0323 (1.0323) | BETTER |
| gap error 2.0 s (m) | +0.0232 / +0.0206 | +0.0219 | [+0.0089, +0.0362] | 0.0159 (0.0159) | WORSE |
| gap error 4.0 s (m) | +0.0033 / -0.1485 | -0.0726 | [-0.2386, +0.0876] | 0.0840 (0.0840) | OPEN |
| gap error 6.0 s (m) | -1.0219 / -1.4860 | -1.2539 | [-2.0688, -0.5614] | 0.3766 (0.3766) | BETTER |
| time-gap error 2.0 s (s) | +0.0202 / +0.0690 | +0.0446 | [+0.0008, +0.1175] | 0.0104 (0.0104) | WORSE |
| time-gap error over the scored steps (s) | -0.8421 / -0.5501 | -0.6961 | [-1.0973, -0.3865] | 0.2853 (0.2853) | BETTER |

## I10RK R0G50Hb20 minus I10B0 R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0142 / +0.0089 | +0.0116 | [+0.0014, +0.0243] | 0.0267 (0.0267) | INSIDE-SPREAD |
| C contact rate | +0.0062 / +0.0078 | +0.0070 | [+0.0000, +0.0150] | 0.0156 (0.0156) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0109 / -0.0234 | -0.0171 | [-0.0393, +0.0014] | 0.0312 (0.0312) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0274 / -0.0359 | -0.0317 | [-0.0514, -0.0130] | 0.0088 (0.0088) | BETTER |
| gap bias 4.0 s (m) | -0.1150 / -0.2729 | -0.1939 | [-0.3356, -0.0548] | 0.1837 (0.1837) | BETTER |
| gap bias 6.0 s (m) | -0.7303 / -1.0164 | -0.8733 | [-1.2605, -0.5292] | 1.0323 (1.0323) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0232 / +0.0206 | +0.0219 | [+0.0089, +0.0362] | 0.0159 (0.0159) | WORSE |
| gap error 4.0 s (m) | +0.0847 / +0.0152 | +0.0499 | [-0.0413, +0.1477] | 0.0840 (0.0840) | OPEN |
| gap error 6.0 s (m) | -0.1283 / -0.2897 | -0.2090 | [-0.5245, +0.0830] | 0.3766 (0.3766) | OPEN |
| time-gap error 2.0 s (s) | +0.0202 / +0.0690 | +0.0446 | [+0.0008, +0.1175] | 0.0104 (0.0104) | WORSE |
| time-gap error over the scored steps (s) | -0.4603 / -0.2418 | -0.3510 | [-0.6441, -0.1149] | 0.2853 (0.2853) | BETTER |

## I10RK R0G0Hb20 minus I10B0 R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0018 / +0.0036 | +0.0027 | [-0.0028, +0.0090] | 0.0267 (0.0267) | OPEN |
| C contact rate | +0.0047 / +0.0047 | +0.0047 | [+0.0004, +0.0098] | 0.0156 (0.0156) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0171 / -0.0062 | -0.0117 | [-0.0257, -0.0001] | 0.0312 (0.0312) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0274 / -0.0359 | -0.0317 | [-0.0514, -0.0130] | 0.0088 (0.0088) | BETTER |
| gap bias 4.0 s (m) | -0.0605 / -0.1504 | -0.1055 | [-0.1996, -0.0169] | 0.1837 (0.1837) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.2361 / -0.3508 | -0.2934 | [-0.4893, -0.1071] | 1.0323 (1.0323) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0232 / +0.0206 | +0.0219 | [+0.0089, +0.0362] | 0.0159 (0.0159) | WORSE |
| gap error 4.0 s (m) | +0.0674 / +0.0754 | +0.0714 | [+0.0227, +0.1204] | 0.0840 (0.0840) | INSIDE-SPREAD |
| gap error 6.0 s (m) | -0.0694 / +0.1332 | +0.0319 | [-0.1663, +0.2079] | 0.3766 (0.3766) | OPEN |
| time-gap error 2.0 s (s) | +0.0202 / +0.0690 | +0.0446 | [+0.0008, +0.1175] | 0.0104 (0.0104) | WORSE |
| time-gap error over the scored steps (s) | -0.0389 / -0.0198 | -0.0293 | [-0.1079, +0.0432] | 0.2853 (0.2853) | OPEN |

## I10RK R0KHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0071 / -0.0018 | +0.0027 | [-0.0047, +0.0101] | 0.0267 (0.0267) | OPEN |
| C contact rate | +0.0031 / +0.0016 | +0.0023 | [-0.0021, +0.0078] | 0.0156 (0.0156) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0187 / +0.0016 | -0.0086 | [-0.0259, +0.0069] | 0.0312 (0.0312) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0274 / -0.0359 | -0.0317 | [-0.0514, -0.0130] | 0.0088 (0.0088) | BETTER |
| gap bias 4.0 s (m) | -0.0665 / -0.1263 | -0.0964 | [-0.1741, -0.0210] | 0.1837 (0.1837) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.1863 / -0.2921 | -0.2392 | [-0.4121, -0.0805] | 1.0323 (1.0323) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0232 / +0.0206 | +0.0219 | [+0.0089, +0.0362] | 0.0159 (0.0159) | WORSE |
| gap error 4.0 s (m) | +0.0639 / +0.0721 | +0.0680 | [+0.0222, +0.1142] | 0.0840 (0.0840) | INSIDE-SPREAD |
| gap error 6.0 s (m) | -0.0112 / +0.0952 | +0.0420 | [-0.0926, +0.1780] | 0.3766 (0.3766) | OPEN |
| time-gap error 2.0 s (s) | +0.0202 / +0.0690 | +0.0446 | [+0.0008, +0.1175] | 0.0104 (0.0104) | WORSE |
| time-gap error over the scored steps (s) | -0.0370 / -0.0288 | -0.0329 | [-0.1185, +0.0529] | 0.2853 (0.2853) | OPEN |

## I10RK R0Hb20 minus I10RH R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0018 / -0.0036 | -0.0009 | [-0.0069, +0.0061] | 0.0267 (0.0267) | OPEN |
| C contact rate | +0.0016 / -0.0016 | +0.0000 | [-0.0049, +0.0072] | 0.0156 (0.0156) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0031 / +0.0062 | +0.0047 | [-0.0031, +0.0131] | 0.0312 (0.0312) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0030 / +0.0037 | +0.0033 | [-0.0034, +0.0108] | 0.0088 (0.0088) | OPEN |
| gap bias 4.0 s (m) | +0.0071 / +0.0524 | +0.0297 | [-0.0311, +0.0810] | 0.1837 (0.1837) | OPEN |
| gap bias 6.0 s (m) | +0.0688 / +0.1639 | +0.1164 | [-0.0424, +0.2685] | 1.0323 (1.0323) | OPEN |
| gap error 2.0 s (m) | -0.0085 / -0.0013 | -0.0049 | [-0.0127, +0.0012] | 0.0159 (0.0159) | OPEN |
| gap error 4.0 s (m) | -0.0474 / +0.0046 | -0.0214 | [-0.0697, +0.0186] | 0.0840 (0.0840) | OPEN |
| gap error 6.0 s (m) | +0.0739 / +0.0970 | +0.0855 | [-0.0248, +0.1991] | 0.3766 (0.3766) | OPEN |
| time-gap error 2.0 s (s) | -0.0070 / +0.0214 | +0.0072 | [-0.0224, +0.0361] | 0.0104 (0.0104) | OPEN |
| time-gap error over the scored steps (s) | -0.0331 / +0.0090 | -0.0121 | [-0.1072, +0.0738] | 0.2853 (0.2853) | OPEN |

## I10RK R0C10XHb20 minus I10RH R0C10XHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0000 | +0.0000 | [-0.0049, +0.0052] | 0.0267 (0.0267) | OPEN |
| C contact rate | +0.0016 / -0.0047 | -0.0016 | [-0.0083, +0.0031] | 0.0156 (0.0156) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0000 / +0.0093 | +0.0047 | [-0.0048, +0.0166] | 0.0312 (0.0312) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0030 / +0.0037 | +0.0033 | [-0.0034, +0.0108] | 0.0088 (0.0088) | OPEN |
| gap bias 4.0 s (m) | +0.0266 / +0.0827 | +0.0546 | [-0.0116, +0.1115] | 0.1837 (0.1837) | OPEN |
| gap bias 6.0 s (m) | +0.1582 / +0.3006 | +0.2294 | [+0.0496, +0.3884] | 1.0323 (1.0323) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0085 / -0.0013 | -0.0049 | [-0.0127, +0.0012] | 0.0159 (0.0159) | OPEN |
| gap error 4.0 s (m) | -0.0282 / +0.0298 | +0.0008 | [-0.0505, +0.0487] | 0.0840 (0.0840) | OPEN |
| gap error 6.0 s (m) | +0.1633 / +0.2324 | +0.1979 | [+0.0752, +0.3143] | 0.3766 (0.3766) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.0070 / +0.0214 | +0.0072 | [-0.0224, +0.0361] | 0.0104 (0.0104) | OPEN |
| time-gap error over the scored steps (s) | +0.0346 / +0.0798 | +0.0572 | [-0.0611, +0.1813] | 0.2853 (0.2853) | OPEN |

## I10RK R0C10XHb20 minus I10RL R0C10XHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0018 / -0.0107 | -0.0044 | [-0.0145, +0.0059] | 0.0267 (0.0267) | OPEN |
| C contact rate | -0.0031 / -0.0047 | -0.0039 | [-0.0112, +0.0007] | 0.0156 (0.0156) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0031 / +0.0016 | +0.0023 | [-0.0071, +0.0127] | 0.0312 (0.0312) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0100 / +0.0054 | -0.0023 | [-0.0138, +0.0103] | 0.0088 (0.0088) | OPEN |
| gap bias 4.0 s (m) | +0.0438 / +0.0501 | +0.0469 | [-0.0011, +0.0955] | 0.1837 (0.1837) | OPEN |
| gap bias 6.0 s (m) | +0.2256 / +0.1707 | +0.1981 | [+0.0383, +0.3479] | 1.0323 (1.0323) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0097 / -0.0111 | -0.0104 | [-0.0164, -0.0049] | 0.0159 (0.0159) | INSIDE-SPREAD |
| gap error 4.0 s (m) | -0.0154 / -0.0291 | -0.0223 | [-0.0699, +0.0189] | 0.0840 (0.0840) | OPEN |
| gap error 6.0 s (m) | +0.1307 / +0.0494 | +0.0900 | [-0.0633, +0.2164] | 0.3766 (0.3766) | OPEN |
| time-gap error 2.0 s (s) | +0.0045 / +0.0063 | +0.0054 | [-0.0194, +0.0378] | 0.0104 (0.0104) | OPEN |
| time-gap error over the scored steps (s) | +0.0791 / +0.0420 | +0.0605 | [-0.0352, +0.1718] | 0.2853 (0.2853) | OPEN |

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
| I10RK_R0C10Hb20 (0,1) | 0.0552 [0.0285, 0.0881] | 0.0366 [0.0140, 0.0647] | 0.2150 [0.1597, 0.2736] | 0.0574 [0.0060, 0.1149] | 1.7132 [1.2762, 2.1753] | 5.3978 [3.9580, 6.8593] | 0.4535 [0.4051, 0.5040] | 2.8868 [2.5140, 3.2911] | 7.5029 [6.5478, 8.5607] | 0.9438 [0.6493, 1.2872] | 3.0657 [2.2504, 3.9241] |
| I10RK_R0C10XHb20 (0,1) | 0.0472 [0.0218, 0.0794] | 0.0343 [0.0119, 0.0615] | 0.2181 [0.1653, 0.2734] | 0.0574 [0.0060, 0.1149] | 1.8434 [1.4074, 2.3049] | 5.6998 [4.3149, 7.0799] | 0.4535 [0.4051, 0.5040] | 2.9253 [2.5492, 3.3379] | 7.7182 [6.7864, 8.7757] | 0.9438 [0.6493, 1.2872] | 3.2017 [2.3638, 4.0642] |
| I10RK_R0G0Hb20 (0,1) | 0.0970 [0.0590, 0.1383] | 0.0639 [0.0329, 0.0992] | 0.1059 [0.0713, 0.1466] | 0.0574 [0.0060, 0.1149] | 0.5844 [0.1856, 1.0273] | 1.5159 [0.3292, 2.8306] | 0.4535 [0.4051, 0.5040] | 2.4516 [2.1472, 2.7882] | 6.0696 [5.2862, 6.9330] | 0.9438 [0.6493, 1.2872] | 2.1513 [1.4985, 2.8669] |
| I10RK_R0G50Hb20 (0,1) | 0.0721 [0.0426, 0.1078] | 0.0460 [0.0203, 0.0761] | 0.1729 [0.1261, 0.2216] | 0.0574 [0.0060, 0.1149] | 1.2370 [0.8185, 1.6881] | 3.8542 [2.5215, 5.1882] | 0.4535 [0.4051, 0.5040] | 2.6607 [2.3229, 3.0427] | 6.8141 [5.9651, 7.8062] | 0.9438 [0.6493, 1.2872] | 2.7518 [1.9804, 3.6041] |
| I10RK_R0Hb20 (0,1) | 0.0810 [0.0419, 0.1253] | 0.0436 [0.0205, 0.0737] | 0.1682 [0.1189, 0.2216] | 0.0574 [0.0060, 0.1149] | 1.1898 [0.7456, 1.6664] | 3.7760 [2.3276, 5.2929] | 0.4535 [0.4051, 0.5040] | 2.6742 [2.3478, 3.0404] | 7.0352 [6.1377, 8.0300] | 0.9438 [0.6493, 1.2872] | 2.5693 [1.8718, 3.3086] |
| I10RK_R0KHb20 (0,1) | 0.0952 [0.0577, 0.1364] | 0.0623 [0.0311, 0.0975] | 0.1059 [0.0717, 0.1435] | 0.0574 [0.0060, 0.1149] | 0.6240 [0.2330, 1.0626] | 1.5066 [0.2765, 2.8260] | 0.4535 [0.4051, 0.5040] | 2.4340 [2.1289, 2.7668] | 6.1010 [5.3272, 6.9552] | 0.9438 [0.6493, 1.2872] | 2.1445 [1.4936, 2.8639] |
| I10RL_R0C10XHb20 (0,1) | 0.0516 [0.0285, 0.0794] | 0.0382 [0.0142, 0.0654] | 0.2157 [0.1641, 0.2703] | 0.0597 [0.0027, 0.1226] | 1.7965 [1.3424, 2.2679] | 5.5017 [4.0939, 6.9383] | 0.4639 [0.4124, 0.5177] | 2.9476 [2.5755, 3.3560] | 7.6282 [6.7086, 8.6878] | 0.9384 [0.6505, 1.2729] | 3.1412 [2.3012, 4.0059] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
