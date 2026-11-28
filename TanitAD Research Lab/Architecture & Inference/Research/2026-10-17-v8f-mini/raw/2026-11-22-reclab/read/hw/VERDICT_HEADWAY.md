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
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RL_R0C10XHb20:I10B0_R0C10XHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RL_R0C10XHb20:I10RL_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RL_R0G50Hb20:I10RL_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RL_R0C10Hb20:I10RL_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0C10XHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0G50Hb20:I10B0_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10B0_R0C10Hb20:I10B0_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RL_R0C10XHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RL_R0Hb20:I10B0_R0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RL_R0C10Hb20:I10B0_R0C10Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RL_R0G50Hb20:I10B0_R0G50Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RL_R0G0Hb20:I10B0_R0G0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RL_R0KHb20:I10B0_R0KHb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RL_R0Hb20:I10RH_R0Hb20", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RL_R0C10XHb20:I10RH_R0C10XHb20", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10RL_R0Hb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10RL_R0C10XHb20 minus I10B0_R0C10XHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RL_R0C10XHb20 minus I10RL_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RL_R0G50Hb20 minus I10RL_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RL_R0C10Hb20 minus I10RL_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0C10XHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0G50Hb20 minus I10B0_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10B0_R0C10Hb20 minus I10B0_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RL_R0C10XHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RL_R0Hb20 minus I10B0_R0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RL_R0C10Hb20 minus I10B0_R0C10Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RL_R0G50Hb20 minus I10B0_R0G50Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RL_R0G0Hb20 minus I10B0_R0G0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RL_R0KHb20 minus I10B0_R0KHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RL_R0Hb20 minus I10RH_R0Hb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RL_R0C10XHb20 minus I10RH_R0C10XHb20 over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10RL R0C10XHb20 minus I10B0 R0C10XHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0160 / +0.0285 | +0.0222 | [+0.0070, +0.0394] | 0.0178 (0.0178) | WORSE |
| C contact rate | +0.0218 / +0.0125 | +0.0171 | [+0.0044, +0.0315] | 0.0202 (0.0202) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0717 / -0.0561 | -0.0639 | [-0.1012, -0.0363] | 0.0312 (0.0312) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0173 / -0.0414 | -0.0294 | [-0.0576, -0.0046] | 0.0243 (0.0243) | BETTER |
| gap bias 4.0 s (m) | -0.3154 / -0.4679 | -0.3917 | [-0.5697, -0.2274] | 0.1890 (0.1890) | BETTER |
| gap bias 6.0 s (m) | -1.8950 / -2.0252 | -1.9601 | [-2.6387, -1.3878] | 1.1265 (1.1265) | BETTER |
| gap error 2.0 s (m) | +0.0328 / +0.0317 | +0.0322 | [+0.0172, +0.0493] | 0.0145 (0.0145) | WORSE |
| gap error 4.0 s (m) | +0.0583 / -0.1028 | -0.0223 | [-0.1753, +0.1345] | 0.0990 (0.0990) | OPEN |
| gap error 6.0 s (m) | -0.8459 / -1.1911 | -1.0185 | [-1.6494, -0.4748] | 0.4500 (0.4500) | BETTER |
| time-gap error 2.0 s (s) | +0.0157 / +0.0628 | +0.0392 | [-0.0130, +0.1064] | 0.0086 (0.0086) | OPEN |
| time-gap error over the scored steps (s) | -0.7455 / -0.6755 | -0.7105 | [-1.0236, -0.4494] | 0.1629 (0.1629) | BETTER |

## I10RL R0C10XHb20 minus I10RL R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0409 / -0.0463 | -0.0436 | [-0.0664, -0.0218] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0218 / -0.0280 | -0.0249 | [-0.0418, -0.0116] | 0.0202 (0.0202) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0903 / +0.1246 | +0.1075 | [+0.0741, +0.1478] | 0.0312 (0.0312) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0243 (0.0243) | OPEN |
| gap bias 4.0 s (m) | +1.0552 / +1.2658 | +1.1605 | [+0.9710, +1.3787] | 0.1890 (0.1890) | WORSE |
| gap bias 6.0 s (m) | +3.3994 / +4.5130 | +3.9562 | [+3.1166, +4.9304] | 1.1265 (1.1265) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0145 (0.0145) | OPEN |
| gap error 4.0 s (m) | +0.4911 / +0.4722 | +0.4816 | [+0.3061, +0.6588] | 0.0990 (0.0990) | WORSE |
| gap error 6.0 s (m) | +1.3010 / +1.6435 | +1.4722 | [+0.8120, +2.2317] | 0.4500 (0.4500) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0086 (0.0086) | OPEN |
| time-gap error over the scored steps (s) | +0.8184 / +1.1482 | +0.9833 | [+0.5797, +1.4465] | 0.1629 (0.1629) | WORSE |

## I10RL R0G50Hb20 minus I10RL R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0142 / -0.0320 | -0.0231 | [-0.0453, -0.0070] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0140 / -0.0187 | -0.0164 | [-0.0282, -0.0067] | 0.0202 (0.0202) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0607 / +0.0639 | +0.0623 | [+0.0414, +0.0835] | 0.0312 (0.0312) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0243 (0.0243) | OPEN |
| gap bias 4.0 s (m) | +0.5691 / +0.6793 | +0.6242 | [+0.5213, +0.7411] | 0.1890 (0.1890) | WORSE |
| gap bias 6.0 s (m) | +1.9559 / +2.6783 | +2.3171 | [+1.7819, +2.9381] | 1.1265 (1.1265) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0145 (0.0145) | OPEN |
| gap error 4.0 s (m) | +0.2325 / +0.1793 | +0.2059 | [+0.0975, +0.3113] | 0.0990 (0.0990) | WORSE |
| gap error 6.0 s (m) | +0.7321 / +0.7266 | +0.7293 | [+0.2926, +1.1705] | 0.4500 (0.4500) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0086 (0.0086) | OPEN |
| time-gap error over the scored steps (s) | +0.4868 / +0.8495 | +0.6681 | [+0.3051, +1.0847] | 0.1629 (0.1629) | WORSE |

## I10RL R0C10Hb20 minus I10RL R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0071 / -0.0160 | -0.0116 | [-0.0220, -0.0020] | 0.0178 (0.0178) | INSIDE-SPREAD |
| C contact rate | -0.0016 / -0.0062 | -0.0039 | [-0.0117, +0.0032] | 0.0202 (0.0202) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0234 / +0.0576 | +0.0405 | [+0.0177, +0.0705] | 0.0312 (0.0312) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0243 (0.0243) | OPEN |
| gap bias 4.0 s (m) | +0.3335 / +0.4883 | +0.4109 | [+0.3050, +0.5388] | 0.1890 (0.1890) | WORSE |
| gap bias 6.0 s (m) | +0.9242 / +1.5162 | +1.2202 | [+0.7524, +1.7129] | 1.1265 (1.1265) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0145 (0.0145) | OPEN |
| gap error 4.0 s (m) | +0.1949 / +0.2309 | +0.2129 | [+0.1447, +0.2929] | 0.0990 (0.0990) | WORSE |
| gap error 6.0 s (m) | +0.4237 / +0.6988 | +0.5613 | [+0.2740, +0.8743] | 0.4500 (0.4500) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0086 (0.0086) | OPEN |
| time-gap error over the scored steps (s) | +0.1424 / +0.1626 | +0.1525 | [-0.0354, +0.3479] | 0.1629 (0.1629) | OPEN |

## I10B0 R0C10XHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0516 / -0.0747 | -0.0632 | [-0.0977, -0.0308] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0405 / -0.0374 | -0.0389 | [-0.0614, -0.0194] | 0.0202 (0.0202) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.1511 / +0.1791 | +0.1651 | [+0.1215, +0.2139] | 0.0312 (0.0312) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0243 (0.0243) | OPEN |
| gap bias 4.0 s (m) | +1.3409 / +1.5946 | +1.4678 | [+1.2207, +1.7382] | 0.1890 (0.1890) | WORSE |
| gap bias 6.0 s (m) | +5.1910 / +6.2410 | +5.7160 | [+4.6498, +6.9298] | 1.1265 (1.1265) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0145 (0.0145) | OPEN |
| gap error 4.0 s (m) | +0.5314 / +0.6764 | +0.6039 | [+0.3704, +0.8402] | 0.0990 (0.0990) | WORSE |
| gap error 6.0 s (m) | +2.2039 / +2.9716 | +2.5877 | [+1.5457, +3.7505] | 0.4500 (0.4500) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0086 (0.0086) | OPEN |
| time-gap error over the scored steps (s) | +1.5846 / +1.7639 | +1.6743 | [+1.1307, +2.2340] | 0.1629 (0.1629) | WORSE |

## I10B0 R0G50Hb20 minus I10B0 R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0285 / -0.0391 | -0.0338 | [-0.0540, -0.0149] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0202 / -0.0202 | -0.0202 | [-0.0352, -0.0082] | 0.0202 (0.0202) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0561 / +0.0888 | +0.0724 | [+0.0441, +0.1092] | 0.0312 (0.0312) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0243 (0.0243) | OPEN |
| gap bias 4.0 s (m) | +0.6755 / +0.8065 | +0.7410 | [+0.6149, +0.8796] | 0.1890 (0.1890) | WORSE |
| gap bias 6.0 s (m) | +2.5213 / +3.3152 | +2.9182 | [+2.2914, +3.6283] | 1.1265 (1.1265) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0145 (0.0145) | OPEN |
| gap error 4.0 s (m) | +0.2169 / +0.2442 | +0.2305 | [+0.1141, +0.3530] | 0.0990 (0.0990) | WORSE |
| gap error 6.0 s (m) | +0.8243 / +1.1466 | +0.9854 | [+0.4232, +1.5805] | 0.4500 (0.4500) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0086 (0.0086) | OPEN |
| time-gap error over the scored steps (s) | +0.8436 / +1.0007 | +0.9222 | [+0.5077, +1.3596] | 0.1629 (0.1629) | WORSE |

## I10B0 R0C10Hb20 minus I10B0 R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0249 / -0.0302 | -0.0276 | [-0.0440, -0.0132] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0218 / -0.0171 | -0.0195 | [-0.0334, -0.0077] | 0.0202 (0.0202) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0950 / +0.1012 | +0.0981 | [+0.0698, +0.1293] | 0.0312 (0.0312) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0243 (0.0243) | OPEN |
| gap bias 4.0 s (m) | +0.6707 / +0.7715 | +0.7211 | [+0.6054, +0.8476] | 0.1890 (0.1890) | WORSE |
| gap bias 6.0 s (m) | +2.7210 / +3.1187 | +2.9198 | [+2.3181, +3.5892] | 1.1265 (1.1265) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0145 (0.0145) | OPEN |
| gap error 4.0 s (m) | +0.2994 / +0.3979 | +0.3486 | [+0.2279, +0.4805] | 0.0990 (0.0990) | WORSE |
| gap error 6.0 s (m) | +1.4038 / +2.0636 | +1.7337 | [+1.0930, +2.4937] | 0.4500 (0.4500) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0086 (0.0086) | OPEN |
| time-gap error over the scored steps (s) | +0.6962 / +0.6219 | +0.6590 | [+0.3831, +0.9571] | 0.1629 (0.1629) | WORSE |

## I10RL R0C10XHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0356 / -0.0463 | -0.0409 | [-0.0627, -0.0210] | 0.0178 (0.0178) | BETTER |
| C contact rate | -0.0187 / -0.0249 | -0.0218 | [-0.0387, -0.0082] | 0.0202 (0.0202) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0794 / +0.1231 | +0.1012 | [+0.0643, +0.1469] | 0.0312 (0.0312) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0173 / -0.0414 | -0.0294 | [-0.0576, -0.0046] | 0.0243 (0.0243) | BETTER |
| gap bias 4.0 s (m) | +1.0255 / +1.1267 | +1.0761 | [+0.9101, +1.2666] | 0.1890 (0.1890) | WORSE |
| gap bias 6.0 s (m) | +3.2959 / +4.2158 | +3.7559 | [+2.9573, +4.6612] | 1.1265 (1.1265) | WORSE |
| gap error 2.0 s (m) | +0.0328 / +0.0317 | +0.0322 | [+0.0172, +0.0493] | 0.0145 (0.0145) | WORSE |
| gap error 4.0 s (m) | +0.5896 / +0.5736 | +0.5816 | [+0.4009, +0.7640] | 0.0990 (0.0990) | WORSE |
| gap error 6.0 s (m) | +1.3580 / +1.7805 | +1.5692 | [+0.8841, +2.3648] | 0.4500 (0.4500) | WORSE |
| time-gap error 2.0 s (s) | +0.0157 / +0.0628 | +0.0392 | [-0.0130, +0.1064] | 0.0086 (0.0086) | OPEN |
| time-gap error over the scored steps (s) | +0.8391 / +1.0884 | +0.9638 | [+0.5881, +1.3726] | 0.1629 (0.1629) | WORSE |

## I10RL R0Hb20 minus I10B0 R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0231 / +0.0249 | +0.0240 | [+0.0090, +0.0408] | 0.0178 (0.0178) | WORSE |
| C contact rate | +0.0249 / +0.0140 | +0.0195 | [+0.0056, +0.0384] | 0.0202 (0.0202) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0249 / -0.0327 | -0.0288 | [-0.0476, -0.0129] | 0.0312 (0.0312) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0173 / -0.0414 | -0.0294 | [-0.0576, -0.0046] | 0.0243 (0.0243) | BETTER |
| gap bias 4.0 s (m) | -0.2679 / -0.3521 | -0.3100 | [-0.4445, -0.1765] | 0.1890 (0.1890) | BETTER |
| gap bias 6.0 s (m) | -1.5746 / -1.7493 | -1.6619 | [-2.1223, -1.2155] | 1.1265 (1.1265) | BETTER |
| gap error 2.0 s (m) | +0.0328 / +0.0317 | +0.0322 | [+0.0172, +0.0493] | 0.0145 (0.0145) | WORSE |
| gap error 4.0 s (m) | +0.1032 / -0.0182 | +0.0425 | [-0.0768, +0.1788] | 0.0990 (0.0990) | OPEN |
| gap error 6.0 s (m) | -0.1338 / -0.5559 | -0.3449 | [-0.8220, +0.1444] | 0.4500 (0.4500) | OPEN |
| time-gap error 2.0 s (s) | +0.0157 / +0.0628 | +0.0392 | [-0.0130, +0.1064] | 0.0086 (0.0086) | OPEN |
| time-gap error over the scored steps (s) | -0.5404 / -0.6500 | -0.5952 | [-0.8738, -0.3381] | 0.1629 (0.1629) | BETTER |

## I10RL R0C10Hb20 minus I10B0 R0C10Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0320 / +0.0249 | +0.0285 | [+0.0102, +0.0526] | 0.0178 (0.0178) | WORSE |
| C contact rate | +0.0296 / +0.0187 | +0.0241 | [+0.0091, +0.0424] | 0.0202 (0.0202) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0810 / -0.0717 | -0.0763 | [-0.1122, -0.0474] | 0.0312 (0.0312) | BETTER |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0173 / -0.0414 | -0.0294 | [-0.0576, -0.0046] | 0.0243 (0.0243) | BETTER |
| gap bias 4.0 s (m) | -0.4730 / -0.5735 | -0.5232 | [-0.6994, -0.3646] | 0.1890 (0.1890) | BETTER |
| gap bias 6.0 s (m) | -2.5615 / -2.5810 | -2.5713 | [-3.3172, -1.9032] | 1.1265 (1.1265) | BETTER |
| gap error 2.0 s (m) | +0.0328 / +0.0317 | +0.0322 | [+0.0172, +0.0493] | 0.0145 (0.0145) | WORSE |
| gap error 4.0 s (m) | +0.0128 / -0.1129 | -0.0501 | [-0.1960, +0.1068] | 0.0990 (0.0990) | OPEN |
| gap error 6.0 s (m) | -1.0819 / -1.5591 | -1.3205 | [-2.0941, -0.6296] | 0.4500 (0.4500) | BETTER |
| time-gap error 2.0 s (s) | +0.0157 / +0.0628 | +0.0392 | [-0.0130, +0.1064] | 0.0086 (0.0086) | OPEN |
| time-gap error over the scored steps (s) | -0.9031 / -0.6254 | -0.7643 | [-1.1926, -0.4500] | 0.1629 (0.1629) | BETTER |

## I10RL R0G50Hb20 minus I10B0 R0G50Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0142 / +0.0107 | +0.0125 | [+0.0015, +0.0258] | 0.0178 (0.0178) | INSIDE-SPREAD |
| C contact rate | +0.0093 / +0.0078 | +0.0086 | [+0.0021, +0.0169] | 0.0202 (0.0202) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0093 / -0.0280 | -0.0187 | [-0.0400, -0.0007] | 0.0312 (0.0312) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0173 / -0.0414 | -0.0294 | [-0.0576, -0.0046] | 0.0243 (0.0243) | BETTER |
| gap bias 4.0 s (m) | -0.1358 / -0.2903 | -0.2130 | [-0.3618, -0.0720] | 0.1890 (0.1890) | BETTER |
| gap bias 6.0 s (m) | -0.7646 / -0.9786 | -0.8716 | [-1.2136, -0.5651] | 1.1265 (1.1265) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0328 / +0.0317 | +0.0322 | [+0.0172, +0.0493] | 0.0145 (0.0145) | WORSE |
| gap error 4.0 s (m) | +0.1173 / +0.0540 | +0.0857 | [-0.0016, +0.1775] | 0.0990 (0.0990) | OPEN |
| gap error 6.0 s (m) | -0.1018 / -0.1943 | -0.1480 | [-0.3968, +0.0969] | 0.4500 (0.4500) | OPEN |
| time-gap error 2.0 s (s) | +0.0157 / +0.0628 | +0.0392 | [-0.0130, +0.1064] | 0.0086 (0.0086) | OPEN |
| time-gap error over the scored steps (s) | -0.3494 / -0.1661 | -0.2577 | [-0.4690, -0.0806] | 0.1629 (0.1629) | BETTER |

## I10RL R0G0Hb20 minus I10B0 R0G0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0036 | +0.0018 | [-0.0047, +0.0108] | 0.0178 (0.0178) | OPEN |
| C contact rate | +0.0031 / +0.0062 | +0.0047 | [+0.0000, +0.0105] | 0.0202 (0.0202) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0140 / -0.0031 | -0.0086 | [-0.0228, +0.0035] | 0.0312 (0.0312) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0173 / -0.0414 | -0.0294 | [-0.0576, -0.0046] | 0.0243 (0.0243) | BETTER |
| gap bias 4.0 s (m) | -0.0293 / -0.1630 | -0.0961 | [-0.2188, +0.0160] | 0.1890 (0.1890) | OPEN |
| gap bias 6.0 s (m) | -0.1993 / -0.3416 | -0.2705 | [-0.4804, -0.0835] | 1.1265 (1.1265) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0328 / +0.0317 | +0.0322 | [+0.0172, +0.0493] | 0.0145 (0.0145) | WORSE |
| gap error 4.0 s (m) | +0.1017 / +0.1189 | +0.1103 | [+0.0583, +0.1646] | 0.0990 (0.0990) | WORSE |
| gap error 6.0 s (m) | -0.0096 / +0.2258 | +0.1081 | [-0.0981, +0.3131] | 0.4500 (0.4500) | OPEN |
| time-gap error 2.0 s (s) | +0.0157 / +0.0628 | +0.0392 | [-0.0130, +0.1064] | 0.0086 (0.0086) | OPEN |
| time-gap error over the scored steps (s) | +0.0075 / -0.0148 | -0.0037 | [-0.0732, +0.0584] | 0.1629 (0.1629) | OPEN |

## I10RL R0KHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0053 / +0.0000 | +0.0027 | [-0.0027, +0.0096] | 0.0178 (0.0178) | OPEN |
| C contact rate | +0.0031 / +0.0031 | +0.0031 | [-0.0004, +0.0071] | 0.0202 (0.0202) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0109 / -0.0016 | -0.0062 | [-0.0188, +0.0046] | 0.0312 (0.0312) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0173 / -0.0414 | -0.0294 | [-0.0576, -0.0046] | 0.0243 (0.0243) | BETTER |
| gap bias 4.0 s (m) | -0.0297 / -0.1391 | -0.0844 | [-0.1934, +0.0169] | 0.1890 (0.1890) | OPEN |
| gap bias 6.0 s (m) | -0.1034 / -0.2973 | -0.2003 | [-0.4237, -0.0048] | 1.1265 (1.1265) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0328 / +0.0317 | +0.0322 | [+0.0172, +0.0493] | 0.0145 (0.0145) | WORSE |
| gap error 4.0 s (m) | +0.0985 / +0.1014 | +0.1000 | [+0.0493, +0.1547] | 0.0990 (0.0990) | WORSE |
| gap error 6.0 s (m) | +0.0570 / +0.1369 | +0.0970 | [-0.0324, +0.2379] | 0.4500 (0.4500) | OPEN |
| time-gap error 2.0 s (s) | +0.0157 / +0.0628 | +0.0392 | [-0.0130, +0.1064] | 0.0086 (0.0086) | OPEN |
| time-gap error over the scored steps (s) | +0.0208 / -0.0598 | -0.0195 | [-0.1195, +0.0592] | 0.1629 (0.1629) | OPEN |

## I10RL R0Hb20 minus I10RH R0Hb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0018 / +0.0053 | +0.0036 | [-0.0055, +0.0126] | 0.0178 (0.0178) | OPEN |
| C contact rate | +0.0078 / +0.0000 | +0.0039 | [-0.0027, +0.0139] | 0.0202 (0.0202) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0093 / +0.0125 | +0.0109 | [+0.0009, +0.0222] | 0.0312 (0.0312) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0130 / -0.0018 | +0.0056 | [-0.0065, +0.0168] | 0.0243 (0.0243) | OPEN |
| gap bias 4.0 s (m) | -0.0378 / +0.0127 | -0.0126 | [-0.0659, +0.0440] | 0.1890 (0.1890) | OPEN |
| gap bias 6.0 s (m) | -0.1284 / +0.0609 | -0.0338 | [-0.2294, +0.1743] | 1.1265 (1.1265) | OPEN |
| gap error 2.0 s (m) | +0.0012 / +0.0098 | +0.0055 | [-0.0027, +0.0127] | 0.0145 (0.0145) | OPEN |
| gap error 4.0 s (m) | -0.0148 / +0.0223 | +0.0037 | [-0.0378, +0.0447] | 0.0990 (0.0990) | OPEN |
| gap error 6.0 s (m) | +0.0339 / +0.1305 | +0.0822 | [-0.0606, +0.2229] | 0.4500 (0.4500) | OPEN |
| time-gap error 2.0 s (s) | -0.0115 / +0.0152 | +0.0019 | [-0.0216, +0.0285] | 0.0086 (0.0086) | OPEN |
| time-gap error over the scored steps (s) | -0.0054 / -0.0858 | -0.0456 | [-0.1622, +0.0749] | 0.1629 (0.1629) | OPEN |

## I10RL R0C10XHb20 minus I10RH R0C10XHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0018 / +0.0107 | +0.0044 | [-0.0065, +0.0155] | 0.0178 (0.0178) | OPEN |
| C contact rate | +0.0047 / +0.0000 | +0.0023 | [-0.0016, +0.0072] | 0.0202 (0.0202) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0031 / +0.0078 | +0.0023 | [-0.0112, +0.0162] | 0.0312 (0.0312) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0130 / -0.0018 | +0.0056 | [-0.0065, +0.0168] | 0.0243 (0.0243) | OPEN |
| gap bias 4.0 s (m) | -0.0173 / +0.0326 | +0.0077 | [-0.0459, +0.0641] | 0.1890 (0.1890) | OPEN |
| gap bias 6.0 s (m) | -0.0674 / +0.1299 | +0.0313 | [-0.1461, +0.2180] | 1.1265 (1.1265) | OPEN |
| gap error 2.0 s (m) | +0.0012 / +0.0098 | +0.0055 | [-0.0027, +0.0127] | 0.0145 (0.0145) | OPEN |
| gap error 4.0 s (m) | -0.0128 / +0.0589 | +0.0230 | [-0.0379, +0.0871] | 0.0990 (0.0990) | OPEN |
| gap error 6.0 s (m) | +0.0327 / +0.1831 | +0.1079 | [-0.0386, +0.2637] | 0.4500 (0.4500) | OPEN |
| time-gap error 2.0 s (s) | -0.0115 / +0.0152 | +0.0019 | [-0.0216, +0.0285] | 0.0086 (0.0086) | OPEN |
| time-gap error over the scored steps (s) | -0.0444 / +0.0379 | -0.0033 | [-0.1222, +0.1094] | 0.1629 (0.1629) | OPEN |

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
| I10RL_R0C10Hb20 (0,1) | 0.0614 [0.0319, 0.0993] | 0.0436 [0.0172, 0.0748] | 0.2118 [0.1609, 0.2663] | 0.0597 [0.0027, 0.1226] | 1.6288 [1.1709, 2.0891] | 5.0761 [3.5755, 6.5695] | 0.4639 [0.4124, 0.5177] | 2.9093 [2.5560, 3.2933] | 7.4363 [6.5179, 8.4758] | 0.9384 [0.6505, 1.2729] | 2.9976 [2.1691, 3.8570] |
| I10RL_R0C10XHb20 (0,1) | 0.0516 [0.0285, 0.0794] | 0.0382 [0.0142, 0.0654] | 0.2157 [0.1641, 0.2703] | 0.0597 [0.0027, 0.1226] | 1.7965 [1.3424, 2.2679] | 5.5017 [4.0939, 6.9383] | 0.4639 [0.4124, 0.5177] | 2.9476 [2.5755, 3.3560] | 7.6282 [6.7086, 8.6878] | 0.9384 [0.6505, 1.2729] | 3.1412 [2.3012, 4.0059] |
| I10RL_R0G0Hb20 (0,1) | 0.0961 [0.0580, 0.1377] | 0.0639 [0.0331, 0.0990] | 0.1090 [0.0725, 0.1498] | 0.0597 [0.0027, 0.1226] | 0.5937 [0.1752, 1.0443] | 1.5388 [0.3367, 2.8829] | 0.4639 [0.4124, 0.5177] | 2.4905 [2.1756, 2.8273] | 6.1457 [5.3438, 7.0159] | 0.9384 [0.6505, 1.2729] | 2.1770 [1.5215, 2.9138] |
| I10RL_R0G50Hb20 (0,1) | 0.0730 [0.0423, 0.1087] | 0.0475 [0.0205, 0.0793] | 0.1713 [0.1261, 0.2202] | 0.0597 [0.0027, 0.1226] | 1.2179 [0.7942, 1.6849] | 3.8559 [2.4483, 5.2204] | 0.4639 [0.4124, 0.5177] | 2.6964 [2.3606, 3.0819] | 6.8751 [6.0024, 7.8848] | 0.9384 [0.6505, 1.2729] | 2.8451 [2.0287, 3.7328] |
| I10RL_R0Hb20 (0,1) | 0.0854 [0.0471, 0.1306] | 0.0475 [0.0225, 0.0800] | 0.1745 [0.1249, 0.2275] | 0.0597 [0.0027, 0.1226] | 1.1475 [0.6897, 1.6350] | 3.6258 [2.1166, 5.1470] | 0.4639 [0.4124, 0.5177] | 2.6993 [2.3723, 3.0638] | 7.0319 [6.1337, 8.0468] | 0.9384 [0.6505, 1.2729] | 2.5358 [1.8404, 3.3034] |
| I10RL_R0KHb20 (0,1) | 0.0952 [0.0571, 0.1366] | 0.0631 [0.0321, 0.0970] | 0.1083 [0.0718, 0.1493] | 0.0597 [0.0027, 0.1226] | 0.6360 [0.2254, 1.0800] | 1.5455 [0.2951, 2.8713] | 0.4639 [0.4124, 0.5177] | 2.4659 [2.1571, 2.7962] | 6.1560 [5.3781, 7.0003] | 0.9384 [0.6505, 1.2729] | 2.1579 [1.5091, 2.8907] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
