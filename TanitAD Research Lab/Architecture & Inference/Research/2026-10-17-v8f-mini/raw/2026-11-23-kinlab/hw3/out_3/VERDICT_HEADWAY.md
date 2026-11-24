# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: F:/TanitAD-scratch/strat1/read/inputs/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 14, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09891171799189782, "A_hi": 0.16379738797148666, "spread_max": 0.003558718861209967, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0KHb20:I10B0_R0KHb20", "seeds": [0, 1, 2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "I10RH_R0C10XHb20:I10B0_R0C10XHb20", "seeds": [0, 1, 2]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm I10UEA_R0C10XHb20, seeds present [0, 1] of requested [0, 1]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, I10RH_R0KHb20 minus I10B0_R0KHb20 over seeds [0, 1, 2] of [0, 1, 2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, I10RH_R0C10XHb20 minus I10B0_R0C10XHb20 over seeds [0, 1, 2] of [0, 1, 2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## I10RH R0KHb20 minus I10B0 R0KHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0071 / -0.0018 / +0.0071 | +0.0042 | [-0.0017, +0.0107] | 0.0036 (0.0036) | OPEN |
| C contact rate | -0.0016 / +0.0031 / +0.0031 | +0.0016 | [-0.0027, +0.0055] | 0.0062 (0.0062) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0109 / -0.0062 / -0.0062 | -0.0078 | [-0.0145, -0.0014] | 0.0561 (0.0561) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0303 / -0.0396 / -0.0289 | -0.0329 | [-0.0518, -0.0162] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.0565 / -0.1507 / -0.0917 | -0.0996 | [-0.1717, -0.0325] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.1106 / -0.3363 / -0.2585 | -0.2351 | [-0.3959, -0.0799] | 1.8616 (1.8616) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0316 / +0.0219 / +0.0232 | +0.0256 | [+0.0137, +0.0395] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.0904 / +0.0547 / +0.0895 | +0.0782 | [+0.0326, +0.1265] | 0.4100 (0.4100) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.0379 / +0.0351 / +0.0347 | +0.0359 | [-0.0550, +0.1336] | 1.5504 (1.5504) | OPEN |
| time-gap error 2.0 s (s) | +0.0272 / +0.0476 / +0.0869 | +0.0539 | [+0.0064, +0.1143] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | +0.0146 / -0.0607 / -0.0558 | -0.0340 | [-0.1370, +0.0519] | 0.2075 (0.2075) | OPEN |

## I10RH R0C10XHb20 minus I10B0 R0C10XHb20

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0178 / +0.0178 / +0.0125 | +0.0160 | [+0.0070, +0.0266] | 0.0036 (0.0036) | WORSE |
| C contact rate | +0.0171 / +0.0125 / +0.0078 | +0.0125 | [+0.0035, +0.0230] | 0.0062 (0.0062) | WORSE |
| F falls behind by more than 10 m at the last scored step | -0.0685 / -0.0639 / -0.0343 | -0.0556 | [-0.0863, -0.0276] | 0.0561 (0.0561) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0303 / -0.0396 / -0.0289 | -0.0329 | [-0.0518, -0.0162] | 0.0490 (0.0490) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | -0.2982 / -0.5005 / -0.2610 | -0.3532 | [-0.5268, -0.2064] | 0.6885 (0.6885) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -1.8277 / -2.1552 / -1.3479 | -1.7769 | [-2.3840, -1.2315] | 1.8616 (1.8616) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0316 / +0.0219 / +0.0232 | +0.0256 | [+0.0137, +0.0395] | 0.0069 (0.0069) | WORSE |
| gap error 4.0 s (m) | +0.0711 / -0.1617 / +0.0519 | -0.0129 | [-0.1677, +0.1257] | 0.4100 (0.4100) | OPEN |
| gap error 6.0 s (m) | -0.8786 / -1.3742 / -0.5772 | -0.9433 | [-1.5562, -0.4073] | 1.5504 (1.5504) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0272 / +0.0476 / +0.0869 | +0.0539 | [+0.0064, +0.1143] | 0.0330 (0.0330) | WORSE |
| time-gap error over the scored steps (s) | -0.7011 / -0.7134 / -0.6156 | -0.6767 | [-0.9282, -0.4141] | 0.2075 (0.2075) | BETTER |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I10B0_R0C10XHb20 (0,1,2) | 0.0338 [0.0160, 0.0561] | 0.0280 [0.0120, 0.0491] | 0.2534 [0.1963, 0.3111] | 0.0825 [0.0440, 0.1205] | 1.9786 [1.5086, 2.4609] | 6.8855 [5.4947, 8.3163] | 0.4345 [0.4011, 0.4717] | 2.8420 [2.4993, 3.1894] | 8.2678 [7.2504, 9.3259] | 0.8694 [0.6502, 1.1299] | 3.6743 [2.8697, 4.5360] |
| I10B0_R0KHb20 (0,1,2) | 0.0907 [0.0619, 0.1235] | 0.0613 [0.0346, 0.0894] | 0.1075 [0.0778, 0.1416] | 0.0825 [0.0440, 0.1205] | 0.6351 [0.3160, 0.9720] | 1.5603 [0.5663, 2.6372] | 0.4345 [0.4011, 0.4717] | 2.3281 [2.1008, 2.5863] | 5.9715 [5.3593, 6.6863] | 0.8694 [0.6502, 1.1299] | 2.1497 [1.5890, 2.7973] |
| I10RH_R0C10XHb20 (0,1,2) | 0.0498 [0.0281, 0.0766] | 0.0405 [0.0192, 0.0656] | 0.1978 [0.1517, 0.2441] | 0.0496 [0.0035, 0.0975] | 1.6254 [1.1913, 2.0746] | 5.1086 [3.9325, 6.3203] | 0.4600 [0.4209, 0.5043] | 2.8291 [2.4967, 3.1936] | 7.3244 [6.5338, 8.1804] | 0.9233 [0.6837, 1.2096] | 2.9976 [2.3166, 3.7679] |
| I10RH_R0KHb20 (0,1,2) | 0.0949 [0.0634, 0.1309] | 0.0628 [0.0358, 0.0911] | 0.0997 [0.0720, 0.1324] | 0.0496 [0.0035, 0.0975] | 0.5355 [0.1917, 0.8968] | 1.3252 [0.2944, 2.4492] | 0.4600 [0.4209, 0.5043] | 2.4063 [2.1616, 2.6819] | 6.0073 [5.3872, 6.7301] | 0.9233 [0.6837, 1.2096] | 2.1157 [1.5754, 2.7452] |
| I10UEA_R0C10XHb20 (0,1) | 0.0623 [0.0388, 0.0910] | 0.0280 [0.0123, 0.0462] | 0.3910 [0.3191, 0.4592] | -0.2236 [-0.2757, -0.1682] | 3.0145 [2.3455, 3.6779] | 10.3479 [8.3469, 12.6031] | 0.4729 [0.4299, 0.5219] | 3.5797 [3.0625, 4.1116] | 11.7414 [10.0248, 13.7045] | 1.0435 [0.7059, 1.4212] | 4.8571 [3.8136, 5.9361] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
