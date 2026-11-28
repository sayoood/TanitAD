# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: F:/TanitAD-scratch/v8proof/rhacc10k/pins/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 12, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09891171799189782, "A_hi": 0.16379738797148666, "spread_max": 0.005338078291814943, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "H_RLcK:U_RLcK", "seeds": [0, 1, 2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "H_RQ50K:U_RQ50K", "seeds": [0, 1, 2]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm U_RLcK, seeds present [0, 1, 2] of requested [0, 1, 2]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, H_RLcK minus U_RLcK over seeds [0, 1, 2] of [0, 1, 2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, H_RQ50K minus U_RQ50K over seeds [0, 1, 2] of [0, 1, 2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## H RLcK minus U RLcK

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0018 / +0.0089 / +0.0036 | +0.0036 | [-0.0055, +0.0124] | 0.0053 (0.0053) | OPEN |
| C contact rate | +0.0000 / +0.0000 / -0.0016 | -0.0005 | [-0.0022, +0.0000] | 0.0016 (0.0016) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0249 / -0.0187 / +0.0031 | +0.0031 | [-0.0200, +0.0262] | 0.0358 (0.0296) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2877 / +0.2129 / +0.2107 | +0.2371 | [+0.1925, +0.2881] | 0.0607 (0.0537) | WORSE |
| gap bias 4.0 s (m) | +0.5673 / +0.0080 / +0.2390 | +0.2714 | [+0.0103, +0.5526] | 0.4477 (0.3725) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.9997 / -0.4600 / +0.1434 | +0.2277 | [-0.4514, +0.9393] | 1.1182 (0.9461) | OPEN |
| gap error 2.0 s (m) | +0.0050 / +0.0036 / +0.0123 | +0.0070 | [-0.0129, +0.0280] | 0.0092 (0.0048) | OPEN |
| gap error 4.0 s (m) | +0.2945 / +0.0335 / +0.0477 | +0.1253 | [+0.0002, +0.2943] | 0.1594 (0.0930) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.3749 / -0.1364 / -0.1821 | +0.0188 | [-0.2829, +0.3675] | 0.4287 (0.2775) | OPEN |
| time-gap error 2.0 s (s) | -0.1007 / -0.1374 / -0.1305 | -0.1229 | [-0.2704, -0.0035] | 0.0461 (0.0327) | BETTER |
| time-gap error over the scored steps (s) | +0.1268 / -0.3068 / -0.2852 | -0.1551 | [-0.4512, +0.1602] | 0.3014 (0.2062) | OPEN |

## H RQ50K minus U RQ50K

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0018 / +0.0036 / +0.0089 | +0.0047 | [-0.0035, +0.0139] | 0.0053 (0.0053) | OPEN |
| C contact rate | +0.0062 / +0.0062 / +0.0093 | +0.0073 | [+0.0022, +0.0143] | 0.0016 (0.0016) | WORSE |
| F falls behind by more than 10 m at the last scored step | +0.0140 / -0.0234 / +0.0016 | -0.0026 | [-0.0241, +0.0170] | 0.0358 (0.0296) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2494 / +0.1991 / +0.1745 | +0.2076 | [+0.1656, +0.2555] | 0.0607 (0.0537) | WORSE |
| gap bias 4.0 s (m) | +0.4799 / -0.0008 / +0.1984 | +0.2259 | [-0.0069, +0.4688] | 0.4477 (0.3725) | OPEN |
| gap bias 6.0 s (m) | +0.8280 / -0.4813 / +0.0502 | +0.1323 | [-0.4897, +0.7738] | 1.1182 (0.9461) | OPEN |
| gap error 2.0 s (m) | +0.0155 / +0.0110 / +0.0214 | +0.0160 | [-0.0027, +0.0345] | 0.0092 (0.0048) | OPEN |
| gap error 4.0 s (m) | +0.2795 / +0.0687 / +0.0964 | +0.1482 | [+0.0373, +0.2905] | 0.1594 (0.0930) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.2768 / -0.0949 / -0.1569 | +0.0083 | [-0.2612, +0.2928] | 0.4287 (0.2775) | OPEN |
| time-gap error 2.0 s (s) | -0.1073 / -0.1353 / -0.1258 | -0.1228 | [-0.2680, -0.0044] | 0.0461 (0.0327) | BETTER |
| time-gap error over the scored steps (s) | +0.1261 / -0.2498 / -0.2159 | -0.1132 | [-0.3766, +0.1724] | 0.3014 (0.2062) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H_RLcK (0,1,2) | 0.0409 [0.0212, 0.0642] | 0.0000 [0.0000, 0.0000] | 0.1511 [0.1099, 0.1977] | 0.5144 [0.2920, 0.7679] | 1.7782 [1.2417, 2.3763] | 3.7368 [2.7085, 4.7992] | 0.8017 [0.5912, 1.0434] | 2.7591 [2.3148, 3.2701] | 5.6843 [4.9906, 6.4977] | 0.9692 [0.7264, 1.2570] | 2.1041 [1.5574, 2.7253] |
| H_RQ50K (0,1,2) | 0.0747 [0.0464, 0.1060] | 0.0363 [0.0171, 0.0575] | 0.1636 [0.1222, 0.2056] | 0.6197 [0.3730, 0.8805] | 1.9001 [1.3104, 2.5168] | 3.8743 [2.7267, 5.0904] | 0.9244 [0.7043, 1.1591] | 3.0874 [2.6423, 3.5753] | 6.3648 [5.6273, 7.2000] | 1.0783 [0.8082, 1.3770] | 2.3566 [1.7622, 3.0473] |
| U_RLcK (0,1,2) | 0.0374 [0.0199, 0.0568] | 0.0005 [0.0000, 0.0022] | 0.1480 [0.1053, 0.1956] | 0.2773 [0.0427, 0.5439] | 1.5068 [0.9344, 2.1183] | 3.5091 [2.3635, 4.6444] | 0.7948 [0.5900, 1.0296] | 2.6338 [2.2017, 3.1470] | 5.6655 [4.9544, 6.5091] | 1.0920 [0.8200, 1.4077] | 2.2591 [1.7071, 2.8892] |
| U_RQ50K (0,1,2) | 0.0700 [0.0445, 0.0979] | 0.0291 [0.0123, 0.0482] | 0.1661 [0.1306, 0.2036] | 0.4120 [0.1644, 0.6729] | 1.6742 [1.1936, 2.2008] | 3.7420 [2.7440, 4.8090] | 0.9084 [0.6948, 1.1445] | 2.9392 [2.5561, 3.3744] | 6.3564 [5.6746, 7.1161] | 1.2011 [0.9076, 1.5367] | 2.4698 [1.8794, 3.1463] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
