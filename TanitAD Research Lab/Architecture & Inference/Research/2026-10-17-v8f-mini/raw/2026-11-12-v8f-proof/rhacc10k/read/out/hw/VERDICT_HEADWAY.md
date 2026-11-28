# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: F:/TanitAD-scratch/v8proof/rhacc10k/pins/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 18, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09891171799189782, "A_hi": 0.16379738797148666, "spread_max": 0.005338078291814943, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "H_R0A:H_R0K", "seeds": [0, 1, 2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "H_RLcA:U_RLcK", "seeds": [0, 1, 2]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "H_RQ50A:U_RQ50K", "seeds": [0, 1, 2]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm U_RLcK, seeds present [0, 1, 2] of requested [0, 1, 2]; complete for the card (six seeds): False.

## Names of section 3 of the card

* None, cell U, H_R0A minus H_R0K over seeds [0, 1, 2] of [0, 1, 2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, H_RLcA minus U_RLcK over seeds [0, 1, 2] of [0, 1, 2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, H_RQ50A minus U_RQ50K over seeds [0, 1, 2] of [0, 1, 2]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## H R0A minus H R0K

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0409 / -0.0552 / -0.0391 | -0.0451 | [-0.0666, -0.0252] | 0.0053 (0.0053) | BETTER |
| C contact rate | -0.0218 / -0.0280 / -0.0171 | -0.0223 | [-0.0368, -0.0111] | 0.0016 (0.0016) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.0935 / +0.1215 / +0.0794 | +0.0981 | [+0.0681, +0.1307] | 0.0358 (0.0296) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0607 (0.0537) | OPEN |
| gap bias 4.0 s (m) | +1.0992 / +1.2448 / +0.9256 | +1.0898 | [+0.9027, +1.2886] | 0.4477 (0.3725) | WORSE |
| gap bias 6.0 s (m) | +3.4739 / +4.4221 / +3.4542 | +3.7834 | [+3.2119, +4.5537] | 1.1182 (0.9461) | WORSE |
| gap error 2.0 s (m) | +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0092 (0.0048) | OPEN |
| gap error 4.0 s (m) | +0.5120 / +0.4600 / +0.2962 | +0.4228 | [+0.2521, +0.5981] | 0.1594 (0.0930) | WORSE |
| gap error 6.0 s (m) | +1.2874 / +1.5622 / +1.1017 | +1.3171 | [+0.7569, +1.9200] | 0.4287 (0.2775) | WORSE |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0461 (0.0327) | OPEN |
| time-gap error over the scored steps (s) | +0.8690 / +1.1113 / +0.6655 | +0.8819 | [+0.5497, +1.3071] | 0.3014 (0.2062) | WORSE |

## H RLcA minus U RLcK

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0071 / -0.0160 / -0.0160 | -0.0130 | [-0.0277, +0.0005] | 0.0053 (0.0053) | OPEN |
| C contact rate | +0.0000 / +0.0000 / -0.0016 | -0.0005 | [-0.0022, +0.0000] | 0.0016 (0.0016) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0981 / +0.0794 / +0.0639 | +0.0805 | [+0.0491, +0.1104] | 0.0358 (0.0296) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2877 / +0.2129 / +0.2107 | +0.2371 | [+0.1925, +0.2881] | 0.0607 (0.0537) | WORSE |
| gap bias 4.0 s (m) | +1.3376 / +0.8668 / +0.8186 | +1.0077 | [+0.6764, +1.3772] | 0.4477 (0.3725) | WORSE |
| gap bias 6.0 s (m) | +3.2712 / +2.6268 / +2.3957 | +2.7646 | [+2.0148, +3.5209] | 1.1182 (0.9461) | WORSE |
| gap error 2.0 s (m) | +0.0050 / +0.0036 / +0.0123 | +0.0070 | [-0.0129, +0.0280] | 0.0092 (0.0048) | OPEN |
| gap error 4.0 s (m) | +0.7403 / +0.4444 / +0.2835 | +0.4894 | [+0.1930, +0.8121] | 0.1594 (0.0930) | WORSE |
| gap error 6.0 s (m) | +1.7947 / +1.6290 / +1.1048 | +1.5095 | [+0.8247, +2.1853] | 0.4287 (0.2775) | WORSE |
| time-gap error 2.0 s (s) | -0.1007 / -0.1374 / -0.1305 | -0.1229 | [-0.2704, -0.0035] | 0.0461 (0.0327) | BETTER |
| time-gap error over the scored steps (s) | +1.0059 / +0.8311 / +0.3950 | +0.7440 | [+0.3527, +1.1646] | 0.3014 (0.2062) | WORSE |

## H RQ50A minus U RQ50K

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0267 / -0.0320 / -0.0125 | -0.0237 | [-0.0425, -0.0089] | 0.0053 (0.0053) | BETTER |
| C contact rate | -0.0093 / -0.0187 / +0.0031 | -0.0083 | [-0.0229, +0.0041] | 0.0016 (0.0016) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0670 / +0.0623 / +0.0483 | +0.0592 | [+0.0287, +0.0889] | 0.0358 (0.0296) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2494 / +0.1991 / +0.1745 | +0.2076 | [+0.1656, +0.2555] | 0.0607 (0.0537) | WORSE |
| gap bias 4.0 s (m) | +1.1083 / +0.8322 / +0.7325 | +0.8910 | [+0.6177, +1.1858] | 0.4477 (0.3725) | WORSE |
| gap bias 6.0 s (m) | +2.6395 / +2.4865 / +2.2074 | +2.4445 | [+1.7842, +3.0726] | 1.1182 (0.9461) | WORSE |
| gap error 2.0 s (m) | +0.0155 / +0.0110 / +0.0214 | +0.0160 | [-0.0027, +0.0345] | 0.0092 (0.0048) | OPEN |
| gap error 4.0 s (m) | +0.6005 / +0.3889 / +0.3174 | +0.4356 | [+0.1999, +0.6955] | 0.1594 (0.0930) | WORSE |
| gap error 6.0 s (m) | +1.1849 / +1.4700 / +0.9006 | +1.1852 | [+0.5635, +1.7925] | 0.4287 (0.2775) | WORSE |
| time-gap error 2.0 s (s) | -0.1073 / -0.1353 / -0.1258 | -0.1228 | [-0.2680, -0.0044] | 0.0461 (0.0327) | BETTER |
| time-gap error over the scored steps (s) | +0.9575 / +0.7570 / +0.3469 | +0.6871 | [+0.2769, +1.1052] | 0.3014 (0.2062) | WORSE |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H_R0A (0,1,2) | 0.0498 [0.0281, 0.0766] | 0.0405 [0.0192, 0.0656] | 0.1978 [0.1517, 0.2441] | 0.0496 [0.0035, 0.0975] | 1.6254 [1.1913, 2.0746] | 5.1086 [3.9325, 6.3203] | 0.4600 [0.4209, 0.5043] | 2.8291 [2.4967, 3.1936] | 7.3244 [6.5338, 8.1804] | 0.9233 [0.6837, 1.2096] | 2.9976 [2.3166, 3.7679] |
| H_R0K (0,1,2) | 0.0949 [0.0634, 0.1309] | 0.0628 [0.0358, 0.0911] | 0.0997 [0.0720, 0.1324] | 0.0496 [0.0035, 0.0975] | 0.5355 [0.1917, 0.8968] | 1.3252 [0.2944, 2.4492] | 0.4600 [0.4209, 0.5043] | 2.4063 [2.1616, 2.6819] | 6.0073 [5.3872, 6.7301] | 0.9233 [0.6837, 1.2096] | 2.1157 [1.5754, 2.7452] |
| H_RLcA (0,1,2) | 0.0243 [0.0111, 0.0412] | 0.0000 [0.0000, 0.0000] | 0.2285 [0.1792, 0.2793] | 0.5144 [0.2920, 0.7679] | 2.5145 [2.0144, 3.0463] | 6.2737 [5.2591, 7.3498] | 0.8017 [0.5912, 1.0434] | 3.1232 [2.7043, 3.5933] | 7.1750 [6.3675, 8.0603] | 0.9692 [0.7264, 1.2570] | 3.0031 [2.3183, 3.7677] |
| H_RQ50A (0,1,2) | 0.0463 [0.0272, 0.0687] | 0.0208 [0.0058, 0.0391] | 0.2253 [0.1819, 0.2697] | 0.6197 [0.3730, 0.8805] | 2.5653 [2.0790, 3.0728] | 6.1865 [5.2491, 7.1667] | 0.9244 [0.7043, 1.1591] | 3.3748 [2.9371, 3.8412] | 7.5416 [6.7810, 8.3452] | 1.0783 [0.8082, 1.3770] | 3.1569 [2.4490, 3.9453] |
| U_RLcK (0,1,2) | 0.0374 [0.0199, 0.0568] | 0.0005 [0.0000, 0.0022] | 0.1480 [0.1053, 0.1956] | 0.2773 [0.0427, 0.5439] | 1.5068 [0.9344, 2.1183] | 3.5091 [2.3635, 4.6444] | 0.7948 [0.5900, 1.0296] | 2.6338 [2.2017, 3.1470] | 5.6655 [4.9544, 6.5091] | 1.0920 [0.8200, 1.4077] | 2.2591 [1.7071, 2.8892] |
| U_RQ50K (0,1,2) | 0.0700 [0.0445, 0.0979] | 0.0291 [0.0123, 0.0482] | 0.1661 [0.1306, 0.2036] | 0.4120 [0.1644, 0.6729] | 1.6742 [1.1936, 2.2008] | 3.7420 [2.7440, 4.8090] | 0.9084 [0.6948, 1.1445] | 2.9392 [2.5561, 3.3744] | 6.3564 [5.6746, 7.1161] | 1.2011 [0.9076, 1.5367] | 2.4698 [1.8794, 3.1463] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
