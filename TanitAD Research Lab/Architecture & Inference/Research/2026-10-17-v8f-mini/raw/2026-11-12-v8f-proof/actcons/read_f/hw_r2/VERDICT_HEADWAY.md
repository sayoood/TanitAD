# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 14, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLA3:RLFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLA3:RLAm", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLA2:RLFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "RLA2:RLAm", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, RLA3 minus RLFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, RLA3 minus RLAm over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, RLA2 minus RLFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, RLA2 minus RLAm over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## RLA3 minus RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0160 / -0.0125 | +0.0018 | [-0.0199, +0.0242] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0109 / +0.0016 | +0.0062 | [-0.0025, +0.0150] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0047 / -0.0296 | -0.0125 | [-0.0448, +0.0151] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0143 / +0.0008 | -0.0068 | [-0.0184, +0.0049] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.0672 / +0.0081 | -0.0296 | [-0.1332, +0.0590] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -0.0027 / -0.1672 | -0.0849 | [-0.4561, +0.2352] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0035 / +0.0034 | +0.0034 | [-0.0020, +0.0084] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0448 / -0.0821 | -0.0187 | [-0.1199, +0.0996] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.3894 / -0.3906 | -0.0006 | [-0.5987, +0.5558] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0465 / +0.0223 | +0.0344 | [-0.0179, +0.0865] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.1377 / -0.1963 | -0.0293 | [-0.2930, +0.1974] | 1.7849 (0.8089) | OPEN |

## RLA3 minus RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0053 / -0.0160 | -0.0053 | [-0.0229, +0.0117] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0062 / +0.0078 | +0.0070 | [+0.0022, +0.0134] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0093 / -0.0312 | -0.0109 | [-0.0421, +0.0191] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.0025 / +0.0085 | +0.0030 | [-0.0063, +0.0119] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.0761 / +0.0563 | -0.0099 | [-0.1515, +0.1056] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.0833 / -0.3483 | -0.1325 | [-0.6114, +0.3491] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0003 / -0.0034 | -0.0015 | [-0.0068, +0.0040] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.0439 / +0.0059 | -0.0190 | [-0.1155, +0.0625] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.2402 / -0.4574 | -0.1086 | [-0.6567, +0.4455] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0236 / +0.0254 | +0.0009 | [-0.0489, +0.0583] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0470 / -0.0414 | +0.0028 | [-0.1280, +0.1361] | 1.7849 (0.8089) | OPEN |

## RLA2 minus RLFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0107 / +0.0071 | +0.0089 | [-0.0035, +0.0214] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0016 / +0.0093 | +0.0055 | [-0.0015, +0.0135] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0047 / -0.0358 | -0.0202 | [-0.0479, +0.0085] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0017 / -0.0064 | -0.0024 | [-0.0119, +0.0072] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.0891 / -0.2058 | -0.1475 | [-0.2621, -0.0355] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.1029 / -1.0691 | -0.5860 | [-1.3330, +0.1017] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0010 / +0.0060 | +0.0035 | [-0.0041, +0.0107] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0363 / -0.1649 | -0.0643 | [-0.2083, +0.0713] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.3583 / -0.8238 | -0.2327 | [-1.0718, +0.5491] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0377 / -0.0196 | +0.0090 | [-0.0480, +0.0600] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0089 / -0.3415 | -0.1663 | [-0.4420, +0.0405] | 1.7849 (0.8089) | OPEN |

## RLA2 minus RLAm

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0036 | +0.0018 | [-0.0094, +0.0139] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0031 / +0.0156 | +0.0062 | [-0.0059, +0.0211] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0000 / -0.0374 | -0.0187 | [-0.0483, +0.0124] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0135 / +0.0013 | +0.0074 | [-0.0022, +0.0192] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.0980 / -0.1576 | -0.1278 | [-0.2106, -0.0268] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -0.0169 / -1.2502 | -0.6336 | [-1.5607, +0.2611] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0022 / -0.0007 | -0.0015 | [-0.0081, +0.0044] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.0524 / -0.0769 | -0.0646 | [-0.1398, +0.0088] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.2091 / -0.8906 | -0.3407 | [-1.1669, +0.4374] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0324 / -0.0166 | -0.0245 | [-0.0701, +0.0177] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.0818 / -0.1865 | -0.1342 | [-0.2732, -0.0236] | 1.7849 (0.8089) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| RLA2 (0,1) | 0.1335 [0.0836, 0.1926] | 0.0475 [0.0246, 0.0751] | 0.1628 [0.1149, 0.2206] | -0.2145 [-0.2635, -0.1644] | 0.6086 [-0.1028, 1.2660] | 4.3319 [2.4546, 6.4086] | 0.4872 [0.4375, 0.5391] | 2.5939 [2.3285, 2.8799] | 7.3458 [6.0691, 8.9299] | 1.0646 [0.7218, 1.4582] | 1.8825 [1.3453, 2.5181] |
| RLA3 (0,1) | 0.1263 [0.0838, 0.1732] | 0.0483 [0.0245, 0.0761] | 0.1706 [0.1201, 0.2288] | -0.2188 [-0.2691, -0.1701] | 0.7265 [0.0899, 1.2902] | 4.8330 [3.2531, 6.4849] | 0.4872 [0.4370, 0.5396] | 2.6395 [2.3568, 2.9477] | 7.5779 [6.4087, 9.0271] | 1.0899 [0.7426, 1.4986] | 2.0194 [1.4759, 2.6579] |
| RLAm (0,1) | 0.1317 [0.0831, 0.1892] | 0.0413 [0.0194, 0.0678] | 0.1815 [0.1372, 0.2287] | -0.2218 [-0.2701, -0.1740] | 0.7364 [0.0292, 1.3799] | 4.9655 [3.4659, 6.5884] | 0.4887 [0.4391, 0.5413] | 2.6585 [2.3764, 2.9619] | 7.6865 [6.6276, 8.9726] | 1.0890 [0.7424, 1.4765] | 2.0166 [1.4838, 2.6347] |
| RLFT (0,1) | 0.1246 [0.0752, 0.1873] | 0.0421 [0.0206, 0.0660] | 0.1830 [0.1376, 0.2315] | -0.2121 [-0.2594, -0.1651] | 0.7561 [0.0790, 1.3428] | 4.9179 [3.3324, 6.5198] | 0.4837 [0.4329, 0.5361] | 2.6582 [2.3576, 2.9983] | 7.5785 [6.6350, 8.6730] | 1.0555 [0.7232, 1.4345] | 2.0488 [1.4877, 2.6843] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
