# VERDICT_HIERREP (PREREG_V8F_PROOF_ADD11; MEASURED, tier T1-kin, a diagnostic, never driving performance)

Estimator: (A) two-level bootstrap seeds then episodes, B 2000, rng 0 (decides); (B) exact sign-flip over seed differences; (C) t interval (decides nothing). Seeds read: [0, 1, 2, 3, 4, 5] (n = 6); excluded: []. Windows: {'n_windows': 2399, 'n_episodes': 141, 'n_crit': 1305, 'n_bulk': 1094}.
Difference = hierarchical arm minus flat arm, ADE6 in metres, lower is better; negative = hierarchy better. Each cell: D [95 % interval A]; d_s are the per-seed differences.

## Outcome (primary protocol, Stage 2 DAG vs FLATDAG): **HIER-DOMINATES**; Stage 1 (secondary): **NOT-ESTABLISHED**

## Stage 1 (from scratch, 19,164 steps): HIER - FLAT  (outcome NOT-ESTABLISHED)
| cell | class | n windows per seed (ep) | seeds | D [A] | d_s | sign-flip p (B) | t interval (C) | label |
|---|---|---|---|---|---|---|---|---|
| ADE6 | critical | 1305 (110) | 6 | -0.1582 [-0.2656, -0.0338] | +0.109, -0.186, -0.121, -0.274, -0.283, -0.194 | 0.06250 | [-0.3091, -0.0073] | OPEN |
| ADE6 | bulk | 1094 (122) | 6 | +0.3025 [+0.1437, +0.4550] | +0.482, +0.082, +0.507, +0.383, +0.083, +0.280 | 0.03125 | [+0.1045, +0.5005] | FLAT |
| ADE6 | all | 2399 (141) | 6 | +0.0519 [-0.0607, +0.1686] | +0.279, -0.064, +0.165, +0.026, -0.116, +0.022 | 0.43750 | [-0.1020, +0.2058] | OPEN |
| tactical-band ADE 2.2-6.0 s | critical | 1305 (110) | 6 | -0.2448 [-0.4028, -0.0627] | +0.140, -0.280, -0.191, -0.416, -0.428, -0.294 | 0.06250 | [-0.4638, -0.0259] | OPEN |
| tactical-band ADE 2.2-6.0 s | bulk | 1094 (122) | 6 | +0.4430 [+0.2113, +0.6647] | +0.705, +0.124, +0.738, +0.557, +0.110, +0.424 | 0.03125 | [+0.1529, +0.7331] | FLAT |
| tactical-band ADE 2.2-6.0 s | all | 2399 (141) | 6 | +0.0688 [-0.0954, +0.2379] | +0.398, -0.096, +0.233, +0.027, -0.183, +0.034 | 0.50000 | [-0.1555, +0.2932] | OPEN |

Noise floor beside the differences (same arm type, 15 seed pairs, ADE6): median |difference| m, pairs separated by the paired episode bootstrap
| arm | class | median abs m | separated / pairs |
|---|---|---|---|
| HIER | critical | 0.1313 | 7 / 15 |
| HIER | bulk | 0.1424 | 5 / 15 |
| HIER | all | 0.1263 | 8 / 15 |
| FLAT | critical | 0.0184 | 0 / 15 |
| FLAT | bulk | 0.1801 | 8 / 15 |
| FLAT | all | 0.0813 | 5 / 15 |

Four families on all windows (hierarchy minus flat; speed, along-track, cross-track, heading, curvature, yaw rate, mode error rates, tactical-band ADE):
| row | n windows (ep) | D [A] | sign-flip p | label |
|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | 2399 (141) | +0.29835 [+0.17314, +0.41734] | 0.03125 | FLAT |
| LONGITUDINAL:along_mae_m | 2399 (141) | +0.21279 [+0.08458, +0.34458] | 0.03125 | FLAT |
| LATERAL:cross_mae_m | 2399 (141) | -0.18431 [-0.26374, -0.11241] | 0.03125 | HIER |
| LATERAL:heading_mae_deg | 2373 (141) | -0.99453 [-1.78783, -0.27848] | 0.06250 | OPEN |
| LATERAL:curvature_mae_1pm | 2366 (141) | -0.01378 [-0.05626, +0.02521] | 0.71875 | OPEN |
| LATERAL:yaw_rate_mae_degps | 2366 (141) | +0.59703 [-3.66252, +4.44322] | 0.78125 | OPEN |
| TACTICAL:lon_err_rate | 2399 (141) | +0.09692 [+0.06255, +0.13024] | 0.03125 | FLAT |
| TACTICAL:lat_err_rate | 2399 (141) | +0.00424 [-0.03815, +0.03628] | 0.90625 | OPEN |
| TACTICAL:joint_err_rate | 2399 (141) | +0.06475 [+0.02097, +0.10379] | 0.06250 | OPEN |
| TACTICAL:ade_tac_m | 2399 (141) | +0.06884 [-0.09544, +0.23788] | 0.50000 | OPEN |

## Stage 2 (own-state fine-tune, 3,000 steps) = PRIMARY protocol: DAG - FLATDAG  (outcome HIER-DOMINATES)
| cell | class | n windows per seed (ep) | seeds | D [A] | d_s | sign-flip p (B) | t interval (C) | label |
|---|---|---|---|---|---|---|---|---|
| ADE6 | critical | 1305 (110) | 6 | -0.1953 [-0.2930, -0.0982] | -0.129, -0.087, -0.298, -0.110, -0.313, -0.235 | 0.03125 | [-0.2996, -0.0910] | HIER |
| ADE6 | bulk | 1094 (122) | 6 | +0.0305 [-0.0541, +0.1202] | -0.025, +0.061, -0.074, +0.092, -0.037, +0.166 | 0.50000 | [-0.0654, +0.1264] | OPEN |
| ADE6 | all | 2399 (141) | 6 | -0.0923 [-0.1678, -0.0246] | -0.081, -0.020, -0.196, -0.018, -0.187, -0.052 | 0.03125 | [-0.1767, -0.0079] | HIER |
| tactical-band ADE 2.2-6.0 s | critical | 1305 (110) | 6 | -0.2997 [-0.4428, -0.1570] | -0.204, -0.137, -0.457, -0.180, -0.471, -0.350 | 0.03125 | [-0.4531, -0.1462] | HIER |
| tactical-band ADE 2.2-6.0 s | bulk | 1094 (122) | 6 | +0.0409 [-0.0833, +0.1751] | -0.039, +0.084, -0.116, +0.123, -0.054, +0.248 | 0.53125 | [-0.1012, +0.1830] | OPEN |
| tactical-band ADE 2.2-6.0 s | all | 2399 (141) | 6 | -0.1444 [-0.2551, -0.0428] | -0.129, -0.036, -0.302, -0.042, -0.281, -0.077 | 0.03125 | [-0.2689, -0.0198] | HIER |

Noise floor beside the differences (same arm type, 15 seed pairs, ADE6): median |difference| m, pairs separated by the paired episode bootstrap
| arm | class | median abs m | separated / pairs |
|---|---|---|---|
| DAG | critical | 0.0684 | 5 / 15 |
| DAG | bulk | 0.1303 | 8 / 15 |
| DAG | all | 0.0791 | 6 / 15 |
| FLATDAG | critical | 0.0544 | 2 / 15 |
| FLATDAG | bulk | 0.0727 | 5 / 15 |
| FLATDAG | all | 0.0417 | 2 / 15 |

Four families on all windows (hierarchy minus flat; speed, along-track, cross-track, heading, curvature, yaw rate, mode error rates, tactical-band ADE):
| row | n windows (ep) | D [A] | sign-flip p | label |
|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | 2399 (141) | -0.08724 [-0.19659, +0.00345] | 0.09375 | OPEN |
| LONGITUDINAL:along_mae_m | 2399 (141) | +0.02355 [-0.04889, +0.10313] | 0.56250 | OPEN |
| LATERAL:cross_mae_m | 2399 (141) | -0.16531 [-0.21910, -0.09679] | 0.03125 | HIER |
| LATERAL:heading_mae_deg | 2373 (141) | -1.76080 [-2.24329, -1.28273] | 0.03125 | HIER |
| LATERAL:curvature_mae_1pm | 2366 (141) | -0.02608 [-0.04964, -0.00378] | 0.12500 | OPEN |
| LATERAL:yaw_rate_mae_degps | 2366 (141) | -4.82335 [-8.65650, -1.64263] | 0.06250 | OPEN |
| TACTICAL:lon_err_rate | 2399 (141) | -0.09997 [-0.13514, -0.06345] | 0.03125 | HIER |
| TACTICAL:lat_err_rate | 2399 (141) | -0.04460 [-0.06100, -0.02681] | 0.03125 | HIER |
| TACTICAL:joint_err_rate | 2399 (141) | -0.10400 [-0.14284, -0.06490] | 0.03125 | HIER |
| TACTICAL:ade_tac_m | 2399 (141) | -0.14435 [-0.25515, -0.04282] | 0.03125 | HIER |

## Named gaps
* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: PAUSED (PI item 25), never a pass, never omitted

## Controls
* r5_completeness: {"PASS": true, "note": "an arm without its final step is not read; a seed enters only with all four arms", "seeds_read": [0, 1, 2, 3, 4, 5], "seeds_excluded": []}
* r3_param_count: {"PASS": true, "expected": 7214495, "mismatch": []}
* window_counts: {"PASS": true, "card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "checked": true}
* r4 label swap (20 draws, Stage 2 critical, must read OPEN in >= 18): PASS; OPEN in 20 of 20
* r1 harness: PASS {"control": "r1 harness", "tol_m": 0.005, "pod_ade6_all_windows_m": 2.2310719951188216, "ref_ade6_all_windows_m": 2.2310735035518494, "abs_dev_m": 1.5084330278192226e-06, "n_windows_pod": 2399, "n_windows_ref": 2399}
* r2_gate: {"note": "read from the chain's token files and STAGEB_STATUS.json, not from the dumps", "status": "see chain log"}

## What it cannot show
Six seeds, one corpus (4,512 training clips), one evaluation set of 141 episodes, a kinematic rollout on logged frames. Nothing about another corpus, a re-perception simulator or a larger model. The intervals answer whether another draw of seeds and episodes would say this; inference-seed variance is not sampled (the planner is deterministic).
