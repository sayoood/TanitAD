# VERDICT_GATEMIX (PREREG_V8F_PROOF_ADD8, card sha256 807bb2115013; MEASURED, tier T1-kin on logged frames: a diagnostic, never driving performance; CPU, zero training and zero rollouts)

## Overall outcome: **GATE-COSTS**

Per pair and seed (decided on composite minus arm, ADE6, paired episode-cluster bootstrap, B 2,000, seed 0):

| pair | seed | outcome | composite minus hierarchy | composite minus flat |
|---|---|---|---|---|
| P1 (DAG vs FLATDAG) | 0 | **GATE-TIES** | +0.0041 [-0.0031, +0.0126] | -0.3410 [-0.4358, -0.2465] |
| P1 (DAG vs FLATDAG) | 1 | **GATE-WORSE** | +0.0568 [+0.0208, +0.1022] | +0.0023 [-0.0913, +0.0957] |
| P2 (DAG0 vs FLATDAG0) | 0 | **GATE-WORSE** | +0.0986 [+0.0222, +0.1806] | +0.1448 [+0.0024, +0.3274] |
| P2 (DAG0 vs FLATDAG0) | 1 | **GATE-TIES** | -0.1426 [-0.2901, +0.0032] | +0.0170 [-0.0003, +0.0514] |

Floors beside every number: F = 0.12075 m (seed floor), tactical replicate floor 0.287 m. Estimator of every interval: paired episode-cluster bootstrap over the 141 episodes (taniteval.ci draws), B 2,000, seed 0; the tau and the common-case class are fixed per fold and are not re-fitted inside the draws. One training run per arm per seed: a separated interval is necessary and not sufficient. STRATEGIC family: **PAUSED** (PI item 25).

## Controls

* g1 (tau below every score = hierarchy, above = flat, 0.0 m): PASS; max abs deviation P1_s0 0.0e+00/0.0e+00, P1_s1 0.0e+00/0.0e+00, P2_s0 0.0e+00/0.0e+00, P2_s1 0.0e+00/0.0e+00
* g2 (banked ADE6 differences reproduced, bar 1e-4 m on point and bounds): PASS
  * P1_s0: reproduced -0.3451 [-0.4421, -0.2506] vs banked -0.3451 [-0.4421, -0.2506], max dev 0.0e+00 PASS
  * P2_h1a_route_s0: reproduced -0.4973 [-0.6075, -0.3874] vs banked -0.4973 [-0.6075, -0.3874], max dev 0.0e+00 PASS
  * P2_read_selfref_s0: reproduced +0.0462 [-0.1259, +0.2419] vs banked +0.0462 [-0.1259, +0.2419], max dev 0.0e+00 PASS
  * P1_s1: reproduced -0.0544 [-0.1605, +0.0547] vs banked -0.0544 [-0.1605, +0.0547], max dev 0.0e+00 PASS
  * P2_h1a_route_s1: reproduced -0.2450 [-0.3553, -0.1316] vs banked -0.2450 [-0.3553, -0.1316], max dev 0.0e+00 PASS
  * P2_read_selfref_s1: reproduced +0.1597 [+0.0089, +0.3145] vs banked +0.1597 [+0.0089, +0.3145], max dev 0.0e+00 PASS
* g3 (shuffled score: composite minus the better arm not negative-and-separated in >= 18 of 20 shuffles): PASS; P1_s0 20/20, P1_s1 20/20, P2_s0 20/20, P2_s1 19/20
* g4 (ORACLE composite equals the share-weighted group means, bar 1e-9 m): PASS; P1_s0 0.0e+00, P1_s1 0.0e+00, P2_s0 0.0e+00, P2_s1 0.0e+00

## P1 (DAG vs FLATDAG), seed 0: GATE-TIES

n = 2,399 windows, 141 episodes; selection-critical 1,305; fold sizes (windows) [493, 476, 478, 476, 476]; common-case joint class (lat*7+lon) per fold [6, 6, 6, 6, 6]; tau per fold [0.0357, 0.1164, 0.1272, 0.1471, 0.1515]. Mean ADE6 (m): hierarchy 2.2777, flat 2.6228, composite 2.2818, ORACLE 2.3619.

| metric | group | n win / n ep | composite minus hier | composite minus flat | ORACLE minus hier | ORACLE minus flat | floors (composite minus flat) |
|---|---|---|---|---|---|---|---|
| ade6_m | all | 2,399 / 141 | +0.0041 [-0.0031, +0.0126] | -0.3410 [-0.4358, -0.2465] | +0.0842 [+0.0388, +0.1331] | -0.2608 [-0.3434, -0.1827] | > F, > 0.287 |
| ade6_m | critical | 1,305 / 110 | +0.0041 [-0.0059, +0.0180] | -0.4754 [-0.6182, -0.3438] | +0.0000 [+0.0000, +0.0000] | -0.4795 [-0.6222, -0.3445] | > F, > 0.287 |
| ade6_m | bulk | 1,094 / 122 | +0.0042 [-0.0072, +0.0157] | -0.1806 [-0.2879, -0.0819] | +0.1847 [+0.0866, +0.2933] | +0.0000 [+0.0000, +0.0000] | > F |
| ade26_m | all | 2,399 / 141 | +0.0050 [-0.0058, +0.0176] | -0.5132 [-0.6532, -0.3733] | +0.1270 [+0.0599, +0.1995] | -0.3912 [-0.5137, -0.2759] | > F, > 0.287 |
| ade26_m | critical | 1,305 / 110 | +0.0059 [-0.0091, +0.0267] | -0.7132 [-0.9239, -0.5174] | +0.0000 [+0.0000, +0.0000] | -0.7191 [-0.9298, -0.5202] | > F, > 0.287 |
| ade26_m | bulk | 1,094 / 122 | +0.0039 [-0.0130, +0.0210] | -0.2746 [-0.4312, -0.1289] | +0.2784 [+0.1293, +0.4379] | +0.0000 [+0.0000, +0.0000] | > F |

Share of windows driven by the hierarchy: all +0.8950 [+0.8554, +0.9308], critical +0.9571 [+0.9257, +0.9802], bulk +0.8208 [+0.7563, +0.8826]; ORACLE share 0.5440. AUROC of s(w) for the logged selection-critical label: +0.6349 [+0.5744, +0.6924] (n = 2,399, 141 episodes; 1,305 critical, 1,094 bulk; 0.5 = chance).

Four families, composite minus each arm (mean over windows where both are finite; n windows per row):

| family | metric | n win | composite minus hier | composite minus flat |
|---|---|---|---|---|
| LONGITUDINAL | speed_mae_mps | 2,399 | -0.0007 [-0.0156, +0.0155] | -0.3248 [-0.4417, -0.2159] |
| LONGITUDINAL | along_mae_m | 2,399 | -0.0032 [-0.0098, +0.0031] | -0.1653 [-0.2600, -0.0696] |
| LATERAL | cross_mae_m | 2,399 | +0.0112 [+0.0032, +0.0212] | -0.2876 [-0.3602, -0.2199] |
| LATERAL | heading_mae_deg | 2,373 | +0.0207 [+0.0014, +0.0470] | -3.1487 [-4.2345, -2.2288] |
| LATERAL | curvature_mae_1pm | 2,365 | +0.0000 [-0.0000, +0.0001] | -0.0461 [-0.0699, -0.0244] |
| LATERAL | yaw_rate_mae_degps | 2,365 | +0.0066 [-0.0201, +0.0420] | -11.6206 [-16.8377, -7.2923] |
| TACTICAL | AUROC of s(w) | 2,399 | +0.6349 [+0.5744, +0.6924] | share driven by hier +0.8950 [+0.8554, +0.9308] |
| STRATEGIC | PAUSED (PI item 25) | | | |

## P1 (DAG vs FLATDAG), seed 1: GATE-WORSE

n = 2,399 windows, 141 episodes; selection-critical 1,305; fold sizes (windows) [493, 476, 478, 476, 476]; common-case joint class (lat*7+lon) per fold [6, 6, 6, 6, 6]; tau per fold [-inf, 0.0952, 0.0953, 0.1154, 0.5518]. Mean ADE6 (m): hierarchy 2.4565, flat 2.5109, composite 2.5132, ORACLE 2.4244.

| metric | group | n win / n ep | composite minus hier | composite minus flat | ORACLE minus hier | ORACLE minus flat | floors (composite minus flat) |
|---|---|---|---|---|---|---|---|
| ade6_m | all | 2,399 / 141 | +0.0568 [+0.0208, +0.1022] | +0.0023 [-0.0913, +0.0957] | -0.0320 [-0.1017, +0.0268] | -0.0865 [-0.1668, -0.0061] | <= F |
| ade6_m | critical | 1,305 / 110 | +0.0855 [+0.0255, +0.1673] | -0.0735 [-0.1953, +0.0459] | +0.0000 [+0.0000, +0.0000] | -0.1589 [-0.3043, -0.0117] | <= F |
| ade6_m | bulk | 1,094 / 122 | +0.0225 [-0.0074, +0.0566] | +0.0927 [-0.0336, +0.2391] | -0.0702 [-0.2201, +0.0610] | +0.0000 [+0.0000, +0.0000] | <= F |
| ade26_m | all | 2,399 / 141 | +0.0845 [+0.0309, +0.1518] | -0.0133 [-0.1513, +0.1229] | -0.0379 [-0.1385, +0.0483] | -0.1357 [-0.2548, -0.0188] | <= F |
| ade26_m | critical | 1,305 / 110 | +0.1267 [+0.0387, +0.2476] | -0.1227 [-0.3043, +0.0527] | +0.0000 [+0.0000, +0.0000] | -0.2494 [-0.4643, -0.0345] | > F |
| ade26_m | bulk | 1,094 / 122 | +0.0342 [-0.0099, +0.0847] | +0.1172 [-0.0672, +0.3294] | -0.0830 [-0.2991, +0.1060] | +0.0000 [+0.0000, +0.0000] | <= F |

Share of windows driven by the hierarchy: all +0.8291 [+0.7727, +0.8807], critical +0.8575 [+0.7941, +0.9142], bulk +0.7952 [+0.7063, +0.8725]; ORACLE share 0.5440. AUROC of s(w) for the logged selection-critical label: +0.6393 [+0.5777, +0.7005] (n = 2,399, 141 episodes; 1,305 critical, 1,094 bulk; 0.5 = chance).

Four families, composite minus each arm (mean over windows where both are finite; n windows per row):

| family | metric | n win | composite minus hier | composite minus flat |
|---|---|---|---|---|
| LONGITUDINAL | speed_mae_mps | 2,399 | +0.0380 [-0.0025, +0.0968] | +0.0831 [-0.0064, +0.1698] |
| LONGITUDINAL | along_mae_m | 2,399 | +0.0367 [+0.0082, +0.0686] | +0.0474 [-0.0468, +0.1549] |
| LATERAL | cross_mae_m | 2,399 | +0.0327 [+0.0035, +0.0737] | -0.0464 [-0.1026, +0.0025] |
| LATERAL | heading_mae_deg | 2,373 | +0.2123 [+0.0047, +0.4622] | -1.1801 [-1.7735, -0.7044] |
| LATERAL | curvature_mae_1pm | 2,365 | -0.0030 [-0.0125, +0.0053] | -0.0068 [-0.0194, +0.0055] |
| LATERAL | yaw_rate_mae_degps | 2,365 | +0.1638 [-0.6722, +1.1718] | -0.4469 [-2.6012, +2.1433] |
| TACTICAL | AUROC of s(w) | 2,399 | +0.6393 [+0.5777, +0.7005] | share driven by hier +0.8291 [+0.7727, +0.8807] |
| STRATEGIC | PAUSED (PI item 25) | | | |

## P2 (DAG0 vs FLATDAG0), seed 0: GATE-WORSE

n = 2,399 windows, 141 episodes; selection-critical 1,305; fold sizes (windows) [493, 476, 478, 476, 476]; common-case joint class (lat*7+lon) per fold [6, 6, 6, 6, 6]; tau per fold [-inf, 0.1292, 0.9756, 0.9866, 0.9977]. Mean ADE6 (m): hierarchy 2.8212, flat 2.7750, composite 2.9198, ORACLE 2.6762.

| metric | group | n win / n ep | composite minus hier | composite minus flat | ORACLE minus hier | ORACLE minus flat | floors (composite minus flat) |
|---|---|---|---|---|---|---|---|
| ade6_m | all | 2,399 / 141 | +0.0986 [+0.0222, +0.1806] | +0.1448 [+0.0024, +0.3274] | -0.1450 [-0.3349, +0.0036] | -0.0988 [-0.1685, -0.0356] | > F |
| ade6_m | critical | 1,305 / 110 | +0.1121 [+0.0176, +0.2149] | -0.0695 [-0.1376, -0.0019] | +0.0000 [+0.0000, +0.0000] | -0.1817 [-0.3054, -0.0686] | <= F |
| ade6_m | bulk | 1,094 / 122 | +0.0825 [-0.0325, +0.1925] | +0.4005 [+0.1049, +0.7665] | -0.3180 [-0.7169, +0.0079] | +0.0000 [+0.0000, +0.0000] | > F, > 0.287 |
| ade26_m | all | 2,399 / 141 | +0.1478 [+0.0360, +0.2692] | +0.2103 [+0.0014, +0.4749] | -0.2107 [-0.4883, +0.0075] | -0.1482 [-0.2522, -0.0547] | > F |
| ade26_m | critical | 1,305 / 110 | +0.1691 [+0.0280, +0.3214] | -0.1034 [-0.2058, -0.0020] | +0.0000 [+0.0000, +0.0000] | -0.2725 [-0.4572, -0.1036] | <= F |
| ade26_m | bulk | 1,094 / 122 | +0.1224 [-0.0487, +0.2855] | +0.5845 [+0.1531, +1.1192] | -0.4621 [-1.0470, +0.0167] | +0.0000 [+0.0000, +0.0000] | > F, > 0.287 |

Share of windows driven by the hierarchy: all +0.3881 [+0.3079, +0.4644], critical +0.3969 [+0.3009, +0.4920], bulk +0.3775 [+0.2853, +0.4783]; ORACLE share 0.5440. AUROC of s(w) for the logged selection-critical label: +0.6368 [+0.5761, +0.6972] (n = 2,399, 141 episodes; 1,305 critical, 1,094 bulk; 0.5 = chance).

Four families, composite minus each arm (mean over windows where both are finite; n windows per row):

| family | metric | n win | composite minus hier | composite minus flat |
|---|---|---|---|---|
| LONGITUDINAL | speed_mae_mps | 2,399 | -0.0530 [-0.1278, +0.0234] | +0.1624 [+0.0346, +0.3171] |
| LONGITUDINAL | along_mae_m | 2,399 | +0.0254 [-0.0421, +0.0994] | +0.2223 [+0.0520, +0.4393] |
| LATERAL | cross_mae_m | 2,399 | +0.1154 [+0.0559, +0.1763] | -0.0777 [-0.1306, -0.0309] |
| LATERAL | heading_mae_deg | 2,373 | +0.7523 [+0.4334, +1.1418] | -0.3625 [-0.6680, -0.0611] |
| LATERAL | curvature_mae_1pm | 2,365 | +0.0403 [+0.0205, +0.0634] | -0.0075 [-0.0232, +0.0075] |
| LATERAL | yaw_rate_mae_degps | 2,365 | +3.4942 [+1.7742, +5.3524] | -1.2648 [-2.7734, +0.0267] |
| TACTICAL | AUROC of s(w) | 2,399 | +0.6368 [+0.5761, +0.6972] | share driven by hier +0.3881 [+0.3079, +0.4644] |
| STRATEGIC | PAUSED (PI item 25) | | | |

## P2 (DAG0 vs FLATDAG0), seed 1: GATE-TIES

n = 2,399 windows, 141 episodes; selection-critical 1,305; fold sizes (windows) [493, 476, 478, 476, 476]; common-case joint class (lat*7+lon) per fold [6, 6, 6, 6, 6]; tau per fold [0.9732, 0.9946, inf]. Mean ADE6 (m): hierarchy 2.8611, flat 2.7014, composite 2.7185, ORACLE 2.6364.

| metric | group | n win / n ep | composite minus hier | composite minus flat | ORACLE minus hier | ORACLE minus flat | floors (composite minus flat) |
|---|---|---|---|---|---|---|---|
| ade6_m | all | 2,399 / 141 | -0.1426 [-0.2901, +0.0032] | +0.0170 [-0.0003, +0.0514] | -0.2247 [-0.3561, -0.1121] | -0.0651 [-0.1422, +0.0130] | <= F |
| ade6_m | critical | 1,305 / 110 | +0.1341 [-0.0076, +0.2730] | +0.0145 [-0.0006, +0.0465] | +0.0000 [+0.0000, +0.0000] | -0.1196 [-0.2619, +0.0243] | <= F |
| ade6_m | bulk | 1,094 / 122 | -0.4728 [-0.7378, -0.2409] | +0.0200 [+0.0000, +0.0617] | -0.4928 [-0.7552, -0.2488] | +0.0000 [+0.0000, +0.0000] | <= F |
| ade26_m | all | 2,399 / 141 | -0.2017 [-0.4193, +0.0139] | +0.0252 [-0.0006, +0.0761] | -0.3287 [-0.5229, -0.1636] | -0.1018 [-0.2147, +0.0153] | <= F |
| ade26_m | critical | 1,305 / 110 | +0.2086 [-0.0013, +0.4149] | +0.0214 [-0.0010, +0.0688] | +0.0000 [+0.0000, +0.0000] | -0.1872 [-0.3956, +0.0286] | <= F |
| ade26_m | bulk | 1,094 / 122 | -0.6912 [-1.0794, -0.3506] | +0.0296 [+0.0000, +0.0912] | -0.7208 [-1.1070, -0.3633] | +0.0000 [+0.0000, +0.0000] | <= F |

Share of windows driven by the hierarchy: all +0.0033 [+0.0000, +0.0092], critical +0.0031 [+0.0000, +0.0084], bulk +0.0037 [+0.0000, +0.0113]; ORACLE share 0.5440. AUROC of s(w) for the logged selection-critical label: +0.6424 [+0.5816, +0.7028] (n = 2,399, 141 episodes; 1,305 critical, 1,094 bulk; 0.5 = chance).

Four families, composite minus each arm (mean over windows where both are finite; n windows per row):

| family | metric | n win | composite minus hier | composite minus flat |
|---|---|---|---|---|
| LONGITUDINAL | speed_mae_mps | 2,399 | -0.3716 [-0.5080, -0.2404] | +0.0151 [+0.0000, +0.0452] |
| LONGITUDINAL | along_mae_m | 2,399 | -0.1511 [-0.3158, -0.0023] | +0.0193 [-0.0006, +0.0587] |
| LATERAL | cross_mae_m | 2,399 | -0.0271 [-0.1003, +0.0499] | +0.0010 [-0.0001, +0.0030] |
| LATERAL | heading_mae_deg | 2,373 | +1.4497 [+0.7717, +2.2129] | +0.0078 [+0.0000, +0.0227] |
| LATERAL | curvature_mae_1pm | 2,365 | +0.0599 [+0.0324, +0.0920] | +0.0000 [-0.0000, +0.0000] |
| LATERAL | yaw_rate_mae_degps | 2,365 | +0.9519 [-2.4274, +4.0995] | +0.0119 [-0.0006, +0.0364] |
| TACTICAL | AUROC of s(w) | 2,399 | +0.6424 [+0.5816, +0.7028] | share driven by hier +0.0033 [+0.0000, +0.0092] |
| STRATEGIC | PAUSED (PI item 25) | | | |

## Card bars next to the measured values

| bar | measured | result |
|---|---|---|
| P1 s0: composite minus hierarchy negative, interval excludes 0 | +0.0041 [-0.0031, +0.0126] | FAIL |
| P1 s0: composite minus flat negative, interval excludes 0 | -0.3410 [-0.4358, -0.2465] | PASS |
| P1 s0: no cell GATE-WORSE (neither difference positive with interval excluding 0) | GATE-TIES | PASS |
| P1 s1: composite minus hierarchy negative, interval excludes 0 | +0.0568 [+0.0208, +0.1022] | FAIL |
| P1 s1: composite minus flat negative, interval excludes 0 | +0.0023 [-0.0913, +0.0957] | FAIL |
| P1 s1: no cell GATE-WORSE (neither difference positive with interval excluding 0) | GATE-WORSE | FAIL |
| P2 s0: composite minus hierarchy negative, interval excludes 0 | +0.0986 [+0.0222, +0.1806] | FAIL |
| P2 s0: composite minus flat negative, interval excludes 0 | +0.1448 [+0.0024, +0.3274] | FAIL |
| P2 s0: no cell GATE-WORSE (neither difference positive with interval excluding 0) | GATE-WORSE | FAIL |
| P2 s1: composite minus hierarchy negative, interval excludes 0 | -0.1426 [-0.2901, +0.0032] | FAIL |
| P2 s1: composite minus flat negative, interval excludes 0 | +0.0170 [-0.0003, +0.0514] | FAIL |
| P2 s1: no cell GATE-WORSE (neither difference positive with interval excluding 0) | GATE-TIES | PASS |
| P1: GATE-BEATS-BOTH at both seeds | ['GATE-TIES', 'GATE-WORSE'] | FAIL |
| P2: GATE-BEATS-BOTH at both seeds | ['GATE-WORSE', 'GATE-TIES'] | FAIL |
| overall GATE-DOMINATES | GATE-COSTS | FAIL |
| control g1 | see Controls | PASS |
| control g2 | see Controls | PASS |
| control g3 | see Controls | PASS |
| control g4 | see Controls | PASS |

## Limits of the card (stated in advance, kept)

Post-hoc on banked runs whose primary outcome is NOT ESTABLISHED. One training run per arm per seed, no replicate arm: the bootstrap resamples episodes with the models held fixed and cannot see training variance. tau is cross-fitted on the evaluation clips (5 folds by episode), so the composite is an estimate of a rule, not a deployable threshold. The composite needs both models to run. ORACLE is privileged (the logged label) and is a ceiling, not an arm. This read never supports a statement that the hierarchy beats flat as an architecture. Definitions and deviations (R1...): GATE_DEFS.md.
