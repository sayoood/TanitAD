# VERDICT_HIERWHERE (PREREG_V8F_PROOF_ADD6; MEASURED, tier T1-kin on logged frames, a diagnostic, never driving performance)

Estimator: paired episode-cluster bootstrap on taniteval.ci draws (pinned tree E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a), B 2000, seed 0; one training run per arm per seed: a separated interval is necessary, not sufficient. n = 2399 windows, 141 episodes per seed. Floors beside every term: seed floor F = 0.12075 m (banked bar 2F = 0.2415 m), tactical replicate floor 0.287 m. STRATEGIC: PAUSED (PI item 25).
One training run per arm per seed, no replicate arm: a separated interval is necessary and not sufficient. Nothing here says the hierarchy beats flat. Definitions and deviations: HIER_DEFS.md.

## Outcomes: R3 = **MIXED**, R1 = **CRIT-CONCENTRATED**; controls A CONTROL FAILED
Controls failed (the card says the read is then not valid until the control reads its known value): h4_s0. h3 is a flagged definition gap, not a failure (card rule).
h5 sensitivity of the R3 name: x0.75 -> MIXED, x1.25 -> MIXED; the decided outcome is the stated-threshold one.

## R3 exact decomposition of D = ADE6(DAG) - ADE6(FLAT-DAG), metres, point [95 % CI]
| term | n windows (ep) s0 | s0 | n windows (ep) s1 | s1 |
|---|---|---|---|---|
| D | 2399 (141) | -0.3451 [-0.4421, -0.2506] sep | 2399 (141) | -0.0544 [-0.1605, +0.0547] |
| WITHIN | 763 (113) | -0.0320 [-0.0684, -0.0027] sep | 768 (115) | -0.0303 [-0.0653, +0.0024] |
| SELECTION | 577 (124) | -0.0536 [-0.0853, -0.0235] sep | 595 (123) | +0.0216 [-0.0276, +0.0777] |
| BOTH-MISS | 1059 (119) | -0.2594 [-0.3408, -0.1832] sep | 1036 (123) | -0.0458 [-0.1152, +0.0193] |

Same three terms, 2-6 s ADE (knots t = 2.2 to 6.0 s):
| term | s0 | s1 |
|---|---|---|
| D | -0.5181 [-0.6613, -0.3781] sep | -0.0978 [-0.2543, +0.0618] |
| WITHIN | -0.0482 [-0.1019, -0.0043] sep | -0.0494 [-0.1020, -0.0008] sep |
| SELECTION | -0.0825 [-0.1300, -0.0377] sep | +0.0255 [-0.0458, +0.1061] |
| BOTH-MISS | -0.3875 [-0.5083, -0.2751] sep | -0.0738 [-0.1758, +0.0224] |

## R3 card bars next to measured values
| outcome | condition | measured | result |
|---|---|---|---|
| SELECTION-CARRIES | s0_SELECTION_lt_0 | SELECTION s0 -0.0536 | PASS |
| SELECTION-CARRIES | s0_SELECTION_interval_excludes_0 | [-0.0853, -0.0235] | PASS |
| SELECTION-CARRIES | s0_abs_SELECTION_ge_abs_WITHIN | |SEL| 0.0536 vs |WITHIN| 0.0320 | PASS |
| SELECTION-CARRIES | s1_SELECTION_le_WITHIN | SEL +0.0216 vs WITHIN -0.0303 (s1) | FAIL |
| WITHIN-CARRIES | s0_WITHIN_lt_0 | WITHIN s0 -0.0320 | PASS |
| WITHIN-CARRIES | s0_WITHIN_interval_excludes_0 | [-0.0684, -0.0027] | PASS |
| WITHIN-CARRIES | s0_abs_WITHIN_gt_abs_SELECTION | |WITHIN| 0.0320 vs |SEL| 0.0536 | FAIL |
| CRIT-CONCENTRATED | s0 crit diff more negative than bulk diff | crit -0.4795 vs bulk -0.1847 | PASS |
| CRIT-CONCENTRATED | paired (crit - bulk) interval excludes 0 on the negative side | -0.2948 [-0.4675, -0.1284] sep | PASS |

## R1 WHERE: ADE6(DAG) - ADE6(FLAT-DAG), metres, by window class
### seed 0
| window class (logged) | n windows | n episodes | difference [95 % CI] |
|---|---|---|---|
| SELECTION-CRITICAL | 1305 | 110 | -0.4795 [-0.6222, -0.3445] sep |
| BULK | 1094 | 122 | -0.1847 [-0.2933, -0.0866] sep |
| crit - bulk | | | -0.2948 [-0.4675, -0.1284] sep |
| STOP | 141 | 26 | -0.3906 [-0.9524, +0.1448] |
| DECEL | 401 | 72 | -0.0007 [-0.1664, +0.1632] |
| ACCEL | 457 | 69 | -0.8855 [-1.0893, -0.6765] sep |
| KEEP | 1400 | 138 | -0.2628 [-0.3686, -0.1676] sep |
| LEFT | 219 | 37 | -0.6471 [-1.0460, -0.2751] sep |
| RIGHT | 429 | 52 | -0.4845 [-0.6817, -0.2805] sep |
| STRAIGHT | 1751 | 136 | -0.2731 [-0.3832, -0.1727] sep |

### seed 1
| window class (logged) | n windows | n episodes | difference [95 % CI] |
|---|---|---|---|
| SELECTION-CRITICAL | 1305 | 110 | -0.1589 [-0.3043, -0.0117] sep |
| BULK | 1094 | 122 | +0.0702 [-0.0610, +0.2201] |
| crit - bulk | | | -0.2291 [-0.4299, -0.0303] sep |
| STOP | 141 | 26 | -0.1690 [-0.8561, +0.3334] |
| DECEL | 401 | 72 | +0.1853 [+0.0076, +0.3625] sep |
| ACCEL | 457 | 69 | -0.3608 [-0.5997, -0.1125] sep |
| KEEP | 1400 | 138 | -0.0116 [-0.1280, +0.1189] |
| LEFT | 219 | 37 | -0.5251 [-1.0307, -0.0804] sep |
| RIGHT | 429 | 52 | -0.1576 [-0.4172, +0.1020] |
| STRAIGHT | 1751 | 136 | +0.0297 [-0.0750, +0.1428] |

## R2 selection (mode) error rates, 1 - mean c_a (fraction of windows whose arm trajectory misses the logged mode)
| seed | family | DAG | FLAT-DAG | DAG - FLAT-DAG [95 % CI] |
|---|---|---|---|---|
| s0 | longitudinal | 0.4314 | 0.4969 | -0.0654 [-0.1033, -0.0283] sep |
| s0 | lateral | 0.2055 | 0.2597 | -0.0542 [-0.0779, -0.0308] sep |
| s0 | joint | 0.5261 | 0.5973 | -0.0713 [-0.1033, -0.0367] sep |
| s1 | longitudinal | 0.4544 | 0.4739 | -0.0196 [-0.0620, +0.0238] |
| s1 | lateral | 0.2134 | 0.2305 | -0.0171 [-0.0392, +0.0038] |
| s1 | joint | 0.5444 | 0.5673 | -0.0229 [-0.0650, +0.0200] |

## Four families on the two window groups, DAG - FLAT-DAG, point [95 % CI] (LONGITUDINAL: speed, along-track; LATERAL: cross-track, heading, curvature, yaw rate; TACTICAL = R2; STRATEGIC PAUSED)
| seed | row | group | n windows (ep) | difference |
|---|---|---|---|---|
| s0 | LONGITUDINAL:speed_mae_mps | crit | 1305 (110) | -0.4734 [-0.6330, -0.3221] sep |
| s0 | LONGITUDINAL:speed_mae_mps | bulk | 1094 (122) | -0.1460 [-0.2701, -0.0340] sep |
| s0 | LONGITUDINAL:along_mae_m | crit | 1305 (110) | -0.2697 [-0.4081, -0.1342] sep |
| s0 | LONGITUDINAL:along_mae_m | bulk | 1094 (122) | -0.0336 [-0.1493, +0.0776] |
| s0 | LATERAL:cross_mae_m | crit | 1305 (110) | -0.3590 [-0.4576, -0.2715] sep |
| s0 | LATERAL:cross_mae_m | bulk | 1094 (122) | -0.2271 [-0.3273, -0.1353] sep |
| s0 | LATERAL:heading_mae_deg | crit | 1279 (110) | -4.0513 [-5.2931, -2.8695] sep |
| s0 | LATERAL:heading_mae_deg | bulk | 1094 (122) | -2.1384 [-3.3797, -1.1853] sep |
| s0 | LATERAL:curvature_mae_1pm | crit | 1274 (110) | -0.0520 [-0.0820, -0.0253] sep |
| s0 | LATERAL:curvature_mae_1pm | bulk | 1091 (121) | -0.0394 [-0.0695, -0.0135] sep |
| s0 | LATERAL:yaw_rate_mae_degps | crit | 1274 (110) | -14.1169 [-20.5308, -8.8614] sep |
| s0 | LATERAL:yaw_rate_mae_degps | bulk | 1091 (121) | -8.7198 [-15.1536, -4.0278] sep |
| s1 | LONGITUDINAL:speed_mae_mps | crit | 1305 (110) | -0.0481 [-0.2011, +0.0886] |
| s1 | LONGITUDINAL:speed_mae_mps | bulk | 1094 (122) | +0.1562 [+0.0330, +0.3002] sep |
| s1 | LONGITUDINAL:along_mae_m | crit | 1305 (110) | -0.0600 [-0.1868, +0.0680] |
| s1 | LONGITUDINAL:along_mae_m | bulk | 1094 (122) | +0.0950 [-0.0583, +0.2894] |
| s1 | LATERAL:cross_mae_m | crit | 1305 (110) | -0.1275 [-0.2349, -0.0253] sep |
| s1 | LATERAL:cross_mae_m | bulk | 1094 (122) | -0.0214 [-0.0913, +0.0348] |
| s1 | LATERAL:heading_mae_deg | crit | 1279 (110) | -1.7726 [-2.5803, -1.0786] sep |
| s1 | LATERAL:heading_mae_deg | bulk | 1094 (122) | -0.9480 [-1.8160, -0.3343] sep |
| s1 | LATERAL:curvature_mae_1pm | crit | 1274 (110) | -0.0090 [-0.0318, +0.0154] |
| s1 | LATERAL:curvature_mae_1pm | bulk | 1091 (121) | +0.0024 [-0.0198, +0.0233] |
| s1 | LATERAL:yaw_rate_mae_degps | crit | 1274 (110) | -1.6399 [-5.0289, +1.9305] |
| s1 | LATERAL:yaw_rate_mae_degps | bulk | 1091 (121) | +0.5911 [-1.8880, +3.2240] |

## Controls
* h1 s0: PASS; reproduced -0.3451 [-0.4421, -0.2506] vs banked -0.3451 [-0.4421, -0.2506], max |dev| 0.00e+00 m (bar 1e-4); own bootstrap vs taniteval.ci max |dev| 3.4e-05.
* h1 s1: PASS; reproduced -0.0544 [-0.1605, +0.0547] vs banked -0.0544 [-0.1605, +0.0547], max |dev| 0.00e+00 m (bar 1e-4); own bootstrap vs taniteval.ci max |dev| 4.0e-05.
* h2 s0: PASS; |WITHIN + SELECTION + BOTH-MISS - D| = 0.0e+00 m (ADE6), 0.0e+00 m (2-6 s); bar 1e-9.
* h2 s1: PASS; |WITHIN + SELECTION + BOTH-MISS - D| = 0.0e+00 m (ADE6), 0.0e+00 m (2-6 s); bar 1e-9.
* h3 s0: mode-function STOP vs existing stop-window label: Jaccard 0.3140 (bar 0.90; FLAGGED definition gap), label 131, mode 141, both 65, recall 0.4962, precision 0.4610, accuracy over all windows 0.9408.
  R1 on the existing label (s0): STOP-label windows -0.0781 [-0.6157, +0.4590] (n 131); other windows -0.3605 [-0.4545, -0.2689] sep (n 2268).
* h3 s1: mode-function STOP vs existing stop-window label: Jaccard 0.3140 (bar 0.90; FLAGGED definition gap), label 131, mode 141, both 65, recall 0.4962, precision 0.4610, accuracy over all windows 0.9408.
  R1 on the existing label (s1): STOP-label windows +0.1160 [-0.3529, +0.4735] (n 131); other windows -0.0643 [-0.1691, +0.0460] (n 2268).
* h4 s0 (swap seed 0): FAIL; SELECTION +0.0253 [+0.0015, +0.0523] sep, WITHIN +0.0034 [-0.0110, +0.0182]; n swapped 1224. Informational (not part of the card control): over 19 further swap seeds both intervals include 0 in 19 of 19; mean swap point SELECTION +0.0002, WITHIN +0.0004.
* h4 s1 (swap seed 0): PASS; SELECTION +0.0067 [-0.0174, +0.0303], WITHIN +0.0085 [-0.0181, +0.0374]; n swapped 1224. Informational (not part of the card control): over 19 further swap seeds both intervals include 0 in 17 of 19; mean swap point SELECTION +0.0020, WITHIN -0.0018.

## h5 sensitivity (all thresholds x0.75 and x1.25; decided at the stated thresholds)
| seed | scale | n SELECTION-CRITICAL | WITHIN | SELECTION | BOTH-MISS | crit diff | bulk diff | sign change WITHIN / SELECTION |
|---|---|---|---|---|---|---|---|---|
| s0 | x1.0 (stated) | 1305 | -0.0320 [-0.0684, -0.0027] sep | -0.0536 [-0.0853, -0.0235] sep | -0.2594 [-0.3408, -0.1832] sep | -0.4795 [-0.6222, -0.3445] sep | -0.1847 [-0.2933, -0.0866] sep | |
| s0 | x0.75 | 1504 | -0.0306 [-0.0561, -0.0062] sep | -0.0496 [-0.0824, -0.0182] sep | -0.2650 [-0.3436, -0.1873] sep | -0.4479 [-0.5755, -0.3239] sep | -0.1723 [-0.2746, -0.0739] sep | False / False |
| s0 | x1.25 | 1153 | -0.0297 [-0.0669, +0.0018] | -0.0792 [-0.1121, -0.0466] sep | -0.2362 [-0.3082, -0.1636] sep | -0.4864 [-0.6416, -0.3456] sep | -0.2143 [-0.3179, -0.1185] sep | False / False |
| s1 | x1.0 (stated) | 1305 | -0.0303 [-0.0653, +0.0024] | +0.0216 [-0.0276, +0.0777] | -0.0458 [-0.1152, +0.0193] | -0.1589 [-0.3043, -0.0117] sep | +0.0702 [-0.0610, +0.2201] | |
| s1 | x0.75 | 1504 | -0.0233 [-0.0582, +0.0084] | +0.0234 [-0.0216, +0.0736] | -0.0545 [-0.1240, +0.0141] | -0.1462 [-0.2784, -0.0160] sep | +0.0997 [-0.0411, +0.2755] | False / False |
| s1 | x1.25 | 1153 | -0.0331 [-0.0714, +0.0003] | +0.0190 [-0.0301, +0.0786] | -0.0403 [-0.1075, +0.0225] | -0.1966 [-0.3592, -0.0318] sep | +0.0771 [-0.0421, +0.2113] | False / False |

h5 named changes (sign changes of SELECTION or WITHIN, and changes of whether an interval excludes 0): s0 x1.25 WITHIN separated -> False

## Logged mode counts (stated thresholds)
* s0: SELECTION-CRITICAL 1305, BULK 1094; longitudinal {'STOP': 141, 'DECEL': 401, 'ACCEL': 457, 'KEEP': 1400}; lateral {'LEFT': 219, 'RIGHT': 429, 'STRAIGHT': 1751}.
* s1: SELECTION-CRITICAL 1305, BULK 1094; longitudinal {'STOP': 141, 'DECEL': 401, 'ACCEL': 457, 'KEEP': 1400}; lateral {'LEFT': 219, 'RIGHT': 429, 'STRAIGHT': 1751}.

## Limits
A post-hoc partition of a banked comparison. Mode thresholds are a choice; h5 shows how much rides on it. The interval answers another draw of episodes, never another training run or another inference run. One training run per arm per seed. The tier is T1-kin on logged frames. Several classes hold few windows (see n); an interval from a handful of episodes is wide. Windows are partitioned post hoc by a function of the logged future and of each arm's own rollout; the partition is therefore not independent of the error it partitions (BOTH-MISS and SELECTION windows are selected on the arm outputs). Deviations R1 to R5 are in HIER_DEFS.md.
