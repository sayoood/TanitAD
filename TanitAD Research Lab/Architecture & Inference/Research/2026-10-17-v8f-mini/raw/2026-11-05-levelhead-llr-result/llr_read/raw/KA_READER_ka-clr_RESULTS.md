**KNOWN-ANSWER RUN `ka-clr` on EXISTING arrays (no LLR rollout): a test of this reader, not a read of arm LLR.**

# LLR read: arm LLR (lead-aware head + level head, token wiring, inside RAPR-P01's recovery recipe), T1-kin, generated 2026-11-05T19:30:16+0100

**Outcome: P-FAILS-FAMILY-EXHAUSTED**

Next (A2 card, Outcomes): the vision-head family on frozen single-frame features is exhausted at these forms (A2 card, Outcomes: LLR P fails); the remaining levers are the named blockers of option (c) (T2 re-perception, the A40), reported to the PI with the numbers.
Notes: G fails too (the guards read worse alongside the failed P); attribution LLR - CLR on M1 (reindexed) separated-better at 0 of 2 seeds (reported; the card's outcome does not branch on it).

Tier: T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2). Estimator: paired episode-cluster bootstrap B=2000 seed 0 (gapdecomp.decompose); answers another draw of EPISODES only. `*` = interval excludes 0. STRATEGIC: PAUSED (PI ITEM 25, 2026-09-19): reported beside the verdict, never scored as a pass of its own; D3 is evaluated over four families as Read A defines it and the three-family D3 is reported beside it

## Rules (CL card section 4 by reference, CL := LLR; attribution and outcomes of the A2 card; deciding scene = reindexed)

| rule | s0 | s1 | holds |
|---|---|---|---|
| P: M1 separated-better and share >= 0.5 | sep True, share 0.233 | sep True, share 0.130 | **False** |
| G: ADE6 <= +0.0604 m, no non-crawl row separated-worse | ADE6 +0.0323, worse: DK-TG, DK-VIOL | ADE6 +0.0585, worse: ADE6, DK-VIOL, along-track MAE | **False** |
| F: abs(RILLR-RIC0)[M1] > abs(C0REP-C0)[M1] | 0.0124 vs 0.0362 | 0.0077 vs 0.0038 | **False** |
| LEAK-BORNE: logged share - reindexed share > half the logged share | logged 0.229, reindexed 0.233 | logged 0.127, reindexed 0.130 | **False** |
| D: RILLRC - P10 read A (D1, D2, D3 four families) + P + G | D1 True, D2 False, D3 True | D1 True, D2 False, D3 True | **False** |
| ATTRIBUTION (reported; the outcome does not branch on it): M1 of RILLR - RICLR (reindexed) separated-better at >= 1 seed | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | **False** |
| beside: M1 of LLR - CLR (logged), CLR = the banked rollout | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0 of 2 seeds separated-better |

Beside: P mean-over-seeds form False (mean share 0.181); P on logged scene False; D three-family form False; F logged form False; hold_now (FZLLR vs FZC0) reading PARTIAL, shares s0 0.235, s1 0.110 (never deciding).

## D failing terms

* s0: D2 (separated-worse: M1 crawl share)
* s1: D2 (separated-worse: M1 crawl share)

## 19 rows: RILLR - RIC0 (reindexed, the deciding contrast)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | +0.0323 [-0.0224, +0.0813] | +0.0585 [+0.0040, +0.1098] * |
| speed MAE | LONGITUDINAL | +0.0102 [-0.0197, +0.0398] | +0.0271 [-0.0056, +0.0588] |
| along-track MAE | LONGITUDINAL | +0.0507 [-0.0049, +0.1055] | +0.0937 [+0.0395, +0.1475] * |
| accel MAE | LONGITUDINAL | -0.0017 [-0.0604, +0.0504] | -0.0014 [-0.0530, +0.0507] |
| DK-VIOL | LONGITUDINAL | +0.0269 [+0.0071, +0.0552] * | +0.0185 [+0.0017, +0.0421] * |
| DK-HW | LONGITUDINAL | +0.1456 [-0.0111, +0.3210] | +0.1249 [-0.0525, +0.3097] |
| DK-TG | LONGITUDINAL | +0.0515 [+0.0050, +0.1261] * | -0.0320 [-0.1042, +0.0192] |
| heading MAE | LATERAL | -0.6007 [-0.9182, -0.2758] * | -0.9769 [-1.3748, -0.6397] * |
| yaw-rate MAE | LATERAL | -2.4881 [-3.4907, -1.5642] * | -4.1139 [-6.6423, -2.1421] * |
| curvature MAE | LATERAL | -0.0315 [-0.0430, -0.0210] * | -0.0345 [-0.0575, -0.0168] * |
| cross-track MAE | LATERAL | -0.0267 [-0.0586, +0.0027] | -0.0331 [-0.0739, +0.0045] |
| L1 lateral-accel MAE | LATERAL | -0.1406 [-0.3493, +0.0470] | -0.3399 [-0.6025, -0.1149] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0182 [-0.0287, -0.0098] * | -0.0193 [-0.0348, -0.0060] * |
| TAC-LAT | TACTICAL | +0.0142 [+0.0033, +0.0254] * | +0.0392 [+0.0216, +0.0608] * |
| TAC-LON | TACTICAL | -0.0021 [-0.0246, +0.0196] | +0.0188 [-0.0096, +0.0471] |
| S-POS@30 | STRATEGIC | -0.2586 [-0.3926, -0.1320] * | -0.5166 [-0.7280, -0.3323] * |
| S-POS@50 | STRATEGIC | -0.4309 [-0.6626, -0.2081] * | -0.8016 [-1.1666, -0.4855] * |
| S-POS@80 | STRATEGIC | -0.6855 [-1.0730, -0.3042] * | -1.2148 [-1.8401, -0.6877] * |
| M1 crawl share | crawl | -0.0124 [-0.0173, -0.0076] * | -0.0077 [-0.0132, -0.0028] * |

## 19 rows: LLR - C0 (logged)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | +0.0317 [-0.0228, +0.0810] | +0.0577 [+0.0033, +0.1092] * |
| speed MAE | LONGITUDINAL | +0.0109 [-0.0190, +0.0406] | +0.0242 [-0.0081, +0.0557] |
| along-track MAE | LONGITUDINAL | +0.0507 [-0.0058, +0.1055] | +0.0936 [+0.0393, +0.1474] * |
| accel MAE | LONGITUDINAL | +0.0029 [-0.0562, +0.0551] | -0.0013 [-0.0528, +0.0526] |
| DK-VIOL | LONGITUDINAL | +0.0269 [+0.0071, +0.0552] * | +0.0185 [+0.0017, +0.0421] * |
| DK-HW | LONGITUDINAL | +0.1436 [-0.0105, +0.3176] | +0.1206 [-0.0592, +0.3099] |
| DK-TG | LONGITUDINAL | +0.0465 [+0.0038, +0.1119] * | -0.0329 [-0.1054, +0.0186] |
| heading MAE | LATERAL | -0.6017 [-0.9182, -0.2756] * | -0.9612 [-1.3489, -0.6274] * |
| yaw-rate MAE | LATERAL | -2.5520 [-3.5703, -1.5821] * | -4.1900 [-6.7549, -2.1807] * |
| curvature MAE | LATERAL | -0.0307 [-0.0428, -0.0201] * | -0.0351 [-0.0581, -0.0177] * |
| cross-track MAE | LATERAL | -0.0271 [-0.0591, +0.0019] | -0.0340 [-0.0752, +0.0032] |
| L1 lateral-accel MAE | LATERAL | -0.1295 [-0.3286, +0.0523] | -0.3397 [-0.6100, -0.1071] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0178 [-0.0287, -0.0096] * | -0.0139 [-0.0263, -0.0038] * |
| TAC-LAT | TACTICAL | +0.0113 [+0.0004, +0.0233] * | +0.0388 [+0.0221, +0.0601] * |
| TAC-LON | TACTICAL | -0.0021 [-0.0242, +0.0196] | +0.0171 [-0.0113, +0.0455] |
| S-POS@30 | STRATEGIC | -0.2738 [-0.4024, -0.1534] * | -0.5144 [-0.7223, -0.3357] * |
| S-POS@50 | STRATEGIC | -0.4417 [-0.6599, -0.2294] * | -0.7938 [-1.1466, -0.4895] * |
| S-POS@80 | STRATEGIC | -0.6545 [-1.0130, -0.2927] * | -1.1825 [-1.8090, -0.6670] * |
| M1 crawl share | crawl | -0.0122 [-0.0171, -0.0076] * | -0.0075 [-0.0126, -0.0031] * |

## 19 rows: RILLRC - P10 (reindexed composite, D)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -1.1242 [-1.3223, -0.9308] * | -0.9545 [-1.1506, -0.7548] * |
| speed MAE | LONGITUDINAL | -0.8521 [-1.0818, -0.6550] * | -0.7750 [-0.9966, -0.5792] * |
| along-track MAE | LONGITUDINAL | -0.5863 [-0.7550, -0.4218] * | -0.4047 [-0.5833, -0.2252] * |
| accel MAE | LONGITUDINAL | -1.1399 [-1.5995, -0.7555] * | -1.1599 [-1.6017, -0.7922] * |
| DK-VIOL | LONGITUDINAL | +0.0269 [-0.0047, +0.0626] | +0.0017 [-0.0276, +0.0349] |
| DK-HW | LONGITUDINAL | -0.3661 [-0.7732, -0.0267] * | -0.3850 [-0.7763, -0.0334] * |
| DK-TG | LONGITUDINAL | -0.0905 [-0.1642, -0.0293] * | -0.1295 [-0.2187, -0.0608] * |
| heading MAE | LATERAL | -2.7186 [-3.5936, -1.8423] * | -1.3465 [-2.4321, -0.1376] * |
| yaw-rate MAE | LATERAL | -13.1148 [-18.6123, -8.5542] * | -12.6447 [-17.8519, -8.2452] * |
| curvature MAE | LATERAL | -0.0366 [-0.0531, -0.0224] * | -0.0354 [-0.0517, -0.0213] * |
| cross-track MAE | LATERAL | -0.9752 [-1.1306, -0.8368] * | -0.9419 [-1.0930, -0.8102] * |
| L1 lateral-accel MAE | LATERAL | -3.6355 [-5.2070, -2.2897] * | -3.5731 [-5.1044, -2.2653] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0014 [-0.0045, +0.0018] | -0.0003 [-0.0040, +0.0039] |
| TAC-LAT | TACTICAL | +0.2130 [+0.1743, +0.2527] * | +0.1822 [+0.1426, +0.2230] * |
| TAC-LON | TACTICAL | +0.1551 [+0.0958, +0.2117] * | +0.1634 [+0.1066, +0.2171] * |
| S-POS@30 | STRATEGIC | -0.6621 [-0.9329, -0.4156] * | -0.9607 [-1.2518, -0.6979] * |
| S-POS@50 | STRATEGIC | -2.7924 [-3.6268, -2.0142] * | -3.4100 [-4.2509, -2.6427] * |
| S-POS@80 | STRATEGIC | -7.5973 [-9.3560, -5.8520] * | -8.7860 [-10.7291, -6.9827] * |
| M1 crawl share | crawl | +0.0479 [+0.0300, +0.0689] * | +0.0527 [+0.0308, +0.0783] * |

## 19 rows: RILLR - RICLR (reindexed, ATTRIBUTION: LLR against CLR's own checkpoint, the banked CLR arrays)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| speed MAE | LONGITUDINAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| along-track MAE | LONGITUDINAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| accel MAE | LONGITUDINAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| DK-VIOL | LONGITUDINAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| DK-HW | LONGITUDINAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| DK-TG | LONGITUDINAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| heading MAE | LATERAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| yaw-rate MAE | LATERAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| curvature MAE | LATERAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| cross-track MAE | LATERAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| L1 lateral-accel MAE | LATERAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| TAC-LAT | TACTICAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| TAC-LON | TACTICAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| S-POS@30 | STRATEGIC | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| S-POS@50 | STRATEGIC | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| S-POS@80 | STRATEGIC | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| M1 crawl share | crawl | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

## 19 rows: LLR - CLR (logged, attribution beside; CLR = the banked rollout)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| speed MAE | LONGITUDINAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| along-track MAE | LONGITUDINAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| accel MAE | LONGITUDINAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| DK-VIOL | LONGITUDINAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| DK-HW | LONGITUDINAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| DK-TG | LONGITUDINAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| heading MAE | LATERAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| yaw-rate MAE | LATERAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| curvature MAE | LATERAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| cross-track MAE | LATERAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| L1 lateral-accel MAE | LATERAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| TAC-LAT | TACTICAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| TAC-LON | TACTICAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| S-POS@30 | STRATEGIC | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| S-POS@50 | STRATEGIC | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| S-POS@80 | STRATEGIC | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| M1 crawl share | crawl | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

## Families (separated-better / separated-worse row counts)

| contrast | family | s0 better / worse | s1 better / worse |
|---|---|---|---|
| RILLR_minus_RIC0 | LONGITUDINAL | 0 / 2 (of 6) | 0 / 2 (of 6) |
| RILLR_minus_RIC0 | LATERAL | 4 / 0 (of 6) | 5 / 0 (of 6) |
| RILLR_minus_RIC0 | TACTICAL | 1 / 0 (of 2) | 1 / 0 (of 2) |
| RILLR_minus_RIC0 | STRATEGIC | 3 / 0 (of 3) | 3 / 0 (of 3) |
| LLR_minus_C0 | LONGITUDINAL | 0 / 2 (of 6) | 0 / 2 (of 6) |
| LLR_minus_C0 | LATERAL | 4 / 0 (of 6) | 5 / 0 (of 6) |
| LLR_minus_C0 | TACTICAL | 1 / 0 (of 2) | 1 / 0 (of 2) |
| LLR_minus_C0 | STRATEGIC | 3 / 0 (of 3) | 3 / 0 (of 3) |
| RILLRC_minus_P10 | LONGITUDINAL | 5 / 0 (of 6) | 5 / 0 (of 6) |
| RILLRC_minus_P10 | LATERAL | 5 / 0 (of 6) | 5 / 0 (of 6) |
| RILLRC_minus_P10 | TACTICAL | 2 / 0 (of 2) | 2 / 0 (of 2) |
| RILLRC_minus_P10 | STRATEGIC | 3 / 0 (of 3) | 3 / 0 (of 3) |
| LLRC_minus_P10 | LONGITUDINAL | 5 / 0 (of 6) | 5 / 0 (of 6) |
| LLRC_minus_P10 | LATERAL | 5 / 0 (of 6) | 5 / 0 (of 6) |
| LLRC_minus_P10 | TACTICAL | 2 / 0 (of 2) | 2 / 0 (of 2) |
| LLRC_minus_P10 | STRATEGIC | 3 / 0 (of 3) | 3 / 0 (of 3) |
| FZLLR_minus_FZC0 | LONGITUDINAL | 0 / 2 (of 6) | 0 / 1 (of 6) |
| FZLLR_minus_FZC0 | LATERAL | 5 / 0 (of 6) | 6 / 0 (of 6) |
| FZLLR_minus_FZC0 | TACTICAL | 1 / 0 (of 2) | 1 / 0 (of 2) |
| FZLLR_minus_FZC0 | STRATEGIC | 3 / 0 (of 3) | 3 / 0 (of 3) |
| RILLR_minus_RICLR | LONGITUDINAL | 0 / 0 (of 6) | 0 / 0 (of 6) |
| RILLR_minus_RICLR | LATERAL | 0 / 0 (of 6) | 0 / 0 (of 6) |
| RILLR_minus_RICLR | TACTICAL | 0 / 0 (of 2) | 0 / 0 (of 2) |
| RILLR_minus_RICLR | STRATEGIC | 0 / 0 (of 3) | 0 / 0 (of 3) |
| LLR_minus_CLR | LONGITUDINAL | 0 / 0 (of 6) | 0 / 0 (of 6) |
| LLR_minus_CLR | LATERAL | 0 / 0 (of 6) | 0 / 0 (of 6) |
| LLR_minus_CLR | TACTICAL | 0 / 0 (of 2) | 0 / 0 (of 2) |
| LLR_minus_CLR | STRATEGIC | 0 / 0 (of 3) | 0 / 0 (of 3) |

## Controls

| control | PASS |
|---|---|
| reused_arrays | True |

## Variance named

* episodes: answered by the paired episode-cluster bootstrap (every interval here)
* training_run: answered ONLY by the C0REP floor (rule F); a separated CI is necessary, not sufficient (H-ESTIM-SEED-1)
* inference: the planner is deterministic at eval (no sampling): no inference seed applies
