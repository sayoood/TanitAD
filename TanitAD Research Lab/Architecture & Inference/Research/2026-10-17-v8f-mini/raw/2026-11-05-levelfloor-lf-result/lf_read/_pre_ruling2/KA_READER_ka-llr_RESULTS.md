**KNOWN-ANSWER RUN `ka-llr` on EXISTING arrays (no LF rollout): a test of this reader, not a read of arms LF050 / LF075.**

# LF read: arms LF050 and LF075 (the LLR heads as a floor on O's along-track progress, planner C0), T1-kin, generated 2026-11-05T21:20:22+0100

**Headline: P fails at both doses and M1 is separated-better at dose 050 and 075: the floor binds but the head's estimate is too weak to lift the crawl; the level estimator is the bottleneck. Its levers are motion from vision (option (c): T2 re-perception, the A40) or a stronger estimator trained on more data, reported to the PI with the binding and v_f-error instruments. Two doses were read (kappa 0.50 and kappa 0.75): a counted DOMINANT names its dose and says that two were read (A3 card section 4, multiplicity).**

* dose 050 (kappa 0.50): P-FAILS-M1-MOVED -> the floor binds but the head's estimate is too weak to lift the crawl: the level estimator is the bottleneck (levers: motion from vision, option (c) T2 re-perception / the A40, or a stronger estimator on more data)
* dose 075 (kappa 0.75): P-FAILS-M1-MOVED -> the floor binds but the head's estimate is too weak to lift the crawl: the level estimator is the bottleneck (levers: motion from vision, option (c) T2 re-perception / the A40, or a stronger estimator on more data)

Tier: T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2). Estimator: paired episode-cluster bootstrap B=2000 seed 0 (gapdecomp.decompose); answers another draw of EPISODES only. `*` = interval excludes 0. STRATEGIC: PAUSED (PI ITEM 25, 2026-09-19): reported beside the verdict, never scored as a pass of its own; D3 is evaluated over four families as Read A defines it and the three-family D3 is reported beside it

`hold_now` is not read (a frozen frame makes v_hat constant: no information about a frame-reading floor; A3 card section 3).


## Dose 050 (kappa 0.50): **P-FAILS-M1-MOVED**

Next: the floor binds but the head's estimate is too weak to lift the crawl: the level estimator is the bottleneck (levers: motion from vision, option (c) T2 re-perception / the A40, or a stronger estimator on more data).


| rule | s0 | s1 | holds |
|---|---|---|---|
| P: M1 separated-better and share >= 0.5 | sep True, share 0.235 | sep True, share 0.195 | **False** |
| G: ADE6 <= +0.0604 m, no non-crawl row separated-worse | ADE6 +0.0309, worse: DK-TG, DK-VIOL | ADE6 +0.0540, worse: DK-HW, DK-VIOL, along-track MAE | **False** |
| F: abs(RILF-RIC0)[M1] > abs(C0REP-C0)[M1] | 0.0125 vs 0.0362 | 0.0116 vs 0.0038 | **False** |
| LEAK-BORNE: logged share - reindexed share > half the logged share | logged 0.240, reindexed 0.235 | logged 0.192, reindexed 0.195 | **False** |
| D: RILFC - P10 read A (D1, D2, D3 four families) + P + G | D1 True, D2 False, D3 True | D1 True, D2 False, D3 True | **False** |

Beside: P on the logged scene False; D three-family form False; M1 moved (RILF - RIC0 separated-better at >= 1 seed) True.

D failing terms: s0: D2 (separated-worse: M1 crawl share); s1: D2 (separated-worse: M1 crawl share)

Harm rows: none


### 19 rows, dose 050: RILF - RIC0 (reindexed, the deciding contrast)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | +0.0309 [-0.0295, +0.0876] | +0.0540 [-0.0099, +0.1155] |
| speed MAE | LONGITUDINAL | +0.0270 [-0.0126, +0.0686] | +0.0347 [-0.0022, +0.0685] |
| along-track MAE | LONGITUDINAL | +0.0413 [-0.0249, +0.1045] | +0.0942 [+0.0272, +0.1592] * |
| accel MAE | LONGITUDINAL | +0.0569 [-0.0098, +0.1268] | -0.0254 [-0.0739, +0.0237] |
| DK-VIOL | LONGITUDINAL | +0.0252 [+0.0037, +0.0571] * | +0.0286 [+0.0059, +0.0634] * |
| DK-HW | LONGITUDINAL | +0.1321 [-0.0613, +0.3692] | +0.2479 [+0.0469, +0.4682] * |
| DK-TG | LONGITUDINAL | +0.0288 [+0.0052, +0.0555] * | +0.0204 [-0.0252, +0.0743] |
| heading MAE | LATERAL | -0.3881 [-0.6463, -0.1224] * | -1.4117 [-1.9687, -0.9616] * |
| yaw-rate MAE | LATERAL | -2.2962 [-3.3214, -1.3607] * | -5.7947 [-9.0097, -3.3565] * |
| curvature MAE | LATERAL | -0.0278 [-0.0397, -0.0171] * | -0.0524 [-0.0821, -0.0293] * |
| cross-track MAE | LATERAL | -0.0198 [-0.0488, +0.0082] | -0.0436 [-0.0872, -0.0041] * |
| L1 lateral-accel MAE | LATERAL | -0.0478 [-0.2398, +0.1285] | -0.3624 [-0.6062, -0.1685] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0109 [-0.0214, -0.0008] * | -0.0297 [-0.0496, -0.0128] * |
| TAC-LAT | TACTICAL | +0.0133 [+0.0029, +0.0242] * | +0.0358 [+0.0192, +0.0547] * |
| TAC-LON | TACTICAL | +0.0083 [-0.0204, +0.0346] | +0.0175 [-0.0104, +0.0467] |
| S-POS@30 | STRATEGIC | -0.2235 [-0.3863, -0.0576] * | -0.4221 [-0.6363, -0.2337] * |
| S-POS@50 | STRATEGIC | -0.4106 [-0.7013, -0.1409] * | -0.6821 [-1.0535, -0.3380] * |
| S-POS@80 | STRATEGIC | -0.5825 [-1.0930, -0.1025] * | -1.0269 [-1.6660, -0.4330] * |
| M1 crawl share | crawl | -0.0125 [-0.0177, -0.0080] * | -0.0116 [-0.0163, -0.0074] * |

### 19 rows, dose 050: RILFC - P10 (reindexed composite, D)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -1.1141 [-1.3120, -0.9149] * | -0.9574 [-1.1594, -0.7587] * |
| speed MAE | LONGITUDINAL | -0.8508 [-1.0766, -0.6600] * | -0.7782 [-0.9973, -0.5842] * |
| along-track MAE | LONGITUDINAL | -0.5846 [-0.7551, -0.4265] * | -0.4143 [-0.5938, -0.2434] * |
| accel MAE | LONGITUDINAL | -1.1207 [-1.5691, -0.7458] * | -1.1501 [-1.5828, -0.7764] * |
| DK-VIOL | LONGITUDINAL | +0.0252 [-0.0078, +0.0641] | +0.0118 [-0.0224, +0.0538] |
| DK-HW | LONGITUDINAL | -0.3501 [-0.7472, +0.0189] | -0.3280 [-0.7311, +0.0453] |
| DK-TG | LONGITUDINAL | -0.1158 [-0.1982, -0.0498] * | -0.0727 [-0.1688, +0.0308] |
| heading MAE | LATERAL | -2.1468 [-3.2859, -0.9060] * | -1.6609 [-2.6746, -0.6085] * |
| yaw-rate MAE | LATERAL | -12.8428 [-18.1772, -8.3345] * | -12.8162 [-18.1223, -8.3859] * |
| curvature MAE | LATERAL | -0.0360 [-0.0523, -0.0219] * | -0.0361 [-0.0525, -0.0220] * |
| cross-track MAE | LATERAL | -0.9646 [-1.1157, -0.8251] * | -0.9331 [-1.0846, -0.7919] * |
| L1 lateral-accel MAE | LATERAL | -3.5538 [-5.1135, -2.2169] * | -3.5803 [-5.1069, -2.2643] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0012 [-0.0043, +0.0021] | -0.0014 [-0.0046, +0.0021] |
| TAC-LAT | TACTICAL | +0.2051 [+0.1659, +0.2454] * | +0.1822 [+0.1408, +0.2243] * |
| TAC-LON | TACTICAL | +0.1647 [+0.1050, +0.2213] * | +0.1672 [+0.1117, +0.2175] * |
| S-POS@30 | STRATEGIC | -0.6738 [-0.9598, -0.4182] * | -0.9158 [-1.2078, -0.6408] * |
| S-POS@50 | STRATEGIC | -2.8669 [-3.7295, -2.0577] * | -3.3565 [-4.1822, -2.5896] * |
| S-POS@80 | STRATEGIC | -7.5496 [-9.3679, -5.7769] * | -8.7333 [-10.6256, -6.9529] * |
| M1 crawl share | crawl | +0.0474 [+0.0297, +0.0685] * | +0.0504 [+0.0295, +0.0752] * |

### 19 rows, dose 050: LF - C0 (logged, LEAK-BORNE only)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | +0.0313 [-0.0292, +0.0882] | +0.0541 [-0.0101, +0.1158] |
| speed MAE | LONGITUDINAL | +0.0286 [-0.0114, +0.0714] | +0.0338 [-0.0032, +0.0676] |
| along-track MAE | LONGITUDINAL | +0.0419 [-0.0246, +0.1053] | +0.0943 [+0.0269, +0.1596] * |
| accel MAE | LONGITUDINAL | +0.0653 [+0.0006, +0.1356] * | -0.0261 [-0.0773, +0.0264] |
| DK-VIOL | LONGITUDINAL | +0.0252 [+0.0037, +0.0571] * | +0.0269 [+0.0048, +0.0621] * |
| DK-HW | LONGITUDINAL | +0.1247 [-0.0682, +0.3607] | +0.2068 [+0.0132, +0.4209] * |
| DK-TG | LONGITUDINAL | +0.0218 [-0.0018, +0.0471] | +0.0193 [-0.0260, +0.0729] |
| heading MAE | LATERAL | -0.3679 [-0.6184, -0.0943] * | -1.3979 [-1.9288, -0.9507] * |
| yaw-rate MAE | LATERAL | -2.2776 [-3.3323, -1.3246] * | -5.7962 [-9.1085, -3.3108] * |
| curvature MAE | LATERAL | -0.0263 [-0.0377, -0.0163] * | -0.0512 [-0.0815, -0.0282] * |
| cross-track MAE | LATERAL | -0.0205 [-0.0491, +0.0072] | -0.0441 [-0.0874, -0.0045] * |
| L1 lateral-accel MAE | LATERAL | -0.0289 [-0.2117, +0.1463] | -0.3642 [-0.6176, -0.1686] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0108 [-0.0208, -0.0013] * | -0.0247 [-0.0430, -0.0110] * |
| TAC-LAT | TACTICAL | +0.0104 [-0.0004, +0.0217] | +0.0342 [+0.0183, +0.0533] * |
| TAC-LON | TACTICAL | +0.0092 [-0.0196, +0.0359] | +0.0171 [-0.0113, +0.0467] |
| S-POS@30 | STRATEGIC | -0.2438 [-0.4009, -0.0818] * | -0.4235 [-0.6328, -0.2348] * |
| S-POS@50 | STRATEGIC | -0.4351 [-0.7186, -0.1622] * | -0.6789 [-1.0300, -0.3408] * |
| S-POS@80 | STRATEGIC | -0.5731 [-1.0831, -0.1029] * | -0.9908 [-1.5967, -0.4183] * |
| M1 crawl share | crawl | -0.0128 [-0.0179, -0.0081] * | -0.0114 [-0.0160, -0.0073] * |

## Dose 075 (kappa 0.75): **P-FAILS-M1-MOVED**

Next: the floor binds but the head's estimate is too weak to lift the crawl: the level estimator is the bottleneck (levers: motion from vision, option (c) T2 re-perception / the A40, or a stronger estimator on more data).


| rule | s0 | s1 | holds |
|---|---|---|---|
| P: M1 separated-better and share >= 0.5 | sep True, share 0.235 | sep True, share 0.195 | **False** |
| G: ADE6 <= +0.0604 m, no non-crawl row separated-worse | ADE6 +0.0309, worse: DK-TG, DK-VIOL | ADE6 +0.0540, worse: DK-HW, DK-VIOL, along-track MAE | **False** |
| F: abs(RILF-RIC0)[M1] > abs(C0REP-C0)[M1] | 0.0125 vs 0.0362 | 0.0116 vs 0.0038 | **False** |
| LEAK-BORNE: logged share - reindexed share > half the logged share | logged 0.240, reindexed 0.235 | logged 0.192, reindexed 0.195 | **False** |
| D: RILFC - P10 read A (D1, D2, D3 four families) + P + G | D1 True, D2 False, D3 True | D1 True, D2 False, D3 True | **False** |

Beside: P on the logged scene False; D three-family form False; M1 moved (RILF - RIC0 separated-better at >= 1 seed) True.

D failing terms: s0: D2 (separated-worse: M1 crawl share); s1: D2 (separated-worse: M1 crawl share)

Harm rows: none


### 19 rows, dose 075: RILF - RIC0 (reindexed, the deciding contrast)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | +0.0309 [-0.0295, +0.0876] | +0.0540 [-0.0099, +0.1155] |
| speed MAE | LONGITUDINAL | +0.0270 [-0.0126, +0.0686] | +0.0347 [-0.0022, +0.0685] |
| along-track MAE | LONGITUDINAL | +0.0413 [-0.0249, +0.1045] | +0.0942 [+0.0272, +0.1592] * |
| accel MAE | LONGITUDINAL | +0.0569 [-0.0098, +0.1268] | -0.0254 [-0.0739, +0.0237] |
| DK-VIOL | LONGITUDINAL | +0.0252 [+0.0037, +0.0571] * | +0.0286 [+0.0059, +0.0634] * |
| DK-HW | LONGITUDINAL | +0.1321 [-0.0613, +0.3692] | +0.2479 [+0.0469, +0.4682] * |
| DK-TG | LONGITUDINAL | +0.0288 [+0.0052, +0.0555] * | +0.0204 [-0.0252, +0.0743] |
| heading MAE | LATERAL | -0.3881 [-0.6463, -0.1224] * | -1.4117 [-1.9687, -0.9616] * |
| yaw-rate MAE | LATERAL | -2.2962 [-3.3214, -1.3607] * | -5.7947 [-9.0097, -3.3565] * |
| curvature MAE | LATERAL | -0.0278 [-0.0397, -0.0171] * | -0.0524 [-0.0821, -0.0293] * |
| cross-track MAE | LATERAL | -0.0198 [-0.0488, +0.0082] | -0.0436 [-0.0872, -0.0041] * |
| L1 lateral-accel MAE | LATERAL | -0.0478 [-0.2398, +0.1285] | -0.3624 [-0.6062, -0.1685] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0109 [-0.0214, -0.0008] * | -0.0297 [-0.0496, -0.0128] * |
| TAC-LAT | TACTICAL | +0.0133 [+0.0029, +0.0242] * | +0.0358 [+0.0192, +0.0547] * |
| TAC-LON | TACTICAL | +0.0083 [-0.0204, +0.0346] | +0.0175 [-0.0104, +0.0467] |
| S-POS@30 | STRATEGIC | -0.2235 [-0.3863, -0.0576] * | -0.4221 [-0.6363, -0.2337] * |
| S-POS@50 | STRATEGIC | -0.4106 [-0.7013, -0.1409] * | -0.6821 [-1.0535, -0.3380] * |
| S-POS@80 | STRATEGIC | -0.5825 [-1.0930, -0.1025] * | -1.0269 [-1.6660, -0.4330] * |
| M1 crawl share | crawl | -0.0125 [-0.0177, -0.0080] * | -0.0116 [-0.0163, -0.0074] * |

### 19 rows, dose 075: RILFC - P10 (reindexed composite, D)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -1.1141 [-1.3120, -0.9149] * | -0.9574 [-1.1594, -0.7587] * |
| speed MAE | LONGITUDINAL | -0.8508 [-1.0766, -0.6600] * | -0.7782 [-0.9973, -0.5842] * |
| along-track MAE | LONGITUDINAL | -0.5846 [-0.7551, -0.4265] * | -0.4143 [-0.5938, -0.2434] * |
| accel MAE | LONGITUDINAL | -1.1207 [-1.5691, -0.7458] * | -1.1501 [-1.5828, -0.7764] * |
| DK-VIOL | LONGITUDINAL | +0.0252 [-0.0078, +0.0641] | +0.0118 [-0.0224, +0.0538] |
| DK-HW | LONGITUDINAL | -0.3501 [-0.7472, +0.0189] | -0.3280 [-0.7311, +0.0453] |
| DK-TG | LONGITUDINAL | -0.1158 [-0.1982, -0.0498] * | -0.0727 [-0.1688, +0.0308] |
| heading MAE | LATERAL | -2.1468 [-3.2859, -0.9060] * | -1.6609 [-2.6746, -0.6085] * |
| yaw-rate MAE | LATERAL | -12.8428 [-18.1772, -8.3345] * | -12.8162 [-18.1223, -8.3859] * |
| curvature MAE | LATERAL | -0.0360 [-0.0523, -0.0219] * | -0.0361 [-0.0525, -0.0220] * |
| cross-track MAE | LATERAL | -0.9646 [-1.1157, -0.8251] * | -0.9331 [-1.0846, -0.7919] * |
| L1 lateral-accel MAE | LATERAL | -3.5538 [-5.1135, -2.2169] * | -3.5803 [-5.1069, -2.2643] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0012 [-0.0043, +0.0021] | -0.0014 [-0.0046, +0.0021] |
| TAC-LAT | TACTICAL | +0.2051 [+0.1659, +0.2454] * | +0.1822 [+0.1408, +0.2243] * |
| TAC-LON | TACTICAL | +0.1647 [+0.1050, +0.2213] * | +0.1672 [+0.1117, +0.2175] * |
| S-POS@30 | STRATEGIC | -0.6738 [-0.9598, -0.4182] * | -0.9158 [-1.2078, -0.6408] * |
| S-POS@50 | STRATEGIC | -2.8669 [-3.7295, -2.0577] * | -3.3565 [-4.1822, -2.5896] * |
| S-POS@80 | STRATEGIC | -7.5496 [-9.3679, -5.7769] * | -8.7333 [-10.6256, -6.9529] * |
| M1 crawl share | crawl | +0.0474 [+0.0297, +0.0685] * | +0.0504 [+0.0295, +0.0752] * |

### 19 rows, dose 075: LF - C0 (logged, LEAK-BORNE only)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | +0.0313 [-0.0292, +0.0882] | +0.0541 [-0.0101, +0.1158] |
| speed MAE | LONGITUDINAL | +0.0286 [-0.0114, +0.0714] | +0.0338 [-0.0032, +0.0676] |
| along-track MAE | LONGITUDINAL | +0.0419 [-0.0246, +0.1053] | +0.0943 [+0.0269, +0.1596] * |
| accel MAE | LONGITUDINAL | +0.0653 [+0.0006, +0.1356] * | -0.0261 [-0.0773, +0.0264] |
| DK-VIOL | LONGITUDINAL | +0.0252 [+0.0037, +0.0571] * | +0.0269 [+0.0048, +0.0621] * |
| DK-HW | LONGITUDINAL | +0.1247 [-0.0682, +0.3607] | +0.2068 [+0.0132, +0.4209] * |
| DK-TG | LONGITUDINAL | +0.0218 [-0.0018, +0.0471] | +0.0193 [-0.0260, +0.0729] |
| heading MAE | LATERAL | -0.3679 [-0.6184, -0.0943] * | -1.3979 [-1.9288, -0.9507] * |
| yaw-rate MAE | LATERAL | -2.2776 [-3.3323, -1.3246] * | -5.7962 [-9.1085, -3.3108] * |
| curvature MAE | LATERAL | -0.0263 [-0.0377, -0.0163] * | -0.0512 [-0.0815, -0.0282] * |
| cross-track MAE | LATERAL | -0.0205 [-0.0491, +0.0072] | -0.0441 [-0.0874, -0.0045] * |
| L1 lateral-accel MAE | LATERAL | -0.0289 [-0.2117, +0.1463] | -0.3642 [-0.6176, -0.1686] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0108 [-0.0208, -0.0013] * | -0.0247 [-0.0430, -0.0110] * |
| TAC-LAT | TACTICAL | +0.0104 [-0.0004, +0.0217] | +0.0342 [+0.0183, +0.0533] * |
| TAC-LON | TACTICAL | +0.0092 [-0.0196, +0.0359] | +0.0171 [-0.0113, +0.0467] |
| S-POS@30 | STRATEGIC | -0.2438 [-0.4009, -0.0818] * | -0.4235 [-0.6328, -0.2348] * |
| S-POS@50 | STRATEGIC | -0.4351 [-0.7186, -0.1622] * | -0.6789 [-1.0300, -0.3408] * |
| S-POS@80 | STRATEGIC | -0.5731 [-1.0831, -0.1029] * | -0.9908 [-1.5967, -0.4183] * |
| M1 crawl share | crawl | -0.0128 [-0.0179, -0.0081] * | -0.0114 [-0.0160, -0.0073] * |

## Controls

| control | PASS |
|---|---|
| reused_arrays | True |

## Variance named

* episodes: answered by the paired episode-cluster bootstrap (every interval here)
* training_run: O is C0's own checkpoint in both arms of every LF - C0 contrast, so O's training variance is absent from it; the heads come from one LLR run per seed (not measured); rule F reads against the C0REP floor
* inference: the planner and F are deterministic: no inference seed applies
