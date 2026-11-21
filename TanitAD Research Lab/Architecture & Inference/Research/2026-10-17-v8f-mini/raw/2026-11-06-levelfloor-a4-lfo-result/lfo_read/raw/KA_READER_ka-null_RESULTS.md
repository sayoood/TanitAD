**KNOWN-ANSWER RUN `ka-null` on EXISTING arrays (no LFO rollout): a test of this reader, not a read of arm LFO075.**

# LFO read: arm LFO075 (LF075 with the level estimate replaced by the LOGGED speed), CEILING, T1-kin, generated 2026-11-06T04:13:44+0100

**Outcome: CEILING-SHORT (CEILING diagnostic).**

**Headline: CEILING-SHORT (CEILING diagnostic): P_O fails (share closed s0 -0.000, s1 -0.000; bar 0.5), G_O holds (ADE6 s0 +0.0000 m, s1 +0.0000 m; guard +0.0604 m). CEILING: the level input is the human's logged route speed, a privileged channel that is NOT admissible at inference. This is not a driving result, not a capability claim and not a counted DOMINANT; it ranks levers (A4 card).**

Tier: T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2); CEILING: privileged level input. Estimator: paired episode-cluster bootstrap B=2000 seed 0 (gapdecomp.decompose); answers another draw of EPISODES only. `*` = interval excludes 0. STRATEGIC: PAUSED (PI ITEM 25, 2026-09-19): reported beside the verdict, never scored as a pass of its own; D3 is evaluated over four families as Read A defines it and the three-family D3 is reported beside it

`logged` is not read (A4 section 3: the level input is the human's speed by construction, so the LEAK-BORNE rule has no question left; disclosed). `hold_now` is not read.

Next: an exact level at kappa 0.75 does not close half the crawl, so a floor on progress is not the form that closes M1 at this dose; the estimator levers are NOT signalled for this form; what remains is reported to the PI: T2 re-perception (option (c)), or the operator at kappa 1 in its own addendum.
Notes: M1 moved (RILFO - RIC0 separated-better at >= 1 seed): False.

| rule | s0 | s1 | holds |
|---|---|---|---|
| P_O: M1 separated-better and share >= 0.5 | sep False, share 0.000 | sep False, share 0.000 | **False** |
| G_O: ADE6 <= +0.0604 m, no non-crawl row separated-worse | ADE6 +0.0000, worse: none | ADE6 +0.0000, worse: none | **True** |
| F_O (beside the label): abs(RILFO-RIC0)[M1] > abs(C0REP-C0)[M1] | 0.0000 vs 0.0362 | 0.0000 vs 0.0038 | **False** |
| CEILING of D (never counted): RILFOC - P10 read A (D1, D2, D3 four families) | D1 True, D2 False, D3 True | D1 True, D2 False, D3 True | read A both seeds False (counted: False) |

Beside: three-family form of the ceiling of D False; M1 moved (RILFO - RIC0 separated-better at >= 1 seed) False.

Ceiling-of-D failing terms: s0: D2 (separated-worse: M1 crawl share); s1: D2 (separated-worse: M1 crawl share)

Failing rows (G_O, whatever the label): none


### 19 rows: RILFO - RIC0 (reindexed, the deciding contrast)

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

### 19 rows: ATTRIBUTION: RILFO - RILF075 (the level estimate's error at a fixed operator, dose, planner and lead cap; reported, never deciding)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -0.3683 [-0.5400, -0.2134] * | -0.2172 [-0.3406, -0.0971] * |
| speed MAE | LONGITUDINAL | -0.1250 [-0.2269, -0.0378] * | -0.0379 [-0.0938, +0.0199] |
| along-track MAE | LONGITUDINAL | -0.2753 [-0.4183, -0.1348] * | -0.1657 [-0.2827, -0.0462] * |
| accel MAE | LONGITUDINAL | +0.0357 [-0.0847, +0.1515] | +0.0379 [-0.0252, +0.1012] |
| DK-VIOL | LONGITUDINAL | -0.0269 [-0.0541, -0.0058] * | -0.0269 [-0.0513, -0.0084] * |
| DK-HW | LONGITUDINAL | -0.2961 [-0.6715, -0.0513] * | -0.1978 [-0.4163, -0.0052] * |
| DK-TG | LONGITUDINAL | -0.0779 [-0.2087, +0.0138] | -0.0351 [-0.1391, +0.0819] |
| heading MAE | LATERAL | +0.9023 [+0.3415, +1.5232] * | +0.9247 [+0.3716, +1.7145] * |
| yaw-rate MAE | LATERAL | +7.4724 [+4.6626, +11.1972] * | +7.5591 [+4.3665, +11.8067] * |
| curvature MAE | LATERAL | +0.0762 [+0.0485, +0.1084] * | +0.0800 [+0.0458, +0.1217] * |
| cross-track MAE | LATERAL | -0.1589 [-0.2533, -0.0843] * | -0.0868 [-0.1426, -0.0403] * |
| L1 lateral-accel MAE | LATERAL | +0.2673 [-0.1396, +0.7324] | +0.3303 [+0.1128, +0.6195] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0380 [+0.0184, +0.0654] * | +0.0403 [+0.0152, +0.0695] * |
| TAC-LAT | TACTICAL | -0.0283 [-0.0421, -0.0158] * | -0.0313 [-0.0446, -0.0184] * |
| TAC-LON | TACTICAL | -0.0013 [-0.0229, +0.0204] | -0.0275 [-0.0475, -0.0096] * |
| S-POS@30 | STRATEGIC | +0.3952 [+0.1979, +0.6138] * | +0.5271 [+0.2974, +0.7941] * |
| S-POS@50 | STRATEGIC | +0.7201 [+0.3769, +1.1097] * | +0.8314 [+0.4553, +1.2779] * |
| S-POS@80 | STRATEGIC | +1.2356 [+0.6888, +1.8567] * | +1.2768 [+0.6579, +2.0227] * |
| M1 crawl share | crawl | +0.0211 [+0.0112, +0.0334] * | +0.0234 [+0.0131, +0.0363] * |

### 19 rows: RILFOC - P10 (reindexed composite, the CEILING of D, never counted)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -1.1190 [-1.3168, -0.9317] * | -1.0066 [-1.2041, -0.8140] * |
| speed MAE | LONGITUDINAL | -0.8483 [-1.0788, -0.6500] * | -0.8059 [-1.0255, -0.6065] * |
| along-track MAE | LONGITUDINAL | -0.6271 [-0.8003, -0.4654] * | -0.5078 [-0.6866, -0.3370] * |
| accel MAE | LONGITUDINAL | -1.1260 [-1.5667, -0.7514] * | -1.1388 [-1.5765, -0.7725] * |
| DK-VIOL | LONGITUDINAL | -0.0034 [-0.0259, +0.0184] | -0.0151 [-0.0343, +0.0035] |
| DK-HW | LONGITUDINAL | -0.4782 [-0.8708, -0.1506] * | -0.5274 [-0.8882, -0.2614] * |
| DK-TG | LONGITUDINAL | -0.1141 [-0.1813, -0.0623] * | -0.1330 [-0.2106, -0.0714] * |
| heading MAE | LATERAL | -1.6234 [-2.6178, -0.5062] * | -1.0985 [-2.0588, -0.0322] * |
| yaw-rate MAE | LATERAL | -12.6815 [-18.0451, -8.1958] * | -12.3250 [-17.5593, -7.9859] * |
| curvature MAE | LATERAL | -0.0347 [-0.0509, -0.0208] * | -0.0319 [-0.0479, -0.0181] * |
| cross-track MAE | LATERAL | -0.9014 [-1.0484, -0.7701] * | -0.8797 [-1.0170, -0.7529] * |
| L1 lateral-accel MAE | LATERAL | -3.5435 [-5.0974, -2.2030] * | -3.5436 [-5.0744, -2.2392] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0001 [-0.0030, +0.0034] | +0.0019 [-0.0020, +0.0064] |
| TAC-LAT | TACTICAL | +0.1876 [+0.1483, +0.2282] * | +0.1455 [+0.1034, +0.1897] * |
| TAC-LON | TACTICAL | +0.1538 [+0.0916, +0.2175] * | +0.1467 [+0.0883, +0.2043] * |
| S-POS@30 | STRATEGIC | -0.7606 [-1.0518, -0.5032] * | -0.8581 [-1.1543, -0.5746] * |
| S-POS@50 | STRATEGIC | -3.0065 [-3.8752, -2.2283] * | -3.3098 [-4.1397, -2.5430] * |
| S-POS@80 | STRATEGIC | -7.8456 [-9.6499, -6.0467] * | -8.9263 [-10.8606, -7.0816] * |
| M1 crawl share | crawl | +0.0591 [+0.0403, +0.0812] * | +0.0619 [+0.0395, +0.0889] * |

### 19 rows: attribution on the composite: RILFOC - RILF075C (beside)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -0.3664 [-0.5356, -0.2050] * | -0.2024 [-0.3258, -0.0811] * |
| speed MAE | LONGITUDINAL | -0.1470 [-0.2407, -0.0619] * | -0.0555 [-0.1231, +0.0111] |
| along-track MAE | LONGITUDINAL | -0.3205 [-0.4758, -0.1696] * | -0.1674 [-0.2836, -0.0479] * |
| accel MAE | LONGITUDINAL | -0.0745 [-0.1577, -0.0024] * | -0.0371 [-0.1253, +0.0225] |
| DK-VIOL | LONGITUDINAL | -0.0420 [-0.0806, -0.0126] * | -0.0336 [-0.0619, -0.0106] * |
| DK-HW | LONGITUDINAL | -0.4062 [-0.8353, -0.0838] * | -0.2905 [-0.5321, -0.0857] * |
| DK-TG | LONGITUDINAL | -0.1220 [-0.3085, +0.0107] | -0.1146 [-0.2482, -0.0244] * |
| heading MAE | LATERAL | +0.7089 [+0.2953, +1.2011] * | +0.8639 [+0.2355, +1.7212] * |
| yaw-rate MAE | LATERAL | +0.6464 [+0.3116, +0.9914] * | +0.7541 [+0.4434, +1.0712] * |
| curvature MAE | LATERAL | +0.0116 [+0.0076, +0.0160] * | +0.0121 [+0.0078, +0.0170] * |
| cross-track MAE | LATERAL | -0.1042 [-0.1627, -0.0534] * | -0.0608 [-0.1068, -0.0220] * |
| L1 lateral-accel MAE | LATERAL | -0.0891 [-0.1820, -0.0104] * | -0.0436 [-0.1493, +0.0276] |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0047 [+0.0028, +0.0070] * | +0.0055 [+0.0028, +0.0089] * |
| TAC-LAT | TACTICAL | -0.0258 [-0.0396, -0.0133] * | -0.0246 [-0.0354, -0.0146] * |
| TAC-LON | TACTICAL | +0.0000 [-0.0229, +0.0221] | -0.0258 [-0.0467, -0.0067] * |
| S-POS@30 | STRATEGIC | +0.3808 [+0.2147, +0.5578] * | +0.3888 [+0.1997, +0.6099] * |
| S-POS@50 | STRATEGIC | +0.6718 [+0.3686, +0.9863] * | +0.6383 [+0.3155, +1.0275] * |
| S-POS@80 | STRATEGIC | +1.0328 [+0.5670, +1.5220] * | +1.0107 [+0.5053, +1.6353] * |
| M1 crawl share | crawl | +0.0220 [+0.0115, +0.0347] * | +0.0232 [+0.0132, +0.0356] * |

### 19 rows: beside: RILF075 - RIC0 (the heads-source arm's own contrast, the LF read's)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | +0.3683 [+0.2134, +0.5400] * | +0.2172 [+0.0971, +0.3406] * |
| speed MAE | LONGITUDINAL | +0.1250 [+0.0378, +0.2269] * | +0.0379 [-0.0199, +0.0938] |
| along-track MAE | LONGITUDINAL | +0.2753 [+0.1348, +0.4183] * | +0.1657 [+0.0462, +0.2827] * |
| accel MAE | LONGITUDINAL | -0.0357 [-0.1515, +0.0847] | -0.0379 [-0.1012, +0.0252] |
| DK-VIOL | LONGITUDINAL | +0.0269 [+0.0058, +0.0541] * | +0.0269 [+0.0084, +0.0513] * |
| DK-HW | LONGITUDINAL | +0.2961 [+0.0513, +0.6715] * | +0.1978 [+0.0052, +0.4163] * |
| DK-TG | LONGITUDINAL | +0.0779 [-0.0138, +0.2087] | +0.0351 [-0.0819, +0.1391] |
| heading MAE | LATERAL | -0.9023 [-1.5232, -0.3415] * | -0.9247 [-1.7145, -0.3716] * |
| yaw-rate MAE | LATERAL | -7.4724 [-11.1972, -4.6626] * | -7.5591 [-11.8067, -4.3665] * |
| curvature MAE | LATERAL | -0.0762 [-0.1084, -0.0485] * | -0.0800 [-0.1217, -0.0458] * |
| cross-track MAE | LATERAL | +0.1589 [+0.0843, +0.2533] * | +0.0868 [+0.0403, +0.1426] * |
| L1 lateral-accel MAE | LATERAL | -0.2673 [-0.7324, +0.1396] | -0.3303 [-0.6195, -0.1128] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0380 [-0.0654, -0.0184] * | -0.0403 [-0.0695, -0.0152] * |
| TAC-LAT | TACTICAL | +0.0283 [+0.0158, +0.0421] * | +0.0313 [+0.0184, +0.0446] * |
| TAC-LON | TACTICAL | +0.0013 [-0.0204, +0.0229] | +0.0275 [+0.0096, +0.0475] * |
| S-POS@30 | STRATEGIC | -0.3952 [-0.6138, -0.1979] * | -0.5271 [-0.7941, -0.2974] * |
| S-POS@50 | STRATEGIC | -0.7201 [-1.1097, -0.3769] * | -0.8314 [-1.2779, -0.4553] * |
| S-POS@80 | STRATEGIC | -1.2356 [-1.8567, -0.6888] * | -1.2768 [-2.0227, -0.6579] * |
| M1 crawl share | crawl | -0.0211 [-0.0334, -0.0112] * | -0.0234 [-0.0363, -0.0131] * |

## Controls

| control | PASS |
|---|---|
| reused_arrays | True |

## Variance named

* episodes: answered by the paired episode-cluster bootstrap (every interval here)
* training_run: O is C0's own checkpoint in every contrast with C0 / RIC0, so O's training variance is absent from it; the lead cap comes from one LLR run per seed (not measured); rule F_O reads against the C0REP floor
* inference: the planner, F and the oracle input are deterministic: no inference seed applies (the oracle input carries no draw)
