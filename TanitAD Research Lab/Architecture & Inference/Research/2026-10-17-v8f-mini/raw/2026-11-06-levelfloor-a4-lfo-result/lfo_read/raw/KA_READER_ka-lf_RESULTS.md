**KNOWN-ANSWER RUN `ka-lf` on EXISTING arrays (no LFO rollout): a test of this reader, not a read of arm LFO075.**

# LFO read: arm LFO075 (LF075 with the level estimate replaced by the LOGGED speed), CEILING, T1-kin, generated 2026-11-06T04:13:42+0100

**Outcome: CEILING-SHORT (CEILING diagnostic).**

**Headline: CEILING-SHORT (CEILING diagnostic): P_O fails (share closed s0 0.395, s1 0.392; bar 0.5), G_O fails (ADE6 s0 +0.3683 m, s1 +0.2172 m; guard +0.0604 m). CEILING: the level input is the human's logged route speed, a privileged channel that is NOT admissible at inference. This is not a driving result, not a capability claim and not a counted DOMINANT; it ranks levers (A4 card).**

Tier: T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2); CEILING: privileged level input. Estimator: paired episode-cluster bootstrap B=2000 seed 0 (gapdecomp.decompose); answers another draw of EPISODES only. `*` = interval excludes 0. STRATEGIC: PAUSED (PI ITEM 25, 2026-09-19): reported beside the verdict, never scored as a pass of its own; D3 is evaluated over four families as Read A defines it and the three-family D3 is reported beside it

`logged` is not read (A4 section 3: the level input is the human's speed by construction, so the LEAK-BORNE rule has no question left; disclosed). `hold_now` is not read.

Next: an exact level at kappa 0.75 does not close half the crawl, so a floor on progress is not the form that closes M1 at this dose; the estimator levers are NOT signalled for this form; what remains is reported to the PI: T2 re-perception (option (c)), or the operator at kappa 1 in its own addendum.
Notes: M1 moved (RILFO - RIC0 separated-better at >= 1 seed): True.

| rule | s0 | s1 | holds |
|---|---|---|---|
| P_O: M1 separated-better and share >= 0.5 | sep True, share 0.395 | sep True, share 0.392 | **False** |
| G_O: ADE6 <= +0.0604 m, no non-crawl row separated-worse | ADE6 +0.3683, worse: ADE6, DK-HW, DK-VIOL, along-track MAE, cross-track MAE, speed MAE | ADE6 +0.2172, worse: ADE6, DK-HW, DK-VIOL, along-track MAE, cross-track MAE | **False** |
| F_O (beside the label): abs(RILFO-RIC0)[M1] > abs(C0REP-C0)[M1] | 0.0211 vs 0.0362 | 0.0234 vs 0.0038 | **False** |
| CEILING of D (never counted): RILFOC - P10 read A (D1, D2, D3 four families) | D1 True, D2 False, D3 True | D1 True, D2 False, D3 True | read A both seeds False (counted: False) |

Beside: three-family form of the ceiling of D False; M1 moved (RILFO - RIC0 separated-better at >= 1 seed) True.

Ceiling-of-D failing terms: s0: D2 (separated-worse: DK-VIOL, M1 crawl share); s1: D2 (separated-worse: M1 crawl share)

Failing rows (G_O, whatever the label): ADE6, DK-HW, DK-VIOL, along-track MAE, cross-track MAE, speed MAE


### 19 rows: RILFO - RIC0 (reindexed, the deciding contrast)

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

### 19 rows: ATTRIBUTION: RILFO - RILF075 (the level estimate's error at a fixed operator, dose, planner and lead cap; reported, never deciding)

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

### 19 rows: RILFOC - P10 (reindexed composite, the CEILING of D, never counted)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -0.7526 [-0.9765, -0.5462] * | -0.8043 [-1.0031, -0.6152] * |
| speed MAE | LONGITUDINAL | -0.7013 [-0.9073, -0.5211] * | -0.7503 [-0.9614, -0.5681] * |
| along-track MAE | LONGITUDINAL | -0.3066 [-0.4944, -0.1278] * | -0.3404 [-0.5153, -0.1736] * |
| accel MAE | LONGITUDINAL | -1.0515 [-1.4899, -0.6825] * | -1.1017 [-1.5373, -0.7323] * |
| DK-VIOL | LONGITUDINAL | +0.0387 [+0.0064, +0.0768] * | +0.0185 [-0.0068, +0.0452] |
| DK-HW | LONGITUDINAL | -0.0652 [-0.5703, +0.4591] | -0.2033 [-0.5906, +0.1355] |
| DK-TG | LONGITUDINAL | +0.0128 [-0.1325, +0.2118] | -0.0154 [-0.1145, +0.0964] |
| heading MAE | LATERAL | -2.3322 [-3.3822, -1.1386] * | -1.9624 [-3.0219, -0.8111] * |
| yaw-rate MAE | LATERAL | -13.3281 [-18.7954, -8.6686] * | -13.1525 [-18.4533, -8.6073] * |
| curvature MAE | LATERAL | -0.0464 [-0.0646, -0.0306] * | -0.0441 [-0.0617, -0.0286] * |
| cross-track MAE | LATERAL | -0.7971 [-0.9497, -0.6572] * | -0.8188 [-0.9576, -0.6892] * |
| L1 lateral-accel MAE | LATERAL | -3.4544 [-4.9967, -2.1195] * | -3.5205 [-5.0108, -2.2177] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0045 [-0.0072, -0.0019] * | -0.0036 [-0.0064, -0.0009] * |
| TAC-LAT | TACTICAL | +0.2134 [+0.1751, +0.2519] * | +0.1701 [+0.1284, +0.2147] * |
| TAC-LON | TACTICAL | +0.1538 [+0.0939, +0.2107] * | +0.1726 [+0.1159, +0.2280] * |
| S-POS@30 | STRATEGIC | -0.8442 [-1.1577, -0.5656] * | -0.9916 [-1.2938, -0.7119] * |
| S-POS@50 | STRATEGIC | -3.1794 [-4.0872, -2.3779] * | -3.4455 [-4.3221, -2.6290] * |
| S-POS@80 | STRATEGIC | -7.9963 [-9.8842, -6.1361] * | -9.0507 [-11.0584, -7.1729] * |
| M1 crawl share | crawl | +0.0380 [+0.0206, +0.0575] * | +0.0401 [+0.0211, +0.0619] * |

### 19 rows: attribution on the composite: RILFOC - RILF075C (beside)

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
