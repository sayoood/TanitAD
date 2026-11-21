**KNOWN-ANSWER RUN `ka-null` on EXISTING arrays (no LF rollout): a test of this reader, not a read of arms LF050 / LF075.**

# LF read: arms LF050 and LF075 (the LLR heads as a floor on O's along-track progress, planner C0), T1-kin, generated 2026-11-05T23:38:35+0100

**Headline: P fails at both doses and M1 is not moved: the floor does not bind where the crawl is; the instrument is checked first. Two doses were read (kappa 0.50 and kappa 0.75): a counted DOMINANT names its dose and says that two were read (A3 card section 4, multiplicity).**

* dose 050 (kappa 0.50): P-FAILS-M1-NOT-MOVED -> P fails and M1 is not moved: the floor does not bind where the crawl is; the instrument is checked first
* dose 075 (kappa 0.75): P-FAILS-M1-NOT-MOVED -> P fails and M1 is not moved: the floor does not bind where the crawl is; the instrument is checked first

Tier: T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2). Estimator: paired episode-cluster bootstrap B=2000 seed 0 (gapdecomp.decompose); answers another draw of EPISODES only. `*` = interval excludes 0. STRATEGIC: PAUSED (PI ITEM 25, 2026-09-19): reported beside the verdict, never scored as a pass of its own; D3 is evaluated over four families as Read A defines it and the three-family D3 is reported beside it

`hold_now` is not read (a frozen frame makes v_hat constant: no information about a frame-reading floor; A3 card section 3).


## Dose 050 (kappa 0.50): **P-FAILS-M1-NOT-MOVED**

Next: P fails and M1 is not moved: the floor does not bind where the crawl is; the instrument is checked first.


| rule | s0 | s1 | holds |
|---|---|---|---|
| P: M1 separated-better and share >= 0.5 | sep False, share 0.000 | sep False, share 0.000 | **False** |
| G: ADE6 <= +0.0604 m, no non-crawl row separated-worse | ADE6 +0.0000, worse: none | ADE6 +0.0000, worse: none | **True** |
| F: abs(RILF-RIC0)[M1] > abs(C0REP-C0)[M1] | 0.0000 vs 0.0362 | 0.0000 vs 0.0038 | **False** |
| LEAK-BORNE: logged share - reindexed share > half the logged share | logged 0.000, reindexed 0.000 | logged 0.000, reindexed 0.000 | **False** |
| D: RILFC - P10 read A (D1, D2, D3 four families) + P + G | D1 True, D2 False, D3 True | D1 True, D2 False, D3 True | **False** |

Beside: P on the logged scene False; D three-family form False; M1 moved (RILF - RIC0 separated-better at >= 1 seed) False.

D failing terms: s0: D2 (separated-worse: M1 crawl share); s1: D2 (separated-worse: M1 crawl share)

Harm rows: none


### 19 rows, dose 050: RILF - RIC0 (reindexed, the deciding contrast)

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

### 19 rows, dose 050: RILFC - P10 (reindexed composite, D)

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

### 19 rows, dose 050: LF - C0 (logged, LEAK-BORNE only)

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

## Dose 075 (kappa 0.75): **P-FAILS-M1-NOT-MOVED**

Next: P fails and M1 is not moved: the floor does not bind where the crawl is; the instrument is checked first.


| rule | s0 | s1 | holds |
|---|---|---|---|
| P: M1 separated-better and share >= 0.5 | sep False, share 0.000 | sep False, share 0.000 | **False** |
| G: ADE6 <= +0.0604 m, no non-crawl row separated-worse | ADE6 +0.0000, worse: none | ADE6 +0.0000, worse: none | **True** |
| F: abs(RILF-RIC0)[M1] > abs(C0REP-C0)[M1] | 0.0000 vs 0.0362 | 0.0000 vs 0.0038 | **False** |
| LEAK-BORNE: logged share - reindexed share > half the logged share | logged 0.000, reindexed 0.000 | logged 0.000, reindexed 0.000 | **False** |
| D: RILFC - P10 read A (D1, D2, D3 four families) + P + G | D1 True, D2 False, D3 True | D1 True, D2 False, D3 True | **False** |

Beside: P on the logged scene False; D three-family form False; M1 moved (RILF - RIC0 separated-better at >= 1 seed) False.

D failing terms: s0: D2 (separated-worse: M1 crawl share); s1: D2 (separated-worse: M1 crawl share)

Harm rows: none


### 19 rows, dose 075: RILF - RIC0 (reindexed, the deciding contrast)

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

### 19 rows, dose 075: RILFC - P10 (reindexed composite, D)

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

### 19 rows, dose 075: LF - C0 (logged, LEAK-BORNE only)

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

## Controls

| control | PASS |
|---|---|
| reused_arrays | True |

## Variance named

* episodes: answered by the paired episode-cluster bootstrap (every interval here)
* training_run: O is C0's own checkpoint in both arms of every LF - C0 contrast, so O's training variance is absent from it; the heads come from one LLR run per seed (not measured); rule F reads against the C0REP floor
* inference: the planner and F are deterministic: no inference seed applies
