**KNOWN-ANSWER RUN `ka-ri` on EXISTING arrays (no CLR rollout): a test of this reader, not a read of arm CLR.**

# CLR read: arm CLR (level-from-image head inside RAPR-P01's recovery recipe), T1-kin, generated 2026-11-05T16:56:12+0100

**Outcome: P-FAILS-ATTR-NOT-SEPARATED-BETTER**

Next (A2 card, Outcomes): the level adds nothing even when needed (CLR - RAPRP01 on M1 not separated-better): LLR (the lead-aware head) is the next arm, wiring per the O-sensitivity of CLR (readout/ and WIRING.md).


Tier: T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2). Estimator: paired episode-cluster bootstrap B=2000 seed 0 (gapdecomp.decompose); answers another draw of EPISODES only. `*` = interval excludes 0. STRATEGIC: PAUSED (PI ITEM 25, 2026-09-19): reported beside the verdict, never scored as a pass of its own; D3 is evaluated over four families as Read A defines it and the three-family D3 is reported beside it

## Rules (CL card section 4 by reference, CL := CLR; attribution and outcomes of the A2 card; deciding scene = reindexed)

| rule | s0 | s1 | holds |
|---|---|---|---|
| P: M1 separated-better and share >= 0.5 | sep False, share 0.000 | sep False, share 0.000 | **False** |
| G: ADE6 <= +0.0604 m, no non-crawl row separated-worse | ADE6 +0.0000, worse: none | ADE6 +0.0000, worse: none | **True** |
| F: abs(RICLR-RIC0)[M1] > abs(C0REP-C0)[M1] | 0.0000 vs 0.0362 | 0.0000 vs 0.0038 | **False** |
| LEAK-BORNE: logged share - reindexed share > half the logged share | logged -0.002, reindexed 0.000 | logged -0.003, reindexed 0.000 | **False** |
| D: RICLRC - P10 read A (D1, D2, D3 four families) + P + G | D1 True, D2 False, D3 True | D1 True, D2 False, D3 True | **False** |
| ATTRIBUTION: M1 of RICLR - RIRAPRP01 (reindexed) separated-better at >= 1 seed | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | **False** |
| beside: M1 of CLR - RAPRP01 (logged), RAPRP01 = the banked rollout | +0.0001 [-0.0005, +0.0007] | +0.0002 [-0.0005, +0.0008] | 0 of 2 seeds separated-better |

Beside: P mean-over-seeds form False (mean share 0.000); P on logged scene False; D three-family form False; F logged form False; hold_now (FZCLR vs FZC0) reading PARTIAL, shares s0 0.000, s1 0.000 (never deciding).

## D failing terms

* s0: D2 (separated-worse: M1 crawl share)
* s1: D2 (separated-worse: M1 crawl share)

## 19 rows: RICLR - RIC0 (reindexed, the deciding contrast)

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

## 19 rows: CLR - C0 (logged)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | +0.0018 [+0.0005, +0.0032] * | +0.0012 [-0.0001, +0.0027] |
| speed MAE | LONGITUDINAL | +0.0015 [-0.0005, +0.0040] | -0.0000 [-0.0013, +0.0014] |
| along-track MAE | LONGITUDINAL | +0.0031 [+0.0014, +0.0049] * | +0.0014 [+0.0002, +0.0029] * |
| accel MAE | LONGITUDINAL | +0.0040 [-0.0015, +0.0110] | -0.0008 [-0.0038, +0.0017] |
| DK-VIOL | LONGITUDINAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| DK-HW | LONGITUDINAL | +0.0004 [-0.0094, +0.0119] | -0.0014 [-0.0056, +0.0021] |
| DK-TG | LONGITUDINAL | -0.0036 [-0.0107, -0.0000] * | +0.0003 [-0.0012, +0.0018] |
| heading MAE | LATERAL | +0.0298 [+0.0052, +0.0575] * | +0.0088 [-0.0213, +0.0405] |
| yaw-rate MAE | LATERAL | -0.0347 [-0.2754, +0.1586] | -0.0248 [-0.1844, +0.1172] |
| curvature MAE | LATERAL | +0.0016 [-0.0003, +0.0040] | +0.0003 [-0.0014, +0.0022] |
| cross-track MAE | LATERAL | -0.0006 [-0.0015, +0.0003] | -0.0001 [-0.0008, +0.0006] |
| L1 lateral-accel MAE | LATERAL | +0.0098 [-0.0214, +0.0445] | -0.0044 [-0.0181, +0.0073] |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0013 [-0.0004, +0.0032] | +0.0043 [-0.0040, +0.0174] |
| TAC-LAT | TACTICAL | -0.0029 [-0.0058, +0.0000] | +0.0004 [-0.0033, +0.0042] |
| TAC-LON | TACTICAL | +0.0033 [+0.0008, +0.0063] * | -0.0008 [-0.0042, +0.0025] |
| S-POS@30 | STRATEGIC | -0.0069 [-0.0351, +0.0219] | +0.0043 [-0.0297, +0.0383] |
| S-POS@50 | STRATEGIC | +0.0065 [-0.0449, +0.0599] | +0.0262 [-0.0386, +0.0936] |
| S-POS@80 | STRATEGIC | +0.0500 [-0.0459, +0.1535] | +0.0796 [-0.0396, +0.2065] |
| M1 crawl share | crawl | +0.0001 [-0.0005, +0.0007] | +0.0002 [-0.0005, +0.0008] |

## 19 rows: RICLRC - P10 (reindexed composite, D)

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

## 19 rows: RICLR - RIRAPRP01 (reindexed, ATTRIBUTION: CLR against RAPR-P01's own checkpoint)

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

## 19 rows: CLR - RAPRP01 (logged, attribution beside; RAPRP01 = the banked rollout)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | +0.0018 [+0.0005, +0.0032] * | +0.0012 [-0.0001, +0.0027] |
| speed MAE | LONGITUDINAL | +0.0015 [-0.0005, +0.0040] | -0.0000 [-0.0013, +0.0014] |
| along-track MAE | LONGITUDINAL | +0.0031 [+0.0014, +0.0049] * | +0.0014 [+0.0002, +0.0029] * |
| accel MAE | LONGITUDINAL | +0.0040 [-0.0015, +0.0110] | -0.0008 [-0.0038, +0.0017] |
| DK-VIOL | LONGITUDINAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| DK-HW | LONGITUDINAL | +0.0004 [-0.0094, +0.0119] | -0.0014 [-0.0056, +0.0021] |
| DK-TG | LONGITUDINAL | -0.0036 [-0.0107, -0.0000] * | +0.0003 [-0.0012, +0.0018] |
| heading MAE | LATERAL | +0.0298 [+0.0052, +0.0575] * | +0.0088 [-0.0213, +0.0405] |
| yaw-rate MAE | LATERAL | -0.0347 [-0.2754, +0.1586] | -0.0248 [-0.1844, +0.1172] |
| curvature MAE | LATERAL | +0.0016 [-0.0003, +0.0040] | +0.0003 [-0.0014, +0.0022] |
| cross-track MAE | LATERAL | -0.0006 [-0.0015, +0.0003] | -0.0001 [-0.0008, +0.0006] |
| L1 lateral-accel MAE | LATERAL | +0.0098 [-0.0214, +0.0445] | -0.0044 [-0.0181, +0.0073] |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0013 [-0.0004, +0.0032] | +0.0043 [-0.0040, +0.0174] |
| TAC-LAT | TACTICAL | -0.0029 [-0.0058, +0.0000] | +0.0004 [-0.0033, +0.0042] |
| TAC-LON | TACTICAL | +0.0033 [+0.0008, +0.0063] * | -0.0008 [-0.0042, +0.0025] |
| S-POS@30 | STRATEGIC | -0.0069 [-0.0351, +0.0219] | +0.0043 [-0.0297, +0.0383] |
| S-POS@50 | STRATEGIC | +0.0065 [-0.0449, +0.0599] | +0.0262 [-0.0386, +0.0936] |
| S-POS@80 | STRATEGIC | +0.0500 [-0.0459, +0.1535] | +0.0796 [-0.0396, +0.2065] |
| M1 crawl share | crawl | +0.0001 [-0.0005, +0.0007] | +0.0002 [-0.0005, +0.0008] |

## Families (separated-better / separated-worse row counts)

| contrast | family | s0 better / worse | s1 better / worse |
|---|---|---|---|
| RICLR_minus_RIC0 | LONGITUDINAL | 0 / 0 (of 6) | 0 / 0 (of 6) |
| RICLR_minus_RIC0 | LATERAL | 0 / 0 (of 6) | 0 / 0 (of 6) |
| RICLR_minus_RIC0 | TACTICAL | 0 / 0 (of 2) | 0 / 0 (of 2) |
| RICLR_minus_RIC0 | STRATEGIC | 0 / 0 (of 3) | 0 / 0 (of 3) |
| CLR_minus_C0 | LONGITUDINAL | 1 / 1 (of 6) | 0 / 1 (of 6) |
| CLR_minus_C0 | LATERAL | 0 / 1 (of 6) | 0 / 0 (of 6) |
| CLR_minus_C0 | TACTICAL | 1 / 0 (of 2) | 0 / 0 (of 2) |
| CLR_minus_C0 | STRATEGIC | 0 / 0 (of 3) | 0 / 0 (of 3) |
| RICLRC_minus_P10 | LONGITUDINAL | 5 / 0 (of 6) | 5 / 0 (of 6) |
| RICLRC_minus_P10 | LATERAL | 5 / 0 (of 6) | 5 / 0 (of 6) |
| RICLRC_minus_P10 | TACTICAL | 2 / 0 (of 2) | 2 / 0 (of 2) |
| RICLRC_minus_P10 | STRATEGIC | 3 / 0 (of 3) | 3 / 0 (of 3) |
| CLRC_minus_P10 | LONGITUDINAL | 5 / 0 (of 6) | 5 / 0 (of 6) |
| CLRC_minus_P10 | LATERAL | 5 / 0 (of 6) | 5 / 0 (of 6) |
| CLRC_minus_P10 | TACTICAL | 2 / 0 (of 2) | 2 / 0 (of 2) |
| CLRC_minus_P10 | STRATEGIC | 3 / 0 (of 3) | 3 / 0 (of 3) |
| FZCLR_minus_FZC0 | LONGITUDINAL | 0 / 0 (of 6) | 0 / 0 (of 6) |
| FZCLR_minus_FZC0 | LATERAL | 0 / 0 (of 6) | 0 / 0 (of 6) |
| FZCLR_minus_FZC0 | TACTICAL | 0 / 0 (of 2) | 0 / 0 (of 2) |
| FZCLR_minus_FZC0 | STRATEGIC | 0 / 0 (of 3) | 0 / 0 (of 3) |
| RICLR_minus_RIRAPRP01 | LONGITUDINAL | 0 / 0 (of 6) | 0 / 0 (of 6) |
| RICLR_minus_RIRAPRP01 | LATERAL | 0 / 0 (of 6) | 0 / 0 (of 6) |
| RICLR_minus_RIRAPRP01 | TACTICAL | 0 / 0 (of 2) | 0 / 0 (of 2) |
| RICLR_minus_RIRAPRP01 | STRATEGIC | 0 / 0 (of 3) | 0 / 0 (of 3) |
| CLR_minus_RAPRP01 | LONGITUDINAL | 1 / 1 (of 6) | 0 / 1 (of 6) |
| CLR_minus_RAPRP01 | LATERAL | 0 / 1 (of 6) | 0 / 0 (of 6) |
| CLR_minus_RAPRP01 | TACTICAL | 1 / 0 (of 2) | 0 / 0 (of 2) |
| CLR_minus_RAPRP01 | STRATEGIC | 0 / 0 (of 3) | 0 / 0 (of 3) |

## Controls

| control | PASS |
|---|---|
| reused_arrays | True |

## Variance named

* episodes: answered by the paired episode-cluster bootstrap (every interval here)
* training_run: answered ONLY by the C0REP floor (rule F); a separated CI is necessary, not sufficient (H-ESTIM-SEED-1)
* inference: the planner is deterministic at eval (no sampling): no inference seed applies
