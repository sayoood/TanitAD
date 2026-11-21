# CL read: arm CL (level-from-image head), T1-kin, generated 2026-11-05T15:35:02+0100

**Outcome: P-FAILS-M1-NOT-MOVED**

Next (card section 5): the head's level does not reach O's decision: next is the head read-out check (does v_hat track the logged speed on eval at T0? R2 and the M1-window R2) and, if it does, O's use of it (attribution read with e_level zeroed at inference); if it does not, a lead-aware head (lead presence and distance from the frame, CRAWL-LEAD: half the crawl sits behind a lead).
Notes: G fails too (the guards read worse alongside the failed P).

Tier: T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2). Estimator: paired episode-cluster bootstrap B=2000 seed 0 (gapdecomp.decompose); answers another draw of EPISODES only. `*` = interval excludes 0. STRATEGIC: PAUSED (PI ITEM 25, 2026-09-19): reported beside the verdict, never scored as a pass of its own; D3 is evaluated over four families as Read A defines it and the three-family D3 is reported beside it

## Rules (card section 4; deciding scene = reindexed)

| rule | s0 | s1 | holds |
|---|---|---|---|
| P: M1 separated-better and share >= 0.5 | sep False, share 0.030 | sep False, share 0.006 | **False** |
| G: ADE6 <= +0.0604 m, no non-crawl row separated-worse | ADE6 +0.0007, worse: L1 lateral-accel MAE, heading MAE | ADE6 -0.0206, worse: none | **False** |
| F: abs(RICL-RIC0)[M1] > abs(C0REP-C0)[M1] | 0.0016 vs 0.0362 | 0.0003 vs 0.0038 | **False** |
| LEAK-BORNE: logged share - reindexed share > half the logged share | logged 0.038, reindexed 0.030 | logged 0.006, reindexed 0.006 | **False** |
| D: RICLC - P10 read A (D1, D2, D3 four families) + P + G | D1 True, D2 False, D3 True | D1 True, D2 False, D3 True | **False** |

Beside: P mean-over-seeds form False (mean share 0.018); P on logged scene False; D three-family form False; F logged form False; hold_now (FZCL vs FZC0) reading PARTIAL, shares s0 0.043, s1 -0.025 (never deciding).

## D failing terms

* s0: D2 (separated-worse: M1 crawl share)
* s1: D2 (separated-worse: M1 crawl share)

## 19 rows: RICL - RIC0 (reindexed, the deciding contrast)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | +0.0007 [-0.0195, +0.0206] | -0.0206 [-0.0434, +0.0014] |
| speed MAE | LONGITUDINAL | +0.0069 [-0.0176, +0.0339] | -0.0227 [-0.0469, +0.0009] |
| along-track MAE | LONGITUDINAL | -0.0074 [-0.0279, +0.0120] | -0.0199 [-0.0394, -0.0018] * |
| accel MAE | LONGITUDINAL | +0.0189 [-0.0217, +0.0600] | -0.0264 [-0.0861, +0.0274] |
| DK-VIOL | LONGITUDINAL | -0.0017 [-0.0101, +0.0077] | +0.0034 [-0.0036, +0.0125] |
| DK-HW | LONGITUDINAL | +0.0129 [-0.0657, +0.0932] | +0.0400 [-0.0119, +0.0922] |
| DK-TG | LONGITUDINAL | +0.0001 [-0.0244, +0.0275] | +0.0057 [-0.0318, +0.0393] |
| heading MAE | LATERAL | +0.2257 [+0.0662, +0.4048] * | -0.3794 [-0.7439, -0.1061] * |
| yaw-rate MAE | LATERAL | +0.7160 [-0.0079, +1.6327] | -1.3155 [-3.0641, -0.0105] * |
| curvature MAE | LATERAL | -0.0005 [-0.0074, +0.0065] | -0.0008 [-0.0082, +0.0067] |
| cross-track MAE | LATERAL | +0.0039 [-0.0122, +0.0193] | -0.0039 [-0.0214, +0.0137] |
| L1 lateral-accel MAE | LATERAL | +0.2199 [+0.0644, +0.4111] * | -0.2074 [-0.4649, -0.0038] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0029 [-0.0069, +0.0011] | -0.0043 [-0.0181, +0.0071] |
| TAC-LAT | TACTICAL | +0.0013 [-0.0046, +0.0071] | +0.0063 [-0.0017, +0.0142] |
| TAC-LON | TACTICAL | -0.0021 [-0.0146, +0.0113] | +0.0004 [-0.0138, +0.0154] |
| S-POS@30 | STRATEGIC | +0.0426 [-0.0684, +0.1553] | -0.1996 [-0.3179, -0.0948] * |
| S-POS@50 | STRATEGIC | +0.0092 [-0.1767, +0.1952] | -0.2534 [-0.4470, -0.0716] * |
| S-POS@80 | STRATEGIC | -0.1344 [-0.4623, +0.1867] | -0.2847 [-0.6000, +0.0061] |
| M1 crawl share | crawl | -0.0016 [-0.0042, +0.0011] | -0.0003 [-0.0026, +0.0020] |

## 19 rows: CL - C0 (logged)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | +0.0006 [-0.0197, +0.0206] | -0.0206 [-0.0429, +0.0015] |
| speed MAE | LONGITUDINAL | +0.0058 [-0.0188, +0.0324] | -0.0234 [-0.0480, +0.0000] |
| along-track MAE | LONGITUDINAL | -0.0072 [-0.0275, +0.0120] | -0.0198 [-0.0394, -0.0013] * |
| accel MAE | LONGITUDINAL | +0.0191 [-0.0201, +0.0599] | -0.0280 [-0.0897, +0.0260] |
| DK-VIOL | LONGITUDINAL | -0.0017 [-0.0101, +0.0077] | +0.0034 [-0.0036, +0.0125] |
| DK-HW | LONGITUDINAL | +0.0130 [-0.0708, +0.0978] | +0.0427 [-0.0124, +0.1002] |
| DK-TG | LONGITUDINAL | -0.0066 [-0.0281, +0.0136] | +0.0088 [-0.0296, +0.0460] |
| heading MAE | LATERAL | +0.2100 [+0.0586, +0.3804] * | -0.3857 [-0.7290, -0.1229] * |
| yaw-rate MAE | LATERAL | +0.6242 [-0.0464, +1.4427] | -1.3199 [-3.0274, -0.0165] * |
| curvature MAE | LATERAL | +0.0008 [-0.0073, +0.0088] | +0.0008 [-0.0074, +0.0092] |
| cross-track MAE | LATERAL | +0.0035 [-0.0126, +0.0190] | -0.0038 [-0.0213, +0.0140] |
| L1 lateral-accel MAE | LATERAL | +0.2054 [+0.0596, +0.3841] * | -0.2178 [-0.4837, -0.0014] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0031 [-0.0078, +0.0014] | -0.0020 [-0.0124, +0.0072] |
| TAC-LAT | TACTICAL | -0.0033 [-0.0113, +0.0046] | +0.0071 [-0.0004, +0.0146] |
| TAC-LON | TACTICAL | +0.0017 [-0.0100, +0.0146] | +0.0000 [-0.0150, +0.0162] |
| S-POS@30 | STRATEGIC | +0.0158 [-0.0868, +0.1231] | -0.2062 [-0.3235, -0.1039] * |
| S-POS@50 | STRATEGIC | -0.0450 [-0.2178, +0.1224] | -0.2466 [-0.4312, -0.0707] * |
| S-POS@80 | STRATEGIC | -0.2218 [-0.5017, +0.0611] | -0.2745 [-0.5768, +0.0084] |
| M1 crawl share | crawl | -0.0020 [-0.0045, +0.0003] | -0.0004 [-0.0025, +0.0020] |

## 19 rows: RICLC - P10 (reindexed composite, D)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -1.1331 [-1.3272, -0.9541] * | -1.0342 [-1.2297, -0.8371] * |
| speed MAE | LONGITUDINAL | -0.8569 [-1.0883, -0.6621] * | -0.8241 [-1.0468, -0.6233] * |
| along-track MAE | LONGITUDINAL | -0.6478 [-0.8174, -0.4887] * | -0.5401 [-0.7155, -0.3704] * |
| accel MAE | LONGITUDINAL | -1.1336 [-1.5808, -0.7575] * | -1.1802 [-1.6246, -0.7969] * |
| DK-VIOL | LONGITUDINAL | -0.0017 [-0.0265, +0.0216] | -0.0134 [-0.0381, +0.0100] |
| DK-HW | LONGITUDINAL | -0.5187 [-0.9121, -0.1870] * | -0.5059 [-0.8702, -0.2360] * |
| DK-TG | LONGITUDINAL | -0.1193 [-0.1889, -0.0646] * | -0.1011 [-0.1567, -0.0581] * |
| heading MAE | LATERAL | -2.3260 [-3.4334, -1.3107] * | -1.2529 [-2.2251, -0.1964] * |
| yaw-rate MAE | LATERAL | -12.7565 [-18.1572, -8.2508] * | -12.5704 [-17.9514, -8.1358] * |
| curvature MAE | LATERAL | -0.0333 [-0.0496, -0.0194] * | -0.0328 [-0.0491, -0.0189] * |
| cross-track MAE | LATERAL | -0.9008 [-1.0420, -0.7734] * | -0.8803 [-1.0206, -0.7485] * |
| L1 lateral-accel MAE | LATERAL | -3.5726 [-5.1614, -2.2182] * | -3.5774 [-5.1267, -2.2568] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0000 [-0.0031, +0.0032] | +0.0004 [-0.0029, +0.0041] |
| TAC-LAT | TACTICAL | +0.2001 [+0.1604, +0.2401] * | +0.1526 [+0.1110, +0.1966] * |
| TAC-LON | TACTICAL | +0.1534 [+0.0875, +0.2181] * | +0.1442 [+0.0834, +0.2009] * |
| S-POS@30 | STRATEGIC | -0.6837 [-0.9584, -0.4353] * | -0.9003 [-1.1934, -0.6212] * |
| S-POS@50 | STRATEGIC | -2.9183 [-3.7367, -2.1499] * | -3.3450 [-4.1809, -2.5746] * |
| S-POS@80 | STRATEGIC | -8.0602 [-9.8518, -6.2865] * | -8.7830 [-10.6780, -7.0064] * |
| M1 crawl share | crawl | +0.0583 [+0.0393, +0.0809] * | +0.0586 [+0.0373, +0.0846] * |

## Families (separated-better / separated-worse row counts)

| contrast | family | s0 better / worse | s1 better / worse |
|---|---|---|---|
| RICL_minus_RIC0 | LONGITUDINAL | 0 / 0 (of 6) | 1 / 0 (of 6) |
| RICL_minus_RIC0 | LATERAL | 0 / 2 (of 6) | 3 / 0 (of 6) |
| RICL_minus_RIC0 | TACTICAL | 0 / 0 (of 2) | 0 / 0 (of 2) |
| RICL_minus_RIC0 | STRATEGIC | 0 / 0 (of 3) | 2 / 0 (of 3) |
| CL_minus_C0 | LONGITUDINAL | 0 / 0 (of 6) | 1 / 0 (of 6) |
| CL_minus_C0 | LATERAL | 0 / 2 (of 6) | 3 / 0 (of 6) |
| CL_minus_C0 | TACTICAL | 0 / 0 (of 2) | 0 / 0 (of 2) |
| CL_minus_C0 | STRATEGIC | 0 / 0 (of 3) | 2 / 0 (of 3) |
| RICLC_minus_P10 | LONGITUDINAL | 5 / 0 (of 6) | 5 / 0 (of 6) |
| RICLC_minus_P10 | LATERAL | 5 / 0 (of 6) | 5 / 0 (of 6) |
| RICLC_minus_P10 | TACTICAL | 2 / 0 (of 2) | 2 / 0 (of 2) |
| RICLC_minus_P10 | STRATEGIC | 3 / 0 (of 3) | 3 / 0 (of 3) |
| CLC_minus_P10 | LONGITUDINAL | 5 / 0 (of 6) | 5 / 0 (of 6) |
| CLC_minus_P10 | LATERAL | 5 / 0 (of 6) | 5 / 0 (of 6) |
| CLC_minus_P10 | TACTICAL | 2 / 0 (of 2) | 2 / 0 (of 2) |
| CLC_minus_P10 | STRATEGIC | 3 / 0 (of 3) | 3 / 0 (of 3) |
| FZCL_minus_FZC0 | LONGITUDINAL | 0 / 0 (of 6) | 1 / 0 (of 6) |
| FZCL_minus_FZC0 | LATERAL | 0 / 0 (of 6) | 1 / 0 (of 6) |
| FZCL_minus_FZC0 | TACTICAL | 0 / 0 (of 2) | 0 / 0 (of 2) |
| FZCL_minus_FZC0 | STRATEGIC | 0 / 0 (of 3) | 3 / 0 (of 3) |

## Controls

| control | PASS |
|---|---|
| rollout_CL_s0 | True |
| rollout_CLC_s0 | True |
| rollout_FZCL_s0 | True |
| rollout_RICL_s0 | True |
| rollout_RICLC_s0 | True |
| rollout_CL_s1 | True |
| rollout_CLC_s1 | True |
| rollout_FZCL_s1 | True |
| rollout_RICL_s1 | True |
| rollout_RICLC_s1 | True |
| one_checkpoint_per_seed | True |
| KA_NEW | True |
| CL_PRECHECK | True |
| card_rules | True |
| reused_arrays | True |

## Variance named

* episodes: answered by the paired episode-cluster bootstrap (every interval here)
* training_run: answered ONLY by the C0REP floor (rule F); a separated CI is necessary, not sufficient (H-ESTIM-SEED-1)
* inference: the planner is deterministic at eval (no sampling): no inference seed applies
