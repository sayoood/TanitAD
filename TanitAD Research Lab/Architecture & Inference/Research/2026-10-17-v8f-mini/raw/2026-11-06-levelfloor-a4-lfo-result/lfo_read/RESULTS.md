# LFO read: arm LFO075 (LF075 with the level estimate replaced by the LOGGED speed), CEILING, T1-kin, generated 2026-11-06T04:33:09+0100

**Outcome: CEILING-SHORT (CEILING diagnostic).**

**Headline: CEILING-SHORT (CEILING diagnostic): P_O fails (share closed s0 0.126, s1 0.132; bar 0.5), G_O fails (ADE6 s0 -0.0233 m, s1 -0.0246 m; guard +0.0604 m). CEILING: the level input is the human's logged route speed, a privileged channel that is NOT admissible at inference. This is not a driving result, not a capability claim and not a counted DOMINANT; it ranks levers (A4 card).**

Tier: T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2); CEILING: privileged level input. Estimator: paired episode-cluster bootstrap B=2000 seed 0 (gapdecomp.decompose); answers another draw of EPISODES only. `*` = interval excludes 0. STRATEGIC: PAUSED (PI ITEM 25, 2026-09-19): reported beside the verdict, never scored as a pass of its own; D3 is evaluated over four families as Read A defines it and the three-family D3 is reported beside it

`logged` is not read (A4 section 3: the level input is the human's speed by construction, so the LEAK-BORNE rule has no question left; disclosed). `hold_now` is not read.

Next: an exact level at kappa 0.75 does not close half the crawl, so a floor on progress is not the form that closes M1 at this dose; the estimator levers are NOT signalled for this form; what remains is reported to the PI: T2 re-perception (option (c)), or the operator at kappa 1 in its own addendum.
Notes: M1 moved (RILFO - RIC0 separated-better at >= 1 seed): True.

Reach control (F binds on >= 1 % of the replans inside the M1 crawl ticks, each seed): s0 0.536 of 237 ticks; s1 0.621 of 240 ticks -> **True**

| rule | s0 | s1 | holds |
|---|---|---|---|
| P_O: M1 separated-better and share >= 0.5 | sep True, share 0.126 | sep True, share 0.132 | **False** |
| G_O: ADE6 <= +0.0604 m, no non-crawl row separated-worse | ADE6 -0.0233, worse: accel MAE, cross-track MAE | ADE6 -0.0246, worse: L1 lateral-accel MAE, accel MAE, cross-track MAE | **False** |
| F_O (beside the label): abs(RILFO-RIC0)[M1] > abs(C0REP-C0)[M1] | 0.0067 vs 0.0362 | 0.0079 vs 0.0038 | **False** |
| CEILING of D (never counted): RILFOC - P10 read A (D1, D2, D3 four families) | D1 True, D2 False, D3 True | D1 True, D2 False, D3 True | read A both seeds False (counted: False) |

Beside: three-family form of the ceiling of D False; M1 moved (RILFO - RIC0 separated-better at >= 1 seed) True.

Ceiling-of-D failing terms: s0: D2 (separated-worse: M1 crawl share); s1: D2 (separated-worse: M1 crawl share)

Failing rows (G_O, whatever the label): L1 lateral-accel MAE, accel MAE, cross-track MAE


### 19 rows: RILFO - RIC0 (reindexed, the deciding contrast)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -0.0233 [-0.0483, +0.0052] | -0.0246 [-0.0553, +0.0096] |
| speed MAE | LONGITUDINAL | +0.0049 [-0.0320, +0.0521] | -0.0075 [-0.0332, +0.0257] |
| along-track MAE | LONGITUDINAL | -0.0405 [-0.0665, -0.0115] * | -0.0449 [-0.0732, -0.0177] * |
| accel MAE | LONGITUDINAL | +0.0678 [+0.0013, +0.1529] * | +0.0866 [+0.0209, +0.1673] * |
| DK-VIOL | LONGITUDINAL | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| DK-HW | LONGITUDINAL | +0.0080 [-0.0048, +0.0240] | +0.0291 [-0.0006, +0.0929] |
| DK-TG | LONGITUDINAL | +0.0045 [-0.0009, +0.0137] | +0.0025 [-0.0001, +0.0076] |
| heading MAE | LATERAL | -0.0324 [-0.1338, +0.0776] | +0.0146 [-0.1222, +0.2344] |
| yaw-rate MAE | LATERAL | -1.6337 [-2.5295, -0.8804] * | -1.3848 [-2.0559, -0.7734] * |
| curvature MAE | LATERAL | -0.0236 [-0.0361, -0.0136] * | -0.0234 [-0.0358, -0.0133] * |
| cross-track MAE | LATERAL | +0.0278 [+0.0109, +0.0493] * | +0.0284 [+0.0058, +0.0584] * |
| L1 lateral-accel MAE | LATERAL | +0.1518 [-0.0125, +0.3516] | +0.1807 [+0.0106, +0.4016] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0291 [-0.0485, -0.0137] * | -0.0352 [-0.0601, -0.0141] * |
| TAC-LAT | TACTICAL | +0.0029 [-0.0021, +0.0092] | +0.0083 [+0.0042, +0.0138] * |
| TAC-LON | TACTICAL | +0.0108 [+0.0042, +0.0184] * | +0.0146 [+0.0063, +0.0233] * |
| S-POS@30 | STRATEGIC | -0.0826 [-0.1449, -0.0279] * | -0.0923 [-0.1907, +0.0006] |
| S-POS@50 | STRATEGIC | -0.1594 [-0.2753, -0.0553] * | -0.1213 [-0.2876, +0.0376] |
| S-POS@80 | STRATEGIC | -0.2772 [-0.4882, -0.0947] * | -0.1675 [-0.4324, +0.1010] |
| M1 crawl share | crawl | -0.0067 [-0.0106, -0.0032] * | -0.0079 [-0.0121, -0.0043] * |

### 19 rows: ATTRIBUTION: RILFO - RILF075 (the level estimate's error at a fixed operator, dose, planner and lead cap; reported, never deciding)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -0.3916 [-0.5562, -0.2435] * | -0.2418 [-0.3638, -0.1234] * |
| speed MAE | LONGITUDINAL | -0.1201 [-0.2144, -0.0315] * | -0.0454 [-0.1057, +0.0175] |
| along-track MAE | LONGITUDINAL | -0.3158 [-0.4491, -0.1770] * | -0.2105 [-0.3245, -0.0991] * |
| accel MAE | LONGITUDINAL | +0.1035 [-0.0338, +0.2426] | +0.1245 [+0.0375, +0.2253] * |
| DK-VIOL | LONGITUDINAL | -0.0269 [-0.0541, -0.0058] * | -0.0269 [-0.0513, -0.0084] * |
| DK-HW | LONGITUDINAL | -0.2881 [-0.6625, -0.0411] * | -0.1686 [-0.3785, +0.0139] |
| DK-TG | LONGITUDINAL | -0.0735 [-0.2060, +0.0189] | -0.0325 [-0.1369, +0.0833] |
| heading MAE | LATERAL | +0.8699 [+0.2958, +1.4997] * | +0.9394 [+0.3552, +1.7146] * |
| yaw-rate MAE | LATERAL | +5.8387 [+3.1881, +9.5440] * | +6.1743 [+3.1116, +10.1920] * |
| curvature MAE | LATERAL | +0.0526 [+0.0289, +0.0806] * | +0.0565 [+0.0261, +0.0939] * |
| cross-track MAE | LATERAL | -0.1310 [-0.2203, -0.0583] * | -0.0584 [-0.1196, -0.0051] * |
| L1 lateral-accel MAE | LATERAL | +0.4191 [+0.0036, +0.9067] * | +0.5110 [+0.2290, +0.8508] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0089 [+0.0014, +0.0210] * | +0.0051 [-0.0036, +0.0154] |
| TAC-LAT | TACTICAL | -0.0254 [-0.0400, -0.0125] * | -0.0229 [-0.0363, -0.0100] * |
| TAC-LON | TACTICAL | +0.0096 [-0.0096, +0.0292] | -0.0129 [-0.0304, +0.0042] |
| S-POS@30 | STRATEGIC | +0.3126 [+0.1240, +0.5194] * | +0.4349 [+0.2063, +0.6974] * |
| S-POS@50 | STRATEGIC | +0.5607 [+0.2418, +0.9208] * | +0.7101 [+0.3072, +1.1608] * |
| S-POS@80 | STRATEGIC | +0.9584 [+0.4411, +1.5316] * | +1.1093 [+0.4436, +1.8704] * |
| M1 crawl share | crawl | +0.0143 [+0.0041, +0.0260] * | +0.0155 [+0.0065, +0.0266] * |

### 19 rows: RILFOC - P10 (reindexed composite, the CEILING of D, never counted)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -1.1473 [-1.3364, -0.9682] * | -1.0608 [-1.2521, -0.8735] * |
| speed MAE | LONGITUDINAL | -0.8431 [-1.0719, -0.6529] * | -0.8409 [-1.0574, -0.6517] * |
| along-track MAE | LONGITUDINAL | -0.6675 [-0.8294, -0.5197] * | -0.5705 [-0.7428, -0.4118] * |
| accel MAE | LONGITUDINAL | -1.0415 [-1.4953, -0.6654] * | -1.0917 [-1.5333, -0.7165] * |
| DK-VIOL | LONGITUDINAL | -0.0017 [-0.0228, +0.0190] | -0.0151 [-0.0343, +0.0035] |
| DK-HW | LONGITUDINAL | -0.4686 [-0.8590, -0.1422] * | -0.4931 [-0.8473, -0.2286] * |
| DK-TG | LONGITUDINAL | -0.1121 [-0.1790, -0.0592] * | -0.1232 [-0.1977, -0.0649] * |
| heading MAE | LATERAL | -1.5710 [-2.6184, -0.4298] * | -1.0253 [-2.0103, +0.0450] |
| yaw-rate MAE | LATERAL | -12.5302 [-17.8608, -8.0362] * | -12.4530 [-17.7458, -8.0699] * |
| curvature MAE | LATERAL | -0.0373 [-0.0535, -0.0233] * | -0.0352 [-0.0513, -0.0212] * |
| cross-track MAE | LATERAL | -0.8818 [-1.0324, -0.7441] * | -0.8708 [-1.0123, -0.7409] * |
| L1 lateral-accel MAE | LATERAL | -3.3892 [-4.9894, -2.0319] * | -3.5154 [-5.0572, -2.1976] * |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0027 [-0.0053, -0.0003] * | -0.0024 [-0.0053, +0.0006] |
| TAC-LAT | TACTICAL | +0.1905 [+0.1516, +0.2301] * | +0.1517 [+0.1097, +0.1953] * |
| TAC-LON | TACTICAL | +0.1655 [+0.1055, +0.2245] * | +0.1622 [+0.1047, +0.2175] * |
| S-POS@30 | STRATEGIC | -0.9152 [-1.2188, -0.6387] * | -1.0299 [-1.3319, -0.7459] * |
| S-POS@50 | STRATEGIC | -3.2310 [-4.1237, -2.4506] * | -3.5501 [-4.4186, -2.7413] * |
| S-POS@80 | STRATEGIC | -8.0777 [-9.9117, -6.2848] * | -9.1323 [-11.0550, -7.2594] * |
| M1 crawl share | crawl | +0.0527 [+0.0346, +0.0738] * | +0.0563 [+0.0343, +0.0823] * |

### 19 rows: attribution on the composite: RILFOC - RILF075C (beside)

| row | family | s0 | s1 |
|---|---|---|---|
| ADE6 | PRIMARY | -0.3947 [-0.5569, -0.2421] * | -0.2565 [-0.3754, -0.1457] * |
| speed MAE | LONGITUDINAL | -0.1417 [-0.2246, -0.0638] * | -0.0905 [-0.1553, -0.0330] * |
| along-track MAE | LONGITUDINAL | -0.3609 [-0.5060, -0.2151] * | -0.2302 [-0.3413, -0.1216] * |
| accel MAE | LONGITUDINAL | +0.0100 [-0.0783, +0.1054] | +0.0100 [-0.0884, +0.0930] |
| DK-VIOL | LONGITUDINAL | -0.0403 [-0.0778, -0.0117] * | -0.0336 [-0.0619, -0.0106] * |
| DK-HW | LONGITUDINAL | -0.3966 [-0.8264, -0.0774] * | -0.2561 [-0.4913, -0.0611] * |
| DK-TG | LONGITUDINAL | -0.1199 [-0.3062, +0.0133] | -0.1048 [-0.2335, -0.0203] * |
| heading MAE | LATERAL | +0.7612 [+0.3447, +1.2416] * | +0.9371 [+0.2832, +1.8220] * |
| yaw-rate MAE | LATERAL | +0.7975 [+0.4391, +1.1861] * | +0.6994 [+0.3674, +1.0492] * |
| curvature MAE | LATERAL | +0.0090 [+0.0054, +0.0132] * | +0.0089 [+0.0050, +0.0132] * |
| cross-track MAE | LATERAL | -0.0847 [-0.1431, -0.0365] * | -0.0520 [-0.1017, -0.0123] * |
| L1 lateral-accel MAE | LATERAL | +0.0652 [-0.0758, +0.2208] | +0.0051 [-0.1074, +0.0970] |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0018 [+0.0008, +0.0029] * | +0.0012 [-0.0000, +0.0026] |
| TAC-LAT | TACTICAL | -0.0229 [-0.0362, -0.0104] * | -0.0183 [-0.0304, -0.0071] * |
| TAC-LON | TACTICAL | +0.0117 [-0.0083, +0.0317] | -0.0104 [-0.0288, +0.0071] |
| S-POS@30 | STRATEGIC | +0.3069 [+0.1434, +0.4840] * | +0.2698 [+0.0794, +0.4753] * |
| S-POS@50 | STRATEGIC | +0.5210 [+0.2244, +0.8415] * | +0.4698 [+0.1326, +0.8344] * |
| S-POS@80 | STRATEGIC | +0.7618 [+0.2944, +1.2414] * | +0.7814 [+0.2221, +1.3842] * |
| M1 crawl share | crawl | +0.0146 [+0.0040, +0.0274] * | +0.0162 [+0.0065, +0.0276] * |

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

## Instruments (from the tick dumps; beside, never deciding)

| rollout | bind overall | bind in M1 ticks (n) | cap overall | cap in M1 ticks | mean lift where bound, m/s | negative chord lift: n, share, min m/s | no-route ticks | v_log - v_f: check vs (1-kappa) v_log, max abs m/s | heads' MAE on these frames, m/s (share heads > logged) |
|---|---|---|---|---|---|---|---|---|---|
| RILFO075_s0 | 0.085 | 0.536 (237) | 0.051 | 0.042 | 0.493 | 31, 0.051, -0.243 | 987 (0.137) | 9.50e-07 | 2.583 (0.515) |
| RILFO075_s1 | 0.097 | 0.621 (240) | 0.051 | 0.067 | 0.511 | 24, 0.034, -0.116 | 987 (0.137) | 9.50e-07 | 2.588 (0.498) |
| RILFO075C_s0 | 0.088 | 0.576 (236) | 0.051 | 0.042 | 0.493 | 33, 0.052, -0.597 | 987 (0.137) | 9.50e-07 | 2.579 (0.514) |
| RILFO075C_s1 | 0.099 | 0.649 (239) | 0.051 | 0.067 | 0.522 | 29, 0.041, -0.126 | 987 (0.137) | 9.50e-07 | 2.583 (0.497) |

Beside, the heads-source arm LF075 (the LF read's rollouts, same instruments): 

| rollout | bind overall | bind in M1 ticks (n) | cap overall | mean lift where bound, m/s | v_f vs logged: MAE m/s, share v_f > logged (n) |
|---|---|---|---|---|---|
| RILF075_s0 | 0.187 | 0.455 (200) | 0.069 | 0.666 | 3.894, 0.191 (5793) |
| RILF075_s1 | 0.187 | 0.462 (199) | 0.050 | 0.662 | 3.979, 0.174 (5710) |

## Controls

| control | PASS |
|---|---|
| patched_tool_pin | True |
| rollout_RILFO075_s0 | True |
| rollout_RILFO075_s1 | True |
| rollout_RILFO075C_s0 | True |
| rollout_RILFO075C_s1 | True |
| one_checkpoint_per_seed | True |
| LFO_PRECHECK | True |
| card_a4 | True |
| card_a3 | True |
| cl_card_rules | True |
| llr_read_py | True |
| lf_read_py | True |
| reused_arrays | True |

## Variance named

* episodes: answered by the paired episode-cluster bootstrap (every interval here)
* training_run: O is C0's own checkpoint in every contrast with C0 / RIC0, so O's training variance is absent from it; the lead cap comes from one LLR run per seed (not measured); rule F_O reads against the C0REP floor
* inference: the planner, F and the oracle input are deterministic: no inference seed applies (the oracle input carries no draw)
