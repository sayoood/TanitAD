# RESULT_PLANTEST: is today's hierarchical latent space plannable? (T0, zero training; agent, 2026-11-08)
Card `../PREREG_V8F_PLANTEST.md` (prefix sha256 `d5dbcc63...7783`), IMPL `IMPL_CHOICES_PLANTEST.md` (sha256 in `IMPL_CHOICES_PLANTEST.sha256`). **Tier T0: a world-model diagnostic with an oracle (true-future) goal, never a driving result.** Evidence class: MEASURED (this run, `runs/*.npz`, `RESULT_PLANTEST.json`) unless marked.
Estimator on every interval: EC-BOOT = paired episode-cluster bootstrap over 141 episodes, B 2,000, seed 0, 95 % percentile; windows n = 1976 (dropped for lacking the 2.4 s frame: 0). Separated-better = interval wholly below 0 (MAE, lower is better). The interval answers 'another draw of episodes'; it is blind to training-seed and planner-seed variance, hence both training seeds are required and the planner-seed floor is printed beside.

**Controls: all four PASS for TD3 and for FLAT1 (`RESULT_PLANTEST_controls.md`). The last-2-state FLAT does not pass control 1 (card log, 2026-11-08 10:45), so FLAT1 is the flat arm for every outcome and for HIER-ADDS, and FLAT is reported beside with no verdict.**

## Verdict (the card's committed table, per planner and family; both training seeds required)
| planner | family | M2b_s0 | M2b_s1 | decision label |
|---|---|---|---|---|
| TD3 | LONGITUDINAL | NO-GAIN | NO-GAIN | **NO-GAIN** |
| TD3 | LATERAL | NO-GAIN | NO-GAIN | **NO-GAIN** |
| FLAT1 | LONGITUDINAL | NO-GAIN | NO-GAIN | **NO-GAIN** |
| FLAT1 | LATERAL | NO-GAIN | NO-GAIN | **NO-GAIN** |
TACTICAL: not computed (n = 0: this read has no decision output). STRATEGIC: PAUSED (PI item 25, `PI_DECISION_QUEUE.md`).

HIER-ADDS (TD3 minus FLAT1 under G-TRUE, separated-better at both seeds): LONGITUDINAL: NO; LATERAL: NO

## Deciding rows: MAE against the 12 logged actions (EC-BOOT mean [95 % interval])
### LONGITUDINAL (acc_mae, m/s^2)
| model | HOLD | PRIOR | TD3 true | TD3 rand | TD3 stay | FLAT1 true | FLAT1 rand | FLAT1 stay | FLAT true (beside, no verdict) |
|---|---|---|---|---|---|---|---|---|---|
| M2b_s0 | 0.3296 [0.2973, 0.3623] | 0.4931 [0.4397, 0.5467] | 1.4228 [1.3800, 1.4647] | 1.6497 [1.6019, 1.6985] | 1.5581 [1.5122, 1.6031] | 1.3925 [1.3537, 1.4312] | 1.6367 [1.5836, 1.6908] | 1.5434 [1.4933, 1.5902] | 1.3898 [1.3514, 1.4283] |
| M2b_s1 | 0.3296 [0.2973, 0.3623] | 0.4931 [0.4397, 0.5467] | 1.3355 [1.2882, 1.3823] | 1.5411 [1.4902, 1.5939] | 1.4607 [1.4022, 1.5169] | 1.2984 [1.2536, 1.3435] | 1.5255 [1.4640, 1.5857] | 1.4265 [1.3654, 1.4850] | 1.2963 [1.2516, 1.3415] |

| model | planner | TRUE - HOLD | sep | TRUE - RAND | sep | TRUE - STAY (beside) | TRUE - PRIOR (beside) | label |
|---|---|---|---|---|---|---|---|---|
| M2b_s0 | td3 | 1.0932 [1.0337, 1.1532] | worse | -0.2270 [-0.2933, -0.1624] | better | -0.1354 [-0.1782, -0.0922] | 0.9297 [0.8492, 1.0118] | NO-GAIN |
| M2b_s0 | flat1 | 1.0629 [1.0098, 1.1170] | worse | -0.2442 [-0.3147, -0.1749] | better | -0.1509 [-0.1932, -0.1078] | 0.8994 [0.8252, 0.9728] | NO-GAIN |
| M2b_s1 | td3 | 1.0059 [0.9476, 1.0628] | worse | -0.2056 [-0.2768, -0.1353] | better | -0.1252 [-0.1644, -0.0851] | 0.8424 [0.7678, 0.9162] | NO-GAIN |
| M2b_s1 | flat1 | 0.9688 [0.9152, 1.0214] | worse | -0.2270 [-0.3090, -0.1460] | better | -0.1281 [-0.1705, -0.0853] | 0.8053 [0.7341, 0.8765] | NO-GAIN |

### LATERAL (yaw_mae, rad/s)
| model | HOLD | PRIOR | TD3 true | TD3 rand | TD3 stay | FLAT1 true | FLAT1 rand | FLAT1 stay | FLAT true (beside, no verdict) |
|---|---|---|---|---|---|---|---|---|---|
| M2b_s0 | 0.0224 [0.0185, 0.0263] | 0.0371 [0.0300, 0.0449] | 0.1338 [0.1272, 0.1406] | 0.1644 [0.1584, 0.1702] | 0.1270 [0.1206, 0.1341] | 0.1344 [0.1285, 0.1405] | 0.1594 [0.1526, 0.1660] | 0.1290 [0.1226, 0.1361] | 0.1344 [0.1286, 0.1403] |
| M2b_s1 | 0.0224 [0.0185, 0.0263] | 0.0371 [0.0300, 0.0449] | 0.1422 [0.1361, 0.1483] | 0.1640 [0.1580, 0.1695] | 0.1360 [0.1298, 0.1425] | 0.1374 [0.1315, 0.1430] | 0.1616 [0.1547, 0.1680] | 0.1315 [0.1250, 0.1379] | 0.1370 [0.1312, 0.1426] |

| model | planner | TRUE - HOLD | sep | TRUE - RAND | sep | TRUE - STAY (beside) | TRUE - PRIOR (beside) | label |
|---|---|---|---|---|---|---|---|---|
| M2b_s0 | td3 | 0.1114 [0.1057, 0.1175] | worse | -0.0306 [-0.0381, -0.0229] | better | 0.0068 [0.0037, 0.0098] | 0.0967 [0.0892, 0.1039] | NO-GAIN |
| M2b_s0 | flat1 | 0.1121 [0.1067, 0.1175] | worse | -0.0250 [-0.0346, -0.0158] | better | 0.0054 [0.0017, 0.0091] | 0.0973 [0.0895, 0.1045] | NO-GAIN |
| M2b_s1 | td3 | 0.1199 [0.1144, 0.1255] | worse | -0.0217 [-0.0288, -0.0148] | better | 0.0063 [0.0026, 0.0098] | 0.1051 [0.0967, 0.1129] | NO-GAIN |
| M2b_s1 | flat1 | 0.1150 [0.1098, 0.1203] | worse | -0.0243 [-0.0320, -0.0159] | better | 0.0058 [0.0027, 0.0091] | 0.1003 [0.0921, 0.1077] | NO-GAIN |

## HIER-ADDS rows: TD3 minus FLAT1 under G-TRUE (negative = TD3 better)
| family | model | TD3 - FLAT1 | sep | TD3 - FLAT (beside, no verdict) | sep |
|---|---|---|---|---|---|
| LONGITUDINAL | M2b_s0 | 0.0302 [0.0029, 0.0559] | worse | 0.0329 [0.0065, 0.0576] | worse |
| LONGITUDINAL | M2b_s1 | 0.0371 [0.0091, 0.0636] | worse | 0.0392 [0.0115, 0.0659] | worse |
| LATERAL | M2b_s0 | -0.0006 [-0.0048, 0.0036] | none | -0.0006 [-0.0047, 0.0036] | none |
| LATERAL | M2b_s1 | 0.0049 [0.0009, 0.0087] | worse | 0.0052 [0.0013, 0.0091] | worse |

## Beside, no bar: speed error, heading error and ADE of the unicycle integration, plan vs logged actions (G-TRUE; EC-BOOT mean [interval])
| model | planner/ref | speed err at 2.4 s [m/s] | heading err at 2.4 s [rad] | ADE [m] |
|---|---|---|---|---|
| M2b_s0 | hold | 0.6796 [0.6111, 0.7470] | 0.0499 [0.0408, 0.0591] | 0.2164 [0.1974, 0.2352] |
| M2b_s0 | prior | 1.0680 [0.9415, 1.1927] | 0.0837 [0.0668, 0.1019] | 0.6385 [0.5629, 0.7188] |
| M2b_s0 | td3_true | 2.0832 [1.9482, 2.2206] | 0.2202 [0.2038, 0.2379] | 1.6245 [1.4986, 1.7524] |
| M2b_s0 | td3_rand | 3.2449 [3.0918, 3.4003] | 0.2751 [0.2568, 0.2934] | 2.2172 [2.0716, 2.3722] |
| M2b_s0 | flat1_true | 2.0005 [1.8840, 2.1284] | 0.2109 [0.1964, 0.2270] | 1.7107 [1.5850, 1.8394] |
| M2b_s0 | flat1_rand | 3.3765 [3.2036, 3.5421] | 0.2724 [0.2537, 0.2920] | 2.3505 [2.1906, 2.5169] |
| M2b_s0 | flat_true | 2.0132 [1.8929, 2.1415] | 0.2121 [0.1980, 0.2283] | 1.7059 [1.5806, 1.8332] |
| M2b_s0 | flat_rand | 3.3577 [3.1890, 3.5252] | 0.2704 [0.2518, 0.2903] | 2.3409 [2.1831, 2.5067] |
| M2b_s1 | hold | 0.6796 [0.6111, 0.7470] | 0.0499 [0.0408, 0.0591] | 0.2164 [0.1974, 0.2352] |
| M2b_s1 | prior | 1.0680 [0.9415, 1.1927] | 0.0837 [0.0668, 0.1019] | 0.6385 [0.5629, 0.7188] |
| M2b_s1 | td3_true | 2.0306 [1.8990, 2.1623] | 0.2389 [0.2232, 0.2549] | 1.6616 [1.5401, 1.7940] |
| M2b_s1 | td3_rand | 2.8286 [2.6564, 2.9996] | 0.2844 [0.2664, 0.3017] | 2.1011 [1.9471, 2.2563] |
| M2b_s1 | flat1_true | 1.9293 [1.8117, 2.0459] | 0.2200 [0.2058, 0.2345] | 1.6923 [1.5843, 1.8125] |
| M2b_s1 | flat1_rand | 3.0033 [2.8284, 3.1854] | 0.2825 [0.2646, 0.2992] | 2.2690 [2.1063, 2.4457] |
| M2b_s1 | flat_true | 1.9237 [1.8074, 2.0385] | 0.2191 [0.2051, 0.2335] | 1.6686 [1.5618, 1.7854] |
| M2b_s1 | flat_rand | 3.0329 [2.8559, 3.2145] | 0.2811 [0.2636, 0.2978] | 2.2705 [2.1059, 2.4498] |

## Beside, no bar: cost monotonicity per level (Tab. 10 statistic: -Spearman(goal distance, time); 1.0 = perfectly decreasing) and fraction of windows where planning lowered the cost below the best initial candidate
| run | level | monotonicity | frac lowered | mean iterations |
|---|---|---|---|---|
| M2b_s0|td3|true | L1 | 0.2575 [0.1764, 0.3361] | 1.0000 [1.0000, 1.0000] | 62.6 |
| M2b_s0|td3|true | L2 | 0.9205 [0.9057, 0.9342] | 1.0000 [1.0000, 1.0000] | 90.0 |
| M2b_s0|td3|true | L3 | 0.8982 [0.8770, 0.9167] | 1.0000 [1.0000, 1.0000] | 90.0 |
| M2b_s0|flat|true | L1 | -0.0967 [-0.1773, -0.0236] | 1.0000 [1.0000, 1.0000] | 54.8 |
| M2b_s0|flat1|true | L1 | -0.0997 [-0.1794, -0.0264] | 1.0000 [1.0000, 1.0000] | 55.5 |
| M2b_s1|td3|true | L1 | 0.2128 [0.1401, 0.2842] | 1.0000 [1.0000, 1.0000] | 68.0 |
| M2b_s1|td3|true | L2 | 0.9260 [0.9118, 0.9381] | 1.0000 [1.0000, 1.0000] | 90.0 |
| M2b_s1|td3|true | L3 | 0.8946 [0.8758, 0.9118] | 1.0000 [1.0000, 1.0000] | 90.0 |
| M2b_s1|flat|true | L1 | -0.1211 [-0.1979, -0.0446] | 1.0000 [1.0000, 1.0000] | 56.2 |
| M2b_s1|flat1|true | L1 | -0.1253 [-0.2025, -0.0473] | 1.0000 [1.0000, 1.0000] | 56.9 |

## Beside: planner-seed floor (planner seed 1 minus seed 0, G-TRUE, same checkpoint, EC-BOOT; a difference of this size is inference noise, not an effect)
* M2b_s0|td3|LONGITUDINAL: 0.0066 [0.0010, 0.0126] (worse)
* M2b_s0|td3|LATERAL: -0.0001 [-0.0007, 0.0006] (none)
* M2b_s0|flat|LONGITUDINAL: 0.0024 [-0.0007, 0.0055] (none)
* M2b_s0|flat|LATERAL: 0.0003 [-0.0001, 0.0006] (none)
* M2b_s1|td3|LONGITUDINAL: 0.0014 [-0.0047, 0.0079] (none)
* M2b_s1|td3|LATERAL: 0.0001 [-0.0006, 0.0007] (none)
* M2b_s1|flat|LONGITUDINAL: 0.0009 [-0.0020, 0.0042] (none)
* M2b_s1|flat|LATERAL: 0.0001 [-0.0003, 0.0005] (none)

## Beside: two untrained stacks (seeds 5000 + r) on M2b_s0's level-1 latents, 256 windows (the card's no-learned-dynamics reading; EC-BOOT on the subset; the label applies the card's rule for display only)
* rand0|td3 LONGITUDINAL: TRUE 1.6674 [1.6443, 1.6908], RAND 1.6672 [1.6439, 1.6904], HOLD 0.3371 [0.2945, 0.3806]; TRUE-RAND 0.0001 [-0.0008, 0.0010]; TRUE-HOLD 1.3302 [1.2886, 1.3695]; rule reads GOAL-BLIND
* rand0|td3 LATERAL: TRUE 0.1914 [0.1870, 0.1960], RAND 0.1914 [0.1869, 0.1960], HOLD 0.0230 [0.0180, 0.0279]; TRUE-RAND 0.0000 [-0.0001, 0.0001]; TRUE-HOLD 0.1684 [0.1638, 0.1727]; rule reads GOAL-BLIND
* rand0|flat LONGITUDINAL: TRUE 1.6462 [1.6224, 1.6716], RAND 1.6538 [1.6278, 1.6821], HOLD 0.3371 [0.2945, 0.3806]; TRUE-RAND -0.0076 [-0.0381, 0.0219]; TRUE-HOLD 1.3090 [1.2653, 1.3538]; rule reads GOAL-BLIND
* rand0|flat LATERAL: TRUE 0.1900 [0.1854, 0.1954], RAND 0.1909 [0.1861, 0.1960], HOLD 0.0230 [0.0180, 0.0279]; TRUE-RAND -0.0009 [-0.0040, 0.0019]; TRUE-HOLD 0.1670 [0.1622, 0.1719]; rule reads GOAL-BLIND
* rand1|td3 LONGITUDINAL: TRUE 1.6150 [1.5794, 1.6513], RAND 1.6141 [1.5782, 1.6511], HOLD 0.3371 [0.2945, 0.3806]; TRUE-RAND 0.0010 [-0.0015, 0.0044]; TRUE-HOLD 1.2779 [1.2355, 1.3226]; rule reads GOAL-BLIND
* rand1|td3 LATERAL: TRUE 0.1937 [0.1890, 0.1992], RAND 0.1937 [0.1890, 0.1992], HOLD 0.0230 [0.0180, 0.0279]; TRUE-RAND 0.0000 [-0.0002, 0.0002]; TRUE-HOLD 0.1708 [0.1656, 0.1759]; rule reads GOAL-BLIND
* rand1|flat LONGITUDINAL: TRUE 1.6318 [1.5980, 1.6675], RAND 1.5940 [1.5522, 1.6365], HOLD 0.3371 [0.2945, 0.3806]; TRUE-RAND 0.0379 [-0.0052, 0.0827]; TRUE-HOLD 1.2947 [1.2450, 1.3454]; rule reads GOAL-BLIND
* rand1|flat LATERAL: TRUE 0.1868 [0.1825, 0.1917], RAND 0.1881 [0.1832, 0.1939], HOLD 0.0230 [0.0180, 0.0279]; TRUE-RAND -0.0013 [-0.0046, 0.0020]; TRUE-HOLD 0.1639 [0.1592, 0.1685]; rule reads GOAL-BLIND


## Reading and what it triggers (Master Mind, 2026-11-08)
* **The committed trigger runs.** NO-GAIN on a family commits the planner's cost and solver next, then the world model. `PREREG_V8F_PLANTEST2.md` (prefix sha256 `d8217dfa...`, HOLD-seeded and regularised solver, TUNE / SCORE split over the 141 episodes) was frozen from these deciding rows and is running; its result decides the solver of the three-level planner.
* **What the goal dependence means.** Trained models separate G-TRUE from G-RAND on both families (for example LONGITUDINAL TD3 TRUE - RAND -0.2270 [-0.2933, -0.1624] at M2b_s0). Two untrained stacks read GOAL-BLIND on all eight rows (TRUE - RAND intervals all contain zero, largest magnitude 0.0379 [-0.0052, 0.0827], 256 windows). So the goal dependence is learned, not produced by the instrument. Twelve untrained-stack units failed in the first run with a missing import path (`run_plantest.run1.log`, rc 1, phase beside); they were rerun with that path added (`run_plantest.rand.log`, 12 ok, 0 errors) and the report rebuilt. The deciding rows and the planner-seed floor come from the first run.
* **What the size means.** Planning to a true goal is worse than holding the action by 0.97 to 1.09 m/s^2 of acceleration MAE, with a planner-seed floor of at most 0.0066 m/s^2 (M2b_s0 TD3), so the gap is not sampling noise of the planner. HIER-ADDS is NO: the three-level planner does not beat the flat one here.
* **Scope, by the PI's horizon ruling (D-HWM-BANDS-1).** These are M2b world models, trained one step at 0.2 / 0.4 / 0.8 s, so no level is trained inside its band. This read is about them and is T0 with an oracle goal: never a driving result. The ladder retrain (0.2 / 0.6 / 3.0 s, trained reach 2 / 6 / 18 s, `PREREG_V8F_LADDER.md`) carries PLANTEST as a required gate and is not dropped on this negative.
* TACTICAL: not computed, n = 0, this read has no decision output. STRATEGIC: PAUSED (PI item 25).
* The per-unit plans (`runs/*.npz`, 34 MB) stay at `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/plantest/runs`.
