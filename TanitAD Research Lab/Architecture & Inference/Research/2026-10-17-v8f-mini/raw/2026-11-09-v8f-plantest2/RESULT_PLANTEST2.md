# RESULT_PLANTEST2: does a seeded and regularised solver make the level planners beat holding the action? (T0, zero training; agent, 2026-11-08)
Card `../PREREG_V8F_PLANTEST2.md` (prefix sha256 `d8217dfa...1a89`), IMPL `IMPL_CHOICES_PLANTEST2.md` (sha256 in `IMPL_CHOICES_PLANTEST2.sha256`). **Tier T0: a world-model diagnostic with an oracle (true-future) goal, never a driving result.** Evidence class: MEASURED (`runs/*.npz`, `RESULT_PLANTEST2.json`).
Estimator on every interval: EC-BOOT = paired episode-cluster bootstrap over the 71 SCORE episodes (994 windows), B 2,000, seed 0, 95 % percentile. Separated-better = interval wholly below 0 (MAE, lower is better). Lambda was chosen on the TUNE half only (hash split, salt `plantest2-v1`); every number below is SCORE. The interval answers 'another draw of episodes'; it is blind to training-seed and planner-seed variance, hence both training seeds are required and the planner-seed replicate is printed beside.

## Verdict (the card's committed table, per planner and family; both training seeds required)
| planner | family | chosen lambda s0 / s1 | M2b_s0 | M2b_s1 | decision label |
|---|---|---|---|---|---|
| TD3 | LONGITUDINAL | 1 / 1 | NO-GAIN | NO-GAIN | **NO-GAIN** |
| TD3 | LATERAL | 1 / 1 | NO-GAIN | NO-GAIN | **NO-GAIN** |
| FLAT1 | LONGITUDINAL | 1 / 1 | NO-GAIN | NO-GAIN | **NO-GAIN** |
| FLAT1 | LATERAL | 1 / 1 | NO-GAIN | NO-GAIN | **NO-GAIN** |
TACTICAL: not computed (n = 0: this read has no decision output). STRATEGIC: PAUSED (PI item 25, `PI_DECISION_QUEUE.md`).

HIER-ADDS (TD3 minus FLAT1 under G-TRUE, each planner at its own chosen lambda, separated-better at both seeds): LONGITUDINAL: NO; LATERAL: NO

## Deciding rows: MAE against the 12 logged actions, SCORE (EC-BOOT mean [95 % interval])
### LONGITUDINAL (acc_mae, m/s^2)
| model | planner | lambda | HOLD | G-TRUE | G-RAND | TRUE - HOLD | sep | TRUE - RAND | sep | plan distance from HOLD, G-TRUE (G-RAND) [prior-std units] | base label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M2b_s0 | td3 | 1 | 0.3348 [0.2880, 0.3826] | 1.2344 [1.1718, 1.2978] | 1.5091 [1.4375, 1.5769] | 0.8995 [0.8156, 0.9813] | worse | -0.2747 [-0.3666, -0.1886] | better | 1.442 (1.783) | NO-GAIN |
| M2b_s0 | flat1 | 1 | 0.3348 [0.2880, 0.3826] | 1.0430 [0.9830, 1.1065] | 1.5065 [1.4334, 1.5770] | 0.7082 [0.6355, 0.7833] | worse | -0.4635 [-0.5624, -0.3663] | better | 1.205 (1.763) | NO-GAIN |
| M2b_s1 | td3 | 1 | 0.3348 [0.2880, 0.3826] | 1.1820 [1.1115, 1.2519] | 1.3906 [1.3092, 1.4632] | 0.8472 [0.7694, 0.9245] | worse | -0.2086 [-0.3100, -0.1043] | better | 1.358 (1.623) | NO-GAIN |
| M2b_s1 | flat1 | 1 | 0.3348 [0.2880, 0.3826] | 1.0033 [0.9353, 1.0729] | 1.3980 [1.3228, 1.4710] | 0.6685 [0.5984, 0.7405] | worse | -0.3947 [-0.5003, -0.2880] | better | 1.145 (1.637) | NO-GAIN |

### LATERAL (yaw_mae, rad/s)
| model | planner | lambda | HOLD | G-TRUE | G-RAND | TRUE - HOLD | sep | TRUE - RAND | sep | plan distance from HOLD, G-TRUE (G-RAND) [prior-std units] | base label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M2b_s0 | td3 | 1 | 0.0256 [0.0197, 0.0320] | 0.1146 [0.1046, 0.1258] | 0.1494 [0.1399, 0.1586] | 0.0891 [0.0806, 0.0979] | worse | -0.0348 [-0.0459, -0.0228] | better | 1.214 (1.552) | NO-GAIN |
| M2b_s0 | flat1 | 1 | 0.0256 [0.0197, 0.0320] | 0.1020 [0.0940, 0.1107] | 0.1511 [0.1423, 0.1602] | 0.0765 [0.0693, 0.0840] | worse | -0.0491 [-0.0619, -0.0361] | better | 1.067 (1.582) | NO-GAIN |
| M2b_s1 | td3 | 1 | 0.0256 [0.0197, 0.0320] | 0.1188 [0.1092, 0.1293] | 0.1500 [0.1401, 0.1596] | 0.0932 [0.0853, 0.1012] | worse | -0.0312 [-0.0424, -0.0196] | better | 1.251 (1.547) | NO-GAIN |
| M2b_s1 | flat1 | 1 | 0.0256 [0.0197, 0.0320] | 0.1025 [0.0940, 0.1114] | 0.1493 [0.1399, 0.1582] | 0.0769 [0.0703, 0.0839] | worse | -0.0467 [-0.0584, -0.0349] | better | 1.069 (1.552) | NO-GAIN |

HOLD-COLLAPSE flag: mean absolute distance of the G-TRUE plan from HOLD below 0.10 of the TRAIN prior std of the channel (IMPL section 7); a flagged row does not count as planning.

## HIER-ADDS rows: TD3 minus FLAT1 under G-TRUE (negative = TD3 better), SCORE
| family | model | lambda TD3 / FLAT1 | TD3 - FLAT1 | sep |
|---|---|---|---|---|
| LONGITUDINAL | M2b_s0 | 1 / 1 | 0.1914 [0.1545, 0.2283] | worse |
| LONGITUDINAL | M2b_s1 | 1 / 1 | 0.1788 [0.1308, 0.2253] | worse |
| LATERAL | M2b_s0 | 1 / 1 | 0.0126 [0.0064, 0.0184] | worse |
| LATERAL | M2b_s1 | 1 / 1 | 0.0163 [0.0107, 0.0221] | worse |

## Beside, no bar: speed error, heading error and ADE at 2.4 s of the unicycle integration, plan vs logged actions (SCORE; at the family's chosen lambda; EC-BOOT mean [interval])
| model | planner | family (lambda) | ref | speed err [m/s] | heading err [rad] | ADE [m] |
|---|---|---|---|---|---|---|
| M2b_s0 | td3 | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | td3 | LONGITUDINAL (1) | true | 1.8637 [1.6989, 2.0315] | 0.1990 [0.1752, 0.2252] | 1.5183 [1.3602, 1.6805] |
| M2b_s0 | td3 | LONGITUDINAL (1) | rand | 3.0064 [2.7666, 3.2368] | 0.2545 [0.2288, 0.2811] | 2.0865 [1.9131, 2.2647] |
| M2b_s0 | td3 | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | td3 | LATERAL (1) | true | 1.8637 [1.6989, 2.0315] | 0.1990 [0.1752, 0.2252] | 1.5183 [1.3602, 1.6805] |
| M2b_s0 | td3 | LATERAL (1) | rand | 3.0064 [2.7666, 3.2368] | 0.2545 [0.2288, 0.2811] | 2.0865 [1.9131, 2.2647] |
| M2b_s0 | flat1 | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | flat1 | LONGITUDINAL (1) | true | 1.6545 [1.5140, 1.8051] | 0.1830 [0.1656, 0.2037] | 1.3656 [1.2278, 1.5293] |
| M2b_s0 | flat1 | LONGITUDINAL (1) | rand | 3.1187 [2.8812, 3.3538] | 0.2636 [0.2372, 0.2903] | 2.2701 [2.0578, 2.4931] |
| M2b_s0 | flat1 | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | flat1 | LATERAL (1) | true | 1.6545 [1.5140, 1.8051] | 0.1830 [0.1656, 0.2037] | 1.3656 [1.2278, 1.5293] |
| M2b_s0 | flat1 | LATERAL (1) | rand | 3.1187 [2.8812, 3.3538] | 0.2636 [0.2372, 0.2903] | 2.2701 [2.0578, 2.4931] |
| M2b_s1 | td3 | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | td3 | LONGITUDINAL (1) | true | 1.9321 [1.7580, 2.1219] | 0.2178 [0.1950, 0.2429] | 1.5389 [1.3932, 1.7057] |
| M2b_s1 | td3 | LONGITUDINAL (1) | rand | 2.5772 [2.3310, 2.8019] | 0.2710 [0.2472, 0.2971] | 2.0307 [1.8207, 2.2619] |
| M2b_s1 | td3 | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | td3 | LATERAL (1) | true | 1.9321 [1.7580, 2.1219] | 0.2178 [0.1950, 0.2429] | 1.5389 [1.3932, 1.7057] |
| M2b_s1 | td3 | LATERAL (1) | rand | 2.5772 [2.3310, 2.8019] | 0.2710 [0.2472, 0.2971] | 2.0307 [1.8207, 2.2619] |
| M2b_s1 | flat1 | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | flat1 | LONGITUDINAL (1) | true | 1.7039 [1.5543, 1.8629] | 0.1836 [0.1641, 0.2037] | 1.3593 [1.2378, 1.4985] |
| M2b_s1 | flat1 | LONGITUDINAL (1) | rand | 2.7750 [2.5452, 3.0004] | 0.2780 [0.2544, 0.3010] | 2.1796 [1.9530, 2.4315] |
| M2b_s1 | flat1 | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | flat1 | LATERAL (1) | true | 1.7039 [1.5543, 1.8629] | 0.1836 [0.1641, 0.2037] | 1.3593 [1.2378, 1.4985] |
| M2b_s1 | flat1 | LATERAL (1) | rand | 2.7750 [2.5452, 3.0004] | 0.2780 [0.2544, 0.3010] | 2.1796 [1.9530, 2.4315] |

## Beside, no bar: cost monotonicity per level (Tab. 10 statistic: -Spearman(goal distance, time); 1.0 = perfectly decreasing) and fraction of windows where planning lowered the cost below the best initial candidate (G-TRUE, SCORE; the cost includes REG when lambda > 0)
| run | level | monotonicity | frac lowered |
|---|---|---|---|
| M2b_s0|td3|LONGITUDINAL (lambda 1) | L1 | 0.2557 [0.1451, 0.3578] | 1.0000 [1.0000, 1.0000] |
| M2b_s0|td3|LONGITUDINAL (lambda 1) | L2 | 0.9179 [0.8968, 0.9354] | 1.0000 [1.0000, 1.0000] |
| M2b_s0|td3|LONGITUDINAL (lambda 1) | L3 | 0.8984 [0.8684, 0.9223] | 1.0000 [1.0000, 1.0000] |
| M2b_s0|td3|LATERAL (lambda 1) | L1 | 0.2557 [0.1451, 0.3578] | 1.0000 [1.0000, 1.0000] |
| M2b_s0|td3|LATERAL (lambda 1) | L2 | 0.9179 [0.8968, 0.9354] | 1.0000 [1.0000, 1.0000] |
| M2b_s0|td3|LATERAL (lambda 1) | L3 | 0.8984 [0.8684, 0.9223] | 1.0000 [1.0000, 1.0000] |
| M2b_s0|flat1|LONGITUDINAL (lambda 1) | L1 | -0.1475 [-0.2489, -0.0467] | 1.0000 [1.0000, 1.0000] |
| M2b_s0|flat1|LATERAL (lambda 1) | L1 | -0.1475 [-0.2489, -0.0467] | 1.0000 [1.0000, 1.0000] |
| M2b_s1|td3|LONGITUDINAL (lambda 1) | L1 | 0.1596 [0.0563, 0.2650] | 1.0000 [1.0000, 1.0000] |
| M2b_s1|td3|LONGITUDINAL (lambda 1) | L2 | 0.9290 [0.9054, 0.9487] | 1.0000 [1.0000, 1.0000] |
| M2b_s1|td3|LONGITUDINAL (lambda 1) | L3 | 0.8948 [0.8702, 0.9169] | 1.0000 [1.0000, 1.0000] |
| M2b_s1|td3|LATERAL (lambda 1) | L1 | 0.1596 [0.0563, 0.2650] | 1.0000 [1.0000, 1.0000] |
| M2b_s1|td3|LATERAL (lambda 1) | L2 | 0.9290 [0.9054, 0.9487] | 1.0000 [1.0000, 1.0000] |
| M2b_s1|td3|LATERAL (lambda 1) | L3 | 0.8948 [0.8702, 0.9169] | 1.0000 [1.0000, 1.0000] |
| M2b_s1|flat1|LONGITUDINAL (lambda 1) | L1 | -0.1746 [-0.2753, -0.0631] | 1.0000 [1.0000, 1.0000] |
| M2b_s1|flat1|LATERAL (lambda 1) | L1 | -0.1746 [-0.2753, -0.0631] | 1.0000 [1.0000, 1.0000] |

## Beside: planner-seed replicate (planner seed 1 minus seed 0, G-TRUE, same checkpoint and lambda, SCORE, EC-BOOT; a difference of this size is inference noise, not an effect)
* M2b_s0|td3|LONGITUDINAL (lambda 1): -0.0036 [-0.0102, 0.0033] (none)
* M2b_s0|td3|LATERAL (lambda 1): -0.0001 [-0.0011, 0.0007] (none)
* M2b_s0|flat1|LONGITUDINAL (lambda 1): 0.0015 [-0.0021, 0.0050] (none)
* M2b_s0|flat1|LATERAL (lambda 1): -0.0000 [-0.0005, 0.0005] (none)
* M2b_s1|td3|LONGITUDINAL (lambda 1): 0.0043 [-0.0025, 0.0112] (none)
* M2b_s1|td3|LATERAL (lambda 1): -0.0003 [-0.0013, 0.0006] (none)
* M2b_s1|flat1|LONGITUDINAL (lambda 1): 0.0014 [-0.0014, 0.0044] (none)
* M2b_s1|flat1|LATERAL (lambda 1): 0.0000 [-0.0003, 0.0004] (none)

## Beside: the TUNE grid (G-TRUE, planner seed 0, mean MAE over the TUNE windows; lambda is chosen as the row minimum per family; this is the selection record, not a result)
| model | planner | lambda | LONGITUDINAL acc MAE [m/s^2] | LATERAL yaw MAE [rad/s] |
|---|---|---|---|---|
| M2b_s0 | td3 | 0 | 1.4100 | 0.1260 |
| M2b_s0 | td3 | 0.01 | 1.4079 | 0.1259 |
| M2b_s0 | td3 | 0.03 | 1.4041 | 0.1254 |
| M2b_s0 | td3 | 0.1 | 1.3911 | 0.1238 |
| M2b_s0 | td3 | 0.3 | 1.3530 | 0.1190 |
| M2b_s0 | td3 | 1 | 1.2174 (chosen) | 0.1049 (chosen) |
| M2b_s0 | flat1 | 0 | 1.3946 | 0.1303 |
| M2b_s0 | flat1 | 0.01 | 1.3911 | 0.1300 |
| M2b_s0 | flat1 | 0.03 | 1.3827 | 0.1293 |
| M2b_s0 | flat1 | 0.1 | 1.3569 | 0.1270 |
| M2b_s0 | flat1 | 0.3 | 1.2822 | 0.1204 |
| M2b_s0 | flat1 | 1 | 1.0450 (chosen) | 0.0985 (chosen) |
| M2b_s1 | td3 | 0 | 1.2969 | 0.1385 |
| M2b_s1 | td3 | 0.01 | 1.2955 | 0.1383 |
| M2b_s1 | td3 | 0.03 | 1.2913 | 0.1378 |
| M2b_s1 | td3 | 0.1 | 1.2745 | 0.1358 |
| M2b_s1 | td3 | 0.3 | 1.2299 | 0.1304 |
| M2b_s1 | td3 | 1 | 1.0962 (chosen) | 0.1151 (chosen) |
| M2b_s1 | flat1 | 0 | 1.2780 | 0.1313 |
| M2b_s1 | flat1 | 0.01 | 1.2756 | 0.1309 |
| M2b_s1 | flat1 | 0.03 | 1.2680 | 0.1301 |
| M2b_s1 | flat1 | 0.1 | 1.2425 | 0.1275 |
| M2b_s1 | flat1 | 0.3 | 1.1719 | 0.1198 |
| M2b_s1 | flat1 | 1 | 0.9569 (chosen) | 0.0956 (chosen) |

Fraction of SCORE windows whose HOLD lies outside the level-1 box (HOLD_c differs from HOLD): M2b_s0 0.090, M2b_s1 0.090

## What each outcome triggers (committed in the card, Rule Zero)
* NO-GAIN appears in the table above: if NO-GAIN holds at every lambda, the next lever is the cost space (distance in decoded, action-relevant quantities instead of the raw latent), read on the ladder world model.
* Every outcome: PLANTEST and this solver are required gates of the ladder arms; the ladder is not dropped on a negative here (one-step loss, 0.8 s top step).

## Master Mind reading of the committed triggers (2026-11-09, dev box; added after the report script ran, every number below is quoted from the tables above)
* **Outcome: NO-GAIN** on TD3 and FLAT1, LONGITUDINAL and LATERAL, at both training seeds, at the chosen lambda on SCORE (T0, oracle goal, EC-BOOT). PLANNABLE did not occur on either family, so the card's first trigger does not fire.
* **The lambda grid.** Every lambda of the card's grid (0, 0.01, 0.03, 0.1, 0.3, 1) read G-TRUE above HOLD on the TUNE half: lowest acceleration MAE 0.9569 m/s^2 and lowest yaw MAE 0.0956 rad/s (M2b_s1 FLAT1, lambda 1), against HOLD 0.3348 and 0.0256 on SCORE. The two halves are different windows, so that comparison is indicative; the separated comparison is the SCORE table. Scope: lambda 1 is the top of the grid and MAE falls with lambda in every row, so the grid holds no interior optimum. What it shows is that moving the plan toward HOLD lowers the error; the plans at lambda 1 still sit 1.067 to 1.442 prior-std units from HOLD.
* **Trigger that fires:** "NO-GAIN at every lambda: the next lever is the cost space". The planners do lower their cost in every window (frac lowered 1.0000) and the goal does steer them (TRUE minus RAND separated-better in all 8 rows), so the search and the goal path run. Level 1's distance is the weak term: monotonicity 0.1596 to 0.2557 for TD3 and -0.1746 to -0.1475 for FLAT1, against 0.8948 to 0.9290 at levels 2 and 3. A level-1 latent distance that hardly falls along the true future is not a cost an action search can use.
* **HWM-PLAN solver slot:** PLANTEST2 did not read PLANNABLE, so the HWM-PLAN card takes PLANTEST's initialisation (`seed_mode = prior`, `lam = 0`), as that card committed.
* **The ladder stays:** this read is the current world models (one-step loss, 0.8 s top step). PLANTEST and this solver remain required gates of the ladder arms.
