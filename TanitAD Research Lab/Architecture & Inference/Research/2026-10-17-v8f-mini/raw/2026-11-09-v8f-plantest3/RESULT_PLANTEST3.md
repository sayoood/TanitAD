# RESULT_PLANTEST3: does a cost in decoded, action-relevant quantities make the level planners beat holding the action? (T0, zero training; agent, 2026-11-09)
Card `../PREREG_V8F_PLANTEST3.md` (prefix sha256 `1f9609b4...b348f9`), IMPL `IMPL_CHOICES_PLANTEST3.md` (sha256 in `IMPL_CHOICES_PLANTEST3.sha256`). **Tier T0: a world-model diagnostic with an oracle (true-future) goal, never a driving result.** Evidence class: MEASURED (`runs/*.npz`, `RESULT_PLANTEST3.json`).
Estimator on every interval: EC-BOOT = paired episode-cluster bootstrap over the 71 SCORE episodes (994 windows), B 2,000, seed 0, 95 % percentile. Separated-better = interval wholly below 0 (MAE, lower is better). Decoders and lambda were chosen on the TUNE half only (hash split, salt `plantest2-v1`); every number below is SCORE. The interval answers 'another draw of episodes'; it is blind to training-seed and planner-seed variance, hence both training seeds are required and the planner-seed replicate is printed beside.
A read is VALID only if the six controls of `RESULT_PLANTEST3_controls.md` passed; this report does not check that file.

## Decoder gate (before planning; TUNE inner split, held-out episodes)
**DECODER-WEAK: YES** (M2b_s0, M2b_s1); bars: held-out R2 on encoded latents below 0.70 (speed) or 0.50 (yaw rate) at either seed. Missing gate records: none.
| model | read | R2 speed | R2 yaw rate | n speed / n yaw |
|---|---|---|---|---|
| M2b_s0 | D_1 encoded, all knots (the DECODER-WEAK quantity) | 0.2707 | -0.1224 | 3808 / 3808 |
| M2b_s0 | D_1 encoded, k = 12 | 0.2862 | -0.0629 | 238 / 238 |
| M2b_s0 | D_1 on the rollout at k = 12 under the logged actions | 0.3377 | -0.0051 | 238 / 238 |
| M2b_s0 | D_2 encoded, all knots | 0.3401 | 0.1536 | 3808 / 3808 |
| M2b_s1 | D_1 encoded, all knots (the DECODER-WEAK quantity) | 0.3145 | 0.2680 | 3808 / 3808 |
| M2b_s1 | D_1 encoded, k = 12 | 0.3230 | 0.2861 | 238 / 238 |
| M2b_s1 | D_1 on the rollout at k = 12 under the logged actions | -0.0062 | 0.2177 | 238 / 238 |
| M2b_s1 | D_2 encoded, all knots | 0.2941 | 0.1756 | 3808 / 3808 |

## Verdict (the card's committed table; per arm, planner and family; both training seeds required)
| planner | arm | form | family | chosen lambda s0 / s1 | M2b_s0 | M2b_s1 | decision label |
|---|---|---|---|---|---|---|---|
| FLAT1 | DEC-END | plain | LONGITUDINAL | 1 / 1 | NO-GAIN | NO-GAIN | **NO-GAIN** |
| FLAT1 | DEC-END | plain | LATERAL | 1 / 1 | NO-GAIN | NO-GAIN | **NO-GAIN** |
| FLAT1 | DEC-END | shrink | LONGITUDINAL | 1 / 1 | NO-GAIN (collapse) | NO-GAIN (collapse) | **NO-GAIN +HOLD-COLLAPSE(M2b_s0,M2b_s1)** |
| FLAT1 | DEC-END | shrink | LATERAL | 1 / 1 | NO-GAIN | NO-GAIN | **NO-GAIN** |
| FLAT1 | DEC-PATH | plain | LONGITUDINAL | 1 / 1 | NO-GAIN | NO-GAIN | **NO-GAIN** |
| FLAT1 | DEC-PATH | plain | LATERAL | 1 / 1 | NO-GAIN | NO-GAIN | **NO-GAIN** |
| FLAT1 | DEC-PATH | shrink | LONGITUDINAL | 1 / 1 | NO-GAIN | NO-GAIN | **NO-GAIN** |
| FLAT1 | DEC-PATH | shrink | LATERAL | 1 / 1 | NO-GAIN | NO-GAIN | **NO-GAIN** |
| TD3 | DEC-END | shrink / shrink | LONGITUDINAL | 1 / 1 | NO-GAIN | NO-GAIN | **NO-GAIN** |
| TD3 | DEC-END | shrink / shrink | LATERAL | 1 / 1 | NO-GAIN | NO-GAIN | **NO-GAIN** |
TACTICAL: not computed (n = 0: this read has no decision output). STRATEGIC: PAUSED (PI item 25, `PI_DECISION_QUEUE.md`).

HIER-ADDS (TD3 DEC-END minus FLAT1 DEC-END of the same form and lambda, G-TRUE, separated-better at both seeds): LONGITUDINAL: NO; LATERAL: NO

## KIN-CEILING (no world model, no planner): actions from the decoded true-future speed differences and yaw rates, MAE against the 12 logged actions, SCORE
| model | family | unit | HOLD | KIN | KIN-RAND (donor's decoded future) | KIN - HOLD | sep | KIN - KIN-RAND | sep |
|---|---|---|---|---|---|---|---|---|---|
| M2b_s0 | LONGITUDINAL | m/s^2 | 0.3348 [0.2880, 0.3826] | 4.4463 [4.2071, 4.7019] | 4.5498 [4.3209, 4.7981] | 4.1115 [3.8557, 4.3708] | worse | -0.1035 [-0.4312, 0.2559] | none |
| M2b_s0 | LATERAL | rad/s | 0.0256 [0.0197, 0.0320] | 0.0629 [0.0524, 0.0745] | 0.0662 [0.0550, 0.0781] | 0.0373 [0.0301, 0.0453] | worse | -0.0033 [-0.0123, 0.0052] | none |
| M2b_s1 | LONGITUDINAL | m/s^2 | 0.3348 [0.2880, 0.3826] | 4.3712 [4.1200, 4.6308] | 4.3936 [4.1471, 4.6570] | 4.0364 [3.7708, 4.3019] | worse | -0.0224 [-0.3704, 0.3399] | none |
| M2b_s1 | LATERAL | rad/s | 0.0256 [0.0197, 0.0320] | 0.0605 [0.0506, 0.0717] | 0.0655 [0.0546, 0.0772] | 0.0349 [0.0282, 0.0425] | worse | -0.0049 [-0.0140, 0.0031] | none |
KIN beats HOLD (separated-better at both seeds): LONGITUDINAL: NO; LATERAL: NO

## Deciding rows: MAE against the 12 logged actions, SCORE (EC-BOOT mean [95 % interval])
### LONGITUDINAL (acc_mae, m/s^2)
| model | planner | arm | form | lambda | HOLD | G-TRUE | G-RAND | TRUE - HOLD | sep | TRUE - RAND | sep | distance from HOLD, G-TRUE (G-RAND) [prior-std units] | base label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M2b_s0 | flat1 | DEC-END | plain | 1 | 0.3348 [0.2880, 0.3826] | 0.3441 [0.3003, 0.3890] | 0.4036 [0.3521, 0.4523] | 0.0093 [0.0021, 0.0172] | worse | -0.0595 [-0.0853, -0.0367] | better | 0.102 (0.217) | NO-GAIN |
| M2b_s0 | flat1 | DEC-END | shrink | 1 | 0.3348 [0.2880, 0.3826] | 0.3336 [0.2892, 0.3791] | 0.3566 [0.3104, 0.4018] | -0.0012 [-0.0060, 0.0031] | none | -0.0230 [-0.0351, -0.0124] | better | 0.058 (0.120) COLLAPSE | NO-GAIN |
| M2b_s0 | flat1 | DEC-PATH | plain | 1 | 0.3348 [0.2880, 0.3826] | 0.5094 [0.4673, 0.5504] | 0.8528 [0.7617, 0.9445] | 0.1746 [0.1445, 0.2091] | worse | -0.3434 [-0.4320, -0.2570] | better | 0.383 (0.846) | NO-GAIN |
| M2b_s0 | flat1 | DEC-PATH | shrink | 1 | 0.3348 [0.2880, 0.3826] | 0.4022 [0.3591, 0.4455] | 0.6902 [0.6057, 0.7757] | 0.0674 [0.0514, 0.0849] | worse | -0.2879 [-0.3658, -0.2143] | better | 0.207 (0.609) | NO-GAIN |
| M2b_s0 | td3 | DEC-END | shrink | 1 | 0.3348 [0.2880, 0.3826] | 0.3895 [0.3430, 0.4379] | 0.4706 [0.4112, 0.5337] | 0.0547 [0.0387, 0.0744] | worse | -0.0810 [-0.1311, -0.0451] | better | 0.168 (0.288) | NO-GAIN |
| M2b_s1 | flat1 | DEC-END | plain | 1 | 0.3348 [0.2880, 0.3826] | 0.3519 [0.3070, 0.3970] | 0.4326 [0.3760, 0.4939] | 0.0171 [0.0083, 0.0277] | worse | -0.0806 [-0.1262, -0.0421] | better | 0.127 (0.277) | NO-GAIN |
| M2b_s1 | flat1 | DEC-END | shrink | 1 | 0.3348 [0.2880, 0.3826] | 0.3365 [0.2918, 0.3817] | 0.3754 [0.3280, 0.4239] | 0.0017 [-0.0032, 0.0072] | none | -0.0389 [-0.0606, -0.0207] | better | 0.075 (0.164) COLLAPSE | NO-GAIN |
| M2b_s1 | flat1 | DEC-PATH | plain | 1 | 0.3348 [0.2880, 0.3826] | 0.5384 [0.4827, 0.5962] | 0.8908 [0.7947, 0.9923] | 0.2035 [0.1677, 0.2457] | worse | -0.3524 [-0.4518, -0.2555] | better | 0.429 (0.907) | NO-GAIN |
| M2b_s1 | flat1 | DEC-PATH | shrink | 1 | 0.3348 [0.2880, 0.3826] | 0.4381 [0.3918, 0.4860] | 0.7548 [0.6659, 0.8468] | 0.1033 [0.0814, 0.1289] | worse | -0.3167 [-0.4029, -0.2353] | better | 0.274 (0.713) | NO-GAIN |
| M2b_s1 | td3 | DEC-END | shrink | 1 | 0.3348 [0.2880, 0.3826] | 0.3851 [0.3399, 0.4311] | 0.4239 [0.3700, 0.4822] | 0.0503 [0.0313, 0.0725] | worse | -0.0388 [-0.0663, -0.0140] | better | 0.172 (0.247) | NO-GAIN |

### LATERAL (yaw_mae, rad/s)
| model | planner | arm | form | lambda | HOLD | G-TRUE | G-RAND | TRUE - HOLD | sep | TRUE - RAND | sep | distance from HOLD, G-TRUE (G-RAND) [prior-std units] | base label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M2b_s0 | flat1 | DEC-END | plain | 1 | 0.0256 [0.0197, 0.0320] | 0.0308 [0.0247, 0.0376] | 0.0379 [0.0316, 0.0448] | 0.0052 [0.0036, 0.0070] | worse | -0.0071 [-0.0102, -0.0044] | better | 0.150 (0.259) | NO-GAIN |
| M2b_s0 | flat1 | DEC-END | shrink | 1 | 0.0256 [0.0197, 0.0320] | 0.0282 [0.0220, 0.0351] | 0.0323 [0.0260, 0.0389] | 0.0026 [0.0014, 0.0040] | worse | -0.0040 [-0.0055, -0.0028] | better | 0.104 (0.177) | NO-GAIN |
| M2b_s0 | flat1 | DEC-PATH | plain | 1 | 0.0256 [0.0197, 0.0320] | 0.0572 [0.0507, 0.0642] | 0.0963 [0.0866, 0.1067] | 0.0316 [0.0270, 0.0364] | worse | -0.0391 [-0.0497, -0.0294] | better | 0.481 (0.930) | NO-GAIN |
| M2b_s0 | flat1 | DEC-PATH | shrink | 1 | 0.0256 [0.0197, 0.0320] | 0.0425 [0.0364, 0.0493] | 0.0792 [0.0699, 0.0890] | 0.0169 [0.0141, 0.0199] | worse | -0.0367 [-0.0460, -0.0282] | better | 0.302 (0.731) | NO-GAIN |
| M2b_s0 | td3 | DEC-END | shrink | 1 | 0.0256 [0.0197, 0.0320] | 0.0330 [0.0265, 0.0405] | 0.0412 [0.0339, 0.0496] | 0.0074 [0.0055, 0.0098] | worse | -0.0081 [-0.0134, -0.0042] | better | 0.181 (0.271) | NO-GAIN |
| M2b_s1 | flat1 | DEC-END | plain | 1 | 0.0256 [0.0197, 0.0320] | 0.0298 [0.0238, 0.0364] | 0.0390 [0.0324, 0.0460] | 0.0042 [0.0029, 0.0056] | worse | -0.0092 [-0.0132, -0.0059] | better | 0.144 (0.270) | NO-GAIN |
| M2b_s1 | flat1 | DEC-END | shrink | 1 | 0.0256 [0.0197, 0.0320] | 0.0277 [0.0217, 0.0344] | 0.0321 [0.0258, 0.0390] | 0.0021 [0.0010, 0.0034] | worse | -0.0044 [-0.0060, -0.0030] | better | 0.105 (0.174) | NO-GAIN |
| M2b_s1 | flat1 | DEC-PATH | plain | 1 | 0.0256 [0.0197, 0.0320] | 0.0577 [0.0512, 0.0645] | 0.1087 [0.0973, 0.1204] | 0.0321 [0.0274, 0.0373] | worse | -0.0511 [-0.0632, -0.0382] | better | 0.498 (1.069) | NO-GAIN |
| M2b_s1 | flat1 | DEC-PATH | shrink | 1 | 0.0256 [0.0197, 0.0320] | 0.0434 [0.0376, 0.0500] | 0.0905 [0.0794, 0.1016] | 0.0178 [0.0147, 0.0211] | worse | -0.0471 [-0.0577, -0.0365] | better | 0.326 (0.854) | NO-GAIN |
| M2b_s1 | td3 | DEC-END | shrink | 1 | 0.0256 [0.0197, 0.0320] | 0.0319 [0.0250, 0.0397] | 0.0359 [0.0281, 0.0444] | 0.0063 [0.0041, 0.0092] | worse | -0.0039 [-0.0071, -0.0012] | better | 0.175 (0.214) | NO-GAIN |

HOLD-COLLAPSE flag: mean absolute distance of the G-TRUE plan from HOLD below 0.10 of the TRAIN prior std of the channel; a flagged row does not count as planning.

## HIER-ADDS rows: TD3 minus FLAT1 under G-TRUE, DEC-END (negative = TD3 better), SCORE
| family | model | form / lambda | TD3 - FLAT1 | sep |
|---|---|---|---|---|
| LONGITUDINAL | M2b_s0 | shrink / 1 | 0.0559 [0.0402, 0.0744] | worse |
| LONGITUDINAL | M2b_s1 | shrink / 1 | 0.0486 [0.0321, 0.0684] | worse |
| LATERAL | M2b_s0 | shrink / 1 | 0.0048 [0.0035, 0.0064] | worse |
| LATERAL | M2b_s1 | shrink / 1 | 0.0042 [0.0024, 0.0064] | worse |

## Beside, no bar: speed error, heading error and ADE at 2.4 s of the unicycle integration, plan vs logged actions (SCORE; at the family's chosen lambda; EC-BOOT mean [interval])
| model | planner | arm | form | family (lambda) | ref | speed err [m/s] | heading err [rad] | ADE [m] |
|---|---|---|---|---|---|---|---|---|
| M2b_s0 | flat1 | DEC-END | plain | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | flat1 | DEC-END | plain | LONGITUDINAL (1) | true | 0.6794 [0.5925, 0.7724] | 0.0663 [0.0530, 0.0814] | 0.2855 [0.2532, 0.3212] |
| M2b_s0 | flat1 | DEC-END | plain | LONGITUDINAL (1) | rand | 0.8055 [0.7029, 0.9067] | 0.0820 [0.0681, 0.0973] | 0.4027 [0.3541, 0.4603] |
| M2b_s0 | flat1 | DEC-END | plain | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | flat1 | DEC-END | plain | LATERAL (1) | true | 0.6794 [0.5925, 0.7724] | 0.0663 [0.0530, 0.0814] | 0.2855 [0.2532, 0.3212] |
| M2b_s0 | flat1 | DEC-END | plain | LATERAL (1) | rand | 0.8055 [0.7029, 0.9067] | 0.0820 [0.0681, 0.0973] | 0.4027 [0.3541, 0.4603] |
| M2b_s0 | flat1 | DEC-END | shrink | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | flat1 | DEC-END | shrink | LONGITUDINAL (1) | true | 0.6654 [0.5756, 0.7623] | 0.0613 [0.0480, 0.0764] | 0.2594 [0.2271, 0.2937] |
| M2b_s0 | flat1 | DEC-END | shrink | LONGITUDINAL (1) | rand | 0.7152 [0.6233, 0.8082] | 0.0707 [0.0569, 0.0854] | 0.3126 [0.2777, 0.3490] |
| M2b_s0 | flat1 | DEC-END | shrink | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | flat1 | DEC-END | shrink | LATERAL (1) | true | 0.6654 [0.5756, 0.7623] | 0.0613 [0.0480, 0.0764] | 0.2594 [0.2271, 0.2937] |
| M2b_s0 | flat1 | DEC-END | shrink | LATERAL (1) | rand | 0.7152 [0.6233, 0.8082] | 0.0707 [0.0569, 0.0854] | 0.3126 [0.2777, 0.3490] |
| M2b_s0 | flat1 | DEC-PATH | plain | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | flat1 | DEC-PATH | plain | LONGITUDINAL (1) | true | 0.9526 [0.8695, 1.0347] | 0.1199 [0.1060, 0.1350] | 0.8584 [0.7378, 0.9942] |
| M2b_s0 | flat1 | DEC-PATH | plain | LONGITUDINAL (1) | rand | 1.6832 [1.4918, 1.8797] | 0.2017 [0.1795, 0.2245] | 1.5840 [1.4009, 1.7821] |
| M2b_s0 | flat1 | DEC-PATH | plain | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | flat1 | DEC-PATH | plain | LATERAL (1) | true | 0.9526 [0.8695, 1.0347] | 0.1199 [0.1060, 0.1350] | 0.8584 [0.7378, 0.9942] |
| M2b_s0 | flat1 | DEC-PATH | plain | LATERAL (1) | rand | 1.6832 [1.4918, 1.8797] | 0.2017 [0.1795, 0.2245] | 1.5840 [1.4009, 1.7821] |
| M2b_s0 | flat1 | DEC-PATH | shrink | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | flat1 | DEC-PATH | shrink | LONGITUDINAL (1) | true | 0.7643 [0.6867, 0.8476] | 0.0910 [0.0776, 0.1057] | 0.5510 [0.4837, 0.6221] |
| M2b_s0 | flat1 | DEC-PATH | shrink | LONGITUDINAL (1) | rand | 1.3790 [1.2039, 1.5652] | 0.1736 [0.1527, 0.1954] | 1.2821 [1.0989, 1.4807] |
| M2b_s0 | flat1 | DEC-PATH | shrink | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | flat1 | DEC-PATH | shrink | LATERAL (1) | true | 0.7643 [0.6867, 0.8476] | 0.0910 [0.0776, 0.1057] | 0.5510 [0.4837, 0.6221] |
| M2b_s0 | flat1 | DEC-PATH | shrink | LATERAL (1) | rand | 1.3790 [1.2039, 1.5652] | 0.1736 [0.1527, 0.1954] | 1.2821 [1.0989, 1.4807] |
| M2b_s0 | td3 | DEC-END | shrink | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | td3 | DEC-END | shrink | LONGITUDINAL (1) | true | 0.7703 [0.6785, 0.8669] | 0.0701 [0.0560, 0.0867] | 0.3883 [0.3425, 0.4393] |
| M2b_s0 | td3 | DEC-END | shrink | LONGITUDINAL (1) | rand | 0.9576 [0.8289, 1.0988] | 0.0896 [0.0730, 0.1086] | 0.5799 [0.4778, 0.7210] |
| M2b_s0 | td3 | DEC-END | shrink | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | td3 | DEC-END | shrink | LATERAL (1) | true | 0.7703 [0.6785, 0.8669] | 0.0701 [0.0560, 0.0867] | 0.3883 [0.3425, 0.4393] |
| M2b_s0 | td3 | DEC-END | shrink | LATERAL (1) | rand | 0.9576 [0.8289, 1.0988] | 0.0896 [0.0730, 0.1086] | 0.5799 [0.4778, 0.7210] |
| M2b_s1 | flat1 | DEC-END | plain | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | flat1 | DEC-END | plain | LONGITUDINAL (1) | true | 0.6895 [0.6021, 0.7871] | 0.0640 [0.0511, 0.0787] | 0.2831 [0.2528, 0.3179] |
| M2b_s1 | flat1 | DEC-END | plain | LONGITUDINAL (1) | rand | 0.8697 [0.7437, 1.0107] | 0.0830 [0.0685, 0.0983] | 0.4460 [0.3776, 0.5300] |
| M2b_s1 | flat1 | DEC-END | plain | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | flat1 | DEC-END | plain | LATERAL (1) | true | 0.6895 [0.6021, 0.7871] | 0.0640 [0.0511, 0.0787] | 0.2831 [0.2528, 0.3179] |
| M2b_s1 | flat1 | DEC-END | plain | LATERAL (1) | rand | 0.8697 [0.7437, 1.0107] | 0.0830 [0.0685, 0.0983] | 0.4460 [0.3776, 0.5300] |
| M2b_s1 | flat1 | DEC-END | shrink | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | flat1 | DEC-END | shrink | LONGITUDINAL (1) | true | 0.6677 [0.5795, 0.7629] | 0.0599 [0.0465, 0.0746] | 0.2562 [0.2263, 0.2908] |
| M2b_s1 | flat1 | DEC-END | shrink | LONGITUDINAL (1) | rand | 0.7536 [0.6567, 0.8532] | 0.0697 [0.0557, 0.0849] | 0.3349 [0.2944, 0.3818] |
| M2b_s1 | flat1 | DEC-END | shrink | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | flat1 | DEC-END | shrink | LATERAL (1) | true | 0.6677 [0.5795, 0.7629] | 0.0599 [0.0465, 0.0746] | 0.2562 [0.2263, 0.2908] |
| M2b_s1 | flat1 | DEC-END | shrink | LATERAL (1) | rand | 0.7536 [0.6567, 0.8532] | 0.0697 [0.0557, 0.0849] | 0.3349 [0.2944, 0.3818] |
| M2b_s1 | flat1 | DEC-PATH | plain | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | flat1 | DEC-PATH | plain | LONGITUDINAL (1) | true | 1.0092 [0.9031, 1.1287] | 0.1220 [0.1075, 0.1377] | 0.8836 [0.7719, 1.0212] |
| M2b_s1 | flat1 | DEC-PATH | plain | LONGITUDINAL (1) | rand | 1.7892 [1.5660, 2.0319] | 0.2311 [0.2056, 0.2579] | 1.7791 [1.5362, 2.0430] |
| M2b_s1 | flat1 | DEC-PATH | plain | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | flat1 | DEC-PATH | plain | LATERAL (1) | true | 1.0092 [0.9031, 1.1287] | 0.1220 [0.1075, 0.1377] | 0.8836 [0.7719, 1.0212] |
| M2b_s1 | flat1 | DEC-PATH | plain | LATERAL (1) | rand | 1.7892 [1.5660, 2.0319] | 0.2311 [0.2056, 0.2579] | 1.7791 [1.5362, 2.0430] |
| M2b_s1 | flat1 | DEC-PATH | shrink | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | flat1 | DEC-PATH | shrink | LONGITUDINAL (1) | true | 0.8300 [0.7369, 0.9320] | 0.0926 [0.0797, 0.1071] | 0.5995 [0.5264, 0.6824] |
| M2b_s1 | flat1 | DEC-PATH | shrink | LONGITUDINAL (1) | rand | 1.5389 [1.3349, 1.7568] | 0.1947 [0.1708, 0.2189] | 1.5115 [1.2903, 1.7479] |
| M2b_s1 | flat1 | DEC-PATH | shrink | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | flat1 | DEC-PATH | shrink | LATERAL (1) | true | 0.8300 [0.7369, 0.9320] | 0.0926 [0.0797, 0.1071] | 0.5995 [0.5264, 0.6824] |
| M2b_s1 | flat1 | DEC-PATH | shrink | LATERAL (1) | rand | 1.5389 [1.3349, 1.7568] | 0.1947 [0.1708, 0.2189] | 1.5115 [1.2903, 1.7479] |
| M2b_s1 | td3 | DEC-END | shrink | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | td3 | DEC-END | shrink | LONGITUDINAL (1) | true | 0.7436 [0.6524, 0.8414] | 0.0681 [0.0528, 0.0851] | 0.3691 [0.3137, 0.4389] |
| M2b_s1 | td3 | DEC-END | shrink | LONGITUDINAL (1) | rand | 0.8426 [0.7350, 0.9581] | 0.0780 [0.0608, 0.0973] | 0.4594 [0.3791, 0.5564] |
| M2b_s1 | td3 | DEC-END | shrink | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | td3 | DEC-END | shrink | LATERAL (1) | true | 0.7436 [0.6524, 0.8414] | 0.0681 [0.0528, 0.0851] | 0.3691 [0.3137, 0.4389] |
| M2b_s1 | td3 | DEC-END | shrink | LATERAL (1) | rand | 0.8426 [0.7350, 0.9581] | 0.0780 [0.0608, 0.0973] | 0.4594 [0.3791, 0.5564] |

## Beside, no bar: cost monotonicity (Tab. 10 statistic: -Spearman(distance, time); 1.0 = perfectly decreasing) along the chosen G-TRUE plan, level 1: decoded distance to the END goal, and latent distance to the END goal; fraction of windows where planning lowered the cost below the best initial candidate (the cost includes REG when lambda > 0)
| run | decoded monotonicity | latent monotonicity | frac lowered |
|---|---|---|---|
| M2b_s0|flat1|end|plain|LONGITUDINAL (lambda 1) | -0.2318 [-0.3202, -0.1391] | -0.4603 [-0.5338, -0.3850] | 1.0000 [1.0000, 1.0000] |
| M2b_s0|flat1|end|plain|LATERAL (lambda 1) | -0.2318 [-0.3202, -0.1391] | -0.4603 [-0.5338, -0.3850] | 1.0000 [1.0000, 1.0000] |
| M2b_s0|flat1|end|shrink|LONGITUDINAL (lambda 1) | -0.1167 [-0.2038, -0.0293] | -0.0192 [-0.1061, 0.0689] | 0.9990 [0.9970, 1.0000] |
| M2b_s0|flat1|end|shrink|LATERAL (lambda 1) | -0.1167 [-0.2038, -0.0293] | -0.0192 [-0.1061, 0.0689] | 0.9990 [0.9970, 1.0000] |
| M2b_s0|flat1|path|plain|LONGITUDINAL (lambda 1) | -0.0924 [-0.1842, -0.0002] | -0.4447 [-0.5182, -0.3721] | 1.0000 [1.0000, 1.0000] |
| M2b_s0|flat1|path|plain|LATERAL (lambda 1) | -0.0924 [-0.1842, -0.0002] | -0.4447 [-0.5182, -0.3721] | 1.0000 [1.0000, 1.0000] |
| M2b_s0|flat1|path|shrink|LONGITUDINAL (lambda 1) | -0.0602 [-0.1460, 0.0279] | -0.0019 [-0.0896, 0.0844] | 1.0000 [1.0000, 1.0000] |
| M2b_s0|flat1|path|shrink|LATERAL (lambda 1) | -0.0602 [-0.1460, 0.0279] | -0.0019 [-0.0896, 0.0844] | 1.0000 [1.0000, 1.0000] |
| M2b_s0|td3|end|shrink|LONGITUDINAL (lambda 1) | -0.1266 [-0.2153, -0.0369] | 0.1062 [0.0163, 0.1970] | 0.9990 [0.9970, 1.0000] |
| M2b_s0|td3|end|shrink|LATERAL (lambda 1) | -0.1266 [-0.2153, -0.0369] | 0.1062 [0.0163, 0.1970] | 0.9990 [0.9970, 1.0000] |
| M2b_s1|flat1|end|plain|LONGITUDINAL (lambda 1) | -0.1037 [-0.1871, -0.0211] | -0.4489 [-0.5256, -0.3621] | 0.9990 [0.9970, 1.0000] |
| M2b_s1|flat1|end|plain|LATERAL (lambda 1) | -0.1037 [-0.1871, -0.0211] | -0.4489 [-0.5256, -0.3621] | 0.9990 [0.9970, 1.0000] |
| M2b_s1|flat1|end|shrink|LONGITUDINAL (lambda 1) | -0.0694 [-0.1484, 0.0074] | -0.0740 [-0.1638, 0.0239] | 1.0000 [1.0000, 1.0000] |
| M2b_s1|flat1|end|shrink|LATERAL (lambda 1) | -0.0694 [-0.1484, 0.0074] | -0.0740 [-0.1638, 0.0239] | 1.0000 [1.0000, 1.0000] |
| M2b_s1|flat1|path|plain|LONGITUDINAL (lambda 1) | 0.0454 [-0.0416, 0.1262] | -0.4262 [-0.5076, -0.3373] | 1.0000 [1.0000, 1.0000] |
| M2b_s1|flat1|path|plain|LATERAL (lambda 1) | 0.0454 [-0.0416, 0.1262] | -0.4262 [-0.5076, -0.3373] | 1.0000 [1.0000, 1.0000] |
| M2b_s1|flat1|path|shrink|LONGITUDINAL (lambda 1) | 0.0112 [-0.0700, 0.0871] | -0.0582 [-0.1505, 0.0417] | 0.9990 [0.9970, 1.0000] |
| M2b_s1|flat1|path|shrink|LATERAL (lambda 1) | 0.0112 [-0.0700, 0.0871] | -0.0582 [-0.1505, 0.0417] | 0.9990 [0.9970, 1.0000] |
| M2b_s1|td3|end|shrink|LONGITUDINAL (lambda 1) | -0.0819 [-0.1607, -0.0064] | 0.0342 [-0.0543, 0.1258] | 1.0000 [1.0000, 1.0000] |
| M2b_s1|td3|end|shrink|LATERAL (lambda 1) | -0.0819 [-0.1607, -0.0064] | 0.0342 [-0.0543, 0.1258] | 1.0000 [1.0000, 1.0000] |

## Beside: planner-seed replicate (planner seed 1 minus seed 0, G-TRUE, same checkpoint and lambda, SCORE, EC-BOOT; a difference of this size is inference noise, not an effect)
* M2b_s0|flat1|end|plain|LONGITUDINAL (lambda 1): 0.0000 [-0.0000, 0.0000] (none)
* M2b_s0|flat1|end|plain|LATERAL (lambda 1): -0.0000 [-0.0000, 0.0000] (none)
* M2b_s0|flat1|end|shrink|LONGITUDINAL (lambda 1): -0.0000 [-0.0000, 0.0000] (none)
* M2b_s0|flat1|end|shrink|LATERAL (lambda 1): -0.0000 [-0.0000, 0.0000] (none)
* M2b_s0|flat1|path|plain|LONGITUDINAL (lambda 1): -0.0000 [-0.0003, 0.0003] (none)
* M2b_s0|flat1|path|plain|LATERAL (lambda 1): -0.0000 [-0.0001, 0.0000] (none)
* M2b_s0|flat1|path|shrink|LONGITUDINAL (lambda 1): 0.0002 [-0.0001, 0.0005] (none)
* M2b_s0|flat1|path|shrink|LATERAL (lambda 1): -0.0000 [-0.0000, 0.0000] (none)
* M2b_s0|td3|end|shrink|LONGITUDINAL (lambda 1): -0.0003 [-0.0025, 0.0018] (none)
* M2b_s0|td3|end|shrink|LATERAL (lambda 1): 0.0002 [0.0000, 0.0004] (worse)
* M2b_s1|flat1|end|plain|LONGITUDINAL (lambda 1): -0.0000 [-0.0001, 0.0000] (none)
* M2b_s1|flat1|end|plain|LATERAL (lambda 1): -0.0000 [-0.0000, 0.0000] (none)
* M2b_s1|flat1|end|shrink|LONGITUDINAL (lambda 1): 0.0000 [-0.0000, 0.0001] (none)
* M2b_s1|flat1|end|shrink|LATERAL (lambda 1): 0.0000 [-0.0000, 0.0000] (none)
* M2b_s1|flat1|path|plain|LONGITUDINAL (lambda 1): 0.0001 [-0.0002, 0.0005] (none)
* M2b_s1|flat1|path|plain|LATERAL (lambda 1): -0.0000 [-0.0000, 0.0000] (none)
* M2b_s1|flat1|path|shrink|LONGITUDINAL (lambda 1): -0.0000 [-0.0003, 0.0001] (none)
* M2b_s1|flat1|path|shrink|LATERAL (lambda 1): -0.0000 [-0.0000, 0.0000] (none)
* M2b_s1|td3|end|shrink|LONGITUDINAL (lambda 1): -0.0013 [-0.0035, 0.0009] (none)
* M2b_s1|td3|end|shrink|LATERAL (lambda 1): 0.0001 [-0.0002, 0.0003] (none)

## Beside: the TUNE grid (FLAT1, G-TRUE, planner seed 0, mean MAE over the TUNE windows; lambda is chosen as the row minimum per family; this is the selection record, not a result) and the TD3 form pick
| model | arm | form | lambda | LONGITUDINAL acc MAE [m/s^2] | LATERAL yaw MAE [rad/s] |
|---|---|---|---|---|---|
| M2b_s0 | DEC-END | plain | 0 | 1.4647 | 0.1665 |
| M2b_s0 | DEC-END | plain | 0.1 | 0.6831 | 0.0791 |
| M2b_s0 | DEC-END | plain | 1 | 0.3419 (chosen) | 0.0259 (chosen) |
| M2b_s0 | DEC-END | shrink | 0 | 1.4912 | 0.1679 |
| M2b_s0 | DEC-END | shrink | 0.1 | 0.4905 | 0.0588 |
| M2b_s0 | DEC-END | shrink | 1 | 0.3265 (chosen) | 0.0227 (chosen) |
| M2b_s0 | DEC-PATH | plain | 0 | 1.5103 | 0.1675 |
| M2b_s0 | DEC-PATH | plain | 0.1 | 1.1598 | 0.1360 |
| M2b_s0 | DEC-PATH | plain | 1 | 0.5369 (chosen) | 0.0598 (chosen) |
| M2b_s0 | DEC-PATH | shrink | 0 | 1.5240 | 0.1685 |
| M2b_s0 | DEC-PATH | shrink | 0.1 | 1.0251 | 0.1228 |
| M2b_s0 | DEC-PATH | shrink | 1 | 0.4132 (chosen) | 0.0427 (chosen) |
| M2b_s1 | DEC-END | plain | 0 | 1.4204 | 0.1594 |
| M2b_s1 | DEC-END | plain | 0.1 | 0.6783 | 0.0745 |
| M2b_s1 | DEC-END | plain | 1 | 0.3468 (chosen) | 0.0244 (chosen) |
| M2b_s1 | DEC-END | shrink | 0 | 1.4245 | 0.1640 |
| M2b_s1 | DEC-END | shrink | 0.1 | 0.5106 | 0.0537 |
| M2b_s1 | DEC-END | shrink | 1 | 0.3256 (chosen) | 0.0219 (chosen) |
| M2b_s1 | DEC-PATH | plain | 0 | 1.4452 | 0.1614 |
| M2b_s1 | DEC-PATH | plain | 0.1 | 1.1138 | 0.1274 |
| M2b_s1 | DEC-PATH | plain | 1 | 0.5295 (chosen) | 0.0525 (chosen) |
| M2b_s1 | DEC-PATH | shrink | 0 | 1.4386 | 0.1673 |
| M2b_s1 | DEC-PATH | shrink | 0.1 | 1.0009 | 0.1202 |
| M2b_s1 | DEC-PATH | shrink | 1 | 0.4194 (chosen) | 0.0373 (chosen) |
TD3 form and lambda (the FLAT1 DEC-END choice per training seed and family): M2b_s0|LONGITUDINAL: shrink / 1; M2b_s0|LATERAL: shrink / 1; M2b_s1|LONGITUDINAL: shrink / 1; M2b_s1|LATERAL: shrink / 1

Fraction of SCORE windows whose HOLD lies outside the level-1 box (HOLD_c differs from HOLD): M2b_s0 0.090, M2b_s1 0.090

## What each outcome triggers (committed in the card, Rule Zero; evaluated mechanically on the table above, the Master Mind applies it)
* NO-GAIN on every decoded arm and (KIN-CEILING does not beat HOLD in any family, or DECODER-WEAK): the decoded quantities are the limit; next is a nonlinear readout and multi-frame decoding on the ladder latents.
* Every outcome: the harness is read on the ladder world models when they finish; PLANTEST and PLANTEST2 stay required gates.
