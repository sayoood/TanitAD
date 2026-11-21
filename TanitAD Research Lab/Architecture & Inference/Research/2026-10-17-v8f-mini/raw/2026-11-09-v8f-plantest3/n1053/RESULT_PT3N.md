# RESULT_PT3N: PLANTEST3's FLAT1 DEC-END with the decoder fitted on N1053
Card `../../PREREG_V8F_PT3N.md`. Tier **T0**, an oracle-goal diagnostic, never a driving result. TD3 and DEC-PATH are not run, so their lines below read INCOMPLETE or NO by construction. MEASURED: `plantest3/n1053/runs/`.

## Validity first (the read is INVALID if one control fails)
* **Verdict on validity: all controls pass.**
* Control 1, tie to DECN (M2b_s0, speed): N1053 D_1 +0.67955, DECN ridge m=1 +0.67955, difference 0.00000 (bar 0.01): pass.
* Control 1, tie to DECN (M2b_s0, yaw): N1053 D_1 -0.04298, DECN ridge m=1 -0.04298, difference 0.00000 (bar 0.01): pass.
* Control 2, N53 reference (M2b_s0, speed): +0.27074 against PLANTEST3 +0.27074, difference 0.000000 (bar 0.001): pass.
* Control 2, N53 reference (M2b_s0, yaw): -0.12238 against PLANTEST3 -0.12238, difference 0.000000 (bar 0.001): pass.
* Control 1, tie to DECN (M2b_s1, speed): N1053 D_1 +0.64713, DECN ridge m=1 +0.64713, difference 0.00000 (bar 0.01): pass.
* Control 1, tie to DECN (M2b_s1, yaw): N1053 D_1 +0.10532, DECN ridge m=1 +0.10532, difference 0.00000 (bar 0.01): pass.
* Control 2, N53 reference (M2b_s1, speed): +0.31447 against PLANTEST3 +0.31447, difference 0.000000 (bar 0.001): pass.
* Control 2, N53 reference (M2b_s1, yaw): +0.26797 against PLANTEST3 +0.26797, difference 0.000000 (bar 0.001): pass.
* Control 3, harness reproduction (M2b_s0 flat1 end plain true L1 p0): identical True, largest absolute difference 0.000e+00: pass.
* Control 4, split: asserted in `run_pt3n.py` `split_guard` on real ids before every fit; `guard_mutation` requires three planted violations to raise, run at start. Pass (the run reached planning).
* Control 5, clips: the 1,000 shards are DECN's, md5-verified at pull (DECN control 6); count and order asserted in `clip_order`. Pass.
* Control 6, isolation: PLANTEST3's `tune_choice.json` sha256 unchanged True; run errors 0: pass.

## PT3N minus PLANTEST3, G-TRUE, same SCORE windows (paired EC-BOOT, B 2,000; beside, no bar; negative is lower error)

| seed | form | family | lambda (N1053 / N53-to-70 fit) | difference [95 %] | separated |
|---|---|---|---|---|---|
| M2b_s0 | plain | LONGITUDINAL | 1 / 1 | -0.0104 [-0.0156, -0.0054] m/s^2 | better |
| M2b_s0 | plain | LATERAL | 1 / 1 | -0.0023 [-0.0035, -0.0013] rad/s | better |
| M2b_s0 | shrink | LONGITUDINAL | 1 / 1 | -0.0028 [-0.0054, -0.0002] m/s^2 | better |
| M2b_s0 | shrink | LATERAL | 1 / 1 | -0.0008 [-0.0013, -0.0003] rad/s | better |
| M2b_s1 | plain | LONGITUDINAL | 1 / 1 | -0.0124 [-0.0191, -0.0066] m/s^2 | better |
| M2b_s1 | plain | LATERAL | 1 / 1 | -0.0018 [-0.0027, -0.0009] rad/s | better |
| M2b_s1 | shrink | LONGITUDINAL | 1 / 1 | -0.0046 [-0.0076, -0.0018] m/s^2 | better |
| M2b_s1 | shrink | LATERAL | 1 / 1 | -0.0007 [-0.0011, -0.0002] rad/s | better |

Card `../PREREG_V8F_PLANTEST3.md` (prefix sha256 `1f9609b4...b348f9`), IMPL `IMPL_CHOICES_PLANTEST3.md` (sha256 in `IMPL_CHOICES_PLANTEST3.sha256`). **Tier T0: a world-model diagnostic with an oracle (true-future) goal, never a driving result.** Evidence class: MEASURED (`runs/*.npz`, `RESULT_PLANTEST3.json`).
Estimator on every interval: EC-BOOT = paired episode-cluster bootstrap over the 71 SCORE episodes (994 windows), B 2,000, seed 0, 95 % percentile. Separated-better = interval wholly below 0 (MAE, lower is better). Decoders and lambda were chosen on the TUNE half only (hash split, salt `plantest2-v1`); every number below is SCORE. The interval answers 'another draw of episodes'; it is blind to training-seed and planner-seed variance, hence both training seeds are required and the planner-seed replicate is printed beside.
A read is VALID only if the six controls of `RESULT_PLANTEST3_controls.md` passed; this report does not check that file.

## Decoder gate (before planning; TUNE inner split, held-out episodes)
**DECODER-WEAK: YES** (M2b_s0, M2b_s1); bars: held-out R2 on encoded latents below 0.70 (speed) or 0.50 (yaw rate) at either seed. Missing gate records: none.
| model | read | R2 speed | R2 yaw rate | n speed / n yaw |
|---|---|---|---|---|
| M2b_s0 | D_1 encoded, all knots (the DECODER-WEAK quantity) | 0.6796 | -0.0430 | 3808 / 3808 |
| M2b_s0 | D_1 encoded, k = 12 | 0.6686 | -0.0825 | 238 / 238 |
| M2b_s0 | D_1 on the rollout at k = 12 under the logged actions | 0.6106 | -0.1204 | 238 / 238 |
| M2b_s0 | D_2 encoded, all knots | 0.6338 | 0.1917 | 3808 / 3808 |
| M2b_s1 | D_1 encoded, all knots (the DECODER-WEAK quantity) | 0.6471 | 0.1053 | 3808 / 3808 |
| M2b_s1 | D_1 encoded, k = 12 | 0.6405 | 0.0768 | 238 / 238 |
| M2b_s1 | D_1 on the rollout at k = 12 under the logged actions | 0.5712 | 0.0341 | 238 / 238 |
| M2b_s1 | D_2 encoded, all knots | 0.6086 | 0.4439 | 3808 / 3808 |

## Verdict (the card's committed table; per arm, planner and family; both training seeds required)
| planner | arm | form | family | chosen lambda s0 / s1 | M2b_s0 | M2b_s1 | decision label |
|---|---|---|---|---|---|---|---|
| FLAT1 | DEC-END | plain | LONGITUDINAL | 1 / 1 | NO-GAIN (collapse) | NO-GAIN (collapse) | **NO-GAIN +HOLD-COLLAPSE(M2b_s0,M2b_s1)** |
| FLAT1 | DEC-END | plain | LATERAL | 1 / 1 | NO-GAIN | NO-GAIN | **NO-GAIN** |
| FLAT1 | DEC-END | shrink | LONGITUDINAL | 1 / 1 | PLANNABLE (collapse) | NO-GAIN (collapse) | **SEEDS-DISAGREE(PLANNABLE|NO-GAIN) +HOLD-COLLAPSE(M2b_s0,M2b_s1)** |
| FLAT1 | DEC-END | shrink | LATERAL | 1 / 1 | NO-GAIN (collapse) | NO-GAIN (collapse) | **NO-GAIN +HOLD-COLLAPSE(M2b_s0,M2b_s1)** |
TACTICAL: not computed (n = 0: this read has no decision output). STRATEGIC: PAUSED (PI item 25, `PI_DECISION_QUEUE.md`).

HIER-ADDS (TD3 DEC-END minus FLAT1 DEC-END of the same form and lambda, G-TRUE, separated-better at both seeds): LONGITUDINAL: NO; LATERAL: NO

## KIN-CEILING (no world model, no planner): actions from the decoded true-future speed differences and yaw rates, MAE against the 12 logged actions, SCORE
| model | family | unit | HOLD | KIN | KIN-RAND (donor's decoded future) | KIN - HOLD | sep | KIN - KIN-RAND | sep |
|---|---|---|---|---|---|---|---|---|---|
| M2b_s0 | LONGITUDINAL | m/s^2 | 0.3348 [0.2880, 0.3826] | 3.4086 [3.1987, 3.6267] | 3.5453 [3.3618, 3.7393] | 3.0738 [2.8479, 3.2981] | worse | -0.1367 [-0.4135, 0.1534] | none |
| M2b_s0 | LATERAL | rad/s | 0.0256 [0.0197, 0.0320] | 0.0501 [0.0415, 0.0595] | 0.0565 [0.0469, 0.0671] | 0.0245 [0.0198, 0.0293] | worse | -0.0064 [-0.0104, -0.0022] | better |
| M2b_s1 | LONGITUDINAL | m/s^2 | 0.3348 [0.2880, 0.3826] | 3.3879 [3.1732, 3.6216] | 3.4984 [3.3003, 3.7116] | 3.0530 [2.8160, 3.2904] | worse | -0.1105 [-0.3822, 0.1917] | none |
| M2b_s1 | LATERAL | rad/s | 0.0256 [0.0197, 0.0320] | 0.0497 [0.0413, 0.0589] | 0.0561 [0.0463, 0.0668] | 0.0241 [0.0195, 0.0289] | worse | -0.0064 [-0.0109, -0.0021] | better |
KIN beats HOLD (separated-better at both seeds): LONGITUDINAL: NO; LATERAL: NO

## Deciding rows: MAE against the 12 logged actions, SCORE (EC-BOOT mean [95 % interval])
### LONGITUDINAL (acc_mae, m/s^2)
| model | planner | arm | form | lambda | HOLD | G-TRUE | G-RAND | TRUE - HOLD | sep | TRUE - RAND | sep | distance from HOLD, G-TRUE (G-RAND) [prior-std units] | base label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M2b_s0 | flat1 | DEC-END | plain | 1 | 0.3348 [0.2880, 0.3826] | 0.3337 [0.2903, 0.3790] | 0.3762 [0.3290, 0.4240] | -0.0011 [-0.0069, 0.0049] | none | -0.0424 [-0.0649, -0.0245] | better | 0.071 (0.177) COLLAPSE | NO-GAIN |
| M2b_s0 | flat1 | DEC-END | shrink | 1 | 0.3348 [0.2880, 0.3826] | 0.3308 [0.2860, 0.3765] | 0.3496 [0.3043, 0.3952] | -0.0040 [-0.0082, -0.0001] | better | -0.0188 [-0.0291, -0.0104] | better | 0.048 (0.109) COLLAPSE | PLANNABLE |
| M2b_s1 | flat1 | DEC-END | plain | 1 | 0.3348 [0.2880, 0.3826] | 0.3396 [0.2964, 0.3847] | 0.3872 [0.3393, 0.4353] | 0.0048 [-0.0022, 0.0120] | none | -0.0476 [-0.0708, -0.0275] | better | 0.090 (0.202) COLLAPSE | NO-GAIN |
| M2b_s1 | flat1 | DEC-END | shrink | 1 | 0.3348 [0.2880, 0.3826] | 0.3319 [0.2884, 0.3771] | 0.3572 [0.3118, 0.4023] | -0.0029 [-0.0073, 0.0014] | none | -0.0253 [-0.0375, -0.0149] | better | 0.056 (0.132) COLLAPSE | NO-GAIN |

### LATERAL (yaw_mae, rad/s)
| model | planner | arm | form | lambda | HOLD | G-TRUE | G-RAND | TRUE - HOLD | sep | TRUE - RAND | sep | distance from HOLD, G-TRUE (G-RAND) [prior-std units] | base label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M2b_s0 | flat1 | DEC-END | plain | 1 | 0.0256 [0.0197, 0.0320] | 0.0285 [0.0225, 0.0352] | 0.0319 [0.0259, 0.0386] | 0.0029 [0.0017, 0.0043] | worse | -0.0034 [-0.0048, -0.0021] | better | 0.116 (0.168) | NO-GAIN |
| M2b_s0 | flat1 | DEC-END | shrink | 1 | 0.0256 [0.0197, 0.0320] | 0.0274 [0.0214, 0.0342] | 0.0293 [0.0232, 0.0361] | 0.0018 [0.0009, 0.0032] | worse | -0.0019 [-0.0027, -0.0012] | better | 0.091 (0.125) COLLAPSE | NO-GAIN |
| M2b_s1 | flat1 | DEC-END | plain | 1 | 0.0256 [0.0197, 0.0320] | 0.0280 [0.0219, 0.0347] | 0.0321 [0.0256, 0.0392] | 0.0024 [0.0013, 0.0038] | worse | -0.0041 [-0.0064, -0.0024] | better | 0.107 (0.165) | NO-GAIN |
| M2b_s1 | flat1 | DEC-END | shrink | 1 | 0.0256 [0.0197, 0.0320] | 0.0270 [0.0209, 0.0338] | 0.0291 [0.0228, 0.0360] | 0.0015 [0.0005, 0.0028] | worse | -0.0020 [-0.0030, -0.0012] | better | 0.082 (0.117) COLLAPSE | NO-GAIN |

HOLD-COLLAPSE flag: mean absolute distance of the G-TRUE plan from HOLD below 0.10 of the TRAIN prior std of the channel; a flagged row does not count as planning.

## HIER-ADDS rows: TD3 minus FLAT1 under G-TRUE, DEC-END (negative = TD3 better), SCORE
| family | model | form / lambda | TD3 - FLAT1 | sep |
|---|---|---|---|---|

## Beside, no bar: speed error, heading error and ADE at 2.4 s of the unicycle integration, plan vs logged actions (SCORE; at the family's chosen lambda; EC-BOOT mean [interval])
| model | planner | arm | form | family (lambda) | ref | speed err [m/s] | heading err [rad] | ADE [m] |
|---|---|---|---|---|---|---|---|---|
| M2b_s0 | flat1 | DEC-END | plain | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | flat1 | DEC-END | plain | LONGITUDINAL (1) | true | 0.6621 [0.5763, 0.7543] | 0.0616 [0.0483, 0.0762] | 0.2571 [0.2264, 0.2911] |
| M2b_s0 | flat1 | DEC-END | plain | LONGITUDINAL (1) | rand | 0.7567 [0.6616, 0.8521] | 0.0689 [0.0554, 0.0838] | 0.3337 [0.2950, 0.3748] |
| M2b_s0 | flat1 | DEC-END | plain | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | flat1 | DEC-END | plain | LATERAL (1) | true | 0.6621 [0.5763, 0.7543] | 0.0616 [0.0483, 0.0762] | 0.2571 [0.2264, 0.2911] |
| M2b_s0 | flat1 | DEC-END | plain | LATERAL (1) | rand | 0.7567 [0.6616, 0.8521] | 0.0689 [0.0554, 0.0838] | 0.3337 [0.2950, 0.3748] |
| M2b_s0 | flat1 | DEC-END | shrink | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | flat1 | DEC-END | shrink | LONGITUDINAL (1) | true | 0.6620 [0.5734, 0.7572] | 0.0595 [0.0460, 0.0744] | 0.2490 [0.2179, 0.2834] |
| M2b_s0 | flat1 | DEC-END | shrink | LONGITUDINAL (1) | rand | 0.7019 [0.6127, 0.7921] | 0.0636 [0.0501, 0.0786] | 0.2864 [0.2543, 0.3200] |
| M2b_s0 | flat1 | DEC-END | shrink | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | flat1 | DEC-END | shrink | LATERAL (1) | true | 0.6620 [0.5734, 0.7572] | 0.0595 [0.0460, 0.0744] | 0.2490 [0.2179, 0.2834] |
| M2b_s0 | flat1 | DEC-END | shrink | LATERAL (1) | rand | 0.7019 [0.6127, 0.7921] | 0.0636 [0.0501, 0.0786] | 0.2864 [0.2543, 0.3200] |
| M2b_s1 | flat1 | DEC-END | plain | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | flat1 | DEC-END | plain | LONGITUDINAL (1) | true | 0.6701 [0.5836, 0.7639] | 0.0604 [0.0471, 0.0751] | 0.2630 [0.2298, 0.2980] |
| M2b_s1 | flat1 | DEC-END | plain | LONGITUDINAL (1) | rand | 0.7812 [0.6829, 0.8808] | 0.0691 [0.0547, 0.0852] | 0.3629 [0.3081, 0.4381] |
| M2b_s1 | flat1 | DEC-END | plain | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | flat1 | DEC-END | plain | LATERAL (1) | true | 0.6701 [0.5836, 0.7639] | 0.0604 [0.0471, 0.0751] | 0.2630 [0.2298, 0.2980] |
| M2b_s1 | flat1 | DEC-END | plain | LATERAL (1) | rand | 0.7812 [0.6829, 0.8808] | 0.0691 [0.0547, 0.0852] | 0.3629 [0.3081, 0.4381] |
| M2b_s1 | flat1 | DEC-END | shrink | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | flat1 | DEC-END | shrink | LONGITUDINAL (1) | true | 0.6607 [0.5725, 0.7558] | 0.0585 [0.0450, 0.0737] | 0.2470 [0.2166, 0.2812] |
| M2b_s1 | flat1 | DEC-END | shrink | LONGITUDINAL (1) | rand | 0.7168 [0.6291, 0.8069] | 0.0633 [0.0492, 0.0789] | 0.3029 [0.2675, 0.3435] |
| M2b_s1 | flat1 | DEC-END | shrink | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | flat1 | DEC-END | shrink | LATERAL (1) | true | 0.6607 [0.5725, 0.7558] | 0.0585 [0.0450, 0.0737] | 0.2470 [0.2166, 0.2812] |
| M2b_s1 | flat1 | DEC-END | shrink | LATERAL (1) | rand | 0.7168 [0.6291, 0.8069] | 0.0633 [0.0492, 0.0789] | 0.3029 [0.2675, 0.3435] |

## Beside, no bar: cost monotonicity (Tab. 10 statistic: -Spearman(distance, time); 1.0 = perfectly decreasing) along the chosen G-TRUE plan, level 1: decoded distance to the END goal, and latent distance to the END goal; fraction of windows where planning lowered the cost below the best initial candidate (the cost includes REG when lambda > 0)
| run | decoded monotonicity | latent monotonicity | frac lowered |
|---|---|---|---|
| M2b_s0|flat1|end|plain|LONGITUDINAL (lambda 1) | -0.1309 [-0.2195, -0.0373] | -0.4581 [-0.5321, -0.3834] | 1.0000 [1.0000, 1.0000] |
| M2b_s0|flat1|end|plain|LATERAL (lambda 1) | -0.1309 [-0.2195, -0.0373] | -0.4581 [-0.5321, -0.3834] | 1.0000 [1.0000, 1.0000] |
| M2b_s0|flat1|end|shrink|LONGITUDINAL (lambda 1) | -0.0489 [-0.1245, 0.0307] | -0.0198 [-0.1074, 0.0699] | 1.0000 [1.0000, 1.0000] |
| M2b_s0|flat1|end|shrink|LATERAL (lambda 1) | -0.0489 [-0.1245, 0.0307] | -0.0198 [-0.1074, 0.0699] | 1.0000 [1.0000, 1.0000] |
| M2b_s1|flat1|end|plain|LONGITUDINAL (lambda 1) | -0.1883 [-0.2946, -0.0811] | -0.4516 [-0.5287, -0.3637] | 0.9990 [0.9970, 1.0000] |
| M2b_s1|flat1|end|plain|LATERAL (lambda 1) | -0.1883 [-0.2946, -0.0811] | -0.4516 [-0.5287, -0.3637] | 0.9990 [0.9970, 1.0000] |
| M2b_s1|flat1|end|shrink|LONGITUDINAL (lambda 1) | -0.0746 [-0.1668, 0.0237] | -0.0730 [-0.1620, 0.0248] | 1.0000 [1.0000, 1.0000] |
| M2b_s1|flat1|end|shrink|LATERAL (lambda 1) | -0.0746 [-0.1668, 0.0237] | -0.0730 [-0.1620, 0.0248] | 1.0000 [1.0000, 1.0000] |

## Beside: planner-seed replicate (planner seed 1 minus seed 0, G-TRUE, same checkpoint and lambda, SCORE, EC-BOOT; a difference of this size is inference noise, not an effect)
* M2b_s0|flat1|end|plain|LONGITUDINAL (lambda 1): -0.0000 [-0.0000, 0.0000] (none)
* M2b_s0|flat1|end|plain|LATERAL (lambda 1): 0.0000 [-0.0000, 0.0000] (none)
* M2b_s0|flat1|end|shrink|LONGITUDINAL (lambda 1): -0.0000 [-0.0001, 0.0000] (none)
* M2b_s0|flat1|end|shrink|LATERAL (lambda 1): 0.0000 [-0.0000, 0.0000] (none)
* M2b_s1|flat1|end|plain|LONGITUDINAL (lambda 1): 0.0000 [-0.0000, 0.0000] (none)
* M2b_s1|flat1|end|plain|LATERAL (lambda 1): -0.0000 [-0.0000, 0.0000] (none)
* M2b_s1|flat1|end|shrink|LONGITUDINAL (lambda 1): -0.0000 [-0.0000, 0.0000] (none)
* M2b_s1|flat1|end|shrink|LATERAL (lambda 1): 0.0000 [-0.0000, 0.0000] (none)

## Beside: the TUNE grid (FLAT1, G-TRUE, planner seed 0, mean MAE over the TUNE windows; lambda is chosen as the row minimum per family; this is the selection record, not a result) and the TD3 form pick
| model | arm | form | lambda | LONGITUDINAL acc MAE [m/s^2] | LATERAL yaw MAE [rad/s] |
|---|---|---|---|---|---|
| M2b_s0 | DEC-END | plain | 0 | 1.3546 | 0.1506 |
| M2b_s0 | DEC-END | plain | 0.1 | 0.4636 | 0.0501 |
| M2b_s0 | DEC-END | plain | 1 | 0.3226 (chosen) | 0.0224 (chosen) |
| M2b_s0 | DEC-END | shrink | 0 | 1.3773 | 0.1610 |
| M2b_s0 | DEC-END | shrink | 0.1 | 0.3752 | 0.0406 |
| M2b_s0 | DEC-END | shrink | 1 | 0.3205 (chosen) | 0.0213 (chosen) |
| M2b_s1 | DEC-END | plain | 0 | 1.3118 | 0.1467 |
| M2b_s1 | DEC-END | plain | 0.1 | 0.5056 | 0.0437 |
| M2b_s1 | DEC-END | plain | 1 | 0.3258 (chosen) | 0.0215 (chosen) |
| M2b_s1 | DEC-END | shrink | 0 | 1.3114 | 0.1529 |
| M2b_s1 | DEC-END | shrink | 0.1 | 0.4032 | 0.0330 |
| M2b_s1 | DEC-END | shrink | 1 | 0.3213 (chosen) | 0.0207 (chosen) |
TD3 form and lambda (the FLAT1 DEC-END choice per training seed and family): M2b_s0|LONGITUDINAL: shrink / 1; M2b_s0|LATERAL: shrink / 1; M2b_s1|LONGITUDINAL: shrink / 1; M2b_s1|LATERAL: shrink / 1

Fraction of SCORE windows whose HOLD lies outside the level-1 box (HOLD_c differs from HOLD): M2b_s0 0.090, M2b_s1 0.090

## What each outcome triggers (committed in the card, Rule Zero; evaluated mechanically on the table above, the Master Mind applies it)
* Every outcome: the harness is read on the ladder world models when they finish; PLANTEST and PLANTEST2 stay required gates.
