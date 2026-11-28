# RESULT_PT3M: FLAT1 DEC-END with the four-frame per-target reader in the cost, fitted on N1053
Card `../../PREREG_V8F_PT3M.md`. Tier **T0**, an oracle-goal diagnostic, never a driving result. TD3 and DEC-PATH are not run, so their lines below read INCOMPLETE or NO by construction. In the gate table D_1 is this reader on four frames; D_2 is not fitted (nan). In the KIN-CEILING table knots 3 to 11 come from the reader and knots 0 to 2 stay at HOLD. The triggers are in the card; the Master Mind applies them. MEASURED: `plantest3/mlp4/runs/`.

## Validity first (the read is INVALID if one control fails)
* **Verdict on validity: all controls pass.**
* Control 1, tie to DECN (M2b_s0, speed_ridge_m4): +0.69457 against +0.69457, difference 0.00000 (bar 0.01): pass.
* Control 1, tie to DECN (M2b_s0, yaw_mlp_m4_lr1e-3): +0.56960 against +0.56960, difference 0.00000 (bar 0.01): pass.
* Control 2, reachability (M2b_s0): mean absolute gradient per channel 1.379e-02, 1.403e-02; under SHRINK(0) largest absolute gradient 0.0e+00: pass.
* Control 3, harness tie (M2b_s0): largest relative cost difference 1.71e-14 (bar 1e-4), gradient-norm difference 0.00e+00 (bar 1e-3): pass.
* Control 1, tie to DECN (M2b_s1, speed_ridge_m4): +0.68567 against +0.68567, difference 0.00000 (bar 0.01): pass.
* Control 1, tie to DECN (M2b_s1, yaw_mlp_m4_lr1e-3): +0.44091 against +0.44091, difference 0.00000 (bar 0.01): pass.
* Control 2, reachability (M2b_s1): mean absolute gradient per channel 1.801e-02, 2.050e-02; under SHRINK(0) largest absolute gradient 0.0e+00: pass.
* Control 3, harness tie (M2b_s1): largest relative cost difference 2.66e-14 (bar 1e-4), gradient-norm difference 0.00e+00 (bar 1e-3): pass.
* Control 4, split: DECN's `split_guard_n` on real ids before each fit; `guard_mutation` requires four planted violations to raise, run at start. Pass (the run reached planning).
* Control 5, clips: the 1,000 shards are DECN's; count and order asserted in `clip_order`. Pass.
* Control 6, isolation: PLANTEST3's and PT3N's `tune_choice.json` sha256 unchanged True; run errors 0: pass.

Reader fit (no bar): 
* M2b_s0: chosen 0.0001; lr 0.001: early-stopping loss 0.00544, held-out R2 +0.570, epochs 20, lr 0.0003: early-stopping loss 0.00540, held-out R2 +0.472, epochs 14, lr 0.0001: early-stopping loss 0.00537, held-out R2 +0.430, epochs 14; shuffled-target yaw rate at the chosen lr -0.352 (no bar)
* M2b_s1: chosen 0.0001; lr 0.001: early-stopping loss 0.00636, held-out R2 +0.441, epochs 13, lr 0.0003: early-stopping loss 0.00610, held-out R2 +0.432, epochs 13, lr 0.0001: early-stopping loss 0.00600, held-out R2 +0.396, epochs 14; shuffled-target yaw rate at the chosen lr -0.314 (no bar)

## END ceilings (no world model, no planner; SCORE; paired EC-BOOT, B 2,000)
ORACLE-END: constant acceleration to the true speed at knot 12, yaw-rate ramp to the true value at knot 12. READ-END: the same with those two values from the reader on the encoded true future, starting from the measured current speed and yaw rate.

| model | family | unit | HOLD | ORACLE-END | READ-END | ORACLE - HOLD | sep | READ - HOLD | sep |
|---|---|---|---|---|---|---|---|---|---|
| M2b_s0 | LONGITUDINAL | m/s^2 | 0.3348 [0.2880, 0.3826] | 0.2161 [0.1855, 0.2472] | 1.1510 [0.9938, 1.3277] | -0.1187 [-0.1364, -0.1014] | better | 0.8162 [0.6495, 0.9964] | worse |
| M2b_s0 | LATERAL | rad/s | 0.0256 [0.0197, 0.0320] | 0.0142 [0.0113, 0.0172] | 0.0255 [0.0204, 0.0313] | -0.0114 [-0.0151, -0.0082] | better | -0.0001 [-0.0030, 0.0026] | none |
| M2b_s1 | LONGITUDINAL | m/s^2 | 0.3348 [0.2880, 0.3826] | 0.2161 [0.1855, 0.2472] | 1.1147 [0.9694, 1.2771] | -0.1187 [-0.1364, -0.1014] | better | 0.7799 [0.6254, 0.9495] | worse |
| M2b_s1 | LATERAL | rad/s | 0.0256 [0.0197, 0.0320] | 0.0142 [0.0113, 0.0172] | 0.0261 [0.0212, 0.0317] | -0.0114 [-0.0151, -0.0082] | better | 0.0005 [-0.0026, 0.0036] | none |
Separated-better than HOLD at both training seeds: LONGITUDINAL: ORACLE-END YES, READ-END NO; LATERAL: ORACLE-END YES, READ-END NO

## PT3M minus PT3N, G-TRUE, same SCORE windows (paired EC-BOOT, B 2,000; beside, no bar; negative is lower error)

| seed | form | family | lambda (PT3M / PT3N) | difference [95 %] | separated |
|---|---|---|---|---|---|
| M2b_s0 | plain | LONGITUDINAL | 1 / 1 | 0.0162 [0.0059, 0.0309] m/s^2 | worse |
| M2b_s0 | plain | LATERAL | 1 / 1 | -0.0001 [-0.0014, 0.0014] rad/s | none |
| M2b_s0 | shrink | LONGITUDINAL | 1 / 1 | 0.0106 [0.0021, 0.0223] m/s^2 | worse |
| M2b_s0 | shrink | LATERAL | 1 / 1 | 0.0004 [-0.0006, 0.0017] rad/s | none |
| M2b_s1 | plain | LONGITUDINAL | 1 / 1 | 0.0240 [0.0104, 0.0420] m/s^2 | worse |
| M2b_s1 | plain | LATERAL | 1 / 1 | 0.0007 [-0.0003, 0.0019] rad/s | none |
| M2b_s1 | shrink | LONGITUDINAL | 1 / 1 | 0.0120 [0.0043, 0.0225] m/s^2 | worse |
| M2b_s1 | shrink | LATERAL | 1 / 1 | 0.0007 [-0.0002, 0.0016] rad/s | none |

Card `../PREREG_V8F_PLANTEST3.md` (prefix sha256 `1f9609b4...b348f9`), IMPL `IMPL_CHOICES_PLANTEST3.md` (sha256 in `IMPL_CHOICES_PLANTEST3.sha256`). **Tier T0: a world-model diagnostic with an oracle (true-future) goal, never a driving result.** Evidence class: MEASURED (`runs/*.npz`, `RESULT_PLANTEST3.json`).
Estimator on every interval: EC-BOOT = paired episode-cluster bootstrap over the 71 SCORE episodes (994 windows), B 2,000, seed 0, 95 % percentile. Separated-better = interval wholly below 0 (MAE, lower is better). Decoders and lambda were chosen on the TUNE half only (hash split, salt `plantest2-v1`); every number below is SCORE. The interval answers 'another draw of episodes'; it is blind to training-seed and planner-seed variance, hence both training seeds are required and the planner-seed replicate is printed beside.

## Decoder gate (before planning; TUNE inner split, held-out episodes)
**DECODER-WEAK: YES** (M2b_s0, M2b_s1); bars: held-out R2 on encoded latents below 0.70 (speed) or 0.50 (yaw rate) at either seed. Missing gate records: none.
| model | read | R2 speed | R2 yaw rate | n speed / n yaw |
|---|---|---|---|---|
| M2b_s0 | D_1 encoded, all knots (the DECODER-WEAK quantity) | 0.6946 | 0.4300 | 3094 / 3094 |
| M2b_s0 | D_1 encoded, k = 12 | 0.6937 | 0.4763 | 238 / 238 |
| M2b_s0 | D_1 on the rollout at k = 12 under the logged actions | 0.6179 | 0.6212 | 238 / 238 |
| M2b_s0 | D_2 encoded, all knots | nan | nan | 0 / 0 |
| M2b_s1 | D_1 encoded, all knots (the DECODER-WEAK quantity) | 0.6857 | 0.3958 | 3094 / 3094 |
| M2b_s1 | D_1 encoded, k = 12 | 0.6889 | 0.4291 | 238 / 238 |
| M2b_s1 | D_1 on the rollout at k = 12 under the logged actions | 0.6026 | 0.5362 | 238 / 238 |
| M2b_s1 | D_2 encoded, all knots | nan | nan | 0 / 0 |

## Verdict (the card's committed table; per arm, planner and family; both training seeds required)
| planner | arm | form | family | chosen lambda s0 / s1 | M2b_s0 | M2b_s1 | decision label |
|---|---|---|---|---|---|---|---|
| FLAT1 | DEC-END | plain | LONGITUDINAL | 1 / 1 | NO-GAIN (collapse) | NO-GAIN | **NO-GAIN +HOLD-COLLAPSE(M2b_s0)** |
| FLAT1 | DEC-END | plain | LATERAL | 1 / 1 | NO-GAIN | NO-GAIN | **NO-GAIN** |
| FLAT1 | DEC-END | shrink | LONGITUDINAL | 1 / 1 | NO-GAIN (collapse) | NO-GAIN (collapse) | **NO-GAIN +HOLD-COLLAPSE(M2b_s0,M2b_s1)** |
| FLAT1 | DEC-END | shrink | LATERAL | 1 / 1 | NO-GAIN | NO-GAIN | **NO-GAIN** |
TACTICAL: not computed (n = 0: this read has no decision output). STRATEGIC: PAUSED (PI item 25, `PI_DECISION_QUEUE.md`).

HIER-ADDS (TD3 DEC-END minus FLAT1 DEC-END of the same form and lambda, G-TRUE, separated-better at both seeds): LONGITUDINAL: NO; LATERAL: NO

## KIN-CEILING (no world model, no planner): actions from the decoded true-future speed differences and yaw rates, MAE against the 12 logged actions, SCORE
| model | family | unit | HOLD | KIN | KIN-RAND (donor's decoded future) | KIN - HOLD | sep | KIN - KIN-RAND | sep |
|---|---|---|---|---|---|---|---|---|---|
| M2b_s0 | LONGITUDINAL | m/s^2 | 0.3348 [0.2880, 0.3826] | 1.4049 [1.3231, 1.4918] | 1.4661 [1.3960, 1.5425] | 1.0701 [0.9766, 1.1643] | worse | -0.0612 [-0.1744, 0.0563] | none |
| M2b_s0 | LATERAL | rad/s | 0.0256 [0.0197, 0.0320] | 0.0302 [0.0240, 0.0375] | 0.0434 [0.0342, 0.0533] | 0.0046 [0.0008, 0.0087] | worse | -0.0132 [-0.0218, -0.0059] | better |
| M2b_s1 | LONGITUDINAL | m/s^2 | 0.3348 [0.2880, 0.3826] | 1.4036 [1.3190, 1.4920] | 1.4244 [1.3542, 1.4989] | 1.0688 [0.9672, 1.1698] | worse | -0.0208 [-0.1365, 0.0960] | none |
| M2b_s1 | LATERAL | rad/s | 0.0256 [0.0197, 0.0320] | 0.0313 [0.0251, 0.0381] | 0.0440 [0.0345, 0.0540] | 0.0057 [0.0019, 0.0099] | worse | -0.0127 [-0.0212, -0.0058] | better |
KIN beats HOLD (separated-better at both seeds): LONGITUDINAL: NO; LATERAL: NO

## Deciding rows: MAE against the 12 logged actions, SCORE (EC-BOOT mean [95 % interval])
### LONGITUDINAL (acc_mae, m/s^2)
| model | planner | arm | form | lambda | HOLD | G-TRUE | G-RAND | TRUE - HOLD | sep | TRUE - RAND | sep | distance from HOLD, G-TRUE (G-RAND) [prior-std units] | base label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M2b_s0 | flat1 | DEC-END | plain | 1 | 0.3348 [0.2880, 0.3826] | 0.3499 [0.3032, 0.3988] | 0.3871 [0.3350, 0.4396] | 0.0151 [0.0030, 0.0328] | worse | -0.0372 [-0.0614, -0.0170] | better | 0.098 (0.189) COLLAPSE | NO-GAIN |
| M2b_s0 | flat1 | DEC-END | shrink | 1 | 0.3348 [0.2880, 0.3826] | 0.3413 [0.2951, 0.3897] | 0.3609 [0.3132, 0.4106] | 0.0065 [-0.0032, 0.0200] | none | -0.0196 [-0.0360, -0.0063] | better | 0.072 (0.124) COLLAPSE | NO-GAIN |
| M2b_s1 | flat1 | DEC-END | plain | 1 | 0.3348 [0.2880, 0.3826] | 0.3636 [0.3165, 0.4104] | 0.4018 [0.3518, 0.4520] | 0.0287 [0.0132, 0.0486] | worse | -0.0383 [-0.0621, -0.0175] | better | 0.130 (0.219) | NO-GAIN |
| M2b_s1 | flat1 | DEC-END | shrink | 1 | 0.3348 [0.2880, 0.3826] | 0.3439 [0.2992, 0.3909] | 0.3719 [0.3260, 0.4201] | 0.0091 [-0.0000, 0.0214] | none | -0.0280 [-0.0437, -0.0144] | better | 0.083 (0.152) COLLAPSE | NO-GAIN |

### LATERAL (yaw_mae, rad/s)
| model | planner | arm | form | lambda | HOLD | G-TRUE | G-RAND | TRUE - HOLD | sep | TRUE - RAND | sep | distance from HOLD, G-TRUE (G-RAND) [prior-std units] | base label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M2b_s0 | flat1 | DEC-END | plain | 1 | 0.0256 [0.0197, 0.0320] | 0.0283 [0.0217, 0.0357] | 0.0353 [0.0269, 0.0456] | 0.0028 [0.0009, 0.0052] | worse | -0.0070 [-0.0114, -0.0034] | better | 0.123 (0.197) | NO-GAIN |
| M2b_s0 | flat1 | DEC-END | shrink | 1 | 0.0256 [0.0197, 0.0320] | 0.0279 [0.0212, 0.0354] | 0.0320 [0.0241, 0.0414] | 0.0023 [0.0006, 0.0045] | worse | -0.0041 [-0.0077, -0.0016] | better | 0.100 (0.149) | NO-GAIN |
| M2b_s1 | flat1 | DEC-END | plain | 1 | 0.0256 [0.0197, 0.0320] | 0.0287 [0.0221, 0.0359] | 0.0354 [0.0277, 0.0441] | 0.0031 [0.0017, 0.0048] | worse | -0.0067 [-0.0096, -0.0041] | better | 0.128 (0.203) | NO-GAIN |
| M2b_s1 | flat1 | DEC-END | shrink | 1 | 0.0256 [0.0197, 0.0320] | 0.0277 [0.0211, 0.0350] | 0.0321 [0.0245, 0.0408] | 0.0021 [0.0007, 0.0039] | worse | -0.0045 [-0.0071, -0.0023] | better | 0.105 (0.155) | NO-GAIN |

HOLD-COLLAPSE flag: mean absolute distance of the G-TRUE plan from HOLD below 0.10 of the TRAIN prior std of the channel; a flagged row does not count as planning.

## HIER-ADDS rows: TD3 minus FLAT1 under G-TRUE, DEC-END (negative = TD3 better), SCORE
| family | model | form / lambda | TD3 - FLAT1 | sep |
|---|---|---|---|---|

## Beside, no bar: speed error, heading error and ADE at 2.4 s of the unicycle integration, plan vs logged actions (SCORE; at the family's chosen lambda; EC-BOOT mean [interval])
| model | planner | arm | form | family (lambda) | ref | speed err [m/s] | heading err [rad] | ADE [m] |
|---|---|---|---|---|---|---|---|---|
| M2b_s0 | flat1 | DEC-END | plain | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | flat1 | DEC-END | plain | LONGITUDINAL (1) | true | 0.6855 [0.5927, 0.7831] | 0.0604 [0.0459, 0.0770] | 0.2765 [0.2339, 0.3279] |
| M2b_s0 | flat1 | DEC-END | plain | LONGITUDINAL (1) | rand | 0.7540 [0.6588, 0.8507] | 0.0767 [0.0571, 0.1002] | 0.3783 [0.3145, 0.4524] |
| M2b_s0 | flat1 | DEC-END | plain | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | flat1 | DEC-END | plain | LATERAL (1) | true | 0.6855 [0.5927, 0.7831] | 0.0604 [0.0459, 0.0770] | 0.2765 [0.2339, 0.3279] |
| M2b_s0 | flat1 | DEC-END | plain | LATERAL (1) | rand | 0.7540 [0.6588, 0.8507] | 0.0767 [0.0571, 0.1002] | 0.3783 [0.3145, 0.4524] |
| M2b_s0 | flat1 | DEC-END | shrink | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | flat1 | DEC-END | shrink | LONGITUDINAL (1) | true | 0.6713 [0.5793, 0.7686] | 0.0599 [0.0453, 0.0768] | 0.2608 [0.2229, 0.3050] |
| M2b_s0 | flat1 | DEC-END | shrink | LONGITUDINAL (1) | rand | 0.7068 [0.6143, 0.8022] | 0.0699 [0.0521, 0.0919] | 0.3167 [0.2646, 0.3798] |
| M2b_s0 | flat1 | DEC-END | shrink | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s0 | flat1 | DEC-END | shrink | LATERAL (1) | true | 0.6713 [0.5793, 0.7686] | 0.0599 [0.0453, 0.0768] | 0.2608 [0.2229, 0.3050] |
| M2b_s0 | flat1 | DEC-END | shrink | LATERAL (1) | rand | 0.7068 [0.6143, 0.8022] | 0.0699 [0.0521, 0.0919] | 0.3167 [0.2646, 0.3798] |
| M2b_s1 | flat1 | DEC-END | plain | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | flat1 | DEC-END | plain | LONGITUDINAL (1) | true | 0.7111 [0.6190, 0.8079] | 0.0608 [0.0469, 0.0765] | 0.2880 [0.2488, 0.3328] |
| M2b_s1 | flat1 | DEC-END | plain | LONGITUDINAL (1) | rand | 0.7859 [0.6849, 0.8841] | 0.0767 [0.0592, 0.0958] | 0.4119 [0.3481, 0.4904] |
| M2b_s1 | flat1 | DEC-END | plain | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | flat1 | DEC-END | plain | LATERAL (1) | true | 0.7111 [0.6190, 0.8079] | 0.0608 [0.0469, 0.0765] | 0.2880 [0.2488, 0.3328] |
| M2b_s1 | flat1 | DEC-END | plain | LATERAL (1) | rand | 0.7859 [0.6849, 0.8841] | 0.0767 [0.0592, 0.0958] | 0.4119 [0.3481, 0.4904] |
| M2b_s1 | flat1 | DEC-END | shrink | LONGITUDINAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | flat1 | DEC-END | shrink | LONGITUDINAL (1) | true | 0.6788 [0.5878, 0.7759] | 0.0591 [0.0448, 0.0751] | 0.2597 [0.2248, 0.3004] |
| M2b_s1 | flat1 | DEC-END | shrink | LONGITUDINAL (1) | rand | 0.7238 [0.6339, 0.8135] | 0.0700 [0.0530, 0.0893] | 0.3371 [0.2895, 0.3939] |
| M2b_s1 | flat1 | DEC-END | shrink | LATERAL (1) | hold | 0.6903 [0.5894, 0.7955] | 0.0572 [0.0437, 0.0721] | 0.2284 [0.2016, 0.2582] |
| M2b_s1 | flat1 | DEC-END | shrink | LATERAL (1) | true | 0.6788 [0.5878, 0.7759] | 0.0591 [0.0448, 0.0751] | 0.2597 [0.2248, 0.3004] |
| M2b_s1 | flat1 | DEC-END | shrink | LATERAL (1) | rand | 0.7238 [0.6339, 0.8135] | 0.0700 [0.0530, 0.0893] | 0.3371 [0.2895, 0.3939] |

## Beside, no bar: cost monotonicity (Tab. 10 statistic: -Spearman(distance, time); 1.0 = perfectly decreasing) along the chosen G-TRUE plan, level 1: decoded distance to the END goal, and latent distance to the END goal; fraction of windows where planning lowered the cost below the best initial candidate (the cost includes REG when lambda > 0)
| run | decoded monotonicity | latent monotonicity | frac lowered |
|---|---|---|---|
| M2b_s0|flat1|end|plain|LONGITUDINAL (lambda 1) | -0.1396 [-0.2287, -0.0493] | -0.4550 [-0.5274, -0.3824] | 0.9970 [0.9939, 1.0000] |
| M2b_s0|flat1|end|plain|LATERAL (lambda 1) | -0.1396 [-0.2287, -0.0493] | -0.4550 [-0.5274, -0.3824] | 0.9970 [0.9939, 1.0000] |
| M2b_s0|flat1|end|shrink|LONGITUDINAL (lambda 1) | -0.0533 [-0.1279, 0.0266] | -0.0147 [-0.1013, 0.0729] | 0.9920 [0.9849, 0.9970] |
| M2b_s0|flat1|end|shrink|LATERAL (lambda 1) | -0.0533 [-0.1279, 0.0266] | -0.0147 [-0.1013, 0.0729] | 0.9920 [0.9849, 0.9970] |
| M2b_s1|flat1|end|plain|LONGITUDINAL (lambda 1) | -0.1925 [-0.2952, -0.0887] | -0.4483 [-0.5251, -0.3610] | 1.0000 [1.0000, 1.0000] |
| M2b_s1|flat1|end|plain|LATERAL (lambda 1) | -0.1925 [-0.2952, -0.0887] | -0.4483 [-0.5251, -0.3610] | 1.0000 [1.0000, 1.0000] |
| M2b_s1|flat1|end|shrink|LONGITUDINAL (lambda 1) | -0.0745 [-0.1668, 0.0235] | -0.0720 [-0.1612, 0.0243] | 0.9909 [0.9829, 0.9970] |
| M2b_s1|flat1|end|shrink|LATERAL (lambda 1) | -0.0745 [-0.1668, 0.0235] | -0.0720 [-0.1612, 0.0243] | 0.9909 [0.9829, 0.9970] |

## Beside: planner-seed replicate (planner seed 1 minus seed 0, G-TRUE, same checkpoint and lambda, SCORE, EC-BOOT; a difference of this size is inference noise, not an effect)
* M2b_s0|flat1|end|plain|LONGITUDINAL (lambda 1): 0.0004 [-0.0000, 0.0012] (none)
* M2b_s0|flat1|end|plain|LATERAL (lambda 1): 0.0001 [-0.0001, 0.0004] (none)
* M2b_s0|flat1|end|shrink|LONGITUDINAL (lambda 1): 0.0001 [-0.0000, 0.0004] (none)
* M2b_s0|flat1|end|shrink|LATERAL (lambda 1): -0.0000 [-0.0001, 0.0000] (none)
* M2b_s1|flat1|end|plain|LONGITUDINAL (lambda 1): 0.0000 [-0.0000, 0.0001] (none)
* M2b_s1|flat1|end|plain|LATERAL (lambda 1): 0.0000 [-0.0000, 0.0000] (none)
* M2b_s1|flat1|end|shrink|LONGITUDINAL (lambda 1): 0.0000 [-0.0000, 0.0001] (none)
* M2b_s1|flat1|end|shrink|LATERAL (lambda 1): 0.0000 [-0.0000, 0.0000] (none)

## Beside: the TUNE grid (FLAT1, G-TRUE, planner seed 0, mean MAE over the TUNE windows; lambda is chosen as the row minimum per family; this is the selection record, not a result) and the TD3 form pick
| model | arm | form | lambda | LONGITUDINAL acc MAE [m/s^2] | LATERAL yaw MAE [rad/s] |
|---|---|---|---|---|---|
| M2b_s0 | DEC-END | plain | 0 | 1.3498 | 0.1475 |
| M2b_s0 | DEC-END | plain | 0.1 | 0.4818 | 0.0360 |
| M2b_s0 | DEC-END | plain | 1 | 0.3378 (chosen) | 0.0214 (chosen) |
| M2b_s0 | DEC-END | shrink | 0 | 1.3795 | 0.1555 |
| M2b_s0 | DEC-END | shrink | 0.1 | 0.4015 | 0.0285 |
| M2b_s0 | DEC-END | shrink | 1 | 0.3274 (chosen) | 0.0207 (chosen) |
| M2b_s1 | DEC-END | plain | 0 | 1.2813 | 0.1352 |
| M2b_s1 | DEC-END | plain | 0.1 | 0.5327 | 0.0382 |
| M2b_s1 | DEC-END | plain | 1 | 0.3418 (chosen) | 0.0214 (chosen) |
| M2b_s1 | DEC-END | shrink | 0 | 1.3170 | 0.1425 |
| M2b_s1 | DEC-END | shrink | 0.1 | 0.4322 | 0.0310 |
| M2b_s1 | DEC-END | shrink | 1 | 0.3296 (chosen) | 0.0209 (chosen) |
TD3 form and lambda (the FLAT1 DEC-END choice per training seed and family): M2b_s0|LONGITUDINAL: shrink / 1; M2b_s0|LATERAL: shrink / 1; M2b_s1|LONGITUDINAL: shrink / 1; M2b_s1|LATERAL: shrink / 1

Fraction of SCORE windows whose HOLD lies outside the level-1 box (HOLD_c differs from HOLD): M2b_s0 0.090, M2b_s1 0.090
