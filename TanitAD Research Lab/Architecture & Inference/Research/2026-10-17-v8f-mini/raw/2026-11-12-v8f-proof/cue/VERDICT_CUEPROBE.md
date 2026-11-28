# VERDICT_CUEPROBE (PREREG_V8F_PROOF_ADD5, frozen 2026-11-13)

Evidence class MEASURED (ours, zero training, CPU). Tier T0 for the probe reads (frozen tokens, closed-form kernel ridge, 5 folds by episode), T1-kin on logged frames for the shield reads. n windows 2399 (131 stop windows, 2365 labelled), 141 episodes; paired episode-cluster bootstrap B 2000; fold seeds [0, 1, 2, 3, 4] (outcomes read at fold seed 0, the others shown as spread). A diagnostic of the input interface, never driving performance. STRATEGIC family: PAUSED (PI item 25), not a pass, not omitted.
Full window set: True; smoke: False.

## Controls
* all controls pass: **False**  (a control failed: the reads below are not valid)
* k1 shuffled labels PASS: AUROC {"E": 0.471, "V40": 0.515, "V160": 0.511, "V40+E": 0.514, "V160+E": 0.511}; DIST better than floor with CI: {"E": false, "V40": false, "V160": false, "V40+E": false, "V160+E": false}
* k2 OF-OD vs banked ORACLE PASS: seed 0: delta -3.5099312965 vs -3.5099312965 (diff 0.0e+00), overrun share 0.0229 vs 0.0229, infeasible-at-start stop windows 3, evaluated True; seed 1: delta -3.3735825716 vs -3.3735825716 (diff 0.0e+00), overrun share 0.0153 vs 0.0153, infeasible-at-start stop windows 2, evaluated True
* k3 no active window bit-identical to NULL PASS
* k4 leak guard FAIL: guard fires on tokens 2.0 s later True; AUROC real {"V40": 0.711, "V160": 0.705, "V40+E": 0.797, "V160+E": 0.799} vs late {"V40": 0.748, "V160": 0.749, "V40+E": 0.789, "V160+E": 0.795}

## Read 1 FLAG (T0, AUROC over labelled windows, fold seed 0)

| set | AUROC | 95 % CI | minus E | CI | separated |
|---|---|---|---|---|---|
| E | 0.8101 | [0.7437457249882872, 0.8675406763745092] | - | - | - |
| V40 | 0.7113 | [0.6309797864676895, 0.7965787592726449] | -0.0988 | [-0.1776, -0.0067] | False |
| V160 | 0.7052 | [0.6273203901400459, 0.7907968018705291] | -0.1049 | [-0.1845, -0.0092] | False |
| V40+E | 0.7973 | [0.7295186075154129, 0.8652551113021879] | -0.0127 | [-0.0684, +0.0447] | False |
| V160+E | 0.7987 | [0.731305392530206, 0.8662101336243047] | -0.0114 | [-0.0613, +0.0431] | False |

Outcome: **VISION-FLAT** (V160+E minus E -0.0114, CI [-0.06130826448451483, 0.04313789159213602]); V40+E minus E: **VISION-FLAT** (-0.0127, CI [-0.06840448124820447, 0.04473662443173737]). Bar: diff >= 0.02 and CI excludes 0: FAIL.

AUROC spread over fold seeds (V160+E minus E): fs0 -0.0114, fs1 -0.0407, fs2 -0.0372, fs3 +0.0149, fs4 -0.1034

## Read 2 DIST (T0, plan seed 0, n stop 131, fold seed 0)

| set | median abs err m | mean abs err m | R2 |
|---|---|---|---|
| constant (training-fold median) | 10.335 | - | -0.055 |
| E | 3.819 | 4.899 | 0.845 |
| V40 | 10.519 | 13.152 | -0.160 |
| V160 | 10.649 | 13.201 | -0.159 |
| V40+E | 10.595 | 12.517 | -0.032 |
| V160+E | 12.454 | 14.898 | -0.466 |

Best set E: median 3.819 m (bar <= 5.0 usable, <= 8.0 coarse); gain over constant floor +6.516 m CI [+4.930, +8.849] better: True; over E +0.000 m CI [+0.000, +0.000] better: False.
Best vision set V40: vs const -0.184 [-2.145, +1.799]; vs E -6.700 [-9.710, -4.433].
Outcome: **CUE-ABSENT**. Median error of the best set over fold seeds: fs0 3.82, fs1 4.01, fs2 3.76, fs3 3.78, fs4 3.80

## Read 2 DIST (T0, plan seed 1, n stop 131, fold seed 0)

| set | median abs err m | mean abs err m | R2 |
|---|---|---|---|
| constant (training-fold median) | 10.369 | - | -0.053 |
| E | 3.967 | 4.797 | 0.851 |
| V40 | 10.942 | 13.091 | -0.163 |
| V160 | 11.250 | 13.143 | -0.163 |
| V40+E | 10.581 | 12.467 | -0.034 |
| V160+E | 12.399 | 14.992 | -0.520 |

Best set E: median 3.967 m (bar <= 5.0 usable, <= 8.0 coarse); gain over constant floor +6.402 m CI [+4.984, +8.469] better: True; over E +0.000 m CI [+0.000, +0.000] better: False.
Best vision set V40+E: vs const -0.212 [-1.534, +1.193]; vs E -6.614 [-8.746, -4.618].
Outcome: **CUE-ABSENT**. Median error of the best set over fold seeds: fs0 3.97, fs1 3.83, fs2 3.57, fs3 3.71, fs4 3.80

DIST outcome (needs both plan seeds, the weaker label is reported): **CUE-ABSENT**; per seed {'0': 'CUE-ABSENT', '1': 'CUE-ABSENT'}

## Read 3 SHIELD with the probe cue (T1-kin, fold seed 0; a = 3.0 m/s2, dt 0.2 s; paired episode-cluster bootstrap against NULL)

### Plan seed 0: **CUE-PROBE-COSTS**

| bar | measured | result |
|---|---|---|
| stop-window 2-6 s ADE change negative, CI excludes 0 | +0.0911 m [-0.5322, +0.6390] (n 131) | FAIL |
| size >= 0.30 of the ORACLE gain at this seed | 0.026 (ORACLE -3.5099 m) | FAIL |
| non-stop 0-6 s ADE change, upper CI end <= +0.05 m | +0.0523 m [+0.0269, +0.0838] (n 2268) | FAIL |

| arm | stop 2-6 s ADE | non-stop 0-6 s ADE | all 0-6 s ADE | overrun share (stop, NULL) | overrun mean m | hit rate | FP rate | active | TP/FP/FN/TN | lon label changed |
|---|---|---|---|---|---|---|---|---|---|---|
| PF-PD | +0.0911 [-0.5322, +0.6390] | +0.0523 [+0.0269, +0.0838]* | +0.0535 [+0.0185, +0.0920]* | 0.725 (0.916) | 7.842 | 0.267 | 0.0215 | 83 | 35/48/96/2186 | 0.005 |
| OF-PD | +0.0830 [-1.1598, +1.3461] | +0.0000 [+0.0000, +0.0000] | +0.0052 [-0.0479, +0.0527] | 0.107 (0.916) | 0.347 | 0.267 | 0.0215 | 83 | 35/48/96/2186 | 0.010 |
| PF-OD | -0.6257 [-1.1249, -0.2138]* | +0.0523 [+0.0269, +0.0838]* | +0.0264 [-0.0102, +0.0653] | 0.710 (0.916) | 7.826 | 0.267 | 0.0215 | 83 | 35/48/96/2186 | 0.005 |
| OF-OD | -3.5099 [-4.3835, -2.6164]* | +0.0000 [+0.0000, +0.0000] | -0.1288 [-0.1952, -0.0684]* | 0.023 (0.916) | 0.019 | 0.267 | 0.0215 | 83 | 35/48/96/2186 | 0.010 |

Secondary (not a card outcome): tactical rates at the inner cross-fitted threshold: {'hit_rate': 0.3969465648854962, 'fp_rate': 0.03804834377797672, 'active': 137}.

All cells (family; variant minus NULL; * = CI excludes 0):

| arm | cell | delta [lo, hi] | n |
|---|---|---|---|
| PF-PD | LONGITUDINAL OVERRUN_vs_logged_rest_m|stop | -1.8292 [-3.0612, -0.7919]* | 131 |
| PF-PD | LONGITUDINAL full_0_6s|accel_mae_mps2|all | +0.0004 [-0.0057, +0.0061] | 2399 |
| PF-PD | LONGITUDINAL full_0_6s|accel_mae_mps2|nonstop | +0.0033 [+0.0002, +0.0072]* | 2268 |
| PF-PD | LONGITUDINAL full_0_6s|accel_mae_mps2|stop | -0.0509 [-0.1307, +0.0182] | 131 |
| PF-PD | LONGITUDINAL full_0_6s|ade_m|all | +0.0535 [+0.0185, +0.0920]* | 2399 |
| PF-PD | LONGITUDINAL full_0_6s|ade_m|nonstop | +0.0523 [+0.0269, +0.0838]* | 2268 |
| PF-PD | LONGITUDINAL full_0_6s|ade_m|stop | +0.0758 [-0.3522, +0.4522] | 131 |
| PF-PD | LONGITUDINAL full_0_6s|along_mae_m|all | +0.0593 [+0.0248, +0.0975]* | 2399 |
| PF-PD | LONGITUDINAL full_0_6s|along_mae_m|nonstop | +0.0549 [+0.0293, +0.0874]* | 2268 |
| PF-PD | LONGITUDINAL full_0_6s|along_mae_m|stop | +0.1362 [-0.2744, +0.5015] | 131 |
| PF-PD | LATERAL full_0_6s|cross_mae_m|all | +0.0009 [-0.0106, +0.0119] | 2399 |
| PF-PD | LATERAL full_0_6s|cross_mae_m|nonstop | +0.0053 [-0.0023, +0.0150] | 2268 |
| PF-PD | LATERAL full_0_6s|cross_mae_m|stop | -0.0747 [-0.2125, +0.0228] | 131 |
| PF-PD | LATERAL full_0_6s|curvature_mae_1pm|all | -0.0026 [-0.0054, -0.0002]* | 2325 |
| PF-PD | LATERAL full_0_6s|curvature_mae_1pm|nonstop | -0.0021 [-0.0048, +0.0002] | 2195 |
| PF-PD | LATERAL full_0_6s|curvature_mae_1pm|stop | -0.0110 [-0.0178, -0.0053]* | 130 |
| PF-PD | LONGITUDINAL full_0_6s|fde_m|all | +0.1127 [-0.0065, +0.2422] | 2399 |
| PF-PD | LONGITUDINAL full_0_6s|fde_m|nonstop | +0.1476 [+0.0623, +0.2616]* | 2268 |
| PF-PD | LONGITUDINAL full_0_6s|fde_m|stop | -0.4912 [-1.7467, +0.6047] | 131 |
| PF-PD | LATERAL full_0_6s|heading_mae_deg|all | -0.1763 [-0.3262, -0.0612]* | 2370 |
| PF-PD | LATERAL full_0_6s|heading_mae_deg|nonstop | -0.1295 [-0.2828, -0.0308]* | 2240 |
| PF-PD | LATERAL full_0_6s|heading_mae_deg|stop | -0.9830 [-2.0341, -0.3149]* | 130 |
| PF-PD | LONGITUDINAL full_0_6s|speed_mae_mps|all | +0.0212 [+0.0006, +0.0446]* | 2399 |
| PF-PD | LONGITUDINAL full_0_6s|speed_mae_mps|nonstop | +0.0295 [+0.0140, +0.0506]* | 2268 |
| PF-PD | LONGITUDINAL full_0_6s|speed_mae_mps|stop | -0.1230 [-0.3259, +0.0447] | 131 |
| PF-PD | LATERAL full_0_6s|yaw_rate_mae_degps|all | -0.0779 [-0.2339, +0.0700] | 2325 |
| PF-PD | LATERAL full_0_6s|yaw_rate_mae_degps|nonstop | -0.0775 [-0.2248, +0.0645] | 2195 |
| PF-PD | LATERAL full_0_6s|yaw_rate_mae_degps|stop | -0.0847 [-0.8567, +0.7820] | 130 |
| PF-PD | LONGITUDINAL operative_0_2s|accel_mae_mps2|all | +0.0077 [+0.0022, +0.0132]* | 2399 |
| PF-PD | LONGITUDINAL operative_0_2s|accel_mae_mps2|nonstop | +0.0087 [+0.0047, +0.0133]* | 2268 |
| PF-PD | LONGITUDINAL operative_0_2s|accel_mae_mps2|stop | -0.0082 [-0.0920, +0.0562] | 131 |
| PF-PD | LONGITUDINAL operative_0_2s|ade_m|all | +0.0066 [+0.0032, +0.0106]* | 2399 |
| PF-PD | LONGITUDINAL operative_0_2s|ade_m|nonstop | +0.0044 [+0.0017, +0.0073]* | 2268 |
| PF-PD | LONGITUDINAL operative_0_2s|ade_m|stop | +0.0453 [+0.0063, +0.0844]* | 131 |
| PF-PD | LONGITUDINAL operative_0_2s|along_mae_m|all | +0.0071 [+0.0035, +0.0111]* | 2399 |
| PF-PD | LONGITUDINAL operative_0_2s|along_mae_m|nonstop | +0.0047 [+0.0020, +0.0077]* | 2268 |
| PF-PD | LONGITUDINAL operative_0_2s|along_mae_m|stop | +0.0480 [+0.0092, +0.0875]* | 131 |
| PF-PD | LATERAL operative_0_2s|cross_mae_m|all | -0.0001 [-0.0007, +0.0005] | 2399 |
| PF-PD | LATERAL operative_0_2s|cross_mae_m|nonstop | -0.0003 [-0.0008, +0.0001] | 2268 |
| PF-PD | LATERAL operative_0_2s|cross_mae_m|stop | +0.0035 [-0.0046, +0.0134] | 131 |
| PF-PD | LATERAL operative_0_2s|curvature_mae_1pm|all | -0.0052 [-0.0108, -0.0014]* | 2305 |
| PF-PD | LATERAL operative_0_2s|curvature_mae_1pm|nonstop | -0.0046 [-0.0093, -0.0014]* | 2175 |
| PF-PD | LATERAL operative_0_2s|curvature_mae_1pm|stop | -0.0154 [-0.0377, -0.0011]* | 130 |
| PF-PD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|all | +0.0044 [-0.0042, +0.0135] | 705 |
| PF-PD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|nonstop | -0.0004 [-0.0065, +0.0040] | 658 |
| PF-PD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|stop | +0.0718 [-0.0012, +0.1514] | 47 |
| PF-PD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|all | +0.0008 [-0.0009, +0.0026] | 647 |
| PF-PD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|nonstop | -0.0000 [-0.0013, +0.0009] | 601 |
| PF-PD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|stop | +0.0117 [-0.0001, +0.0239] | 46 |
| PF-PD | LONGITUDINAL operative_0_2s|fde_m|all | +0.0233 [+0.0137, +0.0341]* | 2399 |
| PF-PD | LONGITUDINAL operative_0_2s|fde_m|nonstop | +0.0153 [+0.0081, +0.0235]* | 2268 |
| PF-PD | LONGITUDINAL operative_0_2s|fde_m|stop | +0.1606 [+0.0442, +0.2766]* | 131 |
| PF-PD | LATERAL operative_0_2s|heading_mae_deg|all | -0.0299 [-0.0607, -0.0082]* | 2314 |
| PF-PD | LATERAL operative_0_2s|heading_mae_deg|nonstop | -0.0247 [-0.0503, -0.0075]* | 2184 |
| PF-PD | LATERAL operative_0_2s|heading_mae_deg|stop | -0.1172 [-0.3023, +0.0454] | 130 |
| PF-PD | LONGITUDINAL operative_0_2s|speed_mae_mps|all | +0.0113 [+0.0068, +0.0164]* | 2399 |
| PF-PD | LONGITUDINAL operative_0_2s|speed_mae_mps|nonstop | +0.0076 [+0.0042, +0.0115]* | 2268 |
| PF-PD | LONGITUDINAL operative_0_2s|speed_mae_mps|stop | +0.0746 [+0.0191, +0.1280]* | 131 |
| PF-PD | LATERAL operative_0_2s|yaw_rate_mae_degps|all | -0.4299 [-0.9034, -0.1406]* | 2305 |
| PF-PD | LATERAL operative_0_2s|yaw_rate_mae_degps|nonstop | -0.3096 [-0.5994, -0.1129]* | 2175 |
| PF-PD | LATERAL operative_0_2s|yaw_rate_mae_degps|stop | -2.4421 [-5.5486, -0.5309]* | 130 |
| PF-PD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|all | -0.0040 [-0.0113, +0.0027] | 2399 |
| PF-PD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|nonstop | +0.0002 [-0.0039, +0.0048] | 2268 |
| PF-PD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|stop | -0.0766 [-0.1628, -0.0011]* | 131 |
| PF-PD | LONGITUDINAL tactical_2_6s|ade_m|all | +0.0770 [+0.0251, +0.1336]* | 2399 |
| PF-PD | LONGITUDINAL tactical_2_6s|ade_m|nonstop | +0.0762 [+0.0388, +0.1235]* | 2268 |
| PF-PD | LONGITUDINAL tactical_2_6s|ade_m|stop | +0.0911 [-0.5322, +0.6390] | 131 |
| PF-PD | LONGITUDINAL tactical_2_6s|along_mae_m|all | +0.0854 [+0.0343, +0.1410]* | 2399 |
| PF-PD | LONGITUDINAL tactical_2_6s|along_mae_m|nonstop | +0.0800 [+0.0421, +0.1284]* | 2268 |
| PF-PD | LONGITUDINAL tactical_2_6s|along_mae_m|stop | +0.1803 [-0.4192, +0.7152] | 131 |
| PF-PD | LATERAL tactical_2_6s|cross_mae_m|all | +0.0015 [-0.0157, +0.0180] | 2399 |
| PF-PD | LATERAL tactical_2_6s|cross_mae_m|nonstop | +0.0081 [-0.0034, +0.0226] | 2268 |
| PF-PD | LATERAL tactical_2_6s|cross_mae_m|stop | -0.1137 [-0.3185, +0.0295] | 131 |
| PF-PD | LATERAL tactical_2_6s|curvature_mae_1pm|all | -0.0005 [-0.0012, -0.0000]* | 2256 |
| PF-PD | LATERAL tactical_2_6s|curvature_mae_1pm|nonstop | -0.0000 [-0.0001, +0.0001] | 2145 |
| PF-PD | LATERAL tactical_2_6s|curvature_mae_1pm|stop | -0.0092 [-0.0224, -0.0001]* | 111 |
| PF-PD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|all | -0.0079 [-0.0279, +0.0073] | 588 |
| PF-PD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|nonstop | +0.0027 [-0.0077, +0.0127] | 554 |
| PF-PD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|stop | -0.1808 [-0.3995, +0.0000] | 34 |
| PF-PD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|all | -0.0003 [-0.0046, +0.0040] | 530 |
| PF-PD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|nonstop | +0.0019 [-0.0009, +0.0060] | 497 |
| PF-PD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|stop | -0.0338 [-0.0754, +0.0000] | 33 |
| PF-PD | LONGITUDINAL tactical_2_6s|fde_m|all | +0.1127 [-0.0065, +0.2422] | 2399 |
| PF-PD | LONGITUDINAL tactical_2_6s|fde_m|nonstop | +0.1476 [+0.0623, +0.2616]* | 2268 |
| PF-PD | LONGITUDINAL tactical_2_6s|fde_m|stop | -0.4912 [-1.7467, +0.6047] | 131 |
| PF-PD | LATERAL tactical_2_6s|heading_mae_deg|all | -0.0808 [-0.1782, -0.0214]* | 2312 |
| PF-PD | LATERAL tactical_2_6s|heading_mae_deg|nonstop | -0.0648 [-0.1655, -0.0063]* | 2196 |
| PF-PD | LATERAL tactical_2_6s|heading_mae_deg|stop | -0.3846 [-0.7187, -0.1357]* | 116 |
| PF-PD | LONGITUDINAL tactical_2_6s|speed_mae_mps|all | +0.0262 [-0.0035, +0.0608] | 2399 |
| PF-PD | LONGITUDINAL tactical_2_6s|speed_mae_mps|nonstop | +0.0405 [+0.0180, +0.0708]* | 2268 |
| PF-PD | LONGITUDINAL tactical_2_6s|speed_mae_mps|stop | -0.2218 [-0.5087, +0.0137] | 131 |
| PF-PD | LATERAL tactical_2_6s|yaw_rate_mae_degps|all | -0.0470 [-0.0794, -0.0183]* | 2256 |
| PF-PD | LATERAL tactical_2_6s|yaw_rate_mae_degps|nonstop | -0.0176 [-0.0342, -0.0010]* | 2145 |
| PF-PD | LATERAL tactical_2_6s|yaw_rate_mae_degps|stop | -0.6140 [-1.1410, -0.1952]* | 111 |
| OF-PD | LONGITUDINAL OVERRUN_vs_logged_rest_m|stop | -9.3246 [-11.0866, -7.3430]* | 131 |
| OF-PD | LONGITUDINAL full_0_6s|accel_mae_mps2|all | +0.0029 [-0.0091, +0.0127] | 2399 |
| OF-PD | LONGITUDINAL full_0_6s|accel_mae_mps2|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL full_0_6s|accel_mae_mps2|stop | +0.0524 [-0.1560, +0.2365] | 131 |
| OF-PD | LONGITUDINAL full_0_6s|ade_m|all | +0.0052 [-0.0479, +0.0527] | 2399 |
| OF-PD | LONGITUDINAL full_0_6s|ade_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL full_0_6s|ade_m|stop | +0.0954 [-0.7575, +0.9704] | 131 |
| OF-PD | LONGITUDINAL full_0_6s|along_mae_m|all | +0.0157 [-0.0372, +0.0628] | 2399 |
| OF-PD | LONGITUDINAL full_0_6s|along_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL full_0_6s|along_mae_m|stop | +0.2875 [-0.5910, +1.1551] | 131 |
| OF-PD | LATERAL full_0_6s|cross_mae_m|all | -0.0103 [-0.0258, +0.0056] | 2399 |
| OF-PD | LATERAL full_0_6s|cross_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LATERAL full_0_6s|cross_mae_m|stop | -0.1886 [-0.4651, +0.1012] | 131 |
| OF-PD | LATERAL full_0_6s|curvature_mae_1pm|all | -0.0011 [-0.0019, -0.0005]* | 2326 |
| OF-PD | LATERAL full_0_6s|curvature_mae_1pm|nonstop | +0.0000 [+0.0000, +0.0000] | 2196 |
| OF-PD | LATERAL full_0_6s|curvature_mae_1pm|stop | -0.0205 [-0.0301, -0.0117]* | 130 |
| OF-PD | LONGITUDINAL full_0_6s|fde_m|all | -0.1415 [-0.3074, -0.0007]* | 2399 |
| OF-PD | LONGITUDINAL full_0_6s|fde_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL full_0_6s|fde_m|stop | -2.5911 [-5.0364, -0.2064]* | 131 |
| OF-PD | LATERAL full_0_6s|heading_mae_deg|all | -0.1265 [-0.2256, -0.0562]* | 2370 |
| OF-PD | LATERAL full_0_6s|heading_mae_deg|nonstop | +0.0000 [+0.0000, +0.0000] | 2240 |
| OF-PD | LATERAL full_0_6s|heading_mae_deg|stop | -2.3064 [-3.6584, -1.2421]* | 130 |
| OF-PD | LONGITUDINAL full_0_6s|speed_mae_mps|all | -0.0222 [-0.0485, +0.0009] | 2399 |
| OF-PD | LONGITUDINAL full_0_6s|speed_mae_mps|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL full_0_6s|speed_mae_mps|stop | -0.4072 [-0.8002, -0.0209]* | 131 |
| OF-PD | LATERAL full_0_6s|yaw_rate_mae_degps|all | -0.0505 [-0.1029, +0.0019] | 2326 |
| OF-PD | LATERAL full_0_6s|yaw_rate_mae_degps|nonstop | +0.0000 [+0.0000, +0.0000] | 2196 |
| OF-PD | LATERAL full_0_6s|yaw_rate_mae_degps|stop | -0.9034 [-1.7261, -0.0095]* | 130 |
| OF-PD | LONGITUDINAL operative_0_2s|accel_mae_mps2|all | +0.0184 [+0.0066, +0.0313]* | 2399 |
| OF-PD | LONGITUDINAL operative_0_2s|accel_mae_mps2|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL operative_0_2s|accel_mae_mps2|stop | +0.3372 [+0.1454, +0.5304]* | 131 |
| OF-PD | LONGITUDINAL operative_0_2s|ade_m|all | +0.0066 [+0.0009, +0.0130]* | 2399 |
| OF-PD | LONGITUDINAL operative_0_2s|ade_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL operative_0_2s|ade_m|stop | +0.1203 [+0.0258, +0.2287]* | 131 |
| OF-PD | LONGITUDINAL operative_0_2s|along_mae_m|all | +0.0068 [+0.0012, +0.0132]* | 2399 |
| OF-PD | LONGITUDINAL operative_0_2s|along_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL operative_0_2s|along_mae_m|stop | +0.1245 [+0.0340, +0.2295]* | 131 |
| OF-PD | LATERAL operative_0_2s|cross_mae_m|all | +0.0004 [-0.0008, +0.0022] | 2399 |
| OF-PD | LATERAL operative_0_2s|cross_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LATERAL operative_0_2s|cross_mae_m|stop | +0.0074 [-0.0143, +0.0386] | 131 |
| OF-PD | LATERAL operative_0_2s|curvature_mae_1pm|all | -0.0013 [-0.0029, -0.0003]* | 2305 |
| OF-PD | LATERAL operative_0_2s|curvature_mae_1pm|nonstop | +0.0000 [+0.0000, +0.0000] | 2175 |
| OF-PD | LATERAL operative_0_2s|curvature_mae_1pm|stop | -0.0229 [-0.0452, -0.0067]* | 130 |
| OF-PD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|all | +0.0215 [+0.0001, +0.0532]* | 705 |
| OF-PD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|nonstop | +0.0000 [+0.0000, +0.0000] | 659 |
| OF-PD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|stop | +0.3288 [+0.0218, +0.7448]* | 46 |
| OF-PD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|all | +0.0062 [+0.0009, +0.0134]* | 648 |
| OF-PD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|nonstop | +0.0000 [+0.0000, +0.0000] | 602 |
| OF-PD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|stop | +0.0873 [+0.0219, +0.1746]* | 46 |
| OF-PD | LONGITUDINAL operative_0_2s|fde_m|all | +0.0222 [+0.0064, +0.0419]* | 2399 |
| OF-PD | LONGITUDINAL operative_0_2s|fde_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL operative_0_2s|fde_m|stop | +0.4056 [+0.1370, +0.7190]* | 131 |
| OF-PD | LATERAL operative_0_2s|heading_mae_deg|all | -0.0112 [-0.0244, +0.0021] | 2314 |
| OF-PD | LATERAL operative_0_2s|heading_mae_deg|nonstop | +0.0000 [+0.0000, +0.0000] | 2184 |
| OF-PD | LATERAL operative_0_2s|heading_mae_deg|stop | -0.1995 [-0.4286, +0.0185] | 130 |
| OF-PD | LONGITUDINAL operative_0_2s|speed_mae_mps|all | +0.0133 [+0.0050, +0.0236]* | 2399 |
| OF-PD | LONGITUDINAL operative_0_2s|speed_mae_mps|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL operative_0_2s|speed_mae_mps|stop | +0.2442 [+0.1151, +0.3942]* | 131 |
| OF-PD | LATERAL operative_0_2s|yaw_rate_mae_degps|all | -0.2050 [-0.4147, -0.0745]* | 2305 |
| OF-PD | LATERAL operative_0_2s|yaw_rate_mae_degps|nonstop | +0.0000 [+0.0000, +0.0000] | 2175 |
| OF-PD | LATERAL operative_0_2s|yaw_rate_mae_degps|stop | -3.6345 [-6.7034, -1.7022]* | 130 |
| OF-PD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|all | -0.0061 [-0.0208, +0.0050] | 2399 |
| OF-PD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|stop | -0.1125 [-0.3463, +0.1018] | 131 |
| OF-PD | LONGITUDINAL tactical_2_6s|ade_m|all | +0.0045 [-0.0728, +0.0744] | 2399 |
| OF-PD | LONGITUDINAL tactical_2_6s|ade_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL tactical_2_6s|ade_m|stop | +0.0830 [-1.1598, +1.3461] | 131 |
| OF-PD | LONGITUDINAL tactical_2_6s|along_mae_m|all | +0.0202 [-0.0568, +0.0887] | 2399 |
| OF-PD | LONGITUDINAL tactical_2_6s|along_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL tactical_2_6s|along_mae_m|stop | +0.3691 [-0.9010, +1.6494] | 131 |
| OF-PD | LATERAL tactical_2_6s|cross_mae_m|all | -0.0157 [-0.0386, +0.0075] | 2399 |
| OF-PD | LATERAL tactical_2_6s|cross_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LATERAL tactical_2_6s|cross_mae_m|stop | -0.2866 [-0.6939, +0.1345] | 131 |
| OF-PD | LATERAL tactical_2_6s|curvature_mae_1pm|all | -0.0006 [-0.0014, +0.0000] | 2254 |
| OF-PD | LATERAL tactical_2_6s|curvature_mae_1pm|nonstop | +0.0000 [+0.0000, +0.0000] | 2161 |
| OF-PD | LATERAL tactical_2_6s|curvature_mae_1pm|stop | -0.0137 [-0.0302, +0.0001] | 93 |
| OF-PD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|all | +0.0516 [-0.0306, +0.1497] | 587 |
| OF-PD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|nonstop | +0.0000 [+0.0000, +0.0000] | 554 |
| OF-PD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|stop | +0.9185 [-0.6410, +2.1762] | 33 |
| OF-PD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|all | +0.0063 [-0.0063, +0.0211] | 530 |
| OF-PD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|nonstop | +0.0000 [+0.0000, +0.0000] | 497 |
| OF-PD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|stop | +0.1019 [-0.1175, +0.3293] | 33 |
| OF-PD | LONGITUDINAL tactical_2_6s|fde_m|all | -0.1415 [-0.3074, -0.0007]* | 2399 |
| OF-PD | LONGITUDINAL tactical_2_6s|fde_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL tactical_2_6s|fde_m|stop | -2.5911 [-5.0364, -0.2064]* | 131 |
| OF-PD | LATERAL tactical_2_6s|heading_mae_deg|all | -0.0851 [-0.1477, -0.0389]* | 2310 |
| OF-PD | LATERAL tactical_2_6s|heading_mae_deg|nonstop | +0.0000 [+0.0000, +0.0000] | 2211 |
| OF-PD | LATERAL tactical_2_6s|heading_mae_deg|stop | -1.9856 [-3.1853, -1.0467]* | 99 |
| OF-PD | LONGITUDINAL tactical_2_6s|speed_mae_mps|all | -0.0400 [-0.0791, -0.0067]* | 2399 |
| OF-PD | LONGITUDINAL tactical_2_6s|speed_mae_mps|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL tactical_2_6s|speed_mae_mps|stop | -0.7329 [-1.2713, -0.2019]* | 131 |
| OF-PD | LATERAL tactical_2_6s|yaw_rate_mae_degps|all | -0.0706 [-0.1344, -0.0242]* | 2254 |
| OF-PD | LATERAL tactical_2_6s|yaw_rate_mae_degps|nonstop | +0.0000 [+0.0000, +0.0000] | 2161 |
| OF-PD | LATERAL tactical_2_6s|yaw_rate_mae_degps|stop | -1.7118 [-3.0146, -0.6466]* | 93 |
| PF-OD | LONGITUDINAL OVERRUN_vs_logged_rest_m|stop | -1.8454 [-3.1052, -0.8061]* | 131 |
| PF-OD | LONGITUDINAL full_0_6s|accel_mae_mps2|all | -0.0004 [-0.0065, +0.0053] | 2399 |
| PF-OD | LONGITUDINAL full_0_6s|accel_mae_mps2|nonstop | +0.0033 [+0.0002, +0.0072]* | 2268 |
| PF-OD | LONGITUDINAL full_0_6s|accel_mae_mps2|stop | -0.0652 [-0.1418, -0.0028]* | 131 |
| PF-OD | LONGITUDINAL full_0_6s|ade_m|all | +0.0264 [-0.0102, +0.0653] | 2399 |
| PF-OD | LONGITUDINAL full_0_6s|ade_m|nonstop | +0.0523 [+0.0269, +0.0838]* | 2268 |
| PF-OD | LONGITUDINAL full_0_6s|ade_m|stop | -0.4206 [-0.7577, -0.1428]* | 131 |
| PF-OD | LONGITUDINAL full_0_6s|along_mae_m|all | +0.0322 [-0.0049, +0.0703] | 2399 |
| PF-OD | LONGITUDINAL full_0_6s|along_mae_m|nonstop | +0.0549 [+0.0293, +0.0874]* | 2268 |
| PF-OD | LONGITUDINAL full_0_6s|along_mae_m|stop | -0.3604 [-0.6917, -0.0954]* | 131 |
| PF-OD | LATERAL full_0_6s|cross_mae_m|all | -0.0030 [-0.0157, +0.0094] | 2399 |
| PF-OD | LATERAL full_0_6s|cross_mae_m|nonstop | +0.0053 [-0.0023, +0.0150] | 2268 |
| PF-OD | LATERAL full_0_6s|cross_mae_m|stop | -0.1462 [-0.3001, -0.0250]* | 131 |
| PF-OD | LATERAL full_0_6s|curvature_mae_1pm|all | -0.0012 [-0.0044, +0.0019] | 2325 |
| PF-OD | LATERAL full_0_6s|curvature_mae_1pm|nonstop | -0.0021 [-0.0048, +0.0002] | 2195 |
| PF-OD | LATERAL full_0_6s|curvature_mae_1pm|stop | +0.0135 [-0.0079, +0.0464] | 130 |
| PF-OD | LONGITUDINAL full_0_6s|fde_m|all | +0.0580 [-0.0652, +0.1922] | 2399 |
| PF-OD | LONGITUDINAL full_0_6s|fde_m|nonstop | +0.1476 [+0.0623, +0.2616]* | 2268 |
| PF-OD | LONGITUDINAL full_0_6s|fde_m|stop | -1.4920 [-2.6578, -0.5578]* | 131 |
| PF-OD | LATERAL full_0_6s|heading_mae_deg|all | -0.1511 [-0.2949, -0.0479]* | 2370 |
| PF-OD | LATERAL full_0_6s|heading_mae_deg|nonstop | -0.1295 [-0.2828, -0.0308]* | 2240 |
| PF-OD | LATERAL full_0_6s|heading_mae_deg|stop | -0.5233 [-1.1755, -0.0565]* | 130 |
| PF-OD | LONGITUDINAL full_0_6s|speed_mae_mps|all | +0.0144 [-0.0073, +0.0391] | 2399 |
| PF-OD | LONGITUDINAL full_0_6s|speed_mae_mps|nonstop | +0.0295 [+0.0140, +0.0506]* | 2268 |
| PF-OD | LONGITUDINAL full_0_6s|speed_mae_mps|stop | -0.2485 [-0.4433, -0.0924]* | 131 |
| PF-OD | LATERAL full_0_6s|yaw_rate_mae_degps|all | -0.0103 [-0.2058, +0.2019] | 2325 |
| PF-OD | LATERAL full_0_6s|yaw_rate_mae_degps|nonstop | -0.0775 [-0.2248, +0.0645] | 2195 |
| PF-OD | LATERAL full_0_6s|yaw_rate_mae_degps|stop | +1.1240 [-0.6294, +3.9513] | 130 |
| PF-OD | LONGITUDINAL operative_0_2s|accel_mae_mps2|all | +0.0045 [-0.0011, +0.0097] | 2399 |
| PF-OD | LONGITUDINAL operative_0_2s|accel_mae_mps2|nonstop | +0.0087 [+0.0047, +0.0133]* | 2268 |
| PF-OD | LONGITUDINAL operative_0_2s|accel_mae_mps2|stop | -0.0682 [-0.1428, -0.0129]* | 131 |
| PF-OD | LONGITUDINAL operative_0_2s|ade_m|all | +0.0036 [+0.0010, +0.0064]* | 2399 |
| PF-OD | LONGITUDINAL operative_0_2s|ade_m|nonstop | +0.0044 [+0.0017, +0.0073]* | 2268 |
| PF-OD | LONGITUDINAL operative_0_2s|ade_m|stop | -0.0104 [-0.0250, -0.0005]* | 131 |
| PF-OD | LONGITUDINAL operative_0_2s|along_mae_m|all | +0.0040 [+0.0013, +0.0068]* | 2399 |
| PF-OD | LONGITUDINAL operative_0_2s|along_mae_m|nonstop | +0.0047 [+0.0020, +0.0077]* | 2268 |
| PF-OD | LONGITUDINAL operative_0_2s|along_mae_m|stop | -0.0088 [-0.0216, -0.0001]* | 131 |
| PF-OD | LATERAL operative_0_2s|cross_mae_m|all | -0.0004 [-0.0009, +0.0000] | 2399 |
| PF-OD | LATERAL operative_0_2s|cross_mae_m|nonstop | -0.0003 [-0.0008, +0.0001] | 2268 |
| PF-OD | LATERAL operative_0_2s|cross_mae_m|stop | -0.0026 [-0.0072, +0.0002] | 131 |
| PF-OD | LATERAL operative_0_2s|curvature_mae_1pm|all | -0.0043 [-0.0083, -0.0013]* | 2305 |
| PF-OD | LATERAL operative_0_2s|curvature_mae_1pm|nonstop | -0.0046 [-0.0093, -0.0014]* | 2175 |
| PF-OD | LATERAL operative_0_2s|curvature_mae_1pm|stop | +0.0018 [-0.0064, +0.0114] | 130 |
| PF-OD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|all | -0.0010 [-0.0068, +0.0031] | 705 |
| PF-OD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|nonstop | -0.0004 [-0.0065, +0.0040] | 658 |
| PF-OD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|stop | -0.0094 [-0.0278, +0.0014] | 47 |
| PF-OD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|all | -0.0002 [-0.0013, +0.0005] | 647 |
| PF-OD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|nonstop | -0.0000 [-0.0013, +0.0009] | 601 |
| PF-OD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|stop | -0.0024 [-0.0070, +0.0002] | 46 |
| PF-OD | LONGITUDINAL operative_0_2s|fde_m|all | +0.0123 [+0.0049, +0.0203]* | 2399 |
| PF-OD | LONGITUDINAL operative_0_2s|fde_m|nonstop | +0.0153 [+0.0081, +0.0235]* | 2268 |
| PF-OD | LONGITUDINAL operative_0_2s|fde_m|stop | -0.0397 [-0.0845, -0.0059]* | 131 |
| PF-OD | LATERAL operative_0_2s|heading_mae_deg|all | -0.0315 [-0.0610, -0.0107]* | 2314 |
| PF-OD | LATERAL operative_0_2s|heading_mae_deg|nonstop | -0.0247 [-0.0503, -0.0075]* | 2184 |
| PF-OD | LATERAL operative_0_2s|heading_mae_deg|stop | -0.1460 [-0.3199, -0.0307]* | 130 |
| PF-OD | LONGITUDINAL operative_0_2s|speed_mae_mps|all | +0.0059 [+0.0023, +0.0098]* | 2399 |
| PF-OD | LONGITUDINAL operative_0_2s|speed_mae_mps|nonstop | +0.0076 [+0.0042, +0.0115]* | 2268 |
| PF-OD | LONGITUDINAL operative_0_2s|speed_mae_mps|stop | -0.0239 [-0.0489, -0.0038]* | 131 |
| PF-OD | LATERAL operative_0_2s|yaw_rate_mae_degps|all | -0.3388 [-0.6618, -0.1125]* | 2305 |
| PF-OD | LATERAL operative_0_2s|yaw_rate_mae_degps|nonstop | -0.3096 [-0.5994, -0.1129]* | 2175 |
| PF-OD | LATERAL operative_0_2s|yaw_rate_mae_degps|stop | -0.8267 [-1.7955, +0.0240] | 130 |
| PF-OD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|all | -0.0035 [-0.0111, +0.0032] | 2399 |
| PF-OD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|nonstop | +0.0002 [-0.0039, +0.0048] | 2268 |
| PF-OD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|stop | -0.0677 [-0.1571, +0.0049] | 131 |
| PF-OD | LONGITUDINAL tactical_2_6s|ade_m|all | +0.0378 [-0.0165, +0.0949] | 2399 |
| PF-OD | LONGITUDINAL tactical_2_6s|ade_m|nonstop | +0.0762 [+0.0388, +0.1235]* | 2268 |
| PF-OD | LONGITUDINAL tactical_2_6s|ade_m|stop | -0.6257 [-1.1249, -0.2138]* | 131 |
| PF-OD | LONGITUDINAL tactical_2_6s|along_mae_m|all | +0.0463 [-0.0077, +0.1022] | 2399 |
| PF-OD | LONGITUDINAL tactical_2_6s|along_mae_m|nonstop | +0.0800 [+0.0421, +0.1284]* | 2268 |
| PF-OD | LONGITUDINAL tactical_2_6s|along_mae_m|stop | -0.5361 [-1.0275, -0.1427]* | 131 |
| PF-OD | LATERAL tactical_2_6s|cross_mae_m|all | -0.0042 [-0.0234, +0.0146] | 2399 |
| PF-OD | LATERAL tactical_2_6s|cross_mae_m|nonstop | +0.0081 [-0.0034, +0.0226] | 2268 |
| PF-OD | LATERAL tactical_2_6s|cross_mae_m|stop | -0.2180 [-0.4471, -0.0374]* | 131 |
| PF-OD | LATERAL tactical_2_6s|curvature_mae_1pm|all | -0.0005 [-0.0017, +0.0004] | 2266 |
| PF-OD | LATERAL tactical_2_6s|curvature_mae_1pm|nonstop | -0.0000 [-0.0001, +0.0001] | 2145 |
| PF-OD | LATERAL tactical_2_6s|curvature_mae_1pm|stop | -0.0081 [-0.0300, +0.0080] | 121 |
| PF-OD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|all | -0.0198 [-0.0578, +0.0061] | 588 |
| PF-OD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|nonstop | +0.0027 [-0.0077, +0.0127] | 554 |
| PF-OD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|stop | -0.3876 [-0.8323, +0.0000] | 34 |
| PF-OD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|all | -0.0024 [-0.0098, +0.0034] | 530 |
| PF-OD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|nonstop | +0.0019 [-0.0009, +0.0060] | 497 |
| PF-OD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|stop | -0.0681 [-0.1451, +0.0000] | 33 |
| PF-OD | LONGITUDINAL tactical_2_6s|fde_m|all | +0.0580 [-0.0652, +0.1922] | 2399 |
| PF-OD | LONGITUDINAL tactical_2_6s|fde_m|nonstop | +0.1476 [+0.0623, +0.2616]* | 2268 |
| PF-OD | LONGITUDINAL tactical_2_6s|fde_m|stop | -1.4920 [-2.6578, -0.5578]* | 131 |
| PF-OD | LATERAL tactical_2_6s|heading_mae_deg|all | -0.0565 [-0.1533, +0.0083] | 2320 |
| PF-OD | LATERAL tactical_2_6s|heading_mae_deg|nonstop | -0.0648 [-0.1655, -0.0063]* | 2196 |
| PF-OD | LATERAL tactical_2_6s|heading_mae_deg|stop | +0.0899 [-0.3012, +0.5032] | 124 |
| PF-OD | LONGITUDINAL tactical_2_6s|speed_mae_mps|all | +0.0186 [-0.0126, +0.0542] | 2399 |
| PF-OD | LONGITUDINAL tactical_2_6s|speed_mae_mps|nonstop | +0.0405 [+0.0180, +0.0708]* | 2268 |
| PF-OD | LONGITUDINAL tactical_2_6s|speed_mae_mps|stop | -0.3609 [-0.6435, -0.1335]* | 131 |
| PF-OD | LATERAL tactical_2_6s|yaw_rate_mae_degps|all | -0.0611 [-0.1206, -0.0113]* | 2266 |
| PF-OD | LATERAL tactical_2_6s|yaw_rate_mae_degps|nonstop | -0.0176 [-0.0342, -0.0010]* | 2145 |
| PF-OD | LATERAL tactical_2_6s|yaw_rate_mae_degps|stop | -0.8316 [-1.7931, -0.0066]* | 121 |
| OF-OD | LONGITUDINAL OVERRUN_vs_logged_rest_m|stop | -9.6526 [-11.5661, -7.5234]* | 131 |
| OF-OD | LONGITUDINAL full_0_6s|accel_mae_mps2|all | -0.0111 [-0.0225, -0.0020]* | 2399 |
| OF-OD | LONGITUDINAL full_0_6s|accel_mae_mps2|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL full_0_6s|accel_mae_mps2|stop | -0.2029 [-0.3767, -0.0574]* | 131 |
| OF-OD | LONGITUDINAL full_0_6s|ade_m|all | -0.1288 [-0.1952, -0.0684]* | 2399 |
| OF-OD | LONGITUDINAL full_0_6s|ade_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL full_0_6s|ade_m|stop | -2.3583 [-2.9477, -1.7592]* | 131 |
| OF-OD | LONGITUDINAL full_0_6s|along_mae_m|all | -0.1203 [-0.1877, -0.0612]* | 2399 |
| OF-OD | LONGITUDINAL full_0_6s|along_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL full_0_6s|along_mae_m|stop | -2.2031 [-2.8640, -1.5585]* | 131 |
| OF-OD | LATERAL full_0_6s|cross_mae_m|all | -0.0277 [-0.0448, -0.0128]* | 2399 |
| OF-OD | LATERAL full_0_6s|cross_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LATERAL full_0_6s|cross_mae_m|stop | -0.5067 [-0.7460, -0.2865]* | 131 |
| OF-OD | LATERAL full_0_6s|curvature_mae_1pm|all | +0.0005 [-0.0008, +0.0024] | 2326 |
| OF-OD | LATERAL full_0_6s|curvature_mae_1pm|nonstop | +0.0000 [+0.0000, +0.0000] | 2196 |
| OF-OD | LATERAL full_0_6s|curvature_mae_1pm|stop | +0.0083 [-0.0132, +0.0405] | 130 |
| OF-OD | LONGITUDINAL full_0_6s|fde_m|all | -0.4626 [-0.6943, -0.2503]* | 2399 |
| OF-OD | LONGITUDINAL full_0_6s|fde_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL full_0_6s|fde_m|stop | -8.4709 [-10.4962, -6.3816]* | 131 |
| OF-OD | LATERAL full_0_6s|heading_mae_deg|all | -0.0766 [-0.1257, -0.0344]* | 2370 |
| OF-OD | LATERAL full_0_6s|heading_mae_deg|nonstop | +0.0000 [+0.0000, +0.0000] | 2240 |
| OF-OD | LATERAL full_0_6s|heading_mae_deg|stop | -1.3957 [-2.1085, -0.7707]* | 130 |
| OF-OD | LONGITUDINAL full_0_6s|speed_mae_mps|all | -0.0682 [-0.1044, -0.0359]* | 2399 |
| OF-OD | LONGITUDINAL full_0_6s|speed_mae_mps|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL full_0_6s|speed_mae_mps|stop | -1.2498 [-1.5914, -0.9129]* | 131 |
| OF-OD | LATERAL full_0_6s|yaw_rate_mae_degps|all | +0.0346 [-0.0659, +0.1988] | 2326 |
| OF-OD | LATERAL full_0_6s|yaw_rate_mae_degps|nonstop | +0.0000 [+0.0000, +0.0000] | 2196 |
| OF-OD | LATERAL full_0_6s|yaw_rate_mae_degps|stop | +0.6194 [-1.1550, +3.4113] | 130 |
| OF-OD | LONGITUDINAL operative_0_2s|accel_mae_mps2|all | -0.0059 [-0.0128, -0.0014]* | 2399 |
| OF-OD | LONGITUDINAL operative_0_2s|accel_mae_mps2|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL operative_0_2s|accel_mae_mps2|stop | -0.1084 [-0.2206, -0.0305]* | 131 |
| OF-OD | LONGITUDINAL operative_0_2s|ade_m|all | -0.0030 [-0.0055, -0.0010]* | 2399 |
| OF-OD | LONGITUDINAL operative_0_2s|ade_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL operative_0_2s|ade_m|stop | -0.0550 [-0.0903, -0.0238]* | 131 |
| OF-OD | LONGITUDINAL operative_0_2s|along_mae_m|all | -0.0030 [-0.0055, -0.0010]* | 2399 |
| OF-OD | LONGITUDINAL operative_0_2s|along_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL operative_0_2s|along_mae_m|stop | -0.0548 [-0.0910, -0.0233]* | 131 |
| OF-OD | LATERAL operative_0_2s|cross_mae_m|all | -0.0003 [-0.0008, -0.0001]* | 2399 |
| OF-OD | LATERAL operative_0_2s|cross_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LATERAL operative_0_2s|cross_mae_m|stop | -0.0063 [-0.0128, -0.0016]* | 131 |
| OF-OD | LATERAL operative_0_2s|curvature_mae_1pm|all | -0.0001 [-0.0006, +0.0005] | 2305 |
| OF-OD | LATERAL operative_0_2s|curvature_mae_1pm|nonstop | +0.0000 [+0.0000, +0.0000] | 2175 |
| OF-OD | LATERAL operative_0_2s|curvature_mae_1pm|stop | -0.0016 [-0.0102, +0.0089] | 130 |
| OF-OD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|all | -0.0035 [-0.0076, -0.0005]* | 706 |
| OF-OD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|nonstop | +0.0000 [+0.0000, +0.0000] | 659 |
| OF-OD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|stop | -0.0529 [-0.0986, -0.0149]* | 47 |
| OF-OD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|all | -0.0007 [-0.0014, -0.0001]* | 648 |
| OF-OD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|nonstop | +0.0000 [+0.0000, +0.0000] | 602 |
| OF-OD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|stop | -0.0097 [-0.0173, -0.0030]* | 46 |
| OF-OD | LONGITUDINAL operative_0_2s|fde_m|all | -0.0113 [-0.0198, -0.0043]* | 2399 |
| OF-OD | LONGITUDINAL operative_0_2s|fde_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL operative_0_2s|fde_m|stop | -0.2062 [-0.3263, -0.1017]* | 131 |
| OF-OD | LATERAL operative_0_2s|heading_mae_deg|all | -0.0111 [-0.0206, -0.0035]* | 2314 |
| OF-OD | LATERAL operative_0_2s|heading_mae_deg|nonstop | +0.0000 [+0.0000, +0.0000] | 2184 |
| OF-OD | LATERAL operative_0_2s|heading_mae_deg|stop | -0.1972 [-0.3666, -0.0789]* | 130 |
| OF-OD | LONGITUDINAL operative_0_2s|speed_mae_mps|all | -0.0055 [-0.0099, -0.0021]* | 2399 |
| OF-OD | LONGITUDINAL operative_0_2s|speed_mae_mps|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL operative_0_2s|speed_mae_mps|stop | -0.1006 [-0.1612, -0.0473]* | 131 |
| OF-OD | LATERAL operative_0_2s|yaw_rate_mae_degps|all | -0.0674 [-0.1326, -0.0171]* | 2305 |
| OF-OD | LATERAL operative_0_2s|yaw_rate_mae_degps|nonstop | +0.0000 [+0.0000, +0.0000] | 2175 |
| OF-OD | LATERAL operative_0_2s|yaw_rate_mae_degps|stop | -1.1959 [-2.1069, -0.3672]* | 130 |
| OF-OD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|all | -0.0143 [-0.0290, -0.0025]* | 2399 |
| OF-OD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|stop | -0.2619 [-0.4779, -0.0710]* | 131 |
| OF-OD | LONGITUDINAL tactical_2_6s|ade_m|all | -0.1917 [-0.2904, -0.1022]* | 2399 |
| OF-OD | LONGITUDINAL tactical_2_6s|ade_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL tactical_2_6s|ade_m|stop | -3.5099 [-4.3835, -2.6164]* | 131 |
| OF-OD | LONGITUDINAL tactical_2_6s|along_mae_m|all | -0.1790 [-0.2794, -0.0911]* | 2399 |
| OF-OD | LONGITUDINAL tactical_2_6s|along_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL tactical_2_6s|along_mae_m|stop | -3.2773 [-4.2469, -2.3180]* | 131 |
| OF-OD | LATERAL tactical_2_6s|cross_mae_m|all | -0.0413 [-0.0670, -0.0192]* | 2399 |
| OF-OD | LATERAL tactical_2_6s|cross_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LATERAL tactical_2_6s|cross_mae_m|stop | -0.7568 [-1.1130, -0.4280]* | 131 |
| OF-OD | LATERAL tactical_2_6s|curvature_mae_1pm|all | -0.0005 [-0.0018, +0.0003] | 2278 |
| OF-OD | LATERAL tactical_2_6s|curvature_mae_1pm|nonstop | +0.0000 [+0.0000, +0.0000] | 2161 |
| OF-OD | LATERAL tactical_2_6s|curvature_mae_1pm|stop | -0.0107 [-0.0330, +0.0059] | 117 |
| OF-OD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|all | -0.1841 [-0.3648, -0.0533]* | 588 |
| OF-OD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|nonstop | +0.0000 [+0.0000, +0.0000] | 554 |
| OF-OD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|stop | -3.1837 [-4.3575, -1.7976]* | 34 |
| OF-OD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|all | -0.0332 [-0.0604, -0.0117]* | 530 |
| OF-OD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|nonstop | +0.0000 [+0.0000, +0.0000] | 497 |
| OF-OD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|stop | -0.5330 [-0.6783, -0.3718]* | 33 |
| OF-OD | LONGITUDINAL tactical_2_6s|fde_m|all | -0.4626 [-0.6943, -0.2503]* | 2399 |
| OF-OD | LONGITUDINAL tactical_2_6s|fde_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL tactical_2_6s|fde_m|stop | -8.4709 [-10.4962, -6.3816]* | 131 |
| OF-OD | LATERAL tactical_2_6s|heading_mae_deg|all | -0.0566 [-0.1120, -0.0122]* | 2331 |
| OF-OD | LATERAL tactical_2_6s|heading_mae_deg|nonstop | +0.0000 [+0.0000, +0.0000] | 2211 |
| OF-OD | LATERAL tactical_2_6s|heading_mae_deg|stop | -1.0999 [-1.9892, -0.2857]* | 120 |
| OF-OD | LONGITUDINAL tactical_2_6s|speed_mae_mps|all | -0.0996 [-0.1518, -0.0532]* | 2399 |
| OF-OD | LONGITUDINAL tactical_2_6s|speed_mae_mps|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL tactical_2_6s|speed_mae_mps|stop | -1.8245 [-2.3102, -1.3403]* | 131 |
| OF-OD | LATERAL tactical_2_6s|yaw_rate_mae_degps|all | -0.0979 [-0.1722, -0.0364]* | 2278 |
| OF-OD | LATERAL tactical_2_6s|yaw_rate_mae_degps|nonstop | +0.0000 [+0.0000, +0.0000] | 2161 |
| OF-OD | LATERAL tactical_2_6s|yaw_rate_mae_degps|stop | -1.9055 [-2.9869, -0.8797]* | 117 |

PF-PD point deltas at the other fold seeds (spread, no interval): fs1 stop +0.028 / non-stop +0.046; fs2 stop +0.072 / non-stop +0.041; fs3 stop +0.158 / non-stop +0.048; fs4 stop +0.139 / non-stop +0.045

### Plan seed 1: **CUE-PROBE-COSTS**

| bar | measured | result |
|---|---|---|
| stop-window 2-6 s ADE change negative, CI excludes 0 | +0.0860 m [-0.4514, +0.6069] (n 131) | FAIL |
| size >= 0.30 of the ORACLE gain at this seed | 0.026 (ORACLE -3.3736 m) | FAIL |
| non-stop 0-6 s ADE change, upper CI end <= +0.05 m | +0.0487 m [+0.0189, +0.0825] (n 2268) | FAIL |

| arm | stop 2-6 s ADE | non-stop 0-6 s ADE | all 0-6 s ADE | overrun share (stop, NULL) | overrun mean m | hit rate | FP rate | active | TP/FP/FN/TN | lon label changed |
|---|---|---|---|---|---|---|---|---|---|---|
| PF-PD | +0.0860 [-0.4514, +0.6069] | +0.0487 [+0.0189, +0.0825]* | +0.0501 [+0.0134, +0.0890]* | 0.664 (0.847) | 7.383 | 0.267 | 0.0215 | 83 | 35/48/96/2186 | 0.008 |
| OF-PD | +0.0600 [-1.2530, +1.3670] | +0.0000 [+0.0000, +0.0000] | +0.0044 [-0.0501, +0.0535] | 0.092 (0.847) | 0.336 | 0.267 | 0.0215 | 83 | 35/48/96/2186 | 0.012 |
| PF-OD | -0.6325 [-1.1129, -0.2257]* | +0.0487 [+0.0189, +0.0825]* | +0.0228 [-0.0168, +0.0638] | 0.664 (0.847) | 7.383 | 0.267 | 0.0215 | 83 | 35/48/96/2186 | 0.008 |
| OF-OD | -3.3736 [-4.3290, -2.3353]* | +0.0000 [+0.0000, +0.0000] | -0.1239 [-0.1936, -0.0633]* | 0.015 (0.847) | 0.015 | 0.267 | 0.0215 | 83 | 35/48/96/2186 | 0.012 |

Secondary (not a card outcome): tactical rates at the inner cross-fitted threshold: {'hit_rate': 0.3969465648854962, 'fp_rate': 0.03804834377797672, 'active': 137}.

All cells (family; variant minus NULL; * = CI excludes 0):

| arm | cell | delta [lo, hi] | n |
|---|---|---|---|
| PF-PD | LONGITUDINAL OVERRUN_vs_logged_rest_m|stop | -1.7324 [-2.9289, -0.6706]* | 131 |
| PF-PD | LONGITUDINAL full_0_6s|accel_mae_mps2|all | +0.0005 [-0.0051, +0.0059] | 2399 |
| PF-PD | LONGITUDINAL full_0_6s|accel_mae_mps2|nonstop | +0.0026 [-0.0007, +0.0066] | 2268 |
| PF-PD | LONGITUDINAL full_0_6s|accel_mae_mps2|stop | -0.0356 [-0.1051, +0.0279] | 131 |
| PF-PD | LONGITUDINAL full_0_6s|ade_m|all | +0.0501 [+0.0134, +0.0890]* | 2399 |
| PF-PD | LONGITUDINAL full_0_6s|ade_m|nonstop | +0.0487 [+0.0189, +0.0825]* | 2268 |
| PF-PD | LONGITUDINAL full_0_6s|ade_m|stop | +0.0737 [-0.2935, +0.4306] | 131 |
| PF-PD | LONGITUDINAL full_0_6s|along_mae_m|all | +0.0536 [+0.0148, +0.0962]* | 2399 |
| PF-PD | LONGITUDINAL full_0_6s|along_mae_m|nonstop | +0.0508 [+0.0185, +0.0891]* | 2268 |
| PF-PD | LONGITUDINAL full_0_6s|along_mae_m|stop | +0.1019 [-0.2694, +0.4658] | 131 |
| PF-PD | LATERAL full_0_6s|cross_mae_m|all | +0.0011 [-0.0069, +0.0095] | 2399 |
| PF-PD | LATERAL full_0_6s|cross_mae_m|nonstop | +0.0045 [-0.0019, +0.0125] | 2268 |
| PF-PD | LATERAL full_0_6s|cross_mae_m|stop | -0.0575 [-0.1333, +0.0021] | 131 |
| PF-PD | LATERAL full_0_6s|curvature_mae_1pm|all | -0.0034 [-0.0068, -0.0003]* | 2322 |
| PF-PD | LATERAL full_0_6s|curvature_mae_1pm|nonstop | -0.0016 [-0.0044, +0.0008] | 2192 |
| PF-PD | LATERAL full_0_6s|curvature_mae_1pm|stop | -0.0327 [-0.0786, -0.0075]* | 130 |
| PF-PD | LONGITUDINAL full_0_6s|fde_m|all | +0.1113 [-0.0104, +0.2429] | 2399 |
| PF-PD | LONGITUDINAL full_0_6s|fde_m|nonstop | +0.1468 [+0.0504, +0.2630]* | 2268 |
| PF-PD | LONGITUDINAL full_0_6s|fde_m|stop | -0.5039 [-1.6515, +0.5483] | 131 |
| PF-PD | LATERAL full_0_6s|heading_mae_deg|all | -0.1532 [-0.2546, -0.0684]* | 2360 |
| PF-PD | LATERAL full_0_6s|heading_mae_deg|nonstop | -0.1100 [-0.2044, -0.0438]* | 2230 |
| PF-PD | LATERAL full_0_6s|heading_mae_deg|stop | -0.8944 [-1.8570, -0.2169]* | 130 |
| PF-PD | LONGITUDINAL full_0_6s|speed_mae_mps|all | +0.0219 [+0.0018, +0.0449]* | 2399 |
| PF-PD | LONGITUDINAL full_0_6s|speed_mae_mps|nonstop | +0.0296 [+0.0138, +0.0498]* | 2268 |
| PF-PD | LONGITUDINAL full_0_6s|speed_mae_mps|stop | -0.1115 [-0.2969, +0.0499] | 131 |
| PF-PD | LATERAL full_0_6s|yaw_rate_mae_degps|all | -0.1253 [-0.3483, +0.0571] | 2322 |
| PF-PD | LATERAL full_0_6s|yaw_rate_mae_degps|nonstop | -0.0652 [-0.2096, +0.0788] | 2192 |
| PF-PD | LATERAL full_0_6s|yaw_rate_mae_degps|stop | -1.1401 [-4.6921, +0.9226] | 130 |
| PF-PD | LONGITUDINAL operative_0_2s|accel_mae_mps2|all | +0.0097 [+0.0047, +0.0151]* | 2399 |
| PF-PD | LONGITUDINAL operative_0_2s|accel_mae_mps2|nonstop | +0.0092 [+0.0047, +0.0144]* | 2268 |
| PF-PD | LONGITUDINAL operative_0_2s|accel_mae_mps2|stop | +0.0197 [-0.0315, +0.0691] | 131 |
| PF-PD | LONGITUDINAL operative_0_2s|ade_m|all | +0.0066 [+0.0024, +0.0107]* | 2399 |
| PF-PD | LONGITUDINAL operative_0_2s|ade_m|nonstop | +0.0041 [+0.0004, +0.0076]* | 2268 |
| PF-PD | LONGITUDINAL operative_0_2s|ade_m|stop | +0.0491 [+0.0162, +0.0848]* | 131 |
| PF-PD | LONGITUDINAL operative_0_2s|along_mae_m|all | +0.0069 [+0.0026, +0.0113]* | 2399 |
| PF-PD | LONGITUDINAL operative_0_2s|along_mae_m|nonstop | +0.0045 [+0.0006, +0.0083]* | 2268 |
| PF-PD | LONGITUDINAL operative_0_2s|along_mae_m|stop | +0.0495 [+0.0166, +0.0855]* | 131 |
| PF-PD | LATERAL operative_0_2s|cross_mae_m|all | -0.0005 [-0.0013, +0.0002] | 2399 |
| PF-PD | LATERAL operative_0_2s|cross_mae_m|nonstop | -0.0006 [-0.0014, -0.0000]* | 2268 |
| PF-PD | LATERAL operative_0_2s|cross_mae_m|stop | +0.0020 [-0.0060, +0.0111] | 131 |
| PF-PD | LATERAL operative_0_2s|curvature_mae_1pm|all | -0.0064 [-0.0128, -0.0018]* | 2303 |
| PF-PD | LATERAL operative_0_2s|curvature_mae_1pm|nonstop | -0.0043 [-0.0088, -0.0012]* | 2173 |
| PF-PD | LATERAL operative_0_2s|curvature_mae_1pm|stop | -0.0406 [-0.0978, -0.0040]* | 130 |
| PF-PD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|all | -0.0003 [-0.0118, +0.0110] | 700 |
| PF-PD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|nonstop | -0.0043 [-0.0129, +0.0014] | 653 |
| PF-PD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|stop | +0.0553 [-0.0429, +0.1447] | 47 |
| PF-PD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|all | -0.0012 [-0.0053, +0.0019] | 647 |
| PF-PD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|nonstop | -0.0013 [-0.0042, +0.0005] | 601 |
| PF-PD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|stop | -0.0005 [-0.0397, +0.0217] | 46 |
| PF-PD | LONGITUDINAL operative_0_2s|fde_m|all | +0.0218 [+0.0110, +0.0332]* | 2399 |
| PF-PD | LONGITUDINAL operative_0_2s|fde_m|nonstop | +0.0139 [+0.0042, +0.0236]* | 2268 |
| PF-PD | LONGITUDINAL operative_0_2s|fde_m|stop | +0.1600 [+0.0699, +0.2576]* | 131 |
| PF-PD | LATERAL operative_0_2s|heading_mae_deg|all | -0.0447 [-0.0944, -0.0111]* | 2314 |
| PF-PD | LATERAL operative_0_2s|heading_mae_deg|nonstop | -0.0266 [-0.0534, -0.0091]* | 2184 |
| PF-PD | LATERAL operative_0_2s|heading_mae_deg|stop | -0.3475 [-0.8551, +0.0170] | 130 |
| PF-PD | LONGITUDINAL operative_0_2s|speed_mae_mps|all | +0.0110 [+0.0058, +0.0165]* | 2399 |
| PF-PD | LONGITUDINAL operative_0_2s|speed_mae_mps|nonstop | +0.0072 [+0.0027, +0.0121]* | 2268 |
| PF-PD | LONGITUDINAL operative_0_2s|speed_mae_mps|stop | +0.0767 [+0.0327, +0.1226]* | 131 |
| PF-PD | LATERAL operative_0_2s|yaw_rate_mae_degps|all | -0.5051 [-0.9655, -0.1821]* | 2303 |
| PF-PD | LATERAL operative_0_2s|yaw_rate_mae_degps|nonstop | -0.3055 [-0.5891, -0.1134]* | 2173 |
| PF-PD | LATERAL operative_0_2s|yaw_rate_mae_degps|stop | -3.8419 [-8.3251, -0.8587]* | 130 |
| PF-PD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|all | -0.0047 [-0.0120, +0.0019] | 2399 |
| PF-PD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|nonstop | -0.0012 [-0.0056, +0.0036] | 2268 |
| PF-PD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|stop | -0.0658 [-0.1491, +0.0102] | 131 |
| PF-PD | LONGITUDINAL tactical_2_6s|ade_m|all | +0.0718 [+0.0181, +0.1290]* | 2399 |
| PF-PD | LONGITUDINAL tactical_2_6s|ade_m|nonstop | +0.0710 [+0.0276, +0.1203]* | 2268 |
| PF-PD | LONGITUDINAL tactical_2_6s|ade_m|stop | +0.0860 [-0.4514, +0.6069] | 131 |
| PF-PD | LONGITUDINAL tactical_2_6s|along_mae_m|all | +0.0770 [+0.0207, +0.1390]* | 2399 |
| PF-PD | LONGITUDINAL tactical_2_6s|along_mae_m|nonstop | +0.0740 [+0.0266, +0.1299]* | 2268 |
| PF-PD | LONGITUDINAL tactical_2_6s|along_mae_m|stop | +0.1281 [-0.4162, +0.6635] | 131 |
| PF-PD | LATERAL tactical_2_6s|cross_mae_m|all | +0.0019 [-0.0101, +0.0145] | 2399 |
| PF-PD | LATERAL tactical_2_6s|cross_mae_m|nonstop | +0.0071 [-0.0027, +0.0192] | 2268 |
| PF-PD | LATERAL tactical_2_6s|cross_mae_m|stop | -0.0873 [-0.1984, -0.0001]* | 131 |
| PF-PD | LATERAL tactical_2_6s|curvature_mae_1pm|all | -0.0007 [-0.0018, -0.0000]* | 2247 |
| PF-PD | LATERAL tactical_2_6s|curvature_mae_1pm|nonstop | -0.0005 [-0.0014, +0.0001] | 2137 |
| PF-PD | LATERAL tactical_2_6s|curvature_mae_1pm|stop | -0.0060 [-0.0153, +0.0010] | 110 |
| PF-PD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|all | -0.0234 [-0.0676, +0.0091] | 589 |
| PF-PD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|nonstop | +0.0006 [-0.0154, +0.0217] | 553 |
| PF-PD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|stop | -0.3923 [-0.7890, +0.0000] | 36 |
| PF-PD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|all | -0.0058 [-0.0208, +0.0079] | 531 |
| PF-PD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|nonstop | +0.0006 [-0.0085, +0.0121] | 496 |
| PF-PD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|stop | -0.0955 [-0.2010, +0.0000] | 35 |
| PF-PD | LONGITUDINAL tactical_2_6s|fde_m|all | +0.1113 [-0.0104, +0.2429] | 2399 |
| PF-PD | LONGITUDINAL tactical_2_6s|fde_m|nonstop | +0.1468 [+0.0504, +0.2630]* | 2268 |
| PF-PD | LONGITUDINAL tactical_2_6s|fde_m|stop | -0.5039 [-1.6515, +0.5483] | 131 |
| PF-PD | LATERAL tactical_2_6s|heading_mae_deg|all | -0.0866 [-0.1849, -0.0228]* | 2302 |
| PF-PD | LATERAL tactical_2_6s|heading_mae_deg|nonstop | -0.0798 [-0.1826, -0.0187]* | 2185 |
| PF-PD | LATERAL tactical_2_6s|heading_mae_deg|stop | -0.2138 [-0.5752, +0.1658] | 117 |
| PF-PD | LONGITUDINAL tactical_2_6s|speed_mae_mps|all | +0.0274 [-0.0019, +0.0609] | 2399 |
| PF-PD | LONGITUDINAL tactical_2_6s|speed_mae_mps|nonstop | +0.0408 [+0.0182, +0.0699]* | 2268 |
| PF-PD | LONGITUDINAL tactical_2_6s|speed_mae_mps|stop | -0.2056 [-0.4735, +0.0236] | 131 |
| PF-PD | LATERAL tactical_2_6s|yaw_rate_mae_degps|all | -0.0307 [-0.0785, +0.0067] | 2247 |
| PF-PD | LATERAL tactical_2_6s|yaw_rate_mae_degps|nonstop | -0.0288 [-0.0757, +0.0022] | 2137 |
| PF-PD | LATERAL tactical_2_6s|yaw_rate_mae_degps|stop | -0.0681 [-0.4519, +0.3241] | 110 |
| OF-PD | LONGITUDINAL OVERRUN_vs_logged_rest_m|stop | -8.7799 [-10.9269, -6.3907]* | 131 |
| OF-PD | LONGITUDINAL full_0_6s|accel_mae_mps2|all | +0.0042 [-0.0073, +0.0139] | 2399 |
| OF-PD | LONGITUDINAL full_0_6s|accel_mae_mps2|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL full_0_6s|accel_mae_mps2|stop | +0.0771 [-0.1248, +0.2605] | 131 |
| OF-PD | LONGITUDINAL full_0_6s|ade_m|all | +0.0044 [-0.0501, +0.0535] | 2399 |
| OF-PD | LONGITUDINAL full_0_6s|ade_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL full_0_6s|ade_m|stop | +0.0799 [-0.8169, +0.9838] | 131 |
| OF-PD | LONGITUDINAL full_0_6s|along_mae_m|all | +0.0141 [-0.0401, +0.0634] | 2399 |
| OF-PD | LONGITUDINAL full_0_6s|along_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL full_0_6s|along_mae_m|stop | +0.2581 [-0.6456, +1.1569] | 131 |
| OF-PD | LATERAL full_0_6s|cross_mae_m|all | -0.0152 [-0.0339, +0.0009] | 2399 |
| OF-PD | LATERAL full_0_6s|cross_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LATERAL full_0_6s|cross_mae_m|stop | -0.2787 [-0.5852, +0.0036] | 131 |
| OF-PD | LATERAL full_0_6s|curvature_mae_1pm|all | -0.0027 [-0.0051, -0.0009]* | 2323 |
| OF-PD | LATERAL full_0_6s|curvature_mae_1pm|nonstop | +0.0000 [+0.0000, +0.0000] | 2193 |
| OF-PD | LATERAL full_0_6s|curvature_mae_1pm|stop | -0.0476 [-0.0945, -0.0198]* | 130 |
| OF-PD | LONGITUDINAL full_0_6s|fde_m|all | -0.1332 [-0.3078, +0.0178] | 2399 |
| OF-PD | LONGITUDINAL full_0_6s|fde_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL full_0_6s|fde_m|stop | -2.4388 [-5.0823, +0.2105] | 131 |
| OF-PD | LATERAL full_0_6s|heading_mae_deg|all | -0.1207 [-0.2052, -0.0561]* | 2360 |
| OF-PD | LATERAL full_0_6s|heading_mae_deg|nonstop | +0.0000 [+0.0000, +0.0000] | 2230 |
| OF-PD | LATERAL full_0_6s|heading_mae_deg|stop | -2.1918 [-3.4019, -1.2052]* | 130 |
| OF-PD | LONGITUDINAL full_0_6s|speed_mae_mps|all | -0.0201 [-0.0479, +0.0038] | 2399 |
| OF-PD | LONGITUDINAL full_0_6s|speed_mae_mps|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL full_0_6s|speed_mae_mps|stop | -0.3676 [-0.7916, +0.0552] | 131 |
| OF-PD | LATERAL full_0_6s|yaw_rate_mae_degps|all | -0.1171 [-0.2924, -0.0044]* | 2323 |
| OF-PD | LATERAL full_0_6s|yaw_rate_mae_degps|nonstop | +0.0000 [+0.0000, +0.0000] | 2193 |
| OF-PD | LATERAL full_0_6s|yaw_rate_mae_degps|stop | -2.0917 [-5.6311, +0.0076] | 130 |
| OF-PD | LONGITUDINAL operative_0_2s|accel_mae_mps2|all | +0.0168 [+0.0063, +0.0284]* | 2399 |
| OF-PD | LONGITUDINAL operative_0_2s|accel_mae_mps2|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL operative_0_2s|accel_mae_mps2|stop | +0.3069 [+0.1446, +0.4830]* | 131 |
| OF-PD | LONGITUDINAL operative_0_2s|ade_m|all | +0.0065 [+0.0011, +0.0129]* | 2399 |
| OF-PD | LONGITUDINAL operative_0_2s|ade_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL operative_0_2s|ade_m|stop | +0.1198 [+0.0318, +0.2214]* | 131 |
| OF-PD | LONGITUDINAL operative_0_2s|along_mae_m|all | +0.0065 [+0.0012, +0.0128]* | 2399 |
| OF-PD | LONGITUDINAL operative_0_2s|along_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL operative_0_2s|along_mae_m|stop | +0.1199 [+0.0333, +0.2180]* | 131 |
| OF-PD | LATERAL operative_0_2s|cross_mae_m|all | +0.0002 [-0.0010, +0.0020] | 2399 |
| OF-PD | LATERAL operative_0_2s|cross_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LATERAL operative_0_2s|cross_mae_m|stop | +0.0043 [-0.0179, +0.0352] | 131 |
| OF-PD | LATERAL operative_0_2s|curvature_mae_1pm|all | -0.0029 [-0.0059, -0.0006]* | 2303 |
| OF-PD | LATERAL operative_0_2s|curvature_mae_1pm|nonstop | +0.0000 [+0.0000, +0.0000] | 2173 |
| OF-PD | LATERAL operative_0_2s|curvature_mae_1pm|stop | -0.0510 [-0.1054, -0.0140]* | 130 |
| OF-PD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|all | +0.0204 [-0.0018, +0.0530] | 700 |
| OF-PD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|nonstop | +0.0000 [+0.0000, +0.0000] | 653 |
| OF-PD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|stop | +0.3042 [-0.0283, +0.7007] | 47 |
| OF-PD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|all | +0.0050 [-0.0005, +0.0124] | 647 |
| OF-PD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|nonstop | +0.0000 [+0.0000, +0.0000] | 601 |
| OF-PD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|stop | +0.0698 [-0.0049, +0.1612] | 46 |
| OF-PD | LONGITUDINAL operative_0_2s|fde_m|all | +0.0208 [+0.0051, +0.0396]* | 2399 |
| OF-PD | LONGITUDINAL operative_0_2s|fde_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL operative_0_2s|fde_m|stop | +0.3806 [+0.1238, +0.6842]* | 131 |
| OF-PD | LATERAL operative_0_2s|heading_mae_deg|all | -0.0241 [-0.0529, +0.0001] | 2314 |
| OF-PD | LATERAL operative_0_2s|heading_mae_deg|nonstop | +0.0000 [+0.0000, +0.0000] | 2184 |
| OF-PD | LATERAL operative_0_2s|heading_mae_deg|stop | -0.4299 [-0.9560, -0.0363]* | 130 |
| OF-PD | LONGITUDINAL operative_0_2s|speed_mae_mps|all | +0.0119 [+0.0042, +0.0214]* | 2399 |
| OF-PD | LONGITUDINAL operative_0_2s|speed_mae_mps|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL operative_0_2s|speed_mae_mps|stop | +0.2182 [+0.0958, +0.3656]* | 131 |
| OF-PD | LATERAL operative_0_2s|yaw_rate_mae_degps|all | -0.2868 [-0.5246, -0.1014]* | 2303 |
| OF-PD | LATERAL operative_0_2s|yaw_rate_mae_degps|nonstop | +0.0000 [+0.0000, +0.0000] | 2173 |
| OF-PD | LATERAL operative_0_2s|yaw_rate_mae_degps|stop | -5.0811 [-9.3803, -2.2337]* | 130 |
| OF-PD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|all | -0.0033 [-0.0176, +0.0083] | 2399 |
| OF-PD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|stop | -0.0603 [-0.2997, +0.1607] | 131 |
| OF-PD | LONGITUDINAL tactical_2_6s|ade_m|all | +0.0033 [-0.0770, +0.0756] | 2399 |
| OF-PD | LONGITUDINAL tactical_2_6s|ade_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL tactical_2_6s|ade_m|stop | +0.0600 [-1.2530, +1.3670] | 131 |
| OF-PD | LONGITUDINAL tactical_2_6s|along_mae_m|all | +0.0179 [-0.0609, +0.0892] | 2399 |
| OF-PD | LONGITUDINAL tactical_2_6s|along_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL tactical_2_6s|along_mae_m|stop | +0.3272 [-0.9865, +1.6369] | 131 |
| OF-PD | LATERAL tactical_2_6s|cross_mae_m|all | -0.0229 [-0.0506, +0.0005] | 2399 |
| OF-PD | LATERAL tactical_2_6s|cross_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LATERAL tactical_2_6s|cross_mae_m|stop | -0.4201 [-0.8758, -0.0106]* | 131 |
| OF-PD | LATERAL tactical_2_6s|curvature_mae_1pm|all | -0.0008 [-0.0017, -0.0002]* | 2249 |
| OF-PD | LATERAL tactical_2_6s|curvature_mae_1pm|nonstop | +0.0000 [+0.0000, +0.0000] | 2156 |
| OF-PD | LATERAL tactical_2_6s|curvature_mae_1pm|stop | -0.0202 [-0.0383, -0.0056]* | 93 |
| OF-PD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|all | +0.0473 [-0.0700, +0.1654] | 588 |
| OF-PD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|nonstop | +0.0000 [+0.0000, +0.0000] | 553 |
| OF-PD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|stop | +0.7939 [-1.3943, +2.3660] | 35 |
| OF-PD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|all | +0.0051 [-0.0158, +0.0261] | 531 |
| OF-PD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|nonstop | +0.0000 [+0.0000, +0.0000] | 496 |
| OF-PD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|stop | +0.0777 [-0.2891, +0.3738] | 35 |
| OF-PD | LONGITUDINAL tactical_2_6s|fde_m|all | -0.1332 [-0.3078, +0.0178] | 2399 |
| OF-PD | LONGITUDINAL tactical_2_6s|fde_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL tactical_2_6s|fde_m|stop | -2.4388 [-5.0823, +0.2105] | 131 |
| OF-PD | LATERAL tactical_2_6s|heading_mae_deg|all | -0.0765 [-0.1371, -0.0271]* | 2303 |
| OF-PD | LATERAL tactical_2_6s|heading_mae_deg|nonstop | +0.0000 [+0.0000, +0.0000] | 2202 |
| OF-PD | LATERAL tactical_2_6s|heading_mae_deg|stop | -1.7441 [-2.8827, -0.6681]* | 101 |
| OF-PD | LONGITUDINAL tactical_2_6s|speed_mae_mps|all | -0.0361 [-0.0766, -0.0014]* | 2399 |
| OF-PD | LONGITUDINAL tactical_2_6s|speed_mae_mps|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-PD | LONGITUDINAL tactical_2_6s|speed_mae_mps|stop | -0.6605 [-1.2459, -0.0625]* | 131 |
| OF-PD | LATERAL tactical_2_6s|yaw_rate_mae_degps|all | -0.0735 [-0.1411, -0.0232]* | 2249 |
| OF-PD | LATERAL tactical_2_6s|yaw_rate_mae_degps|nonstop | +0.0000 [+0.0000, +0.0000] | 2156 |
| OF-PD | LATERAL tactical_2_6s|yaw_rate_mae_degps|stop | -1.7762 [-3.2001, -0.5420]* | 93 |
| PF-OD | LONGITUDINAL OVERRUN_vs_logged_rest_m|stop | -1.7324 [-2.9289, -0.6706]* | 131 |
| PF-OD | LONGITUDINAL full_0_6s|accel_mae_mps2|all | -0.0002 [-0.0054, +0.0049] | 2399 |
| PF-OD | LONGITUDINAL full_0_6s|accel_mae_mps2|nonstop | +0.0026 [-0.0007, +0.0066] | 2268 |
| PF-OD | LONGITUDINAL full_0_6s|accel_mae_mps2|stop | -0.0476 [-0.1065, -0.0028]* | 131 |
| PF-OD | LONGITUDINAL full_0_6s|ade_m|all | +0.0228 [-0.0168, +0.0638] | 2399 |
| PF-OD | LONGITUDINAL full_0_6s|ade_m|nonstop | +0.0487 [+0.0189, +0.0825]* | 2268 |
| PF-OD | LONGITUDINAL full_0_6s|ade_m|stop | -0.4248 [-0.7475, -0.1508]* | 131 |
| PF-OD | LONGITUDINAL full_0_6s|along_mae_m|all | +0.0264 [-0.0152, +0.0710] | 2399 |
| PF-OD | LONGITUDINAL full_0_6s|along_mae_m|nonstop | +0.0508 [+0.0185, +0.0891]* | 2268 |
| PF-OD | LONGITUDINAL full_0_6s|along_mae_m|stop | -0.3971 [-0.7163, -0.1341]* | 131 |
| PF-OD | LATERAL full_0_6s|cross_mae_m|all | -0.0022 [-0.0125, +0.0074] | 2399 |
| PF-OD | LATERAL full_0_6s|cross_mae_m|nonstop | +0.0045 [-0.0019, +0.0125] | 2268 |
| PF-OD | LATERAL full_0_6s|cross_mae_m|stop | -0.1184 [-0.2437, -0.0191]* | 131 |
| PF-OD | LATERAL full_0_6s|curvature_mae_1pm|all | -0.0008 [-0.0041, +0.0023] | 2322 |
| PF-OD | LATERAL full_0_6s|curvature_mae_1pm|nonstop | -0.0016 [-0.0044, +0.0008] | 2192 |
| PF-OD | LATERAL full_0_6s|curvature_mae_1pm|stop | +0.0124 [-0.0066, +0.0419] | 130 |
| PF-OD | LONGITUDINAL full_0_6s|fde_m|all | +0.0575 [-0.0744, +0.1926] | 2399 |
| PF-OD | LONGITUDINAL full_0_6s|fde_m|nonstop | +0.1468 [+0.0504, +0.2630]* | 2268 |
| PF-OD | LONGITUDINAL full_0_6s|fde_m|stop | -1.4891 [-2.6048, -0.5314]* | 131 |
| PF-OD | LATERAL full_0_6s|heading_mae_deg|all | -0.1206 [-0.2106, -0.0529]* | 2360 |
| PF-OD | LATERAL full_0_6s|heading_mae_deg|nonstop | -0.1100 [-0.2044, -0.0438]* | 2230 |
| PF-OD | LATERAL full_0_6s|heading_mae_deg|stop | -0.3013 [-0.7121, +0.0203] | 130 |
| PF-OD | LONGITUDINAL full_0_6s|speed_mae_mps|all | +0.0153 [-0.0055, +0.0391] | 2399 |
| PF-OD | LONGITUDINAL full_0_6s|speed_mae_mps|nonstop | +0.0296 [+0.0138, +0.0498]* | 2268 |
| PF-OD | LONGITUDINAL full_0_6s|speed_mae_mps|stop | -0.2330 [-0.4110, -0.0831]* | 131 |
| PF-OD | LATERAL full_0_6s|yaw_rate_mae_degps|all | +0.0214 [-0.1688, +0.2383] | 2322 |
| PF-OD | LATERAL full_0_6s|yaw_rate_mae_degps|nonstop | -0.0652 [-0.2096, +0.0788] | 2192 |
| PF-OD | LATERAL full_0_6s|yaw_rate_mae_degps|stop | +1.4804 [-0.1391, +4.3667] | 130 |
| PF-OD | LONGITUDINAL operative_0_2s|accel_mae_mps2|all | +0.0063 [+0.0014, +0.0116]* | 2399 |
| PF-OD | LONGITUDINAL operative_0_2s|accel_mae_mps2|nonstop | +0.0092 [+0.0047, +0.0144]* | 2268 |
| PF-OD | LONGITUDINAL operative_0_2s|accel_mae_mps2|stop | -0.0427 [-0.0806, -0.0090]* | 131 |
| PF-OD | LONGITUDINAL operative_0_2s|ade_m|all | +0.0034 [-0.0001, +0.0067] | 2399 |
| PF-OD | LONGITUDINAL operative_0_2s|ade_m|nonstop | +0.0041 [+0.0004, +0.0076]* | 2268 |
| PF-OD | LONGITUDINAL operative_0_2s|ade_m|stop | -0.0095 [-0.0197, -0.0019]* | 131 |
| PF-OD | LONGITUDINAL operative_0_2s|along_mae_m|all | +0.0037 [+0.0001, +0.0073]* | 2399 |
| PF-OD | LONGITUDINAL operative_0_2s|along_mae_m|nonstop | +0.0045 [+0.0006, +0.0083]* | 2268 |
| PF-OD | LONGITUDINAL operative_0_2s|along_mae_m|stop | -0.0092 [-0.0190, -0.0015]* | 131 |
| PF-OD | LATERAL operative_0_2s|cross_mae_m|all | -0.0007 [-0.0015, -0.0001]* | 2399 |
| PF-OD | LATERAL operative_0_2s|cross_mae_m|nonstop | -0.0006 [-0.0014, -0.0000]* | 2268 |
| PF-OD | LATERAL operative_0_2s|cross_mae_m|stop | -0.0027 [-0.0072, -0.0001]* | 131 |
| PF-OD | LATERAL operative_0_2s|curvature_mae_1pm|all | -0.0043 [-0.0085, -0.0012]* | 2303 |
| PF-OD | LATERAL operative_0_2s|curvature_mae_1pm|nonstop | -0.0043 [-0.0088, -0.0012]* | 2173 |
| PF-OD | LATERAL operative_0_2s|curvature_mae_1pm|stop | -0.0035 [-0.0063, -0.0010]* | 130 |
| PF-OD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|all | -0.0051 [-0.0130, +0.0002] | 700 |
| PF-OD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|nonstop | -0.0043 [-0.0129, +0.0014] | 653 |
| PF-OD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|stop | -0.0168 [-0.0406, +0.0016] | 47 |
| PF-OD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|all | -0.0015 [-0.0041, +0.0001] | 647 |
| PF-OD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|nonstop | -0.0013 [-0.0042, +0.0005] | 601 |
| PF-OD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|stop | -0.0043 [-0.0102, +0.0002] | 46 |
| PF-OD | LONGITUDINAL operative_0_2s|fde_m|all | +0.0111 [+0.0018, +0.0206]* | 2399 |
| PF-OD | LONGITUDINAL operative_0_2s|fde_m|nonstop | +0.0139 [+0.0042, +0.0236]* | 2268 |
| PF-OD | LONGITUDINAL operative_0_2s|fde_m|stop | -0.0374 [-0.0700, -0.0104]* | 131 |
| PF-OD | LATERAL operative_0_2s|heading_mae_deg|all | -0.0273 [-0.0519, -0.0105]* | 2314 |
| PF-OD | LATERAL operative_0_2s|heading_mae_deg|nonstop | -0.0266 [-0.0534, -0.0091]* | 2184 |
| PF-OD | LATERAL operative_0_2s|heading_mae_deg|stop | -0.0384 [-0.0660, -0.0143]* | 130 |
| PF-OD | LONGITUDINAL operative_0_2s|speed_mae_mps|all | +0.0057 [+0.0012, +0.0103]* | 2399 |
| PF-OD | LONGITUDINAL operative_0_2s|speed_mae_mps|nonstop | +0.0072 [+0.0027, +0.0121]* | 2268 |
| PF-OD | LONGITUDINAL operative_0_2s|speed_mae_mps|stop | -0.0209 [-0.0377, -0.0073]* | 131 |
| PF-OD | LATERAL operative_0_2s|yaw_rate_mae_degps|all | -0.3279 [-0.6253, -0.1222]* | 2303 |
| PF-OD | LATERAL operative_0_2s|yaw_rate_mae_degps|nonstop | -0.3055 [-0.5891, -0.1134]* | 2173 |
| PF-OD | LATERAL operative_0_2s|yaw_rate_mae_degps|stop | -0.7025 [-1.3582, -0.2185]* | 130 |
| PF-OD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|all | -0.0041 [-0.0110, +0.0021] | 2399 |
| PF-OD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|nonstop | -0.0012 [-0.0056, +0.0036] | 2268 |
| PF-OD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|stop | -0.0535 [-0.1307, +0.0044] | 131 |
| PF-OD | LONGITUDINAL tactical_2_6s|ade_m|all | +0.0326 [-0.0258, +0.0932] | 2399 |
| PF-OD | LONGITUDINAL tactical_2_6s|ade_m|nonstop | +0.0710 [+0.0276, +0.1203]* | 2268 |
| PF-OD | LONGITUDINAL tactical_2_6s|ade_m|stop | -0.6325 [-1.1129, -0.2257]* | 131 |
| PF-OD | LONGITUDINAL tactical_2_6s|along_mae_m|all | +0.0377 [-0.0235, +0.1026] | 2399 |
| PF-OD | LONGITUDINAL tactical_2_6s|along_mae_m|nonstop | +0.0740 [+0.0266, +0.1299]* | 2268 |
| PF-OD | LONGITUDINAL tactical_2_6s|along_mae_m|stop | -0.5910 [-1.0663, -0.1995]* | 131 |
| PF-OD | LATERAL tactical_2_6s|cross_mae_m|all | -0.0029 [-0.0183, +0.0116] | 2399 |
| PF-OD | LATERAL tactical_2_6s|cross_mae_m|nonstop | +0.0071 [-0.0027, +0.0192] | 2268 |
| PF-OD | LATERAL tactical_2_6s|cross_mae_m|stop | -0.1763 [-0.3633, -0.0285]* | 131 |
| PF-OD | LATERAL tactical_2_6s|curvature_mae_1pm|all | -0.0007 [-0.0020, +0.0003] | 2258 |
| PF-OD | LATERAL tactical_2_6s|curvature_mae_1pm|nonstop | -0.0005 [-0.0014, +0.0001] | 2137 |
| PF-OD | LATERAL tactical_2_6s|curvature_mae_1pm|stop | -0.0052 [-0.0218, +0.0104] | 121 |
| PF-OD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|all | -0.0366 [-0.1017, +0.0079] | 590 |
| PF-OD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|nonstop | +0.0006 [-0.0154, +0.0217] | 553 |
| PF-OD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|stop | -0.5933 [-1.1716, +0.0000] | 37 |
| PF-OD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|all | -0.0085 [-0.0271, +0.0074] | 532 |
| PF-OD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|nonstop | +0.0006 [-0.0085, +0.0121] | 496 |
| PF-OD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|stop | -0.1340 [-0.2620, +0.0000] | 36 |
| PF-OD | LONGITUDINAL tactical_2_6s|fde_m|all | +0.0575 [-0.0744, +0.1926] | 2399 |
| PF-OD | LONGITUDINAL tactical_2_6s|fde_m|nonstop | +0.1468 [+0.0504, +0.2630]* | 2268 |
| PF-OD | LONGITUDINAL tactical_2_6s|fde_m|stop | -1.4891 [-2.6048, -0.5314]* | 131 |
| PF-OD | LATERAL tactical_2_6s|heading_mae_deg|all | -0.0843 [-0.1820, -0.0214]* | 2310 |
| PF-OD | LATERAL tactical_2_6s|heading_mae_deg|nonstop | -0.0798 [-0.1826, -0.0187]* | 2185 |
| PF-OD | LATERAL tactical_2_6s|heading_mae_deg|stop | -0.1638 [-0.6043, +0.1443] | 125 |
| PF-OD | LONGITUDINAL tactical_2_6s|speed_mae_mps|all | +0.0201 [-0.0100, +0.0546] | 2399 |
| PF-OD | LONGITUDINAL tactical_2_6s|speed_mae_mps|nonstop | +0.0408 [+0.0182, +0.0699]* | 2268 |
| PF-OD | LONGITUDINAL tactical_2_6s|speed_mae_mps|stop | -0.3391 [-0.5979, -0.1199]* | 131 |
| PF-OD | LATERAL tactical_2_6s|yaw_rate_mae_degps|all | -0.0320 [-0.0811, +0.0086] | 2258 |
| PF-OD | LATERAL tactical_2_6s|yaw_rate_mae_degps|nonstop | -0.0288 [-0.0757, +0.0022] | 2137 |
| PF-OD | LATERAL tactical_2_6s|yaw_rate_mae_degps|stop | -0.0896 [-0.6055, +0.4145] | 121 |
| OF-OD | LONGITUDINAL OVERRUN_vs_logged_rest_m|stop | -9.1010 [-11.3623, -6.5887]* | 131 |
| OF-OD | LONGITUDINAL full_0_6s|accel_mae_mps2|all | -0.0106 [-0.0213, -0.0023]* | 2399 |
| OF-OD | LONGITUDINAL full_0_6s|accel_mae_mps2|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL full_0_6s|accel_mae_mps2|stop | -0.1943 [-0.3483, -0.0627]* | 131 |
| OF-OD | LONGITUDINAL full_0_6s|ade_m|all | -0.1239 [-0.1936, -0.0633]* | 2399 |
| OF-OD | LONGITUDINAL full_0_6s|ade_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL full_0_6s|ade_m|stop | -2.2681 [-2.9145, -1.5702]* | 131 |
| OF-OD | LONGITUDINAL full_0_6s|along_mae_m|all | -0.1166 [-0.1860, -0.0570]* | 2399 |
| OF-OD | LONGITUDINAL full_0_6s|along_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL full_0_6s|along_mae_m|stop | -2.1355 [-2.8267, -1.4291]* | 131 |
| OF-OD | LATERAL full_0_6s|cross_mae_m|all | -0.0298 [-0.0500, -0.0128]* | 2399 |
| OF-OD | LATERAL full_0_6s|cross_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LATERAL full_0_6s|cross_mae_m|stop | -0.5466 [-0.8346, -0.2906]* | 131 |
| OF-OD | LATERAL full_0_6s|curvature_mae_1pm|all | +0.0005 [-0.0006, +0.0022] | 2323 |
| OF-OD | LATERAL full_0_6s|curvature_mae_1pm|nonstop | +0.0000 [+0.0000, +0.0000] | 2193 |
| OF-OD | LATERAL full_0_6s|curvature_mae_1pm|stop | +0.0082 [-0.0101, +0.0372] | 130 |
| OF-OD | LONGITUDINAL full_0_6s|fde_m|all | -0.4394 [-0.6827, -0.2243]* | 2399 |
| OF-OD | LONGITUDINAL full_0_6s|fde_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL full_0_6s|fde_m|stop | -8.0459 [-10.2021, -5.5965]* | 131 |
| OF-OD | LATERAL full_0_6s|heading_mae_deg|all | -0.0551 [-0.0965, -0.0220]* | 2360 |
| OF-OD | LATERAL full_0_6s|heading_mae_deg|nonstop | +0.0000 [+0.0000, +0.0000] | 2230 |
| OF-OD | LATERAL full_0_6s|heading_mae_deg|stop | -0.9994 [-1.6264, -0.4765]* | 130 |
| OF-OD | LONGITUDINAL full_0_6s|speed_mae_mps|all | -0.0641 [-0.1001, -0.0320]* | 2399 |
| OF-OD | LONGITUDINAL full_0_6s|speed_mae_mps|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL full_0_6s|speed_mae_mps|stop | -1.1730 [-1.5349, -0.7932]* | 131 |
| OF-OD | LATERAL full_0_6s|yaw_rate_mae_degps|all | +0.0539 [-0.0400, +0.2175] | 2323 |
| OF-OD | LATERAL full_0_6s|yaw_rate_mae_degps|nonstop | +0.0000 [+0.0000, +0.0000] | 2193 |
| OF-OD | LATERAL full_0_6s|yaw_rate_mae_degps|stop | +0.9632 [-0.7030, +3.7786] | 130 |
| OF-OD | LONGITUDINAL operative_0_2s|accel_mae_mps2|all | -0.0049 [-0.0098, -0.0012]* | 2399 |
| OF-OD | LONGITUDINAL operative_0_2s|accel_mae_mps2|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL operative_0_2s|accel_mae_mps2|stop | -0.0906 [-0.1638, -0.0277]* | 131 |
| OF-OD | LONGITUDINAL operative_0_2s|ade_m|all | -0.0031 [-0.0055, -0.0011]* | 2399 |
| OF-OD | LONGITUDINAL operative_0_2s|ade_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL operative_0_2s|ade_m|stop | -0.0571 [-0.0915, -0.0258]* | 131 |
| OF-OD | LONGITUDINAL operative_0_2s|along_mae_m|all | -0.0032 [-0.0057, -0.0011]* | 2399 |
| OF-OD | LONGITUDINAL operative_0_2s|along_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL operative_0_2s|along_mae_m|stop | -0.0582 [-0.0936, -0.0268]* | 131 |
| OF-OD | LATERAL operative_0_2s|cross_mae_m|all | -0.0004 [-0.0008, -0.0001]* | 2399 |
| OF-OD | LATERAL operative_0_2s|cross_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LATERAL operative_0_2s|cross_mae_m|stop | -0.0067 [-0.0139, -0.0016]* | 131 |
| OF-OD | LATERAL operative_0_2s|curvature_mae_1pm|all | -0.0003 [-0.0005, -0.0001]* | 2303 |
| OF-OD | LATERAL operative_0_2s|curvature_mae_1pm|nonstop | +0.0000 [+0.0000, +0.0000] | 2173 |
| OF-OD | LATERAL operative_0_2s|curvature_mae_1pm|stop | -0.0055 [-0.0083, -0.0027]* | 130 |
| OF-OD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|all | -0.0056 [-0.0127, -0.0004]* | 700 |
| OF-OD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|nonstop | +0.0000 [+0.0000, +0.0000] | 653 |
| OF-OD | LONGITUDINAL operative_0_2s|dk_hw_abs_err_m|stop | -0.0834 [-0.1665, -0.0122]* | 47 |
| OF-OD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|all | -0.0011 [-0.0024, -0.0002]* | 647 |
| OF-OD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|nonstop | +0.0000 [+0.0000, +0.0000] | 601 |
| OF-OD | LONGITUDINAL operative_0_2s|dk_tg_abs_err_s|stop | -0.0161 [-0.0292, -0.0039]* | 46 |
| OF-OD | LONGITUDINAL operative_0_2s|fde_m|all | -0.0112 [-0.0196, -0.0042]* | 2399 |
| OF-OD | LONGITUDINAL operative_0_2s|fde_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL operative_0_2s|fde_m|stop | -0.2046 [-0.3173, -0.1038]* | 131 |
| OF-OD | LATERAL operative_0_2s|heading_mae_deg|all | -0.0046 [-0.0084, -0.0017]* | 2314 |
| OF-OD | LATERAL operative_0_2s|heading_mae_deg|nonstop | +0.0000 [+0.0000, +0.0000] | 2184 |
| OF-OD | LATERAL operative_0_2s|heading_mae_deg|stop | -0.0814 [-0.1326, -0.0379]* | 130 |
| OF-OD | LONGITUDINAL operative_0_2s|speed_mae_mps|all | -0.0058 [-0.0099, -0.0024]* | 2399 |
| OF-OD | LONGITUDINAL operative_0_2s|speed_mae_mps|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL operative_0_2s|speed_mae_mps|stop | -0.1055 [-0.1587, -0.0581]* | 131 |
| OF-OD | LATERAL operative_0_2s|yaw_rate_mae_degps|all | -0.0605 [-0.1093, -0.0250]* | 2303 |
| OF-OD | LATERAL operative_0_2s|yaw_rate_mae_degps|nonstop | +0.0000 [+0.0000, +0.0000] | 2173 |
| OF-OD | LATERAL operative_0_2s|yaw_rate_mae_degps|stop | -1.0722 [-1.6996, -0.5672]* | 130 |
| OF-OD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|all | -0.0141 [-0.0282, -0.0029]* | 2399 |
| OF-OD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL tactical_2_6s|accel_mae_mps2|stop | -0.2577 [-0.4614, -0.0791]* | 131 |
| OF-OD | LONGITUDINAL tactical_2_6s|ade_m|all | -0.1842 [-0.2884, -0.0944]* | 2399 |
| OF-OD | LONGITUDINAL tactical_2_6s|ade_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL tactical_2_6s|ade_m|stop | -3.3736 [-4.3290, -2.3353]* | 131 |
| OF-OD | LONGITUDINAL tactical_2_6s|along_mae_m|all | -0.1733 [-0.2763, -0.0845]* | 2399 |
| OF-OD | LONGITUDINAL tactical_2_6s|along_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL tactical_2_6s|along_mae_m|stop | -3.1741 [-4.1947, -2.1273]* | 131 |
| OF-OD | LATERAL tactical_2_6s|cross_mae_m|all | -0.0446 [-0.0747, -0.0190]* | 2399 |
| OF-OD | LATERAL tactical_2_6s|cross_mae_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LATERAL tactical_2_6s|cross_mae_m|stop | -0.8166 [-1.2472, -0.4336]* | 131 |
| OF-OD | LATERAL tactical_2_6s|curvature_mae_1pm|all | -0.0006 [-0.0014, +0.0002] | 2275 |
| OF-OD | LATERAL tactical_2_6s|curvature_mae_1pm|nonstop | +0.0000 [+0.0000, +0.0000] | 2156 |
| OF-OD | LATERAL tactical_2_6s|curvature_mae_1pm|stop | -0.0107 [-0.0274, +0.0035] | 119 |
| OF-OD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|all | -0.1792 [-0.3672, -0.0476]* | 590 |
| OF-OD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|nonstop | +0.0000 [+0.0000, +0.0000] | 553 |
| OF-OD | LONGITUDINAL tactical_2_6s|dk_hw_abs_err_m|stop | -2.8569 [-4.3560, -1.2828]* | 37 |
| OF-OD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|all | -0.0329 [-0.0620, -0.0115]* | 532 |
| OF-OD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|nonstop | +0.0000 [+0.0000, +0.0000] | 496 |
| OF-OD | LONGITUDINAL tactical_2_6s|dk_tg_abs_err_s|stop | -0.4863 [-0.7036, -0.2932]* | 36 |
| OF-OD | LONGITUDINAL tactical_2_6s|fde_m|all | -0.4394 [-0.6827, -0.2243]* | 2399 |
| OF-OD | LONGITUDINAL tactical_2_6s|fde_m|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL tactical_2_6s|fde_m|stop | -8.0459 [-10.2021, -5.5965]* | 131 |
| OF-OD | LATERAL tactical_2_6s|heading_mae_deg|all | -0.0631 [-0.1208, -0.0176]* | 2323 |
| OF-OD | LATERAL tactical_2_6s|heading_mae_deg|nonstop | +0.0000 [+0.0000, +0.0000] | 2202 |
| OF-OD | LATERAL tactical_2_6s|heading_mae_deg|stop | -1.2115 [-2.1810, -0.3939]* | 121 |
| OF-OD | LONGITUDINAL tactical_2_6s|speed_mae_mps|all | -0.0932 [-0.1458, -0.0465]* | 2399 |
| OF-OD | LONGITUDINAL tactical_2_6s|speed_mae_mps|nonstop | +0.0000 [+0.0000, +0.0000] | 2268 |
| OF-OD | LONGITUDINAL tactical_2_6s|speed_mae_mps|stop | -1.7068 [-2.2323, -1.1563]* | 131 |
| OF-OD | LATERAL tactical_2_6s|yaw_rate_mae_degps|all | -0.0724 [-0.1264, -0.0270]* | 2275 |
| OF-OD | LATERAL tactical_2_6s|yaw_rate_mae_degps|nonstop | +0.0000 [+0.0000, +0.0000] | 2156 |
| OF-OD | LATERAL tactical_2_6s|yaw_rate_mae_degps|stop | -1.3841 [-2.2259, -0.5253]* | 119 |

PF-PD point deltas at the other fold seeds (spread, no interval): fs1 stop +0.002 / non-stop +0.045; fs2 stop +0.108 / non-stop +0.038; fs3 stop +0.135 / non-stop +0.045; fs4 stop +0.053 / non-stop +0.044

SHIELD outcome (PF-PD, needs both seeds): **CUE-PROBE-COSTS**; per seed {'0': 'CUE-PROBE-COSTS', '1': 'CUE-PROBE-COSTS'}

## Read 4 Coverage (held-out stop windows with d_0 <= d_true, nominal 0.9, band 0.84 to 0.95 for n = 131)

* plan seed 0: share 0.9083969465648855 (119/131), median margin q 7.431709656824817: **COVERAGE-HOLDS**; other fold seeds: fs1 0.8854961832061069, fs2 0.9083969465648855, fs3 0.9541984732824428, fs4 0.916030534351145
* plan seed 1: share 0.9083969465648855 (119/131), median margin q 7.265879707315605: **COVERAGE-HOLDS**; other fold seeds: fs1 0.8854961832061069, fs2 0.916030534351145, fs3 0.9541984732824428, fs4 0.9236641221374046

## Outcomes in the card's words

* FLAG: VISION-FLAT  (V40+E: VISION-FLAT)
* DIST: CUE-ABSENT
* SHIELD (PF-PD): CUE-PROBE-COSTS
* COVERAGE: {'0': 'COVERAGE-HOLDS', '1': 'COVERAGE-HOLDS'}

Reading choices and deviations R1.. are in CUE_DEFS.md (same folder). A separated interval answers 'another draw of episodes' only; the closed-form probes have no training seed, the fold assignment is the one free choice and its spread is shown above.
