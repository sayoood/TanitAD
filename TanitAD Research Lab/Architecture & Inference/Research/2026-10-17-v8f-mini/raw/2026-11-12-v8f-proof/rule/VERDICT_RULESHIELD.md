# VERDICT_RULESHIELD (PREREG_V8F_PROOF_ADD3, frozen 2026-11-13)

Evidence class: MEASURED (ours; zero training; first-tick T1-kin plans of banked `stopbal_BAL-NULL_s0/_s1`, dumps under `F:/TanitAD-scratch/stopbal_read/t1`, STOPDEC rows under `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopbal/read/out/stopdec`). Tier **T1-kin**, diagnostic, never driving performance.
Estimator: paired episode-cluster bootstrap, B 2,000, seed 0, shield minus NULL on the same windows (answers only 'another draw of episodes'; no replicate arm, so H-ESTIM-SEED-1 applies: a separated CI is necessary, not sufficient).
STRATEGIC family: **PAUSED** (PI item 25), not a pass, not omitted. TACTICAL: T's goal/lon/lat outputs are unchanged by construction (the shield acts below T); plan-derived lon label changed share is printed.

## Outcome (ORACLE, the card's label): **SHIELD-HOLDS-AND-GAINS**

```
{
 "cost_cells_both_seeds": [],
 "cost_cells_one_seed": {},
 "ade_gain_by_seed": {
  "0": true,
  "1": true
 }
}
```

T-CUE (diagnostic, R6): **SHIELD-HOLDS-NO-GAIN**; T-CUE availability: **AVAILABLE (weak): T emits stop_x_m / stop_y_m (v8f_mini.py:49-52 CONS_NAMES, cons slots 16/17), finite on every window; median stop-xy error on the stop windows 11.8 m (seed 0) / 9.6 m (seed 1); flag p>=0.5 on 19 (seed 0, 10 stop) / 33 (seed 1, 18 stop) windows. Work item: a better stop-distance output + its conformal envelope (theorem T3 residual risk = cue miss rate).**

Reference beside (not a gate): STOPPERC S0-C forced the logged stop in the tactical code at coverage 0.7 and read -0.957 / -0.855 m tactical ADE.

## Windows and controls (per seed)

| seed | n | stop | non-stop | T-CUE flagged | flagged and stop | T-CUE active |
|---|---|---|---|---|---|---|
| 0 | 2399 | 131 | 2268 | 19 | 10 | 19 |
| 1 | 2399 | 131 | 2268 | 33 | 18 | 33 |

| seed | variant | cued | feasible | INFEASIBLE-AT-START | s1 max overrun m (tail) | s1 | s2 NULL overruns / mutation overruns | s2 | s3 max cross m | s3 | s4 | infeasible at measured v0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | ORACLE | 131 | 128 | 3 | 0.00e+00 (0.00e+00) | True | 120 / 117 | True | 1.78e-15 | True | True | 3 |
| 0 | T-CUE | 19 | 19 | 0 | 0.00e+00 (0.00e+00) | True | 2 / 2 | True | 7.77e-16 | True | True | 0 |
| 1 | ORACLE | 131 | 129 | 2 | 0.00e+00 (0.00e+00) | True | 111 / 109 | True | 1.78e-15 | True | True | 2 |
| 1 | T-CUE | 33 | 33 | 0 | 0.00e+00 (0.00e+00) | True | 7 / 7 | True | 8.88e-16 | True | True | 1 |

## OVERRUN on the 131 stop windows against the LOGGED rest point (path distance along the plan's own path)

| seed | arm | share overrun | mean m | share incl. braking tail | mean tail m |
|---|---|---|---|---|---|
| 0 | NULL | 0.916 | 9.671 | 0.916 | 14.347 |
| 0 | ORACLE | 0.023 | 0.019 | 0.023 | 0.019 |
| 0 | T-CUE | 0.908 | 9.688 | 0.908 | 14.357 |
| 1 | NULL | 0.847 | 9.116 | 0.870 | 13.584 |
| 1 | ORACLE | 0.015 | 0.015 | 0.015 | 0.015 |
| 1 | T-CUE | 0.847 | 9.118 | 0.878 | 13.585 |

seed 0: ORACLE feasible-only {"n": 128, "NULL_share": 0.9140625, "shield_share": 0.0}; INFEASIBLE-AT-START (apart) {"n": 3, "NULL_share_overrun": 1.0, "braked_share_overrun": 1.0}; T-CUE cued stop windows 10, missed 121; plan lon label changed share {"ORACLE": 0.010004168403501459, "T-CUE": 0.0}

seed 1: ORACLE feasible-only {"n": 129, "NULL_share": 0.8449612403100775, "shield_share": 0.0}; INFEASIBLE-AT-START (apart) {"n": 2, "NULL_share_overrun": 1.0, "braked_share_overrun": 1.0}; T-CUE cued stop windows 18, missed 113; plan lon label changed share {"ORACLE": 0.01208837015423093, "T-CUE": 0.0}

## Cells (variant minus NULL; delta [lo, hi]; * = CI excludes 0)

### ORACLE

| cell | seed 0 | seed 1 |
|---|---|---|
| OVERRUN_vs_logged_rest_m|stop_131 | -9.6526 [-11.5661, -7.5234]* (n 131) | -9.1010 [-11.3623, -6.5887]* (n 131) |
| full_0_6s|accel_mae_mps2|all_2399 | -0.0111 [-0.0225, -0.0020]* (n 2399) | -0.0106 [-0.0213, -0.0023]* (n 2399) |
| full_0_6s|accel_mae_mps2|stop_131 | -0.2029 [-0.3767, -0.0574]* (n 131) | -0.1943 [-0.3483, -0.0627]* (n 131) |
| full_0_6s|ade_m|all_2399 | -0.1288 [-0.1952, -0.0684]* (n 2399) | -0.1239 [-0.1936, -0.0633]* (n 2399) |
| full_0_6s|ade_m|stop_131 | -2.3583 [-2.9477, -1.7592]* (n 131) | -2.2681 [-2.9145, -1.5702]* (n 131) |
| full_0_6s|along_mae_m|all_2399 | -0.1203 [-0.1877, -0.0612]* (n 2399) | -0.1166 [-0.1860, -0.0570]* (n 2399) |
| full_0_6s|along_mae_m|stop_131 | -2.2031 [-2.8640, -1.5585]* (n 131) | -2.1355 [-2.8267, -1.4291]* (n 131) |
| full_0_6s|cross_mae_m|all_2399 | -0.0277 [-0.0448, -0.0128]* (n 2399) | -0.0298 [-0.0500, -0.0128]* (n 2399) |
| full_0_6s|cross_mae_m|stop_131 | -0.5067 [-0.7460, -0.2865]* (n 131) | -0.5466 [-0.8346, -0.2906]* (n 131) |
| full_0_6s|curvature_mae_1pm|all_2399 | +0.0005 [-0.0008, +0.0024] (n 2326) | +0.0005 [-0.0006, +0.0022] (n 2323) |
| full_0_6s|curvature_mae_1pm|stop_131 | +0.0083 [-0.0132, +0.0405] (n 130) | +0.0082 [-0.0101, +0.0372] (n 130) |
| full_0_6s|fde_m|all_2399 | -0.4626 [-0.6943, -0.2503]* (n 2399) | -0.4394 [-0.6827, -0.2243]* (n 2399) |
| full_0_6s|fde_m|stop_131 | -8.4709 [-10.4962, -6.3816]* (n 131) | -8.0459 [-10.2021, -5.5965]* (n 131) |
| full_0_6s|heading_mae_deg|all_2399 | -0.0766 [-0.1257, -0.0344]* (n 2370) | -0.0551 [-0.0965, -0.0220]* (n 2360) |
| full_0_6s|heading_mae_deg|stop_131 | -1.3957 [-2.1085, -0.7707]* (n 130) | -0.9994 [-1.6264, -0.4765]* (n 130) |
| full_0_6s|speed_mae_mps|all_2399 | -0.0682 [-0.1044, -0.0359]* (n 2399) | -0.0641 [-0.1001, -0.0320]* (n 2399) |
| full_0_6s|speed_mae_mps|stop_131 | -1.2498 [-1.5914, -0.9129]* (n 131) | -1.1730 [-1.5349, -0.7932]* (n 131) |
| full_0_6s|yaw_rate_mae_degps|all_2399 | +0.0346 [-0.0659, +0.1988] (n 2326) | +0.0539 [-0.0400, +0.2175] (n 2323) |
| full_0_6s|yaw_rate_mae_degps|stop_131 | +0.6194 [-1.1550, +3.4113] (n 130) | +0.9632 [-0.7030, +3.7786] (n 130) |
| operative_0_2s|accel_mae_mps2|all_2399 | -0.0059 [-0.0128, -0.0014]* (n 2399) | -0.0049 [-0.0098, -0.0012]* (n 2399) |
| operative_0_2s|accel_mae_mps2|stop_131 | -0.1084 [-0.2206, -0.0305]* (n 131) | -0.0906 [-0.1638, -0.0277]* (n 131) |
| operative_0_2s|ade_m|all_2399 | -0.0030 [-0.0055, -0.0010]* (n 2399) | -0.0031 [-0.0055, -0.0011]* (n 2399) |
| operative_0_2s|ade_m|stop_131 | -0.0550 [-0.0903, -0.0238]* (n 131) | -0.0571 [-0.0915, -0.0258]* (n 131) |
| operative_0_2s|along_mae_m|all_2399 | -0.0030 [-0.0055, -0.0010]* (n 2399) | -0.0032 [-0.0057, -0.0011]* (n 2399) |
| operative_0_2s|along_mae_m|stop_131 | -0.0548 [-0.0910, -0.0233]* (n 131) | -0.0582 [-0.0936, -0.0268]* (n 131) |
| operative_0_2s|cross_mae_m|all_2399 | -0.0003 [-0.0008, -0.0001]* (n 2399) | -0.0004 [-0.0008, -0.0001]* (n 2399) |
| operative_0_2s|cross_mae_m|stop_131 | -0.0063 [-0.0128, -0.0016]* (n 131) | -0.0067 [-0.0139, -0.0016]* (n 131) |
| operative_0_2s|curvature_mae_1pm|all_2399 | -0.0001 [-0.0006, +0.0005] (n 2305) | -0.0003 [-0.0005, -0.0001]* (n 2303) |
| operative_0_2s|curvature_mae_1pm|stop_131 | -0.0016 [-0.0102, +0.0089] (n 130) | -0.0055 [-0.0083, -0.0027]* (n 130) |
| operative_0_2s|dk_hw_abs_err_m|all_2399 | -0.0035 [-0.0076, -0.0005]* (n 706) | -0.0056 [-0.0127, -0.0004]* (n 700) |
| operative_0_2s|dk_hw_abs_err_m|stop_131 | -0.0529 [-0.0986, -0.0149]* (n 47) | -0.0834 [-0.1665, -0.0122]* (n 47) |
| operative_0_2s|dk_tg_abs_err_s|all_2399 | -0.0007 [-0.0014, -0.0001]* (n 648) | -0.0011 [-0.0024, -0.0002]* (n 647) |
| operative_0_2s|dk_tg_abs_err_s|stop_131 | -0.0097 [-0.0173, -0.0030]* (n 46) | -0.0161 [-0.0292, -0.0039]* (n 46) |
| operative_0_2s|fde_m|all_2399 | -0.0113 [-0.0198, -0.0043]* (n 2399) | -0.0112 [-0.0196, -0.0042]* (n 2399) |
| operative_0_2s|fde_m|stop_131 | -0.2062 [-0.3263, -0.1017]* (n 131) | -0.2046 [-0.3173, -0.1038]* (n 131) |
| operative_0_2s|heading_mae_deg|all_2399 | -0.0111 [-0.0206, -0.0035]* (n 2314) | -0.0046 [-0.0084, -0.0017]* (n 2314) |
| operative_0_2s|heading_mae_deg|stop_131 | -0.1972 [-0.3666, -0.0789]* (n 130) | -0.0814 [-0.1326, -0.0379]* (n 130) |
| operative_0_2s|speed_mae_mps|all_2399 | -0.0055 [-0.0099, -0.0021]* (n 2399) | -0.0058 [-0.0099, -0.0024]* (n 2399) |
| operative_0_2s|speed_mae_mps|stop_131 | -0.1006 [-0.1612, -0.0473]* (n 131) | -0.1055 [-0.1587, -0.0581]* (n 131) |
| operative_0_2s|yaw_rate_mae_degps|all_2399 | -0.0674 [-0.1326, -0.0171]* (n 2305) | -0.0605 [-0.1093, -0.0250]* (n 2303) |
| operative_0_2s|yaw_rate_mae_degps|stop_131 | -1.1959 [-2.1069, -0.3672]* (n 130) | -1.0722 [-1.6996, -0.5672]* (n 130) |
| tactical_2_6s|accel_mae_mps2|all_2399 | -0.0143 [-0.0290, -0.0025]* (n 2399) | -0.0141 [-0.0282, -0.0029]* (n 2399) |
| tactical_2_6s|accel_mae_mps2|stop_131 | -0.2619 [-0.4779, -0.0710]* (n 131) | -0.2577 [-0.4614, -0.0791]* (n 131) |
| tactical_2_6s|ade_m|all_2399 | -0.1917 [-0.2904, -0.1022]* (n 2399) | -0.1842 [-0.2884, -0.0944]* (n 2399) |
| tactical_2_6s|ade_m|stop_131 | -3.5099 [-4.3835, -2.6164]* (n 131) | -3.3736 [-4.3290, -2.3353]* (n 131) |
| tactical_2_6s|along_mae_m|all_2399 | -0.1790 [-0.2794, -0.0911]* (n 2399) | -0.1733 [-0.2763, -0.0845]* (n 2399) |
| tactical_2_6s|along_mae_m|stop_131 | -3.2773 [-4.2469, -2.3180]* (n 131) | -3.1741 [-4.1947, -2.1273]* (n 131) |
| tactical_2_6s|cross_mae_m|all_2399 | -0.0413 [-0.0670, -0.0192]* (n 2399) | -0.0446 [-0.0747, -0.0190]* (n 2399) |
| tactical_2_6s|cross_mae_m|stop_131 | -0.7568 [-1.1130, -0.4280]* (n 131) | -0.8166 [-1.2472, -0.4336]* (n 131) |
| tactical_2_6s|curvature_mae_1pm|all_2399 | -0.0005 [-0.0018, +0.0003] (n 2278) | -0.0006 [-0.0014, +0.0002] (n 2275) |
| tactical_2_6s|curvature_mae_1pm|stop_131 | -0.0107 [-0.0330, +0.0059] (n 117) | -0.0107 [-0.0274, +0.0035] (n 119) |
| tactical_2_6s|dk_hw_abs_err_m|all_2399 | -0.1841 [-0.3648, -0.0533]* (n 588) | -0.1792 [-0.3672, -0.0476]* (n 590) |
| tactical_2_6s|dk_hw_abs_err_m|stop_131 | -3.1837 [-4.3575, -1.7976]* (n 34) | -2.8569 [-4.3560, -1.2828]* (n 37) |
| tactical_2_6s|dk_tg_abs_err_s|all_2399 | -0.0332 [-0.0604, -0.0117]* (n 530) | -0.0329 [-0.0620, -0.0115]* (n 532) |
| tactical_2_6s|dk_tg_abs_err_s|stop_131 | -0.5330 [-0.6783, -0.3718]* (n 33) | -0.4863 [-0.7036, -0.2932]* (n 36) |
| tactical_2_6s|fde_m|all_2399 | -0.4626 [-0.6943, -0.2503]* (n 2399) | -0.4394 [-0.6827, -0.2243]* (n 2399) |
| tactical_2_6s|fde_m|stop_131 | -8.4709 [-10.4962, -6.3816]* (n 131) | -8.0459 [-10.2021, -5.5965]* (n 131) |
| tactical_2_6s|heading_mae_deg|all_2399 | -0.0566 [-0.1120, -0.0122]* (n 2331) | -0.0631 [-0.1208, -0.0176]* (n 2323) |
| tactical_2_6s|heading_mae_deg|stop_131 | -1.0999 [-1.9892, -0.2857]* (n 120) | -1.2115 [-2.1810, -0.3939]* (n 121) |
| tactical_2_6s|speed_mae_mps|all_2399 | -0.0996 [-0.1518, -0.0532]* (n 2399) | -0.0932 [-0.1458, -0.0465]* (n 2399) |
| tactical_2_6s|speed_mae_mps|stop_131 | -1.8245 [-2.3102, -1.3403]* (n 131) | -1.7068 [-2.2323, -1.1563]* (n 131) |
| tactical_2_6s|yaw_rate_mae_degps|all_2399 | -0.0979 [-0.1722, -0.0364]* (n 2278) | -0.0724 [-0.1264, -0.0270]* (n 2275) |
| tactical_2_6s|yaw_rate_mae_degps|stop_131 | -1.9055 [-2.9869, -0.8797]* (n 117) | -1.3841 [-2.2259, -0.5253]* (n 119) |

### T-CUE

| cell | seed 0 | seed 1 |
|---|---|---|
| OVERRUN_vs_logged_rest_m|stop_131 | +0.0171 [-0.0066, +0.0587] (n 131) | +0.0023 [-0.0372, +0.0446] (n 131) |
| full_0_6s|accel_mae_mps2|all_2399 | -0.0001 [-0.0006, +0.0001] (n 2399) | +0.0003 [-0.0002, +0.0012] (n 2399) |
| full_0_6s|accel_mae_mps2|stop_131 | -0.0038 [-0.0115, +0.0000] (n 131) | -0.0011 [-0.0028, +0.0003] (n 131) |
| full_0_6s|ade_m|all_2399 | +0.0010 [-0.0000, +0.0024] (n 2399) | +0.0018 [-0.0002, +0.0047] (n 2399) |
| full_0_6s|ade_m|stop_131 | +0.0103 [-0.0011, +0.0302] (n 131) | +0.0031 [-0.0150, +0.0317] (n 131) |
| full_0_6s|along_mae_m|all_2399 | +0.0010 [-0.0000, +0.0024] (n 2399) | +0.0019 [-0.0003, +0.0048] (n 2399) |
| full_0_6s|along_mae_m|stop_131 | +0.0108 [-0.0011, +0.0305] (n 131) | +0.0044 [-0.0157, +0.0348] (n 131) |
| full_0_6s|cross_mae_m|all_2399 | +0.0002 [+0.0000, +0.0004]* (n 2399) | -0.0001 [-0.0005, +0.0002] (n 2399) |
| full_0_6s|cross_mae_m|stop_131 | +0.0028 [+0.0000, +0.0083] (n 131) | -0.0020 [-0.0081, +0.0040] (n 131) |
| full_0_6s|curvature_mae_1pm|all_2399 | +0.0001 [-0.0000, +0.0002] (n 2326) | -0.0002 [-0.0005, +0.0001] (n 2323) |
| full_0_6s|curvature_mae_1pm|stop_131 | +0.0018 [-0.0001, +0.0049] (n 130) | -0.0019 [-0.0070, +0.0029] (n 130) |
| full_0_6s|fde_m|all_2399 | +0.0028 [-0.0000, +0.0067] (n 2399) | +0.0074 [-0.0000, +0.0165] (n 2399) |
| full_0_6s|fde_m|stop_131 | +0.0257 [-0.0018, +0.0710] (n 131) | +0.0209 [-0.0325, +0.1132] (n 131) |
| full_0_6s|heading_mae_deg|all_2399 | +0.0023 [-0.0003, +0.0064] (n 2370) | +0.0002 [-0.0029, +0.0038] (n 2360) |
| full_0_6s|heading_mae_deg|stop_131 | +0.0463 [+0.0000, +0.1190] (n 130) | +0.0177 [-0.0351, +0.0815] (n 130) |
| full_0_6s|speed_mae_mps|all_2399 | +0.0002 [-0.0002, +0.0007] (n 2399) | +0.0011 [-0.0001, +0.0026] (n 2399) |
| full_0_6s|speed_mae_mps|stop_131 | -0.0014 [-0.0034, +0.0000] (n 131) | +0.0030 [-0.0051, +0.0170] (n 131) |
| full_0_6s|yaw_rate_mae_degps|all_2399 | +0.0028 [-0.0015, +0.0092] (n 2326) | -0.0066 [-0.0220, +0.0073] (n 2323) |
| full_0_6s|yaw_rate_mae_degps|stop_131 | +0.0473 [-0.0230, +0.1635] (n 130) | +0.0391 [-0.1792, +0.2641] (n 130) |
| operative_0_2s|accel_mae_mps2|all_2399 | -0.0009 [-0.0023, +0.0000] (n 2399) | -0.0007 [-0.0021, +0.0004] (n 2399) |
| operative_0_2s|accel_mae_mps2|stop_131 | -0.0162 [-0.0409, +0.0000] (n 131) | -0.0191 [-0.0410, -0.0039]* (n 131) |
| operative_0_2s|ade_m|all_2399 | +0.0002 [-0.0000, +0.0006] (n 2399) | +0.0004 [-0.0001, +0.0011] (n 2399) |
| operative_0_2s|ade_m|stop_131 | +0.0033 [-0.0004, +0.0102] (n 131) | +0.0017 [-0.0026, +0.0076] (n 131) |
| operative_0_2s|along_mae_m|all_2399 | +0.0002 [-0.0000, +0.0006] (n 2399) | +0.0004 [-0.0001, +0.0011] (n 2399) |
| operative_0_2s|along_mae_m|stop_131 | +0.0033 [-0.0005, +0.0104] (n 131) | +0.0014 [-0.0030, +0.0075] (n 131) |
| operative_0_2s|cross_mae_m|all_2399 | +0.0000 [+0.0000, +0.0000]* (n 2399) | +0.0000 [-0.0000, +0.0000] (n 2399) |
| operative_0_2s|cross_mae_m|stop_131 | +0.0001 [+0.0000, +0.0003] (n 131) | +0.0000 [-0.0002, +0.0004] (n 131) |
| operative_0_2s|curvature_mae_1pm|all_2399 | -0.0000 [-0.0001, -0.0000]* (n 2305) | -0.0002 [-0.0003, -0.0000]* (n 2303) |
| operative_0_2s|curvature_mae_1pm|stop_131 | -0.0003 [-0.0008, +0.0000] (n 130) | -0.0014 [-0.0031, -0.0002]* (n 130) |
| operative_0_2s|dk_hw_abs_err_m|all_2399 | +0.0000 [+0.0000, +0.0001] (n 706) | +0.0001 [-0.0000, +0.0003] (n 700) |
| operative_0_2s|dk_hw_abs_err_m|stop_131 | +0.0003 [+0.0000, +0.0012] (n 47) | +0.0012 [-0.0002, +0.0048] (n 47) |
| operative_0_2s|dk_tg_abs_err_s|all_2399 | +0.0000 [+0.0000, +0.0000] (n 648) | +0.0000 [-0.0000, +0.0001] (n 647) |
| operative_0_2s|dk_tg_abs_err_s|stop_131 | +0.0000 [+0.0000, +0.0002] (n 46) | +0.0002 [-0.0000, +0.0008] (n 46) |
| operative_0_2s|fde_m|all_2399 | +0.0005 [-0.0001, +0.0016] (n 2399) | +0.0008 [-0.0006, +0.0026] (n 2399) |
| operative_0_2s|fde_m|stop_131 | +0.0091 [-0.0019, +0.0287] (n 131) | +0.0004 [-0.0127, +0.0162] (n 131) |
| operative_0_2s|heading_mae_deg|all_2399 | -0.0004 [-0.0010, -0.0000]* (n 2314) | -0.0015 [-0.0032, -0.0002]* (n 2314) |
| operative_0_2s|heading_mae_deg|stop_131 | -0.0063 [-0.0168, +0.0000] (n 130) | -0.0206 [-0.0436, -0.0021]* (n 130) |
| operative_0_2s|speed_mae_mps|all_2399 | +0.0000 [-0.0002, +0.0002] (n 2399) | +0.0001 [-0.0006, +0.0010] (n 2399) |
| operative_0_2s|speed_mae_mps|stop_131 | +0.0002 [-0.0029, +0.0037] (n 131) | -0.0049 [-0.0116, +0.0009] (n 131) |
| operative_0_2s|yaw_rate_mae_degps|all_2399 | -0.0054 [-0.0138, -0.0003]* (n 2305) | -0.0184 [-0.0361, -0.0045]* (n 2303) |
| operative_0_2s|yaw_rate_mae_degps|stop_131 | -0.0863 [-0.2270, +0.0000] (n 130) | -0.2045 [-0.4054, -0.0390]* (n 130) |
| tactical_2_6s|accel_mae_mps2|all_2399 | +0.0001 [-0.0001, +0.0004] (n 2399) | +0.0007 [-0.0002, +0.0022] (n 2399) |
| tactical_2_6s|accel_mae_mps2|stop_131 | +0.0007 [-0.0023, +0.0045] (n 131) | +0.0054 [+0.0002, +0.0132]* (n 131) |
| tactical_2_6s|ade_m|all_2399 | +0.0014 [-0.0000, +0.0033] (n 2399) | +0.0026 [-0.0004, +0.0066] (n 2399) |
| tactical_2_6s|ade_m|stop_131 | +0.0137 [-0.0014, +0.0402] (n 131) | +0.0038 [-0.0239, +0.0459] (n 131) |
| tactical_2_6s|along_mae_m|all_2399 | +0.0014 [-0.0000, +0.0033] (n 2399) | +0.0027 [-0.0004, +0.0069] (n 2399) |
| tactical_2_6s|along_mae_m|stop_131 | +0.0146 [-0.0014, +0.0410] (n 131) | +0.0060 [-0.0252, +0.0508] (n 131) |
| tactical_2_6s|cross_mae_m|all_2399 | +0.0002 [+0.0000, +0.0007]* (n 2399) | -0.0001 [-0.0007, +0.0004] (n 2399) |
| tactical_2_6s|cross_mae_m|stop_131 | +0.0042 [+0.0000, +0.0124] (n 131) | -0.0031 [-0.0122, +0.0060] (n 131) |
| tactical_2_6s|curvature_mae_1pm|all_2399 | +0.0002 [-0.0001, +0.0006] (n 2285) | -0.0004 [-0.0012, +0.0003] (n 2280) |
| tactical_2_6s|curvature_mae_1pm|stop_131 | +0.0043 [+0.0000, +0.0110] (n 124) | -0.0070 [-0.0219, +0.0061] (n 124) |
| tactical_2_6s|dk_hw_abs_err_m|all_2399 | +0.0000 [+0.0000, +0.0000] (n 588) | +0.0000 [+0.0000, +0.0000] (n 590) |
| tactical_2_6s|dk_hw_abs_err_m|stop_131 | +0.0000 [+0.0000, +0.0000] (n 34) | +0.0000 [+0.0000, +0.0000] (n 37) |
| tactical_2_6s|dk_tg_abs_err_s|all_2399 | +0.0000 [+0.0000, +0.0000] (n 530) | +0.0000 [+0.0000, +0.0000] (n 532) |
| tactical_2_6s|dk_tg_abs_err_s|stop_131 | +0.0000 [+0.0000, +0.0000] (n 33) | +0.0000 [+0.0000, +0.0000] (n 36) |
| tactical_2_6s|fde_m|all_2399 | +0.0028 [-0.0000, +0.0067] (n 2399) | +0.0074 [-0.0000, +0.0165] (n 2399) |
| tactical_2_6s|fde_m|stop_131 | +0.0257 [-0.0018, +0.0710] (n 131) | +0.0209 [-0.0325, +0.1132] (n 131) |
| tactical_2_6s|heading_mae_deg|all_2399 | +0.0058 [-0.0003, +0.0155] (n 2337) | +0.0035 [-0.0003, +0.0091] (n 2329) |
| tactical_2_6s|heading_mae_deg|stop_131 | +0.1133 [+0.0000, +0.2803] (n 126) | +0.0674 [+0.0010, +0.1623]* (n 128) |
| tactical_2_6s|speed_mae_mps|all_2399 | +0.0002 [-0.0003, +0.0010] (n 2399) | +0.0016 [-0.0002, +0.0038] (n 2399) |
| tactical_2_6s|speed_mae_mps|stop_131 | -0.0022 [-0.0060, +0.0002] (n 131) | +0.0070 [-0.0058, +0.0296] (n 131) |
| tactical_2_6s|yaw_rate_mae_degps|all_2399 | +0.0097 [-0.0021, +0.0257] (n 2285) | -0.0034 [-0.0323, +0.0205] (n 2280) |
| tactical_2_6s|yaw_rate_mae_degps|stop_131 | +0.2160 [+0.0000, +0.4874] (n 124) | +0.0102 [-0.5256, +0.4630] (n 124) |

## Reading choices

```
(R1-R6, printed in the verdict; none moves a bar):
 R1 plan grid: the dumped plan is 30 knots at 0.2 s (not 60 at 0.1 s): the shield runs at dt = 0.2 s, q = a*dt = 0.6 m/s, a = 3.0 m/s2.
 R2 start state: v_0 = the plan's own first-segment speed (path origin -> knot 1 / 0.2 s); B(v_0) > d_0 = INFEASIBLE-AT-START. The measured v0 is read beside (infeasible_at_measured_v0).
 R3 d_0 (both variants): arc length ON THE PLAN'S OWN PATH (extended straight beyond the last knot) of the point nearest to the rest point (x, y) in the tick frame.
    ORACLE: the logged rest point = STOPDEC cons_gt stop_x_m / stop_y_m (finite exactly on the stop_hit windows); T-CUE: T's own t_cons_pred stop_x_m / stop_y_m (de-standardised) where p(STOP_POINT) >= 0.5.
 R4 STOP window set = stop_hit (goal_y STOP_POINT & goal_w, stopdec_read.py:100) = 131; OVERRUN = max(0, plan path length at 6 s - d_0); `overrun_tail` adds the braking distance of the step after the last segment, B(max(v_end - q, 0)). s2 is required for ORACLE; for T-CUE it is read where NULL overruns and shown as None when it has no such window.
 R5 verdict precedence: controls fail -> SHIELD-BROKEN; else SHIELD-COSTS if an ORACLE lateral/longitudinal cell (incl. ADE/FDE, which the card files under LONGITUDINAL) is worse with a CI excluding 0 at BOTH seeds; else HOLDS-AND-GAINS if the
    ORACLE tactical-band (2-6 s) ADE on stop windows has a CI excluding 0 on the low side at both seeds; else HOLDS-NO-GAIN. Cost cells at one seed only are printed. A separated CI is necessary, not sufficient (H-ESTIM-SEED-1: no replicate arm here).
 R6 T-CUE outcome is computed by the same rule and labelled diagnostic; its cue-miss residual (overrun against the LOGGED rest point) is reported apart.
```
