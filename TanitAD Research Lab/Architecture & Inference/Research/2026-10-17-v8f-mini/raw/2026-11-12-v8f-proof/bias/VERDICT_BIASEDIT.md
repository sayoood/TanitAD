# VERDICT BIASEDIT (addendum 4: stop-weighting lever as a one-scalar weight edit); MEASURED, T1-kin first tick (diagnostic), eval-141

Outcome: **BIAS-MATCHES-ROC**
Estimator: taniteval.ci paired / episode-cluster bootstrap, B=2000, seed 0; answers another draw of EPISODES only
Controls: 0 problem(s): none
Replicate gap |NULL_rep - NULL_s0|: hit 0.0684, fpr 0.0103 (one replicate pair, seed 0; used for both seeds).

STRATEGIC family: PAUSED (PI item 25).

## STOPDEC hit / false-positive rate at p >= 0.5

| tag | hit [CI] | fpr [CI] |
|---|---|---|
| NULL_s0 | 0.0598 [0.0083, 0.1406] | 0.0054 [0.0018, 0.0104] |
| NULL_rep | 0.1282 [0.0385, 0.2400] | 0.0157 [0.0071, 0.0264] |
| G_s0 | 0.2222 [0.0970, 0.3726] | 0.0278 [0.0160, 0.0421] |
| G2_s0 | 0.4530 [0.3167, 0.6000] | 0.1195 [0.0833, 0.1586] |
| NULL_s1 | 0.1197 [0.0455, 0.2119] | 0.0085 [0.0027, 0.0158] |
| G_s1 | 0.2906 [0.1393, 0.4562] | 0.0506 [0.0282, 0.0757] |
| G2_s1 | 0.5299 [0.3681, 0.6870] | 0.1258 [0.0888, 0.1639] |
| GE0_s0 | 0.0598 [0.0083, 0.1406] | 0.0054 [0.0018, 0.0104] |
| GE1_s0 | 0.2564 [0.1322, 0.4017] | 0.0416 [0.0258, 0.0603] |
| GE2_s0 | 0.5299 [0.3964, 0.6667] | 0.1468 [0.1066, 0.1887] |
| GE0_s1 | 0.1197 [0.0455, 0.2119] | 0.0085 [0.0027, 0.0158] |
| GE1_s1 | 0.3590 [0.2018, 0.5391] | 0.0609 [0.0364, 0.0883] |
| GE2_s1 | 0.5897 [0.4407, 0.7391] | 0.1652 [0.1249, 0.2076] |

## Edit vs trained, and b_match

| pair | edit hit / fpr | trained hit / fpr | equal within gap (hit, fpr) | b (ln W) | b_match | hit / fpr at b_match | matches ROC (hit within gap) |
|---|---|---|---|---|---|---|---|
| dose1_s0 | 0.2564 / 0.0416 | 0.2222 / 0.0278 | True, False | 1.518 | 1.086 | 0.2137 / 0.0278 | True |
| dose2_s0 | 0.5299 / 0.1468 | 0.4530 / 0.1195 | False, False | 3.035 | 2.760 | 0.4786 / 0.1195 | True |
| dose1_s1 | 0.3590 / 0.0609 | 0.2906 / 0.0506 | True, True | 1.518 | 1.382 | 0.3077 / 0.0506 | True |
| dose2_s1 | 0.5897 / 0.1652 | 0.5299 / 0.1258 | True, False | 3.035 | 2.564 | 0.5214 / 0.1258 | True |

## Controls

```
{
 "threads_by_tag": {
  "NULL_s0": 2,
  "NULL_rep": 2,
  "G_s0": 2,
  "G2_s0": 2,
  "NULL_s1": 2,
  "G_s1": 2,
  "G2_s1": 2,
  "GE0_s0": 2,
  "GE1_s0": 2,
  "GE2_s0": 2,
  "GE0_s1": 2,
  "GE1_s1": 2,
  "GE2_s1": 2
 },
 "e1_from_BUILD_REPORT": {
  "GE0_s0": true,
  "GE1_s0": true,
  "GE2_s0": true,
  "GE0_s1": true,
  "GE1_s1": true,
  "GE2_s1": true
 },
 "e2_auroc_GE0_s0": {
  "edit": 0.8188179571348775,
  "parent": 0.8188179571348775,
  "abs_diff": 0.0,
  "pass": true
 },
 "e2_auroc_GE1_s0": {
  "edit": 0.8188179571348775,
  "parent": 0.8188179571348775,
  "abs_diff": 0.0,
  "pass": true
 },
 "e2_auroc_GE2_s0": {
  "edit": 0.8188179571348775,
  "parent": 0.8188179571348775,
  "abs_diff": 0.0,
  "pass": true
 },
 "e2_auroc_GE0_s1": {
  "edit": 0.8497386926214142,
  "parent": 0.8497386926214142,
  "abs_diff": 0.0,
  "pass": true
 },
 "e2_auroc_GE1_s1": {
  "edit": 0.8497386926214142,
  "parent": 0.8497386926214142,
  "abs_diff": 0.0,
  "pass": true
 },
 "e2_auroc_GE2_s1": {
  "edit": 0.8497386926214142,
  "parent": 0.8497386926214142,
  "abs_diff": 0.0,
  "pass": true
 },
 "e3_GE0_s0": {
  "dump_bit_identical": {
   "cl": true,
   "ol": true,
   "fl": true,
   "g": true,
   "v0": true
  },
  "stopdec_bit_identical": {
   "goal_prob": true,
   "lon_prob": true,
   "dec_goal_stop": true
  },
  "pass": true
 },
 "e3_GE0_s1": {
  "dump_bit_identical": {
   "cl": true,
   "ol": true,
   "fl": true,
   "g": true,
   "v0": true
  },
  "stopdec_bit_identical": {
   "goal_prob": true,
   "lon_prob": true,
   "dec_goal_stop": true
  },
  "pass": true
 }
}
```

## T1-kin four families for GOALEDIT(b2) (edit minus reference; paired episode-cluster bootstrap)

### GE2_s0_vs_parent

| family | cell | windows | delta [CI] | edit / ref | flag |
|---|---|---|---|---|---|
| PRIMARY | tactical_2_6s:ade_m|all | 2399 | 0.0802 [0.0381, 0.1264] | 3.3178 / 3.2375 | WORSE |
| PRIMARY | tactical_2_6s:ade_m|stop_point | 117 | -0.4286 [-0.6573, -0.1979] | 5.3165 / 5.7451 | better |
| PRIMARY | tactical_2_6s:fde_m|all | 2399 | 0.2724 [0.1585, 0.3982] | 7.1667 / 6.8943 | WORSE |
| LONGITUDINAL | tactical_2_6s:speed_mae_mps|all | 2399 | 0.0681 [0.0383, 0.1018] | 1.5877 / 1.5196 | WORSE |
| LONGITUDINAL | tactical_2_6s:speed_mae_mps|stop_point | 117 | -0.0372 [-0.3234, 0.2408] | 3.7778 / 3.8150 |  |
| LONGITUDINAL | tactical_2_6s:along_mae_m|all | 2399 | 0.0746 [0.0332, 0.1192] | 2.6260 / 2.5514 | WORSE |
| LONGITUDINAL | tactical_2_6s:accel_mae_mps2|all | 2399 | 0.1310 [0.0493, 0.2152] | 1.7392 / 1.6081 | WORSE |
| LATERAL | tactical_2_6s:cross_mae_m|all | 2399 | 0.0210 [-0.0010, 0.0449] | 1.4219 / 1.4009 |  |
| LATERAL | tactical_2_6s:heading_mae_deg|all | 2347 | 1.0653 [0.7065, 1.4575] | 9.7838 / 8.7184 | WORSE |
| LATERAL | tactical_2_6s:yaw_rate_mae_degps|all | 2340 | 5.5339 [3.6486, 7.6672] | 35.8850 / 30.3511 | WORSE |
| LATERAL | tactical_2_6s:curvature_mae_1pm|all | 2340 | 0.0511 [0.0310, 0.0752] | 0.2316 / 0.1805 | WORSE |
| LONGITUDINAL | tactical_2_6s:dk_hw_abs_err_m|all | 569 | -0.0474 [-0.1607, 0.0486] | 2.4212 / 2.4686 |  |
| LONGITUDINAL | tactical_2_6s:dk_tg_abs_err_s|all | 526 | -0.0285 [-0.0686, 0.0041] | 0.3718 / 0.4003 |  |
| PRIMARY | full_0_6s:ade_m|all | 2399 | 0.0529 [0.0245, 0.0841] | 2.2840 / 2.2311 | WORSE |
| PRIMARY | full_0_6s:ade_m|stop_point | 117 | -0.2937 [-0.4464, -0.1388] | 3.6580 / 3.9517 | better |
| PRIMARY | full_0_6s:fde_m|all | 2399 | 0.2724 [0.1585, 0.3982] | 7.1667 / 6.8943 | WORSE |
| LONGITUDINAL | full_0_6s:speed_mae_mps|all | 2399 | 0.0458 [0.0257, 0.0686] | 1.1453 / 1.0996 | WORSE |
| LONGITUDINAL | full_0_6s:speed_mae_mps|stop_point | 117 | -0.0334 [-0.2224, 0.1521] | 2.6590 / 2.6924 |  |
| LONGITUDINAL | full_0_6s:along_mae_m|all | 2399 | 0.0493 [0.0215, 0.0793] | 1.8107 / 1.7614 | WORSE |
| LONGITUDINAL | full_0_6s:accel_mae_mps2|all | 2399 | 0.0885 [0.0347, 0.1445] | 1.3641 / 1.2756 | WORSE |
| LATERAL | full_0_6s:cross_mae_m|all | 2399 | 0.0137 [-0.0009, 0.0297] | 0.9735 / 0.9597 |  |
| LATERAL | full_0_6s:heading_mae_deg|all | 2372 | 0.8165 [0.5301, 1.1358] | 7.6139 / 6.7974 | WORSE |
| LATERAL | full_0_6s:yaw_rate_mae_degps|all | 2366 | 3.7548 [2.3894, 5.3101] | 27.3409 / 23.5861 | WORSE |
| LATERAL | full_0_6s:curvature_mae_1pm|all | 2366 | 0.0342 [0.0202, 0.0506] | 0.1916 / 0.1574 | WORSE |
| TACTICAL | TAC-LAT|all | 2399 | -0.0229 [-0.0342, -0.0121] | 0.7674 / 0.7903 | WORSE |
| TACTICAL | TAC-LON|all | 2399 | -0.0079 [-0.0188, 0.0033] | 0.5123 / 0.5202 |  |
| TACTICAL | TAC-LON-macroF1|all | 2399 | -0.0104 [-0.0220, 0.0013] | 0.5069 / 0.5173 |  |
| STRATEGIC | | | PAUSED (PI item 25) | | |

### GE2_s0_vs_trained_G2

| family | cell | windows | delta [CI] | edit / ref | flag |
|---|---|---|---|---|---|
| PRIMARY | tactical_2_6s:ade_m|all | 2399 | 0.0406 [-0.0084, 0.0859] | 3.3178 / 3.2772 |  |
| PRIMARY | tactical_2_6s:ade_m|stop_point | 117 | -0.2108 [-0.4493, 0.0005] | 5.3165 / 5.5273 |  |
| PRIMARY | tactical_2_6s:fde_m|all | 2399 | 0.1321 [-0.0034, 0.2562] | 7.1667 / 7.0346 |  |
| LONGITUDINAL | tactical_2_6s:speed_mae_mps|all | 2399 | 0.0295 [-0.0051, 0.0648] | 1.5877 / 1.5582 |  |
| LONGITUDINAL | tactical_2_6s:speed_mae_mps|stop_point | 117 | -0.0129 [-0.2628, 0.2216] | 3.7778 / 3.7906 |  |
| LONGITUDINAL | tactical_2_6s:along_mae_m|all | 2399 | 0.0352 [-0.0133, 0.0840] | 2.6260 / 2.5908 |  |
| LONGITUDINAL | tactical_2_6s:accel_mae_mps2|all | 2399 | 0.0917 [-0.0102, 0.1957] | 1.7392 / 1.6475 |  |
| LATERAL | tactical_2_6s:cross_mae_m|all | 2399 | 0.0086 [-0.0179, 0.0363] | 1.4219 / 1.4132 |  |
| LATERAL | tactical_2_6s:heading_mae_deg|all | 2348 | 0.6547 [0.3528, 0.9998] | 9.8312 / 9.1765 | WORSE |
| LATERAL | tactical_2_6s:yaw_rate_mae_degps|all | 2340 | 3.5101 [1.7655, 5.5871] | 35.8850 / 32.3749 | WORSE |
| LATERAL | tactical_2_6s:curvature_mae_1pm|all | 2340 | 0.0272 [0.0077, 0.0496] | 0.2316 / 0.2044 | WORSE |
| LONGITUDINAL | tactical_2_6s:dk_hw_abs_err_m|all | 563 | 0.0138 [-0.0855, 0.1181] | 2.3908 / 2.3769 |  |
| LONGITUDINAL | tactical_2_6s:dk_tg_abs_err_s|all | 523 | 0.0001 [-0.0189, 0.0196] | 0.3674 / 0.3673 |  |
| PRIMARY | full_0_6s:ade_m|all | 2399 | 0.0262 [-0.0066, 0.0566] | 2.2840 / 2.2577 |  |
| PRIMARY | full_0_6s:ade_m|stop_point | 117 | -0.1443 [-0.3038, -0.0011] | 3.6580 / 3.8023 | better |
| PRIMARY | full_0_6s:fde_m|all | 2399 | 0.1321 [-0.0034, 0.2562] | 7.1667 / 7.0346 |  |
| LONGITUDINAL | full_0_6s:speed_mae_mps|all | 2399 | 0.0188 [-0.0047, 0.0426] | 1.1453 / 1.1265 |  |
| LONGITUDINAL | full_0_6s:speed_mae_mps|stop_point | 117 | -0.0143 [-0.1785, 0.1387] | 2.6590 / 2.6733 |  |
| LONGITUDINAL | full_0_6s:along_mae_m|all | 2399 | 0.0227 [-0.0098, 0.0555] | 1.8107 / 1.7879 |  |
| LONGITUDINAL | full_0_6s:accel_mae_mps2|all | 2399 | 0.0595 [-0.0073, 0.1289] | 1.3641 / 1.3046 |  |
| LATERAL | full_0_6s:cross_mae_m|all | 2399 | 0.0056 [-0.0122, 0.0241] | 0.9735 / 0.9679 |  |
| LATERAL | full_0_6s:heading_mae_deg|all | 2373 | 0.4835 [0.2574, 0.7353] | 7.6618 / 7.1783 | WORSE |
| LATERAL | full_0_6s:yaw_rate_mae_degps|all | 2366 | 2.2737 [1.0345, 3.6720] | 27.3409 / 25.0672 | WORSE |
| LATERAL | full_0_6s:curvature_mae_1pm|all | 2366 | 0.0162 [0.0003, 0.0331] | 0.1916 / 0.1754 | WORSE |
| TACTICAL | TAC-LAT|all | 2399 | -0.0117 [-0.0225, -0.0025] | 0.7674 / 0.7791 | WORSE |
| TACTICAL | TAC-LON|all | 2399 | -0.0038 [-0.0246, 0.0200] | 0.5123 / 0.5160 |  |
| TACTICAL | TAC-LON-macroF1|all | 2399 | -0.0059 [-0.0262, 0.0165] | 0.5069 / 0.5128 |  |
| STRATEGIC | | | PAUSED (PI item 25) | | |

### GE2_s1_vs_parent

| family | cell | windows | delta [CI] | edit / ref | flag |
|---|---|---|---|---|---|
| PRIMARY | tactical_2_6s:ade_m|all | 2399 | 0.0625 [0.0262, 0.0986] | 3.4049 / 3.3424 | WORSE |
| PRIMARY | tactical_2_6s:ade_m|stop_point | 117 | -0.2445 [-0.5005, 0.0186] | 5.0128 / 5.2573 |  |
| PRIMARY | tactical_2_6s:fde_m|all | 2399 | 0.2033 [0.1068, 0.2990] | 7.4902 / 7.2869 | WORSE |
| LONGITUDINAL | tactical_2_6s:speed_mae_mps|all | 2399 | 0.0328 [0.0054, 0.0616] | 1.6792 / 1.6464 | WORSE |
| LONGITUDINAL | tactical_2_6s:speed_mae_mps|stop_point | 117 | -0.0521 [-0.2995, 0.2670] | 3.4301 / 3.4822 |  |
| LONGITUDINAL | tactical_2_6s:along_mae_m|all | 2399 | 0.0536 [0.0184, 0.0878] | 2.7367 / 2.6832 | WORSE |
| LONGITUDINAL | tactical_2_6s:accel_mae_mps2|all | 2399 | 0.0967 [0.0138, 0.1973] | 2.1262 / 2.0295 | WORSE |
| LATERAL | tactical_2_6s:cross_mae_m|all | 2399 | 0.0252 [0.0078, 0.0433] | 1.4106 / 1.3854 | WORSE |
| LATERAL | tactical_2_6s:heading_mae_deg|all | 2348 | 0.9536 [0.6414, 1.2999] | 10.8615 / 9.9079 | WORSE |
| LATERAL | tactical_2_6s:yaw_rate_mae_degps|all | 2340 | 4.4580 [2.9145, 6.1262] | 41.6428 / 37.1848 | WORSE |
| LATERAL | tactical_2_6s:curvature_mae_1pm|all | 2340 | 0.0320 [0.0170, 0.0481] | 0.2808 / 0.2488 | WORSE |
| LONGITUDINAL | tactical_2_6s:dk_hw_abs_err_m|all | 562 | 0.0250 [-0.0622, 0.1201] | 2.2766 / 2.2516 |  |
| LONGITUDINAL | tactical_2_6s:dk_tg_abs_err_s|all | 529 | -0.0001 [-0.0264, 0.0251] | 0.3820 / 0.3821 |  |
| PRIMARY | full_0_6s:ade_m|all | 2399 | 0.0409 [0.0164, 0.0654] | 2.3438 / 2.3028 | WORSE |
| PRIMARY | full_0_6s:ade_m|stop_point | 117 | -0.1693 [-0.3425, 0.0085] | 3.4548 / 3.6241 |  |
| PRIMARY | full_0_6s:fde_m|all | 2399 | 0.2033 [0.1068, 0.2990] | 7.4902 / 7.2869 | WORSE |
| LONGITUDINAL | full_0_6s:speed_mae_mps|all | 2399 | 0.0218 [0.0034, 0.0409] | 1.2085 / 1.1867 | WORSE |
| LONGITUDINAL | full_0_6s:speed_mae_mps|stop_point | 117 | -0.0428 [-0.2104, 0.1722] | 2.4303 / 2.4732 |  |
| LONGITUDINAL | full_0_6s:along_mae_m|all | 2399 | 0.0352 [0.0115, 0.0584] | 1.8856 / 1.8505 | WORSE |
| LONGITUDINAL | full_0_6s:accel_mae_mps2|all | 2399 | 0.0656 [0.0106, 0.1325] | 1.6169 / 1.5513 | WORSE |
| LATERAL | full_0_6s:cross_mae_m|all | 2399 | 0.0164 [0.0047, 0.0284] | 0.9671 / 0.9507 | WORSE |
| LATERAL | full_0_6s:heading_mae_deg|all | 2373 | 0.6625 [0.4447, 0.9109] | 8.4110 / 7.7485 | WORSE |
| LATERAL | full_0_6s:yaw_rate_mae_degps|all | 2366 | 2.7791 [1.6384, 3.9353] | 31.1847 / 28.4056 | WORSE |
| LATERAL | full_0_6s:curvature_mae_1pm|all | 2366 | 0.0230 [0.0132, 0.0336] | 0.2206 / 0.1975 | WORSE |
| TACTICAL | TAC-LAT|all | 2399 | -0.0375 [-0.0538, -0.0229] | 0.7549 / 0.7924 | WORSE |
| TACTICAL | TAC-LON|all | 2399 | -0.0104 [-0.0200, -0.0013] | 0.4935 / 0.5040 | WORSE |
| TACTICAL | TAC-LON-macroF1|all | 2399 | -0.0109 [-0.0206, -0.0018] | 0.4933 / 0.5042 | separated |
| STRATEGIC | | | PAUSED (PI item 25) | | |

### GE2_s1_vs_trained_G2

| family | cell | windows | delta [CI] | edit / ref | flag |
|---|---|---|---|---|---|
| PRIMARY | tactical_2_6s:ade_m|all | 2399 | 0.0269 [-0.0248, 0.0806] | 3.4049 / 3.3780 |  |
| PRIMARY | tactical_2_6s:ade_m|stop_point | 117 | -0.3405 [-0.6059, -0.0620] | 5.0128 / 5.3533 | better |
| PRIMARY | tactical_2_6s:fde_m|all | 2399 | 0.1098 [-0.0393, 0.2680] | 7.4902 / 7.3804 |  |
| LONGITUDINAL | tactical_2_6s:speed_mae_mps|all | 2399 | -0.0133 [-0.0613, 0.0339] | 1.6792 / 1.6925 |  |
| LONGITUDINAL | tactical_2_6s:speed_mae_mps|stop_point | 117 | -0.1799 [-0.5116, 0.1412] | 3.4301 / 3.6100 |  |
| LONGITUDINAL | tactical_2_6s:along_mae_m|all | 2399 | 0.0223 [-0.0277, 0.0753] | 2.7367 / 2.7144 |  |
| LONGITUDINAL | tactical_2_6s:accel_mae_mps2|all | 2399 | -0.0372 [-0.2370, 0.1415] | 2.1262 / 2.1634 |  |
| LATERAL | tactical_2_6s:cross_mae_m|all | 2399 | -0.0017 [-0.0377, 0.0350] | 1.4106 / 1.4123 |  |
| LATERAL | tactical_2_6s:heading_mae_deg|all | 2348 | 1.2499 [0.6610, 1.9510] | 10.8615 / 9.6116 | WORSE |
| LATERAL | tactical_2_6s:yaw_rate_mae_degps|all | 2340 | 3.7679 [1.9206, 5.8575] | 41.6428 / 37.8749 | WORSE |
| LATERAL | tactical_2_6s:curvature_mae_1pm|all | 2340 | 0.0441 [0.0203, 0.0717] | 0.2808 / 0.2367 | WORSE |
| LONGITUDINAL | tactical_2_6s:dk_hw_abs_err_m|all | 550 | -0.0244 [-0.1246, 0.0684] | 2.2412 / 2.2657 |  |
| LONGITUDINAL | tactical_2_6s:dk_tg_abs_err_s|all | 526 | -0.0168 [-0.0551, 0.0164] | 0.3889 / 0.4057 |  |
| PRIMARY | full_0_6s:ade_m|all | 2399 | 0.0177 [-0.0170, 0.0537] | 2.3438 / 2.3261 |  |
| PRIMARY | full_0_6s:ade_m|stop_point | 117 | -0.2328 [-0.4123, -0.0444] | 3.4548 / 3.6876 | better |
| PRIMARY | full_0_6s:fde_m|all | 2399 | 0.1098 [-0.0393, 0.2680] | 7.4902 / 7.3804 |  |
| LONGITUDINAL | full_0_6s:speed_mae_mps|all | 2399 | -0.0090 [-0.0413, 0.0228] | 1.2085 / 1.2176 |  |
| LONGITUDINAL | full_0_6s:speed_mae_mps|stop_point | 117 | -0.1254 [-0.3488, 0.0895] | 2.4303 / 2.5558 |  |
| LONGITUDINAL | full_0_6s:along_mae_m|all | 2399 | 0.0145 [-0.0192, 0.0499] | 1.8856 / 1.8711 |  |
| LONGITUDINAL | full_0_6s:accel_mae_mps2|all | 2399 | -0.0239 [-0.1540, 0.0939] | 1.6169 / 1.6408 |  |
| LATERAL | full_0_6s:cross_mae_m|all | 2399 | -0.0012 [-0.0254, 0.0233] | 0.9671 / 0.9683 |  |
| LATERAL | full_0_6s:heading_mae_deg|all | 2373 | 0.8910 [0.4035, 1.5093] | 8.4110 / 7.5200 | WORSE |
| LATERAL | full_0_6s:yaw_rate_mae_degps|all | 2365 | 1.8994 [0.2971, 3.4283] | 31.1526 / 29.2532 | WORSE |
| LATERAL | full_0_6s:curvature_mae_1pm|all | 2365 | 0.0249 [0.0062, 0.0450] | 0.2193 / 0.1945 | WORSE |
| TACTICAL | TAC-LAT|all | 2399 | -0.0271 [-0.0455, -0.0096] | 0.7549 / 0.7820 | WORSE |
| TACTICAL | TAC-LON|all | 2399 | -0.0121 [-0.0275, 0.0029] | 0.4935 / 0.5056 |  |
| TACTICAL | TAC-LON-macroF1|all | 2399 | -0.0129 [-0.0288, 0.0026] | 0.4933 / 0.5062 |  |
| STRATEGIC | | | PAUSED (PI item 25) | | |

