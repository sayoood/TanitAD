# RESULTS_TRAINFIT -- RETIME gains fitted on TRAIN clips, frozen, scored on eval139 (EMA-30k, T1 open loop, single training seed)

prereg (v0 + AMENDMENT 1) sha256 `648cdee6e68f70c2a91f518dbcce2999e846d5de6759a7deaf8a7d07311096f6`; frozen gains sha256 `b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee`; fit source = train, fit clips 139, overlap with eval139 0, registered A6 sample: True; seeds pooled [0, 1] 
Estimator: paired episode-cluster bootstrap over eval episodes (B 2000, seed 0, gains and models fixed) = 'another draw of EVAL EPISODES' only; not another train-clip draw, training run or inference run.

## OUTCOME (arm RETIME, committed table PREREG section 5): **DEPLOYABLE-PASS**

```
{
 "label": "DEPLOYABLE-PASS",
 "E1_verdict": "PASS (single training seed; training-seed floor unmeasured)",
 "E1P_separated_both_seeds": true,
 "ACC_LAT_label": "HOLDS",
 "NR_per_family": {
  "ADE": "PASS",
  "LONGITUDINAL": "PASS",
  "LATERAL": "PASS"
 },
 "MOVES_S2_vs_os_separated_better_both_seeds": true,
 "MOVED_S6_vs_os_separated_better_both_seeds": true,
 "fail_reasons": []
}
```
GATED_RETIME (beside, cannot rescue or sink): DEPLOYABLE-PASS

## CONST_G05 reading (AMENDMENT 1 A1.3; does not change the label): RETIME **MIXED** | GATED_RETIME MIXED
some cells separate the fit from the constant, not all four, none the other way.

train-fit - CONST_G05, ADE, compliant, EMA-30k:

* S2|s0: -0.0010 [-0.0020, -0.0000] **SEP**
* S2|s1: -0.0010 [-0.0020, -0.0001] **SEP**
* S6|s0: +0.0341 [-0.0149, +0.0828]
* S6|s1: +0.0359 [-0.0154, +0.0860]

## Known-value controls: PASS (else the run is withheld). CONST_G1 - os exactly 0.0 with zero-width CI on 112 rows; CONST_G1 bit-equal os and CONST_G0 == independent prior-arc-length walker (<= 1e-06 m) on 4 cells: max walker error 7.29e-11 m.

## Gains per instant: train-fit RETIME vs eval-fit (FROZEN_GAINS_RETIME_ema30k.json, all eval windows, seeds pooled) and the constants

S2: instant(s) 0.5 1 1.5 2
* train-fit : 0.967 0.619 0.551 0.531
* eval-fit  : 0.997 0.583 0.426 0.382
* CONST_G1/G05/G0: 1 / 0.5 / 0

S6: instant(s) 1 2 3 4 5 6
* train-fit : 0.639 0.597 0.655 0.699 0.722 0.745
* eval-fit  : 0.604 0.421 0.464 0.508 0.530 0.555
* CONST_G1/G05/G0: 1 / 0.5 / 0

## Same table, S2 and S6, ceiling-compliant, EMA-30k (ADE; arm minus reference; s0 | s1)

### S2

| arm | - ha0_ext s0 | s1 | - ha0_ext_pose s0 | s1 | - os s0 | s1 |
|---|---|---|---|---|---|---|
| RETIME_TRAINFIT | -0.0267 [-0.0378, -0.0150] **SEP** | -0.0262 [-0.0372, -0.0145] **SEP** | -0.0318 [-0.0433, -0.0201] **SEP** | -0.0313 [-0.0426, -0.0194] **SEP** | -0.0182 [-0.0235, -0.0132] **SEP** | -0.0186 [-0.0238, -0.0137] **SEP** |
| GATED_RETIME_TRAINFIT | -0.0296 [-0.0408, -0.0181] **SEP** | -0.0289 [-0.0393, -0.0175] **SEP** | -0.0348 [-0.0463, -0.0231] **SEP** | -0.0340 [-0.0452, -0.0225] **SEP** | -0.0211 [-0.0271, -0.0156] **SEP** | -0.0213 [-0.0272, -0.0160] **SEP** |
| CONST_G05 (comparator) | -0.0257 [-0.0364, -0.0145] **SEP** | -0.0252 [-0.0359, -0.0141] **SEP** | -0.0308 [-0.0420, -0.0196] **SEP** | -0.0303 [-0.0413, -0.0190] **SEP** | -0.0171 [-0.0233, -0.0115] **SEP** | -0.0176 [-0.0236, -0.0119] **SEP** |
| CONST_G1 (control) | -0.0085 [-0.0229, +0.0065] | -0.0076 [-0.0216, +0.0073] | -0.0136 [-0.0286, +0.0018] | -0.0127 [-0.0272, +0.0024] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| CONST_G0 (control) | -0.0067 [-0.0152, +0.0014] | -0.0062 [-0.0147, +0.0019] | -0.0118 [-0.0209, -0.0035] **SEP** | -0.0113 [-0.0202, -0.0028] **SEP** | +0.0018 [-0.0100, +0.0128] | +0.0014 [-0.0105, +0.0118] |

### S6

| arm | - ha0_ext s0 | s1 | - ha0_ext_pose s0 | s1 | - os s0 | s1 |
|---|---|---|---|---|---|---|
| RETIME_TRAINFIT | -0.9475 [-1.2471, -0.6759] **SEP** | -0.9438 [-1.2358, -0.6722] **SEP** | -0.9763 [-1.2770, -0.7007] **SEP** | -0.9726 [-1.2664, -0.6931] **SEP** | -0.1940 [-0.2518, -0.1366] **SEP** | -0.2009 [-0.2621, -0.1410] **SEP** |
| GATED_RETIME_TRAINFIT | -0.9318 [-1.2275, -0.6615] **SEP** | -0.9276 [-1.2213, -0.6556] **SEP** | -0.9606 [-1.2627, -0.6825] **SEP** | -0.9563 [-1.2503, -0.6770] **SEP** | -0.1783 [-0.2387, -0.1183] **SEP** | -0.1846 [-0.2474, -0.1248] **SEP** |
| CONST_G05 (comparator) | -0.9816 [-1.2509, -0.7400] **SEP** | -0.9797 [-1.2446, -0.7426] **SEP** | -1.0104 [-1.2845, -0.7643] **SEP** | -1.0084 [-1.2794, -0.7669] **SEP** | -0.2281 [-0.3320, -0.1217] **SEP** | -0.2367 [-0.3426, -0.1293] **SEP** |
| CONST_G1 (control) | -0.7535 [-1.0832, -0.4506] **SEP** | -0.7430 [-1.0699, -0.4363] **SEP** | -0.7823 [-1.1159, -0.4748] **SEP** | -0.7717 [-1.1024, -0.4582] **SEP** | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| CONST_G0 (control) | -0.6645 [-0.8911, -0.4677] **SEP** | -0.6590 [-0.8806, -0.4676] **SEP** | -0.6933 [-0.9185, -0.4915] **SEP** | -0.6878 [-0.9095, -0.4865] **SEP** | +0.0890 [-0.0937, +0.2767] | +0.0839 [-0.0994, +0.2719] |

## All four families, RETIME_TRAINFIT, compliant (arm - os, arm - refcv7-50,400; s0 | s1). STRATEGIC: NOT APPLICABLE, n = 0 (--no-strategic).

### S2

| family | row | - os s0 | s1 | - refcv7 s0 | s1 |
|---|---|---|---|---|---|
| ADE | ade_m | -0.0182 [-0.0235, -0.0132] **SEP** | -0.0186 [-0.0238, -0.0137] **SEP** | -0.0231 [-0.0353, -0.0120] **SEP** | -0.0224 [-0.0349, -0.0111] **SEP** |
| LONGITUDINAL | LON_speed_mae_mps | -0.0276 [-0.0347, -0.0207] **SEP** | -0.0286 [-0.0356, -0.0220] **SEP** | -0.0314 [-0.0454, -0.0188] **SEP** | -0.0312 [-0.0455, -0.0187] **SEP** |
| LONGITUDINAL | LON_along_abs_final_m | -0.0469 [-0.0617, -0.0330] **SEP** | -0.0497 [-0.0640, -0.0362] **SEP** | -0.0519 [-0.0804, -0.0259] **SEP** | -0.0512 [-0.0801, -0.0255] **SEP** |
| LATERAL | LAT_heading_mae_deg | -0.0018 [-0.0079, +0.0044] | -0.0012 [-0.0068, +0.0048] | +0.0032 [-0.0391, +0.0468] | +0.0218 [-0.0282, +0.0719] |
| LATERAL | LAT_curvature_mae_1pm_valid | -0.0000 [-0.0001, -0.0000] **SEP** | -0.0000 [-0.0001, +0.0000] | -0.0001 [-0.0004, +0.0002] | +0.0001 [-0.0002, +0.0003] |
| LATERAL | LAT_yaw_rate_mae_radps_valid | -0.0003 [-0.0005, -0.0001] **SEP** | -0.0003 [-0.0004, -0.0001] **SEP** | -0.0005 [-0.0014, +0.0004] | -0.0002 [-0.0011, +0.0008] |

### S6

| family | row | - os s0 | s1 | - refcv7 s0 | s1 |
|---|---|---|---|---|---|
| ADE | ade_m | -0.1940 [-0.2518, -0.1366] **SEP** | -0.2009 [-0.2621, -0.1410] **SEP** | -0.0753 [-0.2136, +0.0644] | -0.0637 [-0.2029, +0.0763] |

## Train vs eval139 os ADE (A = TRAIN, B = EVAL; unpaired episode-cluster bootstrap). Headline = os / ha0_ext per set (difficulty-normalised)

* S2|all|s0: os ADE A 0.2642 B 0.2878 (diff -0.0235 [-0.0579, +0.0156]); os/ha0_ext A 0.8945 B 0.9971 (diff -0.1026 [-0.1744, -0.0308])
* S2|compliant|s0: os ADE A 0.2389 B 0.2722 (diff -0.0333 [-0.0699, +0.0051]); os/ha0_ext A 0.8552 B 0.9696 (diff -0.1145 [-0.1840, -0.0446])
* S6|all|s0: os ADE A 2.4054 B 2.7333 (diff -0.3278 [-0.6274, -0.0015]); os/ha0_ext A 0.6192 B 0.7704 (diff -0.1512 [-0.2370, -0.0713])
* S6|compliant|s0: os ADE A 2.2287 B 2.6275 (diff -0.3988 [-0.7091, -0.1049]); os/ha0_ext A 0.6257 B 0.7771 (diff -0.1515 [-0.2431, -0.0645])
* S2|all|s1: os ADE A 0.2647 B 0.2901 (diff -0.0254 [-0.0591, +0.0131]); os/ha0_ext A 0.8960 B 1.0050 (diff -0.1090 [-0.1813, -0.0348])
* S2|compliant|s1: os ADE A 0.2395 B 0.2731 (diff -0.0335 [-0.0703, +0.0053]); os/ha0_ext A 0.8574 B 0.9728 (diff -0.1155 [-0.1833, -0.0461])
* S6|all|s1: os ADE A 2.4107 B 2.7577 (diff -0.3470 [-0.6483, -0.0251]); os/ha0_ext A 0.6206 B 0.7773 (diff -0.1567 [-0.2435, -0.0767])
* S6|compliant|s1: os ADE A 2.2338 B 2.6381 (diff -0.4043 [-0.7158, -0.1090]); os/ha0_ext A 0.6271 B 0.7803 (diff -0.1532 [-0.2457, -0.0658])

Full cells (both sets, raw-30k beside, GATED, all rows): `FORWARD_TRAINFIT_step30000ema_unc.json`, `FORWARD_TRAINFIT_step30000_unc.json`; gap: `GAP_train_vs_eval.json`.
