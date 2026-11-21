# A0K0_RETIME -- ARC-LENGTH RE-TIMING of refcv8's plan (path kept, progress re-timed; zero training, banked panels)

* **Status:** EARLY READ of banked step 30,000. ARC-LENGTH RE-TIMING (path kept, progress re-timed), motivated AFTER the LON arm failed its pre-registered ACC-LAT (RS_LON_XF - os separated WORSE on S6 heading / curvature / yaw-rate). E1 AS REGISTERED (os - ha0_ext) stays the bar of record. All arms are cross-fitted calibrations fit on EVAL episodes: lever estimates, never a bar pass (a deployable version fits g on TRAIN).
* **Tier:** T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
* **Estimator:** paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, models FIXED: 'another draw of EPISODES' only; NOT another training run (single training seed), NOT another inference run (seed 1 is the replicate beside seed 0), NOT another fold assignment (fold-seed sensitivity reported separately)
* Pre-registration `raw/a0k0_retime/PREREG_RETIME.md` (sha256 `dc02f704fd09`, written before any number of the new arms); script sha256 `842889130005`; parent `A0K0_HEADROOM.json` sha256 `37a6e10bcc27`.
* **Disclosure:** this arm was motivated by the LON arm FAILING ACC-LAT (RS_LON_XF - os separated WORSE on S6 heading / curvature / yaw-rate), seen before the registration; the gains are fit on EVAL episodes (cross-fitted) -- a LEVER estimate, never a bar pass.
* Controls **ALL_PASS = True** (a (g1 == banked os, every row exact; the re-timing at g == 1 is bit-equal to os), b (blend g0 == ha0_ext_pose bit-exact, 0-cell), h (recomputed RS_XF == banked A0K0_HEADROOM.json RS_XF cells, 1e-9), i (every re-timed point lies ON the os polyline: float64 distance <= 1e-6 m), decisions (ws, ep_poses[ws] == pose_last, v0), V1 (ha0_ext rebuilt <= 1e-3 m), mask sha, K). 2026-09-26 reproduction (S2 all windows, ha0_ext_pose - ha0_ext): +0.0055 [+0.0036, +0.0076] **SEP** vs banked +0.0055 [+0.0036, +0.0076] **SEP** -> exact = **True**.
* STRATEGIC family: NOT APPLICABLE, n = 0 (the run argv carries --no-strategic: the layer is off).

## ACCEPTANCE (ACC-LAT): no LATERAL row of arm - os separated WORSE at EITHER seed -- S6, ceiling-compliant, EMA-30k

* **RETIME_XF: ACC-LAT HOLDS** -- separated worse at any seed: none; at both seeds: none
* **GATED_RETIME_XF: ACC-LAT HOLDS** -- separated worse at any seed: none; at both seeds: none

## FROZEN gains -- fit on EMA-30k, ALL windows, both seeds pooled

`FROZEN_GAINS_RETIME_ema30k.json` sha256 `4ecde275554b`; fit set: ALL windows of the surface (all 139 eval episodes), inference seeds 0 and 1 POOLED (each seed's os against the same prior and GT).

* **S2** instants [0.5, 1.0, 1.5, 2.0] s, n 4754 windows / 139 ep: RETIME g per instant = [0.997, 0.583, 0.426, 0.382]; in-sample pooled ADE os 0.2889 / prior 0.2941 / RETIME 0.2639 / GATED_RETIME 0.2593
  * GATED_RETIME v0<2 (n 465): g [0.118, 0.108, 0.128, 0.168]
  * GATED_RETIME v0 2-8 (n 1306): g [0.629, 0.289, 0.339, 0.422]
  * GATED_RETIME v0 8-15 (n 1792): g [0.823, 0.421, 0.324, 0.334]
  * GATED_RETIME v0 15-25 (n 856): g [1.279, 0.823, 0.721, 0.459]
  * GATED_RETIME v0>=25 (n 335): g [1.376, 0.867, 0.834, 0.819]
* **S6** instants [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s, n 3668 windows / 139 ep: RETIME g per instant = [0.604, 0.421, 0.464, 0.508, 0.53, 0.555]; in-sample pooled ADE os 2.7455 / prior 3.5764 / RETIME 2.4792 / GATED_RETIME 2.4552
  * GATED_RETIME v0<2 (n 330): g [0.152, 0.262, 0.335, 0.43, 0.452, 0.473]
  * GATED_RETIME v0 2-8 (n 1017): g [0.326, 0.472, 0.59, 0.638, 0.663, 0.669]
  * GATED_RETIME v0 8-15 (n 1397): g [0.45, 0.363, 0.424, 0.479, 0.512, 0.534]
  * GATED_RETIME v0 15-25 (n 679): g [0.828, 0.403, 0.142, 0.169, 0.203, 0.22]
  * GATED_RETIME v0>=25 (n 245): g [0.864, 0.855, 0.725, 0.543, 0.446, 0.386]

## FORWARD step30000_unc -- path check on a different checkpoint of the SAME run (not an independent test, not a bar) (controls True)

| set / arm | FZR-E1 (-ha0_ext, S2) | FZR-E1P (-ha0_ext_pose, S2) | FZR-E3 (-ha0_ext, S6) | FZR-NR per family | FZR-LAT (S6 vs os) |
|---|---|---|---|---|---|
| compliant|RETIME_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0127 [-0.0237, -0.0015] / -0.0120 [-0.0229, -0.0007] | PASS (single training seed; training-seed floor unmeasured): -0.0178 [-0.0291, -0.0065] / -0.0171 [-0.0282, -0.0053] | PASS (single training seed; training-seed floor unmeasured): -0.9038 [-1.1696, -0.6564] / -0.8986 [-1.1710, -0.6486] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid'} | holds=True; worse at any seed: none |
| compliant|GATED_RETIME_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0168 [-0.0274, -0.0057] / -0.0157 [-0.0264, -0.0044] | PASS (single training seed; training-seed floor unmeasured): -0.0219 [-0.0328, -0.0109] / -0.0208 [-0.0314, -0.0093] | PASS (single training seed; training-seed floor unmeasured): -0.9435 [-1.2085, -0.7033] / -0.9373 [-1.2012, -0.6977] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid'} | holds=True; worse at any seed: none |
| all|RETIME_FROZEN30k | FAILED: -0.0113 [-0.0222, -0.0005] / -0.0098 [-0.0207, +0.0012] | PASS (single training seed; training-seed floor unmeasured): -0.0168 [-0.0287, -0.0054] / -0.0153 [-0.0272, -0.0035] | PASS (single training seed; training-seed floor unmeasured): -0.9794 [-1.2512, -0.7206] / -0.9744 [-1.2557, -0.7127] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid'} | holds=True; worse at any seed: none |
| all|GATED_RETIME_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0153 [-0.0263, -0.0044] / -0.0136 [-0.0249, -0.0025] | PASS (single training seed; training-seed floor unmeasured): -0.0208 [-0.0325, -0.0092] / -0.0192 [-0.0311, -0.0077] | PASS (single training seed; training-seed floor unmeasured): -1.0157 [-1.2910, -0.7542] / -1.0105 [-1.2906, -0.7488] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid'} | holds=True; worse at any seed: none |

## FORWARD step30000ema_unc -- IN-SAMPLE path check (the gains were fit on this tree) (controls True)

| set / arm | FZR-E1 (-ha0_ext, S2) | FZR-E1P (-ha0_ext_pose, S2) | FZR-E3 (-ha0_ext, S6) | FZR-NR per family | FZR-LAT (S6 vs os) |
|---|---|---|---|---|---|
| compliant|RETIME_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0269 [-0.0373, -0.0161] / -0.0264 [-0.0367, -0.0158] | PASS (single training seed; training-seed floor unmeasured): -0.0320 [-0.0427, -0.0211] / -0.0315 [-0.0420, -0.0207] | PASS (single training seed; training-seed floor unmeasured): -0.9839 [-1.2570, -0.7391] / -0.9820 [-1.2499, -0.7405] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'} | holds=True; worse at any seed: none |
| compliant|GATED_RETIME_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0315 [-0.0419, -0.0206] / -0.0306 [-0.0406, -0.0199] | PASS (single training seed; training-seed floor unmeasured): -0.0366 [-0.0474, -0.0257] / -0.0357 [-0.0462, -0.0248] | PASS (single training seed; training-seed floor unmeasured): -1.0066 [-1.2799, -0.7696] / -1.0066 [-1.2736, -0.7710] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'} | holds=True; worse at any seed: none |
| all|RETIME_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0253 [-0.0358, -0.0150] / -0.0242 [-0.0349, -0.0139] | PASS (single training seed; training-seed floor unmeasured): -0.0308 [-0.0420, -0.0199] / -0.0297 [-0.0410, -0.0188] | PASS (single training seed; training-seed floor unmeasured): -1.0747 [-1.3573, -0.8110] / -1.0626 [-1.3411, -0.8007] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'} | holds=True; worse at any seed: none |
| all|GATED_RETIME_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0301 [-0.0408, -0.0192] / -0.0285 [-0.0393, -0.0178] | PASS (single training seed; training-seed floor unmeasured): -0.0357 [-0.0469, -0.0246] / -0.0341 [-0.0454, -0.0227] | PASS (single training seed; training-seed floor unmeasured): -1.0983 [-1.3833, -0.8348] / -1.0870 [-1.3683, -0.8298] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: S2:LAT_heading_mae_deg'} | holds=True; worse at any seed: none |

## FORWARD step40000ema_unc -- FORWARD TEST (PREREG_RETIME bars FZR-*) (controls True)

| set / arm | FZR-E1 (-ha0_ext, S2) | FZR-E1P (-ha0_ext_pose, S2) | FZR-E3 (-ha0_ext, S6) | FZR-NR per family | FZR-LAT (S6 vs os) |
|---|---|---|---|---|---|
| compliant|RETIME_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0270 [-0.0380, -0.0160] / -0.0267 [-0.0374, -0.0159] | PASS (single training seed; training-seed floor unmeasured): -0.0321 [-0.0433, -0.0210] / -0.0319 [-0.0428, -0.0206] | PASS (single training seed; training-seed floor unmeasured): -0.9709 [-1.2492, -0.7168] / -0.9742 [-1.2485, -0.7275] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'} | holds=True; worse at any seed: none |
| compliant|GATED_RETIME_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0299 [-0.0407, -0.0187] / -0.0290 [-0.0397, -0.0182] | PASS (single training seed; training-seed floor unmeasured): -0.0350 [-0.0462, -0.0237] / -0.0341 [-0.0452, -0.0230] | PASS (single training seed; training-seed floor unmeasured): -1.0091 [-1.2870, -0.7611] / -1.0105 [-1.2810, -0.7674] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'} | holds=True; worse at any seed: none |
| all|RETIME_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0248 [-0.0359, -0.0138] / -0.0243 [-0.0354, -0.0136] | PASS (single training seed; training-seed floor unmeasured): -0.0303 [-0.0418, -0.0187] / -0.0298 [-0.0415, -0.0185] | PASS (single training seed; training-seed floor unmeasured): -1.0613 [-1.3568, -0.7917] / -1.0535 [-1.3432, -0.7877] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid'} | holds=True; worse at any seed: none |
| all|GATED_RETIME_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0282 [-0.0392, -0.0171] / -0.0271 [-0.0383, -0.0160] | PASS (single training seed; training-seed floor unmeasured): -0.0337 [-0.0454, -0.0224] / -0.0326 [-0.0444, -0.0211] | PASS (single training seed; training-seed floor unmeasured): -1.0994 [-1.3976, -0.8331] / -1.0903 [-1.3782, -0.8255] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: S2:LAT_heading_mae_deg'} | holds=True; worse at any seed: none |

## FORWARD step48852_unc -- FORWARD TEST (PREREG_RETIME bars FZR-*) (controls True)

| set / arm | FZR-E1 (-ha0_ext, S2) | FZR-E1P (-ha0_ext_pose, S2) | FZR-E3 (-ha0_ext, S6) | FZR-NR per family | FZR-LAT (S6 vs os) |
|---|---|---|---|---|---|
| compliant|RETIME_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0214 [-0.0328, -0.0101] / -0.0205 [-0.0318, -0.0094] | PASS (single training seed; training-seed floor unmeasured): -0.0265 [-0.0385, -0.0149] / -0.0256 [-0.0376, -0.0140] | PASS (single training seed; training-seed floor unmeasured): -0.8891 [-1.1793, -0.6248] / -0.8896 [-1.1699, -0.6291] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid'} | holds=True; worse at any seed: none |
| compliant|GATED_RETIME_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0223 [-0.0346, -0.0097] / -0.0211 [-0.0333, -0.0083] | PASS (single training seed; training-seed floor unmeasured): -0.0274 [-0.0399, -0.0145] / -0.0262 [-0.0386, -0.0135] | PASS (single training seed; training-seed floor unmeasured): -0.9471 [-1.2328, -0.6931] / -0.9510 [-1.2258, -0.6971] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: S2:LAT_heading_mae_deg'} | holds=True; worse at any seed: none |
| all|RETIME_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0197 [-0.0313, -0.0081] / -0.0177 [-0.0294, -0.0061] | PASS (single training seed; training-seed floor unmeasured): -0.0252 [-0.0373, -0.0132] / -0.0232 [-0.0358, -0.0112] | PASS (single training seed; training-seed floor unmeasured): -0.9827 [-1.2769, -0.7049] / -0.9727 [-1.2668, -0.6981] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid'} | holds=True; worse at any seed: none |
| all|GATED_RETIME_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0213 [-0.0336, -0.0089] / -0.0190 [-0.0311, -0.0065] | PASS (single training seed; training-seed floor unmeasured): -0.0268 [-0.0396, -0.0137] / -0.0245 [-0.0372, -0.0113] | PASS (single training seed; training-seed floor unmeasured): -1.0376 [-1.3283, -0.7685] / -1.0310 [-1.3223, -0.7642] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid'} | holds=True; worse at any seed: none |

## FORWARD step48852ema_unc -- FORWARD TEST (PREREG_RETIME bars FZR-*) (controls True)

| set / arm | FZR-E1 (-ha0_ext, S2) | FZR-E1P (-ha0_ext_pose, S2) | FZR-E3 (-ha0_ext, S6) | FZR-NR per family | FZR-LAT (S6 vs os) |
|---|---|---|---|---|---|
| compliant|RETIME_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0233 [-0.0352, -0.0116] / -0.0236 [-0.0349, -0.0126] | PASS (single training seed; training-seed floor unmeasured): -0.0284 [-0.0408, -0.0163] / -0.0288 [-0.0407, -0.0173] | PASS (single training seed; training-seed floor unmeasured): -0.9160 [-1.2045, -0.6508] / -0.9161 [-1.1899, -0.6612] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'} | holds=True; worse at any seed: none |
| compliant|GATED_RETIME_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0240 [-0.0363, -0.0112] / -0.0243 [-0.0358, -0.0124] | PASS (single training seed; training-seed floor unmeasured): -0.0291 [-0.0418, -0.0160] / -0.0294 [-0.0415, -0.0175] | PASS (single training seed; training-seed floor unmeasured): -0.9691 [-1.2530, -0.7121] / -0.9676 [-1.2413, -0.7148] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'} | holds=True; worse at any seed: none |
| all|RETIME_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0212 [-0.0331, -0.0094] / -0.0206 [-0.0324, -0.0092] | PASS (single training seed; training-seed floor unmeasured): -0.0267 [-0.0390, -0.0144] / -0.0261 [-0.0382, -0.0143] | PASS (single training seed; training-seed floor unmeasured): -1.0065 [-1.3019, -0.7297] / -0.9987 [-1.2912, -0.7263] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid'} | holds=True; worse at any seed: none |
| all|GATED_RETIME_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0227 [-0.0354, -0.0101] / -0.0219 [-0.0336, -0.0099] | PASS (single training seed; training-seed floor unmeasured): -0.0282 [-0.0413, -0.0150] / -0.0275 [-0.0400, -0.0147] | PASS (single training seed; training-seed floor unmeasured): -1.0579 [-1.3544, -0.7892] / -1.0491 [-1.3418, -0.7844] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: S2:LAT_heading_mae_deg'} | holds=True; worse at any seed: none |

References: `ha0_ext` = E1 echo (steer kappa0, INADMISSIBLE input); `ha0_ext_pose` = refcv8's own causal prior (admissible). Arms: p = C_os(s'), s' = s_prior + g (s_os - s_prior), g one scalar per instant, 5-episode-fold cross-fitted. RETIME_XF: one table; GATED_RETIME_XF: per v0 bin; RS_XF = the Amendment-2 arm (control, recomputed and equal to the banked cells); _IS in-sample, _WF window-fold (leaky) controls.

## step30000ema_unc

### S2 / compliant -- n 4159 windows / 138 ep

ADE means (s0): os 0.2722, ha0_ext 0.2807, ha0_ext_pose 0.2858, ha0_imu 0.2823, ha0_imu_ax 0.3875, refcv7_50400 0.2771

| read | seed 0 | seed 1 |
|---|---|---|
| **E1 AS REGISTERED: os - ha0_ext (bar of record)** | -0.0085 [-0.0229, +0.0065] | -0.0076 [-0.0216, +0.0073] |
| os - ha0_ext_pose | -0.0136 [-0.0286, +0.0018] | -0.0127 [-0.0272, +0.0024] |
| RS_XF - ha0_ext (LE-E1 form; lever, NOT a bar) | -0.0428 [-0.0538, -0.0321] **SEP** | -0.0427 [-0.0534, -0.0321] **SEP** |
| RS_XF - ha0_ext_pose (LE-E1P form) | -0.0479 [-0.0596, -0.0367] **SEP** | -0.0479 [-0.0593, -0.0368] **SEP** |
| RS_XF - os (ADE) | -0.0343 [-0.0419, -0.0274] **SEP** | -0.0351 [-0.0425, -0.0284] **SEP** |
| RETIME_XF - ha0_ext (LE-E1 form; lever, NOT a bar) | -0.0269 [-0.0376, -0.0156] **SEP** | -0.0265 [-0.0371, -0.0153] **SEP** |
| RETIME_XF - ha0_ext_pose (LE-E1P form) | -0.0321 [-0.0432, -0.0208] **SEP** | -0.0316 [-0.0425, -0.0202] **SEP** |
| RETIME_XF - os (ADE) | -0.0184 [-0.0245, -0.0127] **SEP** | -0.0189 [-0.0249, -0.0134] **SEP** |
| GATED_RETIME_XF - ha0_ext (LE-E1 form; lever, NOT a bar) | -0.0310 [-0.0419, -0.0201] **SEP** | -0.0301 [-0.0402, -0.0191] **SEP** |
| GATED_RETIME_XF - ha0_ext_pose (LE-E1P form) | -0.0362 [-0.0472, -0.0252] **SEP** | -0.0352 [-0.0458, -0.0242] **SEP** |
| GATED_RETIME_XF - os (ADE) | -0.0225 [-0.0296, -0.0161] **SEP** | -0.0225 [-0.0294, -0.0163] **SEP** |
| RETIME_IS - ha0_ext | -0.0271 [-0.0377, -0.0159] **SEP** | -0.0266 [-0.0373, -0.0155] **SEP** |
| RETIME_IS - ha0_ext_pose (LE-E1P form) | -0.0322 [-0.0434, -0.0210] **SEP** | -0.0317 [-0.0425, -0.0204] **SEP** |
| RETIME_IS - os (ADE) | -0.0186 [-0.0247, -0.0129] **SEP** | -0.0190 [-0.0250, -0.0135] **SEP** |
| GATED_RETIME_IS - ha0_ext | -0.0321 [-0.0429, -0.0211] **SEP** | -0.0312 [-0.0415, -0.0203] **SEP** |
| GATED_RETIME_IS - ha0_ext_pose (LE-E1P form) | -0.0372 [-0.0482, -0.0261] **SEP** | -0.0363 [-0.0469, -0.0252] **SEP** |
| GATED_RETIME_IS - os (ADE) | -0.0235 [-0.0308, -0.0169] **SEP** | -0.0235 [-0.0308, -0.0171] **SEP** |
| RETIME_WF - ha0_ext | -0.0271 [-0.0377, -0.0158] **SEP** | -0.0266 [-0.0372, -0.0155] **SEP** |
| RETIME_WF - ha0_ext_pose (LE-E1P form) | -0.0322 [-0.0433, -0.0209] **SEP** | -0.0317 [-0.0425, -0.0204] **SEP** |
| RETIME_WF - os (ADE) | -0.0186 [-0.0247, -0.0129] **SEP** | -0.0190 [-0.0250, -0.0135] **SEP** |
| GATED_RETIME_WF - ha0_ext | -0.0318 [-0.0426, -0.0209] **SEP** | -0.0307 [-0.0410, -0.0199] **SEP** |
| GATED_RETIME_WF - ha0_ext_pose (LE-E1P form) | -0.0369 [-0.0479, -0.0258] **SEP** | -0.0359 [-0.0465, -0.0248] **SEP** |
| GATED_RETIME_WF - os (ADE) | -0.0232 [-0.0305, -0.0167] **SEP** | -0.0231 [-0.0303, -0.0167] **SEP** |

| metric (family) | RETIME_XF - os s0 | s1 | RETIME_XF - refcv7 s0 | s1 | GATED_RETIME_XF - os s0 | s1 | GATED_RETIME_XF - refcv7 s0 | RS_XF (banked Amendment 2) - os s0 |
|---|---|---|---|---|---|---|---|---|
| ade_m (ADE) | -0.0184 [-0.0245, -0.0127] **SEP** | -0.0189 [-0.0249, -0.0134] **SEP** | -0.0234 [-0.0360, -0.0119] **SEP** | -0.0227 [-0.0355, -0.0110] **SEP** | -0.0225 [-0.0296, -0.0161] **SEP** | -0.0225 [-0.0294, -0.0163] **SEP** | -0.0275 [-0.0408, -0.0156] **SEP** | -0.0343 [-0.0419, -0.0274] **SEP** |
| fde_m (ADE) | -0.0464 [-0.0629, -0.0309] **SEP** | -0.0479 [-0.0637, -0.0329] **SEP** | -0.0560 [-0.0878, -0.0273] **SEP** | -0.0542 [-0.0858, -0.0255] **SEP** | -0.0504 [-0.0679, -0.0339] **SEP** | -0.0510 [-0.0675, -0.0355] **SEP** | -0.0599 [-0.0931, -0.0307] **SEP** | -0.0658 [-0.0834, -0.0496] **SEP** |
| LON_speed_mae_mps (LONGITUDINAL) | -0.0264 [-0.0348, -0.0184] **SEP** | -0.0277 [-0.0360, -0.0198] **SEP** | -0.0302 [-0.0449, -0.0175] **SEP** | -0.0302 [-0.0450, -0.0172] **SEP** | -0.0288 [-0.0378, -0.0204] **SEP** | -0.0297 [-0.0383, -0.0214] **SEP** | -0.0326 [-0.0483, -0.0187] **SEP** | -0.0250 [-0.0331, -0.0173] **SEP** |
| LON_along_mae_m (LONGITUDINAL) | -0.0191 [-0.0256, -0.0129] **SEP** | -0.0201 [-0.0265, -0.0141] **SEP** | -0.0214 [-0.0334, -0.0103] **SEP** | -0.0211 [-0.0331, -0.0102] **SEP** | -0.0238 [-0.0311, -0.0171] **SEP** | -0.0243 [-0.0314, -0.0178] **SEP** | -0.0261 [-0.0392, -0.0143] **SEP** | -0.0182 [-0.0245, -0.0124] **SEP** |
| LON_along_abs_final_m (LONGITUDINAL) | -0.0478 [-0.0651, -0.0311] **SEP** | -0.0507 [-0.0674, -0.0346] **SEP** | -0.0527 [-0.0828, -0.0255] **SEP** | -0.0521 [-0.0823, -0.0252] **SEP** | -0.0526 [-0.0704, -0.0359] **SEP** | -0.0546 [-0.0716, -0.0386] **SEP** | -0.0575 [-0.0894, -0.0290] **SEP** | -0.0447 [-0.0610, -0.0292] **SEP** |
| LON_speed_within_0p5 (LONGITUDINAL) | +0.0271 [+0.0185, +0.0359] **SEP** | +0.0269 [+0.0187, +0.0352] **SEP** | +0.0254 [+0.0145, +0.0374] **SEP** | +0.0257 [+0.0148, +0.0377] **SEP** | +0.0256 [+0.0173, +0.0344] **SEP** | +0.0263 [+0.0185, +0.0342] **SEP** | +0.0239 [+0.0124, +0.0365] **SEP** | +0.0245 [+0.0166, +0.0327] **SEP** |
| LON_accel_mae_mps2 (LONGITUDINAL) | -0.0601 [-0.0729, -0.0468] **SEP** | -0.0615 [-0.0747, -0.0485] **SEP** | -0.0720 [-0.0910, -0.0539] **SEP** | -0.0715 [-0.0908, -0.0538] **SEP** | -0.0557 [-0.0679, -0.0435] **SEP** | -0.0568 [-0.0686, -0.0451] **SEP** | -0.0676 [-0.0868, -0.0498] **SEP** | -0.0582 [-0.0703, -0.0460] **SEP** |
| LAT_heading_mae_deg (LATERAL) | -0.0033 [-0.0108, +0.0041] | -0.0023 [-0.0091, +0.0048] | +0.0018 [-0.0406, +0.0450] | +0.0207 [-0.0295, +0.0706] | -0.0021 [-0.0113, +0.0065] | -0.0004 [-0.0084, +0.0077] | +0.0065 [-0.0356, +0.0495] | -0.1436 [-0.1813, -0.1086] **SEP** |
| LAT_curvature_mae_1pm_valid (LATERAL) | -0.0000 [-0.0001, -0.0000] **SEP** | -0.0000 [-0.0001, +0.0000] | -0.0001 [-0.0004, +0.0002] | +0.0001 [-0.0002, +0.0003] | -0.0001 [-0.0001, -0.0000] **SEP** | -0.0001 [-0.0001, -0.0000] **SEP** | -0.0001 [-0.0004, +0.0002] | +0.0001 [-0.0002, +0.0005] |
| LAT_yaw_rate_mae_radps_valid (LATERAL) | -0.0003 [-0.0005, -0.0001] **SEP** | -0.0003 [-0.0005, -0.0001] **SEP** | -0.0005 [-0.0014, +0.0003] | -0.0002 [-0.0011, +0.0008] | -0.0003 [-0.0006, -0.0001] **SEP** | -0.0003 [-0.0005, -0.0001] **SEP** | -0.0005 [-0.0014, +0.0004] | -0.0011 [-0.0017, -0.0006] **SEP** |
| LAT_cross_mae_m (LATERAL) | -0.0010 [-0.0021, -0.0000] **SEP** | -0.0009 [-0.0020, +0.0000] | -0.0027 [-0.0076, +0.0023] | -0.0024 [-0.0077, +0.0027] | -0.0010 [-0.0021, +0.0001] | -0.0009 [-0.0020, +0.0001] | -0.0026 [-0.0076, +0.0024] | -0.0253 [-0.0314, -0.0199] **SEP** |
| LAT_cross_abs_final_m (LATERAL) | -0.0033 [-0.0067, -0.0002] **SEP** | -0.0031 [-0.0065, -0.0001] **SEP** | -0.0047 [-0.0160, +0.0069] | -0.0040 [-0.0159, +0.0079] | -0.0034 [-0.0068, -0.0002] **SEP** | -0.0031 [-0.0064, -0.0002] **SEP** | -0.0048 [-0.0160, +0.0069] | -0.0362 [-0.0466, -0.0266] **SEP** |
| TAC_traj_lat_correct (TACTICAL) | -0.0002 [-0.0017, +0.0010] | -0.0007 [-0.0024, +0.0007] | -0.0019 [-0.0065, +0.0027] | -0.0029 [-0.0078, +0.0017] | +0.0002 [-0.0014, +0.0019] | +0.0000 [-0.0017, +0.0017] | -0.0014 [-0.0064, +0.0036] | +0.0005 [-0.0031, +0.0044] |
| TAC_traj_lon_correct (TACTICAL) | +0.0221 [+0.0102, +0.0344] **SEP** | +0.0233 [+0.0108, +0.0357] **SEP** | +0.0289 [+0.0152, +0.0445] **SEP** | +0.0207 [+0.0067, +0.0359] **SEP** | +0.0207 [+0.0093, +0.0324] **SEP** | +0.0248 [+0.0127, +0.0370] **SEP** | +0.0274 [+0.0136, +0.0431] **SEP** | +0.0219 [+0.0110, +0.0331] **SEP** |

* E1 registered verdict (os floor 0.0009): **FAILED**. Share of RS_XF - os (ADE, s0 point) retained by RETIME_XF: 0.536.
* RETIME_XF (floor 0.0004): LE-E1 (-ha0_ext) **PASS (single training seed; training-seed floor unmeasured)**; LE-E1P (-ha0_ext_pose) PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); LE-NR rollup (headline rows) {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}; ACC-LAT HOLDS (worse at any seed: none).
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): none; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'TAC_traj_lon_correct (TACTICAL)']
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* GATED_RETIME_XF (floor 0.0010): LE-E1 (-ha0_ext) **PASS (single training seed; training-seed floor unmeasured)**; LE-E1P (-ha0_ext_pose) PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); LE-NR rollup (headline rows) {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}; ACC-LAT HOLDS (worse at any seed: none).
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): none; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'TAC_traj_lon_correct (TACTICAL)']
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* geometry RETIME_XF s0: max dist to os path (float64) 7.11e-15 m (after float32 cast 7.11e-07, bound 7.76e-06); non-monotone progress windows 0.0002; extrapolated points 0.0988; fallback-direction windows 0.0260; mean |g-1| 0.369, max 0.578
* geometry RETIME_XF s1: max dist to os path (float64) 7.11e-15 m (after float32 cast 5.81e-07, bound 7.76e-06); non-monotone progress windows 0.0007; extrapolated points 0.0995; fallback-direction windows 0.0260; mean |g-1| 0.368, max 0.582
* geometry GATED_RETIME_XF s0: max dist to os path (float64) 7.11e-15 m (after float32 cast 6.48e-07, bound 7.78e-06); non-monotone progress windows 0.0154; extrapolated points 0.1039; fallback-direction windows 0.0260; mean |g-1| 0.449, max 0.967
* geometry GATED_RETIME_XF s1: max dist to os path (float64) 7.11e-15 m (after float32 cast 6.04e-07, bound 7.77e-06); non-monotone progress windows 0.0149; extrapolated points 0.1047; fallback-direction windows 0.0260; mean |g-1| 0.451, max 0.952
* optimism (-pose, s0): RETIME XF-IS +0.0001 / gated +0.0010, XF-WF +0.0001 / gated +0.0007; IS known sign (<= os): True.
* fold-seed sensitivity (s0, points): RETIME_XF -ha0_ext -0.0270..-0.0264 (10 seeds), GATED_RETIME_XF -ha0_ext -0.0314..-0.0307 (5 seeds).
* in-sample RETIME g per instant (s0): [1.036, 0.6337, 0.4902, 0.4375]
* RETIME_XF fold-mean g (s0) per instant: [1.035, 0.634, 0.49, 0.434]; GATED_RETIME_XF fold-mean g per v0 bin: {'bin0': [0.117, 0.082, 0.117, 0.165], 'bin1': [0.612, 0.295, 0.349, 0.436], 'bin2': [1.024, 0.63, 0.521, 0.467], 'bin3': [1.272, 0.821, 0.749, 0.504], 'bin4': [1.34, 0.86, 0.854, 0.874]}
* RETIME_XF fold-mean g (s1) per instant: [1.037, 0.638, 0.492, 0.433]; GATED_RETIME_XF fold-mean g per v0 bin: {'bin0': [0.119, 0.098, 0.142, 0.187], 'bin1': [0.609, 0.296, 0.355, 0.451], 'bin2': [0.999, 0.621, 0.517, 0.467], 'bin3': [1.289, 0.829, 0.714, 0.461], 'bin4': [1.367, 0.866, 0.864, 0.875]}

### S2 / all -- n 4754 windows / 139 ep

ADE means (s0): os 0.2878, ha0_ext 0.2886, ha0_ext_pose 0.2941, ha0_imu 0.2900, ha0_imu_ax 0.4065, refcv7_50400 0.2784

| read | seed 0 | seed 1 |
|---|---|---|
| **E1 AS REGISTERED: os - ha0_ext (bar of record)** | -0.0008 [-0.0149, +0.0136] | +0.0015 [-0.0131, +0.0163] |
| os - ha0_ext_pose | -0.0064 [-0.0207, +0.0090] | -0.0041 [-0.0189, +0.0110] |
| RS_XF - ha0_ext (LE-E1 form; lever, NOT a bar) | -0.0412 [-0.0511, -0.0317] **SEP** | -0.0401 [-0.0498, -0.0307] **SEP** |
| RS_XF - ha0_ext_pose (LE-E1P form) | -0.0467 [-0.0573, -0.0362] **SEP** | -0.0457 [-0.0561, -0.0354] **SEP** |
| RS_XF - os (ADE) | -0.0403 [-0.0484, -0.0319] **SEP** | -0.0416 [-0.0500, -0.0331] **SEP** |
| RETIME_XF - ha0_ext (LE-E1 form; lever, NOT a bar) | -0.0249 [-0.0355, -0.0145] **SEP** | -0.0236 [-0.0344, -0.0133] **SEP** |
| RETIME_XF - ha0_ext_pose (LE-E1P form) | -0.0304 [-0.0416, -0.0194] **SEP** | -0.0291 [-0.0403, -0.0182] **SEP** |
| RETIME_XF - os (ADE) | -0.0241 [-0.0308, -0.0175] **SEP** | -0.0250 [-0.0320, -0.0184] **SEP** |
| GATED_RETIME_XF - ha0_ext (LE-E1 form; lever, NOT a bar) | -0.0297 [-0.0403, -0.0187] **SEP** | -0.0279 [-0.0388, -0.0171] **SEP** |
| GATED_RETIME_XF - ha0_ext_pose (LE-E1P form) | -0.0352 [-0.0464, -0.0240] **SEP** | -0.0335 [-0.0448, -0.0221] **SEP** |
| GATED_RETIME_XF - os (ADE) | -0.0288 [-0.0359, -0.0220] **SEP** | -0.0294 [-0.0364, -0.0228] **SEP** |
| RETIME_IS - ha0_ext | -0.0253 [-0.0359, -0.0150] **SEP** | -0.0242 [-0.0348, -0.0139] **SEP** |
| RETIME_IS - ha0_ext_pose (LE-E1P form) | -0.0308 [-0.0420, -0.0199] **SEP** | -0.0297 [-0.0410, -0.0188] **SEP** |
| RETIME_IS - os (ADE) | -0.0245 [-0.0311, -0.0180] **SEP** | -0.0256 [-0.0325, -0.0190] **SEP** |
| GATED_RETIME_IS - ha0_ext | -0.0302 [-0.0409, -0.0192] **SEP** | -0.0286 [-0.0393, -0.0178] **SEP** |
| GATED_RETIME_IS - ha0_ext_pose (LE-E1P form) | -0.0357 [-0.0469, -0.0246] **SEP** | -0.0341 [-0.0454, -0.0227] **SEP** |
| GATED_RETIME_IS - os (ADE) | -0.0293 [-0.0364, -0.0225] **SEP** | -0.0300 [-0.0370, -0.0234] **SEP** |
| RETIME_WF - ha0_ext | -0.0253 [-0.0358, -0.0149] **SEP** | -0.0241 [-0.0348, -0.0139] **SEP** |
| RETIME_WF - ha0_ext_pose (LE-E1P form) | -0.0308 [-0.0419, -0.0198] **SEP** | -0.0296 [-0.0409, -0.0188] **SEP** |
| RETIME_WF - os (ADE) | -0.0244 [-0.0311, -0.0179] **SEP** | -0.0255 [-0.0325, -0.0189] **SEP** |
| GATED_RETIME_WF - ha0_ext | -0.0299 [-0.0405, -0.0190] **SEP** | -0.0282 [-0.0390, -0.0175] **SEP** |
| GATED_RETIME_WF - ha0_ext_pose (LE-E1P form) | -0.0354 [-0.0467, -0.0243] **SEP** | -0.0338 [-0.0451, -0.0225] **SEP** |
| GATED_RETIME_WF - os (ADE) | -0.0291 [-0.0361, -0.0223] **SEP** | -0.0297 [-0.0367, -0.0231] **SEP** |

| metric (family) | RETIME_XF - os s0 | s1 | RETIME_XF - refcv7 s0 | s1 | GATED_RETIME_XF - os s0 | s1 | GATED_RETIME_XF - refcv7 s0 | RS_XF (banked Amendment 2) - os s0 |
|---|---|---|---|---|---|---|---|---|
| ade_m (ADE) | -0.0241 [-0.0308, -0.0175] **SEP** | -0.0250 [-0.0320, -0.0184] **SEP** | -0.0147 [-0.0264, -0.0042] **SEP** | -0.0140 [-0.0258, -0.0032] **SEP** | -0.0288 [-0.0359, -0.0220] **SEP** | -0.0294 [-0.0364, -0.0228] **SEP** | -0.0195 [-0.0317, -0.0086] **SEP** | -0.0403 [-0.0484, -0.0319] **SEP** |
| fde_m (ADE) | -0.0607 [-0.0777, -0.0439] **SEP** | -0.0631 [-0.0803, -0.0462] **SEP** | -0.0403 [-0.0679, -0.0150] **SEP** | -0.0385 [-0.0673, -0.0128] **SEP** | -0.0653 [-0.0823, -0.0487] **SEP** | -0.0672 [-0.0842, -0.0512] **SEP** | -0.0449 [-0.0739, -0.0189] **SEP** | -0.0796 [-0.0982, -0.0609] **SEP** |
| LON_speed_mae_mps (LONGITUDINAL) | -0.0348 [-0.0436, -0.0257] **SEP** | -0.0365 [-0.0459, -0.0273] **SEP** | -0.0239 [-0.0371, -0.0121] **SEP** | -0.0238 [-0.0372, -0.0118] **SEP** | -0.0376 [-0.0468, -0.0285] **SEP** | -0.0392 [-0.0487, -0.0303] **SEP** | -0.0267 [-0.0409, -0.0141] **SEP** | -0.0325 [-0.0412, -0.0239] **SEP** |
| LON_along_mae_m (LONGITUDINAL) | -0.0261 [-0.0336, -0.0190] **SEP** | -0.0278 [-0.0355, -0.0203] **SEP** | -0.0144 [-0.0262, -0.0038] **SEP** | -0.0142 [-0.0261, -0.0038] **SEP** | -0.0317 [-0.0395, -0.0245] **SEP** | -0.0331 [-0.0409, -0.0258] **SEP** | -0.0200 [-0.0318, -0.0091] **SEP** | -0.0244 [-0.0317, -0.0174] **SEP** |
| LON_along_abs_final_m (LONGITUDINAL) | -0.0656 [-0.0840, -0.0473] **SEP** | -0.0697 [-0.0888, -0.0514] **SEP** | -0.0415 [-0.0691, -0.0160] **SEP** | -0.0411 [-0.0693, -0.0158] **SEP** | -0.0713 [-0.0893, -0.0533] **SEP** | -0.0749 [-0.0936, -0.0573] **SEP** | -0.0471 [-0.0758, -0.0210] **SEP** | -0.0600 [-0.0780, -0.0424] **SEP** |
| LON_speed_within_0p5 (LONGITUDINAL) | +0.0321 [+0.0233, +0.0406] **SEP** | +0.0333 [+0.0248, +0.0418] **SEP** | +0.0207 [+0.0108, +0.0317] **SEP** | +0.0219 [+0.0118, +0.0324] **SEP** | +0.0305 [+0.0222, +0.0393] **SEP** | +0.0334 [+0.0253, +0.0417] **SEP** | +0.0190 [+0.0087, +0.0301] **SEP** | +0.0295 [+0.0211, +0.0378] **SEP** |
| LON_accel_mae_mps2 (LONGITUDINAL) | -0.0694 [-0.0829, -0.0556] **SEP** | -0.0703 [-0.0836, -0.0571] **SEP** | -0.0670 [-0.0847, -0.0509] **SEP** | -0.0657 [-0.0834, -0.0501] **SEP** | -0.0640 [-0.0765, -0.0519] **SEP** | -0.0662 [-0.0783, -0.0546] **SEP** | -0.0616 [-0.0793, -0.0462] **SEP** | -0.0652 [-0.0782, -0.0521] **SEP** |
| LAT_heading_mae_deg (LATERAL) | -0.0026 [-0.0097, +0.0041] | -0.0011 [-0.0077, +0.0059] | +0.0248 [-0.0151, +0.0646] | +0.0433 [-0.0007, +0.0873] | -0.0002 [-0.0077, +0.0074] | +0.0011 [-0.0058, +0.0088] | +0.0299 [-0.0093, +0.0683] | -0.1437 [-0.1803, -0.1093] **SEP** |
| LAT_curvature_mae_1pm_valid (LATERAL) | -0.0001 [-0.0001, -0.0000] **SEP** | -0.0001 [-0.0001, -0.0000] **SEP** | -0.0000 [-0.0003, +0.0002] | +0.0001 [-0.0001, +0.0003] | -0.0001 [-0.0001, -0.0000] **SEP** | -0.0001 [-0.0001, -0.0000] **SEP** | -0.0000 [-0.0003, +0.0002] | +0.0002 [-0.0001, +0.0006] |
| LAT_yaw_rate_mae_radps_valid (LATERAL) | -0.0003 [-0.0005, -0.0001] **SEP** | -0.0003 [-0.0005, -0.0001] **SEP** | -0.0001 [-0.0009, +0.0007] | +0.0003 [-0.0006, +0.0012] | -0.0003 [-0.0005, -0.0001] **SEP** | -0.0002 [-0.0005, -0.0000] **SEP** | -0.0001 [-0.0009, +0.0007] | -0.0012 [-0.0017, -0.0006] **SEP** |
| LAT_cross_mae_m (LATERAL) | -0.0002 [-0.0013, +0.0008] | +0.0000 [-0.0011, +0.0011] | +0.0010 [-0.0042, +0.0057] | +0.0016 [-0.0036, +0.0066] | -0.0001 [-0.0012, +0.0011] | +0.0002 [-0.0010, +0.0013] | +0.0011 [-0.0041, +0.0059] | -0.0264 [-0.0322, -0.0213] **SEP** |
| LAT_cross_abs_final_m (LATERAL) | -0.0013 [-0.0048, +0.0019] | -0.0007 [-0.0041, +0.0027] | +0.0027 [-0.0086, +0.0136] | +0.0043 [-0.0077, +0.0157] | -0.0012 [-0.0046, +0.0020] | -0.0005 [-0.0039, +0.0028] | +0.0028 [-0.0084, +0.0138] | -0.0369 [-0.0474, -0.0276] **SEP** |
| TAC_traj_lat_correct (TACTICAL) | -0.0002 [-0.0015, +0.0008] | +0.0000 [-0.0015, +0.0015] | -0.0023 [-0.0067, +0.0021] | -0.0046 [-0.0090, -0.0002] **SEP** | +0.0004 [-0.0011, +0.0019] | -0.0002 [-0.0017, +0.0011] | -0.0017 [-0.0063, +0.0029] | +0.0000 [-0.0036, +0.0036] |
| TAC_traj_lon_correct (TACTICAL) | +0.0286 [+0.0168, +0.0412] **SEP** | +0.0290 [+0.0175, +0.0415] **SEP** | +0.0250 [+0.0116, +0.0391] **SEP** | +0.0187 [+0.0063, +0.0322] **SEP** | +0.0229 [+0.0118, +0.0349] **SEP** | +0.0286 [+0.0179, +0.0400] **SEP** | +0.0194 [+0.0065, +0.0338] **SEP** | +0.0290 [+0.0179, +0.0415] **SEP** |

* E1 registered verdict (os floor 0.0023): **FAILED**. Share of RS_XF - os (ADE, s0 point) retained by RETIME_XF: 0.598.
* RETIME_XF (floor 0.0013): LE-E1 (-ha0_ext) **PASS (single training seed; training-seed floor unmeasured)**; LE-E1P (-ha0_ext_pose) PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); LE-NR rollup (headline rows) {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}; ACC-LAT HOLDS (worse at any seed: none).
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): none; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'TAC_traj_lon_correct (TACTICAL)']
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* GATED_RETIME_XF (floor 0.0017): LE-E1 (-ha0_ext) **PASS (single training seed; training-seed floor unmeasured)**; LE-E1P (-ha0_ext_pose) PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); LE-NR rollup (headline rows) {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: LAT_heading_mae_deg'}; ACC-LAT HOLDS (worse at any seed: none).
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): none; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'TAC_traj_lon_correct (TACTICAL)']
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* geometry RETIME_XF s0: max dist to os path (float64) 7.11e-15 m (after float32 cast 6.07e-07, bound 8.69e-06); non-monotone progress windows 0.0002; extrapolated points 0.1081; fallback-direction windows 0.0244; mean |g-1| 0.406, max 0.633
* geometry RETIME_XF s1: max dist to os path (float64) 7.11e-15 m (after float32 cast 5.86e-07, bound 8.68e-06); non-monotone progress windows 0.0006; extrapolated points 0.1092; fallback-direction windows 0.0244; mean |g-1| 0.414, max 0.656
* geometry GATED_RETIME_XF s0: max dist to os path (float64) 7.11e-15 m (after float32 cast 5.83e-07, bound 8.72e-06); non-monotone progress windows 0.0130; extrapolated points 0.1108; fallback-direction windows 0.0244; mean |g-1| 0.510, max 0.944
* geometry GATED_RETIME_XF s1: max dist to os path (float64) 7.11e-15 m (after float32 cast 5.45e-07, bound 8.72e-06); non-monotone progress windows 0.0122; extrapolated points 0.1120; fallback-direction windows 0.0244; mean |g-1| 0.520, max 0.933
* optimism (-pose, s0): RETIME XF-IS +0.0004 / gated +0.0005, XF-WF +0.0004 / gated +0.0002; IS known sign (<= os): True.
* in-sample RETIME g per instant (s0): [0.9998, 0.5837, 0.4319, 0.3862]
* RETIME_XF fold-mean g (s0) per instant: [1.0, 0.583, 0.431, 0.389]; GATED_RETIME_XF fold-mean g per v0 bin: {'bin0': [0.12, 0.093, 0.123, 0.159], 'bin1': [0.628, 0.287, 0.335, 0.417], 'bin2': [0.839, 0.433, 0.329, 0.342], 'bin3': [1.271, 0.821, 0.741, 0.485], 'bin4': [1.376, 0.869, 0.836, 0.819]}
* RETIME_XF fold-mean g (s1) per instant: [0.993, 0.581, 0.418, 0.376]; GATED_RETIME_XF fold-mean g per v0 bin: {'bin0': [0.121, 0.112, 0.147, 0.181], 'bin1': [0.627, 0.29, 0.341, 0.43], 'bin2': [0.804, 0.41, 0.319, 0.328], 'bin3': [1.288, 0.826, 0.701, 0.437], 'bin4': [1.387, 0.872, 0.838, 0.799]}

### S6 / compliant -- n 3251 windows / 138 ep

ADE means (s0): os 2.6275, ha0_ext 3.3810, ha0_ext_pose 3.4098, ha0_imu 3.2977, ha0_imu_ax 3.7217, refcv7_50400 2.5088

| read | seed 0 | seed 1 |
|---|---|---|
| **E1 AS REGISTERED: os - ha0_ext (bar of record)** | -0.7535 [-1.0832, -0.4506] **SEP** | -0.7430 [-1.0699, -0.4363] **SEP** |
| os - ha0_ext_pose | -0.7823 [-1.1159, -0.4748] **SEP** | -0.7717 [-1.1024, -0.4582] **SEP** |
| RS_XF - ha0_ext (LE-E3 form; lever, NOT a bar) | -0.9162 [-1.1794, -0.6723] **SEP** | -0.9161 [-1.1758, -0.6742] **SEP** |
| RS_XF - ha0_ext_pose (S6) | -0.9450 [-1.2100, -0.6967] **SEP** | -0.9449 [-1.2098, -0.6947] **SEP** |
| RS_XF - os (ADE) | -0.1627 [-0.2464, -0.0763] **SEP** | -0.1732 [-0.2554, -0.0869] **SEP** |
| RETIME_XF - ha0_ext (LE-E3 form; lever, NOT a bar) | -0.9825 [-1.2567, -0.7375] **SEP** | -0.9808 [-1.2490, -0.7395] **SEP** |
| RETIME_XF - ha0_ext_pose (S6) | -1.0113 [-1.2913, -0.7596] **SEP** | -1.0095 [-1.2813, -0.7640] **SEP** |
| RETIME_XF - os (ADE) | -0.2290 [-0.3254, -0.1316] **SEP** | -0.2378 [-0.3367, -0.1368] **SEP** |
| GATED_RETIME_XF - ha0_ext (LE-E3 form; lever, NOT a bar) | -0.9932 [-1.2680, -0.7501] **SEP** | -0.9907 [-1.2577, -0.7507] **SEP** |
| GATED_RETIME_XF - ha0_ext_pose (S6) | -1.0219 [-1.3024, -0.7778] **SEP** | -1.0195 [-1.2912, -0.7746] **SEP** |
| GATED_RETIME_XF - os (ADE) | -0.2396 [-0.3416, -0.1369] **SEP** | -0.2478 [-0.3539, -0.1419] **SEP** |
| RETIME_IS - ha0_ext | -0.9840 [-1.2576, -0.7386] **SEP** | -0.9821 [-1.2504, -0.7405] **SEP** |
| RETIME_IS - ha0_ext_pose (S6) | -1.0128 [-1.2916, -0.7608] **SEP** | -1.0108 [-1.2838, -0.7651] **SEP** |
| RETIME_IS - os (ADE) | -0.2305 [-0.3274, -0.1333] **SEP** | -0.2391 [-0.3377, -0.1384] **SEP** |
| GATED_RETIME_IS - ha0_ext | -1.0078 [-1.2815, -0.7681] **SEP** | -1.0079 [-1.2753, -0.7703] **SEP** |
| GATED_RETIME_IS - ha0_ext_pose (S6) | -1.0365 [-1.3153, -0.7939] **SEP** | -1.0366 [-1.3091, -0.7941] **SEP** |
| GATED_RETIME_IS - os (ADE) | -0.2542 [-0.3611, -0.1511] **SEP** | -0.2649 [-0.3775, -0.1558] **SEP** |
| RETIME_WF - ha0_ext | -0.9838 [-1.2573, -0.7383] **SEP** | -0.9818 [-1.2500, -0.7402] **SEP** |
| RETIME_WF - ha0_ext_pose (S6) | -1.0126 [-1.2914, -0.7605] **SEP** | -1.0105 [-1.2835, -0.7647] **SEP** |
| RETIME_WF - os (ADE) | -0.2303 [-0.3270, -0.1330] **SEP** | -0.2388 [-0.3373, -0.1382] **SEP** |
| GATED_RETIME_WF - ha0_ext | -1.0048 [-1.2779, -0.7647] **SEP** | -1.0049 [-1.2729, -0.7671] **SEP** |
| GATED_RETIME_WF - ha0_ext_pose (S6) | -1.0336 [-1.3119, -0.7920] **SEP** | -1.0336 [-1.3063, -0.7904] **SEP** |
| GATED_RETIME_WF - os (ADE) | -0.2513 [-0.3579, -0.1482] **SEP** | -0.2619 [-0.3737, -0.1525] **SEP** |

| metric (family) | RETIME_XF - os s0 | s1 | RETIME_XF - refcv7 s0 | s1 | GATED_RETIME_XF - os s0 | s1 | GATED_RETIME_XF - refcv7 s0 | RS_XF (banked Amendment 2) - os s0 |
|---|---|---|---|---|---|---|---|---|
| ade_m (ADE) | -0.2290 [-0.3254, -0.1316] **SEP** | -0.2378 [-0.3367, -0.1368] **SEP** | -0.1102 [-0.2542, +0.0321] | -0.1006 [-0.2516, +0.0412] | -0.2396 [-0.3416, -0.1369] **SEP** | -0.2478 [-0.3539, -0.1419] **SEP** | -0.1209 [-0.2692, +0.0171] | -0.1627 [-0.2464, -0.0763] **SEP** |
| fde_m (ADE) | -0.6033 [-0.8719, -0.3427] **SEP** | -0.6212 [-0.8868, -0.3504] **SEP** | -0.2045 [-0.5773, +0.1581] | -0.1913 [-0.5612, +0.1791] | -0.6318 [-0.9275, -0.3492] **SEP** | -0.6419 [-0.9450, -0.3570] **SEP** | -0.2330 [-0.6139, +0.1222] | -0.3767 [-0.6047, -0.1394] **SEP** |
| LON_speed_mae_mps (LONGITUDINAL) | -0.0943 [-0.1386, -0.0501] **SEP** | -0.0972 [-0.1423, -0.0533] **SEP** | -0.0314 [-0.0910, +0.0273] | -0.0338 [-0.0934, +0.0259] | -0.1009 [-0.1507, -0.0522] **SEP** | -0.1034 [-0.1534, -0.0543] **SEP** | -0.0380 [-0.0976, +0.0183] | -0.0721 [-0.1074, -0.0350] **SEP** |
| LON_along_mae_m (LONGITUDINAL) | -0.2341 [-0.3314, -0.1383] **SEP** | -0.2440 [-0.3436, -0.1427] **SEP** | -0.1186 [-0.2520, +0.0171] | -0.1154 [-0.2513, +0.0215] | -0.2459 [-0.3500, -0.1446] **SEP** | -0.2548 [-0.3610, -0.1506] **SEP** | -0.1304 [-0.2641, -0.0005] **SEP** | -0.1573 [-0.2468, -0.0661] **SEP** |
| LON_along_abs_final_m (LONGITUDINAL) | -0.6136 [-0.8764, -0.3438] **SEP** | -0.6319 [-0.9029, -0.3569] **SEP** | -0.2406 [-0.5777, +0.1066] | -0.2400 [-0.5886, +0.1267] | -0.6460 [-0.9444, -0.3658] **SEP** | -0.6545 [-0.9519, -0.3685] **SEP** | -0.2730 [-0.6194, +0.0649] | -0.3956 [-0.6363, -0.1433] **SEP** |
| LON_speed_within_0p5 (LONGITUDINAL) | +0.0251 [+0.0132, +0.0376] **SEP** | +0.0268 [+0.0145, +0.0392] **SEP** | -0.0004 [-0.0203, +0.0174] | +0.0018 [-0.0172, +0.0198] | +0.0264 [+0.0136, +0.0392] **SEP** | +0.0301 [+0.0170, +0.0438] **SEP** | +0.0009 [-0.0183, +0.0189] | +0.0201 [+0.0103, +0.0302] **SEP** |
| LON_accel_mae_mps2 (LONGITUDINAL) | -0.0477 [-0.0654, -0.0306] **SEP** | -0.0464 [-0.0638, -0.0292] **SEP** | -0.0095 [-0.0311, +0.0114] | -0.0099 [-0.0313, +0.0108] | -0.0455 [-0.0667, -0.0256] **SEP** | -0.0463 [-0.0675, -0.0265] **SEP** | -0.0073 [-0.0288, +0.0125] | -0.0306 [-0.0450, -0.0157] **SEP** |
| LAT_heading_mae_deg (LATERAL) | -0.0372 [-0.1053, +0.0265] | -0.0466 [-0.1122, +0.0149] | +0.0894 [-0.0981, +0.2637] | +0.1393 [-0.0492, +0.3333] | -0.0275 [-0.0836, +0.0267] | -0.0357 [-0.0877, +0.0138] | +0.0975 [-0.0971, +0.2777] | +0.1773 [+0.0644, +0.3123] **SEP** |
| LAT_curvature_mae_1pm_valid (LATERAL) | -0.0000 [-0.0001, +0.0001] | -0.0000 [-0.0002, +0.0001] | -0.0000 [-0.0004, +0.0004] | +0.0002 [-0.0002, +0.0005] | -0.0001 [-0.0002, +0.0000] | -0.0001 [-0.0002, +0.0000] | -0.0001 [-0.0004, +0.0003] | +0.0007 [+0.0003, +0.0013] **SEP** |
| LAT_yaw_rate_mae_radps_valid (LATERAL) | -0.0005 [-0.0011, +0.0002] | -0.0006 [-0.0012, +0.0000] | -0.0002 [-0.0016, +0.0010] | +0.0001 [-0.0012, +0.0014] | -0.0005 [-0.0010, -0.0000] **SEP** | -0.0006 [-0.0010, -0.0001] **SEP** | -0.0003 [-0.0016, +0.0010] | +0.0019 [+0.0007, +0.0033] **SEP** |
| LAT_cross_mae_m (LATERAL) | -0.0139 [-0.0478, +0.0200] | -0.0168 [-0.0529, +0.0187] | +0.0083 [-0.0543, +0.0692] | +0.0084 [-0.0552, +0.0730] | -0.0136 [-0.0430, +0.0164] | -0.0157 [-0.0469, +0.0161] | +0.0087 [-0.0536, +0.0682] | -0.0235 [-0.0372, -0.0102] **SEP** |
| LAT_cross_abs_final_m (LATERAL) | -0.0398 [-0.1370, +0.0588] | -0.0490 [-0.1501, +0.0534] | +0.0339 [-0.1322, +0.1946] | +0.0280 [-0.1351, +0.1910] | -0.0370 [-0.1199, +0.0504] | -0.0429 [-0.1305, +0.0495] | +0.0367 [-0.1256, +0.1946] | -0.0166 [-0.0421, +0.0107] |
| TAC_traj_lat_correct (TACTICAL) | +0.0015 [-0.0006, +0.0040] | +0.0018 [+0.0000, +0.0040] | +0.0003 [-0.0113, +0.0118] | +0.0018 [-0.0100, +0.0131] | +0.0015 [-0.0006, +0.0037] | +0.0025 [+0.0006, +0.0046] **SEP** | +0.0003 [-0.0118, +0.0121] | -0.0031 [-0.0091, +0.0031] |
| TAC_traj_lon_correct (TACTICAL) | +0.0194 [+0.0009, +0.0399] **SEP** | +0.0169 [-0.0009, +0.0354] | -0.0132 [-0.0471, +0.0174] | -0.0117 [-0.0460, +0.0192] | +0.0206 [-0.0018, +0.0440] | +0.0175 [-0.0022, +0.0388] | -0.0120 [-0.0459, +0.0188] | +0.0015 [-0.0159, +0.0194] |

* E1 registered verdict (os floor 0.0106): **PASS (single training seed; training-seed floor unmeasured)**. Share of RS_XF - os (ADE, s0 point) retained by RETIME_XF: 1.407.
* RETIME_XF (floor 0.0017): LE-E3 (-ha0_ext) **PASS (single training seed; training-seed floor unmeasured)**; LE-E1P (-ha0_ext_pose) PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); LE-NR rollup (headline rows) {'ADE': 'PASS'}; ACC-LAT HOLDS (worse at any seed: none).
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): none; BETTER: none
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)']
* GATED_RETIME_XF (floor 0.0024): LE-E3 (-ha0_ext) **PASS (single training seed; training-seed floor unmeasured)**; LE-E1P (-ha0_ext_pose) PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); LE-NR rollup (headline rows) {'ADE': 'PASS'}; ACC-LAT HOLDS (worse at any seed: none).
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): none; BETTER: none
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)']
* geometry RETIME_XF s0: max dist to os path (float64) 1.42e-14 m (after float32 cast 3.67e-06, bound 2.32e-05); non-monotone progress windows 0.0166; extrapolated points 0.1096; fallback-direction windows 0.0181; mean |g-1| 0.469, max 0.561
* geometry RETIME_XF s1: max dist to os path (float64) 7.12e-15 m (after float32 cast 3.52e-06, bound 2.30e-05); non-monotone progress windows 0.0166; extrapolated points 0.1090; fallback-direction windows 0.0181; mean |g-1| 0.471, max 0.568
* geometry GATED_RETIME_XF s0: max dist to os path (float64) 1.42e-14 m (after float32 cast 3.92e-06, bound 2.32e-05); non-monotone progress windows 0.0163; extrapolated points 0.1082; fallback-direction windows 0.0181; mean |g-1| 0.502, max 0.918
* geometry GATED_RETIME_XF s1: max dist to os path (float64) 1.42e-14 m (after float32 cast 3.67e-06, bound 2.30e-05); non-monotone progress windows 0.0166; extrapolated points 0.1076; fallback-direction windows 0.0181; mean |g-1| 0.507, max 0.926
* optimism (-pose, s0): RETIME XF-IS +0.0015 / gated +0.0146, XF-WF +0.0013 / gated +0.0117; IS known sign (<= os): True.
* fold-seed sensitivity (s0, points): RETIME_XF -ha0_ext -0.9800..-0.9659 (10 seeds), GATED_RETIME_XF -ha0_ext -0.9841..-0.9731 (5 seeds).
* in-sample RETIME g per instant (s0): [0.6436, 0.451, 0.4869, 0.5185, 0.5352, 0.5544]
* RETIME_XF fold-mean g (s0) per instant: [0.643, 0.451, 0.485, 0.518, 0.535, 0.556]; GATED_RETIME_XF fold-mean g per v0 bin: {'bin0': [0.122, 0.227, 0.33, 0.418, 0.448, 0.462], 'bin1': [0.33, 0.451, 0.584, 0.635, 0.668, 0.673], 'bin2': [0.623, 0.453, 0.483, 0.521, 0.537, 0.548], 'bin3': [0.823, 0.465, 0.176, 0.196, 0.226, 0.238], 'bin4': [0.853, 0.876, 0.873, 0.716, 0.617, 0.568]}
* RETIME_XF fold-mean g (s1) per instant: [0.649, 0.445, 0.478, 0.516, 0.532, 0.558]; GATED_RETIME_XF fold-mean g per v0 bin: {'bin0': [0.143, 0.245, 0.306, 0.387, 0.413, 0.422], 'bin1': [0.34, 0.478, 0.602, 0.645, 0.674, 0.689], 'bin2': [0.615, 0.438, 0.476, 0.51, 0.529, 0.547], 'bin3': [0.827, 0.395, 0.173, 0.195, 0.223, 0.237], 'bin4': [0.861, 0.889, 0.837, 0.679, 0.587, 0.524]}

### S6 / all -- n 3668 windows / 139 ep

ADE means (s0): os 2.7333, ha0_ext 3.5479, ha0_ext_pose 3.5764, ha0_imu 3.4480, ha0_imu_ax 3.8578, refcv7_50400 2.5675

| read | seed 0 | seed 1 |
|---|---|---|
| **E1 AS REGISTERED: os - ha0_ext (bar of record)** | -0.8146 [-1.1390, -0.5132] **SEP** | -0.7902 [-1.1101, -0.4921] **SEP** |
| os - ha0_ext_pose | -0.8432 [-1.1730, -0.5332] **SEP** | -0.8187 [-1.1463, -0.5109] **SEP** |
| RS_XF - ha0_ext (LE-E3 form; lever, NOT a bar) | -1.0018 [-1.2678, -0.7533] **SEP** | -0.9895 [-1.2504, -0.7423] **SEP** |
| RS_XF - ha0_ext_pose (S6) | -1.0304 [-1.3075, -0.7709] **SEP** | -1.0181 [-1.2915, -0.7621] **SEP** |
| RS_XF - os (ADE) | -0.1872 [-0.2689, -0.1026] **SEP** | -0.1994 [-0.2823, -0.1141] **SEP** |
| RETIME_XF - ha0_ext (LE-E3 form; lever, NOT a bar) | -1.0717 [-1.3539, -0.8089] **SEP** | -1.0591 [-1.3353, -0.7999] **SEP** |
| RETIME_XF - ha0_ext_pose (S6) | -1.1003 [-1.3924, -0.8258] **SEP** | -1.0877 [-1.3760, -0.8174] **SEP** |
| RETIME_XF - os (ADE) | -0.2571 [-0.3473, -0.1679] **SEP** | -0.2689 [-0.3633, -0.1755] **SEP** |
| GATED_RETIME_XF - ha0_ext (LE-E3 form; lever, NOT a bar) | -1.0813 [-1.3688, -0.8175] **SEP** | -1.0694 [-1.3522, -0.8107] **SEP** |
| GATED_RETIME_XF - ha0_ext_pose (S6) | -1.1099 [-1.4039, -0.8384] **SEP** | -1.0979 [-1.3906, -0.8334] **SEP** |
| GATED_RETIME_XF - os (ADE) | -0.2667 [-0.3644, -0.1702] **SEP** | -0.2792 [-0.3793, -0.1787] **SEP** |
| RETIME_IS - ha0_ext | -1.0747 [-1.3575, -0.8108] **SEP** | -1.0626 [-1.3404, -0.8009] **SEP** |
| RETIME_IS - ha0_ext_pose (S6) | -1.1033 [-1.3953, -0.8290] **SEP** | -1.0912 [-1.3793, -0.8216] **SEP** |
| RETIME_IS - os (ADE) | -0.2601 [-0.3508, -0.1702] **SEP** | -0.2725 [-0.3662, -0.1787] **SEP** |
| GATED_RETIME_IS - ha0_ext | -1.0984 [-1.3834, -0.8347] **SEP** | -1.0870 [-1.3685, -0.8295] **SEP** |
| GATED_RETIME_IS - ha0_ext_pose (S6) | -1.1269 [-1.4198, -0.8535] **SEP** | -1.1156 [-1.4067, -0.8518] **SEP** |
| GATED_RETIME_IS - os (ADE) | -0.2838 [-0.3838, -0.1846] **SEP** | -0.2969 [-0.4011, -0.1943] **SEP** |
| RETIME_WF - ha0_ext | -1.0740 [-1.3564, -0.8102] **SEP** | -1.0619 [-1.3394, -0.7999] **SEP** |
| RETIME_WF - ha0_ext_pose (S6) | -1.1026 [-1.3946, -0.8288] **SEP** | -1.0905 [-1.3788, -0.8213] **SEP** |
| RETIME_WF - os (ADE) | -0.2595 [-0.3497, -0.1695] **SEP** | -0.2718 [-0.3658, -0.1780] **SEP** |
| GATED_RETIME_WF - ha0_ext | -1.0950 [-1.3805, -0.8315] **SEP** | -1.0819 [-1.3642, -0.8241] **SEP** |
| GATED_RETIME_WF - ha0_ext_pose (S6) | -1.1236 [-1.4176, -0.8494] **SEP** | -1.1105 [-1.4017, -0.8451] **SEP** |
| GATED_RETIME_WF - os (ADE) | -0.2804 [-0.3802, -0.1813] **SEP** | -0.2918 [-0.3937, -0.1880] **SEP** |

| metric (family) | RETIME_XF - os s0 | s1 | RETIME_XF - refcv7 s0 | s1 | GATED_RETIME_XF - os s0 | s1 | GATED_RETIME_XF - refcv7 s0 | RS_XF (banked Amendment 2) - os s0 |
|---|---|---|---|---|---|---|---|---|
| ade_m (ADE) | -0.2571 [-0.3473, -0.1679] **SEP** | -0.2689 [-0.3633, -0.1755] **SEP** | -0.0913 [-0.2195, +0.0369] | -0.0795 [-0.2104, +0.0494] | -0.2667 [-0.3644, -0.1702] **SEP** | -0.2792 [-0.3793, -0.1787] **SEP** | -0.1009 [-0.2280, +0.0237] | -0.1872 [-0.2689, -0.1026] **SEP** |
| fde_m (ADE) | -0.6638 [-0.9135, -0.4173] **SEP** | -0.6861 [-0.9449, -0.4324] **SEP** | -0.1875 [-0.5150, +0.1484] | -0.1647 [-0.4949, +0.1727] | -0.6866 [-0.9551, -0.4129] **SEP** | -0.7067 [-0.9827, -0.4317] **SEP** | -0.2103 [-0.5482, +0.1227] | -0.4226 [-0.6448, -0.1990] **SEP** |
| LON_speed_mae_mps (LONGITUDINAL) | -0.1078 [-0.1488, -0.0659] **SEP** | -0.1120 [-0.1539, -0.0694] **SEP** | -0.0299 [-0.0826, +0.0219] | -0.0318 [-0.0860, +0.0204] | -0.1133 [-0.1589, -0.0672] **SEP** | -0.1176 [-0.1635, -0.0713] **SEP** | -0.0354 [-0.0886, +0.0147] | -0.0818 [-0.1165, -0.0474] **SEP** |
| LON_along_mae_m (LONGITUDINAL) | -0.2719 [-0.3598, -0.1763] **SEP** | -0.2871 [-0.3807, -0.1918] **SEP** | -0.1041 [-0.2252, +0.0152] | -0.1023 [-0.2290, +0.0162] | -0.2840 [-0.3822, -0.1808] **SEP** | -0.2995 [-0.4006, -0.1946] **SEP** | -0.1161 [-0.2358, -0.0016] **SEP** | -0.1822 [-0.2677, -0.0922] **SEP** |
| LON_along_abs_final_m (LONGITUDINAL) | -0.6999 [-0.9444, -0.4494] **SEP** | -0.7285 [-0.9781, -0.4728] **SEP** | -0.2286 [-0.5438, +0.0748] | -0.2287 [-0.5666, +0.0848] | -0.7312 [-1.0007, -0.4413] **SEP** | -0.7559 [-1.0306, -0.4667] **SEP** | -0.2599 [-0.5736, +0.0351] | -0.4396 [-0.6699, -0.2044] **SEP** |
| LON_speed_within_0p5 (LONGITUDINAL) | +0.0278 [+0.0169, +0.0387] **SEP** | +0.0302 [+0.0185, +0.0416] **SEP** | -0.0016 [-0.0193, +0.0150] | +0.0024 [-0.0148, +0.0196] | +0.0312 [+0.0188, +0.0439] **SEP** | +0.0324 [+0.0201, +0.0447] **SEP** | +0.0018 [-0.0156, +0.0184] | +0.0209 [+0.0113, +0.0303] **SEP** |
| LON_accel_mae_mps2 (LONGITUDINAL) | -0.0500 [-0.0659, -0.0345] **SEP** | -0.0494 [-0.0655, -0.0338] **SEP** | -0.0107 [-0.0305, +0.0089] | -0.0109 [-0.0304, +0.0086] | -0.0472 [-0.0663, -0.0290] **SEP** | -0.0489 [-0.0675, -0.0311] **SEP** | -0.0079 [-0.0264, +0.0106] | -0.0316 [-0.0453, -0.0173] **SEP** |
| LAT_heading_mae_deg (LATERAL) | -0.0192 [-0.0804, +0.0402] | -0.0252 [-0.0869, +0.0327] | +0.1150 [-0.0474, +0.2885] | +0.1730 [-0.0001, +0.3527] | -0.0064 [-0.0577, +0.0420] | -0.0150 [-0.0632, +0.0292] | +0.1282 [-0.0373, +0.3076] | +0.2030 [+0.0887, +0.3289] **SEP** |
| LAT_curvature_mae_1pm_valid (LATERAL) | -0.0000 [-0.0001, +0.0001] | -0.0000 [-0.0002, +0.0001] | +0.0000 [-0.0003, +0.0004] | +0.0002 [-0.0001, +0.0005] | -0.0001 [-0.0001, +0.0000] | -0.0001 [-0.0002, -0.0000] **SEP** | -0.0000 [-0.0003, +0.0003] | +0.0008 [+0.0004, +0.0013] **SEP** |
| LAT_yaw_rate_mae_radps_valid (LATERAL) | -0.0003 [-0.0009, +0.0003] | -0.0005 [-0.0011, +0.0001] | +0.0000 [-0.0012, +0.0013] | +0.0004 [-0.0008, +0.0017] | -0.0003 [-0.0008, +0.0002] | -0.0005 [-0.0010, -0.0000] **SEP** | +0.0000 [-0.0012, +0.0014] | +0.0022 [+0.0010, +0.0036] **SEP** |
| LAT_cross_mae_m (LATERAL) | -0.0002 [-0.0347, +0.0318] | -0.0014 [-0.0378, +0.0324] | +0.0207 [-0.0381, +0.0806] | +0.0248 [-0.0360, +0.0836] | +0.0022 [-0.0280, +0.0304] | +0.0018 [-0.0306, +0.0324] | +0.0231 [-0.0348, +0.0818] | -0.0233 [-0.0351, -0.0129] **SEP** |
| LAT_cross_abs_final_m (LATERAL) | -0.0006 [-0.0971, +0.0907] | -0.0069 [-0.1097, +0.0877] | +0.0540 [-0.0932, +0.2076] | +0.0593 [-0.0923, +0.2100] | +0.0098 [-0.0756, +0.0926] | +0.0069 [-0.0811, +0.0942] | +0.0644 [-0.0868, +0.2213] | -0.0135 [-0.0332, +0.0053] |
| TAC_traj_lat_correct (TACTICAL) | +0.0019 [-0.0003, +0.0041] | +0.0016 [-0.0003, +0.0036] | -0.0019 [-0.0134, +0.0085] | +0.0005 [-0.0103, +0.0109] | +0.0019 [+0.0000, +0.0038] | +0.0022 [+0.0003, +0.0041] **SEP** | -0.0019 [-0.0136, +0.0087] | +0.0005 [-0.0060, +0.0074] |
| TAC_traj_lon_correct (TACTICAL) | +0.0218 [+0.0046, +0.0390] **SEP** | +0.0215 [+0.0052, +0.0377] **SEP** | -0.0008 [-0.0329, +0.0297] | +0.0027 [-0.0292, +0.0323] | +0.0215 [+0.0003, +0.0422] **SEP** | +0.0188 [-0.0003, +0.0370] | -0.0011 [-0.0330, +0.0287] | -0.0014 [-0.0188, +0.0158] |

* E1 registered verdict (os floor 0.0244): **PASS (single training seed; training-seed floor unmeasured)**. Share of RS_XF - os (ADE, s0 point) retained by RETIME_XF: 1.373.
* RETIME_XF (floor 0.0126): LE-E3 (-ha0_ext) **PASS (single training seed; training-seed floor unmeasured)**; LE-E1P (-ha0_ext_pose) PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); LE-NR rollup (headline rows) {'ADE': 'PASS'}; ACC-LAT HOLDS (worse at any seed: none).
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): none; BETTER: none
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'TAC_traj_lon_correct (TACTICAL)']
* GATED_RETIME_XF (floor 0.0119): LE-E3 (-ha0_ext) **PASS (single training seed; training-seed floor unmeasured)**; LE-E1P (-ha0_ext_pose) PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); LE-NR rollup (headline rows) {'ADE': 'PASS'}; ACC-LAT HOLDS (worse at any seed: none).
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): none; BETTER: none
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)']
* geometry RETIME_XF s0: max dist to os path (float64) 1.43e-14 m (after float32 cast 3.44e-06, bound 2.62e-05); non-monotone progress windows 0.0147; extrapolated points 0.1145; fallback-direction windows 0.0166; mean |g-1| 0.482, max 0.587
* geometry RETIME_XF s1: max dist to os path (float64) 1.42e-14 m (after float32 cast 3.76e-06, bound 2.61e-05); non-monotone progress windows 0.0153; extrapolated points 0.1146; fallback-direction windows 0.0166; mean |g-1| 0.491, max 0.613
* geometry GATED_RETIME_XF s0: max dist to os path (float64) 1.42e-14 m (after float32 cast 3.92e-06, bound 2.62e-05); non-monotone progress windows 0.0147; extrapolated points 0.1134; fallback-direction windows 0.0166; mean |g-1| 0.532, max 0.898
* geometry GATED_RETIME_XF s1: max dist to os path (float64) 1.42e-14 m (after float32 cast 3.86e-06, bound 2.61e-05); non-monotone progress windows 0.0150; extrapolated points 0.1135; fallback-direction windows 0.0166; mean |g-1| 0.540, max 0.928
* optimism (-pose, s0): RETIME XF-IS +0.0030 / gated +0.0170, XF-WF +0.0023 / gated +0.0137; IS known sign (<= os): True.
* in-sample RETIME g per instant (s0): [0.6057, 0.4295, 0.4725, 0.5128, 0.5346, 0.5552]
* RETIME_XF fold-mean g (s0) per instant: [0.604, 0.428, 0.473, 0.51, 0.535, 0.556]; GATED_RETIME_XF fold-mean g per v0 bin: {'bin0': [0.153, 0.232, 0.35, 0.435, 0.462, 0.48], 'bin1': [0.332, 0.462, 0.585, 0.638, 0.663, 0.665], 'bin2': [0.46, 0.375, 0.436, 0.485, 0.515, 0.539], 'bin3': [0.826, 0.441, 0.147, 0.173, 0.21, 0.227], 'bin4': [0.861, 0.85, 0.714, 0.533, 0.432, 0.383]}
* RETIME_XF fold-mean g (s1) per instant: [0.603, 0.412, 0.457, 0.501, 0.525, 0.553]; GATED_RETIME_XF fold-mean g per v0 bin: {'bin0': [0.161, 0.257, 0.329, 0.407, 0.436, 0.455], 'bin1': [0.322, 0.478, 0.6, 0.635, 0.661, 0.673], 'bin2': [0.439, 0.352, 0.417, 0.465, 0.501, 0.534], 'bin3': [0.83, 0.36, 0.144, 0.177, 0.212, 0.231], 'bin4': [0.874, 0.862, 0.706, 0.535, 0.442, 0.405]}

## step30000_unc

### S2 / compliant -- n 4159 windows / 138 ep

ADE means (s0): os 0.2981, ha0_ext 0.2807, ha0_ext_pose 0.2858, ha0_imu 0.2823, ha0_imu_ax 0.3875, refcv7_50400 0.2771

| read | seed 0 | seed 1 |
|---|---|---|
| **E1 AS REGISTERED: os - ha0_ext (bar of record)** | +0.0174 [+0.0014, +0.0349] **SEP** | +0.0160 [+0.0002, +0.0329] **SEP** |
| os - ha0_ext_pose | +0.0123 [-0.0043, +0.0299] | +0.0109 [-0.0056, +0.0283] |
| RS_XF - ha0_ext (LE-E1 form; lever, NOT a bar) | -0.0289 [-0.0382, -0.0199] **SEP** | -0.0286 [-0.0381, -0.0197] **SEP** |
| RS_XF - ha0_ext_pose (LE-E1P form) | -0.0340 [-0.0441, -0.0244] **SEP** | -0.0338 [-0.0439, -0.0243] **SEP** |
| RS_XF - os (ADE) | -0.0463 [-0.0567, -0.0368] **SEP** | -0.0446 [-0.0549, -0.0352] **SEP** |
| RETIME_XF - ha0_ext (LE-E1 form; lever, NOT a bar) | -0.0130 [-0.0233, -0.0027] **SEP** | -0.0122 [-0.0225, -0.0016] **SEP** |
| RETIME_XF - ha0_ext_pose (LE-E1P form) | -0.0181 [-0.0288, -0.0073] **SEP** | -0.0173 [-0.0278, -0.0061] **SEP** |
| RETIME_XF - os (ADE) | -0.0304 [-0.0396, -0.0223] **SEP** | -0.0282 [-0.0370, -0.0198] **SEP** |
| GATED_RETIME_XF - ha0_ext (LE-E1 form; lever, NOT a bar) | -0.0170 [-0.0273, -0.0065] **SEP** | -0.0159 [-0.0261, -0.0051] **SEP** |
| GATED_RETIME_XF - ha0_ext_pose (LE-E1P form) | -0.0221 [-0.0327, -0.0115] **SEP** | -0.0210 [-0.0315, -0.0100] **SEP** |
| GATED_RETIME_XF - os (ADE) | -0.0344 [-0.0442, -0.0261] **SEP** | -0.0319 [-0.0415, -0.0233] **SEP** |
| RETIME_IS - ha0_ext | -0.0135 [-0.0237, -0.0032] **SEP** | -0.0126 [-0.0228, -0.0020] **SEP** |
| RETIME_IS - ha0_ext_pose (LE-E1P form) | -0.0186 [-0.0291, -0.0079] **SEP** | -0.0177 [-0.0280, -0.0065] **SEP** |
| RETIME_IS - os (ADE) | -0.0309 [-0.0402, -0.0227] **SEP** | -0.0286 [-0.0376, -0.0202] **SEP** |
| GATED_RETIME_IS - ha0_ext | -0.0184 [-0.0287, -0.0077] **SEP** | -0.0173 [-0.0276, -0.0065] **SEP** |
| GATED_RETIME_IS - ha0_ext_pose (LE-E1P form) | -0.0235 [-0.0340, -0.0129] **SEP** | -0.0225 [-0.0328, -0.0116] **SEP** |
| GATED_RETIME_IS - os (ADE) | -0.0358 [-0.0457, -0.0274] **SEP** | -0.0333 [-0.0432, -0.0247] **SEP** |
| RETIME_WF - ha0_ext | -0.0134 [-0.0237, -0.0031] **SEP** | -0.0125 [-0.0228, -0.0019] **SEP** |
| RETIME_WF - ha0_ext_pose (LE-E1P form) | -0.0185 [-0.0290, -0.0078] **SEP** | -0.0176 [-0.0279, -0.0064] **SEP** |
| RETIME_WF - os (ADE) | -0.0308 [-0.0402, -0.0227] **SEP** | -0.0285 [-0.0375, -0.0202] **SEP** |
| GATED_RETIME_WF - ha0_ext | -0.0181 [-0.0284, -0.0075] **SEP** | -0.0169 [-0.0273, -0.0060] **SEP** |
| GATED_RETIME_WF - ha0_ext_pose (LE-E1P form) | -0.0232 [-0.0338, -0.0126] **SEP** | -0.0220 [-0.0325, -0.0112] **SEP** |
| GATED_RETIME_WF - os (ADE) | -0.0355 [-0.0454, -0.0271] **SEP** | -0.0329 [-0.0427, -0.0243] **SEP** |

| metric (family) | RETIME_XF - os s0 | s1 | RETIME_XF - refcv7 s0 | s1 | GATED_RETIME_XF - os s0 | s1 | GATED_RETIME_XF - refcv7 s0 | RS_XF (banked Amendment 2) - os s0 |
|---|---|---|---|---|---|---|---|---|
| ade_m (ADE) | -0.0304 [-0.0396, -0.0223] **SEP** | -0.0282 [-0.0370, -0.0198] **SEP** | -0.0094 [-0.0234, +0.0034] | -0.0084 [-0.0220, +0.0044] | -0.0344 [-0.0442, -0.0261] **SEP** | -0.0319 [-0.0415, -0.0233] **SEP** | -0.0134 [-0.0276, -0.0006] **SEP** | -0.0463 [-0.0567, -0.0368] **SEP** |
| fde_m (ADE) | -0.0779 [-0.1007, -0.0574] **SEP** | -0.0716 [-0.0944, -0.0511] **SEP** | -0.0310 [-0.0661, +0.0008] | -0.0284 [-0.0626, +0.0030] | -0.0806 [-0.1030, -0.0600] **SEP** | -0.0740 [-0.0971, -0.0535] **SEP** | -0.0336 [-0.0684, -0.0022] **SEP** | -0.0965 [-0.1211, -0.0742] **SEP** |
| LON_speed_mae_mps (LONGITUDINAL) | -0.0441 [-0.0560, -0.0336] **SEP** | -0.0423 [-0.0543, -0.0313] **SEP** | -0.0185 [-0.0344, -0.0039] **SEP** | -0.0188 [-0.0344, -0.0047] **SEP** | -0.0437 [-0.0555, -0.0330] **SEP** | -0.0412 [-0.0531, -0.0306] **SEP** | -0.0181 [-0.0347, -0.0029] **SEP** | -0.0416 [-0.0536, -0.0312] **SEP** |
| LON_along_mae_m (LONGITUDINAL) | -0.0333 [-0.0435, -0.0243] **SEP** | -0.0310 [-0.0409, -0.0220] **SEP** | -0.0108 [-0.0243, +0.0019] | -0.0104 [-0.0236, +0.0020] | -0.0381 [-0.0488, -0.0289] **SEP** | -0.0356 [-0.0460, -0.0265] **SEP** | -0.0156 [-0.0291, -0.0030] **SEP** | -0.0313 [-0.0414, -0.0226] **SEP** |
| LON_along_abs_final_m (LONGITUDINAL) | -0.0857 [-0.1107, -0.0631] **SEP** | -0.0793 [-0.1038, -0.0570] **SEP** | -0.0364 [-0.0696, -0.0052] **SEP** | -0.0353 [-0.0686, -0.0056] **SEP** | -0.0887 [-0.1133, -0.0665] **SEP** | -0.0820 [-0.1066, -0.0597] **SEP** | -0.0394 [-0.0724, -0.0086] **SEP** | -0.0796 [-0.1044, -0.0575] **SEP** |
| LON_speed_within_0p5 (LONGITUDINAL) | +0.0477 [+0.0362, +0.0596] **SEP** | +0.0439 [+0.0325, +0.0560] **SEP** | +0.0201 [+0.0085, +0.0330] **SEP** | +0.0205 [+0.0087, +0.0329] **SEP** | +0.0426 [+0.0313, +0.0541] **SEP** | +0.0390 [+0.0273, +0.0509] **SEP** | +0.0150 [+0.0022, +0.0284] **SEP** | +0.0454 [+0.0337, +0.0571] **SEP** |
| LON_accel_mae_mps2 (LONGITUDINAL) | -0.0791 [-0.0967, -0.0627] **SEP** | -0.0775 [-0.0947, -0.0612] **SEP** | -0.0575 [-0.0777, -0.0378] **SEP** | -0.0587 [-0.0785, -0.0399] **SEP** | -0.0543 [-0.0685, -0.0413] **SEP** | -0.0524 [-0.0667, -0.0388] **SEP** | -0.0327 [-0.0545, -0.0114] **SEP** | -0.0744 [-0.0919, -0.0585] **SEP** |
| LAT_heading_mae_deg (LATERAL) | +0.0065 [-0.0086, +0.0225] | +0.0060 [-0.0067, +0.0204] | +0.0847 [+0.0245, +0.1604] **SEP** | +0.1015 [+0.0389, +0.1740] **SEP** | +0.0083 [-0.0081, +0.0260] | +0.0072 [-0.0066, +0.0224] | +0.0852 [+0.0251, +0.1595] **SEP** | -0.1467 [-0.1967, -0.1039] **SEP** |
| LAT_curvature_mae_1pm_valid (LATERAL) | -0.0002 [-0.0003, -0.0000] **SEP** | -0.0002 [-0.0003, -0.0000] **SEP** | +0.0003 [-0.0001, +0.0006] | +0.0004 [+0.0001, +0.0008] **SEP** | -0.0002 [-0.0004, -0.0000] **SEP** | -0.0002 [-0.0004, -0.0000] **SEP** | +0.0002 [-0.0001, +0.0006] | +0.0005 [-0.0002, +0.0014] |
| LAT_yaw_rate_mae_radps_valid (LATERAL) | -0.0005 [-0.0009, -0.0001] **SEP** | -0.0004 [-0.0008, -0.0001] **SEP** | +0.0008 [-0.0002, +0.0020] | +0.0012 [+0.0000, +0.0024] **SEP** | -0.0005 [-0.0009, -0.0001] **SEP** | -0.0005 [-0.0009, -0.0001] **SEP** | +0.0008 [-0.0003, +0.0020] | -0.0009 [-0.0019, +0.0001] |
| LAT_cross_mae_m (LATERAL) | +0.0002 [-0.0014, +0.0020] | +0.0005 [-0.0010, +0.0021] | +0.0047 [-0.0010, +0.0107] | +0.0056 [+0.0002, +0.0111] **SEP** | +0.0003 [-0.0014, +0.0021] | +0.0006 [-0.0010, +0.0023] | +0.0048 [-0.0010, +0.0109] | -0.0259 [-0.0321, -0.0203] **SEP** |
| LAT_cross_abs_final_m (LATERAL) | -0.0002 [-0.0047, +0.0043] | +0.0007 [-0.0035, +0.0050] | +0.0104 [-0.0029, +0.0242] | +0.0124 [+0.0001, +0.0254] **SEP** | -0.0005 [-0.0047, +0.0038] | +0.0005 [-0.0035, +0.0045] | +0.0101 [-0.0031, +0.0239] | -0.0353 [-0.0466, -0.0246] **SEP** |
| TAC_traj_lat_correct (TACTICAL) | +0.0010 [-0.0005, +0.0027] | +0.0005 [-0.0012, +0.0023] | -0.0041 [-0.0099, +0.0010] | -0.0053 [-0.0112, -0.0005] **SEP** | +0.0012 [-0.0005, +0.0030] | +0.0002 [-0.0012, +0.0019] | -0.0038 [-0.0097, +0.0015] | +0.0007 [-0.0036, +0.0054] |
| TAC_traj_lon_correct (TACTICAL) | +0.0426 [+0.0266, +0.0605] **SEP** | +0.0361 [+0.0211, +0.0529] **SEP** | +0.0284 [+0.0131, +0.0446] **SEP** | +0.0204 [+0.0070, +0.0363] **SEP** | +0.0392 [+0.0242, +0.0558] **SEP** | +0.0349 [+0.0212, +0.0507] **SEP** | +0.0250 [+0.0098, +0.0413] **SEP** | +0.0385 [+0.0223, +0.0558] **SEP** |

* E1 registered verdict (os floor 0.0014): **FAILED**. Share of RS_XF - os (ADE, s0 point) retained by RETIME_XF: 0.657.
* RETIME_XF (floor 0.0008): LE-E1 (-ha0_ext) **PASS (single training seed; training-seed floor unmeasured)**; LE-E1P (-ha0_ext_pose) PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); LE-NR rollup (headline rows) {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: LAT_heading_mae_deg, LAT_curvature_mae_1pm_valid, LAT_yaw_rate_mae_radps_valid'}; ACC-LAT HOLDS (worse at any seed: none).
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): ['LAT_heading_mae_deg (LATERAL)']; BETTER: ['LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'TAC_traj_lon_correct (TACTICAL)']
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* GATED_RETIME_XF (floor 0.0011): LE-E1 (-ha0_ext) **PASS (single training seed; training-seed floor unmeasured)**; LE-E1P (-ha0_ext_pose) PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); LE-NR rollup (headline rows) {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: LAT_heading_mae_deg, LAT_curvature_mae_1pm_valid'}; ACC-LAT HOLDS (worse at any seed: none).
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): ['LAT_heading_mae_deg (LATERAL)']; BETTER: ['LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'TAC_traj_lon_correct (TACTICAL)']
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* geometry RETIME_XF s0: max dist to os path (float64) 7.11e-15 m (after float32 cast 6.90e-07, bound 7.75e-06); non-monotone progress windows 0.0031; extrapolated points 0.1439; fallback-direction windows 0.0262; mean |g-1| 0.500, max 0.735
* geometry RETIME_XF s1: max dist to os path (float64) 7.12e-15 m (after float32 cast 5.40e-07, bound 7.75e-06); non-monotone progress windows 0.0026; extrapolated points 0.1435; fallback-direction windows 0.0262; mean |g-1| 0.496, max 0.727
* geometry GATED_RETIME_XF s0: max dist to os path (float64) 7.11e-15 m (after float32 cast 6.11e-07, bound 7.78e-06); non-monotone progress windows 0.0164; extrapolated points 0.1444; fallback-direction windows 0.0262; mean |g-1| 0.572, max 1.000
* geometry GATED_RETIME_XF s1: max dist to os path (float64) 7.11e-15 m (after float32 cast 4.84e-07, bound 7.75e-06); non-monotone progress windows 0.0168; extrapolated points 0.1440; fallback-direction windows 0.0262; mean |g-1| 0.565, max 1.000
* optimism (-pose, s0): RETIME XF-IS +0.0005 / gated +0.0014, XF-WF +0.0004 / gated +0.0011; IS known sign (<= os): True.
* in-sample RETIME g per instant (s0): [0.8641, 0.5211, 0.3278, 0.2907]
* RETIME_XF fold-mean g (s0) per instant: [0.864, 0.521, 0.328, 0.291]; GATED_RETIME_XF fold-mean g per v0 bin: {'bin0': [0.085, 0.064, 0.131, 0.197], 'bin1': [0.333, 0.169, 0.255, 0.358], 'bin2': [0.857, 0.516, 0.309, 0.287], 'bin3': [1.172, 0.854, 0.526, 0.022], 'bin4': [1.184, 0.81, 0.866, 0.799]}
* RETIME_XF fold-mean g (s1) per instant: [0.864, 0.522, 0.333, 0.299]; GATED_RETIME_XF fold-mean g per v0 bin: {'bin0': [0.085, 0.053, 0.123, 0.206], 'bin1': [0.321, 0.163, 0.248, 0.355], 'bin2': [0.867, 0.546, 0.358, 0.323], 'bin3': [1.186, 0.855, 0.486, 0.015], 'bin4': [1.186, 0.797, 0.858, 0.779]}

### S2 / all -- n 4754 windows / 139 ep

ADE means (s0): os 0.3128, ha0_ext 0.2886, ha0_ext_pose 0.2941, ha0_imu 0.2900, ha0_imu_ax 0.4065, refcv7_50400 0.2784

| read | seed 0 | seed 1 |
|---|---|---|
| **E1 AS REGISTERED: os - ha0_ext (bar of record)** | +0.0242 [+0.0078, +0.0403] **SEP** | +0.0244 [+0.0082, +0.0408] **SEP** |
| os - ha0_ext_pose | +0.0187 [+0.0021, +0.0351] **SEP** | +0.0189 [+0.0022, +0.0352] **SEP** |
| RS_XF - ha0_ext (LE-E1 form; lever, NOT a bar) | -0.0292 [-0.0381, -0.0209] **SEP** | -0.0282 [-0.0371, -0.0200] **SEP** |
| RS_XF - ha0_ext_pose (LE-E1P form) | -0.0347 [-0.0443, -0.0255] **SEP** | -0.0337 [-0.0433, -0.0247] **SEP** |
| RS_XF - os (ADE) | -0.0534 [-0.0639, -0.0432] **SEP** | -0.0525 [-0.0629, -0.0421] **SEP** |
| RETIME_XF - ha0_ext (LE-E1 form; lever, NOT a bar) | -0.0120 [-0.0222, -0.0019] **SEP** | -0.0103 [-0.0206, +0.0001] |
| RETIME_XF - ha0_ext_pose (LE-E1P form) | -0.0175 [-0.0287, -0.0068] **SEP** | -0.0158 [-0.0271, -0.0047] **SEP** |
| RETIME_XF - os (ADE) | -0.0362 [-0.0446, -0.0278] **SEP** | -0.0347 [-0.0433, -0.0260] **SEP** |
| GATED_RETIME_XF - ha0_ext (LE-E1 form; lever, NOT a bar) | -0.0165 [-0.0268, -0.0061] **SEP** | -0.0148 [-0.0251, -0.0043] **SEP** |
| GATED_RETIME_XF - ha0_ext_pose (LE-E1P form) | -0.0220 [-0.0332, -0.0109] **SEP** | -0.0203 [-0.0316, -0.0091] **SEP** |
| GATED_RETIME_XF - os (ADE) | -0.0407 [-0.0492, -0.0324] **SEP** | -0.0392 [-0.0480, -0.0309] **SEP** |
| RETIME_IS - ha0_ext | -0.0125 [-0.0227, -0.0024] **SEP** | -0.0109 [-0.0212, -0.0007] **SEP** |
| RETIME_IS - ha0_ext_pose (LE-E1P form) | -0.0180 [-0.0291, -0.0074] **SEP** | -0.0165 [-0.0277, -0.0054] **SEP** |
| RETIME_IS - os (ADE) | -0.0367 [-0.0454, -0.0282] **SEP** | -0.0353 [-0.0441, -0.0267] **SEP** |
| GATED_RETIME_IS - ha0_ext | -0.0171 [-0.0274, -0.0066] **SEP** | -0.0154 [-0.0259, -0.0049] **SEP** |
| GATED_RETIME_IS - ha0_ext_pose (LE-E1P form) | -0.0226 [-0.0339, -0.0117] **SEP** | -0.0210 [-0.0323, -0.0098] **SEP** |
| GATED_RETIME_IS - os (ADE) | -0.0413 [-0.0499, -0.0331] **SEP** | -0.0398 [-0.0489, -0.0313] **SEP** |
| RETIME_WF - ha0_ext | -0.0125 [-0.0227, -0.0024] **SEP** | -0.0109 [-0.0212, -0.0007] **SEP** |
| RETIME_WF - ha0_ext_pose (LE-E1P form) | -0.0180 [-0.0291, -0.0073] **SEP** | -0.0164 [-0.0276, -0.0054] **SEP** |
| RETIME_WF - os (ADE) | -0.0367 [-0.0453, -0.0282] **SEP** | -0.0353 [-0.0441, -0.0266] **SEP** |
| GATED_RETIME_WF - ha0_ext | -0.0167 [-0.0269, -0.0062] **SEP** | -0.0151 [-0.0255, -0.0045] **SEP** |
| GATED_RETIME_WF - ha0_ext_pose (LE-E1P form) | -0.0222 [-0.0334, -0.0112] **SEP** | -0.0206 [-0.0319, -0.0094] **SEP** |
| GATED_RETIME_WF - os (ADE) | -0.0409 [-0.0495, -0.0327] **SEP** | -0.0394 [-0.0485, -0.0311] **SEP** |

| metric (family) | RETIME_XF - os s0 | s1 | RETIME_XF - refcv7 s0 | s1 | GATED_RETIME_XF - os s0 | s1 | GATED_RETIME_XF - refcv7 s0 | RS_XF (banked Amendment 2) - os s0 |
|---|---|---|---|---|---|---|---|---|
| ade_m (ADE) | -0.0362 [-0.0446, -0.0278] **SEP** | -0.0347 [-0.0433, -0.0260] **SEP** | -0.0018 [-0.0146, +0.0106] | -0.0007 [-0.0130, +0.0118] | -0.0407 [-0.0492, -0.0324] **SEP** | -0.0392 [-0.0480, -0.0309] **SEP** | -0.0063 [-0.0194, +0.0058] | -0.0534 [-0.0639, -0.0432] **SEP** |
| fde_m (ADE) | -0.0912 [-0.1124, -0.0703] **SEP** | -0.0876 [-0.1089, -0.0671] **SEP** | -0.0161 [-0.0471, +0.0127] | -0.0139 [-0.0433, +0.0152] | -0.0946 [-0.1158, -0.0743] **SEP** | -0.0909 [-0.1125, -0.0707] **SEP** | -0.0195 [-0.0505, +0.0083] | -0.1113 [-0.1354, -0.0876] **SEP** |
| LON_speed_mae_mps (LONGITUDINAL) | -0.0530 [-0.0649, -0.0414] **SEP** | -0.0522 [-0.0640, -0.0410] **SEP** | -0.0149 [-0.0289, -0.0021] **SEP** | -0.0151 [-0.0293, -0.0024] **SEP** | -0.0526 [-0.0642, -0.0414] **SEP** | -0.0514 [-0.0628, -0.0406] **SEP** | -0.0145 [-0.0293, -0.0014] **SEP** | -0.0500 [-0.0618, -0.0390] **SEP** |
| LON_along_mae_m (LONGITUDINAL) | -0.0416 [-0.0515, -0.0316] **SEP** | -0.0402 [-0.0502, -0.0305] **SEP** | -0.0067 [-0.0195, +0.0046] | -0.0065 [-0.0188, +0.0046] | -0.0472 [-0.0572, -0.0376] **SEP** | -0.0457 [-0.0559, -0.0363] **SEP** | -0.0122 [-0.0249, -0.0011] **SEP** | -0.0390 [-0.0490, -0.0291] **SEP** |
| LON_along_abs_final_m (LONGITUDINAL) | -0.1046 [-0.1284, -0.0804] **SEP** | -0.1012 [-0.1253, -0.0782] **SEP** | -0.0299 [-0.0604, -0.0034] **SEP** | -0.0296 [-0.0593, -0.0029] **SEP** | -0.1086 [-0.1326, -0.0858] **SEP** | -0.1051 [-0.1287, -0.0821] **SEP** | -0.0339 [-0.0642, -0.0071] **SEP** | -0.0966 [-0.1208, -0.0738] **SEP** |
| LON_speed_within_0p5 (LONGITUDINAL) | +0.0565 [+0.0451, +0.0681] **SEP** | +0.0532 [+0.0421, +0.0647] **SEP** | +0.0181 [+0.0074, +0.0297] **SEP** | +0.0192 [+0.0087, +0.0303] **SEP** | +0.0504 [+0.0393, +0.0616] **SEP** | +0.0484 [+0.0370, +0.0595] **SEP** | +0.0120 [+0.0013, +0.0241] **SEP** | +0.0530 [+0.0416, +0.0645] **SEP** |
| LON_accel_mae_mps2 (LONGITUDINAL) | -0.0885 [-0.1058, -0.0721] **SEP** | -0.0881 [-0.1051, -0.0717] **SEP** | -0.0540 [-0.0723, -0.0375] **SEP** | -0.0547 [-0.0726, -0.0380] **SEP** | -0.0650 [-0.0778, -0.0525] **SEP** | -0.0634 [-0.0762, -0.0504] **SEP** | -0.0305 [-0.0497, -0.0124] **SEP** | -0.0832 [-0.1001, -0.0674] **SEP** |
| LAT_heading_mae_deg (LATERAL) | +0.0065 [-0.0070, +0.0206] | +0.0053 [-0.0061, +0.0176] | +0.1100 [+0.0498, +0.1776] **SEP** | +0.1260 [+0.0662, +0.1923] **SEP** | +0.0085 [-0.0059, +0.0242] | +0.0073 [-0.0051, +0.0207] | +0.1111 [+0.0503, +0.1799] **SEP** | -0.1521 [-0.2010, -0.1080] **SEP** |
| LAT_curvature_mae_1pm_valid (LATERAL) | -0.0001 [-0.0003, -0.0000] **SEP** | -0.0001 [-0.0003, -0.0000] **SEP** | +0.0003 [-0.0000, +0.0006] | +0.0004 [+0.0001, +0.0008] **SEP** | -0.0002 [-0.0003, -0.0000] **SEP** | -0.0002 [-0.0003, -0.0000] **SEP** | +0.0003 [-0.0000, +0.0006] | +0.0004 [-0.0002, +0.0012] |
| LAT_yaw_rate_mae_radps_valid (LATERAL) | -0.0005 [-0.0008, -0.0001] **SEP** | -0.0004 [-0.0007, -0.0001] **SEP** | +0.0012 [+0.0001, +0.0024] **SEP** | +0.0015 [+0.0004, +0.0027] **SEP** | -0.0004 [-0.0008, -0.0001] **SEP** | -0.0004 [-0.0008, -0.0001] **SEP** | +0.0012 [+0.0002, +0.0024] **SEP** | -0.0011 [-0.0020, -0.0002] **SEP** |
| LAT_cross_mae_m (LATERAL) | +0.0012 [-0.0003, +0.0027] | +0.0013 [-0.0001, +0.0028] | +0.0096 [+0.0035, +0.0159] **SEP** | +0.0110 [+0.0052, +0.0171] **SEP** | +0.0013 [-0.0002, +0.0029] | +0.0014 [+0.0000, +0.0030] **SEP** | +0.0097 [+0.0036, +0.0161] **SEP** | -0.0272 [-0.0336, -0.0215] **SEP** |
| LAT_cross_abs_final_m (LATERAL) | +0.0021 [-0.0020, +0.0063] | +0.0026 [-0.0012, +0.0067] | +0.0210 [+0.0073, +0.0356] **SEP** | +0.0236 [+0.0101, +0.0380] **SEP** | +0.0019 [-0.0020, +0.0059] | +0.0024 [-0.0012, +0.0063] | +0.0208 [+0.0072, +0.0350] **SEP** | -0.0364 [-0.0480, -0.0253] **SEP** |
| TAC_traj_lat_correct (TACTICAL) | +0.0008 [-0.0006, +0.0023] | +0.0004 [-0.0011, +0.0019] | -0.0063 [-0.0116, -0.0013] **SEP** | -0.0072 [-0.0124, -0.0023] **SEP** | +0.0006 [-0.0006, +0.0019] | +0.0002 [-0.0013, +0.0015] | -0.0065 [-0.0118, -0.0013] **SEP** | +0.0000 [-0.0040, +0.0042] |
| TAC_traj_lon_correct (TACTICAL) | +0.0467 [+0.0315, +0.0635] **SEP** | +0.0440 [+0.0292, +0.0605] **SEP** | +0.0250 [+0.0114, +0.0398] **SEP** | +0.0189 [+0.0065, +0.0328] **SEP** | +0.0431 [+0.0297, +0.0589] **SEP** | +0.0406 [+0.0273, +0.0550] **SEP** | +0.0215 [+0.0078, +0.0362] **SEP** | +0.0427 [+0.0280, +0.0585] **SEP** |

* E1 registered verdict (os floor 0.0002): **FAILED**. Share of RS_XF - os (ADE, s0 point) retained by RETIME_XF: 0.678.
* RETIME_XF (floor 0.0017): LE-E1 (-ha0_ext) **FAILED**; LE-E1P (-ha0_ext_pose) PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); LE-NR rollup (headline rows) {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: LAT_heading_mae_deg, LAT_curvature_mae_1pm_valid, LAT_yaw_rate_mae_radps_valid'}; ACC-LAT HOLDS (worse at any seed: none).
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): ['LAT_heading_mae_deg (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)', 'TAC_traj_lat_correct (TACTICAL)']; BETTER: ['LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'TAC_traj_lon_correct (TACTICAL)']
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* GATED_RETIME_XF (floor 0.0017): LE-E1 (-ha0_ext) **PASS (single training seed; training-seed floor unmeasured)**; LE-E1P (-ha0_ext_pose) PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); LE-NR rollup (headline rows) {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: LAT_heading_mae_deg, LAT_curvature_mae_1pm_valid, LAT_yaw_rate_mae_radps_valid'}; ACC-LAT FAILS (worse at any seed: ['LAT_cross_mae_m']).
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): ['LAT_heading_mae_deg (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)', 'TAC_traj_lat_correct (TACTICAL)']; BETTER: ['LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'TAC_traj_lon_correct (TACTICAL)']
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* geometry RETIME_XF s0: max dist to os path (float64) 7.11e-15 m (after float32 cast 5.08e-07, bound 8.69e-06); non-monotone progress windows 0.0023; extrapolated points 0.1483; fallback-direction windows 0.0246; mean |g-1| 0.524, max 0.750
* geometry RETIME_XF s1: max dist to os path (float64) 7.11e-15 m (after float32 cast 5.55e-07, bound 8.68e-06); non-monotone progress windows 0.0025; extrapolated points 0.1483; fallback-direction windows 0.0246; mean |g-1| 0.525, max 0.750
* geometry GATED_RETIME_XF s0: max dist to os path (float64) 7.11e-15 m (after float32 cast 6.11e-07, bound 8.71e-06); non-monotone progress windows 0.0141; extrapolated points 0.1488; fallback-direction windows 0.0246; mean |g-1| 0.615, max 1.000
* geometry GATED_RETIME_XF s1: max dist to os path (float64) 7.11e-15 m (after float32 cast 5.80e-07, bound 8.70e-06); non-monotone progress windows 0.0147; extrapolated points 0.1488; fallback-direction windows 0.0246; mean |g-1| 0.616, max 1.000
* optimism (-pose, s0): RETIME XF-IS +0.0005 / gated +0.0006, XF-WF +0.0005 / gated +0.0002; IS known sign (<= os): True.
* in-sample RETIME g per instant (s0): [0.8425, 0.4845, 0.3011, 0.2779]
* RETIME_XF fold-mean g (s0) per instant: [0.841, 0.484, 0.3, 0.279]; GATED_RETIME_XF fold-mean g per v0 bin: {'bin0': [0.087, 0.07, 0.118, 0.19], 'bin1': [0.358, 0.171, 0.25, 0.347], 'bin2': [0.699, 0.344, 0.223, 0.25], 'bin3': [1.17, 0.849, 0.511, 0.018], 'bin4': [1.174, 0.781, 0.802, 0.684]}
* RETIME_XF fold-mean g (s1) per instant: [0.842, 0.482, 0.3, 0.277]; GATED_RETIME_XF fold-mean g per v0 bin: {'bin0': [0.084, 0.054, 0.104, 0.199], 'bin1': [0.361, 0.168, 0.241, 0.342], 'bin2': [0.706, 0.353, 0.232, 0.258], 'bin3': [1.184, 0.853, 0.478, 0.009], 'bin4': [1.193, 0.792, 0.813, 0.666]}

### S6 / compliant -- n 3251 windows / 138 ep

ADE means (s0): os 2.8070, ha0_ext 3.3810, ha0_ext_pose 3.4098, ha0_imu 3.2977, ha0_imu_ax 3.7217, refcv7_50400 2.5088

| read | seed 0 | seed 1 |
|---|---|---|
| **E1 AS REGISTERED: os - ha0_ext (bar of record)** | -0.5741 [-0.9035, -0.2681] **SEP** | -0.5851 [-0.9183, -0.2759] **SEP** |
| os - ha0_ext_pose | -0.6028 [-0.9347, -0.2914] **SEP** | -0.6138 [-0.9520, -0.2995] **SEP** |
| RS_XF - ha0_ext (LE-E3 form; lever, NOT a bar) | -0.8431 [-1.0855, -0.6161] **SEP** | -0.8444 [-1.0949, -0.6111] **SEP** |
| RS_XF - ha0_ext_pose (S6) | -0.8718 [-1.1246, -0.6370] **SEP** | -0.8732 [-1.1285, -0.6328] **SEP** |
| RS_XF - os (ADE) | -0.2690 [-0.3804, -0.1547] **SEP** | -0.2594 [-0.3723, -0.1447] **SEP** |
| RETIME_XF - ha0_ext (LE-E3 form; lever, NOT a bar) | -0.9052 [-1.1680, -0.6618] **SEP** | -0.8992 [-1.1682, -0.6570] **SEP** |
| RETIME_XF - ha0_ext_pose (S6) | -0.9340 [-1.2058, -0.6833] **SEP** | -0.9280 [-1.1996, -0.6782] **SEP** |
| RETIME_XF - os (ADE) | -0.3312 [-0.4497, -0.2201] **SEP** | -0.3142 [-0.4327, -0.2047] **SEP** |
| GATED_RETIME_XF - ha0_ext (LE-E3 form; lever, NOT a bar) | -0.9279 [-1.1950, -0.6916] **SEP** | -0.9219 [-1.1888, -0.6847] **SEP** |
| GATED_RETIME_XF - ha0_ext_pose (S6) | -0.9567 [-1.2303, -0.7145] **SEP** | -0.9506 [-1.2253, -0.7062] **SEP** |
| GATED_RETIME_XF - os (ADE) | -0.3539 [-0.4873, -0.2348] **SEP** | -0.3368 [-0.4678, -0.2165] **SEP** |
| RETIME_IS - ha0_ext | -0.9065 [-1.1696, -0.6629] **SEP** | -0.9006 [-1.1681, -0.6584] **SEP** |
| RETIME_IS - ha0_ext_pose (S6) | -0.9353 [-1.2057, -0.6849] **SEP** | -0.9293 [-1.2005, -0.6793] **SEP** |
| RETIME_IS - os (ADE) | -0.3325 [-0.4520, -0.2209] **SEP** | -0.3155 [-0.4343, -0.2062] **SEP** |
| GATED_RETIME_IS - ha0_ext | -0.9487 [-1.2126, -0.7155] **SEP** | -0.9429 [-1.2079, -0.7087] **SEP** |
| GATED_RETIME_IS - ha0_ext_pose (S6) | -0.9774 [-1.2505, -0.7374] **SEP** | -0.9716 [-1.2465, -0.7296] **SEP** |
| GATED_RETIME_IS - os (ADE) | -0.3746 [-0.5136, -0.2495] **SEP** | -0.3578 [-0.4924, -0.2336] **SEP** |
| RETIME_WF - ha0_ext | -0.9059 [-1.1688, -0.6620] **SEP** | -0.9001 [-1.1673, -0.6585] **SEP** |
| RETIME_WF - ha0_ext_pose (S6) | -0.9347 [-1.2045, -0.6844] **SEP** | -0.9288 [-1.2000, -0.6787] **SEP** |
| RETIME_WF - os (ADE) | -0.3318 [-0.4517, -0.2203] **SEP** | -0.3150 [-0.4337, -0.2053] **SEP** |
| GATED_RETIME_WF - ha0_ext | -0.9456 [-1.2096, -0.7126] **SEP** | -0.9405 [-1.2050, -0.7054] **SEP** |
| GATED_RETIME_WF - ha0_ext_pose (S6) | -0.9743 [-1.2458, -0.7345] **SEP** | -0.9692 [-1.2436, -0.7272] **SEP** |
| GATED_RETIME_WF - os (ADE) | -0.3715 [-0.5105, -0.2458] **SEP** | -0.3554 [-0.4906, -0.2317] **SEP** |

| metric (family) | RETIME_XF - os s0 | s1 | RETIME_XF - refcv7 s0 | s1 | GATED_RETIME_XF - os s0 | s1 | GATED_RETIME_XF - refcv7 s0 | RS_XF (banked Amendment 2) - os s0 |
|---|---|---|---|---|---|---|---|---|
| ade_m (ADE) | -0.3312 [-0.4497, -0.2201] **SEP** | -0.3142 [-0.4327, -0.2047] **SEP** | -0.0330 [-0.1917, +0.1285] | -0.0191 [-0.1760, +0.1394] | -0.3539 [-0.4873, -0.2348] **SEP** | -0.3368 [-0.4678, -0.2165] **SEP** | -0.0557 [-0.2128, +0.0936] | -0.2690 [-0.3804, -0.1547] **SEP** |
| fde_m (ADE) | -0.8373 [-1.1644, -0.5345] **SEP** | -0.7939 [-1.1118, -0.4960] **SEP** | -0.0379 [-0.4464, +0.3713] | -0.0081 [-0.4080, +0.3953] | -0.9015 [-1.2666, -0.5692] **SEP** | -0.8541 [-1.2161, -0.5259] **SEP** | -0.1021 [-0.5024, +0.2741] | -0.6239 [-0.9177, -0.3093] **SEP** |
| LON_speed_mae_mps (LONGITUDINAL) | -0.1471 [-0.2014, -0.0948] **SEP** | -0.1412 [-0.1938, -0.0903] **SEP** | -0.0224 [-0.0862, +0.0414] | -0.0195 [-0.0823, +0.0446] | -0.1614 [-0.2227, -0.1034] **SEP** | -0.1552 [-0.2158, -0.0997] **SEP** | -0.0367 [-0.0980, +0.0231] | -0.1178 [-0.1659, -0.0692] **SEP** |
| LON_along_mae_m (LONGITUDINAL) | -0.3628 [-0.4829, -0.2499] **SEP** | -0.3521 [-0.4690, -0.2382] **SEP** | -0.0916 [-0.2345, +0.0564] | -0.0802 [-0.2256, +0.0663] | -0.3892 [-0.5227, -0.2661] **SEP** | -0.3760 [-0.5119, -0.2537] **SEP** | -0.1180 [-0.2611, +0.0240] | -0.2740 [-0.3934, -0.1530] **SEP** |
| LON_along_abs_final_m (LONGITUDINAL) | -0.9058 [-1.2267, -0.5974] **SEP** | -0.8799 [-1.1949, -0.5717] **SEP** | -0.2017 [-0.5685, +0.1839] | -0.1757 [-0.5523, +0.2043] | -0.9822 [-1.3553, -0.6419] **SEP** | -0.9430 [-1.3184, -0.6076] **SEP** | -0.2781 [-0.6324, +0.0815] | -0.6578 [-0.9754, -0.3266] **SEP** |
| LON_speed_within_0p5 (LONGITUDINAL) | +0.0422 [+0.0290, +0.0551] **SEP** | +0.0424 [+0.0287, +0.0562] **SEP** | -0.0006 [-0.0212, +0.0184] | -0.0012 [-0.0204, +0.0170] | +0.0445 [+0.0295, +0.0596] **SEP** | +0.0434 [+0.0278, +0.0587] **SEP** | +0.0017 [-0.0181, +0.0212] | +0.0367 [+0.0250, +0.0486] **SEP** |
| LON_accel_mae_mps2 (LONGITUDINAL) | -0.0667 [-0.0867, -0.0480] **SEP** | -0.0639 [-0.0832, -0.0456] **SEP** | -0.0032 [-0.0246, +0.0189] | -0.0044 [-0.0250, +0.0183] | -0.0664 [-0.0910, -0.0444] **SEP** | -0.0630 [-0.0879, -0.0414] **SEP** | -0.0029 [-0.0237, +0.0181] | -0.0463 [-0.0643, -0.0281] **SEP** |
| LAT_heading_mae_deg (LATERAL) | -0.0091 [-0.0911, +0.0672] | -0.0188 [-0.1186, +0.0670] | +0.3747 [+0.1659, +0.5944] **SEP** | +0.4521 [+0.2260, +0.6984] **SEP** | -0.0029 [-0.0732, +0.0646] | -0.0165 [-0.1251, +0.0652] | +0.4206 [+0.1908, +0.6797] **SEP** | +0.2377 [+0.0779, +0.4304] **SEP** |
| LAT_curvature_mae_1pm_valid (LATERAL) | -0.0002 [-0.0005, -0.0000] **SEP** | -0.0002 [-0.0005, -0.0000] **SEP** | +0.0004 [+0.0000, +0.0007] **SEP** | +0.0004 [+0.0000, +0.0009] **SEP** | -0.0003 [-0.0005, -0.0001] **SEP** | -0.0003 [-0.0005, -0.0001] **SEP** | +0.0003 [-0.0001, +0.0006] | +0.0014 [+0.0007, +0.0024] **SEP** |
| LAT_yaw_rate_mae_radps_valid (LATERAL) | -0.0008 [-0.0016, +0.0001] | -0.0005 [-0.0014, +0.0003] | +0.0017 [+0.0003, +0.0031] **SEP** | +0.0018 [+0.0005, +0.0032] **SEP** | -0.0008 [-0.0015, -0.0001] **SEP** | -0.0005 [-0.0012, +0.0002] | +0.0016 [+0.0002, +0.0030] **SEP** | +0.0030 [+0.0011, +0.0053] **SEP** |
| LAT_cross_mae_m (LATERAL) | +0.0121 [-0.0267, +0.0519] | +0.0190 [-0.0201, +0.0576] | +0.0906 [+0.0174, +0.1688] **SEP** | +0.0847 [+0.0147, +0.1573] **SEP** | +0.0127 [-0.0203, +0.0480] | +0.0196 [-0.0138, +0.0545] | +0.0912 [+0.0174, +0.1685] **SEP** | -0.0257 [-0.0420, -0.0094] **SEP** |
| LAT_cross_abs_final_m (LATERAL) | +0.0242 [-0.0829, +0.1389] | +0.0396 [-0.0718, +0.1501] | +0.2479 [+0.0543, +0.4502] **SEP** | +0.2241 [+0.0413, +0.4065] **SEP** | +0.0295 [-0.0634, +0.1329] | +0.0445 [-0.0478, +0.1426] | +0.2532 [+0.0517, +0.4529] **SEP** | -0.0284 [-0.0628, +0.0077] |
| TAC_traj_lat_correct (TACTICAL) | +0.0031 [-0.0009, +0.0075] | +0.0040 [+0.0006, +0.0078] **SEP** | -0.0102 [-0.0269, +0.0065] | -0.0111 [-0.0271, +0.0044] | +0.0018 [-0.0015, +0.0055] | +0.0040 [+0.0009, +0.0078] **SEP** | -0.0114 [-0.0289, +0.0052] | -0.0015 [-0.0104, +0.0075] |
| TAC_traj_lon_correct (TACTICAL) | +0.0237 [+0.0030, +0.0460] **SEP** | +0.0255 [+0.0056, +0.0479] **SEP** | -0.0126 [-0.0464, +0.0207] | -0.0126 [-0.0454, +0.0187] | +0.0252 [+0.0018, +0.0505] **SEP** | +0.0292 [+0.0058, +0.0533] **SEP** | -0.0111 [-0.0441, +0.0204] | +0.0074 [-0.0123, +0.0295] |

* E1 registered verdict (os floor 0.0110): **PASS (single training seed; training-seed floor unmeasured)**. Share of RS_XF - os (ADE, s0 point) retained by RETIME_XF: 1.231.
* RETIME_XF (floor 0.0060): LE-E3 (-ha0_ext) **PASS (single training seed; training-seed floor unmeasured)**; LE-E1P (-ha0_ext_pose) PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); LE-NR rollup (headline rows) {'ADE': 'PASS'}; ACC-LAT HOLDS (worse at any seed: none).
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): ['LAT_heading_mae_deg (LATERAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)']; BETTER: none
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* GATED_RETIME_XF (floor 0.0061): LE-E3 (-ha0_ext) **PASS (single training seed; training-seed floor unmeasured)**; LE-E1P (-ha0_ext_pose) PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); LE-NR rollup (headline rows) {'ADE': 'PASS'}; ACC-LAT HOLDS (worse at any seed: none).
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): ['LAT_heading_mae_deg (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)']; BETTER: none
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* geometry RETIME_XF s0: max dist to os path (float64) 1.42e-14 m (after float32 cast 3.72e-06, bound 2.29e-05); non-monotone progress windows 0.0209; extrapolated points 0.1290; fallback-direction windows 0.0185; mean |g-1| 0.550, max 0.715
* geometry RETIME_XF s1: max dist to os path (float64) 1.43e-14 m (after float32 cast 3.83e-06, bound 2.29e-05); non-monotone progress windows 0.0206; extrapolated points 0.1284; fallback-direction windows 0.0185; mean |g-1| 0.546, max 0.709
* geometry GATED_RETIME_XF s0: max dist to os path (float64) 1.42e-14 m (after float32 cast 3.54e-06, bound 2.34e-05); non-monotone progress windows 0.0200; extrapolated points 0.1271; fallback-direction windows 0.0185; mean |g-1| 0.596, max 1.000
* geometry GATED_RETIME_XF s1: max dist to os path (float64) 1.42e-14 m (after float32 cast 3.88e-06, bound 2.34e-05); non-monotone progress windows 0.0194; extrapolated points 0.1273; fallback-direction windows 0.0185; mean |g-1| 0.590, max 1.000
* optimism (-pose, s0): RETIME XF-IS +0.0013 / gated +0.0207, XF-WF +0.0007 / gated +0.0176; IS known sign (<= os): True.
* in-sample RETIME g per instant (s0): [0.5368, 0.3062, 0.3807, 0.4549, 0.4947, 0.5239]
* RETIME_XF fold-mean g (s0) per instant: [0.538, 0.306, 0.381, 0.455, 0.495, 0.524]; GATED_RETIME_XF fold-mean g per v0 bin: {'bin0': [0.088, 0.284, 0.402, 0.472, 0.508, 0.536], 'bin1': [0.173, 0.354, 0.521, 0.617, 0.66, 0.67], 'bin2': [0.525, 0.291, 0.356, 0.429, 0.478, 0.514], 'bin3': [0.855, 0.011, 0.02, 0.092, 0.137, 0.169], 'bin4': [0.799, 0.824, 0.366, 0.262, 0.233, 0.223]}
* RETIME_XF fold-mean g (s1) per instant: [0.539, 0.311, 0.383, 0.461, 0.503, 0.53]; GATED_RETIME_XF fold-mean g per v0 bin: {'bin0': [0.072, 0.285, 0.401, 0.457, 0.495, 0.532], 'bin1': [0.173, 0.353, 0.523, 0.621, 0.657, 0.675], 'bin2': [0.553, 0.326, 0.385, 0.45, 0.501, 0.526], 'bin3': [0.86, 0.003, 0.005, 0.066, 0.123, 0.156], 'bin4': [0.78, 0.81, 0.347, 0.247, 0.224, 0.224]}

### S6 / all -- n 3668 windows / 139 ep

ADE means (s0): os 2.9189, ha0_ext 3.5479, ha0_ext_pose 3.5764, ha0_imu 3.4480, ha0_imu_ax 3.8578, refcv7_50400 2.5675

| read | seed 0 | seed 1 |
|---|---|---|
| **E1 AS REGISTERED: os - ha0_ext (bar of record)** | -0.6289 [-0.9539, -0.3086] **SEP** | -0.6258 [-0.9564, -0.2987] **SEP** |
| os - ha0_ext_pose | -0.6575 [-0.9943, -0.3332] **SEP** | -0.6544 [-0.9962, -0.3200] **SEP** |
| RS_XF - ha0_ext (LE-E3 form; lever, NOT a bar) | -0.9202 [-1.1782, -0.6834] **SEP** | -0.9178 [-1.1749, -0.6751] **SEP** |
| RS_XF - ha0_ext_pose (S6) | -0.9488 [-1.2063, -0.7037] **SEP** | -0.9464 [-1.2099, -0.6971] **SEP** |
| RS_XF - os (ADE) | -0.2913 [-0.3990, -0.1864] **SEP** | -0.2920 [-0.4033, -0.1848] **SEP** |
| RETIME_XF - ha0_ext (LE-E3 form; lever, NOT a bar) | -0.9780 [-1.2464, -0.7207] **SEP** | -0.9742 [-1.2514, -0.7164] **SEP** |
| RETIME_XF - ha0_ext_pose (S6) | -1.0066 [-1.2872, -0.7443] **SEP** | -1.0028 [-1.2885, -0.7371] **SEP** |
| RETIME_XF - os (ADE) | -0.3491 [-0.4539, -0.2481] **SEP** | -0.3484 [-0.4535, -0.2437] **SEP** |
| GATED_RETIME_XF - ha0_ext (LE-E3 form; lever, NOT a bar) | -1.0071 [-1.2818, -0.7462] **SEP** | -1.0038 [-1.2819, -0.7458] **SEP** |
| GATED_RETIME_XF - ha0_ext_pose (S6) | -1.0357 [-1.3190, -0.7696] **SEP** | -1.0324 [-1.3151, -0.7664] **SEP** |
| GATED_RETIME_XF - os (ADE) | -0.3781 [-0.5023, -0.2632] **SEP** | -0.3780 [-0.5065, -0.2593] **SEP** |
| RETIME_IS - ha0_ext | -0.9815 [-1.2495, -0.7251] **SEP** | -0.9768 [-1.2541, -0.7194] **SEP** |
| RETIME_IS - ha0_ext_pose (S6) | -1.0101 [-1.2903, -0.7467] **SEP** | -1.0053 [-1.2917, -0.7390] **SEP** |
| RETIME_IS - os (ADE) | -0.3525 [-0.4581, -0.2503] **SEP** | -0.3509 [-0.4567, -0.2459] **SEP** |
| GATED_RETIME_IS - ha0_ext | -1.0218 [-1.2947, -0.7651] **SEP** | -1.0175 [-1.2957, -0.7617] **SEP** |
| GATED_RETIME_IS - ha0_ext_pose (S6) | -1.0504 [-1.3319, -0.7880] **SEP** | -1.0460 [-1.3256, -0.7803] **SEP** |
| GATED_RETIME_IS - os (ADE) | -0.3928 [-0.5218, -0.2771] **SEP** | -0.3916 [-0.5228, -0.2702] **SEP** |
| RETIME_WF - ha0_ext | -0.9807 [-1.2490, -0.7247] **SEP** | -0.9761 [-1.2536, -0.7189] **SEP** |
| RETIME_WF - ha0_ext_pose (S6) | -1.0092 [-1.2895, -0.7467] **SEP** | -1.0047 [-1.2911, -0.7380] **SEP** |
| RETIME_WF - os (ADE) | -0.3517 [-0.4573, -0.2500] **SEP** | -0.3503 [-0.4565, -0.2454] **SEP** |
| GATED_RETIME_WF - ha0_ext | -1.0179 [-1.2911, -0.7622] **SEP** | -1.0135 [-1.2916, -0.7569] **SEP** |
| GATED_RETIME_WF - ha0_ext_pose (S6) | -1.0465 [-1.3279, -0.7837] **SEP** | -1.0421 [-1.3213, -0.7762] **SEP** |
| GATED_RETIME_WF - os (ADE) | -0.3890 [-0.5159, -0.2737] **SEP** | -0.3877 [-0.5173, -0.2659] **SEP** |

| metric (family) | RETIME_XF - os s0 | s1 | RETIME_XF - refcv7 s0 | s1 | GATED_RETIME_XF - os s0 | s1 | GATED_RETIME_XF - refcv7 s0 | RS_XF (banked Amendment 2) - os s0 |
|---|---|---|---|---|---|---|---|---|
| ade_m (ADE) | -0.3491 [-0.4539, -0.2481] **SEP** | -0.3484 [-0.4535, -0.2437] **SEP** | +0.0024 [-0.1468, +0.1482] | +0.0054 [-0.1347, +0.1523] | -0.3781 [-0.5023, -0.2632] **SEP** | -0.3780 [-0.5065, -0.2593] **SEP** | -0.0267 [-0.1663, +0.1080] | -0.2913 [-0.3990, -0.1864] **SEP** |
| fde_m (ADE) | -0.8669 [-1.1457, -0.5944] **SEP** | -0.8706 [-1.1575, -0.5981] **SEP** | +0.0285 [-0.3570, +0.4122] | +0.0280 [-0.3331, +0.4052] | -0.9445 [-1.3025, -0.6333] **SEP** | -0.9479 [-1.3003, -0.6274] **SEP** | -0.0492 [-0.4036, +0.2998] | -0.6594 [-0.9429, -0.3797] **SEP** |
| LON_speed_mae_mps (LONGITUDINAL) | -0.1576 [-0.2057, -0.1110] **SEP** | -0.1582 [-0.2056, -0.1121] **SEP** | -0.0204 [-0.0786, +0.0351] | -0.0191 [-0.0763, +0.0375] | -0.1734 [-0.2342, -0.1207] **SEP** | -0.1743 [-0.2343, -0.1206] **SEP** | -0.0362 [-0.0900, +0.0156] | -0.1237 [-0.1687, -0.0799] **SEP** |
| LON_along_mae_m (LONGITUDINAL) | -0.3959 [-0.5038, -0.2934] **SEP** | -0.3983 [-0.5036, -0.2962] **SEP** | -0.0764 [-0.2112, +0.0548] | -0.0695 [-0.2053, +0.0629] | -0.4323 [-0.5599, -0.3132] **SEP** | -0.4333 [-0.5638, -0.3119] **SEP** | -0.1128 [-0.2383, +0.0080] | -0.3006 [-0.4153, -0.1909] **SEP** |
| LON_along_abs_final_m (LONGITUDINAL) | -0.9703 [-1.2528, -0.6978] **SEP** | -0.9849 [-1.2559, -0.7161] **SEP** | -0.1751 [-0.5329, +0.1667] | -0.1627 [-0.5188, +0.1770] | -1.0727 [-1.4289, -0.7506] **SEP** | -1.0800 [-1.4336, -0.7484] **SEP** | -0.2775 [-0.6028, +0.0261] | -0.7006 [-0.9983, -0.4042] **SEP** |
| LON_speed_within_0p5 (LONGITUDINAL) | +0.0426 [+0.0302, +0.0548] **SEP** | +0.0445 [+0.0323, +0.0563] **SEP** | -0.0043 [-0.0220, +0.0139] | -0.0022 [-0.0193, +0.0151] | +0.0477 [+0.0335, +0.0624] **SEP** | +0.0483 [+0.0338, +0.0637] **SEP** | +0.0008 [-0.0166, +0.0186] | +0.0374 [+0.0264, +0.0479] **SEP** |
| LON_accel_mae_mps2 (LONGITUDINAL) | -0.0678 [-0.0861, -0.0513] **SEP** | -0.0677 [-0.0859, -0.0513] **SEP** | -0.0041 [-0.0247, +0.0167] | -0.0061 [-0.0264, +0.0132] | -0.0666 [-0.0900, -0.0463] **SEP** | -0.0664 [-0.0895, -0.0467] **SEP** | -0.0028 [-0.0219, +0.0163] | -0.0468 [-0.0643, -0.0303] **SEP** |
| LAT_heading_mae_deg (LATERAL) | +0.0092 [-0.0605, +0.0771] | +0.0147 [-0.0561, +0.0799] | +0.4607 [+0.2578, +0.6712] **SEP** | +0.5021 [+0.2856, +0.7312] **SEP** | +0.0161 [-0.0435, +0.0756] | +0.0261 [-0.0321, +0.0813] | +0.5025 [+0.2804, +0.7233] **SEP** | +0.2111 [+0.0629, +0.3818] **SEP** |
| LAT_curvature_mae_1pm_valid (LATERAL) | -0.0002 [-0.0005, -0.0001] **SEP** | -0.0002 [-0.0005, -0.0000] **SEP** | +0.0004 [+0.0001, +0.0007] **SEP** | +0.0005 [+0.0001, +0.0008] **SEP** | -0.0003 [-0.0005, -0.0001] **SEP** | -0.0002 [-0.0005, -0.0001] **SEP** | +0.0003 [+0.0000, +0.0007] **SEP** | +0.0014 [+0.0007, +0.0023] **SEP** |
| LAT_yaw_rate_mae_radps_valid (LATERAL) | -0.0007 [-0.0014, +0.0000] | -0.0004 [-0.0012, +0.0003] | +0.0023 [+0.0010, +0.0037] **SEP** | +0.0022 [+0.0009, +0.0035] **SEP** | -0.0006 [-0.0013, -0.0001] **SEP** | -0.0004 [-0.0010, +0.0002] | +0.0022 [+0.0009, +0.0037] **SEP** | +0.0029 [+0.0011, +0.0051] **SEP** |
| LAT_cross_mae_m (LATERAL) | +0.0278 [-0.0066, +0.0607] | +0.0329 [-0.0033, +0.0672] | +0.1225 [+0.0493, +0.1999] **SEP** | +0.1119 [+0.0437, +0.1863] **SEP** | +0.0314 [+0.0009, +0.0609] **SEP** | +0.0365 [+0.0055, +0.0656] **SEP** | +0.1262 [+0.0531, +0.2038] **SEP** | -0.0261 [-0.0411, -0.0121] **SEP** |
| LAT_cross_abs_final_m (LATERAL) | +0.0685 [-0.0293, +0.1650] | +0.0797 [-0.0227, +0.1763] | +0.3160 [+0.1335, +0.5080] **SEP** | +0.2796 [+0.1054, +0.4601] **SEP** | +0.0842 [-0.0027, +0.1682] | +0.0958 [+0.0074, +0.1783] **SEP** | +0.3318 [+0.1457, +0.5314] **SEP** | -0.0240 [-0.0537, +0.0046] |
| TAC_traj_lat_correct (TACTICAL) | +0.0019 [-0.0014, +0.0055] | +0.0035 [+0.0005, +0.0071] **SEP** | -0.0155 [-0.0321, +0.0006] | -0.0139 [-0.0292, +0.0019] | +0.0019 [-0.0008, +0.0052] | +0.0033 [+0.0005, +0.0066] **SEP** | -0.0155 [-0.0323, +0.0008] | -0.0003 [-0.0085, +0.0071] |
| TAC_traj_lon_correct (TACTICAL) | +0.0224 [+0.0038, +0.0424] **SEP** | +0.0286 [+0.0090, +0.0488] **SEP** | -0.0046 [-0.0366, +0.0246] | +0.0003 [-0.0316, +0.0292] | +0.0275 [+0.0052, +0.0523] **SEP** | +0.0327 [+0.0106, +0.0559] **SEP** | +0.0005 [-0.0321, +0.0313] | +0.0065 [-0.0123, +0.0268] |

* E1 registered verdict (os floor 0.0031): **PASS (single training seed; training-seed floor unmeasured)**. Share of RS_XF - os (ADE, s0 point) retained by RETIME_XF: 1.198.
* RETIME_XF (floor 0.0038): LE-E3 (-ha0_ext) **PASS (single training seed; training-seed floor unmeasured)**; LE-E1P (-ha0_ext_pose) PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); LE-NR rollup (headline rows) {'ADE': 'PASS'}; ACC-LAT HOLDS (worse at any seed: none).
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): ['LAT_heading_mae_deg (LATERAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)']; BETTER: none
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* GATED_RETIME_XF (floor 0.0033): LE-E3 (-ha0_ext) **PASS (single training seed; training-seed floor unmeasured)**; LE-E1P (-ha0_ext_pose) PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); LE-NR rollup (headline rows) {'ADE': 'PASS'}; ACC-LAT FAILS (worse at any seed: ['LAT_cross_mae_m', 'LAT_cross_abs_final_m']).
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): ['LAT_heading_mae_deg (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)']; BETTER: none
  * vs refcv8 os, separated WORSE at both seeds: **['LAT_cross_mae_m (LATERAL)']**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* geometry RETIME_XF s0: max dist to os path (float64) 1.42e-14 m (after float32 cast 3.15e-06, bound 2.62e-05); non-monotone progress windows 0.0191; extrapolated points 0.1329; fallback-direction windows 0.0169; mean |g-1| 0.549, max 0.715
* geometry RETIME_XF s1: max dist to os path (float64) 1.43e-14 m (after float32 cast 3.64e-06, bound 2.62e-05); non-monotone progress windows 0.0185; extrapolated points 0.1325; fallback-direction windows 0.0169; mean |g-1| 0.551, max 0.710
* geometry GATED_RETIME_XF s0: max dist to os path (float64) 1.42e-14 m (after float32 cast 3.54e-06, bound 2.62e-05); non-monotone progress windows 0.0183; extrapolated points 0.1311; fallback-direction windows 0.0169; mean |g-1| 0.598, max 1.000
* geometry GATED_RETIME_XF s1: max dist to os path (float64) 1.43e-14 m (after float32 cast 3.58e-06, bound 2.62e-05); non-monotone progress windows 0.0177; extrapolated points 0.1317; fallback-direction windows 0.0169; mean |g-1| 0.601, max 1.000
* optimism (-pose, s0): RETIME XF-IS +0.0035 / gated +0.0147, XF-WF +0.0026 / gated +0.0108; IS known sign (<= os): True.
* in-sample RETIME g per instant (s0): [0.5108, 0.3118, 0.3871, 0.4599, 0.5046, 0.5346]
* RETIME_XF fold-mean g (s0) per instant: [0.511, 0.311, 0.387, 0.459, 0.504, 0.536]; GATED_RETIME_XF fold-mean g per v0 bin: {'bin0': [0.111, 0.301, 0.436, 0.497, 0.536, 0.566], 'bin1': [0.176, 0.368, 0.523, 0.617, 0.656, 0.664], 'bin2': [0.404, 0.279, 0.351, 0.428, 0.482, 0.523], 'bin3': [0.853, 0.003, 0.012, 0.1, 0.144, 0.177], 'bin4': [0.781, 0.772, 0.358, 0.229, 0.219, 0.223]}
* RETIME_XF fold-mean g (s1) per instant: [0.509, 0.311, 0.383, 0.457, 0.502, 0.53]; GATED_RETIME_XF fold-mean g per v0 bin: {'bin0': [0.091, 0.301, 0.439, 0.489, 0.529, 0.564], 'bin1': [0.18, 0.367, 0.53, 0.617, 0.649, 0.662], 'bin2': [0.409, 0.294, 0.356, 0.427, 0.48, 0.511], 'bin3': [0.865, 0.0, 0.006, 0.075, 0.132, 0.17], 'bin4': [0.792, 0.74, 0.313, 0.2, 0.206, 0.227]}

