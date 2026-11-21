# A0K0_HEADROOM -- admissible ego-state headroom for refcv8 (zero training, banked panels)

* **Status:** EARLY READ of banked step 30,000. E1 AS REGISTERED (os - ha0_ext) is the bar of record and is reported first. The admissible-reference read (os - ha0_ext_pose / ha0_imu) is a BESIDE-ROW pre-registered 2026-10-19 before any final-checkpoint data, MOTIVATED AFTER seeing the 30k read. RS arms are cross-fitted calibrations fit on EVAL episodes: lever estimates, never a bar pass (a deployable version fits g on TRAIN).
* **Tier:** T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
* **Estimator:** paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, models FIXED: 'another draw of EPISODES' only; NOT another training run (single training seed), NOT another inference run (seed 1 is the replicate beside seed 0), NOT another fold assignment (fold-seed sensitivity reported separately)
* Pre-registration `raw/a0k0_headroom/PREREG.md` (sha256 `623032b2b53a`, Amendments 1-2 written before any number); script sha256 `a70fb24cb3e5`.
* Premise correction: E:/refcv8_eval_kit/ckpt/config.json argv carries --ego-history and --residual-prior ha0_ext_pose (verified).
* Controls **ALL_PASS = True** (a (g1 == banked os, every row exact), b (g0 == ha0_ext_pose / w0 == ha0_imu bit-exact, 0-cell), decisions (ws, ep_poses[ws] == pose_last, v0), V1 (ha0_ext rebuilt <= 1e-3 m), V2 (imu a0=0 limit <= 1e-4 m), mask sha, K). 2026-09-26 reproduction (S2 all windows, ha0_ext_pose - ha0_ext): +0.0055 [+0.0036, +0.0076] **SEP** vs banked +0.0055 [+0.0036, +0.0076] **SEP** -> exact = **True**.
* STRATEGIC family: NOT APPLICABLE, n = 0 (the run argv carries --no-strategic: the layer is off).

## FROZEN gains (Amendment 3) -- fit on EMA-30k, ALL windows, both seeds pooled

`FROZEN_GAINS_ema30k.json` sha256 `4e0fc55d3e23`; fit set: ALL windows of the surface (all 139 eval episodes), inference seeds 0 and 1 POOLED (each seed's os against the same prior and GT).

* **S2** instants [0.5, 1.0, 1.5, 2.0] s, n 4754 windows / 139 ep: RS g [along, cross] = [[0.997, 0.226], [0.582, 0.382], [0.428, 0.55], [0.41, 0.681]]; in-sample pooled ADE os 0.2889 / prior 0.2941 / RS 0.2472 / GATED 0.2422
  * GATED v0<2 (n 465): [[0.116, 0.131], [0.082, 0.36], [0.122, 0.393], [0.166, 0.384]]
  * GATED v0 2-8 (n 1306): [[0.636, 0.156], [0.292, 0.369], [0.366, 0.568], [0.492, 0.716]]
  * GATED v0 8-15 (n 1792): [[0.831, 0.195], [0.4, 0.366], [0.305, 0.532], [0.332, 0.666]]
  * GATED v0 15-25 (n 856): [[1.269, 0.339], [0.819, 0.415], [0.728, 0.563], [0.483, 0.683]]
  * GATED v0>=25 (n 335): [[1.378, 0.626], [0.868, 0.458], [0.831, 0.494], [0.813, 0.561]]
* **S6** instants [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s, n 3668 windows / 139 ep: RS g [along, cross] = [[0.598, 0.365], [0.443, 0.67], [0.53, 0.849], [0.587, 0.908], [0.635, 0.936], [0.65, 0.952]]; in-sample pooled ADE os 2.7455 / prior 3.5764 / RS 2.5493 / GATED 2.4899
  * GATED v0<2 (n 330): [[0.152, 0.08], [0.263, 0.409], [0.346, 0.434], [0.439, 0.501], [0.46, 0.421], [0.493, 0.34]]
  * GATED v0 2-8 (n 1017): [[0.334, 0.347], [0.527, 0.706], [0.715, 0.868], [0.807, 0.904], [0.832, 0.907], [0.83, 0.922]]
  * GATED v0 8-15 (n 1397): [[0.423, 0.354], [0.348, 0.658], [0.424, 0.839], [0.481, 0.917], [0.512, 0.958], [0.539, 0.981]]
  * GATED v0 15-25 (n 679): [[0.819, 0.395], [0.406, 0.653], [0.159, 0.845], [0.175, 0.935], [0.207, 0.963], [0.227, 0.992]]
  * GATED v0>=25 (n 245): [[0.867, 0.446], [0.853, 0.537], [0.725, 0.626], [0.53, 0.663], [0.423, 0.687], [0.342, 0.704]]

## FORWARD step30000_unc -- path check on a different checkpoint of the SAME run (not an independent test, not a bar) (controls True)

| set / arm | FZ-E1 (-ha0_ext, S2) | FZ-E1P (-ha0_ext_pose, S2) | FZ-E3 (-ha0_ext, S6) | FZ-NR per family |
|---|---|---|---|---|
| compliant|RS_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0289 [-0.0394, -0.0188] / -0.0286 [-0.0391, -0.0186] | PASS (single training seed; training-seed floor unmeasured): -0.0340 [-0.0450, -0.0234] / -0.0337 [-0.0447, -0.0231] | PASS (single training seed; training-seed floor unmeasured): -0.8439 [-1.1007, -0.6009] / -0.8460 [-1.1048, -0.6003] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: S2:LAT_curvature_mae_1pm_valid'} |
| compliant|GATED_RS_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0332 [-0.0434, -0.0235] / -0.0325 [-0.0427, -0.0227] | PASS (single training seed; training-seed floor unmeasured): -0.0383 [-0.0488, -0.0281] / -0.0376 [-0.0483, -0.0274] | PASS (single training seed; training-seed floor unmeasured): -0.9243 [-1.1787, -0.6922] / -0.9271 [-1.1887, -0.6855] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'} |
| all|RS_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0285 [-0.0387, -0.0189] / -0.0275 [-0.0378, -0.0179] | PASS (single training seed; training-seed floor unmeasured): -0.0340 [-0.0449, -0.0236] / -0.0330 [-0.0439, -0.0227] | PASS (single training seed; training-seed floor unmeasured): -0.9172 [-1.1861, -0.6679] / -0.9135 [-1.1820, -0.6572] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: S2:LAT_curvature_mae_1pm_valid'} |
| all|GATED_RS_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0330 [-0.0432, -0.0235] / -0.0319 [-0.0425, -0.0223] | PASS (single training seed; training-seed floor unmeasured): -0.0385 [-0.0495, -0.0283] / -0.0375 [-0.0484, -0.0272] | PASS (single training seed; training-seed floor unmeasured): -0.9970 [-1.2616, -0.7500] / -0.9951 [-1.2639, -0.7439] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'} |

## FORWARD step30000ema_unc -- IN-SAMPLE path check (the gains were fit on this tree) (controls True)

| set / arm | FZ-E1 (-ha0_ext, S2) | FZ-E1P (-ha0_ext_pose, S2) | FZ-E3 (-ha0_ext, S6) | FZ-NR per family |
|---|---|---|---|---|
| compliant|RS_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0429 [-0.0531, -0.0331] / -0.0428 [-0.0529, -0.0329] | PASS (single training seed; training-seed floor unmeasured): -0.0481 [-0.0591, -0.0376] / -0.0479 [-0.0587, -0.0374] | PASS (single training seed; training-seed floor unmeasured): -0.9221 [-1.1839, -0.6802] / -0.9216 [-1.1794, -0.6809] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'} |
| compliant|GATED_RS_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0474 [-0.0575, -0.0377] / -0.0470 [-0.0568, -0.0372] | PASS (single training seed; training-seed floor unmeasured): -0.0526 [-0.0632, -0.0421] / -0.0522 [-0.0626, -0.0418] | PASS (single training seed; training-seed floor unmeasured): -0.9764 [-1.2381, -0.7426] / -0.9778 [-1.2343, -0.7474] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'} |
| all|RS_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0418 [-0.0517, -0.0323] / -0.0410 [-0.0507, -0.0315] | PASS (single training seed; training-seed floor unmeasured): -0.0473 [-0.0580, -0.0370] / -0.0465 [-0.0569, -0.0362] | PASS (single training seed; training-seed floor unmeasured): -1.0045 [-1.2702, -0.7560] / -0.9926 [-1.2534, -0.7441] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: S2:LAT_curvature_mae_1pm_valid'} |
| all|GATED_RS_FROZEN30k | PASS (single training seed; training-seed floor unmeasured): -0.0469 [-0.0569, -0.0376] / -0.0458 [-0.0557, -0.0365] | PASS (single training seed; training-seed floor unmeasured): -0.0525 [-0.0633, -0.0423] / -0.0513 [-0.0620, -0.0412] | PASS (single training seed; training-seed floor unmeasured): -1.0632 [-1.3350, -0.8139] / -1.0527 [-1.3175, -0.8081] | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'} |

References: `ha0_ext` = E1 echo (steer kappa0, INADMISSIBLE input); `ha0_ext_pose` = refcv8's own causal prior (admissible, PRIMARY); `ha0_imu` = hold a0 + yaw rate (admissible, SECONDARY). Arms: RS = residual scaling on ha0_ext_pose (g in [0,2], per instant x axis, 5-episode-fold cross-fitted); GATED_RS = per v0 bin; _IS in-sample, _WF window-fold (leaky) controls; BLEND_imu = convex blend on ha0_imu.

## step30000ema_unc

### S2 / compliant -- n 4159 windows / 138 ep

ADE means (s0): os 0.2722, ha0_ext 0.2807, ha0_ext_pose 0.2858, ha0_imu 0.2823, ha0_imu_ax 0.3875, refcv7_50400 0.2771

| read | seed 0 | seed 1 |
|---|---|---|
| **E1 AS REGISTERED: os - ha0_ext (bar of record)** | -0.0085 [-0.0229, +0.0065] | -0.0076 [-0.0216, +0.0073] |
| os - ha0_ext_pose (residual vs its own prior; admissible ref) | -0.0136 [-0.0286, +0.0018] | -0.0127 [-0.0272, +0.0024] |
| os - ha0_imu (secondary admissible ref) | -0.0101 [-0.0246, +0.0049] | -0.0092 [-0.0230, +0.0056] |
| os - ha0_imu_ax (a0 = measured ax) | -0.1153 [-0.1669, -0.0689] **SEP** | -0.1144 [-0.1661, -0.0678] **SEP** |
| ha0_ext_pose - ha0_ext | +0.0051 [+0.0034, +0.0069] **SEP** | +0.0051 [+0.0034, +0.0069] **SEP** |
| ha0_imu - ha0_ext | +0.0016 [-0.0001, +0.0032] | +0.0016 [-0.0001, +0.0032] |
| RS_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.0428 [-0.0538, -0.0321] **SEP** | -0.0427 [-0.0534, -0.0321] **SEP** |
| RS_XF - ha0_ext_pose | -0.0479 [-0.0596, -0.0367] **SEP** | -0.0479 [-0.0593, -0.0368] **SEP** |
| RS_XF - ha0_imu | -0.0444 [-0.0552, -0.0336] **SEP** | -0.0443 [-0.0547, -0.0336] **SEP** |
| RS_XF - os (ADE) | -0.0343 [-0.0419, -0.0274] **SEP** | -0.0351 [-0.0425, -0.0284] **SEP** |
| GATED_RS_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.0466 [-0.0571, -0.0363] **SEP** | -0.0464 [-0.0564, -0.0360] **SEP** |
| GATED_RS_XF - ha0_ext_pose | -0.0518 [-0.0626, -0.0411] **SEP** | -0.0515 [-0.0621, -0.0407] **SEP** |
| GATED_RS_XF - ha0_imu | -0.0482 [-0.0583, -0.0378] **SEP** | -0.0479 [-0.0579, -0.0376] **SEP** |
| GATED_RS_XF - os (ADE) | -0.0381 [-0.0462, -0.0308] **SEP** | -0.0388 [-0.0467, -0.0313] **SEP** |
| RS_POS_XF - ha0_ext_pose | -0.0456 [-0.0574, -0.0343] **SEP** | -0.0456 [-0.0569, -0.0345] **SEP** |
| RS_POS_XF - os (ADE) | -0.0319 [-0.0391, -0.0253] **SEP** | -0.0329 [-0.0398, -0.0263] **SEP** |
| RS_IS - ha0_ext_pose | -0.0484 [-0.0600, -0.0374] **SEP** | -0.0483 [-0.0598, -0.0374] **SEP** |
| RS_IS - os (ADE) | -0.0347 [-0.0425, -0.0278] **SEP** | -0.0356 [-0.0429, -0.0287] **SEP** |
| GATED_RS_IS - ha0_ext_pose | -0.0535 [-0.0642, -0.0427] **SEP** | -0.0531 [-0.0635, -0.0422] **SEP** |
| GATED_RS_IS - os (ADE) | -0.0398 [-0.0482, -0.0325] **SEP** | -0.0403 [-0.0486, -0.0328] **SEP** |
| RS_WF - ha0_ext_pose | -0.0483 [-0.0599, -0.0373] **SEP** | -0.0483 [-0.0597, -0.0373] **SEP** |
| RS_WF - os (ADE) | -0.0347 [-0.0424, -0.0277] **SEP** | -0.0355 [-0.0429, -0.0287] **SEP** |
| GATED_RS_WF - ha0_ext_pose | -0.0529 [-0.0637, -0.0422] **SEP** | -0.0525 [-0.0629, -0.0417] **SEP** |
| GATED_RS_WF - os (ADE) | -0.0393 [-0.0475, -0.0320] **SEP** | -0.0397 [-0.0480, -0.0323] **SEP** |
| BLEND_imu_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.0434 [-0.0542, -0.0327] **SEP** | -0.0433 [-0.0539, -0.0326] **SEP** |
| BLEND_imu_XF - ha0_ext_pose | -0.0486 [-0.0603, -0.0375] **SEP** | -0.0484 [-0.0598, -0.0374] **SEP** |
| BLEND_imu_XF - ha0_imu | -0.0450 [-0.0557, -0.0343] **SEP** | -0.0449 [-0.0552, -0.0342] **SEP** |
| BLEND_imu_XF - os (ADE) | -0.0349 [-0.0426, -0.0279] **SEP** | -0.0357 [-0.0430, -0.0289] **SEP** |

| metric (family) | os - ha0_ext_pose s0 | RS_XF - refcv7 s0 | s1 | RS_XF - os s0 | s1 | GATED_RS_XF - refcv7 s0 | GATED_RS_XF - os s0 |
|---|---|---|---|---|---|---|---|
| ade_m (ADE) | -0.0136 [-0.0286, +0.0018] | -0.0393 [-0.0518, -0.0278] **SEP** | -0.0389 [-0.0515, -0.0269] **SEP** | -0.0343 [-0.0419, -0.0274] **SEP** | -0.0351 [-0.0425, -0.0284] **SEP** | -0.0431 [-0.0564, -0.0310] **SEP** | -0.0381 [-0.0462, -0.0308] **SEP** |
| fde_m (ADE) | -0.0389 [-0.0769, -0.0020] **SEP** | -0.0753 [-0.1068, -0.0475] **SEP** | -0.0744 [-0.1065, -0.0454] **SEP** | -0.0658 [-0.0834, -0.0496] **SEP** | -0.0682 [-0.0854, -0.0522] **SEP** | -0.0789 [-0.1121, -0.0495] **SEP** | -0.0694 [-0.0879, -0.0528] **SEP** |
| LON_speed_mae_mps (LONGITUDINAL) | +0.0044 [-0.0091, +0.0185] | -0.0288 [-0.0431, -0.0161] **SEP** | -0.0288 [-0.0436, -0.0161] **SEP** | -0.0250 [-0.0331, -0.0173] **SEP** | -0.0263 [-0.0342, -0.0188] **SEP** | -0.0310 [-0.0468, -0.0173] **SEP** | -0.0272 [-0.0358, -0.0190] **SEP** |
| LON_along_mae_m (LONGITUDINAL) | -0.0067 [-0.0187, +0.0064] | -0.0205 [-0.0324, -0.0095] **SEP** | -0.0202 [-0.0322, -0.0095] **SEP** | -0.0182 [-0.0245, -0.0124] **SEP** | -0.0192 [-0.0253, -0.0135] **SEP** | -0.0250 [-0.0381, -0.0136] **SEP** | -0.0227 [-0.0299, -0.0162] **SEP** |
| LON_along_abs_final_m (LONGITUDINAL) | +0.0071 [-0.0212, +0.0357] | -0.0496 [-0.0788, -0.0231] **SEP** | -0.0490 [-0.0787, -0.0225] **SEP** | -0.0447 [-0.0610, -0.0292] **SEP** | -0.0475 [-0.0638, -0.0324] **SEP** | -0.0545 [-0.0866, -0.0270] **SEP** | -0.0496 [-0.0669, -0.0337] **SEP** |
| LON_speed_within_0p5 (LONGITUDINAL) | -0.0170 [-0.0292, -0.0049] **SEP** | +0.0227 [+0.0121, +0.0345] **SEP** | +0.0241 [+0.0130, +0.0358] **SEP** | +0.0245 [+0.0166, +0.0327] **SEP** | +0.0252 [+0.0177, +0.0330] **SEP** | +0.0225 [+0.0114, +0.0351] **SEP** | +0.0242 [+0.0164, +0.0327] **SEP** |
| LON_accel_mae_mps2 (LONGITUDINAL) | +0.0482 [+0.0269, +0.0680] **SEP** | -0.0701 [-0.0890, -0.0530] **SEP** | -0.0698 [-0.0886, -0.0527] **SEP** | -0.0582 [-0.0703, -0.0460] **SEP** | -0.0599 [-0.0726, -0.0478] **SEP** | -0.0644 [-0.0831, -0.0471] **SEP** | -0.0525 [-0.0637, -0.0413] **SEP** |
| LAT_heading_mae_deg (LATERAL) | -0.1232 [-0.2362, -0.0172] **SEP** | -0.1385 [-0.1886, -0.0905] **SEP** | -0.1266 [-0.1751, -0.0783] **SEP** | -0.1436 [-0.1813, -0.1086] **SEP** | -0.1495 [-0.1900, -0.1134] **SEP** | -0.1372 [-0.1877, -0.0896] **SEP** | -0.1460 [-0.1854, -0.1106] **SEP** |
| LAT_curvature_mae_1pm_valid (LATERAL) | -0.0004 [-0.0009, +0.0002] | +0.0001 [-0.0003, +0.0004] | +0.0002 [-0.0001, +0.0005] | +0.0001 [-0.0002, +0.0005] | +0.0001 [-0.0002, +0.0005] | -0.0001 [-0.0004, +0.0002] | -0.0001 [-0.0003, +0.0003] |
| LAT_yaw_rate_mae_radps_valid (LATERAL) | -0.0054 [-0.0082, -0.0026] **SEP** | -0.0014 [-0.0023, -0.0006] **SEP** | -0.0011 [-0.0020, -0.0002] **SEP** | -0.0011 [-0.0017, -0.0006] **SEP** | -0.0012 [-0.0017, -0.0006] **SEP** | -0.0018 [-0.0026, -0.0009] **SEP** | -0.0016 [-0.0020, -0.0011] **SEP** |
| LAT_cross_mae_m (LATERAL) | -0.0098 [-0.0209, +0.0002] | -0.0270 [-0.0349, -0.0196] **SEP** | -0.0271 [-0.0351, -0.0194] **SEP** | -0.0253 [-0.0314, -0.0199] **SEP** | -0.0256 [-0.0316, -0.0199] **SEP** | -0.0267 [-0.0344, -0.0194] **SEP** | -0.0251 [-0.0311, -0.0197] **SEP** |
| LAT_cross_abs_final_m (LATERAL) | -0.0606 [-0.0908, -0.0338] **SEP** | -0.0376 [-0.0529, -0.0224] **SEP** | -0.0378 [-0.0538, -0.0223] **SEP** | -0.0362 [-0.0466, -0.0266] **SEP** | -0.0369 [-0.0475, -0.0267] **SEP** | -0.0371 [-0.0524, -0.0223] **SEP** | -0.0357 [-0.0463, -0.0261] **SEP** |
| TAC_traj_lat_correct (TACTICAL) | +0.0216 [+0.0113, +0.0322] **SEP** | -0.0012 [-0.0063, +0.0039] | -0.0007 [-0.0059, +0.0045] | +0.0005 [-0.0031, +0.0044] | +0.0014 [-0.0021, +0.0052] | -0.0010 [-0.0063, +0.0046] | +0.0007 [-0.0031, +0.0046] |
| TAC_traj_lon_correct (TACTICAL) | -0.0127 [-0.0282, +0.0021] | +0.0286 [+0.0158, +0.0431] **SEP** | +0.0190 [+0.0058, +0.0336] **SEP** | +0.0219 [+0.0110, +0.0331] **SEP** | +0.0216 [+0.0098, +0.0334] **SEP** | +0.0260 [+0.0118, +0.0415] **SEP** | +0.0192 [+0.0076, +0.0315] **SEP** |

* E1 registered verdict (os floor 0.0009): **FAILED**; os - ha0_ext_pose form: FAILED; os - ha0_imu form: FAILED.
* RS_XF (floor 0.0001): -ha0_ext form **PASS (single training seed; training-seed floor unmeasured)**; -ha0_ext_pose PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); NR rollup (headline rows) {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}.
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): none; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_heading_mae_deg (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_heading_mae_deg (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* GATED_RS_XF (floor 0.0003): -ha0_ext form **PASS (single training seed; training-seed floor unmeasured)**; -ha0_ext_pose PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); NR rollup (headline rows) {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}.
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): none; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_heading_mae_deg (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_heading_mae_deg (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* fold-seed sensitivity (s0, 10 seeds, points): RS_XF -ha0_ext -0.0430..-0.0425, GATED_RS_XF -ha0_ext -0.0473..-0.0464; optimism (-pose) XF-IS +0.0005 / gated +0.0017, XF-WF +0.0004.
* in-sample RS gains g_k (s0) per instant [along, cross]: [[1.038, 0.23], [0.64, 0.376], [0.505, 0.55], [0.473, 0.682]]
* RS_XF fold-mean gains g_k (s0) per instant [along, cross]: [[1.04, 0.23], [0.64, 0.38], [0.5, 0.55], [0.47, 0.68]]; GATED_RS_XF fold-mean per v0 bin: {'bin0': [[0.12, 0.13], [0.07, 0.35], [0.12, 0.35], [0.17, 0.34]], 'bin1': [[0.62, 0.14], [0.3, 0.33], [0.38, 0.53], [0.5, 0.68]], 'bin2': [[1.03, 0.21], [0.65, 0.38], [0.54, 0.56], [0.49, 0.69]], 'bin3': [[1.27, 0.34], [0.82, 0.41], [0.75, 0.57], [0.52, 0.69]], 'bin4': [[1.34, 0.58], [0.86, 0.42], [0.86, 0.46], [0.88, 0.56]]}
* RS_XF fold-mean gains g_k (s1) per instant [along, cross]: [[1.04, 0.23], [0.64, 0.38], [0.51, 0.55], [0.48, 0.67]]; GATED_RS_XF fold-mean per v0 bin: {'bin0': [[0.12, 0.13], [0.09, 0.34], [0.14, 0.41], [0.19, 0.44]], 'bin1': [[0.61, 0.14], [0.3, 0.32], [0.38, 0.52], [0.51, 0.67]], 'bin2': [[1.01, 0.21], [0.64, 0.38], [0.55, 0.55], [0.5, 0.68]], 'bin3': [[1.28, 0.34], [0.82, 0.42], [0.73, 0.56], [0.48, 0.68]], 'bin4': [[1.37, 0.56], [0.87, 0.46], [0.86, 0.51], [0.88, 0.56]]}

### S2 / all -- n 4754 windows / 139 ep

ADE means (s0): os 0.2878, ha0_ext 0.2886, ha0_ext_pose 0.2941, ha0_imu 0.2900, ha0_imu_ax 0.4065, refcv7_50400 0.2784

| read | seed 0 | seed 1 |
|---|---|---|
| **E1 AS REGISTERED: os - ha0_ext (bar of record)** | -0.0008 [-0.0149, +0.0136] | +0.0015 [-0.0131, +0.0163] |
| os - ha0_ext_pose (residual vs its own prior; admissible ref) | -0.0064 [-0.0207, +0.0090] | -0.0041 [-0.0189, +0.0110] |
| os - ha0_imu (secondary admissible ref) | -0.0022 [-0.0163, +0.0125] | +0.0001 [-0.0141, +0.0146] |
| os - ha0_imu_ax (a0 = measured ax) | -0.1187 [-0.1691, -0.0729] **SEP** | -0.1165 [-0.1680, -0.0709] **SEP** |
| ha0_ext_pose - ha0_ext | +0.0055 [+0.0036, +0.0076] **SEP** | +0.0055 [+0.0036, +0.0076] **SEP** |
| ha0_imu - ha0_ext | +0.0014 [-0.0003, +0.0030] | +0.0014 [-0.0003, +0.0030] |
| RS_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.0412 [-0.0511, -0.0317] **SEP** | -0.0401 [-0.0498, -0.0307] **SEP** |
| RS_XF - ha0_ext_pose | -0.0467 [-0.0573, -0.0362] **SEP** | -0.0457 [-0.0561, -0.0354] **SEP** |
| RS_XF - ha0_imu | -0.0426 [-0.0523, -0.0330] **SEP** | -0.0415 [-0.0511, -0.0321] **SEP** |
| RS_XF - os (ADE) | -0.0403 [-0.0484, -0.0319] **SEP** | -0.0416 [-0.0500, -0.0331] **SEP** |
| GATED_RS_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.0458 [-0.0559, -0.0366] **SEP** | -0.0446 [-0.0544, -0.0353] **SEP** |
| GATED_RS_XF - ha0_ext_pose | -0.0514 [-0.0621, -0.0413] **SEP** | -0.0501 [-0.0609, -0.0401] **SEP** |
| GATED_RS_XF - ha0_imu | -0.0472 [-0.0573, -0.0380] **SEP** | -0.0460 [-0.0560, -0.0368] **SEP** |
| GATED_RS_XF - os (ADE) | -0.0450 [-0.0534, -0.0367] **SEP** | -0.0461 [-0.0547, -0.0376] **SEP** |
| RS_POS_XF - ha0_ext_pose | -0.0435 [-0.0543, -0.0331] **SEP** | -0.0428 [-0.0533, -0.0323] **SEP** |
| RS_POS_XF - os (ADE) | -0.0372 [-0.0448, -0.0295] **SEP** | -0.0387 [-0.0466, -0.0308] **SEP** |
| RS_IS - ha0_ext_pose | -0.0473 [-0.0581, -0.0370] **SEP** | -0.0465 [-0.0569, -0.0363] **SEP** |
| RS_IS - os (ADE) | -0.0410 [-0.0490, -0.0327] **SEP** | -0.0424 [-0.0507, -0.0340] **SEP** |
| GATED_RS_IS - ha0_ext_pose | -0.0525 [-0.0634, -0.0423] **SEP** | -0.0513 [-0.0620, -0.0412] **SEP** |
| GATED_RS_IS - os (ADE) | -0.0461 [-0.0545, -0.0377] **SEP** | -0.0473 [-0.0559, -0.0388] **SEP** |
| RS_WF - ha0_ext_pose | -0.0472 [-0.0580, -0.0369] **SEP** | -0.0464 [-0.0568, -0.0362] **SEP** |
| RS_WF - os (ADE) | -0.0409 [-0.0489, -0.0325] **SEP** | -0.0423 [-0.0506, -0.0339] **SEP** |
| GATED_RS_WF - ha0_ext_pose | -0.0520 [-0.0629, -0.0419] **SEP** | -0.0509 [-0.0615, -0.0408] **SEP** |
| GATED_RS_WF - os (ADE) | -0.0457 [-0.0541, -0.0373] **SEP** | -0.0468 [-0.0554, -0.0384] **SEP** |
| BLEND_imu_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.0420 [-0.0520, -0.0325] **SEP** | -0.0410 [-0.0506, -0.0315] **SEP** |
| BLEND_imu_XF - ha0_ext_pose | -0.0475 [-0.0583, -0.0371] **SEP** | -0.0465 [-0.0569, -0.0362] **SEP** |
| BLEND_imu_XF - ha0_imu | -0.0434 [-0.0532, -0.0339] **SEP** | -0.0423 [-0.0520, -0.0329] **SEP** |
| BLEND_imu_XF - os (ADE) | -0.0412 [-0.0493, -0.0327] **SEP** | -0.0424 [-0.0508, -0.0338] **SEP** |

| metric (family) | os - ha0_ext_pose s0 | RS_XF - refcv7 s0 | s1 | RS_XF - os s0 | s1 | GATED_RS_XF - refcv7 s0 | GATED_RS_XF - os s0 |
|---|---|---|---|---|---|---|---|
| ade_m (ADE) | -0.0064 [-0.0207, +0.0090] | -0.0310 [-0.0430, -0.0198] **SEP** | -0.0305 [-0.0431, -0.0187] **SEP** | -0.0403 [-0.0484, -0.0319] **SEP** | -0.0416 [-0.0500, -0.0331] **SEP** | -0.0357 [-0.0485, -0.0244] **SEP** | -0.0450 [-0.0534, -0.0367] **SEP** |
| fde_m (ADE) | -0.0273 [-0.0639, +0.0102] | -0.0593 [-0.0889, -0.0326] **SEP** | -0.0582 [-0.0878, -0.0306] **SEP** | -0.0796 [-0.0982, -0.0609] **SEP** | -0.0828 [-0.1019, -0.0641] **SEP** | -0.0638 [-0.0940, -0.0370] **SEP** | -0.0842 [-0.1032, -0.0655] **SEP** |
| LON_speed_mae_mps (LONGITUDINAL) | +0.0148 [+0.0014, +0.0280] **SEP** | -0.0217 [-0.0350, -0.0097] **SEP** | -0.0216 [-0.0353, -0.0093] **SEP** | -0.0325 [-0.0412, -0.0239] **SEP** | -0.0343 [-0.0434, -0.0255] **SEP** | -0.0242 [-0.0384, -0.0118] **SEP** | -0.0350 [-0.0441, -0.0262] **SEP** |
| LON_along_mae_m (LONGITUDINAL) | +0.0031 [-0.0090, +0.0155] | -0.0126 [-0.0245, -0.0020] **SEP** | -0.0124 [-0.0242, -0.0018] **SEP** | -0.0244 [-0.0317, -0.0174] **SEP** | -0.0260 [-0.0336, -0.0187] **SEP** | -0.0181 [-0.0301, -0.0071] **SEP** | -0.0299 [-0.0376, -0.0226] **SEP** |
| LON_along_abs_final_m (LONGITUDINAL) | +0.0260 [-0.0010, +0.0534] | -0.0358 [-0.0639, -0.0105] **SEP** | -0.0353 [-0.0631, -0.0098] **SEP** | -0.0600 [-0.0780, -0.0424] **SEP** | -0.0638 [-0.0822, -0.0460] **SEP** | -0.0418 [-0.0707, -0.0156] **SEP** | -0.0659 [-0.0841, -0.0481] **SEP** |
| LON_speed_within_0p5 (LONGITUDINAL) | -0.0236 [-0.0347, -0.0126] **SEP** | +0.0180 [+0.0082, +0.0289] **SEP** | +0.0201 [+0.0099, +0.0309] **SEP** | +0.0295 [+0.0211, +0.0378] **SEP** | +0.0316 [+0.0233, +0.0398] **SEP** | +0.0166 [+0.0062, +0.0276] **SEP** | +0.0281 [+0.0200, +0.0366] **SEP** |
| LON_accel_mae_mps2 (LONGITUDINAL) | +0.0559 [+0.0363, +0.0751] **SEP** | -0.0628 [-0.0805, -0.0462] **SEP** | -0.0618 [-0.0798, -0.0453] **SEP** | -0.0652 [-0.0782, -0.0521] **SEP** | -0.0664 [-0.0794, -0.0536] **SEP** | -0.0557 [-0.0736, -0.0401] **SEP** | -0.0582 [-0.0700, -0.0469] **SEP** |
| LAT_heading_mae_deg (LATERAL) | -0.1434 [-0.2638, -0.0240] **SEP** | -0.1165 [-0.1591, -0.0762] **SEP** | -0.1018 [-0.1459, -0.0591] **SEP** | -0.1437 [-0.1803, -0.1093] **SEP** | -0.1459 [-0.1851, -0.1090] **SEP** | -0.1189 [-0.1628, -0.0788] **SEP** | -0.1492 [-0.1852, -0.1149] **SEP** |
| LAT_curvature_mae_1pm_valid (LATERAL) | -0.0004 [-0.0010, +0.0001] | +0.0002 [-0.0001, +0.0005] | +0.0004 [+0.0001, +0.0008] **SEP** | +0.0002 [-0.0001, +0.0006] | +0.0003 [-0.0001, +0.0007] | -0.0001 [-0.0004, +0.0002] | -0.0001 [-0.0004, +0.0002] |
| LAT_yaw_rate_mae_radps_valid (LATERAL) | -0.0058 [-0.0088, -0.0028] **SEP** | -0.0010 [-0.0018, -0.0003] **SEP** | -0.0006 [-0.0015, +0.0002] | -0.0012 [-0.0017, -0.0006] **SEP** | -0.0012 [-0.0017, -0.0006] **SEP** | -0.0015 [-0.0023, -0.0008] **SEP** | -0.0017 [-0.0021, -0.0013] **SEP** |
| LAT_cross_mae_m (LATERAL) | -0.0107 [-0.0218, +0.0007] | -0.0252 [-0.0328, -0.0186] **SEP** | -0.0249 [-0.0328, -0.0182] **SEP** | -0.0264 [-0.0322, -0.0213] **SEP** | -0.0265 [-0.0325, -0.0212] **SEP** | -0.0249 [-0.0326, -0.0182] **SEP** | -0.0260 [-0.0319, -0.0209] **SEP** |
| LAT_cross_abs_final_m (LATERAL) | -0.0662 [-0.0977, -0.0358] **SEP** | -0.0329 [-0.0480, -0.0191] **SEP** | -0.0322 [-0.0478, -0.0183] **SEP** | -0.0369 [-0.0474, -0.0276] **SEP** | -0.0371 [-0.0478, -0.0270] **SEP** | -0.0322 [-0.0471, -0.0186] **SEP** | -0.0362 [-0.0469, -0.0268] **SEP** |
| TAC_traj_lat_correct (TACTICAL) | +0.0223 [+0.0114, +0.0331] **SEP** | -0.0021 [-0.0067, +0.0025] | -0.0025 [-0.0074, +0.0023] | +0.0000 [-0.0036, +0.0036] | +0.0021 [-0.0019, +0.0061] | -0.0017 [-0.0063, +0.0032] | +0.0004 [-0.0034, +0.0040] |
| TAC_traj_lon_correct (TACTICAL) | -0.0168 [-0.0317, -0.0019] **SEP** | +0.0255 [+0.0120, +0.0400] **SEP** | +0.0164 [+0.0038, +0.0297] **SEP** | +0.0290 [+0.0179, +0.0415] **SEP** | +0.0267 [+0.0156, +0.0381] **SEP** | +0.0181 [+0.0044, +0.0335] **SEP** | +0.0217 [+0.0105, +0.0335] **SEP** |

* E1 registered verdict (os floor 0.0023): **FAILED**; os - ha0_ext_pose form: FAILED; os - ha0_imu form: FAILED.
* RS_XF (floor 0.0010): -ha0_ext form **PASS (single training seed; training-seed floor unmeasured)**; -ha0_ext_pose PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); NR rollup (headline rows) {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: LAT_curvature_mae_1pm_valid'}.
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): none; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_heading_mae_deg (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_heading_mae_deg (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* GATED_RS_XF (floor 0.0012): -ha0_ext form **PASS (single training seed; training-seed floor unmeasured)**; -ha0_ext_pose PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); NR rollup (headline rows) {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}.
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): none; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_heading_mae_deg (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_heading_mae_deg (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* fold-seed sensitivity (s0, 10 seeds, points): RS_XF -ha0_ext -0.0416..-0.0411, GATED_RS_XF -ha0_ext -0.0457..-0.0452; optimism (-pose) XF-IS +0.0006 / gated +0.0011, XF-WF +0.0005.
* in-sample RS gains g_k (s0) per instant [along, cross]: [[1.001, 0.23], [0.583, 0.383], [0.434, 0.551], [0.415, 0.684]]
* RS_XF fold-mean gains g_k (s0) per instant [along, cross]: [[1.0, 0.23], [0.58, 0.38], [0.43, 0.55], [0.41, 0.68]]; GATED_RS_XF fold-mean per v0 bin: {'bin0': [[0.12, 0.17], [0.09, 0.32], [0.12, 0.34], [0.16, 0.33]], 'bin1': [[0.64, 0.16], [0.29, 0.37], [0.37, 0.56], [0.48, 0.71]], 'bin2': [[0.85, 0.2], [0.41, 0.37], [0.31, 0.54], [0.34, 0.67]], 'bin3': [[1.27, 0.34], [0.82, 0.41], [0.74, 0.56], [0.5, 0.68]], 'bin4': [[1.38, 0.66], [0.87, 0.46], [0.83, 0.49], [0.81, 0.57]]}
* RS_XF fold-mean gains g_k (s1) per instant [along, cross]: [[0.99, 0.22], [0.58, 0.38], [0.42, 0.55], [0.4, 0.68]]; GATED_RS_XF fold-mean per v0 bin: {'bin0': [[0.12, 0.17], [0.1, 0.32], [0.14, 0.41], [0.18, 0.43]], 'bin1': [[0.63, 0.15], [0.3, 0.36], [0.37, 0.57], [0.5, 0.71]], 'bin2': [[0.81, 0.19], [0.39, 0.36], [0.3, 0.53], [0.32, 0.66]], 'bin3': [[1.28, 0.34], [0.82, 0.42], [0.71, 0.56], [0.46, 0.68]], 'bin4': [[1.39, 0.6], [0.87, 0.48], [0.84, 0.51], [0.8, 0.55]]}
* validation (s0): V1 0.00e+00 m, V2 0.00e+00 m, V3 imu vs ext >0.1 m {'2s': 0.1773243584350021}, pose vs ext >0.1 m {'2s': 0.16281026503996635}, pose vs ext max abs 1.588 m.

### S6 / compliant -- n 3251 windows / 138 ep

ADE means (s0): os 2.6275, ha0_ext 3.3810, ha0_ext_pose 3.4098, ha0_imu 3.2977, ha0_imu_ax 3.7217, refcv7_50400 2.5088

| read | seed 0 | seed 1 |
|---|---|---|
| **E1 AS REGISTERED: os - ha0_ext (bar of record)** | -0.7535 [-1.0832, -0.4506] **SEP** | -0.7430 [-1.0699, -0.4363] **SEP** |
| os - ha0_ext_pose (residual vs its own prior; admissible ref) | -0.7823 [-1.1159, -0.4748] **SEP** | -0.7717 [-1.1024, -0.4582] **SEP** |
| os - ha0_imu (secondary admissible ref) | -0.6702 [-0.9721, -0.3947] **SEP** | -0.6596 [-0.9587, -0.3786] **SEP** |
| os - ha0_imu_ax (a0 = measured ax) | -1.0942 [-1.5549, -0.6762] **SEP** | -1.0836 [-1.5431, -0.6645] **SEP** |
| ha0_ext_pose - ha0_ext | +0.0288 [+0.0134, +0.0440] **SEP** | +0.0288 [+0.0134, +0.0440] **SEP** |
| ha0_imu - ha0_ext | -0.0833 [-0.1402, -0.0383] **SEP** | -0.0833 [-0.1402, -0.0383] **SEP** |
| RS_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.9162 [-1.1794, -0.6723] **SEP** | -0.9161 [-1.1758, -0.6742] **SEP** |
| RS_XF - ha0_ext_pose | -0.9450 [-1.2100, -0.6967] **SEP** | -0.9449 [-1.2098, -0.6947] **SEP** |
| RS_XF - ha0_imu | -0.8329 [-1.0611, -0.6159] **SEP** | -0.8328 [-1.0613, -0.6143] **SEP** |
| RS_XF - os (ADE) | -0.1627 [-0.2464, -0.0763] **SEP** | -0.1732 [-0.2554, -0.0869] **SEP** |
| GATED_RS_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.9567 [-1.2171, -0.7214] **SEP** | -0.9582 [-1.2167, -0.7215] **SEP** |
| GATED_RS_XF - ha0_ext_pose | -0.9854 [-1.2506, -0.7412] **SEP** | -0.9869 [-1.2515, -0.7433] **SEP** |
| GATED_RS_XF - ha0_imu | -0.8734 [-1.1084, -0.6583] **SEP** | -0.8748 [-1.1055, -0.6612] **SEP** |
| GATED_RS_XF - os (ADE) | -0.2031 [-0.3023, -0.1035] **SEP** | -0.2152 [-0.3205, -0.1128] **SEP** |
| RS_POS_XF - ha0_ext_pose | -0.9007 [-1.1819, -0.6382] **SEP** | -0.8993 [-1.1779, -0.6359] **SEP** |
| RS_POS_XF - os (ADE) | -0.1184 [-0.1749, -0.0556] **SEP** | -0.1276 [-0.1865, -0.0626] **SEP** |
| RS_IS - ha0_ext_pose | -0.9512 [-1.2151, -0.7030] **SEP** | -0.9509 [-1.2163, -0.7017] **SEP** |
| RS_IS - os (ADE) | -0.1689 [-0.2530, -0.0827] **SEP** | -0.1792 [-0.2635, -0.0927] **SEP** |
| GATED_RS_IS - ha0_ext_pose | -1.0071 [-1.2729, -0.7662] **SEP** | -1.0087 [-1.2734, -0.7690] **SEP** |
| GATED_RS_IS - os (ADE) | -0.2248 [-0.3271, -0.1232] **SEP** | -0.2370 [-0.3427, -0.1342] **SEP** |
| RS_WF - ha0_ext_pose | -0.9497 [-1.2126, -0.7019] **SEP** | -0.9487 [-1.2136, -0.6999] **SEP** |
| RS_WF - os (ADE) | -0.1674 [-0.2513, -0.0810] **SEP** | -0.1770 [-0.2622, -0.0900] **SEP** |
| GATED_RS_WF - ha0_ext_pose | -1.0029 [-1.2690, -0.7619] **SEP** | -1.0036 [-1.2688, -0.7642] **SEP** |
| GATED_RS_WF - os (ADE) | -0.2206 [-0.3239, -0.1194] **SEP** | -0.2319 [-0.3372, -0.1289] **SEP** |
| BLEND_imu_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.9427 [-1.2098, -0.6981] **SEP** | -0.9415 [-1.2069, -0.6974] **SEP** |
| BLEND_imu_XF - ha0_ext_pose | -0.9715 [-1.2438, -0.7232] **SEP** | -0.9703 [-1.2370, -0.7216] **SEP** |
| BLEND_imu_XF - ha0_imu | -0.8594 [-1.0918, -0.6434] **SEP** | -0.8582 [-1.0867, -0.6395] **SEP** |
| BLEND_imu_XF - os (ADE) | -0.1892 [-0.2739, -0.1033] **SEP** | -0.1986 [-0.2837, -0.1109] **SEP** |

| metric (family) | os - ha0_ext_pose s0 | RS_XF - refcv7 s0 | s1 | RS_XF - os s0 | s1 | GATED_RS_XF - refcv7 s0 | GATED_RS_XF - os s0 |
|---|---|---|---|---|---|---|---|
| ade_m (ADE) | -0.7823 [-1.1159, -0.4748] **SEP** | -0.0440 [-0.1872, +0.0975] | -0.0360 [-0.1806, +0.1077] | -0.1627 [-0.2464, -0.0763] **SEP** | -0.1732 [-0.2554, -0.0869] **SEP** | -0.0844 [-0.2326, +0.0537] | -0.2031 [-0.3023, -0.1035] **SEP** |
| fde_m (ADE) | -2.2337 [-3.1803, -1.3815] **SEP** | +0.0221 [-0.3505, +0.4011] | +0.0323 [-0.3437, +0.4129] | -0.3767 [-0.6047, -0.1394] **SEP** | -0.3976 [-0.6205, -0.1595] **SEP** | -0.0845 [-0.4787, +0.2784] | -0.4833 [-0.7707, -0.1956] **SEP** |
| LON_speed_mae_mps (LONGITUDINAL) | -0.0809 [-0.1788, +0.0159] | -0.0092 [-0.0710, +0.0524] | -0.0109 [-0.0727, +0.0499] | -0.0721 [-0.1074, -0.0350] **SEP** | -0.0743 [-0.1103, -0.0364] **SEP** | -0.0214 [-0.0849, +0.0373] | -0.0843 [-0.1325, -0.0380] **SEP** |
| LON_along_mae_m (LONGITUDINAL) | -0.2693 [-0.5174, -0.0421] **SEP** | -0.0418 [-0.1807, +0.0948] | -0.0441 [-0.1864, +0.0948] | -0.1573 [-0.2468, -0.0661] **SEP** | -0.1727 [-0.2607, -0.0821] **SEP** | -0.0830 [-0.2258, +0.0483] | -0.1985 [-0.3024, -0.0970] **SEP** |
| LON_along_abs_final_m (LONGITUDINAL) | -0.8710 [-1.6004, -0.2125] **SEP** | -0.0226 [-0.3931, +0.3445] | -0.0365 [-0.4081, +0.3292] | -0.3956 [-0.6363, -0.1433] **SEP** | -0.4284 [-0.6603, -0.1796] **SEP** | -0.1217 [-0.4941, +0.2219] | -0.4947 [-0.7896, -0.2001] **SEP** |
| LON_speed_within_0p5 (LONGITUDINAL) | -0.0177 [-0.0416, +0.0053] | -0.0054 [-0.0248, +0.0123] | -0.0008 [-0.0197, +0.0169] | +0.0201 [+0.0103, +0.0302] **SEP** | +0.0242 [+0.0139, +0.0349] **SEP** | -0.0019 [-0.0213, +0.0164] | +0.0236 [+0.0114, +0.0359] **SEP** |
| LON_accel_mae_mps2 (LONGITUDINAL) | -0.0277 [-0.0701, +0.0146] | +0.0076 [-0.0149, +0.0309] | +0.0060 [-0.0169, +0.0296] | -0.0306 [-0.0450, -0.0157] **SEP** | -0.0304 [-0.0450, -0.0152] **SEP** | +0.0051 [-0.0169, +0.0259] | -0.0331 [-0.0544, -0.0129] **SEP** |
| LAT_heading_mae_deg (LATERAL) | -2.4598 [-3.3069, -1.6944] **SEP** | +0.3047 [+0.0891, +0.5420] **SEP** | +0.3577 [+0.1229, +0.6116] **SEP** | +0.1773 [+0.0644, +0.3123] **SEP** | +0.1715 [+0.0652, +0.3063] **SEP** | +0.2138 [-0.0008, +0.4411] | +0.0901 [-0.0051, +0.1986] |
| LAT_curvature_mae_1pm_valid (LATERAL) | -0.0020 [-0.0031, -0.0010] **SEP** | +0.0007 [+0.0002, +0.0013] **SEP** | +0.0009 [+0.0004, +0.0015] **SEP** | +0.0007 [+0.0003, +0.0013] **SEP** | +0.0007 [+0.0003, +0.0012] **SEP** | +0.0004 [-0.0000, +0.0008] | +0.0004 [+0.0001, +0.0007] **SEP** |
| LAT_yaw_rate_mae_radps_valid (LATERAL) | -0.0184 [-0.0259, -0.0116] **SEP** | +0.0021 [+0.0002, +0.0041] **SEP** | +0.0024 [+0.0006, +0.0043] **SEP** | +0.0019 [+0.0007, +0.0033] **SEP** | +0.0018 [+0.0006, +0.0032] **SEP** | +0.0014 [-0.0003, +0.0032] | +0.0012 [+0.0001, +0.0023] **SEP** |
| LAT_cross_mae_m (LATERAL) | -0.7751 [-1.0014, -0.5609] **SEP** | -0.0013 [-0.0542, +0.0498] | +0.0020 [-0.0560, +0.0581] | -0.0235 [-0.0372, -0.0102] **SEP** | -0.0232 [-0.0376, -0.0093] **SEP** | -0.0077 [-0.0647, +0.0455] | -0.0300 [-0.0462, -0.0136] **SEP** |
| LAT_cross_abs_final_m (LATERAL) | -2.0827 [-2.6829, -1.5116] **SEP** | +0.0570 [-0.0824, +0.1950] | +0.0610 [-0.0860, +0.2076] | -0.0166 [-0.0421, +0.0107] | -0.0160 [-0.0427, +0.0139] | +0.0330 [-0.1203, +0.1814] | -0.0407 [-0.0811, -0.0019] **SEP** |
| TAC_traj_lat_correct (TACTICAL) | +0.1012 [+0.0763, +0.1264] **SEP** | -0.0043 [-0.0157, +0.0070] | -0.0055 [-0.0184, +0.0061] | -0.0031 [-0.0091, +0.0031] | -0.0055 [-0.0134, +0.0021] | -0.0077 [-0.0205, +0.0054] | -0.0065 [-0.0149, +0.0019] |
| TAC_traj_lon_correct (TACTICAL) | +0.0243 [-0.0128, +0.0618] | -0.0311 [-0.0666, +0.0016] | -0.0268 [-0.0616, +0.0064] | +0.0015 [-0.0159, +0.0194] | +0.0018 [-0.0152, +0.0189] | -0.0258 [-0.0600, +0.0084] | +0.0068 [-0.0161, +0.0299] |

* E1 registered verdict (os floor 0.0106): **PASS (single training seed; training-seed floor unmeasured)**; os - ha0_ext_pose form: PASS (single training seed; training-seed floor unmeasured); os - ha0_imu form: PASS (single training seed; training-seed floor unmeasured).
* RS_XF (floor 0.0001): -ha0_ext form **PASS (single training seed; training-seed floor unmeasured)**; -ha0_ext_pose PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); NR rollup (headline rows) {'ADE': 'PASS'}.
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): ['LAT_heading_mae_deg (LATERAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)']; BETTER: none
  * vs refcv8 os, separated WORSE at both seeds: **['LAT_heading_mae_deg (LATERAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)']**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_cross_mae_m (LATERAL)']
* GATED_RS_XF (floor 0.0015): -ha0_ext form **PASS (single training seed; training-seed floor unmeasured)**; -ha0_ext_pose PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); NR rollup (headline rows) {'ADE': 'PASS'}.
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): none; BETTER: none
  * vs refcv8 os, separated WORSE at both seeds: **['LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)']**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)']
* fold-seed sensitivity (s0, 10 seeds, points): RS_XF -ha0_ext -0.9164..-0.8864, GATED_RS_XF -ha0_ext -0.9570..-0.9336; optimism (-pose) XF-IS +0.0062 / gated +0.0217, XF-WF +0.0047.
* in-sample RS gains g_k (s0) per instant [along, cross]: [[0.647, 0.363], [0.489, 0.675], [0.561, 0.85], [0.614, 0.907], [0.652, 0.933], [0.662, 0.946]]
* RS_XF fold-mean gains g_k (s0) per instant [along, cross]: [[0.64, 0.36], [0.49, 0.68], [0.56, 0.85], [0.61, 0.91], [0.65, 0.93], [0.66, 0.94]]; GATED_RS_XF fold-mean per v0 bin: {'bin0': [[0.12, 0.15], [0.23, 0.35], [0.34, 0.4], [0.43, 0.46], [0.46, 0.38], [0.48, 0.3]], 'bin1': [[0.33, 0.31], [0.52, 0.67], [0.71, 0.85], [0.8, 0.9], [0.83, 0.9], [0.83, 0.91]], 'bin2': [[0.63, 0.38], [0.47, 0.69], [0.51, 0.86], [0.53, 0.92], [0.55, 0.96], [0.56, 0.98]], 'bin3': [[0.82, 0.39], [0.47, 0.66], [0.18, 0.85], [0.2, 0.93], [0.23, 0.96], [0.24, 0.99]], 'bin4': [[0.86, 0.38], [0.88, 0.52], [0.89, 0.63], [0.71, 0.66], [0.62, 0.69], [0.56, 0.7]]}
* RS_XF fold-mean gains g_k (s1) per instant [along, cross]: [[0.65, 0.36], [0.48, 0.66], [0.55, 0.84], [0.61, 0.9], [0.65, 0.93], [0.66, 0.94]]; GATED_RS_XF fold-mean per v0 bin: {'bin0': [[0.15, 0.09], [0.25, 0.41], [0.31, 0.39], [0.39, 0.46], [0.42, 0.38], [0.45, 0.31]], 'bin1': [[0.35, 0.3], [0.53, 0.66], [0.72, 0.84], [0.81, 0.89], [0.83, 0.89], [0.83, 0.91]], 'bin2': [[0.63, 0.37], [0.47, 0.68], [0.51, 0.85], [0.54, 0.92], [0.55, 0.96], [0.57, 0.97]], 'bin3': [[0.81, 0.4], [0.4, 0.65], [0.19, 0.85], [0.2, 0.94], [0.23, 0.97], [0.25, 1.0]], 'bin4': [[0.87, 0.48], [0.89, 0.57], [0.85, 0.66], [0.68, 0.71], [0.59, 0.73], [0.52, 0.74]]}

### S6 / all -- n 3668 windows / 139 ep

ADE means (s0): os 2.7333, ha0_ext 3.5479, ha0_ext_pose 3.5764, ha0_imu 3.4480, ha0_imu_ax 3.8578, refcv7_50400 2.5675

| read | seed 0 | seed 1 |
|---|---|---|
| **E1 AS REGISTERED: os - ha0_ext (bar of record)** | -0.8146 [-1.1390, -0.5132] **SEP** | -0.7902 [-1.1101, -0.4921] **SEP** |
| os - ha0_ext_pose (residual vs its own prior; admissible ref) | -0.8432 [-1.1730, -0.5332] **SEP** | -0.8187 [-1.1463, -0.5109] **SEP** |
| os - ha0_imu (secondary admissible ref) | -0.7148 [-1.0050, -0.4390] **SEP** | -0.6903 [-0.9729, -0.4157] **SEP** |
| os - ha0_imu_ax (a0 = measured ax) | -1.1245 [-1.5607, -0.7413] **SEP** | -1.1001 [-1.5426, -0.7120] **SEP** |
| ha0_ext_pose - ha0_ext | +0.0286 [+0.0108, +0.0472] **SEP** | +0.0286 [+0.0108, +0.0472] **SEP** |
| ha0_imu - ha0_ext | -0.0998 [-0.1492, -0.0525] **SEP** | -0.0998 [-0.1492, -0.0525] **SEP** |
| RS_XF - ha0_ext (E1 form; lever, NOT a bar) | -1.0018 [-1.2678, -0.7533] **SEP** | -0.9895 [-1.2504, -0.7423] **SEP** |
| RS_XF - ha0_ext_pose | -1.0304 [-1.3075, -0.7709] **SEP** | -1.0181 [-1.2915, -0.7621] **SEP** |
| RS_XF - ha0_imu | -0.9020 [-1.1310, -0.6844] **SEP** | -0.8897 [-1.1160, -0.6761] **SEP** |
| RS_XF - os (ADE) | -0.1872 [-0.2689, -0.1026] **SEP** | -0.1994 [-0.2823, -0.1141] **SEP** |
| GATED_RS_XF - ha0_ext (E1 form; lever, NOT a bar) | -1.0450 [-1.3181, -0.7932] **SEP** | -1.0336 [-1.2988, -0.7888] **SEP** |
| GATED_RS_XF - ha0_ext_pose | -1.0736 [-1.3582, -0.8125] **SEP** | -1.0622 [-1.3400, -0.8082] **SEP** |
| GATED_RS_XF - ha0_imu | -0.9452 [-1.1773, -0.7243] **SEP** | -0.9337 [-1.1608, -0.7182] **SEP** |
| GATED_RS_XF - os (ADE) | -0.2304 [-0.3235, -0.1326] **SEP** | -0.2434 [-0.3407, -0.1436] **SEP** |
| RS_POS_XF - ha0_ext_pose | -0.9748 [-1.2618, -0.7094] **SEP** | -0.9616 [-1.2417, -0.6974] **SEP** |
| RS_POS_XF - os (ADE) | -0.1316 [-0.1906, -0.0738] **SEP** | -0.1429 [-0.2062, -0.0809] **SEP** |
| RS_IS - ha0_ext_pose | -1.0331 [-1.3108, -0.7738] **SEP** | -1.0212 [-1.2961, -0.7643] **SEP** |
| RS_IS - os (ADE) | -0.1900 [-0.2709, -0.1049] **SEP** | -0.2025 [-0.2852, -0.1184] **SEP** |
| GATED_RS_IS - ha0_ext_pose | -1.0919 [-1.3797, -0.8320] **SEP** | -1.0814 [-1.3600, -0.8292] **SEP** |
| GATED_RS_IS - os (ADE) | -0.2487 [-0.3450, -0.1502] **SEP** | -0.2626 [-0.3650, -0.1599] **SEP** |
| RS_WF - ha0_ext_pose | -1.0323 [-1.3097, -0.7731] **SEP** | -1.0195 [-1.2929, -0.7633] **SEP** |
| RS_WF - os (ADE) | -0.1892 [-0.2701, -0.1041] **SEP** | -0.2007 [-0.2836, -0.1163] **SEP** |
| GATED_RS_WF - ha0_ext_pose | -1.0857 [-1.3723, -0.8269] **SEP** | -1.0734 [-1.3516, -0.8207] **SEP** |
| GATED_RS_WF - os (ADE) | -0.2426 [-0.3378, -0.1437] **SEP** | -0.2547 [-0.3555, -0.1511] **SEP** |
| BLEND_imu_XF - ha0_ext (E1 form; lever, NOT a bar) | -1.0315 [-1.3059, -0.7753] **SEP** | -1.0193 [-1.2873, -0.7660] **SEP** |
| BLEND_imu_XF - ha0_ext_pose | -1.0601 [-1.3444, -0.7932] **SEP** | -1.0479 [-1.3273, -0.7870] **SEP** |
| BLEND_imu_XF - ha0_imu | -0.9317 [-1.1692, -0.7116] **SEP** | -0.9195 [-1.1536, -0.7013] **SEP** |
| BLEND_imu_XF - os (ADE) | -0.2169 [-0.3000, -0.1310] **SEP** | -0.2292 [-0.3155, -0.1415] **SEP** |

| metric (family) | os - ha0_ext_pose s0 | RS_XF - refcv7 s0 | s1 | RS_XF - os s0 | s1 | GATED_RS_XF - refcv7 s0 | GATED_RS_XF - os s0 |
|---|---|---|---|---|---|---|---|
| ade_m (ADE) | -0.8432 [-1.1730, -0.5332] **SEP** | -0.0214 [-0.1508, +0.1068] | -0.0099 [-0.1400, +0.1202] | -0.1872 [-0.2689, -0.1026] **SEP** | -0.1994 [-0.2823, -0.1141] **SEP** | -0.0646 [-0.1943, +0.0616] | -0.2304 [-0.3235, -0.1326] **SEP** |
| fde_m (ADE) | -2.4630 [-3.3918, -1.5874] **SEP** | +0.0538 [-0.2819, +0.3937] | +0.0762 [-0.2610, +0.4303] | -0.4226 [-0.6448, -0.1990] **SEP** | -0.4453 [-0.6745, -0.2249] **SEP** | -0.0642 [-0.4057, +0.2758] | -0.5405 [-0.8074, -0.2684] **SEP** |
| LON_speed_mae_mps (LONGITUDINAL) | -0.0849 [-0.1762, +0.0028] | -0.0039 [-0.0598, +0.0509] | -0.0043 [-0.0593, +0.0519] | -0.0818 [-0.1165, -0.0474] **SEP** | -0.0845 [-0.1205, -0.0499] **SEP** | -0.0187 [-0.0735, +0.0337] | -0.0966 [-0.1403, -0.0526] **SEP** |
| LON_along_mae_m (LONGITUDINAL) | -0.2900 [-0.5227, -0.0613] **SEP** | -0.0144 [-0.1398, +0.1096] | -0.0149 [-0.1438, +0.1083] | -0.1822 [-0.2677, -0.0922] **SEP** | -0.1996 [-0.2877, -0.1118] **SEP** | -0.0577 [-0.1825, +0.0619] | -0.2255 [-0.3220, -0.1228] **SEP** |
| LON_along_abs_final_m (LONGITUDINAL) | -0.9912 [-1.6651, -0.3300] **SEP** | +0.0317 [-0.3058, +0.3673] | +0.0236 [-0.3185, +0.3612] | -0.4396 [-0.6699, -0.2044] **SEP** | -0.4762 [-0.7130, -0.2484] **SEP** | -0.0787 [-0.4024, +0.2344] | -0.5499 [-0.8218, -0.2660] **SEP** |
| LON_speed_within_0p5 (LONGITUDINAL) | -0.0175 [-0.0381, +0.0044] | -0.0085 [-0.0261, +0.0086] | -0.0049 [-0.0218, +0.0120] | +0.0209 [+0.0113, +0.0303] **SEP** | +0.0230 [+0.0129, +0.0326] **SEP** | -0.0037 [-0.0213, +0.0138] | +0.0257 [+0.0133, +0.0376] **SEP** |
| LON_accel_mae_mps2 (LONGITUDINAL) | -0.0357 [-0.0748, +0.0034] | +0.0077 [-0.0133, +0.0299] | +0.0067 [-0.0143, +0.0286] | -0.0316 [-0.0453, -0.0173] **SEP** | -0.0318 [-0.0462, -0.0170] **SEP** | +0.0043 [-0.0163, +0.0238] | -0.0350 [-0.0531, -0.0176] **SEP** |
| LAT_heading_mae_deg (LATERAL) | -2.6917 [-3.5746, -1.8229] **SEP** | +0.3388 [+0.1424, +0.5511] **SEP** | +0.3917 [+0.1797, +0.6223] **SEP** | +0.2030 [+0.0887, +0.3289] **SEP** | +0.1923 [+0.0821, +0.3147] **SEP** | +0.2022 [+0.0126, +0.4007] **SEP** | +0.0687 [-0.0120, +0.1583] |
| LAT_curvature_mae_1pm_valid (LATERAL) | -0.0022 [-0.0032, -0.0011] **SEP** | +0.0008 [+0.0004, +0.0013] **SEP** | +0.0010 [+0.0005, +0.0016] **SEP** | +0.0008 [+0.0004, +0.0013] **SEP** | +0.0008 [+0.0004, +0.0013] **SEP** | +0.0005 [+0.0001, +0.0009] **SEP** | +0.0004 [+0.0002, +0.0007] **SEP** |
| LAT_yaw_rate_mae_radps_valid (LATERAL) | -0.0199 [-0.0271, -0.0128] **SEP** | +0.0025 [+0.0009, +0.0043] **SEP** | +0.0029 [+0.0013, +0.0047] **SEP** | +0.0022 [+0.0010, +0.0036] **SEP** | +0.0021 [+0.0009, +0.0035] **SEP** | +0.0018 [+0.0002, +0.0036] **SEP** | +0.0015 [+0.0004, +0.0028] **SEP** |
| LAT_cross_mae_m (LATERAL) | -0.8370 [-1.0711, -0.6182] **SEP** | -0.0024 [-0.0496, +0.0443] | +0.0036 [-0.0472, +0.0553] | -0.0233 [-0.0351, -0.0129] **SEP** | -0.0226 [-0.0350, -0.0111] **SEP** | -0.0065 [-0.0572, +0.0441] | -0.0274 [-0.0415, -0.0139] **SEP** |
| LAT_cross_abs_final_m (LATERAL) | -2.2661 [-2.8717, -1.6964] **SEP** | +0.0411 [-0.0818, +0.1671] | +0.0537 [-0.0762, +0.1915] | -0.0135 [-0.0332, +0.0053] | -0.0125 [-0.0346, +0.0090] | +0.0234 [-0.1106, +0.1658] | -0.0312 [-0.0667, +0.0002] |
| TAC_traj_lat_correct (TACTICAL) | +0.1022 [+0.0787, +0.1264] **SEP** | -0.0033 [-0.0136, +0.0071] | -0.0057 [-0.0163, +0.0044] | +0.0005 [-0.0060, +0.0074] | -0.0046 [-0.0120, +0.0030] | -0.0055 [-0.0161, +0.0052] | -0.0016 [-0.0093, +0.0060] |
| TAC_traj_lon_correct (TACTICAL) | +0.0294 [-0.0063, +0.0642] | -0.0240 [-0.0590, +0.0085] | -0.0172 [-0.0515, +0.0145] | -0.0014 [-0.0188, +0.0158] | +0.0016 [-0.0139, +0.0180] | -0.0139 [-0.0486, +0.0178] | +0.0087 [-0.0136, +0.0298] |

* E1 registered verdict (os floor 0.0244): **PASS (single training seed; training-seed floor unmeasured)**; os - ha0_ext_pose form: PASS (single training seed; training-seed floor unmeasured); os - ha0_imu form: PASS (single training seed; training-seed floor unmeasured).
* RS_XF (floor 0.0123): -ha0_ext form **PASS (single training seed; training-seed floor unmeasured)**; -ha0_ext_pose PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); NR rollup (headline rows) {'ADE': 'PASS'}.
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): ['LAT_heading_mae_deg (LATERAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)']; BETTER: none
  * vs refcv8 os, separated WORSE at both seeds: **['LAT_heading_mae_deg (LATERAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)']**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_cross_mae_m (LATERAL)']
* GATED_RS_XF (floor 0.0114): -ha0_ext form **PASS (single training seed; training-seed floor unmeasured)**; -ha0_ext_pose PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); NR rollup (headline rows) {'ADE': 'PASS'}.
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): ['LAT_heading_mae_deg (LATERAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)']; BETTER: none
  * vs refcv8 os, separated WORSE at both seeds: **['LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)']**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_cross_mae_m (LATERAL)']
* fold-seed sensitivity (s0, 10 seeds, points): RS_XF -ha0_ext -1.0022..-0.9785, GATED_RS_XF -ha0_ext -1.0404..-1.0179; optimism (-pose) XF-IS +0.0027 / gated +0.0183, XF-WF +0.0019.
* in-sample RS gains g_k (s0) per instant [along, cross]: [[0.598, 0.367], [0.45, 0.675], [0.535, 0.853], [0.59, 0.911], [0.637, 0.94], [0.651, 0.954]]
* RS_XF fold-mean gains g_k (s0) per instant [along, cross]: [[0.6, 0.37], [0.45, 0.67], [0.53, 0.85], [0.59, 0.91], [0.64, 0.94], [0.65, 0.95]]; GATED_RS_XF fold-mean per v0 bin: {'bin0': [[0.15, 0.17], [0.24, 0.36], [0.36, 0.4], [0.44, 0.49], [0.47, 0.42], [0.52, 0.34]], 'bin1': [[0.33, 0.35], [0.52, 0.7], [0.71, 0.87], [0.8, 0.91], [0.83, 0.91], [0.83, 0.92]], 'bin2': [[0.43, 0.36], [0.36, 0.67], [0.43, 0.84], [0.49, 0.92], [0.52, 0.96], [0.54, 0.98]], 'bin3': [[0.82, 0.39], [0.45, 0.66], [0.16, 0.84], [0.18, 0.93], [0.21, 0.95], [0.24, 0.98]], 'bin4': [[0.86, 0.44], [0.84, 0.52], [0.72, 0.61], [0.51, 0.65], [0.39, 0.67], [0.34, 0.69]]}
* RS_XF fold-mean gains g_k (s1) per instant [along, cross]: [[0.6, 0.36], [0.43, 0.66], [0.52, 0.84], [0.59, 0.91], [0.63, 0.93], [0.65, 0.95]]; GATED_RS_XF fold-mean per v0 bin: {'bin0': [[0.16, 0.16], [0.26, 0.41], [0.34, 0.41], [0.42, 0.49], [0.44, 0.43], [0.48, 0.36]], 'bin1': [[0.33, 0.34], [0.53, 0.7], [0.72, 0.86], [0.81, 0.9], [0.83, 0.9], [0.83, 0.92]], 'bin2': [[0.42, 0.35], [0.34, 0.65], [0.42, 0.83], [0.47, 0.91], [0.51, 0.95], [0.54, 0.97]], 'bin3': [[0.81, 0.4], [0.36, 0.65], [0.16, 0.84], [0.18, 0.93], [0.21, 0.96], [0.24, 1.0]], 'bin4': [[0.88, 0.49], [0.86, 0.56], [0.71, 0.64], [0.51, 0.68], [0.41, 0.71], [0.37, 0.73]]}
* validation (s0): V1 0.00e+00 m, V2 0.00e+00 m, V3 imu vs ext >0.1 m {'2s': 0.18102508178844057, '6s': 0.757360959651036}, pose vs ext >0.1 m {'2s': 0.16739367502726282, '6s': 0.7281897491821155}, pose vs ext max abs 15.883 m.

## step30000_unc

### S2 / compliant -- n 4159 windows / 138 ep

ADE means (s0): os 0.2981, ha0_ext 0.2807, ha0_ext_pose 0.2858, ha0_imu 0.2823, ha0_imu_ax 0.3875, refcv7_50400 0.2771

| read | seed 0 | seed 1 |
|---|---|---|
| **E1 AS REGISTERED: os - ha0_ext (bar of record)** | +0.0174 [+0.0014, +0.0349] **SEP** | +0.0160 [+0.0002, +0.0329] **SEP** |
| os - ha0_ext_pose (residual vs its own prior; admissible ref) | +0.0123 [-0.0043, +0.0299] | +0.0109 [-0.0056, +0.0283] |
| os - ha0_imu (secondary admissible ref) | +0.0159 [-0.0002, +0.0335] | +0.0144 [-0.0017, +0.0315] |
| os - ha0_imu_ax (a0 = measured ax) | -0.0894 [-0.1436, -0.0399] **SEP** | -0.0908 [-0.1442, -0.0420] **SEP** |
| ha0_ext_pose - ha0_ext | +0.0051 [+0.0034, +0.0069] **SEP** | +0.0051 [+0.0034, +0.0069] **SEP** |
| ha0_imu - ha0_ext | +0.0016 [-0.0001, +0.0032] | +0.0016 [-0.0001, +0.0032] |
| RS_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.0289 [-0.0382, -0.0199] **SEP** | -0.0286 [-0.0381, -0.0197] **SEP** |
| RS_XF - ha0_ext_pose | -0.0340 [-0.0441, -0.0244] **SEP** | -0.0338 [-0.0439, -0.0243] **SEP** |
| RS_XF - ha0_imu | -0.0304 [-0.0398, -0.0214] **SEP** | -0.0302 [-0.0395, -0.0212] **SEP** |
| RS_XF - os (ADE) | -0.0463 [-0.0567, -0.0368] **SEP** | -0.0446 [-0.0549, -0.0352] **SEP** |
| GATED_RS_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.0333 [-0.0426, -0.0243] **SEP** | -0.0328 [-0.0422, -0.0239] **SEP** |
| GATED_RS_XF - ha0_ext_pose | -0.0384 [-0.0485, -0.0291] **SEP** | -0.0380 [-0.0479, -0.0284] **SEP** |
| GATED_RS_XF - ha0_imu | -0.0349 [-0.0442, -0.0260] **SEP** | -0.0344 [-0.0435, -0.0253] **SEP** |
| GATED_RS_XF - os (ADE) | -0.0507 [-0.0619, -0.0408] **SEP** | -0.0488 [-0.0598, -0.0389] **SEP** |
| RS_POS_XF - ha0_ext_pose | -0.0302 [-0.0399, -0.0204] **SEP** | -0.0303 [-0.0404, -0.0205] **SEP** |
| RS_POS_XF - os (ADE) | -0.0425 [-0.0524, -0.0336] **SEP** | -0.0411 [-0.0508, -0.0321] **SEP** |
| RS_IS - ha0_ext_pose | -0.0350 [-0.0450, -0.0257] **SEP** | -0.0345 [-0.0444, -0.0252] **SEP** |
| RS_IS - os (ADE) | -0.0473 [-0.0580, -0.0376] **SEP** | -0.0454 [-0.0559, -0.0357] **SEP** |
| GATED_RS_IS - ha0_ext_pose | -0.0406 [-0.0505, -0.0313] **SEP** | -0.0401 [-0.0499, -0.0308] **SEP** |
| GATED_RS_IS - os (ADE) | -0.0529 [-0.0644, -0.0427] **SEP** | -0.0510 [-0.0623, -0.0408] **SEP** |
| RS_WF - ha0_ext_pose | -0.0349 [-0.0449, -0.0256] **SEP** | -0.0345 [-0.0443, -0.0251] **SEP** |
| RS_WF - os (ADE) | -0.0472 [-0.0579, -0.0376] **SEP** | -0.0454 [-0.0559, -0.0357] **SEP** |
| GATED_RS_WF - ha0_ext_pose | -0.0402 [-0.0501, -0.0309] **SEP** | -0.0396 [-0.0493, -0.0302] **SEP** |
| GATED_RS_WF - os (ADE) | -0.0525 [-0.0639, -0.0423] **SEP** | -0.0504 [-0.0617, -0.0403] **SEP** |
| BLEND_imu_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.0298 [-0.0391, -0.0210] **SEP** | -0.0296 [-0.0390, -0.0207] **SEP** |
| BLEND_imu_XF - ha0_ext_pose | -0.0349 [-0.0452, -0.0255] **SEP** | -0.0347 [-0.0448, -0.0251] **SEP** |
| BLEND_imu_XF - ha0_imu | -0.0313 [-0.0407, -0.0223] **SEP** | -0.0311 [-0.0406, -0.0221] **SEP** |
| BLEND_imu_XF - os (ADE) | -0.0472 [-0.0577, -0.0376] **SEP** | -0.0456 [-0.0561, -0.0360] **SEP** |

| metric (family) | os - ha0_ext_pose s0 | RS_XF - refcv7 s0 | s1 | RS_XF - os s0 | s1 | GATED_RS_XF - refcv7 s0 | GATED_RS_XF - os s0 |
|---|---|---|---|---|---|---|---|
| ade_m (ADE) | +0.0123 [-0.0043, +0.0299] | -0.0253 [-0.0394, -0.0120] **SEP** | -0.0248 [-0.0390, -0.0114] **SEP** | -0.0463 [-0.0567, -0.0368] **SEP** | -0.0446 [-0.0549, -0.0352] **SEP** | -0.0298 [-0.0441, -0.0165] **SEP** | -0.0507 [-0.0619, -0.0408] **SEP** |
| fde_m (ADE) | +0.0176 [-0.0224, +0.0575] | -0.0496 [-0.0847, -0.0169] **SEP** | -0.0484 [-0.0835, -0.0164] **SEP** | -0.0965 [-0.1211, -0.0742] **SEP** | -0.0916 [-0.1161, -0.0686] **SEP** | -0.0526 [-0.0877, -0.0201] **SEP** | -0.0996 [-0.1241, -0.0768] **SEP** |
| LON_speed_mae_mps (LONGITUDINAL) | +0.0338 [+0.0190, +0.0500] **SEP** | -0.0160 [-0.0322, -0.0016] **SEP** | -0.0162 [-0.0321, -0.0017] **SEP** | -0.0416 [-0.0536, -0.0312] **SEP** | -0.0396 [-0.0515, -0.0288] **SEP** | -0.0152 [-0.0319, -0.0001] **SEP** | -0.0408 [-0.0524, -0.0301] **SEP** |
| LON_along_mae_m (LONGITUDINAL) | +0.0181 [+0.0043, +0.0336] **SEP** | -0.0088 [-0.0223, +0.0040] | -0.0085 [-0.0218, +0.0040] | -0.0313 [-0.0414, -0.0226] **SEP** | -0.0291 [-0.0390, -0.0202] **SEP** | -0.0136 [-0.0275, -0.0009] **SEP** | -0.0361 [-0.0466, -0.0270] **SEP** |
| LON_along_abs_final_m (LONGITUDINAL) | +0.0614 [+0.0307, +0.0956] **SEP** | -0.0303 [-0.0633, +0.0012] | -0.0295 [-0.0631, +0.0005] | -0.0796 [-0.1044, -0.0575] **SEP** | -0.0735 [-0.0984, -0.0512] **SEP** | -0.0338 [-0.0671, -0.0031] **SEP** | -0.0831 [-0.1082, -0.0608] **SEP** |
| LON_speed_within_0p5 (LONGITUDINAL) | -0.0429 [-0.0581, -0.0285] **SEP** | +0.0178 [+0.0058, +0.0312] **SEP** | +0.0176 [+0.0056, +0.0300] **SEP** | +0.0454 [+0.0337, +0.0571] **SEP** | +0.0410 [+0.0297, +0.0529] **SEP** | +0.0117 [-0.0011, +0.0256] | +0.0393 [+0.0278, +0.0510] **SEP** |
| LON_accel_mae_mps2 (LONGITUDINAL) | +0.0817 [+0.0587, +0.1045] **SEP** | -0.0528 [-0.0733, -0.0334] **SEP** | -0.0533 [-0.0736, -0.0347] **SEP** | -0.0744 [-0.0919, -0.0585] **SEP** | -0.0720 [-0.0894, -0.0561] **SEP** | -0.0249 [-0.0464, -0.0043] **SEP** | -0.0466 [-0.0607, -0.0335] **SEP** |
| LAT_heading_mae_deg (LATERAL) | -0.0549 [-0.1685, +0.0595] | -0.0689 [-0.1252, -0.0096] **SEP** | -0.0601 [-0.1114, -0.0063] **SEP** | -0.1467 [-0.1967, -0.1039] **SEP** | -0.1553 [-0.2054, -0.1119] **SEP** | -0.0837 [-0.1403, -0.0261] **SEP** | -0.1606 [-0.2117, -0.1159] **SEP** |
| LAT_curvature_mae_1pm_valid (LATERAL) | +0.0000 [-0.0006, +0.0006] | +0.0009 [+0.0002, +0.0019] **SEP** | +0.0009 [+0.0004, +0.0017] **SEP** | +0.0005 [-0.0002, +0.0014] | +0.0004 [-0.0002, +0.0010] | +0.0003 [-0.0002, +0.0008] | -0.0001 [-0.0005, +0.0005] |
| LAT_yaw_rate_mae_radps_valid (LATERAL) | -0.0039 [-0.0067, -0.0012] **SEP** | +0.0004 [-0.0009, +0.0018] | +0.0005 [-0.0007, +0.0018] | -0.0009 [-0.0019, +0.0001] | -0.0011 [-0.0020, -0.0003] **SEP** | -0.0003 [-0.0014, +0.0010] | -0.0015 [-0.0023, -0.0008] **SEP** |
| LAT_cross_mae_m (LATERAL) | -0.0037 [-0.0150, +0.0065] | -0.0214 [-0.0298, -0.0129] **SEP** | -0.0214 [-0.0293, -0.0136] **SEP** | -0.0259 [-0.0321, -0.0203] **SEP** | -0.0265 [-0.0327, -0.0208] **SEP** | -0.0218 [-0.0300, -0.0134] **SEP** | -0.0263 [-0.0326, -0.0207] **SEP** |
| LAT_cross_abs_final_m (LATERAL) | -0.0486 [-0.0784, -0.0222] **SEP** | -0.0247 [-0.0417, -0.0072] **SEP** | -0.0245 [-0.0402, -0.0081] **SEP** | -0.0353 [-0.0466, -0.0246] **SEP** | -0.0363 [-0.0474, -0.0256] **SEP** | -0.0254 [-0.0422, -0.0082] **SEP** | -0.0360 [-0.0473, -0.0252] **SEP** |
| TAC_traj_lat_correct (TACTICAL) | +0.0183 [+0.0072, +0.0292] **SEP** | -0.0043 [-0.0102, +0.0014] | -0.0065 [-0.0118, -0.0014] **SEP** | +0.0007 [-0.0036, +0.0054] | -0.0007 [-0.0051, +0.0039] | -0.0050 [-0.0111, +0.0010] | +0.0000 [-0.0043, +0.0045] |
| TAC_traj_lon_correct (TACTICAL) | -0.0337 [-0.0528, -0.0160] **SEP** | +0.0243 [+0.0098, +0.0406] **SEP** | +0.0166 [+0.0022, +0.0324] **SEP** | +0.0385 [+0.0223, +0.0558] **SEP** | +0.0322 [+0.0172, +0.0482] **SEP** | +0.0216 [+0.0073, +0.0371] **SEP** | +0.0358 [+0.0210, +0.0522] **SEP** |

* E1 registered verdict (os floor 0.0014): **FAILED**; os - ha0_ext_pose form: FAILED; os - ha0_imu form: FAILED.
* RS_XF (floor 0.0002): -ha0_ext form **PASS (single training seed; training-seed floor unmeasured)**; -ha0_ext_pose PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); NR rollup (headline rows) {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: LAT_curvature_mae_1pm_valid'}.
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): ['LAT_curvature_mae_1pm_valid (LATERAL)']; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_heading_mae_deg (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_heading_mae_deg (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* GATED_RS_XF (floor 0.0005): -ha0_ext form **PASS (single training seed; training-seed floor unmeasured)**; -ha0_ext_pose PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); NR rollup (headline rows) {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}.
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): none; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_heading_mae_deg (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_heading_mae_deg (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* fold-seed sensitivity (s0, 10 seeds, points): RS_XF -ha0_ext -0.0296..-0.0285, GATED_RS_XF -ha0_ext -0.0344..-0.0327; optimism (-pose) XF-IS +0.0010 / gated +0.0022, XF-WF +0.0009.
* in-sample RS gains g_k (s0) per instant [along, cross]: [[0.861, 0.203], [0.519, 0.344], [0.323, 0.521], [0.309, 0.658]]
* RS_XF fold-mean gains g_k (s0) per instant [along, cross]: [[0.86, 0.2], [0.52, 0.34], [0.32, 0.52], [0.31, 0.66]]; GATED_RS_XF fold-mean per v0 bin: {'bin0': [[0.08, 0.12], [0.05, 0.23], [0.13, 0.24], [0.2, 0.23]], 'bin1': [[0.26, 0.08], [0.12, 0.24], [0.23, 0.46], [0.38, 0.63]], 'bin2': [[0.87, 0.19], [0.53, 0.37], [0.31, 0.56], [0.3, 0.7]], 'bin3': [[1.16, 0.33], [0.84, 0.39], [0.53, 0.54], [0.02, 0.66]], 'bin4': [[1.19, 0.5], [0.81, 0.33], [0.87, 0.33], [0.78, 0.45]]}
* RS_XF fold-mean gains g_k (s1) per instant [along, cross]: [[0.86, 0.2], [0.52, 0.34], [0.33, 0.51], [0.31, 0.65]]; GATED_RS_XF fold-mean per v0 bin: {'bin0': [[0.08, 0.13], [0.05, 0.21], [0.12, 0.23], [0.21, 0.26]], 'bin1': [[0.24, 0.09], [0.12, 0.25], [0.22, 0.46], [0.38, 0.63]], 'bin2': [[0.87, 0.2], [0.56, 0.36], [0.36, 0.56], [0.33, 0.71]], 'bin3': [[1.18, 0.33], [0.84, 0.39], [0.49, 0.53], [0.02, 0.66]], 'bin4': [[1.2, 0.44], [0.81, 0.31], [0.87, 0.32], [0.78, 0.42]]}

### S2 / all -- n 4754 windows / 139 ep

ADE means (s0): os 0.3128, ha0_ext 0.2886, ha0_ext_pose 0.2941, ha0_imu 0.2900, ha0_imu_ax 0.4065, refcv7_50400 0.2784

| read | seed 0 | seed 1 |
|---|---|---|
| **E1 AS REGISTERED: os - ha0_ext (bar of record)** | +0.0242 [+0.0078, +0.0403] **SEP** | +0.0244 [+0.0082, +0.0408] **SEP** |
| os - ha0_ext_pose (residual vs its own prior; admissible ref) | +0.0187 [+0.0021, +0.0351] **SEP** | +0.0189 [+0.0022, +0.0352] **SEP** |
| os - ha0_imu (secondary admissible ref) | +0.0228 [+0.0064, +0.0386] **SEP** | +0.0230 [+0.0069, +0.0390] **SEP** |
| os - ha0_imu_ax (a0 = measured ax) | -0.0937 [-0.1470, -0.0458] **SEP** | -0.0935 [-0.1470, -0.0454] **SEP** |
| ha0_ext_pose - ha0_ext | +0.0055 [+0.0036, +0.0076] **SEP** | +0.0055 [+0.0036, +0.0076] **SEP** |
| ha0_imu - ha0_ext | +0.0014 [-0.0003, +0.0030] | +0.0014 [-0.0003, +0.0030] |
| RS_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.0292 [-0.0381, -0.0209] **SEP** | -0.0282 [-0.0371, -0.0200] **SEP** |
| RS_XF - ha0_ext_pose | -0.0347 [-0.0443, -0.0255] **SEP** | -0.0337 [-0.0433, -0.0247] **SEP** |
| RS_XF - ha0_imu | -0.0306 [-0.0395, -0.0223] **SEP** | -0.0295 [-0.0385, -0.0213] **SEP** |
| RS_XF - os (ADE) | -0.0534 [-0.0639, -0.0432] **SEP** | -0.0525 [-0.0629, -0.0421] **SEP** |
| GATED_RS_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.0340 [-0.0429, -0.0256] **SEP** | -0.0329 [-0.0418, -0.0245] **SEP** |
| GATED_RS_XF - ha0_ext_pose | -0.0395 [-0.0491, -0.0304] **SEP** | -0.0384 [-0.0480, -0.0293] **SEP** |
| GATED_RS_XF - ha0_imu | -0.0353 [-0.0444, -0.0272] **SEP** | -0.0343 [-0.0433, -0.0261] **SEP** |
| GATED_RS_XF - os (ADE) | -0.0582 [-0.0687, -0.0480] **SEP** | -0.0573 [-0.0680, -0.0468] **SEP** |
| RS_POS_XF - ha0_ext_pose | -0.0301 [-0.0398, -0.0210] **SEP** | -0.0294 [-0.0392, -0.0202] **SEP** |
| RS_POS_XF - os (ADE) | -0.0488 [-0.0584, -0.0391] **SEP** | -0.0483 [-0.0581, -0.0385] **SEP** |
| RS_IS - ha0_ext_pose | -0.0356 [-0.0453, -0.0265] **SEP** | -0.0346 [-0.0443, -0.0256] **SEP** |
| RS_IS - os (ADE) | -0.0543 [-0.0648, -0.0440] **SEP** | -0.0535 [-0.0641, -0.0430] **SEP** |
| GATED_RS_IS - ha0_ext_pose | -0.0408 [-0.0506, -0.0317] **SEP** | -0.0398 [-0.0494, -0.0306] **SEP** |
| GATED_RS_IS - os (ADE) | -0.0595 [-0.0702, -0.0492] **SEP** | -0.0587 [-0.0693, -0.0481] **SEP** |
| RS_WF - ha0_ext_pose | -0.0355 [-0.0452, -0.0264] **SEP** | -0.0345 [-0.0442, -0.0255] **SEP** |
| RS_WF - os (ADE) | -0.0542 [-0.0647, -0.0439] **SEP** | -0.0534 [-0.0639, -0.0429] **SEP** |
| GATED_RS_WF - ha0_ext_pose | -0.0401 [-0.0499, -0.0311] **SEP** | -0.0392 [-0.0488, -0.0301] **SEP** |
| GATED_RS_WF - os (ADE) | -0.0588 [-0.0694, -0.0486] **SEP** | -0.0581 [-0.0687, -0.0475] **SEP** |
| BLEND_imu_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.0304 [-0.0392, -0.0220] **SEP** | -0.0294 [-0.0383, -0.0211] **SEP** |
| BLEND_imu_XF - ha0_ext_pose | -0.0359 [-0.0455, -0.0265] **SEP** | -0.0349 [-0.0446, -0.0258] **SEP** |
| BLEND_imu_XF - ha0_imu | -0.0318 [-0.0406, -0.0233] **SEP** | -0.0308 [-0.0397, -0.0224] **SEP** |
| BLEND_imu_XF - os (ADE) | -0.0546 [-0.0652, -0.0442] **SEP** | -0.0537 [-0.0643, -0.0432] **SEP** |

| metric (family) | os - ha0_ext_pose s0 | RS_XF - refcv7 s0 | s1 | RS_XF - os s0 | s1 | GATED_RS_XF - refcv7 s0 | GATED_RS_XF - os s0 |
|---|---|---|---|---|---|---|---|
| ade_m (ADE) | +0.0187 [+0.0021, +0.0351] **SEP** | -0.0190 [-0.0324, -0.0062] **SEP** | -0.0186 [-0.0318, -0.0058] **SEP** | -0.0534 [-0.0639, -0.0432] **SEP** | -0.0525 [-0.0629, -0.0421] **SEP** | -0.0238 [-0.0376, -0.0110] **SEP** | -0.0582 [-0.0687, -0.0480] **SEP** |
| fde_m (ADE) | +0.0274 [-0.0120, +0.0673] | -0.0361 [-0.0689, -0.0055] **SEP** | -0.0352 [-0.0667, -0.0047] **SEP** | -0.1113 [-0.1354, -0.0876] **SEP** | -0.1088 [-0.1330, -0.0845] **SEP** | -0.0396 [-0.0723, -0.0093] **SEP** | -0.1147 [-0.1379, -0.0917] **SEP** |
| LON_speed_mae_mps (LONGITUDINAL) | +0.0422 [+0.0271, +0.0577] **SEP** | -0.0119 [-0.0263, +0.0010] | -0.0120 [-0.0264, +0.0009] | -0.0500 [-0.0618, -0.0390] **SEP** | -0.0492 [-0.0609, -0.0383] **SEP** | -0.0111 [-0.0264, +0.0023] | -0.0493 [-0.0610, -0.0383] **SEP** |
| LON_along_mae_m (LONGITUDINAL) | +0.0263 [+0.0123, +0.0413] **SEP** | -0.0040 [-0.0170, +0.0075] | -0.0040 [-0.0164, +0.0074] | -0.0390 [-0.0490, -0.0291] **SEP** | -0.0376 [-0.0477, -0.0279] **SEP** | -0.0095 [-0.0223, +0.0017] | -0.0444 [-0.0545, -0.0349] **SEP** |
| LON_along_abs_final_m (LONGITUDINAL) | +0.0766 [+0.0449, +0.1092] **SEP** | -0.0218 [-0.0528, +0.0064] | -0.0216 [-0.0520, +0.0070] | -0.0966 [-0.1208, -0.0738] **SEP** | -0.0932 [-0.1174, -0.0704] **SEP** | -0.0260 [-0.0570, +0.0008] | -0.1008 [-0.1250, -0.0777] **SEP** |
| LON_speed_within_0p5 (LONGITUDINAL) | -0.0505 [-0.0651, -0.0366] **SEP** | +0.0146 [+0.0037, +0.0265] **SEP** | +0.0166 [+0.0057, +0.0280] **SEP** | +0.0530 [+0.0416, +0.0645] **SEP** | +0.0506 [+0.0397, +0.0620] **SEP** | +0.0086 [-0.0026, +0.0209] | +0.0470 [+0.0358, +0.0581] **SEP** |
| LON_accel_mae_mps2 (LONGITUDINAL) | +0.0879 [+0.0663, +0.1101] **SEP** | -0.0487 [-0.0675, -0.0314] **SEP** | -0.0486 [-0.0670, -0.0313] **SEP** | -0.0832 [-0.1001, -0.0674] **SEP** | -0.0820 [-0.0988, -0.0658] **SEP** | -0.0230 [-0.0418, -0.0045] **SEP** | -0.0575 [-0.0702, -0.0450] **SEP** |
| LAT_heading_mae_deg (LATERAL) | -0.0709 [-0.1863, +0.0453] | -0.0496 [-0.1042, +0.0080] | -0.0393 [-0.0899, +0.0131] | -0.1521 [-0.2010, -0.1080] **SEP** | -0.1596 [-0.2103, -0.1142] **SEP** | -0.0598 [-0.1124, -0.0056] **SEP** | -0.1617 [-0.2095, -0.1189] **SEP** |
| LAT_curvature_mae_1pm_valid (LATERAL) | -0.0001 [-0.0006, +0.0005] | +0.0008 [+0.0002, +0.0016] **SEP** | +0.0009 [+0.0003, +0.0015] **SEP** | +0.0004 [-0.0002, +0.0012] | +0.0003 [-0.0002, +0.0009] | +0.0002 [-0.0001, +0.0006] | -0.0001 [-0.0005, +0.0003] |
| LAT_yaw_rate_mae_radps_valid (LATERAL) | -0.0044 [-0.0074, -0.0014] **SEP** | +0.0005 [-0.0006, +0.0018] | +0.0007 [-0.0004, +0.0019] | -0.0011 [-0.0020, -0.0002] **SEP** | -0.0012 [-0.0020, -0.0004] **SEP** | -0.0001 [-0.0011, +0.0010] | -0.0017 [-0.0023, -0.0010] **SEP** |
| LAT_cross_mae_m (LATERAL) | -0.0035 [-0.0149, +0.0078] | -0.0188 [-0.0271, -0.0115] **SEP** | -0.0183 [-0.0263, -0.0115] **SEP** | -0.0272 [-0.0336, -0.0215] **SEP** | -0.0280 [-0.0344, -0.0221] **SEP** | -0.0187 [-0.0270, -0.0113] **SEP** | -0.0272 [-0.0335, -0.0215] **SEP** |
| LAT_cross_abs_final_m (LATERAL) | -0.0513 [-0.0822, -0.0219] **SEP** | -0.0175 [-0.0351, -0.0004] **SEP** | -0.0166 [-0.0332, -0.0006] **SEP** | -0.0364 [-0.0480, -0.0253] **SEP** | -0.0376 [-0.0492, -0.0265] **SEP** | -0.0177 [-0.0351, -0.0009] **SEP** | -0.0366 [-0.0485, -0.0255] **SEP** |
| TAC_traj_lat_correct (TACTICAL) | +0.0172 [+0.0061, +0.0277] **SEP** | -0.0072 [-0.0126, -0.0017] **SEP** | -0.0078 [-0.0126, -0.0032] **SEP** | +0.0000 [-0.0040, +0.0042] | -0.0002 [-0.0042, +0.0040] | -0.0072 [-0.0124, -0.0021] **SEP** | +0.0000 [-0.0036, +0.0038] |
| TAC_traj_lon_correct (TACTICAL) | -0.0349 [-0.0537, -0.0177] **SEP** | +0.0210 [+0.0067, +0.0361] **SEP** | +0.0154 [+0.0021, +0.0293] **SEP** | +0.0427 [+0.0280, +0.0585] **SEP** | +0.0404 [+0.0263, +0.0555] **SEP** | +0.0181 [+0.0040, +0.0335] **SEP** | +0.0398 [+0.0261, +0.0555] **SEP** |

* E1 registered verdict (os floor 0.0002): **FAILED**; os - ha0_ext_pose form: FAILED; os - ha0_imu form: FAILED.
* RS_XF (floor 0.0011): -ha0_ext form **PASS (single training seed; training-seed floor unmeasured)**; -ha0_ext_pose PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); NR rollup (headline rows) {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: LAT_curvature_mae_1pm_valid'}.
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): ['LAT_curvature_mae_1pm_valid (LATERAL)', 'TAC_traj_lat_correct (TACTICAL)']; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_heading_mae_deg (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* GATED_RS_XF (floor 0.0011): -ha0_ext form **PASS (single training seed; training-seed floor unmeasured)**; -ha0_ext_pose PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); NR rollup (headline rows) {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'FAIL: LAT_curvature_mae_1pm_valid'}.
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): ['TAC_traj_lat_correct (TACTICAL)']; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)']
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_heading_mae_deg (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)', 'TAC_traj_lon_correct (TACTICAL)']
* fold-seed sensitivity (s0, 10 seeds, points): RS_XF -ha0_ext -0.0298..-0.0293, GATED_RS_XF -ha0_ext -0.0340..-0.0332; optimism (-pose) XF-IS +0.0009 / gated +0.0013, XF-WF +0.0008.
* in-sample RS gains g_k (s0) per instant [along, cross]: [[0.832, 0.197], [0.468, 0.344], [0.293, 0.517], [0.294, 0.657]]
* RS_XF fold-mean gains g_k (s0) per instant [along, cross]: [[0.83, 0.2], [0.47, 0.34], [0.29, 0.52], [0.29, 0.66]]; GATED_RS_XF fold-mean per v0 bin: {'bin0': [[0.09, 0.16], [0.06, 0.22], [0.11, 0.22], [0.2, 0.24]], 'bin1': [[0.3, 0.11], [0.13, 0.29], [0.24, 0.51], [0.38, 0.68]], 'bin2': [[0.69, 0.17], [0.3, 0.33], [0.22, 0.51], [0.25, 0.66]], 'bin3': [[1.16, 0.33], [0.84, 0.39], [0.52, 0.54], [0.03, 0.66]], 'bin4': [[1.18, 0.51], [0.79, 0.32], [0.8, 0.33], [0.68, 0.44]]}
* RS_XF fold-mean gains g_k (s1) per instant [along, cross]: [[0.83, 0.2], [0.46, 0.34], [0.29, 0.51], [0.29, 0.65]]; GATED_RS_XF fold-mean per v0 bin: {'bin0': [[0.09, 0.16], [0.05, 0.2], [0.1, 0.21], [0.19, 0.26]], 'bin1': [[0.3, 0.11], [0.12, 0.3], [0.22, 0.51], [0.38, 0.68]], 'bin2': [[0.7, 0.17], [0.3, 0.32], [0.22, 0.5], [0.26, 0.66]], 'bin3': [[1.18, 0.33], [0.84, 0.39], [0.48, 0.52], [0.03, 0.65]], 'bin4': [[1.2, 0.46], [0.79, 0.32], [0.82, 0.33], [0.66, 0.41]]}
* validation (s0): V1 0.00e+00 m, V2 0.00e+00 m, V3 imu vs ext >0.1 m {'2s': 0.1773243584350021}, pose vs ext >0.1 m {'2s': 0.16281026503996635}, pose vs ext max abs 1.588 m.

### S6 / compliant -- n 3251 windows / 138 ep

ADE means (s0): os 2.8070, ha0_ext 3.3810, ha0_ext_pose 3.4098, ha0_imu 3.2977, ha0_imu_ax 3.7217, refcv7_50400 2.5088

| read | seed 0 | seed 1 |
|---|---|---|
| **E1 AS REGISTERED: os - ha0_ext (bar of record)** | -0.5741 [-0.9035, -0.2681] **SEP** | -0.5851 [-0.9183, -0.2759] **SEP** |
| os - ha0_ext_pose (residual vs its own prior; admissible ref) | -0.6028 [-0.9347, -0.2914] **SEP** | -0.6138 [-0.9520, -0.2995] **SEP** |
| os - ha0_imu (secondary admissible ref) | -0.4907 [-0.8001, -0.2116] **SEP** | -0.5017 [-0.8148, -0.2136] **SEP** |
| os - ha0_imu_ax (a0 = measured ax) | -0.9147 [-1.3737, -0.4957] **SEP** | -0.9257 [-1.3790, -0.5004] **SEP** |
| ha0_ext_pose - ha0_ext | +0.0288 [+0.0134, +0.0440] **SEP** | +0.0288 [+0.0134, +0.0440] **SEP** |
| ha0_imu - ha0_ext | -0.0833 [-0.1402, -0.0383] **SEP** | -0.0833 [-0.1402, -0.0383] **SEP** |
| RS_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.8431 [-1.0855, -0.6161] **SEP** | -0.8444 [-1.0949, -0.6111] **SEP** |
| RS_XF - ha0_ext_pose | -0.8718 [-1.1246, -0.6370] **SEP** | -0.8732 [-1.1285, -0.6328] **SEP** |
| RS_XF - ha0_imu | -0.7597 [-0.9792, -0.5552] **SEP** | -0.7611 [-0.9874, -0.5500] **SEP** |
| RS_XF - os (ADE) | -0.2690 [-0.3804, -0.1547] **SEP** | -0.2594 [-0.3723, -0.1447] **SEP** |
| GATED_RS_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.9027 [-1.1541, -0.6791] **SEP** | -0.9077 [-1.1632, -0.6739] **SEP** |
| GATED_RS_XF - ha0_ext_pose | -0.9315 [-1.1848, -0.7011] **SEP** | -0.9365 [-1.1984, -0.6953] **SEP** |
| GATED_RS_XF - ha0_imu | -0.8194 [-1.0385, -0.6196] **SEP** | -0.8244 [-1.0496, -0.6181] **SEP** |
| GATED_RS_XF - os (ADE) | -0.3287 [-0.4618, -0.2040] **SEP** | -0.3227 [-0.4560, -0.1976] **SEP** |
| RS_POS_XF - ha0_ext_pose | -0.8019 [-1.0573, -0.5598] **SEP** | -0.8031 [-1.0653, -0.5523] **SEP** |
| RS_POS_XF - os (ADE) | -0.1991 [-0.2833, -0.1084] **SEP** | -0.1893 [-0.2731, -0.1022] **SEP** |
| RS_IS - ha0_ext_pose | -0.8786 [-1.1305, -0.6446] **SEP** | -0.8795 [-1.1357, -0.6394] **SEP** |
| RS_IS - os (ADE) | -0.2758 [-0.3853, -0.1620] **SEP** | -0.2657 [-0.3786, -0.1524] **SEP** |
| GATED_RS_IS - ha0_ext_pose | -0.9618 [-1.2130, -0.7308] **SEP** | -0.9653 [-1.2255, -0.7246] **SEP** |
| GATED_RS_IS - os (ADE) | -0.3590 [-0.4938, -0.2297] **SEP** | -0.3515 [-0.4869, -0.2209] **SEP** |
| RS_WF - ha0_ext_pose | -0.8771 [-1.1290, -0.6436] **SEP** | -0.8784 [-1.1347, -0.6390] **SEP** |
| RS_WF - os (ADE) | -0.2743 [-0.3845, -0.1599] **SEP** | -0.2646 [-0.3780, -0.1512] **SEP** |
| GATED_RS_WF - ha0_ext_pose | -0.9550 [-1.2057, -0.7239] **SEP** | -0.9603 [-1.2202, -0.7211] **SEP** |
| GATED_RS_WF - os (ADE) | -0.3522 [-0.4866, -0.2238] **SEP** | -0.3464 [-0.4811, -0.2168] **SEP** |
| BLEND_imu_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.8774 [-1.1274, -0.6455] **SEP** | -0.8770 [-1.1313, -0.6396] **SEP** |
| BLEND_imu_XF - ha0_ext_pose | -0.9062 [-1.1634, -0.6696] **SEP** | -0.9058 [-1.1665, -0.6615] **SEP** |
| BLEND_imu_XF - ha0_imu | -0.7941 [-1.0167, -0.5851] **SEP** | -0.7937 [-1.0194, -0.5805] **SEP** |
| BLEND_imu_XF - os (ADE) | -0.3033 [-0.4154, -0.1926] **SEP** | -0.2919 [-0.4053, -0.1785] **SEP** |

| metric (family) | os - ha0_ext_pose s0 | RS_XF - refcv7 s0 | s1 | RS_XF - os s0 | s1 | GATED_RS_XF - refcv7 s0 | GATED_RS_XF - os s0 |
|---|---|---|---|---|---|---|---|
| ade_m (ADE) | -0.6028 [-0.9347, -0.2914] **SEP** | +0.0292 [-0.1362, +0.1921] | +0.0357 [-0.1230, +0.1961] | -0.2690 [-0.3804, -0.1547] **SEP** | -0.2594 [-0.3723, -0.1447] **SEP** | -0.0304 [-0.1918, +0.1186] | -0.3287 [-0.4618, -0.2040] **SEP** |
| fde_m (ADE) | -1.8331 [-2.7651, -0.9730] **SEP** | +0.1755 [-0.2436, +0.5987] | +0.1819 [-0.2305, +0.6017] | -0.6239 [-0.9177, -0.3093] **SEP** | -0.6040 [-0.8968, -0.3017] **SEP** | +0.0100 [-0.3998, +0.4008] | -0.7895 [-1.1469, -0.4409] **SEP** |
| LON_speed_mae_mps (LONGITUDINAL) | -0.0191 [-0.1251, +0.0870] | +0.0069 [-0.0613, +0.0739] | +0.0086 [-0.0598, +0.0774] | -0.1178 [-0.1659, -0.0692] **SEP** | -0.1131 [-0.1605, -0.0647] **SEP** | -0.0118 [-0.0788, +0.0504] | -0.1365 [-0.1968, -0.0790] **SEP** |
| LON_along_mae_m (LONGITUDINAL) | -0.1136 [-0.3730, +0.1436] | -0.0028 [-0.1601, +0.1475] | +0.0071 [-0.1510, +0.1637] | -0.2740 [-0.3934, -0.1530] **SEP** | -0.2647 [-0.3823, -0.1431] **SEP** | -0.0690 [-0.2194, +0.0730] | -0.3402 [-0.4806, -0.2088] **SEP** |
| LON_along_abs_final_m (LONGITUDINAL) | -0.5399 [-1.2882, +0.1841] | +0.0463 [-0.3522, +0.4381] | +0.0668 [-0.3280, +0.4780] | -0.6578 [-0.9754, -0.3266] **SEP** | -0.6375 [-0.9526, -0.3069] **SEP** | -0.1293 [-0.5043, +0.2294] | -0.8334 [-1.2165, -0.4628] **SEP** |
| LON_speed_within_0p5 (LONGITUDINAL) | -0.0350 [-0.0589, -0.0113] **SEP** | -0.0062 [-0.0261, +0.0130] | -0.0053 [-0.0250, +0.0130] | +0.0367 [+0.0250, +0.0486] **SEP** | +0.0383 [+0.0262, +0.0502] **SEP** | -0.0031 [-0.0233, +0.0170] | +0.0397 [+0.0249, +0.0549] **SEP** |
| LON_accel_mae_mps2 (LONGITUDINAL) | -0.0023 [-0.0476, +0.0420] | +0.0172 [-0.0070, +0.0419] | +0.0158 [-0.0077, +0.0407] | -0.0463 [-0.0643, -0.0281] **SEP** | -0.0438 [-0.0613, -0.0261] **SEP** | +0.0158 [-0.0070, +0.0388] | -0.0477 [-0.0736, -0.0243] **SEP** |
| LAT_heading_mae_deg (LATERAL) | -2.1745 [-2.9980, -1.4254] **SEP** | +0.6271 [+0.3627, +0.9341] **SEP** | +0.6819 [+0.3754, +1.0510] **SEP** | +0.2377 [+0.0779, +0.4304] **SEP** | +0.2058 [+0.0490, +0.3905] **SEP** | +0.5557 [+0.2690, +0.8999] **SEP** | +0.1369 [-0.0551, +0.3887] |
| LAT_curvature_mae_1pm_valid (LATERAL) | -0.0016 [-0.0026, -0.0006] **SEP** | +0.0020 [+0.0012, +0.0030] **SEP** | +0.0020 [+0.0011, +0.0030] **SEP** | +0.0014 [+0.0007, +0.0024] **SEP** | +0.0013 [+0.0005, +0.0023] **SEP** | +0.0021 [+0.0006, +0.0046] **SEP** | +0.0006 [+0.0000, +0.0011] **SEP** |
| LAT_yaw_rate_mae_radps_valid (LATERAL) | -0.0165 [-0.0237, -0.0099] **SEP** | +0.0055 [+0.0030, +0.0084] **SEP** | +0.0051 [+0.0028, +0.0078] **SEP** | +0.0030 [+0.0011, +0.0053] **SEP** | +0.0028 [+0.0009, +0.0049] **SEP** | +0.0047 [+0.0021, +0.0078] **SEP** | +0.0015 [-0.0000, +0.0032] |
| LAT_cross_mae_m (LATERAL) | -0.7188 [-0.9411, -0.5043] **SEP** | +0.0528 [-0.0125, +0.1192] | +0.0401 [-0.0248, +0.1032] | -0.0257 [-0.0420, -0.0094] **SEP** | -0.0256 [-0.0396, -0.0115] **SEP** | +0.0461 [-0.0211, +0.1137] | -0.0324 [-0.0521, -0.0137] **SEP** |
| LAT_cross_abs_final_m (LATERAL) | -1.9327 [-2.5278, -1.3683] **SEP** | +0.1954 [+0.0254, +0.3602] **SEP** | +0.1564 [-0.0068, +0.3187] | -0.0284 [-0.0628, +0.0077] | -0.0281 [-0.0586, +0.0051] | +0.1695 [-0.0096, +0.3463] | -0.0542 [-0.1040, -0.0056] **SEP** |
| TAC_traj_lat_correct (TACTICAL) | +0.0892 [+0.0631, +0.1155] **SEP** | -0.0148 [-0.0308, +0.0000] | -0.0135 [-0.0302, +0.0022] | -0.0015 [-0.0104, +0.0075] | +0.0015 [-0.0065, +0.0104] | -0.0102 [-0.0255, +0.0046] | +0.0031 [-0.0067, +0.0133] |
| TAC_traj_lon_correct (TACTICAL) | +0.0206 [-0.0198, +0.0595] | -0.0289 [-0.0624, +0.0034] | -0.0268 [-0.0621, +0.0063] | +0.0074 [-0.0123, +0.0295] | +0.0114 [-0.0081, +0.0322] | -0.0212 [-0.0562, +0.0117] | +0.0151 [-0.0097, +0.0401] |

* E1 registered verdict (os floor 0.0110): **PASS (single training seed; training-seed floor unmeasured)**; os - ha0_ext_pose form: PASS (single training seed; training-seed floor unmeasured); os - ha0_imu form: PASS (single training seed; training-seed floor unmeasured).
* RS_XF (floor 0.0014): -ha0_ext form **PASS (single training seed; training-seed floor unmeasured)**; -ha0_ext_pose PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); NR rollup (headline rows) {'ADE': 'PASS'}.
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): ['LAT_heading_mae_deg (LATERAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)']; BETTER: none
  * vs refcv8 os, separated WORSE at both seeds: **['LAT_heading_mae_deg (LATERAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)']**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_cross_mae_m (LATERAL)']
* GATED_RS_XF (floor 0.0050): -ha0_ext form **PASS (single training seed; training-seed floor unmeasured)**; -ha0_ext_pose PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); NR rollup (headline rows) {'ADE': 'PASS'}.
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): ['LAT_heading_mae_deg (LATERAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)']; BETTER: none
  * vs refcv8 os, separated WORSE at both seeds: **['LAT_curvature_mae_1pm_valid (LATERAL)']**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_cross_mae_m (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)']
* fold-seed sensitivity (s0, 10 seeds, points): RS_XF -ha0_ext -0.8425..-0.8143, GATED_RS_XF -ha0_ext -0.9074..-0.8781; optimism (-pose) XF-IS +0.0068 / gated +0.0303, XF-WF +0.0053.
* in-sample RS gains g_k (s0) per instant [along, cross]: [[0.535, 0.327], [0.322, 0.651], [0.432, 0.836], [0.524, 0.9], [0.57, 0.92], [0.593, 0.928]]
* RS_XF fold-mean gains g_k (s0) per instant [along, cross]: [[0.54, 0.33], [0.32, 0.65], [0.43, 0.84], [0.52, 0.9], [0.57, 0.92], [0.59, 0.93]]; GATED_RS_XF fold-mean per v0 bin: {'bin0': [[0.08, 0.06], [0.29, 0.22], [0.4, 0.35], [0.48, 0.46], [0.52, 0.45], [0.55, 0.42]], 'bin1': [[0.13, 0.21], [0.38, 0.63], [0.63, 0.84], [0.76, 0.89], [0.8, 0.9], [0.8, 0.9]], 'bin2': [[0.54, 0.37], [0.3, 0.7], [0.37, 0.88], [0.43, 0.93], [0.46, 0.96], [0.5, 0.97]], 'bin3': [[0.84, 0.36], [0.0, 0.63], [0.02, 0.81], [0.09, 0.91], [0.14, 0.95], [0.17, 0.96]], 'bin4': [[0.8, 0.3], [0.81, 0.44], [0.33, 0.51], [0.21, 0.5], [0.19, 0.46], [0.19, 0.45]]}
* RS_XF fold-mean gains g_k (s1) per instant [along, cross]: [[0.54, 0.32], [0.33, 0.65], [0.44, 0.84], [0.53, 0.91], [0.57, 0.92], [0.6, 0.93]]; GATED_RS_XF fold-mean per v0 bin: {'bin0': [[0.07, 0.02], [0.29, 0.32], [0.4, 0.44], [0.46, 0.53], [0.52, 0.5], [0.56, 0.45]], 'bin1': [[0.13, 0.22], [0.39, 0.62], [0.66, 0.82], [0.78, 0.88], [0.81, 0.88], [0.81, 0.9]], 'bin2': [[0.57, 0.36], [0.34, 0.73], [0.4, 0.9], [0.44, 0.96], [0.48, 0.98], [0.51, 0.98]], 'bin3': [[0.85, 0.37], [0.0, 0.63], [0.0, 0.82], [0.07, 0.93], [0.12, 0.96], [0.16, 0.98]], 'bin4': [[0.79, 0.31], [0.8, 0.39], [0.32, 0.46], [0.2, 0.47], [0.18, 0.44], [0.19, 0.42]]}

### S6 / all -- n 3668 windows / 139 ep

ADE means (s0): os 2.9189, ha0_ext 3.5479, ha0_ext_pose 3.5764, ha0_imu 3.4480, ha0_imu_ax 3.8578, refcv7_50400 2.5675

| read | seed 0 | seed 1 |
|---|---|---|
| **E1 AS REGISTERED: os - ha0_ext (bar of record)** | -0.6289 [-0.9539, -0.3086] **SEP** | -0.6258 [-0.9564, -0.2987] **SEP** |
| os - ha0_ext_pose (residual vs its own prior; admissible ref) | -0.6575 [-0.9943, -0.3332] **SEP** | -0.6544 [-0.9962, -0.3200] **SEP** |
| os - ha0_imu (secondary admissible ref) | -0.5291 [-0.8176, -0.2399] **SEP** | -0.5260 [-0.8230, -0.2294] **SEP** |
| os - ha0_imu_ax (a0 = measured ax) | -0.9388 [-1.3837, -0.5450] **SEP** | -0.9357 [-1.3779, -0.5396] **SEP** |
| ha0_ext_pose - ha0_ext | +0.0286 [+0.0108, +0.0472] **SEP** | +0.0286 [+0.0108, +0.0472] **SEP** |
| ha0_imu - ha0_ext | -0.0998 [-0.1492, -0.0525] **SEP** | -0.0998 [-0.1492, -0.0525] **SEP** |
| RS_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.9202 [-1.1782, -0.6834] **SEP** | -0.9178 [-1.1749, -0.6751] **SEP** |
| RS_XF - ha0_ext_pose | -0.9488 [-1.2063, -0.7037] **SEP** | -0.9464 [-1.2099, -0.6971] **SEP** |
| RS_XF - ha0_imu | -0.8204 [-1.0392, -0.6156] **SEP** | -0.8180 [-1.0404, -0.6063] **SEP** |
| RS_XF - os (ADE) | -0.2913 [-0.3990, -0.1864] **SEP** | -0.2920 [-0.4033, -0.1848] **SEP** |
| GATED_RS_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.9892 [-1.2464, -0.7508] **SEP** | -0.9888 [-1.2524, -0.7469] **SEP** |
| GATED_RS_XF - ha0_ext_pose | -1.0178 [-1.2809, -0.7757] **SEP** | -1.0173 [-1.2834, -0.7712] **SEP** |
| GATED_RS_XF - ha0_imu | -0.8894 [-1.1082, -0.6833] **SEP** | -0.8889 [-1.1147, -0.6802] **SEP** |
| GATED_RS_XF - os (ADE) | -0.3602 [-0.4925, -0.2393] **SEP** | -0.3629 [-0.4972, -0.2398] **SEP** |
| RS_POS_XF - ha0_ext_pose | -0.8726 [-1.1327, -0.6179] **SEP** | -0.8675 [-1.1351, -0.6072] **SEP** |
| RS_POS_XF - os (ADE) | -0.2150 [-0.3019, -0.1350] **SEP** | -0.2131 [-0.3016, -0.1300] **SEP** |
| RS_IS - ha0_ext_pose | -0.9524 [-1.2091, -0.7068] **SEP** | -0.9487 [-1.2115, -0.6986] **SEP** |
| RS_IS - os (ADE) | -0.2949 [-0.4017, -0.1897] **SEP** | -0.2943 [-0.4050, -0.1868] **SEP** |
| GATED_RS_IS - ha0_ext_pose | -1.0357 [-1.2979, -0.7943] **SEP** | -1.0348 [-1.3011, -0.7891] **SEP** |
| GATED_RS_IS - os (ADE) | -0.3782 [-0.5135, -0.2557] **SEP** | -0.3804 [-0.5157, -0.2536] **SEP** |
| RS_WF - ha0_ext_pose | -0.9504 [-1.2071, -0.7059] **SEP** | -0.9457 [-1.2088, -0.6966] **SEP** |
| RS_WF - os (ADE) | -0.2929 [-0.4000, -0.1866] **SEP** | -0.2913 [-0.4021, -0.1833] **SEP** |
| GATED_RS_WF - ha0_ext_pose | -1.0265 [-1.2885, -0.7853] **SEP** | -1.0260 [-1.2915, -0.7806] **SEP** |
| GATED_RS_WF - os (ADE) | -0.3689 [-0.5046, -0.2467] **SEP** | -0.3715 [-0.5072, -0.2448] **SEP** |
| BLEND_imu_XF - ha0_ext (E1 form; lever, NOT a bar) | -0.9571 [-1.2207, -0.7105] **SEP** | -0.9537 [-1.2192, -0.7023] **SEP** |
| BLEND_imu_XF - ha0_ext_pose | -0.9857 [-1.2545, -0.7283] **SEP** | -0.9823 [-1.2504, -0.7244] **SEP** |
| BLEND_imu_XF - ha0_imu | -0.8572 [-1.0861, -0.6469] **SEP** | -0.8539 [-1.0828, -0.6413] **SEP** |
| BLEND_imu_XF - os (ADE) | -0.3281 [-0.4325, -0.2258] **SEP** | -0.3279 [-0.4369, -0.2249] **SEP** |

| metric (family) | os - ha0_ext_pose s0 | RS_XF - refcv7 s0 | s1 | RS_XF - os s0 | s1 | GATED_RS_XF - refcv7 s0 | GATED_RS_XF - os s0 |
|---|---|---|---|---|---|---|---|
| ade_m (ADE) | -0.6575 [-0.9943, -0.3332] **SEP** | +0.0602 [-0.0959, +0.2126] | +0.0618 [-0.0834, +0.2107] | -0.2913 [-0.3990, -0.1864] **SEP** | -0.2920 [-0.4033, -0.1848] **SEP** | -0.0088 [-0.1508, +0.1325] | -0.3602 [-0.4925, -0.2393] **SEP** |
| fde_m (ADE) | -2.0440 [-2.9574, -1.1292] **SEP** | +0.2359 [-0.1622, +0.6316] | +0.2303 [-0.1432, +0.6176] | -0.6594 [-0.9429, -0.3797] **SEP** | -0.6683 [-0.9596, -0.3903] **SEP** | +0.0473 [-0.3163, +0.4051] | -0.8480 [-1.2170, -0.5166] **SEP** |
| LON_speed_mae_mps (LONGITUDINAL) | -0.0255 [-0.1215, +0.0732] | +0.0136 [-0.0483, +0.0760] | +0.0147 [-0.0466, +0.0756] | -0.1237 [-0.1687, -0.0799] **SEP** | -0.1245 [-0.1699, -0.0804] **SEP** | -0.0113 [-0.0686, +0.0425] | -0.1485 [-0.2089, -0.0936] **SEP** |
| LON_along_mae_m (LONGITUDINAL) | -0.1384 [-0.3821, +0.1215] | +0.0189 [-0.1224, +0.1583] | +0.0284 [-0.1158, +0.1697] | -0.3006 [-0.4153, -0.1909] **SEP** | -0.3004 [-0.4158, -0.1870] **SEP** | -0.0565 [-0.1888, +0.0681] | -0.3760 [-0.5110, -0.2492] **SEP** |
| LON_along_abs_final_m (LONGITUDINAL) | -0.6672 [-1.3788, +0.0807] | +0.0946 [-0.2806, +0.4641] | +0.1152 [-0.2654, +0.4772] | -0.7006 [-0.9983, -0.4042] **SEP** | -0.7070 [-1.0064, -0.4142] **SEP** | -0.1110 [-0.4562, +0.2100] | -0.9062 [-1.2769, -0.5648] **SEP** |
| LON_speed_within_0p5 (LONGITUDINAL) | -0.0350 [-0.0571, -0.0127] **SEP** | -0.0095 [-0.0278, +0.0089] | -0.0070 [-0.0247, +0.0111] | +0.0374 [+0.0264, +0.0479] **SEP** | +0.0397 [+0.0285, +0.0513] **SEP** | -0.0039 [-0.0220, +0.0146] | +0.0430 [+0.0292, +0.0573] **SEP** |
| LON_accel_mae_mps2 (LONGITUDINAL) | -0.0112 [-0.0528, +0.0323] | +0.0169 [-0.0059, +0.0415] | +0.0151 [-0.0068, +0.0388] | -0.0468 [-0.0643, -0.0303] **SEP** | -0.0465 [-0.0640, -0.0300] **SEP** | +0.0143 [-0.0059, +0.0341] | -0.0494 [-0.0728, -0.0281] **SEP** |
| LAT_heading_mae_deg (LATERAL) | -2.3495 [-3.1927, -1.5339] **SEP** | +0.6691 [+0.4133, +0.9392] **SEP** | +0.6948 [+0.4221, +1.0166] **SEP** | +0.2111 [+0.0629, +0.3818] **SEP** | +0.2031 [+0.0547, +0.3703] **SEP** | +0.5518 [+0.3023, +0.8054] **SEP** | +0.0669 [-0.0568, +0.2213] |
| LAT_curvature_mae_1pm_valid (LATERAL) | -0.0017 [-0.0027, -0.0006] **SEP** | +0.0020 [+0.0013, +0.0030] **SEP** | +0.0020 [+0.0012, +0.0030] **SEP** | +0.0014 [+0.0007, +0.0023] **SEP** | +0.0013 [+0.0006, +0.0022] **SEP** | +0.0017 [+0.0006, +0.0035] **SEP** | +0.0004 [+0.0000, +0.0008] **SEP** |
| LAT_yaw_rate_mae_radps_valid (LATERAL) | -0.0176 [-0.0246, -0.0106] **SEP** | +0.0059 [+0.0036, +0.0085] **SEP** | +0.0054 [+0.0033, +0.0078] **SEP** | +0.0029 [+0.0011, +0.0051] **SEP** | +0.0029 [+0.0011, +0.0049] **SEP** | +0.0045 [+0.0025, +0.0067] **SEP** | +0.0012 [-0.0000, +0.0025] |
| LAT_cross_mae_m (LATERAL) | -0.7632 [-0.9846, -0.5496] **SEP** | +0.0687 [+0.0051, +0.1356] **SEP** | +0.0526 [-0.0109, +0.1158] | -0.0261 [-0.0411, -0.0121] **SEP** | -0.0265 [-0.0406, -0.0132] **SEP** | +0.0629 [-0.0002, +0.1293] | -0.0318 [-0.0497, -0.0149] **SEP** |
| LAT_cross_abs_final_m (LATERAL) | -2.0731 [-2.6703, -1.5052] **SEP** | +0.2236 [+0.0605, +0.3854] **SEP** | +0.1745 [+0.0144, +0.3364] **SEP** | -0.0240 [-0.0537, +0.0046] | -0.0253 [-0.0539, +0.0027] | +0.2061 [+0.0417, +0.3752] **SEP** | -0.0415 [-0.0868, +0.0025] |
| TAC_traj_lat_correct (TACTICAL) | +0.0886 [+0.0641, +0.1124] **SEP** | -0.0177 [-0.0330, -0.0033] **SEP** | -0.0142 [-0.0289, +0.0003] | -0.0003 [-0.0085, +0.0071] | +0.0033 [-0.0046, +0.0115] | -0.0158 [-0.0310, -0.0014] **SEP** | +0.0016 [-0.0068, +0.0101] |
| TAC_traj_lon_correct (TACTICAL) | +0.0251 [-0.0123, +0.0605] | -0.0204 [-0.0544, +0.0101] | -0.0174 [-0.0517, +0.0134] | +0.0065 [-0.0123, +0.0268] | +0.0109 [-0.0071, +0.0294] | -0.0093 [-0.0424, +0.0207] | +0.0177 [-0.0049, +0.0420] |

* E1 registered verdict (os floor 0.0031): **PASS (single training seed; training-seed floor unmeasured)**; os - ha0_ext_pose form: PASS (single training seed; training-seed floor unmeasured); os - ha0_imu form: PASS (single training seed; training-seed floor unmeasured).
* RS_XF (floor 0.0024): -ha0_ext form **PASS (single training seed; training-seed floor unmeasured)**; -ha0_ext_pose PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); NR rollup (headline rows) {'ADE': 'PASS'}.
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): ['LAT_heading_mae_deg (LATERAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)', 'LAT_cross_abs_final_m (LATERAL)']; BETTER: none
  * vs refcv8 os, separated WORSE at both seeds: **['LAT_heading_mae_deg (LATERAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)']**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_cross_mae_m (LATERAL)']
* GATED_RS_XF (floor 0.0004): -ha0_ext form **PASS (single training seed; training-seed floor unmeasured)**; -ha0_ext_pose PASS (single training seed; training-seed floor unmeasured); -os PASS (single training seed; training-seed floor unmeasured); NR rollup (headline rows) {'ADE': 'PASS'}.
  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): ['LAT_heading_mae_deg (LATERAL)', 'LAT_curvature_mae_1pm_valid (LATERAL)', 'LAT_yaw_rate_mae_radps_valid (LATERAL)']; BETTER: none
  * vs refcv8 os, separated WORSE at both seeds: **none**; BETTER: ['ade_m (ADE)', 'fde_m (ADE)', 'LON_speed_mae_mps (LONGITUDINAL)', 'LON_along_mae_m (LONGITUDINAL)', 'LON_along_abs_final_m (LONGITUDINAL)', 'LON_speed_within_0p5 (LONGITUDINAL)', 'LON_accel_mae_mps2 (LONGITUDINAL)', 'LAT_cross_mae_m (LATERAL)']
* fold-seed sensitivity (s0, 10 seeds, points): RS_XF -ha0_ext -0.9210..-0.9006, GATED_RS_XF -ha0_ext -0.9859..-0.9682; optimism (-pose) XF-IS +0.0036 / gated +0.0179, XF-WF +0.0016.
* in-sample RS gains g_k (s0) per instant [along, cross]: [[0.493, 0.32], [0.321, 0.645], [0.432, 0.832], [0.52, 0.898], [0.569, 0.919], [0.598, 0.932]]
* RS_XF fold-mean gains g_k (s0) per instant [along, cross]: [[0.49, 0.32], [0.32, 0.64], [0.43, 0.83], [0.52, 0.9], [0.57, 0.92], [0.6, 0.93]]; GATED_RS_XF fold-mean per v0 bin: {'bin0': [[0.11, 0.12], [0.31, 0.22], [0.44, 0.38], [0.5, 0.51], [0.55, 0.53], [0.59, 0.5]], 'bin1': [[0.14, 0.26], [0.4, 0.68], [0.64, 0.86], [0.76, 0.9], [0.8, 0.91], [0.81, 0.91]], 'bin2': [[0.36, 0.32], [0.27, 0.65], [0.36, 0.84], [0.42, 0.92], [0.46, 0.95], [0.51, 0.96]], 'bin3': [[0.84, 0.36], [0.01, 0.63], [0.02, 0.81], [0.1, 0.91], [0.14, 0.95], [0.18, 0.96]], 'bin4': [[0.79, 0.33], [0.76, 0.44], [0.31, 0.49], [0.17, 0.48], [0.17, 0.45], [0.2, 0.43]]}
* RS_XF fold-mean gains g_k (s1) per instant [along, cross]: [[0.49, 0.32], [0.32, 0.64], [0.43, 0.83], [0.52, 0.9], [0.56, 0.92], [0.59, 0.93]]; GATED_RS_XF fold-mean per v0 bin: {'bin0': [[0.09, 0.08], [0.3, 0.33], [0.44, 0.46], [0.5, 0.58], [0.54, 0.57], [0.59, 0.52]], 'bin1': [[0.14, 0.27], [0.4, 0.67], [0.66, 0.85], [0.77, 0.89], [0.81, 0.9], [0.81, 0.91]], 'bin2': [[0.36, 0.3], [0.28, 0.66], [0.36, 0.86], [0.41, 0.93], [0.46, 0.97], [0.5, 0.98]], 'bin3': [[0.85, 0.37], [0.0, 0.62], [0.01, 0.81], [0.08, 0.91], [0.13, 0.95], [0.17, 0.97]], 'bin4': [[0.8, 0.32], [0.74, 0.4], [0.27, 0.45], [0.14, 0.46], [0.17, 0.43], [0.2, 0.41]]}
* validation (s0): V1 0.00e+00 m, V2 0.00e+00 m, V3 imu vs ext >0.1 m {'2s': 0.18102508178844057, '6s': 0.757360959651036}, pose vs ext >0.1 m {'2s': 0.16739367502726282, '6s': 0.7281897491821155}, pose vs ext max abs 15.883 m.

