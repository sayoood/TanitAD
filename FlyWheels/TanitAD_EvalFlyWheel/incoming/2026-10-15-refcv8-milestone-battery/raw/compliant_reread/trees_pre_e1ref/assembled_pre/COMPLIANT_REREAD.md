# COMPLIANT_REREAD -- the refcv8 bars on CEILING-COMPLIANT windows (AMENDMENT A-R8-CLAMP-2), early reads of the banked milestones

**TIER:** T1 self-action OPEN LOOP on logged eval139 frames (not driving performance). **ESTIMATOR / QUESTION:** episode-cluster bootstrap over the 139 eval episodes, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run -- on the compliant set it is the episodes the set contains. Single training seed, two inference seeds (0, 1).
**STATUS:** EARLY READS under the PI's compliant-window rule (A-R8-CLAMP-2 section 2.5). They never replace a banked verdict; the bars are decided at the final checkpoint (step 48,852).
**Rule:** a window is compliant iff the logged ego's max realised speed over the window's valid 6-s future (`gt_max_speed_6s`) <= the fed ceiling (strict; unknown ceiling = +inf). One mask per tree, from logged data + the fed label only. **Capped** = the landed `cap_plan_to_limit` (de4e15b, brake 2.5 m/s^2) applied post-hoc to the banked emitted plan; **uncapped** = the banked emitted plan. Source: `code/compliant_reread_r8.py`, trees `raw/compliant_reread/trees/<tree>.json`.

## 0. Controls (rule section 3) -- run BEFORE any compliant number was written

| tree | C1 all-True mask = banked cells (uncapped vs PAIRED_READ, capped vs CLAMP_POSTHOC) | C1v all-window verdicts = banked | C2 rolled-35 mask changes the set + a headline cell | C3 mask sha equal across inference seeds + re-derived from panel decisions | ALL |
|---|---|---|---|---|---|
| step5000_unc | True | True | True | True (independent derivation OK: True) | **True** |
| step5000ema_unc | True | True | True | True (independent derivation OK: True) | **True** |
| step10500_unc | True | True | True | True (independent derivation OK: True) | **True** |
| step10500ema_unc | True | True | True | True (independent derivation OK: True) | **True** |
| step15000_unc | True | True | True | True (independent derivation OK: True) | **True** |
| step15000ema_unc | True | True | True | True (independent derivation OK: True) | **True** |
| step20000_unc | True | True | True | True (independent derivation OK: True) | **True** |
| step20000ema_unc | True | True | True | True (independent derivation OK: True) | **True** |
| step30000_unc | True | True | True | True (independent derivation OK: True) | **True** |
| step30000ema_unc | True | True | True | True (independent derivation OK: True) | **True** |

## 1. The compliant set per tree

| tree | seed | S2 compliant / total windows (episodes) | S6 compliant / total windows (episodes) | non-compliant rate, all S2 windows [episode-cluster CI] | capped plans that change on compliant windows | mask sha256 (first 16) |
|---|---|---|---|---|---|---|
| step5000_unc | 0 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 1 | `4ebc6b8604f064f6` |
| step5000_unc | 1 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 0 | `4ebc6b8604f064f6` |
| step5000ema_unc | 0 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 0 | `4ebc6b8604f064f6` |
| step5000ema_unc | 1 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 0 | `4ebc6b8604f064f6` |
| step10500_unc | 0 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 0 | `4ebc6b8604f064f6` |
| step10500_unc | 1 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 0 | `4ebc6b8604f064f6` |
| step10500ema_unc | 0 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 0 | `4ebc6b8604f064f6` |
| step10500ema_unc | 1 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 0 | `4ebc6b8604f064f6` |
| step15000_unc | 0 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 0 | `4ebc6b8604f064f6` |
| step15000_unc | 1 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 0 | `4ebc6b8604f064f6` |
| step15000ema_unc | 0 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 0 | `4ebc6b8604f064f6` |
| step15000ema_unc | 1 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 0 | `4ebc6b8604f064f6` |
| step20000_unc | 0 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 0 | `4ebc6b8604f064f6` |
| step20000_unc | 1 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 0 | `4ebc6b8604f064f6` |
| step20000ema_unc | 0 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 0 | `4ebc6b8604f064f6` |
| step20000ema_unc | 1 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 0 | `4ebc6b8604f064f6` |
| step30000_unc | 0 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 0 | `4ebc6b8604f064f6` |
| step30000_unc | 1 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 0 | `4ebc6b8604f064f6` |
| step30000ema_unc | 0 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 0 | `4ebc6b8604f064f6` |
| step30000ema_unc | 1 | 4159 / 4754 (138) | 3251 / 3668 (138) | 12.52 % [8.94, 16.26] | 0 | `4ebc6b8604f064f6` |

Distinct mask shas over every tree and seed read: **1** (one mask for every arm, tree and seed (C3)).

## 2. The bars: compliant set vs all windows, capped (primary) vs uncapped (beside)

Verdicts exactly as `pair_read_r8` words them: E1/E3 = `os - ha0_ext` ADE < 0 and separated at BOTH inference seeds, NOT PROVEN inside 2x the inference floor (the floor is read on the same set); NR = per family, refcv8 - refcv7-50,400 not separated worse at BOTH seeds. (NR reference side unchanged: `D-R8-NR-INSTRUMENT` stays open.)

| tree | arm | E1 compliant | E1 all-window | E3 compliant | E3 all-window |
|---|---|---|---|---|---|
| step5000_unc | capped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |
| step5000_unc | uncapped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |
| step5000ema_unc | capped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |
| step5000ema_unc | uncapped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |
| step10500_unc | capped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |
| step10500_unc | uncapped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |
| step10500ema_unc | capped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |
| step10500ema_unc | uncapped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |
| step15000_unc | capped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |
| step15000_unc | uncapped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |
| step15000ema_unc | capped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |
| step15000ema_unc | uncapped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |
| step20000_unc | capped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |
| step20000_unc | uncapped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |
| step20000ema_unc | capped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |
| step20000ema_unc | uncapped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |
| step30000_unc | capped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |
| step30000_unc | uncapped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |
| step30000ema_unc | capped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |
| step30000ema_unc | uncapped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) |

NR per family (PASS = not separated worse at both seeds; `FAIL: ...` lists the failing headline cells):

| tree | arm | LONGITUDINAL (compliant / all) | LATERAL (compliant / all) | ADE (compliant / all) |
|---|---|---|---|---|
| step5000_unc | capped | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m / FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid / FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:ade_m, S6:ade_m / FAIL: S2:ade_m, S6:ade_m |
| step5000_unc | uncapped | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m / FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid / FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:ade_m, S6:ade_m / FAIL: S2:ade_m, S6:ade_m |
| step5000ema_unc | capped | PASS / FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | PASS / PASS | PASS / FAIL: S2:ade_m |
| step5000ema_unc | uncapped | PASS / PASS | PASS / PASS | PASS / PASS |
| step10500_unc | capped | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m / FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid / FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:ade_m, S6:ade_m / FAIL: S2:ade_m, S6:ade_m |
| step10500_unc | uncapped | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m / FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid / FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:ade_m, S6:ade_m / FAIL: S2:ade_m, S6:ade_m |
| step10500ema_unc | capped | PASS / FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | PASS / FAIL: S2:LAT_heading_mae_deg | PASS / FAIL: S2:ade_m |
| step10500ema_unc | uncapped | PASS / PASS | PASS / FAIL: S2:LAT_heading_mae_deg | PASS / PASS |
| step15000_unc | capped | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m / FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid / FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:ade_m, S6:ade_m / FAIL: S2:ade_m, S6:ade_m |
| step15000_unc | uncapped | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m / FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid / FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:ade_m, S6:ade_m / FAIL: S2:ade_m, S6:ade_m |
| step15000ema_unc | capped | PASS / FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LAT_heading_mae_deg / FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | PASS / FAIL: S2:ade_m, S6:ade_m |
| step15000ema_unc | uncapped | PASS / PASS | FAIL: S2:LAT_heading_mae_deg / FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | PASS / PASS |
| step20000_unc | capped | PASS / FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LAT_heading_mae_deg, S2:LAT_yaw_rate_mae_radps_valid / FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:ade_m / FAIL: S2:ade_m, S6:ade_m |
| step20000_unc | uncapped | PASS / FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LAT_heading_mae_deg, S2:LAT_yaw_rate_mae_radps_valid / FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:ade_m / FAIL: S2:ade_m, S6:ade_m |
| step20000ema_unc | capped | PASS / FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | PASS / FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid | PASS / FAIL: S2:ade_m, S6:ade_m |
| step20000ema_unc | uncapped | PASS / PASS | PASS / FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid | PASS / PASS |
| step30000_unc | capped | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m / FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid / FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:ade_m, S6:ade_m / FAIL: S2:ade_m, S6:ade_m |
| step30000_unc | uncapped | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m / FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid / FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:ade_m, S6:ade_m / FAIL: S2:ade_m, S6:ade_m |
| step30000ema_unc | capped | PASS / FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | PASS / FAIL: S2:LAT_heading_mae_deg | PASS / FAIL: S2:ade_m, S6:ade_m |
| step30000ema_unc | uncapped | PASS / FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | PASS / FAIL: S2:LAT_heading_mae_deg | PASS / FAIL: S6:ade_m |

## 3. The E1 / E3 cells (`os - ha0_ext`, ADE; negative = better than the echo, B = CI upper < 0)

| tree | arm | seed | E1 compliant (n win / ep) | E1 all-window | E3 compliant (n win / ep) | E3 all-window |
|---|---|---|---|---|---|---|
| step5000_unc | capped | 0 | +0.0491 [+0.0273, +0.0718] (4159 / 138) | +0.1271 [+0.0881, +0.1705] | -0.6365 [-0.9194, -0.3714] **B** (3251 / 138) | -0.4639 [-0.7848, -0.1447] **B** |
| step5000_unc | capped | 1 | +0.0502 [+0.0279, +0.0737] (4159 / 138) | +0.1279 [+0.0894, +0.1718] | -0.6148 [-0.8948, -0.3493] **B** (3251 / 138) | -0.4382 [-0.7633, -0.1050] **B** |
| step5000_unc | uncapped | 0 | +0.0492 [+0.0275, +0.0720] (4159 / 138) | +0.0735 [+0.0509, +0.0987] | -0.6365 [-0.9194, -0.3714] **B** (3251 / 138) | -0.5563 [-0.8477, -0.2767] **B** |
| step5000_unc | uncapped | 1 | +0.0502 [+0.0279, +0.0737] (4159 / 138) | +0.0749 [+0.0511, +0.1001] | -0.6148 [-0.8948, -0.3493] **B** (3251 / 138) | -0.5314 [-0.8296, -0.2411] **B** |
| step5000ema_unc | capped | 0 | -0.0119 [-0.0269, +0.0037] (4159 / 138) | +0.0635 [+0.0280, +0.1045] | -0.9956 [-1.2920, -0.7328] **B** (3251 / 138) | -0.8743 [-1.2084, -0.5591] **B** |
| step5000ema_unc | capped | 1 | -0.0129 [-0.0272, +0.0020] (4159 / 138) | +0.0618 [+0.0263, +0.1028] | -0.9981 [-1.2975, -0.7411] **B** (3251 / 138) | -0.8797 [-1.2096, -0.5744] **B** |
| step5000ema_unc | uncapped | 0 | -0.0119 [-0.0269, +0.0037] (4159 / 138) | -0.0004 [-0.0161, +0.0159] | -0.9956 [-1.2920, -0.7328] **B** (3251 / 138) | -0.9876 [-1.2924, -0.7085] **B** |
| step5000ema_unc | uncapped | 1 | -0.0129 [-0.0272, +0.0020] (4159 / 138) | -0.0022 [-0.0171, +0.0134] | -0.9981 [-1.2975, -0.7411] **B** (3251 / 138) | -0.9922 [-1.2926, -0.7247] **B** |
| step10500_unc | capped | 0 | +0.0619 [+0.0450, +0.0786] (4159 / 138) | +0.1288 [+0.0923, +0.1695] | -0.6275 [-0.9337, -0.3463] **B** (3251 / 138) | -0.5352 [-0.8597, -0.2148] **B** |
| step10500_unc | capped | 1 | +0.0602 [+0.0434, +0.0774] (4159 / 138) | +0.1273 [+0.0906, +0.1677] | -0.6354 [-0.9396, -0.3521] **B** (3251 / 138) | -0.5356 [-0.8713, -0.2161] **B** |
| step10500_unc | uncapped | 0 | +0.0619 [+0.0450, +0.0786] (4159 / 138) | +0.0710 [+0.0533, +0.0891] | -0.6275 [-0.9337, -0.3463] **B** (3251 / 138) | -0.6449 [-0.9381, -0.3577] **B** |
| step10500_unc | uncapped | 1 | +0.0602 [+0.0434, +0.0774] (4159 / 138) | +0.0693 [+0.0514, +0.0877] | -0.6354 [-0.9396, -0.3521] **B** (3251 / 138) | -0.6447 [-0.9414, -0.3646] **B** |
| step10500ema_unc | capped | 0 | -0.0137 [-0.0273, +0.0003] (4159 / 138) | +0.0593 [+0.0239, +0.0996] | -0.9817 [-1.2813, -0.7238] **B** (3251 / 138) | -0.8814 [-1.2010, -0.5709] **B** |
| step10500ema_unc | capped | 1 | -0.0134 [-0.0271, +0.0011] (4159 / 138) | +0.0604 [+0.0245, +0.1010] | -0.9607 [-1.2597, -0.6978] **B** (3251 / 138) | -0.8518 [-1.1677, -0.5464] **B** |
| step10500ema_unc | uncapped | 0 | -0.0137 [-0.0273, +0.0003] (4159 / 138) | -0.0039 [-0.0184, +0.0105] | -0.9817 [-1.2813, -0.7238] **B** (3251 / 138) | -0.9957 [-1.2924, -0.7263] **B** |
| step10500ema_unc | uncapped | 1 | -0.0134 [-0.0271, +0.0011] (4159 / 138) | -0.0026 [-0.0168, +0.0121] | -0.9607 [-1.2597, -0.6978] **B** (3251 / 138) | -0.9652 [-1.2585, -0.6950] **B** |
| step15000_unc | capped | 0 | +0.0289 [+0.0125, +0.0458] (4159 / 138) | +0.1051 [+0.0683, +0.1462] | -0.6073 [-0.9278, -0.3011] **B** (3251 / 138) | -0.5141 [-0.8499, -0.1799] **B** |
| step15000_unc | capped | 1 | +0.0279 [+0.0121, +0.0442] (4159 / 138) | +0.1048 [+0.0680, +0.1448] | -0.6168 [-0.9335, -0.3089] **B** (3251 / 138) | -0.5208 [-0.8611, -0.1885] **B** |
| step15000_unc | uncapped | 0 | +0.0289 [+0.0125, +0.0458] (4159 / 138) | +0.0452 [+0.0275, +0.0625] | -0.6073 [-0.9278, -0.3011] **B** (3251 / 138) | -0.6430 [-0.9576, -0.3396] **B** |
| step15000_unc | uncapped | 1 | +0.0279 [+0.0121, +0.0442] (4159 / 138) | +0.0448 [+0.0276, +0.0614] | -0.6168 [-0.9335, -0.3089] **B** (3251 / 138) | -0.6473 [-0.9657, -0.3405] **B** |
| step15000ema_unc | capped | 0 | -0.0108 [-0.0248, +0.0049] (4159 / 138) | +0.0627 [+0.0264, +0.1042] | -0.8935 [-1.1955, -0.6150] **B** (3251 / 138) | -0.8020 [-1.1273, -0.4823] **B** |
| step15000ema_unc | capped | 1 | -0.0101 [-0.0241, +0.0051] (4159 / 138) | +0.0626 [+0.0263, +0.1040] | -0.8942 [-1.1985, -0.6137] **B** (3251 / 138) | -0.8052 [-1.1278, -0.4849] **B** |
| step15000ema_unc | uncapped | 0 | -0.0108 [-0.0248, +0.0049] (4159 / 138) | +0.0003 [-0.0150, +0.0163] | -0.8935 [-1.1955, -0.6150] **B** (3251 / 138) | -0.9159 [-1.2255, -0.6377] **B** |
| step15000ema_unc | uncapped | 1 | -0.0101 [-0.0241, +0.0051] (4159 / 138) | +0.0001 [-0.0148, +0.0161] | -0.8942 [-1.1985, -0.6137] **B** (3251 / 138) | -0.9219 [-1.2306, -0.6417] **B** |
| step20000_unc | capped | 0 | +0.0093 [-0.0061, +0.0250] (4159 / 138) | +0.0766 [+0.0426, +0.1156] | -0.7801 [-1.0870, -0.5060] **B** (3251 / 138) | -0.7199 [-1.0362, -0.4113] **B** |
| step20000_unc | capped | 1 | +0.0094 [-0.0062, +0.0253] (4159 / 138) | +0.0757 [+0.0414, +0.1144] | -0.7733 [-1.0690, -0.5023] **B** (3251 / 138) | -0.7213 [-1.0352, -0.4136] **B** |
| step20000_unc | uncapped | 0 | +0.0093 [-0.0061, +0.0250] (4159 / 138) | +0.0146 [+0.0003, +0.0293] | -0.7801 [-1.0870, -0.5060] **B** (3251 / 138) | -0.8371 [-1.1368, -0.5600] **B** |
| step20000_unc | uncapped | 1 | +0.0094 [-0.0062, +0.0253] (4159 / 138) | +0.0135 [-0.0006, +0.0283] | -0.7733 [-1.0690, -0.5023] **B** (3251 / 138) | -0.8342 [-1.1203, -0.5570] **B** |
| step20000ema_unc | capped | 0 | -0.0127 [-0.0265, +0.0021] (4159 / 138) | +0.0609 [+0.0251, +0.1013] | -0.8519 [-1.1654, -0.5641] **B** (3251 / 138) | -0.7702 [-1.1053, -0.4392] **B** |
| step20000ema_unc | capped | 1 | -0.0127 [-0.0263, +0.0022] (4159 / 138) | +0.0598 [+0.0239, +0.1000] | -0.8520 [-1.1691, -0.5612] **B** (3251 / 138) | -0.7662 [-1.1028, -0.4437] **B** |
| step20000ema_unc | uncapped | 0 | -0.0127 [-0.0265, +0.0021] (4159 / 138) | -0.0020 [-0.0168, +0.0131] | -0.8519 [-1.1654, -0.5641] **B** (3251 / 138) | -0.8957 [-1.2084, -0.6024] **B** |
| step20000ema_unc | uncapped | 1 | -0.0127 [-0.0263, +0.0022] (4159 / 138) | -0.0025 [-0.0169, +0.0125] | -0.8520 [-1.1691, -0.5612] **B** (3251 / 138) | -0.8878 [-1.1999, -0.5935] **B** |
| step30000_unc | capped | 0 | +0.0174 [+0.0014, +0.0349] (4159 / 138) | +0.0853 [+0.0502, +0.1266] | -0.5741 [-0.9035, -0.2681] **B** (3251 / 138) | -0.5102 [-0.8631, -0.1622] **B** |
| step30000_unc | capped | 1 | +0.0160 [+0.0002, +0.0329] (4159 / 138) | +0.0852 [+0.0499, +0.1266] | -0.5851 [-0.9183, -0.2759] **B** (3251 / 138) | -0.5107 [-0.8694, -0.1574] **B** |
| step30000_unc | uncapped | 0 | +0.0174 [+0.0014, +0.0349] (4159 / 138) | +0.0242 [+0.0078, +0.0403] | -0.5741 [-0.9035, -0.2681] **B** (3251 / 138) | -0.6289 [-0.9539, -0.3086] **B** |
| step30000_unc | uncapped | 1 | +0.0160 [+0.0002, +0.0329] (4159 / 138) | +0.0244 [+0.0082, +0.0408] | -0.5851 [-0.9183, -0.2759] **B** (3251 / 138) | -0.6258 [-0.9564, -0.2987] **B** |
| step30000ema_unc | capped | 0 | -0.0085 [-0.0229, +0.0065] (4159 / 138) | +0.0621 [+0.0267, +0.1022] | -0.7535 [-1.0832, -0.4506] **B** (3251 / 138) | -0.6871 [-1.0261, -0.3526] **B** |
| step30000ema_unc | capped | 1 | -0.0076 [-0.0216, +0.0073] (4159 / 138) | +0.0634 [+0.0279, +0.1039] | -0.7430 [-1.0699, -0.4363] **B** (3251 / 138) | -0.6719 [-1.0117, -0.3391] **B** |
| step30000ema_unc | uncapped | 0 | -0.0085 [-0.0229, +0.0065] (4159 / 138) | -0.0008 [-0.0149, +0.0136] | -0.7535 [-1.0832, -0.4506] **B** (3251 / 138) | -0.8146 [-1.1390, -0.5132] **B** |
| step30000ema_unc | uncapped | 1 | -0.0076 [-0.0216, +0.0073] (4159 / 138) | +0.0015 [-0.0131, +0.0163] | -0.7430 [-1.0699, -0.4363] **B** (3251 / 138) | -0.7902 [-1.1101, -0.4921] **B** |

## 4. The NR headline cells on the primary tree `step30000ema_unc` (refcv8 - refcv7-50,400, error metrics: W = CI lower > 0 = separated WORSE)

| surface | metric | seed | CAPPED compliant (n win / ep) | CAPPED all-window | UNCAPPED compliant | UNCAPPED all-window |
|---|---|---|---|---|---|---|
| S2 | ADE: ADE (mean over the surface's instants) (m) | 0 | -0.0050 [-0.0157, +0.0044] (4159 / 138) | +0.0723 [+0.0379, +0.1134] **W** | -0.0050 [-0.0157, +0.0044] | +0.0093 [-0.0016, +0.0197] |
| S2 | ADE: ADE (mean over the surface's instants) (m) | 1 | -0.0038 [-0.0144, +0.0062] (4159 / 138) | +0.0729 [+0.0383, +0.1133] **W** | -0.0038 [-0.0144, +0.0062] | +0.0110 [-0.0005, +0.0220] |
| S6 | ADE: ADE (mean over the surface's instants) (m) | 0 | +0.1187 [-0.0265, +0.2761] (3251 / 138) | +0.2933 [+0.1132, +0.4805] **W** | +0.1187 [-0.0265, +0.2761] | +0.1658 [+0.0268, +0.3153] **W** |
| S6 | ADE: ADE (mean over the surface's instants) (m) | 1 | +0.1372 [-0.0065, +0.2974] (3251 / 138) | +0.3077 [+0.1272, +0.4962] **W** | +0.1372 [-0.0065, +0.2974] | +0.1895 [+0.0464, +0.3391] **W** |
| S2 | LONGITUDINAL: speed MAE (m/s) | 0 | -0.0038 [-0.0154, +0.0068] (4159 / 138) | +0.0682 [+0.0353, +0.1074] **W** | -0.0038 [-0.0154, +0.0068] | +0.0108 [-0.0012, +0.0224] |
| S2 | LONGITUDINAL: speed MAE (m/s) | 1 | -0.0025 [-0.0140, +0.0083] (4159 / 138) | +0.0689 [+0.0360, +0.1079] **W** | -0.0025 [-0.0140, +0.0083] | +0.0127 [+0.0004, +0.0244] **W** |
| S2 | LONGITUDINAL: |along-track| at the final instant (m) | 0 | -0.0049 [-0.0302, +0.0175] (4159 / 138) | +0.1367 [+0.0708, +0.2158] **W** | -0.0049 [-0.0302, +0.0175] | +0.0241 [-0.0013, +0.0482] |
| S2 | LONGITUDINAL: |along-track| at the final instant (m) | 1 | -0.0014 [-0.0263, +0.0217] (4159 / 138) | +0.1385 [+0.0725, +0.2182] **W** | -0.0014 [-0.0263, +0.0217] | +0.0285 [+0.0024, +0.0532] **W** |
| S2 | LATERAL: heading MAE (deg) | 0 | +0.0050 [-0.0365, +0.0473] (3931 / 138) | +0.0270 [-0.0120, +0.0649] | +0.0050 [-0.0365, +0.0473] | +0.0270 [-0.0120, +0.0649] |
| S2 | LATERAL: heading MAE (deg) | 1 | +0.0230 [-0.0256, +0.0720] (3930 / 138) | +0.0441 [+0.0001, +0.0879] **W** | +0.0230 [-0.0256, +0.0720] | +0.0441 [+0.0001, +0.0879] **W** |
| S2 | LATERAL: curvature MAE (masked) (1/m) | 0 | -0.0001 [-0.0003, +0.0002] (3900 / 138) | +0.0000 [-0.0002, +0.0002] | -0.0001 [-0.0003, +0.0002] | +0.0000 [-0.0002, +0.0002] |
| S2 | LATERAL: curvature MAE (masked) (1/m) | 1 | +0.0001 [-0.0001, +0.0003] (3900 / 138) | +0.0002 [-0.0001, +0.0004] | +0.0001 [-0.0001, +0.0003] | +0.0002 [-0.0001, +0.0004] |
| S2 | LATERAL: yaw-rate MAE (masked) (rad/s) | 0 | -0.0003 [-0.0011, +0.0006] (3900 / 138) | +0.0002 [-0.0006, +0.0010] | -0.0003 [-0.0011, +0.0006] | +0.0002 [-0.0006, +0.0010] |
| S2 | LATERAL: yaw-rate MAE (masked) (rad/s) | 1 | +0.0001 [-0.0009, +0.0011] (3900 / 138) | +0.0005 [-0.0004, +0.0014] | +0.0001 [-0.0009, +0.0011] | +0.0005 [-0.0004, +0.0014] |

## 5. The COMPLIANCE row (LONGITUDINAL family) -- rate of windows whose emitted plan exceeds the fed ceiling (`planned_max_speed`, strict), residual excess, v0 already over

One refcv8 fed ceiling and one window set for every arm; refcv7-50,400 and the echo exist only as the panel's embedded arms, so the cross-arm table is on the panel grids (S2: 0.5-s waypoints 0-2 s; S6: 1-s waypoints 0-6 s). On the compliant set the human did not exceed the ceiling, so a plan above it is an R1 violation he did not commit: **that rate is the R1 metric**. `resid` = excess over the ceiling on the exceeding windows (median / p90 / max, m/s).

### step30000ema_unc, inference seed 0

| grid | arm | all windows: exceed rate [CI] (n) | resid med/p90/max | compliant set: exceed rate [CI] (n) | resid med/p90/max | v0 already over, all / compliant |
|---|---|---|---|---|---|---|
| S2_panel_grid | refcv8_capped | 5.17 % [3.05, 7.54] (246/4754) | 1.2017/3.4383/5.7479 | 0.00 % [0.00, 0.00] (0/4159) | - | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |
| S2_panel_grid | refcv8_uncapped | 5.30 % [3.15, 7.67] (252/4754) | 2.0524/4.2756/6.5197 | 0.00 % [0.00, 0.00] (0/4159) | - | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |
| S2_panel_grid | refcv7_50400 | 8.29 % [5.73, 11.04] (394/4754) | 1.0498/3.5274/6.6021 | 0.53 % [0.17, 0.99] (22/4159) | 0.0863/0.7498/1.0205 | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |
| S2_panel_grid | echo_ha0_ext | 8.73 % [6.15, 11.55] (415/4754) | 1.1862/3.7955/7.0681 | 0.43 % [0.16, 0.78] (18/4159) | 0.6925/1.9349/2.1681 | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |
| S6_panel_grid | refcv8_capped | 5.13 % [2.72, 7.78] (188/3668) | 0.8761/3.165/5.1229 | 0.00 % [0.00, 0.00] (0/3251) | - | 7.06 % [4.37, 10.07] (259) / 0.15 % [0.03, 0.31] (5) |
| S6_panel_grid | refcv8_uncapped | 5.43 % [3.00, 8.15] (199/3668) | 2.1565/4.2866/6.5111 | 0.00 % [0.00, 0.00] (0/3251) | - | 7.06 % [4.37, 10.07] (259) / 0.15 % [0.03, 0.31] (5) |
| S6_panel_grid | refcv7_50400 | 11.31 % [8.23, 14.74] (415/3668) | 1.0006/3.3686/6.5747 | 3.23 % [1.91, 4.77] (105/3251) | 0.5162/2.2437/3.3475 | 7.06 % [4.37, 10.07] (259) / 0.15 % [0.03, 0.31] (5) |
| S6_panel_grid | echo_ha0_ext | 15.62 % [12.35, 18.95] (573/3668) | 1.8057/5.2136/18.3793 | 7.26 % [5.50, 9.20] (236/3251) | 1.2922/4.9448/14.8866 | 7.06 % [4.37, 10.07] (259) / 0.15 % [0.03, 0.31] (5) |
| FULL_plan_0_6s_S2_windows | refcv8_capped | 5.20 % [3.05, 7.57] (247/4754) | 1.192/3.4318/5.7479 | 0.00 % [0.00, 0.00] (0/4159) | - | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |
| FULL_plan_0_6s_S2_windows | refcv8_uncapped | 5.30 % [3.15, 7.67] (252/4754) | 2.1458/4.4699/9.8806 | 0.00 % [0.00, 0.00] (0/4159) | - | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |

### step30000ema_unc, inference seed 1

| grid | arm | all windows: exceed rate [CI] (n) | resid med/p90/max | compliant set: exceed rate [CI] (n) | resid med/p90/max | v0 already over, all / compliant |
|---|---|---|---|---|---|---|
| S2_panel_grid | refcv8_capped | 5.17 % [3.05, 7.50] (246/4754) | 1.2017/3.4383/5.7479 | 0.00 % [0.00, 0.00] (0/4159) | - | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |
| S2_panel_grid | refcv8_uncapped | 5.30 % [3.13, 7.67] (252/4754) | 2.075/4.3184/6.6453 | 0.00 % [0.00, 0.00] (0/4159) | - | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |
| S2_panel_grid | refcv7_50400 | 8.31 % [5.79, 11.07] (395/4754) | 1.0635/3.5233/6.7071 | 0.43 % [0.14, 0.83] (18/4159) | 0.0795/0.4461/0.9997 | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |
| S2_panel_grid | echo_ha0_ext | 8.73 % [6.15, 11.55] (415/4754) | 1.1862/3.7955/7.0681 | 0.43 % [0.16, 0.78] (18/4159) | 0.6925/1.9349/2.1681 | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |
| S6_panel_grid | refcv8_capped | 5.13 % [2.70, 7.79] (188/3668) | 0.8761/3.165/5.1229 | 0.00 % [0.00, 0.00] (0/3251) | - | 7.06 % [4.37, 10.07] (259) / 0.15 % [0.03, 0.31] (5) |
| S6_panel_grid | refcv8_uncapped | 5.45 % [3.02, 8.16] (200/3668) | 2.1114/4.2361/6.5842 | 0.00 % [0.00, 0.00] (0/3251) | - | 7.06 % [4.37, 10.07] (259) / 0.15 % [0.03, 0.31] (5) |
| S6_panel_grid | refcv7_50400 | 11.45 % [8.37, 14.76] (420/3668) | 0.9688/3.2732/6.631 | 3.29 % [1.99, 4.80] (107/3251) | 0.5108/2.0175/3.1693 | 7.06 % [4.37, 10.07] (259) / 0.15 % [0.03, 0.31] (5) |
| S6_panel_grid | echo_ha0_ext | 15.62 % [12.35, 18.95] (573/3668) | 1.8057/5.2136/18.3793 | 7.26 % [5.50, 9.20] (236/3251) | 1.2922/4.9448/14.8866 | 7.06 % [4.37, 10.07] (259) / 0.15 % [0.03, 0.31] (5) |
| FULL_plan_0_6s_S2_windows | refcv8_capped | 5.24 % [3.09, 7.58] (249/4754) | 1.1881/3.4189/5.7479 | 0.00 % [0.00, 0.00] (0/4159) | - | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |
| FULL_plan_0_6s_S2_windows | refcv8_uncapped | 5.30 % [3.13, 7.67] (252/4754) | 2.1323/4.4656/9.3177 | 0.00 % [0.00, 0.00] (0/4159) | - | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |

Compliance rate of the CAPPED refcv8 arm (the R1 metric) on the compliant set, every tree, S2 panel grid, strict > 0 / > 0.01 m/s (the second excludes fp32 dust of a plan capped AT the ceiling):

| tree | seed | capped exceed (strict) | capped exceed (>0.01 m/s) | uncapped exceed (strict) | refcv7-50,400 exceed | echo ha0_ext exceed |
|---|---|---|---|---|---|---|
| step5000_unc | 0 | 0.02 % [0.00, 0.07] (1/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.02 % [0.00, 0.07] (1/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step5000_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step5000ema_unc | 0 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step5000ema_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step10500_unc | 0 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step10500_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step10500ema_unc | 0 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step10500ema_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step15000_unc | 0 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step15000_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step15000ema_unc | 0 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step15000ema_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step20000_unc | 0 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step20000_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step20000ema_unc | 0 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step20000ema_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step30000_unc | 0 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step30000_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step30000ema_unc | 0 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step30000ema_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |

## 6. Strict-cap (brake 0) sensitivity, `step30000ema_unc` (rule section 2.4; no verdict)

Cross-check (the all-True strict read equals CLAMP_POSTHOC's SENSITIVITY_brake.json cell by cell): **{'all_equal': True, 'n_cells': 4}**.

| set | E1 verdict | E1 seed 0 | E1 seed 1 | E3 verdict | landed 2.5 brake, compliant (E1 / E3) |
|---|---|---|---|---|---|
| compliant | FAILED | -0.0085 [-0.0229, +0.0065] | -0.0076 [-0.0216, +0.0073] | PASS (single training seed; training-seed floor unmeasured) | FAILED / PASS (single training seed; training-seed floor unmeasured) |
| all windows | FAILED | +0.1137 [+0.0488, +0.1878] | +0.1149 [+0.0496, +0.1901] | PASS (single training seed; training-seed floor unmeasured) | FAILED / PASS (single training seed; training-seed floor unmeasured) |

Strict-capped plan, seed 0, compliant set, S2 panel grid: exceed (strict) 0.00 % [0.00, 0.00] (0/4159); >0.01 m/s: 0.

Strict-capped plan, seed 1, compliant set, S2 panel grid: exceed (strict) 0.00 % [0.00, 0.00] (0/4159); >0.01 m/s: 0.

## 7. What this is and is not

* TIER T1 self-action OPEN LOOP on logged eval139 frames: a plan scored against the logged human, **not driving performance**.
* The estimator answers 'another draw of EPISODES' (models fixed). It does NOT answer 'another training run' (`H-ESTIM-SEED-1`) or 'another inference run'; the inference floor is printed in the JSON beside every E1/E3 verdict.
* The compliant set removes the windows where the logged human exceeds the fed ceiling (the cap changes only those: `n_changed_plans_on_compliant_windows` above). It is a property of the labels, not a model-dependent filter (C3).
* Banked milestones are early reads; the final battery (step 48,852) decides. `step30000_unc` is read only when its PAIRED_READ is complete (`read --tree step30000_unc`).
* `step5000_corr` (corrected pose clock) is not read: its PAIRED_READ is E1/E3-only (`--no-ref`), the GT differs by the pose shift, and CLAMP_POSTHOC's own tree list excludes it.
