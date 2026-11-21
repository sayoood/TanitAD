# COMPLIANT_REREAD -- the refcv8 bars on CEILING-COMPLIANT windows (AMENDMENT A-R8-CLAMP-2), early reads of the banked milestones

**TIER:** T1 self-action OPEN LOOP on logged eval139 frames (not driving performance). **ESTIMATOR / QUESTION:** episode-cluster bootstrap over the 139 eval episodes, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run -- on the compliant set it is the episodes the set contains. Single training seed, two inference seeds (0, 1).
**STATUS:** EARLY READS under the PI's compliant-window rule (A-R8-CLAMP-2 section 2.5). They never replace a banked verdict; the bars are decided at the final checkpoint (step 48,852).
**Rule:** a window is compliant iff the logged ego's max realised speed over the window's valid 6-s future (`gt_max_speed_6s`) <= the fed ceiling (strict; unknown ceiling = +inf). One mask per tree, from logged data + the fed label only. **Capped** = the landed `cap_plan_to_limit` (de4e15b, brake 2.5 m/s^2) applied post-hoc to the banked emitted plan; **uncapped** = the banked emitted plan. Source: `code/compliant_reread_r8.py`, trees `raw/compliant_reread/trees/<tree>.json`.

## 0. Controls (rule section 3) -- run BEFORE any compliant number was written

| tree | C1 all-True mask = banked cells (uncapped vs PAIRED_READ, capped vs CLAMP_POSTHOC) | C1v all-window ha0_ext verdicts = banked | C2 rolled-35 mask changes the set + a headline cell | C3 mask sha equal across inference seeds + re-derived from panel decisions | C4 ha0_ext_pose arm valid (construction, model-free, 2026-09-26 + A0K0 known values) | ALL |
|---|---|---|---|---|---|---|
| step5000_unc | True | True | True | True (independent derivation OK: True) | True | **True** |
| step5000ema_unc | True | True | True | True (independent derivation OK: True) | True | **True** |
| step10500_unc | True | True | True | True (independent derivation OK: True) | True | **True** |
| step10500ema_unc | True | True | True | True (independent derivation OK: True) | True | **True** |
| step15000_unc | True | True | True | True (independent derivation OK: True) | True | **True** |
| step15000ema_unc | True | True | True | True (independent derivation OK: True) | True | **True** |
| step20000_unc | True | True | True | True (independent derivation OK: True) | True | **True** |
| step20000ema_unc | True | True | True | True (independent derivation OK: True) | True | **True** |
| step30000_unc | True | True | True | True (independent derivation OK: True) | True | **True** |
| step30000ema_unc | True | True | True | True (independent derivation OK: True) | True | **True** |

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

**D-R8-E1-REF (PI 2026-10-19, commit 7525bd7): BAR-R8-E1 / E3 are scored against ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19)** -- refcv8's own causal prior (measured v0, backward-difference acceleration, yaw rate -> kappa0). The registered read against `ha0_ext` (steer-channel kappa0) is kept BESIDE as `BAR-R8-E1_ha0_ext` / `BAR-R8-E3_ha0_ext`. The switch followed the step-30,000 reads; the pre-40k trees below are EARLY READS under it.

Verdicts exactly as `pair_read_r8` words them: E1/E3 = `os - reference` ADE < 0 and separated at BOTH inference seeds, NOT PROVEN inside 2x the inference floor (the floor is read on the same set); NR = per family, refcv8 - refcv7-50,400 not separated worse at BOTH seeds. (NR reference side unchanged: `D-R8-NR-INSTRUMENT` stays open.)

| tree | arm | E1 compliant (pose) | E1 all-window (pose) | E3 compliant (pose) | E3 all-window (pose) | E1 compliant (ha0_ext, beside) | E3 compliant (ha0_ext, beside) |
|---|---|---|---|---|---|---|---|
| step5000_unc | capped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |
| step5000_unc | uncapped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |
| step5000ema_unc | capped | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |
| step5000ema_unc | uncapped | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |
| step10500_unc | capped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |
| step10500_unc | uncapped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |
| step10500ema_unc | capped | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |
| step10500ema_unc | uncapped | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |
| step15000_unc | capped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |
| step15000_unc | uncapped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |
| step15000ema_unc | capped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |
| step15000ema_unc | uncapped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |
| step20000_unc | capped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |
| step20000_unc | uncapped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |
| step20000ema_unc | capped | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |
| step20000ema_unc | uncapped | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |
| step30000_unc | capped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |
| step30000_unc | uncapped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |
| step30000ema_unc | capped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |
| step30000ema_unc | uncapped | FAILED | FAILED | PASS (single training seed; training-seed floor unmeasured) | PASS (single training seed; training-seed floor unmeasured) | FAILED | PASS (single training seed; training-seed floor unmeasured) |

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

## 3. The E1 / E3 cells (`os - ha0_ext_pose` = the bar; `os - ha0_ext` beside; ADE; negative = better than the reference, B = CI upper < 0)

| tree | arm | seed | E1 compliant, pose (n win / ep) | E1 all-window, pose | E3 compliant, pose (n win / ep) | E3 all-window, pose | E1 compliant, ha0_ext | E3 compliant, ha0_ext |
|---|---|---|---|---|---|---|---|---|
| step5000_unc | capped | 0 | +0.0440 [+0.0224, +0.0667] (4159 / 138) | +0.1216 [+0.0833, +0.1653] | -0.6652 [-0.9481, -0.3964] **B** (3251 / 138) | -0.4925 [-0.8163, -0.1616] **B** | +0.0491 [+0.0273, +0.0718] | -0.6365 [-0.9194, -0.3714] **B** |
| step5000_unc | capped | 1 | +0.0451 [+0.0229, +0.0686] (4159 / 138) | +0.1224 [+0.0843, +0.1666] | -0.6436 [-0.9265, -0.3714] **B** (3251 / 138) | -0.4668 [-0.7925, -0.1237] **B** | +0.0502 [+0.0279, +0.0737] | -0.6148 [-0.8948, -0.3493] **B** |
| step5000_unc | uncapped | 0 | +0.0441 [+0.0227, +0.0669] (4159 / 138) | +0.0680 [+0.0450, +0.0932] | -0.6652 [-0.9481, -0.3964] **B** (3251 / 138) | -0.5849 [-0.8818, -0.2971] **B** | +0.0492 [+0.0275, +0.0720] | -0.6365 [-0.9194, -0.3714] **B** |
| step5000_unc | uncapped | 1 | +0.0451 [+0.0229, +0.0686] (4159 / 138) | +0.0694 [+0.0456, +0.0949] | -0.6436 [-0.9265, -0.3714] **B** (3251 / 138) | -0.5600 [-0.8627, -0.2644] **B** | +0.0502 [+0.0279, +0.0737] | -0.6148 [-0.8948, -0.3493] **B** |
| step5000ema_unc | capped | 0 | -0.0170 [-0.0327, -0.0015] **B** (4159 / 138) | +0.0580 [+0.0218, +0.0995] | -1.0243 [-1.3274, -0.7539] **B** (3251 / 138) | -0.9029 [-1.2457, -0.5818] **B** | -0.0119 [-0.0269, +0.0037] | -0.9956 [-1.2920, -0.7328] **B** |
| step5000ema_unc | capped | 1 | -0.0180 [-0.0331, -0.0030] **B** (4159 / 138) | +0.0563 [+0.0202, +0.0976] | -1.0269 [-1.3297, -0.7620] **B** (3251 / 138) | -0.9083 [-1.2433, -0.5992] **B** | -0.0129 [-0.0272, +0.0020] | -0.9981 [-1.2975, -0.7411] **B** |
| step5000ema_unc | uncapped | 0 | -0.0170 [-0.0327, -0.0015] **B** (4159 / 138) | -0.0059 [-0.0222, +0.0106] | -1.0243 [-1.3274, -0.7539] **B** (3251 / 138) | -1.0162 [-1.3276, -0.7299] **B** | -0.0119 [-0.0269, +0.0037] | -0.9956 [-1.2920, -0.7328] **B** |
| step5000ema_unc | uncapped | 1 | -0.0180 [-0.0331, -0.0030] **B** (4159 / 138) | -0.0077 [-0.0231, +0.0084] | -1.0269 [-1.3297, -0.7620] **B** (3251 / 138) | -1.0208 [-1.3218, -0.7478] **B** | -0.0129 [-0.0272, +0.0020] | -0.9981 [-1.2975, -0.7411] **B** |
| step10500_unc | capped | 0 | +0.0568 [+0.0401, +0.0736] (4159 / 138) | +0.1233 [+0.0862, +0.1642] | -0.6563 [-0.9641, -0.3667] **B** (3251 / 138) | -0.5638 [-0.8984, -0.2375] **B** | +0.0619 [+0.0450, +0.0786] | -0.6275 [-0.9337, -0.3463] **B** |
| step10500_unc | capped | 1 | +0.0551 [+0.0381, +0.0720] (4159 / 138) | +0.1218 [+0.0851, +0.1625] | -0.6641 [-0.9728, -0.3740] **B** (3251 / 138) | -0.5642 [-0.9000, -0.2395] **B** | +0.0602 [+0.0434, +0.0774] | -0.6354 [-0.9396, -0.3521] **B** |
| step10500_unc | uncapped | 0 | +0.0568 [+0.0401, +0.0736] (4159 / 138) | +0.0655 [+0.0478, +0.0834] | -0.6563 [-0.9641, -0.3667] **B** (3251 / 138) | -0.6735 [-0.9715, -0.3780] **B** | +0.0619 [+0.0450, +0.0786] | -0.6275 [-0.9337, -0.3463] **B** |
| step10500_unc | uncapped | 1 | +0.0551 [+0.0381, +0.0720] (4159 / 138) | +0.0637 [+0.0457, +0.0820] | -0.6641 [-0.9728, -0.3740] **B** (3251 / 138) | -0.6733 [-0.9755, -0.3804] **B** | +0.0602 [+0.0434, +0.0774] | -0.6354 [-0.9396, -0.3521] **B** |
| step10500ema_unc | capped | 0 | -0.0189 [-0.0331, -0.0049] **B** (4159 / 138) | +0.0538 [+0.0179, +0.0944] | -1.0105 [-1.3104, -0.7440] **B** (3251 / 138) | -0.9100 [-1.2359, -0.5973] **B** | -0.0137 [-0.0273, +0.0003] | -0.9817 [-1.2813, -0.7238] **B** |
| step10500ema_unc | capped | 1 | -0.0185 [-0.0324, -0.0040] **B** (4159 / 138) | +0.0549 [+0.0189, +0.0961] | -0.9895 [-1.2914, -0.7241] **B** (3251 / 138) | -0.8804 [-1.2065, -0.5704] **B** | -0.0134 [-0.0271, +0.0011] | -0.9607 [-1.2597, -0.6978] **B** |
| step10500ema_unc | uncapped | 0 | -0.0189 [-0.0331, -0.0049] **B** (4159 / 138) | -0.0095 [-0.0247, +0.0053] | -1.0105 [-1.3104, -0.7440] **B** (3251 / 138) | -1.0242 [-1.3272, -0.7489] **B** | -0.0137 [-0.0273, +0.0003] | -0.9817 [-1.2813, -0.7238] **B** |
| step10500ema_unc | uncapped | 1 | -0.0185 [-0.0324, -0.0040] **B** (4159 / 138) | -0.0081 [-0.0229, +0.0067] | -0.9895 [-1.2914, -0.7241] **B** (3251 / 138) | -0.9938 [-1.2952, -0.7131] **B** | -0.0134 [-0.0271, +0.0011] | -0.9607 [-1.2597, -0.6978] **B** |
| step15000_unc | capped | 0 | +0.0238 [+0.0066, +0.0408] (4159 / 138) | +0.0996 [+0.0626, +0.1404] | -0.6361 [-0.9590, -0.3238] **B** (3251 / 138) | -0.5427 [-0.8858, -0.2030] **B** | +0.0289 [+0.0125, +0.0458] | -0.6073 [-0.9278, -0.3011] **B** |
| step15000_unc | capped | 1 | +0.0228 [+0.0066, +0.0395] (4159 / 138) | +0.0993 [+0.0624, +0.1398] | -0.6455 [-0.9672, -0.3324] **B** (3251 / 138) | -0.5494 [-0.8945, -0.2153] **B** | +0.0279 [+0.0121, +0.0442] | -0.6168 [-0.9335, -0.3089] **B** |
| step15000_unc | uncapped | 0 | +0.0238 [+0.0066, +0.0408] (4159 / 138) | +0.0397 [+0.0213, +0.0572] | -0.6361 [-0.9590, -0.3238] **B** (3251 / 138) | -0.6716 [-0.9923, -0.3580] **B** | +0.0289 [+0.0125, +0.0458] | -0.6073 [-0.9278, -0.3011] **B** |
| step15000_unc | uncapped | 1 | +0.0228 [+0.0066, +0.0395] (4159 / 138) | +0.0393 [+0.0218, +0.0562] | -0.6455 [-0.9672, -0.3324] **B** (3251 / 138) | -0.6758 [-1.0050, -0.3599] **B** | +0.0279 [+0.0121, +0.0442] | -0.6168 [-0.9335, -0.3089] **B** |
| step15000ema_unc | capped | 0 | -0.0159 [-0.0309, -0.0001] **B** (4159 / 138) | +0.0572 [+0.0210, +0.0989] | -0.9222 [-1.2351, -0.6390] **B** (3251 / 138) | -0.8305 [-1.1642, -0.5029] **B** | -0.0108 [-0.0248, +0.0049] | -0.8935 [-1.1955, -0.6150] **B** |
| step15000ema_unc | capped | 1 | -0.0152 [-0.0296, +0.0002] (4159 / 138) | +0.0571 [+0.0205, +0.0989] | -0.9230 [-1.2366, -0.6370] **B** (3251 / 138) | -0.8338 [-1.1634, -0.5087] **B** | -0.0101 [-0.0241, +0.0051] | -0.8942 [-1.1985, -0.6137] **B** |
| step15000ema_unc | uncapped | 0 | -0.0159 [-0.0309, -0.0001] **B** (4159 / 138) | -0.0052 [-0.0210, +0.0111] | -0.9222 [-1.2351, -0.6390] **B** (3251 / 138) | -0.9445 [-1.2617, -0.6588] **B** | -0.0108 [-0.0248, +0.0049] | -0.8935 [-1.1955, -0.6150] **B** |
| step15000ema_unc | uncapped | 1 | -0.0152 [-0.0296, +0.0002] (4159 / 138) | -0.0054 [-0.0207, +0.0111] | -0.9230 [-1.2366, -0.6370] **B** (3251 / 138) | -0.9505 [-1.2676, -0.6631] **B** | -0.0101 [-0.0241, +0.0051] | -0.8942 [-1.1985, -0.6137] **B** |
| step20000_unc | capped | 0 | +0.0042 [-0.0115, +0.0200] (4159 / 138) | +0.0711 [+0.0366, +0.1104] | -0.8089 [-1.1180, -0.5290] **B** (3251 / 138) | -0.7484 [-1.0716, -0.4326] **B** | +0.0093 [-0.0061, +0.0250] | -0.7801 [-1.0870, -0.5060] **B** |
| step20000_unc | capped | 1 | +0.0043 [-0.0117, +0.0200] (4159 / 138) | +0.0702 [+0.0357, +0.1099] | -0.8020 [-1.1015, -0.5256] **B** (3251 / 138) | -0.7499 [-1.0692, -0.4375] **B** | +0.0094 [-0.0062, +0.0253] | -0.7733 [-1.0690, -0.5023] **B** |
| step20000_unc | uncapped | 0 | +0.0042 [-0.0115, +0.0200] (4159 / 138) | +0.0091 [-0.0053, +0.0243] | -0.8089 [-1.1180, -0.5290] **B** (3251 / 138) | -0.8656 [-1.1669, -0.5805] **B** | +0.0093 [-0.0061, +0.0250] | -0.7801 [-1.0870, -0.5060] **B** |
| step20000_unc | uncapped | 1 | +0.0043 [-0.0117, +0.0200] (4159 / 138) | +0.0080 [-0.0062, +0.0231] | -0.8020 [-1.1015, -0.5256] **B** (3251 / 138) | -0.8628 [-1.1540, -0.5828] **B** | +0.0094 [-0.0062, +0.0253] | -0.7733 [-1.0690, -0.5023] **B** |
| step20000ema_unc | capped | 0 | -0.0179 [-0.0322, -0.0029] **B** (4159 / 138) | +0.0554 [+0.0193, +0.0958] | -0.8807 [-1.2036, -0.5862] **B** (3251 / 138) | -0.7988 [-1.1402, -0.4616] **B** | -0.0127 [-0.0265, +0.0021] | -0.8519 [-1.1654, -0.5641] **B** |
| step20000ema_unc | capped | 1 | -0.0179 [-0.0320, -0.0025] **B** (4159 / 138) | +0.0543 [+0.0179, +0.0948] | -0.8808 [-1.2011, -0.5867] **B** (3251 / 138) | -0.7948 [-1.1404, -0.4617] **B** | -0.0127 [-0.0263, +0.0022] | -0.8520 [-1.1691, -0.5612] **B** |
| step20000ema_unc | uncapped | 0 | -0.0179 [-0.0322, -0.0029] **B** (4159 / 138) | -0.0075 [-0.0229, +0.0076] | -0.8807 [-1.2036, -0.5862] **B** (3251 / 138) | -0.9243 [-1.2480, -0.6291] **B** | -0.0127 [-0.0265, +0.0021] | -0.8519 [-1.1654, -0.5641] **B** |
| step20000ema_unc | uncapped | 1 | -0.0179 [-0.0320, -0.0025] **B** (4159 / 138) | -0.0080 [-0.0228, +0.0076] | -0.8808 [-1.2011, -0.5867] **B** (3251 / 138) | -0.9164 [-1.2379, -0.6153] **B** | -0.0127 [-0.0263, +0.0022] | -0.8520 [-1.1691, -0.5612] **B** |
| step30000_unc | capped | 0 | +0.0123 [-0.0043, +0.0299] (4159 / 138) | +0.0798 [+0.0447, +0.1213] | -0.6028 [-0.9347, -0.2914] **B** (3251 / 138) | -0.5388 [-0.8937, -0.1831] **B** | +0.0174 [+0.0014, +0.0349] | -0.5741 [-0.9035, -0.2681] **B** |
| step30000_unc | capped | 1 | +0.0109 [-0.0056, +0.0283] (4159 / 138) | +0.0797 [+0.0443, +0.1211] | -0.6138 [-0.9520, -0.2995] **B** (3251 / 138) | -0.5392 [-0.9026, -0.1822] **B** | +0.0160 [+0.0002, +0.0329] | -0.5851 [-0.9183, -0.2759] **B** |
| step30000_unc | uncapped | 0 | +0.0123 [-0.0043, +0.0299] (4159 / 138) | +0.0187 [+0.0021, +0.0351] | -0.6028 [-0.9347, -0.2914] **B** (3251 / 138) | -0.6575 [-0.9943, -0.3332] **B** | +0.0174 [+0.0014, +0.0349] | -0.5741 [-0.9035, -0.2681] **B** |
| step30000_unc | uncapped | 1 | +0.0109 [-0.0056, +0.0283] (4159 / 138) | +0.0189 [+0.0022, +0.0352] | -0.6138 [-0.9520, -0.2995] **B** (3251 / 138) | -0.6544 [-0.9962, -0.3200] **B** | +0.0160 [+0.0002, +0.0329] | -0.5851 [-0.9183, -0.2759] **B** |
| step30000ema_unc | capped | 0 | -0.0136 [-0.0286, +0.0018] (4159 / 138) | +0.0566 [+0.0206, +0.0971] | -0.7823 [-1.1159, -0.4748] **B** (3251 / 138) | -0.7157 [-1.0638, -0.3735] **B** | -0.0085 [-0.0229, +0.0065] | -0.7535 [-1.0832, -0.4506] **B** |
| step30000ema_unc | capped | 1 | -0.0127 [-0.0272, +0.0024] (4159 / 138) | +0.0578 [+0.0220, +0.0986] | -0.7717 [-1.1024, -0.4582] **B** (3251 / 138) | -0.7005 [-1.0517, -0.3622] **B** | -0.0076 [-0.0216, +0.0073] | -0.7430 [-1.0699, -0.4363] **B** |
| step30000ema_unc | uncapped | 0 | -0.0136 [-0.0286, +0.0018] (4159 / 138) | -0.0064 [-0.0207, +0.0090] | -0.7823 [-1.1159, -0.4748] **B** (3251 / 138) | -0.8432 [-1.1730, -0.5332] **B** | -0.0085 [-0.0229, +0.0065] | -0.7535 [-1.0832, -0.4506] **B** |
| step30000ema_unc | uncapped | 1 | -0.0127 [-0.0272, +0.0024] (4159 / 138) | -0.0041 [-0.0189, +0.0110] | -0.7717 [-1.1024, -0.4582] **B** (3251 / 138) | -0.8187 [-1.1463, -0.5109] **B** | -0.0076 [-0.0216, +0.0073] | -0.7430 [-1.0699, -0.4363] **B** |

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
| S2_panel_grid | echo_ha0_ext_pose | 8.73 % [6.15, 11.55] (415/4754) | 1.1862/3.7956/7.0681 | 0.43 % [0.16, 0.78] (18/4159) | 0.6925/1.9332/2.1681 | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |
| S6_panel_grid | refcv8_capped | 5.13 % [2.72, 7.78] (188/3668) | 0.8761/3.165/5.1229 | 0.00 % [0.00, 0.00] (0/3251) | - | 7.06 % [4.37, 10.07] (259) / 0.15 % [0.03, 0.31] (5) |
| S6_panel_grid | refcv8_uncapped | 5.43 % [3.00, 8.15] (199/3668) | 2.1565/4.2866/6.5111 | 0.00 % [0.00, 0.00] (0/3251) | - | 7.06 % [4.37, 10.07] (259) / 0.15 % [0.03, 0.31] (5) |
| S6_panel_grid | refcv7_50400 | 11.31 % [8.23, 14.74] (415/3668) | 1.0006/3.3686/6.5747 | 3.23 % [1.91, 4.77] (105/3251) | 0.5162/2.2437/3.3475 | 7.06 % [4.37, 10.07] (259) / 0.15 % [0.03, 0.31] (5) |
| S6_panel_grid | echo_ha0_ext | 15.62 % [12.35, 18.95] (573/3668) | 1.8057/5.2136/18.3793 | 7.26 % [5.50, 9.20] (236/3251) | 1.2922/4.9448/14.8866 | 7.06 % [4.37, 10.07] (259) / 0.15 % [0.03, 0.31] (5) |
| S6_panel_grid | echo_ha0_ext_pose | 15.62 % [12.35, 18.95] (573/3668) | 1.8057/5.2107/18.38 | 7.26 % [5.50, 9.20] (236/3251) | 1.2923/4.9446/14.824 | 7.06 % [4.37, 10.07] (259) / 0.15 % [0.03, 0.31] (5) |
| FULL_plan_0_6s_S2_windows | refcv8_capped | 5.20 % [3.05, 7.57] (247/4754) | 1.192/3.4318/5.7479 | 0.00 % [0.00, 0.00] (0/4159) | - | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |
| FULL_plan_0_6s_S2_windows | refcv8_uncapped | 5.30 % [3.15, 7.67] (252/4754) | 2.1458/4.4699/9.8806 | 0.00 % [0.00, 0.00] (0/4159) | - | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |

### step30000ema_unc, inference seed 1

| grid | arm | all windows: exceed rate [CI] (n) | resid med/p90/max | compliant set: exceed rate [CI] (n) | resid med/p90/max | v0 already over, all / compliant |
|---|---|---|---|---|---|---|
| S2_panel_grid | refcv8_capped | 5.17 % [3.05, 7.50] (246/4754) | 1.2017/3.4383/5.7479 | 0.00 % [0.00, 0.00] (0/4159) | - | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |
| S2_panel_grid | refcv8_uncapped | 5.30 % [3.13, 7.67] (252/4754) | 2.075/4.3184/6.6453 | 0.00 % [0.00, 0.00] (0/4159) | - | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |
| S2_panel_grid | refcv7_50400 | 8.31 % [5.79, 11.07] (395/4754) | 1.0635/3.5233/6.7071 | 0.43 % [0.14, 0.83] (18/4159) | 0.0795/0.4461/0.9997 | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |
| S2_panel_grid | echo_ha0_ext | 8.73 % [6.15, 11.55] (415/4754) | 1.1862/3.7955/7.0681 | 0.43 % [0.16, 0.78] (18/4159) | 0.6925/1.9349/2.1681 | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |
| S2_panel_grid | echo_ha0_ext_pose | 8.73 % [6.15, 11.55] (415/4754) | 1.1862/3.7956/7.0681 | 0.43 % [0.16, 0.78] (18/4159) | 0.6925/1.9332/2.1681 | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |
| S6_panel_grid | refcv8_capped | 5.13 % [2.70, 7.79] (188/3668) | 0.8761/3.165/5.1229 | 0.00 % [0.00, 0.00] (0/3251) | - | 7.06 % [4.37, 10.07] (259) / 0.15 % [0.03, 0.31] (5) |
| S6_panel_grid | refcv8_uncapped | 5.45 % [3.02, 8.16] (200/3668) | 2.1114/4.2361/6.5842 | 0.00 % [0.00, 0.00] (0/3251) | - | 7.06 % [4.37, 10.07] (259) / 0.15 % [0.03, 0.31] (5) |
| S6_panel_grid | refcv7_50400 | 11.45 % [8.37, 14.76] (420/3668) | 0.9688/3.2732/6.631 | 3.29 % [1.99, 4.80] (107/3251) | 0.5108/2.0175/3.1693 | 7.06 % [4.37, 10.07] (259) / 0.15 % [0.03, 0.31] (5) |
| S6_panel_grid | echo_ha0_ext | 15.62 % [12.35, 18.95] (573/3668) | 1.8057/5.2136/18.3793 | 7.26 % [5.50, 9.20] (236/3251) | 1.2922/4.9448/14.8866 | 7.06 % [4.37, 10.07] (259) / 0.15 % [0.03, 0.31] (5) |
| S6_panel_grid | echo_ha0_ext_pose | 15.62 % [12.35, 18.95] (573/3668) | 1.8057/5.2107/18.38 | 7.26 % [5.50, 9.20] (236/3251) | 1.2923/4.9446/14.824 | 7.06 % [4.37, 10.07] (259) / 0.15 % [0.03, 0.31] (5) |
| FULL_plan_0_6s_S2_windows | refcv8_capped | 5.24 % [3.09, 7.58] (249/4754) | 1.1881/3.4189/5.7479 | 0.00 % [0.00, 0.00] (0/4159) | - | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |
| FULL_plan_0_6s_S2_windows | refcv8_uncapped | 5.30 % [3.13, 7.67] (252/4754) | 2.1323/4.4656/9.3177 | 0.00 % [0.00, 0.00] (0/4159) | - | 6.75 % [4.38, 9.31] (321) / 0.12 % [0.02, 0.24] (5) |

Compliance rate of the CAPPED refcv8 arm (the R1 metric) on the compliant set, every tree, S2 panel grid, strict > 0 / > 0.01 m/s (the second excludes fp32 dust of a plan capped AT the ceiling):

| tree | seed | capped exceed (strict) | capped exceed (>0.01 m/s) | uncapped exceed (strict) | refcv7-50,400 exceed | echo ha0_ext exceed | echo ha0_ext_pose exceed |
|---|---|---|---|---|---|---|---|
| step5000_unc | 0 | 0.02 % [0.00, 0.07] (1/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.02 % [0.00, 0.07] (1/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step5000_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step5000ema_unc | 0 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step5000ema_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step10500_unc | 0 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step10500_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step10500ema_unc | 0 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step10500ema_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step15000_unc | 0 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step15000_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step15000ema_unc | 0 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step15000ema_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step20000_unc | 0 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step20000_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step20000ema_unc | 0 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step20000ema_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step30000_unc | 0 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step30000_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step30000ema_unc | 0 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.53 % [0.17, 0.99] (22/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |
| step30000ema_unc | 1 | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.00 % [0.00, 0.00] (0/4159) | 0.43 % [0.14, 0.83] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) | 0.43 % [0.16, 0.78] (18/4159) |

## 6. Strict-cap (brake 0) sensitivity, `step30000ema_unc` (rule section 2.4; no verdict); E1/E3 reference: ha0_ext_pose (D-R8-E1-REF, PI 2026-10-19)

Cross-check (the all-True strict read equals CLAMP_POSTHOC's SENSITIVITY_brake.json cell by cell, on the ha0_ext cell): **{'all_equal': True, 'n_cells': 4}**.

| set | E1 verdict | E1 seed 0 | E1 seed 1 | E3 verdict | landed 2.5 brake, compliant (E1 / E3) |
|---|---|---|---|---|---|
| compliant | FAILED | -0.0136 [-0.0286, +0.0018] | -0.0127 [-0.0272, +0.0024] | PASS (single training seed; training-seed floor unmeasured) | FAILED / PASS (single training seed; training-seed floor unmeasured) |
| all windows | FAILED | +0.1082 [+0.0436, +0.1825] | +0.1094 [+0.0438, +0.1849] | PASS (single training seed; training-seed floor unmeasured) | FAILED / PASS (single training seed; training-seed floor unmeasured) |

Strict-capped plan, seed 0, compliant set, S2 panel grid: exceed (strict) 0.00 % [0.00, 0.00] (0/4159); >0.01 m/s: 0.

Strict-capped plan, seed 1, compliant set, S2 panel grid: exceed (strict) 0.00 % [0.00, 0.00] (0/4159); >0.01 m/s: 0.

## 7. What this is and is not

* TIER T1 self-action OPEN LOOP on logged eval139 frames: a plan scored against the logged human, **not driving performance**.
* The estimator answers 'another draw of EPISODES' (models fixed). It does NOT answer 'another training run' (`H-ESTIM-SEED-1`) or 'another inference run'; the inference floor is printed in the JSON beside every E1/E3 verdict.
* The compliant set removes the windows where the logged human exceeds the fed ceiling (the cap changes only those: `n_changed_plans_on_compliant_windows` above). It is a property of the labels, not a model-dependent filter (C3).
* Banked milestones are early reads; the final battery (step 48,852) decides. `step30000_unc` is read only when its PAIRED_READ is complete (`read --tree step30000_unc`).
* `step5000_corr` (corrected pose clock) is not read: its PAIRED_READ is E1/E3-only (`--no-ref`), the GT differs by the pose shift, and CLAMP_POSTHOC's own tree list excludes it.
