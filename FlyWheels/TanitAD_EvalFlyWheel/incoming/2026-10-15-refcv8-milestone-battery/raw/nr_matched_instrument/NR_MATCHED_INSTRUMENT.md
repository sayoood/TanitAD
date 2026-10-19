# BAR-R8-NR on the MATCHED instrument (refcv8 ceiling filter OFF), inference seed 0

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; estimator: episode-cluster bootstrap over the 139 eval episodes, models FIXED: answers 'another draw of EPISODES' only; NOT another training run, NOT another inference run; B 2000, bootstrap seed 0, 95 % percentile; inference seed **0 only**.
* instrument: ON = refcv8 `os` as banked (max-speed ceiling filter ACTING, as every refcv8 battery was rolled); OFF = the SAME window, same DDIM draw, ceiling filter OFF (`filter_off.traj`) = the instrument the banked refcv7-50,400 reference was rolled with (its filter was INERT: final selection == filter-off, re-verified in the probe).
* reference: `b_refcv7_50400_s0` (the banked refcv7-50,400 `os`, embedded in every panel). Rows are refcv8 minus refcv7; error metrics, negative = refcv8 better.
* NR rule (pair_read_r8.bar_NR_metric): a headline is FAIL iff the paired CI **lower bound > 0** (separated worse). The banked verdict uses BOTH inference seeds; the matched one can only use seed 0 (seed-1 refcv8 rolls are os-only: no `filter_off` arm). So three columns are shown: banked both-seed, banked seed-0-only (ON), matched seed-0-only (OFF). The instrument effect is **ON seed 0 vs OFF seed 0**.
* A separated CI from one-seed arms is necessary, not sufficient (H-ESTIM-SEED-1); the inference-seed floor is not re-estimated here. Single training seed; EARLY reads of a 48,852-step run.

## Controls (all mandatory; per tree)

| tree | (A) identity: `os`->`os` reproduces banked rows exactly (S2 / S6; A1 panel os, A2 extras nav_true) | (B) mispairing changes rows |
|---|---|---|
| step5000_unc | S2: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 ; S6: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 | B_ep 7/7 cells changed, S2 ADE +10.726 m (PASS); B_win 7/7 (PASS) |
| step5000ema_unc | S2: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 ; S6: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 | B_ep 7/7 cells changed, S2 ADE +10.755 m (PASS); B_win 7/7 (PASS) |
| step10500_unc | S2: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 ; S6: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 | B_ep 7/7 cells changed, S2 ADE +10.731 m (PASS); B_win 7/7 (PASS) |
| step10500ema_unc | S2: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 ; S6: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 | B_ep 7/7 cells changed, S2 ADE +10.763 m (PASS); B_win 7/7 (PASS) |
| step15000_unc | S2: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 ; S6: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 | B_ep 7/7 cells changed, S2 ADE +10.706 m (PASS); B_win 7/7 (PASS) |
| step15000ema_unc | S2: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 ; S6: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 | B_ep 7/7 cells changed, S2 ADE +10.760 m (PASS); B_win 7/7 (PASS) |
| step20000_unc | S2: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 ; S6: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 | B_ep 7/7 cells changed, S2 ADE +10.748 m (PASS); B_win 7/7 (PASS) |
| step20000ema_unc | S2: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 ; S6: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 | B_ep 7/7 cells changed, S2 ADE +10.758 m (PASS); B_win 7/7 (PASS) |
| step30000_unc | S2: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 ; S6: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 | B_ep 7/7 cells changed, S2 ADE +10.718 m (PASS); B_win 7/7 (PASS) |
| step30000ema_unc | S2: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 ; S6: A1 28 cells, 0 mismatch, max|mean diff| 0.0e+00; A2 28 cells, 0 mismatch, 0.0e+00 | B_ep 7/7 cells changed, S2 ADE +10.747 m (PASS); B_win 7/7 (PASS) |

## Per-family NR verdicts: banked (ceiling ON) vs matched (ceiling OFF)

| tree | family | banked, both seeds (ON) | seed 0 only, ON | seed 0 only, OFF (matched) | changes ON->OFF at seed 0? | MIXED: seed 0 OFF + seed 1 ON (not matched) |
|---|---|---|---|---|---|---|
| step5000_unc | ADE | FAIL: S2:ade_m, S6:ade_m | FAIL: S2:ade_m, S6:ade_m | FAIL: S2:ade_m, S6:ade_m | no | FAIL: S2:ade_m, S6:ade_m |
| step5000_unc | LONGITUDINAL | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | no | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m |
| step5000_unc | LATERAL | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | no | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid |
| step5000ema_unc | ADE | PASS | PASS | PASS | no | PASS |
| step5000ema_unc | LONGITUDINAL | PASS | PASS | PASS | no | PASS |
| step5000ema_unc | LATERAL | PASS | PASS | PASS | no | PASS |
| step10500_unc | ADE | FAIL: S2:ade_m, S6:ade_m | FAIL: S2:ade_m, S6:ade_m | FAIL: S2:ade_m, S6:ade_m | no | FAIL: S2:ade_m, S6:ade_m |
| step10500_unc | LONGITUDINAL | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | no | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m |
| step10500_unc | LATERAL | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | no | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid |
| step10500ema_unc | ADE | PASS | PASS | PASS | no | PASS |
| step10500ema_unc | LONGITUDINAL | PASS | PASS | PASS | no | PASS |
| step10500ema_unc | LATERAL | FAIL: S2:LAT_heading_mae_deg | PASS | PASS | no | FAIL: S2:LAT_heading_mae_deg |
| step15000_unc | ADE | FAIL: S2:ade_m, S6:ade_m | FAIL: S2:ade_m, S6:ade_m | FAIL: S2:ade_m, S6:ade_m | no | FAIL: S2:ade_m, S6:ade_m |
| step15000_unc | LONGITUDINAL | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | no | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m |
| step15000_unc | LATERAL | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | no | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid |
| step15000ema_unc | ADE | PASS | PASS | PASS | no | PASS |
| step15000ema_unc | LONGITUDINAL | PASS | PASS | PASS | no | PASS |
| step15000ema_unc | LATERAL | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:LAT_heading_mae_deg | PASS | **YES** | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid |
| step20000_unc | ADE | FAIL: S2:ade_m, S6:ade_m | FAIL: S2:ade_m, S6:ade_m | FAIL: S2:ade_m, S6:ade_m | no | FAIL: S2:ade_m, S6:ade_m |
| step20000_unc | LONGITUDINAL | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | no | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m |
| step20000_unc | LATERAL | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:LAT_heading_mae_deg, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:LAT_heading_mae_deg | **YES** | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid |
| step20000ema_unc | ADE | PASS | PASS | PASS | no | PASS |
| step20000ema_unc | LONGITUDINAL | PASS | PASS | PASS | no | PASS |
| step20000ema_unc | LATERAL | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid | FAIL: S2:LAT_heading_mae_deg | PASS | **YES** | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid |
| step30000_unc | ADE | FAIL: S2:ade_m, S6:ade_m | FAIL: S2:ade_m, S6:ade_m | FAIL: S2:ade_m, S6:ade_m | no | FAIL: S2:ade_m, S6:ade_m |
| step30000_unc | LONGITUDINAL | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | no | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m |
| step30000_unc | LATERAL | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid | no | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid |
| step30000ema_unc | ADE | FAIL: S6:ade_m | FAIL: S6:ade_m | FAIL: S6:ade_m | no | FAIL: S6:ade_m |
| step30000ema_unc | LONGITUDINAL | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m | PASS | PASS | no | FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m |
| step30000ema_unc | LATERAL | FAIL: S2:LAT_heading_mae_deg | PASS | PASS | no | FAIL: S2:LAT_heading_mae_deg |

TACTICAL: `pair_read_r8.METRICS` carries NO NR headline for the TACTICAL family (TAC_* rows have an empty headline-surface tuple), so BAR-R8-NR has no TACTICAL verdict under either instrument (NOT EVALUABLE as coded, not a pass); the TAC rows are shown below.

### Verdict changes at seed 0 (ON -> OFF)

| tree | family | seed 0 ON | seed 0 OFF | banked both-seed |
|---|---|---|---|---|
| step15000ema_unc | LATERAL | FAIL: S2:LAT_heading_mae_deg | PASS | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid |
| step20000_unc | LATERAL | FAIL: S2:LAT_heading_mae_deg, S2:LAT_yaw_rate_mae_radps_valid | FAIL: S2:LAT_heading_mae_deg | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid, S2:LAT_yaw_rate_mae_radps_valid |
| step20000ema_unc | LATERAL | FAIL: S2:LAT_heading_mae_deg | PASS | FAIL: S2:LAT_heading_mae_deg, S2:LAT_curvature_mae_1pm_valid |

### Metric-level headline verdict changes at seed 0 (ON -> OFF)

| tree | metric | ON seed 0 delta [CI] | verdict | OFF seed 0 delta [CI] | verdict | banked both-seed | mixed (s0 OFF + s1 ON) |
|---|---|---|---|---|---|---|---|
| step15000ema_unc | S2:LAT_heading_mae_deg | +0.0713 [+0.0277, +0.1167] | FAIL | +0.0385 [-0.0013, +0.0804] | PASS | FAIL | FAIL |
| step20000_unc | S2:LAT_yaw_rate_mae_radps_valid | +0.0011 [+0.0002, +0.0020] | FAIL | +0.0008 [-0.0001, +0.0017] | PASS | FAIL | FAIL |
| step20000ema_unc | S2:LAT_heading_mae_deg | +0.0424 [+0.0009, +0.0849] | FAIL | +0.0126 [-0.0229, +0.0500] | PASS | FAIL | FAIL |

Sign of the instrument shift (OFF delta minus ON delta, all 70 NR headline rows over 8 trees): negative 68, zero 0, positive 2 (negative = removing the ceiling filter moves refcv8 closer to / better than refcv7-50,400).


## LATERAL heading / curvature / yaw-rate rows at 10.5k-20k, S2 0-2 s (does the FAIL survive?)

| tree | metric | ON seed 0: delta [CI] | verdict | OFF seed 0: delta [CI] | verdict | shift (OFF-ON) |
|---|---|---|---|---|---|---|
| step10500_unc | LAT_heading_mae_deg | +0.2450 [+0.1838, +0.3134] | FAIL | +0.1988 [+0.1439, +0.2565] | FAIL | -0.0462 |
| step10500_unc | LAT_curvature_mae_1pm_valid | +0.0008 [+0.0004, +0.0012] | FAIL | +0.0006 [+0.0003, +0.0010] | FAIL | -0.0002 |
| step10500_unc | LAT_yaw_rate_mae_radps_valid | +0.0037 [+0.0025, +0.0050] | FAIL | +0.0027 [+0.0017, +0.0038] | FAIL | -0.0010 |
| step10500ema_unc | LAT_heading_mae_deg | +0.0428 [-0.0006, +0.0868] | PASS | +0.0076 [-0.0315, +0.0471] | PASS | -0.0352 |
| step10500ema_unc | LAT_curvature_mae_1pm_valid | -0.0000 [-0.0002, +0.0002] | PASS | -0.0001 [-0.0003, +0.0001] | PASS | -0.0001 |
| step10500ema_unc | LAT_yaw_rate_mae_radps_valid | +0.0003 [-0.0005, +0.0011] | PASS | -0.0003 [-0.0010, +0.0004] | PASS | -0.0006 |
| step15000_unc | LAT_heading_mae_deg | +0.1261 [+0.0677, +0.1889] | FAIL | +0.1024 [+0.0430, +0.1673] | FAIL | -0.0237 |
| step15000_unc | LAT_curvature_mae_1pm_valid | +0.0005 [+0.0002, +0.0008] | FAIL | +0.0004 [+0.0001, +0.0007] | FAIL | -0.0001 |
| step15000_unc | LAT_yaw_rate_mae_radps_valid | +0.0019 [+0.0008, +0.0031] | FAIL | +0.0014 [+0.0003, +0.0026] | FAIL | -0.0005 |
| step15000ema_unc | LAT_heading_mae_deg | +0.0713 [+0.0277, +0.1167] | FAIL | +0.0385 [-0.0013, +0.0804] | PASS | -0.0328 |
| step15000ema_unc | LAT_curvature_mae_1pm_valid | +0.0002 [-0.0001, +0.0004] | PASS | +0.0001 [-0.0001, +0.0003] | PASS | -0.0001 |
| step15000ema_unc | LAT_yaw_rate_mae_radps_valid | +0.0007 [-0.0001, +0.0016] | PASS | +0.0001 [-0.0007, +0.0010] | PASS | -0.0006 |
| step20000_unc | LAT_heading_mae_deg | +0.0840 [+0.0404, +0.1291] | FAIL | +0.0719 [+0.0296, +0.1165] | FAIL | -0.0121 |
| step20000_unc | LAT_curvature_mae_1pm_valid | +0.0002 [-0.0000, +0.0004] | PASS | +0.0001 [-0.0001, +0.0003] | PASS | -0.0001 |
| step20000_unc | LAT_yaw_rate_mae_radps_valid | +0.0011 [+0.0002, +0.0020] | FAIL | +0.0008 [-0.0001, +0.0017] | PASS | -0.0003 |
| step20000ema_unc | LAT_heading_mae_deg | +0.0424 [+0.0009, +0.0849] | FAIL | +0.0126 [-0.0229, +0.0500] | PASS | -0.0298 |
| step20000ema_unc | LAT_curvature_mae_1pm_valid | +0.0001 [-0.0002, +0.0003] | PASS | +0.0000 [-0.0002, +0.0003] | PASS | -0.0001 |
| step20000ema_unc | LAT_yaw_rate_mae_radps_valid | +0.0002 [-0.0005, +0.0011] | PASS | -0.0003 [-0.0010, +0.0005] | PASS | -0.0005 |

## All NR headline rows, per tree (refcv8 - refcv7-50,400; ON = banked, OFF = matched; seed 0)

| tree | family | surface:metric | refcv8 mean ON / OFF | refcv7 mean | ON delta [CI] | OFF delta [CI] | shift | paired OFF-ON on refcv8 [CI] | ON verdict | OFF verdict | n win / ep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| step5000_unc | ADE | S2:ade_m | 0.36209 / 0.34742 | 0.27842 | +0.0837 [+0.0608, +0.1092] | +0.0690 [+0.0479, +0.0930] | -0.0147 | -0.0147 [-0.0235, -0.0068] SEP | FAIL | FAIL | 4754/139 |
| step5000_unc | ADE | S6:ade_m | 2.9915 / 2.9061 | 2.5675 | +0.4241 [+0.2338, +0.6260] | +0.3386 [+0.1496, +0.5323] | -0.0855 | -0.0854 [-0.1637, -0.0130] SEP | FAIL | FAIL | 3668/139 |
| step5000_unc | LONGITUDINAL | S2:LON_speed_mae_mps | 0.33033 / 0.31644 | 0.25879 | +0.0715 [+0.0482, +0.0962] | +0.0576 [+0.0361, +0.0821] | -0.0139 | -0.0139 [-0.0232, -0.0054] SEP | FAIL | FAIL | 4754/139 |
| step5000_unc | LONGITUDINAL | S2:LON_along_abs_final_m | 0.6302 / 0.60148 | 0.49906 | +0.1311 [+0.0832, +0.1821] | +0.1024 [+0.0584, +0.1507] | -0.0287 | -0.0287 [-0.0484, -0.0113] SEP | FAIL | FAIL | 4754/139 |
| step5000_unc | LATERAL | S2:LAT_heading_mae_deg | 1.7187 / 1.656 | 1.4181 | +0.3007 [+0.2043, +0.4116] | +0.2380 [+0.1505, +0.3339] | -0.0627 | -0.0627 [-0.0941, -0.0375] SEP | FAIL | FAIL | 4479/139 |
| step5000_unc | LATERAL | S2:LAT_curvature_mae_1pm_valid | 0.004607 / 0.0044397 | 0.0037054 | +0.0009 [+0.0005, +0.0014] | +0.0007 [+0.0003, +0.0012] | -0.0002 | -0.0002 [-0.0002, -0.0001] SEP | FAIL | FAIL | 4447/139 |
| step5000_unc | LATERAL | S2:LAT_yaw_rate_mae_radps_valid | 0.023991 / 0.022723 | 0.018931 | +0.0051 [+0.0034, +0.0069] | +0.0038 [+0.0022, +0.0055] | -0.0013 | -0.0013 [-0.0018, -0.0008] SEP | FAIL | FAIL | 4447/139 |
| step5000ema_unc | ADE | S2:ade_m | 0.28818 / 0.27445 | 0.27842 | +0.0098 [-0.0007, +0.0203] | -0.0040 [-0.0108, +0.0020] | -0.0138 | -0.0137 [-0.0221, -0.0065] SEP | PASS | PASS | 4754/139 |
| step5000ema_unc | ADE | S6:ade_m | 2.5602 / 2.4821 | 2.5675 | -0.0072 [-0.1067, +0.0977] | -0.0854 [-0.1604, -0.0107] | -0.0782 | -0.0782 [-0.1539, -0.0131] SEP | PASS | PASS | 3668/139 |
| step5000ema_unc | LONGITUDINAL | S2:LON_speed_mae_mps | 0.26802 / 0.25216 | 0.25879 | +0.0092 [-0.0026, +0.0219] | -0.0066 [-0.0141, -0.0001] | -0.0158 | -0.0159 [-0.0253, -0.0075] SEP | PASS | PASS | 4754/139 |
| step5000ema_unc | LONGITUDINAL | S2:LON_along_abs_final_m | 0.51967 / 0.4873 | 0.49906 | +0.0206 [-0.0037, +0.0463] | -0.0118 [-0.0270, +0.0017] | -0.0324 | -0.0324 [-0.0512, -0.0155] SEP | PASS | PASS | 4754/139 |
| step5000ema_unc | LATERAL | S2:LAT_heading_mae_deg | 1.4654 / 1.445 | 1.4521 | +0.0133 [-0.0161, +0.0443] | -0.0068 [-0.0329, +0.0186] | -0.0201 | -0.0203 [-0.0368, -0.0065] SEP | PASS | PASS | 4519/139 |
| step5000ema_unc | LATERAL | S2:LAT_curvature_mae_1pm_valid | 0.0038443 / 0.0037858 | 0.0038222 | +0.0000 [-0.0001, +0.0002] | -0.0001 [-0.0002, +0.0001] | -0.0001 | -0.0001 [-0.0001, -0.0000] SEP | PASS | PASS | 4487/139 |
| step5000ema_unc | LATERAL | S2:LAT_yaw_rate_mae_radps_valid | 0.01913 / 0.018779 | 0.018942 | +0.0002 [-0.0003, +0.0007] | -0.0002 [-0.0006, +0.0002] | -0.0004 | -0.0004 [-0.0007, -0.0001] SEP | PASS | PASS | 4487/139 |
| step10500_unc | ADE | S2:ade_m | 0.35961 / 0.35316 | 0.27842 | +0.0812 [+0.0650, +0.0972] | +0.0747 [+0.0610, +0.0888] | -0.0065 | -0.0065 [-0.0151, +0.0014] | FAIL | FAIL | 4754/139 |
| step10500_unc | ADE | S6:ade_m | 2.9029 / 2.8762 | 2.5675 | +0.3355 [+0.1831, +0.4958] | +0.3087 [+0.1675, +0.4578] | -0.0268 | -0.0267 [-0.0963, +0.0339] | FAIL | FAIL | 3668/139 |
| step10500_unc | LONGITUDINAL | S2:LON_speed_mae_mps | 0.31935 / 0.31436 | 0.25879 | +0.0606 [+0.0447, +0.0767] | +0.0556 [+0.0420, +0.0700] | -0.0050 | -0.0050 [-0.0137, +0.0029] | FAIL | FAIL | 4754/139 |
| step10500_unc | LONGITUDINAL | S2:LON_along_abs_final_m | 0.61146 / 0.6013 | 0.49906 | +0.1124 [+0.0788, +0.1461] | +0.1022 [+0.0731, +0.1336] | -0.0102 | -0.0102 [-0.0283, +0.0059] | FAIL | FAIL | 4754/139 |
| step10500_unc | LATERAL | S2:LAT_heading_mae_deg | 1.6591 / 1.6129 | 1.4141 | +0.2450 [+0.1838, +0.3134] | +0.1988 [+0.1439, +0.2565] | -0.0462 | -0.0462 [-0.0771, -0.0190] SEP | FAIL | FAIL | 4502/139 |
| step10500_unc | LATERAL | S2:LAT_curvature_mae_1pm_valid | 0.0045269 / 0.0043553 | 0.0037421 | +0.0008 [+0.0004, +0.0012] | +0.0006 [+0.0003, +0.0010] | -0.0002 | -0.0002 [-0.0003, -0.0001] SEP | FAIL | FAIL | 4464/139 |
| step10500_unc | LATERAL | S2:LAT_yaw_rate_mae_radps_valid | 0.022596 / 0.021658 | 0.01894 | +0.0037 [+0.0025, +0.0050] | +0.0027 [+0.0017, +0.0038] | -0.0010 | -0.0009 [-0.0015, -0.0004] SEP | FAIL | FAIL | 4464/139 |
| step10500ema_unc | ADE | S2:ade_m | 0.28466 / 0.27363 | 0.27842 | +0.0062 [-0.0047, +0.0172] | -0.0048 [-0.0145, +0.0035] | -0.0110 | -0.0110 [-0.0179, -0.0055] SEP | PASS | PASS | 4754/139 |
| step10500ema_unc | ADE | S6:ade_m | 2.5522 / 2.4972 | 2.5675 | -0.0153 [-0.1314, +0.1022] | -0.0703 [-0.1732, +0.0377] | -0.0550 | -0.0550 [-0.1098, -0.0089] SEP | PASS | PASS | 3668/139 |
| step10500ema_unc | LONGITUDINAL | S2:LON_speed_mae_mps | 0.26162 / 0.25039 | 0.25879 | +0.0028 [-0.0092, +0.0143] | -0.0084 [-0.0181, +0.0001] | -0.0112 | -0.0112 [-0.0185, -0.0052] SEP | PASS | PASS | 4754/139 |
| step10500ema_unc | LONGITUDINAL | S2:LON_along_abs_final_m | 0.50887 / 0.48501 | 0.49906 | +0.0098 [-0.0153, +0.0337] | -0.0141 [-0.0354, +0.0040] | -0.0239 | -0.0239 [-0.0386, -0.0114] SEP | PASS | PASS | 4754/139 |
| step10500ema_unc | LATERAL | S2:LAT_heading_mae_deg | 1.4557 / 1.4205 | 1.413 | +0.0428 [-0.0006, +0.0868] | +0.0076 [-0.0315, +0.0471] | -0.0352 | -0.0352 [-0.0565, -0.0170] SEP | PASS | PASS | 4510/139 |
| step10500ema_unc | LATERAL | S2:LAT_curvature_mae_1pm_valid | 0.0037984 / 0.0037191 | 0.0037996 | -0.0000 [-0.0002, +0.0002] | -0.0001 [-0.0003, +0.0001] | -0.0001 | -0.0001 [-0.0001, -0.0000] SEP | PASS | PASS | 4481/139 |
| step10500ema_unc | LATERAL | S2:LAT_yaw_rate_mae_radps_valid | 0.019252 / 0.018635 | 0.01894 | +0.0003 [-0.0005, +0.0011] | -0.0003 [-0.0010, +0.0004] | -0.0006 | -0.0006 [-0.0010, -0.0003] SEP | PASS | PASS | 4481/139 |
| step15000_unc | ADE | S2:ade_m | 0.33383 / 0.31498 | 0.27842 | +0.0554 [+0.0401, +0.0713] | +0.0366 [+0.0228, +0.0498] | -0.0188 | -0.0189 [-0.0285, -0.0111] SEP | FAIL | FAIL | 4754/139 |
| step15000_unc | ADE | S6:ade_m | 2.9048 / 2.8213 | 2.5675 | +0.3374 [+0.1520, +0.5167] | +0.2539 [+0.0790, +0.4187] | -0.0835 | -0.0835 [-0.1505, -0.0300] SEP | FAIL | FAIL | 3668/139 |
| step15000_unc | LONGITUDINAL | S2:LON_speed_mae_mps | 0.30488 / 0.28607 | 0.25879 | +0.0461 [+0.0294, +0.0623] | +0.0273 [+0.0135, +0.0413] | -0.0188 | -0.0188 [-0.0288, -0.0106] SEP | FAIL | FAIL | 4754/139 |
| step15000_unc | LONGITUDINAL | S2:LON_along_abs_final_m | 0.58894 / 0.54997 | 0.49906 | +0.0899 [+0.0559, +0.1243] | +0.0509 [+0.0220, +0.0806] | -0.0390 | -0.0390 [-0.0592, -0.0222] SEP | FAIL | FAIL | 4754/139 |
| step15000_unc | LATERAL | S2:LAT_heading_mae_deg | 1.4555 / 1.4318 | 1.3294 | +0.1261 [+0.0677, +0.1889] | +0.1024 [+0.0430, +0.1673] | -0.0237 | -0.0237 [-0.0399, -0.0096] SEP | FAIL | FAIL | 4495/139 |
| step15000_unc | LATERAL | S2:LAT_curvature_mae_1pm_valid | 0.0040732 / 0.0039949 | 0.0035896 | +0.0005 [+0.0002, +0.0008] | +0.0004 [+0.0001, +0.0007] | -0.0001 | -0.0001 [-0.0001, -0.0000] SEP | FAIL | FAIL | 4454/139 |
| step15000_unc | LATERAL | S2:LAT_yaw_rate_mae_radps_valid | 0.020748 / 0.020206 | 0.018801 | +0.0019 [+0.0008, +0.0031] | +0.0014 [+0.0003, +0.0026] | -0.0005 | -0.0005 [-0.0008, -0.0003] SEP | FAIL | FAIL | 4454/139 |
| step15000ema_unc | ADE | S2:ade_m | 0.2889 / 0.27734 | 0.27842 | +0.0105 [-0.0016, +0.0224] | -0.0011 [-0.0119, +0.0095] | -0.0116 | -0.0116 [-0.0187, -0.0057] SEP | PASS | PASS | 4754/139 |
| step15000ema_unc | ADE | S6:ade_m | 2.632 / 2.58 | 2.5675 | +0.0645 [-0.0740, +0.1998] | +0.0125 [-0.1154, +0.1404] | -0.0520 | -0.0520 [-0.1072, -0.0050] SEP | PASS | PASS | 3668/139 |
| step15000ema_unc | LONGITUDINAL | S2:LON_speed_mae_mps | 0.26642 / 0.25513 | 0.25879 | +0.0076 [-0.0054, +0.0205] | -0.0037 [-0.0149, +0.0071] | -0.0113 | -0.0113 [-0.0186, -0.0051] SEP | PASS | PASS | 4754/139 |
| step15000ema_unc | LONGITUDINAL | S2:LON_along_abs_final_m | 0.51734 / 0.49263 | 0.49906 | +0.0183 [-0.0093, +0.0450] | -0.0064 [-0.0307, +0.0164] | -0.0247 | -0.0247 [-0.0401, -0.0119] SEP | PASS | PASS | 4754/139 |
| step15000ema_unc | LATERAL | S2:LAT_heading_mae_deg | 1.4843 / 1.4516 | 1.413 | +0.0713 [+0.0277, +0.1167] | +0.0385 [-0.0013, +0.0804] | -0.0328 | -0.0327 [-0.0558, -0.0139] SEP | FAIL | PASS | 4510/139 |
| step15000ema_unc | LATERAL | S2:LAT_curvature_mae_1pm_valid | 0.0039644 / 0.0038797 | 0.0037953 | +0.0002 [-0.0001, +0.0004] | +0.0001 [-0.0001, +0.0003] | -0.0001 | -0.0001 [-0.0001, -0.0000] SEP | PASS | PASS | 4481/139 |
| step15000ema_unc | LATERAL | S2:LAT_yaw_rate_mae_radps_valid | 0.019663 / 0.019037 | 0.018936 | +0.0007 [-0.0001, +0.0016] | +0.0001 [-0.0007, +0.0010] | -0.0006 | -0.0006 [-0.0010, -0.0003] SEP | PASS | PASS | 4481/139 |
| step20000_unc | ADE | S2:ade_m | 0.30324 / 0.29952 | 0.27842 | +0.0248 [+0.0139, +0.0359] | +0.0211 [+0.0120, +0.0309] | -0.0037 | -0.0037 [-0.0106, +0.0030] | FAIL | FAIL | 4754/139 |
| step20000_unc | ADE | S6:ade_m | 2.7108 / 2.7341 | 2.5675 | +0.1433 [+0.0166, +0.2685] | +0.1666 [+0.0423, +0.2896] | +0.0233 | +0.0233 [-0.0397, +0.0902] | FAIL | FAIL | 3668/139 |
| step20000_unc | LONGITUDINAL | S2:LON_speed_mae_mps | 0.27574 / 0.27354 | 0.25879 | +0.0170 [+0.0058, +0.0269] | +0.0148 [+0.0052, +0.0242] | -0.0022 | -0.0022 [-0.0096, +0.0051] | FAIL | FAIL | 4754/139 |
| step20000_unc | LONGITUDINAL | S2:LON_along_abs_final_m | 0.53142 / 0.52673 | 0.49906 | +0.0324 [+0.0071, +0.0563] | +0.0277 [+0.0055, +0.0494] | -0.0047 | -0.0047 [-0.0202, +0.0106] | FAIL | FAIL | 4754/139 |
| step20000_unc | LATERAL | S2:LAT_heading_mae_deg | 1.494 / 1.482 | 1.41 | +0.0840 [+0.0404, +0.1291] | +0.0719 [+0.0296, +0.1165] | -0.0121 | -0.0120 [-0.0255, -0.0002] SEP | FAIL | FAIL | 4507/139 |
| step20000_unc | LATERAL | S2:LAT_curvature_mae_1pm_valid | 0.0038524 / 0.0038222 | 0.0036887 | +0.0002 [-0.0000, +0.0004] | +0.0001 [-0.0001, +0.0003] | -0.0001 | -0.0000 [-0.0001, +0.0000] | PASS | PASS | 4470/139 |
| step20000_unc | LATERAL | S2:LAT_yaw_rate_mae_radps_valid | 0.019962 / 0.019674 | 0.018871 | +0.0011 [+0.0002, +0.0020] | +0.0008 [-0.0001, +0.0017] | -0.0003 | -0.0003 [-0.0005, -0.0001] SEP | FAIL | PASS | 4470/139 |
| step20000ema_unc | ADE | S2:ade_m | 0.28661 / 0.27504 | 0.27842 | +0.0082 [-0.0042, +0.0202] | -0.0034 [-0.0141, +0.0066] | -0.0116 | -0.0116 [-0.0186, -0.0057] SEP | PASS | PASS | 4754/139 |
| step20000ema_unc | ADE | S6:ade_m | 2.6521 / 2.6034 | 2.5675 | +0.0847 [-0.0568, +0.2250] | +0.0359 [-0.0960, +0.1698] | -0.0488 | -0.0488 [-0.0988, -0.0032] SEP | PASS | PASS | 3668/139 |
| step20000ema_unc | LONGITUDINAL | S2:LON_speed_mae_mps | 0.26612 / 0.25492 | 0.25879 | +0.0073 [-0.0062, +0.0200] | -0.0039 [-0.0151, +0.0066] | -0.0112 | -0.0112 [-0.0183, -0.0052] SEP | PASS | PASS | 4754/139 |
| step20000ema_unc | LONGITUDINAL | S2:LON_along_abs_final_m | 0.51629 / 0.49274 | 0.49906 | +0.0172 [-0.0113, +0.0429] | -0.0063 [-0.0303, +0.0156] | -0.0235 | -0.0236 [-0.0384, -0.0112] SEP | PASS | PASS | 4754/139 |
| step20000ema_unc | LATERAL | S2:LAT_heading_mae_deg | 1.4545 / 1.4248 | 1.4122 | +0.0424 [+0.0009, +0.0849] | +0.0126 [-0.0229, +0.0500] | -0.0298 | -0.0297 [-0.0521, -0.0112] SEP | FAIL | PASS | 4514/139 |
| step20000ema_unc | LATERAL | S2:LAT_curvature_mae_1pm_valid | 0.0038929 / 0.0038178 | 0.0038028 | +0.0001 [-0.0002, +0.0003] | +0.0000 [-0.0002, +0.0003] | -0.0001 | -0.0001 [-0.0001, -0.0000] SEP | PASS | PASS | 4484/139 |
| step20000ema_unc | LATERAL | S2:LAT_yaw_rate_mae_radps_valid | 0.019176 / 0.018614 | 0.018933 | +0.0002 [-0.0005, +0.0011] | -0.0003 [-0.0010, +0.0005] | -0.0005 | -0.0006 [-0.0010, -0.0002] SEP | PASS | PASS | 4484/139 |
| step30000_unc | ADE | S2:ade_m | 0.31283 / 0.30758 | 0.27842 | +0.0344 [+0.0213, +0.0473] | +0.0292 [+0.0173, +0.0409] | -0.0052 | -0.0053 [-0.0106, -0.0003] SEP | FAIL | FAIL | 4754/139 |
| step30000_unc | ADE | S6:ade_m | 2.9189 / 2.9254 | 2.5675 | +0.3514 [+0.1817, +0.5290] | +0.3580 [+0.1963, +0.5334] | +0.0066 | +0.0065 [-0.0375, +0.0506] | FAIL | FAIL | 3668/139 |
| step30000_unc | LONGITUDINAL | S2:LON_speed_mae_mps | 0.29695 / 0.29325 | 0.25879 | +0.0382 [+0.0241, +0.0524] | +0.0345 [+0.0218, +0.0476] | -0.0037 | -0.0037 [-0.0090, +0.0015] | FAIL | FAIL | 4754/139 |
| step30000_unc | LONGITUDINAL | S2:LON_along_abs_final_m | 0.57384 / 0.56639 | 0.49906 | +0.0748 [+0.0444, +0.1049] | +0.0673 [+0.0402, +0.0957] | -0.0075 | -0.0074 [-0.0187, +0.0031] | FAIL | FAIL | 4754/139 |
| step30000_unc | LATERAL | S2:LAT_heading_mae_deg | 1.4761 / 1.454 | 1.3734 | +0.1027 [+0.0482, +0.1654] | +0.0807 [+0.0278, +0.1438] | -0.0220 | -0.0221 [-0.0390, -0.0074] SEP | FAIL | FAIL | 4511/139 |
| step30000_unc | LATERAL | S2:LAT_curvature_mae_1pm_valid | 0.0041062 / 0.0040453 | 0.003721 | +0.0004 [+0.0001, +0.0008] | +0.0003 [+0.0000, +0.0007] | -0.0001 | -0.0001 [-0.0001, -0.0000] SEP | FAIL | FAIL | 4472/139 |
| step30000_unc | LATERAL | S2:LAT_yaw_rate_mae_radps_valid | 0.020519 / 0.020139 | 0.018875 | +0.0016 [+0.0005, +0.0029] | +0.0013 [+0.0002, +0.0025] | -0.0003 | -0.0004 [-0.0007, -0.0001] SEP | FAIL | FAIL | 4472/139 |
| step30000ema_unc | ADE | S2:ade_m | 0.28777 / 0.28012 | 0.27842 | +0.0093 [-0.0016, +0.0197] | +0.0017 [-0.0072, +0.0098] | -0.0076 | -0.0076 [-0.0146, -0.0015] SEP | PASS | PASS | 4754/139 |
| step30000ema_unc | ADE | S6:ade_m | 2.7333 / 2.7122 | 2.5675 | +0.1658 [+0.0268, +0.3153] | +0.1447 [+0.0154, +0.2842] | -0.0211 | -0.0211 [-0.0817, +0.0397] | FAIL | FAIL | 3668/139 |
| step30000ema_unc | LONGITUDINAL | S2:LON_speed_mae_mps | 0.2696 / 0.26251 | 0.25879 | +0.0108 [-0.0012, +0.0224] | +0.0037 [-0.0062, +0.0132] | -0.0071 | -0.0071 [-0.0146, -0.0003] SEP | PASS | PASS | 4754/139 |
| step30000ema_unc | LONGITUDINAL | S2:LON_along_abs_final_m | 0.52321 / 0.50904 | 0.49906 | +0.0241 [-0.0013, +0.0482] | +0.0100 [-0.0108, +0.0300] | -0.0141 | -0.0142 [-0.0293, -0.0003] SEP | PASS | PASS | 4754/139 |
| step30000ema_unc | LATERAL | S2:LAT_heading_mae_deg | 1.479 / 1.4619 | 1.4521 | +0.0270 [-0.0120, +0.0649] | +0.0098 [-0.0264, +0.0474] | -0.0172 | -0.0171 [-0.0308, -0.0051] SEP | PASS | PASS | 4518/139 |
| step30000ema_unc | LATERAL | S2:LAT_curvature_mae_1pm_valid | 0.0038559 / 0.0037879 | 0.0038449 | +0.0000 [-0.0002, +0.0002] | -0.0001 [-0.0003, +0.0002] | -0.0001 | -0.0001 [-0.0001, -0.0000] SEP | PASS | PASS | 4486/139 |
| step30000ema_unc | LATERAL | S2:LAT_yaw_rate_mae_radps_valid | 0.019155 / 0.018697 | 0.018973 | +0.0002 [-0.0006, +0.0010] | -0.0003 [-0.0010, +0.0005] | -0.0005 | -0.0005 [-0.0007, -0.0002] SEP | PASS | PASS | 4486/139 |

## TACTICAL rows (trajectory-derived decisions; no NR headline), S2 and S6, ON vs OFF (refcv8 - refcv7-50,400; higher is better)

| tree | surface | metric | ON delta [CI] | sep. | OFF delta [CI] | sep. |
|---|---|---|---|---|---|---|
| step5000_unc | S2 | TAC_traj_lat_correct | -0.0206 [-0.0301, -0.0120] | True | -0.0151 [-0.0244, -0.0076] | True |
| step5000_unc | S2 | TAC_traj_lon_correct | -0.0177 [-0.0338, -0.0011] | True | -0.0065 [-0.0208, +0.0080] | False |
| step5000_unc | S6 | TAC_traj_lat_correct | -0.0319 [-0.0461, -0.0186] | True | -0.0281 [-0.0417, -0.0152] | True |
| step5000_unc | S6 | TAC_traj_lon_correct | -0.0308 [-0.0648, +0.0027] | False | -0.0431 [-0.0778, -0.0079] | True |
| step5000ema_unc | S2 | TAC_traj_lat_correct | -0.0027 [-0.0063, +0.0006] | False | -0.0002 [-0.0031, +0.0025] | False |
| step5000ema_unc | S2 | TAC_traj_lon_correct | +0.0025 [-0.0080, +0.0133] | False | +0.0093 [+0.0006, +0.0185] | True |
| step5000ema_unc | S6 | TAC_traj_lat_correct | -0.0057 [-0.0123, +0.0014] | False | -0.0027 [-0.0090, +0.0038] | False |
| step5000ema_unc | S6 | TAC_traj_lon_correct | +0.0169 [-0.0024, +0.0368] | False | +0.0188 [+0.0000, +0.0386] | False |
| step10500_unc | S2 | TAC_traj_lat_correct | -0.0109 [-0.0179, -0.0040] | True | -0.0078 [-0.0143, -0.0017] | True |
| step10500_unc | S2 | TAC_traj_lon_correct | +0.0004 [-0.0133, +0.0149] | False | +0.0034 [-0.0091, +0.0171] | False |
| step10500_unc | S6 | TAC_traj_lat_correct | -0.0256 [-0.0409, -0.0117] | True | -0.0218 [-0.0364, -0.0090] | True |
| step10500_unc | S6 | TAC_traj_lon_correct | -0.0387 [-0.0732, -0.0068] | True | -0.0398 [-0.0740, -0.0079] | True |
| step10500ema_unc | S2 | TAC_traj_lat_correct | -0.0027 [-0.0076, +0.0021] | False | +0.0015 [-0.0027, +0.0057] | False |
| step10500ema_unc | S2 | TAC_traj_lon_correct | +0.0053 [-0.0074, +0.0181] | False | +0.0114 [+0.0019, +0.0221] | True |
| step10500ema_unc | S6 | TAC_traj_lat_correct | -0.0087 [-0.0180, +0.0000] | False | -0.0063 [-0.0147, +0.0022] | False |
| step10500ema_unc | S6 | TAC_traj_lon_correct | -0.0035 [-0.0303, +0.0242] | False | +0.0000 [-0.0261, +0.0267] | False |
| step15000_unc | S2 | TAC_traj_lat_correct | -0.0074 [-0.0128, -0.0021] | True | -0.0048 [-0.0101, +0.0002] | False |
| step15000_unc | S2 | TAC_traj_lon_correct | +0.0015 [-0.0129, +0.0164] | False | +0.0088 [-0.0036, +0.0227] | False |
| step15000_unc | S6 | TAC_traj_lat_correct | -0.0406 [-0.0566, -0.0251] | True | -0.0409 [-0.0570, -0.0252] | True |
| step15000_unc | S6 | TAC_traj_lon_correct | -0.0341 [-0.0734, +0.0030] | False | -0.0316 [-0.0700, +0.0054] | False |
| step15000ema_unc | S2 | TAC_traj_lat_correct | -0.0042 [-0.0101, +0.0017] | False | -0.0011 [-0.0065, +0.0042] | False |
| step15000ema_unc | S2 | TAC_traj_lon_correct | +0.0021 [-0.0109, +0.0160] | False | +0.0097 [-0.0008, +0.0213] | False |
| step15000ema_unc | S6 | TAC_traj_lat_correct | -0.0079 [-0.0167, +0.0011] | False | -0.0052 [-0.0139, +0.0033] | False |
| step15000ema_unc | S6 | TAC_traj_lon_correct | -0.0079 [-0.0400, +0.0237] | False | -0.0025 [-0.0341, +0.0300] | False |
| step20000_unc | S2 | TAC_traj_lat_correct | -0.0040 [-0.0088, +0.0008] | False | -0.0029 [-0.0076, +0.0015] | False |
| step20000_unc | S2 | TAC_traj_lon_correct | +0.0057 [-0.0080, +0.0204] | False | +0.0019 [-0.0105, +0.0154] | False |
| step20000_unc | S6 | TAC_traj_lat_correct | -0.0158 [-0.0273, -0.0038] | True | -0.0131 [-0.0247, -0.0016] | True |
| step20000_unc | S6 | TAC_traj_lon_correct | -0.0063 [-0.0368, +0.0231] | False | -0.0123 [-0.0404, +0.0150] | False |
| step20000ema_unc | S2 | TAC_traj_lat_correct | -0.0023 [-0.0069, +0.0021] | False | -0.0004 [-0.0050, +0.0038] | False |
| step20000ema_unc | S2 | TAC_traj_lon_correct | +0.0013 [-0.0120, +0.0155] | False | +0.0069 [-0.0040, +0.0194] | False |
| step20000ema_unc | S6 | TAC_traj_lat_correct | -0.0035 [-0.0138, +0.0057] | False | -0.0019 [-0.0114, +0.0076] | False |
| step20000ema_unc | S6 | TAC_traj_lon_correct | -0.0139 [-0.0470, +0.0175] | False | -0.0095 [-0.0429, +0.0226] | False |
| step30000_unc | S2 | TAC_traj_lat_correct | -0.0072 [-0.0124, -0.0021] | True | -0.0053 [-0.0107, -0.0002] | True |
| step30000_unc | S2 | TAC_traj_lon_correct | -0.0217 [-0.0385, -0.0046] | True | -0.0225 [-0.0391, -0.0059] | True |
| step30000_unc | S6 | TAC_traj_lat_correct | -0.0174 [-0.0344, -0.0011] | True | -0.0155 [-0.0324, +0.0003] | False |
| step30000_unc | S6 | TAC_traj_lon_correct | -0.0270 [-0.0621, +0.0049] | False | -0.0259 [-0.0600, +0.0052] | False |
| step30000ema_unc | S2 | TAC_traj_lat_correct | -0.0021 [-0.0065, +0.0023] | False | -0.0013 [-0.0055, +0.0029] | False |
| step30000ema_unc | S2 | TAC_traj_lon_correct | -0.0036 [-0.0164, +0.0095] | False | -0.0015 [-0.0128, +0.0105] | False |
| step30000ema_unc | S6 | TAC_traj_lat_correct | -0.0038 [-0.0155, +0.0071] | False | -0.0027 [-0.0139, +0.0082] | False |
| step30000ema_unc | S6 | TAC_traj_lon_correct | -0.0226 [-0.0560, +0.0079] | False | -0.0224 [-0.0545, +0.0079] | False |

## Where the filter-OFF plan lives (probe: `S6_FILTER_OFF_PROBE.json`)

* **S2** (`panel_s0`, slots 0-3 = 0.5-2.0 s): the panel carries an `os_filteroff` arm AND `dump_s0/refcv6_extras.npz :: filter_off.traj` [N, 8, 2] holds the full plan. This tool reads the extras (not the panel arm) and uses the panel arm only as a cross-check (max abs difference below).
* **S6** (`s6_s0`, slots [1,3,4,5,6,7] = 1-6 s): the S6 panel and its `decisions/` carry NO filter-off arm (only `plan_full_nav_{true,shuffled,flipped,zero}`), and there is no S6 extras file. The S6 filter-OFF plan is DERIVED: `filter_off.traj[row][:, [1,3,4,5,6,7]]` of the S2 extras, rows matched on (episode_index, ws). The ON-side of that derivation is verified bit-exact (S6 `os` == `nav_true.traj[:, slots]`, and `decisions/plan_full_nav_true` == `nav_true.traj`, max abs 0.0 in all 8 trees); the OFF side uses the same row map and slot map but has never been rolled on its own (UNVERIFIED beyond that identity).
* **Seed 1**: `dump_s1/refcv6_extras.npz` has 20 keys and NO `filter_off.*` (os-only roll): no matched seed-1 read is possible from banked data.
* **refcv7-50,400 side**: its dump_s0 extras read `nav_true.sel_idx != filter_off.sel_idx` in 0 of 4,754 windows and ON-vs-OFF plan max abs 0.0 (re-verified here; the filter was inert).

| tree | S2 windows where the filter changes the selected anchor | S6 windows (of 3,668) | S2 panel `os_filteroff` vs extras max abs | S6 `os` vs extras `nav_true` slots max abs | ON==OFF plan where the anchor is unchanged (max abs) |
|---|---|---|---|---|---|
| step5000_unc | 384 / 4754 (0.081) | 247 (0.067) | 0.0 | 0.0 | 0.0 |
| step5000ema_unc | 320 / 4754 (0.067) | 202 (0.055) | 0.0 | 0.0 | 0.0 |
| step10500_unc | 405 / 4754 (0.085) | 264 (0.072) | 0.0 | 0.0 | 0.0 |
| step10500ema_unc | 289 / 4754 (0.061) | 178 (0.049) | 0.0 | 0.0 | 0.0 |
| step15000_unc | 261 / 4754 (0.055) | 155 (0.042) | 0.0 | 0.0 | 0.0 |
| step15000ema_unc | 279 / 4754 (0.059) | 171 (0.047) | 0.0 | 0.0 | 0.0 |
| step20000_unc | 444 / 4754 (0.093) | 291 (0.079) | 0.0 | 0.0 | 0.0 |
| step20000ema_unc | 289 / 4754 (0.061) | 174 (0.047) | 0.0 | 0.0 | 0.0 |
| step30000ema_unc | 367 / 4754 (0.077) | 239 (0.065) | 0.0 | 0.0 | 0.0 |

Independent numpy cross-check of the matched ADE (no torch, no pair_read_r8): `ADE_CROSSCHECK.json` (worst |mean diff| 1e-8 m over 8 trees x 2 surfaces).

Cells whose CI lower bound is within 2e-4 of zero are flagged `borderline_lo_le_2e-4` in the JSON; `taniteval.ci` renders bounds at adaptive precision so the published sign equals the unrounded test.
