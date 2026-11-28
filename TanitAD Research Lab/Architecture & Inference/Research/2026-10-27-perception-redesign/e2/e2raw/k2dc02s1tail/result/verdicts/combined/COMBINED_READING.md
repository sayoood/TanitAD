# K2DC02s1 tail reading: **VOID**

no dose claim is made: DOSE-VOID

Rule: PREREG_K2DC02S1_2026-11-20.md sec. 5 (tail) / sec. 4 (floor): REPLICATED = SUPPORTED and clears the floor at both seeds; ELIMINATED = REFUTED; UNRESOLVED = anything else; a VOID read decides nothing

Dose-tool outcome (PREREG sec. 3): **DOSE-VOID**. Voids: DOSE-VOID.

## Floor admissibility (sec. 4), LAT_cross_mae_m, metres

| seed | K2DC02s1 - K2 (dose) | K2DC02s1 - K2DC02 (floor) | abs(dose) > abs(floor) |
|---|---|---|---|
| 0 | -0.00520 | -0.00010 | yes |
| 1 | -0.00320 | -0.00240 | yes |

Clears at both seeds: **yes**.

## The floor, every row (K2DC02s1 - K2DC02; training-run noise; never a lever verdict)

Tool label for this comparison (not a verdict): `DOSE-VOID`.

### seed 0

| family | row | delta | lo | hi | separated |
|---|---|---|---|---|---|
| LONGITUDINAL | LON_speed_mae_mps | -0.00900 | -0.01950 | +0.00020 |  |
| LONGITUDINAL | LON_along_mae_m | -0.01020 | -0.02060 | -0.00150 | * |
| LONGITUDINAL | LON_along_abs_final_m | -0.01880 | -0.04070 | +0.00070 |  |
| LONGITUDINAL | LON_speed_within_0p5 | +0.00580 | -0.00230 | +0.01480 |  |
| LONGITUDINAL | LON_accel_mae_mps2 | -0.00450 | -0.01460 | +0.00560 |  |
| LATERAL | LAT_heading_mae_deg | -0.01060 | -0.06780 | +0.04880 |  |
| LATERAL | LAT_curvature_mae_1pm_valid | +0.00010 | -0.00030 | +0.00050 |  |
| LATERAL | LAT_yaw_rate_mae_radps_valid | -0.00010 | -0.00130 | +0.00110 |  |
| LATERAL | LAT_cross_mae_m | -0.00010 | -0.00620 | +0.00540 |  |
| LATERAL | LAT_cross_abs_final_m | -0.00080 | -0.01550 | +0.01220 |  |
| TACTICAL | TAC_traj_lat_correct | +0.00000 | -0.00670 | +0.00670 |  |
| TACTICAL | TAC_traj_lon_correct | -0.00290 | -0.01500 | +0.00960 |  |
| TACTICAL | TAC_traj_man5_correct | -0.00170 | -0.01210 | +0.00920 |  |
| STRATEGIC | - | PAUSED/see strategic/ | | | |

### seed 1

| family | row | delta | lo | hi | separated |
|---|---|---|---|---|---|
| LONGITUDINAL | LON_speed_mae_mps | -0.01300 | -0.02550 | -0.00230 | * |
| LONGITUDINAL | LON_along_mae_m | -0.01300 | -0.02520 | -0.00270 | * |
| LONGITUDINAL | LON_along_abs_final_m | -0.02400 | -0.05020 | -0.00180 | * |
| LONGITUDINAL | LON_speed_within_0p5 | +0.01390 | +0.00540 | +0.02300 | * |
| LONGITUDINAL | LON_accel_mae_mps2 | -0.01150 | -0.02480 | +0.00080 |  |
| LATERAL | LAT_heading_mae_deg | -0.02670 | -0.08790 | +0.03640 |  |
| LATERAL | LAT_curvature_mae_1pm_valid | -0.00000 | -0.00050 | +0.00040 |  |
| LATERAL | LAT_yaw_rate_mae_radps_valid | -0.00050 | -0.00190 | +0.00090 |  |
| LATERAL | LAT_cross_mae_m | -0.00240 | -0.00910 | +0.00350 |  |
| LATERAL | LAT_cross_abs_final_m | -0.00690 | -0.02250 | +0.00660 |  |
| TACTICAL | TAC_traj_lat_correct | +0.00330 | -0.00330 | +0.01050 |  |
| TACTICAL | TAC_traj_lon_correct | +0.00750 | -0.00630 | +0.02050 |  |
| TACTICAL | TAC_traj_man5_correct | +0.00710 | -0.00500 | +0.01920 |  |
| STRATEGIC | - | PAUSED/see strategic/ | | | |

