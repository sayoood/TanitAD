# K2DC02s1 ema reading: **VOID**

no dose claim is made: DOSE-VOID

Rule: PREREG_K2DC02S1_2026-11-20.md sec. 5 (tail) / sec. 4 (floor): REPLICATED = SUPPORTED and clears the floor at both seeds; ELIMINATED = REFUTED; UNRESOLVED = anything else; a VOID read decides nothing

Dose-tool outcome (PREREG sec. 3): **DOSE-VOID**. Voids: DOSE-VOID.

## Floor admissibility (sec. 4), LAT_cross_mae_m, metres

| seed | K2DC02s1 - K2 (dose) | K2DC02s1 - K2DC02 (floor) | abs(dose) > abs(floor) |
|---|---|---|---|
| 0 | +0.00070 | +0.00210 | NO |
| 1 | +0.00060 | +0.00090 | NO |

Clears at both seeds: **no**.

## The floor, every row (K2DC02s1 - K2DC02; training-run noise; never a lever verdict)

Tool label for this comparison (not a verdict): `DOSE-VOID`.

### seed 0

| family | row | delta | lo | hi | separated |
|---|---|---|---|---|---|
| LONGITUDINAL | LON_speed_mae_mps | -0.00580 | -0.01070 | -0.00110 | * |
| LONGITUDINAL | LON_along_mae_m | -0.00490 | -0.00930 | -0.00080 | * |
| LONGITUDINAL | LON_along_abs_final_m | -0.00750 | -0.01770 | +0.00250 |  |
| LONGITUDINAL | LON_speed_within_0p5 | +0.00480 | -0.00060 | +0.01010 |  |
| LONGITUDINAL | LON_accel_mae_mps2 | -0.00540 | -0.01130 | +0.00070 |  |
| LATERAL | LAT_heading_mae_deg | +0.02080 | -0.00390 | +0.04590 |  |
| LATERAL | LAT_curvature_mae_1pm_valid | +0.00000 | -0.00010 | +0.00020 |  |
| LATERAL | LAT_yaw_rate_mae_radps_valid | +0.00030 | -0.00020 | +0.00080 |  |
| LATERAL | LAT_cross_mae_m | +0.00210 | -0.00020 | +0.00480 |  |
| LATERAL | LAT_cross_abs_final_m | +0.00470 | -0.00040 | +0.01110 |  |
| TACTICAL | TAC_traj_lat_correct | -0.00290 | -0.00670 | +0.00040 |  |
| TACTICAL | TAC_traj_lon_correct | +0.00290 | -0.00670 | +0.01210 |  |
| TACTICAL | TAC_traj_man5_correct | -0.00250 | -0.01050 | +0.00580 |  |
| STRATEGIC | - | PAUSED/see strategic/ | | | |

### seed 1

| family | row | delta | lo | hi | separated |
|---|---|---|---|---|---|
| LONGITUDINAL | LON_speed_mae_mps | -0.00290 | -0.00900 | +0.00320 |  |
| LONGITUDINAL | LON_along_mae_m | -0.00230 | -0.00800 | +0.00330 |  |
| LONGITUDINAL | LON_along_abs_final_m | -0.00160 | -0.01400 | +0.01120 |  |
| LONGITUDINAL | LON_speed_within_0p5 | +0.00170 | -0.00470 | +0.00830 |  |
| LONGITUDINAL | LON_accel_mae_mps2 | -0.00180 | -0.00860 | +0.00500 |  |
| LATERAL | LAT_heading_mae_deg | +0.01240 | -0.02030 | +0.04530 |  |
| LATERAL | LAT_curvature_mae_1pm_valid | -0.00000 | -0.00020 | +0.00020 |  |
| LATERAL | LAT_yaw_rate_mae_radps_valid | +0.00010 | -0.00060 | +0.00080 |  |
| LATERAL | LAT_cross_mae_m | +0.00090 | -0.00200 | +0.00380 |  |
| LATERAL | LAT_cross_abs_final_m | +0.00100 | -0.00550 | +0.00770 |  |
| TACTICAL | TAC_traj_lat_correct | -0.00130 | -0.00670 | +0.00420 |  |
| TACTICAL | TAC_traj_lon_correct | -0.00330 | -0.01210 | +0.00500 |  |
| TACTICAL | TAC_traj_man5_correct | -0.00460 | -0.01380 | +0.00500 |  |
| STRATEGIC | - | PAUSED/see strategic/ | | | |

