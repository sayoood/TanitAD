FAIL (E2, E4)

# MAPEDGE Q2 (mechanical; PREREG_MAPEDGE_Q2 sha256 be11ea00bb28; arm label: headline)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance); estimator: paired episode-cluster bootstrap B 2000 seed 0 over EPISODES only; one training seed; one inference seed

| arm (S6c) | off-drivable | edge contact | lane cross | edge minus HUMAN |
|---|---|---|---|---|
| BASE | 0.1792 | 0.2855 | 0.2939 | 0.1370 |
| HUMAN | 0.0125 | 0.1485 | 0.1941 | - |
| OWN-T1 | 0.1657 | 0.2708 | 0.2908 | - |
| OWN-T12 | 0.1660 | 0.2708 | 0.2914 | 0.1223 |
| GT-T12 | 0.0113 | 0.0610 | 0.2747 | -0.0876 |
| OWN-BND-T12 | 0.1492 | 0.2523 | 0.3015 | 0.1038 |
| DER-BND-T12 | 0.2736 | 0.3505 | 0.3597 | 0.2020 |

E1 OWN-BND-T12 edge contact vs BASE: -0.0331 [-0.0483, -0.0187] -> True
E2 no four-family row worse: False ['S6c_stage:ade_m', 'S6c_stage:fde_m', 'S6c_stage:LON_speed_mae_mps', 'S6c_stage:LON_along_mae_m', 'S6c_stage:LON_along_abs_final_m', 'S6c_stage:LON_speed_within_0p5', 'S6c_stage:LON_accel_mae_mps2', 'S6c_stage:LAT_heading_mae_deg', 'S6c_stage:LAT_curvature_mae_1pm_valid', 'S6c_stage:LAT_yaw_rate_mae_radps_valid', 'S6c_stage:LAT_cross_mae_m', 'S6c_stage:LAT_cross_abs_final_m', 'S6c_stage:TAC_traj_lat_correct', 'S6c_stage:TAC_traj_lon_correct', 'S6c_stage:false_decel', 'S2c_stage:ade_m', 'S2c_stage:fde_m', 'S2c_stage:LON_speed_mae_mps', 'S2c_stage:LON_along_mae_m', 'S2c_stage:LON_along_abs_final_m', 'S2c_stage:LON_speed_within_0p5', 'S2c_stage:LON_accel_mae_mps2', 'S2c_stage:LAT_heading_mae_deg', 'S2c_stage:LAT_curvature_mae_1pm_valid', 'S2c_stage:LAT_yaw_rate_mae_radps_valid', 'S2c_stage:LAT_cross_mae_m', 'S2c_stage:LAT_cross_abs_final_m', 'S2c_stage:TAC_traj_lat_correct', 'S2c_stage:TAC_traj_lon_correct']
E3 DER-BND-T12 not separated below BASE: +0.0650 [+0.0384, +0.0930] -> True
E4 STOP guard: False
GT-T12 clears E1/E2/E4: False

controls failed: none

NEXT: the edge rule costs driving quality: a soft edge cost (rank penalty) instead of a hard filter.

ZZMEQ2ENDZZ