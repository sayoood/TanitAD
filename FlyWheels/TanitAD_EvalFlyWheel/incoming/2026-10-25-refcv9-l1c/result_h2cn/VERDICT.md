PASS

# L1C-H2CN (mechanical; PREREG_L1C_H2CN sha256 84820fa6b894)

tier: T1 self-action open loop on logged eval139 frames; estimator: F.cell paired episode-cluster bootstrap B 2000 seed 0, episodes only, models fixed; one training seed

| pair (S2 ADE) | s0 | s1 |
|---|---|---|
| H2cn-H0 | -0.0130 [-0.0197, -0.0064] | -0.0136 [-0.0203, -0.0075] |
| H2cn-H2c | -0.0036 [-0.0076, +0.0005] | -0.0037 [-0.0076, +0.0004] |
| H2c-H0 | -0.0094 [-0.0160, -0.0036] | -0.0099 [-0.0158, -0.0043] |

| four families H2cn-H0 (S2) | s0 | s1 |
|---|---|---|
| ade_m | -0.0130 [-0.0197, -0.0064] | -0.0136 [-0.0203, -0.0075] |
| fde_m | -0.0247 [-0.0408, -0.0085] | -0.0253 [-0.0412, -0.0100] |
| LON_speed_mae_mps | -0.0159 [-0.0255, -0.0064] | -0.0161 [-0.0253, -0.0072] |
| LON_along_mae_m | -0.0155 [-0.0235, -0.0079] | -0.0156 [-0.0232, -0.0085] |
| LON_along_abs_final_m | -0.0292 [-0.0479, -0.0112] | -0.0282 [-0.0461, -0.0108] |
| LON_speed_within_0p5 | +0.0154 [+0.0066, +0.0241] | +0.0135 [+0.0049, +0.0223] |
| LON_accel_mae_mps2 | -0.0099 [-0.0245, +0.0044] | -0.0094 [-0.0233, +0.0048] |
| LAT_heading_mae_deg | -0.0045 [-0.0111, +0.0013] | -0.0037 [-0.0102, +0.0028] |
| LAT_curvature_mae_1pm_valid | -0.0001 [-0.0001, +0.0000] | -0.0001 [-0.0001, -0.0000] |
| LAT_yaw_rate_mae_radps_valid | -0.0002 [-0.0004, -0.0001] | -0.0002 [-0.0003, -0.0000] |
| LAT_cross_mae_m | +0.0004 [-0.0010, +0.0020] | +0.0001 [-0.0012, +0.0017] |
| LAT_cross_abs_final_m | +0.0011 [-0.0029, +0.0060] | +0.0004 [-0.0037, +0.0052] |
| TAC_traj_lat_correct | +0.0002 [-0.0007, +0.0012] | +0.0005 [-0.0007, +0.0019] |
| TAC_traj_lon_correct | +0.0125 [-0.0062, +0.0305] | +0.0108 [-0.0067, +0.0275] |

braking AUC (S6 compliant) {"0": {"L1C_H0": 0.7445319667044412, "L1C_H2c": 0.7242551332901624, "L1C_H2cn": 0.7714918582375478, "stage": 0.7445353394312233}, "1": {"L1C_H0": 0.7360605202093788, "L1C_H2c": 0.7241294992175273, "L1C_H2cn": 0.7714851127839836, "stage": 0.7360613633910744}}; H2cn vs H0: AUC-SAME
rows worse at both seeds: none
K-R reproduction vs N1: {"diff_vs_N1": {"0": 0.0, "1": 0.0}, "reproduces": true}
controls: {'K-ROC': True, 'K-H2c': True}

NEXT: H2cn is a refcv8 STAGE-UPGRADE candidate: request the registered SHIP-R8 battery on the new stage from the PI, with these numbers.

ZZH2CNENDZZ