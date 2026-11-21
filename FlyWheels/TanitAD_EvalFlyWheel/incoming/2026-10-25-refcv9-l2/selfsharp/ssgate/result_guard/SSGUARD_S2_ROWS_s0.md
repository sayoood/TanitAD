# SSGUARD S2 rows, inference seed 0 (DESCRIPTIVE; S2_compliant 4159 w / 138 e, stage on)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap (pair_read_r8.paired_cell), B 2000, seed 0, over EPISODES only; one training seed; one inference seed
controls: {"K1_S2_stage_bit_equal_banked": true, "K2_S2_compliant_n": true, "K-ID_guard_base_base_bit_equal": true}

| S2 row | GUARD-ALL - BASE-os | worse | SELFSHARP - BASE-os | worse |
|---|---|---|---|---|
| ade_m | +0.0006 [-0.0056, +0.0060] | False | +0.0008 [-0.0064, +0.0069] | False |
| fde_m | -0.0030 [-0.0176, +0.0095] | False | -0.0019 [-0.0181, +0.0119] | False |
| LON_speed_mae_mps | -0.0017 [-0.0101, +0.0052] | False | -0.0017 [-0.0101, +0.0053] | False |
| LON_along_mae_m | -0.0002 [-0.0074, +0.0058] | False | -0.0002 [-0.0076, +0.0059] | False |
| LON_along_abs_final_m | -0.0055 [-0.0225, +0.0085] | False | -0.0056 [-0.0225, +0.0087] | False |
| LON_speed_within_0p5 | -0.0005 [-0.0070, +0.0062] | False | -0.0005 [-0.0070, +0.0063] | False |
| LON_accel_mae_mps2 | -0.0067 [-0.0168, +0.0020] | False | -0.0067 [-0.0168, +0.0020] | False |
| LAT_heading_mae_deg | +0.0011 [-0.0044, +0.0069] | False | +0.0442 [+0.0113, +0.0816] | True |
| LAT_curvature_mae_1pm_valid | -0.0000 [-0.0001, +0.0000] | False | +0.0003 [+0.0000, +0.0005] | True |
| LAT_yaw_rate_mae_radps_valid | +0.0000 [-0.0002, +0.0002] | False | +0.0010 [+0.0003, +0.0016] | True |
| LAT_cross_mae_m | +0.0001 [-0.0005, +0.0008] | False | +0.0004 [-0.0024, +0.0031] | False |
| LAT_cross_abs_final_m | +0.0004 [-0.0013, +0.0023] | False | +0.0015 [-0.0046, +0.0074] | False |
| TAC_traj_lat_correct | +0.0000 [-0.0016, +0.0015] | False | -0.0024 [-0.0069, +0.0020] | False |
| TAC_traj_lon_correct | +0.0060 [-0.0021, +0.0143] | False | +0.0060 [-0.0021, +0.0143] | False |

E1-style ADE(0-2 s) minus echo ha0_ext: BASE-os -0.0286 [-0.0410, -0.0162]; SELFSHARP -0.0278 [-0.0392, -0.0165]; GUARD-ALL -0.0280 [-0.0395, -0.0168]
GUARD-ALL rows separated worse: none

ZZSSGS2ENDZZ