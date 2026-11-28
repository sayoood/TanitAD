PASS (GUARD-ALL)

# SSGUARD RESULT seed 1 (mechanical; PREREG_SSGUARD_S1.md sha256 68fc8a0745c4)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, over EPISODES: answers 'would another draw of EPISODES say this?' ONLY; NOT another training run (one refcv8 training seed), NOT another inference run (one inference seed, 0)

MD: BASE-os 0.4710 -> GUARD-ALL 0.3897 (SELFSHARP 0.3897)
FD: BASE-os 0.1374 -> GUARD-ALL 0.1525 (SELFSHARP 0.1525)

## GUARD-ALL vs BASE-os

P1 True -0.0814 [-0.1292, -0.0404]; P2 True +0.0151 [-0.0100, +0.0407]; G-strict True (none)

| S6 row | delta [95% CI] | worse |
|---|---|---|
| ade_m | -0.1139 [-0.2094, -0.0224] | False |
| fde_m | -0.3550 [-0.6182, -0.1094] | False |
| LON_speed_mae_mps | -0.0700 [-0.1179, -0.0278] | False |
| LON_along_mae_m | -0.1381 [-0.2446, -0.0400] | False |
| LON_along_abs_final_m | -0.4239 [-0.7185, -0.1589] | False |
| LON_speed_within_0p5 | +0.0064 [-0.0055, +0.0178] | False |
| LON_accel_mae_mps2 | -0.0274 [-0.0453, -0.0104] | False |
| LAT_heading_mae_deg | +0.0275 [-0.0305, +0.0942] | False |
| LAT_curvature_mae_1pm_valid | +0.0007 [-0.0001, +0.0024] | False |
| LAT_yaw_rate_mae_radps_valid | +0.0005 [-0.0003, +0.0019] | False |
| LAT_cross_mae_m | +0.0084 [-0.0155, +0.0318] | False |
| LAT_cross_abs_final_m | +0.0219 [-0.0480, +0.0919] | False |
| TAC_traj_lat_correct | -0.0015 [-0.0068, +0.0027] | False |
| TAC_traj_lon_correct | +0.0261 [+0.0065, +0.0462] | False |
| false_decel | +0.0151 [-0.0100, +0.0407] | False |
| missed_decel | -0.0814 [-0.1292, -0.0404] | False |
| along_signed_final_m@GTcruise | -0.2629 [-0.6119, +0.0820] | None |
| speed_err_signed_mps@GTcruise | -0.0511 [-0.1109, +0.0089] | None |

controls: {"K-ID": true, "K-SSID": true, "K-ALL": true, "K-NONE": true, "K-HOOK": true}; degenerate base paths: 41

NEXT: the lat/lon-FACTORED decode is the lever: roll SELFSHARP at inference seed 1 and re-read at seed 1, then refcv9's decode design (lon class drives the speed profile only) and the PI for SHIP-R8.

ZZSSGUARDENDZZ