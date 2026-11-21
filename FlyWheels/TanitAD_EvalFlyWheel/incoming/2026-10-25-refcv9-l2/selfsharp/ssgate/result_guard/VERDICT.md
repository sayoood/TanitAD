PASS (GUARD-ALL)

# SSGUARD RESULT (mechanical; PREREG_SSGUARD sha256 2b9d182742c3)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, over EPISODES: answers 'would another draw of EPISODES say this?' ONLY; NOT another training run (one refcv8 training seed), NOT another inference run (one inference seed, 0)

MD: BASE-os 0.4695 -> GUARD-ALL 0.3897 (SELFSHARP 0.3897)
FD: BASE-os 0.1395 -> GUARD-ALL 0.1503 (SELFSHARP 0.1503)

## GUARD-ALL vs BASE-os

P1 True -0.0798 [-0.1317, -0.0357]; P2 True +0.0108 [-0.0165, +0.0358]; G-strict True (none)

| S6 row | delta [95% CI] | worse |
|---|---|---|
| ade_m | -0.1090 [-0.2105, -0.0153] | False |
| fde_m | -0.3410 [-0.6152, -0.0932] | False |
| LON_speed_mae_mps | -0.0671 [-0.1165, -0.0227] | False |
| LON_along_mae_m | -0.1400 [-0.2492, -0.0392] | False |
| LON_along_abs_final_m | -0.4331 [-0.7309, -0.1643] | False |
| LON_speed_within_0p5 | +0.0071 [-0.0051, +0.0187] | False |
| LON_accel_mae_mps2 | -0.0271 [-0.0469, -0.0096] | False |
| LAT_heading_mae_deg | +0.0347 [-0.0106, +0.0945] | False |
| LAT_curvature_mae_1pm_valid | +0.0007 [-0.0002, +0.0024] | False |
| LAT_yaw_rate_mae_radps_valid | +0.0004 [-0.0005, +0.0018] | False |
| LAT_cross_mae_m | +0.0192 [-0.0021, +0.0414] | False |
| LAT_cross_abs_final_m | +0.0556 [-0.0054, +0.1172] | False |
| TAC_traj_lat_correct | -0.0025 [-0.0088, +0.0025] | False |
| TAC_traj_lon_correct | +0.0249 [+0.0049, +0.0437] | False |
| false_decel | +0.0108 [-0.0165, +0.0358] | False |
| missed_decel | -0.0798 [-0.1317, -0.0357] | False |
| along_signed_final_m@GTcruise | -0.2733 [-0.6114, +0.0784] | None |
| speed_err_signed_mps@GTcruise | -0.0501 [-0.1073, +0.0096] | None |

controls: {"K-ID": true, "K-SSID": true, "K-ALL": true, "K-NONE": true, "K-IND": true}; degenerate base paths: 42

NEXT: the lat/lon-FACTORED decode is the lever: roll SELFSHARP at inference seed 1 and re-read at seed 1, then refcv9's decode design (lon class drives the speed profile only) and the PI for SHIP-R8.

ZZSSGUARDENDZZ