FAIL (G_strict)

# SSGATE RESULT (mechanical; PREREG_SSGATE sha256 6e5d4235224f)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, over EPISODES: answers 'would another draw of EPISODES say this?' ONLY; NOT another training run (one refcv8 training seed), NOT another inference run (one inference seed, 0)

MD (S6 compliant, stage on): BASE-os 0.4695 -> SSGATE-CF 0.4319 (SELFSHARP 0.3897)
FD: BASE-os 0.1395 -> SSGATE-CF 0.1455 (SELFSHARP 0.1503)
sharpened windows 1349 / 3668; tau per fold [0.876, inf, 1.374, 0.91, 0.872]

| bar | value | cell |
|---|---|---|
| P1 MD separated below BASE-os | True | -0.0376 [-0.0704, -0.0080] |
| P2 FD not separated above | True | +0.0059 [-0.0016, +0.0141] |
| G-strict | False | LAT_heading_mae_deg, LAT_yaw_rate_mae_radps_valid |

| S6 row (SSGATE-CF - BASE-os) | delta [95% CI] | worse |
|---|---|---|
| ade_m | -0.0374 [-0.0988, +0.0136] | False |
| fde_m | -0.1146 [-0.2802, +0.0201] | False |
| LON_speed_mae_mps | -0.0265 [-0.0540, -0.0042] | False |
| LON_along_mae_m | -0.0489 [-0.1114, +0.0044] | False |
| LON_along_abs_final_m | -0.1525 [-0.3160, -0.0148] | False |
| LON_speed_within_0p5 | +0.0024 [-0.0029, +0.0075] | False |
| LON_accel_mae_mps2 | -0.0109 [-0.0218, -0.0022] | False |
| LAT_heading_mae_deg | +0.0737 [+0.0300, +0.1266] | True |
| LAT_curvature_mae_1pm_valid | +0.0001 [-0.0001, +0.0002] | False |
| LAT_yaw_rate_mae_radps_valid | +0.0005 [+0.0001, +0.0010] | True |
| LAT_cross_mae_m | +0.0086 [-0.0070, +0.0250] | False |
| LAT_cross_abs_final_m | +0.0280 [-0.0175, +0.0755] | False |
| TAC_traj_lat_correct | -0.0049 [-0.0105, +0.0003] | False |
| TAC_traj_lon_correct | +0.0126 [+0.0034, +0.0219] | False |
| false_decel | +0.0059 [-0.0016, +0.0141] | False |
| missed_decel | -0.0376 [-0.0704, -0.0080] | False |
| along_signed_final_m@GTcruise | -0.1286 [-0.2826, +0.0159] | None |
| speed_err_signed_mps@GTcruise | -0.0256 [-0.0514, -0.0010] | None |

controls: {"K-IND": true, "K-ALL": true, "K-NONE": true, "K-SS": true}
full-data tau: {"tau": 1.1147937774658203, "md": 0.42566510172143973, "n_sharpened": 1525}

NEXT: the lateral cost rides on the sharpened windows whatever the margin: the lever is a SOFTER sharpen (temperature on the LON6 logits) with a lateral guard, one GPU roll, pre-registered first.

ZZSSGATEENDZZ