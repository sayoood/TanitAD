INVALID (control failed: ['K-PRIOR'])

# SELCOR RESULT (mechanical; PREREG_SELCOR.md sha256 dc1f9bc6c8ce)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, over EPISODES: answers 'would another draw of EPISODES say this?' ONLY; NOT another training run (one refcv8 training seed), NOT another inference run (one inference seed, 0)

FIT: 7147 fresh S6 compliant rows, 297 episodes; CV chose {"variant": "S", "k": 0, "lam": 10.0, "cv_ade": 2.5923288154077233} (native CV ADE 2.6006)

## seed 0 (primary)

| arm | MD | FD | picks changed | P1 | P2 | G-strict |
|---|---|---|---|---|---|---|
| BASE-os | 0.4695 | 0.1396 | - | - | - | - |
| SELCOR-G | 0.4883 | 0.1202 | 0.044 | True | False | False (LAT_cross_mae_m) |
| PRIOR-ONLY | 0.4742 | 0.1298 | 0.019 | True | True | True (none) |
| DER | 0.4132 | 0.2241 | 0.231 | False | True | False (LAT_cross_abs_final_m, LAT_cross_mae_m, LON_accel_mae_mps2, LON_along_abs_final_m, LON_along_mae_m, LON_speed_mae_mps, LON_speed_within_0p5, TAC_traj_lon_correct, ade_m, ade_m, fde_m) |
| ORACLE (GT, ceiling) | 0.0751 | 0.0566 | - | - | - | - |

| S6 row (SELCOR-G vs BASE) | delta [95% CI] | worse |
|---|---|---|
| ade_m | -0.0148 [-0.0319, +0.0020] | False |
| fde_m | -0.0372 [-0.0857, +0.0083] | False |
| LON_speed_mae_mps | -0.0095 [-0.0182, -0.0014] | False |
| LON_along_mae_m | -0.0216 [-0.0399, -0.0045] | False |
| LON_along_abs_final_m | -0.0554 [-0.1054, -0.0089] | False |
| LON_speed_within_0p5 | +0.0033 [+0.0010, +0.0058] | False |
| LON_accel_mae_mps2 | -0.0039 [-0.0075, -0.0003] | False |
| LAT_heading_mae_deg | -0.0027 [-0.0117, +0.0041] | False |
| LAT_curvature_mae_1pm_valid | -0.0000 [-0.0000, +0.0000] | False |
| LAT_yaw_rate_mae_radps_valid | +0.0000 [-0.0000, +0.0001] | False |
| LAT_cross_mae_m | +0.0052 [+0.0002, +0.0110] | True |
| LAT_cross_abs_final_m | +0.0153 [-0.0004, +0.0338] | False |
| TAC_traj_lat_correct | -0.0003 [-0.0009, +0.0000] | False |
| TAC_traj_lon_correct | +0.0015 [-0.0040, +0.0075] | False |
| false_decel | -0.0194 [-0.0296, -0.0107] | False |
| missed_decel | +0.0188 [+0.0098, +0.0295] | True |
| along_signed_final_m@GTcruise | +0.2456 [+0.1762, +0.3278] | None |
| speed_err_signed_mps@GTcruise | +0.0426 [+0.0309, +0.0562] | None |

## seed 1 (replicate)

| arm | MD | FD | picks changed | P1 | P2 | G-strict |
|---|---|---|---|---|---|---|
| BASE-os | 0.4711 | 0.1374 | - | - | - | - |
| SELCOR-G | 0.4930 | 0.1175 | 0.045 | True | False | False (LAT_cross_abs_final_m, LAT_cross_mae_m, LAT_heading_mae_deg) |
| PRIOR-ONLY | 0.4789 | 0.1298 | 0.017 | True | False | True (none) |
| DER | 0.4084 | 0.2220 | 0.226 | False | True | False (LAT_cross_abs_final_m, LAT_cross_mae_m, LON_accel_mae_mps2, LON_along_abs_final_m, LON_along_mae_m, LON_speed_mae_mps, LON_speed_within_0p5, TAC_traj_lon_correct, ade_m, ade_m, fde_m) |
| ORACLE (GT, ceiling) | 0.0923 | 0.0555 | - | - | - | - |

| S6 row (SELCOR-G vs BASE) | delta [95% CI] | worse |
|---|---|---|
| ade_m | -0.0233 [-0.0476, -0.0026] | False |
| fde_m | -0.0652 [-0.1307, -0.0097] | False |
| LON_speed_mae_mps | -0.0151 [-0.0266, -0.0058] | False |
| LON_along_mae_m | -0.0296 [-0.0554, -0.0087] | False |
| LON_along_abs_final_m | -0.0840 [-0.1529, -0.0268] | False |
| LON_speed_within_0p5 | +0.0029 [-0.0003, +0.0060] | False |
| LON_accel_mae_mps2 | -0.0065 [-0.0112, -0.0027] | False |
| LAT_heading_mae_deg | +0.0054 [+0.0009, +0.0111] | True |
| LAT_curvature_mae_1pm_valid | -0.0000 [-0.0000, +0.0000] | False |
| LAT_yaw_rate_mae_radps_valid | +0.0001 [-0.0000, +0.0002] | False |
| LAT_cross_mae_m | +0.0066 [+0.0005, +0.0135] | True |
| LAT_cross_abs_final_m | +0.0183 [+0.0010, +0.0370] | True |
| TAC_traj_lat_correct | +0.0003 [+0.0000, +0.0009] | False |
| TAC_traj_lon_correct | +0.0043 [-0.0006, +0.0098] | False |
| false_decel | -0.0199 [-0.0282, -0.0128] | False |
| missed_decel | +0.0219 [+0.0124, +0.0331] | True |
| along_signed_final_m@GTcruise | +0.2644 [+0.1934, +0.3452] | None |
| speed_err_signed_mps@GTcruise | +0.0461 [+0.0341, +0.0600] | None |

controls: {"K-FRESH-ROLL": true, "K-JOIN": true, "K-PRIOR": false, "K-ELIG": true, "K-ID_s0": true, "K-EVALJOIN_s0": true, "K-ADE": true, "K-ID_s1": true, "K-EVALJOIN_s1": true}

NEXT: a control failed: fix the instrument, re-score (0 GPU).

ZZSELCORENDZZ