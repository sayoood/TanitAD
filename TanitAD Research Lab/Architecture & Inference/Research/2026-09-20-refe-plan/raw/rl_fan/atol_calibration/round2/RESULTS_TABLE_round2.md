| arm | role | CLI exit | sets | m p50 | m max (worst) | worst / M | sets above M | max-abs delta (round-1 statistic) |
|---|---|---|---|---|---|---|---|---|
| N0_b4 | NULL | 0 | 48 | 0.0041548 | 0.0078805 | 0.32 | 0 | 0.25 |
| N1_b1 | NULL | 0 | 48 | 0.0046106 | 0.011183 | 0.45 | 0 | 0.25 |
| N2_b2 | NULL | 0 | 48 | 0.0043603 | 0.0093462 | 0.37 | 0 | 0.25 |
| N_full | NULL | n/a (CPU) | 1365 | 0.0041548 | 0.012193 | 0.49 | 0 | 0.25 |
| A2_calib_half_scale | DECISION | 5 | 48 | 0.14542 | 0.50515 | 20.21 | 47 | 4 |
| B1_img_hflip | DECISION | 5 | 48 | 0.25526 | 2.0572 | 82.29 | 47 | 10.19 |
| B2_img_bgr_rgb | DECISION | 5 | 48 | 0.036099 | 0.17942 | 7.18 | 29 | 1.25 |
| C1_head_noise10 | DECISION | 5 | 48 | 0.15327 | 0.40586 | 16.23 | 48 | 1.519 |
| C2_snap_epoch021 | DECISION | 4 | 48 | 0.15421 | 0.38648 | 15.46 | 48 | 6.5 |
| A1_calib_other_vehicle | info | 0 | 48 | 0.0067573 | 0.043199 | 1.73 | 2 | 0.25 |
| N3_jpeg | info | 0 | 48 | 0.005609 | 0.026631 | 1.07 | 1 | 0.25 |
| S_focal105 | info | 0 | 48 | 0.0060985 | 0.012481 | 0.50 | 0 | 0.25 |
| S_yaw3 | info | 0 | 48 | 0.022467 | 0.082814 | 3.31 | 24 | 0.5 |
| S_head_noise1 | info | 0 | 48 | 0.016353 | 0.041792 | 1.67 | 14 | 0.25 |
| S_snap027 | info | 4 | 48 | 0.010787 | 0.032556 | 1.30 | 2 | 0.25 |
| S_wrong_scene | info | 4 | not run / no sets | | | | | |
| S_cam_reversed | info | 5 | 48 | 0.44413 | 2.0696 | 82.78 | 48 | 11.62 |

state PASS; F = 0.012192648437499983; M = 0.025; need >= 12 sets above M and worst m >= 5 x M on every DECISION arm.
