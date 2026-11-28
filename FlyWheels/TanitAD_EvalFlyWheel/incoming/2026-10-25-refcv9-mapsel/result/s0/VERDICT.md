# refcv9 MAP-SELECT -- VERDICT: FAIL

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance); GT-* arms = T0-ORACLE BOUND (privileged map)
estimator: paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci, B 2000, seed 0, over EPISODES) = 'another draw of episodes' ONLY; one checkpoint (EMA-48,852), one inference seed (0): NOT another training run, NOT another inference run.
prereg sha256: e9be326e692a6402e70bf79c98097770a5cec24259f27d1d6a01d179c64a3bc9 (seal ok: True)
windows: 4754; lane-change exception active: True

## Controls

| control | pass | detail |
|---|---|---|
| K_G_fields | True | {"n_windows": 4754, "fan_n_seen": [149], "batch_gt1": 0, "rule_seen": ["prior_corrected"], "logit_shapes": [[1, 8, 1000, 600]], "decide_mismatch_cells": 0, "decide_checked": 72, "n_stack": 3} |
| K_B_emitted_equals_fan_at_sel_idx | True | {"max_abs": 0.0} |
| K_C_sel_idx_is_argmax_over_e9_eligible | True | {"share": 1.0, "emitted_eligible_share": 1.0} |
| K_A_roll_equals_L1_BASEos | True | {"n_compared": 4754, "n_bit_equal": 4754, "max_abs": 0.0} |
| K_RAST_all_windows_have_a_raster | True | {"missing": 0, "n": 4754} |
| K_LC_JOIN | True | {"n_checked": 393, "n_agree": 359, "agree": 0.9134860050890585, "clips_checked": 12, "clips_agree_ge_0p99": 11, "ok": true, "n_lc_windows": 48} |
| TAU_CAL | True | {"tau": 0.8627451062202454, "rule": "AMENDMENT 1: tau = quantile_0.99(method=higher) of the HUMAN paths' MAX own-map P^(sidewalk) over the in-range FP5 points (S6-valid human windows with an own raster); a candidate is rejected iff its max P^(sidewalk) > tau; privileged GT trajectories, NO outcome m |
| K_LIVE_own_map_beats_deranged_donor | True | {"acc_own_vs_GT_drivable": 0.888326434838555, "acc_deranged_donor_vs_GT_drivable": 0.782626399115945, "n_gt_windows": 4686, "margin_required": 0.02} |
| K_DERANGED_donor_is_another_episode | True | {"n_no_donor": 0} |
| K_BASE_arm_equals_emitted_pick | True | {"share": 1.0} |
| K_STAGE_BASE_staged_equals_banked_S2_pos | True | {"amended": "A2: battery S2_plan8 staged through the arms' call == S2_pos", "descriptive_BASEos_staged_differs_from_banked_n": 4621} |
| K_S6_alignment | True | {} |
| K_HUMAN_off_drivable_le_2pct | True | {"rate_S6c": 0.012507817385866166, "n": 3198, "max": 0.02} |
| K_HUMAN_prerolled | None | {"n": 3198, "off_drivable": 0.012507817385866166, "edge_contact": 0.1485303314571607, "lane_cross": 0.1985616010006254, "off_drivable_per_episode_q50_q90_max": [0.0, 0.0, 0.8846153846153846], "coverage_nocls_share": 0.0070714134010878, "coverage_unseen_share": 0.012024831034546267} |
| K_SENS_blind_lateral_shift | True | {"shift_m": 3.5, "human_off_rate": 0.012507817385866166, "n": 3198, "plus_left": {"rate": 0.6191369606003753, "delta": 0.6066, "lo": 0.5331, "hi": 0.6751, "separated": true, "n_windows": 3198, "n_episodes": 136, "separated_above": true}, "minus_right": {"rate": 0.8098811757348343, "delta": 0.7974, " |
| K_GTLIVE_GT_T1_lowers_off_drivable | True | {"delta": -0.1626, "hi": -0.129} |

## MAP-COMPLIANCE vs the GT map, S6-compliant (6 instants, 1-6 s): n compliant 3251, with GT 3198 (EMITTED plan after the shipped stage; paired delta vs BASE)

| arm | off-drivable (sidewalk) rate | delta vs BASE [CI] | edge-contact rate | delta vs BASE [CI] | minus HUMAN | lane-cross rate | delta vs BASE [CI] | minus HUMAN | COVERAGE nocls / not-seen (share of in-range check points) |
|---|---|---|---|---|---|---|---|---|---|
| BASE | 0.1792 | - | 0.2855 | - | +0.1370 | 0.2939 | - | +0.0999 | 0.0148 / 0.0118 |
| GT-T1 | 0.0166 | -0.1626 [-0.1996, -0.1290] | 0.2148 | -0.0707 [-0.0902, -0.0539] | +0.0663 | 0.2721 | -0.0218 [-0.0340, -0.0113] | +0.0781 | 0.0117 / 0.0124 |
| GT-T12 | 0.0113 | -0.1679 [-0.2059, -0.1336] | 0.0610 | -0.2245 [-0.2663, -0.1858] | -0.0876 | 0.2747 | -0.0193 [-0.0351, -0.0051] | +0.0806 | 0.0126 / 0.0136 |
| GT-T123 | 0.0119 | -0.1673 [-0.2050, -0.1326] | 0.0666 | -0.2189 [-0.2600, -0.1815] | -0.0819 | 0.0970 | -0.1969 [-0.2332, -0.1629] | -0.0970 | 0.0139 / 0.0141 |
| OWN-T1 | 0.1657 | -0.0134 [-0.0251, -0.0013] | 0.2708 | -0.0147 [-0.0250, -0.0048] | +0.1223 | 0.2908 | -0.0032 [-0.0120, +0.0064] | +0.0967 | 0.0161 / 0.0154 |
| OWN-T12 | 0.1660 | -0.0131 [-0.0250, -0.0009] | 0.2708 | -0.0147 [-0.0249, -0.0047] | +0.1223 | 0.2914 | -0.0025 [-0.0117, +0.0072] | +0.0973 | 0.0161 / 0.0154 |
| OWN-T123 | 0.1676 | -0.0116 [-0.0234, +0.0010] | 0.2711 | -0.0144 [-0.0245, -0.0038] | +0.1226 | 0.2911 | -0.0028 [-0.0121, +0.0072] | +0.0970 | 0.0164 / 0.0155 |
| DER-T1 | 0.2305 | +0.0513 [+0.0289, +0.0746] | 0.3168 | +0.0313 [+0.0100, +0.0521] | +0.1682 | 0.3284 | +0.0345 [+0.0160, +0.0541] | +0.1343 | 0.0286 / 0.0193 |
| DER-T12 | 0.2301 | +0.0510 [+0.0286, +0.0742] | 0.3171 | +0.0316 [+0.0103, +0.0524] | +0.1685 | 0.3281 | +0.0341 [+0.0154, +0.0538] | +0.1340 | 0.0286 / 0.0193 |
| DER-T123 | 0.2398 | +0.0607 [+0.0374, +0.0865] | 0.3233 | +0.0378 [+0.0161, +0.0602] | +0.1748 | 0.3360 | +0.0420 [+0.0215, +0.0650] | +0.1419 | 0.0297 / 0.0197 |
| OWN-CAL-T1 | 0.1767 | -0.0025 [-0.0049, -0.0006] | 0.2846 | -0.0009 [-0.0023, +0.0003] | +0.1360 | 0.2936 | -0.0003 [-0.0025, +0.0019] | +0.0996 | 0.0148 / 0.0118 |
| DER-CAL-T1 | 0.1945 | +0.0153 [+0.0037, +0.0285] | 0.2939 | +0.0084 [-0.0028, +0.0199] | +0.1454 | 0.2968 | +0.0028 [-0.0062, +0.0126] | +0.1027 | 0.0179 / 0.0129 |
| HUMAN | 0.0125 | - | 0.1485 | - | +0.0000 | 0.1941 | - | +0.0000 | 0.0071 / 0.0120 |

## MAP-COMPLIANCE vs the GT map, S2-compliant (4 instants, 0.5-2 s): n compliant 4159, with GT 4091 (EMITTED plan after the shipped stage; paired delta vs BASE)

| arm | off-drivable (sidewalk) rate | delta vs BASE [CI] | edge-contact rate | delta vs BASE [CI] | minus HUMAN | lane-cross rate | delta vs BASE [CI] | minus HUMAN | COVERAGE nocls / not-seen (share of in-range check points) |
|---|---|---|---|---|---|---|---|---|---|
| BASE | 0.1797 | - | 0.2723 | - | +0.1238 | 0.2741 | - | +0.0801 | 0.0169 / 0.0169 |
| GT-T1 | 0.0156 | -0.1640 [-0.2005, -0.1309] | 0.2031 | -0.0692 [-0.0874, -0.0527] | +0.0546 | 0.2524 | -0.0217 [-0.0336, -0.0115] | +0.0583 | 0.0138 / 0.0174 |
| GT-T12 | 0.0112 | -0.1684 [-0.2055, -0.1349] | 0.0601 | -0.2122 [-0.2505, -0.1762] | -0.0884 | 0.2561 | -0.0180 [-0.0320, -0.0051] | +0.0621 | 0.0145 / 0.0183 |
| GT-T123 | 0.0120 | -0.1677 [-0.2040, -0.1346] | 0.0648 | -0.2075 [-0.2464, -0.1721] | -0.0838 | 0.0874 | -0.1867 [-0.2218, -0.1552] | -0.1066 | 0.0156 / 0.0187 |
| OWN-T1 | 0.1621 | -0.0176 [-0.0284, -0.0071] | 0.2569 | -0.0154 [-0.0248, -0.0059] | +0.1084 | 0.2704 | -0.0037 [-0.0146, +0.0072] | +0.0764 | 0.0165 / 0.0196 |
| OWN-T12 | 0.1623 | -0.0174 [-0.0280, -0.0066] | 0.2569 | -0.0154 [-0.0251, -0.0058] | +0.1084 | 0.2712 | -0.0030 [-0.0139, +0.0077] | +0.0771 | 0.0165 / 0.0196 |
| OWN-T123 | 0.1635 | -0.0161 [-0.0268, -0.0052] | 0.2571 | -0.0152 [-0.0253, -0.0051] | +0.1086 | 0.2709 | -0.0032 [-0.0143, +0.0078] | +0.0769 | 0.0168 / 0.0197 |
| DER-T1 | 0.2288 | +0.0491 [+0.0278, +0.0709] | 0.3043 | +0.0320 [+0.0114, +0.0512] | +0.1558 | 0.3109 | +0.0368 [+0.0191, +0.0546] | +0.1169 | 0.0301 / 0.0228 |
| DER-T12 | 0.2286 | +0.0489 [+0.0275, +0.0708] | 0.3046 | +0.0323 [+0.0117, +0.0515] | +0.1560 | 0.3107 | +0.0366 [+0.0188, +0.0546] | +0.1166 | 0.0301 / 0.0228 |
| DER-T123 | 0.2374 | +0.0577 [+0.0359, +0.0801] | 0.3104 | +0.0381 [+0.0171, +0.0581] | +0.1619 | 0.3181 | +0.0440 [+0.0249, +0.0644] | +0.1240 | 0.0314 / 0.0231 |
| OWN-CAL-T1 | 0.1777 | -0.0020 [-0.0039, -0.0005] | 0.2716 | -0.0007 [-0.0018, +0.0002] | +0.1230 | 0.2741 | +0.0000 [-0.0017, +0.0018] | +0.0801 | 0.0168 / 0.0169 |
| DER-CAL-T1 | 0.1936 | +0.0139 [+0.0043, +0.0241] | 0.2791 | +0.0068 [-0.0034, +0.0164] | +0.1306 | 0.2781 | +0.0040 [-0.0036, +0.0120] | +0.0840 | 0.0195 / 0.0178 |
| HUMAN | 0.0125 | - | 0.1485 | - | +0.0000 | 0.1941 | - | +0.0000 | 0.0071 / 0.0120 |

T2 / T3 are read RELATIVE to the HUMAN floor (the 'minus HUMAN' columns), never as an absolute bar. nocls (code 0) is UNKNOWN, not a violation (AMENDMENT 1); its share is the COVERAGE of the check.

## GT vs OWN gap per tier (S6-compliant; share of the GT-oracle gain the deployable map recovers)

| tier | metric | BASE | GT | OWN | DER | GT gain | OWN gain | OWN share of GT gain |
|---|---|---|---|---|---|---|---|---|
| T1 | off_drivable | 0.1792 | 0.0166 | 0.1657 | 0.2305 | 0.1626 | 0.0134 | 0.0827 |
| T1 | edge_contact | 0.2855 | 0.2148 | 0.2708 | 0.3168 | 0.0707 | 0.0147 | 0.2080 |
| T1 | lane_cross | 0.2939 | 0.2721 | 0.2908 | 0.3284 | 0.0218 | 0.0032 | 0.1449 |
| T12 | off_drivable | 0.1792 | 0.0113 | 0.1660 | 0.2301 | 0.1679 | 0.0131 | 0.0782 |
| T12 | edge_contact | 0.2855 | 0.0610 | 0.2708 | 0.3171 | 0.2245 | 0.0147 | 0.0655 |
| T12 | lane_cross | 0.2939 | 0.2747 | 0.2914 | 0.3281 | 0.0193 | 0.0025 | 0.1311 |
| T123 | off_drivable | 0.1792 | 0.0119 | 0.1676 | 0.2398 | 0.1673 | 0.0116 | 0.0692 |
| T123 | edge_contact | 0.2855 | 0.0666 | 0.2711 | 0.3233 | 0.2189 | 0.0144 | 0.0657 |
| T123 | lane_cross | 0.2939 | 0.0970 | 0.2911 | 0.3360 | 0.1969 | 0.0028 | 0.0144 |
| T1_CAL | off_drivable | 0.1792 | 0.0166 | 0.1767 | 0.1945 | 0.1626 | 0.0025 | 0.0154 |
| T1_CAL | edge_contact | 0.2855 | 0.2148 | 0.2846 | 0.2939 | 0.0707 | 0.0009 | 0.0133 |
| T1_CAL | lane_cross | 0.2939 | 0.2721 | 0.2936 | 0.2968 | 0.0218 | 0.0003 | 0.0145 |

## Verdict clauses (OWN arms)

| arm | C1 off-drivable separated below BASE | C2 no four-family row separated worse | C3 deranged does not beat BASE | C4 STOP guard | PASS |
|---|---|---|---|---|---|
| OWN-T1 | True (-0.0134 [-0.0251, -0.0013]) | False (S6c_stage:ade_m, S6c_stage:fde_m, S6c_stage:LON_speed_mae_mps, S6c_stage:LON_along_mae_m, S6c_stage:LON_along_abs_final_m, S6c_stage:LON_accel_mae_mps2) | True | False (path-len -0.6995 [-0.9295, -0.4792] m; stop-share +0.0132 [+0.0062, +0.0221]; guard rows worse: S6c_stage:LON_speed_mae_mps, S6c_stage:LON_along_mae_m, S6c_stage:LON_along_abs_final_m, S6c_stage:LON_accel_mae_mps2, S6c_stage:false_decel, S2c_stage:LON_speed_mae_mps, S2c_stage:LON_along_mae_m, S2c_stage:LON_along_abs_final_m, S2c_stage:LON_speed_within_0p5) | **False** |
| OWN-CAL-T1 | True (-0.0025 [-0.0049, -0.0006]) | False (S6c_stage:LAT_curvature_mae_1pm_valid, S2c_stage:LAT_curvature_mae_1pm_valid) | True | True (path-len -0.0150 [-0.0439, +0.0099] m; stop-share +0.0000 [+0.0000, +0.0000]; guard rows worse: none) | **False** |
| OWN-T12 | True (-0.0131 [-0.0250, -0.0009]) | False (S6c_stage:ade_m, S6c_stage:fde_m, S6c_stage:LON_speed_mae_mps, S6c_stage:LON_along_mae_m, S6c_stage:LON_along_abs_final_m, S6c_stage:LON_accel_mae_mps2) | True | False (path-len -0.7032 [-0.9361, -0.4824] m; stop-share +0.0132 [+0.0062, +0.0221]; guard rows worse: S6c_stage:LON_speed_mae_mps, S6c_stage:LON_along_mae_m, S6c_stage:LON_along_abs_final_m, S6c_stage:LON_accel_mae_mps2, S6c_stage:false_decel, S2c_stage:LON_speed_mae_mps, S2c_stage:LON_along_mae_m, S2c_stage:LON_along_abs_final_m, S2c_stage:LON_speed_within_0p5) | **False** |
| OWN-T123 | False (-0.0116 [-0.0234, +0.0010]) | False (S6c_stage:ade_m, S6c_stage:fde_m, S6c_stage:LON_speed_mae_mps, S6c_stage:LON_along_mae_m, S6c_stage:LON_along_abs_final_m, S6c_stage:LON_accel_mae_mps2) | True | False (path-len -0.7629 [-0.9955, -0.5420] m; stop-share +0.0132 [+0.0062, +0.0221]; guard rows worse: S6c_stage:LON_speed_mae_mps, S6c_stage:LON_along_mae_m, S6c_stage:LON_along_abs_final_m, S6c_stage:LON_accel_mae_mps2, S6c_stage:false_decel, S2c_stage:LON_speed_mae_mps, S2c_stage:LON_along_mae_m, S2c_stage:LON_along_abs_final_m, S2c_stage:LON_speed_within_0p5, S2c_stage:LON_accel_mae_mps2) | **False** |

## STOP-trap guard (S6-compliant, EMITTED staged plan; delta = arm - BASE)

| arm | mean path length to 6 s (m) | delta vs BASE [CI] | stop share (6 s displacement < 1 m) | delta [CI] |
|---|---|---|---|---|
| BASE | 67.727 | - | 0.0261 | - |
| GT-T1 | 67.055 | -0.6717 [-0.8789, -0.4794] | 0.0265 | +0.0003 [+0.0000, +0.0012] |
| GT-T12 | 66.586 | -1.1409 [-1.4969, -0.8187] | 0.0277 | +0.0015 [+0.0003, +0.0034] |
| GT-T123 | 65.484 | -2.2428 [-2.7067, -1.8247] | 0.0295 | +0.0034 [+0.0012, +0.0062] |
| OWN-T1 | 67.027 | -0.6995 [-0.9295, -0.4792] | 0.0394 | +0.0132 [+0.0062, +0.0221] |
| OWN-T12 | 67.023 | -0.7032 [-0.9361, -0.4824] | 0.0394 | +0.0132 [+0.0062, +0.0221] |
| OWN-T123 | 66.964 | -0.7629 [-0.9955, -0.5420] | 0.0394 | +0.0132 [+0.0062, +0.0221] |
| DER-T1 | 66.222 | -1.5043 [-1.9638, -1.0977] | 0.0446 | +0.0185 [+0.0093, +0.0294] |
| DER-T12 | 66.205 | -1.5214 [-1.9884, -1.1153] | 0.0446 | +0.0185 [+0.0093, +0.0294] |
| DER-T123 | 65.926 | -1.8005 [-2.3634, -1.3268] | 0.0449 | +0.0188 [+0.0094, +0.0297] |
| OWN-CAL-T1 | 67.712 | -0.0150 [-0.0439, +0.0099] | 0.0261 | +0.0000 [+0.0000, +0.0000] |
| DER-CAL-T1 | 67.224 | -0.5026 [-0.7436, -0.2927] | 0.0268 | +0.0006 [+0.0000, +0.0019] |
| HUMAN | 66.793 | - | 0.0203 | - |

## CALIBRATED own-map arm (OWN-CAL-T1): tau = 0.8627451062202454; human paths passing at tau 0.9904580152671756, at the hard rule (P >= 0.502) 0.8208833151581243 (= 1 - the plan-level FALSE-ALARM rate of the hard own-map rule on human paths); n human paths 3668. Calibrated ON the eval human paths (privileged GT trajectories, no outcome metric); a train-side calibration is not available in this roll.

## GT-T1 is the T0-ORACLE BOUND for the learned-selector lever (a DrivableCritic / Hydra-MDP++ / DDv2-style selector with map-derived labels); OWN-* is what the head's own map delivers with NO training.

## Own-map tier increments (PREDICTED before the roll: OWN-T12 ~ OWN-T1, the own edge channel is empty)

{
 "OWN-T12_vs_OWN-T1_changed_pick_share": 0.001472444257467396,
 "OWN-T123_vs_OWN-T12_changed_pick_share": 0.01430374421539756,
 "GT-T12_vs_GT-T1_changed_pick_share": 0.15397559949516196,
 "GT-T123_vs_GT-T12_changed_pick_share": 0.1779554059739167,
 "note": "PREDICTED before the roll (prereg sec. 3.4): the own edge channel is empty (IoU 0.001) and lane IoU is 0.2 -> OWN-T12 ~ OWN-T1 (< 2 % of windows differ)"
}


## Four families -- worst rows per arm (S6c and S2c, +stage, delta = arm - BASE; 'worse' = CI excludes 0 on the bad side)

* S6c_stage GT-T1: worse rows ['false_decel']; separated (other direction or unsigned) ['ade_m', 'fde_m', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LAT_heading_mae_deg', 'LAT_cross_mae_m', 'missed_decel', 'along_signed_final_m@GTcruise']
* S6c_stage GT-T12: worse rows ['LON_speed_mae_mps', 'LON_speed_within_0p5', 'LON_accel_mae_mps2', 'LAT_curvature_mae_1pm_valid', 'TAC_traj_lon_correct', 'false_decel']; separated (other direction or unsigned) ['missed_decel', 'along_signed_final_m@GTcruise', 'speed_err_signed_mps@GTcruise']
* S6c_stage GT-T123: worse rows ['ade_m', 'fde_m', 'LON_speed_mae_mps', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LON_speed_within_0p5', 'LON_accel_mae_mps2', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lon_correct', 'false_decel']; separated (other direction or unsigned) ['missed_decel', 'along_signed_final_m@GTcruise', 'speed_err_signed_mps@GTcruise']
* S6c_stage OWN-T1: worse rows ['ade_m', 'fde_m', 'LON_speed_mae_mps', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LON_accel_mae_mps2', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lon_correct', 'false_decel']; separated (other direction or unsigned) ['missed_decel', 'along_signed_final_m@GTcruise', 'speed_err_signed_mps@GTcruise']
* S6c_stage OWN-T12: worse rows ['ade_m', 'fde_m', 'LON_speed_mae_mps', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LON_accel_mae_mps2', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lon_correct', 'false_decel']; separated (other direction or unsigned) ['missed_decel', 'along_signed_final_m@GTcruise', 'speed_err_signed_mps@GTcruise']
* S6c_stage OWN-T123: worse rows ['ade_m', 'fde_m', 'LON_speed_mae_mps', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LON_accel_mae_mps2', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lon_correct', 'false_decel']; separated (other direction or unsigned) ['missed_decel', 'along_signed_final_m@GTcruise', 'speed_err_signed_mps@GTcruise']
* S6c_stage DER-T1: worse rows ['ade_m', 'fde_m', 'LON_speed_mae_mps', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LON_speed_within_0p5', 'LON_accel_mae_mps2', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct', 'TAC_traj_lon_correct', 'false_decel']; separated (other direction or unsigned) ['missed_decel', 'along_signed_final_m@GTcruise', 'speed_err_signed_mps@GTcruise']
* S6c_stage DER-T12: worse rows ['ade_m', 'fde_m', 'LON_speed_mae_mps', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LON_speed_within_0p5', 'LON_accel_mae_mps2', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct', 'TAC_traj_lon_correct', 'false_decel']; separated (other direction or unsigned) ['missed_decel', 'along_signed_final_m@GTcruise', 'speed_err_signed_mps@GTcruise']
* S6c_stage DER-T123: worse rows ['ade_m', 'fde_m', 'LON_speed_mae_mps', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LON_speed_within_0p5', 'LON_accel_mae_mps2', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct', 'TAC_traj_lon_correct', 'false_decel']; separated (other direction or unsigned) ['missed_decel', 'along_signed_final_m@GTcruise', 'speed_err_signed_mps@GTcruise']
* S6c_stage OWN-CAL-T1: worse rows ['LAT_curvature_mae_1pm_valid']; separated (other direction or unsigned) ['along_signed_final_m@GTcruise']
* S6c_stage DER-CAL-T1: worse rows ['ade_m', 'fde_m', 'LON_speed_mae_mps', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LON_speed_within_0p5', 'LON_accel_mae_mps2', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct', 'TAC_traj_lon_correct', 'false_decel']; separated (other direction or unsigned) ['missed_decel', 'speed_err_signed_mps@GTcruise']
* S2c_stage GT-T1: worse rows none; separated (other direction or unsigned) none
* S2c_stage GT-T12: worse rows ['ade_m', 'fde_m', 'LON_speed_mae_mps', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LON_speed_within_0p5', 'LON_accel_mae_mps2', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m']; separated (other direction or unsigned) none
* S2c_stage GT-T123: worse rows ['ade_m', 'fde_m', 'LON_speed_mae_mps', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LON_speed_within_0p5', 'LON_accel_mae_mps2', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct', 'TAC_traj_lon_correct']; separated (other direction or unsigned) none
* S2c_stage OWN-T1: worse rows ['ade_m', 'fde_m', 'LON_speed_mae_mps', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LON_speed_within_0p5', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) none
* S2c_stage OWN-T12: worse rows ['ade_m', 'fde_m', 'LON_speed_mae_mps', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LON_speed_within_0p5', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) none
* S2c_stage OWN-T123: worse rows ['ade_m', 'fde_m', 'LON_speed_mae_mps', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LON_speed_within_0p5', 'LON_accel_mae_mps2', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) none
* S2c_stage DER-T1: worse rows ['ade_m', 'fde_m', 'LON_speed_mae_mps', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LON_speed_within_0p5', 'LON_accel_mae_mps2', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct', 'TAC_traj_lon_correct']; separated (other direction or unsigned) none
* S2c_stage DER-T12: worse rows ['ade_m', 'fde_m', 'LON_speed_mae_mps', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LON_speed_within_0p5', 'LON_accel_mae_mps2', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct', 'TAC_traj_lon_correct']; separated (other direction or unsigned) none
* S2c_stage DER-T123: worse rows ['ade_m', 'fde_m', 'LON_speed_mae_mps', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LON_speed_within_0p5', 'LON_accel_mae_mps2', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct', 'TAC_traj_lon_correct']; separated (other direction or unsigned) none
* S2c_stage OWN-CAL-T1: worse rows ['LAT_curvature_mae_1pm_valid']; separated (other direction or unsigned) none
* S2c_stage DER-CAL-T1: worse rows ['ade_m', 'fde_m', 'LON_speed_mae_mps', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LON_speed_within_0p5', 'LON_accel_mae_mps2', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) none

## Availability of a tier-compliant eligible candidate in the 149-fan (share of windows whose set did not need the fallback)

{
 "gt": {
  "T1_set_nonempty": 0.993689524610854,
  "T12_after_T1": 0.9558266722759782,
  "T123_after_T12": 0.9339503575936053
 },
 "own": {
  "T1_set_nonempty": 0.9511989903239377,
  "T12_after_T1": 0.9991586032814472,
  "T123_after_T12": 0.9987379049221708
 },
 "der": {
  "T1_set_nonempty": 0.9066049642406395,
  "T12_after_T1": 0.9995793016407236,
  "T123_after_T12": 0.9884307951198991
 },
 "cal": {
  "T1_set_nonempty": 0.9993689524610854
 }
}

## Candidate-level (all windows): share of windows whose pick changed; fallback shares (T1, T2, T3)

* BASE: changed 0.0000; fallbacks [0.0, 0.0, 0.0]; kept-set median 143.0
* GT-T1: changed 0.1719; fallbacks [0.006310475389145982, 0.0, 0.0]; kept-set median 66.0
* GT-T12: changed 0.2598; fallbacks [0.006310475389145982, 0.044173327724021876, 0.0]; kept-set median 58.0
* GT-T123: changed 0.3809; fallbacks [0.006310475389145982, 0.044173327724021876, 0.06604964240639462]; kept-set median 36.0
* OWN-T1: changed 0.1205; fallbacks [0.04880100967606226, 0.0, 0.0]; kept-set median 81.0
* OWN-T12: changed 0.1216; fallbacks [0.04880100967606226, 0.0008413967185527977, 0.0]; kept-set median 81.0
* OWN-T123: changed 0.1340; fallbacks [0.04880100967606226, 0.0008413967185527977, 0.0012620950778291964]; kept-set median 77.0
* DER-T1: changed 0.2890; fallbacks [0.09339503575936053, 0.0, 0.0]; kept-set median 79.5
* DER-T12: changed 0.2913; fallbacks [0.09339503575936053, 0.00042069835927639884, 0.0]; kept-set median 79.0
* DER-T123: changed 0.3296; fallbacks [0.09339503575936053, 0.00042069835927639884, 0.011569204880100967]; kept-set median 71.0
* OWN-CAL-T1: changed 0.0093; fallbacks [0.0006310475389145982, 0.0, 0.0]; kept-set median 125.0
* DER-CAL-T1: changed 0.1003; fallbacks [0.007362221287336979, 0.0, 0.0]; kept-set median 124.0

## NEXT LEVER (mechanical, PREREG sec. 6.3)

GT-T1 itself does not clear C1/C2/C4: even with the true map, compliance-first selection costs accuracy or progress (or does not lower off-drivable). The fan/selector, not the map channel, is the bottleneck -> next: a LEARNED listwise selector with map-derived labels over the same fan (not a rule), and a fan-availability audit (availability block).

{
 "gt_T1_clears_C1_C2_C4": false,
 "own_T1_captures_share_of_gt_T1_gain": 0.08269230769230775,
 "own_cal_T1_captures_share": 0.015384615384615363
}


## Evidence class: MEASURED (raw artifacts: MAPSEL_RESULT.json, features_s0.npz). Not computed: distance-keeping and STRATEGIC rows (no lead track in the banked dumps / --no-strategic; n = 0).