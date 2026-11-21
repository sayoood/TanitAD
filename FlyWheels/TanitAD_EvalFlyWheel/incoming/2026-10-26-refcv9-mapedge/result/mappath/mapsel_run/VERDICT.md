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
| K_GTLIVE_GT_T1_lowers_off_drivable | True | {"delta": -0.1573, "hi": -0.1256} |

## MAP-COMPLIANCE vs the GT map, S6-compliant (6 instants, 1-6 s): n compliant 3251, with GT 3198 (EMITTED plan after the shipped stage; paired delta vs BASE)

| arm | off-drivable (sidewalk) rate | delta vs BASE [CI] | edge-contact rate | delta vs BASE [CI] | minus HUMAN | lane-cross rate | delta vs BASE [CI] | minus HUMAN | COVERAGE nocls / not-seen (share of in-range check points) |
|---|---|---|---|---|---|---|---|---|---|
| BASE | 0.1792 | - | 0.2855 | - | +0.1370 | 0.2939 | - | +0.0999 | 0.0148 / 0.0118 |
| GT-T1 | 0.0219 | -0.1573 [-0.1925, -0.1256] | 0.2217 | -0.0638 [-0.0803, -0.0487] | +0.0732 | 0.2813 | -0.0126 [-0.0231, -0.0031] | +0.0872 | 0.0125 / 0.0123 |
| GT-T12 | 0.0200 | -0.1592 [-0.1947, -0.1271] | 0.0782 | -0.2073 [-0.2445, -0.1717] | -0.0704 | 0.2898 | -0.0041 [-0.0175, +0.0090] | +0.0958 | 0.0143 / 0.0128 |
| GT-T123 | 0.0206 | -0.1585 [-0.1940, -0.1265] | 0.0775 | -0.2079 [-0.2456, -0.1717] | -0.0710 | 0.1321 | -0.1618 [-0.1940, -0.1306] | -0.0619 | 0.0146 / 0.0133 |
| OWN-T1 | 0.1723 | -0.0069 [-0.0169, +0.0035] | 0.2814 | -0.0041 [-0.0123, +0.0042] | +0.1329 | 0.2984 | +0.0044 [-0.0025, +0.0127] | +0.1043 | 0.0149 / 0.0119 |
| OWN-T12 | 0.1726 | -0.0066 [-0.0167, +0.0043] | 0.2814 | -0.0041 [-0.0124, +0.0041] | +0.1329 | 0.2993 | +0.0054 [-0.0019, +0.0142] | +0.1052 | 0.0149 / 0.0119 |
| OWN-T123 | 0.1739 | -0.0053 [-0.0159, +0.0054] | 0.2814 | -0.0041 [-0.0121, +0.0045] | +0.1329 | 0.2990 | +0.0051 [-0.0022, +0.0137] | +0.1049 | 0.0151 / 0.0120 |
| DER-T1 | 0.2533 | +0.0741 [+0.0506, +0.1000] | 0.3412 | +0.0557 [+0.0356, +0.0762] | +0.1926 | 0.3451 | +0.0512 [+0.0338, +0.0708] | +0.1511 | 0.0329 / 0.0124 |
| DER-T12 | 0.2539 | +0.0747 [+0.0514, +0.1008] | 0.3418 | +0.0563 [+0.0361, +0.0769] | +0.1932 | 0.3445 | +0.0506 [+0.0334, +0.0702] | +0.1504 | 0.0331 / 0.0124 |
| DER-T123 | 0.2683 | +0.0891 [+0.0632, +0.1189] | 0.3502 | +0.0647 [+0.0435, +0.0877] | +0.2017 | 0.3530 | +0.0591 [+0.0391, +0.0820] | +0.1590 | 0.0357 / 0.0127 |
| OWN-CAL-T1 | 0.1764 | -0.0028 [-0.0063, +0.0000] | 0.2839 | -0.0016 [-0.0040, +0.0003] | +0.1354 | 0.2939 | +0.0000 [-0.0021, +0.0022] | +0.0999 | 0.0149 / 0.0118 |
| DER-CAL-T1 | 0.2036 | +0.0244 [+0.0129, +0.0379] | 0.2986 | +0.0131 [+0.0032, +0.0240] | +0.1501 | 0.3025 | +0.0085 [-0.0009, +0.0186] | +0.1084 | 0.0198 / 0.0125 |
| MAPSOFT-G-XF | 0.1792 | +0.0000 [+0.0000, +0.0000] | 0.2855 | +0.0000 [+0.0000, +0.0000] | +0.1370 | 0.2939 | +0.0000 [+0.0000, +0.0000] | +0.0999 | 0.0148 / 0.0118 |
| DER-MAPSOFT-G-XF | 0.1779 | -0.0013 [-0.0029, +0.0000] | 0.2858 | +0.0003 [-0.0009, +0.0019] | +0.1373 | 0.2936 | -0.0003 [-0.0016, +0.0010] | +0.0996 | 0.0148 / 0.0118 |
| SOFTG0 | 0.1792 | +0.0000 [+0.0000, +0.0000] | 0.2855 | +0.0000 [+0.0000, +0.0000] | +0.1370 | 0.2939 | +0.0000 [+0.0000, +0.0000] | +0.0999 | 0.0148 / 0.0118 |
| HUMAN | 0.0125 | - | 0.1485 | - | +0.0000 | 0.1941 | - | +0.0000 | 0.0071 / 0.0120 |

## MAP-COMPLIANCE vs the GT map, S2-compliant (4 instants, 0.5-2 s): n compliant 4159, with GT 4091 (EMITTED plan after the shipped stage; paired delta vs BASE)

| arm | off-drivable (sidewalk) rate | delta vs BASE [CI] | edge-contact rate | delta vs BASE [CI] | minus HUMAN | lane-cross rate | delta vs BASE [CI] | minus HUMAN | COVERAGE nocls / not-seen (share of in-range check points) |
|---|---|---|---|---|---|---|---|---|---|
| BASE | 0.1797 | - | 0.2723 | - | +0.1238 | 0.2741 | - | +0.0801 | 0.0169 / 0.0169 |
| GT-T1 | 0.0205 | -0.1591 [-0.1936, -0.1282] | 0.2080 | -0.0643 [-0.0811, -0.0482] | +0.0595 | 0.2611 | -0.0131 [-0.0225, -0.0044] | +0.0670 | 0.0149 / 0.0174 |
| GT-T12 | 0.0186 | -0.1611 [-0.1958, -0.1303] | 0.0743 | -0.1980 [-0.2325, -0.1640] | -0.0742 | 0.2682 | -0.0059 [-0.0177, +0.0056] | +0.0742 | 0.0169 / 0.0178 |
| GT-T123 | 0.0193 | -0.1604 [-0.1948, -0.1300] | 0.0738 | -0.1985 [-0.2333, -0.1642] | -0.0747 | 0.1200 | -0.1541 [-0.1831, -0.1258] | -0.0740 | 0.0173 / 0.0181 |
| OWN-T1 | 0.1696 | -0.0100 [-0.0185, -0.0019] | 0.2667 | -0.0056 [-0.0125, +0.0014] | +0.1182 | 0.2798 | +0.0057 [-0.0010, +0.0144] | +0.0858 | 0.0164 / 0.0171 |
| OWN-T12 | 0.1699 | -0.0098 [-0.0185, -0.0013] | 0.2664 | -0.0059 [-0.0130, +0.0012] | +0.1179 | 0.2808 | +0.0067 [-0.0002, +0.0155] | +0.0868 | 0.0164 / 0.0171 |
| OWN-T123 | 0.1711 | -0.0086 [-0.0175, +0.0002] | 0.2667 | -0.0056 [-0.0129, +0.0020] | +0.1182 | 0.2806 | +0.0064 [-0.0008, +0.0153] | +0.0865 | 0.0166 / 0.0171 |
| DER-T1 | 0.2552 | +0.0755 [+0.0545, +0.0977] | 0.3295 | +0.0572 [+0.0383, +0.0761] | +0.1810 | 0.3277 | +0.0536 [+0.0369, +0.0717] | +0.1337 | 0.0352 / 0.0176 |
| DER-T12 | 0.2557 | +0.0760 [+0.0548, +0.0985] | 0.3300 | +0.0577 [+0.0387, +0.0767] | +0.1815 | 0.3272 | +0.0531 [+0.0366, +0.0710] | +0.1332 | 0.0354 / 0.0176 |
| DER-T123 | 0.2679 | +0.0882 [+0.0660, +0.1138] | 0.3376 | +0.0653 [+0.0451, +0.0861] | +0.1890 | 0.3351 | +0.0610 [+0.0421, +0.0814] | +0.1411 | 0.0382 / 0.0179 |
| OWN-CAL-T1 | 0.1775 | -0.0022 [-0.0049, +0.0000] | 0.2711 | -0.0012 [-0.0031, +0.0002] | +0.1226 | 0.2744 | +0.0002 [-0.0015, +0.0020] | +0.0803 | 0.0169 / 0.0169 |
| DER-CAL-T1 | 0.2019 | +0.0222 [+0.0126, +0.0337] | 0.2838 | +0.0115 [+0.0012, +0.0220] | +0.1353 | 0.2828 | +0.0086 [+0.0015, +0.0165] | +0.0887 | 0.0219 / 0.0175 |
| MAPSOFT-G-XF | 0.1797 | +0.0000 [+0.0000, +0.0000] | 0.2723 | +0.0000 [+0.0000, +0.0000] | +0.1238 | 0.2741 | +0.0000 [+0.0000, +0.0000] | +0.0801 | 0.0169 / 0.0169 |
| DER-MAPSOFT-G-XF | 0.1782 | -0.0015 [-0.0036, +0.0000] | 0.2725 | +0.0002 [-0.0007, +0.0015] | +0.1240 | 0.2744 | +0.0002 [-0.0005, +0.0010] | +0.0803 | 0.0168 / 0.0169 |
| SOFTG0 | 0.1797 | +0.0000 [+0.0000, +0.0000] | 0.2723 | +0.0000 [+0.0000, +0.0000] | +0.1238 | 0.2741 | +0.0000 [+0.0000, +0.0000] | +0.0801 | 0.0169 / 0.0169 |
| HUMAN | 0.0125 | - | 0.1485 | - | +0.0000 | 0.1941 | - | +0.0000 | 0.0071 / 0.0120 |

T2 / T3 are read RELATIVE to the HUMAN floor (the 'minus HUMAN' columns), never as an absolute bar. nocls (code 0) is UNKNOWN, not a violation (AMENDMENT 1); its share is the COVERAGE of the check.

## GT vs OWN gap per tier (S6-compliant; share of the GT-oracle gain the deployable map recovers)

| tier | metric | BASE | GT | OWN | DER | GT gain | OWN gain | OWN share of GT gain |
|---|---|---|---|---|---|---|---|---|
| T1 | off_drivable | 0.1792 | 0.0219 | 0.1723 | 0.2533 | 0.1573 | 0.0069 | 0.0437 |
| T1 | edge_contact | 0.2855 | 0.2217 | 0.2814 | 0.3412 | 0.0638 | 0.0041 | 0.0637 |
| T1 | lane_cross | 0.2939 | 0.2813 | 0.2984 | 0.3451 | 0.0126 | -0.0044 | -0.3500 |
| T12 | off_drivable | 0.1792 | 0.0200 | 0.1726 | 0.2539 | 0.1592 | 0.0066 | 0.0413 |
| T12 | edge_contact | 0.2855 | 0.0782 | 0.2814 | 0.3418 | 0.2073 | 0.0041 | 0.0196 |
| T12 | lane_cross | 0.2939 | 0.2898 | 0.2993 | 0.3445 | 0.0041 | -0.0054 | -1.3077 |
| T123 | off_drivable | 0.1792 | 0.0206 | 0.1739 | 0.2683 | 0.1585 | 0.0053 | 0.0335 |
| T123 | edge_contact | 0.2855 | 0.0775 | 0.2814 | 0.3502 | 0.2079 | 0.0041 | 0.0195 |
| T123 | lane_cross | 0.2939 | 0.1321 | 0.2990 | 0.3530 | 0.1618 | -0.0051 | -0.0313 |
| T1_CAL | off_drivable | 0.1792 | 0.0219 | 0.1764 | 0.2036 | 0.1573 | 0.0028 | 0.0179 |
| T1_CAL | edge_contact | 0.2855 | 0.2217 | 0.2839 | 0.2986 | 0.0638 | 0.0016 | 0.0245 |
| T1_CAL | lane_cross | 0.2939 | 0.2813 | 0.2939 | 0.3025 | 0.0126 | 0.0000 | 0.0000 |

## Verdict clauses (OWN arms)

| arm | C1 off-drivable separated below BASE | C2 no four-family row separated worse | C3 deranged does not beat BASE | C4 STOP guard | PASS |
|---|---|---|---|---|---|
| OWN-T1 | False (-0.0069 [-0.0169, +0.0035]) | False (S6c_stage:ade_m, S6c_stage:fde_m, S6c_stage:LON_along_mae_m, S6c_stage:LON_along_abs_final_m, S6c_stage:LAT_heading_mae_deg, S6c_stage:LAT_curvature_mae_1pm_valid) | True | False (path-len -0.0002 [-0.0010, +0.0007] m; stop-share +0.0000 [+0.0000, +0.0000]; guard rows worse: S6c_stage:LON_along_mae_m, S6c_stage:LON_along_abs_final_m) | **False** |
| OWN-CAL-T1 | False (-0.0028 [-0.0063, +0.0000]) | False (S2c_stage:LAT_curvature_mae_1pm_valid, S2c_stage:LAT_cross_abs_final_m, S2c_stage:TAC_traj_lat_correct) | True | True (path-len -0.0001 [-0.0005, +0.0001] m; stop-share +0.0000 [+0.0000, +0.0000]; guard rows worse: none) | **False** |
| OWN-T12 | False (-0.0066 [-0.0167, +0.0043]) | False (S6c_stage:ade_m, S6c_stage:fde_m, S6c_stage:LON_along_mae_m, S6c_stage:LON_along_abs_final_m, S6c_stage:LAT_heading_mae_deg, S6c_stage:LAT_curvature_mae_1pm_valid) | True | False (path-len -0.0002 [-0.0010, +0.0007] m; stop-share +0.0000 [+0.0000, +0.0000]; guard rows worse: S6c_stage:LON_along_mae_m, S6c_stage:LON_along_abs_final_m) | **False** |
| OWN-T123 | False (-0.0053 [-0.0159, +0.0054]) | False (S6c_stage:ade_m, S6c_stage:fde_m, S6c_stage:LON_along_mae_m, S6c_stage:LON_along_abs_final_m, S6c_stage:LAT_heading_mae_deg, S6c_stage:LAT_curvature_mae_1pm_valid) | True | False (path-len -0.0002 [-0.0010, +0.0007] m; stop-share +0.0000 [+0.0000, +0.0000]; guard rows worse: S6c_stage:LON_along_mae_m, S6c_stage:LON_along_abs_final_m) | **False** |

## STOP-trap guard (S6-compliant, EMITTED staged plan; delta = arm - BASE)

| arm | mean path length to 6 s (m) | delta vs BASE [CI] | stop share (6 s displacement < 1 m) | delta [CI] |
|---|---|---|---|---|
| BASE | 67.727 | - | 0.0261 | - |
| GT-T1 | 67.723 | -0.0040 [-0.0069, -0.0017] | 0.0261 | +0.0000 [+0.0000, +0.0000] |
| GT-T12 | 67.717 | -0.0096 [-0.0205, -0.0029] | 0.0261 | +0.0000 [+0.0000, +0.0000] |
| GT-T123 | 67.714 | -0.0127 [-0.0245, -0.0043] | 0.0261 | +0.0000 [+0.0000, +0.0000] |
| OWN-T1 | 67.726 | -0.0002 [-0.0010, +0.0007] | 0.0261 | +0.0000 [+0.0000, +0.0000] |
| OWN-T12 | 67.726 | -0.0002 [-0.0010, +0.0007] | 0.0261 | +0.0000 [+0.0000, +0.0000] |
| OWN-T123 | 67.726 | -0.0002 [-0.0010, +0.0007] | 0.0261 | +0.0000 [+0.0000, +0.0000] |
| DER-T1 | 67.725 | -0.0016 [-0.0039, +0.0002] | 0.0261 | +0.0000 [+0.0000, +0.0000] |
| DER-T12 | 67.725 | -0.0016 [-0.0039, +0.0002] | 0.0261 | +0.0000 [+0.0000, +0.0000] |
| DER-T123 | 67.724 | -0.0026 [-0.0056, -0.0002] | 0.0261 | +0.0000 [+0.0000, +0.0000] |
| OWN-CAL-T1 | 67.726 | -0.0001 [-0.0005, +0.0001] | 0.0261 | +0.0000 [+0.0000, +0.0000] |
| DER-CAL-T1 | 67.726 | -0.0010 [-0.0027, +0.0004] | 0.0261 | +0.0000 [+0.0000, +0.0000] |
| MAPSOFT-G-XF | 67.727 | +0.0000 [+0.0000, +0.0000] | 0.0261 | +0.0000 [+0.0000, +0.0000] |
| DER-MAPSOFT-G-XF | 67.727 | +0.0000 [-0.0000, +0.0001] | 0.0261 | +0.0000 [+0.0000, +0.0000] |
| SOFTG0 | 67.727 | +0.0000 [+0.0000, +0.0000] | 0.0261 | +0.0000 [+0.0000, +0.0000] |
| HUMAN | 66.793 | - | 0.0203 | - |

## CALIBRATED own-map arm (OWN-CAL-T1): tau = 0.8627451062202454; human paths passing at tau 0.9904580152671756, at the hard rule (P >= 0.502) 0.8208833151581243 (= 1 - the plan-level FALSE-ALARM rate of the hard own-map rule on human paths); n human paths 3668. Calibrated ON the eval human paths (privileged GT trajectories, no outcome metric); a train-side calibration is not available in this roll.

## GT-T1 is the T0-ORACLE BOUND for the learned-selector lever (a DrivableCritic / Hydra-MDP++ / DDv2-style selector with map-derived labels); OWN-* is what the head's own map delivers with NO training.

## Own-map tier increments (PREDICTED before the roll: OWN-T12 ~ OWN-T1, the own edge channel is empty)

{
 "OWN-T12_vs_OWN-T1_changed_pick_share": 0.0023138409760201937,
 "OWN-T123_vs_OWN-T12_changed_pick_share": 0.013462347496844763,
 "GT-T12_vs_GT-T1_changed_pick_share": 0.1379890618426588,
 "GT-T123_vs_GT-T12_changed_pick_share": 0.14976861590239798,
 "note": "PREDICTED before the roll (prereg sec. 3.4): the own edge channel is empty (IoU 0.001) and lane IoU is 0.2 -> OWN-T12 ~ OWN-T1 (< 2 % of windows differ)"
}


## Four families -- worst rows per arm (S6c and S2c, +stage, delta = arm - BASE; 'worse' = CI excludes 0 on the bad side)

* S6c_stage GT-T1: worse rows none; separated (other direction or unsigned) ['speed_err_signed_mps@GTcruise']
* S6c_stage GT-T12: worse rows ['LAT_curvature_mae_1pm_valid']; separated (other direction or unsigned) ['speed_err_signed_mps@GTcruise']
* S6c_stage GT-T123: worse rows ['LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid']; separated (other direction or unsigned) ['along_signed_final_m@GTcruise', 'speed_err_signed_mps@GTcruise']
* S6c_stage OWN-T1: worse rows ['ade_m', 'fde_m', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m']; separated (other direction or unsigned) ['along_signed_final_m@GTcruise']
* S6c_stage OWN-T12: worse rows ['ade_m', 'fde_m', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) ['along_signed_final_m@GTcruise']
* S6c_stage OWN-T123: worse rows ['ade_m', 'fde_m', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) ['along_signed_final_m@GTcruise']
* S6c_stage DER-T1: worse rows ['ade_m', 'fde_m', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) ['along_signed_final_m@GTcruise']
* S6c_stage DER-T12: worse rows ['ade_m', 'fde_m', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) ['along_signed_final_m@GTcruise']
* S6c_stage DER-T123: worse rows ['ade_m', 'fde_m', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) ['along_signed_final_m@GTcruise']
* S6c_stage OWN-CAL-T1: worse rows none; separated (other direction or unsigned) ['along_signed_final_m@GTcruise']
* S6c_stage DER-CAL-T1: worse rows ['ade_m', 'fde_m', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) ['along_signed_final_m@GTcruise']
* S6c_stage MAPSOFT-G-XF: worse rows none; separated (other direction or unsigned) none
* S6c_stage DER-MAPSOFT-G-XF: worse rows ['LON_along_mae_m', 'LON_along_abs_final_m']; separated (other direction or unsigned) ['along_signed_final_m@GTcruise']
* S6c_stage SOFTG0: worse rows none; separated (other direction or unsigned) none
* S2c_stage GT-T1: worse rows ['LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid']; separated (other direction or unsigned) ['LON_accel_mae_mps2']
* S2c_stage GT-T12: worse rows ['ade_m', 'fde_m', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) none
* S2c_stage GT-T123: worse rows ['ade_m', 'fde_m', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) none
* S2c_stage OWN-T1: worse rows ['ade_m', 'fde_m', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) none
* S2c_stage OWN-T12: worse rows ['ade_m', 'fde_m', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) none
* S2c_stage OWN-T123: worse rows ['ade_m', 'fde_m', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) none
* S2c_stage DER-T1: worse rows ['ade_m', 'fde_m', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) none
* S2c_stage DER-T12: worse rows ['ade_m', 'fde_m', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) none
* S2c_stage DER-T123: worse rows ['ade_m', 'fde_m', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) none
* S2c_stage OWN-CAL-T1: worse rows ['LAT_curvature_mae_1pm_valid', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) none
* S2c_stage DER-CAL-T1: worse rows ['ade_m', 'fde_m', 'LON_along_mae_m', 'LON_along_abs_final_m', 'LAT_heading_mae_deg', 'LAT_curvature_mae_1pm_valid', 'LAT_yaw_rate_mae_radps_valid', 'LAT_cross_mae_m', 'LAT_cross_abs_final_m', 'TAC_traj_lat_correct']; separated (other direction or unsigned) none
* S2c_stage MAPSOFT-G-XF: worse rows none; separated (other direction or unsigned) none
* S2c_stage DER-MAPSOFT-G-XF: worse rows none; separated (other direction or unsigned) none
* S2c_stage SOFTG0: worse rows none; separated (other direction or unsigned) none

## Availability of a tier-compliant eligible candidate in the 149-fan (share of windows whose set did not need the fallback)

{
 "gt": {
  "T1_set_nonempty": 0.9829617164493059,
  "T12_after_T1": 0.9322675641564998,
  "T123_after_T12": 0.8933529659234329
 },
 "own": {
  "T1_set_nonempty": 0.9167017248632731,
  "T12_after_T1": 0.9993689524610854,
  "T123_after_T12": 0.9976861590239798
 },
 "der": {
  "T1_set_nonempty": 0.878628523348759,
  "T12_after_T1": 0.9985275557425326,
  "T123_after_T12": 0.987379049221708
 },
 "cal": {
  "T1_set_nonempty": 0.9993689524610854
 }
}

## Candidate-level (all windows): share of windows whose pick changed; fallback shares (T1, T2, T3)

* BASE: changed 0.0000; fallbacks [0.0, 0.0, 0.0]; kept-set median 143.0
* GT-T1: changed 0.1611; fallbacks [0.01703828355069415, 0.0, 0.0]; kept-set median 58.0
* GT-T12: changed 0.2474; fallbacks [0.01703828355069415, 0.06773243584350021, 0.0]; kept-set median 49.0
* GT-T123: changed 0.3519; fallbacks [0.01703828355069415, 0.06773243584350021, 0.1066470340765671]; kept-set median 30.0
* OWN-T1: changed 0.0860; fallbacks [0.08329827513672697, 0.0, 0.0]; kept-set median 81.0
* OWN-T12: changed 0.0879; fallbacks [0.08329827513672697, 0.0006310475389145982, 0.0]; kept-set median 80.0
* OWN-T123: changed 0.0999; fallbacks [0.08329827513672697, 0.0006310475389145982, 0.0023138409760201937]; kept-set median 76.0
* DER-T1: changed 0.2610; fallbacks [0.12137147665124107, 0.0, 0.0]; kept-set median 81.0
* DER-T12: changed 0.2636; fallbacks [0.12137147665124107, 0.001472444257467396, 0.0]; kept-set median 81.0
* DER-T123: changed 0.3061; fallbacks [0.12137147665124107, 0.001472444257467396, 0.012620950778291964]; kept-set median 68.0
* OWN-CAL-T1: changed 0.0093; fallbacks [0.0006310475389145982, 0.0, 0.0]; kept-set median 123.0
* DER-CAL-T1: changed 0.0999; fallbacks [0.007782919646613378, 0.0, 0.0]; kept-set median 125.0
* MAPSOFT-G-XF: changed 0.0000; fallbacks [0.0, 0.0, 0.0]; kept-set median 143.0
* DER-MAPSOFT-G-XF: changed 0.0076; fallbacks [0.0, 0.0, 0.0]; kept-set median 143.0
* SOFTG0: changed 0.0000; fallbacks [0.0, 0.0, 0.0]; kept-set median 143.0

## NEXT LEVER (mechanical, PREREG sec. 6.3)

GT-T1 itself does not clear C1/C2/C4: even with the true map, compliance-first selection costs accuracy or progress (or does not lower off-drivable). The fan/selector, not the map channel, is the bottleneck -> next: a LEARNED listwise selector with map-derived labels over the same fan (not a rule), and a fan-availability audit (availability block).

{
 "gt_T1_clears_C1_C2_C4": false,
 "own_T1_captures_share_of_gt_T1_gain": 0.04373757455268384,
 "own_cal_T1_captures_share": 0.017892644135188842
}


## Evidence class: MEASURED (raw artifacts: MAPSEL_RESULT.json, features_s0.npz). Not computed: distance-keeping and STRATEGIC rows (no lead track in the banked dumps / --no-strategic; n = 0).