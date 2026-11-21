# VERDICT OBJSLOT stage 0 (PREREG_V8F_PROOF_ADD20)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25). Zero training; CPU; the banked states and labels.
Ego speeds of the rollout are the logged ones (the action input of a rollout, teacher-forced as in every T0 read). Definitions: OBJSLOT_DEFS.md.

## Controls (first)

- o1_planted_series: PASS :: {"r2_skill": 1.0, "r1_skill": 0.8570436702851005, "n": 1948, "n_planted": 1948, "mean_ego_speed_std_over_window": 0.3094252430354883, "tolerance": 1e-09, "report_only_strict_constant_lead_speed_history_r2_skill": 0.9762532705882924, "n_strict": 1948}
- o3_identity_label: PASS :: {"copy_skill": 0.0, "true_as_rule_skill": 1.0}
- o5_leak_detector: PASS :: {"r1_skill": 0.6915802852415867, "r1_shifted_history_skill": 0.7251984026606367, "n": 413}
- o6_rule_inputs_label: PASS :: {"signatures": {"rule_inputs": ["g0", "c", "V", "K"], "rule_preds": ["inp"], "ls_slope": ["G", "mask"]}}
- o2_shuffled_target_label: PASS :: {"bar": 0.02, "skill_by_rule": {"R1": -0.0046293435089883594, "R2": 0.0030828529904611823, "R3": -0.0005728433620775242}}
- o4_probe_fidelity|P_s0: PASS :: {"level_r2_vision_R0": 0.47819418045649864, "banked_level_r2_ii": 0.47819418045649864, "n": 471, "tol": 0.005, "max_abs_diff_of_the_two_gap_values": 1.4210854715202004e-14}
- o6_no_future_input|P_s0: PASS :: {"max_abs_change_when_states_after_frame_4_are_random": 0.0, "max_abs_change_of_R3_when_heldout_fold_targets_are_random": 0.0}
- o3_identity|P_s0: PASS :: {"copy_skill": 0.0, "true_as_rule_skill": 1.0}
- o2_shuffled_target|P_s0: PASS :: {"bar": 0.02, "skill_by_rule": {"R1": -0.12002594187111493, "R2": -0.1358333352427865, "R3": -0.015133150675719742}}
- o4_probe_fidelity|P_s1: PASS :: {"level_r2_vision_R0": 0.4783835756345852, "banked_level_r2_ii": 0.4783835756345851, "n": 471, "tol": 0.005, "max_abs_diff_of_the_two_gap_values": 7.673861546209082e-13}
- o6_no_future_input|P_s1: PASS :: {"max_abs_change_when_states_after_frame_4_are_random": 0.0, "max_abs_change_of_R3_when_heldout_fold_targets_are_random": 0.0}
- o3_identity|P_s1: PASS :: {"copy_skill": 0.0, "true_as_rule_skill": 1.0}
- o2_shuffled_target|P_s1: PASS :: {"bar": 0.02, "skill_by_rule": {"R1": -0.14984813287553056, "R2": -0.16496319127901704, "R3": -0.0008958524693003511}}
- o4_probe_fidelity|G_s0: PASS :: {"level_r2_vision_R0": 0.4913039673189592, "banked_level_r2_ii": 0.4913039673189592, "n": 471, "tol": 0.005, "max_abs_diff_of_the_two_gap_values": 2.4868995751603507e-14}
- o6_no_future_input|G_s0: PASS :: {"max_abs_change_when_states_after_frame_4_are_random": 0.0, "max_abs_change_of_R3_when_heldout_fold_targets_are_random": 0.0}
- o3_identity|G_s0: PASS :: {"copy_skill": 0.0, "true_as_rule_skill": 1.0}
- o2_shuffled_target|G_s0: PASS :: {"bar": 0.02, "skill_by_rule": {"R1": -0.1355567440474863, "R2": -0.151648591950984, "R3": -0.02773355749814721}}
- o4_probe_fidelity|G_s1: PASS :: {"level_r2_vision_R0": 0.4698345412480035, "banked_level_r2_ii": 0.46983454124800283, "n": 471, "tol": 0.005, "max_abs_diff_of_the_two_gap_values": 9.663381206337363e-13}
- o6_no_future_input|G_s1: PASS :: {"max_abs_change_when_states_after_frame_4_are_random": 0.0, "max_abs_change_of_R3_when_heldout_fold_targets_are_random": 0.0}
- o3_identity|G_s1: PASS :: {"copy_skill": 0.0, "true_as_rule_skill": 1.0}
- o2_shuffled_target|G_s1: PASS :: {"bar": 0.02, "skill_by_rule": {"R1": -0.1390764352699263, "R2": -0.15968926773427472, "R3": -0.0040228437855927}}
- controls_pass: True

## LABEL source (privileged ceiling; one result)

Windows 1948; lead at S0 and at frame 14: 471; dropped by the continuity rule: 42 (jump above 3.0 m: 38; gap missing in between: 4); scored: 429; with the history rule (>= 3 labelled frames): 413 (dropped 16); episodes 48.

- R1: skill over the label copy +0.692 [+0.588, +0.766] n=413 (separated above 0); level R2 +0.941 [+0.915, +0.959]
- R2: skill over the label copy +0.750 [+0.664, +0.811] n=413 (separated above 0); level R2 +0.952 [+0.935, +0.964]
- R3: skill over the label copy +0.745 [+0.668, +0.803] n=413 (separated above 0); level R2 +0.952 [+0.932, +0.964]
- R0 (copy): skill 0 by definition; level R2 +0.810 [+0.725, +0.867]

## VISION source (grid state, level probe of the r-b read; frames 0 to 4 only)

### P_s0

scored windows 429 of 1948; episodes 48

- R1: skill over the vision copy -0.189 [-0.361, -0.064] n=429 (separated below 0); level R2 +0.457 [+0.272, +0.571]
- R2: skill over the vision copy -0.236 [-0.386, -0.125] n=429 (separated below 0); level R2 +0.436 [+0.238, +0.562]
- R3: skill over the vision copy +0.011 [-0.059, +0.080] n=429; level R2 +0.549 [+0.401, +0.650]
- R0 (vision copy): level R2 +0.544 [+0.389, +0.648]
- R3 level R2 +0.549 minus banked rollout (i) [R1] +0.556: -0.007 [-0.055, +0.049] (n=429)
- R3 level R2 +0.549 minus banked rollout (i) [R3h] +0.567: -0.018 [-0.050, +0.018] (n=429)
- R3 level R2 +0.549 minus banked rollout (i) [R3g] +0.565: -0.016 [-0.049, +0.020] (n=429)

### P_s1

scored windows 429 of 1948; episodes 48

- R1: skill over the vision copy -0.281 [-0.486, -0.131] n=429 (separated below 0); level R2 +0.440 [+0.271, +0.545]
- R2: skill over the vision copy -0.329 [-0.509, -0.193] n=429 (separated below 0); level R2 +0.419 [+0.242, +0.535]
- R3: skill over the vision copy +0.003 [-0.080, +0.084] n=429; level R2 +0.565 [+0.430, +0.656]
- R0 (vision copy): level R2 +0.563 [+0.428, +0.657]
- R3 level R2 +0.565 minus banked rollout (i) [R1] +0.571: -0.006 [-0.058, +0.048] (n=429)
- R3 level R2 +0.565 minus banked rollout (i) [R3h] +0.587: -0.023 [-0.062, +0.016] (n=429)
- R3 level R2 +0.565 minus banked rollout (i) [R3g] +0.588: -0.024 [-0.061, +0.013] (n=429)

### G_s0

scored windows 429 of 1948; episodes 48

- R1: skill over the vision copy -0.353 [-0.579, -0.175] n=429 (separated below 0); level R2 +0.409 [+0.164, +0.553]
- R2: skill over the vision copy -0.407 [-0.604, -0.243] n=429 (separated below 0); level R2 +0.386 [+0.125, +0.535]
- R3: skill over the vision copy -0.007 [-0.087, +0.067] n=429; level R2 +0.560 [+0.402, +0.660]
- R0 (vision copy): level R2 +0.564 [+0.409, +0.664]
- R3 level R2 +0.560 minus banked rollout (i) [R1] +0.508: +0.053 [-0.031, +0.139] (n=429)
- R3 level R2 +0.560 minus banked rollout (i) [R3h] +0.518: +0.042 [-0.030, +0.118] (n=429)
- R3 level R2 +0.560 minus banked rollout (i) [R3g] +0.518: +0.043 [-0.031, +0.119] (n=429)

### G_s1

scored windows 429 of 1948; episodes 48

- R1: skill over the vision copy -0.242 [-0.421, -0.086] n=429 (separated below 0); level R2 +0.439 [+0.261, +0.551]
- R2: skill over the vision copy -0.304 [-0.468, -0.160] n=429 (separated below 0); level R2 +0.411 [+0.218, +0.536]
- R3: skill over the vision copy +0.027 [-0.070, +0.123] n=429; level R2 +0.560 [+0.425, +0.650]
- R0 (vision copy): level R2 +0.548 [+0.412, +0.640]
- R3 level R2 +0.560 minus banked rollout (i) [R1] +0.484: +0.076 [-0.034, +0.204] (n=429)
- R3 level R2 +0.560 minus banked rollout (i) [R3h] +0.512: +0.048 [-0.047, +0.150] (n=429)
- R3 level R2 +0.560 minus banked rollout (i) [R3g] +0.512: +0.048 [-0.046, +0.151] (n=429)

## Outcomes (card section 4)

- Label: **KIN-CEILING-HOLDS**
- Vision, arm type P (both seeds needed): **SLOT-FLAT**
- Vision, arm type G (both seeds needed): **SLOT-FLAT**
- Share of the label ceiling (vision R3 skill / label R2 skill; different window sets, decides nothing): {"P_s0": 0.014953740890001718, "P_s1": 0.004350145745115114, "G_s0": -0.009829084573985345, "G_s1": 0.03634005905811486}

## Horizon profile (decides nothing; skill over the copy of the same source)

| horizon s | source | R1 | R2 | R3 |
|---|---|---|---|---|
| 0.4 | LABEL (n=477) | +0.807 [+0.722, +0.864] | +0.818 [+0.734, +0.875] | +0.819 [+0.735, +0.876] |
| 0.4 | P_s0 (n=502) | -0.034 [-0.062, -0.009] | -0.033 [-0.060, -0.009] | -0.039 [-0.067, -0.008] |
| 0.4 | P_s1 (n=502) | -0.041 [-0.073, -0.015] | -0.041 [-0.072, -0.015] | -0.037 [-0.067, -0.001] |
| 0.4 | G_s0 (n=502) | -0.037 [-0.064, -0.014] | -0.036 [-0.062, -0.014] | -0.044 [-0.068, -0.021] |
| 0.4 | G_s1 (n=502) | -0.032 [-0.067, -0.003] | -0.033 [-0.066, -0.004] | -0.048 [-0.097, +0.002] |
| 0.8 | LABEL (n=459) | +0.797 [+0.710, +0.859] | +0.821 [+0.734, +0.879] | +0.818 [+0.736, +0.877] |
| 0.8 | P_s0 (n=480) | -0.082 [-0.140, -0.034] | -0.080 [-0.133, -0.035] | -0.046 [-0.084, -0.007] |
| 0.8 | P_s1 (n=480) | -0.106 [-0.179, -0.049] | -0.103 [-0.173, -0.051] | -0.038 [-0.075, +0.003] |
| 0.8 | G_s0 (n=480) | -0.108 [-0.168, -0.057] | -0.105 [-0.162, -0.059] | -0.054 [-0.085, -0.027] |
| 0.8 | G_s1 (n=480) | -0.089 [-0.166, -0.026] | -0.089 [-0.161, -0.030] | -0.047 [-0.108, +0.014] |
| 1.2 | LABEL (n=443) | +0.780 [+0.700, +0.841] | +0.819 [+0.742, +0.874] | +0.812 [+0.735, +0.867] |
| 1.2 | P_s0 (n=460) | -0.099 [-0.182, -0.030] | -0.101 [-0.175, -0.039] | -0.032 [-0.074, +0.014] |
| 1.2 | P_s1 (n=460) | -0.160 [-0.277, -0.074] | -0.161 [-0.269, -0.083] | -0.037 [-0.079, +0.011] |
| 1.2 | G_s0 (n=460) | -0.201 [-0.316, -0.110] | -0.203 [-0.312, -0.118] | -0.050 [-0.088, -0.013] |
| 1.2 | G_s1 (n=460) | -0.135 [-0.235, -0.044] | -0.142 [-0.239, -0.057] | -0.031 [-0.100, +0.041] |
| 1.6 | LABEL (n=428) | +0.735 [+0.640, +0.801] | +0.786 [+0.701, +0.843] | +0.778 [+0.700, +0.835] |
| 1.6 | P_s0 (n=444) | -0.165 [-0.290, -0.064] | -0.180 [-0.290, -0.093] | -0.018 [-0.072, +0.037] |
| 1.6 | P_s1 (n=444) | -0.258 [-0.450, -0.122] | -0.274 [-0.439, -0.159] | -0.019 [-0.080, +0.042] |
| 1.6 | G_s0 (n=444) | -0.337 [-0.520, -0.191] | -0.350 [-0.517, -0.227] | -0.037 [-0.090, +0.016] |
| 1.6 | G_s1 (n=444) | -0.206 [-0.360, -0.080] | -0.232 [-0.371, -0.112] | -0.010 [-0.097, +0.073] |
| 2.0 | LABEL (n=413) | +0.692 [+0.588, +0.766] | +0.750 [+0.664, +0.811] | +0.745 [+0.668, +0.803] |
| 2.0 | P_s0 (n=429) | -0.189 [-0.361, -0.064] | -0.236 [-0.386, -0.125] | +0.011 [-0.059, +0.080] |
| 2.0 | P_s1 (n=429) | -0.281 [-0.486, -0.131] | -0.329 [-0.509, -0.193] | +0.003 [-0.080, +0.084] |
| 2.0 | G_s0 (n=429) | -0.353 [-0.579, -0.175] | -0.407 [-0.604, -0.243] | -0.007 [-0.087, +0.067] |
| 2.0 | G_s1 (n=429) | -0.242 [-0.421, -0.086] | -0.304 [-0.468, -0.160] | +0.027 [-0.070, +0.123] |

## Limits

- The scored set is small and one training run per seed; a separated interval answers another draw of episodes only (the vision outcome needs both seeds).
- The probe is the level probe of the r-b read, fitted on levels and not on changes; vision history uses all five frames, presence is not a vision output.
- R2 and R3 use the logged ego speeds of the future frames (the action input of a rollout); that is not a goal input and carries nothing of the situation classifier.
- Label and vision rules are compared with their own copy on their own windows; the share of the ceiling compares two different window sets.
