# VERDICT LEADHEAD stage 0b (PREREG_V8F_PROOF_ADD23)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25). Zero training of the model; CPU; the banked train-probe states and labels. Heads fitted on the train probe set (1,200 training clips), scored on the 429 held-out windows of card ADD21 (episode-cluster bootstrap, B 2000, rng 0).
Stages run in this file: ht, lc, hm, pooled. Definitions: LEADHEAD_B_DEFS.md. Ego speeds of the rules are the logged ones (the action input of a rollout).

Train probe set: 121166 rows, 121166 with a finite kinematic vector, 1200 clips; frame step 0.2 s (stride 1: True).

## Controls (first)

- j5_planted_state_train_layout: PASS :: {"heldout_r2_closing": 0.9901983127137075, "bar": 0.9, "n_fit": 104726, "n_heldout": 11640, "noise_sd_per_dim": 0.8641232851171319, "amp_per_metre": 10.0, "chosen_g": 1.0, "layout": "all kept rows and clips of the train probe set, synthetic linear gap per clip"}
- j6_label_path: PASS :: {"r2_skill_this_path": 0.7499068112986798, "r2_skill_objslot_label_pipeline_now": 0.7499068112986798, "r2_skill_banked_ADD20": 0.7499068112986798, "n": 413, "tol": 1e-06}
- j1_disjoint_sets: PASS :: {"n_fit_ids": 1200, "n_eval_ids": 139, "n_common": 0, "common_first5": [], "fit_ids": "the 1,200 train probe clip ids", "eval_ids": "scored-window episode ids and the eval-bank episode ids"}
- j2_alignment|P_s0: PASS :: {"clips_equal": true, "n_rows_state": 121166, "n_rows_probe": 121166, "rows_equal": true, "clip_idx_equal": true, "t_equal": true, "z_is_2d": true, "z_finite": true, "bar": "clips, clip_idx and t equal the probe set exactly; state finite"}
- j3_shuffled_labels|HT|P_s0: PASS :: {"heldout_r2_closing_shuffled_labels": -0.006950176676657094, "bar": 0.02, "n": 413, "chosen_g": 100000.0}
- j7_no_future_input|HT|P_s0: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0}
- j4_in_sample_ceiling|HT|P_s0: PASS :: {"r2_in_sample_closing_fit_clips": 0.22948922047599474, "r2_closing_heldout_tenth_of_train_clips": 0.08896949134011545, "n_fit": 28047, "n_heldout": 2818, "chosen_g": 10000.0, "clips_heldout": 120}
- j8_transfer|HT|P_s0: PASS :: {"reported_not_a_gate": true, "gap_r2_heldout_tenth_of_train_clips": 0.5648358871438484, "gap_r2_eval_windows_same_90pct_head": 0.5933940559062911, "gap_r2_eval_windows_main_head": 0.6001484850158865, "shift_tenth_minus_eval_windows": -0.02855816876244277, "closing_r2_heldout_tenth": 0.08896949134011545, "closing_r2_eval_windows_same_90pct_head": 0.143048519691938, "n_heldout_gap_rows": 2935, "n_heldout_closing_rows": 2818, "chosen_g_gap_90": 1000.0}
- j3_shuffled_labels|HM|P_s0: PASS :: {"heldout_r2_closing_shuffled_labels": -0.17978593688289313, "bar": 0.02, "n": 413}
- j7_no_future_input|HM|P_s0: PASS :: {"max_abs_change_of_head_outputs": 0.0}
- j2_alignment|P_s1: PASS :: {"clips_equal": true, "n_rows_state": 121166, "n_rows_probe": 121166, "rows_equal": true, "clip_idx_equal": true, "t_equal": true, "z_is_2d": true, "z_finite": true, "bar": "clips, clip_idx and t equal the probe set exactly; state finite"}
- j3_shuffled_labels|HT|P_s1: PASS :: {"heldout_r2_closing_shuffled_labels": -0.00266297482386868, "bar": 0.02, "n": 413, "chosen_g": 1000000.0}
- j7_no_future_input|HT|P_s1: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0}
- j4_in_sample_ceiling|HT|P_s1: PASS :: {"r2_in_sample_closing_fit_clips": 0.24080046157143586, "r2_closing_heldout_tenth_of_train_clips": 0.06952660465469152, "n_fit": 28047, "n_heldout": 2818, "chosen_g": 10000.0, "clips_heldout": 120}
- j8_transfer|HT|P_s1: PASS :: {"reported_not_a_gate": true, "gap_r2_heldout_tenth_of_train_clips": 0.5480943978556911, "gap_r2_eval_windows_same_90pct_head": 0.5381640845287738, "gap_r2_eval_windows_main_head": 0.548905738522275, "shift_tenth_minus_eval_windows": 0.009930313326917295, "closing_r2_heldout_tenth": 0.06952660465469152, "closing_r2_eval_windows_same_90pct_head": 0.16062477672881703, "n_heldout_gap_rows": 2935, "n_heldout_closing_rows": 2818, "chosen_g_gap_90": 1000.0}
- j3_shuffled_labels|HM|P_s1: PASS :: {"heldout_r2_closing_shuffled_labels": -0.38868889072825596, "bar": 0.02, "n": 413}
- j7_no_future_input|HM|P_s1: PASS :: {"max_abs_change_of_head_outputs": 0.0}
- j2_alignment|G_s0: PASS :: {"clips_equal": true, "n_rows_state": 121166, "n_rows_probe": 121166, "rows_equal": true, "clip_idx_equal": true, "t_equal": true, "z_is_2d": true, "z_finite": true, "bar": "clips, clip_idx and t equal the probe set exactly; state finite"}
- j3_shuffled_labels|HT|G_s0: PASS :: {"heldout_r2_closing_shuffled_labels": -0.00689297793209831, "bar": 0.02, "n": 413, "chosen_g": 100000.0}
- j7_no_future_input|HT|G_s0: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0}
- j4_in_sample_ceiling|HT|G_s0: PASS :: {"r2_in_sample_closing_fit_clips": 0.2427447445093769, "r2_closing_heldout_tenth_of_train_clips": 0.09800196278645446, "n_fit": 28047, "n_heldout": 2818, "chosen_g": 10000.0, "clips_heldout": 120}
- j8_transfer|HT|G_s0: PASS :: {"reported_not_a_gate": true, "gap_r2_heldout_tenth_of_train_clips": 0.5460922845204315, "gap_r2_eval_windows_same_90pct_head": 0.6281590165890536, "gap_r2_eval_windows_main_head": 0.6199299283712667, "shift_tenth_minus_eval_windows": -0.08206673206862214, "closing_r2_heldout_tenth": 0.09800196278645446, "closing_r2_eval_windows_same_90pct_head": 0.10522605255191197, "n_heldout_gap_rows": 2935, "n_heldout_closing_rows": 2818, "chosen_g_gap_90": 1000.0}
- j3_shuffled_labels|HM|G_s0: PASS :: {"heldout_r2_closing_shuffled_labels": -0.36767151417543675, "bar": 0.02, "n": 413}
- j7_no_future_input|HM|G_s0: PASS :: {"max_abs_change_of_head_outputs": 0.0}
- j2_alignment|G_s1: PASS :: {"clips_equal": true, "n_rows_state": 121166, "n_rows_probe": 121166, "rows_equal": true, "clip_idx_equal": true, "t_equal": true, "z_is_2d": true, "z_finite": true, "bar": "clips, clip_idx and t equal the probe set exactly; state finite"}
- j3_shuffled_labels|HT|G_s1: PASS :: {"heldout_r2_closing_shuffled_labels": -0.004235432321754162, "bar": 0.02, "n": 413, "chosen_g": 100000.0}
- j7_no_future_input|HT|G_s1: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0}
- j4_in_sample_ceiling|HT|G_s1: PASS :: {"r2_in_sample_closing_fit_clips": 0.23952077255796178, "r2_closing_heldout_tenth_of_train_clips": 0.08500494510435674, "n_fit": 28047, "n_heldout": 2818, "chosen_g": 10000.0, "clips_heldout": 120}
- j8_transfer|HT|G_s1: PASS :: {"reported_not_a_gate": true, "gap_r2_heldout_tenth_of_train_clips": 0.5757780525277177, "gap_r2_eval_windows_same_90pct_head": 0.4912311031734292, "gap_r2_eval_windows_main_head": 0.5350524207180497, "shift_tenth_minus_eval_windows": 0.08454694935428853, "closing_r2_heldout_tenth": 0.08500494510435674, "closing_r2_eval_windows_same_90pct_head": 0.1875784875215687, "n_heldout_gap_rows": 2935, "n_heldout_closing_rows": 2818, "chosen_g_gap_90": 1000.0}
- j3_shuffled_labels|HM|G_s1: PASS :: {"heldout_r2_closing_shuffled_labels": -0.2787523269673289, "bar": 0.02, "n": 413}
- j7_no_future_input|HM|G_s1: PASS :: {"max_abs_change_of_head_outputs": 0.0}
- controls_pass: True

## Per arm

### P_s0

Scored windows 429 of 1948 (episodes 48); with the label history rule 413 (dropped 16).
Fit rows (train probe set, 121166 kept rows): with five frames 116366; gap 32025 (548 clips); closing 30865 (525 clips); direct 24533 (484 clips).

#### HT

Chosen penalty multiplier g (gap / closing / direct): 1000 / 10000 / 10000; at the grid edge (1 or 1e7): False / False / False.
Fit rows used: gap 32025, closing 30865, direct 24533. Inner-fold mean squared error (fit rows only): gap 109.248, closing speed 8.0154, direct 24.3904.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 +0.600 [+0.376, +0.732]; RMSE 6.49 m [+5.570, +7.395]
- closing speed at S0: R2 +0.126 [-0.048, +0.227]; RMSE 2.14 m/s [+1.853, +2.449]
- ADD21 beside it (heads on the eval bank, 1689 to 1926 fit rows): gap R2 +0.523, closing R2 -0.094; R3 skill -0.045; direct skill -0.039

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy +0.047 [-0.036, +0.121] n=429; level R2 +0.375 [+0.066, +0.565]
- R2: skill over the head copy -0.055 [-0.137, +0.022] n=429; level R2 +0.309 [-0.043, +0.515]
- R3: skill over the head copy +0.095 [-0.040, +0.217] n=429; level R2 +0.407 [+0.129, +0.588]
- R0 (head copy): level R2 +0.345 [+0.030, +0.546]
- R3 level R2 +0.407 minus banked rollout (i) [R1 file] +0.556: -0.149 [-0.350, +0.009] (n=429)
- Direct read (gap + predicted change over 2.0 s, ego term included) skill over the head copy: +0.100 [+0.023, +0.170] n=429 (separated above 0); RMSE 7.71 m [+6.664, +8.873] against the copy RMSE 8.13 m

#### HM

Fit rows used: gap 32025, closing 30865, direct 24533. Inner-fold mean squared error (fit rows only): gap 154.686, closing speed 11.8815, direct 33.6583.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 -0.017 [-0.712, +0.344]; RMSE 10.34 m [+8.682, +11.978]
- closing speed at S0: R2 -0.469 [-0.989, -0.077]; RMSE 2.77 m/s [+2.449, +3.106]
- ADD21 beside it (heads on the eval bank, 1689 to 1926 fit rows): gap R2 +0.227, closing R2 -0.275; R3 skill -0.065; direct skill -0.138

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy -0.078 [-0.251, +0.079] n=429; level R2 -0.231 [-0.906, +0.175]
- R2: skill over the head copy -0.089 [-0.264, +0.069] n=429; level R2 -0.243 [-0.907, +0.159]
- R3: skill over the head copy -0.051 [-0.148, +0.027] n=429; level R2 -0.200 [-0.849, +0.162]
- R0 (head copy): level R2 -0.142 [-0.802, +0.218]
- R3 level R2 -0.200 minus banked rollout (i) [R1 file] +0.556: -0.756 [-1.293, -0.431] (separated below 0) (n=429)
- Direct read (gap + predicted change over 2.0 s, ego term included) skill over the head copy: +0.060 [-0.095, +0.184] n=429; RMSE 10.40 m [+9.104, +11.677] against the copy RMSE 10.73 m

Selection by the inner folds of the fit rows: closing-speed head = HT; direct head = HT.

#### Learning curve of HT (secondary a; clips drawn with rng 0; whole clips until the labelled rows reach the size; decides nothing)

| labelled fit rows (gap / closing) | closing R2 on scored windows | gap R2 | g (gap / closing) |
|---|---|---|---|
| 2027 / 2032 | -0.155 [-0.435, -0.007] | -0.093 [-0.702, +0.272] | 1000 / 100000 |
| 8054 / 8067 | +0.011 [-0.084, +0.061] | +0.527 [+0.212, +0.715] | 10000 / 100000 |
| 16017 / 16041 | +0.062 [-0.106, +0.173] | +0.581 [+0.291, +0.744] | 10000 / 10000 |
| 32025 / 30865 | +0.126 [-0.048, +0.227] | +0.600 [+0.376, +0.732] | 1000 / 10000 |
| 32025 / 30865 | +0.126 [-0.048, +0.227] | +0.600 [+0.376, +0.732] | 1000 / 10000 |

#### Pooled fit of HT (secondary b: train probe rows and eval-bank rows of the training folds together, cross-fitted over the eval folds; decides nothing)

- gap at S0: R2 +0.602 [+0.372, +0.737]; closing speed at S0: R2 +0.131 [-0.031, +0.228]
- R3 skill over the head copy +0.090 [-0.041, +0.212] n=429; R2 skill -0.057 [-0.132, +0.019] n=429; R1 skill +0.045 [-0.031, +0.118] n=429

### P_s1

Scored windows 429 of 1948 (episodes 48); with the label history rule 413 (dropped 16).
Fit rows (train probe set, 121166 kept rows): with five frames 116366; gap 32025 (548 clips); closing 30865 (525 clips); direct 24533 (484 clips).

#### HT

Chosen penalty multiplier g (gap / closing / direct): 1000 / 10000 / 10000; at the grid edge (1 or 1e7): False / False / False.
Fit rows used: gap 32025, closing 30865, direct 24533. Inner-fold mean squared error (fit rows only): gap 110.251, closing speed 7.8908, direct 24.1446.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 +0.549 [+0.260, +0.712]; RMSE 6.89 m [+5.805, +8.109]
- closing speed at S0: R2 +0.145 [-0.020, +0.245]; RMSE 2.12 m/s [+1.851, +2.397]
- ADD21 beside it (heads on the eval bank, 1689 to 1926 fit rows): gap R2 +0.605, closing R2 -0.108; R3 skill -0.018; direct skill -0.060

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy +0.061 [-0.018, +0.150] n=429; level R2 +0.263 [-0.124, +0.498]
- R2: skill over the head copy -0.050 [-0.119, +0.026] n=429; level R2 +0.176 [-0.231, +0.430]
- R3: skill over the head copy +0.099 [-0.024, +0.226] n=429; level R2 +0.293 [-0.051, +0.508]
- R0 (head copy): level R2 +0.216 [-0.187, +0.457]
- R3 level R2 +0.293 minus banked rollout (i) [R1 file] +0.571: -0.277 [-0.553, -0.105] (separated below 0) (n=429)
- Direct read (gap + predicted change over 2.0 s, ego term included) skill over the head copy: +0.119 [+0.043, +0.203] n=429 (separated above 0); RMSE 8.34 m [+7.073, +9.697] against the copy RMSE 8.89 m

#### HM

Fit rows used: gap 32025, closing 30865, direct 24533. Inner-fold mean squared error (fit rows only): gap 150.762, closing speed 11.3989, direct 34.0922.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 +0.083 [-0.568, +0.423]; RMSE 9.82 m [+8.133, +11.532]
- closing speed at S0: R2 -0.959 [-1.789, -0.421]; RMSE 3.20 m/s [+2.892, +3.554]
- ADD21 beside it (heads on the eval bank, 1689 to 1926 fit rows): gap R2 +0.368, closing R2 -0.457; R3 skill -0.053; direct skill -0.157

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy -0.207 [-0.488, +0.052] n=429; level R2 -0.332 [-1.055, +0.147]
- R2: skill over the head copy -0.234 [-0.514, +0.016] n=429; level R2 -0.362 [-1.111, +0.125]
- R3: skill over the head copy +0.042 [-0.127, +0.163] n=429; level R2 -0.057 [-0.580, +0.243]
- R0 (head copy): level R2 -0.104 [-0.745, +0.253]
- R3 level R2 -0.057 minus banked rollout (i) [R1 file] +0.571: -0.628 [-1.079, -0.359] (separated below 0) (n=429)
- Direct read (gap + predicted change over 2.0 s, ego term included) skill over the head copy: +0.014 [-0.213, +0.219] n=429; RMSE 10.48 m [+8.488, +12.387] against the copy RMSE 10.55 m

Selection by the inner folds of the fit rows: closing-speed head = HT; direct head = HT.

#### Learning curve of HT (secondary a; clips drawn with rng 0; whole clips until the labelled rows reach the size; decides nothing)

| labelled fit rows (gap / closing) | closing R2 on scored windows | gap R2 | g (gap / closing) |
|---|---|---|---|
| 2027 / 2032 | -0.046 [-0.211, +0.007] | +0.338 [-0.038, +0.587] | 10000 / 1e+06 |
| 8054 / 8067 | +0.031 [-0.059, +0.084] | +0.617 [+0.384, +0.752] | 10000 / 100000 |
| 16017 / 16041 | +0.060 [-0.123, +0.175] | +0.647 [+0.420, +0.779] | 10000 / 10000 |
| 32025 / 30865 | +0.145 [-0.020, +0.245] | +0.549 [+0.260, +0.712] | 1000 / 10000 |
| 32025 / 30865 | +0.145 [-0.020, +0.245] | +0.549 [+0.260, +0.712] | 1000 / 10000 |

#### Pooled fit of HT (secondary b: train probe rows and eval-bank rows of the training folds together, cross-fitted over the eval folds; decides nothing)

- gap at S0: R2 +0.555 [+0.266, +0.720]; closing speed at S0: R2 +0.140 [-0.021, +0.237]
- R3 skill over the head copy +0.091 [-0.026, +0.215] n=429; R2 skill -0.061 [-0.130, +0.014] n=429; R1 skill +0.051 [-0.026, +0.137] n=429

### G_s0

Scored windows 429 of 1948 (episodes 48); with the label history rule 413 (dropped 16).
Fit rows (train probe set, 121166 kept rows): with five frames 116366; gap 32025 (548 clips); closing 30865 (525 clips); direct 24533 (484 clips).

#### HT

Chosen penalty multiplier g (gap / closing / direct): 1000 / 10000 / 10000; at the grid edge (1 or 1e7): False / False / False.
Fit rows used: gap 32025, closing 30865, direct 24533. Inner-fold mean squared error (fit rows only): gap 111.564, closing speed 7.8452, direct 23.7166.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 +0.620 [+0.407, +0.744]; RMSE 6.32 m [+5.565, +7.103]
- closing speed at S0: R2 +0.101 [-0.083, +0.206]; RMSE 2.17 m/s [+1.900, +2.461]
- ADD21 beside it (heads on the eval bank, 1689 to 1926 fit rows): gap R2 +0.660, closing R2 -0.098; R3 skill -0.002; direct skill -0.070

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy +0.056 [-0.039, +0.154] n=429; level R2 +0.345 [+0.018, +0.554]
- R2: skill over the head copy -0.063 [-0.147, +0.026] n=429; level R2 +0.262 [-0.100, +0.492]
- R3: skill over the head copy +0.100 [-0.033, +0.229] n=429; level R2 +0.376 [+0.087, +0.570]
- R0 (head copy): level R2 +0.306 [-0.023, +0.510]
- R3 level R2 +0.376 minus banked rollout (i) [R1 file] +0.508: -0.132 [-0.330, +0.026] (n=429)
- Direct read (gap + predicted change over 2.0 s, ego term included) skill over the head copy: +0.115 [+0.029, +0.200] n=429 (separated above 0); RMSE 7.87 m [+6.701, +9.055] against the copy RMSE 8.36 m

#### HM

Fit rows used: gap 32025, closing 30865, direct 24533. Inner-fold mean squared error (fit rows only): gap 147.559, closing speed 12.2994, direct 34.5346.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 -0.020 [-0.885, +0.440]; RMSE 10.36 m [+8.068, +12.681]
- closing speed at S0: R2 -0.754 [-1.461, -0.313]; RMSE 3.03 m/s [+2.682, +3.381]
- ADD21 beside it (heads on the eval bank, 1689 to 1926 fit rows): gap R2 +0.328, closing R2 -0.360; R3 skill -0.049; direct skill -0.239

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy -0.178 [-0.375, +0.024] n=429; level R2 -0.509 [-1.538, +0.050]
- R2: skill over the head copy -0.231 [-0.444, -0.019] n=429 (separated below 0); level R2 -0.577 [-1.631, +0.007]
- R3: skill over the head copy +0.034 [-0.105, +0.134] n=429; level R2 -0.238 [-1.011, +0.205]
- R0 (head copy): level R2 -0.281 [-1.136, +0.195]
- R3 level R2 -0.238 minus banked rollout (i) [R1 file] +0.508: -0.746 [-1.428, -0.342] (separated below 0) (n=429)
- Direct read (gap + predicted change over 2.0 s, ego term included) skill over the head copy: +0.028 [-0.122, +0.174] n=429; RMSE 11.21 m [+9.273, +13.458] against the copy RMSE 11.36 m

Selection by the inner folds of the fit rows: closing-speed head = HT; direct head = HT.

#### Learning curve of HT (secondary a; clips drawn with rng 0; whole clips until the labelled rows reach the size; decides nothing)

| labelled fit rows (gap / closing) | closing R2 on scored windows | gap R2 | g (gap / closing) |
|---|---|---|---|
| 2027 / 2032 | -0.132 [-0.384, -0.013] | +0.251 [-0.183, +0.529] | 1000 / 100000 |
| 8054 / 8067 | +0.023 [-0.071, +0.071] | +0.595 [+0.353, +0.735] | 10000 / 100000 |
| 16017 / 16041 | +0.048 [-0.158, +0.174] | +0.646 [+0.381, +0.784] | 10000 / 10000 |
| 32025 / 30865 | +0.101 [-0.083, +0.206] | +0.620 [+0.407, +0.744] | 1000 / 10000 |
| 32025 / 30865 | +0.101 [-0.083, +0.206] | +0.620 [+0.407, +0.744] | 1000 / 10000 |

#### Pooled fit of HT (secondary b: train probe rows and eval-bank rows of the training folds together, cross-fitted over the eval folds; decides nothing)

- gap at S0: R2 +0.636 [+0.434, +0.757]; closing speed at S0: R2 +0.106 [-0.068, +0.204]
- R3 skill over the head copy +0.091 [-0.039, +0.215] n=429; R2 skill -0.077 [-0.155, +0.010] n=429; R1 skill +0.047 [-0.044, +0.143] n=429

### G_s1

Scored windows 429 of 1948 (episodes 48); with the label history rule 413 (dropped 16).
Fit rows (train probe set, 121166 kept rows): with five frames 116366; gap 32025 (548 clips); closing 30865 (525 clips); direct 24533 (484 clips).

#### HT

Chosen penalty multiplier g (gap / closing / direct): 1000 / 10000 / 10000; at the grid edge (1 or 1e7): False / False / False.
Fit rows used: gap 32025, closing 30865, direct 24533. Inner-fold mean squared error (fit rows only): gap 108.621, closing speed 7.9863, direct 24.3289.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 +0.535 [+0.267, +0.685]; RMSE 6.99 m [+6.108, +7.942]
- closing speed at S0: R2 +0.176 [+0.012, +0.278]; RMSE 2.08 m/s [+1.807, +2.368]
- ADD21 beside it (heads on the eval bank, 1689 to 1926 fit rows): gap R2 +0.618, closing R2 -0.060; R3 skill +0.083; direct skill -0.046

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy +0.031 [-0.044, +0.106] n=429; level R2 +0.228 [-0.110, +0.439]
- R2: skill over the head copy -0.071 [-0.145, +0.009] n=429; level R2 +0.147 [-0.222, +0.375]
- R3: skill over the head copy +0.104 [-0.022, +0.215] n=429; level R2 +0.286 [-0.026, +0.485]
- R0 (head copy): level R2 +0.203 [-0.142, +0.421]
- R3 level R2 +0.286 minus banked rollout (i) [R1 file] +0.484: -0.198 [-0.448, -0.015] (separated below 0) (n=429)
- Direct read (gap + predicted change over 2.0 s, ego term included) skill over the head copy: +0.090 [+0.018, +0.163] n=429 (separated above 0); RMSE 8.55 m [+7.584, +9.639] against the copy RMSE 8.96 m

#### HM

Fit rows used: gap 32025, closing 30865, direct 24533. Inner-fold mean squared error (fit rows only): gap 146.286, closing speed 11.8774, direct 34.3447.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 +0.174 [-0.444, +0.490]; RMSE 9.33 m [+7.804, +10.988]
- closing speed at S0: R2 -0.513 [-1.154, -0.106]; RMSE 2.81 m/s [+2.471, +3.157]
- ADD21 beside it (heads on the eval bank, 1689 to 1926 fit rows): gap R2 +0.313, closing R2 -0.288; R3 skill +0.140; direct skill -0.257

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy -0.014 [-0.291, +0.186] n=429; level R2 -0.117 [-0.703, +0.239]
- R2: skill over the head copy -0.069 [-0.316, +0.126] n=429; level R2 -0.178 [-0.794, +0.191]
- R3: skill over the head copy -0.005 [-0.213, +0.153] n=429; level R2 -0.108 [-0.707, +0.235]
- R0 (head copy): level R2 -0.102 [-0.765, +0.274]
- R3 level R2 -0.108 minus banked rollout (i) [R1 file] +0.484: -0.592 [-1.093, -0.295] (separated below 0) (n=429)
- Direct read (gap + predicted change over 2.0 s, ego term included) skill over the head copy: +0.011 [-0.166, +0.152] n=429; RMSE 10.48 m [+8.980, +12.077] against the copy RMSE 10.54 m

Selection by the inner folds of the fit rows: closing-speed head = HT; direct head = HT.

#### Learning curve of HT (secondary a; clips drawn with rng 0; whole clips until the labelled rows reach the size; decides nothing)

| labelled fit rows (gap / closing) | closing R2 on scored windows | gap R2 | g (gap / closing) |
|---|---|---|---|
| 2027 / 2032 | -0.155 [-0.436, -0.013] | +0.139 [-0.352, +0.442] | 1000 / 100000 |
| 8054 / 8067 | +0.041 [-0.057, +0.091] | +0.351 [+0.007, +0.546] | 1000 / 100000 |
| 16017 / 16041 | +0.101 [-0.067, +0.205] | +0.652 [+0.434, +0.772] | 10000 / 10000 |
| 32025 / 30865 | +0.176 [+0.012, +0.278] | +0.535 [+0.267, +0.685] | 1000 / 10000 |
| 32025 / 30865 | +0.176 [+0.012, +0.278] | +0.535 [+0.267, +0.685] | 1000 / 10000 |

#### Pooled fit of HT (secondary b: train probe rows and eval-bank rows of the training folds together, cross-fitted over the eval folds; decides nothing)

- gap at S0: R2 +0.570 [+0.320, +0.706]; closing speed at S0: R2 +0.174 [+0.014, +0.272]
- R3 skill over the head copy +0.095 [-0.026, +0.203] n=429; R2 skill -0.074 [-0.147, +0.005] n=429; R1 skill +0.031 [-0.041, +0.105] n=429

## Outcomes (card section 4; the head is the one chosen by the inner folds of the fit rows)

- Arm type P (both seeds needed): closing **HEAD-CLOSING-FLAT-B**; slot **HEAD-SLOT-FLAT-B**; direct **STATE-HOLDS-FUTURE-B** (heads {'P_s0': 'HT', 'P_s1': 'HT'}); heads used {'P_s0': 'HT', 'P_s1': 'HT'}; with each head run: {"HT": {"closing": "HEAD-CLOSING-FLAT-B", "slot": "HEAD-SLOT-FLAT-B"}, "HM": {"closing": "HEAD-CLOSING-FLAT-B", "slot": "HEAD-SLOT-FLAT-B"}}
- Arm type G (both seeds needed): closing **HEAD-CLOSING-FLAT-B**; slot **HEAD-SLOT-FLAT-B**; direct **STATE-HOLDS-FUTURE-B** (heads {'G_s0': 'HT', 'G_s1': 'HT'}); heads used {'G_s0': 'HT', 'G_s1': 'HT'}; with each head run: {"HT": {"closing": "HEAD-CLOSING-FLAT-B", "slot": "HEAD-SLOT-FLAT-B"}, "HM": {"closing": "HEAD-CLOSING-FLAT-B", "slot": "HEAD-SLOT-FLAT-B"}}

## Reading of the pair of cards (card section 4) and the learning curve

- P_s0: HT closing-speed R2 on the scored windows at 2,000 / 8,000 / 16,000 / all labelled fit rows: -0.155 / +0.011 / +0.062 / +0.126 (ADD21, 1,689 to 1,926 rows: -0.094).
- P_s1: HT closing-speed R2 on the scored windows at 2,000 / 8,000 / 16,000 / all labelled fit rows: -0.046 / +0.031 / +0.060 / +0.145 (ADD21, 1,689 to 1,926 rows: -0.108).
- G_s0: HT closing-speed R2 on the scored windows at 2,000 / 8,000 / 16,000 / all labelled fit rows: -0.132 / +0.023 / +0.048 / +0.101 (ADD21, 1,689 to 1,926 rows: -0.098).
- G_s1: HT closing-speed R2 on the scored windows at 2,000 / 8,000 / 16,000 / all labelled fit rows: -0.155 / +0.041 / +0.101 / +0.176 (ADD21, 1,689 to 1,926 rows: -0.060).

Card ADD21 read flat for both arm types; the names of this card are flat for both arm types (closing and slot). The learning curve rises at every step up to all labelled fit rows (closing R2 8,000 < 16,000 < all, every arm: True); the all-rows point equals the 32,000 point because the train probe set holds fewer labelled rows than that (32,025 gap rows, 30,865 closing rows). The branch of card section 4 for a flat read with a flat curve therefore does not apply; the branch of card section 5 for a rising curve that has not flattened at all rows applies (the fit set is extended to more training clips before any name is read as a property of the state). Control j8 (shift between held-out train clips and the eval windows) is printed above.


## Limits

- The heads are fitted on states of clips the model was trained on and applied to states of clips it was not; control j8 measures that shift (R2 on the held-out tenth of the train clips beside R2 on the eval windows). A flat read with a large shift is reported as not decisive.
- The scored set is small (429 windows, 48 episodes); one training run per seed, so an interval answers another draw of episodes and the names need both seeds.
- Train labels use the join lead gap in 0 to 60 m with lead_valid; the scored windows use the eval-bank gap as in card ADD21 (LEADHEAD_B_DEFS.md M3).
- R1, R2, R3 and the direct read use the logged ego speeds of the future frames (the action input of a rollout); no goal input, nothing of the situation classifier.
- T0 only: nothing here is driving performance; LONGITUDINAL, LATERAL, TACTICAL not measured; STRATEGIC PAUSED (PI item 25).
