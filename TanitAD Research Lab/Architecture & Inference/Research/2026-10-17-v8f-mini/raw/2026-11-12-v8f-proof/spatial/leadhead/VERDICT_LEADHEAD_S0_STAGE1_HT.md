# VERDICT LEADHEAD stage 0 (PREREG_V8F_PROOF_ADD21)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25). Zero training of the model; CPU; the banked states and labels.
Heads run in this file: HT; direct read: NOT run; horizon profile: NOT run. Definitions: LEADHEAD_DEFS.md. Ego speeds of the rule are the logged ones (the action input of a rollout).

## Controls (first)

- h6_label_path: PASS :: {"r2_skill_this_path": 0.7499068112986798, "r2_skill_objslot_label_pipeline_now": 0.7499068112986798, "r2_skill_banked_ADD20": 0.7499068112986798, "n": 413, "tol": 1e-06}
- h1_planted_state: PASS :: {"heldout_r2_closing": 0.9439669758952696, "bar": 0.9, "n": 1560, "noise_sd_per_dim": 0.8641232851171319, "amp_per_metre": 10.0, "kind": "HT"}
- h3_reference_fidelity|P_s0: PASS :: {"max_abs_diff_to_banked_VG": 0.0, "HL_source": "computed"}
- h2_shuffled_labels|HT|P_s0: PASS :: {"heldout_r2_closing_shuffled_labels": -0.09989907658726205, "bar": 0.02, "n": 413}
- h5_no_future_input|HT|P_s0: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0}
- h4_leak_detector|HT|P_s0: PASS :: {"r2_in_sample_all_folds": 0.27582050195934193, "r2_cross_fitted": -0.09428331466805928, "n": 413}
- h3_reference_fidelity|P_s1: PASS :: {"max_abs_diff_to_banked_VG": 0.0, "HL_source": "computed"}
- h2_shuffled_labels|HT|P_s1: PASS :: {"heldout_r2_closing_shuffled_labels": -0.09549136288596438, "bar": 0.02, "n": 413}
- h5_no_future_input|HT|P_s1: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0}
- h4_leak_detector|HT|P_s1: PASS :: {"r2_in_sample_all_folds": 0.25901594494688995, "r2_cross_fitted": -0.10803521998384458, "n": 413}
- h3_reference_fidelity|G_s0: PASS :: {"max_abs_diff_to_banked_VG": 0.0, "HL_source": "computed"}
- h2_shuffled_labels|HT|G_s0: PASS :: {"heldout_r2_closing_shuffled_labels": -0.10010782301140009, "bar": 0.02, "n": 413}
- h5_no_future_input|HT|G_s0: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0}
- h4_leak_detector|HT|G_s0: PASS :: {"r2_in_sample_all_folds": 0.26185935008556427, "r2_cross_fitted": -0.09792586247385304, "n": 413}
- h3_reference_fidelity|G_s1: PASS :: {"max_abs_diff_to_banked_VG": 0.0, "HL_source": "computed"}
- h2_shuffled_labels|HT|G_s1: PASS :: {"heldout_r2_closing_shuffled_labels": -0.10078656759063542, "bar": 0.02, "n": 413}
- h5_no_future_input|HT|G_s1: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0}
- h4_leak_detector|HT|G_s1: PASS :: {"r2_in_sample_all_folds": 0.2903068892569801, "r2_cross_fitted": -0.060199281626740486, "n": 413}
- controls_pass: True

## Per arm

### P_s0

Scored windows 429 of 1948 (episodes 48); with the label history rule 413 (dropped 16). Window states equal the bank states of frames 0 to 4 (max abs diff 2.87e-01 where the bank row exists; 139 of the window frames 0 to 4 have no bank row, their window state is used as it is).

Fit rows per fold (bank rows of the training folds; with five frames; dropped without; rows with a gap label; with a closing label; with a direct label):

- fold 0: 7441; 6997; dropped 444; gap 1844; closing 1740; direct 1241; episodes 111
- fold 1: 7445; 7001; dropped 444; gap 2026; closing 1926; direct 1372; episodes 111
- fold 2: 7441; 6997; dropped 444; gap 1755; closing 1695; direct 1263; episodes 111
- fold 3: 7445; 7001; dropped 444; gap 1783; closing 1689; direct 1197; episodes 111
- fold 4: 7512; 7064; dropped 448; gap 1944; closing 1862; direct 1391; episodes 112

Reference HL (level probe, one frame; ADD20): R3 skill over its copy +0.011 [-0.059, +0.080] n=429.

#### HT

Chosen penalty multiplier g per fold (gap / closing): f0 10000 / 100000; f1 10000 / 100000; f2 10000 / 100000; f3 10000 / 100000; f4 10000 / 10000
Inner-fold mean squared error (selection of the head, fit rows only): closing speed 6.2933.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 +0.523 [+0.334, +0.640]; RMSE 7.09 m [+6.214, +7.962]
- closing speed at S0: R2 -0.094 [-0.238, +0.012]; RMSE 2.39 m/s [+1.972, +2.862]

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy -0.011 [-0.104, +0.073] n=429; level R2 +0.431 [+0.237, +0.558]
- R2: skill over the head copy -0.113 [-0.237, +0.002] n=429; level R2 +0.373 [+0.148, +0.524]
- R3: skill over the head copy -0.045 [-0.128, +0.030] n=429; level R2 +0.411 [+0.217, +0.539]
- R0 (head copy): level R2 +0.437 [+0.249, +0.561]
- R3 level R2 +0.411 minus banked rollout (i) [R1 file] +0.556: -0.144 [-0.247, -0.053] (separated below 0) (n=429)

### P_s1

Scored windows 429 of 1948 (episodes 48); with the label history rule 413 (dropped 16). Window states equal the bank states of frames 0 to 4 (max abs diff 2.38e-01 where the bank row exists; 139 of the window frames 0 to 4 have no bank row, their window state is used as it is).

Fit rows per fold (bank rows of the training folds; with five frames; dropped without; rows with a gap label; with a closing label; with a direct label):

- fold 0: 7441; 6997; dropped 444; gap 1844; closing 1740; direct 1241; episodes 111
- fold 1: 7445; 7001; dropped 444; gap 2026; closing 1926; direct 1372; episodes 111
- fold 2: 7441; 6997; dropped 444; gap 1755; closing 1695; direct 1263; episodes 111
- fold 3: 7445; 7001; dropped 444; gap 1783; closing 1689; direct 1197; episodes 111
- fold 4: 7512; 7064; dropped 448; gap 1944; closing 1862; direct 1391; episodes 112

Reference HL (level probe, one frame; ADD20): R3 skill over its copy +0.003 [-0.080, +0.084] n=429.

#### HT

Chosen penalty multiplier g per fold (gap / closing): f0 10000 / 100000; f1 1000 / 100000; f2 10000 / 100000; f3 10000 / 100000; f4 10000 / 100000
Inner-fold mean squared error (selection of the head, fit rows only): closing speed 6.3554.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 +0.605 [+0.464, +0.690]; RMSE 6.45 m [+5.666, +7.251]
- closing speed at S0: R2 -0.108 [-0.255, -0.005]; RMSE 2.41 m/s [+1.996, +2.876]

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy -0.050 [-0.149, +0.039] n=429; level R2 +0.498 [+0.330, +0.607]
- R2: skill over the head copy -0.185 [-0.314, -0.062] n=429 (separated below 0); level R2 +0.433 [+0.227, +0.569]
- R3: skill over the head copy -0.018 [-0.154, +0.107] n=429; level R2 +0.513 [+0.358, +0.614]
- R0 (head copy): level R2 +0.521 [+0.366, +0.624]
- R3 level R2 +0.513 minus banked rollout (i) [R1 file] +0.571: -0.058 [-0.153, +0.034] (n=429)

### G_s0

Scored windows 429 of 1948 (episodes 48); with the label history rule 413 (dropped 16). Window states equal the bank states of frames 0 to 4 (max abs diff 2.92e-01 where the bank row exists; 139 of the window frames 0 to 4 have no bank row, their window state is used as it is).

Fit rows per fold (bank rows of the training folds; with five frames; dropped without; rows with a gap label; with a closing label; with a direct label):

- fold 0: 7441; 6997; dropped 444; gap 1844; closing 1740; direct 1241; episodes 111
- fold 1: 7445; 7001; dropped 444; gap 2026; closing 1926; direct 1372; episodes 111
- fold 2: 7441; 6997; dropped 444; gap 1755; closing 1695; direct 1263; episodes 111
- fold 3: 7445; 7001; dropped 444; gap 1783; closing 1689; direct 1197; episodes 111
- fold 4: 7512; 7064; dropped 448; gap 1944; closing 1862; direct 1391; episodes 112

Reference HL (level probe, one frame; ADD20): R3 skill over its copy -0.007 [-0.087, +0.067] n=429.

#### HT

Chosen penalty multiplier g per fold (gap / closing): f0 10000 / 100000; f1 10000 / 100000; f2 10000 / 100000; f3 10000 / 100000; f4 10000 / 100000
Inner-fold mean squared error (selection of the head, fit rows only): closing speed 6.4232.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 +0.660 [+0.519, +0.747]; RMSE 5.98 m [+5.282, +6.717]
- closing speed at S0: R2 -0.098 [-0.233, -0.008]; RMSE 2.40 m/s [+1.977, +2.875]

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy -0.050 [-0.127, +0.023] n=429; level R2 +0.534 [+0.372, +0.637]
- R2: skill over the head copy -0.192 [-0.315, -0.075] n=429 (separated below 0); level R2 +0.470 [+0.269, +0.597]
- R3: skill over the head copy -0.002 [-0.096, +0.087] n=429; level R2 +0.555 [+0.412, +0.646]
- R0 (head copy): level R2 +0.556 [+0.412, +0.648]
- R3 level R2 +0.555 minus banked rollout (i) [R1 file] +0.508: +0.047 [-0.044, +0.147] (n=429)

### G_s1

Scored windows 429 of 1948 (episodes 48); with the label history rule 413 (dropped 16). Window states equal the bank states of frames 0 to 4 (max abs diff 3.39e-01 where the bank row exists; 139 of the window frames 0 to 4 have no bank row, their window state is used as it is).

Fit rows per fold (bank rows of the training folds; with five frames; dropped without; rows with a gap label; with a closing label; with a direct label):

- fold 0: 7441; 6997; dropped 444; gap 1844; closing 1740; direct 1241; episodes 111
- fold 1: 7445; 7001; dropped 444; gap 2026; closing 1926; direct 1372; episodes 111
- fold 2: 7441; 6997; dropped 444; gap 1755; closing 1695; direct 1263; episodes 111
- fold 3: 7445; 7001; dropped 444; gap 1783; closing 1689; direct 1197; episodes 111
- fold 4: 7512; 7064; dropped 448; gap 1944; closing 1862; direct 1391; episodes 112

Reference HL (level probe, one frame; ADD20): R3 skill over its copy +0.027 [-0.070, +0.123] n=429.

#### HT

Chosen penalty multiplier g per fold (gap / closing): f0 10000 / 100000; f1 10000 / 100000; f2 10000 / 100000; f3 10000 / 100000; f4 1000 / 100000
Inner-fold mean squared error (selection of the head, fit rows only): closing speed 6.1537.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 +0.618 [+0.462, +0.717]; RMSE 6.34 m [+5.563, +7.136]
- closing speed at S0: R2 -0.060 [-0.200, +0.026]; RMSE 2.36 m/s [+1.950, +2.822]

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy -0.035 [-0.112, +0.031] n=429; level R2 +0.471 [+0.274, +0.602]
- R2: skill over the head copy -0.194 [-0.314, -0.084] n=429 (separated below 0); level R2 +0.390 [+0.144, +0.551]
- R3: skill over the head copy +0.083 [-0.027, +0.180] n=429; level R2 +0.532 [+0.345, +0.652]
- R0 (head copy): level R2 +0.489 [+0.299, +0.620]
- R3 level R2 +0.532 minus banked rollout (i) [R1 file] +0.484: +0.048 [-0.059, +0.148] (n=429)

## Outcomes (card section 4; the head is the one chosen by the inner folds of the fit rows)

**Provisional: not every head or the direct read has been run in this file; names below are on the heads run and are not the card's final names.**
- Arm type P (both seeds needed): closing **HEAD-CLOSING-FLAT**; slot **HEAD-SLOT-FLAT**; direct read not run; heads used {'P_s0': 'HT', 'P_s1': 'HT'}; with each head run: {"HT": {"closing": "HEAD-CLOSING-FLAT", "slot": "HEAD-SLOT-FLAT"}}
- Arm type G (both seeds needed): closing **HEAD-CLOSING-FLAT**; slot **HEAD-SLOT-FLAT**; direct read not run; heads used {'G_s0': 'HT', 'G_s1': 'HT'}; with each head run: {"HT": {"closing": "HEAD-CLOSING-FLAT", "slot": "HEAD-SLOT-FLAT"}}

Share of the label ceiling (R3 skill of the head over 0.750, the label R2 skill of ADD20; decides nothing):

- P_s0 HT: -0.060
- P_s1 HT: -0.024
- G_s0 HT: -0.002
- G_s1 HT: +0.111

## Limits

- Fit rows are few (see the counts per fold); a negative of HT and HM is a negative for heads of this size on this bank, and the third next step of the card tests the state before pooling.
- The scored set is small (429 windows, 48 episodes); one training run per seed, so an interval answers another draw of episodes and the names need both seeds.
- R1, R2, R3 and the direct read use the logged ego speeds of the future frames (the action input of a rollout); no goal input, nothing of the situation classifier.
- The ADD20 label ceiling uses the privileged gap history; the head sees only the grid states. The share of the ceiling compares different inputs.
