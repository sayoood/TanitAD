# VERDICT LEADHEAD stage 0 (PREREG_V8F_PROOF_ADD21)

Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25). Zero training of the model; CPU; the banked states and labels.
Heads run in this file: HT, HM; direct read: run; horizon profile: run. Definitions: LEADHEAD_DEFS.md. Ego speeds of the rule are the logged ones (the action input of a rollout).

## Controls (first)

- h6_label_path: PASS :: {"r2_skill_this_path": 0.7499068112986798, "r2_skill_objslot_label_pipeline_now": 0.7499068112986798, "r2_skill_banked_ADD20": 0.7499068112986798, "n": 413, "tol": 1e-06}
- h1_planted_state: PASS :: {"heldout_r2_closing": 0.9439669758952696, "bar": 0.9, "n": 1560, "noise_sd_per_dim": 0.8641232851171319, "amp_per_metre": 10.0, "kind": "HT"}
- h3_reference_fidelity|P_s0: PASS :: {"max_abs_diff_to_banked_VG": 0.0, "HL_source": "cached"}
- h2_shuffled_labels|HT|P_s0: PASS :: {"heldout_r2_closing_shuffled_labels": -0.09989907658726205, "bar": 0.02, "n": 413}
- h5_no_future_input|HT|P_s0: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0}
- h4_leak_detector|HT|P_s0: PASS :: {"r2_in_sample_all_folds": 0.27582050195934193, "r2_cross_fitted": -0.09428331466805928, "n": 413}
- h2_shuffled_labels|HM|P_s0: PASS :: {"heldout_r2_closing_shuffled_labels": -1.8143377990326255, "bar": 0.02, "n": 413}
- h5_no_future_input|HM|P_s0: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0}
- h3_reference_fidelity|P_s1: PASS :: {"max_abs_diff_to_banked_VG": 0.0, "HL_source": "cached"}
- h2_shuffled_labels|HT|P_s1: PASS :: {"heldout_r2_closing_shuffled_labels": -0.09549136288596438, "bar": 0.02, "n": 413}
- h5_no_future_input|HT|P_s1: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0}
- h4_leak_detector|HT|P_s1: PASS :: {"r2_in_sample_all_folds": 0.25901594494688995, "r2_cross_fitted": -0.10803521998384458, "n": 413}
- h2_shuffled_labels|HM|P_s1: PASS :: {"heldout_r2_closing_shuffled_labels": -2.48071402196945, "bar": 0.02, "n": 413}
- h5_no_future_input|HM|P_s1: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0}
- h3_reference_fidelity|G_s0: PASS :: {"max_abs_diff_to_banked_VG": 0.0, "HL_source": "cached"}
- h2_shuffled_labels|HT|G_s0: PASS :: {"heldout_r2_closing_shuffled_labels": -0.10010782301140009, "bar": 0.02, "n": 413}
- h5_no_future_input|HT|G_s0: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0}
- h4_leak_detector|HT|G_s0: PASS :: {"r2_in_sample_all_folds": 0.26185935008556427, "r2_cross_fitted": -0.09792586247385304, "n": 413}
- h2_shuffled_labels|HM|G_s0: PASS :: {"heldout_r2_closing_shuffled_labels": -2.8222498915498364, "bar": 0.02, "n": 413}
- h5_no_future_input|HM|G_s0: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0}
- h3_reference_fidelity|G_s1: PASS :: {"max_abs_diff_to_banked_VG": 0.0, "HL_source": "cached"}
- h2_shuffled_labels|HT|G_s1: PASS :: {"heldout_r2_closing_shuffled_labels": -0.10078656759063542, "bar": 0.02, "n": 413}
- h5_no_future_input|HT|G_s1: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0}
- h4_leak_detector|HT|G_s1: PASS :: {"r2_in_sample_all_folds": 0.2903068892569801, "r2_cross_fitted": -0.060199281626740486, "n": 413}
- h2_shuffled_labels|HM|G_s1: PASS :: {"heldout_r2_closing_shuffled_labels": -1.7007234266800557, "bar": 0.02, "n": 413}
- h5_no_future_input|HM|G_s1: PASS :: {"max_abs_change_of_head_outputs": 0.0, "max_abs_change_of_rule_outputs": 0.0}
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

Chosen penalty multiplier g per fold (gap / closing / direct): f0 10000 / 100000 / 100000; f1 10000 / 100000 / 100000; f2 10000 / 100000 / 100000; f3 10000 / 100000 / 100000; f4 10000 / 10000 / 10000
Inner-fold mean squared error (selection of the head, fit rows only): closing speed 6.2933; direct 18.6001.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 +0.523 [+0.334, +0.640]; RMSE 7.09 m [+6.214, +7.962]
- closing speed at S0: R2 -0.094 [-0.238, +0.012]; RMSE 2.39 m/s [+1.972, +2.862]

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy -0.011 [-0.104, +0.073] n=429; level R2 +0.431 [+0.237, +0.558]
- R2: skill over the head copy -0.113 [-0.237, +0.002] n=429; level R2 +0.373 [+0.148, +0.524]
- R3: skill over the head copy -0.045 [-0.128, +0.030] n=429; level R2 +0.411 [+0.217, +0.539]
- R0 (head copy): level R2 +0.437 [+0.249, +0.561]
- R3 level R2 +0.411 minus banked rollout (i) [R1 file] +0.556: -0.144 [-0.247, -0.053] (separated below 0) (n=429)
- Direct read (gap + predicted change over 2.0 s) skill over the head copy: -0.039 [-0.128, +0.039] n=429; RMSE 7.68 m [+6.849, +8.515] against the copy RMSE 7.54 m

#### HM

Inner-fold mean squared error (selection of the head, fit rows only): closing speed 7.5978; direct 21.1182.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 +0.227 [-0.032, +0.397]; RMSE 9.02 m [+7.897, +10.137]
- closing speed at S0: R2 -0.275 [-0.659, +0.013]; RMSE 2.58 m/s [+2.094, +3.149]

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy -0.146 [-0.328, -0.007] n=429 (separated below 0); level R2 +0.045 [-0.239, +0.227]
- R2: skill over the head copy -0.172 [-0.357, -0.020] n=429 (separated below 0); level R2 +0.024 [-0.268, +0.209]
- R3: skill over the head copy -0.065 [-0.129, -0.003] n=429 (separated below 0); level R2 +0.113 [-0.170, +0.305]
- R0 (head copy): level R2 +0.167 [-0.103, +0.350]
- R3 level R2 +0.113 minus banked rollout (i) [R1 file] +0.556: -0.443 [-0.632, -0.285] (separated below 0) (n=429)
- Direct read (gap + predicted change over 2.0 s) skill over the head copy: -0.138 [-0.277, -0.023] n=429 (separated below 0); RMSE 9.78 m [+8.772, +10.884] against the copy RMSE 9.17 m

Selection by the inner folds: closing-speed head = HT; direct head = HT.

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

Chosen penalty multiplier g per fold (gap / closing / direct): f0 10000 / 100000 / 100000; f1 1000 / 100000 / 100000; f2 10000 / 100000 / 100000; f3 10000 / 100000 / 100000; f4 10000 / 100000 / 100000
Inner-fold mean squared error (selection of the head, fit rows only): closing speed 6.3554; direct 19.2414.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 +0.605 [+0.464, +0.690]; RMSE 6.45 m [+5.666, +7.251]
- closing speed at S0: R2 -0.108 [-0.255, -0.005]; RMSE 2.41 m/s [+1.996, +2.876]

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy -0.050 [-0.149, +0.039] n=429; level R2 +0.498 [+0.330, +0.607]
- R2: skill over the head copy -0.185 [-0.314, -0.062] n=429 (separated below 0); level R2 +0.433 [+0.227, +0.569]
- R3: skill over the head copy -0.018 [-0.154, +0.107] n=429; level R2 +0.513 [+0.358, +0.614]
- R0 (head copy): level R2 +0.521 [+0.366, +0.624]
- R3 level R2 +0.513 minus banked rollout (i) [R1 file] +0.571: -0.058 [-0.153, +0.034] (n=429)
- Direct read (gap + predicted change over 2.0 s) skill over the head copy: -0.060 [-0.166, +0.031] n=429; RMSE 7.15 m [+6.298, +8.060] against the copy RMSE 6.95 m

#### HM

Inner-fold mean squared error (selection of the head, fit rows only): closing speed 8.7159; direct 24.1041.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 +0.368 [+0.138, +0.519]; RMSE 8.16 m [+6.920, +9.528]
- closing speed at S0: R2 -0.457 [-0.916, -0.125]; RMSE 2.76 m/s [+2.316, +3.235]

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy -0.345 [-0.646, -0.078] n=429 (separated below 0); level R2 +0.176 [-0.155, +0.414]
- R2: skill over the head copy -0.386 [-0.665, -0.134] n=429 (separated below 0); level R2 +0.152 [-0.169, +0.373]
- R3: skill over the head copy -0.053 [-0.242, +0.104] n=429; level R2 +0.356 [+0.153, +0.506]
- R0 (head copy): level R2 +0.388 [+0.196, +0.524]
- R3 level R2 +0.356 minus banked rollout (i) [R1 file] +0.571: -0.215 [-0.371, -0.080] (separated below 0) (n=429)
- Direct read (gap + predicted change over 2.0 s) skill over the head copy: -0.157 [-0.360, +0.038] n=429; RMSE 8.45 m [+7.073, +9.880] against the copy RMSE 7.86 m

Selection by the inner folds: closing-speed head = HT; direct head = HT.

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

Chosen penalty multiplier g per fold (gap / closing / direct): f0 10000 / 100000 / 100000; f1 10000 / 100000 / 100000; f2 10000 / 100000 / 100000; f3 10000 / 100000 / 100000; f4 10000 / 100000 / 10000
Inner-fold mean squared error (selection of the head, fit rows only): closing speed 6.4232; direct 19.0808.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 +0.660 [+0.519, +0.747]; RMSE 5.98 m [+5.282, +6.717]
- closing speed at S0: R2 -0.098 [-0.233, -0.008]; RMSE 2.40 m/s [+1.977, +2.875]

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy -0.050 [-0.127, +0.023] n=429; level R2 +0.534 [+0.372, +0.637]
- R2: skill over the head copy -0.192 [-0.315, -0.075] n=429 (separated below 0); level R2 +0.470 [+0.269, +0.597]
- R3: skill over the head copy -0.002 [-0.096, +0.087] n=429; level R2 +0.555 [+0.412, +0.646]
- R0 (head copy): level R2 +0.556 [+0.412, +0.648]
- R3 level R2 +0.555 minus banked rollout (i) [R1 file] +0.508: +0.047 [-0.044, +0.147] (n=429)
- Direct read (gap + predicted change over 2.0 s) skill over the head copy: -0.070 [-0.155, +0.010] n=429; RMSE 6.92 m [+6.193, +7.675] against the copy RMSE 6.69 m

#### HM

Inner-fold mean squared error (selection of the head, fit rows only): closing speed 8.6787; direct 21.3599.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 +0.328 [+0.061, +0.503]; RMSE 8.41 m [+7.237, +9.574]
- closing speed at S0: R2 -0.360 [-0.674, -0.130]; RMSE 2.67 m/s [+2.199, +3.174]

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy -0.286 [-0.499, -0.080] n=429 (separated below 0); level R2 +0.110 [-0.219, +0.350]
- R2: skill over the head copy -0.326 [-0.535, -0.120] n=429 (separated below 0); level R2 +0.082 [-0.272, +0.325]
- R3: skill over the head copy -0.049 [-0.155, +0.047] n=429; level R2 +0.274 [+0.008, +0.451]
- R0 (head copy): level R2 +0.308 [+0.040, +0.481]
- R3 level R2 +0.274 minus banked rollout (i) [R1 file] +0.508: -0.233 [-0.428, -0.076] (separated below 0) (n=429)
- Direct read (gap + predicted change over 2.0 s) skill over the head copy: -0.239 [-0.400, -0.080] n=429 (separated below 0); RMSE 9.29 m [+7.935, +10.757] against the copy RMSE 8.35 m

Selection by the inner folds: closing-speed head = HT; direct head = HT.

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

Chosen penalty multiplier g per fold (gap / closing / direct): f0 10000 / 100000 / 100000; f1 10000 / 100000 / 100000; f2 10000 / 100000 / 100000; f3 10000 / 100000 / 100000; f4 1000 / 100000 / 10000
Inner-fold mean squared error (selection of the head, fit rows only): closing speed 6.1537; direct 18.4185.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 +0.618 [+0.462, +0.717]; RMSE 6.34 m [+5.563, +7.136]
- closing speed at S0: R2 -0.060 [-0.200, +0.026]; RMSE 2.36 m/s [+1.950, +2.822]

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy -0.035 [-0.112, +0.031] n=429; level R2 +0.471 [+0.274, +0.602]
- R2: skill over the head copy -0.194 [-0.314, -0.084] n=429 (separated below 0); level R2 +0.390 [+0.144, +0.551]
- R3: skill over the head copy +0.083 [-0.027, +0.180] n=429; level R2 +0.532 [+0.345, +0.652]
- R0 (head copy): level R2 +0.489 [+0.299, +0.620]
- R3 level R2 +0.532 minus banked rollout (i) [R1 file] +0.484: +0.048 [-0.059, +0.148] (n=429)
- Direct read (gap + predicted change over 2.0 s) skill over the head copy: -0.046 [-0.122, +0.019] n=429; RMSE 7.34 m [+6.407, +8.400] against the copy RMSE 7.18 m

#### HM

Inner-fold mean squared error (selection of the head, fit rows only): closing speed 8.1689; direct 20.8489.

Fidelity on held-out windows with the label history rule (n=413):
- gap at S0: R2 +0.313 [+0.055, +0.469]; RMSE 8.50 m [+7.485, +9.491]
- closing speed at S0: R2 -0.288 [-0.552, -0.083]; RMSE 2.60 m/s [+2.141, +3.091]

Rules over the copy of the same head (n=429 scored windows):
- R1: skill over the head copy -0.362 [-0.531, -0.185] n=429 (separated below 0); level R2 -0.004 [-0.400, +0.259]
- R2: skill over the head copy -0.415 [-0.579, -0.242] n=429 (separated below 0); level R2 -0.043 [-0.448, +0.229]
- R3: skill over the head copy +0.140 [-0.022, +0.270] n=429; level R2 +0.366 [+0.134, +0.511]
- R0 (head copy): level R2 +0.263 [+0.001, +0.431]
- R3 level R2 +0.366 minus banked rollout (i) [R1 file] +0.484: -0.118 [-0.296, +0.027] (n=429)
- Direct read (gap + predicted change over 2.0 s) skill over the head copy: -0.257 [-0.518, -0.035] n=429 (separated below 0); RMSE 9.67 m [+8.223, +11.056] against the copy RMSE 8.62 m

Selection by the inner folds: closing-speed head = HT; direct head = HT.

## Outcomes (card section 4; the head is the one chosen by the inner folds of the fit rows)

- Arm type P (both seeds needed): closing **HEAD-CLOSING-FLAT**; slot **HEAD-SLOT-FLAT**; direct **STATE-LACKS-FUTURE** (heads {'P_s0': 'HT', 'P_s1': 'HT'}; with each head: {'HT': 'STATE-LACKS-FUTURE', 'HM': 'STATE-LACKS-FUTURE'}); heads used {'P_s0': 'HT', 'P_s1': 'HT'}; with each head run: {"HT": {"closing": "HEAD-CLOSING-FLAT", "slot": "HEAD-SLOT-FLAT"}, "HM": {"closing": "HEAD-CLOSING-FLAT", "slot": "HEAD-SLOT-FLAT"}}
- Arm type G (both seeds needed): closing **HEAD-CLOSING-FLAT**; slot **HEAD-SLOT-FLAT**; direct **STATE-LACKS-FUTURE** (heads {'G_s0': 'HT', 'G_s1': 'HT'}; with each head: {'HT': 'STATE-LACKS-FUTURE', 'HM': 'STATE-LACKS-FUTURE'}); heads used {'G_s0': 'HT', 'G_s1': 'HT'}; with each head run: {"HT": {"closing": "HEAD-CLOSING-FLAT", "slot": "HEAD-SLOT-FLAT"}, "HM": {"closing": "HEAD-CLOSING-FLAT", "slot": "HEAD-SLOT-FLAT"}}

Share of the label ceiling (R3 skill of the head over 0.750, the label R2 skill of ADD20; decides nothing):

- P_s0 HT: -0.060
- P_s0 HM: -0.086
- P_s1 HT: -0.024
- P_s1 HM: -0.070
- G_s0 HT: -0.002
- G_s0 HM: -0.066
- G_s1 HT: +0.111
- G_s1 HM: +0.187

## Horizon profile (decides nothing; skill over the copy of the same head)

| horizon s | arm | head | R1 | R2 | R3 |
|---|---|---|---|---|---|
| 0.4 | P_s0 (n=502) | HT | -0.001 [-0.021, +0.017] | -0.001 [-0.021, +0.016] | -0.046 [-0.082, -0.007] |
| 0.4 | P_s0 (n=502) | HM | -0.017 [-0.045, +0.008] | -0.017 [-0.044, +0.008] | -0.069 [-0.141, +0.007] |
| 0.4 | P_s1 (n=502) | HT | -0.004 [-0.025, +0.014] | -0.005 [-0.025, +0.012] | -0.100 [-0.178, -0.023] |
| 0.4 | P_s1 (n=502) | HM | -0.043 [-0.088, +0.005] | -0.042 [-0.087, +0.005] | -0.093 [-0.323, +0.092] |
| 0.4 | G_s0 (n=502) | HT | -0.006 [-0.022, +0.009] | -0.006 [-0.022, +0.008] | -0.069 [-0.125, -0.018] |
| 0.4 | G_s0 (n=502) | HM | -0.039 [-0.073, -0.004] | -0.039 [-0.072, -0.003] | -0.088 [-0.219, +0.030] |
| 0.4 | G_s1 (n=502) | HT | -0.005 [-0.021, +0.009] | -0.007 [-0.023, +0.006] | -0.015 [-0.067, +0.039] |
| 0.4 | G_s1 (n=502) | HM | -0.055 [-0.084, -0.025] | -0.055 [-0.083, -0.026] | +0.084 [-0.083, +0.220] |
| 0.8 | P_s0 (n=480) | HT | -0.005 [-0.050, +0.032] | -0.008 [-0.051, +0.029] | -0.048 [-0.096, -0.002] |
| 0.8 | P_s0 (n=480) | HM | -0.041 [-0.098, +0.012] | -0.039 [-0.095, +0.013] | -0.075 [-0.157, +0.003] |
| 0.8 | P_s1 (n=480) | HT | -0.014 [-0.056, +0.025] | -0.020 [-0.061, +0.018] | -0.100 [-0.195, -0.004] |
| 0.8 | P_s1 (n=480) | HM | -0.099 [-0.200, +0.005] | -0.097 [-0.196, +0.001] | -0.089 [-0.327, +0.096] |
| 0.8 | G_s0 (n=480) | HT | -0.016 [-0.052, +0.016] | -0.021 [-0.055, +0.012] | -0.072 [-0.130, -0.012] |
| 0.8 | G_s0 (n=480) | HM | -0.092 [-0.162, -0.020] | -0.090 [-0.159, -0.018] | -0.082 [-0.218, +0.036] |
| 0.8 | G_s1 (n=480) | HT | -0.012 [-0.044, +0.018] | -0.023 [-0.056, +0.007] | -0.006 [-0.071, +0.061] |
| 0.8 | G_s1 (n=480) | HM | -0.121 [-0.183, -0.057] | -0.123 [-0.183, -0.062] | +0.092 [-0.072, +0.232] |
| 1.2 | P_s0 (n=460) | HT | -0.005 [-0.065, +0.050] | -0.022 [-0.084, +0.033] | -0.045 [-0.097, +0.005] |
| 1.2 | P_s0 (n=460) | HM | -0.072 [-0.161, +0.009] | -0.071 [-0.161, +0.006] | -0.071 [-0.148, -0.004] |
| 1.2 | P_s1 (n=460) | HT | -0.022 [-0.082, +0.037] | -0.047 [-0.108, +0.015] | -0.076 [-0.168, +0.019] |
| 1.2 | P_s1 (n=460) | HM | -0.170 [-0.325, -0.016] | -0.173 [-0.321, -0.024] | -0.071 [-0.269, +0.096] |
| 1.2 | G_s0 (n=460) | HT | -0.026 [-0.076, +0.020] | -0.052 [-0.104, -0.002] | -0.051 [-0.109, +0.006] |
| 1.2 | G_s0 (n=460) | HM | -0.150 [-0.273, -0.036] | -0.153 [-0.269, -0.042] | -0.070 [-0.189, +0.038] |
| 1.2 | G_s1 (n=460) | HT | -0.019 [-0.066, +0.023] | -0.056 [-0.107, -0.008] | +0.028 [-0.042, +0.100] |
| 1.2 | G_s1 (n=460) | HM | -0.204 [-0.305, -0.104] | -0.213 [-0.304, -0.119] | +0.125 [-0.041, +0.259] |
| 1.6 | P_s0 (n=444) | HT | -0.010 [-0.096, +0.066] | -0.058 [-0.154, +0.029] | -0.042 [-0.106, +0.016] |
| 1.6 | P_s0 (n=444) | HM | -0.107 [-0.243, -0.001] | -0.114 [-0.248, -0.003] | -0.070 [-0.144, -0.001] |
| 1.6 | P_s1 (n=444) | HT | -0.040 [-0.125, +0.039] | -0.109 [-0.202, -0.024] | -0.045 [-0.161, +0.070] |
| 1.6 | P_s1 (n=444) | HM | -0.260 [-0.487, -0.052] | -0.275 [-0.495, -0.081] | -0.062 [-0.269, +0.101] |
| 1.6 | G_s0 (n=444) | HT | -0.042 [-0.109, +0.022] | -0.113 [-0.195, -0.033] | -0.027 [-0.103, +0.049] |
| 1.6 | G_s0 (n=444) | HM | -0.221 [-0.380, -0.059] | -0.235 [-0.391, -0.079] | -0.060 [-0.182, +0.048] |
| 1.6 | G_s1 (n=444) | HT | -0.029 [-0.094, +0.028] | -0.114 [-0.194, -0.036] | +0.052 [-0.043, +0.139] |
| 1.6 | G_s1 (n=444) | HM | -0.282 [-0.411, -0.142] | -0.307 [-0.432, -0.169] | +0.129 [-0.047, +0.260] |

## Limits

- The heads are fitted on bank states and applied to window states, which differ from the bank states of the same frames by a median 0.015 per element (standard deviation of a dimension 0.864); LEADHEAD_DEFS.md L18.
- Fit rows are few (see the counts per fold); a negative of HT and HM is a negative for heads of this size on this bank, and the third next step of the card tests the state before pooling.
- The scored set is small (429 windows, 48 episodes); one training run per seed, so an interval answers another draw of episodes and the names need both seeds.
- R1, R2, R3 and the direct read use the logged ego speeds of the future frames (the action input of a rollout); no goal input, nothing of the situation classifier.
- The ADD20 label ceiling uses the privileged gap history; the head sees only the grid states. The share of the ceiling compares different inputs.

## Notes written after the run (facts about the read, deciding nothing)

- HT penalty at the edge of the card grid: the closing-speed ridge chose the largest multiplier (1e5) in 19 of 20 arm-fold fits and the direct ridge in 17 of 20; the gap ridge chose 1e3 to 1e4 and never the edge. A head at the top of its grid reads close to the fit-row mean for that target, which is what the held-out closing R2 of -0.06 to -0.11 shows. The grid is the card's and was not extended.
- Control h1 was run with a planted amplitude of 10.0 per metre; the first design number (5.0) read 0.8174 and is kept in LEADHEAD_DEFS.md as a recorded deviation, fixed before any real-target number.
- The HM per-component standardisation of the 64 principal components is also recorded in LEADHEAD_DEFS.md (L6), decided before any real-target number.
- Stage 1 (HT only, no direct read, no horizon profile) is kept as VERDICT_LEADHEAD_S0_STAGE1_HT.md; its HT numbers equal those above.
