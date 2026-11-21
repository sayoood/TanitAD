# VERDICT_TOKPROBE: card PREREG_V8F_PROOF_ADD27 (TOKPROBE) with note N1

Tier T0 (a diagnostic on logged frames, never driving performance). LONGITUDINAL (distance keeping) is the family this lever serves and is not measured by this read; LATERAL, TACTICAL: not measured; STRATEGIC: PAUSED (PI item 25).
Evidence class of every number: MEASURED once run. B 2000, episode-cluster bootstrap, rng 0.

## Names (card section 5; both seeds needed)

- TK16: **not named** (a control failed: p5 (false))
- TK64: **not named** (a control failed: p5 (false))

## Controls

Inputs of the lead: {"p1": true, "p2a": true, "p2b": false, "p5": {"TK16_s0": true, "TK16_s1": true, "TK64_s0": true, "TK64_s1": true}, "notes": {"p1": "real run, first step: gradient norms above 0 on the per-token network and the head, exactly 0.0 on the control parameter, all four arms (podlogs/train)", "p2a": "synthetic, 0.955 / 0.947", "p2b": "synthetic, FAIL (P2B_RESULT.md)", "p5": "as the card words it: lead term first 1,000 / last 1,000 steps 0.42 / 0.27, 0.42 / 0.29, 0.40 / 0.28, 0.39 / 0.26; gap-labelled frames at every logged step at least 51. The stricter check of the tool (every batch inside an interval) fails: 21 of 384 intervals hold a batch with no gap-labelled frame"}} (p2b blocks only the flat name; a missing input counts as not passed).

- a1_records|LSD_s0: PASS
- a1_records|LSD_s1: PASS
- a1_records|TK16SH_s0: PASS
- a1_records|TK16_s0: PASS
- a1_records|TK16_s1: PASS
- a1_records|TK64_s0: PASS
- a1_records|TK64_s1: PASS
- a2_pack_counts: PASS
- a3_same_head_eval_train|LSD_s0: PASS
- a3_same_head_eval_train|LSD_s1: PASS
- a3_same_head_eval_train|TK16_s0: PASS
- a3_same_head_eval_train|TK16_s1: PASS
- a3_same_head_eval_train|TK64_s0: PASS
- a3_same_head_eval_train|TK64_s1: PASS
- a4_window_states|LSD_s0: PASS
- a4_window_states|LSD_s1: PASS
- a4_window_states|TK16_s0: PASS
- a4_window_states|TK16_s1: PASS
- a4_window_states|TK64_s0: PASS
- a4_window_states|TK64_s1: PASS
- a5_train_layout|LSD_s0: PASS
- a5_train_layout|LSD_s1: PASS
- a5_train_layout|TK16_s0: PASS
- a5_train_layout|TK16_s1: PASS
- a5_train_layout|TK64_s0: PASS
- a5_train_layout|TK64_s1: PASS
- k1_label_path: PASS
- k2_head_forward_eval|LSD_s0: PASS
- k2_head_forward_eval|LSD_s1: PASS
- k2_head_forward_eval|TK16_s0: PASS
- k2_head_forward_eval|TK16_s1: PASS
- k2_head_forward_eval|TK64_s0: PASS
- k2_head_forward_eval|TK64_s1: PASS
- k2_head_forward_train|LSD_s0: PASS
- k2_head_forward_train|LSD_s1: PASS
- k2_head_forward_train|TK16_s0: PASS
- k2_head_forward_train|TK16_s1: PASS
- k2_head_forward_train|TK64_s0: PASS
- k2_head_forward_train|TK64_s1: PASS
- k3_card_form_slot|LSD_s0: FAIL (printed, gates nothing)
- k3_card_form_slot|LSD_s1: FAIL (printed, gates nothing)
- k3_card_form_slot|TK16_s0: PASS (printed, gates nothing)
- k3_card_form_slot|TK16_s1: FAIL (printed, gates nothing)
- k3_card_form_slot|TK64_s0: FAIL (printed, gates nothing)
- k3_card_form_slot|TK64_s1: FAIL (printed, gates nothing)
- k3_closing|LSD_s0: PASS
- k3_closing|LSD_s1: PASS
- k3_closing|TK16_s0: PASS
- k3_closing|TK16_s1: PASS
- k3_closing|TK64_s0: PASS
- k3_closing|TK64_s1: PASS
- k3_slot_n1|LSD_s0: PASS
- k3_slot_n1|LSD_s1: PASS
- k3_slot_n1|TK16_s0: PASS
- k3_slot_n1|TK16_s1: PASS
- k3_slot_n1|TK64_s0: PASS
- k3_slot_n1|TK64_s1: PASS
- k4_no_future_input|LSD_s0: PASS
- k4_no_future_input|LSD_s1: PASS
- k4_no_future_input|TK16_s0: PASS
- k4_no_future_input|TK16_s1: PASS
- k4_no_future_input|TK64_s0: PASS
- k4_no_future_input|TK64_s1: PASS
- p3_shuffled_labels: PASS
- paired_planted_shift_read_back: PASS
- paired_self_reads_exactly_0: PASS
- paired_swap_reads_the_negative: PASS
- p3 (TK16SH_s0): closing-speed R2 on the scored windows +0.010, bar at most +0.02: PASS
- failed controls touching TK16: p5 (false)
- failed controls touching TK64: p5 (false)

## Paired difference of M1 (probe arm minus the LSD arm of the same seed index)

- TK16_s0 minus LSD_s0: -0.1798 [-0.446, +0.024]; n=413
- TK16_s1 minus LSD_s1: +0.1143 [-0.086, +0.315]; n=413
- TK64_s0 minus LSD_s0: -0.1359 [-0.360, +0.058]; n=413
- TK64_s1 minus LSD_s1: -0.0902 [-0.467, +0.174]; n=413
- floor (|M1 LSD_s0 - M1 LSD_s1|): 0.0893

## Per arm (reported; M2, M4, DELTA and the train probe clips name nothing)

### TK16_s0
- M1 closing speed at S0 (scored windows): R2 -0.002 [-0.398, +0.212]; RMSE 2.29 m/s [+1.935, +2.666]; n=413
- M2 gap at S0, head: R2 +0.493 [+0.085, +0.741]; RMSE 7.31 m [+5.349, +9.228]; n=413
- M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.126 [+0.047, +0.222] n=429 (separated above 0); level R2 +0.226 [-0.261, +0.528]
- M3 R2: skill over the raw copy of the same head (card form, names nothing) +0.017 [-0.082, +0.116] n=429; level R2 +0.129 [-0.393, +0.455]
- M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.134 [+0.041, +0.236] n=429 (separated above 0)
- M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.188 [+0.068, +0.312] n=429 (separated above 0); level R2 +0.281 [-0.165, +0.558]
- M3 DELTA (note N1; names the slot): +0.020 [-0.032, +0.066] n=429 (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +3.30 m
- M3 R0 (head copy): level R2 +0.114 [-0.424, +0.460] (n scored 429)
- M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)
- M4 presence AUROC at S0: 0.866 [+0.821, +0.907]; n=1844 (positives 534, negatives 1310); names nothing
- train probe clips (in-sample ceiling):
  - M1 closing speed at S0 (train probe clips): R2 +0.235 [+0.191, +0.279]; RMSE 2.61 m/s [+2.475, +2.738]; n=30865
  - M2 gap at S0, head: R2 +0.659 [+0.614, +0.699]; RMSE 9.09 m [+8.517, +9.660]; n=32025
  - M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.088 [+0.050, +0.125] n=24533 (separated above 0); level R2 +0.613 [+0.557, +0.660]
  - M3 R2: skill over the raw copy of the same head (card form, names nothing) +0.007 [-0.027, +0.039] n=24533; level R2 +0.579 [+0.519, +0.628]
  - M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.113 [+0.075, +0.148] n=24533 (separated above 0)
  - M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.157 [+0.110, +0.199] n=24533 (separated above 0); level R2 +0.643 [+0.594, +0.685]
  - M3 DELTA (note N1; names the slot): +0.033 [+0.010, +0.056] n=24533 (separated above 0) (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
  - M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +1.98 m
  - M3 R0 (head copy): level R2 +0.576 [+0.514, +0.625] (n scored 24533)
  - M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)
  - M4 presence AUROC at S0: 0.904 [+0.896, +0.912]; n=116366 (positives 45364, negatives 71002); names nothing

### TK16_s1
- M1 closing speed at S0 (scored windows): R2 +0.203 [-0.023, +0.328]; RMSE 2.04 m/s [+1.751, +2.327]; n=413
- M2 gap at S0, head: R2 +0.704 [+0.534, +0.803]; RMSE 5.58 m [+4.740, +6.529]; n=413
- M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.209 [+0.119, +0.289] n=429 (separated above 0); level R2 +0.469 [+0.206, +0.635]
- M3 R2: skill over the raw copy of the same head (card form, names nothing) +0.057 [-0.034, +0.162] n=429; level R2 +0.367 [+0.044, +0.575]
- M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.256 [+0.110, +0.368] n=429 (separated above 0)
- M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.245 [+0.060, +0.378] n=429 (separated above 0); level R2 +0.494 [+0.266, +0.642]
- M3 DELTA (note N1; names the slot): +0.144 [+0.027, +0.258] n=429 (separated above 0) (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +2.38 m
- M3 R0 (head copy): level R2 +0.329 [-0.033, +0.562] (n scored 429)
- M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)
- M4 presence AUROC at S0: 0.900 [+0.858, +0.935]; n=1844 (positives 534, negatives 1310); names nothing
- train probe clips (in-sample ceiling):
  - M1 closing speed at S0 (train probe clips): R2 +0.186 [+0.142, +0.224]; RMSE 2.69 m/s [+2.546, +2.836]; n=30865
  - M2 gap at S0, head: R2 +0.637 [+0.585, +0.683]; RMSE 9.38 m [+8.729, +10.055]; n=32025
  - M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.120 [+0.070, +0.171] n=24533 (separated above 0); level R2 +0.583 [+0.501, +0.647]
  - M3 R2: skill over the raw copy of the same head (card form, names nothing) +0.038 [-0.011, +0.085] n=24533; level R2 +0.544 [+0.456, +0.611]
  - M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.130 [+0.084, +0.177] n=24533 (separated above 0)
  - M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.167 [+0.107, +0.226] n=24533 (separated above 0); level R2 +0.605 [+0.527, +0.668]
  - M3 DELTA (note N1; names the slot): +0.046 [+0.005, +0.086] n=24533 (separated above 0) (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
  - M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +1.93 m
  - M3 R0 (head copy): level R2 +0.526 [+0.448, +0.589] (n scored 24533)
  - M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)
  - M4 presence AUROC at S0: 0.907 [+0.898, +0.915]; n=116366 (positives 45364, negatives 71002); names nothing

### TK64_s0
- M1 closing speed at S0 (scored windows): R2 +0.042 [-0.311, +0.249]; RMSE 2.24 m/s [+1.919, +2.569]; n=413
- M2 gap at S0, head: R2 +0.563 [+0.286, +0.734]; RMSE 6.78 m [+5.344, +8.295]; n=413
- M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.108 [-0.007, +0.213] n=429; level R2 +0.302 [-0.067, +0.518]
- M3 R2: skill over the raw copy of the same head (card form, names nothing) +0.020 [-0.084, +0.116] n=429; level R2 +0.233 [-0.155, +0.462]
- M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.094 [-0.021, +0.205] n=429
- M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.104 [-0.070, +0.244] n=429; level R2 +0.298 [-0.055, +0.512]
- M3 DELTA (note N1; names the slot): +0.025 [-0.032, +0.085] n=429 (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +2.49 m
- M3 R0 (head copy): level R2 +0.217 [-0.170, +0.461] (n scored 429)
- M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)
- M4 presence AUROC at S0: 0.868 [+0.819, +0.912]; n=1844 (positives 534, negatives 1310); names nothing
- train probe clips (in-sample ceiling):
  - M1 closing speed at S0 (train probe clips): R2 +0.220 [+0.176, +0.258]; RMSE 2.63 m/s [+2.502, +2.773]; n=30865
  - M2 gap at S0, head: R2 +0.646 [+0.603, +0.684]; RMSE 9.26 m [+8.721, +9.810]; n=32025
  - M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.135 [+0.099, +0.170] n=24533 (separated above 0); level R2 +0.610 [+0.559, +0.651]
  - M3 R2: skill over the raw copy of the same head (card form, names nothing) +0.059 [+0.026, +0.094] n=24533 (separated above 0); level R2 +0.576 [+0.521, +0.620]
  - M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.134 [+0.097, +0.170] n=24533 (separated above 0)
  - M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.166 [+0.117, +0.208] n=24533 (separated above 0); level R2 +0.624 [+0.576, +0.664]
  - M3 DELTA (note N1; names the slot): +0.062 [+0.033, +0.092] n=24533 (separated above 0) (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
  - M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +1.78 m
  - M3 R0 (head copy): level R2 +0.549 [+0.486, +0.600] (n scored 24533)
  - M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)
  - M4 presence AUROC at S0: 0.894 [+0.886, +0.903]; n=116366 (positives 45364, negatives 71002); names nothing

### TK64_s1
- M1 closing speed at S0 (scored windows): R2 -0.001 [-0.435, +0.257]; RMSE 2.29 m/s [+1.981, +2.598]; n=413
- M2 gap at S0, head: R2 +0.571 [+0.242, +0.752]; RMSE 6.72 m [+5.252, +8.360]; n=413
- M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.137 [+0.030, +0.224] n=429 (separated above 0); level R2 +0.312 [-0.108, +0.554]
- M3 R2: skill over the raw copy of the same head (card form, names nothing) +0.028 [-0.082, +0.106] n=429; level R2 +0.225 [-0.227, +0.496]
- M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.137 [+0.015, +0.266] n=429 (separated above 0)
- M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.187 [+0.042, +0.317] n=429 (separated above 0); level R2 +0.352 [-0.046, +0.588]
- M3 DELTA (note N1; names the slot): +0.027 [-0.022, +0.077] n=429 (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +2.85 m
- M3 R0 (head copy): level R2 +0.203 [-0.282, +0.505] (n scored 429)
- M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)
- M4 presence AUROC at S0: 0.870 [+0.826, +0.911]; n=1844 (positives 534, negatives 1310); names nothing
- train probe clips (in-sample ceiling):
  - M1 closing speed at S0 (train probe clips): R2 +0.190 [+0.134, +0.238]; RMSE 2.68 m/s [+2.557, +2.806]; n=30865
  - M2 gap at S0, head: R2 +0.676 [+0.635, +0.713]; RMSE 8.86 m [+8.311, +9.422]; n=32025
  - M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.155 [+0.117, +0.193] n=24533 (separated above 0); level R2 +0.646 [+0.595, +0.688]
  - M3 R2: skill over the raw copy of the same head (card form, names nothing) +0.080 [+0.042, +0.117] n=24533 (separated above 0); level R2 +0.615 [+0.560, +0.659]
  - M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.137 [+0.096, +0.176] n=24533 (separated above 0)
  - M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.179 [+0.135, +0.220] n=24533 (separated above 0); level R2 +0.656 [+0.606, +0.697]
  - M3 DELTA (note N1; names the slot): +0.063 [+0.033, +0.093] n=24533 (separated above 0) (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
  - M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +1.91 m
  - M3 R0 (head copy): level R2 +0.581 [+0.520, +0.631] (n scored 24533)
  - M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)
  - M4 presence AUROC at S0: 0.909 [+0.901, +0.917]; n=116366 (positives 45364, negatives 71002); names nothing

### LSD_s0
- M1 closing speed at S0 (scored windows): R2 +0.178 [-0.059, +0.335]; RMSE 2.07 m/s [+1.836, +2.335]; n=413
- M2 gap at S0, head: R2 +0.626 [+0.400, +0.752]; RMSE 6.27 m [+5.273, +7.340]; n=413
- M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.060 [-0.099, +0.197] n=429; level R2 +0.448 [+0.171, +0.606]
- M3 R2: skill over the raw copy of the same head (card form, names nothing) -0.032 [-0.164, +0.075] n=429; level R2 +0.394 [+0.096, +0.560]
- M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.041 [-0.066, +0.148] n=429
- M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.086 [-0.075, +0.226] n=429; level R2 +0.463 [+0.211, +0.619]
- M3 DELTA (note N1; names the slot): +0.001 [-0.044, +0.045] n=429 (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +2.43 m
- M3 R0 (head copy): level R2 +0.413 [+0.097, +0.598] (n scored 429)
- M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)
- M4 presence AUROC at S0: 0.904 [+0.863, +0.936]; n=1844 (positives 534, negatives 1310); names nothing
- train probe clips (in-sample ceiling):
  - M1 closing speed at S0 (train probe clips): R2 +0.203 [+0.161, +0.241]; RMSE 2.66 m/s [+2.518, +2.801]; n=30865
  - M2 gap at S0, head: R2 +0.591 [+0.549, +0.629]; RMSE 9.96 m [+9.416, +10.492]; n=32025
  - M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.043 [+0.007, +0.077] n=24533 (separated above 0); level R2 +0.557 [+0.507, +0.597]
  - M3 R2: skill over the raw copy of the same head (card form, names nothing) -0.038 [-0.071, -0.005] n=24533 (separated below 0); level R2 +0.519 [+0.467, +0.563]
  - M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.085 [+0.052, +0.118] n=24533 (separated above 0)
  - M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.091 [+0.054, +0.126] n=24533 (separated above 0); level R2 +0.579 [+0.531, +0.620]
  - M3 DELTA (note N1; names the slot): +0.010 [-0.007, +0.026] n=24533 (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
  - M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +0.75 m
  - M3 R0 (head copy): level R2 +0.537 [+0.485, +0.582] (n scored 24533)
  - M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)
  - M4 presence AUROC at S0: 0.861 [+0.849, +0.873]; n=116366 (positives 45364, negatives 71002); names nothing

### LSD_s1
- M1 closing speed at S0 (scored windows): R2 +0.089 [-0.127, +0.234]; RMSE 2.18 m/s [+1.851, +2.554]; n=413
- M2 gap at S0, head: R2 +0.544 [+0.217, +0.719]; RMSE 6.93 m [+5.378, +8.747]; n=413
- M3 R1: skill over the raw copy of the same head (card form, names nothing) -0.018 [-0.157, +0.127] n=429; level R2 +0.256 [-0.249, +0.535]
- M3 R2: skill over the raw copy of the same head (card form, names nothing) -0.108 [-0.213, +0.007] n=429; level R2 +0.190 [-0.327, +0.484]
- M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): -0.006 [-0.117, +0.113] n=429
- M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.042 [-0.141, +0.197] n=429; level R2 +0.300 [-0.125, +0.535]
- M3 DELTA (note N1; names the slot): -0.032 [-0.066, +0.004] n=429 (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +2.41 m
- M3 R0 (head copy): level R2 +0.269 [-0.163, +0.526] (n scored 429)
- M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)
- M4 presence AUROC at S0: 0.909 [+0.871, +0.942]; n=1844 (positives 534, negatives 1310); names nothing
- train probe clips (in-sample ceiling):
  - M1 closing speed at S0 (train probe clips): R2 +0.207 [+0.167, +0.247]; RMSE 2.65 m/s [+2.507, +2.802]; n=30865
  - M2 gap at S0, head: R2 +0.564 [+0.518, +0.607]; RMSE 10.27 m [+9.694, +10.829]; n=32025
  - M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.026 [-0.012, +0.061] n=24533; level R2 +0.512 [+0.448, +0.563]
  - M3 R2: skill over the raw copy of the same head (card form, names nothing) -0.053 [-0.087, -0.020] n=24533 (separated below 0); level R2 +0.472 [+0.406, +0.525]
  - M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.083 [+0.051, +0.116] n=24533 (separated above 0)
  - M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.089 [+0.055, +0.123] n=24533 (separated above 0); level R2 +0.543 [+0.486, +0.589]
  - M3 DELTA (note N1; names the slot): +0.006 [-0.010, +0.022] n=24533 (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
  - M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +0.78 m
  - M3 R0 (head copy): level R2 +0.499 [+0.437, +0.548] (n scored 24533)
  - M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)
  - M4 presence AUROC at S0: 0.855 [+0.843, +0.868]; n=116366 (positives 45364, negatives 71002); names nothing

### TK16SH_s0
- M1 closing speed at S0 (scored windows): R2 +0.010 [-0.049, +0.025]; RMSE 2.28 m/s [+1.917, +2.679]; n=413
- M2 gap at S0, head: R2 +0.344 [-0.031, +0.596]; RMSE 8.31 m [+6.555, +10.160]; n=413
- M3 R1: skill over the raw copy of the same head (card form, names nothing) +0.023 [+0.008, +0.038] n=429 (separated above 0); level R2 +0.104 [-0.329, +0.407]
- M3 R2: skill over the raw copy of the same head (card form, names nothing) -0.088 [-0.172, -0.029] n=429 (separated below 0); level R2 +0.003 [-0.458, +0.321]
- M3 R3 skill over the BIAS-CORRECTED copy of the same head (names nothing): +0.197 [+0.040, +0.340] n=429 (separated above 0)
- M3 R3: skill over the raw copy of the same head (card form, names nothing) +0.195 [-0.025, +0.373] n=429; level R2 +0.263 [-0.046, +0.474]
- M3 DELTA (note N1; names the slot): +0.127 [-0.038, +0.268] n=429 (1 - SSE(R3c)/SSE(R3n); R3n = rule without the closing speed of the head)
- M3 head gap offset on the scored windows (mean of head gap minus label gap at S0): +2.29 m
- M3 R0 (head copy): level R2 +0.084 [-0.362, +0.398] (n scored 429)
- M3 difference to the banked rollout read: not available (no banked rollout file for this arm type)
- M4 presence AUROC at S0: 0.891 [+0.849, +0.926]; n=1844 (positives 534, negatives 1310); names nothing

## Limits

A probe on the tokens says what the pooled tokens allow, not what the model does; 429 scored windows of 48 evaluation episodes; one corpus. The paired bootstrap resamples episodes with the arms held fixed; the two seeds differ in initialisation only, and the floor comes from the two LSD runs.

