# VERDICT_TOKPROBE: card PREREG_V8F_PROOF_ADD27 (TOKPROBE) with note N1

Tier T0 (a diagnostic on logged frames, never driving performance). LONGITUDINAL (distance keeping) is the family this lever serves and is not measured by this read; LATERAL, TACTICAL: not measured; STRATEGIC: PAUSED (PI item 25).
Evidence class of every number: MEASURED once run. B 2000, episode-cluster bootstrap, rng 0.

ALIAS RUN: arm names are loaded from other files ({"TK16_s0": "LSD_s0", "TK16_s1": "LSD_s0", "TK64_s0": "LSD_s0", "TK64_s1": "LSD_s0", "TK16SH_s0": "LSD_s0", "LSD_s1": "LSD_s0"}); no number of this run is a result of the probe.

## Names (card section 5; both seeds needed)

- TK16: **not named** (a control failed: p3_shuffled_labels, p3)
- TK64: **not named** (a control failed: p3_shuffled_labels, p3)

## Controls

Inputs of the lead: {"p1": true, "p2a": true, "p2b": true, "p5": true} (p2b blocks only the flat name; a missing input counts as not passed).

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
- k3_card_form_slot|TK16_s0: FAIL (printed, gates nothing)
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
- p3_shuffled_labels: FAIL
- paired_planted_shift_read_back: PASS
- paired_self_reads_exactly_0: PASS
- paired_swap_reads_the_negative: PASS
- p3 (TK16SH_s0): closing-speed R2 on the scored windows +0.178, bar at most +0.02: FAIL
- failed controls touching TK16: p3_shuffled_labels, p3
- failed controls touching TK64: p3_shuffled_labels, p3

## Paired difference of M1 (probe arm minus the LSD arm of the same seed index)

- TK16_s0 minus LSD_s0: +0.0000 [+0.000, +0.000]; n=413
- TK16_s1 minus LSD_s1: +0.0000 [+0.000, +0.000]; n=413
- TK64_s0 minus LSD_s0: +0.0000 [+0.000, +0.000]; n=413
- TK64_s1 minus LSD_s1: +0.0000 [+0.000, +0.000]; n=413
- floor (|M1 LSD_s0 - M1 LSD_s1|): 0.0000

## Per arm (reported; M2, M4, DELTA and the train probe clips name nothing)

### TK16_s0
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

### TK16_s1
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

### TK64_s0
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

### TK64_s1
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

### TK16SH_s0
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

## Limits

A probe on the tokens says what the pooled tokens allow, not what the model does; 429 scored windows of 48 evaluation episodes; one corpus. The paired bootstrap resamples episodes with the arms held fixed; the two seeds differ in initialisation only, and the floor comes from the two LSD runs.

