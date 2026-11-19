# VERDICT_FINEGRID: card PREREG_V8F_PROOF_ADD41 (FINEGRID)

Tier T0 (a diagnostic on logged frames, never driving performance). LONGITUDINAL (distance keeping) is the family this lever serves and it is not measured by this read; LATERAL, TACTICAL: not measured by this read; STRATEGIC: NOT COMPUTED (a token probe reads no strategic quantity; n = 0).
Evidence class of every number: MEASURED once run. B 2000, episode-cluster bootstrap, rng 0, the 429 scored windows of 48 evaluation episodes.

## Controls (card section 4; all must pass before any outcome is named)

- a1_records|F16SH_s0: PASS
- a1_records|F16_s0: PASS
- a1_records|F16_s1: PASS
- a1_records|F64_s0: PASS
- a1_records|F64_s1: PASS
- a1_records|TK16_s0: PASS
- a1_records|TK16_s1: PASS
- a1_records|TK64_s0: PASS
- a1_records|TK64_s1: PASS
- a2_pack_counts: PASS
- a3_same_head_eval_train|F16_s0: PASS
- a3_same_head_eval_train|F16_s1: PASS
- a3_same_head_eval_train|F64_s0: PASS
- a3_same_head_eval_train|F64_s1: PASS
- a3_same_head_eval_train|TK16_s0: PASS
- a3_same_head_eval_train|TK16_s1: PASS
- a3_same_head_eval_train|TK64_s0: PASS
- a3_same_head_eval_train|TK64_s1: PASS
- a4_window_states|F16_s0: PASS
- a4_window_states|F16_s1: PASS
- a4_window_states|F64_s0: PASS
- a4_window_states|F64_s1: PASS
- a4_window_states|TK16_s0: PASS
- a4_window_states|TK16_s1: PASS
- a4_window_states|TK64_s0: PASS
- a4_window_states|TK64_s1: PASS
- a5_train_layout|F16_s0: PASS
- a5_train_layout|F16_s1: PASS
- a5_train_layout|F64_s0: PASS
- a5_train_layout|F64_s1: PASS
- a5_train_layout|TK16_s0: PASS
- a5_train_layout|TK16_s1: PASS
- a5_train_layout|TK64_s0: PASS
- a5_train_layout|TK64_s1: PASS
- g1_given: PASS
- g2_grids_belong_together: PASS
- g3_given: PASS
- g4_given: PASS
- g5_declared_against_built|F16SH_s0: PASS
- g5_declared_against_built|F16_s0: PASS
- g5_declared_against_built|F16_s1: PASS
- g5_declared_against_built|F64_s0: PASS
- g5_declared_against_built|F64_s1: PASS
- g6_forward_given: PASS
- g7_tk_reproduces_banked|TK16_s0: PASS
- g7_tk_reproduces_banked|TK16_s1: PASS
- g7_tk_reproduces_banked|TK64_s0: PASS
- g7_tk_reproduces_banked|TK64_s1: PASS
- k1_label_path: PASS
- k2_head_forward_eval|F16_s0: PASS
- k2_head_forward_eval|F16_s1: PASS
- k2_head_forward_eval|F64_s0: PASS
- k2_head_forward_eval|F64_s1: PASS
- k2_head_forward_eval|TK16_s0: PASS
- k2_head_forward_eval|TK16_s1: PASS
- k2_head_forward_eval|TK64_s0: PASS
- k2_head_forward_eval|TK64_s1: PASS
- k2_head_forward_train|F16_s0: PASS
- k2_head_forward_train|F16_s1: PASS
- k2_head_forward_train|F64_s0: PASS
- k2_head_forward_train|F64_s1: PASS
- k2_head_forward_train|TK16_s0: PASS
- k2_head_forward_train|TK16_s1: PASS
- k2_head_forward_train|TK64_s0: PASS
- k2_head_forward_train|TK64_s1: PASS
- k3_card_form_slot|F16_s0: PASS (printed, gates nothing)
- k3_card_form_slot|F16_s1: PASS (printed, gates nothing)
- k3_card_form_slot|F64_s0: PASS (printed, gates nothing)
- k3_card_form_slot|F64_s1: PASS (printed, gates nothing)
- k3_closing|F16_s0: PASS
- k3_closing|F16_s1: PASS
- k3_closing|F64_s0: PASS
- k3_closing|F64_s1: PASS
- k3_slot_n1|F16_s0: PASS
- k3_slot_n1|F16_s1: PASS
- k3_slot_n1|F64_s0: PASS
- k3_slot_n1|F64_s1: PASS
- k4_no_future_input|F16_s0: PASS
- k4_no_future_input|F16_s1: PASS
- k4_no_future_input|F64_s0: PASS
- k4_no_future_input|F64_s1: PASS
- p3_shuffled_labels: PASS
- p5_train_log|F16SH_s0: FAIL
- p5_train_log|F16_s0: PASS
- p5_train_log|F16_s1: PASS
- p5_train_log|F64_s0: PASS
- p5_train_log|F64_s1: PASS
- paired_planted_shift_read_back: PASS
- paired_self_reads_exactly_0: PASS
- paired_swap_reads_the_negative: PASS
- g2 detail: correlation 0.9995420336778741, median absolute difference 0.00244140625, max absolute difference 0.1875, 20 clips, 200 frames
- p3 (F16SH_s0): closing-speed R2 on the scored windows -0.025, bar at most +0.02

## Names

No name is given and no outcome difference is printed: withheld: a control did not pass (card section 4): p5_train_log|F16SH_s0

## Per arm (reported; M4, the in-sample reads and G64 name nothing)

- F16_s0: M2 gap R2 +0.749 [+0.581, +0.847]; M1 closing R2 -0.006 [-0.266, +0.160]; M4 AUROC 0.907
  - train probe clips (in-sample, names nothing): M2 R2 +0.703; M1 R2 +0.233; M4 AUROC 0.920
- F16_s1: M2 gap R2 +0.713 [+0.407, +0.860]; M1 closing R2 -0.025 [-0.388, +0.187]; M4 AUROC 0.917
  - train probe clips (in-sample, names nothing): M2 R2 +0.704; M1 R2 +0.210; M4 AUROC 0.923
- F64_s0: M2 gap R2 +0.750 [+0.574, +0.845]; M1 closing R2 -0.143 [-0.424, +0.032]; M4 AUROC 0.896
  - train probe clips (in-sample, names nothing): M2 R2 +0.724; M1 R2 +0.149; M4 AUROC 0.923
- F64_s1: M2 gap R2 +0.765 [+0.606, +0.851]; M1 closing R2 -0.059 [-0.309, +0.096]; M4 AUROC 0.905
  - train probe clips (in-sample, names nothing): M2 R2 +0.677; M1 R2 +0.218; M4 AUROC 0.920
- TK16_s0: M2 gap R2 +0.493 [+0.085, +0.741]; M1 closing R2 -0.002 [-0.398, +0.212]; M4 AUROC 0.866
  - train probe clips (in-sample, names nothing): M2 R2 +0.659; M1 R2 +0.235; M4 AUROC 0.904
- TK16_s1: M2 gap R2 +0.704 [+0.534, +0.803]; M1 closing R2 +0.203 [-0.023, +0.328]; M4 AUROC 0.900
  - train probe clips (in-sample, names nothing): M2 R2 +0.637; M1 R2 +0.186; M4 AUROC 0.907
- TK64_s0: M2 gap R2 +0.563 [+0.286, +0.734]; M1 closing R2 +0.042 [-0.311, +0.249]; M4 AUROC 0.868
  - train probe clips (in-sample, names nothing): M2 R2 +0.646; M1 R2 +0.220; M4 AUROC 0.894
- TK64_s1: M2 gap R2 +0.571 [+0.242, +0.752]; M1 closing R2 -0.001 [-0.435, +0.257]; M4 AUROC 0.870
  - train probe clips (in-sample, names nothing): M2 R2 +0.676; M1 R2 +0.190; M4 AUROC 0.909
- F16SH_s0: control arm: closing-speed R2 -0.025 [-0.100, +0.012]

## Limits

A probe on the tokens says what the tokens allow, not what the model does; 429 scored windows of 48 episodes; one corpus; two initialisation seeds per arm type; the reference arms come from an earlier run of the same tool (control g1 covers the code path, not the run); tokens stored as float8; the grid before pooling is not read. The paired bootstrap resamples episodes with the arms held fixed.

