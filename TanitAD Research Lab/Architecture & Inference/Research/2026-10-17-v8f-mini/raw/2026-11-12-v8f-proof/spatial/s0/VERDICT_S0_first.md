# VERDICT_S0: PREREG_V8F_PROOF_ADD12 Stage 0
Tier T0: a world-model diagnostic, never driving performance. Estimator: paired episode-cluster bootstrap over 139 episodes, B 2000, seed 0. Four metric families: LONGITUDINAL, LATERAL, TACTICAL not measured by this read; STRATEGIC: PAUSED (PI item 25, never a pass).
A negative of stage 0 is a negative about a LINEAR predictor on an untrained state; it does not close stage 1. r-c (pixels): not run (S0_DEFS R14: frames64.u8 and LO_r128.i8 deleted; LATVIS prep took 1,895 s of CPU; decoders need the GPU)

## CONTROLS (first)
- s1: PASS :: {"max_r2": -0.016525874218303205, "bar": 0.02, "values": {"ra gap|g": -0.016525874218303205, "ra gap|m": -0.02562310968671988, "ra gap|E": -0.016926104912124096, "rb g|i": -0.04910385762685787, "rb g|ii": -0.06197710684913349, "rb g|iii": -0.06781992907554613, "rb m|i": -0.02936383490421912, "rb m|ii": -0.029834734600610924, "rb m|iii": -0.03231957401894703, "rb ego|iv": -0.03767486194299274}, "presence_auc_permuted": {"pres|g": 0.4816383900760141, "pres|m": 0.45105488906997787, "pres|E": 0.4559502457844818}}
- s2: PASS :: {"auc_S0g": 1.0, "auc_S0m": 0.5176541262328128, "bars": "S0g > 0.95 and S0m <= 0.60"}
- s2p: PASS :: {"auc_S0g": 1.0, "auc_S0m": 0.49491400878282626, "bars": "S0g > 0.95 and S0m <= 0.60", "target": "position of the planted object (cell or mirror cell), f5 amplitude; lemma group P predicts S0m at 0.5"}
- s3: FAIL :: {"ceiling_ok": false, "ego_floor_gap_r2": 0.13274532858514854, "fullres_E": 0.133, "tol": 0.03, "r2_i": {"g": 0.025467389435073096, "m": 0.17945952140125843}, "r2_ii": {"g": -0.030277985110971573, "m": 0.15096563190054746}, "r2_iii": {"g": 0.020400625104529224, "m": 0.17760679675975966}}
- s4: PASS :: {"max_abs_diff": {"S0g": 0.0, "S0m": 0.0}}
- controls_pass: False

## OUTCOME NAMES WITHHELD: a control did not read its known value. Numbers are in RESULT_S0.json, not quotable.

Counts: {"windows_eval141": 2399, "windows_16_real_frames": 1976, "windows_dropped_clip_not_in_bank_set": 28, "windows_dropped_no_bank_row_at_g0_or_g0+10": 139, "windows_dropped_nonfinite_actions": 0, "windows_used": 1809, "bank_rows": 9321, "lead_both_windows": 441, "gap_valid_bank_rows": 2485, "presence_valid_bank_rows": 8838}
