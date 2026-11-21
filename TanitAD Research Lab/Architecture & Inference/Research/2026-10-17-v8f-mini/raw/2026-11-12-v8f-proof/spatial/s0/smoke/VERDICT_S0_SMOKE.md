# VERDICT_S0 (SMOKE: permuted labels): PREREG_V8F_PROOF_ADD12 Stage 0
Tier T0: a world-model diagnostic, never driving performance. Estimator: paired episode-cluster bootstrap over 12 episodes, B 200, seed 0. Four metric families: LONGITUDINAL, LATERAL, TACTICAL not measured by this read; STRATEGIC: PAUSED (PI item 25, never a pass).
A negative of stage 0 is a negative about a LINEAR predictor on an untrained state; it does not close stage 1. r-c (pixels): not run (S0_DEFS R14: frames64.u8 and LO_r128.i8 deleted; LATVIS prep took 1,895 s of CPU; decoders need the GPU)

## CONTROLS (first)
- s2: PASS :: {"auc_S0g": 1.0, "auc_S0m": 0.5205566019032514, "bars": "S0g > 0.95 and S0m <= 0.60"}
- s2p: PASS :: {"auc_S0g": 1.0, "auc_S0m": 0.45470017762209197, "bars": "S0g > 0.95 and S0m <= 0.60", "target": "position of the planted object (cell or mirror cell), f5 amplitude; lemma group P predicts S0m at 0.5"}
- s4: PASS :: {"max_abs_diff": {"S0g": 0.0, "S0m": 0.0}}

SMOKE: lead labels permuted across rows; no outcome name and no R2 is quoted. Shapes and timings are in RESULT_S0_SMOKE.json.

Counts: {"windows_eval141": 2399, "windows_16_real_frames": 1976, "windows_dropped_clip_not_in_bank_set": 1808, "windows_dropped_no_bank_row_at_g0_or_g0+10": 12, "windows_dropped_nonfinite_actions": 0, "windows_used": 156, "bank_rows": 804, "lead_both_windows": 36, "gap_valid_bank_rows": 185, "presence_valid_bank_rows": 769}
