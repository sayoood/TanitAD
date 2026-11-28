> LEAD NOTE: this is the SECOND read (stage 0b, note `PREREG_V8F_PROOF_ADD12_N4.md`). The first read (`VERDICT_S0_first.md`) missed control s3 and printed no outcome name. Stage 0b changes one thing, written down before its run and after every first-read number had been seen: the r-b probe is fitted on the bank frames (the r-a probe of the fold). A stricter second ego floor and control s5 were added; no control was weakened. Quote the outcome names below as "stage 0b".

# VERDICT_S0: PREREG_V8F_PROOF_ADD12 Stage 0
Tier T0: a world-model diagnostic, never driving performance. Estimator: paired episode-cluster bootstrap over 139 episodes, B 2000, seed 0. Four metric families: LONGITUDINAL, LATERAL, TACTICAL not measured by this read; STRATEGIC: PAUSED (PI item 25, never a pass).
A negative of stage 0 is a negative about a LINEAR predictor on an untrained state; it does not close stage 1. r-c (pixels): not run (S0_DEFS R14: frames64.u8 and LO_r128.i8 deleted; LATVIS prep took 1,895 s of CPU; decoders need the GPU)

## CONTROLS (first)
- s1: PASS :: {"max_r2": -0.016525874218303205, "bar": 0.02, "values": {"ra gap|g": -0.016525874218303205, "ra gap|m": -0.02562310968671988, "ra gap|E": -0.016926104912124096, "rb g|i": -0.017844169936201748, "rb g|ii": -0.019982440952670544, "rb g|iii": -0.021203218185129158, "rb m|i": -0.0410573905372662, "rb m|ii": -0.04259059420104294, "rb m|iii": -0.0479586329326358, "rb ego|iv": -0.03767486194299274, "rb ego|ivb": -0.026933820737470526}, "presence_auc_permuted": {"pres|g": 0.4816383900760141, "pres|m": 0.45105488906997787, "pres|E": 0.4559502457844818}}
- s2: PASS :: {"auc_S0g": 1.0, "auc_S0m": 0.5176541262328128, "bars": "S0g > 0.95 and S0m <= 0.60"}
- s2p: PASS :: {"auc_S0g": 1.0, "auc_S0m": 0.49491400878282626, "bars": "S0g > 0.95 and S0m <= 0.60", "target": "position of the planted object (cell or mirror cell), f5 amplitude; lemma group P predicts S0m at 0.5"}
- s3: PASS :: {"ceiling_ok": true, "ego_floor_gap_r2": 0.13274532858514854, "fullres_E": 0.133, "tol": 0.03, "r2_i": {"g": 0.34269254932906246, "m": 0.21239190327083446}, "r2_ii": {"g": 0.35716994782618217, "m": 0.19718653550883247}, "r2_iii": {"g": 0.36593761393892477, "m": 0.2184591290994683}}
- s4: PASS :: {"max_abs_diff": {"S0g": 0.0, "S0m": 0.0}}
- s5: PASS :: {"g": {"n": 441, "max_abs_diff": 8.526512829121202e-13}, "m": {"n": 441, "max_abs_diff": 3.183231456205249e-12}, "bar": "max abs diff <= 1e-9 on rows where both exist"}
- controls_pass: True

## r-a retention (bank frames, n gap rows 2485, presence rows 8838; T0; a world-model diagnostic, never driving performance)
- S0g (640): gap R2 +0.398 [+0.225, +0.523]; presence AUROC 0.761 [+0.675, +0.834]
- S0m (128): gap R2 +0.258 [+0.033, +0.430]; presence AUROC 0.742 [+0.649, +0.820]
- ego floor: gap R2 +0.133 [-0.209, +0.367]; presence AUROC 0.596 [+0.505, +0.685]
- gap R2 S0g minus ego floor: +0.265 [+0.076, +0.535]
- gap R2 S0m minus ego floor: +0.125 [-0.081, +0.379]
- gap R2 S0g minus S0m: +0.140 [-0.004, +0.291]
## r-b imagined lead at +2.0 s (windows with a lead at both times, n 441; T0; a world-model diagnostic, never driving performance)
- S0g: (i) predicted +0.343, (ii) copy +0.357, (iii) true future +0.366; (i)-(ii) -0.014 [-0.059, +0.028]; (i)-(iv) +0.164 [-0.044, +0.449]
- S0m: (i) predicted +0.212, (ii) copy +0.197, (iii) true future +0.218; (i)-(ii) +0.015 [-0.015, +0.048]; (i)-(iv) +0.034 [-0.220, +0.340]
- (iv) ego only: +0.108 [-0.316, +0.368]
- (iv-b) ego only, bank probe on the true future ego vector (note N4): +0.179 [-0.211, +0.421]; floor used by the outcome rule: ivb; r-b probe rows: bank
## OUTCOME NAMES
- S0g: S0-LEAD-KEPT; S0-IMAGINE-FLAT
- S0m: S0-LEAD-LOST; S0-IMAGINE-FLAT
- headline (S0g against S0m): GRID-EQUAL

Counts: {"windows_eval141": 2399, "windows_16_real_frames": 1976, "windows_dropped_clip_not_in_bank_set": 28, "windows_dropped_no_bank_row_at_g0_or_g0+10": 139, "windows_dropped_nonfinite_actions": 0, "windows_used": 1809, "bank_rows": 9321, "lead_both_windows": 441, "gap_valid_bank_rows": 2485, "presence_valid_bank_rows": 8838}
