# VERDICT_S1_W_s0: PREREG_V8F_PROOF_ADD12 note N8, state-file mode
Tier T0: a world-model diagnostic, never driving performance. Numbers are in-sample for the model (train probe clips). Estimator: paired episode-cluster bootstrap with clips as clusters (1200 clips), B 2000, seed 0, one draw shared by every number. Four metric families: LONGITUDINAL, LATERAL, TACTICAL not measured by this read; STRATEGIC: PAUSED (PI item 25, never a pass).
An interval over clips is blind to training variance (H-ESTIM-SEED-1): a difference between two trained arms needs the replicate spread (`spread`) beside it.

## CONTROLS (first)
- t1: PASS :: {"clips_equal": true, "n_rows_state": 121166, "n_rows_probe": 121166, "rows_equal": true, "clip_idx_equal": true, "t_equal": true, "z_is_2d": true, "z_finite": true, "bar": "clips, clip_idx and t equal the probe set exactly; state finite"}
- t4: PASS :: {"declared_dim": 640, "declared_state": "mean", "array_dim": 640, "record_present": true, "dim_array": true, "dim_record": true, "state_mode_record": true, "mode_width_consistent": true, "record_wm_state": "mean", "record_ckpt_sha256": "d262de2c38449b5b9bc62508d8285c4e21f726a14e00d62d39ec7c64d689ebd6", "record_counts_equal": true, "record_split_range": true, "state_file_sha256_equals_record": true, "ckpt_sha256": "equal to the record", "bar": "array width, record width, state mode, counts and the state file's own sha256 equal the record; checkpoint sha256 equal to the one given"}
- t2: PASS :: {"state_mode_values": {"gap_r2": 0.28432310888356316, "pres_auc": 0.7621820704144355, "n_gap_rows": 33114, "n_rows": 121166, "n_clips": 1200, "seconds": 161.39209008216858}, "tol": 0.005, "reference_file": "F:/TanitAD-scratch/v8proof/spatial/s0e/RESULT_S0E.json", "reference_values_read_at_run_time": {"gap_r2": 0.2843231104806241, "pres_auc": 0.76218207500181}, "abs_diff_gap_r2": 1.597060927949201e-09, "abs_diff_pres_auc": 4.5873744802804595e-09, "bar": "both within 0.005 of the stage 0e mean state", "from_cache": true}
- t3: PASS :: {"max_value": -0.00153088578968541, "bar": 0.02, "values": {"ra gap R2 s": -0.0021416883819747046, "ra gap R2 ego": -0.00153088578968541, "change skill H": -0.003079151211640818, "change skill N": -0.002599017973049067, "change skill E": -0.002793257512844205}, "presence_auc_printed": {"presence AUROC s (printed, no bar)": 0.49389701185090146, "presence AUROC ego (printed, no bar)": 0.4870690108064423}, "n_gap_rows": 33114, "n_pairs": 24206, "n_clips": 1200}
- controls_pass: True

## Label: trained state, W_s0
No outcome name of the card is assigned by this tool. State width 640, mode mean, state file W_s0_train.npz, checkpoint sha256 d262de2c38449b5b9bc62508d8285c4e21f726a14e00d62d39ec7c64d689ebd6.
## Reads (a): lead gap and presence from one frame (T0; a world-model diagnostic, never driving performance; in-sample for the model (train probe clips))
Frames 121166, frames with a lead gap in 0 to 60 m 33114, clips 1200; probe fit subsample at most 20000 training rows per fold.
- trained state: gap R2 +0.448 [+0.402, +0.488] (n 33114 frames, 552 clips); presence AUROC 0.792 [+0.774, +0.809] (n 121166 frames, 1200 clips)
- ego vector: gap R2 +0.187 [+0.134, +0.238] (n 33114 frames, 552 clips); presence AUROC 0.559 [+0.538, +0.580] (n 121166 frames, 1200 clips)
- state minus ego: gap R2 +0.260 [+0.213, +0.309]; presence AUROC +0.233 [+0.209, +0.257]
## Reads (b): change of the lead gap over 2.0 s, skill over persistence (T0; a world-model diagnostic, never driving performance; in-sample for the model (train probe clips))
Pairs 24206 of 484 clips; pairs removed by the continuity rule 1740; std(d) 5.195 m, mean -0.195 m, share |d| above 2 m 0.532; fit subsample at most 20000 training pairs per fold.
- H: skill +0.260 [+0.214, +0.302]; plain R2 +0.259 [+0.212, +0.298]; n pairs 24206, clips 484
- N: skill +0.256 [+0.207, +0.299]; plain R2 +0.255 [+0.206, +0.297]; n pairs 24206, clips 484
- E: skill +0.291 [+0.243, +0.334]; plain R2 +0.290 [+0.242, +0.332]; n pairs 24206, clips 484
- H minus E: -0.032 [-0.070, +0.008]
- H minus N: +0.004 [-0.005, +0.014]

Counts: {"pairs_with_three_rows": 103166, "pairs_used": 24206, "pair_rule": "continuous", "pairs_removed_by_the_continuity_rule": 1740, "step_max_m": 3.0, "bank_rows": 121166, "gap_rows": 33114, "rows_dropped_nonfinite_kin": 0, "pairs_per_fold": [4596, 4249, 5618, 4815, 4928]}
Peak RAM 2.34 GB; seconds 702.3
