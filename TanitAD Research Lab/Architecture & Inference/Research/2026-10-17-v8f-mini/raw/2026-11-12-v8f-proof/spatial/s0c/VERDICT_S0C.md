# VERDICT_S0C: PREREG_V8F_PROOF_ADD12 note N5, stage 0c (the change of the lead gap, +2.0 s)
Tier T0: a world-model diagnostic, never driving performance. Estimator: paired episode-cluster bootstrap over 139 episodes, B 2000, seed 0, one draw shared by all arms. Four metric families: LONGITUDINAL, LATERAL, TACTICAL not measured by this read; STRATEGIC: PAUSED (PI item 25, never a pass).
A negative of stage 0c is a negative about a LINEAR read of two pooled frames on an untrained state; stage 1 still runs.

## CONTROLS (first)
- k1: PASS :: {"max_skill": -0.07561952074094735, "bar": 0.02, "values": {"H-g": -0.1373683135306245, "H-m": -0.10966901531399609, "N-g": -0.1460417195205317, "E": -0.07561952074094735}, "n_pairs": 1586, "n_episodes": 50}
- k2: PASS :: {"auc_grid_two_frames": 1.0, "auc_mean_two_frames": 0.49299076195943786, "auc_grid_one_frame": 0.5112712364407225, "bars": "grid two frames > 0.95; mean two frames <= 0.6; grid one frame <= 0.6", "n_pairs": 1586, "n_episodes": 50, "cell": 23, "vector_norm": 7.691012859344482}
- k3: PASS :: {"n_pairs": 1586, "bad_index": 0, "bad_ego": 0, "bad_gap": 0, "gaps_compared": true, "bar": "all counts exactly 0; arrays compared with array_equal", "tokens": {"n_checked": 30, "bad": 0}}
- k4: PASS :: {"grid_gap_r2": 0.3981453992786981, "ego_gap_r2": 0.13274532858514854, "reference": {"grid_gap_r2": 0.3981, "ego_gap_r2": 0.1327}, "bar": "equal to the reference at 4 decimals", "ego_floor_choice": {"scores": {"P1": 0.016116915791232755, "P2": -0.11121577704296559, "linear": 0.13274532858514854}, "chosen": "linear"}}
- controls_pass: True

## Target d = gap(g+10) - gap(g), metres: n pairs 1586, episodes 50; std(d) 4.511 m, mean +0.315 m; share of pairs with |d| above 2 m 0.578 (T0; a world-model diagnostic, never driving performance)
## Skill over persistence (1 - SSE / sum d^2; persistence = 0) and plain R2 beside
- H-g: skill -0.087 [-0.299, +0.079]; plain R2 -0.093 [-0.329, +0.060]; n pairs 1586, episodes 50
- H-m: skill +0.035 [-0.143, +0.183]; plain R2 +0.031 [-0.186, +0.173]; n pairs 1586, episodes 50
- N-g: skill +0.041 [-0.101, +0.151]; plain R2 +0.037 [-0.144, +0.140]; n pairs 1586, episodes 50
- E: skill +0.111 [-0.105, +0.253]; plain R2 +0.106 [-0.152, +0.246]; n pairs 1586, episodes 50
## Paired differences of skill (paired episode-cluster bootstrap)
- H-g minus E: -0.198 [-0.404, +0.041]
- H-m minus E: -0.075 [-0.256, +0.128]
- H-g minus H-m: -0.123 [-0.290, +0.028]
- H-g minus N-g: -0.129 [-0.256, -0.030]
## OUTCOME NAMES
- grid state (H-g against E): S0-CHANGE-FLAT
- mean state (H-m against E): S0-CHANGE-FLAT
- pair (H-g against H-m): GRID-EQUAL-CHANGE
- history (H-g against N-g): HISTORY-EQUAL

Counts: {"pairs_with_three_rows": 7236, "pairs_used": 1586, "pair_rule": "continuous", "pairs_removed_by_the_continuity_rule": 166, "step_max_m": 3.0, "bank_rows": 9321, "pairs_per_fold": [367, 240, 346, 414, 219]}
