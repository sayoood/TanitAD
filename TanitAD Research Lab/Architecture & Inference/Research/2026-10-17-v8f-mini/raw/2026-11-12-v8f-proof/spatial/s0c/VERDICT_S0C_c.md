# VERDICT_S0C: PREREG_V8F_PROOF_ADD12 note N5, stage 0c (the change of the lead gap, +2.0 s)
Tier T0: a world-model diagnostic, never driving performance. Estimator: paired episode-cluster bootstrap over 139 episodes, B 2000, seed 0, one draw shared by all arms. Four metric families: LONGITUDINAL, LATERAL, TACTICAL not measured by this read; STRATEGIC: PAUSED (PI item 25, never a pass).
A negative of stage 0c is a negative about a LINEAR read of two pooled frames on an untrained state; stage 1 still runs.

## CONTROLS (first)
- k1: PASS :: {"max_skill": -0.04540973805191917, "bar": 0.02, "values": {"H-g": -0.0707882547908063, "H-m": -0.04540973805191917, "N-g": -0.05124408807919467, "E": -0.046346607298531683}, "n_pairs": 1752, "n_episodes": 51}
- k2: PASS :: {"auc_grid_two_frames": 1.0, "auc_mean_two_frames": 0.5066270304647914, "auc_grid_one_frame": 0.5239094397064037, "bars": "grid two frames > 0.95; mean two frames <= 0.6; grid one frame <= 0.6", "n_pairs": 1752, "n_episodes": 51, "cell": 23, "vector_norm": 7.691012859344482}
- k3: PASS :: {"n_pairs": 1752, "bad_index": 0, "bad_ego": 0, "bad_gap": 0, "gaps_compared": true, "bar": "all counts exactly 0; arrays compared with array_equal", "tokens": {"n_checked": 30, "bad": 0}}
- k4: PASS :: {"grid_gap_r2": 0.3981453992786981, "ego_gap_r2": 0.13274532858514854, "reference": {"grid_gap_r2": 0.3981, "ego_gap_r2": 0.1327}, "bar": "equal to the reference at 4 decimals", "ego_floor_choice": {"scores": {"P1": 0.016116915791232755, "P2": -0.11121577704296559, "linear": 0.13274532858514854}, "chosen": "linear"}}
- controls_pass: True

## Target d = gap(g+10) - gap(g), metres: n pairs 1752, episodes 51; std(d) 6.279 m, mean +0.300 m; share of pairs with |d| above 2 m 0.612 (T0; a world-model diagnostic, never driving performance)
## Skill over persistence (1 - SSE / sum d^2; persistence = 0) and plain R2 beside
- H-g: skill +0.002 [-0.093, +0.102]; plain R2 -0.000 [-0.104, +0.087]; n pairs 1752, episodes 51
- H-m: skill +0.044 [-0.013, +0.114]; plain R2 +0.041 [-0.020, +0.098]; n pairs 1752, episodes 51
- N-g: skill +0.006 [-0.080, +0.104]; plain R2 +0.004 [-0.087, +0.087]; n pairs 1752, episodes 51
- E: skill -0.033 [-0.151, +0.057]; plain R2 -0.036 [-0.186, +0.052]; n pairs 1752, episodes 51
## Paired differences of skill (paired episode-cluster bootstrap)
- H-g minus E: +0.035 [-0.102, +0.207]
- H-m minus E: +0.077 [-0.020, +0.227]
- H-g minus H-m: -0.042 [-0.111, +0.014]
- H-g minus N-g: -0.004 [-0.026, +0.013]
## OUTCOME NAMES
- grid state (H-g against E): S0-CHANGE-FLAT
- mean state (H-m against E): S0-CHANGE-FLAT
- pair (H-g against H-m): GRID-EQUAL-CHANGE
- history (H-g against N-g): HISTORY-EQUAL

Counts: {"pairs_with_three_rows": 7236, "pairs_used": 1752, "bank_rows": 9321, "pairs_per_fold": [377, 240, 425, 435, 275]}
