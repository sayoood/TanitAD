# dk141 table (descriptive; decides nothing)

Tier: T1-class open loop (self-action: the arm is conditioned on its OWN held upper-layer outputs, never on GT); the lead's logged future does not react, so 'closes on the lead' means 'closes on where the lead was'
Estimator: taniteval.ci.episode_cluster_bootstrap / paired_episode_cluster_bootstrap, B 2000, seed 0, over EPISODES; delta = a - b; answers 'another draw of EPISODES' only
Lead agent layer: lead track = published PhysicalAI-AV `obstacle.offline` via taniteval/tools/build_lead_block_b1.py (B1 per-frame block, 147 v7.2 EVAL clips); NOT the augmented v7.x label release; NOT the v9 scalar lead labels of lead_eval139.npz

## k10: 2.0 s horizon  (window states {'LEAD': 743, 'NO_LEAD': 561, 'NOT_STRAIGHT': 937, 'NO_LABEL': 158}; speed proof max 2.62e-05 m/s)

| run | n hw | headway m | n tg | time gap s | n TTC (closing) | min-TTC s | TTC<1.5 s: n, rate | tg<1.0 s rate | LEAD-VIOL at-risk | progress |
|---|---|---|---|---|---|---|---|---|---|---|
| human | 719 | 27.46 [23.45, 31.49] | 653 | 4.26 [3.32, 5.43] | 719 (365) | 22.75 [21.01, 24.36] | 719, 0.003 [0.000, 0.007] | 0.073 [0.013, 0.156] | 0 by construction | 1 |
| M2b_s0 | 705 | 27.52 [23.73, 31.62] | 661 | 4.14 [3.32, 5.16] | 705 (413) | 20.61 [18.67, 22.44] | 705, 0.009 [0.002, 0.017] | 0.080 [0.022, 0.156] | 5/605 = 0.83 % | 0.980 |
| M2b_s1 | 722 | 27.22 [23.44, 31.36] | 661 | 4.29 [3.35, 5.43] | 722 (396) | 21.36 [19.58, 23.07] | 722, 0.010 [0.003, 0.019] | 0.080 [0.021, 0.161] | 5/605 = 0.83 % | 0.986 |
| P10 | 723 | 27.41 [23.66, 31.59] | 661 | 4.22 [3.33, 5.41] | 723 (354) | 22.55 [20.80, 24.22] | 723, 0.008 [0.002, 0.017] | 0.080 [0.022, 0.157] | 5/605 = 0.83 % | 0.978 |

Paired deltas a - b (delta [lo, hi]; sep = CI excludes 0; positive = MORE gap/time/TTC, or MORE violations on the rate rows):

| a - b | headway m | time gap s | min-TTC s | TTC<1.5 s rate | tg<1.0 s rate | LEAD-VIOL at-risk |
|---|---|---|---|---|---|---|
| M2b_s0 - P10 | -0.22 [-0.44, +0.01] ns | -0.013 [-0.037, +0.015] ns | -1.76 [-2.58, -1.02] sep | +0.000 [-0.006, +0.006] ns | +0.000 [-0.004, +0.004] ns | +0.0000 [-0.0050, +0.0049] ns |
| M2b_s1 - P10 | -0.11 [-0.28, +0.10] ns | -0.008 [-0.033, +0.022] ns | -1.19 [-1.80, -0.64] sep | +0.001 [-0.004, +0.008] ns | +0.000 [-0.004, +0.004] ns | +0.0000 [-0.0050, +0.0049] ns |
| M2b_s0 - M2b_s1 | -0.11 [-0.21, -0.03] sep | -0.005 [-0.021, +0.010] ns | -0.54 [-1.04, -0.06] sep | -0.001 [-0.005, +0.000] ns | +0.000 [+0.000, +0.000] ns | +0.0000 [+0.0000, +0.0000] ns |

## k30: 6.0 s horizon  (window states {'LEAD': 743, 'NO_LEAD': 552, 'NOT_STRAIGHT': 931, 'NO_LABEL': 173}; speed proof max 2.62e-05 m/s)

| run | n hw | headway m | n tg | time gap s | n TTC (closing) | min-TTC s | TTC<1.5 s: n, rate | tg<1.0 s rate | LEAD-VIOL at-risk | progress |
|---|---|---|---|---|---|---|---|---|---|---|
| human | 725 | 25.66 [21.63, 29.79] | 659 | 4.00 [3.10, 5.08] | 725 (447) | 19.79 [17.41, 21.94] | 725, 0.004 [0.000, 0.010] | 0.097 [0.031, 0.176] | 0 by construction | 1 |
| M2b_s0 | 722 | 25.62 [21.84, 29.74] | 665 | 4.02 [3.10, 5.27] | 722 (458) | 19.40 [17.31, 21.34] | 722, 0.039 [0.018, 0.065] | 0.120 [0.056, 0.201] | 21/595 = 3.53 % | 0.863 |
| M2b_s1 | 728 | 25.54 [21.63, 29.65] | 663 | 3.95 [3.06, 5.03] | 728 (445) | 19.52 [17.47, 21.32] | 728, 0.055 [0.029, 0.086] | 0.143 [0.075, 0.221] | 36/595 = 6.05 % | 0.919 |
| P10 | 728 | 25.12 [21.21, 29.50] | 663 | 3.83 [2.96, 4.97] | 728 (411) | 19.95 [17.71, 22.02] | 728, 0.088 [0.051, 0.132] | 0.161 [0.092, 0.243] | 48/595 = 8.07 % | 0.938 |

Paired deltas a - b (delta [lo, hi]; sep = CI excludes 0; positive = MORE gap/time/TTC, or MORE violations on the rate rows):

| a - b | headway m | time gap s | min-TTC s | TTC<1.5 s rate | tg<1.0 s rate | LEAD-VIOL at-risk |
|---|---|---|---|---|---|---|
| M2b_s0 - P10 | +0.53 [+0.08, +0.97] sep | +0.186 [+0.044, +0.360] sep | -0.49 [-1.57, +0.65] ns | -0.050 [-0.078, -0.026] sep | -0.041 [-0.069, -0.017] sep | -0.0454 [-0.0752, -0.0189] sep |
| M2b_s1 - P10 | +0.42 [+0.01, +0.85] sep | +0.117 [-0.004, +0.264] ns | -0.43 [-1.27, +0.52] ns | -0.033 [-0.060, -0.009] sep | -0.018 [-0.044, +0.004] ns | -0.0202 [-0.0490, +0.0050] ns |
| M2b_s0 - M2b_s1 | +0.10 [-0.13, +0.33] ns | +0.070 [-0.009, +0.169] ns | -0.04 [-0.67, +0.64] ns | -0.017 [-0.029, -0.007] sep | -0.023 [-0.038, -0.009] sep | -0.0252 [-0.0428, -0.0105] sep |

