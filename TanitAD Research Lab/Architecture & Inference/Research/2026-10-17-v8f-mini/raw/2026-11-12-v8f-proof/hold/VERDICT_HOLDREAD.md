# VERDICT_HOLDREAD (PREREG_V8F_PROOF_ADD15)

Card sha256 168ad1a7ddc98fabdfb080fbfdc5ea171a114f93f674a92d1bfeac1add777f8b; holdread.py sha256 87e65b6376f3d6e10397b863ec8ef54d8e098195694fb7d1612dc5387b891623.
Evidence class MEASURED. Tier T1-kin (kinematic rollout on the model's own actions, logged frames), zero training, zero rollouts: a diagnostic read, never driving performance.
**The hold rule is an EMULATION on banked rollouts, not a rollout.** A held car in a T1-kin rollout decides again at 2 s and 4 s; here it stands for the whole window.
STRATEGIC family: PAUSED (PI item 25), reported as PAUSED, not as a pass.

## Controls (run first)

Read valid: **YES, every control reads its known value**.

| control | what | observed | known | |
|---|---|---|---|---|
| c5_pair1_s0 | REST windows, REST episodes | [211, 29] | [211, 29] | PASS |
| c1_pair1_s0_H | H(0): max abs difference to the banked track; windows held | [0.0, 0] | [0.0, 0] | PASS |
| c2_pair1_s0_H | H(inf) on REST: hit rate; share of STAY among REST; max change outside REST | [0.25118483412322273, 0.0] | [0.25118483412322273, 0.0] | PASS |
| c3_pair1_s0_H | privileged rule: hit rate on STAY; max change on non-STAY windows | [1.0, 0.0] | [1.0, 0.0] | PASS |
| c4_pair1_s0_H | episodes in both fit and held-out fold, per split; held-out folds partition the REST episodes | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | PASS |
| c1_pair1_s0_F | H(0): max abs difference to the banked track; windows held | [0.0, 0] | [0.0, 0] | PASS |
| c2_pair1_s0_F | H(inf) on REST: hit rate; share of STAY among REST; max change outside REST | [0.25118483412322273, 0.0] | [0.25118483412322273, 0.0] | PASS |
| c3_pair1_s0_F | privileged rule: hit rate on STAY; max change on non-STAY windows | [1.0, 0.0] | [1.0, 0.0] | PASS |
| c4_pair1_s0_F | episodes in both fit and held-out fold, per split; held-out folds partition the REST episodes | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | PASS |
| c5_pair1_s1 | REST windows, REST episodes | [211, 29] | [211, 29] | PASS |
| c1_pair1_s1_H | H(0): max abs difference to the banked track; windows held | [0.0, 0] | [0.0, 0] | PASS |
| c2_pair1_s1_H | H(inf) on REST: hit rate; share of STAY among REST; max change outside REST | [0.25118483412322273, 0.0] | [0.25118483412322273, 0.0] | PASS |
| c3_pair1_s1_H | privileged rule: hit rate on STAY; max change on non-STAY windows | [1.0, 0.0] | [1.0, 0.0] | PASS |
| c4_pair1_s1_H | episodes in both fit and held-out fold, per split; held-out folds partition the REST episodes | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | PASS |
| c1_pair1_s1_F | H(0): max abs difference to the banked track; windows held | [0.0, 0] | [0.0, 0] | PASS |
| c2_pair1_s1_F | H(inf) on REST: hit rate; share of STAY among REST; max change outside REST | [0.25118483412322273, 0.0] | [0.25118483412322273, 0.0] | PASS |
| c3_pair1_s1_F | privileged rule: hit rate on STAY; max change on non-STAY windows | [1.0, 0.0] | [1.0, 0.0] | PASS |
| c4_pair1_s1_F | episodes in both fit and held-out fold, per split; held-out folds partition the REST episodes | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | PASS |
| c5_pair2_s0 | REST windows, REST episodes | [211, 29] | [211, 29] | PASS |
| c1_pair2_s0_H | H(0): max abs difference to the banked track; windows held | [0.0, 0] | [0.0, 0] | PASS |
| c2_pair2_s0_H | H(inf) on REST: hit rate; share of STAY among REST; max change outside REST | [0.25118483412322273, 0.0] | [0.25118483412322273, 0.0] | PASS |
| c3_pair2_s0_H | privileged rule: hit rate on STAY; max change on non-STAY windows | [1.0, 0.0] | [1.0, 0.0] | PASS |
| c4_pair2_s0_H | episodes in both fit and held-out fold, per split; held-out folds partition the REST episodes | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | PASS |
| c1_pair2_s0_F | H(0): max abs difference to the banked track; windows held | [0.0, 0] | [0.0, 0] | PASS |
| c2_pair2_s0_F | H(inf) on REST: hit rate; share of STAY among REST; max change outside REST | [0.25118483412322273, 0.0] | [0.25118483412322273, 0.0] | PASS |
| c3_pair2_s0_F | privileged rule: hit rate on STAY; max change on non-STAY windows | [1.0, 0.0] | [1.0, 0.0] | PASS |
| c4_pair2_s0_F | episodes in both fit and held-out fold, per split; held-out folds partition the REST episodes | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | PASS |
| c5_pair2_s1 | REST windows, REST episodes | [211, 29] | [211, 29] | PASS |
| c1_pair2_s1_H | H(0): max abs difference to the banked track; windows held | [0.0, 0] | [0.0, 0] | PASS |
| c2_pair2_s1_H | H(inf) on REST: hit rate; share of STAY among REST; max change outside REST | [0.25118483412322273, 0.0] | [0.25118483412322273, 0.0] | PASS |
| c3_pair2_s1_H | privileged rule: hit rate on STAY; max change on non-STAY windows | [1.0, 0.0] | [1.0, 0.0] | PASS |
| c4_pair2_s1_H | episodes in both fit and held-out fold, per split; held-out folds partition the REST episodes | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | PASS |
| c1_pair2_s1_F | H(0): max abs difference to the banked track; windows held | [0.0, 0] | [0.0, 0] | PASS |
| c2_pair2_s1_F | H(inf) on REST: hit rate; share of STAY among REST; max change outside REST | [0.25118483412322273, 0.0] | [0.25118483412322273, 0.0] | PASS |
| c3_pair2_s1_F | privileged rule: hit rate on STAY; max change on non-STAY windows | [1.0, 0.0] | [1.0, 0.0] | PASS |
| c4_pair2_s1_F | episodes in both fit and held-out fold, per split; held-out folds partition the REST episodes | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | PASS |
| c6_pair1_s0_H | inverted rule mean ADE6 on REST above the rule as defined (hierarchy) | [4.1197, 2.3729] | inverted > rule | PASS |
| c6_pair1_s1_H | inverted rule mean ADE6 on REST above the rule as defined (hierarchy) | [4.2872, 2.4337] | inverted > rule | PASS |
| c6info_pair1_s0_F | inverted rule mean ADE6 on REST above the rule as defined (flat) | [4.4158, 3.0216] | inverted > rule | PASS |
| c6info_pair1_s1_F | inverted rule mean ADE6 on REST above the rule as defined (flat) | [4.3938, 2.8892] | inverted > rule | PASS |

## Card outcome

* Q1 label (pair 1): **CREEP**; Q2 label (pair 1, hierarchy): **CUE-WEAK**; Q3 card outcome (pair 1, hierarchy, ADE6 on REST): **HOLD-UNDECIDED**.
* Same labels on pair 2 (printed beside, decide nothing): Q1 CREEP; Q2 CUE-WEAK; Q3 HOLD-HELPS.

### Counts, pair1

| seed | REST windows (episodes) | STAY | GO | all windows (episodes) |
|---|---|---|---|---|
| 0 | 211 (29) | 53 (14) | 158 (24) | 2399 (141) |
| 1 | 211 (29) | 53 (14) | 158 (24) | 2399 (141) |

### Counts, pair2

| seed | REST windows (episodes) | STAY | GO | all windows (episodes) |
|---|---|---|---|---|
| 0 | 211 (29) | 53 (14) | 158 (24) | 2399 (141) |
| 1 | 211 (29) | 53 (14) | 158 (24) | 2399 (141) |

## Q1, creep on STAY windows: pair1

T_cl = ||cl[29]|| in metres; interval: episode-cluster bootstrap (B 2,000, seed 0). Hit: ||cl[29] - g[29]|| <= 2.0 m.

| seed | arm | STAY windows (episodes) | median T_cl [95 %] | 90th percentile T_cl [95 %] | hit rate [95 %] |
|---|---|---|---|---|---|
| 0 | hierarchy | 53 (14) | 8.680 [6.869, 10.484] | 16.427 [10.853, 21.049] | 0.057 [0.000, 0.153] |
| 0 | flat | 53 (14) | 12.837 [11.903, 14.512] | 22.033 [14.131, 22.979] | 0.000 [0.000, 0.000] |
| 1 | hierarchy | 53 (14) | 11.832 [8.531, 14.450] | 18.990 [16.274, 27.729] | 0.057 [0.000, 0.129] |
| 1 | flat | 53 (14) | 11.331 [9.805, 13.904] | 15.157 [13.972, 17.957] | 0.038 [0.000, 0.140] |

## Q1, creep on STAY windows: pair2

T_cl = ||cl[29]|| in metres; interval: episode-cluster bootstrap (B 2,000, seed 0). Hit: ||cl[29] - g[29]|| <= 2.0 m.

| seed | arm | STAY windows (episodes) | median T_cl [95 %] | 90th percentile T_cl [95 %] | hit rate [95 %] |
|---|---|---|---|---|---|
| 0 | hierarchy | 53 (14) | 10.959 [9.045, 12.271] | 16.572 [13.031, 17.663] | 0.019 [0.000, 0.067] |
| 0 | flat | 53 (14) | 7.157 [5.867, 11.419] | 16.291 [8.380, 24.805] | 0.038 [0.000, 0.111] |
| 1 | hierarchy | 53 (14) | 10.130 [7.226, 12.242] | 18.737 [12.242, 26.463] | 0.019 [0.000, 0.067] |
| 1 | flat | 53 (14) | 6.809 [4.981, 12.467] | 16.117 [8.454, 20.434] | 0.000 [0.000, 0.000] |

Q1 label, pair 1: **CREEP** (CREEP if the hierarchy's median T_cl on STAY windows is above 2.0 m at both seeds of pair 1, else HOLDS).

## Q2, does the plan at the window start separate GO from STAY at rest: pair1

AUROC of T_ol = ||ol[29]|| for GO against STAY on REST windows, ties counted half; beside it the AUROC of `v0` alone (the label comes from the logged motion, so `v0` partly reads its source).

| seed | arm | AUROC of T_ol [95 %] | AUROC of v0 alone [95 %] | GO / STAY windows |
|---|---|---|---|---|
| 0 | hierarchy | 0.803 [0.711, 0.894] | 0.701 [0.589, 0.796] | 158 / 53 |
| 0 | flat | 0.757 [0.667, 0.845] | 0.701 [0.589, 0.796] | 158 / 53 |
| 1 | hierarchy | 0.807 [0.706, 0.907] | 0.701 [0.589, 0.796] | 158 / 53 |
| 1 | flat | 0.749 [0.656, 0.851] | 0.701 [0.589, 0.796] | 158 / 53 |

## Q2, does the plan at the window start separate GO from STAY at rest: pair2

AUROC of T_ol = ||ol[29]|| for GO against STAY on REST windows, ties counted half; beside it the AUROC of `v0` alone (the label comes from the logged motion, so `v0` partly reads its source).

| seed | arm | AUROC of T_ol [95 %] | AUROC of v0 alone [95 %] | GO / STAY windows |
|---|---|---|---|---|
| 0 | hierarchy | 0.789 [0.712, 0.872] | 0.701 [0.589, 0.796] | 158 / 53 |
| 0 | flat | 0.768 [0.680, 0.858] | 0.701 [0.589, 0.796] | 158 / 53 |
| 1 | hierarchy | 0.815 [0.709, 0.915] | 0.701 [0.589, 0.796] | 158 / 53 |
| 1 | flat | 0.732 [0.633, 0.842] | 0.701 [0.589, 0.796] | 158 / 53 |

Q2 label, pair 1, hierarchy: **CUE-WEAK** (CUE-PRESENT if the lower bound is above 0.80 at both seeds, CUE-ABSENT if the upper bound is below 0.65 at both, else CUE-WEAK).

## Q3, the hold rule (EMULATION), cross-fitted: pair1

Differences are ruled arm minus the arm as banked; the interval is the paired episode-cluster bootstrap with the cross-fitted tracks held fixed. ADE6 in metres; hit-rate change in share points of windows.

| seed | arm | REST: ADE6 change [95 %] | all windows: ADE6 change [95 %] | REST: hit-rate change [95 %] | all windows: hit-rate change [95 %] | banked / ruled ADE6 on REST | banked / ruled hit on REST | windows held (STAY / GO) |
|---|---|---|---|---|---|---|---|---|
| 0 | hierarchy | -0.2390 [-0.5949, +0.0730] | -0.0210 [-0.0566, +0.0059] | +0.1754 [+0.0658, +0.2929] | +0.0154 [+0.0050, +0.0275] | 2.612 / 2.373 | 0.071 / 0.246 | 114 (45 / 69) |
| 0 | flat | -0.5352 [-1.1289, +0.1180] | -0.0471 [-0.1049, +0.0094] | +0.2275 [+0.1046, +0.3660] | +0.0200 [+0.0079, +0.0346] | 3.557 / 3.022 | 0.019 / 0.246 | 135 (49 / 86) |
| 1 | hierarchy | -0.4065 [-0.7716, +0.0157] | -0.0358 [-0.0716, +0.0012] | +0.1991 [+0.0996, +0.3294] | +0.0175 [+0.0079, +0.0304] | 2.840 / 2.434 | 0.052 / 0.251 | 117 (45 / 72) |
| 1 | flat | -0.5132 [-1.0008, +0.1145] | -0.0451 [-0.0944, +0.0086] | +0.2227 [+0.1031, +0.3610] | +0.0196 [+0.0079, +0.0338] | 3.402 / 2.889 | 0.028 / 0.251 | 141 (51 / 90) |

Chosen theta per fold (pair1); grid {0.5, 1, 2, 3, 4, 6, 8, 12} m, lowest mean ADE6 on the four fit folds, ties to the smaller theta:

| seed | arm | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|---|
| 0 | hierarchy | 6 (held-out 31) | 6 (held-out 39) | 2 (held-out 43) | 8 (held-out 51) | 6 (held-out 47) |
| 0 | flat | 8 (held-out 31) | 8 (held-out 39) | 4 (held-out 43) | 8 (held-out 51) | 12 (held-out 47) |
| 1 | hierarchy | 4 (held-out 31) | 4 (held-out 39) | 3 (held-out 43) | 4 (held-out 51) | 6 (held-out 47) |
| 1 | flat | 8 (held-out 31) | 8 (held-out 39) | 6 (held-out 43) | 6 (held-out 51) | 12 (held-out 47) |

Privileged ceiling (pair1): the rule with the logged STAY label in place of the cue (a label, not an input), same differences:

| seed | arm | REST: ADE6 change [95 %] | all windows: ADE6 change [95 %] | REST: hit-rate change [95 %] | all windows: hit-rate change [95 %] | ceiling ADE6 on REST |
|---|---|---|---|---|---|---|
| 0 | hierarchy | -0.5421 [-0.8550, -0.2703] | -0.0477 [-0.0856, -0.0194] | +0.2370 [+0.1330, +0.3552] | +0.0208 [+0.0100, +0.0346] | 2.070 |
| 0 | flat | -0.6770 [-1.0527, -0.3489] | -0.0595 [-0.1030, -0.0258] | +0.2512 [+0.1361, +0.3846] | +0.0221 [+0.0100, +0.0371] | 2.880 |
| 1 | hierarchy | -0.5147 [-0.8261, -0.2544] | -0.0453 [-0.0782, -0.0197] | +0.2370 [+0.1258, +0.3671] | +0.0208 [+0.0096, +0.0354] | 2.325 |
| 1 | flat | -0.5858 [-0.9197, -0.2982] | -0.0515 [-0.0904, -0.0222] | +0.2417 [+0.1270, +0.3760] | +0.0213 [+0.0096, +0.0363] | 2.817 |

## Q3, the hold rule (EMULATION), cross-fitted: pair2

Differences are ruled arm minus the arm as banked; the interval is the paired episode-cluster bootstrap with the cross-fitted tracks held fixed. ADE6 in metres; hit-rate change in share points of windows.

| seed | arm | REST: ADE6 change [95 %] | all windows: ADE6 change [95 %] | REST: hit-rate change [95 %] | all windows: hit-rate change [95 %] | banked / ruled ADE6 on REST | banked / ruled hit on REST | windows held (STAY / GO) |
|---|---|---|---|---|---|---|---|---|
| 0 | hierarchy | -0.9045 [-1.3887, -0.3492] | -0.0796 [-0.1389, -0.0261] | +0.2464 [+0.1319, +0.3810] | +0.0217 [+0.0100, +0.0367] | 3.734 / 2.830 | 0.005 / 0.251 | 142 (53 / 89) |
| 0 | flat | -0.7503 [-1.1489, -0.3667] | -0.0660 [-0.1135, -0.0266] | +0.2417 [+0.1256, +0.3770] | +0.0213 [+0.0092, +0.0363] | 3.428 / 2.678 | 0.028 / 0.270 | 134 (53 / 81) |
| 1 | hierarchy | -0.5812 [-0.9751, -0.1207] | -0.0511 [-0.0954, -0.0090] | +0.2180 [+0.1050, +0.3509] | +0.0192 [+0.0079, +0.0329] | 3.331 / 2.750 | 0.024 / 0.242 | 134 (50 / 84) |
| 1 | flat | -0.5933 [-0.9776, -0.1121] | -0.0522 [-0.0976, -0.0088] | +0.2370 [+0.1255, +0.3689] | +0.0208 [+0.0092, +0.0350] | 3.504 / 2.911 | 0.009 / 0.246 | 141 (51 / 90) |

Chosen theta per fold (pair2); grid {0.5, 1, 2, 3, 4, 6, 8, 12} m, lowest mean ADE6 on the four fit folds, ties to the smaller theta:

| seed | arm | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|---|
| 0 | hierarchy | 6 (held-out 31) | 6 (held-out 39) | 6 (held-out 43) | 6 (held-out 51) | 12 (held-out 47) |
| 0 | flat | 8 (held-out 31) | 8 (held-out 39) | 6 (held-out 43) | 8 (held-out 51) | 8 (held-out 47) |
| 1 | hierarchy | 6 (held-out 31) | 6 (held-out 39) | 6 (held-out 43) | 6 (held-out 51) | 8 (held-out 47) |
| 1 | flat | 8 (held-out 31) | 8 (held-out 39) | 6 (held-out 43) | 8 (held-out 51) | 12 (held-out 47) |

Privileged ceiling (pair2): the rule with the logged STAY label in place of the cue (a label, not an input), same differences:

| seed | arm | REST: ADE6 change [95 %] | all windows: ADE6 change [95 %] | REST: hit-rate change [95 %] | all windows: hit-rate change [95 %] | ceiling ADE6 on REST |
|---|---|---|---|---|---|---|
| 0 | hierarchy | -0.6355 [-0.9656, -0.3299] | -0.0559 [-0.0976, -0.0242] | +0.2464 [+0.1319, +0.3810] | +0.0217 [+0.0100, +0.0367] | 3.099 |
| 0 | flat | -0.5661 [-0.9109, -0.2581] | -0.0498 [-0.0907, -0.0194] | +0.2417 [+0.1256, +0.3770] | +0.0213 [+0.0092, +0.0363] | 2.862 |
| 1 | hierarchy | -0.5791 [-0.9717, -0.2562] | -0.0509 [-0.0960, -0.0186] | +0.2464 [+0.1319, +0.3810] | +0.0217 [+0.0100, +0.0367] | 2.752 |
| 1 | flat | -0.5583 [-0.9006, -0.2629] | -0.0491 [-0.0910, -0.0188] | +0.2512 [+0.1361, +0.3846] | +0.0221 [+0.0100, +0.0371] | 2.946 |

## Four families, ruled arm minus the arm as banked, REST windows (cross-fitted rule; privileged ceiling in brackets)

LONGITUDINAL = mean absolute speed error over the 30 knots (m/s); LATERAL = mean absolute cross-track error at 6 s (m); TACTICAL = share of windows whose joint cell equals the logged cell (share points); STRATEGIC = PAUSED (PI item 25).

| pair | seed | arm | LONGITUDINAL change [95 %] | LATERAL change [95 %] | TACTICAL change [95 %] | STRATEGIC |
|---|---|---|---|---|---|---|
| pair1 | 0 | hierarchy | -0.4279 [-0.8919, -0.0819] (-0.5751) | -1.9442 [-2.7980, -1.1555] (-1.0959) | +0.1232 [+0.0263, +0.2486] (+0.1706) | PAUSED |
| pair1 | 0 | flat | -1.9389 [-2.7223, -1.1952] (-1.3052) | -6.4406 [-8.5239, -4.2964] (-3.3329) | +0.1754 [+0.0649, +0.3128] (+0.1754) | PAUSED |
| pair1 | 1 | hierarchy | -1.2678 [-1.8335, -0.8137] (-0.8963) | -4.4541 [-5.8009, -3.2185] (-2.2587) | +0.1564 [+0.0564, +0.2935] (+0.1754) | PAUSED |
| pair1 | 1 | flat | -1.2150 [-1.7484, -0.7413] (-0.8964) | -5.2568 [-6.5900, -3.9805] (-2.1699) | +0.1517 [+0.0324, +0.2909] (+0.1754) | PAUSED |
| pair2 | 0 | hierarchy | -1.7294 [-2.2747, -1.1333] (-1.0390) | -5.8718 [-7.1157, -4.4328] (-2.0763) | +0.1848 [+0.0756, +0.3171] (+0.1754) | PAUSED |
| pair2 | 0 | flat | -1.8991 [-2.5662, -1.2700] (-1.1994) | -4.4677 [-5.9182, -2.9712] (-2.1092) | +0.1611 [+0.0500, +0.2975] (+0.1706) | PAUSED |
| pair2 | 1 | hierarchy | -1.6814 [-2.3048, -1.0848] (-1.0921) | -4.7079 [-5.7211, -3.4644] (-2.0180) | +0.1801 [+0.0763, +0.3114] (+0.1754) | PAUSED |
| pair2 | 1 | flat | -1.0590 [-1.5600, -0.5924] (-0.7762) | -4.0454 [-4.9816, -3.0344] (-1.8050) | +0.1611 [+0.0471, +0.3000] (+0.1754) | PAUSED |

Windows on REST that used the ego-frame lateral axis because the logged future has fewer than 2 valid steps (counted, as the card requires): {"pair1|s0|H": 31, "pair1|s0|F": 31, "pair1|s1|H": 31, "pair1|s1|F": 31, "pair2|s0|H": 31, "pair2|s0|F": 31, "pair2|s1|H": 31, "pair2|s1|F": 31}.

## What follows (the paragraph the card commits for the outcome reached)

HOLD-UNDECIDED with CUE-WEAK: the rule lacks a cue; the ceiling row says what a cue would be worth, and the lever is a go / stay cue at rest from perception, not the rule.

## Limits (card section 6)

* An emulation: a held car in a T1-kin rollout decides again at 2 s and 4 s; here it stands for the whole window. The rollout version is the committed next step and is the number that would count.
* theta is fitted on evaluation episodes by cross-fitting, not on training clips (no training rollouts are banked on this box).
* 211 windows of 29 episodes: wide intervals are expected. STAY is defined with the logged future, so it is a label, not an input.
* One training run per arm per seed: a separated interval is necessary and not sufficient.

