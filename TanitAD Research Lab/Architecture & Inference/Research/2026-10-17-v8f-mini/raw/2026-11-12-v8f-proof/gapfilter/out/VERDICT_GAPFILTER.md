# VERDICT_GAPFILTER: card PREREG_V8F_PROOF_ADD28 (the closing speed of the lead from a causal filter over the head's own gap reads)

Tier T0 (a diagnostic on logged frames, never driving performance). STRATEGIC: PAUSED (PI item 25). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read. Evidence class MEASURED (ours, T0).

Counts: 1948 windows, 429 scored, 413 scored with a closing-speed label, 48 scored episodes; episode-cluster bootstrap B 2000, rng 0.

## Controls (first)

- f1_label_path: **PASS** R2 skill this path 0.7499068, banked 0.7499068 (n=413)
- f2_sign_units_alignment: **FAIL** E1 K=5 on the label gap: R2 +0.367 (bar at least 0.8), n=413 of 413
- f6_records|LS_s0: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|LS_s0: **PASS** selection with real labels E2_a0.7_b0.1, with noise labels E2_a0.7_b0.1; identical True
- x1_alignment_m1_delta|LS_s0: **PASS** M1 +0.1547 vs N1 result +0.1547; DELTA of the head -0.0036 vs -0.0036
- f3_causal|LS_s0: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|LS_s0: **PASS** M1F with permuted sequences -0.183 (bar at most +0.02), seeds 0 to 4: -0.183, -0.147, -0.265, -0.306, -0.205; identity permutation reproduces F, max abs diff 0
- f6_records|LS_s1: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|LS_s1: **PASS** selection with real labels E2_a0.7_b0.1, with noise labels E2_a0.7_b0.1; identical True
- x1_alignment_m1_delta|LS_s1: **PASS** M1 +0.0136 vs N1 result +0.0136; DELTA of the head -0.0040 vs -0.0040
- f3_causal|LS_s1: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|LS_s1: **PASS** M1F with permuted sequences -0.215 (bar at most +0.02), seeds 0 to 4: -0.215, -0.164, -0.321, -0.238, -0.169; identity permutation reproduces F, max abs diff 0
- f6_records|LSH_s0: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|LSH_s0: **PASS** selection with real labels E2_a0.7_b0.1, with noise labels E2_a0.7_b0.1; identical True
- x1_alignment_m1_delta|LSH_s0: **PASS** M1 +0.0915 vs N1 result +0.0915; DELTA of the head +0.0195 vs +0.0195
- f3_causal|LSH_s0: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|LSH_s0: **PASS** M1F with permuted sequences -0.221 (bar at most +0.02), seeds 0 to 4: -0.221, -0.244, -0.320, -0.316, -0.251; identity permutation reproduces F, max abs diff 0
- f6_records|LSH_s1: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|LSH_s1: **PASS** selection with real labels E2_a0.7_b0.1, with noise labels E2_a0.7_b0.1; identical True
- x1_alignment_m1_delta|LSH_s1: **PASS** M1 +0.0264 vs N1 result +0.0264; DELTA of the head +0.0010 vs +0.0010
- f3_causal|LSH_s1: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|LSH_s1: **PASS** M1F with permuted sequences -0.184 (bar at most +0.02), seeds 0 to 4: -0.184, -0.191, -0.293, -0.261, -0.209; identity permutation reproduces F, max abs diff 0
- f6_records|LSD_s0: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|LSD_s0: **PASS** selection with real labels E2_a0.7_b0.1, with noise labels E2_a0.7_b0.1; identical True
- x1_alignment_m1_delta|LSD_s0: **PASS** M1 +0.1782 vs N1 result +0.1782; DELTA of the head +0.0015 vs +0.0015
- f3_causal|LSD_s0: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|LSD_s0: **PASS** M1F with permuted sequences -0.156 (bar at most +0.02), seeds 0 to 4: -0.156, -0.106, -0.195, -0.144, -0.158; identity permutation reproduces F, max abs diff 0
- f6_records|LSD_s1: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|LSD_s1: **PASS** selection with real labels E2_a0.7_b0.05, with noise labels E2_a0.7_b0.05; identical True
- x1_alignment_m1_delta|LSD_s1: **PASS** M1 +0.0890 vs N1 result +0.0890; DELTA of the head -0.0323 vs -0.0323
- f3_causal|LSD_s1: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|LSD_s1: **PASS** M1F with permuted sequences -0.100 (bar at most +0.02), seeds 0 to 4: -0.100, -0.088, -0.217, -0.162, -0.115; identity permutation reproduces F, max abs diff 0

Controls failed: f2_sign_units_alignment

## Names (card section 5)

No outcome is named. controls failed: f2_sign_units_alignment

Floor: card value 0.089; measured |M1(LSD_s0) - M1(LSD_s1)| = 0.0893.

## Per arm (413 scored windows with a closing-speed label for M1, M1F, paired; 429 scored windows for DELTA)

| arm | selected F | a, b | M1 head closing R2 | M1F R2 | M1F minus M1 | DELTA-F | DELTA-F permuted (seed 0) | F estimate missing (of 413) |
|---|---|---|---|---|---|---|---|---|
| LS_s0 | E2_a0.7_b0.1 | +0.410, -0.052 | +0.155 [-0.043, +0.283] n=413 | +0.130 [-0.058, +0.256] n=413 | -0.025 [-0.163, +0.113] n=413 | +0.001 [-0.041, +0.044] n=429 | -0.003 | 42 |
| LS_s1 | E2_a0.7_b0.1 | +0.368, -0.002 | +0.014 [-0.278, +0.192] n=413 | +0.177 [-0.019, +0.290] n=413 | +0.164 [-0.035, +0.419] n=413 | -0.003 [-0.034, +0.026] n=429 | -0.002 | 51 |
| LSH_s0 | E2_a0.7_b0.1 | +0.433, -0.047 | +0.092 [-0.138, +0.231] n=413 | +0.201 [-0.005, +0.322] n=413 | +0.110 [-0.090, +0.312] n=413 | -0.002 [-0.028, +0.025] n=429 | -0.001 | 33 |
| LSH_s1 | E2_a0.7_b0.1 | +0.423, -0.034 | +0.026 [-0.192, +0.163] n=413 | +0.158 [-0.064, +0.282] n=413 | +0.132 [-0.079, +0.317] n=413 | +0.006 [-0.025, +0.033] n=429 | -0.003 | 38 |
| LSD_s0 | E2_a0.7_b0.1 | +0.370, -0.067 | +0.178 [-0.059, +0.335] n=413 | +0.149 [+0.008, +0.229] n=413 | -0.029 [-0.178, +0.145] n=413 | +0.029 [-0.021, +0.069] n=429 | -0.001 | 42 |
| LSD_s1 | E2_a0.7_b0.05 | +0.543, -0.012 | +0.089 [-0.127, +0.234] n=413 | +0.102 [-0.062, +0.206] n=413 | +0.013 [-0.141, +0.185] n=413 | +0.024 [-0.072, +0.103] n=429 | -0.003 | 53 |

### LS_s0

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K10 (mean fold MSE 7.9199), best E2 E2_a0.7_b0.1 (7.6930); chosen E2_a0.7_b0.1.
- Windows where F is missing and the head closing output is used: 42 of 413 valid, 43 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) -0.004 [-0.026, +0.016]; DELTA-F permuted seeds 0 to 4: -0.003, +0.002, +0.001, -0.010, -0.004.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.139 [+0.109, +0.168]; cross-fitted five-fold +0.137 [+0.107, +0.166]; head closing output +0.209 [+0.167, +0.249].
- E3 blend (names nothing): R2 +0.239 [+0.004, +0.391] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.101 [-0.086, +0.228] n=413, chosen per fold {0: 'E2_a0.7_b0.1', 1: 'E2_a0.7_b0.1', 2: 'E2_a0.7_b0.1', 3: 'E2_a0.7_b0.1', 4: 'E2_a0.7_b0.1'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 7.02 m [+5.696, +8.657]; RMSE of the change between consecutive rows 2.23 m [+1.539, +2.944]; lag-1 autocorrelation +0.946 [+0.899, +0.975]. Rows with head presence at least 0.5 (n=2203): RMSE 5.97 m, change 2.14 m, lag-1 +0.933.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.041 | [-0.086, +0.119] | 413 | 44 | 8.0099 |
| E1_K5 | +0.086 | [-0.060, +0.175] | 413 | 43 | 7.9625 |
| E1_K8 | +0.103 | [-0.044, +0.197] | 413 | 42 | 7.9241 |
| E1_K10 | +0.106 | [-0.046, +0.206] | 413 | 42 | 7.9199 |
| E1_K15 | +0.101 | [-0.063, +0.210] | 413 | 40 | 7.9719 |
| E2_a0.2_b0.02 | +0.082 | [-0.110, +0.205] | 413 | 42 | 7.9792 |
| E2_a0.2_b0.05 | +0.113 | [-0.050, +0.220] | 413 | 42 | 7.8846 |
| E2_a0.2_b0.1 | +0.082 | [-0.050, +0.161] | 413 | 42 | 8.0659 |
| E2_a0.2_b0.2 | +0.040 | [-0.073, +0.106] | 413 | 42 | 8.3337 |
| E2_a0.35_b0.02 | +0.070 | [-0.111, +0.189] | 413 | 42 | 7.9807 |
| E2_a0.35_b0.05 | +0.121 | [-0.066, +0.245] | 413 | 42 | 7.7762 |
| E2_a0.35_b0.1 | +0.114 | [-0.040, +0.212] | 413 | 42 | 7.8682 |
| E2_a0.35_b0.2 | +0.072 | [-0.053, +0.149] | 413 | 42 | 8.1306 |
| E2_a0.5_b0.02 | +0.060 | [-0.116, +0.170] | 413 | 42 | 8.0209 |
| E2_a0.5_b0.05 | +0.115 | [-0.082, +0.245] | 413 | 42 | 7.7561 |
| E2_a0.5_b0.1 | +0.127 | [-0.045, +0.241] | 413 | 42 | 7.7592 |
| E2_a0.5_b0.2 | +0.095 | [-0.044, +0.184] | 413 | 42 | 7.9788 |
| E2_a0.7_b0.02 | +0.049 | [-0.117, +0.154] | 413 | 42 | 8.0743 |
| E2_a0.7_b0.05 | +0.103 | [-0.096, +0.229] | 413 | 42 | 7.7810 |
| E2_a0.7_b0.1 | +0.130 | [-0.058, +0.256] | 413 | 42 | 7.6930 |
| E2_a0.7_b0.2 | +0.115 | [-0.041, +0.214] | 413 | 42 | 7.8407 |

### LS_s1

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K10 (mean fold MSE 8.0198), best E2 E2_a0.7_b0.1 (7.7979); chosen E2_a0.7_b0.1.
- Windows where F is missing and the head closing output is used: 51 of 413 valid, 53 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) -0.004 [-0.059, +0.051]; DELTA-F permuted seeds 0 to 4: -0.002, +0.003, -0.003, +0.012, -0.003.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.128 [+0.094, +0.160]; cross-fitted five-fold +0.126 [+0.091, +0.157]; head closing output +0.182 [+0.129, +0.227].
- E3 blend (names nothing): R2 +0.197 [-0.027, +0.339] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.124 [-0.102, +0.265] n=413, chosen per fold {0: 'E2_a0.7_b0.1', 1: 'E2_a0.7_b0.1', 2: 'E2_a0.7_b0.05', 3: 'E2_a0.7_b0.1', 4: 'E2_a0.7_b0.1'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 8.26 m [+6.904, +9.887]; RMSE of the change between consecutive rows 2.32 m [+1.655, +3.029]; lag-1 autocorrelation +0.960 [+0.924, +0.980]. Rows with head presence at least 0.5 (n=2168): RMSE 7.14 m, change 2.30 m, lag-1 +0.946.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.068 | [-0.054, +0.141] | 413 | 61 | 8.2342 |
| E1_K5 | +0.097 | [-0.045, +0.180] | 413 | 56 | 8.1119 |
| E1_K8 | +0.133 | [-0.019, +0.221] | 413 | 51 | 8.0445 |
| E1_K10 | +0.143 | [-0.013, +0.238] | 413 | 49 | 8.0198 |
| E1_K15 | +0.160 | [-0.003, +0.264] | 413 | 44 | 8.0502 |
| E2_a0.2_b0.02 | +0.137 | [-0.066, +0.251] | 413 | 51 | 8.0283 |
| E2_a0.2_b0.05 | +0.159 | [-0.010, +0.257] | 413 | 51 | 8.0153 |
| E2_a0.2_b0.1 | +0.091 | [-0.047, +0.170] | 413 | 51 | 8.2368 |
| E2_a0.2_b0.2 | +0.048 | [-0.067, +0.114] | 413 | 51 | 8.4565 |
| E2_a0.35_b0.02 | +0.118 | [-0.081, +0.230] | 413 | 51 | 8.0311 |
| E2_a0.35_b0.05 | +0.172 | [-0.027, +0.286] | 413 | 51 | 7.8726 |
| E2_a0.35_b0.1 | +0.139 | [-0.024, +0.233] | 413 | 51 | 8.0122 |
| E2_a0.35_b0.2 | +0.086 | [-0.043, +0.164] | 413 | 51 | 8.2621 |
| E2_a0.5_b0.02 | +0.103 | [-0.086, +0.211] | 413 | 51 | 8.0760 |
| E2_a0.5_b0.05 | +0.168 | [-0.041, +0.287] | 413 | 51 | 7.8339 |
| E2_a0.5_b0.1 | +0.164 | [-0.014, +0.268] | 413 | 51 | 7.8828 |
| E2_a0.5_b0.2 | +0.116 | [-0.023, +0.200] | 413 | 51 | 8.1138 |
| E2_a0.7_b0.02 | +0.089 | [-0.090, +0.194] | 413 | 51 | 8.1358 |
| E2_a0.7_b0.05 | +0.156 | [-0.054, +0.274] | 413 | 51 | 7.8476 |
| E2_a0.7_b0.1 | +0.177 | [-0.019, +0.290] | 413 | 51 | 7.7979 |
| E2_a0.7_b0.2 | +0.146 | [-0.012, +0.239] | 413 | 51 | 7.9741 |

### LSH_s0

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K10 (mean fold MSE 7.7637), best E2 E2_a0.7_b0.1 (7.4634); chosen E2_a0.7_b0.1.
- Windows where F is missing and the head closing output is used: 33 of 413 valid, 34 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) +0.019 [-0.027, +0.067]; DELTA-F permuted seeds 0 to 4: -0.001, -0.000, +0.005, -0.004, -0.006.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.165 [+0.138, +0.192]; cross-fitted five-fold +0.163 [+0.135, +0.190]; head closing output +0.201 [+0.159, +0.239].
- E3 blend (names nothing): R2 +0.256 [+0.021, +0.390] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.146 [-0.035, +0.266] n=413, chosen per fold {0: 'E2_a0.5_b0.05', 1: 'E2_a0.7_b0.05', 2: 'E2_a0.7_b0.05', 3: 'E2_a0.7_b0.05', 4: 'E2_a0.7_b0.1'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 8.05 m [+6.627, +9.720]; RMSE of the change between consecutive rows 2.34 m [+1.632, +3.080]; lag-1 autocorrelation +0.947 [+0.901, +0.975]. Rows with head presence at least 0.5 (n=2277): RMSE 7.00 m, change 2.35 m, lag-1 +0.930.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.089 | [-0.048, +0.162] | 413 | 41 | 8.0022 |
| E1_K5 | +0.115 | [-0.048, +0.205] | 413 | 36 | 7.8710 |
| E1_K8 | +0.150 | [-0.023, +0.244] | 413 | 33 | 7.8073 |
| E1_K10 | +0.147 | [-0.044, +0.254] | 413 | 33 | 7.7637 |
| E1_K15 | +0.147 | [-0.038, +0.257] | 413 | 33 | 7.7961 |
| E2_a0.2_b0.02 | +0.153 | [-0.042, +0.269] | 413 | 33 | 7.7065 |
| E2_a0.2_b0.05 | +0.181 | [+0.001, +0.284] | 413 | 33 | 7.7370 |
| E2_a0.2_b0.1 | +0.105 | [-0.065, +0.195] | 413 | 33 | 7.9880 |
| E2_a0.2_b0.2 | +0.056 | [-0.063, +0.114] | 413 | 33 | 8.2878 |
| E2_a0.35_b0.02 | +0.136 | [-0.054, +0.248] | 413 | 33 | 7.6946 |
| E2_a0.35_b0.05 | +0.194 | [-0.007, +0.311] | 413 | 33 | 7.5446 |
| E2_a0.35_b0.1 | +0.157 | [-0.036, +0.262] | 413 | 33 | 7.7312 |
| E2_a0.35_b0.2 | +0.097 | [-0.048, +0.173] | 413 | 33 | 8.0565 |
| E2_a0.5_b0.02 | +0.124 | [-0.055, +0.230] | 413 | 33 | 7.7390 |
| E2_a0.5_b0.05 | +0.189 | [-0.020, +0.314] | 413 | 33 | 7.4846 |
| E2_a0.5_b0.1 | +0.185 | [-0.017, +0.300] | 413 | 33 | 7.5750 |
| E2_a0.5_b0.2 | +0.131 | [-0.036, +0.220] | 413 | 33 | 7.8781 |
| E2_a0.7_b0.02 | +0.113 | [-0.055, +0.214] | 413 | 33 | 7.8047 |
| E2_a0.7_b0.05 | +0.178 | [-0.026, +0.303] | 413 | 33 | 7.4882 |
| E2_a0.7_b0.1 | +0.201 | [-0.005, +0.322] | 413 | 33 | 7.4634 |
| E2_a0.7_b0.2 | +0.166 | [-0.020, +0.267] | 413 | 33 | 7.7061 |

### LSH_s1

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K15 (mean fold MSE 7.8672), best E2 E2_a0.7_b0.1 (7.6220); chosen E2_a0.7_b0.1.
- Windows where F is missing and the head closing output is used: 38 of 413 valid, 40 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) +0.001 [-0.085, +0.060]; DELTA-F permuted seeds 0 to 4: -0.003, -0.002, +0.003, -0.011, +0.000.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.149 [+0.117, +0.178]; cross-fitted five-fold +0.147 [+0.114, +0.176]; head closing output +0.186 [+0.140, +0.227].
- E3 blend (names nothing): R2 +0.189 [-0.042, +0.337] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.134 [-0.067, +0.255] n=413, chosen per fold {0: 'E2_a0.7_b0.1', 1: 'E2_a0.7_b0.1', 2: 'E2_a0.7_b0.1', 3: 'E2_a0.7_b0.1', 4: 'E2_a0.7_b0.1'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 7.08 m [+5.806, +8.581]; RMSE of the change between consecutive rows 2.31 m [+1.600, +3.026]; lag-1 autocorrelation +0.945 [+0.896, +0.974]. Rows with head presence at least 0.5 (n=2231): RMSE 6.45 m, change 2.34 m, lag-1 +0.933.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.069 | [-0.075, +0.140] | 413 | 44 | 8.1480 |
| E1_K5 | +0.107 | [-0.060, +0.188] | 413 | 42 | 7.9823 |
| E1_K8 | +0.127 | [-0.059, +0.219] | 413 | 39 | 7.9012 |
| E1_K10 | +0.132 | [-0.051, +0.228] | 413 | 37 | 7.8764 |
| E1_K15 | +0.142 | [-0.035, +0.247] | 413 | 36 | 7.8672 |
| E2_a0.2_b0.02 | +0.111 | [-0.079, +0.223] | 413 | 38 | 7.9317 |
| E2_a0.2_b0.05 | +0.140 | [-0.063, +0.254] | 413 | 38 | 7.8690 |
| E2_a0.2_b0.1 | +0.081 | [-0.097, +0.168] | 413 | 38 | 8.0708 |
| E2_a0.2_b0.2 | +0.053 | [-0.064, +0.106] | 413 | 38 | 8.3512 |
| E2_a0.35_b0.02 | +0.090 | [-0.093, +0.201] | 413 | 38 | 7.9335 |
| E2_a0.35_b0.05 | +0.149 | [-0.067, +0.273] | 413 | 38 | 7.7189 |
| E2_a0.35_b0.1 | +0.127 | [-0.079, +0.234] | 413 | 38 | 7.8331 |
| E2_a0.35_b0.2 | +0.090 | [-0.063, +0.161] | 413 | 38 | 8.1168 |
| E2_a0.5_b0.02 | +0.075 | [-0.097, +0.181] | 413 | 38 | 7.9831 |
| E2_a0.5_b0.05 | +0.142 | [-0.078, +0.267] | 413 | 38 | 7.6861 |
| E2_a0.5_b0.1 | +0.150 | [-0.068, +0.268] | 413 | 38 | 7.7023 |
| E2_a0.5_b0.2 | +0.116 | [-0.065, +0.202] | 413 | 38 | 7.9465 |
| E2_a0.7_b0.02 | +0.063 | [-0.097, +0.161] | 413 | 38 | 8.0499 |
| E2_a0.7_b0.05 | +0.127 | [-0.084, +0.251] | 413 | 38 | 7.7107 |
| E2_a0.7_b0.1 | +0.158 | [-0.064, +0.282] | 413 | 38 | 7.6220 |
| E2_a0.7_b0.2 | +0.137 | [-0.067, +0.238] | 413 | 38 | 7.7929 |

### LSD_s0

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K5 (mean fold MSE 8.1620), best E2 E2_a0.7_b0.1 (7.9472); chosen E2_a0.7_b0.1.
- Windows where F is missing and the head closing output is used: 42 of 413 valid, 44 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) +0.001 [-0.044, +0.045]; DELTA-F permuted seeds 0 to 4: -0.001, +0.008, -0.001, -0.009, +0.000.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.111 [+0.086, +0.134]; cross-fitted five-fold +0.109 [+0.083, +0.132]; head closing output +0.203 [+0.161, +0.241].
- E3 blend (names nothing): R2 +0.266 [+0.012, +0.426] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.085 [-0.067, +0.178] n=413, chosen per fold {0: 'E2_a0.7_b0.05', 1: 'E2_a0.7_b0.05', 2: 'E2_a0.7_b0.02', 3: 'E2_a0.7_b0.05', 4: 'E2_a0.5_b0.05'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 7.74 m [+6.422, +9.242]; RMSE of the change between consecutive rows 2.30 m [+1.615, +3.030]; lag-1 autocorrelation +0.952 [+0.914, +0.975]. Rows with head presence at least 0.5 (n=2227): RMSE 6.71 m, change 2.24 m, lag-1 +0.940.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.062 | [-0.052, +0.121] | 413 | 51 | 8.2120 |
| E1_K5 | +0.100 | [-0.011, +0.154] | 413 | 47 | 8.1620 |
| E1_K8 | +0.097 | [-0.023, +0.155] | 413 | 42 | 8.1645 |
| E1_K10 | +0.104 | [-0.020, +0.167] | 413 | 42 | 8.1740 |
| E1_K15 | +0.115 | [-0.004, +0.186] | 413 | 40 | 8.2300 |
| E2_a0.2_b0.02 | +0.120 | [-0.012, +0.202] | 413 | 42 | 8.1551 |
| E2_a0.2_b0.05 | +0.118 | [-0.007, +0.189] | 413 | 42 | 8.1230 |
| E2_a0.2_b0.1 | +0.064 | [-0.053, +0.118] | 413 | 42 | 8.2522 |
| E2_a0.2_b0.2 | +0.033 | [-0.073, +0.073] | 413 | 42 | 8.4273 |
| E2_a0.35_b0.02 | +0.120 | [-0.002, +0.194] | 413 | 42 | 8.1473 |
| E2_a0.35_b0.05 | +0.141 | [+0.001, +0.225] | 413 | 42 | 8.0163 |
| E2_a0.35_b0.1 | +0.107 | [-0.022, +0.170] | 413 | 42 | 8.0970 |
| E2_a0.35_b0.2 | +0.064 | [-0.047, +0.114] | 413 | 42 | 8.2802 |
| E2_a0.5_b0.02 | +0.115 | [-0.000, +0.180] | 413 | 42 | 8.1746 |
| E2_a0.5_b0.05 | +0.148 | [+0.005, +0.234] | 413 | 42 | 7.9861 |
| E2_a0.5_b0.1 | +0.132 | [-0.003, +0.205] | 413 | 42 | 8.0067 |
| E2_a0.5_b0.2 | +0.090 | [-0.030, +0.146] | 413 | 42 | 8.1712 |
| E2_a0.7_b0.02 | +0.107 | [-0.001, +0.166] | 413 | 42 | 8.2152 |
| E2_a0.7_b0.05 | +0.148 | [+0.010, +0.231] | 413 | 42 | 7.9945 |
| E2_a0.7_b0.1 | +0.149 | [+0.008, +0.229] | 413 | 42 | 7.9472 |
| E2_a0.7_b0.2 | +0.114 | [-0.012, +0.178] | 413 | 42 | 8.0707 |

### LSD_s1

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K5 (mean fold MSE 8.1711), best E2 E2_a0.7_b0.05 (7.9212); chosen E2_a0.7_b0.05.
- Windows where F is missing and the head closing output is used: 53 of 413 valid, 56 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) -0.032 [-0.066, +0.004]; DELTA-F permuted seeds 0 to 4: -0.003, +0.005, +0.002, -0.002, -0.000.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.115 [+0.088, +0.142]; cross-fitted five-fold +0.113 [+0.086, +0.141]; head closing output +0.207 [+0.167, +0.247].
- E3 blend (names nothing): R2 +0.150 [-0.119, +0.323] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.083 [-0.098, +0.195] n=413, chosen per fold {0: 'E2_a0.7_b0.1', 1: 'E2_a0.7_b0.05', 2: 'E2_a0.7_b0.1', 3: 'E2_a0.7_b0.1', 4: 'E2_a0.7_b0.1'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 8.80 m [+6.505, +11.899]; RMSE of the change between consecutive rows 2.29 m [+1.599, +3.011]; lag-1 autocorrelation +0.962 [+0.912, +0.985]. Rows with head presence at least 0.5 (n=2146): RMSE 6.69 m, change 2.26 m, lag-1 +0.938.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.023 | [-0.129, +0.106] | 413 | 65 | 8.2161 |
| E1_K5 | +0.081 | [-0.069, +0.160] | 413 | 59 | 8.1711 |
| E1_K8 | +0.093 | [-0.049, +0.170] | 413 | 53 | 8.1907 |
| E1_K10 | +0.088 | [-0.060, +0.170] | 413 | 51 | 8.2092 |
| E1_K15 | +0.085 | [-0.066, +0.175] | 413 | 46 | 8.2750 |
| E2_a0.2_b0.02 | +0.075 | [-0.092, +0.181] | 413 | 53 | 8.0430 |
| E2_a0.2_b0.05 | +0.108 | [-0.043, +0.193] | 413 | 53 | 8.0913 |
| E2_a0.2_b0.1 | +0.074 | [-0.068, +0.147] | 413 | 53 | 8.2640 |
| E2_a0.2_b0.2 | +0.034 | [-0.102, +0.097] | 413 | 53 | 8.4477 |
| E2_a0.35_b0.02 | +0.072 | [-0.091, +0.180] | 413 | 53 | 8.0416 |
| E2_a0.35_b0.05 | +0.115 | [-0.043, +0.208] | 413 | 53 | 7.9654 |
| E2_a0.35_b0.1 | +0.104 | [-0.041, +0.182] | 413 | 53 | 8.1062 |
| E2_a0.35_b0.2 | +0.063 | [-0.076, +0.131] | 413 | 53 | 8.3118 |
| E2_a0.5_b0.02 | +0.066 | [-0.095, +0.173] | 413 | 53 | 8.0754 |
| E2_a0.5_b0.05 | +0.111 | [-0.049, +0.212] | 413 | 53 | 7.9232 |
| E2_a0.5_b0.1 | +0.118 | [-0.034, +0.202] | 413 | 53 | 8.0046 |
| E2_a0.5_b0.2 | +0.084 | [-0.057, +0.157] | 413 | 53 | 8.2036 |
| E2_a0.7_b0.02 | +0.059 | [-0.099, +0.167] | 413 | 53 | 8.1220 |
| E2_a0.7_b0.05 | +0.102 | [-0.062, +0.206] | 413 | 53 | 7.9212 |
| E2_a0.7_b0.1 | +0.123 | [-0.032, +0.212] | 413 | 53 | 7.9268 |
| E2_a0.7_b0.2 | +0.104 | [-0.040, +0.180] | 413 | 53 | 8.0945 |

## f8: the precision a rate needs (label gap plus white noise through E1; closing-speed R2 on the 413 windows, mean over 20 noise draws, range in brackets)

| noise sd (m) | K=5 | K=10 | K=15 |
|---|---|---|---|
| 0 | +0.367 [+0.367, +0.367] | +0.152 [+0.152, +0.152] | +0.325 [+0.325, +0.325] |
| 0.25 | +0.333 [+0.313, +0.360] | +0.140 [+0.133, +0.157] | +0.316 [+0.308, +0.327] |
| 0.5 | +0.232 [+0.190, +0.289] | +0.110 [+0.095, +0.149] | +0.292 [+0.278, +0.319] |
| 1 | -0.168 [-0.260, -0.046] | -0.006 [-0.051, +0.090] | +0.201 [+0.161, +0.276] |
| 2 | -1.764 [-1.999, -1.489] | -0.459 [-0.637, -0.195] | -0.157 [-0.299, +0.073] |
| 4 | -8.141 [-8.899, -7.390] | -2.247 [-2.957, -1.433] | -1.572 [-2.149, -0.794] |

## Limits

Two seeds per arm type; the bootstrap answers the question about episodes, not about training runs; the floor is from the two detached arms. The head saw the train probe clips during training, so its gap reads are cleaner there than on the evaluation episodes. 413 scored windows of 48 episodes; one corpus. A rate from a longer baseline lags a changing closing speed; the label is the closing speed at S0.
