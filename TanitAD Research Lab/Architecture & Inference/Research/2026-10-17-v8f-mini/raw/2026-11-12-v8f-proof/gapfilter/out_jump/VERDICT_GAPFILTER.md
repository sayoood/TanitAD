# VERDICT_GAPFILTER: card PREREG_V8F_PROOF_ADD28 (the closing speed of the lead from a causal filter over the head's own gap reads)

Tier T0 (a diagnostic on logged frames, never driving performance). STRATEGIC: PAUSED (PI item 25). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read. Evidence class MEASURED (ours, T0).

**POST-HOC same-lead variant (step above 3.0 m restarts or truncates the run); written after f2 of the card run failed; names nothing. Not the card's estimator; every number below is information for a later card.**

Counts: 1948 windows, 429 scored, 413 scored with a closing-speed label, 48 scored episodes; episode-cluster bootstrap B 2000, rng 0.

## Controls (first)

- f1_label_path: **PASS** R2 skill this path 0.7499068, banked 0.7499068 (n=413)
- f2_sign_units_alignment: **PASS** E1 K=5 on the label gap: R2 +1.000 (bar at least 0.8), n=413 of 413
- f6_records|LS_s0: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|LS_s0: **PASS** selection with real labels E2_a0.7_b0.1, with noise labels E2_a0.7_b0.1; identical True
- x1_alignment_m1_delta|LS_s0: **PASS** M1 +0.1547 vs N1 result +0.1547; DELTA of the head -0.0036 vs -0.0036
- f3_causal|LS_s0: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|LS_s0: **PASS** M1F with permuted sequences -0.168 (bar at most +0.02), seeds 0 to 4: -0.168, -0.125, -0.264, -0.277, -0.241; identity permutation reproduces F, max abs diff 0
- f6_records|LS_s1: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|LS_s1: **PASS** selection with real labels E2_a0.7_b0.1, with noise labels E2_a0.7_b0.1; identical True
- x1_alignment_m1_delta|LS_s1: **PASS** M1 +0.0136 vs N1 result +0.0136; DELTA of the head -0.0040 vs -0.0040
- f3_causal|LS_s1: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|LS_s1: **PASS** M1F with permuted sequences -0.173 (bar at most +0.02), seeds 0 to 4: -0.173, -0.095, -0.274, -0.166, -0.150; identity permutation reproduces F, max abs diff 0
- f6_records|LSH_s0: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|LSH_s0: **PASS** selection with real labels E2_a0.7_b0.1, with noise labels E2_a0.7_b0.1; identical True
- x1_alignment_m1_delta|LSH_s0: **PASS** M1 +0.0915 vs N1 result +0.0915; DELTA of the head +0.0195 vs +0.0195
- f3_causal|LSH_s0: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|LSH_s0: **PASS** M1F with permuted sequences -0.223 (bar at most +0.02), seeds 0 to 4: -0.223, -0.155, -0.277, -0.264, -0.253; identity permutation reproduces F, max abs diff 0
- f6_records|LSH_s1: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|LSH_s1: **PASS** selection with real labels E2_a0.7_b0.1, with noise labels E2_a0.7_b0.1; identical True
- x1_alignment_m1_delta|LSH_s1: **PASS** M1 +0.0264 vs N1 result +0.0264; DELTA of the head +0.0010 vs +0.0010
- f3_causal|LSH_s1: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|LSH_s1: **PASS** M1F with permuted sequences -0.145 (bar at most +0.02), seeds 0 to 4: -0.145, -0.124, -0.277, -0.214, -0.197; identity permutation reproduces F, max abs diff 0
- f6_records|LSD_s0: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|LSD_s0: **PASS** selection with real labels E2_a0.7_b0.1, with noise labels E2_a0.7_b0.1; identical True
- x1_alignment_m1_delta|LSD_s0: **PASS** M1 +0.1782 vs N1 result +0.1782; DELTA of the head +0.0015 vs +0.0015
- f3_causal|LSD_s0: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|LSD_s0: **PASS** M1F with permuted sequences -0.157 (bar at most +0.02), seeds 0 to 4: -0.157, -0.104, -0.218, -0.134, -0.188; identity permutation reproduces F, max abs diff 0
- f6_records|LSD_s1: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|LSD_s1: **PASS** selection with real labels E2_a0.7_b0.1, with noise labels E2_a0.7_b0.1; identical True
- x1_alignment_m1_delta|LSD_s1: **PASS** M1 +0.0890 vs N1 result +0.0890; DELTA of the head -0.0323 vs -0.0323
- f3_causal|LSD_s1: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|LSD_s1: **PASS** M1F with permuted sequences -0.121 (bar at most +0.02), seeds 0 to 4: -0.121, -0.112, -0.248, -0.194, -0.187; identity permutation reproduces F, max abs diff 0

Controls failed: none

## Names (card section 5)

No outcome is named. POST-HOC same-lead variant (step above 3.0 m restarts or truncates the run); written after f2 of the card run failed; names nothing

Floor: card value 0.089; measured |M1(LSD_s0) - M1(LSD_s1)| = 0.0893.

## Per arm (413 scored windows with a closing-speed label for M1, M1F, paired; 429 scored windows for DELTA)

| arm | selected F | a, b | M1 head closing R2 | M1F R2 | M1F minus M1 | DELTA-F | DELTA-F permuted (seed 0) | F estimate missing (of 413) |
|---|---|---|---|---|---|---|---|---|
| LS_s0 | E2_a0.7_b0.1 | +0.552, +0.003 | +0.155 [-0.043, +0.283] n=413 | +0.106 [-0.095, +0.242] n=413 | -0.048 [-0.180, +0.082] n=413 | -0.001 [-0.047, +0.043] n=429 | -0.002 | 54 |
| LS_s1 | E2_a0.7_b0.1 | +0.464, +0.017 | +0.014 [-0.278, +0.192] n=413 | +0.148 [+0.005, +0.236] n=413 | +0.134 [-0.048, +0.384] n=413 | -0.006 [-0.015, +0.003] n=429 | -0.004 | 67 |
| LSH_s0 | E2_a0.7_b0.1 | +0.576, -0.013 | +0.092 [-0.138, +0.231] n=413 | +0.238 [+0.056, +0.354] n=413 | +0.147 [-0.036, +0.341] n=413 | -0.011 [-0.023, +0.002] n=429 | -0.001 | 61 |
| LSH_s1 | E2_a0.7_b0.1 | +0.540, -0.023 | +0.026 [-0.192, +0.163] n=413 | +0.165 [-0.017, +0.280] n=413 | +0.139 [-0.049, +0.330] n=413 | -0.003 [-0.009, +0.002] n=429 | -0.001 | 49 |
| LSD_s0 | E2_a0.7_b0.1 | +0.468, -0.035 | +0.178 [-0.059, +0.335] n=413 | +0.109 [-0.033, +0.186] n=413 | -0.070 [-0.204, +0.089] n=413 | +0.002 [-0.023, +0.025] n=429 | +0.002 | 62 |
| LSD_s1 | E2_a0.7_b0.1 | +0.471, -0.038 | +0.089 [-0.127, +0.234] n=413 | +0.088 [-0.082, +0.191] n=413 | -0.001 [-0.153, +0.149] n=413 | -0.003 [-0.080, +0.055] n=429 | +0.000 | 71 |

### LS_s0

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K10 (mean fold MSE 7.7021), best E2 E2_a0.7_b0.1 (7.4741); chosen E2_a0.7_b0.1.
- Windows where F is missing and the head closing output is used: 54 of 413 valid, 57 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) -0.004 [-0.026, +0.016]; DELTA-F permuted seeds 0 to 4: -0.002, -0.001, +0.002, -0.012, -0.005.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.163 [+0.133, +0.191]; cross-fitted five-fold +0.161 [+0.131, +0.189]; head closing output +0.209 [+0.167, +0.249].
- E3 blend (names nothing): R2 +0.222 [-0.008, +0.375] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.074 [-0.113, +0.202] n=413, chosen per fold {0: 'E2_a0.7_b0.1', 1: 'E2_a0.7_b0.1', 2: 'E2_a0.7_b0.1', 3: 'E2_a0.7_b0.1', 4: 'E2_a0.7_b0.1'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 7.02 m [+5.696, +8.657]; RMSE of the change between consecutive rows 2.23 m [+1.539, +2.944]; lag-1 autocorrelation +0.946 [+0.899, +0.975]. Rows with head presence at least 0.5 (n=2203): RMSE 5.97 m, change 2.14 m, lag-1 +0.933.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.035 | [-0.105, +0.126] | 413 | 55 | 7.8920 |
| E1_K5 | +0.086 | [-0.079, +0.192] | 413 | 54 | 7.8023 |
| E1_K8 | +0.088 | [-0.091, +0.204] | 413 | 54 | 7.7284 |
| E1_K10 | +0.079 | [-0.108, +0.203] | 413 | 54 | 7.7021 |
| E1_K15 | +0.063 | [-0.131, +0.196] | 413 | 54 | 7.7224 |
| E2_a0.2_b0.02 | +0.057 | [-0.131, +0.186] | 413 | 54 | 7.6837 |
| E2_a0.2_b0.05 | +0.083 | [-0.104, +0.210] | 413 | 54 | 7.6436 |
| E2_a0.2_b0.1 | +0.065 | [-0.093, +0.165] | 413 | 54 | 7.8241 |
| E2_a0.2_b0.2 | +0.048 | [-0.090, +0.135] | 413 | 54 | 8.0780 |
| E2_a0.35_b0.02 | +0.063 | [-0.110, +0.183] | 413 | 54 | 7.6835 |
| E2_a0.35_b0.05 | +0.090 | [-0.108, +0.228] | 413 | 54 | 7.5327 |
| E2_a0.35_b0.1 | +0.093 | [-0.084, +0.209] | 413 | 54 | 7.6424 |
| E2_a0.35_b0.2 | +0.073 | [-0.085, +0.169] | 413 | 54 | 7.8989 |
| E2_a0.5_b0.02 | +0.063 | [-0.098, +0.176] | 413 | 54 | 7.7140 |
| E2_a0.5_b0.05 | +0.092 | [-0.105, +0.228] | 413 | 54 | 7.5061 |
| E2_a0.5_b0.1 | +0.103 | [-0.090, +0.230] | 413 | 54 | 7.5403 |
| E2_a0.5_b0.2 | +0.089 | [-0.077, +0.194] | 413 | 54 | 7.7654 |
| E2_a0.7_b0.02 | +0.061 | [-0.093, +0.168] | 413 | 54 | 7.7561 |
| E2_a0.7_b0.05 | +0.091 | [-0.098, +0.222] | 413 | 54 | 7.5190 |
| E2_a0.7_b0.1 | +0.106 | [-0.095, +0.242] | 413 | 54 | 7.4741 |
| E2_a0.7_b0.2 | +0.100 | [-0.080, +0.217] | 413 | 54 | 7.6407 |

### LS_s1

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K10 (mean fold MSE 7.8985), best E2 E2_a0.7_b0.1 (7.7812); chosen E2_a0.7_b0.1.
- Windows where F is missing and the head closing output is used: 67 of 413 valid, 71 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) -0.004 [-0.059, +0.051]; DELTA-F permuted seeds 0 to 4: -0.004, +0.005, -0.001, +0.011, -0.004.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.129 [+0.096, +0.159]; cross-fitted five-fold +0.127 [+0.094, +0.157]; head closing output +0.182 [+0.129, +0.227].
- E3 blend (names nothing): R2 +0.152 [-0.052, +0.280] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.103 [-0.048, +0.202] n=413, chosen per fold {0: 'E2_a0.7_b0.05', 1: 'E1_K15', 2: 'E2_a0.7_b0.05', 3: 'E2_a0.7_b0.05', 4: 'E2_a0.7_b0.1'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 8.26 m [+6.904, +9.887]; RMSE of the change between consecutive rows 2.32 m [+1.655, +3.029]; lag-1 autocorrelation +0.960 [+0.924, +0.980]. Rows with head presence at least 0.5 (n=2168): RMSE 7.14 m, change 2.30 m, lag-1 +0.946.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.069 | [-0.057, +0.145] | 413 | 75 | 8.0572 |
| E1_K5 | +0.094 | [-0.037, +0.175] | 413 | 71 | 7.9521 |
| E1_K8 | +0.124 | [-0.012, +0.209] | 413 | 67 | 7.9043 |
| E1_K10 | +0.129 | [-0.013, +0.215] | 413 | 66 | 7.8985 |
| E1_K15 | +0.131 | [-0.011, +0.213] | 413 | 64 | 7.9468 |
| E2_a0.2_b0.02 | +0.111 | [-0.027, +0.194] | 413 | 67 | 7.9998 |
| E2_a0.2_b0.05 | +0.124 | [-0.015, +0.209] | 413 | 67 | 7.9139 |
| E2_a0.2_b0.1 | +0.090 | [-0.037, +0.175] | 413 | 67 | 8.0039 |
| E2_a0.2_b0.2 | +0.066 | [-0.055, +0.141] | 413 | 67 | 8.1696 |
| E2_a0.35_b0.02 | +0.111 | [-0.025, +0.194] | 413 | 67 | 7.9884 |
| E2_a0.35_b0.05 | +0.137 | [-0.008, +0.226] | 413 | 67 | 7.8472 |
| E2_a0.35_b0.1 | +0.122 | [-0.017, +0.209] | 413 | 67 | 7.8796 |
| E2_a0.35_b0.2 | +0.092 | [-0.031, +0.172] | 413 | 67 | 8.0349 |
| E2_a0.5_b0.02 | +0.109 | [-0.024, +0.192] | 413 | 67 | 8.0000 |
| E2_a0.5_b0.05 | +0.140 | [-0.004, +0.229] | 413 | 67 | 7.8344 |
| E2_a0.5_b0.1 | +0.138 | [-0.002, +0.225] | 413 | 67 | 7.8162 |
| E2_a0.5_b0.2 | +0.112 | [-0.016, +0.195] | 413 | 67 | 7.9415 |
| E2_a0.7_b0.02 | +0.105 | [-0.025, +0.187] | 413 | 67 | 8.0208 |
| E2_a0.7_b0.05 | +0.138 | [-0.002, +0.226] | 413 | 67 | 7.8462 |
| E2_a0.7_b0.1 | +0.148 | [+0.005, +0.236] | 413 | 67 | 7.7812 |
| E2_a0.7_b0.2 | +0.130 | [-0.000, +0.215] | 413 | 67 | 7.8608 |

### LSH_s0

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K15 (mean fold MSE 7.5786), best E2 E2_a0.7_b0.1 (7.3678); chosen E2_a0.7_b0.1.
- Windows where F is missing and the head closing output is used: 61 of 413 valid, 63 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) +0.019 [-0.027, +0.067]; DELTA-F permuted seeds 0 to 4: -0.001, -0.001, +0.004, -0.008, -0.006.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.175 [+0.146, +0.200]; cross-fitted five-fold +0.174 [+0.145, +0.198]; head closing output +0.201 [+0.159, +0.239].
- E3 blend (names nothing): R2 +0.266 [+0.046, +0.396] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.179 [-0.016, +0.313] n=413, chosen per fold {0: 'E1_K15', 1: 'E2_a0.2_b0.02', 2: 'E2_a0.5_b0.05', 3: 'E2_a0.2_b0.02', 4: 'E2_a0.7_b0.1'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 8.05 m [+6.627, +9.720]; RMSE of the change between consecutive rows 2.34 m [+1.632, +3.080]; lag-1 autocorrelation +0.947 [+0.901, +0.975]. Rows with head presence at least 0.5 (n=2277): RMSE 7.00 m, change 2.35 m, lag-1 +0.930.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.112 | [-0.016, +0.181] | 413 | 68 | 7.8534 |
| E1_K5 | +0.161 | [+0.018, +0.246] | 413 | 64 | 7.6901 |
| E1_K8 | +0.189 | [+0.044, +0.273] | 413 | 61 | 7.6081 |
| E1_K10 | +0.199 | [+0.051, +0.285] | 413 | 61 | 7.5805 |
| E1_K15 | +0.203 | [+0.055, +0.292] | 413 | 61 | 7.5786 |
| E2_a0.2_b0.02 | +0.176 | [-0.019, +0.300] | 413 | 61 | 7.5994 |
| E2_a0.2_b0.05 | +0.218 | [+0.056, +0.318] | 413 | 61 | 7.5459 |
| E2_a0.2_b0.1 | +0.161 | [+0.019, +0.239] | 413 | 61 | 7.7298 |
| E2_a0.2_b0.2 | +0.096 | [-0.033, +0.160] | 413 | 61 | 7.9923 |
| E2_a0.35_b0.02 | +0.159 | [-0.036, +0.283] | 413 | 61 | 7.6081 |
| E2_a0.35_b0.05 | +0.227 | [+0.048, +0.342] | 413 | 61 | 7.4347 |
| E2_a0.35_b0.1 | +0.206 | [+0.048, +0.302] | 413 | 61 | 7.5438 |
| E2_a0.35_b0.2 | +0.144 | [+0.006, +0.222] | 413 | 61 | 7.7975 |
| E2_a0.5_b0.02 | +0.146 | [-0.040, +0.269] | 413 | 61 | 7.6389 |
| E2_a0.5_b0.05 | +0.220 | [+0.027, +0.341] | 413 | 61 | 7.4105 |
| E2_a0.5_b0.1 | +0.229 | [+0.058, +0.335] | 413 | 61 | 7.4365 |
| E2_a0.5_b0.2 | +0.180 | [+0.031, +0.266] | 413 | 61 | 7.6558 |
| E2_a0.7_b0.02 | +0.134 | [-0.046, +0.254] | 413 | 61 | 7.6784 |
| E2_a0.7_b0.05 | +0.206 | [+0.009, +0.333] | 413 | 61 | 7.4264 |
| E2_a0.7_b0.1 | +0.238 | [+0.056, +0.354] | 413 | 61 | 7.3678 |
| E2_a0.7_b0.2 | +0.211 | [+0.053, +0.308] | 413 | 61 | 7.5278 |

### LSH_s1

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K15 (mean fold MSE 7.7718), best E2 E2_a0.7_b0.1 (7.5947); chosen E2_a0.7_b0.1.
- Windows where F is missing and the head closing output is used: 49 of 413 valid, 51 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) +0.001 [-0.085, +0.060]; DELTA-F permuted seeds 0 to 4: -0.001, -0.009, +0.004, -0.005, +0.002.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.152 [+0.120, +0.179]; cross-fitted five-fold +0.151 [+0.119, +0.179]; head closing output +0.186 [+0.140, +0.227].
- E3 blend (names nothing): R2 +0.181 [-0.046, +0.327] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.126 [-0.024, +0.227] n=413, chosen per fold {0: 'E1_K15', 1: 'E2_a0.7_b0.2', 2: 'E2_a0.7_b0.1', 3: 'E1_K10', 4: 'E1_K15'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 7.08 m [+5.806, +8.581]; RMSE of the change between consecutive rows 2.31 m [+1.600, +3.026]; lag-1 autocorrelation +0.945 [+0.896, +0.974]. Rows with head presence at least 0.5 (n=2231): RMSE 6.45 m, change 2.34 m, lag-1 +0.933.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.096 | [-0.024, +0.157] | 413 | 54 | 8.0507 |
| E1_K5 | +0.141 | [+0.017, +0.209] | 413 | 52 | 7.8903 |
| E1_K8 | +0.159 | [+0.014, +0.243] | 413 | 50 | 7.7960 |
| E1_K10 | +0.161 | [+0.010, +0.252] | 413 | 48 | 7.7728 |
| E1_K15 | +0.166 | [+0.002, +0.266] | 413 | 47 | 7.7718 |
| E2_a0.2_b0.02 | +0.104 | [-0.077, +0.219] | 413 | 49 | 7.8500 |
| E2_a0.2_b0.05 | +0.149 | [-0.023, +0.260] | 413 | 49 | 7.7595 |
| E2_a0.2_b0.1 | +0.125 | [-0.008, +0.202] | 413 | 49 | 7.8966 |
| E2_a0.2_b0.2 | +0.086 | [-0.020, +0.142] | 413 | 49 | 8.1595 |
| E2_a0.35_b0.02 | +0.100 | [-0.075, +0.211] | 413 | 49 | 7.8438 |
| E2_a0.35_b0.05 | +0.149 | [-0.040, +0.268] | 413 | 49 | 7.6644 |
| E2_a0.35_b0.1 | +0.153 | [-0.002, +0.249] | 413 | 49 | 7.7327 |
| E2_a0.35_b0.2 | +0.123 | [+0.003, +0.189] | 413 | 49 | 7.9762 |
| E2_a0.5_b0.02 | +0.095 | [-0.067, +0.198] | 413 | 49 | 7.8723 |
| E2_a0.5_b0.05 | +0.145 | [-0.046, +0.265] | 413 | 49 | 7.6467 |
| E2_a0.5_b0.1 | +0.164 | [-0.008, +0.271] | 413 | 49 | 7.6455 |
| E2_a0.5_b0.2 | +0.145 | [+0.015, +0.220] | 413 | 49 | 7.8460 |
| E2_a0.7_b0.02 | +0.090 | [-0.061, +0.187] | 413 | 49 | 7.9137 |
| E2_a0.7_b0.05 | +0.136 | [-0.051, +0.256] | 413 | 49 | 7.6664 |
| E2_a0.7_b0.1 | +0.165 | [-0.017, +0.280] | 413 | 49 | 7.5947 |
| E2_a0.7_b0.2 | +0.160 | [+0.011, +0.249] | 413 | 49 | 7.7314 |

### LSD_s0

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K5 (mean fold MSE 8.0555), best E2 E2_a0.7_b0.1 (7.8780); chosen E2_a0.7_b0.1.
- Windows where F is missing and the head closing output is used: 62 of 413 valid, 65 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) +0.001 [-0.044, +0.045]; DELTA-F permuted seeds 0 to 4: +0.002, +0.004, +0.009, -0.011, -0.001.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.118 [+0.095, +0.140]; cross-fitted five-fold +0.116 [+0.092, +0.137]; head closing output +0.203 [+0.161, +0.241].
- E3 blend (names nothing): R2 +0.230 [-0.019, +0.392] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.071 [-0.077, +0.172] n=413, chosen per fold {0: 'E2_a0.7_b0.05', 1: 'E2_a0.7_b0.05', 2: 'E2_a0.7_b0.02', 3: 'E2_a0.7_b0.02', 4: 'E2_a0.7_b0.05'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 7.74 m [+6.422, +9.242]; RMSE of the change between consecutive rows 2.30 m [+1.615, +3.030]; lag-1 autocorrelation +0.952 [+0.914, +0.975]. Rows with head presence at least 0.5 (n=2227): RMSE 6.71 m, change 2.24 m, lag-1 +0.940.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.060 | [-0.063, +0.126] | 413 | 69 | 8.1168 |
| E1_K5 | +0.078 | [-0.046, +0.140] | 413 | 66 | 8.0555 |
| E1_K8 | +0.060 | [-0.067, +0.120] | 413 | 62 | 8.0586 |
| E1_K10 | +0.062 | [-0.064, +0.123] | 413 | 62 | 8.0735 |
| E1_K15 | +0.063 | [-0.062, +0.127] | 413 | 62 | 8.0971 |
| E2_a0.2_b0.02 | +0.088 | [-0.034, +0.163] | 413 | 62 | 8.0285 |
| E2_a0.2_b0.05 | +0.087 | [-0.051, +0.156] | 413 | 62 | 7.9876 |
| E2_a0.2_b0.1 | +0.054 | [-0.077, +0.119] | 413 | 62 | 8.1057 |
| E2_a0.2_b0.2 | +0.024 | [-0.098, +0.079] | 413 | 62 | 8.2544 |
| E2_a0.35_b0.02 | +0.095 | [-0.027, +0.170] | 413 | 62 | 8.0217 |
| E2_a0.35_b0.05 | +0.101 | [-0.040, +0.176] | 413 | 62 | 7.9193 |
| E2_a0.35_b0.1 | +0.079 | [-0.058, +0.148] | 413 | 62 | 7.9901 |
| E2_a0.35_b0.2 | +0.049 | [-0.084, +0.112] | 413 | 62 | 8.1420 |
| E2_a0.5_b0.02 | +0.097 | [-0.022, +0.174] | 413 | 62 | 8.0360 |
| E2_a0.5_b0.05 | +0.109 | [-0.029, +0.187] | 413 | 62 | 7.9001 |
| E2_a0.5_b0.1 | +0.095 | [-0.045, +0.169] | 413 | 62 | 7.9228 |
| E2_a0.5_b0.2 | +0.068 | [-0.069, +0.137] | 413 | 62 | 8.0590 |
| E2_a0.7_b0.02 | +0.097 | [-0.020, +0.174] | 413 | 62 | 8.0582 |
| E2_a0.7_b0.05 | +0.113 | [-0.020, +0.193] | 413 | 62 | 7.9050 |
| E2_a0.7_b0.1 | +0.109 | [-0.033, +0.186] | 413 | 62 | 7.8780 |
| E2_a0.7_b0.2 | +0.086 | [-0.053, +0.160] | 413 | 62 | 7.9826 |

### LSD_s1

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K5 (mean fold MSE 8.0386), best E2 E2_a0.7_b0.1 (7.8320); chosen E2_a0.7_b0.1.
- Windows where F is missing and the head closing output is used: 71 of 413 valid, 75 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) -0.032 [-0.066, +0.004]; DELTA-F permuted seeds 0 to 4: +0.000, +0.006, -0.003, -0.007, -0.001.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.124 [+0.096, +0.150]; cross-fitted five-fold +0.122 [+0.094, +0.149]; head closing output +0.207 [+0.167, +0.247].
- E3 blend (names nothing): R2 +0.134 [-0.135, +0.302] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.040 [-0.130, +0.151] n=413, chosen per fold {0: 'E2_a0.7_b0.05', 1: 'E1_K15', 2: 'E2_a0.7_b0.05', 3: 'E2_a0.7_b0.1', 4: 'E2_a0.7_b0.1'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 8.80 m [+6.505, +11.899]; RMSE of the change between consecutive rows 2.29 m [+1.599, +3.011]; lag-1 autocorrelation +0.962 [+0.912, +0.985]. Rows with head presence at least 0.5 (n=2146): RMSE 6.69 m, change 2.26 m, lag-1 +0.938.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.019 | [-0.136, +0.105] | 413 | 78 | 8.0973 |
| E1_K5 | +0.064 | [-0.088, +0.147] | 413 | 75 | 8.0386 |
| E1_K8 | +0.076 | [-0.073, +0.161] | 413 | 71 | 8.0410 |
| E1_K10 | +0.069 | [-0.085, +0.158] | 413 | 69 | 8.0564 |
| E1_K15 | +0.073 | [-0.084, +0.165] | 413 | 67 | 8.0880 |
| E2_a0.2_b0.02 | +0.062 | [-0.099, +0.166] | 413 | 71 | 7.9691 |
| E2_a0.2_b0.05 | +0.072 | [-0.093, +0.170] | 413 | 71 | 7.9764 |
| E2_a0.2_b0.1 | +0.060 | [-0.094, +0.142] | 413 | 71 | 8.1015 |
| E2_a0.2_b0.2 | +0.042 | [-0.102, +0.113] | 413 | 71 | 8.2509 |
| E2_a0.35_b0.02 | +0.061 | [-0.098, +0.167] | 413 | 71 | 7.9565 |
| E2_a0.35_b0.05 | +0.080 | [-0.090, +0.185] | 413 | 71 | 7.8793 |
| E2_a0.35_b0.1 | +0.076 | [-0.082, +0.168] | 413 | 71 | 7.9697 |
| E2_a0.35_b0.2 | +0.057 | [-0.093, +0.135] | 413 | 71 | 8.1307 |
| E2_a0.5_b0.02 | +0.058 | [-0.098, +0.163] | 413 | 71 | 7.9695 |
| E2_a0.5_b0.05 | +0.081 | [-0.089, +0.189] | 413 | 71 | 7.8458 |
| E2_a0.5_b0.1 | +0.084 | [-0.077, +0.183] | 413 | 71 | 7.8897 |
| E2_a0.5_b0.2 | +0.068 | [-0.085, +0.153] | 413 | 71 | 8.0405 |
| E2_a0.7_b0.02 | +0.055 | [-0.098, +0.158] | 413 | 71 | 7.9926 |
| E2_a0.7_b0.05 | +0.079 | [-0.090, +0.186] | 413 | 71 | 7.8416 |
| E2_a0.7_b0.1 | +0.088 | [-0.082, +0.191] | 413 | 71 | 7.8320 |
| E2_a0.7_b0.2 | +0.078 | [-0.074, +0.170] | 413 | 71 | 7.9543 |

## f8: the precision a rate needs (label gap plus white noise through E1; closing-speed R2 on the 413 windows, mean over 20 noise draws, range in brackets)

| noise sd (m) | K=5 | K=10 | K=15 |
|---|---|---|---|
| 0 | +1.000 [+1.000, +1.000] | +0.941 [+0.941, +0.941] | +0.886 [+0.886, +0.886] |
| 0.25 | +0.964 [+0.957, +0.968] | +0.931 [+0.924, +0.938] | +0.878 [+0.873, +0.886] |
| 0.5 | +0.856 [+0.844, +0.872] | +0.896 [+0.880, +0.906] | +0.849 [+0.838, +0.862] |
| 1 | +0.179 [+0.020, +0.297] | +0.432 [+0.290, +0.580] | +0.415 [+0.279, +0.560] |
| 2 | -2.612 [-3.379, -1.945] | -2.389 [-3.075, -1.780] | -2.385 [-3.060, -1.785] |
| 4 | -5.856 [-11.105, -3.275] | -5.813 [-11.206, -3.290] | -5.814 [-11.206, -3.290] |

## Limits

Two seeds per arm type; the bootstrap answers the question about episodes, not about training runs; the floor is from the two detached arms. The head saw the train probe clips during training, so its gap reads are cleaner there than on the evaluation episodes. 413 scored windows of 48 episodes; one corpus. A rate from a longer baseline lags a changing closing speed; the label is the closing speed at S0.
