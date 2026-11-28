# VERDICT_GAPFILTER: card PREREG_V8F_PROOF_ADD28 (the closing speed of the lead from a causal filter over the head's own gap reads)

Tier T0 (a diagnostic on logged frames, never driving performance). STRATEGIC: PAUSED (PI item 25). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read. Evidence class MEASURED (ours, T0).

Counts: 1948 windows, 429 scored, 413 scored with a closing-speed label, 48 scored episodes; episode-cluster bootstrap B 2000, rng 0.

## Controls (first)

- f1_label_path: **PASS** R2 skill this path 0.7499068, banked 0.7499068 (n=413)
- f2_sign_units_alignment: **FAIL** E1 K=5 on the label gap: R2 +0.367 (bar at least 0.8), n=413 of 413
- f6_records|TK16_s0: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|TK16_s0: **PASS** selection with real labels E2_a0.7_b0.1, with noise labels E2_a0.7_b0.1; identical True
- x1_alignment_m1_delta|TK16_s0: **PASS** M1 -0.0015 vs N1 result -0.0015; DELTA of the head +0.0195 vs +0.0195
- f3_causal|TK16_s0: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|TK16_s0: **PASS** M1F with permuted sequences -0.229 (bar at most +0.02), seeds 0 to 4: -0.229, -0.226, -0.242, -0.266, -0.252; identity permutation reproduces F, max abs diff 0
- f6_records|TK16_s1: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|TK16_s1: **PASS** selection with real labels E2_a0.7_b0.1, with noise labels E2_a0.7_b0.1; identical True
- x1_alignment_m1_delta|TK16_s1: **PASS** M1 +0.2033 vs N1 result +0.2033; DELTA of the head +0.1440 vs +0.1440
- f3_causal|TK16_s1: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|TK16_s1: **PASS** M1F with permuted sequences -0.262 (bar at most +0.02), seeds 0 to 4: -0.262, -0.198, -0.270, -0.333, -0.279; identity permutation reproduces F, max abs diff 0
- f6_records|TK64_s0: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|TK64_s0: **PASS** selection with real labels E2_a0.7_b0.1, with noise labels E2_a0.7_b0.1; identical True
- x1_alignment_m1_delta|TK64_s0: **PASS** M1 +0.0424 vs N1 result +0.0424; DELTA of the head +0.0248 vs +0.0248
- f3_causal|TK64_s0: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|TK64_s0: **PASS** M1F with permuted sequences -0.191 (bar at most +0.02), seeds 0 to 4: -0.191, -0.204, -0.300, -0.239, -0.242; identity permutation reproduces F, max abs diff 0
- f6_records|TK64_s1: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|TK64_s1: **PASS** selection with real labels E2_a0.7_b0.1, with noise labels E2_a0.7_b0.1; identical True
- x1_alignment_m1_delta|TK64_s1: **PASS** M1 -0.0012 vs N1 result -0.0012; DELTA of the head +0.0274 vs +0.0274
- f3_causal|TK64_s1: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|TK64_s1: **PASS** M1F with permuted sequences -0.202 (bar at most +0.02), seeds 0 to 4: -0.202, -0.204, -0.271, -0.240, -0.223; identity permutation reproduces F, max abs diff 0

Controls failed: f2_sign_units_alignment

## Names (card section 5)

No outcome is named. controls failed: f2_sign_units_alignment

Floor: card value 0.089; measured |M1(LSD_s0) - M1(LSD_s1)| = n/a.

## Per arm (413 scored windows with a closing-speed label for M1, M1F, paired; 429 scored windows for DELTA)

| arm | selected F | a, b | M1 head closing R2 | M1F R2 | M1F minus M1 | DELTA-F | DELTA-F permuted (seed 0) | F estimate missing (of 413) |
|---|---|---|---|---|---|---|---|---|
| TK16_s0 | E2_a0.7_b0.1 | +0.399, -0.046 | -0.002 [-0.398, +0.212] n=413 | +0.199 [-0.002, +0.313] n=413 | +0.201 [-0.016, +0.515] n=413 | -0.017 [-0.038, +0.000] n=429 | +0.001 | 48 |
| TK16_s1 | E2_a0.7_b0.1 | +0.390, -0.033 | +0.203 [-0.023, +0.328] n=413 | +0.217 [+0.013, +0.328] n=413 | +0.014 [-0.144, +0.163] n=413 | -0.033 [-0.081, +0.004] n=429 | -0.001 | 39 |
| TK64_s0 | E2_a0.7_b0.1 | +0.430, -0.023 | +0.042 [-0.311, +0.249] n=413 | +0.155 [-0.056, +0.286] n=413 | +0.113 [-0.063, +0.341] n=413 | -0.012 [-0.029, +0.004] n=429 | -0.003 | 56 |
| TK64_s1 | E2_a0.7_b0.1 | +0.396, -0.035 | -0.001 [-0.435, +0.257] n=413 | +0.149 [-0.057, +0.270] n=413 | +0.150 [-0.094, +0.493] n=413 | -0.007 [-0.025, +0.009] n=429 | +0.000 | 57 |

### TK16_s0

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K10 (mean fold MSE 7.9041), best E2 E2_a0.7_b0.1 (7.5904); chosen E2_a0.7_b0.1.
- Windows where F is missing and the head closing output is used: 48 of 413 valid, 48 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) +0.020 [-0.032, +0.066]; DELTA-F permuted seeds 0 to 4: +0.001, +0.001, +0.003, -0.005, -0.002.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.151 [+0.125, +0.179]; cross-fitted five-fold +0.149 [+0.123, +0.177]; head closing output +0.235 [+0.191, +0.279].
- E3 blend (names nothing): R2 +0.174 [-0.107, +0.338] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.161 [-0.051, +0.292] n=413, chosen per fold {0: 'E2_a0.7_b0.1', 1: 'E2_a0.7_b0.1', 2: 'E2_a0.7_b0.1', 3: 'E2_a0.7_b0.1', 4: 'E2_a0.7_b0.1'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 8.58 m [+6.718, +10.644]; RMSE of the change between consecutive rows 2.35 m [+1.669, +3.086]; lag-1 autocorrelation +0.955 [+0.914, +0.979]. Rows with head presence at least 0.5 (n=2173): RMSE 7.15 m, change 2.22 m, lag-1 +0.942.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.100 | [-0.080, +0.188] | 413 | 60 | 8.0990 |
| E1_K5 | +0.177 | [+0.050, +0.253] | 413 | 54 | 7.9529 |
| E1_K8 | +0.156 | [-0.010, +0.255] | 413 | 48 | 7.9133 |
| E1_K10 | +0.180 | [+0.034, +0.267] | 413 | 45 | 7.9041 |
| E1_K15 | +0.143 | [-0.023, +0.246] | 413 | 42 | 7.9344 |
| E2_a0.2_b0.02 | +0.151 | [-0.018, +0.255] | 413 | 48 | 7.8571 |
| E2_a0.2_b0.05 | +0.180 | [+0.008, +0.274] | 413 | 48 | 7.8334 |
| E2_a0.2_b0.1 | +0.146 | [-0.002, +0.222] | 413 | 48 | 8.0778 |
| E2_a0.2_b0.2 | +0.085 | [-0.036, +0.151] | 413 | 48 | 8.3520 |
| E2_a0.35_b0.02 | +0.134 | [-0.033, +0.236] | 413 | 48 | 7.8583 |
| E2_a0.35_b0.05 | +0.193 | [-0.004, +0.304] | 413 | 48 | 7.6709 |
| E2_a0.35_b0.1 | +0.179 | [-0.000, +0.276] | 413 | 48 | 7.8245 |
| E2_a0.35_b0.2 | +0.122 | [-0.031, +0.203] | 413 | 48 | 8.1203 |
| E2_a0.5_b0.02 | +0.120 | [-0.037, +0.217] | 413 | 48 | 7.9083 |
| E2_a0.5_b0.05 | +0.186 | [-0.015, +0.301] | 413 | 48 | 7.6315 |
| E2_a0.5_b0.1 | +0.194 | [-0.001, +0.302] | 413 | 48 | 7.6814 |
| E2_a0.5_b0.2 | +0.149 | [-0.020, +0.244] | 413 | 48 | 7.9501 |
| E2_a0.7_b0.02 | +0.106 | [-0.040, +0.200] | 413 | 48 | 7.9762 |
| E2_a0.7_b0.05 | +0.170 | [-0.023, +0.286] | 413 | 48 | 7.6515 |
| E2_a0.7_b0.1 | +0.199 | [-0.002, +0.313] | 413 | 48 | 7.5904 |
| E2_a0.7_b0.2 | +0.175 | [-0.001, +0.278] | 413 | 48 | 7.7932 |

### TK16_s1

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K15 (mean fold MSE 7.9143), best E2 E2_a0.7_b0.1 (7.7242); chosen E2_a0.7_b0.1.
- Windows where F is missing and the head closing output is used: 39 of 413 valid, 39 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) +0.144 [+0.027, +0.258]; DELTA-F permuted seeds 0 to 4: -0.001, +0.013, +0.010, -0.003, -0.000.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.136 [+0.108, +0.165]; cross-fitted five-fold +0.134 [+0.106, +0.162]; head closing output +0.186 [+0.142, +0.224].
- E3 blend (names nothing): R2 +0.342 [+0.107, +0.475] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.188 [-0.005, +0.306] n=413, chosen per fold {0: 'E2_a0.7_b0.1', 1: 'E2_a0.7_b0.1', 2: 'E2_a0.7_b0.1', 3: 'E2_a0.7_b0.1', 4: 'E2_a0.7_b0.1'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 7.09 m [+6.086, +8.173]; RMSE of the change between consecutive rows 2.33 m [+1.667, +3.056]; lag-1 autocorrelation +0.938 [+0.896, +0.967]. Rows with head presence at least 0.5 (n=2233): RMSE 6.55 m, change 2.35 m, lag-1 +0.926.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.109 | [-0.049, +0.193] | 413 | 49 | 8.1999 |
| E1_K5 | +0.151 | [-0.028, +0.240] | 413 | 43 | 8.0391 |
| E1_K8 | +0.185 | [+0.015, +0.271] | 413 | 39 | 7.9718 |
| E1_K10 | +0.188 | [+0.013, +0.278] | 413 | 38 | 7.9358 |
| E1_K15 | +0.196 | [+0.037, +0.285] | 413 | 36 | 7.9143 |
| E2_a0.2_b0.02 | +0.149 | [-0.028, +0.253] | 413 | 39 | 8.0147 |
| E2_a0.2_b0.05 | +0.203 | [+0.027, +0.295] | 413 | 39 | 7.9399 |
| E2_a0.2_b0.1 | +0.141 | [-0.014, +0.218] | 413 | 39 | 8.1971 |
| E2_a0.2_b0.2 | +0.079 | [-0.045, +0.134] | 413 | 39 | 8.4850 |
| E2_a0.35_b0.02 | +0.132 | [-0.036, +0.237] | 413 | 39 | 8.0255 |
| E2_a0.35_b0.05 | +0.207 | [+0.011, +0.312] | 413 | 39 | 7.8048 |
| E2_a0.35_b0.1 | +0.190 | [+0.009, +0.281] | 413 | 39 | 7.9495 |
| E2_a0.35_b0.2 | +0.126 | [-0.022, +0.197] | 413 | 39 | 8.2573 |
| E2_a0.5_b0.02 | +0.118 | [-0.041, +0.220] | 413 | 39 | 8.0768 |
| E2_a0.5_b0.05 | +0.196 | [+0.002, +0.307] | 413 | 39 | 7.7805 |
| E2_a0.5_b0.1 | +0.211 | [+0.014, +0.312] | 413 | 39 | 7.8100 |
| E2_a0.5_b0.2 | +0.162 | [-0.008, +0.246] | 413 | 39 | 8.0792 |
| E2_a0.7_b0.02 | +0.105 | [-0.043, +0.202] | 413 | 39 | 8.1424 |
| E2_a0.7_b0.05 | +0.178 | [-0.014, +0.292] | 413 | 39 | 7.8096 |
| E2_a0.7_b0.1 | +0.217 | [+0.013, +0.328] | 413 | 39 | 7.7242 |
| E2_a0.7_b0.2 | +0.192 | [+0.006, +0.288] | 413 | 39 | 7.9132 |

### TK64_s0

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K10 (mean fold MSE 7.7684), best E2 E2_a0.7_b0.1 (7.5095); chosen E2_a0.7_b0.1.
- Windows where F is missing and the head closing output is used: 56 of 413 valid, 58 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) +0.025 [-0.032, +0.085]; DELTA-F permuted seeds 0 to 4: -0.003, -0.004, +0.009, -0.004, -0.001.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.160 [+0.127, +0.188]; cross-fitted five-fold +0.159 [+0.126, +0.187]; head closing output +0.220 [+0.176, +0.258].
- E3 blend (names nothing): R2 +0.182 [-0.144, +0.371] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.122 [-0.101, +0.265] n=413, chosen per fold {0: 'E2_a0.7_b0.1', 1: 'E2_a0.7_b0.1', 2: 'E2_a0.7_b0.1', 3: 'E2_a0.7_b0.1', 4: 'E2_a0.7_b0.1'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 7.82 m [+6.295, +9.584]; RMSE of the change between consecutive rows 2.32 m [+1.645, +3.052]; lag-1 autocorrelation +0.952 [+0.908, +0.977]. Rows with head presence at least 0.5 (n=2117): RMSE 7.07 m, change 2.29 m, lag-1 +0.942.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.007 | [-0.225, +0.135] | 413 | 76 | 8.0250 |
| E1_K5 | +0.110 | [-0.059, +0.204] | 413 | 65 | 7.8487 |
| E1_K8 | +0.128 | [-0.051, +0.230] | 413 | 56 | 7.7836 |
| E1_K10 | +0.129 | [-0.055, +0.237] | 413 | 56 | 7.7684 |
| E1_K15 | +0.124 | [-0.058, +0.235] | 413 | 50 | 7.8166 |
| E2_a0.2_b0.02 | +0.104 | [-0.076, +0.221] | 413 | 56 | 7.8530 |
| E2_a0.2_b0.05 | +0.134 | [-0.057, +0.250] | 413 | 56 | 7.7380 |
| E2_a0.2_b0.1 | +0.086 | [-0.094, +0.186] | 413 | 56 | 7.9759 |
| E2_a0.2_b0.2 | +0.048 | [-0.089, +0.122] | 413 | 56 | 8.2711 |
| E2_a0.35_b0.02 | +0.095 | [-0.075, +0.203] | 413 | 56 | 7.8736 |
| E2_a0.35_b0.05 | +0.146 | [-0.059, +0.273] | 413 | 56 | 7.6063 |
| E2_a0.35_b0.1 | +0.129 | [-0.073, +0.243] | 413 | 56 | 7.7187 |
| E2_a0.35_b0.2 | +0.090 | [-0.066, +0.178] | 413 | 56 | 8.0245 |
| E2_a0.5_b0.02 | +0.083 | [-0.077, +0.182] | 413 | 56 | 7.9315 |
| E2_a0.5_b0.05 | +0.143 | [-0.061, +0.271] | 413 | 56 | 7.5916 |
| E2_a0.5_b0.1 | +0.148 | [-0.062, +0.270] | 413 | 56 | 7.5843 |
| E2_a0.5_b0.2 | +0.117 | [-0.057, +0.215] | 413 | 56 | 7.8428 |
| E2_a0.7_b0.02 | +0.071 | [-0.076, +0.162] | 413 | 56 | 8.0021 |
| E2_a0.7_b0.05 | +0.132 | [-0.065, +0.254] | 413 | 56 | 7.6327 |
| E2_a0.7_b0.1 | +0.155 | [-0.056, +0.286] | 413 | 56 | 7.5095 |
| E2_a0.7_b0.2 | +0.137 | [-0.055, +0.246] | 413 | 56 | 7.6798 |

### TK64_s1

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K15 (mean fold MSE 7.8876), best E2 E2_a0.7_b0.1 (7.6293); chosen E2_a0.7_b0.1.
- Windows where F is missing and the head closing output is used: 57 of 413 valid, 59 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) +0.027 [-0.022, +0.077]; DELTA-F permuted seeds 0 to 4: +0.000, +0.001, +0.015, +0.008, -0.003.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.147 [+0.115, +0.176]; cross-fitted five-fold +0.145 [+0.113, +0.173]; head closing output +0.190 [+0.134, +0.238].
- E3 blend (names nothing): R2 +0.206 [-0.092, +0.374] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.117 [-0.033, +0.211] n=413, chosen per fold {0: 'E2_a0.7_b0.1', 1: 'E2_a0.7_b0.1', 2: 'E2_a0.7_b0.2', 3: 'E2_a0.7_b0.2', 4: 'E2_a0.7_b0.1'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 8.32 m [+6.582, +10.398]; RMSE of the change between consecutive rows 2.46 m [+1.796, +3.196]; lag-1 autocorrelation +0.950 [+0.904, +0.975]. Rows with head presence at least 0.5 (n=2099): RMSE 6.79 m, change 2.41 m, lag-1 +0.931.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.065 | [-0.112, +0.158] | 413 | 78 | 8.1893 |
| E1_K5 | +0.105 | [-0.052, +0.186] | 413 | 67 | 8.0387 |
| E1_K8 | +0.140 | [-0.021, +0.233] | 413 | 57 | 7.9555 |
| E1_K10 | +0.149 | [-0.024, +0.250] | 413 | 53 | 7.9027 |
| E1_K15 | +0.150 | [-0.028, +0.268] | 413 | 45 | 7.8876 |
| E2_a0.2_b0.02 | +0.087 | [-0.112, +0.215] | 413 | 57 | 7.8968 |
| E2_a0.2_b0.05 | +0.110 | [-0.074, +0.220] | 413 | 57 | 7.8557 |
| E2_a0.2_b0.1 | +0.112 | [-0.037, +0.193] | 413 | 57 | 8.1343 |
| E2_a0.2_b0.2 | +0.081 | [-0.042, +0.142] | 413 | 57 | 8.4123 |
| E2_a0.35_b0.02 | +0.088 | [-0.087, +0.208] | 413 | 57 | 7.9193 |
| E2_a0.35_b0.05 | +0.127 | [-0.079, +0.253] | 413 | 57 | 7.7020 |
| E2_a0.35_b0.1 | +0.133 | [-0.041, +0.235] | 413 | 57 | 7.8696 |
| E2_a0.35_b0.2 | +0.112 | [-0.026, +0.186] | 413 | 57 | 8.1853 |
| E2_a0.5_b0.02 | +0.086 | [-0.071, +0.198] | 413 | 57 | 7.9835 |
| E2_a0.5_b0.05 | +0.128 | [-0.079, +0.259] | 413 | 57 | 7.6717 |
| E2_a0.5_b0.1 | +0.144 | [-0.054, +0.256] | 413 | 57 | 7.7227 |
| E2_a0.5_b0.2 | +0.132 | [-0.021, +0.218] | 413 | 57 | 8.0108 |
| E2_a0.7_b0.02 | +0.082 | [-0.063, +0.186] | 413 | 57 | 8.0625 |
| E2_a0.7_b0.05 | +0.124 | [-0.073, +0.252] | 413 | 57 | 7.7016 |
| E2_a0.7_b0.1 | +0.149 | [-0.057, +0.270] | 413 | 57 | 7.6293 |
| E2_a0.7_b0.2 | +0.147 | [-0.028, +0.243] | 413 | 57 | 7.8433 |

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
