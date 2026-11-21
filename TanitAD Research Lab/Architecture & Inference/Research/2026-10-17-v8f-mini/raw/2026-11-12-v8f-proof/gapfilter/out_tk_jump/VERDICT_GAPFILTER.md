# VERDICT_GAPFILTER: card PREREG_V8F_PROOF_ADD28 (the closing speed of the lead from a causal filter over the head's own gap reads)

Tier T0 (a diagnostic on logged frames, never driving performance). STRATEGIC: PAUSED (PI item 25). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read. Evidence class MEASURED (ours, T0).

**POST-HOC same-lead variant (step above 3.0 m restarts or truncates the run); written after f2 of the card run failed; names nothing. Not the card's estimator; every number below is information for a later card.**

Counts: 1948 windows, 429 scored, 413 scored with a closing-speed label, 48 scored episodes; episode-cluster bootstrap B 2000, rng 0.

## Controls (first)

- f1_label_path: **PASS** R2 skill this path 0.7499068, banked 0.7499068 (n=413)
- f2_sign_units_alignment: **PASS** E1 K=5 on the label gap: R2 +1.000 (bar at least 0.8), n=413 of 413
- f6_records|TK16_s0: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|TK16_s0: **PASS** selection with real labels E2_a0.7_b0.1, with noise labels E2_a0.7_b0.1; identical True
- x1_alignment_m1_delta|TK16_s0: **PASS** M1 -0.0015 vs N1 result -0.0015; DELTA of the head +0.0195 vs +0.0195
- f3_causal|TK16_s0: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|TK16_s0: **PASS** M1F with permuted sequences -0.283 (bar at most +0.02), seeds 0 to 4: -0.283, -0.198, -0.264, -0.294, -0.296; identity permutation reproduces F, max abs diff 0
- f6_records|TK16_s1: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|TK16_s1: **PASS** selection with real labels E2_a0.7_b0.1, with noise labels E2_a0.7_b0.1; identical True
- x1_alignment_m1_delta|TK16_s1: **PASS** M1 +0.2033 vs N1 result +0.2033; DELTA of the head +0.1440 vs +0.1440
- f3_causal|TK16_s1: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|TK16_s1: **PASS** M1F with permuted sequences -0.356 (bar at most +0.02), seeds 0 to 4: -0.356, -0.224, -0.265, -0.291, -0.334; identity permutation reproduces F, max abs diff 0
- f6_records|TK64_s0: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|TK64_s0: **PASS** selection with real labels E2_a0.7_b0.1, with noise labels E2_a0.7_b0.1; identical True
- x1_alignment_m1_delta|TK64_s0: **PASS** M1 +0.0424 vs N1 result +0.0424; DELTA of the head +0.0248 vs +0.0248
- f3_causal|TK64_s0: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|TK64_s0: **PASS** M1F with permuted sequences -0.281 (bar at most +0.02), seeds 0 to 4: -0.281, -0.237, -0.404, -0.255, -0.381; identity permutation reproduces F, max abs diff 0
- f6_records|TK64_s1: **PASS** records match, one checkpoint per arm, layouts equal: True
- f5_no_eval_data_in_choice|TK64_s1: **PASS** selection with real labels E2_a0.7_b0.1, with noise labels E2_a0.7_b0.1; identical True
- x1_alignment_m1_delta|TK64_s1: **PASS** M1 -0.0012 vs N1 result -0.0012; DELTA of the head +0.0274 vs +0.0274
- f3_causal|TK64_s1: **PASS** max abs change 0.0, NaN pattern mismatches 0, 429 windows x 21 candidates
- f4_permuted|TK64_s1: **PASS** M1F with permuted sequences -0.258 (bar at most +0.02), seeds 0 to 4: -0.258, -0.234, -0.371, -0.215, -0.269; identity permutation reproduces F, max abs diff 0

Controls failed: none

## Names (card section 5)

No outcome is named. POST-HOC same-lead variant (step above 3.0 m restarts or truncates the run); written after f2 of the card run failed; names nothing

Floor: card value 0.089; measured |M1(LSD_s0) - M1(LSD_s1)| = n/a.

## Per arm (413 scored windows with a closing-speed label for M1, M1F, paired; 429 scored windows for DELTA)

| arm | selected F | a, b | M1 head closing R2 | M1F R2 | M1F minus M1 | DELTA-F | DELTA-F permuted (seed 0) | F estimate missing (of 413) |
|---|---|---|---|---|---|---|---|---|
| TK16_s0 | E2_a0.7_b0.1 | +0.583, -0.040 | -0.002 [-0.398, +0.212] n=413 | +0.199 [-0.014, +0.328] n=413 | +0.201 [+0.043, +0.417] n=413 | +0.006 [-0.027, +0.029] n=429 | -0.003 | 63 |
| TK16_s1 | E2_a0.7_b0.1 | +0.595, +0.014 | +0.203 [-0.023, +0.328] n=413 | +0.227 [+0.031, +0.343] n=413 | +0.024 [-0.131, +0.185] n=413 | -0.012 [-0.038, +0.013] n=429 | -0.003 | 61 |
| TK64_s0 | E2_a0.7_b0.1 | +0.599, +0.027 | +0.042 [-0.311, +0.249] n=413 | +0.103 [-0.187, +0.271] n=413 | +0.060 [-0.074, +0.219] n=413 | -0.000 [-0.020, +0.021] n=429 | -0.008 | 84 |
| TK64_s1 | E2_a0.7_b0.1 | +0.569, -0.028 | -0.001 [-0.435, +0.257] n=413 | +0.093 [-0.161, +0.241] n=413 | +0.094 [-0.116, +0.373] n=413 | -0.004 [-0.010, +0.001] n=429 | -0.003 | 85 |

### TK16_s0

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K15 (mean fold MSE 7.5305), best E2 E2_a0.7_b0.1 (7.2678); chosen E2_a0.7_b0.1.
- Windows where F is missing and the head closing output is used: 63 of 413 valid, 65 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) +0.020 [-0.032, +0.066]; DELTA-F permuted seeds 0 to 4: -0.003, +0.003, +0.003, -0.003, -0.000.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.187 [+0.161, +0.213]; cross-fitted five-fold +0.185 [+0.159, +0.211]; head closing output +0.235 [+0.191, +0.279].
- E3 blend (names nothing): R2 +0.149 [-0.165, +0.334] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.161 [-0.046, +0.288] n=413, chosen per fold {0: 'E2_a0.7_b0.05', 1: 'E2_a0.7_b0.1', 2: 'E2_a0.7_b0.05', 3: 'E2_a0.7_b0.1', 4: 'E2_a0.7_b0.1'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 8.58 m [+6.718, +10.644]; RMSE of the change between consecutive rows 2.35 m [+1.669, +3.086]; lag-1 autocorrelation +0.955 [+0.914, +0.979]. Rows with head presence at least 0.5 (n=2173): RMSE 7.15 m, change 2.22 m, lag-1 +0.942.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.076 | [-0.136, +0.181] | 413 | 70 | 7.8260 |
| E1_K5 | +0.141 | [-0.065, +0.250] | 413 | 66 | 7.6390 |
| E1_K8 | +0.138 | [-0.072, +0.251] | 413 | 63 | 7.5653 |
| E1_K10 | +0.137 | [-0.078, +0.251] | 413 | 62 | 7.5381 |
| E1_K15 | +0.106 | [-0.108, +0.225] | 413 | 60 | 7.5305 |
| E2_a0.2_b0.02 | +0.149 | [-0.056, +0.272] | 413 | 63 | 7.4783 |
| E2_a0.2_b0.05 | +0.171 | [-0.037, +0.299] | 413 | 63 | 7.4344 |
| E2_a0.2_b0.1 | +0.151 | [-0.051, +0.258] | 413 | 63 | 7.6554 |
| E2_a0.2_b0.2 | +0.091 | [-0.099, +0.185] | 413 | 63 | 7.9223 |
| E2_a0.35_b0.02 | +0.153 | [-0.043, +0.270] | 413 | 63 | 7.4788 |
| E2_a0.35_b0.05 | +0.188 | [-0.028, +0.319] | 413 | 63 | 7.3208 |
| E2_a0.35_b0.1 | +0.180 | [-0.028, +0.297] | 413 | 63 | 7.4530 |
| E2_a0.35_b0.2 | +0.128 | [-0.072, +0.232] | 413 | 63 | 7.7274 |
| E2_a0.5_b0.02 | +0.150 | [-0.039, +0.261] | 413 | 63 | 7.5107 |
| E2_a0.5_b0.05 | +0.191 | [-0.020, +0.321] | 413 | 63 | 7.2927 |
| E2_a0.5_b0.1 | +0.193 | [-0.019, +0.319] | 413 | 63 | 7.3402 |
| E2_a0.5_b0.2 | +0.153 | [-0.051, +0.265] | 413 | 63 | 7.5850 |
| E2_a0.7_b0.02 | +0.143 | [-0.041, +0.251] | 413 | 63 | 7.5548 |
| E2_a0.7_b0.05 | +0.187 | [-0.020, +0.311] | 413 | 63 | 7.3054 |
| E2_a0.7_b0.1 | +0.199 | [-0.014, +0.328] | 413 | 63 | 7.2678 |
| E2_a0.7_b0.2 | +0.172 | [-0.036, +0.290] | 413 | 63 | 7.4547 |

### TK16_s1

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K15 (mean fold MSE 7.6882), best E2 E2_a0.7_b0.1 (7.4520); chosen E2_a0.7_b0.1.
- Windows where F is missing and the head closing output is used: 61 of 413 valid, 64 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) +0.144 [+0.027, +0.258]; DELTA-F permuted seeds 0 to 4: -0.003, +0.001, +0.003, -0.002, -0.003.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.165 [+0.137, +0.193]; cross-fitted five-fold +0.164 [+0.135, +0.191]; head closing output +0.186 [+0.142, +0.224].
- E3 blend (names nothing): R2 +0.331 [+0.115, +0.458] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.196 [+0.016, +0.311] n=413, chosen per fold {0: 'E2_a0.7_b0.1', 1: 'E2_a0.7_b0.1', 2: 'E2_a0.7_b0.05', 3: 'E2_a0.7_b0.05', 4: 'E2_a0.7_b0.1'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 7.09 m [+6.086, +8.173]; RMSE of the change between consecutive rows 2.33 m [+1.667, +3.056]; lag-1 autocorrelation +0.938 [+0.896, +0.967]. Rows with head presence at least 0.5 (n=2233): RMSE 6.55 m, change 2.35 m, lag-1 +0.926.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.104 | [-0.070, +0.193] | 413 | 69 | 8.0427 |
| E1_K5 | +0.164 | [-0.023, +0.265] | 413 | 64 | 7.8449 |
| E1_K8 | +0.166 | [-0.040, +0.285] | 413 | 61 | 7.7352 |
| E1_K10 | +0.165 | [-0.046, +0.286] | 413 | 61 | 7.6958 |
| E1_K15 | +0.165 | [-0.047, +0.285] | 413 | 59 | 7.6882 |
| E2_a0.2_b0.02 | +0.204 | [+0.044, +0.304] | 413 | 61 | 7.6980 |
| E2_a0.2_b0.05 | +0.213 | [+0.027, +0.328] | 413 | 61 | 7.6111 |
| E2_a0.2_b0.1 | +0.164 | [-0.018, +0.266] | 413 | 61 | 7.8499 |
| E2_a0.2_b0.2 | +0.112 | [-0.049, +0.199] | 413 | 61 | 8.1531 |
| E2_a0.35_b0.02 | +0.199 | [+0.047, +0.296] | 413 | 61 | 7.7132 |
| E2_a0.35_b0.05 | +0.226 | [+0.040, +0.342] | 413 | 61 | 7.5086 |
| E2_a0.35_b0.1 | +0.198 | [+0.002, +0.309] | 413 | 61 | 7.6373 |
| E2_a0.35_b0.2 | +0.145 | [-0.037, +0.242] | 413 | 61 | 7.9416 |
| E2_a0.5_b0.02 | +0.192 | [+0.046, +0.286] | 413 | 61 | 7.7518 |
| E2_a0.5_b0.05 | +0.228 | [+0.045, +0.340] | 413 | 61 | 7.4928 |
| E2_a0.5_b0.1 | +0.216 | [+0.018, +0.330] | 413 | 61 | 7.5220 |
| E2_a0.5_b0.2 | +0.169 | [-0.024, +0.273] | 413 | 61 | 7.7847 |
| E2_a0.7_b0.02 | +0.184 | [+0.042, +0.278] | 413 | 61 | 7.7992 |
| E2_a0.7_b0.05 | +0.224 | [+0.052, +0.331] | 413 | 61 | 7.5183 |
| E2_a0.7_b0.1 | +0.227 | [+0.031, +0.343] | 413 | 61 | 7.4520 |
| E2_a0.7_b0.2 | +0.191 | [-0.007, +0.299] | 413 | 61 | 7.6414 |

### TK64_s0

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K10 (mean fold MSE 7.5273), best E2 E2_a0.7_b0.1 (7.3100); chosen E2_a0.7_b0.1.
- Windows where F is missing and the head closing output is used: 84 of 413 valid, 88 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) +0.025 [-0.032, +0.085]; DELTA-F permuted seeds 0 to 4: -0.008, -0.005, +0.002, -0.005, +0.002.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.183 [+0.151, +0.211]; cross-fitted five-fold +0.181 [+0.150, +0.209]; head closing output +0.220 [+0.176, +0.258].
- E3 blend (names nothing): R2 +0.165 [-0.163, +0.359] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.085 [-0.201, +0.253] n=413, chosen per fold {0: 'E2_a0.7_b0.1', 1: 'E2_a0.7_b0.1', 2: 'E2_a0.7_b0.1', 3: 'E2_a0.7_b0.1', 4: 'E2_a0.7_b0.1'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 7.82 m [+6.295, +9.584]; RMSE of the change between consecutive rows 2.32 m [+1.645, +3.052]; lag-1 autocorrelation +0.952 [+0.908, +0.977]. Rows with head presence at least 0.5 (n=2117): RMSE 7.07 m, change 2.29 m, lag-1 +0.942.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.020 | [-0.249, +0.164] | 413 | 93 | 7.8332 |
| E1_K5 | +0.071 | [-0.212, +0.222] | 413 | 90 | 7.6254 |
| E1_K8 | +0.086 | [-0.200, +0.238] | 413 | 84 | 7.5461 |
| E1_K10 | +0.084 | [-0.201, +0.242] | 413 | 84 | 7.5273 |
| E1_K15 | +0.066 | [-0.221, +0.224] | 413 | 79 | 7.5493 |
| E2_a0.2_b0.02 | +0.032 | [-0.261, +0.204] | 413 | 84 | 7.6329 |
| E2_a0.2_b0.05 | +0.085 | [-0.202, +0.255] | 413 | 84 | 7.4773 |
| E2_a0.2_b0.1 | +0.075 | [-0.213, +0.227] | 413 | 84 | 7.6211 |
| E2_a0.2_b0.2 | +0.023 | [-0.251, +0.163] | 413 | 84 | 7.9179 |
| E2_a0.35_b0.02 | +0.036 | [-0.253, +0.200] | 413 | 84 | 7.6513 |
| E2_a0.35_b0.05 | +0.085 | [-0.204, +0.260] | 413 | 84 | 7.3925 |
| E2_a0.35_b0.1 | +0.100 | [-0.186, +0.262] | 413 | 84 | 7.4400 |
| E2_a0.35_b0.2 | +0.064 | [-0.213, +0.208] | 413 | 84 | 7.7021 |
| E2_a0.5_b0.02 | +0.034 | [-0.249, +0.195] | 413 | 84 | 7.6911 |
| E2_a0.5_b0.05 | +0.080 | [-0.215, +0.253] | 413 | 84 | 7.3928 |
| E2_a0.5_b0.1 | +0.106 | [-0.183, +0.272] | 413 | 84 | 7.3517 |
| E2_a0.5_b0.2 | +0.088 | [-0.199, +0.240] | 413 | 84 | 7.5569 |
| E2_a0.7_b0.02 | +0.030 | [-0.250, +0.191] | 413 | 84 | 7.7387 |
| E2_a0.7_b0.05 | +0.072 | [-0.226, +0.242] | 413 | 84 | 7.4333 |
| E2_a0.7_b0.1 | +0.103 | [-0.187, +0.271] | 413 | 84 | 7.3100 |
| E2_a0.7_b0.2 | +0.103 | [-0.181, +0.260] | 413 | 84 | 7.4356 |

### TK64_s1

- Selection on the train probe clips (30865 closing rows, five clip-grouped folds): best E1 E1_K15 (mean fold MSE 7.6792), best E2 E2_a0.7_b0.1 (7.4469); chosen E2_a0.7_b0.1.
- Windows where F is missing and the head closing output is used: 85 of 413 valid, 91 of 429 scored; fallback also missing: 0.
- Reported beside, names nothing: DELTA of the head's own closing output (same call) +0.027 [-0.022, +0.077]; DELTA-F permuted seeds 0 to 4: -0.003, -0.001, +0.021, +0.000, -0.002.
- M1F on the train probe clips (names nothing; 30865 closing rows, 1200 clips): in sample +0.167 [+0.138, +0.193]; cross-fitted five-fold +0.166 [+0.136, +0.192]; head closing output +0.190 [+0.134, +0.238].
- E3 blend (names nothing): R2 +0.161 [-0.143, +0.333] n=413. F with every choice cross-fitted inside the evaluation bank (names nothing): R2 +0.067 [-0.161, +0.195] n=413, chosen per fold {0: 'E1_K15', 1: 'E1_K15', 2: 'E2_a0.7_b0.05', 3: 'E1_K15', 4: 'E2_a0.7_b0.1'}.
- f7 (names nothing), head gap error on evaluation rows with a gap label (n=2376 rows, 2312 consecutive pairs): RMSE 8.32 m [+6.582, +10.398]; RMSE of the change between consecutive rows 2.46 m [+1.796, +3.196]; lag-1 autocorrelation +0.950 [+0.904, +0.975]. Rows with head presence at least 0.5 (n=2099): RMSE 6.79 m, change 2.41 m, lag-1 +0.931.

M1F by candidate (calibration fitted on the train probe clips, applied to the 413 windows; names nothing):

| candidate | M1F R2 | interval | n | estimate missing | train CV MSE |
|---|---|---|---|---|---|
| E1_K3 | +0.038 | [-0.210, +0.163] | 413 | 104 | 7.9499 |
| E1_K5 | +0.060 | [-0.192, +0.185] | 413 | 94 | 7.8033 |
| E1_K8 | +0.084 | [-0.158, +0.217] | 413 | 85 | 7.7259 |
| E1_K10 | +0.088 | [-0.160, +0.229] | 413 | 81 | 7.7032 |
| E1_K15 | +0.078 | [-0.177, +0.225] | 413 | 75 | 7.6792 |
| E2_a0.2_b0.02 | +0.038 | [-0.220, +0.190] | 413 | 85 | 7.6392 |
| E2_a0.2_b0.05 | +0.066 | [-0.188, +0.211] | 413 | 85 | 7.5941 |
| E2_a0.2_b0.1 | +0.071 | [-0.163, +0.196] | 413 | 85 | 7.7899 |
| E2_a0.2_b0.2 | +0.048 | [-0.189, +0.162] | 413 | 85 | 8.0663 |
| E2_a0.35_b0.02 | +0.049 | [-0.200, +0.196] | 413 | 85 | 7.6411 |
| E2_a0.35_b0.05 | +0.071 | [-0.190, +0.222] | 413 | 85 | 7.4906 |
| E2_a0.35_b0.1 | +0.083 | [-0.164, +0.220] | 413 | 85 | 7.6150 |
| E2_a0.35_b0.2 | +0.072 | [-0.168, +0.193] | 413 | 85 | 7.8841 |
| E2_a0.5_b0.02 | +0.055 | [-0.190, +0.197] | 413 | 85 | 7.6705 |
| E2_a0.5_b0.05 | +0.074 | [-0.187, +0.226] | 413 | 85 | 7.4664 |
| E2_a0.5_b0.1 | +0.090 | [-0.164, +0.233] | 413 | 85 | 7.5148 |
| E2_a0.5_b0.2 | +0.087 | [-0.151, +0.214] | 413 | 85 | 7.7499 |
| E2_a0.7_b0.02 | +0.060 | [-0.182, +0.197] | 413 | 85 | 7.7100 |
| E2_a0.7_b0.05 | +0.076 | [-0.182, +0.228] | 413 | 85 | 7.4780 |
| E2_a0.7_b0.1 | +0.093 | [-0.161, +0.241] | 413 | 85 | 7.4469 |
| E2_a0.7_b0.2 | +0.097 | [-0.146, +0.232] | 413 | 85 | 7.6248 |

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
