# RESULT_LATSEM2B: does each world-model level encode and imagine the scene, read with a permutation-calibrated instrument?

MEASURED 2026-11-07, dev-box CPU, `run_latsem2b.py` (26 min wall, 4 threads). **Tier T0: a world-model diagnostic, never a driving claim. The planner reads none of these predictions.**

**Controls: all pass on both datasets. The outcomes below are decisions of the card.**

* Card `PREREG_V8F_LATSEM2B.md` (frozen prefix 4,611 bytes, `bf6b3717`), `AMENDMENT_LATSEM2B_1.md` (prefix 2,778 bytes, `120072ff`), `IMPL_CHOICES_LATSEM2B.md` (16,005 bytes, `a90cd9ca`, frozen 13:44:00 before any statistic). The run asserts all three hashes.
* **Deciding set: TRAIN** (WMACT's 141 TRAIN clips, M2b's own training clips, so optimistic for S2): 9,857 bank frames, 141 episodes (77 / 64 by fold), 1,833 windows with 16 real frames, 461 S3 windows (56 episodes). **Beside, post hoc: eval-141** (LATSEM2's bank): 9,460 frames, 139 episodes, 1,948 windows, 505 S3 windows (53 episodes).
* Estimators: the episode-cluster bootstrap (B 2,000, seed 0) answers "another draw of EPISODES"; the two model seeds stand in for "another training run"; the permutation reference (R = 199 draws per cell, plus 20 calibration draws) answers "would this appear with no relation between latent and label?". A row is separated (`†`) only if its interval excludes 0 on the claimed side AND its point is beyond the cell's 97.5th (S3-PARTIAL: 2.5th) reference percentile. `*` marks a row whose interval alone excludes 0 on the claimed side (the first part only). `thr` is the cell's 97.5th percentile, `p` the reference rank of the point.
* Probes: ridge lambda 10 (continuous), logistic C 0.1 (binary, 7-class), cross-fitted over two episode folds, features standardised on the fit fold: run 1's, unchanged. Code `run_latsem2b.py`; raw numbers `RESULT_LATSEM2B.json`; reference draws `draws_train.jsonl`, `draws_eval141.jsonl`; log `run_latsem2b.run1.log`.

## Controls (first; a failing control makes the read INVALID)

| control | committed value | TRAIN observed | eval-141 observed | status |
|---|---|---|---|---|
| copy and per-level checks (not controls) | copies agree to 1e-5; per-frame z2 / z3 equal `encode` to 1e-5 | z1 0.0e+00, actions 0.0, levels 1.2e-06 | z1 1.2e-06, actions 0.0, levels 6.6e-07 | PASS |
| C1 start-state query SKILL | exactly 0.0 on every S2 row | max abs 0.0, interval 0.0 (126 rows) | max abs 0.0, interval 0.0 (126 rows) | PASS |
| C2 identical action sequences, S3 and S3-PARTIAL | exactly 0.0, interval [0, 0] | max abs 0.0 | max abs 0.0 | PASS |
| C4 speed removed | abs corr(residual gap, ego speed) <= 1e-6 on every fit fold | max abs 2.4e-16 (permuted draws: 1.2e-15) | max abs 5.6e-16 (permuted draws: 1.7e-15) | PASS |
| C3 calibration | share declared separated <= 5 % in each of four families | see next table | see next table | PASS |

**C3 in detail** (20 calibration draws, seeds 1001 to 1020, each run through the full two-part rule as if real; rows pool both model seeds, three levels and all 9 targets):

| family | dataset | rows | declared by the two-part rule | share | share by the interval alone (first part only) | <= 5 % |
|---|---|---|---|---|---|---|
| S1_classification | TRAIN | 360 | 10 | 2.78 % | 7.50 % (27) | yes |
| S1_classification | eval-141 | 360 | 9 | 2.50 % | 3.89 % (14) | yes |
| S1_continuous | TRAIN | 720 | 0 | 0.00 % | 0.00 % (0) | yes |
| S1_continuous | eval-141 | 720 | 0 | 0.00 % | 0.00 % (0) | yes |
| S2 | TRAIN | 2,520 | 90 | 3.57 % | 57.98 % (1461) | yes |
| S2 | eval-141 | 2,520 | 76 | 3.02 % | 57.74 % (1455) | yes |
| S3_PARTIAL | TRAIN | 120 | 1 | 0.83 % | 19.17 % (23) | yes |
| S3_PARTIAL | eval-141 | 120 | 1 | 0.83 % | 25.83 % (31) | yes |

Instrument notes: logistic fits in the real rows TRAIN 36 (not converged 0), eval-141 36 (not converged 0); in the draws TRAIN 7,884 (not converged 0), eval-141 7,884 (not converged 0). Distinct permuted label vectors per cell over the 219 draws (model seed s0): minimum 219 on TRAIN and 219 on eval-141 (a diagnostic; targets with few labelled episodes can repeat a vector). Reference points that were NaN and dropped: TRAIN 0, eval-141 0. Balanced-accuracy self-test: perfect 1.0, baseline 0.3333, same-as-base 0.0.
Reproduction check (not a control): the real eval-141 rows of this run against run 1's `RESULT_LATSEM2.json` (same probes, same labels). Max abs difference: S1 points 3.3e-04, S1 interval ends 3.9e-04, S2 decision points 2.4e-03, raw S3 points 3.8e-07. By probe kind: S1 binary 0.0e+00, continuous 2.4e-14, 7-class 3.3e-04; S2 binary 0.0e+00, continuous 2.0e-07, 7-class 2.4e-03. Binary and continuous rows reproduce; the 7-class multinomial rows differ at the 1e-4 to 1e-3 level, most likely the lbfgs stopping point (tol 1e-4, max_iter 5000, no non-convergence) moving with the BLAS thread layout (not tested here). The 7-class rows are not outcome targets of the card (SCENE-AWARE reads lead gap and agents within 30 m; SCENE-IMAGINING reads lead gap).

## Outcomes (the card's definitions, per level; deciding set first, eval-141 beside)

| level | SCENE-AWARE | by lead gap: TRAIN ENCODES both seeds / eval-141 points above 0 | by agents within 30 m: TRAIN / eval-141 | SCENE-IMAGINING (lead gap, longest horizon): TRAIN IMAGINES / eval-141 above 0 | ACTION-SEMANTIC-LEAD: TRAIN S3-PARTIAL separated below 0 / eval-141 below 0 |
|---|---|---|---|---|---|
| L1 | **yes** | no / yes | yes / yes | **no**: no / no | **no**: no / no |
| L2 | **no** | no / no | no / no | **no**: no / yes | **no**: no / no |
| L3 | **no** | no / no | no / no | **no**: no / yes | **no**: no / no |

* **Next lever (card):** SCENE-AWARE at levels [1], SCENE-IMAGINING at []: L1: the reliance arm on that level.
* ENCODES on TRAIN at both seeds (two-part rule), all targets: L1/agents30, L1/agents30_view, L1/lead_present, L1/lon7, L1/nearest, L1/turn, L2/agents30_view, L2/lead_present, L2/lon7, L2/turn, L3/agents30_view, L3/lead_present, L3/lon7, L3/turn.
* IMAGINES on TRAIN at both seeds (two-part rule), all targets and horizons: L3/k3/lon7, L3/k3/turn.
* Beside, on eval-141 (post hoc, its own reference): ENCODES at both seeds: L1/lead_present, L1/lon7, L1/turn, L2/lead_present, L2/lon7, L2/turn, L3/lead_present, L3/lon7, L3/turn; IMAGINES at both seeds: none.
* SCENE-AWARE at L1 holds through `agents30`: s0: TRAIN +0.357 [+0.191, +0.478]†, eval-141 +0.308 [+0.036, +0.429]†; s1: TRAIN +0.329 [+0.122, +0.485]†, eval-141 +0.231 [-0.162, +0.391]. The card asks for the same sign on eval-141 (point above 0 at both seeds); eval-141's own two-part separation is shown here and decides nothing.
* Instrument note: for the continuous S1 targets (R2) the 97.5th reference percentile is negative (range -0.94 to -0.15 over 72 cells), because a probe fitted on permuted labels scores a negative R2 on the other fold. The reference therefore adds little to the interval there, and calibration reads 0.00 % for that family with or without it; the 7-class, binary and S2 rows are where the reference binds (S2: the interval alone declares 58 % of calibration rows, the two-part rule 3.6 % on TRAIN).

## S1: what each level ENCODES (R2 for continuous targets, balanced accuracy minus the majority baseline for classes)

### TRAIN (deciding)

| target | n frames | level | seed s0: point [95 % interval] | seed s1: point [95 % interval] |
|---|---|---|---|---|
| lead present (BA - majority) | 8,964 | L1 | +0.165 [+0.104, +0.225]† thr +0.078, p 0.005 | +0.181 [+0.129, +0.238]† thr +0.066, p 0.005 |
| lead present (BA - majority) | 8,964 | L2 | +0.174 [+0.117, +0.232]† thr +0.069, p 0.005 | +0.160 [+0.096, +0.228]† thr +0.062, p 0.005 |
| lead present (BA - majority) | 8,964 | L3 | +0.157 [+0.104, +0.209]† thr +0.065, p 0.005 | +0.163 [+0.104, +0.220]† thr +0.052, p 0.005 |
| lead gap (R2) | 2,456 | L1 | +0.163 [-0.103, +0.394] thr -0.490, p 0.005 | +0.215 [+0.010, +0.397]† thr -0.529, p 0.005 |
| lead gap (R2) | 2,456 | L2 | +0.166 [-0.141, +0.409] thr -0.425, p 0.005 | -0.112 [-0.506, +0.186] thr -0.384, p 0.010 |
| lead gap (R2) | 2,456 | L3 | +0.099 [-0.168, +0.305] thr -0.468, p 0.005 | -0.018 [-0.369, +0.267] thr -0.404, p 0.005 |
| lead closing speed (R2) | 2,330 | L1 | -1.238 [-2.584, -0.394] thr -0.829, p 0.210 | -1.143 [-2.227, -0.421] thr -0.834, p 0.145 |
| lead closing speed (R2) | 2,330 | L2 | -0.932 [-1.739, -0.374] thr -0.889, p 0.045 | -0.766 [-1.327, -0.364] thr -0.941, p 0.005 |
| lead closing speed (R2) | 2,330 | L3 | -0.595 [-1.190, -0.254] thr -0.730, p 0.015 | -0.535 [-1.118, -0.162] thr -0.756, p 0.010 |
| agents within 30 m (R2) | 8,964 | L1 | +0.357 [+0.191, +0.478]† thr -0.196, p 0.005 | +0.329 [+0.122, +0.485]† thr -0.228, p 0.005 |
| agents within 30 m (R2) | 8,964 | L2 | +0.227 [-0.017, +0.400] thr -0.322, p 0.005 | +0.362 [+0.164, +0.494]† thr -0.257, p 0.005 |
| agents within 30 m (R2) | 8,964 | L3 | +0.156 [-0.015, +0.274] thr -0.236, p 0.005 | +0.246 [+0.066, +0.373]† thr -0.247, p 0.005 |
| nearest-agent distance (R2) | 8,877 | L1 | +0.277 [+0.108, +0.392]† thr -0.345, p 0.005 | +0.198 [+0.033, +0.312]† thr -0.328, p 0.005 |
| nearest-agent distance (R2) | 8,877 | L2 | +0.066 [-0.125, +0.190] thr -0.409, p 0.005 | +0.174 [-0.000, +0.292] thr -0.350, p 0.005 |
| nearest-agent distance (R2) | 8,877 | L3 | +0.112 [-0.006, +0.200] thr -0.252, p 0.005 | +0.238 [+0.116, +0.328]† thr -0.326, p 0.005 |
| 7-class longitudinal (BA - majority) | 8,758 | L1 | +0.095 [+0.055, +0.143]† thr +0.036, p 0.005 | +0.093 [+0.057, +0.134]† thr +0.040, p 0.005 |
| 7-class longitudinal (BA - majority) | 8,758 | L2 | +0.081 [+0.051, +0.115]† thr +0.030, p 0.005 | +0.071 [+0.043, +0.103]† thr +0.029, p 0.005 |
| 7-class longitudinal (BA - majority) | 8,758 | L3 | +0.087 [+0.058, +0.117]† thr +0.033, p 0.005 | +0.089 [+0.061, +0.118]† thr +0.029, p 0.005 |
| turn ahead (BA - majority) | 9,716 | L1 | +0.247 [+0.190, +0.298]† thr +0.049, p 0.005 | +0.246 [+0.192, +0.299]† thr +0.038, p 0.005 |
| turn ahead (BA - majority) | 9,716 | L2 | +0.237 [+0.187, +0.288]† thr +0.054, p 0.005 | +0.252 [+0.202, +0.297]† thr +0.052, p 0.005 |
| turn ahead (BA - majority) | 9,716 | L3 | +0.254 [+0.207, +0.297]† thr +0.051, p 0.005 | +0.273 [+0.220, +0.318]† thr +0.046, p 0.005 |
| beside: closing speed, no 20 m/s cap (R2) | 2,350 | L1 | -0.575 [-1.310, -0.176] thr -0.353, p 0.420 | -0.492 [-1.088, -0.179] thr -0.318, p 0.135 |
| beside: closing speed, no 20 m/s cap (R2) | 2,350 | L2 | -0.486 [-1.048, -0.161] thr -0.370, p 0.070 | -0.392 [-0.785, -0.190] thr -0.387, p 0.035 |
| beside: closing speed, no 20 m/s cap (R2) | 2,350 | L3 | -0.561 [-1.238, -0.215] thr -0.355, p 0.495 | -0.297 [-0.648, -0.114] thr -0.254, p 0.060 |
| beside: agents within 30 m in the frustum (R2) | 8,964 | L1 | +0.394 [+0.269, +0.481]† thr -0.151, p 0.005 | +0.361 [+0.199, +0.463]† thr -0.194, p 0.005 |
| beside: agents within 30 m in the frustum (R2) | 8,964 | L2 | +0.234 [+0.019, +0.369]† thr -0.224, p 0.005 | +0.360 [+0.170, +0.472]† thr -0.284, p 0.005 |
| beside: agents within 30 m in the frustum (R2) | 8,964 | L3 | +0.194 [+0.065, +0.294]† thr -0.213, p 0.005 | +0.302 [+0.133, +0.405]† thr -0.217, p 0.005 |

### eval-141 (beside, post hoc)

| target | n frames | level | seed s0: point [95 % interval] | seed s1: point [95 % interval] |
|---|---|---|---|---|
| lead present (BA - majority) | 8,874 | L1 | +0.196 [+0.132, +0.260]† thr +0.064, p 0.005 | +0.186 [+0.124, +0.248]† thr +0.064, p 0.005 |
| lead present (BA - majority) | 8,874 | L2 | +0.118 [+0.055, +0.178]† thr +0.057, p 0.005 | +0.191 [+0.131, +0.250]† thr +0.060, p 0.005 |
| lead present (BA - majority) | 8,874 | L3 | +0.140 [+0.084, +0.195]† thr +0.061, p 0.005 | +0.242 [+0.180, +0.297]† thr +0.052, p 0.005 |
| lead gap (R2) | 2,489 | L1 | +0.245 [-0.004, +0.456] thr -0.342, p 0.005 | +0.033 [-0.325, +0.313] thr -0.390, p 0.005 |
| lead gap (R2) | 2,489 | L2 | -0.467 [-1.228, +0.049] thr -0.292, p 0.135 | -0.050 [-0.441, +0.258] thr -0.350, p 0.005 |
| lead gap (R2) | 2,489 | L3 | -0.056 [-0.472, +0.255] thr -0.214, p 0.010 | -0.119 [-0.551, +0.243] thr -0.301, p 0.005 |
| lead closing speed (R2) | 2,311 | L1 | -1.171 [-2.041, -0.513] thr -0.530, p 0.685 | -0.539 [-1.027, -0.177] thr -0.537, p 0.035 |
| lead closing speed (R2) | 2,311 | L2 | -0.813 [-1.377, -0.396] thr -0.542, p 0.315 | -0.921 [-1.763, -0.327] thr -0.496, p 0.370 |
| lead closing speed (R2) | 2,311 | L3 | -0.560 [-0.921, -0.282] thr -0.399, p 0.135 | -0.511 [-1.096, -0.164] thr -0.420, p 0.065 |
| agents within 30 m (R2) | 8,874 | L1 | +0.308 [+0.036, +0.429]† thr -0.297, p 0.005 | +0.231 [-0.162, +0.391] thr -0.277, p 0.005 |
| agents within 30 m (R2) | 8,874 | L2 | -0.246 [-1.268, +0.202] thr -0.448, p 0.005 | +0.032 [-0.676, +0.317] thr -0.398, p 0.005 |
| agents within 30 m (R2) | 8,874 | L3 | -0.230 [-1.021, +0.092] thr -0.277, p 0.015 | +0.003 [-0.830, +0.319] thr -0.328, p 0.005 |
| nearest-agent distance (R2) | 8,636 | L1 | +0.112 [-0.143, +0.269] thr -0.560, p 0.005 | +0.138 [-0.124, +0.291] thr -0.543, p 0.005 |
| nearest-agent distance (R2) | 8,636 | L2 | +0.133 [-0.151, +0.307] thr -0.622, p 0.005 | +0.120 [-0.161, +0.298] thr -0.571, p 0.005 |
| nearest-agent distance (R2) | 8,636 | L3 | +0.072 [-0.221, +0.244] thr -0.492, p 0.005 | +0.146 [-0.125, +0.313] thr -0.502, p 0.005 |
| 7-class longitudinal (BA - majority) | 8,834 | L1 | +0.081 [+0.052, +0.111]† thr +0.059, p 0.005 | +0.089 [+0.055, +0.121]† thr +0.043, p 0.005 |
| 7-class longitudinal (BA - majority) | 8,834 | L2 | +0.076 [+0.043, +0.112]† thr +0.035, p 0.010 | +0.122 [+0.050, +0.189]† thr +0.037, p 0.005 |
| 7-class longitudinal (BA - majority) | 8,834 | L3 | +0.101 [+0.066, +0.140]† thr +0.033, p 0.005 | +0.127 [+0.049, +0.196]† thr +0.040, p 0.005 |
| turn ahead (BA - majority) | 9,460 | L1 | +0.161 [+0.113, +0.211]† thr +0.055, p 0.005 | +0.176 [+0.116, +0.235]† thr +0.055, p 0.005 |
| turn ahead (BA - majority) | 9,460 | L2 | +0.203 [+0.138, +0.267]† thr +0.065, p 0.005 | +0.172 [+0.109, +0.237]† thr +0.049, p 0.005 |
| turn ahead (BA - majority) | 9,460 | L3 | +0.209 [+0.157, +0.261]† thr +0.073, p 0.005 | +0.212 [+0.157, +0.268]† thr +0.057, p 0.005 |
| beside: closing speed, no 20 m/s cap (R2) | 2,364 | L1 | -0.663 [-1.881, -0.264] thr -0.552, p 0.055 | -0.390 [-1.179, -0.140] thr -0.600, p 0.005 |
| beside: closing speed, no 20 m/s cap (R2) | 2,364 | L2 | -0.913 [-2.420, -0.384] thr -0.533, p 0.230 | -0.665 [-1.899, -0.259] thr -0.573, p 0.065 |
| beside: closing speed, no 20 m/s cap (R2) | 2,364 | L3 | -0.478 [-1.245, -0.224] thr -0.520, p 0.020 | -0.266 [-0.786, -0.096] thr -0.387, p 0.005 |
| beside: agents within 30 m in the frustum (R2) | 8,874 | L1 | +0.233 [-0.180, +0.356] thr -0.240, p 0.005 | +0.141 [-0.449, +0.310] thr -0.222, p 0.005 |
| beside: agents within 30 m in the frustum (R2) | 8,874 | L2 | -0.199 [-1.467, +0.213] thr -0.342, p 0.005 | -0.055 [-1.108, +0.247] thr -0.359, p 0.005 |
| beside: agents within 30 m in the frustum (R2) | 8,874 | L3 | -0.301 [-1.478, +0.064] thr -0.261, p 0.070 | -0.112 [-1.278, +0.227] thr -0.298, p 0.005 |

## S2: what each level IMAGINES (SKILL = 1 - error of the rolled-out state / error of the start state, true actions; L1 shrunk, L2 and L3 raw)

### Lead gap at every horizon

| dataset | level, horizon | seed s0: SKILL [95 % interval] | seed s1: SKILL [95 % interval] |
|---|---|---|---|
| TRAIN | L1 k1 (0.2 s) | +0.027 [-0.003, +0.061] thr +0.054, p 0.280 | +0.034 [-0.007, +0.071] thr +0.048, p 0.145 |
| TRAIN | L1 k5 (1.0 s) | +0.053 [+0.003, +0.100]* thr +0.093, p 0.345 | +0.103 [+0.029, +0.167]† thr +0.096, p 0.010 |
| TRAIN | L1 k10 (2.0 s) | +0.049 [-0.020, +0.109] thr +0.122, p 0.530 | +0.144 [+0.050, +0.235]† thr +0.120, p 0.010 |
| TRAIN | L2 k1 (0.4 s) | +0.126 [+0.018, +0.218]* thr +0.145, p 0.085 | +0.089 [-0.005, +0.166] thr +0.140, p 0.325 |
| TRAIN | L2 k3 (1.2 s) | +0.192 [+0.062, +0.305]* thr +0.234, p 0.110 | +0.127 [+0.011, +0.223]* thr +0.226, p 0.515 |
| TRAIN | L3 k1 (0.8 s) | +0.085 [-0.082, +0.237] thr +0.221, p 0.765 | +0.129 [-0.011, +0.231] thr +0.235, p 0.440 |
| TRAIN | L3 k3 (2.4 s) | +0.102 [-0.089, +0.270] thr +0.285, p 0.885 | +0.192 [+0.024, +0.315]* thr +0.307, p 0.475 |
| eval-141 | L1 k1 (0.2 s) | -0.011 [-0.055, +0.030] thr +0.050, p 0.990 | +0.007 [-0.026, +0.043] thr +0.063, p 0.855 |
| eval-141 | L1 k5 (1.0 s) | -0.013 [-0.099, +0.057] thr +0.105, p 0.985 | +0.006 [-0.062, +0.073] thr +0.101, p 0.850 |
| eval-141 | L1 k10 (2.0 s) | -0.017 [-0.140, +0.079] thr +0.142, p 0.975 | -0.018 [-0.118, +0.079] thr +0.134, p 0.930 |
| eval-141 | L2 k1 (0.4 s) | +0.074 [+0.017, +0.111]* thr +0.128, p 0.435 | +0.012 [-0.067, +0.084] thr +0.126, p 0.930 |
| eval-141 | L2 k3 (1.2 s) | +0.090 [+0.012, +0.148]* thr +0.211, p 0.715 | +0.029 [-0.117, +0.140] thr +0.212, p 0.945 |
| eval-141 | L3 k1 (0.8 s) | +0.048 [-0.093, +0.163] thr +0.201, p 0.760 | +0.023 [-0.120, +0.118] thr +0.231, p 0.925 |
| eval-141 | L3 k3 (2.4 s) | +0.076 [-0.069, +0.216] thr +0.275, p 0.770 | +0.125 [-0.072, +0.255] thr +0.303, p 0.755 |

### Every target at the longest horizon of each level

#### TRAIN (deciding)

| target | level, horizon | seed s0 | seed s1 |
|---|---|---|---|
| lead present (BA - majority) | L1 k10 | +0.000 [-0.067, +0.064] thr +0.099, p 0.800 | +0.002 [-0.073, +0.075] thr +0.096, p 0.715 |
| lead present (BA - majority) | L2 k3 | +0.015 [-0.108, +0.120] thr +0.143, p 0.860 | +0.005 [-0.060, +0.065] thr +0.134, p 0.885 |
| lead present (BA - majority) | L3 k3 | +0.126 [+0.017, +0.232]* thr +0.190, p 0.300 | +0.083 [-0.020, +0.177] thr +0.194, p 0.640 |
| lead gap (R2) | L1 k10 | +0.049 [-0.020, +0.109] thr +0.122, p 0.530 | +0.144 [+0.050, +0.235]† thr +0.120, p 0.010 |
| lead gap (R2) | L2 k3 | +0.192 [+0.062, +0.305]* thr +0.234, p 0.110 | +0.127 [+0.011, +0.223]* thr +0.226, p 0.515 |
| lead gap (R2) | L3 k3 | +0.102 [-0.089, +0.270] thr +0.285, p 0.885 | +0.192 [+0.024, +0.315]* thr +0.307, p 0.475 |
| lead closing speed (R2) | L1 k10 | +0.100 [+0.026, +0.149]* thr +0.145, p 0.210 | +0.068 [+0.004, +0.144]* thr +0.134, p 0.375 |
| lead closing speed (R2) | L2 k3 | +0.094 [+0.001, +0.186]* thr +0.255, p 0.860 | +0.122 [+0.035, +0.210]* thr +0.256, p 0.710 |
| lead closing speed (R2) | L3 k3 | +0.294 [+0.159, +0.410]* thr +0.407, p 0.285 | +0.261 [+0.120, +0.377]* thr +0.401, p 0.435 |
| agents within 30 m (R2) | L1 k10 | +0.039 [-0.013, +0.082] thr +0.124, p 0.825 | +0.089 [+0.034, +0.141]* thr +0.122, p 0.245 |
| agents within 30 m (R2) | L2 k3 | +0.133 [+0.024, +0.227]* thr +0.226, p 0.575 | +0.088 [-0.010, +0.177] thr +0.229, p 0.880 |
| agents within 30 m (R2) | L3 k3 | +0.088 [+0.003, +0.182]* thr +0.273, p 0.950 | +0.121 [+0.004, +0.248]* thr +0.268, p 0.870 |
| nearest-agent distance (R2) | L1 k10 | +0.011 [-0.022, +0.047] thr +0.139, p 0.965 | +0.009 [-0.038, +0.066] thr +0.139, p 0.990 |
| nearest-agent distance (R2) | L2 k3 | +0.102 [+0.038, +0.169]* thr +0.255, p 0.800 | +0.094 [+0.034, +0.165]* thr +0.253, p 0.920 |
| nearest-agent distance (R2) | L3 k3 | +0.070 [+0.005, +0.138]* thr +0.300, p 0.990 | -0.011 [-0.108, +0.083] thr +0.317, p 1.000 |
| 7-class longitudinal (BA - majority) | L1 k10 | +0.010 [-0.022, +0.037] thr +0.036, p 0.360 | +0.010 [-0.024, +0.047] thr +0.032, p 0.385 |
| 7-class longitudinal (BA - majority) | L2 k3 | +0.052 [+0.014, +0.092]† thr +0.044, p 0.015 | +0.018 [-0.014, +0.053] thr +0.050, p 0.315 |
| 7-class longitudinal (BA - majority) | L3 k3 | +0.089 [+0.048, +0.128]† thr +0.066, p 0.005 | +0.068 [+0.014, +0.124]† thr +0.068, p 0.030 |
| turn ahead (BA - majority) | L1 k10 | +0.115 [+0.047, +0.188]* thr +0.128, p 0.040 | +0.072 [-0.005, +0.147] thr +0.113, p 0.215 |
| turn ahead (BA - majority) | L2 k3 | +0.133 [+0.066, +0.201]* thr +0.202, p 0.305 | +0.155 [+0.055, +0.249]* thr +0.206, p 0.115 |
| turn ahead (BA - majority) | L3 k3 | +0.332 [+0.245, +0.416]† thr +0.227, p 0.005 | +0.312 [+0.192, +0.415]† thr +0.225, p 0.005 |
| beside: closing speed, no 20 m/s cap (R2) | L1 k10 | +0.060 [+0.012, +0.120]* thr +0.087, p 0.195 | +0.056 [+0.018, +0.108]* thr +0.085, p 0.220 |
| beside: closing speed, no 20 m/s cap (R2) | L2 k3 | +0.099 [+0.028, +0.188]* thr +0.161, p 0.410 | +0.091 [+0.024, +0.181]* thr +0.157, p 0.495 |
| beside: closing speed, no 20 m/s cap (R2) | L3 k3 | +0.190 [+0.057, +0.351]* thr +0.281, p 0.260 | +0.160 [+0.048, +0.312]* thr +0.244, p 0.490 |
| beside: agents within 30 m in the frustum (R2) | L1 k10 | +0.079 [+0.038, +0.124]* thr +0.109, p 0.185 | +0.082 [+0.015, +0.154]* thr +0.117, p 0.245 |
| beside: agents within 30 m in the frustum (R2) | L2 k3 | +0.147 [+0.053, +0.243]* thr +0.210, p 0.415 | +0.061 [-0.047, +0.168] thr +0.204, p 0.960 |
| beside: agents within 30 m in the frustum (R2) | L3 k3 | +0.107 [-0.001, +0.207] thr +0.252, p 0.895 | +0.040 [-0.080, +0.187] thr +0.254, p 1.000 |

#### eval-141 (beside, post hoc)

| target | level, horizon | seed s0 | seed s1 |
|---|---|---|---|
| lead present (BA - majority) | L1 k10 | -0.023 [-0.108, +0.047] thr +0.071, p 0.860 | -0.032 [-0.148, +0.082] thr +0.086, p 0.945 |
| lead present (BA - majority) | L2 k3 | +0.130 [+0.050, +0.215]† thr +0.114, p 0.020 | +0.014 [-0.077, +0.101] thr +0.127, p 0.815 |
| lead present (BA - majority) | L3 k3 | +0.079 [-0.045, +0.195] thr +0.186, p 0.595 | -0.016 [-0.182, +0.123] thr +0.187, p 0.990 |
| lead gap (R2) | L1 k10 | -0.017 [-0.140, +0.079] thr +0.142, p 0.975 | -0.018 [-0.118, +0.079] thr +0.134, p 0.930 |
| lead gap (R2) | L2 k3 | +0.090 [+0.012, +0.148]* thr +0.211, p 0.715 | +0.029 [-0.117, +0.140] thr +0.212, p 0.945 |
| lead gap (R2) | L3 k3 | +0.076 [-0.069, +0.216] thr +0.275, p 0.770 | +0.125 [-0.072, +0.255] thr +0.303, p 0.755 |
| lead closing speed (R2) | L1 k10 | +0.066 [-0.004, +0.132] thr +0.137, p 0.450 | +0.021 [-0.043, +0.077] thr +0.111, p 0.570 |
| lead closing speed (R2) | L2 k3 | +0.137 [+0.074, +0.198]* thr +0.213, p 0.415 | +0.118 [+0.038, +0.190]* thr +0.203, p 0.485 |
| lead closing speed (R2) | L3 k3 | +0.072 [-0.044, +0.175] thr +0.271, p 0.845 | +0.077 [-0.015, +0.164] thr +0.327, p 0.870 |
| agents within 30 m (R2) | L1 k10 | +0.037 [-0.059, +0.109] thr +0.153, p 0.880 | +0.099 [+0.017, +0.212]* thr +0.144, p 0.350 |
| agents within 30 m (R2) | L2 k3 | +0.144 [-0.013, +0.308] thr +0.269, p 0.625 | +0.090 [-0.016, +0.203] thr +0.234, p 0.925 |
| agents within 30 m (R2) | L3 k3 | +0.233 [+0.082, +0.405]* thr +0.315, p 0.210 | +0.113 [-0.072, +0.305] thr +0.339, p 0.970 |
| nearest-agent distance (R2) | L1 k10 | +0.043 [-0.002, +0.085] thr +0.159, p 0.905 | -0.003 [-0.052, +0.044] thr +0.156, p 0.990 |
| nearest-agent distance (R2) | L2 k3 | +0.029 [-0.044, +0.104] thr +0.267, p 1.000 | +0.138 [+0.061, +0.217]* thr +0.259, p 0.815 |
| nearest-agent distance (R2) | L3 k3 | +0.090 [-0.045, +0.228] thr +0.319, p 0.990 | +0.153 [+0.059, +0.251]* thr +0.343, p 0.875 |
| 7-class longitudinal (BA - majority) | L1 k10 | +0.021 [-0.004, +0.045] thr +0.028, p 0.110 | +0.011 [-0.016, +0.041] thr +0.038, p 0.390 |
| 7-class longitudinal (BA - majority) | L2 k3 | +0.035 [+0.001, +0.067]* thr +0.040, p 0.050 | +0.018 [-0.015, +0.051] thr +0.046, p 0.365 |
| 7-class longitudinal (BA - majority) | L3 k3 | +0.023 [-0.026, +0.075] thr +0.052, p 0.400 | +0.072 [+0.028, +0.119]† thr +0.070, p 0.025 |
| turn ahead (BA - majority) | L1 k10 | +0.015 [-0.038, +0.061] thr +0.120, p 0.765 | +0.068 [-0.007, +0.142] thr +0.125, p 0.300 |
| turn ahead (BA - majority) | L2 k3 | +0.050 [-0.027, +0.120] thr +0.225, p 0.945 | +0.078 [-0.008, +0.165] thr +0.199, p 0.765 |
| turn ahead (BA - majority) | L3 k3 | +0.187 [+0.099, +0.288]* thr +0.263, p 0.215 | +0.105 [-0.049, +0.224] thr +0.237, p 0.700 |
| beside: closing speed, no 20 m/s cap (R2) | L1 k10 | +0.062 [+0.013, +0.166]* thr +0.139, p 0.560 | +0.029 [-0.047, +0.132] thr +0.139, p 0.630 |
| beside: closing speed, no 20 m/s cap (R2) | L2 k3 | +0.171 [+0.069, +0.311]* thr +0.247, p 0.345 | +0.111 [+0.023, +0.247]* thr +0.291, p 0.615 |
| beside: closing speed, no 20 m/s cap (R2) | L3 k3 | +0.086 [+0.023, +0.197]* thr +0.318, p 0.670 | +0.073 [+0.030, +0.124]* thr +0.364, p 0.870 |
| beside: agents within 30 m in the frustum (R2) | L1 k10 | +0.082 [+0.020, +0.199]* thr +0.142, p 0.430 | +0.098 [+0.016, +0.238]* thr +0.136, p 0.265 |
| beside: agents within 30 m in the frustum (R2) | L2 k3 | +0.152 [-0.002, +0.336] thr +0.258, p 0.535 | +0.103 [-0.023, +0.250] thr +0.243, p 0.885 |
| beside: agents within 30 m in the frustum (R2) | L3 k3 | +0.152 [-0.006, +0.371] thr +0.311, p 0.690 | +0.094 [-0.068, +0.300] thr +0.315, p 0.945 |

## S3-PARTIAL and raw S3 (does the imagined lead gap follow the action; accelerate minus brake, decoded, windows with a lead)

S3-PARTIAL decodes the lead gap after a linear regression on ego speed; claimed side is below 0 (accelerating should shrink the gap). Raw S3 is run 1's quantity (gap probe on the unresidualised gap), reported beside with the same frame-permuted reference.

| dataset | level | seed | S3-PARTIAL [95 % interval] | 2.5th pct of reference, rank p | separated | raw S3 [95 % interval] | 2.5th pct of reference, rank p | identical sequences (partial / raw) |
|---|---|---|---|---|---|---|---|---|
| TRAIN | L1 | s0 | +0.72 [+0.12, +1.37] m | -0.57, 0.995 | no | +1.85 [+1.25, +2.47] m | -0.58, 1.000 | 0.0 / 0.0 |
| TRAIN | L1 | s1 | +1.88 [+1.34, +2.40] m | -0.63, 1.000 | no | +3.19 [+2.71, +3.68] m | -0.68, 1.000 | 0.0 / 0.0 |
| TRAIN | L2 | s0 | +1.21 [+0.77, +1.67] m | -0.75, 1.000 | no | +2.46 [+2.02, +2.88] m | -0.76, 1.000 | 0.0 / 0.0 |
| TRAIN | L2 | s1 | +1.22 [+0.81, +1.67] m | -0.81, 1.000 | no | +2.16 [+1.72, +2.61] m | -0.82, 1.000 | 0.0 / 0.0 |
| TRAIN | L3 | s0 | +3.30 [+2.33, +4.26] m | -0.98, 1.000 | no | +6.66 [+5.71, +7.57] m | -1.03, 1.000 | 0.0 / 0.0 |
| TRAIN | L3 | s1 | +2.46 [+1.80, +3.09] m | -0.94, 1.000 | no | +4.84 [+4.21, +5.45] m | -0.88, 1.000 | 0.0 / 0.0 |
| eval-141 | L1 | s0 | +1.65 [+1.34, +1.97] m | -0.52, 1.000 | no | +2.42 [+2.11, +2.73] m | -0.53, 1.000 | 0.0 / 0.0 |
| eval-141 | L1 | s1 | +1.80 [+1.33, +2.25] m | -0.79, 1.000 | no | +2.92 [+2.46, +3.37] m | -0.78, 1.000 | 0.0 / 0.0 |
| eval-141 | L2 | s0 | +1.85 [+1.47, +2.21] m | -0.98, 1.000 | no | +2.90 [+2.49, +3.28] m | -0.97, 1.000 | 0.0 / 0.0 |
| eval-141 | L2 | s1 | +1.90 [+1.47, +2.31] m | -0.87, 1.000 | no | +2.78 [+2.26, +3.29] m | -0.82, 1.000 | 0.0 / 0.0 |
| eval-141 | L3 | s0 | +5.21 [+4.32, +6.04] m | -1.66, 1.000 | no | +7.64 [+6.58, +8.61] m | -1.71, 1.000 | 0.0 / 0.0 |
| eval-141 | L3 | s1 | +1.32 [+0.47, +2.12] m | -1.06, 0.995 | no | +2.60 [+1.68, +3.39] m | -1.04, 1.000 | 0.0 / 0.0 |

Decoded mean gap under accelerate / brake (m, S3 windows; raw probe | residual probe):

| dataset | level | seed | raw probe: accelerate / brake | residual probe: accelerate / brake |
|---|---|---|---|---|
| TRAIN | L1 | s0 | +19.48 / +17.63 | +1.06 / +0.34 |
| TRAIN | L1 | s1 | +19.08 / +15.88 | +0.79 / -1.09 |
| TRAIN | L2 | s0 | +22.84 / +20.39 | +3.72 / +2.51 |
| TRAIN | L2 | s1 | +21.16 / +19.00 | +1.78 / +0.56 |
| TRAIN | L3 | s0 | +24.19 / +17.53 | +3.86 / +0.56 |
| TRAIN | L3 | s1 | +21.84 / +17.00 | +2.56 / +0.09 |
| eval-141 | L1 | s0 | +22.85 / +20.43 | +3.95 / +2.29 |
| eval-141 | L1 | s1 | +22.82 / +19.89 | +2.60 / +0.80 |
| eval-141 | L2 | s0 | +21.78 / +18.88 | +2.93 / +1.08 |
| eval-141 | L2 | s1 | +22.64 / +19.86 | +2.98 / +1.09 |
| eval-141 | L3 | s0 | +24.31 / +16.66 | +4.53 / -0.68 |
| eval-141 | L3 | s1 | +21.74 / +19.14 | +1.92 / +0.59 |

## Run 1 beside (LATSEM2 `RESULT_LATSEM2.json`: eval-141, bootstrap interval alone, INVALID read; shown for comparison, decides nothing)

| quantity | seed | run 1 (eval-141, one-part rule) | this run, eval-141 (reproduced point, reference thr 97.5 / rank p) |
|---|---|---|---|
| S1 lead_gap L1 R2 | s0 | +0.245 [-0.004, +0.456] | +0.245 [-0.004, +0.456], thr -0.342, p 0.005 |
| S1 lead_gap L1 R2 | s1 | +0.033 [-0.325, +0.313] | +0.033 [-0.325, +0.313], thr -0.390, p 0.005 |
| S1 lead_gap L2 R2 | s0 | -0.467 [-1.228, +0.049] | -0.467 [-1.228, +0.049], thr -0.292, p 0.135 |
| S1 lead_gap L2 R2 | s1 | -0.050 [-0.441, +0.258] | -0.050 [-0.441, +0.258], thr -0.350, p 0.005 |
| S1 lead_gap L3 R2 | s0 | -0.056 [-0.472, +0.255] | -0.056 [-0.472, +0.255], thr -0.214, p 0.010 |
| S1 lead_gap L3 R2 | s1 | -0.119 [-0.551, +0.243] | -0.119 [-0.551, +0.243], thr -0.301, p 0.005 |
| S1 agents30 L1 R2 | s0 | +0.308 [+0.036, +0.429]* | +0.308 [+0.036, +0.429], thr -0.297, p 0.005 |
| S1 agents30 L1 R2 | s1 | +0.231 [-0.162, +0.391] | +0.231 [-0.162, +0.391], thr -0.277, p 0.005 |
| S1 agents30 L2 R2 | s0 | -0.246 [-1.268, +0.202] | -0.246 [-1.268, +0.202], thr -0.448, p 0.005 |
| S1 agents30 L2 R2 | s1 | +0.032 [-0.676, +0.317] | +0.032 [-0.676, +0.317], thr -0.398, p 0.005 |
| S1 agents30 L3 R2 | s0 | -0.230 [-1.021, +0.092] | -0.230 [-1.021, +0.092], thr -0.277, p 0.015 |
| S1 agents30 L3 R2 | s1 | +0.003 [-0.830, +0.319] | +0.003 [-0.830, +0.319], thr -0.328, p 0.005 |
| S2 lead gap L1 k10 SKILL | s0 | -0.017 [-0.140, +0.079] | -0.017 [-0.140, +0.079], thr +0.142, p 0.975 |
| S2 lead gap L1 k10 SKILL | s1 | -0.018 [-0.118, +0.079] | -0.018 [-0.118, +0.079], thr +0.134, p 0.930 |
| S2 lead gap L2 k3 SKILL | s0 | +0.090 [+0.012, +0.148]* | +0.090 [+0.012, +0.148], thr +0.211, p 0.715 |
| S2 lead gap L2 k3 SKILL | s1 | +0.029 [-0.117, +0.140] | +0.029 [-0.117, +0.140], thr +0.212, p 0.945 |
| S2 lead gap L3 k3 SKILL | s0 | +0.076 [-0.069, +0.216] | +0.076 [-0.069, +0.216], thr +0.275, p 0.770 |
| S2 lead gap L3 k3 SKILL | s1 | +0.125 [-0.072, +0.255] | +0.125 [-0.072, +0.255], thr +0.303, p 0.755 |
| S3 raw, accelerate minus brake L1 (m) | s0 | +2.42 [+2.11, +2.73]* | +2.42 [+2.11, +2.73], 2.5th pct -0.53, p 1.000; S3-PARTIAL +1.65 [+1.34, +1.97] |
| S3 raw, accelerate minus brake L1 (m) | s1 | +2.92 [+2.46, +3.37]* | +2.92 [+2.46, +3.37], 2.5th pct -0.78, p 1.000; S3-PARTIAL +1.80 [+1.33, +2.25] |
| S3 raw, accelerate minus brake L2 (m) | s0 | +2.90 [+2.49, +3.28]* | +2.90 [+2.49, +3.28], 2.5th pct -0.97, p 1.000; S3-PARTIAL +1.85 [+1.47, +2.21] |
| S3 raw, accelerate minus brake L2 (m) | s1 | +2.78 [+2.26, +3.29]* | +2.78 [+2.26, +3.29], 2.5th pct -0.82, p 1.000; S3-PARTIAL +1.90 [+1.47, +2.31] |
| S3 raw, accelerate minus brake L3 (m) | s0 | +7.64 [+6.58, +8.61]* | +7.64 [+6.58, +8.61], 2.5th pct -1.71, p 1.000; S3-PARTIAL +5.21 [+4.32, +6.04] |
| S3 raw, accelerate minus brake L3 (m) | s1 | +2.60 [+1.68, +3.39]* | +2.60 [+1.68, +3.39], 2.5th pct -1.04, p 1.000; S3-PARTIAL +1.32 [+0.47, +2.12] |

## What the read does not say

* **It is a probe of the latent, not of the planner.** T0, a world-model diagnostic: nothing here shows what the driving stack uses. The Master Mind's next lever follows the card, not this paragraph.
* **The deciding set is M2b's own training clips.** TRAIN skill and encoding are optimistic for what the model learned on those clips; the eval-141 rows are the held-out view, post hoc, and the card asks for the same sign there. The L1 shrink factor was fitted on these TRAIN windows (in-sample on the deciding set).
* **A permutation reference removes the probe-direction effect, not every confound.** It tests the relation between latent and label across episodes; it cannot say which scene variable a probe reads when two correlate (lead gap and ego speed correlate at r +0.45 on TRAIN, +0.52 on eval-141; S3-PARTIAL removes the linear speed term, nothing nonlinear).
* **The two model seeds are two training runs of one recipe.** Agreement at both seeds is the committed bar; it is not a population estimate over recipes.
* **Label coverage differs by target.** Lead-gap rows rest on 59 TRAIN episodes (29 / 30 in the folds); the permutation reference is bounded by that, and 7 TRAIN clips carry no label but `turn`. n is in each table.
* **The label alignment on TRAIN peaked at a one-raw-frame offset** (corridor distance against lead gap, r 0.682 at -1 against 0.652 at 0). Nothing was shifted; `agents30` is insensitive to a 0.1 s offset, the lead gap is not exempt.
* **An INVALID read decides nothing**, and a PASS decides only the card's outcomes; neither says the model drives well or badly.

## Pictures (no camera frame inside; decide nothing)

Per-level PCA maps of the bank frames (model seed s0), coloured by lead gap (grey: no lead within 40 m or unlabelled).

| file | sha256 |
|---|---|
| `fig/pca_levels_gap_train.png` | `41dfb56f9e171bab79d142cd507ec7c9f944c9c4a081046512a1b622e4259fd7` |
| `fig/pca_levels_gap_eval141.png` | `56db7a9ba964dd3faae4575e116de3543519fcf6df79ff46e9f6a6d7b95c75b7` |

## Files (`theory/latsem2b/`, unstaged on E:)

`IMPL_CHOICES_LATSEM2B.md` (+ `.sha256`), `run_latsem2b.py` (sha256 `1fca3df36a73`), `report_latsem2b.py`, `test_synth_latsem2b.py` (synthetic smoke test of the code paths), `RESULT_LATSEM2B.json`, `RESULT_LATSEM2B.md`, `draws_train.jsonl`, `draws_eval141.jsonl`, `run_latsem2b.run1.log`, `fig/`, labels `labels_latsem2b.py`, `labels_latsem2b_train.*`.
