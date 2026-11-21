# TAC_stepwise050_pf -- TACTICAL read of the RE-TIMED plan (SHIP-R8 clause 5, scored arm RETIME_DEPLOY8)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap, B 2000, bootstrap seed 0; models and gains FIXED ('another draw of EPISODES' only; inference seeds 0 and 1 are the replicate).
gains: deploy sha256 `829562a947b57d7ea9a7ad7f8f00b8407c12329476cac45b4745c17bef65a02b` (from the frozen TRAIN gains sha256 `2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80`); script sha256 `e3e49cdb809374aa3bfebeced470cfe4cbe70f183373f0a0ae8c69b8d52ab3d7`.

**CLAUSE 5: TAC_traj_lat_correct and TAC_traj_lon_correct of the RE-TIMED plan (RETIME_DEPLOY8, AMENDMENT 2) not separated worse than refcv7-50,400 at both inference seeds, on ceiling-compliant windows, on every surface the battery defines them (S2, S6).  ACCURACY metric: separated worse <=> the paired-bootstrap CI upper bound < 0.**

## clause 5 on this tree: **PASS**  (separated-worse cells: none; missing: none)

Compliant windows. Clause-5 cell = RETIME_DEPLOY8 - refcv7-50,400 (accuracy, higher is better; worse+separated = CI upper < 0).

| surface | row | seed | n win / eps | os | RETIME_DEPLOY8 | per-surface DEPLOY | TRAINFIT | refcv7-50,400 | RETIME_DEPLOY8 - refcv7 (delta [lo, hi]) | separated worse |
|---|---|---|---|---|---|---|---|---|---|---|
| S2 | TAC_traj_lat_correct | 0 | 4159 / 138 | 0.9620 | 0.9620 | 0.9625 | 0.9625 | 0.9601 | +0.0019 [-0.0022, +0.0063] | False |
| S2 | TAC_traj_lat_correct | 1 | 4159 / 138 | 0.9615 | 0.9620 | 0.9623 | 0.9623 | 0.9601 | +0.0019 [-0.0024, +0.0058] | False |
| S2 | TAC_traj_lon_correct | 0 | 4159 / 138 | 0.8375 | 0.8637 | 0.8637 | 0.8637 | 0.8348 | +0.0289 [+0.0158, +0.0426] SEP | False |
| S2 | TAC_traj_lon_correct | 1 | 4159 / 138 | 0.8343 | 0.8593 | 0.8593 | 0.8593 | 0.8413 | +0.0180 [+0.0054, +0.0312] SEP | False |
| S6 | TAC_traj_lat_correct | 0 | 3251 / 138 | 0.8791 | 0.8794 | 0.8797 | 0.8797 | 0.8803 | -0.0009 [-0.0103, +0.0079] | False |
| S6 | TAC_traj_lat_correct | 1 | 3251 / 138 | 0.8822 | 0.8819 | 0.8822 | 0.8822 | 0.8816 | +0.0003 [-0.0097, +0.0098] | False |
| S6 | TAC_traj_lon_correct | 0 | 3251 / 138 | 0.5764 | 0.5814 | 0.5814 | 0.5817 | 0.5998 | -0.0185 [-0.0480, +0.0088] | False |
| S6 | TAC_traj_lon_correct | 1 | 3251 / 138 | 0.5749 | 0.5795 | 0.5795 | 0.5801 | 0.5983 | -0.0188 [-0.0491, +0.0085] | False |

## beside (outside clause 5): the per-surface form of Amendment 1 on the same cells reads **PASS** (separated worse: none)

| surface | row | seed | RETIME_DEPLOY8 - os | per-surface DEPLOY - os | TRAINFIT - os | os - refcv7 | per-surface DEPLOY - refcv7 | label changed vs os, RETIME_DEPLOY8 (lat / lon) |
|---|---|---|---|---|---|---|---|---|
| S2 | TAC_traj_lat_correct | 0 | +0.0000 [-0.0017, +0.0017] | +0.0005 [-0.0007, +0.0017] | +0.0005 [-0.0007, +0.0017] | +0.0019 [-0.0022, +0.0064] | +0.0024 [-0.0017, +0.0068] | 0.0031 / 0.0726 |
| S2 | TAC_traj_lat_correct | 1 | +0.0005 [-0.0012, +0.0023] | +0.0007 [-0.0007, +0.0024] | +0.0007 [-0.0007, +0.0024] | +0.0014 [-0.0028, +0.0055] | +0.0022 [-0.0022, +0.0063] | 0.0029 / 0.0697 |
| S2 | TAC_traj_lon_correct | 0 | +0.0262 [+0.0152, +0.0385] SEP | +0.0262 [+0.0152, +0.0385] SEP | +0.0262 [+0.0152, +0.0385] SEP | +0.0026 [-0.0127, +0.0171] | +0.0289 [+0.0158, +0.0426] SEP | 0.0031 / 0.0726 |
| S2 | TAC_traj_lon_correct | 1 | +0.0250 [+0.0156, +0.0359] SEP | +0.0250 [+0.0156, +0.0359] SEP | +0.0250 [+0.0156, +0.0359] SEP | -0.0070 [-0.0216, +0.0067] | +0.0180 [+0.0054, +0.0312] SEP | 0.0029 / 0.0697 |
| S6 | TAC_traj_lat_correct | 0 | +0.0003 [-0.0006, +0.0015] | +0.0006 [+0.0000, +0.0015] | +0.0006 [+0.0000, +0.0015] | -0.0012 [-0.0107, +0.0076] | -0.0006 [-0.0100, +0.0083] | 0.0009 / 0.0289 |
| S6 | TAC_traj_lat_correct | 1 | -0.0003 [-0.0024, +0.0016] | +0.0000 [-0.0019, +0.0019] | +0.0000 [-0.0019, +0.0019] | +0.0006 [-0.0097, +0.0101] | +0.0006 [-0.0095, +0.0100] | 0.0022 / 0.0311 |
| S6 | TAC_traj_lon_correct | 0 | +0.0049 [-0.0007, +0.0107] | +0.0049 [-0.0007, +0.0107] | +0.0052 [-0.0003, +0.0109] | -0.0234 [-0.0531, +0.0055] | -0.0185 [-0.0480, +0.0088] | 0.0009 / 0.0289 |
| S6 | TAC_traj_lon_correct | 1 | +0.0046 [-0.0020, +0.0106] | +0.0046 [-0.0020, +0.0106] | +0.0052 [-0.0013, +0.0112] | -0.0234 [-0.0531, +0.0047] | -0.0188 [-0.0491, +0.0085] | 0.0022 / 0.0311 |

### informational: 8-slot call vs per-surface call, Euclidean point distance (m), compliant / all windows

* S2|s0|compliant: mean 0.00063, max 0.1987 (n 4159)
* S2|s0|all: mean 0.00084, max 0.3526 (n 4754)
* S6|s0|compliant: mean 0.00071, max 0.1222 (n 3251)
* S6|s0|all: mean 0.00073, max 0.1222 (n 3668)
* S2|s1|compliant: mean 0.00065, max 0.2115 (n 4159)
* S2|s1|all: mean 0.00082, max 0.3661 (n 4754)
* S6|s1|compliant: mean 0.00072, max 0.1003 (n 3251)
* S6|s1|all: mean 0.00074, max 0.1003 (n 3668)

## controls (ALL_PASS = **True**)

* K1 unmodified os reproduces the banked PAIRED_READ (all windows, exact): **True**
* K2 unmodified os - refcv7 reproduces the banked compliant-tree cell: **True**
* K3 GT vs GT reads exactly 1.0 (analytic): **True**
* K4 CONST_G1 == os (bit-equal plans, zero cell): **True**
* K5 directly recomputed labels == refav1_arm._components TAC arrays: **True**
* K6 cross-check vs FORWARD_DEPLOY (per-surface) NR cells (determinism only, not correctness): **True**
* K7 cross-check vs FORWARD_DEPLOY8 NR cells for RETIME_DEPLOY8 (determinism only, not correctness): **True**

### K1 literals (os, all windows: mine vs the banked PAIRED_READ `means`)

| surface | seed | row | os mine | os banked | refcv7 mine | refcv7 banked | n | exact |
|---|---|---|---|---|---|---|---|---|
| S2 | s0 | TAC_traj_lat_correct | 0.962557852268219 | 0.962557852268219 | 0.9617164731025696 | 0.9617164731025696 | 4754 | True |
| S2 | s0 | TAC_traj_lon_correct | 0.8199411034584045 | 0.8199411034584045 | 0.8304585814476013 | 0.8304585814476013 | 4754 | True |
| S6 | s0 | TAC_traj_lat_correct | 0.8735005259513855 | 0.8735005259513855 | 0.8770447373390198 | 0.8770447373390198 | 3668 | True |
| S6 | s0 | TAC_traj_lon_correct | 0.5858778357505798 | 0.5858778357505798 | 0.6038713455200195 | 0.6038713455200195 | 3668 | True |
| S2 | s1 | TAC_traj_lat_correct | 0.9623475074768066 | 0.9623475074768066 | 0.9621371626853943 | 0.9621371626853943 | 4754 | True |
| S2 | s1 | TAC_traj_lon_correct | 0.8169962167739868 | 0.8169962167739868 | 0.8357173204421997 | 0.8357173204421997 | 4754 | True |
| S6 | s1 | TAC_traj_lat_correct | 0.8751363158226013 | 0.8751363158226013 | 0.8767721056938171 | 0.8767721056938171 | 3668 | True |
| S6 | s1 | TAC_traj_lon_correct | 0.5842421054840088 | 0.5842421054840088 | 0.6008723974227905 | 0.6008723974227905 | 3668 | True |

Single training seed; T1 open loop; EMA-30k reads are ORIENTATION ONLY unless the tree is step48852ema_unc.
