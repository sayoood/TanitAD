# TAC_step40000ema_pf -- TACTICAL read of the RE-TIMED plan (SHIP-R8 clause 5, scored arm RETIME_DEPLOY8)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap, B 2000, bootstrap seed 0; models and gains FIXED ('another draw of EPISODES' only; inference seeds 0 and 1 are the replicate).
gains: deploy sha256 `04275c14965b2190be12b014c302101e17c263fc27138c9c0e7f44d8bf73af33` (from the frozen TRAIN gains sha256 `2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80`); script sha256 `e3e49cdb809374aa3bfebeced470cfe4cbe70f183373f0a0ae8c69b8d52ab3d7`.

**CLAUSE 5: TAC_traj_lat_correct and TAC_traj_lon_correct of the RE-TIMED plan (RETIME_DEPLOY8, AMENDMENT 2) not separated worse than refcv7-50,400 at both inference seeds, on ceiling-compliant windows, on every surface the battery defines them (S2, S6).  ACCURACY metric: separated worse <=> the paired-bootstrap CI upper bound < 0.**

## clause 5 on this tree: **PASS**  (separated-worse cells: none; missing: none)

Compliant windows. Clause-5 cell = RETIME_DEPLOY8 - refcv7-50,400 (accuracy, higher is better; worse+separated = CI upper < 0).

| surface | row | seed | n win / eps | os | RETIME_DEPLOY8 | per-surface DEPLOY | TRAINFIT | refcv7-50,400 | RETIME_DEPLOY8 - refcv7 (delta [lo, hi]) | separated worse |
|---|---|---|---|---|---|---|---|---|---|---|
| S2 | TAC_traj_lat_correct | 0 | 4159 / 138 | 0.9642 | 0.9649 | 0.9651 | 0.9651 | 0.9601 | +0.0048 [+0.0005, +0.0093] SEP | False |
| S2 | TAC_traj_lat_correct | 1 | 4159 / 138 | 0.9635 | 0.9644 | 0.9644 | 0.9644 | 0.9601 | +0.0043 [+0.0005, +0.0083] SEP | False |
| S2 | TAC_traj_lon_correct | 0 | 4159 / 138 | 0.8401 | 0.8593 | 0.8593 | 0.8593 | 0.8348 | +0.0245 [+0.0116, +0.0382] SEP | False |
| S2 | TAC_traj_lon_correct | 1 | 4159 / 138 | 0.8396 | 0.8569 | 0.8569 | 0.8569 | 0.8413 | +0.0156 [+0.0029, +0.0287] SEP | False |
| S6 | TAC_traj_lat_correct | 0 | 3251 / 138 | 0.8914 | 0.8926 | 0.8926 | 0.8926 | 0.8803 | +0.0123 [+0.0009, +0.0235] SEP | False |
| S6 | TAC_traj_lat_correct | 1 | 3251 / 138 | 0.8908 | 0.8917 | 0.8917 | 0.8917 | 0.8816 | +0.0102 [-0.0010, +0.0209] | False |
| S6 | TAC_traj_lon_correct | 0 | 3251 / 138 | 0.6057 | 0.6109 | 0.6109 | 0.6103 | 0.5998 | +0.0111 [-0.0118, +0.0327] | False |
| S6 | TAC_traj_lon_correct | 1 | 3251 / 138 | 0.6044 | 0.6029 | 0.6032 | 0.6035 | 0.5983 | +0.0046 [-0.0181, +0.0261] | False |

## beside (outside clause 5): the per-surface form of Amendment 1 on the same cells reads **PASS** (separated worse: none)

| surface | row | seed | RETIME_DEPLOY8 - os | per-surface DEPLOY - os | TRAINFIT - os | os - refcv7 | per-surface DEPLOY - refcv7 | label changed vs os, RETIME_DEPLOY8 (lat / lon) |
|---|---|---|---|---|---|---|---|---|
| S2 | TAC_traj_lat_correct | 0 | +0.0007 [-0.0002, +0.0019] | +0.0010 [+0.0002, +0.0020] SEP | +0.0010 [+0.0002, +0.0020] SEP | +0.0041 [-0.0002, +0.0086] | +0.0050 [+0.0007, +0.0096] SEP | 0.0012 / 0.0601 |
| S2 | TAC_traj_lat_correct | 1 | +0.0010 [+0.0000, +0.0022] | +0.0010 [+0.0000, +0.0022] | +0.0010 [+0.0000, +0.0022] | +0.0034 [-0.0007, +0.0075] | +0.0043 [+0.0005, +0.0083] SEP | 0.0014 / 0.0654 |
| S2 | TAC_traj_lon_correct | 0 | +0.0192 [+0.0108, +0.0287] SEP | +0.0192 [+0.0108, +0.0287] SEP | +0.0192 [+0.0108, +0.0287] SEP | +0.0053 [-0.0084, +0.0185] | +0.0245 [+0.0116, +0.0382] SEP | 0.0012 / 0.0601 |
| S2 | TAC_traj_lon_correct | 1 | +0.0173 [+0.0074, +0.0280] SEP | +0.0173 [+0.0074, +0.0280] SEP | +0.0173 [+0.0074, +0.0280] SEP | -0.0017 [-0.0153, +0.0121] | +0.0156 [+0.0029, +0.0287] SEP | 0.0014 / 0.0654 |
| S6 | TAC_traj_lat_correct | 0 | +0.0012 [+0.0003, +0.0025] SEP | +0.0012 [+0.0003, +0.0025] SEP | +0.0012 [+0.0003, +0.0025] SEP | +0.0111 [-0.0003, +0.0225] | +0.0123 [+0.0009, +0.0235] SEP | 0.0012 / 0.0317 |
| S6 | TAC_traj_lat_correct | 1 | +0.0009 [-0.0003, +0.0022] | +0.0009 [-0.0003, +0.0022] | +0.0009 [-0.0003, +0.0022] | +0.0092 [-0.0022, +0.0200] | +0.0102 [-0.0010, +0.0209] | 0.0015 / 0.0252 |
| S6 | TAC_traj_lon_correct | 0 | +0.0052 [-0.0009, +0.0116] | +0.0052 [-0.0009, +0.0116] | +0.0046 [-0.0021, +0.0111] | +0.0058 [-0.0169, +0.0275] | +0.0111 [-0.0118, +0.0327] | 0.0012 / 0.0317 |
| S6 | TAC_traj_lon_correct | 1 | -0.0015 [-0.0064, +0.0033] | -0.0012 [-0.0061, +0.0037] | -0.0009 [-0.0056, +0.0038] | +0.0062 [-0.0167, +0.0283] | +0.0049 [-0.0175, +0.0263] | 0.0015 / 0.0252 |

### informational: 8-slot call vs per-surface call, Euclidean point distance (m), compliant / all windows

* S2|s0|compliant: mean 0.00034, max 0.2658 (n 4159)
* S2|s0|all: mean 0.00044, max 0.2658 (n 4754)
* S6|s0|compliant: mean 0.00090, max 0.1171 (n 3251)
* S6|s0|all: mean 0.00089, max 0.1171 (n 3668)
* S2|s1|compliant: mean 0.00031, max 0.1242 (n 4159)
* S2|s1|all: mean 0.00042, max 0.1242 (n 4754)
* S6|s1|compliant: mean 0.00089, max 0.1203 (n 3251)
* S6|s1|all: mean 0.00089, max 0.1203 (n 3668)

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
| S2 | s0 | TAC_traj_lat_correct | 0.9642406105995178 | 0.9642406105995178 | 0.9617164731025696 | 0.9617164731025696 | 4754 | True |
| S2 | s0 | TAC_traj_lon_correct | 0.8287757635116577 | 0.8287757635116577 | 0.8304585814476013 | 0.8304585814476013 | 4754 | True |
| S6 | s0 | TAC_traj_lat_correct | 0.8874045610427856 | 0.8874045610427856 | 0.8770447373390198 | 0.8770447373390198 | 3668 | True |
| S6 | s0 | TAC_traj_lon_correct | 0.6117775440216064 | 0.6117775440216064 | 0.6038713455200195 | 0.6038713455200195 | 3668 | True |
| S2 | s1 | TAC_traj_lat_correct | 0.963188886642456 | 0.963188886642456 | 0.9621371626853943 | 0.9621371626853943 | 4754 | True |
| S2 | s1 | TAC_traj_lon_correct | 0.8289861083030701 | 0.8289861083030701 | 0.8357173204421997 | 0.8357173204421997 | 4754 | True |
| S6 | s1 | TAC_traj_lat_correct | 0.884678304195404 | 0.884678304195404 | 0.8767721056938171 | 0.8767721056938171 | 3668 | True |
| S6 | s1 | TAC_traj_lon_correct | 0.6090512275695801 | 0.6090512275695801 | 0.6008723974227905 | 0.6008723974227905 | 3668 | True |

Single training seed; T1 open loop; EMA-30k reads are ORIENTATION ONLY unless the tree is step48852ema_unc.
