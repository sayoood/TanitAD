# TAC_step48852ema_pf -- TACTICAL read of the RE-TIMED plan (SHIP-R8 clause 5, scored arm H2CNL_MUT8)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap, B 2000, bootstrap seed 0; models and gains FIXED ('another draw of EPISODES' only; inference seeds 0 and 1 are the replicate).
gains: deploy sha256 `2564bef712e83d14e2529f9da90ae81ced1d0034e3bf620acb924923711f6434` (from the frozen TRAIN gains sha256 `2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80`); script sha256 `e3e49cdb809374aa3bfebeced470cfe4cbe70f183373f0a0ae8c69b8d52ab3d7`.

**CLAUSE 5: TAC_traj_lat_correct and TAC_traj_lon_correct of the RE-TIMED plan (RETIME_DEPLOY8, AMENDMENT 2) not separated worse than refcv7-50,400 at both inference seeds, on ceiling-compliant windows, on every surface the battery defines them (S2, S6).  ACCURACY metric: separated worse <=> the paired-bootstrap CI upper bound < 0.**

## clause 5 on this tree: **PASS**  (separated-worse cells: none; missing: none)

Compliant windows. Clause-5 cell = H2CNL_MUT8 - refcv7-50,400 (accuracy, higher is better; worse+separated = CI upper < 0).

| surface | row | seed | n win / eps | os | H2CNL_MUT8 | per-surface DEPLOY | TRAINFIT | refcv7-50,400 | H2CNL_MUT8 - refcv7 (delta [lo, hi]) | separated worse |
|---|---|---|---|---|---|---|---|---|---|---|
| S2 | TAC_traj_lat_correct | 0 | 4159 / 138 | 0.9627 | 0.9630 | 0.9637 | 0.9637 | 0.9601 | +0.0029 [-0.0014, +0.0075] | False |
| S2 | TAC_traj_lat_correct | 1 | 4159 / 138 | 0.9630 | 0.9637 | 0.9644 | 0.9644 | 0.9601 | +0.0036 [-0.0005, +0.0079] | False |
| S2 | TAC_traj_lon_correct | 0 | 4159 / 138 | 0.8286 | 0.8755 | 0.8541 | 0.8541 | 0.8348 | +0.0406 [+0.0249, +0.0584] SEP | False |
| S2 | TAC_traj_lon_correct | 1 | 4159 / 138 | 0.8269 | 0.8755 | 0.8557 | 0.8557 | 0.8413 | +0.0341 [+0.0198, +0.0507] SEP | False |
| S6 | TAC_traj_lat_correct | 0 | 3251 / 138 | 0.8896 | 0.8893 | 0.8896 | 0.8896 | 0.8803 | +0.0089 [-0.0027, +0.0204] | False |
| S6 | TAC_traj_lat_correct | 1 | 3251 / 138 | 0.8902 | 0.8923 | 0.8911 | 0.8911 | 0.8816 | +0.0108 [-0.0003, +0.0218] | False |
| S6 | TAC_traj_lon_correct | 0 | 3251 / 138 | 0.5930 | 0.6235 | 0.5958 | 0.5952 | 0.5998 | +0.0237 [-0.0085, +0.0553] | False |
| S6 | TAC_traj_lon_correct | 1 | 3251 / 138 | 0.5894 | 0.6238 | 0.5915 | 0.5918 | 0.5983 | +0.0255 [-0.0072, +0.0565] | False |

## beside (outside clause 5): the per-surface form of Amendment 1 on the same cells reads **PASS** (separated worse: none)

| surface | row | seed | H2CNL_MUT8 - os | per-surface DEPLOY - os | TRAINFIT - os | os - refcv7 | per-surface DEPLOY - refcv7 | label changed vs os, H2CNL_MUT8 (lat / lon) |
|---|---|---|---|---|---|---|---|---|
| S2 | TAC_traj_lat_correct | 0 | +0.0002 [-0.0017, +0.0024] | +0.0010 [+0.0002, +0.0020] SEP | +0.0010 [+0.0002, +0.0020] SEP | +0.0026 [-0.0019, +0.0074] | +0.0036 [-0.0007, +0.0083] | 0.0050 / 0.1399 |
| S2 | TAC_traj_lat_correct | 1 | +0.0007 [-0.0014, +0.0030] | +0.0014 [+0.0002, +0.0031] SEP | +0.0014 [+0.0002, +0.0031] SEP | +0.0029 [-0.0015, +0.0071] | +0.0043 [+0.0000, +0.0086] | 0.0055 / 0.1457 |
| S2 | TAC_traj_lon_correct | 0 | +0.0469 [+0.0280, +0.0682] SEP | +0.0255 [+0.0156, +0.0360] SEP | +0.0255 [+0.0156, +0.0360] SEP | -0.0063 [-0.0229, +0.0095] | +0.0192 [+0.0039, +0.0344] SEP | 0.0050 / 0.1399 |
| S2 | TAC_traj_lon_correct | 1 | +0.0486 [+0.0293, +0.0705] SEP | +0.0289 [+0.0168, +0.0418] SEP | +0.0289 [+0.0168, +0.0418] SEP | -0.0144 [-0.0306, +0.0005] | +0.0144 [+0.0007, +0.0282] SEP | 0.0055 / 0.1457 |
| S6 | TAC_traj_lat_correct | 0 | -0.0003 [-0.0043, +0.0034] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0092 [-0.0028, +0.0212] | +0.0092 [-0.0028, +0.0212] | 0.0108 / 0.3482 |
| S6 | TAC_traj_lat_correct | 1 | +0.0022 [-0.0018, +0.0060] | +0.0009 [-0.0006, +0.0025] | +0.0009 [-0.0006, +0.0025] | +0.0086 [-0.0028, +0.0203] | +0.0095 [-0.0022, +0.0214] | 0.0108 / 0.3420 |
| S6 | TAC_traj_lon_correct | 0 | +0.0305 [-0.0092, +0.0714] | +0.0028 [-0.0034, +0.0089] | +0.0022 [-0.0037, +0.0080] | -0.0068 [-0.0330, +0.0168] | -0.0040 [-0.0291, +0.0185] | 0.0108 / 0.3482 |
| S6 | TAC_traj_lon_correct | 1 | +0.0345 [-0.0056, +0.0754] | +0.0022 [-0.0031, +0.0077] | +0.0025 [-0.0028, +0.0079] | -0.0089 [-0.0344, +0.0157] | -0.0068 [-0.0308, +0.0168] | 0.0108 / 0.3420 |

### informational: 8-slot call vs per-surface call, Euclidean point distance (m), compliant / all windows

* S2|s0|compliant: mean 0.11651, max 2.3502 (n 4159)
* S2|s0|all: mean 0.12374, max 2.5807 (n 4754)
* S6|s0|compliant: mean 1.59768, max 54.1747 (n 3251)
* S6|s0|all: mean 1.64324, max 54.1747 (n 3668)
* S2|s1|compliant: mean 0.11626, max 2.3901 (n 4159)
* S2|s1|all: mean 0.12414, max 2.7246 (n 4754)
* S6|s1|compliant: mean 1.59284, max 47.0961 (n 3251)
* S6|s1|all: mean 1.64609, max 47.0961 (n 3668)

## controls (ALL_PASS = **True**)

* K1 unmodified os reproduces the banked PAIRED_READ (all windows, exact): **True**
* K2 unmodified os - refcv7 reproduces the banked compliant-tree cell: **True**
* K3 GT vs GT reads exactly 1.0 (analytic): **True**
* K4 CONST_G1 == os (bit-equal plans, zero cell): **True**
* K5 directly recomputed labels == refav1_arm._components TAC arrays: **True**
* K6 cross-check vs FORWARD_DEPLOY (per-surface) NR cells (determinism only, not correctness): **True**
* K7 cross-check vs FORWARD_DEPLOY8 NR cells for H2CNL_MUT8 (determinism only, not correctness): **True**

### K1 literals (os, all windows: mine vs the banked PAIRED_READ `means`)

| surface | seed | row | os mine | os banked | refcv7 mine | refcv7 banked | n | exact |
|---|---|---|---|---|---|---|---|---|
| S2 | s0 | TAC_traj_lat_correct | 0.9621371626853943 | 0.9621371626853943 | 0.9617164731025696 | 0.9617164731025696 | 4754 | True |
| S2 | s0 | TAC_traj_lon_correct | 0.8174169063568115 | 0.8174169063568115 | 0.8304585814476013 | 0.8304585814476013 | 4754 | True |
| S6 | s0 | TAC_traj_lat_correct | 0.884678304195404 | 0.884678304195404 | 0.8770447373390198 | 0.8770447373390198 | 3668 | True |
| S6 | s0 | TAC_traj_lon_correct | 0.6033260822296143 | 0.6033260822296143 | 0.6038713455200195 | 0.6038713455200195 | 3668 | True |
| S2 | s1 | TAC_traj_lat_correct | 0.962557852268219 | 0.962557852268219 | 0.9621371626853943 | 0.9621371626853943 | 4754 | True |
| S2 | s1 | TAC_traj_lon_correct | 0.8161548376083374 | 0.8161548376083374 | 0.8357173204421997 | 0.8357173204421997 | 4754 | True |
| S6 | s1 | TAC_traj_lat_correct | 0.8830425143241882 | 0.8830425143241882 | 0.8767721056938171 | 0.8767721056938171 | 3668 | True |
| S6 | s1 | TAC_traj_lon_correct | 0.5967829823493958 | 0.5967829823493958 | 0.6008723974227905 | 0.6008723974227905 | 3668 | True |

Single training seed; T1 open loop; EMA-30k reads are ORIENTATION ONLY unless the tree is step48852ema_unc.
