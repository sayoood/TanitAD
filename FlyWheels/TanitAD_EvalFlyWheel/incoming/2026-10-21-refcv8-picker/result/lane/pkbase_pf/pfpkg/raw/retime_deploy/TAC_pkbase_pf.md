# TAC_pkbase_pf -- TACTICAL read of the RE-TIMED plan (SHIP-R8 clause 5, scored arm RETIME_DEPLOY8)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap, B 2000, bootstrap seed 0; models and gains FIXED ('another draw of EPISODES' only; inference seeds 0 and 1 are the replicate).
gains: deploy sha256 `6dbecee1647daa394902587a0ea9f8e59069ea7c418cc9345d3cda26f599d389` (from the frozen TRAIN gains sha256 `2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80`); script sha256 `e3e49cdb809374aa3bfebeced470cfe4cbe70f183373f0a0ae8c69b8d52ab3d7`.

**CLAUSE 5: TAC_traj_lat_correct and TAC_traj_lon_correct of the RE-TIMED plan (RETIME_DEPLOY8, AMENDMENT 2) not separated worse than refcv7-50,400 at both inference seeds, on ceiling-compliant windows, on every surface the battery defines them (S2, S6).  ACCURACY metric: separated worse <=> the paired-bootstrap CI upper bound < 0.**

## clause 5 on this tree: **PASS**  (separated-worse cells: none; missing: none)

Compliant windows. Clause-5 cell = RETIME_DEPLOY8 - refcv7-50,400 (accuracy, higher is better; worse+separated = CI upper < 0).

| surface | row | seed | n win / eps | os | RETIME_DEPLOY8 | per-surface DEPLOY | TRAINFIT | refcv7-50,400 | RETIME_DEPLOY8 - refcv7 (delta [lo, hi]) | separated worse |
|---|---|---|---|---|---|---|---|---|---|---|
| S2 | TAC_traj_lat_correct | 0 | 4159 / 138 | 0.9632 | 0.9644 | 0.9644 | 0.9644 | 0.9601 | +0.0043 [+0.0000, +0.0089] | False |
| S2 | TAC_traj_lat_correct | 1 | 4159 / 138 | 0.9630 | 0.9644 | 0.9644 | 0.9644 | 0.9601 | +0.0043 [+0.0002, +0.0086] SEP | False |
| S2 | TAC_traj_lon_correct | 0 | 4159 / 138 | 0.8483 | 0.8629 | 0.8629 | 0.8629 | 0.8348 | +0.0281 [+0.0142, +0.0428] SEP | False |
| S2 | TAC_traj_lon_correct | 1 | 4159 / 138 | 0.8461 | 0.8615 | 0.8615 | 0.8615 | 0.8413 | +0.0202 [+0.0070, +0.0342] SEP | False |
| S6 | TAC_traj_lat_correct | 0 | 3251 / 138 | 0.8896 | 0.8893 | 0.8893 | 0.8893 | 0.8803 | +0.0089 [-0.0025, +0.0207] | False |
| S6 | TAC_traj_lat_correct | 1 | 3251 / 138 | 0.8902 | 0.8908 | 0.8908 | 0.8908 | 0.8816 | +0.0092 [-0.0022, +0.0206] | False |
| S6 | TAC_traj_lon_correct | 0 | 3251 / 138 | 0.6072 | 0.6112 | 0.6112 | 0.6112 | 0.5998 | +0.0114 [-0.0123, +0.0337] | False |
| S6 | TAC_traj_lon_correct | 1 | 3251 / 138 | 0.6063 | 0.6097 | 0.6097 | 0.6100 | 0.5983 | +0.0114 [-0.0118, +0.0348] | False |

## beside (outside clause 5): the per-surface form of Amendment 1 on the same cells reads **PASS** (separated worse: none)

| surface | row | seed | RETIME_DEPLOY8 - os | per-surface DEPLOY - os | TRAINFIT - os | os - refcv7 | per-surface DEPLOY - refcv7 | label changed vs os, RETIME_DEPLOY8 (lat / lon) |
|---|---|---|---|---|---|---|---|---|
| S2 | TAC_traj_lat_correct | 0 | +0.0012 [-0.0002, +0.0031] | +0.0012 [-0.0002, +0.0031] | +0.0012 [-0.0002, +0.0031] | +0.0031 [-0.0010, +0.0074] | +0.0043 [+0.0000, +0.0089] | 0.0017 / 0.0584 |
| S2 | TAC_traj_lat_correct | 1 | +0.0014 [+0.0002, +0.0028] SEP | +0.0014 [+0.0002, +0.0028] SEP | +0.0014 [+0.0002, +0.0028] SEP | +0.0029 [-0.0010, +0.0069] | +0.0043 [+0.0002, +0.0086] SEP | 0.0019 / 0.0611 |
| S2 | TAC_traj_lon_correct | 0 | +0.0147 [+0.0060, +0.0238] SEP | +0.0147 [+0.0060, +0.0238] SEP | +0.0147 [+0.0060, +0.0238] SEP | +0.0135 [-0.0005, +0.0261] | +0.0281 [+0.0142, +0.0428] SEP | 0.0017 / 0.0584 |
| S2 | TAC_traj_lon_correct | 1 | +0.0154 [+0.0053, +0.0268] SEP | +0.0154 [+0.0053, +0.0268] SEP | +0.0154 [+0.0053, +0.0268] SEP | +0.0048 [-0.0095, +0.0180] | +0.0202 [+0.0070, +0.0342] SEP | 0.0019 / 0.0611 |
| S6 | TAC_traj_lat_correct | 0 | -0.0003 [-0.0009, +0.0000] | -0.0003 [-0.0009, +0.0000] | -0.0003 [-0.0009, +0.0000] | +0.0092 [-0.0021, +0.0209] | +0.0089 [-0.0025, +0.0207] | 0.0003 / 0.0317 |
| S6 | TAC_traj_lat_correct | 1 | +0.0006 [-0.0009, +0.0025] | +0.0006 [-0.0009, +0.0025] | +0.0006 [-0.0009, +0.0025] | +0.0086 [-0.0025, +0.0199] | +0.0092 [-0.0022, +0.0206] | 0.0018 / 0.0323 |
| S6 | TAC_traj_lon_correct | 0 | +0.0040 [-0.0027, +0.0106] | +0.0040 [-0.0027, +0.0106] | +0.0040 [-0.0025, +0.0106] | +0.0074 [-0.0176, +0.0312] | +0.0114 [-0.0123, +0.0337] | 0.0003 / 0.0317 |
| S6 | TAC_traj_lon_correct | 1 | +0.0034 [-0.0019, +0.0091] | +0.0034 [-0.0019, +0.0091] | +0.0037 [-0.0015, +0.0091] | +0.0080 [-0.0168, +0.0317] | +0.0114 [-0.0118, +0.0348] | 0.0018 / 0.0323 |

### informational: 8-slot call vs per-surface call, Euclidean point distance (m), compliant / all windows

* S2|s0|compliant: mean 0.00029, max 0.1351 (n 4159)
* S2|s0|all: mean 0.00041, max 0.1351 (n 4754)
* S6|s0|compliant: mean 0.00089, max 0.1228 (n 3251)
* S6|s0|all: mean 0.00088, max 0.1228 (n 3668)
* S2|s1|compliant: mean 0.00030, max 0.1489 (n 4159)
* S2|s1|all: mean 0.00042, max 0.1489 (n 4754)
* S6|s1|compliant: mean 0.00088, max 0.0983 (n 3251)
* S6|s1|all: mean 0.00087, max 0.0983 (n 3668)

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
| S2 | s0 | TAC_traj_lon_correct | 0.8363483548164368 | 0.8363483548164368 | 0.8304585814476013 | 0.8304585814476013 | 4754 | True |
| S6 | s0 | TAC_traj_lat_correct | 0.8835877776145935 | 0.8835877776145935 | 0.8770447373390198 | 0.8770447373390198 | 3668 | True |
| S6 | s0 | TAC_traj_lon_correct | 0.6150490641593933 | 0.6150490641593933 | 0.6038713455200195 | 0.6038713455200195 | 3668 | True |
| S2 | s1 | TAC_traj_lat_correct | 0.9623475074768066 | 0.9623475074768066 | 0.9621371626853943 | 0.9621371626853943 | 4754 | True |
| S2 | s1 | TAC_traj_lon_correct | 0.8338241577148438 | 0.8338241577148438 | 0.8357173204421997 | 0.8357173204421997 | 4754 | True |
| S6 | s1 | TAC_traj_lat_correct | 0.8822246193885803 | 0.8822246193885803 | 0.8767721056938171 | 0.8767721056938171 | 3668 | True |
| S6 | s1 | TAC_traj_lon_correct | 0.6136859059333801 | 0.6136859059333801 | 0.6008723974227905 | 0.6008723974227905 | 3668 | True |

Single training seed; T1 open loop; EMA-30k reads are ORIENTATION ONLY unless the tree is step48852ema_unc.
