# TAC_stepwplan050_pf -- TACTICAL read of the RE-TIMED plan (SHIP-R8 clause 5, scored arm RETIME_DEPLOY8)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap, B 2000, bootstrap seed 0; models and gains FIXED ('another draw of EPISODES' only; inference seeds 0 and 1 are the replicate).
gains: deploy sha256 `d79793d6447572839b29c4986370bc028a8df0e87a1813e03a2dd7a5fd00b125` (from the frozen TRAIN gains sha256 `2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80`); script sha256 `e3e49cdb809374aa3bfebeced470cfe4cbe70f183373f0a0ae8c69b8d52ab3d7`.

**CLAUSE 5: TAC_traj_lat_correct and TAC_traj_lon_correct of the RE-TIMED plan (RETIME_DEPLOY8, AMENDMENT 2) not separated worse than refcv7-50,400 at both inference seeds, on ceiling-compliant windows, on every surface the battery defines them (S2, S6).  ACCURACY metric: separated worse <=> the paired-bootstrap CI upper bound < 0.**

## clause 5 on this tree: **FAIL**  (separated-worse cells: ['S6|TAC_traj_lon_correct|s0', 'S6|TAC_traj_lon_correct|s1']; missing: none)

Compliant windows. Clause-5 cell = RETIME_DEPLOY8 - refcv7-50,400 (accuracy, higher is better; worse+separated = CI upper < 0).

| surface | row | seed | n win / eps | os | RETIME_DEPLOY8 | per-surface DEPLOY | TRAINFIT | refcv7-50,400 | RETIME_DEPLOY8 - refcv7 (delta [lo, hi]) | separated worse |
|---|---|---|---|---|---|---|---|---|---|---|
| S2 | TAC_traj_lat_correct | 0 | 4159 / 138 | 0.9632 | 0.9637 | 0.9637 | 0.9637 | 0.9601 | +0.0036 [-0.0005, +0.0082] | False |
| S2 | TAC_traj_lat_correct | 1 | 4159 / 138 | 0.9635 | 0.9639 | 0.9639 | 0.9639 | 0.9601 | +0.0038 [-0.0005, +0.0085] | False |
| S2 | TAC_traj_lon_correct | 0 | 4159 / 138 | 0.8173 | 0.8567 | 0.8569 | 0.8569 | 0.8348 | +0.0219 [+0.0063, +0.0381] SEP | False |
| S2 | TAC_traj_lon_correct | 1 | 4159 / 138 | 0.8113 | 0.8574 | 0.8574 | 0.8574 | 0.8413 | +0.0161 [+0.0007, +0.0314] SEP | False |
| S6 | TAC_traj_lat_correct | 0 | 3251 / 138 | 0.8819 | 0.8819 | 0.8822 | 0.8822 | 0.8803 | +0.0015 [-0.0083, +0.0110] | False |
| S6 | TAC_traj_lat_correct | 1 | 3251 / 138 | 0.8782 | 0.8779 | 0.8782 | 0.8782 | 0.8816 | -0.0037 [-0.0161, +0.0077] | False |
| S6 | TAC_traj_lon_correct | 0 | 3251 / 138 | 0.5543 | 0.5558 | 0.5558 | 0.5561 | 0.5998 | -0.0440 [-0.0782, -0.0126] SEP | True |
| S6 | TAC_traj_lon_correct | 1 | 3251 / 138 | 0.5564 | 0.5524 | 0.5528 | 0.5543 | 0.5983 | -0.0458 [-0.0798, -0.0149] SEP | True |

## beside (outside clause 5): the per-surface form of Amendment 1 on the same cells reads **FAIL** (separated worse: ['S6|TAC_traj_lon_correct|s0', 'S6|TAC_traj_lon_correct|s1'])

| surface | row | seed | RETIME_DEPLOY8 - os | per-surface DEPLOY - os | TRAINFIT - os | os - refcv7 | per-surface DEPLOY - refcv7 | label changed vs os, RETIME_DEPLOY8 (lat / lon) |
|---|---|---|---|---|---|---|---|---|
| S2 | TAC_traj_lat_correct | 0 | +0.0005 [-0.0017, +0.0028] | +0.0005 [-0.0005, +0.0018] | +0.0005 [-0.0005, +0.0018] | +0.0031 [-0.0012, +0.0076] | +0.0036 [-0.0007, +0.0081] | 0.0043 / 0.0906 |
| S2 | TAC_traj_lat_correct | 1 | +0.0005 [-0.0012, +0.0026] | +0.0005 [-0.0005, +0.0017] | +0.0005 [-0.0005, +0.0017] | +0.0034 [-0.0010, +0.0078] | +0.0038 [-0.0005, +0.0083] | 0.0029 / 0.1005 |
| S2 | TAC_traj_lon_correct | 0 | +0.0394 [+0.0262, +0.0546] SEP | +0.0397 [+0.0264, +0.0548] SEP | +0.0397 [+0.0264, +0.0548] SEP | -0.0176 [-0.0385, +0.0015] | +0.0221 [+0.0064, +0.0385] SEP | 0.0043 / 0.0906 |
| S2 | TAC_traj_lon_correct | 1 | +0.0462 [+0.0313, +0.0629] SEP | +0.0462 [+0.0313, +0.0629] SEP | +0.0462 [+0.0313, +0.0629] SEP | -0.0301 [-0.0504, -0.0108] SEP | +0.0161 [+0.0007, +0.0314] SEP | 0.0029 / 0.1005 |
| S6 | TAC_traj_lat_correct | 0 | +0.0000 [-0.0012, +0.0012] | +0.0003 [-0.0006, +0.0013] | +0.0003 [-0.0006, +0.0013] | +0.0015 [-0.0084, +0.0111] | +0.0018 [-0.0080, +0.0114] | 0.0012 / 0.0286 |
| S6 | TAC_traj_lat_correct | 1 | -0.0003 [-0.0013, +0.0006] | +0.0000 [-0.0009, +0.0009] | +0.0000 [-0.0009, +0.0009] | -0.0034 [-0.0155, +0.0077] | -0.0034 [-0.0158, +0.0080] | 0.0009 / 0.0246 |
| S6 | TAC_traj_lon_correct | 0 | +0.0015 [-0.0039, +0.0072] | +0.0015 [-0.0039, +0.0072] | +0.0018 [-0.0035, +0.0075] | -0.0455 [-0.0778, -0.0148] SEP | -0.0440 [-0.0782, -0.0126] SEP | 0.0012 / 0.0286 |
| S6 | TAC_traj_lon_correct | 1 | -0.0040 [-0.0091, +0.0012] | -0.0037 [-0.0089, +0.0015] | -0.0022 [-0.0071, +0.0028] | -0.0418 [-0.0748, -0.0113] SEP | -0.0455 [-0.0793, -0.0146] SEP | 0.0009 / 0.0246 |

### informational: 8-slot call vs per-surface call, Euclidean point distance (m), compliant / all windows

* S2|s0|compliant: mean 0.00128, max 0.2231 (n 4159)
* S2|s0|all: mean 0.00152, max 0.2821 (n 4754)
* S6|s0|compliant: mean 0.00072, max 0.1067 (n 3251)
* S6|s0|all: mean 0.00075, max 0.1067 (n 3668)
* S2|s1|compliant: mean 0.00129, max 0.2363 (n 4159)
* S2|s1|all: mean 0.00153, max 0.2826 (n 4754)
* S6|s1|compliant: mean 0.00071, max 0.1027 (n 3251)
* S6|s1|all: mean 0.00074, max 0.1027 (n 3668)

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
| S2 | s0 | TAC_traj_lat_correct | 0.9610853791236877 | 0.9610853791236877 | 0.9617164731025696 | 0.9617164731025696 | 4754 | True |
| S2 | s0 | TAC_traj_lon_correct | 0.7999579310417175 | 0.7999579310417175 | 0.8304585814476013 | 0.8304585814476013 | 4754 | True |
| S6 | s0 | TAC_traj_lat_correct | 0.875408947467804 | 0.875408947467804 | 0.8770447373390198 | 0.8770447373390198 | 3668 | True |
| S6 | s0 | TAC_traj_lon_correct | 0.570065438747406 | 0.570065438747406 | 0.6038713455200195 | 0.6038713455200195 | 3668 | True |
| S2 | s1 | TAC_traj_lat_correct | 0.9621371626853943 | 0.9621371626853943 | 0.9621371626853943 | 0.9621371626853943 | 4754 | True |
| S2 | s1 | TAC_traj_lon_correct | 0.7944888472557068 | 0.7944888472557068 | 0.8357173204421997 | 0.8357173204421997 | 4754 | True |
| S6 | s1 | TAC_traj_lat_correct | 0.8710469007492065 | 0.8710469007492065 | 0.8767721056938171 | 0.8767721056938171 | 3668 | True |
| S6 | s1 | TAC_traj_lon_correct | 0.5706107020378113 | 0.5706107020378113 | 0.6008723974227905 | 0.6008723974227905 | 3668 | True |

Single training seed; T1 open loop; EMA-30k reads are ORIENTATION ONLY unless the tree is step48852ema_unc.
