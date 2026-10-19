# TAC_step30000ema_unc -- TACTICAL read of the RE-TIMED plan (SHIP-R8 clause 5, scored arm RETIME_DEPLOY8)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap, B 2000, bootstrap seed 0; models and gains FIXED ('another draw of EPISODES' only; inference seeds 0 and 1 are the replicate).
gains: deploy sha256 `244f313872004745b91a30104244ec9325b1da11bf633b9f31fb532af7075790` (from the frozen TRAIN gains sha256 `b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee`); script sha256 `e3e49cdb809374aa3bfebeced470cfe4cbe70f183373f0a0ae8c69b8d52ab3d7`.

**CLAUSE 5: TAC_traj_lat_correct and TAC_traj_lon_correct of the RE-TIMED plan (RETIME_DEPLOY8, AMENDMENT 2) not separated worse than refcv7-50,400 at both inference seeds, on ceiling-compliant windows, on every surface the battery defines them (S2, S6).  ACCURACY metric: separated worse <=> the paired-bootstrap CI upper bound < 0.**

## clause 5 on this tree: **PASS**  (separated-worse cells: none; missing: none)

Compliant windows. Clause-5 cell = RETIME_DEPLOY8 - refcv7-50,400 (accuracy, higher is better; worse+separated = CI upper < 0).

| surface | row | seed | n win / eps | os | RETIME_DEPLOY8 | per-surface DEPLOY | TRAINFIT | refcv7-50,400 | RETIME_DEPLOY8 - refcv7 (delta [lo, hi]) | separated worse |
|---|---|---|---|---|---|---|---|---|---|---|
| S2 | TAC_traj_lat_correct | 0 | 4159 / 138 | 0.9584 | 0.9570 | 0.9577 | 0.9577 | 0.9601 | -0.0031 [-0.0077, +0.0014] | False |
| S2 | TAC_traj_lat_correct | 1 | 4159 / 138 | 0.9579 | 0.9572 | 0.9577 | 0.9577 | 0.9601 | -0.0029 [-0.0078, +0.0019] | False |
| S2 | TAC_traj_lon_correct | 0 | 4159 / 138 | 0.8415 | 0.8649 | 0.8649 | 0.8649 | 0.8348 | +0.0301 [+0.0169, +0.0440] SEP | False |
| S2 | TAC_traj_lon_correct | 1 | 4159 / 138 | 0.8387 | 0.8627 | 0.8627 | 0.8627 | 0.8413 | +0.0214 [+0.0083, +0.0358] SEP | False |
| S6 | TAC_traj_lat_correct | 0 | 3251 / 138 | 0.8791 | 0.8810 | 0.8810 | 0.8810 | 0.8803 | +0.0006 [-0.0113, +0.0121] | False |
| S6 | TAC_traj_lat_correct | 1 | 3251 / 138 | 0.8816 | 0.8831 | 0.8834 | 0.8834 | 0.8816 | +0.0015 [-0.0105, +0.0132] | False |
| S6 | TAC_traj_lon_correct | 0 | 3251 / 138 | 0.5672 | 0.5783 | 0.5786 | 0.5783 | 0.5998 | -0.0215 [-0.0566, +0.0088] | False |
| S6 | TAC_traj_lon_correct | 1 | 3251 / 138 | 0.5697 | 0.5869 | 0.5869 | 0.5863 | 0.5983 | -0.0114 [-0.0467, +0.0205] | False |

## beside (outside clause 5): the per-surface form of Amendment 1 on the same cells reads **PASS** (separated worse: none)

| surface | row | seed | RETIME_DEPLOY8 - os | per-surface DEPLOY - os | TRAINFIT - os | os - refcv7 | per-surface DEPLOY - refcv7 | label changed vs os, RETIME_DEPLOY8 (lat / lon) |
|---|---|---|---|---|---|---|---|---|
| S2 | TAC_traj_lat_correct | 0 | -0.0014 [-0.0032, +0.0000] | -0.0007 [-0.0021, +0.0005] | -0.0007 [-0.0021, +0.0005] | -0.0017 [-0.0064, +0.0029] | -0.0024 [-0.0071, +0.0022] | 0.0019 / 0.0757 |
| S2 | TAC_traj_lat_correct | 1 | -0.0007 [-0.0022, +0.0007] | -0.0002 [-0.0015, +0.0010] | -0.0002 [-0.0015, +0.0010] | -0.0022 [-0.0068, +0.0024] | -0.0024 [-0.0072, +0.0022] | 0.0022 / 0.0745 |
| S2 | TAC_traj_lon_correct | 0 | +0.0233 [+0.0129, +0.0340] SEP | +0.0233 [+0.0129, +0.0340] SEP | +0.0233 [+0.0129, +0.0340] SEP | +0.0067 [-0.0055, +0.0195] | +0.0301 [+0.0169, +0.0440] SEP | 0.0019 / 0.0757 |
| S2 | TAC_traj_lon_correct | 1 | +0.0240 [+0.0126, +0.0357] SEP | +0.0240 [+0.0126, +0.0357] SEP | +0.0240 [+0.0126, +0.0357] SEP | -0.0026 [-0.0158, +0.0109] | +0.0214 [+0.0083, +0.0358] SEP | 0.0022 / 0.0745 |
| S6 | TAC_traj_lat_correct | 0 | +0.0018 [+0.0000, +0.0038] | +0.0018 [+0.0000, +0.0038] | +0.0018 [+0.0000, +0.0038] | -0.0012 [-0.0133, +0.0106] | +0.0006 [-0.0113, +0.0121] | 0.0031 / 0.1058 |
| S6 | TAC_traj_lat_correct | 1 | +0.0015 [+0.0000, +0.0031] | +0.0018 [+0.0006, +0.0034] SEP | +0.0018 [+0.0006, +0.0034] SEP | +0.0000 [-0.0122, +0.0117] | +0.0018 [-0.0102, +0.0135] | 0.0028 / 0.1064 |
| S6 | TAC_traj_lon_correct | 0 | +0.0111 [-0.0009, +0.0238] | +0.0114 [-0.0006, +0.0240] | +0.0111 [-0.0009, +0.0239] | -0.0326 [-0.0678, -0.0006] SEP | -0.0212 [-0.0564, +0.0093] | 0.0031 / 0.1058 |
| S6 | TAC_traj_lon_correct | 1 | +0.0172 [+0.0053, +0.0299] SEP | +0.0172 [+0.0053, +0.0299] SEP | +0.0166 [+0.0047, +0.0291] SEP | -0.0286 [-0.0641, +0.0031] | -0.0114 [-0.0467, +0.0205] | 0.0028 / 0.1064 |

### informational: 8-slot call vs per-surface call, Euclidean point distance (m), compliant / all windows

* S2|s0|compliant: mean 0.00040, max 0.2921 (n 4159)
* S2|s0|all: mean 0.00057, max 0.7718 (n 4754)
* S6|s0|compliant: mean 0.00123, max 0.1750 (n 3251)
* S6|s0|all: mean 0.00124, max 0.1750 (n 3668)
* S2|s1|compliant: mean 0.00041, max 0.3026 (n 4159)
* S2|s1|all: mean 0.00058, max 0.7563 (n 4754)
* S6|s1|compliant: mean 0.00124, max 0.1909 (n 3251)
* S6|s1|all: mean 0.00124, max 0.1909 (n 3668)

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
| S2 | s0 | TAC_traj_lat_correct | 0.9596129655838013 | 0.9596129655838013 | 0.9617164731025696 | 0.9617164731025696 | 4754 | True |
| S2 | s0 | TAC_traj_lon_correct | 0.8268826007843018 | 0.8268826007843018 | 0.8304585814476013 | 0.8304585814476013 | 4754 | True |
| S6 | s0 | TAC_traj_lat_correct | 0.8732278943061829 | 0.8732278943061829 | 0.8770447373390198 | 0.8770447373390198 | 3668 | True |
| S6 | s0 | TAC_traj_lon_correct | 0.5812431573867798 | 0.5812431573867798 | 0.6038713455200195 | 0.6038713455200195 | 3668 | True |
| S2 | s1 | TAC_traj_lat_correct | 0.957509458065033 | 0.957509458065033 | 0.9621371626853943 | 0.9621371626853943 | 4754 | True |
| S2 | s1 | TAC_traj_lon_correct | 0.8254101872444153 | 0.8254101872444153 | 0.8357173204421997 | 0.8357173204421997 | 4754 | True |
| S6 | s1 | TAC_traj_lat_correct | 0.8756815791130066 | 0.8756815791130066 | 0.8767721056938171 | 0.8767721056938171 | 3668 | True |
| S6 | s1 | TAC_traj_lon_correct | 0.5820610523223877 | 0.5820610523223877 | 0.6008723974227905 | 0.6008723974227905 | 3668 | True |

Single training seed; T1 open loop; EMA-30k reads are ORIENTATION ONLY unless the tree is step48852ema_unc.
