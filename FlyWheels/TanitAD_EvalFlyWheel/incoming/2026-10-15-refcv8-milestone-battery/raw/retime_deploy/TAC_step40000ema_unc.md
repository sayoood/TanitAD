# TAC_step40000ema_unc -- TACTICAL read of the RE-TIMED plan (SHIP-R8 clause 5, scored arm RETIME_DEPLOY8)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap, B 2000, bootstrap seed 0; models and gains FIXED ('another draw of EPISODES' only; inference seeds 0 and 1 are the replicate).
gains: deploy sha256 `244f313872004745b91a30104244ec9325b1da11bf633b9f31fb532af7075790` (from the frozen TRAIN gains sha256 `b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee`); script sha256 `e3e49cdb809374aa3bfebeced470cfe4cbe70f183373f0a0ae8c69b8d52ab3d7`.

**CLAUSE 5: TAC_traj_lat_correct and TAC_traj_lon_correct of the RE-TIMED plan (RETIME_DEPLOY8, AMENDMENT 2) not separated worse than refcv7-50,400 at both inference seeds, on ceiling-compliant windows, on every surface the battery defines them (S2, S6).  ACCURACY metric: separated worse <=> the paired-bootstrap CI upper bound < 0.**

## clause 5 on this tree: **PASS**  (separated-worse cells: none; missing: none)

Compliant windows. Clause-5 cell = RETIME_DEPLOY8 - refcv7-50,400 (accuracy, higher is better; worse+separated = CI upper < 0).

| surface | row | seed | n win / eps | os | RETIME_DEPLOY8 | per-surface DEPLOY | TRAINFIT | refcv7-50,400 | RETIME_DEPLOY8 - refcv7 (delta [lo, hi]) | separated worse |
|---|---|---|---|---|---|---|---|---|---|---|
| S2 | TAC_traj_lat_correct | 0 | 4159 / 138 | 0.9586 | 0.9596 | 0.9601 | 0.9601 | 0.9601 | -0.0005 [-0.0053, +0.0040] | False |
| S2 | TAC_traj_lat_correct | 1 | 4159 / 138 | 0.9574 | 0.9579 | 0.9584 | 0.9584 | 0.9601 | -0.0022 [-0.0071, +0.0024] | False |
| S2 | TAC_traj_lon_correct | 0 | 4159 / 138 | 0.8346 | 0.8615 | 0.8615 | 0.8615 | 0.8348 | +0.0267 [+0.0128, +0.0414] SEP | False |
| S2 | TAC_traj_lon_correct | 1 | 4159 / 138 | 0.8346 | 0.8591 | 0.8591 | 0.8591 | 0.8413 | +0.0178 [+0.0050, +0.0314] SEP | False |
| S6 | TAC_traj_lat_correct | 0 | 3251 / 138 | 0.8776 | 0.8800 | 0.8803 | 0.8803 | 0.8803 | -0.0003 [-0.0144, +0.0137] | False |
| S6 | TAC_traj_lat_correct | 1 | 3251 / 138 | 0.8745 | 0.8779 | 0.8779 | 0.8779 | 0.8816 | -0.0037 [-0.0187, +0.0104] | False |
| S6 | TAC_traj_lon_correct | 0 | 3251 / 138 | 0.5608 | 0.5724 | 0.5724 | 0.5724 | 0.5998 | -0.0274 [-0.0597, +0.0012] | False |
| S6 | TAC_traj_lon_correct | 1 | 3251 / 138 | 0.5595 | 0.5737 | 0.5737 | 0.5737 | 0.5983 | -0.0246 [-0.0553, +0.0027] | False |

## beside (outside clause 5): the per-surface form of Amendment 1 on the same cells reads **PASS** (separated worse: none)

| surface | row | seed | RETIME_DEPLOY8 - os | per-surface DEPLOY - os | TRAINFIT - os | os - refcv7 | per-surface DEPLOY - refcv7 | label changed vs os, RETIME_DEPLOY8 (lat / lon) |
|---|---|---|---|---|---|---|---|---|
| S2 | TAC_traj_lat_correct | 0 | +0.0010 [-0.0002, +0.0024] | +0.0014 [+0.0005, +0.0027] SEP | +0.0014 [+0.0005, +0.0027] SEP | -0.0014 [-0.0064, +0.0033] | +0.0000 [-0.0048, +0.0046] | 0.0019 / 0.0832 |
| S2 | TAC_traj_lat_correct | 1 | +0.0005 [-0.0007, +0.0019] | +0.0010 [-0.0002, +0.0022] | +0.0010 [-0.0002, +0.0022] | -0.0026 [-0.0078, +0.0024] | -0.0017 [-0.0066, +0.0029] | 0.0019 / 0.0779 |
| S2 | TAC_traj_lon_correct | 0 | +0.0269 [+0.0152, +0.0387] SEP | +0.0269 [+0.0152, +0.0387] SEP | +0.0269 [+0.0152, +0.0387] SEP | -0.0002 [-0.0149, +0.0139] | +0.0267 [+0.0128, +0.0414] SEP | 0.0019 / 0.0832 |
| S2 | TAC_traj_lon_correct | 1 | +0.0245 [+0.0137, +0.0359] SEP | +0.0245 [+0.0137, +0.0359] SEP | +0.0245 [+0.0137, +0.0359] SEP | -0.0067 [-0.0206, +0.0073] | +0.0178 [+0.0050, +0.0314] SEP | 0.0019 / 0.0779 |
| S6 | TAC_traj_lat_correct | 0 | +0.0025 [+0.0006, +0.0043] SEP | +0.0028 [+0.0012, +0.0046] SEP | +0.0028 [+0.0012, +0.0046] SEP | -0.0028 [-0.0169, +0.0116] | +0.0000 [-0.0141, +0.0140] | 0.0031 / 0.1193 |
| S6 | TAC_traj_lat_correct | 1 | +0.0034 [+0.0006, +0.0067] SEP | +0.0034 [+0.0006, +0.0067] SEP | +0.0034 [+0.0006, +0.0067] SEP | -0.0071 [-0.0220, +0.0072] | -0.0037 [-0.0187, +0.0104] | 0.0040 / 0.1138 |
| S6 | TAC_traj_lon_correct | 0 | +0.0117 [-0.0012, +0.0256] | +0.0117 [-0.0012, +0.0256] | +0.0117 [-0.0012, +0.0256] | -0.0391 [-0.0738, -0.0096] SEP | -0.0274 [-0.0597, +0.0012] | 0.0031 / 0.1193 |
| S6 | TAC_traj_lon_correct | 1 | +0.0141 [+0.0006, +0.0272] SEP | +0.0141 [+0.0006, +0.0272] SEP | +0.0141 [+0.0006, +0.0272] SEP | -0.0388 [-0.0730, -0.0088] SEP | -0.0246 [-0.0553, +0.0027] | 0.0040 / 0.1138 |

### informational: 8-slot call vs per-surface call, Euclidean point distance (m), compliant / all windows

* S2|s0|compliant: mean 0.00042, max 0.2996 (n 4159)
* S2|s0|all: mean 0.00057, max 0.2996 (n 4754)
* S6|s0|compliant: mean 0.00132, max 0.1320 (n 3251)
* S6|s0|all: mean 0.00132, max 0.1320 (n 3668)
* S2|s1|compliant: mean 0.00038, max 0.2253 (n 4159)
* S2|s1|all: mean 0.00053, max 0.2253 (n 4754)
* S6|s1|compliant: mean 0.00133, max 0.1845 (n 3251)
* S6|s1|all: mean 0.00133, max 0.1845 (n 3668)

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
| S2 | s0 | TAC_traj_lat_correct | 0.9594026207923889 | 0.9594026207923889 | 0.9617164731025696 | 0.9617164731025696 | 4754 | True |
| S2 | s0 | TAC_traj_lon_correct | 0.8228859901428223 | 0.8228859901428223 | 0.8304585814476013 | 0.8304585814476013 | 4754 | True |
| S6 | s0 | TAC_traj_lat_correct | 0.869956374168396 | 0.869956374168396 | 0.8770447373390198 | 0.8770447373390198 | 3668 | True |
| S6 | s0 | TAC_traj_lon_correct | 0.574700117111206 | 0.574700117111206 | 0.6038713455200195 | 0.6038713455200195 | 3668 | True |
| S2 | s1 | TAC_traj_lat_correct | 0.9579301476478577 | 0.9579301476478577 | 0.9621371626853943 | 0.9621371626853943 | 4754 | True |
| S2 | s1 | TAC_traj_lon_correct | 0.823306679725647 | 0.823306679725647 | 0.8357173204421997 | 0.8357173204421997 | 4754 | True |
| S6 | s1 | TAC_traj_lat_correct | 0.8666848540306091 | 0.8666848540306091 | 0.8767721056938171 | 0.8767721056938171 | 3668 | True |
| S6 | s1 | TAC_traj_lon_correct | 0.572519063949585 | 0.572519063949585 | 0.6008723974227905 | 0.6008723974227905 | 3668 | True |

Single training seed; T1 open loop; EMA-30k reads are ORIENTATION ONLY unless the tree is step48852ema_unc.
