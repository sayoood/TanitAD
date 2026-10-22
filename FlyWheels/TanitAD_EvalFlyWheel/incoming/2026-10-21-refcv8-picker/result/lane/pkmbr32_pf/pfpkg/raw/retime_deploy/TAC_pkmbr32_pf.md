# TAC_pkmbr32_pf -- TACTICAL read of the RE-TIMED plan (SHIP-R8 clause 5, scored arm RETIME_DEPLOY8)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap, B 2000, bootstrap seed 0; models and gains FIXED ('another draw of EPISODES' only; inference seeds 0 and 1 are the replicate).
gains: deploy sha256 `c536533b96ddd2521f80cbd99e06212fb043fdb26360a1a394a829a4bf5c2fd1` (from the frozen TRAIN gains sha256 `2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80`); script sha256 `e3e49cdb809374aa3bfebeced470cfe4cbe70f183373f0a0ae8c69b8d52ab3d7`.

**CLAUSE 5: TAC_traj_lat_correct and TAC_traj_lon_correct of the RE-TIMED plan (RETIME_DEPLOY8, AMENDMENT 2) not separated worse than refcv7-50,400 at both inference seeds, on ceiling-compliant windows, on every surface the battery defines them (S2, S6).  ACCURACY metric: separated worse <=> the paired-bootstrap CI upper bound < 0.**

## clause 5 on this tree: **PASS**  (separated-worse cells: none; missing: none)

Compliant windows. Clause-5 cell = RETIME_DEPLOY8 - refcv7-50,400 (accuracy, higher is better; worse+separated = CI upper < 0).

| surface | row | seed | n win / eps | os | RETIME_DEPLOY8 | per-surface DEPLOY | TRAINFIT | refcv7-50,400 | RETIME_DEPLOY8 - refcv7 (delta [lo, hi]) | separated worse |
|---|---|---|---|---|---|---|---|---|---|---|
| S2 | TAC_traj_lat_correct | 0 | 4159 / 138 | 0.9627 | 0.9632 | 0.9637 | 0.9637 | 0.9601 | +0.0031 [-0.0020, +0.0078] | False |
| S2 | TAC_traj_lat_correct | 1 | 4159 / 138 | 0.9608 | 0.9620 | 0.9620 | 0.9620 | 0.9601 | +0.0019 [-0.0026, +0.0065] | False |
| S2 | TAC_traj_lon_correct | 0 | 4159 / 138 | 0.8584 | 0.8692 | 0.8692 | 0.8692 | 0.8348 | +0.0344 [+0.0205, +0.0499] SEP | False |
| S2 | TAC_traj_lon_correct | 1 | 4159 / 138 | 0.8553 | 0.8690 | 0.8690 | 0.8690 | 0.8413 | +0.0277 [+0.0138, +0.0432] SEP | False |
| S6 | TAC_traj_lat_correct | 0 | 3251 / 138 | 0.8871 | 0.8890 | 0.8890 | 0.8890 | 0.8803 | +0.0086 [-0.0032, +0.0212] | False |
| S6 | TAC_traj_lat_correct | 1 | 3251 / 138 | 0.8828 | 0.8850 | 0.8850 | 0.8850 | 0.8816 | +0.0034 [-0.0090, +0.0155] | False |
| S6 | TAC_traj_lon_correct | 0 | 3251 / 138 | 0.6155 | 0.6180 | 0.6180 | 0.6183 | 0.5998 | +0.0181 [-0.0067, +0.0402] | False |
| S6 | TAC_traj_lon_correct | 1 | 3251 / 138 | 0.6020 | 0.6063 | 0.6063 | 0.6066 | 0.5983 | +0.0080 [-0.0144, +0.0288] | False |

## beside (outside clause 5): the per-surface form of Amendment 1 on the same cells reads **PASS** (separated worse: none)

| surface | row | seed | RETIME_DEPLOY8 - os | per-surface DEPLOY - os | TRAINFIT - os | os - refcv7 | per-surface DEPLOY - refcv7 | label changed vs os, RETIME_DEPLOY8 (lat / lon) |
|---|---|---|---|---|---|---|---|---|
| S2 | TAC_traj_lat_correct | 0 | +0.0005 [-0.0010, +0.0020] | +0.0010 [-0.0002, +0.0024] | +0.0010 [-0.0002, +0.0024] | +0.0026 [-0.0024, +0.0073] | +0.0036 [-0.0016, +0.0086] | 0.0024 / 0.0572 |
| S2 | TAC_traj_lat_correct | 1 | +0.0012 [+0.0002, +0.0026] SEP | +0.0012 [+0.0002, +0.0026] SEP | +0.0012 [+0.0002, +0.0026] SEP | +0.0007 [-0.0038, +0.0053] | +0.0019 [-0.0026, +0.0065] | 0.0012 / 0.0594 |
| S2 | TAC_traj_lon_correct | 0 | +0.0108 [+0.0030, +0.0199] SEP | +0.0108 [+0.0030, +0.0199] SEP | +0.0108 [+0.0030, +0.0199] SEP | +0.0236 [+0.0111, +0.0367] SEP | +0.0344 [+0.0205, +0.0499] SEP | 0.0024 / 0.0572 |
| S2 | TAC_traj_lon_correct | 1 | +0.0137 [+0.0048, +0.0241] SEP | +0.0137 [+0.0048, +0.0241] SEP | +0.0137 [+0.0048, +0.0241] SEP | +0.0139 [+0.0012, +0.0273] SEP | +0.0277 [+0.0138, +0.0432] SEP | 0.0012 / 0.0594 |
| S6 | TAC_traj_lat_correct | 0 | +0.0018 [+0.0006, +0.0034] SEP | +0.0018 [+0.0006, +0.0034] SEP | +0.0018 [+0.0006, +0.0034] SEP | +0.0068 [-0.0056, +0.0197] | +0.0086 [-0.0032, +0.0212] | 0.0018 / 0.0332 |
| S6 | TAC_traj_lat_correct | 1 | +0.0022 [+0.0006, +0.0038] SEP | +0.0022 [+0.0006, +0.0038] SEP | +0.0022 [+0.0006, +0.0038] SEP | +0.0012 [-0.0112, +0.0138] | +0.0034 [-0.0090, +0.0155] | 0.0022 / 0.0298 |
| S6 | TAC_traj_lon_correct | 0 | +0.0025 [-0.0040, +0.0086] | +0.0025 [-0.0040, +0.0086] | +0.0028 [-0.0034, +0.0087] | +0.0157 [-0.0089, +0.0389] | +0.0181 [-0.0067, +0.0402] | 0.0018 / 0.0332 |
| S6 | TAC_traj_lon_correct | 1 | +0.0043 [-0.0015, +0.0099] | +0.0043 [-0.0015, +0.0099] | +0.0046 [-0.0012, +0.0102] | +0.0037 [-0.0193, +0.0256] | +0.0080 [-0.0144, +0.0288] | 0.0022 / 0.0298 |

### informational: 8-slot call vs per-surface call, Euclidean point distance (m), compliant / all windows

* S2|s0|compliant: mean 0.00039, max 0.2017 (n 4159)
* S2|s0|all: mean 0.00051, max 0.2017 (n 4754)
* S6|s0|compliant: mean 0.00079, max 0.0901 (n 3251)
* S6|s0|all: mean 0.00079, max 0.0901 (n 3668)
* S2|s1|compliant: mean 0.00038, max 0.2057 (n 4159)
* S2|s1|all: mean 0.00052, max 0.2057 (n 4754)
* S6|s1|compliant: mean 0.00079, max 0.0862 (n 3251)
* S6|s1|all: mean 0.00079, max 0.0862 (n 3668)

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
| S2 | s0 | TAC_traj_lat_correct | 0.9629785418510437 | 0.9629785418510437 | 0.9617164731025696 | 0.9617164731025696 | 4754 | True |
| S2 | s0 | TAC_traj_lon_correct | 0.8424484729766846 | 0.8424484729766846 | 0.8304585814476013 | 0.8304585814476013 | 4754 | True |
| S6 | s0 | TAC_traj_lat_correct | 0.8819519877433777 | 0.8819519877433777 | 0.8770447373390198 | 0.8770447373390198 | 3668 | True |
| S6 | s0 | TAC_traj_lon_correct | 0.6188658475875854 | 0.6188658475875854 | 0.6038713455200195 | 0.6038713455200195 | 3668 | True |
| S2 | s1 | TAC_traj_lat_correct | 0.9615061283111572 | 0.9615061283111572 | 0.9621371626853943 | 0.9621371626853943 | 4754 | True |
| S2 | s1 | TAC_traj_lon_correct | 0.8367690443992615 | 0.8367690443992615 | 0.8357173204421997 | 0.8357173204421997 | 4754 | True |
| S6 | s1 | TAC_traj_lat_correct | 0.8778625726699829 | 0.8778625726699829 | 0.8767721056938171 | 0.8767721056938171 | 3668 | True |
| S6 | s1 | TAC_traj_lon_correct | 0.6082333922386169 | 0.6082333922386169 | 0.6008723974227905 | 0.6008723974227905 | 3668 | True |

Single training seed; T1 open loop; EMA-30k reads are ORIENTATION ONLY unless the tree is step48852ema_unc.
