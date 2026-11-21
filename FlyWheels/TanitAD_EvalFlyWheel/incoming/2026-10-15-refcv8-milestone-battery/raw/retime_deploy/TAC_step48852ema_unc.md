# TAC_step48852ema_unc -- TACTICAL read of the RE-TIMED plan (SHIP-R8 clause 5, scored arm RETIME_DEPLOY8)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap, B 2000, bootstrap seed 0; models and gains FIXED ('another draw of EPISODES' only; inference seeds 0 and 1 are the replicate).
gains: deploy sha256 `244f313872004745b91a30104244ec9325b1da11bf633b9f31fb532af7075790` (from the frozen TRAIN gains sha256 `b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee`); script sha256 `e3e49cdb809374aa3bfebeced470cfe4cbe70f183373f0a0ae8c69b8d52ab3d7`.

**CLAUSE 5: TAC_traj_lat_correct and TAC_traj_lon_correct of the RE-TIMED plan (RETIME_DEPLOY8, AMENDMENT 2) not separated worse than refcv7-50,400 at both inference seeds, on ceiling-compliant windows, on every surface the battery defines them (S2, S6).  ACCURACY metric: separated worse <=> the paired-bootstrap CI upper bound < 0.**

## clause 5 on this tree: **FAIL**  (separated-worse cells: ['S6|TAC_traj_lon_correct|s0', 'S6|TAC_traj_lon_correct|s1']; missing: none)

Compliant windows. Clause-5 cell = RETIME_DEPLOY8 - refcv7-50,400 (accuracy, higher is better; worse+separated = CI upper < 0).

| surface | row | seed | n win / eps | os | RETIME_DEPLOY8 | per-surface DEPLOY | TRAINFIT | refcv7-50,400 | RETIME_DEPLOY8 - refcv7 (delta [lo, hi]) | separated worse |
|---|---|---|---|---|---|---|---|---|---|---|
| S2 | TAC_traj_lat_correct | 0 | 4159 / 138 | 0.9572 | 0.9582 | 0.9586 | 0.9586 | 0.9601 | -0.0019 [-0.0063, +0.0026] | False |
| S2 | TAC_traj_lat_correct | 1 | 4159 / 138 | 0.9548 | 0.9555 | 0.9560 | 0.9560 | 0.9601 | -0.0046 [-0.0099, +0.0005] | False |
| S2 | TAC_traj_lon_correct | 0 | 4159 / 138 | 0.8214 | 0.8577 | 0.8577 | 0.8577 | 0.8348 | +0.0228 [+0.0080, +0.0385] SEP | False |
| S2 | TAC_traj_lon_correct | 1 | 4159 / 138 | 0.8214 | 0.8584 | 0.8584 | 0.8584 | 0.8413 | +0.0171 [+0.0031, +0.0317] SEP | False |
| S6 | TAC_traj_lat_correct | 0 | 3251 / 138 | 0.8727 | 0.8748 | 0.8751 | 0.8751 | 0.8803 | -0.0055 [-0.0213, +0.0097] | False |
| S6 | TAC_traj_lat_correct | 1 | 3251 / 138 | 0.8727 | 0.8751 | 0.8754 | 0.8754 | 0.8816 | -0.0065 [-0.0221, +0.0078] | False |
| S6 | TAC_traj_lon_correct | 0 | 3251 / 138 | 0.5497 | 0.5666 | 0.5666 | 0.5666 | 0.5998 | -0.0332 [-0.0663, -0.0034] SEP | True |
| S6 | TAC_traj_lon_correct | 1 | 3251 / 138 | 0.5472 | 0.5644 | 0.5644 | 0.5638 | 0.5983 | -0.0338 [-0.0655, -0.0051] SEP | True |

## beside (outside clause 5): the per-surface form of Amendment 1 on the same cells reads **FAIL** (separated worse: ['S6|TAC_traj_lon_correct|s0', 'S6|TAC_traj_lon_correct|s1'])

| surface | row | seed | RETIME_DEPLOY8 - os | per-surface DEPLOY - os | TRAINFIT - os | os - refcv7 | per-surface DEPLOY - refcv7 | label changed vs os, RETIME_DEPLOY8 (lat / lon) |
|---|---|---|---|---|---|---|---|---|
| S2 | TAC_traj_lat_correct | 0 | +0.0010 [-0.0007, +0.0028] | +0.0014 [+0.0000, +0.0031] | +0.0014 [+0.0000, +0.0031] | -0.0029 [-0.0077, +0.0019] | -0.0014 [-0.0058, +0.0031] | 0.0029 / 0.0945 |
| S2 | TAC_traj_lat_correct | 1 | +0.0007 [-0.0010, +0.0024] | +0.0012 [-0.0000, +0.0026] | +0.0012 [-0.0000, +0.0026] | -0.0053 [-0.0108, +0.0000] | -0.0041 [-0.0093, +0.0010] | 0.0031 / 0.0899 |
| S2 | TAC_traj_lon_correct | 0 | +0.0363 [+0.0230, +0.0494] SEP | +0.0363 [+0.0230, +0.0494] SEP | +0.0363 [+0.0230, +0.0494] SEP | -0.0135 [-0.0299, +0.0027] | +0.0228 [+0.0080, +0.0385] SEP | 0.0029 / 0.0945 |
| S2 | TAC_traj_lon_correct | 1 | +0.0370 [+0.0233, +0.0508] SEP | +0.0370 [+0.0233, +0.0508] SEP | +0.0370 [+0.0233, +0.0508] SEP | -0.0200 [-0.0362, -0.0039] SEP | +0.0171 [+0.0031, +0.0317] SEP | 0.0031 / 0.0899 |
| S6 | TAC_traj_lat_correct | 0 | +0.0022 [+0.0003, +0.0044] SEP | +0.0025 [+0.0006, +0.0047] SEP | +0.0025 [+0.0006, +0.0047] SEP | -0.0077 [-0.0237, +0.0076] | -0.0052 [-0.0210, +0.0098] | 0.0046 / 0.1258 |
| S6 | TAC_traj_lat_correct | 1 | +0.0025 [+0.0000, +0.0053] | +0.0028 [+0.0006, +0.0056] SEP | +0.0028 [+0.0006, +0.0056] SEP | -0.0089 [-0.0253, +0.0059] | -0.0062 [-0.0217, +0.0082] | 0.0043 / 0.1197 |
| S6 | TAC_traj_lon_correct | 0 | +0.0169 [+0.0024, +0.0317] SEP | +0.0169 [+0.0024, +0.0317] SEP | +0.0169 [+0.0025, +0.0317] SEP | -0.0501 [-0.0844, -0.0196] SEP | -0.0332 [-0.0663, -0.0034] SEP | 0.0046 / 0.1258 |
| S6 | TAC_traj_lon_correct | 1 | +0.0172 [+0.0031, +0.0309] SEP | +0.0172 [+0.0031, +0.0309] SEP | +0.0166 [+0.0027, +0.0302] SEP | -0.0511 [-0.0868, -0.0196] SEP | -0.0338 [-0.0655, -0.0051] SEP | 0.0043 / 0.1197 |

### informational: 8-slot call vs per-surface call, Euclidean point distance (m), compliant / all windows

* S2|s0|compliant: mean 0.00047, max 0.3123 (n 4159)
* S2|s0|all: mean 0.00065, max 0.3123 (n 4754)
* S6|s0|compliant: mean 0.00135, max 0.2002 (n 3251)
* S6|s0|all: mean 0.00134, max 0.2002 (n 3668)
* S2|s1|compliant: mean 0.00050, max 0.3224 (n 4159)
* S2|s1|all: mean 0.00065, max 0.3224 (n 4754)
* S6|s1|compliant: mean 0.00137, max 0.1791 (n 3251)
* S6|s1|all: mean 0.00136, max 0.1791 (n 3668)

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
| S2 | s0 | TAC_traj_lat_correct | 0.957509458065033 | 0.957509458065033 | 0.9617164731025696 | 0.9617164731025696 | 4754 | True |
| S2 | s0 | TAC_traj_lon_correct | 0.8090029358863831 | 0.8090029358863831 | 0.8304585814476013 | 0.8304585814476013 | 4754 | True |
| S6 | s0 | TAC_traj_lat_correct | 0.8647764325141907 | 0.8647764325141907 | 0.8770447373390198 | 0.8770447373390198 | 3668 | True |
| S6 | s0 | TAC_traj_lon_correct | 0.5665212869644165 | 0.5665212869644165 | 0.6038713455200195 | 0.6038713455200195 | 3668 | True |
| S2 | s1 | TAC_traj_lat_correct | 0.955616295337677 | 0.955616295337677 | 0.9621371626853943 | 0.9621371626853943 | 4754 | True |
| S2 | s1 | TAC_traj_lon_correct | 0.8087925910949707 | 0.8087925910949707 | 0.8357173204421997 | 0.8357173204421997 | 4754 | True |
| S6 | s1 | TAC_traj_lat_correct | 0.8625954389572144 | 0.8625954389572144 | 0.8767721056938171 | 0.8767721056938171 | 3668 | True |
| S6 | s1 | TAC_traj_lon_correct | 0.5618866086006165 | 0.5618866086006165 | 0.6008723974227905 | 0.6008723974227905 | 3668 | True |

Single training seed; T1 open loop; EMA-30k reads are ORIENTATION ONLY unless the tree is step48852ema_unc.
