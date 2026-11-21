# SHIP_R8_step30000ema_unc

**SHIP-R8 on step30000ema_unc: PASS -- ORIENTATION ONLY (the condition's tree is step48852ema_unc; this is not the verdict)**

scored arm: `RETIME_DEPLOY8` (one call on all 8 model slots, AMENDMENT 2)

condition: SHIP-R8 holds iff, on tree `step48852ema_unc` (EMA weights), ceiling-compliant windows, arm `RETIME_TRAINFIT` with the frozen gains UNCHANGED (sha256 b208c9c1e4236075…), both inference seeds 0 and 1, paired episode-cluster bootstrap B 2000: ... Any failed clause = the stage does NOT ship and refcv8 is reported failing that bar.

amendments: AMENDMENT 1: SHIP-R8 clauses (1)–(6) are evaluated on `RETIME_DEPLOY`; `RETIME_TRAINFIT` (per-surface) is printed beside. If `RETIME_DEPLOY` cannot be computed for 48,852, SHIP-R8 reads INCOMPLETE.  AMENDMENT 2: `RETIME_DEPLOY` = ONE call per window on all 8 model slots with Amendment 1's table, prior = `ha0_ext_pose` at the same 8 slots; each surface is scored on that call's outputs at its own instants; the per-surface form is printed beside.  New clause (7) PRIOR PARITY.

| clause | status | reading |
|---|---|---|
| (1) FORWARD_TRAINFIT controls ALL_PASS | **PASS** | FORWARD_TRAINFIT and FORWARD_DEPLOY8 controls ALL_PASS; the frozen gains hash is the pinned one; seeds 0 and 1; B 2000 |
| (2) E1 S2 ADE − `ha0_ext_pose` separated below 0 at both seeds | **PASS** | E1: RETIME_DEPLOY8 - ha0_ext_pose ADE on S2 is separated below 0 at both inference seeds |
| (3) E3 S6 ADE − `ha0_ext_pose` separated below 0 at both seeds | **PASS** | E3: RETIME_DEPLOY8 - ha0_ext_pose ADE on S6 is separated below 0 at both inference seeds |
| (4) LONGITUDINAL / LATERAL / ADE no row separated worse than refcv7-50,400 at both seeds (FZR-NR), and S6 lateral HOLDS (FZR-LAT) | **PASS** | FZR-NR ADE / LONGITUDINAL / LATERAL all PASS (no headline row separated worse than refcv7-50,400 at either seed) and FZR-LAT HOLDS |
| (5) TACTICAL `TAC_traj_lat_correct` and `TAC_traj_lon_correct` of the RE-TIMED plan not separated worse than refcv7-50,400 at both seeds (instrument being built 2026-10-19; if it cannot be computed, SHIP-R8 reads INCOMPLETE, not PASS) | **PASS** | no TAC_traj_lat / TAC_traj_lon cell (S2, S6 x seeds 0, 1) of RETIME_DEPLOY8 is separated worse than refcv7-50,400 |
| (6) STRATEGIC not applicable, n = 0, `--no-strategic` | **N/A** | the battery's strategic block reads NOT APPLICABLE, n = 0 (--no-strategic): satisfied as the condition states |
| (7) PRIOR PARITY: the prior the eval rebuilds from banked poses must equal the model's emitted `residual_prior_path` (built in `RefCModel.forward` via `kinematic_prior.prior_controls('ha0_ext_pose')`) to ≤ 1e-3 m at all 8 slots on ≥ 200 eval windows; if it cannot be shown, SHIP-R8 reads INCOMPLETE | **PASS** | max |delta| <= 0.001 m at all 8 slots on >= 200 windows (smallest panel 4754; 4 panel(s)) |

## evidence

* E1 (S2) RETIME_DEPLOY8 - ha0_ext_pose: s0: -0.0318 [-0.0433, -0.0201] SEP | s1: -0.0313 [-0.0426, -0.0193] SEP
* E3 (S6) RETIME_DEPLOY8 - ha0_ext_pose: s0: -0.9767 [-1.2773, -0.7011] SEP | s1: -0.9730 [-1.2669, -0.6938] SEP
* FZR-NR per family: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}; FZR-LAT HOLDS True

TACTICAL cells (RETIME_DEPLOY8 - refcv7-50,400, accuracy, compliant windows; worse+separated = CI upper < 0):

| cell | delta [lo, hi] |
|---|---|
| S2|TAC_traj_lat_correct|s0 | -0.0031 [-0.0077, +0.0014] |
| S2|TAC_traj_lat_correct|s1 | -0.0029 [-0.0078, +0.0019] |
| S2|TAC_traj_lon_correct|s0 | +0.0301 [+0.0169, +0.0440] SEP |
| S2|TAC_traj_lon_correct|s1 | +0.0214 [+0.0083, +0.0358] SEP |
| S6|TAC_traj_lat_correct|s0 | +0.0006 [-0.0113, +0.0121] |
| S6|TAC_traj_lat_correct|s1 | +0.0015 [-0.0105, +0.0132] |
| S6|TAC_traj_lon_correct|s0 | -0.0215 [-0.0566, +0.0088] |
| S6|TAC_traj_lon_correct|s1 | -0.0114 [-0.0467, +0.0205] |
* PRIOR PARITY max |delta| by slot (m, worst over panels): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; smallest panel 4754 windows

## printed beside (outside the condition)

* pre-ruling ha0_ext (E1 S2 / E3 S6, scored arm): E1_S2: s0: -0.0267 [-0.0379, -0.0150] SEP | s1: -0.0262 [-0.0372, -0.0145] SEP ; E3_S6: s0: -0.9479 [-1.2475, -0.6763] SEP | s1: -0.9443 [-1.2361, -0.6730] SEP
* flat 0.5 (CONST_G05) - ha0_ext_pose: E1_S2_vs_pose: s0: -0.0308 [-0.0420, -0.0196] SEP | s1: -0.0303 [-0.0413, -0.0190] SEP ; E3_S6_vs_pose: s0: -1.0104 [-1.2845, -0.7643] SEP | s1: -1.0084 [-1.2794, -0.7669] SEP
* per-surface RETIME_DEPLOY (Amendment 1's form) - ha0_ext_pose: E1_S2_vs_pose: s0: -0.0318 [-0.0433, -0.0201] SEP | s1: -0.0313 [-0.0426, -0.0194] SEP ; E3_S6_vs_pose: s0: -0.9766 [-1.2771, -0.7010] SEP | s1: -0.9730 [-1.2668, -0.6936] SEP
* RETIME_TRAINFIT (per-surface train tables) - ha0_ext_pose: E1_S2_vs_pose: s0: -0.0318 [-0.0433, -0.0201] SEP | s1: -0.0313 [-0.0426, -0.0194] SEP ; E3_S6_vs_pose: s0: -0.9763 [-1.2770, -0.7007] SEP | s1: -0.9726 [-1.2664, -0.6931] SEP
* raw weights (step30000_unc, RETIME_DEPLOY8): E1_S2_vs_pose: s0: -0.0147 [-0.0271, -0.0019] SEP | s1: -0.0144 [-0.0268, -0.0014] SEP ; E3_S6_vs_pose: s0: -0.8659 [-1.1574, -0.5901] SEP | s1: -0.8662 [-1.1584, -0.5871] SEP

caveats: single training seed; T1 open loop (not driving performance); the stage was found post-hoc; the E1/E3 reference was switched after the 30k reads (D-R8-E1-REF); the train gains under-correct on eval (H-R8-GAINTRAJ-1); the bootstrap answers 'another draw of EPISODES' only (not another training or inference run)
