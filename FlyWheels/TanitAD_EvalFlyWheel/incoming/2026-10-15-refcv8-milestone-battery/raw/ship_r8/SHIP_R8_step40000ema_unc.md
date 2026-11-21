# SHIP_R8_step40000ema_unc

**SHIP-R8 on step40000ema_unc: PASS -- ORIENTATION ONLY (the condition's tree is step48852ema_unc; this is not the verdict)**

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

* E1 (S2) RETIME_DEPLOY8 - ha0_ext_pose: s0: -0.0310 [-0.0431, -0.0186] SEP | s1: -0.0308 [-0.0426, -0.0187] SEP
* E3 (S6) RETIME_DEPLOY8 - ha0_ext_pose: s0: -0.9315 [-1.2398, -0.6464] SEP | s1: -0.9356 [-1.2429, -0.6546] SEP
* FZR-NR per family: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}; FZR-LAT HOLDS True

TACTICAL cells (RETIME_DEPLOY8 - refcv7-50,400, accuracy, compliant windows; worse+separated = CI upper < 0):

| cell | delta [lo, hi] |
|---|---|
| S2|TAC_traj_lat_correct|s0 | -0.0005 [-0.0053, +0.0040] |
| S2|TAC_traj_lat_correct|s1 | -0.0022 [-0.0071, +0.0024] |
| S2|TAC_traj_lon_correct|s0 | +0.0267 [+0.0128, +0.0414] SEP |
| S2|TAC_traj_lon_correct|s1 | +0.0178 [+0.0050, +0.0314] SEP |
| S6|TAC_traj_lat_correct|s0 | -0.0003 [-0.0144, +0.0137] |
| S6|TAC_traj_lat_correct|s1 | -0.0037 [-0.0187, +0.0104] |
| S6|TAC_traj_lon_correct|s0 | -0.0274 [-0.0597, +0.0012] |
| S6|TAC_traj_lon_correct|s1 | -0.0246 [-0.0553, +0.0027] |
* PRIOR PARITY max |delta| by slot (m, worst over panels): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; smallest panel 4754 windows

## printed beside (outside the condition)

* pre-ruling ha0_ext (E1 S2 / E3 S6, scored arm): E1_S2: s0: -0.0259 [-0.0376, -0.0139] SEP | s1: -0.0257 [-0.0372, -0.0138] SEP ; E3_S6: s0: -0.9028 [-1.2096, -0.6256] SEP | s1: -0.9069 [-1.2112, -0.6325] SEP
* flat 0.5 (CONST_G05) - ha0_ext_pose: E1_S2_vs_pose: s0: -0.0305 [-0.0423, -0.0186] SEP | s1: -0.0304 [-0.0418, -0.0187] SEP ; E3_S6_vs_pose: s0: -1.0017 [-1.2846, -0.7514] SEP | s1: -1.0044 [-1.2831, -0.7548] SEP
* per-surface RETIME_DEPLOY (Amendment 1's form) - ha0_ext_pose: E1_S2_vs_pose: s0: -0.0310 [-0.0431, -0.0186] SEP | s1: -0.0308 [-0.0427, -0.0187] SEP ; E3_S6_vs_pose: s0: -0.9314 [-1.2398, -0.6464] SEP | s1: -0.9355 [-1.2428, -0.6545] SEP
* RETIME_TRAINFIT (per-surface train tables) - ha0_ext_pose: E1_S2_vs_pose: s0: -0.0310 [-0.0431, -0.0186] SEP | s1: -0.0308 [-0.0427, -0.0187] SEP ; E3_S6_vs_pose: s0: -0.9308 [-1.2394, -0.6453] SEP | s1: -0.9349 [-1.2424, -0.6538] SEP
* raw weights: no FORWARD_DEPLOY8 / FORWARD_DEPLOY file for the raw-weights tree

caveats: single training seed; T1 open loop (not driving performance); the stage was found post-hoc; the E1/E3 reference was switched after the 30k reads (D-R8-E1-REF); the train gains under-correct on eval (H-R8-GAINTRAJ-1); the bootstrap answers 'another draw of EPISODES' only (not another training or inference run)
