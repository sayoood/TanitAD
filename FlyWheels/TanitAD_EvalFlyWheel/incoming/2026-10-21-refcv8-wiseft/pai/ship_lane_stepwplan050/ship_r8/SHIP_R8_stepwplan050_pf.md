# SHIP_R8_stepwplan050_pf (AMENDMENT 3, per-frame feed)

**SHIP-R8 (AMENDMENT 3) on stepwplan050_pf: FAIL -- ORIENTATION ONLY (the condition's tree is step48852ema_pf)**

scored arm: `RETIME_DEPLOY8` (one call on all 8 model slots, AMENDMENT 2)

condition: SHIP-R8 holds iff, on tree `step48852ema_unc` (EMA weights), ceiling-compliant windows, arm `RETIME_TRAINFIT` with the frozen gains UNCHANGED (sha256 b208c9c1e4236075…), both inference seeds 0 and 1, paired episode-cluster bootstrap B 2000: ... Any failed clause = the stage does NOT ship and refcv8 is reported failing that bar.

amendments: AMENDMENT 1: SHIP-R8 clauses (1)–(6) are evaluated on `RETIME_DEPLOY`; `RETIME_TRAINFIT` (per-surface) is printed beside. If `RETIME_DEPLOY` cannot be computed for 48,852, SHIP-R8 reads INCOMPLETE.  AMENDMENT 2: `RETIME_DEPLOY` = ONE call per window on all 8 model slots with Amendment 1's table, prior = `ha0_ext_pose` at the same 8 slots; each surface is scored on that call's outputs at its own instants; the per-surface form is printed beside.  New clause (7) PRIOR PARITY.  AMENDMENT 3: (i) every SHIP-R8 clause and every refcv8 bar is read on rolls fed `perframe_full` (tree step 48,852 EMA, tag `step48852ema_pf`, inference seeds 0 and 1, paired episode-cluster bootstrap B 2000); (ii) the RETIME table is re-fitted by the unchanged H-R8-RETIME-TRAINFIT-1 procedure on per-frame-fed train rolls (30k EMA, A6 train139, seeds 0+1 pooled, prior ha0_ext_pose), its sha256 registered BEFORE any 48,852 pf eval roll; the clip-fed table b208c9c1... is printed beside only; (iii) new clause (1b) FEED PARITY asserted on the roll artifacts -- if it fails or cannot be shown, SHIP-R8 reads INCOMPLETE; (iv) the per-clip reads (`step48852ema_unc`) are printed beside as the deliberate-regression arm, never binding; (v) a missing clause means INCOMPLETE, never PASS.  Amendments 1 and 2 still apply.

| clause | status | reading |
|---|---|---|
| (1) FORWARD_TRAINFIT controls ALL_PASS | **PASS** | FORWARD_TRAINFIT and FORWARD_DEPLOY8 controls ALL_PASS; the frozen gains hash is the pinned one; seeds 0 and 1; B 2000 | the per-frame table's sha256 was registered before every pf roll of the tree (Amendment 3 (ii)) |
| (1b) FEED PARITY: on the tree's own roll artifacts, both seeds, every window: the fed token equals the v9 per-frame token, `nav_cmd_clip` equals the per-clip token, and `r8_nav` is non-zero exactly where `r8_nav_known`; fails or cannot be shown -> SHIP-R8 INCOMPLETE | **PASS** | both seeds, every window of the tree's dumps: fed token == v9 per-frame token, nav_cmd_clip == per-clip token, r8_nav non-zero exactly where known |
| (2) E1 S2 ADE − `ha0_ext_pose` separated below 0 at both seeds | **FAIL** | E1: not separated below 0 at seed(s) s0, s1 |
| (3) E3 S6 ADE − `ha0_ext_pose` separated below 0 at both seeds | **PASS** | E3: RETIME_DEPLOY8 - ha0_ext_pose ADE on S6 is separated below 0 at both inference seeds |
| (4) LONGITUDINAL / LATERAL / ADE no row separated worse than refcv7-50,400 at both seeds (FZR-NR), and S6 lateral HOLDS (FZR-LAT) | **FAIL** | ADE: FAIL: S2:ade_m, S6:ade_m; LONGITUDINAL: FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m |
| (5) TACTICAL `TAC_traj_lat_correct` and `TAC_traj_lon_correct` of the RE-TIMED plan not separated worse than refcv7-50,400 at both seeds (instrument being built 2026-10-19; if it cannot be computed, SHIP-R8 reads INCOMPLETE, not PASS) | **FAIL** | separated worse than refcv7-50,400: S6|TAC_traj_lon_correct|s0, S6|TAC_traj_lon_correct|s1 |
| (6) STRATEGIC not applicable, n = 0, `--no-strategic` | **N/A** | the battery's strategic block reads NOT APPLICABLE, n = 0 (--no-strategic): satisfied as the condition states |
| (7) PRIOR PARITY: the prior the eval rebuilds from banked poses must equal the model's emitted `residual_prior_path` (built in `RefCModel.forward` via `kinematic_prior.prior_controls('ha0_ext_pose')`) to ≤ 1e-3 m at all 8 slots on ≥ 200 eval windows; if it cannot be shown, SHIP-R8 reads INCOMPLETE | **PASS** | max |delta| <= 0.001 m at all 8 slots on >= 200 windows (smallest panel 4754; 2 panel(s)) |

## evidence

* E1 (S2) RETIME_DEPLOY8 - ha0_ext_pose: s0: +0.0563 [+0.0417, +0.0704] SEP | s1: +0.0575 [+0.0425, +0.0718] SEP
* E3 (S6) RETIME_DEPLOY8 - ha0_ext_pose: s0: -0.6158 [-0.9416, -0.3043] SEP | s1: -0.6055 [-0.9343, -0.2899] SEP
* FZR-NR per family: {'ADE': 'FAIL: S2:ade_m, S6:ade_m', 'LONGITUDINAL': 'FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m', 'LATERAL': 'PASS'}; FZR-LAT HOLDS True

TACTICAL cells (RETIME_DEPLOY8 - refcv7-50,400, accuracy, compliant windows; worse+separated = CI upper < 0):

| cell | delta [lo, hi] |
|---|---|
| S2|TAC_traj_lat_correct|s0 | +0.0036 [-0.0005, +0.0082] |
| S2|TAC_traj_lat_correct|s1 | +0.0038 [-0.0005, +0.0085] |
| S2|TAC_traj_lon_correct|s0 | +0.0219 [+0.0063, +0.0381] SEP |
| S2|TAC_traj_lon_correct|s1 | +0.0161 [+0.0007, +0.0314] SEP |
| S6|TAC_traj_lat_correct|s0 | +0.0015 [-0.0083, +0.0110] |
| S6|TAC_traj_lat_correct|s1 | -0.0037 [-0.0161, +0.0077] |
| S6|TAC_traj_lon_correct|s0 | -0.0440 [-0.0782, -0.0126] SEP |
| S6|TAC_traj_lon_correct|s1 | -0.0458 [-0.0798, -0.0149] SEP |
* PRIOR PARITY max |delta| by slot (m, worst over panels): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; smallest panel 4754 windows

## printed beside (outside the condition)

* pre-ruling ha0_ext (E1 S2 / E3 S6, scored arm): E1_S2: s0: +0.0615 [+0.0472, +0.0748] SEP | s1: +0.0626 [+0.0483, +0.0762] SEP ; E3_S6: s0: -0.5870 [-0.9043, -0.2764] SEP | s1: -0.5767 [-0.9001, -0.2661] SEP
* flat 0.5 (CONST_G05) - ha0_ext_pose: E1_S2_vs_pose: s0: +0.0384 [+0.0250, +0.0512] SEP | s1: +0.0392 [+0.0255, +0.0520] SEP ; E3_S6_vs_pose: s0: -0.8849 [-1.1594, -0.6308] SEP | s1: -0.8805 [-1.1585, -0.6236] SEP
* per-surface RETIME_DEPLOY (Amendment 1's form) - ha0_ext_pose: E1_S2_vs_pose: s0: +0.0563 [+0.0416, +0.0703] SEP | s1: +0.0574 [+0.0425, +0.0717] SEP ; E3_S6_vs_pose: s0: -0.6157 [-0.9413, -0.3041] SEP | s1: -0.6054 [-0.9341, -0.2898] SEP
* RETIME_TRAINFIT (per-surface train tables) - ha0_ext_pose: E1_S2_vs_pose: s0: +0.0563 [+0.0416, +0.0703] SEP | s1: +0.0574 [+0.0425, +0.0717] SEP ; E3_S6_vs_pose: s0: -0.6105 [-0.9364, -0.2985] SEP | s1: -0.6000 [-0.9290, -0.2839] SEP
* raw weights: no FORWARD_DEPLOY8 / FORWARD_DEPLOY file for the raw-weights tree

caveats: single training seed; T1 open loop (not driving performance); the stage was found post-hoc; the E1/E3 reference was switched after the 30k reads (D-R8-E1-REF); the bootstrap answers 'another draw of EPISODES' only (not another training or inference run); the speed-obedience / vmax-zero proxy arms of the battery carry no r8_nav (unknown row); they enter no SHIP-R8 clause
## feed parity (clause 1b), per seed

* s0: status PASS, windows 4754 (tree 4754), token differs from clip on 1157, args known on 2802
* s1: status PASS, windows 4754 (tree 4754), token differs from clip on 1157, args known on 2802

## per-clip verdict (deliberate-regression arm, NEVER binding)

* step48852ema_unc: overall FAIL; per clause {'1': 'PASS', '2': 'PASS', '3': 'PASS', '4': 'PASS', '5': 'FAIL', '6': 'N/A', '7': 'PASS'}

registered per-frame table sha256: `2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80`; clip-fed table `b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee` printed beside only
