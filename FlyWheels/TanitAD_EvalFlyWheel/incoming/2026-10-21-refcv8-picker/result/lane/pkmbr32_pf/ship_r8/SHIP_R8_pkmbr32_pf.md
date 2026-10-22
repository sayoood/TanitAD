# SHIP_R8_pkmbr32_pf (AMENDMENT 3, per-frame feed)

**SHIP-R8 (AMENDMENT 3) on pkmbr32_pf: PASS -- ORIENTATION ONLY (the condition's tree is step48852ema_pf)**

scored arm: `RETIME_DEPLOY8` (one call on all 8 model slots, AMENDMENT 2)

condition: SHIP-R8 holds iff, on tree `step48852ema_unc` (EMA weights), ceiling-compliant windows, arm `RETIME_TRAINFIT` with the frozen gains UNCHANGED (sha256 b208c9c1e4236075…), both inference seeds 0 and 1, paired episode-cluster bootstrap B 2000: ... Any failed clause = the stage does NOT ship and refcv8 is reported failing that bar.

amendments: AMENDMENT 1: SHIP-R8 clauses (1)–(6) are evaluated on `RETIME_DEPLOY`; `RETIME_TRAINFIT` (per-surface) is printed beside. If `RETIME_DEPLOY` cannot be computed for 48,852, SHIP-R8 reads INCOMPLETE.  AMENDMENT 2: `RETIME_DEPLOY` = ONE call per window on all 8 model slots with Amendment 1's table, prior = `ha0_ext_pose` at the same 8 slots; each surface is scored on that call's outputs at its own instants; the per-surface form is printed beside.  New clause (7) PRIOR PARITY.  AMENDMENT 3: (i) every SHIP-R8 clause and every refcv8 bar is read on rolls fed `perframe_full` (tree step 48,852 EMA, tag `step48852ema_pf`, inference seeds 0 and 1, paired episode-cluster bootstrap B 2000); (ii) the RETIME table is re-fitted by the unchanged H-R8-RETIME-TRAINFIT-1 procedure on per-frame-fed train rolls (30k EMA, A6 train139, seeds 0+1 pooled, prior ha0_ext_pose), its sha256 registered BEFORE any 48,852 pf eval roll; the clip-fed table b208c9c1... is printed beside only; (iii) new clause (1b) FEED PARITY asserted on the roll artifacts -- if it fails or cannot be shown, SHIP-R8 reads INCOMPLETE; (iv) the per-clip reads (`step48852ema_unc`) are printed beside as the deliberate-regression arm, never binding; (v) a missing clause means INCOMPLETE, never PASS.  Amendments 1 and 2 still apply.

| clause | status | reading |
|---|---|---|
| (1) FORWARD_TRAINFIT controls ALL_PASS | **PASS** | FORWARD_TRAINFIT and FORWARD_DEPLOY8 controls ALL_PASS; the frozen gains hash is the pinned one; seeds 0 and 1; B 2000 | the per-frame table's sha256 was registered before every pf roll of the tree (Amendment 3 (ii)) |
| (1b) FEED PARITY: on the tree's own roll artifacts, both seeds, every window: the fed token equals the v9 per-frame token, `nav_cmd_clip` equals the per-clip token, and `r8_nav` is non-zero exactly where `r8_nav_known`; fails or cannot be shown -> SHIP-R8 INCOMPLETE | **PASS** | both seeds, every window of the tree's dumps: fed token == v9 per-frame token, nav_cmd_clip == per-clip token, r8_nav non-zero exactly where known |
| (2) E1 S2 ADE − `ha0_ext_pose` separated below 0 at both seeds | **PASS** | E1: RETIME_DEPLOY8 - ha0_ext_pose ADE on S2 is separated below 0 at both inference seeds |
| (3) E3 S6 ADE − `ha0_ext_pose` separated below 0 at both seeds | **PASS** | E3: RETIME_DEPLOY8 - ha0_ext_pose ADE on S6 is separated below 0 at both inference seeds |
| (4) LONGITUDINAL / LATERAL / ADE no row separated worse than refcv7-50,400 at both seeds (FZR-NR), and S6 lateral HOLDS (FZR-LAT) | **PASS** | FZR-NR ADE / LONGITUDINAL / LATERAL all PASS (no headline row separated worse than refcv7-50,400 at either seed) and FZR-LAT HOLDS |
| (5) TACTICAL `TAC_traj_lat_correct` and `TAC_traj_lon_correct` of the RE-TIMED plan not separated worse than refcv7-50,400 at both seeds (instrument being built 2026-10-19; if it cannot be computed, SHIP-R8 reads INCOMPLETE, not PASS) | **PASS** | no TAC_traj_lat / TAC_traj_lon cell (S2, S6 x seeds 0, 1) of RETIME_DEPLOY8 is separated worse than refcv7-50,400 |
| (6) STRATEGIC not applicable, n = 0, `--no-strategic` | **N/A** | the battery's strategic block reads NOT APPLICABLE, n = 0 (--no-strategic): satisfied as the condition states |
| (7) PRIOR PARITY: the prior the eval rebuilds from banked poses must equal the model's emitted `residual_prior_path` (built in `RefCModel.forward` via `kinematic_prior.prior_controls('ha0_ext_pose')`) to ≤ 1e-3 m at all 8 slots on ≥ 200 eval windows; if it cannot be shown, SHIP-R8 reads INCOMPLETE | **PASS** | max |delta| <= 0.001 m at all 8 slots on >= 200 windows (smallest panel 4754; 2 panel(s)) |

## evidence

* E1 (S2) RETIME_DEPLOY8 - ha0_ext_pose: s0: -0.0439 [-0.0546, -0.0337] SEP | s1: -0.0436 [-0.0540, -0.0334] SEP
* E3 (S6) RETIME_DEPLOY8 - ha0_ext_pose: s0: -1.0844 [-1.3899, -0.7834] SEP | s1: -1.0792 [-1.3819, -0.7819] SEP
* FZR-NR per family: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}; FZR-LAT HOLDS True

TACTICAL cells (RETIME_DEPLOY8 - refcv7-50,400, accuracy, compliant windows; worse+separated = CI upper < 0):

| cell | delta [lo, hi] |
|---|---|
| S2|TAC_traj_lat_correct|s0 | +0.0031 [-0.0020, +0.0078] |
| S2|TAC_traj_lat_correct|s1 | +0.0019 [-0.0026, +0.0065] |
| S2|TAC_traj_lon_correct|s0 | +0.0344 [+0.0205, +0.0499] SEP |
| S2|TAC_traj_lon_correct|s1 | +0.0277 [+0.0138, +0.0432] SEP |
| S6|TAC_traj_lat_correct|s0 | +0.0086 [-0.0032, +0.0212] |
| S6|TAC_traj_lat_correct|s1 | +0.0034 [-0.0090, +0.0155] |
| S6|TAC_traj_lon_correct|s0 | +0.0181 [-0.0067, +0.0402] |
| S6|TAC_traj_lon_correct|s1 | +0.0080 [-0.0144, +0.0288] |
* PRIOR PARITY max |delta| by slot (m, worst over panels): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; smallest panel 4754 windows

## printed beside (outside the condition)

* pre-ruling ha0_ext (E1 S2 / E3 S6, scored arm): E1_S2: s0: -0.0387 [-0.0487, -0.0289] SEP | s1: -0.0385 [-0.0487, -0.0290] SEP ; E3_S6: s0: -1.0557 [-1.3551, -0.7615] SEP | s1: -1.0504 [-1.3466, -0.7579] SEP
* flat 0.5 (CONST_G05) - ha0_ext_pose: E1_S2_vs_pose: s0: -0.0418 [-0.0519, -0.0322] SEP | s1: -0.0416 [-0.0516, -0.0321] SEP ; E3_S6_vs_pose: s0: -1.1154 [-1.3761, -0.8654] SEP | s1: -1.1123 [-1.3734, -0.8590] SEP
* per-surface RETIME_DEPLOY (Amendment 1's form) - ha0_ext_pose: E1_S2_vs_pose: s0: -0.0439 [-0.0546, -0.0338] SEP | s1: -0.0437 [-0.0541, -0.0334] SEP ; E3_S6_vs_pose: s0: -1.0844 [-1.3897, -0.7834] SEP | s1: -1.0791 [-1.3819, -0.7818] SEP
* RETIME_TRAINFIT (per-surface train tables) - ha0_ext_pose: E1_S2_vs_pose: s0: -0.0439 [-0.0546, -0.0338] SEP | s1: -0.0437 [-0.0541, -0.0334] SEP ; E3_S6_vs_pose: s0: -1.0842 [-1.3900, -0.7831] SEP | s1: -1.0790 [-1.3820, -0.7816] SEP
* raw weights: no FORWARD_DEPLOY8 / FORWARD_DEPLOY file for the raw-weights tree

caveats: single training seed; T1 open loop (not driving performance); the stage was found post-hoc; the E1/E3 reference was switched after the 30k reads (D-R8-E1-REF); the bootstrap answers 'another draw of EPISODES' only (not another training or inference run); the speed-obedience / vmax-zero proxy arms of the battery carry no r8_nav (unknown row); they enter no SHIP-R8 clause
## feed parity (clause 1b), per seed

* s0: status PASS, windows 4754 (tree 4754), token differs from clip on 1157, args known on 2802
* s1: status PASS, windows 4754 (tree 4754), token differs from clip on 1157, args known on 2802

## per-clip verdict (deliberate-regression arm, NEVER binding)

* step48852ema_unc: overall FAIL; per clause {'1': 'PASS', '2': 'PASS', '3': 'PASS', '4': 'PASS', '5': 'FAIL', '6': 'N/A', '7': 'PASS'}

registered per-frame table sha256: `2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80`; clip-fed table `b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee` printed beside only
