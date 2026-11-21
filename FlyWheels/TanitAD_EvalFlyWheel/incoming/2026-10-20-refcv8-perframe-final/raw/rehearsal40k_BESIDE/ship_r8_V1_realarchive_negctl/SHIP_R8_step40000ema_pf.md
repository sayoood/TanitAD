# SHIP_R8_step40000ema_pf (AMENDMENT 3, per-frame feed)

**SHIP-R8 (AMENDMENT 3) on step40000ema_pf: INCOMPLETE -- ORIENTATION ONLY (the condition's tree is step48852ema_pf)**

scored arm: `RETIME_DEPLOY8` (one call on all 8 model slots, AMENDMENT 2)

condition: SHIP-R8 holds iff, on tree `step48852ema_unc` (EMA weights), ceiling-compliant windows, arm `RETIME_TRAINFIT` with the frozen gains UNCHANGED (sha256 b208c9c1e4236075…), both inference seeds 0 and 1, paired episode-cluster bootstrap B 2000: ... Any failed clause = the stage does NOT ship and refcv8 is reported failing that bar.

amendments: AMENDMENT 1: SHIP-R8 clauses (1)–(6) are evaluated on `RETIME_DEPLOY`; `RETIME_TRAINFIT` (per-surface) is printed beside. If `RETIME_DEPLOY` cannot be computed for 48,852, SHIP-R8 reads INCOMPLETE.  AMENDMENT 2: `RETIME_DEPLOY` = ONE call per window on all 8 model slots with Amendment 1's table, prior = `ha0_ext_pose` at the same 8 slots; each surface is scored on that call's outputs at its own instants; the per-surface form is printed beside.  New clause (7) PRIOR PARITY.  AMENDMENT 3: (i) every SHIP-R8 clause and every refcv8 bar is read on rolls fed `perframe_full` (tree step 48,852 EMA, tag `step48852ema_pf`, inference seeds 0 and 1, paired episode-cluster bootstrap B 2000); (ii) the RETIME table is re-fitted by the unchanged H-R8-RETIME-TRAINFIT-1 procedure on per-frame-fed train rolls (30k EMA, A6 train139, seeds 0+1 pooled, prior ha0_ext_pose), its sha256 registered BEFORE any 48,852 pf eval roll; the clip-fed table b208c9c1... is printed beside only; (iii) new clause (1b) FEED PARITY asserted on the roll artifacts -- if it fails or cannot be shown, SHIP-R8 reads INCOMPLETE; (iv) the per-clip reads (`step48852ema_unc`) are printed beside as the deliberate-regression arm, never binding; (v) a missing clause means INCOMPLETE, never PASS.  Amendments 1 and 2 still apply.

| clause | status | reading |
|---|---|---|
| (1) FORWARD_TRAINFIT controls ALL_PASS | **INCOMPLETE** | FORWARD_TRAINFIT and FORWARD_DEPLOY8 controls ALL_PASS; the frozen gains hash is the pinned one; seeds 0 and 1; B 2000 | REGISTRATION not shown: the tree's roll artifacts carry no time |
| (1b) FEED PARITY: on the tree's own roll artifacts, both seeds, every window: the fed token equals the v9 per-frame token, `nav_cmd_clip` equals the per-clip token, and `r8_nav` is non-zero exactly where `r8_nav_known`; fails or cannot be shown -> SHIP-R8 INCOMPLETE | **INCOMPLETE** | FEED PARITY cannot be shown: s0: parity was run on 'C:/Users/Admin/refcv8_pf/battery40k_pf/step40000ema_pf/dump_s0', not on the tree's dump c:/users/admin/refcv8_eval_archive/battery/step40000ema_pf/dump_s0; s1: parity was run on 'C:/Users/Admin/refcv8_pf/battery40k_pf/step40000ema_pf/dump_s1', not on the tree's dump c:/users/admin/refcv8_eval_archive/battery/step40000ema_pf/dump_s1 |
| (2) E1 S2 ADE − `ha0_ext_pose` separated below 0 at both seeds | **PASS** | E1: RETIME_DEPLOY8 - ha0_ext_pose ADE on S2 is separated below 0 at both inference seeds |
| (3) E3 S6 ADE − `ha0_ext_pose` separated below 0 at both seeds | **PASS** | E3: RETIME_DEPLOY8 - ha0_ext_pose ADE on S6 is separated below 0 at both inference seeds |
| (4) LONGITUDINAL / LATERAL / ADE no row separated worse than refcv7-50,400 at both seeds (FZR-NR), and S6 lateral HOLDS (FZR-LAT) | **PASS** | FZR-NR ADE / LONGITUDINAL / LATERAL all PASS (no headline row separated worse than refcv7-50,400 at either seed) and FZR-LAT HOLDS |
| (5) TACTICAL `TAC_traj_lat_correct` and `TAC_traj_lon_correct` of the RE-TIMED plan not separated worse than refcv7-50,400 at both seeds (instrument being built 2026-10-19; if it cannot be computed, SHIP-R8 reads INCOMPLETE, not PASS) | **PASS** | no TAC_traj_lat / TAC_traj_lon cell (S2, S6 x seeds 0, 1) of RETIME_DEPLOY8 is separated worse than refcv7-50,400 |
| (6) STRATEGIC not applicable, n = 0, `--no-strategic` | **N/A** | UNVERIFIED against a battery file (PAIRED_READ not found); the condition itself pins clause (6) as not applicable, n = 0 |
| (7) PRIOR PARITY: the prior the eval rebuilds from banked poses must equal the model's emitted `residual_prior_path` (built in `RefCModel.forward` via `kinematic_prior.prior_controls('ha0_ext_pose')`) to ≤ 1e-3 m at all 8 slots on ≥ 200 eval windows; if it cannot be shown, SHIP-R8 reads INCOMPLETE | **PASS** | max |delta| <= 0.001 m at all 8 slots on >= 200 windows (smallest panel 4754; 2 panel(s)) |

## evidence

* E1 (S2) RETIME_DEPLOY8 - ha0_ext_pose: s0: -0.0390 [-0.0507, -0.0278] SEP | s1: -0.0377 [-0.0491, -0.0264] SEP
* E3 (S6) RETIME_DEPLOY8 - ha0_ext_pose: s0: -1.0046 [-1.3211, -0.6989] SEP | s1: -0.9993 [-1.3097, -0.6967] SEP
* FZR-NR per family: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}; FZR-LAT HOLDS True

TACTICAL cells (RETIME_DEPLOY8 - refcv7-50,400, accuracy, compliant windows; worse+separated = CI upper < 0):

| cell | delta [lo, hi] |
|---|---|
| S2|TAC_traj_lat_correct|s0 | +0.0048 [+0.0005, +0.0093] SEP |
| S2|TAC_traj_lat_correct|s1 | +0.0043 [+0.0005, +0.0083] SEP |
| S2|TAC_traj_lon_correct|s0 | +0.0245 [+0.0116, +0.0382] SEP |
| S2|TAC_traj_lon_correct|s1 | +0.0156 [+0.0029, +0.0287] SEP |
| S6|TAC_traj_lat_correct|s0 | +0.0123 [+0.0009, +0.0235] SEP |
| S6|TAC_traj_lat_correct|s1 | +0.0102 [-0.0010, +0.0209] |
| S6|TAC_traj_lon_correct|s0 | +0.0111 [-0.0118, +0.0327] |
| S6|TAC_traj_lon_correct|s1 | +0.0046 [-0.0181, +0.0261] |
* PRIOR PARITY max |delta| by slot (m, worst over panels): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; smallest panel 4754 windows

## printed beside (outside the condition)

* pre-ruling ha0_ext (E1 S2 / E3 S6, scored arm): E1_S2: s0: -0.0339 [-0.0449, -0.0232] SEP | s1: -0.0326 [-0.0436, -0.0216] SEP ; E3_S6: s0: -0.9758 [-1.2892, -0.6764] SEP | s1: -0.9705 [-1.2788, -0.6756] SEP
* flat 0.5 (CONST_G05) - ha0_ext_pose: E1_S2_vs_pose: s0: -0.0387 [-0.0497, -0.0285] SEP | s1: -0.0378 [-0.0484, -0.0274] SEP ; E3_S6_vs_pose: s0: -1.1044 [-1.3812, -0.8466] SEP | s1: -1.1061 [-1.3771, -0.8540] SEP
* per-surface RETIME_DEPLOY (Amendment 1's form) - ha0_ext_pose: E1_S2_vs_pose: s0: -0.0391 [-0.0507, -0.0279] SEP | s1: -0.0378 [-0.0491, -0.0264] SEP ; E3_S6_vs_pose: s0: -1.0045 [-1.3211, -0.6989] SEP | s1: -0.9992 [-1.3095, -0.6966] SEP
* RETIME_TRAINFIT (per-surface train tables) - ha0_ext_pose: E1_S2_vs_pose: s0: -0.0391 [-0.0507, -0.0279] SEP | s1: -0.0378 [-0.0491, -0.0264] SEP ; E3_S6_vs_pose: s0: -1.0036 [-1.3203, -0.6974] SEP | s1: -0.9983 [-1.3087, -0.6957] SEP
* raw weights: no FORWARD_DEPLOY8 / FORWARD_DEPLOY file for the raw-weights tree

caveats: single training seed; T1 open loop (not driving performance); the stage was found post-hoc; the E1/E3 reference was switched after the 30k reads (D-R8-E1-REF); the bootstrap answers 'another draw of EPISODES' only (not another training or inference run); the speed-obedience / vmax-zero proxy arms of the battery carry no r8_nav (unknown row); they enter no SHIP-R8 clause
## feed parity (clause 1b), per seed

* s0: status PASS, windows 4754 (tree None), token differs from clip on 1157, args known on 2802
* s1: status PASS, windows 4754 (tree None), token differs from clip on 1157, args known on 2802

## per-clip verdict (deliberate-regression arm, NEVER binding)

* step48852ema_unc: overall NOT YET COMPUTED (the per-clip chain runs after the pf GPU stage); per clause None

registered per-frame table sha256: `2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80`; clip-fed table `b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee` printed beside only
