# SHIP_R8_step48852ema_unc

**SHIP-R8 VERDICT on the condition's tree step48852ema_unc: FAIL**

scored arm: `RETIME_DEPLOY8` (one call on all 8 model slots, AMENDMENT 2)

condition: SHIP-R8 holds iff, on tree `step48852ema_unc` (EMA weights), ceiling-compliant windows, arm `RETIME_TRAINFIT` with the frozen gains UNCHANGED (sha256 b208c9c1e4236075…), both inference seeds 0 and 1, paired episode-cluster bootstrap B 2000: ... Any failed clause = the stage does NOT ship and refcv8 is reported failing that bar.

amendments: AMENDMENT 1: SHIP-R8 clauses (1)–(6) are evaluated on `RETIME_DEPLOY`; `RETIME_TRAINFIT` (per-surface) is printed beside. If `RETIME_DEPLOY` cannot be computed for 48,852, SHIP-R8 reads INCOMPLETE.  AMENDMENT 2: `RETIME_DEPLOY` = ONE call per window on all 8 model slots with Amendment 1's table, prior = `ha0_ext_pose` at the same 8 slots; each surface is scored on that call's outputs at its own instants; the per-surface form is printed beside.  New clause (7) PRIOR PARITY.

| clause | status | reading |
|---|---|---|
| (1) FORWARD_TRAINFIT controls ALL_PASS | **PASS** | FORWARD_TRAINFIT and FORWARD_DEPLOY8 controls ALL_PASS; the frozen gains hash is the pinned one; seeds 0 and 1; B 2000 |
| (2) E1 S2 ADE − `ha0_ext_pose` separated below 0 at both seeds | **PASS** | E1: RETIME_DEPLOY8 - ha0_ext_pose ADE on S2 is separated below 0 at both inference seeds |
| (3) E3 S6 ADE − `ha0_ext_pose` separated below 0 at both seeds | **PASS** | E3: RETIME_DEPLOY8 - ha0_ext_pose ADE on S6 is separated below 0 at both inference seeds |
| (4) LONGITUDINAL / LATERAL / ADE no row separated worse than refcv7-50,400 at both seeds (FZR-NR), and S6 lateral HOLDS (FZR-LAT) | **PASS** | FZR-NR ADE / LONGITUDINAL / LATERAL all PASS (no headline row separated worse than refcv7-50,400 at either seed) and FZR-LAT HOLDS |
| (5) TACTICAL `TAC_traj_lat_correct` and `TAC_traj_lon_correct` of the RE-TIMED plan not separated worse than refcv7-50,400 at both seeds (instrument being built 2026-10-19; if it cannot be computed, SHIP-R8 reads INCOMPLETE, not PASS) | **FAIL** | separated worse than refcv7-50,400: S6|TAC_traj_lon_correct|s0, S6|TAC_traj_lon_correct|s1 |
| (6) STRATEGIC not applicable, n = 0, `--no-strategic` | **N/A** | the battery's strategic block reads NOT APPLICABLE, n = 0 (--no-strategic): satisfied as the condition states |
| (7) PRIOR PARITY: the prior the eval rebuilds from banked poses must equal the model's emitted `residual_prior_path` (built in `RefCModel.forward` via `kinematic_prior.prior_controls('ha0_ext_pose')`) to ≤ 1e-3 m at all 8 slots on ≥ 200 eval windows; if it cannot be shown, SHIP-R8 reads INCOMPLETE | **PASS** | max |delta| <= 0.001 m at all 8 slots on >= 200 windows (smallest panel 4754; 4 panel(s)) |

## evidence

* E1 (S2) RETIME_DEPLOY8 - ha0_ext_pose: s0: -0.0256 [-0.0393, -0.0123] SEP | s1: -0.0262 [-0.0391, -0.0139] SEP
* E3 (S6) RETIME_DEPLOY8 - ha0_ext_pose: s0: -0.8390 [-1.1617, -0.5377] SEP | s1: -0.8431 [-1.1569, -0.5533] SEP
* FZR-NR per family: {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}; FZR-LAT HOLDS True

TACTICAL cells (RETIME_DEPLOY8 - refcv7-50,400, accuracy, compliant windows; worse+separated = CI upper < 0):

| cell | delta [lo, hi] |
|---|---|
| S2|TAC_traj_lat_correct|s0 | -0.0019 [-0.0063, +0.0026] |
| S2|TAC_traj_lat_correct|s1 | -0.0046 [-0.0099, +0.0005] |
| S2|TAC_traj_lon_correct|s0 | +0.0228 [+0.0080, +0.0385] SEP |
| S2|TAC_traj_lon_correct|s1 | +0.0171 [+0.0031, +0.0317] SEP |
| S6|TAC_traj_lat_correct|s0 | -0.0055 [-0.0213, +0.0097] |
| S6|TAC_traj_lat_correct|s1 | -0.0065 [-0.0221, +0.0078] |
| S6|TAC_traj_lon_correct|s0 | -0.0332 [-0.0663, -0.0034] SEP |
| S6|TAC_traj_lon_correct|s1 | -0.0338 [-0.0655, -0.0051] SEP |
* PRIOR PARITY max |delta| by slot (m, worst over panels): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; smallest panel 4754 windows

## printed beside (outside the condition)

* pre-ruling ha0_ext (E1 S2 / E3 S6, scored arm): E1_S2: s0: -0.0205 [-0.0336, -0.0074] SEP | s1: -0.0211 [-0.0333, -0.0093] SEP ; E3_S6: s0: -0.8103 [-1.1291, -0.5176] SEP | s1: -0.8143 [-1.1250, -0.5284] SEP
* flat 0.5 (CONST_G05) - ha0_ext_pose: E1_S2_vs_pose: s0: -0.0260 [-0.0391, -0.0130] SEP | s1: -0.0266 [-0.0389, -0.0146] SEP ; E3_S6_vs_pose: s0: -0.9522 [-1.2418, -0.6872] SEP | s1: -0.9512 [-1.2360, -0.6960] SEP
* per-surface RETIME_DEPLOY (Amendment 1's form) - ha0_ext_pose: E1_S2_vs_pose: s0: -0.0257 [-0.0393, -0.0123] SEP | s1: -0.0262 [-0.0392, -0.0140] SEP ; E3_S6_vs_pose: s0: -0.8389 [-1.1615, -0.5376] SEP | s1: -0.8429 [-1.1567, -0.5532] SEP
* RETIME_TRAINFIT (per-surface train tables) - ha0_ext_pose: E1_S2_vs_pose: s0: -0.0257 [-0.0393, -0.0123] SEP | s1: -0.0262 [-0.0392, -0.0140] SEP ; E3_S6_vs_pose: s0: -0.8380 [-1.1610, -0.5364] SEP | s1: -0.8419 [-1.1563, -0.5521] SEP
* raw weights: no FORWARD_DEPLOY8 / FORWARD_DEPLOY file for the raw-weights tree

caveats: single training seed; T1 open loop (not driving performance); the stage was found post-hoc; the E1/E3 reference was switched after the 30k reads (D-R8-E1-REF); the train gains under-correct on eval (H-R8-GAINTRAJ-1); the bootstrap answers 'another draw of EPISODES' only (not another training or inference run)
