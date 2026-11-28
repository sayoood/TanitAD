## Amendment YAW (2026-11-07, Master Mind; PI go 2026-11-06: "do 1 and 2"). Written BEFORE P7's launch and before any P7 step

**Why.**
- REFe's ego slot 4 (`yaw_rate`) is 0.0 in every training sample of P2, SFT-4 and P6 (`D-REFE-DEADSLOTS-1`). nuPlan's DB loader never reads `ego_pose.angular_rate_z`.
- The measured yaw rate exists:
  - MEASURED (stream YT, `E:/Projects/TanitAD-artifacts/yawrate_wp/evidence/`): corr 0.9988 and slope 1.007 against d(heading)/dt over 12 logs / 439,815 points, CCW-positive, rad/s.
- The PI ruling `D-PI-YAWRATE-1` admits measured ego motion including the yaw rate, never the steering angle.

**Change (one lever, added to §0–§3).**
1. **Input.** `refe/train.py --ego-yaw-table <navtrain table>` sets `ego[4] = YAW_SCALE * angular_rate_z(token)`.
   - `YAW_SCALE = 10.0` is one constant in `refe/ego_yaw_table.py`. REFe's ego inputs are raw; the unscaled yaw is 0.18 % of the `ego_enc` pre-activation spread (stream YF).
   - Slot 5 (steer) stays 0.
   - A missing token, or a bank row whose ego[4] is already non-zero, REFUSES at construction.
2. **Table.** Built ON THE POD by the P7 chain AFTER P6 releases its lock (no CPU load beside P6 training), from `$DATA/navtrain_dbs`, by `ego_yaw_table.py build`.
   - The build uses the same lidar_pc -> ego_pose join as nuPlan's loader. Control: vx/vy/ax/ay bit-equal to `EgoState` and to the bank's ego[0..3].
   - Then `check-bank` must report **100 % coverage of the P7 bank's tokens**; anything less REFUSES.
   - The table's sha256 enters the launch token's data manifest and the checkpoint meta: `ego_yaw`, `ego_yaw_scale`, `ego_yaw_table_sha256`, `ego_yaw_slot`.
3. **Gate.**
   - G-DVB carries the flag.
   - G-LIVE asserts a NON-ZERO gradient on `ego_enc.0.weight[:, 4]`, with the control reading exactly 0.0 without the flag.
   - The deliberate-regression arms `yaw_override_removed` and `yaw_table_not_passed` must FAIL.
   - The refe_p7 profile REQUIRES the flag.

**Endpoint consequence.**
- P7 is a yaw-trained model, so every navtest read of P7 feeds the navtest yaw table (`f5ebe24f...`, 12,146/12,146 tokens) through `refe/ego_yaw.py`'s launcher. Feeding 0 would be out of distribution and is not a P7 number.
- P6's read is unaffected: its harness and `refe/planner.py` stay byte-identical.

**Attribution (stated, not hidden).**
- P7 − P6 is now a bundle of four changes: the head, the reserve's 9.84 % fewer scenes, the yaw input, and noise. §4's bundle claim widens accordingly; P7's ADOPT / REFUTED rule (§5) is unchanged.
- The yaw input's own effect is attributed by the separate fine-tune arms B vs C of `PREREG_YAWFT` (stream YF2). The mechanism endpoint added here, reported and never gating, is the LATERAL family's curvature and yaw-rate errors vs P6.

**Not changed.** Every other flag, pin, bank, reserve rule, kill and comparator in §0–§5.
