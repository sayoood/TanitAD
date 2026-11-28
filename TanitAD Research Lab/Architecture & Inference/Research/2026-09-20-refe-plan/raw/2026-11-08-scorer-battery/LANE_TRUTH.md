# LANE_TRUTH: can the teacher centre-line distance be joined to the C3 table on CPU? (2026-11-08)

**Answer: reachable on CPU in principle; NOT built and NOT run here. Items are marked MEASURED (checked today) or UNVERIFIED.**

The truth: `refe/onpolicy_relabel_teacher_lane.py` (SFT-4's 7th target) keeps the teacher Scorer's `center_line.CenterLine.info` (distance to the centreline, m, max over the horizon) and `.reward` (1.0 centred to 0.75 at the line), plus `off_road.CrossLane.*`, per proposal, from `onpolicy_label.Scorer` (DriveRL calculators). Output: side files keyed (log, token, step, rank).
Why the C3 table lacks it: `table.npz` carries only the 6 PDMS sub-scores (NC, DAC, EP, TTC, C, DDC).

MEASURED today on the dev box:
- all 123 navtest logs of the C3 table have their nuPlan DB at `E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits` (123 of 123); maps at `E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0`.
- ⚠ `_tlane_work/devenv.sh` and `navtrain_scenarios.DEFAULT_DB_ROOT` point at `D:/Projects/TanitAD/data/...`, which does not exist now (drive-letter flip). Set `REFE_NUPLAN_DB_ROOT`, `NUPLAN_DATA_ROOT`, `NUPLAN_MAPS_ROOT` to the E: paths.
- DriveRL teacher code and checkpoints exist at `C:/Users/Admin/dz/DriveZero/DriveRL`; `driverl-eval` is the venv that ran the 20-row smoke (`E:/Projects/TanitAD-artifacts/_tlane_work/smoke`, 2 m 13 s wall for 20 rows including start-up; the per-row cost at scale is NOT measured).
- the 64 u400 proposals at 20 steps are in `refe_ego_probe_20261106/full/results.npz` (`traj_all_A0`, 0.2 s grid); `table.npz` `proposals` has only 8 steps.

UNVERIFIED / not found:
- the Scorer takes queue rows carrying `props` AND a `teacher` rollout per token (`label_log` in the script). I found no navtest queue rows with a teacher rollout. Building them for 1,058 tokens (the teacher on CPU, `DRIVERL_EVAL_DEVICE=cpu`) is the missing piece.
- the table's `pdms` came from the navsim PDM scorer; the lane leaves come from the DriveRL calculators (a different simulator). The join is by (token, proposal rank) and needs a consistency check on DAC/DDC (the script writes `off_road.OffRoad.info`, `ddc.violation`, `dac.violation` for exactly that).
- run time: certainly above the 4-minute rule, so it is a chained job with a handoff.

Proposed next WP "LANE-TRUTH-C3": build the 1,058-token queue (props from `traj_all_A0`, teacher rollout on CPU), run `onpolicy_relabel_teacher_lane.py` sharded on the dev box, join on (token, rank), assert the DAC/DDC consistency leaves, then add the centre-line distance as a seventh true sub-score; `spearman_block` already reserves the `lane` row.
