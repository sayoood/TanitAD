# REFe's ego inputs 4 (yaw_rate) and 5 (tire_steering_angle) were never trained (2026-11-06, Master Mind)

**Verdict (MEASURED, two independent probes).** Every REFe checkpoint trained on the nuPlan/NAVSIM bank (P2 `model_final.pt`, SFT-4 `model_sft4_B_u400.pt`, and, by the same bank code, P6 and P7) has two **dead** ego inputs:
slot 4 `yaw_rate` and slot 5 `tire_steering_angle` were **exactly 0.0 in every training sample**, so their first-layer weights never received a gradient.
A client that feeds non-zero values there drives random-init weights into the ego token.

## Probe 1: source (why the values are zero)
- `nuplan/database/nuplan_db/nuplan_scenario_queries.py:551,594` (nuplan-devkit, `F:/Archive/devbox-C/navsim/nuplan-devkit`): the DB ego state is built with `tire_steering_angle=0.0` hardcoded.
- `nuplan/common/actor_state/ego_state.py:223`: `build_from_rear_axle(..., angular_vel: float = 0.0, ...)`; the DB query never passes it.
  The DB *does* hold a measured yaw rate (`nuplan_db_orm/ego_pose.py:51`, `angular_rate_z`), but this path does not read it.
- REFe banks exactly those fields: `refe/build_targets.py:226-228` and `refe/build_teacher_rollouts.py:383`, through `NuPlanScenario` (`refe/navtrain_scenarios.py:232`).

## Probe 2: the trained weights (what the model actually saw)
Rebuilt REFe at its training init (`refe/train.py:1174-1175`: `torch.manual_seed(0); REFe(cfg)`; cfg from the checkpoint's meta via `ckpt_io.config_for_checkpoint`) and compared `ego_enc.0.weight` [256, 43] column by column with P2 `model_final.pt` (sha256 `4246a320...`, 10,075 steps). Logs: `proof_run.log`, `shrink_residual.log`.

| column | W / W0 (min .. max) | max \|W - c*W0\| | reading |
|---|---|---|---|
| yaw_rate (4) | 0.989766 .. 0.990023 | 2.3e-05 | pure shrink, c = 0.989963 |
| steer (5) | 0.989767 .. 0.990019 | 2.5e-05 | pure shrink, the SAME c |
| vx / vy / ax / ay / speed (controls) | spread 34 .. 252 | 0.10 .. 0.32 | trained |
| goal0 (control) | -0.93 .. 6.19 | 0.045 | trained |

- A pure uniform shrink is AdamW's decoupled weight decay with zero gradient. Both dead columns carry the same shrink factor, as they would if only the decay schedule acted on them.
- A non-zero gradient cannot hide here: Adam normalises, so any step with a non-zero gradient moves an entry by about the learning rate, far above the 2.5e-5 residual. The rebuild is validated by the controls, which differ from init by 0.1–0.3.
- SFT-4 never touched `ego_enc`: u400's `ego_enc.0.weight` is bit-identical to `model_final.pt`'s (`proof_run.log`).

## What it changes
1. **Closed-loop clients must feed 0.0 in slots 4 and 5.** TanitScenarioFlyWheel's T1 client fed `yaw_rate = v*tan(steer)/2.9` and pure-pursuit `steer` (up to ±0.6 rad), which injects untrained weights into the ego token during every turn. The fix has been sent to that session.
   - Our own `refe/planner.py:448-451` passes `angular_velocity` and `tire_steering_angle` from whatever `EgoState` it is given.
   - In NAVSIM navtest those come from the logs, so they are zero and **no banked navtest number is affected**.
   - In a nuPlan closed-loop simulation the simulator's state is non-zero, so that use would need the same zeroing.
   - `planner.py` is pinned by the P6 read instrument and is NOT edited here.
2. **The register's `D-PI-YAWRATE-3` premise is corrected.** That row treats slot 5 as nuPlan's measured steering, and says "yaw rate is already slot 4" carries measured motion. Neither holds: slot 5 carried no information at all, and REFe has **no yaw-rate input**.
3. **A lever, not applied: a real yaw-rate input.** Populate slot 4 from nuPlan's measured `ego_pose.angular_rate_z`, and keep slot 5 at 0 (compliant with `D-PI-YAWRATE-1`). This needs a bank rebuild and changes the paper-faithful recipe, so it is a PI decision for the run after P7. P6 and P7 keep the dead slots.

## Not covered here
- Whether REFe's speed behaviour in closed loop (the "crawl") is a policy property. That is the separate T0 input-reliance probe `E:/Projects/TanitAD-artifacts/refe_ego_probe_20261106/` (arms ax=0, ±0.5, dead-slot values, grey and swapped images).

Scripts: `init_match.py` (rebuild init, column diffs), `decay_ratio.py` (W/W0), `shrink_residual.py` (residual from a pure shrink), `steer_col.py` (u400 vs model_final). CPU only (`CUDA_VISIBLE_DEVICES=-1`), venv `C:/Users/Admin/venvs/tanitad`; model code `refe/model.py` md5 `08d7c216`, `refe/ckpt_io.py` md5 `bc3cb24e`.
