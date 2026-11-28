# T2-A inventory — what a closed-loop (T2) eval of refcv7-50,400 needs, and where it is

Evidence classes: `MEASURED (path)` = measured on the dev box today, artifact named; `INHERITED` = read from another document,
not re-verified; `UNVERIFIED` = nobody has run it; `ESTIMATED` = arithmetic.
Tier stamp for everything this package will emit: **T2** (perception-closed loop). Scope sentence that travels with every number:
*the loop is closed on PERCEPTION, not on INTERACTION* (nothing in the scene responds to the ego; no map; no collision response).
Tier definition: `Project Steering/EVAL_DOCTRINE.md`. Feasibility study: `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-hjepa-literature/T2_FEASIBILITY.md` (INHERITED).

## 1. The harness

`stack/experiments/alpasim-gsplat/closedloop_drive.py` (in-house NuRec/gsplat, NOT AlpaSim). One tick =
render (gsplat f-theta, 1920x1080) -> canonicalise -> `policy.plan(frames, intr, v0, nav_cmd, nav_known)` -> `wp_to_control`
(pure pursuit on the 0.5 s waypoint, steer clamp 0.05 rad) -> kinematic bicycle -> `GroundFollower` -> re-render.
Banked config used by every Thor number and reused here unchanged: 10 Hz, 50 ticks, 9 starts (0,17,...,136), warm-up of 10
logged frames, `--cull-scale-quantile 0.95 --sky-gain 0.3`, `empty` condition (MEASURED: source read; `code/run_on_box2.py` carries the flags).

**Modification made (unstaged, one file):** `closedloop_drive.py` +35/-1 lines — an `EXTRA_ARMS` registry, and a
`set_history(pose_hist)` hook fed from a 16-entry deque of (x,y,yaw,v). The hook is inert for every historical policy class: the 29
pre-existing tests in `stack/tests/test_closedloop_floor.py` passed after the edit (MEASURED earlier in this session); the new suite is
`code/tests/test_t2a_dry.py`, 20 fast + 1 slow, all green (`raw/pytest_fast_final.log`, `raw/refcv7_cpu_tick.log`).

## 2. Inputs probed

| need | path probed | found | size | class |
|---|---|---|---|---|
| refcv7-50,400 checkpoint | `E:/refcv7_eval_kit/ckpt/ckpt_50400.pt` | yes, md5 `b418d0fc4a92a6848c246a6a7c50207b` | 402,708,019 B | MEASURED |
| its config | `E:/refcv7_eval_kit/ckpt/config.json` | yes | 66,500 B | MEASURED |
| scene 1 (`00040136-e651-4abd-991d-0655ccda9430`) | `E:/thor_archive/2026-09-14/nurec_scenes/sample_set/26.04_release/<uuid>/volume.nurec` + `rig_trajectories.json` | yes | 639,393,135 + 2,155,480 B | MEASURED |
| scene 2 (`7c72937c-c620-4776-9555-d57222c0081f`) | same root; rig json under `extracted/` | yes | 603,238,216 + 2,136,806 B | MEASURED |
| P10 floor weights | `C:/Users/Admin/s1_floors_scratch/` (before today) -> now `assets/p10_S2.pt` | yes | 21,011 B | MEASURED |
| refcv6-kit data the refcv7 loader reads | `D:/refcv6_eval_kit/data/` (4 files) | yes | 13,477 + 3,046 + 2,410 + 3,367,838 B | MEASURED |
| code at the launch commit | `git archive fec3a0dc stack taniteval` | yes (≈29 MB pruned) | — | MEASURED |
| control: refc-base ckpt + config | `E:/thor_archive/2026-09-14/models/refc-base/` | yes | 1,250,838,325 + 3,174 B | MEASURED |
| control: banked Thor panel | `stack/experiments/alpasim-gsplat/results/closedloop-hq-render/rollouts/HQ_refc-base_empty.json` | yes | 620,303 B | MEASURED |

Not needed: the 186 MB vis1 sidecar and the 703 MB joins. `refcv7_loader.build_model` builds strictly from `config.json` plus the four
small kit files (MEASURED, `raw/cpu_model_smoke.json`).

**Working-tree trap (MEASURED):** the working-tree branch `agent/arch-inf-20260803` cannot build refcv7 (no `kinematic_prior.py`, no
`refcv7_loader`). The kit pins the launch commit `fec3a0dc` via `git archive`, plus an overlay of the patched harness and of
`floors_kin.py` (which is **untracked** in the working tree).

## 3. refcv7 contract with the harness (MEASURED, `raw/cpu_model_smoke.json`)

- Strict state-dict load, step 50400, 1131 keys, 0 missing / 0 unexpected, `param_breakdown_equal`, `declared_vs_built_mismatches: []`,
  100,468,987 params, DDIM sampler, 2 decoder steps, image 416x1024 cylindrical (f_ref 488.92398517830253, 120° HFOV), 9 input channels,
  horizons 5..60 ticks, `no_strategic: true`, residual prior `ha0_ext_pose`.
- Eval-item contract (what a plan must be fed): `frames [8,9,416,1024]`, `pose_hist [8,4]` (x,y,yaw,v), `nav_cmd`, `nav_valid`,
  `v_max_ms` + `v_max_valid` (4-way one-hot ceiling), per-clip lift geometry from `HiresLiftGeometryBank` keyed by episode id.
- `RefCV7Policy` refuses on a horizon, image-size or seam mismatch; builds a one-clip lift bank from the scene's own extrinsics (NuRec
  `T_sensor_rig` converted to the PhysicalAI quaternion convention; Frobenius residual to the nearest PhysicalAI row 0.0052 / 0.0033,
  `raw/scene_info.json`); seeds DDIM per (inference seed, tick).
- **Which code trained it:** `config.json` is run `refcv7-r101-s0` with `--conflict-every 10` (the pre-R4 launch; `E:/refcv7_eval_kit/KIT.log`: launch `fec3a0dc`). `0fcb1c1c` (the R4 restart commit, which carries the SPEC 26.1 speed-ceiling fix) is a descendant; the kit carries both trees and strict-loads the checkpoint on either (MEASURED, CPU build).
- **Argv reaches the model (MEASURED through `main()`, `raw/refcv7_argv_reach_*.json`):** `--t2a-infer-seed 3` moves the plan by 0.72 m against seed 0; `--t2a-vmax-mode logged_2_6s` feeds `v_max_valid` 1.0 and 15.0 m/s; `--t2a-vmax-mode fixed --t2a-vmax-ms 5` feeds 5.0. The ceiling's effect on the PLAN is not established (BLOCKERS B5).
- **One real CPU forward through the stub loop (MEASURED, `raw/refcv7_cpu_tick.json`):** plan finite, 4x2; first waypoint x = 7.5412 m
  at 0.5 s = 15.08 m/s against the stub's 15 m/s world, lateral 0.023 m; `observed_frac` 0.9035. 165-201 s per tick on CPU, so a full
  CPU rollout is not feasible and the real run is GPU only. The plan is **bit-identical with and without the per-frame canon cache**.

## 4. Scenes (MEASURED, `raw/scene_info.json`)

| | scene 1 `00040136` | scene 2 `7c72937c` |
|---|---|---|
| duration / 10 Hz ticks | 19.93 s / 199 | 19.93 s / 199 |
| speed min / mean / max | 12.1 / 15.6 / 20.4 m/s | 0.00 / 6.3 / 15.2 m/s |
| max over [2 s, 6 s] from the clip start | 19.96 m/s -> 100 km/h bin (margin 71.9 km/h to the nearest edge) | 4.90 m/s -> 30 km/h bin (margin 17.6 km/h) |
| camera | 1920x1080 f-theta, cx 958.44, cy 751.18, rolling shutter top-to-bottom | cx 957.17, cy 753.62 |
| seen in refcv7 training? | no (not in the 4,572 train view, eval139, or the 17-id NuRec overlap; `raw/leak_check.json`) | no |

Both scenes are junction-free, straight-ish drives (degenerate tactical/strategic content). Scene 2 starts from standstill and its
oracle ceiling is **definition-sensitive**: the whole-clip max sits in the 100 km/h bin, the 2-6 s window in the 30 km/h bin.

## 5. Dependencies and VRAM against 8 GB

| item | status |
|---|---|
| torch + CUDA, numpy | box 2 must provide; the dev-box venv runs the CPU dry path |
| `msgpack` | needed by `nurec_loader`; missing in the dev-box venv (`scene_info.py` parses the json directly instead) |
| `gsplat==1.5.3` | pinned: every banked Thor number used it. Needs a CUDA toolchain (nvcc, MSVC `cl`, ninja) because the f-theta UT kernels JIT-compile. **Windows build UNVERIFIED** |
| cv2, grpc | not needed on the in-process path |
| VRAM, renderer | 3.1 M gaussians, scene tensors ≈0.73 GB plus rasterisation buffers: UNVERIFIED |
| VRAM, refcv7 | eval-loss peak **5.85 GiB** on the dev box's RTX 4060 (batched; a single window should be lower): INHERITED from the refcv7 eval battery log (`battery/*/g0.log`, read, not re-run); UNVERIFIED for this use |
| both resident on 8 GB | **UNVERIFIED**. `preflight_box2.py` measures renderer alone, then model + renderer with two real ticks, against the card's total |

## 6. Metrics the harness emits (`cl_metrics.py`, scored by `t2a_score.py`)

| family | emitted | notes |
|---|---|---|
| ADE | `ade_0_2s`, `de_h`, `dist_to_gt_traj_m` | episode-cluster bootstrap CI |
| LONGITUDINAL | speed error, executed-speed error, along-track error | headway / TTC only with a real lead (the `objects` condition); `empty` has none -> UNAVAILABLE per metric |
| LATERAL | cross-track, heading, curvature, yaw-rate error | |
| TACTICAL | plan-derived manoeuvre vs the log | head-based classes UNAVAILABLE for refcv7; degenerate on both straight scenes |
| STRATEGIC | **PAUSED** (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25) | raw values written "for reference, NOT A RESULT"; never a pass, never omitted |
| progress | `progress_rel` | available |
| collision | **UNAVAILABLE** | no collision response; `empty` has no actor |
| off-road | **UNAVAILABLE** | no map in the harness; `route_corridor_departure_rate` is a ±2 m corridor around the LOG, not a road edge |

Statistics: the paired episode-cluster bootstrap answers only *"would another draw of rollout starts say this?"*. Training-seed variance
(one refcv7 checkpoint) is unmeasured; inference-seed variance is measured by the `refcv7_vmax_off_s1` replicate arm. These are three
different questions and the result files name which one an interval answers.
