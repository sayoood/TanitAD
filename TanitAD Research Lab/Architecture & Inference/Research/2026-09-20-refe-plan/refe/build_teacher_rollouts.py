"""Per-frame teacher rollouts -- THE PAPER'S CONSTRUCTION, replacing the closed-loop harvest.

⛔⛔ WHY THIS FILE EXISTS. Until 2026-09-21 the REFe target bank was harvested out of a SINGLE
CONTINUOUS CLOSED-LOOP simulation: `run_mini_teacher.sh` runs the full nuPlan closed-loop sim with
the DriveRL planner driving the whole scenario, and `build_targets.py` then sliced it -- "at
simulation step i, take the ego state and the next 20 poses". In a closed-loop run the ego
ACTUALLY DRIVES AWAY FROM THE LOGGED PATH, while camera frames exist ONLY for the logged
trajectory and are matched BY TIMESTAMP.

MEASURED over all 10 logs: **1,237 / 1,491 = 83.0 %** of states sit more than 0.5 m from the
logged ego whose camera frame the tuple uses; worst log max **24.70 m**, mean 7.99 m. The model
would have been shown the world from one pose and trained to predict a trajectory from another.
⚠️ It would NOT have shown up in the loss -- the targets are self-consistent.

⭐ THE PAPER SAYS EXACTLY HOW TO DO IT, and we had quoted the sentence without acting on it:
  p. 7: "a camera-only end-to-end planner trained **open-loop on logged frames**, and the training
         targets come from **rolling out the frozen teacher at each frame**"
  p. 8: "the frozen teacher receives the structured state and a goal point, and is then rolled out
         with **all background actors following the driving log**"
⇒ For EACH LOGGED FRAME: seed the ego at THAT FRAME'S LOGGED STATE, roll the teacher forward
`SPAN` steps with the background actors replaying the log, and keep that rollout as the target.
The ego never drifts from the log, so the camera frame and the ego state describe the SAME pose;
only the 4-second FUTURE is simulated.

⚠️ The closed-loop runs are not wrong -- they are the right artifact for SCORING the teacher (our
89.755 on Test14-hard stands). One artifact was reused for a second purpose its construction does
not support. That is the `CLAUDE.md` epcache class: price the artifact the CONSUMER opens.

## The fidelity control, and why it is the whole point

A new rollout harness that is subtly wrong is worse than the defect it replaces. So this file
carries `--verify-fidelity`: a rollout seeded at **frame 0** must reproduce the EXISTING closed-loop
simulation's first `SPAN` steps, because at frame 0 the two constructions are by definition the
same run -- same initial state, same planner, same controller, same log-replay agents. Any
divergence there is a bug in THIS file, not a property of the data.
⛔ Do not trust a single rollout from this harness until that control passes.

Usage:
  python build_teacher_rollouts.py --run C:/dzo/m-nr-n --out <dir> [--rank 0]
  python build_teacher_rollouts.py --run C:/dzo/m-nr-n --verify-fidelity      # the control
  python build_teacher_rollouts.py --run C:/dzo/m-nr-n --out <dir> --limit-frames 8   # pilot
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# the target geometry, identical to build_targets.py so the two banks are interchangeable
HORIZON = 20          # samples kept -- PUBLISHED, Table A12: "20 steps at 5 Hz"
TARGET_DT_S = 0.2     # PUBLISHED, Table A12: 5 Hz
HORIZON_S = HORIZON * TARGET_DT_S          # 4.0 s
# ⛔⛔ STRIDE AND SPAN ARE NO LONGER CONSTANTS, AND THE CONSTANT VERSION WAS SILENTLY WRONG BY 2x
# ON navtrain. They used to read `STRIDE = 2 # sim is 10 Hz` and `SPAN = 40 # 4.0 s`, which is
# correct for a scenario whose `database_interval` is 0.1 -- and every val14 scenario's is, because
# THEIR SIMULATION SUBSAMPLES the database. MEASURED:
#     val14 closed-loop sim log : database_interval 0.10 -> SPAN 40 = 4.00 s, STRIDE 2 = 5.0 Hz  OK
#     navtrain per-token scenario: database_interval 0.05 -> SPAN 40 = 2.00 s, STRIDE 2 = 10.0 Hz BAD
# The raw nuPlan lidar_pc stream is **20 Hz** (measured dt 0.0500 s). So the per-token navtrain bank
# was being built with HALF the paper's horizon at DOUBLE its rate, and nothing downstream could
# see it: 20 samples x 3 numbers is the right SHAPE either way. 2,117 rows were discarded.
# ⭐ This is R6/R11/R12 a fourth time -- a constant that encodes an assumption about ONE
# construction, carried into a NEW construction where the assumption no longer holds. The durable
# fix is to stop encoding it: derive the geometry from the scenario and REFUSE if it does not
# divide cleanly.
STRIDE = 2            # legacy default for a 0.1 s scenario; call geometry_for() instead
SPAN = HORIZON * STRIDE


def geometry_for(scenario):
    """(stride, span, history_rows) for THIS scenario's own sample rate.

    Returns the number of scenario iterations that realise the PUBLISHED geometry: 20 samples at
    5 Hz over 4.0 s, seeded by 2.0 s of history. Refuses rather than rounding, because a rate that
    does not divide is a different experiment, not a near-enough one.
    """
    dt = float(scenario.database_interval)
    stride = int(round(TARGET_DT_S / dt))
    if stride < 1 or abs(stride * dt - TARGET_DT_S) > 1e-6:
        raise ValueError(
            f"database_interval {dt} s cannot express the paper's {TARGET_DT_S} s sample period "
            f"(nearest stride {stride} gives {stride * dt} s). Refusing to approximate the target "
            f"geometry.")
    return stride, HORIZON * stride, int(round(BUFFER_S / dt))


def ego_frame(px, py, pyaw, xs, ys):
    """World -> ego frame at (px, py, pyaw). Same convention as build_targets.ego_frame."""
    c, s = math.cos(-pyaw), math.sin(-pyaw)
    dx, dy = xs - px, ys - py
    return np.stack([dx * c - dy * s, dx * s + dy * c], axis=-1)


def _reset_planner_state(planner) -> None:
    """⛔ THE PLANNER IS STATEFUL AND MUST BE RESET BETWEEN ROLLOUTS.

    `DriveRLNuPlanPlanner` forwards its policy at 5 Hz while the sim steps at 10 Hz, gated by
    `_should_forward_policy(current_time_us)` against `_last_policy_time_us`; it also caches
    `_last_action` and `_cached_policy_acceleration_target`. Carrying those across rollout
    boundaries would make rollout k depend on rollout k-1 -- the rollouts would not be
    independent, which is exactly the property the paper's construction is FOR.
    ⚠️ Set to None rather than deleted: the attributes are read before they are written.
    """
    for attr in ("_last_action", "_last_policy_time_us", "_cached_policy_acceleration_target",
                 "_last_goal_points", "_last_model_input_debug"):
        if hasattr(planner, attr):
            setattr(planner, attr, None)


# the device the teacher network runs on, recorded in every row (set by main / augment_search).
# MEASURED 2026-09-24 on the pod: CPU and CUDA rollouts of one frame differ by 0 - 7 cm (float
# noise amplified through the closed loop) -- not a systematic difference like the sim rate, but a
# row should still say which it is.
TEACHER_DEVICE = "cuda"


def build_planner(device: str = "cuda"):
    """Instantiate the teacher EXACTLY as their runner does, from their own yaml.

    ⚠️ Built by hydra instantiation of the shipped config rather than by hand-constructing the
    dataclass: a hand-built config is a restatement of theirs and drifts from it silently. The
    overrides below are the ones `run_driverl_nuplan_eval.sh` passes on its hydra line.
    """
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    dz = os.environ.get("DZ_ROOT", "C:/Users/Admin/dz/DriveZero/DriveRL")
    y = (f"{dz}/nuplan-devkit/nuplan/planning/script/config/simulation/planner/"
         f"driverl_nuplan_planner.yaml")
    cfg = OmegaConf.load(y)["driverl_nuplan_planner"]
    OmegaConf.set_struct(cfg, False)
    cfg.device = device
    cfg.config_path = f"{dz}/release/configs/driverl_teacher.yaml"
    cfg.checkpoint_path = f"{dz}/release/checkpoints/checkpoint_2400.pt"
    cfg.route_goal_horizon_s = 12.0
    cfg.route_goal_min_speed_mps = 5.0
    cfg.route_goal_pair_mode = "legacy"
    cfg.tts_enabled = False
    # ⭐ exact memoisation of the teacher's STATIC map geometry (~90 % of a rollout, MEASURED) --
    # refe/map_feature_cache.py; REFE_MAP_CACHE=0 is the uncached reference arm
    if os.environ.get("REFE_MAP_CACHE", "1") != "0":
        import map_feature_cache
        map_feature_cache.install()
        print("  map-feature cache: ON (exact; REFE_MAP_CACHE=0 disables)")
    # ⭐ exact memoisation of the repeated nuPlan DB queries (66 % of a map-cached rollout,
    # MEASURED) -- refe/query_cache.py; REFE_QUERY_CACHE=0 is the uncached reference arm
    if os.environ.get("REFE_QUERY_CACHE", "1") != "0":
        import query_cache
        query_cache.install()
        print("  query cache: ON (exact; REFE_QUERY_CACHE=0 disables)")
    # ⭐ the teacher builds a per-step VISUALISATION dict (`_extract_model_input_debug`: every lane
    # and agent re-projected to the global frame, 9.7 s of 96 s on the pod = 10 %, MEASURED
    # 2026-09-23) that only lands in `debug_info["model_input"]`; nothing in REFe reads it
    # (`_reset_planner_state` clears `_last_model_input_debug` between rollouts). Skipping it cannot
    # change a row -- diag_lever_ab.py proves that on real frames. REFE_SKIP_DEBUG=0 restores it.
    if os.environ.get("REFE_SKIP_DEBUG", "1") != "0":
        import driverl.nuplan.planner as DRP
        DRP._extract_model_input_debug = lambda agent, features, ego_state: None
        print("  teacher visualisation debug: SKIPPED (REFE_SKIP_DEBUG=0 restores)")
    # ⭐ exact PER-STEP map-feature levers (container overhead, a union cache, vectorised anchor
    # rotation) -- refe/map_step_speedups.py. ON by default since 2026-09-24 03:37 Berlin, after
    # rows AND augmentation stats were byte-identical on BOTH machines and BOTH paths, with a
    # mutation arm that changed rows each time (raw/2026-09-24-levers-v2): pod rank 0 1.21x,
    # pod augmentation 1.31x, dev box 1.05x / 1.21x. REFE_MAP_STEP_FAST=0 is the reference arm.
    if os.environ.get("REFE_MAP_STEP_FAST", "1") != "0":
        import map_step_speedups
        map_step_speedups.install()
        print("  map-step fast path: ON (exact; REFE_MAP_STEP_FAST=0 disables)")
    return instantiate(cfg)


def build_controller(scenario, device: str = "cuda"):
    """Their ego controller, again from their yaml, with the runner's two overrides.

    ⚠️ TAKES THE SCENARIO: `DriveRLOneStageController.__init__` requires it, which is why nuPlan
    builds a controller PER SCENARIO rather than once per run. Hoisting it out of the loop to save
    construction time would have silently bound every log's rollouts to the first log's scenario.
    """
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    dz = os.environ.get("DZ_ROOT", "C:/Users/Admin/dz/DriveZero/DriveRL")
    y = (f"{dz}/nuplan-devkit/nuplan/planning/script/config/simulation/ego_controller/"
         f"driverl_one_stage_controller.yaml")
    cfg = OmegaConf.load(y)
    OmegaConf.set_struct(cfg, False)
    cfg.device = device
    cfg.nuplan_bicycle_max_acceleration = 4.0
    cfg.nuplan_bicycle_max_steering_rate = 0.8
    return instantiate(cfg, scenario=scenario)


# ⛔ THEIR SIMULATION SEEDS A 2.0 s HISTORY BUFFER (`default_simulation.yaml:44`,
# `simulation_history_buffer_duration: 2.0`). My first version seeded FOUR states (0.3 s) and the
# fidelity control caught it: two logs still matched to 0.008/0.027 m, but the most dynamic log
# diverged **0.355 m**. ⭐ That asymmetry is the tell -- a loop bug breaks every log, a history
# deficit only bites where the past actually carries information. A control that had run on one
# log would have passed and shipped the defect.
BUFFER_S = 2.0


def rollout_from_frame(scenario, planner, controller, i: int, span: int,
                       buffer_s: float = BUFFER_S):
    """Roll the teacher `span` steps starting from the LOGGED state at iteration `i`.

    Returns the list of ego states produced (length `span`), or None if the scenario is too short.

    ⭐ The ego is SEEDED at the logged pose and the observations come from the log, which is the
    paper's "all background actors following the driving log". Nothing here lets the ego drift
    between frames: every rollout starts from the log again.
    """
    from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
    from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
        SimulationIteration)

    n = scenario.get_number_of_iterations()
    if i + span >= n:
        return None

    # ⛔ SEED THE BUFFER THE WAY THEIR SIMULATION DOES -- FROM THE SCENARIO'S PAST, NOT BY INDEXING
    # ITERATIONS. `SimulationHistoryBuffer.initialize_from_scenario` calls
    # `get_ego_past_trajectory` / `get_past_tracked_objects`, which reach **BEFORE** the iteration
    # into the log. My first version built the buffer as `range(max(0, i-N+1), i+1)`, which cannot
    # go earlier than iteration 0 at all.
    # ⚠️ AND THAT IS WHY MY BUFFER-SENSITIVITY TEST WAS WORTHLESS: I ran it at frame 0, where the
    # clamp collapses the window to a SINGLE state for every N, so 0.2 s, 2.0 s and 4.0 s all gave
    # byte-identical answers and I nearly concluded history was irrelevant. A knob tested at the
    # one operating point where it cannot move is not a tested knob.
    buffer_size = max(int(round(buffer_s / scenario.database_interval)), 2)
    egos = list(scenario.get_ego_past_trajectory(
        iteration=i, time_horizon=buffer_s, num_samples=buffer_size))
    obs = list(scenario.get_past_tracked_objects(
        iteration=i, time_horizon=buffer_s, num_samples=buffer_size))
    # the buffer their sim holds ends at the CURRENT state, so append it
    egos.append(scenario.get_ego_state_at_iteration(i))
    obs.append(scenario.get_tracked_objects_at_iteration(i))
    buf = SimulationHistoryBuffer.initialize_from_list(
        len(egos), egos, obs, scenario.database_interval)

    # ⚠️ NO `TracksObservation` HERE, DELIBERATELY. An earlier version built one and never used it
    # -- dead code that makes a reader believe the observation pipeline is wired when in fact the
    # direct `scenario.get_tracked_objects_at_iteration(...)` calls below are doing the work.
    # They are exactly equivalent for the non-reactive protocol (TracksObservation IS log replay),
    # and being explicit is what makes "all background actors following the driving log" checkable
    # against the paper rather than hidden behind an object.
    controller.reset()
    _reset_planner_state(planner)

    # the controller's state IS the ego; seed it at the logged pose
    if hasattr(controller, "_current_state"):
        controller._current_state = egos[-1]

    from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
    out = []
    goal_used = None
    for k in range(span):
        it = SimulationIteration(time_point=scenario.get_time_point(i + k), index=i + k)
        nxt = SimulationIteration(time_point=scenario.get_time_point(i + k + 1), index=i + k + 1)
        planner_input = PlannerInput(iteration=it, history=buf,
                                     traffic_light_data=list(
                                         scenario.get_traffic_light_status_at_iteration(i + k)))
        traj = planner.compute_planner_trajectory(planner_input)
        if k == 0:
            # ⭐ THE GOAL THE TEACHER ACTUALLY RECEIVED, at the LOGGED frame, captured from the
            # planner itself on the first step of THIS rollout. The paper: the student's command
            # "is derived from the goal point GIVEN TO THE TEACHER, so student and teacher share
            # one driving intent". MEASURED 2026-09-21: reading it from the closed-loop log instead
            # (the drifted ego's frame) put it up to **8.39 m** away from this one (mean 4.15 m),
            # identical at step 0 and diverging after -- the pose drift again, in a second field.
            gp = getattr(planner, "_last_goal_points", None)
            goal_used = (np.asarray(gp, dtype=float).reshape(-1).tolist()
                         if gp is not None else None)
        state = controller.get_state()
        controller.update_state(it, nxt, state, traj)
        new_state = controller.get_state()
        # background actors REPLAY THE LOG -- this is the paper's protocol, not a simplification
        buf.append(new_state, scenario.get_tracked_objects_at_iteration(i + k + 1))
        out.append(new_state)
    rollout_from_frame.last_goal = goal_used          # read by the caller; see main()
    return out


def verify_fidelity(run: str, db_dir: str, device: str, tol_m: float = 0.25) -> int:
    """⭐ THE CONTROL: a rollout seeded at frame 0 must reproduce the CLOSED-LOOP sim's first SPAN
    steps, because at frame 0 the two constructions are the same run by definition.

    This is an INDEPENDENT reference -- the closed-loop logs were produced by their runner, not by
    this file -- which is what makes it admissible. A harness validated only against itself would
    measure determinism, not correctness.
    """
    from nuplan.planning.simulation.simulation_log import SimulationLog
    logs = sorted(glob.glob(f"{run}/**/*.msgpack.xz", recursive=True))
    if not logs:
        print(f"  no simulation logs under {run!r}")
        return 2
    planner = build_planner(device)
    worst, n_ok, n_bad = 0.0, 0, 0
    for lp in logs[:3]:
        log = SimulationLog.load_data(file_path=Path(lp))
        sc = log.scenario
        controller = build_controller(sc, device)   # per scenario -- see build_controller
        _st, _sp, _hi = geometry_for(sc)
        ref = [s.ego_state for s in log.simulation_history.data[:_sp + 1]]
        planner.initialize(__import__(
            "nuplan.planning.simulation.planner.abstract_planner", fromlist=["x"]
        ).PlannerInitialization(
            route_roadblock_ids=sc.get_route_roadblock_ids(),
            mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
        got = rollout_from_frame(sc, planner, controller, 0, _sp)
        if got is None:
            print(f"  {sc.log_name[:28]}: too short, skipped")
            continue
        d = [math.dist((g.rear_axle.x, g.rear_axle.y), (r.rear_axle.x, r.rear_axle.y))
             for g, r in zip(got, ref[1:])]
        w = max(d)
        worst = max(worst, w)
        ok = w <= tol_m
        n_ok += ok
        n_bad += (not ok)
        print(f"  {sc.log_name[:28]}  max |delta| {w:7.3f} m  mean {sum(d)/len(d):6.3f} m  "
              f"{'MATCH' if ok else 'DIVERGES'}")
    print(f"\n  worst across logs: {worst:.3f} m   (tolerance {tol_m} m)")
    good = n_bad == 0 and n_ok > 0
    print("\n" + ("FIDELITY_OK -- the per-frame harness reproduces their closed-loop run"
                  if good else
                  "FIDELITY_FAILED -- this harness does NOT reproduce their run; do not build a bank"))
    return 0 if good else 1


def make_row(sc, planner, controller, i: int, cams, cam_ts, rank: int):
    """One training tuple for frame `i` of `sc`, or None if the frame cannot be used.

    ⭐ EXTRACTED SO THE TWO SCENARIO SOURCES SHARE ONE IMPLEMENTATION. The `run` source reads
    scenarios out of closed-loop sim logs (val14); the `navtrain` source builds them from
    `(log, token)` via `navtrain_scenarios.py`. If each carried its own copy of this body they
    would drift, and the equivalence control between them would be comparing two different
    constructions rather than one construction over two inputs. The control is only worth running
    because this is shared.
    """
    import build_targets as BT
    # ⛔ derived per scenario, never the module constant -- see geometry_for(). val14 sim logs run
    # at 0.1 s and navtrain DB scenarios at 0.05 s, so a fixed stride halves the horizon on one of
    # them while producing an identically-shaped 20x3 target.
    stride, span, _hist = geometry_for(sc)
    poses = rollout_from_frame(sc, planner, controller, i, span)
    if poses is None:
        return None
    # ⭐ THE EGO AND THE CAMERA NOW DESCRIBE THE SAME POSE: both come from the LOG at i.
    st = sc.get_ego_state_at_iteration(i)
    px, py, pyaw = st.rear_axle.x, st.rear_axle.y, st.rear_axle.heading
    sel = poses[stride - 1::stride][:HORIZON]
    if len(sel) < HORIZON:
        return None
    xs = np.array([p.rear_axle.x for p in sel])
    ys = np.array([p.rear_axle.y for p in sel])
    yaw = np.array([((p.rear_axle.heading - pyaw + math.pi) % (2 * math.pi)) - math.pi
                    for p in sel])
    traj = np.concatenate([ego_frame(px, py, pyaw, xs, ys), yaw[:, None]], axis=-1)

    ts_i = int(st.time_point.time_us)
    rels, dts, bad = [], [], False
    for ch in BT.CAMERAS:
        jj = int(np.argmin(np.abs(cam_ts[ch] - ts_i)))
        d = abs(int(cam_ts[ch][jj]) - ts_i) / 1000.0
        if d > 60.0:
            bad = True
            break
        rels.append(cams[ch][jj][1]); dts.append(d)
    if bad:
        return None
    v = st.dynamic_car_state.rear_axle_velocity_2d
    acc = st.dynamic_car_state.rear_axle_acceleration_2d
    ego = [float(v.x), float(v.y), float(acc.x), float(acc.y),
           float(st.dynamic_car_state.angular_velocity),
           float(st.tire_steering_angle), float(math.hypot(v.x, v.y))]
    # ⛔ NOT from `log.simulation_history.data[i].trajectory.goal_points` -- that is the goal
    # of the CLOSED-LOOP run at step i, expressed in the frame of an ego that had drifted
    # off the log. Use the goal THIS rollout's teacher received at the logged frame.
    goal = getattr(rollout_from_frame, "last_goal", None)
    if goal is None:
        return None                  # no goal => no consistent intent; drop, do not pad
    goal = goal[:4] if len(goal) >= 4 else goal + [0.0] * (4 - len(goal))
    # the same rollout in WORLD coordinates, so the SCORER bank can build its candidates
    # around the teacher's per-frame rollout instead of around the closed-loop slice --
    # otherwise the second training signal keeps the drift the first one just lost.
    traj_world = [[float(p.rear_axle.x), float(p.rear_axle.y), float(p.rear_axle.heading)]
                  for p in sel]
    return {"image": rels, "cameras": list(BT.CAMERAS), "ego": ego, "goal": goal,
            "traj": traj.tolist(), "traj_world": traj_world,
            "origin_world": [float(px), float(py), float(pyaw)],
            "rank": rank, "step": i, "dt_ms": max(dts),
            "scenario_type": sc.scenario_type, "token": sc.scenario_name,
            "log_name": sc.log_name, "source": "per_frame_teacher_rollout",
            # the closed-loop rate of THIS rollout (diag_sim_rate.py: 10 vs 20 Hz moves the
            # target by ADE 0.23 m / FDE 0.60 m), so a bank can never silently mix the two
            "sim_hz": int(round(1.0 / float(sc.database_interval))),
            "teacher_device": TEACHER_DEVICE}


def camera_arrays(db_path: str):
    """(cams, cam_ts) for a log, or (None, None) if any of the four channels is absent.

    ⚠️ Indexed PER LOG and reused across that log's frames. In the navtrain source every frame is
    its own scenario, so doing this per scenario would re-read the image table ~85 times per log.
    """
    import build_targets as BT
    cams = BT.camera_index(db_path)
    cam_ts = {ch: (np.array([t for t, _ in v], dtype=np.int64) if v else np.zeros(0, np.int64))
              for ch, v in cams.items()}
    if any(cam_ts[ch].size == 0 for ch in BT.CAMERAS):
        return None, None
    return cams, cam_ts


def build_navtrain(a, planner, W) -> None:
    """Rows for the NAVSIM navtrain split, scenarios built from DB tokens.

    ⛔ ONE SCENARIO PER FRAME, not per log: `get_route_roadblock_ids()` is keyed on the scenario's
    `initial_lidar_token`, so a whole-log scenario would hand the teacher the route from the log's
    FIRST frame for every one of its ~85 navtrain frames. See `navtrain_scenarios` for the measured
    size of that error (median 2,300 iterations away) and for why it would pass every gate we own.

    MEASURED cost of the per-frame shape: build_controller 0.013 s, warm planner.initialize
    0.005 s, against a 5.94 s rollout -- 0.3 %.
    """
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    import navtrain_scenarios as NS
    # ⛔ ONE RATE PER BANK. Resuming a 20 Hz bank at 10 Hz (or back) would write rows whose targets
    # differ by construction (MEASURED ADE 0.23 m) into the same training set, keyed identically.
    if W.rates - {NS.SIM_HZ}:
        print(f"  REFUSING: this bank already holds rows at {sorted(W.rates)} Hz and REFE_SIM_HZ is "
              f"{NS.SIM_HZ}. Use a fresh --out (and --resume-glob) for a different rate.")
        raise SystemExit(5)
    print(f"  simulation rate: {NS.SIM_HZ} Hz (REFE_SIM_HZ)")

    kw = {}
    if a.navtrain_yaml:
        kw["yaml_path"] = a.navtrain_yaml
    if a.db_root:
        kw["db_root"] = a.db_root
    per_log, rep = NS.resolve(**kw)
    dbs = NS.index_dbs(**({"db_root": a.db_root} if a.db_root else {}))
    print(f"  navtrain: {rep['split_tokens']:,} tokens over {rep['split_logs']} logs; "
          f"local DBs cover {rep['covered_logs']} logs / {rep['frames']:,} frames")

    names = sorted(per_log)
    if getattr(a, "logs_file", None):
        # ⭐ TARGETED BUILDS. The default order is by NAME, and names sort by DATE -- MEASURED
        # 2026-09-23 the first 1,300 corrected rows were **100 % Las Vegas** (14 logs). Any pilot run
        # on a partial bank inherits that city, and LV is the city where lane-rank augmentation looks
        # best (>=3 lanes on 80 % of frames vs 0 % in Singapore / Pittsburgh). A stratified sample
        # needs the builder to take a named list.
        want = {l.strip() for l in open(a.logs_file, encoding="utf-8") if l.strip()}
        missing = sorted(want - set(names))
        if missing and a.logs_missing != "skip":
            print(f"  --logs-file names {len(missing)} log(s) with no local navtrain frames: "
                  f"{missing[:3]}"); raise SystemExit(2)
        if missing:
            print(f"  --logs-file: {len(missing)} named log(s) have no local DB YET -- skipped this "
                  f"run (--logs-missing skip); a relaunch after their fetch picks them up")
        # ⛔ SHARD OVER THE NAMED LIST, NOT OVER WHAT IS LOCAL NOW. MEASURED 2026-09-24 on the dev
        # box: 11 shards started while their DBs were still arriving saw 233..267 local logs each,
        # so `names[i::n]` sliced DIFFERENT lists -- overlapping (duplicate work) and leaving gaps.
        # The file's sorted list is the same for every shard whatever has landed, so the partition
        # is too; availability only decides what a shard can do THIS run.
        names = sorted(want)
        local = set(per_log)
        print(f"  --logs-file: {len(names)} named logs ({len(want & local)} with a local DB now)")
    if a.limit_logs:
        names = names[:a.limit_logs]
    if a.log_shard:
        # ⛔ the shard is applied AFTER --limit-logs so the two compose predictably: N shards of a
        # limited set partition exactly that set. Applied the other way round, `--limit-logs 4
        # --log-shard 1/2` would take logs 1,3,5,7 of ALL 214 and silently do different work.
        i, n = (int(x) for x in a.log_shard.split("/"))
        if not (n > 0 and 0 <= i < n):
            print(f"  --log-shard {a.log_shard} invalid: need 0 <= i < N and N > 0"); raise SystemExit(2)
        names = names[i::n]
        print(f"  shard {i}/{n}: {len(names)} logs, "
              f"{sum(len(per_log.get(x, ())) for x in names):,} frames available now")
        # ⭐ SHUFFLED LOG ORDER (2026-09-24, --grow training). Logs used to be processed in SORTED
        # name order, i.e. by DATE -- so a bank read while it was still growing was the EARLY drives
        # only (one city, one season). Training now starts before data prep ends, so every partial
        # bank must be an unbiased sample. A seeded shuffle AFTER the shard split changes the order
        # the work is done in, never which rows exist or what they contain. REFE_SHUFFLE_LOGS=0 restores it.
    if os.environ.get("REFE_SHUFFLE_LOGS", "1") != "0":
        import random
        random.Random(20260924).shuffle(names)
        print("  log order: SHUFFLED (seed 20260924) -- a partial bank is an unbiased sample")
    names = [x for x in names if x in per_log]       # a named log without a local DB: next run
    t0 = time.time()
    for lg in names:
        toks = per_log[lg]
        if a.limit_frames:
            toks = toks[:a.limit_frames]
        # ⭐ resume check BEFORE the camera index: a fully-finished log must cost no DB read at all,
        # otherwise resuming a 30 h run still pays ~214 image-table scans to discover it has nothing
        # to do.
        todo = [t for t in toks if not W.has(lg, t, 0)]
        if not todo:
            continue
        cams, cam_ts = camera_arrays(dbs[lg])
        if cams is None:
            print(f"    {lg[:28]}: missing a camera channel, skipped")
            continue
        # ⭐ `step` STAYS 0 for every navtrain row, and that is correct, not a collapsed key.
        # `NuPlanScenario.scenario_name` returns `token` returns `_initial_lidar_token`, so the
        # row's `token` field already carries the per-FRAME token here -- each navtrain frame is
        # its own scenario. `(log_name, token, step)` is therefore unique with step 0.
        # ⛔ Do NOT "fix" this by writing the token into `step`: `build_scorer_targets.logged_ego`
        # consumes `step` as an ITERATION INDEX, and the navtrain frame IS iteration 0 of its own
        # scenario. A hex string there would break the scorer while looking like a key repair.
        kept = 0
        for sc in NS.build_scenarios_for_log(dbs[lg], todo):
            # ⛔⛔ ONE BAD FRAME MUST NEVER KILL A 16-HOUR SHARD. Twice now a shard has died on a
            # single frame -- the devkit's NULL-prev_token read -- and each death was silent from
            # outside: a dead shard and a slow shard produce the same row count. The frame-level
            # guard in navtrain_scenarios is the FIX; this is the BACKSTOP, so the next unforeseen
            # per-frame fault costs one frame and a logged line, not a quarter of the run.
            # ⚠️ Counted and printed, never silent -- a backstop that swallows errors without a
            # count is how a 25 % data loss hides (R12).
            try:
                controller = build_controller(sc, a.device)
                planner.initialize(PlannerInitialization(
                    route_roadblock_ids=sc.get_route_roadblock_ids(),
                    mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
                # the navtrain frame IS iteration 0 of its own scenario; the 2.0 s history is read
                # backward out of the DB, not out of the extraction window (navtrain_scenarios).
                r = make_row(sc, planner, controller, 0, cams, cam_ts, a.rank)
            except Exception as exc:
                _reset_planner_state(planner)
                n_frame_errors = getattr(build_navtrain, "_errors", 0) + 1
                build_navtrain._errors = n_frame_errors
                print(f"    FRAME_ERROR #{n_frame_errors} {lg[:28]} token {sc._initial_lidar_token}: "
                      f"{type(exc).__name__}: {str(exc)[:80]}", flush=True)
                continue
            if r is not None:
                W.write(r); kept += 1
        W.flush()
        print(f"    {lg[:28]:28s} frames {len(todo):4d}  kept {kept:4d}  {time.time()-t0:7.1f}s")


def open_append(path: str):
    """Append-mode open that first TERMINATES a torn last line.

    ⛔ MEASURED RISK 2026-09-24: rows are flushed per log through an 8 KB buffer, so a kill (a shard
    restart, a pod stop) can leave the file ending in HALF a row with no newline. Every reader
    skips that fragment -- but a resumed writer appended the next row DIRECTLY onto it, fusing a
    good row into the unparseable line, so the frame was silently lost from the bank and, being
    absent from the resume set only until the next restart, never re-derived. A newline first
    isolates the fragment on its own line.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            torn = f.read(1) != b"\n"
        if torn:
            with open(path, "ab") as f:
                f.write(b"\n")
    return open(path, "a", encoding="utf-8")


class RowWriter:
    """Append rows to the bank AS THEY ARE PRODUCED, and skip ones already on disk.

    ⛔ The builder used to accumulate every row in memory and write once at the end. At val14 scale
    (1,101 rollouts, ~2 h) that was merely unpleasant. At navtrain scale it is not survivable:
    MEASURED 5.94 s/frame means the local 18,179-frame subset is **30 h** and the full 103,288-frame
    split is **170 h** single-process, and a crash at hour 29 wrote nothing at all.

    ⭐ `--resume` keys on `(log_name, token, step)` -- the SAME key the scorer bank and the
    consistency gate join on. Re-deriving it here from the written rows rather than from a side-car
    progress file means the resume set cannot disagree with the bank: the file IS the progress.
    ⚠️ Rows are flushed per log, not per row, so a kill loses at most one log's work; and the file
    is opened append-only, so a resume can never truncate what an earlier run earned.
    """

    def __init__(self, path: str, resume: bool, also: str | None = None):
        self.path = path
        self.seen: set = set()
        # rates of the rows already banked (a navtrain row without the field predates it: 20 Hz)
        self.rates: set = set()
        # ⭐ A SHARED RESUME SET (2026-09-23, first REFe pod). With `also` = a glob over the OTHER
        # shards' files, a row done by ANY shard -- or uploaded from another machine into its own
        # shard dir -- is skipped. Without it, changing the shard count between launches moves
        # logs to shards whose file never saw them: finished work is redone and the merge gets
        # duplicates.
        if resume and also:
            extra = 0
            for q in sorted(glob.glob(also)):
                if os.path.abspath(q) == os.path.abspath(path):
                    continue
                with open(q, "r", encoding="utf-8") as fq:
                    for line in fq:
                        try:
                            r = json.loads(line) if line.strip() else None
                        except json.JSONDecodeError:
                            continue
                        if r is not None:
                            self.seen.add((r.get("log_name"), r.get("token"), r.get("step")))
                            self.rates.add(int(r.get("sim_hz", 20)))
                            extra += 1
            print(f"  resume: {extra:,} rows already done in other files matching {also}")
        if resume and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        # a row torn by a kill mid-write: everything before it stands, and this
                        # one is simply re-derived. Do NOT discard the file over one bad line.
                        continue
                    self.seen.add((r.get("log_name"), r.get("token"), r.get("step")))
                    self.rates.add(int(r.get("sim_hz", 20)))
            print(f"  resume: {len(self.seen):,} rows known (own file {os.path.basename(path)} "
                  f"included)")
        elif not resume and os.path.exists(path):
            os.remove(path)
        self.f = open_append(path)
        self.n = 0

    def has(self, log_name, token, step) -> bool:
        return (log_name, token, step) in self.seen

    def write(self, row) -> None:
        self.f.write(json.dumps(row) + "\n")
        self.seen.add((row.get("log_name"), row.get("token"), row.get("step")))
        self.n += 1

    def flush(self) -> None:
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self) -> None:
        self.flush()
        self.f.close()

    def total(self) -> int:
        return len(self.seen)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="closed-loop sim dir, for scenarios + frame list")
    ap.add_argument("--source", choices=("run", "navtrain"), default="run",
                    help="'run': scenarios from closed-loop sim logs (val14). "
                         "'navtrain': scenarios built from the NAVSIM split's DB tokens.")
    ap.add_argument("--navtrain-yaml", default=None)
    ap.add_argument("--db-root", default=None, help="navtrain: nuPlan splits root")
    ap.add_argument("--limit-logs", type=int, default=0, help="navtrain pilot: first N logs")
    ap.add_argument("--logs-file", default=None,
                    help="navtrain: restrict to the log names listed one per line (stratified pilots)")
    ap.add_argument("--log-shard", default=None, metavar="i/N",
                    help="navtrain: process only logs i, i+N, i+2N ... Shard BY LOG, never by "
                         "frame -- each navtrain frame is a single-step scenario, so a frame-level "
                         "shard selects nothing and writes an empty bank with exit 0.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--db-dir", default="D:/Projects/TanitAD/data/nuplan/dblinks/driverl_val14")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-frames", type=int, default=0, help="pilot: frames per log")
    ap.add_argument("--logs-missing", choices=("refuse", "skip"), default="refuse",
                    help="a --logs-file log with no local DB: refuse (default, the list is a "
                         "contract) or skip it this run (DBs still being fetched)")
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing bank, skipping (log, token, step) already written")
    ap.add_argument("--resume-glob", default=None,
                    help="with --resume: ALSO skip rows present in any file matching this glob "
                         "(the other shards' banks, or rows uploaded from another machine)")
    ap.add_argument("--verify-fidelity", action="store_true")
    a = ap.parse_args(argv)
    if a.source == "run" and not a.run:
        print("  --run is required for --source run"); return 2

    # ⛔ THE RANK MUST SELECT THE ROUTE IN THIS PROCESS -- see route_rank.py (R11). Applied
    # before BOTH modes, so the fidelity control verifies the same route the build will roll.
    import route_rank
    print(f"  {route_rank.apply(a.rank)}")

    if a.verify_fidelity:
        return verify_fidelity(a.run, a.db_dir, a.device)
    if not a.out:
        print("  --out is required unless --verify-fidelity"); return 2

    from nuplan.planning.simulation.simulation_log import SimulationLog
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    import build_targets as BT

    global TEACHER_DEVICE
    TEACHER_DEVICE = a.device
    planner = build_planner(a.device)
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"targets_rank{a.rank}.jsonl")
    W = RowWriter(out, a.resume, a.resume_glob)
    t0 = time.time()
    if a.source == "navtrain":
        build_navtrain(a, planner, W)
    else:
      for lp in sorted(glob.glob(f"{a.run}/**/*.msgpack.xz", recursive=True)):
        log = SimulationLog.load_data(file_path=Path(lp))
        sc = log.scenario
        controller = build_controller(sc, a.device)   # per scenario -- see build_controller
        planner.initialize(PlannerInitialization(
            route_roadblock_ids=sc.get_route_roadblock_ids(),
            mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
        cams, cam_ts = camera_arrays(os.path.join(a.db_dir, f"{sc.log_name}.db"))
        if cams is None:
            print(f"  {sc.log_name[:28]}: missing a camera channel, skipped")
            continue
        n = sc.get_number_of_iterations()
        _st, _sp, _hi = geometry_for(sc)
        frames = list(range(0, n - _sp, max(a.frame_stride, 1)))
        if a.limit_frames:
            frames = frames[:a.limit_frames]
        kept = 0
        for i in frames:
            if W.has(sc.log_name, sc.scenario_name, i):
                continue
            r = make_row(sc, planner, controller, i, cams, cam_ts, a.rank)
            if r is not None:
                W.write(r); kept += 1
        W.flush()
        print(f"    {sc.log_name[:28]:28s} frames {len(frames):4d}  kept {kept:4d}  "
              f"{time.time()-t0:7.1f}s")
    W.close()
    print(f"\n  wrote {W.n:,} tuples this run to {out}  (bank total {W.total():,})")
    print(f"  wall clock {(time.time()-t0)/60:.1f} min  "
          f"=> {(time.time()-t0)/max(W.n,1):.2f} s/tuple")
    for mod in ("map_feature_cache", "query_cache", "map_step_speedups"):
        if mod in sys.modules:
            print(f"  {sys.modules[mod].summary()}")
    print("TEACHER_ROLLOUTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
