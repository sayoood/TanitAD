"""Stage 2c -- the six-component PDM targets REFe's scorer head is trained against.

WHAT THIS PRODUCES. For each frame in the trajectory bank: K candidate trajectories and, for each,
the six released PDM calculators' verdicts, scored THE WAY THE RL ENGINE SCORES THEM -- once per
horizon step, aggregated over the horizon. Events take their max ("did it ever happen"), rewards
take their min ("the worst it ever got").

⛔ HOW THIS DIFFERS FROM DriveZero, AND WHY. They score the STUDENT'S OWN 64 proposals online,
because the scorer is trained jointly with the proposal head. That costs a full calculator rollout
per proposal per optimisation step, which this rig cannot pay. We instead precompute targets for a
FIXED, STRUCTURED candidate set around the teacher's realised path. The consequence is stated
rather than hidden: the scorer learns to rank trajectories drawn from OUR perturbation basis, so a
proposal far outside that basis is extrapolation. Report it with any scorer number.

⭐ THE CANDIDATE SET is built to contain known-good and known-bad examples by CONSTRUCTION, not by
hope. It carries the teacher's own path (must score clean) and, when a curb is in range, a path
aimed through it (must score off-road). Those two are the per-frame controls and this tool ABORTS a
frame whose controls do not separate -- a bank built from an inert scorer is worse than no bank,
because every downstream number would look plausible. That failure has already happened once here:
see raw/refe_scorer_INERT.txt and its retraction.

COST, MEASURED 2026-09-20 on the dev box, single core, CPU:
  simulation-log load          ~11 s  ONCE PER LOG (not per frame -- this file exists to make that
                                      distinction, an earlier draft reloaded per frame)
  ScenarioData build            see --report-cost
  one calculator pass           41.6 ms  (deepcopy 3.0 ms + calculators 38.6 ms)
  rollout, stride 1 (20 steps)  842 ms per candidate
  rollout, stride 2 (11 steps)  444 ms per candidate
⇒ state --stride and --candidates with any number produced from this bank.

Usage:
  python build_scorer_targets.py --run C:/dzo/m-nr-16 --out D:/.../refe_scorer_targets \
      [--stride 2] [--candidates 8] [--max-frames 40] [--report-cost]
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

HORIZON, STRIDE_HIST = 20, 2          # matches build_targets.py: 20 samples @ 5 Hz = 4.0 s


class LogSession:
    """One simulation log, loaded ONCE, serving a ScenarioData per step.

    ⛔ The naive shape -- call a `scenario_data_from_log(path, step)` helper per frame -- reloads
    and xz-decompresses the whole log every time. MEASURED 11.9 s per call: over the 982 frames of
    one rank that is 3.2 HOURS of pure decompression before a single calculator has run.
    """

    def __init__(self, log_path: str = None, scenario=None):
        """`log_path`: a closed-loop sim log (val14). `scenario`: a ready scenario (navtrain).

        ⛔ navtrain has NO simulation logs, so there is nothing to decompress and `self.samples` is
        None. That is not a degraded mode -- `samples` only ever feeds the `logged=False` path,
        which is the CLOSED-LOOP construction the paper does not use and which
        `scenario_data(logged=True)` exists to replace. Asking for it here is refused loudly rather
        than silently returning a context built around a pose the camera never saw.
        """
        import scorer_gate as G
        from driverl.nuplan.feature_builder import DriveRLNuPlanFeatureBuilder
        self._G = G
        if scenario is not None:
            self.scenario = scenario
            self.samples = None
            self.n = scenario.get_number_of_iterations()
        else:
            from nuplan.planning.simulation.simulation_log import SimulationLog
            log = SimulationLog.load_data(file_path=Path(log_path))
            self.scenario = log.scenario
            self.samples = log.simulation_history.data
            self.n = len(self.samples)
        dr, _ = G.load_domain_randomization()
        self.builder = (DriveRLNuPlanFeatureBuilder(domain_randomization=dr) if dr is not None
                        else DriveRLNuPlanFeatureBuilder())
        self.init = None

    def logged_ego(self, step: int):
        """The ego AS LOGGED at `step` -- the pose the camera frame was actually taken from."""
        return self.scenario.get_ego_state_at_iteration(step)

    def scenario_data(self, step: int, logged: bool = False):
        """⛔ `logged=True` IS THE PAPER'S CONSTRUCTION; `False` is the historical closed-loop one.

        With `logged=False` the history comes from `self.samples` -- the CLOSED-LOOP run, whose ego
        drifted off the log (MEASURED 83.0 % of states > 0.5 m, max 24.70 m). Every scoring
        context built that way is centred on a pose the camera never saw. `logged=True` seeds the
        history from the SCENARIO'S PAST (`get_ego_past_trajectory`), exactly as
        `build_teacher_rollouts.py` does and as nuPlan's own `initialize_from_scenario` does, so
        the PDM targets are computed around the same pose as the image and the trajectory target.
        """
        from nuplan.planning.simulation.planner.abstract_planner import (
            PlannerInitialization, PlannerInput)
        from nuplan.planning.simulation.history.simulation_history_buffer import (
            SimulationHistoryBuffer)
        from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
            SimulationIteration)
        sc = self.scenario
        if not logged and self.samples is None:
            raise RuntimeError(
                "scenario_data(logged=False) needs the closed-loop samples, and this session was "
                "built from a navtrain scenario which has none. logged=False is the construction "
                "whose ego drifted off the log (83.0 % of states > 0.5 m); refusing to substitute.")
        if logged:
            egos = list(sc.get_ego_past_trajectory(iteration=step, time_horizon=0.8, num_samples=8))
            obs = list(sc.get_past_tracked_objects(iteration=step, time_horizon=0.8, num_samples=8))
            egos.append(sc.get_ego_state_at_iteration(step))
            obs.append(sc.get_tracked_objects_at_iteration(step))
        else:
            egos = [s.ego_state for s in self.samples[: step + 1]]
            obs = [s.observation for s in self.samples[: step + 1]]
        keep = min(len(egos), 8)
        buf = SimulationHistoryBuffer.initialize_from_list(
            buffer_size=keep, ego_states=egos[-keep:], observations=obs[-keep:],
            sample_interval=0.1)
        if self.init is None:
            self.init = PlannerInitialization(
                route_roadblock_ids=list(sc.get_route_roadblock_ids()),
                mission_goal=sc.get_mission_goal(), map_api=sc.map_api)
        pin = PlannerInput(
            iteration=SimulationIteration(time_point=egos[-1].time_point, index=step),
            history=buf,
            traffic_light_data=list(sc.get_traffic_light_status_at_iteration(step)))
        sd = self.builder.build(pin, self.init)
        try:
            log_sd = self.builder.build_log_scenario_data(sd, sc, step, HORIZON)
        except Exception:
            log_sd = sd
        return sd, log_sd


def candidate_set(sd, teacher_xy, teacher_yaw, k: int):
    """Teacher + structured perturbations + (when reachable) an analytic over-curb control.

    The lateral family is the one that matters for off-road/centre-line; the longitudinal family is
    the one that matters for the 88.7 % of our oracle gap that is longitudinal. Both are present so
    the scorer sees variation on both axes rather than a single direction.
    """
    import scorer_gate as G
    cands = [("teacher", teacher_xy, teacher_yaw)]
    lat = [-4.0, -2.0, 2.0, 4.0]
    lon = [0.5, 1.5]
    for d in lat:
        xy = teacher_xy.clone()
        xy[:, 1] += torch.linspace(0, d, xy.shape[0])
        cands.append((f"lat{d:+g}", xy, teacher_yaw))
    for f in lon:
        xy = teacher_xy.clone() * float(f)
        cands.append((f"lon x{f:g}", xy, teacher_yaw))
    cands.append(("stopped", torch.zeros_like(teacher_xy), torch.zeros_like(teacher_yaw)))
    # ⛔ A COMFORT-VIOLATING CANDIDATE, because without one the comfort component has nothing to
    # rank. MEASURED 2026-09-20: `comfort` read exactly 1.0000 for every candidate of every frame.
    # The first diagnosis was that the injected velocities were identical -- true, and fixed -- but
    # the deeper reason is that EVERY candidate in this set is SMOOTH. Lateral offsets, longitudinal
    # scalings, a stationary ego and a straight run at a curb are all comfortable paths, so a
    # correct comfort calculator returns 1.0 for all of them and the component is constant BY
    # CONSTRUCTION OF THE CANDIDATE SET, not by any defect in the calculator.
    # ⭐ The fix belongs here rather than in the scorer: give the set something uncomfortable. A
    # sawtooth lateral oscillation produces large lateral jerk while staying near the teacher's
    # path, so it is bad for exactly one reason and stays in-distribution for the others.
    jerk = teacher_xy.clone()
    sign = torch.tensor([1.0 if (k // 2) % 2 == 0 else -1.0 for k in range(jerk.shape[0])])
    jerk[:, 1] = jerk[:, 1] + sign * 1.5
    cands.append(("jerky", jerk, teacher_yaw))
    # ⛔ A DIRECTION-VIOLATING CANDIDATE, for the same reason the jerky one exists.
    # MEASURED: `off_road.OffRoad.info` took only {0, 1} across the whole bank -- categories 3 and
    # 4 (wrong-way and severe wrong-way) NEVER occurred, so driving-direction compliance carried
    # zero supervision. That is not a defect in their detector: it fires on BACKWARD progress
    # (thresholds 2.0 m and 6.0 m over 1.0 s) and not one candidate in the set ever drove
    # backwards. A component cannot be supervised by a set that never violates it.
    rev = -teacher_xy.clone()
    cands.append(("reverse", rev, teacher_yaw))
    tgt, curb_d, _n = G.nearest_curb_target(sd, over_m=8.0)
    if tgt is not None and curb_d is not None and curb_d < 40.0:
        xy = torch.stack([tgt * s for s in torch.linspace(1.0 / HORIZON, 1.0, HORIZON)], 0)
        yaw = torch.full((HORIZON,), math.atan2(float(tgt[1]), float(tgt[0])))
        cands.append(("over-curb", xy, yaw))
    return cands[:k] if k and k < len(cands) else cands


def teacher_future(samples, step):
    P = np.array([[s.ego_state.rear_axle.x, s.ego_state.rear_axle.y, s.ego_state.rear_axle.heading]
                  for s in samples])
    px, py, pyaw = P[step]
    sel = slice(step + STRIDE_HIST, step + HORIZON * STRIDE_HIST + 1, STRIDE_HIST)
    c, s = math.cos(-pyaw), math.sin(-pyaw)
    dx, dy = P[sel, 0] - px, P[sel, 1] - py
    xy = np.stack([dx * c - dy * s, dx * s + dy * c], axis=-1)
    yaw = ((P[sel, 2] - pyaw + math.pi) % (2 * math.pi)) - math.pi
    return torch.tensor(xy, dtype=torch.float32), torch.tensor(yaw, dtype=torch.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--source", choices=("run", "navtrain"), default="run",
                    help="'run': sessions from closed-loop sim logs. 'navtrain': sessions built "
                         "from the per-frame bank's (log, token) pairs via navtrain_scenarios.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--rank", type=int, default=0,
                    help="WHICH AUGMENTED ROUTE these rollouts came from. ⛔ NOT cosmetic: the "
                         "paper requires the navigation command, the teacher trajectory and the "
                         "proposal scores to be MUTUALLY CONSISTENT, i.e. all from the SAME "
                         "augmented route. Without this field a rank-1 training tuple silently "
                         "receives rank-0 proposal scores. MEASURED 2026-09-20: that was happening "
                         "to 540 of 1,080 covered tuples, with the two routes' goals differing in "
                         "1,064/1,091 tuples (median 3.80 m, max 28.84 m) and the teacher targets "
                         "differing by a median of 2.418 m.")
    ap.add_argument("--candidates", type=int, default=0, help="0 = the whole set")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = every eligible frame")
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--frame-offset", type=int, default=0,
                    help="shift the frame grid; several passes with the SAME stride and DIFFERENT "
                         "offsets densify a bank in parallel without scoring a frame twice")
    ap.add_argument("--max-logs", type=int, default=0)
    ap.add_argument("--log-shard", default=None, metavar="i/N",
                    help="navtrain: process only logs i, i+N, i+2N ... Shard BY LOG; the frame-level "
                         "knobs cannot shard navtrain (each frame is a single-step scenario).")
    ap.add_argument("--report-cost", action="store_true")
    # ⭐ THE PAPER'S CONSTRUCTION. Without this flag the scorer builds every PDM target around the
    # CLOSED-LOOP run's ego, which drifted off the log -- in FOUR places: the scoring context's
    # history, the teacher trajectory the candidates are built around, the lane-graph anchor, and
    # therefore the candidate set itself. With it, the teacher trajectory is the per-frame rollout
    # from `build_teacher_rollouts.py` and the context is built at the LOGGED ego, so all three
    # training signals -- image, trajectory target, PDM targets -- describe the same pose.
    ap.add_argument("--perframe-bank", default=None,
                    help="per-frame teacher-rollout bank dir (build_teacher_rollouts.py output)")
    ap.add_argument("--resume", action="store_true",
                    help="APPEND to the output and skip frames already scored (it used to open with "
                         "'w' and TRUNCATE: MEASURED 2026-09-23, a console close killed four dev-box "
                         "shards at 71 %% and a restart would have redone all of it)")
    ap.add_argument("--resume-glob", default=None,
                    help="with --resume: also skip frames scored in any file matching this glob")
    ap.add_argument("--logs-file", default=None,
                    help="navtrain: score only the logs named in this file (split work across machines)")
    ap.add_argument("--perframe-file", default=None,
                    help="explicit bank file, e.g. <dir>/targets_aug.jsonl from augment_search.py; "
                         "rows carrying `aug` are scored under THEIR OWN augmented route/goal")
    a = ap.parse_args(argv)

    # ⛔ THE SCORER NEEDS THE SAME ROUTE AS THE TEACHER. `route_arc_length` walks the route the
    # feature builder chose, so without the rank-k patch a rank-1 bank would measure PROGRESS
    # along the RANK-0 route -- a fourth inconsistent signal. See route_rank.py (R11).
    import route_rank
    print(f"  {route_rank.apply(a.rank)}")
    # the scorer's feature builder rebuilds the same static map geometry per frame
    if os.environ.get("REFE_MAP_CACHE", "1") != "0":
        import map_feature_cache
        map_feature_cache.install()
        print("  map-feature cache: ON (exact; REFE_MAP_CACHE=0 disables)")
    if os.environ.get("REFE_QUERY_CACHE", "1") != "0":
        import query_cache
        query_cache.install()
        print("  query cache: ON (exact; REFE_QUERY_CACHE=0 disables)")

    perframe = None
    if a.perframe_bank:
        perframe = {}
        # `--perframe-file` names the bank file explicitly: the diversity search writes
        # `targets_aug.jsonl`, which the rank-keyed glob below could never find -- as first written,
        # the augmented half could not be scored AT ALL.
        _files = ([a.perframe_file] if a.perframe_file else
                  sorted(glob.glob(os.path.join(a.perframe_bank, f"targets_rank{a.rank}.jsonl"))))
        for fp in _files:
            with open(fp, "r", encoding="utf-8") as fh_:
                for line in fh_:
                    if line.strip():
                        r = json.loads(line)
                        # ⛔ KEY ON THE SCENARIO TOKEN, NOT JUST THE LOG. MEASURED 2026-09-21:
                        # one log carries up to THREE scenarios, so (log, step) collided --
                        # 60 rollouts collapsed to 42 keys, and the losers were SCORED WITH
                        # ANOTHER SCENARIO'S TEACHER TRAJECTORY while the builder printed
                        # SCORER_TARGETS_OK. This package had already logged this exact defect
                        # as D-REFE-KEYCOLLIDE-1; I reintroduced it in a new code path.
                        perframe[(r["log_name"], r["token"], int(r["step"]))] = r
        if not perframe:
            print(f"  --perframe-bank {a.perframe_bank!r} holds no rank-{a.rank} rows -- refusing")
            return 4
        print(f"  PER-FRAME mode: {len(perframe):,} teacher rollouts from {a.perframe_bank}")

    # ⛔ AUGMENTED-BANK MODE. Engaged whenever any banked row carries `aug`.
    aug_mode = perframe is not None and any("aug" in r for r in perframe.values())
    BASE_GOAL_HORIZON_S = 12.0            # build_teacher_rollouts.build_planner's route_goal_horizon_s
    RLP = None
    if aug_mode:
        # ⛔ Refused rather than tolerated: with --rank 0 the route patch is never installed, so
        # `set_rank(k)` would change a variable nothing reads and every lane-rank row would be
        # scored WITHOUT its rank -- consistent-looking, wrong, and invisible to every gate. And the
        # output `rank` must be 1 so these rows join the augmented target rows in `ScorerBank`.
        if int(a.rank) != 1:
            print(f"  REFUSING: this bank carries per-row `aug` parameters; score it with --rank 1 "
                  f"(got --rank {a.rank}), or the route patch is never installed."); return 2
        import route_lane_rank_patch as RLP
        from nuplan.planning.script import driverl_runtime_map_features as _M
        if not (getattr(RLP, "_INSTALLED", False) and hasattr(_M._choose_route_edge, "__wrapped__")):
            print("  REFUSING: augmented mode needs the route patch installed and wrapping "
                  "`_choose_route_edge`, and it is not."); return 2
        n_aug = sum(1 for r in perframe.values() if "aug" in r)
        print(f"  AUGMENTED mode: {n_aug:,} rows carry per-row `aug`; each is scored under its own "
              f"lane rank / goal horizon")

    import scorer_gate as G
    import score_proposals as SP
    import augment_routes as AR

    cfg, _used = G.build_engine_config()
    calcs = SP.build_calculators(cfg)
    if a.source == "navtrain":
        # ⛔ navtrain sessions are driven BY THE PER-FRAME BANK, not by a directory scan: the
        # scorer must score exactly the frames that have a trajectory target, or the two training
        # signals are computed at different poses -- the defect the per-frame rebuild existed to
        # remove. A navtrain run without --perframe-bank has nothing to align to.
        if perframe is None:
            print("  --source navtrain requires --perframe-bank"); return 2
        # ⛔ A navtrain unit has exactly ONE step (0), so the frame-level shard knobs -- which are
        # how the val14 path is parallelised -- would produce an EMPTY bank with exit 0:
        # [0][1::5] == []. Shard navtrain by LOG (--max-logs / a log slice), never by frame.
        if a.frame_offset:
            print(f"  --frame-offset {a.frame_offset} is meaningless for --source navtrain: each "
                  f"frame is its own scenario with a single step 0, so any offset > 0 selects "
                  f"NOTHING and would write an empty bank. Shard by log instead."); return 2
        import navtrain_scenarios as NS
        dbs = NS.index_dbs()
        by_log = {}
        for (lg, tk, st) in perframe:
            by_log.setdefault(lg, []).append(tk)
        logs = sorted(by_log)
        if a.logs_file:
            want = {l.strip() for l in open(a.logs_file, encoding="utf-8") if l.strip()}
            logs = [l for l in logs if l in want]
            print(f"  --logs-file: {len(logs)} of the bank's logs")
        if a.max_logs:
            logs = logs[: a.max_logs]
        if a.log_shard:
            # applied AFTER --max-logs, exactly as in build_teacher_rollouts, so N shards of a
            # limited set partition that set rather than striding the whole corpus
            si, sn = (int(x) for x in a.log_shard.split("/"))
            if not (sn > 0 and 0 <= si < sn):
                print(f"  --log-shard {a.log_shard} invalid: need 0 <= i < N and N > 0"); return 2
            logs = logs[si::sn]
            print(f"  shard {si}/{sn}")
        # ⭐ SHUFFLED LOG ORDER (2026-09-24, --grow training). Logs used to be processed in SORTED
        # name order, i.e. by DATE -- so a bank read while it was still growing was the EARLY drives
        # only (one city, one season). Training now starts before data prep ends, so every partial
        # bank must be an unbiased sample. A seeded shuffle AFTER the shard split changes the order
        # the work is done in, never which rows exist or what they contain. REFE_SHUFFLE_LOGS=0 restores it.
        if os.environ.get("REFE_SHUFFLE_LOGS", "1") != "0":
            import random
            random.Random(20260924).shuffle(logs)
            print("  log order: SHUFFLED (seed 20260924)")
        print(f"  {len(logs)} navtrain logs, {sum(len(by_log[l]) for l in logs)} banked frames")
    else:
        if not a.run:
            print("  --run is required for --source run"); return 2
        logs = sorted(glob.glob(f"{a.run}/**/*.msgpack.xz", recursive=True))
        if a.max_logs:
            logs = logs[: a.max_logs]
        print(f"  {len(logs)} simulation logs under {a.run}")
    print(f"  stride {a.stride}  frame-stride {a.frame_stride}  "
          f"candidates {a.candidates or 'all'}")
    os.makedirs(a.out, exist_ok=True)
    suffix = ("" if not a.frame_offset else f"_off{a.frame_offset}") +              ("" if not a.rank else f"_rank{a.rank}")
    out_path = os.path.join(a.out, f"scorer_targets{suffix}.jsonl")
    # ⭐ RESUME. Frames are written one after another, each as ONE flushed write, so only a file's
    # LAST frame can be torn by a kill: every frame except each file's last is done; the last is
    # re-scored (the merge keeps the first copy of every candidate row, so nothing doubles).
    done: set = set()
    if a.resume:
        def _scan(path):
            keys, last = [], None
            if not os.path.exists(path):
                return set()
            with open(path, encoding="utf-8") as f_:
                for line in f_:
                    try:
                        r_ = json.loads(line) if line.strip() else None
                    except json.JSONDecodeError:
                        r_ = None
                    if r_ is None:
                        continue
                    k_ = (r_.get("log_name"), r_.get("token"), int(r_.get("step", 0)),
                          int(r_.get("rank", 0)))
                    if k_ != last:
                        keys.append(k_)
                        last = k_
            return set(keys[:-1])
        done |= _scan(out_path)
        if a.resume_glob:
            for q in sorted(glob.glob(a.resume_glob)):
                if os.path.abspath(q) != os.path.abspath(out_path):
                    done |= _scan(q)
        print(f"  resume: {len(done):,} frames already scored (own file + {a.resume_glob or 'none'})")
    n_rows = n_frames = n_abort = n_teacher_offroad = 0
    n_abort_ndiff = n_curb_not_offroad = 0
    abort_ndiff_hist: dict = {}
    t_start = time.time()
    if a.resume and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        # a kill can leave half a frame with no newline: end it, so the next frame is not fused
        # onto the fragment (same defect as build_teacher_rollouts.open_append)
        with open(out_path, "rb") as f_:
            f_.seek(-1, os.SEEK_END)
            if f_.read(1) != b"\n":
                with open(out_path, "ab") as g_:
                    g_.write(b"\n")
    with open(out_path, "a" if a.resume else "w", encoding="utf-8") as fh:
        # ⭐ ONE loop over "units". For `run` a unit is a sim-log PATH; for navtrain it is a ready
        # SCENARIO, because each navtrain frame is its own scenario. Flattening here rather than
        # nesting a second loop keeps the val14 body byte-for-byte what it was -- and the
        # perframe-driven `steps` line below needs no change at all: it keys on
        # (log_name, scenario_name), and for navtrain that already resolves to exactly [0].
        if a.source == "navtrain":
            units = []
            for lg in logs:
                # a resumed frame costs NO scenario build: filter before building
                toks = [tk for tk in by_log[lg] if (lg, tk, 0, int(a.rank)) not in done]
                if toks:
                    units.extend(NS.build_scenarios_for_log(dbs[lg], toks))
            print(f"  {len(units)} navtrain scenarios (one per frame)")
        else:
            units = logs
        for li, lp in enumerate(units):
            t0 = time.time()
            try:
                sess = LogSession(scenario=lp) if a.source == "navtrain" else LogSession(lp)
            except Exception as exc:
                print(f"    [{li}] LOAD FAILED {type(exc).__name__}: {str(exc)[:70]}")
                continue
            t_load = time.time() - t0
            if perframe is not None:
                # score exactly the frames the per-frame bank holds for THIS log, so every PDM
                # target joins a trajectory target built at the same logged pose
                lname, tok = sess.scenario.log_name, sess.scenario.scenario_name
                steps = sorted(s for (lg, tk, s) in perframe if lg == lname and tk == tok)
                steps = steps[a.frame_offset::max(a.frame_stride, 1)]
            else:
                steps = list(range(8 + a.frame_offset, sess.n - HORIZON * STRIDE_HIST - 1,
                                   a.frame_stride))
            if a.max_frames:
                steps = steps[: a.max_frames]
            kept = 0
            for step in steps:
                t1 = time.time()
                # ⛔⛔ PER-ROW AUGMENTATION. The diversity search writes rows whose alternative intent
                # is a PER-ROW parameter -- `lane_rank k` OR `goal_horizon_s h` -- and the paper
                # requires the SAME augmented route to score the student's proposals. A process-global
                # `--rank` would score every augmented row against lane-rank 1: R11 again, at the
                # row level. So the row's own parameter is applied here, and EVERY row sets its
                # state explicitly -- a non-augmented row resets to base, otherwise a goal horizon
                # from the previous row would leak into the next one's scoring context.
                if aug_mode:
                    _row = perframe.get((sess.scenario.log_name, sess.scenario.scenario_name, step))
                    _aug = (_row or {}).get("aug")
                    if _aug and _aug.get("kind") == "lane_rank":
                        RLP.set_rank(int(_aug["value"]))
                        sess.builder.config.route_goal_horizon_s = BASE_GOAL_HORIZON_S
                    elif _aug and _aug.get("kind") == "goal_horizon_s":
                        RLP.set_rank(0)
                        sess.builder.config.route_goal_horizon_s = float(_aug["value"])
                    else:
                        RLP.set_rank(0)
                        sess.builder.config.route_goal_horizon_s = BASE_GOAL_HORIZON_S
                sd, log_sd = sess.scenario_data(step, logged=perframe is not None)
                t_build = time.time() - t1
                if perframe is not None:
                    tr = np.asarray(perframe[(sess.scenario.log_name, sess.scenario.scenario_name,
                                              step)]["traj"], dtype=float)
                    gxy = torch.tensor(tr[:, :2], dtype=torch.float32)
                    gyw = torch.tensor(tr[:, 2], dtype=torch.float32)
                    anchor_state = sess.logged_ego(step)
                else:
                    gxy, gyw = teacher_future(sess.samples, step)
                    anchor_state = sess.samples[step].ego_state
                sd, _st = SP.enrich_lane_graph(
                    sd, sess.scenario.map_api,
                    AR._anchor_from_ego(anchor_state),
                    sess.init.route_roadblock_ids)
                cands = candidate_set(sd, gxy, gyw, a.candidates)
                t2 = time.time()
                scored = [(nm, SP.score_proposal_rollout(sd, log_sd, calcs, xy, yw,
                                                         stride=a.stride))
                          for nm, xy, yw in cands]
                t_score = time.time() - t2
                by = dict(scored)
                # PROGRESS -> the paper's bounded EP. PDM's ego-progress term is a RATIO against a
                # reference planner's progress, clipped to [0, 1]; our reference is the TEACHER's
                # own advance on this frame, which is the expert we are distilling. Clipping at 1.0
                # is the part that matters: a candidate that travels 1.5x further than the teacher
                # scores 1.0, not 1.5, so the component cannot reward overspeed the way the old
                # goal-reaching proxy did.
                t_adv = by["teacher"].get("progress.advance_m", float("nan"))
                for _nm, _r in by.items():
                    adv = _r.get("progress.advance_m", float("nan"))
                    if adv == adv and t_adv == t_adv and t_adv > 1e-3:
                        _r["progress.ep"] = max(0.0, min(1.0, adv / t_adv))
                    else:
                        _r["progress.ep"] = float("nan")
                # ⛔ PER-FRAME CONTROL -- AND ITS FIRST VERSION WAS WRONG IN A WAY THAT BIASED
                # THE BANK. It required the teacher's own path to score CLEAN. MEASURED 2026-09-20:
                # at step 100 of token 99ca544752f255ad the teacher GENUINELY crosses a curb --
                # OffRoad.info 1.0, reward -3.8131, CurbClearance.info 1.0 -- and the frame was
                # discarded. The teacher is a 93.61-PDMS policy, not a perfect one.
                # => that control encoded an assumption about the DATA, not a property of the
                # INSTRUMENT, and it silently removed exactly the hardest, most informative frames:
                # the ones where a scorer most needs to learn that a path is bad. All three bank
                # passes aborted the SAME frame, which is what gave it away -- three different frame
                # grids losing the same count is not random.
                # ⭐ THE CORRECT CONTROL IS ABOUT DISCRIMINATION, NOT ABOUT VIRTUE: refuse a frame
                # only when the scorer cannot tell the candidates apart at all, and RECORD the
                # teacher's verdict instead of using it as an entry condition.
                keys = [k for k in by["teacher"]
                        if not k.startswith("_") and not k.endswith("@last")]
                ndiff = 0
                for k in keys:
                    vals = set()
                    for c in by:
                        v = by[c].get(k)
                        if not isinstance(v, str) and v is not None:
                            vals.add(round(float(v), 6))
                    if len(vals) > 1:
                        ndiff += 1
                ok_curb = True
                if "over-curb" in by:
                    ok_curb = by["over-curb"].get("off_road.OffRoad.info", 0.0) == 1.0
                teacher_off_road = by["teacher"].get("off_road.OffRoad.info", 0.0) != 0.0
                # ⛔⛔ `ok_curb` WAS AN ENTRY CONDITION AND IT DISCARDED 44 % OF NAVTRAIN, FOR A
                # REASON THAT HAS NOTHING TO DO WITH THE FRAME. It requires the analytic
                # "over-curb" candidate to actually read off-road. On val14's curated urban
                # scenarios it always did, so it never fired: **0 aborts**. On the first navtrain
                # pilot it fired on **8 of 18** frames -- and the split counter shows **8/8 were
                # `!ok_curb` and 0/8 were `ndiff < 3`**, with ndiff on those very frames at
                # **19-20 signals separating**. The frames were maximally discriminative and were
                # thrown away anyway, under a message ("scorer could not tell candidates apart")
                # that was false for every one of them.
                # ⭐ This file's own rule, six lines up, already says why that is wrong: "THE
                # CORRECT CONTROL IS ABOUT DISCRIMINATION, NOT ABOUT VIRTUE: refuse a frame only
                # when the scorer cannot tell the candidates apart at all." `ok_curb` is precisely
                # the virtue condition that rule forbids. It is now a DIAGNOSTIC, not a gate.
                # ⚠️ Nothing is mislabelled by keeping the frame: every candidate's PDM values are
                # COMPUTED, so an over-curb candidate that stays on-road simply reports that.
                # (Same class as R12: a guard whose semantics do not match its purpose, losing data
                # in a biased way -- there by log length, here by road geometry.)
                if not ok_curb:
                    n_curb_not_offroad += 1
                if ndiff < 3:
                    n_abort += 1
                    n_abort_ndiff += 1
                    abort_ndiff_hist[ndiff] = abort_ndiff_hist.get(ndiff, 0) + 1
                    continue
                n_teacher_offroad += int(teacher_off_road)
                buf = []
                for nm, xy, yw in cands:
                    buf.append(json.dumps({
                        "log_name": sess.scenario.log_name,
                        "token": sess.scenario.scenario_name,
                        "scenario_type": sess.scenario.scenario_type,
                        "step": step, "candidate": nm, "rank": int(a.rank),
                        # the augmentation this frame was SCORED under -- auditable against the
                        # target row's own `aug`, which is what makes the pairing checkable
                        "aug": ((perframe.get((sess.scenario.log_name, sess.scenario.scenario_name,
                                               step)) or {}).get("aug") if aug_mode else None),
                        "teacher_off_road": bool(teacher_off_road),
                        "n_signals_differing": int(ndiff),
                        "traj": xy.tolist(), "yaw": yw.tolist(),
                        "targets": {k: v for k, v in by[nm].items()
                                    if not k.endswith("@last") and not k.startswith("_")},
                        "stride": a.stride,
                    }) + "\n")
                    n_rows += 1
                # ONE write per frame, flushed: a kill can tear at most the file's last frame
                fh.write("".join(buf))
                fh.flush()
                kept += 1
                n_frames += 1
                if a.report_cost and kept == 1:
                    print(f"      cost: log load {t_load:5.2f} s | ScenarioData {t_build*1000:6.0f} ms"
                          f" | {len(cands)} candidates scored in {t_score:5.2f} s")
            # ⚠️ a unit is a PATH for `run` and a SCENARIO for navtrain, so the label cannot be
            # Path(lp) -- that raised TypeError and killed the whole navtrain run at the progress
            # line, after every frame had already been scored.
            label = (sess.scenario.log_name if a.source == "navtrain" else Path(lp).parts[-3])
            print(f"    [{li+1}/{len(units)}] {label[:34]:34s} "
                  f"frames {kept:3d}/{len(steps):3d}  rows {n_rows:6d}  "
                  f"aborted {n_abort:3d}  {time.time()-t0:6.1f} s")
    dt = time.time() - t_start
    print(f"\n  wrote {n_rows:,} rows over {n_frames:,} frames to {out_path}")
    print(f"  aborted frames (scorer could not tell candidates apart): {n_abort}")
    print(f"    all aborts are ndiff < 3 (the DISCRIMINATION rule): {n_abort_ndiff}  "
          f"histogram {dict(sorted(abort_ndiff_hist.items()))}")
    print(f"  DIAGNOSTIC (not a gate): frames whose over-curb candidate stayed ON-road: "
          f"{n_curb_not_offroad}  -- these are KEPT; on navtrain this was 8/18, and gating on it "
          f"discarded frames with 19-20 separating signals")
    print(f"  frames where the TEACHER ITSELF reads off-road: {n_teacher_offroad} "
          f"({100.0*n_teacher_offroad/max(n_frames,1):.1f} %) -- BANKED and flagged, not discarded")
    print(f"  wall clock {dt/60:.1f} min  =>  {dt/max(n_frames,1):.2f} s/frame")
    # the ego-view lever's own accounting: a run where every prefix FELL BACK is not a run that
    # used the lever, and an A/B over it would pass vacuously (diag_scorer_ego_view.py asserts it)
    print(f"  ego view: {'ON' if SP.EGO_VIEW else 'OFF'} {json.dumps(SP.EGO_VIEW_STATS)}")
    json.dump({"run": a.run, "rank": a.rank, "rows": n_rows, "frames": n_frames,
               "aborted": n_abort,
               "teacher_off_road_frames": n_teacher_offroad,
               "stride": a.stride, "frame_stride": a.frame_stride,
               "candidates": a.candidates or "all", "seconds": dt,
               "ego_view": bool(SP.EGO_VIEW), "ego_view_stats": dict(SP.EGO_VIEW_STATS)},
              open(os.path.join(a.out, "scorer_targets_stats.json"), "w"), indent=1)
    print("SCORER_TARGETS_OK" if n_rows else "SCORER_TARGETS_EMPTY")
    return 0 if n_rows else 1


if __name__ == "__main__":
    sys.exit(main())
