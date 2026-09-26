"""K=2 with GUARANTEED diversity: search the augmentation space per frame until the teacher moves.

⛔ WHY A FIXED RANK IS NOT GOOD ENOUGH. `route_lane_rank_patch` picks
`ordered[min(rank, len(ordered) - 1)]` -- it CLAMPS to the last lane. MEASURED over 280 navtrain
frames in 12 logs spanning all four cities, the route's FIRST roadblock carries:

    1 lane  43.9 %   |   2 lanes 28.9 %   |   >=3 lanes 27.1 %

so a fixed rank 1 returns THE SAME EDGE as rank 0 on ~44 % of navtrain frames, and rank 2 repeats
rank 1 on ~73 %. Worse, the frames where a high rank IS distinct are **80 % Las Vegas** (Boston
2.7 %, Singapore 0 %, Pittsburgh 0 %), so a fixed-rank K=3 would have paid 3x to add a third target
that is mostly one city. ⚠️ Duplicated ranks are the R6/R11 failure: both banks stay internally
consistent, every gate passes, and the duplicate reads as data.

⭐ THE FIX, per the paper. DriveZero augments *"the route intent and its associated navigation
command"* before generating teacher supervision, and requires the SAME augmented route to be used
when scoring student proposals. Nothing says the augmentation must be one fixed knob. So this pass
walks a LADDER per frame and keeps the first candidate whose teacher trajectory actually differs:

    1. lane ranks 1 .. min(L-1, MAX_RANK)   -- skipped entirely when L == 1, which a cheap map
                                               query settles BEFORE paying a 5.94 s rollout
    2. goal horizons 8 / 16 / 20 / 6 s      -- works on single-lane roads, where no rank can help,
                                               and carries no city bias

Both knobs are per-frame settable and that was verified, not assumed: `route_lane_rank_patch`
exposes `set_rank()`, and `feature_builder._build_goal_positions` reads
`self.config.route_goal_horizon_s` on EVERY call (l.552), so mutating
`planner._feature_builder.config.route_goal_horizon_s` changes the next rollout.

⭐⭐ THE ACCEPTED PARAMETER IS WRITTEN INTO THE ROW (`aug`). This is the part that matters most.
Today the rank is a PROCESS-GLOBAL flag, and R11 happened precisely because the scorer did not share
it. A per-row, self-describing augmentation means the scorer reads the parameter out of the very
bank it is scoring, so the teacher trajectory, the navigation command and the PDM scores cannot
disagree by construction rather than by discipline.

⛔ This is a SECOND PASS over a finished rank-0 bank; it never emits rank-0 rows. The rank-0 control
is therefore untouched: that bank is built by a process that never installs the patch at all.

Usage:
  python augment_search.py --bank <rank0 dir> --out <dir> [--tau 0.5] [--limit-frames N]
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

import build_teacher_rollouts as BR
import navtrain_scenarios as NS

# ⚠️ A DESIGN CHOICE, NOT A MEASUREMENT, and stated as one. From the val14 bank (n=1,101) the
# rank-1 divergence is median 0.673 m, p10 0.031 m, p90 2.532 m. 0.5 m sits above the rig's
# camera-to-state sync term (MEASURED max 0.543 m in the P1 arm) and below the P2 gate's 1.0 m.
# A target differing by a few centimetres is not a second target -- under winner-takes-all it just
# re-weights that frame.
DEFAULT_TAU_M = 0.5
MAX_RANK = 3
GOAL_HORIZONS = (8.0, 16.0, 20.0, 6.0)   # base is 12.0
BASE_HORIZON = 12.0


def lane_count(sc) -> int:
    """Lanes at the route's FIRST roadblock -- the only place the rank patch acts.

    Cheap (~3 ms) and it decides, BEFORE any rollout, whether a lane rank can possibly differ.
    Without this the 43.9 % single-lane frames would each burn a 5.94 s rollout to rediscover that
    the patch clamped.
    """
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    try:
        ids = sc.get_route_roadblock_ids()
        if not ids:
            return 0
        m = sc.map_api
        rb = m.get_map_object(str(ids[0]), SemanticMapLayer.ROADBLOCK)
        if rb is None:
            rb = m.get_map_object(str(ids[0]), SemanticMapLayer.ROADBLOCK_CONNECTOR)
        return len(rb.interior_edges) if rb is not None else 0
    except Exception:
        return 0


def divergence_m(traj_a, traj_b) -> float:
    """Max per-step displacement between two ego-frame trajectories, in metres."""
    if traj_a is None or traj_b is None or len(traj_a) != len(traj_b):
        return float("inf")
    return max(math.dist(p[:2], q[:2]) for p, q in zip(traj_a, traj_b))


def candidates(L: int):
    """The ladder, cheapest and most paper-aligned first."""
    out = []
    for k in range(1, min(L, MAX_RANK + 1)):
        out.append(("lane_rank", k))
    for h in GOAL_HORIZONS:
        out.append(("goal_horizon_s", h))
    return out


def apply_candidate(planner, kind, value) -> None:
    import route_lane_rank_patch as P
    if kind == "lane_rank":
        P.set_rank(int(value))
        planner._feature_builder.config.route_goal_horizon_s = BASE_HORIZON
    else:
        P.set_rank(0)
        planner._feature_builder.config.route_goal_horizon_s = float(value)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help="finished rank-0 bank directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU_M)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-frames", type=int, default=0)
    ap.add_argument("--log-shard", default=None, metavar="i/N")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-glob", default=None,
                    help="with --resume: a glob over OTHER shards' output DIRS; frames they have "
                         "already searched (accepted OR not) are skipped")
    ap.add_argument("--logs-file", default=None,
                    help="search only the logs named in this file (split work across machines)")
    a = ap.parse_args(argv)

    import route_lane_rank_patch as P
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization

    # install once with a non-zero rank so the wrapper exists; set_rank() drives it per frame.
    if not P.install(1):
        print("  FAILED to install the route patch -- refusing to search"); return 2
    from nuplan.planning.script import driverl_runtime_map_features as M
    if not hasattr(M._choose_route_edge, "__wrapped__"):
        print("  patch reports installed but _choose_route_edge is not wrapped"); return 2
    print("  route patch ACTIVE, rank driven per frame by set_rank()")

    rows = []
    for fp in sorted(glob.glob(os.path.join(a.bank, "targets_rank0.jsonl"))) or \
              sorted(glob.glob(os.path.join(a.bank, "r0_shard*", "targets_rank0.jsonl"))):
        rows += [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
    if not rows:
        print(f"  no rank-0 rows under {a.bank} -- refusing"); return 4
    by_log = {}
    for r in rows:
        by_log.setdefault(r["log_name"], []).append(r)
    names = sorted(by_log)
    if a.logs_file:
        want = {l.strip() for l in open(a.logs_file, encoding="utf-8") if l.strip()}
        names = [n for n in names if n in want]
        print(f"  --logs-file: {len(names)} of the bank's logs")
    if a.log_shard:
        i, n = (int(x) for x in a.log_shard.split("/"))
        names = names[i::n]
        print(f"  shard {i}/{n}")
    # ⭐ SHUFFLED LOG ORDER (2026-09-24, --grow training). Logs used to be processed in SORTED
    # name order, i.e. by DATE -- so a bank read while it was still growing was the EARLY drives
    # only (one city, one season). Training now starts before data prep ends, so every partial
    # bank must be an unbiased sample. A seeded shuffle AFTER the shard split changes the order
    # the work is done in, never which rows exist or what they contain. REFE_SHUFFLE_LOGS=0 restores it.
    if os.environ.get("REFE_SHUFFLE_LOGS", "1") != "0":
        import random
        random.Random(20260924).shuffle(names)
        print("  log order: SHUFFLED (seed 20260924)")
    print(f"  {len(rows):,} rank-0 rows over {len(by_log)} logs; searching {len(names)} logs, "
          f"tau={a.tau} m")

    # ⛔ an augmented twin must be rolled out at the SAME rate as the rank-0 row it pairs with
    bank_rates = {int(r.get("sim_hz", 20)) for r in rows}
    if bank_rates != {NS.SIM_HZ}:
        print(f"  REFUSING: the rank-0 bank is at {sorted(bank_rates)} Hz but REFE_SIM_HZ is "
              f"{NS.SIM_HZ} -- a twin and its rank-0 row would come from different simulations.")
        return 5
    print(f"  simulation rate: {NS.SIM_HZ} Hz (REFE_SIM_HZ), matching the rank-0 bank")
    os.makedirs(a.out, exist_ok=True)
    W = BR.RowWriter(os.path.join(a.out, "targets_aug.jsonl"), a.resume,
                     os.path.join(a.resume_glob, "targets_aug.jsonl") if a.resume_glob else None)
    # ⛔ A FRAME WITH NO ALTERNATIVE WRITES NO ROW -- only its stats line. Resuming on rows alone
    # re-searched every such frame (~29 %, up to 7 rollouts each). The stats lines are the progress.
    searched: set = set()
    if a.resume:
        dirs = [a.out] + (sorted(glob.glob(a.resume_glob)) if a.resume_glob else [])
        for d in dirs:
            sp = os.path.join(d, "aug_stats.jsonl")
            if not os.path.exists(sp):
                continue
            with open(sp, encoding="utf-8") as fs:
                for line in fs:
                    try:
                        q = json.loads(line) if line.strip() else None
                    except json.JSONDecodeError:
                        continue
                    if q is not None:
                        searched.add((q.get("log_name"), q.get("token")))
        print(f"  resume: {len(searched):,} frames already searched (stats lines, all shards)")
    SF = BR.open_append(os.path.join(a.out, "aug_stats.jsonl"))
    dbs = NS.index_dbs()
    BR.TEACHER_DEVICE = a.device                 # recorded in every row this search writes
    planner = BR.build_planner(a.device)
    t0 = time.time()
    stats = {"frames": 0, "accepted": 0, "attempts": 0, "no_aug": 0, "by_kind": {}, "lane1": 0}

    for lg in names:
        base = {r["token"]: r for r in by_log[lg]}
        todo = [t for t in base if not W.has(lg, t, 0) and (lg, t) not in searched]
        if a.limit_frames:
            todo = todo[:a.limit_frames]
        if not todo:
            continue
        cams, cam_ts = BR.camera_arrays(dbs[lg])
        if cams is None:
            print(f"    {lg[:28]}: missing a camera channel, skipped"); continue
        kept = 0
        for sc in NS.build_scenarios_for_log(dbs[lg], todo):
            tok = sc._initial_lidar_token
            ref = base[tok]["traj"]
            L = lane_count(sc)
            if L <= 1:
                stats["lane1"] += 1
            controller = BR.build_controller(sc, a.device)
            chosen = None
            tried = []
            for kind, value in candidates(L):
                apply_candidate(planner, kind, value)
                planner.initialize(PlannerInitialization(
                    route_roadblock_ids=sc.get_route_roadblock_ids(),
                    mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
                stats["attempts"] += 1
                try:
                    row = BR.make_row(sc, planner, controller, 0, cams, cam_ts, 1)
                except Exception as exc:
                    # same backstop as build_teacher_rollouts: one bad attempt costs one attempt
                    BR._reset_planner_state(planner)
                    tried.append([kind, value, f"ERR {type(exc).__name__}"])
                    print(f"    ATTEMPT_ERROR {lg[:28]} {tok} {kind}={value}: "
                          f"{type(exc).__name__}: {str(exc)[:70]}", flush=True)
                    continue
                if row is None:
                    tried.append([kind, value, None])
                    continue
                d = divergence_m(row["traj"], ref)
                tried.append([kind, value, round(d, 4)])
                if d >= a.tau:
                    row["aug"] = {"kind": kind, "value": value, "divergence_m": round(d, 4),
                                  "lanes": L}
                    chosen = row
                    stats["by_kind"][kind] = stats["by_kind"].get(kind, 0) + 1
                    break
            stats["frames"] += 1
            if chosen is not None:
                W.write(chosen); kept += 1; stats["accepted"] += 1
                # the row must be DURABLE before its stats line says the frame is done, or a kill
                # between the two would lose an accepted target that resume then never re-searches
                W.flush()
            else:
                # ⛔ NOT padded with a near-duplicate. A frame with no reachable alternative intent
                # contributes ONE target, and the count says so.
                stats["no_aug"] += 1
            # ⭐ PER-FRAME STATS, APPENDED AS THEY HAPPEN. The first pilot printed its cost summary
            # only at the end, was killed before the end, and left 20 accepted rows but NOT ONE of
            # the numbers it existed to measure (attempts per frame, no-alternative rate). A pilot
            # whose findings live only in its closing print is a pilot that can report nothing.
            SF.write(json.dumps({"log_name": lg, "token": tok, "lanes": L, "attempts": len(tried),
                                 "tried": tried,
                                 "accepted": (None if chosen is None else
                                              [chosen["aug"]["kind"], chosen["aug"]["value"]])})
                     + "\n")
            SF.flush()
        W.flush()
        print(f"    {lg[:28]:28s} frames {len(todo):4d}  augmented {kept:4d}  "
              f"{time.time()-t0:7.1f}s")
    W.close()
    # restore the base configuration so nothing downstream inherits a search state
    apply_candidate(planner, "goal_horizon_s", BASE_HORIZON); P.set_rank(0)
    f = max(stats["frames"], 1)
    print(f"\n  frames searched      : {stats['frames']:,}")
    print(f"  augmented (>= {a.tau} m) : {stats['accepted']:,} ({100*stats['accepted']/f:.1f} %)")
    print(f"  NO alternative found : {stats['no_aug']:,} ({100*stats['no_aug']/f:.1f} %)")
    print(f"  single-lane frames   : {stats['lane1']:,} ({100*stats['lane1']/f:.1f} %)")
    print(f"  rollouts per frame   : {stats['attempts']/f:.2f}  <- the cost multiplier")
    print(f"  accepted by knob     : {stats['by_kind']}")
    print(f"  wall {(time.time()-t0)/60:.1f} min for {stats['attempts']:,} rollouts")
    print("AUGMENT_SEARCH_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
