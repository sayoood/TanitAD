"""P6 / D7 -- make a lane-rank TWIN be scored on ITS OWN route, in the process that scores it.

⛔⛔ THE DEFECT THIS CLOSES (MEASURED 2026-10-16, Master Mind probe `d7_probe.py` on the pod, twin 2021.06.23.18.23.38_veh-26_00663_01217 / 88362ada700b543d):
P6's chain REFUSED at D7 with `ZZD7_DISTINCT_BAD k=1 sampled=40 differ=0` -- all 32,929 rank-1 twin parts were byte-for-byte the LOGGED route's parts.  Two causes:

  H1  `build_metric_cache.cmd_build` applied the DriveRL route patch (`route_rank.apply`) in the PARENT only; its `spawn` pool workers start fresh, and the teacher venv has
      no DriveRL `sitecustomize` (the system one is `/usr/lib/python3.12/sitecustomize.py`), so a worker never had the patch.
  H2  (decisive) EVEN IN-PROCESS the patch changes nothing: `driverl_runtime_map_features._choose_route_edge` is called 0 times during
      `navsim_pdm_targets.metric_cache_parts`.  NAVSIM's PDM-Closed chooses its own starting lane (`AbstractPDMPlanner._get_starting_lane`, navsim @ 3e8291b
      abstract_pdm_planner.py:99-127: the on-route lane that contains the ego's rear axle with the lowest heading error) and runs its own centreline search from it
      (`_get_discrete_centerline`, :75-97, a Dijkstra over the route's lane graph).  It never consults DriveRL's route builder.

THE FIX.  The twin's route is "the rank-k lane at the route's FIRST roadblock, later roadblocks by the normal rule" (code/route_lane_rank_patch.py).  PDM-Closed's only
lane-level choice is its STARTING LANE; its centreline is the Dijkstra path from there through the SAME roadblocks.  So this module wraps `_get_starting_lane`:

    lane0   = PDM's own choice (so its roadblock = the roadblock the ego is on, after PDM's own route correction)
    edges   = the interior lanes of lane0's roadblock that have a baseline path
    ordered = sorted(edges, key=DriveRL `_route_edge_anchor_score(edge, anchor))`     <- THE SAME ORDERING and THE SAME ANCHOR as DriveRL
    return    ordered[min(k, len(ordered) - 1)]                                       <- the SAME clamp as route_lane_rank_patch.patched

with `anchor` = DriveRL's `_frame_from_ego_state(ego_state)` (feature_builder.py:960-974): the ego REAR AXLE x, y, heading.  Roadblocks, drivable-area map, route lane ids and
every scorer input other than the centreline / PDM-Closed's reference trajectory are untouched: only the starting lane, and hence the centreline, differs.

⭐ CONTROL BY CONSTRUCTION: rank 0 installs NOTHING (NAVSIM is byte-untouched) and `apply(0)` REFUSES if a patch is already there -- exactly `route_rank.apply`.
⭐ THE SECOND CAUSE OF THE DEFECT WAS A PROCESS BOUNDARY: a patch installed in the parent never reaches a `spawn` worker.  So this module's state is asserted IN THE PROCESS THAT SCORES
(`ensure()`, called by the pool initializer AND again by every job), on the very planner object that will build the parts (`assert_active(obj=...)`), and never inferred from a flag.

Pure standard library; NAVSIM / DriveRL imports are lazy and replaceable (`_planner_cls`, `_driverl`) so the logic is unit-testable with fakes (test_pdm_route_rank.py).
"""
from __future__ import annotations

import os

ENV = "DRIVERL_EVAL_ROUTE_LANE_RANK"
MARK = "_d7_rank"                    # attribute on the installed wrapper: the rank it serves
MUTATE_ENV = "REFE_D7_MUTATE"        # TEST-ONLY mutation arm, read by build_metric_cache.cmd_build / _build_log; unset in every launch


# ------------------------------------------------------------------------------------------------ lazy imports (replaceable in tests)
def _planner_cls():
    from navsim.planning.simulation.planner.pdm_planner.abstract_pdm_planner import AbstractPDMPlanner
    return AbstractPDMPlanner


def _driverl():
    from nuplan.planning.script import driverl_runtime_map_features as M
    return M


# ------------------------------------------------------------------------------------------------ the rule
def anchor_from_ego(ego_state, M=None):
    """DriveRL's anchor, field for field: `driverl.nuplan.feature_builder._frame_from_ego_state` (DriveRL @ /workspace/dz/DriveZero/DriveRL, feature_builder.py:960-974)."""
    M = M or _driverl()
    dyn = ego_state.dynamic_car_state
    return M.FrameRow(
        token=b"current", token_hex="current", timestamp_us=int(ego_state.time_us), scene_token=None,
        x=float(ego_state.rear_axle.x), y=float(ego_state.rear_axle.y), yaw=float(ego_state.rear_axle.heading),
        vx=float(dyn.rear_axle_velocity_2d.x), vy=float(dyn.rear_axle_velocity_2d.y),
        ax=float(dyn.rear_axle_acceleration_2d.x), ay=float(dyn.rear_axle_acceleration_2d.y), yaw_rate=float(dyn.angular_velocity))


def ordered_lanes(edges, anchor, M=None) -> list:
    """The roadblock's lanes with a baseline path, nearest-first by DriveRL's own score (`_route_edge_anchor_score`: distance + 5 * heading error)."""
    M = M or _driverl()
    cands = [e for e in edges if M._edge_centerline_coords(e)]
    return sorted(cands, key=lambda e: M._route_edge_anchor_score(e, anchor))


def rank_k_lane(edges, anchor, rank: int, M=None):
    """`route_lane_rank_patch.patched` at prev_edge=None, verbatim: ordered[min(rank, len(ordered) - 1)] (it CLAMPS); None when no lane has a baseline."""
    ordered = ordered_lanes(edges, anchor, M)
    if not ordered:
        return None
    return ordered[min(int(rank), len(ordered) - 1)]


# ------------------------------------------------------------------------------------------------ the patch
def _make_patched(original, rank: int, M):
    def _get_starting_lane(self, ego_state):
        lane0 = original(self, ego_state)
        if lane0 is None:
            raise RuntimeError("pdm_route_rank: PDM-Closed found no starting lane; a rank-%d twin cannot be derived" % rank)
        block = (getattr(self, "_route_roadblock_dict", None) or {}).get(lane0.get_roadblock_id())
        if block is None:
            raise RuntimeError("pdm_route_rank: PDM-Closed's starting roadblock %r is not in its route dict" % (lane0.get_roadblock_id(),))
        ordered = ordered_lanes(list(block.interior_edges), anchor_from_ego(ego_state, M), M)
        if not ordered:
            raise RuntimeError("pdm_route_rank: roadblock %r has no lane with a baseline path" % (block.id,))
        lane = ordered[min(rank, len(ordered) - 1)]
        # diagnostics for the verification harness (never read by the build): what was chosen from what
        self._d7_last = {"rank": rank, "lane0": getattr(lane0, "id", None), "lane": getattr(lane, "id", None), "n_lanes": len(ordered),
                         "clamped": rank > len(ordered) - 1, "order": [getattr(e, "id", None) for e in ordered],
                         "lane0_pos": next((i for i, e in enumerate(ordered) if getattr(e, "id", None) == getattr(lane0, "id", None)), None)}
        return lane
    setattr(_get_starting_lane, MARK, rank)
    _get_starting_lane.__wrapped__ = original
    return _get_starting_lane


def installed_rank(cls=None):
    """The rank the class-level patch serves, or None when `_get_starting_lane` is NAVSIM's own."""
    cls = cls or _planner_cls()
    return getattr(cls._get_starting_lane, MARK, None)


def apply(rank: int) -> str:
    """Install (rank > 0) or verify the absence of (rank 0) the PDM starting-lane patch in THIS process; return a one-line state.  SystemExit on a silent mismatch."""
    rank = int(rank)
    if rank < 0:
        raise SystemExit("pdm_route_rank: rank %d < 0" % rank)
    env = os.environ.get(ENV, "").strip()
    if env and int(env) != rank:
        raise SystemExit(f"pdm_route_rank: env {ENV}={env} but rank {rank}. Refusing: the process would score one route and label it another.")
    cls = _planner_cls()
    cur = cls._get_starting_lane
    have = getattr(cur, MARK, None)
    if rank == 0:
        if have is not None or hasattr(cur, "__wrapped__"):
            raise SystemExit("pdm_route_rank: rank 0 requested but a starting-lane patch is ALREADY installed (rank %s). Rank 0 must run NAVSIM's unpatched code." % have)
        return "pdm route rank 0: NAVSIM's own _get_starting_lane (no patch installed; byte-identical to the logged-route build by construction)"
    if have is not None:
        if have == rank:
            return "pdm route rank %d: patch already active in this process" % rank
        raise SystemExit("pdm_route_rank: a rank-%s patch is already installed but rank %d was requested: one process serves ONE rank" % (have, rank))
    cls._get_starting_lane = _make_patched(cur, rank, _driverl())
    assert_active(rank)
    return "pdm route rank %d: _get_starting_lane PATCHED (AbstractPDMPlanner), verified wrapped" % rank


def assert_active(rank: int, obj=None) -> None:
    """Raise SystemExit unless the state matches `rank` -- on the CLASS and, when given, on the very planner OBJECT that will build the parts (a subclass override or an
    instance attribute would shadow a class-level patch and leave a success flag disconnected from the thing it describes)."""
    rank = int(rank)
    fns = [("class AbstractPDMPlanner", _planner_cls()._get_starting_lane)]
    if obj is not None:
        b = obj._get_starting_lane
        fns.append(("planner object %s" % type(obj).__name__, getattr(b, "__func__", b)))
    for what, fn in fns:
        have = getattr(fn, MARK, None)
        if rank == 0:
            if have is not None or hasattr(fn, "__wrapped__"):
                raise SystemExit("pdm_route_rank: rank 0 but %s._get_starting_lane is patched (rank %s)" % (what, have))
        else:
            if have != rank or not callable(getattr(fn, "__wrapped__", None)):
                raise SystemExit("pdm_route_rank: rank %d requested but %s._get_starting_lane is %s (a success flag disconnected from the thing it describes)"
                                 % (rank, what, "NOT wrapped" if not hasattr(fn, "__wrapped__") else "rank %s" % have))


def ensure(rank: int, obj=None) -> str:
    """apply(rank) then assert_active(rank, obj): the one call a worker makes.  `obj` = the live PDM-Closed planner (navsim_pdm_targets._setup()["proc"]._pdm_closed)."""
    msg = apply(rank)
    assert_active(rank, obj)
    return msg
