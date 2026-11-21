"""Runtime injection for Stage-1 goal augmentation: choose the RANK-k lane instead of the nearest.

THE PROBLEM. Their runner exposes goal-SHAPING knobs (route_goal_horizon_s, min_speed_mps,
pair_mode) but NO way to replace the route. Roadblock-id injection is available
(`feature_builder._corrected_route_roadblock_ids`, set after `planner.initialize`) but it cannot
express a LANE alternative: a lane change keeps the same roadblocks and differs only in which
interior edge is chosen. That choice lives in
`nuplan.planning.script.driverl_runtime_map_features._choose_route_edge`.

THE PATCH. Wrap that one function. On the FIRST roadblock of a route build (prev_edge is None) it
returns the rank-k nearest lane instead of the nearest; every later roadblock falls through to THEIR
unmodified rule, so continuity is theirs, not ours.

⭐ THE CONTROL, and the reason this design was chosen: with rank 0 -- or with the variable unset --
the wrapper is NOT installed at all, so the run is byte-identical to Stage 0 by construction rather
than by argument. Enable with DRIVERL_EVAL_ROUTE_LANE_RANK=<k>.

Installed via sitecustomize.py in the venv (the same mechanism as the fcntl shim), because their
harness spawns its own python and there is no other injection seam.
"""
from __future__ import annotations
import os

_INSTALLED = False
_RANK = 0


def set_rank(k: int) -> None:
    """Change the rank the NEXT route build uses, without reinstalling the patch.

    ⭐ FOR THE PER-FRAME DIVERSITY SEARCH. A FIXED rank is a weak augmentation on navtrain, because
    `patched` selects `ordered[min(rank, len(ordered) - 1)]` -- it CLAMPS. MEASURED over 280
    navtrain frames in 12 logs across all four cities, the route's FIRST roadblock has:
        1 lane  43.9 %   |   2 lanes 28.9 %   |   >=3 lanes 27.1 %
    so rank 1 silently reproduces rank 0 on ~44 % of frames, and rank 2 reproduces rank 1 on ~73 %
    -- with the distinct cases 80 % Las Vegas (Boston 2.7 %, Singapore 0 %, Pittsburgh 0 %), i.e. a
    geographic bias on top of the duplication. The search therefore varies the parameter PER FRAME
    until the teacher's trajectory actually moves, which requires this to be mutable mid-process.

    ⛔ THE RANK-0 CONTROL IS NOT WEAKENED, because it never depended on this function. `set_rank(0)`
    makes `patched` DELEGATE to the original -- behaviourally identical, but only "by argument".
    The real control is unchanged: the rank-0 bank is built by a process that never calls
    `install()`, so the wrapper is ABSENT and the run is byte-identical to Stage 0 **by
    construction**. The search is a SECOND PASS over that finished bank and never emits rank-0 rows.
    """
    global _RANK
    _RANK = int(k)


def current_rank() -> int:
    return _RANK


def install(rank: int | None = None) -> bool:
    """Patch _choose_route_edge to prefer the rank-k lane at the route's first roadblock."""
    global _INSTALLED, _RANK
    if _INSTALLED:
        if rank is not None:
            _RANK = int(rank)
        return True
    if rank is None:
        raw = os.environ.get("DRIVERL_EVAL_ROUTE_LANE_RANK", "").strip()
        if not raw:
            return False
        rank = int(raw)
    if rank == 0:
        return False                      # identity: leave THEIR code untouched
    _RANK = int(rank)
    from nuplan.planning.script import driverl_runtime_map_features as M
    original = M._choose_route_edge

    def patched(edges, anchor, prev_edge):
        if prev_edge is not None:
            return original(edges, anchor, prev_edge)      # their rule, untouched
        if _RANK == 0:
            return original(edges, anchor, prev_edge)      # delegate -- see set_rank()
        cands = [e for e in edges if M._edge_centerline_coords(e)]
        if not cands:
            return None
        ordered = sorted(cands, key=lambda e: M._route_edge_anchor_score(e, anchor))
        return ordered[min(_RANK, len(ordered) - 1)]

    patched.__wrapped__ = original
    M._choose_route_edge = patched
    _INSTALLED = True
    print(f"[route_lane_rank_patch] ACTIVE rank={rank} -- route_0 semantics are NOT in effect",
          flush=True)
    return True
