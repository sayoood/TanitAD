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

⛔⛔ THE ROUTE MUST BE HELD FIXED OVER A ROLLOUT (2026-10-16, `raw/2026-10-16-twin-reroll/`).
The teacher's feature builder REBUILDS its route on EVERY step from the CURRENT ego pose
(`feature_builder._build_map_features` -> `_sample_route_points_from_map(anchor=<current ego>)`,
DriveRL `src/driverl/nuplan/feature_builder.py:398-418`). The original wrapper therefore re-applied
"rank k" against the CURRENT anchor at every step: once the ego moved toward the rank-1 lane, the
"rank-1 lane" relative to the new pose became the LOGGED lane again, and the teacher was pulled back.
MEASURED: the banked rank-K twins were closer to the rank-K centreline than to the logged one on
1/34, 0/7, 0/4 (P6 D7 check (c)) -- perturbations of the logged route, not alternative intents.
THE FIX: inside a rollout scope (`held_route()`, entered by `build_teacher_rollouts.rollout_from_frame`)
the lane chain chosen by the FIRST route build -- rank-k at the step-0 anchor, their continuity rule
after it -- is RECORDED, and every later build of that rollout returns the recorded lane for any
roadblock it covers. The route is the step-0 rank-k route for the whole rollout, as the paper's
"augmented route intent" requires. `REFE_ROUTE_HOLD=0` restores the per-step re-ranking (the
reference arm, kept ONLY to reproduce the defect).
  * rank 0 is UNTOUCHED: `_RANK == 0` delegates before any hold logic, so goal-horizon twins
    (`set_rank(0)`) and the rank-0 bank keep THEIR per-step nearest-lane rule byte for byte.
  * outside a `held_route()` scope nothing changes (single-build callers such as
    `build_scorer_targets.py` and `pdm_route_rank.py`'s oracle see the old behaviour exactly).
  * NOT thread-safe by design: one rollout at a time per process, as every harness here runs.

Installed via sitecustomize.py in the venv (the same mechanism as the fcntl shim), because their
harness spawns its own python and there is no other injection seam.
"""
from __future__ import annotations
import contextlib
import os

_INSTALLED = False
_RANK = 0
_HOLD = None          # the active rollout's hold state, or None outside a held_route() scope
LAST_HOLD = None      # the most recent scope's final state (read by drivers and tests)


def hold_enabled() -> bool:
    """The route is held fixed over a rollout unless REFE_ROUTE_HOLD=0 (reference arm only)."""
    return os.environ.get("REFE_ROUTE_HOLD", "1").strip() != "0"


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


def _new_hold() -> dict:
    return {"builds": 0, "recording": False, "rec": [], "chain_ids": None, "hits": 0, "misses": 0,
            "rank": _RANK}


@contextlib.contextmanager
def held_route(enabled: bool | None = None):
    """Scope ONE teacher rollout: the first route build's lane chain is held for every later build.

    Yields the hold state dict (or None when holding is disabled). After the scope, `LAST_HOLD`
    holds the final state: `chain_ids` (the step-0 lane ids, in order in `rec`), `builds`, and
    `hits` / `misses` (later builds that found / did not find a recorded lane on a roadblock).
    """
    global _HOLD, LAST_HOLD
    if enabled is None:
        enabled = hold_enabled()
    if not enabled:
        LAST_HOLD = None
        yield None
        return
    prev = _HOLD
    _HOLD = _new_hold()
    try:
        yield _HOLD
    finally:
        h = _HOLD
        if h["recording"]:                      # a one-build scope: its chain is what it recorded
            h["recording"] = False
            h["chain_ids"] = tuple(h["rec"])
        LAST_HOLD = h
        _HOLD = prev


def _choose(original, M, edges, anchor, prev_edge):
    """The un-held choice: rank-k at the first roadblock of a build, THEIR rule after it."""
    if prev_edge is not None:
        return original(edges, anchor, prev_edge)      # their rule, untouched
    cands = [e for e in edges if M._edge_centerline_coords(e)]
    if not cands:
        return None
    ordered = sorted(cands, key=lambda e: M._route_edge_anchor_score(e, anchor))
    return ordered[min(_RANK, len(ordered) - 1)]


def make_patched(original, M):
    """The wrapper, built over `original` and the module `M` (exposed so tests can build it on a fake M)."""

    def patched(edges, anchor, prev_edge):
        if _RANK == 0:
            return original(edges, anchor, prev_edge)      # delegate -- see set_rank(); no hold at rank 0
        h = _HOLD
        if h is not None and prev_edge is None:            # a NEW route build starts at its first roadblock
            h["builds"] += 1
            if h["builds"] == 1:
                h["recording"] = True
                h["rank"] = _RANK
            elif h["recording"]:
                h["recording"] = False
                h["chain_ids"] = tuple(h["rec"])
        if h is not None and h["chain_ids"] is not None:
            held = set(h["chain_ids"])
            for e in edges:
                if str(getattr(e, "id", "")) in held and M._edge_centerline_coords(e):
                    h["hits"] += 1
                    return e
            h["misses"] += 1                               # a roadblock the step-0 route never reached
        chosen = _choose(original, M, edges, anchor, prev_edge)
        if h is not None and h["recording"] and chosen is not None:
            h["rec"].append(str(getattr(chosen, "id", "")))
        return chosen

    patched.__wrapped__ = original
    patched.__route_hold__ = True                          # tests assert the held wrapper is the live one
    return patched


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
    M._choose_route_edge = make_patched(M._choose_route_edge, M)
    _INSTALLED = True
    print(f"[route_lane_rank_patch] ACTIVE rank={rank} route_hold={'ON' if hold_enabled() else 'OFF (reference arm)'}"
          f" -- route_0 semantics are NOT in effect", flush=True)
    return True
