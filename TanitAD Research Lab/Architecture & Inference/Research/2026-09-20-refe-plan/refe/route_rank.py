"""Make `--rank` SELECT the route, instead of stamping a label on whatever route happened to run.

⛔⛔ THE DEFECT THIS CLOSES, AND IT HAS NOW HAPPENED TWICE IN ONE NIGHT.
The rank-k goal augmentation is produced by `code/route_lane_rank_patch.py`, which wraps
`driverl_runtime_map_features._choose_route_edge` to prefer the rank-k lane at the route's first
roadblock. It is switched on by `sitecustomize.py`, which calls `install()` at interpreter start and
reads the env var `DRIVERL_EVAL_ROUTE_LANE_RANK`. So:

  * R6 (closed-loop harvest): `--rank 1` built from the RANK-0 run directory -> a relabelled copy.
    The fix was "point rank 1 at `m-nr-r1`", which is correct FOR A HARVESTER, because the harvested
    rollouts already carry their route.
  * R11 (per-frame harness, 2026-09-21): the new builder ROLLS THE TEACHER ITSELF, so the route is
    chosen IN THIS PROCESS -- and this process never set the env var. `--rank 1 --run m-nr-r1`
    therefore rolled the rank-0 route on the rank-1 logs. MEASURED: **1,101 / 1,101** rank-1
    trajectories and goals byte-identical to rank 0. The consistency gate's step-0 goal control
    (G1) and the rank-distinctness gate both went RED, which is the only reason it did not become
    training data.

⭐ The general lesson: a fix that locates a parameter in ONE construction ("rank lives in the run
directory") is silently wrong in the NEXT construction, where it lives somewhere else ("rank lives
in the route patch"). This module makes the rank an EXPLICIT, ASSERTED property of the process
instead of a side effect of the environment.

Two silent mismatches are REFUSED, not tolerated:
  1. the env var names a DIFFERENT rank than the one requested;
  2. rank 0 was requested but a rank-k patch is already installed (sitecustomize installed it from a
     stale env var) -- rank 0 must run their UNPATCHED code, byte-identical to Stage 0.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_CODE = str(Path(__file__).resolve().parent.parent / "code")
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)


def apply(rank: int) -> str:
    """Install (or verify the absence of) the route patch for `rank`; return a one-line state.

    Raises SystemExit on either silent mismatch -- a run on the wrong route produces a bank that
    passes every internal-consistency check and is wrong.
    """
    import route_lane_rank_patch as P
    env = os.environ.get("DRIVERL_EVAL_ROUTE_LANE_RANK", "").strip()
    if env and int(env) != int(rank):
        raise SystemExit(f"route_rank: env DRIVERL_EVAL_ROUTE_LANE_RANK={env} but --rank {rank}. "
                         f"Refusing: the process would roll one route and label it another.")
    if int(rank) == 0:
        if getattr(P, "_INSTALLED", False):
            raise SystemExit("route_rank: --rank 0 requested but a route patch is ALREADY installed "
                             "(sitecustomize read a stale env var). Rank 0 must run THEIR unpatched "
                             "code. Unset DRIVERL_EVAL_ROUTE_LANE_RANK and rerun.")
        return "route rank 0: THEIR unpatched route (byte-identical to Stage 0 by construction)"
    ok = P.install(int(rank))
    if not ok or not getattr(P, "_INSTALLED", False):
        raise SystemExit(f"route_rank: failed to install the rank-{rank} route patch -- refusing to "
                         f"roll the rank-0 route and label it rank {rank}.")
    from nuplan.planning.script import driverl_runtime_map_features as M
    if not hasattr(M._choose_route_edge, "__wrapped__"):
        raise SystemExit("route_rank: the patch reports installed but `_choose_route_edge` is not "
                         "wrapped -- a success flag disconnected from the thing it describes.")
    return f"route rank {rank}: patch ACTIVE and verified on `_choose_route_edge`"
