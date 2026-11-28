"""Stage 1 -- goal augmentation for REFe, built on THEIR route construction, not a re-derivation.

WHY THIS IS THE LOAD-BEARING STAGE (PUBLISHED, DriveZero Table 7 supervision ablation):
    human trajectories           93.92
    DriveRL teacher              93.61   <- the teacher ALONE LOSES to plain imitation
    teacher + goal augmentation  94.41   <- +0.80, and the only reason to use an RL teacher
So goal augmentation is not one element of REFe; it is the element that decides whether the teacher
is worth having at all.

HOW THEIR ROUTE IS BUILT (read from source, nuplan-devkit/.../driverl_runtime_map_features.py):
  _sample_route_points_from_map(map_api, route_roadblock_ids, anchor)
    -> _route_start_index      : start at the roadblock nearest the ego
    -> _choose_route_edge      : per roadblock pick ONE interior edge (lane); first by proximity to
                                 the ego, thereafter by connectivity to the previous edge then by
                                 _route_edge_continuity_score
    -> _stitch_route_edge_coords -> anchor frame -> _fit_route_polyline(..., ROUTE_POINTS=100)

THE AUGMENTATION SEAM is therefore exactly `_choose_route_edge`: their rule takes the nearest lane at
the first roadblock. Taking the k-th nearest instead yields a LATERAL alternative -- a different lane
-- and their own continuity rule then carries it forward unchanged. This is Table A3's lane semantics
and the `changing_lane_to_left` case, and it costs no new geometry code.

⭐ CONTROL, by construction and asserted at runtime: route_0 calls THEIR function verbatim, so it
must reproduce the un-augmented polyline EXACTLY. `verify()` asserts that and refuses otherwise --
a discriminating control that must read a known value, not a plausible one.

Usage: python augment_routes.py [--k 4] [<simulation_log.msgpack.xz>]
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np

from nuplan.planning.script import driverl_runtime_map_features as M
from nuplan.planning.simulation.simulation_log import SimulationLog


def _anchor_from_ego(ego) -> M.FrameRow:
    p = ego.rear_axle
    v = ego.dynamic_car_state.rear_axle_velocity_2d
    return M.FrameRow(token=b"", token_hex="", timestamp_us=0, scene_token=None,
                      x=float(p.x), y=float(p.y), yaw=float(p.heading),
                      vx=float(v.x), vy=float(v.y), ax=0.0, ay=0.0, yaw_rate=0.0)


def _route_with_lane_rank(map_api, route_roadblock_ids, anchor, rank: int):
    """Their pipeline verbatim, except the FIRST roadblock takes its rank-th nearest lane."""
    roadblocks = [rb for rid in route_roadblock_ids
                  if (rb := M._route_roadblock_for_id(map_api, rid)) is not None]
    if not roadblocks:
        return None, 0
    edges, prev, used_rank = [], None, 0
    for i, rb in enumerate(roadblocks[M._route_start_index(roadblocks, anchor):]):
        cands = [e for e in (getattr(rb, "interior_edges", []) or []) if M._edge_centerline_coords(e)]
        if not cands:
            continue
        if prev is None:
            ordered = sorted(cands, key=lambda e: M._route_edge_anchor_score(e, anchor))
            used_rank = min(rank, len(ordered) - 1)
            edge = ordered[used_rank]
        else:
            edge = M._choose_route_edge(cands, anchor, prev)   # THEIR rule, unchanged
        if edge is None:
            continue
        edges.append(edge); prev = edge
    coords = M._stitch_route_edge_coords(edges)
    if not coords:
        return None, used_rank
    local = np.asarray([M._global_to_anchor_xy(x, y, anchor) for x, y in coords], dtype=np.float32)
    return M._fit_route_polyline(local, M.ROUTE_POINTS), used_rank


def augment_routes(map_api, route_roadblock_ids, ego, k: int = 4):
    """route_0 == theirs; route_1..k-1 == lateral alternatives. Duplicates are dropped."""
    anchor = _anchor_from_ego(ego)
    base = M._sample_route_points_from_map(map_api=map_api,
                                           route_roadblock_ids=route_roadblock_ids, anchor=anchor)
    out = [("route_0_logged", base, 0)]
    seen = [base]
    for r in range(1, k):
        poly, used = _route_with_lane_rank(map_api, route_roadblock_ids, anchor, r)
        if poly is None or used != r:
            continue                                   # fewer lanes than requested rank
        if any(np.allclose(poly, s, atol=1e-4) for s in seen):
            continue                                   # identical to one we already have
        seen.append(poly); out.append((f"route_{r}_lane{r}", poly, r))
    return out, anchor


def verify(map_api, route_roadblock_ids, ego):
    """The control: our rank-0 path must equal THEIR function exactly."""
    anchor = _anchor_from_ego(ego)
    theirs = M._sample_route_points_from_map(map_api=map_api,
                                             route_roadblock_ids=route_roadblock_ids, anchor=anchor)
    ours, _ = _route_with_lane_rank(map_api, route_roadblock_ids, anchor, 0)
    if theirs is None or ours is None:
        return False, "one side produced no polyline -- INCONCLUSIVE, not a pass"
    if theirs.shape != ours.shape:
        return False, f"shape {theirs.shape} vs {ours.shape}"
    d = float(np.abs(theirs - ours).max())
    return d <= 1e-5, f"max|theirs-ours| = {d:.3e}"


def main(argv):
    k = 4
    if "--k" in argv:
        k = int(argv[argv.index("--k") + 1]); argv = [a for i, a in enumerate(argv)
                                                      if i not in (argv.index("--k"), argv.index("--k") + 1)]
    lp = argv[0] if argv else glob.glob(r"C:\dzo\m-nr-n\**\*.msgpack.xz", recursive=True)[0]
    log = SimulationLog.load_data(file_path=Path(lp))
    sc = log.scenario
    ego = log.simulation_history.data[0].ego_state
    rb = sc.get_route_roadblock_ids()
    print(f"scenario {sc.scenario_type}/{sc.scenario_name}  roadblocks={len(rb)}")

    ok, msg = verify(sc.map_api, rb, ego)
    print(f"  CONTROL route_0 == their _sample_route_points_from_map : {'PASS' if ok else 'FAIL'}  ({msg})")
    if not ok:
        return 3

    routes, anchor = augment_routes(sc.map_api, rb, ego, k=k)
    print(f"  generated {len(routes)} distinct routes (requested up to {k})")
    base = routes[0][1]
    for name, poly, rank in routes:
        lat = float(np.abs(poly[:, 1] - base[:, 1]).max())
        end = float(np.hypot(*(poly[-1] - base[-1])))
        print(f"    {name:18s} lane-rank {rank}  max lateral offset vs logged {lat:6.2f} m  "
              f"endpoint shift {end:6.2f} m")
    if len(routes) > 1:
        offs = [float(np.abs(p[:, 1] - base[:, 1]).max()) for _, p, _ in routes[1:]]
        print(f"  lateral offsets {['%.2f' % o for o in offs]} m  "
              f"(a nuPlan lane is ~3.5 m -- an alternative that moves <1 m is NOT a lane change)")
    print("AUGMENT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
