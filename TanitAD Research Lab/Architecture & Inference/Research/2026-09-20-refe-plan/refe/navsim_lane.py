"""NAVSIM-faithful DRIVING-DIRECTION label and a map LANE-KEEPING label for a set of proposals (LANE-1 follow-on, SFT-2).

WHY (LANE-1, raw/2026-10-04-lane-discipline/). The scorer's direction label is DriveRL's wrong-way category (OffRoad.info
3/4, 2.7 % of held-out proposals) -- a different event from NAVSIM's: the scorer's direction head reaches only a within-set
AUC of ~0.60 against NAVSIM's oncoming test on navtest, and the selection rule never uses it. No lane-keeping signal exists
in the labels at all (the solid-line category fires 0 times in 200,768 held-out proposals).

WHAT, from NAVSIM v1.1's source (navsim @ 3e8291b, pdm_scorer.py):
  * on-route lanes -- PDM-Closed's `_load_route_dicts`: every `interior_edges` lane of every route ROADBLOCK or
    ROADBLOCK_CONNECTOR (the metric cache's `route_lane_ids`); the route is the CORRECTED one (`route_roadblock_correction`)
  * oncoming step -- the footprint CENTRE lies in no on-route lane polygon (`_calculate_ego_area`, EgoAreaIndex.ONCOMING)
  * DDC -- centre progress per 0.1 s step, zeroed on non-oncoming steps, summed over the trailing window
    [t - 10, t] (driving_direction_horizon 1.0 s / 0.1 s), max over t: < 2 m -> 1.0, < 6 m -> 0.5, else 0.0
  * states -- NAVSIM's own simulation of the proposal (navsim_dac.simulated_states: LQR + kinematic bicycle, 41 x 0.1 s)
LANE KEEPING (not a NAVSIM v1 metric; LANE-1's lk10): the centre's distance to the centreline of the lane polygon that
contains it, on steps outside every intersection; violation = > 1.0 m for >= 1.0 s (10 consecutive steps).
Labels may use the map (labels may use privileged signals); the map is never a model input.
    python refe/navsim_lane.py --selftest
"""
from __future__ import annotations

import sys

import numpy as np

DDC_HORIZON_STEPS = 10            # 1.0 s / 0.1 s
DDC_COMPLY_M, DDC_VIOLATE_M = 2.0, 6.0
LK_DEV_M, LK_RUN_STEPS = 1.0, 10


def ddc_from_centres(centres: np.ndarray, on_route: np.ndarray) -> np.ndarray:
    """centres [P, T, 2] (T = 41, t = 0 included), on_route [P, T] bool -> [P] DDC in {1.0, 0.5, 0.0}."""
    prog = np.zeros(on_route.shape, dtype=np.float64)
    prog[:, 1:] = np.linalg.norm(centres[:, 1:] - centres[:, :-1], axis=-1)
    prog[on_route] = 0.0
    T = prog.shape[1]
    win = np.stack([prog[:, max(0, t - DDC_HORIZON_STEPS): t + 1].sum(1) for t in range(T)], 1).max(1)
    return np.where(win < DDC_COMPLY_M, 1.0, np.where(win < DDC_VIOLATE_M, 0.5, 0.0))


def longest_run(mask: np.ndarray) -> np.ndarray:
    """[P, T] bool -> [P] longest run of True"""
    out = np.zeros(mask.shape[0], dtype=np.int64)
    cur = np.zeros(mask.shape[0], dtype=np.int64)
    for t in range(mask.shape[1]):
        cur = np.where(mask[:, t], cur + 1, 0)
        out = np.maximum(out, cur)
    return out


def point_polyline_dist(pts: np.ndarray, line: np.ndarray) -> np.ndarray:
    a, b = line[:-1], line[1:]
    ab = b - a
    ap = pts[:, None, :] - a[None]
    t = np.clip((ap * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9)[None], 0.0, 1.0)
    q = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None, :] - q, axis=-1).min(1)


def lane_keep_from_centres(centres: np.ndarray, lanes: list, junctions: list) -> np.ndarray:
    """centres [P, T, 2]; lanes = [(polygon, centreline [n, 2])]; junctions = [polygon] -> [P] 1.0 = violation (off the
    containing lane's centreline by > LK_DEV_M for >= LK_RUN_STEPS steps on non-junction lane steps, t >= 1)."""
    import shapely
    P, T, _ = centres.shape
    c = centres[:, 1:].reshape(-1, 2)
    d = np.full(len(c), np.inf)
    in_lane = np.zeros(len(c), bool)
    for poly, cl in lanes:
        m = shapely.contains_xy(poly, c[:, 0], c[:, 1])
        if m.any():
            in_lane |= m
            d[m] = np.minimum(d[m], point_polyline_dist(c[m], np.asarray(cl, np.float64)))
    in_j = np.zeros(len(c), bool)
    for poly in junctions:
        in_j |= shapely.contains_xy(poly, c[:, 0], c[:, 1])
    bad = (in_lane & ~in_j & (d > LK_DEV_M)).reshape(P, T - 1)
    return (longest_run(bad) >= LK_RUN_STEPS).astype(np.float64)


def route_lane_polygons(map_api, route_roadblock_ids) -> list:
    """PDM-Closed's `_load_route_dicts`: interior lanes of every route ROADBLOCK / ROADBLOCK_CONNECTOR."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    out = []
    for rid in route_roadblock_ids:
        blk = map_api.get_map_object(str(rid), L.ROADBLOCK) or map_api.get_map_object(str(rid), L.ROADBLOCK_CONNECTOR)
        if blk is not None:
            out += [ln.polygon for ln in blk.interior_edges]
    return out


def corrected_route_ids(map_api, rear_axle, ids) -> tuple:
    """NAVSIM's PDM route correction (vendored verbatim in pdm_route_correction.py) -- the same steps as
    planner.pdm_corrected_route_ids, kept here so the pod labeller does not import the model package."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    import pdm_route_correction as PRC
    rd, n_unres = {}, 0
    for i in ids:
        b = map_api.get_map_object(i, L.ROADBLOCK) or map_api.get_map_object(i, L.ROADBLOCK_CONNECTOR)
        if b is None:
            n_unres += 1
            continue
        rd[b.id] = b
    if not rd:
        return list(ids), {"changed": False, "n_unresolved": n_unres, "empty": True}
    new = list(PRC.route_roadblock_correction(rear_axle, map_api, rd))
    return new, {"changed": new != list(ids), "n_unresolved": n_unres}


def centres_from_states(sim: np.ndarray) -> np.ndarray:
    """NAVSIM simulated states [P, 41, S] -> footprint centres [P, 41, 2] (the scorer's BBCoordsIndex.CENTER)."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parent))
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from navsim_pdm.pdm_array_representation import state_array_to_coords_array
    from navsim_pdm.pdm_enums import BBCoordsIndex
    return state_array_to_coords_array(sim, get_pacifica_parameters())[:, :, int(BBCoordsIndex.CENTER)]


def navsim_ddc_and_lane(sim: np.ndarray, map_api, ego_state, route_roadblock_ids, radius: float = 100.0) -> tuple:
    """ONE simulation (the labeller's `navsim_dac.simulated_states`) -> ([P] NAVSIM DDC, [P] lane-keeping violation)."""
    import shapely
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    cen = centres_from_states(sim)
    rpolys = route_lane_polygons(map_api, route_roadblock_ids)
    flat = cen.reshape(-1, 2)
    on = np.zeros(len(flat), bool)
    for poly in rpolys:
        on |= shapely.contains_xy(poly, flat[:, 0], flat[:, 1])
    ddc = ddc_from_centres(cen, on.reshape(cen.shape[:2]))
    objs = map_api.get_proximal_map_objects(ego_state.center.point, radius, [L.LANE, L.INTERSECTION])
    lanes = [(ln.polygon, np.array([[q.x, q.y] for q in ln.baseline_path.discrete_path])) for ln in objs[L.LANE]]
    lk = lane_keep_from_centres(cen, lanes, [o.polygon for o in objs[L.INTERSECTION]])
    return ddc, lk


def selftest(mutate: str = "") -> int:
    """Analytic targets on a synthetic two-lane road: route lane |y| <= 1.75, oncoming lane 1.75 < y <= 5.25, x in
    [-50, 200]; centreline of each lane at its middle. 41 steps at 0.1 s."""
    import shapely
    route = shapely.box(-50, -1.75, 200, 1.75)
    oncl = shapely.box(-50, 1.75, 200, 5.25)
    t = np.arange(41) * 0.1

    def path(v, y):
        return np.stack([v * t, np.full_like(t, y)], -1)

    cases = {                                    # name: (centres, expected DDC, expected LK)
        "in_lane_10mps": (path(10, 0.0), 1.0, 0.0),
        "oncoming_10mps": (path(10, 3.5), 0.0, 0.0),      # 1.0 s window sums 10 m >= 6 m
        "oncoming_3mps": (path(3, 3.5), 0.5, 0.0),        # 3 m: >= 2, < 6
        "oncoming_1mps": (path(1, 3.5), 1.0, 0.0),        # 1 m < 2
        "stopped_oncoming": (path(0, 3.5), 1.0, 0.0),     # no progress, no violation
        "offset_1p3_in_route": (path(10, 1.3), 1.0, 1.0),   # 1.3 m from the centreline for the whole horizon
        "offset_0p6_in_route": (path(10, 0.6), 1.0, 0.0),
    }
    if mutate == "no_progress_zeroing":
        global ddc_from_centres
        orig = ddc_from_centres

        def ddc_from_centres(centres, on_route):                                    # noqa: F811
            return orig(centres, np.zeros_like(on_route))
    ok = True
    for name, (c, e_ddc, e_lk) in cases.items():
        C = c[None]
        on = shapely.contains_xy(route, C[..., 0], C[..., 1])
        d = float(ddc_from_centres(C, on)[0])
        lk = float(lane_keep_from_centres(C, [(route, np.array([[-50, 0.0], [200, 0.0]])),
                                              (oncl, np.array([[200, 3.5], [-50, 3.5]]))], [])[0])
        good = d == e_ddc and lk == e_lk
        ok &= good
        print(f"  {name:22s} DDC {d:.1f} (exp {e_ddc:.1f})  LK {lk:.0f} (exp {e_lk:.0f})  {'ok' if good else 'FAIL'}")
    print(("ZZNAVSIMLANE_SELFTEST_PASS" if ok else "ZZNAVSIMLANE_SELFTEST_FAIL") + (f" (mutation {mutate})" if mutate else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        rc = selftest()
        rc_m = selftest("no_progress_zeroing")
        print("mutation must FAIL:", "RED as required" if rc_m != 0 else "STILL GREEN -- the test cannot see it")
        sys.exit(rc if rc_m != 0 else 2)
