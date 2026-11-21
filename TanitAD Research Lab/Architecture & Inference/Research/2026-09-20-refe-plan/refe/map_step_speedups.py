"""Exact PER-STEP speed levers for the teacher's runtime map features -- data prep, 2026-09-24.

MEASURED (cProfile, 10 CPU-teacher rollouts at 10 Hz on the dev box, map-feature + query caches ON,
`scratchpad/prof_aug.py`): of 52.8 s of planner time the feature builder took 39.8 s and
`_build_map_features` 33.0 s. What the static-geometry cache (map_feature_cache.py) leaves is
PER-STEP work whose cost is mostly CONTAINERS, not geometry:
  * `NuPlanMap._get_proximity_map_object` 4.96 s -- 3.65 s of it is geopandas building a filtered
    GeoDataFrame (`layer_df[mask]['fid']`) on every call; the `intersects` test itself is 0.60 s;
  * `_drivable_area_boundaries` 6.43 s -- a shapely `unary_union` + per-polygon validity walk
    (3.69 s) that is a PURE FUNCTION of the ordered list of nearby polygons, which at 10 Hz is the
    same list step after step, plus 1.69 s of the same GeoDataFrame selection;
  * `_sample_route_points_from_map` 7.74 s -- 3.29 s rotates every route point into the anchor
    frame one Python call at a time;
  * `_write_segment_candidates` 3.83 s -- the same per-point rotation for up to 1,024 segments;
  * `_edge_distance_to_frames` -- each lane's distance is computed for the lane sort, the
    connector sort AND again for the merged sort of the same step.

EXACT BY CONSTRUCTION -- every replacement computes the same values in the same order:
  * ROW SELECTION: the same boolean mask from the same `intersects` call, turned into positions
    with `np.flatnonzero` (row order, as boolean indexing keeps it) and read from lists built ONCE
    per layer object with `.tolist()` -- the values `Series.__iter__` yields. Layers are cached and
    immutable in the map API; entries hold the layer object, so an id can never be recycled.
  * UNION CACHE: keyed on those row positions per layer object, so a hit is the result of the very
    same geometry list; every hit returns fresh lists (callers may mutate them).
  * ROTATION: the same float64 expression `c*dx + s*dy`, `-s*dx + c*dy`, evaluated elementwise by
    numpy -- one IEEE-rounded op per scalar op, same order, no reassociation, no FMA across ufuncs
    -- then cast to float32 exactly as the per-element float32 assignment cast it.
  * DISTANCE MEMO: keyed on the edge OBJECT (held) and the frames' coordinates, valid for one set
    of frames at a time; it returns the float the original computed.
  * `math.hypot` sites are left untouched: numpy's hypot is not guaranteed to round like CPython's.
`install()` swaps module / class attributes; every internal call site resolves them at call time.
REFE_MAP_STEP_FAST=1 enables it (build_teacher_rollouts.build_planner); `diag_lever_ab.py` must
show byte-identical rows against =0 before the default changes.
"""
from __future__ import annotations

import math
import os

import numpy as np

_MUTATE = os.environ.get("REFE_MAP_STEP_MUTATE", "")      # diag only, see diag_lever_ab.py arms
_INSTALLED = False
STATS = {"prox_calls": 0, "da_hit": 0, "da_miss": 0, "dist_hit": 0, "dist_miss": 0,
         "route_calls": 0, "write_calls": 0}
_UNION_CACHE_MAX = 512


def _rotate_rows(xy: np.ndarray, anchor) -> tuple[np.ndarray, np.ndarray]:
    """`_rotate_to_anchor(x - anchor.x, y - anchor.y, anchor.yaw)` for every row of a float64 [n, 2]."""
    c = math.cos(anchor.yaw)
    s = math.sin(anchor.yaw)
    dx = xy[:, 0] - anchor.x
    dy = xy[:, 1] - anchor.y
    return c * dx + s * dy, -s * dx + c * dy


def install() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    from nuplan.planning.script import driverl_runtime_map_features as M
    from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap

    layer_cols: dict = {}             # id(layer_df) -> (layer_df, geometry series, {col: list})

    def _cols(layer_df, col):
        ent = layer_cols.get(id(layer_df))
        if ent is None or ent[0] is not layer_df:
            ent = (layer_df, layer_df["geometry"], {})
            layer_cols[id(layer_df)] = ent
        lst = ent[2].get(col)
        if lst is None:
            lst = layer_df[col].tolist()
            ent[2][col] = lst
        return ent[1], lst

    # ---- 1. proximal map objects: no filtered GeoDataFrame per call --------------------------
    def _get_proximity_map_object(self, patch, layer):
        layer_df = self._get_vector_map_layer(layer)
        geo, fids = _cols(layer_df, "fid")
        pos = np.flatnonzero(geo.intersects(patch).to_numpy())
        STATS["prox_calls"] += 1
        return [self.get_map_object(fids[i], layer) for i in pos]

    NuPlanMap._get_proximity_map_object = _get_proximity_map_object

    # ---- 2. drivable-area boundaries: same selection, union cached on the selection ----------
    union_cache: dict = {}

    def _drivable_area_boundaries(*, map_api, query_frames, radius_m):
        if (M.Point is None or M.SemanticMapLayer is None or M.unary_union is None
                or not query_frames):
            return []
        get_vector_map_layer = getattr(map_api, "_get_vector_map_layer", None)
        if get_vector_map_layer is None:
            return []
        query_region = M.unary_union(
            [M.Point(frame.x, frame.y).buffer(radius_m) for frame in query_frames])
        geometries: list = []
        key: list = []
        held: list = []
        for layer in (M.SemanticMapLayer.DRIVABLE_AREA, M.SemanticMapLayer.CARPARK_AREA):
            try:
                layer_df = get_vector_map_layer(layer)
            except (KeyError, ValueError):
                continue
            geo, geoms = _cols(layer_df, "geometry")
            pos = np.flatnonzero(geo.intersects(query_region).to_numpy())
            geometries.extend(geoms[i] for i in pos)
            key.append((id(layer_df), pos.tobytes()))
            held.append(layer_df)
        k = tuple(key)
        ent = union_cache.get(k)
        if ent is not None and all(a is b for a, b in zip(ent[0], held)):
            STATS["da_hit"] += 1
            return [list(c) for c in ent[1]]
        STATS["da_miss"] += 1
        val = M._merged_drivable_area_boundaries(geometries)
        if len(union_cache) >= _UNION_CACHE_MAX:
            union_cache.pop(next(iter(union_cache)))
        union_cache[k] = (held, tuple(tuple(c) for c in val))
        return [list(c) for c in val]

    M._drivable_area_boundaries = _drivable_area_boundaries

    # ---- 3. route points: the anchor rotation vectorised -------------------------------------
    def _sample_route_points_from_map(*, map_api, route_roadblock_ids, anchor):
        if map_api is None or M.SemanticMapLayer is None or not route_roadblock_ids:
            return None
        roadblocks = [
            roadblock
            for roadblock_id in route_roadblock_ids
            if (roadblock := M._route_roadblock_for_id(map_api, roadblock_id)) is not None
        ]
        if not roadblocks:
            return None
        route_edges: list = []
        prev_edge = None
        for roadblock in roadblocks[M._route_start_index(roadblocks, anchor):]:
            edge = M._choose_route_edge(
                list(getattr(roadblock, "interior_edges", []) or []), anchor, prev_edge)
            if edge is None:
                continue
            route_edges.append(edge)
            prev_edge = edge
        coords = M._stitch_route_edge_coords(route_edges)
        if not coords:
            return None
        lx, ly = _rotate_rows(np.asarray(coords, dtype=np.float64), anchor)
        if _MUTATE == "route_1mm":            # diag only: the A/B must be able to go RED
            lx = lx + 1e-3
        local = np.empty((lx.shape[0], 2), dtype=np.float32)
        local[:, 0] = lx
        local[:, 1] = ly
        STATS["route_calls"] += 1
        return M._fit_route_polyline(local, M.ROUTE_POINTS)

    orig_route = M._sample_route_points_from_map
    M._sample_route_points_from_map = _sample_route_points_from_map
    # ⛔ THE FEATURE BUILDER IMPORTS THIS FUNCTION BY NAME (`from ... import
    # _sample_route_points_from_map`), so patching the module attribute alone never reaches its
    # call site. MEASURED on the first A/B: `route 0` calls in a rank-0 rollout while the builder
    # samples the route on every step -- the first mutation arm aimed here and proved nothing. The
    # builder's binding is replaced too, but only if it still holds the ORIGINAL function.
    import driverl.nuplan.feature_builder as FB
    if getattr(FB, "_sample_route_points_from_map", None) is orig_route:
        FB._sample_route_points_from_map = _sample_route_points_from_map

    # ---- 4. segment writer: the anchor rotation and the slot writes vectorised ----------------
    def _write_segment_candidates(*, candidates, anchor, points, masks, attributes, groups, ids,
                                  max_segments):
        selected = sorted(candidates, key=lambda item: (item.distance_m, item.order))[
            :max_segments]
        selected.sort(key=lambda item: item.order)
        n = len(selected)
        STATS["write_calls"] += 1
        if n == 0:
            return selected
        p = np.asarray([(c.p0[0], c.p0[1], c.p1[0], c.p1[1]) for c in selected],
                       dtype=np.float64)
        x0, y0 = _rotate_rows(p[:, 0:2], anchor)
        x1, y1 = _rotate_rows(p[:, 2:4], anchor)
        if _MUTATE == "segments_1mm":          # diag only -- this writer runs on EVERY step, unlike
            x0 = x0 + 1e-3                     # the route sampler (0 calls in a rank-0 rollout: the
                                               # first mutation arm aimed there and proved nothing)
        points[:n, 0, 0] = x0
        points[:n, 0, 1] = y0
        points[:n, 1, 0] = x1
        points[:n, 1, 1] = y1
        masks[:n] = True
        attributes[:n] = [(c.lane_type, c.speed_mps, 0.0) for c in selected]
        groups[:n] = [np.uint32(c.group_id) if c.group_id >= 0 else M.INVALID_GROUP_ID
                      for c in selected]
        ids[:n] = [M._stable_int64_id(c.raw_id) for c in selected]
        return selected

    M._write_segment_candidates = _write_segment_candidates

    # ---- 5. per-step edge-distance memo ----------------------------------------------------
    orig_dist = M._edge_distance_to_frames
    memo = {"fk": None, "d": {}}

    def _edge_distance_to_frames(edge, frames):
        fk = tuple((f.x, f.y) for f in frames) if frames else ()
        if fk != memo["fk"]:
            memo["fk"] = fk
            memo["d"] = {}
        ent = memo["d"].get(id(edge))
        if ent is not None and ent[0] is edge:
            STATS["dist_hit"] += 1
            return ent[1]
        STATS["dist_miss"] += 1
        v = orig_dist(edge, frames)
        memo["d"][id(edge)] = (edge, v)
        return v

    M._edge_distance_to_frames = _edge_distance_to_frames
    _INSTALLED = True
    return True


def summary() -> str:
    return (f"map-step fast path: proximal {STATS['prox_calls']} calls | drivable-area union "
            f"{STATS['da_hit']} hit / {STATS['da_miss']} miss | edge distance "
            f"{STATS['dist_hit']} hit / {STATS['dist_miss']} miss | route {STATS['route_calls']} | "
            f"writes {STATS['write_calls']}")
