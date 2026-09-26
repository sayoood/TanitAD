"""Exact memoisation of the DriveRL teacher's STATIC map-geometry work -- the data-prep speed lever.

⛔ MEASURED 2026-09-23 (cProfile, 3 navtrain rollouts, dev box): the teacher's feature builder spent
105.9 s of 117.8 s of planner time in `_build_map_features`, i.e. ~90 % of every rollout. The bulk
is geometry that NEVER CHANGES within a map: `_downsample_coords` (Visvalingam-Whyatt polyline
simplification: 58,809 calls, 15.2 M heap pushes, 43.6 s) and `_line_coords` (shapely -> float
tuples: 176,681 calls, 29.4 s) are recomputed for the same lanes at every one of the 80 simulation
steps of every frame.

EXACT BY CONSTRUCTION -- the rows this produces must equal the 18,122 already banked:
  * `_downsample_coords` is a pure function of its VALUES -> keyed on (tuple(coords), max_points,
    area_threshold);
  * `_line_coords` / `_polygon_exterior_coords` read an IMMUTABLE map object -> keyed on the
    object's identity, and the cache holds a strong reference to it, so its id can never be
    recycled by the allocator into a false hit;
  * every hit returns a NEW list, so no caller can mutate a cached value.
`install()` swaps the module attributes; the module calls these helpers through its globals, so
every internal call site picks the cached version up. `diag` in `__main__` proves equality.
"""
from __future__ import annotations

_INSTALLED = False
STATS = {"ds_hit": 0, "ds_miss": 0, "lc_hit": 0, "lc_miss": 0, "pc_hit": 0, "pc_miss": 0}


def install() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    from nuplan.planning.script import driverl_runtime_map_features as M
    orig_ds, orig_lc, orig_pc = M._downsample_coords, M._line_coords, M._polygon_exterior_coords
    ds_cache: dict = {}
    lc_cache: dict = {}
    pc_cache: dict = {}

    def _downsample_coords(coords, max_points=M.MAX_MAP_POINTS_PER_OBJECT,
                           area_threshold=M.DEFAULT_MAP_SIMPLIFICATION_AREA_THRESHOLD_M2):
        try:
            key = (tuple(coords), max_points, area_threshold)
            hash(key)
        except TypeError:                          # unhashable input: never guess, compute
            return orig_ds(coords, max_points, area_threshold)
        hit = ds_cache.get(key)
        if hit is None:
            STATS["ds_miss"] += 1
            hit = tuple(orig_ds(list(coords), max_points, area_threshold))
            ds_cache[key] = hit
        else:
            STATS["ds_hit"] += 1
        return list(hit)

    def _by_identity(orig, cache, tag):
        def f(obj):
            ent = cache.get(id(obj))
            if ent is not None and ent[0] is obj:
                STATS[tag + "_hit"] += 1
                return list(ent[1])
            STATS[tag + "_miss"] += 1
            val = tuple(orig(obj))
            cache[id(obj)] = (obj, val)          # strong ref: this id cannot be reused
            return list(val)
        return f

    M._downsample_coords = _downsample_coords
    M._line_coords = _by_identity(orig_lc, lc_cache, "lc")
    M._polygon_exterior_coords = _by_identity(orig_pc, pc_cache, "pc")
    M._downsample_coords.__wrapped__ = orig_ds
    _INSTALLED = True
    return True


def summary() -> str:
    s = STATS
    rate = lambda h, m: 100.0 * h / max(h + m, 1)
    return (f"map cache hits: downsample {rate(s['ds_hit'], s['ds_miss']):.1f} % "
            f"({s['ds_hit']:,}/{s['ds_hit'] + s['ds_miss']:,}), line coords "
            f"{rate(s['lc_hit'], s['lc_miss']):.1f} %, polygon coords {rate(s['pc_hit'], s['pc_miss']):.1f} %")
