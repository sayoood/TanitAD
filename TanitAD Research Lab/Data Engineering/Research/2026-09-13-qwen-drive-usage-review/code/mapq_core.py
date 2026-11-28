"""Map / occupancy quality WITHOUT map ground truth: agreement with independent LiDAR and label evidence.

Every metric below is computed identically on (a) Qwen-Drive's nuPlan demo frames, where the TRUE map and
occupancy exist -- that calibrates what a correct raster scores -- and (b) our PhysicalAI frames.

Raster conventions (verified 2026-09-13 against the demo's own GT map + LiDAR; the wrong mirrorings put
11-25 % of static obstacles on driveable, the right one 0.2 %):
  map  [i, j]    : y = -15 + (i + 0.5) * 0.15 , x = -30 + (j + 0.5) * 0.15      (ego frame, 200 x 400)
  occ  [ix,iy,iz]: x = -50 + (ix + 0.5) * 0.5 , y = -50 + (iy + 0.5) * 0.5       (ego frame, 200 x 200 x 16)
Classes: map 0 background, 1 driveable_surface, 2 road_line, 3 road_edge, 4 crosswalk, 5 walkway.
         occ 0-6 objects, 7 driveable, 8 background, 9 empty.

Metrics (all fractions of INDEPENDENT evidence, so a sparse sweep changes n, not the scale):
  MAP_A  static-obstacle points on predicted driveable_surface ............ lower is better
  MAP_C  vehicle GT-box centres on predicted on-road {1, 2, 4} ............. higher is better
  MAP_E  predicted road_edge cells within 0.6 m of a LiDAR height step or a static obstacle, among
         road_edge cells inside LiDAR coverage ................................ higher is better
  MAP_B  (ours only) ego's own driven path (+-30 m) on predicted on-road ..... higher is better
  MAP_D  (ours only) high-intensity ground returns (paint) within 0.45 m of predicted road_line or
         crosswalk ................................................................ higher is better
  OCC_A  static-obstacle points whose 0.5 m column is predicted obstacle (occ 0-6, 8) ... higher
  OCC_B  points inside agent boxes whose column is predicted an object class (0-6) ...... higher
  OCC_C  predicted obstacle columns (<= 30 m) holding >= 1 LiDAR point above ground + 0.3 m ... higher
Only evidence within 30 m (x in the map window) is used, where a single sweep is dense.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import ndimage

ON_ROAD = (1, 2, 4)
OBJ = (0, 1, 2, 3, 4, 5, 6)
OBST = (0, 1, 2, 3, 4, 5, 6, 8)
R_MAX = 30.0


def split_points(pts_ego: np.ndarray, boxes_ego: np.ndarray):
    """-> (static_obstacle, ground, agent) point sets, ego frame. Local ground = 2 m cell minimum."""
    p = pts_ego[np.hypot(pts_ego[:, 0], pts_ego[:, 1]) <= R_MAX + 2.0]      # extra columns (intensity) ride along
    # local ground = 2 m cell minimum, but ONLY where the cell holds >= 20 returns; a sparse cell (a 32-beam
    # sweep, MEASURED on a nuScenes demo frame: 9,870 of 13,030 "obstacles" were ground) falls back to 6 m
    key = (np.floor((p[:, 0] + 64) / 2).astype(int) * 64 + np.floor((p[:, 1] + 64) / 2).astype(int))
    zmin = np.full(64 * 64 + 64, np.inf); np.minimum.at(zmin, key, p[:, 2])
    cnt = np.bincount(key, minlength=len(zmin))
    key6 = (np.floor((p[:, 0] + 66) / 6).astype(int) * 23 + np.floor((p[:, 1] + 66) / 6).astype(int))
    zmin6 = np.full(23 * 23 + 23, np.inf); np.minimum.at(zmin6, key6, p[:, 2])
    g = np.where(cnt[key] >= 20, zmin[key], zmin6[key6])
    inbox = np.zeros(len(p), bool); inbox_tight = np.zeros(len(p), bool)
    for b in boxes_ego:
        c, s = math.cos(-b[6]), math.sin(-b[6]); dx, dy = p[:, 0] - b[0], p[:, 1] - b[1]
        lx, ly = dx * c - dy * s, dx * s + dy * c
        inbox |= (np.abs(lx) <= b[3] / 2 + 0.5) & (np.abs(ly) <= b[4] / 2 + 0.5)
        inbox_tight |= ((np.abs(lx) <= b[3] / 2) & (np.abs(ly) <= b[4] / 2)
                        & (p[:, 2] >= b[2] + 0.2) & (p[:, 2] <= b[2] + b[5]))
    static = p[(~inbox) & (p[:, 2] > g + 0.5) & (p[:, 2] < g + 3.0)]
    # TALL static obstacles, ego footprint excluded: removes unlabelled cones/barriers/bollards (PhysicalAI's box
    # enum has none; nuPlan's does) and any self-returns, so both corpora meet the same evidence class
    ego = (p[:, 0] > -2.0) & (p[:, 0] < 5.0) & (np.abs(p[:, 1]) < 1.5)
    tall = p[(~inbox) & (~ego) & (p[:, 2] > g + 1.2) & (p[:, 2] < g + 3.0)]
    ground = p[(~inbox) & (p[:, 2] < g + 0.25)]
    agent = p[inbox_tight]
    above = p[(p[:, 2] > g + 0.3) & (p[:, 2] < g + 3.0)]
    return static, ground, agent, above, tall


def map_cls(m: np.ndarray, xy: np.ndarray):
    i = np.floor((xy[:, 1] + 15.0) / 0.15).astype(int); j = np.floor((xy[:, 0] + 30.0) / 0.15).astype(int)
    k = (i >= 0) & (i < 200) & (j >= 0) & (j < 400)
    return m[i[k], j[k]], k


def occ_cols(occ: np.ndarray):
    o = occ.reshape(200, 200, 16)
    return np.isin(o, OBST).any(axis=2), np.isin(o, OBJ).any(axis=2)


def col_lookup(mask: np.ndarray, xy: np.ndarray):
    ix = np.floor((xy[:, 0] + 50.0) / 0.5).astype(int); iy = np.floor((xy[:, 1] + 50.0) / 0.5).astype(int)
    k = (ix >= 0) & (ix < 200) & (iy >= 0) & (iy < 200)
    return mask[ix[k], iy[k]], k


def edge_evidence(ground: np.ndarray, static: np.ndarray):
    """0.3 m grid over the map window: a ground height step 0.08-0.35 m, or a static obstacle, dilated 0.6 m."""
    H, W = 100, 200                                      # y 30 m / 0.3, x 60 m / 0.3
    gi = np.floor((ground[:, 1] + 15) / 0.3).astype(int); gj = np.floor((ground[:, 0] + 30) / 0.3).astype(int)
    k = (gi >= 0) & (gi < H) & (gj >= 0) & (gj < W)
    tmp = np.full(H * W, np.inf); np.minimum.at(tmp, gi[k] * W + gj[k], ground[k, 2]); tmp[np.isinf(tmp)] = np.nan
    zg = tmp.reshape(H, W)
    step = np.zeros((H, W), bool)
    with np.errstate(invalid="ignore"):
        for a, b, sl_a, sl_b in ((zg[:, :-1], zg[:, 1:], (slice(None), slice(0, -1)), (slice(None), slice(1, None))),
                                 (zg[:-1, :], zg[1:, :], (slice(0, -1), slice(None)), (slice(1, None), slice(None)))):
            d = np.abs(a - b); hit = (d >= 0.08) & (d <= 0.35)      # NaN compares False: no wrap, no fill
            step[sl_a] |= hit; step[sl_b] |= hit
    obst = np.zeros((H, W), bool)
    si = np.floor((static[:, 1] + 15) / 0.3).astype(int); sj = np.floor((static[:, 0] + 30) / 0.3).astype(int)
    k2 = (si >= 0) & (si < H) & (sj >= 0) & (sj < W); obst[si[k2], sj[k2]] = True
    ev = ndimage.binary_dilation(step | obst, iterations=2)
    covered = ndimage.binary_dilation(~np.isnan(zg) | obst, iterations=2)
    return ev, covered


def up03(mask_03: np.ndarray):
    """0.3 m (100 x 200) mask -> map raster resolution 0.15 m (200 x 400)."""
    return np.repeat(np.repeat(mask_03, 2, axis=0), 2, axis=1)


def frame_metrics(pred_map, pred_occ, pts_ego, boxes_ego, vehicle_boxes_ego, ego_path_xy=None, paint_xy=None,
                  split=None, edge=None):
    """`split` / `edge` may be precomputed once per frame and reused across arms (they depend on evidence only)."""
    static, ground, agent, above, tall = split if split is not None else split_points(pts_ego, boxes_ego)
    out = {}
    c, k = map_cls(pred_map, static[:, :2]); out["MAP_A"] = (float((c == 1).sum()), int(len(c)))
    c, k = map_cls(pred_map, tall[:, :2]); out["MAP_A2"] = (float((c == 1).sum()), int(len(c)))
    if len(vehicle_boxes_ego):
        c, k = map_cls(pred_map, vehicle_boxes_ego[:, :2]); out["MAP_C"] = (float(np.isin(c, ON_ROAD).sum()), int(len(c)))
    ev, cov = edge if edge is not None else edge_evidence(ground, static)
    edge_cells = pred_map == 3
    inside = edge_cells & up03(cov)
    out["MAP_E"] = (float((inside & up03(ev)).sum()), int(inside.sum()))
    if ego_path_xy is not None and len(ego_path_xy):
        c, k = map_cls(pred_map, ego_path_xy); out["MAP_B"] = (float(np.isin(c, ON_ROAD).sum()), int(len(c)))
    if paint_xy is not None and len(paint_xy):
        painted = ndimage.binary_dilation(np.isin(pred_map, (2, 4)), iterations=3)
        c, k = map_cls(painted.astype(int), paint_xy); out["MAP_D"] = (float(c.sum()), int(len(c)))
    if pred_occ is not None:
        obst_col, obj_col = occ_cols(pred_occ)
        v, k = col_lookup(obst_col, static[:, :2]); out["OCC_A"] = (float(v.sum()), int(len(v)))
        v, k = col_lookup(obst_col, tall[:, :2]); out["OCC_A2"] = (float(v.sum()), int(len(v)))
        if len(agent):
            v, k = col_lookup(obj_col, agent[:, :2]); out["OCC_B"] = (float(v.sum()), int(len(v)))
        sup = np.zeros((200, 200), bool)
        ai = np.floor((above[:, 0] + 50) / 0.5).astype(int); aj = np.floor((above[:, 1] + 50) / 0.5).astype(int)
        kk = (ai >= 0) & (ai < 200) & (aj >= 0) & (aj < 200); sup[ai[kk], aj[kk]] = True
        xx = (np.arange(200) + 0.5) * 0.5 - 50.0
        near = np.hypot(xx[:, None], xx[None, :]) <= R_MAX
        pc = obst_col & near
        out["OCC_C"] = (float((pc & sup).sum()), int(pc.sum()))
    return out


def aggregate(rows):
    tot = {}
    for r in rows:
        for k, (a, n) in r.items():
            t = tot.setdefault(k, [0.0, 0]); t[0] += a; t[1] += n
    return {k: {"value": round(a / n, 4) if n else None, "n": n} for k, (a, n) in tot.items()}
