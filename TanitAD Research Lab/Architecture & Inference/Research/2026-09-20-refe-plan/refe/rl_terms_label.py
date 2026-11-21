#!/usr/bin/env python3
"""RL-S1 part 2: the `rl_terms` labeller -- the per-proposal inputs rl_reward.py needs and the PDM labels do not carry.

    D  drivable-area excess   per-step gap g(t) = max over the 4 bbox corners of the distance to the union of the DAC polygons (0 inside)
    K  lane consistency       offset / heading error to the PDM-Closed centreline, the scorer's INTERSECTION switch, straddle fraction
    L  lane change            lc_executed [M], label_fires (scene), dither [M]            (review 3.3; PI D3: R2 arm IN)

Design: Research/2026-10-11-refe-data-and-rl/RL_POSTTRAIN_REVIEW.md sections 3.2 (b, c, d), 3.3, 4.  The REDUCTIONS (g -> D, traces -> K, L itself)
are NOT re-implemented here: they are rl_reward.drivable_excess / lane_consistency, called on the traces this module produces.

EVALUATED ON THE EXECUTED STATES (review principle P5).  The traces are read off the 41-step LQR-tracked states -- the same ones
PDMScorer._states holds -- NOT off the 8-pose plan.  `executed_states` runs the front half of score_plans (NAVSIM's own PDMSimulator); the identity run (eval/rl_terms_identity.py) re-asserts bit equality with the real scorer's `_states` on the scenes it uses.

WHICH POLYGONS (the identity is against OUR harness, NAVSIM v1.1 @ 3e8291b, not the v2 devkit the review quotes):
    pdm_scorer.py `_calculate_ego_area`: a step is off the drivable area when FEWER THAN 4 corners lie in at least ONE polygon of type
    ROADBLOCK / INTERSECTION / DRIVABLE_AREA / CARPARK_AREA.  v1.1's `PDMDrivableMap.from_simulation` adds the ROADBLOCK polygons and the INTERSECTION
    and CARPARK_AREA ones; a ROADBLOCK_CONNECTOR's own polygon is NOT added (only its interior lanes, typed LANE_CONNECTOR, which the DAC set excludes).
    `DAC_LAYERS` therefore selects by type name from the scene's PDMDrivableMap, so the labeller follows the scorer, not the review's prose.

INTERPRETATIONS the review leaves open (flagged, each is a field in the record so it can be undone without re-labelling):
  1. K's straddle k3 = mean_t [bbox crosses a lane boundary (the scorer's MULTIPLE_LANES rule) AND not in an intersection], and 0 when a lane-change
     label fires.  The review's guard list ("suppress in intersections and when a lane-change label fires") is applied to ALL of K: when label_fires,
     K is computed with the INTERSECTION mask OR-ed with the scene flag (k1 = k2 = 0), because rl_reward.lane_consistency has no label argument.
     The unsuppressed value is stored as `k_raw`.
  2. dither: signed lateral offset to the PDM-Closed centreline; v_lat over 0.5 s; |v| < 0.15 m/s is ignored; >= 2 sign changes in the 4 s => 1.
  3. lc_executed: label (i)'s rule applied to the proposal's EXECUTED rear-axle positions at 0.5 s steps: a pose in another LANE of the start lane's
     roadblock AND the final pose > 2.0 m from the start lane's baseline path.
  4. label (iii) "route needs it": the start lane's outgoing edges (lane connectors, then the lanes they feed) reach neither of the next two route
     elements, while an adjacent lane's do.  Definition only in the review [U]; its measured rate is in RESULT_RL_TERMS.md.

Files: <out>/rl_terms_<set file name> (the glob scorer_finetune's EPO loader reads, `--epo-terms-train/-heldout <out>`); the record carries every field that
loader reads (d, k, lc_executed, label_fires, dither) with exactly its types, plus extras it ignores.
Record (one per (log, token, step, rank, ckpt_step), same key as the pdm_targets side files):
    {"kind": "rl_terms", "key": [log, token, step, rank], "ckpt_step": c, "label_version": v, "plans": "executed (A7 repair)",
     "d": [M], "gmax": [M], "k": [M], "k_raw": [M], "lc_executed": [M] (0/1), "label_fires": bool, "dither": [M] (0/1), "labels": {...}}
`gmax` (max over t of g) lets G-REWARD-ID be re-run on the pod from side files alone: join to pdm_targets `dac` on the key.

    python refe/rl_terms_label.py --sets <dir> --out <dir> [--workers 16] [--served-only] [--limit N] [--shard I/N] [--terms DKL]
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import warnings
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))
import rl_reward as RW                                                         # noqa: E402  -- the reductions live there, once

# nuPlan Pacifica (get_pacifica_parameters(): front 4.049, rear 1.127, width 2.297); the identity run asserts these against nuPlan itself.
PACIFICA = {"length": 5.176, "width": 2.297, "rear_axle_to_center": 1.461}
EXECUTED_STEPS = 41                      # PDMScorer's grid: t = 0 .. 4 s at 0.1 s, INCLUDING t = 0
DAC_LAYERS = ("ROADBLOCK", "INTERSECTION", "DRIVABLE_AREA", "CARPARK_AREA")   # pdm_scorer.py:_calculate_ego_area (v1.1)
LANE_LAYERS = ("LANE", "LANE_CONNECTOR")
NO_MAP_GAP = 1.0e3                       # no drivable polygon in range: every corner is infinitely far (D saturates at 1)
LC_OFFSET_M = 2.0                        # review 3.3 label (i)
DITHER_V_DEAD = 0.15                     # m/s
DITHER_MIN_CHANGES = 2
LC_POSE_STRIDE = 5                       # 0.5 s poses out of the 0.1 s executed states
RL_TERMS_PREFIX = "rl_terms_"            # scorer_finetune.RL_TERMS_GLOB = "rl_terms_*.jsonl" (ac4d3b8): the consumer globs this prefix


# ------------------------------------------------------------------ pure geometry (numpy + shapely)
def _sh():
    import shapely
    return shapely


def bbox_coords(states, params: dict = PACIFICA) -> dict:
    """states [..., T, >=3] = (x, y, heading) of the REAR AXLE -> {'centers': [..., T, 2], 'corners': [..., T, 4, 2]} (navsim state_array_to_coords_array)."""
    s = np.asarray(states, dtype=np.float64)
    if s.ndim < 2 or s.shape[-1] < 3:
        raise ValueError(f"states need shape [..., T, >=3], got {s.shape}")
    h = s[..., 2]
    ch, sh = np.cos(h), np.sin(h)
    centers = s[..., :2] + params["rear_axle_to_center"] * np.stack([ch, sh], axis=-1)
    hl, hw = params["length"] / 2.0, params["width"] / 2.0
    corners = np.empty(centers.shape[:-1] + (4, 2))
    for i, (lo, la) in enumerate(((hl, hw), (hl, -hw), (-hl, hw), (-hl, -hw))):
        corners[..., i, 0] = centers[..., 0] + lo * ch - la * sh
        corners[..., i, 1] = centers[..., 1] + lo * sh + la * ch
    return {"centers": centers, "corners": corners}


def point_gap(points, polys) -> np.ndarray:
    """distance of each point [..., 2] to the UNION of `polys` (0 inside or on the boundary).  min over polygons == distance to the union."""
    pts = np.asarray(points, dtype=np.float64)
    flat = pts.reshape(-1, 2)
    if len(polys) == 0:
        return np.full(pts.shape[:-1], NO_MAP_GAP)
    sh = _sh()
    tree = sh.STRtree(list(polys))
    idx, dist = tree.query_nearest(sh.points(flat), return_distance=True, all_matches=False)
    out = np.full(len(flat), np.inf)
    out[idx[0]] = dist
    return out.reshape(pts.shape[:-1])


def corner_gap(corners, polys) -> np.ndarray:
    """g(t): corners [..., T, n, 2] -> [..., T] = MAX over the n points of the distance to the union of the drivable polygons (0 when all are inside)."""
    c = np.asarray(corners, dtype=np.float64)
    return point_gap(c, polys).max(axis=-1)


def dac_from_gap(gap, tol: float = 0.0) -> np.ndarray:
    """The identity's prediction: DAC = 1 iff g(t) <= tol at every step.  [..., T] -> [...] bool (True = compliant)."""
    return (np.asarray(gap) <= tol).all(axis=-1)


def inside_any(points, polys) -> np.ndarray:
    """[..., 2] -> [...] bool: the point lies strictly inside at least one polygon (shapely `within`, the occupancy map's `is_in_layer` rule)."""
    pts = np.asarray(points, dtype=np.float64)
    out = np.zeros(pts.shape[:-1], dtype=bool)
    if len(polys) == 0:
        return out
    sh = _sh()
    idx = sh.STRtree(list(polys)).query(sh.points(pts.reshape(-1, 2)), predicate="within")
    flat = out.reshape(-1)
    flat[idx[0]] = True
    return flat.reshape(out.shape)


def multiple_lanes(corners, lane_polys) -> np.ndarray:
    """[..., T, 4, 2] -> [..., T] bool, the scorer's MULTIPLE_LANES rule: more than one lane polygon holds >= 1 corner AND no lane polygon holds all of them."""
    c = np.asarray(corners, dtype=np.float64)
    lead, n = c.shape[:-2], c.shape[-2]
    out = np.zeros(lead, dtype=bool)
    if len(lane_polys) == 0:
        return out
    sh = _sh()
    idx = sh.STRtree(list(lane_polys)).query(sh.points(c.reshape(-1, 2)), predicate="within")
    nl, ns = len(lane_polys), int(np.prod(lead))
    cnt = np.bincount((idx[0] // n) * nl + idx[1], minlength=ns * nl).reshape(ns, nl)
    mult = ((cnt > 0).sum(axis=1) > 1) & (cnt != n).all(axis=1)
    return mult.reshape(lead)


def wrap_angle(a):
    return (np.asarray(a, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def _interp_linear(xp, fp, x):
    """scipy.interpolate.interp1d(kind='linear') semantics, which PDMPath.interpolate uses: at a DUPLICATED knot it returns the FIRST duplicate's value
    (np.interp returns the last).  MEASURED on navtest: the PDM-Closed centreline repeats vertices, and np.interp was 0.66 deg off at those points."""
    xp, fp, x = np.asarray(xp, dtype=np.float64), np.asarray(fp, dtype=np.float64), np.asarray(x, dtype=np.float64)
    hi = np.clip(np.searchsorted(xp, x), 1, len(xp) - 1)
    lo = hi - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (fp[hi] - fp[lo]) / (xp[hi] - xp[lo])
    return slope * (x - xp[lo]) + fp[lo]


class CenterLine:
    """The PDM-Closed centreline as a polyline [N, 3] = (x, y, heading): the scorer's lateral deviation (Point.distance to the linestring) plus the
    heading error and the signed lateral coordinate.  Mirrors PDMPath: linestring from xy, progress = cumulative length, heading = linear interpolation of
    the UNWRAPPED heading at the projection, clipped to [1e-5, length]."""

    def __init__(self, xyh):
        a = np.asarray(xyh, dtype=np.float64)
        if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 3:
            raise ValueError(f"centreline needs shape [N>=2, 3], got {a.shape}")
        self.xy = a[:, :2]
        self.h = np.unwrap(a[:, 2])
        self.s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))])
        self.line = _sh().linestrings(self.xy)

    def trace(self, centers, headings) -> dict:
        """centers [..., 2], headings [...] -> offset (scorer's deviation, m), dpsi_deg (ego - path heading, wrapped, deg), lat (signed, left positive, m)."""
        sh = _sh()
        c = np.asarray(centers, dtype=np.float64)
        pts = sh.points(c.reshape(-1, 2))
        off = sh.distance(pts, self.line)
        with warnings.catch_warnings():              # PDMPath.project silences the same shapely 2.0 "invalid value in line_locate_point"
            warnings.simplefilter("ignore", RuntimeWarning)
            s = np.clip(sh.line_locate_point(self.line, pts), 1e-5, self.s[-1])
        hp = _interp_linear(self.s, self.h, s)
        xp, yp = _interp_linear(self.s, self.xy[:, 0], s), _interp_linear(self.s, self.xy[:, 1], s)
        lat = -(c.reshape(-1, 2)[:, 0] - xp) * np.sin(hp) + (c.reshape(-1, 2)[:, 1] - yp) * np.cos(hp)
        dpsi = np.degrees(wrap_angle(np.asarray(headings, dtype=np.float64).reshape(-1) - hp))
        shp = c.shape[:-1]
        return {"offset": off.reshape(shp), "dpsi_deg": dpsi.reshape(shp), "lat": lat.reshape(shp)}


def dither_flag(lat, dt: float = 0.1, win: int = 5, v_dead: float = DITHER_V_DEAD, min_changes: int = DITHER_MIN_CHANGES) -> np.ndarray:
    """lat [M, T] signed lateral coordinate -> [M] 0/1: >= `min_changes` sign changes of the lateral velocity (difference over `win` steps), |v| < v_dead ignored."""
    lat = np.asarray(lat, dtype=np.float64)
    v = (lat[:, win:] - lat[:, :-win]) / (win * dt)
    sg = np.sign(v) * (np.abs(v) >= v_dead)
    out = np.zeros(len(lat))
    for i, row in enumerate(sg):
        r = row[row != 0]
        out[i] = float((np.diff(r) != 0).sum() >= min_changes) if len(r) > 1 else 0.0
    return out


def executed_lane_change(poses_xy, sibling_polys: dict, start_id: str, baseline, off_m: float = LC_OFFSET_M) -> np.ndarray:
    """Label (i)'s rule on [M, K, 2] rear-axle poses: some pose in a LANE of the start roadblock OTHER than the start lane, AND the final pose > off_m from
    the start lane's baseline path.  sibling_polys {lane_id: polygon}; baseline a shapely LineString (None => nothing can be judged => False)."""
    p = np.asarray(poses_xy, dtype=np.float64)
    if baseline is None or not sibling_polys:
        return np.zeros(len(p), dtype=bool)
    sh = _sh()
    ids = list(sibling_polys)
    idx = sh.STRtree([sibling_polys[i] for i in ids]).query(sh.points(p.reshape(-1, 2)), predicate="within")
    other = np.zeros(len(p) * p.shape[1], dtype=bool)
    for pt, ln in zip(idx[0], idx[1]):
        if ids[ln] != start_id:
            other[pt] = True
    other = other.reshape(len(p), p.shape[1]).any(axis=1)
    off_end = sh.distance(sh.points(p[:, -1, :]), baseline)
    return other & (off_end > off_m)


def route_needs_lane_change(rb0_on_route: bool, start_reaches: bool, adjacent_reaches) -> bool:
    """Label (iii): the start lane cannot reach the next on-route roadblock while an adjacent lane can."""
    return bool(rb0_on_route and (not start_reaches) and any(adjacent_reaches))


def check_executed(states) -> np.ndarray:
    s = np.asarray(states, dtype=np.float64)
    if s.ndim != 3 or s.shape[1] != EXECUTED_STEPS or s.shape[2] < 3:
        raise ValueError(f"expected EXECUTED states [M, {EXECUTED_STEPS}, >=3] (the LQR-tracked 0.1 s grid), got {s.shape}")
    return s


# ------------------------------------------------------------------ one scene, from states + a geometry bundle
class SceneGeometry:
    """Plain data a scene's terms are computed from (shapely objects only): the DAC polygons, the INTERSECTION polygons, the LANE+LANE_CONNECTOR polygons,
    the PDM-Closed centreline, and (optionally) the start-lane context for L."""

    def __init__(self, drivable, intersection, lanes, centerline, lane_ctx=None):
        self.drivable, self.intersection, self.lanes = list(drivable), list(intersection), list(lanes)
        self.centerline = centerline
        self.lane_ctx = lane_ctx                  # {'start_id', 'siblings', 'baseline', 'human_lc', 'route_needs', 'poses'} or None


def terms_from_states(states, geo: SceneGeometry, want: str = "DKL", *, traces: bool = False) -> dict:
    """states [M, 41, >=3] (rear-axle x, y, heading) -> {'d','gmax','k','k_raw','lc_executed','label_fires','dither'}; the parts not in `want` are absent."""
    s = check_executed(states)
    co = bbox_coords(s)
    out: dict = {}
    lab = bool(geo.lane_ctx and (geo.lane_ctx["human_lc"] or geo.lane_ctx["route_needs"]))
    if "D" in want:
        g = corner_gap(co["corners"], geo.drivable)
        out["d"] = RW.drivable_excess(g)
        out["gmax"] = g.max(axis=-1)
        if traces:
            out["g"] = g
    if "K" in want or "L" in want:
        ct = geo.centerline.trace(co["centers"], s[..., 2])
    if "K" in want:
        in_int = inside_any(co["centers"], geo.intersection)
        mult = multiple_lanes(co["corners"], geo.lanes)
        strad_raw = (mult & ~in_int).mean(axis=-1)
        out["k_raw"] = RW.lane_consistency(ct["offset"], ct["dpsi_deg"], in_int, strad_raw)
        out["k"] = RW.lane_consistency(ct["offset"], ct["dpsi_deg"], in_int | lab, strad_raw * (0.0 if lab else 1.0))
        if traces:
            out.update(offset=ct["offset"], dpsi_deg=ct["dpsi_deg"], in_intersection=in_int, multiple_lanes=mult)
    if "L" in want:
        lc = geo.lane_ctx
        poses = s[:, ::LC_POSE_STRIDE, :2][:, 1:]
        out["lc_executed"] = (executed_lane_change(poses, lc["siblings"], lc["start_id"], lc["baseline"]).astype(int)
                              if lc else np.zeros(len(s), dtype=int))
        out["label_fires"] = lab
        out["dither"] = dither_flag(ct["lat"]).astype(int)
    return out


# ------------------------------------------------------------------ nuPlan / NAVSIM adapters (imported lazily; dev box: the navsim v1.1 venv)
def _layer_name(t) -> str:
    return getattr(t, "name", str(t))


def scene_geometry(parts, scenario=None, want: str = "DKL") -> SceneGeometry:
    """navsim_pdm_targets.metric_cache_parts(...) (+ the scenario for L) -> SceneGeometry."""
    _, _, _, centerline, _, dam = parts
    polys = dam._geometries
    names = [_layer_name(t) for t in dam.map_types]
    pick = lambda keys: [polys[i] for i, n in enumerate(names) if n in keys]                    # noqa: E731
    cl = CenterLine(np.array([[p.x, p.y, p.heading] for p in centerline.discrete_path]))
    ctx = lane_context(scenario) if (scenario is not None and ("K" in want or "L" in want)) else None
    return SceneGeometry(pick(DAC_LAYERS), pick(("INTERSECTION",)), pick(LANE_LAYERS), cl, ctx)


def lane_context(sc) -> dict:
    """Start-lane context and the scene-level labels (i) / (iii) from the nuPlan map API.  None when the ego is not on a LANE (connector / off map)."""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as ML
    sh = _sh()
    e0 = sc.initial_ego_state
    m = sc.map_api
    lanes = m.get_all_map_objects(Point2D(e0.rear_axle.x, e0.rear_axle.y), ML.LANE)
    if not lanes:
        return {"start_id": None, "siblings": {}, "baseline": None, "human_lc": False, "route_needs": False, "none": True}
    l0 = lanes[0]
    rb0 = str(l0.get_roadblock_id())
    rb = m.get_map_object(rb0, ML.ROADBLOCK)
    sib = {str(x.id): x.polygon for x in (rb.interior_edges if rb is not None else [l0])}
    baseline = sh.linestrings(np.array([(p.x, p.y) for p in l0.baseline_path.discrete_path]))
    hum = np.array([(s.rear_axle.x, s.rear_axle.y) for s in sc.get_ego_future_trajectory(0, 4.0, 8)])[None]
    human_lc = bool(executed_lane_change(hum, sib, str(l0.id), baseline)[0])
    route = [str(x) for x in sc.get_route_roadblock_ids()]
    on_route = rb0 in route
    nxt = set(route[route.index(rb0) + 1: route.index(rb0) + 3]) if on_route else set()

    def reaches(lane) -> bool:
        for e in lane.outgoing_edges:
            if str(e.get_roadblock_id()) in nxt:
                return True
            if any(str(e2.get_roadblock_id()) in nxt for e2 in e.outgoing_edges):
                return True
        return False
    adj = [a for a in l0.adjacent_edges if a is not None]
    needs = route_needs_lane_change(on_route, reaches(l0), [reaches(a) for a in adj])
    return {"start_id": str(l0.id), "siblings": sib, "baseline": baseline, "human_lc": human_lc, "route_needs": needs, "none": False}


def executed_states(parts, plans8) -> np.ndarray:
    """[M, 8, 3] NAVSIM-frame plans -> the executed LQR-tracked states [M, 41, 11] (PDM-Closed's row dropped).  This is the front half of
    navsim_pdm_targets.score_plans, line for line (test_rl_terms_label pins that textually), so the states are the very ones PDMScorer._states holds;
    the identity run asserts bit equality on every scene.  (navsim_dac.simulated_states, the vendored simulator, differs from them by ~3e-6.)"""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    import navsim_pdm_targets as PT
    S = PT._setup()
    pdm_traj, ego = parts[0], parts[1]
    states = [get_trajectory_as_array(pdm_traj, S["samp"], ego.time_point)]
    for p in np.asarray(plans8, dtype=np.float32):
        it = transform_trajectory(Trajectory(p, S["samp8"]), ego)
        states.append(get_trajectory_as_array(it, S["samp"], ego.time_point))
    return S["sim"].simulate_proposals(np.stack(states, 0), ego)[1:]


def label_scene(sc, plans8, want: str = "DKL", parts=None) -> dict:
    import navsim_pdm_targets as PT
    parts = parts or PT.metric_cache_parts(sc)
    geo = scene_geometry(parts, sc, want)
    t = terms_from_states(executed_states(parts, plans8), geo, want)
    if geo.lane_ctx is not None:
        t["labels"] = {"human_lc": bool(geo.lane_ctx["human_lc"]), "route_needs": bool(geo.lane_ctx["route_needs"]),
                       "start_lane": not geo.lane_ctx.get("none", False)}
    return t


def make_record(key, ckpt_step: int, label_version: int, t: dict) -> dict:
    rec = {"kind": "rl_terms", "key": list(key), "ckpt_step": int(ckpt_step), "label_version": int(label_version), "plans": "executed (A7 repair)"}
    for n in ("d", "gmax", "k", "k_raw"):
        if n in t:
            rec[n] = [round(float(x), 5) for x in t[n]]
    for n in ("lc_executed", "dither"):
        if n in t:
            rec[n] = [int(x) for x in t[n]]
    if "label_fires" in t:
        rec["label_fires"] = bool(t["label_fires"])
    if "labels" in t:
        rec["labels"] = t["labels"]
    return rec


def side_file_name(src_path: str) -> str:
    """onpolicy_<tag>.jsonl -> rl_terms_onpolicy_<tag>.jsonl (the pdm side files are pdm_<same name>); the name the EPO loader globs."""
    return RL_TERMS_PREFIX + os.path.basename(src_path)


def check_terms(terms: str) -> str:
    """--terms: a non-empty subset of D, K, L."""
    if not terms or not set(terms) <= set("DKL"):
        raise ValueError(f"--terms {terms!r}: need a non-empty subset of DKL")
    return terms


def shard_of(log: str, n: int) -> int:
    """Log-disjoint, deterministic sharding: one log -> one shard, on every machine."""
    return zlib.crc32(log.encode("utf-8")) % n


# ------------------------------------------------------------------ CLI (mirrors refe/onpolicy_relabel_pdm.py)
_S: dict = {}


def _init():
    import navtrain_scenarios as NS
    _S["NS"] = NS
    _S["dbs"] = NS.index_dbs()


def label_log(job):
    import navsim_pdm_targets as PT
    log, rows, want = job
    out, failed, skipped = [], 0, 0
    try:
        if not _S:
            _init()
        scs = {sc.scenario_name: sc for sc in _S["NS"].build_scenarios_for_log(
            _S["dbs"][log], sorted({r["token"] for r in rows}), history_rows=40, future_rows=100)}
    except Exception as e:                                                   # noqa: BLE001
        return [], len(rows), 0, f"{log}: scenario build {type(e).__name__}: {str(e)[:120]}"
    for r in rows:
        sc = scs.get(r["token"])
        if sc is None:                                  # the 5 s future guard (NAVSIM's observation horizon) excludes it
            skipped += 1
            continue
        try:
            props = np.concatenate([np.asarray(r["traj"], np.float64), np.asarray(r["yaw"], np.float64)[..., None]], -1)
            t = label_scene(sc, PT.executed_plans8(props, repair=True), want)
            rec = make_record([r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0))], r["ckpt_step"],
                              r.get("label_version", 1), t)
            out.append(json.dumps(rec) + "\n")
        except Exception:                                                    # noqa: BLE001
            failed += 1
            if failed <= 2:
                print(f"  {r.get('token')}: {traceback.format_exc()[-400:]}", flush=True)
    return out, failed, skipped, None


def main() -> int:
    import onpolicy_relabel_pdm as OR
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--files", default="onpolicy_*.jsonl")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--served-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="I/N: label only the logs whose crc32 mod N == I (log-disjoint across machines)")
    ap.add_argument("--terms", default="DKL", help="subset of D, K, L")
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split("/"))
    if not (0 <= si < sn):
        raise SystemExit(f"--shard {a.shard}: need 0 <= I < N")
    try:
        check_terms(a.terms)
    except ValueError as e:
        raise SystemExit(str(e))
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.sets, a.files)))
    best = OR.served_keys(files) if a.served_only else None
    if best is not None:
        print(f"served sets: {len(best):,}", flush=True)
    n_sets = n_fail = n_skip = 0
    t0 = time.time()
    CH = 3000
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for fpath in files:
            dst = os.path.join(a.out, side_file_name(fpath))
            done = set()
            if os.path.exists(dst):
                for ln in open(dst, encoding="utf-8"):
                    try:
                        d = json.loads(ln)
                        done.add((tuple(d["key"]), d["ckpt_step"]))
                    except (json.JSONDecodeError, KeyError):
                        pass

            def flush(by, f):
                c = [0, 0, 0]
                for lines, failed, skipped, err in pool.imap_unordered(label_log, [(lg, rs, a.terms) for lg, rs in by.items()]):
                    if err:
                        print("  " + err, flush=True)
                    f.writelines(lines)
                    f.flush()
                    c[0] += len(lines); c[1] += failed; c[2] += skipped
                return c

            by: dict = {}
            k = 0
            with open(dst, "a", encoding="utf-8") as f:
                for ln in open(fpath, encoding="utf-8"):
                    try:
                        r = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    if r.get("kind") != "onpolicy_set" or shard_of(r["log_name"], sn) != si:
                        continue
                    key = (r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0)))
                    rk = (int(r["ckpt_step"]), int(r.get("label_version", 1)))
                    if best is not None and best.get(key) != rk:
                        continue
                    if (key, rk[0]) in done:
                        continue
                    by.setdefault(r["log_name"], []).append({"log_name": r["log_name"], "token": r.get("token", ""),
                                                             "step": r["step"], "rank": r.get("rank", 0),
                                                             "ckpt_step": r["ckpt_step"], "label_version": r.get("label_version", 1),
                                                             "traj": r["traj"], "yaw": r["yaw"]})
                    k += 1
                    if k % CH == 0:
                        c = flush(by, f)
                        n_sets += c[0]; n_fail += c[1]; n_skip += c[2]
                        by = {}
                        print(f"  {os.path.basename(fpath)}: {k:,} read, {n_sets:,} labelled, {n_skip} skipped (5 s guard), "
                              f"{n_fail} failed, {(time.time() - t0) / max(n_sets, 1):.3f} s/set", flush=True)
                    if a.limit and k >= a.limit:
                        break
                if by:
                    c = flush(by, f)
                    n_sets += c[0]; n_fail += c[1]; n_skip += c[2]
            print(f"{os.path.basename(fpath)}: total {n_sets:,} labelled, {n_skip} skipped, {n_fail} failed", flush=True)
    print(f"ZZRELABEL_RLT_DONE {n_sets} {n_skip} {n_fail}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
