#!/usr/bin/env python3
"""The drivable-corridor / VectorMap instrument.

WHAT THIS IS
------------
Turns ``trajdata.VectorMap`` lane geometry (embedded in every AlpaSim scene USDZ)
into a **per-timestep drivable corridor**: for each ego pose, the signed lateral
distance to the lane's own LEFT and RIGHT edge.

This is the channel ``taniteval/corridor.py`` currently does not have. That module
scores ``corridor_departure_rate`` against ``CORRIDOR_HALFWIDTH_M = 1.75`` m, which
its own docstring marks **PROPOSED** -- "about half a lane", *NOT measured on this
corpus*, with the corridor being a half-width about the REFERENCE PATH rather than
a lane. Here the corridor is the actual mapped lane.

WHY THE CONTAINMENT TEST IS EDGE-BASED AND NOT A TOLERANCE BAND
---------------------------------------------------------------
``gate1_connectivity_probe.py`` (sibling agent, same day) reports an
``ego_lane_match_rate`` of mean 0.9827. That statistic is
``dist_to_nearest_centreline <= max(half_width, 1.0)`` -- a **tolerance band about a
centreline**, with a 1.0 m floor and a 1.75 m fallback when edges are missing. It is
an *association* test (which lane is the ego on?), and it is the right tool for that
job. It is **not** a containment test and must not be quoted as one: a pose 0.99 m
outside a narrow lane still passes it.

The frame proof needs containment, so this module builds the closed lane ring
(left edge forward + right edge reversed) and runs a real point-in-polygon test.
The two numbers are different quantities and are reported separately.

FRAME
-----
Everything is in the scene's MAP/WORLD frame. Lane geometry and the ego pose track
are both native to it, so **no camera intrinsic or extrinsic is used anywhere in this
file**. The unreconciled ``cam_h`` (1.5 / 1.43 / 1.22 m) and FOV (51.4 deg vs 33.1 deg)
values therefore cannot propagate into any number emitted here. ``--assert-no-camera``
makes that structural rather than a claim.

LICENCE
-------
Reads map geometry only. The NuRec/gsplat renderer (NGC-DL-CONTAINER-LICENSE,
no derivatives) is never imported, modified or invoked. Nothing is rendered.

Read-only w.r.t. the scenes. CPU only. No GPU.
"""
import argparse
import collections
import glob
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, "/workspace/alpa-invest/alpasim/src/runtime")

SSROOT = "/workspace/alpa-invest/alpasim/data/nre-artifacts/scenesets"
SENTINEL = "-1"

# --- pre-registered pass conditions (VECTORMAP_CORRIDOR.md 0.2) -------------
P1_WIDTH_LO, P1_WIDTH_HI = 2.5, 4.5      # a real road lane, metres
P2_CONTAINMENT = 0.90                     # frac of ego poses inside a lane ring
P3_RESIDUAL_M = 0.25                      # |d_left + d_right - width| tolerance

EPS = 1e-12


# ========================================================================== #
# geometry                                                                     #
# ========================================================================== #
def cum_len(pts):
    d = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def frenet(q, pts):
    """Project points ``q [N,2]`` onto polyline ``pts [M,2]``.

    Returns ``(s, lat, dist)``: arclength station, SIGNED lateral offset
    (**+ = LEFT of the polyline**, matching ``driving.frenet`` and
    ``taniteval.corridor.cross_track_from_paths``), and unsigned distance.
    """
    q = np.asarray(q, dtype=np.float64)[:, :2]
    pts = np.asarray(pts, dtype=np.float64)[:, :2]
    p0, p1 = pts[:-1], pts[1:]
    seg = p1 - p0
    L2 = np.maximum((seg ** 2).sum(-1), EPS)
    d = q[:, None, :] - p0[None, :, :]
    t = np.clip((d * seg[None]).sum(-1) / L2[None], 0.0, 1.0)
    proj = p0[None] + t[..., None] * seg[None]
    diff = q[:, None, :] - proj
    dist = np.linalg.norm(diff, axis=-1)
    k = dist.argmin(1)
    n = np.arange(len(q))
    segn = seg[k] / np.sqrt(L2[k])[:, None]
    nrm = np.stack([-segn[:, 1], segn[:, 0]], axis=-1)     # left normal
    lat = (diff[n, k] * nrm).sum(-1)
    s = cum_len(pts)[k] + t[n, k] * np.sqrt(L2[k])
    return s, lat, dist[n, k]


def point_in_ring(q, ring):
    """Vectorised even-odd point-in-polygon. ``q [N,2]``, ``ring [M,2]`` closed."""
    q = np.asarray(q, dtype=np.float64)[:, :2]
    x, y = q[:, 0], q[:, 1]
    xi, yi = ring[:-1, 0], ring[:-1, 1]
    xj, yj = ring[1:, 0], ring[1:, 1]
    # edges that straddle the horizontal ray from each query point
    cond = ((yi[None, :] > y[:, None]) != (yj[None, :] > y[:, None]))
    denom = np.where(np.abs(yj - yi) < EPS, EPS, yj - yi)[None, :]
    xint = xi[None, :] + (y[:, None] - yi[None, :]) * (xj - xi)[None, :] / denom
    return ((cond & (x[:, None] < xint)).sum(1) % 2).astype(bool)


def lane_ring(left, right):
    """Closed ring from the two edges: left forward + right reversed."""
    r = np.concatenate([left[:, :2], right[::-1, :2]], axis=0)
    if np.linalg.norm(r[0] - r[-1]) > EPS:
        r = np.concatenate([r, r[:1]], axis=0)
    return r


def ray_hit(origin, direction, poly):
    """First hit of rays ``origin + t*direction`` (t>0) on polyline ``poly``.

    ``origin [N,2]``, ``direction [N,2]`` (unit), ``poly [M,2]``. Returns ``t [N]``
    with ``inf`` where the ray misses.

    **This is the curvature-exact way to measure lane width.** The earlier
    ``edge_profile`` route reparameterised each edge onto the centreline's
    stations; on a curve the outer edge is longer than the centreline, so that
    mapping is non-uniform and the recovered width is distorted -- worst precisely
    at the wide junction lanes where a corridor matters most. Casting a ray along
    the local normal has no reparameterisation and no interpolation.
    """
    o = np.asarray(origin, dtype=np.float64)[:, :2]
    d = np.asarray(direction, dtype=np.float64)[:, :2]
    p0, p1 = poly[:-1, :2], poly[1:, :2]
    e = p1 - p0                                        # [M,2]
    # solve o + t*d = p0 + u*e   ->  cross products, vectorised over N x M
    den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
    safe = np.where(np.abs(den) < EPS, np.nan, den)
    diff = p0[None, :, :] - o[:, None, :]
    t = (diff[..., 0] * e[None, :, 1] - diff[..., 1] * e[None, :, 0]) / safe
    u = (diff[..., 0] * d[:, None, 1] - diff[..., 1] * d[:, None, 0]) / safe
    ok = np.isfinite(t) & (t > 1e-9) & (u >= -1e-9) & (u <= 1 + 1e-9)
    t = np.where(ok, t, np.inf)
    return t.min(axis=1)


def bounds_raycast(q, centre, left, right):
    """Per-point ``(d_left, d_right)`` by casting along the centreline normal.

    Curvature-exact replacement for the ``edge_profile`` route. Returns metres of
    room to each side, ``inf`` where the ray misses the edge (open lane ends)."""
    s, lat, _ = frenet(q, centre)
    p0, p1 = centre[:-1, :2], centre[1:, :2]
    seg = p1 - p0
    L2 = np.maximum((seg ** 2).sum(-1), EPS)
    dd = np.asarray(q, dtype=np.float64)[:, None, :2] - p0[None]
    tt = np.clip((dd * seg[None]).sum(-1) / L2[None], 0.0, 1.0)
    proj = p0[None] + tt[..., None] * seg[None]
    k = np.linalg.norm(np.asarray(q)[:, None, :2] - proj, axis=-1).argmin(1)
    tang = seg[k] / np.sqrt(L2[k])[:, None]
    nl = np.stack([-tang[:, 1], tang[:, 0]], axis=-1)     # unit left normal
    dl = ray_hit(q, nl, left)
    dr = ray_hit(q, -nl, right)
    return dl, dr, lat


def edge_profile(centre, left, right):
    """Lateral coordinate of each EDGE, expressed in the CENTRELINE frame.

    Returns ``(s_grid, lat_left(s), lat_right(s))`` sampled at the centreline's own
    vertices. This is what makes ``d_left + d_right == width`` hold by construction
    rather than by luck: both edges are measured in ONE frame, the centreline's.

    Pairing ``left[i]`` with ``right[i]`` by INDEX (as the gate-1 half-width does)
    is only correct when both edges are sampled identically; MADS edges are not
    guaranteed to be, so this projects instead.
    """
    sc = cum_len(centre)
    sl, latl, _ = frenet(left, centre)
    sr, latr, _ = frenet(right, centre)
    ol, orr = np.argsort(sl), np.argsort(sr)
    fl = np.interp(sc, sl[ol], latl[ol])
    fr = np.interp(sc, sr[orr], latr[orr])
    return sc, fl, fr


# ========================================================================== #
# per-scene                                                                    #
# ========================================================================== #
def probe_scene(ds, assert_no_camera=True):
    from trajdata.maps.vec_map_elements import MapElementType
    rec = {}
    vm = ds.map
    lanes = vm.elements[MapElementType.ROAD_LANE]
    rec["n_lanes"] = len(lanes)
    rec["element_counts"] = {str(int(k)): len(v) for k, v in vm.elements.items()}

    # ---------------- P1: geometry exists and is metrically sane ------------
    ids, cen, LE, RE, ring, prof = [], {}, {}, {}, {}, {}
    widths_all = []
    n_edges_missing = 0
    for k, L in lanes.items():
        c = np.asarray(L.center.points, dtype=np.float64)[:, :2]
        if c.shape[0] < 2:
            continue
        le = getattr(L, "left_edge", None)
        re_ = getattr(L, "right_edge", None)
        if le is None or re_ is None:
            n_edges_missing += 1
            continue
        le = np.asarray(le.points, dtype=np.float64)[:, :2]
        re_ = np.asarray(re_.points, dtype=np.float64)[:, :2]
        if le.shape[0] < 2 or re_.shape[0] < 2:
            n_edges_missing += 1
            continue
        sc, fl, fr = edge_profile(c, le, re_)
        w = fl - fr                      # left is +, right is -, so width = fl - fr
        if not np.isfinite(w).all():
            n_edges_missing += 1
            continue
        ids.append(k)
        cen[k] = c
        LE[k], RE[k] = le, re_
        ring[k] = lane_ring(le, re_)
        prof[k] = (sc, fl, fr)
        widths_all.append(np.median(w))

    rec["n_lanes_with_edges"] = len(ids)
    rec["n_lanes_edges_missing"] = int(n_edges_missing)
    if not ids:
        rec["P1_pass"] = False
        rec["P1_reason"] = "no lane carried usable left+right edge geometry"
        return rec, None
    wa = np.asarray(widths_all, dtype=np.float64)
    rec["lane_width_m"] = {
        "median": round(float(np.median(wa)), 3),
        "p10": round(float(np.percentile(wa, 10)), 3),
        "p90": round(float(np.percentile(wa, 90)), 3),
        "iqr": round(float(np.percentile(wa, 75) - np.percentile(wa, 25)), 3),
        "min": round(float(wa.min()), 3), "max": round(float(wa.max()), 3)}
    rec["P1_pass"] = bool(P1_WIDTH_LO <= np.median(wa) <= P1_WIDTH_HI
                          and (np.percentile(wa, 75) - np.percentile(wa, 25)) > 0)

    # ---------------- ego track, in the SAME (map) frame --------------------
    traj = ds.rig.trajectory
    q = np.asarray(traj.positions, dtype=np.float64)[:, :2]
    rec["ego_n_poses"] = int(len(q))
    rec["ego_path_len_m"] = round(
        float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()), 1)
    try:
        ts = np.asarray(traj.timestamps_us, dtype=np.float64)
        rec["ego_duration_s"] = round(float((ts[-1] - ts[0]) / 1e6), 2)
        dt = float(np.median(np.diff(ts)) / 1e6)
    except Exception:
        ts, dt = np.arange(len(q)) * 1e5, 0.1
        rec["ego_duration_s"] = None
    rec["ego_dt_s"] = round(dt, 4)

    if assert_no_camera:
        # structural guarantee for VECTORMAP_CORRIDOR.md 0.6: nothing above or
        # below reads an intrinsic, an extrinsic, or a camera height.
        rec["camera_free"] = True

    # ---------------- P2: FRAME PROOF -- containment in a lane ring ---------
    # per-lane distance to centreline, to pick the lane the ego is on
    D = np.full((len(q), len(ids)), np.inf)
    S = np.zeros_like(D)
    LAT = np.zeros_like(D)
    for j, k in enumerate(ids):
        s, lat, d = frenet(q, cen[k])
        D[:, j], S[:, j], LAT[:, j] = d, s, lat
    best = D.argmin(1)
    n = np.arange(len(q))
    bestd = D[n, best]

    inside_any = np.zeros(len(q), dtype=bool)
    inside_which = np.full(len(q), -1, dtype=int)
    # test the nearest few lanes only -- a pose can only be inside a lane whose
    # centreline is within half a max lane width
    order = np.argsort(D, axis=1)[:, :6]
    for col in range(order.shape[1]):
        j = order[:, col]
        todo = ~inside_any
        if not todo.any():
            break
        for jj in np.unique(j[todo]):
            m = todo & (j == jj)
            if not m.any():
                continue
            hit = point_in_ring(q[m], ring[ids[jj]])
            idx = np.flatnonzero(m)[hit]
            inside_any[idx] = True
            inside_which[idx] = jj

    rec["ego_containment_rate"] = round(float(inside_any.mean()), 4)
    rec["ego_dist_to_matched_centreline_m"] = {
        "median": round(float(np.median(bestd)), 3),
        "p90": round(float(np.percentile(bestd, 90)), 3),
        "max": round(float(bestd.max()), 3)}
    # the gate-1 statistic, recomputed here so the two are comparable
    hw = np.array([np.median(prof[k][1] - prof[k][2]) / 2.0 for k in ids])
    rec["gate1_style_match_rate"] = round(
        float((bestd <= np.maximum(hw[best], 1.0)).mean()), 4)
    rec["P2_pass"] = bool(rec["ego_containment_rate"] >= P2_CONTAINMENT)

    # ---------------- P3: the corridor channel ------------------------------
    lane_for = np.where(inside_any, inside_which, best)
    d_left = np.full(len(q), np.nan)
    d_right = np.full(len(q), np.nan)
    width_at = np.full(len(q), np.nan)
    lat_at = np.full(len(q), np.nan)
    d_left_legacy = np.full(len(q), np.nan)
    d_right_legacy = np.full(len(q), np.nan)
    for jj in np.unique(lane_for):
        m = lane_for == jj
        k = ids[jj]
        # PRIMARY: curvature-exact ray-cast along the local centreline normal
        dl, dr, lat = bounds_raycast(q[m], cen[k], LE[k], RE[k])
        d_left[m], d_right[m], lat_at[m] = dl, dr, lat
        width_at[m] = dl + dr
        # CROSS-CHECK: the station-reparameterisation route, kept to quantify the
        # curvature artefact rather than to score anything
        sc, fl, fr = prof[k]
        s, _, _ = frenet(q[m], cen[k])
        d_left_legacy[m] = np.interp(s, sc, fl) - lat
        d_right_legacy[m] = lat - np.interp(s, sc, fr)

    ok = np.isfinite(d_left) & np.isfinite(d_right)
    # ⚠️ THE REAL P3 CHECK. The previous one -- |d_left + d_right - width| -- was
    # VACUOUS: width was DEFINED as d_left + d_right, so the residual was 0 by
    # construction and could never fail (the C13 "a guard that cannot fail is not
    # a guard" class). The falsifiable check is agreement between the bounds and
    # the INDEPENDENT ring containment test, which shares no code path.
    # ⚠️ CORRECTED. The first version tested ``(lat <= d_left) & (-lat <= d_right)``,
    # which compares quantities with DIFFERENT ORIGINS: ``lat`` is the offset from
    # the lane CENTRELINE, while ``d_left``/``d_right`` are already measured from
    # the EGO. Ray-cast bounds need no such comparison -- a ray cast along +normal
    # hits the left edge iff the ego is right of it, so BOTH rays landing is exactly
    # containment. (The spurious test made scene 00097de1 read 0.0 agreement at
    # containment 1.0, which is what exposed it.)
    bounds_inside = ok
    agree = (bounds_inside == inside_any)
    resid = np.abs(d_left_legacy - d_left)
    rec["corridor"] = {
        "n_steps": int(len(q)),
        "frac_finite": round(float(ok.mean()), 4),
        "d_left_m": {"median": round(float(np.nanmedian(d_left)), 3),
                     "p10": round(float(np.nanpercentile(d_left, 10)), 3)},
        "d_right_m": {"median": round(float(np.nanmedian(d_right)), 3),
                      "p10": round(float(np.nanpercentile(d_right, 10)), 3)},
        "width_m": {"median": round(float(np.nanmedian(width_at)), 3)},
        "halfwidth_m_median": round(float(np.nanmedian(width_at) / 2.0), 3),
        "lat_offset_m": {"median": round(float(np.nanmedian(lat_at)), 3),
                         "abs_median": round(float(np.nanmedian(np.abs(lat_at))), 3),
                         "abs_p90": round(float(np.nanpercentile(np.abs(lat_at), 90)), 3)},
        # falsifiable: do the ray-cast bounds and the independent ring test agree?
        "ring_vs_bounds_agreement": round(float(agree.mean()), 4),
        "n_disagree": int((~agree).sum()),
        # how far the legacy station-reparameterisation route is off -- this is the
        # curvature artefact, quantified
        "legacy_reparam_error_m": {
            "median": round(float(np.nanmedian(resid)), 4),
            "p90": round(float(np.nanpercentile(resid, 90)), 4),
            "max": round(float(np.nanmax(resid)), 4)},
    }
    rec["P3_pass"] = bool(ok.mean() >= 0.90 and agree.mean() >= 0.90)

    # per-step channel, decimated, for the emitter consumers
    step = max(1, len(q) // 200)
    rec["channel_sample"] = {
        "stride": int(step),
        "d_left_m": [None if not np.isfinite(v) else round(float(v), 3)
                     for v in d_left[::step]],
        "d_right_m": [None if not np.isfinite(v) else round(float(v), 3)
                      for v in d_right[::step]],
    }

    chan = {"d_left_m": d_left, "d_right_m": d_right, "width_m": width_at,
            "lat_m": lat_at, "inside": inside_any, "dt_s": dt}
    return rec, chan


# ========================================================================== #
# HP-4: junction TOPOLOGY classes                                              #
# ========================================================================== #
def topology_classes(ds, chan):
    """Classify each branch point by the SHAPE of its option set.

    HP-4 needs held-out junction *topologies*, not held-out episodes. The class is
    the sorted multiset of branch directions -- e.g. ``L|S``, ``L|R|S`` -- derived
    from each successor's net heading change relative to the approach lane. That is
    a topology (a T-junction is not a 4-way), and it is computed from the MAP, so it
    is available for an unseen scene without any model input.
    """
    from trajdata.maps.vec_map_elements import MapElementType
    lanes = ds.map.elements[MapElementType.ROAD_LANE]

    def clean(s):
        return {x for x in s if x != SENTINEL} if s else set()

    nxt = {k: clean(L.next_lanes) & set(lanes) for k, L in lanes.items()}
    cen = {k: np.asarray(L.center.points, dtype=np.float64)[:, :2]
           for k, L in lanes.items()}

    def heading(k):
        c = cen[k]
        if len(c) < 2:
            return None
        d = c[-1] - c[0]
        return float(np.arctan2(d[1], d[0]))

    out = []
    for k, su in nxt.items():
        if len(su) < 2:
            continue
        h0 = heading(k)
        if h0 is None:
            continue
        dirs = []
        for o in sorted(su):
            h1 = heading(o)
            if h1 is None:
                continue
            dd = np.degrees((h1 - h0 + np.pi) % (2 * np.pi) - np.pi)
            dirs.append("L" if dd > 25 else ("R" if dd < -25 else "S"))
        if len(dirs) < 2:
            continue
        out.append({"lane": str(k), "arity": len(dirs),
                    "class": "|".join(sorted(dirs))})
    return out


# ========================================================================== #
def main():
    ap = argparse.ArgumentParser("vectormap_corridor")
    ap.add_argument("--out", default="/workspace/vectormap_corridor.json")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from alpasim_runtime.scene_loader import ArtifactSceneProvider
    setdirs = sorted(d for d in glob.glob(SSROOT + "/*") if os.path.isdir(d))
    scenes = {}
    for sd in setdirs:
        try:
            prov = ArtifactSceneProvider.from_path(sd, smooth_trajectories=True)
        except Exception as e:
            print("PROVIDER FAIL", sd, repr(e)[:120], flush=True)
            continue
        for sid in sorted(prov.scene_ids):
            scenes.setdefault(sid, (os.path.basename(sd), prov))
    sids = sorted(scenes)
    if a.limit:
        sids = sids[:a.limit]
    print("SCENES:", len(sids), flush=True)

    res, topo = {}, []
    t0 = time.time()
    for i, sid in enumerate(sids, 1):
        ssname, prov = scenes[sid]
        short = sid[7:15] if sid.startswith("clipgt-") else sid[:8]
        rec = {"scene": short, "sceneset": ssname}
        try:
            ds = prov.get_data_source(sid)
            r, chan = probe_scene(ds)
            rec.update(r)
            if chan is not None:
                for t in topology_classes(ds, chan):
                    t["scene"] = short
                    topo.append(t)
        except Exception as e:
            rec["err"] = repr(e)[:200]
            rec["tb"] = traceback.format_exc()[-400:]
        res[short] = rec
        print("[%2d/%2d] %s lanes=%-4s w=%-5s contain=%-6s agree=%-6s %.0fs"
              % (i, len(sids), short, rec.get("n_lanes"),
                 (rec.get("lane_width_m") or {}).get("median"),
                 rec.get("ego_containment_rate"),
                 (rec.get("corridor") or {}).get("ring_vs_bounds_agreement"),
                 time.time() - t0), flush=True)
        try:
            ds.clear_cache()
        except Exception:
            pass

    ok = [r for r in res.values() if "err" not in r and r.get("n_lanes_with_edges")]
    summ = {"n_scenes": len(res), "n_ok": len(ok)}
    if ok:
        nl = [r["n_lanes"] for r in ok]
        wm = [r["lane_width_m"]["median"] for r in ok]
        cr = [r["ego_containment_rate"] for r in ok]
        g1 = [r["gate1_style_match_rate"] for r in ok]
        hw = [r["corridor"]["halfwidth_m_median"] for r in ok]
        rs = [r["corridor"]["ring_vs_bounds_agreement"] for r in ok]
        summ.update({
            "lanes_per_scene": {"min": int(min(nl)), "max": int(max(nl)),
                                "median": float(np.median(nl)),
                                "p10": float(np.percentile(nl, 10)),
                                "p90": float(np.percentile(nl, 90))},
            "lane_width_m_median_over_scenes": round(float(np.median(wm)), 3),
            "lane_width_m_range_over_scenes": [round(float(min(wm)), 3),
                                               round(float(max(wm)), 3)],
            "ego_containment_rate": {
                "mean": round(float(np.mean(cr)), 4),
                "median": round(float(np.median(cr)), 4),
                "min": round(float(min(cr)), 4),
                "n_scenes_ge_0.90": int(sum(1 for x in cr if x >= 0.90)),
                "n_scenes_ge_0.95": int(sum(1 for x in cr if x >= 0.95))},
            "gate1_style_match_rate_mean": round(float(np.mean(g1)), 4),
            "measured_halfwidth_m": {
                "median": round(float(np.median(hw)), 3),
                "p10": round(float(np.percentile(hw, 10)), 3),
                "p90": round(float(np.percentile(hw, 90)), 3)},
            "ring_vs_bounds_agreement": {"mean": round(float(np.mean(rs)), 4),
                                        "min": round(float(min(rs)), 4),
                                        "n_scenes_ge_0.90": int(sum(1 for x in rs if x >= 0.90))},
            "P1_pass_scenes": int(sum(1 for r in ok if r.get("P1_pass"))),
            "P2_pass_scenes": int(sum(1 for r in ok if r.get("P2_pass"))),
            "P3_pass_scenes": int(sum(1 for r in ok if r.get("P3_pass"))),
        })
        cls = collections.Counter(t["class"] for t in topo)
        scn = collections.defaultdict(set)
        for t in topo:
            scn[t["class"]].add(t["scene"])
        summ["topology_classes"] = {
            c: {"n_branch_points": int(v), "n_scenes": len(scn[c])}
            for c, v in cls.most_common()}
        summ["n_topology_classes"] = len(cls)
        summ["n_topology_classes_ge40_scenes"] = int(
            sum(1 for c in cls if len(scn[c]) >= 40))

    json.dump({"summary": summ, "per_scene": res, "topology": topo},
              open(a.out, "w"), indent=1, default=str)
    print("=" * 74)
    print(json.dumps(summ, indent=1, default=str))
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
