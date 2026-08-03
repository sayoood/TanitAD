#!/usr/bin/env python3
"""Does this NuRec scene's ego trajectory traverse a junction — and was there a CHOICE?

Two INDEPENDENT sources, deliberately not merged:

  * ``clipgt/`` (NVIDIA's own labels) — ``intersection_area`` polygons,
    ``wait_line`` ENTRY/EXIT, ``lane.lane_direction``, and the 202-pose
    ``egomotion_estimate``.  All already in one common clip frame, so no
    alignment is needed and nothing can be blamed on a bad transform.
  * ``map.xodr`` (the ASAM OpenDRIVE HD map shipped in the same USDZ) —
    junctions, junction-internal connecting roads, and their laneLinks.  The
    ego track for this source is re-derived from ``pose_record.json``
    geodetic coordinates through a Transverse-Mercator forward, i.e. it never
    touches the clipgt numbers.

They are then cross-checked against each other.  A component-vs-family
self-consistency control is mandatory in this programme, so the two sources
must agree on WHERE the junctions are before either is quoted.

Three controls run before any number is emitted (``--controls``):
  C1 positive — a point sampled on a junction-internal lane centreline must
     score "inside the junction" (distance 0).
  C2 negative — the same ego track translated 500 m must score zero
     traversals and a large minimum distance.
  C3 discrimination — a pose on a plain non-junction road must score a
     clearly positive distance.  If C1 and C3 are not separated, the metric
     cannot discriminate and nothing downstream is admissible.

Usage
-----
    python junction_probe.py --clipgt <dir> --xodr <map.xodr> \
        [--pose-record <pose_record.json>] [--out report.json] [--controls]

Only numpy is required for the OpenDRIVE half; ``pyarrow`` is required for the
clipgt half and is optional (``--no-clipgt`` skips it).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xodr_map import load_xodr, parse_geo_reference  # noqa: E402

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563


# --------------------------------------------------------------------------
# Transverse Mercator forward (Redfearn series) — used ONLY for the xodr half
# --------------------------------------------------------------------------
def tm_forward(lat_deg, lon_deg, lat0_deg, lon0_deg, k0=1.0, a=WGS84_A, f=WGS84_F):
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    phi = np.radians(np.asarray(lat_deg, float))
    lam = np.radians(np.asarray(lon_deg, float))
    phi0 = math.radians(lat0_deg)
    lam0 = math.radians(lon0_deg)

    def meridional_arc(p):
        n = f / (2 - f)
        A_ = a / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
        return A_ * (p
                     - (3 * n / 2 - 9 * n ** 3 / 16) * np.sin(2 * p)
                     + (15 * n ** 2 / 16 - 15 * n ** 4 / 32) * np.sin(4 * p)
                     - (35 * n ** 3 / 48) * np.sin(6 * p)
                     + (315 * n ** 4 / 512) * np.sin(8 * p))

    N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    T = np.tan(phi) ** 2
    C = ep2 * np.cos(phi) ** 2
    A_ = (lam - lam0) * np.cos(phi)
    M = meridional_arc(phi)
    M0 = meridional_arc(phi0)
    E = k0 * N * (A_ + (1 - T + C) * A_ ** 3 / 6
                  + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A_ ** 5 / 120)
    Nn = k0 * (M - M0 + N * np.tan(phi)
               * (A_ ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A_ ** 4 / 24
                  + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A_ ** 6 / 720))
    return E, Nn


# --------------------------------------------------------------------------
# small geometry helpers
# --------------------------------------------------------------------------
def point_seg_dist(P, A, B):
    """(N,2) points -> min distance to the polyline A->B segments (M,2)/(M,2)."""
    AB = B - A
    denom = np.einsum("ij,ij->i", AB, AB)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    AP = P[:, None, :] - A[None, :, :]
    t = np.einsum("nmj,mj->nm", AP, AB) / denom[None, :]
    t = np.clip(t, 0.0, 1.0)
    proj = A[None, :, :] + t[:, :, None] * AB[None, :, :]
    return np.linalg.norm(P[:, None, :] - proj, axis=2)


def polyline_dist(P, poly):
    if len(poly) < 2:
        return np.linalg.norm(P[:, None, :] - poly[None, :, :], axis=2).min(1)
    return point_seg_dist(P, poly[:-1], poly[1:]).min(1)


def point_in_poly(P, poly):
    """Ray casting; poly (M,2) assumed closed implicitly."""
    x, y = P[:, 0], P[:, 1]
    inside = np.zeros(len(P), bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > y) != (yj > y))
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xj - xi) * (y - yi) / np.where(yj - yi == 0, np.nan, yj - yi) + xi
        hit = cond & (x < xint)
        inside ^= np.nan_to_num(hit.astype(float), nan=0.0).astype(bool)
        j = i
    return inside


def signed_poly_dist(P, poly):
    """Negative inside, positive outside; magnitude = distance to the boundary."""
    closed = np.vstack([poly, poly[:1]])
    d = polyline_dist(P, closed)
    ins = point_in_poly(P, poly)
    return np.where(ins, -d, d)


def fit_rigid_2d(src, dst):
    """Least-squares rotation+translation mapping src -> dst (no scale)."""
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cd - R @ cs
    res = np.linalg.norm((src @ R.T + t) - dst, axis=1)
    return R, t, res


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------
# clipgt
# --------------------------------------------------------------------------
def read_clipgt_table(clipgt_dir, name):
    import pyarrow.parquet as pq
    p = Path(clipgt_dir) / f"{name}.parquet"
    if not p.exists():
        return None, None
    t = pq.read_table(p)
    keys = t.column("key").to_pylist() if "key" in t.schema.names else [None] * t.num_rows
    payload = t.column(name).to_pylist() if name in t.schema.names else t.to_pylist()
    return keys, payload


def xyz(seq):
    return np.array([[p["x"], p["y"], p["z"]] for p in seq], float)


def clipgt_analysis(clipgt_dir):
    keys, ego = read_clipgt_table(clipgt_dir, "egomotion_estimate")
    if ego is None:
        raise SystemExit("no egomotion_estimate.parquet")
    P = np.array([[d["location"]["x"], d["location"]["y"]] for d in ego], float)
    Q = np.array([[d["orientation"]["w"], d["orientation"]["x"],
                   d["orientation"]["y"], d["orientation"]["z"]] for d in ego], float)
    # yaw from quaternion (w,x,y,z)
    w, xq, yq, zq = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    yaw = np.degrees(np.arctan2(2 * (w * zq + xq * yq),
                                1 - 2 * (yq * yq + zq * zq)))
    ts = np.array([k["timestamp_micros"] for k in keys], float) if keys[0] else None

    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])

    out = {
        "n_poses": int(len(P)),
        "path_length_m": float(seg.sum()),
        "duration_s": float((ts[-1] - ts[0]) / 1e6) if ts is not None else None,
        "mean_speed_mps": float(seg.sum() / ((ts[-1] - ts[0]) / 1e6)) if ts is not None else None,
        "heading_total_change_deg": float(np.sum(np.abs(wrap_deg(np.diff(yaw))))),
        "heading_net_change_deg": float(wrap_deg(yaw[-1] - yaw[0])),
    }

    _, ia = read_clipgt_table(clipgt_dir, "intersection_area")
    areas = []
    if ia:
        for i, d in enumerate(ia):
            poly = xyz(d["location"])[:, :2]
            if len(poly) < 3:
                continue
            sd = signed_poly_dist(P, poly)
            inside = sd < 0
            areas.append({
                "idx": i,
                "category": d.get("category"),
                "is_complete": bool(d.get("is_complete")),
                "n_vertices": int(len(poly)),
                "centroid": [round(float(v), 3) for v in poly.mean(0)],
                "min_signed_dist_m": round(float(sd.min()), 4),
                "n_poses_inside": int(inside.sum()),
                "pose_span_inside": ([int(np.flatnonzero(inside)[0]),
                                      int(np.flatnonzero(inside)[-1])] if inside.any() else None),
                "arc_len_inside_m": (round(float(arc[np.flatnonzero(inside)[-1]]
                                                 - arc[np.flatnonzero(inside)[0]]), 3)
                                     if inside.sum() > 1 else 0.0),
                "heading_change_through_deg": (
                    round(float(wrap_deg(yaw[np.flatnonzero(inside)[-1]]
                                         - yaw[np.flatnonzero(inside)[0]])), 3)
                    if inside.sum() > 1 else None),
                "nearest_pose": int(np.abs(sd).argmin()),
            })
    if areas:
        allsd = np.stack([signed_poly_dist(P, xyz(ia[a["idx"]]["location"])[:, :2])
                          for a in areas])
        d_near = allsd.min(0)
        out["per_pose_signed_dist_to_nearest_intersection_m"] = [round(float(v), 3)
                                                                 for v in d_near]
        out["dist_to_nearest_intersection_m"] = {
            "min": round(float(d_near.min()), 4),
            "p25": round(float(np.percentile(d_near, 25)), 4),
            "median": round(float(np.median(d_near)), 4),
            "p75": round(float(np.percentile(d_near, 75)), 4),
            "max": round(float(d_near.max()), 4),
            "n_poses_inside_some_intersection": int((d_near < 0).sum()),
        }
    out["intersection_areas"] = areas

    # wait lines that the ego path actually crosses
    _, wl = read_clipgt_table(clipgt_dir, "wait_line")
    wls = []
    if wl:
        for d in wl:
            loc = d.get("location") or []
            if not loc or loc[0].get("x") is None:
                continue
            L = xyz(loc)[:, :2]
            dd = polyline_dist(P, L)
            wls.append({
                "category": d.get("category"),
                "subtype": d.get("intersection_subtype"),
                "is_implicit": bool(d.get("is_implicit")) if d.get("is_implicit") is not None else None,
                "min_dist_to_ego_m": round(float(dd.min()), 3),
                "at_pose": int(dd.argmin()),
                "arc_len_m": round(float(arc[int(dd.argmin())]), 2),
            })
        wls.sort(key=lambda r: r["min_dist_to_ego_m"])
    out["wait_lines_nearest_8"] = wls[:8]
    out["n_wait_lines"] = len(wls)
    out["n_wait_lines_within_3m_of_path"] = sum(1 for r in wls if r["min_dist_to_ego_m"] < 3.0)

    # lane snapping in the clip frame + lane_direction of the snapped lane
    _, lanes = read_clipgt_table(clipgt_dir, "lane")
    lane_recs = []
    if lanes:
        for i, d in enumerate(lanes):
            lr, rr = d.get("left_rail") or [], d.get("right_rail") or []
            if not lr or not rr:
                continue
            L, R = xyz(lr)[:, :2], xyz(rr)[:, :2]
            n = min(len(L), len(R))
            cen = 0.5 * (L[:n] + R[:n])
            lane_recs.append({"i": i, "dir": d.get("lane_direction"),
                              "cen": cen,
                              "half_w": 0.5 * np.linalg.norm(L[:n] - R[:n], axis=1).mean()})
    if lane_recs:
        D = np.stack([polyline_dist(P, r["cen"]) for r in lane_recs])   # (L, N)
        snap = D.argmin(0)
        snap_d = D.min(0)
        dirs = [lane_recs[j]["dir"] for j in snap]
        from collections import Counter
        out["clip_lane_snap"] = {
            "median_snap_dist_m": round(float(np.median(snap_d)), 3),
            "p90_snap_dist_m": round(float(np.percentile(snap_d, 90)), 3),
            "max_snap_dist_m": round(float(snap_d.max()), 3),
            "snapped_lane_direction_hist": dict(Counter(dirs)),
            "snapped_lane_direction_seq": _rle(dirs),
            "n_distinct_lanes": int(len(set(snap.tolist()))),
        }
        # how close does the ego come to a lane whose direction is a real turn?
        turn = {}
        for want in ("LEFT_TURN", "RIGHT_TURN", "U_TURN", "BRANCH_LEFT",
                     "BRANCH_RIGHT", "BRANCH_STRAIGHT", "STRAIGHT_TURN"):
            cands = [r for r in lane_recs if r["dir"] == want]
            if not cands:
                continue
            turn[want] = round(float(min(polyline_dist(P, r["cen"]).min()
                                         for r in cands)), 3)
        out["clip_min_dist_to_turn_lane_m"] = turn
    out["_ego_xy"] = P
    out["_yaw_deg"] = yaw
    out["_arc"] = arc
    out["_ts"] = ts
    return out


def _rle(seq):
    out = []
    for v in seq:
        if not out or out[-1][0] != v:
            out.append([v, 1])
        else:
            out[-1][1] += 1
    return [[v, n] for v, n in out]


# --------------------------------------------------------------------------
# xodr half
# --------------------------------------------------------------------------
def junction_surface(m, step=1.0):
    """Sample every driving lane of every junction-internal road.

    Returns {jid: (pts (K,2), halfwidth (K,))} and a road->jid index.
    """
    surf, road2j = {}, {}
    for r in m.roads.values():
        if not r.in_junction:
            continue
        road2j[r.rid] = r.junction
        pts, hw = [], []
        for lid in r.driving_lane_ids():
            p, w = r.sample_lane(lid, step=step)
            if len(p):
                pts.append(p); hw.append(0.5 * w)
        if pts:
            surf.setdefault(r.junction, [[], []])
            surf[r.junction][0].append(np.vstack(pts))
            surf[r.junction][1].append(np.concatenate(hw))
    return ({k: (np.vstack(v[0]), np.concatenate(v[1])) for k, v in surf.items()},
            road2j)


def all_lane_centerlines(m, step=1.0):
    out = {}
    for r in m.roads.values():
        for lid in r.driving_lane_ids():
            p, w = r.sample_lane(lid, step=step)
            if len(p) >= 2:
                out[f"{r.rid}:{lid}"] = p
    return out


def dist_to_junctions(P, surf):
    """(N,) clearance to the nearest junction drivable surface, 0 when inside,
    plus the argmin junction id per pose."""
    if not surf:
        return np.full(len(P), np.inf), [None] * len(P)
    keys = list(surf)
    D = np.zeros((len(keys), len(P)))
    for i, k in enumerate(keys):
        pts, hw = surf[k]
        d = np.linalg.norm(P[:, None, :] - pts[None, :, :], axis=2)
        j = d.argmin(1)
        D[i] = np.maximum(0.0, d[np.arange(len(P)), j] - hw[j])
    a = D.argmin(0)
    return D.min(0), [keys[i] for i in a]


def xodr_analysis(xodr_path, pose_record_path, step=1.0):
    m = load_xodr(xodr_path)
    geo = parse_geo_reference(m.header.get("geoReference", ""))
    lat0, lon0 = geo.get("lat_0"), geo.get("lon_0")

    rec = json.loads(Path(pose_record_path).read_text())
    records = rec["record"] if isinstance(rec, dict) else rec
    lats, lons = [], []
    for r in records:
        ll = r["alignment_world_pose"]["lat_lng_alt"]
        lats.append(ll["latitude"]); lons.append(ll["longitude"])
    E, N = tm_forward(np.array(lats), np.array(lons), lat0, lon0)
    Pm = np.stack([E, N], 1)

    surf, road2j = junction_surface(m, step=step)
    cls = all_lane_centerlines(m, step=step)
    keys = list(cls)
    D = np.stack([polyline_dist(Pm, cls[k]) for k in keys])
    snap_i = D.argmin(0)
    snap_d = D.min(0)
    snap_lane = [keys[i] for i in snap_i]
    snap_road = [k.split(":")[0] for k in snap_lane]

    dj, dj_which = dist_to_junctions(Pm, surf)

    return {
        "map": m, "geo": geo, "Pm": Pm, "surf": surf, "road2j": road2j,
        "centerlines": cls, "snap_lane": snap_lane, "snap_road": snap_road,
        "snap_dist": snap_d, "d_junction": dj, "d_junction_which": dj_which,
        "counts": {
            "roads": len(m.roads), "junctions": len(m.junctions),
            "junction_internal_roads": len(road2j),
            "driving_lanes": len(cls),
            "junctions_with_surface": len(surf),
        },
        "origin_check_m": float(np.linalg.norm(Pm[0])),
    }


def decision_options(m, incoming_road, incoming_lane, jid):
    """How many DISTINCT ways out of junction ``jid`` exist for this incoming lane?"""
    j = m.junctions.get(jid)
    if j is None:
        return None
    opts = []
    for c in j.connections:
        if c["incomingRoad"] != incoming_road:
            continue
        if incoming_lane is not None and c["laneLinks"]:
            if not any(a == incoming_lane for a, _ in c["laneLinks"]):
                continue
        opts.append(c["connectingRoad"])
    return sorted(set(opts))


# --------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------
def run_controls(xa):
    m, surf = xa["map"], xa["surf"]
    res = {}
    # C1 positive: a point on a junction-internal lane centreline
    jr = [r for r in m.roads.values() if r.in_junction and r.driving_lane_ids()]
    ok = []
    for r in jr[:20]:
        lid = r.driving_lane_ids()[0]
        c = r.lane_center_xy(r.length * 0.5, lid)
        if c is None:
            continue
        d, _ = dist_to_junctions(np.array([[c[0], c[1]]]), surf)
        ok.append(float(d[0]))
    res["C1_positive_on_junction_lane_center"] = {
        "n": len(ok), "max_dist_m": round(max(ok), 6) if ok else None,
        "PASS": bool(ok) and max(ok) < 1e-6,
    }
    # C3 discrimination: mid-point of the longest NON-junction road
    nonj = sorted((r for r in m.roads.values()
                   if not r.in_junction and r.driving_lane_ids()),
                  key=lambda r: -r.length)
    ds = []
    for r in nonj[:20]:
        lid = r.driving_lane_ids()[0]
        c = r.lane_center_xy(r.length * 0.5, lid)
        if c is None:
            continue
        d, _ = dist_to_junctions(np.array([[c[0], c[1]]]), surf)
        ds.append(float(d[0]))
    res["C3_discrimination_on_plain_road"] = {
        "n": len(ds), "median_dist_m": round(float(np.median(ds)), 3) if ds else None,
        "min_dist_m": round(min(ds), 3) if ds else None,
        "PASS": bool(ds) and float(np.median(ds)) > 5.0,
    }
    # C2 negative: the ego track shifted 500 m
    P = xa["Pm"] + np.array([500.0, 500.0])
    d, _ = dist_to_junctions(P, surf)
    res["C2_negative_ego_shifted_500m"] = {
        "min_dist_m": round(float(d.min()), 2),
        "n_inside": int((d <= 0).sum()),
        "PASS": bool(d.min() > 100.0 and (d <= 0).sum() == 0),
    }
    res["ALL_PASS"] = all(v.get("PASS") for v in res.values() if isinstance(v, dict))
    return res


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clipgt", default=None)
    ap.add_argument("--xodr", default=None)
    ap.add_argument("--pose-record", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--scene-id", default=None)
    a = ap.parse_args()

    rep = {"tool": "junction_probe.py", "evidence_class": "MEASURED",
           "scene_id": a.scene_id, "clipgt": a.clipgt, "xodr": a.xodr}

    ca = None
    if a.clipgt:
        ca = clipgt_analysis(a.clipgt)
        pub = {k: v for k, v in ca.items() if not k.startswith("_")}
        rep["clipgt_source"] = pub

    xa = None
    if a.xodr and a.pose_record:
        xa = xodr_analysis(a.xodr, a.pose_record, step=a.step)
        if a.controls:
            rep["controls"] = run_controls(xa)
            if not rep["controls"]["ALL_PASS"]:
                rep["ADMISSIBLE"] = False
        dj = xa["d_junction"]
        rep["xodr_source"] = {
            "counts": xa["counts"],
            "geo_lat0": xa["geo"].get("lat_0"), "geo_lon0": xa["geo"].get("lon_0"),
            "origin_residual_m": round(xa["origin_check_m"], 6),
            "n_pose_record": int(len(xa["Pm"])),
            "snap_dist_m": {"median": round(float(np.median(xa["snap_dist"])), 3),
                            "p90": round(float(np.percentile(xa["snap_dist"], 90)), 3),
                            "max": round(float(xa["snap_dist"].max()), 3)},
            "dist_to_junction_m_full_record": {
                "min": round(float(dj.min()), 3),
                "median": round(float(np.median(dj)), 3),
                "max": round(float(dj.max()), 3),
                "n_inside": int((dj <= 0).sum()),
            },
            "road_sequence": [v for v, _ in _rle(xa["snap_road"])],
            "junction_road_runs": _junction_runs(xa),
        }

    # ---- cross-source: fit clip frame -> map frame using the two ego tracks
    if ca is not None and xa is not None:
        Pc = ca["_ego_xy"]
        Pm = xa["Pm"]
        n = min(len(Pc), len(Pm))
        # the clip window is the LAST n of the pose record iff timestamps align;
        # try every offset and keep the best rigid fit.
        best = None
        for off in range(0, len(Pm) - n + 1):
            R, t, res = fit_rigid_2d(Pc[:n], Pm[off:off + n])
            rms = float(np.sqrt((res ** 2).mean()))
            if best is None or rms < best[0]:
                best = (rms, off, R, t, res)
        rms, off, R, t, res = best
        rep["cross_source_alignment"] = {
            "method": "rigid 2D SVD fit, clipgt egomotion -> geodetic map frame",
            "n": int(n), "pose_record_offset": int(off),
            "rms_residual_m": round(rms, 4),
            "max_residual_m": round(float(res.max()), 4),
            "rotation_deg": round(float(np.degrees(math.atan2(R[1, 0], R[0, 0]))), 4),
            "translation_m": [round(float(v), 3) for v in t],
        }
        if rms < 2.0:
            # transform the clipgt intersection polygons into the map frame and
            # ask the xodr where they land
            import pyarrow.parquet as pq  # noqa: F401
            _, ia = read_clipgt_table(a.clipgt, "intersection_area")
            agree = []
            for i, d in enumerate(ia or []):
                poly = xyz(d["location"])[:, :2]
                if len(poly) < 3:
                    continue
                pm = poly @ R.T + t
                dd, which = dist_to_junctions(pm, xa["surf"])
                agree.append({
                    "clipgt_idx": i, "category": d.get("category"),
                    "median_dist_of_polygon_to_nearest_xodr_junction_m": round(float(np.median(dd)), 3),
                    "frac_vertices_on_a_junction_surface": round(float((dd <= 0.5).mean()), 3),
                    "nearest_xodr_junction_id": max(set(which), key=which.count),
                })
            rep["cross_source_agreement"] = agree
            # the ego in the map frame, restricted to the clip window
            Pw = Pc @ R.T + t
            dj_clip, which_clip = dist_to_junctions(Pw, xa["surf"])
            rep["xodr_source"]["dist_to_junction_m_CLIP_WINDOW"] = {
                "n_poses": int(len(Pw)),
                "min": round(float(dj_clip.min()), 3),
                "p25": round(float(np.percentile(dj_clip, 25)), 3),
                "median": round(float(np.median(dj_clip)), 3),
                "p75": round(float(np.percentile(dj_clip, 75)), 3),
                "max": round(float(dj_clip.max()), 3),
                "n_poses_inside_a_junction": int((dj_clip <= 0).sum()),
            }
            rep["xodr_source"]["per_pose_dist_to_junction_m_CLIP_WINDOW"] = [
                round(float(v), 3) for v in dj_clip]
            rep["xodr_source"]["junctions_touched_in_clip_window"] = sorted(
                {w for w, dd in zip(which_clip, dj_clip) if dd <= 0.0})
            # decision analysis
            rep["decisions"] = _decision_analysis(xa, Pw, ca["_yaw_deg"])

    txt = json.dumps(rep, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(txt)
        print(f"wrote {a.out} ({len(txt)} bytes)")
    else:
        print(txt)
    return rep


def _junction_runs(xa):
    """Maximal runs of consecutive poses snapped to a junction-internal road."""
    road2j = xa["road2j"]
    runs, cur = [], None
    for i, rid in enumerate(xa["snap_road"]):
        j = road2j.get(rid)
        if j is None:
            if cur:
                runs.append(cur); cur = None
            continue
        if cur and cur["junction"] == j and i == cur["end"] + 1:
            cur["end"] = i
        else:
            if cur:
                runs.append(cur)
            cur = {"junction": j, "road": rid, "start": i, "end": i}
    if cur:
        runs.append(cur)
    return runs


def _resolve_incoming(m, jid, taken_road, taken_lane):
    """Which (incomingRoad, incomingLane) fed the connecting road the ego took?

    Resolved from the junction's own ``<connection>`` table, NOT by scanning
    backwards over snapped poses.  The backwards scan is wrong: a connector as
    short as 1.02 m (road 189 in scene 00040136) never wins a nearest-lane
    snap, so the scan reported ``incomingRoad=20`` for junction 239 whose table
    only knows ``incomingRoad=189`` — and the option count came out 0 instead
    of 1.  A zero there would have read as "no continuation exists", which is a
    different and false claim.
    """
    j = m.junctions.get(jid)
    if j is None:
        return None, None, "junction-not-in-map"
    cands = [c for c in j.connections if c["connectingRoad"] == taken_road]
    if not cands:
        return None, None, "connecting-road-not-in-table"
    for c in cands:
        for a, b in c["laneLinks"]:
            if taken_lane is None or b == taken_lane:
                return c["incomingRoad"], a, "ok"
    c = cands[0]
    return c["incomingRoad"], (c["laneLinks"][0][0] if c["laneLinks"] else None), "lane-fallback"


def _decision_analysis(xa, Pw, yaw):
    """For every junction the ego enters in the clip window, how many ways out?"""
    m, surf, road2j = xa["map"], xa["surf"], xa["road2j"]
    cls = xa["centerlines"]
    keys = list(cls)
    D = np.stack([polyline_dist(Pw, cls[k]) for k in keys])
    lane = [keys[i] for i in D.argmin(0)]
    road = [k.split(":")[0] for k in lane]
    lid = [int(k.split(":")[1]) for k in lane]
    dj, which = dist_to_junctions(Pw, surf)

    out = []
    inside = dj <= 0.0
    i = 0
    while i < len(inside):
        if not inside[i]:
            i += 1
            continue
        j0 = i
        while i + 1 < len(inside) and inside[i + 1] and which[i + 1] == which[j0]:
            i += 1
        j1 = i
        jid = which[j0]
        # the connecting road the ego actually took = the most common
        # junction-internal snapped road over the traversal
        taken = [road[k] for k in range(j0, j1 + 1) if road2j.get(road[k]) == jid]
        taken_road = max(set(taken), key=taken.count) if taken else None
        taken_lane = None
        if taken_road is not None:
            tl = [lid[k] for k in range(j0, j1 + 1) if road[k] == taken_road]
            taken_lane = max(set(tl), key=tl.count) if tl else None
        inc_r, inc_l, how = _resolve_incoming(m, jid, taken_road, taken_lane)
        opts = decision_options(m, inc_r, inc_l, jid) if inc_r else None
        opts_any = decision_options(m, inc_r, None, jid) if inc_r else None
        out.append({
            "junction_id": jid,
            "pose_span": [int(j0), int(j1)],
            "n_poses_inside": int(j1 - j0 + 1),
            "connecting_road_taken": taken_road, "connecting_lane_taken": taken_lane,
            "incoming_road": inc_r, "incoming_lane": inc_l, "resolved_by": how,
            "connecting_roads_for_this_lane": opts,
            "n_options_for_this_lane": (len(opts) if opts is not None else None),
            "connecting_roads_for_this_road_any_lane": opts_any,
            "n_options_for_this_road": (len(opts_any) if opts_any is not None else None),
            "heading_change_through_deg": round(float(wrap_deg(yaw[j1] - yaw[j0])), 3),
            "IS_A_LANE_LEVEL_DECISION": bool(opts is not None and len(opts) >= 2),
            "IS_A_ROAD_LEVEL_DECISION": bool(opts_any is not None and len(opts_any) >= 2),
        })
        i += 1
    return out


if __name__ == "__main__":
    main()
