#!/usr/bin/env python3
"""
xodr_probe.py -- probe the OpenDRIVE (map.xodr) inside a NVIDIA NuRec .usdz scene.

TanitAD / STREAM B, 2026-08-02.

Why this exists
---------------
The programme concluded over five probes that PhysicalAI-AV carries no map, no lane
graph, no junction annotation, no traffic-light feature and no route/goal signal, and
that the strategic brain's topology "must come from AlpaSim or an external corpus".
The NuRec sample-set .usdz contains a `map.xodr`. This script answers, with counts:

  1. what is IN the OpenDRIVE (roads / lanes / junctions / signals / speeds / geometry)
  2. can a LANE GRAPH be derived (nodes, edges, components, sources, sinks)
  3. can a ROUTE / GOAL signal be derived (snap the ego rig trajectory onto lanes)
  4. is it GEOREFERENCED (proj4 -> WGS84 lat/lon, i.e. OSM map-matchable)
  5. does the map cover the whole 20 s scene, or only a fragment

Design constraints (Thor edge venv):
  * stdlib only + numpy. No lxml, no pyproj, no shapely, no pandas on the edge venv.
  * never unzip the 1.96 GB usdz -- read single members with zipfile.
  * every derived number is recomputed here, not copied from prose.

Usage
-----
  python3 xodr_probe.py --usdz /path/to/scene.usdz --out /path/to/workdir
  python3 xodr_probe.py --xodr /path/to/map.xodr  --out /path/to/workdir   # already extracted

Outputs (in --out):
  map.xodr, rig_trajectories.json, data_info.json, pose_record.json   (extracted members)
  xodr_probe_output.json                                              (every count below)
  lane_centerlines.json                                               (sampled driving-lane polylines)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

WANT_MEMBERS = [
    "map.xodr",
    "data_info.json",
    "rig_trajectories.json",
    "pose_record.json",
    "metadata.yaml",
]

# --------------------------------------------------------------------------------------
# 0. extraction
# --------------------------------------------------------------------------------------


def extract_members(usdz, outdir, members=WANT_MEMBERS):
    """Read named members out of the usdz WITHOUT unpacking the whole archive."""
    got = {}
    with zipfile.ZipFile(usdz) as z:
        names = set(z.namelist())
        for m in members:
            if m not in names:
                got[m] = {"present": False}
                continue
            blob = z.read(m)
            dst = os.path.join(outdir, m.replace("/", "_"))
            with open(dst, "wb") as fh:
                fh.write(blob)
            got[m] = {
                "present": True,
                "bytes": len(blob),
                "md5": hashlib.md5(blob).hexdigest(),
                "path": dst,
            }
    return got


# --------------------------------------------------------------------------------------
# 1. georeferencing: proj4 parse + Transverse Mercator forward/inverse (Redfearn series)
# --------------------------------------------------------------------------------------

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563


def parse_proj4(s):
    """Parse a proj4 string into a dict.

    NOTE: NuRec's string contains the malformed token '+=alt_0=0' (a vendor typo for
    '+alt_0=0'). A strict proj4 parser rejects it. We record it rather than hide it.
    """
    out, malformed = {}, []
    for tok in s.split():
        if not tok.startswith("+"):
            malformed.append(tok)
            continue
        body = tok[1:]
        if body.startswith("="):  # the '+=alt_0=0' case
            malformed.append(tok)
            body = body[1:]
        if "=" in body:
            k, v = body.split("=", 1)
            out[k] = v
        else:
            out[body] = True
    return out, malformed


def _meridional_arc(phi, a, e2):
    return a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * phi
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * phi)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * phi)
        - (35 * e2**3 / 3072) * math.sin(6 * phi)
    )


def tm_forward(lat_deg, lon_deg, lat0_deg, lon0_deg, k0=1.0, a=WGS84_A, f=WGS84_F):
    """WGS84 lat/lon -> transverse-Mercator local metres (false E/N = 0)."""
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    phi, lam = math.radians(lat_deg), math.radians(lon_deg)
    phi0, lam0 = math.radians(lat0_deg), math.radians(lon0_deg)
    N = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    T = math.tan(phi) ** 2
    C = ep2 * math.cos(phi) ** 2
    A = (lam - lam0) * math.cos(phi)
    E = k0 * N * (A + (1 - T + C) * A**3 / 6 + (5 - 18 * T + T**2 + 72 * C - 58 * ep2) * A**5 / 120)
    Nn = k0 * (
        _meridional_arc(phi, a, e2)
        - _meridional_arc(phi0, a, e2)
        + N
        * math.tan(phi)
        * (
            A**2 / 2
            + (5 - T + 9 * C + 4 * C**2) * A**4 / 24
            + (61 - 58 * T + T**2 + 600 * C - 330 * ep2) * A**6 / 720
        )
    )
    return E, Nn


def tm_inverse(E, Nn, lat0_deg, lon0_deg, k0=1.0, a=WGS84_A, f=WGS84_F):
    """Transverse-Mercator local metres -> WGS84 lat/lon."""
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    phi0, lam0 = math.radians(lat0_deg), math.radians(lon0_deg)
    M = _meridional_arc(phi0, a, e2) + Nn / k0
    mu = M / (a * (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    phi1 = (
        mu
        + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
        + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
        + (151 * e1**3 / 96) * math.sin(6 * mu)
        + (1097 * e1**4 / 512) * math.sin(8 * mu)
    )
    C1 = ep2 * math.cos(phi1) ** 2
    T1 = math.tan(phi1) ** 2
    N1 = a / math.sqrt(1 - e2 * math.sin(phi1) ** 2)
    R1 = a * (1 - e2) / (1 - e2 * math.sin(phi1) ** 2) ** 1.5
    D = E / (N1 * k0)
    phi = phi1 - (N1 * math.tan(phi1) / R1) * (
        D**2 / 2
        - (5 + 3 * T1 + 10 * C1 - 4 * C1**2 - 9 * ep2) * D**4 / 24
        + (61 + 90 * T1 + 298 * C1 + 45 * T1**2 - 252 * ep2 - 3 * C1**2) * D**6 / 720
    )
    lam = lam0 + (
        D
        - (1 + 2 * T1 + C1) * D**3 / 6
        + (5 - 2 * C1 + 28 * T1 - 3 * C1**2 + 8 * ep2 + 24 * T1**2) * D**5 / 120
    ) / math.cos(phi1)
    return math.degrees(phi), math.degrees(lam)


# --------------------------------------------------------------------------------------
# 2. OpenDRIVE geometry sampling (line / arc / spiral / poly3 / paramPoly3)
# --------------------------------------------------------------------------------------


def _fresnel_spiral(x0, y0, hdg0, length, curv_start, curv_end, n):
    """Numeric integration of a clothoid (curvature linear in s)."""
    xs, ys, hs, ss = [], [], [], []
    steps = max(n, 8)
    ds = length / steps
    x, y, h = x0, y0, hdg0
    for i in range(steps + 1):
        s = i * ds
        xs.append(x)
        ys.append(y)
        hs.append(h)
        ss.append(s)
        if i == steps:
            break
        k = curv_start + (curv_end - curv_start) * (s / length if length else 0.0)
        # midpoint integration
        h_mid = h + 0.5 * k * ds
        x += ds * math.cos(h_mid)
        y += ds * math.sin(h_mid)
        h += k * ds
    return ss, xs, ys, hs


def sample_geometry(geom_el, step=1.0):
    """Return (s_local[], x[], y[], hdg[]) sampled along one <geometry>."""
    x0 = float(geom_el.get("x"))
    y0 = float(geom_el.get("y"))
    hdg0 = float(geom_el.get("hdg"))
    L = float(geom_el.get("length"))
    n = max(2, int(math.ceil(L / step)))

    child = None
    for c in geom_el:
        if c.tag in ("line", "arc", "spiral", "poly3", "paramPoly3"):
            child = c
            break
    kind = child.tag if child is not None else "line"

    if kind == "line":
        ss = [L * i / n for i in range(n + 1)]
        xs = [x0 + s * math.cos(hdg0) for s in ss]
        ys = [y0 + s * math.sin(hdg0) for s in ss]
        hs = [hdg0] * (n + 1)
        return kind, ss, xs, ys, hs

    if kind == "arc":
        k = float(child.get("curvature"))
        ss = [L * i / n for i in range(n + 1)]
        if abs(k) < 1e-12:
            xs = [x0 + s * math.cos(hdg0) for s in ss]
            ys = [y0 + s * math.sin(hdg0) for s in ss]
            hs = [hdg0] * (n + 1)
        else:
            xs = [x0 + (math.sin(hdg0 + k * s) - math.sin(hdg0)) / k for s in ss]
            ys = [y0 - (math.cos(hdg0 + k * s) - math.cos(hdg0)) / k for s in ss]
            hs = [hdg0 + k * s for s in ss]
        return kind, ss, xs, ys, hs

    if kind == "spiral":
        cs = float(child.get("curvStart"))
        ce = float(child.get("curvEnd"))
        ss, xs, ys, hs = _fresnel_spiral(x0, y0, hdg0, L, cs, ce, n)
        return kind, ss, xs, ys, hs

    if kind == "poly3":
        a = float(child.get("a")); b = float(child.get("b"))
        c = float(child.get("c")); d = float(child.get("d"))
        ss = [L * i / n for i in range(n + 1)]
        xs, ys, hs = [], [], []
        for u in ss:
            v = a + b * u + c * u**2 + d * u**3
            dv = b + 2 * c * u + 3 * d * u**2
            xs.append(x0 + u * math.cos(hdg0) - v * math.sin(hdg0))
            ys.append(y0 + u * math.sin(hdg0) + v * math.cos(hdg0))
            hs.append(hdg0 + math.atan(dv))
        return kind, ss, xs, ys, hs

    # paramPoly3
    aU = float(child.get("aU")); bU = float(child.get("bU"))
    cU = float(child.get("cU")); dU = float(child.get("dU"))
    aV = float(child.get("aV")); bV = float(child.get("bV"))
    cV = float(child.get("cV")); dV = float(child.get("dV"))
    prange = child.get("pRange", "normalized")
    pmax = L if prange == "arcLength" else 1.0
    ps = [pmax * i / n for i in range(n + 1)]
    xs, ys, hs, ss = [], [], [], []
    for i, p in enumerate(ps):
        u = aU + bU * p + cU * p**2 + dU * p**3
        v = aV + bV * p + cV * p**2 + dV * p**3
        du = bU + 2 * cU * p + 3 * dU * p**2
        dv = bV + 2 * cV * p + 3 * dV * p**2
        xs.append(x0 + u * math.cos(hdg0) - v * math.sin(hdg0))
        ys.append(y0 + u * math.sin(hdg0) + v * math.cos(hdg0))
        hs.append(hdg0 + math.atan2(dv, du))
        ss.append(L * i / n)
    return kind, ss, xs, ys, hs


def poly3_eval(el, ds):
    a = float(el.get("a", 0)); b = float(el.get("b", 0))
    c = float(el.get("c", 0)); d = float(el.get("d", 0))
    return a + b * ds + c * ds**2 + d * ds**3


# --------------------------------------------------------------------------------------
# 3. OpenDRIVE parse
# --------------------------------------------------------------------------------------


def parse_opendrive(path, sample_step=1.0):
    root = ET.parse(path).getroot()
    rep = {"root_tag": root.tag}

    # ---- header -------------------------------------------------------------------
    hdr = root.find("header")
    H = {}
    if hdr is not None:
        H = dict(hdr.attrib)
        geo = hdr.find("geoReference")
        if geo is not None:
            raw = (geo.text or "").strip()
            H["geoReference_raw"] = raw
            proj, malformed = parse_proj4(raw)
            H["geoReference_parsed"] = proj
            H["geoReference_malformed_tokens"] = malformed
        else:
            H["geoReference_raw"] = None
        for oe in hdr.findall("offset"):
            H["offset"] = dict(oe.attrib)
    rep["header"] = H

    # ---- roads --------------------------------------------------------------------
    roads = root.findall("road")
    rep["counts"] = {"road": len(roads), "junction": len(root.findall("junction"))}

    geom_kinds = Counter()
    lane_types = Counter()
    lane_types_by_side = Counter()
    roadmark_types = Counter()
    speed_road = Counter()
    speed_lane = Counter()
    road_types = Counter()
    n_lane_sections = 0
    n_left = n_right = n_center = 0
    n_signals = 0
    n_objects = 0
    n_lane_links = 0
    total_road_len = 0.0
    road_name_vals = []
    road_len_list = []
    roads_in_junction = 0
    link_pred_kind = Counter()
    link_succ_kind = Counter()
    has_elev = 0
    has_lateral = 0
    multi_lanesection = 0

    signals_dump = []
    objects_dump = []

    xs_all, ys_all = [], []
    centerlines = {}   # (road_id, lane_id) -> [[x,y], ...]
    road_ref = {}      # road_id -> sampled reference line
    road_meta = {}

    for r in roads:
        rid = r.get("id")
        rname = r.get("name", "")
        rlen = float(r.get("length", 0.0))
        rjunc = r.get("junction", "-1")
        total_road_len += rlen
        road_len_list.append(rlen)
        road_name_vals.append(rname)
        if rjunc not in ("-1", "", None):
            roads_in_junction += 1

        lk = r.find("link")
        pred_el = succ_el = None
        if lk is not None:
            pred_el = lk.find("predecessor")
            succ_el = lk.find("successor")
            link_pred_kind[pred_el.get("elementType") if pred_el is not None else "none"] += 1
            link_succ_kind[succ_el.get("elementType") if succ_el is not None else "none"] += 1
        else:
            link_pred_kind["none"] += 1
            link_succ_kind["none"] += 1

        for t in r.findall("type"):
            road_types[t.get("type", "?")] += 1
            for sp in t.findall("speed"):
                speed_road[(sp.get("max", "?"), sp.get("unit", "?"))] += 1

        if r.find("elevationProfile") is not None and len(r.find("elevationProfile")):
            has_elev += 1
        if r.find("lateralProfile") is not None and len(r.find("lateralProfile")):
            has_lateral += 1

        # --- reference line -------------------------------------------------------
        ref_s, ref_x, ref_y, ref_h = [], [], [], []
        s_base = 0.0
        pv = r.find("planView")
        if pv is not None:
            for g in pv.findall("geometry"):
                kind, ss, gx, gy, gh = sample_geometry(g, step=sample_step)
                geom_kinds[kind] += 1
                gs0 = float(g.get("s", 0.0))
                for i in range(len(ss)):
                    if ref_s and abs((gs0 + ss[i]) - ref_s[-1]) < 1e-9:
                        continue
                    ref_s.append(gs0 + ss[i])
                    ref_x.append(gx[i])
                    ref_y.append(gy[i])
                    ref_h.append(gh[i])
                s_base = gs0 + (ss[-1] if ss else 0.0)
        xs_all.extend(ref_x)
        ys_all.extend(ref_y)
        road_ref[rid] = (ref_s, ref_x, ref_y, ref_h)

        # --- lanes ---------------------------------------------------------------
        lanes_el = r.find("lanes")
        lane_offsets = []
        sections = []
        if lanes_el is not None:
            lane_offsets = lanes_el.findall("laneOffset")
            sections = lanes_el.findall("laneSection")
            n_lane_sections += len(sections)
            if len(sections) > 1:
                multi_lanesection += 1

        def lane_offset_at(s):
            if not lane_offsets:
                return 0.0
            best = lane_offsets[0]
            for lo in lane_offsets:
                if float(lo.get("s", 0)) <= s + 1e-9:
                    best = lo
            return poly3_eval(best, s - float(best.get("s", 0)))

        road_driving_lanes = []
        for si, sec in enumerate(sections):
            s_sec = float(sec.get("s", 0.0))
            s_end = float(sections[si + 1].get("s")) if si + 1 < len(sections) else rlen
            for side in ("left", "center", "right"):
                sd = sec.find(side)
                if sd is None:
                    continue
                if side == "left":
                    n_left += 1
                elif side == "right":
                    n_right += 1
                else:
                    n_center += 1
                lane_els = sd.findall("lane")
                # order matters: inner -> outer
                if side == "left":
                    lane_els = sorted(lane_els, key=lambda e: int(e.get("id")))
                elif side == "right":
                    lane_els = sorted(lane_els, key=lambda e: -int(e.get("id")))
                for le in lane_els:
                    lt = le.get("type", "?")
                    lane_types[lt] += 1
                    lane_types_by_side[(side, lt)] += 1
                    for rm in le.findall("roadMark"):
                        roadmark_types[rm.get("type", "?")] += 1
                    for sp in le.findall("speed"):
                        speed_lane[(sp.get("max", "?"), sp.get("unit", "?"))] += 1
                    ll = le.find("link")
                    if ll is not None:
                        n_lane_links += len(ll.findall("predecessor")) + len(ll.findall("successor"))
                    if lt == "driving" and side in ("left", "right"):
                        road_driving_lanes.append((side, le))

            # ---- lane centerlines for this section --------------------------------
            for side in ("left", "right"):
                sd = sec.find(side)
                if sd is None:
                    continue
                lane_els = sd.findall("lane")
                lane_els = sorted(lane_els, key=lambda e: int(e.get("id")) if side == "left" else -int(e.get("id")))
                cum = {}
                for le in lane_els:
                    lid = int(le.get("id"))
                    pts = []
                    for i, s in enumerate(ref_s):
                        if s < s_sec - 1e-9 or s > s_end + 1e-9:
                            continue
                        ds = s - s_sec
                        # cumulative width from the reference line outwards
                        inner = 0.0
                        for le2 in lane_els:
                            lid2 = int(le2.get("id"))
                            if (side == "left" and lid2 < lid) or (side == "right" and lid2 > lid):
                                w2 = le2.find("width")
                                if w2 is not None:
                                    inner += poly3_eval(w2, ds - float(w2.get("sOffset", 0)))
                        w = le.find("width")
                        wv = poly3_eval(w, ds - float(w.get("sOffset", 0))) if w is not None else 0.0
                        t = lane_offset_at(s) + (inner + wv / 2.0) * (1 if side == "left" else -1)
                        h = ref_h[i]
                        pts.append([ref_x[i] - t * math.sin(h), ref_y[i] + t * math.cos(h)])
                    cum[lid] = pts
                    if le.get("type") == "driving" and pts:
                        centerlines[f"{rid}:{lid}"] = pts

        # --- signals / objects ----------------------------------------------------
        sg = r.find("signals")
        if sg is not None:
            for s_el in sg.findall("signal"):
                n_signals += 1
                d = dict(s_el.attrib)
                d["_road"] = rid
                ud = s_el.find("userData")
                if ud is not None:
                    d["_userData"] = dict(ud.attrib)
                    if (ud.text or "").strip():
                        d["_userData_text"] = (ud.text or "").strip()[:300]
                signals_dump.append(d)
            n_signals_ref = len(sg.findall("signalReference"))
        ob = r.find("objects")
        if ob is not None:
            for o_el in ob.findall("object"):
                n_objects += 1
                objects_dump.append(dict(o_el.attrib))

        road_meta[rid] = {
            "name": rname,
            "length": rlen,
            "junction": rjunc,
            "n_driving_lanes": len(road_driving_lanes),
            "pred": dict(pred_el.attrib) if pred_el is not None else None,
            "succ": dict(succ_el.attrib) if succ_el is not None else None,
        }

    # ---- junctions ------------------------------------------------------------------
    junc_conn = 0
    junc_lanelink = 0
    junc_types = Counter()
    junc_dump = []
    for j in root.findall("junction"):
        jt = j.get("type", "(default)")
        junc_types[jt] += 1
        conns = j.findall("connection")
        junc_conn += len(conns)
        nll = sum(len(c.findall("laneLink")) for c in conns)
        junc_lanelink += nll
        junc_dump.append(
            {"id": j.get("id"), "name": j.get("name", ""), "type": jt,
             "n_connections": len(conns), "n_laneLinks": nll}
        )

    rep["counts"].update({
        "lane_section": n_lane_sections,
        "roads_with_multiple_lanesections": multi_lanesection,
        "lane_total": sum(lane_types.values()),
        "lane_side_left_blocks": n_left,
        "lane_side_right_blocks": n_right,
        "lane_side_center_blocks": n_center,
        "lane_link_entries": n_lane_links,
        "signal": n_signals,
        "object": n_objects,
        "junction_connection": junc_conn,
        "junction_laneLink": junc_lanelink,
        "roads_inside_a_junction": roads_in_junction,
        "roads_with_elevationProfile": has_elev,
        "roads_with_lateralProfile": has_lateral,
        # traffic-light machinery lives in <controller>/<junctionGroup>/<signalReference>.
        # Probe all three by name before claiming absence.
        "controller": len(root.findall("controller")),
        "junctionGroup": len(root.findall("junctionGroup")),
        "station": len(root.findall("station")),
        "signalReference": sum(len(s.findall("signalReference")) for s in root.iter("signals")),
        "signal_dynamic_yes": sum(1 for s in signals_dump if s.get("dynamic") == "yes"),
        "signal_dynamic_no": sum(1 for s in signals_dump if s.get("dynamic") == "no"),
    })
    rep["top_level_tags"] = dict(Counter(c.tag for c in root))
    rep["signal_names"] = dict(Counter(s.get("name", "?") for s in signals_dump))
    rep["geometry_primitives"] = dict(geom_kinds)
    rep["lane_types"] = dict(lane_types)
    rep["lane_types_by_side"] = {f"{k[0]}/{k[1]}": v for k, v in lane_types_by_side.items()}
    rep["roadmark_types"] = dict(roadmark_types)
    rep["road_types"] = dict(road_types)
    rep["speed_limits_road_level"] = {f"{k[0]} {k[1]}": v for k, v in speed_road.items()}
    rep["speed_limits_lane_level"] = {f"{k[0]} {k[1]}": v for k, v in speed_lane.items()}
    rep["road_link_predecessor_kinds"] = dict(link_pred_kind)
    rep["road_link_successor_kinds"] = dict(link_succ_kind)
    rep["junction_types"] = dict(junc_types)
    rep["junctions"] = junc_dump
    rep["signals"] = signals_dump
    rep["objects_sample"] = objects_dump[:20]
    rep["road_length_m"] = {
        "total": total_road_len,
        "min": min(road_len_list) if road_len_list else None,
        "max": max(road_len_list) if road_len_list else None,
        "mean": (total_road_len / len(road_len_list)) if road_len_list else None,
    }

    # road "name" field -- is it an external id?
    numeric = [v for v in road_name_vals if re.fullmatch(r"\d+", v or "")]
    negative = [v for v in road_name_vals if re.fullmatch(r"-\d+", v or "")]
    other = [v for v in road_name_vals if v not in numeric and v not in negative]
    neg_ids = {rid for rid, m in road_meta.items() if re.fullmatch(r"-\d+", m["name"] or "")}
    junc_ids = {rid for rid, m in road_meta.items() if m["junction"] not in ("-1", "", None)}
    rep["road_name_field"] = {
        "n_roads": len(road_name_vals),
        "n_positive_integer": len(numeric),
        "n_negative_integer": len(negative),
        "n_non_integer": len(other),
        "n_empty": sum(1 for v in road_name_vals if not v),
        "n_distinct": len(set(road_name_vals)),
        "positive_min": min(int(v) for v in numeric) if numeric else None,
        "positive_max": max(int(v) for v in numeric) if numeric else None,
        "sample_positive": numeric[:10],
        "sample_negative": negative[:10],
        "non_integer_values": other[:20],
        "negative_named_roads_that_are_junction_internal": len(neg_ids & junc_ids),
        "junction_internal_roads": len(junc_ids),
        "note": "no street-name strings anywhere; the field is an integer feature id only",
    }

    # ---- map extent ------------------------------------------------------------------
    if xs_all:
        rep["map_extent_local_m"] = {
            "x_min": min(xs_all), "x_max": max(xs_all),
            "y_min": min(ys_all), "y_max": max(ys_all),
            "width_x": max(xs_all) - min(xs_all),
            "height_y": max(ys_all) - min(ys_all),
            "n_samples": len(xs_all),
        }
    rep["_centerlines"] = centerlines
    rep["_road_meta"] = road_meta
    rep["_root"] = root
    return rep


# --------------------------------------------------------------------------------------
# 4. lane graph
# --------------------------------------------------------------------------------------


def build_lane_graph(root):
    """Directed graph over driving lanes: node = 'roadId:laneId'."""
    roads = {r.get("id"): r for r in root.findall("road")}
    junctions = {j.get("id"): j for j in root.findall("junction")}

    nodes = set()
    for rid, r in roads.items():
        lanes_el = r.find("lanes")
        if lanes_el is None:
            continue
        for sec in lanes_el.findall("laneSection"):
            for side in ("left", "right"):
                sd = sec.find(side)
                if sd is None:
                    continue
                for le in sd.findall("lane"):
                    if le.get("type") == "driving":
                        nodes.add(f"{rid}:{le.get('id')}")

    edges = set()
    unresolved = Counter()

    # junction lookup: incomingRoad -> [(connectingRoad, {fromLane: toLane})]
    jmap = defaultdict(list)
    for jid, j in junctions.items():
        for c in j.findall("connection"):
            inc = c.get("incomingRoad")
            con = c.get("connectingRoad")
            lm = {}
            for ll in c.findall("laneLink"):
                lm[ll.get("from")] = ll.get("to")
            jmap[inc].append((jid, con, lm, c.get("contactPoint")))

    for rid, r in roads.items():
        lk = r.find("link")
        succ = lk.find("successor") if lk is not None else None
        pred = lk.find("predecessor") if lk is not None else None
        lanes_el = r.find("lanes")
        if lanes_el is None:
            continue
        secs = lanes_el.findall("laneSection")
        if not secs:
            continue
        last = secs[-1]
        first = secs[0]

        def lane_els(sec):
            out = []
            for side in ("left", "right"):
                sd = sec.find(side)
                if sd is None:
                    continue
                out.extend(sd.findall("lane"))
            return out

        # successor direction
        for le in lane_els(last):
            if le.get("type") != "driving":
                continue
            lid = le.get("id")
            src = f"{rid}:{lid}"
            ll = le.find("link")
            tgt_ids = [e.get("id") for e in (ll.findall("successor") if ll is not None else [])]
            if succ is None or not tgt_ids:
                continue
            et, eid = succ.get("elementType"), succ.get("elementId")
            if et == "road":
                for t in tgt_ids:
                    dst = f"{eid}:{t}"
                    if dst in nodes:
                        edges.add((src, dst))
                    else:
                        unresolved["road_successor_lane_missing"] += 1
            elif et == "junction":
                hit = False
                for jid, con, lm, cp in jmap.get(rid, []):
                    if jid != eid:
                        continue
                    if lid in lm:
                        dst = f"{con}:{lm[lid]}"
                        if dst in nodes:
                            edges.add((src, dst))
                            hit = True
                if not hit:
                    unresolved["junction_successor_unmapped"] += 1

        # predecessor direction (gives the incoming edge into this lane)
        for le in lane_els(first):
            if le.get("type") != "driving":
                continue
            lid = le.get("id")
            dst = f"{rid}:{lid}"
            ll = le.find("link")
            src_ids = [e.get("id") for e in (ll.findall("predecessor") if ll is not None else [])]
            if pred is None or not src_ids:
                continue
            et, eid = pred.get("elementType"), pred.get("elementId")
            if et == "road":
                for t in src_ids:
                    s = f"{eid}:{t}"
                    if s in nodes:
                        edges.add((s, dst))
                    else:
                        unresolved["road_predecessor_lane_missing"] += 1
            elif et == "junction":
                hit = False
                for inc, lst in jmap.items():
                    for jid, con, lm, cp in lst:
                        if jid == eid and con == rid:
                            for f_, t_ in lm.items():
                                if t_ == lid:
                                    s = f"{inc}:{f_}"
                                    if s in nodes:
                                        edges.add((s, dst))
                                        hit = True
                if not hit:
                    unresolved["junction_predecessor_unmapped"] += 1

    outdeg = Counter()
    indeg = Counter()
    for a, b in edges:
        outdeg[a] += 1
        indeg[b] += 1

    # weakly connected components
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen, comps = set(), []
    for n0 in nodes:
        if n0 in seen:
            continue
        stack, comp = [n0], []
        seen.add(n0)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(len(comp))
    comps.sort(reverse=True)

    return {
        "n_nodes_driving_lanes": len(nodes),
        "n_directed_edges": len(edges),
        "n_sources_indeg0": sum(1 for n0 in nodes if indeg[n0] == 0),
        "n_sinks_outdeg0": sum(1 for n0 in nodes if outdeg[n0] == 0),
        "n_isolated": sum(1 for n0 in nodes if indeg[n0] == 0 and outdeg[n0] == 0),
        "max_outdegree": max(outdeg.values()) if outdeg else 0,
        "outdegree_hist": dict(Counter(outdeg[n0] for n0 in nodes)),
        "indegree_hist": dict(Counter(indeg[n0] for n0 in nodes)),
        "n_weakly_connected_components": len(comps),
        "component_sizes_top10": comps[:10],
        "unresolved": dict(unresolved),
        "_edges": sorted(edges),
        "_nodes": sorted(nodes),
    }


# --------------------------------------------------------------------------------------
# 5. ego trajectory -> route
# --------------------------------------------------------------------------------------


def ego_positions(rig_path):
    """Return candidate ego XY tracks under both readings of T_rig_world.

    'T_rig_world' is ambiguous: it can mean world->rig (then position = -R^T t)
    or rig->world (then position = t). We compute BOTH and let the map decide,
    rather than assuming. Returns dict name -> (N,3) list.
    """
    d = json.load(open(rig_path))
    rt = d["rig_trajectories"][0]
    mats = rt["T_rig_worlds"]
    ts = rt.get("T_rig_world_timestamps_us")
    direct, inverted = [], []
    for M in mats:
        M = [[float(v) for v in row] for row in M]
        t = [M[0][3], M[1][3], M[2][3]]
        direct.append(t)
        R = [[M[i][j] for j in range(3)] for i in range(3)]
        # -R^T t
        inv = [-sum(R[k][i] * t[k] for k in range(3)) for i in range(3)]
        inverted.append(inv)
    return {
        "T_rig_worlds_translation": direct,
        "T_rig_worlds_inverted": inverted,
    }, ts, d


def path_length(pts):
    return sum(
        math.dist(pts[i][:2], pts[i + 1][:2]) for i in range(len(pts) - 1)
    )


def ego_from_pose_record(pose_path, lat0, lon0, k0=1.0):
    """Ego track in MAP-LOCAL metres, straight from the geodetic pose record.

    This is the frame-assumption-free route: pose_record.json carries explicit WGS84
    lat/lon/alt per pose, and the .xodr geoReference lat_0/lon_0 is bit-identical to
    record[0].alignment_world_pose.lat_lng_alt, so tm_forward() lands directly in the
    map's own coordinate system. No rotation guess, no rig/world direction ambiguity.
    """
    p = json.load(open(pose_path))
    rec = p["record"]
    out = []
    for r in rec:
        lla = r["alignment_world_pose"]["lat_lng_alt"]
        x, y = tm_forward(lla["latitude"], lla["longitude"], lat0, lon0, k0)
        aa = r["alignment_world_pose"]["axis_angle"]
        out.append({
            "t_us": r["timestamp_microseconds"],
            "x": x, "y": y, "alt": lla["altitude"],
            "lat": lla["latitude"], "lon": lla["longitude"],
            "yaw_deg": aa["angle"], "axis_z": aa["qz"],
        })
    return out, p["alignment_origin"]


def fit_rotation_2d(src, dst):
    """Best planar rotation about the origin taking src -> dst (both start at origin).

    Returns (theta_deg, rms_residual_m). Used as an INDEPENDENT cross-check that the
    T_rig_worlds track and the geodetic track are the same path.
    """
    n = min(len(src), len(dst))
    num = sum(src[i][0] * dst[i][1] - src[i][1] * dst[i][0] for i in range(n))
    den = sum(src[i][0] * dst[i][0] + src[i][1] * dst[i][1] for i in range(n))
    th = math.atan2(num, den)
    c, s = math.cos(th), math.sin(th)
    res = [
        math.dist((c * src[i][0] - s * src[i][1], s * src[i][0] + c * src[i][1]), dst[i][:2])
        for i in range(n)
    ]
    rms = math.sqrt(sum(r * r for r in res) / n) if n else float("nan")
    return math.degrees(th) % 360.0, rms, max(res) if res else None


def speed_profile(track):
    """Per-sample ground speed (m/s) from consecutive map-frame positions."""
    v = []
    for i in range(len(track) - 1):
        dt = (track[i + 1]["t_us"] - track[i]["t_us"]) / 1e6
        if dt <= 0:
            v.append(float("nan"))
            continue
        v.append(math.dist((track[i]["x"], track[i]["y"]),
                           (track[i + 1]["x"], track[i + 1]["y"])) / dt)
    return v


def road_speed_limits(root):
    """road_id -> (max_value, unit) from <road><type><speed>."""
    out = {}
    for r in root.findall("road"):
        for t in r.findall("type"):
            sp = t.find("speed")
            if sp is not None:
                out[r.get("id")] = (sp.get("max"), sp.get("unit"))
    return out


def snap_to_lanes(ego_xy, centerlines):
    """Nearest driving-lane centerline sample for each ego position."""
    if np is None:
        raise RuntimeError("numpy required")
    keys, allpts, owner = [], [], []
    for k, pts in centerlines.items():
        for p in pts:
            allpts.append(p)
            owner.append(k)
    A = np.asarray(allpts, dtype=float)
    E = np.asarray([[p[0], p[1]] for p in ego_xy], dtype=float)
    d2 = ((E[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    idx = d2.argmin(1)
    dist = np.sqrt(d2[np.arange(len(E)), idx])
    lanes = [owner[i] for i in idx]
    return lanes, dist


def compress_route(lanes):
    out = []
    for l in lanes:
        if not out or out[-1] != l:
            out.append(l)
    return out


# --------------------------------------------------------------------------------------
# 6. clipgt/ cross-source probe  (needs pyarrow; use a THROWAWAY venv, never tanitad-edge)
# --------------------------------------------------------------------------------------


def probe_clipgt(usdz, outdir, xodr_root=None):
    """Probe the clipgt/*.parquet annotation tables that ship in the same usdz.

    These are a SECOND, independent source for lanes / intersections / traffic lights.
    Required by the 'absence at one location is not absence' rule: the .xodr having no
    traffic light does not license writing 'the scene has no traffic light'.

    An empty table in this format is 1 row whose every field is null AND whose arrow
    type is inferred as `null` -- that is the placeholder pattern, not real data.
    """
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
    except Exception as e:
        return {"available": False, "reason": f"pyarrow missing: {e}"}

    cg = os.path.join(outdir, "clipgt")
    os.makedirs(cg, exist_ok=True)
    with zipfile.ZipFile(usdz) as z:
        for n in z.namelist():
            if n.startswith("clipgt/") and n.endswith(".parquet"):
                with open(os.path.join(outdir, n), "wb") as fh:
                    fh.write(z.read(n))

    def is_placeholder(tbl, payload_field):
        """True when the table is the 1-row all-null empty marker."""
        if tbl.num_rows != 1:
            return False
        t = tbl.schema.field(payload_field).type
        return "null" in str(t) and "double" not in str(t) and "string" not in str(t)

    out = {"available": True, "tables": {}}
    for fn in sorted(os.listdir(cg)):
        if not fn.endswith(".parquet"):
            continue
        name = fn[:-8]
        tbl = pq.read_table(os.path.join(cg, fn))
        payload = name if name in tbl.column_names else tbl.column_names[1]
        out["tables"][name] = {
            "rows": tbl.num_rows,
            "empty_placeholder": is_placeholder(tbl, payload),
            "payload_type": str(tbl.schema.field(payload).type)[:400],
        }

    # ---- lane attributes -------------------------------------------------------------
    lp = os.path.join(cg, "lane.parquet")
    if os.path.exists(lp):
        rows = pq.read_table(lp).to_pylist()
        out["lane"] = {
            "rows": len(rows),
            "lane_direction": dict(Counter(r["lane"]["lane_direction"] for r in rows)),
            "speed_limit_raw": dict(Counter(r["lane"]["speed_limit"] for r in rows)),
            "map_end": dict(Counter(r["lane"]["map_end"] for r in rows)),
            "vehicle_types": dict(Counter(
                ",".join(r["lane"]["vehicle_types"] or []) or "(none)" for r in rows)),
            "n_distinct_map_id": len({r["key"]["map_id"] for r in rows}),
        }
        styles = Counter()
        for r in rows:
            for s in (r["lane"]["left_edge_styles"] or []):
                styles[s] += 1
            for s in (r["lane"]["right_edge_styles"] or []):
                styles[s] += 1
        out["lane"]["edge_styles"] = dict(styles)
        out["lane"]["speed_limit_x1_609344_kmh"] = {
            v: round(float(v) * 1.609344, 4) for v in out["lane"]["speed_limit_raw"]
        }

        # ---- JOIN clipgt lanes onto xodr roads via the shared DeepMap feature id ------
        if xodr_root is not None:
            byid = {}
            for r in rows:
                byid.setdefault(r["key"]["map_id"], []).append(r["lane"])
            pairs = Counter()
            joined = 0
            for r in xodr_root.findall("road"):
                nm = r.get("name")
                t = r.find("type")
                sp = t.find("speed") if t is not None else None
                if sp is None or nm not in byid:
                    continue
                cv = [l["speed_limit"] for l in byid[nm] if l["speed_limit"]]
                if not cv:
                    continue
                joined += 1
                pairs[(sp.get("max"), sp.get("unit"), cv[0])] += 1
            out["xodr_join"] = {
                "n_xodr_road_names": len({r.get("name") for r in xodr_root.findall("road")}),
                "n_joined_on_feature_id": joined,
                "xodr_max_unit__vs__clipgt_speed_limit": {
                    f"xodr {k[0]} {k[1]} | clipgt {k[2]} | clipgt*1.609344 = "
                    f"{float(k[2]) * 1.609344:.4f} km/h": n
                    for k, n in sorted(pairs.items())
                },
                "verdict": (
                    "xodr unit='mph' is WRONG: its values are km/h. clipgt stores the same "
                    "limits genuinely in mph, and clipgt*1.609344 reproduces the xodr number "
                    "exactly on every joined road."
                ),
            }

    for nm in ("wait_line", "intersection_area"):
        p = os.path.join(cg, nm + ".parquet")
        if os.path.exists(p):
            rows = pq.read_table(p).to_pylist()
            out[nm] = {"rows": len(rows)}
            for f in ("category", "intersection_subtype", "is_implicit", "is_complete"):
                vals = [r[nm].get(f) for r in rows if f in r[nm]]
                if vals:
                    out[nm][f] = dict(Counter(str(v) for v in vals))
    return out


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usdz")
    ap.add_argument("--xodr")
    ap.add_argument("--rig")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--clipgt", action="store_true",
                    help="also probe clipgt/*.parquet (needs pyarrow; use a throwaway venv)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    report = {"tool": "xodr_probe.py", "evidence_class": "MEASURED"}

    if args.usdz:
        report["usdz"] = os.path.abspath(args.usdz)
        report["extracted"] = extract_members(args.usdz, args.out)
        xodr = report["extracted"]["map.xodr"]["path"]
        rig = report["extracted"]["rig_trajectories.json"]["path"]
    else:
        xodr = args.xodr
        rig = args.rig
        report["extracted"] = {
            "map.xodr": {
                "present": True,
                "bytes": os.path.getsize(xodr),
                "md5": hashlib.md5(open(xodr, "rb").read()).hexdigest(),
                "path": xodr,
            }
        }

    rep = parse_opendrive(xodr, sample_step=args.step)
    root = rep.pop("_root")
    centerlines = rep.pop("_centerlines")
    road_meta = rep.pop("_road_meta")
    report["opendrive"] = rep

    # --- lane graph ------------------------------------------------------------------
    lg = build_lane_graph(root)
    edges = lg.pop("_edges")
    nodes = lg.pop("_nodes")
    report["lane_graph"] = lg

    # --- georeferencing ---------------------------------------------------------------
    geo = rep["header"].get("geoReference_parsed") or {}
    georep = {"proj4_present": bool(rep["header"].get("geoReference_raw"))}
    if geo.get("proj") == "tmerc" and "lat_0" in geo and "lon_0" in geo:
        lat0, lon0 = float(geo["lat_0"]), float(geo["lon_0"])
        k0 = float(geo.get("k_0", geo.get("k", 1.0)))
        ext = rep.get("map_extent_local_m", {})
        corners = {
            "origin_0_0": (0.0, 0.0),
            "sw": (ext.get("x_min", 0.0), ext.get("y_min", 0.0)),
            "ne": (ext.get("x_max", 0.0), ext.get("y_max", 0.0)),
            "nw": (ext.get("x_min", 0.0), ext.get("y_max", 0.0)),
            "se": (ext.get("x_max", 0.0), ext.get("y_min", 0.0)),
        }
        wgs = {}
        rt_err = []
        for name, (x, y) in corners.items():
            la, lo = tm_inverse(x, y, lat0, lon0, k0)
            wgs[name] = {"lat": la, "lon": lo}
            bx, by = tm_forward(la, lo, lat0, lon0, k0)
            rt_err.append(math.hypot(bx - x, by - y))
        georep.update({
            "projection": "tmerc",
            "lat_0": lat0, "lon_0": lon0, "k_0": k0,
            "ellps": geo.get("ellps"), "units": geo.get("units"),
            "geoidgrids": geo.get("geoidgrids"),
            "malformed_tokens": rep["header"].get("geoReference_malformed_tokens"),
            "corners_wgs84": wgs,
            "roundtrip_max_error_m": max(rt_err) if rt_err else None,
            "osm_matchable": True,
        })
    else:
        georep["osm_matchable"] = False
    report["georeferencing"] = georep

    # --- ego trajectory, GEODETIC path (no frame assumption) --------------------------
    pose_path = os.path.join(args.out, "pose_record.json")
    ext = rep.get("map_extent_local_m", {})
    if georep.get("osm_matchable") and os.path.exists(pose_path):
        lat0, lon0, k0 = georep["lat_0"], georep["lon_0"], georep["k_0"]
        track, align_origin = ego_from_pose_record(pose_path, lat0, lon0, k0)
        xy = [[t["x"], t["y"]] for t in track]

        # data_info gives the RENDERED scene window; the pose record may be longer.
        di_path = os.path.join(args.out, "data_info.json")
        n_scene = None
        if os.path.exists(di_path):
            di = json.load(open(di_path))
            n_scene = di["pose-range"]["num-poses"]

        ego_rep = {
            "source": "pose_record.json record[].alignment_world_pose.lat_lng_alt -> tm_forward",
            "n_poses_in_pose_record": len(track),
            "n_poses_in_scene_window_data_info": n_scene,
            "duration_pose_record_s": (track[-1]["t_us"] - track[0]["t_us"]) / 1e6,
            "alignment_origin": align_origin,
            "xodr_lat0_equals_record0_lat": (lat0 == track[0]["lat"]),
            "xodr_lon0_equals_record0_lon": (lon0 == track[0]["lon"]),
            "lat_range": [min(t["lat"] for t in track), max(t["lat"] for t in track)],
            "lon_range": [min(t["lon"] for t in track), max(t["lon"] for t in track)],
            "alt_range_m": [min(t["alt"] for t in track), max(t["alt"] for t in track)],
            "map_frame_bbox_m": {
                "x_min": min(p[0] for p in xy), "x_max": max(p[0] for p in xy),
                "y_min": min(p[1] for p in xy), "y_max": max(p[1] for p in xy),
            },
            "path_length_m_full": path_length(xy),
        }
        if n_scene:
            ego_rep["path_length_m_scene_window"] = path_length(xy[:n_scene])
            ego_rep["duration_scene_window_s"] = (
                track[min(n_scene, len(track)) - 1]["t_us"] - track[0]["t_us"]
            ) / 1e6

        v = speed_profile(track)
        vv = [x for x in v if x == x]
        ego_rep["speed_mps"] = {
            "min": min(vv), "max": max(vv), "mean": sum(vv) / len(vv),
        }
        ego_rep["speed_kmh"] = {k: x * 3.6 for k, x in ego_rep["speed_mps"].items()}

        # INDEPENDENT cross-check: is T_rig_worlds the same path up to a rotation?
        if rig and os.path.exists(rig):
            cands, ts, _ = ego_positions(rig)
            xc = {}
            for name, pts in cands.items():
                th, rms, mx = fit_rotation_2d([p[:2] for p in pts], xy)
                xc[name] = {"best_rotation_deg": th, "rms_residual_m": rms,
                            "max_residual_m": mx, "n_compared": min(len(pts), len(xy))}
            ego_rep["crosscheck_T_rig_worlds_vs_geodetic"] = xc
            try:
                pr = json.load(open(pose_path))
                ego_rep["record0_axis_angle_deg"] = pr["record"][0]["alignment_world_pose"]["axis_angle"]["angle"]
            except Exception:
                pass
        report["ego"] = ego_rep

        # --- snap -> lane -> route ----------------------------------------------------
        lanes, dist = snap_to_lanes(xy, centerlines)
        route = compress_route(lanes)
        route_roads = compress_route([l.split(":")[0] for l in lanes])
        junc_roads = {rid for rid, m in road_meta.items() if m["junction"] not in ("-1", "", None)}
        limits = road_speed_limits(root)
        posted = Counter(limits.get(l.split(":")[0], ("?", "?")) for l in lanes)
        report["route"] = {
            "n_ego_samples": len(xy),
            "snap_dist_m": {
                "min": float(dist.min()), "max": float(dist.max()),
                "mean": float(dist.mean()), "median": float(np.median(dist)),
                "p90": float(np.percentile(dist, 90)), "p99": float(np.percentile(dist, 99)),
            },
            "frac_snap_under_2m": float((dist < 2.0).mean()),
            "frac_snap_under_1m": float((dist < 1.0).mean()),
            "route_lane_sequence": route,
            "n_distinct_lanes_traversed": len(set(lanes)),
            "route_road_sequence": route_roads,
            "n_distinct_roads_traversed": len(set(l.split(":")[0] for l in lanes)),
            "n_junction_roads_traversed": len(
                {r for r in (l.split(":")[0] for l in lanes) if r in junc_roads}
            ),
            "posted_limit_on_traversed_roads": {f"{k[0]} {k[1]}": n for k, n in posted.items()},
            "ego_max_speed_kmh": ego_rep["speed_kmh"]["max"],
            "ego_mean_speed_kmh": ego_rep["speed_kmh"]["mean"],
        }
        if n_scene:
            ls, ds = lanes[:n_scene], dist[:n_scene]
            report["route"]["scene_window_only"] = {
                "n": int(n_scene),
                "snap_median_m": float(np.median(ds)),
                "snap_p90_m": float(np.percentile(ds, 90)),
                "snap_max_m": float(ds.max()),
                "frac_under_2m": float((ds < 2.0).mean()),
                "route_road_sequence": compress_route([l.split(":")[0] for l in ls]),
                "n_distinct_roads": len(set(l.split(":")[0] for l in ls)),
            }

        # --- coverage cross-check -----------------------------------------------------
        if ext:
            eb = ego_rep["map_frame_bbox_m"]
            report["coverage_crosscheck"] = {
                "map_bbox_m": {"w": ext["width_x"], "h": ext["height_y"]},
                "map_area_km2": ext["width_x"] * ext["height_y"] / 1e6,
                "map_total_road_length_m": rep["road_length_m"]["total"],
                "ego_path_length_m_full": ego_rep["path_length_m_full"],
                "ego_path_length_m_scene_window": ego_rep.get("path_length_m_scene_window"),
                "map_road_len_over_ego_scene_path": (
                    rep["road_length_m"]["total"]
                    / max(1e-9, ego_rep.get("path_length_m_scene_window") or ego_rep["path_length_m_full"])
                ),
                "ego_bbox_m": {"w": eb["x_max"] - eb["x_min"], "h": eb["y_max"] - eb["y_min"]},
                "ego_fully_inside_map_bbox": bool(
                    eb["x_min"] >= ext["x_min"] and eb["x_max"] <= ext["x_max"]
                    and eb["y_min"] >= ext["y_min"] and eb["y_max"] <= ext["y_max"]
                ),
                "map_margin_beyond_ego_m": {
                    "x_min_side": eb["x_min"] - ext["x_min"], "x_max_side": ext["x_max"] - eb["x_max"],
                    "y_min_side": eb["y_min"] - ext["y_min"], "y_max_side": ext["y_max"] - eb["y_max"],
                },
            }

        with open(os.path.join(args.out, "ego_track_map_frame.json"), "w") as fh:
            json.dump({"track": track, "snapped_lane": lanes,
                       "snap_dist_m": [float(x) for x in dist]}, fh)

    # --- clipgt/ second source --------------------------------------------------------
    if args.clipgt and args.usdz:
        report["clipgt"] = probe_clipgt(args.usdz, args.out, xodr_root=root)

    out_json = os.path.join(args.out, "xodr_probe_output.json")
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    with open(os.path.join(args.out, "lane_centerlines.json"), "w") as fh:
        json.dump({k: [[round(c, 3) for c in p] for p in v] for k, v in centerlines.items()}, fh)
    with open(os.path.join(args.out, "lane_graph_edges.json"), "w") as fh:
        json.dump({"nodes": nodes, "edges": edges}, fh)

    print(json.dumps(report, indent=2, default=str)[:20000])
    print("\nWROTE", out_json)


if __name__ == "__main__":
    main()
