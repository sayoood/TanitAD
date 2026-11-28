"""FS19-2 — posted-speed-limit coverage in the nuPlan maps (map layer only, no logs).

nuPlan ships one GeoPackage (SQLite) per city. Lanes and lane connectors carry a
`speed_limit_mps` column that the devkit reads as Optional[float] (NaN/NULL allowed,
see nuplan/common/maps/nuplan_map/lane.py). This script measures, per city and layer:
  * n features, n with a finite positive limit (count coverage),
  * the same weighted by polyline length when a baseline-path layer is joinable,
  * the distinct limit values (a posted limit should sit on a small ladder).

Evidence class of its output: MEASURED (on the map DBs). It does NOT measure the
ego-occupied lane-frame coverage the FS19-2 pre-registration names — that needs the
navtrain logs; the lane-length-weighted number is the stated proxy.

Usage: python nuplan_speed_limit_coverage.py <maps_root> [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import struct
import sys
from collections import Counter
from pathlib import Path

LAYERS = ("lanes_polygons", "lane_connectors", "baseline_paths")


def _finite_pos(v) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and f > 0.0


def _gpkg_linestring_length(blob: bytes, geographic: bool = False) -> float | None:
    """Length of a GeoPackage LINESTRING blob (GP header + WKB). None if not a linestring."""
    if not blob or blob[:2] != b"GP":
        return None
    flags = blob[3]
    env = (flags >> 1) & 0b111
    env_len = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(env, 0)
    wkb = blob[8 + env_len:]
    order = "<" if wkb[0] == 1 else ">"
    gtype = struct.unpack(order + "I", wkb[1:5])[0]
    base = gtype % 1000
    if base != 2:  # LINESTRING only
        return None
    dim = 3 if gtype // 1000 in (1, 2) else (4 if gtype // 1000 == 3 else 2)
    n = struct.unpack(order + "I", wkb[5:9])[0]
    pts = struct.unpack(order + "d" * (n * dim), wkb[9:9 + 8 * n * dim])
    xy = [(pts[i * dim], pts[i * dim + 1]) for i in range(n)]
    if geographic:
        # nuPlan gpkg geometries are EPSG:4326 (x = lon, y = lat, degrees). Local
        # equirectangular metres per segment; error << 1 % at lane-segment scale.
        def seg(a, b):
            lat = math.radians((a[1] + b[1]) / 2)
            return math.hypot((b[0] - a[0]) * 111_320.0 * math.cos(lat), (b[1] - a[1]) * 110_574.0)
        return sum(seg(xy[i], xy[i + 1]) for i in range(n - 1))
    return sum(math.dist(xy[i], xy[i + 1]) for i in range(n - 1))


def audit_city(gpkg: Path) -> dict:
    con = sqlite3.connect(f"file:{gpkg}?mode=ro", uri=True)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    out: dict = {"file": gpkg.as_posix(), "layers": {}}
    for layer in LAYERS:
        if layer not in tables:
            out["layers"][layer] = {"present": False}
            continue
        cols = [r[1] for r in con.execute(f'pragma table_info("{layer}")')]
        rec: dict = {"present": True, "columns_with_speed": [c for c in cols if "speed" in c.lower()]}
        if "speed_limit_mps" in cols:
            vals = [r[0] for r in con.execute(f'select speed_limit_mps from "{layer}"')]
            fin = [float(v) for v in vals if _finite_pos(v)]
            rec.update(n=len(vals), n_finite=len(fin),
                       frac_finite=(len(fin) / len(vals)) if vals else None,
                       ladder_kmh=sorted(Counter(round(v * 3.6) for v in fin).items()),
                       ladder_mph=sorted(Counter(round(v * 2.236936) for v in fin).items()))
        # lanes_polygons also carries min_speed / max_speed; audit them as a second candidate
        for c in ("min_speed", "max_speed"):
            if c in cols:
                vv = [r[0] for r in con.execute(f'select "{c}" from "{layer}"')]
                ff = [float(v) for v in vv if _finite_pos(v)]
                rec[c] = {"n_finite": len(ff),
                          "values": sorted(Counter(round(v, 3) for v in ff).items())[:25]}
        out["layers"][layer] = rec
    # length weighting: baseline_paths carry the lane-centre polyline and link to lanes/connectors
    if "baseline_paths" in tables:
        bcols = [r[1] for r in con.execute('pragma table_info("baseline_paths")')]
        geom = "geom" if "geom" in bcols else None
        srs = con.execute("select srs_id from gpkg_geometry_columns where table_name='baseline_paths'").fetchone()
        geographic = bool(srs) and srs[0] == 4326
        out["baseline_paths_srs"] = srs[0] if srs else None
        # join keys read from the schema: baseline_paths.lane_fid -> lanes_polygons.lane_fid,
        # baseline_paths.lane_connector_fid -> lane_connectors.fid
        for layer, fk, pk in (("lanes_polygons", "lane_fid", "lane_fid"),
                              ("lane_connectors", "lane_connector_fid", "fid")):
            if geom and fk in bcols and layer in tables and "speed_limit_mps" in \
                    [r[1] for r in con.execute(f'pragma table_info("{layer}")')]:
                q = (f'select b.{geom}, l.speed_limit_mps from baseline_paths b '
                     f'join "{layer}" l on b.{fk} = l.{pk}')
                tot = fin = 0.0
                for blob, v in con.execute(q):
                    ln = _gpkg_linestring_length(blob, geographic)
                    if ln is None:
                        continue
                    tot += ln
                    if _finite_pos(v):
                        fin += ln
                out["layers"][layer]["len_total_m"] = round(tot, 1)
                out["layers"][layer]["len_frac_finite"] = (fin / tot) if tot else None
    con.close()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("maps_root", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    files = sorted(a.maps_root.rglob("*.gpkg"))
    if not files:
        print(f"NO .gpkg under {a.maps_root}", file=sys.stderr)
        return 2
    res = [audit_city(f) for f in files]
    for r in res:
        print(r["file"])
        for k, v in r["layers"].items():
            if not v.get("present"):
                print(f"  {k:16s} ABSENT"); continue
            print(f"  {k:16s} n={v.get('n')} finite={v.get('n_finite')} "
                  f"frac={v.get('frac_finite')} len_frac={v.get('len_frac_finite')} "
                  f"speed_cols={v['columns_with_speed']}")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
