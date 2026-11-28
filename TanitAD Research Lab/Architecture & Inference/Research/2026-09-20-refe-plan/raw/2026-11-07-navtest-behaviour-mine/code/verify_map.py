#!/usr/bin/env python
"""Independent brute-force recomputation of the map-based flags for three tokens (no STRtree, no shared code with mine_features.py).
Run: C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I verify_map.py"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import glob, json, math, pickle
import numpy as np, pandas as pd, pyogrio
from shapely.geometry import Point

OUT = r"E:/Projects/TanitAD-artifacts/navtest_behaviour_mine_20261107"
LOG_DIR = r"F:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test"
MAPS = r"F:/Archive/devbox-C/navsim/data/maps"
df = pd.read_parquet(f"{OUT}/features.parquet").set_index("token")
hv = json.load(open(f"{OUT}/hand_verified_tokens.json"))
tokens = {"red_ahead_stop": hv["red_ahead_stop"]["token"], "junction_turn": hv["junction_turn"]["token"], "free_road_cruise(no junction)": hv["free_road_cruise"]["token"]}
res = {}
maps = {}
for name, tok in tokens.items():
    row = df.loc[tok]
    city = row.map_location
    g = glob.glob(f"{MAPS}/{city}/*/map.gpkg")[0]
    if city not in maps:
        meta = pyogrio.read_dataframe(g, layer="meta")
        proj = meta[meta["key"] == "projectedCoordSystem"]["value"].iloc[0]
        inter = pyogrio.read_dataframe(g, layer="intersections", fid_as_index=True).to_crs(proj)
        gen = pyogrio.read_dataframe(g, layer="gen_lane_connectors_scaled_width_polygons").to_crs(proj)
        lc = pyogrio.read_dataframe(g, layer="lane_connectors", fid_as_index=True).to_crs(proj)
        maps[city] = (inter, gen, lc)
    inter, gen, lc = maps[city]
    frames = pickle.load(open(f"{LOG_DIR}/{row.log_name}.pkl", "rb"))
    j = int(row.list_idx)
    f0 = frames[j]
    R = np.asarray(f0["ego2global"], float)
    x0, y0, h0 = R[0, 3], R[1, 3], math.atan2(R[1, 0], R[0, 0])
    polys = list(inter.geometry) + list(gen.geometry)
    inj = lambda x, y: any(p.contains(Point(x, y)) for p in polys)
    jt0 = inj(x0, y0)
    dist = next((d for d in range(0, 61, 5) if inj(x0 + d * math.cos(h0), y0 + d * math.sin(h0))), None)
    matches = []
    for cid, is_red in f0["traffic_lights"]:
        geom = lc.loc[int(cid)].geometry
        cs = list(geom.coords); acc = 0.0; p5 = cs[-1]
        for a, b in zip(cs[:-1], cs[1:]):           # walk the polyline to 5 m (independent of shapely.interpolate)
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            if acc + seg >= 5.0:
                t = (5.0 - acc) / seg; p5 = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])); break
            acc += seg
        eh = math.atan2(p5[1] - cs[0][1], p5[0] - cs[0][0])
        dx, dy = cs[0][0] - x0, cs[0][1] - y0
        ex, ey = math.cos(h0) * dx + math.sin(h0) * dy, -math.sin(h0) * dx + math.cos(h0) * dy
        dh = (eh - h0 + math.pi) % (2 * math.pi) - math.pi
        if 0 < ex < 60 and abs(ey) < 3.5 and abs(dh) < math.radians(35):
            matches.append(dict(conn=int(cid), red=bool(is_red), ex=round(ex, 2), ey=round(ey, 2), dhead_deg=round(math.degrees(dh), 1)))
    red_d = [m["ex"] for m in matches if m["red"]]
    out = dict(token=tok, city=city, junction_t0=dict(brute=bool(jt0), table=bool(row.junction_t0)),
               junction_dist_ahead_m=dict(brute=dist, table=None if pd.isna(row.junction_dist_ahead_m) else float(row.junction_dist_ahead_m)),
               tl_red_ahead=dict(brute=bool(red_d), table=bool(row.tl_red_ahead)), tl_green_ahead=dict(brute=any(not m["red"] for m in matches), table=bool(row.tl_green_ahead)),
               tl_red_ahead_dist_m=dict(brute=min(red_d) if red_d else None, table=None if pd.isna(row.tl_red_ahead_dist_m) else round(float(row.tl_red_ahead_dist_m), 2)),
               matched_connectors=matches)
    out["agree"] = bool(out["junction_t0"]["brute"] == out["junction_t0"]["table"] and out["tl_red_ahead"]["brute"] == out["tl_red_ahead"]["table"]
                        and out["tl_green_ahead"]["brute"] == out["tl_green_ahead"]["table"]
                        and (out["junction_dist_ahead_m"]["brute"] == out["junction_dist_ahead_m"]["table"]))
    res[name] = out
    print(name, "agree" if out["agree"] else "MISMATCH", json.dumps(out)[:600], flush=True)
json.dump(res, open(f"{OUT}/map_handcheck.json", "w"), indent=1)
