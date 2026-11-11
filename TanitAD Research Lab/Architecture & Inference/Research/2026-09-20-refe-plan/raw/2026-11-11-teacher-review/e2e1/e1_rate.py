#!/usr/bin/env python3
"""E1 step 2: sim8 DAC failure + footprint margin of the SAME samples at two sim rates (banked 10 Hz bank refe_navtrain10 vs
banked legacy 20 Hz bank refe_navtrain, rows without a sim_hz field), on the overlap by (log, token), rank 0, step 0.
Per plan (D1/D3 definitions): raw20 = footprint at the 20 poses; interp8 = footprint of the 8-pose plan at 0.1 s; sim8 = NAVSIM
LQR+bicycle footprint (navsim_dac.simulated_corners); margin = min over steps and 4 corners of the signed distance to the
drivable-area union (+ inside), on interp8 poses (p4_dac_margin.margin). class A = raw20 fail, B = interp8 fail (not A), C = sim8 fail only.
python e1_rate.py --refe <dir> --d1 <d1_navtrain.json> --out <jsonl> --shard i/N"""
import argparse, glob, json, os, sys, time
os.environ.setdefault("REFE_SIM_HZ", "10"); os.environ.setdefault("NUPLAN_MAPS_ROOT", "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
B10 = "E:/Projects/TanitAD/data/refe_navtrain10/r0/targets_rank0.jsonl"
B20 = "E:/Projects/TanitAD/data/refe_navtrain/targets_rank0.jsonl"
DB_ROOT = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"

def margin(polys, corners):
    import shapely
    from shapely.ops import unary_union
    u = unary_union(polys); pts = corners.reshape(-1, 2); geo = shapely.points(pts[:, 0], pts[:, 1])
    inside = shapely.contains(u, geo); d = shapely.distance(u.boundary, geo)
    return float(np.min(np.where(inside, d, -d)))

def load(path, want, hz10):
    out = {}
    for ln in open(path, encoding="utf-8"):
        try: r = json.loads(ln)
        except json.JSONDecodeError: continue
        k = (r.get("log_name"), r.get("token"))
        if k in want and int(r.get("step", 0)) == 0 and int(r.get("rank", 0)) == 0 and (int(r.get("sim_hz", 0)) == 10) == hz10 and len(r["traj"]) == 20:
            out[k] = r["traj"]
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--refe", required=True); ap.add_argument("--d1", required=True); ap.add_argument("--out", required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--overlap-only", action="store_true"); a = ap.parse_args()
    sys.path.insert(0, a.refe)
    import navsim_dac as ND, navtrain_scenarios as NS
    si, sn = (int(x) for x in a.shard.split("/"))
    want = {(r["log"], r["token"]) for r in json.load(open(a.d1, encoding="utf-8"))["rows"]}
    t10, t20 = load(B10, want, True), load(B20, want, False)
    keys = sorted(set(t10) & set(t20))
    by = {}
    for k in keys: by.setdefault(k[0], []).append(k[1])
    dbs = NS.index_dbs(db_root=DB_ROOT)
    t0 = time.time(); n = 0
    with open(a.out, "a", encoding="utf-8") as fo:
        for lg in sorted(by)[si::sn]:
            for sc in NS.build_scenarios_for_log(dbs[lg], by[lg]):
                tok = sc._initial_lidar_token
                rec = {"log": lg, "token": tok}
                try:
                    ego = sc.get_ego_state_at_iteration(0); ra = ego.rear_axle
                    polys = ND.drivable_polygons(sc.map_api, ego)
                    for nm, tr in (("hz10", t10), ("hz20", t20)):
                        t = np.asarray(tr[(lg, tok)], np.float64); p8 = ND.to_navsim(t)
                        c_i = ND.corners_global(ND.at_10hz(p8), ra.x, ra.y, ra.heading)
                        rec[nm] = {"raw20": bool(ND.violations(polys, ND.corners_global(t, ra.x, ra.y, ra.heading)[None])[0]),
                                   "interp8": bool(ND.violations(polys, c_i[None])[0]),
                                   "sim8": bool(ND.violations(polys, ND.simulated_corners(p8[None], ego))[0]),
                                   "margin_m": margin(polys, c_i),
                                   "end": [float(x) for x in t[-1]]}
                except Exception as e:  # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                fo.write(json.dumps(rec) + "\n"); fo.flush(); n += 1
    print("ZZE1_SHARD_DONE", a.shard, n, len(keys), round(time.time() - t0, 1))

if __name__ == "__main__":
    main()
