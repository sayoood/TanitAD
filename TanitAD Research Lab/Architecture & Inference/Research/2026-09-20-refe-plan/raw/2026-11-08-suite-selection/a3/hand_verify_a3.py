#!/usr/bin/env python3
"""Hand-verification of ONE candidate per new slot against the RAW nuPlan DB (sqlite, no selector code): ego track, traffic_light_status, lidar_box.
Map facts (connector turn type, polygons) are read straight from the gpkg with pyogrio.  Prints the numbers; writes HAND_VERIFY_A3.txt.
Usage: python hand_verify_a3.py   (picks rank 1 of every new slot from SUITE_CANDIDATES_A3.json; set VERIFY_RANK=n to pick another)"""
import json, math, os, sqlite3, sys, io, glob
import numpy as np

A3 = "E:/Projects/TanitAD-artifacts/tsf_bundle2_20261107/a3"
DB_DIR = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
MAPS = "F:/Archive/devbox-C/navsim/data/maps"
RANK = int(os.environ.get("VERIFY_RANK", "1"))
OUT = io.StringIO()


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); OUT.write(s + "\n")


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def load(db, tok):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cur = con.cursor()
    rows = cur.execute("select lp.token, lp.timestamp, e.x, e.y, e.qw, e.qx, e.qy, e.qz, e.vx, e.vy from lidar_pc lp join ego_pose e on e.token=lp.ego_pose_token order by lp.timestamp").fetchall()
    toks = [bytes(r[0]).hex() for r in rows]
    r0 = toks.index(tok)
    return con, rows, toks, r0


def yaw_of(r):
    qw, qx, qy, qz = r[4:8]
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def tl_status(con, lp_hex, cid):
    return [(r[0]) for r in con.execute("select status from traffic_light_status where lidar_pc_token=? and lane_connector_id=?", (bytes.fromhex(lp_hex), cid)).fetchall()]


def connector(city, cid):
    import pyogrio
    g = glob.glob(f"{MAPS}/{city}/*/map.gpkg")[0]
    lc = pyogrio.read_dataframe(g, layer="lane_connectors", fid_as_index=True, read_geometry=False)
    return lc.loc[cid]


def main():
    d = json.load(open(f"{A3}/SUITE_CANDIDATES_A3.json"))
    for slot in ["J-UL", "J-R", "J-S", "TL-G", "TL-S"]:
        ents = d["slots"][slot]["ordered_one_per_log"]
        if len(ents) < RANK:
            P(f"== {slot}: no rank {RANK}"); continue
        e = ents[RANK - 1]; tok, log = e["token"], e["log"]
        pv = e["predicate_values"]; mf = pv["map_fields"]
        con, rows, toks, r0 = load(f"{DB_DIR}/{log}.db", tok)
        t0 = rows[r0][1] / 1e6
        P(f"\n== {slot} rank {e['rank']} token {tok} log {log} city {pv['city']}  connector {mf['connector']}  (DB lidar_pc rank {r0}, t0 {t0:.3f})")
        idx = [r0 + 2 * k for k in range(-20, 151)]
        pose = np.array([[rows[i][2], rows[i][3], yaw_of(rows[i]), math.hypot(rows[i][8], rows[i][9]), (rows[i][1] / 1e6 - t0)] for i in idx])
        # signal letters at the 2 Hz key frames (every 10th tick), raw status column
        cid = mf["connector"]
        if slot.startswith("TL"):
            P("t_rel  speed(vx,vy)  status(raw DB) of governing connector", cid)
            line = []
            for k in range(0, 171, 5):
                st = tl_status(con, toks[idx[k]], cid)
                line.append(f"{pose[k, 4]:+5.1f}s v={pose[k, 3]:4.1f} {','.join(map(str, st)) or '-'}")
            for i in range(0, len(line), 3):
                P("   " + " | ".join(line[i:i + 3]))
            v = pose[:, 3]; stat = [(tl_status(con, toks[i], cid) or ["-"])[0] for i in idx]
            stat = [s if isinstance(s, str) else str(s) for s in stat]
            k1 = int(np.argmax(v <= 0.1)) if (v <= 0.1).any() else -1
            P(f"first v<=0.1 at t_rel {pose[k1, 4]:+.1f}s (tick {k1}); status there {stat[k1]}; stop point (map) {mf['stop_line_xy']} stop_line_s {mf['stop_line_s']:.2f} m; ego rear axle at k1 {pose[k1, :2].round(2).tolist()}")
            if slot == "TL-G":
                kg = next((k for k in range(k1, 171) if stat[k].lower().startswith("g") or stat[k] == "3"), None)
                P(f"first green after stop at t_rel {pose[kg, 4]:+.1f}s; max speed after green {v[kg:].max():.2f} m/s; v at tick before green {v[kg - 1]:.2f}")
            else:
                run = best = 0
                for k in range(k1, 171):
                    run = run + 1 if v[k] <= 0.1 else 0; best = max(best, run)
                P(f"longest stopped run {(best - 1) * 0.1:.1f} s at 10 Hz; max speed before k1 {v[:k1 + 1].max():.2f}; any green after k1: {any(s.lower().startswith('g') or s == '3' for s in stat[k1:])}")
        else:
            row = connector(pv["city"], cid)
            P(f"gpkg lane_connectors fid {cid}: turn_type_fid {row.turn_type_fid} (0 S, 1 L, 2 R), intersection_fid {row.intersection_fid}")
            h = np.unwrap(pose[:, 2])
            # ego heading change across the connector: from the tick where the rear axle is within 3 m of the connector start to the exit (approx by total in-window change)
            P(f"ego yaw change over the 17 s window {math.degrees(h[-1] - h[0]):+.1f} deg; speed at t0 {pose[20, 3]:.2f} m/s; start/end {pose[0, :2].round(1).tolist()} -> {pose[-1, :2].round(1).tolist()}")
            P("conflict agents (OpenScene track tokens):", mf["conflict_agent_ids"], " zone vertices:", len(mf["conflict_zone_xy"] or []))
            # raw lidar_box check: the first conflict agent's box positions (DB track tokens are the same hex as the OpenScene track tokens)
            for tk in (mf["conflict_agent_ids"] or [])[:1]:
                q = []
                for k in range(0, 171, 5):
                    r = con.execute("select x,y,yaw,vx,vy,length,width from lidar_box where lidar_pc_token=? and track_token=?", (bytes.fromhex(toks[idx[k]]), bytes.fromhex(tk))).fetchall()
                    if r:
                        q.append((pose[k, 4], r[0]))
                P(f"agent {tk}: {len(q)} DB lidar_box samples (every 0.5 s); first/last t_rel {q[0][0]:+.1f}/{q[-1][0]:+.1f}")
                if mf["conflict_zone_xy"]:
                    from shapely.geometry import Polygon, box as _b
                    import shapely.affinity as af
                    Z = Polygon(mf["conflict_zone_xy"])
                    hit = []
                    for t_, (x, y, yaw, vx, vy, L, W) in q:
                        bx = af.rotate(_b(x - L / 2, y - W / 2, x + L / 2, y + W / 2), yaw, origin=(x, y), use_radians=True)
                        if bx.intersects(Z):
                            hit.append((round(t_, 1), round(math.hypot(vx, vy), 2)))
                    P(f"   DB box footprints intersecting the section-5 zone hull at (t_rel, speed): {hit}")
        P(f"map fields: stop_line_xy {mf['stop_line_xy']} ({mf['stop_line_source']}), stop_line_s {mf['stop_line_s']:.2f}, heads {mf['signal_heads']}, governs {mf['signal_governs_ego']}")
    open(f"{A3}/HAND_VERIFY_A3.txt", "w", encoding="utf-8").write(OUT.getvalue())


if __name__ == "__main__":
    main()
