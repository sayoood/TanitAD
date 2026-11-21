#!/usr/bin/env python3
"""From the pool-B pkl table (raw/poolB_frames.json.gz, MEASURED from the local sha-verified metadata tgz):
 (1) per-log: frames, route anchors (NAVSIM geometry), command mix, K20 = logs with >=1 route anchor whose command is LEFT/STRAIGHT/RIGHT;
 (2) re-derive the compact_metadata (INHERITED) numbers from the pkl itself;
 (3) for the 7 pool-B logs whose nuPlan DB is local: pkl pairing (cams data_path) vs the DB rule (planner.FrameResolver: nearest `image` row, <= 120 ms, per camera),
     pkl K/distortion vs the DB `camera` table (calib_table.read_db fields), pkl frame tokens vs DB lidar_pc tokens.
Writes raw/poolB_pkl_summary.json (per-log table + checks)."""
import os, sys, json, gzip, sqlite3, pickle, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dbpickle
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, "..", "raw")
P = json.load(gzip.open(os.path.join(RAW, "poolB_frames.json.gz"), "rt"))
B = {r["log"]: r for r in json.load(open(os.path.join(RAW, "poolB_logs.json")))}
CMDN = ("L", "S", "R", "U")
rows = {}
for log, v in P.items():
    fr = v["frames"]; n = len(fr)
    cmds = collections.Counter(CMDN[int(np.argmax(f["cmd"]))] for f in fr)
    anchors = [j for j in range(3, n - 10) if fr[j]["route"]]
    known = [j for j in anchors if int(np.argmax(fr[j]["cmd"])) != 3]
    rows[log] = {"n_frames": n, "route_frames": sum(f["route"] for f in fr), "anchors_route": len(anchors), "anchors_known_cmd": len(known), "cmd_counts": dict(cmds),
                 "current_frame_idx_known_cmd": known, "pkl_sha256": v["pkl_sha256"]}
chk = {"anchors_route_equals_pool_lists": all(rows[l]["anchors_route"] == B[l]["anchors_route"] for l in rows),
       "n_frames_equals_pool_lists": all(rows[l]["n_frames"] == B[l]["n_frames_2Hz"] for l in rows),
       "total_anchors_route": sum(r["anchors_route"] for r in rows.values()), "total_anchors_known_cmd": sum(r["anchors_known_cmd"] for r in rows.values())}
K = sorted(l for l, r in rows.items() if r["anchors_known_cmd"] >= 1)
chk["K_logs"] = len(K); chk["K_logs_anchors"] = sum(rows[l]["anchors_known_cmd"] for l in K)
chk["routed_logs_all_unknown_cmd"] = sorted(l for l, r in rows.items() if r["anchors_route"] > 0 and r["anchors_known_cmd"] == 0).__len__()
chk["per_log_command_is_log_constant_or_mixed"] = {l: rows[l]["cmd_counts"] for l in K[:5]}
# --- DB checks for the 7 local DBs
ROOT = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval/"
CAMS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
db = {}
for log in P:
    p = ROOT + log + ".db"
    if not os.path.exists(p): continue
    c = sqlite3.connect("file:%s?mode=ro" % p, uri=True)
    lp = {r[0]: r[1] for r in c.execute("SELECT lower(hex(token)), timestamp FROM lidar_pc")}
    per = {}
    for ch in CAMS:
        row = c.execute("SELECT token FROM camera WHERE channel=?", (ch,)).fetchone()
        rs = c.execute("SELECT timestamp, filename_jpg FROM image WHERE camera_token=? ORDER BY timestamp", (row[0],)).fetchall()
        per[ch] = (np.array([r[0] for r in rs], dtype=np.int64), [r[1] for r in rs])
    cal = {}
    cols = [r[1] for r in c.execute("PRAGMA table_info(camera)").fetchall()]
    for ch in CAMS:
        r = dict(zip(cols, c.execute("SELECT * FROM camera WHERE channel=?", (ch,)).fetchone()))
        K_ = dbpickle.loads(r["intrinsic"]); d_ = dbpickle.loads(r["distortion"])[:5]
        cal[ch] = [K_[0], K_[4], K_[2], K_[5]] + list(d_)
    c.close()
    res = {"n_frames": len(P[log]["frames"]), "frames_token_in_db": 0, "frames_ts_equal_db": 0, "pairs": 0, "pairs_equal_pkl_vs_db_rule": 0, "pairs_within_120ms": 0, "max_dt_ms": 0.0, "calib_equal": 0, "calib_pairs": 0, "calib_max_abs_diff": 0.0}
    for f in P[log]["frames"]:
        t = f["token"]
        if t in lp:
            res["frames_token_in_db"] += 1; res["frames_ts_equal_db"] += int(lp[t] == f["ts"])
        else:
            continue
        for ci, ch in enumerate(CAMS):
            ts, names = per[ch]
            j = int(np.argmin(np.abs(ts - lp[t]))); dt = abs(int(ts[j]) - int(lp[t])) / 1000.0
            res["pairs"] += 1; res["max_dt_ms"] = max(res["max_dt_ms"], dt); res["pairs_within_120ms"] += int(dt <= 120.0)
            res["pairs_equal_pkl_vs_db_rule"] += int(os.path.basename(names[j]) == os.path.basename(f["jpg"][ci]))
    f0 = P[log]["frames"][0]
    for ci, ch in enumerate(CAMS):
        d = float(np.max(np.abs(np.asarray(f0["K"][ci]) - np.asarray(cal[ch])))); res["calib_pairs"] += 1; res["calib_max_abs_diff"] = max(res["calib_max_abs_diff"], d); res["calib_equal"] += int(d < 1e-6)
    db[log] = res
chk["db_local_logs_checked"] = len(db); chk["db_checks"] = db
chk["db_pairs_total"] = sum(r["pairs"] for r in db.values()); chk["db_pairs_equal"] = sum(r["pairs_equal_pkl_vs_db_rule"] for r in db.values())
json.dump({"checks": chk, "per_log": rows}, open(os.path.join(RAW, "poolB_pkl_summary.json"), "w"))
print(json.dumps({k: v for k, v in chk.items() if k != "per_log_command_is_log_constant_or_mixed"}, indent=1))
