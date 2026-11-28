"""Aggregate probe_120.json -> the distinct value sets + control readouts."""
import json
import sys
from collections import Counter

d = json.load(open(sys.argv[1]))
res = d["results"]
print("=" * 78)
print("LOGS PROBED: %d   (corpus on disk: %d db files across %d veh ids)" % (
    len(res), sum(d["veh_census"].values()), len(d["veh_census"])))
print("=" * 78)

# ---- PROBE A: devkit route ---------------------------------------------
okA = [r["A_devkit"] for r in res if r["A_devkit"].get("ok")]
failA = [r for r in res if not r["A_devkit"].get("ok")]
print("\n[PROBE A] devkit get_ego_state_for_lidarpc_token_from_db -> "
      "car_footprint.vehicle_parameters")
print("  ok=%d  failed=%d" % (len(okA), len(failA)))
for f in failA[:3]:
    print("   FAIL", f["basename"], f["A_devkit"].get("error", "")[:200])
FIELDS = ["length", "width", "half_width", "half_length", "front_length",
          "rear_length", "wheel_base", "rear_axle_to_center",
          "cog_position_from_rear_axle", "height", "vehicle_name", "vehicle_type"]
for k in FIELDS:
    vals = Counter(a.get(k) for a in okA)
    print("   %-28s distinct=%d  ->  %s" % (
        k, len(vals), dict(list(vals.items())[:4])))

# ---- VARIANCE CONTROL: did we really open DIFFERENT dbs? ---------------
print("\n[CONTROL V] pose/time MUST vary across logs (else we probed one file N times)")
for k in ["ctrl_rear_axle_x", "ctrl_rear_axle_y", "ctrl_time_us"]:
    vals = {a.get(k) for a in okA}
    print("   %-20s distinct=%d / %d   (must be ~N)" % (k, len(vals), len(okA)))

# ---- PROBE B: raw sqlite, independent of devkit ------------------------
okB = [r["B_raw_sqlite"] for r in res if r["B_raw_sqlite"].get("ok")]
print("\n[PROBE B] raw sqlite3, devkit code NOT involved")
print("  ok=%d / %d" % (len(okB), len(res)))
vn = Counter(b["log_rows"][0].get("vehicle_name") for b in okB if b.get("log_rows"))
print("  CONTROL C3 log.vehicle_name distinct=%d -> %s" % (len(vn), dict(vn)))
locs = Counter(b["log_rows"][0].get("location") for b in okB if b.get("log_rows"))
print("  log.location distinct=%d -> %s" % (len(locs), dict(locs)))
logcols = Counter(tuple(b["log_cols"]) for b in okB)
print("  log table columns (distinct schemas=%d): %s" % (
    len(logcols), list(logcols)[0]))

# the key negative, with its control
EGO_DIM = [c for c in sorted({c for b in okB for c in b["dim_cols"]})]
print("\n  ALL dimension-named columns found anywhere in the schema:")
for c in EGO_DIM:
    print("     ", c)
ego_named = [c for c in EGO_DIM if c.split(".")[0] in ("ego_pose", "log", "scene",
                                                       "lidar_pc")]
print("  -> of those, belonging to an EGO table (ego_pose/log/scene/lidar_pc): %d"
      % len(ego_named), ego_named)
print("  CONTROL C1 (scan CAN see dimension cols): lidar_box has w/l/h in %d/%d logs"
      % (sum(1 for b in okB if b.get("lidar_box_has_dims")), len(okB)))
print("  CONTROL C1 sample OTHER-AGENT dims (per-object, real data): %s"
      % okB[0].get("lidar_box_sample"))
print("  CONTROL C2 total columns per db: %s" % sorted({b["total_cols"] for b in okB}))
print("  lidar_box row counts (non-zero proves tables are populated): min=%d max=%d"
      % (min(b.get("lidar_box_n", 0) for b in okB),
         max(b.get("lidar_box_n", 0) for b in okB)))

# ---- cross-probe agreement --------------------------------------------
print("\n[CROSS] vehicle ids covered by the probe: %d" % len(vn))
print("[CROSS] for every log, does devkit vp.vehicle_name == db log.vehicle_name?")
mism = 0
for r in res:
    a, b = r["A_devkit"], r["B_raw_sqlite"]
    if a.get("ok") and b.get("ok") and b.get("log_rows"):
        if a.get("vehicle_name") != b["log_rows"][0].get("vehicle_name"):
            mism += 1
print("   MISMATCH in %d / %d logs  <-- devkit says 'pacifica', db says 'veh-NN'"
      % (mism, len(res)))
