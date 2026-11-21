"""
ADVERSARIAL PROBE: are nuPlan ego VehicleParameters.length/.width constant across the corpus?

TWO INDEPENDENT PROBES (different mechanisms, not the same command twice):
  A) DEVKIT ROUTE  -- nuplan.database.nuplan_db.nuplan_scenario_queries
                      .get_ego_state_for_lidarpc_token_from_db(log, token)
                      -> ego_state.car_footprint.vehicle_parameters
     This is the route the DB->EgoState pipeline actually takes.
  B) RAW SQLITE ROUTE -- open each .db read-only, scan EVERY table's EVERY column
                      for any name that could carry an ego vehicle dimension, and
                      read log.vehicle_name.

SAME-BREATH CONTROLS (a zero must be a claim about the DATA, not about the SEARCH):
  C1) column-name scan must find the OTHER-AGENT dimension columns (lidar_box.width/
      length/height). If the scan finds those but no ego dimension, the absence is real.
  C2) total column count across tables must be non-zero.
  C3) log.vehicle_name must read non-empty (proves the log table is readable).

Usage: python probe_vehicle_params.py <out.json> [maxlogs]
"""
import json
import os
import sqlite3
import sys
import traceback

DIRS = [
    r"D:\Projects\TanitAD\data\nuplan\dblinks\driverl_val14",
    r"D:\Projects\TanitAD\data\nuplan\nuplan-v1.1\splits\test",
]

# column names that would carry an EGO vehicle dimension if the DB stored one
DIM_PAT = ("width", "length", "wheel", "axle", "height", "dimension",
           "extent", "size", "footprint", "bumper", "cog")


def collect_logs(maxlogs):
    """Stratify by the veh-NN token in the filename so we maximise distinct physical vehicles."""
    by_veh = {}
    for d in DIRS:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".db"):
                continue
            veh = "UNKNOWN"
            for part in fn.split("_"):
                if part.startswith("veh-"):
                    veh = part
                    break
            by_veh.setdefault(veh, []).append(os.path.join(d, fn))
    # round-robin across vehicle ids
    out, i = [], 0
    keys = sorted(by_veh)
    while len(out) < maxlogs:
        added = False
        for k in keys:
            if i < len(by_veh[k]):
                out.append(by_veh[k][i])
                added = True
                if len(out) >= maxlogs:
                    break
        if not added:
            break
        i += 1
    return out, {k: len(v) for k, v in by_veh.items()}


def probe_b_raw_sqlite(path):
    """Independent route: raw sqlite3, no devkit code involved at all."""
    r = {"ok": False}
    con = sqlite3.connect("file:%s?mode=ro" % path.replace("\\", "/"), uri=True)
    try:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        tabs = [x[0] for x in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        total_cols = 0
        dim_cols = []          # every (table, column) whose name could be a dimension
        for t in tabs:
            cols = [x[1] for x in cur.execute("PRAGMA table_info(%s)" % t).fetchall()]
            total_cols += len(cols)
            for c in cols:
                if any(p in c.lower() for p in DIM_PAT):
                    dim_cols.append("%s.%s" % (t, c))
        r["tables"] = sorted(tabs)
        r["n_tables"] = len(tabs)
        r["total_cols"] = total_cols              # CONTROL C2
        r["dim_cols"] = sorted(dim_cols)          # CONTROL C1 lives in here
        # log table: vehicle_name (CONTROL C3) and every other log column
        logcols = [x[1] for x in cur.execute("PRAGMA table_info(log)").fetchall()]
        r["log_cols"] = logcols
        rows = cur.execute("SELECT * FROM log").fetchall()
        r["n_log_rows"] = len(rows)
        r["log_rows"] = [{k: (row[k].hex() if isinstance(row[k], (bytes, bytearray)) else row[k])
                          for k in row.keys()} for row in rows]
        # C1 hard control: other-agent boxes DO carry dimensions -> prove the scan sees them
        if "lidar_box" in tabs:
            lb = [x[1] for x in cur.execute("PRAGMA table_info(lidar_box)").fetchall()]
            r["lidar_box_cols"] = lb
            r["lidar_box_has_dims"] = all(c in lb for c in ("width", "length", "height"))
            row = cur.execute(
                "SELECT width,length,height FROM lidar_box LIMIT 1").fetchone()
            r["lidar_box_sample"] = dict(row) if row is not None else None
            r["lidar_box_n"] = cur.execute(
                "SELECT COUNT(*) FROM lidar_box").fetchone()[0]
        r["ok"] = True
    except Exception:
        r["error"] = traceback.format_exc(limit=3)
    finally:
        con.close()
    return r


def probe_a_devkit(path):
    """The route the code uses: devkit DB->EgoState, then read its vehicle_parameters."""
    from nuplan.database.nuplan_db.nuplan_scenario_queries import (
        get_ego_state_for_lidarpc_token_from_db,
    )
    r = {"ok": False}
    con = sqlite3.connect("file:%s?mode=ro" % path.replace("\\", "/"), uri=True)
    try:
        tok = con.execute("SELECT token FROM lidar_pc LIMIT 1").fetchone()
    finally:
        con.close()
    if tok is None:
        r["error"] = "no lidar_pc rows"
        return r
    token_hex = tok[0].hex() if isinstance(tok[0], (bytes, bytearray)) else str(tok[0])
    r["lidarpc_token"] = token_hex
    try:
        es = get_ego_state_for_lidarpc_token_from_db(path, token_hex)
        vp = es.car_footprint.vehicle_parameters
        r.update({
            "ok": True,
            "length": vp.length,
            "width": vp.width,
            "half_width": vp.half_width,
            "half_length": vp.half_length,
            "front_length": vp.front_length,
            "rear_length": vp.rear_length,
            "wheel_base": vp.wheel_base,
            "rear_axle_to_center": vp.rear_axle_to_center,
            "cog_position_from_rear_axle": vp.cog_position_from_rear_axle,
            "height": vp.height,
            "vehicle_name": vp.vehicle_name,
            "vehicle_type": vp.vehicle_type,
            # VARIANCE CONTROL: pose must DIFFER across logs, proving each probe really
            # opened a DIFFERENT db. A constant here would mean we probed one file 100x.
            "ctrl_rear_axle_x": es.rear_axle.x,
            "ctrl_rear_axle_y": es.rear_axle.y,
            "ctrl_time_us": es.time_us,
        })
    except Exception:
        r["error"] = traceback.format_exc(limit=4)
    return r


def main():
    out_path = sys.argv[1]
    maxlogs = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    logs, veh_census = collect_logs(maxlogs)
    print("corpus census (files per veh id, ALL dirs):", json.dumps(veh_census))
    print("probing %d logs" % len(logs))
    results = []
    for i, p in enumerate(logs):
        rec = {"idx": i, "path": p, "basename": os.path.basename(p),
               "size_bytes": os.path.getsize(p)}
        rec["B_raw_sqlite"] = probe_b_raw_sqlite(p)
        rec["A_devkit"] = probe_a_devkit(p)
        results.append(rec)
        a = rec["A_devkit"]
        b = rec["B_raw_sqlite"]
        vn = (b.get("log_rows") or [{}])[0].get("vehicle_name")
        print("[%3d] %-58s A:len=%s w=%s | B:log.vehicle_name=%s dimcols=%d" % (
            i, rec["basename"][:58],
            a.get("length"), a.get("width"), vn, len(b.get("dim_cols", []))))
        sys.stdout.flush()
    with open(out_path, "w") as f:
        json.dump({"veh_census": veh_census, "n_probed": len(results),
                   "results": results}, f, indent=1)
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
