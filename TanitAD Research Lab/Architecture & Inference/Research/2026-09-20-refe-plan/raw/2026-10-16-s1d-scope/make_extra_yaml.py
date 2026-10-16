"""S1d: write a navtrain-style token-universe yaml for logs that are NOT in navtrain, so `refe/aux_label_store.py logmap/build` can label them
through its DB-origin path (the path the 102 DB-only navtrain tokens already use).  Read-only on the nuPlan DBs.

Token policy (S1D-TOK): per log, lidar_pc rows ordered by timestamp, every STRIDE-th row starting at STRIDE//2 (default 60 rows = 3.0 s at 20 Hz).
Why 60: on the 462 navtrain logs with a local DB the navtrain tokens are 0.0165 of the lidar_pc rows (count_frames.json, MEASURED) = one frame per ~60.8 rows,
so stride 60 reproduces navtrain's temporal density (~0.33 Hz).  NOT NAVSIM's scene filter (has_route / history / future-window rules are not applied): UNVERIFIED
whether the trunk/scorer would accept every such frame; a head-only consumer needs only the image + calibration.

  python make_extra_yaml.py --logs-json inventory_db.json[:key] --out extra.yaml [--stride 60] [--limit N] [--control-navtrain LOG]
"""
import argparse, json, os, sqlite3, sys
ROOT = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"

def db_path(log):
    for sub in ("trainval", ""):
        p = f"{ROOT}/{sub}/{log}.db" if sub else f"{ROOT}/{log}.db"
        if os.path.exists(p): return p
    raise FileNotFoundError(log)

def tokens(log, stride):
    con = sqlite3.connect(f"file:{db_path(log)}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT lower(hex(token)) FROM lidar_pc ORDER BY timestamp").fetchall()
    finally:
        con.close()
    return [r[0] for r in rows][stride // 2::stride]

ap = argparse.ArgumentParser()
ap.add_argument("--inventory", default="C:/Users/Admin/s1d_scope/raw/2026-10-16-s1d-scope/inventory_db.json")
ap.add_argument("--logs", default=None, help="comma-separated log names (overrides the inventory's non-navtrain list)")
ap.add_argument("--out", required=True)
ap.add_argument("--stride", type=int, default=60)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
logs = a.logs.split(",") if a.logs else json.load(open(a.inventory))["neither_trainval_logs"]
if a.limit: logs = logs[:a.limit]
with open(a.out, "w", encoding="utf-8", newline="\n") as f:
    f.write("_target_: navsim.common.dataclasses.SceneFilter\n_convert_: 'all'\nlog_names:\n")
    for l in logs: f.write(f"  - '{l}'\n")
    f.write("tokens:\n")
    n = 0
    for l in logs:
        for t in tokens(l, a.stride):
            f.write(f"  - '{t}'\n"); n += 1
print(f"wrote {a.out}: {len(logs)} logs, {n} tokens (stride {a.stride})")
