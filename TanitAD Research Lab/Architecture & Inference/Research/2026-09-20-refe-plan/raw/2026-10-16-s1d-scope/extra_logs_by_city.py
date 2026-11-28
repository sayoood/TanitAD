"""S1d scope probe 4: city composition of the 1,167 non-navtrain trainval DBs (read-only: SELECT map_version FROM log + lidar_pc count).
Compares against the navtrain train pool (sim_s1c_draws.json train_pool) and navtest's city margins (S1c RESULT.md). Run with -I."""
import json, os, sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
ROOT = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
D = "C:/Users/Admin/s1d_scope/raw/2026-10-16-s1d-scope/"
inv = json.load(open(D + "inventory_db.json"))
cnt = json.load(open(D + "count_frames.json"))
ratio = cnt["tokens_per_lidar_pc_row"]

def info(n):
    p = f"{ROOT}/trainval/{n}.db"
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    try:
        m = con.execute("SELECT map_version FROM log").fetchone()[0]
        c = con.execute("SELECT count(*) FROM lidar_pc").fetchone()[0]
    finally:
        con.close()
    return n, m, c
with ThreadPoolExecutor(8) as ex:
    rows = list(ex.map(info, inv["neither_trainval_logs"]))
logs = Counter(m for _, m, _ in rows)
fr = defaultdict(float)
for _, m, c in rows:
    fr[m] += c * ratio
res = {"extra_logs_by_map": dict(logs), "extra_candidate_frames_by_map_at_navtrain_ratio": {k: round(v) for k, v in fr.items()},
       "total_frames": round(sum(fr.values()))}
tp = json.load(open(D + "sim_s1c_draws.json"))["train_pool"]
tot = sum(tp["frames_by_city"].values())
res["navtrain_pool_frames_by_map"] = tp["frames_by_city"]
res["share_pool"] = {k: round(v / tot, 4) for k, v in tp["frames_by_city"].items()}
res["share_extra"] = {k: round(v / sum(fr.values()), 4) for k, v in fr.items()}
combo = {k: tp["frames_by_city"].get(k, 0) + fr.get(k, 0) for k in set(tp["frames_by_city"]) | set(fr)}
res["share_pool_plus_extra"] = {k: round(v / sum(combo.values()), 4) for k, v in combo.items()}
json.dump(res, open(D + "extra_logs_by_city.json", "w"), indent=1)
print(json.dumps(res, indent=1))
