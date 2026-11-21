"""S1d scope probe 6 (read-only): city mix of the 1,213 test-dir DBs that are NOT navtest logs (NOT proposed as training data: same split family as navtest => place-leak risk).
Plus the arithmetic of the minority-city-only extension of the training pool."""
import json, os, re, sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
D = "C:/Users/Admin/s1d_scope/raw/2026-10-16-s1d-scope/"
ROOT = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test/"
nt = json.load(open("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-estimator-and-route-leak/raw/cluster_maps/navtest.json"))["token_to_log_name"]
ntl = set(nt.values())
names = [f[:-3] for f in os.listdir(ROOT) if f.endswith(".db") and f[:-3] not in ntl]
ratio = json.load(open(D + "count_frames.json"))["tokens_per_lidar_pc_row"]
def info(n):
    con = sqlite3.connect(f"file:{ROOT}{n}.db?mode=ro", uri=True)
    try:
        return con.execute("SELECT map_version FROM log").fetchone()[0], con.execute("SELECT count(*) FROM lidar_pc").fetchone()[0]
    finally:
        con.close()
with ThreadPoolExecutor(8) as ex:
    r = list(ex.map(info, names))
logs = Counter(m for m, _ in r); fr = Counter()
for m, c in r: fr[m] += c * ratio
res = {"test_dir_non_navtest_logs": len(names), "by_map_logs": dict(logs), "by_map_frames_at_navtrain_ratio": {k: round(v) for k, v in fr.items()}}
# minority-city-only extension of the navtrain train pool with the 463 trainval logs outside navtrain
pool = json.load(open(D + "sim_s1c_draws.json"))["train_pool"]["frames_by_city"]
ext = json.load(open(D + "extra_logs_by_city.json"))["extra_candidate_frames_by_map_at_navtrain_ratio"]
minor = {k: v for k, v in ext.items() if k != "us-nv-las-vegas-strip"}
comb = {k: pool[k] + minor.get(k, 0) for k in pool}
tot = sum(comb.values())
res["minority_extension"] = {"added_logs": json.load(open(D + "extra_logs_by_city.json"))["extra_logs_by_map"], "added_frames": {k: round(v) for k, v in minor.items()},
                             "pool_after": {k: round(v) for k, v in comb.items()}, "share_after": {k: round(v / tot, 4) for k, v in comb.items()}, "total_after": round(tot)}
nat = {"us-ma-boston": 0.3072, "us-nv-las-vegas-strip": 0.3346, "us-pa-pittsburgh-hazelwood": 0.2006, "sg-one-north": 0.1576}
res["navtest_natural_city_margins"] = nat
json.dump(res, open(D + "test_dir_cities.json", "w"), indent=1)
print(json.dumps(res, indent=1))
