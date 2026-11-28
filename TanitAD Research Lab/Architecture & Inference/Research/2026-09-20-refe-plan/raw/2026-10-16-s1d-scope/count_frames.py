"""S1d scope probe 2: how many candidate label frames do the 1,167 non-navtrain trainval DBs on the dev box hold?
Method: per DB, count lidar_pc rows (20 Hz) read-only; calibrate the 2 Hz navtrain-token ratio on the 463 navtrain logs that have a local DB
(tokens-in-yaml-per-log / lidar_pc rows). Listing + SELECT count(*) only; nothing is written to a DB. Run with -I."""
import json, os, re, sqlite3, sys, time
from concurrent.futures import ThreadPoolExecutor
PK = "C:/Users/Admin/s1d_scope/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
ROOT = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
INV = json.load(open("C:/Users/Admin/s1d_scope/raw/2026-10-16-s1d-scope/inventory_db.json"))
OUT = "C:/Users/Admin/s1d_scope/raw/2026-10-16-s1d-scope/count_frames.json"

# navtrain tokens per log: the yaml has no log->token map, so use the bank (rank 0) rows: 103,037 distinct (log, token)
from collections import Counter
bank = "D:/Projects/TanitAD-artifacts/_rlfan_work/bank_full/targets_rank0.jsonl"
per_log = Counter(); seen = set()
with open(bank, encoding="utf-8") as fh:
    for l in fh:
        r = json.loads(l); k = (r["log_name"], r["token"])
        if k not in seen:
            seen.add(k); per_log[r["log_name"]] += 1
print("bank logs", len(per_log), "rows", sum(per_log.values()))

def find(n):
    for sub in ("trainval", ""):
        p = f"{ROOT}/{sub}/{n}.db" if sub else f"{ROOT}/{n}.db"
        if os.path.exists(p): return p
    return None

def count(n):
    p = find(n)
    if p is None: return n, None, None, None
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    try:
        c, t0, t1 = con.execute("SELECT count(*), min(timestamp), max(timestamp) FROM lidar_pc").fetchone()
    finally:
        con.close()
    return n, c, (t1 - t0) / 1e6 if c else 0.0, os.path.getsize(p)

neither = INV["neither_trainval_logs"]
nav_with_db = [n for n in per_log if find(n)]
t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    nav = list(ex.map(count, nav_with_db))
    ext = list(ex.map(count, neither))
print("scan seconds", round(time.time() - t0, 1))
ratio_num = sum(per_log[n] for n, c, d, s in nav if c)
ratio_den = sum(c for n, c, d, s in nav if c)
ratio = ratio_num / ratio_den
ext_ok = [(n, c, d, s) for n, c, d, s in ext if c]
res = {
    "navtrain_logs_with_db": len(nav), "navtrain_tokens_in_those_logs_bank_rank0": ratio_num, "navtrain_lidar_pc_rows_in_those_logs": ratio_den,
    "tokens_per_lidar_pc_row": ratio,
    "navtrain_median_tokens_per_log": sorted(per_log.values())[len(per_log) // 2],
    "extra_logs": len(neither), "extra_logs_counted": len(ext_ok),
    "extra_lidar_pc_rows": sum(c for _, c, _, _ in ext_ok), "extra_hours": sum(d for _, _, d, _ in ext_ok) / 3600,
    "extra_db_bytes": sum(s for _, _, _, s in ext_ok),
    "extra_candidate_frames_at_navtrain_ratio": sum(c for _, c, _, _ in ext_ok) * ratio,
    "extra_candidate_frames_at_2hz": sum(d for _, _, d, _ in ext_ok) * 2.0,
    "navtrain_with_db_hours": sum(d for n, c, d, s in nav if c) / 3600,
    "per_db_sample_first5": ext_ok[:5],
}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
