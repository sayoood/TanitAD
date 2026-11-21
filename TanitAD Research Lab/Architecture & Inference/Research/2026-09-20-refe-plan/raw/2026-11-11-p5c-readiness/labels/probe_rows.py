import json, sys, collections, hashlib, glob, os
B = "C:/Users/Admin/p5c_scratch/bank"
sys.path.insert(0, "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe")
out = {}
logs = set()
keys = collections.Counter()
for rk in (0, 1):
    c = collections.Counter(); hz = collections.Counter(); st = collections.Counter(); src = collections.Counter(); lg = set(); bad = 0
    for line in open(f"{B}/targets_rank{rk}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        c["rows"] += 1; hz[r.get("sim_hz")] += 1; st[r.get("step")] += 1; src[r.get("source")] += 1
        lg.add(r["log_name"]); keys[(r["log_name"], r["token"], int(r.get("rank", -1)), r.get("step"))] += 1
        if int(r.get("rank", -1)) != rk: bad += 1
        if len(r["traj"]) != 20: bad += 1
    out[f"rank{rk}"] = {"rows": c["rows"], "sim_hz": dict(hz), "step": dict(st), "source": dict(src), "logs": len(lg), "rank_field_or_len_bad": bad}
    logs |= lg
out["distinct_logs"] = len(logs)
out["dup_keys_log_token_rank_step"] = sum(1 for v in keys.values() if v > 1)
out["rows_with_nonzero_step"] = sum(v for k, v in keys.items() if k[3] != 0)
dbs = {}
for s in ("mini", "test", "trainval"):
    for p in glob.glob(f"E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/{s}/*.db"): dbs.setdefault(os.path.basename(p)[:-3], p)
for p in glob.glob("E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/*.db"): dbs.setdefault(os.path.basename(p)[:-3], p)
miss = sorted(l for l in logs if l not in dbs)
out["db_index_size"] = len(dbs); out["logs_with_db"] = len(logs) - len(miss); out["logs_missing_db"] = len(miss); out["missing_examples"] = miss[:10]
json.dump(out, open("E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c/labels/probe_rows.json", "w"), indent=1)
print(json.dumps(out, indent=1))
