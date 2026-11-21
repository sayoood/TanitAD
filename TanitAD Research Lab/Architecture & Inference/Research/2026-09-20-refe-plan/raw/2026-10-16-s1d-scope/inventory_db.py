"""S1d scope probe 1: which nuPlan DBs on the dev box are navtrain / navtest / neither? (read-only, listing only)
Inputs: PK/splits/navtrain.yaml (log_names block), navtest.json (tracked), the DB dirs under D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits."""
import os, re, json
PK = "C:/Users/Admin/s1d_scope/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
ROOT = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
NT_JSON = "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-estimator-and-route-leak/raw/cluster_maps/navtest.json"
OUT = "C:/Users/Admin/s1d_scope/raw/2026-10-16-s1d-scope/inventory_db.json"

txt = open(PK + "/splits/navtrain.yaml", encoding="utf-8").read()
sec = txt.split("log_names:", 1)[1].split("\ntokens:", 1)[0]     # only the log_names block (the file also carries 103,288 token entries)
tr_logs = re.findall(r"^\s+- '([^']+)'", sec, re.M)
tr = set(tr_logs)
print("navtrain logs (yaml log_names block): list", len(tr_logs), "unique", len(tr))

def logs_of(o, acc):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(k, str) and re.match(r"^\d{4}\.\d\d\.\d\d\.", k): acc.add(k)
            logs_of(v, acc)
    elif isinstance(o, list):
        for v in o: logs_of(v, acc)
    elif isinstance(o, str):
        m = re.match(r"^(\d{4}\.\d\d\.\d\d\.\d\d\.\d\d\.\d\d_veh-\d+_\d+_\d+)", o)
        if m: acc.add(m.group(1))
nt = set(); logs_of(json.load(open(NT_JSON)), nt)
print("navtest logs (json):", len(nt))

dirs = {}; found = {}
for sub in ("", "trainval", "test", "mini"):
    p = ROOT + ("/" + sub if sub else "")
    names = [f[:-3] for f in os.listdir(p) if f.endswith(".db")]
    dirs[sub or "flat"] = {"n": len(names), "navtrain": sum(n in tr for n in names),
                           "navtest": sum(n in nt for n in names),
                           "neither": sum((n not in tr and n not in nt) for n in names)}
    for n in names: found.setdefault(n, []).append(sub or "flat")
print(json.dumps(dirs, indent=1))
nav_with_db = {n for n in found if n in tr}
held = {n for n, w in found.items() if any(x in ("flat", "trainval") for x in w)}
neither = sorted(n for n in held if n not in tr and n not in nt)
print("distinct navtrain logs with a local DB:", len(nav_with_db), "of", len(tr))
print("flat+trainval distinct DBs:", len(held), "| navtrain", sum(n in tr for n in held),
      "| navtest", sum(n in nt for n in held), "| neither (trainval logs outside navtrain)", len(neither))
json.dump({"navtrain_yaml_logs": len(tr), "navtest_json_logs": len(nt), "dirs": dirs,
           "navtrain_with_local_db": len(nav_with_db), "flat_trainval_distinct": len(held),
           "neither_trainval_logs_n": len(neither), "neither_trainval_logs": neither}, open(OUT, "w"), indent=1)
