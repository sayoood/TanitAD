import json, os, collections
from pathlib import Path
P = "/workspace/data/refe_navtrain/train_grow_cmd/targets_rank1.jsonl"
MC = "/workspace/data/refe_navtrain/metric_cache"
MCR = "/workspace/data/refe_navtrain/metric_cache_twins_rejected"
DB = "/workspace/data/navtrain_dbs"
keys = []
for l in open(P):
    if '"lane_rank"' not in l: continue
    r = json.loads(l); a = r["aug"]
    if a.get("kind") == "lane_rank":
        keys.append((r["log_name"], r["token"], int(a["value"]), r.get("step")))
print("twins", len(keys))
byk = collections.Counter(k[2] for k in keys); print(dict(byk))
logs = collections.Counter(k[0] for k in keys); print("logs", len(logs), "twins/log mean", sum(logs.values())/len(logs), "max", max(logs.values()), "min", min(logs.values()))
nolog = sum(1 for k in keys if not (Path(MC)/k[0]/f"{k[1]}.pkl.xz").is_file())
print("missing LOGGED parts for twin frames:", nolog)
norej = sum(1 for k in keys if not (Path(MCR)/k[0]/f"{k[1]}__lane_rank_{k[2]}.pkl.xz").is_file())
print("missing rejected-cache twin parts:", norej)
print("db root sample:", sorted(os.listdir(DB))[:3], len(os.listdir(DB)))
nodb = [lg for lg in logs if not any(Path(DB).glob(lg + "*"))]
print("logs with no db entry (glob):", len(nodb), nodb[:3])
# lpt balance
sizes = sorted(logs.values(), reverse=True)
for N in (90, 360):
    load = [0]*N
    for s in sizes:
        i = load.index(min(load)); load[i] += s
    print("LPT N", N, "max", max(load), "mean", sum(load)/N)
