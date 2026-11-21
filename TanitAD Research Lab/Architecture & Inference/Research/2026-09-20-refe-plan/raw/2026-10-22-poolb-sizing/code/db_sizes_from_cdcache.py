#!/usr/bin/env python3
"""EXACT nuPlan DB member sizes for pool B from the LOCAL central-directory cache (fetch_front_camera.CD_CACHE), 0 download.
Cache key = sha256(f"{url}|{total}")[:32] (fetch_front_camera.central_directory); url/total from the S3 listing
(raw/s3_list_nuplan_v11.json of the prior probe, re-listed here as raw/s3_list_nuplan_v11_poolb.json). Parses every cached DB-zip CD with the project's own parser."""
import sys, os, json, hashlib, struct
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, "..", "raw")
sys.path.insert(0, RAW)
import importlib.machinery, importlib.util
ldr = importlib.machinery.SourceFileLoader("ffc", os.path.join(RAW, "fetch_front_camera.py.323a13d"))
spec = importlib.util.spec_from_loader("ffc", ldr); ffc = importlib.util.module_from_spec(spec); ldr.exec_module(ffc)
CDC = "D:/Projects/TanitAD/data/nuplan-camera/_cdcache"
BASE = "https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/"
keys = json.load(open(os.path.join(RAW, "s3_list_nuplan_v11_poolb.json")))["keys"]
B = {r["log"]: r for r in json.load(open(os.path.join(RAW, "poolB_logs.json")))}
res = {"zips": {}, "logs": {}}
for k in keys:
    if not k["key"].endswith(".zip") or "maps" in k["key"]:
        continue
    url = BASE + k["key"]; total = k["size"]
    ck = hashlib.sha256(f"{url}|{total}".encode()).hexdigest()[:32]
    p = os.path.join(CDC, ck + ".cd")
    ent = {"url": url, "bytes": total, "cd_cache_key": ck, "cached": os.path.exists(p)}
    if ent["cached"]:
        cd = open(p, "rb").read()
        es = [e for e in ffc.entries(cd) if e["name"].endswith(".db")]
        ent.update({"cd_bytes": len(cd), "n_db": len(es)})
        for e in es:
            log = os.path.basename(e["name"])[:-3]
            if log in B:
                res["logs"].setdefault(log, []).append({"zip": os.path.basename(k["key"]), "name": e["name"], "csz": e["csz"], "usz": e["usz"], "crc32": "%08x" % e["crc"], "method": e["method"], "lho": e["lho"]})
    res["zips"][os.path.basename(k["key"])] = ent
json.dump(res, open(os.path.join(RAW, "db_sizes_poolB.json"), "w"), indent=1)
print({z: (v["cached"], v.get("n_db")) for z, v in res["zips"].items()})
print(len(res["logs"]), "pool-B logs found in cached DB-zip central directories")
c = sum(v[0]["csz"] for v in res["logs"].values()); u = sum(v[0]["usz"] for v in res["logs"].values()); print("compressed", c, "uncompressed", u)
