#!/usr/bin/env python3
"""Profile every <log>.pkl in a LOCAL directory with the restricted unpickler of profile_logs.py (no network). Usage: profile_dir.py <dir> <out.json>"""
import sys, os, io, json, hashlib, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_logs import ArrayOnlyUnpickler, profile
d, out = sys.argv[1:3]; res = {}; t0 = time.time()
for f in sorted(os.listdir(d)):
    if not f.endswith(".pkl"):
        continue
    data = open(os.path.join(d, f), "rb").read()
    try:
        p = profile(ArrayOnlyUnpickler(io.BytesIO(data)).load()); p["pkl_bytes"] = len(data); p["pkl_sha256"] = hashlib.sha256(data).hexdigest()
    except Exception as e:
        p = {"error": repr(e)[:300], "pkl_bytes": len(data)}
    res[f[:-4]] = p
json.dump({"dir": d, "profiled": len(res), "seconds": round(time.time() - t0, 1), "logs": res}, open(out, "w"))
print("done", len(res), round(time.time() - t0, 1), "s")
