#!/usr/bin/env python3
"""Search the dev box (C:, D:, E:) for ANY copy of a pool-B nuPlan DB (<log>.db) or a zip/archive named after a pool-B log.
Read-only walk. Writes raw/find_poolb_dbs.json {roots, n_dirs, n_files, db_hits:{log:[path,size]}, any_name_hits:[...], other_nuplan_db_dirs:{dir:count}}."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, "..", "raw")
B = {r["log"] for r in json.load(open(os.path.join(RAW, "poolB_logs.json")))}
roots = sys.argv[1:] or ["C:/", "D:/", "E:/"]
SKIP = ("C:/Windows", "C:/Program Files", "C:/Program Files (x86)", "C:/ProgramData/Microsoft", "C:/$Recycle.Bin", "C:/System Volume Information", "D:/$RECYCLE.BIN", "D:/System Volume Information")
out = {"roots": roots, "db_hits": {}, "any_name_hits": [], "db_dirs": {}, "n_files": 0, "n_dirs": 0, "errors": 0}
t0 = time.time()
for root in roots:
    if not os.path.exists(root):
        out.setdefault("missing_roots", []).append(root); continue
    for dp, dn, fn in os.walk(root, onerror=lambda e: out.__setitem__("errors", out["errors"] + 1)):
        dpn = dp.replace("\\", "/")
        if any(dpn.startswith(s) for s in SKIP):
            dn[:] = []; continue
        out["n_dirs"] += 1; out["n_files"] += len(fn)
        for f in fn:
            if f.endswith(".db"):
                out["db_dirs"][dpn] = out["db_dirs"].get(dpn, 0) + 1
            stem = f[:-3] if f.endswith(".db") else f
            base = f.split(".db")[0] if ".db" in f else None
            if f.endswith(".db") and stem in B:
                try: sz = os.path.getsize(os.path.join(dp, f))
                except OSError: sz = None
                out["db_hits"].setdefault(stem, []).append([dpn + "/" + f, sz])
            else:
                for l in B:
                    if l in f:
                        out["any_name_hits"].append(dpn + "/" + f); break
json.dump(out, open(os.path.join(RAW, "find_poolb_dbs.json"), "w"), indent=0)
print(len(out["db_hits"]), "pool-B logs with a DB copy;", len(out["any_name_hits"]), "other files named after a pool-B log;", out["n_files"], "files scanned in", round(time.time() - t0), "s")
