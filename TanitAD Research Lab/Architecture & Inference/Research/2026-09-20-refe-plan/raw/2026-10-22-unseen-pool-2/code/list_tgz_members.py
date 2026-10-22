#!/usr/bin/env python3
"""List the member names + sizes of a local .tgz (streaming, read-only, no extraction). Used for openscene_metadata_{trainval,mini}.tgz."""
import sys, tarfile, json, time, hashlib
src, out = sys.argv[1], sys.argv[2]
t0 = time.time(); rows = []
with tarfile.open(src, mode="r|gz") as tf:
    for m in tf:
        rows.append({"name": m.name, "size": m.size, "isfile": m.isfile()})
json.dump({"src": src, "n": len(rows), "seconds": round(time.time() - t0, 1), "members": rows}, open(out, "w"))
print("done", len(rows), round(time.time() - t0, 1), "s")
