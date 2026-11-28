#!/usr/bin/env python3
"""Stream the member HEADERS of a LOCAL OpenScene camera tgz (read-only, local file; no network).
Writes {archive, members:[[name,size],...]} in archive order. Used to learn the in-archive order (log -> camera) and the
per-camera byte share of an OpenScene sensor archive, so a prefix-scan cost can be priced."""
import sys, json, tarfile
src, out = sys.argv[1], sys.argv[2]
rows = []
with tarfile.open(src, mode="r|gz") as tf:
    for m in tf:
        if m.isfile():
            rows.append([m.name, m.size])
json.dump({"archive": src, "n": len(rows), "members": rows}, open(out, "w"))
print(len(rows), "members ->", out)
