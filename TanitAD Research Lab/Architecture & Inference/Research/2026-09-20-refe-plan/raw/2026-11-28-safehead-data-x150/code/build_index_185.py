#!/usr/bin/env python3
"""Build the 185-log index = old index (35 logs, F:/safehead_cache/index.csv, read only) + x150 index (rows of logs whose npz re-hashes to its merge.json sha).
    build_index_185.py [--out F:/safehead_cache/x150/index_185.csv]
Reads only; writes --out (default under x150/). Fails loudly on a duplicate log, a missing file or a sha mismatch. The two cache roots stay separate
(the index "file" column is made absolute: old rows F:/safehead_cache/out/..., new rows F:/safehead_cache/x150/out/...); sealing may copy them into one root afterwards."""
import argparse
import csv
import hashlib
import json
import os
import sys

OLD, NEW = "F:/safehead_cache/index.csv", "F:/safehead_cache/x150/index.csv"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="F:/safehead_cache/x150/index_185.csv")
a = ap.parse_args()
rows, seen, files = [], {}, {}
for src in (OLD, NEW):
    for r in csv.DictReader(open(src)):
        if r["log"] in seen and seen[r["log"]] != src:
            sys.exit(f"DUPLICATE log across indexes: {r['log']}")
        seen[r["log"]] = src
        root = os.path.dirname(src) + "/out/"
        files[root + r["file"]] = r["sha256"]
        rows.append(dict(r, file=root + r["file"]))   # absolute path; the 35 old rows keep pointing at F:/safehead_cache/out
for f, h in files.items():
    if not os.path.exists(f) or sha(f) != h:
        sys.exit(f"MISSING/SHA MISMATCH {f}")
toks = [r["token"] for r in rows]
assert len(toks) == len(set(toks)), "duplicate token"
with open(a.out, "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=["token", "log", "split", "file", "sha256"])
    w.writeheader()
    w.writerows(rows)
print(json.dumps({"logs": len(seen), "tokens": len(rows), "heldout_logs": len({r["log"] for r in rows if r["split"] == "heldout"}), "out": a.out}))
