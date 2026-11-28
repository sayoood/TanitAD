#!/usr/bin/env python
"""mk_tiny_grid.py SRC_GRID.json OUT_GRID.json N_CLIPS -- writes a small grid for a tool check: the first N_CLIPS clips of the source grid (in its order) with their windows, every other
field kept as it is, "n_windows" recounted. Refuses an existing output file. Prints the structure it found (controls only; no outcome is read)."""
import io
import json
import os
import sys

src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
if os.path.exists(out):
    sys.exit("REFUSED: %s exists" % out)
g = json.load(io.open(src, encoding="utf-8"))
print("keys", {k: type(v).__name__ for k, v in g.items()})
order = list(g["order"])[:n]
w = g["windows"]
if isinstance(w, dict):
    w2 = {k: w[k] for k in order}
    cnt = sum(len(v) for v in w2.values())
else:
    sys.exit("REFUSED: the field windows is not a dict (%s)" % type(w).__name__)
t = dict(g)
t["order"], t["windows"] = order, w2
if "n_windows" in t:
    t["n_windows"] = cnt
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
json.dump(t, io.open(out, "w", encoding="utf-8", newline="\n"))
print("wrote %s clips %d windows %d" % (out, len(order), cnt))
