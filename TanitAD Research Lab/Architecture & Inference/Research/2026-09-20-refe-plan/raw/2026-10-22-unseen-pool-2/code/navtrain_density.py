#!/usr/bin/env python3
"""Per-log frame / route / anchor counts for navtrain logs from TanitAstra's compact_metadata (local) joined to the repo's navtrain token table:
gives tokens-per-eligible-anchor, used to ESTIMATE how many NAVSIM-style tokens an unseen log would carry. Read-only."""
import json, gzip, csv, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CM = "D:/TanitAstraResearch/data/e024_corpus_20261010/compact_metadata/"
tok = collections.Counter()
with gzip.open("raw/src/navtrain_token_table.tsv.gz", "rt", newline="") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["log"]:
            tok[r["log"]] += 1
nt = {l.strip() for l in open("raw/src/navtrain_logs.txt") if l.strip()}
out = {}
for fn in sorted(os.listdir(CM)):
    if not fn.endswith(".json"):
        continue
    seg = fn[:-5]
    if seg not in nt:
        continue
    d = json.load(open(CM + fn))
    fr = d["frames"]; n = len(fr)
    route = [bool(f and f.get("has_route")) for f in fr]
    ts = [f["timestamp"] if f else None for f in fr]
    out[seg] = {"n_frames": n, "n_route_frames": sum(route), "anchors_geom": max(0, n - 13), "anchors_route": sum(1 for j in range(3, n - 10) if route[j]),
                "tokens": tok.get(seg, 0)}
json.dump(out, open("raw/navtrain_density.json", "w"))
A = sum(v["anchors_geom"] for v in out.values()); R = sum(v["anchors_route"] for v in out.values()); T = sum(v["tokens"] for v in out.values())
print("navtrain logs with compact metadata:", len(out), "of", len(nt), "frames", sum(v["n_frames"] for v in out.values()), "anchors_geom", A, "anchors_route", R, "tokens", T, "tokens/anchors_route", round(T / R, 4), "tokens/anchors_geom", round(T / A, 4))
noroute = [s for s, v in out.items() if v["n_route_frames"] == 0]
print("navtrain logs with zero route frames:", len(noroute))
