"""Merge label shards -> LABELS_DONE.json (counts) after both DONE_s*.json exist. Reads only. Run with navsim-crun python."""
import json, glob, sys, collections
O = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c/labels/out"
if len(glob.glob(O + "/DONE_s*of2.json")) != 2: sys.exit("shards not both DONE")
c = collections.Counter(); keys = set(); dup = 0
for fp in sorted(glob.glob(O + "/labels_s*of2.jsonl")):
    for line in open(fp, encoding="utf-8"):
        r = json.loads(line); k = (r["log"], r["token"], r["rank"], r["traj_sha256"])
        if k in keys: dup += 1
        keys.add(k)
        c["rank%d_%s" % (r["rank"], "ok" if "tc20s" in r else ("nodb" if r["error"] == "no local DB" else "error"))] += 1
res = {"rows_total": len(keys), "expected": 150603, "duplicates": dup, "counts": dict(c)}
json.dump(res, open(O + "/LABELS_DONE.json", "w"), indent=1); print("ZZLABELS_FINAL", json.dumps(res))
