#!/usr/bin/env python3
"""Score the 10 s long-rollout diagnostic (PREREG addendum 3): per arm and rank K, median f(t) at 1..10 s, the share of twins with
f(10 s) >= 0.75 (COMMITS) and the committed reading on K=1: median f(10) >= 0.75 => candidate generator; < 0.5 => no knob of the released
goal pipeline makes this teacher follow the rank-K route.  argv: <dir with *.jsonl>  -> prints JSON."""
import glob, json, os, statistics as st, sys
from collections import defaultdict
D = sys.argv[1]
arms = defaultdict(list)
for p in sorted(glob.glob(os.path.join(D, "*.jsonl"))):
    tag = os.path.basename(p).rsplit("_s", 1)[0]
    for l in open(p, encoding="utf-8"):
        arms[tag].append(json.loads(l))
out = {}
for tag, rows in sorted(arms.items()):
    ok = [r for r in rows if "f_at_s" in r]
    o = {"rows": len(rows), "errors": len(rows) - len(ok)}
    for k in sorted({r["k"] for r in ok}):
        rk = [r for r in ok if r["k"] == k]
        med = [round(st.median(r["f_at_s"][i] for r in rk), 3) for i in range(len(rk[0]["f_at_s"]))]
        o[f"K{k}"] = {"n": len(rk), "logs": len({r["log_name"] for r in rk}), "median_f_1_10s": med,
                      "share_f10_ge_0.75": round(sum(r["f_at_s"][-1] >= 0.75 for r in rk) / len(rk), 3),
                      "share_f10_ge_0.5": round(sum(r["f_at_s"][-1] >= 0.5 for r in rk) / len(rk), 3)}
    k1 = o.get("K1")
    if k1:
        m = k1["median_f_1_10s"][-1]
        o["reading_K1"] = "COMMITS (candidate generator)" if m >= 0.75 else ("NO KNOB (< 0.5)" if m < 0.5 else "BETWEEN 0.5 and 0.75")
    out[tag] = o
print(json.dumps(out, indent=1))
