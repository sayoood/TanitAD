"""Replay rl_s0 metrics.jsonl through the A1 kill-switch (halt_check), step by step."""
import json, sys
import tanitad.rl.ddv2_refcv7 as R
print("[replay] halt_check from", R.__file__)
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
hist, first = [], {}
for row in rows:
    if not all(k in row for k in R.HALT_KEYS):
        continue
    hist.append({k: row[k] for k in R.HALT_KEYS})
    for name, kw in (("K1", dict(gn_median=5.0, oob_median=0, pdn=0)),
                     ("K2", dict(gn_median=0, oob_median=0.40, pdn=0)),
                     ("K3", dict(gn_median=0, oob_median=0, pdn=3.0))):
        if name not in first and R.halt_check(hist, window=20, **kw):
            first[name] = row.get("step")
    if "ALL" not in first:
        msg = R.halt_check(hist, gn_median=5.0, oob_median=0.40, pdn=3.0, window=20)
        if msg:
            first["ALL"] = (row.get("step"), msg)
print("rows", len(rows), "with halt keys", len(hist))
for k in ("K1", "K2", "K3", "ALL"):
    print(k, first.get(k))
