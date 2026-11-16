"""Planner-side in-run eval (T0 teacher-forced losses, eval139 --eval-batches 8): RAW vs EMA per arm. Read-only."""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
RUNS = {"K0": "thor/r9e2-K0/metrics.jsonl", "K2M1N20": "thor/r9e2-K2M1N20/metrics.jsonl", "K2DC": "thor/r9e2-K2DC/metrics.jsonl"}
KEYS = ["lat", "lon", "lat_tac", "anchor_acc", "goal2s_err_m", "sel_v3", "r8_drv", "cascade"]
res = {}
for arm, p in RUNS.items():
    R = [json.loads(l) for l in open(os.path.join(BASE, p), encoding="utf-8") if l.strip()]
    ev = [r for r in R if "eval_loss" in r]
    res[arm] = {}
    print("\n##", arm, "  columns: key raw/ema (ema-raw)")
    for r in ev:
        s = r["step"]
        if s not in (1000, 3000, 5000, 5500, 6000):
            continue
        cells = []
        for k in KEYS:
            a, b = r.get("eval_" + k), r.get("ema_eval_" + k)
            res[arm].setdefault(str(s), {})[k] = [a, b]
            if a is None or b is None:
                cells.append("%s -" % k)
            else:
                cells.append("%s %.4f/%.4f (%+.4f)" % (k, a, b, b - a))
        print(" %5d | %s" % (s, " | ".join(cells)))
json.dump(res, open(os.path.join(BASE, "plan_traj.json"), "w"), indent=1)
