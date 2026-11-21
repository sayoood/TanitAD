#!/usr/bin/env python3
"""Read argv / plan / cost blocks out of refav1_arm records. ASCII only (cp1252 dev box)."""
import json
import os
import sys

P4 = r"C:\Users\Admin\refav1_margin\p4out"
names = sys.argv[1:] or ["gkappa", "wk15", "ta_wk15_s0", "ta_wk15_s1", "wk7", "best", "best_seed1"]
for n in names:
    p = os.path.join(P4, "rec_%s.json" % n)
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print("=====", n, "ERR", e)
        continue
    print("=====", n, "arm=", d.get("arm"), "n_windows=", d.get("n_windows"))
    print("  plan:", json.dumps(d.get("plan", {}))[:260])
    c = d.get("cost", {})
    print("  cost.metric=", c.get("metric"), " weights=", json.dumps(c.get("weights", {})),
          " src=", c.get("weights_source"))
    print("  cost.w_kappa_by_goal=", json.dumps(c.get("w_kappa_by_goal")))
    argv = d.get("argv") or d.get("config", {}).get("argv")
    if argv:
        print("  ARGV:", " ".join(str(x) for x in argv))
    else:
        print("  (no argv key) top keys:", sorted(d.keys()))
