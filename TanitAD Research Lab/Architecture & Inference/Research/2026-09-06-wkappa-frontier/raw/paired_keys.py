#!/usr/bin/env python3
"""What paired deltas do the refav1 records actually carry? ASCII only."""
import json
import os
import sys

P4 = r"C:\Users\Admin\refav1_margin\p4out"
p = os.path.join(P4, "rec_%s.json" % (sys.argv[1] if len(sys.argv) > 1 else "wk7"))
d = json.load(open(p, encoding="utf-8"))
pd = d.get("paired_decision_grade") or {}
print("paired_decision_grade type:", type(pd).__name__)
if isinstance(pd, dict):
    for k, v in list(pd.items())[:8]:
        print(" ", k, "->", json.dumps(v)[:400])
elif isinstance(pd, list):
    for row in pd[:12]:
        print(" ", json.dumps(row)[:300])
print()
arm = d["arms"]["cl"]
print("cl intervals.metrics keys:", sorted((arm.get("intervals", {}).get("metrics") or {}).keys()))
print()
ff = arm["four_families"]
print("lateral keys:", sorted(ff["lateral"].keys()))
print("tactical keys:", sorted(ff["tactical"].keys()))
