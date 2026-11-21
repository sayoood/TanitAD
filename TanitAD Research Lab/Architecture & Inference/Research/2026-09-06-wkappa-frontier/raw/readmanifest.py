#!/usr/bin/env python3
"""Dump a refav1 dump manifest so the launch command can be reconstructed. ASCII only."""
import json
import sys

p = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\Users\Admin\refav1_margin\p4out\dump_gkappa\manifest.json"
d = json.load(open(p, encoding="utf-8"))
for k in ("tool", "model", "corpus", "grid", "plan_cfg", "cost", "goal", "goal_rule",
          "goal_vocab", "action_units", "nav_shuffle", "arms", "tiers", "speed_channel",
          "hold_action_rule", "hold_v0_rule", "arm_meaning"):
    v = d.get(k)
    s = json.dumps(v)
    if len(s) > 1200:
        s = s[:1200] + " ...TRUNC"
    print("== %-16s %s" % (k, s))
eps = d.get("episodes")
if isinstance(eps, list):
    print("== episodes  n=%d  first=%s" % (len(eps), json.dumps(eps[0])[:400]))
else:
    print("== episodes  %s" % json.dumps(eps)[:400])
