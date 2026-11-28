"""check_bmax.py -- tool check of the braking limit of the lead rule before card PREREG_V8F_PROOF_ADD57 is frozen. No model, no GPU, no data of any card: synthetic straight
plans behind a slower lead. Shows that dist_rule.distance_rule takes b_max, that the executed plan never brakes harder than b_max, and that the number of saturated executed
steps (the cap would need more braking than the limit allows) falls as the limit rises: the direction control b5 of the card relies on.
Run on the pod: DISTRULE_DIR=<build> python check_bmax.py <wrapper.py>     Last line: ZZ CHECK-BMAX all hold: True/False"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.environ.get("DISTRULE_DIR", os.path.dirname(os.path.abspath(__file__))))
import dist_rule as R  # noqa: E402

DT = 0.2
N = 30
rng = np.random.default_rng(57)
B = 400
v0 = rng.uniform(4.0, 25.0, B)
sp = np.clip(v0[:, None] + np.cumsum(rng.normal(0.0, 0.25, (B, N)), 1), 0.0, None)
x = np.cumsum(sp * DT, 1)
plan = np.stack([x, np.zeros_like(x)], -1)
gap = rng.uniform(4.0, 45.0, B)
v_L = np.clip(v0 - rng.uniform(0.0, 12.0, B), 0.0, None)
present = np.ones(B, bool)
ok_all = True
res = {}
for bm in (6.0, 8.0, 10.0):
    out = R.distance_rule(plan, v0, present, gap, gap, v_L, m=10, b_max=bm, floor_lifts=False)
    p = np.asarray(out.plan, np.float64)
    s = np.diff(np.concatenate([np.zeros((B, 1)), p[:, :, 0]], 1), axis=1) / DT
    prev = np.concatenate([v0[:, None], s[:, :-1]], 1)
    dec = ((prev - s) / DT)[out.changed]                                       # braking between steps of the changed plans, m/s2
    wish = ((prev - sp) / DT)                                                  # what the plan itself asks for from the rule's speed before
    hardest_own = float(np.max(np.concatenate([v0[:, None], sp[:, :-1]], 1) - sp)) / DT
    res[bm] = {"sat_exec": int(out.n_sat_exec.sum()), "rows_sat": int((out.n_sat_exec > 0).sum()), "changed": int(out.changed.sum()), "hardest": float(dec.max()), "per_row": out.n_sat_exec.copy()}
    good = res[bm]["hardest"] <= max(bm, hardest_own) + 1e-6
    ok_all = ok_all and good
    print("b_max %4.1f: plans changed %d of %d, saturated executed steps %d in %d rows, hardest braking in a changed plan %.3f m/s2 (the plans' own hardest %.3f) -> %s" % (
        bm, res[bm]["changed"], B, res[bm]["sat_exec"], res[bm]["rows_sat"], res[bm]["hardest"], hardest_own, "holds" if good else "DIFFERENT"))
fall = res[6.0]["sat_exec"] > res[8.0]["sat_exec"] > res[10.0]["sat_exec"] and res[6.0]["sat_exec"] > 0
rows_up = int((res[8.0]["per_row"] > res[6.0]["per_row"]).sum())
print("saturated executed steps fall as the limit rises: %s (rows where 8.0 has more than 6.0: %d of %d)" % (fall, rows_up, B))
ok_all = ok_all and fall
d68 = float(np.abs(np.asarray(R.distance_rule(plan, v0, present, gap, gap, v_L, m=10, b_max=6.0, floor_lifts=False).plan) - np.asarray(R.distance_rule(plan, v0, present, gap, gap, v_L, m=10, floor_lifts=False).plan)).max())
print("b_max 6.0 given as a parameter equals the default: %s" % (d68 == 0.0))
ok_all = ok_all and d68 == 0.0 and R.B_MAX == 6.0
src = open(sys.argv[1], encoding="utf-8").read()
wr = bool(re.search(r'rule_kw\["b_max"\] = float\(mine\.b_max\)', src)) and '"B_MAX": float(rule_kw.get("b_max", R.B_MAX))' in src and 'W.MODE_CFG["mix"] = ("mix", dict(rule_kw))' in src
print("the wrapper passes --b-max into the rule's keyword arguments and records it: %s" % wr)
ok_all = ok_all and wr
print("ZZ CHECK-BMAX all hold: %s" % ok_all)
