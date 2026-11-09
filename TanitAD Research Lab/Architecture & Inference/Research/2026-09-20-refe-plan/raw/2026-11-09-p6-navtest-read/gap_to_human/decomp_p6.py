#!/usr/bin/env python3
"""P6's PDMS gap to the logged human on full navtest, same exact decomposition as raw/2026-10-11-ep-gap-decomposition/ep_lowtail.py:
gate term on the human's weight, then EP/TTC/C on the arm's gate. CONTROL: u400 must reproduce the banked 4.15 / 0.86 / 1.31 / -0.01 and 529 zeros, or this exits 1.
Point decomposition, T1-class, one seed, no interval; another term order gives another split. Usage: decomp_p6.py OUT.json. CPU, reads the banked harness CSVs."""
import csv, glob, json, sys
import numpy as np
R = "E:/Projects/TanitAD/data/refe_navtest/score/"
KEYS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "score")
def load(name):
    fs = [f for f in glob.glob(R + name + "/*.csv") if "_logs" not in f]
    assert len(fs) == 1, (name, fs)
    out = {}
    for r in csv.DictReader(open(fs[0], newline="", encoding="utf-8")):
        if r["valid"] == "True" and len(r["token"]) == 16:
            out[r["token"]] = [float(r[k]) for k in KEYS]
    return fs[0].replace("\\", "/"), out
_, h = load("refe_p4_human_navtest")
res = {}
for label, name in (("u400_control", "refe_stage2_sft4_B_u400_navtest_full"), ("P6", "refe_p6final")):
    p, a = load(name)
    toks = sorted(a)
    assert set(toks) == set(h) and len(toks) == 12146, (len(toks), len(h))
    A = np.array([a[t] for t in toks]); H = np.array([h[t] for t in toks])
    gA, gH = A[:, 0] * A[:, 1], H[:, 0] * H[:, 1]
    rA = (5 * A[:, 2] + 5 * A[:, 3] + 2 * A[:, 4]) / 12
    rH = (5 * H[:, 2] + 5 * H[:, 3] + 2 * H[:, 4]) / 12
    assert np.abs(gA * rA - A[:, 5]).max() < 1e-9 and np.abs(gH * rH - H[:, 5]).max() < 1e-9
    terms = dict(gate=((gH - gA) * rH).mean() * 100, ep=(gA * 5 * (H[:, 2] - A[:, 2]) / 12).mean() * 100, ttc=(gA * 5 * (H[:, 3] - A[:, 3]) / 12).mean() * 100, comfort=(gA * 2 * (H[:, 4] - A[:, 4]) / 12).mean() * 100)
    gated = gA > 0
    res[label] = dict(csv=p, n=len(toks), arm=A[:, 5].mean() * 100, human=H[:, 5].mean() * 100, gap=(H[:, 5] - A[:, 5]).mean() * 100, **terms, closure=sum(terms.values()) - (H[:, 5] - A[:, 5]).mean() * 100,
                      zeros=int((A[:, 5] == 0).sum()), dac0_pct=float((A[:, 1] == 0).mean() * 100), nc0_pct=float((A[:, 0] == 0).mean() * 100), nc05_pct=float((A[:, 0] == 0.5).mean() * 100), ttc0_pct=float((A[:, 3] == 0).mean() * 100),
                      ttc0_on_gated_pct=float((A[gated, 3] == 0).mean() * 100), human_ttc0_pct=float((H[:, 3] == 0).mean() * 100), ep_on_gated=float(A[gated, 2].mean() * 100), human_ep_same_tokens=float(H[gated, 2].mean() * 100))
c = res["u400_control"]
ok = abs(c["gap"] - 6.30) < 0.006 and abs(c["gate"] - 4.15) < 0.006 and abs(c["ep"] - 0.86) < 0.006 and abs(c["ttc"] - 1.31) < 0.006 and abs(c["comfort"] + 0.01) < 0.006 and c["zeros"] == 529
res["control_reproduces_banked"] = bool(ok)
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
sys.exit(0 if ok else 1)
