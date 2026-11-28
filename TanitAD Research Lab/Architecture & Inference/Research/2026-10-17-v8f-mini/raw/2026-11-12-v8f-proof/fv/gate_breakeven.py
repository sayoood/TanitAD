"""Group J applied to the measured class means of HIERWHERE / HIERWHERE-2 (banked, T1-kin). ESTIMATED: equal-variance binormal flag
model plus premise A-J (homogeneity). Post hoc and descriptive; decides nothing."""
import json
from statistics import NormalDist
ND = NormalDist()
P = 1305 / 2399
CELLS = {"P1_s0": (-0.4795, -0.1847), "P1_s1": (-0.1589, 0.0702), "P2_s0": (-0.1817, 0.3180), "P2_s1": (-0.1196, 0.4928)}
MEASURED_AUROC = {"P1_s0": 0.6349, "P1_s1": 0.6393, "P2_s0": 0.6368, "P2_s1": 0.6424}
GRID = [i / 2000 for i in range(1, 2000)]


def best_gain(dc, db, auroc):
    """Most negative (composite minus better arm) over thresholds of a binormal flag with this AUROC."""
    dp = 2 ** 0.5 * ND.inv_cdf(auroc)
    D = P * dc + (1 - P) * db
    best = 0.0
    for f in GRID:
        t = ND.cdf(dp + ND.inv_cdf(f))
        DF = P * t * dc + (1 - P) * f * db
        DH = -P * (1 - t) * dc - (1 - P) * (1 - f) * db
        g = DF if D > 0 else DH          # better arm = flat when D > 0, else the hierarchy
        best = min(best, g)
    return best


out = {"p_critical": P, "cells": {}}
for k, (dc, db) in CELLS.items():
    D = P * dc + (1 - P) * db
    row = {"dc": dc, "db": db, "D_hier_minus_flat": round(D, 4)}
    if db <= 0:
        row["case"] = "hierarchy wins both classes (VC-J7): no commit rule can beat it"
        row["oracle_minus_hier"] = round(-(1 - P) * db, 4)
    else:
        s = (1 - P) * db / (P * -dc)
        oracle = P * dc if D > 0 else -(1 - P) * db
        row.update({"s_bulk_cost_per_critical_gain": round(s, 3), "better_arm": "flat" if D > 0 else "hier",
                    "oracle_gain_over_better_arm_m": round(oracle, 4),
                    "need": ("t/f > s (positive likelihood ratio)" if s > 1 else "(1-t)/(1-f) < s (negative likelihood ratio)")})
        g_meas = best_gain(dc, db, MEASURED_AUROC[k])
        row["best_gain_at_measured_auroc_m"] = round(g_meas, 4)
        row["share_of_oracle_at_measured_auroc"] = round(g_meas / oracle, 3)
        need = {}
        for share in (0.25, 0.5, 0.75):
            a = None
            for i in range(51, 100):
                if best_gain(dc, db, i / 100) <= share * oracle:
                    a = i / 100
                    break
            need[str(share)] = a
        row["auroc_needed_for_share_of_oracle"] = need
    out["cells"][k] = row
with open("F:/TanitAD-scratch/v8proof/fv/GATE_BREAKEVEN.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
for k, r in out["cells"].items():
    print(k, json.dumps(r))
