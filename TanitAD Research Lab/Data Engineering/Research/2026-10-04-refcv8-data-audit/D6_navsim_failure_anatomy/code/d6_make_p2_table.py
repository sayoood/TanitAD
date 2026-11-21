"""D6 -- render raw/d6_p2_tables.md from raw/d6_p2.json (generated, not hand-copied)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")
d = json.load(open(os.path.join(RAW, "d6_p2.json"), encoding="utf-8"))
L = []
w = L.append
w("| arm | stratum | n | DAC0 arm / A1 | delta pp [95 % log CI] (seed floor) | resolved | NC0 arm / A1 (delta pp) | command compliance arm / A1 (delta pp) | WRONG-SIDE / ROUTE-FOLLOWING / OVER-STEER share: arm vs A1 |")
w("|---|---|---|---|---|---|---|---|---|")
for arm, r in d["arms"].items():
    for st in ("S_prem", "S_turn", "P2_all"):
        v = r["strata"][st]
        a, n, c, cs = v["DAC0"], v["NC0"], v.get("compliance"), v["class_share"]
        comp = f"{100*c['arm']:.1f} / {100*c['A1']:.1f} ({c['delta_pp']:+.1f})" if c else "n/a"
        cls = " / ".join(f"{100*cs[k]['arm']:.1f} vs {100*cs[k]['A1']:.1f}" for k in ("WRONG-SIDE", "ROUTE-FOLLOWING", "OVER-STEER"))
        w(f"| {arm} | {st} | {v['n']} | {100*a['arm_rate']:.1f} / {100*a['A1_rate']:.1f} | {a['delta_pp']:+.2f} [{a['ci95_pp'][0]:+.2f}, {a['ci95_pp'][1]:+.2f}] ({a['seed_floor_delta_pp']:+.2f}) | {'YES' if a['RESOLVED'] else 'no'} | {100*n['arm_rate']:.1f} / {100*n['A1_rate']:.1f} ({n['delta_pp']:+.2f}) | {comp} | {cls} |")
open(os.path.join(RAW, "d6_p2_tables.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print(len(L))
