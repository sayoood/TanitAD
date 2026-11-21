"""D6 part 5 -- threshold sensitivity of the DAC-zero class mix.  Every literal threshold in part 2 is moved x0.5 and x1.5
(one at a time and all together); the class shares among A1's DAC-zero scenes (both stages) are re-derived and the
range per class is reported, so a class whose share only exists at one choice of threshold is visible as such.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d6_common as C
import d6_part2_anatomy as P

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")
NAMES = ("T_TURN", "T_HEAD", "T_LAT", "T_STOP", "T_AWAY", "SPEED_RATIO", "SPEED_ABS")


def main():
    G, _ = P.load_jsonl(os.path.join(RAW, "geom_all.jsonl"))
    tab = pd.read_csv(os.path.join(RAW, "d6_scene_table_step30000.csv")).set_index("token")
    A1 = C.load_arm(30000, "R7_A1")
    toks = [t for t in tab.index if t in G and tab.loc[t, "A1_DAC"] == 0]
    base = {n: getattr(P, n) for n in NAMES}
    scen = [("base", {})]
    for n in NAMES:
        scen.append((f"{n}x0.5", {n: base[n] * 0.5}))
        scen.append((f"{n}x1.5", {n: base[n] * 1.5}))
    scen.append(("ALLx0.5", {n: base[n] * 0.5 for n in NAMES}))
    scen.append(("ALLx1.5", {n: base[n] * 1.5 for n in NAMES}))
    res = {}
    for name, ch in scen:
        for n, v in base.items():
            setattr(P, n, v)
        for n, v in ch.items():
            setattr(P, n, v)
        cls = []
        for t in toks:
            f = P.plan_features(G[t], "MAIN", A1.loc[t, "plan"], float(A1.loc[t, "v0"]))
            cls.append(P.classify(f))
        c = pd.Series(cls).value_counts(normalize=True)
        res[name] = {k: float(v) for k, v in c.items()}
    for n, v in base.items():
        setattr(P, n, v)
    allc = sorted({k for r in res.values() for k in r})
    summ = {c: {"base": res["base"].get(c, 0.0), "min": min(r.get(c, 0.0) for r in res.values()), "max": max(r.get(c, 0.0) for r in res.values())} for c in allc}
    rank_top3 = {name: [k for k, _ in sorted(r.items(), key=lambda kv: -kv[1])[:3]] for name, r in res.items()}
    out = {"n_DACzero_scenes": len(toks), "scenarios": res, "range_by_class": summ, "top3_by_scenario": rank_top3}
    json.dump(out, open(os.path.join(RAW, "d6_part5_sensitivity.json"), "w", encoding="utf-8"), indent=1)
    print({c: (round(v["min"], 3), round(v["base"], 3), round(v["max"], 3)) for c, v in summ.items()})


if __name__ == "__main__":
    main()
