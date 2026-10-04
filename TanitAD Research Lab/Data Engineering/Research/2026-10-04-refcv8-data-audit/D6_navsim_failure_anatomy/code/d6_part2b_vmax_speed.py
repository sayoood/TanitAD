"""D6 part 2b -- does the max-speed row move the EMITTED SPEED?  plan 4-s speed minus the PDM-closed reference's 4-s speed
(finite difference), by known / unknown stratum, for A1 (fed the row) and PRIOR (input-free kinematic control) on navhard.
A1's stratum gap beyond PRIOR's gap is the part not explained by scene mix."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d6_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")


def speed4(P):
    return float(np.hypot(*(P[-1, :2] - P[-2, :2])) / 0.5)


def main():
    D = pd.read_csv(os.path.join(RAW, "d6_scene_geometry_step30000.csv")).set_index("token")
    out = {}
    for name, step, arm in (("A1", 30000, "R7_A1"), ("PRIOR", 30000, "PRIOR_ha0p"), ("A1_5k", 5000, "R7_A1")):
        df = C.load_arm(step, arm, with_hooks=True)
        sp = pd.Series({t: speed4(df.loc[t, "plan"]) for t in D.index})
        d = sp - D["ref_speed4"]
        out[name] = {}
        for k, g in D.groupby("vmax_known"):
            out[name]["known" if k else "unknown"] = {"n": int(len(g)), "median_plan_minus_ref_speed4": float(d[g.index].median()),
                                                      "mean_plan_speed4": float(sp[g.index].mean()), "mean_ref_speed4": float(D.loc[g.index, "ref_speed4"].mean())}
        # standardised over (stage, v0 band, cmd) cells
        num = den = 0.0
        for _, g in D.groupby(["stage", "v0_band", "cmd"]):
            u, kn = g[~g.vmax_known], g[g.vmax_known]
            if len(u) >= 5 and len(kn) >= 5:
                num += len(g) * (d[u.index].mean() - d[kn.index].mean())
                den += len(g)
        out[name]["standardised_unknown_minus_known_mean_plan_minus_ref_m_s"] = num / den
    json.dump(out, open(os.path.join(RAW, "d6_part2b_vmax_speed.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=0))


if __name__ == "__main__":
    main()
