"""How much M1 closure would a recovery arm need for its composite to pass D2 on the M1 crawl share?  ESTIMATED, post hoc, not a
registered rule: a descriptive line fit over the five recovery arms already read (R, RTL, RP01, RAPR, RAPR-P01), per seed.

For each arm X: x = M1 crawl share X - C0 (the arm's own move, T1-kin, from its read JSON) and y = M1 crawl share XC - P10 (the
composite's D2 row).  Fit y = a + b x per seed (n = 5).  D2 passes on M1 when y is not separated-worse, i.e. its 95 % interval
reaches 0; with the half-width h of the five intervals held at their mean, that is y <= h.  Solve for x and divide by the M1 excess
(mean over arms of |x| / share_closed, the denominator the reader uses) to get the share of the excess the arm would need to close.
Inputs: the five read JSONs (paths below).  Output: d2_closure_need.json next to this script.  Run: python -I d2_closure_need.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DAG = HERE.parent.parent
ARMS = {"R": DAG / "recover_read/raw/RECOVER_READ.json", "RTL": DAG / "recover_next/read/raw/RECOVER_READ_RTL.json",
        "RP01": DAG / "recover_next/read/raw/RECOVER_READ_RP01.json", "RAPR": DAG / "recover_next/read/raw/RECOVER_READ_RAPR.json",
        "RAPRP01": DAG / "recover_next/read/raw/RECOVER_READ_RAPRP01.json"}
K = "M1 crawl share"
out = {"evidence_class": "ESTIMATED (post-hoc line fit over five MEASURED arms; not a registered rule)", "tier": "T1-kin", "seeds": {}}
J = {a: json.load(open(p, encoding="utf-8")) for a, p in ARMS.items()}
for s in ("s0", "s1"):
    rows = {}
    for a, d in J.items():
        x, y = d["seeds"][s]["R_minus_C0"][K], d["seeds"][s]["RC_minus_P10"][K]
        rows[a] = {"x_arm_minus_C0": x["delta"], "share_closed": d["primary"]["share_closed"][s], "y_comp_minus_P10": y["delta"],
                   "y_lo": y["lo"], "y_hi": y["hi"], "y_separated": y["separated"]}
    x = np.array([r["x_arm_minus_C0"] for r in rows.values()])
    y = np.array([r["y_comp_minus_P10"] for r in rows.values()])
    b, a0 = np.polyfit(x, y, 1)
    excess = float(np.mean([abs(r["x_arm_minus_C0"]) / r["share_closed"] for r in rows.values()]))
    h = float(np.mean([(r["y_hi"] - r["y_lo"]) / 2 for r in rows.values()]))
    x_need = (h - a0) / b
    out["seeds"][s] = {"arms": rows, "fit_intercept": float(a0), "fit_slope": float(b), "pearson_r": float(np.corrcoef(x, y)[0, 1]), "n": len(x),
                       "m1_excess": excess, "mean_half_width": h, "x_needed": float(x_need), "share_needed": float(abs(x_need) / excess),
                       "share_needed_for_point_zero": float(abs(-a0 / b) / excess), "max_share_measured": float(max(r["share_closed"] for r in rows.values())),
                       "extrapolation_factor": float((abs(x_need) / excess) / max(r["share_closed"] for r in rows.values()))}
json.dump(out, open(HERE / "d2_closure_need.json", "w", encoding="utf-8", newline="\n"), indent=1)
for s, v in out["seeds"].items():
    print(f"{s}: y = {v['fit_intercept']:.4f} + {v['fit_slope']:.3f} x (r {v['pearson_r']:.3f}, n {v['n']}); share needed {v['share_needed']:.2f} "
          f"(point zero {v['share_needed_for_point_zero']:.2f}); max measured {v['max_share_measured']:.4f}; extrapolation x{v['extrapolation_factor']:.2f}")
