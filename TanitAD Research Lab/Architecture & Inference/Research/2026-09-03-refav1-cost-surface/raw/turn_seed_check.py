"""THE DECISIVE PER-STRATUM READ: on the windows where the tactical decoder ASKS FOR A
TURN, does the canonical turn control that `plan()` itself injects as a seed
(`refa_v1.py:1790-1793`) beat constant velocity — and if not, by how much, and where does
the loss come from?

This is the question the whole R10 panel exists to answer, restricted to the only
population where it can be asked, and it is answered by the ARITHMETIC of the three live
cost terms rather than by an argmin over a grid.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}
for tag, f in (("incumbent", "cost_surface_incumbent.json"),
               ("ep2", "cost_surface_ep2.json")):
    p = os.path.join(HERE, f)
    if not os.path.exists(p):
        continue
    with open(p, encoding="utf-8") as fh:
        j = json.load(fh)
    turn = [r for r in j["rows"] if float(r.get("goal_kappa_max") or 0.0) > 0.0]
    blk = {"n_windows_total": len(j["rows"]), "n_goal_carries_curvature": len(turn)}
    print(f"== {tag}: windows whose imagined goal CARRIES CURVATURE: "
          f"{len(turn)} / {len(j['rows'])}")
    if not turn:
        print("   (none — the decoder never asks for a turn on this checkpoint)")
        out[tag] = blk
        continue
    for conv in ("A", "B"):
        g = {k: np.array([r[f"named_{conv}_goal_canonical_{k}"] for r in turn])
             for k in ("c_total", "c_goal", "c_jerk", "c_kappa")}
        c = {k: np.array([r[f"named_{conv}_cv_{k}"] for r in turn])
             for k in ("c_total", "c_goal", "c_jerk", "c_kappa")}
        adv = c["c_goal"] - g["c_goal"]          # what the turn BUYS in the model
        charge = g["c_kappa"] + g["c_jerk"]      # what it PAYS explicitly
        blk[conv] = {
            "goal_canonical_c_total_median": float(np.median(g["c_total"])),
            "goal_canonical_T1_median": float(np.median(g["c_goal"])),
            "goal_canonical_jerk_median": float(np.median(g["c_jerk"])),
            "goal_canonical_kappa2_median": float(np.median(g["c_kappa"])),
            "cv_c_total_median": float(np.median(c["c_total"])),
            "cv_T1_median": float(np.median(c["c_goal"])),
            "n_beats_cv": int((g["c_total"] < c["c_total"]).sum()),
            "T1_advantage_median": float(np.median(adv)),
            "explicit_charge_median": float(np.median(charge)),
            "advantage_over_charge_median": float(
                np.median(adv) / max(np.median(charge), 1e-300)),
            "n_advantage_exceeds_charge": int((adv > charge).sum())}
        b = blk[conv]
        print(f"  [{conv}] goal_canonical total {b['goal_canonical_c_total_median']:.4e}"
              f" = T1 {b['goal_canonical_T1_median']:.4e}"
              f" + jerk {b['goal_canonical_jerk_median']:.4e}"
              f" + k2 {b['goal_canonical_kappa2_median']:.4e}")
        print(f"       cv total {b['cv_c_total_median']:.4e} "
              f"(T1 {b['cv_T1_median']:.4e});  the turn beats cv on "
              f"{b['n_beats_cv']}/{len(turn)}")
        print(f"       what the turn BUYS in the model (cv_T1 - turn_T1): "
              f"{b['T1_advantage_median']:+.4e}   what it PAYS explicitly: "
              f"{b['explicit_charge_median']:.4e}   ratio "
              f"{b['advantage_over_charge_median']:.4f}   buys>pays on "
              f"{b['n_advantage_exceeds_charge']}/{len(turn)}")
    out[tag] = blk
    print()

with open(os.path.join(HERE, "turn_seed_check.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("written: turn_seed_check.json")
