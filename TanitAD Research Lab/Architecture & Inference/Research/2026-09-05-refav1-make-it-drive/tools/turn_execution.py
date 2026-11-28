#!/usr/bin/env python
"""When the goal head is RIGHT, does the trained planner actually turn?

⭐ THE QUESTION. Gate #1 (`GATE_ON_REAL_MODEL.md`) says a LANE_KEEP decode
forces zero curvature. This asks the complement, on the SAME banked dumps and
the SAME trained checkpoint: on the windows where the head DID decode
TURN_L/TURN_R, what curvature did the planner actually emit?

Two answers are possible and they mean completely different things:
  * a real turn  -> the head's recall is the ONLY problem, and a decision rule
                    that fires more often is the whole fix;
  * ~zero        -> the turn was PROPOSED AND REJECTED, a SECOND gate that a
                    decision rule cannot open, and the fix is elsewhere.

⛔ CONTROLS:
  * the LANE_KEEP windows of the same dump are the same-breath negative
    reference (they must read ~0 — that is gate #1, already established);
  * `ha0_ext`, banked in the same file, is a NON-PLANNER arm and must show
    NON-ZERO curvature somewhere, or the dump's curvature column is dead and
    every zero here is a read error rather than a finding;
  * n is printed for every cell.

Run: python turn_execution.py --dump <dir> --label cos --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--label", required=True, help="cost metric label")
    ap.add_argument("--stack", default="C:/Users/Admin/tanitad-wt/stack")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.stack)
    from tanitad.models.v6 import tactical_lat_actions
    LAT = list(tactical_lat_actions("v7.0"))

    CL, GLAT, HAX, SRC = [], [], [], []
    for f in sorted(glob.glob(os.path.join(a.dump, "ep*.npz"))):
        d_f = os.path.join(a.dump, "decisions", os.path.basename(f))
        if not os.path.exists(d_f):
            continue
        with np.load(d_f) as d:
            CL.append(d["cl_controls"]); GLAT.append(d["goal_lat_cl"])
            HAX.append(d["ha0_ext_controls"])
            SRC.append(d["plan_source_cl"])
    if not CL:
        print("NO WINDOWS READ — inconclusive")
        return 2
    CL = np.concatenate(CL); GLAT = np.concatenate(GLAT)
    HAX = np.concatenate(HAX); SRC = np.concatenate(SRC)
    kap = np.abs(CL[:, :, 1]).max(axis=1)          # per-window |curvature|max
    n = len(kap)

    TURN = {LAT.index("TURN_L"), LAT.index("TURN_R")}
    ZERO = {LAT.index("LANE_KEEP"), LAT.index("ABORT_LC")}
    is_turn = np.array([int(g) in TURN for g in GLAT])
    is_zero = np.array([int(g) in ZERO for g in GLAT])

    def blk(m, name):
        if m.sum() == 0:
            return {"name": name, "n": 0}
        k = kap[m]
        return {"name": name, "n": int(m.sum()),
                "frac_windows_with_curvature_gt_1e-3": float((k > 1e-3).mean()),
                "kappa_absmax_mean": float(k.mean()),
                "kappa_absmax_p50": float(np.percentile(k, 50)),
                "kappa_absmax_max": float(k.max()),
                "plan_source_hist": {int(s): int((SRC[m] == s).sum())
                                     for s in np.unique(SRC[m])}}

    R = {"dump": a.dump, "cost_metric_label": a.label, "n_windows": int(n),
         "decoded_TURN_windows": blk(is_turn, "decoded TURN_L/TURN_R"),
         "decoded_ZERO_windows": blk(is_zero, "decoded LANE_KEEP/ABORT_LC"),
         "decoded_lat_hist": {LAT[i]: int((GLAT == i).sum())
                              for i in range(len(LAT)) if (GLAT == i).sum()}}
    # CONTROL: a non-planner arm in the SAME file must carry curvature
    # somewhere, or the column we are reading is dead.
    hax_k = np.abs(HAX[:, :, 1]).max(axis=1)
    R["CONTROL_ha0_ext_curvature"] = {
        "frac_nonzero": float((hax_k > 1e-6).mean()),
        "absmax": float(hax_k.max()),
        "note": "a non-planner arm; must be non-zero somewhere or the "
                "curvature column is dead and every zero above is a read error"}
    R["CONTROL_column_is_live"] = bool(hax_k.max() > 1e-6)
    tb = R["decoded_TURN_windows"]
    R["verdict"] = (
        "NO SECOND GATE: the planner executes the decoded turn"
        if tb.get("frac_windows_with_curvature_gt_1e-3", 0) > 0.5 else
        "SECOND GATE: the decoded turn is PROPOSED AND REJECTED")
    with open(a.out, "w") as f:
        json.dump(R, f, indent=1)
    print(json.dumps(R, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
