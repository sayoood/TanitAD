#!/usr/bin/env python3
"""WHY DOES A SYMMETRIC |kappa| PENALTY PRODUCE A SIGNED OUTCOME?

MEASURED (kappa_by_goal_rungs.txt): at W_KAPPA 15.11245 the TURN_L-goal windows
collapse to med|k|max 0.02066 while TURN_R-goal windows hold at 0.08000. The
penalty w*kappa^2 cannot see the sign, so the asymmetry must live in the term it
trades against -- the `ccos` goal term -- or in the windows themselves.

This reads the decisions sidecar's own cost columns per decoded goal class:
  finecost_cv_cl   -- the do-nothing (constant-velocity) candidate's cost
  finecost_plan_cl -- the cost of the candidate the planner actually chose
The MARGIN (cv - plan) is how much the goal term is willing to pay for motion in
that window. If TURN_L windows carry a SMALLER margin, a symmetric penalty
naturally wins there first, and the outcome is signed without the penalty being so.

⛔ CONTROL: `w_kappa_eff_cl` must read the class's own weight on every window
(15.11245 everywhere on a scalar arm; the by-goal map on `gkappa`), which is what
proves `goal_lat_cl` is the field the cost actually keyed on.
ASCII only.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Admin\tanitad-ctg\stack")
from tanitad.models.vocab_v7 import TACTICAL_LAT_ACTIONS_V7 as LAT   # noqa: E402

P4 = r"C:\Users\Admin\refav1_margin\p4out"


def load(dumpdir, key):
    out = []
    for f in sorted(glob.glob(os.path.join(dumpdir, "decisions", "ep*.npz"))):
        z = np.load(f, allow_pickle=True)
        if key not in z.files:
            return None
        out.append(np.asarray(z[key]).reshape(-1))
    return np.concatenate(out) if out else None


def kmax(dumpdir):
    out = []
    for f in sorted(glob.glob(os.path.join(dumpdir, "ep*.npz"))):
        z = np.load(f, allow_pickle=True)
        p = np.concatenate([np.zeros((z["cl"].shape[0], 1, 2)), z["cl"]], axis=1)
        d = np.diff(p, axis=1)
        ds = np.linalg.norm(d, axis=-1)
        th = np.arctan2(d[..., 1], d[..., 0])
        dth = (np.diff(th, axis=1) + np.pi) % (2 * np.pi) - np.pi
        seg = 0.5 * (ds[:, :-1] + ds[:, 1:])
        m = seg > 1e-6
        k = np.where(m, dth / np.maximum(seg, 1e-9), 0.0)
        out.append(np.abs(k).max(1))
    return np.concatenate(out) if out else None


def main():
    print("lateral vocabulary: %s" % list(LAT))
    for tag in sys.argv[1:]:
        d = os.path.join(P4, "dump_%s" % tag)
        if not os.path.isdir(d):
            d = os.path.join(r"C:\Users\Admin\wkfront\out", "dump_%s" % tag)
        g = load(d, "goal_lat_cl")
        cv = load(d, "finecost_cv_cl")
        pl = load(d, "finecost_plan_cl")
        we = load(d, "w_kappa_eff_cl")
        km = kmax(d)
        if g is None or cv is None or pl is None:
            print("%-14s columns absent" % tag)
            continue
        print("\n==== %s   n=%d" % (tag, len(g)))
        print("  %-10s %4s %10s %10s %12s %10s %10s"
              % ("goal", "n", "w_kappa", "med|k|max", "med margin", "med cv",
                 "med plan"))
        for t in sorted(set(int(x) for x in g if x >= 0)):
            m = g == t
            name = LAT[t] if 0 <= t < len(LAT) else str(t)
            marg = cv[m] - pl[m]
            w = np.median(we[m]) if we is not None else float("nan")
            print("  %-10s %4d %10.5f %10.5f %12.5f %10.4f %10.4f"
                  % (name, int(m.sum()), w, np.median(km[m]) if km is not None
                     else float("nan"), np.median(marg), np.median(cv[m]),
                     np.median(pl[m])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
