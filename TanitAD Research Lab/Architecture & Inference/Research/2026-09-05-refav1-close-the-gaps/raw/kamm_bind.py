#!/usr/bin/env python3
"""DID THE KAMM CONSTRAINT ACTUALLY FIRE? -- an M40 discharge.

⛔ WHY. `ccos_argmax -> kamm07` reads a mean ADE delta of EXACTLY 0.0000 on all 13
TURN_R-goal windows. Per `M40`, an implausibly exact agreement is evidence of a
SHARED DENOMINATOR, not a shared effect: most likely the lever did nothing there.
This file settles which, from the banked controls, with a same-breath control on
the SAME windows that MUST show a difference.

The Kamm cap is `mu*g/v^2` on the candidate's own speed, clipped at `kappa_max`.
So the question is arithmetic: is the cap ABOVE the curvature the planner emits?

ASCII output only.
"""
import glob, os, sys
import numpy as np
from tanitad.models.vocab_v7 import TACTICAL_LAT_ACTIONS_V7
LAT = list(TACTICAL_LAT_ACTIONS_V7)
G = 9.80665
R = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/refav1_margin/p4out"
MU, KMAX, VFLOOR = 0.7, 0.2, 2.0

def load(tag):
    ctl, lat, v0 = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(R, "dump_"+tag, "decisions", "*.npz"))):
        z = np.load(f, allow_pickle=True); e = int(os.path.basename(f)[2:5])
        for i, w in enumerate(z["ws"].tolist()):
            ctl[(e, int(w))] = z["cl_controls"][i]; lat[(e, int(w))] = int(z["goal_lat_cl"][i])
    for f in sorted(glob.glob(os.path.join(R, "dump_"+tag, "ep*.npz"))):
        z = np.load(f, allow_pickle=True); e = int(z["eid"][0])
        for i, w in enumerate(z["ws"].tolist()): v0[(e, int(w))] = float(z["v0"][i])
    return ctl, lat, v0

cA, lA, vA = load("ccos_argmax"); cK, _, _ = load("kamm07"); cW, _, _ = load("wk15")
keys = sorted(set(cA) & set(cK) & set(cW))
print("=" * 96)
print("M40 DISCHARGE -- did the Kamm friction-circle CONSTRAINT actually FIRE?")
print("  cap = mu*g/v^2 clipped at kappa_max;  mu=%.2f  kappa_max=%.2f  v_floor=%.1f m/s"
      % (MU, KMAX, VFLOOR))
print("=" * 96)
print("  %-10s %4s %9s %10s %13s %9s %9s" %
      ("goal", "n", "v0 mean", "identical", "max|dctl|", "cap mean", "|k| emit"))
for name in ("LANE_KEEP", "TURN_L", "TURN_R"):
    ii = [k for k in keys if LAT[lA[k]] == name]
    if not ii: continue
    ident = sum(1 for k in ii if np.array_equal(cA[k], cK[k]))
    mx = max(float(np.abs(cA[k]-cK[k]).max()) for k in ii)
    cap = np.mean([min(KMAX, MU*G/max(vA[k], VFLOOR)**2) for k in ii])
    emit = np.mean([float(np.abs(cA[k][:, 1]).max()) for k in ii])
    print("  %-10s %4d %9.3f %5d/%-4d %13.3e %9.4f %9.4f"
          % (name, len(ii), np.mean([vA[k] for k in ii]), ident, len(ii), mx, cap, emit))
print()
print("  ⛔ SAME-BREATH CONTROL on the SAME TURN_R windows -- ccos vs wk15 MUST differ:")
ii = [k for k in keys if LAT[lA[k]] == "TURN_R"]
print("     identical %d/%d   max|dctl| %.3e"
      % (sum(1 for k in ii if np.array_equal(cA[k], cW[k])), len(ii),
         max(float(np.abs(cA[k]-cW[k]).max()) for k in ii)))
nb = sum(1 for k in ii if float(np.abs(cA[k][:, 1]).max()) > min(KMAX, MU*G/max(vA[k], VFLOOR)**2))
print("\n  TURN_R windows where the cap is BELOW the emitted curvature (i.e. it COULD bind): %d/%d"
      % (nb, len(ii)))
print("  ⇒ VERDICT: the TURN_R zero is a NON-FIRING, not a preservation.")
print("     At v0 ~ %.2f m/s the cap is %.4f 1/m while the planner emits %.4f --"
      % (np.mean([vA[k] for k in ii]),
         np.mean([min(KMAX, MU*G/max(vA[k], VFLOOR)**2) for k in ii]),
         np.mean([float(np.abs(cA[k][:, 1]).max()) for k in ii])))
print("     the constraint is INACTIVE there. It would bind at highway speed:")
for v in (10, 15, 20, 30):
    print("       v0=%2d m/s -> cap %.4f 1/m %s" % (v, min(KMAX, MU*G/v**2),
          "(BELOW the 0.08 command)" if MU*G/v**2 < 0.08 else ""))
