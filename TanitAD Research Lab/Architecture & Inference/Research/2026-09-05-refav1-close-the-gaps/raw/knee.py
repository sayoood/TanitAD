#!/usr/bin/env python3
"""AN ANALYTIC PREDICTION FOR A2's KNEE -- 0 GPU, from the banked cost columns.

⭐ THE ARITHMETIC. With `W_JERK = 0` and `target_speed = None` (so `W_VEND` is a
DEAD TERM -- the tool never passes `target_speed`), the cost of ANY candidate is

    c(candidate) = goal_term(candidate) + W_KAPPA * mean(kappa^2)

The do-nothing candidate has kappa == 0, so its cost is `basecost_cv`, INDEPENDENT
of W_KAPPA. The winning plan at W_KAPPA = 0 has cost `plan_cost` and its own
mean(kappa^2). So the plan stops beating do-nothing exactly when

    plan_cost + W * mean(kappa_plan^2) > basecost_cv
    <=>  W > (basecost_cv - plan_cost) / mean(kappa_plan^2)   =:  W_flip

⇒ `W_flip` is a PER-WINDOW prediction of the weight at which that window goes
straight, computable from `dump_ccos_argmax` alone.

⛔ WHAT THIS IS NOT. It is a TWO-CANDIDATE bound. iCEM searches a continuum, so
the real behaviour is a gradual magnitude reduction (MEASURED: TURN_L med|k|max
0.08000 -> 0.02066 -> 0.00840 across W = 0, 15.11, 151.1), not a binary flip at
W_flip. ⇒ W_flip OVER-estimates the knee: intermediate curvatures survive past it.
Reported as an UPPER BOUND on where turning dies, never as the knee itself.

⛔ CONTROL: the same quantity on LANE_KEEP windows must be SMALLER (they are the
windows the penalty is supposed to straighten first), or the arithmetic is wrong.

ASCII output only.
"""
import glob, os, sys
import numpy as np
from tanitad.models.vocab_v7 import TACTICAL_LAT_ACTIONS_V7
LAT = list(TACTICAL_LAT_ACTIONS_V7)
R = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/refav1_margin/p4out"

rows = []
for f in sorted(glob.glob(os.path.join(R, "dump_ccos_argmax", "decisions", "*.npz"))):
    z = np.load(f, allow_pickle=True)
    for i, w in enumerate(z["ws"].tolist()):
        k = z["cl_controls"][i][:, 1]
        mk2 = float((k ** 2).mean())
        rows.append(dict(ep=int(os.path.basename(f)[2:5]), w=int(w),
                         tok=LAT[int(z["goal_lat_cl"][i])],
                         mk2=mk2, pc=float(z["plan_cost_cl"][i]),
                         cv=float(z["basecost_cv_cl"][i]),
                         hv=float(z["basecost_hold_v0_cl"][i])))
print("=" * 96)
print("ANALYTIC UPPER BOUND ON A2's KNEE: W_flip = (basecost_cv - plan_cost) / mean(kappa^2)")
print("  valid because W_JERK = 0 and W_VEND is a DEAD TERM (target_speed is never passed)")
print("=" * 96)
print("  %-12s %5s %11s %11s %11s %13s" %
      ("goal token", "n", "mean k^2", "cv-plan", "median", "W_flip range"))
print("  %-12s %5s %11s %11s %11s %13s" % ("", "", "(nonzero)", "advantage", "W_flip", "p25 - p75"))
print("  " + "-" * 78)
for tok in ("LANE_KEEP", "TURN_L", "TURN_R"):
    rr = [r for r in rows if r["tok"] == tok and r["mk2"] > 1e-12]
    n_zero = sum(1 for r in rows if r["tok"] == tok and r["mk2"] <= 1e-12)
    if not rr:
        print("  %-12s %5d  (all windows already straight at W_KAPPA=0)" % (tok, n_zero))
        continue
    wf = np.array([(r["cv"] - r["pc"]) / r["mk2"] for r in rr])
    wf = wf[np.isfinite(wf)]
    print("  %-12s %5d %11.6f %11.5f %11.3f  %5.2f - %-6.2f"
          % (tok, len(rr), np.mean([r["mk2"] for r in rr]),
             np.mean([r["cv"] - r["pc"] for r in rr]), float(np.median(wf)),
             float(np.percentile(wf, 25)), float(np.percentile(wf, 75))))
    if n_zero:
        print("               (+%d window(s) already straight at W_KAPPA=0, excluded)" % n_zero)
print()
print("  ⭐ COMMITTED PREDICTION FOR A2, from this arithmetic:")
tl = [r for r in rows if r["tok"] == "TURN_L" and r["mk2"] > 1e-12]
if tl:
    wf = np.array([(r["cv"] - r["pc"]) / r["mk2"] for r in tl]); wf = wf[np.isfinite(wf)]
    for W in (1.0, 3.0, 7.0, 15.11245):
        frac = float((wf > W).mean())
        print("     W_KAPPA = %-9.5g -> %4.1f %% of TURN_L windows still prefer the turn "
              "(UPPER bound)" % (W, 100 * frac))
    print("     n_true(turn_left) on this panel = 11, recall at W=0 is 0.3636 (4/11).")
print("  ⛔ UPPER BOUND: iCEM reduces MAGNITUDE before it flips to straight, so real")
print("     turn recall dies EARLIER than these numbers say. If A2 measures turn_left")
print("     recall 0.0000 at a W where this predicts most windows still 'prefer' the")
print("     turn, that CONFIRMS the two-candidate model is an upper bound -- it does")
print("     not refute the arithmetic.")
