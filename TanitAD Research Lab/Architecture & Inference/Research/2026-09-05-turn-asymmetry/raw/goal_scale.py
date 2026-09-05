"""THE MECHANISM TEST: is the GOAL TERM's own decision range smaller for a
TURN_L goal than for a TURN_R goal?  Zero GPU, from the banked dumps.

Under `ccos` with W_KAPPA = 0 the ENTIRE cost is the goal term, so
    goal decision = basecost_cv - plan_cost
is exactly how much the goal term can buy by moving off the do-nothing plan.
The curvature charge W_KAPPA * mean(kappa^2) is sign-symmetric (asserted), so
whichever goal has the SMALLER decision range is the one a fixed charge kills
first -- with no left/right term anywhere in the cost.
"""
import glob, os, sys, collections
import numpy as np
WT = "C:/Users/Admin/tanitad-wt"
for p in (WT + "/stack", WT + "/taniteval", WT):
    if p not in sys.path: sys.path.insert(0, p)
from tanitad.models.vocab_v7 import TACTICAL_LAT_ACTIONS_V7 as LATV
P4 = "C:/Users/Admin/refav1_margin/p4out"
I_LK, I_TL, I_TR = LATV.index("LANE_KEEP"), LATV.index("TURN_L"), LATV.index("TURN_R")

def load(tag, keys):
    c = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(P4, "dump_" + tag, "decisions", "ep*.npz"))):
        q = np.load(p)
        for k in keys: c[k].append(q[k])
    return {k: np.concatenate(v) for k, v in c.items()}

K = ["goal_lat_cl", "plan_cost_cl", "basecost_cv_cl", "cl_controls",
     "plan_source_cl", "finecost_plan_cl", "finecost_cv_cl"]
A = load("ccos_argmax", K)          # W_KAPPA = 0 -> cost IS the goal term
W = load("wk15", K)                 # W_KAPPA = 15.11245
gl = A["goal_lat_cl"]
dec = A["basecost_cv_cl"] - A["plan_cost_cl"]        # the goal decision, per window
kap = np.abs(A["cl_controls"][:, :, 1])
charge15 = 15.11245 * (A["cl_controls"][:, :, 1] ** 2).mean(1)

print("=" * 92)
print("THE GOAL TERM'S OWN DECISION RANGE, per DECODED GOAL TOKEN (ccos, W_KAPPA=0)")
print("=" * 92)
print("  %-10s %3s | %10s %10s %10s | %11s %11s" %
      ("goal", "n", "dec med", "dec mean", "dec min", "charge@15 med", "ratio c/d"))
res = {}
for tid, nm in ((I_LK, "LANE_KEEP"), (I_TL, "TURN_L"), (I_TR, "TURN_R")):
    m = gl == tid
    if not m.any(): continue
    d, ch = dec[m], charge15[m]
    res[nm] = d
    print("  %-10s %3d | %10.5f %10.5f %10.5f | %11.5f %11.2f" %
          (nm, m.sum(), np.median(d), d.mean(), d.min(),
           np.median(ch), np.median(ch) / max(np.median(d), 1e-12)))
if "TURN_L" in res and "TURN_R" in res:
    print("\n  ** TURN_L median goal decision = %.5f ; TURN_R = %.5f  ->  RATIO %.3f **"
          % (np.median(res["TURN_L"]), np.median(res["TURN_R"]),
             np.median(res["TURN_L"]) / max(np.median(res["TURN_R"]), 1e-12)))
    from scipy import stats
    u = stats.mannwhitneyu(res["TURN_L"], res["TURN_R"], alternative="two-sided")
    print("     MWU p = %.4g  (n_L=%d n_R=%d; DESCRIPTIVE -- decision estimator is the "
          "episode-cluster bootstrap)" % (u.pvalue, len(res["TURN_L"]), len(res["TURN_R"])))
    print("     CONTROL that must differ: LANE_KEEP median = %.5f" % np.median(res["LANE_KEEP"]))

print("\n" + "=" * 92)
print("PER-WINDOW: goal decision vs the charge wk15 would levy on the FULL 0.08 goal")
print("=" * 92)
full = 15.11245 * (0.08 ** 2) * (10.0 / 10.0)   # a goal-shaped profile held for the horizon
print("  a plan that fully tracks the goal (|kappa| = 0.08 over the horizon) pays")
print("  W_KAPPA * 0.08^2 = %.5f in curvature charge." % full)
print("  %4s %-10s | %11s %11s | %-10s %-10s" %
      ("win", "goal", "goal dec", "dec > charge?", "k(ccos)", "k(wk15)"))
order = np.argsort(-dec)
for i in order:
    if gl[i] == I_LK: continue
    ka = A["cl_controls"][i, :, 1]; kw = W["cl_controls"][i, :, 1]
    ka = ka[np.abs(ka).argmax()]; kw = kw[np.abs(kw).argmax()]
    print("  %4d %-10s | %11.5f %11s | %+9.4f %+9.4f"
          % (i, LATV[gl[i]], dec[i], "YES" if dec[i] > full else "no", ka, kw))

print("\n" + "=" * 92)
print("THE PREDICTION THIS MAKES, AND ITS TEST")
print("=" * 92)
keptL = keptR = totL = totR = 0
for i in range(len(gl)):
    if gl[i] not in (I_TL, I_TR): continue
    kw = W["cl_controls"][i, :, 1]; kw = kw[np.abs(kw).argmax()]
    kept = abs(kw) > 0.06
    if gl[i] == I_TL: totL += 1; keptL += kept
    else: totR += 1; keptR += kept
print("  windows whose GOAL is TURN_L that wk15 tracks at full |kappa| > 0.06: %d/%d"
      % (keptL, totL))
print("  windows whose GOAL is TURN_R that wk15 tracks at full |kappa| > 0.06: %d/%d"
      % (keptR, totR))
print("  CONTROL (W_KAPPA = 0, same goals, same seed):")
kL = kR = 0
for i in range(len(gl)):
    if gl[i] not in (I_TL, I_TR): continue
    ka = A["cl_controls"][i, :, 1]; ka = ka[np.abs(ka).argmax()]
    if gl[i] == I_TL: kL += abs(ka) > 0.06
    else: kR += abs(ka) > 0.06
print("    TURN_L %d/%d   TURN_R %d/%d  (must be near-equal, or the asymmetry "
      "predates the charge)" % (kL, totL, kR, totR))
