"""WHY does the SAME charge crush TURN_L and not TURN_R?  Zero GPU.

Decomposition. Under wk15 the total cost is  goal_term + W*mean(kappa^2)
(W_JERK = 0, and W_VEND acts on the accel channel only). So:
    goal_cost(plan) = plan_cost - W*mean(kappa^2)
is recoverable per window for BOTH arms, and the two arms share the SAME goal
field (control: basecost_cv must be bit-identical, since cv has kappa = 0).
"""
import glob, os, sys, collections
import numpy as np
WT = "C:/Users/Admin/tanitad-wt"
for p in (WT + "/stack", WT + "/taniteval", WT):
    if p not in sys.path: sys.path.insert(0, p)
from tanitad.models.vocab_v7 import TACTICAL_LAT_ACTIONS_V7 as LATV
P4 = "C:/Users/Admin/refav1_margin/p4out"
I_LK, I_TL, I_TR = LATV.index("LANE_KEEP"), LATV.index("TURN_L"), LATV.index("TURN_R")
WK = 15.11245

def load(tag, keys):
    c = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(P4, "dump_" + tag, "decisions", "ep*.npz"))):
        q = np.load(p)
        for k in keys: c[k].append(q[k])
    return {k: np.concatenate(v) for k, v in c.items()}

K = ["goal_lat_cl", "plan_cost_cl", "basecost_cv_cl", "cl_controls", "plan_source_cl"]
A, W = load("ccos_argmax", K), load("wk15", K)
gl = A["goal_lat_cl"]

print("=" * 96)
print("CONTROL 0 -- do the two arms share the SAME goal field?")
print("=" * 96)
same = np.isclose(A["basecost_cv_cl"], W["basecost_cv_cl"], rtol=0, atol=1e-6)
print("  basecost_cv identical (cv has kappa=0, so it is PURE goal term): %d/40" % same.sum())
print("  max |diff| = %.3e   CONTROL that must be NON-zero: max|basecost_cv - 1.0| = %.4f"
      % (np.abs(A["basecost_cv_cl"] - W["basecost_cv_cl"]).max(),
         np.abs(A["basecost_cv_cl"] - 1.0).max()))
if same.sum() != 40:
    print("  ** the goal field DIFFERS between arms -- every comparison below is void **")

kA = A["cl_controls"][:, :, 1]; kW = W["cl_controls"][:, :, 1]
chA, chW = WK * (kA ** 2).mean(1), WK * (kW ** 2).mean(1)
goalA = A["plan_cost_cl"]                       # W_KAPPA = 0 -> cost IS the goal term
goalW = W["plan_cost_cl"] - chW                 # strip the charge
totA_underW = goalA + chA                       # the ccos winner, PRICED under wk15

print()
print("=" * 96)
print("THE DECOMPOSITION -- per window, by decoded goal token")
print("=" * 96)
print("  %4s %-8s %3s | %8s %8s | %8s %8s %8s | %8s %8s | %s" %
      ("win", "goal", "src", "k_ccos", "k_wk15", "goal_A", "goal_W", "d_goal",
       "chg_A", "chg_W", "wk15 total vs ccos-cand-under-W"))
rows = {"TURN_L": [], "TURN_R": []}
for i in np.argsort(gl):
    if gl[i] not in (I_TL, I_TR): continue
    nm = LATV[gl[i]]
    ka = kA[i][np.abs(kA[i]).argmax()]; kw = kW[i][np.abs(kW[i]).argmax()]
    dg = goalW[i] - goalA[i]
    rows[nm].append((abs(ka), abs(kw), goalA[i], goalW[i], dg, chA[i], chW[i],
                     W["plan_cost_cl"][i], totA_underW[i]))
    print("  %4d %-8s %3d | %+8.4f %+8.4f | %8.5f %8.5f %+8.5f | %8.5f %8.5f | %8.5f %s %8.5f"
          % (i, nm, W["plan_source_cl"][i], ka, kw, goalA[i], goalW[i], dg,
             chA[i], chW[i], W["plan_cost_cl"][i],
             "<" if W["plan_cost_cl"][i] < totA_underW[i] else ">=", totA_underW[i]))

print()
print("=" * 96)
print("THE NUMBER THAT EXPLAINS IT: how much GOAL COST does giving up the turn cost you?")
print("=" * 96)
print("  d_goal = goal_cost(wk15's crushed plan) - goal_cost(the full-kappa ccos plan).")
print("  A LARGE d_goal means abandoning the turn is EXPENSIVE in goal terms, so the")
print("  charge cannot win. A SMALL d_goal means the goal barely notices, so it can.")
print()
print("  %-8s %3s | %10s %10s %10s | %10s" %
      ("goal", "n", "d_goal med", "d_goal mean", "d_goal max", "chg saved med"))
for nm in ("TURN_L", "TURN_R"):
    r = np.array(rows[nm])
    print("  %-8s %3d | %10.5f %10.5f %10.5f | %10.5f"
          % (nm, len(r), np.median(r[:, 4]), r[:, 4].mean(), r[:, 4].max(),
             np.median(r[:, 5] - r[:, 6])))
L, R = np.array(rows["TURN_L"]), np.array(rows["TURN_R"])
from scipy import stats
u = stats.mannwhitneyu(L[:, 4], R[:, 4], alternative="two-sided")
print("\n  MWU on d_goal: p = %.4g  (n_L=%d n_R=%d)" % (u.pvalue, len(L), len(R)))
print("  -> TURN_L gives up %.5f of goal cost to save %.5f of charge  (NET %+.5f)"
      % (np.median(L[:, 4]), np.median(L[:, 5] - L[:, 6]),
         np.median(L[:, 4]) - np.median(L[:, 5] - L[:, 6])))
print("  -> TURN_R would give up %.5f to save %.5f  (NET %+.5f) -- so it keeps the turn"
      % (np.median(R[:, 4]), np.median(R[:, 5] - R[:, 6]),
         np.median(R[:, 4]) - np.median(R[:, 5] - R[:, 6])))

print()
print("=" * 96)
print("AND THE CONTROL THAT MUST READ THE OTHER WAY: goal cost of the FULL-kappa plan")
print("=" * 96)
print("  %-8s %3s | %12s %12s" % ("goal", "n", "goal_A med", "goal_A max"))
for nm in ("TURN_L", "TURN_R"):
    r = np.array(rows[nm])
    print("  %-8s %3d | %12.5f %12.5f" % (nm, len(r), np.median(r[:, 2]), r[:, 2].max()))
print("  (if these differ a lot, the world model's latent already treats the two")
print("   directions differently BEFORE any charge is levied)")
