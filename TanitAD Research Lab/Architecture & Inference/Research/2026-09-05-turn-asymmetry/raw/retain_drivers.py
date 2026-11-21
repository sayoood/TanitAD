"""Is curvature RETENTION governed by a continuous variable that merely
CORRELATES with turn direction on this panel?  Zero GPU, banked dumps.

If it is, the "left/right asymmetry" is that variable wearing a direction
costume -- the same shape as the speed confound I proposed and refuted, so this
probe has to be able to come out either way and is written to say so.
"""
import collections
import glob
import os
import sys

import numpy as np
import torch

WT = "C:/Users/Admin/tanitad-wt"
for p in (WT + "/stack", WT + "/taniteval", WT):
    if p not in sys.path:
        sys.path.insert(0, p)
from taniteval import four_families as ff                          # noqa: E402
from tanitad.models.vocab_v7 import TACTICAL_LAT_ACTIONS_V7 as LATV  # noqa: E402
from tanitad.refs.refc_tactical import factor_from_kinematics       # noqa: E402

P4, DT, WK = "C:/Users/Admin/refav1_margin/p4out", 0.2, 15.11245
I_TL, I_TR = LATV.index("TURN_L"), LATV.index("TURN_R")
KEEP = 0.06


def load(tag):
    c = collections.defaultdict(list)
    d0 = os.path.join(P4, "dump_" + tag)
    for p in sorted(glob.glob(os.path.join(d0, "ep*.npz"))):
        d = np.load(p)
        n = d["g"].shape[0]
        for k in ("g", "cl"):
            c[k].append(d[k])
        c["v0"].append(d["v0"])
        c["eid"].append(np.repeat(d["eid"], n))
        q = np.load(os.path.join(d0, "decisions", os.path.basename(p)))
        for k in ("goal_lat_cl", "cl_controls", "plan_cost_cl",
                  "basecost_cv_cl", "plan_source_cl"):
            c[k].append(q[k])
    return {k: np.concatenate(v) for k, v in c.items()}


A, W = load("ccos_argmax"), load("wk15")
gl = A["goal_lat_cl"]
turn = (gl == I_TL) | (gl == I_TR)
kW = W["cl_controls"][:, :, 1]
kw = kW[np.arange(len(kW)), np.abs(kW).argmax(1)]
keep = np.abs(kw) > KEEP
goal_cost_full = A["plan_cost_cl"]                  # W_KAPPA = 0 -> pure goal
v0 = A["v0"]
gt = torch.as_tensor(A["g"]).float()
dy, dv, gv0, gv1, _ = ff.maneuver_kinematics(gt, DT)
lat_g = factor_from_kinematics(dy, dv, gv0, gv1)[0].numpy()

print("=" * 100)
print("THE 22 TURN-GOAL WINDOWS, sorted by v0 -- retained (R) or crushed (.)")
print("=" * 100)
print("  %4s %-7s %-7s %6s %10s %10s %8s %4s" %
      ("win", "goal", "GT", "v0", "goal_cost", "basecv", "k(wk15)", "keep"))
idx = [i for i in np.argsort(v0) if turn[i]]
for i in idx:
    print("  %4d %-7s %-7s %6.2f %10.5f %10.5f %+8.4f  %s"
          % (i, LATV[gl[i]][-6:], ("LK", "TL", "TR")[lat_g[i]], v0[i],
             goal_cost_full[i], A["basecost_cv_cl"][i], kw[i],
             "R" if keep[i] else "."))

print()
print("=" * 100)
print("CAN ANY SINGLE VARIABLE SEPARATE retained FROM crushed AS WELL AS "
      "DIRECTION DOES?")
print("=" * 100)
print("  direction itself: TURN_L kept %d/%d, TURN_R kept %d/%d"
      % (keep[gl == I_TL].sum(), (gl == I_TL).sum(),
         keep[gl == I_TR].sum(), (gl == I_TR).sum()))


def best_split(x, y, name, unit=""):
    """Best single threshold on x, by accuracy, over the 22 turn-goal windows."""
    xs = np.sort(np.unique(x))
    best = (0.0, None, None)
    for c in (xs[:-1] + xs[1:]) / 2 if len(xs) > 1 else []:
        for sign in (1, -1):
            pred = (x * sign) > (c * sign)
            acc = float((pred == y).mean())
            if acc > best[0]:
                best = (acc, c, sign)
    print("  %-22s best threshold %s %s%.4f%s -> accuracy %.4f  (%d/%d)"
          % (name, ">" if best[2] == 1 else "<", "", best[1] if best[1] is not None else float("nan"),
             unit, best[0], round(best[0] * y.size), y.size))
    return best[0]


y = keep[turn]
accs = {}
for nm, x, u in (("v0 (m/s)", v0[turn], " m/s"),
                 ("goal cost at full k", goal_cost_full[turn], ""),
                 ("basecost_cv", A["basecost_cv_cl"][turn], ""),
                 ("episode id", A["eid"][turn].astype(float), ""),
                 ("GT |dyaw|", np.abs(dy.numpy())[turn], " rad")):
    accs[nm] = best_split(x, y, nm, u)
dir_acc = float(((gl[turn] == I_TR) == y).mean())
print("  %-22s %s -> accuracy %.4f  (%d/%d)"
      % ("DIRECTION (goal token)", " " * 26, dir_acc, round(dir_acc * y.size), y.size))
print()
if dir_acc > max(accs.values()) + 1e-9:
    print("  ==> DIRECTION separates BETTER than any single continuous variable "
          "tried. No confound found among these; the split is directional ON "
          "THIS PANEL.")
else:
    k = max(accs, key=accs.get)
    print("  ==> %r separates at least as well as direction (%.4f vs %.4f) -- "
          "a CONFOUND candidate, not a left/right effect." % (k, accs[k], dir_acc))

print()
print("=" * 100)
print("CONTROL: the same split on the W_KAPPA = 0 arm must be UNDEFINED "
      "(everything retained)")
print("=" * 100)
kA = A["cl_controls"][:, :, 1]
ka = kA[np.arange(len(kA)), np.abs(kA).argmax(1)]
keepA = np.abs(ka) > KEEP
print("  W_KAPPA=0: TURN_L kept %d/%d, TURN_R kept %d/%d -- %s"
      % (keepA[gl == I_TL].sum(), (gl == I_TL).sum(),
         keepA[gl == I_TR].sum(), (gl == I_TR).sum(),
         "no variable can separate a constant" if keepA[turn].all()
         else "** NOT all retained -- the control is not a constant **"))

print()
print("=" * 100)
print("EPISODE CONFOUND: which episodes carry each goal token?")
print("=" * 100)
for tid, nm in ((I_TL, "TURN_L"), (I_TR, "TURN_R")):
    m = gl == tid
    c = collections.Counter(A["eid"][m].tolist())
    print("  %-7s n=%2d over %d episodes: %s"
          % (nm, m.sum(), len(c), dict(sorted(c.items()))))
print("  ⛔ if the two goal tokens live in DISJOINT episodes, direction and "
      "episode are the same variable on this panel and neither can be "
      "attributed -- which is exactly what the wider panel is for.")
ce = set(A["eid"][gl == I_TL].tolist()) & set(A["eid"][gl == I_TR].tolist())
print("  episodes carrying BOTH goal tokens: %s" % (sorted(ce) or "NONE"))
