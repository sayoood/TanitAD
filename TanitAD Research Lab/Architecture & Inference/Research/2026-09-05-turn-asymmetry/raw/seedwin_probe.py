"""Is the canonical GOAL SEED present in iteration 0, and does it WIN?

Zero GPU, banked dumps. refa_v1.py:2571-2599 seeds the decoded goal's own
canonical control into the iteration-0 population UNCONDITIONALLY, so a full
|kappa| = GOAL_KAPPA_TURN candidate is HANDED to the search on every turn-goal
window. If the plan realises exactly that magnitude, the seed WON; if not, it was
present and LOST ON COST -- which is a statement about the cost comparison, not
about the search's ability to find a candidate.
"""
import collections, glob, os, sys
import numpy as np
WT = "C:/Users/Admin/tanitad-wt"
for p in (WT + "/stack", WT + "/taniteval", WT):
    if p not in sys.path:
        sys.path.insert(0, p)
from tanitad.models.vocab_v7 import TACTICAL_LAT_ACTIONS_V7 as LATV
from tanitad.refs.refa_v1 import GOAL_KAPPA_TURN
P4 = "C:/Users/Admin/refav1_margin/p4out"
I_TL, I_TR = LATV.index("TURN_L"), LATV.index("TURN_R")
print("GOAL_KAPPA_TURN imported from source = %.5f  (never hand-written)"
      % GOAL_KAPPA_TURN)


def load(tag):
    c = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(P4, "dump_" + tag, "decisions",
                                           "ep*.npz"))):
        q = np.load(p)
        for k in ("goal_lat_cl", "cl_controls"):
            c[k].append(q[k])
    if not c:
        return None
    return {k: np.concatenate(v) for k, v in c.items()}


tags = [t for t in sys.argv[1:]] or ["ccos_argmax", "ccos_seed1", "wk15",
                                     "wk151", "cos_wk"]
print("\n%-14s | %-14s | %-12s | %-12s" % ("arm", "exact 0.080000",
                                           "goal TURN_L", "goal TURN_R"))
for tag in tags:
    D = load(tag)
    if D is None:
        print("  %-12s  <no dump>" % tag)
        continue
    gl = D["goal_lat_cl"]
    k = D["cl_controls"][:, :, 1]
    kw = k[np.arange(len(k)), np.abs(k).argmax(1)]
    turn = (gl == I_TL) | (gl == I_TR)
    ex = np.isclose(np.abs(kw), GOAL_KAPPA_TURN, atol=1e-6)
    print("  %-12s | %5d / %-6d | %4d / %-5d | %4d / %-5d"
          % (tag, int(ex[turn].sum()), int(turn.sum()),
             int((ex & (gl == I_TL)).sum()), int((gl == I_TL).sum()),
             int((ex & (gl == I_TR)).sum()), int((gl == I_TR).sum())))
D = load(tags[0])
gl = D["goal_lat_cl"]
k = D["cl_controls"][:, :, 1]
kw = k[np.arange(len(k)), np.abs(k).argmax(1)]
turn = (gl == I_TL) | (gl == I_TR)
print("\n  CONTROL, %s: distinct realised |max kappa| on turn-goal windows = %s"
      % (tags[0], sorted(set(np.round(np.abs(kw[turn]), 6).tolist()))))
print("  (two values -- the canonical seed and the clip -- is what 'the plan IS")
print("   the seed' looks like; a spread of values would refute it.)")
