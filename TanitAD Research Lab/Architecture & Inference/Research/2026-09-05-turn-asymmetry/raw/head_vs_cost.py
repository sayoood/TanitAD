"""THE DECISIVE CELL: where the GOAL HEAD is RIGHT, does the PLAN still fail?

SPEC section 3.16 registered that a plan-recall gap is PARTLY INHERITED from the
goal head, and that only the INCREMENT over the head is attributable to the cost.
This computes the sharpest form of that: restrict to windows where the head
decoded the CORRECT turn token, and read the plan's recall there. A plan that
fails where the head is right cannot be blamed on the head.

⛔ EVERY CELL PRINTS ITS EPISODE-CLUSTER COUNT. The goal tokens are episode-
degenerate on this slice (the head only decodes TURN_L in the left-heavy
episodes), so the LEFT cell has few clusters and that is stated beside it rather
than discovered afterwards -- RETRACTION_LOG #17.
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
from taniteval import ci as _ci                                    # noqa: E402
from taniteval import four_families as ff                          # noqa: E402
from tanitad.models.vocab_v7 import TACTICAL_LAT_ACTIONS_V7 as LATV  # noqa: E402
from tanitad.refs.refa_v1 import GOAL_KAPPA_TURN                   # noqa: E402
from tanitad.refs.refc_tactical import factor_from_kinematics      # noqa: E402

P4, DT = "C:/Users/Admin/refav1_margin/p4out", 0.2
I_TL, I_TR = LATV.index("TURN_L"), LATV.index("TURN_R")
LAT = {0: "lane_keep", 1: "turn_left", 2: "turn_right"}


def load(tag):
    c = collections.defaultdict(list)
    d0 = os.path.join(P4, "dump_" + tag)
    for p in sorted(glob.glob(os.path.join(d0, "ep*.npz"))):
        d = np.load(p)
        n = d["g"].shape[0]
        for k in ("g", "cl", "ha0_ext"):
            c[k].append(d[k])
        c["eid"].append(np.repeat(d["eid"], n))
        q = np.load(os.path.join(d0, "decisions", os.path.basename(p)))
        c["goal"].append(q["goal_lat_cl"])
        c["ctrl"].append(q["cl_controls"])
    if not c:
        raise SystemExit("[head_vs_cost] no dump for %r" % tag)
    return {k: np.concatenate(v) for k, v in c.items()}


def lab(A):
    t = torch.as_tensor(A).float()
    dy, dv, v0, v1, _ = ff.maneuver_kinematics(t, DT)
    return factor_from_kinematics(dy, dv, v0, v1)[0].numpy()


for tag in (sys.argv[1:] or ["ta_wk15_s0"]):
    D = load(tag)
    gt, pl, eid, goal = lab(D["g"]), lab(D["cl"]), D["eid"], D["goal"]
    head_ok = np.where(gt == 1, goal == I_TL,
                       np.where(gt == 2, goal == I_TR, False))
    plan_ok = pl == gt
    k = D["ctrl"][:, :, 1]
    kw = k[np.arange(len(k)), np.abs(k).argmax(1)]

    print("=" * 96)
    print("ARM %s  (n=%d windows / %d episodes)"
          % (tag, len(gt), len(set(eid.tolist()))))
    print("=" * 96)
    print("  ep | GT L | head L | plan L || GT R | head R | plan R")
    for e in sorted(set(eid.tolist())):
        mL, mR = (eid == e) & (gt == 1), (eid == e) & (gt == 2)
        f = lambda m, v: ("%d/%-2d" % (v[m].sum(), m.sum())) if m.any() else "  -  "
        print("  %2d | %4d | %6s | %6s || %4d | %6s | %6s"
              % (e, mL.sum(), f(mL, head_ok), f(mL, plan_ok),
                 mR.sum(), f(mR, head_ok), f(mR, plan_ok)))

    print("\n  ** THE DECISIVE CELL — windows where the HEAD decoded the CORRECT "
          "turn token **")
    print("  %-9s %4s %9s | %-22s | %-24s"
          % ("stratum", "n", "clusters", "plan recall", "realised peak kappa"))
    for d_, nm, tok in ((1, "GT-LEFT", I_TL), (2, "GT-RIGHT", I_TR)):
        m = (gt == d_) & (goal == tok)
        if not m.any():
            print("  %-9s    0         — | (empty)" % nm)
            continue
        ncl = len(set(eid[m].tolist()))
        r = _ci.episode_cluster_bootstrap(plan_ok[m].astype(float), eid[m],
                                          n_boot=2000)
        full = int((np.abs(kw[m]) > 0.06).sum())
        print("  %-9s %4d %9s | %d/%d = %.4f [%.4f, %.4f] | median %+0.5f, "
              "full |k|>0.06 on %d/%d"
              % (nm, int(m.sum()),
                 "%d %s" % (ncl, "MET" if ncl >= 5 else "**<5**"),
                 int(plan_ok[m].sum()), int(m.sum()), r["mean"], r["lo"], r["hi"],
                 float(np.median(kw[m])), full, int(m.sum())))
    print("  ⇒ a plan that fails where the head is RIGHT cannot be blamed on the "
          "head; the loss is in the COST comparison.")
    print("  ⚠️ read the cluster column: a cell below 5 clusters is a POWER LIMIT, "
          "not a negative.")
    print("  CONTROL: GOAL_KAPPA_TURN = %.5f imported from source; the floor "
          "`ha0_ext` recall on the same windows is L %.4f / R %.4f"
          % (GOAL_KAPPA_TURN,
             (lab(D["ha0_ext"]) == gt)[gt == 1].mean(),
             (lab(D["ha0_ext"]) == gt)[gt == 2].mean()))
    print()
