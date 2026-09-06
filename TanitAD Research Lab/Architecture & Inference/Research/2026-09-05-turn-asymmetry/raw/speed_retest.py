"""RE-TEST of the speed confound on the WIDE panel.

⛔ WHY THIS EXISTS. I refuted the speed-confound hypothesis on the banked
40-window panel, and then LATER established that that panel was structurally
degenerate (direction and episode were the same variable there). A refutation
made on inadmissible evidence is not a refutation, whichever way it came out, so
it is re-run here at n = 30/30 with 6 clusters per direction.

The test: if the SLOW/FAST split matched the LEFT/RIGHT split, speed would be the
confound. If SLOW ~ FAST while LEFT << RIGHT, the effect is DIRECTION.
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
from tanitad.refs.refc_tactical import factor_from_kinematics      # noqa: E402

P4, DT = "C:/Users/Admin/refav1_margin/p4out", 0.2


def lab(A):
    t = torch.as_tensor(A).float()
    dy, dv, v0, v1, _ = ff.maneuver_kinematics(t, DT)
    return factor_from_kinematics(dy, dv, v0, v1)[0].numpy()


def load(tag):
    d0 = os.path.join(P4, "dump_" + tag)
    o = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(d0, "ep*.npz"))):
        d = np.load(p)
        n = d["g"].shape[0]
        o["g"].append(d["g"]); o["cl"].append(d["cl"]); o["v0"].append(d["v0"])
        o["eid"].append(np.repeat(d["eid"], n))
    if not o:
        raise SystemExit("[speed_retest] no dump for %r" % tag)
    return {k: np.concatenate(v) for k, v in o.items()}


tags = sys.argv[1:] or ["ta_ccos_s0", "ta_wk15_s0", "ta_wk15_s1"]
ref = load(tags[0])
gt, v0, eid = lab(ref["g"]), ref["v0"], ref["eid"]
turn = gt > 0
med = float(np.median(v0[turn]))

print("=" * 96)
print("GT-turn v0 BY DIRECTION (wide panel)")
print("=" * 96)
for d_, nm in ((1, "turn_left"), (2, "turn_right")):
    m = gt == d_
    print("  %-11s n=%2d  v0 median %6.3f  mean %6.3f  min %5.3f  max %6.3f"
          % (nm, int(m.sum()), np.median(v0[m]), v0[m].mean(), v0[m].min(),
             v0[m].max()))
try:
    from scipy import stats
    print("  MWU p=%.4g  (DESCRIPTIVE only; windows overlap and the decision "
          "estimator is the episode-cluster bootstrap)"
          % stats.mannwhitneyu(v0[gt == 1], v0[gt == 2],
                               alternative="two-sided").pvalue)
except Exception:
    pass

print("\n" + "=" * 96)
print("RECALL BY SPEED BAND vs BY DIRECTION   (median v0 over GT turns = %.3f)" % med)
print("=" * 96)
print("  %-12s | %-22s | %-22s || %-22s | %s"
      % ("arm", "SLOW", "FAST", "LEFT", "RIGHT"))
for tag in tags:
    D = load(tag)
    hit = (lab(D["cl"]) == gt).astype(float)
    def f(m):
        r = _ci.episode_cluster_bootstrap(hit[m], eid[m], n_boot=2000)
        return "%.4f [%.4f,%.4f]" % (r["mean"], r["lo"], r["hi"])
    print("  %-12s | %-22s | %-22s || %-22s | %s"
          % (tag, f(turn & (v0 <= med)), f(turn & (v0 > med)),
             f(gt == 1), f(gt == 2)))

print("\n" + "=" * 96)
print("MATCHED-SPEED CONTRAST — does LEFT still lose INSIDE a speed band?")
print("=" * 96)
for band, mb in (("SLOW", turn & (v0 <= med)), ("FAST", turn & (v0 > med))):
    for tag in tags:
        D = load(tag)
        hit = lab(D["cl"]) == gt
        L, R = mb & (gt == 1), mb & (gt == 2)
        print("  %-5s %-12s LEFT %d/%-2d   RIGHT %d/%-2d"
              % (band, tag, int(hit[L].sum()), int(L.sum()),
                 int(hit[R].sum()), int(R.sum())))
print("\n  ⇒ SLOW ~ FAST with LEFT << RIGHT, and LEFT losing inside BOTH bands, "
      "is DIRECTION, not speed.")
