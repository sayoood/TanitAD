"""Is the L/R turn-recall asymmetry a SPEED confound?

Zero GPU, from the banked 40-window dumps. The dyaw turn gate is
|dyaw| > 0.15 rad and dyaw ~= kappa * v * T, so the CURVATURE a plan needs in
order to be LABELLED a turn scales as 1/v -- and the curvature CHARGE it pays
scales as 1/v^2. If GT-left turns are systematically slower than GT-right ones,
a perfectly sign-symmetric cost suppresses LEFT first.
"""
import glob, json, os, sys, math, collections
import numpy as np
import torch

WT = "C:/Users/Admin/tanitad-wt"
for p in (WT + "/stack", WT + "/taniteval", WT):
    if p not in sys.path:
        sys.path.insert(0, p)
from tanitad.refs.refc_tactical import factor_from_kinematics, YAW_TURN_RAD
from taniteval import four_families as ff

P4 = "C:/Users/Admin/refav1_margin/p4out"
DT = 0.2
LAT = {0: "lane_keep", 1: "turn_left", 2: "turn_right"}

def load(tag):
    cat = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(P4, "dump_" + tag, "ep*.npz"))):
        d = np.load(p)
        n = d["g"].shape[0]
        for k in ("g", "cl", "ha0_ext", "ol", "ha"):
            cat[k].append(d[k])
        cat["v0"].append(d["v0"]); cat["eid"].append(np.repeat(d["eid"], n))
        cat["ws"].append(d["ws"])
    return {k: np.concatenate(v) for k, v in cat.items()}

def labels(A):
    t = torch.as_tensor(A).float()
    dy, dv, v0, v1, _ = ff.maneuver_kinematics(t, DT)
    lat, lon = factor_from_kinematics(dy, dv, v0, v1)
    return lat.numpy(), dy.numpy()

print("=" * 96)
print("1. THE SPEED OF A GT TURN, BY DIRECTION -- banked 40-window panel")
print("=" * 96)
D = load("wk15")
lat_g, dy_g = labels(D["g"])
v0 = D["v0"]
# the curvature a plan MUST sustain to clear the 0.15 rad gate at this speed
T = D["g"].shape[1] * DT
k_need = YAW_TURN_RAD / np.maximum(v0, 0.1) / T
print("  %-11s %3s | %8s %8s %8s | %10s %12s" %
      ("stratum", "n", "v0 med", "v0 mean", "v0 min", "k_need med", "charge~k^2 med"))
for c in (1, 2, 0):
    m = lat_g == c
    if not m.any():
        continue
    print("  %-11s %3d | %8.3f %8.3f %8.3f | %10.5f %12.3e" %
          (LAT[c], m.sum(), np.median(v0[m]), v0[m].mean(), v0[m].min(),
           np.median(k_need[m]), np.median(k_need[m] ** 2)))
rl, rr = np.median(k_need[lat_g == 1]), np.median(k_need[lat_g == 2])
print("  -> LEFT needs %.2fx the curvature of RIGHT to be LABELLED a turn, "
      "and pays %.2fx the W_KAPPA charge for it." % (rl / rr, (rl / rr) ** 2))

print()
print("=" * 96)
print("2. WHICH windows does each arm get right? -- ordered by v0")
print("=" * 96)
tags = ["cos_argmax", "ccos_argmax", "wk15", "wk151", "cos_wk"]
Ds = {t: load(t) for t in tags}
lat_p = {t: labels(Ds[t]["cl"])[0] for t in tags}
lat_floor = labels(D["ha0_ext"])[0]
lat_ol = labels(D["ol"])[0]
idx = np.argsort(v0)
print("  win  eid    v0   GT      | " + " ".join("%-11s" % t[:11] for t in tags)
      + " | ha0_ext     ol")
for i in idx:
    if lat_g[i] == 0:
        continue
    print("  %3d  %3d %6.2f  %-9s| " % (i, D["eid"][i], v0[i], LAT[lat_g[i]])
          + " ".join("%-11s" % LAT[lat_p[t][i]][:11] for t in tags)
          + " | %-11s %-11s" % (LAT[lat_floor[i]][:11], LAT[lat_ol[i]][:11]))

print()
print("=" * 96)
print("3. RECALL BY SPEED BAND (both directions pooled) -- the confound test")
print("=" * 96)
turn = lat_g > 0
med = np.median(v0[turn])
print("  median v0 over GT-turn windows = %.3f m/s" % med)
for t in tags:
    hit = (lat_p[t] == lat_g)
    slow = turn & (v0 <= med); fast = turn & (v0 > med)
    L = turn & (lat_g == 1); R = turn & (lat_g == 2)
    print("  %-13s  SLOW %d/%d = %.4f   FAST %d/%d = %.4f  |  "
          "LEFT %d/%d = %.4f   RIGHT %d/%d = %.4f"
          % (t, hit[slow].sum(), slow.sum(), hit[slow].mean(),
             hit[fast].sum(), fast.sum(), hit[fast].mean(),
             hit[L].sum(), L.sum(), hit[L].mean(),
             hit[R].sum(), R.sum(), hit[R].mean()))
print("  %-13s  (CONTROL, no planner) " % "ha0_ext", end="")
hit = (lat_floor == lat_g)
slow = turn & (v0 <= med); fast = turn & (v0 > med)
L = turn & (lat_g == 1); R = turn & (lat_g == 2)
print("SLOW %.4f FAST %.4f | LEFT %.4f RIGHT %.4f"
      % (hit[slow].mean(), hit[fast].mean(), hit[L].mean(), hit[R].mean()))

print()
print("=" * 96)
print("4. MATCHED-SPEED CONTRAST -- is there ANY left/right gap at matched v0?")
print("=" * 96)
print("  GT-turn windows with v0 > %.2f m/s (the band where RIGHT lives):" % med)
for t in tags:
    hit = (lat_p[t] == lat_g)
    L = turn & (lat_g == 1) & (v0 > med); R = turn & (lat_g == 2) & (v0 > med)
    print("    %-13s LEFT %d/%d   RIGHT %d/%d" % (t, hit[L].sum(), L.sum(),
                                                  hit[R].sum(), R.sum()))
print("  GT-turn windows with v0 <= %.2f m/s (the band where LEFT lives):" % med)
for t in tags:
    hit = (lat_p[t] == lat_g)
    L = turn & (lat_g == 1) & (v0 <= med); R = turn & (lat_g == 2) & (v0 <= med)
    print("    %-13s LEFT %d/%d   RIGHT %d/%d" % (t, hit[L].sum(), L.sum(),
                                                  hit[R].sum(), R.sum()))
