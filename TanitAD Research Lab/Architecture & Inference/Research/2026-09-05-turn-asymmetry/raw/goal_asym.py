"""P4 -- the DECODED GOAL TOKEN, and the plan's SIGNED CURVATURE. Zero GPU."""
import glob, os, sys, collections
import numpy as np
import torch
WT = "C:/Users/Admin/tanitad-wt"
for p in (WT + "/stack", WT + "/taniteval", WT):
    if p not in sys.path: sys.path.insert(0, p)
from tanitad.refs.refc_tactical import factor_from_kinematics
from tanitad.models.vocab_v7 import TACTICAL_LAT_ACTIONS_V7 as LATV
from taniteval import four_families as ff
P4, DT = "C:/Users/Admin/refav1_margin/p4out", 0.2
LAT = {0: "lane_keep", 1: "turn_left", 2: "turn_right"}
I_LK, I_TL, I_TR = LATV.index("LANE_KEEP"), LATV.index("TURN_L"), LATV.index("TURN_R")
print("vocab: LANE_KEEP=%d TURN_L=%d TURN_R=%d  (imported, not hand-written)"
      % (I_LK, I_TL, I_TR))

def load(tag):
    c = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(P4, "dump_" + tag, "ep*.npz"))):
        d = np.load(p); n = d["g"].shape[0]
        for k in ("g", "cl", "ha0_ext", "ol"): c[k].append(d[k])
        c["v0"].append(d["v0"]); c["eid"].append(np.repeat(d["eid"], n))
        q = np.load(os.path.join(P4, "dump_" + tag, "decisions", os.path.basename(p)))
        for k in ("goal_lat_cl", "cl_controls", "plan_source_cl", "nav_cmd"):
            c[k].append(q[k])
    return {k: np.concatenate(v) for k, v in c.items()}

def lab(A):
    t = torch.as_tensor(A).float()
    dy, dv, v0, v1, _ = ff.maneuver_kinematics(t, DT)
    return factor_from_kinematics(dy, dv, v0, v1)[0].numpy()

tags = ["cos_argmax", "ccos_argmax", "ccos_seed1", "ccosh_w000", "wk15", "wk151", "cos_wk"]
D = {t: load(t) for t in tags}
lat_g = lab(D["wk15"]["g"]); v0 = D["wk15"]["v0"]

print("\n" + "=" * 94)
print("1. CONTROL -- is the DECODED goal token identical across arms at the same seed?")
print("=" * 94)
ref = D["ccos_argmax"]["goal_lat_cl"]
for t in tags:
    same = int((D[t]["goal_lat_cl"] == ref).sum())
    print("  %-13s goal_lat identical to ccos_argmax on %2d/40 %s"
          % (t, same, "" if same == 40 else "  <-- DIFFERS"))
print("  (ccos_seed1 is the SAME head at a different PLAN seed: the goal decode "
      "must be identical; if it is, the goal is not a seed artefact.)")

print("\n" + "=" * 94)
print("2. THE DECODED GOAL vs the GROUND TRUTH DIRECTION (ccos_argmax head, all arms share it)")
print("=" * 94)
gl = ref
print("  decoded goal counts: LANE_KEEP=%d TURN_L=%d TURN_R=%d  other=%d"
      % ((gl == I_LK).sum(), (gl == I_TL).sum(), (gl == I_TR).sum(),
         ((gl != I_LK) & (gl != I_TL) & (gl != I_TR)).sum()))
print("  GT counts:           lane_keep=%d turn_left=%d turn_right=%d"
      % ((lat_g == 0).sum(), (lat_g == 1).sum(), (lat_g == 2).sum()))
print("\n  goal-token confusion (GT rows, decoded-goal cols):")
print("        %10s %10s %10s %10s" % ("LANE_KEEP", "TURN_L", "TURN_R", "other"))
for c, nm in ((0, "GT LK"), (1, "GT TL"), (2, "GT TR")):
    m = lat_g == c
    o = m.sum() - (gl[m] == I_LK).sum() - (gl[m] == I_TL).sum() - (gl[m] == I_TR).sum()
    print("  %-5s %10d %10d %10d %10d" % (nm, (gl[m] == I_LK).sum(),
          (gl[m] == I_TL).sum(), (gl[m] == I_TR).sum(), o))
print("\n  ** GOAL RECALL (the head, before ANY cost): "
      "left %d/%d = %.4f   right %d/%d = %.4f **"
      % ((gl[lat_g == 1] == I_TL).sum(), (lat_g == 1).sum(),
         (gl[lat_g == 1] == I_TL).mean(),
         (gl[lat_g == 2] == I_TR).sum(), (lat_g == 2).sum(),
         (gl[lat_g == 2] == I_TR).mean()))

print("\n" + "=" * 94)
print("3. THE PLAN'S SIGNED CURVATURE -- emission by direction, all 40 windows")
print("=" * 94)
print("  %-13s | %8s %8s | %8s %8s | %10s" %
      ("arm", "n k>+.02", "n k<-.02", "sum k+", "sum k-", "mean signed"))
for t in tags:
    k = D[t]["cl_controls"][:, :, 1]
    kw = k[np.arange(k.shape[0]), np.abs(k).argmax(1)]      # the window's peak signed kappa
    print("  %-13s | %8d %8d | %8.3f %8.3f | %10.5f"
          % (t, (kw > 0.02).sum(), (kw < -0.02).sum(),
             kw[kw > 0].sum(), kw[kw < 0].sum(), kw.mean()))
print("  (+kappa = LEFT; kinematic.py: x forward, y left)")

print("\n" + "=" * 94)
print("4. ON GT-LEFT WINDOWS ONLY: what does each arm emit, and what was the goal?")
print("=" * 94)
mL = lat_g == 1
print("  win eid   v0  goal_tok | " + " ".join("%-9s" % t[:9] for t in tags))
for i in np.flatnonzero(mL)[np.argsort(v0[mL])]:
    g_ = LATV[gl[i]]
    row = []
    for t in tags:
        k = D[t]["cl_controls"][i, :, 1]
        kw = k[np.abs(k).argmax()]
        row.append("%+.4f" % kw)
    print("  %3d %3d %5.2f  %-9s| %s" % (i, D["wk15"]["eid"][i], v0[i], g_[:9],
                                         " ".join("%-9s" % r for r in row)))
print("\n  GT-RIGHT windows:")
for i in np.flatnonzero(lat_g == 2)[np.argsort(v0[lat_g == 2])]:
    g_ = LATV[gl[i]]
    row = []
    for t in tags:
        k = D[t]["cl_controls"][i, :, 1]
        kw = k[np.abs(k).argmax()]
        row.append("%+.4f" % kw)
    print("  %3d %3d %5.2f  %-9s| %s" % (i, D["wk15"]["eid"][i], v0[i], g_[:9],
                                         " ".join("%-9s" % r for r in row)))
