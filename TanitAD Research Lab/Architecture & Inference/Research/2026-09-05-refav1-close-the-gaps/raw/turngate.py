#!/usr/bin/env python3
"""CAN THE PLANNER PASS THE TURN GATE AT ALL, AT THIS PANEL'S SPEEDS? 0 GPU.

⛔ THE ARITHMETIC NOBODY RAN. The tactical labeller's v1 lateral gate is
`|dyaw| > YAW_TURN_RAD = 0.15` rad (refc_tactical.py:148), and dyaw over the
2.0 s plan window is `integral(kappa * v) dt`. At this panel's TURN_L speed
(v0 mean 2.851 m/s) that is ~kappa * 5.7 m. So the gate demands

    kappa >= 0.15 / (v0 * T)

and NO planner can register a turn below it, whatever the goal commands.

This file computes, per window, the dyaw each arm's plan ACTUALLY produces
(integrating the plan's own speed profile, not a constant-speed approximation)
and checks it against BOTH gates the programme owns:
    v1  |dyaw| > 0.15 rad          <- the one the eval uses (kappa=None branch)
    v2  |kappa_mean| >= 1/60       <- CURV_TURN_MAN_PER_M

⛔ CONTROL: the GROUND TRUTH ('g') must pass the gate on the windows the eval
labels as turns -- otherwise the gate and the label disagree and nothing here
is readable.

ASCII output only.
"""
import glob, os, sys
import numpy as np
from tanitad.models.vocab_v7 import TACTICAL_LAT_ACTIONS_V7
from tanitad.refs.refc_tactical import YAW_TURN_RAD, CURV_TURN_MAN_PER_M
LAT = list(TACTICAL_LAT_ACTIONS_V7)
R = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/refav1_margin/p4out"
DT = 0.2

def load(tag):
    d = {}
    for f in sorted(glob.glob(os.path.join(R, "dump_"+tag, "decisions", "*.npz"))):
        z = np.load(f, allow_pickle=True); e = int(os.path.basename(f)[2:5])
        for i, w in enumerate(z["ws"].tolist()):
            d[(e, int(w))] = dict(ctl=z["cl_controls"][i], tok=LAT[int(z["goal_lat_cl"][i])],
                                  lab=int(z["lat_label"][i]) if "lat_label" in z.files else -100)
    for f in sorted(glob.glob(os.path.join(R, "dump_"+tag, "ep*.npz"))):
        z = np.load(f, allow_pickle=True); e = int(z["eid"][0])
        for i, w in enumerate(z["ws"].tolist()):
            k = (e, int(w))
            if k in d: d[k]["v0"] = float(z["v0"][i]); d[k]["g"] = z["g"][i]
    return d

def dyaw_of(ctl, v0):
    """integrate the plan's OWN speed through its own accel channel."""
    a, k = ctl[:, 0], ctl[:, 1]
    v = v0 + np.cumsum(a) * DT - a * DT          # speed each step OPENS at
    v = np.maximum(v, 0.0)
    return float((k * v * DT).sum()), float(np.abs(k).mean())

print("=" * 100)
print("CAN THE PLANNER PASS THE TURN GATE AT ALL? v1 gate |dyaw| > %.2f rad; v2 |kappa| >= %.5f"
      % (YAW_TURN_RAD, CURV_TURN_MAN_PER_M))
print("=" * 100)
arms = ["ccos_argmax", "wk15", "wk151", "kamm07"]
data = {a: load(a) for a in arms}
keys = sorted(set.intersection(*[set(d) for d in data.values()]))
tl = [k for k in keys if data["ccos_argmax"][k]["tok"] == "TURN_L"]
print("  TURN_L-goal windows: n=%d   v0 mean %.3f m/s" %
      (len(tl), np.mean([data["ccos_argmax"][k]["v0"] for k in tl])))
print()
print("  %-14s %11s %11s %11s %11s" % ("arm", "med |dyaw|", "pass v1", "med |k|", "pass v2"))
print("  " + "-" * 62)
for a in arms:
    dy = [abs(dyaw_of(data[a][k]["ctl"], data[a][k]["v0"])[0]) for k in tl]
    km = [dyaw_of(data[a][k]["ctl"], data[a][k]["v0"])[1] for k in tl]
    print("  %-14s %11.4f %8d/%-3d %11.5f %8d/%-3d"
          % (a, float(np.median(dy)), sum(1 for x in dy if x > YAW_TURN_RAD), len(dy),
             float(np.median(km)), sum(1 for x in km if x >= CURV_TURN_MAN_PER_M), len(km)))
print()
print("  ⛔ THE THRESHOLD CURVATURE THE v1 GATE DEMANDS, per window (kappa_min = 0.15/(v0*T)):")
vs = np.array([data["ccos_argmax"][k]["v0"] for k in tl])
T = data["ccos_argmax"][tl[0]]["ctl"].shape[0] * DT
kmin = YAW_TURN_RAD / np.maximum(vs * T, 1e-6)
print("     T = %.1f s;  v0 p25/med/p75 = %.2f / %.2f / %.2f m/s"
      % (T, *np.percentile(vs, [25, 50, 75])))
print("     kappa_min p25/med/p75 = %.4f / %.4f / %.4f 1/m  (= radius %.0f / %.0f / %.0f m)"
      % (*np.percentile(kmin, [25, 50, 75]), *(1.0/np.percentile(kmin, [25, 50, 75]))))
print("     windows where kappa_min EXCEEDS the shipped command 0.08: %d/%d"
      % (int((kmin > 0.08).sum()), len(kmin)))
print("     windows where kappa_min EXCEEDS the CORRECTED command 0.02: %d/%d"
      % (int((kmin > 0.02).sum()), len(kmin)))
print()
print("  ⛔ CONTROL -- the GROUND TRUTH must pass the v1 gate on these windows:")
gy = []
for k in tl:
    g = data["ccos_argmax"][k]["g"]          # [H,2] displacements
    d = np.diff(g, axis=0)
    th = np.arctan2(d[:, 1], d[:, 0])
    gy.append(abs(float(np.unwrap(th)[-1] - np.unwrap(th)[0])))
print("     GT median |dyaw| = %.4f rad; passes v1 on %d/%d"
      % (float(np.median(gy)), sum(1 for x in gy if x > YAW_TURN_RAD), len(gy)))
