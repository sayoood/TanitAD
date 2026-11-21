# -*- coding: utf-8 -*-
"""P5 - WHERE THE REMAINING +0.2599 LIVES, so the NEXT lever is chosen on
attribution rather than on which hypothesis is most interesting.

`lonshift` cut the longitudinal gap to `ha0_ext` by 47 % (+0.4862 -> +0.2599).
This asks the same question of the residual that `lon_attribution.py` asked of
the original: which windows carry it, and is that a property of the PLANNER or
of the windows? The floor's own error per stratum is the control -- it knows
nothing about the decoded token, so if IT splits the same way the split is a
property of the data and no lever can address it.

Zero GPU. ASCII only.
"""
import glob
import sys

import numpy as np
import torch

from tanitad.models.vocab_v7 import (TACTICAL_LAT_ACTIONS_V7,
                                     TACTICAL_LON_ACTIONS_V7)
from tanitad.refs import refa_v1 as R
from taniteval import four_families as ff

sys.path.insert(0, "C:/Users/Admin/tanitad-wt/taniteval/tools")
from refav1_arm import _components                                   # noqa: E402

LAT_V70, LON_V70 = list(TACTICAL_LAT_ACTIONS_V7), list(TACTICAL_LON_ACTIONS_V7)
DT, K, OP = 0.2, 10, 30
P = "C:/Users/Admin/refav1_margin/p4out"
ARM = sys.argv[1] if len(sys.argv) > 1 else "lonshift"


def load(tag):
    G, V0, CL, HA0E, A0, LAT, LON, CLC = [], [], [], [], [], [], [], []
    d = P + "/dump_" + tag
    for e, w in zip(sorted(glob.glob(d + "/ep*.npz")),
                    sorted(glob.glob(d + "/decisions/ep*.npz"))):
        with np.load(e) as z, np.load(w) as y:
            G.append(z["g"]); V0.append(z["v0"]); CL.append(z["cl"])
            HA0E.append(z["ha0_ext"]); A0.append(y["ha0_ext_controls"][:, 0, 0])
            LAT.append(y["goal_lat_cl"]); LON.append(y["goal_lon_cl"])
            CLC.append(y["cl_controls"])
    return [np.concatenate(x) for x in (G, V0, CL, HA0E, A0, LAT, LON, CLC)]


G, V0, CL, HA0E, A0, LAT, LON, CLC = load(ARM)
n = len(V0)
lat_s = [LAT_V70[i] for i in LAT]
lon_s = [LON_V70[i] for i in LON]
Gg = ff._seq_geometry(torch.as_tensor(G).float(), DT)
dv_gt = Gg["speed"][:, -1].numpy() - V0
c_cl = _components(CL.astype(np.float32), G.astype(np.float32), DT)
c_fl = _components(HA0E.astype(np.float32), G.astype(np.float32), DT)
d_s = c_cl["LON_speed_mae_mps"] - c_fl["LON_speed_mae_mps"]


def v_t_of(lon, v0):
    if lon == "HOLD":
        return 0.0
    if lon == "CREEP":
        return R.GOAL_CREEP_MPS
    if lon == "ADAPT_SPEED_FOR_CURVE":
        return min(v0, R.GOAL_CURVE_VMAX_MPS)
    return max(0.0, v0 + R.GOAL_LON_DV_MPS.get(lon, 0.0))


maint = np.array([abs(v_t_of(lon_s[i], float(V0[i])) - float(V0[i])) < 1e-9
                  for i in range(n)])
clip = np.abs(A0) > R.GOAL_A_MAX
lowv = V0 < 2.0
over = np.abs(CLC[:, :, 0]).max(1) >= R.GOAL_A_MAX - 1e-6   # plan saturates a_max

print("# arm %s   n = %d windows" % (ARM, n))
print("# per-window paired `cl - ha0_ext` on LON_speed_mae; the panel mean is")
print("# the number the episode-cluster bootstrap reports.")
print("  PANEL MEAN = %+0.4f  (median %+0.4f)" % (d_s.mean(), np.median(d_s)))
print()
hdr = "%-38s %5s %11s %11s %11s" % ("stratum", "n", "mean d", "share %", "floor own")
print(hdr)
print("-" * len(hdr))
tot = d_s.sum()
for name, m in (("ALL", np.ones(n, bool)),
                ("MAINTAIN branch (a_shift acts)", maint),
                ("NON-maintain", ~maint),
                ("|a0| > GOAL_A_MAX (hint clipped)", clip),
                ("|a0| <= GOAL_A_MAX", ~clip),
                ("v0 < 2 m/s (near-stationary)", lowv),
                ("v0 >= 2 m/s", ~lowv),
                ("plan SATURATES |a| = GOAL_A_MAX", over),
                ("plan below the a_max clip", ~over),
                ("GT |dv| >= 1.0 (GT-LON stratum)", np.abs(dv_gt) >= 1.0),
                ("GT |dv| <  1.0", np.abs(dv_gt) < 1.0)):
    if m.sum() == 0:
        print("%-38s %5d  (empty)" % (name, 0))
        continue
    print("%-38s %5d %11.4f %11.1f %11.4f"
          % (name, m.sum(), d_s[m].mean(), 100.0 * d_s[m].sum() / tot,
             c_fl["LON_speed_mae_mps"][m].mean()))
print()
print("# `floor own` is ha0_ext's OWN LON speed MAE on that stratum. It knows")
print("# nothing about the decoded token, so where IT splits the same way, the")
print("# split is a property of the DATA and no planner lever can address it.")
print()
print("== the a_max ceiling, which is the one lever this table can price ==")
print("  windows whose plan saturates |a| = GOAL_A_MAX = %.1f : %d / %d (%.1f %%)"
      % (R.GOAL_A_MAX, over.sum(), n, 100.0 * over.mean()))
print("  windows whose HINT a0 was clipped to GOAL_A_MAX     : %d / %d (%.1f %%)"
      % (clip.sum(), n, 100.0 * clip.mean()))
print("  GT |accel| p90 over the horizon                     : %.4f m/s2"
      % np.percentile(np.abs(Gg["accel"].numpy()), 90))
print("  GT |accel| max                                      : %.4f m/s2"
      % np.abs(Gg["accel"].numpy()).max())
