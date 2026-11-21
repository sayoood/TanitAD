# -*- coding: utf-8 -*-
"""P2c - WHERE THE +0.4862 LONGITUDINAL GAP LIVES, and therefore the CEILING on
what `a_sustain` alone can deliver at the ARM level.

`a_sustain` touches ONLY the maintain branch. If the gap to `ha0_ext` were
concentrated on the OTHER windows, the lever would be aimed at the wrong place
and no arm could rescue it -- so this is the number that says in advance how
much of the gap is even addressable. Recorded BEFORE the arm runs.

Also quantifies the `GOAL_A_MAX` clip, because `a_sustain = a0` is clipped to
+-1.5 and the corpus's a0 reaches -2.310 / +1.870.

Zero GPU. ASCII only.
"""
import glob
import os
import sys

import numpy as np
import torch

from tanitad.models.vocab_v7 import (TACTICAL_LAT_ACTIONS_V7,
                                     TACTICAL_LON_ACTIONS_V7)
from tanitad.refs import refa_v1 as R

sys.path.insert(0, "C:/Users/Admin/tanitad-wt/taniteval/tools")
from refav1_arm import _components                                # noqa: E402

LAT_V70, LON_V70 = list(TACTICAL_LAT_ACTIONS_V7), list(TACTICAL_LON_ACTIONS_V7)
DT, K, OP = 0.2, 10, 30
P = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/refav1_margin/p4out/dump_wk15"

G, V0, CL, HA0E, A0, LAT, LON = [], [], [], [], [], [], []
for e, d in zip(sorted(glob.glob(P + "/ep*.npz")),
                sorted(glob.glob(P + "/decisions/ep*.npz"))):
    with np.load(e) as z, np.load(d) as w:
        G.append(z["g"]); V0.append(z["v0"]); CL.append(z["cl"])
        HA0E.append(z["ha0_ext"]); A0.append(w["ha0_ext_controls"][:, 0, 0])
        LAT.append(w["goal_lat_cl"]); LON.append(w["goal_lon_cl"])
G = np.concatenate(G); V0 = np.concatenate(V0); CL = np.concatenate(CL)
HA0E = np.concatenate(HA0E); A0 = np.concatenate(A0)
LAT = np.concatenate(LAT); LON = np.concatenate(LON)
n = len(V0)
lat_s = [LAT_V70[i] for i in LAT]; lon_s = [LON_V70[i] for i in LON]


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
c_cl = _components(CL.astype(np.float32), G.astype(np.float32), DT)
c_fl = _components(HA0E.astype(np.float32), G.astype(np.float32), DT)

print("# dump %s   n = %d windows" % (P, n))
print("# per-window PAIRED difference `cl - ha0_ext`, split by the branch")
print("# `a_sustain` touches. The panel mean over ALL windows is the +0.4862")
print("# reported by the episode-cluster bootstrap; this is its ATTRIBUTION.")
print()
hdr = "%-34s %5s %12s %12s %12s" % ("stratum", "n", "LON speed", "LON accel", "ADE")
print(hdr)
print("-" * len(hdr))
for name, m in (("ALL windows", np.ones(n, bool)),
                ("MAINTAIN branch (a_sustain acts)", maint),
                ("NON-maintain (a_sustain inert)", ~maint)):
    d_s = (c_cl["LON_speed_mae_mps"] - c_fl["LON_speed_mae_mps"])[m]
    d_a = (c_cl["LON_accel_mae_mps2"] - c_fl["LON_accel_mae_mps2"])[m]
    d_e = (c_cl["ade_m"] - c_fl["ade_m"])[m]
    print("%-34s %5d %12.4f %12.4f %12.4f"
          % (name, m.sum(), d_s.mean(), d_a.mean(), d_e.mean()))
tot = (c_cl["LON_speed_mae_mps"] - c_fl["LON_speed_mae_mps"]).sum()
part = (c_cl["LON_speed_mae_mps"] - c_fl["LON_speed_mae_mps"])[maint].sum()
print()
print("  ADDRESSABLE FRACTION of the LON-speed gap (share of the total")
print("  window-summed deficit that sits on the maintain branch): %.1f %%"
      % (100.0 * part / tot))
print()
print("== the GOAL_A_MAX clip on `a_sustain = a0` ==")
clip = np.abs(A0) > R.GOAL_A_MAX
print("  |a0| > GOAL_A_MAX (%.1f): %d / %d windows (%.1f %%); on the maintain"
      % (R.GOAL_A_MAX, clip.sum(), n, 100 * clip.mean()))
print("  branch specifically: %d / %d (%.1f %%)"
      % ((clip & maint).sum(), maint.sum(),
         100.0 * (clip & maint).sum() / max(maint.sum(), 1)))
print("  a0 range: [%+0.3f, %+0.3f]; median |a0| on the maintain branch %.3f"
      % (A0.min(), A0.max(), np.median(np.abs(A0[maint]))))
print()
print("== SANITY: the floor's own LON error, by branch (it should NOT differ")
print("   much -- ha0_ext knows nothing about the decoded token) ==")
for name, m in (("MAINTAIN", maint), ("NON-maintain", ~maint)):
    print("  %-13s ha0_ext LON speed MAE mean %.4f   cl %.4f"
          % (name, c_fl["LON_speed_mae_mps"][m].mean(),
             c_cl["LON_speed_mae_mps"][m].mean()))
