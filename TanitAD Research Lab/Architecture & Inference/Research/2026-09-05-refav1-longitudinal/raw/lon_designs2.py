# -*- coding: utf-8 -*-
"""P5b - D4: raise the GOAL's acceleration clip to the PLANNER's own box.

MEASURED reason (`lon_residual.py` on `lonshift`'s dump): the 6 windows whose
`a0` hint is clipped by `GOAL_A_MAX = 1.5` are 15 % of the grid but carry
**47.4 %** of the remaining +0.2599 longitudinal gap, at mean +0.8212 against
+0.1609 on the other 34 -- a 5.1x concentration -- while the FLOOR's own error
on those windows is only 1.19x its error elsewhere. The split is a property of
the PLANNER, not of the windows.

And the ceiling is not the actuator's: `PlanConfig.a_max = 4.0` while
`GOAL_A_MAX = 1.5`, which the source documents as "DV_BRAKE_MS/s; = the
decel_1.5 floor" -- a BRAKING constant used as a symmetric global clip on the
goal. The corpus reaches |accel| 3.4095 max / 1.8443 p90, inside the planner's
box and outside the goal's.

Designs scored here, all hinted on the MEASURED a0, decoded LAT token held
constant so the lateral channel is not a second variable:

  SHIPPED   the decoded token's canonical profile
  D2        a_shift, clip GOAL_A_MAX = 1.5          (the landed `lonshift`)
  D4a       a_shift, clip = PlanConfig.a_max = 4.0
  D4b       a_shift, clip = 2.5                     (a middle rung)
  FLOOR     ha0_ext

Zero GPU. ASCII only.
"""
import glob
import sys

import numpy as np
import torch

from tanitad.models.vocab_v7 import (TACTICAL_LAT_ACTIONS_V7,
                                     TACTICAL_LON_ACTIONS_V7)
from tanitad.refs import refa_v1 as R
from tanitad.refs.refa_v1_plan import PlanConfig
from taniteval import four_families as ff

sys.path.insert(0, "C:/Users/Admin/tanitad-wt/taniteval/tools")
from refav1_arm import _components, paths_from_controls              # noqa: E402

LAT_V70, LON_V70 = list(TACTICAL_LAT_ACTIONS_V7), list(TACTICAL_LON_ACTIONS_V7)
DT, K, OP = 0.2, 10, 30
P = "C:/Users/Admin/refav1_margin/p4out/dump_wk15"

G, V0, CL, HA0E, A0, LAT, LON = [], [], [], [], [], [], []
for e, w in zip(sorted(glob.glob(P + "/ep*.npz")),
                sorted(glob.glob(P + "/decisions/ep*.npz"))):
    with np.load(e) as z, np.load(w) as y:
        G.append(z["g"]); V0.append(z["v0"]); CL.append(z["cl"])
        HA0E.append(z["ha0_ext"]); A0.append(y["ha0_ext_controls"][:, 0, 0])
        LAT.append(y["goal_lat_cl"]); LON.append(y["goal_lon_cl"])
G = np.concatenate(G); V0 = np.concatenate(V0); CL = np.concatenate(CL)
HA0E = np.concatenate(HA0E); A0 = np.concatenate(A0)
LAT = np.concatenate(LAT); LON = np.concatenate(LON)
n = len(V0)
lat_s = [LAT_V70[i] for i in LAT]
lon_s = [LON_V70[i] for i in LON]
PLAN_A_MAX = float(PlanConfig.a_max)
print("# GOAL_A_MAX = %.2f   PlanConfig.a_max = %.2f   ratio %.2fx"
      % (R.GOAL_A_MAX, PLAN_A_MAX, PLAN_A_MAX / R.GOAL_A_MAX))
print("# n = %d windows (dump_wk15, the shared baseline grid)" % n)
print()


def v_t_of(lon, v0):
    if lon == "HOLD":
        return 0.0
    if lon == "CREEP":
        return R.GOAL_CREEP_MPS
    if lon == "ADAPT_SPEED_FOR_CURVE":
        return min(v0, R.GOAL_CURVE_VMAX_MPS)
    return max(0.0, v0 + R.GOAL_LON_DV_MPS.get(lon, 0.0))


def relative(lon, v0):
    return lon in R.GOAL_LON_DV_MPS or (lon == "ADAPT_SPEED_FOR_CURVE"
                                        and v0 <= R.GOAL_CURVE_VMAX_MPS)


def build(shift, clip):
    """`shift=False` -> the shipped profile. Otherwise D2 with the given clip
    on BOTH the hint and the profile, which is what changing GOAL_A_MAX does."""
    out = np.zeros((n, K, 2), np.float32)
    for i in range(n):
        v0 = float(V0[i]); lon = lon_s[i]
        c = R.canonical_controls(lat_s[i], lon, v0, OP, DT)[:K].clone()
        if shift and relative(lon, v0):
            vt = v_t_of(lon, v0) + float(np.clip(A0[i], -clip, clip)) * R.GOAL_REACH_S
            a = np.zeros(OP); v = v0
            for t in range(OP):
                ai = float(np.clip((vt - v) / R.GOAL_REACH_S, -clip, clip))
                a[t] = ai; v += ai * DT
            c[:, 0] = torch.as_tensor(a[:K], dtype=c.dtype)
        out[i] = paths_from_controls(c, v0, DT, K)[0].numpy()
    return out


def build_tau(tau):
    """D5: keep D2's TARGET (v_t + a0 * 2.0 s, the plan horizon) but reach it
    with a SHORTER time constant, so the token delivers its own semantics
    INSIDE the optimised 2 s window instead of 65 % of them."""
    out = np.zeros((n, K, 2), np.float32)
    for i in range(n):
        v0 = float(V0[i]); lon = lon_s[i]
        c = R.canonical_controls(lat_s[i], lon, v0, OP, DT)[:K].clone()
        if relative(lon, v0):
            vt = v_t_of(lon, v0) + float(np.clip(A0[i], -R.GOAL_A_MAX,
                                                 R.GOAL_A_MAX)) * R.GOAL_REACH_S
            a = np.zeros(OP); v = v0
            for t in range(OP):
                ai = float(np.clip((vt - v) / tau, -R.GOAL_A_MAX, R.GOAL_A_MAX))
                a[t] = ai; v += ai * DT
            c[:, 0] = torch.as_tensor(a[:K], dtype=c.dtype)
        out[i] = paths_from_controls(c, v0, DT, K)[0].numpy()
    return out


rows = [("GT identity control", G),
        ("ha0_ext FLOOR", HA0E),
        ("SHIPPED canon @ decoded", build(False, R.GOAL_A_MAX)),
        ("D2  a_shift clip 1.5 (landed)", build(True, R.GOAL_A_MAX)),
        ("D4b a_shift clip 2.5", build(True, 2.5)),
        ("D4a a_shift clip 4.0 = plan box", build(True, PLAN_A_MAX)),
        ("D5a a_shift tau 1.0 s", build_tau(1.0)),
        ("D5b a_shift tau 0.6 s", build_tau(0.6)),
        ("D5c a_shift tau 2.0 s = D2 control", build_tau(R.GOAL_REACH_S))]

cf = _components(HA0E.astype(np.float32), G.astype(np.float32), DT)
clip6 = np.abs(A0) > R.GOAL_A_MAX
print("== MEAN PAIRED difference vs the ha0_ext FLOOR (negative = beats it) ==")
h = ("%-32s %13s %13s %11s %14s"
     % ("design", "d LON speed", "d LON accel", "d ADE", "d LON on the 6"))
print(h); print("-" * len(h))
for nm, Pth in rows:
    c = _components(np.asarray(Pth, np.float32), G.astype(np.float32), DT)
    d = c["LON_speed_mae_mps"] - cf["LON_speed_mae_mps"]
    print("%-32s %13.4f %13.4f %11.4f %14.4f"
          % (nm, d.mean(),
             (c["LON_accel_mae_mps2"] - cf["LON_accel_mae_mps2"]).mean(),
             (c["ade_m"] - cf["ade_m"]).mean(), d[clip6].mean()))
print()
print("# 'd LON on the 6' is the mean paired difference on the SIX windows whose")
print("# a0 hint GOAL_A_MAX clips -- the 15 %% of the grid carrying 47.4 %% of")
print("# lonshift's residual. That column is what D4 exists to move.")
print("# GT |accel| p90 %.4f  max %.4f  -- inside the plan box, outside the goal's."
      % (np.percentile(np.abs(ff._seq_geometry(torch.as_tensor(G).float(), DT)["accel"].numpy()), 90),
         np.abs(ff._seq_geometry(torch.as_tensor(G).float(), DT)["accel"].numpy()).max()))
