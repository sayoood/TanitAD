# -*- coding: utf-8 -*-
"""P1e - THE DESIGN BAKE-OFF. `a_sustain` (D1) touches only the maintain branch,
which carries 61.8 % of the longitudinal gap; this asks whether a design that
covers ALL 40 windows does better, and it does it BEFORE spending a GPU hour.

Designs, all scored on the same windows, same integrator, same decoded LAT
token (so the LATERAL channel is held constant and this is one variable):

  SHIPPED  the decoded token's canonical profile
  D1       maintain branch only -> CONSTANT a = a0                (implemented)
  D2       ALL tokens -> target speed shifted by the a0 extrapolation:
           v_t' = v_t + a0 * GOAL_REACH_S, profile computed as shipped.
           "The tokens name a speed change relative to WHERE YOU ARE GOING,
            not relative to where you are." Reduces to a[0] = a0 on the
            maintain branch, so D1 is its special case at the first step.
  D3       ALL tokens -> CONSTANT a = clip(a0 + a_token[0])
  FLOOR    ha0_ext, for reference

Every design's hint is the MEASURED a0 (backward difference of past speeds at
t0), so every column below is REALISED w.r.t. the chooser -- there is no oracle
row here at all. ⛔ They remain vocabulary-expressivity rows, NOT planner
results.

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
from refav1_arm import _components, paths_from_controls              # noqa: E402

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
Gg = ff._seq_geometry(torch.as_tensor(G).float(), DT)
dv_gt = Gg["speed"][:, -1].numpy() - V0
THR = 1.0
stratum = np.abs(dv_gt) >= THR
CLIP = R.GOAL_A_MAX


def v_t_of(lon, v0):
    if lon == "HOLD":
        return 0.0
    if lon == "CREEP":
        return R.GOAL_CREEP_MPS
    if lon == "ADAPT_SPEED_FOR_CURVE":
        return min(v0, R.GOAL_CURVE_VMAX_MPS)
    return max(0.0, v0 + R.GOAL_LON_DV_MPS.get(lon, 0.0))


def profile(v0, v_t):
    a = np.zeros(OP); v = float(v0)
    for i in range(OP):
        ai = max(-CLIP, min(CLIP, (v_t - v) / R.GOAL_REACH_S))
        a[i] = ai; v += ai * DT
    return a


def build(design):
    out = np.zeros((n, K, 2), np.float32)
    for i in range(n):
        v0 = float(V0[i]); lon = lon_s[i]; a0 = float(A0[i])
        c = R.canonical_controls(lat_s[i], lon, v0, OP, DT)[:K].clone()
        vt = v_t_of(lon, v0)
        if design == "D1" and abs(vt - v0) < 1e-9:
            c[:, 0] = max(-CLIP, min(CLIP, a0))
        elif design == "D2":
            c[:, 0] = torch.as_tensor(
                profile(v0, vt + a0 * R.GOAL_REACH_S)[:K], dtype=c.dtype)
        elif design == "D3":
            a_tok = float(R.canonical_controls(lat_s[i], lon, v0, OP, DT)[0, 0])
            c[:, 0] = max(-CLIP, min(CLIP, a0 + a_tok))
        out[i] = paths_from_controls(c, v0, DT, K)[0].numpy()
    return out


rows = [("GT identity control", G, "must read EXACTLY 0"),
        ("cl = wk15 (T1 planner)", CL, ""),
        ("ha0_ext FLOOR", HA0E, ""),
        ("SHIPPED canon @ decoded", build("SHIPPED"), ""),
        ("D1 maintain -> const a0", build("D1"), "implemented"),
        ("D2 v_t + a0*REACH_S", build("D2"), "all tokens"),
        ("D3 const (a0 + a_token)", build("D3"), "all tokens")]

hdr = ("%-26s %12s %12s %12s %10s  %s"
       % ("design", "LONspd medAE", "LONacc medAE", "LONalong", "ADE med", "note"))
for title, mask in (("GT-LON stratum (n=%d)" % stratum.sum(), stratum),
                    ("ALL %d windows" % n, np.ones(n, bool))):
    print("== %s ==" % title)
    print(hdr); print("-" * len(hdr))
    for nm, Pth, note in rows:
        c = _components(np.asarray(Pth, np.float32), G.astype(np.float32), DT)
        print("%-26s %12.4f %12.4f %12.4f %10.4f  %s"
              % (nm, np.median(c["LON_speed_mae_mps"][mask]),
                 np.median(c["LON_accel_mae_mps2"][mask]),
                 np.median(c["LON_along_mae_m"][mask]),
                 np.median(c["ade_m"][mask]), note))
    print()

print("== MEAN PAIRED difference vs the FLOOR (the arm-level statistic) ==")
cf = _components(HA0E.astype(np.float32), G.astype(np.float32), DT)
print("%-26s %14s %14s %12s" % ("design", "d LON speed", "d LON accel", "d ADE"))
print("-" * 70)
for nm, Pth, _ in rows[1:]:
    c = _components(np.asarray(Pth, np.float32), G.astype(np.float32), DT)
    print("%-26s %14.4f %14.4f %12.4f"
          % (nm, (c["LON_speed_mae_mps"] - cf["LON_speed_mae_mps"]).mean(),
             (c["LON_accel_mae_mps2"] - cf["LON_accel_mae_mps2"]).mean(),
             (c["ade_m"] - cf["ade_m"]).mean()))
print()
print("# negative = better than ha0_ext. The `cl` row must reproduce the banked")
print("# +0.4862 / +0.4117 / +0.0162 -- if it does not, this script and the")
print("# episode-cluster bootstrap are not reading the same thing.")
