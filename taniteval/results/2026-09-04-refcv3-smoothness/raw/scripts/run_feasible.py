"""WP-6: IS THE ANCHOR VOCABULARY ACTUALLY KINEMATICALLY FEASIBLE?

`synth_anchor_pool` (refc.py:176-207) integrates a CONSTANT yaw_rate that is
INDEPENDENT of v, and clamps v at 0. `rollout_unicycle` (kinematic.py) — the
Alpamayo action space — uses ``yaw_rate = v * curvature`` and its docstring says
a head that "turns at a standstill is producing a path no vehicle drives".
The two integrators DISAGREE, and the anchors come from the first one.
"""
import sys, numpy as np, torch
sys.path.insert(0, r"C:/Users/Admin/_wp56"); sys.path.insert(0, r"C:/Users/Admin/_wp56/wp56")
import smooth_lib as S
from tanitad.refs.refc import default_anchors

MU_G = 0.7 * 9.81          # 6.867 m/s^2 — a dry-road Kamm circle
for nm, hz, t in (("refcv3   (6 s)", (5,10,15,20,30,40,50,60), S.T8),
                  ("refc-base (2 s)", (5,10,15,20), S.T4)):
    A = default_anchors(hz, 128, 4096, 0).numpy().astype(np.float64)
    g = S.geom(S.with_origin(A), t)
    v = g["v"]                                     # [128, S] per-segment speed
    kap = np.abs(g["kappa"])                       # [128, S-1]
    vv = 0.5 * (v[:, :-1] + v[:, 1:])
    a_lat = vv ** 2 * kap
    a_lon = np.abs(g["a"])
    a_tot = np.sqrt(a_lat ** 2 + a_lon ** 2)
    stall = (v.min(1) < 0.5)
    turn_while_stalled = stall & (np.degrees(np.abs(g["turn"])).max(1) > 1.0)
    print(f"\n== {nm} ==")
    print(f"  per-segment implied speed: min {v.min():6.2f}  p50 {np.median(v):6.2f}  max {v.max():6.2f} m/s")
    print(f"  lateral accel v^2*kappa  : p50 {np.nanmedian(a_lat):6.2f}  p90 "
          f"{np.nanpercentile(a_lat,90):6.2f}  max {np.nanmax(a_lat):7.2f} m/s^2")
    print(f"  |a_lon|                  : p50 {np.median(a_lon):6.2f}  max {a_lon.max():6.2f} m/s^2")
    print(f"  KAMM-CIRCLE VIOLATIONS (|a| > {MU_G:.2f} m/s^2): "
          f"{100*np.nanmean(a_tot>MU_G):5.2f} % of segment-vertices; "
          f"{int((np.nan_to_num(a_tot)>MU_G).any(1).sum()):3d}/128 anchors violate somewhere")
    print(f"  |a_lon| > 3.0 m/s^2 (comfort): {100*(a_lon>3.0).mean():5.2f} % of segments")
    print(f"  anchors that STALL (v<0.5 m/s somewhere): {stall.sum():3d}/128; "
          f"of those, still TURNING while stalled: {turn_while_stalled.sum():3d}"
          f"  <-- physically impossible under yaw_rate = v*kappa")
