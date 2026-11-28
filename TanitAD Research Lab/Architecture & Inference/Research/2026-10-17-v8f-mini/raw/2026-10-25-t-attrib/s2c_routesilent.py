"""s2c_routesilent.py -- who are the 210 route-silent turn-label windows? (EXPLORATORY, MEASURED eval-141.) Raw turn_t_start_s = std*sd+mu (s1 standardiser)."""
import numpy as np
from tcommon import *
T = load(); has = T["has"]; G = T["gt_std"]; mu, sd = T["mu"], T["sd"]
turn = np.isfinite(G[:, 1]) & has
nav0 = T["nav_tok"] == 0
ts = G[:, 1] * sd[1] + mu[1]
te = G[:, 2] * sd[2] + mu[2]
E0, P = sq(T, "M2b_s0")
cols = list(range(1, 10))
print("turn-label windows by raw turn_t_start_s (s) x nav token (FOLLOW=route silent):")
print(f"{'t_start bin':14s} {'nav FOLLOW n':>12s} {'nav TURN n':>10s} | turn-cell RMSE (s0): {'FOLLOW':>7s} {'TURN':>7s} | share of all turn-cell SSE: {'FOLLOW':>7s} {'TURN':>7s}")
tot = E0[turn][:, cols].sum()
for lo, hi in ((-9, 0.2), (0.2, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 9.0)):
    b = turn & (ts >= lo) & (ts < hi)
    a, c = b & nav0, b & ~nav0
    r = lambda m: np.sqrt(E0[m][:, cols].sum() / max(P[m][:, cols].sum(), 1)) if m.sum() else float("nan")
    print(f"[{lo:4.1f},{hi:4.1f})    {a.sum():12d} {c.sum():10d} |                       {r(a):7.3f} {r(c):7.3f} |                           {E0[a][:, cols].sum() / tot:7.3f} {E0[c][:, cols].sum() / tot:7.3f}")
print("\nroute-silent turn windows: lat_cls mix:", {LAT_NAMES[k]: int(((T['lat_cls'] == k) & turn & nav0).sum()) for k in (-100, 0, 1, 2)},
      "; yaw-rate/curvature: GT |dpsi6| median", round(float(np.median(np.abs(T['dpsi6'][turn & nav0]))), 1), "deg vs route-speaking", round(float(np.median(np.abs(T['dpsi6'][turn & ~nav0]))), 1))
print("nav_d (m, route distance to next turn) valid in route-silent turn windows: n/a (token FOLLOW carries args_valid only for TURN tokens by construction, v8f_mini_data.py:199-206)")
