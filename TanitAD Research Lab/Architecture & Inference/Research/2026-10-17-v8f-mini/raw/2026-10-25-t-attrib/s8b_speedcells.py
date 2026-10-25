"""s8b_speedcells.py -- the SPEED-kind cells (v_a, v_end, v_lo, v_hi, lon_v_target) against the trivial 'carry the ego speed' predictor.
EXPLORATORY, MEASURED on eval-141 (2,365 windows with a v9 row). Raw units (m/s) = SD units x sd + mu with the standardiser recovered in s1.
Predictors that use only a number T already reads at inference (T's input p10f carries the ego kinematics, v8f_mini.py:556-558 / _ego): the ego speed at the T tick
(poses[2(t-a_t),3]); linear least squares of the cell on it (fit on the SAME windows: an optimistic in-sample bound for the trivial predictor).
"""
import numpy as np
from tcommon import *

T = load()
has = T["has"]
mu, sd = T["mu"], T["sd"]
G = T["gt_std"]
v = T["v_T"]
print(f"{'cell':16s} {'n':>5s} {'sd m/s':>6s} | {'corr(raw,v_T)':>13s} | {'RMSE m/s: v_T as is':>19s} {'lin fit on v_T':>14s} {'T s0':>6s} {'T s1':>6s} | {'bias(T-GT) s0 m/s':>17s} {'T RMSE vs v_T-lin':>17s}")
for c in (22, 23, 24, 25, 12):
    m = np.isfinite(G[:, c]) & has
    raw = G[m, c] * sd[c] + mu[c]
    x = v[m]
    cor = np.corrcoef(raw, x)[0, 1]
    r_asis = np.sqrt(np.mean((x - raw) ** 2))
    A = np.vstack([x, np.ones(len(x))]).T
    co = np.linalg.lstsq(A, raw, rcond=None)[0]
    r_lin = np.sqrt(np.mean((A @ co - raw) ** 2))
    out = []
    for arm in ("M2b_s0", "M2b_s1"):
        r = resid(T, arm)[m, c] * sd[c]
        out.append((np.sqrt(np.mean(r ** 2)), r.mean()))
    print(f"{CONS_NAMES[c]:16s} {m.sum():5d} {sd[c]:6.2f} | {cor:13.4f} | {r_asis:19.3f} {r_lin:14.3f} {out[0][0]:6.3f} {out[1][0]:6.3f} | {out[0][1]:17.3f} {out[0][0] / r_lin:17.2f}")
# where is T's v_a error: by speed band
c = 22
m = np.isfinite(G[:, c]) & has
raw = G[m, c] * sd[c] + mu[c]
r = resid(T, "M2b_s0")[m, c] * sd[c]
b = band(v[m], [1, 5, 10, 15], ["<1", "1-5", "5-10", "10-15", ">=15"])
print("\nT s0 v_a error (m/s) by ego speed band:")
for k in ["<1", "1-5", "5-10", "10-15", ">=15"]:
    mm = b == k
    print(f"   {k:6s} n {mm.sum():4d} RMSE {np.sqrt(np.mean(r[mm] ** 2)):.3f} bias {r[mm].mean():+.3f}")
