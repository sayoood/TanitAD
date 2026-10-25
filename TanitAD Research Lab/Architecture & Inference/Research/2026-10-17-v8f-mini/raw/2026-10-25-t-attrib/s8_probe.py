"""s8_probe.py -- Q4 information probe: is the missing information in a SIMPLE non-vision input set, or in privileged lead information, or in neither?
(EXPLORATORY, ESTIMATED; ONLY 141 episodes / 2,365 windows are local, so the probes are trained on far less than T's 4,512 clips -- a probe that
matches T is informative, a probe that loses to T is NOT evidence that the information is absent.)

Probe A  = what T already reads that is not vision: ego speed / accel / yaw rate at the T tick + 1 s and 2 s speed history, nav token + nav distance, speed limit.
Probe B  = A + PRIVILEGED lead information (lead present, gap, causal closing speed from the lead-gap series); NOT an admissible input as built (it is
           label-side tracking), used only to bound how much lead information could add.
Both: HistGradientBoostingRegressor (depth 3, 80 iters, lr 0.1, min_leaf 30), 5-fold GroupKFold by EPISODE, per cell, on the cells present; scored in the SAME SD units
as T (M2b_s0 and M2b_s1) over the same windows/cells.
"""
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from tcommon import *
from s3_tail import KIND

warnings.filterwarnings("ignore")


def main():
    T = load()
    Kz = dict(np.load(S + "raw_in/eval141_kin.npz", allow_pickle=True))
    Jz = dict(np.load(S + "raw_in/join_eval141.npz", allow_pickle=True))
    eid, tt, a_t, has = T["ci"], T["tt"], T["a_t"], T["has"]
    G = T["gt_std"]
    poses = Kz["poses"].astype(np.float64)
    koff = np.concatenate([[0], np.cumsum(Kz["n"])])
    N = len(eid)
    F = np.zeros((N, 8))
    lead = np.zeros((N, 3))
    for k in range(len(Kz["clip"])):
        P = poses[koff[k]:koff[k + 1]]
        m = np.where(eid == k)[0]
        t = tt[m] - a_t[m]
        k2 = 2 * t
        v = P[:, 3]
        F[m, 0] = v[k2]
        F[m, 1] = (v[k2] - v[np.maximum(k2 - 2, 0)]) / 0.2
        yaw = P[:, 2]
        F[m, 2] = (yaw[k2] - yaw[np.maximum(k2 - 1, 0)]) / 0.1
        F[m, 3] = v[np.maximum(k2 - 5, 0)]
        F[m, 4] = v[np.maximum(k2 - 10, 0)]
        o = int(Jz["off"][k])
        n = int(Jz["n"][k])
        gap = Jz["lead_gap"][o:o + n].astype(np.float64)
        lv = Jz["lead_valid"][o:o + n].astype(np.float64)
        lead[m, 0] = lv[t]
        lead[m, 1] = np.nan_to_num(gap[t], nan=-1.0)
        g_prev = gap[np.maximum(t - 2, 0)]
        lead[m, 2] = np.nan_to_num((g_prev - gap[t]) / 0.4, nan=0.0)         # closing speed (m/s), causal
    nav = T["nav_tok"]
    navd = Jz["nav"][T["rowT"], 3]
    F[:, 5] = nav
    F[:, 6] = navd
    F[:, 7] = Jz["limit_kmh"][eid].astype(np.float64)
    XA = F
    XB = np.hstack([F, lead])
    E_T = {a: sq(T, a)[0] for a in ("M2b_s0", "M2b_s1")}
    kinds = sorted(set(KIND.values()))
    res = {}
    gkf = GroupKFold(n_splits=5)
    for c in range(26):
        m = np.isfinite(G[:, c]) & has
        if m.sum() < 200:
            continue
        y = G[m, c]
        g = eid[m]
        pa, pb = np.zeros(m.sum()), np.zeros(m.sum())
        for tr, te in gkf.split(y, y, g):
            for X, out in ((XA[m], pa), (XB[m], pb)):
                mdl = HistGradientBoostingRegressor(max_depth=3, max_iter=80, learning_rate=0.1, min_samples_leaf=30, random_state=0)
                mdl.fit(X[tr], y[tr])
                out[te] = mdl.predict(X[te])
        res[c] = (m.sum(), ((pa - y) ** 2).sum(), ((pb - y) ** 2).sum(), (y ** 2).sum(), E_T["M2b_s0"][m, c].sum(), E_T["M2b_s1"][m, c].sum())
    print(f"{'cell':20s} {'n':>5s} | {'RMSE const(0)':>13s} {'probeA':>7s} {'probeB(+lead)':>13s} {'T s0':>6s} {'T s1':>6s}   (SD units, same windows/cells)")
    for c, (n, ea, eb, e0, et0, et1) in res.items():
        print(f"{CONS_NAMES[c]:20s} {n:5d} | {np.sqrt(e0 / n):13.3f} {np.sqrt(ea / n):7.3f} {np.sqrt(eb / n):13.3f} {np.sqrt(et0 / n):6.3f} {np.sqrt(et1 / n):6.3f}")
    print("\nBy cell kind (pooled over cells; SSE-weighted):")
    print(f"   {'kind':40s} {'const':>6s} {'probeA':>7s} {'probeB':>7s} {'T s0':>6s} {'T s1':>6s}   verdict")
    for k in kinds:
        cs = [c for c in res if KIND[c] == k]
        n = sum(res[c][0] for c in cs)
        r = [np.sqrt(sum(res[c][j] for c in cs) / n) for j in (3, 1, 2, 4, 5)]
        v = []
        if r[1] <= r[3] * 1.05:
            v.append("non-vision probe A matches T (vision adds ~nothing here)")
        elif r[2] <= r[3] * 0.9:
            v.append("privileged lead info beats T by >10% (lead information is missing from T)")
        else:
            v.append("both probes lose to T (inconclusive: probe sees 141 episodes)")
        print(f"   {k:40s} {r[0]:6.3f} {r[1]:7.3f} {r[2]:7.3f} {r[3]:6.3f} {r[4]:6.3f}   {v[0]}")
    n = sum(res[c][0] for c in res)
    print(f"   ALL cells with n>=200: const {np.sqrt(sum(res[c][3] for c in res) / n):.3f}, probeA {np.sqrt(sum(res[c][1] for c in res) / n):.3f}, "
          f"probeB {np.sqrt(sum(res[c][2] for c in res) / n):.3f}, T s0 {np.sqrt(sum(res[c][4] for c in res) / n):.3f}, T s1 {np.sqrt(sum(res[c][5] for c in res) / n):.3f}")


if __name__ == "__main__":
    main()
