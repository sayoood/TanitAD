"""PROBE 4 (DESCRIPTIVE): the speed and curvature profiles (mean over windows, per 0.2 s step / pair) of the windows that hold a GT-fast & HACC-slow pair
(probe 3's set), for HACC (cl), P10 (fl), plain B (logged run) and the GT; plus the signed curvature on the slow pairs. python probe4_speedprofile.py > probe4.log"""
import sys
import numpy as np
sys.path.insert(0, "C:/Users/Admin/lcdiag")
import common as C  # noqa: E402
np.set_printoptions(linewidth=250, precision=2, suppress=True)
for s in (0, 1):
    d = C.load("causal", s); b = C.load("plainB_logged", s); lg = C.load("logged", s)
    G, X, F = d["g"], d["cl"], d["fl"]
    pa, pf, pg = C.pair_err(X, G), C.pair_err(F, G), C.pair_err(G, G)
    ok = (pa["c"] > 0) & (pf["c"] > 0)
    slow = pa["fast"] & (pa["vX"] < C.VMIN) & ok[:, None]
    W = slow.any(1)
    def spd(T):
        p = np.concatenate([np.zeros((len(T), 1, 2)), T.astype(np.float64)], 1)
        return np.linalg.norm(np.diff(p, axis=1), axis=-1) / C.DT
    print(f"== s{s}: {W.sum()} windows (causal HACC), speed per 0.2 s step (m/s), mean over those windows")
    for nm, T in (("GT", G), ("HACC", X), ("P10", F), ("B(logged)", b["cl"]), ("HACC(logged)", lg["cl"])):
        print(f"  {nm:12s}", spd(T[W]).mean(0))
    print("  HACC-GT along-track offset at knots 9/19/29 (m, GT tangent frame), median:")
    for nm, T in (("HACC", X), ("P10", F)):
        offs = []
        for k in (9, 19, 29):
            tg = G[W, k] - G[W, k - 1]; tg = tg / np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
            offs.append(float(np.median(((T[W, k] - G[W, k]) * tg).sum(1))))
        print(f"    {nm}: {np.round(offs, 2)}")
    print("  |kappa| (1/m) per pair, mean over those windows: GT / HACC / P10")
    for nm, P_ in (("GT", pg), ("HACC", pa), ("P10", pf)):
        k = np.where(pa["fast"][W] | True, np.abs(P_["kX"][W]), np.nan)
        print(f"  {nm:5s}", np.nanmedian(k, 0))
    print("  on the slow pairs: median |kG| %.3f |kH| %.3f |kF| %.3f ; median |kH-kG| %.3f |kF-kG| %.3f" % (
        np.median(np.abs(pa["kG"][slow])), np.median(np.abs(pa["kX"][slow])), np.median(np.abs(pf["kX"][slow])),
        np.median(pa["e"][slow]), np.median(pf["e"][slow])))
    # do the slow windows also hold the excess of the joint-fast pairs? split windows
    exc = (pa["sh"] - pf["sh"]).sum(1)
    print(f"  L2 excess mean: slow windows {exc[W & ok].sum() / ok.sum():+.4f} (sum share), others {exc[~W & ok].sum() / ok.sum():+.4f}")
    # HACC speed below GT: how much along-track speed deficit, and the speed MAE of HACC vs P10 in these windows
    sH, sF, sG = spd(X[W]), spd(F[W]), spd(G[W])
    print(f"  speed MAE in those windows: HACC {np.abs(sH - sG).mean():.3f}  P10 {np.abs(sF - sG).mean():.3f} ; mean signed HACC-GT {np.mean(sH - sG):+.3f} P10-GT {np.mean(sF - sG):+.3f}")
