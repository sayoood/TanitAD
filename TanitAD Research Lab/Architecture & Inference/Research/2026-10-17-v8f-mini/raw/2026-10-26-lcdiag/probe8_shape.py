"""PROBE 8 (DESCRIPTIVE): the SHAPE of HACC's executed curvature in the slow-window set (plan 2, pairs 20-28): oscillation (sign flips between consecutive
pairs) or a sustained bend (one sign), and the lateral offset it produces; the 8 windows carrying the most excess, printed.
python probe8_shape.py > probe8.log"""
import sys

import numpy as np

sys.path.insert(0, "C:/Users/Admin/lcdiag")
import common as C  # noqa: E402

np.set_printoptions(linewidth=220, precision=3, suppress=True)
for s in (0, 1):
    d = C.load("causal", s)
    G, X, F = d["g"].astype(np.float64), d["cl"].astype(np.float64), d["fl"].astype(np.float64)
    pa, pf, pg = C.pair_err(X, G), C.pair_err(F, G), C.pair_err(G, G)
    ok = (pa["c"] > 0) & (pf["c"] > 0)
    W = (pa["fast"] & (pa["vX"] < C.VMIN) & ok[:, None]).any(1)
    exc = np.where(ok, (pa["sh"] - pf["sh"]).sum(1), 0.0)
    for nm, P_ in (("HACC", pa), ("P10", pf), ("GT", pg)):
        k = P_["kX"][W][:, 20:29]
        sg = np.sign(k)
        flips = (sg[:, 1:] * sg[:, :-1] < 0).mean()
        net = np.abs(np.nansum(k * P_["vX"][W][:, 20:29] * C.DT, 1))           # |net heading change| over plan 2 (rad)
        tot = np.nansum(np.abs(k) * P_["vX"][W][:, 20:29] * C.DT, 1)            # total absolute heading change (rad)
        print(f"s{s} {nm:4s} slow-set plan 2: sign-flip rate {flips:.3f} | median net |dpsi| {np.median(net):.4f} rad, total |dpsi| {np.median(tot):.4f} rad, ratio net/total {np.median(net / np.maximum(tot, 1e-9)):.3f}")
    # lateral offset at 6 s in the GT tangent frame
    tg = G[:, 29] - G[:, 28]
    tg = tg / np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
    nrm = np.stack([-tg[:, 1], tg[:, 0]], 1)
    for nm, T in (("HACC", X), ("P10", F)):
        lat = ((T[:, 29] - G[:, 29]) * nrm).sum(1)
        print(f"s{s} {nm:4s} |cross offset at 6 s| median: slow-set {np.median(np.abs(lat[W])):.3f} m, others {np.median(np.abs(lat[~W & ok])):.3f} m")
    top = np.argsort(-exc)[:8]
    print(f"s{s} share of the L2 excess (sum over windows / N) in the top 8 windows: {exc[top].sum() / ok.sum():+.4f} of {exc.sum() / ok.sum():+.4f}; top-30: {exc[np.argsort(-exc)[:30]].sum() / ok.sum():+.4f}; episodes in top-30: {len(np.unique(d['eid'][np.argsort(-exc)[:30]]))}")
    for i in top[:4]:
        print(f"  window {i} eid {d['eid'][i]} v0 {d['v0'][i]:.2f} excess*N {exc[i]:+.3f}")
        print("    speed GT  ", pg["vX"][i][::2]); print("    speed HACC", pa["vX"][i][::2]); print("    speed P10 ", pf["vX"][i][::2])
        print("    kappa GT  ", pg["kX"][i][::2]); print("    kappa HACC", pa["kX"][i][::2]); print("    kappa P10 ", pf["kX"][i][::2])
