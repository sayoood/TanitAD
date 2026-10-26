"""PROBE 12 (TRAIN ONLY; the PREREG_DRAFT lever's fit and its train-side mechanism check; no eval array is opened).
HACC-MC ("motion-consistent"): w_along[k] = c for the knots k = 1..12 (every knot a replan period <= 2.2 s executes or reads into the state), HACC's banked
w_along beyond, w_cross = HACC's banked. c per seed = argmin over [0, 1] of HACC's OWN objective (v8f_hacc.py: positions + lambda* x D2 with the origin
prepended, lambda* = the banked per-seed value) on the TRAIN tick-0 samples (= HBLEND's 9,582 open-loop TRAIN windows), along axis; a 1-D convex quadratic.
Deliberate-regression arm STEEP: w_along[k] for k = 8..12 replaced by a ramp 0 -> 0.6 (5x HACC's slope there), everything else HACC's.
Reports, in-sample on TRAIN: open-loop ADE2 / ADE6 of P10, HACC, MC, STEEP; and on TRAIN ticks 0, 1 the replan-state ramp term
(ALONG ramp term: state(H) - state(H with its along axis re-blended in the increment domain, same w); identically 0 for MC) and a0 / v0 relative to P10's state.
python probe12_mc_trainfit.py > probe12.log"""
import json
import sys

import numpy as np

sys.path.insert(0, "C:/Users/Admin/lcdiag")
import statekin as SK  # noqa: E402

WJ = "C:/Users/Admin/t1k/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-25-hacc/run/weights_HACC_M2b_s{s}.json"
K_FLAT = 12


def d2(e):                                       # [n, 30] -> [n, 29]: D2 with the origin prepended (v8f_hacc I3)
    p = np.concatenate([np.zeros((len(e), 1)), e], 1)
    return p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]


def apply(P, T, w):
    return P + w[None] * (T - P)


def inc(P, T, w):
    D = T - P
    dD = np.diff(np.concatenate([np.zeros((len(D), 1, 2)), D], 1), axis=1)
    return P + np.cumsum(w[None] * dD, 1)


def ade(X, Y):
    e = np.linalg.norm(X - Y, axis=-1)
    return round(float(e[:, :10].mean()), 4), round(float(e.mean()), 4)


def st(A):
    return SK.state_from_knots(np.vstack([np.zeros((1, 2)), A]), 0)[0]


out = {}
for s in (0, 1):
    wj = json.load(open(WJ.format(s=s)))
    wh = np.stack([wj["w_along_by_step"], wj["w_cross_by_step"]], 1)
    lam = float(wj["lam_star"])
    z = np.load(f"C:/Users/Admin/hlat/dumps_train/ticks_HACC_s{s}_m10_train.npz")
    m0 = (z["tick"] == 0) & z["gt_ok"]
    P, T, Y = z["P"][m0][..., 0], z["tau"][m0][..., 0], z["gt"][m0][..., 0]      # along axis
    D = T - P
    flat = np.arange(30) < K_FLAT
    u = np.where(flat[None], D, 0.0)
    e0 = (P - Y) + np.where(flat[None], 0.0, wh[None, :, 0] * D)
    num = (e0 * u).sum() + lam * (d2(e0) * d2(u)).sum()
    den = (u * u).sum() + lam * (d2(u) ** 2).sum()
    c = float(np.clip(-num / den, 0.0, 1.0))
    # objective check: c is the minimiser on a grid (control of the closed form)
    def J(cc):
        e = e0 + cc * u
        return float((e ** 2).sum() + lam * (d2(e) ** 2).sum())
    grid = np.linspace(0, 1, 201)
    cg = float(grid[np.argmin([J(g) for g in grid])])
    wmc = wh.copy(); wmc[:K_FLAT, 0] = c
    wst = wh.copy(); wst[7:12, 0] = np.linspace(0.0, 0.6, 5)
    PP, TT, YY = z["P"][m0], z["tau"][m0], z["gt"][m0]
    res = {"lam_star": lam, "c_mc": c, "c_grid_check": cg, "w_along_mc": wmc[:, 0].round(4).tolist(), "w_along_steep": wst[:, 0].round(4).tolist(),
           "train_openloop_ade2_ade6": {"P10": ade(PP, YY), "HACC": ade(apply(PP, TT, wh), YY), "MC": ade(apply(PP, TT, wmc), YY), "STEEP": ade(apply(PP, TT, wst), YY)}}
    print(f"== s{s}: lambda* {lam:.1f}  c_MC {c:.4f} (grid check {cg:.3f}) | train open-loop ADE2/ADE6:", res["train_openloop_ade2_ade6"])
    rows = []
    for i in np.nonzero(z["tick"] <= 1)[0][::4]:
        Pi, Ti = z["P"][i][None], z["tau"][i][None]
        sp = st(z["P"][i])
        r = [z["tick"][i], z["v0"][i]]
        for w in (wh, wmc, wst):
            hp = apply(Pi, Ti, w)[0]
            hi_ = hp.copy(); hi_[:, 0] = inc(Pi, Ti, w)[0][:, 0]          # ALONG axis only swapped to the increment domain (the cross axis kept)
            sh, si = st(hp), st(hi_)
            r += [sh[1] - si[1], sh[1] - sp[1], sh[0] - sp[0]]
        rows.append(r)
    A = np.array(rows)
    res["state"] = {}
    for t in (0, 1):
        for lo, hi in ((0, 5.2), (5.2, 99)):
            m = (A[:, 0] == t) & (A[:, 1] >= lo) & (A[:, 1] < hi)
            cell = {"n": int(m.sum())}
            for j, nm in enumerate(("HACC", "MC", "STEEP")):
                cell[f"{nm}_ramp_a0_med"] = round(float(np.median(A[m, 2 + 3 * j])), 4)
                cell[f"{nm}_ramp_a0_maxabs"] = round(float(np.abs(A[m, 2 + 3 * j]).max()), 6)
                cell[f"{nm}_a0_minus_P10_med"] = round(float(np.median(A[m, 3 + 3 * j])), 4)
                cell[f"{nm}_v0_minus_P10_med"] = round(float(np.median(A[m, 4 + 3 * j])), 4)
            res["state"][f"plan@tick{t} v0 {lo}-{hi}"] = cell
            print(f"   plan@tick{t} v0 {lo}-{hi}:", cell)
    out[f"s{s}"] = res
json.dump(out, open("C:/Users/Admin/lcdiag/probe12.json", "w"), indent=1)
