"""PROBE 11 (TRAIN ONLY; feasibility for PREREG_DRAFT's lever, no eval array is opened): can an INCREMENT-domain blend
    H_inc(k) = P(k) + sum_{j<=k} w_j dD_j,   dD_j = D(j) - D(j-1), D = tau - P, D(0) = 0
match the POSITION-domain HACC family in open loop on the TRAIN windows, and does it remove the replan-state deceleration?
Fit set: the TRAIN tick-0 samples (logged state; HACC-LAT's C-T2 showed they equal the open-loop tau / P / v0 exactly), gt_ok only.
Both families at lambda = 0 (positions only), same BVLS solver and bounds [0, 1] as v8f_hacc.py I4. In-sample train ADE2 / ADE6 for: P10, O (w = 1),
HACC banked w (position domain), POS refit at lambda 0, INC refit at lambda 0. Then, on TRAIN ticks 0 -> 1 and 1 -> 2, the replan-state a0 of the
INC-refit plan built from the dumped P / tau, against the dumped HACC plan's (statekin; same pose).
python probe11_inc_trainfit.py > probe11.log"""
import json
import sys

import numpy as np
from scipy.optimize import lsq_linear

sys.path.insert(0, "C:/Users/Admin/lcdiag")
import statekin as SK  # noqa: E402

WJ = "C:/Users/Admin/t1k/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-25-hacc/run/weights_HACC_M2b_s{s}.json"


def fit(A_fn, P, tau, y):
    w = np.zeros((30, 2))
    for ax in (0, 1):
        A, b = A_fn(P[..., ax], tau[..., ax], y[..., ax])
        r = lsq_linear(A, b, bounds=(0.0, 1.0), method="bvls", tol=1e-12, max_iter=500)
        w[:, ax] = r.x
    return w


def A_pos(P, T, Y):
    D = T - P                                    # [n, 30]
    n = len(D)
    A = np.zeros((n * 30, 30))
    A[np.arange(n * 30), np.tile(np.arange(30), n)] = D.reshape(-1)
    return A, (Y - P).reshape(-1)


def A_inc(P, T, Y):
    D = T - P
    dD = np.diff(np.concatenate([np.zeros((len(D), 1)), D], 1), axis=1)
    tri = np.tril(np.ones((30, 30)))             # row k, col j: j <= k
    A = (tri[None] * dD[:, None, :]).reshape(-1, 30)
    return A, (Y - P).reshape(-1)


def apply_pos(P, T, w):
    return P + w[None] * (T - P)


def apply_inc(P, T, w):
    D = T - P
    dD = np.diff(np.concatenate([np.zeros((len(D), 1, 2)), D], 1), axis=1)
    return P + np.cumsum(w[None] * dD, 1)


def ade(X, Y):
    e = np.linalg.norm(X - Y, axis=-1)
    return float(e[:, :10].mean()), float(e.mean())


out = {}
for s in (0, 1):
    z = np.load(f"C:/Users/Admin/hlat/dumps_train/ticks_HACC_s{s}_m10_train.npz")
    m0 = (z["tick"] == 0) & z["gt_ok"]
    P, T, Y = z["P"][m0], z["tau"][m0], z["gt"][m0]
    wj = json.load(open(WJ.format(s=s)))
    wh = np.stack([wj["w_along_by_step"], wj["w_cross_by_step"]], 1)
    ctl = float(np.abs(apply_pos(P, T, wh) - z["plan"][m0]).max())          # control: the banked HACC w reproduces the dumped tick-0 plan
    wpos, winc = fit(A_pos, P, T, Y), fit(A_inc, P, T, Y)
    ctl1 = float(np.abs(apply_inc(P, T, np.ones((30, 2))) - T).max())       # INC with w = 1 is O's plan; w = 0 is P10's
    ctl0 = float(np.abs(apply_inc(P, T, np.zeros((30, 2))) - P).max())
    rows = {"n_windows": int(m0.sum()), "ctl_banked_w_reproduces_plan_max_abs_m": ctl, "ctl_inc_w1_is_O": ctl1, "ctl_inc_w0_is_P": ctl0,
            "ade2_ade6": {"P10": ade(P, Y), "O": ade(T, Y), "HACC_banked": ade(apply_pos(P, T, wh), Y), "POS_refit_l0": ade(apply_pos(P, T, wpos), Y),
                          "INC_refit_l0": ade(apply_inc(P, T, winc), Y)},
            "w_inc_l0": winc.round(4).tolist(), "w_pos_l0": wpos.round(4).tolist()}
    print(f"== s{s}: n {rows['n_windows']} | controls: banked w -> dumped plan {ctl:.1e}, INC(w=1)=O {ctl1:.1e}, INC(w=0)=P {ctl0:.1e}")
    for k, v in rows["ade2_ade6"].items():
        print(f"   {k:14s} ADE2 {v[0]:.4f}  ADE6 {v[1]:.4f}")
    print("   w_inc along", np.round(winc[:, 0], 3)); print("   w_inc cross", np.round(winc[:, 1], 3))
    # replan-state a0 / v0 of the INC-refit plan vs the HACC plan, train ticks 0, 1 (own state at tick 1)
    st = []
    for i in np.nonzero(z["tick"] <= 1)[0][::4]:
        Pi, Ti = z["P"][i][None], z["tau"][i][None]
        hi = SK.state_from_knots(np.vstack([np.zeros((1, 2)), z["plan"][i]]), 0)[0]
        ii = SK.state_from_knots(np.vstack([np.zeros((1, 2)), apply_inc(Pi, Ti, winc)[0]]), 0)[0]
        pi = SK.state_from_knots(np.vstack([np.zeros((1, 2)), z["P"][i]]), 0)[0]
        st.append((z["tick"][i], z["v0"][i], hi[1] - pi[1], ii[1] - pi[1], hi[0] - pi[0], ii[0] - pi[0]))
    S = np.array(st)
    rows["state_vs_P10_state"] = {}
    for t in (0, 1):
        for lo, hi_ in ((0, 5.2), (5.2, 99)):
            m = (S[:, 0] == t) & (S[:, 1] >= lo) & (S[:, 1] < hi_)
            c = {"n": int(m.sum()), "HACC_a0_minus_P10_med": float(np.median(S[m, 2])), "INC_a0_minus_P10_med": float(np.median(S[m, 3])),
                 "HACC_v0_minus_P10_med": float(np.median(S[m, 4])), "INC_v0_minus_P10_med": float(np.median(S[m, 5]))}
            rows["state_vs_P10_state"][f"plan@tick{t} v0 {lo}-{hi_}"] = c
            print(f"   plan@tick{t} v0 {lo}-{hi_}:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()})
    out[f"s{s}"] = rows
json.dump(out, open("C:/Users/Admin/lcdiag/probe11.json", "w"), indent=1)
