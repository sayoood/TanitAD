"""PROBE 10 (DESCRIPTIVE): does the post-map's STEP-VARYING weight inject motion into the replan state?
HACC's executed knots are H(k) = P(k) + w(k) D(k), D = O - P. Their velocity is v_P + w v_D + (dw/dt) D: the last term is in NEITHER plan. The replan
state reads knots 8..11 (statekin), so it carries dw/dt * D at knots 8..11 (w_along 0.14 -> 0.32 there at s0).
 For each plan issued at the logged state (EVAL tick 0: P recovered from HACC = P + w(O - P), O = plain-B's tick-0 plan, both exact) and at own states
 (TRAIN ticks 0, 1: P, tau, plan dumped): the tick-(n+1) state (v0, a0, kappa0) rebuilt from
   H     = HACC's plan (what ran),
   LIN   = P_state + w10 (O_state - P_state)   (the blend of the two plans' own states, w at knot 10: what a ramp-free blend would hand over),
   INC   = the INCREMENT-domain blend  H_inc(k) = sum_{j<=k} [dP(j) + w(j) dD(j)]  (same w, applied to the 0.2 s steps instead of the positions),
 and the ramp term = state(H) - state(INC). Split by the probe-3 slow-window set (eval) and by GT speed band (train).
python probe10_ramp.py > probe10.log"""
import json
import sys

import numpy as np

sys.path.insert(0, "C:/Users/Admin/lcdiag")
import common as C  # noqa: E402
import statekin as SK  # noqa: E402

WJ = "C:/Users/Admin/t1k/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-25-hacc/run/weights_HACC_M2b_s{s}.json"


def st(A):
    return SK.state_from_knots(np.vstack([np.zeros((1, 2)), A]), 0)[0]


def inc_blend(P, O, w):
    dP = np.diff(np.vstack([np.zeros((1, 2)), P]), axis=0)
    dD = np.diff(np.vstack([np.zeros((1, 2)), O - P]), axis=0)
    return np.cumsum(dP + w * dD, axis=0)


out = {}
for s in (0, 1):
    wj = json.load(open(WJ.format(s=s)))
    w = np.stack([wj["w_along_by_step"], wj["w_cross_by_step"]], 1)
    # ---------------- EVAL, plan issued at the logged state
    d, b, ref = C.load("logged", s), C.load("plainB_logged", s), C.load("causal", s)
    H, O, F = d["ol"].astype(np.float64), b["ol"].astype(np.float64), d["fl"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        Pr = np.where(w[None] < 1, (H - w[None] * O) / (1 - w[None]), np.nan)
    Pr[:, :10] = np.where(np.isnan(Pr[:, :10]), F[:, :10], Pr[:, :10])
    ok_rec = np.isfinite(Pr[:, :12]).all((1, 2))
    Hc = Pr + w[None] * (O - Pr)
    ctl = float(np.nanmax(np.abs(Hc[:, :12] - H[:, :12])))               # control: the blend of the recovered P and O reproduces HACC's knots 1..12
    pa, pf = C.pair_err(ref["cl"], ref["g"]), C.pair_err(ref["fl"], ref["g"])
    okr = (pa["c"] > 0) & (pf["c"] > 0)
    Wset = (pa["fast"] & (pa["vX"] < C.VMIN) & okr[:, None]).any(1)
    SH = np.array([st(H[i]) for i in range(len(H))])
    SP = np.array([st(Pr[i]) for i in range(len(H))])
    SO = np.array([st(O[i]) for i in range(len(H))])
    SI = np.array([st(inc_blend(Pr[i], O[i], w)) for i in range(len(H))])
    SG = np.array([st(d["g"][i].astype(np.float64)) for i in range(len(H))])
    lin = SP + w[9, 0] * (SO - SP)                                       # along weight at knot 10 for v0, a0
    lin[:, 2] = SP[:, 2] + w[9, 1] * (SO[:, 2] - SP[:, 2])
    ev = {"ctl_blend_reproduces_HACC_knots_1_12_max_abs_m": ctl, "n_recovered": int(ok_rec.sum())}
    print(f"== EVAL s{s} (plan issued at the logged state; tick-1 state). control blend(P_rec, O) vs HACC ol knots 1-12: {ctl:.2e} m")
    for grp, m in (("slow-set", Wset & ok_rec), ("other", ~Wset & ok_rec)):
        cell = {"n": int(m.sum())}
        for nm, S_ in (("HACC", SH), ("LIN", lin), ("INC", SI), ("P10", SP), ("O", SO)):
            cell[f"{nm}_v0_minus_GT_med"] = float(np.median(S_[m, 0] - SG[m, 0]))
            cell[f"{nm}_a0_minus_GT_med"] = float(np.median(S_[m, 1] - SG[m, 1]))
        cell["ramp_a0_HACC_minus_INC_med"] = float(np.median(SH[m, 1] - SI[m, 1]))
        cell["ramp_a0_HACC_minus_INC_mean"] = float(np.mean(SH[m, 1] - SI[m, 1]))
        cell["ramp_v0_HACC_minus_INC_med"] = float(np.median(SH[m, 0] - SI[m, 0]))
        cell["HACC_minus_LIN_a0_med"] = float(np.median(SH[m, 1] - lin[m, 1]))
        cell["abs_dkappa0_vs_GT_med_HACC_INC"] = [float(np.median(np.abs(SH[m, 2] - SG[m, 2]))), float(np.median(np.abs(SI[m, 2] - SG[m, 2])))]
        ev[grp] = cell
        print(f"  {grp:9s}", {k: (round(v, 3) if isinstance(v, float) else (np.round(v, 4).tolist() if isinstance(v, list) else v)) for k, v in cell.items()})
    out[f"eval_s{s}"] = ev
    # ---------------- TRAIN ticks: plans issued at tick 0 (logged) and tick 1 (own state)
    z = np.load(f"C:/Users/Admin/hlat/dumps_train/ticks_HACC_s{s}_m10_train.npz")
    R = dict(np.load(f"C:/Users/Admin/lcdiag/probe6_train_s{s}.npz"))
    idx = {(int(a), int(t)): i for i, (a, t) in enumerate(zip(z["win"], z["tick"]))}
    rows = []
    for j, wi in enumerate(R["w"]):
        for n in (0, 1):
            i = idx[(int(wi), n)]
            Hn, Pn, On = z["plan"][i], z["P"][i], z["tau"][i]
            sh, si = st(Hn), st(inc_blend(Pn, On, w))
            rows.append((n, R[f"vG{n + 1}"][j], sh[0], sh[1], si[0], si[1], float(np.abs(Hn - (Pn + w * (On - Pn))).max())))
    A = np.array(rows)
    tr = {"ctl_plan_equals_P_plus_w_tau_minus_P_max_abs_m": float(A[:, 6].max())}
    print(f"== TRAIN s{s}: control plan == P + w (tau - P): max|d| {tr['ctl_plan_equals_P_plus_w_tau_minus_P_max_abs_m']:.2e} m")
    for n in (0, 1):
        for lo, hi in ((0, 5.2), (5.2, 8.6), (8.6, 99)):
            m = (A[:, 0] == n) & (A[:, 1] >= lo) & (A[:, 1] < hi)
            cell = {"n": int(m.sum()), "ramp_a0_med": float(np.median(A[m, 3] - A[m, 5])), "ramp_a0_mean": float(np.mean(A[m, 3] - A[m, 5])),
                    "ramp_a0_absmed": float(np.median(np.abs(A[m, 3] - A[m, 5]))), "ramp_v0_med": float(np.median(A[m, 2] - A[m, 4]))}
            tr[f"plan@tick{n} GT {lo}-{hi}"] = cell
            print(f"  plan issued at tick {n} -> state at tick {n + 1}, GT {lo}-{hi}:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in cell.items()})
    out[f"train_s{s}"] = tr
json.dump(out, open("C:/Users/Admin/lcdiag/probe10.json", "w"), indent=1)
