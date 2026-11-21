"""PROBE 7 (DESCRIPTIVE): does a plan inherit the curvature its predecessor left at the replan point?
 EVAL (executed tracks): signed chord curvature at the last executed pair of plan n (pair 8 / 18: plan-n knots 8, 9, 10, the proxy of the next state's
   kappa0) against the mean signed curvature the next plan executes (pairs 10-18 / 20-28), for HACC, plain B, P10 and the GT; slow-window set vs other
   GT-fast windows. A loop that re-issues the state's curvature as a constant arc gives slope ~1 and |kappa| that grows plan to plan.
 TRAIN ticks (exact states, probe 6 npz): which component of the plan issued at tick n injects the kappa0 of tick n+1 -- kappa0 rebuilt from the HACC plan's
   own knots (what happened) vs from the P10 plan P's knots and O's knots tau, same pose; GT-straight windows (|kG| < 0.01), by GT speed band.
python probe7_persistence.py > probe7.log"""
import json
import sys

import numpy as np

sys.path.insert(0, "C:/Users/Admin/lcdiag")
import common as C  # noqa: E402
import statekin as SK  # noqa: E402

out = {}
for s in (0, 1):
    ref = C.load("causal", s)
    b = C.load("plainB_logged", s)
    lg = C.load("logged", s)
    G = ref["g"]
    pa, pf = C.pair_err(ref["cl"], G), C.pair_err(ref["fl"], G)
    okr = (pa["c"] > 0) & (pf["c"] > 0)
    Wset = (pa["fast"] & (pa["vX"] < C.VMIN) & okr[:, None]).any(1)
    vg = np.linalg.norm(G[:, 19] - G[:, 18], axis=1) / 0.2
    r = {}
    for nm, X in (("HACC", ref["cl"]), ("B_logged", b["cl"]), ("HACC_logged", lg["cl"]), ("P10", ref["fl"]), ("GT", G)):
        k = C.pair_err(X, X)["kX"]
        for grp, m in (("slow-set", Wset), ("other GT>=5.2", ~Wset & (vg >= 5.2))):
            e8, e18 = k[m, 8], k[m, 18]
            p1, p2 = np.nanmean(k[m, 10:19], 1), np.nanmean(k[m, 20:29], 1)
            okk = np.isfinite(e8) & np.isfinite(p1) & np.isfinite(e18) & np.isfinite(p2)
            sl1 = float(np.polyfit(e8[okk], p1[okk], 1)[0]) if okk.sum() > 10 else None
            sl2 = float(np.polyfit(e18[okk], p2[okk], 1)[0]) if okk.sum() > 10 else None
            r[f"{nm} | {grp}"] = {"n": int(okk.sum()), "med_abs_k_plan0_1_2": [float(np.nanmedian(np.abs(np.nanmean(k[m, a:a + 9], 1)))) for a in (0, 10, 20)],
                                 "slope_next_plan_on_end_pair": [sl1, sl2],
                                 "corr_next_plan_on_end_pair": [float(np.corrcoef(e8[okk], p1[okk])[0, 1]), float(np.corrcoef(e18[okk], p2[okk])[0, 1])]}
            print(f"s{s} {nm:12s} {grp:14s}", {k_: (np.round(v, 4).tolist() if isinstance(v, list) and v and v[0] is not None else v) for k_, v in r[f'{nm} | {grp}'].items()})
    out[f"eval_s{s}"] = r

    # TRAIN: injection
    z = np.load(f"C:/Users/Admin/hlat/dumps_train/ticks_HACC_s{s}_m10_train.npz")
    R = dict(np.load(f"C:/Users/Admin/lcdiag/probe6_train_s{s}.npz"))
    plan, P, tau, pose, tick, win = z["plan"], z["P"], z["tau"], z["pose"], z["tick"], z["win"]
    idx = {(int(win[i]), int(tick[i])): i for i in range(len(win))}
    rows = []
    for j, w in enumerate(R["w"]):
        for n in (1, 2):
            ia = idx[(int(w), n - 1)]
            kk = {}
            for nm, A in (("H", plan[ia]), ("P", P[ia]), ("O", tau[ia])):
                Kw = np.vstack([pose[ia][None, :2], SK.T1.plan_to_world(A, pose[ia])])
                kk[nm] = SK.state_from_knots(Kw, 0)[0][2]
            rows.append((n, R[f"vG{n}"][j], R[f"k_G{n}"][j], R[f"kin_G{n}"][j][2], R[f"kin_H{n - 1}"][j][2] if n == 2 else R["kin_G1"][j][2] * np.nan,
                         kk["H"], kk["P"], kk["O"]))
    A = np.array(rows, dtype=np.float64)
    tr = {}
    for n in (1, 2):
        for lo, hi in ((5.2, 8.6), (8.6, 99)):
            m = (A[:, 0] == n) & (A[:, 1] >= lo) & (A[:, 1] < hi) & (np.abs(A[:, 2]) < 0.01)
            dG = A[m, 3]
            cell = {"n": int(m.sum()), "med_abs_kappa0_err_from_HACC_plan": float(np.median(np.abs(A[m, 5] - dG))),
                    "from_P_plan": float(np.median(np.abs(A[m, 6] - dG))), "from_O_plan": float(np.median(np.abs(A[m, 7] - dG)))}
            tr[f"tick{n} GT {lo}-{hi} straight"] = cell
            print(f"s{s} TRAIN state at tick {n} (from the plan issued at tick {n - 1}), GT {lo}-{hi} straight:", {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in cell.items()})
    out[f"train_s{s}"] = tr
json.dump(out, open("C:/Users/Admin/lcdiag/probe7.json", "w"), indent=1)
