"""PROBE 6 (DESCRIPTIVE): the replan STATE (v0, a0, kappa0) the T1 tool hands the model, rebuilt with the tool's own functions (statekin.py).
 TRAIN ticks (HACC loop, both seeds; exact): control S-1 = the rebuilt frame-2j (x, y, psi, v) equals the dumped tick pose; S-2 = the checkpoint's own P10
   prior evaluated on the rebuilt kin3 equals the dumped P (the plan the model's prior made from that state).
 Then: kappa0 / a0 of HACC's own state vs the GT-equivalent state (the same Hermite + one-frame rule applied to the logged knots), per tick and GT speed band.
 EVAL tick 1 (logged bucket; exact from the tick-0 plans): HACC (ol), plain B (B.ol), P10 (its tick-0 plan, recovered from HACC = P + w (O - P);
   control: equals fl[:10]); split by probe 3's slow-window set.
python probe6_state.py > probe6.log"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "C:/Users/Admin/lcdiag")
import common as C  # noqa: E402
import statekin as SK  # noqa: E402
from tanitad.data import v8f_mini_data as D_  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
from taniteval import floors_kin as fk  # noqa: E402

rng = np.random.default_rng(0)
OUT = {}
WJ = "C:/Users/Admin/t1k/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-25-hacc/run/weights_HACC_M2b_s{s}.json"


def load_prior(seed):
    ck = torch.load(f"E:/Projects/TanitAD-artifacts/v8f-mini-hacc/M2b_s{seed}_hacc.pt", map_location="cpu", weights_only=False)
    sd = None
    for key in ("model", "state", "state_dict", "model_state"):
        if isinstance(ck, dict) and key in ck and isinstance(ck[key], dict):
            sd = ck[key]
            break
    sd = sd if sd is not None else ck
    ps = {k[len("prior."):]: v for k, v in sd.items() if k.startswith("prior.")}
    hid = ps["net.0.weight"].shape[0]
    pr = M.P10Prior(ps["xm"], ps["xs"], ps["ym"], hidden=hid, k=M.K30)
    pr.load_state_dict(ps)
    del ck, sd
    return pr.eval()


def prior_plan(pr, kin):
    kin = np.asarray(kin, np.float64)[None]
    f8 = torch.as_tensor(fk.kin_features(kin).astype(np.float32))
    base = torch.as_tensor(D_.kdx_base(kin, D_.KIN_UNITS["yaw"]))
    with torch.no_grad():
        return pr(f8.float(), base.float()).double().numpy()[0]


def mean_k(A, lo, hi):
    pe = C.pair_err(A[None], A[None])
    return float(np.nanmean(pe["kX"][0][lo:hi]))


for seed in (0, 1):
    pr = load_prior(seed)
    z = np.load(f"C:/Users/Admin/hlat/dumps_train/ticks_HACC_s{seed}_m10_train.npz")
    plan, P, tau, gt, pose, tick, win, gok, v0d = z["plan"], z["P"], z["tau"], z["gt"], z["pose"], z["tick"], z["win"], z["gt_ok"], z["v0"]
    idx = {(int(win[i]), int(tick[i])): i for i in range(len(win))}
    sel = np.array([w for w in np.unique(win) if all((int(w), t) in idx for t in (0, 1, 2)) and gok[idx[(int(w), 0)]]])
    n_all = len(sel)
    if len(sel) > 2500:
        sel = rng.choice(sel, 2500, replace=False)
    rows, s1, s2 = [], [], []
    for w in sel:
        i0, i1, i2 = idx[(int(w), 0)], idx[(int(w), 1)], idx[(int(w), 2)]
        g0 = gt[i0]
        Kg = np.vstack([np.zeros((1, 2)), g0])
        rec = {"w": int(w), "v0": float(v0d[i0])}
        for n, (ia, ib) in enumerate(((i0, i1), (i1, i2)), start=1):
            Kw = np.vstack([pose[ia][None, :2], SK.T1.plan_to_world(plan[ia], pose[ia])])
            kin, f2j = SK.state_from_knots(Kw, 0)
            dpsi = float(np.angle(np.exp(1j * (f2j[2] - pose[ib][2]))))
            s1.append(max(abs(f2j[0] - pose[ib][0]), abs(f2j[1] - pose[ib][1]), abs(dpsi), abs(f2j[3] - pose[ib][3])))
            s2.append(float(np.abs(prior_plan(pr, kin) - P[ib]).max()))
            kinG, _ = SK.state_from_knots(Kg, 10 * (n - 1))
            rec[f"kin_H{n}"], rec[f"kin_G{n}"] = kin.tolist(), kinG.tolist()
            for nm, A in (("P", P[ib]), ("O", tau[ib]), ("H", plan[ib])):
                rec[f"k_{nm}{n}"] = mean_k(A[:11], 1, 9)              # mean signed chord curvature of the plan issued at tick n, pairs 1..8 (the executed part)
            rec[f"k_G{n}"] = mean_k(g0, 10 * n + 1, 10 * n + 9)
            rec[f"vG{n}"] = float(np.linalg.norm(g0[10 * n] - g0[10 * n - 1]) / 0.2)
        rows.append(rec)
    S1, S2 = float(np.max(s1)), float(np.max(s2))
    print(f"== TRAIN s{seed}: {len(rows)} of {n_all} windows | S-1 rebuilt frame-2j vs dumped pose max|d| {S1:.2e} | S-2 prior(rebuilt kin) vs dumped P max|d| {S2:.2e} m")
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    res = {"S1_max": S1, "S2_max": S2, "n": len(rows), "n_all": n_all, "cells": []}
    for n in (1, 2):
        H, G = R[f"kin_H{n}"], R[f"kin_G{n}"]
        for lo, hi in ((0, 5.2), (5.2, 8.6), (8.6, 99)):
            m = (R[f"vG{n}"] >= lo) & (R[f"vG{n}"] < hi)
            st = m & (np.abs(R[f"k_G{n}"]) < 0.01)
            cell = {"tick": n, "gt_speed_band": [lo, hi], "n": int(m.sum()),
                    "abs_dkappa0_med": float(np.median(np.abs(H[m, 2] - G[m, 2]))), "abs_dkappa0_mean": float(np.mean(np.abs(H[m, 2] - G[m, 2]))),
                    "da0_med": float(np.median(H[m, 1] - G[m, 1])), "abs_da0_med": float(np.median(np.abs(H[m, 1] - G[m, 1]))),
                    "dv0_med": float(np.median(H[m, 0] - G[m, 0])),
                    "corr_kP_kappa0H": float(np.corrcoef(R[f"k_P{n}"][m], H[m, 2])[0, 1]), "corr_kH_kappa0H": float(np.corrcoef(R[f"k_H{n}"][m], H[m, 2])[0, 1]),
                    "straight_n": int(st.sum())}
            if st.sum() > 10:
                cell.update({"straight_abs_kP_med": float(np.median(np.abs(R[f"k_P{n}"][st]))), "straight_abs_kO_med": float(np.median(np.abs(R[f"k_O{n}"][st]))),
                             "straight_abs_kH_med": float(np.median(np.abs(R[f"k_H{n}"][st]))),
                             "straight_abs_kappa0H_med": float(np.median(np.abs(H[st, 2]))), "straight_abs_kappa0G_med": float(np.median(np.abs(G[st, 2])))})
            res["cells"].append(cell)
            print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cell.items()})
    e1, e2 = R["kin_H1"][:, 2] - R["kin_G1"][:, 2], R["kin_H2"][:, 2] - R["kin_G2"][:, 2]
    m = (R["vG1"] >= 5.2) & (R["vG2"] >= 5.2)
    res["persist_corr_e1_e2_fast"] = float(np.corrcoef(e1[m], e2[m])[0, 1])
    res["abs_e_t1_t2_fast_mean"] = [float(np.abs(e1[m]).mean()), float(np.abs(e2[m]).mean())]
    print(f"   persistence corr(e1, e2) GT>=5.2: {res['persist_corr_e1_e2_fast']:+.3f}; mean |dkappa0| tick1 {res['abs_e_t1_t2_fast_mean'][0]:.4f} tick2 {res['abs_e_t1_t2_fast_mean'][1]:.4f} (n {int(m.sum())})")
    OUT[f"train_s{seed}"] = res
    np.savez_compressed(f"C:/Users/Admin/lcdiag/probe6_train_s{seed}.npz", **R)

    # ---------------- EVAL tick 1 (logged bucket)
    d, b = C.load("logged", seed), C.load("plainB_logged", seed)
    wj = json.load(open(WJ.format(s=seed)))
    w = np.stack([wj["w_along_by_step"], wj["w_cross_by_step"]], 1)
    H, O, G, F = d["ol"].astype(np.float64), b["ol"].astype(np.float64), d["g"].astype(np.float64), d["fl"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        Prec = np.where(w[None] < 1, (H - w[None] * O) / (1 - w[None]), np.nan)
    ctl = float(np.nanmax(np.abs(Prec[:, :10] - F[:, :10])))
    Prec[:, :10] = np.where(np.isnan(Prec[:, :10]), F[:, :10], Prec[:, :10])
    ctl_O = float(np.abs(d["cl"][:, :10] - d["ol"][:, :10]).max())
    print(f"== EVAL s{seed}: control P-recovery vs fl[:10] max|d| {ctl:.2e} m; HACC cl[:10] vs ol[:10] {ctl_O:.1e}")
    ref = C.load("causal", seed)
    pa, pf = C.pair_err(ref["cl"], ref["g"]), C.pair_err(ref["fl"], ref["g"])
    okr = (pa["c"] > 0) & (pf["c"] > 0)
    Wset = (pa["fast"] & (pa["vX"] < C.VMIN) & okr[:, None]).any(1)
    kin = {nm: np.array([SK.state_from_knots(np.vstack([np.zeros((1, 2)), A[i]]), 0)[0] for i in range(len(A))]) for nm, A in
           (("HACC", H), ("B", O), ("P10", Prec), ("GT", G))}
    vg = np.linalg.norm(G[:, 9] - G[:, 8], axis=1) / 0.2
    ev = {"P_recovery_ctl_max_abs_m": ctl, "cells": []}
    for grp, m in (("slow-window set", Wset), ("other windows, GT>=5.2 at 2 s", ~Wset & (vg >= 5.2))):
        cell = {"group": grp, "n": int(m.sum())}
        for nm in ("HACC", "B", "P10"):
            cell[f"{nm}_abs_dkappa0_med"] = float(np.median(np.abs(kin[nm][m, 2] - kin["GT"][m, 2])))
            cell[f"{nm}_abs_dkappa0_mean"] = float(np.mean(np.abs(kin[nm][m, 2] - kin["GT"][m, 2])))
            cell[f"{nm}_da0_med"] = float(np.median(kin[nm][m, 1] - kin["GT"][m, 1]))
            cell[f"{nm}_dv0_med"] = float(np.median(kin[nm][m, 0] - kin["GT"][m, 0]))
        ev["cells"].append(cell)
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cell.items()})
    OUT[f"eval_s{seed}"] = ev
    np.savez_compressed(f"C:/Users/Admin/lcdiag/probe6_eval_s{seed}.npz", Wset=Wset, **{f"kin_{k}": v for k, v in kin.items()})
json.dump(OUT, open("C:/Users/Admin/lcdiag/probe6.json", "w"), indent=1)
