#!/usr/bin/env python
"""log_assoc.py -- note PREREG_V8F_PROOF_ADD36_N2 (LOGASSOC). Written by the lead. A statistic of the logged eval windows; no model is read.

x = -2.0 * a (a = mean ego acceleration over 2.0 s): the kinematic change of the gap caused by the ego's own acceleration.  y = logged change of the label gap over 2.0 s.
b_raw = slope of y on x; b_cond = coefficient of x with the gap at t, the ego speed at t and the closing rate over the last 0.8 s as covariates. Episode-cluster bootstrap.
"""
import argparse
import io
import json
import os

import numpy as np

F_T, F_1, F_2, DT = 4, 9, 14, 0.2


def fit(y, X):
    A = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def slopes(y, x, C):
    b_raw = fit(y, x[:, None])[1]
    b_cond = fit(y, np.column_stack([x, C]))[1]
    return float(b_raw), float(b_cond)


def boot(y, x, C, eid, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    ep = np.unique(eid)
    rows = {e: np.flatnonzero(eid == e) for e in ep}
    out = np.zeros((B, 2))
    for i in range(B):
        idx = np.concatenate([rows[e] for e in rng.choice(ep, len(ep), replace=True)])
        out[i] = slopes(y[idx], x[idx], C[idx])
    return [[float(np.percentile(out[:, k], 2.5)), float(np.percentile(out[:, k], 97.5))] for k in (0, 1)]


def stat(G, V, eid, f_end, B=2000):
    T = (f_end - F_T) * DT
    a = (V[:, f_end] - V[:, F_T]) / T
    x = -0.5 * a * T * T
    y = G[:, f_end] - G[:, F_T]
    C = np.column_stack([G[:, F_T], V[:, F_T], (G[:, F_T] - G[:, 0]) / (F_T * DT)])
    b_raw, b_cond = slopes(y, x, C)
    ci = boot(y, x, C, eid, B)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    return {"T_s": T, "b_raw": b_raw, "b_raw_ci": ci[0], "b_cond": b_cond, "b_cond_ci": ci[1], "r2_x": r2, "corr_xy": float(np.corrcoef(x, y)[0, 1]),
            "std_x_m": float(x.std()), "std_y_m": float(y.std()), "mean_a_mps2": float(a.mean()), "std_a_mps2": float(a.std())}, x, y, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = np.load(a.pack, allow_pickle=True)
    G, V, eid = np.asarray(P["G15"], np.float64), np.asarray(P["V"], np.float64), np.asarray(P["eid"]).astype(str)
    ok = np.isfinite(G[:, 0:F_T + 1]).all(1) & np.isfinite(G[:, F_2]) & np.isfinite(V[:, [F_T, F_1, F_2]]).all(1)
    if "pres0" in P.files:
        ok &= np.asarray(P["pres0"], np.float64) > 0.5
    n_all = len(G)
    G, V, eid = G[ok], V[ok], eid[ok]
    res = {"note": "PREREG_V8F_PROOF_ADD36_N2", "pack": os.path.basename(a.pack), "n_windows_pack": int(n_all), "n_windows": int(ok.sum()), "n_episodes": int(len(np.unique(eid)))}
    s2, x, y, C = stat(G, V, eid, F_2)
    res["T2.0"] = s2
    ok1 = np.isfinite(G[:, F_1])
    res["T1.0"] = stat(G[ok1], V[ok1], eid[ok1], F_1)[0]
    rng = np.random.default_rng(1)
    # e1 known kinematic: y = x + independent noise of the spread of y
    y1 = x + rng.normal(0.0, y.std(), len(x))
    e1 = slopes(y1, x, C)
    # e2 known zero: x permuted across windows
    xp = rng.permutation(x)
    e2_ci = boot(y, xp, C, eid)[1]
    # e4 sign convention: a planted lead at constant speed
    vL = V[:, F_T] + rng.normal(0.0, 2.0, len(x))
    y4 = (vL - V[:, F_T]) * 2.0 + x
    C4 = np.column_stack([C, vL - V[:, F_T]])
    e4 = float(fit(y4, np.column_stack([x, C4]))[1])
    res["controls"] = {
        "e1_known_kinematic": {"b_raw": e1[0], "b_cond": e1[1], "PASS": bool(0.9 <= e1[0] <= 1.1 and 0.9 <= e1[1] <= 1.1)},
        "e2_permuted": {"b_cond_ci": e2_ci, "PASS": bool(e2_ci[0] <= 0.0 <= e2_ci[1])},
        "e3_counts": {"n_windows": res["n_windows"], "n_episodes": res["n_episodes"], "PASS": bool(res["n_windows"] >= 200 and res["n_episodes"] >= 20)},
        "e4_planted_constant_lead": {"b_cond": e4, "PASS": bool(abs(e4 - 1.0) < 1e-6)},
    }
    allp = all(v["PASS"] for v in res["controls"].values())
    res["controls_pass"] = bool(allp)
    b, ci = s2["b_cond"], s2["b_cond_ci"]
    if not allp:
        name = "NOT NAMED (a control failed)"
    elif b < 0 and ci[1] < 0:
        name = "LOG-ASSOCIATION-REVERSED"
    elif b >= 0.5 and ci[0] > 0:
        name = "LOG-ASSOCIATION-KINEMATIC"
    else:
        name = "LOG-ASSOCIATION-WEAK"
    res["name"] = name
    res["beside_model_slopes"] = {"LS": -0.452, "RL": -0.888, "LA": 0.855}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    r3 = lambda d: {k: ([round(u, 3) for u in v] if isinstance(v, list) else (round(v, 3) if isinstance(v, float) else v)) for k, v in d.items()}
    print("n", res["n_windows"], "of", n_all, "episodes", res["n_episodes"])
    print("T2.0", json.dumps(r3(s2)))
    print("T1.0", json.dumps(r3(res["T1.0"])))
    print("controls", json.dumps({k: (r3(v)) for k, v in res["controls"].items()}))
    print("NAME", name)


if __name__ == "__main__":
    main()
