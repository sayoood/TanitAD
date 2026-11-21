"""l2_fit.py -- refcv9 L2 (PREREG_L2.md section 4): the TRAIN-fitted calibration of refcv8's planner-feeding longitudinal tactical head.

    calibrated logits  lc = (l + b) / T          over the 8 tacv6 lon logits; b = [0, 0, (masked), bD, bD, bD, (masked), bA] on LON6 columns
                                                 (FOLLOW, CRUISE, BRAKE_TO, CREEP, HOLD, ACCELERATE), 0 on the two masked columns (v7 ids 2, 6)
    group probability  q_g = sum of softmax6(lc) over the group:  decel = {BRAKE_TO, CREEP, HOLD}; cruise = {FOLLOW, CRUISE}; accel = {ACCELERATE}
    parameters         theta = (bD, bA, log T)  -- ONE bias per non-reference group (a within-group preference cannot be identified from a 3-class
                       target); the cruise group {FOLLOW, CRUISE} is the reference (bias 0: a FOLLOW bias is not identified either -- a first dev fit
                       on EMA-30k drove it to its bound -8, i.e. deleted the FOLLOW hypothesis, which is not a calibration); T is a global temperature.
    objective          group NLL against the TRAJECTORY-derived class of the human's 6 s future (the class the bars are defined on), on TRAIN rows
                       only (S6-valid, ceiling-compliant), plus ridge lam * (bD^2 + bA^2 + logT^2); lam by 5-fold EPISODE cross-validation
                       on TRAIN (fold seed 0; replicate seed 1).  Nothing here ever reads an eval row.

    python l2_fit.py selftest
    python l2_fit.py fit --dump-dir <A6 train139 roll dump> --out <json> [--tag ..]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_common as C  # noqa: E402

GROUP_OF6 = np.array([1, 1, 0, 0, 0, 2])          # LON6 index -> group (0 decel, 1 cruise, 2 accel)
YID = {"decel": 0, "cruise": 1, "accel": 2}
LAMBDAS = (0.0, 1e-3, 1e-2, 1e-1, 1.0)
NFOLD = 5


def bvec6(theta):
    bD, bA = float(theta[0]), float(theta[1])
    return np.array([0.0, 0.0, bD, bD, bD, bA])


def bvec8(theta):
    b8 = np.zeros(8)
    b8[list(C.LON6_V7_IDS)] = bvec6(theta)
    return b8


def temperature(theta):
    return float(np.exp(theta[2]))


def apply8(logits8, theta=None, b8=None, T=None):
    """The calibration map on the 8 raw lon logits (what the hook applies at inference)."""
    if theta is not None:
        b8, T = bvec8(theta), temperature(theta)
    return (np.asarray(logits8, np.float64) + np.asarray(b8)[None, :]) / float(T)


def group_logq(L6, theta):
    z = (np.asarray(L6, np.float64) + bvec6(theta)[None, :]) / temperature(theta)
    lp = z - logsumexp(z, axis=1, keepdims=True)
    return np.stack([logsumexp(lp[:, GROUP_OF6 == g], axis=1) for g in (0, 1, 2)], axis=1)


def nll(theta, L6, y, lam=0.0):
    lq = group_logq(L6, theta)
    return float(-lq[np.arange(len(y)), y].mean() + lam * float(np.dot(theta, theta)))


def fit_theta(L6, y, lam):
    r = minimize(nll, np.zeros(3), args=(L6, y, lam), method="L-BFGS-B", bounds=[(-8, 8)] * 2 + [(-2.0, 3.0)], options={"maxiter": 500})
    return r.x


def folds_by_episode(eid, seed):
    u = np.unique(eid)
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    f = {int(e): i % NFOLD for i, e in enumerate(u)}
    return np.array([f[int(e)] for e in eid])


def cv(L6, y, eid, seed):
    fo = folds_by_episode(eid, seed)
    out = {}
    for lam in LAMBDAS:
        tot, n = 0.0, 0
        for k in range(NFOLD):
            tr, te = fo != k, fo == k
            th = fit_theta(L6[tr], y[tr], lam)
            lq = group_logq(L6[te], th)
            tot += float(-lq[np.arange(int(te.sum())), y[te]].sum())
            n += int(te.sum())
        out[lam] = tot / n
    best = min(LAMBDAS, key=lambda l: (round(out[l], 12), -l))       # ties -> larger lam
    return best, out


def const_nll(y, eid, seed):
    """Held-out NLL of the group-prior-only predictor (the constant-only control), same folds."""
    fo = folds_by_episode(eid, seed)
    tot = 0.0
    for k in range(NFOLD):
        tr, te = fo != k, fo == k
        p = np.bincount(y[tr], minlength=3) + 1.0
        p = p / p.sum()
        tot += float(-np.log(p[y[te]]).sum())
    return tot / len(y)


def fit_from_arrays(L6, y, eid):
    res = {"n": int(len(y)), "n_episodes": int(len(np.unique(eid))), "class_counts": {c: int((y == i).sum()) for c, i in YID.items()}}
    lam0, cv0 = cv(L6, y, eid, 0)
    lam1, cv1 = cv(L6, y, eid, 1)
    th0, th1 = fit_theta(L6, y, lam0), fit_theta(L6, y, lam1)
    res.update(lam=lam0, theta=[float(v) for v in th0], cv_table_seed0={str(k): v for k, v in cv0.items()}, cv_table_seed1={str(k): v for k, v in cv1.items()},
               lam_replicate=lam1, theta_replicate=[float(v) for v in th1],
               identity_nll_in_sample=nll(np.zeros(3), L6, y), fitted_nll_in_sample=nll(th0, L6, y), const_cv_nll_seed0=const_nll(y, eid, 0),
               cv_nll_at_chosen_lam_seed0=cv0[lam0])
    res["b"] = {"bD": res["theta"][0], "bA": res["theta"][1]}
    res["T"] = float(np.exp(res["theta"][2]))
    res["b8"] = [float(v) for v in bvec8(th0)]
    return res


def synth(theta_true, n=20000, seed=0, group_of6=GROUP_OF6):
    rng = np.random.default_rng(seed)
    L6 = rng.normal(0, 2.0, size=(n, 6))
    z = (L6 + bvec6(theta_true)[None, :]) / temperature(theta_true)
    lp = z - logsumexp(z, axis=1, keepdims=True)
    q = np.stack([np.exp(logsumexp(lp[:, group_of6 == g], axis=1)) for g in (0, 1, 2)], axis=1)
    u = rng.random(n)[:, None]
    y = (u > q.cumsum(1)).sum(1).clip(0, 2)
    return L6, y, np.arange(n) // 20


def selftest() -> int:
    out = {}
    th_true = np.array([-1.0, 0.5, np.log(1.8)])
    L6, y, eid = synth(th_true)
    th = fit_theta(L6, y, 0.0)
    out["S1_recovery_max_abs"] = float(np.abs(th - th_true).max())
    out["S1_pass"] = bool(out["S1_recovery_max_abs"] < 0.12)
    L6, y, eid = synth(np.zeros(3), seed=1)
    th = fit_theta(L6, y, 0.0)
    out["S2_identity_max_abs"] = float(np.abs(th).max())
    out["S2_pass"] = bool(out["S2_identity_max_abs"] < 0.10)
    # mutation: data generated with BRAKE_TO in the CRUISE group; the fit (correct grouping) must NOT recover the true parameters
    L6, y, eid = synth(th_true, group_of6=np.array([1, 1, 1, 0, 0, 2]), seed=2)
    th = fit_theta(L6, y, 0.0)
    out["S3_mutation_max_abs"] = float(np.abs(th - th_true).max())
    out["S3_RED"] = bool(out["S3_mutation_max_abs"] > 0.25)
    # a shuffled-label control must not beat the constant predictor out of fold
    L6, y, eid = synth(th_true, n=6000, seed=3)
    lam, tab = cv(L6, y, eid, 0)
    rng = np.random.default_rng(5)
    ys = y.copy()
    rng.shuffle(ys)
    lam_s, tab_s = cv(L6, ys, eid, 0)
    out["S4_cv_best_lam"] = lam
    out["S4_shuffled_cv_nll_best"] = float(min(tab_s.values()))
    out["S4_shuffled_const_cv_nll"] = const_nll(ys, eid, 0)
    out["S4_pass"] = bool(min(tab_s.values()) >= const_nll(ys, eid, 0) - 0.01)
    out["ALL_PASS"] = bool(out["S1_pass"] and out["S2_pass"] and out["S3_RED"] and out["S4_pass"])
    print(json.dumps(out, indent=1))
    return 0 if out["ALL_PASS"] else 1


def cmd_fit(a) -> int:
    t0 = time.time()
    T = C.load_train_dump(a.dump_dir, a.tag)
    m = T["valid_s6"] & T["compliant"]
    y = np.array([YID[c] for c in T["gt"][m]])
    L6 = C.lon6(T["logits"][m])
    eid = T["eid"][m]
    res = fit_from_arrays(L6, y, eid)
    res.update(tool="l2_fit.py", tag=T["tag"], dump=T["dump"], extras_sha256=T["extras_sha256"], population="TRAIN S6-valid ceiling-compliant rows",
               n_rows_total=T["n_rows"], n_s6_valid=int(T["valid_s6"].sum()), script_sha256=C.sha256_file(__file__), wall_s=round(time.time() - t0, 1),
               map="lc = (l + b8) / T on the 8 tacv6 lon logits; b8 zero on the masked columns 2 and 6",
               target="trajectory-derived class of the human's 6 s future (s6_lon_decomp.plan_lon_class: last chord vs v0, +-1.5 m/s, dt 1)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    C.jdump(res, a.out)
    keys = ("tag", "n", "n_episodes", "class_counts", "lam", "theta", "T", "b", "identity_nll_in_sample", "fitted_nll_in_sample",
            "cv_nll_at_chosen_lam_seed0", "const_cv_nll_seed0", "lam_replicate", "theta_replicate")
    print(json.dumps({k: res[k] for k in keys}, indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    f = sp.add_parser("fit")
    f.add_argument("--dump-dir", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--tag", default="")
    a = ap.parse_args()
    sys.exit(selftest() if a.cmd == "selftest" else cmd_fit(a))
