#!/usr/bin/env python3
"""verify_mm.py -- the Master Mind's INDEPENDENT re-derivation of HACC's headline (descriptive; changes no row of the read).
Independent of the build agent's code: its own normal equations, a DIFFERENT solver (L-BFGS-B on 0.5 w'Aw - b'w with bounds [0, 1]), its own
D2 (origin prepended), its own lambda0, its own ADE / accel (float32, the family's definition) and its own episode-cluster bootstrap (numpy,
B = 2000, rng 20261025). Reuses only HSPEC's load_world (the eval arrays HBLEND / HSPEC already read) and the banked train fit windows."""
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

SCR = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"
sys.path.insert(0, SCR + "/hspec")
import v8f_hspec as S  # noqa: E402

DT = np.float32(0.2)


def d2mat(n=30):
    D = np.zeros((n - 1, n))                         # rows k = 1..29 on [p1..p30], p0 = 0 (origin): p_{k+1} - 2 p_k + p_{k-1}
    for k in range(1, n):
        D[k - 1, k] = 1.0
        D[k - 1, k - 1] = -2.0
        if k - 2 >= 0:
            D[k - 1, k - 2] = 1.0
    return D


def normal_eq(r, d, lam, D):
    """sum_n ||w*r_n - d_n||^2 + lam ||D (w*r_n - d_n)||^2 = w'Aw - 2 b'w + c, per axis (r, d: [N, 30])."""
    G = D.T @ D
    A = np.diag((r * r).sum(0)) + lam * (G * (r.T @ r))           # sum_n diag(r_n) G diag(r_n) = G (.) (r' r)
    b = (r * d).sum(0) + lam * ((r * (d @ G.T))).sum(0)            # sum_n diag(r_n) G d_n
    return A, b


def solve(A, b):
    f = lambda w: (0.5 * w @ A @ w - b @ w, A @ w - b)            # noqa: E731
    res = minimize(f, np.full(len(b), 0.5), jac=True, method="L-BFGS-B", bounds=[(0, 1)] * len(b), options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 20000})
    return res.x


def terms(r, d, w, D):
    e = w * r - d
    return float((e * e).sum()), float(((e @ D.T) ** 2).sum())


def accel(p):
    p = p.astype(np.float32)
    q = np.concatenate([np.zeros_like(p[:, :1]), p], 1)
    v = np.linalg.norm(np.diff(q, axis=1), axis=-1) / DT
    return np.diff(v, axis=1) / DT


def pw(plan, gt):
    e = np.linalg.norm(plan - gt, axis=-1)
    return {"ade2": e[:, :10].mean(1), "ade6": e.mean(1), "accel": np.abs(accel(plan) - accel(gt)).mean(1).astype(np.float64)}


def boot(x, eid, nb=2000, seed=20261025):
    u, inv = np.unique(eid, return_inverse=True)
    s = np.bincount(inv, weights=x); c = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(nb, len(u)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return {"mean": float(x.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5))}


def main():
    D = d2mat()
    out = {}
    for seed in ("M2b_s0", "M2b_s1"):
        z = np.load(f"{SCR}/hblend/fit_windows_{seed}.npz")
        ok = z["gt_ok"].astype(bool)
        tau, P, y = (z[k].astype(np.float64) for k in ("traj30", "p30", "gt"))
        Wb = json.load(open(f"{SCR}/hblend/weights_{seed}.json"))
        wh = np.stack([np.asarray(Wb["w_along_by_step"]), np.asarray(Wb["w_cross_by_step"])], 1)
        Hc = json.load(open(f"{SCR}/hacc/weights_HACC_{seed}.json"))
        wb = np.stack([np.asarray(Hc["w_along_by_step"]), np.asarray(Hc["w_cross_by_step"])], 1)
        rec = {"n_train": int(len(y)), "n_train_gt_ok": int(ok.sum()), "m_star_banked": Hc["m_star"], "lam_star_banked": Hc["lam_star"], "lambda0_banked": Hc["lambda0"]}
        for tag, sel in (("all", np.ones(len(y), bool)), ("gt_ok", ok)):
            r, d = (tau - P)[sel], (y - P)[sel]
            pos = sum(terms(r[:, :, a], d[:, :, a], wh[:, a], D)[0] for a in range(2))
            dd = sum(terms(r[:, :, a], d[:, :, a], wh[:, a], D)[1] for a in range(2))
            lam = Hc["m_star"] * pos / dd
            w = np.stack([solve(*normal_eq(r[:, :, a], d[:, :, a], lam, D)) for a in range(2)], 1)
            rec[f"{tag}: lambda0_mine"] = pos / dd
            rec[f"{tag}: w_mine_vs_banked_maxabs"] = float(np.abs(w - wb).max())
        W = S.load_world(seed)
        P_e, tau_e, gt, eid = W["P"], W["tau"], W["gt"], W["eid"]
        hacc = P_e + wb[None] * (tau_e - P_e)
        a, p, h = pw(hacc, gt), pw(W["P_file"], gt), pw(W["H"], gt)
        rec["ade6_mine_vs_arm_pw_P10_maxabs"] = float(np.abs(p["ade6"] - np.asarray(W["PW"]["P10"]["ade6"], np.float64)).max())
        rec["accel_mine_vs_arm_pw_P10_maxabs"] = float(np.abs(p["accel"] - np.asarray(W["PW"]["P10"]["fw_lon_accel_mae"], np.float64)).max())
        for k in ("ade2", "ade6", "accel"):
            rec[f"HACC-P10 {k}"] = boot(a[k] - p[k], eid)
        rec["KEEP HACC-H ade6"] = boot(a["ade6"] - h["ade6"], eid)
        out[seed] = rec
        print(seed, json.dumps(rec, indent=1), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_mm.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
