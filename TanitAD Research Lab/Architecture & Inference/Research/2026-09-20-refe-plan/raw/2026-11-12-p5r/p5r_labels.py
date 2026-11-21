"""P5-R: regularised tracker inversion (SG/p5r/PREREG_P5R.md, FROZEN sha256 cfd23e06..., 2026-11-12T08:03:43Z).

Pure numpy. The tracker and `to_navsim` are INJECTED, so this module imports no nuPlan and no scipy, and unit tests can
drive it with a synthetic tracker. The driver (`p5r_run.py`) injects NAVSIM's LQR + bicycle via
`p4_tracker_comp.tracked8(ND, P, ego)` and `navsim_dac.to_navsim`, the exact path `iterate20` uses.

Objective (card §2.1), per row:
    J(d; lam) = ||e(d)||^2 + lam * ||g(d)||^2            [m^2]
    e(d) = per knot [T_x - s_x, T_y - s_y, L * wrap(T_h - s_h)],  s = track(to_navsim(traj + d))
    g(d) = [D d_x, D d_y, L * D d_h],  D = second difference of [0, 0, d_1..d_20] (clamped start, unit diagonal)
Optimiser (card §2.2): Levenberg-Marquardt, one forward-difference Jacobian of the tracker in P-space (25 proposals in
one batched call, shared by every lam on the row), good-Broyden rank-1 updates, 3 damping candidates per iteration in one
batched call, K_MAX 10, stop after 3 consecutive failed iterations or an accepted step with ||delta||_inf < 1e-9.
lam = inf returns traj bit for bit.

Imports p5_target_labels (blob 294644df) UNCHANGED for `learnability`, `boot_weights`, `paired_ci`.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

PKG = os.environ.get("P5C_PKG", "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/"
                                "2026-09-20-refe-plan")
for _p in (PKG + "/eval", PKG + "/refe", PKG + "/code"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import p5_target_labels as P5  # noqa: E402  (unchanged; module level imports numpy and the stdlib only)

L_M_PER_RAD = (4.049 + 1.127) / 2.0          # = navsim_dac.HALF_LENGTH, asserted equal by the driver
LAMBDA_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
K_MAX = 10
FD_H = 1e-3                                  # tracker Jacobian step in P-space (m / rad)
A_H = 1e-3                                   # to_navsim Jacobian step
MU0_FRAC = 1e-3
MU_MIN_FRAC = 1e-12
STEP_TOL = 1e-9
MAX_FAILS = 3
N_LAB, N_KNOT = 20, 8

learnability = P5.learnability
boot_weights = P5.boot_weights
paired_ci = P5.paired_ci


def wrap(a):
    return (np.asarray(a, np.float64) + np.pi) % (2 * np.pi) - np.pi


def dmat(n: int = N_LAB) -> np.ndarray:
    """(D d)_k = d_k - 2 d_{k-1} + d_{k-2}, d_0 = d_{-1} = 0: lower triangular, unit diagonal, invertible."""
    D = np.zeros((n, n))
    for k in range(n):
        D[k, k] = 1.0
        if k >= 1:
            D[k, k - 1] = -2.0
        if k >= 2:
            D[k, k - 2] = 1.0
    return D


D20 = dmat()
JG = np.zeros((3 * N_LAB, 3 * N_LAB))        # acts on vec(d) = channel-major [d_x(20), d_y(20), d_h(20)]
JG[:N_LAB, :N_LAB] = D20
JG[N_LAB:2 * N_LAB, N_LAB:2 * N_LAB] = D20
JG[2 * N_LAB:, 2 * N_LAB:] = L_M_PER_RAD * D20
JGTJG = JG.T @ JG
WVEC = np.tile([1.0, 1.0, L_M_PER_RAD], N_KNOT)   # residual weights, knot-major [x, y, h] x 8


def vec(d: np.ndarray) -> np.ndarray:
    return np.asarray(d, np.float64).T.reshape(-1)


def unvec(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, np.float64).reshape(3, N_LAB).T.copy()


def resid(T: np.ndarray, s: np.ndarray) -> np.ndarray:
    """e, knot-major [8*3]: [T_x - s_x, T_y - s_y, L wrap(T_h - s_h)]."""
    return np.concatenate([T[:, :2] - s[:, :2], L_M_PER_RAD * wrap(T[:, 2] - s[:, 2])[:, None]], -1).reshape(-1)


def sdiff(s1: np.ndarray, s0: np.ndarray) -> np.ndarray:
    """s1 - s0 with the heading difference wrapped, knot-major [24]."""
    return np.concatenate([s1[:, :2] - s0[:, :2], wrap(s1[:, 2] - s0[:, 2])[:, None]], -1).reshape(-1)


def pdiff(p1: np.ndarray, p0: np.ndarray) -> np.ndarray:
    return sdiff(p1, p0)


def reg_vec(d: np.ndarray) -> np.ndarray:
    return JG @ vec(d)


def a_jacobian(to_navsim, r20: np.ndarray) -> np.ndarray:
    """dP/dr20 [24, 60] by forward differences of to_navsim itself (no tracker call)."""
    p0 = to_navsim(r20)
    A = np.zeros((3 * N_KNOT, 3 * N_LAB))
    for j in range(3 * N_LAB):
        rp = r20.copy()
        rp[j % N_LAB, j // N_LAB] += A_H
        A[:, j] = pdiff(to_navsim(rp), p0) / A_H
    return A


class Counter:
    """Wraps a batched tracker, counting calls and proposals (card §2.3)."""

    def __init__(self, track):
        self.track, self.calls, self.props, self.log = track, 0, 0, []

    def __call__(self, Pb):
        import time
        t = time.perf_counter()
        out = np.asarray(self.track(np.asarray(Pb, np.float64)), np.float64)
        self.calls += 1
        self.props += int(len(Pb))
        self.log.append((int(len(Pb)), time.perf_counter() - t))
        return out


def solve(traj, d0, lams, track, to_navsim, *, mutate=None, k_max=K_MAX):
    """Solve P5-R for every lam in `lams` on one row, jointly batched. Returns {lam: result}.

    traj [20,3] raw label; d0 [20,3] warm start (Tc20s - traj, or zeros); track(P[B,8,3]) -> s[B,8,3] (ego frame);
    mutate: None | "drop_lambda" (deliberate regression: the lam term never reaches J nor the step).
    """
    traj = np.asarray(traj, np.float64)
    d0 = np.asarray(d0, np.float64)
    T = to_navsim(traj)
    out = {}
    finite = [lm for lm in lams if not math.isinf(lm)]
    for lm in lams:
        if math.isinf(lm):
            out[lm] = {"label": traj.copy(), "d": np.zeros_like(traj), "lam": lm, "path": "identity_inf",
                       "s": None, "J": None, "J_track": None, "J_reg": None, "iters": 0, "accepted": 0,
                       "stop": "lam_inf", "props": 0}
    if not finite:
        return out
    r0 = traj + d0
    P0 = to_navsim(r0)
    batch = [P0]
    for i in range(3 * N_KNOT):
        Pp = P0.copy()
        Pp[i // 3, i % 3] += FD_H
        batch.append(Pp)
    S = track(np.stack(batch))
    s0 = S[0]
    Js0 = np.stack([sdiff(S[1 + i], s0) / FD_H for i in range(3 * N_KNOT)], 1)     # [24, 24]
    A = a_jacobian(to_navsim, r0)
    use_reg = mutate != "drop_lambda"

    def jval(lm, d, s):
        e = resid(T, s)
        jt = float(e @ e)
        if use_reg and lm != 0.0:
            g = reg_vec(d)
            jr = float(lm * (g @ g))
        else:
            jr = 0.0
        return jt + jr, jt, jr, e

    st = {}
    for lm in finite:
        J, jt, jr, e = jval(lm, d0, s0)
        Je = -(WVEC[:, None] * (Js0 @ A))
        H = Je.T @ Je + (lm * JGTJG if use_reg else 0.0)
        mx = float(np.max(np.diag(H)))
        mu_min = MU_MIN_FRAC * mx if mx > 0 else 1e-12
        st[lm] = {"d": d0.copy(), "P": P0.copy(), "s": s0.copy(), "e": e, "J": J, "jt": jt, "jr": jr,
                  "Js": Js0.copy(), "mu": max(MU0_FRAC * mx, mu_min), "mu_min": mu_min, "fails": 0, "k": 0,
                  "acc": 0, "done": False, "stop": None, "props": 25 if lm == finite[0] else 0, "jt0": jt}
    for _k in range(k_max):
        active = [lm for lm in finite if not st[lm]["done"]]
        if not active:
            break
        cands, owners = [], []
        for lm in active:
            z = st[lm]
            Je = -(WVEC[:, None] * (z["Js"] @ A))
            Hb = Je.T @ Je
            grad = Je.T @ z["e"]
            if use_reg and lm != 0.0:
                Hb = Hb + lm * JGTJG
                grad = grad + lm * (JG.T @ reg_vec(z["d"]))
            for mu in (z["mu"] / 10.0, z["mu"], z["mu"] * 10.0):
                mu = max(mu, z["mu_min"])
                dl = np.linalg.solve(Hb + mu * np.eye(3 * N_LAB), -grad)
                dn = z["d"] + unvec(dl)
                cands.append((dn, to_navsim(traj + dn), mu, dl))
                owners.append(lm)
        Sc = track(np.stack([c[1] for c in cands]))
        for lm in active:
            z = st[lm]
            z["k"] += 1
            z["props"] += 3
            idx = [i for i, o in enumerate(owners) if o == lm]
            vals = [(jval(lm, cands[i][0], Sc[i]), i) for i in idx]
            (Jb, jtb, jrb, eb), ib = min(vals, key=lambda v: v[0][0])
            dn, Pn, mub, dl = cands[ib]
            dP = pdiff(Pn, z["P"])
            if float(dP @ dP) > 1e-24:
                ds = sdiff(Sc[ib], z["s"])
                z["Js"] = z["Js"] + np.outer(ds - z["Js"] @ dP, dP) / float(dP @ dP)
            if Jb < z["J"]:
                z.update(d=dn, P=Pn, s=Sc[ib].copy(), e=eb, J=Jb, jt=jtb, jr=jrb, mu=mub, fails=0)
                z["acc"] += 1
                if float(np.max(np.abs(dl))) < STEP_TOL:
                    z["done"], z["stop"] = True, "step_tol"
            else:
                z["mu"] = z["mu"] * 100.0
                z["fails"] += 1
                if z["fails"] >= MAX_FAILS:
                    z["done"], z["stop"] = True, "stall"
    for lm in finite:
        z = st[lm]
        out[lm] = {"label": traj + z["d"], "d": z["d"], "lam": lm, "path": "lm", "s": z["s"], "J": z["J"],
                   "J_track": z["jt"], "J_reg": z["jr"], "iters": z["k"], "accepted": z["acc"],
                   "stop": z["stop"] or "k_max", "props": z["props"], "J_track0": z["jt0"]}
    return out


def tb6(labels: np.ndarray, trajs: np.ndarray) -> dict:
    """Card §5.1: ratio of medians of per-row max|acc| and max|heading rate| (label vs traj)."""
    ll = np.array([learnability(q) for q in labels])
    lt = np.array([learnability(q) for q in trajs])
    return {"R_acc": float(np.median(ll[:, 0]) / np.median(lt[:, 0])),
            "R_hr": float(np.median(ll[:, 1]) / np.median(lt[:, 1])),
            "med_acc_label": float(np.median(ll[:, 0])), "med_acc_traj": float(np.median(lt[:, 0])),
            "med_hr_label": float(np.median(ll[:, 1])), "med_hr_traj": float(np.median(lt[:, 1])),
            "p95_acc_label": float(np.percentile(ll[:, 0], 95)), "p95_hr_label": float(np.percentile(ll[:, 1], 95)),
            "n": int(len(ll))}


def wmedian(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v, kind="stable")
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, 0.5 * c[-1], side="left")])


def projection(lab_m: np.ndarray, raw_m: np.ndarray, bank_raw_m: np.ndarray) -> dict:
    """Card §5.3: full-bank projection of one TB-6 ratio. lab_m/raw_m: the sample rows' label/raw metric;
    bank_raw_m: the raw metric over ALL bank rows of the rank."""
    edges = np.percentile(bank_raw_m, np.arange(10, 100, 10))
    b = np.searchsorted(edges, raw_m, side="right")
    n = len(raw_m)
    cnt = np.bincount(b, minlength=10)
    w = np.array([0.1 / (cnt[i] / n) for i in b])
    dropped = float(0.1 * int((cnt == 0).sum()))
    den = float(np.median(bank_raw_m))
    num = wmedian(lab_m, w)
    return {"R_proj": num / den, "num_wmedian_label": num, "den_bank_median_raw": den, "dropped_mass": dropped,
            "bin_counts": cnt.tolist()}


def tb6_proj(labels, trajs, bank_acc, bank_hr) -> dict:
    ll = np.array([learnability(q) for q in labels])
    lt = np.array([learnability(q) for q in trajs])
    pa = projection(ll[:, 0], lt[:, 0], np.asarray(bank_acc))
    ph = projection(ll[:, 1], lt[:, 1], np.asarray(bank_hr))
    return {"R_acc_proj": pa["R_proj"], "R_hr_proj": ph["R_proj"], "acc": pa, "hr": ph}
