#!/usr/bin/env python3
"""v8f_hlat.py -- executes PREREG_V8F_HACC_LAT.md (dev box, CPU only, ZERO training): fits, bakes and the controls of the two arms.

    A  HACC-T1  HACC's [30, 2] weights re-fitted on the TICK samples of the HACC model's own T1-kin loop (m = 10) over HBLEND's 141 TRAIN clips: HACC's objective (position + lambda * D2, origin
               prepended, w in [0, 1], no monotone constraint) and its train-only lambda rule on the grid m in {0, 1, 4, 16, 64}.
    B  HACC-V  HACC unchanged except that the CROSS weight is 0 (the cross axis falls back to P) when the measured tick speed v0 < v*; v* in {2, 4, 6, 8} by the PREREG's rule on the train tick samples.

  fit            --seed 0|1 --ticks ticks_*.npz --out-dir DIR      Phase A, TRAIN tick samples only: arm A's lambda rule + C-A2, arm B's v* rule, the deliberate-regression control
  bake           --seed 0|1 --arm A|B|CA1|CB1|CB2 --out-dir DIR    plain scored checkpoint + the table (+ the gate) -> a NEW checkpoint file, verified (never overwrites anything)
  compare-dumps  --a DIR --b DIR --label C-A1 --tol 1e-9           C-A1 / C-B1: two T1 dump directories equal to <= tol on cl / ol / fl
  check-t2       --seed 0|1 --ticks FILE                           C-T2: the tick-0 samples equal the open-loop batch's tau and P at tick-0 ages (and the GT, and v0)
  check-b2       --ticks FILE                                      C-B2: with v* = +inf every tick's cross-axis plan equals P's cross plan
  check-ct       --a DIR --b DIR                                   C-T: the tool's output with the flag absent equals the original tool's, bit for bit

DECLARED INTERPRETATIONS (each a choice the PREREG text leaves open; ALL fixed before any tick sample existed):
  J1  lambda0 = S1 / S2 pooled over samples, steps and axes at HBLEND's banked w (``hblend/weights_M2b_s*.json``), EVALUATED ON THE TICK SAMPLES (HACC evaluated it on its fit set); lambda = m * lambda0.
  J2  "P10" in the lambda rule is the tick sample's own P (the checkpoint's frozen prior at the tick); an arm's MAE is the mean over the samples where BOTH it and P10 are finite (HACC's I5).
  J3  A tick sample without a complete logged 6 s future (``gt_ok`` False: the clip ends inside the plan) cannot enter HACC's objective and is dropped; the count is reported.
  J4  Arm B's rule, read literally: v* is the smallest grid value with  (i) paired mean curvature MAE of (HACC-V - P10) over the first 10 knots <= 0  AND  (ii) |ADE6(HACC-V) - ADE6(HACC)| <= F/2, F/2 = 0.0604 m as
      the PREREG prints it; the one-sided reading (HACC-V - HACC <= F/2) and the exact F/2 = 0.06037 are reported beside (``reading_sensitivity``). None qualifies => v* = 8.
  J5  Arm B's reference weights are HACC's deployed table (``weights_HACC_M2b_s*.json``, canonical sha pinned); the v* rule scores them on the train tick samples of the HACC loop (not of arm B's own loop).
  J6  A / B reuse HACC's fitter by VERBATIM copies of its pure functions (``tests/test_v8f_hlat.py`` pins each copy against the source file by AST equality).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np  # noqa: E402
from scipy.optimize import lsq_linear  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SC = os.environ.get("HLAT_SCRATCH", "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad")
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
HACC_RUN = REPO / PK / "raw" / "2026-10-25-hacc" / "run"
SEEDS = ("M2b_s0", "M2b_s1")
DT, NSTEP, NAX = 0.2, 30, 2
GRID_M_A = (0, 1, 4, 16, 64)                       # PREREG_V8F_HACC_LAT section 2, arm A (HACC's grid without 256)
GRID_V_B = (2.0, 4.0, 6.0, 8.0)                    # arm B
V_B_FALLBACK = 8.0
RULE_RATIO = 0.98
KKT_TOL = 1e-6
F_PRINTED, F_HALF_PRINTED, F_HALF_EXACT = 0.1207, 0.0604, 0.06037
GATE_AXIS = 1                                      # cross
N_TRAIN_WINDOWS = 9582
KEY = {"ade2": "ade2", "ade6": "ade6", "accel": "fw_lon_accel_mae", "curv": "fw_lat_curvature_mae_1pm"}

PLAIN_CKPT = {s: f"{SC}/viz/work/ckpt/{s}.pt" for s in SEEDS}
HACC_CKPT = {s: f"E:/Projects/TanitAD-artifacts/v8f-mini-hacc/{s}_hacc.pt" for s in SEEDS}
BANKED_W = {s: str(REPO / PK / "raw" / "2026-10-25-hblend" / f"weights_{s}.json") for s in SEEDS}      # HBLEND's banked w (repo copy; sha pinned below)
HACC_W_JSON = {s: str(HACC_RUN / f"weights_HACC_{s}.json") for s in SEEDS}
TRAIN_CACHE, TRAIN_JOIN, FIT_CLIPS = f"{SC}/hblend/cache", f"{SC}/hblend/box2/join_train.npz", f"{SC}/hblend/fit_clips.json"
EVAL_JOIN = str(REPO / PK / "raw" / "join_eval141.npz")
TRACKS = f"{SC}/sfresh/inputs/native_tracks.pt"
SHA = {"plain": {"M2b_s0": "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8", "M2b_s1": "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"},
       "hacc": {"M2b_s0": "71de3c6195ce651ee656d26579bd68a0cc5d7370090044140ed634fe10b627c0", "M2b_s1": "32e4809dc4204e70544bceb5c3c57d54310c387decd3b95ae0a830177616523a"},
       "banked_w": {"M2b_s0": "dd260b4eaf114454bdc2f0a55c9eb601fd62de4e4c13ec0dad590b4b30eeff9d", "M2b_s1": "987a0b5f07393e399d1482f96ca9a01c338386d952594776edb3426cc6d65227"},
       "hacc_w_canonical": {"M2b_s0": "5c330a9ba98d66ccdaf3770b518f2a23ffee2612987684776a76a89eb33e9b7f", "M2b_s1": "eee64e066792404710eff44009e4eb6e961466060a72ce378a8706d271d4e0a1"},
       "fit_clips": "e419b10e3b1234986eb6b0d0a3372b04", "train_join": "9ebc7a48cbca45cc634eab2aab5a3960", "eval_join": "532fcd644c0e0835e847b2f1cee011ad"}   # the last three are md5


class KKTError(RuntimeError):
    """A fitted w is not the bounded optimum (relative projected gradient above the bar)."""


class EvalLeakError(RuntimeError):
    """A fit input contains an eval-141 window: REFUSED."""


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canon(w):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(w, np.float64)).tobytes()).hexdigest()


def jdump(o, path):
    def enc(x):
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(o, fh, indent=1, default=enc)


# ================================================================================================ HACC's pure functions, VERBATIM (v8f_hacc.py; pinned by AST equality in test_v8f_hlat.py)
def d2_matrix(K: int = NSTEP) -> np.ndarray:
    """[K-1, K]: row k-1 (cell k = 1..K-1) = e_(k-1) - 2 e_k + e_(k+1) acting on (e_1..e_K) with the ORIGIN prepended e_0 = 0 (its column drops out of the matrix)."""
    full = np.zeros((K - 1, K + 1))                                                # columns e_0 .. e_K
    for k in range(1, K):
        full[k - 1, k - 1], full[k - 1, k], full[k - 1, k + 1] = 1.0, -2.0, 1.0
    return full[:, 1:]


def stack_system(r: np.ndarray, d: np.ndarray, lam: float, D2: np.ndarray):
    """One axis. r, d [N, 30] -> (A, b) with 0.5||A w - b||^2 = sum_n [ sum_t (w_t r_nt - d_nt)^2 + lam sum_k (D2 (w r_n - d_n))_k^2 ]."""
    N = len(r)
    idx = np.arange(NSTEP)
    pos = np.zeros((N, NSTEP, NSTEP))
    pos[:, idx, idx] = r
    rows, tgt = [pos.reshape(-1, NSTEP)], [np.asarray(d, np.float64).reshape(-1)]
    if lam > 0:
        s = np.sqrt(lam)
        rows.append((s * D2[None, :, :] * r[:, None, :]).reshape(-1, NSTEP))
        tgt.append((s * (d @ D2.T)).reshape(-1))
    return np.concatenate(rows), np.concatenate(tgt)


def kkt_rel(A: np.ndarray, b: np.ndarray, w: np.ndarray, eps: float = 1e-9) -> float:
    """Relative projected-gradient norm of 0.5||A w - b||^2 over [0, 1]^n at w (0 = exact optimum)."""
    g = A.T @ (A @ w - b)
    pg = np.where(w <= eps, np.minimum(g, 0.0), np.where(w >= 1 - eps, np.maximum(g, 0.0), g))
    scale = max(float(np.abs(A.T @ b).max()), 1e-12 * float(np.abs(A.T @ A).max()), 1e-300)
    return float(np.abs(pg).max() / scale)


def terms_at(tau, P, y, w, D2: np.ndarray | None = None):
    """(S1, S2): the position and the D2 terms of the objective at w, summed over windows, steps and BOTH axes (lambda = 1)."""
    D2 = d2_matrix() if D2 is None else D2
    e = np.asarray(w, np.float64)[None] * (np.asarray(tau, np.float64) - np.asarray(P, np.float64)) - (np.asarray(y, np.float64) - np.asarray(P, np.float64))
    return float((e ** 2).sum()), float((np.einsum("kt,nta->nka", D2, e) ** 2).sum())


def qualifies(r: dict) -> bool:
    return bool(r["accel"] <= RULE_RATIO * r["p_accel"] and r["curv"] <= RULE_RATIO * r["p_curv"] and r["ade2"] < r["p_ade2"] and r["ade6"] < r["p_ade6"])


def choose_lambda(rows: list):
    """The PREREG's rule on the grid rows (pure).  -> (row, branch)."""
    rows = sorted(rows, key=lambda r: r["lam"])
    for r in rows:
        if qualifies(r):
            return r, "qualified: the smallest grid lambda meeting all four train conditions"
    cand = [r for r in rows if r["ade2"] < r["p_ade2"]]
    if cand:
        return min(cand, key=lambda r: r["accel"]), "fallback 1: no grid lambda qualifies; the lowest train accel MAE among those with train ADE2 < P10's"
    return rows[0], "fallback 2: no grid lambda qualifies and none has train ADE2 < P10's; lambda* = 0"


def apply_w(P, tau, w):
    """H = P + w * (tau - P), w broadcast over windows ([30, 2] or scalar)."""
    P, tau = np.asarray(P, np.float64), np.asarray(tau, np.float64)
    return P + np.asarray(w, np.float64) * (tau - P)


# ================================================================================================ HACC's fitter, with the one scipy import it needs
def fit_w(tau, P, y, lam: float = 0.0, D2: np.ndarray | None = None, solver=lsq_linear) -> dict:
    """The PREREG's fit.  tau, P, y [N, 30, 2] -> {'w' [30, 2], 'kkt' [2], 'cost' [2], 'nit' [2], 'status' [2]}.  No isotonic step.  Raises KKTError if a fit is not optimal."""
    tau, P, y = (np.asarray(v, np.float64) for v in (tau, P, y))
    r, d = tau - P, y - P
    D2 = d2_matrix() if D2 is None else D2
    w, kkt, cost, nit, status = np.zeros((NSTEP, NAX)), [], [], [], []
    for a in range(NAX):
        A, b = stack_system(r[:, :, a], d[:, :, a], lam, D2)
        res = solver(A, b, bounds=(0.0, 1.0), method="bvls", tol=1e-12, max_iter=500)
        w[:, a] = res.x
        k = kkt_rel(A, b, res.x)
        if not (k <= KKT_TOL):
            raise KKTError(f"axis {a}, lambda {lam:.6g}: KKT residual {k:.3e} > {KKT_TOL:g}")
        kkt.append(k)
        cost.append(float(res.cost))
        nit.append(int(res.nit))
        status.append(int(res.status))
    return {"w": w, "kkt": kkt, "cost": cost, "nit": nit, "status": status}


def clipped_closed_form(tau, P, y):
    """HBLEND's ``fit_weights`` up to the clip (no isotonic step): raw = sum r d / sum r^2 per step and axis, clipped to [0, 1] -- the lambda = 0 optimum of the box-constrained fit."""
    tau, P, y = (np.asarray(v, np.float64) for v in (tau, P, y))
    r, d = tau - P, y - P
    num, den = (r * d).sum(0), (r * r).sum(0)
    if not (den > 0).all():
        raise ValueError("a step/axis with sum r^2 == 0: the closed form is undefined")
    return np.clip(num / den, 0.0, 1.0)


# ================================================================================================ the train-only tick bundle
@dataclass(frozen=True)
class TickBundle:
    """The TRAIN tick samples and nothing else (HACC's TrainBundle, with the speed of the tick). ``from_ticks`` REFUSES a file that is not the train split of HBLEND's 141 clips, or that shares a clip with eval-141."""
    tau: np.ndarray
    P: np.ndarray
    y: np.ndarray
    v0: np.ndarray
    banked_w: np.ndarray
    split: str = "train"
    source: str = ""

    def __post_init__(self):
        if self.split != "train":
            raise ValueError(f"TickBundle holds TRAIN tick samples only (split={self.split!r})")
        if not (self.tau.shape == self.P.shape == self.y.shape and self.tau.ndim == 3 and self.tau.shape[1:] == (NSTEP, NAX)):
            raise ValueError(f"bad tick shapes {self.tau.shape} {self.P.shape} {self.y.shape}")
        if self.v0.shape != (len(self.tau),) or self.banked_w.shape != (NSTEP, NAX):
            raise ValueError("bad v0 / banked_w shapes")
        if not (np.isfinite(self.tau).all() and np.isfinite(self.P).all() and np.isfinite(self.y).all() and np.isfinite(self.v0).all()):
            raise ValueError("a non-finite tick sample reached the bundle")

    @classmethod
    def from_ticks(cls, path, seed: str, *, eval_join=EVAL_JOIN, fit_clips=FIT_CLIPS, banked_w=None, expect_windows: int | None = N_TRAIN_WINDOWS):
        z = np.load(path, allow_pickle=False)
        if str(z["split"]) != "train":
            raise EvalLeakError(f"{path}: split {str(z['split'])!r} is not 'train'")
        clips = {str(c) for c in z["clip"]}
        fit = set(json.load(open(fit_clips, encoding="utf-8"))["clips"])
        ev = {str(c) for c in np.load(eval_join, allow_pickle=False)["clip"]}
        if clips & ev or fit & ev:
            raise EvalLeakError(f"{path}: {len(clips & ev)} sample clips / {len(fit & ev)} fit clips are in the eval-141 join")
        if not clips <= fit:
            raise EvalLeakError(f"{path}: {len(clips - fit)} sample clips are not in fit_clips.json")
        if expect_windows is not None and int(z["n_windows"]) != expect_windows:
            raise ValueError(f"{path}: {int(z['n_windows'])} windows, expected {expect_windows}")
        ok = z["gt_ok"].astype(bool)
        bw = banked_w if banked_w is not None else read_table(BANKED_W[seed], SHA["banked_w"][seed])
        return cls(tau=z["tau"][ok].astype(np.float64), P=z["P"][ok].astype(np.float64), y=z["gt"][ok].astype(np.float64), v0=z["v0"][ok].astype(np.float64),
                   banked_w=bw, split="train", source=str(path)), {"n_samples": int(len(ok)), "n_gt_ok": int(ok.sum()), "n_windows": int(z["n_windows"]),
                                                                    "by_tick": {int(t): [int((z["tick"] == t).sum()), int(((z["tick"] == t) & ok).sum())] for t in np.unique(z["tick"])},
                                                                    "n_clips": len(clips), "sha256": sha256_file(path), "checked": "split == train; clips subset of fit_clips.json; fit_clips and sample clips disjoint from the eval-141 join"}


def read_table(path, sha=None) -> np.ndarray:
    """A weights JSON (``w_along_by_step`` / ``w_cross_by_step``) -> float64 [30, 2]; ``sha`` pins the FILE bytes when given."""
    if sha is not None and sha256_file(path) != sha:
        raise ValueError(f"{path}: not the pinned weights file")
    d = json.load(open(path, encoding="utf-8"))
    return np.stack([np.asarray(d["w_along_by_step"], np.float64), np.asarray(d["w_cross_by_step"], np.float64)], 1)


def write_table(path, w, extra: dict | None = None):
    w = np.asarray(w, np.float64)
    jdump({"w_along_by_step": w[:, 0], "w_cross_by_step": w[:, 1], **(extra or {})}, path)


# ================================================================================================ metrics on tick samples
def _be():
    from taniteval import v8f_beta_eval as BE
    return BE


def arm_pw(pred, gt, BE=None):
    """Per-sample scalars of one arm: ADE / FDE at 1..6 s and the family components (taniteval's own ``family_windows``) -- HBLEND's ``arm_pw`` verbatim."""
    BE = BE or _be()
    pred = np.asarray(pred, np.float64)
    d = np.linalg.norm(pred - gt, axis=-1)
    pw = {}
    for h in range(1, 7):
        pw[f"ade{h}"] = d[:, :5 * h].mean(1)
        pw[f"fde{h}"] = d[:, 5 * h - 1]
    for k, v in BE.family_windows(pred, gt, DT).items():
        pw["fw_" + k] = v
    return pw


def paired_means(pw_a: dict, pw_p: dict) -> dict:
    """{ade2, ade6, accel, curv, p_ade2, ...}: each pair of means over the samples where BOTH arms are finite (HACC's I5)."""
    out = {}
    for nm, k in KEY.items():
        a, p = np.asarray(pw_a[k], np.float64), np.asarray(pw_p[k], np.float64)
        ok = np.isfinite(a) & np.isfinite(p)
        out[nm], out["p_" + nm] = float(a[ok].mean()), float(p[ok].mean())
    return out


def lambda0(T: TickBundle) -> float:
    """lambda0 = (position term) / (D2 term), both evaluated at HBLEND's banked w on the TICK samples (pooled over axes) -- J1."""
    s1, s2 = terms_at(T.tau, T.P, T.y, T.banked_w)
    return s1 / s2


def select_lambda(T: TickBundle, grid=GRID_M_A, metrics_fn=None, fit=None) -> dict:
    """lambda0, the grid fits, the train metrics and lambda*. Its inputs are the TRAIN tick samples alone. ``fit`` is the test seam (default :func:`fit_w`)."""
    if not isinstance(T, TickBundle) or T.split != "train":
        raise TypeError("the lambda selection receives a train TickBundle alone")
    metrics_fn, fit = metrics_fn or arm_pw, fit or fit_w
    D2 = d2_matrix()
    lam0 = lambda0(T)
    pwP = metrics_fn(T.P, T.y)
    rows = []
    for m in grid:
        lam = float(m * lam0)
        t0 = time.time()
        f = fit(T.tau, T.P, T.y, lam, D2)
        plan = apply_w(T.P, T.tau, f["w"])
        row = {"m": m, "lam": lam, **paired_means(metrics_fn(plan, T.y), pwP), "w": f["w"], "kkt": f["kkt"], "nit": f["nit"], "status": f["status"], "seconds": time.time() - t0}
        row["qualifies"] = qualifies(row)
        rows.append(row)
        print(f"[fit] m {m:>3} lambda {lam:.6g}  ADE2 {row['ade2']:.4f}/{row['p_ade2']:.4f}  ADE6 {row['ade6']:.4f}/{row['p_ade6']:.4f}  accel {row['accel']:.4f}/{row['p_accel']:.4f}  "
              f"curv {row['curv']:.5f}/{row['p_curv']:.5f}  qualifies {row['qualifies']}  kkt {max(f['kkt']):.1e}  {row['seconds']:.0f}s", flush=True)
    star, branch = choose_lambda(rows)
    return {"lambda0": lam0, "grid": rows, "m_star": star["m"], "lam_star": star["lam"], "w_star": star["w"], "rule_branch": branch}


# ================================================================================================ arm B: the speed gate
def gated_w(w, v0, vstar: float) -> np.ndarray:
    """Per-sample weights [N, 30, 2]: HACC's table, with the CROSS column 0 where the measured tick speed v0 < v* (the model's ``_hacc_post_map`` in float64)."""
    v0 = np.asarray(v0, np.float64)
    W = np.broadcast_to(np.asarray(w, np.float64)[None], (len(v0), NSTEP, NAX)).copy()
    W[v0 < float(vstar), :, GATE_AXIS] = 0.0
    return W


def apply_gated(P, tau, w, v0, vstar: float):
    P, tau = np.asarray(P, np.float64), np.asarray(tau, np.float64)
    return P + gated_w(w, v0, vstar) * (tau - P)


def curv10(plan, y):
    """Per-sample curvature MAE over the FIRST 10 knots (2 s: the part a replan at m = 10 executes)."""
    return _be().family_windows(np.asarray(plan)[:, :10], np.asarray(y)[:, :10], DT)["lat_curvature_mae_1pm"]


def paired_mean_diff(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    return float((a[ok] - b[ok]).mean())


def select_vstar(T: TickBundle, w_hacc, grid=GRID_V_B, f_half: float = F_HALF_PRINTED) -> dict:
    """PREREG arm B: v* = the smallest grid value with (i) the paired mean curvature MAE (first 10 knots) of HACC-V - P10 <= 0 AND (ii) |ADE6(HACC-V) - ADE6(HACC)| <= F/2 on the train tick samples; else 8. J4."""
    if not isinstance(T, TickBundle) or T.split != "train":
        raise TypeError("the v* selection receives a train TickBundle alone")
    ade6 = lambda p: np.linalg.norm(p - T.y, axis=-1).mean(1)
    hacc = apply_w(T.P, T.tau, w_hacc)
    a6_h, c10_p, c10_h = ade6(hacc).mean(), curv10(T.P, T.y), curv10(hacc, T.y)
    rows = []
    for v in grid:
        plan = apply_gated(T.P, T.tau, w_hacc, T.v0, v)
        d_curv = paired_mean_diff(curv10(plan, T.y), c10_p)
        d_ade6 = float(ade6(plan).mean() - a6_h)
        row = {"v_star": float(v), "curv10_minus_p10": d_curv, "ade6": float(ade6(plan).mean()), "ade6_minus_hacc": d_ade6, "frac_fired": float((T.v0 < v).mean()),
               "cond_curv": bool(d_curv <= 0.0), "cond_ade6": bool(abs(d_ade6) <= f_half), "cond_ade6_one_sided": bool(d_ade6 <= f_half), "cond_ade6_exactF": bool(abs(d_ade6) <= F_HALF_EXACT)}
        row["qualifies"] = row["cond_curv"] and row["cond_ade6"]
        rows.append(row)
    first = lambda key: next((r["v_star"] for r in rows if r["cond_curv"] and r[key]), V_B_FALLBACK)
    star = first("cond_ade6")
    return {"v_star": star, "branch": "qualified" if any(r["qualifies"] for r in rows) else f"fallback: none qualifies, v* = {V_B_FALLBACK:g}", "grid": rows,
            "reference": {"ade6_hacc": float(a6_h), "curv10_hacc_minus_p10": paired_mean_diff(c10_h, c10_p), "curv10_p10": float(np.nanmean(c10_p)), "n": int(len(T.tau))},
            "f_half_used": f_half, "reading_sensitivity": {"one_sided_ade6": first("cond_ade6_one_sided"), "exact_F_half": first("cond_ade6_exactF"),
                                                              "same_as_used": first("cond_ade6_one_sided") == star and first("cond_ade6_exactF") == star}}


# ================================================================================================ fit
def cmd_fit(a) -> int:
    seed = SEEDS[a.seed]
    t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    pins = {"fit_clips_md5": (FIT_CLIPS, SHA["fit_clips"]), "eval_join_md5": (EVAL_JOIN, SHA["eval_join"])}
    for nm, (p, h) in pins.items():
        if md5_file(p) != h:
            raise SystemExit(f"REFUSED: {nm} {p} is not the pinned file")
    T, tinfo = TickBundle.from_ticks(a.ticks, seed, expect_windows=a.expect_windows)
    w_hacc = read_table(HACC_W_JSON[seed])
    if canon(w_hacc) != SHA["hacc_w_canonical"][seed]:
        raise SystemExit("REFUSED: HACC's weights table is not the scored one")
    print(f"[fit {seed}] tick samples {tinfo['n_samples']} gt_ok {tinfo['n_gt_ok']} by tick {tinfo['by_tick']} clips {tinfo['n_clips']}", flush=True)
    # ---- arm A
    sel = select_lambda(T)
    g0 = next(r for r in sel["grid"] if r["m"] == 0)
    cf = clipped_closed_form(T.tau, T.P, T.y)
    ca2 = float(np.abs(g0["w"] - cf).max())                                                       # C-A2
    star = next(r for r in sel["grid"] if r["m"] == sel["m_star"])
    reg = fit_w(T.tau, T.P, T.P, star["lam"])                                                     # deliberate regression: GT := P  =>  w = 0
    reg_w = float(np.abs(reg["w"]).max())
    controls = {"C-A2_lam0_vs_clipped_closed_form_maxabs": ca2, "C-A2_pass": bool(ca2 <= 1e-12), "regression_gt_is_P_w_maxabs_at_lam_star": reg_w, "regression_pass": bool(reg_w <= 1e-12),
                "kkt_max_over_fits": float(max(max(r["kkt"]) for r in sel["grid"] + [{"kkt": reg["kkt"]}])), "kkt_pass": bool(max(max(r["kkt"]) for r in sel["grid"]) <= KKT_TOL)}
    # ---- arm B
    vb = select_vstar(T, w_hacc)
    write_table(f"{a.out_dir}/weights_A_{seed}.json", sel["w_star"], {"arm": "A", "seed": seed, "m_star": sel["m_star"], "lam_star": sel["lam_star"]})
    jdump({"arm": "B", "seed": seed, "v_star": vb["v_star"], "weights": f"weights_HACC_{seed}.json (canonical sha {SHA['hacc_w_canonical'][seed][:12]})"}, f"{a.out_dir}/gate_B_{seed}.json")
    rec = {"seed": seed, "prereg": "PREREG_V8F_HACC_LAT", "tier": "T1-kin tick samples, TRAIN windows, HACC model's own loop (m = 10); fit offline, zero training", "ticks": tinfo, "grid_m": list(GRID_M_A),
           "lambda0": sel["lambda0"], "m_star": sel["m_star"], "lam_star": sel["lam_star"], "rule_branch": sel["rule_branch"], "grid": sel["grid"], "w_star": sel["w_star"], "w_hacc": w_hacc,
           "banked_w_sha256": SHA["banked_w"][seed], "controls": controls, "arm_B": vb, "seconds": time.time() - t0,
           "mean_abs_w_star_minus_hacc": float(np.abs(sel["w_star"] - w_hacc).mean()), "max_abs_w_star_minus_hacc": float(np.abs(sel["w_star"] - w_hacc).max())}
    jdump(rec, f"{a.out_dir}/hlat_fit_{seed}.json")
    print(f"[fit {seed}] m* {sel['m_star']} lambda* {sel['lam_star']:.6g} ({sel['rule_branch'][:30]})  v* {vb['v_star']:g} ({vb['branch']}; same under other readings: "
          f"{vb['reading_sensitivity']['same_as_used']})  controls {json.dumps({k: v for k, v in controls.items() if k.endswith('pass')})}  {time.time() - t0:.0f}s", flush=True)
    return 0 if (controls["C-A2_pass"] and controls["regression_pass"] and controls["kkt_pass"]) else 2


# ================================================================================================ bake
ARMS = ("A", "B", "CA1", "CB1", "CB2")


def _build(ck):
    import torch
    from tanitad.models import v8f_mini as M
    m = M.V8FMini(M.V8FMiniConfig(**ck["cfg"]), M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    m.load_state_dict(ck["model"])                                   # strict, exactly as the T1 tool and the eval harness
    return m.eval()


def arm_spec(arm: str, seed: str, fit_dir: str | None):
    """-> (weights [30, 2], gate v* or None, provenance dict). A: the fitted table; B: HACC's table + the chosen v*; CA1: A's pipeline fed HACC's own table; CB1: HACC + v* = 0; CB2: HACC + v* = +inf."""
    w_h = read_table(HACC_W_JSON[seed])
    if canon(w_h) != SHA["hacc_w_canonical"][seed]:
        raise ValueError("HACC's weights table is not the scored one")
    if arm == "A":
        p = f"{fit_dir}/weights_A_{seed}.json"
        return read_table(p), None, {"weights": p, "weights_sha256": sha256_file(p), "fit": f"{fit_dir}/hlat_fit_{seed}.json"}
    if arm == "B":
        p = f"{fit_dir}/gate_B_{seed}.json"
        return w_h, float(json.load(open(p))["v_star"]), {"gate": p, "gate_sha256": sha256_file(p), "weights": HACC_W_JSON[seed]}
    if arm == "CA1":
        return w_h, None, {"weights": HACC_W_JSON[seed], "note": "arm A's bake pipeline fed HACC's own table (control C-A1)"}
    if arm == "CB1":
        return w_h, 0.0, {"weights": HACC_W_JSON[seed], "note": "HACC + gate v* = 0: never fires (control C-B1)"}
    if arm == "CB2":
        return w_h, float("inf"), {"weights": HACC_W_JSON[seed], "note": "HACC + gate v* = +inf: always fires (control C-B2)"}
    raise ValueError(f"arm must be one of {ARMS}")


def bake(arm: str, seed: str, out_path: str, fit_dir: str | None = None, plain_path: str | None = None, verify_batch=None) -> dict:
    """The scored plain checkpoint (sha pinned) + the arm's table (+ gate) -> a NEW file (REFUSES to overwrite), then verified: every source tensor carried bit for bit plus exactly the new buffer(s); the
    baked file loads strictly and its ``forward_infer`` equals the runtime-set model's bit for bit on every output tensor; CA1 additionally equals HACC's shipped checkpoint tensor for tensor."""
    import torch
    plain = plain_path or PLAIN_CKPT[seed]
    if plain_path is None and sha256_file(plain) != SHA["plain"][seed]:
        raise ValueError(f"{plain}: not the scored checkpoint")
    if os.path.exists(out_path):
        raise FileExistsError(f"{out_path} exists: a baked checkpoint is never overwritten")
    w, v, prov = arm_spec(arm, seed, fit_dir)
    ck = torch.load(plain, map_location="cpu", weights_only=False)
    m_src, m_rt = _build(ck), _build(ck)
    m_rt.set_hacc_weights(w)
    if v is not None:
        m_rt.set_hacc_gate(v)
    sd = m_rt.state_dict()
    prov = {"prereg": "PREREG_V8F_HACC_LAT", "arm": arm, "seed": seed, "source_ckpt_sha256": SHA["plain"][seed] if plain_path is None else sha256_file(plain), "gate_v_star": v,
            "weights_canonical_sha256": canon(w), **prov, "what": "forward_infer returns P + w (O - P); with a gate the CROSS weight is 0 where the measured tick speed < v*; training (_forward) is untouched"}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save({**ck, "model": sd, "hlat": prov}, out_path)
    ck2 = torch.load(out_path, map_location="cpu", weights_only=False)
    new = {"hacc_w"} | ({"hacc_gate_v"} if v is not None else set())
    assert set(ck2["model"]) - set(ck["model"]) == new and set(ck["model"]) <= set(ck2["model"]), set(ck2["model"]) ^ set(ck["model"])
    assert all(torch.equal(ck2["model"][k], ck["model"][k]) for k in ck["model"]), "a source tensor changed"
    assert torch.equal(ck2["model"]["hacc_w"], torch.as_tensor(w, dtype=torch.float32)), "hacc_w is not the table"
    if v is not None:
        assert torch.equal(ck2["model"]["hacc_gate_v"], torch.tensor(v, dtype=torch.float32)), "hacc_gate_v is not v*"
    m_file = _build(ck2)
    assert m_file.hacc_active and m_file.hacc_gate_active == (v is not None)
    rep = {"out": out_path, "sha256": sha256_file(out_path), "bytes": os.path.getsize(out_path), "arm": arm, "seed": seed, "gate_v_star": v, "carried_bit_for_bit": True, "new_keys": sorted(new)}
    if verify_batch is not None:
        infer = lambda m: {k: t for k, t in m.forward_infer(verify_batch).items() if torch.is_tensor(t)}
        with torch.no_grad():
            y_file, y_rt, y_src = infer(m_file), infer(m_rt), infer(m_src)
        assert set(y_file) == set(y_rt) and all(torch.equal(y_file[k], y_rt[k]) for k in y_file), "the baked file's forward_infer differs from the runtime-set model"
        rep["baked_vs_runtime"] = "bit-identical on every output tensor"
        rep["max_abs_baked_vs_plain_m"] = float((y_file["traj30"] - y_src["traj30"]).abs().max())
        rep["n_fired"] = int(y_file["hacc_gate_fired"].sum()) if "hacc_gate_fired" in y_file else None
        rep["n_verify_windows"] = int(y_file["traj30"].shape[0])
    if arm == "CA1":
        hk = torch.load(HACC_CKPT[seed], map_location="cpu", weights_only=False)
        if sha256_file(HACC_CKPT[seed]) != SHA["hacc"][seed]:
            raise ValueError("HACC's shipped checkpoint is not the pinned file")
        assert set(hk["model"]) == set(ck2["model"]) and all(torch.equal(hk["model"][k], ck2["model"][k]) for k in hk["model"]), "CA1 is not HACC's shipped checkpoint tensor for tensor"
        rep["equals_hacc_shipped_checkpoint_tensor_for_tensor"] = True
    return rep


def verify_batch_from_train(cfg_dict, n_clips: int = 6, per_clip: int = 8):
    """A real batch of train windows (the first ``n_clips`` fit clips, ``per_clip`` spread windows each) for the bake's bit-identity check."""
    import torch
    from tanitad.data import v8f_mini_data as D
    from tanitad.models import v8f_mini as M
    cfg = M.V8FMiniConfig(**cfg_dict)
    tracks = D.load_native_tracks(TRACKS)
    J = D.load_join(TRAIN_JOIN)
    store = D.ClipStore(TRAIN_CACHE, "train", lru=4, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    wins = D.train_windows(store, J, json.load(open(FIT_CLIPS, encoding="utf-8"))["clips"][:n_clips], hist=cfg.hist, start=cfg.window_start)
    ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    by = {}
    for ci, t, _, _ in wins:
        by.setdefault(ci, []).append(t)
    items = []
    for ci, ts in by.items():
        for t in np.linspace(0, len(ts) - 1, per_clip).round().astype(int):
            items.append(ds.item(ci, ts[t], 1, 3))
    full = D.collate(items)
    import v8f_t1_eval as T1
    return {k: full[k] for k in T1.MODEL_KEYS}


def cmd_bake(a) -> int:
    seed = SEEDS[a.seed]
    ck = None
    import torch
    ck = torch.load(PLAIN_CKPT[seed], map_location="cpu", weights_only=False)
    vb = verify_batch_from_train(ck["cfg"])
    out = f"{a.out_dir}/hlat{a.arm}_s{a.seed}.pt"
    rep = bake(a.arm, seed, out, fit_dir=a.fit_dir, verify_batch=vb)
    jdump(rep, f"{a.out_dir}/BAKE_{a.arm}_s{a.seed}.json")
    print("[bake]", json.dumps({k: v for k, v in rep.items() if k != "out"}), flush=True)
    return 0


# ================================================================================================ the controls on dumps
def cmd_compare_dumps(a) -> int:
    """C-A1 / C-B1: every ``ep*.npz`` of dump dir A equals dump dir B's on cl / ol / fl (and the ids) to <= tol."""
    fa = sorted(f for f in os.listdir(a.a) if f.startswith("ep") and f.endswith(".npz"))
    fb = sorted(f for f in os.listdir(a.b) if f.startswith("ep") and f.endswith(".npz"))
    res = {"label": a.label, "a": a.a, "b": a.b, "tol_m": a.tol, "n_files_a": len(fa), "n_files_b": len(fb), "same_files": fa == fb, "max_abs": {}, "ids_equal": True}
    if fa == fb and fa:
        for k in ("cl", "ol", "fl"):
            res["max_abs"][k] = 0.0
        for f in fa:
            za, zb = np.load(os.path.join(a.a, f)), np.load(os.path.join(a.b, f))
            for k in ("cl", "ol", "fl"):
                res["max_abs"][k] = max(res["max_abs"][k], float(np.abs(za[k] - zb[k]).max()))
            res["ids_equal"] &= all(np.array_equal(za[k], zb[k]) for k in ("ws", "eid", "clip_index", "g", "v0"))
    res["max_abs_all"] = max(res["max_abs"].values()) if res["max_abs"] else None
    res["pass"] = bool(res["same_files"] and res["ids_equal"] and res["max_abs_all"] is not None and res["max_abs_all"] <= a.tol)
    jdump(res, a.out)
    print(f"[{a.label}] files {len(fa)}/{len(fb)} max|d| {res['max_abs_all']}  ids_equal {res['ids_equal']}  PASS {res['pass']}", flush=True)
    return 0 if res["pass"] else 2


def cmd_check_ct(a) -> int:
    """C-T: every dump array and every JSON of run dir B equals run dir A's (the original tool's) bit for bit."""
    res = {"label": "C-T", "a": a.a, "b": a.b, "files": {}, "pass": True}
    for root_a, _, files in os.walk(a.a):
        for f in sorted(files):
            pa = os.path.join(root_a, f)
            pb = os.path.join(a.b, os.path.relpath(pa, a.a))
            if not os.path.exists(pb):
                res["files"][os.path.relpath(pa, a.a)] = "MISSING in b"
                res["pass"] = False
            elif f.endswith(".npz"):
                za, zb = np.load(pa), np.load(pb)
                same = set(za.files) == set(zb.files) and all(np.array_equal(za[k], zb[k], equal_nan=True) and za[k].dtype == zb[k].dtype for k in za.files)
                res["files"][os.path.relpath(pa, a.a)] = "bit-identical" if same else "DIFFERS"
                res["pass"] &= same
            elif f.endswith(".json"):
                same = json.load(open(pa)) == json.load(open(pb))
                res["files"][os.path.relpath(pa, a.a)] = "identical" if same else "DIFFERS"
                res["pass"] &= same
    res["n_files"] = len(res["files"])
    res["n_extra_in_b"] = sum(len(fs) for _, _, fs in os.walk(a.b)) - res["n_files"]
    res["pass"] &= res["n_files"] > 0 and res["n_extra_in_b"] == 0
    jdump(res, a.out)
    print(f"[C-T] files {res['n_files']}  extra in b {res['n_extra_in_b']}  PASS {res['pass']}", flush=True)
    return 0 if res["pass"] else 2


def cmd_check_b2(a) -> int:
    """C-B2: with v* = +inf, every tick's cross-axis plan equals P's cross plan to <= 1e-6 m (every tick of every window, gt_ok or not); and the ALONG axis is NOT P's (the gate acts on one axis only)."""
    z = np.load(a.ticks, allow_pickle=False)
    cross = float(np.abs(z["plan"][:, :, GATE_AXIS] - z["P"][:, :, GATE_AXIS]).max())
    along = float(np.abs(z["plan"][:, :, 1 - GATE_AXIS] - z["P"][:, :, 1 - GATE_AXIS]).max())
    res = {"label": "C-B2", "ticks": a.ticks, "n_samples": int(len(z["plan"])), "split": str(z["split"]), "n_fired": int(z["fired"].sum()), "cross_max_abs_m": cross, "along_max_abs_m_power": along,
           "pass": bool(cross <= 1e-6 and along > 1e-3 and z["fired"].all())}
    jdump(res, a.out)
    print(f"[C-B2] n {res['n_samples']} fired {res['n_fired']} cross max|plan-P| {cross:.3g} (bar 1e-6)  along {along:.3g} (> 1e-3: the gate is one axis)  PASS {res['pass']}", flush=True)
    return 0 if res["pass"] else 2


def cmd_check_t2(a) -> int:
    """C-T2: for the first ``n_chunks`` chunks of the run, the tick-0 samples equal the OPEN-LOOP batch's tau (traj30_o) and P (p10) at tick-0 ages (0, 0), the same chunk composition; v0 = p10f[-1][0]; GT = the store's."""
    import torch
    import v8f_t1_eval as T1
    from tanitad.data import v8f_mini_data as D
    seed = SEEDS[a.seed]
    z = np.load(a.ticks, allow_pickle=False)
    ck = torch.load(a.ckpt or HACC_CKPT[seed], map_location="cpu", weights_only=False)
    model = _build(ck)
    cfg = model.cfg
    tracks = D.load_native_tracks(TRACKS)
    J = D.load_join(TRAIN_JOIN)
    store = D.ClipStore(TRAIN_CACHE, "train", lru=8, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
    wins = D.train_windows(store, J, json.load(open(FIT_CLIPS, encoding="utf-8"))["clips"], hist=cfg.hist, start=cfg.window_start)
    if a.max_windows and a.max_windows < len(wins):
        wins = [wins[i] for i in np.linspace(0, len(wins) - 1, a.max_windows).round().astype(int)]
    assert len(wins) == int(z["n_windows"]), (len(wins), int(z["n_windows"]))
    ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    t0m = (z["tick"] == 0)
    worst = {"tau": 0.0, "P": 0.0, "v0": 0.0, "gt_float32_bits_equal": True, "ids": True}
    n = 0
    for b0 in range(0, min(len(wins), a.n_chunks * a.batch), a.batch):
        chunk = wins[b0:b0 + a.batch]
        full = D.collate([ds.item(ci, t, 0, 0) for ci, t, _, _ in chunk])
        batch = {k: full[k] for k in T1.MODEL_KEYS}
        with torch.no_grad():
            out = model.forward_infer(batch)
        tau, P = out.get("traj30_o", out["traj30"]).double().numpy(), out["p10"].double().numpy()
        v0 = batch["p10f"][:, -1, 0].double().numpy()
        idx = np.where(t0m & (z["win"] >= b0) & (z["win"] < b0 + len(chunk)))[0]
        idx = idx[np.argsort(z["win"][idx])]
        assert len(idx) == len(chunk), (len(idx), len(chunk))
        worst["tau"], worst["P"] = max(worst["tau"], float(np.abs(z["tau"][idx] - tau).max())), max(worst["P"], float(np.abs(z["P"][idx] - P).max()))
        worst["v0"] = max(worst["v0"], float(np.abs(z["v0"][idx] - v0).max()))
        gt_store = full["gt"].numpy()
        worst["gt_float32_bits_equal"] &= bool(np.array_equal(z["gt"][idx].astype(np.float32), gt_store)) and bool(z["gt_ok"][idx].all())
        worst["ids"] &= bool((z["ci"][idx] == np.array([w[0] for w in chunk])).all() and (z["t0"][idx] == np.array([w[1] for w in chunk])).all() and (z["j"][idx] == z["t0"][idx]).all())
        n += len(chunk)
    res = {"label": "C-T2", "ticks": a.ticks, "ckpt": a.ckpt or HACC_CKPT[seed], "n_windows_checked": n, "ages": [0, 0], "max_abs_m": worst, "bar_m": a.tol,
           "pass": bool(worst["tau"] <= a.tol and worst["P"] <= a.tol and worst["v0"] <= a.tol and worst["gt_float32_bits_equal"] and worst["ids"])}
    jdump(res, a.out)
    print(f"[C-T2] windows {n}  tau {worst['tau']:.3g}  P {worst['P']:.3g}  v0 {worst['v0']:.3g}  gt bits {worst['gt_float32_bits_equal']}  ids {worst['ids']}  PASS {res['pass']}", flush=True)
    return 0 if res["pass"] else 2


# ================================================================================================ CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--seed", type=int, choices=(0, 1), required=True)
    p.add_argument("--ticks", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--expect-windows", type=int, default=N_TRAIN_WINDOWS, help="the number of train windows the tick file must hold (a DRY RUN on a subset passes its own; the chain never does)")
    p = sub.add_parser("bake")
    p.add_argument("--seed", type=int, choices=(0, 1), required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fit-dir", default=None)
    p = sub.add_parser("compare-dumps")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--out", required=True)
    p = sub.add_parser("check-ct")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("check-b2")
    p.add_argument("--ticks", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("check-t2")
    p.add_argument("--seed", type=int, choices=(0, 1), required=True)
    p.add_argument("--ticks", required=True)
    p.add_argument("--ckpt", default="")
    p.add_argument("--n-chunks", type=int, default=8)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--out", required=True)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    return {"fit": cmd_fit, "bake": cmd_bake, "compare-dumps": cmd_compare_dumps, "check-ct": cmd_check_ct, "check-b2": cmd_check_b2, "check-t2": cmd_check_t2}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
