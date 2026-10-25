#!/usr/bin/env python3
"""v8f_hacc.py -- executes PREREG_V8F_HACC.md exactly as written (dev box, CPU only, ZERO training, offline, from banked outputs).

  python v8f_hacc.py auto [--n-boot 2000] [--out DIR]     fit (train only) -> controls -> score both seeds -> DK -> READ.json -> RESULT_V8F_HACC.md
  python v8f_hacc.py fit   --seed M2b_s0                  Phase A, TRAIN arrays only: controls 1, 2, 4, 5, lambda0, the grid, lambda*, w
  python v8f_hacc.py score --seed M2b_s0                  Phase B: eval controls 3, 5, 6, 7, then every PREREG row
  python v8f_hacc.py dk | read | x1

THE ARM (PREREG, fixed).  Per seed s and axis a (0 along / 1 cross), r = tau - P, d = y - P, H_w = P + w (.) r:
    w_{s,a} = argmin_{w in [0,1]^30}  sum_n [ sum_t (w_t r_nt - d_nt)^2  +  lambda_s sum_{k=1..29} (D2[0;H_w]_nk - D2[0;y]_nk)^2 ]
  D2[0;p]_k = p_(k+1) - 2 p_k + p_(k-1) with the ORIGIN prepended (p_0 = the ego at NOW).  Fit set = HBLEND's 9,582 TRAIN windows.  NO isotonic step.
  lambda rule (TRAIN only): lambda0 = S1/S2 at HBLEND's banked w; grid lambda = m lambda0, m in {0,1,4,16,64,256}; lambda* = the smallest grid lambda at which, on the
  TRAIN windows, HACC's accel MAE and curvature MAE are both <= 0.98 x P10's AND ADE2, ADE6 are both < P10's; else the grid lambda with the lowest train accel MAE among
  those with train ADE2 < P10's; else 0.

DECLARED INTERPRETATIONS (each a choice the PREREG text leaves open; ALL fixed BEFORE any HACC number was computed -- the only quantities looked at beforehand were the
two machinery ties of controls 1 and 2 on the TRAIN windows, which carry no HACC outcome: lambda=0 vs the clipped closed form 4e-14 / 5e-14, + isotonic vs the banked w 3e-14 / 4e-14):
  I1  Axes: w[:, a], a = 0 along (x forward), a = 1 cross (y left): the axis order of tau/P/y [N, 30, 2]. Only the 30 knots at 0.2 s are fitted and scored (HBLEND's I1).
  I2  ONE lambda per seed, shared by both axes; lambda0 = S1/S2 POOLED over windows, steps and axes at HBLEND's banked w (the PREREG's lambda_s carries only the seed).
  I3  The D2 term: e = H_w - y (= w r - d) with e_0 = 0 (both paths start at the ego), so D2 has 29 rows per axis -- the same 29 cells as the accel family.  D2[0;H_w] - D2[0;y]
      is linear in e, hence D2 e; the literal np.diff-with-prepended-origin form is the tests' oracle.
  I4  The solver: `scipy.optimize.lsq_linear(method='bvls', bounds=(0,1), tol=1e-12, max_iter=500)` on the LITERAL stacked system (position rows + sqrt(lambda) D2 rows;
      565,338 x 30 at lambda>0; at lambda = 0 the D2 rows carry zero weight and are omitted).  KKT: relative projected gradient of 0.5||A w - b||^2 on the same stacked system,
      max|pg| / max|A^T b| (HSPEC's definition), bar 1e-6; a fit that fails it RAISES (KKTError).
  I5  The train metrics of the lambda rule: per-window `H.arm_pw` (the eval's own family functions, BE.family_windows) on the TRAIN arrays; an arm's MAE is the mean over the
      windows where BOTH the arm and P10 are finite (paired convention; curvature MAE is NaN on windows with no moving step pair).  ADE2/ADE6 = mean of the first 10 / all 30 steps.
      "P10" on train = the checkpoint's own frozen prior P = fit_windows p30.  The comparisons are on POINT means (train, in-sample): "<=" 0.98x for accel/curvature, "<" for ADE.
  I6  Outcome row 3 "NO SMOOTHING" = HACC - H accel MAE is not separated-negative at EITHER seed (i.e. at neither); MIXED = anything else.  Row order as in the PREREG.
  I7  "separated-better / separated-worse" = `taniteval.v8f_beta_eval.paired_row`'s flags; distance keeping: worse = headway or time-gap separated NEGATIVE, LEAD-VIOL at-risk
      separated POSITIVE; "separated-negative" of HACC - H accel/curvature = `better` (lower is better).  HSPEC's conventions, via its `dk_flags`, DOM_BETTER, DOM_NOT_WORSE.
  I8  PRIMARY-KEEP bar = F/2 with the exact F = |ADE6(s0) - ADE6(s1)| = 0.12074 m of the banked npz, i.e. +0.06037 m (the printed +0.0604 variant is reported beside).
  I9  Control thresholds: c1, c2: 1e-12 abs;  c3: plan 1e-4 m, per-window ADE2/ADE6/FDE6 1e-4 m, w0 ADE6 mean 2.4870 +- 1e-3;  c4: 1e-6 at EVERY fit (the lambda=0 fit, the six grid
      fits, both deliberate-regression fits, the eval-ceiling fit);  c5: max w <= 1e-12 and ADE6 diff <= 1e-4 m (regression, at every grid lambda and at lambda*); permuted-r mean w <= 0.10
      (at lambda*, rng default_rng(0), the SAME permutation of whole windows on both axes; tau_perm = P + (tau - P)[perm]);  c6: runtime refusal of an eval world;  c7: 2e-6.
  I10 Added machinery tie X1 (does not touch a bar): my dominance reader reproduces HSPEC's banked PRIMARY-DOM verdicts and every per-row better/worse flag from HSPEC's own
      banked score and DK files (both stages, both seeds).  A failure is rc 2 like any control.
  I11 Tactical / strategic: HACC post-maps the plan only; T's decision heads and S's checkpoint outputs are the banked B arrays, so HACC - H is identical to 0 by construction
      (the structural check re-runs the eval's own evaluate() on the HACC plan and requires its tactical block to equal the banked B's).  `v8f_paired_families.py` reads eval `run`
      dumps and HACC produces none through `evaluate()`, so it is not run; the plan-derived tactical agreements are in the dominance row.  STRATEGIC is NOT paused for v8F.
Everything else (the eval's evaluate(), families, DK reader, paired bootstrap, loaders, w, fit windows) is HBLEND's / HSPEC's, imported read-only, not re-implemented.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np  # noqa: E402
from scipy.optimize import lsq_linear  # noqa: E402

SCR = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"
HB, HS, OUT = f"{SCR}/hblend", f"{SCR}/hspec", f"{SCR}/hacc"
for _p in (HS, HB):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import v8f_hblend as H  # noqa: E402  (read-only reuse: loaders, apply_w, arm_pw, rows_for, single, jdump, pav, fit_weights)
import v8f_hspec as S  # noqa: E402  (read-only reuse: load_world, cons_viol_arrays, dk_flags, DOM_BETTER, DOM_NOT_WORSE)


def _local(name: str, modname: str):
    """Import HACC's OWN file by path under a unique module name: hspec/ and hblend/ carry same-named files (accel_profile.py, build_result.py) that sys.path would find first."""
    import importlib.util
    sp = importlib.util.spec_from_file_location(modname, f"{OUT}/{name}.py")
    m = importlib.util.module_from_spec(sp)
    sys.modules[modname] = m
    sp.loader.exec_module(m)
    return m


AP = _local("accel_profile", "hacc_accel_profile")      # HACC's copy of the instrument; the original (hspec/accel_profile.py) is untouched

SEEDS, B_BOOT, DT = H.SEEDS, H.B_BOOT, 0.2
NSTEP, NAX = 30, 2
GRID_M = (0, 1, 4, 16, 64, 256)
RULE_RATIO = 0.98
KKT_TOL = 1e-6
TIER = "T1-class open loop (own held outputs; no recorded future action)"
TIER_CEILING = "T0 diagnostic (the fit reads the EVAL GT): never a result"
N_TRAIN = 9582
# the TRAIN inputs (HBLEND's banked fit windows and weights) are pinned by content: a file that is not these bytes is not the training set
TRAIN_SHA = {"M2b_s0": {"fit": "39179937bfef122038c395a0c6477dc56893a33e08b9c8c57cb3f3555e9095b3", "w": "dd260b4eaf114454bdc2f0a55c9eb601fd62de4e4c13ec0dad590b4b30eeff9d"},
             "M2b_s1": {"fit": "d6ae1421bb0d7b403cd2c5bacb44bf12bf33b188d84dc683764cc66fd371137f", "w": "987a0b5f07393e399d1482f96ca9a01c338386d952594776edb3426cc6d65227"}}
KEY = {"ade2": "ade2", "ade6": "ade6", "accel": "fw_lon_accel_mae", "curv": "fw_lat_curvature_mae_1pm"}
NEXT = {
    "DOMINATES P10": ("HACC becomes the deployed post-map: 30 x 2 buffers per model, zero training, folded into `forward_infer` under its own landing with tests. The evidence videos "
                      "are re-rendered on HACC. The T1-kin closed-loop read on HACC (`D-V8F-T1KIN-1`) is the next tier and needs a PI go for the pooled eval cache"),
    "SMOOTHER, NOT DOMINANT": "OACC (box 2) runs as queued, and its HX read adds HACC's fit on OACC's checkpoint",
    "NO SMOOTHING": "the roughness is not a weighting artefact; OACC is the lever",
    "MIXED": "reported per seed; seeds before windows",
}


class KKTError(RuntimeError):
    """A fitted w is not the bounded optimum (relative projected gradient above the bar)."""


# ------------------------------------------------------------------------------------------------ pure arithmetic (unit-tested)
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


def isotonic(w: np.ndarray) -> np.ndarray:
    """HBLEND's pool-adjacent-violators, per axis (control 1 only; the arm has NO isotonic step)."""
    return np.stack([H.pav_nondecreasing(w[:, a]) for a in range(w.shape[1])], 1)


def terms_at(tau, P, y, w, D2: np.ndarray | None = None):
    """(S1, S2): the position and the D2 terms of the objective at w, summed over windows, steps and BOTH axes (lambda = 1)."""
    D2 = d2_matrix() if D2 is None else D2
    e = np.asarray(w, np.float64)[None] * (np.asarray(tau, np.float64) - np.asarray(P, np.float64)) - (np.asarray(y, np.float64) - np.asarray(P, np.float64))
    return float((e ** 2).sum()), float((np.einsum("kt,nta->nka", D2, e) ** 2).sum())


# ------------------------------------------------------------------------------------------------ the train-only bundle and the lambda rule
@dataclass(frozen=True)
class TrainBundle:
    """The TRAIN arrays and nothing else.  The lambda rule accepts this type alone; an eval array cannot be wrapped in it (split must be 'train')."""
    tau: np.ndarray
    P: np.ndarray
    y: np.ndarray
    banked_w: np.ndarray
    split: str = "train"
    source: str = ""

    def __post_init__(self):
        if self.split != "train":
            raise ValueError(f"TrainBundle holds TRAIN arrays only (split={self.split!r})")
        if not (self.tau.shape == self.P.shape == self.y.shape and self.tau.ndim == 3 and self.tau.shape[1:] == (NSTEP, NAX)):
            raise ValueError(f"bad train shapes {self.tau.shape} {self.P.shape} {self.y.shape}")
        if self.banked_w.shape != (NSTEP, NAX):
            raise ValueError(f"banked_w must be [30, 2], got {self.banked_w.shape}")


def _require_train(T) -> None:
    if not isinstance(T, TrainBundle):
        raise TypeError(f"the lambda selection receives a TrainBundle (training arrays) alone, got {type(T).__name__}")
    if T.split != "train":
        raise ValueError(f"the lambda selection refuses a non-train bundle (split={T.split!r})")


def train_bundle(seed: str) -> TrainBundle:
    fp, wp = f"{HB}/fit_windows_{seed}.npz", f"{HB}/weights_{seed}.json"
    if H.sha256_file(fp) != TRAIN_SHA[seed]["fit"] or H.sha256_file(wp) != TRAIN_SHA[seed]["w"]:
        raise ValueError(f"{seed}: the fit-window / weight files are not the pinned TRAIN inputs")
    fw = np.load(fp)
    assert fw["gt_ok"].all() and len(fw["traj30"]) == N_TRAIN
    Wb = json.load(open(wp))
    bw = np.stack([np.asarray(Wb["w_along_by_step"], np.float64), np.asarray(Wb["w_cross_by_step"], np.float64)], 1)
    tau, P, y = (fw[k].astype(np.float64) for k in ("traj30", "p30", "gt"))
    return TrainBundle(tau=tau, P=P, y=y, banked_w=bw, split="train", source=fp)


def lambda0(T: TrainBundle) -> float:
    """lambda0 = (position term) / (D2 term), both evaluated at HBLEND's banked w on the fit set (pooled over axes)."""
    _require_train(T)
    s1, s2 = terms_at(T.tau, T.P, T.y, T.banked_w)
    return s1 / s2


def _be():
    from taniteval import v8f_beta_eval as BE
    return BE


def default_metrics(plan, y):
    return H.arm_pw(plan, y, _be())


def paired_means(pw_a: dict, pw_p: dict) -> dict:
    """{ade2, ade6, accel, curv, p_ade2, ...}: each pair of means over the windows where BOTH arms are finite (I5)."""
    out = {}
    for nm, k in KEY.items():
        a, p = np.asarray(pw_a[k], np.float64), np.asarray(pw_p[k], np.float64)
        ok = np.isfinite(a) & np.isfinite(p)
        out[nm], out["p_" + nm] = float(a[ok].mean()), float(p[ok].mean())
    return out


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


def select_lambda(T: TrainBundle, metrics_fn=None) -> dict:
    """lambda0, the grid fits, the train metrics and lambda*.  Its inputs are TRAIN arrays alone (control 6)."""
    _require_train(T)
    metrics_fn = metrics_fn or default_metrics
    D2 = d2_matrix()
    lam0 = lambda0(T)
    pwP = metrics_fn(T.P, T.y)
    rows = []
    for m in GRID_M:
        lam = float(m * lam0)
        f = fit_w(T.tau, T.P, T.y, lam, D2)
        plan = H.apply_w(T.P, T.tau, f["w"])
        row = {"m": m, "lam": lam, **paired_means(metrics_fn(plan, T.y), pwP), "w": f["w"], "kkt": f["kkt"], "nit": f["nit"], "status": f["status"]}
        row["qualifies"] = qualifies(row)
        rows.append(row)
    star, branch = choose_lambda(rows)
    return {"lambda0": lam0, "grid": rows, "m_star": star["m"], "lam_star": star["lam"], "w_star": star["w"], "rule_branch": branch}


# ------------------------------------------------------------------------------------------------ Phase A: the TRAIN-only fit, controls 1, 2, 4, 5
def fit_seed(seed: str, out: str = OUT) -> dict:
    t0 = time.time()
    T = train_bundle(seed)
    D2 = d2_matrix()
    # control 2: lambda = 0 without the isotonic step equals the clipped closed form;  control 1: + HBLEND's isotonic step equals HBLEND's banked w
    f0 = fit_w(T.tau, T.P, T.y, 0.0, D2)
    hb = H.fit_weights(T.tau, T.P, T.y)
    tc = {"c2_lam0_vs_clipped_closed_form_maxabs": float(np.abs(f0["w"] - hb["clipped"]).max()),
          "c1_lam0_plus_isotonic_vs_banked_w_maxabs": float(np.abs(isotonic(f0["w"]) - T.banked_w).max()),
          "c1_hblend_refit_vs_banked_w_maxabs": float(np.abs(hb["w"] - T.banked_w).max())}
    sel = select_lambda(T)
    lam_star = sel["lam_star"]
    # control 5: the deliberate regressions (at lambda* and at every grid lambda)
    reg, perm_means = {}, {}
    perm = np.random.default_rng(0).permutation(len(T.tau))
    tau_perm = T.P + (T.tau - T.P)[perm]
    all_kkt = [max(f0["kkt"])] + [max(g["kkt"]) for g in sel["grid"]]
    for g in sel["grid"]:
        fr = fit_w(T.tau, T.P, T.P, g["lam"], D2)                                  # GT := P10
        reg[g["m"]] = float(np.abs(fr["w"]).max())
        fp = fit_w(tau_perm, T.P, T.y, g["lam"], D2)                               # r permuted across training windows
        perm_means[g["m"]] = float(fp["w"].mean())
        all_kkt += [max(fr["kkt"]), max(fp["kkt"])]
        if g["m"] == sel["m_star"]:
            tc["c5_regression_w_star"] = fr["w"]
            tc["c5_permuted_w_star"] = fp["w"]
    tc.update({"c5_regression_w_maxabs_by_m": {str(k): v for k, v in reg.items()}, "c5_regression_w_maxabs_at_lam_star": reg[sel["m_star"]],
               "c5_permuted_r_mean_w_by_m": {str(k): v for k, v in perm_means.items()}, "c5_permuted_r_mean_w_at_lam_star": perm_means[sel["m_star"]],
               "c4_kkt_max_over_all_train_fits": float(max(all_kkt))})
    tc["pass"] = {"c1_isotonic_tie": bool(tc["c1_lam0_plus_isotonic_vs_banked_w_maxabs"] <= 1e-12),
                  "c2_clipped_tie": bool(tc["c2_lam0_vs_clipped_closed_form_maxabs"] <= 1e-12),
                  "c4_kkt": bool(tc["c4_kkt_max_over_all_train_fits"] <= KKT_TOL),
                  "c5_regression_w_zero": bool(max(reg.values()) <= 1e-12),
                  "c5_permuted_mean_w": bool(tc["c5_permuted_r_mean_w_at_lam_star"] <= 0.10)}
    tc["all_pass"] = bool(all(tc["pass"].values()))
    rec = {"seed": seed, "tier": TIER, "n_windows": int(len(T.tau)), "fit_windows_sha256": TRAIN_SHA[seed]["fit"], "banked_w_sha256": TRAIN_SHA[seed]["w"],
           "lambda0": sel["lambda0"], "grid_m": list(GRID_M), "grid": sel["grid"], "m_star": sel["m_star"], "lam_star": lam_star, "w_star": sel["w_star"],
           "rule_branch": sel["rule_branch"], "rule": "PREREG: smallest grid lambda with train accel & curvature <= 0.98 x P10 and train ADE2 & ADE6 < P10; fallbacks as registered",
           "train_controls": tc, "w_lam0_unconstrained": f0["w"], "banked_w": T.banked_w, "seconds": time.time() - t0}
    H.jdump(rec, f"{out}/fit_{seed}.json")
    g = next(r for r in sel["grid"] if r["m"] == sel["m_star"])
    print(f"[fit {seed}] {time.time() - t0:.0f}s  lambda0 {sel['lambda0']:.5g}  m* {sel['m_star']}  lambda* {lam_star:.5g}  ({sel['rule_branch'][:40]})  "
          f"train accel {g['accel']:.4f} vs P10 {g['p_accel']:.4f}  curv {g['curv']:.5f} vs {g['p_curv']:.5f}  ADE2 {g['ade2']:.4f} vs {g['p_ade2']:.4f}  ADE6 {g['ade6']:.4f} vs {g['p_ade6']:.4f}  "
          f"w mean {sel['w_star'].mean():.3f}  controls {json.dumps(tc['pass'])}", flush=True)
    return rec


# ------------------------------------------------------------------------------------------------ the dominance reader (HSPEC's conventions)
def dominance(paired_p10: dict, viol_any_row: dict, dk_pair: dict) -> dict:
    """PRIMARY-DOM for ONE seed from the three paired reads of (arm - P10): the family/ADE rows (H.rows_for), CONSISTENCY(a) 'any', and the DK pair read."""
    rows, ok_better, ok_nw = {}, True, True
    for nm, key in S.DOM_BETTER.items():
        r = paired_p10[key]
        rows[nm] = {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "better": r["better"], "worse": r["worse"], "need": "separated-better"}
        ok_better &= bool(r["better"])
    for nm, key in S.DOM_NOT_WORSE.items():
        r = paired_p10[key]
        rows[nm] = {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "better": r["better"], "worse": r["worse"], "need": "not separated-worse"}
        ok_nw &= not bool(r["worse"])
    r = viol_any_row
    rows["CONSISTENCY(a) 'any'"] = {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "better": r["better"], "worse": r["worse"], "need": "not separated-worse"}
    ok_nw &= not bool(r["worse"])
    for nm, f in S.dk_flags(dk_pair).items():
        rows[f"DK {nm}"] = {"mean": f["delta"], "lo": f["lo"], "hi": f["hi"], "separated": f["separated"], "better": f["better"], "worse": f["worse"],
                            "need": "not separated-worse", "two_sided_sensitivity_separated": f["worse_two_sided_sensitivity"]}
        ok_nw &= not f["worse"]
    return {"pass": bool(ok_better and ok_nw), "all_better_ok": bool(ok_better), "all_not_worse_ok": bool(ok_nw), "rows": rows}


def keep_read(ade6_vs_h: dict, F_exact: float) -> dict:
    """PRIMARY-KEEP: ADE6 (arm - H) upper CI bound <= +F/2."""
    return {"delta": ade6_vs_h["mean"], "lo": ade6_vs_h["lo"], "hi": ade6_vs_h["hi"], "bar_hi_le": F_exact / 2, "pass": bool(ade6_vs_h["hi"] <= F_exact / 2),
            "hi_le_prereg_printed_0.0604": bool(ade6_vs_h["hi"] <= 0.0604)}


def smooth_flags(paired_vs_h: dict) -> dict:
    a, c = paired_vs_h["fw_lon_accel_mae"], paired_vs_h["fw_lat_curvature_mae_1pm"]
    return {"accel": {k: a[k] for k in ("mean", "lo", "hi", "separated")}, "curvature": {k: c[k] for k in ("mean", "lo", "hi", "separated")},
            "accel_sep_neg": bool(a["better"]), "curv_sep_neg": bool(c["better"])}


def outcome_of(per_seed: dict) -> str:
    """The PREREG outcome table, in its order.  per_seed[s] = {'dom': bool, 'keep': bool, 'accel_sep_neg': bool, 'curv_sep_neg': bool}."""
    seeds = list(per_seed)
    if all(per_seed[s]["dom"] and per_seed[s]["keep"] for s in seeds):
        return "DOMINATES P10"
    if all(per_seed[s]["accel_sep_neg"] and per_seed[s]["curv_sep_neg"] for s in seeds):
        return "SMOOTHER, NOT DOMINANT"
    if not any(per_seed[s]["accel_sep_neg"] for s in seeds):
        return "NO SMOOTHING"
    return "MIXED"


def x1_hspec_tie() -> dict:
    """Machinery tie (I10): `dominance` / `keep_read` reproduce HSPEC's banked verdicts and every per-row flag from HSPEC's own banked files (JSON only)."""
    R = json.load(open(f"{HS}/READ.json"))
    out, bad = {"checked": 0, "mismatches": []}, []
    for st in ("1", "2"):
        dkj = json.load(open(f"{HS}/dk_read_hs{st}.json"))
        for s in SEEDS:
            k = s.split("_")[1]
            sc = json.load(open(f"{HS}/score_hs{st}_{s}.json"))
            mine = dominance(sc["paired"]["HS_minus_P10"], sc["consistency"]["paired"]["HS_minus_P10"]["viol_any"], dkj["pairs"][f"HS_{k}__minus__P10"])
            ref = R["stages"][st]["per_seed"][s]
            kp = keep_read(sc["paired"]["HS_minus_H"]["ade6"], R["F_exact_m"])
            if mine["pass"] != ref["PRIMARY_DOM"]["pass"]:
                bad.append(f"stage {st} {s}: PRIMARY-DOM {mine['pass']} vs banked {ref['PRIMARY_DOM']['pass']}")
            if kp["pass"] != ref["PRIMARY_KEEP"]["pass"]:
                bad.append(f"stage {st} {s}: PRIMARY-KEEP {kp['pass']} vs banked {ref['PRIMARY_KEEP']['pass']}")
            for nm, row in ref["PRIMARY_DOM"]["rows"].items():
                out["checked"] += 1
                for f in ("better", "worse", "separated"):
                    if mine["rows"][nm][f] != row[f]:
                        bad.append(f"stage {st} {s} {nm}: {f} {mine['rows'][nm][f]} vs banked {row[f]}")
    out["mismatches"] = bad
    out["pass"] = bool(not bad and out["checked"] > 0)
    return out


# ------------------------------------------------------------------------------------------------ Phase B: the eval controls and the read
def eval_controls(W: dict, F: dict) -> dict:
    tau, P, gt, z, fl, Pf = (W[k] for k in ("tau", "P", "gt", "z", "fl", "P_file"))
    tc = F["train_controls"]
    H1, H0 = H.apply_w(P, tau, 1.0), H.apply_w(P, tau, 0.0)
    d1, d0 = np.linalg.norm(H1 - gt, axis=-1), np.linalg.norm(H0 - gt, axis=-1)
    c = {"c3_w1_plan_vs_banked_pred_maxabs_m": float(np.abs(H1 - tau).max()), "c3_w1_ade2_perwindow_maxabs_vs_banked": float(np.abs(d1[:, :10].mean(1) - z["ade2"]).max()),
         "c3_w1_ade6_perwindow_maxabs_vs_banked": float(np.abs(d1.mean(1) - z["ade6"]).max()), "c3_w1_fde6_perwindow_maxabs_vs_banked": float(np.abs(d1[:, -1] - z["fde6"]).max()),
         "c3_w0_plan_vs_P10_file_maxabs_m": float(np.abs(H0 - Pf).max()), "c3_w0_ade6_mean": float(d0.mean()),
         "c3_w0_ade6_perwindow_maxabs_vs_floors_P10_refit_(report)": float(np.abs(d0.mean(1) - fl["P10_refit__ade6"]).max()),
         "P10_file_ade6_mean": float(np.linalg.norm(Pf - gt, axis=-1).mean())}
    # control 5 (eval half): the regression arm READS as P10
    plan_reg = H.apply_w(P, tau, np.asarray(tc["c5_regression_w_star"]))
    c["c5_regression_ade6_minus_P10_file_mean_m"] = float(np.linalg.norm(plan_reg - gt, axis=-1).mean() - np.linalg.norm(Pf - gt, axis=-1).mean())
    # control 6 (runtime): the selection refuses an eval world
    refused = {}
    for nm, obj in (("eval_world_dict", W), ("eval_tau_ndarray", tau)):
        try:
            select_lambda(obj)
            refused[nm] = False
        except (TypeError, ValueError):
            refused[nm] = True
    try:
        TrainBundle(tau=tau[:2], P=P[:2], y=gt[:2], banked_w=np.zeros((NSTEP, NAX)), split="eval")
        refused["eval_split_bundle"] = False
    except ValueError:
        refused["eval_split_bundle"] = True
    c["c6_eval_input_refused"] = refused
    # control 7: the accel instrument
    c["c7_accel_instrument_vs_banked_maxabs"] = {n: AP.control(arr, gt, W["PW"][n]["fw_lon_accel_mae"]) for n, arr in (("H", W["H"]), ("P10", W["P_file"]))}
    c["c7_accel_instrument_vs_arm_pw_(report)"] = {"B": AP.control(tau, gt, W["PW"]["B"]["fw_lon_accel_mae"])}
    c["pass"] = {"train_side_controls_1_2_4_5": bool(tc["all_pass"]),
                 "c3_w1_equals_B": bool(c["c3_w1_plan_vs_banked_pred_maxabs_m"] <= 1e-4 and c["c3_w1_ade2_perwindow_maxabs_vs_banked"] <= 1e-4
                                        and c["c3_w1_ade6_perwindow_maxabs_vs_banked"] <= 1e-4 and c["c3_w1_fde6_perwindow_maxabs_vs_banked"] <= 1e-4),
                 "c3_w0_equals_P10": bool(c["c3_w0_plan_vs_P10_file_maxabs_m"] <= 1e-4 and abs(c["c3_w0_ade6_mean"] - 2.4870) <= 1e-3),
                 "c5_regression_reads_as_P10": bool(abs(c["c5_regression_ade6_minus_P10_file_mean_m"]) <= 1e-4),
                 "c6_selection_train_only": bool(all(refused.values())),
                 "c7_accel_instrument": bool(max(c["c7_accel_instrument_vs_banked_maxabs"].values()) <= AP.CONTROL_BAR),
                 "world_P_and_H_reproduce_hblend": bool(W["P_vs_hblend_saved_maxabs"] == 0.0 and W["H_vs_recomputed_maxabs"] <= 1e-9)}
    c["all_pass"] = bool(all(c["pass"].values()))
    return c


def score_seed(seed: str, nb: int, out: str = OUT, W: dict | None = None) -> int:
    t0 = time.time()
    F = json.load(open(f"{out}/fit_{seed}.json"))
    x1 = json.load(open(f"{out}/control_X1.json")) if os.path.exists(f"{out}/control_X1.json") else x1_hspec_tie()
    W = W or S.load_world(seed)
    from taniteval import ci as CI
    BE, tau, P, gt, eid = (W[k] for k in ("BE", "tau", "P", "gt", "eid"))
    ctl = eval_controls(W, F)
    ctl["X1_hspec_reader_tie"] = {"pass": x1["pass"], "checked_rows": x1["checked"], "mismatches": x1["mismatches"][:10]}
    ctl["all_pass"] = bool(ctl["all_pass"] and x1["pass"])
    print(f"[score {seed}] CONTROLS {json.dumps(ctl['pass'])}  X1 {x1['pass']}  all_pass={ctl['all_pass']}", flush=True)
    if not ctl["all_pass"]:
        H.jdump({"seed": seed, "controls": ctl}, f"{out}/score_{seed}.json")
        print(f"[score {seed}] CONTROL FAILED -- rc 2, NO READ: {json.dumps(ctl, default=float)[:2500]}", flush=True)
        return 2
    w = np.asarray(F["w_star"], np.float64)
    plan = H.apply_w(P, tau, w)
    pwB, pwH, pwP = W["PW"]["B"], W["PW"]["H"], W["PW"]["P10"]
    pwA = H.arm_pw(plan, gt, BE)
    D2 = d2_matrix()
    o = {"seed": seed, "ckpt_sha256": H.SHA[seed], "n_windows": int(len(eid)), "n_episodes": int(len(set(eid.tolist()))), "tier": TIER,
         "estimator": f"paired episode-cluster bootstrap, B={nb}, 141 episodes, seed 0 (taniteval/ci.py). Answers 'another draw of EPISODES' ONLY; training-seed variance only via the two "
                      "seeds side by side and F = |ADE6(s0)-ADE6(s1)| (H-ESTIM-SEED-1); inference is deterministic (no inference-seed floor).",
         "controls": ctl, "w_star": w, "lam_star": F["lam_star"], "m_star": F["m_star"], "fit_json_sha256": H.sha256_file(f"{out}/fit_{seed}.json"),
         "w_note": "w[t, a]: t = step 0..29 (0.2 s knots), a = 0 along / 1 cross (I1)"}
    keys = list(pwB)
    o["single"] = {n: {k: H.single(pw[k], eid, CI, nb) for k in keys} for n, pw in (("B", pwB), ("H", pwH), ("HACC", pwA), ("P10", pwP))}
    o["paired"] = {"HACC_minus_P10": H.rows_for(pwA, pwP, eid, BE, nb), "HACC_minus_H": H.rows_for(pwA, pwH, eid, BE, nb), "HACC_minus_B": H.rows_for(pwA, pwB, eid, BE, nb),
                   "H_minus_P10": H.rows_for(pwH, pwP, eid, BE, nb)}
    res, viol = S.cons_viol_arrays(W, plan, nb)
    o["blocks"] = {"HACC": res}
    o["consistency"] = {"label": "CONSISTENCY(a) violation rate of the PLAN vs the GT-derived constraints (lower is better); (b) = violation of the model's OWN predicted constraints",
                        "rate": {("HACC" if n == "HS" else n): {k: float(vv[k].mean()) for k in vv} for n, vv in viol.items()},
                        "paired": {f"HACC_minus_{q}": {k: BE.paired_row(viol["HS"][k].astype(float), viol[q][k].astype(float), eid, nb) for k in viol["HS"] if k in viol[q]}
                                   for q in ("P10", "H", "B")}}
    strat = res.get("strategic")
    o["structural_identity"] = {
        "tactical_block_T_heads_HACC_equals_banked_B": bool(H.jround(res["tactical"]) == H.jround(W["bj"]["raw"]["tactical"])),
        "strategic_block_HACC_equals_banked_B": (bool(H.jround(strat) == H.jround(W["bj"]["raw"]["strategic"])) if strat is not None else None),
        "strategic": "ACTIVE for v8F (PI-V8F-2026-10-17; R72/R73), NOT paused. S's checkpoint outputs are the banked B arrays: HACC - H is identical by construction (the post-map touches the plan only).",
        "banked_B_strategic_block": W["bj"]["raw"]["strategic"],
        "paired_families_tool": "v8f_paired_families.py not run: it reads eval `run` dumps and HACC produces none through evaluate(); T's decisions and S are H's own (I11)."}
    o["F_exact_m"] = abs(float(np.load(f"{H.BANK}/eval_M2b_s0-kind-own.npz")["ade6"].mean()) - float(np.load(f"{H.BANK}/eval_M2b_s1-kind-own.npz")["ade6"].mean()))
    # CEILING (T0 diagnostic, never a result): the same objective at lambda*, fitted on the EVAL GT
    ce = fit_w(tau, P, gt, F["lam_star"], D2)
    plan_c = H.apply_w(P, tau, ce["w"])
    pwC = H.arm_pw(plan_c, gt, BE)
    ck = ("ade2", "ade6", "fw_lon_accel_mae", "fw_lat_curvature_mae_1pm", "fw_lat_yaw_rate_mae_degps", "fw_lat_heading_mae_deg")
    o["ceiling_T0"] = {"label": TIER_CEILING, "tier": TIER_CEILING, "w": ce["w"], "kkt": ce["kkt"], "single": {k: H.single(pwC[k], eid, CI, nb) for k in ck},
                       "vs_P10": H.rows_for({k: pwC[k] for k in ck}, {k: pwP[k] for k in ck}, eid, BE, nb), "vs_H": H.rows_for({k: pwC[k] for k in ck}, {k: pwH[k] for k in ck}, eid, BE, nb),
                       "vs_HACC": H.rows_for({k: pwC[k] for k in ck}, {k: pwA[k] for k in ck}, eid, BE, nb)}
    # BESIDE: the per-step accel profile of HACC / H / P10 / B
    o["accel_profile"] = AP.profile({"HACC": plan, "H": W["H"], "P10": W["P_file"], "B": tau}, gt)
    o["seconds"] = time.time() - t0
    H.jdump(o, f"{out}/score_{seed}.json")
    np.savez_compressed(f"{out}/score_{seed}_perwindow.npz", HACC=plan, eid=eid, **{f"HACC__{k}": v for k, v in pwA.items()},
                        **{f"viol_{('HACC' if n == 'HS' else n)}__{k}": vv[k] for n, vv in viol.items() for k in vv})
    p = o["paired"]["HACC_minus_P10"]
    print(f"[score {seed}] DONE {time.time() - t0:.0f}s  HACC-P10 ADE2 {p['ade2']['mean']:+.4f} [{p['ade2']['lo']:+.4f},{p['ade2']['hi']:+.4f}]  ADE6 {p['ade6']['mean']:+.4f} "
          f"[{p['ade6']['lo']:+.4f},{p['ade6']['hi']:+.4f}]  accel {p['fw_lon_accel_mae']['mean']:+.4f} [{p['fw_lon_accel_mae']['lo']:+.4f},{p['fw_lon_accel_mae']['hi']:+.4f}]  "
          f"curv {p['fw_lat_curvature_mae_1pm']['mean']:+.5f} [{p['fw_lat_curvature_mae_1pm']['lo']:+.5f},{p['fw_lat_curvature_mae_1pm']['hi']:+.5f}]", flush=True)
    return 0


def run_dk(nb: int, out: str = OUT) -> int:
    """LONGITUDINAL distance keeping through the programme's OWN reader (`v8f_dk_read.read_dk`, K = 30 block): B / H / HACC per seed and P10 -- cmd_dk of HSPEC, HACC's plans."""
    import importlib.util
    sp = importlib.util.spec_from_file_location("v8f_dk_read", f"{H.OUT}/deps/v8f_dk_read.py")
    dk = importlib.util.module_from_spec(sp)
    sys.modules["v8f_dk_read"] = dk
    sp.loader.exec_module(dk)
    grid, J, X = H.eval_world()
    order = list(grid["order"])
    preds, saved = {}, {}
    for s in SEEDS:
        z = np.load(f"{H.BANK}/eval_{s}-kind-own.npz")
        k = s.split("_")[1]
        preds[f"B_{k}"], preds[f"H_{k}"] = z["pred"].astype(np.float64), np.load(f"{HB}/score_{s}_perwindow.npz")["H"].astype(np.float64)
        preds[f"HACC_{k}"] = np.load(f"{out}/score_{s}_perwindow.npz")["HACC"].astype(np.float64)
        saved[f"B_{k}"] = z["ade6"]
    preds["P10"] = np.load(H.P10_FILE).astype(np.float64)
    fl = np.load(f"{H.OUT}/inputs/full_eval_floors_perwindow.npz")["P10_refit__ade6"]
    blk = dk.load_block(f"{H.OUT}/inputs/b1_eval_lead_block_k30.npz")
    pairs = []
    for k in ("s0", "s1"):
        pairs += [(f"HACC_{k}", "P10"), (f"HACC_{k}", f"H_{k}"), (f"HACC_{k}", f"B_{k}")]
    res = dk.read_dk(preds=preds, gt=X["gt"].astype(np.float64), eid=X["eid"], clip=np.array([order[e] for e in X["eid"]]), frame=2 * np.asarray(X["t"], np.int64),
                     v0=np.asarray(X["kin"][:, 0], np.float64), block=blk, k=30, dt=DT, n_boot=nb, seed=0, saved_ade6=saved, p10_floor_ade6=fl, pairs=pairs)
    H.jdump(dk._clean(res["read"]), f"{out}/dk_read.json")
    np.savez_compressed(f"{out}/dk_perwindow.npz", **res["perwindow"])
    r = res["read"]
    print(f"[dk] states {r['join']['window_states']} contract {r['human_reference']['kinematic_contract']['status']}", flush=True)
    for nm, ab in r["arms"].items():
        m = ab["metrics"]
        print(f"  {nm:7s} headway {m['headway_min_m'].get('mean')} time-gap {m['time_gap_min_s'].get('mean')} LEAD-VIOL at-risk {ab['lead_viol'].get('share_of_at_risk')}", flush=True)
    return 0


def read_all(out: str = OUT) -> dict:
    Sc = {s: json.load(open(f"{out}/score_{s}.json")) for s in SEEDS}
    for s in SEEDS:
        if "single" not in Sc[s]:
            raise SystemExit(f"{s}: controls did not pass -- no read")
    Fi = {s: json.load(open(f"{out}/fit_{s}.json")) for s in SEEDS}
    dkj = json.load(open(f"{out}/dk_read.json"))
    F_exact = Sc["M2b_s0"]["F_exact_m"]
    per, brief = {}, {}
    for s in SEEDS:
        k = s.split("_")[1]
        dom = dominance(Sc[s]["paired"]["HACC_minus_P10"], Sc[s]["consistency"]["paired"]["HACC_minus_P10"]["viol_any"], dkj["pairs"][f"HACC_{k}__minus__P10"])
        kp = keep_read(Sc[s]["paired"]["HACC_minus_H"]["ade6"], F_exact)
        sm = smooth_flags(Sc[s]["paired"]["HACC_minus_H"])
        per[s] = {"PRIMARY_DOM": dom, "PRIMARY_KEEP": kp, "HACC_minus_H_accel_curvature": sm, "lam_star": Fi[s]["lam_star"], "m_star": Fi[s]["m_star"], "rule_branch": Fi[s]["rule_branch"]}
        brief[s] = {"dom": dom["pass"], "keep": kp["pass"], "accel_sep_neg": sm["accel_sep_neg"], "curv_sep_neg": sm["curv_sep_neg"]}
    outcome = outcome_of(brief)
    R = {"outcome": outcome, "committed_next_step_verbatim": NEXT[outcome], "tier": TIER, "estimator": Sc["M2b_s0"]["estimator"], "F_exact_m": F_exact, "KEEP_bar_m": F_exact / 2,
         "DOM_both": all(brief[s]["dom"] for s in SEEDS), "KEEP_both": all(brief[s]["keep"] for s in SEEDS),
         "SMOOTHER_both": all(brief[s]["accel_sep_neg"] and brief[s]["curv_sep_neg"] for s in SEEDS), "ACCEL_NEG_either": any(brief[s]["accel_sep_neg"] for s in SEEDS),
         "controls_all_pass": all(Sc[s]["controls"]["all_pass"] for s in SEEDS), "per_seed": per, "brief": brief}
    H.jdump(R, f"{out}/READ.json")
    print(f"[read] OUTCOME: {outcome}  DOM_both {R['DOM_both']}  KEEP_both {R['KEEP_both']}  SMOOTHER_both {R['SMOOTHER_both']}  accel-neg-either {R['ACCEL_NEG_either']}", flush=True)
    return R


# ------------------------------------------------------------------------------------------------ orchestration
def cmd_auto(nb: int, out: str = OUT) -> int:
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    for s in SEEDS:                                                           # Phase A: TRAIN arrays only
        rec = fit_seed(s, out)
        if not rec["train_controls"]["all_pass"]:
            print(f"[auto] {s}: a TRAIN-side control FAILED -> rc 2, no read: {json.dumps(rec['train_controls']['pass'])}", flush=True)
            return 2
    x1 = x1_hspec_tie()
    H.jdump(x1, f"{out}/control_X1.json")
    print(f"[auto] X1 (HSPEC reader tie) pass={x1['pass']}  rows checked {x1['checked']}  {x1['mismatches'][:3]}", flush=True)
    if not x1["pass"]:
        return 2
    for s in SEEDS:                                                           # Phase B
        rc = score_seed(s, nb, out)
        if rc:
            return rc
    run_dk(nb, out)
    R = read_all(out)
    _local("build_result", "hacc_build_result").main(out)
    print(f"[auto] DONE {time.time() - t0:.0f}s -> {out}/READ.json, RESULT_V8F_HACC.md", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("auto", "dk", "read", "x1"):
        p = sub.add_parser(name)
        p.add_argument("--n-boot", type=int, default=B_BOOT)
        p.add_argument("--out", default=OUT)
    for name in ("fit", "score"):
        p = sub.add_parser(name)
        p.add_argument("--seed", required=True, choices=SEEDS)
        p.add_argument("--n-boot", type=int, default=B_BOOT)
        p.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "auto":
        return cmd_auto(a.n_boot, a.out)
    if a.mode == "fit":
        fit_seed(a.seed, a.out)
        return 0
    if a.mode == "score":
        return score_seed(a.seed, a.n_boot, a.out)
    if a.mode == "dk":
        return run_dk(a.n_boot, a.out)
    if a.mode == "x1":
        x = x1_hspec_tie()
        H.jdump(x, f"{a.out}/control_X1.json")
        print(json.dumps(x, default=float)[:600])
        return 0 if x["pass"] else 2
    read_all(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
