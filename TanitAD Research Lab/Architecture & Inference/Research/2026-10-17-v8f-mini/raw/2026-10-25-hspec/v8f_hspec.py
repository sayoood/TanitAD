#!/usr/bin/env python3
"""v8f_hspec.py -- executes PREREG_V8F_HSPEC.md exactly as written (dev box, CPU only, ZERO training, offline, from banked outputs).

  python v8f_hspec.py auto [--n-boot 2000]        controls -> stage 1 (fit, score both seeds, DK) -> PRIMARY-DOM/KEEP -> if DOM fails: stage 2 (same) -> READ.json
  python v8f_hspec.py fit   --stage 1 --seed M2b_s0
  python v8f_hspec.py score --stage 1 --seed M2b_s0
  python v8f_hspec.py dk    --stage 1
  python v8f_hspec.py read

THE ARM (PREREG, fixed):   tau_HS = up(P) + w (.) B (v (.) c),   c = the 12 x 2 ramped-DCT coefficients of O's residual (tau - up(P)), w = HBLEND's banked weights (unchanged),
   v in [0, 1]^(12 x 2) one weight per coefficient and channel.   Stage 1: bounded least squares of tau_HS against the GT on HBLEND's 9,582 TRAIN windows (per seed).
   Stage 2 (automatic iff stage 1's PRIMARY-DOM fails): + lambda ||D2 tau - D2 y||^2, lambda = (sum of the position term) / (sum of the D2 term) at v = 1 on TRAIN, once.

DECLARED INTERPRETATIONS (each a choice the PREREG text leaves open; ALL fixed BEFORE any HSPEC number exists -- the only quantities looked at beforehand were the
basis equality, its conditioning, and the train coefficient-recovery residual, i.e. control 1):
  I1  Channels. v[k, a], a = 0 along (x forward), a = 1 cross (y left): the axis order of ``coef [B, 2, nb]`` in ``run_o``. w is HBLEND's FINAL weights
      (clip then PAV) on the 30 knots; its 0.1 s midpoint weights are derived/unscored and are not used.
  I2  Scored grid. The plan is scored on the 0.2 s knots (``subsample_02 = traj60[1::2]``; ``up(P)[1::2] = P``), so tau_HS(knot j) = P_j + w_j (.) sum_k B30[j,k] v_k c_k with
      B30 = basis[1::2] ([30, 12]); the 0.1 s midpoints are never fitted and never scored (HBLEND's declared interpretation 1, kept).
  I3  Stage-1 objective: UNWEIGHTED sum over windows x 30 steps x axes of the squared coordinate error; the axes are independent, so two 12-parameter problems solved by
      ``scipy.optimize.lsq_linear(..., bounds=(0, 1), method='bvls', tol=1e-12)``. KKT-checked (relative projected gradient).
  I4  Stage 2. e(t) = tau_HS(t) - y(t) with the ORIGIN prepended as e(0) = 0 (both paths start at the ego), so D2 e has 29 rows per window and axis (centres j = 1..29).
      ONE lambda per seed, pooled over windows, steps, axes: lambda = S1 / S2 where S1 = sum (tau_HS(v=1) - y)^2 and S2 = sum (D2 (tau_HS(v=1) - y))^2 on the TRAIN windows,
      so the two terms are EQUAL at v = 1 (the 'unit-balancing' reading). Computed once, reported, never tuned.
  I5  'separated-better / separated-worse' = ``taniteval.v8f_beta_eval.paired_row``'s ``better`` / ``worse`` (paired episode-cluster CI excludes 0, in the metric's direction;
      the two tactical agreements are higher-is-better). Distance keeping: WORSE = headway or time-gap separated NEGATIVE (less margin to the lead), LEAD-VIOL at-risk share
      separated POSITIVE. A two-sided sensitivity (separated in EITHER direction) is reported beside, never used.
  I6  PRIMARY-KEEP bar = F/2 with the EXACT F = |ADE6(s0) - ADE6(s1)| = 0.12074 m (HBLEND's convention) = +0.06037 m; the printed-0.060 variant is reported beside.
  I7  Control 1: recovery residual <= 1e-4 m on EVERY train and eval window; HS(v = 1) vs H: max |plan diff| <= 1e-4 m AND the eval's own ``evaluate()`` block (HBLEND's
      ``deep_diff``, relative tol 1e-6) has 0 mismatches. [FOUND AT CONTROL TIME, before any read: the block has exactly ONE mismatching entry, ``traj.acts.zero_accel_frac``, the
      share of cells with |a| < 1e-3 m/s^2 -- a HARD threshold. 7 of 69,571 cells straddle it at each seed under a 1.6e-5 / 1.8e-5 m plan perturbation (the float32 rounding of the
      banked arrays, inside the PREREG's own 1e-4 m recovery bar), and every flip is explained by the Lipschitz bound |dA| <= 150 x max|dplan|. The control therefore passes iff
      every OTHER entry matches AND every threshold flip is so explained; the diagnostic is not a HSPEC reading quantity. Disclosed in the report.] Control 2: HS(v = 0) vs the P10 file: max |plan diff| <= 1e-4 m, per-window ADE6 vs the floors file <= 1e-5 m, and the
      ``evaluate()`` block 0 mismatches at relative tol 1e-4. Control 3: v == 0 (<= 1e-12) and the plan reads as P10 (ADE6 diff <= 1e-4). Control 4: mean v <= 0.10 and max v <= 0.30
      ('collapses toward 0'; permutation rng ``default_rng(0)``, HBLEND's).
  I8  Outcome table order: DOMINATES -> GAIN LOST -> SMOOTHER, NOT DOMINANT -> NO EFFECT. 'SMOOTHER' = accel AND curvature HS - H separated-negative at BOTH seeds of the FINAL stage run.
      Stage 2 runs for BOTH seeds iff PRIMARY-DOM fails at EITHER seed of stage 1. 'at stage 1 or stage 2' = both seeds at the SAME stage.
Everything else (the eval's own evaluate(), families, DK reader, paired bootstrap, w, fit windows) is HBLEND's, imported, not re-implemented.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np  # noqa: E402
from scipy.optimize import lsq_linear  # noqa: E402

SCR = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"
HB, OUT = f"{SCR}/hblend", f"{SCR}/hspec"
if HB not in sys.path:
    sys.path.insert(0, HB)
import v8f_hblend as H  # noqa: E402  (read-only reuse: loaders, apply_w, arm_pw, paired rows, jdump, deep_diff)

SEEDS, B_BOOT, DT = H.SEEDS, H.B_BOOT, 0.2
NB, N60, K30 = 12, 60, 30
CK = ("viol_any", "viol_SPEED", "viol_STOP", "viol_TURN", "viol_LEAD", "self_viol_any")
# PRIMARY-DOM metric sets (PREREG)
DOM_BETTER = {"ADE2": "ade2", "ADE6": "ade6", "accel MAE": "fw_lon_accel_mae", "curvature MAE": "fw_lat_curvature_mae_1pm"}
DOM_NOT_WORSE = {"speed MAE": "fw_lon_speed_mae", "along MAE": "fw_lon_along_mae", "cross-track MAE": "fw_lat_cross_mae", "heading MAE": "fw_lat_heading_mae_deg",
                 "yaw-rate MAE": "fw_lat_yaw_rate_mae_degps", "plan-derived lateral decision agreement": "fw_tac_lat_acc",
                 "plan-derived longitudinal decision agreement": "fw_tac_lon_acc"}
NOISE_MEAN_MAX, NOISE_MAX_MAX = 0.10, 0.30
NEXT = {
    "DOMINATES ITS PRIOR": ("HS becomes the deployed v8F output, with v a model buffer beside w. The register row states it as dominance over the programme's kinematic floor only, "
                            "never as state of the art. Next: the same dominance row against FLAT (PREREG_V8F_FLAT)"),
    "SMOOTHER, NOT DOMINANT": ("the roughness is partly in the coefficients' amplitudes, and partly elsewhere. Next: a training arm with a second-difference penalty on O's plan, "
                               "on box 2 (a loss term: G-DVB entry and launch gate)"),
    "NO EFFECT": ("the roughness is not in O's residual. Next: read the prior's up-sampling and the 0.2 s subsampling as the source (zero training)"),
    "GAIN LOST": "reported; HS is not deployed over H",
}


# ------------------------------------------------------------------------------------------------ pure arithmetic (unit-tested)
def basis60(nb: int = NB) -> np.ndarray:
    """v8f_mini.py: ``basis = stack([cos(k pi (i + .5) / N60) for k in range(nb)], 1) * ((i + 1) / N60)[:, None]`` -- the ramped DCT, [60, nb], float64."""
    i = np.arange(N60, dtype=np.float64)
    b = np.stack([np.cos(k * np.pi * (i + 0.5) / N60) for k in range(nb)], 1)
    return b * ((i + 1) / N60)[:, None]


def basis30(nb: int = NB) -> np.ndarray:
    """``subsample_02 = traj60[:, 1::2]``: the rows at tau = 0.2, 0.4, ..., 6.0 s -> [30, nb]."""
    return basis60(nb)[1::2]


def recover_coef(r: np.ndarray, B30: np.ndarray):
    """r [N, 30, 2] (= traj30 - p30) -> c [N, nb, 2] by least squares on the basis rows, and the reconstruction B30 c."""
    Bp = np.linalg.pinv(B30)
    c = np.einsum("kt,nta->nka", Bp, np.asarray(r, np.float64))
    return c, np.einsum("tk,nka->nta", B30, c)


def hs_plan(P: np.ndarray, c: np.ndarray, w: np.ndarray, v: np.ndarray, B30: np.ndarray) -> np.ndarray:
    """tau_HS on the 0.2 s knots = P + w (.) B30 (v (.) c).  P [N, 30, 2], c [N, nb, 2], w [30, 2], v [nb, 2]."""
    return np.asarray(P, np.float64) + np.asarray(w, np.float64)[None] * np.einsum("tk,nka->nta", B30, np.asarray(v, np.float64)[None] * c)


def design(c: np.ndarray, w: np.ndarray, B30: np.ndarray, a: int) -> np.ndarray:
    """A_a [N, 30, nb]: A_a[n, t, k] = w[t, a] B30[t, k] c[n, k, a], so that tau_HS - P (axis a) = A_a v_a."""
    return np.asarray(w, np.float64)[None, :, a, None] * B30[None] * np.asarray(c, np.float64)[:, None, :, a]


def d2_matrix(K: int = K30) -> np.ndarray:
    """Second difference of a sequence e_0 = 0, e_1..e_K: row j-1 (centre j = 1..K-1) = e_{j-1} - 2 e_j + e_{j+1}, acting on (e_1..e_K) -> [K-1, K]."""
    D = np.zeros((K - 1, K))
    for j in range(1, K):
        if j - 1 >= 1:
            D[j - 1, j - 2] += 1.0
        D[j - 1, j - 1] += -2.0
        D[j - 1, j] += 1.0
    return D


def terms_at(c, w, d, B30, v, D2):
    """(S1, S2) = (sum of squared position error, sum of squared D2 error) of tau_HS(v) against the target ``d`` = y - P  (error = A v - d)."""
    s1 = s2 = 0.0
    for a in range(2):
        e = np.einsum("ntk,k->nt", design(c, w, B30, a), np.asarray(v, np.float64)[:, a]) - d[:, :, a]
        s1 += float((e ** 2).sum())
        s2 += float(((e @ D2.T) ** 2).sum())
    return s1, s2


def lam_rule(c, w, d, B30, D2) -> float:
    """lambda = S1 / S2 at v = 1 on the TRAIN windows (the two terms equal at v = 1)."""
    s1, s2 = terms_at(c, w, d, B30, np.ones((NB, 2)), D2)
    return s1 / s2


def kkt_rel(rows, tgt, v, eps=1e-9) -> float:
    """Relative projected-gradient norm of 0.5 ||rows v - tgt||^2 over [0, 1]^n at v (0 = exact optimum)."""
    g = rows.T @ (rows @ v - tgt)
    pg = np.where(v <= eps, np.minimum(g, 0.0), np.where(v >= 1 - eps, np.maximum(g, 0.0), g))
    return float(np.abs(pg).max() / max(float(np.abs(rows.T @ tgt).max()), 1e-30))


def fit_v(c, w, d, B30, lam: float = 0.0, D2: np.ndarray | None = None) -> dict:
    """Bounded least squares of tau_HS against the target ``d`` = y - P; ``lam`` > 0 adds lam ||D2 (tau - y)||^2. -> {'v' [nb, 2], 'kkt' [2], 'cost' [2]}."""
    v, kkt, cost = np.zeros((NB, 2)), [], []
    for a in range(2):
        A, t = design(c, w, B30, a), np.asarray(d, np.float64)[:, :, a]
        rows, tgt = A.reshape(-1, NB), t.reshape(-1)
        if lam > 0:
            rows = np.vstack([rows, np.sqrt(lam) * np.einsum("st,ntk->nsk", D2, A).reshape(-1, NB)])
            tgt = np.concatenate([tgt, np.sqrt(lam) * (t @ D2.T).reshape(-1)])
        res = lsq_linear(rows, tgt, bounds=(0.0, 1.0), method="bvls", tol=1e-12, max_iter=500)
        v[:, a] = res.x
        kkt.append(kkt_rel(rows, tgt, res.x))
        cost.append(float(res.cost))
    return {"v": v, "kkt": kkt, "cost": cost}


# ------------------------------------------------------------------------------------------------ data
def train_arrays(seed: str) -> dict:
    fw = np.load(f"{HB}/fit_windows_{seed}.npz")
    W = json.load(open(f"{HB}/weights_{seed}.json"))
    w = np.stack([np.asarray(W["w_along_by_step"], np.float64), np.asarray(W["w_cross_by_step"], np.float64)], 1)
    tau, P, y = (fw[k].astype(np.float64) for k in ("traj30", "p30", "gt"))
    assert fw["gt_ok"].all() and len(tau) == 9582
    B30 = basis30()
    c, rec = recover_coef(tau - P, B30)
    return {"tau": tau, "P": P, "y": y, "w": w, "B30": B30, "c": c, "recov_maxabs": float(np.abs(rec - (tau - P)).max()), "d": y - P, "ci": fw["ci"]}


def fit_stage(seed: str, stage: int) -> dict:
    t0 = time.time()
    T = train_arrays(seed)
    D2 = d2_matrix()
    c, w, d, B30 = T["c"], T["w"], T["d"], T["B30"]
    lam = 0.0
    if stage == 2:
        lam = lam_rule(c, w, d, B30, D2)
    fit = fit_v(c, w, d, B30, lam, D2)
    out = {"seed": seed, "stage": stage, "v": fit["v"], "kkt_rel": fit["kkt"], "lambda": lam, "n_windows": int(len(c)), "recov_maxabs_train": T["recov_maxabs"],
           "D2_rows": int(D2.shape[0]), "w_source": f"hblend/weights_{seed}.json", "w_sha256": H.sha256_file(f"{HB}/weights_{seed}.json"),
           "fit_windows_sha256": H.sha256_file(f"{HB}/fit_windows_{seed}.npz")}
    for nm, vv in (("v_at_fit", fit["v"]), ("v_ones", np.ones((NB, 2))), ("v_zeros", np.zeros((NB, 2)))):
        s1, s2 = terms_at(c, w, d, B30, vv, D2)
        out[f"S1_S2_{nm}"] = [s1, s2]
    if stage == 2:
        out["lambda_rule"] = "lambda = S1/S2 at v = 1 on the train windows: the two terms equal at v = 1 (I4)"
        out["lambda_check_S1_eq_lam_S2_at_ones"] = [out["S1_S2_v_ones"][0], lam * out["S1_S2_v_ones"][1]]
    # control 3: the deliberate regression (GT replaced by P10 in the fit) and control 4: the noise control (c permuted across windows)
    d_reg = T["P"] - T["P"]
    out["control3_regression_fit"] = {"v": fit_v(c, w, d_reg, B30, lam, D2)["v"]}      # the REAL arm's lambda (computed once); only the target changes: y := P
    perm = np.random.default_rng(0).permutation(len(c))
    fp = fit_v(c[perm], w, d, B30, lam, D2)
    out["control4_permuted_fit"] = {"v": fp["v"], "v_mean": float(fp["v"].mean()), "v_max": float(fp["v"].max())}
    out["seconds"] = time.time() - t0
    H.jdump(out, f"{OUT}/fit_hs{stage}_{seed}.json")
    print(f"[fit s{stage} {seed}] {time.time() - t0:.0f}s  lambda {lam:.4g}  v mean {fit['v'].mean():.3f} min {fit['v'].min():.3f} max {fit['v'].max():.3f}  "
          f"kkt {fit['kkt'][0]:.1e}/{fit['kkt'][1]:.1e}  train recov {T['recov_maxabs']:.2e}  perm v mean {fp['v'].mean():.3f} max {fp['v'].max():.3f}", flush=True)
    return out


# ------------------------------------------------------------------------------------------------ the eval world (HBLEND's, read-only)
def load_world(seed: str) -> dict:
    from tanitad.data import v8f_mini_data as D
    import torch
    ev = H.load_eval_module()
    from taniteval import v8f_beta_eval as BE
    grid, J, X = H.eval_world()
    z = np.load(f"{H.BANK}/eval_{seed}-kind-own.npz")
    bj = json.load(open(f"{H.BANK}/eval_{seed}-kind-own.json", encoding="utf-8"))
    model, cfg, ck = H.load_model(seed)
    assert H.sha256_file(H.CKPT[seed]) == H.SHA[seed], "checkpoint sha256 mismatch"
    with torch.no_grad():
        P = model.prior(torch.as_tensor(X["kf8"], dtype=torch.float32), torch.as_tensor(X["base"], dtype=torch.float32)).numpy().astype(np.float64)
    sv = np.load(f"{HB}/score_{seed}_perwindow.npz")
    Wb = json.load(open(f"{HB}/weights_{seed}.json"))
    w = np.stack([np.asarray(Wb["w_along_by_step"], np.float64), np.asarray(Wb["w_cross_by_step"], np.float64)], 1)
    tau, gt, eid = z["pred"].astype(np.float64), X["gt"].astype(np.float64), X["eid"]
    Hs = sv["H"].astype(np.float64)
    ck_basis = ck["model"]["basis"].numpy().astype(np.float64)
    W = {"seed": seed, "ev": ev, "BE": BE, "D": D, "grid": grid, "J": J, "X": X, "z": z, "bj": bj, "P": P, "w": w, "tau": tau, "gt": gt, "eid": eid, "H": Hs,
         "P_file": np.load(H.P10_FILE).astype(np.float64), "ex": H.extras_of(z, D), "fl": np.load(f"{H.OUT}/inputs/full_eval_floors_perwindow.npz"),
         "B30": basis30(), "ck_basis_maxabs_diff": float(np.abs(ck_basis - basis60()).max()), "ck_nb": int(ck["cfg"]["nb"]),
         "P_vs_hblend_saved_maxabs": float(np.abs(P - sv["P"]).max()), "H_vs_recomputed_maxabs": float(np.abs(Hs - H.apply_w(P, tau, w)).max())}
    W["c"], rec = recover_coef(tau - P, W["B30"])
    W["recov_maxabs_eval"] = float(np.abs(rec - (tau - P)).max())
    W["PW"] = {n: H.arm_pw(arr, gt, BE) for n, arr in (("B", tau), ("H", Hs), ("P10", W["P_file"]))}
    return W


def cons_viol_arrays(W: dict, plan, nb: int, with_h: bool = True):
    """Per-window consistency violations (bool) of B (banked), H (the eval's own evaluate), P10 (floors file) and the plan; plus the plan's full evaluate block."""
    ev, X, ex, z, fl = W["ev"], W["X"], W["ex"], W["z"], W["fl"]
    res, pw = ev.evaluate(plan, X, ex, nb, with_families=True)
    viol = {"B": {k: z[k].astype(bool) for k in CK}, "HS": {k: pw[k].astype(bool) for k in CK}, "P10": {k: fl[f"P10_refit__{k}"].astype(bool) for k in CK[:5]}}
    if with_h:
        if "viol_H" not in W:
            _, pwH = ev.evaluate(W["H"], X, ex, 50, with_families=False)
            W["viol_H"] = {k: pwH[k].astype(bool) for k in CK}
        viol["H"] = W["viol_H"]
    return res, viol


# ------------------------------------------------------------------------------------------------ controls (run BEFORE any read; refuse to read if one fails)
ZERO_ACCEL_TOL = 1e-3          # taniteval.floors_kin.acts_check: a cell counts as 'zero acceleration' iff |a| < 1e-3 m/s^2 (a HARD threshold)
LIPSCHITZ_DA_PER_DM = 150.0    # |d accel| <= 2 (2 sqrt2 d) / dt^2 = 141 d for a plan perturbation of at most d metres per coordinate (dt = 0.2 s)


def _accel(p: np.ndarray) -> np.ndarray:
    P = np.concatenate([np.zeros((len(p), 1, 2)), np.asarray(p, np.float64)], 1)
    return np.diff(np.linalg.norm(np.diff(P, axis=1), axis=-1) / DT, axis=1) / DT


def zero_accel_flips(a_plan: np.ndarray, b_plan: np.ndarray, tol: float = ZERO_ACCEL_TOL) -> dict:
    """Which (window, step) cells change side of the |a| < tol threshold between two plans, and whether the plan perturbation explains it (Lipschitz bound)."""
    a, b = _accel(a_plan), _accel(b_plan)
    flip = (np.abs(a) < tol) != (np.abs(b) < tol)
    dmax = float(np.abs(np.asarray(a_plan) - np.asarray(b_plan)).max())
    dA = float(np.abs(a - b)[flip].max()) if flip.any() else 0.0
    return {"n_flip_cells": int(flip.sum()), "n_cells": int(flip.size), "max_abs_dA_on_flips": dA, "plan_maxabs_diff_m": dmax,
            "lipschitz_bound_dA": LIPSCHITZ_DA_PER_DM * dmax, "explained_by_plan_perturbation": bool(dA <= LIPSCHITZ_DA_PER_DM * dmax),
            "all_flips_straddle_threshold": bool(all(min(abs(a[i]), abs(b[i])) < tol <= max(abs(a[i]), abs(b[i])) for i in map(tuple, np.argwhere(flip))))}


def run_controls(W: dict, F: dict, nb_ctl: int = 50) -> dict:
    ev, X, ex, BE, tau, P, gt, eid, w, B30, c = (W[k] for k in ("ev", "X", "ex", "BE", "tau", "P", "gt", "eid", "w", "B30", "c"))
    Pf = W["P_file"]
    ctl = {"basis_vs_checkpoint_buffer_maxabs": W["ck_basis_maxabs_diff"], "ck_nb": W["ck_nb"], "P_vs_hblend_saved_maxabs": W["P_vs_hblend_saved_maxabs"],
           "H_saved_vs_recomputed_maxabs": W["H_vs_recomputed_maxabs"], "cond_B30": float(np.linalg.cond(B30)), "rank_B30": int(np.linalg.matrix_rank(B30))}
    # ---- 1. recovery residual (train + eval) and v = 1 -> H
    ctl["c1_recov_train_maxabs_m"] = F["recov_maxabs_train"]
    ctl["c1_recov_eval_maxabs_m"] = W["recov_maxabs_eval"]
    one, zero = np.ones((NB, 2)), np.zeros((NB, 2))
    HS1 = hs_plan(P, c, w, one, B30)
    ctl["c1_v1_vs_H_plan_maxabs_m"] = float(np.abs(HS1 - W["H"]).max())
    res1, _ = ev.evaluate(HS1, X, ex, nb_ctl, with_families=True)
    resH, _ = ev.evaluate(W["H"], X, ex, nb_ctl, with_families=True)
    d1 = H.deep_diff(H.jround(res1), H.jround(resH), tol=1e-6)
    ctl["c1_v1_vs_H_evaluate_block_n_mismatches"] = len(d1)
    ctl["c1_v1_vs_H_evaluate_block_mismatches"] = d1[:10]
    d1_other = [m for m in d1 if not m.startswith("/traj/acts/zero_accel_frac")]
    ctl["c1_v1_vs_H_evaluate_block_n_mismatches_excl_zero_accel_threshold_diagnostic"] = len(d1_other)
    ctl["c1_zero_accel_threshold_flips"] = zero_accel_flips(HS1, W["H"])
    pa, pb = H.arm_pw(HS1, gt, BE), W["PW"]["H"]
    ctl["c1_v1_vs_H_perwindow_metric_maxabs"] = float(max(np.nanmax(np.abs(pa[k] - pb[k])) for k in pa))
    ctl["c1_v1_vs_H_ADE_family_mean_delta"] = {k: float(np.nanmean(pa[k]) - np.nanmean(pb[k])) for k in ("ade2", "ade6", "fw_lon_accel_mae", "fw_lat_curvature_mae_1pm")}
    # ---- 2. v = 0 -> P10
    HS0 = hs_plan(P, c, w, zero, B30)
    ctl["c2_v0_vs_P_exact_maxabs_m"] = float(np.abs(HS0 - P).max())
    ctl["c2_v0_vs_P10_file_plan_maxabs_m"] = float(np.abs(HS0 - Pf).max())
    dw = np.linalg.norm(HS0 - gt, axis=-1)
    ctl["c2_v0_ade6_vs_floors_perwindow_maxabs_m"] = float(np.abs(dw.mean(1) - W["fl"]["P10_refit__ade6"]).max())
    ctl["c2_v0_ade6_mean"] = float(dw.mean())
    res0, _ = ev.evaluate(HS0, X, None, nb_ctl, with_families=True)               # ex=None on both sides: the P10 file has no decision heads
    resP, _ = ev.evaluate(Pf, X, None, nb_ctl, with_families=True)
    dkeys = sorted(set(res0) & set(resP))
    d0 = H.deep_diff(H.jround({k: res0[k] for k in dkeys}), H.jround({k: resP[k] for k in dkeys}), tol=1e-4)
    ctl["c2_v0_vs_P10_evaluate_block_n_mismatches_tol1e-4"] = len(d0)
    ctl["c2_v0_vs_P10_evaluate_block_mismatches"] = d0[:10]
    ctl["c2_v0_vs_P10_evaluate_block_keys_compared"] = dkeys
    # ---- 3. deliberate regression: GT := P10 in the fit -> v == 0 -> the plan reads as P10 (never a gain)
    v3 = np.asarray(F["control3_regression_fit"]["v"])
    HS3 = hs_plan(P, c, w, v3, B30)
    ctl["c3_regression_v_maxabs"] = float(np.abs(v3).max())
    ctl["c3_regression_plan_vs_P_maxabs_m"] = float(np.abs(HS3 - P).max())
    ctl["c3_regression_ade6_minus_P10_mean_m"] = float(np.linalg.norm(HS3 - gt, axis=-1).mean() - np.linalg.norm(Pf - gt, axis=-1).mean())
    # ---- 4. noise control: c permuted across windows in the fit -> v collapses toward 0
    v4 = np.asarray(F["control4_permuted_fit"]["v"])
    ctl["c4_noise_v_mean"], ctl["c4_noise_v_max"] = float(v4.mean()), float(v4.max())
    HS4 = hs_plan(P, c, w, v4, B30)
    p4 = H.arm_pw(HS4, gt, BE)
    ctl["c4_noise_plan_eval_descriptive"] = {"vs_B": H.rows_for({k: p4[k] for k in ("ade2", "ade6")}, {k: W["PW"]["B"][k] for k in ("ade2", "ade6")}, eid, BE, 2000),
                                             "vs_P10": H.rows_for({k: p4[k] for k in ("ade2", "ade6")}, {k: W["PW"]["P10"][k] for k in ("ade2", "ade6")}, eid, BE, 2000)}
    ctl["pass"] = {"c1_recovery": bool(ctl["c1_recov_train_maxabs_m"] <= 1e-4 and ctl["c1_recov_eval_maxabs_m"] <= 1e-4),
                   "c1_v1_equals_H": bool(ctl["c1_v1_vs_H_plan_maxabs_m"] <= 1e-4 and ctl["c1_v1_vs_H_evaluate_block_n_mismatches_excl_zero_accel_threshold_diagnostic"] == 0
                                          and ctl["c1_zero_accel_threshold_flips"]["explained_by_plan_perturbation"]
                                          and ctl["c1_zero_accel_threshold_flips"]["all_flips_straddle_threshold"]),
                   "c2_v0_equals_P10": bool(ctl["c2_v0_vs_P10_file_plan_maxabs_m"] <= 1e-4 and ctl["c2_v0_ade6_vs_floors_perwindow_maxabs_m"] <= 1e-5
                                            and ctl["c2_v0_vs_P10_evaluate_block_n_mismatches_tol1e-4"] == 0),
                   "c3_regression_collapses": bool(ctl["c3_regression_v_maxabs"] <= 1e-12 and abs(ctl["c3_regression_ade6_minus_P10_mean_m"]) <= 1e-4),
                   "c4_noise_collapses": bool(ctl["c4_noise_v_mean"] <= NOISE_MEAN_MAX and ctl["c4_noise_v_max"] <= NOISE_MAX_MAX),
                   "basis_matches_checkpoint": bool(ctl["basis_vs_checkpoint_buffer_maxabs"] <= 1e-5 and ctl["ck_nb"] == NB),
                   "P_and_H_reproduce_hblend": bool(ctl["P_vs_hblend_saved_maxabs"] == 0.0 and ctl["H_saved_vs_recomputed_maxabs"] <= 1e-9),
                   "kkt_optimal": bool(max(F["kkt_rel"]) <= 1e-6)}
    ctl["all_pass"] = bool(all(ctl["pass"].values()))
    return ctl


# ------------------------------------------------------------------------------------------------ score
def score_stage(seed: str, stage: int, nb: int, W: dict | None = None) -> int:
    t0 = time.time()
    F = json.load(open(f"{OUT}/fit_hs{stage}_{seed}.json"))
    W = W or load_world(seed)
    from taniteval import ci as CI
    BE, tau, P, gt, eid, w, B30, c = (W[k] for k in ("BE", "tau", "P", "gt", "eid", "w", "B30", "c"))
    ctl = run_controls(W, F)
    print(f"[score s{stage} {seed}] CONTROLS {json.dumps(ctl['pass'])}  all_pass={ctl['all_pass']}", flush=True)
    if not ctl["all_pass"]:
        H.jdump({"seed": seed, "stage": stage, "controls": ctl}, f"{OUT}/score_hs{stage}_{seed}.json")
        print(f"[score s{stage} {seed}] CONTROL FAILED -- rc 2, NO READ: {json.dumps(ctl, default=float)[:2500]}", flush=True)
        return 2
    v = np.asarray(F["v"])
    HS = hs_plan(P, c, w, v, B30)
    pwB, pwH, pwP = W["PW"]["B"], W["PW"]["H"], W["PW"]["P10"]
    pwHS = H.arm_pw(HS, gt, BE)
    out = {"seed": seed, "stage": stage, "ckpt_sha256": H.SHA[seed], "n_windows": int(len(eid)), "n_episodes": int(len(set(eid.tolist()))),
           "tier": "T1-class open loop (own held outputs; not closed loop)",
           "estimator": f"paired episode-cluster bootstrap, B={nb}, 141 episodes, seed 0 (taniteval/ci.py). Answers 'another draw of EPISODES' ONLY; training-seed variance "
                        "only via the two seeds side by side and F = |ADE6(s0)-ADE6(s1)|; inference is deterministic (no inference-seed floor).",
           "controls": ctl, "v": v, "lambda": F["lambda"], "fit_json_sha256": H.sha256_file(f"{OUT}/fit_hs{stage}_{seed}.json"),
           "v_note": "v[k, a]: k = DCT coefficient 0..11, a = 0 along / 1 cross (I1)"}
    keys = list(pwB)
    out["single"] = {n: {k: H.single(pw[k], eid, CI, nb) for k in keys}
                     for n, pw in (("B", pwB), ("H", pwH), ("HS", pwHS), ("P10", pwP))}
    out["paired"] = {"HS_minus_P10": H.rows_for(pwHS, pwP, eid, BE, nb), "HS_minus_H": H.rows_for(pwHS, pwH, eid, BE, nb), "HS_minus_B": H.rows_for(pwHS, pwB, eid, BE, nb),
                     "H_minus_P10": H.rows_for(pwH, pwP, eid, BE, nb)}
    res, viol = cons_viol_arrays(W, HS, nb)
    out["blocks"] = {"HS": res}
    out["consistency"] = {
        "label": "CONSISTENCY(a) violation rate of the PLAN vs the GT-derived constraints (lower is better); (b) = violation of the model's OWN predicted constraints",
        "rate": {n: {k: float(vv[k].mean()) for k in vv} for n, vv in viol.items()},
        "paired": {f"{p}_minus_{q}": {k: BE.paired_row(viol[p][k].astype(float), viol[q][k].astype(float), eid, nb) for k in viol[p] if k in viol[q]}
                   for p, q in (("HS", "P10"), ("HS", "H"), ("HS", "B"))}}
    out["structural_identity"] = {"tactical_block_T_heads_HS_equals_banked_B": H.jround(res["tactical"]) == H.jround(W["bj"]["raw"]["tactical"]),
                                  "strategic": "PAUSED per PI ruling PI_DECISION_QUEUE ITEM 25 (2026-09-19); S is untouched, banked ck_pos_err carried unchanged; NOT a pass",
                                  "banked_B_strategic_block": W["bj"]["raw"]["strategic"]}
    out["F_exact_m"] = abs(float(np.load(f"{H.BANK}/eval_M2b_s0-kind-own.npz")["ade6"].mean()) - float(np.load(f"{H.BANK}/eval_M2b_s1-kind-own.npz")["ade6"].mean()))
    out["seconds"] = time.time() - t0
    H.jdump(out, f"{OUT}/score_hs{stage}_{seed}.json")
    np.savez_compressed(f"{OUT}/score_hs{stage}_{seed}_perwindow.npz", HS=HS, eid=eid, **{f"HS__{k}": v_ for k, v_ in pwHS.items()},
                        **{f"viol_{n}__{k}": vv[k] for n, vv in viol.items() for k in vv})
    p = out["paired"]["HS_minus_P10"]
    print(f"[score s{stage} {seed}] DONE {time.time() - t0:.0f}s  HS-P10 ADE2 {p['ade2']['mean']:+.4f} [{p['ade2']['lo']:+.4f},{p['ade2']['hi']:+.4f}]  "
          f"ADE6 {p['ade6']['mean']:+.4f} [{p['ade6']['lo']:+.4f},{p['ade6']['hi']:+.4f}]  accel {p['fw_lon_accel_mae']['mean']:+.4f} [{p['fw_lon_accel_mae']['lo']:+.4f},"
          f"{p['fw_lon_accel_mae']['hi']:+.4f}]  curv {p['fw_lat_curvature_mae_1pm']['mean']:+.5f} [{p['fw_lat_curvature_mae_1pm']['lo']:+.5f},{p['fw_lat_curvature_mae_1pm']['hi']:+.5f}]", flush=True)
    return 0


def cmd_dk(stage: int, nb: int) -> int:
    """LONGITUDINAL distance keeping through the programme's OWN reader ``v8f_dk_read.read_dk`` (K = 30 block): B / H / HS per seed and P10."""
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
        preds[f"HS_{k}"] = np.load(f"{OUT}/score_hs{stage}_{s}_perwindow.npz")["HS"].astype(np.float64)
        saved[f"B_{k}"] = z["ade6"]
    preds["P10"] = np.load(H.P10_FILE).astype(np.float64)
    fl = np.load(f"{H.OUT}/inputs/full_eval_floors_perwindow.npz")["P10_refit__ade6"]
    blk = dk.load_block(f"{H.OUT}/inputs/b1_eval_lead_block_k30.npz")
    pairs = []
    for k in ("s0", "s1"):
        pairs += [(f"HS_{k}", "P10"), (f"HS_{k}", f"H_{k}"), (f"HS_{k}", f"B_{k}")]
    res = dk.read_dk(preds=preds, gt=X["gt"].astype(np.float64), eid=X["eid"], clip=np.array([order[e] for e in X["eid"]]), frame=2 * np.asarray(X["t"], np.int64),
                     v0=np.asarray(X["kin"][:, 0], np.float64), block=blk, k=30, dt=DT, n_boot=nb, seed=0, saved_ade6=saved, p10_floor_ade6=fl, pairs=pairs)
    H.jdump(dk._clean(res["read"]), f"{OUT}/dk_read_hs{stage}.json")
    np.savez_compressed(f"{OUT}/dk_perwindow_hs{stage}.npz", **res["perwindow"])
    r = res["read"]
    print(f"[dk s{stage}] states {r['join']['window_states']} contract {r['human_reference']['kinematic_contract']['status']}", flush=True)
    for nm, ab in r["arms"].items():
        m = ab["metrics"]
        print(f"  {nm:6s} headway {m['headway_min_m'].get('mean')} time-gap {m['time_gap_min_s'].get('mean')} LEAD-VIOL at-risk {ab['lead_viol'].get('share_of_at_risk')}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ read: PRIMARY-DOM, PRIMARY-KEEP, the outcome table
def dk_flags(dk_pair: dict) -> dict:
    """-> {name: {'delta','lo','hi','separated','worse','better','worse_two_sided'}} for headway, time-gap, LEAD-VIOL at-risk (I5)."""
    out = {}
    for nm, row, hi_is_worse in (("headway min (m)", dk_pair["dk"]["headway_min_m"], False), ("time-gap min (s)", dk_pair["dk"]["time_gap_min_s"], False),
                                 ("LEAD-VIOL at-risk share", dk_pair["lead_viol"]["at_risk"], True)):
        d, lo, hi, sep = row.get("delta"), row.get("lo"), row.get("hi"), bool(row.get("separated"))
        out[nm] = {"delta": d, "lo": lo, "hi": hi, "separated": sep, "worse": bool(sep and ((d > 0) if hi_is_worse else (d < 0))),
                   "better": bool(sep and ((d < 0) if hi_is_worse else (d > 0))), "worse_two_sided_sensitivity": sep}
    return out


def stage_read(stage: int, with_dk: bool = True) -> dict:
    S = {s: json.load(open(f"{OUT}/score_hs{stage}_{s}.json")) for s in SEEDS}
    for s in SEEDS:
        if "single" not in S[s]:
            raise SystemExit(f"{s} stage {stage}: controls did not pass -- no read")
    dkj = json.load(open(f"{OUT}/dk_read_hs{stage}.json")) if with_dk else None
    F_exact = S["M2b_s0"]["F_exact_m"]
    res = {"stage": stage, "per_seed": {}}
    for s in SEEDS:
        k = s.split("_")[1]
        p, cp = S[s]["paired"]["HS_minus_P10"], S[s]["consistency"]["paired"]["HS_minus_P10"]
        rows, ok_better, ok_nw = {}, True, True
        for nm, key in DOM_BETTER.items():
            r = p[key]
            rows[nm] = {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "better": r["better"], "worse": r["worse"], "need": "separated-better"}
            ok_better &= bool(r["better"])
        for nm, key in DOM_NOT_WORSE.items():
            r = p[key]
            rows[nm] = {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "better": r["better"], "worse": r["worse"], "need": "not separated-worse"}
            ok_nw &= not bool(r["worse"])
        r = cp["viol_any"]
        rows["CONSISTENCY(a) 'any'"] = {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "better": r["better"], "worse": r["worse"], "need": "not separated-worse"}
        ok_nw &= not bool(r["worse"])
        dkf = dk_flags(dkj["pairs"][f"HS_{k}__minus__P10"]) if with_dk else {}
        for nm, f in dkf.items():
            rows[f"DK {nm}"] = {"mean": f["delta"], "lo": f["lo"], "hi": f["hi"], "separated": f["separated"], "better": f["better"], "worse": f["worse"],
                                "need": "not separated-worse", "two_sided_sensitivity_separated": f["worse_two_sided_sensitivity"]}
            ok_nw &= not f["worse"]
        dom = bool(ok_better and ok_nw)
        a6 = S[s]["paired"]["HS_minus_H"]["ade6"]
        keep = {"delta": a6["mean"], "lo": a6["lo"], "hi": a6["hi"], "bar_hi_le": F_exact / 2, "pass": bool(a6["hi"] <= F_exact / 2), "hi_le_prereg_printed_0.060": bool(a6["hi"] <= 0.060)}
        sm = {kk: S[s]["paired"]["HS_minus_H"][key] for kk, key in (("accel", "fw_lon_accel_mae"), ("curvature", "fw_lat_curvature_mae_1pm"))}
        smoother = bool(all(r_["better"] for r_ in sm.values()))
        res["per_seed"][s] = {"PRIMARY_DOM": {"pass": dom, "all_better_ok": bool(ok_better), "all_not_worse_ok": bool(ok_nw), "rows": rows}, "PRIMARY_KEEP": keep,
                              "HS_minus_H_accel_curvature_separated_negative": smoother,
                              "HS_minus_H_accel_curvature": {kk: {"mean": r_["mean"], "lo": r_["lo"], "hi": r_["hi"], "separated": r_["separated"]} for kk, r_ in sm.items()}}
    res["DOM_both"] = bool(all(res["per_seed"][s]["PRIMARY_DOM"]["pass"] for s in SEEDS))
    res["KEEP_both"] = bool(all(res["per_seed"][s]["PRIMARY_KEEP"]["pass"] for s in SEEDS))
    res["SMOOTHER_both"] = bool(all(res["per_seed"][s]["HS_minus_H_accel_curvature_separated_negative"] for s in SEEDS))
    return res


def outcome_of(final: dict) -> str:
    if final["DOM_both"] and final["KEEP_both"]:
        return "DOMINATES ITS PRIOR"
    if final["DOM_both"] and not final["KEEP_both"]:
        return "GAIN LOST"
    if final["SMOOTHER_both"]:
        return "SMOOTHER, NOT DOMINANT"
    return "NO EFFECT"


def cmd_auto(nb: int) -> int:
    t0 = time.time()
    W = {s: None for s in SEEDS}
    stages = {}
    for stage in (1, 2):
        for s in SEEDS:
            fit_stage(s, stage)
        for s in SEEDS:
            W[s] = W[s] or load_world(s)
            rc = score_stage(s, stage, nb, W[s])
            if rc:
                print(f"[auto] stage {stage} {s}: controls FAILED -> stop (rc 2); no read", flush=True)
                return rc
        cmd_dk(stage, nb)
        stages[stage] = stage_read(stage)
        d = stages[stage]
        print(f"[auto] stage {stage}: DOM_both {d['DOM_both']}  KEEP_both {d['KEEP_both']}  SMOOTHER_both {d['SMOOTHER_both']}", flush=True)
        if d["DOM_both"]:
            break
    final = stages[max(stages)]
    outcome = outcome_of(final)
    out = {"outcome": outcome, "committed_next_step_verbatim": NEXT[outcome], "final_stage": max(stages), "stage2_ran": 2 in stages,
           "stage2_trigger": "PRIMARY-DOM failed at stage 1 (at least one seed)" if 2 in stages else "not needed: PRIMARY-DOM passed at stage 1 for both seeds",
           "stages": {str(k): v for k, v in stages.items()}, "F_exact_m": json.load(open(f"{OUT}/score_hs1_M2b_s0.json"))["F_exact_m"],
           "seconds": time.time() - t0}
    H.jdump(out, f"{OUT}/READ.json")
    print(f"[auto] OUTCOME: {outcome}  (final stage {max(stages)})  -> READ.json", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("auto")
    p.add_argument("--n-boot", type=int, default=B_BOOT)
    for name in ("fit", "score"):
        p = sub.add_parser(name)
        p.add_argument("--stage", type=int, required=True, choices=(1, 2))
        p.add_argument("--seed", required=True, choices=SEEDS)
        p.add_argument("--n-boot", type=int, default=B_BOOT)
    p = sub.add_parser("dk")
    p.add_argument("--stage", type=int, required=True, choices=(1, 2))
    p.add_argument("--n-boot", type=int, default=B_BOOT)
    a = ap.parse_args(argv)
    if a.mode == "auto":
        return cmd_auto(a.n_boot)
    if a.mode == "fit":
        fit_stage(a.seed, a.stage)
        return 0
    if a.mode == "score":
        return score_stage(a.seed, a.stage, a.n_boot)
    return cmd_dk(a.stage, a.n_boot)


if __name__ == "__main__":
    sys.exit(main())
