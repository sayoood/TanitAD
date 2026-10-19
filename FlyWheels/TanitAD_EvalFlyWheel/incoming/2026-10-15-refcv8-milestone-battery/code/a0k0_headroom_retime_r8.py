"""a0k0_headroom_retime_r8.py -- ARC-LENGTH RE-TIMING of refcv8's own plan (a COPY of a0k0_headroom_lon_r8.py, which is banked provenance and untouched).
Keep refcv8's PATH exactly (every re-timed point lies ON the os polyline) and change only the PROGRESS along it:

    C        = polyline through (0,0) and the os points at the S-grid instants
    s_os(t)  = cumulative arc length of os;  s_prior(t) = cumulative arc length of the causal prior ha0_ext_pose (same instants)
    s'(t)    = s_prior(t) + g_t * (s_os(t) - s_prior(t))   [evaluated as L + (1-g)(M-L): g == 1 -> s' == s_os bit for bit], clipped at >= 0
    p(t)     = C(s'(t)), linear interpolation along the polyline, beyond the end along the last positive-length segment's direction

CPU only, banked panels only, zero training.  Pre-registration (written before any real-data number of these arms): <PKG>/raw/a0k0_retime/PREREG_RETIME.md.
Output: <PKG>/raw/a0k0_retime/A0K0_RETIME.{json,md}, FROZEN_GAINS_RETIME_ema30k.json, FORWARD_RETIME_<tree>.json, RESULTS_BRIEF.md.

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8     PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code'       python -X utf8 a0k0_headroom_retime_r8.py compute      # -> A0K0_RETIME.json (+ .md)
      python -X utf8 a0k0_headroom_retime_r8.py render | freeze | forward --tree <tree> | brief

ADMISSIBILITY (PI 2026-10-18 ~23:58, relayed): measured a0 and yaw rate at t0 ARE admissible (IMU-measurable); curvature is NOT.
refcv8 is NOT "fed v0 only": its argv carries --ego-history and --residual-prior ha0_ext_pose (kinematic_prior.py), so its plan is
a causal prior (a0, kappa0 = omega0 / max(v0, 2)) + a learned residual.  The E1 echo ha0_ext reads kappa0 from the STEER channel.

REFERENCES (model-free; built from the panel's own decisions/ep_poses, observed window ws-7..ws):
  ha0_ext       the banked E1 echo (panel arm) -- E1 AS REGISTERED, the bar of record.
  ha0_ext_pose  kinematic_prior.prior_controls('ha0_ext_pose') -> prior_path: refcv8's own prior.  PRIMARY admissible reference and the progress base of the re-timing.
  ha0_imu       hold a0 and yaw rate psi_dot0 constant (SECONDARY);  ha0_imu_ax: the same with a0 = measured ax.
ARMS (g per instant, ONE scalar per instant; fit = coarse grid 0.05 + bounded Brent refinement of the mean window ADE, the loss is NOT convex):
  RETIME_XF / GATED_RETIME_XF  5 EPISODE folds (clip_index, rng 20261018), fit on 4 apply to 1;  GATED = one table per v0 bin.
  RETIME_IS / GATED_RETIME_IS  in-sample (known sign: mean ADE <= os);  RETIME_WF / GATED_RETIME_WF  folds per WINDOW (leaky control).
  RS_XF                        the Amendment-2 arm, recomputed with the unchanged L-BFGS machinery as a CONTROL against the banked A0K0_HEADROOM.json cells.
ESTIMATOR  pair_read_r8.paired_cell -> taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0: "another draw of EPISODES" only.
TIER       T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).  EARLY READ of banked step 30,000.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PKG = HERE.parent
OUT = PKG / "raw" / "a0k0_retime"
BANKED = PKG / "raw" / "a0k0_headroom"      # the parent deliverable: read-only here
PREREG = OUT / "PREREG_RETIME.md"
TREES = ("step30000ema_unc", "step30000_unc")
SEEDS = (0, 1)
SURFS = ("S2", "S6")
SETS = ("compliant", "all")
MASK_SHA = "4ebc6b8604f064f69e9e94c15e099062f68339a8ef38d666e8a29d29cb1c4129"
N_BOOT = 2000
K_FOLDS = 5
FOLD_SEED = 20261018
SENS_SEEDS = tuple(range(1, 11))          # RETIME_XF fold-seed sensitivity (EMA-30k, inference seed 0 only)
SENS_SEEDS_GATED = tuple(range(1, 6))     # GATED_RETIME_XF (5 seeds: each is 30 fits)
MIN_BIN_FIT = 30
W_OBS = 8
DT = 0.1
BOX_RS = (0.0, 2.0)
BOX_BL = (0.0, 1.0)
RT_BOX = (0.0, 2.0)                 # the re-timing gain box (scalar g per instant)
RT_GRID_STEP = 0.05                 # coarse grid; bounded Brent refinement inside the best cell
RT_ON_PATH_TOL_M = 1e-6             # control (b): re-timed points lie on the os polyline (float64 positions)
FROZEN_ARMS = ("RETIME", "GATED_RETIME")
VERIFY_STEP = 0.1
V1_TOL_M = 1e-3
V2_TOL_M = 1e-4
REF0926 = "C:/lgs/Lfb47a58376/TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-residual-prior/raw/measure_prior_vs_echo.json"
CELL_KEYS = ("delta", "lo", "hi", "n_windows", "n_episodes", "separated", "n_dropped_nonfinite")
TIER = "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)"
QUESTION = ("paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, models FIXED: 'another draw of EPISODES' "
            "only; NOT another training run (single training seed), NOT another inference run (seed 1 is the replicate beside seed 0), "
            "NOT another fold assignment (fold-seed sensitivity reported separately)")
STATUS = ("EARLY READ of banked step 30,000. ARC-LENGTH RE-TIMING (path kept, progress re-timed), motivated AFTER the LON arm failed its pre-registered ACC-LAT "
          "(RS_LON_XF - os separated WORSE on S6 heading / curvature / yaw-rate). E1 AS REGISTERED (os - ha0_ext) stays the bar of record. All arms are cross-fitted "
          "calibrations fit on EVAL episodes: lever estimates, never a bar pass (a deployable version fits g on TRAIN).")


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(obj, p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, p)


# ------------------------------------------------------------------------------------------ pure fitting (pinned by the pytest)
def ade_np(P, G):
    return np.linalg.norm(np.asarray(P, np.float64) - np.asarray(G, np.float64), axis=-1).mean(1)


def _lg_axis(w, rx, ry, dx, dy):
    ex, ey = rx + w[0] * dx, ry + w[1] * dy
    n = np.sqrt(ex * ex + ey * ey)
    s = np.where(n > 0, n, 1.0)
    return float(n.mean()), np.array([float(np.where(n > 0, ex * dx / s, 0.0).mean()), float(np.where(n > 0, ey * dy / s, 0.0).mean())])


def _lg_pos(w, rx, ry, dx, dy):
    ex, ey = rx + w[0] * dx, ry + w[0] * dy
    n = np.sqrt(ex * ex + ey * ey)
    s = np.where(n > 0, n, 1.0)
    return float(n.mean()), np.array([float(np.where(n > 0, (ex * dx + ey * dy) / s, 0.0).mean())])


def _boxes(box, per_axis) -> list:
    """box = (lo, hi) -> that box on both axes; box = ((lo, hi), (lo, hi)) -> (along, cross).  A degenerate cross box (lo == hi) FIXES that gain."""
    if isinstance(box[0], (tuple, list, np.ndarray)):
        if not per_axis:
            raise ValueError("a per-axis box needs per_axis=True")
        return [tuple(float(v) for v in box[0]), tuple(float(v) for v in box[1])]
    b = (float(box[0]), float(box[1]))
    return [b, b] if per_axis else [b]


def _grid_min(rx, ry, dx, dy, box, per_axis):
    bxs = _boxes(box, per_axis)
    g = np.arange(bxs[0][0], bxs[0][1] + 1e-9, VERIFY_STEP)
    gy = np.arange(bxs[1][0], bxs[1][1] + 1e-9, VERIFY_STEP) if per_axis else g
    best, arg = np.inf, None
    for gx in g:
        if per_axis:
            ex2 = (rx + gx * dx) ** 2
            L = np.sqrt(ex2[None, :] + (ry[None, :] + gy[:, None] * dy[None, :]) ** 2).mean(1)
            j = int(np.argmin(L))
            if L[j] < best:
                best, arg = float(L[j]), np.array([gx, gy[j]])
        else:
            v = float(np.sqrt((rx + gx * dx) ** 2 + (ry + gx * dy) ** 2).mean())
            if v < best:
                best, arg = v, np.array([gx])
    return best, arg


def fit_gains(O, B, G, box=BOX_RS, per_axis=True, stats: dict | None = None) -> np.ndarray:
    """[T, 2] weights on O (1 - w on B) minimising mean_n mean_k ||w*O + (1-w)*B - G|| per instant (separable, convex), in `box`."""
    from scipy.optimize import minimize
    O, B, G = (np.asarray(x, np.float64) for x in (O, B, G))
    T = O.shape[1]
    W = np.ones((T, 2))
    if O.shape[0] == 0:
        return W
    lg = _lg_axis if per_axis else _lg_pos
    bxs = _boxes(box, per_axis)
    x0 = np.array([min(max(1.0, b[0]), b[1]) for b in bxs])
    lo_, hi_ = np.array([b[0] for b in bxs]), np.array([b[1] for b in bxs])
    for k in range(T):
        rx, ry = B[:, k, 0] - G[:, k, 0], B[:, k, 1] - G[:, k, 1]
        dx, dy = O[:, k, 0] - B[:, k, 0], O[:, k, 1] - B[:, k, 1]
        opt = dict(ftol=1e-15, gtol=1e-12, maxiter=1000)
        r = minimize(lg, x0, args=(rx, ry, dx, dy), jac=True, method="L-BFGS-B", bounds=bxs, options=opt)
        gmin, garg = _grid_min(rx, ry, dx, dy, box, per_axis)
        w, f = np.clip(r.x, lo_, hi_), lg(np.clip(r.x, lo_, hi_), rx, ry, dx, dy)[0]
        if f > gmin + 1e-12:
            r2 = minimize(lg, garg, args=(rx, ry, dx, dy), jac=True, method="L-BFGS-B", bounds=bxs, options=opt)
            w2 = np.clip(r2.x, lo_, hi_)
            f2 = lg(w2, rx, ry, dx, dy)[0]
            w, f = (w2, f2) if f2 <= gmin else (garg, gmin)
            if stats is not None:
                stats["n_restarts"] = stats.get("n_restarts", 0) + 1
        if stats is not None:
            stats["n_fits"] = stats.get("n_fits", 0) + 1
            stats["max_grid_gap"] = max(stats.get("max_grid_gap", -np.inf), f - gmin)
        W[k] = w if per_axis else (w[0], w[0])
    return W


def blend(O, B, Wn) -> np.ndarray:
    """p = w*O + (1-w)*B in float64, cast to O's dtype.  The convex form makes w = 1 -> O and w = 0 -> B bit-exact."""
    Wn = np.asarray(Wn, np.float64)
    P = Wn * np.asarray(O, np.float64) + (1.0 - Wn) * np.asarray(B, np.float64)
    return P.astype(np.asarray(O).dtype)


def episode_folds(clip, k=K_FOLDS, seed=FOLD_SEED) -> np.ndarray:
    u = np.unique(np.asarray(clip))
    perm = np.random.default_rng(seed).permutation(u.size)
    fold_of = {int(u[p]): r % k for r, p in enumerate(perm)}
    return np.asarray([fold_of[int(c)] for c in clip], dtype=np.int64)


def window_folds(n, k=K_FOLDS, seed=FOLD_SEED) -> np.ndarray:
    return (np.random.default_rng(seed).permutation(n) % k).astype(np.int64)


def v0_bins(v0) -> np.ndarray:
    import s6_lon_decomp as SD
    v = np.asarray(v0, np.float64)
    lab = np.full(v.size, -1, dtype=np.int64)
    for b, (lo, hi, _n) in enumerate(SD.V0_EDGES):
        lab[(v >= lo) & (v < hi)] = b
    if (lab < 0).any():
        raise SystemExit("[a0k0] a v0 fell outside V0_EDGES")
    return lab


def crossfit(O, B, G, folds, groups=None, leak=False, box=BOX_RS, per_axis=True, stats=None) -> tuple[np.ndarray, dict]:
    """Per-window weights [n, T, 2]: fold f gets weights fit on folds != f (leak=True includes fold f: the DELIBERATE REGRESSION).
    groups (int labels, e.g. v0 bin): one fit per group within the fit folds; < MIN_BIN_FIT fit windows -> the fold's ungated fit."""
    n, T = O.shape[0], O.shape[1]
    Wn = np.ones((n, T, 2))
    rec = {}
    fk = dict(box=box, per_axis=per_axis, stats=stats)
    for f in np.unique(folds):
        fit = np.ones(n, bool) if leak else (folds != f)
        app = folds == f
        base = fit_gains(O[fit], B[fit], G[fit], **fk)
        Wn[app] = base
        rec[f"fold{int(f)}"] = {"n_fit": int(fit.sum()), "n_apply": int(app.sum()), "w": np.round(base, 3).tolist()}
        if groups is not None:
            for g in np.unique(groups):
                fm, am = fit & (groups == g), app & (groups == g)
                if fm.sum() >= MIN_BIN_FIT:
                    wg = fit_gains(O[fm], B[fm], G[fm], **fk)
                    rec[f"fold{int(f)}|bin{int(g)}"] = {"n_fit": int(fm.sum()), "n_apply": int(am.sum()), "w": np.round(wg, 3).tolist()}
                else:
                    wg = base
                    rec[f"fold{int(f)}|bin{int(g)}"] = {"n_fit": int(fm.sum()), "n_apply": int(am.sum()), "fallback": "fold ungated weights"}
                Wn[am] = wg
    return Wn, rec


def insample(O, B, G, groups=None, **kw) -> tuple[np.ndarray, dict]:
    return crossfit(O, B, G, np.zeros(O.shape[0], np.int64), groups=groups, leak=True, **kw)


# ------------------------------------------------------------------------------------------ ARC-LENGTH RE-TIMING (pure; pinned by the pytest)
def polyline_geometry(O) -> dict:
    """The os polyline of every window.  Vertices V_0 = (0,0), V_k = O[:, k-1] (k = 1..S); cumulative arc length L [n, S+1] (L[:, 0] = 0, L[:, k] = s_os(t_k));
    unit directions u [n, S, 2] (0 on a zero-length segment); u_end [n, 2] = direction of the LAST POSITIVE-LENGTH segment, +x if the whole polyline has zero
    length (fallback [n] flags those windows).  float64.  Non-finite input fails loud."""
    O = np.asarray(O, np.float64)
    if O.ndim != 3 or O.shape[-1] != 2:
        raise ValueError(f"plans must be [n, S, 2], got {O.shape}")
    if not np.isfinite(O).all():
        raise ValueError("non-finite plan position: the re-timing is undefined")
    n, S, _ = O.shape
    V = np.concatenate([np.zeros((n, 1, 2)), O], axis=1)
    seg = V[:, 1:] - V[:, :-1]
    ln = np.hypot(seg[..., 0], seg[..., 1])
    L = np.concatenate([np.zeros((n, 1)), np.cumsum(ln, axis=1)], axis=1)
    pos = ln > 0
    u = np.where(pos[..., None], seg / np.where(pos, ln, 1.0)[..., None], 0.0)
    last = np.where(pos.any(1), S - 1 - np.argmax(pos[:, ::-1], axis=1), 0)
    u_end = u[np.arange(n), last].copy()
    fb = ~pos.any(1)
    u_end[fb] = (1.0, 0.0)
    return {"V": V, "L": L, "u": u, "u_end": u_end, "fallback": fb}


def _geo_take(geo: dict, m) -> dict:
    return {k: v[m] for k, v in geo.items()}


def eval_polyline(geo: dict, s) -> np.ndarray:
    """C(s): linear interpolation along the polyline at arc length s [n] or [n, m] (s >= 0); at s >= L_S the point is V_S + (s - L_S) * u_end
    (anchored at the END vertex, so s == L_S returns V_S bit for bit).  Returns [n, 2] or [n, m, 2] float64."""
    V, L, u, u_end = geo["V"], geo["L"], geo["u"], geo["u_end"]
    s = np.asarray(s, np.float64)
    one = s.ndim == 1
    if one:
        s = s[:, None]
    n, S = s.shape[0], u.shape[1]
    idx = np.zeros(s.shape, np.int64)
    for j in range(1, S):
        idx += (s >= L[:, j:j + 1])
    rows = np.arange(n)[:, None]
    P = V[rows, idx] + (s - L[rows, idx])[..., None] * u[rows, idx]
    beyond = s >= L[:, S:S + 1]
    Pe = V[:, S:S + 1] + (s - L[:, S:S + 1])[..., None] * u_end[:, None, :]
    P = np.where(beyond[..., None], Pe, P)
    return P[:, 0] if one else P


def retime_progress(L, M, Gn) -> np.ndarray:
    """s' = L + (1 - g)(M - L) = s_prior + g (s_os - s_prior), clipped at >= 0.  L, M = s_os, s_prior [n, T]; Gn [n, T] or [T].  g == 1 -> s' == L bit for bit."""
    Gn = np.broadcast_to(np.asarray(Gn, np.float64), L.shape)
    return np.maximum(L + (1.0 - Gn) * (M - L), 0.0)


def retime_positions(geoO: dict, M, Gn):
    """Float64 re-timed positions [n, T, 2] and the progress s' [n, T]."""
    L = geoO["L"][:, 1:]
    s = retime_progress(L, M, Gn)
    P = np.empty(s.shape + (2,))
    for k in range(s.shape[1]):
        P[:, k] = eval_polyline(geoO, s[:, k])
    return P, s


def prior_arc(B) -> np.ndarray:
    """s_prior(t_k) = cumulative chord arc length of the prior through (0,0) and its points at the S-grid instants [n, T]."""
    return polyline_geometry(B)["L"][:, 1:]


def retime_apply(O, B, Gn, cast=True) -> np.ndarray:
    """p = C_os(s'(g)), cast to O's dtype by default (the same cast as `blend`)."""
    P, _s = retime_positions(polyline_geometry(O), prior_arc(B), Gn)
    return P.astype(np.asarray(O).dtype) if cast else P


def dist_to_path(P, geo: dict, ray=True) -> np.ndarray:
    """Distance from each point [n, T, 2] to the os polyline (union its final extension ray if ray=True) -> [n, T]."""
    V, u_end = geo["V"], geo["u_end"]
    P = np.asarray(P, np.float64)
    S = V.shape[1] - 1
    best = np.full(P.shape[:2], np.inf)
    for j in range(S):
        A, Bv = V[:, j][:, None, :], V[:, j + 1][:, None, :]
        AB = Bv - A
        den = (AB * AB).sum(-1)
        t = np.clip(np.where(den > 0, ((P - A) * AB).sum(-1) / np.where(den > 0, den, 1.0), 0.0), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(P - (A + t[..., None] * AB), axis=-1))
    if ray:
        A, D = V[:, S][:, None, :], u_end[:, None, :]
        t = np.maximum(((P - A) * D).sum(-1), 0.0)
        best = np.minimum(best, np.linalg.norm(P - (A + t[..., None] * D), axis=-1))
    return best


def _rt_loss(geoO, so, sb, Gk, gv) -> np.ndarray:
    """Mean over windows of ||C(s'(g)) - G|| for every g in gv [m] at ONE instant -> [m]."""
    s = np.maximum(so[:, None] + (1.0 - gv)[None, :] * (sb - so)[:, None], 0.0)
    d = eval_polyline(geoO, s) - Gk[:, None, :]
    return np.hypot(d[..., 0], d[..., 1]).mean(0)


def retime_fit_pre(geoO: dict, M, G, box=RT_BOX, step=RT_GRID_STEP, stats: dict | None = None) -> np.ndarray:
    """[T] gains minimising mean_n ||C(s'_k(g)) - G_k|| per instant (separable).  NOT convex: coarse grid (step), then a bounded scalar refinement inside the best
    cell; ties (|dF| <= 1e-12) go to the g closest to 1.  The refinement is accepted only if it beats the grid minimum by > 1e-12."""
    from scipy.optimize import minimize_scalar
    G = np.asarray(G, np.float64)
    T = M.shape[1]
    g = np.ones(T)
    if M.shape[0] == 0:
        return g
    L = geoO["L"][:, 1:]
    gv = np.arange(box[0], box[1] + 1e-9, step)
    for k in range(T):
        F = _rt_loss(geoO, L[:, k], M[:, k], G[:, k], gv)
        near = np.flatnonzero(F <= F.min() + 1e-12)
        i = int(near[np.argmin(np.abs(gv[near] - 1.0))])
        gb, fb = float(gv[i]), float(F[i])
        lo, hi = max(box[0], gv[i] - step), min(box[1], gv[i] + step)
        r = minimize_scalar(lambda x: float(_rt_loss(geoO, L[:, k], M[:, k], G[:, k], np.array([x]))[0]), bounds=(lo, hi), method="bounded", options={"xatol": 1e-9})
        xr = float(np.clip(r.x, box[0], box[1]))
        fr = float(_rt_loss(geoO, L[:, k], M[:, k], G[:, k], np.array([xr]))[0])
        if fr < fb - 1e-12:
            if stats is not None:
                stats["n_refined"] = stats.get("n_refined", 0) + 1
                stats["max_refine_gain_m"] = max(stats.get("max_refine_gain_m", 0.0), fb - fr)
            gb, fb = xr, fr
        if stats is not None:
            stats["n_fits"] = stats.get("n_fits", 0) + 1
        g[k] = gb
    return g


def retime_fit(O, B, G, **kw) -> np.ndarray:
    return retime_fit_pre(polyline_geometry(O), prior_arc(B), G, **kw)


def retime_crossfit(O, B, G, folds, groups=None, leak=False, stats=None) -> tuple[np.ndarray, dict]:
    """Per-window gains [n, T]: fold f gets gains fit on folds != f (leak=True includes fold f: the DELIBERATE REGRESSION).  groups (int labels, e.g. v0 bin):
    one fit per group within the fit folds; < MIN_BIN_FIT fit windows -> the fold's ungated fit.  Mirrors `crossfit`."""
    geoO, M = polyline_geometry(O), prior_arc(B)
    G = np.asarray(G, np.float64)
    n, T = M.shape
    Gn = np.ones((n, T))
    rec = {}
    fk = dict(stats=stats)
    for f in np.unique(folds):
        fit = np.ones(n, bool) if leak else (folds != f)
        app = folds == f
        base = retime_fit_pre(_geo_take(geoO, fit), M[fit], G[fit], **fk)
        Gn[app] = base
        rec[f"fold{int(f)}"] = {"n_fit": int(fit.sum()), "n_apply": int(app.sum()), "g": np.round(base, 4).tolist()}
        if groups is not None:
            for gg in np.unique(groups):
                fm, am = fit & (groups == gg), app & (groups == gg)
                if fm.sum() >= MIN_BIN_FIT:
                    wg = retime_fit_pre(_geo_take(geoO, fm), M[fm], G[fm], **fk)
                    rec[f"fold{int(f)}|bin{int(gg)}"] = {"n_fit": int(fm.sum()), "n_apply": int(am.sum()), "g": np.round(wg, 4).tolist()}
                else:
                    wg = base
                    rec[f"fold{int(f)}|bin{int(gg)}"] = {"n_fit": int(fm.sum()), "n_apply": int(am.sum()), "fallback": "fold ungated gains"}
                Gn[am] = wg
    return Gn, rec


def retime_insample(O, B, G, groups=None, **kw) -> tuple[np.ndarray, dict]:
    return retime_crossfit(O, B, G, np.zeros(np.asarray(O).shape[0], np.int64), groups=groups, leak=True, **kw)


def retime_diag(O, B, Gn) -> dict:
    """Geometry of the re-timed plan: control (b) on-path distance, and the descriptive fractions (NOT bars)."""
    geoO = polyline_geometry(O)
    P, s = retime_positions(geoO, prior_arc(B), Gn)
    d = dist_to_path(P, geoO)
    LS = geoO["L"][:, -1:]
    P32 = P.astype(np.asarray(O).dtype)
    d32 = dist_to_path(P32, geoO)
    bound32 = float(np.finfo(P32.dtype).eps * np.abs(P).max()) if P32.dtype.kind == "f" else float("nan")
    Gb = np.broadcast_to(np.asarray(Gn, np.float64), s.shape)
    return {"n_windows": int(len(s)), "max_dist_to_path_f64_m": float(d.max()), "max_dist_to_path_after_cast_m": float(d32.max()), "float32_rounding_bound_m": bound32,
            "frac_nonmonotone_windows": float((np.diff(s, axis=1) < -1e-12).any(1).mean()) if s.shape[1] > 1 else 0.0,
            "frac_extrapolated_points": float((s > LS + 1e-9).mean()), "frac_fallback_dir_windows": float(geoO["fallback"].mean()),
            "max_abs_g_minus_1": float(np.abs(Gb - 1.0).max()), "mean_abs_g_minus_1": float(np.abs(Gb - 1.0).mean())}


# ------------------------------------------------------------------------------------------ admissible references (library functions)
def imu_path(a0, psi_dot0, v0, horizons, tick=DT):
    """ha0_imu: constant a0 and constant HEADING RATE psi_dot0 from (0, 0, 0, v0), rollout_unicycle's update order (position on the
    start-of-step speed, heading next, v LAST floored at 0), float32, index-selected at horizons - 1 -> [B, S, 2] torch."""
    import torch
    a = torch.as_tensor(a0, dtype=torch.float32).reshape(-1)
    w = torch.as_tensor(psi_dot0, dtype=torch.float32).reshape(-1)
    v = torch.as_tensor(v0, dtype=torch.float32).reshape(-1).clone()
    x = torch.zeros_like(v)
    y = torch.zeros_like(v)
    yaw = torch.zeros_like(v)
    out = []
    for _k in range(max(int(h) for h in horizons)):
        x = x + v * torch.cos(yaw) * tick
        y = y + v * torch.sin(yaw) * tick
        yaw = yaw + w * tick
        v = (v + a * tick).clamp_min(0.0)
        out.append(torch.stack([x, y], -1))
    P = torch.stack(out, 1)
    return P.index_select(1, torch.tensor([int(h) - 1 for h in horizons], dtype=torch.long))


def load_decisions(NR, pdir: str, panel_ws: np.ndarray, panel_v0: np.ndarray) -> dict:
    _m, files, _e = NR.panel_layout(pdir)
    win, pl, et, ws = [], [], [], []
    for f in files:
        dd = np.load(os.path.join(pdir, "decisions", os.path.basename(f)))
        ep, w = dd["ep_poses"], dd["ws"].astype(np.int64)
        idx = w[:, None] + np.arange(-(W_OBS - 1), 1)[None, :]
        if idx.min() < 0 or idx.max() >= ep.shape[0]:
            raise SystemExit(f"[a0k0] {f}: observed window out of the episode's pose range")
        win.append(ep[idx])
        pl.append(dd["pose_last"])
        et.append(dd["ego_t0"])
        ws.append(w)
    d = {"win": np.concatenate(win), "pose_last": np.concatenate(pl), "ego_t0": np.concatenate(et), "ws": np.concatenate(ws)}
    chk = {"ws_eq_panel": bool(np.array_equal(d["ws"], panel_ws)),
           "ep_poses_at_ws_eq_pose_last": bool(np.array_equal(d["win"][:, -1], d["pose_last"])),
           "pose_v0_eq_panel_v0": bool(np.array_equal(d["win"][:, -1, 3].astype(np.float32), np.asarray(panel_v0, np.float32))),
           "ego_t0_v0_eq_pose_v0": bool(np.array_equal(d["ego_t0"][:, 0].astype(np.float32), d["win"][:, -1, 3].astype(np.float32))),
           "dtype_ep_poses": str(d["win"].dtype)}
    chk["PASS"] = bool(chk["ws_eq_panel"] and chk["ep_poses_at_ws_eq_pose_last"] and chk["pose_v0_eq_panel_v0"] and chk["ego_t0_v0_eq_pose_v0"])
    d["chk"] = chk
    return d


def references(dec: dict, HX: np.ndarray, horizons) -> tuple[dict, dict]:
    import torch
    from tanitad.models import kinematic_prior as KP
    from tanitad.models.ego_history import ego_channels_from_poses
    pw = torch.from_numpy(np.ascontiguousarray(dec["win"], dtype=np.float32))
    a0, kp = KP.prior_controls("ha0_ext_pose", pw)
    ch = ego_channels_from_poses(pw, W_OBS, dt=DT)
    v0, om = ch[:, -1, 0], ch[:, -1, 2]
    k_steer = torch.from_numpy(np.ascontiguousarray(dec["ego_t0"][:, 3], dtype=np.float32))
    ax = torch.from_numpy(np.ascontiguousarray(dec["ego_t0"][:, 1], dtype=np.float32))
    hz = [int(h) for h in horizons]
    R = {"HP": KP.prior_path(a0, kp, v0, hz).numpy().astype(np.float32),
         "HI": imu_path(a0, om, v0, hz).numpy().astype(np.float32),
         "HIax": imu_path(ax, om, v0, hz).numpy().astype(np.float32)}
    hx_rebuilt = KP.prior_path(a0, k_steer, v0, hz).numpy()
    z = torch.zeros_like(a0)
    v2_imu = imu_path(z, k_steer * v0, v0, hz).numpy()
    v2_ref = KP.prior_path(z, k_steer, v0, hz).numpy()
    d_ip = np.linalg.norm(R["HI"].astype(np.float64) - HX.astype(np.float64), axis=-1)
    d_pp = np.linalg.norm(R["HP"].astype(np.float64) - HX.astype(np.float64), axis=-1)
    at = {f"{h / 10:g}s": i for i, h in enumerate(hz) if h in (20, 60)}
    V = {"V1_ha0_ext_rebuilt_vs_panel_max_abs_m": float(np.abs(hx_rebuilt.astype(np.float64) - HX.astype(np.float64)).max()),
         "V1_frac_windows_bit_equal": float(np.mean(np.all(hx_rebuilt.reshape(len(hx_rebuilt), -1) == HX.reshape(len(HX), -1), axis=1))),
         "V2_imu_with_psidot_kappa0v0_a0zero_vs_unicycle_kappa0_max_abs_m": float(np.abs(v2_imu.astype(np.float64) - v2_ref.astype(np.float64)).max()),
         "V2_lateral_max_abs_m": float(np.abs(v2_imu[..., 1].astype(np.float64) - v2_ref[..., 1].astype(np.float64)).max()),
         "V3_frac_imu_vs_ha0_ext_gt_0p1m": {k: float((d_ip[:, i] > 0.1).mean()) for k, i in at.items()},
         "V3_frac_pose_vs_ha0_ext_gt_0p1m": {k: float((d_pp[:, i] > 0.1).mean()) for k, i in at.items()},
         "ha0_ext_pose_vs_ha0_ext_max_abs_m": float(np.abs(R["HP"].astype(np.float64) - HX.astype(np.float64)).max()),
         "ha0_ext_pose_vs_ha0_ext_mean_abs_m": float(np.abs(R["HP"].astype(np.float64) - HX.astype(np.float64)).mean()),
         "ha0_ext_pose_frac_windows_bit_equal_ha0_ext": float(np.mean(np.all(R["HP"].reshape(len(HX), -1) == HX.reshape(len(HX), -1), axis=1))),
         "a0_backward_vs_measured_ax": {"median_abs_ms2": float(np.median(np.abs(a0.numpy() - ax.numpy()))), "p90_abs_ms2": float(np.quantile(np.abs(a0.numpy() - ax.numpy()), 0.9))},
         "psidot0_vs_kappa0steer_v0": {"median_abs_radps": float(np.median(np.abs(om.numpy() - (k_steer * v0).numpy()))),
                                       "p90_abs_radps": float(np.quantile(np.abs(om.numpy() - (k_steer * v0).numpy()), 0.9))}}
    V["PASS"] = bool(V["V1_ha0_ext_rebuilt_vs_panel_max_abs_m"] <= V1_TOL_M and V["V2_imu_with_psidot_kappa0v0_a0zero_vs_unicycle_kappa0_max_abs_m"] <= V2_TOL_M)
    return R, V


# ------------------------------------------------------------------------------------------ reads
def tools_load() -> dict:
    import clamp_posthoc_r8 as CP
    NR, BP, PR, ra, sha12 = CP.get_readers()
    return {"NR": NR, "BP": BP, "PR": PR, "ra": ra, "sha12": sha12, "V": None, "rc": CP.run_cfg()}


def cells_equal(mine: dict, banked: dict) -> dict:
    diff = {k: (mine.get(k), banked.get(k)) for k in CELL_KEYS if k in banked and mine.get(k) != banked.get(k)}
    missing = [k for k in ("delta", "lo", "hi", "n_windows", "n_episodes", "separated") if k not in banked]
    return {"exact": (not diff) and (not missing), "diff": diff, "banked_missing": missing}


def slim(c: dict) -> dict:
    return {k: c.get(k) for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "n_dropped_nonfinite", "mean_new", "mean_ref")}


ARMS = {  # name: (kind, fold kind, gated, full read)      kind "rs" = the Amendment-2 L-BFGS machinery (control); "rt" = the arc-length re-timing
    "RS_XF": ("rs", "ep", False, False),             # CONTROL: must reproduce the banked A0K0_HEADROOM.json RS_XF ADE cells
    "RETIME_XF": ("rt", "ep", False, True),
    "GATED_RETIME_XF": ("rt", "ep", True, True),
    "RETIME_IS": ("rt", "is", False, False),
    "GATED_RETIME_IS": ("rt", "is", True, False),
    "RETIME_WF": ("rt", "wf", False, False),
    "GATED_RETIME_WF": ("rt", "wf", True, False),
}
PRIMARY_ARMS = ("RETIME_XF", "GATED_RETIME_XF")      # the arms the bars / acceptance read
RT_ARMS = tuple(n for n, v in ARMS.items() if v[0] == "rt")
LATERAL_KEYS = ("LAT_heading_mae_deg", "LAT_curvature_mae_1pm_valid", "LAT_yaw_rate_mae_radps_valid", "LAT_cross_mae_m", "LAT_cross_abs_final_m")


def cells_close(mine: dict, banked: dict, tol=1e-9) -> bool:
    return all(abs(float(mine[k]) - float(banked[k])) <= tol for k in ("delta", "lo", "hi")) and all(mine[k] == banked[k] for k in ("n_windows", "n_episodes"))


def read_set(tools, X: dict, eid, clip, v0, dt, banked_set: dict, log, bk_arms: dict | None = None, sens: bool = False) -> dict:
    """Every reference, arm and control on ONE (tree, seed, surface, window set).  X = {O, HX, HP, HI, HIax, G, R}."""
    PR, ra = tools["PR"], tools["ra"]
    keys = [m[1] for m in PR.METRICS]
    G = X["G"]
    comp = lambda P: PR.components(ra, P, G, dt)  # noqa: E731
    C = {k: comp(X[k]) for k in ("O", "HX", "HP", "HI", "HIax", "R")}
    cell = lambda a, b, k="ade_m": slim(PR.paired_cell(a[k], b[k], eid, N_BOOT))  # noqa: E731
    out = {"n_windows": int(len(eid)), "n_episodes": int(len(np.unique(eid))),
           "ade_means": {n: float(np.nanmean(C[k]["ade_m"])) for n, k in (("os", "O"), ("ha0_ext", "HX"), ("ha0_ext_pose", "HP"), ("ha0_imu", "HI"),
                                                                           ("ha0_imu_ax", "HIax"), ("refcv7_50400", "R"))},
           "controls": {}, "refs": {}, "arms": {}, "fit_stats": {}}
    T = X["O"].shape[1]
    # (a) g = 1 reproduces the banked os cells EXACTLY (blend machinery AND the re-timing machinery)
    P1 = blend(X["O"], X["HP"], np.ones((T, 2)))
    c1 = comp(P1)
    a = {"bit_exact_vs_os": bool(np.array_equal(P1, X["O"])), "rows": {}}
    a["rows"]["ade_m|ha0_ext"] = cells_equal(PR.paired_cell(c1["ade_m"], C["HX"]["ade_m"], eid, N_BOOT), banked_set["rows"]["ade_m"]["os_minus_ha0_ext"])
    for k in keys:
        a["rows"][f"{k}|refcv7"] = cells_equal(PR.paired_cell(c1[k], C["R"][k], eid, N_BOOT), banked_set["rows"][k]["os_minus_refcv7_50400"])
    a["n_rows"], a["n_exact"] = len(a["rows"]), int(sum(v["exact"] for v in a["rows"].values()))
    Pr1 = retime_apply(X["O"], X["HP"], np.ones((len(eid), T)))
    Pr1f = retime_apply(X["O"], X["HP"], np.ones((len(eid), T)), cast=False)
    a["retime_g1_bit_exact_vs_os"] = bool(np.array_equal(Pr1, X["O"]))
    a["retime_g1_max_abs_diff_f64_m"] = float(np.abs(Pr1f - np.asarray(X["O"], np.float64)).max())
    a["PASS"] = bool(a["bit_exact_vs_os"] and a["n_exact"] == a["n_rows"] and a["retime_g1_bit_exact_vs_os"] and a["retime_g1_max_abs_diff_f64_m"] <= 1e-6)
    out["controls"]["a_g1_reproduces_banked_os"] = a
    # (b) g = 0 of the blend machinery is the prior bit for bit; its cell against the prior reads exactly 0 (and w = 0 is ha0_imu for BLEND_imu)
    P0, Q0 = blend(X["O"], X["HP"], np.zeros((T, 2))), blend(X["O"], X["HI"], np.zeros((T, 2)))
    z = PR.paired_cell(comp(P0)["ade_m"], C["HP"]["ade_m"], eid, N_BOOT)
    out["controls"]["b_g0_is_prior"] = {"bit_exact_vs_ha0_ext_pose": bool(np.array_equal(P0, X["HP"])), "bit_exact_w0_vs_ha0_imu": bool(np.array_equal(Q0, X["HI"])),
                                        "cell_vs_prior": slim(z), "PASS": bool(np.array_equal(P0, X["HP"]) and np.array_equal(Q0, X["HI"]) and z["delta"] == 0 and z["lo"] == 0 and z["hi"] == 0)}
    del P1, c1, P0, Q0, Pr1, Pr1f
    log(f"      controls a={a['PASS']} ({a['n_exact']}/{a['n_rows']}) b={out['controls']['b_g0_is_prior']['PASS']}")
    # references -- E1 AS REGISTERED first
    R = out["refs"]
    R["E1_registered_os_minus_ha0_ext"] = cell(C["O"], C["HX"])
    R["os_minus_ha0_ext_pose"] = {k: cell(C["O"], C["HP"], k) for k in keys}
    R["os_minus_ha0_imu"] = cell(C["O"], C["HI"])
    R["os_minus_ha0_imu_ax"] = cell(C["O"], C["HIax"])
    R["ha0_ext_pose_minus_ha0_ext"] = cell(C["HP"], C["HX"])
    R["ha0_imu_minus_ha0_ext"] = cell(C["HI"], C["HX"])
    R["ha0_imu_minus_ha0_ext_pose"] = cell(C["HI"], C["HP"])
    log(f"      E1reg {R['E1_registered_os_minus_ha0_ext']['delta']:+.4f}  os-pose {R['os_minus_ha0_ext_pose']['ade_m']['delta']:+.4f} "
        f"[{R['os_minus_ha0_ext_pose']['ade_m']['lo']:+.4f}, {R['os_minus_ha0_ext_pose']['ade_m']['hi']:+.4f}]  pose-ext {R['ha0_ext_pose_minus_ha0_ext']['delta']:+.4f}")
    folds, wfolds, vb = episode_folds(clip), window_folds(len(eid)), v0_bins(v0)
    prim_P = {}
    for name, (kind, fk, gated, full) in ARMS.items():
        st = {}
        Bb = X["HP"]
        if kind == "rs":
            kw = dict(groups=vb if gated else None, box=BOX_RS, per_axis=True, stats=st)
            Wn, rec = insample(X["O"], Bb, G, **kw) if fk == "is" else crossfit(X["O"], Bb, G, folds if fk == "ep" else wfolds, **kw)
            P = blend(X["O"], Bb, Wn)
        else:
            kw = dict(groups=vb if gated else None, stats=st)
            Wn, rec = retime_insample(X["O"], Bb, G, **kw) if fk == "is" else retime_crossfit(X["O"], Bb, G, folds if fk == "ep" else wfolds, **kw)
            P = retime_apply(X["O"], Bb, Wn)
        if name in PRIMARY_ARMS:
            prim_P[name] = P
        cB = comp(P)
        arm = {"base": "ha0_ext_pose", "kind": kind, "weights": rec, "mean_ade_m": float(np.nanmean(cB["ade_m"])),
               "minus_ha0_ext": cell(cB, C["HX"]), "minus_ha0_ext_pose": cell(cB, C["HP"]), "minus_ha0_imu": cell(cB, C["HI"])}
        if kind == "rt":    # control (b): the re-timed path lies ON the os polyline; and the geometry diagnostics (NOT bars)
            arm["geometry"] = retime_diag(X["O"], Bb, Wn)
            gm = arm["geometry"]
            gm["on_path_PASS"] = bool(gm["max_dist_to_path_f64_m"] <= RT_ON_PATH_TOL_M and gm["max_dist_to_path_after_cast_m"] <= gm["float32_rounding_bound_m"] + RT_ON_PATH_TOL_M)
        if name == "RS_XF" and bk_arms is not None:   # control (h): the recomputed Amendment-2 arm equals the banked one
            hh = {k: cells_close(arm[k], bk_arms["RS_XF"][k]) for k in ("minus_ha0_ext", "minus_ha0_ext_pose")}
            hh["minus_os_ade"] = cells_close(cell(cB, C["O"]), bk_arms["RS_XF"]["minus_os"]["ade_m"])
            hh["PASS"] = bool(all(hh.values()))
            out["controls"]["h_RS_XF_reproduces_banked"] = hh
            log(f"      control h RS_XF == banked cells: {hh['PASS']}")
        if full:
            arm["NR_minus_refcv7"] = {k: cell(cB, C["R"], k) for k in keys}
            arm["minus_os"] = {k: cell(cB, C["O"], k) for k in keys}
        else:
            arm["minus_os"] = {"ade_m": cell(cB, C["O"])}
        out["arms"][name] = arm
        out["fit_stats"][name] = st
        gtxt = f" g_mean_abs_dev {arm['geometry']['mean_abs_g_minus_1']:.3f} nonmono {arm['geometry']['frac_nonmonotone_windows']:.3f} extrap {arm['geometry']['frac_extrapolated_points']:.3f}" if kind == "rt" else ""
        log(f"      {name:15s} -ext {arm['minus_ha0_ext']['delta']:+.4f} [{arm['minus_ha0_ext']['lo']:+.4f}, {arm['minus_ha0_ext']['hi']:+.4f}]  "
            f"-pose {arm['minus_ha0_ext_pose']['delta']:+.4f}  -os {arm['minus_os']['ade_m']['delta']:+.4f} [{arm['minus_os']['ade_m']['lo']:+.4f}, {arm['minus_os']['ade_m']['hi']:+.4f}]{gtxt}")
        del cB, P, Wn
    # fold-seed sensitivity (points, numpy ADE) for the PRIMARY arms -- EMA-30k, inference seed 0 only (PREREG_RETIME compute protocol)
    aX, aP = ade_np(X["HX"], G).mean(), ade_np(X["HP"], G).mean()
    if sens:
        sn = {}
        for name, gated, sds in (("RETIME_XF", False, SENS_SEEDS), ("GATED_RETIME_XF", True, SENS_SEEDS_GATED)):
            pts = []
            for s in sds:
                Wn, _ = retime_crossfit(X["O"], X["HP"], G, episode_folds(clip, seed=s), groups=vb if gated else None)
                a_ = ade_np(retime_apply(X["O"], X["HP"], Wn), G).mean()
                pts.append((float(a_ - aX), float(a_ - aP)))
            a_ = ade_np(prim_P[name], G).mean()
            sn[name] = {"fold_seeds": list(sds), "minus_ha0_ext_points": [p[0] for p in pts], "minus_ha0_ext_pose_points": [p[1] for p in pts],
                        "minus_ha0_ext_range": [min(p[0] for p in pts), max(p[0] for p in pts)],
                        "minus_ha0_ext_pose_range": [min(p[1] for p in pts), max(p[1] for p in pts)],
                        "primary_seed_points": [float(a_ - aX), float(a_ - aP)]}
        out["fold_seed_sensitivity"] = sn
    A = out["arms"]
    d = lambda x, k="minus_ha0_ext_pose": A[x][k]["delta"]  # noqa: E731
    out["controls"]["c_d_optimism"] = {"RETIME_XF_minus_IS": d("RETIME_XF") - d("RETIME_IS"), "GATED_RETIME_XF_minus_IS": d("GATED_RETIME_XF") - d("GATED_RETIME_IS"),
                                       "RETIME_XF_minus_WF": d("RETIME_XF") - d("RETIME_WF"), "GATED_RETIME_XF_minus_WF": d("GATED_RETIME_XF") - d("GATED_RETIME_WF"),
                                       "IS_mean_ade_lt_XF_mean_ade": bool(A["RETIME_IS"]["mean_ade_m"] < A["RETIME_XF"]["mean_ade_m"]),
                                       "WF_mean_ade_le_XF_mean_ade": bool(A["RETIME_WF"]["mean_ade_m"] <= A["RETIME_XF"]["mean_ade_m"] + 1e-9),
                                       "IS_known_sign_le_os": bool(A["RETIME_IS"]["mean_ade_m"] <= out["ade_means"]["os"] + 1e-9
                                                                   and A["GATED_RETIME_IS"]["mean_ade_m"] <= out["ade_means"]["os"] + 1e-9)}
    # (i) the re-timed path lies ON the os polyline for EVERY re-timing arm (gates the write)
    out["controls"]["i_retime_on_os_path"] = {n: {k: A[n]["geometry"][k] for k in ("max_dist_to_path_f64_m", "max_dist_to_path_after_cast_m", "on_path_PASS")} for n in RT_ARMS}
    out["controls"]["i_retime_on_os_path"]["PASS"] = bool(all(A[n]["geometry"]["on_path_PASS"] for n in RT_ARMS))
    return out


def surface_inputs(tools, ctx: dict, surf: str, first: dict, tree: str) -> dict:
    """The arrays of ONE (tree, seed, surface): panel arms + the rebuilt model-free references, asserted identical across seeds."""
    P = ctx["panels"][surf]
    rows, eid, A, dt = P["rows"], P["eid"], P["A"], P["dt"]
    clip, v0, ws = ctx["clip"][rows], ctx["v0"][rows], ctx["ws"][rows]
    hz = [int(h) for h in P["man"]["grid"]["horizons_steps"]]
    dec = load_decisions(tools["NR"], P["dir"], ws, v0)
    REF, V = references(dec, A["ha0_ext"], hz)
    key = np.stack([clip, ws])
    if surf in first:   # model-free references and window order must be identical across inference seeds
        if not (all(np.array_equal(REF[k], first[surf][k]) for k in REF) and np.array_equal(first[surf]["_key"], key)
                and np.array_equal(first[surf]["G"], A["g"]) and np.array_equal(first[surf]["HX"], A["ha0_ext"])):
            raise SystemExit(f"[a0k0] {tree} {surf}: references / GT / window order differ between inference seeds")
    else:
        first[surf] = {**REF, "_key": key, "G": A["g"], "HX": A["ha0_ext"]}
    X = {"O": A["os"], "HX": A["ha0_ext"], "HP": REF["HP"], "HI": REF["HI"], "HIax": REF["HIax"], "G": A["g"], "R": A[ctx["refarm"]]}
    return {"X": X, "eid": eid, "clip": clip, "v0": v0, "ws": ws, "mask": ctx["comp"][rows], "dt": dt, "hz": hz, "dec": dec, "V": V, "dir": P["dir"]}


def _sha_arrays(*arrs) -> str:
    h = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _sha_dir(pdir: str) -> dict:
    import glob
    fs = sorted(glob.glob(os.path.join(pdir, "ep*.npz"))) + sorted(glob.glob(os.path.join(pdir, "decisions", "ep*.npz"))) + [os.path.join(pdir, "manifest.json")]
    h = hashlib.sha256()
    for f in fs:
        h.update(os.path.relpath(f, pdir).replace("\\", "/").encode())
        h.update(hashlib.sha256(Path(f).read_bytes()).digest())
    return {"dir": pdir, "n_files": len(fs), "sha256_of_file_sha256s": h.hexdigest()}


FREEZE_TREE = "step30000ema_unc"
FROZEN_JSON = OUT / "FROZEN_GAINS_RETIME_ema30k.json"


def acc_lat(cells_by_seed: dict) -> dict:
    """ACC-LAT (PREREG_RETIME, identical to PREREG_LON): arm - os on the LATERAL rows, S6.  HOLDS iff NO LATERAL row is separated WORSE (error metric: lo > 0) at EITHER seed.
    cells_by_seed = {seed: {metric_key: cell}}.  A row whose cell carries no interval is UNDETERMINED (HOLDS = None), never counted as clean."""
    rows, bad_any, bad_both, und = {}, [], [], []
    for k in LATERAL_KEYS:
        per = {s: cells_by_seed[s][k] for s in cells_by_seed}
        if any(x.get("lo") is None for x in per.values()):
            und.append(k)
            rows[k] = {"per_seed": {str(s): None for s in per}, "worse_at_seeds": None}
            continue
        w = [s for s, x in per.items() if x["lo"] > 0]
        rows[k] = {"per_seed": {str(s): {kk: x.get(kk) for kk in ("delta", "lo", "hi", "separated")} for s, x in per.items()}, "worse_at_seeds": w}
        if w:
            bad_any.append(k)
        if len(w) == len(per):
            bad_both.append(k)
    return {"HOLDS": (None if und else not bad_any), "separated_worse_at_any_seed": bad_any, "separated_worse_at_both_seeds": bad_both, "undetermined": und, "rows": rows}


def freeze() -> int:
    """FROZEN gains (PREREG_RETIME, same method as Amendment 3): RETIME / GATED_RETIME fit on ALL windows of EMA-30k, both inference seeds POOLED, one table per surface
    grid.  The tables are [T] (RETIME) and per v0 bin [T] (GATED_RETIME); one SCALAR gain per instant."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[a0k0-retime] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    import s6_lon_decomp as SD
    import compliant_reread_r8 as CR
    t0 = time.time()
    tools = tools_load()
    first, per = {}, {s: {} for s in SURFS}
    inputs = {}
    for seed in SEEDS:
        ctx = CR.prepare_seed(FREEZE_TREE, seed, tools, tools["rc"])
        if ctx["sha"] != MASK_SHA:
            raise SystemExit("[a0k0-retime freeze] mask sha mismatch")
        for surf in SURFS:
            S = surface_inputs(tools, ctx, surf, first, FREEZE_TREE)
            if not (S["dec"]["chk"]["PASS"] and S["V"]["PASS"]):
                raise SystemExit(f"[a0k0-retime freeze] {surf} s{seed}: decisions / validation failed")
            per[surf][seed] = S
            inputs[f"{surf}|s{seed}"] = _sha_dir(S["dir"])
        del ctx
    out = {"tool": "a0k0_headroom_retime_r8.py freeze", "generated": time.strftime("%FT%T"), "prereg": "raw/a0k0_retime/PREREG_RETIME.md (frozen-gains section)",
           "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "tree": f"{CR.ARCHIVE}/{FREEZE_TREE}",
           "fit_set": "ALL windows of the surface (all 139 eval episodes), inference seeds 0 and 1 POOLED (each seed's os against the same prior and GT)",
           "fitter": f"coarse grid {RT_GRID_STEP} over {list(RT_BOX)} + bounded scalar refinement per instant (the loss is not convex)",
           "prior": "ha0_ext_pose (kinematic_prior.prior_controls/prior_path)",
           "plan": "p_k = C_os(s'_k), s'_k = s_prior(t_k) + g_k (s_os(t_k) - s_prior(t_k)) clipped >= 0, C_os = polyline through (0,0) and the os points at the surface grid",
           "v0_edges": [list(e) for e in SD.V0_EDGES], "min_bin_fit_unique_windows": MIN_BIN_FIT, "input_files": inputs, "surfaces": {}}
    for surf in SURFS:
        s0, s1 = per[surf][0], per[surf][1]
        HP, G, v0 = s0["X"]["HP"], s0["X"]["G"], s0["v0"]
        vb = v0_bins(v0)
        rec = {"instants_s": [h / 10 for h in s0["hz"]], "horizons_steps": s0["hz"], "n_windows_unique": int(len(v0)),
               "n_episodes": int(len(np.unique(s0["eid"]))),
               "input_sha256": _sha_arrays(s0["X"]["O"], s1["X"]["O"], HP, G, np.asarray(v0, np.float32))}
        for set_name in ("all", "compliant_info_only"):
            sel = np.ones(len(v0), bool) if set_name == "all" else s0["mask"]
            O2 = np.concatenate([s0["X"]["O"][sel], s1["X"]["O"][sel]])
            B2, G2, vb2 = np.concatenate([HP[sel], HP[sel]]), np.concatenate([G[sel], G[sel]]), np.concatenate([vb[sel], vb[sel]])
            st = {}
            geo2, M2 = polyline_geometry(O2), prior_arc(B2)
            rt = retime_fit_pre(geo2, M2, G2, stats=st)
            gated = {}
            for b, (_lo, _hi, name) in enumerate(SD.V0_EDGES):
                nu = int((vb[sel] == b).sum())
                if nu >= MIN_BIN_FIT:
                    mb = vb2 == b
                    gated[name] = {"bin": b, "n_fit_unique": nu, "g": retime_fit_pre(_geo_take(geo2, mb), M2[mb], G2[mb], stats=st).tolist()}
                else:
                    gated[name] = {"bin": b, "n_fit_unique": nu, "g": rt.tolist(), "fallback": "RETIME table"}
            Wg = np.stack([np.asarray(gated[n]["g"]) for _a, _b, n in SD.V0_EDGES])[vb2]
            rec[set_name] = {"n_fit_rows_pooled": int(len(O2)), "RETIME": rt.tolist(), "GATED_RETIME": gated, "fit_stats": st,
                             "in_sample_pooled_mean_ade": {"os": float(ade_np(O2, G2).mean()), "ha0_ext_pose": float(ade_np(B2, G2).mean()),
                                                           "RETIME": float(ade_np(retime_apply(O2, B2, rt), G2).mean()),
                                                           "GATED_RETIME": float(ade_np(retime_apply(O2, B2, Wg), G2).mean())}}
            if not (np.all(np.asarray(rt) >= RT_BOX[0]) and np.all(np.asarray(rt) <= RT_BOX[1])):
                raise SystemExit("[a0k0-retime freeze] a gain is outside its box")
        out["surfaces"][surf] = rec
    out["wall_s"] = round(time.time() - t0, 1)
    jdump(out, FROZEN_JSON)
    print(f"WROTE {FROZEN_JSON} ({out['wall_s']} s)")
    for surf in SURFS:
        print(surf, "RETIME g", np.round(np.asarray(out["surfaces"][surf]["all"]["RETIME"]), 3).tolist(), out["surfaces"][surf]["all"]["in_sample_pooled_mean_ade"])
    return 0


def frozen_weights(fz: dict, surf: str, v0) -> dict:
    import s6_lon_decomp as SD
    t = fz["surfaces"][surf]["all"]
    vb = v0_bins(v0)
    tab = np.stack([np.asarray(t["GATED_RETIME"][n]["g"], np.float64) for _a, _b, n in SD.V0_EDGES])
    return {"RETIME_FROZEN30k": np.asarray(t["RETIME"], np.float64), "GATED_RETIME_FROZEN30k": tab[vb]}


def forward(tree: str) -> int:
    """Apply the FROZEN EMA-30k RETIME gains, unchanged, to `tree`; bars FZR-E1 / FZR-E1P / FZR-E3 / FZR-NR / FZR-LAT."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[a0k0-retime] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    import compliant_reread_r8 as CR
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)  # noqa: E731
    fz = jload(FROZEN_JSON)
    tools = tools_load()
    PR = tools["PR"]
    keys = [m[1] for m in PR.METRICS]
    tjp = BANKED.parent / "compliant_reread" / "trees" / f"{tree}.json"
    tj = jload(str(tjp)) if tjp.exists() else None
    res = {"tool": "a0k0_headroom_retime_r8.py forward", "generated": time.strftime("%FT%T"), "tree": f"{CR.ARCHIVE}/{tree}", "tier": TIER, "estimator": QUESTION,
           "prereg": "raw/a0k0_retime/PREREG_RETIME.md (frozen-gains section)", "frozen_gains": str(FROZEN_JSON), "frozen_gains_sha256": hashlib.sha256(FROZEN_JSON.read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "role": ("IN-SAMPLE path check (the gains were fit on this tree)" if tree == FREEZE_TREE else
                    "path check on a different checkpoint of the SAME run (not an independent test, not a bar)" if tree == "step30000_unc" else
                    "FORWARD TEST (PREREG_RETIME bars FZR-*)"),
           "seeds": {}, "controls": {}}
    ok, first = True, {}
    for seed in SEEDS:
        ctx = CR.prepare_seed(tree, seed, tools, tools["rc"])
        sr = {"mask_sha": ctx["sha"], "mask_sha_equal_30k": ctx["sha"] == MASK_SHA, "surfaces": {}}
        for surf in SURFS:
            S = surface_inputs(tools, ctx, surf, first, tree)
            ok &= S["dec"]["chk"]["PASS"] and S["V"]["PASS"]
            fw = frozen_weights(fz, surf, S["v0"])
            ss = {"validation": S["V"], "decisions_check": S["dec"]["chk"], "sets": {}}
            for set_name in SETS:
                sel = S["mask"] if set_name == "compliant" else np.ones(len(S["eid"]), bool)
                X = {k: v[sel] for k, v in S["X"].items()}
                e = S["eid"][sel]
                comp = lambda P: PR.components(tools["ra"], P, X["G"], S["dt"])  # noqa: E731
                C = {k: comp(X[k]) for k in ("O", "HX", "HP", "R")}
                cell = lambda a, b, k="ade_m": slim(PR.paired_cell(a[k], b[k], e, N_BOOT))  # noqa: E731
                r = {"n_windows": int(sel.sum()), "n_episodes": int(len(np.unique(e))),
                     "E1_registered_os_minus_ha0_ext": cell(C["O"], C["HX"]), "os_minus_ha0_ext_pose": cell(C["O"], C["HP"]), "arms": {}}
                if tj is not None:
                    bk = tj["seeds"][str(seed)]["surfaces"][surf]["compliant" if set_name == "compliant" else "all_window"]["uncapped"]
                    r["control_os_cell_eq_banked"] = cells_equal(PR.paired_cell(C["O"]["ade_m"], C["HX"]["ade_m"], e, N_BOOT), bk["rows"]["ade_m"]["os_minus_ha0_ext"])
                    ok &= r["control_os_cell_eq_banked"]["exact"]
                for arm, W in fw.items():
                    Wn = W if W.ndim == 1 else W[sel]
                    Pm = retime_apply(X["O"], X["HP"], Wn)
                    cB = comp(Pm)
                    r["arms"][arm] = {"mean_ade_m": float(np.nanmean(cB["ade_m"])), "minus_ha0_ext": cell(cB, C["HX"]), "minus_ha0_ext_pose": cell(cB, C["HP"]),
                                      "NR_minus_refcv7": {k: cell(cB, C["R"], k) for k in keys}, "minus_os": {k: cell(cB, C["O"], k) for k in keys},
                                      "geometry": retime_diag(X["O"], X["HP"], Wn)}
                    gm = r["arms"][arm]["geometry"]
                    gm["on_path_PASS"] = bool(gm["max_dist_to_path_f64_m"] <= RT_ON_PATH_TOL_M and gm["max_dist_to_path_after_cast_m"] <= gm["float32_rounding_bound_m"] + RT_ON_PATH_TOL_M)
                    ok &= gm["on_path_PASS"]
                    log(f"  {tree} s{seed} {surf} {set_name} {arm}: -ext {c(r['arms'][arm]['minus_ha0_ext'])}  -pose {c(r['arms'][arm]['minus_ha0_ext_pose'])}")
                ss["sets"][set_name] = r
            sr["surfaces"][surf] = ss
        res["seeds"][str(seed)] = sr
        del ctx
    bars = {}
    for st in SETS:
        g = lambda s, surf: res["seeds"][str(s)]["surfaces"][surf]["sets"][st]  # noqa: E731
        for arm in fw:
            fl = {surf: abs(g(0, surf)["arms"][arm]["mean_ade_m"] - g(1, surf)["arms"][arm]["mean_ade_m"]) for surf in SURFS}
            nr = {}
            for fam, k, _l, _u, _lower, surfs_ in PR.METRICS:
                for surf in surfs_:
                    nr[f"{surf}:{k}"] = {"family": fam, **PR.bar_NR_metric({s: g(s, surf)["arms"][arm]["NR_minus_refcv7"][k] for s in SEEDS})}
            bars[f"{st}|{arm}"] = {"arm_floor": fl,
                                   "FZR-E1": PR.bar_E({s: g(s, "S2")["arms"][arm]["minus_ha0_ext"] for s in SEEDS}, fl["S2"]),
                                   "FZR-E1P": PR.bar_E({s: g(s, "S2")["arms"][arm]["minus_ha0_ext_pose"] for s in SEEDS}, fl["S2"]),
                                   "FZR-E3": PR.bar_E({s: g(s, "S6")["arms"][arm]["minus_ha0_ext"] for s in SEEDS}, fl["S6"]),
                                   "FZR-NR": {"metrics": nr, "per_family": PR.rollup_NR(nr, {k: v["family"] for k, v in nr.items()})},
                                   "FZR-LAT": acc_lat({s: {k: g(s, "S6")["arms"][arm]["minus_os"][k] for k in LATERAL_KEYS} for s in SEEDS})}
    res["bars"] = bars
    res["controls"]["ALL_PASS"] = bool(ok)
    res["wall_s"] = round(time.time() - t0, 1)
    p = OUT / f"FORWARD_RETIME_{tree}.json"
    jdump(res, p)
    for k, v in bars.items():
        log(f"{k}: FZR-E1 {v['FZR-E1']['verdict']} | FZR-E1P {v['FZR-E1P']['verdict']} | FZR-E3 {v['FZR-E3']['verdict']} | FZR-NR {v['FZR-NR']['per_family']} | "
            f"FZR-LAT holds={v['FZR-LAT']['HOLDS']} worse={v['FZR-LAT']['separated_worse_at_any_seed']}")
    log(f"WROTE {p} controls={ok}")
    return 0 if ok else 3


def compute(trees, seeds, surfs, sets_) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[a0k0-retime] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    import compliant_reread_r8 as CR
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)  # noqa: E731
    tools = tools_load()
    PR = tools["PR"]
    ref0926 = jload(REF0926)
    bk = jload(str(BANKED / "A0K0_HEADROOM.json"))
    res = {"tool": "a0k0_headroom_retime_r8.py", "generated": time.strftime("%FT%T"), "tier": TIER, "estimator": QUESTION, "status": STATUS,
           "n_boot": N_BOOT, "k_folds": K_FOLDS, "fold_seed": FOLD_SEED, "min_bin_fit": MIN_BIN_FIT,
           "boxes": {"RS_XF_control": BOX_RS, "RETIME": RT_BOX},
           "fitter": {"RETIME": f"coarse grid {RT_GRID_STEP} over {list(RT_BOX)} + bounded scalar refinement per instant (non-convex loss)",
                      "RS_XF_control": f"L-BFGS-B (analytic gradient) per instant x axis, verified on a {VERIFY_STEP} grid"}, "prereg": str(PREREG),
           "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "parent_deliverable": {"path": str(BANKED / "A0K0_HEADROOM.json"), "sha256": hashlib.sha256((BANKED / "A0K0_HEADROOM.json").read_bytes()).hexdigest()},
           "argv_refcv8": "E:/refcv8_eval_kit/ckpt/config.json argv carries --ego-history and --residual-prior ha0_ext_pose (verified in the parent deliverable)",
           "ref_2026_09_26": {"path": REF0926, "S2_P_ha0_ext_pose_minus_echo": ref0926["B_ade_0_2s"]["P_ha0_ext_pose_minus_echo"]},
           "metrics": [list(m[:5]) for m in PR.METRICS], "strategic": "NOT APPLICABLE, n = 0 (the run argv carries --no-strategic: the layer is off)",
           "trees": {}}
    ok = True
    for tree in trees:
        tj = jload(str(PKG / "raw" / "compliant_reread" / "trees" / f"{tree}.json"))
        if not tj["controls"]["ALL_PASS"]:
            raise SystemExit(f"[a0k0-retime] {tree}: compliant reread controls not ALL_PASS -- refusing")
        tr = {"archive": f"{CR.ARCHIVE}/{tree}", "seeds": {}}
        first = {}
        for seed in seeds:
            log(f"{tree} seed {seed}: loading")
            ctx = CR.prepare_seed(tree, seed, tools, tools["rc"])
            sd = tj["seeds"][str(seed)]
            sr = {"mask_sha_equal": ctx["sha"] == MASK_SHA, "K_capped_eq_uncapped_on_compliant": sd["n_changed_plans_on_compliant_windows"] == 0,
                  "refarm": ctx["refarm"], "surfaces": {}}
            ok &= sr["mask_sha_equal"] and sr["K_capped_eq_uncapped_on_compliant"]
            for surf in surfs:
                S = surface_inputs(tools, ctx, surf, first, tree)
                X_all, eid, clip, v0, mk, dt, hz, dec, V = (S[k] for k in ("X", "eid", "clip", "v0", "mask", "dt", "hz", "dec", "V"))
                ok &= dec["chk"]["PASS"] and V["PASS"]
                ss = {"n_panel_windows": int(len(eid)), "dt_s": dt, "horizons_steps": hz, "decisions_check": dec["chk"], "validation": V, "sets": {}}
                log(f"  {tree} s{seed} {surf}: V1 {V['V1_ha0_ext_rebuilt_vs_panel_max_abs_m']:.2e}  dec {dec['chk']['PASS']}")
                for set_name in sets_:
                    sel = mk if set_name == "compliant" else np.ones(len(eid), bool)
                    banked = sd["surfaces"][surf]["compliant" if set_name == "compliant" else "all_window"]["uncapped"]
                    try:
                        bka = bk["trees"][tree]["seeds"][str(seed)]["surfaces"][surf]["sets"][set_name]["arms"]
                    except KeyError:
                        bka = None
                    log(f"  {tree} s{seed} {surf} {set_name}: n {int(sel.sum())}")
                    r = read_set(tools, {k: v[sel] for k, v in X_all.items()}, eid[sel], clip[sel], v0[sel], dt, banked, log, bk_arms=bka,
                                 sens=bool(tree == "step30000ema_unc" and seed == 0 and set_name == "compliant"))
                    ok &= (r["controls"]["a_g1_reproduces_banked_os"]["PASS"] and r["controls"]["b_g0_is_prior"]["PASS"]
                           and r["controls"]["i_retime_on_os_path"]["PASS"]
                           and bool(r["controls"].get("h_RS_XF_reproduces_banked", {}).get("PASS", False)))
                    ss["sets"][set_name] = r
                sr["surfaces"][surf] = ss
            tr["seeds"][str(seed)] = sr
            del ctx
        tr["summary"] = summarise(PR, tr, surfs, sets_, seeds)
        res["trees"][tree] = tr
        OUT.mkdir(parents=True, exist_ok=True)
        jdump(res, OUT / "A0K0_RETIME.partial.json")
    rep = None
    for tree in trees:
        try:
            rep = res["trees"][tree]["seeds"][str(seeds[0])]["surfaces"]["S2"]["sets"]["all"]["refs"]["ha0_ext_pose_minus_ha0_ext"]
            break
        except KeyError:
            continue
    want = res["ref_2026_09_26"]["S2_P_ha0_ext_pose_minus_echo"]
    res["controls_summary"] = {"R0926_S2_all_ha0_ext_pose_minus_ha0_ext": rep, "R0926_banked": want,
                               "R0926_exact": bool(rep is not None and all(rep.get(k) == want.get(k) for k in ("delta", "lo", "hi", "n_windows", "n_episodes")))}
    res["controls_summary"]["ALL_PASS"] = bool(ok)
    res["controls_summary"]["note"] = ("a (g1 == banked os, every row exact; the re-timing at g == 1 is bit-equal to os), b (blend g0 == ha0_ext_pose bit-exact, 0-cell), "
                                       "h (recomputed RS_XF == banked A0K0_HEADROOM.json RS_XF cells, 1e-9), i (every re-timed point lies ON the os polyline: "
                                       "float64 distance <= 1e-6 m), decisions (ws, ep_poses[ws] == pose_last, v0), V1 (ha0_ext rebuilt <= 1e-3 m), mask sha, K")
    res["wall_s"] = round(time.time() - t0, 1)
    full = tuple(trees) == TREES and tuple(seeds) == SEEDS and tuple(surfs) == SURFS and tuple(sets_) == SETS
    if not full:
        res["SMOKE"] = "subset run -- not the deliverable"
        jdump(res, OUT / "A0K0_RETIME.smoke.json")
        log(f"WROTE smoke ok={ok} R0926_exact={res['controls_summary']['R0926_exact']}")
        return 0 if ok else 3
    if not ok:
        jdump(res, OUT / "CONTROL_FAILED_RETIME.json")
        print("CONTROL FAILED")
        return 3
    jdump(res, OUT / "A0K0_RETIME.json")
    try:
        os.remove(OUT / "A0K0_RETIME.partial.json")
    except OSError:
        pass
    render()
    log(f"WROTE {OUT / 'A0K0_RETIME.json'}")
    return 0


def acc_verdict(a: dict, moved: bool) -> str:
    """ACC-LAT label: FAILS / UNDETERMINED / HOLDS (arm moved vs os on S6 ADE, separated better at both seeds) / HOLDS VACUOUSLY (PREREG_RETIME vacuity guard)."""
    if a["HOLDS"] is None:
        return "UNDETERMINED"
    if not a["HOLDS"]:
        return "FAILS"
    return "HOLDS" if moved else "HOLDS VACUOUSLY (the arm did not move S6 ADE vs os, separated at both seeds)"


def summarise(PR, tr, surfs, sets_, seeds) -> dict:
    """Verdict FORMS (never bar verdicts for the arms) with the floors on the same set, and the ACC-LAT acceptance condition."""
    out = {}
    if len(seeds) < 2:
        return {"note": "fewer than 2 inference seeds"}
    for surf in surfs:
        for st in sets_:
            g = lambda s: tr["seeds"][str(s)]["surfaces"][surf]["sets"][st]  # noqa: E731
            fl_os = abs(g(0)["ade_means"]["os"] - g(1)["ade_means"]["os"])
            row = {"os_floor": fl_os,
                   "E1_registered_verdict": PR.bar_E({s: g(s)["refs"]["E1_registered_os_minus_ha0_ext"] for s in seeds}, fl_os)["verdict"]}
            for arm in PRIMARY_ARMS:
                fl = abs(g(0)["arms"][arm]["mean_ade_m"] - g(1)["arms"][arm]["mean_ade_m"])
                nr = {k: {"family": fam, **PR.bar_NR_metric({s: g(s)["arms"][arm]["NR_minus_refcv7"][k] for s in seeds})}
                      for fam, k, _l, _u, _lo, surfs_ in PR.METRICS if surf in surfs_}

                def rows_where(block, worse, arm=arm, g=g):
                    """direction-aware over EVERY METRICS row: an error metric is worse when lo > 0, an accuracy metric when hi < 0."""
                    res_ = []
                    for fam, k, _l, _u, lower, *_r in PR.METRICS:
                        cs = [g(s)["arms"][arm][block][k] for s in seeds]
                        if all(((x["lo"] > 0) if (lower == worse) else (x["hi"] < 0)) for x in cs):
                            res_.append(f"{k} ({fam})")
                    return res_
                moved = all((g(s)["arms"][arm]["minus_os"]["ade_m"]["delta"] < 0 and g(s)["arms"][arm]["minus_os"]["ade_m"]["hi"] < 0) for s in seeds)
                al = acc_lat({s: {k: g(s)["arms"][arm]["minus_os"][k] for k in LATERAL_KEYS} for s in seeds})
                row[arm] = {"arm_floor": fl,
                            "LE_E1_form_minus_ha0_ext": PR.bar_E({s: g(s)["arms"][arm]["minus_ha0_ext"] for s in seeds}, fl)["verdict"],
                            "LE_E1P_form_minus_ha0_ext_pose": PR.bar_E({s: g(s)["arms"][arm]["minus_ha0_ext_pose"] for s in seeds}, fl)["verdict"],
                            "LE_OS_form_minus_os_ade": PR.bar_E({s: g(s)["arms"][arm]["minus_os"]["ade_m"] for s in seeds}, fl_os)["verdict"],
                            "LE_NR_rollup_headline_rows": PR.rollup_NR(nr, {k: v["family"] for k, v in nr.items()}),
                            "vs_refcv7_separated_worse_both_seeds_all_rows": rows_where("NR_minus_refcv7", True),
                            "vs_refcv7_separated_better_both_seeds_all_rows": rows_where("NR_minus_refcv7", False),
                            "vs_os_separated_worse_both_seeds_all_rows": rows_where("minus_os", True),
                            "vs_os_separated_better_both_seeds_all_rows": rows_where("minus_os", False),
                            "moved_vs_os_ADE_separated_better_both_seeds": bool(moved),
                            "ACC_LAT": al, "ACC_LAT_label": acc_verdict(al, moved)}
            # descriptive: the share of the RS_XF - os ADE gain that the re-timing retains (s0 point ratio; not an estimate with an interval)
            dX, dL = g(0)["arms"]["RS_XF"]["minus_os"]["ade_m"]["delta"], g(0)["arms"]["RETIME_XF"]["minus_os"]["ade_m"]["delta"]
            row["share_of_RS_XF_minus_os_retained_by_RETIME_s0"] = (dL / dX) if dX else None
            out[f"{surf}|{st}"] = row
    return out


# ------------------------------------------------------------------------------------------ markdown
def c(x, dp=4):
    if not isinstance(x, dict) or x.get("delta") is None:
        return "n/a"
    s = " **SEP**" if x.get("separated") else ""
    return f"{x['delta']:+.{dp}f} [{x['lo']:+.{dp}f}, {x['hi']:+.{dp}f}]{s}"


def fold_mean(rec: dict):
    ws = [np.asarray(v["g"]) for k, v in rec.items() if "|" not in k and "g" in v]
    return np.round(np.mean(ws, 0), 3).tolist() if ws else None


def fold_mean_bins(rec: dict) -> dict:
    out = {}
    for b in range(5):
        ws = [np.asarray(v["g"]) for k, v in rec.items() if k.endswith(f"|bin{b}") and "g" in v]
        out[f"bin{b}"] = np.round(np.mean(ws, 0), 3).tolist() if ws else "fallback"
    return out


def render_frozen_forward() -> list:
    import glob
    L = []
    if FROZEN_JSON.exists():
        fz = jload(FROZEN_JSON)
        L += ["## FROZEN gains -- fit on EMA-30k, ALL windows, both seeds pooled", "",
              f"`{FROZEN_JSON.name}` sha256 `{hashlib.sha256(FROZEN_JSON.read_bytes()).hexdigest()[:12]}`; fit set: {fz['fit_set']}.", ""]
        for surf, t in fz["surfaces"].items():
            a = t["all"]
            ip = a["in_sample_pooled_mean_ade"]
            L.append(f"* **{surf}** instants {t['instants_s']} s, n {t['n_windows_unique']} windows / {t['n_episodes']} ep: RETIME g per instant = "
                     f"{np.round(np.asarray(a['RETIME']), 3).tolist()}; in-sample pooled ADE os {ip['os']:.4f} / prior {ip['ha0_ext_pose']:.4f} / RETIME {ip['RETIME']:.4f} / "
                     f"GATED_RETIME {ip['GATED_RETIME']:.4f}")
            for name, v in a["GATED_RETIME"].items():
                L.append(f"  * GATED_RETIME {name} (n {v['n_fit_unique']}{', ' + v['fallback'] if v.get('fallback') else ''}): g {np.round(np.asarray(v['g']), 3).tolist()}")
        L.append("")
    for p in sorted(glob.glob(str(OUT / "FORWARD_RETIME_*.json"))):
        fw = jload(p)
        L += [f"## FORWARD {os.path.basename(fw['tree'])} -- {fw['role']} (controls {fw['controls']['ALL_PASS']})", "",
              "| set / arm | FZR-E1 (-ha0_ext, S2) | FZR-E1P (-ha0_ext_pose, S2) | FZR-E3 (-ha0_ext, S6) | FZR-NR per family | FZR-LAT (S6 vs os) |", "|---|---|---|---|---|---|"]
        for k, v in fw["bars"].items():
            pe = lambda b: " / ".join(c(v[b]["per_inference_seed"][s]) for s in sorted(v[b]["per_inference_seed"]))  # noqa: E731
            L.append(f"| {k} | {v['FZR-E1']['verdict']}: {pe('FZR-E1')} | {v['FZR-E1P']['verdict']}: {pe('FZR-E1P')} | {v['FZR-E3']['verdict']}: {pe('FZR-E3')} | "
                     f"{v['FZR-NR']['per_family']} | holds={v['FZR-LAT']['HOLDS']}; worse at any seed: {v['FZR-LAT']['separated_worse_at_any_seed'] or 'none'} |")
        L.append("")
    return L


def render() -> int:
    r = jload(OUT / "A0K0_RETIME.json")
    try:
        bkj = jload(str(BANKED / "A0K0_HEADROOM.json"))
    except OSError:
        bkj = None
    cs = r["controls_summary"]
    L = ["# A0K0_RETIME -- ARC-LENGTH RE-TIMING of refcv8's plan (path kept, progress re-timed; zero training, banked panels)", "",
         f"* **Status:** {r['status']}", f"* **Tier:** {r['tier']}", f"* **Estimator:** {r['estimator']}",
         f"* Pre-registration `raw/a0k0_retime/PREREG_RETIME.md` (sha256 `{r['prereg_sha256'][:12]}`, written before any number of the new arms); script sha256 `{r['script_sha256'][:12]}`; "
         f"parent `A0K0_HEADROOM.json` sha256 `{r['parent_deliverable']['sha256'][:12]}`.",
         "* **Disclosure:** this arm was motivated by the LON arm FAILING ACC-LAT (RS_LON_XF - os separated WORSE on S6 heading / curvature / yaw-rate), seen before the registration; the gains are "
         "fit on EVAL episodes (cross-fitted) -- a LEVER estimate, never a bar pass.",
         f"* Controls **ALL_PASS = {cs['ALL_PASS']}** ({cs['note']}). 2026-09-26 reproduction (S2 all windows, ha0_ext_pose - ha0_ext): "
         f"{c(cs['R0926_S2_all_ha0_ext_pose_minus_ha0_ext'])} vs banked {c(cs['R0926_banked'])} -> exact = **{cs['R0926_exact']}**.",
         f"* STRATEGIC family: {r['strategic']}.", ""]
    hd = r["trees"].get("step30000ema_unc", {}).get("summary", {}).get("S6|compliant")
    if hd:
        L += ["## ACCEPTANCE (ACC-LAT): no LATERAL row of arm - os separated WORSE at EITHER seed -- S6, ceiling-compliant, EMA-30k", ""]
        for arm in PRIMARY_ARMS:
            a = hd[arm]["ACC_LAT"]
            L.append(f"* **{arm}: ACC-LAT {hd[arm]['ACC_LAT_label']}** -- separated worse at any seed: "
                     f"{a['separated_worse_at_any_seed'] or 'none'}; at both seeds: {a['separated_worse_at_both_seeds'] or 'none'}")
        L.append("")
    L += render_frozen_forward() + [
        "References: `ha0_ext` = E1 echo (steer kappa0, INADMISSIBLE input); `ha0_ext_pose` = refcv8's own causal prior (admissible). Arms: p = C_os(s'), s' = s_prior + g (s_os - s_prior), "
        "g one scalar per instant, 5-episode-fold cross-fitted. RETIME_XF: one table; GATED_RETIME_XF: per v0 bin; RS_XF = the Amendment-2 arm (control, recomputed and equal to the "
        "banked cells); _IS in-sample, _WF window-fold (leaky) controls.", ""]
    for tree, tr in r["trees"].items():
        L += [f"## {tree}", ""]
        for surf in SURFS:
            for st in SETS:
                try:
                    g = {s: tr["seeds"][s]["surfaces"][surf]["sets"][st] for s in ("0", "1")}
                except KeyError:
                    continue
                m = g["0"]["ade_means"]
                lbl_e = "LE-E1" if surf == "S2" else "LE-E3"
                L += [f"### {surf} / {st} -- n {g['0']['n_windows']} windows / {g['0']['n_episodes']} ep", "",
                      "ADE means (s0): " + ", ".join(f"{k} {v:.4f}" for k, v in m.items()), "",
                      "| read | seed 0 | seed 1 |", "|---|---|---|",
                      f"| **E1 AS REGISTERED: os - ha0_ext (bar of record)** | {c(g['0']['refs']['E1_registered_os_minus_ha0_ext'])} | {c(g['1']['refs']['E1_registered_os_minus_ha0_ext'])} |",
                      f"| os - ha0_ext_pose | {c(g['0']['refs']['os_minus_ha0_ext_pose']['ade_m'])} | {c(g['1']['refs']['os_minus_ha0_ext_pose']['ade_m'])} |"]
                for arm in ARMS:
                    lab = f"{arm} - ha0_ext ({lbl_e} form; lever, NOT a bar)" if arm in PRIMARY_ARMS or arm == "RS_XF" else f"{arm} - ha0_ext"
                    L.append(f"| {lab} | {c(g['0']['arms'][arm]['minus_ha0_ext'])} | {c(g['1']['arms'][arm]['minus_ha0_ext'])} |")
                    L.append(f"| {arm} - ha0_ext_pose ({'LE-E1P form' if surf == 'S2' else 'S6'}) | {c(g['0']['arms'][arm]['minus_ha0_ext_pose'])} | {c(g['1']['arms'][arm]['minus_ha0_ext_pose'])} |")
                    L.append(f"| {arm} - os (ADE) | {c(g['0']['arms'][arm]['minus_os']['ade_m'])} | {c(g['1']['arms'][arm]['minus_os']['ade_m'])} |")
                L.append("")
                bkarm = None
                try:
                    bkarm = {s: bkj["trees"][tree]["seeds"][s]["surfaces"][surf]["sets"][st]["arms"]["RS_XF"] for s in ("0", "1")} if bkj else None
                except KeyError:
                    bkarm = None
                L += ["| metric (family) | RETIME_XF - os s0 | s1 | RETIME_XF - refcv7 s0 | s1 | GATED_RETIME_XF - os s0 | s1 | GATED_RETIME_XF - refcv7 s0 | RS_XF (banked Amendment 2) - os s0 |",
                      "|---|---|---|---|---|---|---|---|---|"]
                for fam, k, *_r in r["metrics"]:
                    a0, a1, q0, q1 = g["0"]["arms"]["RETIME_XF"], g["1"]["arms"]["RETIME_XF"], g["0"]["arms"]["GATED_RETIME_XF"], g["1"]["arms"]["GATED_RETIME_XF"]
                    xb = c(bkarm["0"]["minus_os"][k]) if bkarm and k in bkarm["0"].get("minus_os", {}) else "n/a"
                    L.append(f"| {k} ({fam}) | {c(a0['minus_os'][k])} | {c(a1['minus_os'][k])} | {c(a0['NR_minus_refcv7'][k])} | {c(a1['NR_minus_refcv7'][k])} | "
                             f"{c(q0['minus_os'][k])} | {c(q1['minus_os'][k])} | {c(q0['NR_minus_refcv7'][k])} | {xb} |")
                sm = tr["summary"].get(f"{surf}|{st}", {})
                if sm:
                    shr = sm["share_of_RS_XF_minus_os_retained_by_RETIME_s0"]
                    L.append(f"\n* E1 registered verdict (os floor {sm['os_floor']:.4f}): **{sm['E1_registered_verdict']}**. Share of RS_XF - os (ADE, s0 point) retained by RETIME_XF: "
                             + (f"{shr:.3f}." if shr is not None else "n/a."))
                    for arm in PRIMARY_ARMS:
                        x = sm[arm]
                        a = x["ACC_LAT"]
                        L.append(f"* {arm} (floor {x['arm_floor']:.4f}): {lbl_e} (-ha0_ext) **{x['LE_E1_form_minus_ha0_ext']}**; LE-E1P (-ha0_ext_pose) {x['LE_E1P_form_minus_ha0_ext_pose']}; "
                                 f"-os {x['LE_OS_form_minus_os_ade']}; LE-NR rollup (headline rows) {x['LE_NR_rollup_headline_rows']}; ACC-LAT "
                                 f"{x['ACC_LAT_label']} (worse at any seed: {a['separated_worse_at_any_seed'] or 'none'}).")
                        L.append(f"  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): {x['vs_refcv7_separated_worse_both_seeds_all_rows'] or 'none'}; "
                                 f"BETTER: {x['vs_refcv7_separated_better_both_seeds_all_rows'] or 'none'}")
                        L.append(f"  * vs refcv8 os, separated WORSE at both seeds: **{x['vs_os_separated_worse_both_seeds_all_rows'] or 'none'}**; "
                                 f"BETTER: {x['vs_os_separated_better_both_seeds_all_rows'] or 'none'}")
                for arm in RT_ARMS[:2]:
                    for sd_ in ("0", "1"):
                        gm = g[sd_]["arms"][arm]["geometry"]
                        L.append(f"* geometry {arm} s{sd_}: max dist to os path (float64) {gm['max_dist_to_path_f64_m']:.2e} m (after float32 cast {gm['max_dist_to_path_after_cast_m']:.2e}, bound "
                                 f"{gm['float32_rounding_bound_m']:.2e}); non-monotone progress windows {gm['frac_nonmonotone_windows']:.4f}; extrapolated points {gm['frac_extrapolated_points']:.4f}; "
                                 f"fallback-direction windows {gm['frac_fallback_dir_windows']:.4f}; mean |g-1| {gm['mean_abs_g_minus_1']:.3f}, max {gm['max_abs_g_minus_1']:.3f}")
                op = g["0"]["controls"]["c_d_optimism"]
                L.append(f"* optimism (-pose, s0): RETIME XF-IS {op['RETIME_XF_minus_IS']:+.4f} / gated {op['GATED_RETIME_XF_minus_IS']:+.4f}, XF-WF {op['RETIME_XF_minus_WF']:+.4f} / gated "
                         f"{op['GATED_RETIME_XF_minus_WF']:+.4f}; IS known sign (<= os): {op['IS_known_sign_le_os']}.")
                fs = g["0"].get("fold_seed_sensitivity")
                if fs:
                    L.append(f"* fold-seed sensitivity (s0, points): RETIME_XF -ha0_ext {fs['RETIME_XF']['minus_ha0_ext_range'][0]:+.4f}..{fs['RETIME_XF']['minus_ha0_ext_range'][1]:+.4f} "
                             f"({len(fs['RETIME_XF']['fold_seeds'])} seeds), GATED_RETIME_XF -ha0_ext {fs['GATED_RETIME_XF']['minus_ha0_ext_range'][0]:+.4f}.."
                             f"{fs['GATED_RETIME_XF']['minus_ha0_ext_range'][1]:+.4f} ({len(fs['GATED_RETIME_XF']['fold_seeds'])} seeds).")
                L.append(f"* in-sample RETIME g per instant (s0): {g['0']['arms']['RETIME_IS']['weights']['fold0']['g']}")
                for sd_ in ("0", "1"):
                    gb = fold_mean_bins(g[sd_]["arms"]["GATED_RETIME_XF"]["weights"])
                    L.append(f"* RETIME_XF fold-mean g (s{sd_}) per instant: {fold_mean(g[sd_]['arms']['RETIME_XF']['weights'])}; GATED_RETIME_XF fold-mean g per v0 bin: {gb}")
                L.append("")
    (OUT / "A0K0_RETIME.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return 0


def brief() -> int:
    """RESULTS_BRIEF.md: exactly the hand-back cells (E1-form, S6 lateral vs os / refcv7, fitted gains, ACC-LAT), read from the JSONs."""
    r = jload(OUT / "A0K0_RETIME.json")
    L = ["# RESULTS_BRIEF -- arc-length re-timing (all cells T1 self-action OPEN LOOP on logged eval139 frames; paired episode-cluster bootstrap B 2000, "
         "'another draw of EPISODES' only, single training seed; n windows / episodes in each header)", "",
         f"controls ALL_PASS = {r['controls_summary']['ALL_PASS']}; PREREG_RETIME sha256 {r['prereg_sha256'][:12]}; script sha256 {r['script_sha256'][:12]}", ""]
    for tree in ("step30000ema_unc", "step30000_unc"):
        tr = r["trees"][tree]
        for st in ("compliant", "all"):
            for surf in SURFS:
                g = {s: tr["seeds"][s]["surfaces"][surf]["sets"][st] for s in ("0", "1")}
                lbl = "LE-E1" if surf == "S2" else "LE-E3"
                L += [f"## {tree} / {surf} / {st} -- n {g['0']['n_windows']} windows / {g['0']['n_episodes']} episodes", "",
                      "| arm | - ha0_ext (" + lbl + ") s0 | s1 | - ha0_ext_pose s0 | s1 | - os (ADE) s0 | s1 |", "|---|---|---|---|---|---|---|"]
                for arm in ("RETIME_XF", "GATED_RETIME_XF", "RS_XF"):
                    a0, a1 = g["0"]["arms"][arm], g["1"]["arms"][arm]
                    L.append(f"| {arm} | {c(a0['minus_ha0_ext'])} | {c(a1['minus_ha0_ext'])} | {c(a0['minus_ha0_ext_pose'])} | {c(a1['minus_ha0_ext_pose'])} | "
                             f"{c(a0['minus_os']['ade_m'])} | {c(a1['minus_os']['ade_m'])} |")
                L.append("")
                if surf == "S6":
                    L += [f"S6 LATERAL rows ({tree} / {st}) -- arm - os and arm - refcv7-50,400, s0 ; s1:", "", "| row | RETIME - os | GATED_RETIME - os | RETIME - refcv7 | GATED_RETIME - refcv7 |", "|---|---|---|---|---|"]
                    for k in LATERAL_KEYS:
                        f2 = lambda arm, blk: " ; ".join(c(g[s]["arms"][arm][blk][k]) for s in ("0", "1"))  # noqa: E731
                        L.append(f"| {k} | {f2('RETIME_XF', 'minus_os')} | {f2('GATED_RETIME_XF', 'minus_os')} | "
                                 f"{f2('RETIME_XF', 'NR_minus_refcv7')} | {f2('GATED_RETIME_XF', 'NR_minus_refcv7')} |")
                    L.append("")
                sm = tr["summary"].get(f"{surf}|{st}", {})
                for arm in PRIMARY_ARMS:
                    if arm in sm:
                        a = sm[arm]["ACC_LAT"]
                        L.append(f"* {arm}: {lbl} verdict form {sm[arm]['LE_E1_form_minus_ha0_ext']}; ACC-LAT {sm[arm]['ACC_LAT_label']} "
                                 f"(separated worse at any seed: {a['separated_worse_at_any_seed'] or 'none'})")
                if sm:
                    L.append(f"* share of RS_XF - os (ADE, s0 point) retained by RETIME_XF: {sm['share_of_RS_XF_minus_os_retained_by_RETIME_s0']}")
                L.append("")
    if FROZEN_JSON.exists():
        fz = jload(FROZEN_JSON)
        L += ["## FITTED GAINS (EMA-30k, ALL windows, both seeds pooled; one scalar g per instant)", ""]
        for surf, t in fz["surfaces"].items():
            a = t["all"]
            L.append(f"* {surf} instants {t['instants_s']} s: RETIME g = {np.round(np.asarray(a['RETIME']), 3).tolist()}")
            for name, v in a["GATED_RETIME"].items():
                L.append(f"  * GATED {name} (n {v['n_fit_unique']}): g {np.round(np.asarray(v['g']), 3).tolist()}")
    (OUT / "RESULTS_BRIEF.md").write_text(chr(10).join(L) + chr(10), encoding="utf-8")
    print("WROTE", OUT / "RESULTS_BRIEF.md")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compute", "render", "freeze", "forward", "brief"])
    ap.add_argument("--tree", default=None, help="forward: the tree to apply the frozen gains to")
    ap.add_argument("--trees", nargs="*", default=list(TREES))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--surfs", nargs="*", default=list(SURFS))
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    a = ap.parse_args()
    if a.cmd == "render":
        return render()
    if a.cmd == "brief":
        return brief()
    if a.cmd == "freeze":
        return freeze()
    if a.cmd == "forward":
        if not a.tree:
            raise SystemExit("forward needs --tree")
        return forward(a.tree)
    return compute(a.trees, tuple(a.seeds), a.surfs, a.sets)


if __name__ == "__main__":
    sys.exit(main())
