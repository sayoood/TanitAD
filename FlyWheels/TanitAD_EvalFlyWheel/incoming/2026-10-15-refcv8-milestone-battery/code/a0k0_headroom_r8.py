"""a0k0_headroom_r8.py -- ADMISSIBLE ego-state headroom for refcv8 at inference: residual scaling on its own causal prior, and the
admissible references (ha0_ext_pose, ha0_imu) beside the E1 echo.  CPU only, banked panels only, zero training.  NEW file.
Pre-registration (incl. Amendments 1 and 2, all written before any number): <PKG>/raw/a0k0_headroom/PREREG.md.

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 \
    PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' \
      python -X utf8 a0k0_headroom_r8.py compute      # -> <PKG>/raw/a0k0_headroom/A0K0_HEADROOM.json (+ .md)
      python -X utf8 a0k0_headroom_r8.py render       # re-render the .md from the JSON

ADMISSIBILITY (PI 2026-10-18 ~23:58, relayed): measured a0 and yaw rate at t0 ARE admissible (IMU-measurable); curvature is NOT.
refcv8 is NOT "fed v0 only": its argv carries --ego-history and --residual-prior ha0_ext_pose (kinematic_prior.py), so its plan is
a causal prior (a0, kappa0 = omega0 / max(v0, 2)) + a learned residual.  The E1 echo ha0_ext reads kappa0 from the STEER channel.

REFERENCES (model-free; built from the panel's own decisions/ep_poses, observed window ws-7..ws):
  ha0_ext       the banked E1 echo (panel arm) -- E1 AS REGISTERED, the bar of record.
  ha0_ext_pose  kinematic_prior.prior_controls('ha0_ext_pose') -> prior_path: refcv8's own prior.  PRIMARY admissible reference.
  ha0_imu       hold a0 (backward speed diff) and yaw rate psi_dot0 (backward yaw diff) constant; rollout_unicycle's update order with
                heading rate psi_dot0 held.  SECONDARY.  ha0_imu_ax: the same with a0 = measured ax (ego_t0[:, 1]).
ARMS (fit = continuous minimiser of mean window ADE per instant, L-BFGS-B + analytic gradient in the box, verified on a 0.1 grid):
  RS / GATED_RS / RS_POS  p = g*os + (1-g)*ha0_ext_pose, g in [0, 2] per instant x axis (GATED: per v0 bin; POS: one g per instant).
  BLEND_imu               p = w*os + (1-w)*ha0_imu, w in [0, 1].
  _XF = 5 EPISODE folds (clip_index, rng 20261018), fit on 4 apply to 1;  _IS in-sample;  _WF folds per WINDOW (leaky control).
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
OUT = PKG / "raw" / "a0k0_headroom"
TREES = ("step30000ema_unc", "step30000_unc")
SEEDS = (0, 1)
SURFS = ("S2", "S6")
SETS = ("compliant", "all")
MASK_SHA = "4ebc6b8604f064f69e9e94c15e099062f68339a8ef38d666e8a29d29cb1c4129"
N_BOOT = 2000
K_FOLDS = 5
FOLD_SEED = 20261018
SENS_SEEDS = tuple(range(1, 11))
MIN_BIN_FIT = 30
W_OBS = 8
DT = 0.1
BOX_RS = (0.0, 2.0)
BOX_BL = (0.0, 1.0)
VERIFY_STEP = 0.1
V1_TOL_M = 1e-3
V2_TOL_M = 1e-4
REF0926 = "C:/lgs/Lfb47a58376/TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-residual-prior/raw/measure_prior_vs_echo.json"
CELL_KEYS = ("delta", "lo", "hi", "n_windows", "n_episodes", "separated", "n_dropped_nonfinite")
TIER = "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)"
QUESTION = ("paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, models FIXED: 'another draw of EPISODES' "
            "only; NOT another training run (single training seed), NOT another inference run (seed 1 is the replicate beside seed 0), "
            "NOT another fold assignment (fold-seed sensitivity reported separately)")
STATUS = ("EARLY READ of banked step 30,000. E1 AS REGISTERED (os - ha0_ext) is the bar of record and is reported first. The admissible-reference "
          "read (os - ha0_ext_pose / ha0_imu) is a BESIDE-ROW pre-registered 2026-10-19 before any final-checkpoint data, MOTIVATED AFTER seeing "
          "the 30k read. RS arms are cross-fitted calibrations fit on EVAL episodes: lever estimates, never a bar pass (a deployable version fits g on TRAIN).")


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


def _grid_min(rx, ry, dx, dy, box, per_axis):
    g = np.arange(box[0], box[1] + 1e-9, VERIFY_STEP)
    best, arg = np.inf, None
    for gx in g:
        if per_axis:
            ex2 = (rx + gx * dx) ** 2
            L = np.sqrt(ex2[None, :] + (ry[None, :] + g[:, None] * dy[None, :]) ** 2).mean(1)
            j = int(np.argmin(L))
            if L[j] < best:
                best, arg = float(L[j]), np.array([gx, g[j]])
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
    x0 = np.full(2 if per_axis else 1, min(max(1.0, box[0]), box[1]))
    for k in range(T):
        rx, ry = B[:, k, 0] - G[:, k, 0], B[:, k, 1] - G[:, k, 1]
        dx, dy = O[:, k, 0] - B[:, k, 0], O[:, k, 1] - B[:, k, 1]
        opt = dict(ftol=1e-15, gtol=1e-12, maxiter=1000)
        r = minimize(lg, x0, args=(rx, ry, dx, dy), jac=True, method="L-BFGS-B", bounds=[box] * x0.size, options=opt)
        gmin, garg = _grid_min(rx, ry, dx, dy, box, per_axis)
        w, f = np.clip(r.x, *box), lg(np.clip(r.x, *box), rx, ry, dx, dy)[0]
        if f > gmin + 1e-12:
            r2 = minimize(lg, garg, args=(rx, ry, dx, dy), jac=True, method="L-BFGS-B", bounds=[box] * x0.size, options=opt)
            w2 = np.clip(r2.x, *box)
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


ARMS = {  # name: (base, box, per_axis, fold kind, gated, full read)
    "RS_XF": ("HP", BOX_RS, True, "ep", False, True),
    "GATED_RS_XF": ("HP", BOX_RS, True, "ep", True, True),
    "RS_POS_XF": ("HP", BOX_RS, False, "ep", False, False),
    "RS_IS": ("HP", BOX_RS, True, "is", False, False),
    "GATED_RS_IS": ("HP", BOX_RS, True, "is", True, False),
    "RS_WF": ("HP", BOX_RS, True, "wf", False, False),
    "GATED_RS_WF": ("HP", BOX_RS, True, "wf", True, False),
    "BLEND_imu_XF": ("HI", BOX_BL, True, "ep", False, False),
}


def read_set(tools, X: dict, eid, clip, v0, dt, banked_set: dict, log) -> dict:
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
    # (a) g = 1 reproduces the banked os cells EXACTLY
    P1 = blend(X["O"], X["HP"], np.ones((T, 2)))
    c1 = comp(P1)
    a = {"bit_exact_vs_os": bool(np.array_equal(P1, X["O"])), "rows": {}}
    a["rows"]["ade_m|ha0_ext"] = cells_equal(PR.paired_cell(c1["ade_m"], C["HX"]["ade_m"], eid, N_BOOT), banked_set["rows"]["ade_m"]["os_minus_ha0_ext"])
    for k in keys:
        a["rows"][f"{k}|refcv7"] = cells_equal(PR.paired_cell(c1[k], C["R"][k], eid, N_BOOT), banked_set["rows"][k]["os_minus_refcv7_50400"])
    a["n_rows"], a["n_exact"] = len(a["rows"]), int(sum(v["exact"] for v in a["rows"].values()))
    a["PASS"] = bool(a["bit_exact_vs_os"] and a["n_exact"] == a["n_rows"])
    out["controls"]["a_g1_reproduces_banked_os"] = a
    # (b) g = 0 is the prior bit for bit; its cell against the prior reads exactly 0 (and w = 0 is ha0_imu for BLEND_imu)
    P0, Q0 = blend(X["O"], X["HP"], np.zeros((T, 2))), blend(X["O"], X["HI"], np.zeros((T, 2)))
    z = PR.paired_cell(comp(P0)["ade_m"], C["HP"]["ade_m"], eid, N_BOOT)
    out["controls"]["b_g0_is_prior"] = {"bit_exact_vs_ha0_ext_pose": bool(np.array_equal(P0, X["HP"])), "bit_exact_w0_vs_ha0_imu": bool(np.array_equal(Q0, X["HI"])),
                                        "cell_vs_prior": slim(z), "PASS": bool(np.array_equal(P0, X["HP"]) and np.array_equal(Q0, X["HI"]) and z["delta"] == 0 and z["lo"] == 0 and z["hi"] == 0)}
    del P1, c1, P0, Q0
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
    for name, (base, box, per_axis, fk, gated, full) in ARMS.items():
        st = {}
        kw = dict(groups=vb if gated else None, box=box, per_axis=per_axis, stats=st)
        Bb = X[base]
        if fk == "is":
            Wn, rec = insample(X["O"], Bb, G, **kw)
        else:
            Wn, rec = crossfit(X["O"], Bb, G, folds if fk == "ep" else wfolds, **kw)
        P = blend(X["O"], Bb, Wn)
        cB = comp(P)
        arm = {"base": {"HP": "ha0_ext_pose", "HI": "ha0_imu"}[base], "box": list(box), "weights": rec, "mean_ade_m": float(np.nanmean(cB["ade_m"])),
               "minus_ha0_ext": cell(cB, C["HX"]), "minus_ha0_ext_pose": cell(cB, C["HP"]), "minus_ha0_imu": cell(cB, C["HI"])}
        if full:
            arm["NR_minus_refcv7"] = {k: cell(cB, C["R"], k) for k in keys}
            arm["minus_os"] = {k: cell(cB, C["O"], k) for k in keys}
        else:
            arm["minus_os"] = {"ade_m": cell(cB, C["O"])}
        out["arms"][name] = arm
        out["fit_stats"][name] = st
        log(f"      {name:13s} -ext {arm['minus_ha0_ext']['delta']:+.4f} [{arm['minus_ha0_ext']['lo']:+.4f}, {arm['minus_ha0_ext']['hi']:+.4f}]  "
            f"-pose {arm['minus_ha0_ext_pose']['delta']:+.4f}  -os {arm['minus_os']['ade_m']['delta']:+.4f} [{arm['minus_os']['ade_m']['lo']:+.4f}, {arm['minus_os']['ade_m']['hi']:+.4f}]")
        del cB, P, Wn
    # fold-seed sensitivity (points, numpy ADE) for the PRIMARY arms
    aX, aP = ade_np(X["HX"], G).mean(), ade_np(X["HP"], G).mean()
    sens = {}
    for name, gated in (("RS_XF", False), ("GATED_RS_XF", True)):
        pts = []
        for s in SENS_SEEDS:
            Wn, _ = crossfit(X["O"], X["HP"], G, episode_folds(clip, seed=s), groups=vb if gated else None)
            a_ = ade_np(blend(X["O"], X["HP"], Wn), G).mean()
            pts.append((float(a_ - aX), float(a_ - aP)))
        Wn, _ = crossfit(X["O"], X["HP"], G, folds, groups=vb if gated else None)
        a_ = ade_np(blend(X["O"], X["HP"], Wn), G).mean()
        sens[name] = {"fold_seeds": list(SENS_SEEDS), "minus_ha0_ext_points": [p[0] for p in pts], "minus_ha0_ext_pose_points": [p[1] for p in pts],
                      "minus_ha0_ext_range": [min(p[0] for p in pts), max(p[0] for p in pts)],
                      "minus_ha0_ext_pose_range": [min(p[1] for p in pts), max(p[1] for p in pts)],
                      "primary_seed_points": [float(a_ - aX), float(a_ - aP)]}
    out["fold_seed_sensitivity"] = sens
    A = out["arms"]
    d = lambda x, k="minus_ha0_ext_pose": A[x][k]["delta"]  # noqa: E731
    out["controls"]["c_d_optimism"] = {"RS_XF_minus_IS": d("RS_XF") - d("RS_IS"), "GATED_RS_XF_minus_IS": d("GATED_RS_XF") - d("GATED_RS_IS"),
                                       "RS_XF_minus_WF": d("RS_XF") - d("RS_WF"), "GATED_RS_XF_minus_WF": d("GATED_RS_XF") - d("GATED_RS_WF"),
                                       "IS_mean_ade_lt_XF_mean_ade": bool(A["RS_IS"]["mean_ade_m"] < A["RS_XF"]["mean_ade_m"]),
                                       "IS_known_sign_le_prior_and_os": bool(A["RS_IS"]["mean_ade_m"] <= out["ade_means"]["ha0_ext_pose"] + 1e-9
                                                                             and A["RS_IS"]["mean_ade_m"] <= out["ade_means"]["os"] + 1e-9)}
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
FROZEN_JSON = OUT / "FROZEN_GAINS_ema30k.json"


def freeze() -> int:
    """Amendment 3: RS / GATED_RS gains fit on ALL windows of EMA-30k, both inference seeds POOLED, one table per surface grid."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[a0k0] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    import s6_lon_decomp as SD
    import compliant_reread_r8 as CR
    t0 = time.time()
    tools = tools_load()
    first, per = {}, {s: {} for s in SURFS}
    inputs = {}
    for seed in SEEDS:
        ctx = CR.prepare_seed(FREEZE_TREE, seed, tools, tools["rc"])
        if ctx["sha"] != MASK_SHA:
            raise SystemExit("[a0k0 freeze] mask sha mismatch")
        for surf in SURFS:
            S = surface_inputs(tools, ctx, surf, first, FREEZE_TREE)
            if not (S["dec"]["chk"]["PASS"] and S["V"]["PASS"]):
                raise SystemExit(f"[a0k0 freeze] {surf} s{seed}: decisions / validation failed")
            per[surf][seed] = S
            inputs[f"{surf}|s{seed}"] = _sha_dir(S["dir"])
        del ctx
    out = {"tool": "a0k0_headroom_r8.py freeze", "generated": time.strftime("%FT%T"), "prereg": "PREREG.md Amendment 3",
           "prereg_sha256": hashlib.sha256((OUT / "PREREG.md").read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "tree": f"{CR.ARCHIVE}/{FREEZE_TREE}",
           "fit_set": "ALL windows of the surface (all 139 eval episodes), inference seeds 0 and 1 POOLED (each seed's os against the same prior and GT)",
           "fitter": f"L-BFGS-B per instant x axis, box RS {list(BOX_RS)}, verified on a {VERIFY_STEP} grid", "prior": "ha0_ext_pose (kinematic_prior.prior_controls/prior_path)",
           "plan": "p = g * os + (1 - g) * ha0_ext_pose, g[k, c] for instant k (the surface grid) and axis c in (along x, cross y) of the ego frame at t0",
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
            rs = fit_gains(O2, B2, G2, box=BOX_RS, stats=st)
            gated = {}
            for b, (_lo, _hi, name) in enumerate(SD.V0_EDGES):
                nu = int((vb[sel] == b).sum())
                if nu >= MIN_BIN_FIT:
                    gated[name] = {"bin": b, "n_fit_unique": nu, "g": fit_gains(O2[vb2 == b], B2[vb2 == b], G2[vb2 == b], box=BOX_RS, stats=st).tolist()}
                else:
                    gated[name] = {"bin": b, "n_fit_unique": nu, "g": rs.tolist(), "fallback": "RS table"}
            Wg = np.stack([np.asarray(gated[n]["g"]) for _a, _b, n in SD.V0_EDGES])[vb2]
            rec[set_name] = {"n_fit_rows_pooled": int(len(O2)), "RS": rs.tolist(), "GATED_RS": gated, "fit_stats": st,
                             "in_sample_pooled_mean_ade": {"os": float(ade_np(O2, G2).mean()), "ha0_ext_pose": float(ade_np(B2, G2).mean()),
                                                           "RS": float(ade_np(blend(O2, B2, rs), G2).mean()), "GATED_RS": float(ade_np(blend(O2, B2, Wg), G2).mean())}}
        out["surfaces"][surf] = rec
    out["wall_s"] = round(time.time() - t0, 1)
    jdump(out, FROZEN_JSON)
    print(f"WROTE {FROZEN_JSON} ({out['wall_s']} s)")
    for surf in SURFS:
        print(surf, "RS", np.round(out["surfaces"][surf]["all"]["RS"], 3).tolist(), out["surfaces"][surf]["all"]["in_sample_pooled_mean_ade"])
    return 0


def frozen_weights(fz: dict, surf: str, v0) -> dict:
    import s6_lon_decomp as SD
    t = fz["surfaces"][surf]["all"]
    vb = v0_bins(v0)
    tab = np.stack([np.asarray(t["GATED_RS"][n]["g"], np.float64) for _a, _b, n in SD.V0_EDGES])
    return {"RS_FROZEN30k": np.asarray(t["RS"], np.float64), "GATED_RS_FROZEN30k": tab[vb]}


def forward(tree: str) -> int:
    """Amendment 3: apply the FROZEN EMA-30k gains, unchanged, to `tree`; the four bars FZ-E1 / FZ-E1P / FZ-NR / FZ-E3."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[a0k0] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    import compliant_reread_r8 as CR
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)  # noqa: E731
    fz = jload(FROZEN_JSON)
    tools = tools_load()
    PR = tools["PR"]
    keys = [m[1] for m in PR.METRICS]
    tjp = PKG / "raw" / "compliant_reread" / "trees" / f"{tree}.json"
    tj = jload(str(tjp)) if tjp.exists() else None
    res = {"tool": "a0k0_headroom_r8.py forward", "generated": time.strftime("%FT%T"), "tree": f"{CR.ARCHIVE}/{tree}", "tier": TIER, "estimator": QUESTION,
           "prereg": "PREREG.md Amendment 3", "frozen_gains": str(FROZEN_JSON), "frozen_gains_sha256": hashlib.sha256(FROZEN_JSON.read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "role": ("IN-SAMPLE path check (the gains were fit on this tree)" if tree == FREEZE_TREE else
                    "path check on a different checkpoint of the SAME run (not an independent test, not a bar)" if tree == "step30000_unc" else
                    "FORWARD TEST (Amendment 3 bars)"),
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
                    Wn = W if W.ndim == 2 else W[sel]
                    cB = comp(blend(X["O"], X["HP"], Wn))
                    r["arms"][arm] = {"mean_ade_m": float(np.nanmean(cB["ade_m"])), "minus_ha0_ext": cell(cB, C["HX"]), "minus_ha0_ext_pose": cell(cB, C["HP"]),
                                      "NR_minus_refcv7": {k: cell(cB, C["R"], k) for k in keys}, "minus_os": {k: cell(cB, C["O"], k) for k in keys}}
                    log(f"  {tree} s{seed} {surf} {set_name} {arm}: -ext {c(r['arms'][arm]['minus_ha0_ext'])}  -pose {c(r['arms'][arm]['minus_ha0_ext_pose'])}")
                ss["sets"][set_name] = r
            sr["surfaces"][surf] = ss
        res["seeds"][str(seed)] = sr
        del ctx
    # the bars (compliant primary, all beside)
    bars = {}
    for st in SETS:
        g = lambda s, surf: res["seeds"][str(s)]["surfaces"][surf]["sets"][st]  # noqa: E731
        for arm in ("RS_FROZEN30k", "GATED_RS_FROZEN30k"):
            fl = {surf: abs(g(0, surf)["arms"][arm]["mean_ade_m"] - g(1, surf)["arms"][arm]["mean_ade_m"]) for surf in SURFS}
            nr = {}
            for fam, k, _l, _u, _lower, surfs_ in PR.METRICS:
                for surf in surfs_:
                    nr[f"{surf}:{k}"] = {"family": fam, **PR.bar_NR_metric({s: g(s, surf)["arms"][arm]["NR_minus_refcv7"][k] for s in SEEDS})}
            bars[f"{st}|{arm}"] = {"arm_floor": fl,
                                   "FZ-E1": PR.bar_E({s: g(s, "S2")["arms"][arm]["minus_ha0_ext"] for s in SEEDS}, fl["S2"]),
                                   "FZ-E1P": PR.bar_E({s: g(s, "S2")["arms"][arm]["minus_ha0_ext_pose"] for s in SEEDS}, fl["S2"]),
                                   "FZ-E3": PR.bar_E({s: g(s, "S6")["arms"][arm]["minus_ha0_ext"] for s in SEEDS}, fl["S6"]),
                                   "FZ-NR": {"metrics": nr, "per_family": PR.rollup_NR(nr, {k: v["family"] for k, v in nr.items()})}}
    res["bars"] = bars
    res["controls"]["ALL_PASS"] = bool(ok)
    res["wall_s"] = round(time.time() - t0, 1)
    p = OUT / f"FORWARD_{tree}.json"
    jdump(res, p)
    for k, v in bars.items():
        log(f"{k}: FZ-E1 {v['FZ-E1']['verdict']} | FZ-E1P {v['FZ-E1P']['verdict']} | FZ-E3 {v['FZ-E3']['verdict']} | FZ-NR {v['FZ-NR']['per_family']}")
    log(f"WROTE {p} controls={ok}")
    return 0 if ok else 3


def compute(trees, seeds, surfs, sets_) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[a0k0] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    import compliant_reread_r8 as CR
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)  # noqa: E731
    tools = tools_load()
    PR = tools["PR"]
    ref0926 = jload(REF0926)
    res = {"tool": "a0k0_headroom_r8.py", "generated": time.strftime("%FT%T"), "tier": TIER, "estimator": QUESTION, "status": STATUS,
           "n_boot": N_BOOT, "k_folds": K_FOLDS, "fold_seed": FOLD_SEED, "min_bin_fit": MIN_BIN_FIT, "boxes": {"RS": BOX_RS, "BLEND_imu": BOX_BL},
           "fitter": f"L-BFGS-B (analytic gradient) per instant, verified on a {VERIFY_STEP} grid", "prereg": str(OUT / "PREREG.md"),
           "prereg_sha256": hashlib.sha256((OUT / "PREREG.md").read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "argv_refcv8": "E:/refcv8_eval_kit/ckpt/config.json argv carries --ego-history and --residual-prior ha0_ext_pose (verified)",
           "ref_2026_09_26": {"path": REF0926, "S2_P_ha0_ext_pose_minus_echo": ref0926["B_ade_0_2s"]["P_ha0_ext_pose_minus_echo"],
                              "S6_8slot_P_ha0_ext_pose_minus_P_ha0_ext": ref0926["C_ade_0_6s_8slots"]["P_ha0_ext_pose_minus_P_ha0_ext"],
                              "echo_banked_ade_2s": ref0926["B_ade_0_2s"]["ade_mean_m"]["echo_banked"]},
           "metrics": [list(m[:5]) for m in PR.METRICS], "strategic": "NOT APPLICABLE, n = 0 (the run argv carries --no-strategic: the layer is off)",
           "trees": {}}
    ok = True
    for tree in trees:
        tj = jload(str(PKG / "raw" / "compliant_reread" / "trees" / f"{tree}.json"))
        if not tj["controls"]["ALL_PASS"]:
            raise SystemExit(f"[a0k0] {tree}: compliant reread controls not ALL_PASS -- refusing")
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
                log(f"  {tree} s{seed} {surf}: V1 {V['V1_ha0_ext_rebuilt_vs_panel_max_abs_m']:.2e}  V2 {V['V2_imu_with_psidot_kappa0v0_a0zero_vs_unicycle_kappa0_max_abs_m']:.2e}  "
                    f"V3 {V['V3_frac_imu_vs_ha0_ext_gt_0p1m']}  dec {dec['chk']['PASS']}")
                for set_name in sets_:
                    sel = mk if set_name == "compliant" else np.ones(len(eid), bool)
                    banked = sd["surfaces"][surf]["compliant" if set_name == "compliant" else "all_window"]["uncapped"]
                    log(f"  {tree} s{seed} {surf} {set_name}: n {int(sel.sum())}")
                    r = read_set(tools, {k: v[sel] for k, v in X_all.items()}, eid[sel], clip[sel], v0[sel], dt, banked, log)
                    ok &= r["controls"]["a_g1_reproduces_banked_os"]["PASS"] and r["controls"]["b_g0_is_prior"]["PASS"]
                    ss["sets"][set_name] = r
                sr["surfaces"][surf] = ss
            tr["seeds"][str(seed)] = sr
            del ctx
        tr["summary"] = summarise(PR, tr, surfs, sets_, seeds)
        res["trees"][tree] = tr
        OUT.mkdir(parents=True, exist_ok=True)
        jdump(res, OUT / "A0K0_HEADROOM.partial.json")
    # the 2026-09-26 reproduction (S2, all windows, any tree / seed: model-free)
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
    res["controls_summary"]["note"] = ("a (g1 == banked os, every row exact), b (g0 == ha0_ext_pose / w0 == ha0_imu bit-exact, 0-cell), decisions "
                                       "(ws, ep_poses[ws] == pose_last, v0), V1 (ha0_ext rebuilt <= 1e-3 m), V2 (imu a0=0 limit <= 1e-4 m), mask sha, K")
    res["wall_s"] = round(time.time() - t0, 1)
    full = tuple(trees) == TREES and tuple(seeds) == SEEDS and tuple(surfs) == SURFS and tuple(sets_) == SETS
    if not full:
        res["SMOKE"] = "subset run -- not the deliverable"
        jdump(res, OUT / "A0K0_HEADROOM.smoke.json")
        log(f"WROTE smoke ok={ok} R0926_exact={res['controls_summary']['R0926_exact']}")
        return 0 if ok else 3
    if not ok:
        jdump(res, OUT / "CONTROL_FAILED.json")
        print("CONTROL FAILED")
        return 3
    jdump(res, OUT / "A0K0_HEADROOM.json")
    try:
        os.remove(OUT / "A0K0_HEADROOM.partial.json")
    except OSError:
        pass
    render()
    log(f"WROTE {OUT / 'A0K0_HEADROOM.json'}")
    return 0


def resummarise() -> int:
    """Recompute the per-tree SUMMARY blocks from the cells already in A0K0_HEADROOM.json (no cell is recomputed); stamps the summary's
    own script sha beside the compute-time script_sha256.  Exists because the first summary applied pair_read_r8.bar_NR_metric (an
    ERROR-metric rule, lo > 0 = worse) to the two ACCURACY rows (LON_speed_within_0p5, TAC_traj_lon_correct) and so listed separated
    IMPROVEMENTS as 'FAIL'."""
    import clamp_posthoc_r8 as CP
    _NR, _BP, PR, _ra, _sha12 = CP.get_readers()
    p = OUT / "A0K0_HEADROOM.json"
    r = jload(p)
    for tree, tr in r["trees"].items():
        tr["summary"] = summarise(PR, tr, SURFS, SETS, SEEDS)
    r["summary_recomputed"] = {"by": "a0k0_headroom_r8.py resummarise", "at": time.strftime("%FT%T"),
                               "summary_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "reason": resummarise.__doc__.split("Exists because ")[1].strip(),
                               "cells_unchanged": "every cell is the compute-time value (script_sha256 above)"}
    jdump(r, p)
    render()
    print("RESUMMARISED", p)
    return 0


def summarise(PR, tr, surfs, sets_, seeds) -> dict:
    """Verdict FORMS (never bar verdicts for the arms) with the floors on the same set."""
    out = {}
    if len(seeds) < 2:
        return {"note": "fewer than 2 inference seeds"}
    for surf in surfs:
        for st in sets_:
            g = lambda s: tr["seeds"][str(s)]["surfaces"][surf]["sets"][st]  # noqa: E731
            fl_os = abs(g(0)["ade_means"]["os"] - g(1)["ade_means"]["os"])
            row = {"os_floor": fl_os,
                   "E1_registered_verdict": PR.bar_E({s: g(s)["refs"]["E1_registered_os_minus_ha0_ext"] for s in seeds}, fl_os)["verdict"],
                   "os_minus_ha0_ext_pose_verdict_form": PR.bar_E({s: g(s)["refs"]["os_minus_ha0_ext_pose"]["ade_m"] for s in seeds}, fl_os)["verdict"],
                   "os_minus_ha0_imu_verdict_form": PR.bar_E({s: g(s)["refs"]["os_minus_ha0_imu"] for s in seeds}, fl_os)["verdict"]}
            for arm in ("RS_XF", "GATED_RS_XF"):
                fl = abs(g(0)["arms"][arm]["mean_ade_m"] - g(1)["arms"][arm]["mean_ade_m"])
                # NR rollup on the battery's NR HEADLINE rows of this surface only (all ERROR metrics, as pair_read_r8.bars_of reads them)
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
                row[arm] = {"arm_floor": fl,
                            "minus_ha0_ext_E1_FORM_lever_not_bar": PR.bar_E({s: g(s)["arms"][arm]["minus_ha0_ext"] for s in seeds}, fl)["verdict"],
                            "minus_ha0_ext_pose_form": PR.bar_E({s: g(s)["arms"][arm]["minus_ha0_ext_pose"] for s in seeds}, fl)["verdict"],
                            "minus_os_form": PR.bar_E({s: g(s)["arms"][arm]["minus_os"]["ade_m"] for s in seeds}, fl_os)["verdict"],
                            "NR_rollup_headline_rows": PR.rollup_NR(nr, {k: v["family"] for k, v in nr.items()}),
                            "vs_refcv7_separated_worse_both_seeds_all_rows": rows_where("NR_minus_refcv7", True),
                            "vs_refcv7_separated_better_both_seeds_all_rows": rows_where("NR_minus_refcv7", False),
                            "vs_os_separated_worse_both_seeds_all_rows": rows_where("minus_os", True),
                            "vs_os_separated_better_both_seeds_all_rows": rows_where("minus_os", False)}
            out[f"{surf}|{st}"] = row
    return out


# ------------------------------------------------------------------------------------------ markdown
def c(x, dp=4):
    if not isinstance(x, dict) or x.get("delta") is None:
        return "n/a"
    s = " **SEP**" if x.get("separated") else ""
    return f"{x['delta']:+.{dp}f} [{x['lo']:+.{dp}f}, {x['hi']:+.{dp}f}]{s}"


def fold_mean(rec: dict):
    ws = [np.asarray(v["w"]) for k, v in rec.items() if "|" not in k and "w" in v]
    return np.round(np.mean(ws, 0), 2).tolist() if ws else None


def fold_mean_bins(rec: dict) -> dict:
    out = {}
    for b in range(5):
        ws = [np.asarray(v["w"]) for k, v in rec.items() if k.endswith(f"|bin{b}") and "w" in v]
        out[f"bin{b}"] = np.round(np.mean(ws, 0), 2).tolist() if ws else "fallback"
    return out


def render_frozen_forward() -> list:
    import glob
    L = []
    if FROZEN_JSON.exists():
        fz = jload(FROZEN_JSON)
        L += ["## FROZEN gains (Amendment 3) -- fit on EMA-30k, ALL windows, both seeds pooled", "",
              f"`{FROZEN_JSON.name}` sha256 `{hashlib.sha256(FROZEN_JSON.read_bytes()).hexdigest()[:12]}`; fit set: {fz['fit_set']}.", ""]
        for surf, t in fz["surfaces"].items():
            a = t["all"]
            L.append(f"* **{surf}** instants {t['instants_s']} s, n {t['n_windows_unique']} windows / {t['n_episodes']} ep: RS g [along, cross] = "
                     f"{np.round(a['RS'], 3).tolist()}; in-sample pooled ADE os {a['in_sample_pooled_mean_ade']['os']:.4f} / prior "
                     f"{a['in_sample_pooled_mean_ade']['ha0_ext_pose']:.4f} / RS {a['in_sample_pooled_mean_ade']['RS']:.4f} / GATED {a['in_sample_pooled_mean_ade']['GATED_RS']:.4f}")
            for name, v in a["GATED_RS"].items():
                L.append(f"  * GATED {name} (n {v['n_fit_unique']}{', ' + v['fallback'] if v.get('fallback') else ''}): {np.round(v['g'], 3).tolist()}")
        L.append("")
    for p in sorted(glob.glob(str(OUT / "FORWARD_*.json"))):
        fw = jload(p)
        L += [f"## FORWARD {os.path.basename(fw['tree'])} -- {fw['role']} (controls {fw['controls']['ALL_PASS']})", "",
              "| set / arm | FZ-E1 (-ha0_ext, S2) | FZ-E1P (-ha0_ext_pose, S2) | FZ-E3 (-ha0_ext, S6) | FZ-NR per family |", "|---|---|---|---|---|"]
        for k, v in fw["bars"].items():
            pe = lambda b: " / ".join(c(v[b]["per_inference_seed"][s]) for s in sorted(v[b]["per_inference_seed"]))  # noqa: E731
            L.append(f"| {k} | {v['FZ-E1']['verdict']}: {pe('FZ-E1')} | {v['FZ-E1P']['verdict']}: {pe('FZ-E1P')} | {v['FZ-E3']['verdict']}: {pe('FZ-E3')} | {v['FZ-NR']['per_family']} |")
        L.append("")
    return L


def render() -> int:
    r = jload(OUT / "A0K0_HEADROOM.json")
    cs = r["controls_summary"]
    L = ["# A0K0_HEADROOM -- admissible ego-state headroom for refcv8 (zero training, banked panels)", "",
         f"* **Status:** {r['status']}", f"* **Tier:** {r['tier']}", f"* **Estimator:** {r['estimator']}",
         f"* Pre-registration `raw/a0k0_headroom/PREREG.md` (sha256 `{r['prereg_sha256'][:12]}`, Amendments 1-2 written before any number); script sha256 `{r['script_sha256'][:12]}`.",
         f"* Premise correction: {r['argv_refcv8']}.",
         f"* Controls **ALL_PASS = {cs['ALL_PASS']}** ({cs['note']}). 2026-09-26 reproduction (S2 all windows, ha0_ext_pose - ha0_ext): "
         f"{c(cs['R0926_S2_all_ha0_ext_pose_minus_ha0_ext'])} vs banked {c(cs['R0926_banked'])} -> exact = **{cs['R0926_exact']}**.",
         f"* STRATEGIC family: {r['strategic']}.", ""] + render_frozen_forward() + [
         "References: `ha0_ext` = E1 echo (steer kappa0, INADMISSIBLE input); `ha0_ext_pose` = refcv8's own causal prior (admissible, PRIMARY); "
         "`ha0_imu` = hold a0 + yaw rate (admissible, SECONDARY). Arms: RS = residual scaling on ha0_ext_pose (g in [0,2], per instant x axis, "
         "5-episode-fold cross-fitted); GATED_RS = per v0 bin; _IS in-sample, _WF window-fold (leaky) controls; BLEND_imu = convex blend on ha0_imu.", ""]
    for tree, tr in r["trees"].items():
        L += [f"## {tree}", ""]
        for surf in SURFS:
            for st in SETS:
                try:
                    g = {s: tr["seeds"][s]["surfaces"][surf]["sets"][st] for s in ("0", "1")}
                except KeyError:
                    continue
                V = tr["seeds"]["0"]["surfaces"][surf]["validation"]
                m = g["0"]["ade_means"]
                L += [f"### {surf} / {st} -- n {g['0']['n_windows']} windows / {g['0']['n_episodes']} ep", "",
                      "ADE means (s0): " + ", ".join(f"{k} {v:.4f}" for k, v in m.items()), "",
                      "| read | seed 0 | seed 1 |", "|---|---|---|"]
                for lab, f in (("**E1 AS REGISTERED: os - ha0_ext (bar of record)**", lambda x: x["refs"]["E1_registered_os_minus_ha0_ext"]),
                               ("os - ha0_ext_pose (residual vs its own prior; admissible ref)", lambda x: x["refs"]["os_minus_ha0_ext_pose"]["ade_m"]),
                               ("os - ha0_imu (secondary admissible ref)", lambda x: x["refs"]["os_minus_ha0_imu"]),
                               ("os - ha0_imu_ax (a0 = measured ax)", lambda x: x["refs"]["os_minus_ha0_imu_ax"]),
                               ("ha0_ext_pose - ha0_ext", lambda x: x["refs"]["ha0_ext_pose_minus_ha0_ext"]),
                               ("ha0_imu - ha0_ext", lambda x: x["refs"]["ha0_imu_minus_ha0_ext"])):
                    L.append(f"| {lab} | {c(f(g['0']))} | {c(f(g['1']))} |")
                for arm in ARMS:
                    for k, lab in (("minus_ha0_ext", "- ha0_ext (E1 form; lever, NOT a bar)"), ("minus_ha0_ext_pose", "- ha0_ext_pose"), ("minus_ha0_imu", "- ha0_imu")):
                        if arm in ("RS_XF", "GATED_RS_XF", "BLEND_imu_XF") or k == "minus_ha0_ext_pose":
                            L.append(f"| {arm} {lab} | {c(g['0']['arms'][arm][k])} | {c(g['1']['arms'][arm][k])} |")
                    L.append(f"| {arm} - os (ADE) | {c(g['0']['arms'][arm]['minus_os']['ade_m'])} | {c(g['1']['arms'][arm]['minus_os']['ade_m'])} |")
                L.append("")
                L += ["| metric (family) | os - ha0_ext_pose s0 | RS_XF - refcv7 s0 | s1 | RS_XF - os s0 | s1 | GATED_RS_XF - refcv7 s0 | GATED_RS_XF - os s0 |",
                      "|---|---|---|---|---|---|---|---|"]
                for fam, k, *_r in r["metrics"]:
                    b0, b1, q0 = g["0"]["arms"]["RS_XF"], g["1"]["arms"]["RS_XF"], g["0"]["arms"]["GATED_RS_XF"]
                    L.append(f"| {k} ({fam}) | {c(g['0']['refs']['os_minus_ha0_ext_pose'][k])} | {c(b0['NR_minus_refcv7'][k])} | {c(b1['NR_minus_refcv7'][k])} | "
                             f"{c(b0['minus_os'][k])} | {c(b1['minus_os'][k])} | {c(q0['NR_minus_refcv7'][k])} | {c(q0['minus_os'][k])} |")
                sm = tr["summary"].get(f"{surf}|{st}", {})
                if sm:
                    L.append(f"\n* E1 registered verdict (os floor {sm['os_floor']:.4f}): **{sm['E1_registered_verdict']}**; os - ha0_ext_pose form: "
                             f"{sm['os_minus_ha0_ext_pose_verdict_form']}; os - ha0_imu form: {sm['os_minus_ha0_imu_verdict_form']}.")
                    for arm in ("RS_XF", "GATED_RS_XF"):
                        x = sm[arm]
                        L.append(f"* {arm} (floor {x['arm_floor']:.4f}): -ha0_ext form **{x['minus_ha0_ext_E1_FORM_lever_not_bar']}**; -ha0_ext_pose "
                                 f"{x['minus_ha0_ext_pose_form']}; -os {x['minus_os_form']}; NR rollup (headline rows) {x['NR_rollup_headline_rows']}.")
                        L.append(f"  * vs refcv7, separated WORSE at both seeds (all rows, direction-aware): {x['vs_refcv7_separated_worse_both_seeds_all_rows'] or 'none'}; "
                                 f"BETTER: {x['vs_refcv7_separated_better_both_seeds_all_rows'] or 'none'}")
                        L.append(f"  * vs refcv8 os, separated WORSE at both seeds: **{x['vs_os_separated_worse_both_seeds_all_rows'] or 'none'}**; "
                                 f"BETTER: {x['vs_os_separated_better_both_seeds_all_rows'] or 'none'}")
                fs = g["0"]["fold_seed_sensitivity"]
                op = g["0"]["controls"]["c_d_optimism"]
                L.append(f"* fold-seed sensitivity (s0, 10 seeds, points): RS_XF -ha0_ext {fs['RS_XF']['minus_ha0_ext_range'][0]:+.4f}..{fs['RS_XF']['minus_ha0_ext_range'][1]:+.4f}, "
                         f"GATED_RS_XF -ha0_ext {fs['GATED_RS_XF']['minus_ha0_ext_range'][0]:+.4f}..{fs['GATED_RS_XF']['minus_ha0_ext_range'][1]:+.4f}; optimism "
                         f"(-pose) XF-IS {op['RS_XF_minus_IS']:+.4f} / gated {op['GATED_RS_XF_minus_IS']:+.4f}, XF-WF {op['RS_XF_minus_WF']:+.4f}.")
                L.append(f"* in-sample RS gains g_k (s0) per instant [along, cross]: {g['0']['arms']['RS_IS']['weights']['fold0']['w']}")
                for sd_ in ("0", "1"):
                    L.append(f"* RS_XF fold-mean gains g_k (s{sd_}) per instant [along, cross]: {fold_mean(g[sd_]['arms']['RS_XF']['weights'])}; "
                             f"GATED_RS_XF fold-mean per v0 bin: {fold_mean_bins(g[sd_]['arms']['GATED_RS_XF']['weights'])}")
                if st == "all":
                    L.append(f"* validation (s0): V1 {V['V1_ha0_ext_rebuilt_vs_panel_max_abs_m']:.2e} m, V2 {V['V2_imu_with_psidot_kappa0v0_a0zero_vs_unicycle_kappa0_max_abs_m']:.2e} m, "
                             f"V3 imu vs ext >0.1 m {V['V3_frac_imu_vs_ha0_ext_gt_0p1m']}, pose vs ext >0.1 m {V['V3_frac_pose_vs_ha0_ext_gt_0p1m']}, "
                             f"pose vs ext max abs {V['ha0_ext_pose_vs_ha0_ext_max_abs_m']:.3f} m.")
                L.append("")
    (OUT / "A0K0_HEADROOM.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compute", "render", "freeze", "forward", "resummarise"])
    ap.add_argument("--tree", default=None, help="forward: the tree to apply the frozen gains to")
    ap.add_argument("--trees", nargs="*", default=list(TREES))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--surfs", nargs="*", default=list(SURFS))
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    a = ap.parse_args()
    if a.cmd == "render":
        return render()
    if a.cmd == "resummarise":
        return resummarise()
    if a.cmd == "freeze":
        return freeze()
    if a.cmd == "forward":
        if not a.tree:
            raise SystemExit("forward needs --tree")
        return forward(a.tree)
    return compute(a.trees, tuple(a.seeds), a.surfs, a.sets)


if __name__ == "__main__":
    sys.exit(main())
