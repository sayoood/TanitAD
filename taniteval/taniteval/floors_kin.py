"""floors_kin -- the learned / fixed KINEMATIC FLOORS as pure functions of t0 kinematics.

H-V8F-FLOOR-1 (``Project Steering/PREREG_HJ_STAGE0.md``, stream S1): does a driver beat the learned
kinematic floor?  Every function here maps (past poses / t0 kinematics) -> a trajectory.  Nothing reads a
frame, a model or a label, and nothing here reads a FUTURE value except ``gt_waypoints`` (the target).

PROVENANCE -- ported VERBATIM, never re-derived (a second implementation of a shared floor is how two
"independent" checks agree on a wrong answer):

* ``kin_features``  <- ``probe.py::kin_feats`` (refav1 trunk-probe package, 2026-09-27)
* ``mlp_curve`` / ``fit_p10`` <- ``probe_te.py::mlp_curve`` / ``fit_mlp_kin`` (the SPEC-A3 "P10" arm):
  one hidden layer (h=256, GELU), zero-initialised output layer (starts AT the prior), AdamW lr 1e-3,
  batch 256, 60 epochs, wd in {1e-4, 1e-2} and the epoch count chosen by a 4-fold GROUPED inner CV on the
  fit rows only, then refit on all fit rows.
* ``retime`` <- ``build_explore.py::retime`` (kd_x), vectorised; ``tests/test_floors_kin.py`` pins the
  vectorised form against a verbatim copy of the scalar original.
* The integrator reproduces ``tanitad.models.kinematic.rollout_unicycle``: position and heading advance on
  the speed at the START of the step, v is updated LAST and clamped at 0.

THE EIGHT FEATURES -- ``kin = (v0, a0, k0)`` is exactly ``(v0, ext[0], ext[1])`` of
``refav1_arm.hold_ext_controls``: ``a0 = (v[i0] - v[i0 - stride]) / dt`` (backward difference, every index
<= i0) and ``k0 = actions[i0, 0]`` -- the RECORDED CHANNEL-1 VALUE, which is a road-wheel STEER ANGLE
(``physicalai.py`` writes ``arctan(L * kappa)``), NOT a curvature.  The features take it as recorded
(unchanged from P10); only the INTEGRATION converts (``action_units="steer"`` -> ``kappa = tan(steer)/L``).

THE TWO FLOORS THAT ARE NOT LEARNED
* ``damp50 = 0.5 * ha0 + 0.5 * ha0_ext`` -- ``ha0`` is the constant-velocity straight line and ``ha0_ext``
  the constant-(a0, k0) extrapolation.
* ``kd_x`` = the damp50 PATH, re-timed so its along-track distance at each step equals ``ha0_ext``'s own
  travelled distance (lateral geometry from damp50, longitudinal profile from ha0_ext).
"""
from __future__ import annotations

import math

import numpy as np

STEER_WHEELBASE_M = 2.9          # the ENCODER's L (kinematic.STEER_WHEELBASE_M)
N_INNER = 4                      # probe.py::N_INNER
_EPS_SEG = 1e-9


# --------------------------------------------------------------------------- #
# kinematics at t0                                                             #
# --------------------------------------------------------------------------- #
def curvature_of_steer(steer, wheelbase: float = STEER_WHEELBASE_M):
    """kappa = tan(steer) / L  (kinematic.kappa_of_steer)."""
    return np.tan(np.asarray(steer, dtype=np.float64)) / float(wheelbase)


def t0_kinematics(v, steer, t0, *, stride: int, dt: float) -> np.ndarray:
    """``[n, 3]`` = (v0, a0, k0) at window origins ``t0`` of one episode.

    ``v`` = ``poses[:, 3]``, ``steer`` = ``actions[:, 0]`` (both on the 10 Hz frame grid).
    ``a0 = (v[t0] - v[t0 - stride]) / dt``; every index <= t0 (nothing recorded after t0 enters).
    ``stride``/``dt`` are the GRID's: (1, 0.1) for the refcv grid, (2, 0.2) for refav1's.
    """
    t0 = np.asarray(t0, dtype=np.int64)
    if t0.size and int(t0.min()) < stride:
        raise ValueError(f"ha0_ext needs one closed step before t0 (stride={stride}, min t0={int(t0.min())})")
    v = np.asarray(v, dtype=np.float64)
    steer = np.asarray(steer, dtype=np.float64)
    return np.stack([v[t0], (v[t0] - v[t0 - stride]) / float(dt), steer[t0]], axis=-1)


def kin_features(kin) -> np.ndarray:
    """``[n, 8]`` = (v, a, k, v^2, v*a, v*k, v^2*k, a*k) -- ``probe.py::kin_feats`` verbatim."""
    kin = np.asarray(kin, dtype=np.float64)
    v, a, k = kin[:, 0], kin[:, 1], kin[:, 2]
    return np.stack([v, a, k, v * v, v * a, v * k, v * v * k, a * k], 1)


# --------------------------------------------------------------------------- #
# constant-control paths (the model-free floors)                               #
# --------------------------------------------------------------------------- #
def const_control_paths(v0, a, k, dt: float, n: int, *, action_units: str) -> np.ndarray:
    """``[n_w, n, 2]`` ego-frame (x forward, y left) paths under CONSTANT (accel ``a``, channel-1 ``k``).

    Reproduces ``tanitad.models.kinematic.rollout_unicycle`` in float64.  ``action_units="steer"`` applies
    ``kappa = tan(k)/L`` first (the correct reading of a RECORDED action); ``"kappa"`` integrates ``k`` as
    supplied (the legacy refav1 probe path, kept so the known-value control is the same code path).

    REQUIRED, KEYWORD-ONLY, NO DEFAULT (BACKLOG HJ-UNITS; RETRACTION_LOG R46, 2026-10-17). A call that omits
    it raises ``TypeError``; an unknown value raises ``ValueError`` naming both legal values.  It used to
    default to ``"kappa"``, and the module's own main input -- ``kin = (v0, a0, k0)`` with ``k0 = actions[:, 0]``,
    a road-wheel STEER ANGLE (see the module docstring) -- is exactly what a defaulted call would integrate
    as a curvature: the over-rotated echo control (5.356 m instead of 2.838 m) that manufactured a fake
    -1.90 m "lever".  The unit is a property of WHERE THE CONTROLS CAME FROM, so the caller must state it.
    """
    if action_units not in ("kappa", "steer"):
        raise ValueError(f"action_units must be kappa|steer, got {action_units!r}")
    v = np.asarray(v0, dtype=np.float64).copy()
    a = np.broadcast_to(np.asarray(a, dtype=np.float64), v.shape)
    kap = np.broadcast_to(np.asarray(k, dtype=np.float64), v.shape)
    if action_units == "steer":
        kap = curvature_of_steer(kap)
    x = np.zeros_like(v)
    y = np.zeros_like(v)
    yaw = np.zeros_like(v)
    out = np.empty(v.shape + (int(n), 2), dtype=np.float64)
    for j in range(int(n)):
        x = x + v * np.cos(yaw) * dt
        y = y + v * np.sin(yaw) * dt
        yaw = yaw + v * kap * dt
        v = np.maximum(v + a * dt, 0.0)
        out[..., j, 0] = x
        out[..., j, 1] = y
    return out


def ha0_paths(v0, dt: float, n: int) -> np.ndarray:
    """The constant-velocity STRAIGHT line (a = 0, k = 0) -- exactly zero in either action unit."""
    z = np.zeros_like(np.asarray(v0, dtype=np.float64))
    # zero controls are 0 in either unit; "kappa" applies no conversion (= the pre-HJ-UNITS call, bit for bit)
    return const_control_paths(v0, z, z, dt, n, action_units="kappa")


def ha0_ext_paths(kin, dt: float, n: int, *, action_units: str) -> np.ndarray:
    """``ha0_ext``: constant (a0, k0) extrapolation of the MEASURED t0 state. ``kin`` = ``[n_w, 3]``.

    ``action_units`` is REQUIRED, keyword-only (BACKLOG HJ-UNITS / R46): ``kin[:, 2]`` is the RECORDED channel-1
    value, a road-wheel STEER ANGLE, so a recorded-action caller states ``"steer"``; only a caller that holds a
    true curvature states ``"kappa"``.  See :func:`const_control_paths`."""
    kin = np.asarray(kin, dtype=np.float64)
    return const_control_paths(kin[:, 0], kin[:, 1], kin[:, 2], dt, n, action_units=action_units)


def damp50(ha0, ha0_ext) -> np.ndarray:
    """``0.5 * ha0 + 0.5 * ha0_ext`` -- the damped floor."""
    return 0.5 * np.asarray(ha0, dtype=np.float64) + 0.5 * np.asarray(ha0_ext, dtype=np.float64)


# --------------------------------------------------------------------------- #
# kd_x -- vectorised ``build_explore.py::retime``                              #
# --------------------------------------------------------------------------- #
def _arclen_batch(p):
    """p [n, K, 2] positions at k = 1..K (origin implicit) -> P [n,K+1,2], seg, L [n,K], S [n,K+1]."""
    n = p.shape[0]
    P = np.concatenate([np.zeros((n, 1, 2), p.dtype), p], axis=1)
    seg = np.diff(P, axis=1)
    L = np.linalg.norm(seg, axis=-1)
    S = np.concatenate([np.zeros((n, 1), L.dtype), np.cumsum(L, axis=1)], axis=1)
    return P, seg, L, S


def retime(path, lon_src, *, chunk: int = 2048) -> np.ndarray:
    """Re-time ``path`` so its along-track distance at step k equals ``lon_src``'s own travelled distance.

    ``path`` and ``lon_src``: ``[n, K, 2]`` (or ``[K, 2]``), positions at steps 1..K, origin implicit.
    Returns float32, like the original.  Semantics, case by case, are ``build_explore.py::retime``:

    * degenerate damped path (total length < 1e-6, or no segment longer than 1e-9): straight ahead ``(s, 0)``;
    * ``s <= total length``: linear interpolation inside the segment that holds ``s`` (a zero-length segment
      is skipped forward); ``s`` beyond the end: extrapolate along the LAST non-zero segment.
    """
    single = np.asarray(path).ndim == 2
    path = np.asarray(path, dtype=np.float64)
    lon_src = np.asarray(lon_src, dtype=np.float64)
    if single:
        path, lon_src = path[None], lon_src[None]
    n, K, _ = path.shape
    Ks = lon_src.shape[1]
    out = np.empty((n, Ks, 2), dtype=np.float64)
    for c0 in range(0, n, chunk):
        sl = slice(c0, min(n, c0 + chunk))
        P, seg, L, S = _arclen_batch(path[sl])
        s_src = _arclen_batch(lon_src[sl])[3][:, 1:]                         # [m, Ks]
        m = P.shape[0]
        nz = L > _EPS_SEG
        deg = (S[:, -1] < 1e-6) | ~nz.any(1)
        # last non-zero segment, and "next non-zero at or after i" (the original's while-loop)
        jj = (K - 1) - np.argmax(nz[:, ::-1], axis=1)
        nxt = np.empty((m, K), dtype=np.int64)
        nxt[:, K - 1] = K - 1
        for i in range(K - 2, -1, -1):
            nxt[:, i] = np.where(nz[:, i], i, nxt[:, i + 1])
        j = (S[:, None, :] <= s_src[:, :, None]).sum(-1) - 1                  # searchsorted(S, s, 'right') - 1
        j = np.clip(j, 0, K - 1)
        j = np.take_along_axis(nxt, j, axis=1)
        take = lambda arr, idx: np.take_along_axis(arr, idx, axis=1)           # noqa: E731
        Lj = take(L, j)
        Pj = np.take_along_axis(P, j[..., None].repeat(2, -1), axis=1)
        segj = np.take_along_axis(seg, j[..., None].repeat(2, -1), axis=1)
        Sj = take(S, j)
        inside = Pj + ((s_src - Sj) / np.maximum(Lj, 1e-12))[..., None] * segj
        res = np.where((Lj > _EPS_SEG)[..., None], inside, Pj)
        # beyond the end of the damped path
        jj_c = np.clip(jj, 0, K - 1)[:, None]
        Ljj = take(L, jj_c)
        segjj = np.take_along_axis(seg, jj_c[..., None].repeat(2, -1), axis=1)
        beyond = P[:, -1:, :] + ((s_src - S[:, -1:]) / np.maximum(Ljj, 1e-12))[..., None] * segjj
        res = np.where((s_src <= S[:, -1:])[..., None], res, beyond)
        straight = np.stack([s_src, np.zeros_like(s_src)], axis=-1)
        res = np.where(deg[:, None, None], straight, res)
        out[sl] = res
    out = out.astype(np.float32)
    return out[0] if single else out


def kd_x(ha0, ha0_ext) -> np.ndarray:
    """``kd_x``: the damp50 path re-timed by ``ha0_ext``'s own travelled distance (float32).

    The STEP RESOLUTION of ``ha0``/``ha0_ext`` is part of the definition (arc length is measured on the
    polyline you pass): the refav1 package's ``kd_x_6`` is the 30 x 0.2 s polyline; the refcv7 grids here
    are re-timed on the native 60 x 0.1 s polyline (S6) / 20 x 0.1 s (S2) and then index-selected.
    """
    return retime(damp50(ha0, ha0_ext), np.asarray(ha0_ext))


# --------------------------------------------------------------------------- #
# ground truth + scores                                                        #
# --------------------------------------------------------------------------- #
def gt_waypoints(poses, t0, frames) -> np.ndarray:
    """Ego-frame GT waypoints: ``[len(t0), len(frames), 2]`` at ``t0 + frames`` (10 Hz frame offsets).

    ``poses`` ``[T, 4]`` = (x, y, yaw, v); rotation by ``-yaw(t0)`` (metric_dynamics.ego_delta).
    """
    poses = np.asarray(poses, dtype=np.float64)
    t0 = np.asarray(t0, dtype=np.int64)
    fr = np.asarray(frames, dtype=np.int64)
    d = poses[t0[:, None] + fr[None, :], :2] - poses[t0, None, :2]
    c, s = np.cos(-poses[t0, 2])[:, None], np.sin(-poses[t0, 2])[:, None]
    return np.stack([d[..., 0] * c - d[..., 1] * s, d[..., 0] * s + d[..., 1] * c], axis=-1)


def ade(pred, g) -> np.ndarray:
    """Per-window mean L2 over the horizon (``probe.py::ade``)."""
    return np.linalg.norm(np.asarray(pred) - np.asarray(g), axis=-1).mean(-1)


def fde(pred, g) -> np.ndarray:
    """Per-window L2 at the LAST step."""
    return np.linalg.norm(np.asarray(pred)[:, -1] - np.asarray(g)[:, -1], axis=-1)


def acts_check(path, dt: float, *, zero_tol: float = 1e-3) -> dict:
    """The "acts" check: does the arm ACCELERATE at all?

    Accelerations are the second difference of path-implied speed, ``speed[k] = |p[k] - p[k-1]| / dt`` with
    the origin prepended and the derivative taken forward -- the convention of ``kinematic.
    unicycle_controls_from_path`` and ``four_families._seq_geometry``.  ``mean_abs_a`` is the mean |a| over
    all windows and steps (m/s^2); ``zero_accel_frac`` the fraction of (window, step) cells with
    |a| < ``zero_tol`` m/s^2.  A constant-velocity line is 1.0 on both-ends cells.
    """
    p = np.asarray(path, dtype=np.float64)
    P = np.concatenate([np.zeros((p.shape[0], 1, 2)), p], axis=1)
    speed = np.linalg.norm(np.diff(P, axis=1), axis=-1) / float(dt)           # [n, K]
    acc = np.diff(speed, axis=1) / float(dt)                                  # [n, K-1]
    return {"mean_abs_a_mps2": float(np.abs(acc).mean()),
            "zero_accel_frac": float((np.abs(acc) < zero_tol).mean()),
            "zero_tol_mps2": float(zero_tol), "n_windows": int(p.shape[0]), "n_steps": int(acc.shape[1])}


# --------------------------------------------------------------------------- #
# P10 -- the learned kinematic floor (probe_te.py::mlp_curve / fit_mlp_kin)    #
# --------------------------------------------------------------------------- #
def _torch():
    import torch
    return torch


def _make_mlp(d_in: int, d_out: int, h: int = 256):
    torch = _torch()
    net = torch.nn.Sequential(torch.nn.Linear(d_in, h), torch.nn.GELU(), torch.nn.Linear(h, d_out))
    torch.nn.init.zeros_(net[2].weight)
    torch.nn.init.zeros_(net[2].bias)                      # starts AT the prior
    return net


def mlp_curve(Xtr, ytr, Xva, base_va, g_va, wd, epochs, seed=0):
    """``probe_te.py::mlp_curve`` verbatim (same RNG draws, same optimiser, same batch order)."""
    torch = _torch()
    torch.manual_seed(seed)
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8
    ym = ytr.mean(0)
    net = _make_mlp(Xtr.shape[1], ytr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    Xt = torch.tensor((Xtr - xm) / xs, dtype=torch.float32)
    Yt = torch.tensor(ytr - ym, dtype=torch.float32)
    Xv = None if Xva is None else torch.tensor((Xva - xm) / xs, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    curve = []
    for _ep in range(epochs):
        perm = rng.permutation(len(Xt))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if Xva is not None:
            with torch.no_grad():
                yh = net(Xv).numpy() + ym
            curve.append(float(ade(base_va + yh.reshape(-1, *base_va.shape[1:]), g_va).mean()))
    return net, (xm, xs, ym), curve


def fit_p10(kf, y, base, g, groups, wds=(1e-4, 1e-2), epochs=60, seed=0):
    """P10: the MLP on the kinematic features ONLY, residual on ``base`` (kd_x).

    ``kf`` ``[n, 8]`` (``kin_features``), ``y`` ``[n, K*2]`` = ``(g - base).reshape(n, -1)``, ``base``/``g``
    ``[n, K, 2]``, ``groups`` the EPISODE id per row.  Inner 4-fold GROUPED CV (fold = rank of the episode
    mod 4) over (weight decay, epoch count); the winner is refit on every row given.  -> (net, stats, best)
    with ``best = (0, wd, epochs)`` and ``stats = (xm, xs, ym)``.  ``seed`` is the TRAINING seed (init + batch
    order); 0 is the banked ``fit_mlp_kin`` behaviour, other values are replicate fits for the floor's own seed variance.
    """
    ug = np.unique(groups)
    inner = {e: i % N_INNER for i, e in enumerate(ug)}
    fold = np.array([inner[e] for e in groups])
    score = {}
    for f in range(N_INNER):
        tr, te = fold != f, fold == f
        for wd in wds:
            _, _, cur = mlp_curve(kf[tr], y[tr], kf[te], base[te], g[te], wd, epochs, seed=seed)
            score.setdefault(wd, []).append(np.array(cur) * te.sum())
    tot = len(groups)
    best, best_s = None, np.inf
    for wd, v in score.items():
        c = np.sum(v, 0) / tot
        e = int(np.argmin(c))
        if c[e] < best_s:
            best, best_s = (0, wd, e + 1), float(c[e])
    net, st, _ = mlp_curve(kf, y, None, None, None, best[1], best[2], seed=seed)
    return net, st, best


def predict_p10(model, kf, base) -> np.ndarray:
    """``base + MLP(kf)``  ->  ``[n, K, 2]`` float64."""
    torch = _torch()
    net, (xm, xs, ym), _best = model
    with torch.no_grad():
        yh = net(torch.tensor((np.asarray(kf) - xm) / xs, dtype=torch.float32)).numpy() + ym
    base = np.asarray(base)
    return base + yh.reshape(-1, *base.shape[1:])


def fit_predict_p10(kf_fit, y_fit, base_fit, g_fit, groups_fit, kf_pred, base_pred, **kw):
    """Fit on the fit rows, predict the rest; -> (pred, best)."""
    net, st, best = fit_p10(kf_fit, y_fit, base_fit, g_fit, groups_fit, **kw)
    return predict_p10((net, st, best), kf_pred, base_pred), best


def grouped_folds(groups, n_outer: int = 5) -> np.ndarray:
    """Outer fold per row = (rank of the episode in sorted order) mod ``n_outer`` -- folds BY EPISODE."""
    ug = np.unique(groups)
    rank = {e: i % n_outer for i, e in enumerate(ug)}
    return np.array([rank[e] for e in groups])


def late_fusion(os_pred, p10_pred, alpha) -> np.ndarray:
    """EXPLORATORY: ``alpha * os + (1 - alpha) * P10``."""
    return float(alpha) * np.asarray(os_pred, dtype=np.float64) + (1.0 - float(alpha)) * np.asarray(p10_pred, dtype=np.float64)


def fit_alpha(os_pred, p10_pred, g, alphas=None) -> float:
    """Closed-form grid argmin of the mean ADE of the blend over ``alphas`` (default 0..1 step 0.05)."""
    alphas = np.linspace(0.0, 1.0, 21) if alphas is None else np.asarray(alphas)
    best, best_a = math.inf, 0.0
    for a in alphas:
        m = float(ade(late_fusion(os_pred, p10_pred, a), g).mean())
        if m < best:
            best, best_a = m, float(a)
    return best_a
