"""Slowed copies of a trajectory -- the same PATH, traversed `factor` times as fast. ONE implementation.

REFe's trajectory format: [..., T, 3] = (x, y, heading) in the t0 rear-axle ego frame, sampled at t = dt, 2 dt, ...,
T dt (planner.TRAJ_DT_S = 0.2 s, T = 20). The ego is at the origin (0, 0, 0) at t = 0 and is NOT part of the array.

    slow_copy(traj, factor, dt=0.2) -> same shape, same dtype, same type (numpy array or torch tensor)
        out[..., k, :] = traj(factor * t_k),   t_k = (k + 1) * dt,   k = 0 .. T-1

traj(tau) BETWEEN samples follows the seam converter's own rule (eval/refe_navtest_seam.py::to_navsim): a sample
time is copied verbatim; in between, x and y are linear in time and the heading is the start heading plus the
WRAPPED heading difference times the weight, wrapped to [-pi, pi); the origin (0, 0, 0) is the sample at tau = 0.
Consequences (pinned by eval/selftest_slow_copies.py, including deliberately broken variants that must fail):
  * factor 1.0 returns the input BIT-FOR-BIT (every query time is a sample time);
  * every output point lies ON the input polyline origin -> p_1 -> ... -> p_T (the path is kept, not re-shaped);
  * where both ends of an output step fall inside one input segment, the step's speed is exactly
    factor x that segment's speed -- the speed profile is v_copy(t) = factor * v(factor * t);
  * the heading is the input's OWN heading at that point of the path (the model's orientation; a tangent recomputed
    from the positions would change the pose even at factor 1.0);
  * at t = 0 the copy moves at factor x the input's initial speed: a copy of a plan that starts at the ego's speed
    asks for an immediate slow-down. That is what a time-rescaled plan is; it is stated, not hidden.
Domain: 0 < factor <= 1 (a faster copy would need poses beyond the horizon, which do not exist).

Arithmetic: float64 internally, returned in the input's dtype. The interpolation plan (which samples, which weight)
depends only on (T, factor, dt) and is computed ONCE with scalar float64 arithmetic, then applied element-wise over
all leading dimensions -- so a batch, a single trajectory and the numpy and torch paths all use identical weights.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["slow_copy", "slow_copy_plan"]

_TOL = 1e-9          # a query time within this of a sample time IS that sample (the converter's tolerance)


def _wrap(a):
    """to [-pi, pi); numpy or torch (both `%` are floor-modulo)"""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def slow_copy_plan(T: int, factor: float, dt: float = 0.2):
    """[(j, w)] per output step: j indexes [origin, p_1, ..., p_T]; w None = copy sample j verbatim, else interpolate
    between sample j-1 and sample j with weight w (np.float64)."""
    f = float(factor)
    if not (0.0 < f <= 1.0):
        raise ValueError(f"factor must be in (0, 1], got {factor!r}")
    dt = float(dt)
    src_t = dt * np.arange(0, T + 1)
    plan = []
    for k in range(T):
        tq = f * dt * (k + 1)
        j = int(np.searchsorted(src_t, tq - _TOL))
        if j < len(src_t) and abs(src_t[j] - tq) < _TOL:
            plan.append((j, None))
            continue
        if j == 0 or j >= len(src_t):
            raise ValueError(f"query time {tq} outside the source grid [0, {src_t[-1]}]")
        plan.append((j, (tq - src_t[j - 1]) / (src_t[j] - src_t[j - 1])))
    return plan


def slow_copy(traj, factor: float, dt: float = 0.2):
    """[..., T, 3] (x, y, heading), ego frame, samples at dt..T*dt -> the same path at `factor` x the speed.

    numpy in -> numpy out; torch in -> torch out (same device). float64 internally, the input's dtype out."""
    if traj.shape[-1] != 3:
        raise ValueError(f"expected [..., T, 3] (x, y, heading), got shape {tuple(traj.shape)}")
    plan = slow_copy_plan(int(traj.shape[-2]), factor, dt)
    if not isinstance(traj, np.ndarray) and hasattr(traj, "new_zeros"):          # a torch tensor
        import torch
        x = traj.to(torch.float64)
        src = torch.cat([torch.zeros_like(x[..., :1, :]), x], dim=-2)
        steps = []
        for j, w in plan:
            if w is None:
                steps.append(src[..., j, :])
                continue
            w = float(w)
            a, b = src[..., j - 1, :], src[..., j, :]
            xy = (1 - w) * a[..., :2] + w * b[..., :2]
            h = _wrap(a[..., 2] + w * _wrap(b[..., 2] - a[..., 2]))
            steps.append(torch.cat([xy, h[..., None]], dim=-1))
        return torch.stack(steps, dim=-2).to(traj.dtype)
    x = np.asarray(traj)
    dtype = x.dtype
    x = x.astype(np.float64)
    src = np.concatenate([np.zeros(x.shape[:-2] + (1, 3)), x], axis=-2)
    out = np.empty_like(x)
    for k, (j, w) in enumerate(plan):
        if w is None:
            out[..., k, :] = src[..., j, :]
            continue
        a, b = src[..., j - 1, :], src[..., j, :]
        out[..., k, :2] = (1 - w) * a[..., :2] + w * b[..., :2]
        out[..., k, 2] = _wrap(a[..., 2] + w * _wrap(b[..., 2] - a[..., 2]))
    return out.astype(dtype)
