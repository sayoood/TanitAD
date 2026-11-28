"""The `hold` executor of the v8F T2 harness (stage 2): the playback executor with a heading that follows the tangent only when the executed step EXCEEDS eps.

Card of the other read: `motion_c/exec_standstill/PREREG_EXEC_STANDSTILL.md` section 3 (variant executor `hold`, eps = 0.1 m per 0.2 s knot = the reader's own standstill mask,
ds <= 0.5 m/s x 0.2 s). The frozen T2 card runs `base`. `hold` only ever runs under a later dated amendment.

This file is an INDEPENDENT implementation (it does not read the other agent's code). It does not edit `t2v8f_exec.py` or the T1 tool; it
  * restates the tool's `hermite_frames` with one extra parameter (`eps`): the heading is updated iff the tangent norm per 0.2 s exceeds eps (the tool: iff it is >= EPS_M = 1e-3),
  * subclasses `PlaybackExecutor` and replaces only `commit` (everything else is inherited),
  * is switched on by `install(X, eps)` (it rebinds `X.PlaybackExecutor`, which `run_t2_rollout` looks up at call time) and off by `uninstall(X)`.
Positions, speeds, the replans (3 x 10 knots), the windows and the causal bucket are untouched: only the yaw column of the executed track can differ.

Gates (tests/test_t2v8f_stage2.py, each with a RED arm):
  G-ID    eps = 1e-3 reproduces the base executor bit for bit (random, moving and standstill-noise plans); RED: eps = 0.1 differs on standstill-noise plans
  G-MOV   when every executed step exceeds eps, hold(0.1) equals base exactly
  G-PHYS  under hold, no raw row whose tangent is <= eps changes the heading; RED: hold disabled (eps = 1e-3) breaks it on a standstill-noise plan
"""
from __future__ import annotations

import math

import numpy as np

HOLD_EPS_DEFAULT = 0.1                       # m per 0.2 s knot (= 0.5 m/s)
DT = 0.2


def hermite_frames_eps(k_prev, knots, m: int, psi0: float, eps: float, dt: float = DT):
    """`hermite_frames` of the T1 tool (cubic Hermite, Catmull-Rom tangents), restated, with the heading rule `norm > eps` instead of `norm >= EPS_M`.
    Returns (xy [2m, 2], psi [2m], v [2m]); the odd raw frame of interval i is the Hermite midpoint, the even frame the knot itself."""
    K = np.vstack([np.asarray(k_prev, np.float64)[None], np.asarray(knots, np.float64)])
    nk = len(K)
    if nk < m + 2:
        raise ValueError(f"need at least m+1={m + 1} knots, got {nk - 1}")

    def tang(i):
        return 0.5 * (K[i + 1] - K[i - 1]) if i + 1 < nk else K[i] - K[i - 1]

    xy, psi, v = [], [], []
    cur = float(psi0)
    for i in range(1, m + 1):
        t0, t1 = tang(i), tang(i + 1)
        mid = 0.5 * (K[i] + K[i + 1]) + (t0 - t1) / 8.0
        dmid = 1.5 * (K[i + 1] - K[i]) - 0.25 * (t0 + t1)
        for pos, d in ((mid, dmid), (K[i + 1], t1)):
            nrm = float(np.hypot(d[0], d[1]))
            if nrm > eps:
                cur = math.atan2(d[1], d[0])
            xy.append(pos)
            psi.append(cur)
            v.append(nrm / dt)
    return np.asarray(xy), np.asarray(psi), np.asarray(v)


def make_hold_class(base_cls, eps: float):
    """A subclass of the module's `PlaybackExecutor` whose `commit` uses `hermite_frames_eps(..., eps)`. `eps` is fixed per class (a test can build several)."""

    class HoldExecutor(base_cls):
        HOLD_EPS = float(eps)

        def commit(self, plan_ego, m: int = 10):
            T, j = self.tool, self.j
            pose = self.track[2 * j, :3]
            plan_ego, self.hit = T.envelope(plan_ego)
            W = T.plan_to_world(plan_ego, pose)
            if self.skew:
                W = np.roll(W, self.skew, axis=0)
            knots = np.vstack([pose[None, :2], W])
            xy, psi, v = hermite_frames_eps(self.track[2 * (j - 1), :2], knots, m, float(pose[2]), self.HOLD_EPS)
            new = np.column_stack([xy, psi, v])
            self.track = np.vstack([self.track, new])
            return new

    HoldExecutor.__name__ = "HoldExecutor"
    return HoldExecutor


def install(X, eps: float = HOLD_EPS_DEFAULT):
    """Rebind `X.PlaybackExecutor` (X = the `t2v8f_exec` module) to the hold executor with `eps`. Idempotent; the original class is kept in `X._PLAYBACK_BASE`."""
    if not hasattr(X, "_PLAYBACK_BASE"):
        X._PLAYBACK_BASE = X.PlaybackExecutor
    X.PlaybackExecutor = make_hold_class(X._PLAYBACK_BASE, eps)
    return X.PlaybackExecutor


def uninstall(X):
    """Restore the base executor (a no-op when `install` never ran)."""
    if hasattr(X, "_PLAYBACK_BASE"):
        X.PlaybackExecutor = X._PLAYBACK_BASE
    return X.PlaybackExecutor


def active(X) -> str:
    """'hold' or 'base', read from the class `run_t2_rollout` will instantiate."""
    return "hold" if hasattr(X.PlaybackExecutor, "HOLD_EPS") else "base"
