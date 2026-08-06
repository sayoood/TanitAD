"""Differentiable kinematic bicycle layer (H14 Track 1).

The trajectory head predicts *controls*; this layer integrates them through
explicit vehicle kinematics, so every decoded trajectory is physically
realizable by construction. Also provides the Kamm-circle friction penalty
(|a_lat^2 + a_lon^2| <= (mu g)^2) used as a standing metric and loss barrier.
"""

from __future__ import annotations

import torch
from torch import Tensor


def rollout_bicycle(state0: Tensor, controls: Tensor, dt: float = 0.1,
                    wheelbase: float = 2.7) -> Tensor:
    """Integrate the kinematic bicycle model.

    state0: [B, 4] = (x, y, yaw, v); controls: [B, K, 2] = (accel, steer);
    returns states [B, K, 4] after each step. Differentiable throughout.
    """
    x, y, yaw, v = state0.unbind(-1)
    out = []
    for k in range(controls.shape[1]):
        accel, steer = controls[:, k, 0], controls[:, k, 1]
        x = x + v * torch.cos(yaw) * dt
        y = y + v * torch.sin(yaw) * dt
        yaw = yaw + v / wheelbase * torch.tan(steer) * dt
        v = (v + accel * dt).clamp_min(0.0)
        out.append(torch.stack([x, y, yaw, v], dim=-1))
    return torch.stack(out, dim=1)


def kamm_circle_violation(controls: Tensor, v: Tensor, wheelbase: float = 2.7,
                          mu: float = 0.8, g: float = 9.81) -> Tensor:
    """Mean rectified friction-circle violation of a control sequence.

    controls: [B, K, 2] = (accel, steer); v: [B] entry speed (approximation:
    constant per short horizon). Returns scalar >= 0; 0 means fully feasible.
    """
    a_lon = controls[..., 0]
    a_lat = v.unsqueeze(-1).pow(2) / wheelbase * torch.tan(controls[..., 1]).abs()
    # clamp_min inside sqrt: at total accel = 0, sqrt'(0)=inf and relu'=0 give
    # the 0*inf=NaN sqrt-relu trap on the backward pass — bound the magnitude so
    # a fully-feasible (near-zero accel) control never NaNs the gradient.
    total = (a_lon.pow(2) + a_lat.pow(2)).clamp_min(1e-12).sqrt()
    return torch.relu(total - mu * g).mean()


# --------------------------------------------------------------------------- #
# UNICYCLE (accel, curvature) — the Alpamayo 2 Super action space               #
#                                                                              #
# ⭐ WHY THIS EXISTS, and it is a MEASURED motivation, not an architectural     #
# preference. On 39 paired OOD-val clips (2026-08-06, four binding families)    #
# our flagship's ACCELERATION MAE is 1.7644 m/s^2 against Alpamayo's 0.5077 —   #
# 3.48x worse — while our LATERAL family is competitive and we BEAT it on       #
# curvature MAE (0.007551 vs 0.009162 1/m). The deficit is longitudinal and it  #
# is specifically in the ACCELERATION PROFILE, which a free-waypoint head does  #
# not represent and a scalar ADE cannot see.                                    #
#                                                                              #
# A head that emits (accel, curvature) and integrates them is regularised on    #
# exactly the quantity we are worst at: acceleration is a decision variable     #
# rather than a second difference of an unconstrained output.                   #
#                                                                              #
# ⛔ WHY UNICYCLE AND NOT THE BICYCLE ABOVE. `rollout_bicycle` takes            #
# (accel, STEER) and applies `v/L * tan(steer)`. That couples the lateral       #
# control to a wheelbase constant we do not measure per-clip, and `tan` blows   #
# up near +-pi/2 — a head free to emit large steer produces an unbounded yaw    #
# rate and a NaN-adjacent gradient. Curvature is the geometric quantity the     #
# vehicle actually follows, is directly comparable to the `curvature_mae_1pm`   #
# we already report in the LATERAL family, and needs no wheelbase. Alpamayo's   #
# `UnicycleAccelCurvatureActionSpace` makes the same choice, with a = [-9.8,    #
# 9.8] m/s^2 and kappa = [-0.33, 0.33] 1/m (PUBLISHED, their config).            #
#                                                                              #
# ⚠️ `rollout_bicycle` above is currently DEAD CODE — exported from             #
# `models/__init__` and NaN-tested, but imported by NO model or trainer         #
# (verified 2026-08-06 by grep over `stack/`). The flagship emits free          #
# waypoints. Wiring a control-integrating head is therefore a NEW capability,   #
# not a switch; this function is its kernel and is deliberately kept pure so it #
# can be unit-tested against the four-family instrument before any GPU is spent.#
# --------------------------------------------------------------------------- #

#: PUBLISHED — Alpamayo 2 Super's own action-space bounds. Kept as named
#: constants so an arm that deviates has to say so rather than drift.
A2S_ACCEL_LIMIT = 9.8          # m/s^2
A2S_CURVATURE_LIMIT = 0.33     # 1/m

#: Below this step SPEED the path tangent — and therefore curvature — carries no
#: information. Same constant and same meaning as ``four_families.MIN_DS_MPS``, kept
#: here rather than imported because ``tanitad`` must not depend on ``taniteval``.
MIN_DS_MPS = 0.5
_DS_EPS = 1e-8




def _squash(x: Tensor, limit: float) -> Tensor:
    """Bound ``x`` to (-limit, limit) while keeping a NON-ZERO gradient everywhere.

    ⛔ THREE CANDIDATES, TWO OF THEM TRAPS.
    ``clamp`` has exactly zero gradient outside the range, so a head that initialises
    outside it can never learn back in — a silent dead head.
    ``tanh`` looks like the fix and is not: MEASURED 2026-08-06, ``1 - tanh(51)**2``
    evaluates to **exactly 0.0** in float32, so a control 500 m/s² against a 9.8 limit
    lands in a region where the gradient has UNDERFLOWED to zero. tanh does not remove
    the cliff, it moves it out to where nobody tests.
    Softsign ``x / (1 + |x|/limit)`` decays as 1/x² instead of exponentially: at the
    same 51× overshoot the gradient is ~3.7e-4 — small, but representable and nonzero,
    so the head recovers instead of dying.
    """
    return x / (1.0 + x.abs() / limit)


def rollout_unicycle(state0: Tensor, controls: Tensor, dt: float = 0.1,
                     accel_limit: float | None = None,
                     curvature_limit: float | None = None) -> Tensor:
    """Integrate the unicycle model under (accel, curvature) controls.

    ``state0`` [B, 4] = (x, y, yaw, v); ``controls`` [B, K, 2] = (accel, curvature);
    returns states [B, K, 4] after each step. Differentiable throughout.

    ``yaw_rate = v * curvature`` — the defining identity of the model, and the reason
    a stopped vehicle cannot turn here (v = 0 => yaw_rate = 0). That is physically
    right and is NOT an edge case to patch around: a head that "turns" at a standstill
    is producing a path no vehicle drives.

    ⛔ THE ORDER OF THE UPDATE IS LOAD-BEARING and matches ``rollout_bicycle`` exactly:
    position and heading advance on the speed at the START of the step, and ``v`` is
    updated LAST. Integrating with the post-update speed would make every trajectory
    systematically longer — an over-progress bias, which is precisely the defect this
    module exists to attack. The two rollouts must stay consistent, or a bicycle-vs-
    unicycle ablation would be measuring the integrator rather than the action space.
    ⇒ A CONSEQUENCE, and it is the one that bit the first draft of this file: step
    ``k``'s displacement reveals the speed BEFORE ``accel[k]`` was applied, so the
    inverse map is SHIFTED BY ONE. See :func:`unicycle_controls_from_path`.

    ``accel_limit`` / ``curvature_limit`` apply :func:`_squash` when given. Pass
    ``None`` (the default) to integrate the controls as supplied, which is what a head
    that already bounds its own output wants — squashing twice is not a no-op.
    """
    if controls.shape[-1] != 2:
        raise ValueError(f"controls must be [B, K, 2] = (accel, curvature), "
                         f"got trailing dim {controls.shape[-1]}")
    accel = controls[..., 0]
    curvature = controls[..., 1]
    if accel_limit is not None:
        accel = _squash(accel, accel_limit)
    if curvature_limit is not None:
        curvature = _squash(curvature, curvature_limit)

    x, y, yaw, v = state0.unbind(-1)
    out = []
    for k in range(controls.shape[1]):
        x = x + v * torch.cos(yaw) * dt
        y = y + v * torch.sin(yaw) * dt
        yaw = yaw + v * curvature[:, k] * dt          # yaw_rate = v * kappa
        v = (v + accel[:, k] * dt).clamp_min(0.0)
        out.append(torch.stack([x, y, yaw, v], dim=-1))
    return torch.stack(out, dim=1)


def unicycle_controls_from_path(path: Tensor, dt: float = 0.1) -> Tensor:
    """Inverse map: an ego-frame path [B, K, 2] -> the (accel, curvature) that
    produces it under :func:`rollout_unicycle`. Returns [B, K, 2].

    ⭐ WHAT THIS IS FOR. It is the only honest way to answer *"would a control head
    have helped?"* without training one: take an existing arm's waypoints, recover the
    controls it implicitly commanded, and score THOSE. A difference there is evidence
    about the action space; a difference in ADE after a retrain is evidence about
    everything at once.

    ⛔ IT USES THE SAME CONVENTION AS ``four_families._seq_geometry`` — ``speed[k] =
    |p[k] - p[k-1]| / dt`` with the origin prepended, and derivatives taken FORWARD
    across it. That is deliberate and non-negotiable: the whole point is to compare
    recovered controls against the ``accel_mae_mps2`` the binding LONGITUDINAL family
    already publishes, and two different finite-difference conventions would make that
    comparison meaningless.

    ⛔ THE LAST STEP'S CONTROL IS UNOBSERVABLE and is replicated from the second-to-
    last. This is exact, not a fudge: ``controls[K-1]`` only updates ``v`` and ``yaw``
    AFTER the final displacement, so no waypoint depends on it. Replicating it keeps
    the shape contract without inventing information.

    ⚠️ Controls are recovered from finite differences of a discrete path, so they are
    the CHORD-average over each step, not the instantaneous control. On a tight curve
    the chord under-reads speed (the same ~1.1 % effect ``four_families`` documents).
    Do not quote a recovered control as the model's output; quote it as what the path
    implies.

    ⚠️ ENTRY SPEED IS NOT AN INPUT HERE. Whether the arm's first step is consistent
    with the ego's actual ``v0`` is a SEPARATE question with a separate answer —
    :func:`entry_speed_mismatch`. Folding it into ``accel[0]`` would mix "launched from
    the wrong speed" with "accelerated too hard", and the launch transient is exactly
    the thing we are trying to see.
    """
    if path.shape[-1] != 2:
        raise ValueError(f"path must be [B, K, 2], got trailing dim {path.shape[-1]}")
    if path.shape[1] < 2:
        raise ValueError("need at least 2 waypoints to recover a control")
    zero = torch.zeros_like(path[:, :1])
    p = torch.cat([zero, path], dim=1)                     # [B, K+1, 2]
    d = p[:, 1:] - p[:, :-1]                               # [B, K, 2]
    ds = torch.linalg.norm(d, dim=-1)                      # [B, K]
    speed = ds / dt

    accel = (speed[:, 1:] - speed[:, :-1]) / dt            # [B, K-1]
    accel = torch.cat([accel, accel[:, -1:]], dim=1)       # last is unobservable

    heading = torch.atan2(d[..., 1], d[..., 0])
    dh = heading[:, 1:] - heading[:, :-1]
    dh = (dh + torch.pi) % (2 * torch.pi) - torch.pi
    # curvature = dheading / ds. ⛔ The arc the heading turned through between step k
    # and k+1 is travelled at speed[k] — NOT the mean of the two steps. The mean is
    # right for a symmetric central difference (which is what `_seq_geometry` reports
    # as a metric); here we need the control that the INTEGRATOR consumed, and the
    # integrator turns by `v_{k-1} * kappa[k] * dt` = `ds[k] * kappa[k]`.
    # ⛔ WHERE THE EGO IS NOT MOVING, CURVATURE IS UNDETERMINED — NOT LARGE.
    # `yaw_rate = v * kappa`, so at v ~ 0 EVERY kappa produces the same (zero) heading
    # change and the path carries no information about it. Dividing by ds anyway
    # returns an enormous number that is not wrong-by-a-little, it is meaningless.
    # MEASURED 2026-08-06, and this is why the guard exists rather than a comment: run
    # over the 39 paired OOD-val clips WITHOUT it, the implied-curvature MAE came back
    # as 1.6e6 and 7.6e3 1/m for the two arms — a "result" that would have been
    # reported. Returning 0 is the physically right answer AND round-trips exactly,
    # because a stopped unicycle does not turn under any control.
    moving = ds[:, :-1] > (MIN_DS_MPS * dt)
    curv = torch.where(moving, dh / ds[:, :-1].clamp_min(_DS_EPS),
                       torch.zeros_like(dh))                # [B, K-1]
    curv = torch.cat([curv, curv[:, -1:]], dim=1)
    return torch.stack([accel, curv], dim=-1)


def entry_speed_mismatch(path: Tensor, v0: Tensor, dt: float = 0.1) -> Tensor:
    """[B] — the acceleration the arm's FIRST step implies relative to the ego's
    actual entry speed: ``(|p[0]| / dt - v0) / dt``.

    ⭐ WHY IT IS ITS OWN NUMBER. Under :func:`rollout_unicycle` the first displacement
    is forced to use ``v0``, so a path whose first step disagrees with ``v0`` is not
    *reachable* by any control — it is a LAUNCH TRANSIENT, a discontinuity at the
    window origin. MEASURED context: our flagship ends 2 s windows +0.8176 m ahead of
    the human with an accel MAE of 1.7644 m/s² (39 clips, 2026-08-06); separating "it
    jumps off the line" from "it accelerates too hard" is the first thing a control
    head has to be told apart on, and a pooled accel error cannot do it.
    """
    if path.shape[-1] != 2:
        raise ValueError(f"path must be [B, K, 2], got trailing dim {path.shape[-1]}")
    return (torch.linalg.norm(path[:, 0], dim=-1) / dt - v0) / dt
