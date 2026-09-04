"""Differentiable kinematic bicycle layer (H14 Track 1).

The trajectory head predicts *controls*; this layer integrates them through
explicit vehicle kinematics, so every decoded trajectory is physically
realizable by construction. Also provides the Kamm-circle friction penalty
(|a_lat^2 + a_lon^2| <= (mu g)^2) used as a standing metric and loss barrier.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# ⭐ THE COMMAND ↔ GEOMETRY BRIDGE (PI ruling 2026-09-03)
#
# The programme has exactly TWO unit domains for action channel 1 and every
# module belongs to exactly one of them:
#
#   COMMAND   ``steer``  road-wheel angle [rad] — what a driver commands, what
#             `physicalai.signals_at` stores in v2ep ``actions[:, 0]``, and what
#             every trained refav1 / v6 / v7 checkpoint has ever consumed.
#   GEOMETRY  ``kappa``  path curvature [1/m] — what `rollout_unicycle`
#             integrates (``yaw_rate = v * kappa``), what ``GOAL_KAPPA_*`` and
#             ``PlanConfig.kappa_max`` are expressed in, and the only unit in
#             which a metre-valued trajectory can be produced.
#
# Crossing between them REQUIRES one of the two exact inverses below. A crossing
# that applies neither is the defect measured on 2026-09-03: reading the stored
# COMMAND channel as GEOMETRY over-rotates by x2.870 and misses the human's
# lateral position by 0.716 m on curved windows — worse than a straight line.
#
# ⛔ ``STEER_WHEELBASE_M`` IS AN ENCODING CONSTANT, NOT A VEHICLE PROPERTY.
# `physicalai.signals_at` wrote ``steer = arctan(wheelbase * curvature)`` with
# the value the BUILD passed, and every shipped cache passed the legacy 2.9
# (`physicalai.DEFAULT_WHEELBASE_MODE == "const2p9"`; `scripts/v2_compressed.py`
# does not plumb a wheelbase at all). The inverse must use THAT number.
# MEASURED 2026-09-03 by algebraic inversion against the producer's own input
# (the raw egomotion ``curvature`` column) on 20/20 local eval clips:
#   L_enc = 2.9000000 (spread 3e-8, pointwise IQR 1.2e-7), and the forward map
#   at exactly 2.9 reproduces the stored channel to 1.5e-8 rad.
# The clips' TRUE wheelbases — from the release's own
# ``calibration/vehicle_dimensions`` — are {2.73, 3.135, 3.165, 3.216}, mean
# 3.085, and NOT ONE of them is 2.9. Using a real wheelbase here would inject an
# error the encoding never had.
# (`TanitAD Research Lab/Data Engineering/Research/2026-09-03-steer-curvature-interface/`)
# --------------------------------------------------------------------------- #
STEER_WHEELBASE_M = 2.9
#: The two legal spellings of action channel 1. Every API that can receive
#: either states which one it got; there is no "guess from the magnitude".
ACTION_UNITS = ("kappa", "steer")


def kappa_of_steer(steer: Tensor, wheelbase: float = STEER_WHEELBASE_M) -> Tensor:
    """COMMAND -> GEOMETRY: ``kappa = tan(steer) / L_enc`` [1/m]."""
    return torch.tan(steer) / float(wheelbase)


def steer_of_kappa(kappa: Tensor, wheelbase: float = STEER_WHEELBASE_M) -> Tensor:
    """GEOMETRY -> COMMAND: ``steer = arctan(L_enc * kappa)`` [rad]."""
    return torch.atan(float(wheelbase) * kappa)


def _check_units(units: str) -> str:
    if units not in ACTION_UNITS:
        raise ValueError(f"action_units must be one of {ACTION_UNITS}, "
                         f"got {units!r}")
    return units


def as_curvature(controls: Tensor, units: str,
                 wheelbase: float = STEER_WHEELBASE_M) -> Tensor:
    """``[..., >=2]`` controls whose channel 1 is in ``units`` -> channel 1 in
    GEOMETRY (kappa). ``units="kappa"`` returns the input object UNCHANGED — not
    a copy, not a re-scale — so a legacy caller is byte-identical."""
    if _check_units(units) == "kappa":
        return controls
    return torch.cat([controls[..., :1],
                      kappa_of_steer(controls[..., 1:2], wheelbase),
                      controls[..., 2:]], dim=-1)


def as_command(controls: Tensor, units: str,
               wheelbase: float = STEER_WHEELBASE_M) -> Tensor:
    """``[..., >=2]`` controls whose channel 1 is in GEOMETRY (kappa) -> channel
    1 in ``units``. ``units="kappa"`` returns the input UNCHANGED (the legacy,
    unconverted path); ``units="steer"`` applies ``arctan(L_enc * kappa)``."""
    if _check_units(units) == "kappa":
        return controls
    return torch.cat([controls[..., :1],
                      steer_of_kappa(controls[..., 1:2], wheelbase),
                      controls[..., 2:]], dim=-1)


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
#: Waypoint spacing of the deployed 2 s / 20-waypoint plan.
DT_RETIME = 0.1

A2S_ACCEL_LIMIT = 9.8          # m/s^2
A2S_CURVATURE_LIMIT = 0.33     # 1/m

#: Below this step SPEED the path tangent — and therefore curvature — carries no
#: information. Same constant and same meaning as ``four_families.MIN_DS_MPS``, kept
#: here rather than imported because ``tanitad`` must not depend on ``taniteval``.
MIN_DS_MPS = 0.5
_DS_EPS = 1e-8




#: Fraction of ``limit`` below which :func:`_squash` is EXACTLY the identity.
SQUASH_KNEE = 0.9


def _squash(x: Tensor, limit: float, knee: float = SQUASH_KNEE) -> Tensor:
    """Bound ``x`` to (-limit, limit), identity well inside it, non-zero gradient outside.

    ⛔ FOUR CANDIDATES, THREE OF THEM TRAPS.
    ``clamp`` has exactly zero gradient outside the range, so a head that initialises
    outside it can never learn back in — a silent dead head.
    ``tanh`` looks like the fix and is not: MEASURED 2026-08-06, ``1 - tanh(51)**2``
    evaluates to **exactly 0.0** in float32, so a control 500 m/s² against a 9.8 limit
    lands in a region where the gradient has UNDERFLOWED to zero. tanh does not remove
    the cliff, it moves it out to where nobody tests.
    ⚠️ **RE-MEASURED 2026-08-15 — and the ``tanh(51)`` example UNDERSTATES the cliff by
    5×.** The gradient is exactly ``0.0`` from **``raw ≥ 10``**, because ``tanh`` rounds
    to exactly ``1.0f`` there and ``1 - 1*1`` is exactly zero; ``raw = 9`` still carries
    ``1.19e-07``. So this is a genuine cliff at an ORDINARY pre-activation, not a far-field
    curiosity — v6's S-W run logged a ``gnorm 354 076`` spike, which is exactly the regime
    that pushes a pre-activation past 10 and leaves the head unable to learn back.
    Pinned by ``tests/test_emission_squash.py::test_tanh_gradient_is_EXACTLY_zero_from_raw_10_not_51``.
    Plain softsign ``x / (1 + |x|/limit)`` keeps the gradient — but it is NOT the
    identity inside the range: MEASURED 2026-08-06, a curvature of 0.04 against the
    0.33 limit came back as 0.0357, an 11 % shrink on a control that was never near
    the bound. Composed through a decode that must reproduce its own anchor, that cost
    **0.594 m** — the anchor vocabulary became unreachable and the head would have had
    to learn to undo the squash before it could learn anything.
    ⇒ Identity below ``knee * limit``, then a C¹ rational tail that saturates at
    ``limit`` with a 1/x² gradient decay. Exact where it matters, safe where it does not.
    """
    a = knee * limit
    span = limit - a
    excess = (x.abs() - a).clamp_min(0.0)
    tail = a + span * excess / (excess + span)
    return torch.where(x.abs() <= a, x, torch.sign(x) * tail)


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
    # ⛔ epsilon INSIDE the sqrt: norm's gradient at exactly 0 is NaN, and the accel
    # barrier backpropagates through this on paths that contain stopped steps. Same
    # F-5/6/7 class as `_headings` above; masking after the op does not help.
    ds = (d.pow(2).sum(-1) + 1e-12).sqrt()                 # [B, K]
    speed = ds / dt

    accel = (speed[:, 1:] - speed[:, :-1]) / dt            # [B, K-1]
    accel = torch.cat([accel, accel[:, -1:]], dim=1)       # last is unobservable

    _mv = ds > (MIN_DS_MPS * dt)
    _d_safe = torch.where(_mv.unsqueeze(-1), d,
                          torch.stack([torch.ones_like(ds), torch.zeros_like(ds)], -1))
    heading = torch.atan2(_d_safe[..., 1], _d_safe[..., 0])
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


# --------------------------------------------------------------------------- #
# RETIMING — a RETRAIN-FREE fix for a frozen free-waypoint head                 #
#                                                                              #
# ⭐ WHY THIS EXISTS. Every measurement so far describes flagship v1 without    #
# changing it. MEASURED 2026-08-06 on 6,834 windows: intra-plan jerk RMS        #
# 52.21 m/s^3 against a human floor of 1.71 (30.6x); implied accel RMS 4.17     #
# against 0.80 (5.18x); launch transient 1.54 against a 0.42 floor; and the     #
# commanded acceleration revised by 1.10 m/s^2 EVERY 0.1 s frame while the      #
# human's entire accel RMS is 0.80. All four are LONGITUDINAL. The arm's        #
# LATERAL channel is fine — it beats a 34.3 B six-camera model on curvature.    #
#                                                                              #
# ⇒ So do not touch the geometry. Keep the curve the model drew and fix only    #
# the SCHEDULE ALONG IT: re-time the path under a true initial speed and        #
# bounded acceleration and jerk. "Where it wants to go" is preserved exactly;   #
# "how fast it gets there" is made feasible.                                    #
#                                                                              #
# ⛔ WHY RE-TIMING AND NOT RE-INTEGRATING WITH THE RECOVERED CURVATURE. Under   #
# the unicycle, yaw_rate = v * kappa. Change the speed profile and re-integrate #
# the SAME kappa and the heading history changes with it — the path bends       #
# differently and the lateral channel, which is our one healthy channel, is     #
# silently corrupted. Re-sampling the ORIGINAL curve at new arc lengths cannot  #
# do that: the geometric curve is bit-identical, only the sample times move.    #
#                                                                              #
# ⚠️ THIS IS NOT A SUBSTITUTE FOR THE UNICYCLE HEAD. It is a projection applied #
# after the fact: the model still THINKS in free waypoints and still needs a    #
# retrain to plan feasibly. This buys the deployable arm a feasible output      #
# today, and gives the retrain a measured baseline to beat.                     #
# --------------------------------------------------------------------------- #

#: Human 2 s jerk RMS on PhysicalAI OOD-val, MEASURED 2026-08-06 (n=6,834 windows;
#: Alpamayo 2 Super scores 1.79 through the identical instrument). The default
#: barrier is set well ABOVE it — the aim is to remove the 30x thrash, not to
#: flatten legitimate emergency braking.
HUMAN_JERK_RMS_MPS3 = 1.71


def retime_path(path: Tensor, v0: Tensor, dt: float = DT_RETIME,
                accel_limit: float = 4.0,
                jerk_limit: float = 12.0) -> Tensor:
    """Re-time an ego-frame path [B,K,2] so its SPEED PROFILE is feasible.

    Returns [B,K,2]: the SAME geometric curve, re-sampled at arc lengths produced
    by integrating a bounded-acceleration, bounded-jerk schedule from the ego's
    TRUE entry speed ``v0`` [B].

    Three defects are removed by construction:
      * **launch transient** — the schedule starts at ``v0``, so the first step
        cannot disagree with the ego's real speed;
      * **accel magnitude** — every commanded accel is clamped to ``accel_limit``;
      * **jerk** — successive accels may differ by at most ``jerk_limit * dt``.

    ⛔ A HARD CLAMP IS CORRECT HERE and would be wrong in a trained head. There is
    no gradient to preserve: this runs at inference on a frozen model, so the
    dead-head argument that forced softsign in :func:`rollout_unicycle` does not
    apply. Clamping is exact, cheap and has no tuning surface.

    ⚠️ The re-timed path may run PAST the end of the original curve (if the arm was
    under-driving) — the tail is then extrapolated along the final tangent, which is
    a straight-line continuation and is flagged in the docstring rather than hidden.
    Running SHORT is not a problem: the samples simply stop earlier along the curve.
    """
    if path.dim() != 3 or path.shape[-1] != 2:
        raise ValueError(f"path must be [B, K, 2], got {tuple(path.shape)}")
    B, K, _ = path.shape
    v0 = torch.as_tensor(v0, dtype=path.dtype, device=path.device).reshape(B)

    # --- arc length of the ORIGINAL curve, origin prepended ---------------------
    p = torch.cat([torch.zeros_like(path[:, :1]), path], dim=1)      # [B,K+1,2]
    seg = torch.linalg.norm(p[:, 1:] - p[:, :-1], dim=-1)            # [B,K]
    s = torch.cumsum(seg, dim=1)                                     # [B,K]
    s = torch.cat([torch.zeros_like(s[:, :1]), s], dim=1)            # [B,K+1]

    # --- the arm's OWN implied speed profile, which we retime TOWARDS -----------
    want = seg / dt                                                  # [B,K]

    # --- feasible schedule: bounded accel, bounded jerk, exact v0 ---------------
    v = v0.clone()
    a_prev = torch.zeros_like(v0)
    s_new, dist = [], torch.zeros_like(v0)
    for k in range(K):
        a = (want[:, k] - v) / dt                       # accel that would hit it
        a = a.clamp(-accel_limit, accel_limit)
        a = torch.max(torch.min(a, a_prev + jerk_limit * dt),
                      a_prev - jerk_limit * dt)          # jerk barrier
        dist = dist + v * dt                             # advance on the PRE-update
        s_new.append(dist)                               # speed, as rollout_* does
        v = (v + a * dt).clamp_min(0.0)
        a_prev = a
    s_new = torch.stack(s_new, dim=1)                                # [B,K]

    # --- resample the ORIGINAL curve at the new arc lengths ---------------------
    out = torch.empty_like(path)
    for b in range(B):
        out[b, :, 0] = _interp1d(s_new[b], s[b], p[b, :, 0])
        out[b, :, 1] = _interp1d(s_new[b], s[b], p[b, :, 1])
    # ⛔ BEYOND THE END OF THE CURVE, CONTINUE THE ARC — NOT THE TANGENT.
    # MEASURED 2026-08-06: 17.9 % of windows (all low-speed, mean v0 4.77 m/s) have a
    # schedule that OUTRUNS the curve, because the ego's true v0 exceeds what the arm's
    # own first waypoint implied. Under-running windows are off-curve by 4.44e-16 m —
    # machine zero — but a straight-tangent extrapolation put over-running ones up to
    # 0.2146 m off, and that showed up at full scale as a REAL curvature regression
    # (MAE 0.006103 -> 0.006922 over 6,834 windows). Continuing with the curve's final
    # curvature keeps the lateral channel intact by construction.
    _extend_by_arc(out, p, s, s_new)
    return out


def _extend_by_arc(out: Tensor, p: Tensor, s: Tensor, s_new: Tensor) -> None:
    """In-place: rewrite samples that lie PAST the curve as a constant-curvature
    continuation of its final arc. A straight continuation is the ``kappa -> 0`` limit
    of this and is handled by the same branch, so a straight path is untouched."""
    B = out.shape[0]
    seg = s[:, 1:] - s[:, :-1]                                   # [B,K]
    d = p[:, 1:] - p[:, :-1]
    head = torch.atan2(d[..., 1], d[..., 0])                     # [B,K]
    dh = (head[:, 1:] - head[:, :-1] + math.pi) % (2 * math.pi) - math.pi
    for b in range(B):
        beyond = s_new[b] > s[b, -1]
        if not bool(beyond.any()):
            continue
        # final heading and curvature of the curve, from its last usable segment
        valid = seg[b] > _DS_EPS
        if not bool(valid.any()):
            continue
        j = int(torch.nonzero(valid).max())
        h_end = head[b, j]
        k_end = (dh[b, j - 1] / seg[b, j].clamp_min(_DS_EPS)) if j >= 1 else \
            torch.zeros((), dtype=out.dtype, device=out.device)
        k_end = k_end.clamp(-A2S_CURVATURE_LIMIT, A2S_CURVATURE_LIMIT)
        u = (s_new[b][beyond] - s[b, -1]).clamp_min(0.0)         # extra arc length
        if bool((k_end.abs() < 1e-6).item()):
            dx, dy = u * torch.cos(h_end), u * torch.sin(h_end)
        else:
            h2 = h_end + k_end * u
            dx = (torch.sin(h2) - torch.sin(h_end)) / k_end
            dy = -(torch.cos(h2) - torch.cos(h_end)) / k_end
        out[b, beyond, 0] = p[b, -1, 0] + dx
        out[b, beyond, 1] = p[b, -1, 1] + dy


def _interp1d(q: Tensor, xs: Tensor, ys: Tensor) -> Tensor:
    """Linear interpolation of ``ys`` at query points ``q`` over knots ``xs``.

    ⛔ Beyond the last knot this EXTRAPOLATES along the final segment's direction
    rather than clamping. Clamping would pile every over-running sample onto the
    curve's endpoint and manufacture a hard stop that the arm never planned — a
    speed fix that invents a braking event is worse than no fix.
    """
    idx = torch.searchsorted(xs.contiguous(), q.contiguous()).clamp(1, xs.numel() - 1)
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    w = (q - x0) / (x1 - x0).clamp_min(1e-8)
    return y0 + w * (y1 - y0)


# --------------------------------------------------------------------------- #
# TRAINING-SIDE: kinematic loss terms, and the differentiable unicycle decode   #
#                                                                              #
# ⛔ WHY THE HEAD LEARNS BAD KINEMATICS. `flagship_v15.v15_losses` supervises   #
# trajectories with `(recon - traj_tgt).abs().mean()` — PURE POSITION L1.       #
# Nothing in it constrains heading, curvature, acceleration or jerk. Every      #
# defect measured on v1arch follows directly: accel RMS 4.21x human, jerk       #
# 30.6x, and a heading channel that only looks healthy because ADE never        #
# scored it. A term that is not in the loss is not being learned.               #
#                                                                              #
# ⚠️ POSITION L1 DOES NOT IMPLY HEADING. Two paths can agree at every waypoint  #
# to within centimetres and still have different tangents at every step,        #
# because L1 on positions is invariant to how the samples are distributed       #
# along the curve. That is not a corner case — it is what re-timing v1arch      #
# demonstrated: MEASURED 2026-08-06, cross-track improved 20 % while net-yaw    #
# error got 62 % WORSE on the same paths.                                       #
# --------------------------------------------------------------------------- #


def _headings(path: Tensor, dt: float = DT_RETIME):
    """[B,S,2] -> (heading [B,S], step length [B,S], moving mask [B,S]).

    ⛔ BACKWARD-SAFE AT ZERO DISPLACEMENT, and this is not optional. ``atan2`` and
    ``norm`` both have NaN GRADIENTS at exactly (0, 0) even though their forward
    values are fine, and masking the OUTPUT does not help: ``0 * NaN = NaN`` in the
    backward pass. MEASURED 2026-08-06: the first unicycle-readout training run
    NaN'd from its first logged step because the train corpus contains STOPPED
    episodes (v = 0.00) — while every eval and baseline pass was clean, because
    none of them call backward(). This is the F-5/6/7 sqrt-relu trap
    (`kamm_circle_violation`, `test_kinematic_nan.py`) in its third costume.
    ⇒ norms get an epsilon INSIDE the sqrt; atan2's INPUT is substituted with a
    safe (1, 0) on non-moving steps BEFORE the op, not masked after it."""
    p = torch.cat([torch.zeros_like(path[:, :1]), path], dim=1)
    d = p[:, 1:] - p[:, :-1]
    ds = (d.pow(2).sum(-1) + 1e-12).sqrt()
    moving = ds > (MIN_DS_MPS * dt)
    d_safe = torch.where(moving.unsqueeze(-1), d,
                         torch.stack([torch.ones_like(ds), torch.zeros_like(ds)], -1))
    return torch.atan2(d_safe[..., 1], d_safe[..., 0]), ds, moving


def kinematic_losses(pred: Tensor, tgt: Tensor, dt: float = DT_RETIME,
                     accel_limit: float = 2.785,
                     jerk_limit: float = 6.369) -> dict:
    """Heading / net-yaw / accel / jerk terms for a [B,S,2] trajectory pair.

    ⭐ ``heading`` and ``net_yaw`` are the two the position loss cannot see, and
    ``net_yaw`` is the SAMPLING-INDEPENDENT one — it is the quantity that
    regressed 62 % under re-timing while cross-track improved, so it is the
    honest target for "fix the heading".

    ⛔ ``accel`` and ``jerk`` are BARRIERS, not shrinkage: only the excess above
    the limit is penalised. A plain ``lambda * jerk**2`` would also punish
    legitimate emergency braking and hard avoidance, i.e. it would train the arm
    to be smooth when it should be decisive. Defaults are the human's own p99 on
    PhysicalAI OOD-val (MEASURED 2026-08-06, 6,834 windows).

    ⚠️ Steps below ``MIN_DS_MPS`` are masked out of the heading terms. A stopped
    ego has no path tangent and its heading is free to flip; including those
    steps would train the head against noise. Same gate as
    ``four_families.lateral`` and ``unicycle_controls_from_path``.
    """
    hp, _, mp = _headings(pred, dt)
    hg, _, mg = _headings(tgt, dt)
    both = mp & mg
    dh = (hp - hg + math.pi) % (2 * math.pi) - math.pi
    n = both.sum().clamp_min(1)
    head = (dh.abs() * both).sum() / n

    # net yaw over the window: sum of WRAPPED per-step turns, masked
    wrap = lambda x: (x + math.pi) % (2 * math.pi) - math.pi      # noqa: E731
    vp = mp[:, 1:] & mp[:, :-1]
    vg = mg[:, 1:] & mg[:, :-1]
    ny_p = (wrap(hp[:, 1:] - hp[:, :-1]) * vp).sum(1)
    ny_g = (wrap(hg[:, 1:] - hg[:, :-1]) * vg).sum(1)
    net_yaw = (ny_p - ny_g).abs().mean()

    ca = unicycle_controls_from_path(pred, dt)[..., 0]
    accel = torch.relu(ca.abs() - accel_limit).mean()
    jerk = torch.relu(((ca[:, 1:] - ca[:, :-1]) / dt).abs() - jerk_limit).mean()
    return {"heading": head, "net_yaw": net_yaw, "accel": accel, "jerk": jerk}


def unicycle_decode(base: Tensor, delta: Tensor, v0: Tensor,
                    dt: float = DT_RETIME,
                    accel_limit: float = A2S_ACCEL_LIMIT,
                    curvature_limit: float = A2S_CURVATURE_LIMIT) -> Tensor:
    """⭐ THE RETRAIN PATH: compose a correction in CONTROL space, not position.

    ``base`` [B,S,2] an anchor trajectory, ``delta`` [B,S,2] the head's output
    read as ``(d_accel, d_curvature)``, ``v0`` [B] the ego's true entry speed.
    Returns [B,S,2], differentiable end-to-end through :func:`rollout_unicycle`.

    ⛔ WHY THIS IS THE RIGHT PLACE TO CHANGE THE HEAD. The anchor VOCABULARY is
    already kinematically feasible — ``fourbrain._synth_anchor_pool`` builds it
    from random unicycle rollouts. It is the free-form per-waypoint OFFSET that
    destroys feasibility: nothing stops it moving waypoint k by 2 m and waypoint
    k+1 by −2 m, which is a 400 m/s³ jerk the position loss never sees. Adding
    the offset in control space instead makes every output feasible BY
    CONSTRUCTION, and it changes ~one line of the decode rather than the
    diffusion machinery, the anchors, the selection head or the losses.

    ⇒ It also fixes the heading channel structurally: heading is
    ``yaw += v * kappa * dt``, so the head now emits the quantity heading is
    made of, instead of positions from which heading is a fragile by-product.

    ⚠️ The bounds are applied by :func:`rollout_unicycle`'s softsign — saturating
    but never zero-gradient, so a head that initialises outside the range can
    still learn back in. Do NOT swap them for a hard clamp here; that is the
    dead-head trap, and it is only safe in the inference-time
    :func:`retime_path`, which has no gradient to preserve.
    """
    if base.shape != delta.shape:
        raise ValueError(f"base {tuple(base.shape)} != delta {tuple(delta.shape)}")
    c = unicycle_controls_from_path(base, dt) + delta
    B = base.shape[0]
    v0 = torch.as_tensor(v0, dtype=base.dtype, device=base.device).reshape(B)
    z = torch.zeros_like(v0)
    state0 = torch.stack([z, z, z, v0], dim=-1)
    return rollout_unicycle(state0, c, dt, accel_limit=accel_limit,
                            curvature_limit=curvature_limit)[..., :2]


# --------------------------------------------------------------------------- #
# ⭐ VARIABLE-STEP UNICYCLE — the L4 kernel for REF-C v4 (2026-09-04)           #
#                                                                             #
# ⛔ WHY THIS EXISTS, AND WHY THE SCALAR-dt FUNCTIONS ABOVE CANNOT BE USED FOR #
# REF-C's EMISSION. MEASURED 2026-09-04: refc_v3's fan is emitted on the       #
# ``horizons`` grid ``[5, 10, 15, 20, 30, 40, 50, 60]`` frames at the 10 Hz    #
# tick, i.e. slot times 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0 s. The per-slot #
# spacing is therefore                                                         #
#       [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]  s                            #
# which is NOT UNIFORM — it doubles at the 2 s seam.                           #
#                                                                             #
# ``rollout_unicycle`` / ``unicycle_controls_from_path`` / ``unicycle_decode`` #
# each take a SCALAR ``dt``. Wiring them into an 8-slot REF-C emission would   #
# integrate eight equal steps across a grid whose second half is twice as long #
# — a 2x time error on every waypoint from 3 s out, which is exactly the range #
# where the programme's measured longitudinal deficit lives.                   #
#                                                                             #
# ⛔ AND IT WOULD FAIL SILENTLY. The output is still [B, S, 2], the L1 still   #
# falls, ADE still computes. Nothing raises. This is the ``df`` / Thor         #
# ``free`` / cgroup ``usage_in_bytes`` / ``step_s`` family in an integrator's  #
# costume: a correct formula quoted outside its scope, read as an answer. It   #
# is also the ``HORIZON = round(6.0 * 10.0 / STRIDE)`` derived-constant trap   #
# from ``v2_compressed.py`` with the constant moved into the time axis.        #
#                                                                             #
# ⇒ Two admissible ways to emit a unicycle path on this grid, and BOTH are     #
#   provided here so an arm has to say which it chose:                         #
#   (a) integrate at the NATIVE uniform tick and SUBSAMPLE to the slots        #
#       (:func:`rollout_unicycle_grid`) — physically the honest one, and the   #
#       convention ``refa_v1``'s ladder already uses ("the abstracted levels   #
#       SUBSAMPLE the operative target grid"); jerk is then differenced on a   #
#       uniform sequence, where it means what its units say.                   #
#   (b) integrate the slots directly under a PER-STEP dt                       #
#       (:func:`rollout_unicycle_varstep`) — cheaper (8 steps, not 60), exact  #
#       for the waypoints, but its differences straddle the 2 s seam, so a     #
#       jerk computed on them is NOT comparable to a 10 Hz jerk.               #
#                                                                             #
# ⚠️ A NOTE ON THE CHANNEL CONVENTION, because a briefing inverted it and the  #
# inversion would have mis-aimed the whole cost budget. In THIS module's       #
# unicycle convention ``controls[..., 0] = accel`` and                          #
# ``controls[..., 1] = curvature`` (see :func:`rollout_unicycle`). MEASURED    #
# from source 2026-09-04, ``refa_v1``'s planner cost uses the SAME convention: #
#   ``jerk = (controls[:, 1:, 0] - controls[:, :-1, 0]) / pc.dt``  -> LONGITUDINAL
#   ``c += 0.05 * controls[..., 1].pow(2).mean(-1)   # curvature``  -> lateral MAGNITUDE
#   ``v_end = v0 + controls[..., 0].sum(-1) * pc.dt``  -> channel 0 integrates to SPEED
# So refav1 ALREADY penalises longitudinal jerk; what it has never had is a    #
# curvature-RATE (lateral jerk) term. Do not repeat the claim that its         #
# channel 0 is steer.                                                          #
# --------------------------------------------------------------------------- #

def slot_dts(horizons, tick: float = 0.1) -> list[float]:
    """Per-slot dt [s] for a ``horizons`` grid given in FRAMES at ``tick``.

    ``slot_dts([5, 10, 20])`` -> ``[0.5, 0.5, 1.0]``. The first entry is the gap
    from t=0 (the ego's own position) to the first slot, matching the
    ``torch.cat([zeros, path])`` origin-prepend convention every inverse map in
    this module uses.
    """
    h = [int(x) for x in horizons]
    if not h:
        raise ValueError("horizons must be non-empty")
    if any(b <= a for a, b in zip(h, h[1:])):
        raise ValueError(f"horizons must be strictly increasing, got {h}")
    if h[0] <= 0:
        raise ValueError(f"horizons must be positive, got {h}")
    return [float(b - a) * float(tick) for a, b in zip([0] + h[:-1], h)]


def uniform_slot_dt(horizons, tick: float = 0.1) -> float:
    """The single dt of a UNIFORM ``horizons`` grid, or raise.

    ⭐ Call this before passing a scalar ``dt`` to any function above. It turns
    the silent 2x time error described in this section's header into a loud
    refusal at wiring time, which is the only point at which it is cheap.
    """
    d = slot_dts(horizons, tick)
    if max(d) - min(d) > 1e-9:
        raise ValueError(
            f"horizons {list(horizons)} give NON-UNIFORM slot spacing {d} s — a "
            f"scalar dt is inexpressible here. Use rollout_unicycle_grid (integrate "
            f"at the native tick, subsample) or the *_varstep functions, and say "
            f"in the arm's config which one was chosen.")
    return d[0]


def _as_dts(dts, K: int, ref: Tensor) -> Tensor:
    """Normalise ``dts`` (float | sequence | Tensor) to a [1, K] or [B, K] tensor."""
    t = torch.as_tensor(dts, dtype=ref.dtype, device=ref.device)
    if t.ndim == 0:
        t = t.reshape(1, 1).expand(1, K)
    elif t.ndim == 1:
        if t.shape[0] != K:
            raise ValueError(f"dts has length {t.shape[0]}, expected {K}")
        t = t.reshape(1, K)
    elif t.ndim != 2 or t.shape[1] != K:
        raise ValueError(f"dts must be scalar, [K] or [B, K] with K={K}, "
                         f"got {tuple(t.shape)}")
    if bool((t <= 0).any()):
        raise ValueError("every dt must be > 0")
    return t


def rollout_unicycle_varstep(state0: Tensor, controls: Tensor, dts,
                             accel_limit: float | None = None,
                             curvature_limit: float | None = None) -> Tensor:
    """:func:`rollout_unicycle` with a PER-STEP dt.

    ``state0`` [B, 4] = (x, y, yaw, v); ``controls`` [B, K, 2] = (accel, curvature);
    ``dts`` scalar | [K] | [B, K]. Returns states [B, K, 4]. Differentiable.

    ⛔ The update ORDER is copied verbatim from :func:`rollout_unicycle` and is
    load-bearing: position and heading advance on the speed at the START of the
    step and ``v`` is updated LAST. The two must stay consistent or a
    fixed-vs-variable-step ablation would be measuring the integrator.
    """
    if controls.shape[-1] != 2:
        raise ValueError(f"controls must be [B, K, 2] = (accel, curvature), "
                         f"got trailing dim {controls.shape[-1]}")
    K = controls.shape[1]
    dt = _as_dts(dts, K, controls)
    accel = controls[..., 0]
    curvature = controls[..., 1]
    if accel_limit is not None:
        accel = _squash(accel, accel_limit)
    if curvature_limit is not None:
        curvature = _squash(curvature, curvature_limit)

    x, y, yaw, v = state0.unbind(-1)
    out = []
    for k in range(K):
        h = dt[:, k]
        x = x + v * torch.cos(yaw) * h
        y = y + v * torch.sin(yaw) * h
        yaw = yaw + v * curvature[:, k] * h
        v = (v + accel[:, k] * h).clamp_min(0.0)
        out.append(torch.stack([x, y, yaw, v], dim=-1))
    return torch.stack(out, dim=1)


def unicycle_controls_from_path_varstep(path: Tensor, dts) -> Tensor:
    """:func:`unicycle_controls_from_path` with a PER-STEP dt.

    Every convention is inherited verbatim — origin prepended, forward
    differences, the last control replicated because it is unobservable, the
    ``MIN_DS_MPS`` moving-gate, epsilon INSIDE the sqrt, and curvature returned
    as 0 (not a huge number) where the ego is not moving. Read that function's
    docstring before changing anything here; each of those is a measured trap.

    ⚠️ ``accel[k]`` is divided by ``dt[k]`` because the integrator applies it
    over step k (``v_{k+1} = v_k + a_k * dt_k``). ``curvature`` is ``dh / ds``
    and carries no dt at all, so it is unchanged by the variable grid.
    """
    if path.shape[-1] != 2:
        raise ValueError(f"path must be [B, K, 2], got trailing dim {path.shape[-1]}")
    if path.shape[1] < 2:
        raise ValueError("need at least 2 waypoints to recover a control")
    K = path.shape[1]
    dt = _as_dts(dts, K, path)

    zero = torch.zeros_like(path[:, :1])
    p = torch.cat([zero, path], dim=1)
    d = p[:, 1:] - p[:, :-1]
    ds = (d.pow(2).sum(-1) + 1e-12).sqrt()
    speed = ds / dt

    accel = (speed[:, 1:] - speed[:, :-1]) / dt[:, :-1]
    accel = torch.cat([accel, accel[:, -1:]], dim=1)

    _mv = ds > (MIN_DS_MPS * dt)
    _d_safe = torch.where(_mv.unsqueeze(-1), d,
                          torch.stack([torch.ones_like(ds), torch.zeros_like(ds)], -1))
    heading = torch.atan2(_d_safe[..., 1], _d_safe[..., 0])
    dh = heading[:, 1:] - heading[:, :-1]
    dh = (dh + torch.pi) % (2 * torch.pi) - torch.pi
    moving = ds[:, :-1] > (MIN_DS_MPS * dt[:, :-1])
    curv = torch.where(moving, dh / ds[:, :-1].clamp_min(_DS_EPS),
                       torch.zeros_like(dh))
    curv = torch.cat([curv, curv[:, -1:]], dim=1)
    return torch.stack([accel, curv], dim=-1)


def control_smoothness_losses(controls: Tensor, dts,
                              jerk_limit: float,
                              curvature_rate_limit: float) -> dict:
    """BOTH-CHANNEL smoothness barriers for a (accel, curvature) sequence.

    Returns ``{"jerk_lon": ..., "curv_rate": ...}`` — the longitudinal jerk
    ``d(accel)/dt`` [m/s^3] and the lateral curvature rate ``d(kappa)/dt``
    [1/(m s)], each rectified above its limit and averaged.

    ⛔ BARRIERS, NOT SHRINKAGE — the same argument :func:`kinematic_losses`
    makes and for the same reason: a plain ``lambda * jerk**2`` also punishes
    legitimate emergency braking and hard avoidance, i.e. it trains the arm to
    be smooth exactly when it should be decisive.

    ⛔ BOTH LIMITS ARE REQUIRED ARGUMENTS AND NEITHER HAS A DEFAULT. The
    accel/jerk limits used elsewhere in this module (2.785 / 6.369) are the
    human's own p99 MEASURED on PhysicalAI OOD-val over 6,834 windows. An
    invented curvature-rate limit would be a number with no evidence class
    deciding a GPU-day, which the operating standard forbids — measure the p99
    on the arm's OWN corpus and pass it, and record the value and the n in the
    arm's config.

    ⚠️ These differences are only physically meaningful on a UNIFORM control
    sequence. On REF-C's native ``horizons`` grid the spacing doubles at the 2 s
    seam, so a "jerk" differenced across it mixes two time bases. Emit through
    :func:`rollout_unicycle_grid` and compute this on the 10 Hz controls, or
    state explicitly that the value is a seam-straddling proxy.
    """
    if controls.shape[-1] != 2:
        raise ValueError(f"controls must be [B, K, 2], got trailing dim "
                         f"{controls.shape[-1]}")
    if controls.shape[1] < 2:
        raise ValueError("need at least 2 control steps to difference")
    K = controls.shape[1]
    dt = _as_dts(dts, K, controls)[:, :-1]
    d_acc = (controls[:, 1:, 0] - controls[:, :-1, 0]) / dt
    d_curv = (controls[:, 1:, 1] - controls[:, :-1, 1]) / dt
    return {
        "jerk_lon": torch.relu(d_acc.abs() - float(jerk_limit)).mean(),
        "curv_rate": torch.relu(d_curv.abs() - float(curvature_rate_limit)).mean(),
    }


def rollout_unicycle_grid(state0: Tensor, controls: Tensor, horizons,
                          tick: float = 0.1,
                          accel_limit: float | None = None,
                          curvature_limit: float | None = None) -> Tensor:
    """⭐ THE RECOMMENDED EMISSION: integrate at the NATIVE tick, then subsample.

    ``controls`` [B, T, 2] at the uniform ``tick`` (T must reach ``max(horizons)``);
    returns states [B, len(horizons), 4] at the requested frame indices.

    This is the option that keeps jerk meaningful: the head emits a uniform 10 Hz
    control sequence, the physics is integrated on that uniform grid, and only the
    READOUT is sparse and non-uniform. Differencing for
    :func:`control_smoothness_losses` happens on ``controls``, before the subsample.
    """
    h = [int(x) for x in horizons]
    T = controls.shape[1]
    if h[-1] > T:
        raise ValueError(f"controls has {T} steps but horizons reach {h[-1]}")
    st = rollout_unicycle(state0, controls, dt=float(tick),
                          accel_limit=accel_limit, curvature_limit=curvature_limit)
    idx = torch.as_tensor([x - 1 for x in h], device=st.device, dtype=torch.long)
    return st.index_select(1, idx)


def unicycle_decode_varstep(base: Tensor, delta: Tensor, v0: Tensor, dts,
                            accel_limit: float = A2S_ACCEL_LIMIT,
                            curvature_limit: float = A2S_CURVATURE_LIMIT) -> Tensor:
    """:func:`unicycle_decode` on a non-uniform slot grid.

    ``base`` [B, S, 2] anchor waypoints AT THE SLOT TIMES, ``delta`` [B, S, 2] the
    head's ``(d_accel, d_curvature)``, ``v0`` [B] the ego's measured entry speed,
    ``dts`` the per-slot spacing from :func:`slot_dts`. Returns [B, S, 2].

    ⭐ The feasibility argument is :func:`unicycle_decode`'s and is unchanged: it
    is the free-form per-waypoint OFFSET that destroys feasibility, and composing
    the correction in CONTROL space makes every emitted path drivable by
    construction.

    ⚠️ ONE PREMISE DOES CHANGE under a data-driven vocabulary, and in our favour.
    :func:`unicycle_decode` argues the base is feasible because
    ``fourbrain._synth_anchor_pool`` builds it from unicycle rollouts. A vocabulary
    clustered from REAL human trajectories is not built that way — it is feasible
    for the stronger reason that a human actually drove it.
    """
    if base.shape != delta.shape:
        raise ValueError(f"base {tuple(base.shape)} != delta {tuple(delta.shape)}")
    c = unicycle_controls_from_path_varstep(base, dts) + delta
    B = base.shape[0]
    v0 = torch.as_tensor(v0, dtype=base.dtype, device=base.device).reshape(B)
    z = torch.zeros_like(v0)
    state0 = torch.stack([z, z, z, v0], dim=-1)
    return rollout_unicycle_varstep(state0, c, dts, accel_limit=accel_limit,
                                    curvature_limit=curvature_limit)[..., :2]
