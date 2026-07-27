"""Ego/action input for the tactical plan — the geometry, and the graft.

WHY THIS MODULE EXISTS
----------------------
The pseudo-simulation arm panel (2026-07-27) found that a **zero-parameter
constant-velocity baseline ranks first** on the only closed-loop surface this
program owns that is a MEASUREMENT, and named a mechanism: the tactical planner
never sees the ego speed. Reading the code (2026-07-28) the mechanism is real
but is TWO gaps, not one:

1. ``TacticalPolicy.forward(states, ctx, ego=None)`` DOES take an ego vector
   (``tanitad/models/fourbrain.py:328``), but ``self.ego_emb`` is built only when
   ``cfg.v2_ego_to_planners`` is on — and it is ``False`` by default
   (``tanitad/config.py:196``). Every panel arm records
   ``ego_input_on_planners = False``.
2. ⭐ Independently, EVERY eval/planner call site calls
   ``tactical_policy(states, ctx)`` positionally and never passes ``ego=`` —
   ``taniteval/closedloop.py:245,317``, ``planner_p2.py:279,340``,
   ``planning.py:166``, ``corpus_overlay.py:307``, ``blindimag.py:101``,
   ``probe_overlay.py:49``. So an ego-TRAINED checkpoint would still be
   evaluated ego-BLIND, silently and with no error.

This module holds the two things that follow, and **modifies nothing that a
running trainer imports** (pod1 is training on ``train_flagship4b``; not one
byte of ``fourbrain.py`` / ``config.py`` / ``train_flagship4b.py`` is touched).

WHAT IS HERE
------------
``arc_length`` / ``resample_by_arclength`` / ``retime``
    A plan is a **curve** plus a **schedule**: ``gamma(s)`` traced in the plane,
    and how far along it the plan is at each step. A tactical head that never
    saw ``v0`` can still emit a good curve while getting the schedule wrong — and
    an ego-speed input's entire job is the schedule. These functions factor the
    two so the contribution of a speed input can be measured **without training
    anything**, by re-timing an existing plan.

    ⚠️ Not a planner and not a fix — a MEASUREMENT INSTRUMENT. Re-timing a plan
    with the true ``v0`` is an upper bound on what a perfect speed input could
    buy the longitudinal axis; it is not a deployable controller.

``attach_ego_input``
    The architectural piece for v5: graft the shipped ``ego_emb`` seam onto an
    already-built ``TacticalPolicy`` / ``StrategicPolicy``, ZERO-INITIALISED so
    the grafted module is numerically IDENTICAL to the original until the graft
    is trained. This is the same ``nn.Linear(2, d_cond)`` the ``--v2`` lever
    builds, so a grafted model is state-dict-compatible with a ``--v2`` run.

THE EGO VECTOR — the contract, copied from the trainer so it cannot drift
------------------------------------------------------------------------
``ego = [v0 / pose_scale, yr0]``, both from OBSERVED poses only (t and t-1),
never a future (``tanitad/train/flagship_losses.py:202-210``). ⚠️ Note this is
**ego PROPRIOCEPTION, not an action**: the operative predictor's speed channel is
a third *action* channel at ``v0 / SPEED_SCALE`` with ``SPEED_SCALE = 10.0``
(``flagship_losses.py:228``), which is a different scaling and a different port.
Conflating them decodes garbage.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

#: Divisor for the operative predictor's third ACTION channel. Hard contract
#: with ``eval_grounded_rollout_4b_speed.py`` — see registry §1.2.
SPEED_SCALE = 10.0

#: Below this, a segment carries no usable direction.
_EPS = 1e-9


# =========================================================================== #
# the geometry: a plan is a CURVE plus a SCHEDULE                              #
# =========================================================================== #
def _polyline(traj: Tensor) -> Tensor:
    """``[n, H, 2]`` plan -> ``[n, H+1, 2]`` polyline that starts at the ego
    origin. ``traj[:, 0]`` is the position at step 1, so the ego's own position
    at step 0 is the implicit ``(0, 0)`` and must be prepended before any arc
    length is taken."""
    if traj.ndim != 3 or traj.shape[-1] != 2:
        raise ValueError(f"traj must be [n, H, 2], got {tuple(traj.shape)}")
    n = traj.shape[0]
    zero = torch.zeros(n, 1, 2, dtype=traj.dtype, device=traj.device)
    return torch.cat([zero, traj], dim=1)


def arc_length(traj: Tensor) -> Tensor:
    """Cumulative arc length along the plan, ``[n, H+1]``, starting at 0.

    ``out[:, k]`` is the distance travelled along the plan's own path by step
    ``k``. ``out[:, -1]`` is the total path length — the quantity a speed-blind
    tactical head has no way to calibrate."""
    p = _polyline(traj)
    seg = p[:, 1:] - p[:, :-1]                       # [n, H, 2]
    lens = seg.norm(dim=-1)                          # [n, H]
    zero = torch.zeros(lens.shape[0], 1, dtype=lens.dtype, device=lens.device)
    return torch.cat([zero, lens.cumsum(dim=-1)], dim=-1)


def terminal_tangent(traj: Tensor) -> tuple[Tensor, Tensor]:
    """Unit direction of the LAST non-degenerate segment, ``([n, 2], [n] bool)``.

    The bool marks rows that have no usable direction at all (a plan that never
    moves). Those rows cannot be extrapolated and are reported, never guessed:
    a silently-invented tangent is exactly how a degenerate arm sneaks a score."""
    p = _polyline(traj)
    seg = p[:, 1:] - p[:, :-1]                       # [n, H, 2]
    lens = seg.norm(dim=-1)                          # [n, H]
    usable = lens > _EPS
    any_usable = usable.any(dim=-1)
    # index of the LAST usable segment (0 for rows with none; masked out below)
    h = lens.shape[1]
    ix = torch.arange(h, device=traj.device).expand_as(lens)
    last = torch.where(usable, ix, torch.full_like(ix, -1)).amax(dim=-1)
    last = last.clamp_min(0)
    pick = seg[torch.arange(seg.shape[0], device=traj.device), last]   # [n, 2]
    norm = pick.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    tangent = torch.where(any_usable[:, None], pick / norm,
                          torch.zeros_like(pick))
    return tangent, ~any_usable


def resample_by_arclength(traj: Tensor, s_new: Tensor) -> dict[str, Tensor]:
    """Walk the plan's OWN path to arc lengths ``s_new`` — the shape is kept
    exactly, only the schedule changes.

    ``traj`` ``[n, H, 2]`` · ``s_new`` ``[n, H]`` (must be non-negative).
    Returns ``{"traj", "extrapolated", "degenerate", "frac_extrapolated"}``.

    Beyond the end of the path the curve is extended along its **terminal
    tangent** — a straight continuation of where the plan was last heading. Rows
    whose plan never moves have no tangent; they are returned unchanged (all
    zeros) and flagged in ``degenerate``, never silently extended."""
    if s_new.shape != traj.shape[:2]:
        raise ValueError(f"s_new must be [n, H] = {tuple(traj.shape[:2])}, "
                         f"got {tuple(s_new.shape)}")
    if bool((s_new < 0).any()):
        raise ValueError("s_new must be non-negative (arc length)")
    p = _polyline(traj)                              # [n, H+1, 2]
    seg = p[:, 1:] - p[:, :-1]                       # [n, H, 2]
    lens = seg.norm(dim=-1)                          # [n, H]
    s_cum = arc_length(traj)                         # [n, H+1]
    total = s_cum[:, -1:]                            # [n, 1]

    # --- interpolation: which segment does each target arc length fall in? --- #
    # searchsorted on the [n, H+1] cumulative array; clamp into [1, H] so the
    # (j-1) segment index is always valid. Rows past the end are overwritten by
    # the extrapolation branch below.
    j = torch.searchsorted(s_cum.contiguous(), s_new.contiguous().clamp(
        min=0.0)).clamp(1, lens.shape[1])            # [n, H]
    base = torch.gather(s_cum, 1, j - 1)             # [n, H]
    seg_len = torch.gather(lens, 1, j - 1)           # [n, H]
    u = ((s_new - base) / seg_len.clamp_min(_EPS)).clamp(0.0, 1.0)
    p0 = torch.gather(p, 1, (j - 1)[..., None].expand(-1, -1, 2))
    dv = torch.gather(seg, 1, (j - 1)[..., None].expand(-1, -1, 2))
    out = p0 + u[..., None] * dv                     # [n, H, 2]

    # --- extrapolation past the end, along the terminal tangent ------------- #
    tangent, degenerate = terminal_tangent(traj)     # [n, 2], [n]
    beyond = s_new > total                           # [n, H]
    if bool(beyond.any()):
        overshoot = (s_new - total).clamp_min(0.0)   # [n, H]
        ext = p[:, -1][:, None, :] + overshoot[..., None] * tangent[:, None, :]
        out = torch.where(beyond[..., None], ext, out)

    # a plan that never moved has no direction; leave it at the origin
    out = torch.where(degenerate[:, None, None], torch.zeros_like(out), out)
    beyond = beyond & ~degenerate[:, None]
    return {
        "traj": out,
        "extrapolated": beyond,
        "degenerate": degenerate,
        "frac_extrapolated": beyond.any(dim=-1).float().mean(),
    }


def constant_speed_schedule(v0: Tensor, horizon: int, dt: float,
                            scale: float = 1.0) -> Tensor:
    """``[n]`` speed -> ``[n, H]`` arc-length schedule ``scale * v0 * t * dt``.

    This is exactly what ``cv_holdv0`` assumes, expressed as a schedule so it can
    be composed with ANY shape. ``scale`` exists for the deliberate-degradation
    control (``scale = 0.5`` must score WORSE — the composite is known to rise
    for a slowed planner because a barely-moving plan is scored NaN)."""
    t = torch.arange(1, horizon + 1, dtype=v0.dtype, device=v0.device) * dt
    return float(scale) * v0.abs()[:, None] * t[None, :]


def straight_plan(schedule: Tensor) -> Tensor:
    """``[n, H]`` arc-length schedule -> ``[n, H, 2]`` straight-ahead plan.

    The zero-steering shape. Composed with ``constant_speed_schedule`` it
    reproduces ``cv_holdv0`` exactly, which is what makes the factorial in the
    2026-07-28 report self-validating: two of its four cells are already-
    published arms."""
    return torch.stack([schedule, torch.zeros_like(schedule)], dim=-1)


def retime(traj: Tensor, schedule: Tensor) -> dict[str, Tensor]:
    """Alias for :func:`resample_by_arclength` — the plan's shape, a new
    schedule. Kept under the name the report uses so code and prose agree."""
    return resample_by_arclength(traj, schedule)


# =========================================================================== #
# the architectural piece: the ego seam, zero-init                             #
# =========================================================================== #
def ego_vector(v0: Tensor, yaw_rate: Tensor, pose_scale: float) -> Tensor:
    """``[n]``, ``[n]`` -> ``[n, 2]`` ``[v0 / pose_scale, yr0]``.

    VERBATIM the trainer's contract (``train/flagship_losses.py:202-210``) so the
    two cannot drift. ⚠️ ``pose_scale``, NOT ``SPEED_SCALE`` — the operative
    action channel uses ``/10.0`` and the planner ego vector uses the pose
    scale; they are different ports and swapping them decodes garbage."""
    if pose_scale == 0:
        raise ValueError("pose_scale must be non-zero")
    return torch.stack([v0 / pose_scale, yaw_rate], dim=-1)


class EgoInputDropped(AssertionError):
    """A policy that OWNS trained ego weights was called without an ego vector."""


def assert_ego_is_fed(policy: nn.Module, ego: Tensor | None, *,
                      where: str = "<caller>") -> None:
    """⛔ THE GUARD FOR THE SILENT GAP. Call before every planner forward.

    ``TacticalPolicy`` / ``StrategicPolicy`` accept ``ego=None`` and quietly skip
    the ego term. That is correct for a policy built WITHOUT the lever, and it is
    a **silent evaluation bug** for one built with it: the trained ``ego_emb``
    weights are simply never exercised, no error is raised, no log line is
    written, and the arm gets scored as though the lever did nothing.

    MEASURED 2026-07-28: **all 8** eval/planner call sites in the repo pass the
    policy positionally with two arguments and none passes ``ego=`` —
    ``taniteval/closedloop.py:245,317``, ``planner_p2.py:279,340``,
    ``planning.py:166``, ``corpus_overlay.py:307``, ``blindimag.py:101``,
    ``probe_overlay.py:49``, ``panel_run.py:137,165``, ``refs/refa.py:260``. So
    ``flagship-v2corpus-30k`` — training with ``v2_ego_to_planners = true`` —
    would be evaluated ego-blind by every one of them.

    A policy with ``ego_emb is None`` is unaffected: this is a no-op for every
    arm in the 2026-07-27 panel, so adding the call changes no published number.
    """
    if getattr(policy, "ego_emb", None) is not None and ego is None:
        raise EgoInputDropped(
            f"{where}: {type(policy).__name__} has TRAINED ego weights "
            f"(ego_emb: {policy.ego_emb.in_features}->{policy.ego_emb.out_features}) "
            f"but was called with ego=None, so they are silently unused. Pass "
            f"ego = [v0/pose_scale, yr0] (tanitad.ego_plan.ego_vector), or "
            f"explicitly pass a zero vector if an ego-ablated arm is intended — "
            f"zeros are IN-DISTRIBUTION when the run used v2_ego_dropout, "
            f"ego=None is not the same thing (it skips the bias too).")


@torch.no_grad()
def attach_ego_input(policy: nn.Module, d_cond: int | None = None) -> nn.Module:
    """Graft the shipped ``ego_emb`` seam onto a built policy, ZERO-INITIALISED.

    ``TacticalPolicy`` and ``StrategicPolicy`` both already accept ``ego=`` and
    both already own an ``ego_emb`` slot that is ``None`` unless
    ``cfg.v2_ego_to_planners`` was on at construction. This attaches that exact
    module (``nn.Linear(2, d_cond)``) to a policy that was built without it, with
    **weight and bias zeroed**, so the grafted policy is numerically IDENTICAL to
    the original for every input until the graft is trained.

    ⇒ the ego/no-ego comparison is a **paired, within-checkpoint** contrast that
    differs in ``2 * d_cond + d_cond`` parameters and nothing else, and its
    control direction is guaranteed rather than hoped for.

    Raises if the policy already has an ``ego_emb`` — silently replacing trained
    weights is exactly the class of error this program keeps paying for."""
    if not hasattr(policy, "ego_emb"):
        raise TypeError(f"{type(policy).__name__} has no ego_emb slot — this "
                        f"graft only fits TacticalPolicy / StrategicPolicy")
    if policy.ego_emb is not None:
        raise ValueError(f"{type(policy).__name__} already has a trained "
                         f"ego_emb; refusing to overwrite it")
    if d_cond is None:
        # TacticalPolicy: ego_emb -> d_cond (the strategic ctx dim, = in_features
        # of the FiLM cond). StrategicPolicy: ego_emb -> cfg.d_cmd.
        d_cond = getattr(getattr(policy, "cfg", None), "d_cmd", None)
        if d_cond is None:
            raise ValueError("d_cond could not be inferred; pass it explicitly")
    emb = nn.Linear(2, int(d_cond))
    emb.weight.zero_()
    emb.bias.zero_()
    ref = next(policy.parameters())
    policy.ego_emb = emb.to(device=ref.device, dtype=ref.dtype)
    return policy
