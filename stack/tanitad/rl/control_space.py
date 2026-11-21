"""``tanitad.rl.control_space`` — the DD-v2 two-scalar scale policy, in CONTROL space.

⛔ **WHY THIS MODULE EXISTS, and it is not a nicety.** The refcv4b RL design
(`E-DDA-3c` / `E-DDA-3b`) states that the preferred exploration acts on the
candidate's **controls** ``(accel, curvature)`` and is re-rolled through
``rollout_unicycle``, *"so every explored candidate is flyable by construction"*
(`D-REFCV5-PLAN-4`), and it registers **metre-space noise as the DELIBERATE
REGRESSION arm that must FAIL flyability**.

⚠️ **MEASURED 2026-09-06, by reading the shipped code rather than the design:
CONTROL SPACE DID NOT EXIST.** ``refcv3_adapter.sample_offsets`` scales the
**offset waypoints** — ``scale = mean.abs() * cfg.noise_scale`` — and the strings
``control`` / ``rollout_unicycle`` / ``a_lon`` / ``alat`` appear **zero times** in
that file (content-verified read, 9,078 bytes). ⇒ **the arm that would have run is
the deliberate regression**, and the result table would have carried it as the
hypothesis. This module closes that gap without touching the adapter.

⭐ **WHAT IS FAITHFUL TO THE PUBLISHED MECHANISM** (`2512.07745`, sha256
`076ce47e…`, + released code at `1cd12a1`): DD-v2's RL stage draws **two scalars
per trajectory** — one longitudinal, one lateral — from ``randn([B, N, 1, 1])``
per axis and broadcasts them over the waypoints; the additive DDPM term is
**multiplied by zero**, so the two scalars are the *entire* stochasticity. The
exploration σ floor is **0.04** and the likelihood σ floor is **0.10**. Both are
reproduced here. ⛔ What is deliberately NOT reproduced is the *space*: theirs
scales waypoints, ours scales controls, because ours must land inside a friction
envelope that theirs delegates to a map-based drivable-area term we do not have.

⛔⛔ **THE LIKELIHOOD IS NOT THE DENSITY OF THE SAMPLER, AND THAT IS STATED, NOT
HIDDEN.** ``logp`` here is the density of the **two scalars**, not of the emitted
trajectory: the clamp to the envelope is a non-injective pushforward, so the
trajectory's true density has an atom on the envelope boundary. This is *the same
class of object as the published estimator* — DD-v2 evaluates an isotropic
16-coordinate Gaussian on samples drawn from a 2-scalar family, which is a
directional finite-difference along μ rather than the paper's per-step REINFORCE
(the DDv2 analysis §1.3 #3). ⇒ It is a valid **scale-perturbation** gradient and
is quoted as one. A design that claimed an exact policy gradient here would be
wrong.

⛔ **NO EGO-FUTURE, NO SITUATION CLASSIFIER, NO SELECTOR OUTPUT** enters anything
in this file. It transforms controls and integrates physics.

Tier: T0 training-side. Evidence class of the mechanism: `PUBLISHED-CODE`
(the two-scalar family and both σ floors); of the flyability property: **MEASURED
(ours)** — ``test_rl_control_space.py`` asserts it on adversarial inputs.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from ..models.kinematic import rollout_unicycle, unicycle_controls_from_path
from . import rewards as R

__all__ = [
    "SIGMA_EXPLORE_FLOOR",
    "SIGMA_LIKELIHOOD_FLOOR",
    "LOGP_MODES",
    "MIN_LOGP_SCALE",
    "ControlSpaceError",
    "scale_controls",
    "sample_control_scales",
    "policy_logp",
    "assert_carries_policy_gradient",
    "roll_controls",
    "sample_control_space",
    "controls_from_path",
    "envelope_violation",
]

#: DD-v2's exploration σ floor (`_model_rl.py`: `clip(σ_t, min=0.04)`), which its
#: own schedule never exceeds — so it is the whole exploration magnitude, not a
#: floor that occasionally binds.
SIGMA_EXPLORE_FLOOR = 0.04

#: DD-v2's likelihood σ floor (`clip(σ_t, 0.1)`). Kept distinct from the
#: exploration σ **on purpose**: they are different numbers in the published code
#: and collapsing them silently changes the gradient's scale.
SIGMA_LIKELIHOOD_FLOOR = 0.10


#: How ``sample_control_space`` computes ``logp``.
#:   "policy" (DEFAULT) -- the SCORE-FUNCTION density of the DRAWN CONTROLS under
#:        ``N(base_controls, (|base_controls| * likelihood_sigma)^2)``, with the
#:        sample DETACHED and the density's parameters live. Differentiable w.r.t.
#:        the decoder, which is the whole point of a policy gradient.
#:   "scalar" -- the density of the TWO SCALARS. ⛔ A DIAGNOSTIC ONLY: it is a
#:        function of the DRAW alone, so it is CONSTANT in the policy parameters
#:        and carries NO gradient to them.
LOGP_MODES = ("policy", "scalar")

#: Floor on the per-coordinate density scale. A control the decoder leaves at
#: exactly 0 would otherwise give a zero-width Gaussian, ``logp = -inf`` and a
#: NaN loss -- which surfaces as a dead run rather than as the degenerate control
#: it is. Same guard, same reason, as ``refcv3_adapter.sample_offsets``.
MIN_LOGP_SCALE = 1e-3

LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


class ControlSpaceError(ValueError):
    """Raised when a control-space sample cannot be drawn honestly."""


def _check_controls(controls: Tensor) -> None:
    if controls.dim() < 2 or controls.shape[-1] != 2:
        raise ControlSpaceError(
            f"controls must be [..., K, 2] = (accel, curvature), got "
            f"{tuple(controls.shape)}")
    if not torch.isfinite(controls).all():
        raise ControlSpaceError("controls carry non-finite entries")


def scale_controls(controls: Tensor, s_lon: Tensor, s_lat: Tensor, *,
                   a_max: float = R.A_MAX_MPS2,
                   kappa_max: float = R.KAPPA_MAX_1PM) -> Tensor:
    """Apply the two scale factors and CLAMP to the envelope.

    ``controls [..., K, 2]`` = (accel, curvature); ``s_lon``/``s_lat`` broadcast
    against ``controls[..., 0]``.

    ⭐ **THE CLAMP IS THE POINT.** Scaling alone can push a candidate outside the
    envelope, which is exactly the defect the control-space design exists to
    avoid; clamping makes flyability a **property of the construction** rather
    than something the reward has to police afterwards. ⚠️ It also means the
    sampler is not a pure location-scale family — see the module docstring's
    statement about the likelihood.
    """
    _check_controls(controls)
    a = controls[..., 0] * s_lon
    k = controls[..., 1] * s_lat
    a = a.clamp(-abs(a_max), abs(a_max))
    k = k.clamp(-abs(kappa_max), abs(kappa_max))
    return torch.stack((a, k), dim=-1)


def sample_control_scales(shape, *, sigma: float = SIGMA_EXPLORE_FLOOR,
                          likelihood_sigma: float = SIGMA_LIKELIHOOD_FLOOR,
                          generator: torch.Generator | None = None,
                          device=None, dtype=torch.float32):
    """Draw ``(s_lon, s_lat, logp)``; ``s = 1 + eps``, ``eps ~ N(0, sigma^2)``.

    ``shape`` is the leading shape (e.g. ``[B, N, G]``). ``logp`` is the summed
    log-density of the **two scalars** under ``N(0, likelihood_sigma^2)`` — two
    terms, not ``2*K``, because two scalars is the whole sample.
    """
    if sigma <= 0 or likelihood_sigma <= 0:
        raise ControlSpaceError("sigma and likelihood_sigma must be positive")
    eps = torch.randn(*shape, 2, generator=generator, device=device, dtype=dtype)
    eps = eps * float(sigma)
    s = 1.0 + eps
    ls = float(likelihood_sigma)
    logp = (-(eps ** 2) / (2 * ls * ls) - math.log(ls)
            - 0.5 * math.log(2 * math.pi)).sum(dim=-1)
    return s[..., 0], s[..., 1], logp


def policy_logp(base: Tensor, drawn: Tensor, *,
                likelihood_sigma: float = SIGMA_LIKELIHOOD_FLOOR,
                min_scale: float = MIN_LOGP_SCALE) -> Tensor:
    """Score-function log-density of ``drawn`` controls under the policy at ``base``.

    ⛔⛔ WHY THIS FUNCTION EXISTS, AND IT IS A MEASURED DEFECT IN THIS MODULE'S
    OWN FIRST DRAFT. ``sample_control_scales`` returns the density of the TWO
    SCALARS, which is a function of the DRAW alone. MEASURED 2026-09-06 by
    running it: the ``logp`` returned by ``sample_control_space`` had
    ``requires_grad`` **False** and no ``grad_fn`` at all, so
    ``(-(logp * advantage).sum()).backward()`` raised *"element 0 of tensors does
    not require grad"*. ⇒ **Wiring that ``logp`` into an RL stage gives either a
    crash at the first backward or, if it is summed with a term that does carry
    grad, a policy gradient that is IDENTICALLY ZERO** -- a silent no-op arm
    wearing the hypothesis' name. That is the same family of defect
    ``refcv3_adapter`` already carries a correction for, and the reason its
    comment insists the gradient lives in ``(sample.detach() - mean)``.

    The correct REINFORCE form treats the DRAWN sample as a constant and the
    density's parameters as live::

        scale   = clamp(|base| * likelihood_sigma, min_scale)
        eps_eff = (drawn.detach() - base) / scale
        logp    = sum(-0.5 * eps_eff^2 - log(scale) - log(sqrt(2 pi)))

    summed over the (K, 2) control axes.

    ⛔ STATED LIMIT, unchanged and not hidden: the sampler has RANK 2 (two
    scalars) while this density is written per coordinate, so the 2K terms are
    perfectly correlated. That is the SAME class of estimator the published
    method trains with -- a directional finite difference along the mean -- and
    it is quoted as a **scale-perturbation** gradient, never as an exact policy
    gradient. ``likelihood_sigma`` is DD-v2's 0.10 and is deliberately distinct
    from the exploration sigma 0.04.
    """
    if likelihood_sigma <= 0:
        raise ControlSpaceError("likelihood_sigma must be positive")
    _check_controls(base)
    _check_controls(drawn)
    scale = (base.abs() * float(likelihood_sigma)).clamp_min(float(min_scale))
    eps_eff = (drawn.detach() - base) / scale
    lp = -0.5 * eps_eff.pow(2) - torch.log(scale) - LOG_SQRT_2PI
    return lp.sum(dim=(-1, -2))


def assert_carries_policy_gradient(logp: Tensor, what: str = "logp") -> Tensor:
    """⛔ REFUSE a ``logp`` that cannot move the policy. Returns it unchanged.

    A positive assertion, run BEFORE any compute is spent: a detached ``logp``
    makes the REINFORCE term identically zero, and every downstream number then
    describes an arm that did nothing. This is cheap and it fails loudly.
    """
    if not torch.is_tensor(logp):
        raise ControlSpaceError(f"{what} is not a tensor")
    if not logp.requires_grad or logp.grad_fn is None:
        raise ControlSpaceError(
            f"{what} carries NO gradient to the policy (requires_grad="
            f"{bool(logp.requires_grad)}, grad_fn={logp.grad_fn!r}). The "
            "REINFORCE term would be identically zero and the arm a SILENT "
            "NO-OP. Use logp_mode='policy'.")
    return logp


def controls_from_path(path: Tensor, *, dt: float) -> Tensor:
    """``[..., K, 2]`` ego-frame waypoints -> the ``(accel, curvature)`` they imply.

    A thin, shape-preserving wrapper over the programme's SINGLE inverse map,
    ``models.kinematic.unicycle_controls_from_path`` -- imported rather than
    re-derived, because a second inverse would make any control-space ablation
    measure the inverse instead of the action space. ⛔ The path must be on a
    UNIFORM ``dt`` grid; the caller states which prefix it sliced.
    """
    if path.dim() < 2 or path.shape[-1] != 2:
        raise ControlSpaceError(
            f"path must be [..., K, 2], got {tuple(path.shape)}")
    lead = path.shape[:-2]
    K = int(path.shape[-2])
    ctl = unicycle_controls_from_path(path.reshape(-1, K, 2).to(torch.float32),
                                      dt=float(dt))
    return ctl.reshape(*lead, K, 2)


def roll_controls(state0: Tensor, controls: Tensor, *, dt: float,
                  a_max: float = R.A_MAX_MPS2,
                  kappa_max: float = R.KAPPA_MAX_1PM) -> Tensor:
    """Integrate controls to ego-frame waypoints ``[..., K, 2]``.

    ``state0 [..., 4]`` = (x, y, yaw, v). Flattens the leading axes, calls
    ``rollout_unicycle`` (the programme's single integrator — a second one would
    make any control-space ablation measure the integrator), and returns xy.
    """
    _check_controls(controls)
    lead = controls.shape[:-2]
    K = int(controls.shape[-2])
    if state0.shape[:-1] != lead:
        raise ControlSpaceError(
            f"state0 leading shape {tuple(state0.shape[:-1])} != controls' "
            f"{tuple(lead)}")
    st = rollout_unicycle(state0.reshape(-1, 4).to(torch.float32),
                          controls.reshape(-1, K, 2).to(torch.float32),
                          dt=float(dt), accel_limit=float(a_max),
                          curvature_limit=float(kappa_max))
    return st[..., :2].reshape(*lead, K, 2)


def sample_control_space(state0: Tensor, controls: Tensor, *, group: int,
                         dt: float, sigma: float = SIGMA_EXPLORE_FLOOR,
                         likelihood_sigma: float = SIGMA_LIKELIHOOD_FLOOR,
                         a_max: float = R.A_MAX_MPS2,
                         kappa_max: float = R.KAPPA_MAX_1PM,
                         generator: torch.Generator | None = None,
                         logp_mode: str = "policy"):
    """The whole sampler: ``(traj [B, N, G, K, 2], logp [B, N, G], ctl [B, N, G, K, 2])``.

    ``state0 [B, 4]``, ``controls [B, N, K, 2]``, ``group`` = G intra-anchor
    samples (DD-v2 uses **4**). ⛔ ``group < 2`` is refused: a group-relative
    advantage over one sample is identically zero, which is a silent no-op arm —
    the family of defect this programme has already paid a GPU-hour for.
    """
    g = int(group)
    if g < 2:
        raise ControlSpaceError(
            f"group must be >= 2 for a group-relative advantage, got {g}")
    _check_controls(controls)
    if state0.dim() != 2 or state0.shape[-1] != 4:
        raise ControlSpaceError(
            f"state0 must be [B, 4] = (x, y, yaw, v), got {tuple(state0.shape)}")
    B, N, K, _ = controls.shape
    if state0.shape[0] != B:
        raise ControlSpaceError(f"state0 batch {state0.shape[0]} != controls' {B}")

    if logp_mode not in LOGP_MODES:
        raise ControlSpaceError(
            f"logp_mode must be one of {LOGP_MODES}, got {logp_mode!r}")
    s_lon, s_lat, logp_scalar = sample_control_scales(
        (B, N, g), sigma=sigma, likelihood_sigma=likelihood_sigma,
        generator=generator, device=controls.device, dtype=controls.dtype)
    base = controls.unsqueeze(2).expand(B, N, g, K, 2)
    ctl = scale_controls(base, s_lon[..., None], s_lat[..., None],
                         a_max=a_max, kappa_max=kappa_max)
    st0 = state0[:, None, None, :].expand(B, N, g, 4)
    traj = roll_controls(st0, ctl, dt=dt, a_max=a_max, kappa_max=kappa_max)
    logp = (policy_logp(base, ctl, likelihood_sigma=likelihood_sigma)
            if logp_mode == "policy" else logp_scalar)
    return traj, logp, ctl


def envelope_violation(controls: Tensor, *, a_max: float = R.A_MAX_MPS2,
                       kappa_max: float = R.KAPPA_MAX_1PM) -> Tensor:
    """Per-candidate envelope overshoot, ``[...]``, **0.0 when flyable**.

    ⭐ The instrument the flyability claim is read on. ``max(|a|/a_max,
    |kappa|/kappa_max) - 1``, clamped at 0 — so a control-space sample must read
    **exactly 0.0** and a metre-space one (the deliberate regression) need not.
    """
    _check_controls(controls)
    over = torch.maximum((controls[..., 0].abs() / abs(a_max)),
                         (controls[..., 1].abs() / abs(kappa_max)))
    return (over.amax(dim=-1) - 1.0).clamp_min(0.0)
