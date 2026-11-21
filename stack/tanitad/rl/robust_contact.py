"""``tanitad.rl.robust_contact`` — contact under AGENT-MOTION PREDICTION ERROR.

⭐ WHY THIS TERM AND NOT ANOTHER. A constraint can forbid; it cannot rank. RL's target is
therefore the residual a constraint **provably cannot discharge**, and on this programme
that residual is now named and measured:

* contact at ``dt = 0`` is closed **by construction** — ``refs/contact_projection.py``
  takes the fan from 3.4277 % to a structural **0.000000 %** at **+0.0000 m** ADE, zero
  GPU (`D-RL-COLL-PRICE-1`). ⛔ Spending reward budget on it buys nothing;
* but contact under **lead-track timing error** is not closed and cannot be: MEASURED
  `fan_contact` **0.034277** at ``dt = 0`` → **0.039811** (1.16x) at ``dt = -0.5 s`` →
  **0.066243** (1.93x) at ``dt = -1.0 s``, **monotone across the whole -1.0…+2.0 s sweep
  and roughly linear, with NO THRESHOLD to sit safely below** (`D-RL-TIMING-SURFACE-1`,
  `.../2026-09-05-rl-generator-collisions/raw/p8_timing_error_surface.json`).

⭐ **That shape is the entire argument.** A residual with a knee can be widened away by a
bigger safety margin; a smooth monotone one cannot, because every margin you pick is still
on the slope. That is what a learned policy is for, and it is why this term — not the
``dt = 0`` contact term the refcv3 campaign aimed at — is where RL has headroom.

⚠️ **SIGN SEMANTICS ARE GEOMETRY, NEVER THE ADJECTIVE "EARLY".** ``dt < 0`` places the lead
**EARLIER ALONG ITS OWN PATH**, i.e. **CLOSER to a following ego** — the risk direction.
Reading it the other way inverts the whole table. The docstring states the geometry because
the label is ambiguous and a mislabelled sign has already cost this programme a retraction.

⛔ WHAT THIS TERM IS NOT. It does not read the ego's own recorded future in any form — no
ADE, no speed-profile match, no "stay near the logged path". Those are the imitation loss in
a reward costume (the DDv2 analysis §3.3 echo table). It reads **other agents' recorded
tracks**, perturbed in time: the environment, not the answer.

⚠️ STATED LIMIT — NON-REACTIVE AGENTS. The lead was recorded reacting to the *human*, so a
strongly deviating policy meets a replay that will not yield. This is NAVSIM's own caveat
and it bounds the term's realism, not its admissibility. Perturbing the track in time does
**not** repair it; it measures robustness to *our* prediction error, which is a different
and narrower claim, and the only one made here.

Tier: T0 training-side reward component. Evidence class of the surface it targets:
MEASURED (ours), artifact cited above.
"""
from __future__ import annotations

import torch
from torch import Tensor

from . import rewards as R

__all__ = ["DEFAULT_SHIFTS_S", "shift_track", "contact_under_shift",
           "robust_contact", "make_robust_contact_component"]

#: The shift grid. Weighted toward the RISK direction (``dt < 0``) because that is where
#: the measured surface rises; ``0.0`` is included so the term reduces exactly to
#: ``rewards._collision`` when the weight is concentrated there — the T1 control below.
DEFAULT_SHIFTS_S: tuple[float, ...] = (-1.0, -0.5, 0.0, +0.5)


def shift_track(lead: Tensor, dt_shift_s: float, dt_s: float) -> Tensor:
    """Resample ``lead [..., S, 2]`` at times shifted by ``dt_shift_s`` ALONG ITS OWN PATH.

    Linear interpolation on the track's own index grid with **edge clamping** (a shift
    past either end holds the endpoint rather than extrapolating a fictional
    continuation — an invented lead position is a fabricated obstacle).

    ``dt_shift_s < 0`` samples EARLIER indices, placing the lead further back along its
    path, i.e. closer to a following ego. See the sign note in the module docstring.
    """
    if dt_s <= 0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    S = lead.shape[-2]
    idx = torch.arange(S, device=lead.device, dtype=lead.dtype)
    src = (idx + dt_shift_s / dt_s).clamp(0.0, S - 1.0)
    lo = src.floor().long()
    hi = src.ceil().long()
    w = (src - lo.to(lead.dtype)).unsqueeze(-1)            # [S, 1]
    a = lead.index_select(-2, lo)
    b = lead.index_select(-2, hi)
    return a * (1.0 - w) + b * w


def contact_under_shift(traj: Tensor, ctx: dict, dt_shift_s: float) -> Tensor:
    """``rewards._collision`` evaluated against a lead track shifted by ``dt_shift_s``.

    Returns the component's own value in ``[-1, 0]`` (−1 = contact), so it composes with
    the rest of the library unchanged.
    """
    lead = ctx.get("lead_path")
    if lead is None:
        # ABSENCE MEANS NO CONSTRAINT THEREFORE NO PENALTY -- rewards.py:152-171. A term
        # that punished a missing scene fact would punish the corpus, not the policy.
        return torch.zeros(traj.shape[:-2], dtype=traj.dtype, device=traj.device)
    dt_s = float(ctx.get("dt", R.DT_S))
    shifted = lead if dt_shift_s == 0.0 else shift_track(lead, dt_shift_s, dt_s)
    return R.COMPONENTS["collision"](traj, {**ctx, "lead_path": shifted})


def robust_contact(traj: Tensor, ctx: dict) -> Tensor:
    """Expected contact over a distribution of lead-track TIMING error. ``[-1, 0]``.

    ``ctx`` may carry ``robust_shifts_s`` (tuple) and ``robust_shift_weights`` (tuple,
    normalised here). Defaults: :data:`DEFAULT_SHIFTS_S`, uniform.

    ⭐ THE T1 CONTROL, AND IT IS EXACT: with ``robust_shifts_s = (0.0,)`` this returns
    ``rewards._collision`` bit-for-bit. A robustness term that does not reduce to the
    point estimate at zero perturbation is measuring something else, and the test suite
    asserts the identity rather than trusting it.
    """
    shifts = tuple(ctx.get("robust_shifts_s", DEFAULT_SHIFTS_S))
    if not shifts:
        raise ValueError("robust_contact needs at least one shift")
    w = ctx.get("robust_shift_weights")
    if w is None:
        w = (1.0,) * len(shifts)
    if len(w) != len(shifts):
        raise ValueError(f"{len(w)} weights for {len(shifts)} shifts")
    tot = float(sum(w))
    if tot <= 0:
        raise ValueError("robust_shift_weights must sum to a positive number")
    out = None
    for s, wi in zip(shifts, w):
        term = contact_under_shift(traj, ctx, float(s)) * (float(wi) / tot)
        out = term if out is None else out + term
    return out


def make_robust_contact_component() -> R.RewardComponent:
    """The component record, so ``audit``/``COMPONENTS`` treat it like any other.

    ``degenerate`` is stated honestly: standing still never contacts under ANY shift, so
    this term is hackable alone exactly as ``collision`` is, and it needs ``progress`` to
    be checked — the pairing ``DEFAULT_WEIGHTS`` is built around (`rewards.py:580-585`).
    """
    return R.RewardComponent(
        "robust_contact", robust_contact, -1.0, 0.0,
        degenerate="stand still forever (never contacts under any timing error)",
        grounded="gt-derived", hackable_alone=True, neutral=0.0)
