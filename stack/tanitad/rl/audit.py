"""Reward-hacking and disjointness audits — the guard that must be able to FAIL.

⛔ WHY THIS FILE IS THE POINT OF THE LIBRARY
--------------------------------------------
A reward you cannot hack is a reward you have not thought about. This module
runs a fixed panel of DEGENERATE POLICIES — hand-built trajectories that drive
badly in a specific way — against a composed reward and reports which of them
win. The rule, from the programme's quality system: **a guard must be shown ABLE
TO FAIL before its PASS means anything.** So `rewards.HACKABLE_WEIGHTS`
(progress-only) ships on purpose as the deliberate-regression arm:

    audit_reward(RewardSpec(HACKABLE_WEIGHTS))  ->  MUST return flagged=True

If that ever returns clean, the audit is broken and every PASS it ever gave is
worthless. `tests/test_rl_audit.py` pins both directions.

THE THREE AUDITS
----------------
1. ``assert_selector_disjoint`` — STRUCTURAL. The reward context may not carry
   any model-produced ranking (`rewards.FORBIDDEN_REWARD_INPUTS`). This is the
   winner's-curse firewall in code: a reward built from the selector's score
   trains the generator to agree with the stack's weakest component, and the
   resulting number is an echo, not a capability. DDv2 names the same defect
   (selector over-reliance) and our SEL-1 refusal measured it independently.
2. ``audit_reward`` — BEHAVIOURAL. Does a degenerate policy beat a good one?
3. ``report_component_coverage`` — HONESTY. A component that never fires (no
   obstacles in frame, no lead vehicle) contributes a constant and is therefore
   invisible in the advantage. Reported explicitly, with its n, rather than
   silently trusted. *(A collision term that never fires is not safety — it is
   an absent measurement wearing safety's name.)*

Evidence class: MEASURED (ours). Tier: N/A — this is an instrument audit, not
a capability claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from . import rewards as R


# ---------------------------------------------------------------------------
# 1. Structural: the reward may never see the selector
# ---------------------------------------------------------------------------

def assert_selector_disjoint(ctx: dict) -> None:
    """Raise if a reward context carries any model-produced ranking.

    ⛔ Call this on EVERY reward evaluation path, not just in tests — the whole
    point is that it is impossible to wire the selector into the reward by
    accident during a late-night debugging session.
    """
    bad = sorted(set(ctx) & R.FORBIDDEN_REWARD_INPUTS)
    if bad:
        raise ValueError(
            "REWARD/SELECTOR DISJOINTNESS VIOLATED: the reward context carries "
            f"model-produced ranking key(s) {bad}. The reward must never be the "
            "selector's own score (see rewards.py module docstring: DDv2's named "
            "defect is selector over-reliance; our SEL-1 winner's-curse refusal "
            "measured the same thing). Pass scene facts and geometry only.")


# ---------------------------------------------------------------------------
# 2. Behavioural: the degenerate-policy panel
# ---------------------------------------------------------------------------

def _line(n_steps: int, v: float, dt: float = R.DT_S,
          lateral: float = 0.0, device=None) -> Tensor:
    """A straight constant-speed path, optionally offset laterally."""
    t = torch.arange(n_steps, dtype=torch.float32, device=device)
    return torch.stack([v * dt * t, torch.full_like(t, lateral)], dim=-1)


def degenerate_panel(n_steps: int = 21, dt: float = R.DT_S,
                     device=None) -> dict[str, Tensor]:
    """The fixed panel of badly-driving trajectories, each ``[S, 2]``.

    Each entry is named for the failure it embodies, so a flag reads as a
    diagnosis rather than a score.
    """
    t = torch.arange(n_steps, dtype=torch.float32, device=device)
    zeros = torch.zeros_like(t)
    return {
        # maximises `progress`; drives through whatever is ahead
        "bullet_straight": _line(n_steps, 30.0, dt, device=device),
        # maximises feasibility/comfort/collision-avoidance; never moves
        "frozen": torch.stack([zeros, zeros], dim=-1),
        # physically impossible: accelerates far beyond a_max
        "teleport": torch.stack([0.5 * 40.0 * (t * dt) ** 2, zeros], dim=-1),
        # violates kappa_max: a tight spiral. ⚠️ radius 2.0 is deliberate —
        # at radius 5.0 the curvature is 1/5 = 0.2 = EXACTLY KAPPA_MAX, so the
        # entry read as FEASIBLE (0.985) and did not embody the failure it is
        # named for. Caught by test_teleport_and_spinner_are_actually_infeasible,
        # which is why that test exists: a panel entry that does not misbehave
        # silently weakens every audit that uses it.
        "spinner": torch.stack([2.0 * torch.cos(t * 0.6) - 2.0,
                                2.0 * torch.sin(t * 0.6)], dim=-1),
        # jerk-maximal zig-zag; smooth in neither axis
        "shaky": torch.stack([10.0 * dt * t,
                              0.7 * ((-1.0) ** t)], dim=-1),
    }


def sane_reference_trajectory(n_steps: int = 21, dt: float = R.DT_S,
                              v: float = 10.0, device=None) -> Tensor:
    """A sane TRAJECTORY: straight, lawful speed, zero curvature, zero jerk.

    ⚠️ RENAMED 2026-08-29. This was ``reference_policy`` — which collided with
    ``anchor.ReferencePolicy``, a FROZEN MODEL. A trajectory and a policy are
    different objects, and the package exported both under one name the day
    after TRAIN-C5 established that conflated names hide design changes. The
    old name remains as a deprecated alias so existing callers do not break.

    This is deliberately MODEST — it does not maximise progress. A reward that
    ranks `bullet_straight` above this one is telling you it prefers speed to
    everything else, which is the finding we want surfaced.
    """
    return _line(n_steps, v, dt, device=device)


#: deprecated alias — prefer ``sane_reference_trajectory`` (see the rename note)
reference_policy = sane_reference_trajectory


@dataclass
class AuditReport:
    """Three states, never two.

    ⛔ ``inconclusive`` exists because a two-state audit LIES in the most
    common case. MEASURED 2026-08-29: auditing the default reward with an empty
    context reported ``bullet_straight`` as a winner — but with no obstacles, no
    lead vehicle and no GT in the context, ``collision``/``headway``/
    ``gt_similarity`` are all identically zero, so **progress is the only term
    that discriminates and the fastest policy wins by construction**. That is
    not a hackable reward; it is an underpowered audit. Reporting it as FLAGGED
    would be a false positive, and reporting it as clean would be worse.

    Same doctrine as the rank gate: *a criterion that cannot RULE at the
    configured settings must say so loudly* rather than return a verdict.
    """
    flagged: bool
    reason: str
    scores: dict[str, float]
    reference: float
    winners: list[str]
    inconclusive: bool = False
    dead_components: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.dead_components is None:
            self.dead_components = []

    @property
    def verdict(self) -> str:
        if self.inconclusive:
            return "INCONCLUSIVE"
        return "FLAGGED" if self.flagged else "clean"

    def __str__(self) -> str:
        rows = "\n".join(f"    {k:<18} {v:+.4f}"
                         for k, v in sorted(self.scores.items(),
                                            key=lambda kv: -kv[1]))
        return (f"[reward audit] {self.verdict}: {self.reason}\n"
                f"    {'REFERENCE':<18} {self.reference:+.4f}\n{rows}")


def audit_reward(spec: R.RewardSpec, ctx: dict | None = None, *,
                 n_steps: int = 21, margin: float = 0.0,
                 device=None) -> AuditReport:
    """Score the degenerate panel against a sane reference.

    Returns FLAGGED if a degenerate policy beats the reference, ``clean`` if
    none does AND every weighted component actually discriminated, and
    INCONCLUSIVE if some weighted component was constant across the whole panel
    (an audit run in a context that cannot exercise the reward).

    ``margin`` is how far above the reference a degenerate must score to count
    as winning; 0.0 means "ties are not failures, strictly beating is".
    """
    ctx = dict(ctx or {})
    assert_selector_disjoint(ctx)
    ctx.setdefault("dt", spec.dt)

    ref_traj = sane_reference_trajectory(n_steps, spec.dt, device=device)
    ref = float(spec(ref_traj, ctx))

    panel = degenerate_panel(n_steps, spec.dt, device)
    scores = {name: float(spec(traj, ctx)) for name, traj in panel.items()}
    winners = [k for k, v in scores.items() if v > ref + margin]

    # which weighted components actually varied across panel + reference?
    stacked = torch.stack(list(panel.values()) + [ref_traj])
    dead = [n for n, v in spec.per_component(stacked, ctx).items()
            if spec.weights.get(n, 0.0) != 0.0
            and float(v.max() - v.min()) <= 1e-9]

    if dead:
        reason = (f"audit is UNDERPOWERED: weighted component(s) {dead} were "
                  "CONSTANT across the whole panel, so they cannot influence the "
                  "ranking. Supply a context that exercises them (obstacles / "
                  "lead_path / gt_traj) before trusting any verdict. "
                  f"(raw winners under this context: {winners or 'none'})")
        return AuditReport(False, reason, scores, ref, winners,
                           inconclusive=True, dead_components=dead)
    if winners:
        reason = (f"{len(winners)} degenerate polic(y/ies) score above the sane "
                  f"reference: {winners}. The reward is hackable as weighted.")
    else:
        reason = ("no degenerate policy in the panel beats the sane reference "
                  "(this is necessary, NOT sufficient — the panel is fixed and "
                  "finite, so it bounds the claim)")
    return AuditReport(bool(winners), reason, scores, ref, winners)


# ---------------------------------------------------------------------------
# 3. Honesty: which components actually carried information?
# ---------------------------------------------------------------------------

def report_component_coverage(spec: R.RewardSpec, traj: Tensor,
                              ctx: dict) -> dict[str, dict]:
    """Per component: its spread over the candidate axis, and whether it fired.

    A component whose value is CONSTANT across candidates contributes exactly
    nothing to a group-relative advantage (it cancels in the centring). Such a
    component is not a safeguard — it is an absent measurement. Reported with
    its n so the absence is visible in the run record.
    """
    assert_selector_disjoint(ctx)
    out: dict[str, dict] = {}
    for name, val in spec.per_component(traj, ctx).items():
        v = val.detach().reshape(-1)
        spread = float(v.max() - v.min()) if v.numel() else 0.0
        out[name] = {
            "weight": spec.weights[name],
            "mean": float(v.mean()) if v.numel() else float("nan"),
            "spread": spread,
            "n": int(v.numel()),
            "fired": spread > 1e-9,
            "_note": ("CONSTANT across candidates -> cancels in the "
                      "group-relative advantage and carries NO signal"
                      if spread <= 1e-9 else ""),
        }
    return out
