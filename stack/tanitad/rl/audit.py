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
2. ``audit_reward`` — BEHAVIOURAL. Does a degenerate policy beat a good one, and
   — added 2026-09-11 — **can the reward tell two degenerate policies apart at
   all?** ⛔ Those are different questions and the second one caught a reward the
   first called ``clean``: at the pilot's ``dt = 0.5`` the progress-only
   regression arm scored ``bullet_straight`` / ``teleport`` / ``shaky`` at the
   identical **1.5**. Nothing was constant, so the liveness half was satisfied;
   the RANKING had collapsed anyway. See `audit_reward`'s separation check.
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


#: The panel's own two speeds — the reference's and ``bullet_straight``'s. Named
#: here so :func:`exercising_ctx` derives its geometry from the panel instead of
#: carrying a second, independently-drifting copy of it.
PANEL_V_REF_MPS = 10.0
PANEL_V_BULLET_MPS = 30.0


def exercising_ctx(n_steps: int = 21, dt: float = R.DT_S, device=None) -> dict:
    """A context that actually EXERCISES ``collision`` and ``headway``.

    ⛔ WHY THIS FUNCTION EXISTS. `audit_reward` called with no context cannot
    rule on any reward that weights ``collision`` or ``headway``: with no
    obstacle and no lead they are identically constant, and the audit correctly
    returns INCONCLUSIVE. MEASURED 2026-09-10 — the 2,000-step pilot's P1 arm
    (DEFAULT_WEIGHTS) recorded exactly that: *"weighted component(s)
    ['collision', 'headway'] were CONSTANT across the whole panel … supply a
    context that exercises them before trusting any verdict."* ⇒ **no P1 verdict
    was admissible at all**, with or without the >=GT bar. This is the context it
    asked for.

    ⭐ EVERY NUMBER IS DERIVED FROM THE PANEL, NOT CHOSEN. Two placements were
    wrong before this geometry and both are instructive (see the history in
    `tests/test_rl_audit.py::rich_ctx`): an obstacle *under* the reference makes
    ``frozen`` legitimately win, and one off to the side makes ``collision``
    constant again.

      * ``obstacle_x`` = the MIDPOINT between the reference's reach and
        ``bullet_straight``'s — far enough that the lawful reference never gets
        there, close enough that the 30 m/s policy and ``teleport`` drive
        straight through it. *That is the scene the reward must get right:
        reckless speed meets an obstacle the careful policy never reaches.*
      * ``lead_path`` = the reference offset by ``lead_len_m + target_time_gap_s
        * v_ref``, i.e. the standoff ``_headway`` itself is written against — no
        new constant, and nothing tunable.

    ⭐ THE CROSS-CHECK THAT MAKES IT TRUSTWORTHY: at the module default
    ``dt = 0.1`` this derivation reproduces the banked literal ``[40.0, 0.0]``
    and the standoff ``24.5`` EXACTLY — values that were found empirically, by
    hand, months earlier. An independently authored derivation landing on the
    number already in the tree is evidence; re-running the producer's own
    arithmetic would only have measured determinism.

    ⚠️ IT SCALES WITH ``dt``, AND IT MUST. The panel is built from
    ``v * dt * t``, so at the pilot's ``dt = 0.5`` the reference reaches 100 m,
    not 20 m — and the banked literal 40.0 then sits UNDER the reference, which
    is the first of the two historical mistakes. MEASURED at both: ``default``
    reads **clean** (dead=[]) at dt 0.1 and at dt 0.5; ``hackable`` reads
    **FLAGGED** at both.
    """
    ref = sane_reference_trajectory(n_steps, dt, v=PANEL_V_REF_MPS, device=device)
    span = dt * (n_steps - 1)
    obstacle_x = 0.5 * (PANEL_V_REF_MPS + PANEL_V_BULLET_MPS) * span
    standoff = 4.5 + 2.0 * PANEL_V_REF_MPS       # lead_len_m + target_time_gap_s*v
    return {
        "obstacles": torch.tensor([[obstacle_x, 0.0]], device=device),
        "gt_traj": ref,
        "lead_path": ref + torch.tensor([standoff, 0.0], device=device),
    }


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
    #: ⭐ THE PANEL ENTRIES THAT TIE AT THE PANEL'S OWN MAXIMUM (see
    #: `audit_reward`). Reported in EVERY branch, including INCONCLUSIVE, so the
    #: fact is visible even where it does not decide the verdict.
    tied_at_max: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.dead_components is None:
            self.dead_components = []
        if self.tied_at_max is None:
            self.tied_at_max = []

    @property
    def verdict(self) -> str:
        if self.inconclusive:
            return "INCONCLUSIVE"
        return "FLAGGED" if self.flagged else "clean"

    def __str__(self) -> str:
        rows = "\n".join(f"    {k:<18} {v:+.4f}"
                         for k, v in sorted(self.scores.items(),
                                            key=lambda kv: -kv[1]))
        tie = (f"    TIED AT MAX        {self.tied_at_max}\n"
               if len(self.tied_at_max) >= 2 else "")
        return (f"[reward audit] {self.verdict}: {self.reason}\n"
                f"    {'REFERENCE':<18} {self.reference:+.4f}\n{rows}\n{tie}")


#: ⭐ THE TIE TOLERANCE FOR THE SEPARATION CHECK, WRITTEN AS A LITERAL AND NAMED
#: ONCE. Deliberately the same magnitude as the dead-component tolerance below
#: it: both ask "did this quantity actually move?", and two answers to one
#: question drifting apart is the `frac_above_bar` two-epsilon defect
#: (`advantage.gt_bar_mask`) in a second costume.
TIE_ATOL = 1e-9


def audit_reward(spec: R.RewardSpec, ctx: dict | None = None, *,
                 n_steps: int = 21, margin: float = 0.0,
                 tie_atol: float = TIE_ATOL,
                 device=None) -> AuditReport:
    """Score the degenerate panel against a sane reference.

    Returns FLAGGED if a degenerate policy beats the reference **or if two or
    more panel policies TIE at the panel's own maximum**, ``clean`` if neither
    holds AND every weighted component actually discriminated, and INCONCLUSIVE
    if some weighted component was constant across the whole panel (an audit run
    in a context that cannot exercise the reward).

    ``margin`` is how far above the reference a degenerate must score to count
    as winning; 0.0 means "ties are not failures, strictly beating is".

    ⛔⛔ THE SEPARATION CHECK, AND WHY LIVENESS WAS NOT ENOUGH (2026-09-11).
    MEASURED on P2 of the 2,000-step pilot, whose reward is `HACKABLE_WEIGHTS`
    (progress-only) run at the pilot's ``dt = 0.5``: this function returned
    **``verdict: clean``, ``flagged: False``, ``dead_components: []``** while its
    own scores read ``bullet_straight`` **1.5**, ``teleport`` **1.5**, ``shaky``
    **1.5** — three distinct degenerate policies pinned at the identical value —
    against ``spinner`` −0.0104 and ``frozen`` 0.0, with the sane reference also
    at 1.5. ⛔ A reward that cannot separate *drive straight through everything*,
    *teleport* and *shake violently* is gameable by construction: every one of
    them is an optimum, so the policy is free to pick any of them.

    ⚠️ ⭐ SCOPE THIS HONESTLY — THE OLD VERDICT WAS NOT A BUG, IT WAS A NARROWER
    QUESTION. ``dead_components`` reports component **LIVENESS**: did every
    weighted term vary enough to influence the ranking. Nothing was constant
    here (``progress`` spans −0.0104 to 1.5), so ``clean`` was *correct by that
    definition*. **The gap is that liveness is not gameability.** A term can vary
    across the panel and still be SATURATED at the top of its clamp for every
    policy that matters — which is exactly what ``progress`` does at ``dt = 0.5``,
    where its ``hi`` bound of 1.5 is reached by anything fast enough. Liveness
    looks at the term; separation looks at the RANKING the term produces.

    ⭐ WHY THE TIE IS TAKEN ON THE **PANEL'S** MAXIMUM AND NOT THE REFERENCE'S.
    MEASURED at both dt values on 2026-09-11, and this is the discriminator that
    makes the check safe rather than a blanket refusal:

    ======================  ==========  =========================  ==========
    spec                    panel max   panel entries at that max  fires?
    ======================  ==========  =========================  ==========
    ``default``  dt 0.1     1.45        ``['bullet_straight']``    no
    ``default``  dt 0.5     1.45        ``['bullet_straight']``    no
    ``default``  dt 0.1 +   0.349503    ``['spinner']``            no
    rich ctx
    ``hackable`` dt 0.1     1.5         ``[bullet, teleport]``     **yes**
    ``hackable`` dt 0.5     1.5         ``[bullet, teleport,       **yes**
                                          shaky]``
    ======================  ==========  =========================  ==========

    A single degenerate tying the reference is already the ``margin`` policy's
    business ("ties are not failures"); this check is about the reward being
    unable to RANK TWO PROBES APART, which no margin can express.

    ⛔ ORDER: the underpowered (``dead``) branch still wins. An audit that cannot
    exercise its own components cannot support a gameability verdict either — but
    ``tied_at_max`` is reported in that branch too, so the fact is never hidden.
    """
    ctx = dict(ctx or {})
    assert_selector_disjoint(ctx)
    ctx.setdefault("dt", spec.dt)

    ref_traj = sane_reference_trajectory(n_steps, spec.dt, device=device)
    ref = float(spec(ref_traj, ctx))

    panel = degenerate_panel(n_steps, spec.dt, device)
    scores = {name: float(spec(traj, ctx)) for name, traj in panel.items()}
    winners = [k for k, v in scores.items() if v > ref + margin]

    # ⭐ THE SEPARATION SET — computed in every branch, used in one.
    panel_max = max(scores.values())
    tied = sorted(k for k, v in scores.items() if abs(v - panel_max) <= tie_atol)

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
                  f"(raw winners under this context: {winners or 'none'}; "
                  f"panel entries tied at the maximum: {tied})")
        return AuditReport(False, reason, scores, ref, winners,
                           inconclusive=True, dead_components=dead,
                           tied_at_max=tied)
    if len(tied) >= 2:
        reason = (f"NOT SEPARABLE: {len(tied)} degenerate policies tie at the "
                  f"panel maximum {panel_max:+.6f} — {tied}. The reward assigns "
                  "them the identical score, so it cannot rank them and a policy "
                  "is free to occupy any of them. Component LIVENESS is not "
                  "gameability: every weighted term varied, and the ranking still "
                  "collapsed (typically a clamped term SATURATED at its `hi` "
                  "bound). This is a hackability finding, not an underpowered "
                  f"panel. (winners over the reference: {winners or 'none'})")
        return AuditReport(True, reason, scores, ref, winners, tied_at_max=tied)
    if winners:
        reason = (f"{len(winners)} degenerate polic(y/ies) score above the sane "
                  f"reference: {winners}. The reward is hackable as weighted.")
    else:
        reason = ("no degenerate policy in the panel beats the sane reference "
                  "and no two tie at the panel maximum (this is necessary, NOT "
                  "sufficient — the panel is fixed and finite, so it bounds the "
                  "claim)")
    return AuditReport(bool(winners), reason, scores, ref, winners,
                       tied_at_max=tied)


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
