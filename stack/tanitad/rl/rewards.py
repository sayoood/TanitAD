"""Rule-based reward components for RL post-training of the refcv3 planner.

⛔ THE ONE RULE THIS MODULE EXISTS TO ENFORCE
---------------------------------------------
**The reward must NEVER be the selector's own score.** Every component here is a
pure function of (a) the candidate trajectory's own geometry, (b) ground-truth
scene facts, and (c) fixed physical constants. None of them may read
``sel_score``, ``sel_score_v3``, ``anchor_logits``, ``goal_gate`` or any other
model-produced ranking. ``tanitad.rl.audit.assert_selector_disjoint`` enforces
this structurally, and ``FORBIDDEN_REWARD_INPUTS`` below is its allow-list
complement. Rationale: DiffusionDriveV2's own named defect is **selector
over-reliance** ("a downstream selector … often less robust than the generator
… prone to failure … particularly in out-of-distribution scenarios"), and our
SEL-1 winner's-curse refusal measured the same defect independently. A reward
built from the selector's score trains the generator to agree with the weakest
component in the stack, and the resulting number would be an echo — the same
family as the nav-echo that scored 1.0000 by reproducing its own input.

WHY THESE ARE PURE TENSOR FUNCTIONS AND NOT taniteval CALLS
-----------------------------------------------------------
``taniteval/taniteval/lead_metrics.py`` (headway / time-gap / per-step gap),
``four_families.py`` (the four metric families) and ``lateral.py`` are the
REFERENCE implementations for the eval-side numbers, and any *reported* metric
must come from them. But ``taniteval`` is a SIBLING of ``stack/`` and reaching
it needs ``PYTHONPATH=<repo>/stack:<repo>/taniteval``; a hard import here would
make the RL library un-importable exactly where it must run (dev box, Colab,
a pod). So the reward components are self-contained and analytically testable,
and the eval-side comparison is done by the audit, not by the training loop.
⚠️ STATED, not hidden: that means a reward component and its taniteval namesake
are two implementations of one idea. ``tests/test_rl_rewards.py`` pins each
component to its ANALYTIC value, which is the thing that keeps them honest.

EVERY COMPONENT DECLARES ITS OWN DEGENERATE POLICY
---------------------------------------------------
A reward you cannot hack is a reward you have not thought about. Each component
below carries ``.degenerate`` — the policy that maximises it while driving
badly. ``audit.py`` runs those policies against the composed reward and FLAGS
when one of them wins. ⛔ ``progress`` alone is the canonical hackable reward
(maximised by driving straight and fast through everything); it is retained
deliberately as the DELIBERATE-REGRESSION arm, and an audit that does not flag
it is a broken audit.

Geometry contract: trajectories are ego-frame waypoints ``[..., S, 2]`` in
METRES at a fixed ``dt`` (default 0.1 s = 10 Hz, matching the corpus cadence and
``four_families.DT_S``), ego starts at the origin heading +x. This is the same
convention ``refc.synth_anchor_pool`` integrates and ``refc_v3`` emits as
``anchor_traj``.

Evidence class of anything computed here: MEASURED (ours; rule-based).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

DT_S = 0.1
EPS = 1e-8

#: ⛔ Keys that may NEVER reach a reward function. Enforced by
#: ``audit.assert_selector_disjoint``; see the module docstring.
FORBIDDEN_REWARD_INPUTS = frozenset({
    "sel_score", "sel_score_v3", "sel_idx", "sel_idx_base", "sel_tele",
    "anchor_logits", "prefinal_logits", "goal_gate", "score", "blended",
    "rank", "traj",          # `traj` is the SELECTED trajectory (post-argmax)
    "traj_base",
})

#: Physical limits for the (a, kappa) unicycle action space. Defaults match
#: `refc.synth_anchor_pool`'s sampling envelope (accel +-3 m/s^2, yaw rate
#: +-0.35 rad/s) widened to the trainer's own `--a-max` / `--kappa-max` scale.
A_MAX_MPS2 = 4.0
KAPPA_MAX_1PM = 0.2
LAT_ACC_MAX_MPS2 = 4.0
JERK_MAX_MPS3 = 8.0


# ---------------------------------------------------------------------------
# Kinematics — one derivation, used by every component
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Kinematics:
    """Per-step kinematics derived from ego-frame waypoints.

    All fields are ``[..., S-k]`` for the appropriate k (differencing loses a
    step each time). ``speed`` is [..., S-1], ``accel``/``yaw_rate``/``kappa``
    are [..., S-2], ``jerk`` is [..., S-3].
    """
    speed: Tensor
    heading: Tensor
    yaw_rate: Tensor
    kappa: Tensor
    accel: Tensor
    jerk: Tensor
    lat_acc: Tensor
    arc_len: Tensor          # [...] total path length
    along: Tensor            # [...] net +x displacement (signed progress)


def kinematics(traj: Tensor, dt: float = DT_S) -> Kinematics:
    """Derive kinematics from waypoints ``[..., S, 2]``.

    ⚠️ Curvature is ``yaw_rate / speed`` and is UNDEFINED at zero speed. We
    clamp the denominator at ``MIN_SPEED`` rather than ``EPS`` so a stationary
    trajectory reads kappa 0 instead of a spurious 1e8 — the same guard
    ``four_families.MIN_DS_MPS`` applies for the same reason (below ~0.5 m/s a
    step carries no reliable heading).
    """
    if traj.shape[-1] != 2:
        raise ValueError(f"expected [..., S, 2] waypoints, got {tuple(traj.shape)}")
    if traj.shape[-2] < 4:
        raise ValueError(f"need >=4 waypoints to derive jerk, got {traj.shape[-2]}")

    d = traj[..., 1:, :] - traj[..., :-1, :]          # [..., S-1, 2]
    step = d.norm(dim=-1)                              # [..., S-1]
    speed = step / dt
    heading = torch.atan2(d[..., 1], d[..., 0])        # [..., S-1]

    dtheta = heading[..., 1:] - heading[..., :-1]
    dtheta = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))   # wrap to +-pi
    yaw_rate = dtheta / dt                             # [..., S-2]

    min_speed = 0.5
    v_mid = speed[..., :-1].clamp_min(min_speed)
    kappa = yaw_rate / v_mid
    accel = (speed[..., 1:] - speed[..., :-1]) / dt    # [..., S-2]
    jerk = (accel[..., 1:] - accel[..., :-1]) / dt     # [..., S-3]
    lat_acc = v_mid.pow(2) * kappa

    return Kinematics(speed=speed, heading=heading, yaw_rate=yaw_rate,
                      kappa=kappa, accel=accel, jerk=jerk, lat_acc=lat_acc,
                      arc_len=step.sum(dim=-1),
                      along=traj[..., -1, 0] - traj[..., 0, 0])


# ---------------------------------------------------------------------------
# The component protocol
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewardComponent:
    """One named, bounded, rule-based reward term.

    ``fn`` maps (traj, ctx) -> ``[...]`` reward, HIGHER IS BETTER, and every
    component is bounded in [lo, hi] so a composition cannot be dominated by one
    unbounded term (the classic reward-hacking amplifier).

    ⛔ ``neutral`` — THE VALUE FOR "THIS CONSTRAINT IS NOT PRESENT IN THE SCENE".
    ⚠️ CORRECTED 2026-08-29 after a peer review of this module. The absence
    semantics were INCONSISTENT and the inconsistency was silent::

        collision      absent -> 0.0  = its BEST  (range [-1, 0])
        headway        absent -> 0.0  = its WORST (range [0, 1])
        gt_similarity  absent -> 0.0  = its WORST (range [0, 1])

    So a window with no obstacle was rewarded while a window with no lead
    vehicle was punished — for the same underlying fact, *nothing to constrain
    against*. On our corpus the second case is common, so the default reward was
    quietly penalising ordinary open road.

    **The rule now: ABSENCE MEANS NO CONSTRAINT, THEREFORE NO PENALTY** — every
    component returns its BEST value when its scene fact is missing, and
    ``audit.report_component_coverage`` flags it as not-fired so the absence is
    visible instead of being read as safety. *(A constant cancels in a
    group-relative advantage, so this does not bias the gradient; it biased the
    audit and any absolute reporting, which is where a wrong number gets
    quoted.)*
    """
    name: str
    fn: Callable[..., Tensor]
    lo: float
    hi: float
    degenerate: str                  # the policy that maximises it while driving badly
    grounded: str                    # "rule-based" | "gt-derived" | "learned"
    hackable_alone: bool
    neutral: float = 0.0             # value when the scene fact is absent

    def __call__(self, traj: Tensor, ctx: dict) -> Tensor:
        r = self.fn(traj, ctx)
        return r.clamp(self.lo, self.hi)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def _progress(traj: Tensor, ctx: dict) -> Tensor:
    """Net along-track displacement, normalised by a reference distance.

    ⛔ THE CANONICAL HACKABLE REWARD. Maximised by driving straight and as fast
    as possible, THROUGH anything in the way. Retained on purpose as the
    deliberate-regression arm.
    """
    ref = float(ctx.get("progress_ref_m", 30.0))
    return kinematics(traj, ctx.get("dt", DT_S)).along / max(ref, EPS)


def _collision(traj: Tensor, ctx: dict) -> Tensor:
    """-1 if the path comes within (ego_r + obs_r) of any obstacle, else 0.

    Obstacles are ``ctx["obstacles"]`` ``[..., K, 2]`` ego-frame centres (a
    static snapshot; a moving-obstacle variant belongs with the agent-track
    join, not here). Absent obstacles -> 0 for every candidate, which is
    HONEST (no information) rather than a free pass disguised as safety:
    ``audit.report_component_coverage`` counts how often this fires, and a
    component that never fires is reported, not silently trusted.
    """
    obs = ctx.get("obstacles")
    if obs is None or obs.numel() == 0:
        return torch.zeros(traj.shape[:-2], device=traj.device, dtype=traj.dtype)
    r = float(ctx.get("ego_radius_m", 1.0)) + float(ctx.get("obs_radius_m", 1.0))
    # traj [..., S, 2] vs obs [..., K, 2] -> pairwise [..., S, K]
    d = (traj.unsqueeze(-2) - obs.unsqueeze(-3)).norm(dim=-1)
    hit = (d < r).any(dim=-1).any(dim=-1)
    return torch.where(hit, -torch.ones_like(hit, dtype=traj.dtype),
                       torch.zeros_like(hit, dtype=traj.dtype))


def _headway(traj: Tensor, ctx: dict) -> Tensor:
    """GRADED, ASYMMETRIC time-gap reward. Peaks at the target gap T*.

    ⛔ REDESIGNED 2026-08-29 AFTER A0 MEASURED THE FIRST VERSION INERT.
    The original was ``min_t (gap_t / v_t) / T*`` clamped to [0, 1] — a
    SATURATING CONSTRAINT. A0 (240 held-out windows, `raw/a0_coverage.json`)
    measured it firing on 28.3 % of windows with a **median spread across the fan
    of 0.0000**: with a lead 40 m ahead at 10 m/s the time gap is already ≈2 s
    = T*, so nearly every candidate pinned at 1.0. A term that is identical for
    every candidate cancels EXACTLY in a group-relative advantage — it was
    present and carrying no signal.

    The shape now ranks on BOTH sides of T* (Master Mind design ruling, as model-
    design owner; do NOT demote this to a pure hard constraint — that would
    discard the distance-keeping signal, and longitudinal error is 88.7 % of our
    historical oracle gap):

        t < T*      steep  — (t/T*)**k, k>1. Tailgating is punished hard and
                             the penalty accelerates as the gap closes.
        T* <= t <= FAR*T*  mild  — linear decline by `dawdle` over the span.
                             Sitting at a 4 s gap is wrong, but nothing like as
                             wrong as following at 0.8 s.
        t > FAR*T*  saturated at the FAR value — beyond genuinely-far there is
                             nothing left to rank.

    Peak is exactly 1.0 at t == T*, so the reward's argmax IS the target gap.

    ⚠️ The hard safety constraint is NOT here. TTC / collision-imminent is a
    VETO applied OUTSIDE the advantage (``ttc_violation``), because a constraint
    and a ranking signal are different objects and fusing them into one term is
    what produced the inert component in the first place.
    """
    lead = ctx.get("lead_path")
    if lead is None:
        # NO LEAD => no headway constraint => no penalty (the `neutral` rule).
        return torch.ones(traj.shape[:-2], device=traj.device, dtype=traj.dtype)
    t_star = float(ctx.get("target_time_gap_s", 2.0))
    k = float(ctx.get("headway_tailgate_exp", 2.0))
    dawdle = float(ctx.get("headway_dawdle_penalty", 0.25))
    far = float(ctx.get("headway_far_mult", 2.0))

    kin = kinematics(traj, ctx.get("dt", DT_S))
    gap = (lead[..., 1:, :] - traj[..., 1:, :]).norm(dim=-1)     # [..., S-1]
    gap = (gap - float(ctx.get("lead_len_m", 4.5))).clamp_min(0.0)
    tg = (gap / kin.speed.clamp_min(0.5)).amin(dim=-1)           # worst step

    ratio = tg / max(t_star, EPS)
    below = ratio.clamp(0.0, 1.0).pow(k)                          # steep
    span = max(far - 1.0, EPS)
    above = 1.0 - dawdle * ((ratio - 1.0).clamp(0.0, span) / span)
    return torch.where(ratio < 1.0, below, above)


def ttc_violation(traj: Tensor, ctx: dict) -> Tensor:
    """⛔ The HARD safety VETO — a constraint, never a ranking term.

    True where a candidate's time-to-collision with the lead drops below
    ``ttc_min_s``. Applied OUTSIDE the group-relative advantage (see
    ``advantage.truncated_inter_anchor_advantage``), exactly like a collision:
    a vetoed candidate is pinned, not merely ranked lower.

    Separating this from ``_headway`` is the point. One term that tried to be
    both a constraint and a ranking signal saturated and became inert; the
    constraint half belongs where constraints go.

    TTC uses the CLOSING speed (how fast the gap shrinks), not the ego speed —
    matching a lead at 30 m/s at a 10 m gap is stable, not imminent.
    """
    lead = ctx.get("lead_path")
    if lead is None:
        return torch.zeros(traj.shape[:-2], device=traj.device, dtype=torch.bool)
    ttc_min = float(ctx.get("ttc_min_s", 1.5))
    dt = float(ctx.get("dt", DT_S))
    gap = (lead[..., 1:, :] - traj[..., 1:, :]).norm(dim=-1)
    gap = (gap - float(ctx.get("lead_len_m", 4.5))).clamp_min(0.0)
    closing = (gap[..., :-1] - gap[..., 1:]) / dt                 # >0 = closing
    ttc = gap[..., :-1] / closing.clamp_min(1e-3)
    imminent = (closing > 0) & (ttc < ttc_min)
    return imminent.any(dim=-1)


def _motion_gate(kin: Kinematics, min_m: float = 1.0) -> Tensor:
    """0 for a path that does not move, ramping to 1 by ``min_m`` metres.

    ⛔ WHY THIS EXISTS — MEASURED 2026-08-29 BY OUR OWN AUDIT. Without it,
    ``feasibility`` and ``comfort`` both read **1.0** for a trajectory that
    stands perfectly still, handing the "frozen" degenerate policy **0.70 of
    free reward** (0.50 + 0.20 at the default weights) for doing nothing. The
    audit caught it on the library's own default weight vector.

    The fix is semantic, not a fudge: *"feasible"* and *"comfortable"* are
    properties OF A RIDE. A path that never moves has no ride to be comfortable
    about, so the honest value is 0, not 1 — the same reasoning that makes an
    absent collision reading 0 rather than "safe".
    """
    return (kin.arc_len / max(min_m, EPS)).clamp(0.0, 1.0)


def _kinematic_feasibility(traj: Tensor, ctx: dict) -> Tensor:
    """1 when the whole path respects the (a, kappa) envelope, decaying to 0.

    This is the component that makes the action space real: a trajectory the
    vehicle cannot execute is worthless however good it looks in L2. Gated on
    actually moving (see ``_motion_gate``).
    """
    kin = kinematics(traj, ctx.get("dt", DT_S))
    a_ex = (kin.accel.abs() / ctx.get("a_max", A_MAX_MPS2) - 1.0).clamp_min(0.0)
    k_ex = (kin.kappa.abs() / ctx.get("kappa_max", KAPPA_MAX_1PM) - 1.0).clamp_min(0.0)
    ex = a_ex.amax(dim=-1) + k_ex.amax(dim=-1)
    return torch.exp(-ex) * _motion_gate(kin, ctx.get("min_motion_m", 1.0))


def _comfort(traj: Tensor, ctx: dict) -> Tensor:
    """1 for a smooth path, decaying with jerk and lateral acceleration.

    Gated on actually moving (see ``_motion_gate``).
    """
    kin = kinematics(traj, ctx.get("dt", DT_S))
    j = (kin.jerk.abs().amax(dim=-1) / ctx.get("jerk_max", JERK_MAX_MPS3)).clamp_min(0.0)
    la = (kin.lat_acc.abs().amax(dim=-1)
          / ctx.get("lat_acc_max", LAT_ACC_MAX_MPS2)).clamp_min(0.0)
    return torch.exp(-(j + la)) * _motion_gate(kin, ctx.get("min_motion_m", 1.0))


def _gt_similarity(traj: Tensor, ctx: dict) -> Tensor:
    """Negative ADE to the logged future, mapped to (0, 1].

    ⚠️ THIS IS THE IMITATION ANCHOR, NOT A CAPABILITY SIGNAL. It exists so RL
    cannot wander off the demonstration manifold (DDv2 keeps an IL regulariser
    for exactly this reason). It is `gt-derived`, so it obeys D-LABEL-GT: it may
    read privileged future state because it is a TRAINING signal, never an
    inference input.
    ⛔ Do NOT read a high value here as "drives well" — it is ADE wearing a
    reward's clothes, and ADE alone is an incomplete result (four families).
    """
    gt = ctx.get("gt_traj")
    if gt is None:
        # NO TARGET => no imitation penalty (the `neutral` rule). Coverage
        # reporting flags it as not-fired so the absence is visible.
        return torch.ones(traj.shape[:-2], device=traj.device, dtype=traj.dtype)
    scale = float(ctx.get("ade_scale_m", 2.0))
    while gt.dim() < traj.dim():
        gt = gt.unsqueeze(-3)
    ade = (traj - gt).norm(dim=-1).mean(dim=-1)
    return torch.exp(-ade / max(scale, EPS))


COMPONENTS: dict[str, RewardComponent] = {
    c.name: c for c in (
        RewardComponent("progress", _progress, -1.0, 1.5,
                        degenerate="straight-line max-speed through obstacles",
                        grounded="rule-based", hackable_alone=True,
                        neutral=0.0),
        RewardComponent("collision", _collision, -1.0, 0.0,
                        degenerate="stand still forever (never collides)",
                        grounded="gt-derived", hackable_alone=True, neutral=0.0),
        RewardComponent("headway", _headway, 0.0, 1.0,
                        degenerate="fall infinitely far behind the lead",
                        grounded="gt-derived", hackable_alone=True, neutral=1.0),
        RewardComponent("feasibility", _kinematic_feasibility, 0.0, 1.0,
                        degenerate="stand still (trivially feasible)",
                        grounded="rule-based", hackable_alone=True, neutral=1.0),
        RewardComponent("comfort", _comfort, 0.0, 1.0,
                        degenerate="stand still (perfectly smooth)",
                        grounded="rule-based", hackable_alone=True, neutral=1.0),
        RewardComponent("gt_similarity", _gt_similarity, 0.0, 1.0,
                        degenerate="none known — but it is ADE, not driving skill",
                        grounded="gt-derived", hackable_alone=False, neutral=1.0),
    )
}

#: The recommended default. ⭐ Note it pairs every "go" term with a "don't"
#: term: progress is checked by collision + headway, and the two stand-still
#: degenerates (feasibility, comfort) are checked by progress. A weight vector
#: that breaks that pairing is what `audit.audit_reward` is for.
#:
#: ⛔ ``gt_similarity`` IS DELIBERATELY ABSENT — REMOVED 2026-08-29 after peer
#: review, and this is the most important line in the file.
#:
#: It was here at weight 0.40 *while* ``PostTrainConfig.w_imitation`` added a
#: separate imitation loss at 1.0, so the imitation signal was counted TWICE.
#: The double count is not the real damage. The real damage is WHERE the second
#: copy sat:
#:
#:   A group-relative advantage is computed ACROSS the candidates of one fan.
#:   A term that says "be closer to the single logged expert path" therefore
#:   assigns its highest advantage to whichever candidate is nearest that path,
#:   and pushes probability mass onto it and OFF every other mode.
#:   ⇒ **Inside the advantage, an imitation term is a FAN-COLLAPSE objective.**
#:
#: That destroys precisely the property the whole method exists to buy: DDv2's
#: raw-fan floor holds at **84.4** where DiffusionDrive's falls to **75.3**
#: (top-1 -> top-10). Collapsing the fan onto the expert would have made our
#: numbers look fine on the selected trajectory while the fan rotted underneath
#: — which is the SAME failure shape as selector over-reliance, arrived at from
#: the other side.
#:
#: ⇒ The imitation anchor belongs OUTSIDE the advantage, as DDv2 has it:
#: ``L = L_RL + lambda * L_IL`` (``config.w_imitation``, applied in
#: ``posttrain.rl_objective``). The component stays in ``COMPONENTS`` because it
#: is a legitimate DIAGNOSTIC and someone may want it deliberately — but
#: ``PostTrainConfig.validate()`` now REFUSES the combination.
DEFAULT_WEIGHTS: dict[str, float] = {
    "progress": 0.30,
    "collision": 1.00,
    "headway": 0.30,
    "feasibility": 0.50,
    "comfort": 0.20,
}

#: ⛔ The DELIBERATE-REGRESSION arm. `audit.audit_reward` MUST flag this. If it
#: passes, the audit is broken — that is the point of shipping it.
HACKABLE_WEIGHTS: dict[str, float] = {"progress": 1.0}


@dataclass
class RewardSpec:
    """A named, weighted composition of components."""
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    dt: float = DT_S

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(COMPONENTS)
        if unknown:
            raise KeyError(f"unknown reward components: {sorted(unknown)}; "
                           f"known = {sorted(COMPONENTS)}")
        if not self.weights:
            raise ValueError("a reward with no components is not a reward")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.weights))

    def per_component(self, traj: Tensor, ctx: dict) -> dict[str, Tensor]:
        ctx = {**ctx, "dt": ctx.get("dt", self.dt)}
        return {n: COMPONENTS[n](traj, ctx) for n in self.names}

    def __call__(self, traj: Tensor, ctx: dict) -> Tensor:
        parts = self.per_component(traj, ctx)
        out = None
        for n, v in parts.items():
            term = self.weights[n] * v
            out = term if out is None else out + term
        return out
