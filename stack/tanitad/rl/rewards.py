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
    """Along-track displacement relative to MAINTAINING THE CURRENT SPEED.

    ⭐ PER-WINDOW REFERENCE, adopted 2026-08-29 (Master Mind decision) after A0
    measured the fixed-reference version SATURATING: median spread **1.5000** =
    exactly its clamp width, mean **1.1478** > 1.0, i.e. a large share of
    candidates pinned at the `hi` bound and ranking was compressed among exactly
    the fastest candidates the reward most needs to order.

    The reference is now ``max(v0 * horizon_s, min_ref_m)``:

    * ``v0`` is the window's OWN current ego speed — inference-admissible ego
      state at labels-time.
      ⛔ **It is deliberately NOT derived from the fan and NOT from the logged
      expert.** A fan-derived reference would be tuning on what we score (the
      probe-that-tunes-on-its-own-data class); an expert-derived one would smuggle
      the imitation signal back into a term we removed it from.
    * ``min_ref_m`` floors near-stopped windows so the normalisation cannot
      explode as v0 -> 0.

    Semantics, and this is why the choice is right rather than merely convenient:
    **1.0 means "kept the current speed"**, >1 accelerating, <1 decelerating. The
    zero point is a driving fact, not an arbitrary metre count — a fixed 30 m
    reference silently bakes in a speed prior that is wrong at both ends of the
    speed distribution.

    ⚠️ Falls back to a FIXED reference when ``v0`` is absent from the context.
    That is a degraded mode: it restores the saturation A0 measured, so callers
    should supply ``v0``.

    ⭐⭐ LEAD CAP, ADDED 2026-09-06 (`H-RL-PROGRESS-LEADCAP-1`, pre-registered in
    `.../Research/2026-09-06-refcv4b-rl-repair/PREREG.md` and committed BEFORE
    this change). ⛔ THE DEFECT IT REPAIRS IS MEASURED, NOT ARGUED: on 6,089
    RL-fit windows / 73 episodes the per-term weighted mean gap
    ``hold_v0 - human`` for THIS TERM **FLIPS SIGN** as the scene tightens --
    -0.005600 over all windows, **+0.003593** at a <= 3.0 s human time gap,
    **+0.006368** at <= 2.0 s, **+0.008251** at <= 1.5 s. In close following the
    human slows for the lead and a constant-velocity path does not, so an
    UNCAPPED along-track reference **PAYS THE TRIVIAL PATH FOR NOT SLOWING
    DOWN**, precisely where the safety question is live. `progress` is the only
    one of the three ranking terms that changes sign.

    The numerator is therefore capped at what is ACHIEVABLE given the lead's own
    recorded position at the horizon end (:func:`achievable_along_ref`), behind
    ``ctx["progress_lead_cap"]`` (default ``True``). ⛔ It is **inert without a
    lead**, so the no-lead population is unchanged BY CONSTRUCTION rather than by
    luck, and ``progress_lead_cap=False`` restores the exact pre-repair term for
    reproducing any banked number.

    ⛔ WHAT IT IS NOT. DD-v2's ``EP = progress_i / max(progress_GT, progress_i)``
    caps at the HUMAN'S OWN progress. That is the human's SHAPE inside a RANK
    term and is inadmissible here (`E-DDA-3c` §6.1: the bar may use the human's
    SCORE, never their SHAPE). This cap comes from the LEAD AGENT'S recorded
    track, which §6.1 already admits into the RANK channel.
    """
    kin = kinematics(traj, ctx.get("dt", DT_S))
    v0 = ctx.get("v0")
    if v0 is None:
        ref = float(ctx.get("progress_ref_m", 30.0))
        return _apply_lead_cap(kin.along, max(ref, EPS), ctx)
    horizon_s = (traj.shape[-2] - 1) * float(ctx.get("dt", DT_S))
    min_ref = float(ctx.get("progress_min_ref_m", 5.0))
    if isinstance(v0, Tensor):
        ref = (v0 * horizon_s).clamp_min(min_ref)
        while ref.dim() < kin.along.dim():
            ref = ref.unsqueeze(-1)
    else:
        ref = max(float(v0) * horizon_s, min_ref)
    return _apply_lead_cap(kin.along, ref, ctx)


#: The legal values of ``ctx["progress_lead_cap"]``.
#:   "achievable" (default, and ``True``) -- the PRE-REGISTERED form:
#:        ref_ach = clamp(min(ref_free, lead_x[-1] - standoff), min_ref)
#:   "lead"  -- the SENSITIVITY variant: cap at the lead-imposed bound ONLY, so a
#:        candidate may still earn progress above ``v0 * H`` when the lead is far.
#:        Kept because the pre-registered form carries a SECOND cap (at ref_free)
#:        that was not the stated intent, and a difference that is priced is worth
#:        more than one that is hidden.
#:   "off" (and ``False``) -- the exact pre-repair term, bit-identical.
PROGRESS_LEAD_CAP_MODES = ("achievable", "lead", "off")


def _progress_cap_mode(ctx: dict) -> str:
    m = ctx.get("progress_lead_cap", True)
    if m is True:
        return "achievable"
    if m is False or m is None:
        return "off"
    m = str(m)
    if m not in PROGRESS_LEAD_CAP_MODES:
        raise ValueError(
            f"progress_lead_cap must be one of {PROGRESS_LEAD_CAP_MODES} "
            f"(or a bool), got {m!r}")
    return m


def achievable_along_ref(ctx: dict, ref_free):
    """The ACHIEVABLE along-track reference given the LEAD'S OWN RECORDED POSITION.

    Returns ``None`` when the cap is inert -- no lead, or the mode is ``off`` --
    in which case ``_progress`` takes the byte-identical pre-repair path.

    ⛔⛔ COORDINATE DISCIPLINE, AND IT IS THE REASON THIS FUNCTION LOOKS THE WAY
    IT DOES. Every quantity here is a **POSITION QUERY** on the lead's recorded
    track (``lead_path[..., -1, 0]`` -- the lead's x at the END of the horizon)
    or a **scene scalar measured at t0** (``v0``, ``lead_len_m``,
    ``target_time_gap_s``). ⛔ **NO CLOSING RATE APPEARS ANYWHERE.** `M84`
    MEASURED that refcv4b-lineage latents decode the lead's POSITION (paired vs
    pixels +0.4145 [+0.2018, +0.6120], constant control exactly +0.000000) while
    its CLOSING RATE is a clean null on every arm (+0.0061, and an explicit
    temporal difference recovers nothing). A RANK term whose ORDERING rested on
    that rate would inherit a coordinate the latent does not carry -- which is
    exactly why TTC stays a VETO and is not graded into the ranking.
    ⇒ A consequence that is TESTED, not asserted: perturbing the lead's
    INTERMEDIATE track samples while holding its endpoint fixed must leave this
    reference EXACTLY unchanged (``test_rl_progress_leadcap.py``). A rate-based
    reference could not pass that test.

    ⛔ THE CAP IS A PROPERTY OF THE SCENE, NOT OF THE CANDIDATE -- identical for
    every candidate in a window -- so it cannot rank on anything the reward is
    not allowed to see. And it uses NO NEW CONSTANT: ``lead_len_m`` and
    ``target_time_gap_s`` are the same two ``_headway`` already consumes and
    ``THRESHOLD_CALIBRATION`` already carries. A standoff with a free knob in it
    could have been tuned until the answer came out right; this one cannot.
    """
    if _progress_cap_mode(ctx) == "off":
        return None
    lead = ctx.get("lead_path")
    if lead is None:
        return None                      # ABSENCE MEANS NO CONSTRAINT (the neutral rule)
    lead_end_x = lead[..., -1, 0]                                  # POSITION at t = H
    v0 = ctx.get("v0")
    t_star = float(ctx.get("target_time_gap_s", 2.0))
    lead_len = float(ctx.get("lead_len_m", 4.5))
    if isinstance(v0, Tensor):
        standoff = lead_len + t_star * v0
        while standoff.dim() < lead_end_x.dim():
            standoff = standoff.unsqueeze(-1)
    else:
        standoff = lead_len + t_star * (0.0 if v0 is None else float(v0))
    ref_lead = lead_end_x - standoff
    min_ref = float(ctx.get("progress_min_ref_m", 5.0))
    if _progress_cap_mode(ctx) == "achievable":
        ref_free_t = (ref_free if isinstance(ref_free, Tensor)
                      else torch.as_tensor(float(ref_free), dtype=ref_lead.dtype,
                                           device=ref_lead.device))
        ref_lead = torch.minimum(ref_free_t, ref_lead)
    return ref_lead.clamp_min(min_ref)


def _apply_lead_cap(along: Tensor, ref_free, ctx: dict) -> Tensor:
    """``along / ref_free``, with the numerator CAPPED at the achievable bound.

    ⭐ IT REMOVES A PAYMENT; IT DOES NOT ADD A PENALTY. Exceeding the achievable
    bound earns nothing further, and the floor is the cap, never a negative
    value: going too fast is ALREADY punished by ``headway`` (MEASURED weighted
    mean gap -0.015899 at a <= 2.0 s human time gap) and by the collision and TTC
    channels, and a second penalty for one fact is double-dipping.
    """
    ref_ach = achievable_along_ref(ctx, ref_free)
    if ref_ach is None:
        return along / ref_free
    return torch.minimum(along, ref_ach) / ref_free


def segment_point_distance(p0: Tensor, p1: Tensor, q: Tensor) -> Tensor:
    """Min distance from each point ``q`` to each SEGMENT ``p0 -> p1``.

    ``p0``, ``p1`` ``[..., S, 2]``; ``q`` ``[..., K, 2]``; returns ``[..., S, K]``.

    ⛔ **This exists because the point test has a hole with a SIZE.** Sampled
    waypoints are ``v*dt`` apart; with a combined radius ``r`` an obstacle sitting
    more than ``r`` from BOTH endpoints but inside the segment is invisible to a
    point test, so the undetected corridor per segment is ``v*dt - 2r`` — metres
    on our grid, not a rounding concern. A finer time grid costs compute forever
    and still leaves a smaller hole; the segment test closes it exactly.

    A degenerate (zero-length) segment — a stopped ego — falls back to the point
    test by construction: the numerator is then exactly 0, so ``t = 0`` and the
    closest point is ``p0``. The clamp of the denominator only avoids 0/0.
    """
    d = p1 - p0                                       # [..., S, 2]
    qq = q.unsqueeze(-3)                              # [..., 1, K, 2]
    p0e = p0.unsqueeze(-2)                            # [..., S, 1, 2]
    de = d.unsqueeze(-2)                              # [..., S, 1, 2]
    denom = (de * de).sum(-1).clamp_min(1e-12)        # [..., S, 1]
    t = (((qq - p0e) * de).sum(-1) / denom).clamp(0.0, 1.0).unsqueeze(-1)
    return (qq - (p0e + t * de)).norm(dim=-1)         # [..., S, K]


def _swept_hit(path: Tensor, obs: Tensor, r: float) -> Tensor:
    """``[...]`` bool: does the swept path come within ``r`` of any obstacle?

    ORs the segment test with the point test. The segment test already covers
    every waypoint (each is an endpoint of some segment, and ``t`` includes 0 and
    1), so the OR is a belt-and-braces guarantee that the swept form can NEVER
    detect LESS than the form it replaces — which is what makes the banked
    numbers a lower bound rather than an incomparable measurement. With fewer
    than two waypoints there is no segment and the point test is all there is.
    """
    pt = (path.unsqueeze(-2) - obs.unsqueeze(-3)).norm(dim=-1) < r      # [...,S,K]
    hit = pt.any(dim=-1).any(dim=-1)
    if path.shape[-2] >= 2:
        seg = segment_point_distance(path[..., :-1, :], path[..., 1:, :], obs) < r
        hit = hit | seg.any(dim=-1).any(dim=-1)
    return hit


def _collision(traj: Tensor, ctx: dict) -> Tensor:
    """-1 if the path comes within (ego_r + obs_r) of any obstacle, else 0.

    Two obstacle models: a STATIC set ``ctx["obstacles"]`` (every step against
    every obstacle), or — when no static set is given — a MOVING
    ``ctx["lead_path"]`` checked TIME-ALIGNED per step (see the branch below).

    Obstacles are ``ctx["obstacles"]`` ``[..., K, 2]`` ego-frame centres (a
    static snapshot; a moving-obstacle variant belongs with the agent-track
    join, not here). Absent obstacles -> 0 for every candidate, which is
    HONEST (no information) rather than a free pass disguised as safety:
    ``audit.report_component_coverage`` counts how often this fires, and a
    component that never fires is reported, not silently trusted.
    """
    obs = ctx.get("obstacles")
    r = float(ctx.get("ego_radius_m", 1.0)) + float(ctx.get("obs_radius_m", 1.0))
    if obs is None or obs.numel() == 0:
        # ⭐ MOVING-LEAD CONTACT (2026-09-05, REF-C RL-readiness WP). When no
        # static obstacle set is supplied but a `lead_path [..., S, 2]` is, contact
        # is TIME-ALIGNED: step s of the candidate against step s of the lead —
        # the per-step convention `_headway` and `ttc_violation` already use.
        # WHY: holding the lead STATIC at its first sample turns every competent
        # follower into a "collision" — the ego reaches the lead's t0 position
        # after one time-gap, so any human path with a time gap shorter than the
        # horizon is flagged (the H-RL-THRESH-1 class: a safety term that fires
        # on the demonstration). Measured on the refcv3 RL-fit clips by
        # `rl_refcv3_min.py --mode humanflag`. Step 0 is skipped (the ego is at
        # its own origin; the lead is ahead by construction).
        lead = ctx.get("lead_path")
        if lead is None:
            return torch.zeros(traj.shape[:-2], device=traj.device, dtype=traj.dtype)
        #: ⭐ SWEPT, and swept in the RELATIVE frame so the time alignment
        #: survives: contact is |lead_s - traj_s| < r, i.e. the relative path
        #: entering a disc of radius r about the ORIGIN. Sweeping the relative
        #: segment therefore accounts for the lead's own motion over the step;
        #: sweeping the ego path against a STATIC lead would re-introduce the
        #: H-RL-THRESH-1 failure the docstring above exists to prevent.
        rel = lead[..., 1:, :] - traj[..., 1:, :]                   # [..., S-1, 2]
        origin = torch.zeros_like(rel[..., :1, :])                  # [..., 1, 2]
        hit = _swept_hit(rel, origin, r)
        return torch.where(hit, -torch.ones_like(hit, dtype=traj.dtype),
                           torch.zeros_like(hit, dtype=traj.dtype))
    #: traj [..., S, 2] vs obs [..., K, 2]. ⭐ SWEPT: the point test missed any
    #: obstacle more than r from both endpoints of a segment it sits inside.
    hit = _swept_hit(traj, obs, r)
    return torch.where(hit, -torch.ones_like(hit, dtype=traj.dtype),
                       torch.zeros_like(hit, dtype=traj.dtype))


def _reduce_time_gap(tg_steps: Tensor, ctx: dict) -> Tensor:
    """Reduce the per-step time gaps to ONE number. ``ctx["headway_reduce"]``.

    ⛔⛔ WHY THIS IS A KNOB AND NOT A CONSTANT — `H-RL-HEADWAY-QUANTILE-1`,
    pre-registered in `.../2026-09-06-refcv4b-rl-repair/PREREG_HEADWAY.md` and
    committed BEFORE this change.

    ``"min"`` (the DEFAULT, and BIT-IDENTICAL to the pre-2026-09-06 term) is
    ``amin`` — a **MIN-OVER-N ORDER STATISTIC** over the horizon's steps. A
    minimum fires on the single worst step, so the term is **RARE AND LARGE**:
    silent in most windows and very loud in a few. MEASURED 2026-09-06: with
    `progress` repaired, `headway` carries essentially the whole `hold_v0 −
    human` gap at the conflict rungs (**−0.015899 at a ≤ 2.0 s human time gap,
    142× `progress`'s +0.000112**) while the RATE still reads 0.5546 — i.e. the
    rate/mean divergence with one term left holding it.
    ⭐ MEASURED in a unit test rather than argued
    (`test_rl_progress_leadcap.py::test_headway_is_blind_to_this_perturbation…`):
    moving three of five lead samples by up to 5 m leaves this term
    **bit-identical**, because the worst step is elsewhere — while a quantile
    over the same per-step gaps does move.
    ⭐ Same family as `fan_floor@k` read alone: a min/max-over-N quantity
    answering a narrower question than the one it is quoted for.

    ``"q<Q>"`` takes the Q-quantile instead, e.g. ``"q0.25"``. ⛔ Every step is a
    POSITION query on the lead's recorded track; no closing rate enters the
    ranking, so the coordinate discipline of `M84` is unchanged and TTC stays a
    VETO.
    """
    mode = str(ctx.get("headway_reduce", "min"))
    if mode == "min":
        return tg_steps.amin(dim=-1)
    if mode.startswith("q"):
        try:
            q = float(mode[1:])
        except ValueError:
            raise ValueError(f"headway_reduce {mode!r}: expected 'q<float>'")
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"headway_reduce quantile must be in [0, 1], got {q}")
        return torch.quantile(tg_steps.to(torch.float32), q, dim=-1).to(tg_steps.dtype)
    raise ValueError(
        f"headway_reduce must be 'min' or 'q<float>', got {mode!r}")


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
    tg = _reduce_time_gap(gap / kin.speed.clamp_min(0.5), ctx)

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


def _proximity(traj: Tensor, ctx: dict) -> Tensor:
    """⭐ A BARRIER on closeness to obstacles — NOT a distance-maximiser.

    ``0`` at or beyond ``d_safe``; falls steeply to ``-1`` at contact::

        r(d) = 0                        for d >= d_safe
        r(d) = -(1 - d/d_safe) ** p     for d <  d_safe

    ⛔ THE SHAPE IS THE POINT, and it is a design constraint from the Master
    Mind (2026-08-29): **penalise below a threshold, go FLAT above it.** An
    unbounded "further from obstacles is better" reward buys the timid-driving
    pathology — hugging empty space, refusing gaps a competent driver takes —
    and would be the saturating-headway mistake in a new costume: a term whose
    optimum sits OUTSIDE the driving envelope. A barrier constrains bad
    candidates and leaves good ones **unranked by it**, which is exactly what we
    want on the ~50 % of windows with no obstacle in the corridor.

    ⚠️ WHY THIS EXISTS (measured, not speculative). The sweep decomposition
    showed ``collision`` is BINARY per candidate and contributes ~0 to ΔR1 at
    every anchor strength, while ``feasibility`` — continuous, with headroom on
    a raw fan — supplied **87 %** of the reward gain and did so by making the fan
    blander (+44 % ADE drift). The gradient followed the EASIEST term, not the
    heaviest. A graded barrier gives the group-relative advantage something
    continuous to rank in the scene-grounded direction.

    ⚠️ It CANNOT be gamed upward: its maximum is 0, reached by any candidate
    that simply stays ``d_safe`` away. There is no reward for going further, so
    the degenerate "flee all obstacles" policy gains nothing over a competent one.
    """
    obs = ctx.get("obstacles")
    if obs is None or obs.numel() == 0:
        return torch.zeros(traj.shape[:-2], device=traj.device, dtype=traj.dtype)
    d_safe = float(ctx.get("proximity_safe_m", 5.0))
    p = float(ctx.get("proximity_exp", 2.0))
    clr = clearance(traj, ctx)                       # ⭐ the ONE definition
    deficit = (1.0 - clr / max(d_safe, EPS)).clamp(0.0, 1.0)
    return -deficit.pow(p)


def clearance(traj: Tensor, ctx: dict) -> Tensor:
    """Worst-case surface-to-surface clearance of a path, in metres. Pure.

    ``traj [..., S, 2]`` against ``ctx["obstacles"] [..., K, 2]`` -> ``[...]``,
    the minimum over BOTH steps and obstacles, floored at 0 (a path through an
    obstacle reads 0, not negative -- penetration depth is the collision term's
    business, not this one's).

    ⚠️ SHAPE CONTRACT, and it is easy to get wrong: the obstacle tensor's leading
    dims must broadcast against ``traj``'s WITHOUT the ``S`` axis. For the usual
    fan ``[B, N, S, 2]`` that means ``[B, 1, K, 2]`` -- **not** ``[B, K, 2]``,
    which carries no axis for the N candidates. ``[B, K, 2]`` happens to broadcast
    correctly when ``B == 1`` and raises a bare ``RuntimeError`` when it does not,
    so a caller can pass the wrong shape, see it work on a single-window smoke
    test, and have it fail only once the batch grows.

    ⭐ EXTRACTED so the quantity has ONE definition. ``_proximity`` shapes it into
    a barrier and ``R5`` in the pilot readout thresholds it; before this they each
    recomputed it, which is how two "clearances" drift apart and a report compares
    a barrier's notion of close with a readout's.

    ⚠️ Returns ``nan`` when there are no obstacles -- NOT 0 and NOT +inf. A window
    with nothing to be close to is UNDEFINED for this quantity, and the caller must
    decide whether to drop it. Scoring absence as "clear" is how a thin obstacle
    join silently reports safety (the same family as TRAIN-C2, where a median
    pooled over windows where the term is undefined became a different
    measurement, and as the E-DETECT-1 all-zero floor).
    """
    obs = ctx.get("obstacles")
    if obs is None or obs.numel() == 0:
        return torch.full(traj.shape[:-2], float("nan"),
                          device=traj.device, dtype=traj.dtype)
    r = float(ctx.get("ego_radius_m", 1.0)) + float(ctx.get("obs_radius_m", 1.0))
    d = (traj.unsqueeze(-2) - obs.unsqueeze(-3)).norm(dim=-1)      # [..., S, K]
    return (d - r).clamp_min(0.0).amin(dim=-1).amin(dim=-1)


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
                        degenerate="straight-line max-speed through obstacles "
                                   "(now: 1.5x the current speed; with the lead "
                                   "cap on, no further payment above the "
                                   "lead-achievable bound)",
                        grounded="rule-based", hackable_alone=True,
                        neutral=0.0),
        RewardComponent("proximity", _proximity, -1.0, 0.0,
                        degenerate="none — a BARRIER caps at 0, so fleeing "
                                   "obstacles gains nothing over competence",
                        grounded="gt-derived", hackable_alone=False, neutral=0.0),
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


#: ⭐ EVERY THRESHOLD CONSTANT CARRIES ITS CALIBRATION AGAINST THE
#: DEMONSTRATION DISTRIBUTION — or the test suite refuses the module.
#:
#: ⛔ THE FAILURE THIS PREVENTS. ``proximity_safe_m = 5.0`` was a round number
#: with no derivation. MEASURED 2026-08-29: it flags the HUMAN DRIVER'S OWN
#: FUTURE on 34.9 % of windows with an in-lane lead (45.1 % on the wider join
#: gate). The trust region pulls the policy toward the demonstration
#: distribution while the barrier pushes it away, with the threshold set where
#: the human lives — opposed by construction, and the reason a whole RL campaign
#: measured nothing. No test could have caught it, because nothing required a
#: constant to be justified.
#:
#: ⚠️ WHAT THIS REGISTRY IS AND IS NOT. It is NOT a re-measurement — the rates
#: below come from a probe over a real corpus (``threshold_sweep.py``), which CI
#: cannot run. It is a COVERAGE CONTRACT: every threshold-like constant must
#: appear here with a measured rate and the artifact that produced it, so a NEW
#: constant fails the suite until somebody measures it. The numbers are
#: evidence; the test enforces that evidence exists.
#:
#: ``flags_human_frac`` = fraction of DEMONSTRATION windows the constant flags,
#: over the population meeting that constant's OWN applicability gate — never a
#: subset selected by the outcome (TRAIN-C8).
#:
#: ⭐ READ THE RATE AS A DESIGN STATEMENT: ~0.05–0.15 is a threshold doing its
#: job (real driving contains real risk). Above ~0.25 the constant is either
#: mis-calibrated or asserting that competent human driving is unsafe — which is
#: a claim to defend explicitly, not to inherit from a default.
THRESHOLD_CALIBRATION: dict[str, dict] = {
    "proximity_safe_m": {
        "value": 5.0, "flags_human_frac": 0.349, "n": 43,
        "verdict": "MISCALIBRATED",
        "note": "flags a third of competent human driving; superseding value "
                "pre-registered in PREREG_D_SAFE_CAL.md (2.0 m, ~0.155). "
                "45.1 % on the wider join gate — quote neither without its gate.",
    },
    "target_time_gap_s": {
        "value": 2.0, "flags_human_frac": 0.118, "n": 34,
        "verdict": "REVIEW",
        "note": "mis-set in the OPPOSITE direction: the human median gap is "
                "2.992 s, so T* rewards following CLOSER than humans drive. A "
                "graded peak, not a barrier, so it shapes rather than vetoes. "
                "P4-12 — needs its own both-outcomes prereg.",
    },
    "ttc_min_s": {
        "value": 1.5, "flags_human_frac": 0.059, "n": 34, "verdict": "OK",
        "note": "hard veto; a low rate is correct for one.",
    },
    "kappa_max_1pm": {
        "value": 0.2, "flags_human_frac": 0.042, "n": 120, "verdict": "OK",
        "note": "feasibility envelope, curvature.",
    },
    "jerk_max_mps3": {
        "value": 8.0, "flags_human_frac": 0.033, "n": 120, "verdict": "OK",
        "note": "identical to pseudosim COMFORT_LIMITS jerk_max_mps3.",
    },
    "a_max_mps2": {
        "value": 4.0, "flags_human_frac": 0.008, "n": 120, "verdict": "OK",
        "note": "feasibility envelope, longitudinal acceleration.",
    },
    "lat_acc_max_mps2": {
        "value": 4.0, "flags_human_frac": 0.000, "n": 120, "verdict": "OK",
        "note": "never flags the human on this corpus — non-binding, not wrong.",
    },
}

#: The measurement behind every row above. ⛔ Do not edit a rate without
#: re-running this and updating the artifact path in the same change.
THRESHOLD_CALIBRATION_SOURCE = (
    "TanitAD Research Lab/Architecture & Inference/Research/"
    "2026-08-29-rl-posttrain-library/code/threshold_sweep.py -> "
    "raw/p_rc21_rerun/threshold_sweep.json (MEASURED 2026-08-29, ours, "
    "NON-PARITY pilot corpus, T0)"
)
