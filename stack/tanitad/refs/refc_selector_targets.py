"""refcv5 WP-7 / ``E-DDA-2b`` — the RULE-BASED, ENVIRONMENT-GROUNDED targets
the sub-metric selector is trained against.

## The one rule this whole file exists to enforce

⛔ **The ego's logged future never enters an environment target.** Five of the
six heads are supervised by predicates evaluated against the **replayed scene**
(``obstacle.offline`` agents, the lead's own track, the candidate's own
geometry, the nav command). The sixth, ``tactical``, is derived from an
ego-future label and is **QUARANTINED**: flagged in the provenance dict,
reported separately, and OFF in the score composition at
``SelectorConfig`` defaults.

**Why this is not paranoia.** ``refc_rescorer.py`` (REF-C v1.2) was trained
toward *the fan's own GT-distance winner* and was refused as SEL-1's winner's
curse — a selector trained on "which candidate is nearest the human" is an ADE
head wearing a safety head's clothes, and it cannot be better than the fan's
oracle by construction. The same trap has a second door: a *target* that quietly
reads the recorded future (a speed profile, a route, a banded distance to the
logged path) is the imitation loss again, with variance. :func:`
assert_environment_only` is the door lock, and :data:`FORBIDDEN_TARGET_INPUTS`
is its allow-list — deliberately the same shape as
``rl.rewards.FORBIDDEN_REWARD_INPUTS``, which exists for the same reason one
layer down.

## Everything reusable is REUSED

| predicate | comes from | not re-implemented because |
|---|---|---|
| contact vs a moving lead / static obstacles | ``rl.rewards._collision`` | the time-aligned moving-lead fix (``D-RL-READY-1`` #3) lives there; a second copy would inherit the *static-lead* defect that flags every competent follower |
| graded time-gap | ``rl.rewards._headway`` | its asymmetric shape is the 2026-08-29 redesign after A0 measured the first version INERT (median fan spread **0.0000**) |
| imminent-TTC veto | ``rl.rewards.ttc_violation`` | closing-speed TTC, not ego-speed TTC |
| along-track progress vs ``v0`` | ``rl.rewards._progress`` | its per-window reference is a Master Mind decision after the fixed-reference version SATURATED at its clamp |
| jerk / lat-acc comfort, (a, kappa) feasibility | ``rl.rewards._comfort``, ``_kinematic_feasibility`` | including ``_motion_gate``, without which "stand perfectly still" collects 0.70 of free reward |
| all per-step kinematics | ``rl.rewards.kinematics`` | one derivation, one curvature guard (``MIN_SPEED`` 0.5 m/s) |
| the ``(lat, lon)`` tactical cell | ``refs.refc_tactical.factor_from_kinematics`` | it reproduces the v1 AND v2 labeler gates exactly; a hand-written gate is how two label definitions drift |
| the nav enum | ``refs.refc.NAV_COMMANDS`` | ``nav_compliance.py`` pins its own copy against this one; a third spelling is a third thing to drift |

⚠️ **Two things could NOT simply be called and are re-derived here, marked as
such at their definitions**, because the existing functions have the wrong
*interface*, not the wrong *maths*:

1. **Kamm / friction load** — ``instruments.flyability.friction_load`` takes
   ``controls [N, 2]`` (the ANCHOR bank) and ``v0 [B]``, and *re-integrates*
   constant controls into a rollout. Our candidates are already-rolled
   ``[B, N, S, 2]`` paths with per-step varying load, so that entry point does
   not fit. :func:`kamm_load` computes the **same quantity** —
   ``sqrt(a_lon^2 + a_lat^2) / G`` — from ``rewards.kinematics``' realised
   ``accel`` and ``lat_acc``, and imports ``flyability.G`` so the gravity
   constant has exactly one home. ``models.kinematic.kamm_circle_violation``
   also does not fit: it takes ``(accel, STEER)`` and returns a **scalar mean
   over the whole batch**, not a per-candidate value.
2. **Nav compliance** — ``taniteval.nav_compliance.complies`` /
   ``terminal_heading`` / ``commanded_side`` are **numpy** and live in the other
   package (``tanitad`` must not depend on ``taniteval``; the four existing
   cross-package imports are all lazy, inside functions). :func:`
   compliance_target` is a torch mirror whose **definition of record is that
   file**, constant for constant.

## Shapes, stated once because a broadcast error here is silent

Candidates are ``[B, N, S, 2]`` ego-frame waypoints. Per-window scene facts are
given in their **natural** per-window shape and this module inserts the
candidate axis:

* ``lead_path``   ``[B, S, 2]``  -> ``[B, 1, S, 2]``
* ``obstacles``   ``[B, K, 2]``  -> ``[B, 1, K, 2]``
* ``v0``          ``[B]``        (broadcast inside ``_progress``)

⛔ Passing an already-unsqueezed tensor is REFUSED rather than guessed. The
``obstacles`` path in ``_collision`` does ``traj.unsqueeze(-2) -
obs.unsqueeze(-3)``; with a rank-3 ``obs`` against a rank-4 ``traj`` the ranks
align *wrongly* and produce a plausible-looking tensor of the wrong meaning.

## Masks: absence is reported, never silently dropped

Every target comes with a boolean mask. ``False`` means **this predicate is
undefined for this candidate in this window** — comfort on a candidate that
does not move, compliance on a window whose nav command is ``follow`` /
``straight``. ``CLAUDE.md``: *"say so per family with the reason and the n,
rather than silently dropping it"*, which is why
:func:`selector_targets` returns a ``coverage`` block with a count per head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from tanitad.instruments.flyability import G
from tanitad.refs import refc_tactical as _tac
from tanitad.refs.refc import NAV_COMMANDS
from tanitad.refs.refc_selector import ENV_HEADS, GT_DERIVED_HEADS, HEAD_NAMES
from tanitad.rl import rewards as R

__all__ = [
    "FORBIDDEN_TARGET_INPUTS", "NAV_SIDE", "TargetConfig",
    "assert_environment_only", "broadcast_scene", "kamm_load",
    "nc_target", "ttc_target", "progress_target", "comfort_target",
    "compliance_target", "tactical_target", "selector_targets",
]


# ---------------------------------------------------------------------------
# The door lock
# ---------------------------------------------------------------------------
#: ⛔ Keys that may NEVER reach an ENVIRONMENT target. Same shape and same
#: purpose as ``rl.rewards.FORBIDDEN_REWARD_INPUTS`` one layer down.
#:
#: ``gt_traj`` / ``gt_future`` — the ego's logged future. Any target reading it
#: is the imitation loss and re-creates the SEL-1 winner's curse.
#: ``sel_*`` / ``anchor_logits`` / ``score`` — a model-produced RANKING. A
#: selector trained toward the generator's own confidence learns the
#: generator's ranking, which is the ``D-REFC-DDAUDIT-3`` defect laundered
#: through a second network.
#: ``nav_true`` — the ORACLE nav label. ``nav_cmd`` (the command the model is
#: told at inference) is admissible for the compliance head; the oracle label
#: is not, because that head's whole read is "does the plan obey what it was
#: TOLD", not "does it obey what a future-reading oracle knows".
FORBIDDEN_TARGET_INPUTS: frozenset[str] = frozenset({
    "gt_traj", "gt_future", "gt_path", "future_poses", "expert_traj",
    "sel_score", "sel_score_v3", "sel_idx", "sel_idx_base", "sel_tele",
    "anchor_logits", "refined_logits", "prefinal_logits", "score", "rank",
    "nav_true",
})

#: ``+1`` left, ``-1`` right, ``0`` follow/straight (= NOT informative).
#: Mirrors ``taniteval.nav_compliance.commanded_side``; the enum itself is
#: imported from ``refc.NAV_COMMANDS`` so there is one spelling.
NAV_SIDE: dict[int, int] = {
    NAV_COMMANDS.index("follow"): 0,
    NAV_COMMANDS.index("left"): 1,
    NAV_COMMANDS.index("right"): -1,
    NAV_COMMANDS.index("straight"): 0,
}


def assert_environment_only(ctx: dict) -> None:
    """⛔ Refuse a context that carries the answer. Call it before every
    environment target; :func:`selector_targets` calls it for you.

    Raises ``KeyError`` naming the offending keys, rather than warning — a
    warning in a training loop is a line in a log nobody re-reads, which is
    exactly how an orthogonality instrument sat unmerged for 10 days.
    """
    bad = sorted(set(ctx) & FORBIDDEN_TARGET_INPUTS)
    if bad:
        raise KeyError(
            f"environment targets may not read {bad}. These carry either the "
            f"ego's logged future (the imitation signal) or a model-produced "
            f"ranking (the D-REFC-DDAUDIT-3 defect laundered through a second "
            f"network). The tactical head is the ONE quarantined exception and "
            f"takes its label through `tactical_target`'s explicit arguments, "
            f"never through this ctx.")


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
def broadcast_scene(ctx: dict) -> dict:
    """Insert the candidate axis into per-window scene facts.

    ``lead_path [B, S, 2] -> [B, 1, S, 2]``; ``obstacles [B, K, 2] ->
    [B, 1, K, 2]``. Anything already carrying a candidate axis is passed
    through untouched **only if it is unambiguous** (rank 4); an ambiguous rank
    raises rather than being guessed.
    """
    out = dict(ctx)
    lead = out.get("lead_path")
    if isinstance(lead, Tensor):
        if lead.dim() == 3:
            out["lead_path"] = lead.unsqueeze(1)
        elif lead.dim() != 4:
            raise ValueError(
                f"lead_path must be [B, S, 2] or [B, 1, S, 2], got "
                f"{tuple(lead.shape)}")
    obs = out.get("obstacles")
    if isinstance(obs, Tensor) and obs.numel():
        if obs.dim() == 3:
            out["obstacles"] = obs.unsqueeze(1)
        elif obs.dim() != 4:
            raise ValueError(
                f"obstacles must be [B, K, 2] or [B, 1, K, 2], got "
                f"{tuple(obs.shape)}")
    return out


@dataclass
class TargetConfig:
    """Every threshold that decides a target. ⭐ Each one is either INHERITED
    from a calibrated source (named below) or is a DECLARED DECISION.

    ⛔ ``rl.rewards`` carries the calibration registry for its own constants
    (``proximity_safe_m = 5.0`` flagged **the human driver's own future on
    34.9 %** of lead windows — a threshold set where the human lives, opposing
    the trust region by construction, and no test could catch it because
    nothing required a constant to be justified). The same discipline applies
    here: a threshold with no derivation may not ship.
    """

    dt: float = 0.1
    #: INHERITED from ``rewards`` defaults, which carry their own calibration.
    ego_radius_m: float = 1.0
    obs_radius_m: float = 1.0
    lead_len_m: float = 4.5
    target_time_gap_s: float = 2.0
    ttc_min_s: float = 1.5
    #: DECISION: comfort/feasibility are UNDEFINED below this arc length, so
    #: the target is MASKED rather than scored. Same value and same meaning as
    #: ``rewards._motion_gate``'s ``min_m`` default; here it is a mask, not a
    #: multiplicative ramp, because a soft ramp toward 0 would teach the head
    #: that a stationary candidate is *uncomfortable* rather than *not a ride*.
    min_motion_m: float = 1.0
    #: INHERITED: the (a, kappa) envelope from ``rewards``.
    a_max: float = R.A_MAX_MPS2
    kappa_max: float = R.KAPPA_MAX_1PM
    jerk_max: float = R.JERK_MAX_MPS3
    lat_acc_max: float = R.LAT_ACC_MAX_MPS2
    #: DECISION: the friction coefficient the Kamm target is scored against.
    #: 0.8 = dry asphalt, the middle of ``flyability.kamm_report``'s published
    #: sweep ``(0.7, 0.8, 0.9, 1.0)``. ⚠️ An arm on a wet/OOD split must state
    #: its own mu; this is not a universal constant.
    mu: float = 0.8
    #: DECISION: whether the Kamm load enters the comfort target at all.
    #: Default OFF — comfort is jerk + lateral acceleration (V2's C sub-score
    #: shape); the friction circle is a FEASIBILITY fact and is reported
    #: separately in ``extras`` so an arm can see it without it silently
    #: changing what the comfort head means.
    comfort_uses_kamm: bool = False
    #: DECISION: fuse the imminent-TTC veto into the ttc target by zeroing it.
    #: The unfused components are ALWAYS returned in ``extras`` so the panel can
    #: read the constraint apart from the ranking.
    ttc_fuse_veto: bool = True
    #: ⚠️ tau for the compliance predicate, radians of terminal heading.
    #: INHERITED SHAPE from ``nav_compliance.complies(signal, side, tau)``;
    #: the VALUE is derived per corpus by
    #: ``nav_compliance.derive_tolerance(pos, neg)`` and MUST be passed by the
    #: arm. The default here is deliberately ``None`` -> compliance is not
    #: computed, rather than computed against an invented tolerance.
    compliance_tau_rad: float | None = None
    #: ⚠️ The tactical label's horizon is 2 s (``refc_tactical.LABEL_HORIZON``
    #: = 20 steps @ 10 Hz); the candidate is 6 s. Truncate the candidate to
    #: this many STEPS before deriving its cell, or the two are not the same
    #: quantity. ``None`` -> use the whole candidate AND say so in provenance.
    tactical_horizon_steps: int | None = None


# ---------------------------------------------------------------------------
# Kamm — the same quantity flyability computes, on a rolled path
# ---------------------------------------------------------------------------
def kamm_load(cand: Tensor, dt: float = 0.1) -> Tensor:
    """Peak realised friction load of every candidate, in **g**. ``[B, N]``.

    ``sqrt(a_lon^2 + a_lat^2) / G`` maximised over steps — term for term the
    ``peak_g`` that ``instruments.flyability.friction_load`` returns, computed
    from ``rewards.kinematics``' REALISED ``accel`` and ``lat_acc`` instead of
    from a constant anchor control re-integrated through ``rollout_unicycle``.

    ⚠️ It is re-derived, not re-invented, and the reason is an INTERFACE
    mismatch stated in the module docstring: ``friction_load`` is anchor-shaped
    (``controls [N, 2]`` x ``v0 [B]``) and our candidates are already rolled and
    per-step varying. ``G`` is imported from ``flyability`` so gravity has one
    home; if that constant ever changes, this follows.
    """
    kin = R.kinematics(cand, dt)
    a_tot = (kin.accel.pow(2) + kin.lat_acc.pow(2)).clamp_min(1e-12).sqrt()
    return a_tot.amax(dim=-1) / G


# ---------------------------------------------------------------------------
# The six targets
# ---------------------------------------------------------------------------
def nc_target(cand: Tensor, ctx: dict, cfg: TargetConfig | None = None
              ) -> tuple[Tensor, Tensor]:
    """NO-CONTACT. ``1.0`` = clean, ``0.0`` = contact. ``([B, N], mask)``.

    Delegates entirely to ``rewards._collision`` (which returns ``-1`` on
    contact, ``0`` otherwise) and shifts by ``+1``. Both of its obstacle models
    are inherited: a static ``obstacles`` set, or — when none is given — a
    **time-aligned moving** ``lead_path``.

    ⛔ Do NOT "improve" this by holding the lead static at its first sample.
    MEASURED (``rl_refcv3_min.py --mode humanflag``): that turns every competent
    follower into a collision, because the ego reaches the lead's t0 position
    after one time gap. It is the ``H-RL-THRESH-1`` class — a safety term that
    fires on the demonstration.

    ⚠️ ABSENCE READS 1.0 (the ``neutral`` rule: no agents ⇒ nothing to hit ⇒ no
    penalty). That is honest but it is also *no information*, so
    :func:`selector_targets` counts how often the predicate actually fires and
    publishes it in ``coverage``. A head trained on a corpus where NC never
    fires has learned the constant 1.0, and only the coverage count can say so.
    """
    cfg = cfg or TargetConfig()
    c = R._collision(cand, {**ctx, "dt": cfg.dt,
                            "ego_radius_m": cfg.ego_radius_m,
                            "obs_radius_m": cfg.obs_radius_m})
    return (1.0 + c).clamp(0.0, 1.0), torch.ones_like(c, dtype=torch.bool)


def ttc_target(cand: Tensor, ctx: dict, cfg: TargetConfig | None = None
               ) -> tuple[Tensor, Tensor, dict]:
    """TIME-GAP quality, zeroed where TTC is imminent. ``([B,N], mask, extras)``

    ``rewards._headway`` supplies the graded, asymmetric time-gap shape (peak
    exactly 1.0 at the target gap; steep below, mild above) and
    ``rewards.ttc_violation`` supplies the hard imminent-collision veto.

    ⚠️ ``rewards.py`` argues — correctly — that fusing a CONSTRAINT and a
    RANKING into one *reward term* is what made the first headway component
    inert. This is not that failure and the difference matters: the inert
    version saturated, with a **median fan spread of 0.0000**, because *every*
    candidate pinned at the constraint's satisfied value. Here the veto only
    ever *adds* a hard zero to candidates that are imminent, so it strictly
    increases spread. Still, the fusion is a DECISION and it is switchable
    (``cfg.ttc_fuse_veto``), and **both components are always returned
    unfused** in ``extras`` so a panel can read them apart.
    """
    cfg = cfg or TargetConfig()
    c = {**ctx, "dt": cfg.dt, "lead_len_m": cfg.lead_len_m,
         "target_time_gap_s": cfg.target_time_gap_s, "ttc_min_s": cfg.ttc_min_s}
    gap = R._headway(cand, c).clamp(0.0, 1.0)
    veto = R.ttc_violation(cand, c)
    extras = {"headway": gap, "ttc_veto": veto,
              "ttc_veto_rate": float(veto.to(torch.float32).mean())}
    tgt = gap.masked_fill(veto, 0.0) if cfg.ttc_fuse_veto else gap
    has_lead = ctx.get("lead_path") is not None
    mask = torch.full_like(tgt, float(has_lead)).bool()
    return tgt, mask, extras


def progress_target(cand: Tensor, ctx: dict, cfg: TargetConfig | None = None
                    ) -> tuple[Tensor, Tensor]:
    """ALONG-TRACK progress vs ``v0``, affinely mapped to ``[0, 1]``.

    ``rewards._progress`` reads ``1.0`` for "kept the current speed", >1 for
    accelerating, <1 for decelerating, and is clamped by its component to
    ``[lo, hi]``. The BCE head needs ``[0, 1]``, so the map is
    ``(p - lo) / (hi - lo)``.

    ⭐ ``lo`` and ``hi`` are READ FROM ``rewards.COMPONENTS["progress"]``, not
    written here. If that clamp is ever re-tuned, this mapping follows it
    automatically; a hardcoded pair would silently re-scale every target the
    day the component changed — the *derived-constant* trap (``CLAUDE.md``:
    ``HORIZON`` moved from ``7`` to ``round(6.0*10.0/STRIDE)`` and a
    "reproduction" quietly became a different experiment).

    ⛔ This head is the HACKABLE one and it is a head, never the only head.
    ``rewards.HACKABLE_WEIGHTS = {"progress": 1.0}`` is the deliberate-
    regression arm; the test suite's progress-only selector must pick a
    colliding candidate, or the collision readout is blind and the panel VOID.
    """
    cfg = cfg or TargetConfig()
    comp = R.COMPONENTS["progress"]
    p = comp(cand, {**ctx, "dt": cfg.dt})
    span = max(comp.hi - comp.lo, 1e-8)
    return ((p - comp.lo) / span).clamp(0.0, 1.0), \
        torch.ones_like(p, dtype=torch.bool)


def comfort_target(cand: Tensor, ctx: dict, cfg: TargetConfig | None = None
                   ) -> tuple[Tensor, Tensor, dict]:
    """COMFORT x FEASIBILITY, MASKED where the candidate does not move.

    ``rewards._comfort`` (jerk + lateral acceleration) and
    ``rewards._kinematic_feasibility`` (the (a, kappa) envelope) both already
    apply ``_motion_gate``, which ramps them to 0 for a stationary path — that
    gate exists because without it "stand perfectly still" collected **0.70 of
    free reward** at the default weights.

    ⚠️ For a TARGET the gate is the wrong tool and the mask is the right one.
    A stationary candidate is not *uncomfortable*; it has **no ride to be
    comfortable about**, and training a head to predict "0" there teaches it to
    equate stillness with harshness. So: divide the gate back out, and mask.
    The plan's own words are *"comfort masked where undefined"*.
    """
    cfg = cfg or TargetConfig()
    c = {**ctx, "dt": cfg.dt, "a_max": cfg.a_max, "kappa_max": cfg.kappa_max,
         "jerk_max": cfg.jerk_max, "lat_acc_max": cfg.lat_acc_max,
         "min_motion_m": cfg.min_motion_m}
    kin = R.kinematics(cand, cfg.dt)
    gate = R._motion_gate(kin, cfg.min_motion_m)
    # ⛔ MASK ON gate == 1 EXACTLY, not on gate > 0. `_motion_gate` RAMPS from 0
    # to 1 over `min_motion_m`, so a candidate that moves 0.01 m has gate 0.01
    # and dividing it out multiplies the comfort value by 100 before the clamp
    # — every barely-moving candidate would read a perfect 1.0. Partial motion
    # is exactly the regime where "is this ride comfortable" has no answer, so
    # it is UNDEFINED (masked), not maximal.
    moving = kin.arc_len >= float(cfg.min_motion_m)
    safe = gate.clamp_min(1e-6)
    comfort = (R._comfort(cand, c) / safe).clamp(0.0, 1.0)
    feas = (R._kinematic_feasibility(cand, c) / safe).clamp(0.0, 1.0)
    peak_g = kamm_load(cand, cfg.dt)
    kamm_ok = (peak_g <= cfg.mu).to(comfort.dtype)
    extras = {"comfort_raw": comfort, "feasibility": feas,
              "peak_g": peak_g, "kamm_ok": kamm_ok,
              "kamm_over_mu": float((peak_g > cfg.mu).to(torch.float32).mean())}
    tgt = comfort * feas
    if cfg.comfort_uses_kamm:
        tgt = tgt * kamm_ok
    return tgt.clamp(0.0, 1.0), moving, extras


def compliance_target(cand: Tensor, nav_cmd: Tensor, tau_rad: float,
                      stall_m: float = 0.05) -> tuple[Tensor, Tensor]:
    """Does the candidate's TERMINAL HEADING obey the nav command it was told?

    A torch mirror of ``taniteval.nav_compliance``:
    ``complies(terminal_heading(path), commanded_side(nav), tau)`` — 1.0 where
    the signed terminal heading has the commanded sign **and** a magnitude of at
    least ``tau``; 0.0 otherwise. A stalled last segment (``< stall_m``, the
    same 0.05 m) reads heading exactly 0.0 and therefore never complies.

    ⛔ The MASK is ``side != 0``: ``follow`` and ``straight`` carry no commanded
    side, so the predicate is **undefined**, not "failed". This is the
    ``informative`` split ``nav_compliance`` already makes; scoring the
    uninformative windows would put a large block of trivially-0 targets into
    the head and let it hit a fine BCE by predicting "never complies".

    ⚠️ ``tau_rad`` has NO DEFAULT. It is derived per corpus by
    ``nav_compliance.derive_tolerance(pos, neg)`` and the arm must state the
    value it used. An invented tolerance here is a number with no evidence
    class deciding what "compliant" means.

    ⚠️ **This head is TOLD the command** — that is S7 in learned form. The
    honest read is therefore the SHUFFLE / ZERO delta (I2), never the absolute
    compliance rate: a head fed the command and scored on obeying it can reach
    a high rate by echoing its own input, which is exactly the defect MEASURED
    on flagship v1's route head (an exact bijection of its nav input, 369/369
    and 81/81, scoring 1.0000).
    """
    if cand.dim() != 4:
        raise ValueError(f"cand must be [B, N, S, 2], got {tuple(cand.shape)}")
    d = cand[..., -1, :] - cand[..., -2, :]
    seg = d.norm(dim=-1)
    th = torch.atan2(d[..., 1], d[..., 0])
    th = torch.where(seg < stall_m, torch.zeros_like(th), th)   # [B, N]

    nav = nav_cmd.reshape(-1).to(torch.long)
    side = torch.zeros(nav.shape, device=cand.device, dtype=cand.dtype)
    for k, v in NAV_SIDE.items():
        side = torch.where(nav == k,
                           torch.full_like(side, float(v)), side)
    side = side.unsqueeze(-1)                                    # [B, 1]
    ok = (torch.sign(th) == torch.sign(side)) & (th.abs() >= float(tau_rad))
    mask = (side != 0).expand_as(ok)
    return (ok & mask).to(cand.dtype), mask


def tactical_target(cand: Tensor, v0: Tensor, lat_label: Tensor,
                    lon_label: Tensor, cfg: TargetConfig | None = None
                    ) -> tuple[Tensor, Tensor, dict]:
    """⚠️ QUARANTINED. Does the candidate's ``(lat, lon)`` cell match the label?

    The cell comes from ``refc_tactical.factor_from_kinematics`` — the SAME
    function that produces the v7.2 factored label, so the candidate and the
    label are classified by one gate rather than two that can drift.

    ⛔ **THIS TARGET READS A LABEL DERIVED FROM THE EGO FUTURE.** It is a
    *partial echo* (V2 analysis §3.3). It is admissible as a selector target
    only under three conditions, all of which this module enforces or records:
      1. it is REPORTED SEPARATELY (its own head, its own loss key);
      2. it is OFF in the score composition by default
         (``SelectorConfig.w_tactical = 0.0``);
      3. ``H-DDA-6``: the class may be the RL **grouping key** XOR a reward
         term, never both. Nothing here may be wired into a reward.

    ⚠️ **HORIZON MISMATCH, stated because it silently makes the two quantities
    different things.** ``refc_tactical.LABEL_HORIZON`` is **20 steps = 2 s**;
    the candidate fan is **6 s**. A cell derived from the full 6 s candidate is
    NOT the quantity the label names. Pass
    ``cfg.tactical_horizon_steps`` to truncate; leaving it ``None`` is allowed
    but is recorded in the provenance dict as ``horizon_mismatch``.
    """
    cfg = cfg or TargetConfig()
    path = cand
    trunc = cfg.tactical_horizon_steps
    if trunc is not None:
        if trunc < 4:
            raise ValueError("need >=4 waypoints to derive kinematics, "
                             f"got tactical_horizon_steps={trunc}")
        path = cand[..., :int(trunc), :]
    kin = R.kinematics(path, cfg.dt)
    # The candidate is EGO-FRAME and starts along +x, so the terminal segment's
    # heading IS the yaw change over the window. Same convention as
    # `_terminal_heading` in refc_selector and as nav_compliance's.
    dyaw = kin.heading[..., -1]                                  # [B, N]
    v_end = kin.speed[..., -1]                                   # [B, N]
    v0b = v0.reshape(-1, 1).to(cand.dtype).expand_as(v_end)
    kap = kin.kappa.mean(dim=-1)
    lat, lon = _tac.factor_from_kinematics(dyaw, v_end - v0b, v0b, v_end, kap)
    lat_t = lat_label.reshape(-1, 1).to(lat.device)
    lon_t = lon_label.reshape(-1, 1).to(lon.device)
    agree = ((lat == lat_t) & (lon == lon_t)).to(cand.dtype)
    extras = {"cand_lat": lat, "cand_lon": lon,
              "horizon_mismatch": bool(trunc is None
                                       or trunc != _tac.LABEL_HORIZON),
              "label_horizon_steps": int(_tac.LABEL_HORIZON),
              "used_steps": int(path.shape[-2])}
    return agree, torch.ones_like(agree, dtype=torch.bool), extras


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------
def selector_targets(cand: Tensor, ctx: dict,
                     cfg: TargetConfig | None = None,
                     nav_cmd: Tensor | None = None,
                     lat_label: Tensor | None = None,
                     lon_label: Tensor | None = None) -> dict:
    """All available targets for one batch of emitted fans.

    ``cand`` ``[B, N, S, 2]``. ``ctx`` carries the per-window scene facts in
    their natural shapes (see the module docstring) plus ``v0 [B]``.

    Returns ``{"targets": {...}, "masks": {...}, "extras": {...},
    "coverage": {...}, "provenance": {...}}``.

    ⭐ ``coverage`` exists because **absence and safety look identical in a
    mean**. ``report_component_coverage`` makes the same argument for the reward
    library. A head whose predicate never fired has learned a constant, and the
    only thing that can say so is a count.

    ⛔ A head is OMITTED — not zero-filled — when its inputs are absent. A
    zero-filled compliance head trains toward "never complies"; an omitted one
    is visibly untrained.
    """
    cfg = cfg or TargetConfig()
    assert_environment_only(ctx)
    ctx = broadcast_scene({**ctx, "dt": cfg.dt})

    tg: dict[str, Tensor] = {}
    mk: dict[str, Tensor] = {}
    ex: dict = {}
    prov: dict = {"env_heads": sorted(ENV_HEADS),
                  "gt_derived_heads": sorted(GT_DERIVED_HEADS),
                  "omitted": [], "reasons": {}}

    tg["nc"], mk["nc"] = nc_target(cand, ctx, cfg)
    tg["progress"], mk["progress"] = progress_target(cand, ctx, cfg)
    tg["comfort"], mk["comfort"], ex["comfort"] = comfort_target(cand, ctx, cfg)

    if ctx.get("lead_path") is None:
        prov["omitted"].append("ttc")
        prov["reasons"]["ttc"] = "no lead_path in this window"
    else:
        tg["ttc"], mk["ttc"], ex["ttc"] = ttc_target(cand, ctx, cfg)

    if nav_cmd is None or cfg.compliance_tau_rad is None:
        prov["omitted"].append("compliance")
        prov["reasons"]["compliance"] = (
            "no nav_cmd" if nav_cmd is None else
            "TargetConfig.compliance_tau_rad is None — derive it with "
            "nav_compliance.derive_tolerance and pass it; an invented "
            "tolerance is a number with no evidence class")
    else:
        tg["compliance"], mk["compliance"] = compliance_target(
            cand, nav_cmd, float(cfg.compliance_tau_rad))

    if lat_label is None or lon_label is None:
        prov["omitted"].append("tactical")
        prov["reasons"]["tactical"] = "no (lat, lon) v7.2 label supplied"
    else:
        tg["tactical"], mk["tactical"], ex["tactical"] = tactical_target(
            cand, ctx["v0"], lat_label, lon_label, cfg)
        prov["tactical_quarantine"] = (
            "GT-DERIVED: the label comes from the ego future. Report "
            "separately; w_tactical defaults to 0.0; H-DDA-6 forbids it being "
            "both an RL grouping key and a reward term.")

    cov: dict = {}
    for h in HEAD_NAMES:
        if h not in tg:
            cov[h] = {"present": False}
            continue
        m = mk[h]
        t = tg[h]
        n = int(m.sum())
        cov[h] = {
            "present": True,
            "n_defined": n,
            "n_total": int(m.numel()),
            "frac_defined": (n / m.numel()) if m.numel() else 0.0,
            "mean_where_defined": (
                float((t * m).sum() / max(n, 1)) if n else float("nan")),
            "spread_where_defined": (
                float(t[m].std()) if n > 1 else float("nan")),
        }
    # ⚠️ A0's lesson in one number: a target whose spread ACROSS THE FAN is 0
    # cancels exactly in any ranking objective. It is present and carries no
    # signal, and only this field says so.
    cov["_note"] = ("spread_where_defined == 0 means the target is identical "
                    "for every candidate and cannot rank anything (the A0 "
                    "inert-headway failure, MEASURED median spread 0.0000)")
    return {"targets": tg, "masks": mk, "extras": ex, "coverage": cov,
            "provenance": prov}
