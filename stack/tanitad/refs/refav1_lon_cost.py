#!/usr/bin/env python3
"""refav1's LONGITUDINAL distance-keeping cost term -- the first non-degenerate
cost on the acceleration channel.

⛔ WHY THIS FILE EXISTS. `refa_v1.plan`'s `_cost_chunk` prices exactly three
things: the goal term, `w_jerk * mean(jerk^2)` and `w_kappa * mean(kappa^2)`,
plus `w_vend * (v_end - target_speed)^2` **only when `target_speed` is armed --
and `refav1_arm.py`'s single `.plan(...)` call never passes it**. So on every
banked refav1 window the longitudinal channel carried NO cost at all, and the
all-zero control is the JOINT minimiser of both surviving regularisers
(`mean(jerk^2)` is 0 for ANY constant acceleration, including 0;
`mean(kappa^2)` is 0 only at kappa = 0). MEASURED consequence on the 21109
open-loop panel: `cl` is bit-identical to the constant-velocity control `ha0` on
**270/282** windows, the emitted controls are exactly zero on 270/282, and
kappa is identically zero on 282/282. A planner with no longitudinal cost does
not keep distance; it holds speed.

⭐ WHY *THIS* TERM AND NOT `target_speed`. `W_VEND` prices a speed against a
target the planner must be TOLD. Distance-keeping prices a gap the planner can
SEE -- and M84 (`H-REFAV1-PERCEPT-1`, 2026-09-06) measured that it can:
decoding `lead_gap_m` (lead <= 30 m, vehicles only) from the planner's own
`_last_state` latent with a ridge head scores **R^2_skill +0.3632
[+0.2069, +0.5088]** over 42 episode clusters against a **raw-pixel floor of
-0.0513** and a constant control of **exactly +0.000000**; the paired
`field - pix` delta is **+0.4145 [+0.2018, +0.6120], excluding zero**, and the
within-clip skill is **+0.3995** against a within-clip shuffle of -0.0013, so it
tracks the lead as it MOVES rather than merely identifying the clip.
⇒ the objective below is REPRESENTABLE in the coordinate the planner already
has, which is the whole argument for putting it in the cost.

⛔⛔ THE ASSUMPTION THIS TERM IS BUILT ON, STATED UP FRONT BECAUSE IT IS
FALSIFIED-BY-CONSTRUCTION AND NOT HIDDEN. The same probe found the **closing
rate is absent from the latent entirely** -- `field` +0.0061 [-0.0406, +0.0513],
`dino` +0.0114, pixels -0.0001, nonlinear included, every paired delta spanning
zero -- and its named next lever (hand the cost an explicit temporal difference,
`field_pair` = [state, state - state_prev]) recovered **+0.0145, still spanning
zero**, against a same-breath non-zero control on the *gap* target. The rate is
not merely collapsed out of `_last_state`; it is not in the field sequence
either. ⇒ **this term CANNOT price closing, and does not pretend to.** The lead's
future speed is closed at :data:`LEAD_SPEED_STEADY` (the lead travels at the
ego's own measured `v0`), which is the maximum-entropy choice given that null and
makes the predicted gap depend ONLY on the candidate's excess displacement over
constant velocity. Any arm may override it with a measured/decoded lead speed the
moment one exists; the closure travels in the result so no reader has to guess.

⭐ THE COORDINATE, which is the question this file answers. Everything below is a
function of exactly two things:
  * ``gap0_m`` -- the causal gap at t0, the ONE perception input, and the one M84
    proved decodable (necessary, not sufficient -- C131);
  * the candidate's OWN kinematics -- integrated here with `rollout_unicycle`'s
    convention *exactly* (see :func:`predicted_speeds`), so the gap this cost
    minimises is the gap `taniteval.lead_metrics.per_step_gap` will later score,
    not a second geometry that happens to have the same name.
Nothing else is read. There is no future ego state, no future lead state, and no
label at inference -- the vision-only rule (Sayed 2026-08-03) is satisfied by
construction, and an ORACLE-gap arm is an ORACLE arm and must be stamped as one.

⛔ GAP CONVENTION -- inherited unchanged from `lead_state_gate` / `lead_metrics`
/ `lead_source`, because two gap conventions in one programme is a retraction
waiting to happen: ``gap = along - size_x/2``, i.e. **rig origin to the lead's
REAR FACE**, never bumper-to-bumper. `gap0_m` handed in here must already be in
that convention -- which is what the B1 eval lead block's ``gap0_m`` column is.

⚠️ THE DESIRED GAP IS IDM-SHAPED, AND THAT IS A REGIME DECISION, NOT A DEFAULT.
``s*(v) = d0 + tau * v`` (metres). A pure time-gap barrier ``gap/v >= tau`` is
undefined as ``v -> 0`` and would rate a 1 m gap at 0.5 m/s as a 2 s headway --
safe by arithmetic, a collision in fact. A pure distance barrier is speed-blind
and would brake at 1 m/s for a 10 m gap. The affine form is the standard
car-following invariant and covers both regimes with one expression.
⇒ this is the M74 lesson applied before it bites: **a threshold carries its
regime**, and this one carries d0 for the crawl and tau for the cruise.

⚠️ ONE-SIDED, ALWAYS. The penalty is ``relu(s*(v) - gap)^2``: it is exactly zero
once the gap is adequate, so the term can never reward accelerating INTO a large
gap (that is the target-speed term's job and mixing them would make the two
non-attributable), and it saturates rather than rewarding ever-harder braking --
the existing `w_jerk` prices the deceleration that buys nothing more.

⛔ DEFAULT-OFF. ``w_dk = 0`` or a non-finite ``gap0_m`` returns EXACTLY zeros, so
a caller that does not arm it is bit-identical to every arm banked before
2026-09-06. That is the same discipline `jerk_seam_a0=None` and
`w_kappa_by_goal=None` already follow in `refa_v1.plan`.

Registered as ``D-REFAV1-DK-COST``. Tests: ``stack/tests/test_refav1_lon_cost.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import Tensor

__all__ = [
    "DK_TAU_TARGET_S", "DK_D0_M", "DK_W", "LEAD_SPEED_STEADY", "LEAD_SPEED_STATIC",
    "LEAD_SPEED_CLOSURES",
    "GAP_SOURCE_ORACLE", "GAP_SOURCE_DECODED_GAP", "GAP_SOURCE_DECODED",
    "GAP_SOURCE_UNIT_TEST", "GAP_SOURCE_UNSET", "GAP_SOURCES", "GAP_SOURCE_KIND",
    "GapHeadProvenance",
    "DistanceKeepingSpec", "predicted_speeds", "predicted_along", "predicted_gap",
    "desired_gap", "distance_keeping_cost", "gap_violation",
]

#: desired TIME gap, seconds. 1.5 s is the conventional following target and sits
#: inside this corpus's own measured distribution rather than above it: the 21109
#: panel's realised ``mean_time_gap_min_s`` is 4.6913 s over n = 81, so a 1.5 s
#: target is a floor the human already clears on average -- it prices the tail,
#: not the median. ⚠️ It is a WEIGHT-CLASS choice, not a physical constant; any
#: arm that moves it declares it, exactly as `W_KAPPA` must be declared.
DK_TAU_TARGET_S = 1.5
#: standstill gap, metres -- the term that keeps the barrier meaningful at v ~ 0,
#: where a time gap is not. Rig origin to the lead's rear face (see the module
#: docstring's convention note), so it is a stopping distance from the SENSOR
#: origin and is deliberately larger than a bumper-to-bumper figure.
DK_D0_M = 5.0
#: default weight. ⛔ 0.0 -- the shipped path is OFF, and an arm that arms this
#: term states the number it used. Its units are cost per m^2 of gap shortfall,
#: which is NOT commensurate with `W_KAPPA` (per (1/m)^2) or `W_JERK`
#: (per (m/s^3)^2); the three are calibrated against each other by measurement,
#: never by intuition.
DK_W = 0.0

#: lead-speed closures. ⛔ `LEAD_SPEED_STEADY` is the ONLY one admissible at
#: inference today -- see the module docstring on M84's closing-rate null.
LEAD_SPEED_STEADY = "steady_v0"
LEAD_SPEED_STATIC = "static"
LEAD_SPEED_CLOSURES = (LEAD_SPEED_STEADY, LEAD_SPEED_STATIC)


def _check_closure(closure: str) -> str:
    if closure not in LEAD_SPEED_CLOSURES:
        raise ValueError(
            f"lead_speed_closure must be one of {LEAD_SPEED_CLOSURES}, got "
            f"{closure!r}. A closure is an ASSUMPTION about an unobservable "
            f"(M84: the closing rate does not decode) and must be named, never "
            f"defaulted silently.")
    return closure


# --------------------------------------------------------------------------- #
# 0. WHERE `gap0_m` CAME FROM -- a CLOSED vocabulary (D-REFAV1-DK-DECODED)      #
# --------------------------------------------------------------------------- #
# ⛔⛔ THE VOCABULARY IS CLOSED AND THE REFUSAL IS THE POINT. Before 2026-09-06
# this field was a free-form string: ANY value was accepted and stamped into the
# dump verbatim, so a typo, a stale flag or an optimistic label would have been
# recorded as fact. The fix for "the decoded head does not exist yet" was never
# to relax the check -- it is to EXTEND the vocabulary and make every member
# carry what it is. An unrecognised source now RAISES.
#
# ⭐ THE THREE PRODUCTION SOURCES ARE A LADDER, one variable per step, because
# the arming GATE and the gap VALUE are two different oracles and conflating
# them would make the result non-attributable:
#
#   source            | lead PRESENT gate | gap0 value | what it is
#   ------------------|-------------------|------------|---------------------
#   oracle_label      | ORACLE (block)    | ORACLE     | a CEILING, never a
#                     |                   |            | driving capability
#   decoded_gap       | ORACLE (block)    | DECODED    | isolates the GAP decode
#   decoded           | DECODED           | DECODED    | ⭐ VISION-ONLY, the only
#                     |                   |            | deployable member
#
#: gap AND gate from the B1 lead block's labels. ⛔ A CEILING ARM. It answers
#: "does the cost geometry work when perception is perfect?" -- which is the
#: question M53/M54 left open when a PERFECT GOAL made this planner 2.03x WORSE.
GAP_SOURCE_ORACLE = "oracle_label"
#: gap from a vision head, arming gate still from the label. ⛔ NOT vision-only:
#: it is the middle rung, and exists so a failure can be attributed to the gap
#: decode rather than to the presence decode.
GAP_SOURCE_DECODED_GAP = "decoded_gap"
#: gap AND gate from a vision head. ⭐ The only member that satisfies the binding
#: vision-only rule (Sayed 2026-08-03) at inference.
GAP_SOURCE_DECODED = "decoded"
#: reserved for unit tests of the ARITHMETIC. ⛔ Stamped non-deployable so a dump
#: carrying it can never be read as a run.
GAP_SOURCE_UNIT_TEST = "unit-test"
#: the default. Legal only while the term is UNARMED -- see `__post_init__`.
GAP_SOURCE_UNSET = "unset"

GAP_SOURCES = (GAP_SOURCE_ORACLE, GAP_SOURCE_DECODED_GAP, GAP_SOURCE_DECODED,
               GAP_SOURCE_UNIT_TEST, GAP_SOURCE_UNSET)

#: what each source IS, stamped into every record so no reader has to remember.
#: ``needs_head`` is what makes an unstamped decoded arm impossible.
GAP_SOURCE_KIND = {
    GAP_SOURCE_ORACLE:      {"gate": "oracle", "gap": "oracle",
                             "vision_only": False, "is_ceiling": True,
                             "deployable": False, "needs_head": False},
    GAP_SOURCE_DECODED_GAP: {"gate": "oracle", "gap": "decoded",
                             "vision_only": False, "is_ceiling": True,
                             "deployable": False, "needs_head": True},
    GAP_SOURCE_DECODED:     {"gate": "decoded", "gap": "decoded",
                             "vision_only": True, "is_ceiling": False,
                             "deployable": True, "needs_head": True},
    GAP_SOURCE_UNIT_TEST:   {"gate": "none", "gap": "synthetic",
                             "vision_only": False, "is_ceiling": False,
                             "deployable": False, "needs_head": False},
    GAP_SOURCE_UNSET:       {"gate": "none", "gap": "none",
                             "vision_only": False, "is_ceiling": False,
                             "deployable": False, "needs_head": False},
}


def _check_gap_source(src: str) -> str:
    if src not in GAP_SOURCES:
        raise ValueError(
            f"gap_source must be one of {GAP_SOURCES}, got {src!r}. The "
            f"vocabulary is CLOSED on purpose: a free-form source string is "
            f"stamped into the dump verbatim, so an unrecognised one would be "
            f"BANKED AS FACT. Add the source to GAP_SOURCES with its "
            f"GAP_SOURCE_KIND entry -- never pass it through.")
    return src


@dataclass(frozen=True)
class GapHeadProvenance:
    """What produced a DECODED gap. ⛔ Mandatory for any decoded source, and
    forbidden for the oracle one, so the two can never be confused in a dump.

    ⚠️ Every field here answers a question a reader would otherwise have to
    reconstruct: *which* head, fit against *which* trunk, over *which* corpus,
    under *which* split. A `.pt` opened in isolation is opened far more often
    than its run record (the `anchors.pt` units retraction, 2026-09-04), so the
    identity travels in the dump and not only in the bundle.
    """

    head_path: str = ""
    head_sha256: str = ""
    head_version: str = ""
    #: the trunk the head was fit against. ⛔ A head fit on a DIFFERENT
    #: checkpoint decodes a different latent and its numbers are meaningless.
    ckpt_sha256: str = ""
    ckpt_step: int = -1
    fit_corpus: str = ""
    fit_scheme: str = ""
    n_fit_rows: int = 0
    n_fit_clips: int = 0
    pool: tuple = ()
    grid: tuple = ()
    win: int = 0

    def __post_init__(self) -> None:
        missing = [k for k in ("head_path", "head_sha256", "head_version",
                               "ckpt_sha256", "fit_corpus", "fit_scheme")
                   if not str(getattr(self, k))]
        if missing:
            raise ValueError(
                f"GapHeadProvenance is missing {missing}. An unstamped decoded "
                f"gap is indistinguishable from an oracle one in the dump, "
                f"which is the exact confusion this class exists to prevent.")
        if len(self.head_sha256) != 64:
            raise ValueError(
                f"head_sha256 must be a 64-char sha256, got "
                f"{len(self.head_sha256)} chars. A short or empty hash is how a "
                f"'verified' comparison passes on two failed reads.")

    def record(self) -> dict:
        d = asdict(self)
        d["pool"] = list(self.pool)
        d["grid"] = list(self.grid)
        return d


@dataclass(frozen=True)
class DistanceKeepingSpec:
    """The full declaration of one distance-keeping term. Travels into the run
    record so a number is never quoted without the assumption behind it."""

    w_dk: float = DK_W
    tau_target_s: float = DK_TAU_TARGET_S
    d0_m: float = DK_D0_M
    lead_speed_closure: str = LEAD_SPEED_STEADY
    #: what produced ``gap0_m`` -- a member of the CLOSED :data:`GAP_SOURCES`
    #: vocabulary. "oracle_label" is a CEILING arm and must never be reported as
    #: a driving capability; "decoded" is the vision-only path.
    gap_source: str = GAP_SOURCE_UNSET
    #: ⛔ MANDATORY when ``gap_source`` decodes, FORBIDDEN when it does not.
    gap_head: "GapHeadProvenance | None" = None

    def __post_init__(self) -> None:
        _check_closure(self.lead_speed_closure)
        _check_gap_source(self.gap_source)
        if not self.w_dk >= 0.0:
            raise ValueError(f"w_dk must be >= 0, got {self.w_dk}")
        if not self.tau_target_s >= 0.0:
            raise ValueError(f"tau_target_s must be >= 0, got {self.tau_target_s}")
        if not self.d0_m >= 0.0:
            raise ValueError(f"d0_m must be >= 0, got {self.d0_m}")
        # ⛔ AN ARMED TERM MUST DECLARE WHERE ITS DISTANCE CAME FROM. The whole
        # point of `D-REFAV1-DK-COST` is that the gap is the ONE perception
        # input; an armed cost whose source is "unset" banks a number nobody can
        # place on the oracle/vision ladder.
        if self.w_dk > 0.0 and self.gap_source == GAP_SOURCE_UNSET:
            raise ValueError(
                "an ARMED distance-keeping term (w_dk > 0) must declare its "
                f"gap_source -- one of {GAP_SOURCES[:-1]}. 'unset' is legal only "
                "while the term is off.")
        kind = GAP_SOURCE_KIND[self.gap_source]
        # ⛔ AND THE STAMP MUST MATCH THE SOURCE, BOTH WAYS. A decoded arm with
        # no head provenance is unauditable; an ORACLE arm carrying head
        # provenance is worse, because it reads like a vision result.
        if kind["needs_head"] and self.gap_head is None:
            raise ValueError(
                f"gap_source={self.gap_source!r} decodes the gap and therefore "
                f"REQUIRES a GapHeadProvenance. Refusing to bank an arm whose "
                f"record could not name the head that produced its distances.")
        if not kind["needs_head"] and self.gap_head is not None:
            raise ValueError(
                f"gap_source={self.gap_source!r} does not decode, so it must "
                f"NOT carry head provenance -- an oracle arm stamped with a head "
                f"reads as a vision result, which is the more dangerous error.")

    @property
    def armed(self) -> bool:
        return self.w_dk > 0.0

    @property
    def vision_only(self) -> bool:
        """⭐ True ONLY for :data:`GAP_SOURCE_DECODED`. The middle rung
        (`decoded_gap`) still takes its ARMING GATE from the label and is not
        vision-only, however good its gap is."""
        return bool(GAP_SOURCE_KIND[self.gap_source]["vision_only"])

    @property
    def is_ceiling(self) -> bool:
        return bool(GAP_SOURCE_KIND[self.gap_source]["is_ceiling"])

    def record(self) -> dict:
        d = asdict(self)
        d["gap_head"] = (self.gap_head.record() if self.gap_head is not None
                         else None)
        # ⭐ the source's MEANING travels with it, so a dump can be read without
        # this module to hand.
        d["gap_source_kind"] = dict(GAP_SOURCE_KIND[self.gap_source])
        d["vision_only"] = self.vision_only
        d["is_ceiling"] = self.is_ceiling
        d["gap_convention"] = ("rig-origin to lead rear face (along - size_x/2); "
                               "NOT bumper-to-bumper -- lead_metrics.per_step_gap")
        d["closure_note"] = (
            "the lead's future speed is UNOBSERVABLE on this trunk (M84: closing "
            "rate R2_skill +0.0061 [-0.0406, +0.0513], every paired delta spans 0, "
            "the explicit temporal difference recovers nothing). "
            + ("assumed equal to the ego's measured v0 -- the predicted gap then "
               "depends ONLY on the candidate's excess displacement over constant "
               "velocity, and is EXACTLY gap0 for a == 0."
               if self.lead_speed_closure == LEAD_SPEED_STEADY else
               "assumed stationary -- a worst-case closure that fires on every "
               "lead and is NOT a following model."))
        return d


# --------------------------------------------------------------------------- #
# 1. the candidate's own kinematics -- rollout_unicycle's convention, exactly   #
# --------------------------------------------------------------------------- #
def predicted_speeds(accel: Tensor, v0: Tensor | float, dt: float) -> Tensor:
    """Speed at the START of each step, ``[n, H]``.

    ⛔ THE ORDER IS LOAD-BEARING and is copied from
    ``tanitad.models.kinematic.rollout_unicycle``: position advances on the speed
    at the START of the step and ``v`` is updated LAST, with ``clamp_min(0)``.
    Integrating with the post-update speed makes every trajectory systematically
    longer -- the over-progress bias -- and would make this cost's gap disagree
    with the gap ``lead_metrics`` scores. ⇒ ``v[:, 0] == v0`` always, and
    ``a[:, H-1]`` never affects any displacement inside the horizon.

    ⚠️ The clamp is not decorative: a hard-braking candidate hits zero and STAYS
    there, so its displacement stops growing. A closed-form ``v0 + cumsum(a)*dt``
    would go negative and credit the candidate with driving backwards.
    """
    if accel.ndim != 2:
        raise ValueError(f"accel must be [n, H], got {tuple(accel.shape)}")
    n, h = accel.shape
    v = (accel.new_full((n,), float(v0)) if not torch.is_tensor(v0)
         else v0.to(accel.dtype).to(accel.device).expand(n).clone())
    out = []
    for k in range(h):
        out.append(v)
        v = (v + accel[:, k] * dt).clamp_min(0.0)
    return torch.stack(out, dim=1)


def predicted_along(v: Tensor, dt: float) -> Tensor:
    """Along-track displacement after each step, ``[n, H]``, from the step-start
    speeds :func:`predicted_speeds` returns: ``s[k] = dt * sum_{j<=k} v[j]``.

    ⚠️ STRAIGHT-LINE. The true along-track distance of a curving candidate is
    shorter than its arc length by O(kappa^2); at this planner's own
    ``kappa_max`` over a 2 s horizon that is sub-percent, and the corridor gate
    the metric applies means a candidate curving far enough for it to matter has
    lost its lead anyway. Stated rather than assumed away.
    """
    return torch.cumsum(v, dim=1) * dt


def predicted_gap(accel: Tensor, *, v0: float, gap0_m: float, dt: float,
                  lead_speed_closure: str = LEAD_SPEED_STEADY,
                  lead_speed_mps: float | None = None) -> tuple[Tensor, Tensor]:
    """-> ``(gap [n, H], v [n, H])``: the predicted gap after each step.

    ``gap[k] = gap0 + s_lead(t_k) - s_ego(t_k)`` with ``t_k = (k + 1) * dt``.

    ⭐ Under :data:`LEAD_SPEED_STEADY` this collapses to
    ``gap[k] = gap0 - (s_ego(t_k) - v0 * t_k)`` -- the candidate's **excess
    displacement over constant velocity** -- so it is EXACTLY ``gap0`` for the
    all-zero control. That is the correct behaviour and the reason the term is
    one-sided: a constant-velocity plan neither closes nor opens the gap, and is
    penalised only if the gap is ALREADY inadequate.
    """
    _check_closure(lead_speed_closure)
    v = predicted_speeds(accel, v0, dt)
    s_ego = predicted_along(v, dt)
    h = accel.shape[1]
    t = torch.arange(1, h + 1, device=accel.device, dtype=accel.dtype) * dt
    if lead_speed_mps is not None:
        v_lead = float(lead_speed_mps)
    elif lead_speed_closure == LEAD_SPEED_STEADY:
        v_lead = float(v0)
    else:
        v_lead = 0.0
    return gap0_m + v_lead * t - s_ego, v


def desired_gap(v: Tensor, *, tau_target_s: float = DK_TAU_TARGET_S,
                d0_m: float = DK_D0_M) -> Tensor:
    """IDM-shaped desired gap ``s*(v) = d0 + tau * v``, metres. See the module
    docstring for why neither half may be dropped."""
    return d0_m + tau_target_s * v


def gap_violation(accel: Tensor, *, v0: float, gap0_m: float, dt: float,
                  tau_target_s: float = DK_TAU_TARGET_S, d0_m: float = DK_D0_M,
                  lead_speed_closure: str = LEAD_SPEED_STEADY,
                  lead_speed_mps: float | None = None) -> Tensor:
    """-> ``[n, H]`` shortfall ``relu(s*(v) - gap)`` in METRES, before squaring.

    Exposed separately from the cost because the shortfall is the readable
    diagnostic: "this plan comes 3.2 m closer than it should" is auditable, a
    weighted square is not.
    """
    gap, v = predicted_gap(accel, v0=v0, gap0_m=gap0_m, dt=dt,
                           lead_speed_closure=lead_speed_closure,
                           lead_speed_mps=lead_speed_mps)
    return torch.clamp(desired_gap(v, tau_target_s=tau_target_s, d0_m=d0_m) - gap,
                       min=0.0)


def distance_keeping_cost(controls: Tensor, *, v0: float, gap0_m: float | None,
                          dt: float, spec: DistanceKeepingSpec | None = None,
                          lead_speed_mps: float | None = None) -> Tensor:
    """-> ``[n]`` distance-keeping cost for a candidate batch ``[n, H, 2]``.

    ``w_dk * mean_k( relu(s*(v_k) - gap_k)^2 )``, in cost per m^2.

    ⛔ RETURNS EXACT ZEROS -- not "approximately zero" -- when the term is not
    armed (``w_dk == 0``), when there is no lead (``gap0_m`` is None or
    non-finite), or when the horizon is empty. A caller on the shipped path is
    therefore bit-identical to every arm banked before this term existed, which
    is the property that makes an A/B of it attributable.
    """
    spec = spec or DistanceKeepingSpec()
    if controls.ndim != 3 or controls.shape[-1] != 2:
        raise ValueError(f"controls must be [n, H, 2] = (accel, curvature), got "
                         f"{tuple(controls.shape)}")
    n = controls.shape[0]
    zero = controls.new_zeros(n)
    if not spec.armed or controls.shape[1] == 0:
        return zero
    if gap0_m is None:
        return zero
    g0 = float(gap0_m)
    if not (g0 == g0) or g0 in (float("inf"), float("-inf")):   # NaN / inf
        return zero
    viol = gap_violation(controls[..., 0], v0=v0, gap0_m=g0, dt=dt,
                         tau_target_s=spec.tau_target_s, d0_m=spec.d0_m,
                         lead_speed_closure=spec.lead_speed_closure,
                         lead_speed_mps=lead_speed_mps)
    return spec.w_dk * viol.pow(2).mean(-1)
