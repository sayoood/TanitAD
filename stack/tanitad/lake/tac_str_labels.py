"""Tactical + strategic label composition for the Alpamayo-augmented corpus.

⭐ THIS MODULE **IS** THE APPROVED CONCEPT, IN CODE — one file, so the rules
cannot drift from the document that authorised them
(`…/Research/2026-08-18-alpamayo-screening/TACTICAL_STRATEGIC_LABEL_CONCEPT.md`,
PI-approved 2026-08-18; the relative-lane-target addendum in
`RELATIVE_LANE_TARGET_PROPOSAL.md`).

⛔ THE ONE PRINCIPLE EVERYTHING ELSE FOLLOWS FROM. The three tiers do **not**
vote on the same field. They produce **different kinds of thing**, so there is no
fusion heuristic, no cross-tier echo, and precedence is needed only *within* a
field:

    Alpamayo  ->  the action CLASS        (lateral; a lane-topological fact)
    VLM       ->  the REASON + REFERENT   (longitudinal; a semantic fact)
                  and the RELATIVE LANE TARGET (a visual fact)
    ego       ->  the goal ARGUMENTS      (physical units; a measurement)

⭐ AND THE RULE THAT MAKES THE EGO TIER SAFE:
**ego QUANTIFIES what the VLM has NAMED — it may never fire where the VLM
abstained.** `vtarget_guarded` was rejected as *"hindsight EGO geometry — what
speed did this driver settle at, not what speed is permitted here"*, with
nothing beating repeating v0's band (0.4066 free vs 0.2465 trained). An ego
quantity gated on a named external referent measures a scene feature the VLM
independently asserted exists; ungated, it is the driver's idiosyncrasy wearing
a label's clothes.

⛔ WHAT THIS MODULE REFUSES, BY CONSTRUCTION
* a longitudinal action from Alpamayo's magnitude axis (a TYPE error: theirs is
  magnitude-typed, ours is reason-typed — *"Gentle Deceleration"* cannot say
  whether the ego FOLLOWs a lead or BRAKE_TOs a stop line);
* `ROUTE_TO` from anything EXCEPT one audited path — a direction sign that was
  READ, verified EXTERIOR to the vehicle, and demonstrably FOLLOWED
  (:func:`route_to_from_sign`, PI 2026-08-19). G1 closed it at 0/31 because a
  VLM asked for a destination would INVENT one; a read sign puts the
  destination in the pixels and the future path says which branch was taken, so
  the label becomes two observations rather than a fabrication;
* any label on the 6 clips that leak into val40;
* an ego argument with no VLM referent;
* an anonymous label — every field carries the leg that produced it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tanitad.models.v6 import (STRATEGIC_ACTION_TOKENS, STRATEGIC_GOAL_TOKENS,
                               TACTICAL_GOAL_TOKENS, TACTICAL_LON_ACTIONS,
                               tactical_lat_actions)

__all__ = ["Leg", "LabelField", "TacStrLabel", "ALPAMAYO_LANE_TO_LAT",
           "LON_ADMISSIBLE", "LANE_TARGET_REL", "NOT_APPLICABLE", "ABSTAIN",
           "compose", "lon_is_admissible", "strategic_from_lane_target",
           "route_to_from_sign", "ROUTE_BRANCH"]

ABSTAIN = "ABSTAIN"
NOT_APPLICABLE = "NOT_APPLICABLE"


class Leg(str, Enum):
    """Who produced a field. ⛔ There is no ``UNKNOWN`` — an anonymous label
    cannot have its blast radius computed when it is later retracted."""

    ALPAMAYO = "alpamayo"
    VLM = "vlm"
    EGO = "ego"
    GEOMETRY = "geometry"          # ego path via Engine A (route_from_future_v3)
    DERIVED = "derived"            # deterministic consequence of other fields
    NONE = "none"                  # abstained, with a reason


@dataclass(frozen=True)
class LabelField:
    """A value, the leg that produced it, and — when absent — WHY.

    ``reason`` is mandatory on an abstention. *"The 186 unrepresentable clips
    being DECLARED rather than COERCED"* is the discipline this encodes.
    """

    value: Any
    leg: Leg
    reason: str = ""
    corroborated_by: tuple[Leg, ...] = ()

    def __post_init__(self) -> None:
        if self.value in (None, ABSTAIN, NOT_APPLICABLE) and not self.reason:
            raise ValueError("an absent value MUST carry a reason — a silent "
                             "abstention is indistinguishable from a bug")
        if self.value not in (None, ABSTAIN, NOT_APPLICABLE) and self.leg is Leg.NONE:
            raise ValueError("a present value cannot have leg=NONE")


# --------------------------------------------------------------------------- #
# TIER 1 — Alpamayo
# --------------------------------------------------------------------------- #
#: Alpamayo LANE axis -> `a_tac_lat`. MEASURED over 4,729 clips. Under v6.1 this
#: covers 100 % of parseable rows; ``ABORT_LC`` has no Alpamayo source at all.
ALPAMAYO_LANE_TO_LAT: dict[str, str] = {
    "lane keep": "LANE_KEEP",
    "left lane change": "LANE_CHANGE_L",
    "right lane change": "LANE_CHANGE_R",
    "slightly shift left": "NUDGE_L",
    "slightly shift right": "NUDGE_R",
    "turn left": "TURN_L",          # ⭐ v6.1
    "turn right": "TURN_R",         # ⭐ v6.1
}

#: ⛔ Alpamayo's LONGITUDINAL axis is a **PRIOR, NEVER A LABEL** (§2.2 of the
#: concept). It constrains which reason the VLM may return; a VLM answer outside
#: the set is a FLAGGED DISAGREEMENT routed to review — never silently accepted,
#: never silently dropped.
LON_ADMISSIBLE: dict[str, frozenset[str]] = {
    "strong deceleration": frozenset({"BRAKE_TO", "FOLLOW", "YIELD_MERGE", "CREEP"}),
    "gentle deceleration": frozenset({"BRAKE_TO", "FOLLOW", "YIELD_MERGE", "CREEP"}),
    "maintain speed": frozenset({"CRUISE", "FOLLOW"}),
    "gentle acceleration": frozenset({"CRUISE", "FOLLOW"}),
    "strong acceleration": frozenset({"CRUISE", "FOLLOW"}),
    "stop": frozenset({"HOLD", "BRAKE_TO"}),
}


def lon_is_admissible(alpamayo_lon: str | None, tok: str) -> bool:
    """Does ``tok`` survive Alpamayo's magnitude prior? Unknown magnitude or an
    abstention never constrains — an unmapped prior must not silently reject."""
    if tok in (ABSTAIN, None) or not alpamayo_lon:
        return True
    return tok in LON_ADMISSIBLE.get(alpamayo_lon.strip().lower(),
                                     frozenset(TACTICAL_LON_ACTIONS))


# --------------------------------------------------------------------------- #
# TIER 2 — the VLM's relative lane target, and the cascade it unlocks
# --------------------------------------------------------------------------- #
#: PI 2026-08-18: *"keep target lane simple in a first step: current lane, to
#: the left, to the right"*. Sign convention is the programme's existing declared
#: one (`s2_derive`): **+1 = left, −1 = right** (ego frame +y = left).
LANE_TARGET_REL: dict[str, int] = {"LEFT": +1, "CURRENT": 0, "RIGHT": -1}


def strategic_from_lane_target(rel: str, *, exit_ahead_on_side: bool | None = None
                               ) -> tuple[LabelField, LabelField]:
    """``lane_target_rel`` -> (``g_str``, ``a_str``). The cascade.

    ⭐ Three tokens fall out of ONE visual question, which is the whole leverage
    of the relative encoding:
      * ``LANE_TARGET``        when the target is not the current lane;
      * ``PREPARE_LANE_CHANGE`` — its own rule is *"only when the ROUTE requires
        a lane the ego is not in"*, which under a RELATIVE target reduces to
        ``rel != CURRENT`` and **stops needing a route at all**;
      * ``HOLD_CORRIDOR``     as the complement — and this finally gives that
        token a definition that does not require a map: not *"stay in the
        corridor"* but **"the target lane IS the current lane"**.

    ``exit_ahead_on_side`` is the ONE extra visual predicate ``PREPARE_EXIT``
    needs; ``None`` means the VLM was not asked or abstained, and the action
    then falls back rather than guessing.
    """
    if rel == ABSTAIN:
        r = "VLM abstained on the relative lane target"
        return (LabelField(ABSTAIN, Leg.NONE, r), LabelField(ABSTAIN, Leg.NONE, r))
    if rel not in LANE_TARGET_REL:
        raise ValueError(f"lane_target_rel must be one of "
                         f"{sorted(LANE_TARGET_REL)} or {ABSTAIN}, got {rel!r}")

    if rel == "CURRENT":
        return (LabelField("KEEP_CORRIDOR", Leg.DERIVED),
                LabelField("HOLD_CORRIDOR", Leg.DERIVED))
    if exit_ahead_on_side:
        return (LabelField("EXIT_LEFT" if rel == "LEFT" else "EXIT_RIGHT", Leg.DERIVED),
                LabelField("PREPARE_EXIT", Leg.DERIVED))
    return (LabelField("LANE_TARGET", Leg.DERIVED),
            LabelField("PREPARE_LANE_CHANGE", Leg.DERIVED))


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #
#: ⭐ `ROUTE_TO`'s categorical argument — the branch of the SIGNED route the ego
#: followed. Bounded on purpose: a destination TOPONYM is an unbounded
#: vocabulary and the planner does not need to know the word "Berlin"; what it
#: needs is that this turn was ROUTE-DETERMINED rather than opportunistic, and
#: which way the route went.
ROUTE_BRANCH: tuple[str, ...] = ("left", "straight", "right")


def route_to_from_sign(sign_branch: str | None, *, sign_is_exterior: bool | None,
                       ego_followed_it: bool | None) -> LabelField:
    """⭐ THE PI's MECHANISM — `ROUTE_TO` from a READ sign, not an invented one.

    `ROUTE_TO` was gated because it names a navigation destination and *"no
    destination exists anywhere in the corpus; a VLM asked for one would invent
    it"* (G1 CLOSED at 0/31). A **direction sign puts the destination in the
    pixels**, and the future path says which signed branch was taken — so the
    label becomes two OBSERVATIONS rather than a fabrication.

    ⛔ THREE CONDITIONS, ALL REQUIRED, and the first is not boilerplate.
    MEASURED in this corpus: the sign detector's two highest-confidence
    detections were a **dashboard `30` roundel (0.927)** and a **hoarding
    (0.778)**, both *above* true signs — *"a confidence threshold removes the
    harmless errors and KEEPS the harmful ones"*. The roundel is the EGO
    SPEEDOMETER, i.e. **an ego echo arriving through the vision channel**, which
    a vision-only admissibility audit does not watch. An interior sign is
    therefore a hard refusal, not a low-confidence one.
    """
    if sign_branch in (None, ABSTAIN):
        return LabelField(ABSTAIN, Leg.NONE,
                          "no legible direction sign; ROUTE_TO stays gated "
                          "(G1 closed at 0/31)")
    if sign_branch not in ROUTE_BRANCH:
        raise ValueError(f"sign_branch must be one of {ROUTE_BRANCH} or "
                         f"{ABSTAIN}, got {sign_branch!r}")
    if not sign_is_exterior:
        return LabelField(ABSTAIN, Leg.NONE,
                          "the sign was not verified as EXTERIOR to the vehicle "
                          "— a dashboard/windscreen read is an ego echo through "
                          "the vision channel (measured: roundel 0.927 ranked "
                          "ABOVE true signs)")
    if not ego_followed_it:
        return LabelField(ABSTAIN, Leg.NONE,
                          "a sign was read but the ego did not follow any branch "
                          "it named — the route is not demonstrated")
    return LabelField("ROUTE_TO", Leg.VLM, corroborated_by=(Leg.GEOMETRY,))


@dataclass
class TacStrLabel:
    """One clip's tactical + strategic label, every field with its provenance."""

    clip_id: str
    a_tac_lat: LabelField
    a_tac_lon: LabelField
    g_tac: LabelField
    g_tac_args: dict[str, float] = field(default_factory=dict)
    g_str: LabelField | None = None
    a_str: LabelField | None = None
    flags: list[str] = field(default_factory=list)
    vocab_version: str = "v6.1"

    def to_dict(self) -> dict:
        def f(x: LabelField | None) -> dict | None:
            if x is None:
                return None
            return {"value": x.value, "leg": x.leg.value, "reason": x.reason,
                    "corroborated_by": [c.value for c in x.corroborated_by]}
        return {"clip_id": self.clip_id, "vocab_version": self.vocab_version,
                "a_tac_lat": f(self.a_tac_lat), "a_tac_lon": f(self.a_tac_lon),
                "g_tac": f(self.g_tac), "g_tac_args": self.g_tac_args,
                "g_str": f(self.g_str), "a_str": f(self.a_str),
                "flags": self.flags}


def compose(*, clip_id: str,
            alpamayo_lane: str | None,
            alpamayo_lon: str | None,
            vlm_lon: str | None = None,
            vlm_referent: str | None = None,
            vlm_goal: str | None = None,
            vlm_lane_target_rel: str | None = None,
            vlm_exit_ahead: bool | None = None,
            vlm_sign_branch: str | None = None,
            vlm_sign_is_exterior: bool | None = None,
            vlm_ego_followed_sign: bool | None = None,
            ego_args: dict[str, float] | None = None,
            vocab_version: str = "v6.1") -> TacStrLabel:
    """Compose one label from the three tiers. Pure — no I/O, no model calls.

    Every refusal below is a rule from the approved concept, not a defensive
    check: they are the reason the composition needs no conflict policy.
    """
    lat_vocab = tactical_lat_actions(vocab_version)
    flags: list[str] = []

    # ---- TIER 1: lateral class ------------------------------------------- #
    if alpamayo_lane is None:
        # The 304 one-axis `Stop` rows: the missing axis is NOT-APPLICABLE,
        # never a value to impute. A structural fact about the emitter.
        lat = LabelField(NOT_APPLICABLE, Leg.NONE,
                         "Alpamayo emitted one axis (a stopped vehicle); the "
                         "lateral axis is not-applicable, never imputed")
    else:
        tok = ALPAMAYO_LANE_TO_LAT.get(alpamayo_lane.strip().lower())
        if tok is None:
            lat = LabelField(ABSTAIN, Leg.NONE,
                             f"no v6 lateral token for Alpamayo {alpamayo_lane!r}")
        elif tok not in lat_vocab:
            lat = LabelField(ABSTAIN, Leg.NONE,
                             f"{tok} requires vocabulary v6.1; this build is "
                             f"{vocab_version}")
            flags.append("VOCAB_TOO_OLD_FOR_TOKEN")
        else:
            lat = LabelField(tok, Leg.ALPAMAYO)

    # ---- TIER 2: longitudinal reason, constrained by the Tier-1 prior ---- #
    if vlm_lon in (None, ABSTAIN):
        lon = LabelField(ABSTAIN, Leg.NONE,
                         "no VLM reason; Alpamayo's magnitude axis is "
                         "REASON_REQUIRED and casts no longitudinal vote")
    elif vlm_lon not in TACTICAL_LON_ACTIONS:
        raise ValueError(f"vlm_lon {vlm_lon!r} is outside TACTICAL_LON_ACTIONS")
    elif not vlm_referent:
        # "Evidence before verdict": a reason whose referent cannot be named is
        # not a reason label. Make the failure visible instead of silent.
        lon = LabelField(ABSTAIN, Leg.NONE,
                         f"VLM returned {vlm_lon} with no named referent")
        flags.append("VLM_REASON_WITHOUT_REFERENT")
    elif not lon_is_admissible(alpamayo_lon, vlm_lon):
        lon = LabelField(ABSTAIN, Leg.NONE,
                         f"VLM {vlm_lon} inadmissible under Alpamayo "
                         f"{alpamayo_lon!r} — routed to review")
        flags.append("LON_PRIOR_DISAGREEMENT")
    else:
        lon = LabelField(vlm_lon, Leg.VLM,
                         corroborated_by=(Leg.ALPAMAYO,) if alpamayo_lon else ())

    # ---- TIER 2: the tactical goal token --------------------------------- #
    if vlm_goal in (None, ABSTAIN):
        goal = LabelField(ABSTAIN, Leg.NONE, "no VLM goal token")
    elif vlm_goal not in TACTICAL_GOAL_TOKENS:
        raise ValueError(f"vlm_goal {vlm_goal!r} is outside TACTICAL_GOAL_TOKENS")
    else:
        goal = LabelField(vlm_goal, Leg.VLM)

    # ---- TIER 3: ego arguments — GATED on a VLM referent ------------------ #
    args: dict[str, float] = {}
    if ego_args:
        if goal.value in (ABSTAIN, None) or not vlm_referent:
            # ⛔ THE RULE THAT KEEPS EGO OUT OF ECHO TERRITORY.
            flags.append("EGO_ARGS_DROPPED_NO_REFERENT")
        else:
            args = dict(ego_args)

    # ---- STRATEGIC: the relative-lane-target cascade ---------------------- #
    g_str = a_str = None
    # ⭐ A READ SIGN outranks the lane-target cascade: it carries strictly more
    # information (the turn was ROUTE-DETERMINED, and which way the route went),
    # and it is only reachable when all three sign conditions held.
    route_fld = route_to_from_sign(vlm_sign_branch,
                                   sign_is_exterior=vlm_sign_is_exterior,
                                   ego_followed_it=vlm_ego_followed_sign)
    if route_fld.value == "ROUTE_TO":
        g_str = route_fld
        a_str = LabelField(ABSTAIN, Leg.NONE,
                           "strategic action needs the route's TIMING, which the "
                           "sign read does not supply")
        flags.append("ROUTE_TO_FROM_SIGN")
        flags.append("STRATEGIC_TIMING_FROM_GEOMETRY_REQUIRED")
    elif vlm_lane_target_rel is not None:
        g_str, a_str = strategic_from_lane_target(
            vlm_lane_target_rel, exit_ahead_on_side=vlm_exit_ahead)
    elif alpamayo_lane and alpamayo_lane.strip().lower() in ("turn left", "turn right"):
        # Alpamayo names the TYPE and DIRECTION of a turn; ⛔ it carries NO
        # timestamp (the per-clip record is clip_id/cot/lane/lateral/
        # longitudinal), so geometry must supply the timing and the args. The
        # token is emitted; its placement is not claimed here.
        side = "TURN_LEFT" if "left" in alpamayo_lane.lower() else "TURN_RIGHT"
        g_str = LabelField(side, Leg.ALPAMAYO)
        a_str = LabelField(ABSTAIN, Leg.NONE,
                           "strategic action needs the turn's TIMING, which the "
                           "clip-level Alpamayo record cannot supply")
        flags.append("STRATEGIC_TIMING_FROM_GEOMETRY_REQUIRED")

    # ⛔ ROUTE_TO is reachable through EXACTLY ONE audited path: a sign that was
    # READ, verified EXTERIOR, and demonstrably FOLLOWED (route_to_from_sign).
    # Any other route to it is still the fabrication G1 closed at 0/31.
    for fld in (g_str, a_str):
        if fld is not None and fld.value == "ROUTE_TO" and \
                "ROUTE_TO_FROM_SIGN" not in flags:
            raise ValueError("ROUTE_TO may only be emitted via "
                             "route_to_from_sign() — it names a navigation "
                             "destination, and without a read sign that is a "
                             "guess (G1 closed at 0/31)")
    if g_str is not None and g_str.value not in (ABSTAIN, NOT_APPLICABLE):
        assert g_str.value in STRATEGIC_GOAL_TOKENS, g_str.value
    if a_str is not None and a_str.value not in (ABSTAIN, NOT_APPLICABLE):
        assert a_str.value in STRATEGIC_ACTION_TOKENS, a_str.value

    return TacStrLabel(clip_id=clip_id, a_tac_lat=lat, a_tac_lon=lon, g_tac=goal,
                       g_tac_args=args, g_str=g_str, a_str=a_str, flags=flags,
                       vocab_version=vocab_version)
