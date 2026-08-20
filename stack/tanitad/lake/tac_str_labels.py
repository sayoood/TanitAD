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

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tanitad.models.v6 import (STRATEGIC_ACTION_TOKENS, STRATEGIC_GOAL_TOKENS,
                               TACTICAL_GOAL_TOKENS, TACTICAL_LON_ACTIONS,
                               tactical_lat_actions)

__all__ = ["Leg", "LabelField", "TacStrLabel", "ALPAMAYO_LANE_TO_LAT",
           "LON_ADMISSIBLE", "LANE_TARGET_REL", "NOT_APPLICABLE", "ABSTAIN",
           "ALPAMAYO_COT_REFERENTS", "referents_from_cot", "lon_from_alpamayo",
           "strategic_from_alpamayo", "JUNCTION_REFERENTS", "STOP_AT_REFERENTS",
           "lon_from_ego", "EGO_STOP_EPS_MS", "EGO_CREEP_MAX_MS",
           "magnitude_from_ego", "EGO_MAGNITUDE_BANDS",
           "compose", "lon_is_admissible", "strategic_from_lane_target",
           "lane_target_is_admissible", "LAT_REQUIRES_LANE_TARGET",
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


#: ⛔ MEASURED 2026-08-19 by the first real VLM smoke — the composition DID
#: silently emit an internally contradictory record. Clip 8dc5d14d is Alpamayo
#: `Right Lane Change`; the VLM read the geometry carefully and correctly
#: ("the ego vehicle stays between the same two markings ... no crossing of a
#: lane marking") and returned CURRENT. compose() then produced
#: **a_tac_lat=LANE_CHANGE_R together with g_str=HOLD_CORRIDOR** — "change lane
#: right" and "the target lane IS the current lane" in one label.
#:
#: The docstring claimed the tier refusals meant "the composition needs no
#: conflict policy". That was true for the LONGITUDINAL axis, which has
#: LON_ADMISSIBLE, and false for the LATERAL one, which had no counterpart.
#:
#: ⚠️ The map is deliberately ASYMMETRIC, and the asymmetry is the design:
#:   * a DECLARED lane change constrains the target lane — it must be the lane
#:     being moved into, so CURRENT (or the opposite side) is a contradiction;
#:   * LANE_KEEP constrains NOTHING — a non-current target under a lane keep is
#:     exactly PREPARE_LANE_CHANGE, the informative case, not a conflict;
#:   * turns and the one-axis `Stop` rows are not lane changes and constrain
#:     nothing.
#: A symmetric "they must always agree" rule would destroy the one case the
#: relative encoding was introduced to capture.
LAT_REQUIRES_LANE_TARGET: dict[str, str] = {
    "LANE_CHANGE_L": "LEFT",
    "LANE_CHANGE_R": "RIGHT",
}


def lane_target_is_admissible(lat_token: str | None, rel: str | None) -> bool:
    """Is ``rel`` consistent with the Alpamayo lateral class? Unconstrained
    pairs are admissible — see LAT_REQUIRES_LANE_TARGET on the asymmetry."""
    if lat_token is None or rel is None or rel == ABSTAIN:
        return True
    required = LAT_REQUIRES_LANE_TARGET.get(lat_token)
    return required is None or rel == required


#: ⭐⭐ THE ALPAMAYO `cot` FIELD — 100 % POPULATED, AND IT NAMES THE REFERENT.
#: MEASURED over the 4,729-clip taxonomy: **88.7 % of clips name a referent** and
#: **52.5 % resolve to a SINGLE longitudinal action** from the text alone. We were
#: paying a 9B VLM ~240 s per generation to produce a referent that ships with the
#: dataset, free and deterministic, on every clip — while `compose()` declared
#: *"Alpamayo's magnitude axis is REASON_REQUIRED and casts no longitudinal
#: vote"* and dropped the whole field. (The PI flagged this repeatedly.)
#:
#: Verbatim from the corpus:
#:   "Slow down due to the lead vehicle ahead."     -> FOLLOW    (lead vehicle)
#:   "Keep lane because the road ahead is clear."   -> CRUISE    (clear road)
#:   "Slow down due to the speed bump ahead"        -> BRAKE_TO  (speed bump)
#:
#: ⚠️ `implies` is None where the referent genuinely does NOT determine the
#: longitudinal action — a curve, a parked car, an intersection, or a traffic
#: light whose COLOUR decides it. Those contribute a REFERENT but never an
#: action; guessing there is the nearest-token repair this module forbids.
ALPAMAYO_COT_REFERENTS: tuple[tuple[str, str, str | None], ...] = (
    ("lead_vehicle", r"\blead(ing)? (vehicle|car)\b|\bvehicle ahead\b|\bcar ahead\b", "FOLLOW"),
    ("traffic_light", r"\btraffic light\b|\bred light\b|\bgreen light\b|\btraffic signal\b", None),
    ("stop_sign", r"\bstop sign\b", "BRAKE_TO"),
    ("pedestrian", r"\bpedestrian|\bcrosswalk\b|\bzebra\b|\bcrossing\b", "YIELD_MERGE"),
    ("speed_bump", r"\bspeed bump\b|\bspeed hump\b", "BRAKE_TO"),
    ("curve", r"\bcurve\b|\bcurvature\b|\bbend\b", None),
    ("parked", r"\bparked (car|vehicle|van|truck)\b", None),
    ("cyclist", r"\bcyclist\b|\bbicycl", None),
    ("oncoming", r"\boncoming\b", None),
    ("clear", r"\b(road|lane|path) ahead is clear\b|\bclear (road|path|lane)\b", "CRUISE"),
    ("merge_yield", r"\bmerg|\byield", "YIELD_MERGE"),
    ("queue", r"\bqueue\b|\bstopped traffic\b|\btraffic ahead\b", "FOLLOW"),
    ("intersection", r"\bintersection\b|\bjunction\b", None),
    ("roundabout", r"\broundabout\b", None),
)


def referents_from_cot(cot: str | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """-> (referent names found, longitudinal actions they imply).

    Names and actions are returned SEPARATELY because a clip may name a referent
    that determines nothing (a curve). That still satisfies the referent
    requirement while casting no longitudinal vote.
    """
    if not cot:
        return (), ()
    low = cot.lower()
    names: list[str] = []
    acts: list[str] = []
    for name, pat, implies in ALPAMAYO_COT_REFERENTS:
        if re.search(pat, low):
            names.append(name)
            if implies:
                acts.append(implies)
    return tuple(names), tuple(dict.fromkeys(acts))


def lon_from_alpamayo(alpamayo_lon: str | None, cot: str | None
                      ) -> tuple[str | None, str | None]:
    """-> (longitudinal action, referent phrase), or (None, None).

    ⛔ TWO GATES, BOTH REQUIRED — a derivation, never a guess:
      1. the cot's referents imply EXACTLY ONE action (ambiguity abstains);
      2. that action is ADMISSIBLE under the magnitude (``LON_ADMISSIBLE``), so
         the stated reason and the measured magnitude cannot contradict.
    """
    _names, acts = referents_from_cot(cot)
    if len(acts) != 1:
        return None, None
    act = acts[0]
    if not lon_is_admissible(alpamayo_lon, act):
        return None, None
    return act, (cot or "").strip()


#: ⭐ THE EGO TIER — and the line that makes it admissible.
#:
#: The module rule above says *"ego QUANTIFIES what the VLM has NAMED — it may
#: never fire where the VLM abstained"*. That rule is about **goal ARGUMENTS**:
#: a NUMBER attached to a scene referent ("stop at 12 m") is the driver's
#: idiosyncrasy unless something independently asserted the referent exists.
#:
#: ⚠️ THE DISTINCTION THIS TIER RESTS ON: ``HOLD`` and ``CREEP`` are not reasons,
#: they are **ego KINEMATIC STATES**. "The vehicle is stopped and stays stopped"
#: is a fact about the ego, not a claim about the scene, and it needs no
#: referent to be true. The reason-typed tokens (FOLLOW, YIELD_MERGE, BRAKE_TO)
#: remain closed to ego, because each asserts something about an external object.
#: ⇒ **ego may assert what the ego DID; never why.**
#:
#: PI 2026-08-18 tier-3, verbatim: *"use it only in very clear situations
#: without large interpretation"* — hence the wide dead-band and the refusal to
#: guess anywhere in between.
EGO_STOP_EPS_MS: float = 0.5      # at/below this the vehicle is stopped
EGO_CREEP_MAX_MS: float = 2.0     # ~7 km/h — crawling, not driving

#: ⚠️ **DECLARED, NOT CALIBRATED.** Alpamayo does not publish the thresholds
#: behind its magnitude words, so these bands are OURS. They exist so a magnitude
#: can be measured over **the same window the label describes**, which Alpamayo's
#: own magnitude is not — see `WINDOW_ALIGNMENT_DEFECT.md` §4. Boundaries are
#: m/s², upper-exclusive; anything at or above the last band is strong accel.
EGO_MAGNITUDE_BANDS: tuple[tuple[float, str], ...] = (
    (-1.5, "strong deceleration"),
    (-0.3, "gentle deceleration"),
    (+0.3, "maintain speed"),
    (+1.5, "gentle acceleration"),
)


def magnitude_from_ego(v0_ms: float | None, v_end_ms: float | None,
                       window_s: float = 6.0) -> str | None:
    """-> an Alpamayo-SHAPED magnitude word measured over OUR window, or None.

    ⛔ This is **not** a reproduction of Alpamayo's labelling. It returns a word
    from the same vocabulary so it can be fed to :func:`lon_is_admissible`, and
    nothing more. Its value is that its SUPPORT is the labelled window.

    ⚠️ Permitted because **labels may use ego** (PI 2026-08-03); inference stays
    vision-only. This never runs at inference.
    """
    if v0_ms is None or v_end_ms is None or not window_s:
        return None
    if v_end_ms <= EGO_STOP_EPS_MS:
        return "stop"
    a = (v_end_ms - v0_ms) / window_s
    for hi, word in EGO_MAGNITUDE_BANDS:
        if a < hi:
            return word
    return "strong acceleration"



def lon_from_ego(alpamayo_lon: str | None, v0_ms: float | None,
                 v_end_ms: float | None, *,
                 magnitude_source: str = "alpamayo",
                 window_s: float = 6.0
                 ) -> tuple[str | None, str | None]:
    """-> (longitudinal action, reason) from EGO KINEMATICS alone, or (None, None).

    ⛔ Only the two ego-state tokens are reachable here, and only in the clear
    cases. Anything requiring an external referent stays with the VLM.
    """
    if v0_ms is None or v_end_ms is None:
        return None, None
    mag = (alpamayo_lon or "").strip().lower()
    if v_end_ms <= EGO_STOP_EPS_MS and (mag == "stop" or v0_ms <= EGO_CREEP_MAX_MS):
        tok, why = "HOLD", (f"ego stopped and stays stopped "
                            f"(v_end {v_end_ms:.2f} <= {EGO_STOP_EPS_MS} m/s)")
    elif (EGO_STOP_EPS_MS < v_end_ms <= EGO_CREEP_MAX_MS
            and v0_ms <= EGO_CREEP_MAX_MS):
        tok, why = "CREEP", (f"ego crawls throughout (v0 {v0_ms:.2f}, "
                             f"v_end {v_end_ms:.2f} m/s, both <= "
                             f"{EGO_CREEP_MAX_MS})")
    else:
        return None, None
    # ⚠️ the magnitude prior still binds: an ego reading that contradicts the
    # measured magnitude is a disagreement, not a label.
    #
    # ⛔ WHICH magnitude, though. MEASURED 2026-08-20 (WINDOW_ALIGNMENT_DEFECT.md):
    # Alpamayo's magnitude is measured over an EARLIER interval than the window
    # we label, so it vetoed HOLD on 3 of 3 clips that were provably at rest for
    # the whole window. `magnitude_source="ego"` gates on a magnitude measured
    # over THIS window instead. ⚠️ It stays OFF by default — switching it is the
    # PI's call, not the composer's, and the default keeps every banked label
    # byte-identical.
    gate_mag, gate_src = alpamayo_lon, "alpamayo"
    if magnitude_source == "ego":
        derived = magnitude_from_ego(v0_ms, v_end_ms, window_s)
        if derived is not None:
            gate_mag, gate_src = derived, "ego"
    elif magnitude_source != "alpamayo":
        raise ValueError(f"magnitude_source must be 'alpamayo' or 'ego', "
                         f"got {magnitude_source!r}")
    if not lon_is_admissible(gate_mag, tok):
        return None, None
    if gate_src == "ego":
        why = f"{why}; gated on ego-derived magnitude {gate_mag!r} over {window_s}s"
    return tok, why


#: Alpamayo's `lateral` axis. "Go Straight" is 2,504 of 4,729 clips (53 %) — a
#: ROUTE-level statement the pipeline previously discarded.
ALPAMAYO_GOES_STRAIGHT: frozenset[str] = frozenset({"go straight"})
#: cot referents marking a JUNCTION — the discriminator between "proceeding
#: through a junction" and "continuing along the road".
JUNCTION_REFERENTS: frozenset[str] = frozenset(
    {"intersection", "roundabout", "traffic_light", "stop_sign"})
#: referents that can license STOP_AT — a stop must be AT something.
STOP_AT_REFERENTS: frozenset[str] = frozenset(
    {"traffic_light", "stop_sign", "pedestrian"})


def strategic_from_alpamayo(alpamayo_lane: str | None,
                            alpamayo_lateral: str | None,
                            alpamayo_lon: str | None,
                            cot: str | None
                            ) -> tuple[LabelField | None, LabelField | None]:
    """-> (g_str, a_str) from Alpamayo alone, or (None, None).

    ⭐ WHY THIS EXISTS. Before it the strategic cascade could emit only
    KEEP_CORRIDOR / LANE_TARGET / TURN_* / ROUTE_TO — **5 of the 11
    STRATEGIC_GOAL_TOKENS were unreachable by any path** — and, worse, a
    lane-keep on a main road was labelled ``KEEP_CORRIDOR``, a LANE-level claim
    standing in for a ROUTE-level one. The strategic layer was emitting a
    tactical-grade token, which undercuts the hierarchy thesis at exactly the
    layer meant to demonstrate it.

    This makes STRAIGHT_THROUGH, FOLLOW_MAIN_ROAD and STOP_AT reachable from
    signal Alpamayo already ships. EXIT_LEFT / EXIT_RIGHT still need the exit
    predicate and remain the only gap.

    ⚠️ The junction discriminator is the point: "Go Straight" means
    STRAIGHT_THROUGH only AT a junction; on open road the honest token is
    FOLLOW_MAIN_ROAD. Collapsing them would stamp half the corpus with a
    junction claim it cannot support.
    """
    names, _acts = referents_from_cot(cot)
    lat = (alpamayo_lateral or "").strip().lower()
    lane = (alpamayo_lane or "").strip().lower()
    lon = (alpamayo_lon or "").strip().lower()

    if lon == "stop" and (set(names) & STOP_AT_REFERENTS):
        return (LabelField("STOP_AT", Leg.ALPAMAYO,
                           corroborated_by=(Leg.ALPAMAYO,)),
                LabelField("PREPARE_STOP", Leg.ALPAMAYO,
                           corroborated_by=(Leg.ALPAMAYO,)))

    if lane in ("turn left", "turn right"):
        # ⚠️ CONSOLIDATED HERE 2026-08-20. compose() used to carry a separate
        # alpamayo-turn branch; once this function became the route-level tier
        # that branch was SHADOWED and turns silently lost their g_str. All
        # Alpamayo-derived strategy now lives in one place so a reordering
        # cannot orphan one of its cases again.
        side = "TURN_LEFT" if "left" in lane else "TURN_RIGHT"
        return (LabelField(side, Leg.ALPAMAYO,
                           corroborated_by=(Leg.ALPAMAYO,)),
                LabelField(ABSTAIN, Leg.NONE,
                           "strategic action needs the turn's TIMING, which "
                           "the clip-level Alpamayo record cannot supply"))

    if lane in ("left lane change", "right lane change"):
        # the ACTION is stated; the GOAL still needs a target lane
        return (None,
                LabelField("PREPARE_LANE_CHANGE", Leg.ALPAMAYO,
                           corroborated_by=(Leg.ALPAMAYO,)))

    if lat in ALPAMAYO_GOES_STRAIGHT and lane == "lane keep":
        if set(names) & JUNCTION_REFERENTS:
            return (LabelField("STRAIGHT_THROUGH", Leg.ALPAMAYO,
                               corroborated_by=(Leg.ALPAMAYO,)),
                    LabelField(ABSTAIN, Leg.NONE,
                               "strategic action needs the junction's TIMING, "
                               "which the clip-level record cannot supply"))
        return (LabelField("FOLLOW_MAIN_ROAD", Leg.ALPAMAYO,
                           corroborated_by=(Leg.ALPAMAYO,)),
                LabelField(ABSTAIN, Leg.NONE,
                           "no strategic action implied by continuing along "
                           "the road"))
    return None, None


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
            alpamayo_lateral: str | None = None,
            alpamayo_cot: str | None = None,
            ego_v0_ms: float | None = None,
            ego_v_end_ms: float | None = None,
            allow_ego_lon: bool = False,
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
    check. ⚠️ The claim this docstring used to make — that they meant "the
    composition needs no conflict policy" — was FALSIFIED by the first real VLM
    smoke (2026-08-19): it held for the longitudinal axis, which has
    LON_ADMISSIBLE, and not for the lateral one, which had no counterpart and
    emitted LANE_CHANGE_R together with HOLD_CORRIDOR. The policy is now
    explicit on BOTH axes; see LAT_REQUIRES_LANE_TARGET.
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
        # ⭐ THE ALPAMAYO REASON TIER. This branch used to refuse outright —
        # "Alpamayo's magnitude axis is REASON_REQUIRED and casts no
        # longitudinal vote" — which is true of the MAGNITUDE and threw away the
        # `cot`, a field that names the referent on 88.7 % of clips and
        # determines the action outright on 52.5 %. A VLM abstention, or a
        # generation that never ran, no longer empties a label the dataset
        # already answered.
        _alp_act, _alp_ref = lon_from_alpamayo(alpamayo_lon, alpamayo_cot)
        if _alp_act is not None:
            lon = LabelField(_alp_act, Leg.ALPAMAYO,
                             corroborated_by=(Leg.ALPAMAYO,))
            flags.append("LON_FROM_ALPAMAYO_COT")
        else:
            # ⭐ THE EGO TIER, third and last. Only the two ego-STATE tokens are
            # reachable (HOLD / CREEP) — see lon_from_ego for why that is the
            # admissible line and the reason-typed tokens are not.
            # ⛔ OPT-IN, DEFAULT OFF — and the default is not timidity.
            # `test_ego_never_assigns_a_class_only_arguments` encoded a MEASURED
            # refusal: `vtarget_guarded` was rejected as "hindsight EGO geometry
            # — what speed did this driver settle at, not what speed is
            # permitted here", beating nothing (0.4066 free vs 0.2465 trained).
            # HOLD/CREEP are a narrower case (ego STATE, not a permitted speed)
            # and the PI's binding rule explicitly allows ego for LABEL
            # derivation — but the risk is real and specific:
            # ⚠️ A HOLD label derived from v_end~0 is often predictable from
            # v0~0, which the model IS handed as the integrator constant. That
            # is the ego-speed echo family. ⇒ ANY head trained on
            # LON_FROM_EGO_KINEMATICS labels MUST be run through the v0-shuffle
            # echo control before its number is quotable.
            _ego_act, _ego_why = (lon_from_ego(alpamayo_lon, ego_v0_ms,
                                               ego_v_end_ms)
                                  if allow_ego_lon else (None, None))
            if _ego_act is not None:
                lon = LabelField(_ego_act, Leg.EGO,
                                 corroborated_by=(Leg.ALPAMAYO,)
                                 if alpamayo_lon else ())
                flags.append("LON_FROM_EGO_KINEMATICS")
            else:
                _names, _acts = referents_from_cot(alpamayo_cot)
                lon = LabelField(
                    ABSTAIN, Leg.NONE,
                    f"no VLM reason; Alpamayo cot names {list(_names) or 'nothing'} "
                    f"implying {list(_acts) or 'no single action'}; ego "
                    f"(v0={ego_v0_ms}, v_end={ego_v_end_ms}) not a clear "
                    f"HOLD/CREEP — unresolvable under magnitude {alpamayo_lon!r}")
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
    elif vlm_lane_target_rel in (None, ABSTAIN) or (
            vlm_lane_target_rel == "CURRENT"
            and strategic_from_alpamayo(alpamayo_lane, alpamayo_lateral,
                                        alpamayo_lon, alpamayo_cot)[0] is not None):
        # ⭐ ROUTE-LEVEL BEATS LANE-LEVEL, and the ordering is the whole point.
        # MEASURED 2026-08-20: with the lane-target branch first, this tier was
        # UNREACHABLE — `vlm_lane_target_rel` is never None (the caller passes
        # ABSTAIN), so the cascade always fired and `g_str` came out
        # KEEP_CORRIDOR on every lane-keep clip.
        #
        # ⚠️ KEEP_CORRIDOR is a LANE-level claim ("stay in this lane"). On a
        # lane-keep clip the STRATEGIC fact is the ROUTE one — FOLLOW_MAIN_ROAD
        # on open road, STRAIGHT_THROUGH at a junction. Emitting the lane token
        # in the route slot is the hierarchy collapsing by one level, which is
        # precisely what the strategic layer exists to avoid.
        #
        # A NON-CURRENT target still goes to the cascade below: "the target is
        # the lane to my left" is more specific than "I am following the road",
        # so LANE_TARGET wins there.
        _g_alp, _a_alp = strategic_from_alpamayo(
            alpamayo_lane, alpamayo_lateral, alpamayo_lon, alpamayo_cot)
        if _g_alp is not None or _a_alp is not None:
            g_str, a_str = _g_alp, _a_alp
            flags.append("STRATEGIC_FROM_ALPAMAYO")
            if _a_alp is not None and "TIMING" in (_a_alp.reason or "").upper():
                flags.append("STRATEGIC_TIMING_FROM_GEOMETRY_REQUIRED")
        elif vlm_lane_target_rel is not None:
            g_str, a_str = strategic_from_lane_target(
                vlm_lane_target_rel, exit_ahead_on_side=vlm_exit_ahead)
    elif not lane_target_is_admissible(lat.value, vlm_lane_target_rel):
        # ⛔ Refuse rather than compose a self-contradictory record. Same shape
        # as LON_PRIOR_DISAGREEMENT: the two legs disagree, so the STRATEGIC
        # field abstains WITH ITS REASON and the clip is routed to review. The
        # tactical lateral token is untouched — Alpamayo remains the class
        # authority; what is refused is the strategic claim built on top of a
        # target the class contradicts.
        r = (f"Alpamayo {lat.value} requires lane target "
             f"{LAT_REQUIRES_LANE_TARGET[lat.value]}, VLM read "
             f"{vlm_lane_target_rel} — routed to review")
        g_str = LabelField(ABSTAIN, Leg.NONE, r)
        a_str = LabelField(ABSTAIN, Leg.NONE, r)
        flags.append("LAT_LANE_TARGET_DISAGREEMENT")
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
    else:
        # ⭐ THE ALPAMAYO STRATEGIC TIER — makes STRAIGHT_THROUGH,
        # FOLLOW_MAIN_ROAD and STOP_AT reachable. Before this, 5 of the 11
        # STRATEGIC_GOAL_TOKENS could not be emitted by ANY path, and a lane
        # keep on a main road was labelled KEEP_CORRIDOR — a LANE-level token
        # standing in for a ROUTE-level one.
        _g_alp, _a_alp = strategic_from_alpamayo(
            alpamayo_lane, alpamayo_lateral, alpamayo_lon, alpamayo_cot)
        if _g_alp is not None or _a_alp is not None:
            g_str, a_str = _g_alp, _a_alp
            flags.append("STRATEGIC_FROM_ALPAMAYO")

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
