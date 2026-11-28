"""Cross-family consistency guard for the strategic/tactical label pipeline.

⭐ WHY THIS EXISTS. The 39-clip validation sample (2026-08-23,
`products/P2-data-pipelines/2026-08-23-label-validation-sample/`) found that on
2 of 6 defective clips OUR OWN TACTICAL FAMILY already contradicted the
strategic label, and on a third Alpamayo did. The information needed to catch
those defects was ALREADY IN THE PIPELINE and nothing consumed it. This module
consumes it.

The guard is deliberately CHEAP and LOCAL: it takes labels that have already
been derived plus ego-pose geometry, and returns findings. It derives nothing
itself, so it cannot introduce a new labelling opinion — it can only object to
one the pipeline already formed.

⚠️ SEVERITY IS NOT SUBTLE HERE, so it is typed:
  * ``REFUSE``  — the label contradicts hindsight geometry that the labeller
                  itself had access to. Do not emit; emit a turn token or
                  abstain instead.
  * ``FLAG``    — two families disagree, or the label under-describes the
                  event. Emit, but record the finding for review.
Callers decide what to do with each; the guard never mutates a label.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# The yaw a manoeuvre must reach before "the ego went straight" is untenable.
# Matches the validation harness and refb_labels' turn gate.
TURN_DEG = 25.0
# A speed gain that makes "preparing to stop" untenable.
DV_ACCEL_MS = 2.0
# Below this the ego is treated as at rest (refb_labels.STOP_V_MS lineage).
V_REST_MS = 1.0
# A stop worth describing: the ego was actually moving beforehand.
V_WAS_MOVING_MS = 3.0

Severity = Literal["REFUSE", "FLAG"]

# g_str tokens that ASSERT no junction manoeuvre happens on the horizon.
_NO_MANOEUVRE_GOALS = frozenset({"FOLLOW_MAIN_ROAD", "NONE_ABSTAIN",
                                 "STRAIGHT_THROUGH", "KEEP_CORRIDOR"})
_TURN_SIGN = {"TURN_LEFT": +1, "TURN_RIGHT": -1}
_TAC_SIGN = {"turn_left": +1, "turn_right": -1}


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: Severity
    message: str


@dataclass
class GuardReport:
    clip_id: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return any(f.severity == "REFUSE" for f in self.findings)

    @property
    def clean(self) -> bool:
        return not self.findings

    def as_dict(self) -> dict:
        return {"clip_id": self.clip_id, "refused": self.refused,
                "findings": [{"rule": f.rule, "severity": f.severity,
                              "message": f.message} for f in self.findings]}


def check(*, clip_id: str, g_str: str | None, a_str: str | None,
          peak_yaw_deg: float, v_at_key_ms: float, v_end_ms: float,
          v_min_future_ms: float, tac_lat: str | None = None,
          tac_lon: str | None = None, lateral_class: str | None = None,
          stop_type: str | None = None) -> GuardReport:
    """Run every consistency rule over one clip's already-derived labels.

    ``peak_yaw_deg`` is the signed yaw excursion of largest magnitude on the
    available horizon, measured from the key frame (+ = left). All speeds m/s.
    ``tac_lat``/``tac_lon`` are the factored tactical classes when available;
    the cross-family rules simply do not fire when they are None.
    """
    out = GuardReport(clip_id=clip_id)
    dv = v_end_ms - v_at_key_ms
    stops = v_min_future_ms < 0.5

    # -- G1: the fallback token must not absorb an undetected junction turn ---
    # This is the sample's ONE systematic defect: FOLLOW_MAIN_ROAD ran 70 %
    # (7/10) and all three misses were 63-72 deg turns. Because it is the
    # FALLBACK, a missed turn becomes a confident "carry on straight" rather
    # than an abstention — the same substitute-a-confident-claim failure the
    # lane-change gate was retired for.
    # ⭐ PREFER THE MANOEUVRE CLASSIFIER WHEN THE CALLER SUPPLIES IT.
    # Raw yaw CANNOT separate a junction turn from a curving main road, and
    # refusing a correct FOLLOW_MAIN_ROAD on a bend would repeat exactly the
    # LANE_TARGET mistake. `ego_manoeuvre.analyse` makes that split on
    # instantaneous curvature (kappa = omega/v), so when its verdict is
    # available it decides; the yaw form remains only as a fallback for callers
    # that have not been migrated.
    if g_str in _NO_MANOEUVRE_GOALS:
        if lateral_class is not None:
            if lateral_class.startswith("JUNCTION_TURN"):
                out.findings.append(Finding(
                    "G1-fallback-absorbs-turn", "REFUSE",
                    f"{g_str} asserts no junction manoeuvre, but the ego "
                    f"executes {lateral_class} ({peak_yaw_deg:+.1f} deg). Emit "
                    f"a turn token or abstain."))
        elif abs(peak_yaw_deg) >= TURN_DEG:
            out.findings.append(Finding(
                "G1-fallback-absorbs-turn", "REFUSE",
                f"{g_str} asserts no junction manoeuvre, but the hindsight path "
                f"turns {peak_yaw_deg:+.1f} deg on the horizon. Emit a turn "
                f"token or abstain."))

    # -- G6: a stop the strategic layer does not distinguish ------------------
    # PI 2026-08-23: "we need better detection prepare-to-stop to differentiate
    # from traffic jam situations at least strategically." A queue and a red
    # light look identical for the first seconds; what separates them is
    # REPETITION (a jam stops repeatedly) and RECOVERY (a light releases to
    # cruise). PREPARE_STOP over a QUEUE is a different strategic situation and
    # must not be emitted as if it were a controlled stop.
    if a_str == "PREPARE_STOP" and stop_type == "QUEUE":
        out.findings.append(Finding(
            "G6-queue-vs-controlled-stop", "FLAG",
            "PREPARE_STOP over STOP_AND_GO/QUEUE kinematics — this is "
            "queueing traffic, not a controlled stop; the strategic layer "
            "should distinguish them."))

    # -- G2: strategic and tactical families disagree on the DIRECTION -------
    # ⚠️ Only a SIGN CONFLICT is a finding. A strategic TURN_* with tactical
    # lane_keep is NOT one: tactical spans 2.0 s and 10 of 21 sampled turns
    # begin later than that (median onset 3.7 s). Scoring that as a defect
    # would manufacture a ~50 % false failure rate.
    gs, ts = _TURN_SIGN.get(g_str or ""), _TAC_SIGN.get(tac_lat or "")
    if gs and ts and gs != ts:
        out.findings.append(Finding(
            "G2-lat-sign-conflict", "FLAG",
            f"strategic {g_str} but tactical {tac_lat} — the two families "
            f"disagree on turn direction."))

    # -- G3: longitudinal inversion ------------------------------------------
    if a_str == "PREPARE_STOP" and v_at_key_ms < V_REST_MS and dv > DV_ACCEL_MS:
        out.findings.append(Finding(
            "G3-lon-inverted", "REFUSE",
            f"PREPARE_STOP while accelerating away from rest "
            f"({v_at_key_ms:.2f} -> {v_end_ms:.2f} m/s). RESUME_CRUISE is the "
            f"token this profile carries elsewhere in the corpus."))
    if a_str == "RESUME_CRUISE" and dv < 0:
        out.findings.append(Finding(
            "G3-lon-inverted", "FLAG",
            f"RESUME_CRUISE while decelerating ({dv:+.2f} m/s)."))

    # -- G4: an action that fails to describe a full stop ---------------------
    if a_str == "HOLD_CORRIDOR" and stops and v_at_key_ms > V_WAS_MOVING_MS:
        out.findings.append(Finding(
            "G4-stop-undescribed", "FLAG",
            f"HOLD_CORRIDOR through a full stop from {v_at_key_ms:.2f} m/s — "
            f"the action does not describe the dominant longitudinal event."))

    # -- G5: tactical LON contradicts the action family ----------------------
    if a_str == "PREPARE_STOP" and tac_lon == "accelerate":
        out.findings.append(Finding(
            "G5-lon-family-conflict", "FLAG",
            "strategic PREPARE_STOP but tactical LON reads accelerate."))
    return out


def check_many(rows: list[dict]) -> list[GuardReport]:
    """Convenience batch wrapper; each row is a ``check`` kwargs dict."""
    return [check(**r) for r in rows]
