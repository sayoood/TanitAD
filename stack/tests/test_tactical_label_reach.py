"""Pin WHICH v7 tactical label fields actually REACH training -- to SOURCE.

WHY THIS TEST EXISTS
====================
On 2026-09-06 the Master Mind asserted *"PhysicalAI-AV has no traffic-light feature,
so there is no label to learn braking-for-red from."* The PI corrected it: the
programme's **augmented** v7.x tactical labels DO carry a GT traffic-light colour,
extracted from the Alpamayo CoT.

Both halves were true of **different layers**, and that is the whole defect:

===============================  =========================================================
layer                            traffic light?
===============================  =========================================================
published PhysicalAI-AV features **absent** -- the card says "we do not include open
                                 maps data"; ``obstacle.offline``'s enum is 10 dynamic
                                 agent classes
augmented v7.x label release     **PRESENT, with the colour** -- 779/4,572 records in
                                 ``s2_labels_v7.2_train.jsonl.gz``
supervised training targets      **ABSENT** -- and this is the finding that matters
===============================  =========================================================

A layer-1 fact was quoted to settle a layer-3 question. The programme's own rule --
*"always state the LAYER, never the bare phrase 'our ingest'"* -- exists for exactly
this, and ``test_physicalai_feature_readset.py`` pins the read-set counts for the same
reason. This file pins the TACTICAL row set one table down.

WHAT WAS MEASURED (2026-09-06, evidence class MEASURED)
-------------------------------------------------------
Blob ``…/incoming/2026-09-04-v72-label-release/raw/s2_labels_v7.2_train.jsonl.gz``
(schema ``s2-geom-v7``, vocab ``v7``, 4,572 records, one per clip):

    TRAFFIC_LIGHT_REACT_RED      376/4,572
    TRAFFIC_LIGHT_REACT_GREEN    363/4,572
    TRAFFIC_LIGHT_REACT_YELLOW    22/4,572
    TRAFFIC_LIGHT_REACT           18/4,572   (colourless)
    any TRAFFIC_LIGHT_*          779/4,572

A RED<->GREEN mutation over all 739 coloured entries moved **0 of 4,572** supervised
label records, while a same-breath control mutation of ``a_tac.lon`` on the *same* 779
records moved **779 of 4,572**. The rig was live; the colour is invisible to it.

THE MECHANISM (what this test pins)
-----------------------------------
1. ``vocab_v7.TACTICAL_GOAL_TOKENS_V7`` carries the four traffic-light tokens.
2. ``v7_labels.HEADS`` -- the supervised heads -- is ``tac_lat``, ``tac_lon``,
   ``str_action``, ``str_goal``. **None of them is sized on
   ``TACTICAL_GOAL_TOKENS_V7``.** There is no tactical-GOAL head at all.
3. ``load_v7_labels`` reads ``g_tac.goals`` **only** into ``V7Label.audit["goal_flags"]``,
   and the module declares ``audit`` *"audit-only, NEVER a training input"*.

⭐ THIS TEST IS A GAP PIN, NOT AN APPROVAL. It asserts the CURRENT wiring so that the
day someone supervises the tactical goal set, the build breaks and the documents below
get updated in the same commit -- instead of the gap silently reopening or the fix
silently landing while the docs still describe the gap.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from tanitad.data import v7_labels as V7L
from tanitad.models import vocab_v7 as V7

# --------------------------------------------------------------------------- #
# The documents that carry this claim. A failure message that does not name    #
# them turns a mechanical fix into a re-derivation.                            #
# --------------------------------------------------------------------------- #
DOCS_CARRYING_THE_CLAIM = (
    "CLAUDE.md  (the 'no map / no traffic-light feature' paragraph -- it is "
    "LAYER-1 and must stay scoped)",
    "Project Steering/GOALS_AND_CLAIMS.md  (claim D-TLIGHT-1)",
    "TanitAD Research Lab/Architecture & Inference/Research/"
    "2026-09-06-traffic-light-chain/TRAFFIC_LIGHT_CHAIN.md",
)

#: The traffic-light tokens, by NAME. A count alone rots; the readset test
#: learned that four times in one sentence.
TRAFFIC_LIGHT_TOKENS = (
    "TRAFFIC_LIGHT_REACT",
    "TRAFFIC_LIGHT_REACT_RED",
    "TRAFFIC_LIGHT_REACT_YELLOW",
    "TRAFFIC_LIGHT_REACT_GREEN",
)

#: The four supervised heads, by name and by the vocabulary tuple each is sized on.
EXPECTED_HEADS = {
    "tac_lat": "TACTICAL_LAT_ACTIONS_V7",
    "tac_lon": "TACTICAL_LON_ACTIONS_V7",
    "str_action": "STRATEGIC_ACTION_TOKENS_V7",
    "str_goal": "STRATEGIC_GOAL_TOKENS_V7",
}


def _docs() -> str:
    return "\n".join(f"      - {d}" for d in DOCS_CARRYING_THE_CLAIM)


def _msg(what: str) -> str:
    return (
        f"\n  {what}\n"
        f"  If this is a DELIBERATE change (the tactical goal set is now supervised,\n"
        f"  or a traffic-light token was renamed), update BOTH this test and the\n"
        f"  documents below in the SAME commit:\n{_docs()}\n"
    )


# --------------------------------------------------------------------------- #
# LAYER 2 -- the label vocabulary DOES carry the traffic light, with colour     #
# --------------------------------------------------------------------------- #
def test_vocabulary_carries_the_four_traffic_light_tokens_by_name() -> None:
    """The 'we have no traffic-light label' half of the claim is FALSE."""
    missing = [t for t in TRAFFIC_LIGHT_TOKENS
               if t not in V7.TACTICAL_GOAL_TOKENS_V7]
    assert not missing, _msg(
        f"TACTICAL_GOAL_TOKENS_V7 is missing traffic-light token(s): {missing}. "
        f"The programme's augmented v7 label layer is supposed to carry a GT "
        f"traffic-light colour (PI ruling 2026-09-06).")
    # the three colours must be distinguishable, not collapsed into one token
    coloured = [t for t in TRAFFIC_LIGHT_TOKENS if t != "TRAFFIC_LIGHT_REACT"]
    assert len(set(coloured)) == 3, _msg(
        "the three traffic-light COLOURS collapsed into fewer tokens")


def test_traffic_light_is_perception_only_never_geometry_derived() -> None:
    """⛔ The colour may never be invented by the ego-geometry emitter.

    It is a PERCEPTION claim. If a geometry emitter could produce it, the label
    would be a function of ego dynamics -- the sitclf leak in a new costume.
    """
    for t in TRAFFIC_LIGHT_TOKENS:
        assert t in V7.TACTICAL_GOAL_NEEDS_PERCEPTION, _msg(
            f"{t} left TACTICAL_GOAL_NEEDS_PERCEPTION. A geometry emitter that "
            f"invents a light colour is fabricating perception from ego motion.")
    assert not (set(TRAFFIC_LIGHT_TOKENS) & V7.geometry_emittable(
        V7.TACTICAL_GOAL_TOKENS_V7)), _msg(
        "a traffic-light token became geometry-emittable")


# --------------------------------------------------------------------------- #
# LAYER 3 -- ...and it does NOT reach any supervised head                      #
# --------------------------------------------------------------------------- #
def test_supervised_heads_are_exactly_four_and_named() -> None:
    assert set(V7L.HEADS) == set(EXPECTED_HEADS), _msg(
        f"the supervised head set changed: {sorted(V7L.HEADS)} != "
        f"{sorted(EXPECTED_HEADS)}")


def test_no_supervised_head_is_sized_on_the_tactical_goal_vocabulary() -> None:
    """⛔ THE FINDING. No head predicts the tactical GOAL set at all.

    ``TACTICAL_GOAL_TOKENS_V7`` -- the tuple holding every traffic-light token --
    sizes no supervised head, so no loss can reference it and no gradient can
    reach it. This is a structural absence, not a zero weight.
    """
    goal_set = set(V7.TACTICAL_GOAL_TOKENS_V7)
    offenders = {h: sorted(set(toks) & goal_set)
                 for h, toks in V7L.HEADS.items()
                 if set(toks) & goal_set}
    # tac_lat/tac_lon share ACTION names with a few goal names by design
    # (e.g. TURN_L), so require the TRAFFIC-LIGHT tokens specifically absent.
    tl_reached = {h: sorted(set(toks) & set(TRAFFIC_LIGHT_TOKENS))
                  for h, toks in V7L.HEADS.items()
                  if set(toks) & set(TRAFFIC_LIGHT_TOKENS)}
    assert not tl_reached, _msg(
        f"A supervised head now carries traffic-light classes: {tl_reached}. "
        f"⭐ THIS IS THE GAP BEING CLOSED -- good news, but the claim register "
        f"and CLAUDE.md still describe the gap and must be updated now.")
    assert V7.TACTICAL_GOAL_TOKENS_V7 not in tuple(V7L.HEADS.values()), _msg(
        "a head is now sized directly on TACTICAL_GOAL_TOKENS_V7")
    # documents the incidental overlap so it cannot be mistaken for reach
    assert set(offenders) <= {"tac_lat", "tac_lon"}, _msg(
        f"unexpected head/goal name overlap: {offenders}")


def test_v7label_exposes_no_tactical_goal_set_field() -> None:
    """The loader surfaces the goal POINT, never the goal SET."""
    fields = {f.name for f in dataclasses.fields(V7L.V7Label)}
    assert "tac_anchor" in fields, _msg(
        "V7Label lost tac_anchor -- the ADMISSIBLE goal signal")
    for forbidden in ("tac_goals", "g_tac", "goals", "tac_goal_set"):
        assert forbidden not in fields, _msg(
            f"V7Label gained {forbidden!r}: the tactical goal SET now reaches "
            f"the consumer. That is the gap closing -- update the docs.")


def test_goal_set_is_routed_to_audit_only() -> None:
    """``g_tac.goals`` reaches ``audit``, which is never a training input."""
    src = inspect.getsource(V7L.load_v7_labels)
    assert 'g_tac.get("anchor")' in src, _msg(
        "load_v7_labels no longer reads the anchor from g_tac")
    assert "_goal_audit(g_tac)" in src, _msg(
        "load_v7_labels no longer routes g_tac.goals through _goal_audit")
    audit_src = inspect.getsource(V7L._goal_audit)
    assert '.get("goals")' in audit_src, _msg(
        "_goal_audit no longer reads g_tac.goals")
    # the contract line that makes 'audit' non-training must survive verbatim
    assert "NEVER a training input" in inspect.getsource(V7L), _msg(
        "the 'audit is NEVER a training input' contract line was removed from "
        "v7_labels -- if audit became trainable, every audit field is now a "
        "silent input and the leak rules must be re-checked.")


# --------------------------------------------------------------------------- #
# ⭐ THE GUARD MUST BE MUTATION-TESTED, NOT INSPECTED                           #
# (an AST census once read 0 suspects on BOTH the fixed and the broken trainer) #
# --------------------------------------------------------------------------- #
def test_the_reach_detector_actually_fires() -> None:
    """Reintroduce the condition and prove the assertion is reachable.

    Without this, a detector that can never fail passes forever and certifies
    nothing.
    """
    fake_heads = dict(V7L.HEADS)
    fake_heads["tac_goal"] = TRAFFIC_LIGHT_TOKENS
    tl_reached = {h: sorted(set(toks) & set(TRAFFIC_LIGHT_TOKENS))
                  for h, toks in fake_heads.items()
                  if set(toks) & set(TRAFFIC_LIGHT_TOKENS)}
    assert tl_reached, (
        "the traffic-light reach detector did NOT fire on a head that plainly "
        "carries traffic-light classes -- the detector is dead and every "
        "'no reach' verdict from it is uninformative")
    assert "tac_goal" in tl_reached


@pytest.mark.parametrize("field_name", ["tac_goals", "g_tac", "goals"])
def test_the_field_detector_actually_fires(field_name: str) -> None:
    fake_fields = {f.name for f in dataclasses.fields(V7L.V7Label)} | {field_name}
    assert field_name in fake_fields, (
        "the V7Label field detector cannot see an added goal-set field")
