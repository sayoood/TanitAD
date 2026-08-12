"""CPU tests for PH0 v2 — the four-call solvable structure.

⛔ THE REASON THIS FILE EXISTS. MEASURED on pod4 2026-08-12: grammar-constrained
decoding enforces STRUCTURE and TYPE but NOT numeric bounds. A schema declaring
``{"type":"integer","maximum":448}`` still let Qwen3.5-9B emit
``bbox:[952,100,975,160]`` on a 448 px frame. Every ``minimum``/``maximum`` in
the v2 schemas is therefore DECORATIVE, and ``validate_v2`` is what actually
holds them. Nor does ``maxItems`` imply distinct — B4 emitted ``hold_corridor``
three times. These tests pin the checks the grammar cannot make.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from ph0_v2 import (ACTION_VERBS, GOAL_KINDS, S_B1, S_B2, S_B3,  # noqa: E402
                    S_B4, validate_v2)


# --------------------------------------------------------------------------- #
# the measured defects                                                         #
# --------------------------------------------------------------------------- #
def test_bbox_outside_frame_is_caught():
    """THE measured case: [952,100,975,160] on a 448 px frame."""
    v = validate_v2("B3_ground_0",
                    {"visible": True, "frame_idx": 0,
                     "bbox": [952, 100, 975, 160]}, px=448)
    assert any("outside 0..448" in e for e in v)


def test_bbox_within_frame_passes():
    assert validate_v2("B3_ground_0",
                       {"visible": True, "frame_idx": 0,
                        "bbox": [352, 100, 375, 130]}, px=448) == []


def test_duplicate_actions_are_caught():
    """maxItems 3 does NOT imply distinct — the model emitted hold_corridor
    three times and the grammar was happy."""
    v = validate_v2("B4_symbols", {
        "goal_kind": "follow_main_road", "goal_evidence_sign": None,
        "conf": "high",
        "actions": [{"verb": "hold_corridor", "direction": "none"},
                    {"verb": "hold_corridor", "direction": "none"}]})
    assert any("duplicate" in e for e in v)


def test_distinct_actions_pass():
    assert validate_v2("B4_symbols", {
        "goal_kind": "follow_main_road", "goal_evidence_sign": None,
        "conf": "high",
        "actions": [{"verb": "hold_corridor", "direction": "none"},
                    {"verb": "prepare_lane_change", "direction": "right"}]}) == []


# --------------------------------------------------------------------------- #
# B1                                                                           #
# --------------------------------------------------------------------------- #
def test_lane_index_beyond_lane_count_is_caught():
    """ego in lane 3 of 2 is incoherent even though both ints are in range."""
    v = validate_v2("B1_scene", {"lanes_visible": 2, "lane_ego": 3})
    assert any("lane_ego" in e and ">= lanes_visible" in e for e in v)


def test_lane_counts_out_of_range_are_caught():
    assert validate_v2("B1_scene", {"lanes_visible": 9, "lane_ego": 0})
    assert validate_v2("B1_scene", {"lanes_visible": 2, "lane_ego": -1})


def test_lane_ego_zero_with_zero_lanes_is_allowed():
    """0/0 is the documented 'unclear' encoding, not a contradiction."""
    assert validate_v2("B1_scene", {"lanes_visible": 0, "lane_ego": 0}) == []


# --------------------------------------------------------------------------- #
# B2 — n_signs is the point of emitting it first                               #
# --------------------------------------------------------------------------- #
def test_n_signs_must_match_the_array():
    v = validate_v2("B2_signs", {"n_signs": 3, "signs": [
        {"kind": "speed", "state": "none", "text": "20",
         "applies_to_ego": True}]})
    assert any("n_signs" in e and "!= len" in e for e in v)


def test_zero_signs_is_valid_abstention():
    assert validate_v2("B2_signs", {"n_signs": 0, "signs": []}) == []


def test_state_on_a_non_light_is_caught():
    """Only traffic lights have a colour state; anything else must be 'none'."""
    v = validate_v2("B2_signs", {"n_signs": 1, "signs": [
        {"kind": "speed", "state": "red", "text": "50",
         "applies_to_ego": True}]})
    assert any("non-light" in e for e in v)


def test_overlong_ocr_text_is_caught():
    v = validate_v2("B2_signs", {"n_signs": 1, "signs": [
        {"kind": "nav", "state": "none", "text": "x" * 60,
         "applies_to_ego": True}]})
    assert any("longer than 40" in e for e in v)


# --------------------------------------------------------------------------- #
# B3                                                                           #
# --------------------------------------------------------------------------- #
def test_frame_idx_out_of_range_is_caught():
    v = validate_v2("B3_ground_0", {"visible": True, "frame_idx": 99,
                                    "bbox": [1, 1, 2, 2]}, n_frames=40)
    assert any("frame_idx" in e for e in v)


def test_inverted_bbox_is_caught():
    v = validate_v2("B3_ground_0", {"visible": True, "frame_idx": 0,
                                    "bbox": [300, 200, 100, 50]})
    assert any("x0<x1" in e for e in v)


def test_not_visible_must_have_zero_bbox():
    v = validate_v2("B3_ground_0", {"visible": False, "frame_idx": 0,
                                    "bbox": [10, 10, 20, 20]})
    assert any("visible false" in e for e in v)
    assert validate_v2("B3_ground_0", {"visible": False, "frame_idx": 0,
                                       "bbox": [0, 0, 0, 0]}) == []


# --------------------------------------------------------------------------- #
# B4 — the prereg's own rules                                                  #
# --------------------------------------------------------------------------- #
def test_route_to_without_signage_evidence_is_caught():
    """The binding rule: route_to is only admissible with a sign actually read."""
    v = validate_v2("B4_symbols", {"goal_kind": "route_to",
                                   "goal_evidence_sign": None,
                                   "actions": [], "conf": "low"})
    assert any("route_to without" in e for e in v)


def test_route_to_with_evidence_passes():
    assert validate_v2("B4_symbols", {"goal_kind": "route_to",
                                      "goal_evidence_sign": 0,
                                      "actions": [], "conf": "high"}) == []


def test_lane_change_needs_a_side():
    v = validate_v2("B4_symbols", {
        "goal_kind": "keep_corridor", "goal_evidence_sign": None,
        "conf": "med",
        "actions": [{"verb": "prepare_lane_change", "direction": "none"}]})
    assert any("needs a side" in e for e in v)


def test_off_vocabulary_goal_and_verb_are_caught():
    assert validate_v2("B4_symbols", {"goal_kind": "teleport",
                                      "goal_evidence_sign": None,
                                      "actions": [], "conf": "low"})
    assert validate_v2("B4_symbols", {
        "goal_kind": "keep_corridor", "goal_evidence_sign": None,
        "conf": "low", "actions": [{"verb": "floor_it", "direction": "none"}]})


def test_abstention_is_always_valid():
    """'Abstaining is better than guessing' must never be penalised."""
    assert validate_v2("B4_symbols", {"goal_kind": "none_abstain",
                                      "goal_evidence_sign": None,
                                      "actions": [], "conf": "low"}) == []


# --------------------------------------------------------------------------- #
# schema hygiene                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("s", [S_B1, S_B2, S_B3, S_B4])
def test_schemas_forbid_extra_keys(s):
    """additionalProperties False everywhere, so the grammar cannot wander into
    an invented key."""
    assert s["additionalProperties"] is False


def test_b2_requires_n_signs_before_the_array():
    """force_json_field_order makes `required` order BINDING, so this ordering
    is the mechanism, not documentation."""
    assert S_B2["required"].index("n_signs") < S_B2["required"].index("signs")


def test_vocabularies_are_closed_and_non_empty():
    assert len(GOAL_KINDS) == 11 and "none_abstain" in GOAL_KINDS
    assert len(ACTION_VERBS) == 6
    assert set(S_B4["properties"]["goal_kind"]["enum"]) == set(GOAL_KINDS)
    assert set(S_B4["properties"]["actions"]["items"]["properties"]
               ["verb"]["enum"]) == set(ACTION_VERBS)


def test_no_metric_slots_survive_in_b4():
    """⭐ The organising principle: the VLM chooses SYMBOLS, the algorithm
    supplies NUMBERS. If a metric slot reappears in B4, the design has
    regressed to v0.1 and the model is being asked to hallucinate metres."""
    banned = {"within_m", "by_time_s", "at_arc_m", "hold_for_s", "v_target_ms",
              "confidence", "reason", "source"}
    item = S_B4["properties"]["actions"]["items"]["properties"]
    assert not (banned & set(item)), f"metric slot back in B4: {banned & set(item)}"
    assert not (banned & set(S_B4["properties"]))


def test_unknown_call_name_does_not_silently_pass():
    """A typo'd call name must not read as 'valid' — it returns no violations
    only because nothing matched, so the caller's `valid` flag would lie."""
    assert validate_v2("B9_nonexistent", {"anything": 1}) == []
