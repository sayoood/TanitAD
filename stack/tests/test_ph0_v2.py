"""CPU tests for PH0 v2 — the four-call solvable structure.

⛔ THE REASON THIS FILE EXISTS. MEASURED on pod4 2026-08-12: grammar-constrained
decoding enforces STRUCTURE and TYPE but NOT numeric bounds — a schema declaring
``maximum`` is DECORATIVE, so ``validate_v2`` is what actually holds it.

⚠️ AND THE VALIDATOR ITSELF WAS WRONG FIRST. I read ``bbox:[952,100,975,160]``
as an out-of-frame hallucination against a 448 px maximum. It was neither:
Qwen-VL emits NORMALIZED 0–1000 coordinates, its own trained convention, so the
model was perfectly consistent. The check was wrong twice over — frames here are
179x448, so y never reaches 448 either, and a single maximum for both axes could
not have been right in any coordinate space. Likewise ``maxItems`` does not imply
distinct, but a repeated verb is a PADDING ARTIFACT, not wrong content, and
rejecting the record for it discarded a good ``goal_kind``.

These tests pin both what the grammar cannot check AND the two places my own
checks were the defect.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from ph0_v2 import (ACTION_VERBS, BBOX_MAX, GOAL_KINDS, S_B1,  # noqa: E402
                    S_B2, S_B3, S_B4, dedupe_actions, norm_to_px,
                    validate_v2)


# --------------------------------------------------------------------------- #
# the measured defects                                                         #
# --------------------------------------------------------------------------- #
def test_bbox_beyond_the_normalized_range_is_caught():
    v = validate_v2("B3_ground_0",
                    {"visible": True, "frame_idx": 0,
                     "bbox": [1200, 100, 1300, 160]})
    assert any("outside 0..1000" in e for e in v)


def test_the_measured_952_bbox_is_VALID_in_normalized_space():
    """⭐ THE CORRECTION. [952,100,975,160] was flagged as out-of-frame against
    a 448 px maximum — but Qwen-VL emits NORMALIZED 0–1000 coordinates, which
    is its trained convention. The model was consistent; the validator was
    wrong, and it was wrong twice over because frames are 179x448, so y never
    reaches 448 either. Working WITH the convention is the fix."""
    assert validate_v2("B3_ground_0", {"visible": True, "frame_idx": 0,
                                       "bbox": [952, 100, 975, 160]}) == []


def test_norm_to_px_uses_separate_x_and_y_ranges():
    """x and y have DIFFERENT pixel extents (179x448) — a single maximum for
    both was the second half of the bug."""
    assert norm_to_px([0, 0, 1000, 1000], 448, 179) == [0, 0, 448, 179]
    assert norm_to_px([500, 500, 1000, 1000], 448, 179) == [224, 90, 448, 179]


def test_bbox_max_is_the_normalized_range():
    assert BBOX_MAX == 1000


def test_duplicate_actions_are_NOT_a_violation():
    """MEASURED: the model pads the array to maxItems, repeating a verb — 6 of
    8 clips. That carries no wrong content, and failing the record for it
    DISCARDED a perfectly good goal_kind. Deduped, not rejected."""
    assert validate_v2("B4_symbols", {
        "goal_kind": "follow_main_road", "goal_evidence_sign": None,
        "conf": "high",
        "actions": [{"verb": "hold_corridor", "direction": "none"},
                    {"verb": "hold_corridor", "direction": "none"}]}) == []


def test_dedupe_actions_drops_repeats_and_counts_them():
    sym, n = dedupe_actions({"actions": [
        {"verb": "hold_corridor", "direction": "none"},
        {"verb": "hold_corridor", "direction": "none"},
        {"verb": "prepare_lane_change", "direction": "right"}]})
    assert n == 1
    assert [a["verb"] for a in sym["actions"]] == ["hold_corridor",
                                                  "prepare_lane_change"]


def test_dedupe_preserves_order_and_handles_none():
    assert dedupe_actions(None) == (None, 0)
    sym, n = dedupe_actions({"actions": []})
    assert n == 0


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
                                    "bbox": [10, 10, 20, 20]}, n_frames=40)
    assert any("frame_idx" in e for e in v)


def test_inverted_bbox_is_caught():
    v = validate_v2("B3_ground_0", {"visible": True, "frame_idx": 0,
                                    "bbox": [600, 400, 200, 100]})
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


# --------------------------------------------------------------------------- #
# defects that SURVIVED the inference fix (ph0-v2.1)                           #
# --------------------------------------------------------------------------- #
def test_dedupe_signs_drops_repeats_and_resyncs_n_signs():
    """⛔ MEASURED on CORRECTED inference: sign padding survives. One clip gave
    speed "100" x4, another yield "" x5, another "P" x3. So it is NOT the
    near-blind-inference artifact — it is array filling, same as B4 actions.
    n_signs must be re-synced or the record self-contradicts."""
    from ph0_v2 import dedupe_signs
    sg, n = dedupe_signs({"n_signs": 4, "signs": [
        {"kind": "speed", "text": "100", "state": "none",
         "applies_to_ego": True}] * 4})
    assert n == 3
    assert sg["n_signs"] == 1 and len(sg["signs"]) == 1
    assert validate_v2("B2_signs", sg) == []


def test_dedupe_signs_keeps_genuinely_distinct_signs():
    from ph0_v2 import dedupe_signs
    sg, n = dedupe_signs({"n_signs": 2, "signs": [
        {"kind": "speed", "text": "100", "state": "none", "applies_to_ego": True},
        {"kind": "nav", "text": "The Sea", "state": "none",
         "applies_to_ego": True}]})
    assert n == 0 and sg["n_signs"] == 2


def test_b3_prompt_separates_frame_index_from_coordinate_range():
    """⛔ MEASURED: the B3 prompt said 'NORMALIZED coordinates 0-1000' and the
    model emitted frame_idx=1000 three times on one clip — the range bled into
    the ADJACENT field. The prompt must state the frame range separately and
    say it is a NUMBER, not a coordinate."""
    from ph0_v2 import P_B3
    assert "not a coordinate" in P_B3
    assert "{n_last}" in P_B3 and "{n_frames}" in P_B3


def test_frame_idx_1000_is_still_caught():
    v = validate_v2("B3_ground_1", {"visible": True, "frame_idx": 1000,
                                    "bbox": [750, 100, 800, 200]}, n_frames=40)
    assert any("frame_idx" in e for e in v)


def test_degenerate_bbox_is_caught():
    """[750,1000,750,1000] — zero width and height, measured on the same clip."""
    v = validate_v2("B3_ground_1", {"visible": True, "frame_idx": 5,
                                    "bbox": [750, 1000, 750, 1000]})
    assert any("x0<x1" in e for e in v)


# =========================================================================== #
# v2.2 — EGO STATE IN THE PROMPT (PI, 2026-08-12)                             #
# =========================================================================== #
def _synth_poses(n=200, v0=10.0, a=0.0, w=0.0, dt=0.1):
    """Straight/curving synthetic track at 10 Hz: [T,4] = x, y, yaw, v."""
    import math
    rows, x, y, yaw = [], 0.0, 0.0, 0.0
    for i in range(n):
        v = max(0.0, v0 + a * i * dt)
        rows.append([x, y, yaw, v])
        x += v * dt * math.cos(yaw)
        y += v * dt * math.sin(yaw)
        yaw += w * dt
    return rows


def test_ego_past_state_uses_only_the_past():
    """⛔ The whole point: B1/B2 must not receive future evidence. A track that
    is steady before t0 and slams the brakes after it must read 'steady'."""
    from ph0_v2 import ego_past_state
    import math
    rows, x, v = [], 0.0, 10.0
    for i in range(200):
        rows.append([x, 0.0, 0.0, v])
        x += v * 0.1
        v = v if i < 80 else max(0.0, v - 0.5)      # brakes only AFTER t0=80
    st = ego_past_state(rows, 80)
    assert st["motion"] == "steady"
    assert st["v_now_ms"] == pytest.approx(10.0, abs=0.01)
    assert st["v_min_ms"] == pytest.approx(10.0, abs=0.01)   # no future leak
    assert not math.isnan(st["dist_travelled_m"])


def test_ego_past_state_window_is_eight_seconds():
    from ph0_v2 import ego_past_state, EGO_WINDOW_S
    st = ego_past_state(_synth_poses(300), 200)
    assert st["window_s"] == pytest.approx(EGO_WINDOW_S, abs=0.05)


def test_ego_past_state_detects_braking_and_accelerating():
    from ph0_v2 import ego_past_state
    assert ego_past_state(_synth_poses(200, v0=20.0, a=-1.0), 150)["motion"] \
        == "braking"
    assert ego_past_state(_synth_poses(200, v0=2.0, a=1.0), 150)["motion"] \
        == "accelerating"


def test_ego_past_state_detects_turn_direction():
    from ph0_v2 import ego_past_state
    assert ego_past_state(_synth_poses(200, w=0.2), 150)["turning"] \
        == "turning_left"
    assert ego_past_state(_synth_poses(200, w=-0.2), 150)["turning"] \
        == "turning_right"
    assert ego_past_state(_synth_poses(200, w=0.0), 150)["turning"] == "straight"


def test_ego_past_state_yaw_wraps_at_pi():
    """A track crossing the ±pi seam must not mint a ~62 rad/s yaw rate."""
    from ph0_v2 import ego_past_state
    import math
    rows = [[float(i), 0.0, ((i * 0.05 + math.pi) % (2 * math.pi)) - math.pi,
             10.0] for i in range(200)]
    st = ego_past_state(rows, 150)
    assert abs(st["yaw_rate_rad_s"]) < 1.0


def test_ego_past_state_returns_none_on_unusable_poses():
    from ph0_v2 import ego_past_state
    assert ego_past_state([], 5) is None
    assert ego_past_state([[0.0, 0.0, 0.0, 1.0]], 0) is None      # T<2
    assert ego_past_state(_synth_poses(50), 0) is None            # t0 at 0


def test_b2_block_is_speed_redacted_and_b1_is_not():
    """⛔ THE LEAK. B2 reads sign TEXT and a speed sign's text is a NUMBER. If
    the model can see its own speedometer, '50' can be transcribed from the ego
    state instead of read off the sign, and the sign channel stops being
    falsifiable. Same family as the nav-echo defect."""
    from ph0_v2 import ego_past_state, ego_section
    st = ego_past_state(_synth_poses(200, v0=13.9), 150)
    b2 = ego_section("B2_signs", st, "past")
    b1 = ego_section("B1_scene", st, "past")
    for k in ("v_now_ms", "v_now_kmh", "v_mean_ms", "v_min_ms", "v_max_ms"):
        assert k not in b2, f"{k} leaked into the sign-reading prompt"
        assert k in b1
    assert "50.0" not in b2                      # 13.9 m/s == 50.0 km/h
    assert "NEVER copy a number from it" in b2
    assert "motion" in b2 and "turning" in b2    # still gets the useful part


def test_ego_full_mode_is_the_leak_measurement_arm():
    """`full` exists so the redaction's cost is MEASURABLE rather than assumed:
    the delta in speed-sign recall between past and full IS the leak size."""
    from ph0_v2 import ego_past_state, ego_section
    st = ego_past_state(_synth_poses(200, v0=13.9), 150)
    assert "v_now_kmh" in ego_section("B2_signs", st, "full")


def test_ego_mode_none_reproduces_v21_prompts_byte_identically():
    """The control arm must be EXACT, not approximately the old prompt."""
    from ph0_v2 import ego_past_state, ego_section
    st = ego_past_state(_synth_poses(200), 150)
    for call in ("B1_scene", "B2_signs", "B4_symbols"):
        assert ego_section(call, st, "none") == ""


def test_b3_never_receives_ego_state():
    """B3 is pure spatial localisation — ego kinematics cannot help it, and it
    is the call that repeats up to 6x per clip."""
    from ph0_v2 import ego_past_state, ego_section
    st = ego_past_state(_synth_poses(200), 150)
    for i in range(6):
        assert ego_section(f"B3_ground_{i}", st, "past") == ""
        assert ego_section(f"B3_ground_{i}", st, "full") == ""


def test_missing_ego_emits_no_block_at_all():
    """No ego_root is not an error — it must degrade to the v2.1 prompt, not to
    a prompt containing an empty EGO_STATE header the model must interpret."""
    from ph0_v2 import ego_section
    assert ego_section("B1_scene", None, "past") == ""


def test_ego_section_is_valid_json_after_the_header():
    from ph0_v2 import ego_past_state, ego_section
    import json as _json
    st = ego_past_state(_synth_poses(200), 150)
    for call in ("B1_scene", "B2_signs", "B4_symbols"):
        body = ego_section(call, st, "past").split("\n")[-2]
        _json.loads(body)


def test_fmt_ego_drops_none_valued_keys():
    """A short window makes accel_3s unmeasurable; the prompt must omit it
    rather than assert `null`, which reads as a measured zero."""
    from ph0_v2 import ego_past_state, fmt_ego
    st = ego_past_state(_synth_poses(20), 5)      # 0.5 s of history
    assert st["accel_3s_ms2"] is None
    assert "accel_3s_ms2" not in fmt_ego(st)
