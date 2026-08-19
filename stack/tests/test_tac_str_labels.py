"""Every rule the PI approved, pinned — or it is not a rule.

The concept (`…/2026-08-18-alpamayo-screening/TACTICAL_STRATEGIC_LABEL_CONCEPT.md`,
approved 2026-08-18) is prose. `tac_str_labels.py` is the executable form. These
tests are what keeps the two from drifting — C126's lesson: a correction that
lives only in prose decays, because no prose can reach a glob.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.lake.tac_str_labels import (ABSTAIN, ALPAMAYO_LANE_TO_LAT,
                                         LANE_TARGET_REL, LON_ADMISSIBLE,
                                         NOT_APPLICABLE, LabelField, Leg,
                                         compose, lon_is_admissible,
                                         strategic_from_lane_target)
from tanitad.models.v6 import (STRATEGIC_ACTION_TOKENS, STRATEGIC_GOAL_TOKENS,
                               TACTICAL_LON_ACTIONS)


def _c(**kw):
    base = dict(clip_id="c0", alpamayo_lane="Lane Keep",
                alpamayo_lon="Maintain Speed")
    base.update(kw)
    return compose(**base)


# ------------------------------------------------- TIER 1: the lateral class --
@pytest.mark.parametrize("lane,tok", [
    ("Lane Keep", "LANE_KEEP"), ("Left Lane Change", "LANE_CHANGE_L"),
    ("Right Lane Change", "LANE_CHANGE_R"), ("Slightly Shift Left", "NUDGE_L"),
    ("Slightly Shift Right", "NUDGE_R"), ("Turn Left", "TURN_L"),
    ("Turn Right", "TURN_R")])
def test_every_alpamayo_lane_value_maps_under_v6_1(lane, tok):
    out = _c(alpamayo_lane=lane)
    assert out.a_tac_lat.value == tok
    assert out.a_tac_lat.leg is Leg.ALPAMAYO


def test_the_mapping_is_case_and_whitespace_robust():
    assert _c(alpamayo_lane="  TURN left ").a_tac_lat.value == "TURN_L"


def test_turn_tokens_are_REFUSED_under_v6_0_rather_than_silently_dropped():
    out = _c(alpamayo_lane="Turn Left", vocab_version="v6.0")
    assert out.a_tac_lat.value == ABSTAIN
    assert "v6.1" in out.a_tac_lat.reason
    assert "VOCAB_TOO_OLD_FOR_TOKEN" in out.flags


def test_a_one_axis_stop_row_is_NOT_APPLICABLE_never_imputed():
    """The 304 unparsed lateral rows ARE the 304 `Stop` rows — a structural fact
    about the emitter, not a coverage defect."""
    out = _c(alpamayo_lane=None, alpamayo_lon="Stop")
    assert out.a_tac_lat.value == NOT_APPLICABLE
    assert "not-applicable" in out.a_tac_lat.reason.lower()
    assert out.a_tac_lat.leg is Leg.NONE


def test_ABORT_LC_has_no_alpamayo_source():
    assert "ABORT_LC" not in ALPAMAYO_LANE_TO_LAT.values()


# ---------------------------------- TIER 1 as a PRIOR, never a longitudinal label --
def test_alpamayo_alone_NEVER_produces_a_longitudinal_label():
    """The type mismatch is structural: magnitude cannot decide a reason."""
    for mag in LON_ADMISSIBLE:
        out = _c(alpamayo_lon=mag, vlm_lon=None)
        assert out.a_tac_lon.value == ABSTAIN
        assert "REASON_REQUIRED" in out.a_tac_lon.reason


def test_the_prior_REJECTS_an_inadmissible_vlm_answer_and_FLAGS_it():
    out = _c(alpamayo_lon="Maintain Speed", vlm_lon="BRAKE_TO",
             vlm_referent="the stop line")
    assert out.a_tac_lon.value == ABSTAIN
    assert "LON_PRIOR_DISAGREEMENT" in out.flags
    assert "review" in out.a_tac_lon.reason


def test_the_prior_ACCEPTS_an_admissible_answer_and_records_corroboration():
    out = _c(alpamayo_lon="Gentle Deceleration", vlm_lon="FOLLOW",
             vlm_referent="the white van ahead")
    assert out.a_tac_lon.value == "FOLLOW"
    assert out.a_tac_lon.leg is Leg.VLM
    assert Leg.ALPAMAYO in out.a_tac_lon.corroborated_by


def test_an_unknown_magnitude_does_NOT_silently_reject():
    assert lon_is_admissible("Reverse", "CRUISE") is True
    assert lon_is_admissible(None, "BRAKE_TO") is True


def test_a_verdict_outside_our_vocabulary_is_an_ERROR_not_an_abstention():
    with pytest.raises(ValueError, match="TACTICAL_LON_ACTIONS"):
        _c(vlm_lon="Gentle Deceleration", vlm_referent="x")


# ------------------------------------------- evidence before verdict (the rule) --
def test_a_reason_without_a_named_referent_is_REFUSED():
    out = _c(alpamayo_lon="Gentle Deceleration", vlm_lon="FOLLOW",
             vlm_referent=None)
    assert out.a_tac_lon.value == ABSTAIN
    assert "VLM_REASON_WITHOUT_REFERENT" in out.flags


# ------------------------------------------------------ TIER 3: the ego gate --
def test_ego_args_are_DROPPED_where_the_vlm_abstained():
    """⛔ The rule that keeps ego out of echo territory."""
    out = _c(vlm_goal=None, ego_args={"within_m": 23.4})
    assert out.g_tac_args == {}
    assert "EGO_ARGS_DROPPED_NO_REFERENT" in out.flags


def test_ego_args_SURVIVE_when_the_vlm_named_a_referent_and_a_goal():
    out = _c(vlm_goal="STOP_POINT", vlm_referent="the stop line",
             ego_args={"within_m": 23.4, "by_time_s": 3.1})
    assert out.g_tac.value == "STOP_POINT"
    assert out.g_tac_args == {"within_m": 23.4, "by_time_s": 3.1}


def test_ego_never_assigns_a_class_only_arguments():
    """There is no parameter by which ego can set a_tac_lon or a_tac_lat."""
    import inspect
    sig = inspect.signature(compose).parameters
    ego_params = [p for p in sig if p.startswith("ego_")]
    assert ego_params == ["ego_args"], ego_params


# --------------------------------- the relative-lane-target cascade (PI proposal) --
def test_the_sign_convention_matches_the_programmes_declared_one():
    assert LANE_TARGET_REL == {"LEFT": +1, "CURRENT": 0, "RIGHT": -1}


def test_CURRENT_yields_HOLD_CORRIDOR_which_finally_has_a_map_free_definition():
    g, a = strategic_from_lane_target("CURRENT")
    assert (g.value, a.value) == ("KEEP_CORRIDOR", "HOLD_CORRIDOR")


@pytest.mark.parametrize("rel", ["LEFT", "RIGHT"])
def test_a_non_current_target_yields_LANE_TARGET_and_PREPARE_LANE_CHANGE(rel):
    """PREPARE_LANE_CHANGE stops needing a route: its own rule reduces to
    `lane_target_rel != CURRENT` under the relative encoding."""
    g, a = strategic_from_lane_target(rel)
    assert (g.value, a.value) == ("LANE_TARGET", "PREPARE_LANE_CHANGE")


@pytest.mark.parametrize("rel,tok", [("LEFT", "EXIT_LEFT"), ("RIGHT", "EXIT_RIGHT")])
def test_the_exit_predicate_promotes_the_pair(rel, tok):
    g, a = strategic_from_lane_target(rel, exit_ahead_on_side=True)
    assert (g.value, a.value) == (tok, "PREPARE_EXIT")


def test_an_abstained_lane_target_abstains_BOTH_strategic_fields_with_a_reason():
    g, a = strategic_from_lane_target(ABSTAIN)
    assert g.value == a.value == ABSTAIN
    assert g.reason and a.reason


def test_an_unknown_relative_target_is_REFUSED():
    with pytest.raises(ValueError, match="lane_target_rel"):
        strategic_from_lane_target("SLIGHTLY_LEFT")


def test_every_emitted_strategic_token_is_in_the_v6_vocabulary():
    for rel in ("LEFT", "CURRENT", "RIGHT"):
        for ex in (True, False, None):
            g, a = strategic_from_lane_target(rel, exit_ahead_on_side=ex)
            assert g.value in STRATEGIC_GOAL_TOKENS
            assert a.value in STRATEGIC_ACTION_TOKENS


# ------------------------------------------------------------- ROUTE_TO gate --
def test_ROUTE_TO_can_never_be_emitted_by_any_path():
    """G1 CLOSED at 0/31 — it names a destination that does not exist here."""
    outs = [strategic_from_lane_target(r, exit_ahead_on_side=e)
            for r in ("LEFT", "CURRENT", "RIGHT") for e in (True, False)]
    assert all(g.value != "ROUTE_TO" and a.value != "ROUTE_TO" for g, a in outs)


# ------------------------------------- Alpamayo turns: type+direction, NOT timing --
def test_an_alpamayo_turn_gives_the_strategic_TYPE_but_defers_the_TIMING():
    out = _c(alpamayo_lane="Turn Left")
    assert out.g_str.value == "TURN_LEFT" and out.g_str.leg is Leg.ALPAMAYO
    assert out.a_str.value == ABSTAIN
    assert "TIMING" in out.a_str.reason.upper()
    assert "STRATEGIC_TIMING_FROM_GEOMETRY_REQUIRED" in out.flags


def test_the_vlm_lane_target_TAKES_PRECEDENCE_over_the_alpamayo_turn_fallback():
    out = _c(alpamayo_lane="Turn Left", vlm_lane_target_rel="CURRENT")
    assert out.g_str.value == "KEEP_CORRIDOR"


# ------------------------------------------------------------- provenance ----
def test_no_field_can_be_anonymous():
    with pytest.raises(ValueError, match="reason"):
        LabelField(ABSTAIN, Leg.NONE)          # abstained, no reason
    with pytest.raises(ValueError, match="leg=NONE"):
        LabelField("LANE_KEEP", Leg.NONE, "x")  # present, no author


def test_the_serialised_record_carries_the_leg_of_every_field():
    d = _c(vlm_lon="CRUISE", vlm_referent="open road",
           vlm_goal="SPEED_BAND").to_dict()
    for k in ("a_tac_lat", "a_tac_lon", "g_tac"):
        assert d[k]["leg"], k
    assert d["vocab_version"] == "v6.1"


# ------------------------------------------------------------- the prompts ---
def test_the_prompts_ask_one_axis_each_and_offer_ABSTAIN():
    import vlm_tac_prompts as P
    ctx = P.ClipContext(v0_ms=10.0)
    lon, lane = P.build_lon_prompt(ctx), P.build_lane_prompt(ctx)
    assert ABSTAIN in lon and ABSTAIN in lane
    for t in TACTICAL_LON_ACTIONS:
        assert t in lon
    assert "LEFT" in lane and "RIGHT" in lane and "CURRENT" in lane
    # one axis per call: the lateral tokens must not appear in the lon prompt
    assert "LANE_CHANGE_L" not in lon


def test_the_lane_prompt_asks_the_CURVATURE_DISCRIMINATOR():
    """The retired geometric gate died on this exact confusion (15 of 19)."""
    import vlm_tac_prompts as P
    lane = P.build_lane_prompt(P.ClipContext())
    assert "CROSS" in lane and "CURVE" in lane.upper()
    assert "the road curved" in lane


def test_the_echo_control_withholds_ego_numbers():
    import vlm_tac_prompts as P
    ctx = P.ClipContext(v0_ms=11.5, v_end_ms=7.0)
    assert "11.5" in P.build_lon_prompt(ctx, with_ego=True)
    assert "11.5" not in P.build_lon_prompt(ctx, with_ego=False)


def test_alpamayo_is_presented_as_a_PRIOR_that_may_be_wrong():
    import vlm_tac_prompts as P
    p = P.build_lon_prompt(P.ClipContext(alpamayo_magnitude="Gentle Deceleration"))
    assert "PRIOR ONLY" in p and "do not defer" in p


def test_sampling_does_NOT_use_the_cards_general_presence_penalty():
    """1.5 would bias against the frequent classes; LANE_KEEP is 85.33 %."""
    import vlm_tac_prompts as P
    assert P.SAMPLING["presence_penalty"] == 0.0
    assert P.SAMPLING["temperature"] == 0.6


def test_an_unparseable_answer_becomes_ABSTAIN_not_a_nearest_match():
    import vlm_tac_prompts as P
    assert P.parse_verdict("I think it is probably following", "lon")[0] == ABSTAIN
    assert P.parse_verdict("VERDICT: SLOW_DOWN", "lon")[0] == ABSTAIN
    v, think = P.parse_verdict("<think>reasoning</think>\nVERDICT: FOLLOW", "lon")
    assert v == "FOLLOW" and think == "reasoning"


def test_the_last_verdict_wins_so_a_restated_answer_is_not_misread():
    import vlm_tac_prompts as P
    assert P.parse_verdict("VERDICT: CRUISE\nwait\nVERDICT: FOLLOW", "lon")[0] == "FOLLOW"
