"""Tests for the cross-family label guard.

⭐ THE ARM THAT MATTERS MOST is ``test_turn_with_tactical_lane_keep_is_silent``.
A strategic TURN_* sitting beside a tactical ``lane_keep`` looks exactly like a
contradiction and is NOT one — the tactical horizon is 2.0 s and 10 of 21
sampled turns start later than that. A guard that fired on it would invent a
~50 % failure rate on a healthy corpus, which is worse than no guard at all.

Per the quality rules every guard here is also shown ABLE TO FAIL: each rule
has a positive arm that must fire and a negative arm that must stay silent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tanitad.data import label_guard as lg

SAMPLE = (Path(__file__).resolve().parents[2] / "products" /
          "P2-data-pipelines" / "2026-08-23-label-validation-sample" /
          "raw" / "sample_v2_slim.json")

CLEAN = dict(clip_id="clean", g_str="TURN_LEFT", a_str="HOLD_CORRIDOR",
             peak_yaw_deg=+62.0, v_at_key_ms=5.8, v_end_ms=6.1,
             v_min_future_ms=4.9, tac_lat="turn_left", tac_lon="steady")


def test_clean_clip_produces_no_findings():
    assert lg.check(**CLEAN).clean


# --------------------------------------------------------------------------
# G1 — the fallback token must not absorb a junction turn
# --------------------------------------------------------------------------
@pytest.mark.parametrize("goal", ["FOLLOW_MAIN_ROAD", "NONE_ABSTAIN"])
def test_g1_fires_on_fallback_hiding_a_turn(goal):
    r = lg.check(**{**CLEAN, "g_str": goal, "tac_lat": None,
                    "peak_yaw_deg": +69.4})
    assert r.refused
    assert [f.rule for f in r.findings] == ["G1-fallback-absorbs-turn"]


def test_g1_silent_when_the_path_really_is_straight():
    r = lg.check(**{**CLEAN, "g_str": "FOLLOW_MAIN_ROAD", "tac_lat": None,
                    "peak_yaw_deg": -13.3})
    assert r.clean


def test_g1_boundary_is_inclusive_at_the_turn_gate():
    below = lg.check(**{**CLEAN, "g_str": "FOLLOW_MAIN_ROAD", "tac_lat": None,
                        "peak_yaw_deg": lg.TURN_DEG - 0.1})
    at = lg.check(**{**CLEAN, "g_str": "FOLLOW_MAIN_ROAD", "tac_lat": None,
                     "peak_yaw_deg": lg.TURN_DEG})
    assert below.clean and at.refused


# --------------------------------------------------------------------------
# G2 — direction conflict between families, and the false positive it must not make
# --------------------------------------------------------------------------
def test_g2_fires_on_a_genuine_sign_conflict():
    r = lg.check(**{**CLEAN, "g_str": "TURN_LEFT", "tac_lat": "turn_right"})
    assert [f.rule for f in r.findings] == ["G2-lat-sign-conflict"]
    assert not r.refused          # a disagreement is a FLAG, not a refusal


def test_turn_with_tactical_lane_keep_is_silent():
    """The 52 % case. A late turn is not a contradiction."""
    for goal in ("TURN_LEFT", "TURN_RIGHT"):
        r = lg.check(**{**CLEAN, "g_str": goal, "tac_lat": "lane_keep",
                        "peak_yaw_deg": +62.0 if goal == "TURN_LEFT" else -62.0})
        assert r.clean, f"guard invented a defect on a late {goal}"


# --------------------------------------------------------------------------
# G3/G4/G5 — longitudinal
# --------------------------------------------------------------------------
def test_g3_refuses_prepare_stop_while_accelerating_from_rest():
    r = lg.check(**{**CLEAN, "g_str": "FOLLOW_MAIN_ROAD", "a_str": "PREPARE_STOP",
                    "tac_lat": None, "peak_yaw_deg": +0.2, "v_at_key_ms": 0.17,
                    "v_end_ms": 13.27, "v_min_future_ms": 0.17})
    assert r.refused
    assert "G3-lon-inverted" in [f.rule for f in r.findings]


def test_g3_silent_on_the_same_profile_labelled_resume_cruise():
    """Two clips with this profile ARE correctly labelled in the corpus."""
    r = lg.check(**{**CLEAN, "g_str": "FOLLOW_MAIN_ROAD", "a_str": "RESUME_CRUISE",
                    "tac_lat": None, "peak_yaw_deg": -1.0, "v_at_key_ms": 0.0,
                    "v_end_ms": 14.06, "v_min_future_ms": 0.0})
    assert r.clean


def test_g4_flags_a_full_stop_described_as_hold_corridor():
    r = lg.check(**{**CLEAN, "g_str": "FOLLOW_MAIN_ROAD", "a_str": "HOLD_CORRIDOR",
                    "tac_lat": None, "peak_yaw_deg": +0.1, "v_at_key_ms": 9.69,
                    "v_end_ms": 0.0, "v_min_future_ms": 0.0})
    assert [f.rule for f in r.findings] == ["G4-stop-undescribed"]


def test_g4_silent_when_the_ego_was_already_stopped():
    r = lg.check(**{**CLEAN, "a_str": "HOLD_CORRIDOR", "v_at_key_ms": 0.4,
                    "v_end_ms": 0.0, "v_min_future_ms": 0.0})
    assert r.clean


def test_g5_flags_tactical_lon_contradicting_the_action():
    r = lg.check(**{**CLEAN, "a_str": "PREPARE_STOP", "tac_lon": "accelerate",
                    "v_at_key_ms": 6.0, "v_end_ms": 6.2, "v_min_future_ms": 5.0})
    assert "G5-lon-family-conflict" in [f.rule for f in r.findings]


# --------------------------------------------------------------------------
# The real-data arm — the guard must reproduce the hand adjudication
# --------------------------------------------------------------------------
KNOWN_DEFECTS = {"5b4eef8f", "4d389996", "5aef0388", "b0499b70", "a8a381bf"}


@pytest.mark.skipif(not SAMPLE.exists(), reason="validation sample not banked")
def test_guard_reproduces_the_hand_adjudication_on_the_real_sample():
    rows = json.loads(SAMPLE.read_text(encoding="utf-8"))
    caught = set()
    for r in rows:
        geo, tac = r["geometry"], r["tactical"]["v2"]
        rep = lg.check(clip_id=r["clip_id"], g_str=r["g_str"]["token"],
                       a_str=r["a_str"]["token"],
                       peak_yaw_deg=geo["peak_yaw_after_key_deg"],
                       v_at_key_ms=geo["v_at_key_mps"],
                       v_end_ms=geo["v_end_mps"],
                       v_min_future_ms=geo["v_min_future_mps"],
                       tac_lat=tac["lat"], tac_lon=tac["lon"])
        if rep.findings:
            caught.add(r["clip_id"][:8])
    missed = KNOWN_DEFECTS - caught
    assert not missed, f"guard missed hand-adjudicated defects: {sorted(missed)}"
