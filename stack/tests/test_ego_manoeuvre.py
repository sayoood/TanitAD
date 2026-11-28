"""Tests for ego-kinematic manoeuvre detection and Alpamayo CoT semantics.

⭐ THE ARM THAT MATTERS MOST is ``test_curving_main_road_is_not_a_junction_turn``.
It pins the estimator error that cost a whole analysis pass: measuring turn
radius as arc/delta-yaw over the horizon divides a turn's heading change by an
arc that includes the STRAIGHT ROAD AFTER IT, so a tight junction turn reads as
a gentle bend (12.4 m read as 83.7 m on clip 5b4eef8f). The synthetic fixtures
below are built so the two estimators DISAGREE, which is the only way a test can
hold the correct one in place.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tanitad.data import alpamayo_semantics as S
from tanitad.data import ego_manoeuvre as EM

HZ = 10.0


def _arc(v, radius, dyaw_deg, n_pre=0, n_post=0, hz=HZ):
    """Build poses: optional straight run, a constant-radius arc, optional run."""
    xs, ys, yaws, vs = [], [], [], []
    x = y = yaw = 0.0
    for _ in range(n_pre):
        x += v / hz * math.cos(yaw); y += v / hz * math.sin(yaw)
        xs.append(x); ys.append(y); yaws.append(yaw); vs.append(v)
    total = math.radians(dyaw_deg)
    arc_len = abs(total) * radius
    steps = max(2, int(round(arc_len / (v / hz))))
    for _ in range(steps):
        yaw += total / steps
        x += v / hz * math.cos(yaw); y += v / hz * math.sin(yaw)
        xs.append(x); ys.append(y); yaws.append(yaw); vs.append(v)
    for _ in range(n_post):
        x += v / hz * math.cos(yaw); y += v / hz * math.sin(yaw)
        xs.append(x); ys.append(y); yaws.append(yaw); vs.append(v)
    return np.stack([xs, ys, yaws, vs], axis=1)


def test_tight_junction_turn_is_detected():
    m = EM.analyse(_arc(6.0, 12.0, 90.0, n_post=40))
    assert m.lateral_class == "JUNCTION_TURN_L"
    assert m.turn_radius_m == pytest.approx(12.0, rel=0.25)
    assert m.confidence == "HIGH"


def test_curving_main_road_is_not_a_junction_turn():
    """A long, large-radius bend is road geometry, not a decision."""
    m = EM.analyse(_arc(14.0, 220.0, 45.0))
    assert m.lateral_class == "ROAD_BEND_L", m.lateral_class
    assert m.turn_radius_m > EM.R_BEND_M


def test_the_estimator_error_is_pinned():
    """A 12 m turn followed by a long straight must NOT read as a bend.

    arc/delta-yaw over the whole horizon gives ~4x the true radius here; only
    instantaneous curvature keeps the verdict correct. This is the exact shape
    of clip 5b4eef8f.
    """
    poses = _arc(6.0, 12.0, 70.0, n_post=200)
    m = EM.analyse(poses)
    # what the BAD estimator would have said:
    x, y, yaw = poses[:, 0], poses[:, 1], poses[:, 2]
    arc_all = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    bad_R = arc_all / abs(yaw[-1] - yaw[0])
    assert bad_R > EM.R_BEND_M, "fixture must actually fool the bad estimator"
    assert m.turn_radius_m < EM.R_JUNCTION_M
    assert m.lateral_class == "JUNCTION_TURN_L"


def test_straight_is_straight():
    assert EM.analyse(_arc(10.0, 1e6, 0.2, n_post=60)).lateral_class == "STRAIGHT"


# -- longitudinal ----------------------------------------------------------
def _profile(speeds, hz=HZ):
    x = np.cumsum(np.asarray(speeds, float) / hz)
    return np.stack([x, np.zeros_like(x), np.zeros_like(x),
                     np.asarray(speeds, float)], axis=1)


def test_controlled_stop_one_decel_then_clean_release():
    v = list(np.linspace(9, 0, 40)) + [0] * 25 + list(np.linspace(0, 10, 40))
    m = EM.analyse(_profile(v))
    assert m.stop_type == "CONTROLLED"
    assert m.n_stop_episodes == 1


def test_queue_is_repetition_not_depth():
    """The discriminator the PI asked for: a jam stops REPEATEDLY."""
    cyc = list(np.linspace(4, 0, 12)) + [0] * 8 + list(np.linspace(0, 4, 12))
    m = EM.analyse(_profile(cyc * 3))
    assert m.stop_type == "QUEUE"
    assert m.longitudinal_class == "STOP_AND_GO"
    assert m.n_stop_episodes >= 2


def test_a_queue_and_a_light_stop_are_not_confused():
    light = list(np.linspace(9, 0, 40)) + [0] * 25 + list(np.linspace(0, 10, 40))
    cyc = list(np.linspace(4, 0, 12)) + [0] * 8 + list(np.linspace(0, 4, 12))
    assert EM.analyse(_profile(light)).stop_type != \
           EM.analyse(_profile(cyc * 3)).stop_type


def test_launch_from_rest():
    m = EM.analyse(_profile([0] * 20 + list(np.linspace(0, 13, 60))))
    assert m.longitudinal_class == "LAUNCH"


def test_cruise_has_no_stop():
    m = EM.analyse(_profile([10.0] * 80))
    assert m.stop_type == "NONE" and m.longitudinal_class == "CRUISE"


def test_decel_events_counts_descents_not_samples():
    assert EM.decel_events(np.array([10.0] * 30)) == 0
    assert EM.decel_events(np.concatenate([np.linspace(10, 0, 40)])) == 1
    cyc = np.concatenate([np.linspace(4, 0, 12), np.linspace(0, 4, 12)])
    assert EM.decel_events(np.concatenate([cyc] * 3)) >= 2


# -- Alpamayo CoT semantics -------------------------------------------------
@pytest.mark.parametrize("cot,ref,reason", [
    ("Stop for the red traffic light.", "TRAFFIC_LIGHT_RED", "light"),
    ("Slow down due to the stop sign ahead.", "STOP_SIGN", "sign"),
    ("Slow down due to the yield sign at the intersection", "YIELD_SIGN", "sign"),
    ("Decelerate to maintain a safe distance from the lead vehicle.",
     "LEAD_VEHICLE", "queue"),
    ("Slow down due to pedestrians crossing the crosswalk", "PEDESTRIAN", "hazard"),
    ("Nudge left to pass the cyclist in the same lane", "CYCLIST", "hazard"),
    ("Adapt speed for the right curve ahead.", "CURVE_AHEAD", None),
])
def test_cot_referents_and_stop_reason(cot, ref, reason):
    sem = S.extract(cot)
    assert ref in sem.referents, sem.referents
    assert sem.stop_reason == reason


def test_empty_cot_is_silent():
    sem = S.extract(None)
    assert sem.referents == () and sem.stop_reason is None


def test_cot_yields_the_blocked_tactical_tokens():
    """The tokens HIERARCHY_VOCABULARY listed as needing non-ego inputs."""
    got = {t.token for c in [
        "Nudge left to pass the cyclist in the same lane",
        "Yield to the oncoming traffic before turning",
        "Stop for the red traffic light.",
        "Decelerate to maintain a safe distance from the lead vehicle.",
    ] for t in S.propose_tokens(c)}
    assert {"EVADE_IN_CORRIDOR", "YIELD_AT", "TRAFFIC_LIGHT_REACT",
            "GAP_TARGET", "STOP_POINT"} <= got, got


def test_every_proposed_token_is_disputed_and_carries_its_evidence():
    """An ungrounded VLM claim may never present as a measurement."""
    for t in S.propose_tokens("Stop for the red traffic light."):
        assert t.disputed is True
        assert t.provenance == "vlm-cot"
        assert "red traffic light" in t.evidence


def test_traffic_light_state_is_carried_not_guessed():
    for cot, want in [("Stop for the red traffic light.", "red"),
                      ("Slow down because of yellow traffic light ahead.", "yellow"),
                      ("Resume speed from stop since the traffic light turns green.",
                       "green")]:
        tok = next(t for t in S.propose_tokens(cot)
                   if t.token == "TRAFFIC_LIGHT_REACT")
        assert tok.args["state"] == want


def test_reconcile_surfaces_a_conflict_rather_than_resolving_it():
    sem = S.extract("Stop for the red traffic light.")
    resolved, prov = S.reconcile("QUEUE", sem)
    assert resolved == "QUEUE" and "CONFLICT" in prov


def test_reconcile_lets_the_vlm_name_a_queue():
    sem = S.extract("Decelerate to maintain a safe distance from the lead vehicle.")
    resolved, _ = S.reconcile("CONTROLLED", sem)
    assert resolved == "QUEUE"
