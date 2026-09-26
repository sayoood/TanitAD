"""``turn_following.net_heading`` against analytic targets; the ratio against a hand-built case.

    PYTHONPATH=D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval \
      C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q tests/test_turn_following.py

The "refcv4b turns 0.149x as much as the human on curves" finding rests on this function, and it
deliberately reads POSITIONS, never the pose's heading channel (E2's spline-tangent headings can
spin through several turns). So the tests include a pose whose heading channel is garbage.
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("w3_turn", PKG / "code" / "turn_following.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["w3_turn"] = m
    spec.loader.exec_module(m)
    return m


M = _mod()


def _path(pts, heading=0.0):
    p = np.zeros((8, 3))
    p[:, :2] = pts
    p[:, 2] = heading
    return p


def test_a_straight_path_has_zero_net_heading():
    assert M.net_heading(_path([[k, 0.0] for k in range(1, 9)])) == 0.0


def test_the_last_moving_segment_sets_the_net_heading_and_the_heading_channel_is_ignored():
    """The final segment points at exactly 30 deg; the heading CHANNEL says 12 rad (a spun spline
    tangent). The answer must be 30 deg."""
    pts = [[k, 0.0] for k in range(1, 8)]
    pts.append([7 + math.cos(math.radians(30)), math.sin(math.radians(30))])
    assert math.degrees(M.net_heading(_path(pts, heading=12.0))) == pytest.approx(30.0, abs=1e-9)


def test_a_plan_that_never_moves_has_no_net_heading_not_a_zero():
    """⛔ A stopped plan has no direction. Scoring it as 0 deg would count it as 'did not turn',
    which is a claim about behaviour the plan never exhibited."""
    assert M.net_heading(_path([[0.01 * k, 0.0] for k in range(1, 9)])) is None


def test_the_ratio_and_the_same_sign_share_on_a_hand_built_case():
    """Human turns +20 deg under STRAIGHT on two tokens; the model turns +10 deg on one and -10 deg
    on the other: ratios 0.5 and -0.5 -> median 0.0; same sign 50 %."""
    def arc(deg):
        pts = [[k, 0.0] for k in range(1, 8)]
        pts.append([7 + math.cos(math.radians(deg)), math.sin(math.radians(deg))])
        return _path(pts)
    human = {"a": arc(20), "b": arc(20)}
    model = {"a": arc(10), "b": arc(-10)}
    r = M.turn_following(model, human, {"a": "STRAIGHT", "b": "STRAIGHT"})
    assert r["STRAIGHT"]["n_turning"] == 2
    assert r["STRAIGHT"]["median_model_over_human"] == pytest.approx(0.0, abs=1e-9)
    assert r["STRAIGHT"]["same_sign_pct"] == 50.0


def test_a_human_path_below_the_turn_threshold_is_not_counted():
    human = {"a": _path([[k, 0.0] for k in range(1, 9)])}
    model = {"a": _path([[k, 0.0] for k in range(1, 9)])}
    assert M.turn_following(model, human, {"a": "LEFT"}) == {}
