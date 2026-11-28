"""test_r8v_data.py -- controls for the video data layer.  Every expectation is a LITERAL derived by hand (never from the code under test).

    python -m pytest -q -p no:cacheprovider test_r8v_data.py     (pure numpy; no banked data needed)
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r8v_data as D  # noqa: E402


def test_dv_class_first_match_literals():
    assert D.gt_dv_class(10.0, 10.5) == "cruising"
    assert D.gt_dv_class(10.0, 5.0) == "decelerating"
    assert D.gt_dv_class(10.0, 0.5) == "stopping"          # v0 >= 2 and v6 < 1 wins over decelerating
    assert D.gt_dv_class(0.5, 5.0) == "start_from_stop"
    assert D.gt_dv_class(1.0, 1.5) == "slow_both"
    assert D.gt_dv_class(10.0, 12.0) == "accelerating"


def test_dv_class_boundary_hand_checked():
    assert D.gt_dv_class(10.0, 8.6) == "cruising"
    assert D.gt_dv_class(10.0, 8.5) == "decelerating"      # dv = -1.5 -> <= -1.5


def test_plan_class_literals():
    # slots 1..6 s; last second = |P6 - P5|.  v0 = 10 m/s.
    def plan(last_step):
        P = np.zeros((6, 2))
        P[:, 0] = [10, 20, 30, 40, 50, 50 + last_step]
        return P
    assert D.plan_lon_class_one(plan(5.0), 10.0)[0] == "decel"      # 5 - 10 = -5
    assert D.plan_lon_class_one(plan(10.0), 10.0)[0] == "cruise"
    assert D.plan_lon_class_one(plan(12.0), 10.0)[0] == "accel"     # +2
    assert abs(D.plan_lon_class_one(plan(5.0), 10.0)[1] - 5.0) < 1e-12


def test_false_brake_truth_table():
    P8 = np.zeros((8, 2))
    P8[:, 0] = [4, 8, 12, 16, 22, 28, 34, 36]                      # slots 3..: 1 s,2 s,3 s,4 s,5 s,6 s = [8,16,22,28,34,36] -> last second 2 m -> 2 m/s
    fb, gc, pc, ve = D.false_brake(P8, v0=10.0, v6_gt=10.2, gt_valid=True)
    assert (fb, gc, pc) == (True, "cruising", "decel") and abs(ve - 2.0) < 1e-12
    assert D.false_brake(P8, 10.0, 0.4, True)[0] is False           # GT is stopping: a decel plan there is NOT a false brake
    assert D.false_brake(P8, 10.0, 10.2, False)[0] is None          # no GT 6 s -> not scored, never False


def test_reframe_literals():
    A = np.array([0.0, 0.0, 0.0, 0.0])
    B = np.array([10.0, 0.0, 0.0, 0.0])
    assert np.allclose(D.reframe([[20.0, 0.0]], A, B), [[10.0, 0.0]])          # ego drove 10 m forward
    C = np.array([0.0, 0.0, math.pi / 2, 0.0])                                  # ego turned 90 deg left (heading +y)
    assert np.allclose(D.reframe([[0.0, 5.0]], A, C), [[5.0, 0.0]], atol=1e-12)  # a point 5 m to the LEFT is now 5 m AHEAD
    assert np.allclose(D.reframe([[3.0, 4.0]], A, A), [[3.0, 4.0]])             # identity


def test_plan_speeds_constant_5mps():
    P8 = np.array([[2.5, 0], [5, 0], [7.5, 0], [10, 0], [15, 0], [20, 0], [25, 0], [30, 0]], float)   # 5 m/s at every slot interval
    assert np.allclose(D.plan_speeds(P8), 5.0)


def test_softmax_sums_to_one():
    p = D.softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(p.sum() - 1.0) < 1e-12 and p.argmax() == 2
