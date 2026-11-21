"""Controls C3 (analytic path geometry) and C8 (analytic BEV IoU) -- literal expectations, never expressions
over the code under test (CLAUDE.md: a cross-check must be derived independently of the value it checks)."""
import math

import numpy as np

import route_metrics as rm

T = np.array(rm.HORIZONS) * 0.1


def arc(R, v, sign=+1):
    """a circular arc of radius R at speed v, turning LEFT (+1) / RIGHT (-1); heading at time t is v t / R."""
    ph = v * T / R
    return np.stack([R * np.sin(ph), sign * R * (1 - np.cos(ph))], -1)


def test_arc_terminal_heading_is_the_chord_angle():
    # the chord from t=5 s to t=6 s of a circle has heading = mean of the two tangent angles (exact geometry)
    R, v = 20.0, 5.0
    th = rm.terminal_heading(arc(R, v))
    assert abs(th - 0.5 * (5.0 * 5 / 20 + 6.0 * 5 / 20)) < 1e-9          # 1.375 rad
    assert abs(rm.terminal_heading(arc(R, v, -1)) + 1.375) < 1e-9          # mirror -> negated


def test_straight_line_reads_zero_and_stall_reads_zero():
    P = np.stack([T * 10.0, np.zeros(8)], -1)
    assert rm.terminal_heading(P) == 0.0
    S = np.zeros((8, 2))
    S[:, 0] = 1.0                                                        # parked at x = 1 m
    assert rm.terminal_heading(S) == 0.0


def test_dir_class_threshold_is_inclusive_like_compliance_target():
    assert rm.dir_class(rm.TAU_C) == 1 and rm.dir_class(-rm.TAU_C) == -1 and rm.dir_class(0.18) == 0


def test_gt_class():
    gt = np.stack([arc(20.0, 5.0), np.stack([T * 10.0, np.zeros(8)], -1), arc(200.0, 10.0)])
    cls, th = rm.gt_class(gt, np.ones((3, 8), bool))
    assert list(cls) == ["turnL", "straight", "gentle"]    # 1.375 rad = 78.8 deg ; 0 ; 0.275 rad = 15.8 deg


def test_iou_analytic():
    assert abs(rm.bev_iou((0, 0, 4, 2, 0.0), (0, 0, 4, 2, 0.0)) - 1.0) < 1e-9
    assert rm.bev_iou((0, 0, 4, 2, 0.0), (10, 0, 4, 2, 0.0)) == 0.0
    assert abs(rm.bev_iou((0, 0, 4, 2, 0.0), (2, 0, 4, 2, 0.0)) - 1.0 / 3.0) < 1e-9
    assert abs(rm.bev_iou((1, 1, 2, 2, 0.0), (1, 1, 2, 2, math.pi / 2)) - 1.0) < 1e-9
    # a 2x2 square vs the same square rotated 45 deg about its centre: overlap = regular octagon
    oct_area = 2 * (1 + math.sqrt(2)) * (2 * math.tan(math.pi / 8)) ** 2      # side of the octagon = 2 tan(22.5)
    exp = oct_area / (8 - oct_area)
    assert abs(rm.bev_iou((0, 0, 2, 2, 0.0), (0, 0, 2, 2, math.pi / 4)) - exp) < 1e-9


def test_nms():
    p = np.array([0.9, 0.8, 0.7])
    xy = np.array([[0, 0], [1, 0], [5, 0]], float)
    assert list(rm.nms_keep(p, xy, "centre", 1.5)) == [True, False, True]
    assert list(rm.nms_keep(p, xy, "none", 0)) == [True, True, True]
    bx = [(0, 0, 4, 2, 0.0), (1, 0, 4, 2, 0.0), (5, 0, 4, 2, 0.0)]
    assert list(rm.nms_keep(p, xy, "iou", 0.5, boxes=bx)) == [True, False, True]     # IoU(0,1) = 3/5
    assert list(rm.nms_keep(p, xy, "iou", 0.7, boxes=bx)) == [True, True, True]


def test_spearman_and_bootstrap():
    assert abs(rm.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12
    assert abs(rm.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    pt, bs, _ = rm.boot_ratio([1, 1, 0, 0], [1, 1, 1, 1], ["a", "a", "b", "b"], B=200)
    assert pt == 0.5
