"""Pure-function tests for taniteval/tools/render_refcv6_gt_validation.py (no stack, no data).

Every expectation is a LITERAL derived by hand, never an expression over the code under test.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

_P = Path(__file__).resolve().parents[1] / "tools" / "render_refcv6_gt_validation.py"
_spec = importlib.util.spec_from_file_location("gtval_under_test", str(_P))
gv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gv)


def test_filter_reasons_literal_points():
    # (x, y): kept / behind / outside the field AND |y|>16 / beyond 60 / in the field but |y|>16
    cx = np.array([10.0, -5.0, 5.0, 70.0, 40.0, 10.0])
    cy = np.array([0.0, 0.0, 20.0, 0.0, 20.0, 0.0])
    valid = np.array([True, True, True, True, True, False])
    fr = gv.filter_reasons(cx, cy, valid)
    assert fr["keep"].tolist() == [True, False, False, False, False, False]
    # primary = the FIRST reason in REASONS order; -1 = kept or invalid
    assert fr["primary"].tolist() == [-1, 0, 1, 2, 3, -1]
    # (5, 20) carries two reasons: outside the field AND |y| > 16
    assert fr["reasons"][:, 2].tolist() == [False, True, False, True]
    # an invalid row carries no reason at all
    assert not fr["reasons"][:, 5].any()


def test_filter_reasons_field_edge_is_60_degrees():
    # atan2(|y|, x) = 59.9 deg is kept, 60.1 deg is not (x = 5 m)
    y_in, y_out = 5.0 * math.tan(math.radians(59.9)), 5.0 * math.tan(math.radians(60.1))
    fr = gv.filter_reasons(np.array([5.0, 5.0]), np.array([y_in, y_out]), np.array([True, True]))
    assert fr["keep"].tolist() == [True, False]


def test_bev_iou_literals():
    assert gv.bev_iou((0, 0, 2, 1, 0), (0, 0, 2, 1, 0)) == pytest.approx(1.0, abs=1e-12)
    assert gv.bev_iou((0, 0, 2, 1, 0), (10, 0, 2, 1, 0)) == 0.0
    # 2 x 1 boxes offset 1 m along x: intersection 1 x 1 = 1, union 2 + 2 - 1 = 3
    assert gv.bev_iou((0, 0, 2, 1, 0), (1, 0, 2, 1, 0)) == pytest.approx(1 / 3, abs=1e-9)
    # the same box turned 90 deg: a 1 x 1 cross -> 1 / 3
    assert gv.bev_iou((0, 0, 2, 1, 0), (0, 0, 2, 1, math.pi / 2)) == pytest.approx(1 / 3, abs=1e-9)


def test_duplicate_pairs_finds_the_copy_only():
    b = np.array([[5.0, 1.0, 4.5, 1.8, 0.3], [5.0, 1.0, 4.5, 1.8, 0.3], [30.0, -4.0, 4.5, 1.8, 0.0]])
    out = gv.duplicate_pairs(b)
    assert [(i, j) for i, j, _ in out] == [(0, 1)]
    assert out[0][2] == pytest.approx(1.0, abs=1e-12)


def test_norm_flags_pi_examples():
    assert gv.norm_flags("person", 2.4, 0.6, 1.7) == ["L 2.40 m > 2"]
    assert gv.norm_flags("automobile", 4.5, 1.8, 3.2) == ["H 3.20 m > 3"]
    assert gv.norm_flags("automobile", 4.5, 1.8, None) == []
    assert gv.norm_flags("protruding_object", 9.0, 9.0, 9.0) == []


def test_bev_pixel_roundtrip_and_region():
    for x, y in ((0.0, 0.0), (37.5, -12.25), (-20.0, 32.0), (80.0, -32.0)):
        px, py = gv.bevb_px(x, y)
        assert gv.bevb_m(px, py) == pytest.approx((x, y), abs=1e-12)
    # the ego sits 20 m above the bottom edge at the horizontal centre: (192, 480) px at 6 px/m
    assert gv.bevb_px(0.0, 0.0) == (192.0, 480.0)
    reg = gv.target_region_polygon()
    # the 60 deg wedge meets |y| = 16 m at x = 16 / tan(60 deg) = 9.2376 m
    assert reg[1] == pytest.approx([9.2376, 16.0], abs=1e-4)


def test_range_hist_literal():
    h = gv.range_hist([0.0, 9.99, 10.0, 59.0, 60.0, 200.0])
    assert h["0-10m"] == 2 and h["10-20m"] == 1 and h["40-60m"] == 1 and h["60-80m"] == 1
    assert h[">=120m"] == 1


def test_convex_hull_literal():
    h = gv.convex_hull([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1], [1, 0]])
    assert sorted(map(tuple, h.tolist())) == [(0.0, 0.0), (0.0, 2.0), (2.0, 0.0), (2.0, 2.0)]


def test_visible_fractions_literal():
    sq = lambda x0, y0, s: [(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s)]
    near = sq(10, 10, 40)            # depth 5, covers x 10..50
    far = sq(30, 10, 40)             # depth 10, x 30..70: its left half is behind `near`
    hidden = sq(20, 20, 10)          # depth 20, entirely inside `near`
    vf = gv.visible_fractions([near, far, hidden, None], [5.0, 10.0, 20.0, 1.0], w=100, h=100)
    assert vf[0] == 1.0
    assert vf[1] == pytest.approx(0.5, abs=0.05)      # analytic 20 / 40 columns
    assert vf[2] == 0.0
    assert np.isnan(vf[3])


def test_angle_poly_literal():
    # a box straight ahead, x 10..14 m, y -1..1 m, z 0..1.5 m; camera centre at (0, 0, 1)
    cor = np.array([[x, y, z] for z in (0.0, 1.5) for x, y in ((14, 1), (14, -1), (10, -1), (10, 1))])
    p = gv.angle_poly(cor, (0.0, 0.0, 1.0))
    half = math.degrees(math.atan2(1.0, 10.0)) / gv.ANG_RES_DEG       # the NEAR face spans +-5.71 deg
    assert p[:, 0].min() == pytest.approx(900.0 - half, abs=1e-9)
    assert p[:, 0].max() == pytest.approx(900.0 + half, abs=1e-9)
    assert gv.angle_poly(cor, (11.0, 0.0, 1.0)) is None               # a corner behind the camera


def test_cyl3_literal():
    class P:                                      # identity mount, f 100, centre (50, 20)
        R, t, f, cx, cy, proj = np.eye(3), np.zeros(3), 100.0, 50.0, 20.0, "cylindrical"
    assert gv.cyl3(P, (0.0, 0.0, 10.0)) == pytest.approx((50.0, 20.0))
    # 45 deg to the right: u = 50 + 100 * pi / 4
    assert gv.cyl3(P, (10.0, 0.0, 10.0)) == pytest.approx((50.0 + 100.0 * math.pi / 4, 20.0))
    assert gv.cyl3(P, (0.0, 0.0, -1.0)) is None
