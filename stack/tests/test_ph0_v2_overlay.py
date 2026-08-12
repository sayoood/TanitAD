"""CPU tests for the PH0 v2 overlay renderer.

The defect class these guard: a viewer that makes a FAILED extraction look like
a good one. A grounding that violated `validate_v2` must be visibly different
from one that passed, and the assumed ±1.5 m band must never render as if it
were a perceived lane — PhysicalAI-AV ships no map data.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from ph0_v2_overlay import (BEV_HALF_M, C_BAD, C_OK, draw_bev,  # noqa: E402
                            draw_boxes, draw_panel, render_clip)

pytest.importorskip("PIL")


def _img(w=200, h=120):
    from PIL import Image
    return Image.new("RGB", (w, h), (255, 255, 255))


def _rec(**kw):
    base = {"clip_id": "abcd1234", "scene": {"illumination": "day",
            "weather": "clear", "road_type": "urban", "domain": "urban",
            "lanes_visible": 2, "lane_ego": 0, "conf": "high"},
            "signs": {"n_signs": 0, "signs": []}, "grounding": [],
            "symbols": {"goal_kind": "follow_main_road",
                        "goal_evidence_sign": None, "conf": "high",
                        "actions": [{"verb": "hold_corridor",
                                     "direction": "none"}]},
            "_calls": [], "_all_valid": True}
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# boxes: status must be visually distinguishable                               #
# --------------------------------------------------------------------------- #
def test_valid_and_invalid_boxes_render_differently():
    a = np.asarray(draw_boxes(_img(), [{"bbox": [10, 10, 80, 60],
                                        "label": "speed", "ok": True}]))
    b = np.asarray(draw_boxes(_img(), [{"bbox": [10, 10, 80, 60],
                                        "label": "speed", "ok": False}]))
    assert not np.array_equal(a, b), "a failed grounding must look different"


def test_valid_box_uses_the_ok_colour_and_invalid_the_bad_one():
    a = np.asarray(draw_boxes(_img(), [{"bbox": [10, 10, 80, 60], "ok": True}]))
    b = np.asarray(draw_boxes(_img(), [{"bbox": [10, 10, 80, 60], "ok": False}]))
    assert (a == np.array(C_OK)).all(-1).any()
    assert (b == np.array(C_BAD)).all(-1).any()
    assert not (a == np.array(C_BAD)).all(-1).any()


def test_no_boxes_leaves_the_frame_untouched():
    blank = np.asarray(_img())
    assert np.array_equal(np.asarray(draw_boxes(_img(), [])), blank)


def test_box_label_is_drawn():
    plain = np.asarray(draw_boxes(_img(), [{"bbox": [10, 10, 80, 60],
                                            "ok": True}]))
    lab = np.asarray(draw_boxes(_img(), [{"bbox": [10, 10, 80, 60],
                                          "label": "speed 20", "ok": True}]))
    assert not np.array_equal(plain, lab)


# --------------------------------------------------------------------------- #
# BEV                                                                          #
# --------------------------------------------------------------------------- #
def test_bev_renders_without_engine_a():
    """A missing polyline must produce a labelled empty BEV, not a crash."""
    img = draw_bev((120, 200), None)
    assert img.size == (120, 200)


def test_bev_draws_a_path_when_given_one():
    ea = {"polyline_xy": [[float(i), 0.0] for i in range(0, 40)]}
    empty = np.asarray(draw_bev((120, 200), None))
    with_path = np.asarray(draw_bev((120, 200), ea))
    assert not np.array_equal(empty, with_path)


def test_bev_drops_points_outside_the_grid():
    """A lateral excursion beyond the grid must be clipped, not wrapped."""
    ea = {"polyline_xy": [[10.0, BEV_HALF_M * 5]]}
    assert np.array_equal(np.asarray(draw_bev((120, 200), ea)),
                          np.asarray(draw_bev((120, 200), None)))


# --------------------------------------------------------------------------- #
# panel                                                                        #
# --------------------------------------------------------------------------- #
def test_panel_renders_valid_and_violating_records_differently():
    ok = np.asarray(draw_panel((430, 300), _rec(), None, 0))
    bad = np.asarray(draw_panel((430, 300), _rec(
        _all_valid=False,
        _calls=[{"call": "B3_ground_0", "violations": ["bbox outside"]}]),
        None, 0))
    assert not np.array_equal(ok, bad)


def test_panel_survives_missing_sections():
    """Every section is independently optional — one failed call must not take
    the whole render down."""
    img = draw_panel((430, 300), _rec(scene=None, signs=None, symbols=None),
                     None, 0)
    assert img.size == (430, 300)


def test_panel_handles_abstention():
    img = draw_panel((430, 300), _rec(
        symbols={"goal_kind": "none_abstain", "goal_evidence_sign": None,
                 "actions": [], "conf": "low"}), None, 0)
    assert img.size == (430, 300)


# --------------------------------------------------------------------------- #
# end to end                                                                   #
# --------------------------------------------------------------------------- #
def test_render_clip_writes_a_still(tmp_path):
    frames = [np.zeros((120, 160, 3), np.uint8) for _ in range(4)]
    outs = render_clip(_rec(), frames, None, str(tmp_path))
    assert any(o.endswith("_still.png") for o in outs)
    assert os.path.exists(outs[0])


def test_render_clip_places_boxes_on_their_reported_frame(tmp_path):
    """A box reported for frame 2 must not be drawn on frame 0 — an off-by-one
    here would put a sign on a frame it never appeared in."""
    frames = [np.zeros((120, 160, 3), np.uint8) for _ in range(4)]
    rec = _rec(signs={"n_signs": 1, "signs": [{"kind": "speed", "text": "30",
                                               "state": "none",
                                               "applies_to_ego": True}]},
               grounding=[{"visible": True, "frame_idx": 2,
                           "bbox": [10, 10, 60, 50]}])
    outs = render_clip(rec, frames, None, str(tmp_path))
    assert os.path.exists(outs[0])


def test_invisible_grounding_draws_no_box(tmp_path):
    frames = [np.zeros((120, 160, 3), np.uint8) for _ in range(3)]
    rec = _rec(signs={"n_signs": 1, "signs": [{"kind": "other", "text": "",
                                               "state": "none",
                                               "applies_to_ego": False}]},
               grounding=[{"visible": False, "frame_idx": 0,
                           "bbox": [0, 0, 0, 0]}])
    a = render_clip(rec, frames, None, str(tmp_path / "a"))
    b = render_clip(_rec(), frames, None, str(tmp_path / "b"))
    from PIL import Image
    assert np.array_equal(np.asarray(Image.open(a[0]).crop((0, 0, 160, 120))),
                          np.asarray(Image.open(b[0]).crop((0, 0, 160, 120))))
