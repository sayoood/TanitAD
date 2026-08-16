"""CPU tests for the full-pipeline renderer.

⛔ WHY render_clip IS EXERCISED END-TO-END HERE. The first pod run died on
``sorted(..., default=None)`` — `default` is a min/max kwarg, not a sorted one.
A smoke test that called only the three drawing helpers passed happily, because
the bug was in the function that STITCHES them. Drawing helpers are not the
risky part; the assembly is.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))


def _frames(n=6, h=179, w=448):
    rng = np.random.default_rng(0)
    return [(rng.random((h, w, 3)) * 255).astype(np.uint8) for _ in range(n)]


def _rec():
    return {"scene": {"illumination": "day", "weather": "clear",
                      "road_type": "urban", "domain": "urban",
                      "lanes_visible": 2, "lane_ego": 0, "conf": "med"},
            "signs": {"signs": [{"kind": "speed", "text": "50",
                                 "state": "none", "applies_to_ego": True}]},
            "symbols": {"goal_kind": "follow_main_road",
                        "actions": [{"verb": "hold_corridor",
                                     "direction": "none"}]},
            "ego_state": {"v_now_kmh": 45.9, "motion": "steady",
                          "turning": "straight"},
            "grounding": [{"visible": True, "frame_idx": 2,
                           "bbox": [400, 300, 450, 350]}]}


def _sam():
    return {"per_concept_hits": {"car": 3, "traffic sign": 1},
            "vlm_cross_check": [{"matched": False}],
            "frames": {"0": {"n_det": 1, "det": [
                {"concept": "car", "score": 0.9,
                 "box_xyxy": [100, 60, 160, 110],
                 "rle_rows": [[r, 100, 160] for r in range(60, 110)]}]},
                "4": {"n_det": 0, "det": []}}}


def _ea():
    return {"route": {"token": "turn_left", "token_valid": True,
                      "dist_m": 40.2, "arc_m": 55.1,
                      "maneuver_dyaw_rad": -1.2},
            "speed_profile": {"v_t0_ms": 12.7, "v_min_future_ms": 3.1,
                              "v_max_future_ms": 14.0, "net_dv_ms": -2.0,
                              "stops": False},
            "situations": {"lane_change": False, "intersection": True,
                           "roundabout": False},
            "polyline_xy": [[i * 0.5, 0.02 * i * i] for i in range(200)],
            "t0_idx": 80}


def test_clip_layout_sorts_sam3_keys_without_crashing():
    """⛔ THE ACTUAL POD FAILURE: `sorted(..., default=None)` — `default` is a
    min/max kwarg. It lived in the ASSEMBLY, which was previously reachable only
    through a video encoder that is not installed everywhere, so a helper-level
    smoke test passed while the batch died."""
    from ph0_rich_overlay import clip_layout
    lay = clip_layout(_rec(), _sam(), _frames())
    assert lay["keys"] == [0, 4]
    assert lay["W"] == 448 * 2 + 470 and lay["H"] == 179 * 2 + 210
    assert len(lay["vlm_boxes"]) == 1
    assert lay["vlm_boxes"][0]["frame_idx"] == 2


def test_clip_layout_handles_a_clip_sam3_skipped():
    from ph0_rich_overlay import clip_layout
    assert clip_layout(_rec(), None, _frames())["keys"] == []


def test_dets_for_frame_bounds_the_temporal_gap():
    """⚠️ At 2 fps sampling, snapping to the nearest SAM3 frame across an
    unbounded gap would paint detections from seconds of different driving onto
    the current frame. The gap is bounded and empty is returned instead."""
    from ph0_rich_overlay import clip_layout, dets_for_frame
    lay = clip_layout(_rec(), _sam(), _frames(n=20))
    assert len(dets_for_frame(lay, 0)) == 1          # exact key
    assert len(dets_for_frame(lay, 2)) == 1          # within max_gap of 0
    assert dets_for_frame(lay, 12) == []             # 8 away from key 4 -> none
    assert dets_for_frame(lay, 4) == []              # key 4 has n_det 0


def test_compose_frame_builds_the_full_canvas_without_an_encoder():
    from ph0_rich_overlay import clip_layout, compose_frame
    fr = _frames()
    lay = clip_layout(_rec(), _sam(), fr)
    img = compose_frame(_rec(), _sam(), fr, _ea(), lay, 0)
    assert img.size == (lay["W"], lay["H"])


def test_compose_frame_survives_absent_ego_and_absent_sam3():
    from ph0_rich_overlay import clip_layout, compose_frame
    fr = _frames()
    lay = clip_layout(_rec(), None, fr)
    assert compose_frame(_rec(), None, fr, None, lay, 1).size == (lay["W"],
                                                                  lay["H"])


def test_canvas_geometry_is_camera_plus_panel_and_bev():
    from ph0_rich_overlay import (BEV_H, PANEL_W, SCALE, draw_bev,
                                  draw_camera, draw_panel)
    f = _frames(1)[0]
    cam = draw_camera(f, [], [])
    assert cam.size == (448 * SCALE, 179 * SCALE)
    H = 179 * SCALE + BEV_H
    assert draw_panel((PANEL_W, H), _rec(), _sam(), _ea(), 0).size == (PANEL_W, H)
    assert draw_bev((448 * SCALE, BEV_H), _ea(), 3, 6).size == (448 * SCALE, BEV_H)


def test_vlm_boxes_are_drawn_over_sam3_not_under():
    """⭐ The colour rule is the point of the figure: colour says WHICH ENGINE
    produced a mark. The VLM's dashed white boxes are drawn last so a SAM3 mask
    can never hide them — merging the two would hide the disagreement the
    cross-engine check exists to surface."""
    import inspect
    from ph0_rich_overlay import draw_camera
    src = inspect.getsource(draw_camera)
    assert src.index("for det in sam_dets") < src.index("for vb in vlm_boxes")


def test_bev_reports_absence_instead_of_drawing_an_empty_box():
    from ph0_rich_overlay import draw_bev
    img = draw_bev((300, 120), {"polyline_xy": []}, 0, 1)
    assert img.size == (300, 120)          # renders a stated "no ego poses"


# =========================================================================== #
# The PI's viz standard: perception AND the strategic label it feeds, in one   #
# frame — plus the C77 liveness control ON the figure                          #
# =========================================================================== #
def _s2():
    return {"clip_id": "c0", "g_str": {
        "token": "TURN_LEFT", "args": [49.997, 0, 0, 0, 0, 0, 0, 0],
        "arg_mask": [1, 0, 0, 0, 0, 0, 0, 0], "provenance": "path",
        "sources": ["engine_a.route_v3=turn_left"], "confidence": 0.9},
        "a_str": {"token": "HOLD_CORRIDOR",
                  "args": [0, 0, 0, 0, 0, 0, 67.9, 0],
                  "arg_mask": [0, 0, 0, 0, 0, 0, 1, 0],
                  "provenance": "path", "confidence": 0.7},
        "disjointness": {"situation_classifier_output_used": False}}


def test_masked_off_arguments_are_not_printed_as_decisions():
    """⛔ An arg slot with `arg_mask == 0` carries a filler 0.0. Printing it
    would read as 'stop in 0 m' — a decision the label never took."""
    from ph0_rich_overlay import _tok_args
    assert _tok_args(_s2()["g_str"]) == "TURN_LEFT(a0=49.997)"
    assert _tok_args({"token": "FOLLOW_MAIN_ROAD", "args": [0.0] * 8,
                      "arg_mask": [0] * 8}) == "FOLLOW_MAIN_ROAD"
    assert _tok_args(None) == "—"


def test_strategic_label_is_drawn_beside_the_perception_that_feeds_it():
    """The PI's standing viz standard: camera + metric BEV + the decoded
    strategic goal in ONE frame, so perception and the label it produces are
    judged together rather than in two documents."""
    from ph0_rich_overlay import draw_panel
    import numpy as np
    with_s2 = np.asarray(draw_panel((470, 600), _rec(), _sam(), _ea(), 0,
                                    _s2()))
    without = np.asarray(draw_panel((470, 600), _rec(), _sam(), _ea(), 0,
                                    None))
    assert with_s2.shape == without.shape
    assert not np.array_equal(with_s2, without), "the S2 block must render"


def test_the_liveness_alarm_reaches_the_figure():
    """⛔ C77 ON THE FIGURE. A viewer looking at an all-zero clip must be able
    to tell an empty scene from a dead engine WITHOUT opening the JSON."""
    from ph0_rich_overlay import draw_panel
    import numpy as np
    dead = dict(_sam(), per_concept_hits={"car": 0}, n_err_total=42,
                err_kinds={"RuntimeError": 42},
                liveness={"n_det": {"road": 0, "sky": 0}, "live": False})
    alive = dict(_sam(),
                 liveness={"n_det": {"road": 2, "sky": 1}, "live": True})
    a = np.asarray(draw_panel((470, 600), _rec(), alive, _ea(), 0))
    d = np.asarray(draw_panel((470, 600), _rec(), dead, _ea(), 0))
    assert not np.array_equal(a, d)


def test_the_alignment_warning_is_read_off_the_record_not_hard_coded():
    """⚠️ The banner was hard-coded while the frame-snapping bug existed.
    `run_clip_frames` now runs every grounded frame exactly and stamps
    `frame_aligned`, so a fixed banner would print a stale claim onto every
    future figure."""
    import inspect
    from ph0_rich_overlay import draw_panel
    src = inspect.getsource(draw_panel)
    assert 'c.get("frame_aligned")' in src
    assert src.index("ALIGNMENT UNVERIFIED") > src.index('frame_aligned')


def test_render_clip_still_composes_with_a_strategic_label(tmp_path):
    from ph0_rich_overlay import clip_layout, compose_frame
    fr = _frames()
    lay = clip_layout(_rec(), _sam(), fr)
    img = compose_frame(_rec(), _sam(), fr, _ea(), lay, 0, _s2())
    assert img.size == (lay["W"], lay["H"])


def test_bev_carries_a_metric_scale_bar():
    """⭐ The PI's standard asks for a METRIC BEV. Equal-aspect axes plus one
    scale bar state the whole mapping; without it a 20 m lane change and a
    200 m sweep draw the same squiggle and the panel is decoration."""
    import inspect
    from ph0_rich_overlay import draw_bev
    src = inspect.getsource(draw_bev)
    assert "m fwd" in src and "{target:g} m" in src
    import numpy as np
    near = np.asarray(draw_bev((400, 210), _ea(), 0, 6))
    ea_far = dict(_ea(), polyline_xy=[[i * 5.0, 0.0] for i in range(200)])
    far = np.asarray(draw_bev((400, 210), ea_far, 0, 6))
    assert not np.array_equal(near, far), "the scale must react to extent"


def test_font_fallback_reaches_a_real_ttf_when_one_exists():
    """⚠️ `ImageFont.load_default()` has no em dash, arrow or warning glyph —
    on a host without DejaVu the panel's load-bearing clauses became tofu
    boxes. MEASURED on the Windows dev box 2026-08-16."""
    import inspect
    from ph0_rich_overlay import _font
    src = inspect.getsource(_font)
    assert "C:\\Windows\\Fonts" in src or r"C:\Windows\Fonts" in src
    f = _font(12)
    assert f is not None


def test_the_figure_recomputes_liveness_instead_of_trusting_the_flag():
    """⛔ `live` is a DERIVED field whose rule already changed once (all -> any,
    after an underpass returned `road 2 · sky 0` on a working engine). The
    counts are banked, so the figure recomputes — and it distinguishes an
    occluded control from a dead engine rather than printing one alarm for
    both."""
    import inspect
    import numpy as np
    from ph0_rich_overlay import draw_panel
    src = inspect.getsource(draw_panel)
    assert "one control occluded" in src
    assert 'alive = any(int(v) > 0' in src
    occluded = dict(_sam(),
                    liveness={"n_det": {"road": 2, "sky": 0}, "live": False})
    dead = dict(_sam(), liveness={"n_det": {"road": 0, "sky": 0},
                                  "live": True})   # a LYING stored flag
    a = np.asarray(draw_panel((470, 600), _rec(), occluded, _ea(), 0))
    d = np.asarray(draw_panel((470, 600), _rec(), dead, _ea(), 0))
    assert not np.array_equal(a, d), "the stored flag must not drive the panel"
