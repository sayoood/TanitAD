"""Pins ``render_refcv7_video``'s PURE geometry, panel orientation, plan errors and the fixed clip rule.

Every expected value is a LITERAL worked out by hand -- never an expression over the code under test --
and the orientation checks ship with a lateral-mirror mutation that must go RED (the classic BEV defect:
left drawn on the right).

Runs without the model, the data or the ``tanitad`` package: the tool is loaded by path and only its pure
functions are called (it imports numpy at module level; PIL inside the panel functions).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os

import numpy as np
import pytest

pytest.importorskip("PIL")

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOL = os.path.join(os.path.dirname(_HERE), "tools", "render_refcv7_video.py")


def _load():
    spec = importlib.util.spec_from_file_location("render_refcv7_video_under_test", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()


# ------------------------------------------------------------------------------------------- #
# metric -> panel pixels                                                                      #
# ------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("xy, px", [
    ((0.0, 0.0), (202.0, 673.0)),        # the ego: bottom centre
    ((100.0, 0.0), (202.0, 0.0)),        # 100 m ahead: top centre
    ((0.0, 30.0), (0.0, 673.0)),         # 30 m LEFT: the panel's LEFT edge
    ((0.0, -30.0), (404.0, 673.0)),      # 30 m RIGHT: the right edge
    ((50.0, -15.0), (303.0, 336.5)),
])
def test_m2px_known_values(xy, px):
    got = M.m2px(*xy)
    assert got[0] == pytest.approx(px[0], abs=1e-9)
    assert got[1] == pytest.approx(px[1], abs=1e-9)


def test_cell_of_known_values():
    assert M.cell_of(0.05, -29.95) == (0, 0)             # nearest row, RIGHTMOST column
    assert M.cell_of(37.26, 12.36) == (372, 423)
    assert M.cell_of(99.95, 29.95) == (999, 599)
    assert M.cell_of(-0.01, 0.0) is None and M.cell_of(10.0, 30.0) is None


def test_grid_to_image_puts_nearest_right_cell_at_bottom_right():
    g = np.zeros((1000, 600), np.uint8)
    g[0, 0] = 7                                           # x in [0, 0.1) m, y in [-30, -29.9) m (RIGHT)
    img = M.grid_to_image(g)
    assert img[999, 599] == 7 and int(img.sum()) == 7


def _block_codes():
    """SAM3 codes: everything NOT SEEN except an 'edge' (5) block 35-40 m ahead, 10-15 m to the LEFT."""
    g = np.full((1000, 600), 255, np.uint8)
    g[350:400, 400:450] = 5
    return g


def test_map_panel_block_lands_left_ahead_and_mirror_goes_red():
    g = _block_codes()
    lv = np.ones_like(g, dtype=bool)
    img = M.map_panel(g, g, lv, kind="gt")
    assert img.shape == (673, 404, 3)
    x, y = M.m2px(37.5, 12.5)                              # (117.83, 420.6): interior of the block
    assert tuple(int(v) for v in img[int(y), int(x)]) == (214, 64, 64)
    # ⛔ MUTATION: a lateral mirror (the classic orientation defect) must NOT find the block here
    xm, ym = M.m2px(37.5, -12.5)
    assert tuple(int(v) for v in img[int(ym), int(xm)]) != (214, 64, 64)
    mirrored = M.map_panel(g[:, ::-1].copy(), g[:, ::-1].copy(), lv, kind="gt")
    assert tuple(int(v) for v in mirrored[int(y), int(x)]) != (214, 64, 64)


def test_map_panel_unseen_is_hatched_and_unscored_is_dimmed():
    g = _block_codes()
    lv = np.zeros_like(g, dtype=bool)                     # the lift reaches nothing: drawn, not scored
    img = M.map_panel(g, g, lv, kind="gt")
    x, y = M.m2px(37.5, 12.5)
    assert tuple(int(v) for v in img[int(y), int(x)]) == (89, 26, 26)     # floor(214*.42), floor(64*.42)
    xh, yh = M.m2px(80.0, 0.0)                            # SAM3 never saw it: a hatch colour, never a class
    assert tuple(int(v) for v in img[int(yh), int(xh)]) in {(14, 16, 21), (70, 76, 90)}


def test_pred_panel_hatches_where_the_lift_cannot_reach():
    pred = np.full((1000, 600), 1, np.uint8)              # drivable everywhere
    gt = np.full((1000, 600), 1, np.uint8)
    lv = np.zeros_like(pred, dtype=bool)
    lv[:200] = True                                       # the lift reaches 0-20 m only
    img = M.map_panel(pred, gt, lv, kind="pred")
    x, y = M.m2px(10.0, 0.0)
    assert tuple(int(v) for v in img[int(y), int(x)]) == (46, 98, 178)
    xf, yf = M.m2px(60.0, 0.0)
    assert tuple(int(v) for v in img[int(yf), int(xf)]) in {(14, 16, 21), (150, 140, 96)}


# ------------------------------------------------------------------------------------------- #
# boxes and plan errors                                                                       #
# ------------------------------------------------------------------------------------------- #
def test_box_corners_known_values():
    np.testing.assert_allclose(M.box_corners_xy(10, 2, 4, 2, 0.0),
                               [[12, 3], [12, 1], [8, 1], [8, 3]], atol=1e-12)
    np.testing.assert_allclose(M.box_corners_xy(10, 2, 4, 2, math.pi / 2),
                               [[9, 4], [11, 4], [11, 0], [9, 0]], atol=1e-12)


def test_plan_errors_known_values():
    gt = np.array([[2.5 * k, 0.0] for k in range(1, 9)])
    r = M.plan_errors(gt + np.array([0.0, 1.0]), gt, [True] * 8)
    assert r["ade_m"] == pytest.approx(1.0) and r["fde_m"] == pytest.approx(1.0)
    r = M.plan_errors(gt + np.array([0.0, 1.0]), gt, [True] * 7 + [False])
    assert r["ade_m"] == pytest.approx(1.0) and r["fde_m"] is None and r["n_valid_slots"] == 7
    sel = gt.copy()
    sel[-1] += np.array([3.0, 4.0])
    r = M.plan_errors(sel, gt, [True] * 8)
    assert r["ade_m"] == pytest.approx(0.625) and r["fde_m"] == pytest.approx(5.0)


# ------------------------------------------------------------------------------------------- #
# the fixed clip rule                                                                         #
# ------------------------------------------------------------------------------------------- #
def _c(s, nav, v, vru, lead, full=True):
    return {"sha12": s, "nav": nav, "full_cov": full, "v_mean": v, "n_vru_win": vru, "n_lead_win": lead}


def _pool():
    out = []
    for nav, p in (("left", "a"), ("right", "b"), ("follow", "c")):
        out += [_c(p + "01", nav, 2.0, 0, 5), _c(p + "02", nav, 3.0, 4, 1), _c(p + "03", nav, 4.0, 4, 0),
                _c(p + "04", nav, 10.0, 0, 0), _c(p + "05", nav, 11.0, 0, 0), _c(p + "06", nav, 12.0, 2, 9),
                _c(p + "99", nav, 1.0, 50, 50, full=False)]       # NOT fully covered: never eligible
    return out


def test_select_clips_known_picks():
    chosen, rep = M.select_clips(_pool())
    # per token: median of (2,3,4,10,11,12) = 7.0 -> LOW = 01,02,03 ; HIGH = 04,05,06
    # LOW : VRU max 4 tied by 02/03 -> 02 (smaller sha12); LEAD of the rest (01:5, 03:0) -> 01
    # HIGH: VRU max 2 -> 06; LEAD of the rest (04:0, 05:0) all zero -> smallest sha12 04
    assert [c["sha12"] for c in chosen] == ["a02", "a01", "a06", "a04", "b02", "b01", "b06", "b04",
                                            "c02", "c01", "c06", "c04"]
    assert rep["by_token"]["left"]["v_mean_median_ms"] == 7.0
    assert "smallest sha12" in chosen[3]["why"]


def test_select_clips_refuses_without_any_vru():
    pool = [dict(c, n_vru_win=0) for c in _pool()]
    with pytest.raises(SystemExit):
        M.select_clips(pool)


# ------------------------------------------------------------------------------------------- #
# hygiene                                                                                     #
# ------------------------------------------------------------------------------------------- #
def test_scrub_and_refusal_of_raw_uuids():
    uid = "01234567-89ab-cdef-0123-456789abcdef"
    txt = M.dumps({"k": f"clip {uid}"})
    assert uid not in txt and "sha12:" in txt
    assert json.loads(txt)["k"].endswith(M.sha12(uid))


def test_palette_is_disjoint_from_paths():
    rep = M.assert_palette_disjoint()
    assert rep["orange_vs_amber_L1"] > 90
