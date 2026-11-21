"""Tests for the border-clipped, sampled cuboid overlay (box-head audit 2026-09-27).

Literal expectations on a nominal 416x1024 cylindrical camera (f_ref 488.924, 1.5 m high, 2.0 m forward):

* a fully visible car draws all 12 edges, and each edge's run starts and ends EXACTLY on its projected corners;
* a bus alongside, crossing the +-60 deg border, draws strictly MORE edges than the old both-corners rule and
  every run end that is not a corner lies ON the frame border;
* a long horizontal edge bends: the sampled curve departs from the corner-to-corner chord by > 3 px;
* a box behind the camera draws nothing.
⛔ The RED arm: the old rule (edge kept iff both corners in frame) is re-implemented here and must FAIL the
border test -- the property the patch exists for.

Run: PYTHONPATH=<tree>/stack python -m pytest test_draw_cuboid_clipped.py
The module under test is the patched ``render_refcv6_map_video.py`` next to this file, else
``taniteval/tools/render_refcv6_map_video.py`` of the repo this file sits in.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
CAND = [HERE / "render_refcv6_map_video.py", HERE.parent / "tools" / "render_refcv6_map_video.py"]


def _load():
    p = next((c for c in CAND if c.exists()), None)
    if p is None:
        pytest.skip("render_refcv6_map_video.py not found next to the test or in ../tools")
    spec = importlib.util.spec_from_file_location("rmv_under_test", str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _load()


def _cam():
    from tanitad.data.calib import CanonicalFrame
    from tanitad.data.rig_projection import RigCamera
    fr = CanonicalFrame(height=416, width=1024, f_ref=488.92398517830253, projection="cylindrical")
    return RigCamera.nominal(fr, height_m=1.5, x_m=2.0)


def _corners_px(cam, cor):
    c, r, ok = cam.project(torch.as_tensor(cor, dtype=torch.float64))
    return c.numpy(), r.numpy(), ok.numpy().astype(bool)


def _old_rule_edges(cam, cor):
    _c, _r, ok = _corners_px(cam, cor)
    return sum(1 for a, b in M.CUBOID_EDGES if ok[a] and ok[b])


class _Rec:
    """A PIL-ImageDraw stand-in that records polylines."""
    def __init__(self):
        self.lines = []

    def line(self, pts, fill=None, width=0, joint=None):
        self.lines.append(list(pts))


def test_fully_visible_car_draws_12_edges_with_exact_corner_ends():
    cam = _cam()
    cor = M.cuboid_corners(15.0, 0.5, 0.8, 4.5, 1.9, 1.6, 0.3)
    runs = M.cuboid_edge_runs(cam, cor)
    assert len(runs) == 12 and sorted(e for e, _ in runs) == list(range(12))
    c, r, ok = _corners_px(cam, cor)
    assert ok.all()
    for e, pts in runs:
        a, b = M.CUBOID_EDGES[e]
        assert math.hypot(pts[0][0] - c[a], pts[0][1] - r[a]) < 1e-9
        assert math.hypot(pts[-1][0] - c[b], pts[-1][1] - r[b]) < 1e-9
    rec = _Rec()
    assert M.draw_cuboid_cam(rec, cam, cor, (0, 255, 0)) == 12 and len(rec.lines) == 12


def _on_border(p, W=1024, H=416, tol=0.05):
    x, y = p
    return min(abs(x), abs(x - (W - 1)), abs(y), abs(y - (H - 1))) < tol


def test_bus_crossing_the_border_is_clipped_not_dropped():
    cam = _cam()
    # the PI's bus #39 geometry: ~12 x 3 x 3.2 m, front at x ~ 12 m, 9.5 m to the LEFT -> rear corners beyond 60 deg
    cor = M.cuboid_corners(8.0, 9.5, 1.6, 12.0, 3.0, 3.2, 0.0)
    c, r, ok = _corners_px(cam, cor)
    assert (~ok).any() and ok.any(), "fixture must straddle the frame border"
    runs = M.cuboid_edge_runs(cam, cor)
    n_new = len({e for e, _ in runs})
    n_old = _old_rule_edges(cam, cor)
    assert n_new > n_old, (n_new, n_old)
    corner_px = [(c[k], r[k]) for k in range(8) if ok[k]]
    for _e, pts in runs:
        for end in (pts[0], pts[-1]):
            is_corner = any(math.hypot(end[0] - q[0], end[1] - q[1]) < 1e-6 for q in corner_px)
            assert is_corner or _on_border(end), end


def test_red_arm_old_rule_fails_the_border_property():
    """⛔ must FAIL for the old rule: on the bus it keeps fewer edges than there are edges with an in-frame part."""
    cam = _cam()
    cor = M.cuboid_corners(8.0, 9.5, 1.6, 12.0, 3.0, 3.2, 0.0)
    n_with_part = len({e for e, _ in M.cuboid_edge_runs(cam, cor)})
    assert _old_rule_edges(cam, cor) < n_with_part


def test_long_horizontal_edge_bends_on_the_cylinder():
    cam = _cam()
    # a 12 m side 4 m to the right, running from 4 m to 16 m ahead, at 3 m height (a bus roof edge)
    cor = M.cuboid_corners(10.0, -4.0, 1.5, 12.0, 0.5, 3.0, 0.0)
    runs = dict((e, pts) for e, pts in M.cuboid_edge_runs(cam, cor))
    dev = 0.0
    for e, pts in runs.items():
        p = np.asarray(pts)
        a, b = p[0], p[-1]
        ab = b - a
        n = np.linalg.norm(ab)
        if n < 1:
            continue
        d = np.abs((p[:, 0] - a[0]) * ab[1] - (p[:, 1] - a[1]) * ab[0]) / n
        dev = max(dev, float(d.max()))
    assert dev > 3.0, dev


def test_box_behind_the_camera_draws_nothing():
    cam = _cam()
    cor = M.cuboid_corners(-10.0, 0.0, 0.8, 4.5, 1.9, 1.6, 0.0)
    assert M.cuboid_edge_runs(cam, cor) == []
    assert M.draw_cuboid_cam(_Rec(), cam, cor, (0, 255, 0)) == 0
