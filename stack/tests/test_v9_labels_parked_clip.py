"""WP-A v9 builder: a PARKED recording (whole 10 Hz track shorter than the 0.5 m smoothing step) must build.

MEASURED 2026-10-12 while building the 4,572-clip train release: ``12c90362-a57f-4c3d-bf0d-ebe47a687186`` (one of the
203 ADD clips) has x/y range exactly 0.0 and v == 0 for its whole 20.35 s. ``smooth_path`` made a ONE-point arc grid and
``np.gradient`` raised ValueError, killing ``lead-pass`` and ``build`` for the whole corpus. None of the 4,369 older clips
is that short, so nothing had ever exercised it. Every expectation below is a LITERAL (a parked car has arc 0.0 m and
two path points 0.5 m apart), never an expression over the code under test.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
B = pytest.importorskip("build_v9_labels")


def parked_log(t_end=20.0, dt=0.01):
    t = np.arange(0.0, t_end + 1e-9, dt)
    z = np.zeros_like(t)
    return B.log_from_arrays(t, z, z, z, z)


def test_parked_track_has_zero_arc():
    tr = B.track10(parked_log())
    assert float(tr.s[-1]) == 0.0                     # literal: the ego never moves


def test_smooth_path_on_a_parked_track_returns_two_points_not_a_crash():
    tr = B.track10(parked_log())
    for sigma in (0.0, 8.0, 25.0):
        p = B.smooth_path(tr, sigma)
        assert len(p.s) == 2 and float(p.s[0]) == 0.0 and float(p.s[1]) == 0.5
        assert np.all(p.x == 0.0) and np.all(p.y == 0.0)


def test_clip_geo_and_labels_build_for_a_parked_recording():
    log = parked_log()
    G = B.clip_geo(log, None)                          # raised ValueError before the patch
    now = np.array([5.0, 6.0, 7.0])
    k = np.round(now / 0.1).astype(int)
    F = B.clip_labels(G, now, k, None, None, None, lc_labels=False)
    assert len(F["lat_cls_a"]) == 3
    assert list(F["lat_cls_a"]) == [0, 0, 0]          # LANE_KEEP: LAT9[0]
    assert list(F["lon_cls"]) == [0, 0, 0]            # HOLD: LON9[0] -- a parked car holds
    assert list(F["nav_token"]) == [0, 0, 0]          # FOLLOW: no turn ahead of a car that never moves
    assert not np.any(F["rc_A50_valid"] == 1)         # a 0.5 m path cannot supply a 50 m route checkpoint
    assert not np.any(F["rcH_A50_valid"] == 1)


def test_lead_for_frame_on_a_parked_ego_reports_no_lead():
    log = parked_log()
    tr = B.track10(log)
    p0 = B.smooth_path(tr, 0.0)
    L, *_ = B.lead_for_frame(log, tr, p0, 10.0, [{"cls": "automobile", "cx": 12.0, "cy": 0.0, "l": 4.6}])
    assert L is False                                  # a 0.5 m path cannot reach the 2 m minimum lead distance
