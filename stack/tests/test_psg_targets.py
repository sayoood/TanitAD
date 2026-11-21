"""E-DEC-18 — PSG targets: registration, the leak guard, and the FOV boundary.

⚠️ The registration tests are the load-bearing ones. A sign error here does not
crash and does not show in a loss curve -- it teaches the encoder a MIRRORED
world, which is worse than no supervision.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tanitad.data.psg_targets import (PSG_HFOV_DEG, PSG_N_COLS, azimuth_column,
                                      clip_split, frame_target,
                                      unsupervised_probes)


def test_plus_y_is_left_so_positive_cy_lands_left_of_centre():
    """+x fwd, +y LEFT (MEASURED, bev_raster.py / build_obstacle_join.py:15).
    Image column 0 is the LEFTMOST column, so a POSITIVE cy must give a SMALLER
    column index than a negative one."""
    left = azimuth_column(20.0, +10.0)
    right = azimuth_column(20.0, -10.0)
    assert left is not None and right is not None
    assert left < right, (
        f"+y is LEFT: cy=+10 gave col {left}, cy=-10 gave col {right}. "
        "If this inverts, every agent is supervised into the mirrored column.")


def test_symmetric_pairs_are_mirror_images_about_the_centre():
    for cy in (2.0, 6.0, 14.0):
        l = azimuth_column(20.0, +cy)
        r = azimuth_column(20.0, -cy)
        assert l is not None and r is not None
        assert l + r == PSG_N_COLS - 1, (cy, l, r)


def test_agents_behind_the_camera_get_no_column():
    """⛔ Folding cx<=0 into an edge column would put traffic BEHIND the car
    into the leftmost/rightmost cell."""
    assert azimuth_column(-5.0, 0.0) is None
    assert azimuth_column(0.0, 3.0) is None


def test_outside_the_120_degree_field_gets_no_column():
    beyond = math.tan(math.radians(PSG_HFOV_DEG / 2 + 5)) * 10.0
    assert azimuth_column(10.0, beyond) is None
    assert azimuth_column(10.0, -beyond) is None
    inside = math.tan(math.radians(PSG_HFOV_DEG / 2 - 5)) * 10.0
    assert azimuth_column(10.0, inside) is not None


def test_frame_target_shape_and_empty_scene_is_all_zero():
    t = frame_target([])
    assert t.shape == (PSG_N_COLS, 2)
    assert not t.any(), "an empty scene must read exactly zero in both channels"


def test_nearness_channel_is_bounded_and_monotone_in_range():
    near = frame_target([{"cx": 5.0, "cy": 0.0}])[:, 1].max()
    far = frame_target([{"cx": 60.0, "cy": 0.0}])[:, 1].max()
    assert 0.0 <= far < near <= 1.0
    beyond = frame_target([{"cx": 500.0, "cy": 0.0}])[:, 1].max()
    assert beyond == 0.0, "past r_max must clamp to 0, never go negative"


def test_count_channel_is_log1p_not_raw():
    """Two agents that genuinely share a column must accumulate."""
    a = {"cx": 20.0, "cy": -3.0}          # az = -8.5 deg  -> col 4
    b = {"cx": 22.0, "cy": -3.3}          # az = -8.5 deg  -> col 4
    assert azimuth_column(a["cx"], a["cy"]) == azimuth_column(b["cx"], b["cy"])
    one = frame_target([a])[:, 0].max()
    two = frame_target([a, b])[:, 0].max()
    assert one == pytest.approx(math.log1p(1))
    assert two == pytest.approx(math.log1p(2))


def test_there_is_no_centre_column_and_the_lead_straddles_two():
    """⚠️ DOCUMENTING TEST -- a real property of the settled geometry, found by
    this suite. With an EVEN column count the optical axis falls exactly ON the
    3|4 boundary, so the single most important agent -- the lead vehicle, which
    sits near az = 0 by definition -- is split between columns 3 and 4 and will
    JITTER between them frame to frame as its lateral offset changes sign.

    This is not a defect to fix by shifting the bins: the columns must match the
    readout's 8 (E-DEC-2), and re-centring them would de-register the target
    from the thing it supervises. It IS a constraint on any consumer: a
    lead-related quantity must be read from columns 3 AND 4 together, never from
    "the centre column", which does not exist.
    """
    assert PSG_N_COLS % 2 == 0
    assert azimuth_column(20.0, +0.01) == PSG_N_COLS // 2 - 1
    assert azimuth_column(20.0, -0.01) == PSG_N_COLS // 2


def test_clip_split_is_disjoint_deterministic_and_covers_everything():
    """⛔ THE LEAK GUARD. The PSG target DETERMINES n_agents and lead_gap_m, so
    supervising and scoring on the same clips would measure the training label."""
    ids = [f"clip{i:03d}" for i in range(130)]
    tr, ho = clip_split(ids)
    assert set(tr).isdisjoint(ho), "a clip in both splits is the leak itself"
    assert sorted(tr + ho) == sorted(ids)
    assert clip_split(ids) == clip_split(list(reversed(ids))), "must be order-free"
    assert 0 < len(ho) < len(ids)


def test_unsupervised_probes_are_absent_from_the_supervised_target():
    """These two must be derivable from the cuboids yet NOT recoverable from
    frame_target -- otherwise the 'generalisation' test is another leak."""
    a = [{"cx": 20.0, "cy": 1.0, "occ": 1, "yaw": 0.5},
         {"cx": 20.0, "cy": 1.0, "occ": 0, "yaw": -0.5}]
    b = [{"cx": 20.0, "cy": 1.0, "occ": 0, "yaw": 0.0},
         {"cx": 20.0, "cy": 1.0, "occ": 0, "yaw": 0.0}]
    assert np.array_equal(frame_target(a), frame_target(b)), (
        "the two scenes must be INDISTINGUISHABLE to the supervised target")
    pa, pb = unsupervised_probes(a), unsupervised_probes(b)
    assert pa["frac_occluded"] != pb["frac_occluded"]
    assert pa["mean_abs_yaw"] != pb["mean_abs_yaw"]
