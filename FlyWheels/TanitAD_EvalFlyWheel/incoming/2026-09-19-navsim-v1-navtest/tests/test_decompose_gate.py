"""The per-token lateral read against ANALYTIC targets — never against its own output.

    PYTHONPATH=D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval \
      C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q tests/test_decompose_gate.py

``decompose_gate.lateral_at_horizon`` is the only new arithmetic behind the claim that refcv4b's
drivable-area failures are LATERAL (cross-track 4.6x, heading 3.1x, FDE ~equal on the 200-token
read). A cross-check derived from the value it checks measures determinism, not correctness, so
every expectation here is a closed-form geometric fact written as a literal.
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("w3_decomp", PKG / "code" / "decompose_gate.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["w3_decomp"] = m
    spec.loader.exec_module(m)
    return m


M = _mod()


def _traj(x, y, h):
    t = np.zeros((8, 3))
    t[-1] = (x, y, h)
    return t


def test_identical_paths_read_exactly_zero():
    r = M.lateral_at_horizon(_traj(20.0, 1.0, 0.1), _traj(20.0, 1.0, 0.1))
    assert r == {"heading_err_deg_4s": 0.0, "xtrack_m_4s": 0.0, "fde_m_4s": 0.0}


def test_a_pure_lateral_offset_is_all_cross_track():
    """Human ends heading +x; the model ends 1 m to its left: cross-track 1, FDE 1, heading 0."""
    r = M.lateral_at_horizon(_traj(20.0, 1.0, 0.0), _traj(20.0, 0.0, 0.0))
    assert r["xtrack_m_4s"] == pytest.approx(1.0, abs=1e-12)
    assert r["fde_m_4s"] == pytest.approx(1.0, abs=1e-12)
    assert r["heading_err_deg_4s"] == 0.0


def test_a_pure_along_track_offset_has_zero_cross_track():
    """⭐ The discriminating case: an endpoint 3 m SHORT of the human's is a longitudinal error.
    FDE sees 3 m; cross-track must see 0 — which is exactly why FDE alone cannot tell the two
    failure kinds apart."""
    r = M.lateral_at_horizon(_traj(17.0, 0.0, 0.0), _traj(20.0, 0.0, 0.0))
    assert r["fde_m_4s"] == pytest.approx(3.0, abs=1e-12)
    assert r["xtrack_m_4s"] == pytest.approx(0.0, abs=1e-12)


def test_cross_track_is_measured_in_the_humans_frame_not_the_egos():
    """Human ends heading +y (pi/2). A model 1 m further along world -x is 1 m to the human's
    LEFT — a world-frame dy would read 0 here and a world-frame dx would read the wrong axis."""
    r = M.lateral_at_horizon(_traj(-1.0, 10.0, math.pi / 2), _traj(0.0, 10.0, math.pi / 2))
    assert r["xtrack_m_4s"] == pytest.approx(1.0, abs=1e-12)


def test_heading_error_wraps_through_pi():
    """179 deg vs -179 deg is a 2 deg error, not 358."""
    r = M.lateral_at_horizon(_traj(0.0, 0.0, math.radians(179.0)),
                             _traj(0.0, 0.0, math.radians(-179.0)))
    assert r["heading_err_deg_4s"] == pytest.approx(2.0, abs=1e-9)
