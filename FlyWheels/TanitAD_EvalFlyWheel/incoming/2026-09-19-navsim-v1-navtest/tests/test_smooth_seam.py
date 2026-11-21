"""SPEC §7b's cubic smoother against ANALYTIC targets — never against its own output.

    PYTHONPATH=D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval \
      C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q tests/test_smooth_seam.py

A smoother that changes a path it should leave alone would manufacture an A1S effect. So the
strongest checks are the fixed points: a path that IS an origin-anchored cubic must come back
unchanged, and a straight constant-speed path must stay exactly straight.
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("w3_smooth", PKG / "code" / "smooth_seam.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["w3_smooth"] = m
    spec.loader.exec_module(m)
    return m


M = _mod()
T = np.arange(1, 9) * 0.5


def test_an_anchored_cubic_is_a_fixed_point():
    """x = 5t + 0.2t^2 - 0.01t^3, y = 0.1t^2 + 0.02t^3 is exactly representable: it must return
    unchanged, heading included (atan2 of the exact derivative)."""
    x = 5 * T + 0.2 * T ** 2 - 0.01 * T ** 3
    y = 0.1 * T ** 2 + 0.02 * T ** 3
    h = np.arctan2(0.2 * T + 0.06 * T ** 2, 5 + 0.4 * T - 0.03 * T ** 2)
    p = np.stack([x, y, h], 1)
    assert np.abs(M.cubic_smooth(p) - p).max() < 1e-9


def test_a_straight_constant_speed_path_stays_exactly_straight():
    p = np.stack([10.0 * T, np.zeros(8), np.zeros(8)], 1)
    out = M.cubic_smooth(p)
    assert np.abs(out[:, 1]).max() < 1e-12
    assert np.abs(out[:, 2]).max() < 1e-12
    assert np.abs(out[:, 0] - 10.0 * T).max() < 1e-9


def test_lateral_jitter_loses_its_sign_flips():
    """A straight path with ±0.3 m alternating lateral jitter flips its heading change at every
    step. The fitted path is a parametric cubic, whose signed curvature numerator
    x'y'' - y'x'' is a polynomial of degree <= 2 in t — so it can flip sign at most TWICE,
    whatever the input. That bound is analytic, not a property of this implementation."""
    y = 0.3 * np.array([(-1) ** k for k in range(1, 9)], dtype=float)
    p = np.stack([8.0 * T, y, np.zeros(8)], 1)
    out = M.cubic_smooth(p)

    def flips(q):
        d = np.diff(np.vstack([[0, 0, 0], q])[:, :2], axis=0)
        hd = np.arctan2(d[:, 1], d[:, 0])
        dh = np.diff(hd)
        return int(np.sum(np.sign(dh[:-1]) * np.sign(dh[1:]) < 0))

    assert flips(p) >= 5                      # the control: the input really is jittery
    assert flips(out) <= 2


def test_below_the_speed_floor_the_input_heading_is_kept():
    """A near-standstill plan has no defined direction of travel; atan2 of a ~zero derivative
    would invent one. Below 0.5 m/s the model's own heading must pass through untouched."""
    p = np.stack([0.01 * T, np.zeros(8), np.full(8, 0.7)], 1)   # 0.02 m/s, heading 0.7 rad
    out = M.cubic_smooth(p)
    assert np.allclose(out[:, 2], 0.7, atol=0.0)


def test_the_basis_has_no_constant_term_so_the_fit_is_anchored_at_the_ego():
    """p(0) = 0 by construction: every basis column vanishes at t = 0."""
    assert M.BASIS.shape == (8, 3)
    assert np.allclose(M.BASIS[:, 0], T) and np.allclose(M.BASIS[:, 2], T ** 3)
    assert M.V_MIN == 0.5
