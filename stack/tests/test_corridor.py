"""Tests for E7.1: strategic corridor labels (scripts/refb_labels.py, E7.1
section).

Pins:
  (1) straight highway episode -> CORRIDOR_FOLLOW everywhere judgeable; valid
      exactly where >= min_steps of future exist.
  (2) R=20 m junction turn placed 5-15 s ahead -> CORRIDOR_LEFT / _RIGHT with
      the correct side, valid=True (tight AND transient fires).
  (3) gentle R=300 m sweep netting > 45 deg -> CORRIDOR_FOLLOW (curvature-
      relative), AND the v1 net-heading rule (`nav_command`) would have said
      LEFT on the same poses — the documented v1 degeneracy, pinned so the
      difference stays measured, not asserted.
  (4) too-short future -> valid=False everywhere, labels parked at FOLLOW
      (masked placeholder, never a judgement — the D2 lesson).
  (5) corridor_sequence: t_idx == arange(0, T, stride) and every row equals
      corridor_labels at that t (default and non-default stride).
  (6) stratified_corridor_report on a synthetic 2-class map: counts / fracs /
      valid_frac / turn_frac per class, overall totals, unmapped clip
      reported (never silently dropped).
CPU-only, synthetic trajectories.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refb_labels as R  # noqa: E402

DT = R.DT_DEFAULT


def _poses(T, dt=DT, v0=8.0, yaw_rate=0.0):
    rows, x, y, yaw, v = [], 0.0, 0.0, 0.0, v0
    for _ in range(T):
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
    return torch.tensor(rows, dtype=torch.float32)


def _poses_piecewise(segments, dt=DT, v0=8.0):
    """Poses from [(n_steps, yaw_rate), ...] at constant speed."""
    rows, x, y, yaw, v = [], 0.0, 0.0, 0.0, v0
    for n, yr in segments:
        for _ in range(n):
            rows.append([x, y, yaw, v])
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            yaw += yr * dt
    return torch.tensor(rows, dtype=torch.float32)


def _junction_episode(side=+1, T=300, v=8.0, radius_m=20.0, pre_s=5.0):
    """Straight for pre_s seconds, then a 90-deg turn at `radius_m`, then
    straight — the turn sits ~5-9 s ahead of t=0 (inside the 5-15 s band)."""
    yaw_rate = side * v / radius_m                       # 0.4 rad/s at defaults
    pre = int(round(pre_s / DT))
    turn = int(round((math.pi / 2) / abs(yaw_rate) / DT))  # ~39 steps
    post = T - pre - turn
    assert post > 0
    return _poses_piecewise([(pre, 0.0), (turn, yaw_rate), (post, 0.0)], v0=v)


# ---------- (1) straight highway -> FOLLOW, valid ----------------------------

def test_straight_highway_follow_and_valid_pattern():
    T = 500
    p = _poses(T, v0=30.0)
    lab, val = R.corridor_labels(p)
    assert lab.shape == (T,) and lab.dtype == torch.long
    assert val.shape == (T,) and val.dtype == torch.bool
    # valid iff >= NAV_MIN_STEPS (150) of future exist: t <= T-1-150
    cut = T - R.NAV_MIN_STEPS
    assert val[:cut].all() and not val[cut:].any()
    assert (lab[val] == R.CORRIDOR_FOLLOW).all()


# ---------- (2) junction turn ahead -> correct side, valid -------------------

def test_junction_turn_left_ahead():
    p = _junction_episode(side=+1)
    lab, val = R.corridor_labels(p)
    assert bool(val[0])
    assert int(lab[0]) == R.CORRIDOR_LEFT


def test_junction_turn_right_ahead():
    p = _junction_episode(side=-1)
    lab, val = R.corridor_labels(p)
    assert bool(val[0])
    assert int(lab[0]) == R.CORRIDOR_RIGHT


# ---------- (3) gentle sweep: FOLLOW here, LEFT under the v1 rule ------------

def test_gentle_sweep_follows_but_v1_net_heading_would_turn():
    # R = v / yaw_rate = 30 / 0.1 = 300 m; over 25 s the road sweeps
    # 0.1 * 25 = 2.5 rad ~ 143 deg of net heading — pure lane-keeping.
    p = _poses(260, v0=30.0, yaw_rate=0.1)
    lab, val = R.corridor_labels(p)
    assert bool(val[0])
    assert int(lab[0]) == R.CORRIDOR_FOLLOW          # curvature-relative: road
    # The v1 net-heading rule mints a LEFT on the very same poses (143 > 45
    # deg) — the degeneracy the module docstring documents. Pinned so the
    # corridor labeler's deviation from `nav_command` stays a measured fact.
    nav, nav_valid = R.nav_command(p, 0)
    assert nav_valid and nav == R.NAV_LEFT


# ---------- (4) too-short future -> valid=False ------------------------------

def test_too_short_future_invalid_everywhere():
    p = _poses(100, v0=8.0)                          # < NAV_MIN_STEPS anywhere
    lab, val = R.corridor_labels(p)
    assert not val.any()
    assert (lab == R.CORRIDOR_FOLLOW).all()          # masked placeholder only


# ---------- (5) corridor_sequence stride / t-index ---------------------------

def test_corridor_sequence_matches_dense_labels():
    p = _junction_episode(side=+1, T=300)
    lab, val = R.corridor_labels(p)
    sl, sv, st = R.corridor_sequence(p, stride=10)
    assert (st == torch.arange(0, 300, 10)).all()
    assert sl.shape == sv.shape == st.shape == (30,)
    assert sl.dtype == torch.long and sv.dtype == torch.bool
    assert (sl == lab[st]).all()
    assert (sv == val[st]).all()


def test_corridor_sequence_nondefault_stride_and_kwargs():
    p = _poses(200, v0=10.0)
    sl, sv, st = R.corridor_sequence(p, stride=7, min_steps=50)
    n = math.ceil(200 / 7)
    assert st.shape == (n,) and int(st[-1]) == 7 * (n - 1)
    # min_steps forwarded: valid iff >= 50 steps of future at that t
    expect = (200 - 1 - st) >= 50
    assert (sv == expect).all()
    assert (sl[sv] == R.CORRIDOR_FOLLOW).all()


# ---------- (6) stratified report on a synthetic 2-class map -----------------

def test_stratified_corridor_report_two_classes_and_unmapped():
    F, L, Rt = R.CORRIDOR_FOLLOW, R.CORRIDOR_LEFT, R.CORRIDOR_RIGHT
    labels_by_clip = {
        "clip_int": (torch.tensor([F, L, Rt, F]),
                     torch.tensor([True, True, True, False])),
        "clip_hwy": (torch.tensor([F] * 5), torch.tensor([True] * 5)),
        "clip_lost": ([F, L], [True, True]),         # plain lists also accepted
    }
    road = {"clip_int": "intersection_rich", "clip_hwy": "highway"}
    rep = R.stratified_corridor_report(labels_by_clip, road)

    ic = rep["per_class"]["intersection_rich"]
    assert ic["n_clips"] == 1 and ic["n_windows"] == 4 and ic["n_valid"] == 3
    assert ic["counts"] == {"follow": 1, "left": 1, "right": 1}
    assert abs(ic["valid_frac"] - 0.75) < 1e-9
    assert abs(ic["turn_frac"] - 2.0 / 3.0) < 1e-9

    hw = rep["per_class"]["highway"]
    assert hw["n_windows"] == 5 and hw["n_valid"] == 5
    assert hw["turn_frac"] == 0.0 and hw["valid_frac"] == 1.0

    # the point of the stratification: intersection-rich carries the turns
    assert ic["turn_frac"] > hw["turn_frac"]

    # unmapped clip is reported AND counted under its own bucket, not dropped
    assert rep["unmapped_clips"] == ["clip_lost"]
    um = rep["per_class"]["unmapped"]
    assert um["n_windows"] == 2 and um["counts"]["left"] == 1

    ov = rep["overall"]
    assert ov["n_clips"] == 3 and ov["n_windows"] == 11 and ov["n_valid"] == 10
    assert ov["counts"] == {"follow": 7, "left": 2, "right": 1}


def test_stratified_report_rejects_length_mismatch_and_bad_class():
    import pytest
    with pytest.raises(ValueError):
        R.stratified_corridor_report({"a": ([0, 1], [True])}, {"a": "urban"})
    with pytest.raises(ValueError):
        R.stratified_corridor_report({"a": ([7], [True])}, {"a": "urban"})
