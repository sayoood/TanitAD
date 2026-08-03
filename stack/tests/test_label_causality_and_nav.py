"""Tests for the three LABEL defects fixed 2026-08-03 (REF-C corpus-and-labels stream).

Each test pins ONE defect and its fix, and states the defect in its own docstring so
a future reader does not have to reconstruct it from a changelog:

  D1  situations.kinematics built `alon_pre` / `omega_pre` on `np.gradient`, a CENTRED
      difference, under a comment reading STRICTLY CAUSAL. Both channels therefore read
      one frame (0.1 s) past t.
  D2  route_from_future_v21 dropped v2's `min_steps` parameter, so the two live callers
      that still pass it hit a TypeError their bare `except` swallowed — their
      scene-length-adapted nav fallback had never run.
  D3  nav_command_v21 maps ROUTE_UNKNOWN and ROUTE_STRAIGHT onto the SAME NAV_FOLLOW, so
      the model INPUT cannot distinguish "the road goes straight" from "I do not know".

Plus a pin that the FACTORED tactical label projects EXACTLY onto the shipped 5-way one
(the property the retrain recommendation rests on) and that the v2.2 route labeler can
never regress v2.1's coverage.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_STACK = Path(__file__).resolve().parents[1]
if str(_STACK / "scripts") not in sys.path:
    sys.path.insert(0, str(_STACK / "scripts"))

import refb_labels as R                                       # noqa: E402
from tanitad.data import situations as S                      # noqa: E402
from tanitad.refs import refc_tactical as TAC                 # noqa: E402


# --------------------------------------------------------------------------- #
# D1 — the centred-difference causality break                                   #
# --------------------------------------------------------------------------- #
def _poses(T=200, v0=10.0, yaw_rate=0.0, dt=0.1):
    t = np.arange(T) * dt
    yaw = yaw_rate * t
    x = np.cumsum(np.cos(yaw) * v0 * dt)
    y = np.cumsum(np.sin(yaw) * v0 * dt)
    return np.stack([x, y, yaw, np.full(T, v0)], 1).astype(np.float64)


def test_backward_diff_is_strictly_causal():
    """A causal derivative at t may not change when the FUTURE changes."""
    x = np.arange(20, dtype=np.float64)
    d = S.backward_diff(x)
    x2 = x.copy()
    x2[10:] += 1000.0                       # perturb everything from t=10 onward
    d2 = S.backward_diff(x2)
    assert np.allclose(d[:10], d2[:10]), "backward_diff at t < 10 saw the future"
    # and np.gradient — the thing it replaces — DOES see it, one frame early
    g, g2 = np.gradient(x, S.DT), np.gradient(x2, S.DT)
    assert not np.allclose(g[9], g2[9]), (
        "np.gradient is supposed to be centred; if this fires, numpy changed and "
        "the premise of this whole fix needs re-checking")


def test_pre_channels_do_not_read_the_future():
    """`alon_pre` / `omega_pre` at t must be invariant to every sample after t.

    THE DEFECT, precisely: the trailing mean was causal but the derivative under it
    was `np.gradient`, so the composition read t+1. Perturbing the track strictly
    after frame k must leave every `_pre` value at index <= k untouched.
    """
    P = _poses(T=120, v0=8.0, yaw_rate=0.05)
    k = 60
    Q = P.copy()
    Q[k + 1:, 3] += 5.0                       # speed jumps AFTER k
    Q[k + 1:, 2] += 0.5                       # and so does yaw
    A, B = S.kinematics(P), S.kinematics(Q)
    for ch in ("alon_pre", "omega_pre"):
        assert np.allclose(A[ch][: k + 1], B[ch][: k + 1], atol=1e-9), (
            f"{ch} at t <= {k} changed when only t > {k} was perturbed — "
            "the channel is not causal")
    # the LEGACY channels are expected to fail exactly this test; that is the bug.
    assert not np.allclose(A["alon_pre_centred"][k], B["alon_pre_centred"][k]), (
        "the legacy centred channel should still leak — if it does not, the "
        "regression fixture stopped exercising the defect")


def test_causal_pre_is_the_default_and_legacy_is_reproducible():
    P = _poses(T=80, v0=6.0, yaw_rate=0.1)
    K = S.kinematics(P)
    assert K["pre_mode"] == "causal_backward_diff"
    assert np.allclose(K["alon_pre"], K["alon_pre_causal"])
    assert np.allclose(K["omega_pre"], K["omega_pre_causal"])
    L = S.kinematics(P, causal_pre=False)
    assert "LEGACY" in L["pre_mode"]
    assert np.allclose(L["alon_pre"], K["alon_pre_centred"])
    assert np.allclose(L["omega_pre"], K["omega_pre_centred"])


def test_detector_channels_are_untouched_by_the_fix():
    """⛔ The frozen thresholds must not move. `omega` / `kappa` / `alon` are LABEL
    channels (the PI allows labels to use the future) and their smoothing is centred
    BY DESIGN; only the `_pre` channels were meant to be causal."""
    P = _poses(T=150, v0=5.0, yaw_rate=0.2)
    K, L = S.kinematics(P), S.kinematics(P, causal_pre=False)
    for ch in ("omega", "kappa", "alon", "psi", "v", "x", "y"):
        assert np.allclose(K[ch], L[ch]), f"{ch} moved with the causal_pre switch"


def test_detectors_still_fire_after_the_fix():
    """A regression guard on the pre-registered detectors themselves."""
    T = 200
    yaw = np.zeros(T)
    yaw[80:110] = np.linspace(0, math.radians(90), 30)
    yaw[110:] = math.radians(90)
    P = np.zeros((T, 4))
    P[:, 2] = yaw
    P[:, 3] = 6.0
    P[:, 0] = np.cumsum(np.cos(yaw) * 0.6)
    P[:, 1] = np.cumsum(np.sin(yaw) * 0.6)
    K = S.kinematics(P)
    _ev, turns, _x = S.detect_intersection(K, cross=None)
    assert turns, "the quarter-turn fixture stopped producing a turn event"


# --------------------------------------------------------------------------- #
# D2 — the dropped `min_steps` parameter                                        #
# --------------------------------------------------------------------------- #
def test_route_v21_accepts_and_honours_min_steps():
    """Two live callers pass `min_steps=10` today and swallow the TypeError.

    MEASURED before the fix: every rollout row under
    `stack/experiments/alpasim-gsplat/results/openloop-thor-2026-08-03/rollouts/`
    carries `nav_short_err: TypeError(... unexpected keyword argument 'min_steps')`
    with `nav_short = 0, nav_short_valid = False`.
    """
    p = torch.zeros(200, 4)
    p[:, 0] = torch.arange(200).float()
    p[:, 3] = 10.0
    # accepted (this raised TypeError before the fix)
    R.nav_command_v21(p, 0, horizon_steps=50, min_steps=10)
    # honoured: a floor larger than the available future must refuse to judge
    r = R.route_from_future_v21(p, 0, horizon_steps=50, min_steps=60)
    assert r["reason"] == "no_future" and r["route"] == R.ROUTE_UNKNOWN
    assert r["valid"] is False
    # and None keeps pure-arc v2.1 semantics
    assert R.route_from_future_v21(p, 0, horizon_steps=50)["reason"] != "no_future"
    with pytest.raises(ValueError):
        R.route_from_future_v21(p, 0, min_steps=0)


def test_min_steps_default_is_byte_identical_to_before():
    p = torch.zeros(300, 4)
    p[:, 0] = torch.arange(300).float() * 0.8
    p[:, 3] = 8.0
    for t in (0, 50, 150, 250):
        a = R.route_from_future_v21(p, t)
        b = R.route_from_future_v21(p, t, min_steps=None)
        assert a == b


# --------------------------------------------------------------------------- #
# D3 — the nav INPUT collapses UNKNOWN onto FOLLOW                              #
# --------------------------------------------------------------------------- #
def _stopped(T=200):
    return torch.zeros(T, 4)


def _straight(T=200, v=10.0):
    p = torch.zeros(T, 4)
    p[:, 0] = torch.arange(T).float() * v * 0.1
    p[:, 3] = v
    return p


def test_nav_command_v21_really_does_collapse_unknown_onto_follow():
    """The DEFECT, pinned. Both a real judgement and a confession emit nav = 0."""
    known = R.nav_command_v21(_straight(), 0)
    unknown = R.nav_command_v21(_stopped(), 0)
    assert known[0] == unknown[0] == R.NAV_FOLLOW
    assert known[1] is True and unknown[1] is False


def test_nav_command_v21_ex_exposes_the_sentinel():
    k = R.nav_command_v21_ex(_straight(), 0)
    u = R.nav_command_v21_ex(_stopped(), 0)
    assert k["unknown_sentinel"] is False and k["reason"] == "road_following"
    assert u["unknown_sentinel"] is True and u["route"] == R.ROUTE_UNKNOWN
    # the shipped 2-tuple must be reproduced exactly
    for p in (_straight(), _stopped()):
        d = R.nav_command_v21_ex(p, 0)
        assert (d["nav"], d["valid"]) == R.nav_command_v21(p, 0)


def test_nav_input_v22_pairs_the_command_with_a_known_bit():
    assert R.nav_input_v22(_straight(), 0) == (R.NAV_FOLLOW, 1.0)
    assert R.nav_input_v22(_stopped(), 0) == (R.NAV_FOLLOW, 0.0)


# --------------------------------------------------------------------------- #
# v2.2 — scene-adaptive transience may never regress v2.1 coverage              #
# --------------------------------------------------------------------------- #
def _junction(T=200, v=6.0, turn_at=80, turn_len=25, ang=math.pi / 2):
    yaw = np.zeros(T)
    yaw[turn_at:turn_at + turn_len] = np.linspace(0, ang, turn_len)
    yaw[turn_at + turn_len:] = ang
    p = np.zeros((T, 4))
    p[:, 2] = yaw
    p[:, 3] = v
    p[:, 0] = np.cumsum(np.cos(yaw) * v * 0.1)
    p[:, 1] = np.cumsum(np.sin(yaw) * v * 0.1)
    return torch.from_numpy(p.astype(np.float32))


@pytest.mark.parametrize("p", [_straight(), _stopped(), _junction()])
def test_v22_carries_the_v21_decision_for_audit(p):
    for t in (0, 40, 120):
        if t >= p.shape[0] - 1:
            continue
        a = R.route_from_future_v21(p, t)
        b = R.route_from_future_v22(p, t)
        assert b["v21_route"] == a["route"] and b["v21_reason"] == a["reason"]
        assert b["changed_from_v21"] == (b["route"] != a["route"])


def test_v22_never_turns_a_decided_window_into_no_data():
    """v2.2 re-decides only the TRANSIENCE half. `no_future` / `no_arc` — the two
    'there is nothing to look at' reasons — must pass through untouched, so
    coverage cannot regress the way v2 -> v2.1 was written to prevent."""
    for p in (_stopped(), _straight(3)):
        r = R.route_from_future_v22(p, 0)
        assert r["reason"] == r["v21_reason"]
        assert r["changed_from_v21"] is False


def test_v22_transience_becomes_measurable_at_60m_not_150m():
    p = _straight(200, v=10.0)                 # 200 m of road over 20 s
    long_arc = R.route_from_future_v22(p, 0)
    assert long_arc["transience_measurable"] is True
    short = R.route_from_future_v22(p, 0, horizon_steps=70)   # ~70 m of road
    assert short["transience_measurable"] is True
    tiny = R.route_from_future_v22(p, 0, horizon_steps=30)    # ~30 m of road
    assert tiny["transience_measurable"] is False, (
        "below 3 x CONC_ARC_MIN_M the gate must stay OFF — claiming transience "
        "on a window shorter than three junction lengths is the v2.1 bug in a "
        "new costume")
    # and above 3 x CONC_ARC_M the sub-window saturates at v2.1's own constant
    assert R.route_from_future_v22(p, 0)["conc_arc_m_used"] == R.CONC_ARC_M


def test_route_target_v22_returns_target_and_mask_together():
    tgt, valid = R.route_target_v22(_stopped(), 0)
    assert tgt == R.ROUTE_UNKNOWN and valid is False
    tgt, valid = R.route_target_v22(_straight(), 0)
    assert tgt in (R.ROUTE_LEFT, R.ROUTE_STRAIGHT, R.ROUTE_RIGHT) and valid is True


# --------------------------------------------------------------------------- #
# the retrain premise: the factored label projects EXACTLY onto the 5-way one    #
# --------------------------------------------------------------------------- #
def test_factored_label_collapses_onto_the_shipped_5way_label():
    """The recommendation to retrain REF-C on (lat x lon) rests on this identity:
    the factored target is a REFINEMENT of the shipped one at the SAME thresholds,
    so switching cannot silently relabel anything. MEASURED over 11,504 real
    windows in `…/2026-08-03-refc-corpus-and-labels/results/labelqa_pai_*.json`
    (`collapse_self_check_mismatches: 0`); this is the synthetic pin."""
    g = torch.Generator().manual_seed(7)
    B, H = 512, R.LABEL_HORIZON
    pose_last = torch.randn(B, 4, generator=g)
    pose_last[:, 3] = pose_last[:, 3].abs() * 12.0
    fut = torch.randn(B, H, 4, generator=g)
    fut[:, :, 3] = fut[:, :, 3].abs() * 12.0
    lat, lon = TAC.window_factored_labels(pose_last, fut, H)
    assert torch.equal(TAC.collapse(lat, lon),
                       R.window_maneuver_labels(pose_last, fut, H))
    lat2, lon2 = TAC.window_factored_labels_v2(pose_last, fut, H)
    assert torch.equal(TAC.collapse(lat2, lon2),
                       R.window_maneuver_labels_v2(pose_last, fut, H))


def test_the_5way_target_is_provably_lossy():
    """A turn absorbs the longitudinal class: COLLAPSE_TABLE's turn rows are
    constant in `lon`. That is the whole defect, in one assertion."""
    for lat in (TAC.LAT_TURN_LEFT, TAC.LAT_TURN_RIGHT):
        row = TAC.COLLAPSE_TABLE[lat]
        assert len(set(row)) == 1, "a turn row must destroy the longitudinal axis"
    assert len(set(TAC.COLLAPSE_TABLE[TAC.LAT_LANE_KEEP])) == 3
