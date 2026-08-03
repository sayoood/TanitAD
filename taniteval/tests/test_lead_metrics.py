"""Synthetic-geometry tests for the distance-keeping metrics.

Every case here has a hand-computable answer, so a regression shows up as a wrong NUMBER rather
than as a plausible-looking one. That is deliberate: the programme's C63 retraction came from an
imported metric whose precondition was never measured, and the cheapest guard against repeating it
is a metric whose arithmetic is pinned.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# resolved via the taniteval package

from taniteval.lead_metrics import (
    MIN_SPEED_MPS, TTC_CAP_S, distance_keeping, path_headings, per_step_gap,
)


def test_path_headings_straight_ahead_is_zero():
    p = np.stack([np.arange(1, 5) * 5.0, np.zeros(4)], axis=1)
    assert np.allclose(path_headings(p), 0.0)


def test_path_headings_left_turn_is_positive_and_uses_t0_for_step0():
    p = np.array([[1.0, 0.0], [2.0, 1.0]])
    h = path_headings(p)
    assert h[0] == pytest.approx(0.0)               # from the window origin, straight
    assert h[1] == pytest.approx(np.pi / 4)         # +1 forward, +1 left


def test_path_headings_stationary_step_carries_the_last_heading():
    p = np.array([[1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert np.allclose(path_headings(p), 0.0)       # not an atan2(0,0) swing


def test_gap_is_rig_origin_to_lead_rear_face():
    # ego at x=10, lead centre at x=40, lead 5 m long -> rear face at 37.5 -> gap 27.5
    gap, lat, ok = per_step_gap(np.array([[10.0, 0.0]]), np.array([[40.0, 0.0]]), 5.0)
    assert gap[0] == pytest.approx(27.5)
    assert lat[0] == pytest.approx(0.0)
    assert ok[0]


def test_lead_outside_the_corridor_is_not_a_lead():
    gap, lat, ok = per_step_gap(np.array([[0.0, 0.0]]), np.array([[30.0, 6.0]]), 4.0)
    assert lat[0] == pytest.approx(6.0)
    assert not ok[0]                                # 6 m lateral: a neighbouring lane


def test_corridor_follows_the_predicted_heading_not_the_t0_axes():
    """An arm turning 45 deg keeps a lead that sits along ITS path, not along the t0 axis."""
    path = np.array([[10.0, 10.0], [20.0, 20.0]])       # heading +45 deg
    lead = np.array([[15.0, 15.0], [25.0, 25.0]])       # straight ahead of that path
    _, lat, ok = per_step_gap(path, lead, 4.0)
    assert np.allclose(lat, 0.0, atol=1e-9)
    assert ok.all()


def test_ttc_of_a_closing_lead_is_gap_over_closing_rate():
    # ego does 10 m/s, lead is static. dt=0.5 -> gap shrinks 5 m per step -> rate 10 m/s.
    k, dt = 4, 0.5
    path = np.stack([np.arange(1, k + 1) * 5.0, np.zeros(k)], axis=1)
    lead = np.stack([np.full(k, 60.0), np.zeros(k)], axis=1)
    r = distance_keeping(path[None], lead[None], [4.0], [10.0], dt)
    assert r["status"] == "OK"
    # tightest gap at the last step: 60 - 2 - 20 = 38
    assert r["headway_min_m"][0] == pytest.approx(38.0)
    assert r["time_gap_min_s"][0] == pytest.approx(3.8)      # 38 / 10
    # last closing step starts at gap 43 and closes at 10 m/s -> 4.3 s
    assert r["min_ttc_s"][0] == pytest.approx(4.3)
    assert r["n_closing"] == 1


def test_a_lead_pulling_away_is_censored_at_the_cap_not_reported_as_safe():
    k, dt = 4, 0.5
    path = np.stack([np.arange(1, k + 1) * 5.0, np.zeros(k)], axis=1)
    lead = np.stack([40.0 + np.arange(1, k + 1) * 10.0, np.zeros(k)], axis=1)
    r = distance_keeping(path[None], lead[None], [4.0], [10.0], dt)
    assert r["min_ttc_s"][0] == pytest.approx(TTC_CAP_S)
    assert r["n_closing"] == 0
    assert "censor" in r["censoring_note"].lower()


def test_time_gap_is_undefined_at_standstill_rather_than_enormous():
    path = np.zeros((2, 2))
    lead = np.stack([np.full(2, 20.0), np.zeros(2)], axis=1)
    r = distance_keeping(path[None], lead[None], [4.0], [MIN_SPEED_MPS / 2], 0.5)
    assert np.isfinite(r["headway_min_m"][0])
    assert np.isnan(r["time_gap_min_s"][0])
    assert r["n_time_gap"] == 0


def test_no_lead_anywhere_reports_not_applicable_with_n_and_a_reason():
    path = np.stack([np.arange(1, 5) * 5.0, np.zeros(4)], axis=1)
    lead = np.full((4, 2), np.nan)
    r = distance_keeping(path[None], lead[None], [np.nan], [10.0], 0.5)
    assert r["status"] == "NOT-APPLICABLE"
    assert r["n"] == 0 and r["n_windows"] == 1
    assert "free-flow" in r["reason"]                        # not silently dropped


def test_a_faster_arm_closes_more_which_is_what_makes_the_family_discriminate():
    """The whole point of the instrument: a too-fast arm must score a tighter gap and lower TTC."""
    k, dt = 4, 0.5
    lead = np.stack([np.full(k, 60.0), np.zeros(k)], axis=1)
    slow = np.stack([np.arange(1, k + 1) * 2.5, np.zeros(k)], axis=1)
    fast = np.stack([np.arange(1, k + 1) * 7.5, np.zeros(k)], axis=1)
    r = distance_keeping(np.stack([slow, fast]), np.stack([lead, lead]),
                         [4.0, 4.0], [5.0, 15.0], dt)
    assert r["headway_min_m"][1] < r["headway_min_m"][0]
    assert r["min_ttc_s"][1] < r["min_ttc_s"][0]


def test_shape_contracts_are_enforced():
    with pytest.raises(ValueError):
        distance_keeping(np.zeros((1, 4, 2)), np.zeros((1, 3, 2)), [4.0], [10.0], 0.5)
    with pytest.raises(ValueError):
        distance_keeping(np.zeros((1, 4, 2)), np.zeros((1, 4, 2)), [4.0], [10.0], 0.0)
    with pytest.raises(ValueError):
        distance_keeping(np.zeros((4, 2)), np.zeros((4, 2)), [4.0], [10.0], 0.5)


# --------------------------------------------------------------------------- #
# WIRING — the family must actually flip from UNAVAILABLE to OK                #
# --------------------------------------------------------------------------- #
def _win(k=4):
    p = np.stack([np.arange(1, k + 1) * 5.0, np.zeros(k)], axis=1)
    return {"pred": p[None].copy(), "gt": p[None].copy(),
            "wp_steps": list(range(1, k + 1)), "dt_s": 0.5}


def test_longitudinal_without_lead_is_a_work_item_not_a_pass():
    from taniteval.four_families import all_families
    fam = all_families(_win())
    dk = fam["longitudinal"]["distance_keeping"]
    assert dk["status"] == "UNAVAILABLE" and dk["n"] == 0
    assert "WORK ITEM" in dk["reason"]
    assert fam["_complete"] is False


def test_supplying_a_lead_track_turns_distance_keeping_on():
    from taniteval.four_families import all_families
    k = 4
    w = _win(k)
    w["lead"] = {"leads": np.stack([np.full(k, 60.0), np.zeros(k)], axis=1)[None],
                 "lead_lens": np.array([4.0]), "speeds": np.array([10.0])}
    dk = all_families(w)["longitudinal"]["distance_keeping"]
    assert dk["status"] == "OK" and dk["n"] == 1
    assert dk["mean_headway_min_m"] == pytest.approx(38.0)     # 60 - 2 - 20
    assert "D-LEAD-1" in dk["admitted_by"]
    assert set(dk["_per_window"]) == {"headway_min_m", "time_gap_min_s",
                                      "min_ttc_s", "n_steps_in_corridor"}
