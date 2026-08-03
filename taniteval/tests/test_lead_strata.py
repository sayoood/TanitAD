"""Tests for the SPEED-STRATIFIED distance-keeping read and its paired estimator.

⛔ Why stratification is tested as hard as the metric: a pooled headway/time-gap/TTC averages over
regimes that do not resemble each other on this corpus, and the crawling regime — where a time gap
is least meaningful — dominates the window count. A stratum that cannot be computed must say so
with its n and its reason, never vanish.
"""
from __future__ import annotations

import numpy as np
import pytest

from taniteval.lead_metrics import (MIN_STRATUM_N, SPEED_BANDS, assign_bands,
                                    band_label, distance_keeping_by_speed,
                                    paired_distance_keeping)
from taniteval.lead_source import LEAD, NO_LABEL, NO_LEAD


def _dk(headway, time_gap=None, ttc=None):
    h = np.asarray(headway, float)
    return {"headway_min_m": h,
            "time_gap_min_s": (h / 10.0 if time_gap is None else np.asarray(time_gap, float)),
            "min_ttc_s": (h / 2.0 if ttc is None else np.asarray(ttc, float))}


def test_band_labels_and_assignment_are_half_open():
    assert band_label(1.0, 3.0) == "1-3"
    assert band_label(15.0, float("inf")) == "15+"
    b = assign_bands([0.5, 1.0, 2.999, 3.0, 14.999, 15.0, 100.0, np.nan])
    assert list(b) == [0, 1, 1, 2, 4, 5, 5, -1]


def test_strata_carry_n_and_a_reason_when_empty():
    speeds = np.full(60, 12.0)                      # everything in the 10-15 band
    eid = np.repeat(np.arange(6).astype(object), 10)
    r = distance_keeping_by_speed(_dk(np.full(60, 20.0)), speeds, eid, n_boot=200)
    assert r["strata"]["10-15"]["status"] == "OK"
    assert r["strata"]["10-15"]["n_with_lead"] == 60
    for other in ("0-1", "1-3", "3-6", "6-10", "15+"):
        assert r["strata"][other]["status"] == "EMPTY"
        assert "reason" in r["strata"][other]
        assert r["strata"][other]["n_windows"] == 0


def test_a_thin_stratum_is_UNPOWERED_not_a_number():
    n = MIN_STRATUM_N - 1
    speeds = np.full(n, 12.0)
    eid = np.arange(n).astype(object)
    r = distance_keeping_by_speed(_dk(np.full(n, 20.0)), speeds, eid, n_boot=200)
    blk = r["strata"]["10-15"]
    assert blk["status"] == "UNPOWERED"
    assert blk["n_with_lead"] == n
    assert "headway_min_m" not in blk               # a number is not offered


def test_a_band_with_windows_but_no_lead_is_NOT_APPLICABLE_with_its_denominator():
    speeds = np.full(50, 12.0)
    eid = np.repeat(np.arange(5).astype(object), 10)
    r = distance_keeping_by_speed(_dk(np.full(50, np.nan)), speeds, eid, n_boot=200)
    blk = r["strata"]["10-15"]
    assert blk["status"] == "NOT-APPLICABLE"
    assert blk["n_windows"] == 50 and blk["n_with_lead"] == 0
    assert "free flow" in blk["reason"]


def test_window_states_are_reported_per_stratum_and_NO_LABEL_stays_separate():
    """⛔ The bias guard. NO_LABEL must never be folded into NO_LEAD: one is 'we could not see',
    the other is 'the road was clear'."""
    speeds = np.concatenate([np.full(40, 12.0), np.full(20, 2.0)])
    eid = np.repeat(np.arange(6).astype(object), 10)
    head = np.concatenate([np.full(40, 20.0), np.full(20, np.nan)])
    states = np.array([LEAD] * 40 + [NO_LEAD] * 10 + [NO_LABEL] * 10, dtype=object)
    r = distance_keeping_by_speed(_dk(head), speeds, eid, states=states, n_boot=200)
    assert r["window_states_total"] == {LEAD: 40, NO_LABEL: 10, NO_LEAD: 10}
    assert r["strata"]["10-15"]["window_states"] == {LEAD: 40}
    assert r["strata"]["1-3"]["window_states"] == {NO_LABEL: 10, NO_LEAD: 10}


def test_the_low_speed_band_carries_the_time_gap_caveat():
    speeds = np.full(60, 0.4)
    eid = np.repeat(np.arange(6).astype(object), 10)
    r = distance_keeping_by_speed(_dk(np.full(60, 5.0)), speeds, eid, n_boot=200)
    assert "time_gap_caveat" in r["strata"]["0-1"]


def test_point_estimate_is_the_full_set_mean_not_a_split_mean():
    """⛔ `overlapping_holdout_se` biases the POINT ESTIMATE (mean-of-split-means). The
    episode-cluster bootstrap must leave the mean exactly where the full set puts it."""
    rng = np.random.default_rng(0)
    head = rng.normal(20.0, 4.0, 300)
    speeds = np.full(300, 12.0)
    eid = np.repeat(np.arange(10).astype(object), 30)   # deliberately unequal-information clusters
    r = distance_keeping_by_speed(_dk(head), speeds, eid, n_boot=400)
    # ci.py rounds its reported mean to 4 dp; what matters is that it is the full-set value and
    # not a mean-of-split-means, which would move it far beyond rounding.
    assert r["strata"]["10-15"]["headway_min_m"]["mean"] == pytest.approx(float(head.mean()),
                                                                         abs=1e-4)


def test_it_accepts_the_four_families_shape_with_per_window_nested():
    speeds = np.full(60, 12.0)
    eid = np.repeat(np.arange(6).astype(object), 10)
    nested = {"status": "OK", "_per_window": _dk(np.full(60, 20.0))}
    r = distance_keeping_by_speed(nested, speeds, eid, n_boot=200)
    assert r["strata"]["10-15"]["status"] == "OK"


def test_it_refuses_a_result_without_per_window_arrays():
    with pytest.raises(KeyError):
        distance_keeping_by_speed({"status": "OK", "mean_headway_min_m": 3.0},
                                  np.zeros(3), np.arange(3).astype(object))


def test_it_refuses_mismatched_lengths():
    with pytest.raises(ValueError):
        distance_keeping_by_speed(_dk(np.zeros(10)), np.zeros(9),
                                  np.arange(10).astype(object))


# --------------------------------------------------------------------------- #
# paired estimator                                                             #
# --------------------------------------------------------------------------- #
def test_paired_delta_uses_only_jointly_finite_windows_and_reports_the_rest():
    a = np.array([10.0, 12.0, np.nan, 14.0, 9.0, 11.0])
    b = np.array([8.0, 9.0, 7.0, np.nan, 6.0, 8.0])
    eid = np.array(list("aabbcc"), dtype=object)
    r = paired_distance_keeping(_dk(a), _dk(b), eid, n_boot=300)
    m = r["metrics"]["headway_min_m"]
    assert m["status"] == "OK"
    assert m["n_used"] == 4 and m["n_a_only"] == 1 and m["n_b_only"] == 1
    assert m["delta"] > 0                       # A keeps more distance on the paired windows
    assert m["estimator"].startswith("paired_episode_cluster_bootstrap")


def test_paired_reports_NOT_APPLICABLE_rather_than_zero_when_nothing_pairs():
    a = np.array([10.0, np.nan])
    b = np.array([np.nan, 8.0])
    r = paired_distance_keeping(_dk(a), _dk(b), np.array(["x", "y"], dtype=object), n_boot=100)
    m = r["metrics"]["headway_min_m"]
    assert m["status"] == "NOT-APPLICABLE" and m["n_used"] == 0
    assert "delta" not in m


def test_paired_refuses_mismatched_eid_length():
    with pytest.raises(ValueError):
        paired_distance_keeping(_dk(np.zeros(4)), _dk(np.zeros(4)),
                                np.array(["a", "b"], dtype=object))


# --------------------------------------------------------------------------- #
# the four_families wiring                                                     #
# --------------------------------------------------------------------------- #
def _lead_for(n, k=4, gap=25.0):
    rel = np.arange(1, k + 1) * 0.5
    return {"leads": np.stack([np.column_stack([np.full(k, gap), np.zeros(k)])] * n),
            "lead_lens": np.full(n, 4.0), "speeds": np.full(n, 12.0),
            "state": np.array([LEAD] * n, dtype=object),
            "eid": np.repeat(np.arange(max(n // 10, 1)).astype(object),
                             int(np.ceil(n / max(n // 10, 1))))[:n],
            "n_boot": 200}, rel


def test_all_families_emits_by_speed_when_the_lead_block_carries_eid():
    from taniteval.four_families import all_families
    n = 60
    lead, rel = _lead_for(n)
    pred = np.stack([np.column_stack([10.0 * rel, np.zeros(4)])] * n)
    win = {"pred": pred, "gt": pred, "wp_steps": [5, 10, 15, 20], "lead": lead}
    dk = all_families(win)["longitudinal"]["distance_keeping"]
    assert dk["status"] == "OK"
    assert dk["by_speed"]["strata"]["10-15"]["status"] == "OK"
    assert dk["by_speed"]["window_states_total"] == {LEAD: n}


def test_all_families_says_why_when_the_lead_block_has_no_eid():
    """⛔ A pooled number is not silently substituted for the stratified one."""
    from taniteval.four_families import all_families
    n = 60
    lead, rel = _lead_for(n)
    lead.pop("eid")
    pred = np.stack([np.column_stack([10.0 * rel, np.zeros(4)])] * n)
    win = {"pred": pred, "gt": pred, "wp_steps": [5, 10, 15, 20], "lead": lead}
    bs = all_families(win)["longitudinal"]["distance_keeping"]["by_speed"]
    assert bs["status"] == "UNAVAILABLE" and bs["n"] == 0
    assert "episode-cluster bootstrap" in bs["reason"]


def test_bands_cover_the_real_line_without_gaps_or_overlap():
    lo = [b[0] for b in SPEED_BANDS]
    hi = [b[1] for b in SPEED_BANDS]
    assert lo[0] == 0.0 and hi[-1] == float("inf")
    assert lo[1:] == hi[:-1]                     # contiguous, half-open
