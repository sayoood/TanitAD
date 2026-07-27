"""comma2k19 yaw_rate ADMISSIBILITY — the decision that outranks the repair.

RETRACTION_LOG **C42**: `hold_heading_through_standstill` returned an
`observable` mask that no caller consumed, so a WHOLLY-STATIONARY clip (300
frames, zero observable frames, v_max 0.039 m/s) kept **84 physically impossible
yaw labels up to 15.28 rad/s** through the repair and pinned a published-grade
R2 at ~0.

⛔ A guard that cannot fire is worse than none (class **C13**). Every refusal
below is exercised **on the input that must trigger it**, and the two
directions that matter — the guard firing, and the guard NOT over-reaching on
clean data — are both asserted.
"""
from __future__ import annotations

import numpy as np
import pytest

from tanitad.data.comma2k19 import (DEFAULT_YAW_RATE_ADMISSIBILITY,
                                    HEADING_OBSERVABLE_V_MPS,
                                    KEEP_INADMISSIBLE_YAW_REASON,
                                    InadmissibleYawLabel,
                                    admissible_from_poses,
                                    assert_yaw_rate_admissible,
                                    heading_admissible_centers,
                                    hold_heading_through_standstill,
                                    yaw_rate_from_heading)

DT = 0.1


def _naive_rate(yaw, centers, dt=DT):
    """The pre-2026-07-27 derivation, independently reimplemented — calling the
    shipped path on both sides of a pin would let one bug satisfy it twice."""
    y = np.asarray(yaw, float)
    t = np.asarray(centers, np.int64)
    d = y[t + 1] - y[t - 1]
    return ((d + np.pi) % (2 * np.pi) - np.pi) / (2 * dt)


# --------------------------------------------------------------------------- #
# the rule, in one place                                                       #
# --------------------------------------------------------------------------- #
def test_admissible_centers_require_t_minus_1_t_and_t_plus_1():
    obs = np.array([True, True, False, True, True, True])
    got = heading_admissible_centers(obs, np.array([1, 2, 3, 4]))
    # t=1 needs 0,1,2 -> False;  t=2 needs 1,2,3 -> False
    # t=3 needs 2,3,4 -> False;  t=4 needs 3,4,5 -> True
    assert got.tolist() == [False, False, False, True]


def test_admissible_centers_refuse_a_center_without_both_neighbours():
    obs = np.ones(5, bool)
    with pytest.raises(ValueError, match="centred difference"):
        heading_admissible_centers(obs, np.array([0]))
    with pytest.raises(ValueError, match="centred difference"):
        heading_admissible_centers(obs, np.array([4]))


def test_admissibility_needs_no_new_storage_it_is_a_function_of_poses():
    """The argument for NOT widening the episode contract, made executable."""
    poses = np.zeros((6, 4))
    poses[:, 3] = [0.0, 0.4, 0.5, 9.0, 0.49999, 3.0]
    obs = admissible_from_poses(poses)
    assert obs.tolist() == [False, False, True, True, False, True]
    assert obs.tolist() == (poses[:, 3] >= HEADING_OBSERVABLE_V_MPS).tolist()


def test_admissible_from_poses_refuses_a_wrong_shaped_array():
    with pytest.raises(ValueError, match=r"\[T,>=4\]"):
        admissible_from_poses(np.zeros((5, 3)))


# --------------------------------------------------------------------------- #
# the policies                                                                 #
# --------------------------------------------------------------------------- #
def _seg_with_standstill():
    """6 standstill frames (GNSS noise, random direction) then 24 at 10 m/s."""
    rng = np.random.default_rng(0)
    v = np.concatenate([np.full(6, 0.01), np.full(24, 10.0)])
    yaw = np.concatenate([rng.uniform(-np.pi, np.pi, 6), np.zeros(24)])
    return yaw, v


def test_default_policy_is_nan_and_nan_cannot_be_quoted_by_accident():
    yaw, v = _seg_with_standstill()
    fixed, obs = hold_heading_through_standstill(yaw, v)
    centers = np.arange(1, len(yaw) - 1)
    assert DEFAULT_YAW_RATE_ADMISSIBILITY == "nan"
    rate, adm = yaw_rate_from_heading(fixed, obs, centers, dt=DT)

    assert rate.size == centers.size            # n is UNCHANGED — comparable
    assert np.isnan(rate[~adm]).all()
    assert not np.isnan(rate[adm]).any()
    # the whole point: a metric over an undefined label is now visibly undefined
    assert np.isnan(float(np.mean(np.abs(rate))))
    assert not np.isnan(float(np.mean(np.abs(rate[adm]))))


def test_drop_policy_returns_only_defined_windows_and_the_realignment_mask():
    yaw, v = _seg_with_standstill()
    fixed, obs = hold_heading_through_standstill(yaw, v)
    centers = np.arange(1, len(yaw) - 1)
    rate, adm = yaw_rate_from_heading(fixed, obs, centers, dt=DT,
                                      admissibility="drop")
    assert rate.size == int(adm.sum()) < centers.size
    assert not np.isnan(rate).any()
    nan_rate, _ = yaw_rate_from_heading(fixed, obs, centers, dt=DT)
    assert np.array_equal(rate, nan_rate[adm])          # the mask realigns


def test_keep_is_REFUSED_without_an_acknowledgement():
    yaw, v = _seg_with_standstill()
    fixed, obs = hold_heading_through_standstill(yaw, v)
    centers = np.arange(1, len(yaw) - 1)
    with pytest.raises(InadmissibleYawLabel, match="INADMISSIBLE"):
        yaw_rate_from_heading(fixed, obs, centers, dt=DT, admissibility="keep")


@pytest.mark.parametrize("flag,reason", [
    (True, ""),                       # flag, no reason
    (True, "   "),                    # flag, blank reason
    (False, KEEP_INADMISSIBLE_YAW_REASON),   # reason, no flag
])
def test_keep_is_REFUSED_on_every_half_acknowledgement(flag, reason):
    yaw, v = _seg_with_standstill()
    fixed, obs = hold_heading_through_standstill(yaw, v)
    centers = np.arange(1, len(yaw) - 1)
    with pytest.raises(InadmissibleYawLabel):
        yaw_rate_from_heading(fixed, obs, centers, dt=DT, admissibility="keep",
                              allow_inadmissible=flag, reason=reason)


def test_keep_WITH_both_reproduces_the_pre_2026_07_27_label_bit_identically():
    """The reproduction path must exist and must be exact, or people will route
    around the guard instead of using it."""
    yaw, v = _seg_with_standstill()
    fixed, obs = hold_heading_through_standstill(yaw, v)
    centers = np.arange(1, len(yaw) - 1)
    rate, adm = yaw_rate_from_heading(
        fixed, obs, centers, dt=DT, admissibility="keep",
        allow_inadmissible=True, reason=KEEP_INADMISSIBLE_YAW_REASON)
    assert np.array_equal(rate, _naive_rate(fixed, centers))
    assert not adm.all()                       # it really was the dirty set


def test_a_typo_in_the_policy_is_not_quietly_treated_as_keep():
    yaw, v = _seg_with_standstill()
    fixed, obs = hold_heading_through_standstill(yaw, v)
    with pytest.raises(ValueError, match="unknown admissibility"):
        yaw_rate_from_heading(fixed, obs, np.arange(1, 29), dt=DT,
                              admissibility="kep")


# --------------------------------------------------------------------------- #
# ⭐ the MEASURED case: a wholly-stationary clip                                #
# --------------------------------------------------------------------------- #
def test_wholly_stationary_clip_the_repair_is_a_noop_and_admissibility_is_not():
    """C42's mechanism, reproduced. 300 frames, zero observable, v_max 0.039."""
    rng = np.random.default_rng(7)
    T = 300
    yaw = rng.uniform(-np.pi, np.pi, T)          # arctan2 of ~zero velocity
    v = rng.uniform(0.0, 0.039, T)
    fixed, obs = hold_heading_through_standstill(yaw, v)

    assert v.max() < HEADING_OBSERVABLE_V_MPS
    assert obs.sum() == 0
    assert np.array_equal(fixed, yaw), "the repair is deliberately a NO-OP here"

    centers = np.arange(1, T - 1)
    kept, adm = yaw_rate_from_heading(
        fixed, obs, centers, dt=DT, admissibility="keep",
        allow_inadmissible=True, reason=KEEP_INADMISSIBLE_YAW_REASON)
    # the repair alone leaves physically impossible labels standing …
    assert (np.abs(kept) > 1.5).sum() > 0
    # … and admissibility removes EVERY window, because none is defined
    assert adm.sum() == 0
    dropped, _ = yaw_rate_from_heading(fixed, obs, centers, dt=DT,
                                       admissibility="drop")
    assert dropped.size == 0
    assert (np.abs(dropped) > 1.5).sum() == 0


def test_raising_v_min_cannot_substitute_for_admissibility():
    """A stationary clip is stationary at EVERY threshold — 84-85 survivors at
    1.0 / 2.0 / 4.0 m/s in the measured case."""
    rng = np.random.default_rng(7)
    T = 300
    yaw = rng.uniform(-np.pi, np.pi, T)
    v = rng.uniform(0.0, 0.039, T)
    centers = np.arange(1, T - 1)
    for v_min in (0.5, 1.0, 2.0, 4.0):
        fixed, obs = hold_heading_through_standstill(yaw, v, v_min=v_min)
        kept, _ = yaw_rate_from_heading(
            fixed, obs, centers, dt=DT, admissibility="keep",
            allow_inadmissible=True, reason=KEEP_INADMISSIBLE_YAW_REASON)
        assert (np.abs(kept) > 1.5).sum() > 0, f"v_min={v_min} should not help"


# --------------------------------------------------------------------------- #
# the guard must NOT over-reach                                                #
# --------------------------------------------------------------------------- #
def test_a_fully_observable_segment_loses_nothing():
    yaw = np.linspace(0.0, 1.0, 40)
    v = np.full(40, 12.0)
    fixed, obs = hold_heading_through_standstill(yaw, v)
    centers = np.arange(1, 39)
    rate, adm = yaw_rate_from_heading(fixed, obs, centers, dt=DT)
    assert adm.all()
    assert not np.isnan(rate).any()
    assert np.array_equal(rate, _naive_rate(fixed, centers))   # exact derivation
    dropped, _ = yaw_rate_from_heading(fixed, obs, centers, dt=DT,
                                       admissibility="drop")
    assert dropped.size == centers.size


def test_the_repair_is_a_float_ROUND_TRIP_on_observable_frames_not_a_noop():
    """⚠️ MEASURED here, and it corrects a small over-claim: the repair rebuilds
    every heading as `arctan2(sin, cos)`, so an OBSERVABLE frame is not
    bit-identical in general — it moves by up to **1 ULP (1.11e-16 rad)**, which
    reaches the yaw-rate label at **5.6e-16 rad/s**. HEADING_DEFAULT.md §2.1's
    'moving-part heading: bit-identical' held for its fixture (whose moving part
    had yaw exactly 0.0); it is not true of arbitrary headings. Physically
    irrelevant, but 'bit-identical' is a claim with a definite meaning."""
    rng = np.random.default_rng(3)
    yaw = rng.uniform(-np.pi, np.pi, 10_000)
    fixed, obs = hold_heading_through_standstill(yaw, np.full(10_000, 12.0))
    assert obs.all()
    d = np.abs(fixed - yaw)
    assert d.max() <= 2.3e-16, d.max()
    assert (d > 0).any(), "if this ever becomes exact, delete this test"
    assert np.allclose(fixed, yaw, rtol=0, atol=1e-15)


def test_assert_guard_is_a_noop_on_a_clean_label_and_fires_on_a_dirty_one():
    assert_yaw_rate_admissible(np.zeros(5), np.ones(5, bool))       # no raise
    with pytest.raises(InadmissibleYawLabel, match="worst such label"):
        assert_yaw_rate_admissible(np.array([0.0, 15.28, 0.0]),
                                   np.array([True, False, True]))
    # acknowledged -> allowed, and the wording is shipped
    assert KEEP_INADMISSIBLE_YAW_REASON.strip()
    assert_yaw_rate_admissible(np.array([0.0, 15.28, 0.0]),
                               np.array([True, False, True]),
                               allow_inadmissible=True,
                               reason=KEEP_INADMISSIBLE_YAW_REASON)


def test_the_refusal_message_carries_the_measured_mechanism():
    """Whoever hits this must not have to go find out why — the `zeros_naive`
    priced-trap construction."""
    with pytest.raises(InadmissibleYawLabel) as e:
        assert_yaw_rate_admissible(np.array([9.0]), np.array([False]))
    msg = str(e.value)
    for token in ("REPAIR DOES NOT FIX", "wholly-stationary", "v_min",
                  "'drop'", "allow_inadmissible=True"):
        assert token in msg, token


# --------------------------------------------------------------------------- #
# repair and admissibility are ORTHOGONAL — the headline lesson                #
# --------------------------------------------------------------------------- #
def test_repair_and_admissibility_are_different_things():
    yaw, v = _seg_with_standstill()
    centers = np.arange(1, len(yaw) - 1)
    fixed, obs = hold_heading_through_standstill(yaw, v)

    raw_rate = _naive_rate(yaw, centers)          # no repair, no admissibility
    rep_rate = _naive_rate(fixed, centers)        # repair only
    adm = heading_admissible_centers(obs, centers)

    assert not np.array_equal(raw_rate, rep_rate), "the repair changed labels"
    # …and the repair did not, by itself, decide admissibility
    assert (~adm).sum() > 0
    assert np.abs(raw_rate).max() > 1.5           # impossible before
    assert np.abs(rep_rate[adm]).max() <= 1.5     # possible where DEFINED
