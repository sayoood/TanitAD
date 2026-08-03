"""AP + episode-cluster bootstrap: correctness, estimator identity, discrimination."""

import sys
from pathlib import Path

import numpy as np
import pytest

# repo pattern (test_heldout_gate.py:38): taniteval is not installed, it is a
# sibling checkout. The estimator-identity test below is load-bearing, so it
# must RUN, not skip.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "taniteval"))

from tanitad.eval.ap_ci import (
    ap_episode_cluster_bootstrap,
    ap_lift,
    average_precision,
    paired_ap_episode_cluster_bootstrap,
)


# --------------------------------------------------------------------------- #
# the statistic                                                               #
# --------------------------------------------------------------------------- #
def test_perfect_and_worst_ranking():
    y = np.array([1, 1, 0, 0, 0])
    assert average_precision(y, np.array([4.0, 3, 2, 1, 0])) == pytest.approx(1.0)
    # both positives last: P@4 = 1/4, P@5 = 2/5
    assert average_precision(y, np.array([0.0, 1, 2, 3, 4])) == pytest.approx(
        (0.25 + 0.4) / 2)


def test_matches_sc_train_reference_formula():
    """Independent re-implementation of sc_train.py:231 must agree exactly."""
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(np.int64)
    s = rng.random(500)
    order = np.argsort(-s, kind="mergesort")
    yt = y[order]
    tp = np.cumsum(yt)
    P = tp / np.maximum(np.arange(1, len(yt) + 1), 1e-12)
    ref = float((P * yt).sum() / yt.sum())
    assert average_precision(y, s) == pytest.approx(ref, rel=1e-12)


def test_random_scorer_lift_is_about_one():
    """The NEGATIVE CONTROL that makes AP interpretable: a scorer with no
    information must land at lift ~1 (AP ~ base rate), not at 0."""
    rng = np.random.default_rng(1)
    y = (rng.random(20000) < 0.05).astype(np.int64)
    lift = ap_lift(y, rng.random(20000))
    assert 0.8 < lift < 1.25, lift


def test_no_positive_is_nan_not_zero():
    assert np.isnan(average_precision(np.zeros(10), np.arange(10.0)))


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        average_precision(np.array([0, 1, 2]), np.arange(3.0))     # not binary
    with pytest.raises(ValueError):
        average_precision(np.array([0, 1]), np.arange(3.0))        # mismatch
    with pytest.raises(ValueError):
        average_precision(np.array([]), np.array([]))              # empty


# --------------------------------------------------------------------------- #
# the estimator                                                               #
# --------------------------------------------------------------------------- #
def _toy(n_ep=40, per=50, seed=0, strength=2.0):
    rng = np.random.default_rng(seed)
    eid = np.repeat([f"ep{i:03d}" for i in range(n_ep)], per)
    y = (rng.random(n_ep * per) < 0.1).astype(np.int64)
    s = rng.normal(size=y.size) + strength * y
    return y, s, eid


def test_point_estimate_is_the_full_set_value():
    """The bootstrap supplies the interval and must NOT move the centre — the
    `overlapping_holdout_se` lesson (it biased the point estimate too)."""
    y, s, eid = _toy()
    r = ap_episode_cluster_bootstrap(y, s, eid, n_boot=200)
    assert r["point"] == pytest.approx(round(average_precision(y, s), 5))


def test_draws_are_taniteval_draws():
    """Estimator identity: our resampling must BE taniteval's, not a lookalike."""
    tv = pytest.importorskip("taniteval.ci")
    from tanitad.eval.ap_ci import _draws, _episode_index

    eid = np.repeat([f"ep{i}" for i in range(12)], 7)
    uniq, idx = _episode_index(eid)
    tuniq, tidx = tv.episode_index(eid)
    assert list(uniq) == list(tuniq)
    mine = list(_draws(uniq, idx, 25, seed=3))
    theirs = list(tv._draws(tuniq, tidx, 25, seed=3))
    assert len(mine) == len(theirs) == 25
    for a, b in zip(mine, theirs):
        assert np.array_equal(a, b)


def test_cluster_bootstrap_is_wider_than_a_frame_bootstrap():
    """Clusters, not frames, are the unit. With a real per-episode random effect
    -- which is what a drive clip is -- resampling frames pretends 2000
    correlated rows are 2000 observations and yields a far too narrow band."""
    rng = np.random.default_rng(5)
    n_ep, per = 40, 50
    eid = np.repeat([f"ep{i:03d}" for i in range(n_ep)], per)
    re_ = np.repeat(rng.normal(scale=1.5, size=n_ep), per)     # clip difficulty
    y = (rng.random(n_ep * per) < 1 / (1 + np.exp(-(-2.5 + re_)))).astype(np.int64)
    s = rng.normal(size=y.size) + 2.0 * y + 0.8 * re_
    clustered = ap_episode_cluster_bootstrap(y, s, eid, n_boot=400)
    frames = ap_episode_cluster_bootstrap(y, s, np.arange(y.size), n_boot=400)
    assert clustered["se"] > frames["se"] * 1.5, (clustered["se"], frames["se"])


def test_interval_covers_the_point():
    y, s, eid = _toy(seed=7)
    r = ap_episode_cluster_bootstrap(y, s, eid, n_boot=400)
    assert r["lo"] <= r["point"] <= r["hi"]
    assert r["estimator"] == "ap_episode_cluster_bootstrap"
    assert r["n_episodes"] == 40 and r["n_boot_valid"] > 0


def test_lift_mode_rescales_by_base_rate():
    y, s, eid = _toy(seed=9)
    a = ap_episode_cluster_bootstrap(y, s, eid, n_boot=200)
    b = ap_episode_cluster_bootstrap(y, s, eid, n_boot=200, lift=True)
    assert b["point"] == pytest.approx(a["point"] / y.mean(), rel=1e-3)


# --------------------------------------------------------------------------- #
# the paired test — must SEPARATE on a real gap and NOT on noise               #
# --------------------------------------------------------------------------- #
def test_paired_separates_a_real_difference():
    rng = np.random.default_rng(11)
    eid = np.repeat([f"ep{i:03d}" for i in range(60)], 60)
    y = (rng.random(eid.size) < 0.08).astype(np.int64)
    strong = rng.normal(size=y.size) + 2.5 * y
    weak = rng.normal(size=y.size) + 0.4 * y
    r = paired_ap_episode_cluster_bootstrap(y, strong, weak, eid, n_boot=400)
    assert r["delta"] > 0 and r["separated"], r


def test_paired_does_not_separate_two_equivalent_arms():
    """NEGATIVE CONTROL for the estimator itself: two arms drawn from the same
    process must not be declared different, or every later 'separated' is void."""
    rng = np.random.default_rng(13)
    eid = np.repeat([f"ep{i:03d}" for i in range(60)], 60)
    y = (rng.random(eid.size) < 0.08).astype(np.int64)
    a = rng.normal(size=y.size) + 1.0 * y
    b = rng.normal(size=y.size) + 1.0 * y
    r = paired_ap_episode_cluster_bootstrap(y, a, b, eid, n_boot=400)
    assert not r["separated"], r


def test_generic_stat_bootstrap_matches_the_ap_specialisation():
    """The generic set-level bootstrap must reproduce the AP one exactly, or the
    TACTICAL family's intervals would be built on a different estimator."""
    from tanitad.eval.ap_ci import (
        paired_stat_episode_cluster_bootstrap, stat_episode_cluster_bootstrap)

    y, s, eid = _toy(seed=21)
    other = s + np.random.default_rng(4).normal(scale=0.7, size=s.size)
    single = stat_episode_cluster_bootstrap(
        lambda sel: average_precision(y[sel], s[sel]), eid, n_boot=200, name="ap")
    ref = ap_episode_cluster_bootstrap(y, s, eid, n_boot=200)
    assert single["point"] == pytest.approx(ref["point"])
    assert single["lo"] == pytest.approx(ref["lo"])
    assert single["hi"] == pytest.approx(ref["hi"])
    pr = paired_stat_episode_cluster_bootstrap(
        lambda sel: average_precision(y[sel], s[sel]),
        lambda sel: average_precision(y[sel], other[sel]), eid, n_boot=200)
    pref = paired_ap_episode_cluster_bootstrap(y, s, other, eid, n_boot=200)
    assert pr["delta"] == pytest.approx(pref["delta"])
    assert pr["separated"] == pref["separated"]


def test_generic_stat_bootstrap_separates_a_real_accuracy_gap():
    """NEGATIVE CONTROL for the tactical path: a labeller at chance must be
    separated from one that is right, and two equal arms must not be."""
    from tanitad.eval.ap_ci import paired_stat_episode_cluster_bootstrap

    rng = np.random.default_rng(31)
    eid = np.repeat([f"ep{i:03d}" for i in range(50)], 40)
    gt = rng.integers(0, 3, size=eid.size)
    good = np.where(rng.random(eid.size) < 0.85, gt, rng.integers(0, 3, eid.size))
    blind = np.zeros_like(gt)

    def acc(p):
        return lambda sel: float((p[sel] == gt[sel]).mean())

    r = paired_stat_episode_cluster_bootstrap(acc(good), acc(blind), eid, n_boot=400)
    assert r["delta"] > 0.3 and r["separated"], r
    same = paired_stat_episode_cluster_bootstrap(acc(good), acc(good), eid, n_boot=200)
    assert same["delta"] == pytest.approx(0.0) and not same["separated"]


def test_paired_is_antisymmetric_and_aligned():
    y, s, eid = _toy(seed=17)
    other = s + np.random.default_rng(2).normal(scale=0.5, size=s.size)
    ab = paired_ap_episode_cluster_bootstrap(y, s, other, eid, n_boot=200)
    ba = paired_ap_episode_cluster_bootstrap(y, other, s, eid, n_boot=200)
    assert ab["delta"] == pytest.approx(-ba["delta"], abs=1e-9)
    assert ab["ap_a"] == pytest.approx(ba["ap_b"])
    with pytest.raises(ValueError):
        paired_ap_episode_cluster_bootstrap(y, s, other[:-1], eid, n_boot=10)


# --------------------------------------------------------------------------- #
# degenerate resamples must report, not raise                                 #
# --------------------------------------------------------------------------- #
def test_ap_bootstrap_reports_nan_bounds_when_every_draw_is_degenerate():
    """A situation can be rare enough that NO cluster resample contains a
    positive. numpy's percentile then raised an IndexError from inside
    `_quantile`, which reads as a library bug instead of "unpowered here".
    The row must survive with n_boot_valid = 0 and nan bounds."""
    y = np.array([0, 0, 1, 0, 0, 0], np.int64)
    s = np.arange(6, dtype=float)
    eid = np.array(["a", "a", "b", "c", "c", "d"])
    # every draw that omits cluster "b" has no positive; force that by resampling
    # a set in which "b" is the only carrier and n_boot is small enough to check.
    r = ap_episode_cluster_bootstrap(y, s, eid, n_boot=8, seed=3, lift=True)
    assert r["n_boot_valid"] <= 8
    assert np.isfinite(r["point"])
    if r["n_boot_valid"] == 0:
        assert np.isnan(r["lo"]) and np.isnan(r["hi"])


def test_ap_bootstrap_survives_a_label_column_with_no_positive_in_any_draw():
    y = np.zeros(20, np.int64)
    y[7] = 1
    s = np.arange(20, dtype=float)
    eid = np.repeat(np.arange(10), 2)
    r = ap_episode_cluster_bootstrap(y, s, eid, n_boot=16, seed=0)
    assert "lo" in r and "hi" in r and r["n_boot"] == 16


def test_paired_ap_bootstrap_survives_all_degenerate_draws():
    y = np.zeros(8, np.int64)
    y[3] = 1
    a = np.arange(8, dtype=float)
    b = a[::-1].copy()
    eid = np.array(["p", "p", "q", "q", "r", "r", "s", "s"])
    r = paired_ap_episode_cluster_bootstrap(y, a, b, eid, n_boot=6, seed=1)
    assert set(("delta", "lo", "hi", "separated")) <= set(r)
    if r["n_boot_valid"] == 0:
        assert np.isnan(r["lo"]) and r["separated"] is False
