"""Known-answer tests for the E1 paired helpers in ``taniteval/ci.py`` (refcv9 C7 / E1 rig):
``align_windows``, ``paired_stat_cluster_bootstrap``, ``e1_floor_read``. Literal expectations only -- no expected value is
computed by the code under test."""
import numpy as np
import pytest

from taniteval import ci


def _eids(n_ep, per):
    return ["ep%02d" % e for e in range(n_ep) for _ in range(per)]


def test_identical_arms_read_exactly_zero_and_not_separated():
    eid = _eids(10, 5)
    items = list(np.linspace(0.0, 1.0, 50))
    r = ci.paired_stat_cluster_bootstrap(items, list(items), eid, lambda xs: float(np.mean(xs)), n_boot=200, seed=0)
    assert r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0
    assert r["separated"] is False


def test_constant_shift_is_recovered_exactly_and_separated():
    eid = _eids(12, 4)
    base = [float(i % 7) for i in range(48)]
    a = [x + 0.25 for x in base]
    r = ci.paired_stat_cluster_bootstrap(a, base, eid, lambda xs: float(np.mean(xs)), n_boot=300, seed=0)
    # the mean of (a - b) is 0.25 on EVERY resample, so the interval is the point
    assert r["delta"] == 0.25 and r["lo"] == 0.25 and r["hi"] == 0.25 and r["separated"] is True
    assert r["a"] - r["b"] == pytest.approx(0.25)
    assert r["n_windows"] == 48 and r["n_episodes"] == 12 and r["n_boot"] == 300


def test_ratio_of_pooled_counts_known_answer():
    # per-window (tp, n_conf): pooled precision = sum tp / sum n_conf; arm A = 3/4 on every window, arm B = 1/2
    eid = _eids(8, 3)
    a = [(3, 4)] * 24
    b = [(1, 2)] * 24
    prec = lambda xs: sum(t for t, _ in xs) / sum(n for _, n in xs)
    r = ci.paired_stat_cluster_bootstrap(a, b, eid, prec, n_boot=100, seed=0)
    assert r["a"] == 0.75 and r["b"] == 0.5 and r["delta"] == 0.25 and r["lo"] == 0.25 and r["hi"] == 0.25


def test_mean_statistic_reproduces_the_scalar_paired_bootstrap_bounds():
    # same seed, same episode draws, same percentile rule: the generic helper with stat = mean must agree with the
    # per-window-scalar paired estimator the programme already uses
    rng = np.random.default_rng(3)
    eid = _eids(15, 4)
    a = rng.normal(0.3, 1.0, 60)
    b = rng.normal(0.0, 1.0, 60)
    g = ci.paired_stat_cluster_bootstrap(list(a), list(b), eid, lambda xs: float(np.mean(xs)), n_boot=500, seed=0)
    s = ci.paired_episode_cluster_bootstrap(a, b, eid, n_boot=500, seed=0)
    assert g["lo"] == pytest.approx(s["lo"], abs=1e-4) and g["hi"] == pytest.approx(s["hi"], abs=1e-4)
    assert g["separated"] == s["separated"]


def test_resampling_unit_is_the_episode_not_the_window():
    # DELIBERATE-REGRESSION ARM. Every window of an episode carries the SAME value, so episode resampling is the only
    # honest uncertainty; a window-level resampler (the defect this estimator replaced) is far too narrow.
    n_ep, per = 12, 20
    eid = _eids(n_ep, per)
    ep_val = np.linspace(-1.0, 1.0, n_ep)
    a = [float(ep_val[i // per]) + 0.05 for i in range(n_ep * per)]
    b = [float(ep_val[i // per]) for i in range(n_ep * per)]
    noise = np.random.default_rng(0).normal(0, 1.0, n_ep * per)
    a2 = [x + 0.5 * n for x, n in zip(a, noise)]          # a window-level noise on top
    mean = lambda xs: float(np.mean(xs))
    good = ci.paired_stat_cluster_bootstrap(a2, b, eid, mean, n_boot=400, seed=0)
    rng = np.random.default_rng(0)
    wl = np.array([mean([a2[j] for j in rng.integers(0, len(a2), len(a2))]) - mean([b[j] for j in rng.integers(0, len(b), len(b))])
                   for _ in range(400)])
    wl_w = np.percentile(wl, 97.5) - np.percentile(wl, 2.5)
    assert (good["hi"] - good["lo"]) > 0.0 and wl_w > 0.0
    # the window-level resampler also moves B independently (UNPAIRED), so it is wider here; the point of the arm is the
    # next line: the episode-level interval must NOT be the window-level one (they differ by > 20 %)
    assert abs((good["hi"] - good["lo"]) - wl_w) / wl_w > 0.2


def test_align_windows_pairs_on_keys_not_position_and_reports_the_drop():
    ka = [("c1", 0), ("c1", 8), ("c2", 0), ("c3", 0)]
    kb = [("c2", 0), ("c1", 8), ("c1", 0), ("c9", 0)]
    a, b, eid, rep = ci.align_windows(ka, ["a10", "a18", "a20", "a30"], kb, ["b20", "b18", "b10", "b90"])
    assert a == ["a10", "a18", "a20"] and b == ["b10", "b18", "b20"]
    assert eid == ["c1", "c1", "c2"]
    assert rep == {"n_common": 3, "n_only_a": 1, "n_only_b": 1}


def test_align_windows_refuses_duplicate_keys_and_disjoint_arms():
    with pytest.raises(ValueError):
        ci.align_windows([("c", 0), ("c", 0)], [1, 2], [("c", 0)], [3])
    with pytest.raises(ValueError):
        ci.align_windows([("c", 0)], [1], [("d", 0)], [2])


def test_e1_floor_read_literals():
    # floor |0.31 - 0.30| = 0.01; delta +0.10 is 10x the floor, separated, higher-is-better  => keep
    r = ci.e1_floor_read(0.10, True, 0.30, 0.31)
    assert r["floor_abs_c1_minus_c0"] == pytest.approx(0.01) and r["keep_on_this_metric"] is True
    assert r["margin_over_floor"] == pytest.approx(10.0) and r["floor_degenerate"] is False
    # delta +0.015 is separated but < 2x the 0.01 floor => NOT kept
    r = ci.e1_floor_read(0.015, True, 0.30, 0.31)
    assert r["separated_good_direction"] is True and r["exceeds_2x_floor"] is False and r["keep_on_this_metric"] is False
    # separated in the BAD direction (higher-is-better metric fell): never kept, flagged
    r = ci.e1_floor_read(-0.20, True, 0.30, 0.31)
    assert r["separated_bad_direction"] is True and r["keep_on_this_metric"] is False
    # lower-is-better metric (boxes per object): a fall IS the good direction
    r = ci.e1_floor_read(-0.50, True, 2.14, 2.20, higher_is_better=False)
    assert r["keep_on_this_metric"] is True
    # not separated => never kept, however large the point delta
    r = ci.e1_floor_read(0.50, False, 0.30, 0.31)
    assert r["keep_on_this_metric"] is False
    # a floor of exactly 0 is not a floor
    r = ci.e1_floor_read(0.05, True, 0.30, 0.30)
    assert r["floor_degenerate"] is True and r["margin_over_floor"] == float("inf") and r["keep_on_this_metric"] is True


def test_dict_valued_stat_and_single_arm_sibling():
    eid = _eids(6, 2)
    a = [(3, 4)] * 12
    b = [(1, 2)] * 12
    two = lambda xs: {"prec": sum(t for t, _ in xs) / sum(n for _, n in xs), "n": float(len(xs))}
    r = ci.paired_stat_cluster_bootstrap(a, b, eid, two, n_boot=50, seed=0)
    assert set(r) == {"prec", "n"} and r["prec"]["delta"] == 0.25 and r["n"]["delta"] == 0.0
    s = ci.stat_cluster_bootstrap(a, eid, two, n_boot=50, seed=0)
    assert s["prec"]["point"] == 0.75 and s["prec"]["lo"] == 0.75 and s["prec"]["hi"] == 0.75
    assert s["n"]["point"] == 12.0 and s["n"]["lo"] == 12.0   # every resample of 6 equal-size episodes keeps 12 windows
