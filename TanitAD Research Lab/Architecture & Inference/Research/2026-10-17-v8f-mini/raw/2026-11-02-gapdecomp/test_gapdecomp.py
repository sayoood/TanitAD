"""Known-answer tests for the pure layer of gapdecomp.py (numpy only; no model, no checkpoint, no GPU).

    CUDA_VISIBLE_DEVICES=-1 python -I -m pytest -q -p no:cacheprovider test_gapdecomp.py
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("gapdecomp", os.path.join(HERE, "gapdecomp.py"))
gd = importlib.util.module_from_spec(_spec)
sys.modules["gapdecomp"] = gd
_spec.loader.exec_module(gd)
CI = gd.load_ci()


def make_windows(rng, n_ep=30, per=20):
    """Synthetic eval windows: eid, v0, g with every class present (constant-speed tracks, late launches, early bursts)."""
    eid = np.repeat(np.arange(n_ep), per)
    n = len(eid)
    v0 = rng.choice([0.0, 0.2, 0.8, 3.0, 9.0], size=n)
    g = np.zeros((n, 30, 2))
    for i in range(n):
        kind = rng.integers(0, 6)
        if kind < 3:                                                    # constant GT speed in m/s
            sp = [0.0, 0.3, 4.0][kind]
            g[i, :, 0] = np.arange(1, 31) * 0.2 * sp
        elif kind == 3:                                                 # a late launch: stopped, then 5 m/s at the end
            g[i, :, 0] = np.concatenate([np.zeros(24), np.cumsum(np.full(6, 1.0))])
        elif kind == 4:                                                 # an early burst, then stopped again
            g[i, :, 0] = np.concatenate([np.cumsum(np.full(10, 1.0)), np.full(20, 10.0)])
        else:
            g[i, :, 0] = np.arange(1, 31) * 0.2 * 7.0
    return eid, v0, g


def test_partition_is_exact_and_nan_v0_raises():
    rng = np.random.default_rng(0)
    eid, v0, g = make_windows(rng)
    cls, masks = gd.classify(v0, g)
    assert sum(int(m.sum()) for m in masks.values()) == len(v0)
    assert all(int(m.sum()) > 0 for m in masks.values())
    assert np.array_equal(np.bincount(cls, minlength=4), [int(masks[c].sum()) for c in gd.CLASSES])
    v_bad = v0.copy()
    v_bad[3] = np.nan
    with pytest.raises(ValueError):
        gd.classify(v_bad, g)


def test_class_definitions_on_hand_built_windows():
    g = np.zeros((4, 30, 2))
    g[0, :, 0] = np.concatenate([np.zeros(24), np.cumsum(np.full(6, 1.0))])        # stopped then 5 m/s at the end
    g[1] = 0.0                                                                      # stays stopped
    g[2, :, 0] = np.concatenate([np.cumsum(np.full(10, 1.0)), np.full(20, 10.0)])   # a burst early, then stopped again
    g[3, :, 0] = np.arange(1, 31) * 1.0                                             # 5 m/s throughout
    v0 = np.array([0.0, 0.0, 0.0, 5.0])                                             # 5 m/s at v0 = 5: GT chord 5 < v0 + 1, so moving, not a launch
    cls, masks = gd.classify(v0, g)
    assert [gd.CLASSES[i] for i in cls] == ["L", "S", "O", "M"]


def test_threshold_edges():
    g = np.zeros((3, 30, 2))
    g[:, :, 0] = np.arange(1, 31)[None, :] * 0.2 * 0.4                              # 0.4 m/s throughout: top = 0.4 < v0 + 1
    cls, _ = gd.classify(np.array([0.49, 0.5, 6.0]), g)
    assert [gd.CLASSES[i] for i in cls] == ["S", "M", "M"]                           # v0 = 0.5 is moving; v0 = 6 is outside the launch range


def planted(rng, shift_classes, shift=1.0, nan_frac=0.0):
    eid, v0, g = make_windows(rng)
    cls, _ = gd.classify(v0, g)
    base = rng.normal(0, 1, len(eid))
    a = base.copy()
    for c in shift_classes:
        a[cls == gd.CLASSES.index(c)] += shift
    b = base.copy()
    if nan_frac:
        drop = rng.random(len(eid)) < nan_frac
        a[drop], b[drop] = np.nan, np.nan
    return a, b, eid, cls


def test_identity_sum_of_class_contributions_equals_total():
    rng = np.random.default_rng(1)
    a, b, eid, cls = planted(rng, ("L", "M"), 0.7, nan_frac=0.1)
    a[5] = np.nan                                                                    # a pair valid on one side only is dropped
    r = gd.decompose(a, b, eid, cls, "masked")
    ok = np.isfinite(a) & np.isfinite(b)
    assert abs(r["delta"] - (a[ok] - b[ok]).mean()) < 1e-12
    assert r["_class_sum_identity_abs_dev"] < 1e-12
    assert sum(r["classes"][c]["n"] for c in gd.CLASSES) == int(ok.sum()) == r["n_windows"]
    assert abs(sum(r["classes"][c]["share"] for c in gd.CLASSES) - 1.0) < 1e-12
    assert r["_share_sum_max_abs_dev_over_draws"] < 1e-9


def test_planted_effect_in_trigger_windows_reads_share_one():
    rng = np.random.default_rng(2)
    a, b, eid, cls = planted(rng, ("L", "S"), 1.0)
    r = gd.decompose(a, b, eid, cls, "masked")
    for c in ("O", "M"):
        assert abs(r["classes"][c]["delta"]) < 1e-12 and abs(r["classes"][c]["share"]) < 1e-12
    assert abs(r["trigger"]["share"] - 1.0) < 1e-12 and r["trigger"]["lo"] > 1.0 - 1e-9 and r["trigger"]["hi"] < 1.0 + 1e-9
    assert abs(r["trigger_zeroed"]["delta"]) < 1e-12 and r["trigger_zeroed"]["lo"] == 0.0 and r["trigger_zeroed"]["hi"] == 0.0
    assert abs(r["classes"]["L"]["delta"] - 1.0) < 1e-12 and abs(r["classes"]["S"]["delta"] - 1.0) < 1e-12
    assert gd.trigger_verdict(r) == "TRIGGER-BOUND"


def test_planted_effect_outside_trigger_windows_reads_share_zero():
    rng = np.random.default_rng(3)
    a, b, eid, cls = planted(rng, ("O", "M"), 1.0)
    r = gd.decompose(a, b, eid, cls, "masked")
    assert abs(r["trigger"]["share"]) < 1e-12 and r["trigger"]["hi"] < 1e-9
    assert abs(r["trigger_zeroed"]["delta"] - r["delta"]) < 1e-12
    assert gd.trigger_verdict(r) == "NOT TRIGGER-BOUND"


def test_planted_even_split_is_mixed():
    rng = np.random.default_rng(4)
    a, b, eid, cls = planted(rng, ("L", "M"), 1.0)
    r = gd.decompose(a, b, eid, cls, "masked")
    assert 0.0 < r["trigger"]["share"] < 1.0
    assert gd.trigger_verdict(r) == "MIXED"


def test_identical_arms_read_exactly_zero_in_every_class():
    rng = np.random.default_rng(5)
    a, _, eid, cls = planted(rng, (), 0.0, nan_frac=0.05)
    r = gd.decompose(a, a, eid, cls, "all")
    assert r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0
    assert all(r["classes"][c]["delta"] == 0.0 for c in gd.CLASSES if r["classes"][c]["n"] > 0)
    assert all(r["classes"][c]["n"] > 0 for c in gd.CLASSES)
    assert gd.trigger_verdict(r) == "NOT DEFINED"                                    # a zero difference has no share


def test_a_difference_whose_interval_contains_zero_has_no_share():
    rng = np.random.default_rng(6)
    eid, v0, g = make_windows(rng)
    cls, _ = gd.classify(v0, g)
    a = rng.normal(0, 1, len(eid))
    a = a - a.mean()                                                                  # the point difference is exactly 0
    r = gd.decompose(a, np.zeros_like(a), eid, cls, "masked")
    assert not r["separated"] and gd.trigger_verdict(r) == "NOT DEFINED"


@pytest.mark.parametrize("pool", ["masked", "all"])
def test_total_interval_equals_the_instrument_on_the_same_arrays(pool):
    rng = np.random.default_rng(7)
    a, b, eid, cls = planted(rng, ("L", "O"), 0.4, nan_frac=0.15)
    r = gd.decompose(a, b, eid, cls, pool)
    inst = gd.instrument_row(CI, a, b, eid, pool)
    dp = inst.get("display_dp", 4)
    for k in ("delta", "lo", "hi"):
        assert round(r[k], dp) == inst[k], (k, r[k], inst[k])
    ok = np.isfinite(a) & np.isfinite(b)
    assert r["n_windows"] == inst["n_windows"] if pool == "masked" else r["n_windows"] == int(ok.sum())
    if pool == "masked":
        assert r["n_episodes_pool"] == inst["n_episodes"]


def test_a_class_shuffle_moves_the_share():
    """The planted effect is attributed to the class that carries it: permuting the class labels must break the share-one reading."""
    rng = np.random.default_rng(8)
    a, b, eid, cls = planted(rng, ("L", "S"), 1.0)
    shuffled = np.random.default_rng(99).permutation(cls)
    assert abs(gd.decompose(a, b, eid, cls, "masked")["trigger"]["share"] - 1.0) < 1e-12
    assert gd.decompose(a, b, eid, shuffled, "masked")["trigger"]["share"] < 0.9


def test_sep_worse_directions():
    assert gd.sep_worse("yaw-rate MAE", 1.0, 0.5, 1.5)
    assert not gd.sep_worse("yaw-rate MAE", -1.0, -1.5, -0.5)
    assert not gd.sep_worse("yaw-rate MAE", 1.0, -0.5, 1.5)
    assert gd.sep_worse("TAC-LAT", -0.1, -0.2, -0.05)
    assert not gd.sep_worse("TAC-LAT", 0.1, 0.05, 0.2)


def test_arm_verdict_combinations():
    V = {0: "TRIGGER-BOUND", 1: "TRIGGER-BOUND"}
    assert gd.arm_verdict(V) == "TRIGGER-BOUND"
    assert gd.arm_verdict({0: "NOT TRIGGER-BOUND", 1: "NOT TRIGGER-BOUND"}) == "NOT TRIGGER-BOUND"
    assert gd.arm_verdict({0: "TRIGGER-BOUND", 1: "MIXED"}) == "MIXED"
    assert gd.arm_verdict({0: "TRIGGER-BOUND", 1: "NOT TRIGGER-BOUND"}) == "MIXED"
    assert gd.arm_verdict({0: "NOT DEFINED", 1: "MIXED"}).startswith("NOT DEFINED at s0")
