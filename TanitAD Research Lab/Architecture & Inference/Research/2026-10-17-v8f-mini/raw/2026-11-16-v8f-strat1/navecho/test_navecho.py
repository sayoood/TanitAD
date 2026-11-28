"""test_navecho.py -- closed-form known values for navecho.py, and one source mutation per formula that must turn a check RED.
Run: python -m pytest test_navecho.py -q   (CPU, TMP on F:/TanitAD-scratch/strat1/navecho/tmp)"""
import os
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import navecho as N  # noqa: E402

SRC = open(os.path.join(HERE, "navecho.py"), encoding="utf-8").read()


def mutant(old, new):
    assert SRC.count(old) >= 1, f"mutation anchor not found: {old!r}"
    m = types.ModuleType("navecho_mut")
    m.__file__ = os.path.join(HERE, "navecho.py")
    exec(compile(SRC.replace(old, new, 1), "navecho_mut", "exec"), m.__dict__)
    return m


# ---- synthetic world: 8 episodes x 5 windows; the label is a deterministic function of the key except 4 noise windows
NAV = np.zeros((40, 5))
KEY = np.tile([0, 0, 1, 2, 3], 8)
for i, k in enumerate(KEY):
    if k < 3:
        NAV[i, k] = 1.0
NAV[:, 3] = np.tile([0.0, 0.2, 0.5, 0.8, 0.0], 8)
NAV[:, 4] = np.tile([0, 1, 1, 1, 0], 8)
EID = np.repeat(np.arange(8), 5)
Y = np.where(KEY == 1, 1, np.where(KEY == 2, 2, 0)).astype(np.int64)


def test_nav_key_known():
    assert N.nav_key(NAV).tolist() == KEY.tolist()
    assert N.informative(NAV).tolist() == np.isin(KEY, (1, 2)).tolist()


def test_nav_key_dist_known():
    nav = np.array([[1, 0, 0, 0.1, 1], [1, 0, 0, 0.35, 1], [0, 1, 0, 0.7, 1], [0, 0, 1, 2.0, 1], [0, 1, 0, 0.7, 0], [0, 0, 0, 0, 0]], float)
    w = len(N.DIST_EDGES_M) + 2
    assert N.nav_key_dist(nav).tolist() == [0 * w + 1, 0 * w + 2, 1 * w + 3, 2 * w + 4, 1 * w + 0, 3 * w + 0]


def test_fit_map_known():
    mp, g = N.fit_map(np.array([0, 0, 0, 1, 1, 2]), np.array([0, 0, 1, 1, 2, N.IGNORE]))
    assert mp == {0: 0, 1: 1} and g == 0                      # key 1 ties 1 vs 2 -> lowest; the ignored label is not a vote
    assert N.apply_map(mp, g, np.array([0, 1, 7])).tolist() == [0, 1, 0]


def test_macro_f1_known():
    y = np.array([0, 0, 1, 1, 2, 2]); p = np.array([0, 0, 1, 0, 2, 2])
    assert N.macro_f1(y, p) == pytest.approx((0.8 + 2 / 3 + 1.0) / 3)
    assert N.majority_f1(np.array([0, 0, 0, 1])) == pytest.approx((2 * 3 / (3 + 4) + 0.0) / 2)


def test_crossfit_never_fits_on_scoring_episode():
    A = N.episode_halves(EID, 0)
    assert A.sum() == 20 and set(EID[A]).isdisjoint(set(EID[~A]))
    out, pred, A2 = N.crossfit(KEY, Y, EID, 0)
    assert np.array_equal(A, A2)
    assert out["mean"] == pytest.approx(1.0) and out["pooled_oof_macro_f1"] == pytest.approx(1.0)      # the map is a function of the key
    # a label independent of the key: out-of-fold F1 must not be perfect
    rng = np.random.default_rng(3)
    out2, _, _ = N.crossfit(KEY, rng.integers(0, 3, 40), EID, 0)
    assert out2["mean"] < 0.9


def test_shuffled_reads_near_majority():
    rng = np.random.default_rng(1)
    eid = np.repeat(np.arange(40), 25)
    key = rng.integers(0, 3, len(eid))
    y = np.where(rng.random(len(eid)) < 0.9, 0, 1)
    s = N.shuffled_nav(key, y, eid, n=5)
    assert abs(s["mean"] - N.majority_f1(y)) < 0.1


def test_split_masks_and_counts():
    v9 = np.zeros((4, 5)); v9[1, 1] = 1; v9[0, 0] = 1; v9[2, 0] = 1; v9[3, 0] = 1
    tx = np.zeros((4, 5)); tx[2, 2] = 1; tx[0, 0] = 1; tx[1, 0] = 1; tx[3, 0] = 1
    m = N.split_masks(v9, tx)
    assert m["UNINF_BOTH"].tolist() == [True, False, False, True] and m["INF_ANY"].tolist() == [False, True, True, False]
    assert m["UNINF_V9"].tolist() == [True, False, True, True] and m["INF_V9"].tolist() == [False, True, False, False]
    c = N.split_counts(np.array([0, 1, 2, N.IGNORE]), np.array([0, 0, 1, 1]), m)
    assert c["UNINF_BOTH"]["n_labelled"] == 1 and c["INF_ANY"]["label_support"]["TURN_R"] == 1


def test_net_correct_known():
    y = np.array([0, 1, 2, 0, N.IGNORE]); a = np.array([0, 1, 2, 1, 0]); b = np.array([1, 1, 0, 0, 0])
    # windows: 0 a-only, 1 both, 2 a-only, 3 b-only, 4 ignored -> 2 - 1
    assert N.net_correct(y, a, b, np.ones(5, bool)) == 1
    assert N.net_correct(y, a, b, np.array([1, 1, 0, 0, 0], bool)) == 1


def test_reading_rules():
    nots = {"separated": False, "delta": -0.01}
    sep = {"separated": True, "delta": -0.1}
    sepup = {"separated": True, "delta": 0.1}
    hi, lo = {"s0": 0.95, "s1": 0.92}, {"s0": 0.95, "s1": 0.5}
    assert N.reading(hi, {"s0": nots, "s1": nots}) == "ECHO"
    assert N.reading(hi, {"s0": sep, "s1": sep}) == "SKILL-COST"
    assert N.reading(lo, {"s0": sep, "s1": sep}) == "SKILL-COST"
    assert N.reading(lo, {"s0": nots, "s1": nots}) == "MIXED"
    assert N.reading(hi, {"s0": sep, "s1": nots}) == "MIXED"
    assert N.reading({"s0": 0.90, "s1": 0.90}, {"s0": nots, "s1": nots}) == "ECHO"       # >= 90 %
    assert N.reading(lo, {"s0": sep, "s1": sepup}) == "MIXED"                             # opposite signs are not one reading


# ---- mutations: each formula changed once, the matching known-value check must fail
def _red(fn):
    with pytest.raises((AssertionError, ValueError)):
        fn()


def test_mut_no_row_key():
    m = mutant("np.where(nav[:, :3].sum(1) == 0, NO_ROW, k)", "np.where(nav[:, :3].sum(1) == 0, 0, k)")
    _red(lambda: np.testing.assert_array_equal(m.nav_key(NAV), KEY))


def test_mut_informative_set():
    m = mutant("np.isin(nav_key(nav), (TURN_L, TURN_R))", "np.isin(nav_key(nav), (TURN_L,))")
    _red(lambda: _must(m.informative(NAV).tolist() == np.isin(KEY, (1, 2)).tolist()))


def _must(c):
    assert c


def test_mut_dist_bin_side():
    m = mutant('side="right"', 'side="left"')
    nav = np.array([[1, 0, 0, 0.3, 1]], float)             # exactly on the 30 m edge
    assert N.nav_key_dist(nav)[0] != m.nav_key_dist(nav)[0]


def test_mut_dist_invalid_bin():
    m = mutant("nav[:, 4] == 1, 1 + np.searchsorted", "nav[:, 4] == 0, 1 + np.searchsorted")
    nav = np.array([[0, 1, 0, 0.7, 0]], float)
    assert N.nav_key_dist(nav)[0] != m.nav_key_dist(nav)[0]


def test_mut_fit_majority():
    m = mutant("int(np.bincount(y[m & (key == k)], minlength=K).argmax())", "int(np.bincount(y[m & (key == k)], minlength=K).argmin())")
    _red(lambda: _must(m.fit_map(np.array([0, 0, 0]), np.array([0, 0, 1]))[0] == {0: 0}))


def test_mut_fit_ignore_mask():
    m = mutant("m = (y != ignore) & (y >= 0) & (y < K)\n    if not m.any():\n        return {}, 0", "m = (y >= -999) & (y < K)\n    if not m.any():\n        return {}, 0")
    _red(lambda: _must(m.fit_map(np.array([0, 0, 0]), np.array([N.IGNORE, N.IGNORE, 1]))[1] == 1 and m.fit_map(np.array([0, 0, 0]), np.array([N.IGNORE, N.IGNORE, 1]))[0] == {0: 1}))


def test_mut_unseen_key():
    m = mutant("mp.get(int(k), glob)", "mp.get(int(k), 0)")
    _red(lambda: _must(m.apply_map({0: 1}, 2, np.array([5])).tolist() == [2]))


def test_mut_halves_overlap():
    m = mutant("rng.permutation(u)[: len(u) // 2]", "rng.permutation(u)[: len(u) // 2 + 1]")
    _red(lambda: _must(int(m.episode_halves(EID, 0).sum()) == 20))


def test_mut_crossfit_fits_on_scored_half():
    m = mutant('("fit_A_score_B", A, ~A), ("fit_B_score_A", ~A, A)', '("fit_A_score_B", A, A), ("fit_B_score_A", ~A, ~A)')
    rng = np.random.default_rng(3)
    yy = rng.integers(0, 3, 40)
    assert m.crossfit(KEY, yy, EID, 0)[0]["mean"] > N.crossfit(KEY, yy, EID, 0)[0]["mean"]    # fitting on the scored windows inflates the score


def test_mut_crossfit_mean():
    m = mutant('0.5 * (out["fit_A_score_B"]["macro_f1"] + out["fit_B_score_A"]["macro_f1"])', 'out["fit_A_score_B"]["macro_f1"]')
    rng = np.random.default_rng(5)
    yy = rng.integers(0, 3, 40)
    assert m.crossfit(KEY, yy, EID, 0)[0]["mean"] != N.crossfit(KEY, yy, EID, 0)[0]["mean"]


def test_mut_net_correct_sign():
    m = mutant("int((m & ra & ~rb).sum()) - int((m & ~ra & rb).sum())", "int((m & ~ra & rb).sum()) - int((m & ra & ~rb).sum())")
    y = np.array([0, 1, 2, 0, N.IGNORE]); a = np.array([0, 1, 2, 1, 0]); b = np.array([1, 1, 0, 0, 0])
    assert m.net_correct(y, a, b, np.ones(5, bool)) != N.net_correct(y, a, b, np.ones(5, bool))


def test_mut_masked():
    m = mutant("return np.where(np.asarray(mask), np.asarray(y), IGNORE)", "return np.where(np.asarray(mask), IGNORE, np.asarray(y))")
    _red(lambda: _must(m.masked(np.array([1, 2]), np.array([True, False])).tolist() == [1, N.IGNORE]))


def test_mut_split_both():
    m = mutant('"UNINF_BOTH": ~iv & ~it', '"UNINF_BOTH": ~iv | ~it')
    v9 = np.zeros((2, 5)); v9[0, 1] = 1; v9[1, 0] = 1
    tx = np.zeros((2, 5)); tx[0, 0] = 1; tx[1, 0] = 1
    assert m.split_masks(v9, tx)["UNINF_BOTH"].tolist() != N.split_masks(v9, tx)["UNINF_BOTH"].tolist()


def test_mut_echo_threshold():
    m = mutant("ECHO_RATIO = 0.90", "ECHO_RATIO = 0.60")
    nots = {"separated": False, "delta": -0.01}
    assert m.reading({"s0": 0.7, "s1": 0.7}, {"s0": nots, "s1": nots}) != N.reading({"s0": 0.7, "s1": 0.7}, {"s0": nots, "s1": nots})


def test_mut_echo_both_seeds():
    m = mutant("echo_nav = all(ratios[s] >= ECHO_RATIO", "echo_nav = any(ratios[s] >= ECHO_RATIO")
    nots = {"separated": False, "delta": -0.01}
    r = {"s0": 0.95, "s1": 0.5}
    assert m.reading(r, {"s0": nots, "s1": nots}) != N.reading(r, {"s0": nots, "s1": nots})


def test_mut_skill_needs_both():
    m = mutant("if all(sep.values()) and same_sign", "if any(sep.values()) and same_sign")
    nots, sep = {"separated": False, "delta": -0.01}, {"separated": True, "delta": -0.1}
    r = {"s0": 0.5, "s1": 0.5}
    assert m.reading(r, {"s0": sep, "s1": nots}) != N.reading(r, {"s0": sep, "s1": nots})


def test_mut_echo_requires_unseparated():
    m = mutant("if echo_nav and not any(sep.values())", "if echo_nav and not all(sep.values())")
    nots, sep = {"separated": False, "delta": -0.01}, {"separated": True, "delta": -0.1}
    r = {"s0": 0.95, "s1": 0.95}
    assert m.reading(r, {"s0": sep, "s1": nots}) != N.reading(r, {"s0": sep, "s1": nots})
