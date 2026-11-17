"""test_navecho2.py -- grouping, mutations, controls for navecho2.py.  Run: python -m pytest test_navecho2.py -q  (CPU; TMP on F:/TanitAD-scratch/strat1/navecho2/tmp)"""
import os
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import navecho2 as M  # noqa: E402

SRC = open(os.path.join(HERE, "navecho2.py"), encoding="utf-8").read()


def mutant(old, new):
    assert SRC.count(old) >= 1, f"anchor not found: {old!r}"
    m = types.ModuleType("navecho2_mut")
    m.__file__ = os.path.join(HERE, "navecho2.py")
    exec(compile(SRC.replace(old, new, 1), "navecho2_mut", "exec"), m.__dict__)
    return m


# 30 episodes x 6 windows; X carries the label exactly (a clean out-of-fold fit reads 1.0); a second world has X pure noise
rng = np.random.default_rng(1)
EID = np.repeat(np.arange(30), 6)
Y = rng.integers(0, 5, len(EID))
X_SIG = np.c_[M.label_onehot(Y), np.zeros(len(Y))]
X_NOISE = rng.normal(size=(len(Y), 5))


def test_folds_never_split_an_episode():
    f = M.episode_folds(EID)
    for e in np.unique(EID):
        assert len(np.unique(f[EID == e])) == 1
    assert sorted(np.unique(f).tolist()) == list(range(5))
    assert M.episode_folds(EID).tolist() == f.tolist()


def test_oof_reaches_one_on_signal_and_noise_is_near_chance():
    p, _ = M.crossfit_hgb(X_SIG, Y, EID)
    assert M.macro_f1(Y, p) == 1.0
    pn, _ = M.crossfit_hgb(X_NOISE, Y, EID)
    assert M.macro_f1(Y, pn) < 0.35


def test_label_as_input_reach_control_and_shuffle():
    p, _ = M.crossfit_hgb(M.label_onehot(Y), Y, EID)
    assert M.macro_f1(Y, p) == 1.0
    s = M.shuffled_control(X_SIG, Y, EID, n=3)
    assert s["mean"] < 0.4


def test_ignore_windows_excluded_from_fit_and_score():
    y2 = Y.copy()
    y2[::7] = M.IGNORE
    p, _ = M.crossfit_hgb(X_SIG, y2, EID)
    lab = y2 != M.IGNORE
    assert (p[lab] == y2[lab]).all()


def test_fit_never_sees_the_scored_fold():
    f = M.episode_folds(EID)
    y2 = Y.copy()
    y2[f == 0] = (Y[f == 0] + 1) % 5          # fold 0 labels disagree with the rule every other fold obeys
    p, _ = M.crossfit_hgb(X_SIG, y2, EID)
    assert (p[f == 0] == Y[f == 0]).all()     # predicted by the rule learned elsewhere, not by its own shifted labels


def test_mutation_fit_on_scored_fold_goes_red():
    m = mutant("te, tr = fold == f, (fold != f) & lab", "te, tr = fold == f, (fold == f) & lab")
    f = M.episode_folds(EID)
    y2 = Y.copy()
    y2[f == 0] = (Y[f == 0] + 1) % 5
    p, _ = m.crossfit_hgb(X_SIG, y2, EID)
    assert not (p[f == 0] == Y[f == 0]).all()   # the leaking fit reads its own labels: test_fit_never_sees_the_scored_fold would fail


def test_mutation_window_level_folds_goes_red():
    m = mutant("return np.asarray([fold_of[int(e)] for e in eid], np.int64)", "return np.arange(len(eid)) % k")
    f = m.episode_folds(EID)
    assert any(len(np.unique(f[EID == e])) > 1 for e in np.unique(EID))
    pn, _ = m.crossfit_hgb(X_NOISE, Y, EID)     # same-episode windows now straddle folds; the grouping test above would fail
    assert pn.shape == Y.shape


def test_reading_rules():
    r = {"s0": 0.95, "s1": 0.91}
    assert M.reading(r, {"s0": 0.0, "s1": 0.0}, {"s0": 0.1, "s1": 0.1}) == "ECHO-CONT"
    r2 = {"s0": 0.95, "s1": 0.80}
    assert M.reading(r2, {"s0": 0.07, "s1": 0.07}, {"s0": 0.1, "s1": 0.1}) == "FEED-INFO"
    assert M.reading(r2, {"s0": 0.06, "s1": 0.07}, {"s0": 0.1, "s1": 0.1}) == "MODEL"
    assert M.reading(r2, {"s0": 0.2, "s1": 0.2}, {"s0": -0.1, "s1": 0.1}) == "MODEL"


def test_mutations_in_reading_go_red():
    m = mutant("ratio[s] >= ECHO_RATIO", "ratio[s] > 2.0")
    assert m.reading({"s0": 0.95, "s1": 0.95}, {"s0": 0, "s1": 0}, {"s0": 0.1, "s1": 0.1}) != "ECHO-CONT"
    m = mutant('all(ratio[s] >= ECHO_RATIO for s in ("s0", "s1"))', 'any(ratio[s] >= ECHO_RATIO for s in ("s0", "s1"))')
    assert m.reading({"s0": 0.95, "s1": 0.5}, {"s0": 0, "s1": 0}, {"s0": 0.1, "s1": 0.1}) == "ECHO-CONT"
    assert M.reading({"s0": 0.95, "s1": 0.5}, {"s0": 0, "s1": 0}, {"s0": 0.1, "s1": 0.1}) != "ECHO-CONT"
    m = mutant("diff[s] >= FEED_FRAC * gap[s]", "diff[s] >= gap[s]")
    assert m.reading({"s0": 0.5, "s1": 0.5}, {"s0": 0.07, "s1": 0.07}, {"s0": 0.1, "s1": 0.1}) != "FEED-INFO"


def test_hyperparameters_pinned():
    assert M.HGB_PARAMS == dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, class_weight="balanced", random_state=0, early_stopping=False)
