"""pure-numpy checks of PROBE-N1D-FEAT's fold rule and feature builders (no sklearn / torch needed)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n1dwprobe_probe as P  # noqa: E402


def test_folds_grouped_by_episode_and_deterministic():
    rng = np.random.RandomState(3)
    eid = rng.randint(0, 40, 500)
    f = P.fold_ids(eid)
    for e in np.unique(eid):
        assert len(set(f[eid == e])) == 1                 # no episode in two folds
    assert set(f) == set(range(5)) and np.array_equal(f, P.fold_ids(eid))
    assert not np.array_equal(f, P.fold_ids(eid, seed=1))  # the seed is what fixes the assignment


def test_regression_arm_zeroes_c0_and_exit_columns_only():
    T = np.arange(1, 31, dtype=np.float32).reshape(2, 15)
    X = P.feature_matrix({"F_TEXT": T}, "F-TEXT-REG")
    assert (X[:, :13] == 0).all() and np.array_equal(X[:, 13:], T[:, 13:])
    assert (P.feature_matrix({"F_TEXT": T}, "F-TEXT-REG-ALL") == 0).all()


def test_plan_plus_text_concatenation():
    F = {"F_PLAN": np.ones((3, 4), np.float32), "F_TEXT": np.zeros((3, 15), np.float32)}
    assert P.feature_matrix(F, "F-PLAN+TEXT").shape == (3, 19)
