"""Unit tests (numpy only) for the S1b pure rules.  Each RED check is a mutation: the swapped / shifted rule MUST disagree with the pinned one."""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1b_rules as R  # noqa: E402


def test_cv_plan_is_the_devkit_straight_line():
    p = R.cv_plan(6.0)
    assert p.shape == (8, 3) and p.dtype == np.float32
    assert np.allclose(p[:, 0], 0.5 * 6.0 * np.arange(1, 9)) and not p[:, 1:].any()      # constant_velocity_agent.py:41-44
    assert np.allclose(R.cv_plan(0.0), 0.0)


def test_keep_rule_truth_table_and_boundaries():
    assert R.keep_rule(0.5, 0.9) and R.keep_rule(0.8, 0.8)                                   # "exceeds 0.8" / "less than 0.8" remove: the equalities stay
    assert not R.keep_rule(0.81, 0.9)        # simplistic: CV passes
    assert not R.keep_rule(0.5, 0.79)        # the human fails
    assert not R.keep_rule(0.81, 0.79)


def test_RED_swapped_directions_collapse_on_a_known_keep_population():
    rng = np.random.default_rng(0)
    cv, h = rng.uniform(0, 1, 5000), rng.uniform(0, 1, 5000)
    keep = np.array([R.keep_rule(a, b) for a, b in zip(cv, h)])
    swapped = (cv > 0.8) & (h < 0.8)
    assert keep.sum() > 0 and (keep & swapped).sum() == 0          # disjoint by construction: precision of the swapped rule against `keep` is exactly 0


def test_permutation_is_S1s_and_is_a_permutation():
    lg = "2021.06.07.12.01.13_veh-47_00730_00915"
    a = R.perm_for(20261020, lg, 356)
    b = np.random.default_rng([20261020, int(hashlib.sha256(lg.encode()).hexdigest()[:8], 16)]).permutation(356)
    assert (a == b).all() and sorted(a.tolist()) == list(range(356))
    assert not (R.perm_for(20261021, lg, 356) == a).all()                # RED: another seed gives another order
