"""Unit tests for prh_seed on planted data. PRH_SRC_DIR (env) selects the directory holding the prh_seed.py under test (the mutation runner points it at a mutated copy)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PRH_SRC_DIR", os.path.dirname(os.path.abspath(__file__))))
import prh_seed as P  # noqa: E402

N, D = 1200, 40
FIT, SCORE = np.arange(0, N, 2), np.arange(1, N, 2)


def test_same_subspace_gives_one():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((N, D))
    T = rng.standard_normal((D, D))
    assert P.cca_M(A, A @ T + 3.0, FIT, SCORE) > 1 - 1e-4


def test_independent_subspace_high_dim_is_low():
    rng = np.random.default_rng(1)
    A, B = rng.standard_normal((N, D)), rng.standard_normal((N, D))
    assert abs(P.cca_M(A, B, FIT, SCORE)) < 0.1


def test_rotation_invariance():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((N, D))
    B = A @ rng.standard_normal((D, D)) + 2.0 * rng.standard_normal((N, D))
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    assert abs(P.cca_M(A, B, FIT, SCORE) - P.cca_M(A, B @ Q, FIT, SCORE)) < 1e-6


def test_bootstrap_point_matches_direct():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((N, D))
    B = A + rng.standard_normal((N, D))
    cs = (np.repeat(np.arange(30), N // 30))[SCORE]
    ids = np.unique(cs)
    counts = np.random.default_rng(0).multinomial(len(ids), np.ones(len(ids)) / len(ids), size=50).astype(float)
    pt, boot = P.pair_boot(A, B, FIT, SCORE, cs, len(ids), counts)
    assert abs(pt - P.cca_M(A, B, FIT, SCORE)) < 1e-9
    assert boot.shape == (50,)


def test_separation_rule():
    assert P.is_sep_pos(0.01, 0.05)
    assert not P.is_sep_pos(-0.01, 0.05)
    assert not P.is_sep_pos(-0.05, -0.01)
    assert P.is_sep_any(-0.05, -0.01) and not P.is_sep_any(-0.01, 0.01)


def test_outcome_logic_all_three():
    assert P.outcome(True, True) == "O1 PLATONIC-SLOW"
    assert P.outcome(True, False) == "O2 PLATONIC-OTHER"
    assert P.outcome(False, True) == "O3 NOT-PLATONIC"
    assert P.outcome(False, False) == "O3 NOT-PLATONIC"


def test_outcome_requires_all_four_p2():
    cis = [(0.01, 0.02), (0.01, 0.02), (0.01, 0.02), (-0.01, 0.02)]
    assert P.outcome(True, all(P.is_sep_pos(lo, hi) for lo, hi in cis)) == "O2 PLATONIC-OTHER"
    cis[3] = (0.005, 0.02)
    assert P.outcome(True, all(P.is_sep_pos(lo, hi) for lo, hi in cis)) == "O1 PLATONIC-SLOW"
    assert P.outcome(P.is_sep_pos(-0.01, 0.02), True) == "O3 NOT-PLATONIC"
