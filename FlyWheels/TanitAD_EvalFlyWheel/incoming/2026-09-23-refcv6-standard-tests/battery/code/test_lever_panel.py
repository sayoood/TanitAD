"""A4 lever panel: the cross-fit must score every window with the OTHER fold's weight, and its
identity control must be exact. Expected values are LITERALS; the leak test fails on a mutation that
scores a fold with its own weights.

run:  PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval" python -m pytest -q test_lever_panel.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lever_panel as LP  # noqa: E402

rng = np.random.default_rng(0)
N, K = 8, 4
G = rng.normal(size=(N, K, 2)).astype(np.float32)
EID = np.array([0, 0, 1, 1, 2, 2, 3, 3])          # fold = eid % 2 -> eps 0,2 fold 0; eps 1,3 fold 1


def test_fit_w_picks_the_exact_plan_and_rejects_a_bad_one():
    rows = np.arange(N)
    assert LP.fit_w(G.copy(), G + 3.0, G, rows) == [1.0, 1.0, 1.0, 1.0]
    assert LP.fit_w(G + 3.0, G.copy(), G, rows) == [0.0, 0.0, 0.0, 0.0]


def test_identity_grid_returns_the_base_bit_exactly():
    os_ = G + rng.normal(size=G.shape).astype(np.float32)
    base = G + rng.normal(size=G.shape).astype(np.float32)
    out, _ = LP.cross_fit_blend(os_, base, G, EID, grid=np.array([0.0]))
    assert np.array_equal(out, base)


def test_each_fold_is_scored_with_the_other_folds_weight():
    fold = EID % 2
    base = G + 1.0                                    # uniformly mediocre
    os_ = np.where((fold == 0)[:, None, None], G, G + 10.0).astype(np.float32)   # perfect on fold 0 only
    out, w = LP.cross_fit_blend(os_, base, G, EID)
    assert w["w_fit_on_fold0_scores_fold1"] == [1.0, 1.0, 1.0, 1.0]
    assert w["w_fit_on_fold1_scores_fold0"] == [0.0, 0.0, 0.0, 0.0]
    # honest cross-fit: fold 1 gets w = 1 (its own os is terrible), fold 0 gets w = 0 (its own os is perfect)
    assert np.array_equal(out[fold == 1], os_[fold == 1])
    assert np.array_equal(out[fold == 0], base[fold == 0])


def test_seed_average_of_identical_seeds_is_the_seed():
    o = G + 0.5
    assert np.array_equal(0.5 * (o + o), o)
