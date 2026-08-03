"""Guards for E-SEL-1D's two genuinely new pieces of arithmetic.

Both are the kind of thing that is silently wrong: a "derangement" with fixed
points still LOOKS shuffled, and a leave-one-out selector that peeks at its own
fold still LOOKS like a held-out number. So each test is written to FIRE on the
defect, not merely to pass on the happy path — a guard never observed to raise
is indistinguishable from a guard that cannot raise (C13's class).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from refc_sel_dump_deploy import sattolo                          # noqa: E402
from refc_s3_deploy_probe import (S3_THRESHOLDS, loeo_alpha,      # noqa: E402
                                  rho_on_reachable, zscore_rows)


# --------------------------------------------------------------------------- #
# C-ctxswap's precondition: EVERY window must see a DIFFERENT window's context  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [2, 5, 40, 881])
@pytest.mark.parametrize("seed", [0, 20260803, 7])
def test_sattolo_is_a_derangement(n, seed):
    p = sattolo(n, seed)
    assert sorted(p.tolist()) == list(range(n)), "not a permutation"
    assert (p != np.arange(n)).all(), (
        "FIXED POINT: that window would be scored against its OWN context, so "
        "the control leg is silently NOT swapped there")


def test_sattolo_is_deterministic_and_a_plain_shuffle_is_not_a_derangement():
    assert (sattolo(881, 20260803) == sattolo(881, 20260803)).all()
    # the defect the helper exists to avoid: `permutation` leaves fixed points.
    fixed = sum(int((np.random.default_rng(s).permutation(881)
                     == np.arange(881)).any()) for s in range(200))
    assert fixed > 0, ("a plain permutation is expected to leave a self-map "
                       "sometimes; if this never happens the comparison this "
                       "test documents is meaningless")


# --------------------------------------------------------------------------- #
# LOEO must not peek at the fold it scores                                     #
# --------------------------------------------------------------------------- #

def _toy():
    """2 episodes that prefer OPPOSITE gates, so leakage changes the answer."""
    de = torch.tensor([[1.0, 9.0],                 # ep0: candidate 0 is good
                       [1.0, 9.0],
                       [9.0, 1.0],                 # ep1: candidate 1 is good
                       [9.0, 1.0]])
    logits = torch.zeros(4, 2)
    z = torch.tensor([[1.0, -1.0], [1.0, -1.0],    # score prefers candidate 0
                      [1.0, -1.0], [1.0, -1.0]])
    return de, logits, z, np.array([0, 0, 1, 1])


def test_loeo_does_not_select_alpha_on_its_own_fold():
    de, logits, z, eid = _toy()
    alphas = [1.0, -1.0]
    _, ade, chosen = loeo_alpha(de, logits, z, eid, alphas)
    # ep0 is scored with the alpha that is best on ep1 (which prefers candidate
    # 1 => alpha = -1), and that alpha is BAD on ep0 => 9.0, not 1.0.
    assert chosen[0] == -1.0 and chosen[1] == 1.0
    assert ade[:2].tolist() == [9.0, 9.0], (
        "LEAK: ep0 got the alpha that is best ON ep0. A leave-one-out selector "
        "that peeks at its own fold reports a tuned number as a held-out one.")
    assert ade[2:].tolist() == [9.0, 9.0]


def test_loeo_matches_a_single_alpha_when_all_episodes_agree():
    de = torch.tensor([[1.0, 9.0]] * 6)
    logits = torch.zeros(6, 2)
    z = torch.tensor([[1.0, -1.0]] * 6)
    _, ade, chosen = loeo_alpha(de, logits, z, np.array([0, 0, 1, 1, 2, 2]),
                                [1.0, -1.0])
    assert set(chosen.values()) == {1.0}
    assert ade.tolist() == [1.0] * 6


def test_registered_alpha_grid_is_symmetric_and_contains_zero():
    a = S3_THRESHOLDS["alphas"]
    assert 0.0 in a, "alpha = 0 IS the shipped ranker; the sweep must contain it"
    assert all(-x in a for x in a), (
        "an asymmetric grid cannot express a NEGATIVE gate, and a negative gate "
        "is exactly what an ANTI-correlated consequence score needs")


# --------------------------------------------------------------------------- #
# the reachable-subset rho                                                     #
# --------------------------------------------------------------------------- #

def test_rho_on_reachable_drops_degenerate_windows_instead_of_inventing_a_value():
    score = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    neg = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    mask = np.array([[True, True, True, True], [True, True, False, False]])
    per, used = rho_on_reachable(score, neg, mask)
    assert used == 1 and np.isnan(per[1]), (
        "a window with < 3 survivors has no usable rank correlation; reporting "
        "one there would manufacture signal")
    assert per[0] == pytest.approx(1.0)


def test_zscore_rows_is_per_window_and_survives_a_constant_row():
    x = torch.tensor([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    z = zscore_rows(x)
    assert z[0].mean().abs() < 1e-6 and abs(float(z[0].std()) - 1.0) < 1e-5
    assert torch.isfinite(z[1]).all(), "a constant row must not produce inf/nan"
