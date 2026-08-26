"""Pins the measured null, and the reading it forces.

⛔ WHY THESE ASSERTIONS AND NOT OTHERS. On 2026-08-25/26 three published claims
were retracted because t-statistics between 2.0 and 3.1 were read as significant
in panels whose null reaches **3.49**. The load-bearing test below is
`test_t_of_2_is_NOT_significant`: it fails loudly if anyone reintroduces the
convention that killed those claims.
"""
from __future__ import annotations

import pytest

# ⚠️ THE PACKAGE IS `taniteval/taniteval/`, NOT `taniteval/`. The outer directory
# is a plain folder that `conftest.py` puts on sys.path; the importable package is
# the one INSIDE it. Placing this module in the outer directory made
# `taniteval.null_calibration` unimportable while `taniteval.ci` resolved fine —
# which reads exactly like the namespace-shadow trap and is really a wrong-directory
# error. `parents[2]` is the outer `taniteval/`, matching every sibling test.

from taniteval.null_calibration import (
    N_DRAWS, NULL_MAX, NULL_P95, NULL_P99, describe, p_value, verdict,
)


def test_the_null_is_deduplicated_to_104_independent_draws():
    """⚠️ It was first reported as 128: two runs shared `seed = 1000 + s` for
    their first six seeds, duplicating 24 draws. A pooled statistic must be
    deduplicated by SEED, not assumed independent because the invocations were."""
    assert N_DRAWS == 104


def test_the_null_reaches_three_point_four_nine():
    """The single number that matters — a provably-null input produced this."""
    assert NULL_MAX == pytest.approx(3.49, abs=1e-9)
    assert NULL_P99 == pytest.approx(2.93, abs=1e-9)
    assert 2.5 <= NULL_P95 <= 2.7


def test_t_of_2_is_NOT_significant():
    """⛔ THE LOAD-BEARING TEST. |t| = 2.0 is the convention that cost this
    programme three retracted claims. Against the measured null it is nowhere
    near significant, and this test exists so the convention cannot come back."""
    label, p = verdict(2.0)
    assert label == "INSIDE_NULL", (label, p)
    assert p > 0.05


def test_the_two_retracted_claims_read_as_inside_the_null():
    """The exact values that were retracted (C162 / E-DEC-56), pinned so the
    module reproduces the decision that was actually made."""
    assert verdict(2.56)[0] == "INSIDE_NULL"      # action -> delta-speed
    assert verdict(1.83)[0] == "INSIDE_NULL"      # E-DEC-48b n_free_cols marginal
    assert p_value(2.56) == pytest.approx(0.067, abs=0.005)


def test_the_surviving_claims_survive():
    """The identity control, the scene control and action -> delta-yaw."""
    for t in (23.74, 12.58, 5.09, 4.57):
        label, p = verdict(t)
        assert label == "SURVIVES", (t, label, p)


def test_a_value_just_under_the_null_max_is_MARGINAL_not_SURVIVES():
    """⭐ E-DEC-48b's headline marginal is t 3.50 against a null max of 3.49 — it
    clears by 0.01. A classifier that called that SURVIVES without qualification
    would be overselling it, so the band between alpha and the null max has its
    own name."""
    label, _ = verdict(3.40)
    assert label == "MARGINAL", label
    assert verdict(3.60)[0] == "SURVIVES"


def test_p_value_is_floored_at_one_over_n():
    """⚠️ With 104 draws the smallest resolvable p is ~0.0096. Reporting 0.000
    would claim precision the sample does not have."""
    assert p_value(1e6) == pytest.approx(1.0 / N_DRAWS)
    assert p_value(0.0) == pytest.approx(1.0)


def test_p_value_is_symmetric_and_monotone():
    assert p_value(-3.0) == p_value(3.0)
    assert p_value(1.0) >= p_value(2.0) >= p_value(3.0)


def test_describe_carries_the_constant_into_any_report_that_quotes_it():
    d = describe()
    assert "104" in d and "3.49" in d and "E-DEC-54/56" in d
