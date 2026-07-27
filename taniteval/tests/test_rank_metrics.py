"""The test that would have caught the H2 "chance" comparator.

THE DEFECT, restated so a future reader does not have to reconstruct it:
``h2c_eval.py:138`` built the chance comparator as ``np.zeros_like(y)`` and
``h2c_stats.average_precision`` ranked it with a STABLE argsort. A stable sort on
an all-tied score returns row order, and the H2 row order is
``[all left-camera rows, then all right-camera rows]`` — so the "constant score"
was really *"fire the left camera everywhere"*. MEASURED on the committed
``scores_heldout.npz``: **AP 0.005269 vs a base rate of 0.0030527 = 1.7259x
chance**, and the resulting sentence *"NO ARM IS ABOVE CHANCE"* gated a ~52 GB
corpus expansion.

Every assertion below is driven with input DESIGNED TO MAKE THE GUARD FAIL
(the ``e1c_selftest`` pattern): a guard only ever fed input it accepts has not
been tested.
"""
import numpy as np
import pytest

from taniteval.rank_metrics import (CHANCE_RTOL, ComparatorNotChance,
                                    assert_chance_comparator,
                                    average_precision, chance_ap,
                                    comparator_audit, random_ranking_ap)


def _front_loaded(n=1000, n_pos=50, frac_in_front=0.9):
    """Positives concentrated in the FIRST half of the row order.

    This is the H2 layout in miniature: rows are ``[all left, all right]`` and
    the left half carries most of the positives. On a constant score, row-order
    tie-breaking reads that layout as skill; collapsing ties cannot.
    """
    y = np.zeros(n)
    n_front = int(round(n_pos * frac_in_front))
    rng = np.random.default_rng(0)
    y[rng.choice(n // 2, n_front, replace=False)] = 1.0
    y[n // 2 + rng.choice(n // 2, n_pos - n_front, replace=False)] = 1.0
    return y


# ---------------------------------------------------------------- the defect --
def test_constant_score_under_row_order_ties_is_NOT_chance():
    """THE BUG, pinned. Input designed to make the naive comparator wrong."""
    y = _front_loaded()
    const = np.zeros_like(y)
    base = chance_ap(y)

    ap_row = average_precision(y, const, ties="row_order")
    assert ap_row > base * 1.3, (
        "the legacy row-order policy must still exhibit the defect, otherwise "
        "this test is not pinning anything")

    ap_ok = average_precision(y, const, ties="collapse")
    assert ap_ok == pytest.approx(base, abs=1e-12), (
        "a constant score MUST score exactly the base rate once ties collapse")


def test_the_bias_direction_is_toward_the_null():
    """The defect does not merely add noise — it always inflates.

    An inflated comparator understates every ``AP - chance`` delta, i.e. biases
    every above-chance test toward "not separated". For a CONTROL that is bias
    toward the desired verdict, which is why it survived review.
    """
    for frac in (0.7, 0.8, 0.9, 1.0):
        y = _front_loaded(frac_in_front=frac)
        const = np.zeros_like(y)
        assert average_precision(y, const, ties="row_order") > chance_ap(y)


def test_guard_REFUSES_the_row_order_comparator():
    """The guard must FIRE on the exact input that shipped."""
    y = _front_loaded()
    const = np.zeros_like(y)

    with pytest.raises(ComparatorNotChance) as e:
        assert_chance_comparator(y, const, name="chance", ties="row_order")
    msg = str(e.value)
    assert "TIE-HANDLING" in msg and "ROW ORDER" in msg, msg
    assert "x chance" in msg, "the guard must quote HOW FAR off it is"

    # ...and must ACCEPT the repaired one, or it is a guard that always fires.
    rec = assert_chance_comparator(y, const, name="chance", ties="collapse")
    assert rec["is_chance"] and rec["is_constant"] and rec["row_order_leaks"]


def test_guard_REFUSES_an_informative_comparator_that_is_not_constant():
    """A second failing mode: a comparator that is not tied at all but is sold
    as chance. The message must NOT blame tie handling in that case."""
    y = _front_loaded()
    informative = y + np.random.default_rng(1).normal(0, 0.1, y.size)
    with pytest.raises(ComparatorNotChance) as e:
        assert_chance_comparator(y, informative, name="chance")
    assert "carries information" in str(e.value)
    assert "TIE-HANDLING" not in str(e.value)


# ------------------------------------------------------- positive properties --
def test_collapsed_ties_match_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    for tied in (1, 3, 17, 1000):            # 1 => fully constant
        y = (rng.random(400) < 0.2).astype(float)
        s = rng.integers(0, tied, 400).astype(float)   # heavy ties on purpose
        assert average_precision(y, s) == pytest.approx(
            sk.average_precision_score(y, s), abs=1e-12), f"tied={tied}"


def test_a_true_random_ranking_scatters_around_the_base_rate():
    y = _front_loaded()
    r = random_ranking_ap(y, n_seeds=32)
    assert r["p2.5"] < chance_ap(y) < r["p97.5"], (
        "a genuine chance comparator must BRACKET the base rate; the constant "
        "comparator was deterministic and deterministically above it")


def test_perfect_and_inverted_rankings_bracket_the_metric():
    y = _front_loaded()
    assert average_precision(y, y) == pytest.approx(1.0)
    assert average_precision(y, -y) < chance_ap(y) * 1.05


def test_row_order_policy_is_reachable_and_reproduces_the_legacy_value():
    """``ties='row_order'`` exists ONLY to reproduce committed numbers, so it
    must stay bit-reachable — and it must differ from the default, or the fix
    silently did nothing."""
    y = _front_loaded()
    const = np.zeros_like(y)
    assert (average_precision(y, const, ties="row_order")
            != average_precision(y, const, ties="collapse"))
    with pytest.raises(ValueError):
        average_precision(y, const, ties="stable")


def test_audit_reports_both_policies_without_raising():
    y = _front_loaded()
    rec = comparator_audit(y, np.zeros_like(y), name="chance")
    assert rec["inflation_vs_chance"] == pytest.approx(1.0, abs=1e-9)
    assert rec["inflation_vs_chance_row_order"] > 1.3
    assert rec["is_chance"] and rec["row_order_leaks"]


def test_degenerate_inputs_do_not_silently_return_a_number():
    with pytest.raises(ValueError):
        average_precision(np.zeros(4), np.zeros(5))
    assert np.isnan(average_precision(np.zeros(10), np.arange(10.0)))
    assert abs(CHANCE_RTOL) < 1e-6
