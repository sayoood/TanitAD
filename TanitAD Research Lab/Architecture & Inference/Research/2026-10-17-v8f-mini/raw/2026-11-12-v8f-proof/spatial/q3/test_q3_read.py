"""Tests of the rule functions of q3_read.py (pure functions; no dumps needed)."""
import math
import q3_read as Q


def _c(D, lo, hi):
    return {"D": D, "A_lo": lo, "A_hi": hi}


def test_worse_needs_interval_and_spread():
    assert Q.cell_rule(_c(0.30, 0.10, 0.50), 0.20) == ("WORSE", True)
    assert Q.cell_rule(_c(0.15, 0.10, 0.50), 0.20) == ("INSIDE-SPREAD", False)
    assert Q.cell_rule(_c(0.30, -0.10, 0.50), 0.20) == ("OPEN", True)


def test_better_is_mirror():
    assert Q.cell_rule(_c(-0.30, -0.50, -0.10), 0.20) == ("BETTER", True)
    assert Q.cell_rule(_c(-0.15, -0.50, -0.10), 0.20) == ("INSIDE-SPREAD", False)


def test_no_spread_means_never_outside():
    v, out = Q.cell_rule(_c(0.30, 0.10, 0.50), float("nan"))
    assert v == "INSIDE-SPREAD" and out is False


def test_non_finite():
    assert Q.cell_rule(_c(float("nan"), 0.1, 0.2), 0.1)[0] == "NON-FINITE"


def test_names():
    cells = {"LONGITUDINAL:a": {"verdict": "OPEN"}, "LATERAL:b": {"verdict": "INSIDE-SPREAD"}}
    q = Q.q3_name(cells, True)
    assert q["name"] == "NO-COST" and q["final"]
    cells["LATERAL:c"] = {"verdict": "WORSE"}
    cells["TACTICAL:d"] = {"verdict": "BETTER"}
    q = Q.q3_name(cells, True)
    assert q["name"] == "COST-LATERAL GAIN-TACTICAL"
    q = Q.q3_name(cells, False)
    assert q["name"].startswith("PRELIMINARY") and not q["final"] and q["name_by_rule"] == "COST-LATERAL GAIN-TACTICAL"


def test_same_sign_rule():
    c = {"D": 0.30, "A_lo": 0.10, "A_hi": 0.50, "d_s": [0.7, -0.1]}
    assert Q.cell_rule(c, 0.20, same_sign=True) == ("MIXED-SIGN", True)
    assert Q.cell_rule(c, 0.20) == ("WORSE", True)
    c["d_s"] = [0.4, 0.2]
    assert Q.cell_rule(c, 0.20, same_sign=True) == ("WORSE", True)
