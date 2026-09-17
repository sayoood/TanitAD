"""Refusals 2 and 9 — both directions, and the mutation that earns each.

⛔ Every check here is tested clean AND with its defect reintroduced. A guard-
removal audit (`code/guard_removal_audit.py`, run against this module too) is what
proves the branches are load-bearing rather than decorative.
"""
from __future__ import annotations

import pytest

from tanitad.train.panel_refusals import (
    N_FLOOR, ControlDidNotReadItsKnownValue, TacticalReportRefused,
    refuse_control_not_reading_known_value,
    refuse_pooled_or_underpowered_tactical,
)


def _controls():
    return {"constant_only": {"expected": 0.0, "observed": 0.0},
            "all_zero_base_rate": {"expected": 0.016197554333,
                                   "observed": 0.016197554333}}


def _report():
    return {"per_token": {"LANE_KEEP": {"ap": 0.71, "n": 4210},
                          "TURN_L": {"ap": 0.64, "n": 980},
                          "GAP_TARGET": {"ap": 0.31, "n": 140}},
            "headline": ["LANE_KEEP", "TURN_L"]}


# ------------------------------------------------------------- refusal 2

def test_controls_that_read_their_known_values_EXACTLY_pass():
    assert refuse_control_not_reading_known_value(_controls())["all_exact"]


def test_MUT_a_control_off_by_the_recorded_0_881_percent_is_REFUSED():
    """MUTATION: the real historical defect — an all-zero control reading
    0.016340208 against a base rate of 0.016197554. ⛔ No tolerance."""
    c = _controls()
    c["all_zero_base_rate"]["observed"] = 0.016340208
    with pytest.raises(ControlDidNotReadItsKnownValue, match="EXACTLY"):
        refuse_control_not_reading_known_value(c)


def test_MUT_a_float_ulp_is_ALSO_refused_because_there_is_no_tolerance():
    """⛔ Deliberate: the moment a tolerance exists, the 0.881 % case has to argue
    about its size. Exactness is the property that makes the check unarguable."""
    c = _controls()
    c["constant_only"]["observed"] = 1e-18
    with pytest.raises(ControlDidNotReadItsKnownValue):
        refuse_control_not_reading_known_value(c)


def test_MUT_a_control_with_no_declared_expectation_is_REFUSED():
    """⛔ The one that matters: an undeclared expectation turns a control into a
    second measurement, and whatever it reads looks like the answer."""
    with pytest.raises(ControlDidNotReadItsKnownValue, match="not a control"):
        refuse_control_not_reading_known_value({"c": {"observed": 0.0}})
    with pytest.raises(ControlDidNotReadItsKnownValue, match="not a control"):
        refuse_control_not_reading_known_value({"c": {"expected": 0.0}})


def test_MUT_a_panel_with_NO_controls_is_REFUSED():
    with pytest.raises(ControlDidNotReadItsKnownValue, match="NO controls"):
        refuse_control_not_reading_known_value({})


# ------------------------------------------------------------- refusal 9

def test_a_per_class_report_with_a_well_supported_headline_passes():
    rep = refuse_pooled_or_underpowered_tactical(_report())
    assert rep["n_tokens"] == 3 and rep["below_floor"] == ["GAP_TARGET"]


def test_MUT_a_pooled_key_anywhere_is_REFUSED():
    """MUTATION: the pooled average — individually correct numbers supporting a
    false sentence."""
    r = _report()
    r["pooled_ap"] = 0.55
    with pytest.raises(TacticalReportRefused, match="reported PER CLASS"):
        refuse_pooled_or_underpowered_tactical(r)


def test_MUT_a_headline_built_on_a_token_under_the_floor_is_REFUSED():
    r = _report()
    r["headline"] = ["LANE_KEEP", "GAP_TARGET"]
    with pytest.raises(TacticalReportRefused, match="under the n = 200 floor"):
        refuse_pooled_or_underpowered_tactical(r)


def test_a_token_under_the_floor_may_be_REPORTED_just_not_quoted():
    """⭐ The distinction the refusal exists to preserve: reporting is required,
    leading is refused."""
    r = _report()
    assert "GAP_TARGET" in r["per_token"]
    assert refuse_pooled_or_underpowered_tactical(r)["n_below_floor"] == 1


def test_MUT_a_token_reporting_AP_without_n_is_REFUSED():
    """⛔ An AP with no support cannot be placed against the floor, and an
    unplaceable number gets quoted as if it had been placed."""
    r = _report()
    del r["per_token"]["TURN_L"]["n"]
    with pytest.raises(TacticalReportRefused, match="report no `n`"):
        refuse_pooled_or_underpowered_tactical(r)


def test_MUT_a_report_with_no_per_token_block_is_REFUSED():
    with pytest.raises(TacticalReportRefused, match="no `per_token` block"):
        refuse_pooled_or_underpowered_tactical({"headline": []})


def test_the_floor_is_the_documented_one():
    assert N_FLOOR == 200
