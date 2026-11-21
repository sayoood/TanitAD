"""Refusals 2 and 9 — both directions, and the mutation that earns each.

⛔ Every check here is tested clean AND with its defect reintroduced. A guard-
removal audit (`code/guard_removal_audit.py`, run against this module too) is what
proves the branches are load-bearing rather than decorative.
"""
from __future__ import annotations

import pytest

from tanitad.train.panel_refusals import (
    N_FLOOR, NEGATIVES_POLICIES, POLICY_FACTS,
    ControlDidNotReadItsKnownValue, TacticalReportRefused,
    refuse_control_not_reading_known_value,
    refuse_pooled_or_underpowered_tactical,
    refuse_unnamed_negatives_policy,
    refuse_capped_weight_without_note, pos_weight_cap,
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


# -------------------------------------------------- refusal 12 (ERRATUM §E7)

def test_both_policies_pass_and_carry_their_own_counts():
    """⭐ The two policies are NOT interchangeable: 17/22 with 3 capped vs 21/22
    with 9. That difference is the reason the refusal exists."""
    a = refuse_unnamed_negatives_policy({"negatives_policy": "measured"})
    b = refuse_unnamed_negatives_policy(
        {"negatives_policy": "cot-absence-negative"})
    assert (a["trainable"], a["on_cap"]) == (17, 3)
    assert (b["trainable"], b["on_cap"]) == (21, 9)
    assert a["under_floor"] == b["under_floor"] == 10


def test_MUT_a_report_with_NO_policy_is_REFUSED_not_defaulted():
    """⛔ The real error: the pre-registration quoted the OPT-IN policy's numbers
    while the trainer defaults to the other one. Assuming a default here would
    reproduce exactly that."""
    with pytest.raises(TacticalReportRefused, match="does not name its negatives"):
        refuse_unnamed_negatives_policy({"per_token": {}})


def test_MUT_a_policy_name_the_trainer_does_not_know_is_REFUSED():
    with pytest.raises(TacticalReportRefused, match="not one the trainer knows"):
        refuse_unnamed_negatives_policy({"negatives_policy": "absence-negative"})


def test_MUT_naming_one_policy_while_reporting_the_OTHERS_counts_is_REFUSED():
    """⛔ Worse than naming none, because it reads as checked."""
    with pytest.raises(TacticalReportRefused, match="contradict it"):
        refuse_unnamed_negatives_policy(
            {"negatives_policy": "measured", "trainable": 21, "on_cap": 9})


def test_the_policy_vocabulary_matches_the_trainer_and_is_not_invented():
    """⛔ A checker with its own copy of a vocabulary is a check that shares the
    defect it checks for — so this pins the names against the trainer's own
    `--tac-goal-negatives` choices."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts"
           / "refc_v3_train.py").read_text(encoding="utf-8", errors="replace")
    # ⛔ Scoped to the SINGLE add_argument call. A non-greedy `.*?choices=`
    # spans into the NEXT flag's choices — my first version did exactly that and
    # pulled in `auto/on/off` from an unrelated argument.
    i = src.index('add_argument("--tac-goal-negatives"')
    j = src.index("choices=[", i)
    declared = set(re.findall(r'"([a-z-]+)"', src[j:src.index("]", j)]))
    assert set(NEGATIVES_POLICIES) == declared, (
        f"checker knows {sorted(NEGATIVES_POLICIES)}, trainer declares "
        f"{sorted(declared)}")
    # ⚠️ POLICY_FACTS is deliberately a SUBSET: `geometry` and `all` are accepted
    # by the trainer and their label census has NOT been measured here. Asserting
    # equality would force a fabricated census, which would then be checked
    # against itself and pass.
    assert set(POLICY_FACTS) < declared
    assert set(POLICY_FACTS) == {"measured", "cot-absence-negative"}


def test_a_policy_the_trainer_accepts_but_we_have_NOT_measured_is_flagged_UNVERIFIED():
    """⛔ 'accepted by the trainer' and 'its census is known' are different facts.
    Collapsing them is how an unchecked number reads as a checked one."""
    r = refuse_unnamed_negatives_policy({"negatives_policy": "geometry"})
    assert r["counts_verified"] is False and "UNVERIFIED" in r["why"]
    ok = refuse_unnamed_negatives_policy({"negatives_policy": "measured"})
    assert ok["counts_verified"] is True


# ------------------------------------------------- refusal 10 (ERRATUM §E4/§E7)

def _capped_report(cap):
    return {"per_token": {
        "LANE_KEEP": {"pos_weight": 3.2},
        "LANE_CHANGE_R": {"pos_weight": cap, "capped": True,
                          "cap_note": "the cap sets this weight, not the data "
                                      "(uncapped it implies 303.8)"},
    }}


def test_the_cap_is_READ_from_the_function_that_applies_it_not_a_literal():
    """⛔ The trainer reads it the same way and says why: a literal would be a
    second copy that goes stale. This module was already caught once tonight
    carrying its own copy of a vocabulary."""
    import inspect

    from tanitad.data import v7_labels
    assert pos_weight_cap() == float(
        inspect.signature(v7_labels.goal_pos_weight).parameters["cap"].default)


def test_a_capped_token_WITH_its_note_passes_and_uncapped_tokens_are_ignored():
    c = pos_weight_cap()
    r = refuse_capped_weight_without_note(_capped_report(c))
    assert r["n_on_cap"] == 1 and r["cap"] == c and r["n_tokens"] == 2


def test_MUT_a_capped_weight_quoted_BARE_is_REFUSED():
    """⭐ The refusal §E4 doubted could exist, because it was a rule about PROSE.
    It is enforceable once the obligation moves onto the PANEL: the report cannot
    be written without the sentence, because the sentence is a field."""
    c = pos_weight_cap()
    rep = _capped_report(c)
    del rep["per_token"]["LANE_CHANGE_R"]["capped"]
    del rep["per_token"]["LANE_CHANGE_R"]["cap_note"]
    with pytest.raises(TacticalReportRefused, match="quoted without it"):
        refuse_capped_weight_without_note(rep)


def test_MUT_an_EMPTY_cap_note_does_not_satisfy_it():
    """⛔ A field present but blank is the box-ticking failure — it passes a
    key-presence check while telling the reader nothing."""
    c = pos_weight_cap()
    rep = _capped_report(c)
    rep["per_token"]["LANE_CHANGE_R"]["cap_note"] = "   "
    with pytest.raises(TacticalReportRefused, match="quoted without it"):
        refuse_capped_weight_without_note(rep)


def test_MUT_claiming_capped_on_a_token_that_is_NOT_capped_is_REFUSED():
    """⛔ A false cap note is worse than none: it makes an uncapped weight look
    like one somebody checked."""
    c = pos_weight_cap()
    rep = _capped_report(c)
    rep["per_token"]["LANE_KEEP"].update(capped=True, cap_note="x")
    with pytest.raises(TacticalReportRefused, match="claim `capped` but are NOT"):
        refuse_capped_weight_without_note(rep)


def test_MUT_a_report_with_no_per_token_block_is_REFUSED_here_too():
    with pytest.raises(TacticalReportRefused, match="cannot be checked"):
        refuse_capped_weight_without_note({"negatives_policy": "measured"})
