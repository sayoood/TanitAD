"""S-S INVALIDATES S-T's CERTIFICATE — pin the gate that now refuses it.

THE DEFECT THIS PINS. ``STAGE_PRECONDITION`` is a FORWARD check: "did the stage
below pass?". The v6 ladder also runs backwards. S-S trains ``layer_str`` alone
(``v6.py:995``), and its output flows

    goal_head_str -> e_g_str -> goal_head_tac(cond=e_g_str) -> e_g_tac
                                                    (v6.py:1520-1528)

into the SELECTOR, whose only input is ``e_g_tac`` (``v6.py:655``, scoring at
``v6.py:619``). ``goal_head_tac`` and the selector are FROZEN in S-S — but their
input distribution moves. S-T certified ``sel_gap`` against the S-T-era
``e_g_tac``; that certificate does not survive S-S.

Before this gate an S-S run could report PASS on ``STRATEGIC_family`` alone and
S-J would launch on an uncertified selector. That is registry §1.14's
consumer-invalidation one level up, inside the ladder — the place it is easiest
to miss, because every forward check is green.

These tests are cheap and CPU-only on purpose: the failure they prevent costs
GPU-days and surfaces as "the model got worse", never as an error.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_v6_staged import (  # noqa: E402
    STAGE_GATE_SPEC, STAGE_INVALIDATES, STAGE_INVALIDATION_MECHANISM,
    STAGE_PRECONDITION, GatePreconditionError, assert_stage_precondition,
    stage_gate_dict, write_stage_gate)


def _probes(names, verdict=True):
    return {n: {"pass": verdict} for n in names}


# ---------------------------------------------------------------- the data --

def test_only_s_s_invalidates_and_it_invalidates_s_t():
    """The backward edge exists for exactly one stage, and it is S-S->S-T.

    S-T is deliberately EMPTY: it trains on a frozen S-W trunk whose inputs
    (pixels) do not move, so S-W's certificate still applies verbatim. S-J is
    empty because its own ``no_harm`` probe IS the revalidation.
    """
    assert STAGE_INVALIDATES["S-S"] == ("S-T",)
    assert STAGE_INVALIDATES["S-W"] == ()
    assert STAGE_INVALIDATES["S-T"] == ()
    assert STAGE_INVALIDATES["S-J"] == ()
    assert set(STAGE_INVALIDATES) == set(STAGE_PRECONDITION)


def test_mechanism_names_the_actual_seam():
    """A gate whose rationale is lost gets overridden by the next person.

    Pin the SEAM, not prose: if someone re-wires the goal path, this fails and
    the invalidation has to be re-derived rather than silently inherited.
    """
    m = STAGE_INVALIDATION_MECHANISM["S-S"]
    for token in ("layer_str", "goal_head_str", "e_g_str", "goal_head_tac",
                  "e_g_tac", "FROZEN"):
        assert token in m, f"mechanism text no longer names {token!r}"


# --------------------------------------------------------------- the gate --

def test_s_s_gate_requires_the_revalidations():
    req = STAGE_GATE_SPEC["S-S"]["required"]
    assert "sel_gap_revalidated" in req
    assert "TACTICAL_revalidated" in req
    # and each one names an owner, so a missing probe says WHAT was not
    # reachable (rule 2: absence at one location is not absence).
    for name in ("sel_gap_revalidated", "TACTICAL_revalidated"):
        assert STAGE_GATE_SPEC["S-S"]["owners"].get(name)
        assert STAGE_GATE_SPEC["S-S"]["criteria"].get(name)


def test_strategic_alone_is_inconclusive_not_pass():
    """THE ACTUAL DEFECT: this used to be a PASS."""
    g = stage_gate_dict("S-S", _probes(["STRATEGIC_family"]))
    assert g["pass"] is None
    assert g["verdict"] == "INCONCLUSIVE"
    assert set(g["missing_required"]) == {"sel_gap_revalidated",
                                          "TACTICAL_revalidated"}


def test_full_revalidation_passes():
    g = stage_gate_dict("S-S", _probes(STAGE_GATE_SPEC["S-S"]["required"]))
    assert g["pass"] is True
    assert g["verdict"] == "PASS"


def test_a_regressed_selector_FAILS_and_has_no_override():
    """A revalidation that ran and REGRESSED is a FAIL, not an inconclusive.

    X5 gives INCONCLUSIVE an override and FAIL none. A selector measured as
    broken under the new goal must therefore stop the ladder outright.
    """
    p = dict(_probes(STAGE_GATE_SPEC["S-S"]["required"]))
    p["sel_gap_revalidated"] = {"pass": False}
    g = stage_gate_dict("S-S", p)
    assert g["pass"] is False
    assert g["failed_required"] == ["sel_gap_revalidated"]


def test_gate_file_carries_the_mechanism():
    g = stage_gate_dict("S-S", _probes(STAGE_GATE_SPEC["S-S"]["required"]))
    assert g["revalidates"]["stages"] == ["S-T"]
    assert "e_g_tac" in g["revalidates"]["mechanism"]
    # stages with no backward edge carry no block at all — no empty key to
    # misread as "checked and clean".
    assert stage_gate_dict("S-W", _probes(["P1", "P3", "P6"]))["revalidates"] \
        is None


# ------------------------------------------------------- the ladder refuses --

def test_s_j_refuses_an_s_s_gate_that_skipped_the_revalidation(tmp_path):
    p = write_stage_gate(tmp_path, stage_gate_dict(
        "S-S", _probes(["STRATEGIC_family"])))
    with pytest.raises(GatePreconditionError):
        assert_stage_precondition("S-J", p)
    # the refusal names what was not run, so the override is a conscious act
    rep = assert_stage_precondition(
        "S-J", p, allow_inconclusive=True,
        off_reason="PI waived the tactical revalidation for a smoke run")
    assert rep["prev_verdict"] == "INCONCLUSIVE"
    body = json.loads(p.read_text())
    assert "sel_gap_revalidated" in body["missing_required"]


def test_s_j_launches_on_a_fully_revalidated_s_s(tmp_path):
    p = write_stage_gate(tmp_path, stage_gate_dict(
        "S-S", _probes(STAGE_GATE_SPEC["S-S"]["required"])))
    assert assert_stage_precondition("S-J", p)["prev_verdict"] == "PASS"
