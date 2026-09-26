"""The STRATEGIC family's n/a is stated PER CORPUS — and NAVSIM is not PhysicalAI-AV.

⛔ WHY (2026-09-26, E3 of the navtest single-stage package). Every NavSim artifact's STRATEGIC ``reason`` —
the field the criteria check and every report quote — read *"PhysicalAI-AV carries NO map, NO lane graph …"*.
True of PhysicalAI-AV; wrong for NAVSIM, which HAS a map, a lane graph and a route. The NavSim fact sat in a
side key (``navsim_specific_reason``) that the criteria check never read, and the one test named for the
defect asserted only that the side key existed — green while the primary reason was wrong.
The class: TRUE BUT WRONG FOR THE READER — it tells the reader the family is blocked on a CORPUS fact, when on
NAVSIM it is blocked on eval engineering, which is the difference between "nothing to do" and a work item.

Every expected string below is a LITERAL, never read back from the code under test.
"""
from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TE = os.path.dirname(_HERE)
if _TE not in sys.path:
    sys.path.insert(0, _TE)

from taniteval import four_families as ff                   # noqa: E402
from taniteval.bench.navsim import summarize as SUM          # noqa: E402

PHYSICALAI_ONLY = ("PhysicalAI-AV carries NO map", "we do not include open maps data", "lat/lon")


def test_navsim_reason_states_the_navsim_fact():
    s = ff.strategic_unavailable(218, tier="T1", corpus="navsim")
    assert (s["status"], s["n"], s["n_windows_it_would_have_had"], s["corpus"], s["tier"]) == \
        ("UNAVAILABLE", 218, 218, "navsim", "T1")
    for must in ("DOES carry a map", "`driving_command` is an agent INPUT", "ROUTE-LEVEL ORACLE",
                 "TANITEVAL_AUDIT.md, seam 7"):
        assert must in s["reason"], must
    for wrong in PHYSICALAI_ONLY:
        assert wrong not in s["reason"], wrong


def test_navsim_work_item_is_eval_engineering_not_a_corpus_fact():
    s = ff.strategic_unavailable(5, corpus="navsim")
    assert "EVAL ENGINEERING" in s["_is_a_work_item"] and "CORPUS fact" not in s["_is_a_work_item"]
    assert "AlpaSim" not in s["_settled"]
    assert "strategic_optionset" in s["instrument_that_would_close_it"]
    assert "VLM" not in s["instrument_that_would_close_it"]


def test_the_default_is_still_physicalai_verbatim():
    """Every existing PhysicalAI caller passes no corpus — its statement must not move."""
    s = ff.strategic_unavailable(881)
    assert s["corpus"] == "physicalai"
    assert s["reason"].startswith("PhysicalAI-AV carries NO map, NO lane graph")
    assert "CORPUS fact" in s["_is_a_work_item"]
    assert "PH2" in s["instrument_that_would_close_it"]


def test_an_unknown_corpus_is_refused_not_given_another_corpus_reason():
    with pytest.raises(ValueError, match="never borrow another corpus"):
        ff.strategic_unavailable(3, corpus="nuscenes")


def test_the_corpus_reaches_the_family_through_strategic():
    s = ff.strategic({}, no_label={"n": 7, "tier": "T1", "corpus": "navsim"})
    assert s["corpus"] == "navsim" and "DOES carry a map" in s["reason"]
    legacy = ff.strategic({}, no_label={"n": 7, "tier": "T1"})       # a no_label without a corpus: PhysicalAI
    assert legacy["corpus"] == "physicalai"


def test_a_legacy_navsim_artifact_renders_the_navsim_fact_first():
    """Banked artifacts from before 2026-09-26 carry the PhysicalAI reason PLUS the side key. The summary must
    show the NavSim fact, not append it behind the other corpus's reason."""
    legacy_block = {"status": "UNAVAILABLE", "n": 16,
                    "reason": ff.STRATEGIC_UNAVAILABLE_REASON,
                    "navsim_specific_reason": "⚠️ The generic reason above is PhysicalAI-specific (no map in "
                                              "the corpus). On NavSim the fact is DIFFERENT and must not be "
                                              "conflated: the lane graph and route EXIST"}
    out = SUM.families_from_artifact({"four_families": {"strategic": legacy_block}}, scope="test")
    r = out["strategic"]["reason"]
    assert r.startswith("⚠️ The generic reason above is PhysicalAI-specific")
    assert "PhysicalAI-AV carries NO map" not in r
    assert "[legacy artifact" in r
