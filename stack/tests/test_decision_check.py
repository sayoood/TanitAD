"""The instrument that would have prevented C129 must actually find C129's rows.

⛔ A guard that cannot fail is not a guard. These tests pin the two specific
lookups I skipped — the standing decision and the arm that already ran the
experiment — so the tool cannot silently stop surfacing them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "stack" / "scripts"))

import decision_check as DC  # noqa: E402


def test_the_record_files_the_tool_depends_on_exist():
    """⚠️ If one of these moves, the tool degrades to silence — which reads as
    'no decision exists'. That is the failure mode it was built to remove."""
    missing = [p.name for _, p, _ in DC.SOURCES if not p.exists()]
    assert not missing, f"record file(s) absent: {missing}"


def test_frozen_encoder_surfaces_D003_the_decision_C129_violated():
    hits = DC.search(["frozen", "encoder"], context=2, require_all=True)
    anchors = {a for _, _, a, _ in hits}
    assert "D-003" in anchors, (
        "D-003 says frozen-encoder is a COMPARISON ARM, NOT A HEDGE TO ADOPT. "
        f"The tool must surface it for 'frozen encoder'. Got: {sorted(anchors)}")


def test_frozen_encoder_also_surfaces_the_comparative_arm_decision():
    """D-033 specifies refa-v3 as a frozen-encoder MATRIX, explicitly not the
    naive configuration REF-A v1 actually was."""
    hits = DC.search(["frozen", "encoder"], context=2, require_all=True)
    assert "D-033" in {a for _, _, a, _ in hits}


def test_refa_surfaces_the_measured_arm():
    """The '(b) half' of the rule: the arm that already tested it."""
    hits = DC.search(["REF-A"], context=2, require_all=True)
    assert hits, "REF-A must be findable — it is the measured frozen arm"
    assert any("MODEL_REGISTRY" in f for _, f, _, _ in hits)


def test_retraction_log_is_searched_too():
    hits = DC.search(["overlapping_holdout_se"], context=1, require_all=True)
    assert any(lab == "RETRACTION" for lab, _, _, _ in hits), (
        "retraction classes must be searchable — they are how the programme "
        "learns, and only if they are read before asserting in a known class")


def test_empty_result_is_reported_as_word_miss_not_as_absence():
    """⚠️ 'nothing matched these WORDS' is not 'no decision exists'."""
    assert DC.search(["zzzz-no-such-token-zzzz"], 1, True) == []
    assert DC.main(["zzzz-no-such-token-zzzz"]) == 1     # non-zero, not silent


def test_all_vs_any_semantics_differ():
    a = DC.search(["frozen", "encoder"], 1, require_all=True)
    b = DC.search(["frozen", "encoder"], 1, require_all=False)
    assert len(b) >= len(a)
