"""The hierarchy panel's defect gate must fire on a NESTED `_defects` list.

⛔ WHY THIS EXISTS.  `taniteval/tools/refcv3_arm.py` appends defects to
`rec["refcv3"]["_defects"]` while its own comment claims they are "collected at the top
level".  `run_hierarchy_panel.record_ok` read only `rec["_defects"]`, so a record carrying
a REAL defect returned ok=True and the panel published a family-shaped hole as a result.
That is how the STRATEGIC family stayed silently absent from every refcv3 eval.

⭐ MUTATION PROOF.  `test_nested_defect_is_caught` and `test_both_levels_are_collected`
FAIL on the unfixed tree (they were the whole defect); the remaining three pass on both
trees and exist so a regression cannot quietly delete the gate instead of fixing it.
A test suite that passes on both the fixed and the broken tree has proved nothing.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PANEL = os.path.normpath(
    os.path.join(_HERE, "..", "..", "taniteval", "tools", "run_hierarchy_panel.py")
)


def _load_panel():
    if not os.path.isfile(_PANEL):
        pytest.skip(f"panel driver not present at {_PANEL}")
    spec = importlib.util.spec_from_file_location("_hier_panel_under_test", _PANEL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write(tmp_path, rec) -> str:
    p = tmp_path / "rec.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    return str(p)


def test_nested_defect_is_caught(tmp_path):
    """⛔ FAILS on the unfixed tree: this is the defect itself."""
    panel = _load_panel()
    rec = {
        "refcv3": {
            "_defects": [
                {"where": "strategic.nav_compliance", "type": "TypeError",
                 "detail": "labels path was a dict"}
            ]
        }
    }
    ok, why = panel.record_ok(_write(tmp_path, rec))
    assert ok is False, "a record carrying a nested defect must NOT count as produced"
    assert "DEFECT" in why


def test_both_levels_are_collected(tmp_path):
    """⛔ FAILS on the unfixed tree: only the top-level one was ever counted."""
    panel = _load_panel()
    rec = {"_defects": [{"where": "top"}], "refcv3": {"_defects": [{"where": "nested"}]}}
    assert panel.collect_defects(rec) == [{"where": "top"}, {"where": "nested"}]
    ok, why = panel.record_ok(_write(tmp_path, rec))
    assert ok is False
    assert "2 DEFECT" in why


def test_top_level_defect_still_caught(tmp_path):
    """Regression control: the behaviour that already worked must survive the fix."""
    panel = _load_panel()
    rec = {"_defects": [{"where": "strategic", "type": "ValueError"}]}
    ok, _ = panel.record_ok(_write(tmp_path, rec))
    assert ok is False


def test_clean_record_passes(tmp_path):
    """⭐ The known-value control: a defect-free record must read ok=True.

    Without this, 'return False always' would satisfy every other test here.
    """
    panel = _load_panel()
    rec = {"refcv3": {"strategic": {"nav_compliance": {"status": "OK"}}}, "arm": "x"}
    ok, why = panel.record_ok(_write(tmp_path, rec))
    assert ok is True, why
    assert panel.collect_defects(rec) == []


def test_non_dict_values_do_not_raise(tmp_path):
    """The scan walks every value; scalars and lists must not break it."""
    panel = _load_panel()
    rec = {"n_windows": 40, "arms": ["cl", "ha0"], "dt_s": 0.2, "lead_block": None}
    assert panel.collect_defects(rec) == []
    ok, _ = panel.record_ok(_write(tmp_path, rec))
    assert ok is True
