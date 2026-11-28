"""``code/d_only_register_lines.py`` — the sub-classes that decide what is handed over, on LITERAL lines.

The expensive mistakes this guards against, each MEASURED on 2026-09-26 while building the handover:
* re-appending a line the tip since CORRECTED (the HISTORICAL class — git-backed, exercised by the live run);
* handing over a clip-handle REDACTION of an existing row as if it were new content;
* reading a DIFFERENT item written from the same template as an EDIT (PI_DECISION_QUEUE.md: a REFe item's
  heading was paired with a refcv7 item's by a 0.5 similarity cut).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "code" / "d_only_register_lines.py"


@pytest.fixture(scope="module")
def D():
    spec = importlib.util.spec_from_file_location("d_only_register_lines", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TIP = [
    "## 🔴 NEW ITEM (2026-09-26 ~14:30 Berlin) — refcv7 output head: should the planner predict a residual?",
    "| C135 | the turn guard on clip `cafef00d` is a false positive | **RETRACTED** | argued from the frames |",
    "| **D-REFE-SEL-1** | ⭐ **SELECTION-BOUND at the pre-registered rule.** MEASURED on 200 tokens. |",
    "| **D-X-1** | the gradient on every `layer.agent_gate`. ⛛ **PARITY, BITWISE, INCLUDING GRADIENTS** holds |",
    "an unrelated line that stays on the tip exactly as written",
]


def test_a_different_item_from_the_same_template_is_NEW_not_an_edit(D):
    line = "## ✅ ITEM (2026-09-26 ~13:25 Berlin) — REFe finishes ~1 h past the 7-day line: DECIDED, extension accepted"
    assert D.subclass(line, TIP)["sub"] == "NEW"


def test_a_clip_handle_rewritten_to_sha12_is_a_REDACTION(D):
    line = "| C135 | the turn guard on clip `sha12:000000000001` is a false positive | **RETRACTED** | argued from the frames |"
    assert D.subclass(line, TIP)["sub"] == "SHA12_REDACTION"


def test_a_row_that_keeps_every_tip_character_and_adds_is_a_DESCENDANT(D):
    line = TIP[2][:-1] + "**UPDATE — epoch 12 snapshot, MEASURED.** |"
    r = D.subclass(line, TIP)
    assert r["sub"] == "DESCENDANT" and r["tip_chars"] == len(TIP[2]) and r["added_chars"] > 0


def test_a_single_glyph_change_is_an_OTHER_EDIT_never_auto_applied(D):
    line = TIP[3].replace("⛛", "⛔")
    r = D.subclass(line, TIP)
    assert r["sub"] == "OTHER_EDIT" and r["chars_changed_on_tip_side"] == 1


def test_a_line_with_no_near_tip_line_is_NEW(D):
    assert D.subclass("| **D-REFE-RULE-1** | ⭐ **REFe NOW SELECTS WITH NAVSIM v1's OWN FORMULA.** |", TIP)["sub"] == "NEW"


def test_the_row_id_pattern_reads_ids_and_ignores_prose_cells(D):
    got = {t: (m.group(1) if (m := D.ROW_ID.match(t)) else None) for t in
           ("| C135 | x", "| **D-REFE-SEL-1** | x", "| ⛔⛔ **D-TLIGHT-1** | x", "| Validation of x |")}
    assert list(got.values()) == ["C135", "D-REFE-SEL-1", "D-TLIGHT-1", None]


# ---------------------------------------------------------------- the emitted APPEND block is the block, whole
def _rep(tmp_path, W, rows):
    return {"_W": W, "file": str(tmp_path / "REG.md"), "tip": "0" * 40, "rows": rows}


def test_a_line_already_on_the_tip_inside_a_new_block_is_kept(D, tmp_path):
    """A new entry may repeat a common line (a separator, a stock phrase). Dropping it would corrupt the block."""
    W = ["old", "### R25 new entry", "", "**Fixed:** nothing.", "", "last new line", "tail"]
    rows = [{"n": 2, "sub": "NEW"}, {"n": 6, "sub": "NEW"}]          # line 4 is on the tip, so it is not a row
    D.emit(_rep(tmp_path, W, rows), tmp_path)
    body = (tmp_path / "APPEND_REG.md").read_text(encoding="utf-8")
    assert "**Fixed:** nothing." in body and body.count("### R25 new entry") == 1 and "tail" not in body


def test_each_region_names_its_anchor_by_tip_line_number_not_by_copied_text(D, tmp_path):
    """The anchor must locate the insertion point WITHOUT copying the anchor line's text: that text may carry a
    clip handle, and a new file carrying one grows the count the clip-id guard exists to hold."""
    W = ["# head", "an anchor line with clip `cafef00d`", "### R25 new entry", "body"]
    T = ["# head", "x", "an anchor line with clip `cafef00d`"]
    rep = {"_W": W, "_T": T, "file": str(tmp_path / "REG.md"), "tip": "0" * 40,
           "rows": [{"n": 3, "sub": "NEW"}, {"n": 4, "sub": "NEW"}]}
    D.emit(rep, tmp_path)
    body = (tmp_path / "APPEND_REG.md").read_text(encoding="utf-8")
    assert "insert after tip line 3 " in body and "cafef00d" not in body


def test_rows_after_a_replaced_row_are_anchored_to_its_id(D):
    W = ["| **D-REFE-SEL-1** | extended |", "| **D-REFE-RULE-1** | new |"]
    rep = {"_W": W, "_T": [], "rows": [{"n": 1, "sub": "DESCENDANT", "row_id": "D-REFE-SEL-1"}, {"n": 2, "sub": "NEW"}]}
    assert D.anchor(rep, 2) == "insert after the row D-REFE-SEL-1 (it is replaced -- see REPLACE file)"


def test_a_replaced_field_row_without_an_id_is_cited_by_its_tip_line(D):
    """MODEL_REGISTRY's `| **Status** |` rows carry no id; the anchor once read 'insert after the row None'."""
    tip = ["| **Status** | TRAINING. |", "| **Other** | y |"]
    r = D.subclass("| **Status** | TRAINING. EVALUATED on 200 tokens. |", tip)
    assert r["sub"] == "DESCENDANT" and r["tip_line"] == 1
    rep = {"_W": ["| **Status** | TRAINING. EVALUATED on 200 tokens. |", "| **New row** | z |"], "_T": tip,
           "rows": [{"n": 1, "row_id": None, **r}, {"n": 2, "sub": "NEW"}]}
    assert D.anchor(rep, 2) == "insert after the row at tip line 1 (it is replaced -- see REPLACE file)"


def test_another_d_only_line_between_two_new_lines_splits_the_block(D, tmp_path):
    W = ["### A new", "a redacted row", "### B new"]
    rows = [{"n": 1, "sub": "NEW"}, {"n": 2, "sub": "SHA12_REDACTION"}, {"n": 3, "sub": "NEW"}]
    D.emit(_rep(tmp_path, W, rows), tmp_path)
    body = (tmp_path / "APPEND_REG.md").read_text(encoding="utf-8")
    assert "a redacted row" not in body and "### A new" in body and "### B new" in body
    assert "2 region(s)" in body
