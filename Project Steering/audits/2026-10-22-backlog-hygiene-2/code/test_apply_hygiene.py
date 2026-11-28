"""`apply_hygiene.py`: does it strike what it should, refuse what it must, and lose nothing?

Run from the package root (`code/`) or with `--rootdir`:
    python -m pytest -q -p no:cacheprovider test_apply_hygiene.py

The integration tests read the TIP BLOB the script was written against
(`../raw/BACKLOG.tip_blob.md`, sha256 064960afc7329c47...); they SKIP, loudly, when
it is absent rather than pass vacuously. Everything else uses verbatim live rows
from the tool's own fixture, so no test here invents a row.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import apply_hygiene as ah  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "backlog_drift", HERE / "tools" / "backlog_drift.py")
bd = importlib.util.module_from_spec(_spec)
sys.modules["backlog_drift"] = bd
_spec.loader.exec_module(bd)

FIXTURE = HERE / "tools" / "tests" / "fixtures" / "backlog_rows_2026-10-22.md"
TIP_BLOB = HERE.parent / "raw" / "BACKLOG.tip_blob.md"
TIP_SHA = "064960afc7329c47ceb034e977f0b504bf5ad266eec12b09a87f058c1b320777"


def _fixture_text() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def _verdicts(text: str) -> dict:
    rows, _s, _b, missed = bd.parse(text)
    assert missed == [], missed
    return {v["id"]: v for v in bd.classify(rows)}


def _row(text: str, rid: str) -> str:
    hits = [ln for ln in text.split("\n") if ah._row_id(ln) == rid]
    assert len(hits) == 1, (rid, len(hits))
    return hits[0]


def _apply(text: str, **kw):
    return ah.apply(text, allow_absent=True, **kw)


# --------------------------------------------------------------------------
# on verbatim live rows
# --------------------------------------------------------------------------
def test_every_row_the_MM_listed_is_CLOSED_after_the_sweep():
    new, rep = _apply(_fixture_text())
    v = _verdicts(new)
    for rid in ("EP-R1UB", "HEADROW-PROBE", "PICKRULE-AUDIT-NC", "NC-FASTCAL",
                "EP-RELIAB-PICKS", "EP-ADMISSION-R1", "EP-ADMISSION-R2-CONFIRM",
                "PRIVGEOM-SAFE", "SEARCH-DOSE", "EPADM-SEARCHBANK-S0",
                "PRIVGEOM-DAC", "L2-FRESH-STEPA"):
        assert v[rid]["verdict"] == "CLOSED", (rid, v[rid]["verdict"])
        assert rid in rep["struck"]


def test_the_old_family_rows_are_struck_in_the_files_own_form():
    new, _rep = _apply(_fixture_text())
    assert _row(new, "A11").startswith("| ~~A11~~ |")
    assert _row(new, "R1").startswith("| ~~**R1**~~ |")
    v = _verdicts(new)
    for rid in ("A11", "A13", "B4", "C3", "R1", "R2", "R4"):
        assert v[rid]["verdict"] == "CLOSED", (rid, v[rid]["verdict"])


def test_rows_the_sweep_must_NOT_touch_are_byte_identical():
    """The negative control: the P7-WIRE rows, every multi-step row, and ZG-WIRE
    (DONE with explicit OPEN items (a)-(c) -- left for the Master Mind)."""
    old = _fixture_text()
    new, rep = _apply(old)
    for rid in ("P7-WIRE-RESERVE", "P7-WIRE-HEAD", "SEL-RANK", "P6-HN", "PAPER-720",
                "R0E-PICK", "SH5-DAC-PROX", "EP-ADMISSION", "P6-3C",
                "SEARCH-PESSIMISTIC", "FASTNEG-F25-RERUN", "SEARCH-NAVTEST-PIPE",
                "ZG-WIRE"):
        assert _row(new, rid) == _row(old, rid), rid
        assert rid not in rep["struck"] + rep["relabelled"]
    assert _verdicts(new)["ZG-WIRE"]["verdict"] == "DONE-BUT-OPEN"


def test_the_MM_rulings_relabel_the_LEAD():
    old = _fixture_text()
    new, rep = _apply(old)
    v = _verdicts(new)
    assert v["SEARCH-P2-CONFIRM"]["verdict"] == "CLOSED"
    assert v["LATENCY-BUDGET"]["verdict"] == "CLOSED"
    row = _row(new, "SEARCH-P2-CONFIRM")
    assert "CLOSED 2026-10-22: C2_FAIL_LOWER_CI" in row
    assert "DECIDED 2026-10-20 (PI, D-LATENCY-BUDGET)" in _row(new, "LATENCY-BUDGET")
    for rid in ("SEARCH-HOST-PORT", "THOR-PICK-TICK", "SEARCH-THOR-LATENCY"):
        r = _row(new, rid)
        assert not ah._FIRST_CELL.match(r).group(1).count("~~"), "DEFERRED is not CLOSED"
        assert v[rid]["verdict"] == "OPEN"
        body = r.split("|", 2)[2].strip()
        assert bd.lead_status(body) == "deferred", (rid, body[:80])
    assert ("the search is CLOSED on P2's scorer (C2, 2026-10-22); re-opens with a "
            "P6/P7 scorer").lower() in _row(new, "SEARCH-THOR-LATENCY").lower()


def test_C2_loses_its_STALE_BLOCKER_verdict_without_losing_a_word():
    old = _fixture_text()
    assert _verdicts(old)["C2"]["verdict"] == "STALE-BLOCKER"
    new, rep = _apply(old)
    assert rep["restated"] == ["C2"]
    r_old, r_new = _row(old, "C2"), _row(new, "C2")
    assert "~~v2corpus" not in r_new and "v2corpus reaching 30" in r_new
    assert "UNVERIFIABLE, not cleared" in r_new
    assert "RE-STATED 2026-10-22 (BACKLOG-HYGIENE-2)" in r_new
    assert ah._no_text_lost(r_old, r_new, [ah.C2_INLINE_TAG, ah.C2_NOTE])
    assert _verdicts(new)["C2"]["verdict"] == "OPEN"


# --------------------------------------------------------------------------
# the script's own contract
# --------------------------------------------------------------------------
def test_applying_twice_is_applying_once():
    once, r1 = _apply(_fixture_text())
    twice, r2 = _apply(once)
    assert twice == once
    assert r2["struck"] == [] and r2["relabelled"] == [] and r2["restated"] == []
    assert set(r1["struck"]) <= set(r2["already_struck"])


def test_a_DUPLICATE_id_is_always_a_refusal(capsys):
    text = _fixture_text() + "\n## again\n\n| # | item |\n|---|---|\n| EP-R1UB | x |\n"
    with pytest.raises(SystemExit) as e:
        ah.apply(text, allow_absent=True)           # even with --allow-absent
    assert e.value.code == 2
    assert "EP-R1UB: 2 occurrence" in capsys.readouterr().out


def test_an_ABSENT_id_refuses_by_default_and_skips_under_allow_absent(capsys):
    text = _fixture_text()
    with pytest.raises(SystemExit) as e:
        ah.apply(text)                 # the fixture holds only some of the targets
    assert e.value.code == 2 and "0 occurrence" in capsys.readouterr().out
    _new, rep = ah.apply(text, allow_absent=True)
    assert rep["absent"], "allow_absent must REPORT what it skipped"


def test_a_row_whose_closure_evidence_is_gone_is_NOT_struck(capsys):
    """The bytes are newer than the script: a row reopened since must not be
    struck blindly. Same-breath control: the untouched row beside it IS struck."""
    old = _fixture_text()
    reopened = old.replace("DONE 2026-10-19: MIXED", "NEXT: a second probe")
    assert reopened != old
    with pytest.raises(SystemExit) as e:
        ah.apply(reopened, allow_absent=True)
    assert e.value.code == 2
    out = capsys.readouterr().out
    assert "HEADROW-PROBE" in out and "no longer in the row" in out
    assert "EP-R1UB" not in out, "an unrelated row must not be reported"


def test_an_already_struck_row_is_skipped_not_refused():
    old = _fixture_text()
    pre = old.replace("| A11 |", "| ~~A11~~ |", 1)
    _new, rep = _apply(pre)
    assert "A11" in rep["already_struck"] and "A11" not in rep["struck"]


def test_NO_TEXT_IS_EVER_DELETED_and_the_guard_can_fail():
    """The contract, on every changed line of the fixture -- plus the control that
    proves `_no_text_lost` can say NO (a guard that cannot fail is not one)."""
    old = _fixture_text()
    new, _rep = _apply(old)
    changed = 0
    for a, b in zip(old.split("\n"), new.split("\n")):
        if a != b:
            changed += 1
            stripped = b
            assert len(b) > len(a) or "~~" in a, "an edit must add, or un-strike"
            pieces = ([ah.C2_INLINE_TAG, ah.C2_NOTE] if ah._row_id(a) == "C2"
                      else _inserted_pieces(a, b))
            assert ah._no_text_lost(a, stripped, pieces), a[:60]
    assert changed > 30, changed
    assert not ah._no_text_lost("| A1 | keep this text | y |", "| ~~A1~~ | keep | y |", [])


def _inserted_pieces(a: str, b: str) -> list:
    """Everything `b` has that `a` lacks, found by locating `a`'s tokens in order."""
    import difflib
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return ["".join(b[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes()
            if tag == "insert"]


@pytest.mark.parametrize("eol", ["\n", "\r\n"])
def test_line_endings_are_preserved(eol):
    base = _fixture_text()
    src = base.replace("\n", eol)
    new, _rep = _apply(src)
    assert ("\r\n" in new) == (eol == "\r\n")
    assert new.replace("\r\n", "\n") == _apply(base)[0]
    if eol == "\n":
        assert "\r" not in new
    else:
        # EVERY line keeps its CRLF, including the lines the sweep edited. (A
        # mutation that re-terminated only the edited lines with LF survived the
        # weaker assertions above, which is how this one was earned.)
        assert "\n" not in new.replace("\r\n", ""), "a line lost its CR"


# --------------------------------------------------------------------------
# the real tip blob
# --------------------------------------------------------------------------
needs_blob = pytest.mark.skipif(
    not TIP_BLOB.exists(), reason="tip blob raw/BACKLOG.tip_blob.md not present")


@needs_blob
def test_the_tip_blob_is_the_one_the_script_was_written_against():
    assert hashlib.sha256(TIP_BLOB.read_bytes()).hexdigest() == TIP_SHA


@needs_blob
def test_on_the_TIP_the_sweep_leaves_exactly_the_rows_listed_for_the_MM():
    """Before: 54 rows drift. After: ONE (ZG-WIRE: DONE with explicit open items).
    Every id the script strikes was flagged by the tool, and the tool flags nothing
    the script neither strikes nor lists -- so script and detector agree."""
    old = TIP_BLOB.read_text(encoding="utf-8")
    before = _verdicts(old)
    flagged = {k for k, v in before.items()
               if v["verdict"] in ("DONE-BUT-OPEN", "CLOSED-LATER", "STALE-BLOCKER")}
    struck_ids = {rid for rid, _a in ah.STRIKES}
    assert flagged == struck_ids | {"ZG-WIRE", "C2"}, flagged ^ (struck_ids | {"ZG-WIRE", "C2"})
    new, rep = ah.apply(old)                     # strict: NO --allow-absent on the tip
    after = _verdicts(new)
    left = {k for k, v in after.items()
            if v["verdict"] in ("DONE-BUT-OPEN", "CLOSED-LATER", "STALE-BLOCKER")}
    assert left == {"ZG-WIRE"}, left
    assert sum(v["verdict"] == "CLOSED" for v in after.values()) == \
        sum(v["verdict"] == "CLOSED" for v in before.values()) + 54
    assert len(rep["struck"]) == 52 and len(rep["relabelled"]) == 5
    assert ah.apply(new)[0] == new, "not idempotent on the tip"


@needs_blob
def test_on_the_TIP_every_other_line_is_byte_identical():
    old = TIP_BLOB.read_text(encoding="utf-8").split("\n")
    new = ah.apply("\n".join(old))[0].split("\n")
    assert len(old) == len(new)
    changed = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
    assert len(changed) == 52 + 5 + 1, len(changed)


def test_the_no_deletion_contract_is_ENFORCED_by_apply_not_just_by_a_helper(
        monkeypatch, capsys):
    """Earned by a surviving mutation (`A3`): `_no_text_lost` was tested directly,
    but nothing proved `apply()` CALLS it. Inject an edit that truncates a row and
    require the refusal -- with a same-breath control that the unpatched call passes."""
    text = _fixture_text()
    ah.apply(text, allow_absent=True)                        # control: green
    real = ah._strike_id_cell
    monkeypatch.setattr(ah, "_strike_id_cell",
                        lambda body: real(body)[: len(body) // 2])
    with pytest.raises(SystemExit) as e:
        ah.apply(text, allow_absent=True)
    assert e.value.code == 2
    assert "would delete text" in capsys.readouterr().out
