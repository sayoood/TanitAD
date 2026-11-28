"""Regression fixtures for `backlog_claim_check.py`.

⛔ WHY THESE EXIST. The checker has **zero true positives on the live backlog** —
the one real case (A12) was found and fixed by hand before the tool was written.
So a green run proves nothing on its own: a checker that always returns 0 would
score identically. What makes it trustworthy is that it FIRES on the case that
motivated it and stays SILENT on the two shapes that fooled its first version.

That is the same discipline the programme applies to model gates: an arm that
re-introduces the defect must fail, or a pass on the fixed arm means nothing.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "scripts" / "backlog_claim_check.py"
_spec = importlib.util.spec_from_file_location("backlog_claim_check", _SRC)
bcc = importlib.util.module_from_spec(_spec)
sys.modules["backlog_claim_check"] = bcc
_spec.loader.exec_module(bcc)

REPO = Path(__file__).resolve().parents[2]

#: ⛔ HERMETIC BY CONSTRUCTION. The first version of these tests passed
#: `--repo REPO` and let the checker shell out to `git ls-files`. Run from the
#: off-Drive mirror that is not a git repo, git returned nothing, the checker
#: correctly REFUSED, and four fixtures failed for a reason that had nothing to
#: do with the logic under test. A fixture whose verdict depends on which tree
#: it happens to run in is not testing the tool.
#:
#: So the index is supplied directly. These are the real tracked paths the
#: fixtures reference (verified present in the repo on 2026-09-02).
FAKE_INDEX = {
    "stack/scripts/sel_winners_curse_law.py",
    "stack/scripts/lf0_bev_lead.py",
    "stack/scripts/p8_bev_reel.py",
    "stack/scripts/refc_train.py",
    "stack/tests/test_resim.py",
    "stack/tests/test_rig_clean_fix.py",
}


@pytest.fixture(autouse=True)
def _fixed_index(monkeypatch):
    """Every test below runs against FAKE_INDEX, never against live git."""
    monkeypatch.setattr(bcc, "tracked_paths", lambda repo: set(FAKE_INDEX))

#: A12 exactly as it read before 2026-09-02. `sel_winners_curse_law.py` is
#: annotated "(absent)" and is in fact tracked at `stack/scripts/`.
A12_PRE_FIX = (
    "| A12 | **Ship the 4 stale/absent files to Thor** — "
    "`sel_winners_curse_law.py` (absent), `lf0_bev_lead.py`, "
    "`p8_bev_reel.py`, `refc_train.py` | the real pre-fix text |\n"
)

#: F5: the absent-word belongs to a PACKAGE (`onnx`), and the filename names
#: where a failure lives. Cell-membership matching flagged this; proximity must
#: not.
F5_ONNX = (
    "| F5 | **The 23 standing pytest failures** — `onnx` absent (1), a "
    "Windows-basename assert in `test_resim.py` | must NOT fire |\n"
)

#: F7: a file to be CREATED. Untracked is correct, not a finding.
F7_PLANNED = (
    "| F7 | **Pod-side preflight** | `stack/scripts/v6_preflight.sh` (new) "
    "— must NOT fire |\n"
)

HEADER = "| # | item | why |\n|---|---|---|\n"


def _run(tmp_path: Path, body: str) -> tuple[int, str]:
    d = tmp_path / "Project Steering"
    d.mkdir(parents=True, exist_ok=True)
    bl = d / "BACKLOG.md"
    bl.write_text(HEADER + body, encoding="utf-8")
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = bcc.main(["--repo", str(REPO), "--backlog", str(bl)])
    return rc, buf.getvalue()


def test_it_fires_on_the_case_that_motivated_it(tmp_path):
    """⭐ THE NEGATIVE CONTROL. Without this, a green run is meaningless."""
    rc, out = _run(tmp_path, A12_PRE_FIX)
    assert rc == 1, "the pre-fix A12 row must be flagged"
    assert "sel_winners_curse_law.py" in out
    assert "STALE-ABSENT" in out


def test_it_is_silent_on_a_package_absent_word(tmp_path):
    """`onnx` absent must not be attributed to `test_resim.py` beside it."""
    rc, out = _run(tmp_path, F5_ONNX)
    assert "test_resim.py" not in out, out
    assert rc == 0, out


def test_it_is_silent_on_a_file_marked_new(tmp_path):
    """A planned file is not a missing one."""
    rc, out = _run(tmp_path, F7_PLANNED)
    assert "v6_preflight.sh" not in out, out
    assert rc == 0, out


def test_a_closed_row_is_not_re_flagged(tmp_path):
    """⛔ The echo trap: a row closed by explaining WHY it was stale still
    contains the word "absent". Reading that post-mortem as a live claim is what
    the first version did, flagging seven already-fixed rows."""
    closed = ("| ~~A12~~ | ~~Ship the files~~ DONE — the row's own \"(absent)\" "
              "annotation was STALE; `sel_winners_curse_law.py` exists | done |\n")
    rc, out = _run(tmp_path, closed)
    assert rc == 0, out
    assert "sel_winners_curse_law.py" not in out


def test_an_unreadable_index_refuses_rather_than_reporting_every_claim_wrong(
        tmp_path, monkeypatch):
    """⛔ An empty `git ls-files` is a FAILED READ, not an empty repo — and on
    this mount that distinction is not theoretical. Reporting "every file is
    missing" off a failed read is the exact class of error this tool exists to
    catch, so it must refuse instead."""
    monkeypatch.setattr(bcc, "tracked_paths", lambda repo: set())
    rc, out = _run(tmp_path, A12_PRE_FIX)
    assert rc == 2, out
    assert "REFUSING" in out
