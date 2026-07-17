"""Falsifier tests for ci_gate — each drives a tiny synthetic pytest project so
the gate's real behaviour (it shells out to pytest) is exercised end-to-end,
fast (<2 s each). These ARE the backlog falsifiers made executable:

  * green suite            -> gate returns 0
  * a failing test         -> gate returns non-zero  (correctness)
  * a collection ImportError-> gate returns non-zero (the 2026-07-17 breakage)
  * a slow test over budget-> gate returns non-zero  (the "slow fixture" falsifier)
  * a required tripwire     -> present+green passes; missing/failing fails
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ci_gate  # noqa: E402


def _project(tmp_path: Path, body: str, name: str = "test_synth.py") -> Path:
    d = tmp_path / "proj" / "tests"
    d.mkdir(parents=True)
    (d / name).write_text(body, encoding="utf-8")
    return tmp_path / "proj"


def _run(root: Path, **kw) -> int:
    return ci_gate.main(["--rootdir", str(root), "--require", "", *kw.pop("args", [])])


def test_green_suite_passes(tmp_path):
    root = _project(tmp_path, "def test_ok():\n    assert 1 + 1 == 2\n")
    assert _run(root) == 0


def test_failing_test_fails_gate(tmp_path):
    root = _project(tmp_path, "def test_bad():\n    assert False\n")
    assert _run(root) == 1


def test_collection_import_error_fails_gate(tmp_path):
    # exactly the 2026-07-17 class of breakage: a test importing a symbol its
    # implementation never shipped -> pytest exit 2 at collection, gate must catch.
    root = _project(tmp_path,
                    "from nonexistent_pkg import missing_symbol\n"
                    "def test_never_runs():\n    assert True\n")
    assert _run(root) == 1


def test_slow_test_over_budget_fails_gate(tmp_path):
    root = _project(tmp_path,
                    "import time\n"
                    "def test_slow():\n    time.sleep(0.6)\n    assert True\n")
    # passes correctness but blows a tight per-test budget -> gate fails.
    rc = ci_gate.main(["--rootdir", str(root), "--require", "",
                       "--max-test-seconds", "0.2"])
    assert rc == 1


def test_slow_test_within_budget_passes(tmp_path):
    root = _project(tmp_path,
                    "import time\n"
                    "def test_slow():\n    time.sleep(0.6)\n    assert True\n")
    rc = ci_gate.main(["--rootdir", str(root), "--require", "",
                       "--max-test-seconds", "5.0"])
    assert rc == 0


def test_wall_budget_enforced(tmp_path):
    root = _project(tmp_path, "def test_ok():\n    assert True\n")
    rc = ci_gate.main(["--rootdir", str(root), "--require", "",
                       "--max-wall-seconds", "0.0001"])
    assert rc == 1


def test_required_tripwire_present_and_green_passes(tmp_path):
    root = _project(tmp_path, "def test_i2_guard():\n    assert True\n")
    rc = ci_gate.main(["--rootdir", str(root), "--require", "test_i2_guard"])
    assert rc == 0


def test_required_tripwire_missing_fails(tmp_path):
    root = _project(tmp_path, "def test_something_else():\n    assert True\n")
    rc = ci_gate.main(["--rootdir", str(root), "--require", "test_i2_guard"])
    assert rc == 1


def test_required_tripwire_failing_fails(tmp_path):
    root = _project(tmp_path, "def test_i2_guard():\n    assert False\n")
    rc = ci_gate.main(["--rootdir", str(root), "--require", "test_i2_guard"])
    assert rc == 1


def test_no_tests_collected_fails(tmp_path):
    (tmp_path / "empty").mkdir()
    rc = ci_gate.main(["--rootdir", str(tmp_path / "empty"), "--require", ""])
    assert rc == 1


def test_parse_junit_classifies_statuses(tmp_path):
    xml = tmp_path / "j.xml"
    xml.write_text(
        '<testsuites><testsuite>'
        '<testcase classname="m" name="a" time="0.1"/>'
        '<testcase classname="m" name="b" time="2.0"><failure/></testcase>'
        '<testcase classname="m" name="c" time="0.0"><error/></testcase>'
        '<testcase classname="m" name="d" time="0.0"><skipped/></testcase>'
        '</testsuite></testsuites>', encoding="utf-8")
    cases = ci_gate.parse_junit(xml)
    by = {c.name: c.status for c in cases}
    assert by == {"a": "passed", "b": "failed", "c": "error", "d": "skipped"}
    assert next(c for c in cases if c.name == "b").time == 2.0
