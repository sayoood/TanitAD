"""Falsifier tests for ci_gate — each drives a tiny synthetic pytest project so
the gate's real behaviour (it shells out to pytest) is exercised end-to-end,
fast (<2 s each). These ARE the backlog falsifiers made executable:

  v1  * green suite             -> gate returns 0
      * a failing test          -> non-zero  (correctness)
      * a collection ImportError-> non-zero  (the 2026-07-17 breakage)
      * a slow test over budget -> non-zero  (the "slow fixture" falsifier)
      * a required tripwire      -> present+green passes; missing/failing fails

  v2  * a required SUITE below its collected floor -> non-zero (module deleted
        or renamed — the failure a single-node tripwire cannot see)
      * a required suite present but red            -> non-zero
      * --min-total below the collected count       -> non-zero
      * the spec parser: path/dotted/`>=N` forms, and a bad floor raises
      * --gpu-smoke require with no CUDA visible     -> non-zero
      * --gpu-smoke warn with no CUDA visible        -> still 0 (advisory only)
      * --json writes a machine-readable report with the reasons
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ci_gate  # noqa: E402

# Synthetic projects have neither the manifest suites nor 390 tests, so the v2
# whole-tree defaults are switched off unless a test is specifically about them.
NO_DEFAULTS = ["--no-default-suites", "--min-total", "0"]


def _project(tmp_path: Path, body: str, name: str = "test_synth.py") -> Path:
    d = tmp_path / "proj" / "tests"
    d.mkdir(parents=True)
    (d / name).write_text(body, encoding="utf-8")
    return tmp_path / "proj"


def _run(root: Path, *extra: str, require: str = "") -> int:
    return ci_gate.main(["--rootdir", str(root), "--require", require,
                         *NO_DEFAULTS, *extra])


# ----------------------------------------------------------------- v1 falsifiers


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
    assert _run(root, "--max-test-seconds", "0.2") == 1


def test_slow_test_within_budget_passes(tmp_path):
    root = _project(tmp_path,
                    "import time\n"
                    "def test_slow():\n    time.sleep(0.6)\n    assert True\n")
    assert _run(root, "--max-test-seconds", "5.0") == 0


def test_wall_budget_enforced(tmp_path):
    root = _project(tmp_path, "def test_ok():\n    assert True\n")
    assert _run(root, "--max-wall-seconds", "0.0001") == 1


def test_required_tripwire_present_and_green_passes(tmp_path):
    root = _project(tmp_path, "def test_i2_guard():\n    assert True\n")
    assert _run(root, require="test_i2_guard") == 0


def test_required_tripwire_missing_fails(tmp_path):
    root = _project(tmp_path, "def test_something_else():\n    assert True\n")
    assert _run(root, require="test_i2_guard") == 1


def test_required_tripwire_failing_fails(tmp_path):
    root = _project(tmp_path, "def test_i2_guard():\n    assert False\n")
    assert _run(root, require="test_i2_guard") == 1


def test_no_tests_collected_fails(tmp_path):
    (tmp_path / "empty").mkdir()
    assert ci_gate.main(["--rootdir", str(tmp_path / "empty"), "--require", "",
                         *NO_DEFAULTS]) == 1


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


# ----------------------------------------------------------------- v2 falsifiers


def test_required_suite_at_floor_passes(tmp_path):
    root = _project(tmp_path,
                    "def test_a():\n    assert True\n"
                    "def test_b():\n    assert True\n")
    assert _run(root, "--require-suite", "tests/test_synth.py>=2") == 0


def test_required_suite_below_floor_fails(tmp_path):
    """The failure a single-node tripwire cannot see: the module is still there
    and green, but somebody removed half of it."""
    root = _project(tmp_path, "def test_a():\n    assert True\n")
    assert _run(root, "--require-suite", "tests/test_synth.py>=2") == 1


def test_required_suite_renamed_away_fails(tmp_path):
    root = _project(tmp_path, "def test_a():\n    assert True\n",
                    name="test_renamed.py")
    assert _run(root, "--require-suite", "tests/test_synth.py>=1") == 1


def test_required_suite_present_but_red_fails(tmp_path):
    root = _project(tmp_path,
                    "def test_a():\n    assert True\n"
                    "def test_b():\n    assert False\n")
    # correctness alone would already fail; assert the SUITE reason is the one
    # reported, so the operator is told which module lost its green.
    report = ci_gate.Report(exit_code=1, wall_s=0.1, cases=[
        ci_gate.Case("test_a", "tests.test_synth", 0.0, "passed"),
        ci_gate.Case("test_b", "tests.test_synth", 0.0, "failed"),
    ])
    reasons = ci_gate.evaluate(report, 15.0, 90.0, [],
                               suites={"tests.test_synth": 2})
    assert any("not green" in r and "tests.test_synth" in r for r in reasons)
    assert _run(root, "--require-suite", "tests/test_synth.py>=2") == 1


def test_min_total_floor_fails_on_shrunken_suite(tmp_path):
    root = _project(tmp_path, "def test_a():\n    assert True\n")
    rc = ci_gate.main(["--rootdir", str(root), "--require", "",
                       "--no-default-suites", "--min-total", "5"])
    assert rc == 1


def test_min_total_zero_disables_the_floor(tmp_path):
    root = _project(tmp_path, "def test_a():\n    assert True\n")
    assert _run(root) == 0          # NO_DEFAULTS sets --min-total 0


@pytest.mark.parametrize("spec,expected", [
    ("tests/test_lake.py>=9", ("tests.test_lake", 9)),
    ("tests\\test_lake.py>=9", ("tests.test_lake", 9)),
    ("tests.test_lake >= 9", ("tests.test_lake", 9)),
    ("tests/test_lake.py", ("tests.test_lake", 1)),
])
def test_suite_spec_parsing(spec, expected):
    assert ci_gate.parse_suite_spec(spec) == expected


def test_suite_spec_bad_floor_raises():
    with pytest.raises(ValueError):
        ci_gate.parse_suite_spec("tests/test_lake.py>=nine")


def test_builtin_manifest_is_normalized_and_nonempty():
    """A manifest entry written in path form must survive normalization to the
    JUnit classname form, or every suite check silently passes vacuously."""
    assert ci_gate.SUITE_MANIFEST
    for k in ci_gate.SUITE_MANIFEST:
        assert ci_gate.suite_key(k).startswith("tests.")
        assert "/" not in ci_gate.suite_key(k)


def test_gpu_probe_failure_is_a_gate_reason():
    report = ci_gate.Report(exit_code=0, wall_s=1.0, cases=[
        ci_gate.Case("test_a", "tests.test_synth", 0.0, "passed")])
    report.gpu = {"available": True, "error": None, "probes": [
        {"name": "P1_encode_parity", "ok": False, "detail": "max dev 7e-1"},
        {"name": "P4_backward_finite", "ok": True, "detail": "ok"},
    ]}
    reasons = ci_gate.evaluate(report, 15.0, 90.0, [])
    assert len(reasons) == 1 and "P1_encode_parity" in reasons[0]


def test_gpu_absent_is_a_gate_reason_when_required():
    report = ci_gate.Report(exit_code=0, wall_s=1.0, cases=[
        ci_gate.Case("test_a", "tests.test_synth", 0.0, "passed")])
    report.gpu = {"available": False, "error": "no CUDA device visible",
                  "probes": []}
    reasons = ci_gate.evaluate(report, 15.0, 90.0, [])
    assert any("no CUDA device" in r for r in reasons)


def test_gpu_smoke_warn_never_blocks(tmp_path):
    """'warn' prints the report but must not contribute a gate reason — that is
    what makes it safe to leave on in every agent's local run."""
    root = _project(tmp_path, "def test_a():\n    assert True\n")
    assert _run(root, "--gpu-smoke", "warn") == 0


def test_json_report_written(tmp_path):
    root = _project(tmp_path, "def test_a():\n    assert False\n")
    out = tmp_path / "gate.json"
    rc = _run(root, "--json", str(out))
    assert rc == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["pass"] is False
    assert payload["collected"] == 1
    assert payload["counts"]["failed"] == 1
    assert payload["reasons"]


def test_required_suite_with_a_legitimate_skip_stays_green():
    """Measured 2026-07-20: tests.test_scena skips one MiniLM-download test. A
    skip is not a failure -- gating on it would make the manifest unusable."""
    report = ci_gate.Report(exit_code=0, wall_s=1.0, cases=[
        ci_gate.Case("test_a", "tests.test_scena", 0.0, "passed"),
        ci_gate.Case("test_minilm", "tests.test_scena", 0.0, "skipped"),
    ])
    assert ci_gate.evaluate(report, 15.0, 90.0, [],
                            suites={"tests.test_scena": 2}) == []


def test_entirely_skipped_suite_fails():
    """...but a module that skips *everything* is coverage loss wearing a green
    hat -- the exact thing a collected-count floor alone cannot see."""
    report = ci_gate.Report(exit_code=0, wall_s=1.0, cases=[
        ci_gate.Case("test_a", "tests.test_scena", 0.0, "skipped"),
        ci_gate.Case("test_b", "tests.test_scena", 0.0, "skipped"),
    ])
    reasons = ci_gate.evaluate(report, 15.0, 90.0, [],
                               suites={"tests.test_scena": 2})
    assert len(reasons) == 1 and "entirely SKIPPED" in reasons[0]


# --- budget rebasing + SLOW_EXEMPT (2026-08-03) ---------------------------
# The 150 s wall budget was calibrated against a 531-test suite. Measured
# 2026-08-03 the suite is 1708 collected / 1694 passed in 186.7 s, so the gate
# failed on SIZE while nothing was wrong (0 failed, 0 error). A gate that is
# red for everyone gets ignored -- the muted-monitor failure mode.

def _case(nodeid, t, status="passed"):
    cls, _, name = nodeid.rpartition("::")
    return ci_gate.Case(name=name, classname=cls, time=t, status=status)


def _report(cases, wall):
    return ci_gate.Report(exit_code=0, wall_s=wall, cases=cases)


EXEMPT_ID = next(iter(ci_gate.SLOW_EXEMPT))


def test_an_exempt_test_does_not_fail_the_gate_but_a_new_slow_one_does():
    """The exemption must be surgical: it clears exactly one named test."""
    rep = _report([_case(EXEMPT_ID, 41.8),
                   _case("tests.test_other::test_regressed", 40.0)], wall=100)
    reasons = ci_gate.evaluate(rep, 15.0, 300.0, require=[])
    assert not any(EXEMPT_ID in r for r in reasons)
    assert any("test_regressed" in r and "slow test" in r for r in reasons)


def test_the_rebased_wall_budget_passes_todays_measured_suite():
    """186.7 s measured 2026-08-03 must pass; the old 150 s must not."""
    rep = _report([_case("tests.test_a::test_x", 1.0)], wall=186.7)
    assert ci_gate.evaluate(rep, 15.0, 300.0, require=[]) == []
    assert ci_gate.evaluate(rep, 15.0, 150.0, require=[])   # the stale budget


def test_the_wall_budget_still_fires_above_the_shard_ceiling():
    """Inverted control: rebasing must not disable the check. 300 s stays
    under the 5-minute 'shard it instead' ceiling, so crossing it is real."""
    rep = _report([_case("tests.test_a::test_x", 1.0)], wall=310.0)
    reasons = ci_gate.evaluate(rep, 15.0, 300.0, require=[])
    assert any("suite wall" in r for r in reasons)


def test_every_exemption_documents_why_the_cost_is_intrinsic():
    """Guards the list against becoming a dumping ground."""
    assert ci_gate.SLOW_EXEMPT
    for nodeid, why in ci_gate.SLOW_EXEMPT.items():
        assert "::" in nodeid
        assert len(why) > 40 and "s (" in why      # a measured time and a reason
