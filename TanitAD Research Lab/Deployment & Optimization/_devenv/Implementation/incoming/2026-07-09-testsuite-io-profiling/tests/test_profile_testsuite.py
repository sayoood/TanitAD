"""Standalone tests for the test-suite profiler parsers + budget logic.

Pure: feeds canned pytest text to the parsers; never spawns pytest. Run with
    pytest "TanitAD Research Hub/Tools&DevEnv/Implementation/incoming/2026-07-09-testsuite-io-profiling/tests" -q
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from profile_testsuite import (  # noqa: E402
    Duration,
    RunReport,
    Summary,
    cmd_check,
    parse_durations,
    parse_summary,
)

# real captured tail from the 2026-07-09 dev-machine run
SAMPLE = """\
........................................................................ [ 39%]
............................................                             [100%]
============================ slowest 25 durations =============================
3.02s call     tests/test_smoke_train.py::test_smoke_training
1.09s call     tests/test_imagination.py::test_base250_parameter_budget
0.03s setup    tests/test_comma2k19.py::test_actions_and_poses_math
0.02s call     tests/test_instruments.py::test_i2_batch_consistency_of_encoder
181 passed, 1 skipped in 9.23s
"""

FAIL_SAMPLE = """\
1.10s call     tests/test_x.py::test_slow
5 passed, 2 failed, 1 skipped in 1.10s
"""


def test_parse_durations_extracts_all_rows():
    ds = parse_durations(SAMPLE)
    assert len(ds) == 4
    assert ds[0] == Duration(3.02, "call", "tests/test_smoke_train.py::test_smoke_training")
    # the setup-phase row is captured with its phase, not misread as a call
    setup = [d for d in ds if d.phase == "setup"]
    assert len(setup) == 1 and setup[0].seconds == 0.03


def test_parse_durations_ignores_dots_and_headers():
    # progress dots and the header line must not parse as durations
    assert parse_durations("....... [100%]\nslowest 25 durations\n") == []


def test_parse_summary_passed_skipped():
    s = parse_summary(SAMPLE)
    assert s == Summary(passed=181, failed=0, skipped=1, reported_sec=9.23)


def test_parse_summary_with_failures():
    s = parse_summary(FAIL_SAMPLE)
    assert s.passed == 5 and s.failed == 2 and s.skipped == 1
    assert s.reported_sec == 1.10


def test_parse_summary_raises_when_absent():
    try:
        parse_summary("no summary here\njust dots ....")
    except ValueError:
        return
    raise AssertionError("expected ValueError on missing summary")


def test_overhead_is_wall_minus_reported():
    r = RunReport("warm", wall_sec=10.7, summary=Summary(passed=181, reported_sec=9.2))
    assert r.overhead_sec == 1.5


def _check_args(**kw):
    base = dict(stack_dir="stack", durations=25, max_warm_overhead=4.0, max_test=6.0)
    base.update(kw)
    return argparse.Namespace(**base)


def test_check_passes_on_healthy_run(monkeypatch):
    import profile_testsuite as m
    healthy = RunReport(
        "warm", wall_sec=10.7,
        summary=Summary(passed=181, skipped=1, reported_sec=9.2),
        slowest=[Duration(3.02, "call", "tests/test_smoke_train.py::test_smoke_training")],
    )
    monkeypatch.setattr(m, "measure", lambda *a, **k: healthy)
    assert cmd_check(_check_args()) == 0


def test_check_fails_on_slow_test_and_overhead(monkeypatch):
    import profile_testsuite as m
    bad = RunReport(
        "warm", wall_sec=30.0,
        summary=Summary(passed=181, reported_sec=20.0),  # overhead 10 > 4
        slowest=[Duration(9.0, "call", "tests/test_new.py::test_heavy")],  # 9 > 6
    )
    monkeypatch.setattr(m, "measure", lambda *a, **k: bad)
    assert cmd_check(_check_args()) == 1


def test_check_fails_on_test_failure(monkeypatch):
    import profile_testsuite as m
    failing = RunReport("warm", wall_sec=11.0,
                        summary=Summary(passed=180, failed=1, reported_sec=9.2))
    monkeypatch.setattr(m, "measure", lambda *a, **k: failing)
    assert cmd_check(_check_args()) == 1
