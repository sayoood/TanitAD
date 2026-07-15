"""Unit tests for the CI commit-gate logic (scripts/ci_check.py).

These drive the pure helpers on synthetic pytest output — no nested pytest
process — so they are fast and deterministic. The subprocess orchestration in
``main`` is exercised end-to-end by ``scripts/ci.ps1`` in the research note.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ci_check as ci  # noqa: E402


SAMPLE = """\
============================ slowest durations =============================
2.15s call     tests/test_driving_diagnostic.py::test_pipeline_end_to_end_cpu
1.21s call     tests/test_smoke_train.py::test_smoke_training
1.02s setup    tests/test_resim.py::test_index_served
0.53s call     tests/test_refb.py::test_ckpt_roundtrip_and_resume
0.004s teardown tests/test_x.py::test_trivial
========================= 351 passed, 2 skipped in 21.11s =========================
"""

FAIL_SAMPLE = """\
0.5s call tests/test_a.py::test_a
========================= 1 failed, 350 passed in 22.0s =========================
"""


def test_parse_durations_extracts_all_rows():
    ds = ci.parse_durations(SAMPLE)
    assert len(ds) == 5
    assert ds[0].seconds == 2.15
    assert ds[0].phase == "call"
    assert ds[0].nodeid.endswith("test_pipeline_end_to_end_cpu")


def test_max_call_duration_ignores_setup_teardown():
    ds = ci.parse_durations(SAMPLE)
    slowest = ci.max_call_duration(ds)
    # the 1.02s row is 'setup' and must NOT win over the 2.15s call
    assert slowest.phase == "call"
    assert slowest.seconds == 2.15


def test_max_call_duration_none_when_no_calls():
    ds = [ci.Duration(1.0, "setup", "x"), ci.Duration(2.0, "teardown", "y")]
    assert ci.max_call_duration(ds) is None


def test_suite_passed_true_and_false():
    assert ci.suite_passed(SAMPLE) is True
    assert ci.suite_passed(FAIL_SAMPLE) is False
    assert ci.suite_passed("no summary here") is False


def test_reported_wall_reads_summary_time():
    assert ci.reported_wall(SAMPLE) == 21.11
    assert ci.reported_wall("nothing") is None


def test_evaluate_all_green():
    v = ci.evaluate(
        i2_passed=True, suite_ok=True,
        slowest=ci.Duration(2.15, "call", "n"),
        slow_budget_s=6.0, warm_wall_s=21.0, warm_budget_s=0.0,
    )
    assert v.ok and v.reasons == []


def test_evaluate_slow_test_fails_the_gate():
    # This is the registered falsifier: one over-budget call fails CI.
    v = ci.evaluate(
        i2_passed=True, suite_ok=True,
        slowest=ci.Duration(7.3, "call", "tests/test_slow.py::test_sleepy"),
        slow_budget_s=6.0, warm_wall_s=21.0, warm_budget_s=0.0,
    )
    assert not v.ok
    assert any("test_sleepy" in r and "7.3" in r for r in v.reasons)


def test_evaluate_i2_failure_dominates():
    v = ci.evaluate(
        i2_passed=False, suite_ok=True, slowest=None,
        slow_budget_s=6.0, warm_wall_s=1.0, warm_budget_s=0.0,
    )
    assert not v.ok
    assert any("I2 tripwire" in r for r in v.reasons)


def test_evaluate_warm_budget_only_when_positive():
    common = dict(i2_passed=True, suite_ok=True,
                  slowest=ci.Duration(1.0, "call", "n"), slow_budget_s=6.0,
                  warm_wall_s=40.0)
    assert ci.evaluate(warm_budget_s=0.0, **common).ok          # disabled
    assert not ci.evaluate(warm_budget_s=30.0, **common).ok     # 40 > 30 fails


def test_evaluate_suite_failure_flagged():
    v = ci.evaluate(
        i2_passed=True, suite_ok=False, slowest=None,
        slow_budget_s=6.0, warm_wall_s=1.0, warm_budget_s=0.0,
    )
    assert not v.ok
    assert any("did not pass" in r for r in v.reasons)


def test_quick_suite_targets_exist():
    """The curated subset must reference real test files (guards against drift
    when a safety test file is renamed)."""
    tests_dir = Path(__file__).resolve().parent
    for rel in ci.QUICK_SUITE:
        assert (tests_dir.parent / rel).exists(), f"QUICK_SUITE stale: {rel} missing"


def test_i2_node_points_at_a_real_test():
    path, _, name = ci.I2_NODE.partition("::")
    src = (Path(__file__).resolve().parent.parent / path).read_text(encoding="utf-8")
    assert f"def {name}(" in src
