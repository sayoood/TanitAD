"""RED-tests for the navtest quarantine gate (behaviour_gate.py) and its wiring into online_labeller._job.  Pure numpy/stdlib: runs under real pytest in
C:/Users/Admin/venvs/tanitad as well as under tests/run_tests.py.  ⛔ navtest is EVAL ONLY: every refusal below is asserted AND its deliberate-regression arm
(REFE_LAB_MUTATE=no_navtest_gate) is asserted to let the same input through -- so the test goes RED if the gate is ever removed."""
import gzip
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
A2 = os.path.dirname(HERE)
TIP = os.environ.get("A2_TIP_REFE", "F:/a2_scratch/tip_refe")
for p in (TIP, os.path.join(A2, "overlay", "refe")):      # overlay LAST in the list => first on sys.path
    if p not in sys.path:
        sys.path.insert(0, p)
import behaviour_gate as BG   # noqa: E402

MUT = BG.MUTATE_ENV


@pytest.fixture
def clean_env():
    old = os.environ.pop(MUT, None)
    os.environ.pop("A2_GATE_DATA", None)
    BG._CACHE.clear()
    yield
    os.environ.pop(MUT, None)
    os.environ.pop("A2_GATE_DATA", None)
    if old is not None:
        os.environ[MUT] = old
    BG._CACHE.clear()


def _a_navtest_pair():
    logs, toks = BG.load()
    return sorted(logs)[0], sorted(toks)[0]


def test_lists_are_the_expected_size(clean_env):
    logs, toks = BG.load()
    meta = json.load(open(os.path.join(A2, "gate_data", "navtest_gate.json")))
    assert len(logs) == meta["n_logs"] == 147
    assert len(toks) == meta["n_tokens"] > 50000


def test_navtest_log_is_refused_by_log_gate_alone(clean_env):
    log, _ = _a_navtest_pair()
    with pytest.raises(BG.NavtestLeak, match="LOG"):
        BG.assert_not_navtest(log, "0123456789abcdef")           # token NOT in navtest: the LOG key alone refuses
    with pytest.raises(BG.NavtestLeak):
        BG.assert_not_navtest(log + ".db", "0123456789abcdef")   # a DB filename is the same log


def test_navtest_token_is_refused_by_token_gate_alone(clean_env):
    _, tok = _a_navtest_pair()
    with pytest.raises(BG.NavtestLeak, match="TOKEN"):
        BG.assert_not_navtest("2021.01.01.00.00.00_veh-00_00000_00001", tok)   # renamed log, real navtest token
    with pytest.raises(BG.NavtestLeak):
        BG.assert_not_navtest("x_log", tok.upper())


def test_non_navtest_control_passes_and_unkeyed_is_refused(clean_env):
    BG.assert_not_navtest("2021.01.01.00.00.00_veh-00_00000_00001", "0123456789abcdef")   # control: must NOT raise
    for log, tok in (("", "0123456789abcdef"), ("some_log", ""), (None, None)):
        with pytest.raises(BG.NavtestLeak, match="unkeyed"):
            BG.assert_not_navtest(log, tok)


def test_deliberate_regression_arm_lets_navtest_through(clean_env):
    log, tok = _a_navtest_pair()
    os.environ[MUT] = BG.MUTATION
    BG.assert_not_navtest(log, tok)                               # gate OFF: no raise -> the "is refused" tests above would be RED
    del os.environ[MUT]
    with pytest.raises(BG.NavtestLeak):
        BG.assert_not_navtest(log, tok)


def test_fails_closed_on_altered_or_missing_lists(clean_env):
    d = tempfile.mkdtemp(prefix="a2gate_", dir=os.environ.get("A2_TMP", None))
    try:
        for f in os.listdir(os.path.join(A2, "gate_data")):
            shutil.copy(os.path.join(A2, "gate_data", f), d)
        os.environ["A2_GATE_DATA"] = d
        BG._CACHE.clear()
        BG.load()                                                 # intact copy loads
        with gzip.open(os.path.join(d, "navtest_tokens.txt.gz"), "wt") as fh:      # tamper: an EMPTY token list must not mean "pass everything"
            fh.write("")
        BG._CACHE.clear()
        with pytest.raises(BG.GateDataError):
            BG.assert_not_navtest("some_log", "0123456789abcdef")
        os.remove(os.path.join(d, "navtest_logs.txt"))
        BG._CACHE.clear()
        with pytest.raises(BG.GateDataError):
            BG.assert_not_navtest("some_log", "0123456789abcdef")
    finally:
        shutil.rmtree(d, ignore_errors=True)


class _FakePT:           # the worker's PT module stand-in: must never be reached for a navtest sample
    def executed_plans8(self, *a, **k):
        raise AssertionError("scoring reached for a navtest sample")

    score_plans = executed_plans8


def test_job_refuses_navtest_when_behaviour_on_and_is_unchanged_when_off(clean_env):
    import online_labeller as OL
    log, tok = _a_navtest_pair()
    props = np.zeros((4, 20, 3))
    meta = {"log_name": log, "token": tok}
    OL._W.clear()
    OL._W.update(PT=_FakePT(), root="X:/does_not_exist", lru={}, lru_n=2, behaviour=True)
    with pytest.raises(BG.NavtestLeak):
        OL._job("pdm", meta, props)
    os.environ[MUT] = BG.MUTATION                                  # regression arm: the gate is off -> the job proceeds past the gate (to the missing parts file)
    r = OL._job("pdm", meta, props)
    assert r["labels"] is None and r["reason"] == "no_parts"
    del os.environ[MUT]
    OL._W["behaviour"] = False                                     # behaviour OFF => the pre-A2 path, untouched (its own navtest guard is the trainer's launch gate)
    r = OL._job("pdm", meta, props)
    assert r["labels"] is None and r["reason"] == "no_parts"
    OL._W.clear()
