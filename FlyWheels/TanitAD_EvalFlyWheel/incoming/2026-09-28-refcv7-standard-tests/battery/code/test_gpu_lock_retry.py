"""Literal tests for the 2026-10-04 lock-gate fix (`gpu_gate.PROBE_ERRORS`, `gpu_lock.acquire`,
`final_waiter.py`'s start condition). No GPU, no nvidia-smi, no real lock file.

    PYTHONPATH="C:/Users/Admin/ev7/stack;C:/Users/Admin/ev7/taniteval" python -m pytest -q test_gpu_lock_retry.py

MEASURED failure this pins (`D:/refcv7_eval_kit/chain/chain_50400.log`, 2026-10-04 03:35 Berlin): ONE
`nvidia-smi` call timed out after 60 s under host memory pressure; `subprocess.TimeoutExpired` escaped
`gpu_gate.gate()` and `gpu_lock.acquire()`, the lock tool wrote no `--out`, and the chain ended
`ZZCHAINNOLOCK50400ZZ`. Every expectation is a LITERAL; the deliberate-regression arm restores the
unguarded probe and must see that crash again.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

TIMEOUT = subprocess.TimeoutExpired(
    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], 60)


def _flaky_facts(n_fail, healthy=(1000, [], 16.0)):
    calls = {"n": 0}

    def facts():
        calls["n"] += 1
        if calls["n"] <= n_fail:
            raise TIMEOUT
        return healthy
    return facts, calls


@pytest.fixture
def lockfile(tmp_path, monkeypatch):
    import gpu_lock
    p = tmp_path / "devbox_gpu.lock"
    monkeypatch.setattr(gpu_lock, "LOCK", str(p))
    return p


def test_gate_fails_closed_on_a_probe_timeout(monkeypatch):
    import gpu_gate
    facts, _ = _flaky_facts(1)
    monkeypatch.setattr(gpu_gate, "facts", facts)
    g = gpu_gate.gate(self_pid=123)
    assert g["ok"] is False
    assert g["probe_ok"] is False
    assert g["probe_error"].startswith("TimeoutExpired")
    assert g["gpu_used_mib"] is None
    g2 = gpu_gate.gate(self_pid=123)               # second call: the healthy facts
    assert g2["ok"] is True and g2["probe_ok"] is True and g2["gpu_used_mib"] == 1000


def test_smi_rule_is_fail_closed_on_a_failed_probe():
    import gpu_lock
    assert gpu_lock.smi_ok({"probe_ok": True, "gpu_used_mib": 1000, "python_compute": []}) is True
    assert gpu_lock.smi_ok({"probe_ok": False, "gpu_used_mib": None, "python_compute": None}) is False
    assert gpu_lock.smi_ok({"probe_ok": True, "gpu_used_mib": 4300, "python_compute": []}) is False
    assert gpu_lock.smi_ok({"probe_ok": True, "gpu_used_mib": 10,
                            "python_compute": [(7, "python.exe")]}) is False


def test_acquire_retries_through_three_probe_timeouts(monkeypatch, lockfile):
    import gpu_gate
    import gpu_lock
    facts, calls = _flaky_facts(3)
    monkeypatch.setattr(gpu_gate, "facts", facts)
    r = gpu_lock.acquire("job-x", 4242, max_wait_s=30, poll_s=0, log=lambda *a, **k: None)
    assert r["ok"] is True
    assert r["polls"] == 4
    assert r["probe_failures"] == 3
    assert r["last_probe_error"].startswith("TimeoutExpired")
    assert calls["n"] == 4
    held = json.loads(lockfile.read_text(encoding="utf-8"))
    assert held["job"] == "job-x" and held["pid"] == 4242


def test_acquire_never_takes_the_lock_while_the_probe_keeps_failing(monkeypatch, lockfile):
    import gpu_gate
    import gpu_lock
    facts, _ = _flaky_facts(10 ** 9)
    monkeypatch.setattr(gpu_gate, "facts", facts)
    r = gpu_lock.acquire("job-y", 1, max_wait_s=0.3, poll_s=0.05, log=lambda *a, **k: None)
    assert r["ok"] is False
    assert r["probe_failures"] >= 2
    assert not lockfile.exists()


def test_acquire_still_waits_on_a_held_lock(monkeypatch, lockfile):
    import gpu_gate
    import gpu_lock
    facts, _ = _flaky_facts(0)
    monkeypatch.setattr(gpu_gate, "facts", facts)
    lockfile.write_text(json.dumps({"job": "refcv7-navsim-navtest", "pid": 31600}), encoding="utf-8")
    r = gpu_lock.acquire("job-z", 2, max_wait_s=0.2, poll_s=0.05, log=lambda *a, **k: None)
    assert r["ok"] is False
    assert r["holder"]["job"] == "refcv7-navsim-navtest"
    assert json.loads(lockfile.read_text(encoding="utf-8"))["pid"] == 31600   # never broken


def test_REGRESSION_unguarded_probe_crashes_acquire(monkeypatch, lockfile):
    """DELIBERATE REGRESSION: the guard removed (an empty tuple catches nothing) must reproduce the
    03:35 crash -- so the retry test above has power."""
    import gpu_gate
    import gpu_lock
    facts, _ = _flaky_facts(3)
    monkeypatch.setattr(gpu_gate, "facts", facts)
    monkeypatch.setattr(gpu_gate, "PROBE_ERRORS", ())
    with pytest.raises(subprocess.TimeoutExpired):
        gpu_lock.acquire("job-r", 3, max_wait_s=30, poll_s=0, log=lambda *a, **k: None)
    assert not lockfile.exists()


def test_cli_writes_out_even_when_acquire_raises(monkeypatch, tmp_path, lockfile):
    """The `--out` artifact exists on every path (the chain reads it, never `$?`)."""
    import runpy
    import gpu_gate
    facts, _ = _flaky_facts(10 ** 9)
    monkeypatch.setattr(gpu_gate, "facts", facts)
    monkeypatch.setattr(gpu_gate, "PROBE_ERRORS", ())          # force an UNEXPECTED raise
    out = tmp_path / "lock_out.json"
    monkeypatch.setenv("REFCV7_GPU_LOCK", str(lockfile))
    monkeypatch.setattr(sys, "argv", ["gpu_lock.py", "acquire", "--job", "j", "--pid", "9",
                                      "--max-wait-s", "5", "--out", str(out)])
    with pytest.raises(SystemExit) as ex:
        runpy.run_path(str(Path(__file__).resolve().parent / "gpu_lock.py"), run_name="__main__")
    assert ex.value.code == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["ok"] is False and r["error"].startswith("TimeoutExpired")
    assert not lockfile.exists()


# ------------------------------------------------------------------------------------------- #
# final_waiter.py -- the start condition for the step-50,400 chain                              #
# ------------------------------------------------------------------------------------------- #
def test_waiter_condition_literals():
    import final_waiter as W
    ok = dict(lock=None, free_disk_gb=10.0, free_commit_gb=6.0, min_disk_gb=10.0, min_commit_gb=6.0)
    assert W.decide(**ok)["go"] is True
    assert W.decide(**{**ok, "lock": {"job": "refcv7-navsim-navtest"}})["go"] is False
    assert W.decide(**{**ok, "free_disk_gb": 9.99})["go"] is False
    assert W.decide(**{**ok, "free_commit_gb": 5.99})["go"] is False
    # a probe that could not be read is NOT a pass (fail closed)
    assert W.decide(**{**ok, "free_disk_gb": None})["go"] is False
    assert W.decide(**{**ok, "free_commit_gb": None})["go"] is False
    assert W.decide(**{**ok, "lock": {"unreadable": "PermissionError"}})["go"] is False
    why = W.decide(**{**ok, "free_disk_gb": 7.7, "lock": {"job": "x"}})["why"]
    assert any("lock held" in w for w in why) and any("disk" in w for w in why)
