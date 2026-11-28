"""The dev-box GPU lock protocol (any python, no GPU): never touch a foreign lock; reap only THIS
suite's lock whose pid is dead; release only with the owner's token.

MEASURED 2026-09-28: a runner killed by PID seconds after acquiring left its lock for 42 min and
blocked the battery and the Research Lab -- the reaping rule exists for that, and the foreign-lock
rule (Master Mind 06:45: the Research Lab and the battery share the SAME file) bounds it.
"""
from __future__ import annotations

import importlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))


@pytest.fixture()
def gl(tmp_path, monkeypatch):
    monkeypatch.setenv("R7_GPU_LOCK", str(tmp_path / "devbox_gpu.lock"))
    import gpu_lock
    return importlib.reload(gpu_lock)


def _write(gl, rec):
    with open(gl.LOCK, "w", encoding="utf-8") as fh:
        json.dump(rec, fh)


def _dead_pid() -> int:
    import psutil
    p = 999999
    while psutil.pid_exists(p):
        p += 1
    return p


def test_own_stale_lock_is_reaped(gl):
    _write(gl, {"job": "refcv7-navsim-warmup", "pid": _dead_pid(), "token": "t1"})
    assert gl.reap_own_stale(lambda m: None) is True and gl.read() is None


def test_own_live_lock_is_kept(gl):
    _write(gl, {"job": "refcv7-navsim-warmup", "pid": os.getpid(), "token": "t2"})
    assert gl.reap_own_stale(lambda m: None) is False and gl.read()["token"] == "t2"


@pytest.mark.parametrize("job", ["refcv7-g0-5000", "refcv6-38k-perc-dump", "research-refe-eval"])
def test_foreign_lock_is_never_touched_even_with_a_dead_pid(gl, job):
    rec = {"job": job, "pid": _dead_pid(), "win_pid_of_locker": 1, "host": "x", "acquired": "t"}
    _write(gl, rec)
    assert gl.reap_own_stale(lambda m: None) is False
    assert gl.release("anything") is False
    assert gl.read() == rec


def test_release_needs_the_owners_token(gl):
    _write(gl, {"job": "refcv7-navsim-navtest", "pid": os.getpid(), "token": "mine"})
    assert gl.release("not-mine") is False and gl.read() is not None
    assert gl.release("mine") is True and gl.read() is None
