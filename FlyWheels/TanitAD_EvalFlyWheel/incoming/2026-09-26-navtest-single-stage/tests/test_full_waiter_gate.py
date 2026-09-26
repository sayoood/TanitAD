"""The full-split floors' START GATE (``code/navtest_full_after_final.py``) — rewired 2026-09-26 after the PI stopped
refcv6, and fixed the same evening for two defects the rewire exposed.

1. ⛔ **The queue check could not see the queue.** It looked for ``run_battery.py``; tonight's battery runs as
   ``a6_chain.sh -> a6_roll.py`` and never names that file, and navhard@30k (which the Master Mind said must
   FINISH first) never matched either. MEASURED 2026-09-26 ~20:50 Berlin on the live box: the old check read
   ``battery_pids: []`` while A6 held the GPU; the new one read exactly the six live queue-ahead processes.
2. ⛔ **A RAM wait that never looked back.** With FINAL-ended as the gate, "check once, then wait on RAM" was
   sound — FINAL-ended is a LATCH. The GPU gate and the queue are not; and the old RAM wait also started the
   run when its own 24 h wait had FAILED.

Every command line below is a LITERAL copied from the live process table, never built from the code under test.
"""
from __future__ import annotations

import importlib.util
import json
import re
import time
from pathlib import Path

import pytest

WAITER = Path(__file__).resolve().parents[1] / "code" / "navtest_full_after_final.py"


@pytest.fixture(scope="module")
def W():
    spec = importlib.util.spec_from_file_location("w8_full_waiter", WAITER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert Path(mod.__file__).resolve() == WAITER          # the file under test, not a stale copy elsewhere
    return mod


# ---------------------------------------------------------------- literal command lines (live table, 2026-09-26)
A6_ROLL = ("C:\\Users\\Admin\\venvs\\tanitad\\Scripts\\python.exe a6_roll.py --ckpt D:/refcv6_eval_kit/ckpt/ckpt_5000.pt "
           "--config D:/refcv6_eval_kit/ckpt/config.json --seed 0 --episodes D:/refcv6_eval_kit/data/"
           "refcv6-b1-416x1024-train-a6 --labels D:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz "
           "--dump-dir C:/Users/Admin/ev6_battery/raw/a6/step5000/dump_s0 "
           "--out-json C:/Users/Admin/ev6_battery/raw/a6/step5000/roll_s0.json")
GATE_WAIT = ("C:/Users/Admin/venvs/tanitad/Scripts/python.exe -c \"import sys; sys.path.insert(0, "
             "r'C:/Users/Admin/ev6_battery/code'); import run_battery as RB; "
             "RB.gate_wait(r'C:/Users/Admin/ev6_battery/raw/a6/step5000/gate_s1.json')\"")
NAVHARD = ("C:\\Users\\Admin\\venvs\\tanitad\\Scripts\\python.exe C:/Users/Admin/ev6/FlyWheels/TanitAD_EvalFlyWheel/"
           "incoming/2026-09-23-refcv6-standard-tests/navsim/code/run_navsim_refcv6.py --fetch 30000 --splits navhard "
           "--device auto --gpu-wait-s 900")
SCORER = ("C:/Users/Admin/navsim-crun/venv/Scripts/python.exe D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/"
          "incoming/2026-09-19-navsim-warmup-reference-epdms/code/navsim_win.py --script pdm_score "
          "--label R6_A1__navhard_two_stage")
# ⛔ must NEVER hold the floors
SELF = ("C:\\Users\\Admin\\venvs\\tanitad\\Scripts\\python.exe ../FlyWheels/TanitAD_EvalFlyWheel/incoming/"
        "2026-09-26-navtest-single-stage/code/navtest_full_after_final.py")
GPU_GATE_SCRIPT = ("C:/Users/Admin/venvs/tanitad/Scripts/python.exe D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/"
                   "incoming/2026-09-23-refcv6-standard-tests/battery/code/gpu_gate.py")
REFE = ("C:\\Users\\Admin\\venvs\\driverl-eval\\Scripts\\python.exe eval_checkpoint.py --ckpt "
        "D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch013.pt --name a5confirm_ep013 "
        "--tokens D:/Projects/TanitAD/data/refe_navtest/a5_confirm_tokens.json")
HFPUSH = "C:\\Users\\Admin\\venvs\\tanitad\\Scripts\\python.exe hfpush_refcv6.py"


# ---------------------------------------------------------------- the classifier sees the queue
@pytest.mark.parametrize("cmd,label", [
    (A6_ROLL, "battery (ev6_battery chain)"),
    (GATE_WAIT, "battery (run_battery)"),
    (NAVHARD, "navhard@30k (run_navsim_refcv6)"),
    (SCORER, "NavSim scorer (navsim_win.py)"),
])
def test_every_queued_job_holds_the_floors(W, cmd, label):
    got = W.classify_queue_ahead([(101, cmd)])
    assert [(x["pid"], x["what"]) for x in got] == [(101, label)]


@pytest.mark.parametrize("cmd", [SELF, GPU_GATE_SCRIPT, REFE, HFPUSH, "", None])
def test_nothing_else_holds_the_floors(W, cmd):
    """The gate script's own path contains 'battery' — a pattern broad enough to match it would block forever."""
    assert W.classify_queue_ahead([(202, cmd)]) == []


def test_own_pids_are_never_queue_ahead(W):
    assert W.classify_queue_ahead([(7, NAVHARD), (8, A6_ROLL)], self_pids={7, 8}) == []


def test_MUTATION_the_old_rule_is_blind_to_tonights_battery_and_navhard(W, monkeypatch):
    """Reintroduce the pre-fix rule: it must MISS A6 and navhard@30k. If it did not, the positive tests above
    would not be evidence that the new patterns added anything."""
    monkeypatch.setattr(W, "QUEUE_AHEAD", (("old", re.compile(r"run_battery\.py")),))
    assert W.classify_queue_ahead([(1, A6_ROLL), (2, NAVHARD), (3, SCORER)]) == []


def test_MUTATION_a_too_broad_battery_pattern_would_match_the_gate_itself(W, monkeypatch):
    monkeypatch.setattr(W, "QUEUE_AHEAD", (("broad", re.compile(r"battery")),))
    assert [x["pid"] for x in W.classify_queue_ahead([(9, GPU_GATE_SCRIPT)])] == [9]


# ---------------------------------------------------------------- the process table fails CLOSED
def test_an_unreadable_table_is_a_wait(W, monkeypatch):
    def boom():
        raise RuntimeError("process query rc 1")
    monkeypatch.setattr(W, "python_processes", boom)
    q = W.queue_ahead()
    assert len(q) == 1 and q[0]["pid"] == -1 and "unreadable" in q[0]["what"]


def test_a_table_missing_this_process_is_a_wait(W, monkeypatch):
    """Positive control: the waiter is itself a python process, so an EMPTY table was not read."""
    monkeypatch.setattr(W, "python_processes", lambda: [])
    q = W.queue_ahead()
    assert len(q) == 1 and q[0]["pid"] == -1 and "own pid absent" in q[0]["what"]


def test_a_table_with_only_this_process_is_clear(W, monkeypatch):
    import os
    monkeypatch.setattr(W, "python_processes", lambda: [(os.getpid(), SELF)])
    assert W.queue_ahead() == []


# ---------------------------------------------------------------- every gate is re-read before a start
class _Args:
    start_avail_mb, poll_s, max_wait_h = 11500.0, 0.0, 72.0


def _gate(ok: bool, pid: int | None = None) -> dict:
    q = [] if pid is None else [{"pid": pid, "what": "battery (ev6_battery chain)", "cmd": "x"}]
    return {"cache_done": {"ok": True}, "gpu_gate": {"ok": True}, "queue_ahead": q, "ok": ok}


@pytest.fixture
def quiet_env(W, monkeypatch, tmp_path):
    monkeypatch.setattr(W, "LOG", tmp_path / "waiter.log")
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    return W


def test_a_job_that_starts_during_the_RAM_window_blocks_the_start(quiet_env, monkeypatch):
    """gates OK -> RAM window OK -> the battery re-appeared -> must NOT start; it starts only once all hold again."""
    W = quiet_env
    seq = iter([_gate(True), _gate(False, 15708), _gate(True), _gate(True)])
    calls = []
    monkeypatch.setattr(W, "gates", lambda prof: calls.append(1) or next(seq))
    monkeypatch.setattr(W.CB, "wait_for_ram", lambda *a, **k: {"ok": True})
    rec = W.wait_until_quiet(_Args(), None, time.time())
    assert rec is not None and rec["queue_ahead"] == [] and len(calls) == 4


def test_a_failed_RAM_wait_never_starts_a_run(quiet_env, monkeypatch):
    """The old loop ran even when its 24 h RAM wait returned ok=False."""
    W = quiet_env
    monkeypatch.setattr(W, "gates", lambda prof: _gate(True))
    monkeypatch.setattr(W.CB, "wait_for_ram", lambda *a, **k: {"ok": False})
    a = _Args()
    a.max_wait_h = 0.0
    assert W.wait_until_quiet(a, None, time.time() - 1.0) is None


def test_main_gives_up_without_running_when_the_box_never_quiets(quiet_env, monkeypatch, tmp_path):
    W = quiet_env
    (tmp_path / "raw").mkdir()
    monkeypatch.setattr(W, "PKG", tmp_path)
    monkeypatch.setattr(W, "wait_until_quiet", lambda a, prof, t0: None)
    def must_not_run(*a, **k):
        raise AssertionError("the floors started on a box that never quieted")
    monkeypatch.setattr(W, "run_cli", must_not_run)
    assert W.main([]) == 2
    rec = json.loads((tmp_path / "raw" / "full_waiter.json").read_text(encoding="utf-8"))
    assert rec["status"] == "NOT_RUN" and "first run" in rec["reason"]
    assert "ZZW8GAVEUPZZ" in (tmp_path / "waiter.log").read_text(encoding="utf-8")
