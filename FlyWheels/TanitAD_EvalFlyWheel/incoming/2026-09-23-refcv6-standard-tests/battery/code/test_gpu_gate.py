"""Regression tests for the dev-box GPU gate (2026-09-24 self-deadlock). Run: python -m pytest -q test_gpu_gate.py

The defect: run_battery's `gate_wait()` called the gate WITHOUT the caller's pid while the caller held
a CUDA context, so it waited on itself. The expected values below are LITERALS (never an expression
over the code under test), and the mutation arm reproduces the historical defect and must read NOT ok.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_gate as G  # noqa: E402

SELF = 30624
OTHER = 29244
PY = r"C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe"
EXPLORER = r"C:\Windows\explorer.exe"


def test_own_pid_never_blocks_the_gate():
    g = G.evaluate(1826, [(SELF, PY), (9164, EXPLORER)], 12.0, self_pids=[SELF])
    assert g["ok"] is True
    assert g["python_compute"] == []


def test_mutation_the_historical_defect_blocks_on_itself():
    # the 2026-09-24 state: no self pid passed -> the caller's own context blocks it
    g = G.evaluate(1826, [(SELF, PY)], 12.0, self_pids=[])
    assert g["ok"] is False
    assert g["python_compute"] == [(SELF, PY)]


def test_another_python_still_blocks():
    g = G.evaluate(1826, [(SELF, PY), (OTHER, PY)], 12.0, self_pids=[SELF])
    assert g["ok"] is False
    assert g["python_compute"] == [(OTHER, PY)]


def test_ram_and_memory_rules_unchanged():
    assert G.evaluate(1826, [], 7.99, self_pids=[SELF])["ok"] is False      # the 8 GB rule stands
    assert G.evaluate(4300, [], 12.0, self_pids=[SELF])["ok"] is False      # used < 4300 MiB
    assert G.evaluate(4299, [], 8.0, self_pids=[SELF])["ok"] is True


def test_gate_excludes_the_calling_process_by_default(monkeypatch):
    monkeypatch.setattr(G, "facts", lambda: (1826, [(os.getpid(), PY)], 12.0))
    g = G.gate()
    assert g["ok"] is True
    assert os.getpid() in g["self_pids_excluded"]


def test_run_battery_waits_with_its_own_pid_and_rolls_in_a_subprocess():
    src = (Path(__file__).resolve().parent / "run_battery.py").read_text(encoding="utf-8")
    assert "gpu_gate.gate(self_pid=os.getpid())" in src
    assert "roll_seed.py" in src               # every seed roll runs in a child process
