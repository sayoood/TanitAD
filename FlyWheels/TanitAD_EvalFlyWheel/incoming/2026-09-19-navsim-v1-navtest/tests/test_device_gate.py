"""The suite device gate for model inference (orchestrator arbitration 2026-09-20), as W3 applies it.

    PYTHONPATH=D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval \
      C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q tests/test_device_gate.py

Rules under test — literals, one arm per branch, and the REFUSAL arm is the one that matters:
  auto|cuda                          -> GPU_GAP (the launcher waits)
  cpu, no training alive             -> CPU_ALLOWED
  cpu, training alive, no override   -> REFUSED_TRAINING_ALIVE  (must NOT be allowed)
  cpu, training alive, override      -> CPU_ACCEPTED_TRAINING_BOX_LOAD, and the accepted process
                                        is RECORDED (that record is what lands in bench_run.json)
The probe itself is W1's ``taniteval.bench.gpu_gap``; these tests monkeypatch it so the box's real
state cannot make the test pass or fail by accident.
"""
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO / "taniteval")]

from taniteval.bench import gpu_gap as GG                              # noqa: E402
from taniteval.bench.plugins import navsim_v1 as P                     # noqa: E402


def _args(device="cpu", accept=False):
    return types.SimpleNamespace(device=device, accept_training_box_load=accept)


@pytest.fixture
def _no_shared_gate(monkeypatch):
    """Exercise W3's FALLBACK composition of W1's primitives (the route used before W1 exported
    `device_gate`, and the one that runs if that export is ever withdrawn)."""
    for nm in ("device_gate", "cpu_gate", "check_cpu_box", "acquire_cpu", "cpu_box_gate"):
        monkeypatch.delattr(GG, nm, raising=False)
    monkeypatch.delenv("W3_ACCEPT_TRAINING_BOX_LOAD", raising=False)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("W3_ACCEPT_TRAINING_BOX_LOAD", raising=False)


def test_auto_and_cuda_wait_for_a_gap(monkeypatch, _no_shared_gate):
    monkeypatch.setattr(GG, "probe_training_pids", lambda *a, **k: [4242])
    for d in ("auto", "cuda"):
        g = P.device_decision(_args(d), log=lambda *a: None)
        assert g["decision"] == "GPU_GAP" and g["allowed"] is True and g["requested"] == d


def test_cpu_allowed_when_no_training(monkeypatch, _no_shared_gate):
    monkeypatch.setattr(GG, "probe_training_pids", lambda *a, **k: [])
    g = P.device_decision(_args("cpu"), log=lambda *a: None)
    assert g["decision"] == "CPU_ALLOWED" and g["allowed"] is True
    assert g["training_processes"] == []


def test_cpu_refused_when_training_alive(monkeypatch, _no_shared_gate):
    monkeypatch.setattr(GG, "probe_training_pids", lambda *a, **k: [4242])
    g = P.device_decision(_args("cpu"), log=lambda *a: None)
    assert g["decision"] == "REFUSED_TRAINING_ALIVE"
    assert g["allowed"] is False and "4242" in g["reason"]
    assert g["accept_training_box_load"] is False


def test_cpu_allowed_with_the_named_override_and_the_process_is_recorded(monkeypatch, _no_shared_gate):
    monkeypatch.setattr(GG, "probe_training_pids", lambda *a, **k: [4242])
    g = P.device_decision(_args("cpu", accept=True), log=lambda *a: None)
    assert g["decision"] == "CPU_ACCEPTED_TRAINING_BOX_LOAD" and g["allowed"] is True
    assert g["accept_via"] == "--accept-training-box-load"
    assert [p["pid"] for p in g["training_processes"]] == [4242]


def test_env_override_is_recorded_as_such(monkeypatch, _no_shared_gate):
    monkeypatch.setattr(GG, "probe_training_pids", lambda *a, **k: [7])
    monkeypatch.setenv("W3_ACCEPT_TRAINING_BOX_LOAD", "1")
    g = P.device_decision(_args("cpu"), log=lambda *a: None)
    assert g["allowed"] is True and g["accept_via"] == "env W3_ACCEPT_TRAINING_BOX_LOAD=1"


def test_w1s_shared_gate_is_called_with_ITS_signature(monkeypatch):
    """W1's gate takes the device STRING (+ kwargs), not the args namespace. Passing the namespace
    raised ValueError — MEASURED 2026-09-20 — so the call shape is pinned here."""
    seen = {}

    def fake(requested, *, accept_training_box_load=False, queue_hint="", **kw):
        seen.update(requested=requested, accept=accept_training_box_load, hint=queue_hint)
        return {"decision": "CPU", "reason": "stub", "record": {"override_used": False}}

    monkeypatch.setattr(GG, "device_gate", fake, raising=False)
    g = P.device_decision(_args("cpu", accept=True), log=lambda *a: None)
    assert seen["requested"] == "cpu" and seen["accept"] is True and seen["hint"]
    assert g["decision"] == "CPU" and g["allowed"] is True          # W1's vocabulary, mapped
    assert g["gate_source"] == "taniteval.bench.gpu_gap.device_gate"
    assert g["record"] == {"override_used": False}                  # W1's record is preserved


def test_the_real_w1_gate_answers_without_raising():
    """The live wiring: whatever this box's state, the shared gate must answer in W1's vocabulary.
    (This is the test that would have caught the namespace-vs-string call error.)"""
    if not hasattr(GG, "device_gate"):
        pytest.skip("W1 has not exported the shared gate yet")
    for dev in ("auto", "cpu", "cuda"):
        g = P.device_decision(_args(dev), log=lambda *a: None)
        assert g["decision"] in ("CPU", "CUDA", "REFUSE")
        assert isinstance(g["allowed"], bool)
        assert g["allowed"] == (g["decision"] in ("CPU", "CUDA"))


def test_cuda_with_the_override_still_refuses_the_card(monkeypatch):
    """The override accepts CPU load only, never the card (orchestrator FYI 2026-09-20)."""
    def fake(requested, *, accept_training_box_load=False, queue_hint="", **kw):
        if requested in ("auto", "cuda"):
            return {"decision": "REFUSE", "reason": "no gap", "record": {}}
        return {"decision": "CPU", "reason": "cpu ok", "record": {}}

    monkeypatch.setattr(GG, "device_gate", fake, raising=False)
    g = P.device_decision(_args("cuda", accept=True), log=lambda *a: None)
    assert g["decision"] == "REFUSE" and g["allowed"] is False


# --- the reuse key + the re-check (ruling (a) and (b)) ----------------------------- #
PKG = Path(__file__).resolve().parents[1]


def test_recheck_accepts_a_real_smoke_csv_and_counts_its_summary_row():
    csv = PKG / "raw" / "CV_smoke20" / "CV_smoke20.csv"
    if not csv.exists():
        pytest.skip("smoke CSV absent")
    toks = {r for r in _tokens(csv)}
    rc = P.recheck_csv(csv, toks)
    assert rc["ok"] is True
    assert rc["n_token_rows"] == rc["n_valid"] == 20 and rc["n_summary_rows"] == 1
    assert rc["C1_pdms_formula_max_abs_delta"] <= 1e-12 and len(rc["sha256"]) == 64


def test_recheck_rejects_a_truncated_csv(tmp_path):
    csv = PKG / "raw" / "CV_smoke20" / "CV_smoke20.csv"
    if not csv.exists():
        pytest.skip("smoke CSV absent")
    toks = set(_tokens(csv))
    lines = csv.read_text(encoding="utf-8").splitlines()
    bad = tmp_path / "truncated.csv"
    bad.write_text("\n".join(lines[:-3]) + "\n", encoding="utf-8")     # loses rows + the average
    rc = P.recheck_csv(bad, toks)
    assert rc["ok"] is False and (rc["n_missing"] > 0 or rc["n_summary_rows"] != 1)


def test_the_reuse_key_names_the_agent_class_per_arm():
    assert P.AGENT_CLASS["CV"].endswith("ConstantVelocityAgent")
    assert P.AGENT_CLASS["HUMAN"].endswith("HumanAgent")
    assert P.AGENT_CLASS["STOP"] == "w3_agents_v1.StopAgent"
    assert len(set(P.AGENT_CLASS.values())) == 3          # no arm can inherit another's CSV


def _tokens(csv):
    import csv as _csv
    return [r["token"] for r in _csv.DictReader(open(csv, encoding="utf-8"))
            if r["token"] != "average"]
