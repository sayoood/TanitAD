"""The GPU-GAP LAUNCHER (W1) — PI 2026-09-19 *"wait for a gap in the gpu"*.

⚠️ UNVERIFIED ON HARDWARE: the GPU belongs to training (A8, then the S1 pass / A7), and this
package is CPU-only by brief, so every test here injects the probes. What IS pinned: the predicate's
literals, the auto/cpu/cuda decisions, the wait loop, the back-off, and a RED mutation per rule.
"""
from __future__ import annotations

import pytest

from taniteval.bench import gpu_gap as G


@pytest.mark.parametrize("pids,mem,expect,why", [
    ([], 0.0, True, None),
    ([], 1023.9, True, None),
    ([], 1024.0, False, "memory"),                     # "< 1 GB" — 1024 MiB is NOT a gap
    ([], 6163.0, False, "memory"),                     # MEASURED while A8 trained
    ([16996], 0.0, False, "training"),                 # A8's PID
    ([16996], 5000.0, False, "training"),
    ([], None, False, "UNKNOWN"),                      # nvidia-smi gave no reading
])
def test_gap_predicate_literals(pids, mem, expect, why):
    v = G.gap_verdict(pids, mem, limit_mib=1024)
    assert v["gap"] is expect
    if why:
        assert any(why.lower() in r.lower() for r in v["reasons"]), v


def test_no_cuda_no_gap():
    assert G.gap_verdict([], 0.0, cuda_available=False)["gap"] is False


@pytest.mark.parametrize("cmdline,is_training", [
    (["python.exe", "-u", "scripts/refc_v3_train.py", "--arm", "hier"], True),   # A8, MEASURED
    (["python", "train_v6_staged.py"], True),
    (["python", "-m", "torch.distributed.run", "x.py"], True),
    (["python", "scripts/eval_flagship_v4.py"], False),
    (["python", "-m", "taniteval.bench", "navsim_v2"], False),
])
def test_training_process_detection(cmdline, is_training):
    assert G.is_training_cmdline(cmdline) is is_training


def _launcher(seq, requested="auto", **kw):
    """``seq`` = list of (training_pids, mem_used) returned by successive probes."""
    box = {"i": 0}

    def probe():
        v = seq[min(box["i"], len(seq) - 1)]
        box["i"] += 1
        return v
    return G.GpuGapLauncher(requested=requested, probe=probe, cuda_available=lambda: True,
                            sleep=lambda s: None, clock=lambda: 0.0 + box["i"], log=lambda m: None, **kw)


def test_auto_takes_the_gap_when_there_is_one():
    g = _launcher([([], 12.0)])
    assert g.acquire() == "cuda"
    assert g.record()["used"] == "cuda"
    g.stop()


def test_auto_stays_on_cpu_when_training_is_alive():
    g = _launcher([([21724], 6163.0)])
    assert g.acquire() == "cpu"
    ev = [e["event"] for e in g.events]
    assert "NO_GAP_CPU" in ev and "GAP_ACQUIRED" not in ev


def test_cpu_requested_never_probes():
    def boom():
        raise AssertionError("probed although --device cpu")
    g = G.GpuGapLauncher("cpu", probe=boom, cuda_available=lambda: True, log=lambda m: None)
    assert g.acquire() == "cpu"


def test_cuda_waits_for_a_gap_then_takes_it():
    g = _launcher([([21724], 6163.0), ([21724], 6163.0), ([], 200.0)], requested="cuda", interval_s=0.0)
    assert g.acquire() == "cuda"
    assert sum(e["event"] == "WAITING_FOR_GAP" for e in g.events) == 2
    g.stop()


def test_cuda_wait_times_out_to_cpu():
    g = _launcher([([21724], 6163.0)], requested="cuda", interval_s=0.0, max_wait_s=2.0)
    assert g.acquire() == "cpu"
    assert any(e["event"] == "GAP_WAIT_TIMEOUT" for e in g.events)


def test_watchdog_backs_off_when_training_appears():
    import time as _t
    g = _launcher([([], 10.0), ([21724], 600.0)], interval_s=0.005)
    assert g.acquire() == "cuda"
    deadline = _t.monotonic() + 5.0
    while _t.monotonic() < deadline and not g.should_back_off():
        _t.sleep(0.01)
    assert g.should_back_off() is True, [e["event"] for e in g.events]
    g.backed_off()
    assert g.device == "cpu" and g.record()["used"] == "mixed"


def test_own_usage_is_not_read_as_foreign_load():
    """While WE hold the card our own memory must not trip the limit; foreign load must."""
    g = _launcher([([], 100.0)], interval_s=1e6)
    assert g.acquire() == "cuda"
    g.used_before_mib, g.own_mib = 100.0, 4000.0
    assert g.check(own_mib=g.own_mib)["gap"] is True       # 100 + 4000 ours -> foreign 0
    g._probe = lambda: ([], 100.0 + 4000.0 + 1500.0)       # a foreign job takes 1.5 GB
    assert g.check(own_mib=g.own_mib)["gap"] is False
    g.stop()


def test_deliberate_regression_inverted_comparison_goes_red():
    """M-GAP: flip the memory comparison -> the literal table above must fail."""
    def mutant(training_pids, mem_used_mib, *, limit_mib=1024, cuda_available=True):
        reasons = []
        if training_pids:
            reasons.append("training")
        if mem_used_mib is not None and mem_used_mib < limit_mib:     # INVERTED
            reasons.append("memory")
        return {"gap": not reasons, "reasons": reasons}
    assert mutant([], 6163.0)["gap"] is True                           # would pass the gap while A8 trains
    assert G.gap_verdict([], 6163.0)["gap"] is False                   # the real predicate refuses


# --------------------------------------------------------------------------- #
# THE SHARED DEVICE GATE (orchestrator arbitration, 2026-09-20)                #
# --------------------------------------------------------------------------- #
def _policy(requested, pids, mem, accept=False):
    return G.device_policy(requested, accept_training_box_load=accept,
                           probe=lambda: (pids, mem), cuda_available=lambda: True,
                           queue_hint="python -m taniteval.bench … --device cpu")


@pytest.mark.parametrize("requested,pids,mem,accept,decision", [
    ("auto", [], 10.0, False, "CUDA"),          # a gap is open -> take it
    ("cuda", [], 10.0, False, "CUDA"),
    ("auto", [21724], 6163.0, False, "REFUSE"),  # model arms WAIT for a gap (never silently CPU)
    ("auto", [], 4000.0, False, "REFUSE"),       # no training, but the card is busy
    ("cuda", [21724], 6163.0, False, "REFUSE"),
    ("cpu", [], 6163.0, False, "CPU"),           # explicit CPU, nothing training -> allowed
    ("cpu", [21724], 10.0, False, "REFUSE"),     # explicit CPU beside a trainer -> refused
    ("cpu", [21724], 10.0, True, "CPU"),         # … unless the operator accepts it, on the record
])
def test_shared_device_gate_literals(requested, pids, mem, accept, decision):
    p = _policy(requested, pids, mem, accept)
    assert p["decision"] == decision, p["reason"]
    assert p["record"]["requested"] == requested and p["record"]["training_pids"] == sorted(pids)


def test_the_override_and_the_accepted_pids_are_recorded():
    p = _policy("cpu", [21724, 16996], 10.0, accept=True)
    assert p["decision"] == "CPU"
    assert p["record"]["override_used"] is True
    assert p["record"]["training_pids"] == [16996, 21724]
    assert G.OVERRIDE_FLAG == "--accept-training-box-load" and G.OVERRIDE_FLAG in p["record"]["override_flag"]


def test_refusals_carry_the_queue_command():
    for req, pids in (("auto", [21724]), ("cpu", [21724])):
        p = _policy(req, pids, 6163.0)
        assert p["decision"] == "REFUSE" and "--device cpu" in p["reason"]
    assert G.OVERRIDE_FLAG in _policy("cpu", [21724], 10.0)["reason"]


def test_override_does_not_open_the_gpu():
    """⛔ The override accepts CPU load only — it can never take the card from a trainer."""
    p = _policy("cuda", [21724], 10.0, accept=True)
    assert p["decision"] == "REFUSE"


def test_deliberate_regression_gate_that_ignores_training_goes_red():
    def mutant(requested, *, accept_training_box_load=False, **kw):
        return {"decision": "CPU", "reason": "sure", "record": {}}
    assert mutant("cpu")["decision"] == "CPU"
    assert _policy("cpu", [21724], 10.0)["decision"] == "REFUSE"


def test_the_named_shared_gate_is_exported_for_w3():
    """W3's plugin probes for a named gate on taniteval.bench.gpu_gap and calls it if present."""
    assert G.device_gate is G.device_policy
    p = G.device_gate("cpu", probe=lambda: ([], 0.0), cuda_available=lambda: True)
    assert p["decision"] == "CPU"


def test_the_record_carries_pid_and_cmdline_of_the_accepted_trainer():
    procs = [{"pid": 21724, "name": "python.exe", "cmdline": "python -u scripts/refc_v3_train.py --arm hier"}]
    p = G.device_policy("cpu", accept_training_box_load=True, probe=lambda: ([21724], 10.0),
                        cuda_available=lambda: True, processes=procs)
    assert p["decision"] == "CPU"
    r = p["record"]
    assert r["override_used"] is True and r["override_passed_as"].endswith(G.OVERRIDE_FLAG)
    assert r["training_processes"] == procs and r["training_pids"] == [21724]
    assert "TRAINING_SCRIPT_RE" in r["training_process_definition"]
