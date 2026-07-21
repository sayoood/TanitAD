"""Falsifiers for tools/fleet_probe.py.

The property under test is not "does it print a table" — it is the one rule the
old monitor broke four times:

    **absence of evidence must never score GREEN.**

So most tests here construct a host that *looks* fine to a naive checker (ssh
answered, a process exists, the GPU is warm) and assert the probe still raises
the alarm. Two tests are deliberately inverted — they prove the checks CAN pass,
so a green run means something.

Fixtures are real captures from the 2026-07-21 live fleet, trimmed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fleet_probe as fp                                   # noqa: E402

NOW = 1_784_658_100.0

# --- real captures (trimmed) ----------------------------------------------
PS_POD1 = [
    "111 1 1286966 90000 /usr/bin/python /usr/local/bin/jupyter-lab --allow-root",
    "1400645 900 9635 4000 bash -c cd /workspace/TanitAD/stack && PYTHONPATH=/workspace/TanitAD/stack nohup python3 scripts/train_flagship4b.py --config flagship4b --out /workspace/experiments/flagship4b-v3enc-expA-nodrop-2k > /tmp/expA_nodrop_2k.out 2>&1 & echo LAUNCHED-PID=$!",
    "1400646 1400645 9635 8000000 python3 scripts/train_flagship4b.py --config flagship4b --out /workspace/experiments/flagship4b-v3enc-expA-nodrop-2k",
    "1400779 1400646 9628 500000 python3 scripts/train_flagship4b.py --config flagship4b --out /workspace/experiments/flagship4b-v3enc-expA-nodrop-2k",
]
PS_POD3_VLM = [
    "40783 1 34775 3000 bash -c cd /root/vlmprod && nohup bash chain5.sh > /root/vlmprod/train_queue.log 2>&1 & echo queued",
    "53351 40783 18572 9000000 /workspace/venv/bin/python vlm_semantic_labels.py --val /workspace/pai_epcache/x --out /root/vlmprod/trainstrat --tag train_strat",
]
LOGS_POD1 = [f"{NOW - 8}|39689|/tmp/expA_nodrop_2k.out",
             f"{NOW - 20000}|409844|/tmp/flagship_v3enc.log"]

GPU_BUSY = [{"index": 0, "name": "A6000", "util": 64,
             "mem_used": 14862, "mem_total": 49140}]
GPU_COLD = [{"index": 0, "name": "A40", "util": 0,
             "mem_used": 0, "mem_total": 46068}]
GPU_ORPHAN = [{"index": 0, "name": "A40", "util": 0,
               "mem_used": 18729, "mem_total": 46068}]


def _host(role="train", **kw) -> fp.HostReport:
    rep = fp.HostReport(name="pX", ssh="x", role=role, reachable=True)
    for k, v in kw.items():
        setattr(rep, k, v)
    return rep


def _verdict(rep, prev=None, gap=None):
    fp.judge(rep, prev or {}, NOW, gap)
    return rep.verdict


# --- discovery -------------------------------------------------------------
def test_discovers_trainer_without_any_hardcoded_name():
    """The whole point: no run name, log name or script name is configured."""
    jobs = fp.discover_jobs(fp.parse_ps(PS_POD1))
    assert len(jobs) == 1
    assert jobs[0].key.endswith("flagship4b-v3enc-expA-nodrop-2k")
    assert len(jobs[0].pids) == 3          # launcher + 2 workers, one run


def test_jupyter_is_not_a_trainer():
    assert fp.discover_jobs(fp.parse_ps(PS_POD1[:1])) == []


def test_log_found_via_launcher_redirect_then_bound_by_mtime():
    jobs = fp.discover_jobs(fp.parse_ps(PS_POD1))
    fp.attach_logs(jobs, fp.parse_logs(LOGS_POD1), NOW)
    assert jobs[0].log == "/tmp/expA_nodrop_2k.out"
    assert jobs[0].log_age_s == pytest.approx(8, abs=1)


def test_log_found_through_the_ppid_chain():
    """pod3's VLM worker carries no redirect; its grandparent bash does."""
    jobs = fp.discover_jobs(fp.parse_ps(PS_POD3_VLM))
    assert any(j.log == "/root/vlmprod/train_queue.log" for j in jobs)


def test_crlf_payload_is_normalised_to_lf_bytes():
    """A CRLF checkout turns every `fi` into `fi\\r` and bash dies with a
    misleading 'unexpected end of file' (cost: one live-run debug cycle)."""
    out = fp._lf("if x\r\nthen\ry\r\nfi\r\n")
    assert isinstance(out, bytes) and b"\r" not in out


# --- the four RED signatures ----------------------------------------------
def test_red_when_train_host_is_idle():
    rep = _host(gpus=GPU_COLD, jobs=[])
    assert _verdict(rep) == "RED"
    assert any("GPU_IDLE_NO_TRAINER" in f for f in rep.findings)


def test_red_when_gpu_memory_is_resident_with_no_owner():
    """The dead-trainer signature: the process died, the allocation didn't."""
    rep = _host(gpus=GPU_ORPHAN, jobs=[])
    assert _verdict(rep) == "RED"
    assert any("ORPHANED_GPU_MEMORY" in f for f in rep.findings)


def test_red_when_process_is_alive_but_the_log_froze():
    """Futex-deadlock class: `ps` says healthy, the log has said nothing for
    hours. This is what a liveness check based on process presence misses."""
    job = fp.Job(key="/w/run", script="train.py", pids=[1], log="/w/run.log",
                 log_age_s=3600, step=1050)
    rep = _host(gpus=GPU_BUSY, jobs=[job])
    assert _verdict(rep) == "RED"
    assert any("LOG_STALE" in f for f in rep.findings)


def test_red_when_step_has_not_moved_since_the_last_probe():
    job = fp.Job(key="/w/run", script="t.py", pids=[1], log="/w/r.log",
                 log_age_s=10, step=1050)
    rep = _host(gpus=GPU_BUSY, jobs=[job])
    prev = {"pX:/w/run": {"step": 1050, "at": NOW - 7200}}
    assert _verdict(rep, prev, gap=7200) == "RED"
    assert any("STEP_NOT_ADVANCING" in f for f in rep.findings)


def test_step_advancing_is_not_flagged():
    """Inverted control: the stall check must not fire on healthy progress."""
    job = fp.Job(key="/w/run", script="t.py", pids=[1], log="/w/r.log",
                 log_age_s=10, step=1100)
    rep = _host(gpus=GPU_BUSY, jobs=[job])
    prev = {"pX:/w/run": {"step": 1050, "at": NOW - 7200}}
    assert _verdict(rep, prev, gap=7200) == "GREEN"


def test_red_when_the_disk_cannot_take_a_write():
    """`df` would have reported the multi-TB cluster and lied; only a real
    write sees the per-pod MooseFS quota."""
    rep = _host(gpus=GPU_BUSY, jobs=[fp.Job(key="/w/r", script="t.py",
                                            pids=[1], log="/w/r.log",
                                            log_age_s=5, step=9)],
                disk_note="FAIL")
    assert _verdict(rep) == "RED"
    assert any("DISK_FULL" in f for f in rep.findings)


def test_red_when_unreachable():
    rep = fp.HostReport(name="pX", ssh="x", role="train", reachable=False,
                        error="ssh timeout")
    assert _verdict(rep) == "RED"


# --- the anti-false-green rules -------------------------------------------
def test_running_job_with_no_discoverable_log_is_amber_not_green():
    """THE regression this tool exists for. The old monitor grepped a log path
    that no longer existed, found nothing, and printed no anomaly."""
    rep = _host(gpus=GPU_BUSY,
                jobs=[fp.Job(key="/w/run", script="t.py", pids=[7], log=None)])
    assert _verdict(rep) == "AMBER"
    assert any("NO_LOG_BOUND" in f for f in rep.findings)


def test_unparseable_step_is_amber_not_green():
    rep = _host(gpus=GPU_BUSY,
                jobs=[fp.Job(key="/w/run", script="t.py", pids=[7],
                             log="/w/r.log", log_age_s=5, step=None)])
    assert _verdict(rep) == "AMBER"


def test_two_concurrent_runs_on_one_host_is_amber():
    jobs = [fp.Job(key="/w/a", script="t.py", pids=[1], log="/a.log",
                   log_age_s=5, step=1),
            fp.Job(key="/w/b", script="t.py", pids=[2], log="/b.log",
                   log_age_s=5, step=1)]
    rep = _host(gpus=GPU_BUSY, jobs=jobs)
    assert _verdict(rep) == "AMBER"
    assert any("MULTIPLE_RUNS" in f for f in rep.findings)


def test_idle_burst_host_is_green_by_design():
    """The eval pod is *supposed* to be idle between jobs — role decides."""
    assert _verdict(_host(role="burst", gpus=GPU_COLD, jobs=[])) == "GREEN"


def test_healthy_training_host_is_green():
    """Inverted control #2: a genuinely healthy host must reach GREEN, or the
    probe is just a red-light generator nobody will read."""
    job = fp.Job(key="/w/run", script="train.py", pids=[1, 2], log="/w/r.log",
                 log_age_s=8, step=1050)
    rep = _host(gpus=GPU_BUSY, jobs=[job], disk_mbps=943.0, disk_note="OK")
    assert _verdict(rep) == "GREEN"
    assert rep.findings == []


# --- parsing edges ---------------------------------------------------------
def test_sections_survive_noise_and_blank_lines():
    sec = fp.split_sections("##NOW\n123\n\n##HOST\nh1\n##END\n")
    assert sec["NOW"] == ["123"] and sec["HOST"] == ["h1"]


def test_step_regex_takes_the_last_occurrence_style_value():
    assert fp.STEP_RE.findall('{"step": 50}{"step":  1050}') == ["50", "1050"]


def test_exit_code_ranks_red_above_amber(monkeypatch, tmp_path):
    reps = [_host(), _host()]
    reps[0].verdict, reps[1].verdict = "AMBER", "RED"
    monkeypatch.setattr(fp, "probe_host",
                        lambda name, spec, *a, **k: reps.pop())
    monkeypatch.setattr(fp, "judge", lambda *a, **k: None)
    rc = fp.main(["--hosts", "pod1", "pod2", "--state", str(tmp_path / "s.json")])
    assert rc == 2
