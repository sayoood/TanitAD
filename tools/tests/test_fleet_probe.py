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
    # `dd_path_hinted=True` = a curated pod with a declared /workspace, which
    # every ROLE_HINTS entry is. Only a host whose scratch dir was GUESSED
    # downgrades a failed write from RED to AMBER (see the 2026-08-03 tests).
    rep = fp.HostReport(name="pX", ssh="x", role=role, reachable=True,
                        dd_path_hinted=True)
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


# ==========================================================================
# v2 — fleet DISCOVERY (2026-08-03)
#
# v1 discovered logs but hardcoded the four-host FLEET dict, which is the same
# defect one level up. Measured that day: the live ssh config held pod4/v2arch,
# pod5/new and thor, none of them in that dict, while POD_SHUTDOWN_2026-08-02
# records an arm training on pod4. These falsifiers pin the property that a
# host present in the config is never silently absent from the probe.
# ==========================================================================

SSH_CONFIG_LIVE = """\
Host tanitad-pod
    HostName 38.147.83.15
    Port 39198
    User root

Host tanitad-pod2
    HostName 69.30.85.123
    Port 22091
    User root

Host tanitad-eval
    HostName 69.30.85.106
    Port 22073
    User root

Host tanitad-thor
    HostName 192.168.178.194
    User nvidia

Host tanitad-pod4 tanitad-v2arch
    HostName 69.30.85.48
    Port 22192
    User root

Host tanitad-pod9000 tanitad-brandnew
    HostName 10.0.0.9
    Port 2222
    User root

Host github.com
    HostName github.com

Host *
    ServerAliveInterval 30
"""


def _fleet(text=SSH_CONFIG_LIVE):
    return fp.discover_fleet(fp.parse_ssh_config(text))


def test_a_never_configured_host_is_still_discovered():
    """The whole point: a pod nobody added to ROLE_HINTS is STILL probed.

    Falsifies v1, whose FLEET dict would have omitted it entirely.
    """
    fleet = _fleet()
    endpoints = {spec["endpoint"] for spec in fleet.values()}
    assert "10.0.0.9:2222" in endpoints
    assert "69.30.85.48:22192" in endpoints        # pod4, the one that trained


def test_unhinted_host_is_amber_never_dropped_and_never_green():
    """An unclassified host must be loud, not absent and not quietly fine."""
    fleet = _fleet()
    unclassified = [n for n, s in fleet.items() if s["role"] == "unknown"]
    assert unclassified, "a host with no ROLE_HINTS entry must exist here"
    rep = fp.HostReport(name=unclassified[0], ssh="x", role="unknown")
    rep.reachable = True
    rep.gpus = [{"index": 0, "name": "A40", "util": 0,
                 "mem_used": 0, "mem_total": 46068}]
    fp.judge(rep, {}, NOW, None)
    assert rep.verdict == "AMBER"
    assert any("IDLE_UNCLASSIFIED_HOST" in f for f in rep.findings)


def test_two_aliases_one_endpoint_are_probed_once():
    """pod4 and v2arch are one A40 — probing twice double-counts the fleet."""
    fleet = _fleet()
    pod4 = [s for s in fleet.values() if s["endpoint"] == "69.30.85.48:22192"]
    assert len(pod4) == 1
    assert pod4[0]["aliases"] == ["tanitad-pod4", "tanitad-v2arch"]
    assert pod4[0]["ssh"] == "tanitad-pod4"        # the hinted alias wins


def test_non_fleet_and_wildcard_hosts_are_excluded():
    fleet = _fleet()
    assert not any("github" in a for s in fleet.values() for a in s["aliases"])
    assert not any("*" in a for s in fleet.values() for a in s["aliases"])


def test_missing_port_defaults_and_distinct_ports_stay_distinct():
    """eval and a same-IP different-port pod are two machines, not one."""
    cfg = fp.parse_ssh_config(SSH_CONFIG_LIVE + """
Host tanitad-pod5
    HostName 69.30.85.106
    Port 22039
""")
    fleet = fp.discover_fleet(cfg)
    eps = {s["endpoint"] for s in fleet.values()}
    assert "69.30.85.106:22073" in eps and "69.30.85.106:22039" in eps
    thor = [s for s in fleet.values() if s["role"] == "edge"][0]
    assert thor["endpoint"].endswith(":22")       # no Port line -> default


def test_a_hinted_alias_vanishing_from_the_config_is_reported(tmp_path):
    """A released pod must be visible as ALIAS_GONE, not silently forgotten."""
    cfg = tmp_path / "config"
    cfg.write_text(SSH_CONFIG_LIVE, encoding="utf-8")
    fleet, warns = fp.load_fleet(cfg)
    assert fleet
    gone = [w for w in warns if "ALIAS_GONE" in w]
    assert any("tanitad-pod3" in w for w in gone)   # in ROLE_HINTS, not in cfg
    assert all(w.startswith(("AMBER", "RED")) for w in warns)


def test_no_ssh_config_is_RED_not_an_empty_green_fleet(tmp_path):
    """The failure mode this whole tool exists to prevent, at the top level."""
    fleet, warns = fp.load_fleet(tmp_path / "does-not-exist")
    assert fleet == {}
    assert warns and warns[0].startswith("RED NO_SSH_CONFIG")


def test_a_config_with_no_tanitad_host_is_RED_not_green(tmp_path):
    cfg = tmp_path / "config"
    cfg.write_text("Host github.com\n  HostName github.com\n", encoding="utf-8")
    fleet, warns = fp.load_fleet(cfg)
    assert fleet == {}
    assert any(w.startswith("RED NO_FLEET_DISCOVERED") for w in warns)


def test_main_exits_RED_when_the_fleet_cannot_be_discovered(tmp_path, capsys):
    """Inverted falsifier: prove the discovery failure reaches the exit code."""
    rc = fp.main(["--ssh-config", str(tmp_path / "nope"), "--no-dd"])
    assert rc == 2


def test_render_folds_discovery_warnings_into_the_fleet_verdict():
    """A GREEN host list must not print FLEET: GREEN when discovery warned."""
    rep = fp.HostReport(name="pod1", ssh="tanitad-pod", role="train")
    rep.reachable, rep.verdict = True, "GREEN"
    clean = fp.render([rep], [])
    assert "FLEET: GREEN" in clean
    warned = fp.render([rep], ["AMBER HOST_UNCLASSIFIED: pod9000 ..."])
    assert "FLEET: AMBER" in warned


def test_dd_path_is_per_host_so_thor_is_not_probed_at_workspace():
    """Thor has no /workspace; a hardcoded dd there is a manufactured RED."""
    fleet = _fleet()
    thor = [s for s in fleet.values() if s["role"] == "edge"][0]
    assert thor["dd_path"] == "/home/nvidia"
    pods = [s for s in fleet.values() if s["role"] != "edge"]
    assert all(s["dd_path"] == "/workspace" for s in pods)


def test_failed_dd_at_a_GUESSED_path_is_amber_not_a_manufactured_red():
    """Measured 2026-08-03: thor-wifi has no /workspace, and defaulting the dd
    path there produced `RED DISK_FULL` on a device with 847 MB/s of headroom.
    RED asserts a cause (quota); an unwritable *guessed* path only establishes
    "unknown". Absence of evidence stays an alarm — but the honest one."""
    rep = fp.HostReport(name="mystery", ssh="x", role="burst",
                        dd_path="/workspace", dd_path_hinted=False)
    rep.reachable, rep.disk_note = True, "FAIL"
    fp.judge(rep, {}, NOW, None)
    assert rep.verdict == "AMBER"
    assert any("DISK_UNVERIFIED" in f for f in rep.findings)
    assert not any("DISK_FULL" in f for f in rep.findings)

    hinted = fp.HostReport(name="pod2", ssh="x", role="burst",
                           dd_path="/workspace", dd_path_hinted=True)
    hinted.reachable, hinted.disk_note = True, "FAIL"
    fp.judge(hinted, {}, NOW, None)
    assert hinted.verdict == "RED"                 # the real quota case stays RED
    assert any("DISK_FULL" in f for f in hinted.findings)


def test_thor_wifi_is_the_same_device_not_an_unclassified_host():
    cfg = fp.parse_ssh_config(SSH_CONFIG_LIVE + """
Host tanitad-thor-wifi
    HostName 192.168.178.93
    User nvidia
""")
    fleet = fp.discover_fleet(cfg)
    wifi = fleet["thor-wifi"]
    assert wifi["role"] == "edge" and wifi["dd_path"] == "/home/nvidia"


# --- /proc/<pid>/fd/1 binding (2026-08-03) --------------------------------
# Measured that day on pod5: the v5f flagship ran with
#   --out /workspace/experiments/flagship-v5f-w120-30k
# while writing to /workspace/v5f_run.log. No prefix or basename rule links
# those two, and the launcher (`ssh -f`) left no shell parent carrying a
# redirect — so the probe reported the live flagship as liveness UNVERIFIED.
# The kernel knew the answer the whole time.

def test_proc_fd_binds_a_log_that_no_name_heuristic_could_reach():
    job = fp.Job(key="/workspace/experiments/flagship-v5f-w120-30k",
                 script="scripts/train_flagship_v4.py", pids=[19412, 19430])
    logs = [(NOW - 30, 4096, "/workspace/v5f_run.log"),
            (NOW - 5, 99, "/workspace/experiments/unrelated.log")]
    fp.attach_logs([job], logs, NOW, {19412: "/workspace/v5f_run.log"})
    assert job.log == "/workspace/v5f_run.log"
    assert job.log_bound_by == "proc_fd"
    assert job.log_age_s == pytest.approx(30, abs=1)


def test_without_proc_fd_the_same_job_is_correctly_UNVERIFIED():
    """The inverted case: the heuristics genuinely cannot solve this, so the
    v1 AMBER was right, not a bug. Only new evidence resolves it."""
    job = fp.Job(key="/workspace/experiments/flagship-v5f-w120-30k",
                 script="scripts/train_flagship_v4.py", pids=[19412])
    fp.attach_logs([job], [(NOW - 30, 4096, "/workspace/v5f_run.log")],
                   NOW, {})
    assert job.log is None


def test_proc_fd_ignores_ttys_pipes_and_devnull():
    """A process on a tty has no log; binding one would invent evidence."""
    m = fp.parse_fd(["1|/dev/null", "2|/dev/pts/0", "3|pipe:[12345]",
                     "4|socket:[9]", "5|/workspace/real.log", "x|/bad"])
    assert m == {5: "/workspace/real.log"}


def test_proc_fd_binding_outside_the_find_window_is_amber_not_green():
    """Bound with certainty, freshness unknown -> still an alarm, not a pass."""
    job = fp.Job(key="/w/run", script="t.py", pids=[7])
    fp.attach_logs([job], [], NOW, {7: "/w/old.log"})
    assert job.log == "/w/old.log" and job.log_age_s is None
    rep = _host(gpus=GPU_BUSY, jobs=[job])
    fp.judge(rep, {}, NOW, None)
    assert rep.verdict == "AMBER"
    assert any("LOG_AGE_UNKNOWN" in f for f in rep.findings)
