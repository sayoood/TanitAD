"""Control-flow tests for force_s2_launcher_r8.sh -- a SANDBOX dry run (fake logs, fake free-commit probe, a STUB stage script): no GPU, no model, no real
roll, nothing outside tmp_path is read or written.  Needs Git-for-Windows `sh`.

Run:  python -m pytest -q -p no:cacheprovider test_force_s2_launcher_r8.py

Every wait gate is paired with the state that must OPEN it and the state that must NOT; in particular the fp64 gate is tested against the STALE terminal
marker that the real run.log carries from the aborted 19:56 attempt (a plain grep would have opened the gate at once)."""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SH = shutil.which("sh") or "C:/Program Files/Git/bin/sh.exe"
needs_sh = pytest.mark.skipif(not Path(SH).exists() and not shutil.which("sh"), reason="no sh")
LAUNCHER = HERE / "force_s2_launcher_r8.sh"
PYX = Path(sys.executable).as_posix()

Q_OK = "ZZQMSG0OKZZ x\nZZQQUEUEENDZZ 2026-10-18T23:30:00+0200 all stages attempted\n"
Q_NO = "ZZQMSG0OKZZ x\nZZQRELEASEDZZ refcv8-wrapfix-step30000\n"
F_STALE_ONLY = ("ZZF64STARTZZ 19:56:01\nZZF64ARMOKZZ F32\nZZF64COMBINEFAILZZ 19:56:34\nZZF64WAITZZ 19:58:42 for 30k queue end + 10 GB headroom\n")
FL_DONE = "ZZFLPROGZZ 20/100\nZZFLDRIVERENDZZ rolled=500/500\n"


def P(p):
    return Path(p).as_posix()


def run_launcher(tmp, *, q=Q_OK, f=F_STALE_ONLY + "ZZF64DONEZZ 2026-10-18T23:40:00+0200 ok\n", fl=FL_DONE, fl_age_s=0, free="echo 20", f64_alive="echo 1",
                 max_wait=4, dry=True, extra_env=None, py=None, pkg=None, timeout=120):
    tmp = Path(tmp)
    qp, fp, flp, lg, scr = tmp / "q.log", tmp / "f64.log", tmp / "fl.log", tmp / "force_s2.log", tmp / "scr"
    if q is not None:
        qp.write_text(q)
    if f is not None:
        fp.write_text(f)
    if fl is not None:
        flp.write_text(fl)
        t = time.time() - fl_age_s
        os.utime(flp, (t, t))
    if lg.exists():
        lg.unlink()
    env = dict(os.environ)
    env.update({"FS2_QUEUE_LOG": P(qp), "FS2_F64_LOG": P(fp), "FS2_FL_LOG": P(flp), "FS2_LOG": P(lg), "FS2_SCR": P(scr), "FS2_POLL_S": "1", "FS2_GAP_S": "1",
                "FS2_MAX_WAIT_S": str(max_wait), "FS2_FREE_CMD": free, "FS2_F64_ALIVE_CMD": f64_alive, "FS2_PY": py or PYX,
                "FS2_PKG": P(pkg or tmp / "pkg")})
    if dry:
        env["FS2_DRY"] = "1"
    env.update(extra_env or {})
    r = subprocess.run([SH, P(LAUNCHER)], env=env, capture_output=True, text=True, timeout=timeout)
    return (lg.read_text() if lg.exists() else ""), r


def opened(log):
    return "ZZFS2GATEOPENZZ" in log


def timed_out(log):
    return "ZZFS2TIMEOUTZZ" in log


# ====================================================================================================== the four gates
@needs_sh
def test_all_gates_true_opens_and_dry_run_stops_before_any_stage(tmp_path):
    log, _ = run_launcher(tmp_path)
    assert "ZZFS2ARMEDZZ" in log and opened(log) and "ZZFS2DRYENDZZ" in log and "free commit 20/20 GB" in log
    assert "ZZFS2SELECTSTARTZZ" not in log and not (tmp_path / "scr" / "select.log").exists()


@needs_sh
def test_queue_end_marker_missing_keeps_the_gate_shut(tmp_path):
    log, _ = run_launcher(tmp_path, q=Q_NO)
    assert timed_out(log) and not opened(log)


@needs_sh
def test_a_stale_fp64_terminal_marker_BEFORE_the_last_wait_line_does_not_open_the_gate(tmp_path):
    # the real run.log today: START / ARMOK / ARMOK / COMBINEFAIL (19:56 attempt) then WAIT (19:58 v2 script) and nothing after it
    log, _ = run_launcher(tmp_path, f=F_STALE_ONLY, f64_alive="echo 1")
    assert timed_out(log) and not opened(log) and "F64DEAD" not in log


@needs_sh
@pytest.mark.parametrize("term", ["DONE", "ARMFAIL", "COMBINEFAIL", "TIMEOUT"])
def test_every_terminal_marker_AFTER_the_last_wait_line_opens_the_gate(tmp_path, term):
    log, _ = run_launcher(tmp_path, f=F_STALE_ONLY + f"ZZF64STARTZZ 23:35\nZZF64{term}ZZ 23:40 x\n")
    assert opened(log) and not timed_out(log) and "F64DEAD" not in log


@needs_sh
def test_a_dead_fp64_script_with_no_terminal_marker_opens_with_a_loud_marker(tmp_path):
    log, _ = run_launcher(tmp_path, f=F_STALE_ONLY, f64_alive="echo 0")
    assert "ZZFS2F64DEADZZ" in log and opened(log)


@needs_sh
def test_missing_fp64_log_with_a_live_probe_waits(tmp_path):
    log, _ = run_launcher(tmp_path, f=None, f64_alive="echo 1")
    assert timed_out(log) and not opened(log)


@needs_sh
def test_force_lon_roll_alive_blocks_and_done_or_stale_does_not(tmp_path):
    busy, _ = run_launcher(tmp_path, fl="ZZFLPROGZZ 20/100\n", fl_age_s=30)                 # no end marker, written 30 s ago: alive
    assert timed_out(busy) and not opened(busy)
    done, _ = run_launcher(tmp_path, fl="ZZFLPROGZZ 20/100\nZZFLDRIVERENDZZ rolled=500/500\n", fl_age_s=30)
    assert opened(done)
    stale, _ = run_launcher(tmp_path, fl="ZZFLPROGZZ 20/100\n", fl_age_s=3600)             # no end marker but silent for an hour: not alive (today's real state)
    assert opened(stale)
    none, _ = run_launcher(tmp_path, fl=None)                                                # no force_lon log at all
    assert opened(none)


# ====================================================================================================== free commit: two consecutive checks
@needs_sh
@pytest.mark.parametrize("free", ["echo 5", "echo 11", "echo", "echo abc", "echo -3", "echo 11.9"])
def test_insufficient_or_unreadable_free_commit_never_opens(tmp_path, free):
    log, _ = run_launcher(tmp_path, free=free)
    assert timed_out(log) and not opened(log)


@needs_sh
def test_exactly_the_threshold_counts_and_a_single_good_check_is_not_enough(tmp_path):
    ok, _ = run_launcher(tmp_path, free="echo 12")
    assert opened(ok) and "free commit 12/12 GB" in ok
    cnt, seq = tmp_path / "cnt", tmp_path / "seq"
    seq.write_text("20\n5\n20\n20\n20\n20\n")                                                 # good, BAD, good, good -> only the 3rd+4th pair opens it
    cmd = f'n=$(cat {P(cnt)} 2>/dev/null); n=$((${{n:-0}}+1)); echo $n > {P(cnt)}; sed -n "${{n}}p" {P(seq)}'
    log, _ = run_launcher(tmp_path, free=cmd, max_wait=30)
    assert opened(log) and "free commit 20/20 GB" in log and "second check 5 GB" in log
    assert cnt.read_text().strip() == "4"                                                    # exactly four probes: a single-check gate would have stopped at 2


# ====================================================================================================== the single-launcher guard
@needs_sh
def test_a_live_lock_holder_refuses_and_a_dead_one_is_reclaimed(tmp_path):
    scr = tmp_path / "scr"
    (scr / "launcher.lock").mkdir(parents=True)
    pidf = tmp_path / "pid"
    holder = subprocess.Popen([SH, "-c", f"echo $$ > {P(pidf)}; exec sleep 25"])
    try:
        for _ in range(50):
            if pidf.exists() and pidf.read_text().strip():
                break
            time.sleep(0.1)
        (scr / "launcher.lock" / "pid").write_text(pidf.read_text().strip())
        log, r = run_launcher(tmp_path)
        assert "ZZFS2ALREADYZZ" in log and not opened(log) and r.returncode == 6
    finally:
        holder.terminate()                                                                    # our own helper process
    (scr / "launcher.lock" / "pid").write_text("99999999")
    log, r = run_launcher(tmp_path)
    assert opened(log) and "ALREADY" not in log
    assert not (scr / "launcher.lock").exists()                                              # released on exit


# ====================================================================================================== the stage sequencing (stub stage script)
STUB = r'''
import json, os, sys
from pathlib import Path
scr = Path(os.environ["FS2_SCR"]); out = Path(os.environ["FS2_PKG"]) / "raw" / "force_s2"; mode = os.environ.get("STUB_MODE", "ok")
stage = sys.argv[1]
if stage == "select":
    scr.mkdir(parents=True, exist_ok=True)
    (scr / "plan_raw.json").write_text(json.dumps({"entries": [{"clip": f"c{i}", "ws": 1} for i in range(4)]}))
    (scr / "s2_ref.npz").write_bytes(b"x")
    if mode != "select_nomarker":
        print("ZZFS2SELECTOKZZ")
elif stage == "roll":
    n = int((scr / "rolls").read_text()) + 1 if (scr / "rolls").exists() else 1
    (scr / "rolls").write_text(str(n))
    k = {"ok": 4, "partial_then_ok": 2 if n == 1 else 4, "never": 2}.get(mode, 4)
    (scr / "roll_full.json").write_text(json.dumps({"windows": {f"c{i}|1": {} for i in range(k)}}))
elif stage == "status":
    plan = json.loads((scr / "plan_raw.json").read_text())["entries"]
    rp = scr / "roll_full.json"
    have = set(json.loads(rp.read_text())["windows"]) if rp.exists() else set()
    print("ZZFS2STATUSZZ rolled=%d/%d" % (len(have & {f"{e['clip']}|{e['ws']}" for e in plan}), len(plan)))
elif stage == "analyze":
    out.mkdir(parents=True, exist_ok=True)
    (out / "FORCE_S2.json").write_text("{}")
    if mode != "analyze_nomarker":
        print('  "verdict": "TIMING-CONTROLLABLE",'); print("ZZFS2ANALYZEDONEZZ")
'''


def sandbox(tmp):
    pkg = Path(tmp) / "pkg"
    (pkg / "code").mkdir(parents=True, exist_ok=True)
    (pkg / "code" / "force_s2_reroll_r8.py").write_text(STUB)
    return pkg


@needs_sh
def test_stage_sequence_happy_path_judges_each_stage_by_its_artifact(tmp_path):
    pkg = sandbox(tmp_path)
    log, _ = run_launcher(tmp_path, dry=False, pkg=pkg, extra_env={"STUB_MODE": "ok"})
    order = [m for m in ("ZZFS2GATEOPENZZ", "ZZFS2SELECTSTARTZZ", "ZZFS2SELECTOKZZ", "ZZFS2ROLLSTARTZZ", "ZZFS2ROLLDONEZZ", "ZZFS2ANALYZESTARTZZ", "ZZFS2ANALYZEDONEZZ")]
    pos = [log.find(m) for m in order]
    assert all(p >= 0 for p in pos) and pos == sorted(pos)
    assert "rolled=4/4" in log and 'ANALYZEDONEZZ' in log and '"verdict": "TIMING-CONTROLLABLE"' in log
    assert (tmp_path / "scr" / "rolls").read_text() == "1"                                    # the roll ran once: pass 1 was complete


@needs_sh
def test_select_without_its_end_marker_stops_the_launcher_before_the_roll(tmp_path):
    pkg = sandbox(tmp_path)
    log, r = run_launcher(tmp_path, dry=False, pkg=pkg, extra_env={"STUB_MODE": "select_nomarker"})
    assert "ZZFS2SELECTFAILZZ" in log and "ZZFS2ROLLSTARTZZ" not in log and r.returncode == 3


@needs_sh
def test_incomplete_roll_is_retried_once_resumably_then_analyzed(tmp_path):
    pkg = sandbox(tmp_path)
    log, _ = run_launcher(tmp_path, dry=False, pkg=pkg, extra_env={"STUB_MODE": "partial_then_ok"})
    assert "ZZFS2ROLLINCOMPLETEZZ" in log and "rolled=2/4 after pass 1" in log and "ZZFS2ROLLDONEZZ" in log and "ZZFS2ANALYZEDONEZZ" in log
    assert (tmp_path / "scr" / "rolls").read_text() == "2"


@needs_sh
def test_a_roll_that_never_completes_never_reaches_analyze(tmp_path):
    pkg = sandbox(tmp_path)
    log, r = run_launcher(tmp_path, dry=False, pkg=pkg, extra_env={"STUB_MODE": "never"})
    assert "ZZFS2ROLLFAILZZ" in log and "ZZFS2ANALYZESTARTZZ" not in log and r.returncode == 5
    assert (tmp_path / "scr" / "rolls").read_text() == "2"                                    # exactly two passes, no more


@needs_sh
def test_analyze_without_its_end_marker_is_a_failure_even_though_the_json_exists(tmp_path):
    pkg = sandbox(tmp_path)
    log, r = run_launcher(tmp_path, dry=False, pkg=pkg, extra_env={"STUB_MODE": "analyze_nomarker"})
    assert (pkg / "raw" / "force_s2" / "FORCE_S2.json").exists()
    assert "ZZFS2ANALYZEFAILZZ" in log and "ZZFS2ANALYZEDONEZZ" not in log and r.returncode == 7
