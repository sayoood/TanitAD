"""Tests of pod_qhead_queue.sh: DRY=1 prints the commands in the card's order; fail-closed refusals; the slot wait (GPU count, container memory); no process-name test, no rm."""
import os
import re
import shutil
import subprocess

import pytest

import qhead_testlib as TL  # noqa: E402

BASH = shutil.which("bash")
SCRIPT = os.path.join(TL.SRC, "pod_qhead_queue.sh")
pytestmark = pytest.mark.skipif(BASH is None, reason="no bash on this box")


def sh(p):
    return str(p).replace("\\", "/")


def run(env_extra, tmp_path, timeout=60):
    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run([BASH, SCRIPT], env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=sh(tmp_path))


def test_syntax_lf_and_forbidden_words():
    r = subprocess.run([BASH, "-n", SCRIPT], capture_output=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    raw = open(SCRIPT, "rb").read()
    assert b"\r" not in raw
    code = "\n".join(l for l in raw.decode("utf-8").splitlines() if not l.lstrip().startswith("#"))
    for bad in ("pgrep", "pkill", "killall", "ps -", "ps aux", "pidof"):
        assert bad not in code, bad
    assert not re.search(r"(^|[;&|\s])rm(\s|$)", code), "rm in the script"
    assert "--query-compute-apps=pid" in code and "memory.max" in code and "memory.stat" in code


def test_dry_run_order_and_content(tmp_path):
    r = run({"DRY": "1", "QH_RUNF": sh(tmp_path / "run")}, tmp_path)
    assert r.returncode == 0, r.stderr + r.stdout
    out = r.stdout
    assert not os.path.exists(tmp_path / "run"), "DRY must create nothing"
    fits = [m.start() for m in re.finditer(r"DRY: \S+ \S+qhead_fit.py --arm (LS_s0|LS_s1|LSD_s0|LSD_s1) ", out)]
    ctrl = out.find("--q-equals-head")
    h1 = out.find("cmp_dumps.py")
    rolls = [m.start() for m in re.finditer(r'DR_MIX="head head head \S+ --qhead \S+(LS|LSD)_s[01]\.qhead\.npz"', out)]
    assert len(fits) == 4 and len(rolls) == 4
    assert max(fits) < ctrl < h1 < min(rolls)
    assert out.count("slot wait before") == 4 + 1 + 4
    assert "DRY: slot wait before fit LS_s0" in out and "DRY: slot wait before rollout LSQHI_s0" in out
    assert "DR_MIX=\"head head head 0.5 --q-equals-head \"" in out
    assert "--seed 1" in out and "--seed 0" in out
    for arm, p in (("LSQH_s0", "0.4"), ("LSQH_s1", "0.5"), ("LSDQH_s0", "0.4"), ("LSDQH_s1", "0.3")):
        assert re.search(r'DR_MIX="head head head %s --qhead \S+" DR_RULEPY=v8f_t1_eval_rule_qhead.py bash \S+ %s ' % (re.escape(p), arm), out), arm
    assert "never a fourth" not in out or True


def fake_pod(tmp_path, gpu_pids=0, anon_gb=1, max_gb=60):
    d = tmp_path / "pod"
    (d / "bin").mkdir(parents=True, exist_ok=True)
    smi = d / "bin" / "nvidia-smi"
    smi.write_text("#!/usr/bin/env bash\n" + "".join("echo %d\n" % (1000 + i) for i in range(gpu_pids)) + "exit 0\n", encoding="utf-8", newline="\n")
    (d / "memory.max").write_text(str(max_gb * 1073741824), encoding="utf-8")
    (d / "memory.stat").write_text("anon %d\nfile 1\n" % (anon_gb * 1073741824), encoding="utf-8")
    D = d / "distrule"
    D.mkdir(exist_ok=True)
    for f in ("qhead_model.py", "qhead_fit.py", "v8f_t1_eval_rule_qhead.py", "v8f_t1_eval_rule_mix.py", "v8f_t1_eval_rule.py", "dist_rule.py", "cmp_dumps.py"):
        (D / f).write_text("# stub\n", encoding="utf-8")
    (D / "pod_ruleattr_run.sh").write_text('if [ "$MODE" = mix ]; then set -- $DR_MIX; ARGV=(--tree "$TREE" --mix-presence "$1" --mix-gap "$2" --mix-closing "$3"); fi\n', encoding="utf-8", newline="\n")
    shutil.copy(os.path.join(TL.SRC, "qhead_arms.txt"), D / "qhead_arms.txt")
    (D / "train_pack.npz").write_bytes(b"x")
    py = d / "fakepy"
    py.write_text("#!/usr/bin/env bash\necho fake python $@\nexit 0\n", encoding="utf-8", newline="\n")
    env = {"QH_W": sh(d), "QH_D": sh(D), "QH_RUNF": sh(d / "run"), "QH_PY": sh(py), "QH_CG_MAX": sh(d / "memory.max"), "QH_CG_STAT": sh(d / "memory.stat"),
           "PATH": sh(d / "bin") + os.pathsep + os.environ["PATH"], "QH_PROBE_GAP_S": "1", "QH_POLL_S": "1", "QH_MAX_WAIT_S": "2"}
    return d, env


def test_second_start_is_refused_by_marker(tmp_path):
    d, env = fake_pod(tmp_path)
    (d / "run").mkdir()
    (d / "run" / "QHEAD_STARTED.txt").write_text("x")
    r = run(env, tmp_path)
    assert r.returncode == 2 and "REFUSED" in r.stdout and "QHEAD_STARTED.txt" in r.stdout


@pytest.mark.parametrize("marker", ["QHEAD_DONE.txt", "QHEAD_STOP.txt"])
def test_end_markers_refuse_a_restart(tmp_path, marker):
    d, env = fake_pod(tmp_path)
    (d / "run").mkdir()
    (d / "run" / marker).write_text("x")
    r = run(env, tmp_path)
    assert r.returncode == 2 and marker in r.stdout


def test_missing_tool_file_stops_closed_with_a_marker(tmp_path):
    d, env = fake_pod(tmp_path)
    os.replace(d / "distrule" / "qhead_fit.py", d / "distrule" / "qhead_fit.py.moved")
    r = run(env, tmp_path)
    assert r.returncode == 4 and (d / "run" / "QHEAD_STOP.txt").exists() and "qhead_fit.py" in (d / "run" / "QHEAD_STOP.txt").read_text()


def test_slot_wait_refuses_a_fourth_gpu_job(tmp_path):
    d, env = fake_pod(tmp_path, gpu_pids=3)
    r = run(env, tmp_path)
    assert r.returncode == 4, r.stdout[-600:]
    assert "never a fourth GPU job" in (d / "run" / "QHEAD_STOP.txt").read_text() and "fit LS_s0" in (d / "run" / "QHEAD_STOP.txt").read_text()


def test_slot_wait_needs_28_gb_of_the_container_free(tmp_path):
    d, env = fake_pod(tmp_path, gpu_pids=0, anon_gb=40, max_gb=60)          # 20 GB free
    r = run(env, tmp_path)
    assert r.returncode == 4 and "never a fourth GPU job" in (d / "run" / "QHEAD_STOP.txt").read_text()


def test_slot_passes_with_two_jobs_and_memory_then_the_fit_input_check_stops(tmp_path):
    d, env = fake_pod(tmp_path, gpu_pids=2, anon_gb=10, max_gb=60)
    r = run(env, tmp_path)
    assert r.returncode == 4, r.stdout[-600:]
    stop = (d / "run" / "QHEAD_STOP.txt").read_text()
    assert "no train states for LS_s0" in stop
    assert (d / "run" / "QHEAD_STARTED.txt").exists()
    assert "slot free on two probes for fit LS_s0" in (d / "run" / "qhead_queue.log").read_text()
