"""test_constclose_queue.py -- tests of make_constclose_queue.py and of the queue it writes (pod_constclose10k_queue.sh): the generator runs into a temporary file, the queue is checked by bash -n and by
its DRY=1 output (no pod path is touched: DRY prints every command and runs nothing). CC_TOOL_DIR (default: this folder) names the folder of the generator under test.
"""
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.environ.get("CC_TOOL_DIR") or HERE
SRC = os.path.normpath(os.path.join(HERE, "..", "estclose", "pod_estclose10k_queue.sh"))
def _bash():
    for p in (os.environ.get("CC_BASH", ""), "C:/Program Files/Git/usr/bin/bash.exe", "C:/Program Files/Git/bin/bash.exe", "/usr/bin/bash", "/bin/bash"):
        if p and os.path.isfile(p):
            return p
    return "bash"


BASH = _bash()
ORDER = [("P30Hb20", "C20"), ("P30Hb20", "C10"), ("P30Hb20", "C40"), ("Hb20", "C20"), ("Hb20", "C10"), ("Hb20", "C40")]


@pytest.fixture(scope="module")
def queue(tmp_path_factory):
    d = tmp_path_factory.mktemp("q")
    out = d / "q.sh"
    env = dict(os.environ, CC_SRC=SRC, CC_OUT=str(out), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "make_constclose_queue.py"), "ADD80"], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=TOOLS)
    assert r.returncode == 0, r.stdout + r.stderr
    return out, out.read_text(encoding="utf-8")


def dry(path):
    env = dict(os.environ, DRY="1", I10_W="/x/y")
    r = subprocess.run([BASH, path.as_posix()], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    return r.stdout


def test_bash_syntax_and_refusal_of_overwrite(queue, tmp_path):
    path, text = queue
    assert subprocess.run([BASH, "-n", path.as_posix()]).returncode == 0
    env = dict(os.environ, CC_SRC=SRC, CC_OUT=str(path), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "make_constclose_queue.py"), "ADD80"], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "REFUSED" in (r.stdout + r.stderr)
    bad = tmp_path / "other.sh"
    bad.write_text("echo not the queue of card ADD79\n")
    env = dict(os.environ, CC_SRC=str(bad), CC_OUT=str(tmp_path / "o.sh"))
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "make_constclose_queue.py"), "ADD80"], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and not (tmp_path / "o.sh").exists()


def test_text_markers_names_and_checks(queue):
    path, text = queue
    for m in ("START_CONSTCLOSE10K.txt", "CONSTCLOSE10K_DONE.txt", "CONSTCLOSE10K_STOP.txt", "queue_constclose.log"):
        assert m in text, m
    for m in ("ESTCLOSE10K", "queue_estclose", "START_ESTCLOSE", "estclose_test", "N10", "L25"):
        assert m not in text, m
    assert "a747abeb51ef041f" in text and "^ZZ CHECK-CONSTCLOSE PASS" in text and "CHECK_CONST_DONE_$CKTAG.txt" in text and "CHECK_CONSTCLOSE_$CKTAG.txt.hashes" in text
    assert 'CKTAG=${CC_CHECK_TAG:-ck1}' in text
    assert 'grep -q "$got" "$W/constclose_test/CHECK_CONSTCLOSE_$CKTAG.txt.hashes"' in text and 'sha256sum "$ESTC/const_C20_H1.npz"' in text
    assert 'NO-CONST-FILE' in text and "for c in C10 C20 C40; do" in text
    assert 'ESTC=${I10_ESTC:-$V/closeread/est_const}' in text
    assert 'echo "$(ts) all steps done" > "$W/CONSTCLOSE10K_DONE.txt"' in text and 'echo "$(ts) $*" > "$W/CONSTCLOSE10K_STOP.txt"' in text and '> "$W/START_CONSTCLOSE10K.txt"' in text
    assert 'LOG=$W/queue_constclose.log' in text and 'SM=$W/small_cc' in text
    assert "I10RQ50cKF5P30Hb20" in text and "I10RQ50cKF5Hb20" in text and "I10R0Hb20" in text
    assert text.count("\r") == 0


def test_twelve_rollouts_in_order_with_the_right_files(queue):
    path, text = queue
    out = dry(path)
    cmds = [l for l in out.splitlines() if l.startswith("CMD:")]
    assert len(cmds) == 12
    want = []
    for setting, c in ORDER:
        for k in (0, 1):
            want.append(("I10RQ50e%sKF5%s_s%d" % (c, setting, k), c, setting, k))
    for cmd, (arm, c, setting, k) in zip(cmds, want):
        assert " --form RRe " in cmd
        assert "--closing-est /workspace/v8proof/closeread/est_const/const_%s_H1.npz" % c in cmd
        assert "--rule-every 5" in cmd and "--readout causal_g" in cmd and "--acc-gain 0" in cmd and "--path-exec" in cmd and "--kappa-max 0.2" in cmd
        assert "--reader /x/y/lr16/TK16_s%d/ckpt_last.pt" % k in cmd and "--qmap /x/y/qmap_sweep/QMAP_LR16_s%d_q50.json" % k in cmd and "/x/y/ckpt/UE_s%d/A/ckpt_last.pt" % k in cmd
        assert "--out-dir /x/y/t1/%s " % arm in cmd and ("--tag %s_m10_causal_heads" % arm) in cmd
        assert "v8f_t1_eval_reader_rule_rzgfhqe.py" in cmd
        if setting == "P30Hb20":
            assert "--preroll-steps 30" in cmd and "grid_p30.json" in cmd
        else:
            assert "--preroll-steps 0" in cmd and "eval_grid_windows.json" in cmd
    assert out.strip().splitlines()[-1] == "DRY DONE"


def test_log_markers_in_dry(queue):
    out = dry(queue[0])
    assert "CONSTCLOSE10K" in out and "12 rollouts" in out
    assert not re.search(r"ESTCLOSE|MIDLOOK", out)
