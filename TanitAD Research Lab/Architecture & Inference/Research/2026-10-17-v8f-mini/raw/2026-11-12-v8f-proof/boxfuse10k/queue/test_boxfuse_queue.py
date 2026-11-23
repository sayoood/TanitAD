"""test_boxfuse_queue.py -- card ADD131 (BOXFUSE10K), queue kit: the generated queue in DRY mode (creates nothing), step 1 run on tiny synthetic records and files
with stand-in functions, and the BUSY/IDLE fragment. The source queue of card ADD115 is taken from ../../presattrc/queue (same relative place in the pod stage).
Child processes run in the folder named by the env BOXFUSE_Q_TEST_CWD (default: the test's tmp_path)."""
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(HERE, "pod_boxfuse10k_queue.sh")
GEN = os.path.join(HERE, "make_boxfuse_queue.py")
BASH = shutil.which("bash")
PYX = sys.executable.replace("\\", "/")


def rd(p):
    return io.open(str(p), encoding="utf-8", newline="").read()


def fwd(p):
    return str(p).replace("\\", "/")


def wr(p, txt):
    io.open(str(p), "w", encoding="utf-8", newline="\n").write(txt)


def cwd_of(tmp_path):
    return os.environ.get("BOXFUSE_Q_TEST_CWD") or str(tmp_path)


def run_bash(script_path, tmp_path, env=None, timeout=120):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([BASH, fwd(script_path)], capture_output=True, text=True, encoding="utf-8", env=e, cwd=cwd_of(tmp_path), timeout=timeout)


def cmds():
    r = subprocess.run([BASH, fwd(Q)], capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, DRY="1"), timeout=120)
    assert r.returncode == 0, r.stdout[-500:]
    return [x for x in r.stdout.splitlines() if x.startswith("CMD: DISTRULE_DIR")], r.stdout


def test_generated_file_equals_the_committed_one():
    r = subprocess.run([sys.executable, GEN, "--print"], capture_output=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.replace("\r\n", "\n") == rd(Q)


def test_lf_only_and_syntax():
    assert "\r" not in rd(Q)
    r = subprocess.run([BASH, "-n", fwd(Q)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr


def test_six_commands_in_order_with_the_card_arguments():
    c, out = cmds()
    assert len(c) == 6
    names = [re.search(r"--tag (\S+)_m10_causal_heads", x).group(1) for x in c]
    assert names == ["I10RQ50KF5Hb20gJEc_s0", "I10RQ50KF5Hb20gJEc_s1", "I10RQ50KF5Hb20gJA90_s0", "I10RQ50KF5Hb20gJA90_s1",
                     "I10RQ50Hb20gJEc_s0", "I10RQ50Hb20gJEc_s1"]
    gf = "/workspace/v8proof/boxfuse10k/gapfiles"
    want_gap = [gf + "/VGAPJE.npz"] * 2 + [gf + "/VGAPJE90.npz"] * 2 + [gf + "/VGAPJE.npz"] * 2
    for i, x in enumerate(c):
        k = i % 2
        assert "v8f_t1_eval_reader_rule_rzgfhqetjv.py" in x and "--kappa-max 0.2" in x and " --form RR " in x and "--rule-join" not in x and "--closing-table" not in x
        assert "--vis-gap-file %s --vis-closing" % want_gap[i] in x
        assert "/ckpt/UE_s%d/A/ckpt_last.pt" % k in x and "--preroll-steps 0" in x and "eval_grid_windows.json" in x
        assert "--qmap /workspace/v8proof/integ10k/qmap_sweep/QMAP_LR16_s%d_q50.json" % k in x and "--reader /workspace/v8proof/integ10k/lr16/TK16_s%d/ckpt_last.pt" % k in x
    for x in c[0:4]:
        assert "--readout causal_g" in x and "--acc-gain 0" in x and "--path-exec" in x and "--rule-every 5" in x
    for x in c[4:6]:
        assert "--readout pinned" in x and "--acc-gain" not in x and "--path-exec" not in x and "--rule-every" not in x
    assert "DRY DONE" in out and "GATE: needs /workspace/v8proof/boxfuse10k/lead_tests/lead_pytest_queue.out" in out
    assert "FORKSTATE:" in out and "make_margin" not in out and "--scales" not in out


def test_tags_and_env_lines_present_once_each_in_their_place():
    t = rd(Q)
    qm = 'QMAP_FILE="$W/qmap_sweep/QMAP_LR16_s${k}_q50.json"'
    k0 = "READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 " + qm
    cv = "READOUT=pinned KAPPA=0.2 " + qm
    lines = [
        k0 + ' VGAP="$GF/VGAPJE.npz" VCLOSE=1 rollout I10RQ50KF5Hb20gJEc RR "$k" "$k"',
        k0 + ' VGAP="$GF/VGAPJE90.npz" VCLOSE=1 rollout I10RQ50KF5Hb20gJA90 RR "$k" "$k"',
        cv + ' VGAP="$GF/VGAPJE.npz" VCLOSE=1 rollout I10RQ50Hb20gJEc RR "$k" "$k"',
    ]
    pos = []
    for ln in lines:
        assert t.count(ln) == 1, ln
        pos.append(t.index(ln))
    assert pos == sorted(pos)
    for tag in ("I10RQ50KF5Hb20gJEc", "I10RQ50KF5Hb20gJA90", "I10RQ50Hb20gJEc"):
        assert t.count(" rollout %s RR" % tag) == 1
    assert t.count(" rollout I10") == 3
    for nm in ("I10RLcKF5Hb20", "I10RQ50KF5Hb20", "I10RQ50KF5Hb20gGEc", "I10RQ50KF5Hb20gA90", "I10RLcHb20", "I10RQ50Hb20", "I10RQ50Hb20gGEc"):
        assert re.search(r"for nm in [^\n]*\b%s\b" % nm, t), nm


def test_marker_formats_names_and_stop_rule():
    t = rd(Q)
    assert '( set -C; echo "$(date +%s) $(ts) pid $$" > "$GP/BOXFUSE10K_START.txt" )' in t
    assert 'echo "ZZ BOXFUSE10K DONE $(ts) all steps done" > "$GP/BOXFUSE10K_DONE.txt"' in t
    assert '>> "$GP/BOXFUSE10K_STOP.txt"' in t and "GE=${BF_ROOT:-/workspace/v8proof/boxfuse10k}" in t and "MAXGPU=3" in t and "MINGB=${I10_MIN_FREE_GB:-28}" in t
    assert "STOPS=${BF_STOP_S:-4320}" in t and "LOG=$GP/boxfuse10k_queue.log" in t and "GF=${GF:-$GE/gapfiles}" in t
    for a in ("PRESATTR", "presattr", "PC_ROOT", "JLE", "$PJ", "gapmargin", "GAPMARGIN"):
        assert a not in t, a
    code = "\n".join(x for x in t.splitlines() if not x.lstrip().startswith("#"))
    assert not re.search(r"(^|[;&|]\s*|\s)(rm|mv|rmdir|kill|pkill)\s", code)


def step_one(tmp_path, rec, files, tamper=None):
    """Run the step_inputs function of the queue on a tiny synthetic gap folder; return (returncode, stdout, log text)."""
    gf = tmp_path / "gf"
    gf.mkdir(exist_ok=True)
    body = {}
    for nm, data in files.items():
        io.open(str(gf / (nm + ".npz")), "wb").write(data)
        body[nm] = {"sha256": hashlib.sha256(data).hexdigest()}
    if tamper:
        body[tamper] = {"sha256": "0" * 64}
    if rec is not None:
        r = dict(rec)
        r["files"] = body
        wr(gf / "BOXFUSE10K_GAPFILES_RECORD.json", json.dumps(r))
    m = re.search(r"^step_inputs\(\) \{.*?^\}\n", rd(Q), re.S | re.M)
    assert m
    log = tmp_path / "step.log"
    harness = tmp_path / "harness.sh"
    wr(harness, "\n".join([
        "DRY=0",
        'PY="%s"' % PYX,
        'GF="%s"' % fwd(gf),
        'BFREC="$GF/BOXFUSE10K_GAPFILES_RECORD.json"',
        'LOG="%s"' % fwd(log),
        "ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }",
        'log() { echo "$*" >> "$LOG"; }',
        'stop() { echo "STOP $*"; exit 3; }',
        m.group(0),
        "step_inputs",
        "echo STEP-ONE-RETURNED",
        ""]))
    r = run_bash(harness, tmp_path)
    return r, (rd(log) if log.exists() else "")


GOOD = {"pass_j0": True, "pass_j1": True, "pass_j2": True, "pass_j3": True, "pass_j4": True}
FILES = {"VGAPJE": b"tiny joined file", "VGAPJE90": b"tiny margin file"}


def test_step_one_admits_a_good_record(tmp_path):
    r, log = step_one(tmp_path, GOOD, FILES)
    assert r.returncode == 0 and "STEP-ONE-RETURNED" in r.stdout, r.stdout + r.stderr
    assert "ADMIT-OK gap files" in log


def test_step_one_refuses_a_failed_j3(tmp_path):
    r, _ = step_one(tmp_path, dict(GOOD, pass_j3=False), FILES)
    assert r.returncode == 3 and "STOP INPUT-ADMISSION PASS_J3-NOT-PASS" in r.stdout and "STEP-ONE-RETURNED" not in r.stdout


def test_step_one_refuses_each_pass_key_that_is_not_json_true(tmp_path):
    for i, k in enumerate(("pass_j0", "pass_j1", "pass_j2", "pass_j4")):
        d = tmp_path / ("c%d" % i)
        d.mkdir()
        r, _ = step_one(d, dict(GOOD, **{k: "true"}), FILES)
        assert r.returncode == 3 and "NOT-PASS" in r.stdout, k
    d = tmp_path / "missingkey"
    d.mkdir()
    rec = dict(GOOD)
    del rec["pass_j2"]
    r, _ = step_one(d, rec, FILES)
    assert r.returncode == 3 and "PASS_J2-NOT-PASS" in r.stdout


def test_step_one_refuses_a_missing_file_and_a_missing_record(tmp_path):
    d1 = tmp_path / "a"
    d1.mkdir()
    r, _ = step_one(d1, GOOD, {"VGAPJE": FILES["VGAPJE"]})
    assert r.returncode == 3 and "STOP NO-GAP-FILE" in r.stdout and "VGAPJE90.npz" in r.stdout
    d2 = tmp_path / "b"
    d2.mkdir()
    r, _ = step_one(d2, None, FILES)
    assert r.returncode == 3 and "STOP NO-GAP-RECORD" in r.stdout


def test_step_one_refuses_a_wrong_hash(tmp_path):
    for i, nm in enumerate(("VGAPJE", "VGAPJE90")):
        d = tmp_path / ("h%d" % i)
        d.mkdir()
        r, _ = step_one(d, GOOD, FILES, tamper=nm)
        assert r.returncode == 3 and "GAP-FILE-IS-NOT-THE-RECORDED-ONE " + nm in r.stdout


def fork_fragment():
    m = re.search(r"^# fork-state begin\n.*?^# fork-state end\n", rd(Q), re.S | re.M)
    assert m
    return m.group(0)


def fork_world(tmp_path, tail, state_env):
    harness = tmp_path / "fork.sh"
    wr(harness, "\n".join([
        "DRY=0",
        "ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }",
        'stop() { echo "STOP $*"; exit 3; }',
        fork_fragment(),
        tail,
        ""]))
    env = {"FORK_STATE_FILE": state_env} if state_env is not None else {}
    return run_bash(harness, tmp_path, env)


def test_busy_before_the_first_rollout_and_idle_on_every_exit_path(tmp_path):
    for i, (tail, rc) in enumerate((("exit 0", 0), ("stop SOMETHING", 3), ("exit 7", 7), ("kill -TERM $$; sleep 5", 143))):
        d = tmp_path / ("e%d" % i)
        d.mkdir()
        sf = d / "FORK_STATE.txt"
        wr(sf, "OLD LINE\n")
        r = fork_world(d, tail, fwd(sf))
        assert r.returncode == rc, (tail, r.returncode, r.stdout, r.stderr)
        got = rd(sf).splitlines()
        assert got[0] == "OLD LINE" and len(got) == 3, got
        assert re.match(r"^BUSY card ADD131 rollouts of the v8proof fork \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$", got[1])
        assert re.match(r"^IDLE \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$", got[2])


def test_empty_state_file_variable_writes_nothing(tmp_path):
    r = fork_world(tmp_path, "exit 0", "")
    assert r.returncode == 0 and not (tmp_path / "FORK_STATE.txt").exists()
    assert [p for p in os.listdir(str(tmp_path)) if p not in ("fork.sh",)] == []


def test_idle_is_written_only_after_a_busy_line(tmp_path):
    t = rd(Q)
    assert t.index("step_inputs\n# fork-state begin") > 0
    assert t.index("# fork-state begin") < t.index("rollout I10RQ50KF5Hb20gJEc RR")
    assert t.count("BUSY card ADD131 rollouts of the v8proof fork") == 2 and t.count("IDLE $(ts)") == 1


def test_gate_refuses_without_the_lead_test_output(tmp_path):
    r = run_bash(Q, tmp_path, {"BF_ROOT": fwd(tmp_path / "ge")})
    assert r.returncode == 4 and "REFUSED: step 0 gate" in r.stdout
    assert not (tmp_path / "ge").exists()


def test_wording_of_the_kit_files():
    parts = [("vo", "id"), ("super", "seded"), ("retr", "acted"), ("form", "erly"), ("correc", "tion"), ("with", "drawn"), ("no lon", "ger"), ("closed", " loop")]
    for fn in os.listdir(HERE):
        p = os.path.join(HERE, fn)
        if os.path.isfile(p) and fn.endswith((".py", ".sh", ".md")):
            low = rd(p).lower()
            for a, b in parts:
                assert (a + b) not in low, (fn, a + b)
