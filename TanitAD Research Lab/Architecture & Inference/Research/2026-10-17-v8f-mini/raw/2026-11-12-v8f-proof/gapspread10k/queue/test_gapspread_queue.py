"""test_gapspread_queue.py -- card ADD134 (GAPSPREAD10K), queue kit: the generated queue in DRY mode (creates nothing), the gate, step 1 on tiny synthetic records and files,
the stop rule, the refusal of an existing run folder, the BUSY/IDLE fragment, and the four run lines against the banked A90 lines of the same model seed.
The banked lines are read from the DRY output of the queue of card ADD117 (model seeds 0 and 1, ../../gapmargin/queue) and of card ADD121 (model seeds 2 and 3, ../../repl/queue).
Child processes run in the folder named by the env GAPSPREAD_Q_TEST_CWD (default: the test's tmp_path)."""
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(HERE, "pod_gapspread10k_queue.sh")
GEN = os.path.join(HERE, "make_gapspread_queue.py")
BANK = {"01": os.path.join(HERE, "..", "..", "gapmargin", "queue", "pod_gapmargin10k_queue.sh"),
        "23": os.path.join(HERE, "..", "..", "repl", "queue", "pod_repl10k_queue.sh")}
BASH = shutil.which("bash")
PYX = sys.executable.replace("\\", "/")
ROOT = "/workspace/v8proof/gapspread10k"
NAME = "I10RQ50KF5Hb20gSP90"
A90 = "I10RQ50KF5Hb20gA90"


def rd(p):
    return io.open(str(p), encoding="utf-8", newline="").read()


def fwd(p):
    return str(p).replace("\\", "/")


def wr(p, txt):
    io.open(str(p), "w", encoding="utf-8", newline="\n").write(txt)


def cwd_of(tmp_path):
    return os.environ.get("GAPSPREAD_Q_TEST_CWD") or str(tmp_path)


def run_bash(script_path, tmp_path, env=None, timeout=120):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([BASH, fwd(script_path)], capture_output=True, text=True, encoding="utf-8", env=e, cwd=cwd_of(tmp_path), timeout=timeout)


def dry(path):
    r = subprocess.run([BASH, fwd(path)], capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, DRY="1"), timeout=120)
    assert r.returncode == 0, r.stdout[-500:]
    return [x for x in r.stdout.splitlines() if x.startswith("CMD: DISTRULE_DIR")], r.stdout


def cmds():
    return dry(Q)


def test_generated_file_equals_the_committed_one():
    r = subprocess.run([sys.executable, GEN, "--print"], capture_output=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.replace("\r\n", "\n") == rd(Q)


def test_lf_only_and_syntax():
    assert "\r" not in rd(Q)
    r = subprocess.run([BASH, "-n", fwd(Q)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr


def load_gen():
    spec = importlib.util.spec_from_file_location("make_gapspread_queue_mod", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generator_refuses_a_wrong_source_and_a_missing_text(tmp_path):
    mod = load_gen()
    src = io.open(mod.SRC, "rb").read().replace(b"\r\n", b"\n")
    bad = tmp_path / "src.sh"
    io.open(str(bad), "wb").write(src + b"# one more line\n")
    r = subprocess.run([sys.executable, GEN, "--src", str(bad), "--out", str(tmp_path / "o.sh")], capture_output=True, encoding="utf-8", timeout=60)
    assert r.returncode != 0 and "source sha256 prefix" in (r.stderr + r.stdout)
    assert not (tmp_path / "o.sh").exists()
    text = src.decode("utf-8")
    for a in ("STOPS=${BF_STOP_S:-4320}", "ARMS=(0 1); PS=\"\"", "gate_ok() { [ -f \"$GE/lead_tests/lead_pytest_queue.out\" ]; }\n"):
        assert text.count(a) == 1, a
        try:
            mod.build(text.replace(a, "# gone\n"))
        except SystemExit as e:
            assert "expected once, found 0" in str(e) or "found 0 times" in str(e) or "anchors" in str(e)
        else:
            raise AssertionError("no refusal for " + a)
    try:
        mod.build(text + "STOPS=${BF_STOP_S:-4320}\n")
    except SystemExit as e:
        assert "found 2" in str(e)
    else:
        raise AssertionError("no refusal for a doubled text")


def want_cmds_from_banked():
    """The banked A90 lines at K0 of model seeds 0 to 3, turned into the lines the new runs must have (tag and gap file exchanged)."""
    out = {}
    for seeds, path in (("01", BANK["01"]), ("23", BANK["23"])):
        c, _ = dry(path)
        for k in seeds:
            hit = [x for x in c if "--tag %s_s%s_m10_causal_heads" % (A90, k) in x]
            assert len(hit) == 1, (seeds, k, len(hit))
            out[k] = hit[0]
    return out


def test_four_run_lines_equal_the_banked_a90_lines_except_tag_and_gap_file():
    c, out = cmds()
    assert len(c) == 4
    banked = want_cmds_from_banked()
    old_gap = "/workspace/v8proof/gapmargin/gapfiles/VGAPLE_A90.npz"
    new_gap = ROOT + "/gapfiles/VGAPSP90.npz"
    for k in "0123":
        assert banked[k].count(A90) >= 3 and banked[k].count(old_gap) == 1, k
        want = banked[k].replace(A90, NAME).replace(old_gap, new_gap)
        got = [x for x in c if "--tag %s_s%s_m10_causal_heads" % (NAME, k) in x]
        assert got == [want], (k, got, want)
        assert "--vis-gap-file %s --vis-closing" % new_gap in got[0]
        assert "A90" not in got[0] and "VGAPLE" not in got[0]
    assert [re.search(r"--tag (\S+)_m10_causal_heads", x).group(1) for x in c] == ["%s_s%d" % (NAME, k) for k in range(4)]


def test_seed_mapping_of_checkpoint_reader_and_map():
    c, out = cmds()
    for k in range(4):
        r = k - 2 if k >= 2 else k
        x = c[k]
        assert "--ckpt /workspace/v8proof/integ10k/ckpt/UE_s%d/A/ckpt_last.pt" % k in x
        assert "--reader /workspace/v8proof/integ10k/lr16/TK16_s%d/ckpt_last.pt" % r in x
        assert "--qmap /workspace/v8proof/integ10k/qmap_sweep/QMAP_LR16_s%d_q50.json" % r in x
        assert "--out-dir /workspace/v8proof/integ10k/t1/%s_s%d " % (NAME, k) in x
        assert "--readout causal_g" in x and "--acc-gain 0" in x and "--path-exec" in x and "--rule-every 5" in x and "--preroll-steps 0" in x
        assert "--kappa-max 0.2" in x and " --form RR " in x and "v8f_t1_eval_reader_rule_rzgfhqetjv.py" in x and "eval_grid_windows.json" in x
    assert "DRY DONE" in out and "GATE: needs %s/lead_tests/lead_pytest_queue.out" % ROOT in out and "FORKSTATE:" in out


def test_marker_formats_names_and_stop_rule_text():
    t = rd(Q)
    assert '( set -C; echo "$(date +%s) $(ts) pid $$" > "$GP/GAPSPREAD10K_START.txt" )' in t
    assert 'echo "ZZ GAPSPREAD10K DONE $(ts) all steps done" > "$GP/GAPSPREAD10K_DONE.txt"' in t
    assert '>> "$GP/GAPSPREAD10K_STOP.txt"' in t and "GE=${GS_ROOT:-%s}" % ROOT in t and "MAXGPU=3" in t and "MINGB=${I10_MIN_FREE_GB:-28}" in t
    assert "STOPS=${GS_STOP_S:-3600}" in t and "LOG=$GP/gapspread10k_queue.log" in t and "GF=${GF:-$GE/gapfiles}" in t and "ARMS=(0 1 2 3)" in t
    assert t.count(" rollout I10") == 1 and t.count("VCLOSE=1 rollout " + NAME + ' RR "$k" "$r"') == 1
    for a in ("boxfuse", "BOXFUSE", "BF_", "JEc", "JA90", "VGAPJE"):
        assert a not in t, a
    code = "\n".join(x for x in t.splitlines() if not x.lstrip().startswith("#"))
    assert not re.search(r"(^|[;&|]\s*|\s)(rm|mv|rmdir|kill|pkill)\s", code)


def test_dry_mode_creates_nothing(tmp_path):
    ge = tmp_path / "ge"
    w = tmp_path / "w"
    r = run_bash(Q, tmp_path, {"DRY": "1", "GS_ROOT": fwd(ge), "I10_W": fwd(w), "FORK_STATE_FILE": fwd(tmp_path / "fs.txt")})
    assert r.returncode == 0 and "DRY DONE" in r.stdout and r.stdout.count("CMD: ") == 4, r.stdout[-400:] + r.stderr[-300:]
    assert not ge.exists() and not w.exists() and not (tmp_path / "fs.txt").exists()


def gate_world(tmp_path, content):
    ge = tmp_path / "ge"
    if content is not None:
        (ge / "lead_tests").mkdir(parents=True)
        io.open(str(ge / "lead_tests" / "lead_pytest_queue.out"), "wb").write(content)
    return ge, run_bash(Q, tmp_path, {"GS_ROOT": fwd(ge), "I10_W": fwd(tmp_path / "nowork")})


def test_gate_refuses_without_the_lead_test_output(tmp_path):
    ge, r = gate_world(tmp_path, None)
    lines = [x for x in r.stdout.splitlines() if x.startswith("GATE")]
    assert r.returncode == 4 and len(lines) == 1 and fwd(ge) + "/lead_tests/lead_pytest_queue.out" in lines[0] and "missing" in lines[0]
    assert not (ge / "GAPSPREAD10K_START.txt").exists()


def test_gate_refuses_an_output_that_does_not_end_in_a_pass_line(tmp_path):
    for i, body in enumerate((b"3 failed, 5 passed in 2s\n", b"1 passed, 1 error in 2s\n", b"no tests ran in 0.1s\n", b"", b"15 passed in 3s\nsome later line\n")):
        d = tmp_path / ("g%d" % i)
        d.mkdir()
        ge, r = gate_world(d, body)
        assert r.returncode == 4 and any(x.startswith("GATE") for x in r.stdout.splitlines()), (body, r.stdout)


def test_gate_admits_pass_lines_and_stops_later_at_the_missing_work_folder(tmp_path):
    for i, body in enumerate((b"15 passed in 3.37s\n", b"15 passed in 3.37s\r\n\r\n\n", b"============ 15 passed in 3.37s ============\n", b"..\n  \n12 passed, 1 warning in 1s\n")):
        d = tmp_path / ("p%d" % i)
        d.mkdir()
        ge, r = gate_world(d, body)
        assert r.returncode == 2 and "REFUSED: no " in r.stdout and not any(x.startswith("GATE") for x in r.stdout.splitlines()), (body, r.stdout)


def step_one(tmp_path, rec, files, tamper=None):
    """Run the step_inputs function of the queue on a tiny synthetic gap folder; return (result, log text)."""
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
        wr(gf / "GAPSPREAD10K_GAPFILE_RECORD.json", json.dumps(r))
    m = re.search(r"^step_inputs\(\) \{.*?^\}\n", rd(Q), re.S | re.M)
    assert m
    log = tmp_path / "step.log"
    harness = tmp_path / "harness.sh"
    wr(harness, "\n".join([
        "DRY=0",
        'PY="%s"' % PYX,
        'GF="%s"' % fwd(gf),
        'GSREC="$GF/GAPSPREAD10K_GAPFILE_RECORD.json"',
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


GOOD = {"card": "PREREG_V8F_PROOF_ADD134", "pass_p0": True}
FILES = {"VGAPSP90": b"tiny spread file"}


def test_step_one_admits_a_good_record(tmp_path):
    r, log = step_one(tmp_path, GOOD, FILES)
    assert r.returncode == 0 and "STEP-ONE-RETURNED" in r.stdout, r.stdout + r.stderr
    assert "ADMIT-OK gap file" in log


def test_step_one_refuses_a_pass_p0_that_is_not_json_true(tmp_path):
    for i, v in enumerate((False, "true", 1, None)):
        d = tmp_path / ("c%d" % i)
        d.mkdir()
        r, _ = step_one(d, dict(GOOD, pass_p0=v), FILES)
        assert r.returncode == 3 and "STOP INPUT-ADMISSION PASS_P0-NOT-PASS" in r.stdout and "STEP-ONE-RETURNED" not in r.stdout, v
    d = tmp_path / "missingkey"
    d.mkdir()
    r, _ = step_one(d, {"card": GOOD["card"]}, FILES)
    assert r.returncode == 3 and "PASS_P0-NOT-PASS" in r.stdout


def test_step_one_refuses_a_record_of_another_card(tmp_path):
    for i, v in enumerate(("PREREG_V8F_PROOF_ADD133", None, "prereg_v8f_proof_add134")):
        d = tmp_path / ("k%d" % i)
        d.mkdir()
        rec = dict(GOOD, card=v)
        if v is None:
            del rec["card"]
        r, _ = step_one(d, rec, FILES)
        assert r.returncode == 3 and "RECORD-IS-NOT-OF-THIS-CARD" in r.stdout, v


def test_step_one_refuses_a_missing_file_and_a_missing_record(tmp_path):
    d1 = tmp_path / "a"
    d1.mkdir()
    r, _ = step_one(d1, GOOD, {})
    assert r.returncode == 3 and "STOP NO-GAP-FILE" in r.stdout and "VGAPSP90.npz" in r.stdout
    d2 = tmp_path / "b"
    d2.mkdir()
    r, _ = step_one(d2, None, FILES)
    assert r.returncode == 3 and "STOP NO-GAP-RECORD" in r.stdout


def test_step_one_refuses_a_wrong_or_absent_hash(tmp_path):
    d = tmp_path / "h0"
    d.mkdir()
    r, _ = step_one(d, GOOD, FILES, tamper="VGAPSP90")
    assert r.returncode == 3 and "GAP-FILE-IS-NOT-THE-RECORDED-ONE VGAPSP90" in r.stdout
    d = tmp_path / "h1"
    d.mkdir()
    r, _ = step_one(d, GOOD, {"VGAPSP90": b"x"})
    assert r.returncode == 0
    gf = d / "gf"
    io.open(str(gf / "VGAPSP90.npz"), "wb").write(b"changed after the record")
    m = re.search(r"^step_inputs\(\) \{.*?^\}\n", rd(Q), re.S | re.M)
    harness = d / "again.sh"
    wr(harness, "\n".join(["DRY=0", 'PY="%s"' % PYX, 'GF="%s"' % fwd(gf), 'GSREC="$GF/GAPSPREAD10K_GAPFILE_RECORD.json"', 'log() { :; }',
                           'stop() { echo "STOP $*"; exit 3; }', m.group(0), "step_inputs", "echo RETURNED", ""]))
    r = run_bash(harness, d)
    assert r.returncode == 3 and "GAP-FILE-IS-NOT-THE-RECORDED-ONE" in r.stdout
    # an entry under another name does not stand in for VGAPSP90
    d = tmp_path / "h2"
    d.mkdir()
    r, _ = step_one(d, GOOD, {"VGAPSP90": b"y", "VGAPOTHER": b"z"}, tamper="VGAPSP90")
    assert r.returncode == 3


def stoprule_world(tmp_path, start_line, stops):
    ge = tmp_path / "ge"
    ge.mkdir(exist_ok=True)
    if start_line is not None:
        wr(ge / "GAPSPREAD10K_START.txt", start_line + "\n")
    m = re.search(r"^stop\(\) \{.*?\n", rd(Q), re.M)
    m2 = re.search(r"^check_time\(\) \{.*?^\}\n", rd(Q), re.S | re.M)
    assert m and m2
    harness = tmp_path / "stoprule.sh"
    wr(harness, "\n".join(["DRY=0", 'GP="%s"' % fwd(ge), "STOPS=%d" % stops, "ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }", "log() { :; }", m.group(0), m2.group(0),
                           'check_time "rollout X"', "echo RULE-RETURNED", ""]))
    return ge, run_bash(harness, tmp_path)


def test_stop_rule_passes_inside_the_limit(tmp_path):
    import time
    ge, r = stoprule_world(tmp_path, "%d 2026-11-23T10:00:00Z pid 1" % (int(time.time()) - 100), 3600)
    assert r.returncode == 0 and "RULE-RETURNED" in r.stdout and not (ge / "GAPSPREAD10K_STOP.txt").exists()


def test_stop_rule_stops_after_the_limit_and_writes_the_marker(tmp_path):
    import time
    ge, r = stoprule_world(tmp_path, "%d 2026-11-23T10:00:00Z pid 1" % (int(time.time()) - 3700), 3600)
    assert r.returncode == 3 and "RULE-RETURNED" not in r.stdout
    assert "STOP-RULE" in rd(ge / "GAPSPREAD10K_STOP.txt") and "limit 3600 s" in rd(ge / "GAPSPREAD10K_STOP.txt")


def test_stop_rule_stops_on_an_unreadable_start_line(tmp_path):
    ge, r = stoprule_world(tmp_path, "garbage line", 3600)
    assert r.returncode == 3 and "START-LINE-UNREADABLE" in rd(ge / "GAPSPREAD10K_STOP.txt")
    d = tmp_path / "nostart"
    d.mkdir()
    ge, r = stoprule_world(d, None, 3600)
    assert r.returncode == 3 and "START-LINE-UNREADABLE" in rd(ge / "GAPSPREAD10K_STOP.txt")


def test_a_run_folder_that_exists_is_refused(tmp_path):
    w = tmp_path / "w"
    (w / "t1" / (NAME + "_s2")).mkdir(parents=True)
    m = re.search(r"^rollout\(\) \{.*?^\}\n", rd(Q), re.S | re.M)
    assert m
    harness = tmp_path / "roll.sh"
    wr(harness, "\n".join(["DRY=0", 'W="%s"' % fwd(w), 'GP="%s"' % fwd(tmp_path / "gp"), 'GRID=g.json', 'WRAP=w.py', 'PY=py',
                           "ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }",
                           "done_step() { return 1; }", "check_time() { echo CHECKED; }", "wait_slot() { echo WAITED; }",
                           'stop() { echo "STOP $*"; exit 3; }', m.group(0),
                           'READOUT=causal_g VGAP=x VCLOSE=1 QMAP_FILE=q rollout %s RR 2 0' % NAME, "echo ROLLOUT-RETURNED", ""]))
    r = run_bash(harness, tmp_path)
    assert r.returncode == 3 and "STOP OUTPUT-EXISTS" in r.stdout and "CHECKED" not in r.stdout and "WAITED" not in r.stdout and "ROLLOUT-RETURNED" not in r.stdout
    assert not (w / "t1" / (NAME + "_s2") / "ticks").exists()


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
        assert re.match(r"^BUSY card ADD134 rollouts of the v8proof fork \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$", got[1])
        assert re.match(r"^IDLE \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$", got[2])


def test_empty_state_file_variable_writes_nothing(tmp_path):
    r = fork_world(tmp_path, "exit 0", "")
    assert r.returncode == 0 and not (tmp_path / "FORK_STATE.txt").exists()
    assert [p for p in os.listdir(str(tmp_path)) if p not in ("fork.sh",)] == []


def test_busy_comes_after_step_one_and_before_the_first_rollout():
    t = rd(Q)
    assert t.index("step_inputs\n# fork-state begin") > 0
    assert t.index("# fork-state begin") < t.index("rollout " + NAME + " RR")
    assert t.count("BUSY card ADD134 rollouts of the v8proof fork") == 2 and t.count("IDLE $(ts)") == 1


def test_wording_of_the_kit_files():
    parts = [("vo", "id"), ("super", "seded"), ("retr", "acted"), ("form", "erly"), ("correc", "tion"), ("with", "drawn"), ("no lon", "ger"), ("closed", " loop")]
    for fn in os.listdir(HERE):
        p = os.path.join(HERE, fn)
        if os.path.isfile(p) and fn.endswith((".py", ".sh", ".md")):
            low = rd(p).lower()
            for a, b in parts:
                assert (a + b) not in low, (fn, a + b)
