"""test_presattrc_queue.py -- card ADD115 (PRESATTRC10K), queue kit: the generated queue in DRY mode (creates nothing) and the wait script on a small stand-in world.
The source queue of card ADD114 is taken from ../../presattr/queue (same relative place on the dev box and in the pod stage)."""
import io
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(HERE, "pod_presattrc10k_queue.sh")
AFTER = os.path.join(HERE, "pod_presattrc_after.sh")
GEN = os.path.join(HERE, "make_presattrc_queue.py")
BASH = shutil.which("bash")


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read()


def fwd(p):
    return str(p).replace("\\", "/")


def cmds():
    r = subprocess.run([BASH, fwd(Q)], capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, DRY="1"), timeout=120)
    assert r.returncode == 0, r.stdout[-500:]
    return [x for x in r.stdout.splitlines() if x.startswith("CMD: DISTRULE_DIR")], r.stdout


def test_generated_file_is_what_the_generator_prints():
    r = subprocess.run([sys.executable, GEN, "--print"], capture_output=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.replace("\r\n", "\n") == rd(Q)


def test_lf_only_and_syntax():
    for p in (Q, AFTER):
        assert "\r" not in rd(p)
        r = subprocess.run([BASH, "-n", fwd(p)], capture_output=True, text=True, encoding="utf-8", timeout=60)
        assert r.returncode == 0, r.stderr


def test_four_commands_in_order_with_the_card_arguments():
    c, out = cmds()
    assert len(c) == 4
    names = [re.search(r"--tag (\S+)_m10_causal_heads", x).group(1) for x in c]
    assert names == ["I10RQ50Hb20aLA_s0", "I10RQ50Hb20aLA_s1", "I10RLcHb20pLE_s0", "I10RLcHb20pLE_s1"]
    for i, x in enumerate(c):
        k = i % 2
        assert "v8f_t1_eval_reader_rule_rzgfhqetjv.py" in x and "--kappa-max 0.2" in x and "--readout pinned" in x
        assert "--acc-gain" not in x and "--path-exec" not in x and "--rule-every" not in x and "--closing-table" not in x
        assert "/ckpt/UE_s%d/A/ckpt_last.pt" % k in x and "--preroll-steps 0" in x and "eval_grid_windows.json" in x
    for i, x in enumerate(c[0:2]):
        assert " --form RR " in x and "--vis-gap-file /workspace/v8proof/fallattr/gapfiles/VGAPA_LA.npz --vis-closing" in x and "--rule-join" not in x
        assert "--qmap /workspace/v8proof/integ10k/qmap_sweep/QMAP_LR16_s%d_q50.json" % i in x and "--reader /workspace/v8proof/integ10k/lr16/TK16_s%d/ckpt_last.pt" % i in x
    for x in c[2:4]:
        assert " --form RLc " in x and "--rule-join /workspace/v8proof/presattr/joins/JLE.npz" in x and "--vis-gap-file" not in x and "--vis-closing" not in x and "--reader " not in x and "--qmap" not in x
    assert "DRY DONE" in out and "GATE: needs /workspace/v8proof/presattrc/lead_tests/lead_pytest_queue.out" in out
    assert "CHECK: /workspace/v8proof/presattr/joins is the join folder of card ADD114; it has to exist and is not built here" in out
    assert "make_planted_join" not in out


def test_marker_formats_and_names():
    t = rd(Q)
    assert '( set -C; echo "$(date +%s) $(ts) pid $$" > "$GP/PRESATTRC10K_START.txt" )' in t
    assert 'echo "ZZ PRESATTRC10K DONE $(ts) all steps done" > "$GP/PRESATTRC10K_DONE.txt"' in t
    assert '>> "$GP/PRESATTRC10K_STOP.txt"' in t and "GE=${PC_ROOT:-/workspace/v8proof/presattrc}" in t and "MAXGPU=3" in t and "MINGB=${I10_MIN_FREE_GB:-28}" in t
    assert "STOPS=${PC_STOP_S:-7200}" in t and "PJ=${PC_JOINS:-/workspace/v8proof/presattr/joins}" in t
    assert '[ -d "$PJ" ] || stop "NO-JOIN-FOLDER-OF-CARD-ADD114 $PJ"' in t
    rolls = [x for x in t.splitlines() if " rollout I10" in x]
    assert len(rolls) == 2 and all("READOUT=pinned KAPPA=0.2 " in x and "EVERY" not in x and "PREROLL" not in x and "RGRID" not in x for x in rolls)
    for a in ("PRESATTR10K_", "PA_ROOT", "PA_KIT", "small_pa", "KF5"):
        assert a not in t, a
    for bad in (t, rd(AFTER)):
        code = "\n".join(x for x in bad.splitlines() if not x.lstrip().startswith("#"))
        assert not re.search(r"(^|[;&|]\s*|\s)(rm|mv|rmdir)\s", code)
    for nm in ("I10RLcHb20", "I10RQ50Hb20", "I10RQ50Hb20gGEc", "I10R0Hb20"):
        assert re.search(r"for nm in [^\n]*\b%s\b" % nm, t)


def after_world(tmp_path, marker):
    prev, ge, kit = tmp_path / "prev", tmp_path / "ge", tmp_path / "kit"
    for d in (prev, ge, kit):
        d.mkdir()
    io.open(str(kit / "pod_presattrc10k_queue.sh"), "w", encoding="utf-8", newline="\n").write('#!/usr/bin/env bash\necho ran > "$PC_ROOT/STUB_RAN.txt"\n')
    if marker:
        io.open(str(prev / marker), "w", encoding="utf-8", newline="\n").write("x\n")
    env = dict(os.environ, PC_PREV=fwd(prev), PC_ROOT=fwd(ge), PC_KIT=fwd(kit), PC_WAIT_S="0", PC_STEP_S="1")
    r = subprocess.run([BASH, fwd(AFTER)], capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
    return r, ge


def test_after_script_starts_the_queue_on_the_done_marker(tmp_path):
    r, ge = after_world(tmp_path, "PRESATTR10K_DONE.txt")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (ge / "STUB_RAN.txt").is_file() and not (ge / "PRESATTRC_AFTER_NOT_STARTED.txt").exists()
    lines = rd(str(ge / "PRESATTRC_AFTER_START.txt")).splitlines()
    assert len(lines) == 2 and re.match(r"^\d+ \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ pid \d+ waiting for ", lines[0]) and "the queue starts" in lines[1]


def test_after_script_starts_nothing_on_a_stop_marker(tmp_path):
    r, ge = after_world(tmp_path, "PRESATTR10K_STOP.txt")
    assert r.returncode == 5 and not (ge / "STUB_RAN.txt").exists() and (ge / "PRESATTRC_AFTER_NOT_STARTED.txt").is_file()


def test_after_script_starts_nothing_when_the_wait_runs_out(tmp_path):
    r, ge = after_world(tmp_path, "")
    assert r.returncode == 6 and not (ge / "STUB_RAN.txt").exists() and (ge / "PRESATTRC_AFTER_NOT_STARTED.txt").is_file()


def test_after_script_refuses_a_second_start(tmp_path):
    r, ge = after_world(tmp_path, "PRESATTR10K_DONE.txt")
    assert r.returncode == 0
    env = dict(os.environ, PC_PREV=fwd(tmp_path / "prev"), PC_ROOT=fwd(ge), PC_KIT=fwd(tmp_path / "kit"), PC_WAIT_S="0", PC_STEP_S="1")
    r2 = subprocess.run([BASH, fwd(AFTER)], capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
    assert r2.returncode == 2 and "REFUSED" in r2.stdout


def test_wording_of_the_kit_files():
    parts = [("vo", "id"), ("super", "seded"), ("retr", "acted"), ("form", "erly"), ("correc", "tion"), ("with", "drawn"), ("no lon", "ger"), ("closed", " loop")]
    for fn in os.listdir(HERE):
        p = os.path.join(HERE, fn)
        if os.path.isfile(p) and fn.endswith((".py", ".sh", ".md")):
            low = rd(p).lower()
            for a, b in parts:
                assert (a + b) not in low, (fn, a + b)
