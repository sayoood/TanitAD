"""test_candset_queue.py -- card ADD113 (CANDSET10K), queue kit. Runs the generator and the generated queue in DRY mode (creates nothing on a pod)."""
import io
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(HERE, "pod_candset10k_queue.sh")
GEN = os.path.join(HERE, "make_candset_queue.py")
BASH = shutil.which("bash")


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read()


def dry():
    env = dict(os.environ, DRY="1")
    r = subprocess.run([BASH, Q.replace("\\", "/")], capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
    return r.returncode, r.stdout


def cmds():
    rc, out = dry()
    assert rc == 0, out[-500:]
    return [x for x in out.splitlines() if x.startswith("CMD:")], out


def test_generated_file_is_what_the_generator_prints():
    r = subprocess.run([sys.executable, GEN, "--print"], capture_output=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.replace("\r\n", "\n") == rd(Q)


def test_lf_only_and_syntax():
    assert "\r" not in rd(Q)
    r = subprocess.run([BASH, "-n", Q.replace("\\", "/")], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr


def test_six_commands_in_order_with_the_card_arguments():
    c, out = cmds()
    assert len(c) == 6
    names = [re.search(r"--tag (\S+)_m10_causal_heads", x).group(1) for x in c]
    assert names == ["I10RQ50Hb20gGEc_s0", "I10RQ50Hb20gGEc_s1", "I10RQ50KF5Hb20gGEc_s0", "I10RQ50KF5Hb20gGEc_s1", "I10RQ50KF5P60Hb20gGEc_s0", "I10RQ50KF5P60Hb20gGEc_s1"]
    for i, x in enumerate(c):
        k = i % 2
        assert " --form RR " in x and "v8f_t1_eval_reader_rule_rzgfhqetjv.py" in x
        assert "--vis-gap-file /workspace/v8proof/cropscale/gapfiles/VGAPLE.npz --vis-closing" in x
        assert "--kappa-max 0.2" in x
        assert "--qmap /workspace/v8proof/integ10k/qmap_sweep/QMAP_LR16_s%d_q50.json" % k in x
        assert "--reader /workspace/v8proof/integ10k/lr16/TK16_s%d/ckpt_last.pt" % k in x
        assert "/ckpt/UE_s%d/A/ckpt_last.pt" % k in x
        assert "--rule-join" not in x and "--closing-table" not in x
    for x in c[0:2]:
        assert "--readout pinned" in x and "--preroll-steps 0" in x and "--rule-every" not in x and "--path-exec" not in x and "eval_grid_windows.json" in x
    for x in c[2:4]:
        assert "--readout causal_g" in x and "--acc-gain 0" in x and "--path-exec" in x and "--rule-every 5" in x and "--preroll-steps 0" in x and "eval_grid_windows.json" in x
    for x in c[4:6]:
        assert "--readout causal_g" in x and "--acc-gain 0" in x and "--path-exec" in x and "--rule-every 5" in x and "--preroll-steps 60" in x and "longroll/grid_p60.json" in x
    assert "DRY DONE" in out and "GATE: needs /workspace/v8proof/candset/lead_tests/lead_pytest_queue.out" in out


def test_marker_formats_are_those_of_the_source_queue():
    t = rd(Q)
    assert '( set -C; echo "$(date +%s) $(ts) pid $$" > "$GP/CANDSET10K_START.txt" )' in t
    assert 'echo "ZZ CANDSET10K DONE $(ts) all steps done" > "$GP/CANDSET10K_DONE.txt"' in t
    assert '>> "$GP/CANDSET10K_STOP.txt"' in t
    assert "GE=${CS_ROOT:-/workspace/v8proof/candset}" in t and "STOPS=${CS_STOP_S:-10800}" in t and "MAXGPU=3" in t and "MINGB=${I10_MIN_FREE_GB:-28}" in t


def test_nothing_of_the_source_card_is_left_and_nothing_is_removed():
    t = rd(Q)
    # the source's variable lines KG, KR and LAF stay in the file and are used nowhere: taken out before the check, as the generator does
    unused = ("KG=$GE/kit_gap\n", "KR=$GE/kit_read\n", "LAF=/workspace/v8proof/fallattr/gapfiles/VGAPA_LA.npz\n")
    for u in unused:
        assert t.count(u) == 1
        t = t.replace(u, "")
    for bad in ("GAPERR", "gaperr_queue", "VGAPA_", "make_gaperr", "step_gapfiles", "$KG", "$KR", "$LAF"):
        assert bad not in t
    code = "\n".join(x for x in t.splitlines() if not x.lstrip().startswith("#"))
    assert not re.search(r"(^|[;&|]\s*|\s)(rm|mv|rmdir)\s", code)
    assert t.count(" rollout I10") == 3 and 'sha256sum "$BASEF" | cut -c1-8); [ "$got" = "55de2f7c" ]' in t


def test_banked_runs_are_checked():
    t = rd(Q)
    for nm in ("I10RQ50Hb20", "I10RLcHb20", "I10R0Hb20", "I10RQ50KF5Hb20", "I10RLcKF5Hb20", "I10R0KHb20", "I10RQ50KF5P60Hb20", "I10RLcKF5P60Hb20", "I10R0KP60Hb20", "I10RQ50KF5P30Hb20gGEc"):
        assert re.search(r"for nm in [^\n]*\b%s\b" % nm, t)


def test_wording_of_the_kit_files():
    parts = [("vo", "id"), ("super", "seded"), ("retr", "acted"), ("form", "erly"), ("correc", "tion"), ("with", "drawn"), ("no lon", "ger"), ("closed", " loop")]
    for fn in os.listdir(HERE):
        p = os.path.join(HERE, fn)
        if os.path.isfile(p) and fn.endswith((".py", ".sh", ".md")):
            low = rd(p).lower()
            for a, b in parts:
                assert (a + b) not in low, (fn, a + b)
