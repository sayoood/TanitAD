"""test_gapmargin_queue.py -- card ADD117 (GAPMARGIN10K), queue kit (lead-written): the generated queue in DRY mode (creates nothing), the wait script on a small stand-in
world, and the gap-file tool on a synthetic base file. The source queue of card ADD115 is taken from ../../presattrc/queue (same relative place in the pod stage)."""
import io
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(HERE, "pod_gapmargin10k_queue.sh")
AFTER = os.path.join(HERE, "pod_gapmargin_after.sh")
GEN = os.path.join(HERE, "make_gapmargin_queue.py")
BASH = shutil.which("bash")
sys.path.insert(0, HERE)
import make_margin_gapfiles as MG  # noqa: E402


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


def test_eight_commands_in_order_with_the_card_arguments():
    c, out = cmds()
    assert len(c) == 8
    names = [re.search(r"--tag (\S+)_m10_causal_heads", x).group(1) for x in c]
    assert names == ["I10RQ50KF5Hb20gA90_s0", "I10RQ50KF5Hb20gA90_s1", "I10RQ50KF5Hb20gA80_s0", "I10RQ50KF5Hb20gA80_s1",
                     "I10RQ50Hb20gA90_s0", "I10RQ50Hb20gA90_s1", "I10RQ50Hb20gA80_s0", "I10RQ50Hb20gA80_s1"]
    for i, x in enumerate(c):
        k = i % 2
        a = "A90" if (i // 2) % 2 == 0 else "A80"
        assert "v8f_t1_eval_reader_rule_rzgfhqetjv.py" in x and "--kappa-max 0.2" in x and " --form RR " in x and "--rule-join" not in x and "--closing-table" not in x
        assert "--vis-gap-file /workspace/v8proof/gapmargin/gapfiles/VGAPLE_%s.npz --vis-closing" % a in x
        assert "/ckpt/UE_s%d/A/ckpt_last.pt" % k in x and "--preroll-steps 0" in x and "eval_grid_windows.json" in x
        assert "--qmap /workspace/v8proof/integ10k/qmap_sweep/QMAP_LR16_s%d_q50.json" % k in x and "--reader /workspace/v8proof/integ10k/lr16/TK16_s%d/ckpt_last.pt" % k in x
    for x in c[0:4]:
        assert "--readout causal_g" in x and "--acc-gain 0" in x and "--path-exec" in x and "--rule-every 5" in x
    for x in c[4:8]:
        assert "--readout pinned" in x and "--acc-gain" not in x and "--path-exec" not in x and "--rule-every" not in x
    assert "DRY DONE" in out and "GATE: needs /workspace/v8proof/gapmargin/lead_tests/lead_pytest_queue.out" in out
    assert "make_margin_gapfiles.py --base /workspace/v8proof/cropscale/gapfiles/VGAPLE.npz --out-dir /workspace/v8proof/gapmargin/gapfiles --scales 0.90,0.80" in out


def test_marker_formats_and_names():
    t = rd(Q)
    assert '( set -C; echo "$(date +%s) $(ts) pid $$" > "$GP/GAPMARGIN10K_START.txt" )' in t
    assert 'echo "ZZ GAPMARGIN10K DONE $(ts) all steps done" > "$GP/GAPMARGIN10K_DONE.txt"' in t
    assert '>> "$GP/GAPMARGIN10K_STOP.txt"' in t and "GE=${GM_ROOT:-/workspace/v8proof/gapmargin}" in t and "MAXGPU=3" in t and "MINGB=${I10_MIN_FREE_GB:-28}" in t
    assert "STOPS=${GM_STOP_S:-10800}" in t and "GMREC=$GF/GAPMARGIN_GAPFILES_RECORD.json" in t
    for a in ("PRESATTR", "presattr", "PC_ROOT", "JLE", "$PJ"):
        assert a not in t, a
    for txt in (t, rd(AFTER)):
        code = "\n".join(x for x in txt.splitlines() if not x.lstrip().startswith("#"))
        assert not re.search(r"(^|[;&|]\s*|\s)(rm|mv|rmdir)\s", code)
    for nm in ("I10RLcKF5Hb20", "I10RQ50KF5Hb20", "I10RQ50KF5Hb20gGEc", "I10RLcHb20", "I10RQ50Hb20", "I10RQ50Hb20gGEc", "I10R0Hb20"):
        assert re.search(r"for nm in [^\n]*\b%s\b" % nm, t)


def after_world(tmp_path, marker):
    prev, ge, kit = tmp_path / "prev", tmp_path / "ge", tmp_path / "kit"
    for d in (prev, ge, kit):
        d.mkdir()
    io.open(str(kit / "pod_gapmargin10k_queue.sh"), "w", encoding="utf-8", newline="\n").write('#!/usr/bin/env bash\necho ran > "$GM_ROOT/STUB_RAN.txt"\n')
    if marker:
        io.open(str(prev / marker), "w", encoding="utf-8", newline="\n").write("x\n")
    env = dict(os.environ, GM_PREV=fwd(prev), GM_ROOT=fwd(ge), GM_KIT=fwd(kit), GM_WAIT_S="0", GM_STEP_S="1")
    r = subprocess.run([BASH, fwd(AFTER)], capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
    return r, ge, env


def test_after_script_starts_the_queue_on_the_done_marker_and_refuses_a_second_start(tmp_path):
    r, ge, env = after_world(tmp_path, "PRESATTRC10K_DONE.txt")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (ge / "STUB_RAN.txt").is_file() and not (ge / "GAPMARGIN_AFTER_NOT_STARTED.txt").exists()
    lines = rd(str(ge / "GAPMARGIN_AFTER_START.txt")).splitlines()
    assert len(lines) == 2 and re.match(r"^\d+ \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ pid \d+ waiting for ", lines[0]) and "the queue starts" in lines[1]
    r2 = subprocess.run([BASH, fwd(AFTER)], capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
    assert r2.returncode == 2 and "REFUSED" in r2.stdout


def test_after_script_starts_nothing_on_a_stop_marker(tmp_path):
    r, ge, _ = after_world(tmp_path, "PRESATTRC10K_STOP.txt")
    assert r.returncode == 5 and not (ge / "STUB_RAN.txt").exists() and (ge / "GAPMARGIN_AFTER_NOT_STARTED.txt").is_file()


def test_after_script_starts_nothing_when_the_wait_runs_out(tmp_path):
    r, ge, _ = after_world(tmp_path, "")
    assert r.returncode == 6 and not (ge / "STUB_RAN.txt").exists() and (ge / "GAPMARGIN_AFTER_NOT_STARTED.txt").is_file()


def base_file(tmp_path):
    B = MG.synthetic_base()
    bp = str(tmp_path / "base.npz")
    np.savez(bp, **B)
    return B, bp


def test_scaled_files_on_a_clean_base(tmp_path):
    B, bp = base_file(tmp_path)
    rec = MG.build(bp, str(tmp_path / "out"), (0.90, 0.80))
    assert rec["pass_k0"] is True and rec["scale_one_gives_the_base_arrays"] is True and sorted(rec["files"]) == ["VGAPLE_A80", "VGAPLE_A90"]
    has = B["has_out"]
    for nm, a in (("VGAPLE_A90", 0.90), ("VGAPLE_A80", 0.80)):
        S = MG.load(str(tmp_path / "out" / (nm + ".npz")))
        assert sorted(S) == sorted(B) and np.allclose(S["gap"][has], a * B["gap"][has], atol=1e-12, rtol=0) and np.isnan(S["gap"][~has]).all()
        for k in B:
            if k != "gap":
                assert np.array_equal(S[k], B[k])
        assert all(rec["clauses"][nm].values())
        want = int((has & (B["gap"] > 60.0) & (a * B["gap"] <= 60.0)).sum())
        assert rec["files"][nm]["n_rows_base_gap_above_60_and_scaled_at_most_60"] == want and want > 0
        assert rec["files"][nm]["sha256"] == MG.sha_file(str(tmp_path / "out" / (nm + ".npz")))
    rj = json.load(io.open(str(tmp_path / "out" / "GAPMARGIN_GAPFILES_RECORD.json"), encoding="utf-8"))
    assert rj["pass_k0"] is True and rj["base"]["sha256"] == MG.sha_file(bp) and rj["card"] == "PREREG_V8F_PROOF_ADD117"


def test_scaled_files_faults_turn_k0_red(tmp_path):
    B, bp = base_file(tmp_path)
    for fault in ("shift", "nan_filled", "other_field"):
        rec = MG.build(bp, str(tmp_path / ("out_" + fault)), (0.90, 0.80), fault)
        assert rec["pass_k0"] is False, fault


def test_scaled_files_refuse_an_existing_output_and_a_scale_out_of_range(tmp_path):
    B, bp = base_file(tmp_path)
    MG.build(bp, str(tmp_path / "out"), (0.90,))
    try:
        MG.build(bp, str(tmp_path / "out"), (0.90,))
    except SystemExit as e:
        assert "REFUSED" in str(e)
    else:
        raise AssertionError("a second build into the same folder was not refused")
    assert MG.main(["--base", bp, "--out-dir", str(tmp_path / "o2"), "--scales", "1.2"]) == 2
    assert MG.tag(0.90) == "A90" and MG.tag(0.80) == "A80"


def test_selftest_runs(tmp_path):
    assert MG.selftest(str(tmp_path / "st")) == 0


def test_wording_of_the_kit_files():
    parts = [("vo", "id"), ("super", "seded"), ("retr", "acted"), ("form", "erly"), ("correc", "tion"), ("with", "drawn"), ("no lon", "ger"), ("closed", " loop")]
    for fn in os.listdir(HERE):
        p = os.path.join(HERE, fn)
        if os.path.isfile(p) and fn.endswith((".py", ".sh", ".md")):
            low = rd(p).lower()
            for a, b in parts:
                assert (a + b) not in low, (fn, a + b)
