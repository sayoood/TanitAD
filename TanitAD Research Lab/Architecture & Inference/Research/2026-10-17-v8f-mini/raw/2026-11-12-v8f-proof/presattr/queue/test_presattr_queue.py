"""test_presattr_queue.py -- card ADD114 (PRESATTR10K), queue kit: the generated queue in DRY mode (creates nothing) and the planted-join tool on a synthetic world.
The gap tool of card ADD111 is taken from PRESATTR_KIT_GAP (pod) or from ../../gaperr/gap (dev box)."""
import io
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(HERE, "pod_presattr10k_queue.sh")
GEN = os.path.join(HERE, "make_presattr_queue.py")
KIT = os.environ.get("PRESATTR_KIT_GAP", os.path.join(HERE, "..", "..", "gaperr", "gap"))
BASH = shutil.which("bash")
sys.path.insert(0, HERE)
import make_planted_join_le as JL  # noqa: E402

G, KITP = JL.load_kit(KIT)


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read()


def cmds():
    r = subprocess.run([BASH, Q.replace("\\", "/")], capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, DRY="1"), timeout=120)
    assert r.returncode == 0, r.stdout[-500:]
    return [x for x in r.stdout.splitlines() if x.startswith("CMD: DISTRULE_DIR")], r.stdout


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
    assert names == ["I10RQ50KF5Hb20aLA_s0", "I10RQ50KF5Hb20aLA_s1", "I10RLcKF5Hb20pLE_s0", "I10RLcKF5Hb20pLE_s1", "I10RLcKF5P30Hb20pLE_s0", "I10RLcKF5P30Hb20pLE_s1"]
    for i, x in enumerate(c):
        k = i % 2
        assert "v8f_t1_eval_reader_rule_rzgfhqetjv.py" in x and "--kappa-max 0.2" in x and "--readout causal_g" in x and "--acc-gain 0" in x and "--path-exec" in x and "--rule-every 5" in x
        assert "/ckpt/UE_s%d/A/ckpt_last.pt" % k in x and "--closing-table" not in x
    for i, x in enumerate(c[0:2]):
        assert " --form RR " in x and "--vis-gap-file /workspace/v8proof/fallattr/gapfiles/VGAPA_LA.npz --vis-closing" in x and "--rule-join" not in x
        assert "--qmap /workspace/v8proof/integ10k/qmap_sweep/QMAP_LR16_s%d_q50.json" % i in x and "--reader /workspace/v8proof/integ10k/lr16/TK16_s%d/ckpt_last.pt" % i in x
        assert "--preroll-steps 0" in x and "eval_grid_windows.json" in x
    for x in c[2:6]:
        assert " --form RLc " in x and "--rule-join /workspace/v8proof/presattr/joins/JLE.npz" in x and "--vis-gap-file" not in x and "--vis-closing" not in x and "--reader " not in x and "--qmap" not in x
    for x in c[2:4]:
        assert "--preroll-steps 0" in x and "eval_grid_windows.json" in x
    for x in c[4:6]:
        assert "--preroll-steps 30" in x and "longroll/grid_p30.json" in x
    assert "DRY DONE" in out and "GATE: needs /workspace/v8proof/presattr/lead_tests/lead_pytest_queue.out" in out
    assert "make_planted_join_le.py --kit-gap /workspace/v8proof/gaperr/kit_gap --join /workspace/v8proof/hierrep/pins/join_eval141.npz --base /workspace/v8proof/cropscale/gapfiles/VGAPLE.npz" in out


def test_marker_formats_and_names():
    t = rd(Q)
    assert '( set -C; echo "$(date +%s) $(ts) pid $$" > "$GP/PRESATTR10K_START.txt" )' in t
    assert 'echo "ZZ PRESATTR10K DONE $(ts) all steps done" > "$GP/PRESATTR10K_DONE.txt"' in t
    assert '>> "$GP/PRESATTR10K_STOP.txt"' in t and "GE=${PA_ROOT:-/workspace/v8proof/presattr}" in t and "MAXGPU=3" in t and "MINGB=${I10_MIN_FREE_GB:-28}" in t
    for bad in ("CANDSET", "candset", "GAPERR10K", "step_base"):
        assert bad not in t
    code = "\n".join(x for x in t.splitlines() if not x.lstrip().startswith("#"))
    assert not re.search(r"(^|[;&|]\s*|\s)(rm|mv|rmdir)\s", code)
    for nm in ("I10RLcKF5Hb20", "I10RQ50KF5Hb20", "I10RQ50KF5Hb20gGEc", "I10RLcKF5P30Hb20", "I10RQ50KF5P30Hb20", "I10RQ50KF5P30Hb20gGEc", "I10RQ50KF5P30Hb20aLA"):
        assert re.search(r"for nm in [^\n]*\b%s\b" % nm, t)


def world(tmp_path):
    J = G.synthetic_join()
    B = G.synthetic_base(J)
    jp, bp = str(tmp_path / "join.npz"), str(tmp_path / "base.npz")
    np.savez(jp, **J)
    np.savez(bp, **B)
    return J, B, jp, bp


def test_planted_join_clean_world(tmp_path):
    J, B, jp, bp = world(tmp_path)
    rec = JL.build(G, KITP, jp, bp, str(tmp_path / "out"))
    assert rec["pass_e0"] is True and all(rec["clauses"].values())
    c = rec["counts"]
    assert c["n_replaced_rows"] > 0 and c["n_replaced_rows"] + c["n_label_rows_kept"] == c["n_label_rows"]
    S = G.load_npz(str(tmp_path / "out" / "JLE.npz"))
    assert sorted(S) == sorted(J)
    names, n, cum, idx, ci, fr = G.join_frames(J)
    g, lab = G.label_arrays(J, idx)
    bg, bh = G.lay_base(names, n, cum, B)
    mix = lab & bh & np.isfinite(bg)
    assert np.allclose(np.asarray(S["lead_gap"], np.float64)[idx[mix]], bg[mix], atol=1e-4)
    keep = np.ones(len(J["lead_gap"]), bool)
    keep[idx[mix]] = False
    assert np.array_equal(np.asarray(S["lead_gap"])[keep], np.asarray(J["lead_gap"])[keep], equal_nan=True)
    assert np.array_equal(S["lead_valid"], J["lead_valid"])
    rj = json.load(io.open(str(tmp_path / "out" / "JLE_RECORD.json"), encoding="utf-8"))
    assert rj["join"]["sha256"] == JL.sha_file(str(tmp_path / "out" / "JLE.npz")) and rj["eval_join_sha256"] == JL.sha_file(jp) and rj["base"]["sha256"] == JL.sha_file(bp)


def test_planted_join_faults_turn_e0_red(tmp_path):
    J, B, jp, bp = world(tmp_path)
    for fault in ("scale", "shift", "outside", "other_field"):
        rec = JL.build(G, KITP, jp, bp, str(tmp_path / ("out_" + fault)), fault)
        assert rec["pass_e0"] is False, fault


def test_planted_join_refuses_an_existing_output(tmp_path):
    J, B, jp, bp = world(tmp_path)
    JL.build(G, KITP, jp, bp, str(tmp_path / "out"))
    try:
        JL.build(G, KITP, jp, bp, str(tmp_path / "out"))
    except SystemExit as e:
        assert "REFUSED" in str(e)
    else:
        raise AssertionError("a second build into the same folder was not refused")


def test_selftest_runs(tmp_path):
    ok, res = JL.selftest(G, KITP, str(tmp_path / "st"))
    assert ok and res["clean"]["pass_e0"]


def test_wording_of_the_kit_files():
    parts = [("vo", "id"), ("super", "seded"), ("retr", "acted"), ("form", "erly"), ("correc", "tion"), ("with", "drawn"), ("no lon", "ger"), ("closed", " loop")]
    for fn in os.listdir(HERE):
        p = os.path.join(HERE, fn)
        if os.path.isfile(p) and fn.endswith((".py", ".sh", ".md")):
            low = rd(p).lower()
            for a, b in parts:
                assert (a + b) not in low, (fn, a + b)
