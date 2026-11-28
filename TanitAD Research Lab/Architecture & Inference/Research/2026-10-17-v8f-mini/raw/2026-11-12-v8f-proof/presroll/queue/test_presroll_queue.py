"""test_presroll_queue.py -- card ADD120 (PRESROLL10K): the generated queue against its generator, the queue's dry run (nine rollouts, the identity run first and its compare
before any other rollout), the first line of the run text, and the rule log compare of the identity step on tiny arrays. The queue of card ADD117 is taken from
../../gapmargin/queue (same relative place in the pod stage). No pod result file is opened."""
import io
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BASH = shutil.which("bash")
sys.path.insert(0, HERE)
import make_presroll_queue as MK  # noqa: E402

QUEUE = os.path.join(HERE, MK.OUT_NAME)
PD = "/workspace/v8proof/preshead2/dump"
T1 = "/workspace/v8proof/integ10k/t1"


def rd(p):
    return io.open(p, "rb").read().replace(b"\r\n", b"\n").decode("utf-8")


def test_generated_file_equals_the_generator_output(tmp_path):
    assert rd(QUEUE) == MK.make()
    assert MK.main(["--out-dir", str(tmp_path)]) == 0
    with pytest.raises(FileExistsError):
        MK.main(["--out-dir", str(tmp_path)])


def test_generator_refuses_a_changed_source(monkeypatch):
    monkeypatch.setattr(MK, "SOURCE", (MK.SOURCE[0], "0" * 16))
    with pytest.raises(SystemExit):
        MK.make()


def dry(tmp_path):
    env = dict(os.environ)
    env.update({"DRY": "1", "PR_ROOT": str(tmp_path / "root").replace(os.sep, "/")})
    r = subprocess.run([BASH, QUEUE.replace(os.sep, "/")], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert not os.path.exists(str(tmp_path / "root"))
    return r


def test_queue_dry_run_prints_the_nine_rollouts_in_order(tmp_path):
    r = dry(tmp_path)
    out = r.stdout.splitlines()
    idx = [i for i, x in enumerate(out) if x.startswith("CMD: DISTRULE_DIR")]
    cmds = [out[i] for i in idx]
    assert len(cmds) == 9, r.stdout[-800:] + r.stderr[-400:]
    for x in cmds:
        assert "/kit_wrap/v8f_t1_eval_reader_rule_rzgfhqetjvp.py --tree" in x and "DISTRULE_DIR=/workspace/v8proof/integ10k/build " in x
        assert "--form RR" in x and "--preroll-steps 0" in x and "eval_grid_windows.json" in x and "--kappa-max 0.2" in x and "--vis-closing" in x and "--rule-join" not in x
    first = cmds[0]
    assert "I10RQ50Hb20gGEcIDP_s0" in first and "--vis-pres" not in first and "--vis-gap-file /workspace/v8proof/cropscale/gapfiles/VGAPLE.npz --vis-closing" in first
    assert "/ckpt/UE_s0/A/ckpt_last.pt" in first and "--readout pinned" in first and "--rule-every" not in first
    cmp_i = [i for i, x in enumerate(out) if x.startswith("CMD: CUDA_VISIBLE_DEVICES=-1") and "cmp_dumps.py" in x]
    assert len(cmp_i) == 1 and idx[0] < cmp_i[0] < idx[1]
    assert ("%s/I10RQ50Hb20gGEcIDP_s0/dump_I10RQ50Hb20gGEcIDP_s0_m10_causal_heads %s/I10RQ50Hb20gGEc_s0/dump_I10RQ50Hb20gGEc_s0_m10_causal_heads" % (T1, T1)) in out[cmp_i[0]]
    gaps = {"GEc": "/workspace/v8proof/cropscale/gapfiles/VGAPLE.npz", "A90": "/workspace/v8proof/gapmargin/gapfiles/VGAPLE_A90.npz"}
    want = [(pre, arm, k) for pre in ("", "KF5") for arm in ("GEc", "A90") for k in (0, 1)]
    for x, (pre, arm, k) in zip(cmds[1:], want):
        assert ("I10RQ50%sHb20g%sPO_s%d" % (pre, arm, k)) in x
        assert ("--vis-gap-file %s --vis-closing --vis-pres-file %s/D_H5_s%d/pres.npz --vis-pres-mode or run " % (gaps[arm], PD, k)) in x
        assert ("/ckpt/UE_s%d/A/ckpt_last.pt" % k) in x and ("QMAP_LR16_s%d_q50.json" % k) in x and ("/lr16/TK16_s%d/ckpt_last.pt" % k) in x
        if pre:
            assert "--rule-every 5" in x and "--readout causal_g" in x and "--path-exec" in x and "--acc-gain 0" in x
        else:
            assert "--rule-every" not in x and "--readout pinned" in x and "--path-exec" not in x and "--acc-gain" not in x
    assert out[-1] == "DRY DONE" and r.returncode == 0
    assert sum(1 for x in out if x.startswith("CHECK: sha256 of %s/D_H5_s" % PD)) == 2


def test_first_line_of_the_run_text_and_the_gates():
    t = rd(QUEUE)
    assert t.count('vis-closing ${VCLOSE:-0}}${VPRES:+ vis-pres-file $VPRES vis pres sha256 $(sha256sum "$VPRES" | cut -c1-64) vis-pres-mode ${VPMODE:-}}" > "$out/ticks/run_$arm.txt"') == 1
    assert 'wrapper sha256 $(sha256sum "$WRAP" | cut -c1-64)' in t
    assert '[ "$got" = "b2a1d5f7242c3700" ] || stop "WRAPPER-P-SHA $got"' in t and '[ "$got" = "28bc31f21ecb4ccb" ] || stop "WRAPPER-SHA $got"' in t
    assert '= "ZZ CHECK-WRAPPER-P PASS" ] || stop "NO-PASS-OF-THE-WRAPPER-P-CHECK"' in t
    for nm in MK.REFS:
        assert (" " + nm + " ") in t or (" " + nm + ";") in t
    i_run, i_cmp, i_next = t.index("rollout I10RQ50Hb20gGEcIDP RR 0 0"), t.index("\nstep_identity\n"), t.index("rollout I10RQ50Hb20gGEcPO RR")
    assert i_run < i_cmp < i_next
    assert "ZZ PRESROLL10K DONE $(ts) all steps done" in t and '( set -C; echo "$(date +%s) $(ts) pid $$" > "$GP/PRESROLL10K_START.txt" )' in t
    assert " rm " not in t and "\nrm " not in t and " mv " not in t


def rule_log_py():
    t = rd(QUEUE)
    a = t.index('"$W/t1/$b/ticks/rule_log_$b" <<\'PYEOF\'\n') + len('"$W/t1/$b/ticks/rule_log_$b" <<\'PYEOF\'\n')
    return t[a:t.index("\nPYEOF\n", a)] + "\n"


def write_logs(base, shift=0.0, p_used=False, drop=None):
    n = 12
    rng = np.random.default_rng(5)
    cols = {"ci": np.arange(n), "t0": np.arange(n) * 5, "gap": rng.random(n), "closing": rng.random(n), "present": rng.random(n) > 0.5, "a_cmd": rng.random(n),
            "p_src": rng.random(n), "g_src": rng.random(n), "clips": np.asarray(["c%02d" % i for i in range(n)]), "form": np.asarray("RR"), "a_max": np.asarray(2.0)}
    cols["gap"][3] = np.nan
    cols["gap"][5] += shift
    if drop:
        cols.pop(drop)
    side = {"p": rng.random(n), "g_read": rng.random(n), "g_used": rng.random(n), "ok": rng.random(n) > 0.2, "form": np.asarray("RR")}
    if p_used:
        side["p_used"] = side["p"].copy()
    np.savez(base + ".npz", **cols)
    np.savez(base + "_rd.npz", **side)


def run_cmp(tmp_path, a, b):
    script = tmp_path / "cmp_rule_log.py"
    script.write_text(rule_log_py(), encoding="utf-8")
    return subprocess.run([sys.executable, str(script), a, b], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)


def test_rule_log_compare_of_the_identity_step(tmp_path):
    a, b = str(tmp_path / "rule_log_A"), str(tmp_path / "rule_log_B")
    write_logs(a)
    write_logs(b)
    r = run_cmp(tmp_path, a, b)
    assert r.returncode == 0 and r.stdout.startswith("ADMIT-OK identity of the rule log: arrays compared 14,"), r.stdout + r.stderr
    c = str(tmp_path / "rule_log_C")
    write_logs(c, shift=1e-9)
    r = run_cmp(tmp_path, c, b)
    assert r.returncode == 1 and "RULE-LOG-DIFFERS" in r.stdout and "gap.npz" in r.stdout
    d = str(tmp_path / "rule_log_D")
    write_logs(d, p_used=True)
    r = run_cmp(tmp_path, d, b)
    assert r.returncode == 1 and "p_used-in-the-side-file" in r.stdout and "keys_rd.npz" in r.stdout
    e = str(tmp_path / "rule_log_E")
    write_logs(e, drop="closing")
    r = run_cmp(tmp_path, e, b)
    assert r.returncode == 1 and "keys.npz" in r.stdout
    r = run_cmp(tmp_path, str(tmp_path / "rule_log_none"), b)
    assert r.returncode == 1 and "missing.npz" in r.stdout
