"""test_repl_queue.py -- card ADD121 (REPL10K): the generated queue against its generator, the dry runs of both stages, and the rule log compare of the identity stage
on tiny arrays. The queue of card ADD117 is taken from ../../gapmargin/queue (same relative place in the pod stage). No pod result file is opened."""
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
import make_repl_queue as MK  # noqa: E402

QUEUE = os.path.join(HERE, MK.OUT_NAME)
W = "/workspace/v8proof/integ10k"


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


def dry(tmp_path, stage=None):
    env = dict(os.environ)
    env.update({"DRY": "1", "RP_ROOT": str(tmp_path / "root").replace(os.sep, "/")})
    env.pop("RP_STAGE", None)
    if stage:
        env["RP_STAGE"] = stage
    r = subprocess.run([BASH, QUEUE.replace(os.sep, "/")], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert not os.path.exists(str(tmp_path / "root"))
    return r


def test_main_stage_dry_run_prints_the_26_rollouts(tmp_path):
    r = dry(tmp_path)
    out = r.stdout.splitlines()
    cmds = [x for x in out if x.startswith("CMD: DISTRULE_DIR")]
    assert len(cmds) == 26 and out[-1] == "DRY DONE" and r.returncode == 0, r.stdout[-800:] + r.stderr[-400:]
    want = [(m, run) for m in (2, 3) for run in MK.RUNS]
    for x, (m, (setting, name, form, qmap, gap)) in zip(cmds, want):
        rs = m - 2
        assert ("/build/v8f_t1_eval_reader_rule_rzgfhqetjv.py --tree" in x) and ("--form %s " % form) in x
        assert ("--ckpt %s/ckpt/UE_s%d/A/ckpt_last.pt --tag %s_s%d_m10_causal_heads" % (W, m, name, m)) in x and "--kappa-max 0.2" in x
        assert (("--qmap %s/qmap_sweep/QMAP_LR16_s%d_q50.json" % (W, rs)) in x) == qmap and (("--reader %s/lr16/TK16_s%d/ckpt_last.pt" % (W, rs)) in x) == (form == "RR")
        if gap:
            f = "/workspace/v8proof/cropscale/gapfiles/VGAPLE.npz" if gap == "$BASEF" else "/workspace/v8proof/gapmargin/gapfiles/VGAPLE_A90.npz"
            assert ("--vis-gap-file %s --vis-closing" % f) in x
        else:
            assert "--vis-gap-file" not in x and "--vis-closing" not in x
        if "KF5" in name:
            assert "--rule-every 5" in x and "--readout causal_g" in x and "--path-exec" in x and "--acc-gain 0" in x
        else:
            assert "--rule-every" not in x and "--readout pinned" in x and "--path-exec" not in x
        if "P30" in name:
            assert "--preroll-steps 30" in x and "grid_p30.json" in x
        else:
            assert "--preroll-steps 0" in x and "eval_grid_windows.json" in x
        assert ("--rule-log" in x) == (form != "R0") and "IDR" not in x
    assert any("REPL_MODELS_RECORD.json names for UE_s2 and UE_s3" in x for x in out)


def test_identity_stage_dry_run_prints_three_runs_each_with_its_compare(tmp_path):
    r = dry(tmp_path, "identity")
    out = r.stdout.splitlines()
    idx = [i for i, x in enumerate(out) if x.startswith("CMD: DISTRULE_DIR")]
    cmp_i = [i for i, x in enumerate(out) if x.startswith("CMD: CUDA_VISIBLE_DEVICES=-1") and "cmp_dumps.py" in x]
    assert len(idx) == 3 and len(cmp_i) == 3 and out[-1] == "DRY DONE", r.stdout[-800:] + r.stderr[-400:]
    for i, c, (setting, name, form, qmap, banked) in zip(idx, cmp_i, MK.IDRUNS):
        assert i < c and ("--tag %s_s0_m10_causal_heads" % name) in out[i] and "/ckpt/UE_s0/A/ckpt_last.pt" in out[i] and ("--form %s " % form) in out[i]
        assert ("%s/t1/%s_s0/dump_%s_s0_m10_causal_heads %s/t1/%s_s0/dump_%s_s0_m10_causal_heads" % (W, name, name, W, banked, banked)) in out[c]
    assert dry(tmp_path, "other").returncode == 2


def test_markers_and_gates_in_the_text():
    t = rd(QUEUE)
    assert 'MK=REPL10K; CKS="${ARMS[*]}"' in t and "identity) MK=REPL10K_ID; CKS=0 ;;" in t and "ARMS=(2 3); FIRST=(0 1)" in t
    assert 'echo "ZZ ${MK} DONE $(ts) all steps done" > "$GP/${MK}_DONE.txt"' in t and '( set -C; echo "$(date +%s) $(ts) pid $$" > "$GP/${MK}_START.txt" )' in t
    assert '[ "$DRY" = 1 ] || [ -f "$GP/REPL10K_ID_DONE.txt" ] || stop "NO-IDENTITY-STAGE"' in t and '[ "$got" = "28bc31f21ecb4ccb" ] || stop "WRAPPER-SHA $got"' in t
    assert " rm " not in t and "\nrm " not in t and " mv " not in t


def cmp_py():
    t = rd(QUEUE)
    key = '"$W/t1/$b/ticks" "$b" <<\'PYEOF\'\n'
    a = t.index(key) + len(key)
    return t[a:t.index("\nPYEOF\n", a)] + "\n"


def write_logs(d, name, shift=0.0, drop=None, extra=False, side=True, mid=False):
    os.makedirs(d, exist_ok=True)
    n = 10
    rng = np.random.default_rng(3)
    cols = {"ci": np.arange(n), "j": np.arange(n) * 5, "g_src": rng.random(n), "present": rng.random(n) > 0.5, "form": np.asarray("RR"), "D": np.asarray(2.0)}
    cols["g_src"][2] = np.nan
    cols["g_src"][4] += shift
    if drop:
        cols.pop(drop)
    if extra:
        cols["vis_closing"] = np.asarray(0)
    np.savez(os.path.join(d, "rule_log_%s.npz" % name), **cols)
    if mid:
        np.savez(os.path.join(d, "rule_log_mid_%s.npz" % name), **cols)
    if side:
        np.savez(os.path.join(d, "rule_log_%s_rd.npz" % name), p=rng.random(n), ok=rng.random(n) > 0.2)


def run_cmp(tmp_path, da, a, db, b):
    script = tmp_path / "cmp_rule_logs.py"
    script.write_text(cmp_py(), encoding="utf-8")
    return subprocess.run([sys.executable, str(script), da, a, db, b], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)


def test_rule_log_compare_of_the_identity_stage(tmp_path):
    db = str(tmp_path / "B")
    write_logs(db, "B")
    d = str(tmp_path / "A1")
    write_logs(d, "A1", extra=True)                      # a later variant adds a key: counted, no failure
    r = run_cmp(tmp_path, d, "A1", db, "B")
    assert r.returncode == 0 and "files 2, arrays compared 7," in r.stdout and "keys only in the new run 1" in r.stdout, r.stdout + r.stderr
    d = str(tmp_path / "A2")
    write_logs(d, "A2", shift=1e-9)
    r = run_cmp(tmp_path, d, "A2", db, "B")
    assert r.returncode == 1 and "g_src:rule_log_.npz" in r.stdout
    d = str(tmp_path / "A3")
    write_logs(d, "A3", drop="present")
    r = run_cmp(tmp_path, d, "A3", db, "B")
    assert r.returncode == 1 and "lost:present" in r.stdout
    d = str(tmp_path / "A4")
    write_logs(d, "A4", side=False)
    r = run_cmp(tmp_path, d, "A4", db, "B")
    assert r.returncode == 1 and "lost-file:rule_log__rd.npz" in r.stdout
    d = str(tmp_path / "A5")
    write_logs(d, "A5", mid=True)                        # a file only the new run has: counted
    r = run_cmp(tmp_path, d, "A5", db, "B")
    assert r.returncode == 0 and "files 2," in r.stdout
    e1, e2 = str(tmp_path / "E1"), str(tmp_path / "E2")
    os.makedirs(e1)
    os.makedirs(e2)
    r = run_cmp(tmp_path, e1, "X", e2, "Y")              # a form with no rule log (R0): nothing to compare
    assert r.returncode == 0 and "files 0, arrays compared 0," in r.stdout
