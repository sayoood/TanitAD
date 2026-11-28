"""Tests of pod_qhead_dose_fits.sh: DRY=1 prints the two fits with the clip sizes; refusals (tool sha, pack, controls, markers); the memory wait (20 GB); a full run against a stub python; no process-name test, no rm, no mv."""
import hashlib
import os
import re
import subprocess

import pytest

import dose_testlib as TL  # noqa: E402

BASH = TL.BASH
SCRIPT = os.path.join(TL.SRC, "pod_qhead_dose_fits.sh")
pytestmark = pytest.mark.skipif(BASH is None, reason="no bash on this box")

FAKE_PY = r"""#!/usr/bin/env bash
echo "stub $*" >> "$STUB_LOG"
arm=""; out=""; prev=""
for a in "$@"; do [ "$prev" = "--arm" ] && arm=$a; [ "$prev" = "--out-dir" ] && out=$a; prev=$a; done
[ -n "${STUB_FAIL_ARM:-}" ] && [ "$arm" = "$STUB_FAIL_ARM" ] && { echo "REFUSED: stub"; exit 3; }
echo x > "$out/$arm.qhead.npz"; echo '{}' > "$out/$arm.qhead.json"
echo "$arm: stub fit line"
exit 0
"""


def sh(p):
    return TL.sh(p)


def run(env_extra, tmp_path, timeout=120):
    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run([BASH, SCRIPT], env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=sh(tmp_path))


def fake_pod(tmp_path, anon_gb=1, max_gb=60, d1=True, d2=("LSD_s0", "LSD_s1")):
    d = tmp_path / "pod"
    (d / "bin").mkdir(parents=True, exist_ok=True)
    (d / "memory.max").write_text(str(max_gb * 1073741824), encoding="utf-8")
    (d / "memory.stat").write_text("anon %d\nfile 1\n" % (anon_gb * 1073741824), encoding="utf-8")
    O = d / "qhead_dose"
    for sub in ("tools", "states", "pack", "controls", "run"):
        (O / sub).mkdir(parents=True, exist_ok=True)
    (O / "run").rmdir()
    ft, mt = b"# fit tool\n", b"# model\n"
    (O / "tools" / "qhead_fit_n1.py").write_bytes(ft)
    (O / "tools" / "qhead_model.py").write_bytes(mt)
    (O / "pack" / "train_pack_all.npz").write_bytes(b"pack")
    for arm in ("LSD_s0", "LSD_s1"):
        for suf in ("_dose_train.npz", "_dose_train.npz.json", "_dose_train_lead.npz", "_dose_train_lead.npz.head.npz"):
            (O / "states" / (arm + suf)).write_bytes(b"x")
    if d1:
        (O / "controls" / "d1_pack.json").write_text('{\n "pass": true\n}\n')
    for arm in d2:
        (O / "controls" / ("d2_%s.json" % arm)).write_text('{\n "pass": true\n}\n')
    py = d / "fakepy"
    py.write_text(FAKE_PY, encoding="utf-8", newline="\n")
    env = {"QD_W": sh(d), "QD_PY": sh(py), "QD_CG_MAX": sh(d / "memory.max"), "QD_CG_STAT": sh(d / "memory.stat"), "QD_POLL_S": "1", "QD_MAX_WAIT_S": "2", "STUB_LOG": sh(d / "stub.log"),
           "QD_FIT_SHA": hashlib.sha256(ft).hexdigest()[:16], "QD_MODEL_SHA": hashlib.sha256(mt).hexdigest()[:16]}
    return d, env


def test_syntax_lf_and_forbidden_commands():
    r = subprocess.run([BASH, "-n", SCRIPT], capture_output=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    raw = open(SCRIPT, "rb").read()
    assert b"\r" not in raw
    code = "\n".join(l for l in raw.decode("utf-8").splitlines() if not l.lstrip().startswith("#"))
    for bad in ("pgrep", "pkill", "killall", "ps -", "ps aux", "pidof"):
        assert bad not in code, bad
    assert not re.search(r"(^|[;&|\s])(rm|mv|rmdir|shred)(\s|$)", code), "rm or mv in the script"
    assert "memory.max" in code and "memory.stat" in code and "set -o noclobber" in code and "57e1242713e0b42e" in code and "f6ff755e3dd4e212" in code


def test_dry_run_prints_both_fits_with_the_sizes_and_creates_nothing(tmp_path):
    d, env = fake_pod(tmp_path)
    env["DRY"] = "1"
    before = sorted(str(p) for p in d.rglob("*"))
    r = run(env, tmp_path)
    assert r.returncode == 0, r.stderr + r.stdout
    assert sorted(str(p) for p in d.rglob("*")) == before
    fits = re.findall(r"DRY: \S+ \S+qhead_fit_n1\.py --arm (LSD_s[01]) --seed (\d) --states \S+_dose_train\.npz --pack \S+train_pack_all\.npz --out-dir \S+/heads --device cpu "
                      r"--fit-clips 3048 --select-clips 762 --check-clips 762 --lead \S+ --banked-head \S+", r.stdout)
    assert fits == [("LSD_s0", "0"), ("LSD_s1", "1")]
    assert r.stdout.count("DRY: memory wait before fit") == 2 and "more than 20 GB" in r.stdout and "ZZ QHEAD-DOSE FITS DONE <n> of 2" in r.stdout


def test_sizes_add_up_to_the_4572_clips_of_the_note():
    assert 3048 + 762 + 762 == 4572


@pytest.mark.parametrize("marker", ["QHEAD_DOSE_FITS_STARTED.txt", "QHEAD_DOSE_FITS_DONE.txt", "QHEAD_DOSE_FITS_STOP.txt"])
def test_markers_refuse_a_second_start(tmp_path, marker):
    d, env = fake_pod(tmp_path)
    (d / "qhead_dose" / "run").mkdir()
    (d / "qhead_dose" / "run" / marker).write_text("x")
    r = run(env, tmp_path)
    assert r.returncode == 2 and "REFUSED" in r.stdout and marker in r.stdout


def stop_text(d):
    return (d / "qhead_dose" / "run" / "QHEAD_DOSE_FITS_STOP.txt").read_text()


def test_fit_tool_sha_mismatch_stops(tmp_path):
    d, env = fake_pod(tmp_path)
    env["QD_FIT_SHA"] = "0" * 16
    r = run(env, tmp_path)
    assert r.returncode == 4 and "qhead_fit_n1.py sha256" in stop_text(d) and not (d / "stub.log").exists()


def test_model_sha_mismatch_stops(tmp_path):
    d, env = fake_pod(tmp_path)
    env["QD_MODEL_SHA"] = "0" * 16
    r = run(env, tmp_path)
    assert r.returncode == 4 and "qhead_model.py sha256" in stop_text(d)


def test_missing_pack_stops(tmp_path):
    d, env = fake_pod(tmp_path)
    os.replace(d / "qhead_dose" / "pack" / "train_pack_all.npz", d / "qhead_dose" / "pack" / "moved.npz")
    r = run(env, tmp_path)
    assert r.returncode == 4 and "missing label pack" in stop_text(d)


def test_control_d1_not_passed_stops(tmp_path):
    d, env = fake_pod(tmp_path, d1=False)
    r = run(env, tmp_path)
    assert r.returncode == 4 and "control d1 is not passed" in stop_text(d) and not (d / "stub.log").exists()
    (d / "qhead_dose" / "controls" / "d1_pack.json").write_text('{\n "pass": false\n}\n')
    for m in ("QHEAD_DOSE_FITS_STARTED.txt", "QHEAD_DOSE_FITS_STOP.txt"):
        os.replace(d / "qhead_dose" / "run" / m, d / "qhead_dose" / "run" / (m + ".old"))
    r = run(env, tmp_path)
    assert r.returncode == 4 and "control d1 is not passed" in stop_text(d)


def test_control_d2_missing_for_one_arm_stops(tmp_path):
    d, env = fake_pod(tmp_path, d2=("LSD_s0",))
    r = run(env, tmp_path)
    assert r.returncode == 4 and "control d2 is not passed" in stop_text(d) and "LSD_s1" in stop_text(d) and not (d / "stub.log").exists()


def test_memory_wait_needs_20_gb(tmp_path):
    d, env = fake_pod(tmp_path, anon_gb=45, max_gb=60)             # 15 GB free
    r = run(env, tmp_path)
    assert r.returncode == 4 and "less than 20 GB free for fit LSD_s0" in stop_text(d) and not (d / "stub.log").exists()


def test_full_run_two_fits_last_line_and_second_start_refused(tmp_path):
    d, env = fake_pod(tmp_path, anon_gb=10)
    r = run(env, tmp_path)
    assert r.returncode == 0, r.stdout[-800:] + r.stderr
    assert r.stdout.strip().splitlines()[-1].endswith("ZZ QHEAD-DOSE FITS DONE 2 of 2")
    assert (d / "qhead_dose" / "run" / "QHEAD_DOSE_FITS_DONE.txt").read_text().strip() == "2"
    calls = (d / "stub.log").read_text().splitlines()
    assert len(calls) == 2 and "--arm LSD_s0 --seed 0" in calls[0] and "--arm LSD_s1 --seed 1" in calls[1]
    for c in calls:
        assert "--fit-clips 3048 --select-clips 762 --check-clips 762" in c and "--device cpu" in c and "--lead" in c and "--banked-head" in c
    for arm in ("LSD_s0", "LSD_s1"):
        assert (d / "qhead_dose" / "heads" / (arm + ".qhead.json")).exists()
    assert run(env, tmp_path).returncode == 2


def test_a_failed_fit_stops_with_the_reason_and_counts_nothing_done(tmp_path):
    d, env = fake_pod(tmp_path, anon_gb=10)
    env["STUB_FAIL_ARM"] = "LSD_s1"
    r = run(env, tmp_path)
    assert r.returncode == 4 and "fit LSD_s1 failed" in stop_text(d) and "REFUSED: stub" in stop_text(d)
    assert r.stdout.strip().splitlines()[-1].startswith("[") and "ZZ QHEAD-DOSE FITS STOP:" in r.stdout.strip().splitlines()[-1]
    assert not (d / "qhead_dose" / "run" / "QHEAD_DOSE_FITS_DONE.txt").exists()


def test_existing_record_is_not_fitted_again(tmp_path):
    d, env = fake_pod(tmp_path, anon_gb=10)
    h = d / "qhead_dose" / "heads"
    h.mkdir()
    (h / "LSD_s0.qhead.json").write_text("{}")
    r = run(env, tmp_path)
    assert r.returncode == 0 and "fit LSD_s0 already written" in r.stdout
    assert len((d / "stub.log").read_text().splitlines()) == 1
