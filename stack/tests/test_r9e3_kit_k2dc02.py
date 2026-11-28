"""The E3 launch kit of arm K2DC02 (stack/ops/r9e3_kit/k2dc02/: records queue 28, dev-box prep, restart, resume): CPU only, no ssh, no GPU, nothing is run on Thor.

Mirror of test_r9e2_kit_k2dc02cl.py for the refcv9 FULL run (study e3, profile refcv9-full, 73,280 steps) and its early-D1 branch K2DC02_d1:

  * every script has LF endings and valid bash syntax; no executable line names another arm's queue;
  * the queue binds, in this order and with --argv-prefix r9e3: map-banded K2DC02, box K2DC02, then the D1 arm map-banded K2DC02_d1, box K2DC02_d1; every bind has a --dry-run first, the D1 dry-runs are SOFT
    (a refusal is a marker, never fatal to E3's binds), the DRY exit comes before the first GPU step, and the GPU step waits for the lock (flock -n poll) and never forces it;
  * the queue's argv guard (the same python it ships) is RUN on the landed E3 / D1 argv files and refuses mutated ones;
  * the restart: study e3, arm "K2DC02", every named guard, the CAP is the landed approval's max_s_per_step (10.5) for BOTH arms, its NOARM python accepts E3's argv only;
  * the early-D1 pause is WIRED: prep writes the PAUSE_AT_STEP stamp the chain reads, the chain's manifest block exists and skips *_d1, the supervisor pauses only after the checkpoint AND the eval row, the
    one-shot latch makes the second start resume; the resume script refuses without the latch / on tree drift / while a chain lives.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "ops" / "r9e3_kit" / "k2dc02"
G3 = ROOT / "ops" / "r9e1_gate3"
RUNS = ROOT / "ops" / "runs.d"
FILES = ("r9_recqueue28.sh", "prep_q28.sh", "restart_e3.sh", "resume_e3.sh")
PY = sys.executable.replace("\\", "/")


def _bash():
    for c in ("C:/Program Files/Git/usr/bin/bash.exe", "C:/Program Files/Git/bin/bash.exe"):      # never System32's WSL bash
        if Path(c).exists():
            return c
    w = shutil.which("bash")
    if w and "system32" not in w.lower():
        return w
    pytest.skip("no usable bash")


def _t(name):
    return (KIT / name).read_bytes().decode("utf-8")


def _code(t):
    return [ln for ln in t.splitlines() if not ln.lstrip().startswith("#")]


def _argv(name):
    return json.loads((RUNS / f"r9e3_{name}.argv.json").read_text(encoding="utf-8"))["argv"]


@pytest.mark.parametrize("name", FILES)
def test_script_has_lf_endings(name):
    assert b"\r" not in (KIT / name).read_bytes()


@pytest.mark.parametrize("name", FILES)
def test_script_has_valid_bash_syntax(name):
    r = subprocess.run([_bash(), "-n", str(KIT / name)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("name", FILES)
def test_no_executable_line_names_another_arms_queue(name):
    code = "\n".join(_code(_t(name)))
    for other in ("recqueue27", "recqueue26", "recqueue19", "q27_after_read", "K2DC02cl", "K2DC02s1", "r9e2_K", "r9e1/rectree"):
        assert other not in code, other


def test_queue_binds_in_order_with_the_e3_prefix():
    code = "\n".join(_code(_t("r9_recqueue28.sh")))
    assert 'RE3="--argv-prefix r9e3"' in code
    gpu = [m.group(0) for m in re.finditer(r"^\s*gpu (mapb|box)_\$(K2D|K2) (map-banded|box) \$(K2D|K2) \$RE3 --commit \$X --code \$LT", code, re.M)]
    assert [g.split()[1:5] for g in gpu] == [
        ["mapb_$K2", "map-banded", "$K2", "$RE3"], ["box_$K2", "box", "$K2", "$RE3"],
        ["mapb_$K2D", "map-banded", "$K2D", "$RE3"], ["box_$K2D", "box", "$K2D", "$RE3"]], gpu
    assert "K2D=${K2}_d1" in code
    dry = re.findall(r"^\s*dry (mapb|box)_\$(K2D|K2) (map-banded|box) ", code, re.M)
    assert len(dry) == 4
    # the dry-runs precede the DRY exit, which precedes the first gpu step
    assert code.index("dry mapb_$K2 ") < code.index('"${R9Q28_DRY:-}" = 1') < code.index("gpu mapb_$K2 ")
    assert "--dry-run" in code and "ZZRECQ28-DRYONLYZZ" in code


def test_queue_d1_dry_runs_are_soft_and_e3_binds_continue():
    t = _t("r9_recqueue28.sh")
    i = t.index("DRY_SOFT=1\ndry mapb_$K2D")
    j = t.index("DRY_SOFT=0")
    assert "|| D1OK=0" in t[i:j] and t[i:j].count("|| D1OK=0") == 2
    assert 'if [ "$D1OK" = 1 ]; then\n  gpu mapb_$K2D' in t
    assert "D1-DRY-REFUSED" in t and "ZZRECQ28-DONE-E3ONLYZZ" in t and "R9Q28_D1_ONLY" in t
    # the E3 dry-runs are hard (not soft): a refusal fails the queue
    assert t.index("dry mapb_$K2 ") < i and t.index("dry box_$K2 ") < i


def test_queue_waits_for_the_lock_and_never_forces_it():
    code = "\n".join(_code(_t("r9_recqueue28.sh")))
    assert "flock -n $LOCKF true" in code and "sleep 60" in code
    assert "flock -u" not in code and "kill" not in code and "rm -" not in code
    assert "thor_gpu.lock" in code and "ZZRECQ28-FAIL-" in code


def test_queue_records_land_in_the_e3_root_not_e2s():
    code = "\n".join(_code(_t("r9_recqueue28.sh")))
    assert "R3=/home/nvidia/r8ns/r9e3" in code and "r9e2" not in code
    assert "recq28_" in code and "QUEUE28-START" in code and "QUEUE28-DONE" in code


def _queue_guard_src():
    t = _t("r9_recqueue28.sh")
    m = re.search(r'/usr/bin/python3 -c "\n(.*?)" \$RD', t, re.S)
    assert m, "the argv guard python was not found"
    return m.group(1)


def _run_guard(tmp_path, e3, d1):
    d = tmp_path / "rd"
    d.mkdir(exist_ok=True)
    (d / "r9e3_K2DC02.argv.json").write_text(json.dumps({"argv": e3}), encoding="utf-8")
    (d / "r9e3_K2DC02_d1.argv.json").write_text(json.dumps({"argv": d1}), encoding="utf-8")
    src = _queue_guard_src().replace("sys.argv[1]", repr(str(d)))
    return subprocess.run([PY, "-I", "-c", src], capture_output=True, text=True, encoding="utf-8").returncode


def _set(a, k, v):
    a = list(a)
    a[a.index(k) + 1] = v
    return a


def test_queue_argv_guard_accepts_the_landed_e3_and_d1_argv(tmp_path):
    assert _run_guard(tmp_path, _argv("K2DC02"), _argv("K2DC02_d1")) == 0


def test_queue_argv_guard_refuses_mutated_argv(tmp_path):
    e3, d1 = _argv("K2DC02"), _argv("K2DC02_d1")
    assert _run_guard(tmp_path, _set(e3, "--seed", "1"), d1) == 1
    assert _run_guard(tmp_path, _set(e3, "--steps", "36000"), d1) == 1
    assert _run_guard(tmp_path, e3, _set(d1, "--steps", "73280")) == 1
    assert _run_guard(tmp_path, e3, _set(d1, "--wsd-branch-decay", "5000")) == 1
    assert _run_guard(tmp_path, d1, e3) == 1                       # swapped: E3 under D1's name


def test_landed_argv_and_approvals_are_what_the_kit_assumes():
    e3, d1 = _argv("K2DC02"), _argv("K2DC02_d1")
    assert e3.count("--save-at") == 1 and e3[e3.index("--save-at") + 1] == "30000"
    assert d1[d1.index("--wsd-branch-from") + 1].endswith("/r9e3/runs/r9e3-K2DC02/ckpt_full_30000.pt")
    for a in ("K2DC02", "K2DC02_d1"):
        p = json.loads((RUNS / f"r9e3_{a}.pi_cost_approval.json").read_text(encoding="utf-8"))
        assert p["max_s_per_step"] == 10.5


def test_restart_runs_study_e3_for_arm_K2DC02_and_has_every_guard():
    t = _t("restart_e3.sh")
    code = "\n".join(_code(t))
    assert 'V=K2DC02' in code and 'R9_STUDY=e3 R9_PRECHECK_ARMS="$V" bash' in code and "r9gate3_restart.sh" in code and '"$V" $RC12' in code
    for g in ("FDISK", "THORDISK", "THORSHM", "NORECORDS", "WRONGV", "NOKIT", "NOZGENV", "NOPAUSE", "NORECSTAMP", "RECINPUT", "NOAPPROVAL", "NOARM", "CHAINLIVE", "GPULOCK"):
        assert f"bad {g} " in code or f"bad {g}\t" in code or re.search(rf"bad {g}\b", code), g
    assert "R9_STUDY=e2" not in code and "zg_auto" not in code
    assert "r9e3_$V.argv.json" in code and "r9e3_$A.pi_cost_approval.json" in code
    assert 'max_s_per_step\\": $CAP,' in code
    # GPULOCK is REAL-only: it sits inside `if [ -z "$MODE" ]`, before the DRY exit
    assert code.index('if [ -z "$MODE" ]') < code.index("GPULOCK") < code.index('if [ "$MODE" = DRY ]; then say "ZZAUTOE3-DRY-ENDZZ')
    assert "ZZAUTOE3-OKZZ" in code and "SUITE-STARTED" in code


def test_restart_NOARM_python_accepts_e3_argv_only(tmp_path):
    m = re.search(r'-c "(import json,sys; a=json.load\(sys.stdin\).*?)" \| tr', _t("restart_e3.sh"))
    assert m
    src = m.group(1)

    def run(argv):
        return subprocess.run([PY, "-I", "-c", src], input=json.dumps({"argv": argv}), capture_output=True, text=True, encoding="utf-8").stdout.strip()

    assert run(_argv("K2DC02")) == "1"
    assert run(_argv("K2DC02_d1")) == "0"
    assert run(_set(_argv("K2DC02"), "--seed", "1")) == "0"


def test_restart_cap_equals_the_landed_approval():
    caps = {json.loads((RUNS / f"r9e3_{a}.pi_cost_approval.json").read_text(encoding="utf-8"))["max_s_per_step"] for a in ("K2DC02", "K2DC02_d1")}
    assert caps == {10.5}


def test_prep_has_the_guards_and_the_e3_root_setup_and_never_overwrites():
    t = _t("prep_q28.sh")
    code = "\n".join(_code(t))
    for tag in ("NOREC", "NOTREMOTE", "RECINPUT", "NOFILE", "DISK", "FDISK", "KIT", "ZGENV", "RECSTAMP", "PAUSESTAMP", "RECTREE", "RECQSHIP", "DRY", "START"):
        assert f"stop {tag} " in code, tag
    for needle in ("merge-base --is-ancestor", "R9W_ROOT=$R3", "SHIP kit", "SHIP tree", "zerograd_controls.env", "gate3/REC_C12", "gate3/PAUSE_AT_STEP",
                   "R9Q28_DRY=1", "setsid nohup bash kit/r9_recqueue28.sh", "ZZPREPQ28-STARTEDZZ", "ZZPREPQ28-STOP-", "DELLOG"):
        assert needle in code, needle
    assert "R9E3_PAUSE_AT:-30000" in code
    # the DRY runs before the REAL queue, and the root setup before both
    assert code.index("zerograd_controls.env") < code.index("R9Q28_DRY=1") < code.index("setsid nohup bash kit/r9_recqueue28.sh")
    # an existing differing file STOPS (exit 15), it is not overwritten
    assert code.count(" 15") >= 3
    assert "pgrep" not in code and "pkill" not in code


def test_early_d1_pause_is_wired_end_to_end():
    chain = (G3 / "r9gate3_chain.sh").read_text(encoding="utf-8")
    assert "PAUSE-MANIFEST-BEGIN" in chain and "$R/gate3/PAUSE_AT_STEP" in chain and "*_d1) ;;" in chain
    assert 'echo "PAUSE_AT_STEP=$PAUSE_AT" >> $R/runs.d/$ARM.env' in chain
    assert "$A-PAUSED" in chain and '"$RC" = 7' in chain
    sup = (ROOT / "ops" / "sup_refcv7.sh").read_text(encoding="utf-8")
    assert 'PAUSE_AT="${PAUSE_AT_STEP:-}"' in sup and "PAUSED.json present" in sup and "ckpt_full_${PAUSE_AT}.pt AND the step-${PAUSE_AT} eval row" in sup
    # prep writes the stamp the chain reads: the same file name
    assert "gate3/PAUSE_AT_STEP" in _t("prep_q28.sh")


def test_resume_guards_and_lock():
    code = "\n".join(_code(_t("resume_e3.sh")))
    for g in ("NOPAUSED", "ALREADYDONE", "NOMANIFEST", "TREEDRIFT", "CHAINLIVE", "D1NOTDONE"):
        assert f"bad {g} " in code, g
    assert "sup_refcv7.sh $ARM $R/runs.d" in code and "flock -w 7200 9" in code and "GPU-NOT-EMPTY" in code
    assert code.index('if [ "$MODE" = DRY ]') < code.index("setsid nohup")
    assert "pkill" not in code and "pgrep" not in code


# ---- 2026-11-24 fix: queue 28 failed a SUCCESSFUL D1 bind -- its marker regex had no `_`, and arm K2DC02_d1 has one ----------------------------
STATUS_1124 = (                                   # the real Thor lines (recq28_a49c964032cd.status), MEASURED 2026-11-24
    "2026-11-23T23:03:25Z box_K2DC02 OK (ZZR9E1BIND-box-K2DC02-OKZZ)\n"
    "2026-11-23T23:03:25Z K2DC02-RECORDS-READY at a49c964032cd (map-banded + box own, E3 argv)\n"
    "2026-11-23T23:03:25Z mapb_K2DC02_d1 try 1: lock free, calling bind\n"
    "2026-11-24T00:22:57Z mapb_K2DC02_d1 FAILED (no marker): [r9e1-bind:map-banded:K2DC02_d1] harness finished\n"
    "ZZRECQ28-FAIL-mapb_K2DC02_d1ZZ\n")


def test_queue_marker_regexes_accept_an_underscore_arm():
    pats = re.findall(r"grep -oE '(ZZR9E1BIND-[^']*ZZ)'", _t("r9_recqueue28.sh"))
    assert len(pats) == 2, pats                                                     # the skip check + the bind loop
    for p in pats:
        assert "_" in p.split("[", 1)[1].split("]", 1)[0], p                         # the character class carries `_`
        for m in ("ZZR9E1BIND-map-banded-K2DC02_d1-OKZZ", "ZZR9E1BIND-box-K2DC02_d1-OKZZ", "ZZR9E1BIND-box-K2DC02-OKZZ"):
            assert re.fullmatch(p, m), (p, m)
    assert "[A-Za-z0-9-]*ZZ" not in _t("r9_recqueue28.sh")                         # the 11-23 class is gone


def _gpu_fn():
    t = _t("r9_recqueue28.sh")
    i = t.index("gpu() {")
    return t[i:t.index("\n}\n", i) + 3]


def _run_gpu(tmp_path, pre_out):
    L = tmp_path / "logs"
    L.mkdir()
    called = tmp_path / "called"
    if pre_out is not None:
        (L / "recq28_a49c964032cd_mapb_K2DC02_d1.1.out").write_bytes(pre_out.encode())
    stub = tmp_path / "bind.sh"
    stub.write_bytes(("echo CALLED > '%s'\necho ZZR9E1BIND-map-banded-K2DC02_d1-OKZZ\n" % called.as_posix()).encode())
    lock = tmp_path / "lock"
    lock.write_bytes(b"")
    sh = tmp_path / "t.sh"
    sh.write_bytes(("set -u\nL='%s'; C12=a49c964032cd; LOCKF='%s'; BIND='%s'\n"
                    "ev() { echo \"EV $*\"; }\nfail() { echo \"FAIL $*\"; exit 3; }\nflock() { return 0; }   # the dev box has no flock: the lock is free here\n"
                    "%s\ngpu mapb_K2DC02_d1 map-banded K2DC02_d1\necho RC=$?\n"
                    % (L.as_posix(), lock.as_posix(), stub.as_posix(), _gpu_fn())).encode())
    r = subprocess.run([_bash(), str(sh)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    return r.stdout, called.exists()


def test_gpu_step_skips_a_bind_already_ok_at_this_rec(tmp_path):
    out, called = _run_gpu(tmp_path, "[r9e1-bind:map-banded:K2DC02_d1] harness finished\nZZR9E1BIND-map-banded-K2DC02_d1-OKZZ\n")
    assert "RC=0" in out and "OK already" in out and "skipped" in out and not called, out


def test_gpu_step_binds_and_passes_an_underscore_marker(tmp_path):
    out, called = _run_gpu(tmp_path, None)                                         # no earlier .out: the bind runs; its marker must pass
    assert called and "RC=0" in out and "mapb_K2DC02_d1 OK (ZZR9E1BIND-map-banded-K2DC02_d1-OKZZ)" in out and "FAIL" not in out, out


def test_gpu_step_does_not_skip_a_failed_earlier_try(tmp_path):
    out, called = _run_gpu(tmp_path, "[r9e1-bind:map-banded:K2DC02_d1] REFUSED\nZZR9E1BIND-map-banded-K2DC02_d1-REFUSEDZZ\n")
    assert called and "RC=0" in out and "OK already" not in out, out


def test_restart_norecords_reads_e3s_own_records_ready_line(tmp_path):
    code = "\n".join(_code(_t("restart_e3.sh")))
    assert "grep -c ' $V-RECORDS-READY at $RC12 '" in code and "ZZRECQ28-DONE-E3ONLYZZ) ;;" not in code
    st = tmp_path / "s.status"
    st.write_bytes(STATUS_1124.encode())
    def n(v):
        r = subprocess.run([_bash(), "-c", f"grep -c ' {v}-RECORDS-READY at a49c964032cd ' '{st.as_posix()}'"], capture_output=True, text=True, encoding="utf-8")
        return r.stdout.strip()
    assert n("K2DC02") == "1"                                                      # E3's records are ready despite the D1 FAIL line
    assert n("K2DC02_d1") == "0"                                                   # D1's are not: never read as E3's
