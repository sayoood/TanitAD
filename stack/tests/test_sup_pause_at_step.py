"""Opt-in one-shot PAUSE of the refcv7 supervisor + the chain's ``-PAUSED`` event (early D1, PI 2026-11-11), CPU only.

PI: *"yes, D1 is covered, read it early"* -- E3 must stop cleanly once ``ckpt_full_30000.pt`` is written AND its step-30,000
eval row is in ``metrics.jsonl``, let D1 train on the same GPU, and then RESUME as the same run. The wiring under test:

* ``stack/ops/sup_refcv7.sh``: the manifest variable ``PAUSE_AT_STEP`` (default OFF). When both artifacts exist (and are quiet)
  the supervisor SIGTERMs the trainer, writes ``PAUSED.json`` (step, ckpt path + sha256), writes NO done-marker, exits 7 and does
  not relaunch. ``PAUSED.json`` present = the pause is spent: the next start on the same manifest RESUMES.
* ``stack/ops/r9e1_gate3/r9gate3_chain.sh``: ``<ARM>-PAUSED`` (rc 7 + latch) is neither ``-DONE`` nor ``-FAILED``; the manifest
  line ``PAUSE_AT_STEP=`` is emitted only when opted in and never for a ``*_d1`` arm.

Every check has a deliberate-regression arm (the line that emits it is edited out -> the invariant is violated). The supervisor
is the REAL script run by bash against a fake launch gate and fake trainer (the style of test_sup_refcv7_zerograd.py); the chain
blocks are cut out of the REAL chain text and run in bash with stubs. Skipped, not failed, when bash is unusable.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUP = ROOT / "ops" / "sup_refcv7.sh"
CHAIN = ROOT / "ops" / "r9e1_gate3" / "r9gate3_chain.sh"
ARM = "pausetest"
STEPS = 10
AT = 5


def _find_bash():
    for p in (os.environ.get("TANITAD_TEST_BASH"), r"C:\Program Files\Git\bin\bash.exe", shutil.which("bash")):
        if p and os.path.isfile(p) and "system32" not in p.replace("\\", "/").lower():
            try:
                r = subprocess.run([p, "-c", "command -v tee date grep sleep nohup >/dev/null && echo ok"],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
            except Exception:
                continue
            if r.returncode == 0 and "ok" in r.stdout:
                return p
    return None


BASH = _find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="no usable bash on this box")


def _posix(p) -> str:
    return Path(p).as_posix()


def _lf(dst: Path, text: str) -> Path:
    dst.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))
    return dst


def _src(p: Path) -> str:
    return p.read_bytes().decode("utf-8").replace("\r\n", "\n")


# ------------------------------------------------------------------ the supervisor rig
FAKE_LAUNCH_GATE = '''\
import json, os, runpy, sys
def arg(flag):
    a = sys.argv
    return a[a.index(flag) + 1] if flag in a else None
out = arg("--json")
if out:
    with open(out, "w") as f:
        json.dump({"verdict": "MATCH", "reasons": []}, f)
if sys.argv[1] == "exec":
    argv = json.load(open(arg("--argv-file")))
    argv = argv["argv"] if isinstance(argv, dict) else argv
    sys.argv = [os.environ["FAKE_TRAINER"]] + argv        # the REAL gate exec()s: the trainer IS this pid
    runpy.run_path(os.environ["FAKE_TRAINER"], run_name="__main__")
sys.exit(0)
'''

# first launch (no ckpt.pt): rows 1..AT, ckpt.pt, ckpt_full_AT.pt, an eval row (unless FAKE_NO_EVAL), then either LINGER (sleep, to be
# stopped by the pause) or finish the run (FAKE_FINISH_S seconds later). A launch that finds ckpt.pt RESUMES: it lingers
# FAKE_RESUME_LINGER_S (so a pause that wrongly fires a second time has time to) and then finishes the run.
FAKE_TRAINER = '''\
import json, os, sys, time
a = sys.argv[1:]
out = a[a.index("--out") + 1]
steps = int(a[a.index("--steps") + 1])
at = int(os.environ["FAKE_AT"])
with open(os.environ["FAKE_LAUNCH_LOG"], "a") as f:
    f.write("launch\\n")
os.makedirs(out, exist_ok=True)
def rows(lo, hi, mode):
    with open(os.path.join(out, "metrics.jsonl"), mode) as f:
        for s in range(lo, hi + 1):
            f.write(json.dumps({"step": s, "loss": 1.0 / s}) + "\\n")
def finish(lo):
    rows(lo, steps, "a")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"done": True, "step": steps, "arm": "x"}, f)
if os.path.exists(os.path.join(out, "ckpt.pt")):                    # RESUME
    time.sleep(float(os.environ.get("FAKE_RESUME_LINGER_S", "0")))
    finish(at + 1)
    sys.exit(0)
rows(1, at, "w")
open(os.path.join(out, "ckpt.pt"), "wb").write(b"ckpt-pt-bytes")
if os.environ.get("FAKE_NO_FULL") != "1":
    open(os.path.join(out, "ckpt_full_%d.pt" % at), "wb").write(b"ckpt-full-bytes")
if os.environ.get("FAKE_NO_EVAL") != "1":
    with open(os.path.join(out, "metrics.jsonl"), "a") as f:
        f.write(json.dumps({"step": at, "eval_loss": 0.5, "eval_traj": 0.25}) + "\\n")
if os.environ.get("FAKE_FINISH_S"):
    time.sleep(float(os.environ["FAKE_FINISH_S"]))
    finish(at + 1)
    sys.exit(0)
time.sleep(600)
'''


class Rig:
    def __init__(self, tmp: Path, extra="PAUSE_AT_STEP=%d" % AT, env=None):
        self.tmp = tmp
        self.code = tmp / "code"
        (self.code / "stack" / "scripts").mkdir(parents=True)
        (self.code / "stack" / "scripts" / "launch_gate.py").write_text(FAKE_LAUNCH_GATE)
        self.out = tmp / "out"
        self.runs = tmp / "runs.d"
        self.runs.mkdir()
        self.trainer = tmp / "fake_trainer.py"
        self.trainer.write_text(FAKE_TRAINER)
        self.launch_log = tmp / "launches.log"
        (tmp / "tok.json").write_text("{}")
        (tmp / "argv.json").write_text(json.dumps({"argv": ["--out", _posix(self.out), "--steps", str(STEPS), "--arm", "x"]}))
        (self.runs / ("%s.env" % ARM)).write_text(
            "CODE='%s'\nPYTHON='%s'\nGATE_TOKEN='%s'\nGATE_ARGV_FILE='%s'\nPOLL_S=2\nMAX_RELAUNCH=3\nZEROGRAD_GATE=off\n"
            "PAUSE_POLL_S=1\nPAUSE_SETTLE_S=0\n%s\n" % (_posix(self.code), _posix(sys.executable), _posix(tmp / "tok.json"),
                                                        _posix(tmp / "argv.json"), extra))
        stub = tmp / "stubbin"
        stub.mkdir()
        if shutil.which("flock") is None:
            f = stub / "flock"
            f.write_bytes(b"#!/bin/sh\nexit 0\n")
            f.chmod(f.stat().st_mode | stat.S_IEXEC)
        self.stub, self.env = stub, dict(env or {})
        self.script = _lf(tmp / "sup.sh", _src(SUP))

    @property
    def launches(self) -> int:
        return len(self.launch_log.read_text().split()) if self.launch_log.exists() else 0

    def marker(self, name):
        p = self.out / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    @property
    def suplog(self) -> str:
        p = self.out / ("sup_%s.log" % ARM)
        return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

    def run(self, script=None, **env):
        e = dict(os.environ)
        e.update(FAKE_TRAINER=_posix(self.trainer), FAKE_LAUNCH_LOG=_posix(self.launch_log), FAKE_AT=str(AT),
                 PATH=str(self.stub) + os.pathsep + e.get("PATH", ""))
        e.update(self.env)
        e.update({k: str(v) for k, v in env.items()})
        r = subprocess.run([BASH, _posix(script or self.script), ARM, _posix(self.runs)], env=e,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
        self.last = r
        return r.returncode


def _mutant(tmp: Path, old: str, new: str) -> Path:
    text = _src(SUP)
    assert text.count(old) == 1, "the line moved (%d hits): update this test with it: %r" % (text.count(old), old)
    return _lf(tmp / "sup_mutant.sh", text.replace(old, new))


def _paused(r: Rig, rc: int) -> bool:
    """The pause outcome in ONE predicate: rc 7, the latch, no done-marker, no relaunch."""
    return rc == 7 and r.marker("PAUSED.json") is not None and r.marker("summary.json") is None and r.launches == 1


def _first_launch_finishes(r: Rig, **env):
    # the trainer finishes the run 5 s after writing its artifacts: ample time for a (wrong) pause to fire at PAUSE_POLL_S=1
    return r.run(FAKE_FINISH_S=5, **env)


# ------------------------------------------------------------------ (1) the pause
def test_pause_fires_when_both_ckpt_full_and_eval_row_exist_and_writes_the_latch_not_a_done_marker(tmp_path):
    r = Rig(tmp_path)
    rc = r.run()
    assert _paused(r, rc), (rc, r.suplog)
    latch = r.marker("PAUSED.json")
    assert latch["paused_at_step"] == AT and "step" not in latch and "final_step" not in latch and latch["arm"] == ARM
    import hashlib
    assert latch["ckpt_sha256"] == hashlib.sha256(b"ckpt-full-bytes").hexdigest()
    assert latch["ckpt"].replace("\\", "/").endswith("ckpt_full_%d.pt" % AT)
    assert not (r.out / "summary.json").exists() and not (r.out / "ZEROGRAD_FAIL.json").exists()
    assert "ZZPAUSED-%s-%dZZ" % (ARM, AT) in r.suplog and "relaunching" not in r.suplog


def test_no_eval_row_means_no_pause(tmp_path):
    r = Rig(tmp_path)
    assert _first_launch_finishes(r, FAKE_NO_EVAL=1) == 0, r.suplog
    assert r.marker("PAUSED.json") is None and r.marker("summary.json")["done"] is True and "PAUSE:" not in r.suplog


def test_no_ckpt_full_means_no_pause(tmp_path):
    r = Rig(tmp_path)
    assert _first_launch_finishes(r, FAKE_NO_FULL=1) == 0, r.suplog
    assert r.marker("PAUSED.json") is None and r.marker("summary.json")["done"] is True and "PAUSE:" not in r.suplog


def test_checkpoints_still_being_written_means_no_pause_yet(tmp_path):
    r = Rig(tmp_path, extra="PAUSE_AT_STEP=%d\nPAUSE_SETTLE_S=3600" % AT)        # the later assignment wins: nothing is ever quiet
    assert _first_launch_finishes(r) == 0, r.suplog
    assert r.marker("PAUSED.json") is None and r.marker("summary.json")["done"] is True


# ------------------------------------------------------------------ (2) one-shot: the latch makes the next start RESUME
def test_second_start_with_the_latch_resumes_into_the_same_out_and_does_not_pause_again(tmp_path):
    r = Rig(tmp_path)
    assert _paused(r, r.run()), r.suplog
    # the resumed trainer lingers 5 s with the step-AT artifacts all still on disk: a second pause would fire inside that window
    rc = r.run(FAKE_RESUME_LINGER_S=5)
    assert rc == 0, (rc, r.suplog)
    assert r.launches == 2                                              # one launch before the pause, ONE resume launch
    assert r.marker("summary.json")["done"] is True and r.marker("summary.json")["final_step"] == STEPS
    assert "already spent" in r.suplog and r.suplog.count("ZZPAUSED-") == 1
    assert (r.out / "gate_verify_1.json").exists()                      # the resume went through the gate verify again


# ------------------------------------------------------------------ (3) default OFF, byte-identical behaviour
@pytest.mark.parametrize("extra", ["", "PAUSE_AT_STEP="])
def test_default_unset_or_empty_never_pauses_even_with_every_artifact_on_disk(tmp_path, extra):
    r = Rig(tmp_path, extra=extra)
    assert _first_launch_finishes(r) == 0, r.suplog
    assert r.marker("PAUSED.json") is None and r.marker("summary.json")["done"] is True and "PAUSE" not in r.suplog


def test_pause_variables_are_not_inherited_from_the_callers_environment(tmp_path):
    r = Rig(tmp_path, extra="")
    assert _first_launch_finishes(r, PAUSE_AT_STEP=AT) == 0, r.suplog
    assert r.marker("PAUSED.json") is None and "PAUSE" not in r.suplog


@pytest.mark.parametrize("bad", ["abc", "05", "-3", "1.5", "%d" % STEPS, "%d" % (STEPS + 5)])
def test_a_bad_pause_step_refuses_instead_of_disabling_or_pausing(tmp_path, bad):
    r = Rig(tmp_path, extra="PAUSE_AT_STEP=%s" % bad)
    assert r.run() == 2 and r.launches == 0 and "REFUSED: PAUSE_AT_STEP" in r.last.stdout


# ------------------------------------------------------------------ (4) the deliberate-regression arms: edit out the line, the check goes RED
def test_MUTATION_without_the_eval_row_check_it_pauses_before_the_eval_is_written(tmp_path):
    m = _mutant(tmp_path, "if not have_eval:", "if False:")
    r = Rig(tmp_path)
    assert _paused(r, r.run(script=m, FAKE_NO_EVAL=1)), r.suplog       # the real script does not pause here (test above)


def test_MUTATION_without_the_ckpt_full_check_it_pauses_with_no_full_checkpoint(tmp_path):
    text = _src(SUP)
    for old, new in (('  [ -f "${OUT_DIR}/ckpt_full_${PAUSE_AT}.pt" ] || return 0\n', "  true\n"),
                     ("for p in (full, ck):", "for p in (ck,):"),
                     ("if not (quiet(full) and quiet(ck)):", "if not quiet(ck):")):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    m = _lf(tmp_path / "sup_mutant.sh", text)
    r = Rig(tmp_path)
    rc = r.run(script=m, FAKE_NO_FULL=1)
    assert rc == 7 and r.launches == 1 and r.marker("summary.json") is None, r.suplog        # paused WITHOUT the checkpoint D1 needs


def test_MUTATION_without_the_settle_check_it_pauses_inside_a_checkpoint_write(tmp_path):
    m = _mutant(tmp_path, "if not (quiet(full) and quiet(ck)):", "if False:")
    r = Rig(tmp_path, extra="PAUSE_AT_STEP=%d\nPAUSE_SETTLE_S=3600" % AT)
    assert _paused(r, r.run(script=m)), r.suplog


def test_MUTATION_without_the_latch_check_every_start_pauses_again_and_E3_never_resumes(tmp_path):
    m = _mutant(tmp_path, 'if [ -f "$PAUSE_LATCH" ]; then', "if false; then")
    r = Rig(tmp_path)
    assert _paused(r, r.run(script=m)), r.suplog
    rc = r.run(script=m, FAKE_RESUME_LINGER_S=5)
    assert rc == 7 and r.suplog.count("ZZPAUSED-") == 2            # the mutant paused a SECOND time; the real script resumes (test above)


def test_MUTATION_a_default_that_arms_the_pause_changes_every_unopted_run(tmp_path):
    m = _mutant(tmp_path, 'PAUSE_AT="${PAUSE_AT_STEP:-}"', 'PAUSE_AT="${PAUSE_AT_STEP:-%d}"' % AT)
    r = Rig(tmp_path, extra="")
    assert _paused(r, r.run(script=m)), r.suplog


def test_MUTATION_without_the_latch_write_nothing_records_the_pause(tmp_path):
    m = _mutant(tmp_path, "os.replace(tmp, latch)", "pass")
    r = Rig(tmp_path)
    rc = r.run(script=m)
    assert rc == 7 and r.marker("PAUSED.json") is None            # the real script writes it (first test): E3 would pause again on resume


def test_pause_code_sits_in_the_poll_loop_and_never_writes_the_done_marker():
    text = _src(SUP)
    body = text[text.index("\ndo_pause() {"):text.index("\npause_check() {")]
    assert "write_done_marker" not in body and "finalize_run" not in body and "exit 7" in body
    assert text.count("pause_check\n") == 1 and "pause_check" in text[text.index("# ---- poll"):]


# ------------------------------------------------------------------ (5) the chain
def _chain_block(start_marker: str, end_marker: str) -> str:
    t = _src(CHAIN)
    a = t.index(start_marker)
    b = t.index(end_marker, a)
    return t[a:b]


def _run_bash(tmp: Path, script: str, env=None):
    f = _lf(tmp / "blk.sh", script)
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([BASH, _posix(f)], env=e, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)


CLASSIFY_START = """  if [ -s "$OUTD/summary.json" ] && grep -q '"done"[[:space:]]*:[[:space:]]*true' "$OUTD/summary.json" && grep -q '"zerograd_gate"'"""
CLASSIFY_END = '\ndone\n[ -n "$PASSED" ]'


def _classify(tmp: Path, rc: int, files: dict, chain_text=None):
    """Run the chain's end-of-arm classification on a fake run dir -> the event name it logs."""
    t = chain_text if chain_text is not None else _src(CHAIN)
    block = t[t.index(CLASSIFY_START):t.index(CLASSIFY_END)]
    outd = tmp / ("o%d" % len(list(tmp.iterdir())))
    outd.mkdir()
    for n, c in files.items():
        (outd / n).write_text(c)
    ev = tmp / "events.txt"
    ev.write_text("")
    script = ('set -u\nev() { echo "$1" >> %s; }\nA=V; ARM=r9e3_V; R=/r; TRAINED=""; OUTD=%s; RC=%d\n%s\n'
              % (_posix(ev), _posix(outd), rc, block))
    r = _run_bash(tmp, script)
    assert r.returncode == 0, r.stderr
    return ev.read_text().split()


def test_chain_logs_PAUSED_not_FAILED_or_DONE_for_rc7_with_the_latch(tmp_path):
    assert _classify(tmp_path, 7, {"PAUSED.json": "{}"}) == ["V-PAUSED"]


def test_chain_still_classifies_the_other_outcomes_as_before(tmp_path):
    assert _classify(tmp_path, 7, {}) == ["V-FAILED"]                                   # rc 7 WITHOUT the latch is not a pause
    assert _classify(tmp_path, 1, {"PAUSED.json": "{}"}) == ["V-FAILED"]                # a crash of the resumed run is a failure
    assert _classify(tmp_path, 0, {"summary.json": '{"done": true, "zerograd_gate": {}}'}) == ["V-DONE"]
    assert _classify(tmp_path, 6, {"ZEROGRAD_FAIL.json": "{}", "PAUSED.json": "{}"}) == ["V-ZEROGRAD-FAIL"]
    assert _classify(tmp_path, 5, {}) == ["V-REFUSED"]


def test_MUTATION_chain_without_the_PAUSED_branch_logs_FAILED_for_a_paused_arm(tmp_path):
    t = _src(CHAIN)
    lines = [l for l in t.split("\n") if '"$A-PAUSED"' in l]
    assert len(lines) == 1
    assert _classify(tmp_path, 7, {"PAUSED.json": "{}"}, chain_text=t.replace(lines[0] + "\n", "")) == ["V-FAILED"]


def test_chain_PAUSED_branch_comes_before_FAILED_and_the_lock_is_released_first():
    t = _src(CHAIN)
    assert t.index('"$A-PAUSED"') < t.index('"$A-FAILED"')
    assert t.index("flock -u 9; exec 9>&-     # release the GPU lock") < t.index('"$A-PAUSED"')


def _manifest(tmp: Path, arm: str, env: dict, pause_file=None, chain_text=None):
    t = chain_text if chain_text is not None else _src(CHAIN)
    block = t[t.index("  # PAUSE-MANIFEST-BEGIN"):t.index("  # PAUSE-MANIFEST-END")]
    r = tmp / "R"
    (r / "runs.d").mkdir(parents=True, exist_ok=True)
    (r / "gate3").mkdir(parents=True, exist_ok=True)
    if pause_file is not None:
        (r / "gate3" / "PAUSE_AT_STEP").write_text(pause_file)
    mf = r / "runs.d" / ("r9e3_%s.env" % arm)
    mf.write_text("MAX_RELAUNCH=4\nPOLL_S=300\n")
    ev = tmp / "events.txt"
    ev.write_text("")
    script = ('set -u\nev() { echo "$1" >> %s; }\nflock() { :; }\nR=%s; A=%s; ARM=r9e3_%s\nfor X in 1; do\n%s\necho CONTINUED-NOT\ndone\n'
              % (_posix(ev), _posix(r), arm, arm, block))
    p = _run_bash(tmp, script, env)
    assert p.returncode == 0, p.stderr
    return (mf.read_text() if mf.exists() else None), ev.read_text().split(), p.stdout


def test_chain_manifest_has_no_pause_line_by_default(tmp_path):
    env = {"R9_PAUSE_AT_STEP": ""}
    text, events, _ = _manifest(tmp_path, "V", env)
    assert text == "MAX_RELAUNCH=4\nPOLL_S=300\n" and events == []


def test_chain_manifest_gets_the_pause_line_from_the_file_or_the_env_but_never_for_a_D1_arm(tmp_path):
    text, events, _ = _manifest(tmp_path, "V", {"R9_PAUSE_AT_STEP": ""}, pause_file="30000\n")
    assert text.endswith("PAUSE_AT_STEP=30000\n") and events == []
    sub = tmp_path / "e"; sub.mkdir()
    text, _, _ = _manifest(sub, "V", {"R9_PAUSE_AT_STEP": "12000"})
    assert text.endswith("PAUSE_AT_STEP=12000\n")
    sub2 = tmp_path / "d"; sub2.mkdir()
    text, events, _ = _manifest(sub2, "V_d1", {"R9_PAUSE_AT_STEP": "30000"})
    assert "PAUSE_AT_STEP" not in text and events == []


def test_chain_refuses_a_malformed_pause_step_and_removes_the_manifest(tmp_path):
    text, events, out = _manifest(tmp_path, "V", {"R9_PAUSE_AT_STEP": "30k"})
    assert text is None and events == ["V-BADPAUSE"] and "CONTINUED-NOT" not in out


def test_MUTATION_chain_without_the_manifest_line_drops_the_pause_for_the_arm_that_opted_in(tmp_path):
    t = _src(CHAIN)
    old = 'echo "PAUSE_AT_STEP=$PAUSE_AT" >> $R/runs.d/$ARM.env'
    assert t.count(old) == 1
    text, _, _ = _manifest(tmp_path, "V", {"R9_PAUSE_AT_STEP": "30000"}, chain_text=t.replace(old, ":"))
    assert "PAUSE_AT_STEP" not in text                         # the real chain writes it (test above)


def test_the_files_are_LF_only():
    for p in (SUP, CHAIN):
        assert b"\r" not in p.read_bytes(), p
