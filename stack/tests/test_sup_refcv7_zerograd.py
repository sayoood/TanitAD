"""Shell-level contract of the zero-gradient gate hook in ``stack/ops/sup_refcv7.sh`` (G5), 2026-10-18.

``stack/scripts/zerograd_gate.py`` landed with NOTHING calling it (BACKLOG ZG-WIRE). The supervisor now runs it on the
final checkpoint just BEFORE it writes the done-marker. This file drives the REAL supervisor (bash) against a fake
launch gate + fake trainer + the REAL zerograd gate, so the claims are about the script that will run on the box:

* clean run            -> done-marker WITH a ``zerograd_gate`` record (status PASS), exit 0, one launch;
* untrained head       -> NO supervisor done-marker, ``ZEROGRAD_FAIL.json`` (embedding the gate report) naming the head,
                          exit 6, ONE launch -- and no RELAUNCH, even when the supervisor is restarted (with or without
                          the marker; with or without the trainer's own ``summary.json``);
* no optimizer state   -> done-marker with status NOT_APPLICABLE and the words "not applicable (no optimizer state)";
* the gate cannot look -> (unreadable ckpt / no gate JSON / exit code disagrees with the verdict / gate script missing /
                          a crash that exits 1) is UNEVALUATED and fails CLOSED exactly like FAIL: a gate that did not
                          run is not a pass, and a traceback (exit 1) is not a FAIL either;
* ``ZEROGRAD_ALLOW``   -> a frozen-by-design head is exempted, logged and recorded (array and one-per-line string);
  ``ZEROGRAD_GATE=off`` -> logged loudly and recorded; a typo in it REFUSES (exit 2) instead of disabling;
* trainer-written ``summary.json`` ({"done": true}, no zerograd record) is NOT "FINISHED": a fresh supervisor gates it,
  and does so WITHOUT launching the trainer;
* MUTATION (the RED arm): with the hook line removed, the untrained-head scenario WRITES the done-marker and exits 0 --
  so every FAIL assertion above is read from a script that can say otherwise.

Needs ``bash`` (Git-for-Windows bash or any Linux bash) and torch; ``flock`` is stubbed when the box has none. Skipped,
not failed, when bash is unusable.
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

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SUP = ROOT / "ops" / "sup_refcv7.sh"
GATE = ROOT / "scripts" / "zerograd_gate.py"
ARM = "zgtest"
STEPS = 4
HOOK = 'zerograd_check "$fs"   # ZG-HOOK'


# ------------------------------------------------------------------ the box
def _find_bash():
    cands = []
    for p in (os.environ.get("TANITAD_TEST_BASH"), r"C:\Program Files\Git\bin\bash.exe", shutil.which("bash")):
        if p and os.path.isfile(p) and "system32" not in p.replace("\\", "/").lower():
            cands.append(p)
    for p in cands:
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


def _lf(src: Path, dst: Path, text=None):
    t = text if text is not None else src.read_bytes().decode("utf-8")
    dst.write_bytes(t.replace("\r\n", "\n").encode("utf-8"))
    return dst


# ------------------------------------------------------------------ checkpoints (built once)
class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = torch.nn.Linear(4, 6)
        self.head = torch.nn.Linear(6, 2)
        self.dead = torch.nn.Linear(6, 3)          # registered, in the optimizer, NEVER in the loss

    def forward(self, x, use_dead):
        h = torch.relu(self.enc(x))
        y = self.head(h).sum()
        return y + self.dead(h).sum() if use_dead else y


def _ckpt(kind: str):
    stale = kind == "stale"
    kind = "clean" if stale else kind
    torch.manual_seed(0)
    m = _Net()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        m(torch.randn(8, 4), use_dead=(kind == "clean")).backward()
        opt.step()
    ck = {"model": m.state_dict(), "step": STEPS - 2 if stale else STEPS,
          "param_names": [n for n, _ in m.named_parameters()]}
    if kind != "noopt":
        ck["opt"] = opt.state_dict()
    return ck


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    d = tmp_path_factory.mktemp("zg_ckpts")
    for kind in ("clean", "dead", "noopt", "stale"):
        torch.save(_ckpt(kind), d / ("%s.pt" % kind))
    (d / "garbage.pt").write_bytes(b"this is not a torch checkpoint")
    return d


# ------------------------------------------------------------------ the rig
FAKE_LAUNCH_GATE = '''\
import json, os, subprocess, sys
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
    sys.exit(subprocess.call([sys.executable, os.environ["FAKE_TRAINER"]] + argv))
sys.exit(0)
'''

FAKE_TRAINER = '''\
import json, os, shutil, sys
a = sys.argv[1:]
out = a[a.index("--out") + 1]
steps = int(a[a.index("--steps") + 1])
with open(os.environ["FAKE_LAUNCH_LOG"], "a") as f:
    f.write("launch\\n")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "metrics.jsonl"), "w") as f:
    for s in range(1, steps + 1):
        f.write(json.dumps({"step": s, "loss": 1.0 / s}) + "\\n")
if os.environ.get("FAKE_CKPT"):
    shutil.copyfile(os.environ["FAKE_CKPT"], os.path.join(out, "ckpt.pt"))
if os.environ.get("FAKE_TRAINER_SUMMARY") == "1":          # the REAL trainer writes this itself (refc_v3_train.py)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"done": True, "step": steps, "arm": "x"}, f)
'''


class Rig:
    def __init__(self, tmp: Path, ckpts: Path, ckpt="clean", trainer_summary=True, manifest_extra=""):
        self.tmp, self.ckpts = tmp, ckpts
        self.code = tmp / "code"
        (self.code / "stack" / "scripts").mkdir(parents=True)
        (self.code / "stack" / "scripts" / "launch_gate.py").write_text(FAKE_LAUNCH_GATE)
        shutil.copyfile(GATE, self.code / "stack" / "scripts" / "zerograd_gate.py")
        self.out = tmp / "out"
        self.runs = tmp / "runs.d"
        self.runs.mkdir()
        self.trainer = tmp / "fake_trainer.py"
        self.trainer.write_text(FAKE_TRAINER)
        self.launch_log = tmp / "launches.log"
        self.ckpt_kind, self.trainer_summary = ckpt, trainer_summary
        (tmp / "tok.json").write_text("{}")
        (tmp / "argv.json").write_text(json.dumps({"argv": ["--out", _posix(self.out), "--steps", str(STEPS),
                                                            "--arm", "x"]}))
        (self.runs / ("%s.env" % ARM)).write_text(
            "CODE='%s'\nPYTHON='%s'\nGATE_TOKEN='%s'\nGATE_ARGV_FILE='%s'\nPOLL_S=1\nMAX_RELAUNCH=3\n%s\n" % (
                _posix(self.code), _posix(sys.executable), _posix(tmp / "tok.json"), _posix(tmp / "argv.json"),
                manifest_extra))
        stub = tmp / "stubbin"
        stub.mkdir()
        if shutil.which("flock") is None:                  # Git-for-Windows has no flock: the lock is not under test
            f = stub / "flock"
            f.write_bytes(b"#!/bin/sh\nexit 0\n")
            f.chmod(f.stat().st_mode | stat.S_IEXEC)
        self.stub = stub
        self.script = _lf(SUP, tmp / "sup.sh")

    # -- state
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

    def run(self, script=None, ckpt=None, trainer_summary=None):
        kind = self.ckpt_kind if ckpt is None else ckpt
        ts = self.trainer_summary if trainer_summary is None else trainer_summary
        env = dict(os.environ)
        env.update(FAKE_TRAINER=_posix(self.trainer), FAKE_LAUNCH_LOG=_posix(self.launch_log),
                   FAKE_TRAINER_SUMMARY="1" if ts else "0",
                   FAKE_CKPT="" if kind is None else _posix(self.ckpts / ("%s.pt" % kind)),
                   PATH=str(self.stub) + os.pathsep + env.get("PATH", ""))
        r = subprocess.run([BASH, _posix(script or self.script), ARM, _posix(self.runs)], env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
        self.last = r
        return r.returncode


def _rig(tmp_path, ckpts, **kw) -> Rig:
    return Rig(tmp_path, ckpts, **kw)


# ------------------------------------------------------------------ the contract
def test_clean_run_writes_the_done_marker_with_a_zerograd_pass_record(tmp_path, ckpts):
    r = _rig(tmp_path, ckpts, ckpt="clean")
    assert r.run() == 0, r.suplog
    done = r.marker("summary.json")
    assert done["done"] is True and done["final_step"] == STEPS and done["gate_token"]       # the SUPERVISOR's marker
    zg = done["zerograd_gate"]
    assert zg["status"] == "PASS" and zg["n_untrained"] == 0 and zg["ckpt_step_is_final"] is True
    assert zg["param_names_source"] == "ckpt" and zg["n_registered"] == 6
    assert not (r.out / "ZEROGRAD_FAIL.json").exists()
    assert r.launches == 1
    assert (r.out / "zerograd_gate.json").exists() and (r.out / "zerograd_gate.txt").exists()
    assert "ZZZG-%s-PASSZZ" % ARM in r.suplog


def test_a_checkpoint_older_than_the_requested_steps_is_flagged_loudly_not_silently_gated(tmp_path, ckpts):
    r = _rig(tmp_path, ckpts, ckpt="stale")
    assert r.run() == 0, r.suplog
    zg = r.marker("summary.json")["zerograd_gate"]
    assert zg["status"] == "PASS" and zg["ckpt_step"] == STEPS - 2 and zg["ckpt_step_is_final"] is False
    assert "checkpoint whose step is BELOW" in r.suplog


@pytest.mark.parametrize("trainer_summary", [True, False])
def test_untrained_head_refuses_the_marker_writes_the_fail_marker_and_never_relaunches(
        tmp_path, ckpts, trainer_summary):
    r = _rig(tmp_path, ckpts, ckpt="dead", trainer_summary=trainer_summary)
    assert r.run() == 6, r.suplog
    fail = r.marker("ZEROGRAD_FAIL.json")
    assert fail["zerograd_fail"] is True and fail["status"] == "FAIL"
    names = sorted(n for u in fail["zerograd_summary"]["untrained"] for n in u["names"])
    assert names == ["dead.bias", "dead.weight"], names
    assert fail["gate_report"]["verdict"] == "FAIL" and len(fail["gate_report"]["offenders"]) == 2   # the gate JSON, embedded
    summ = r.marker("summary.json")
    if trainer_summary:        # the trainer's own artifact is left UNTOUCHED -- the supervisor did not overwrite it
        assert summ == {"done": True, "step": STEPS, "arm": "x"}
    else:                      # ... and with no trainer summary there is NO done-marker at all
        assert summ is None
    assert r.launches == 1                                              # no relaunch
    assert "relaunching" not in r.suplog and "launch #2" not in r.suplog
    assert "ZZZGFAIL-%s" % ARM in r.suplog
    # ---- the resurrection trap: a restart (marker kept) exits 6 and does not launch or gate again
    before = (r.out / "ZEROGRAD_FAIL.json").read_bytes()
    assert r.run() == 6 and r.launches == 1
    assert "ZEROGRAD_FAIL.json present" in r.suplog or "ZEROGRAD_FAIL.json" in r.suplog
    assert (r.out / "ZEROGRAD_FAIL.json").read_bytes() == before        # not re-written either
    # ---- ... and a restart after the operator DELETED the marker re-gates, still without a launch
    (r.out / "ZEROGRAD_FAIL.json").unlink()
    assert r.run() == 6 and r.launches == 1
    assert r.marker("ZEROGRAD_FAIL.json")["status"] == "FAIL"


def test_a_trainer_written_summary_is_gated_by_a_fresh_supervisor_without_launching(tmp_path, ckpts):
    # the supervisor died between the trainer's completion and its own finalising: the trainer's summary.json says
    # done, there is no zerograd record, and the old startup test would have said "FINISHED" and exited 0.
    for kind, rc in (("dead", 6), ("clean", 0)):
        sub = tmp_path / kind
        sub.mkdir()
        r = _rig(sub, ckpts, ckpt=kind)
        r.out.mkdir()
        (r.out / "metrics.jsonl").write_text("".join(json.dumps({"step": s}) + "\n" for s in range(1, STEPS + 1)))
        shutil.copyfile(ckpts / ("%s.pt" % kind), r.out / "ckpt.pt")
        (r.out / "summary.json").write_text(json.dumps({"done": True, "step": STEPS, "arm": "x"}))
        assert r.run() == rc, r.suplog
        assert r.launches == 0, "the finished run was RELAUNCHED"
        assert "WITHOUT relaunching" in r.suplog
        if rc == 0:
            assert r.marker("summary.json")["zerograd_gate"]["status"] == "PASS"
            assert r.run() == 0 and "run is FINISHED" in r.suplog         # now it carries the record: FINISHED
        else:
            assert r.marker("ZEROGRAD_FAIL.json")["status"] == "FAIL"


def test_a_checkpoint_without_optimizer_state_is_recorded_not_applicable_never_silently_a_pass(tmp_path, ckpts):
    r = _rig(tmp_path, ckpts, ckpt="noopt")
    assert r.run() == 0, r.suplog
    zg = r.marker("summary.json")["zerograd_gate"]
    assert zg["status"] == "NOT_APPLICABLE" and "not applicable (no optimizer state)" in zg["reason"]
    assert zg["gate_exit_code"] == 2 and zg["gate_verdict"] == "ERROR"
    assert "ZZZG-%s-NOT_APPLICABLEZZ" % ARM in r.suplog and r.launches == 1


@pytest.mark.parametrize("why", ["garbage_ckpt", "no_ckpt", "gate_missing", "crash_exit1", "json_disagrees_with_rc"])
def test_a_gate_that_cannot_look_is_unevaluated_and_fails_closed_like_a_fail(tmp_path, ckpts, why):
    r = _rig(tmp_path, ckpts, ckpt={"garbage_ckpt": "garbage", "no_ckpt": None}.get(why, "clean"))
    gate = r.code / "stack" / "scripts" / "zerograd_gate.py"
    if why == "gate_missing":
        gate.unlink()
    elif why == "crash_exit1":          # a traceback exits 1 -- indistinguishable from FAIL by exit code, so: JSON or nothing
        gate.write_text("raise RuntimeError('simulated gate crash')\n")
    elif why == "json_disagrees_with_rc":
        gate.write_text("import json, sys\nout = sys.argv[sys.argv.index('--json-out') + 1]\n"
                        "json.dump({'gate': 'zerograd_gate', 'verdict': 'PASS'}, open(out, 'w'))\nsys.exit(1)\n")
    assert r.run() == 6, r.suplog
    fail = r.marker("ZEROGRAD_FAIL.json")
    assert fail["status"] == "UNEVALUATED", fail["reason"]
    assert r.launches == 1 and "relaunching" not in r.suplog
    assert r.marker("summary.json") == {"done": True, "step": STEPS, "arm": "x"}      # the trainer's, untouched
    if why == "garbage_ckpt":
        assert "unreadable" in fail["reason"]


def test_allow_list_exempts_frozen_by_design_parameters_and_is_logged_and_recorded(tmp_path, ckpts):
    r = _rig(tmp_path, ckpts, ckpt="dead", manifest_extra="ZEROGRAD_ALLOW=('dead\\..*')\n")      # bash array
    assert r.run() == 0, r.suplog
    zg = r.marker("summary.json")["zerograd_gate"]
    assert zg["status"] == "PASS" and zg["allow"] == ["dead\\..*"] and zg["unused_allow"] == []
    assert "zerograd gate allow pattern: dead\\..*" in r.suplog
    # a one-per-line STRING, with a pattern that matches nothing: still PASS, the typo is visible as unused
    sub = tmp_path / "scalar"
    sub.mkdir()
    r2 = _rig(sub, ckpts, ckpt="dead", manifest_extra="ZEROGRAD_ALLOW='dead\\..*\nnot_a_param\\..*'\n")
    assert r2.run() == 0, r2.suplog
    zg2 = r2.marker("summary.json")["zerograd_gate"]
    assert zg2["allow"] == ["dead\\..*", "not_a_param\\..*"] and zg2["unused_allow"] == ["not_a_param\\..*"]
    # an allow that covers only PART of the dead head still FAILS (fullmatch, fail closed)
    sub = tmp_path / "partial"
    sub.mkdir()
    r3 = _rig(sub, ckpts, ckpt="dead", manifest_extra="ZEROGRAD_ALLOW=('dead\\.weight')\n")
    assert r3.run() == 6
    assert [n for u in r3.marker("ZEROGRAD_FAIL.json")["zerograd_summary"]["untrained"] for n in u["names"]] == ["dead.bias"]


def test_a_failed_positive_control_fails_the_run(tmp_path, ckpts):
    r = _rig(tmp_path, ckpts, ckpt="clean", manifest_extra="ZEROGRAD_CONTROL=('enc\\..*' 'nothing_like_this\\..*')\n")
    assert r.run() == 6, r.suplog
    assert r.marker("ZEROGRAD_FAIL.json")["zerograd_summary"]["failed_controls"] == ["nothing_like_this\\..*"]


def test_gate_off_is_loud_and_recorded_and_a_typo_refuses_instead_of_disabling(tmp_path, ckpts):
    r = _rig(tmp_path, ckpts, ckpt="dead", manifest_extra="ZEROGRAD_GATE=off\n")
    assert r.run() == 0, r.suplog
    assert r.marker("summary.json")["zerograd_gate"]["status"] == "DISABLED"
    assert "ZEROGRAD GATE DISABLED" in r.suplog and not (r.out / "zerograd_gate.json").exists()
    sub = tmp_path / "typo"
    sub.mkdir()
    r2 = _rig(sub, ckpts, ckpt="dead", manifest_extra="ZEROGRAD_GATE=of\n")
    assert r2.run() == 2 and r2.launches == 0
    assert "REFUSED" in r2.last.stdout and "ZEROGRAD_GATE=of" in r2.last.stdout


def test_manifest_zerograd_vars_are_not_inherited_from_the_callers_environment(tmp_path, ckpts):
    # the manifest is the ONLY source (the 2026-09-27 STEPS lesson): an exported ZEROGRAD_GATE=off must not leak in
    r = _rig(tmp_path, ckpts, ckpt="dead")
    os.environ["ZEROGRAD_GATE"] = "off"
    try:
        assert r.run() == 6, r.suplog
    finally:
        del os.environ["ZEROGRAD_GATE"]


# ------------------------------------------------------------------ the RED arm
def test_MUTATION_without_the_hook_the_untrained_head_writes_the_done_marker(tmp_path, ckpts):
    text = SUP.read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert text.count(HOOK) == 1, "the ZG-HOOK line moved: update this test with it"
    mutant = _lf(SUP, tmp_path / "sup_mutant.sh", text.replace(HOOK, "ZG_STATUS=PASS   # MUTANT: the gate is not called"))
    r = _rig(tmp_path, ckpts, ckpt="dead", trainer_summary=False)
    assert r.run(script=mutant) == 0, r.suplog                       # the dead head sails through ...
    done = r.marker("summary.json")
    assert done["done"] is True and done["gate_token"]               # ... and the supervisor WROTE the done-marker
    assert done["zerograd_gate"]["status"] == "NOT_RUN"              # with no record of a gate (the red arm's tell)
    assert not (r.out / "ZEROGRAD_FAIL.json").exists()
    # and the SAME rig with the real script is refused: the assertion above is not a vacuous one
    sub = tmp_path / "real"
    sub.mkdir()
    r2 = _rig(sub, ckpts, ckpt="dead", trainer_summary=False)
    assert r2.run() == 6 and r2.marker("summary.json") is None


def test_the_hook_line_sits_in_finalize_run_and_both_completion_paths_call_it():
    text = SUP.read_bytes().decode("utf-8").replace("\r\n", "\n")
    body = text[text.index("\nfinalize_run() {"):text.index("\nmarker_step() {")]
    assert HOOK in body and "write_done_marker" in body
    assert body.index(HOOK) < body.index("write_done_marker")        # the gate runs BEFORE the marker
    # no call of write_done_marker outside finalize_run (a second path would bypass the gate)
    outside = text.replace(body, "")
    calls = [l for l in outside.splitlines() if l.strip().startswith("write_done_marker ")]
    assert calls == [], calls
    assert text.count('finalize_run "$fs"') == 3          # loop-top, after the trainer exits, and finalize-only mode
