"""r9e1_arm_read.sh <ARM> [REF]: the Thor-side per-arm read, driven in DRY mode (GPU step = the stub dump, GPU lock emulated with mkdir when the box has
no flock), exactly like ``test_r9_e1_floor.py`` drives r9e1_read.sh.

The floor is built IN-PROCESS in the real layout (``$RD/dump_C0``, ``$RD/EVAL_C0.json``, ``$RD/FLOOR.json`` ...) by the real r9_e1_dump-format stub ->
r9_e1_eval.py -> r9_e1_floor.py; the shell then runs the real wait / lock / dump / check / eval / verdict steps for a lever arm.  What is pinned:
the order of the steps, that the floor's C0 is NEVER re-dumped, that a non-floor reference is dumped once, that every abort is read from an artifact,
that the pins equal r9e1_read.sh's, and the verdict that comes out the other end.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

_STACK = Path(__file__).resolve().parents[1]
_OPS = _STACK / "ops" / "r9e1_read"
for _p in (_STACK / "scripts", _OPS):
    sys.path.insert(0, str(_p))
import r9_e1_eval as E                                   # noqa: E402
import r9_e1_floor as F                                  # noqa: E402
import r9_e1_verdict as V                                # noqa: E402
import r9e1_read_stub_dump as STUB                       # noqa: E402

SH = _OPS / "r9e1_arm_read.sh"
FLOOR_SH = _OPS / "r9e1_read.sh"
CLIPS = ["%012x" % (0xa00 + i) for i in range(6)]
BOOT = "50"


def _strict_json(path):
    def bad(tok):
        raise ValueError("non-finite JSON token %s" % tok)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=bad)


# ------------------------------------------------------------------------------------------------------------------ static pins
def test_arm_read_script_is_lf_and_its_pins_equal_the_floor_scripts():
    sh, fl = SH.read_text(encoding="utf-8"), FLOOR_SH.read_text(encoding="utf-8")
    assert "\r" not in sh and sh.startswith("#!/usr/bin/env bash\n")
    for var in ("SPLIT_SHA_PIN", "GATES_SHA"):
        a, b = re.search(r'^%s="([0-9a-f]{64})"' % var, sh, re.M), re.search(r'^%s="([0-9a-f]{64})"' % var, fl, re.M)
        assert a and b and a.group(1) == b.group(1), var
    for var in ("SPLIT", "S2", "GATES", "LOCKF"):                                                    # the same defaults: the same split, S2 windows, gates, lock
        a, b = re.search(r'^%s="\$\{R9E1_\w+:-([^}]+)\}"' % var, sh, re.M), re.search(r'^%s="\$\{R9E1_\w+:-([^}]+)\}"' % var, fl, re.M)
        assert a and b and a.group(1) == b.group(1), var
    assert '--split dev --s2-windows "$S2" --require-ema' in sh and '--split dev --s2-windows "$S2" --require-ema' in fl     # the same dump flags
    assert "flock -w" in sh and "LOCK_MAX_S:-43200" in sh                                           # >= 12 h on the lock
    assert "dump_arm C0" not in sh and "dump_arm C1" not in sh                                      # the floor arms are never re-dumped
    assert re.search(r'EVC="--windows \$SPLIT --split dev \$GATEARGS --boot \$BOOT --seed \$SEED"', sh)
    assert re.search(r'EVC="--windows \$SPLIT --split dev \$GATEARGS --boot \$BOOT --seed \$SEED"', fl)


def test_arm_read_script_flags_exist_in_the_scripts_it_calls():
    sh = SH.read_text(encoding="utf-8")

    def flags(p):
        return set(re.findall(r'add_argument\(\s*"(--[\w-]+)"', Path(p).read_text(encoding="utf-8")))
    want = {
        "r9_e1_dump.py": ["--ckpt", "--config", "--windows", "--split", "--s2-windows", "--require-ema", "--out"],
        "r9_e1_eval.py": ["--packs", "--packs-b", "--rast-dir", "--windows", "--split", "--boot", "--seed", "--out", "--gates-json", "--gate", "--lift-valid"],
        "r9_e1_verdict.py": ["--floor", "--floor-eval-c0", "--paired", "--eval-arm", "--eval-ref", "--arm", "--ref", "--rast-arm", "--rast-ref", "--boot",
                             "--seed", "--out-dir", "--status", "--root", "--read-dir", "--out", "--dump", "--lift-valid"],
    }
    for name, fl in want.items():
        have = flags(_STACK / "scripts" / name)
        for f in fl:
            assert f in have, "%s has no %s" % (name, f)
            assert f in sh, "r9e1_arm_read.sh no longer passes %s (to %s)" % (f, name)
    assert set(want["r9_e1_dump.py"]) <= flags(_OPS / "r9e1_read_stub_dump.py")


# ------------------------------------------------------------------------------------------------------------------ the dry-run rig
def _bash():
    cands = [os.environ.get("R9E1_TEST_BASH"), "C:/Program Files/Git/usr/bin/bash.exe", "C:/Program Files/Git/bin/bash.exe", shutil.which("bash")]
    for b in cands:
        if not b:
            continue
        try:
            r = subprocess.run([b, "-c", 'test -f "$1" && command -v sha256sum >/dev/null && command -v awk >/dev/null', "x", SH.as_posix()],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode == 0:
            return b
    return None


def _run_dir(root, arm, spec, *, done=True):
    d = Path(root) / "runs" / ("r9e1-" + arm)
    d.mkdir(parents=True, exist_ok=True)
    s = {"done": done, "final_step": 4000, "steps_requested": 4000, "arm": "flat"}      # the SUPERVISOR's done-marker fields (sup_refcv7.sh)
    if done:
        s["zerograd_gate"] = {"ok": True}
    (d / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    (d / "model_ema.pt").write_text(json.dumps(dict({"step": 4000}, **spec)), encoding="utf-8")
    (d / "config.json").write_text("{}", encoding="utf-8")
    return d


def _write_mask(path):
    """A sidecar lift-valid mask for the six episodes of CLIPS (all cells observed): ``e<episode>`` -> bool [400, 60]."""
    import numpy as np
    np.savez_compressed(str(path), **{"e%d" % i: np.ones(E.LV_SHAPE, bool) for i in range(len(CLIPS))})


def _build_floor_template(tpl, with_mask=False):
    """The floor read, in the real layout, built ONCE per module (the dry tests copy it): the SAME flags the shell uses (--windows as the
    POSIX string the shell passes, so the floor's recorded window file equals the arm evals')."""
    tpl = Path(tpl)
    rd = tpl / "read"
    rd.mkdir(parents=True)
    lv = []
    if with_mask:                                                                   # the floor AND its evals on ONE mask (the shell must use the same)
        _write_mask(tpl / "mask.npz")
        lv = ["--lift-valid", (tpl / "mask.npz").as_posix()]
    split = tpl / "split.json"
    split.write_text(json.dumps({"DEV": {"sha12": CLIPS}, "HOLD": {"sha12": ["%012x" % 0xb00]}}), encoding="utf-8")
    (tpl / "s2.json").write_text("{}", encoding="utf-8")
    (tpl / "config.json").write_text("{}", encoding="utf-8")
    com = ["--windows", split.as_posix(), "--split", "dev", "--gate", "0.5", "--boot", BOOT, "--seed", "0"]
    for name, spec in (("C0", {"miss": [7, 8, 9]}), ("C1", {"miss": [7, 8]})):
        ck = tpl / (name + ".ckpt.json")
        ck.write_text(json.dumps(dict({"step": 4000}, **spec)), encoding="utf-8")
        assert STUB.main(["--ckpt", str(ck), "--config", str(tpl / "config.json"), "--windows", str(split), "--split", "dev",
                          "--s2-windows", str(tpl / "s2.json"), "--out", str(rd / ("dump_" + name)), "--require-ema"]) == 0
        assert E.main(["--packs", str(rd / ("dump_" + name) / "perc_det.pkl"), "--rast-dir", str(rd / ("dump_" + name) / "rast"), *com, *lv,
                       "--out", str(rd / ("EVAL_%s.json" % name))]) == 0
    assert E.main(["--packs", str(rd / "dump_C1" / "perc_det.pkl"), "--packs-b", str(rd / "dump_C0" / "perc_det.pkl"), *com, *lv,
                   "--out", str(rd / "EVAL_C1_vs_C0.json")]) == 0
    assert F.main(["--eval-c0", str(rd / "EVAL_C0.json"), "--eval-c1", str(rd / "EVAL_C1.json"), "--paired", str(rd / "EVAL_C1_vs_C0.json"),
                   "--rast-c0", str(rd / "dump_C0" / "rast"), "--rast-c1", str(rd / "dump_C1" / "rast"), "--boot", BOOT, *lv, "--out-dir", str(rd)]) == 0
    return tpl


@pytest.fixture(scope="module")
def floor_template(tmp_path_factory):
    return _build_floor_template(tmp_path_factory.mktemp("armread_floor") / "tpl")


class Dry:
    def __init__(self, tmp, bash, tpl):
        self.tmp, self.bash = Path(tmp), bash
        self.root = self.tmp / "r9e1"
        self.rd = self.root / "read"
        (self.root / "gate3").mkdir(parents=True)
        shutil.copytree(tpl / "read", self.rd)                                      # the floor's files (FLOOR.json, EVAL_*.json, dump_C0, dump_C1)
        self.status = self.root / "gate3" / "CHAIN_STATUS.txt"
        self.split = tpl / "split.json"
        (self.tmp / "split.sha256").write_text("%s  split.json" % hashlib.sha256(self.split.read_bytes()).hexdigest() + chr(10), encoding="utf-8")
        self.lockf = self.tmp / "gpu.lock"
        self.env = dict(os.environ, R9E1_DRY="1", R9E1_ROOT=self.root.as_posix(), R9E1_READ_DIR=self.rd.as_posix(),
                        R9E1_TREE=(self.root / "launch_tree").as_posix(), R9E1_PY=sys.executable, R9E1_LOCKF=self.lockf.as_posix(),
                        R9E1_POLL_S="1", R9E1_BOOT=BOOT, R9E1_DUMP_SCRIPT=(_OPS / "r9e1_read_stub_dump.py").as_posix(),
                        R9E1_DUMP_CWD=self.tmp.as_posix(), R9E1_EVAL_SCRIPT=(_STACK / "scripts" / "r9_e1_eval.py").as_posix(),
                        R9E1_PYTHONPATH=os.pathsep.join(str(p) for p in (_STACK, _STACK / "scripts", _STACK.parent / "taniteval")),
                        R9E1_GATE_ARGS="--gate 0.5", R9E1_SPLIT=self.split.as_posix(), R9E1_SPLIT_SHA_FILE=(self.tmp / "split.sha256").as_posix(),
                        R9E1_S2=(tpl / "s2.json").as_posix(), CUDA_VISIBLE_DEVICES="-1", R9E1_MAX_WAIT_S="900")
        self.proc = None
        self.out = None

    def chain(self, *ev):
        with open(self.status, "a", encoding="utf-8", newline="\n") as f:
            f.write("".join("2026-11-04T05:%02d:00Z %s\n" % (i % 60, e) for i, e in enumerate(ev)))

    def run_dir(self, arm, spec, **kw):
        return _run_dir(self.root, arm, spec, **kw)

    def ad(self, arm):
        return self.rd / ("arm_" + arm)

    def start(self, arm, ref=None, **env):
        self.out = open(self.tmp / ("sh_%s.out" % arm), "w", encoding="utf-8")
        args = [self.bash, SH.as_posix(), arm] + ([ref] if ref else [])
        self.proc = subprocess.Popen(args, env=dict(self.env, **env), stdout=self.out, stderr=subprocess.STDOUT)

    def events(self, arm):
        f = self.ad(arm) / "READ_STATUS.txt"
        return [ln.split(" ", 2)[1] for ln in f.read_text(encoding="utf-8").splitlines()] if f.exists() else []

    def log(self, arm):
        f = self.ad(arm) / "READ_STATUS.txt"
        return f.read_text(encoding="utf-8") if f.exists() else ""

    def wait(self, pred, timeout=240):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if pred():
                return True
            if self.proc.poll() is not None and not pred():
                time.sleep(0.5)
                return pred()
            time.sleep(0.5)
        return False

    def stdout(self):
        return Path(self.out.name).read_text(encoding="utf-8")

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=30)
        if self.out is not None:
            self.out.close()


@pytest.fixture
def dry(tmp_path, floor_template):
    b = _bash()
    if not b:
        pytest.skip("no usable bash on this box")
    d = Dry(tmp_path, b, floor_template)
    yield d
    d.stop()


def _subsequence(ev, order):
    it = iter(ev)
    return all(any(e == o for e in it) for o in order)


# ------------------------------------------------------------------------------------------------------------------ the tests
def test_dry_run_waits_then_reads_a_lever_arm_against_the_floors_c0_without_redumping_it(dry):
    c0_dump_before = (dry.rd / "dump_C0" / "dump.json").read_bytes()
    c0_mtime = (dry.rd / "dump_C0" / "dump.json").stat().st_mtime_ns
    dry.chain("START commit bbb tree /t arms [B1]")
    dry.start("B1")
    assert dry.wait(lambda: dry.events("B1").count("WAIT") >= 1, 90), (dry.events("B1"), dry.stdout())
    b1 = dry.run_dir("B1", {"miss": []}, done=False)
    dry.chain("B1-LAUNCH supervisor x; run dir %s" % b1.as_posix())
    assert dry.wait(lambda: dry.events("B1").count("WAIT") >= 2, 90), dry.events("B1")                 # a changed picture is logged once
    time.sleep(2)
    assert "LOCK-WAIT" not in dry.events("B1") and dry.proc.poll() is None and not (dry.rd / "arm_B1" / "dump_B1").exists()      # WAIT runs nothing
    dry.run_dir("B1", {"miss": []})                                                                      # the markers appear ...
    dry.chain("B1-DONE rc 0; ok")                                                                        # ... and the chain says so
    assert dry.wait(lambda: "DONE" in dry.events("B1"), 420), (dry.events("B1"), dry.stdout()[-2500:])
    ev = dry.events("B1")
    order = ["START", "WAIT", "WAIT", "READY", "INPUTS-OK", "LOCK-WAIT", "LOCK-TAKEN", "GPU-CHECK", "DUMP-START", "DUMP-DONE", "LOCK-RELEASED", "DUMP-CHECK",
             "EVAL-DONE", "EVAL-DONE", "VERDICT-DONE", "DONE"]
    assert _subsequence(ev, order), ev
    log = dry.log("B1")
    assert "DUMP-START B1" in log and "DUMP-START C0" not in log and "DUMP-START C1" not in log             # the floor's C0 is NEVER re-dumped
    assert "EVAL-SKIP C0" in log                                                                          # nor re-evaluated: its EVAL_C0.json is reused
    assert (dry.rd / "dump_C0" / "dump.json").read_bytes() == c0_dump_before and (dry.rd / "dump_C0" / "dump.json").stat().st_mtime_ns == c0_mtime
    assert dry.proc.wait(timeout=60) == 0 and (dry.ad("B1") / "READ_DONE").exists() and not (dry.ad("B1") / "READ_ABORTED").exists()
    assert not Path(str(dry.lockf) + ".d").exists()                                                      # the (emulated) lock is released
    assert re.search(r"ZZR9E1ARMREAD-DONEZZ", dry.stdout())
    v = _strict_json(dry.ad("B1") / "VERDICT_B1.json")
    assert v["status"] == "OK" and v["arm"] == "B1" and v["ref"] == "C0" and v["verdict"] == "DROP" and 1 in [f["clause"] for f in v["failing_clauses"]]
    assert v["floor"]["sha256"] == hashlib.sha256((dry.rd / "FLOOR.json").read_bytes()).hexdigest()
    assert (dry.ad("B1") / "VERDICT_B1.md").read_text(encoding="utf-8").count("PAUSED (PI 2026-09-19") == 1
    # resumable: a second start with READ_DONE present exits quietly; with it removed, the artifacts are skipped, nothing is re-dumped
    dry.start("B1")
    assert dry.proc.wait(timeout=60) == 0 and "ALREADYDONE" in dry.stdout()
    (dry.ad("B1") / "READ_DONE").unlink()
    n_before = dry.log("B1").count("DUMP-START")
    dry.start("B1")
    assert dry.proc.wait(timeout=240) == 0
    log = dry.log("B1")
    assert log.count("DUMP-START") == n_before and log.count("EVAL-SKIP") >= 3 and "LOCK-WAIT" not in log.split("EVAL-SKIP")[-1]
    assert (dry.ad("B1") / "READ_DONE").exists()


def test_dry_run_reads_b4_against_the_b2_winner_dumping_the_non_floor_reference_once(dry):
    b2 = dry.run_dir("B2", {"miss": [7, 8, 9]})
    b4 = dry.run_dir("B4", {"miss": []})
    dry.chain("START commit bbb tree /t arms [B2 B4]", "B2-LAUNCH s; run dir %s" % b2.as_posix(), "B2-DONE rc 0; ok",
              "B4-LAUNCH s; run dir %s" % b4.as_posix(), "B4-DONE rc 0; ok")
    dry.start("B4")                                                                                      # the design's reference for B4 is B2
    assert dry.wait(lambda: "DONE" in dry.events("B4"), 420), (dry.events("B4"), dry.stdout()[-2500:])
    log = dry.log("B4")
    assert "ref=B2" in log and "DUMP-START B4" in log and "DUMP-START B2" in log and "DUMP-START C0" not in log
    assert (dry.rd / "arm_B2" / "dump_B2" / "dump.json").is_file() and (dry.rd / "arm_B2" / "EVAL_B2.json").is_file()
    assert "DUMP-CHECK" in dry.events("B4") and dry.proc.wait(timeout=60) == 0
    v = _strict_json(dry.ad("B4") / "VERDICT_B4.json")
    assert v["status"] == "OK" and v["ref"] == "B2" and v["verdict"] == "KEEP", (v["verdict"], v["failing_clauses"], v["not_computable"])
    assert v["reads"][0]["key"] == "ap2m_all" and v["reads"][0]["delta"] == pytest.approx(0.15, abs=1e-9)


def test_dry_run_waits_on_the_floor_and_aborts_on_a_refused_floor(dry):
    (dry.rd / "FLOOR.json").rename(dry.rd / "FLOOR.json.bak")
    b1 = dry.run_dir("B1", {"miss": []})
    dry.chain("START commit bbb", "B1-LAUNCH s; run dir %s" % b1.as_posix(), "B1-DONE ok")
    dry.start("B1")
    assert dry.wait(lambda: dry.events("B1").count("WAIT") >= 1, 90), dry.events("B1")
    time.sleep(2)
    assert dry.proc.poll() is None and "LOCK-WAIT" not in dry.events("B1") and "floor read" in dry.log("B1")      # the floor is not read yet: nothing runs
    (dry.rd / "FLOOR.json").write_text(json.dumps({"schema": "r9e1-floor-v1", "status": "REFUSED", "problems": ["x"], "rows": {}}), encoding="utf-8")
    assert dry.wait(lambda: (dry.ad("B1") / "READ_ABORTED").exists(), 90) and dry.proc.wait(timeout=60) == 3
    assert "REFUSED" in (dry.ad("B1") / "READ_ABORTED").read_text(encoding="utf-8") and "LOCK-WAIT" not in dry.events("B1")


@pytest.mark.parametrize("bad", ["B1-FAILED rc 137: no done-marker at /x (read y)", "B1-ZEROGRAD-FAIL rc 0; /x/ZEROGRAD_FAIL.json", "B1-REFUSED supervisor refused"])
def test_dry_run_aborts_on_a_failed_arm_with_the_chain_line_and_never_touches_the_lock(dry, bad):
    dry.chain("START commit bbb", "B1-LAUNCH supervisor x; run dir /x", bad)
    dry.start("B1")
    assert dry.wait(lambda: (dry.ad("B1") / "READ_ABORTED").exists(), 90), dry.events("B1")
    assert dry.proc.wait(timeout=60) == 3
    txt = (dry.ad("B1") / "READ_ABORTED").read_text(encoding="utf-8")
    assert bad in txt and "CHAIN" in txt.splitlines()[0]
    assert "LOCK-WAIT" not in dry.events("B1") and "ABORTED" in dry.events("B1") and not (dry.ad("B1") / "dump_B1").exists()


def test_dry_run_refuses_floor_arms_unknown_references_and_a_split_whose_sha_does_not_match(dry):
    r = subprocess.run([dry.bash, SH.as_posix(), "C0"], env=dry.env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    assert r.returncode == 4 and "FLOOR arm" in r.stderr and not (dry.rd / "arm_C0").exists()
    r = subprocess.run([dry.bash, SH.as_posix(), "../x"], env=dry.env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    assert r.returncode == 4
    # E1-M3N20 (PREREG_M3): M3 now HAS a committed reference (M1; M3N20 -> M1N20), so it no longer aborts here -- it waits for its chain like any lever.
    for arm in ("M1-FROZEN",):                                                                           # the design names no reference: abort at once, before any wait
        r = subprocess.run([dry.bash, SH.as_posix(), arm], env=dry.env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        assert r.returncode == 3 and "NO-REFERENCE" in (dry.ad(arm) / "READ_ABORTED").read_text(encoding="utf-8"), (arm, r.stdout, r.stderr)
    r = subprocess.run([dry.bash, SH.as_posix(), "B1", "M1"], env=dry.env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    assert r.returncode == 3 and "cannot be read against" in (dry.ad("B1") / "READ_ABORTED").read_text(encoding="utf-8")        # B1 against M1: refused by the state check
    (dry.tmp / "split.sha256").write_text("0" * 64 + "  split.json\n", encoding="utf-8")
    b1 = dry.run_dir("B1", {"miss": []})
    (dry.ad("B1") / "READ_ABORTED").unlink()
    dry.chain("START commit bbb", "B1-LAUNCH s; run dir %s" % b1.as_posix(), "B1-DONE ok")
    dry.start("B1")
    assert dry.wait(lambda: (dry.ad("B1") / "READ_ABORTED").exists(), 90) and dry.proc.wait(timeout=60) == 3
    assert "SPLIT-SHA" in (dry.ad("B1") / "READ_ABORTED").read_text(encoding="utf-8") and "LOCK-WAIT" not in dry.events("B1")


def test_dry_run_waits_on_a_held_gpu_lock_and_runs_nothing_without_it(dry):
    b1 = dry.run_dir("B1", {"miss": []})
    dry.chain("START commit bbb", "B1-LAUNCH s; run dir %s" % b1.as_posix(), "B1-DONE ok")
    flock = subprocess.run([dry.bash, "-c", "command -v flock"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60).returncode == 0
    if flock:
        import fcntl
        fh = open(dry.lockf, "w", encoding="utf-8")
        fcntl.flock(fh, fcntl.LOCK_EX)
        release = lambda: (fcntl.flock(fh, fcntl.LOCK_UN), fh.close())                                   # noqa: E731
    else:
        Path(str(dry.lockf) + ".d").mkdir()
        release = lambda: Path(str(dry.lockf) + ".d").rmdir()                                             # noqa: E731
    try:
        dry.start("B1", R9E1_LOCK_MAX_S="3")
        assert dry.wait(lambda: (dry.ad("B1") / "READ_ABORTED").exists(), 90), dry.events("B1")
        assert dry.proc.wait(timeout=60) == 3
        ev = dry.events("B1")
        assert "LOCK-WAIT" in ev and "LOCK-TAKEN" not in ev and "DUMP-START" not in ev and not (dry.ad("B1") / "dump_B1").exists()
        assert "LOCK-TIMEOUT" in (dry.ad("B1") / "READ_ABORTED").read_text(encoding="utf-8")
    finally:
        release()


def test_dry_run_aborts_when_the_arm_was_dumped_on_other_windows(dry):
    """A pre-existing arm dump on a different window set is caught by `dumpcheck` before any eval (order_digest)."""
    b1 = dry.run_dir("B1", {"miss": []})
    dry.chain("START commit bbb", "B1-LAUNCH s; run dir %s" % b1.as_posix(), "B1-DONE ok")
    d = dry.ad("B1") / "dump_B1"
    d.mkdir(parents=True)
    rec = json.loads((dry.rd / "dump_C0" / "dump.json").read_text(encoding="utf-8"))
    (d / "dump.json").write_text(json.dumps(dict(rec, order_digest="0" * 64)), encoding="utf-8")
    dry.start("B1")
    assert dry.wait(lambda: (dry.ad("B1") / "READ_ABORTED").exists(), 90) and dry.proc.wait(timeout=60) == 3
    txt = (dry.ad("B1") / "READ_ABORTED").read_text(encoding="utf-8")
    assert "DUMP-MISMATCH" in txt and "order_digest" in txt and "EVAL-DONE" not in dry.events("B1")


# ------------------------------------------------------------------------------------------------------------------ MAPWIRE: R9E1_LIFT_VALID
@pytest.fixture(scope="module")
def floor_template_lv(tmp_path_factory):
    """The same floor read, but on ONE shared lift-valid mask (``tpl/mask.npz``): FLOOR.json carries the nf_swedge_prec_lv_* rows and the mask sha256."""
    return _build_floor_template(tmp_path_factory.mktemp("armread_floor_lv") / "tpl", with_mask=True)


@pytest.fixture
def dry_lv(tmp_path, floor_template_lv):
    b = _bash()
    if not b:
        pytest.skip("no usable bash on this box")
    d = Dry(tmp_path, b, floor_template_lv)
    d.mask = floor_template_lv / "mask.npz"
    yield d
    d.stop()


def _recorder(dry):
    """A bash shim for R9E1_PY: it appends every argv to a log, then runs the real interpreter -- the shell's calls are read, not guessed."""
    w = dry.tmp / "pyrec.sh"
    w.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$R9E1_ARGV_LOG"\nexec "$R9E1_REAL_PY" "$@"\n', encoding="utf-8", newline="\n")
    log = dry.tmp / "argv.log"
    return dict(R9E1_PY=w.as_posix(), R9E1_REAL_PY=sys.executable, R9E1_ARGV_LOG=log.as_posix()), log


def _calls(log):
    toks = [ln.split() for ln in Path(log).read_text(encoding="utf-8").splitlines() if ln.strip()]
    ev = [t for t in toks if any(x.endswith("r9_e1_eval.py") for x in t)]
    vd = [t for t in toks if any(x.endswith("r9_e1_verdict.py") for x in t) and "--floor" in t]
    return ev, vd


def _flags(toks):
    return [t for t in toks if t.startswith("--")]


def _read_b4(dry, **env):
    b2 = dry.run_dir("B2", {"miss": [7, 8, 9]})
    b4 = dry.run_dir("B4", {"miss": []})
    dry.chain("START commit bbb tree /t arms [B2 B4]", "B2-LAUNCH s; run dir %s" % b2.as_posix(), "B2-DONE rc 0; ok",
              "B4-LAUNCH s; run dir %s" % b4.as_posix(), "B4-DONE rc 0; ok")
    dry.start("B4", **env)
    assert dry.wait(lambda: "DONE" in dry.events("B4") or (dry.ad("B4") / "READ_ABORTED").exists(), 420), (dry.events("B4"), dry.stdout()[-2500:])
    assert dry.proc.wait(timeout=60) == 0, (dry.log("B4"), dry.stdout()[-2500:])


EVAL_FLAGS = ["--packs", "--rast-dir", "--windows", "--split", "--gate", "--boot", "--seed", "--out"]
PAIRED_FLAGS = ["--packs", "--packs-b", "--windows", "--split", "--gate", "--boot", "--seed", "--out"]
VERDICT_FLAGS = ["--floor", "--floor-eval-c0", "--paired", "--eval-arm", "--eval-ref", "--arm", "--ref", "--rast-arm", "--rast-ref", "--boot", "--seed", "--out-dir"]


def test_lift_valid_unset_passes_no_lift_valid_anywhere_and_the_argv_is_the_earlier_one(dry):
    env, log = _recorder(dry)
    _read_b4(dry, **env)
    ev, vd = _calls(log)
    # B4 vs B2: the arm single, the (non-floor) reference single, the paired eval -- three eval calls -- and ONE verdict call
    assert [_flags(t) for t in ev] == [EVAL_FLAGS, EVAL_FLAGS, PAIRED_FLAGS] and [_flags(t) for t in vd] == [VERDICT_FLAGS], (ev, vd)
    assert not [t for t in ev + vd if "--lift-valid" in t] and "LIFT-VALID" not in dry.events("B4")
    v = _strict_json(dry.ad("B4") / "VERDICT_B4.json")
    assert v["status"] == "OK" and v["verdict"] == "KEEP" and v["lift_valid"] == {"verdict": None, "floor": None, "same_mask": True}


def test_lift_valid_set_is_passed_to_both_evals_the_paired_eval_and_the_verdict_and_the_verdict_is_read_on_it(dry_lv):
    env, log = _recorder(dry_lv)
    _read_b4(dry_lv, R9E1_LIFT_VALID=dry_lv.mask.as_posix(), **env)
    ev, vd = _calls(log)
    with_lv = lambda fl, after: fl[:fl.index(after) + 1] + ["--lift-valid"] + fl[fl.index(after) + 1:]            # noqa: E731
    assert [_flags(t) for t in ev] == [with_lv(EVAL_FLAGS, "--seed"), with_lv(EVAL_FLAGS, "--seed"), with_lv(PAIRED_FLAGS, "--seed")], ev    # before --out
    assert [_flags(t) for t in vd] == [with_lv(VERDICT_FLAGS, "--rast-ref")], vd
    for t in ev + vd:
        assert t[t.index("--lift-valid") + 1].endswith("/mask.npz") and t.count("--lift-valid") == 1
    assert "LIFT-VALID" in dry_lv.events("B4") and dry_lv.events("B4").index("LIFT-VALID") < dry_lv.events("B4").index("LOCK-WAIT")       # checked BEFORE the GPU lock
    v = _strict_json(dry_lv.ad("B4") / "VERDICT_B4.json")
    assert v["status"] == "OK" and v["verdict"] == "KEEP", (v["verdict"], v["failing_clauses"], v["not_computable"], v.get("problems"))
    fl = _strict_json(dry_lv.rd / "FLOOR.json")
    assert v["lift_valid"]["same_mask"] is True and v["lift_valid"]["verdict"]["sha256"] == fl["map_inputs"]["lift_valid"]["sha256"]
    assert "nf_swedge_prec_lv_5_20" in fl["rows"] and v["clause3"]["missing_from_floor"] == []                       # the floor HAS the row: nothing is missing
    assert "EQUAL to the floor's" in (dry_lv.ad("B4") / "VERDICT_B4.md").read_text(encoding="utf-8")


def test_lift_valid_set_against_a_floor_read_without_a_mask_is_refused_by_the_verdict(dry, tmp_path):
    """The shell passes the mask; the verdict REFUSES (the floor was read without one) and the shell aborts from the artifact, not from $?."""
    mask = tmp_path / "mask_for_a_maskless_floor.npz"
    _write_mask(mask)
    b1 = dry.run_dir("B1", {"miss": []})
    dry.chain("START commit bbb", "B1-LAUNCH s; run dir %s" % b1.as_posix(), "B1-DONE ok")
    dry.start("B1", R9E1_LIFT_VALID=mask.as_posix())
    assert dry.wait(lambda: (dry.ad("B1") / "READ_ABORTED").exists() or "DONE" in dry.events("B1"), 420), (dry.events("B1"), dry.stdout()[-2500:])
    assert dry.proc.wait(timeout=60) == 3 and "VERDICT-NOT-OK" in (dry.ad("B1") / "READ_ABORTED").read_text(encoding="utf-8")
    v = _strict_json(dry.ad("B1") / "VERDICT_B1.json")
    assert v["status"] == "REFUSED" and "lift-valid mask" in " ".join(v["problems"]) and not (dry.ad("B1") / "READ_DONE").exists()


def test_lift_valid_set_to_a_missing_path_aborts_before_the_lock_is_taken(dry, tmp_path):
    b1 = dry.run_dir("B1", {"miss": []})
    dry.chain("START commit bbb", "B1-LAUNCH s; run dir %s" % b1.as_posix(), "B1-DONE ok")
    dry.start("B1", R9E1_LIFT_VALID=(tmp_path / "no_such_mask.npz").as_posix())
    assert dry.wait(lambda: (dry.ad("B1") / "READ_ABORTED").exists(), 90) and dry.proc.wait(timeout=60) == 3
    assert "NO-LIFT-VALID" in (dry.ad("B1") / "READ_ABORTED").read_text(encoding="utf-8")
    assert "LOCK-WAIT" not in dry.events("B1") and "DUMP-START" not in dry.events("B1") and not (dry.ad("B1") / "dump_B1").exists()
