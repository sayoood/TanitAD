"""r9_e1_floor.py + the E1-READ ops kit (r9e1_read.sh / r9e1_read_state.py / r9e1_read_stub_dump.py): known-answer tests.

KNOWN ANSWERS (literal, built from synthetic dumps through the REAL r9_e1_eval.py -> r9_e1_floor.py path):
  * C0 vs an exact COPY of C0:  delta exactly 0.0, CI [0.0, 0.0], separated False, floor 0.0 (flagged degenerate), every metric,
    boxes and map;
  * C1 with one more detected GT per window (20 GT, AP@2 m 0.95 -> 1.0): delta = +0.05 within float tolerance, separated True,
    floor 0.05, keep threshold 0.10 -- and a lever read through ``taniteval.ci.e1_floor_read`` keeps at 0.11, not at 0.09;
  * the refusals: different window sets, swapped arm order, a different P=R gate, different raster name sets.
The ops half drives the real shell script in DRY mode (GPU step = the stub, GPU lock emulated with mkdir when the box has no
flock): wait / abort / lock / step order.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_STACK = Path(__file__).resolve().parents[1]
_OPS = _STACK / "ops" / "r9e1_read"
for _p in (_STACK / "scripts", _OPS):
    sys.path.insert(0, str(_p))
import r9_e1_eval as E                                   # noqa: E402
import r9_e1_floor as F                                  # noqa: E402
import r9e1_read_state as ST                             # noqa: E402
import r9e1_read_stub_dump as STUB                       # noqa: E402

CLIPS = ["%012x" % (0xa00 + i) for i in range(6)]
BOOT = "50"


def _strict_json(path):
    def bad(tok):
        raise ValueError("non-finite JSON token %s" % tok)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=bad)


# ------------------------------------------------------------------------------------------------------ synthetic arms
def _split(tmp):
    p = Path(tmp) / "split.json"
    p.write_text(json.dumps({"DEV": {"sha12": CLIPS}, "HOLD": {"sha12": ["%012x" % 0xb00]}}), encoding="utf-8")
    return p


def _dump(tmp, name, spec, split):
    tmp = Path(tmp)
    ck = tmp / (name + ".ckpt.json")
    ck.write_text(json.dumps(dict({"step": 4000}, **spec)), encoding="utf-8")
    (tmp / "config.json").write_text("{}", encoding="utf-8")
    out = tmp / ("dump_" + name)
    assert STUB.main(["--ckpt", str(ck), "--config", str(tmp / "config.json"), "--windows", str(split), "--split", "dev",
                      "--s2-windows", str(tmp / "s2.json"), "--out", str(out), "--require-ema"]) == 0
    return out


def _evals(tmp, d0, d1, split, *, gate1="0.5", swap=False, name=""):
    tmp = Path(tmp)
    com = lambda g: ["--windows", str(split), "--split", "dev", "--gate", g, "--boot", BOOT, "--seed", "0"]      # noqa: E731
    e0, e1, ep = tmp / ("EVAL_C0%s.json" % name), tmp / ("EVAL_C1%s.json" % name), tmp / ("EVAL_C1_vs_C0%s.json" % name)
    p0, p1 = str(d0 / "perc_det.pkl"), str(d1 / "perc_det.pkl")
    assert E.main(["--packs", p0, "--rast-dir", str(d0 / "rast"), *com("0.5"), "--out", str(e0)]) == 0
    assert E.main(["--packs", p1, "--rast-dir", str(d1 / "rast"), *com(gate1), "--out", str(e1)]) == 0
    a, b = (p0, p1) if swap else (p1, p0)
    assert E.main(["--packs", a, "--packs-b", b, *com("0.5"), "--out", str(ep)]) == 0
    return e0, e1, ep


def _read(tmp, spec0, spec1, *, name="", **kw):
    tmp = Path(tmp)
    split = _split(tmp)
    d0, d1 = _dump(tmp, "C0" + name, spec0, split), _dump(tmp, "C1" + name, spec1, split)
    if kw.pop("drop_window", False):
        import pickle
        recs = pickle.loads((d1 / "perc_det.pkl").read_bytes())
        (d1 / "perc_det.pkl").write_bytes(pickle.dumps(recs[:-1]))
    if kw.pop("drop_raster", False):
        sorted((d1 / "rast").glob("*.npz"))[-1].unlink()
    e0, e1, ep = _evals(tmp, d0, d1, split, name=name, **kw)
    out = tmp / ("floor" + name)
    rc = F.main(["--eval-c0", str(e0), "--eval-c1", str(e1), "--paired", str(ep), "--rast-c0", str(d0 / "rast"),
                 "--rast-c1", str(d1 / "rast"), "--boot", BOOT, "--out-dir", str(out)])
    return rc, _strict_json(out / "FLOOR.json"), (out / "FLOOR.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def selfpair(tmp_path_factory):
    return _read(tmp_path_factory.mktemp("selfpair"), {"miss": [7], "fp_rows": 5}, {"miss": [7], "fp_rows": 5})


@pytest.fixture(scope="module")
def shifted(tmp_path_factory):
    return _read(tmp_path_factory.mktemp("shifted"), {"miss": [7]}, {"miss": []})


# ------------------------------------------------------------------------------------------------------ KNOWN ANSWER 1
def test_c0_vs_a_copy_of_c0_reads_exactly_zero(selfpair):
    rc, fl, _md = selfpair
    assert rc == 0 and fl["status"] == "OK" and fl["tier"] == "T0 (label level)"
    rows = fl["rows"]
    assert {r["family"] for r in rows.values()} == {"BOXES", "MAP"}
    assert "ap2m_all" in rows and "ap1m_0_20" in rows and "nf_prec_5_10" in rows
    defined = [r for r in rows.values() if not r["undefined"]]
    assert len(rows) == 30 and len(defined) == 30                      # 22 box rows (n_pos is a count; B3 added boxes/object + AP no-NMS by range) + 8 map rows
    for r in defined:
        assert r["delta"] == 0.0 and r["ci_lo"] == 0.0 and r["ci_hi"] == 0.0, r["metric"]
        assert r["separated"] is False and r["floor"] == 0.0 and r["keep_threshold"] == 0.0 and r["floor_degenerate"] is True, r["metric"]
    assert rows["ap2m_all"]["c0"] == rows["ap2m_all"]["c1"] == pytest.approx(0.95)
    assert fl["summary"]["n_separated_c1_vs_c0"] == 0 and fl["summary"]["n_floor_degenerate"] == fl["summary"]["n_defined"] == 29   # 30 rows minus the n_pos count row


# ------------------------------------------------------------------------------------------------------ KNOWN ANSWER 2
def test_deliberate_plus_005_shift_reads_separated_and_exact(shifted):
    rc, fl, _md = shifted
    assert rc == 0
    r = fl["rows"]
    for m in ("ap2m_all", "ap1m_all", "rec_gate_noNMS"):
        assert r[m]["c0"] == pytest.approx(0.95, abs=1e-9) and r[m]["c1"] == pytest.approx(1.0, abs=1e-9), m
        assert r[m]["delta"] == pytest.approx(0.05, abs=1e-9) and r[m]["separated"] is True, m
        assert r[m]["floor"] == pytest.approx(0.05, abs=1e-9) and r[m]["keep_threshold"] == pytest.approx(0.10, abs=1e-9), m
        assert r[m]["ci_lo"] > 0 and r[m]["floor_degenerate"] is False, m
    # the missed GT (index 7, x = 24 m) sits in the 20-40 m band, which holds GT 6..12 = 7 boxes: 1/7 of that band's AP, 0 elsewhere
    assert r["ap2m_20_40"]["delta"] == pytest.approx(1.0 / 7.0, abs=1e-9) and r["ap2m_20_40"]["separated"] is True
    for m in ("ap2m_0_20", "ap2m_40_60", "dx_0_40_mean", "tp_class_acc", "boxes_per_object_noNMS", "nf_prec_5_10"):
        assert r[m]["delta"] == 0.0 and r[m]["separated"] is False and r[m]["floor_degenerate"] is True, m
    assert r["ap2m_all"]["n_windows"] == 12 and r["ap2m_all"]["n_episodes"] == 6
    assert "episode" in r["ap2m_all"]["estimator"] or "paired_episode_cluster_bootstrap_stat" in r["ap2m_all"]["estimator"]


def test_lever_read_is_taniteval_ci_e1_floor_read(shifted):
    _rc, fl, _md = shifted
    t = fl["rows"]["ap2m_all"]["keep_threshold"]
    assert F.lever_read(fl, "ap2m_all", 0.11, True)["keep_on_this_metric"] is True
    assert F.lever_read(fl, "ap2m_all", 0.09, True)["keep_on_this_metric"] is False
    assert F.lever_read(fl, "ap2m_all", 0.11, False)["keep_on_this_metric"] is False            # not separated
    assert F.lever_read(fl, "ap2m_all", -0.11, True)["separated_bad_direction"] is True            # separated the wrong way
    assert F.lever_read(fl, "ap2m_all", t * (1 + 1e-9), True)["exceeds_2x_floor"] is True
    assert F.lever_read(fl, "ap2m_all", t * (1 - 1e-9), True)["exceeds_2x_floor"] is False
    # lower-is-better: a SMALLER dx is the good direction
    assert F.lever_read(fl, "dx_0_40_mean", -0.5, True)["separated_good_direction"] is True
    with pytest.raises(ValueError):
        F.lever_read(fl, "n_pos", 1.0, True)


def test_map_row_shift_has_the_right_sign(tmp_path):
    rc, fl, _md = _read(tmp_path, {}, {"fp_rows": 20})                 # C1 flags 20 false sidewalk rows in the 5-10 m band
    assert rc == 0
    r = fl["rows"]
    assert r["nf_prec_5_10"]["delta"] < 0 and r["nf_prec_5_10"]["separated"] is True
    assert r["nf_prec_5_10"]["floor"] == pytest.approx(abs(r["nf_prec_5_10"]["delta"]))
    assert r["nf_prec_0_5"]["delta"] == 0.0 and r["nf_prec_10_20"]["delta"] == 0.0
    assert r["nf_prec_5_10"]["n_unit"] == "raster files" and r["nf_prec_5_10"]["n_windows"] == 12 and r["nf_prec_5_10"]["n_episodes"] == 6


# ------------------------------------------------------------------------------------------------------ the four families
def test_markdown_states_every_family_and_stamps_every_table(selfpair):
    _rc, fl, md = selfpair
    nc = "not computable: the E1 rig builds no planner (perception-only, T0 label level), n = 0"
    for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL"):
        assert "| %s | %s |" % (fam, nc) in md
    assert re.search(r"\| STRATEGIC \| PAUSED \(PI 2026-09-19", md)
    tables = md.count("| metric | C0 | C1 |")
    assert tables >= 6 and md.count("Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=50 seed=0 | n = ") == tables
    assert "floor = abs(C1 - C0)" in md and "keep threshold = 2 x floor" in md
    fams = {x["family"]: x["status"] for x in fl["families"]}
    assert set(fams) == {"LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"} and fams["STRATEGIC"].startswith("PAUSED")


# ------------------------------------------------------------------------------------------------------ refusals
@pytest.mark.parametrize("kw,needle", [
    ({"drop_window": True}, "same windows"),
    ({"swap": True}, "first arm is the C1 dump"),
    ({"gate1": "0.4"}, "P=R gate"),
    ({"drop_raster": True}, "raster name sets differ"),
])
def test_floor_refuses_when_it_is_not_one_read_of_the_same_windows(tmp_path, kw, needle):
    rc, fl, md = _read(tmp_path, {}, {}, **kw)
    assert rc == 2 and fl["status"] == "REFUSED" and fl["rows"] == {}
    assert any(needle in p for p in fl["problems"]), fl["problems"]
    assert "REFUSED" in md and "| metric |" not in md                   # no table is quoted from a refused read


def test_floor_refuses_a_non_eval_file(tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text(json.dumps({"schema": "nope"}), encoding="utf-8")
    assert F.main(["--eval-c0", str(bad), "--eval-c1", str(bad), "--paired", str(bad), "--out-dir", str(tmp_path / "o")]) == 2


# ------------------------------------------------------------------------------------------------------ the state machine
def _lines(*ev):
    return "".join("2026-11-03T%02d:00:00Z %s\n" % (i % 24, e) for i, e in enumerate(ev))


def _run(root, arm, *, done=True, zerograd=True, fail=False, step=4000, ckpt=True, cfg=True):
    d = Path(root) / "runs" / ("r9e1-" + arm)
    d.mkdir(parents=True, exist_ok=True)
    s = {"done": done, "step": step, "arm": "flat"}
    if zerograd:
        s["zerograd_gate"] = {"ok": True}
    (d / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    if ckpt:
        (d / "model_ema.pt").write_bytes(b"x")
    if cfg:
        (d / "config.json").write_text("{}", encoding="utf-8")
    if fail:
        (d / "ZEROGRAD_FAIL.json").write_text("{}", encoding="utf-8")
    return d


OLD = ["START commit aaa tree /t", "STOP-EXPLORE-NOT-PASS  G-EVAL=FAIL (read ...); suite runner killed", "START commit bbb tree /t",
       "SUITE-STARTED runner pid 1", "EXPLORE-PASS six non-suite checks PASS"]


def test_state_ignores_a_superseded_stop_and_waits(tmp_path):
    s = ST.classify(_lines(*OLD[:2]) + _lines(*OLD[2:]), tmp_path)
    assert s["state"] == "WAIT" and s["stops"] == []                                  # the STOP precedes the last START
    s = ST.classify(_lines(*OLD, "STOP-NO-TOKEN neither arm passed the gate"), tmp_path)
    assert s["state"] == "WAIT" and s["stops"] and "chain stop" in s["reason"]       # a pre-launch stop: wait for a restart
    assert ST.classify("", tmp_path)["state"] == "WAIT"


def test_state_tracks_the_chain_to_ready(tmp_path):
    r0, r1 = tmp_path / "runs" / "r9e1-C0", tmp_path / "runs" / "r9e1-C1"
    s = ST.classify(_lines(*OLD, "C0-LAUNCH supervisor x; run dir %s" % r0), tmp_path)
    assert s["state"] == "WAIT" and s["c0_run"] == str(r0) and "C0 training" in s["reason"]
    _run(tmp_path, "C0")
    s = ST.classify(_lines(*OLD, "C0-LAUNCH s; run dir %s" % r0, "C0-DONE rc 0; ok", "C1-LAUNCH s; run dir %s" % r1), tmp_path)
    assert s["state"] == "WAIT" and "C0 done" in s["reason"] and "C1 training" in s["reason"]
    _run(tmp_path, "C1")
    full = _lines(*OLD, "C0-LAUNCH s; run dir %s" % r0, "C0-DONE rc 0; ok", "C1-LAUNCH s; run dir %s" % r1, "C1-DONE rc 0; ok")
    s = ST.classify(full, tmp_path)
    assert s["state"] == "READY" and s["c1_run"] == str(r1) and "C1-DONE" in s["line"]
    assert ST.classify(full + _lines("ALLDONE chain finished"), tmp_path)["state"] == "READY"       # ALLDONE after both DONE is fine


@pytest.mark.parametrize("bad", ["C0-FAILED rc 1: no done-marker", "C0-ZEROGRAD-FAIL rc 0; x", "C0-REFUSED supervisor refused",
                                 "C0-NOT-LAUNCHED no PASS token", "C1-FAILED rc 137: killed"])
def test_state_aborts_on_a_failed_arm_with_the_line(tmp_path, bad):
    s = ST.classify(_lines(*OLD, "C0-LAUNCH s; run dir /r/C0", bad), tmp_path)
    assert s["state"] == "ABORT" and bad in s["line"]


def test_state_aborts_on_alldone_without_c1_and_on_marker_disagreement(tmp_path):
    r0, r1 = tmp_path / "runs" / "r9e1-C0", tmp_path / "runs" / "r9e1-C1"
    s = ST.classify(_lines(*OLD, "C0-LAUNCH s; run dir %s" % r0, "C0-DONE ok", "ALLDONE chain finished"), tmp_path)
    assert s["state"] == "ABORT" and "ALLDONE without C1-DONE" in s["reason"] and s["line"].endswith("ALLDONE chain finished")
    both = _lines(*OLD, "C0-LAUNCH s; run dir %s" % r0, "C0-DONE ok", "C1-LAUNCH s; run dir %s" % r1, "C1-DONE ok")
    _run(tmp_path, "C0")
    for kw, needle in (({"done": False}, '"done": true'), ({"zerograd": False}, "zerograd_gate"), ({"fail": True}, "ZEROGRAD_FAIL"),
                       ({"ckpt": False}, "model_ema.pt"), ({"cfg": False}, "config.json"), ({"step": 3999}, "different steps")):
        shutil.rmtree(r1, ignore_errors=True)
        _run(tmp_path, "C1", **kw)
        s = ST.classify(both, tmp_path)
        assert s["state"] == "ABORT" and needle in s["reason"], (kw, s["reason"])


def test_state_checks_the_launch_argv_steps(tmp_path):
    r0, r1 = _run(tmp_path, "C0"), _run(tmp_path, "C1")
    ad = tmp_path / "launch_tree" / "stack" / "ops" / "runs.d"
    ad.mkdir(parents=True)
    for a in ("C0", "C1"):
        (ad / ("r9e1_%s.argv.json" % a)).write_text(json.dumps({"steps": 4000}), encoding="utf-8")
    both = _lines("START x", "C0-LAUNCH s; run dir %s" % r0, "C0-DONE ok", "C1-LAUNCH s; run dir %s" % r1, "C1-DONE ok")
    assert ST.classify(both, tmp_path)["state"] == "READY"
    (ad / "r9e1_C1.argv.json").write_text(json.dumps({"steps": 5000}), encoding="utf-8")
    s = ST.classify(both, tmp_path)
    assert s["state"] == "ABORT" and "launch argv's steps 5000" in s["reason"]


def test_state_cli_writes_a_sourceable_file(tmp_path):
    st = tmp_path / "CHAIN_STATUS.txt"
    st.write_text(_lines(*OLD, "C0-FAILED rc 1: it's \"bad\" $(x)"), encoding="utf-8")
    assert ST.main(["--status", str(st), "--root", str(tmp_path), "--out", str(tmp_path / "s.env")]) == 0
    bash = _bash(tmp_path)
    if bash:
        out = subprocess.run([bash, "-c", '. "$1"; printf "%s|%s" "$S_STATE" "$S_LINE"', "x", (tmp_path / "s.env").as_posix()],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60).stdout
        assert out.startswith("ABORT|") and "it's \"bad\" $(x)" in out                     # quoting survived: nothing was executed
    assert ST.main(["--status", str(tmp_path / "absent.txt"), "--root", str(tmp_path), "--out", str(tmp_path / "s2.env")]) == 0
    assert "no status file yet" in (tmp_path / "s2.env").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------------------ the shell, DRY
SH = _OPS / "r9e1_read.sh"


def _bash(tmp):
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


def _flock_in(bash):
    return subprocess.run([bash, "-c", "command -v flock"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60).returncode == 0


def test_shell_script_flags_exist_in_the_scripts_it_calls():
    sh = SH.read_text(encoding="utf-8")
    assert "\r" not in sh and sh.startswith("#!/usr/bin/env bash\n")

    def flags(p):
        return set(re.findall(r'add_argument\(\s*"(--[\w-]+)"', Path(p).read_text(encoding="utf-8")))
    want = {
        "r9_e1_dump.py": ["--ckpt", "--config", "--windows", "--split", "--s2-windows", "--require-ema", "--out"],
        "r9_e1_eval.py": ["--packs", "--packs-b", "--rast-dir", "--windows", "--split", "--boot", "--seed", "--out", "--gates-json", "--gate"],
        "r9_e1_floor.py": ["--eval-c0", "--eval-c1", "--paired", "--rast-c0", "--rast-c1", "--boot", "--seed", "--out-dir"],
        "r9e1_read_state.py": ["--status", "--root", "--out"],
    }
    srcs = {"r9_e1_dump.py": _STACK / "scripts" / "r9_e1_dump.py", "r9_e1_eval.py": _STACK / "scripts" / "r9_e1_eval.py",
            "r9_e1_floor.py": _STACK / "scripts" / "r9_e1_floor.py", "r9e1_read_state.py": _OPS / "r9e1_read_state.py"}
    for name, fl in want.items():
        have = flags(srcs[name])
        for f in fl:
            assert f in have, "%s has no %s" % (name, f)
            assert f in sh, "r9e1_read.sh no longer passes %s (to %s)" % (f, name)
    assert flags(_OPS / "r9e1_read_stub_dump.py") >= set(want["r9_e1_dump.py"])                      # the stub accepts what the shell passes


class Dry:
    def __init__(self, tmp, bash):
        self.tmp, self.bash = Path(tmp), bash
        self.root = self.tmp / "r9e1"
        self.rd = self.root / "read"
        (self.root / "gate3").mkdir(parents=True)
        self.status = self.root / "gate3" / "CHAIN_STATUS.txt"
        split = _split(self.tmp)
        (self.tmp / "split.sha256").write_text("%s  split.json\n" % __import__("hashlib").sha256(split.read_bytes()).hexdigest(), encoding="utf-8")
        (self.tmp / "s2.json").write_text("{}", encoding="utf-8")
        self.lockf = self.tmp / "gpu.lock"
        self.env = dict(os.environ, R9E1_DRY="1", R9E1_ROOT=self.root.as_posix(), R9E1_READ_DIR=self.rd.as_posix(),
                        R9E1_TREE=(self.root / "launch_tree").as_posix(), R9E1_PY=sys.executable, R9E1_LOCKF=self.lockf.as_posix(),
                        R9E1_POLL_S="1", R9E1_BOOT=BOOT, R9E1_DUMP_SCRIPT=(_OPS / "r9e1_read_stub_dump.py").as_posix(),
                        R9E1_DUMP_CWD=self.tmp.as_posix(), R9E1_EVAL_SCRIPT=(_STACK / "scripts" / "r9_e1_eval.py").as_posix(),
                        R9E1_PYTHONPATH=os.pathsep.join(str(p) for p in (_STACK, _STACK / "scripts", _STACK.parent / "taniteval")),
                        R9E1_GATE_ARGS="--gate 0.5", R9E1_SPLIT=split.as_posix(), R9E1_SPLIT_SHA_FILE=(self.tmp / "split.sha256").as_posix(),
                        R9E1_S2=(self.tmp / "s2.json").as_posix(), CUDA_VISIBLE_DEVICES="-1", R9E1_MAX_WAIT_S="900")     # a stray script ends itself
        self.proc = None

    def chain(self, *ev):
        with open(self.status, "a", encoding="utf-8", newline="\n") as f:
            f.write("".join("2026-11-03T05:%02d:00Z %s\n" % (i % 60, e) for i, e in enumerate(ev)))

    def run_dir(self, arm, spec, **kw):
        d = _run(self.root, arm, **kw)
        (d / "model_ema.pt").write_text(json.dumps(dict({"step": 4000}, **spec)), encoding="utf-8")
        return d

    def start(self, **env):
        self.out = open(self.tmp / "sh.out", "w", encoding="utf-8")
        self.proc = subprocess.Popen([self.bash, SH.as_posix()], env=dict(self.env, **env), stdout=self.out, stderr=subprocess.STDOUT)

    def events(self):
        f = self.rd / "READ_STATUS.txt"
        return [ln.split(" ", 2)[1] for ln in f.read_text(encoding="utf-8").splitlines()] if f.exists() else []

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

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=30)
        self.out.close()


@pytest.fixture
def dry(tmp_path):
    b = _bash(tmp_path)
    if not b:
        pytest.skip("no usable bash on this box")
    d = Dry(tmp_path, b)
    yield d
    d.stop()


def test_dry_run_waits_then_reads_in_the_pinned_order(dry):
    c0 = dry.run_dir("C0", {"miss": [7]})
    c1 = dry.run_dir("C1", {"miss": []})
    dry.chain(*OLD)                                                   # an old STOP, then the START that trained
    dry.start()
    assert dry.wait(lambda: dry.events().count("WAIT") >= 1, 90), (dry.events(), (dry.tmp / "sh.out").read_text(encoding="utf-8"))
    dry.chain("C0-LAUNCH supervisor x; run dir %s" % c0.as_posix())
    assert dry.wait(lambda: dry.events().count("WAIT") >= 2, 90), dry.events()          # a changed picture is logged once
    dry.chain("C0-DONE rc 0; ok", "C1-LAUNCH supervisor x; run dir %s" % c1.as_posix())
    assert dry.wait(lambda: dry.events().count("WAIT") >= 3, 90), dry.events()
    time.sleep(2)
    assert "LOCK-WAIT" not in dry.events() and dry.proc.poll() is None               # WAIT never reaches the lock
    assert not (dry.rd / "dump_C0").exists()                                         # and runs nothing
    dry.chain("C1-DONE rc 0; ok", "ALLDONE chain finished")
    assert dry.wait(lambda: "LOCK-RELEASED" in dry.events(), 420), (dry.events(), (dry.tmp / "sh.out").read_text(encoding="utf-8")[-2000:])
    ev = dry.events()
    order = ["START", "WAIT", "READY", "INPUTS-OK", "LOCK-WAIT", "LOCK-TAKEN", "GPU-CHECK", "DUMP-START", "DUMP-DONE", "DUMP-START",
             "DUMP-DONE", "DUMP-CHECK", "EVAL-DONE", "EVAL-DONE", "EVAL-DONE", "FLOOR-DONE", "DONE", "LOCK-RELEASED"]
    it = iter(ev)
    assert all(any(e == o for e in it) for o in order), ev                           # `order` is a subsequence of the events
    log = (dry.rd / "READ_STATUS.txt").read_text(encoding="utf-8")
    assert log.index("DUMP-START C0") < log.index("DUMP-START C1") < log.index("EVAL-DONE C0") < log.index("EVAL-DONE C1_vs_C0")
    assert re.search(r"ZZR9E1READ-DONEZZ", (dry.tmp / "sh.out").read_text(encoding="utf-8"))
    assert dry.proc.wait(timeout=60) == 0 and (dry.rd / "READ_DONE").exists() and not (dry.rd / "READ_ABORTED").exists()
    assert not Path(str(dry.lockf) + ".d").exists()                                  # the (emulated) lock is released
    fl = _strict_json(dry.rd / "FLOOR.json")
    assert fl["status"] == "OK" and fl["rows"]["ap2m_all"]["delta"] == pytest.approx(0.05, abs=1e-9)    # end to end: the +0.05 arm
    assert (dry.rd / "FLOOR.md").read_text(encoding="utf-8").count("PAUSED (PI 2026-09-19") == 1


@pytest.mark.parametrize("bad", ["C0-FAILED rc 137: no done-marker at /x (read y)", "C0-ZEROGRAD-FAIL rc 0; /x/ZEROGRAD_FAIL.json"])
def test_dry_run_aborts_on_a_failed_arm_and_never_touches_the_lock(dry, bad):
    dry.chain(*OLD, "C0-LAUNCH supervisor x; run dir /x", bad)
    dry.start()
    assert dry.wait(lambda: (dry.rd / "READ_ABORTED").exists(), 90), dry.events()
    assert dry.proc.wait(timeout=60) == 3
    txt = (dry.rd / "READ_ABORTED").read_text(encoding="utf-8")
    assert bad in txt and "CHAIN" in txt.splitlines()[0]                              # the chain line, verbatim
    assert "LOCK-WAIT" not in dry.events() and "ABORTED" in dry.events() and not (dry.rd / "dump_C0").exists()


def test_dry_run_waits_on_a_held_gpu_lock_and_runs_nothing_without_it(dry):
    c0, c1 = dry.run_dir("C0", {}), dry.run_dir("C1", {})
    dry.chain(*OLD, "C0-LAUNCH s; run dir %s" % c0.as_posix(), "C0-DONE ok", "C1-LAUNCH s; run dir %s" % c1.as_posix(), "C1-DONE ok")
    if _flock_in(dry.bash):
        import fcntl
        fh = open(dry.lockf, "w", encoding="utf-8")
        fcntl.flock(fh, fcntl.LOCK_EX)
        release = lambda: (fcntl.flock(fh, fcntl.LOCK_UN), fh.close())                   # noqa: E731
    else:
        Path(str(dry.lockf) + ".d").mkdir()
        release = lambda: Path(str(dry.lockf) + ".d").rmdir()                             # noqa: E731
    try:
        dry.start(R9E1_LOCK_MAX_S="3")
        assert dry.wait(lambda: (dry.rd / "READ_ABORTED").exists(), 90), dry.events()
        assert dry.proc.wait(timeout=60) == 3
        ev = dry.events()
        assert "LOCK-WAIT" in ev and "LOCK-TAKEN" not in ev and "DUMP-START" not in ev and not (dry.rd / "dump_C0").exists()
        assert "LOCK-TIMEOUT" in (dry.rd / "READ_ABORTED").read_text(encoding="utf-8")
    finally:
        release()


def test_dry_run_refuses_a_split_whose_sha_does_not_match(dry):
    c0, c1 = dry.run_dir("C0", {}), dry.run_dir("C1", {})
    dry.chain(*OLD, "C0-LAUNCH s; run dir %s" % c0.as_posix(), "C0-DONE ok", "C1-LAUNCH s; run dir %s" % c1.as_posix(), "C1-DONE ok")
    (dry.tmp / "split.sha256").write_text("0" * 64 + "  split.json\n", encoding="utf-8")
    dry.start()
    assert dry.wait(lambda: (dry.rd / "READ_ABORTED").exists(), 90) and dry.proc.wait(timeout=60) == 3
    assert "SPLIT-SHA" in (dry.rd / "READ_ABORTED").read_text(encoding="utf-8") and "LOCK-WAIT" not in dry.events()


def test_dry_run_with_the_scipy_shim_reads_the_same_map_numbers_as_real_scipy(dry):
    """Thor's training venv has no scipy: the CPU steps run on shim_scipy/.  Force it here and require the FLOOR's map rows to equal the
    real-scipy counting done in-process on the very same rasters."""
    pytest.importorskip("scipy.ndimage")
    c0, c1 = dry.run_dir("C0", {"miss": [7], "fp_rows": 3}), dry.run_dir("C1", {"fp_rows": 9})
    dry.chain(*OLD, "C0-LAUNCH s; run dir %s" % c0.as_posix(), "C0-DONE ok", "C1-LAUNCH s; run dir %s" % c1.as_posix(), "C1-DONE ok")
    dry.start(R9E1_FORCE_SCIPY_SHIM="1", R9E1_PATHSEP=os.pathsep)
    assert dry.wait(lambda: "LOCK-RELEASED" in dry.events(), 420), (dry.events(), (dry.tmp / "sh.out").read_text(encoding="utf-8")[-2000:])
    assert dry.proc.wait(timeout=60) == 0 and "SCIPY-SHIM" in dry.events() and dry.events().index("SCIPY-SHIM") < dry.events().index("LOCK-WAIT")
    assert "[scipy-shim]" in (dry.rd / "logs" / "eval_C0.err").read_text(encoding="utf-8")                 # the shim, not scipy, ran
    assert "[scipy-shim]" not in (dry.rd / "logs" / "dump_C0.err").read_text(encoding="utf-8")            # the GPU step never sees it
    fl = _strict_json(dry.rd / "FLOOR.json")
    assert fl["status"] == "OK"
    for arm, key in (("C0", "c0"), ("C1", "c1")):
        names, acc, gtcls = E.nearfield_per_file(sorted(str(q) for q in (dry.rd / ("dump_" + arm) / "rast").glob("*.npz")))        # real scipy
        eps = [E.nearfield_episode_of(n) for n in names]
        units = [sum(acc[i] for i in range(len(names)) if eps[i] == e) for e in sorted(set(eps))]
        for m, v in E.nearfield_stats(units).items():
            assert fl["rows"][m][key] == pytest.approx(v, abs=1e-12), (arm, m)
    assert fl["rows"]["nf_prec_5_10"]["delta"] < 0 and fl["rows"]["nf_prec_5_10"]["separated"] is True       # C1 flags 9 vs 3 false rows


def test_scipy_shim_equals_scipy_binary_dilation():
    nd = pytest.importorskip("scipy.ndimage")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_r9e1_shim_nd", _OPS / "shim_scipy" / "scipy" / "ndimage" / "__init__.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rng = np.random.default_rng(0)
    for shape, p in (((500, 600), 0.001), ((500, 600), 0.3), ((37, 53), 0.05), ((1, 9), 0.5), ((9, 1), 0.5), ((3, 3), 0.0)):
        for it in (1, 2, 5):
            x = rng.random(shape) < p
            assert np.array_equal(m.binary_dilation(x, iterations=it), nd.binary_dilation(x, iterations=it)), (shape, p, it)
    gt = rng.choice(np.array([0, 1, 2, 3, 5, 7, 7, 255], np.uint8), size=(500, 600))                 # the evaluator's own call
    assert np.array_equal(m.binary_dilation(gt == 7, iterations=5), nd.binary_dilation(gt == 7, iterations=5))
    for bad in ({"iterations": 0}, {"structure": np.ones((3, 3))}, {"border_value": 1}, {"mask": np.ones((500, 600), bool)}):
        with pytest.raises(NotImplementedError):
            m.binary_dilation(gt == 7, **dict({"iterations": 5}, **bad))
