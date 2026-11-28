"""thor_gate_runner2.py: the bounded per-file worker pool (THOR_GATE_WORKERS).

N=1 (the default) must be the serial run; N>1 must produce the same per-file results, junit files and record ORDER, only
faster. A deliberate-regression mutant (two workers writing one junit path) must make the comparison go RED."""
import importlib.util
import json
import os
import pathlib
import sys
import tarfile
import xml.etree.ElementTree as ET

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "stack" / "ops" / "gate" / "thor_gate_runner2.py"

TEST_BODY = '''
import os, time


def test_it():
    t0 = time.time()
    time.sleep(1.5)
    with open(os.environ["GATE_MARKS"], "a", encoding="utf-8") as fh:
        fh.write("%s %.3f %.3f\\n" % (os.environ.get("TMPDIR", "-"), t0, time.time()))
    assert {FAIL} == 0
'''


def _load(src, name):
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, f"<{name}>", "exec"), mod.__dict__)
    orig = mod.env_for

    def env_for(tree, tmpdir=None):            # the runner joins PYTHONPATH with ':' (Thor); this box needs os.pathsep
        e = orig(tree, tmpdir)
        e["PYTHONPATH"] = os.pathsep.join([f"{tree}/stack", f"{tree}/stack/scripts", f"{tree}/taniteval", str(mod.PYT)])
        return e
    mod.env_for = env_for
    mod.PY = sys.executable
    return mod


def _fake_suite(root):
    """A tree archive with 4 stack files + 2 taniteval files (one of them failing), and the plan that names them."""
    src = root / "src"
    (src / "stack" / "tanitad").mkdir(parents=True)
    (src / "stack" / "tanitad" / "__init__.py").write_text("", encoding="utf-8")
    (src / "taniteval" / "taniteval").mkdir(parents=True)
    (src / "taniteval" / "taniteval" / "__init__.py").write_text("", encoding="utf-8")
    stack_files = [f"tests/test_s{i}.py" for i in range(4)]
    tev_files = ["tests/test_e0.py", "tests/test_e1.py"]
    for f, sub, fail in [(f, "stack", 0) for f in stack_files] + [(tev_files[0], "taniteval", 0), (tev_files[1], "taniteval", 1)]:
        p = src / sub / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(TEST_BODY.replace("{FAIL}", str(fail)), encoding="utf-8")
    run = root / "run"
    run.mkdir()
    with tarfile.open(run / "tree.tar", "w") as tf:
        for child in sorted(src.iterdir()):
            tf.add(child, arcname=child.name)
    plan = {"archive": "tree.tar", "overlay_md5": {}, "stages": {"S": {"overlay": [], "stack": stack_files, "taniteval": tev_files}}}
    (run / "plan_thor.json").write_text(json.dumps(plan), encoding="utf-8")
    (root / "pyt").mkdir()
    return run, plan


def _run(mod, root, workers, monkeypatch):
    marks = root / f"marks_{workers}.txt"
    monkeypatch.setenv("GATE_MARKS", str(marks))
    monkeypatch.setenv("THOR_GATE_WORKERS", str(workers))
    # Measure the runner's OWN TMPDIR policy from an unset baseline: under a multi-worker gate suite (study e3: 3 workers) the OUTER runner gives
    # this pytest a per-slot TMPDIR, the inner N=1 children inherited it, and "no TMPDIR override" failed on Thor (MEASURED 2026-11-24 02:40Z,
    # suite_ba457a792581) although the runner set none.
    monkeypatch.delenv("TMPDIR", raising=False)
    src_run, plan = _fake_suite(root / f"w{workers}")
    run = src_run
    # the inner pytest must import pytest from where THIS one did: the Thor venv has no pytest (the gate passes its --no-deps side
    # folder as PYT, /home/nvidia/refcv8_audit/pyt), so an empty PYT made every inner file rc 1 with no junit there (K2DC02 gate, 2026-11-18)
    mod.R, mod.PYT, mod.plan, mod.OUT = run, pathlib.Path(pytest.__file__).resolve().parents[1], plan, run / "results"
    mod.OUT.mkdir()
    mod.run_stage("S", plan["stages"]["S"])
    rec = json.loads((mod.OUT / "S.json").read_text(encoding="utf-8"))
    xmls = {p.name: sorted(tc.get("name") + ":" + ("F" if tc.find("failure") is not None else "P")
                           for tc in ET.parse(p).getroot().iter("testcase"))
            for p in (mod.OUT / "S").glob("*.xml")}
    spans = [tuple(map(float, ln.split()[1:])) for ln in marks.read_text(encoding="utf-8").splitlines()] if marks.exists() else []
    return rec, xmls, spans, run


def _shape(rec):
    """Everything the verdict reads from the record, minus timing and the free-text tail."""
    return [(f["cwd"], f["file"], f["rc"], f["junit"]) for f in rec["files"]], rec.get("done"), rec.get("error")


def _peak(spans):
    ev = sorted([(a, 1) for a, _ in spans] + [(b, -1) for _, b in spans])
    cur = peak = 0
    for _, d in ev:
        cur += d
        peak = max(peak, cur)
    return peak


@pytest.fixture(scope="module")
def serial_ref(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        mod = _load(RUNNER.read_text(encoding="utf-8"), "runner_serial")
        return _run(mod, tmp_path_factory.mktemp("ser"), 1, mp)
    finally:
        mp.undo()


def test_n1_is_the_serial_order_and_outputs(serial_ref):
    rec, xmls, spans, run = serial_ref
    assert rec.get("done") is True and "error" not in rec
    assert [f["file"] for f in rec["files"]] == [f"tests/test_s{i}.py" for i in range(4)] + ["tests/test_e0.py", "tests/test_e1.py"]
    assert [f["rc"] for f in rec["files"]] == [0, 0, 0, 0, 0, 1]
    assert sorted(xmls) == sorted([f"stack__tests__test_s{i}.py.xml" for i in range(4)]
                                  + ["taniteval__tests__test_e0.py.xml", "taniteval__tests__test_e1.py.xml"])
    assert _peak(spans) == 1                                   # one pytest at a time
    assert all(t == "-" for t in (ln.split()[0] for ln in (run.parent.parent / "marks_1.txt").read_text(encoding="utf-8").splitlines()))   # no TMPDIR override


def test_n3_gives_the_same_per_file_results(serial_ref, tmp_path, monkeypatch):
    mod = _load(RUNNER.read_text(encoding="utf-8"), "runner_par")
    rec, xmls, spans, run = _run(mod, tmp_path, 3, monkeypatch)
    assert (_shape(rec), xmls) == (_shape(serial_ref[0]), serial_ref[1])
    assert _peak(spans) >= 2                                   # they really overlapped
    assert _peak(spans) <= 3                                   # and never beyond the bound
    tmps = {ln.split()[0] for ln in (tmp_path / "marks_3.txt").read_text(encoding="utf-8").splitlines()}
    assert len(tmps) >= 2 and all("wtmp" in t for t in tmps)  # worker slots have their own TMPDIR
    assert not (run / "wtmp" / "S").exists()                       # and it is cleaned up


def _same(mod, tmp_path, monkeypatch, ref):
    rec, xmls, _, _ = _run(mod, tmp_path, 3, monkeypatch)
    return (_shape(rec), xmls) == (_shape(ref[0]), ref[1])


def test_mutant_two_workers_one_junit_path_goes_red(serial_ref, tmp_path_factory, monkeypatch):
    src = RUNNER.read_text(encoding="utf-8")
    needle = 'f.replace("/", "__")'
    assert src.count(needle) == 1
    mut_src = src.replace(needle, '"X"')                      # every file of a cwd now writes the same junit path
    assert _same(_load(src, "m_ok"), tmp_path_factory.mktemp("ok"), monkeypatch, serial_ref) is True     # the control reads True
    assert _same(_load(mut_src, "m_bad"), tmp_path_factory.mktemp("bad"), monkeypatch, serial_ref) is False
    # and with the collision guard removed too, the silent overwrite (fewer junit files) is still caught
    guard = 'assert len(set(paths)) == len(paths), "two files map to one junit path"'
    assert guard in mut_src
    assert _same(_load(mut_src.replace(guard, "pass"), "m_bad2"), tmp_path_factory.mktemp("bad2"), monkeypatch, serial_ref) is False


def test_default_is_one_worker_and_bad_values_fall_back(monkeypatch):
    mod = _load(RUNNER.read_text(encoding="utf-8"), "runner_env")
    monkeypatch.delenv("THOR_GATE_WORKERS", raising=False)
    assert mod.n_workers() == 1
    for v, want in (("3", 3), ("0", 1), ("-2", 1), ("abc", 1), ("", 1)):
        monkeypatch.setenv("THOR_GATE_WORKERS", v)
        assert mod.n_workers() == want
