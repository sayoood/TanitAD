"""NT-FIX1 reachability: ``linux_run_v1`` (the Colab / Linux launcher) must load W3's harness THROUGH
``run_v1_fix1.load()``, for both ``score`` and ``cache``.

A wrapper that no caller loads fixes nothing (CLAUDE.md, "built, tested, and unreachable from its caller").
The ``--w3-code`` dir is a tmp dir holding a FAKE ``run_v1.py`` (its ``cmd_score`` / ``cmd_cache`` record
``UNFIXED``) AND a FAKE ``run_v1_fix1.py`` (its ``load()`` returns a module whose ``cmd_score`` / ``cmd_cache``
record ``FIX1``). The launcher must reach ``FIX1``. Pointed at the pre-NT-FIX1 launcher
(``LINUX_RUN_V1=<path>``) the reachability tests go RED -- that is the mutation arm.

The last test uses the REAL ``run_v1.py`` + ``run_v1_fix1.py`` from this checkout: ``load_w3`` must return the
byte-verified, patched module (and still apply the Linux path re-pointing).

    python -m pytest -q colab/navtest/test_linux_run_v1_fix1.py
    LINUX_RUN_V1=<pre-NT-FIX1 linux_run_v1.py> python -m pytest -q colab/navtest/test_linux_run_v1_fix1.py   # RED
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CALLER = os.environ.get("LINUX_RUN_V1", os.path.join(HERE, "linux_run_v1.py"))
REAL_W3 = os.path.join(REPO, "FlyWheels", "TanitAD_EvalFlyWheel", "incoming", "2026-09-19-navsim-v1-navtest", "code")

FAKE_RUN_V1 = '''
RAW = PKG = None
ENV = {}
def _who(tag, ns):
    import json, os
    open(os.path.join(RAW, "WHO.json"), "w").write(json.dumps({"who": tag, "label": ns.label}))
    return 0
def cmd_score(ns):
    return _who("UNFIXED", ns)
def cmd_cache(ns):
    return _who("UNFIXED_CACHE", ns)
'''

FAKE_FIX1 = '''
import os, types
def load():
    m = types.ModuleType("fake_fix1_w3")
    m.RAW = m.PKG = None
    m.ENV = {}
    m.FIX1 = {"version": 1}
    def _who(tag, ns):
        import json
        open(os.path.join(m.RAW, "WHO.json"), "w").write(json.dumps({"who": tag, "label": ns.label,
                                                                     "arm": getattr(ns, "arm", None)}))
        return 0
    m.cmd_score = lambda ns: _who("FIX1", ns)
    m.cmd_cache = lambda ns: _who("FIX1_CACHE", ns)
    return m
'''


def _caller():
    spec = importlib.util.spec_from_file_location("linux_run_v1_under_test", CALLER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w3(tmp_path, with_fix1=True):
    d = tmp_path / "w3code"
    d.mkdir()
    (d / "run_v1.py").write_text(FAKE_RUN_V1, encoding="utf-8")
    if with_fix1:
        (d / "run_v1_fix1.py").write_text(FAKE_FIX1, encoding="utf-8")
    return d


def _argv(w3, out, cmd="score", seam=None):
    base = [cmd, "--label", "refe_nt_fix1_probe", "--out", str(out), "--w3-code", str(w3), "--e1-wrapper", "e1.py",
            "--nv-py", "py", "--v11-tree", "v11", "--nuplan-tree", "nup", "--exp", "exp", "--openscene", "osc",
            "--maps", "maps"]
    return base + (["--arm", f"SEAM:{seam}"] if cmd == "score" else [])


def test_score_runs_through_fix1(tmp_path):
    mod = _caller()
    out, seam = tmp_path / "out", tmp_path / "seam.npz"
    seam.write_bytes(b"")
    assert mod.main(_argv(_w3(tmp_path), out, "score", seam)) == 0
    who = json.load(open(out / "WHO.json", encoding="utf-8"))
    assert who["who"] == "FIX1", f"launcher scored through {who['who']}, not run_v1_fix1.load()"
    assert who["label"] == "refe_nt_fix1_probe" and who["arm"].startswith("SEAM:")


def test_cache_runs_through_the_fix1_module_too(tmp_path):
    mod = _caller()
    out = tmp_path / "out"
    assert mod.main(_argv(_w3(tmp_path), out, "cache")) == 0
    assert json.load(open(out / "WHO.json", encoding="utf-8"))["who"] == "FIX1_CACHE"


def test_missing_fix1_is_loud_not_a_silent_fallback(tmp_path):
    mod = _caller()
    out, seam = tmp_path / "out", tmp_path / "seam.npz"
    seam.write_bytes(b"")
    with pytest.raises(SystemExit) as e:
        mod.main(_argv(_w3(tmp_path, with_fix1=False), out, "score", seam))
    assert "run_v1_fix1.py is missing" in str(e.value)
    assert not (out / "WHO.json").exists()


def test_launcher_source_names_fix1_once_and_no_direct_run_v1_exec():
    src = open(CALLER, encoding="utf-8").read()
    assert src.count('"run_v1_fix1.py"') == 1
    assert 'os.path.join(a.w3_code, "run_v1.py")' not in src


def test_real_w3_pair_loads_pinned_and_patched(tmp_path):
    if not (os.path.isfile(os.path.join(REAL_W3, "run_v1.py"))
            and os.path.isfile(os.path.join(REAL_W3, "run_v1_fix1.py"))):
        pytest.skip(f"real W3 pair not in this checkout: {REAL_W3}")
    mod = _caller()
    a = argparse.Namespace(w3_code=REAL_W3, nv_py="NVPY", v11_tree="V11", exp=str(tmp_path / "exp"), openscene="OSC",
                           maps="MAPS", e1_wrapper="E1.py", nuplan_tree="NUP", windowspath_shim=False,
                           out=str(tmp_path / "out"))
    w3 = mod.load_w3(a)
    assert w3.FIX1["version"] == 1 and w3.FIX1["original"]["matches_pinned"] is True
    assert w3.cmd_score.__qualname__.startswith("_wrap_cmd_score"), "cmd_score is not fix1's wrapper"
    assert w3.NV_PY == "NVPY" and w3.WRAP.endswith("navsim_v1_linux.py")          # the Linux re-pointing still applies
    assert w3.RAW == os.path.abspath(a.out)
