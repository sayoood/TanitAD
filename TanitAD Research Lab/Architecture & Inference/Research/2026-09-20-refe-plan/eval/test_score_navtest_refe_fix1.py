"""NT-FIX1 reachability: ``score_navtest_refe.main`` must score THROUGH ``run_v1_fix1.load()``.

A wrapper that no caller loads fixes nothing (CLAUDE.md, "built, tested, and unreachable from its
caller"). The W3 code dir is replaced by a tmp dir holding a FAKE ``run_v1.py`` (its ``cmd_score``
records ``UNFIXED``) AND a FAKE ``run_v1_fix1.py`` (its ``load()`` returns a module whose
``cmd_score`` records ``FIX1``). The caller must reach ``FIX1``. Pointed at the pre-NT-FIX1 caller
(``SCORE_NAVTEST_REFE=<path>``) the first test goes RED -- that is the mutation arm.

    python -m pytest -q eval/test_score_navtest_refe_fix1.py
"""
from __future__ import annotations

import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CALLER = os.environ.get("SCORE_NAVTEST_REFE", os.path.join(HERE, "score_navtest_refe.py"))

FAKE_RUN_V1 = '''
RAW = PKG = None
def cmd_score(ns):
    import json, os
    open(os.path.join(RAW, "WHO.json"), "w").write(json.dumps({"who": "UNFIXED", "label": ns.label}))
    return 0
'''

FAKE_FIX1 = '''
import importlib.util, os, sys, types
def load():
    m = types.ModuleType("fake_fix1_w3")
    m.RAW = m.PKG = None
    m.FIX1 = {"version": 1}
    def cmd_score(ns):
        import json
        open(os.path.join(m.RAW, "WHO.json"), "w").write(json.dumps({"who": "FIX1", "label": ns.label,
                                                                     "arm": ns.arm}))
        return 0
    m.cmd_score = cmd_score
    return m
'''


def _caller(w3_code):
    spec = importlib.util.spec_from_file_location("score_navtest_refe_under_test", CALLER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.W3_CODE = str(w3_code)
    return mod


def _w3(tmp_path, with_fix1=True):
    d = tmp_path / "w3code"
    d.mkdir()
    (d / "run_v1.py").write_text(FAKE_RUN_V1, encoding="utf-8")
    if with_fix1:
        (d / "run_v1_fix1.py").write_text(FAKE_FIX1, encoding="utf-8")
    return d


def test_caller_scores_through_fix1(tmp_path):
    mod = _caller(_w3(tmp_path))
    out = tmp_path / "out"
    seam = tmp_path / "seam.npz"
    seam.write_bytes(b"")
    rc = mod.main(["--label", "refe_nt_fix1_probe", "--seam", str(seam), "--out", str(out)])
    assert rc == 0
    who = json.load(open(out / "WHO.json", encoding="utf-8"))
    assert who["who"] == "FIX1", f"caller scored through {who['who']}, not run_v1_fix1.load()"
    assert who["label"] == "refe_nt_fix1_probe" and who["arm"].startswith("SEAM:")


def test_missing_fix1_is_loud_not_a_silent_fallback(tmp_path):
    mod = _caller(_w3(tmp_path, with_fix1=False))
    seam = tmp_path / "seam.npz"
    seam.write_bytes(b"")
    with pytest.raises(SystemExit) as e:
        mod.main(["--label", "refe_nt_fix1_probe", "--seam", str(seam), "--out", str(tmp_path / "out")])
    assert "run_v1_fix1.py is missing" in str(e.value)
    assert not (tmp_path / "out" / "WHO.json").exists()


def test_caller_source_names_fix1_once_and_no_direct_run_v1_exec():
    src = open(CALLER, encoding="utf-8").read()
    assert src.count('"run_v1_fix1.py"') == 1
    assert 'os.path.join(W3_CODE, "run_v1.py")' not in src
