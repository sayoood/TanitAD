"""NT-FIX1 reachability: ``score_navtest7.main`` must score THROUGH ``run_v1_fix1.load()``.

A wrapper that no caller loads fixes nothing (CLAUDE.md, "built, tested, and unreachable from its caller").
The W3 code dir is replaced by a tmp dir holding a FAKE ``run_v1.py`` (its ``cmd_score`` records ``UNFIXED``)
AND a FAKE ``run_v1_fix1.py`` (its ``load()`` returns a module whose ``cmd_score`` records ``FIX1``). The caller
must reach ``FIX1``. Pointed at the pre-NT-FIX1 caller (``SCORE_NAVTEST7=<path>``) the first test goes RED --
that is the mutation arm. Pattern: ``2026-09-20-refe-plan/eval/test_score_navtest_refe_fix1.py``.

ISOLATION. ``score_navtest7.main`` wraps the scorer in ``ram_governor7`` (a shared scoring-slot lock, a per-job
lock, and a RAM watcher that can SUSPEND the process tree). A test must never touch those, so a FAKE
``ram_governor7`` is injected into ``sys.modules`` before ``main`` runs; ``test_fake_governor_was_the_one_used``
proves the fake -- not the real governor -- was entered around ``cmd_score``.

    python -m pytest -q tests/test_score_navtest7_fix1.py
    SCORE_NAVTEST7=<pre-NT-FIX1 score_navtest7.py> python -m pytest -q tests/test_score_navtest7_fix1.py   # RED
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
CALLER = os.environ.get("SCORE_NAVTEST7", os.path.join(PKG, "code", "score_navtest7.py"))

FAKE_RUN_V1 = '''
RAW = PKG = None
def cmd_score(ns):
    import json, os
    open(os.path.join(RAW, "WHO.json"), "w").write(json.dumps({"who": "UNFIXED", "label": ns.label}))
    return 0
'''

FAKE_FIX1 = '''
import os, types
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


@pytest.fixture
def fake_governor(monkeypatch):
    """Stand-in for ``ram_governor7``: no lock file, no RAM watcher, no process suspension."""
    calls = {"governed": 0, "joblock": 0, "already_scored": 0}
    rg = types.ModuleType("ram_governor7")

    class _JobLock:
        def __init__(self, name):
            calls["joblock"] += 1

        def acquire(self):
            pass

        def release(self):
            pass

    @contextlib.contextmanager
    def governed(label, odir, est_rss_mb=0.0):
        calls["governed"] += 1
        yield types.SimpleNamespace(summary=lambda: {"fake_governor": True})

    def already_scored(counts_path, seam, csv_path):
        calls["already_scored"] += 1
        return None

    rg.JobLock, rg.governed, rg.already_scored = _JobLock, governed, already_scored
    monkeypatch.setitem(__import__("sys").modules, "ram_governor7", rg)
    return calls


def _caller(w3_code):
    spec = importlib.util.spec_from_file_location("score_navtest7_under_test", CALLER)
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


def test_caller_scores_through_fix1(tmp_path, fake_governor):
    mod = _caller(_w3(tmp_path))
    out = tmp_path / "out"
    seam = tmp_path / "seam.npz"
    seam.write_bytes(b"")
    rc = mod.main(["--label", "r7_nt_fix1_probe", "--seam", str(seam), "--out", str(out)])
    assert rc == 0
    who = json.load(open(out / "WHO.json", encoding="utf-8"))
    assert who["who"] == "FIX1", f"caller scored through {who['who']}, not run_v1_fix1.load()"
    assert who["label"] == "r7_nt_fix1_probe" and who["arm"].startswith("SEAM:")


def test_fake_governor_was_the_one_used(tmp_path, fake_governor):
    """Isolation proof: the injected fake governor wrapped cmd_score; the real one (locks, RAM watcher) was not."""
    mod = _caller(_w3(tmp_path))
    seam = tmp_path / "seam.npz"
    seam.write_bytes(b"")
    mod.main(["--label", "r7_nt_fix1_probe", "--seam", str(seam), "--out", str(tmp_path / "out")])
    assert fake_governor == {"governed": 1, "joblock": 1, "already_scored": 1}


def test_official_control_path_also_scores_through_fix1(tmp_path, fake_governor):
    """KH-navtest's ``--official STOP`` path (no seam, so no per-job lock) goes through the same loader."""
    mod = _caller(_w3(tmp_path))
    out = tmp_path / "out"
    rc = mod.main(["--label", "r7_nt_fix1_probe", "--official", "STOP", "--out", str(out)])
    assert rc == 0
    who = json.load(open(out / "WHO.json", encoding="utf-8"))
    assert who["who"] == "FIX1" and who["arm"] == "STOP"
    assert fake_governor["joblock"] == 0 and fake_governor["governed"] == 1


def test_missing_fix1_is_loud_not_a_silent_fallback(tmp_path, fake_governor):
    mod = _caller(_w3(tmp_path, with_fix1=False))
    seam = tmp_path / "seam.npz"
    seam.write_bytes(b"")
    with pytest.raises(SystemExit) as e:
        mod.main(["--label", "r7_nt_fix1_probe", "--seam", str(seam), "--out", str(tmp_path / "out")])
    assert "run_v1_fix1.py is missing" in str(e.value)
    assert not (tmp_path / "out" / "WHO.json").exists()
    assert fake_governor["governed"] == 0            # it stopped BEFORE any scoring started


def test_caller_source_names_fix1_once_and_no_direct_run_v1_exec():
    src = open(CALLER, encoding="utf-8").read()
    assert src.count('"run_v1_fix1.py"') == 1
    assert 'os.path.join(W3_CODE, "run_v1.py")' not in src
