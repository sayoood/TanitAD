"""⛔⛔ The refcv6 trainer must IMPORT on a branch that does not carry the refcv7 modules.

⭐⭐ THE DEFECT, MEASURED 2026-09-23 on the first CLEAN-TREE suite (a `git archive` of the branch
tip plus one commit's files): 16 test modules died at COLLECTION with
``ImportError: cannot import name 'refcv7_heads' from 'tanitad.refs'``. ``refc_v3_train.py``
imported ``refcv7_heads`` / ``refcv7_oracle`` at module scope since ``d014414``, which swept the
refcv7 stream's trainer hooks in from a worktree that ALSO held the stream's unlanded modules --
so every suite run on that worktree stayed green while the branch itself could not import its
own trainer, and a refcv6 launch shipped from the tip would have died at line 99.

What is pinned, each against the REAL import machinery (a subprocess with a meta-path finder
that makes the modules absent, exactly as they are on the tip):
(a) with both refcv7 modules absent the trainer imports, and ``r7h``/``r7o`` are ``None``;
(b) ⛔ THE CONTROL: with a DIFFERENT module absent the import still FAILS -- the tolerance is
    scoped to refcv7's own absence and cannot widen into swallowing real breakage;
(c) ``--refcv7`` REFUSES while the modules are missing, naming them, instead of failing later
    on ``None.wta_loss`` after the run is paid for.
CPU only.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_PROBE = r'''
import importlib.abc, sys
BLOCK = set(sys.argv[1].split(","))
class _Absent(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name in BLOCK:
            raise ImportError("cannot import name %r from 'tanitad.refs'" % name.rsplit(".", 1)[-1])
        return None
sys.meta_path.insert(0, _Absent())
sys.path.insert(0, sys.argv[2])
import refc_v3_train as T
print("R7_NONE", T.r7h is None and T.r7o is None)
'''


def _import_with_absent(modules: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8",
               CUDA_VISIBLE_DEVICES="")
    return subprocess.run(
        [sys.executable, "-c", _PROBE, ",".join(modules), str(ROOT / "scripts")],
        capture_output=True, encoding="utf-8", errors="replace", env=env, timeout=600)


def test_the_trainer_IMPORTS_when_the_refcv7_modules_are_absent():
    # ⚠️ Premise check, not a convenience skip: in a tree whose MODEL file already imports the
    # refcv7 modules (the refcv7 stream's model half is present), their absence is not a
    # configuration that can exist, and the model -- not the trainer -- would fail first.
    if "refcv7_heads" in (ROOT / "tanitad" / "refs" / "refc_v3.py").read_text(encoding="utf-8"):
        pytest.skip("this tree's refc_v3.py itself requires the refcv7 modules")
    r = _import_with_absent(["tanitad.refs.refcv7_heads", "tanitad.refs.refcv7_oracle"])
    assert r.returncode == 0, r.stderr[-1500:]
    assert "R7_NONE True" in r.stdout


def test_CONTROL_any_other_missing_module_still_FAILS_the_import():
    """⛔ Without this, an `except ImportError: pass` would pass the test above."""
    r = _import_with_absent(["tanitad.refs.refcv6_max_speed"])
    assert r.returncode != 0
    assert "refcv6_max_speed" in r.stderr


def test_refcv7_REFUSES_by_name_while_its_modules_are_missing(monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import refc_v3_train as T
    from tanitad.refs import refc_v3 as v3
    monkeypatch.setattr(T, "r7h", None)
    monkeypatch.setattr(T, "r7o", None)
    args = T.build_parser().parse_args(["--arm", "hier", "--out", "x", "--refcv7"])
    with pytest.raises(SystemExit) as e:
        T._pin_refcv7(v3.RefCV3Config(hier=True), args)
    assert "NOT IN THIS TREE" in str(e.value)
