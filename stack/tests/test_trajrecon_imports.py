"""Does `tanitad.data.trajrecon` actually RUN? The byte checks never asked.

WHY THIS EXISTS
---------------
`test_trajrecon_integrity.py` reads every source as **bytes** — no control
characters, valid UTF-8 — and `test_trajrecon_contract.py` exercises the adapter.
Both passed on a package whose documented entry point could not start:

    python -m tanitad.data.trajrecon.pipeline --input-dir ... --output-dir ...
    ModuleNotFoundError: No module named 'trajlib'

Seven of the byte-exact upstream modules import each other under the upstream
distribution name (`pipeline.py:53-62`, `render_video.py:25-34`, `run_demo.py`,
`frames.py`, `diagnose.py`, `validate.py`, `contract.py`), and nothing in this
repo provided `trajlib`. The suite could not see it because **no test imported
anything** — a deliberate choice, since these modules need cv2/scipy/pandas/
matplotlib, which are an optional extra.

So the gap was not carelessness; it was a real constraint that left a real hole:
*byte-integrity was proven and runnability never was.* The fix is a check that
needs neither the heavy extras nor torch — it reads the import statements and
asserts each names a module this package actually ships.

Same shape as the traps in `CLAUDE.md`: a check that answers a different question
than the one you think it answers is worse than no check, because it reads as
coverage.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

PKG = pathlib.Path(__file__).resolve().parents[1] / "tanitad" / "data" / "trajrecon"
SOURCES = sorted(PKG.glob("*.py"))


def _trajlib_imports(path: pathlib.Path) -> set[str]:
    """Every submodule name this file pulls out of `trajlib`.

    Covers both spellings the upstream sources use:
        from trajlib import timesync as TS      -> {"timesync"}
        from trajlib.camera import calibrate    -> {"camera"}
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "trajlib":
                found.update(a.name for a in node.names)
            elif node.module.startswith("trajlib."):
                found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "trajlib" or a.name.startswith("trajlib."):
                    parts = a.name.split(".")
                    if len(parts) > 1:
                        found.add(parts[1])
    return found


def test_the_package_really_does_import_under_the_upstream_name():
    """Guard the guard: if nothing imports `trajlib`, every check below is vacuous."""
    total = set()
    for p in SOURCES:
        total |= _trajlib_imports(p)
    assert total, ("no `trajlib` imports found — either the sources were rewritten "
                   "(which breaks the byte-exactness this package depends on) or "
                   "this test is looking in the wrong place")


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_every_trajlib_import_names_a_module_this_package_ships(path: pathlib.Path):
    """The defect this file was written for.

    An upstream import that resolves to nothing is not a style issue — it is the
    package failing to start, and no byte check can see it.
    """
    for name in sorted(_trajlib_imports(path)):
        assert (PKG / f"{name}.py").is_file(), (
            f"{path.name} does `from trajlib` import {name!r}, but "
            f"{PKG.name}/{name}.py does not exist")


def test_the_alias_is_registered_and_is_this_package():
    """`trajlib` must resolve to us, or the imports above still fail at runtime.

    Imported by reading `__init__.py` rather than importing it: importing would
    pull in the parent `tanitad.data`, which needs torch, and this check has
    nothing to do with torch.
    """
    src = (PKG / "__init__.py").read_text(encoding="utf-8")
    assert re.search(r"sys\.modules\.setdefault\(\s*[\"']trajlib[\"']", src), (
        "the trajlib alias is gone from __init__.py — "
        "`python -m tanitad.data.trajrecon.pipeline` cannot start without it")


def test_the_alias_defers_to_a_real_installed_trajlib():
    """`setdefault`, never assignment.

    If someone installs the genuine upstream package, shadowing it with ours
    would silently swap which code runs — the worst kind of import bug.
    """
    src = (PKG / "__init__.py").read_text(encoding="utf-8")
    assert "sys.modules[\"trajlib\"] =" not in src
    assert "sys.modules['trajlib'] =" not in src


def test_alias_resolves_a_submodule_end_to_end():
    """The dynamic proof, skipped when the optional extras are absent.

    The static checks above cover CI without cv2/scipy/pandas. This one closes
    the loop where the extras exist: `import trajlib.<mod>` must load the file
    that lives in this package, not something else on the path.

    ⚠️ `torch` is required here and that is itself a finding, not an oversight:
    `trajrecon/__init__.py` is carefully lazy so the heavy extras stay optional,
    but importing it goes through the PARENT `tanitad/data/__init__.py`, which
    eagerly imports `toy_driving` -> `torch` (`toy_driving.py:21`). The laziness
    is defeated one level up, so the documented
    `python -m tanitad.data.trajrecon.pipeline` needs torch even though nothing
    in the pipeline itself does.
    """
    pytest.importorskip("numpy")
    pytest.importorskip("torch", reason="tanitad.data.__init__ imports toy_driving")
    import importlib
    import sys

    pkg = importlib.import_module("tanitad.data.trajrecon")
    assert sys.modules.get("trajlib") is pkg

    geo = importlib.import_module("trajlib.geo")
    assert pathlib.Path(geo.__file__).resolve() == (PKG / "geo.py").resolve()
