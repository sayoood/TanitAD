"""W5 must not disturb anything that already exists — two ways it nearly did.

1. **The legacy dashboard module.** ``taniteval/taniteval/report.py`` is imported by
   ``taniteval/taniteval/runner.py:641``, ``taniteval/rerun_all.sh:17`` and
   ``taniteval/tests/test_estimator_closeout.py``. A package called ``report`` would SHADOW it (a
   regular package beats a module file on the same path entry). W5 is therefore ``benchreport`` —
   this test pins that, so a future rename cannot silently break the legacy importers.
2. **Stdlib shadowing.** MEASURED 2026-09-20: this package contained an ``html.py``; the NavSim-venv
   gallery script put the package directory on ``sys.path`` and matplotlib's ``import html.entities``
   resolved to it — the whole gallery died. No module here may share a name with a top-level stdlib
   module, and the gallery script must not put its own directory on ``sys.path``.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "taniteval" / "benchreport"


def test_taniteval_report_still_resolves_to_the_legacy_module():
    mod = importlib.import_module("taniteval.report")
    assert Path(mod.__file__).name == "report.py"
    assert Path(mod.__file__).parent.name == "taniteval"
    for attr in ("build", "_lb_rows", "_primary"):       # what runner.py / rerun_all.sh / the closeout use
        assert callable(getattr(mod, attr)), attr


def test_w5_is_a_separate_package_and_does_not_shadow_it():
    br = importlib.import_module("taniteval.benchreport")
    assert Path(br.__file__).parent.name == "benchreport"
    assert br.REPORT_VERSION.startswith("w5")
    assert not (PKG.parent / "report").is_dir(), "a report/ PACKAGE would shadow the legacy report.py"


def test_no_module_here_shadows_a_top_level_stdlib_module():
    stdlib = set(getattr(sys, "stdlib_module_names", set()))
    assert "html" in stdlib and "json" in stdlib, "sanity: the stdlib name list is populated"
    ours = {p.stem for p in PKG.glob("*.py")} - {"__init__", "__main__"}
    clash = sorted(ours & stdlib)
    assert not clash, (f"{clash} shadow stdlib modules for anything that puts {PKG} on sys.path "
                       f"(this is how matplotlib's `import html.entities` broke the gallery)")


def test_the_navsim_side_script_never_puts_its_directory_on_sys_path():
    src = (PKG / "navsim_gallery.py").read_text(encoding="utf-8")
    assert "sys.path.insert(0, str(HERE))" not in src
    assert "_load_beside" in src, "camproj must be loaded by path, not through sys.path"


def test_importing_the_package_is_cheap():
    """``import taniteval.benchreport`` must not drag in torch: W1 calls it at the end of a run."""
    import subprocess
    code = ("import sys; import taniteval.benchreport as b; "
            "print(int('torch' in sys.modules), int('numpy' in sys.modules), b.REPORT_VERSION)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-500:]
    torch_in, _numpy_in, _v = out.stdout.split()
    assert torch_in == "0", "importing the report package pulled in torch"
