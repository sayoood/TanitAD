"""M-LEGACY — the ``taniteval.bench`` PACKAGE must not break the pre-existing ``taniteval.bench``
MODULE's importers (W1, 2026-09-19). MEASURED importers: runner.py:39, refc_rerank.py:78,
generalization.py:901, six test files, five stack scripts, rerun_all.sh:7 (``-m taniteval.bench --model``).
Companion evidence: the five pre-existing importer test files read 81 passed BEFORE the package and
81 passed AFTER (``…/2026-09-19-suite-core-navsim-v2/raw/legacy_importers_{BEFORE,AFTER}.txt``).
"""
from __future__ import annotations

import importlib
import sys

import pytest


def test_bench_is_the_suite_package():
    from taniteval import bench
    assert hasattr(bench, "__path__") and bench.SUITE_VERSION


def test_legacy_names_resolve_to_the_unmodified_legacy_file():
    from taniteval import bench
    from taniteval.bench import _agg, _suite, diagnostic, run      # the exact imports the old callers use
    leg = sys.modules["taniteval.bench._legacy"]
    assert run is leg.run and _agg is leg._agg and _suite is leg._suite and diagnostic is leg.diagnostic
    assert bench.run.__code__.co_filename.replace("\\", "/").endswith("taniteval/taniteval/bench.py")
    assert bench.run.__module__ == "taniteval.bench._legacy"       # importable in a child process


def test_suite_submodules_do_not_load_the_legacy_panel(monkeypatch):
    import taniteval.bench as b
    monkeypatch.setattr(b, "_load_legacy", lambda: (_ for _ in ()).throw(AssertionError("legacy loaded")))
    for name in ("contract", "schema_check", "gpu_gap", "submission", "plugins", "report_hook"):
        importlib.import_module(f"taniteval.bench.{name}")        # must not call _load_legacy


def test_no_suite_submodule_shadows_a_legacy_name():
    import taniteval.bench as b
    leg = importlib.import_module("taniteval.bench._legacy")
    clash = sorted(set(b._SUITE_SUBMODULES) & {n for n in dir(leg) if not n.startswith("__")})
    assert clash == [], f"suite submodule(s) {clash} shadow legacy attributes of the same name"


def test_unknown_name_raises_attributeerror():
    import taniteval.bench as b
    with pytest.raises(AttributeError):
        b.this_name_does_not_exist_anywhere


def test_deliberate_regression_without_delegation_goes_red(monkeypatch):
    """Remove the PEP 562 delegation -> the legacy import the runner uses must FAIL (the test above
    would go RED). Proves the compat test is not green by construction."""
    import taniteval.bench as b
    importlib.import_module("taniteval.bench._legacy")
    saved = {k: b.__dict__.pop(k) for k in ("run",) if k in b.__dict__}
    monkeypatch.delitem(b.__dict__, "__getattr__")
    try:
        with pytest.raises(AttributeError):
            getattr(b, "run")
    finally:
        b.__dict__.update(saved)


@pytest.mark.parametrize("argv,legacy", [
    (["--model", "flagship4b-speedjerk-30k", "--episodes", "40"], True),     # rerun_all.sh:7
    (["--all"], True),
    (["legacy", "--help"], True),
    (["navsim_v2", "--ckpt", "none", "--split", "warmup_two_stage"], False),
    (["submit", "x", "--target", "hf_navsim_navhard"], False),
    ([], False),
])
def test_cli_routing_literals(argv, legacy):
    from taniteval.bench.__main__ import is_legacy_invocation
    assert is_legacy_invocation(argv) is legacy
