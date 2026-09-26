"""Benchmark PLUGIN slots of the one-command suite.

Orchestrator ruling 2026-09-19 (EvalFlyWheel): each slot below is owned EXCLUSIVELY by the named
work package, which replaces the docstring-only stub W1 wrote with an implementation of
``run_benchmark(ctx)``:

    taniteval/taniteval/bench/plugins/navsim_v1.py     W3  (NAVSIM v1.1, PDMS_v1_navtest)
    taniteval/taniteval/bench/plugins/nuscenes_ol.py   W6  (nuScenes open-loop L2 / collision)

W1 owns this ``__init__`` and the loader. A slot whose module defines no callable
``run_benchmark`` is refused by the CLI with ``PLUGIN_NOT_LANDED`` (exit 2) — never run half-built.

The ``ctx`` contract is :class:`taniteval.bench.contract.BenchContext` (read its docstring); the
record the plugin must leave is validated against ``bench/schema/{bench_run,summary}.schema.json``
by the core after ``run_benchmark`` returns.
"""
from __future__ import annotations

import importlib

PLUGIN_SLOTS = {"navsim_v1": "W3", "nuscenes_ol": "W6"}


class PluginNotLanded(RuntimeError):
    pass


def load(benchmark: str):
    """-> the plugin's ``run_benchmark`` callable, or raise :class:`PluginNotLanded`."""
    if benchmark not in PLUGIN_SLOTS:
        raise KeyError(benchmark)
    owner = PLUGIN_SLOTS[benchmark]
    try:
        mod = importlib.import_module(f"{__name__}.{benchmark}")
    except ModuleNotFoundError as e:
        raise PluginNotLanded(f"PLUGIN_NOT_LANDED: {benchmark} (owner {owner}) — module missing ({e})") from e
    fn = getattr(mod, "run_benchmark", None)
    if not callable(fn):
        raise PluginNotLanded(f"PLUGIN_NOT_LANDED: {benchmark} (owner {owner}) — "
                              f"{mod.__file__} defines no run_benchmark(ctx) yet (docstring-only stub)")
    return fn
