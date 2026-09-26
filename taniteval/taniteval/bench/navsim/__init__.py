"""NAVSIM v2 (EPDMS, two-stage splits) for the one-command suite — W1.

Promoted from EvalFlyWheel E1 (``2026-09-19-navsim-warmup-reference-epdms``: wrapper, controls) and
E2 (``2026-09-19-navsim-refcv4b-bridge``: export, seam agent, scoring driver, parser, artifacts,
bridge). The ``devkit_side/`` files run in the NavSim venv (Python 3.9) and are byte-identical to
their origins (``devkit_side/PROVENANCE.json``); everything else runs in the TanitAD venv.
"""


def run_benchmark(ctx):
    from .benchmark import run_benchmark as _run
    return _run(ctx)
