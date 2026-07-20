"""TanitEval — a world-class evaluation harness for the TanitAD AD world-models.
Benchmarks + profiling + leaderboard + A/B + dashboard. v0.1."""
__version__ = "0.1.0"

# TEMP assess 2026-07-19 (restore from .assess_backup_20260719 after):
# TANITEVAL_STACK_OVERRIDE points `tanitad` at a training pod's own
# working-tree stack copy (v2 arms need v2 model code the default
# /root/TanitAD/stack predates). This package __init__ runs before every
# taniteval submodule, so importing tanitad HERE pins it from the override
# path — later sys.path.insert(0, "/root/TanitAD/stack") calls in the
# submodules cannot re-import it (sys.modules cache wins).
import os as _os
import sys as _sys

_ov = _os.environ.get("TANITEVAL_STACK_OVERRIDE")
if _ov:
    _sys.path.insert(0, _ov)
    import tanitad  # noqa: F401
    print(f"[taniteval] tanitad OVERRIDE -> {_ov} "
          f"({tanitad.__file__})", file=_sys.stderr, flush=True)
