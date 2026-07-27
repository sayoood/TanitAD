"""TanitEval — a world-class evaluation harness for the TanitAD AD world-models.
Benchmarks + profiling + leaderboard + A/B + dashboard. v0.1."""
__version__ = "0.1.0"

# ⛔ STACK PINNING — NOT TEMPORARY, and not safe to "restore" away.
#
# This block used to be labelled `TEMP assess 2026-07-19 (restore from
# .assess_backup_20260719 after)`. It is the CURE for a live defect, and that
# label invited someone to delete it:
#
#   28 modules of this package hardcoded `sys.path.insert(0, "/root/TanitAD/stack")`,
#   which puts that tree IN FRONT of whatever the caller put on PYTHONPATH. On
#   pod2 (MEASURED 2026-07-27) that tree is 12 MB, has no `.git`, no
#   `tanitad/train/heldout_gate.py` and no `resolve_v2_frames` — i.e. pre-v5.
#   ⇒ merely importing `taniteval.<anything>` re-pointed `tanitad` at pre-v5 code.
#   ⚠️ The crash is the LUCKY case: a module present in BOTH trees resolves
#   silently to the stale one and the eval publishes a plausible wrong number.
#   ⚠️ `import tanitad` on its own resolves CORRECTLY, so the obvious sync check
#   does NOT catch this.  (SMALL_VALIDATION.md §5; STALE_IMPORT_GUARD.md)
#
# Importing `tanitad` HERE, from the override, pins it in `sys.modules` before any
# submodule runs, so every later insert loses. The sentinel installed after it
# then refuses any `tanitad.*` that would still come from elsewhere — including
# from a module this package never fixed, or a command copied from a document.
import os as _os
import sys as _sys

from . import stack_guard as _stack_guard  # noqa: E402

_ov = _os.environ.get(_stack_guard.ENV_OVERRIDE)
if _ov:
    _sys.path.insert(0, _ov)
    import tanitad  # noqa: F401,E402
    print(f"[taniteval] tanitad OVERRIDE -> {_ov} "
          f"({tanitad.__file__})", file=_sys.stderr, flush=True)

# ⭐ The path fix + the tripwire, for EVERY import of this package — with or
# without the override, because "the next person copies a command from somewhere
# else". The guard can only fire when the caller NAMED a stack (env var /
# PYTHONPATH / cwd / an already-imported tanitad) and a different one won; when
# nothing is named the deployed `/root/TanitAD/stack` IS the intent and the guard
# has no teeth. See `stack_guard.resolve_intended_stack`.
_stack_pin = _stack_guard.ensure_stack_on_path()

assert_stack = _stack_guard.assert_stack
StackShadowError = _stack_guard.StackShadowError
stack_report = _stack_guard.report
