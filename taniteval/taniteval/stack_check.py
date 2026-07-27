"""``python3 -m taniteval.stack_check`` — the one-second preflight for every eval.

⭐ Put this in FRONT of an eval command and a stale `tanitad` costs one second
instead of one wrong number:

    TANITEVAL_STACK_OVERRIDE=/workspace/TanitAD/stack \\
    PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts:/workspace/TanitAD/taniteval \\
    python3 -m taniteval.stack_check --require v5 --json results/stack_guard_v5.json

Exit **0** = the stack the caller named is the stack that will be imported, and
it carries the required post-v5 symbols. Exit **2** = it is not.

⚠️ This module is a thin entry point on purpose. Running
``python3 -m taniteval.stack_guard`` works too, but ``taniteval/__init__`` has
already imported ``stack_guard`` by then, so ``runpy`` prints a
*"found in sys.modules … may result in unpredictable behaviour"* warning — noise
in a command people are meant to copy. See :mod:`taniteval.stack_guard` for the
mechanism, the three layers and the demonstrated failure.
"""
from taniteval.stack_guard import main

if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
