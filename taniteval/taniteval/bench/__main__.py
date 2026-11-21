"""``python -m taniteval.bench …`` — routes to the suite CLI, or to the LEGACY diagnostic panel.

Legacy routing (so ``taniteval/rerun_all.sh:7`` — ``python3 -m taniteval.bench --model "$k"
--episodes 40`` — keeps working): the legacy panel's CLI takes ONLY the flags
``--model --all --episodes --device``. An argv whose first token is not a suite subcommand and
which carries ``--model`` or ``--all`` is handed, unchanged, to the legacy ``main()``.
``python -m taniteval.bench legacy …`` reaches it explicitly (e.g. ``legacy --help``).
"""
from __future__ import annotations

import sys

LEGACY_FLAGS = ("--model", "--all")


def is_legacy_invocation(argv: list) -> bool:
    from taniteval.bench.cli import SUBCOMMANDS
    if not argv:
        return False
    if argv[0] == "legacy":
        return True
    if argv[0] in SUBCOMMANDS:
        return False
    return any(a == f or a.startswith(f + "=") for a in argv for f in LEGACY_FLAGS)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if is_legacy_invocation(argv):
        from taniteval.bench import _legacy
        rest = argv[1:] if argv and argv[0] == "legacy" else argv
        sys.argv = ["taniteval.bench"] + rest
        print("[taniteval.bench] legacy diagnostic panel (taniteval/taniteval/bench.py) — the suite "
              "CLI is `python -m taniteval.bench <benchmark> …`", file=sys.stderr, flush=True)
        _legacy.main()
        return 0
    from taniteval.bench.cli import main as suite_main
    return suite_main(argv)


if __name__ == "__main__":
    sys.exit(main())
