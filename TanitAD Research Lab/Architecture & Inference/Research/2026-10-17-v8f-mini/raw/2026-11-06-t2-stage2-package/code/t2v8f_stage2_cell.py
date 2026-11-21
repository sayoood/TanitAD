"""One stage-2 rollout cell: `t2v8f_run.py`'s rollout entry with an `--executor {base,hold}` switch (default base) in front.

    python -I t2v8f_stage2_cell.py [--executor base|hold] [--hold-eps 0.1] <the t2v8f_run.py rollout arguments>

`base` patches NOTHING: the cell is exactly what `t2v8f_run.py` would run (the rehearsal asserts the records are bit-equal). `hold` rebinds `t2v8f_exec.PlaybackExecutor` to the
hold executor of `t2v8f_exec_hold.py` before the first rollout. The frozen card runs `base`; `hold` only ever runs under a later dated amendment.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def split_argv(argv):
    """-> (executor, eps, rest): the two switch flags are removed; everything else goes to `t2v8f_run.rollout_main` unchanged."""
    executor, eps, rest, i = "base", 0.1, [], 0
    argv = list(argv)
    while i < len(argv):
        a = argv[i]
        if a == "--executor":
            executor, i = argv[i + 1], i + 2
        elif a.startswith("--executor="):
            executor, i = a.split("=", 1)[1], i + 1
        elif a == "--hold-eps":
            eps, i = float(argv[i + 1]), i + 2
        elif a.startswith("--hold-eps="):
            eps, i = float(a.split("=", 1)[1]), i + 1
        else:
            rest.append(a)
            i += 1
    if executor not in ("base", "hold"):
        raise SystemExit(f"[t2v8f_stage2_cell] --executor must be base or hold, got {executor!r}")
    return executor, eps, rest


def apply_executor(executor, eps=0.1):
    """Bind the executor in the (cached) `t2v8f_run` state. Returns the active name. Safe to call per cell in one process: `base` restores the original class."""
    import t2v8f_exec_hold as H
    import t2v8f_run as R
    R.setup()
    X = R.STATE["X"]
    if executor == "hold":
        H.install(X, eps)
    else:
        H.uninstall(X)
    return H.active(X)


def run_cell(argv):
    executor, eps, rest = split_argv(argv)
    apply_executor(executor, eps)
    import t2v8f_run as R
    R.rollout_main(rest)
    return executor


if __name__ == "__main__":
    run_cell(sys.argv[1:])
