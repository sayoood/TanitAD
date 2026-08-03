"""Run a REAL trainer under the resource probe, unmodified.

The point is fidelity: a training benchmark that reimplements the step measures
the reimplementation. This wrapper imports ``thor_bench_probe`` (which starts a
sampling thread) and then executes the trainer script itself via ``runpy`` with
``__name__ == "__main__"``, so the measured code is byte-identical to what the
A40 pods run.

    THOR_BENCH_OUT=/tmp/b.jsonl python scripts/thor_bench_run.py \
        scripts/refc_train.py --data-root ... --out ... --steps 60

Everything after the target script path is passed through as its ``argv``.
"""

from __future__ import annotations

import os
import runpy
import sys

import thor_bench_probe  # noqa: F401  (import starts the sampler)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = sys.argv[1]
    if not os.path.isfile(target):
        print(f"[thor-bench] target trainer not found: {target}", file=sys.stderr)
        return 2
    sys.argv = [target, *sys.argv[2:]]
    print(f"[thor-bench] probe active -> {os.environ.get('THOR_BENCH_OUT', '(no out)')}",
          flush=True)
    print(f"[thor-bench] exec {target} {' '.join(sys.argv[1:])}", flush=True)
    runpy.run_path(target, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
