"""Run ONE command while holding the dev-box GPU lock (`gpu_lock.py`), and always release it.

    python with_gpu_lock.py --job refcv7-g0-1500 --log <log> --rec <rec.json> [--max-wait-s N] -- <cmd ...>

The lock records THIS wrapper's pid (the job's parent). The child's own python pid is excluded from
the smi check only for the wait; while the child runs, the lock is what keeps other jobs out.
The record (`--rec`) says whether the lock was taken, how long it waited, the child's exit code and
wall time -- a caller asserts on the child's OUTPUT ARTIFACT, never on this exit code.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_lock  # noqa: E402


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: with_gpu_lock.py [opts] -- <cmd ...>")
    i = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--max-wait-s", type=int, default=6 * 3600)
    ap.add_argument("--child-timeout-s", type=int, default=0,
                    help="kill the CHILD (its own handle; nothing else) after N s; the lock is still "
                         "released by this wrapper. 0 = no timeout (the previous behaviour)")
    a = ap.parse_args(argv[:i])
    cmd = argv[i + 1:]
    rec = {"job": a.job, "cmd": cmd, "wrapper_pid": os.getpid(),
           "t_request": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(a.log, "a", encoding="utf-8") as lf:
        g = gpu_lock.acquire(a.job, os.getpid(), a.max_wait_s,
                             log=lambda *x, **k: print(*x, file=lf, flush=True))
    rec["lock"] = g
    if not g.get("ok"):
        rec["status"] = "LOCK NOT ACQUIRED"
        json.dump(rec, open(a.rec, "w", encoding="utf-8"), indent=1, default=str)
        return 3
    t0 = time.time()
    try:
        with open(a.log, "a", encoding="utf-8") as lf:
            try:
                p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                   timeout=(a.child_timeout_s or None))
                rec["child_exit"] = p.returncode
            except subprocess.TimeoutExpired:
                # subprocess.run has already killed THIS child by its own handle
                rec["child_exit"] = None
                rec["child_timed_out_s"] = a.child_timeout_s
    finally:
        rec["child_wall_s"] = round(time.time() - t0, 1)
        rec["release"] = gpu_lock.release(a.job, os.getpid())
        rec["t_done"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        rec["status"] = "RAN"
        json.dump(rec, open(a.rec, "w", encoding="utf-8"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
