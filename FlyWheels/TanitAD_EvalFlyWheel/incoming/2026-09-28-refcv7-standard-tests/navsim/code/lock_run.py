#!/usr/bin/env python3
"""Run ONE command holding the dev-box GPU lock (TANITAD VENV).

    python code/lock_run.py --job refcv7-navsim-X --wait-s 36000 --log <file> -- <cmd ...>

Acquires ``gpu_lock`` (lock ABSENT and no other python compute app), substitutes the literal
argument ``@TOKEN@`` in the command with the token, runs it, and ALWAYS releases the lock (only if
it still carries our token). Writes ``<log>.lockrun.json`` (token, times, rc) -- the artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpu_lock  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--wait-s", type=float, default=0.0)
    ap.add_argument("--log", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    rec = {"job": a.job, "cmd": cmd, "t_request": time.time()}

    def lg(m):
        with open(a.log, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {m}\n")
    tok = gpu_lock.acquire(a.job, a.wait_s, log=lg)
    rec["token"] = tok
    rec["t_acquired"] = time.time() if tok else None
    if not tok:
        lg(f"LOCK NOT ACQUIRED after {a.wait_s} s")
        rec["rc"] = None
        json.dump(rec, open(a.log + ".lockrun.json", "w", encoding="utf-8"), indent=1)
        return 3
    lg(f"LOCK ACQUIRED {tok}")
    try:
        cmd = [tok if c == "@TOKEN@" else c for c in cmd]
        with open(a.log, "a", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    finally:
        rel = gpu_lock.release(tok)
        lg(f"LOCK RELEASED={rel}")
    rec.update({"rc": rc, "t_done": time.time(), "released": rel})
    json.dump(rec, open(a.log + ".lockrun.json", "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
