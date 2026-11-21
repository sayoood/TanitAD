#!/usr/bin/env python3
"""Keep ``LANDING_READY_NAVSIM50K_LEGAL.txt`` current while step-50,400 arms complete (any python, no GPU/RAM load).

    python code/landing_refresher7.py [--every-s 600]

Re-runs ``landing_navsim50k_legal7.py`` every ``--every-s`` seconds and exits after the final pass that follows BOTH
finisher markers (``ZZFINISH50400DONEZZ`` in step50400/finish.log and ``ZZFINISHVMAXOFFDONEZZ`` in
step50400/vmaxoff_legal/finish_vmaxoff.log). Reads artifacts only; never runs git.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MS = os.path.join(os.path.dirname(HERE), "raw", "milestones", "step50400")


def has(path: str, marker: str) -> bool:
    try:
        return marker in open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every-s", type=float, default=600.0)
    a = ap.parse_args()
    while True:
        done = (has(os.path.join(MS, "finish.log"), "ZZFINISH50400DONEZZ")
                and has(os.path.join(MS, "vmaxoff_legal", "finish_vmaxoff.log"), "ZZFINISHVMAXOFFDONEZZ"))
        subprocess.run([sys.executable, os.path.join(HERE, "landing_navsim50k_legal7.py")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if done:
            return 0
        time.sleep(a.every_s)


if __name__ == "__main__":
    sys.exit(main())
