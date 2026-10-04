"""D6 P1 + P2 GATE -- holds the GPU queue back until SPEC_P1P2_A1.md is registered.   tanitad venv, CPU only, no GPU, no lock.

Why: SPEC_P1P2_A1.md (the as-launched model has no WTA heads: universe = FAN117) must be registered BEFORE any P1/P2 number can exist. The GPU lock is held by the battery's
step-50,400 chain, so the queue would normally still be waiting for hours -- but "normally" is not a guarantee. This gate polls for raw/SPEC_A1_REGISTERED.txt, which must
CONTAIN the amendment's sha256 (written by the Master Mind after registering it), and only then starts the battery's ``with_gpu_lock.py`` for ``run_p1p2_chain.py``.
Until the file exists nothing touches the lock or the GPU.  Logs to raw/gate.log.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
WR = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/with_gpu_lock.py")
MARK = os.path.join(RAW, "SPEC_A1_REGISTERED.txt")
SHA = "def9b4c24222d47dc49de0174690279c4819a7da1dc896954d7c6b582f1392bb"


def log(m):
    with open(os.path.join(RAW, "gate.log"), "a", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + m + "\n")


def main() -> int:
    log("gate up; waiting for " + MARK)
    while True:
        if os.path.exists(MARK):
            try:
                txt = open(MARK, encoding="utf-8", errors="replace").read()
            except OSError:
                txt = ""
            if SHA in txt:
                break
            log("marker exists but does not contain the A1 sha256 -> still waiting")
        time.sleep(30)
    log("A1 registered marker found -> starting the GPU queue (with_gpu_lock.py waits for the lock; it never touches a held one)")
    cmd = [PY, WR, "--job", "refcv8-d6-p1p2", "--log", os.path.join(RAW, "gpu_wait.log"), "--rec", os.path.join(RAW, "gpu_rec.json"),
           "--max-wait-s", "43200", "--child-timeout-s", "10800", "--", PY, os.path.join(HERE, "run_p1p2_chain.py")]
    p = subprocess.Popen(cmd)
    log(f"with_gpu_lock launched pid={p.pid}")
    rc = p.wait()
    log(f"with_gpu_lock ended rc={rc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
