"""exit 0 when no other agent's GPU job (LONHEAD l2_roll / lonhead) is alive, else print the blockers and exit 1.  NEW (EvalFlyWheel 2026-10-26)."""
import os
import sys

import psutil

me = {os.getpid(), os.getppid()}
busy = []
for p in psutil.process_iter(["pid", "cmdline"]):
    try:
        c = " ".join(p.info["cmdline"] or [])
    except Exception:  # noqa: BLE001
        continue
    if p.info["pid"] in me or "gpu_free.py" in c:
        continue
    lc = c.lower()
    if "python" in lc and ("l2_roll.py" in lc or "lonhead" in lc or "ztac_roll" in lc or "roll_seed_r8" in lc or "run_battery" in lc):
        busy.append(f"{p.info['pid']}:{c[:140]}")
print("BUSY " + " | ".join(busy) if busy else "FREE")
sys.exit(1 if busy else 0)
