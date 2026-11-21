#!/usr/bin/env python3
"""Zero-token watcher: waits for P2HO_DONE or P2HO_FAILED in the package, then rewrites paths.json (so it lists the final small outputs) and writes FINALIZED.txt.
Launched detached once; it only reads and writes small files in the package."""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
deadline = time.time() + 24 * 3600
while time.time() < deadline:
    done, failed = os.path.exists(os.path.join(ROOT, "P2HO_DONE")), os.path.exists(os.path.join(ROOT, "P2HO_FAILED"))
    if done or failed:
        time.sleep(20)
        subprocess.call([sys.executable, os.path.join(ROOT, "code", "make_paths.py")])
        open(os.path.join(ROOT, "FINALIZED.txt"), "w").write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} marker={'P2HO_DONE' if done else 'P2HO_FAILED'}; paths.json rewritten\n")
        break
    time.sleep(30)
