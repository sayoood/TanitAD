#!/usr/bin/env python3
"""YAXD detached entry point: SMOKE on CPU without the GPU lock (8-token log, 3 CRUISE, + 4 RED seams on the same log, no YAX wait) and, only if markers/axd_smoke.DONE, the FULL chain (waits for YAX's marker)."""
import subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
PY = sys.executable
T = "C:/Users/Admin/yaxd_run/tokens"
rc = 0
if not (HERE / "markers" / "axd_smoke.DONE").exists():   # CPU smoke (mechanics only): the GPU lock's idle bar (3,500 MiB) was not met by the desktop's 3.6-3.7 GB at 09:50
    rc = subprocess.call([PY, str(HERE / "axd_chain.py"), "--tag", "smoke", "--tokens", T + "/tokens_red.json", "--red-tokens", T + "/tokens_red.json", "--no-wait", "--device", "cpu", "--no-lock"],
                         stdout=open(HERE / "markers" / "axd_smoke.out", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=dict(__import__("os").environ, CUDA_VISIBLE_DEVICES="-1"))
if not (HERE / "markers" / "axd_smoke.DONE").exists():
    sys.exit(f"smoke did not finish DONE (rc {rc}); the FULL chain is NOT started")
sys.exit(subprocess.call([PY, str(HERE / "axd_chain.py"), "--tag", "full"], stdout=open(HERE / "markers" / "axd_full.out", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL))
