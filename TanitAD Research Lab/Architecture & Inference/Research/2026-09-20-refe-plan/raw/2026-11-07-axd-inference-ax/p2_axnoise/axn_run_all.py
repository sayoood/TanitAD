#!/usr/bin/env python3
"""YAXN detached entry point: SMOKE (tiny, CPU seams, no GPU lock, RED seams) and, only if markers/axn_smoke.DONE exists, the FULL chain (GPU + the shared lock, queues, never forces).
Launch (inline PowerShell, WMI Win32_Process Create, CreateFlags 0x08000200, stdin NUL): see HANDOFF_P2.md."""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
(HERE / "markers").mkdir(exist_ok=True)
rc = 0
if not (HERE / "markers" / "axn_smoke.DONE").exists():
    rc = subprocess.call([PY, str(HERE / "axn_chain.py"), "--tag", "smoke"], stdout=open(HERE / "markers" / "axn_smoke.out", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8"))
if not (HERE / "markers" / "axn_smoke.DONE").exists():
    sys.exit(f"smoke did not finish DONE (rc {rc}); the FULL chain is NOT started")
sys.exit(subprocess.call([PY, str(HERE / "axn_chain.py"), "--tag", "full"], stdout=open(HERE / "markers" / "axn_full.out", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         env=dict(os.environ, PYTHONIOENCODING="utf-8")))
