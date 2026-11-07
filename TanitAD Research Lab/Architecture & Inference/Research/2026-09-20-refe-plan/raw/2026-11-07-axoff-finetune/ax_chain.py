#!/usr/bin/env python3
"""YAX detached chain: SMOKE pipeline (tiny subset, mechanical) -> only if its marker is DONE -> FULL pipeline.  Markers (last action): markers/yax_chain.DONE | .FAILED (json).
    python ax_chain.py        (launch with launch_detached.ps1 / CIM CreateFlags 0x08000200)"""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
MD = HERE / "markers"
MD.mkdir(exist_ok=True)


def run(tag, extra):
    for m in ("DONE", "FAILED"):
        p = MD / f"yax_{tag}.{m}"
        if p.exists():
            os.remove(p)
    rc = subprocess.run([PY, str(HERE / "ax_pipeline.py"), "--root", str(HERE / f"run_{tag}")] + extra, stdin=subprocess.DEVNULL,
                        stdout=open(MD / f"yax_{tag}.out", "ab"), stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING="utf-8")).returncode
    return rc == 0 and (MD / f"yax_{tag}.DONE").exists()


def main():
    (MD / "yax_chain.STARTED").write_text(json.dumps({"pid": os.getpid(), "t": time.strftime("%Y-%m-%dT%H:%M:%S")}), encoding="utf-8")
    for m in ("DONE", "FAILED"):
        if (MD / f"yax_chain.{m}").exists():
            os.remove(MD / f"yax_chain.{m}")
    try:
        stage = "smoke"
        ok = run("smoke", ["--smoke"])
        if ok:
            stage = "full"
            ok = run("full", [])
        rec = {"state": "DONE" if ok else "FAILED", "failed_stage": None if ok else stage, "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    except Exception:
        ok, rec = False, {"state": "FAILED", "why": traceback.format_exc()[-2500:]}
    (MD / ("yax_chain.DONE" if ok else "yax_chain.FAILED")).write_text(json.dumps(rec, indent=1), encoding="utf-8")        # LAST ACTION
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
