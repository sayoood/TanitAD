#!/usr/bin/env python3
"""ONE detached chain for the scene-tap probe:  stage 1 = scene_ctx extraction (full, resumable; STOPS if it is not the same forward)  ->  stage 2 = the 35-unit grid.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I chain.py --updates 2000 --folds-sha256 <sha>      (launched detached; survives its launcher)

Markers (all in this folder): PROBE_RESULT.json then PROBE_DONE on success; PROBE_FAILED (stage, rc, the tail of the stage log) on any failure.  chain.log is the chain's own log; logs/ holds the stage logs.
A GPU-busy exit (rc 5) of stage 1 is retried (the shared dev-box lock); a NOT-THE-SAME-FORWARD exit (rc 6) is never retried.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
ART_WORK = "D:/Projects/TanitAD-artifacts/scene_tap_probe"
SCENE_DIR = "D:/Projects/TanitAD-artifacts/aux_navtest_forward/c3_full_scene_ctx"


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(HERE, "chain.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def tail(path, n=40):
    try:
        return "".join(open(path, encoding="utf-8", errors="replace").readlines()[-n:])
    except OSError:
        return ""


def fail(stage, rc, logpath, extra=""):
    body = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} stage={stage} rc={rc}\n{extra}\n--- tail of {logpath} ---\n{tail(logpath)}"
    with open(os.path.join(HERE, "PROBE_FAILED"), "w", encoding="utf-8") as f:
        f.write(body)
    log(f"PROBE_FAILED stage={stage} rc={rc}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--folds-sha256", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    os.makedirs(ART_WORK, exist_ok=True)
    for mk in ("PROBE_DONE", "PROBE_FAILED", "PROBE_RESULT.json"):
        p = os.path.join(HERE, mk)
        if os.path.exists(p):
            os.remove(p)
    try:
        log(f"chain start pid {os.getpid()} updates {a.updates}")
        # ---- stage 1: extraction -------------------------------------------------------------------------------------------------------------------------
        l1 = os.path.join(HERE, "logs", "extract_full.log")
        for attempt in range(1, 41):
            with open(l1, "a", encoding="utf-8") as f:
                f.write(f"\n===== attempt {attempt} {time.strftime('%Y-%m-%dT%H:%M:%S')} =====\n")
                f.flush()
                rc = subprocess.call([PY, "-I", os.path.join(HERE, "launch_extract.py")], stdout=f, stderr=subprocess.STDOUT)
            log(f"stage 1 attempt {attempt} rc {rc}")
            if rc == 0:
                break
            if rc == 5:
                time.sleep(120)
                continue
            extra = ""
            sf = os.path.join(SCENE_DIR, "SAME_FORWARD_FAILED.json")
            if rc == 6 and os.path.exists(sf):
                extra = "NOT THE SAME FORWARD (max |d| > 1e-4 vs the c3_full_ctx cache): " + open(sf, encoding="utf-8").read()
            return fail("extract", rc, l1, extra)
        else:
            return fail("extract", "GPU_BUSY_40_ATTEMPTS", l1)
        sm = os.path.join(SCENE_DIR, "extract_summary.json")
        summ = json.load(open(sm, encoding="utf-8")) if os.path.exists(sm) else {}
        if not (summ.get("tokens_extracted") == 1058 and summ.get("same_forward") and summ.get("tap_identity_all") and os.path.exists(os.path.join(SCENE_DIR, "scene_ctx_manifest.json"))):
            return fail("extract-verify", 0, l1, f"summary does not show a complete same-forward extraction: {json.dumps(summ)}")
        log(f"stage 1 OK: {json.dumps(summ)}")
        # ---- stage 2: the grid ---------------------------------------------------------------------------------------------------------------------------
        l2 = os.path.join(HERE, "logs", "grid.log")
        with open(l2, "a", encoding="utf-8") as f:
            rc = subprocess.call([PY, "-I", os.path.join(HERE, "scene_tap_probe.py"), "run", "--out", HERE, "--work", ART_WORK, "--updates", str(a.updates),
                                  "--folds-sha256", a.folds_sha256], stdout=f, stderr=subprocess.STDOUT)
        log(f"stage 2 rc {rc}")
        if rc != 0 or not os.path.exists(os.path.join(HERE, "PROBE_DONE")):
            if not os.path.exists(os.path.join(HERE, "PROBE_FAILED")):
                return fail("grid", rc, l2)
            return 1
        log("chain done")
        return 0
    except BaseException:                                                          # noqa: BLE001
        with open(os.path.join(HERE, "PROBE_FAILED"), "w", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} chain crashed\n{traceback.format_exc()}")
        log("PROBE_FAILED chain crashed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
