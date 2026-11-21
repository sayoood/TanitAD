#!/usr/bin/env python3
"""ONE detached chain for Amendment A2 (2026-10-16): the 10,000-update V seed-0 arm, then its reader.

    stage 1  scene_tap_probe.py run --arms V --seeds 0 --updates 10000 --work <A2 work dir>     (5 units; resumable; shared dev-box GPU lock, one lease per unit)
    stage 2  a2_read.py                                                                        (identity test + zero control FIRST, then H / I / Delta and exactly one of U1..U4)

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -B -I chain_a2.py          (launched detached by launch_a2.ps1; survives its launcher)

Markers (this folder): A2_DONE (json: the row) | A2_FAILED (stage, rc, the tail of the stage log).  A2_RESULT.json / A2_RESULT.md hold the read.  Logs: chain_a2.log (this chain), probe.log (the grid's own),
logs/grid.log, logs/a2_read.log.  Re-running the same command RESUMES (finished units are skipped; a unit of another config is refused, never mixed).  A second chain while one is alive exits at once.
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
A2_WORK = "D:/Projects/TanitAD-artifacts/scene_tap_probe_a2"
FOLDS_SHA256 = "20dec3431663de2201190020f97c74f7975468a8e9e178f3db4d52fbacd5885f"
EXPECT_UNITS = [f"V_s0_f{f}" for f in range(5)]
LOCK = os.path.join(HERE, "chain_a2.lock")


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(HERE, "chain_a2.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def tail(path, n=40):
    try:
        return "".join(open(path, encoding="utf-8", errors="replace").readlines()[-n:])
    except OSError:
        return ""


def pid_alive(pid) -> bool:
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True)
    return f'"{pid}"' in r.stdout.decode("utf-8", "replace")


def take_lock() -> bool:
    for _ in (1, 2):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": os.getpid(), "t": time.time()}).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                holder = json.loads(open(LOCK, encoding="utf-8").read())
            except (OSError, ValueError):
                holder = {}
            if holder.get("pid") and pid_alive(holder["pid"]):
                log(f"another A2 chain is alive (pid {holder['pid']}); exiting")
                return False
            os.remove(LOCK)                                                           # a dead chain's lock
    return False


def fail(stage, rc, logpath, extra=""):
    body = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} stage={stage} rc={rc}\n{extra}\n--- tail of {logpath} ---\n{tail(logpath)}"
    with open(os.path.join(HERE, "A2_FAILED"), "w", encoding="utf-8") as f:
        f.write(body)
    log(f"A2_FAILED stage={stage} rc={rc}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=10000)
    ap.add_argument("--lease-s", type=float, default=600.0, help="re-take the GPU lock before every unit (a unit is ~20 min, so one lease = one unit)")
    ap.add_argument("--work", default=A2_WORK)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    os.makedirs(a.work, exist_ok=True)
    if not take_lock():
        return 0
    try:
        for mk in ("A2_DONE", "A2_FAILED"):
            p = os.path.join(HERE, mk)
            if os.path.exists(p):
                os.remove(p)
        log(f"chain start pid {os.getpid()} updates {a.updates} work {a.work}")
        # ---- stage 1: the grid, filtered to V seed 0 ---------------------------------------------------------------------------------------------------
        l1 = os.path.join(HERE, "logs", "grid.log")
        with open(l1, "a", encoding="utf-8") as f:
            rc = subprocess.call([PY, "-B", "-I", os.path.join(HERE, "scene_tap_probe.py"), "run", "--out", HERE, "--work", a.work, "--updates", str(a.updates),
                                  "--folds-sha256", FOLDS_SHA256, "--arms", "V", "--seeds", "0", "--lease-s", str(a.lease_s)], stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
        log(f"stage 1 (grid) rc {rc}")
        gd = os.path.join(HERE, "GRID_FILTERED_DONE")
        if rc != 0 or not os.path.exists(gd):
            return fail("grid", rc, l1, tail(os.path.join(HERE, "PROBE_FAILED"), 30))
        done = json.load(open(gd, encoding="utf-8"))
        if done.get("units") != EXPECT_UNITS or int(done.get("n_updates", -1)) != a.updates:
            return fail("grid-verify", 0, l1, f"GRID_FILTERED_DONE is not the 5 V-s0 units at {a.updates} updates: {done}")
        # ---- stage 2: the reader -------------------------------------------------------------------------------------------------------------------------
        l2 = os.path.join(HERE, "logs", "a2_read.log")
        with open(l2, "a", encoding="utf-8") as f:
            rc = subprocess.call([PY, "-B", "-I", os.path.join(HERE, "a2_read.py"), "--out", HERE, "--work-a2", a.work, "--updates-a2", str(a.updates)],
                                 stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
        log(f"stage 2 (a2_read) rc {rc}")
        rp = os.path.join(HERE, "A2_RESULT.json")
        if rc != 0 or not os.path.exists(rp):
            return fail("a2_read", rc, l2, tail(os.path.join(HERE, "a2_read.error"), 30) + ("  [rc 3 = INSTRUMENT FAILED: identity test / zero control]" if rc == 3 else ""))
        r = json.load(open(rp, encoding="utf-8"))
        dec = r["decision"]
        if r.get("mode") != "full" or dec["row"] not in ("U1", "U2", "U3", "U4"):
            return fail("a2_read-verify", 0, l2, f"A2_RESULT.json is not a full read with a row: {json.dumps(dec)[:400]}")
        with open(os.path.join(HERE, "A2_DONE"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"row": dec["row"], "H": dec["H"], "I": dec["I"], "delta": dec["delta_point"], "delta_separated_positive": dec["delta_separated_positive"],
                                "row_depends_on_the_ci_clause": r["row_depends_on_the_ci_clause"], "at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + chr(10))
        log(f"A2_DONE {dec['row']} H {dec['H']!r} I {dec['I']!r} delta {dec['delta_point']!r}")
        return 0
    except BaseException:                                                              # noqa: BLE001  (the marker must exist whatever killed the chain)
        with open(os.path.join(HERE, "A2_FAILED"), "w", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} chain crashed" + chr(10) + traceback.format_exc())
        log("A2_FAILED chain crashed")
        return 1
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
