#!/usr/bin/env python
"""Watcher for the detached H-HJ-COLLAPSE-1 job (poll by SCRIPT, not by model turns).

  * every 120 s: count finished episodes (ep/*.json with "done": true);
  * if the diag process is gone and fewer than 40 episodes are done -> relaunch it (resumable,
    max 3 relaunches; the 'window-CLOSE' console event killed one launch during development);
  * when 40/40 are done -> run the read-out and render the numbers, then exit.
Everything is logged to <work>/post_run.log.  Idempotent: safe to run twice.
"""
import glob
import json
import os
import subprocess
import sys
import time

WORK = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/hjcollapse_run1"
PKG = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-hjepa-collapse-diag"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
N_EXPECT = 40
LOG = os.path.join(WORK, "post_run.log")


def log(m):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%F %T')} {m}\n")


def n_done():
    n = 0
    for jp in glob.glob(os.path.join(WORK, "ep", "*.json")):
        try:
            if json.load(open(jp)).get("done"):
                n += 1
        except Exception:
            pass
    return n


def diag_running():
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -like '*hj_collapse_diag*' } | Measure-Object).Count"],
        capture_output=True, text=True)
    try:
        return int(r.stdout.strip() or "0") > 0
    except ValueError:
        return True          # unknown -> do not relaunch


def relaunch():
    r = subprocess.run(
        ["powershell", "-NoProfile", "-File", f"{PKG}/code/launch_detached.ps1",
         "-Work", WORK.replace("/", "\\"), "-Tag", "run_relaunch"],
        capture_output=True, text=True)
    log(f"relaunch: {r.stdout.strip()} {r.stderr.strip()[:200]}")


def main():
    log(f"watcher start work={WORK}")
    relaunches, t0 = 0, time.time()
    while time.time() - t0 < 8 * 3600:
        n = n_done()
        if n >= N_EXPECT:
            break
        if not diag_running():
            time.sleep(20)                       # let a finishing process exit cleanly
            if not diag_running() and n_done() < N_EXPECT:
                if relaunches >= 3:
                    log("diag gone, 3 relaunches used -> giving up (needs an agent)")
                    return
                relaunches += 1
                relaunch()
        log(f"episodes done: {n}/{N_EXPECT}")
        time.sleep(120)
    n = n_done()
    log(f"finished episodes: {n}/{N_EXPECT}")
    if n < 2:
        return
    env = dict(os.environ, PYTHONPATH="E:/Projects/TanitAD/stack;E:/Projects/TanitAD/taniteval",
               PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="6")
    out = f"{PKG}/raw"
    r = subprocess.run([PY, f"{PKG}/code/hj_collapse_readout.py", "--work", WORK, "--out", out],
                       capture_output=True, text=True, env=env)
    log(f"readout rc={r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}")
    if r.returncode == 0:
        r2 = subprocess.run([PY, f"{PKG}/code/render_numbers.py", f"{out}/readout.json"],
                            capture_output=True, text=True, env=env)
        log(f"render rc={r2.returncode} {r2.stdout.strip()} {r2.stderr[-500:]}")
    log("POST-RUN-DONE")


if __name__ == "__main__":
    main()
