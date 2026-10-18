#!/usr/bin/env python3
"""RL-1 step 4b RELAUNCH (Master Mind, 2026-10-18 23:15). nt_run.py's call UNCHANGED except three guards:
  1. score_e2.py (cross-drive relpath fallback) instead of score_e.py;
  2. memory gate on '\\Memory\\Available MBytes' >= 6500 for 3 consecutive 60-s samples (W3's RAM guard aborts below
     3,000 MB; the first run died there because two other sessions' stack suites held 7.3 GB);
  3. before EACH attempt every *.csv already in the devkit run dir is moved aside, so the harness's glob fallback can
     never again pick up a partial reconstruction from an earlier attempt; and OK additionally requires 12,146 token rows.
The first run's files are in <work>/quarantine_ram_abort_20261018/ (not read).
    python nt_run2.py --variant V1 --work C:/Users/Admin/par18/rl/nav
"""
import argparse, csv, glob, json, os, subprocess, sys, time

SP = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/a085b71c-4fb3-409b-ae6b-65f605ed575e/scratchpad"
PY_D = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
RUNS = "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/runs"
N_TOK = 12146
ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="V1")
ap.add_argument("--work", required=True)
ap.add_argument("--min-avail-mb", type=int, default=6500)
a = ap.parse_args()
label = f"refe_rl1_{a.variant}"
seam = f"{a.work}/seam_{a.variant}.npz"
out = f"{a.work}/score"
logd = f"{a.work}/score_logs"
wrapper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_e2.py")
os.makedirs(logd, exist_ok=True)
os.makedirs(out, exist_ok=True)
trail = open(f"{logd}/{label}.nt_run2.log", "a", encoding="utf-8")


def note(msg):
    trail.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    trail.flush()


def avail_mb():
    # psutil, the probe W3's own RAM guard uses; the perf-counter path '\Memory\Available MBytes' does not exist on
    # this German-locale Windows (Get-Counter fails, which would have read 0 forever and never launched).
    import psutil
    return int(psutil.virtual_memory().available >> 20)


def wait_mem():
    ok, t0 = 0, time.time()
    while ok < 3:
        m = avail_mb()
        ok = ok + 1 if m >= a.min_avail_mb else 0
        if ok == 0 and int(time.time() - t0) % 1800 < 60:
            note(f"waiting for memory: available {m} MB < {a.min_avail_mb}")
        time.sleep(60)
    note(f"memory gate passed after {round((time.time() - t0) / 60)} min")


def token_rows(p):
    try:
        return sum(1 for r in csv.DictReader(open(p, encoding="utf-8")) if r["token"] != "average")
    except Exception:
        return -1


status, rows = {}, -1
for att in range(1, 4):
    wait_mem()
    for c in glob.glob(f"{RUNS}/{label}/*.csv"):
        dst = f"{c}.preattempt{att}_{time.strftime('%H%M%S')}.moved"
        os.replace(c, dst)
        note(f"moved stale csv aside: {c} -> {dst}")
    lp = f"{logd}/{label}.attempt{att}.log"
    t0 = time.time()
    note(f"attempt {att} start")
    subprocess.call([PY_D, wrapper, "--label", label, "--seam", seam, "--out", out], cwd=SP,
                    env=dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1"),
                    stdin=subprocess.DEVNULL, stdout=open(lp, "w"), stderr=subprocess.STDOUT)
    lines = open(lp, encoding="utf-8", errors="replace").read().splitlines()
    status = next((json.loads(l) for l in reversed(lines) if l.startswith("{") and '"status"' in l), {})
    status["attempt"], status["seconds"] = att, round(time.time() - t0)
    rows = token_rows(f"{out}/{label}/{label}.csv")
    status["nt_run2_token_rows"] = rows
    json.dump(status, open(f"{logd}/{label}.status.json", "w"), indent=1)
    note(f"attempt {att} end status={status.get('status')} rows={rows} secs={status['seconds']}")
    if status.get("status") == "PASS" and rows == N_TOK:
        break
    time.sleep(60)
ok = status.get("status") == "PASS" and rows == N_TOK
open(f"{a.work}/DONE_{a.variant}", "w").write(("OK" if ok else "FAIL") + "\n" + json.dumps(status) + "\n")
note("DONE " + ("OK" if ok else "FAIL"))
print("ZZNTRUN2", "OK" if ok else "FAIL", flush=True)
