"""Memory-bounded suite runner: ONE pytest process per test file, gated on free host RAM.

    python run_suite_bounded.py <tree> <files.txt> <out.jsonl> [--rundir stack|taniteval]

* The dev box is the PI's desktop and other sessions' jobs use most of its RAM, so each file
  runs in its own process (memory is returned between files) and only starts when free RAM is
  >= START_GB. While it runs, free RAM is polled; if it falls below KILL_GB the child (ours,
  by PID) is killed, the file is marked RAM_ABORT and retried later (at most RETRIES times).
* One JSON line per file: counts parsed from pytest's summary and the FAILED/ERROR node ids
  from ``-rfE``. Resumable: files already in <out.jsonl> with a final status are skipped.
* PYTHONPATH is WINDOWS-style (C:/...), and the first line of every run asserts
  ``tanitad.__file__`` resolves inside <tree> -- an MSYS path is silently dropped and the venv's
  editable install then serves tanitad from G: (the false-pass trap).
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
# FLOOR, CPU-ONLY CHAIN (Master Mind, 2026-09-26 ~22:30 Berlin): start only after THREE
# consecutive samples 30 s apart read >= 7.5 GB available; kill our child below 6.5 GB.
# Reason: the box sat below the brief's 8 GB floor for > 1 h because of other sessions' jobs;
# E1's NavSim scorer aborts only after 120 s below 3 GB (run_navsim_refcv6.py:317), so 6.5 GB
# keeps a 3.5 GB margin. The 8 GB rule stays in force for GPU jobs.
START_GB, KILL_GB, RETRIES, FILE_TIMEOUT_S = 7.5, 6.5, 3, 1800
START_SAMPLES, START_GAP_S = 3, 30.0
MAX_WAIT_S = 6 * 3600
# UNIT: "GB" in this file is GiB -- ullAvailPhys / 2**30 (free_gb below), the same unit the
# programme's other RAM guards print as "GB" (run_navsim_refcv6.py: psutil ... / 2**30).
# 7.5 GiB = 8.05 decimal GB: a decimal probe reads the floor ~7 % LOW.
# ⛔ When the floor is not met within MAX_WAIT_S the file is NOT launched (RAM_WAIT_TIMEOUT):
# the first version fell out of the wait loop and started anyway -- a floor that a timer
# silently turns off is not a floor.
FLOOR_NOTE = ("floor: start >= 7.5 GiB on 3 consecutive samples 30 s apart, kill < 6.5 GiB "
              "(Master Mind 2026-09-26: box below 8 GB for > 1 h from other sessions; the "
              "NavSim scorer aborts only after 120 s < 3 GB, run_navsim_refcv6.py:317; the "
              "8 GB rule is kept for GPU jobs); OMP_NUM_THREADS=4; one pytest file at a time")


def free_gb() -> float:
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


def parse(text: str) -> dict:
    counts = {}
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    for n, k in re.findall(r"(\d+) (passed|failed|skipped|errors?|xfailed|xpassed|deselected)",
                           tail):
        counts["error" if k.startswith("error") else k] = int(n)
    failed = re.findall(r"^FAILED (\S+)", text, re.M)
    errors = re.findall(r"^ERROR (\S+)", text, re.M)
    return {"counts": counts, "failed": failed, "errors": errors, "tail": tail[-300:]}


def run_one(tree: Path, rundir: str, rel: str, env: dict, logdir: Path) -> dict:
    stem = rel.replace("/", "__").replace("\\", "__")
    log = logdir / (stem + ".log")
    junit = logdir / (stem + ".junit.xml")
    cmd = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfE",
           f"--junitxml={junit}", rel]
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.Popen(cmd, cwd=str(tree / rundir), env=env, stdout=fh,
                             stderr=subprocess.STDOUT)
        status, min_free = None, free_gb()
        while p.poll() is None:
            time.sleep(2.0)
            f = free_gb()
            min_free = min(min_free, f)
            if f < KILL_GB:
                p.kill()
                p.wait()
                status = "RAM_ABORT"
                break
            if time.time() - t0 > FILE_TIMEOUT_S:
                p.kill()
                p.wait()
                status = "TIMEOUT"
                break
    text = log.read_text(encoding="utf-8", errors="replace")
    res = {"file": rel, "rc": p.returncode, "seconds": round(time.time() - t0, 1),
           "min_free_gb": round(min_free, 2), **parse(text)}
    res["status"] = status or ("OK" if p.returncode in (0, 1, 5) else f"RC{p.returncode}")
    return res


def main() -> None:
    tree, files_txt, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
    rundir = sys.argv[sys.argv.index("--rundir") + 1] if "--rundir" in sys.argv else "stack"
    files = [x for x in files_txt.read_text(encoding="utf-8").split() if x.strip()]
    logdir = out.parent / (out.stem + "_logs")
    logdir.mkdir(parents=True, exist_ok=True)
    tp = str(tree).replace("\\", "/")
    env = dict(os.environ, PYTHONPATH=f"{tp}/stack;{tp}/stack/scripts;{tp}/taniteval",
               OMP_NUM_THREADS="4", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1",
               CUDA_VISIBLE_DEVICES="")
    chk = subprocess.run([PY, "-c", "import tanitad; print(tanitad.__file__)"], env=env,
                         capture_output=True, text=True, cwd=str(tree / rundir))
    where = chk.stdout.strip()
    if not where.replace("\\", "/").lower().startswith(tp.lower()):
        raise SystemExit(f"tanitad resolves to {where!r}, NOT inside {tp}")
    chk2 = subprocess.run([PY, "-c", "import taniteval.nav_compliance as m; print(m.__file__)"],
                          env=env, capture_output=True, text=True, cwd=str(tree / rundir))
    te_where = chk2.stdout.strip()
    if not te_where.replace("\\", "/").lower().startswith(tp.lower()):
        raise SystemExit(f"taniteval resolves to {te_where!r}, NOT inside {tp}")
    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("status") in ("OK", "TIMEOUT") or str(r.get("status", "")).startswith("RC"):
                done[r["file"]] = r
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"file": "__header__", "tree": tp, "tanitad": where,
                             "taniteval": te_where, "floor": FLOOR_NOTE,
                             "t": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        fh.flush()
        for rel in files:
            if rel in done:
                continue
            for attempt in range(RETRIES + 1):
                w0, ok = time.time(), 0
                while ok < START_SAMPLES and time.time() - w0 < MAX_WAIT_S:
                    ok = ok + 1 if free_gb() >= START_GB else 0
                    if ok < START_SAMPLES:
                        time.sleep(START_GAP_S)
                if ok < START_SAMPLES:       # budget spent: NEVER start into the squeeze
                    res = {"file": rel, "status": "RAM_WAIT_TIMEOUT", "attempt": attempt,
                           "rc": None, "seconds": 0.0, "min_free_gb": round(free_gb(), 2),
                           "counts": {}, "failed": [], "errors": [],
                           "tail": f"floor not met within {MAX_WAIT_S // 3600} h: NOT launched"}
                    fh.write(json.dumps(res) + "\n")
                    fh.flush()
                    print(f"[{time.strftime('%H:%M:%S')}] RAM_WAIT_TIMEOUT {rel} (not launched)",
                          flush=True)
                    break
                res = run_one(tree, rundir, rel, env, logdir)
                res["attempt"] = attempt
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                print(f"[{time.strftime('%H:%M:%S')}] {res['status']:9s} {rel} "
                      f"{res['counts']} free>={res['min_free_gb']}GB {res['seconds']}s",
                      flush=True)
                if res["status"] != "RAM_ABORT":
                    break
    print("ZZSUITEDONEZZ", flush=True)


if __name__ == "__main__":
    main()
