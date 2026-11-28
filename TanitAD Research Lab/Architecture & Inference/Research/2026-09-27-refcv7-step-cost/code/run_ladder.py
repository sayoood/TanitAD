#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a plan of step_profiler.py jobs one at a time, each gated on:

  * the GPU being free of every OTHER compute process: no `python*` process other than this
    runner's own child in `nvidia-smi --query-compute-apps` (on this Windows/WDDM box the list
    always carries desktop apps -- dwm, chrome, ... -- which hold a display context, not a
    compute job; they are named in the record, never ignored silently);
  * host RAM: >= --min-free-gb + --need-gb free before a start, and a watchdog that kills
    OUR child (only ever our own PID tree) if free RAM falls below --min-free-gb.

Usage: python run_ladder.py --plan plan.json --out-root <raw dir> [--dry]
The plan is a JSON list of {"name", "args": [...]} (args passed to step_profiler.py).
Each job's verdict is its profile_*.json, never this runner's exit code.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"


def free_ram_gb() -> float:
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30, m.ullAvailPageFile / 2 ** 30


def gpu_compute_pythons(own: set) -> list:
    """(pid, name) of python processes holding a GPU context, excluding `own`."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
    except Exception as e:                               # noqa: BLE001
        return [(-1, f"nvidia-smi failed: {e!r}")]
    if r.returncode != 0:
        return [(-1, f"nvidia-smi rc {r.returncode}")]
    out = []
    for line in r.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if pid in own:
            continue
        name = proc_name(pid)
        if name.startswith("python") or pid == 38376 or pid == 37904:
            out.append((pid, name))
    return out


def child_pids(root: int) -> set:
    import psutil
    try:
        pr = psutil.Process(root)
        return {root} | {c.pid for c in pr.children(recursive=True)}
    except Exception:                                    # noqa: BLE001
        return {root}


def own_profilers() -> list:
    """PIDs of step_profiler.py jobs already running (ours, from any ladder): jobs never overlap,
    so a CPU job cannot perturb a GPU job's timing and two jobs never share the RAM budget."""
    import psutil
    out = []
    # ⛔ match the ARGV ELEMENT of a python process, never a substring of a whole command line:
    # the bash wrapper that launched a ladder carries the script text and matched itself
    # (MEASURED 2026-09-28 -- the pgrep -f self-match trap, CLAUDE.md).
    for pr in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            nm = (pr.info.get("name") or "").lower()
            av = list(pr.info.get("cmdline") or [])
        except Exception:                                # noqa: BLE001
            continue
        if nm.startswith("python") and any(str(x).replace("\\", "/").endswith("/step_profiler.py")
                                           for x in av[1:3]):
            out.append(pr.info["pid"])
    return out


def gpu_ladder_alive() -> bool:
    import psutil
    for pr in psutil.process_iter(["name", "cmdline"]):
        try:
            nm = (pr.info.get("name") or "").lower()
            av = [str(x) for x in (pr.info.get("cmdline") or [])]
        except Exception:                                # noqa: BLE001
            continue
        if nm.startswith("python") and any(x.endswith("run_ladder.py") for x in av[1:3])                 and "plan_gpu.json" in av:
            return True
    return False


def proc_name(pid: int) -> str:
    import psutil
    try:
        return psutil.Process(pid).name().lower()
    except Exception:                                    # noqa: BLE001
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tree", default="C:/lgt/rc7cost")
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--need-gb", type=float, default=6.0)
    ap.add_argument("--timeout-s", type=int, default=2400)
    ap.add_argument("--gpu-wait-s", type=int, default=6 * 3600)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of job names")
    ap.add_argument("--max-attempts", type=int, default=6,
                    help="re-gate and retry a job killed by the RAM watchdog")
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    root = Path(a.out_root)
    root.mkdir(parents=True, exist_ok=True)
    log = open(root / "ladder_log.jsonl", "a", encoding="utf-8")
    env = dict(os.environ)
    t = a.tree
    env.update(PYTHONPATH=f"{t}/stack;{t}/stack/scripts;{t}/taniteval",
               PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    if a.cpu_only:
        # ⛔ NOT "": Windows DROPS an env var set to the empty string, so the child would see
        # every GPU (MEASURED 2026-09-27 on the first smoke). -1 hides them; the profiler also
        # gets --require-cpu, which asserts device_count() == 0 BEFORE anything else runs.
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    only = set(a.only.split(",")) if a.only else None
    for job in plan:
        name = job["name"]
        if only and name not in only:
            continue
        jd = root / name
        if (jd / "DONE").exists():
            continue
        jd.mkdir(parents=True, exist_ok=True)
        for attempt in range(a.max_attempts):
            v = run_one(a, job, jd, env, log, attempt)
            if v != "KILLED_LOW_RAM":
                break
        if v == "GATE_TIMEOUT":
            return 3
    return 0


def run_one(a, job, jd, env, log, attempt):
    name = job["name"]
    if True:
        # ---- gates ----
        t0 = time.time()
        while True:
            fr, fc = free_ram_gb()
            busy = [] if a.cpu_only else gpu_compute_pythons(set())
            mine = own_profilers()
            if a.cpu_only and gpu_ladder_alive() and not gpu_compute_pythons(set()):
                mine = mine + ["yield-to-gpu-ladder"]     # the GPU is free: GPU jobs go first
            if not busy and not mine and fr >= a.min_free_gb + a.need_gb:
                break
            if time.time() - t0 > a.gpu_wait_s:
                log.write(json.dumps({"job": name, "verdict": "GATE_TIMEOUT", "busy": busy,
                                      "free_gb": round(fr, 2)}) + "\n")
                log.flush()
                return "GATE_TIMEOUT"
            print(f"[ladder] {name}: waiting (busy={busy}, own_running={len(mine)}, "
                  f"free={fr:.1f} GB)", flush=True)
            time.sleep(60)
        cmd = [PY, str(HERE / "step_profiler.py"), "--tree", a.tree, "--out", str(jd),
               *job["args"]] + (["--require-cpu"] if a.cpu_only else [])
        fr0, _ = free_ram_gb()
        rec = {"job": name, "cmd": cmd[1:], "t_start": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "free_gb_at_start": round(fr0, 2), "attempt": attempt}
        with open(jd / "log.txt", "w", encoding="utf-8") as fh:
            p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL)
            min_free, verdict = fr0, None
            tstart = time.time()
            peak_rss = 0.0
            while p.poll() is None:
                time.sleep(5)
                fr, _ = free_ram_gb()
                min_free = min(min_free, fr)
                try:
                    import psutil
                    rss = 0
                    for cp in child_pids(p.pid):
                        try:
                            rss += psutil.Process(cp).memory_info().rss
                        except Exception:                # noqa: BLE001
                            pass
                    peak_rss = max(peak_rss, rss / 2 ** 30)
                except Exception:                        # noqa: BLE001
                    pass
                if fr < a.min_free_gb:
                    for pid in sorted(child_pids(p.pid), reverse=True):
                        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                                       capture_output=True)
                    verdict = "KILLED_LOW_RAM"
                    break
                if time.time() - tstart > a.timeout_s:
                    for pid in sorted(child_pids(p.pid), reverse=True):
                        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                                       capture_output=True)
                    verdict = "KILLED_TIMEOUT"
                    break
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                pass
        rec.update(rc=p.returncode, verdict=verdict or "EXITED", min_free_gb=round(min_free, 2),
                   peak_rss_gb_own_tree=round(peak_rss, 2),
                   wall_s=round(time.time() - tstart, 1),
                   artifacts=sorted(x.name for x in jd.iterdir()))
        prof_err = "no profile json"
        for pj in jd.glob("profile_*.json"):
            try:
                prof_err = json.loads(pj.read_text(encoding="utf-8")).get("error")
            except Exception as e:                       # noqa: BLE001
                prof_err = f"unreadable: {e!r}"
        rec["profile_error"] = prof_err
        if verdict is None and prof_err is None:
            (jd / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
        log.write(json.dumps(rec) + chr(10))
        log.flush()
        msg = (f"[ladder] {name}: {rec['verdict']} rc={p.returncode} {rec['wall_s']} s "
               f"err={prof_err}")
        print(msg.encode("ascii", "backslashreplace").decode("ascii"), flush=True)
        return rec["verdict"]


if __name__ == "__main__":
    raise SystemExit(main())
