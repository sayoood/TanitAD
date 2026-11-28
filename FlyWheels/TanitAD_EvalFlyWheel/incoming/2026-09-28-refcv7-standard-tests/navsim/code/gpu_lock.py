#!/usr/bin/env python3
"""The dev-box GPU: ONE job at a time (the brief). Exclusive lock file + an nvidia-smi gate.

    python code/gpu_lock.py acquire --job refcv7-navsim-warmup [--wait-s 36000]  -> prints a TOKEN
    python code/gpu_lock.py release --token <TOKEN>
    python code/gpu_lock.py status

The lock is ``C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock``, created with ``O_CREAT|O_EXCL``
(atomic: two jobs can never both create it) and holding ``job``, ``pid``, ``token``, ``utc``. A
job may use CUDA only while the file exists AND names its token (``held_by``), and only when
``nvidia-smi --query-compute-apps`` shows no OTHER python compute app. Release removes the file
only if it still carries the caller's token (never another job's lock).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid

LOCK = os.environ.get("R7_GPU_LOCK", "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock")


def read() -> dict | None:
    try:
        with open(LOCK, encoding="utf-8") as fh:
            txt = fh.read()
    except FileNotFoundError:
        return None
    except OSError as e:
        return {"unreadable": repr(e)}
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {"raw": txt[:500]}


def other_python_compute(own_pids=()) -> list:
    """``nvidia-smi`` compute apps whose process name contains 'python', minus ``own_pids``.
    ⚠️ Raises when nvidia-smi fails -- an unreadable card is never read as a free card."""
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed rc={r.returncode}: {r.stderr.strip()[:200]}")
    out = []
    for ln in r.stdout.strip().splitlines():
        parts = [p.strip() for p in ln.split(",", 1)]
        if len(parts) == 2 and "python" in parts[1].lower():
            try:
                pid = int(parts[0])
            except ValueError:
                pid = -1
            if pid not in set(int(p) for p in own_pids):
                out.append({"pid": pid, "name": parts[1][-120:]})
    return out


def reap_own_stale(log=print) -> bool:
    """Remove a lock that THIS suite left behind: job ``refcv7-navsim*`` AND its pid is DEAD.
    ⚠️ MEASURED 2026-09-28: a runner killed by PID seconds after acquiring left its lock for 42 min
    (06:59-07:41 Berlin), blocking the battery and the Research Lab. A foreign lock (any other job)
    is NEVER touched, dead pid or not -- its owner's protocol is not ours to judge."""
    cur = read()
    if not cur or not str(cur.get("job", "")).startswith("refcv7-navsim"):
        return False
    try:
        import psutil
        alive = psutil.pid_exists(int(cur.get("pid", -1)))
    except Exception:                                                     # noqa: BLE001
        return False
    if alive:
        return False
    ok = release(str(cur.get("token", "")))
    log(f"[gpu_lock] reaped this suite's STALE lock {cur} -> {ok}")
    return ok


def acquire(job: str, wait_s: float = 0.0, poll_s: float = 60.0, log=print) -> str | None:
    """-> the token, or None after ``wait_s``. Requires the lock ABSENT and no other python on
    the card, then creates the lock atomically and re-checks the card (a race is released)."""
    t0 = time.time()
    while True:
        reap_own_stale(log)
        cur = read()
        busy = []
        if cur is None:
            try:
                busy = other_python_compute()
            except Exception as e:                                        # noqa: BLE001
                busy = [{"nvidia_smi_error": repr(e)}]
            if not busy:
                tok = f"{job}:{os.getpid()}:{uuid.uuid4().hex[:12]}"
                rec = {"job": job, "pid": os.getpid(), "token": tok,
                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                try:
                    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    cur = read()
                else:
                    with os.fdopen(fd, "w", encoding="utf-8") as fh:
                        json.dump(rec, fh)
                    try:
                        late = other_python_compute()
                    except Exception as e:                                # noqa: BLE001
                        late = [{"nvidia_smi_error": repr(e)}]
                    if not late:
                        return tok
                    release(tok)
                    busy = late
        if time.time() - t0 >= wait_s:
            return None
        log(f"[gpu_lock] waiting: lock={cur} busy={busy}")
        time.sleep(poll_s)


def held_by(token: str) -> bool:
    cur = read()
    return bool(cur) and cur.get("token") == token


def release(token: str) -> bool:
    cur = read()
    if cur and cur.get("token") == token:
        try:
            os.remove(LOCK)
            return True
        except OSError:
            return False
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("acquire", "release", "status"))
    ap.add_argument("--job", default="refcv7-navsim")
    ap.add_argument("--token", default="")
    ap.add_argument("--wait-s", type=float, default=0.0)
    a = ap.parse_args(argv)
    if a.cmd == "acquire":
        t = acquire(a.job, a.wait_s)
        print(t or "")
        return 0 if t else 3
    if a.cmd == "release":
        ok = release(a.token)
        print("RELEASED" if ok else "NOT_HELD")
        return 0 if ok else 1
    print(json.dumps({"lock": read(), "other_python_compute": other_python_compute()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
