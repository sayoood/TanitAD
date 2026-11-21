"""The shared dev-box GPU lock (`C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock`) -- acquire / release / status.

The RTX 4060 is shared by several sessions through ONE lock file of JSON {pid, owner, job, host, t, acquired}.
This helper follows the protocol recorded in memory `devbox-drive-letter-and-detached-jobs` (2026-10-17):
  * acquire creates the file EXCLUSIVELY (O_EXCL) -- two waiters cannot both win;
  * a lock that carries OUR owner marker (`owner` == --owner, default "ra1a") AND whose pid is dead, or that was taken before
    the last boot (pid reuse), is STALE: it is RENAMED ASIDE (`devbox_gpu.lock.stale_<utc>_pid<pid>`), never deleted;
    any OTHER schema (e.g. the refcv8 battery's {job, pid = an MSYS pid, win_pid_of_locker, host, acquired}) is a LIVE job's lock
    -- an MSYS pid is invisible to the Windows pid probe and that schema has no `t` -- so it is waited on, never renamed or removed;
  * a lock held by a live pid is waited on (poll --poll-s, give up after --max-wait-s, exit 2);
  * release removes the file ONLY if it carries our owner AND our pid (never another session's lock).

    python gpu_lock.py acquire --pid <winpid> --owner ra1a --job refav1-hj-pooled-encode [--max-wait-s 28800]
    python gpu_lock.py release --pid <winpid>
    python gpu_lock.py status
Under Git Bash the Windows pid of the running script is `$(cat /proc/$$/winpid)` -- NOT `$$`.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import socket
import sys
import time
from pathlib import Path

LOCK = Path("C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock")
OWNER = "ra1a"                                                       # the owner marker this helper writes by default (--owner)
STILL_ACTIVE = 259


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, int(pid))                       # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        return bool(ok) and code.value == STILL_ACTIVE
    finally:
        k.CloseHandle(h)


def boot_time() -> float:
    return time.time() - ctypes.windll.kernel32.GetTickCount64() / 1000.0


def read(lock: Path):
    try:
        return json.loads(lock.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:                                                # unreadable / torn: treat as held-by-unknown
        return {"pid": -1, "owner": "UNREADABLE", "t": time.time()}


def stale_reason(info: dict, now_boot: float, owner: str = OWNER):
    if info is None:
        return None
    if not isinstance(info, dict) or info.get("owner") != owner:     # a lock WITHOUT our owner marker is a live job's lock: pid and `t` say nothing about it
        return None
    if float(info.get("t", 0)) < now_boot - 5:
        return "acquired before the last boot (pid may have been reused)"
    if not pid_alive(int(info.get("pid", -1))):
        return f"pid {info.get('pid')} is not running"
    return None


def acquire(lock: Path, pid: int, owner: str, job: str, max_wait_s: float, poll_s: float) -> int:
    t0 = time.time()
    lock.parent.mkdir(parents=True, exist_ok=True)
    while True:
        info = read(lock)
        why = stale_reason(info, boot_time(), owner)
        if info is not None and info.get("owner") == "UNREADABLE":
            why = None                                               # do not guess about a torn file
        if info is not None and why:
            aside = lock.with_name(lock.name + time.strftime(".stale_%Y%m%dT%H%M%S", time.gmtime()) + f"_pid{info.get('pid')}")
            try:
                os.replace(lock, aside)
                print(f"[gpu_lock] stale lock ({why}) renamed aside -> {aside.name}", flush=True)
            except OSError as e:
                print(f"[gpu_lock] could not rename stale lock aside: {e}", flush=True)
            continue
        if info is None:
            body = json.dumps({"pid": pid, "owner": owner, "job": job, "host": socket.gethostname(), "t": time.time(),
                               "acquired": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue                                             # lost the race: loop and re-read
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(body)
            print(f"[gpu_lock] ACQUIRED by {owner} pid {pid} after {time.time() - t0:.0f}s", flush=True)
            return 0
        if time.time() - t0 > max_wait_s:
            print(f"[gpu_lock] TIMEOUT after {max_wait_s:.0f}s; held by {info.get('owner')}/{info.get('job')} pid {info.get('pid')}",
                  flush=True)
            return 2
        print(f"[gpu_lock] held by {info.get('owner')}/{info.get('job')} pid {info.get('pid')} since {info.get('acquired')}; "
              f"waiting ({time.time() - t0:.0f}s)", flush=True)
        time.sleep(poll_s)


def release(lock: Path, pid: int, owner: str = OWNER) -> int:
    info = read(lock)
    if info is None:
        print("[gpu_lock] nothing to release", flush=True)
        return 0
    if info.get("owner") != owner or int(info.get("pid", -1)) != int(pid):
        print(f"[gpu_lock] NOT releasing: lock belongs to pid {info.get('pid')} ({info.get('owner')}), not {pid} ({owner})", flush=True)
        return 3
    lock.unlink()
    print("[gpu_lock] released", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["acquire", "release", "status"])
    ap.add_argument("--lock", default=str(LOCK))
    ap.add_argument("--pid", type=int, default=os.getpid())
    ap.add_argument("--owner", default=OWNER)
    ap.add_argument("--job", default="refav1-hj-pooled-encode")
    ap.add_argument("--max-wait-s", type=float, default=28800)
    ap.add_argument("--poll-s", type=float, default=30)
    a = ap.parse_args(argv)
    lock = Path(a.lock)
    if a.cmd == "status":
        info = read(lock)
        print(json.dumps({"held": info is not None, "info": info,
                          "stale": stale_reason(info, boot_time(), a.owner) if info else None}, indent=1))
        return 0
    if a.cmd == "acquire":
        return acquire(lock, a.pid, a.owner, a.job, a.max_wait_s, a.poll_s)
    return release(lock, a.pid, a.owner)


if __name__ == "__main__":
    sys.exit(main())
