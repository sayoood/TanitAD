"""The shared dev-box GPU lock, protocol copied from PK/eval/aux_navtest_forward.py (acquire_gpu_lock / release_gpu_lock / gpu_lock; same LOCK path, same body schema
{"pid","owner","t"}, same wait_idle precondition, same memory bar). Differences, all deliberate: OWNER = "yf2_yaw_cache"; WAIT_IDLE points at the E: copy (D: is absent);
`acquire_blocking` RETRIES a busy lock (a foreign lock is NEVER removed, never forced) until `give_up_s`, so a detached job can queue behind MM's probes without a model turn.
The imports of aux_navtest_forward (torch, aux_drivable_head, nuPlan) are avoided on purpose: this file must run before any heavy import."""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time

LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
WAIT_IDLE = "E:/Projects/TanitAD-artifacts/_rlfan_work/wait_idle.py"
GPU_MAX_USED_MIB = 3500
OWNER = "yf2_yaw_cache"


class GpuBusy(RuntimeError):
    pass


def gpu_used_mib() -> int:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True)
    try:
        return int(r.stdout.decode("utf-8", "replace").strip().splitlines()[0])
    except (ValueError, IndexError):
        return -1


def pid_alive(pid) -> bool:
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True)
    return f'"{pid}"' in r.stdout.decode("utf-8", "replace")


def acquire_gpu_lock(lock: str = LOCK, wait_idle: str | None = WAIT_IDLE, wait_timeout_s: float = 240.0, used_fn=gpu_used_mib, settle_s: float = 4.0, owner: str = OWNER) -> str:
    """Fail CLOSED (verbatim semantics of aux_navtest_forward.acquire_gpu_lock): the idle waiter must exit 0, the lock is created O_EXCL, a lock without OUR owner marker is never
    removed, and memory.used must still read <= GPU_MAX_USED_MIB `settle_s` after taking it. Returns the lock body."""
    if wait_idle:
        try:
            r = subprocess.run([sys.executable, wait_idle], timeout=wait_timeout_s)
        except subprocess.TimeoutExpired:
            raise GpuBusy(f"wait_idle.py did not report an idle GPU within {wait_timeout_s:.0f} s")
        if r.returncode != 0:
            raise GpuBusy(f"wait_idle.py exited {r.returncode}")
    body = json.dumps({"pid": os.getpid(), "owner": owner, "t": time.time()})
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    for attempt in (1, 2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                holder = json.loads(open(lock, encoding="utf-8").read())
                age = time.time() - os.path.getmtime(lock)
            except (OSError, ValueError):
                raise GpuBusy(f"{lock} exists and is unreadable")
            if not isinstance(holder, dict) or holder.get("owner") != owner:
                who = (holder.get("job") or holder.get("owner") or "?") if isinstance(holder, dict) else "?"
                raise GpuBusy(f"{lock} is held by a FOREIGN lock (job/owner {who!r}, age {age:.0f} s): never removed: {holder}")
            if "pid" not in holder or pid_alive(holder["pid"]):
                raise GpuBusy(f"{lock} is held by {holder}")
            if attempt == 2 or age <= 600:
                raise GpuBusy(f"{lock} holder pid {holder['pid']} is gone but the lock is only {age:.0f} s old")
            os.remove(lock)
    else:
        raise GpuBusy("lock not acquired")
    os.write(fd, body.encode("utf-8"))
    os.close(fd)
    time.sleep(settle_s)
    used = used_fn()
    if not 0 <= used <= GPU_MAX_USED_MIB:
        release_gpu_lock(lock, body)
        raise GpuBusy(f"memory.used {used} MiB > {GPU_MAX_USED_MIB} after taking the lock")
    return body


def release_gpu_lock(lock: str, body: str) -> None:
    try:
        cur = json.loads(open(lock, encoding="utf-8").read())
        mine = json.loads(body)
        if cur.get("owner") == mine.get("owner") and cur.get("pid") == mine["pid"]:
            os.remove(lock)
    except (OSError, ValueError, AttributeError):
        pass


def acquire_blocking(give_up_s: float, poll_s: float = 60.0, log=print, **kw) -> str:
    """Queue behind a busy GPU. Polls `acquire_gpu_lock` every `poll_s`; GpuBusy is the expected answer while MM's probes run. Never forces."""
    t0 = time.time()
    n = 0
    while True:
        try:
            return acquire_gpu_lock(**kw)
        except GpuBusy as e:
            n += 1
            if time.time() - t0 > give_up_s:
                raise
            if n == 1 or n % 10 == 0:
                log(f"[{time.strftime('%H:%M:%S')}] GPU busy (attempt {n}): {str(e)[:200]}")
            time.sleep(poll_s)


@contextlib.contextmanager
def gpu_lock(**kw):
    body = acquire_gpu_lock(**kw)
    try:
        yield body
    finally:
        release_gpu_lock(kw.get("lock", LOCK), body)
