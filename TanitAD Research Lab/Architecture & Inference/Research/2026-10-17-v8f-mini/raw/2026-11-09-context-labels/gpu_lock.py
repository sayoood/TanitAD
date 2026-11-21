"""Shared dev-box GPU lock (wait_idle pattern of aux_navtest_forward.acquire_gpu_lock, restated; behaviour identical):
wait_idle.py must exit 0 (GPU idle: < 3500 MiB and no foreign GPU job process), then O_EXCL create of the lock. A lock that does not
carry OUR owner marker is never removed. Released by `release`, and always at the end of a 30-minute job."""
import json
import os
import subprocess
import sys
import time

LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
WAIT_IDLE = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/strat_labels/wait_idle_strat.py"
OWNER = "strat_labels_ctx"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"


HELD = "held-by-labels_first"


class GpuBusy(RuntimeError):
    pass


def pid_alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
    return str(pid) in out


def acquire(wait_timeout_s=6 * 3600, lock=LOCK):
    """Block until the GPU is idle and the lock is ours. Raises GpuBusy on timeout."""
    if os.environ.get("STRAT_LOCK_HELD"):      # labels_first.py holds the lock for this whole chain
        return HELD
    t0 = time.time()
    print(f"[gpu_lock] waiting for idle GPU via {WAIT_IDLE}", flush=True)
    while True:
        try:
            r = subprocess.run([PY, WAIT_IDLE], timeout=max(60, wait_timeout_s - (time.time() - t0)))
        except subprocess.TimeoutExpired:
            raise GpuBusy("wait_idle timeout")
        if r.returncode != 0:
            raise GpuBusy(f"wait_idle rc {r.returncode}")
        body = json.dumps({"pid": os.getpid(), "owner": OWNER, "t": time.time()})
        os.makedirs(os.path.dirname(lock), exist_ok=True)
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, body.encode())
            os.close(fd)
            time.sleep(4)
            print(f"[gpu_lock] lock taken after {time.time() - t0:.0f}s", flush=True)
            return body
        except FileExistsError:
            try:
                holder = json.loads(open(lock, encoding="utf-8").read())
            except (OSError, ValueError):
                holder = {}
            if holder.get("owner") == OWNER and not pid_alive(holder.get("pid", -1)) and time.time() - os.path.getmtime(lock) > 600:
                os.remove(lock)          # our own dead holder only
            elif time.time() - t0 > wait_timeout_s:
                raise GpuBusy(f"lock held by {holder}")
            time.sleep(30)


def release(body, lock=LOCK):
    if body == HELD:
        return
    try:
        if open(lock, encoding="utf-8").read() == body:
            os.remove(lock)
    except OSError:
        pass
