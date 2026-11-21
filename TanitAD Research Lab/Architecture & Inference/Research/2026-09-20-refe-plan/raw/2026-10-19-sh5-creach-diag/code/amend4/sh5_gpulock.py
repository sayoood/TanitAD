#!/usr/bin/env python3
"""Dev-box GPU lock for the SH5 GPU jobs (`sh5_ctx_build.py run`, `sh5_train_eval.py fit`).  The pattern is `hn34/work/gpu_locked.py` as patched for L1 (`l1run/code/launch_stepA_ctype.py`,
PI decision 2026-10-19 "Allow compute-type check") and the TRAPS entry "TWO LOCK SCHEMAS SHARE devbox_gpu.lock":

  * waits while `devbox_gpu.chain_active` exists (another chain owns the GPU between its jobs);
  * takes `devbox_gpu.lock` with O_EXCL and OUR owner marker (`sh5_dacprox`).  An existing lock is NEVER removed, renamed or edited when it is a FOREIGN schema or another owner's
    ({"job", "pid" = an MSYS pid ...} is invisible to tasklist, so a dead-pid test would steal a live job's lock).  The one exception is a lock carrying OUR owner marker whose pid is gone and
    which is older than 10 minutes (a launcher of ours that died);
  * after taking it, requires NO compute-only (type C) process in the nvidia-smi process table (C+G desktop apps are ignored: they hold ~2.3 GB with no compute job) and memory.used <= 6000 MiB
    (an unlocked GPU user means: give the lock back and wait).  An unreadable nvidia-smi fails CLOSED (compute_pids -> [-1]);
  * waits while psutil available RAM < `min_avail_gb`, BEFORE taking the lock, so no lock is held while RAM-starved;
  * releases only the lock whose bytes are EXACTLY the body this process wrote;
  * exit 75 = the GPU never became free within --max-wait and the job was NOT run.

    sh5_gpulock.py probe                                  # read-only: chain_active / lock holder / memory.used; takes nothing
    sh5_gpulock.py run --max-wait 7200 -- <cmd ...>       # run a subprocess under the lock
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
MARK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"
OWNER = "sh5_dacprox"
MAX_USED_MIB = 6000                      # backstop only; the decisive test is compute_pids()
OWN_STALE_S = 600.0
SETTLE_S = 3.0
EX_BUSY = 75


def used_mib():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, timeout=30)
        return int(r.stdout.decode("utf-8", "replace").strip().splitlines()[0])
    except Exception:
        return 10 ** 9                   # fail CLOSED: an unreadable GPU is a busy GPU


def avail_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except Exception:
        return 0.0                       # fail CLOSED


def compute_pids(text=None):
    """pids of type-C (compute-only) processes in the nvidia-smi process table; C+G desktop apps are not counted.  Unreadable nvidia-smi -> [-1] (fail CLOSED)."""
    if text is None:
        try:
            r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=30)
            text = r.stdout.decode("utf-8", "replace")
        except Exception:
            return [-1]
    out = []
    for line in text.splitlines():
        t = line.strip().strip("|").split()
        if len(t) >= 6 and t[0].isdigit() and t[3].isdigit() and t[4] == "C":
            out.append(int(t[3]))
    return out


def pid_alive(pid) -> bool:
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True)        # bytes: the German "no tasks" line carries 0x81
    return f'"{pid}"' in r.stdout.decode("utf-8", "replace")


def read_holder(lock, now_fn=time.time):
    try:
        return json.loads(open(lock, encoding="utf-8").read()), now_fn() - os.path.getmtime(lock)
    except (OSError, ValueError):
        return None, 0.0


def try_lock(lock=LOCK, mark=MARK, used_fn=used_mib, alive_fn=pid_alive, sleep_fn=time.sleep, now_fn=time.time, compute_fn=None, owner=OWNER, job="sh5"):
    """-> (body, 'ok') or (None, reason).  Never touches a lock that is not ours."""
    if os.path.exists(mark):
        return None, "chain_active present"
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    for attempt in (1, 2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            holder, age = read_holder(lock, now_fn)
            if not isinstance(holder, dict) or holder.get("owner") != owner:
                who = (holder.get("job") or holder.get("owner") or "?") if isinstance(holder, dict) else "unreadable"
                return None, f"lock held by a FOREIGN holder ({who}); never removed"
            if "pid" not in holder or alive_fn(holder["pid"]):
                return None, f"lock held by a live {owner} process"
            if attempt == 2 or age <= OWN_STALE_S:
                return None, f"own lock, holder pid gone but only {age:.0f} s old"
            os.remove(lock)              # OUR marker, dead pid, > 10 min: a launcher of ours that died
    else:
        return None, "lock not acquired"
    body = json.dumps({"pid": os.getpid(), "owner": owner, "job": job, "t": now_fn()})
    os.write(fd, body.encode("utf-8"))
    os.close(fd)
    sleep_fn(SETTLE_S)
    cp = (compute_fn or compute_pids)()  # late-bound: a test or a mutation can replace the module's compute_pids
    if cp:
        release(body, lock)
        return None, f"compute process(es) on the GPU {cp} after taking the lock"
    u = used_fn()
    if u > MAX_USED_MIB:
        release(body, lock)
        return None, f"memory.used {u} MiB after taking the lock"
    return body, "ok"


def release(body, lock=LOCK):
    """remove the lock ONLY if its bytes are exactly the body we wrote"""
    try:
        if open(lock, encoding="utf-8").read() == body:
            os.remove(lock)
            return True
    except OSError:
        pass
    return False


def child_env(base=None) -> dict:
    env = {k: v for k, v in (os.environ if base is None else base).items() if k != "CUDA_VISIBLE_DEVICES"}
    env["SH5_GPU_LOCKED"] = "1"
    return env


def acquire(max_wait, lock=LOCK, mark=MARK, used_fn=used_mib, alive_fn=pid_alive, sleep_fn=time.sleep, now_fn=time.time, poll_s=60.0, log=print, compute_fn=None, avail_fn=None,
            min_avail_gb=8.0, owner=OWNER, job="sh5"):
    """-> body or None (max-wait expired).  RAM gate BEFORE the lock."""
    t0, last = now_fn(), ""
    while True:
        ram = (avail_fn or avail_gb)()
        if ram < min_avail_gb:
            body, why = None, f"RAM available {int(ram)} GB < {min_avail_gb:g} GB (lock not taken)"
        else:
            body, why = try_lock(lock, mark, used_fn, alive_fn, sleep_fn, now_fn, compute_fn, owner, job)
        if body:
            return body
        if why != last:
            log(f"ZZGPUWAIT {why}")
            last = why
        if now_fn() - t0 > max_wait:
            log(f"ZZGPU_NOT_RUN waited {max_wait:.0f} s: {why}")
            return None
        sleep_fn(poll_s)


def run_locked_inprocess(fn, max_wait, min_avail_gb=8.0, owner=OWNER, job="sh5", log=print, **kw):
    """Take the lock (waiting), run `fn()` in THIS process, release only our own body.  -> fn's return code, or EX_BUSY (75) when the GPU never became free."""
    body = acquire(max_wait, log=log, min_avail_gb=min_avail_gb, owner=owner, job=job, **kw)
    if body is None:
        return EX_BUSY
    os.environ["SH5_GPU_LOCKED"] = "1"
    log("ZZGPULOCKED")
    try:
        return fn()
    finally:
        done = release(body, kw.get("lock", LOCK))
        log(f"ZZGPURELEASED {'yes' if done else 'NOT (the lock body changed; left untouched)'}")


def require_locked(device: str) -> None:
    """GPU code refuses to start unless it was launched through this module (exit code 4)."""
    if str(device).startswith("cuda") and os.environ.get("SH5_GPU_LOCKED") != "1":
        raise SystemExit(4)


def probe(lock=LOCK, mark=MARK, used_fn=used_mib, compute_fn=None) -> dict:
    holder, age = read_holder(lock) if os.path.exists(lock) else (None, 0.0)
    cp = (compute_fn or compute_pids)()
    return {"chain_active": os.path.exists(mark), "lock_exists": os.path.exists(lock), "lock_holder": holder, "lock_age_s": round(age), "memory_used_mib": used_fn(), "max_used_mib": MAX_USED_MIB,
            "compute_pids": cp, "would_launch": bool(not os.path.exists(mark) and not os.path.exists(lock) and not cp and used_fn() <= MAX_USED_MIB)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("probe")
    r = sp.add_parser("run")
    r.add_argument("--max-wait", type=float, required=True)
    r.add_argument("--min-avail-gb", type=float, default=8.0)
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        print(json.dumps(probe(), indent=1))
        return 0
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        print("no command after --", file=sys.stderr)
        return 2
    return run_locked_inprocess(lambda: subprocess.run(cmd, env=child_env()).returncode, a.max_wait, a.min_avail_gb)


if __name__ == "__main__":
    sys.exit(main())
