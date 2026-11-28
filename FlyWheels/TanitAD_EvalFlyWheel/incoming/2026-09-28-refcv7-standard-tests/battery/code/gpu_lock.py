"""The dev-box GPU lock (brief, binding): ONE GPU job at a time on the RTX 4060, shared with the NavSim
agent. A job CREATES `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock` exclusively (O_CREAT|O_EXCL),
writing its job name and pid, and removes it when done. It also requires `nvidia-smi` to show no
other python compute app (`gpu_gate.evaluate`, which never counts the caller's own pids).

    python gpu_lock.py acquire --job refcv7-g0 --max-wait-s 21600 --out lock.json   # wait, then hold
    python gpu_lock.py release --job refcv7-g0

⛔ A lock held by someone else is NEVER broken by this tool, even if it looks stale: the NavSim
agent may write an MSYS pid that Windows tools cannot see, so "pid not alive" is not evidence.
A stale lock is reported loudly (every poll) for a human / the Master Mind to clear.
Assert on the JSON (`--out`), never on the exit code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

LOCK = os.environ.get("REFCV7_GPU_LOCK", "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock")


def read_lock() -> dict | None:
    try:
        with open(LOCK, encoding="utf-8") as f:
            txt = f.read()
    except FileNotFoundError:
        return None
    except OSError as exc:
        return {"unreadable": str(exc)}
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {"raw": txt[:300]}


def try_acquire(job: str, pid: int) -> bool:
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"job": job, "pid": pid, "win_pid_of_locker": os.getpid(),
                   "host": os.environ.get("COMPUTERNAME"),
                   "acquired": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f)
    return True


def release(job: str, pid: int | None = None) -> dict:
    """Remove the lock ONLY if it is still ours: same job name and, when given, same pid. Extra
    fields written by another session (token, utc, win_pid_of_locker, ...) are tolerated."""
    cur = read_lock()
    if cur is None:
        return {"released": False, "why": "no lock file"}
    if not isinstance(cur, dict) or cur.get("job") != job:
        return {"released": False, "why": f"lock is held by {cur.get('job') if isinstance(cur, dict) else cur!r}, "
                                          f"not {job!r}", "holder": cur}
    if pid is not None and cur.get("pid") not in (pid, str(pid)):
        return {"released": False, "why": f"lock job matches but pid {cur.get('pid')} != {pid}",
                "holder": cur}
    os.remove(LOCK)
    return {"released": True, "holder": cur}


def smi_ok(g: dict) -> bool:
    """The smi rule alone (memory + other python apps); the RAM rule is reported, not waited on.
    ⛔ FAIL CLOSED: a gate whose probe failed (`probe_ok` False, `gpu_used_mib` None) is NOT ok --
    the card's occupancy is unknown, so the lock is never taken on it."""
    used = g.get("gpu_used_mib")
    return (g.get("probe_ok", True) is not False and isinstance(used, (int, float))
            and used < 4300 and not g.get("python_compute"))


def acquire(job: str, pid: int, max_wait_s: float, poll_s: float = 60, log=print) -> dict:
    """Wait for (no lock file) AND (smi rule), then create the lock. A failed `nvidia-smi` probe
    (2026-10-04 03:35: a 60 s TimeoutExpired under host memory pressure killed this function and with
    it the step-50,400 chain) is a WAIT and is RETRIED every poll until `max_wait_s` --
    `gpu_gate.gate` returns it fail-closed; `probe_failures` counts them."""
    import gpu_gate
    t0 = time.time()
    n = 0
    n_fail = 0
    last_err = None
    while True:
        n += 1
        g = gpu_gate.gate(self_pids=[pid, os.getpid()])
        if g.get("probe_ok") is False:
            n_fail += 1
            last_err = g.get("probe_error")
        cur = read_lock()
        ok_smi = smi_ok(g)
        if cur is None and ok_smi and try_acquire(job, pid):
            return {"ok": True, "waited_s": round(time.time() - t0, 1), "polls": n, "gate": g,
                    "probe_failures": n_fail, "last_probe_error": last_err, "lock": read_lock()}
        if time.time() - t0 > max_wait_s:
            return {"ok": False, "waited_s": round(time.time() - t0, 1), "polls": n, "gate": g,
                    "probe_failures": n_fail, "last_probe_error": last_err, "holder": cur}
        log(f"[gpu_lock] WAIT job={job} holder={cur} smi_ok={ok_smi} "
            f"python_compute={g.get('python_compute')} used={g.get('gpu_used_mib')}MiB "
            f"free_ram={g.get('free_ram_gb')}GB"
            + (f" PROBE_FAILED ({g.get('probe_error')}) -- retrying" if g.get("probe_ok") is False
               else ""), flush=True)
        time.sleep(poll_s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("acquire", "release", "show"))
    ap.add_argument("--job", default="refcv7-battery")
    ap.add_argument("--pid", type=int, default=None, help="the pid to record (default: this one)")
    ap.add_argument("--max-wait-s", type=int, default=6 * 3600)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    # ⛔ the `--out` ARTIFACT is written on EVERY path (2026-10-04: the crash wrote none, and the chain's
    # reader then died on FileNotFoundError). An unexpected error is `ok False` with the error named,
    # and a lock this call itself created (same job AND pid) is released first, so a crash can never
    # leave a lock that no EXIT trap owns.
    try:
        if a.cmd == "show":
            r = {"lock": read_lock()}
        elif a.cmd == "release":
            r = release(a.job, a.pid)
        else:
            r = acquire(a.job, a.pid or os.getpid(), a.max_wait_s)
    except Exception as exc:                            # noqa: BLE001 -- recorded, fail closed
        r = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        if a.cmd == "acquire":
            r["release_after_error"] = release(a.job, a.pid or os.getpid())
    print(json.dumps(r, default=str), flush=True)
    if a.out:
        json.dump(r, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    sys.exit(0)
