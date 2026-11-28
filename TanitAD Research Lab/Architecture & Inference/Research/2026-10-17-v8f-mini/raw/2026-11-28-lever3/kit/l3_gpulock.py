#!/usr/bin/env python3
"""l3_gpulock.py -- the shared dev-box GPU lock, in the format the other sessions write: C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock = JSON {"job","pid","host","acquired"}, created exclusively (O_EXCL).
Honours the chain marker devbox_gpu.chain_active (no new holder while it exists). Never forces or removes another holder's lock; release removes the file ONLY if it still names this job and pid.
take(job, wait_s): polls every 30 s up to wait_s (0 = one try); returns True when held. release(job).  CLI: l3_gpulock.py try JOB | release JOB | show"""
import json, os, socket, sys, time

LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
MARK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"


def holder():
    try:
        return json.load(open(LOCK, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def take(job, wait_s=0):
    t0 = time.time()
    while True:
        if not os.path.exists(MARK):
            try:
                fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, json.dumps({"job": job, "pid": os.getpid(), "host": socket.gethostname(), "acquired": time.strftime("%Y-%m-%dT%H:%M:%S%z")}).encode())
                os.close(fd)
                return True
            except FileExistsError:
                pass
        if time.time() - t0 >= wait_s:
            return False
        time.sleep(30)


def release(job):
    h = holder()
    if h and h.get("job") == job and h.get("pid") == os.getpid():
        os.remove(LOCK)
        return True
    return False


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "show":
        print(holder(), "marker" if os.path.exists(MARK) else "no marker")
    elif c == "try":
        print("HELD" if take(sys.argv[2]) else f"BUSY {holder()}")
