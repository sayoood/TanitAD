"""OPS1 test fixture (NOT part of the read; unpinned): a PLANTED stand-in for `pg_extract.py work`.  It is started by the chain exactly as the real extractor is
(`PY -B -W ignore <this> work --subset S --out O`) and, like the Windows venv `python.exe` launcher, it runs the actual work in a CHILD interpreter with a DIFFERENT pid, which is the
one that writes `WORKER_DONE_<its own pid>.txt` and the `<log>.lock` files.  The behaviour of the n-th worker started comes from the OPS1_PLAN environment variable
({"modes": [...], "default": "clean"}).  It opens no real data and no label.
"""
import json
import os
import subprocess
import sys
import time


def arg(a, flag):
    return a[a.index(flag) + 1]


def inner(a):
    subset, out = arg(a, "--subset"), arg(a, "--out")
    logs = json.load(open(subset, encoding="utf-8"))["logs_in_order"]
    plan = json.loads(os.environ["OPS1_PLAN"])
    claims = os.path.join(os.path.dirname(out), "claims")
    os.makedirs(claims, exist_ok=True)
    i = 0
    while True:
        try:
            os.close(os.open(os.path.join(claims, str(i)), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            i += 1
    mode = plan["modes"][i] if i < len(plan["modes"]) else plan["default"]
    pid = os.getpid()
    os.makedirs(out, exist_ok=True)

    def take(name):
        try:
            fd = os.open(f"{out}/{name}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{pid} t".encode())
            os.close(fd)
            return True
        except FileExistsError:
            return False

    def process_all():
        for name in logs:
            if os.path.exists(f"{out}/{name}.npz") or not take(name):
                continue
            with open(f"{out}/{name}.tmp.npz", "wb") as f:
                f.write(b"x")
            os.replace(f"{out}/{name}.tmp.npz", f"{out}/{name}.npz")
            os.remove(f"{out}/{name}.lock")

    def record():
        with open(f"{out}/WORKER_DONE_{pid}.txt", "w", encoding="utf-8") as f:
            f.write("t logs_done 0\n")

    time.sleep(0.2)
    if mode == "clean":
        process_all()
        record()
        return 0
    if mode == "silent0":                                                                 # rc 0, no done record, logs still missing
        return 0
    if mode == "record_only":                                                             # rc 0 and a done record, but it extracted nothing
        record()
        return 0
    if mode == "allnorecord":                                                             # rc 0, everything extracted, NO done record
        process_all()
        return 0
    if mode == "crash_with_record":                                                       # rc 5 with a done record present and a lock left behind
        for name in logs:
            if not os.path.exists(f"{out}/{name}.npz") and take(name):
                break
        record()
        return 5
    raise SystemExit(f"unknown mode {mode}")


def main():
    a = sys.argv
    if "--inner" in a:
        return inner(a)
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--inner"] + a[1:], stdin=subprocess.DEVNULL)      # the 'launcher': child pid != our pid
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
