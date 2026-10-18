#!/usr/bin/env python3
"""Guarded tar-over-ssh copier (dev box side) for SR-TTC S0.  Derived from the held-out completion run's `podcopy.py`; the ssh command prefix comes from env PODSSH (never written to a file).

CHANGED (coordinator's scope change 2026-10-18, "pod reads ARE allowed while P6 trains, under P6's overlap rule"):
  * the old guard (STOP when a GPU compute app / p6_pod.sh is on the pod) is GONE; before EVERY 120 MB batch the tail of P6's chain log on the pod is read and, if it shows `P6_GUARD`
    (a guard PAUSE or slowdown line), the batch is NOT started: exit 3, the caller retries in 10 min;
  * the pod-side reader is `nice -n 19 ionice -c3 tar -cf - -T -` and NOTHING else runs on the pod per batch: ONE stream at a time, no pod-side python, no pod-side hashing (the old
    remote `sha256sum` is gone: it is CPU work on the pod).  Integrity is now: the exact byte count of every member must equal the size listed by the pod's own `stat` (ssh adds a MAC to the stream),
    the local sha256 of every file is recorded in the manifest, and every file is opened by its consumer (sqlite for the DBs, cv2.imdecode for the frames) before it is used;
  * never stages onto the pod, never touches P6's processes or locks (read-only: `tail`, `tar -c`).
stdin-free CLI:  podcopy.py --list LIST.tsv --dest ROOT --manifest OUT.tsv [--max-batch-mb 120] [--min-free-gb 21] [--p6-log /workspace/data/refe_p6/chain.log]
LIST.tsv lines: <bytes>\t<pod_basedir>\t<rel>      (rel is also the path under --dest)
Resumable: a dest file whose size already equals the listed size is skipped (files are written as .part and renamed, so a file at its final name is complete).
Exit codes: 0 all done, 3 P6_GUARD in P6's chain log (retry in 10 min), 4 size mismatch, 5 disk guard, 6 ssh failure.
"""
import argparse
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time

PODSSH = shlex.split(os.environ["PODSSH"])
NICE = "nice -n 19 ionice -c3"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def ssh(cmd, stdin_bytes=b"", timeout=900):
    r = subprocess.run(PODSSH + [cmd], input=stdin_bytes, capture_output=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def p6_guard_ok(p6_log):
    """(ok, line): False when the last 20 lines of P6's chain log contain P6_GUARD, or the tail cannot be read."""
    rc, out, _err = ssh(f"{NICE} tail -n 20 {shlex.quote(p6_log)}", timeout=60)
    if rc != 0:
        return False, f"ssh/tail rc={rc}"
    for ln in out.decode(errors="replace").splitlines():
        if "P6_GUARD" in ln:
            return False, ln.strip()[:200]
    return True, ""


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max-batch-mb", type=float, default=120.0)
    ap.add_argument("--min-free-gb", type=float, default=21.0)
    ap.add_argument("--p6-log", default="/workspace/data/refe_p6/chain.log")
    a = ap.parse_args()
    rows = []
    for ln in open(a.list, encoding="utf-8"):
        ln = ln.rstrip("\r\n")
        if ln:
            b, base, rel = ln.split("\t")
            rows.append((int(b), base, rel))
    done = {}
    if os.path.exists(a.manifest):
        for ln in open(a.manifest, encoding="utf-8"):
            p = ln.rstrip("\n").split("\t")
            if len(p) == 4 and p[3] == "OK":
                done[p[0]] = p
    rows = [r for r in rows if r[2] not in done or not os.path.exists(os.path.join(a.dest, r[2]))]
    log(f"{len(rows)} files to do ({sum(r[0] for r in rows) / 1048576:.1f} MiB); {len(done)} already in the manifest")
    batches, cur, cur_b = [], [], 0
    for r in rows:
        if cur and (r[1] != cur[0][1] or cur_b + r[0] > a.max_batch_mb * 1048576):
            batches.append(cur)
            cur, cur_b = [], 0
        cur.append(r)
        cur_b += r[0]
    if cur:
        batches.append(cur)
    mf = open(a.manifest, "a", encoding="utf-8", newline="\n")
    total_done = 0
    t_start = time.time()
    for bi, batch in enumerate(batches, 1):
        ok, why = p6_guard_ok(a.p6_log)
        if not ok:
            log(f"STOP: P6_GUARD / unreadable chain log ({why}); batch {bi}/{len(batches)} not started")
            return 3
        bbytes = sum(r[0] for r in batch)
        free = shutil.disk_usage(a.dest if os.path.isdir(a.dest) else os.path.dirname(a.dest.rstrip("/\\")) or ".").free
        if free - bbytes < a.min_free_gb * (1 << 30):
            log(f"STOP: disk guard: free {free / (1 << 30):.2f} GiB - batch {bbytes / (1 << 20):.0f} MiB would leave < {a.min_free_gb} GiB")
            return 5
        base = batch[0][1]
        todo = []
        for r in batch:
            d = os.path.join(a.dest, r[2])
            if os.path.exists(d) and os.path.getsize(d) == r[0]:
                mf.write(f"{r[2]}\t{r[0]}\t{sha256_file(d)}\tOK\n")
            else:
                todo.append(r)
        mf.flush()
        if todo:
            t0 = time.time()
            p = subprocess.Popen(PODSSH + [f"cd {shlex.quote(base)} && {NICE} tar -cf - -T -"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            p.stdin.write(("\n".join(r[2] for r in todo) + "\n").encode())
            p.stdin.close()
            want = {r[2]: r for r in todo}
            stall = {"t": time.time(), "done": False}

            def _wd(p=p, stall=stall):
                while not stall["done"]:
                    time.sleep(5)
                    if time.time() - stall["t"] > 180:
                        log("WATCHDOG: no progress for 180 s, killing the ssh client")
                        p.kill()
                        return
            threading.Thread(target=_wd, daemon=True).start()
            got = {}
            with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
                for m in tf:
                    if not m.isfile() or m.name not in want:
                        log(f"unexpected tar member {m.name!r} ignored")
                        continue
                    d = os.path.join(a.dest, m.name)
                    os.makedirs(os.path.dirname(d), exist_ok=True)
                    h = hashlib.sha256()
                    n = 0
                    with open(d + ".part", "wb") as fo:
                        src = tf.extractfile(m)
                        for b in iter(lambda: src.read(1 << 20), b""):
                            fo.write(b)
                            h.update(b)
                            n += len(b)
                    if n != want[m.name][0]:
                        log(f"MISMATCH {m.name}: size {n} vs listed {want[m.name][0]}")
                        os.remove(d + ".part")
                        mf.write(f"{m.name}\t{n}\t{h.hexdigest()}\tMISMATCH\n")
                        mf.flush()
                        return 4
                    os.replace(d + ".part", d)
                    got[m.name] = (n, h.hexdigest())
                    stall["t"] = time.time()
            while p.stdout.read(1 << 20):          # drain the tar end padding, or the ssh client blocks on the pipe and never exits
                pass
            p.wait()
            stall["done"] = True
            if p.returncode != 0 or set(got) != set(want):
                log(f"tar rc={p.returncode} got {len(got)}/{len(want)}; stderr={p.stderr.read()[:200]!r}")
                return 6
            for k, (n, h) in got.items():
                mf.write(f"{k}\t{n}\t{h}\tOK\n")
            mf.flush()
            dt = time.time() - t0
            log(f"batch {bi}/{len(batches)} ({base.rsplit('/', 1)[-1]}): {len(todo)} files {sum(v[0] for v in got.values()) / 1048576:.1f} MiB in {dt:.1f}s = {sum(v[0] for v in got.values()) / 1048576 / max(dt, 1e-6):.1f} MiB/s, size equal to the pod's stat")
        total_done += len(batch)
    log(f"ALL DONE {total_done} files in {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
