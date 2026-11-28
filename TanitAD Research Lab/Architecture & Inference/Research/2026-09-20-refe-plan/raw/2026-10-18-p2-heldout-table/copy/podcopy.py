#!/usr/bin/env python3
"""Guarded tar-over-ssh copier (dev box side). The ssh command prefix comes from env PODSSH (never written to a file).

stdin-free CLI:  podcopy.py --list LIST.tsv --dest ROOT --manifest OUT.tsv [--max-batch-mb 120] [--min-free-gb 10]
LIST.tsv lines: <bytes>\t<pod_basedir>\t<rel>      (rel is also the path under --dest)
Per batch: (1) guard -- STOP if a GPU compute app or a p6_pod.sh is on the pod, (2) disk guard, (3) remote sha256 of the batch,
(4) remote `tar -cf -` streamed into Python's tarfile (only expected member names are written, via .part + os.replace),
(5) local sha256 must equal the remote one.  Resumable: a dest file whose sha256 already equals the remote one is skipped.
Exit codes: 0 all done, 3 pod busy (STOP), 4 sha/size mismatch, 5 disk guard, 6 ssh failure.
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
GUARD = ('n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); '
         'if [ "$n" -ne 0 ] || ps -eo args | grep -v grep | grep -q "refe/p6_pod[.]sh"; then echo "BUSY n=$n"; else echo IDLE; fi')


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def ssh(cmd, stdin_bytes=b"", timeout=900):
    r = subprocess.run(PODSSH + [cmd], input=stdin_bytes, capture_output=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def guard():
    rc, out, err = ssh(GUARD, timeout=60)
    s = out.decode(errors="replace").strip()
    return (rc == 0 and s == "IDLE"), s if rc == 0 else f"ssh rc={rc}"


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
    ap.add_argument("--min-free-gb", type=float, default=10.0)
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
    rows = [r for r in rows if r[2] not in done]
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
        ok, why = guard()
        if not ok:
            log(f"STOP: pod not idle ({why}); batch {bi}/{len(batches)} not started")
            return 3
        bbytes = sum(r[0] for r in batch)
        free = shutil.disk_usage(a.dest if os.path.isdir(a.dest) else os.path.dirname(a.dest.rstrip("/\\")) or ".").free
        if free - bbytes < a.min_free_gb * (1 << 30):
            log(f"STOP: disk guard: free {free / (1 << 30):.2f} GiB - batch {bbytes / (1 << 20):.0f} MiB would leave < {a.min_free_gb} GiB")
            return 5
        base = batch[0][1]
        rels = [r[2] for r in batch]
        lst = ("\n".join(rels) + "\n").encode()
        rc, out, err = ssh(f"cd {shlex.quote(base)} && xargs -d '\\n' sha256sum --", lst)
        if rc != 0:
            log(f"ssh sha256 rc={rc} {err[:200]!r}")
            return 6
        rsha = {}
        for ln in out.decode().splitlines():
            h, _, n = ln.partition("  ")
            rsha[n] = h
        if set(rsha) != set(rels):
            log("remote sha256 listing does not cover the batch")
            return 4
        # skip files already present and identical
        todo = []
        for r in batch:
            d = os.path.join(a.dest, r[2])
            if os.path.exists(d) and os.path.getsize(d) == r[0] and sha256_file(d) == rsha[r[2]]:
                mf.write(f"{r[2]}\t{r[0]}\t{rsha[r[2]]}\tOK\n")
            else:
                todo.append(r)
        mf.flush()
        if todo:
            t0 = time.time()
            p = subprocess.Popen(PODSSH + [f"cd {shlex.quote(base)} && tar -cf - -T -"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            p.stdin.write(("\n".join(r[2] for r in todo) + "\n").encode())
            p.stdin.close()
            want = {r[2]: r for r in todo}
            stall = {"t": time.time(), "done": False}

            def _wd(p=p, stall=stall):
                while not stall["done"]:
                    time.sleep(5)
                    if time.time() - stall["t"] > 120:
                        log("WATCHDOG: no progress for 120 s, killing the ssh client")
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
                    if n != want[m.name][0] or h.hexdigest() != rsha[m.name]:
                        log(f"MISMATCH {m.name}: size {n} vs {want[m.name][0]}, sha {h.hexdigest()[:12]} vs {rsha[m.name][:12]}")
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
            log(f"batch {bi}/{len(batches)} ({base.rsplit('/', 1)[-1]}): {len(todo)} files {sum(v[0] for v in got.values()) / 1048576:.1f} MiB in {dt:.1f}s = {sum(v[0] for v in got.values()) / 1048576 / max(dt, 1e-6):.1f} MiB/s, sha256 equal on both sides")
        total_done += len(batch)
    log(f"ALL DONE {total_done} files in {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
