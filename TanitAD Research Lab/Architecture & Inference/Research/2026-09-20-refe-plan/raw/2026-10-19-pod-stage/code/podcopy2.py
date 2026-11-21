#!/usr/bin/env python3
"""Guarded tar-over-ssh copier (dev box side) for the 2026-10-19 pod reads (SH5 FULL frames, 76 nuPlan DBs).  DERIVED FROM `podcopy.py` of SR-TTC S0 (2026-10-18-sel-rank/code/podcopy.py);
the ssh command prefix comes from env PODSSH (read by the caller from a local file, NEVER written anywhere; every logged string is redacted).

SAME AS S0 (P6 overlap rule): the pod-side reader runs at `nice -n 19 ionice -c3`; ONE stream at a time; batches of <= 120 MB; before EVERY batch the tail of P6's chain log is read and a
`P6_GUARD` line means the batch is NOT started (exit 3, the caller retries in 10 min); never starts a GPU process, never touches P6's processes / files, never stages onto the pod.

DIFFERENT FROM S0 (each one on purpose):
  1. The guard check and the pod-side command run in ONE remote shell (the guard decision and the read cannot be split by a P6 state change; 2 ssh calls per batch instead of 3).
  2. PI-approved for this task (2026-10-19): the pod-side `sha256sum` of every file of the batch, run at the same nice/ionice, right after the batch was read (page-cache warm).  A file is
     'OK' only when  pod sha256 == the sha256 of the bytes received  AND the received byte count == the size listed by the pod's own `stat`.  A mismatch deletes the local file and exits 4.
  3. The stall watchdog is fed per 1 MiB chunk, not per tar member: S0's watchdog (180 s without a COMPLETED member) would kill a 1.6 GB DB at ~9 MiB/s (~180 s) mid-file.
  4. The disk guard counts ALLOCATED bytes (each file rounded up to the exFAT cluster, 1 MiB on D:) -- 4 jpgs of 0.2 MB occupy 4 MiB there.

CLI:  podcopy2.py --list LIST.tsv --dest ROOT --manifest OUT.tsv [--max-batch-mb 120] [--min-free-gb 20] [--p6-log /workspace/data/refe_p6/chain.log]
LIST.tsv lines: <bytes>\t<pod_basedir>\t<rel>      (rel is also the path under --dest)
MANIFEST lines: <rel>\t<bytes>\t<local_sha256>\t<pod_sha256>\tOK        (MISMATCH rows are written for a failed file)
Resumable: a dest file at its final name is complete (written as .part, then renamed); files in the manifest as OK are skipped; a complete file without a manifest row is re-hashed (local + pod).
Exit codes: 0 all done, 3 P6_GUARD in P6's chain log (retry in 10 min), 4 size/sha mismatch, 5 disk guard, 6 ssh/tar failure.
"""
import argparse
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time

PODSSH = shlex.split(os.environ["PODSSH"])
NICE = "nice -n 19 ionice -c3"
CLUSTER = 1 << 20          # D: is exFAT with 1 MiB clusters (MEASURED, Win32_Volume.BlockSize)
RC_GUARD, RC_NOLOG = 93, 94

_REDACT = [re.escape(t) for t in PODSSH if len(t) >= 4]


def red(s):
    """strip anything that identifies the pod (ssh prefix tokens, IPv4s, ports, key paths) from a string before it is logged."""
    if isinstance(s, bytes):
        s = s.decode("utf-8", "replace")
    for pat in _REDACT:
        s = re.sub(pat, "<ssh>", s)
    s = re.sub(r"\b\d{1,3}(\.\d{1,3}){3}\b", "<ip>", s)
    s = re.sub(r"([Pp]ort\s+)\d+", r"\1<n>", s)
    return s


def log(msg):
    print(time.strftime("%H:%M:%S"), red(msg), flush=True)


def guard_prefix(p6_log):
    """remote shell snippet: exit 93 when the last 20 lines of P6's chain log hold `P6_GUARD`, 94 when the log cannot be read."""
    return (f'G=$({NICE} tail -n 20 {shlex.quote(p6_log)} 2>/dev/null) || exit {RC_NOLOG}; '
            f'case "$G" in *P6_GUARD*) exit {RC_GUARD};; esac; ')


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def alloc(n):
    return ((n + CLUSTER - 1) // CLUSTER) * CLUSTER


def pod_sha(base, rels, p6_log):
    """-> (rc, {rel: sha}) from ONE guarded `sha256sum` at nice/ionice (xargs feeds the names from stdin)."""
    cmd = guard_prefix(p6_log) + f"cd {shlex.quote(base)} && {NICE} xargs -d '\\n' sha256sum -b --"
    r = subprocess.run(PODSSH + [cmd], input=("\n".join(rels) + "\n").encode(), capture_output=True, timeout=900)
    out = {}
    if r.returncode == 0:
        for ln in r.stdout.decode("utf-8", "replace").splitlines():
            m = re.match(r"^([0-9a-f]{64}) [ *](.+)$", ln)
            if m:
                out[m.group(2)] = m.group(1)
    else:
        log(f"pod sha256sum rc={r.returncode} stderr={r.stderr[:200]!r}")
    return r.returncode, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max-batch-mb", type=float, default=120.0)
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--p6-log", default="/workspace/data/refe_p6/chain.log")
    ap.add_argument("--max-batches", type=int, default=0, help="stop (rc 0) after this many batches; 0 = all (used for the pilot)")
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
            if len(p) == 5 and p[4] == "OK":
                done[p[0]] = p
    rows = [r for r in rows if r[2] not in done or not os.path.exists(os.path.join(a.dest, r[2]))]
    log(f"{len(rows)} files to do ({sum(r[0] for r in rows) / 1048576:.1f} MiB); {len(done)} already OK in the manifest")
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
    total_done, t_start = 0, time.time()
    for bi, batch in enumerate(batches, 1):
        if a.max_batches and bi > a.max_batches:
            log(f"PILOT STOP after {a.max_batches} batches")
            return 0
        base = batch[0][1]
        want_all = {r[2]: r for r in batch}
        todo, have = [], []
        for r in batch:
            d = os.path.join(a.dest, r[2])
            (have if os.path.exists(d) and os.path.getsize(d) == r[0] else todo).append(r)
        # ---- disk guard on ALLOCATED bytes of what this batch will add
        add = sum(alloc(r[0]) for r in todo)
        probe = a.dest if os.path.isdir(a.dest) else os.path.dirname(a.dest.rstrip("/\\")) or "."
        free = shutil.disk_usage(probe).free
        if free - add < a.min_free_gb * (1 << 30):
            log(f"STOP: disk guard: free {free / (1 << 30):.2f} GiB - batch {add / (1 << 20):.0f} MiB allocated would leave < {a.min_free_gb} GiB")
            return 5
        # ---- local sha of complete files we did not copy in this run
        local = {r[2]: sha256_file(os.path.join(a.dest, r[2])) for r in have}
        t0 = time.time()
        if todo:
            cmd = guard_prefix(a.p6_log) + f"cd {shlex.quote(base)} && {NICE} tar -cf - -T -"
            p = subprocess.Popen(PODSSH + [cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            p.stdin.write(("\n".join(r[2] for r in todo) + "\n").encode())
            p.stdin.close()
            want = {r[2]: r for r in todo}
            stall = {"t": time.time(), "done": False}

            def _wd(p=p, stall=stall):
                while not stall["done"]:
                    time.sleep(5)
                    if time.time() - stall["t"] > 180:
                        log("WATCHDOG: no byte received for 180 s, killing the ssh client")
                        p.kill()
                        return
            threading.Thread(target=_wd, daemon=True).start()
            got = {}
            try:
                with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
                    for m in tf:
                        if not m.isfile() or m.name not in want:
                            log(f"unexpected tar member {m.name!r} ignored")
                            continue
                        d = os.path.join(a.dest, m.name)
                        os.makedirs(os.path.dirname(d), exist_ok=True)
                        h, n = hashlib.sha256(), 0
                        with open(d + ".part", "wb") as fo:
                            src = tf.extractfile(m)
                            for b in iter(lambda: src.read(1 << 20), b""):
                                fo.write(b)
                                h.update(b)
                                n += len(b)
                                stall["t"] = time.time()
                        if n != want[m.name][0]:
                            log(f"MISMATCH {m.name}: size {n} vs listed {want[m.name][0]}")
                            os.remove(d + ".part")
                            mf.write(f"{m.name}\t{n}\t{h.hexdigest()}\t-\tMISMATCH_SIZE\n")
                            mf.flush()
                            stall["done"] = True
                            p.kill()
                            return 4
                        os.replace(d + ".part", d)
                        got[m.name] = h.hexdigest()
            except tarfile.ReadError as e:
                stall["done"] = True
                p.wait()
                rc = p.returncode
                if rc == RC_GUARD or rc == RC_NOLOG:
                    log(f"STOP: P6_GUARD / unreadable chain log (remote rc {rc}); batch {bi}/{len(batches)} not started")
                    return 3
                log(f"tar read error {e}; remote rc={rc} stderr={p.stderr.read()[:200]!r}")
                return 6
            while p.stdout.read(1 << 20):          # drain the tar end padding, or the ssh client blocks on the pipe and never exits
                pass
            p.wait()
            stall["done"] = True
            if p.returncode in (RC_GUARD, RC_NOLOG):
                log(f"STOP: P6_GUARD / unreadable chain log (remote rc {p.returncode}); batch {bi}/{len(batches)} not started")
                return 3
            if p.returncode != 0 or set(got) != set(want):
                log(f"tar rc={p.returncode} got {len(got)}/{len(want)}; stderr={p.stderr.read()[:200]!r}")
                return 6
            local.update(got)
        # ---- pod-side sha256 of the whole batch, guarded, same nice/ionice
        rc, pod = pod_sha(base, [r[2] for r in batch], a.p6_log)
        if rc in (RC_GUARD, RC_NOLOG):
            log(f"STOP: P6_GUARD before the pod-side sha256 of batch {bi}/{len(batches)} (files stay copied, not yet OK in the manifest); remote rc {rc}")
            return 3
        if rc != 0 or set(pod) != set(want_all):
            log(f"pod sha256 rc={rc} got {len(pod)}/{len(want_all)}")
            return 6
        bad = [k for k in want_all if pod[k] != local[k]]
        for k in want_all:
            ok = k not in bad
            mf.write(f"{k}\t{want_all[k][0]}\t{local[k]}\t{pod[k]}\t{'OK' if ok else 'MISMATCH_SHA'}\n")
        mf.flush()
        if bad:
            for k in bad:
                log(f"MISMATCH_SHA {k}: local {local[k][:12]} pod {pod[k][:12]}; local file deleted so a re-run re-copies it")
                os.remove(os.path.join(a.dest, k))
            return 4
        dt = time.time() - t0
        nb = sum(r[0] for r in batch)
        log(f"batch {bi}/{len(batches)} ({base.rsplit('/', 1)[-1]}): {len(batch)} files ({len(todo)} copied) {nb / 1048576:.1f} MiB in {dt:.1f}s = {nb / 1048576 / max(dt, 1e-6):.1f} MiB/s, "
            f"size = pod stat, local sha = pod sha")
        total_done += len(batch)
    log(f"ALL DONE {total_done} files in {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
