#!/usr/bin/env python3
"""Detached chain for the SAFE-HEAD cache: fetch frames (background thread) -> GPU dump -> CPU labels -> merge, per log. Resumable; writes markers/chain.DONE.

    python sh_run.py [--workers 4]          (launched detached via run_safehead.cmd + WMI CreateFlags 0x08000200; log W/CHAIN.log)

Resume rule: a log is DONE iff markers/log_<log>.DONE exists AND out/<log>.npz re-hashes to the sha in out/<log>.merge.json AND its index rows exist.
Gates (as the TRK-SCORE chain): before each dump/label start, available RAM >= 4500 MB and free commit >= 8 GB (MM 2026-11-26: commit limit, not RAM, killed P det) on 5 consecutive 20 s samples.
GPU: sh_dump takes devbox_gpu.lock PER LOG (held only around that log's forwards). C1 weights: W/ckpt/C1/model_final.pt (sha-checked by sh_dump).
Deletions (own temp data, literal paths, logged): a finished log's extracted frames (re-creatable from the public archive) when F: has < 8 GB free.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sh_env as E  # noqa: E402
import sh_lib as L  # noqa: E402
import sh_merge as M  # noqa: E402

assert os.path.abspath(L.W).replace("\\", "/").rstrip("/") != "F:/safehead_cache", "x150 chain must run with SH_WORK=F:/safehead_cache/x150"
F_FLOOR_GB = 8.0
LOGF = L.wp("CHAIN.log")
MK = L.wp("markers")


def say(m):
    L.say(m, LOGF)


def memstat():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]
    s = MS()
    s.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys // (1 << 20), s.ullAvailPageFile // (1 << 20)      # MB (the original returned whole GB)


def memgate(tag, need_mb=None, need_commit=int(os.environ.get("SH_COMMIT_MB", "6000")), samples=5, every=20, give_up_s=86400):
    # MM 2026-11-27: the RAM floor is SH_NEED_MB (default 4500). Chrome held 15.6 GB of physical RAM and the chain waited 30+ min at
    # ~2.4 GB available with 14 GB commit free; the COMMIT floor (8 GB) is what prevents allocation failures, so it stays fixed.
    need_mb = int(os.environ.get("SH_NEED_MB", "4500")) if need_mb is None else need_mb
    ok, t0, n = 0, time.time(), 0
    while time.time() - t0 < give_up_s:
        a, c = memstat()
        ok = ok + 1 if (a >= need_mb and c >= need_commit) else 0
        if n % 15 == 0:
            say(f"memgate {tag}: available {a} MB, commit free {c} MB (ok {ok}/{samples})")
        n += 1
        if ok >= samples:
            return True
        time.sleep(every)
    return False


def run(cmd, env, log, cwd=None):
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n=== {time.strftime('%FT%TZ', time.gmtime())} {' '.join(cmd)[:300]}\n")
        f.flush()
        return subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=cwd)


def log_done(lg):
    mj = L.wp("out", f"{lg}.merge.json")
    if not (os.path.exists(L.wp("markers", f"log_{lg}.DONE")) and os.path.exists(mj)):
        return False
    r = json.load(open(mj))
    return os.path.exists(r["file"]) and L.sha256_file(r["file"]) == r["sha256"]


def fetcher(plan, ready, failed, passive=False):
    for ent in plan["logs"]:
        if ent["source"] == "local":
            rc = run([L.PY_T, os.path.join(HERE, "sh_fetch.py"), "local", ent["log"]], os.environ.copy(), L.wp("logs", "fetch.log"))
            if rc == 0:
                ready.add(ent["log"])
            else:
                failed.add(ent["log"])
                say(f"FETCH local {ent['log']} rc {rc}")
    for a in plan["archives"]:
        if all(log_done(l) for l in a["logs"]):
            ready.update(a["logs"])
            continue
        if passive:                                    # another process (--fetch-only) owns the downloads: wait for its marker, never stream the same archive twice
            t0 = time.time()
            while not os.path.exists(L.wp("markers", f"fetch_{a['name']}.DONE")) and time.time() - t0 < 6 * 3600:
                fj = L.wp("plan", f"fetch_{a['name']}.json")
                if os.path.exists(fj) and json.load(open(fj)).get("missing", 1) > 0 and time.time() - os.path.getmtime(fj) > 60 and not os.path.exists(L.wp("markers", "fetch_running")):
                    break
                time.sleep(20)
            if os.path.exists(L.wp("markers", f"fetch_{a['name']}.DONE")):
                ready.update(a["logs"])
                say(f"FETCHED {a['name']} (by the fetch-only process)")
            else:
                failed.update(a["logs"])
                say(f"FETCH {a['name']} not available (passive wait ended)")
            continue
        est = a.get("n_need_frames", 0) * 0.22 / 1024                       # GB of extracted frames
        tw = time.time()
        while shutil.disk_usage("F:/").free / 1e9 < F_FLOOR_GB + est + 1.0 and time.time() - tw < 6 * 3600:
            time.sleep(30)                                                  # keep F: >= 8 GB: wait for finished logs to free their frames
        for att in (1, 2, 3):
            if os.path.exists(L.wp("markers", f"fetch_{a['name']}.DONE")):
                rc = 0
                break
            say(f"FETCH {a['name']} ({a['bytes'] / 1e9:.2f} GB) attempt {att}")
            rc = run([L.PY_T, os.path.join(HERE, "sh_fetch.py"), "archive", a["name"]], os.environ.copy(), L.wp("logs", "fetch.log"))
            if rc == 0:
                open(L.wp("markers", f"fetch_{a['name']}.DONE"), "w").write(time.strftime("%FT%TZ"))
                break
        if rc == 0:
            ready.update(a["logs"])
            say(f"FETCHED {a['name']}")
        else:
            failed.update(a["logs"])
            say(f"FETCH {a['name']} FAILED rc {rc}")
    say("fetcher finished")


def label_and_merge(ent, workers):
    lg = ent["log"]
    if not memgate(f"label_{lg}"):
        return False
    if not os.path.exists(L.wp("lab", f"{lg}.lab.npz")):
        rc = run([L.PY_N, os.path.join(HERE, "sh_label.py"), "--dump", L.wp("dump", f"{lg}.dump.npz"), "--db", ent["db"], "--out", L.wp("lab", f"{lg}.lab.npz"), "--workers", str(workers)],
                 E.nenv(), L.wp("logs", f"label_{lg}.log"))
        if rc != 0:
            say(f"LABEL {lg} FAILED rc {rc}")
            return False
    M.merge(lg, "heldout" if ent["split"] == "heldout" else "train")
    open(L.wp("markers", f"log_{lg}.DONE"), "w").write(time.strftime("%FT%TZ"))
    say(f"DONE {lg} ({ent['split']})")
    free_gb = shutil.disk_usage("F:/").free / 1e9
    if True:                                                       # x150: always delete a finished log's extracted frames (re-creatable from the public archive)
        d = L.wp("frames", lg)
        if os.path.isdir(d):
            sz = sum(os.path.getsize(os.path.join(p, f)) for p, _, fs in os.walk(d) for f in fs)
            shutil.rmtree(d)
            say(f"DELETED {d} ({sz} B; re-creatable from the public archive; F: had {free_gb:.1f} GB free)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--passive-fetch", action="store_true", help="do not download: wait for the --fetch-only process's markers")
    ap.add_argument("--only", default="", help="comma list of logs: smoke on just these (archives filtered to them); the DONE markers carry over to the full run")
    ap.add_argument("--fetch-only", action="store_true", help="only fetch/extract the frames (no GPU, no labels), then exit")
    a = ap.parse_args()
    for d in ("markers", "logs", "dump", "lab", "out", "plan"):
        os.makedirs(L.wp(d), exist_ok=True)
    plan = json.load(open(L.wp("plan", "PLAN.json")))
    if a.only:
        keep = set(a.only.split(","))
        plan["logs"] = [e for e in plan["logs"] if e["log"] in keep]
        plan["archives"] = [dict(x, logs=[l for l in x["logs"] if l in keep]) for x in plan["archives"] if keep & set(x["logs"])]
        plan["n_tokens"] = sum(e["n_tokens"] for e in plan["logs"])
    by = {e["log"]: e for e in plan["logs"]}
    say(f"chain start pid {os.getpid()}; {len(by)} logs, {plan['n_tokens']} tokens; ckpt {L.C1_CKPT}")
    ready, failed = set(), set()
    if a.fetch_only:
        fetcher(plan, ready, failed)
        say(f"fetch-only done: ready {len(ready)} failed {len(failed)}")
        sys.exit(1 if failed else 0)
    ft = threading.Thread(target=fetcher, args=(plan, ready, failed, a.passive_fetch), daemon=True)
    ft.start()
    lab_threads: list[threading.Thread] = []
    pending = [l for l in by if not log_done(l)]
    dumped = {l for l in by if os.path.exists(L.wp("dump", f"{l}.dump.npz"))}
    started = set()
    while True:
        todo = [l for l in pending if not log_done(l) and l not in failed]
        if not todo:
            break
        batch = [l for l in todo if l in ready and l not in dumped]
        if batch:
            if not memgate("dump"):
                say("memgate gave up")
                sys.exit(8)
            jb = {"frames": L.wp("frames"), "yaw": L.YAW_TRAIN, "commands": L.COMMANDS,
                  "logs": [{"log": l, "db": by[l]["db"], "tokens": by[l]["tokens"], "split": by[l]["split"]} for l in batch]}
            jp = L.wp("plan", "job_dump.json")
            json.dump(jb, open(jp, "w"))
            say(f"DUMP batch {len(batch)} logs: {batch[0]} ...")
            rc = run([L.PY_D, os.path.join(HERE, "sh_dump.py"), "--job", jp, "--ckpt", L.C1_CKPT, "--expect-sha256", L.C1_SHA, "--out-dir", L.wp("dump"), "--jobname", "refe-safehead-dump"],
                     E.denv(), L.wp("logs", "dump.log"))
            if rc != 0:
                say(f"DUMP FAILED rc {rc}; stopping (see logs/dump.log)")
                open(L.wp("markers", "chain.FAILED"), "w").write(f"dump rc {rc}")
                sys.exit(1)
            for l in batch:
                if os.path.exists(L.wp("dump", f"{l}.dump.npz")):
                    dumped.add(l)
        for l in list(dumped):
            if l not in started and not log_done(l):
                started.add(l)
                th = threading.Thread(target=label_and_merge, args=(by[l], a.workers))
                th.start()
                lab_threads.append(th)
                th.join()                              # one label job at a time (RAM); the fetcher thread keeps downloading meanwhile
        if not batch and not [l for l in dumped if not log_done(l)]:
            time.sleep(30)
    ft.join(timeout=1)
    bad = [l for l in by if not log_done(l)]
    if bad:
        say(f"INCOMPLETE: {len(bad)} logs not done: {bad[:5]}")
        open(L.wp("markers", "chain.FAILED"), "w").write(f"incomplete {len(bad)}")
        sys.exit(1)
    open(L.wp("markers", "chain.DONE"), "w").write(time.strftime("%FT%TZ"))
    say("chain DONE")


if __name__ == "__main__":
    main()
