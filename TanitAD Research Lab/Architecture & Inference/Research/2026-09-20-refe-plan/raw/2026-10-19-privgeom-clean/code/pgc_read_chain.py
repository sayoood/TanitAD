"""PRIVGEOM-CLEAN READ CHAIN (DETACHED).  Started ONCE, after the seal, by a WMI `Win32_Process.Create` (`cmd /c "... < NUL >> run_clean/read.log 2>&1"`, CreateFlags 0x08000200).  CPU only; no GPU, no lock, no pod.

    python -B -W ignore pgc_read_chain.py

Order (MM ruling, run order; PREREG Revision 1):
  0. PREFLIGHT (label-free, before a CPU-hour is spent):  seal-check (+ the label-free input files), the registered branch table, `select`.
  1. EXTRACTION: exactly 2 `pg_extract.py work` workers (the FROZEN extractor), each STARTED only when psutil available RAM >= 6 GB (the frozen extractor adds its own 4 GB start / 3 GB running gates).
  2. JOIN (label-free) -> GATE: PHASE A label-free (C-JOIN, C-ID-EXTRACT, suites incl. C-GRAD, branch table, C-GRAD-REAL), PHASE B = the first label read (C-AGREE, A-ADMIT with its RED arm, C-REACH).  A failure STOPS the run; Phase A stops before any label is opened.
  3. FIT x 2 shards (the 15 fits incl. C-NULL x 5 and the C-POWER plant) -> ANALYZE (the call, controls first).
Markers (grep the log; poll with a script, never with model turns):  ZZPGC_START  ZZPGC_PREFLIGHT_OK  ZZPGC_EXTRACT_START  ZZPGC_PROGRESS  ZZPGC_EXTRACT_DONE  ZZPGC_JOIN_OK  ZZPGC_GATE_PASS  ZZPGC_FIT_DONE
              ZZPGC_ANALYZE_OK  and the LAST line:  ZZPGC_DONE <call>  |  ZZPGC_STOP <reason>.   Files: run_clean/CHAIN_DONE.txt | run_clean/CHAIN_STOP.txt.
Never `pkill -f`: the chain only ever signals ITS OWN worker children by PID.
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
RC = f"{ROOT}/run_clean"
SCAL = f"{RC}/scalars"
SUBSET = f"{RC}/subset_clean.json"
RUNPY, EXTRACT = f"{HERE}/pgc_run.py", f"{HERE}/pg_extract.py"
N_LOGS, N_WORKERS, MIN_GB = 76, 2, 6.0
STALE_MIN, MAX_RESPAWN = 150, 3
DONE_SLACK_S = 2.0          # filesystem mtime granularity: a done record counts if written at or after (worker start - this)


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def say(msg: str) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def avail() -> float:
    return psutil.virtual_memory().available / 2 ** 30


def wait_ram(tag: str) -> None:
    t0, last = time.time(), 0.0
    while avail() < MIN_GB:
        if time.time() - last > 600:
            say(f"ZZPGC_RAMWAIT {tag}: available {avail():.2f} GB < {MIN_GB} GB (waiting {time.time() - t0:.0f} s)")
            last = time.time()
        time.sleep(30)


def stop(reason: str, code: int = 1):
    say(f"ZZPGC_STOP {reason}")
    os.makedirs(RC, exist_ok=True)
    with open(f"{RC}/CHAIN_STOP.txt", "w", encoding="utf-8") as f:
        f.write(f"{now()} {reason}\n")
    raise SystemExit(code)


def base_env(**kw) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PGC_")}
    env.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", **kw)
    return env


def stage(name: str, args: list, **envkw) -> int:
    """one pgc_run.py stage, output appended to run_clean/stages.log; returns the exit code."""
    say(f"stage {name} ...")
    with open(f"{RC}/stages.log", "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {now()} {name} =====\n")
        lf.flush()
        r = subprocess.run([PY, "-B", "-W", "ignore", RUNPY] + args, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=base_env(**envkw), cwd=HERE)
    say(f"stage {name} rc={r.returncode}")
    return r.returncode


def alive_chain_pid():
    p = f"{RC}/CHAIN_PID.txt"
    if os.path.exists(p):
        try:
            pid = int(open(p).read().split()[0])
            pr = psutil.Process(pid)
            if pr.is_running() and any("pgc_read_chain" in c for c in pr.cmdline()):
                return pid
        except Exception:                                                                # noqa: BLE001
            return None
    return None


def other_workers() -> list:
    out = []
    for pr in psutil.process_iter(["pid", "cmdline"]):
        c = pr.info.get("cmdline") or []
        if any(x.endswith("pg_extract.py") for x in c) and any("subset_clean.json" in x and "pgclean" in x for x in c) and pr.info["pid"] != os.getpid():
            out.append(pr.info["pid"])
    return out


def counts():
    ls = os.listdir(SCAL) if os.path.isdir(SCAL) else []
    return (sum(1 for f in ls if f.endswith(".npz") and ".tmp." not in f), [f for f in ls if f.endswith(".lock")], [f for f in ls if f.endswith(".failed")])


def done_records() -> list:
    """[(file name, mtime)] of every WORKER_DONE_<pid>.txt in the scalar store.  On Windows the venv `python.exe` that Popen starts is a LAUNCHER: the interpreter that runs pg_extract.py (and writes
    WORKER_DONE_<its own pid>.txt) has a DIFFERENT pid from `Popen.pid`, so a record can never be looked up by the launcher's pid (OPS1: the first read stopped on exactly that)."""
    out = []
    if os.path.isdir(SCAL):
        for f in os.listdir(SCAL):
            if f.startswith("WORKER_DONE_") and f.endswith(".txt"):
                try:
                    out.append((f, os.path.getmtime(f"{SCAL}/{f}")))
                except OSError:
                    pass
    return out


def worker_finished(rc, t_start: float, records: list, claimed: set, n: int, locks: list, failed: list):
    """(finished, how, record claimed).  A worker that exited rc 0 is FINISHED iff a WORKER_DONE record written at/after its start (under ANY pid, not already claimed by another worker) exists, OR all N_LOGS
    are extracted with no lock held and no .failed marker.  A nonzero rc, or rc 0 with logs still missing and no done record, is a CRASH."""
    if rc != 0:
        return False, f"rc={rc}", None
    for name, mt in sorted(records, key=lambda r: r[1]):
        if name not in claimed and mt >= t_start - DONE_SLACK_S:
            return True, f"done record {name}", name
    if n >= N_LOGS and not locks and not failed:
        return True, f"all {N_LOGS} logs extracted, no lock held", None
    return False, "rc=0 without a done record and with logs missing", None


def lock_pid(path: str):
    try:
        return int(open(path).read().split()[0])
    except Exception:                                                                    # noqa: BLE001
        return None


def stale_lock(pid) -> bool:
    """True iff the lock's owner is gone (no such process, or a process that is not a pg_extract.py).  The lock records the INTERPRETER's pid (os.getpid() in the worker), never the launcher's Popen.pid."""
    if pid is None:
        return False                                                                     # empty/unreadable: possibly being written right now -> leave it
    try:
        pr = psutil.Process(pid)
        return not (pr.is_running() and any("pg_extract" in c for c in pr.cmdline()))
    except psutil.NoSuchProcess:
        return True
    except psutil.Error:
        return False


def spawn(k: int):
    wait_ram(f"worker{k}")
    logf = open(f"{RC}/worker{k}.log", "a", encoding="utf-8")
    t0 = time.time()                                                                     # OPS1: a done record counts for this worker only if written at/after this instant
    p = subprocess.Popen([PY, "-B", "-W", "ignore", EXTRACT, "work", "--subset", SUBSET, "--out", SCAL], stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         env=base_env(OMP_NUM_THREADS="1"), cwd=HERE)
    p.t_start = t0
    say(f"worker{k} started pid {p.pid} (available RAM {avail():.2f} GB)")
    return p


def main() -> int:
    os.makedirs(RC, exist_ok=True)
    bad_env = sorted(k for k in os.environ if k.startswith("PGC_"))
    if bad_env:
        stop(f"environment override(s) present {bad_env}: the read runs with NO PGC_* variable (the rehearsal bypasses are not for the read)")
    other = alive_chain_pid()
    if other:
        stop(f"another read chain is alive (pid {other})")
    if other_workers():
        stop(f"extraction worker(s) already running (pids {other_workers()}); refusing to start a second set")
    with open(f"{RC}/CHAIN_PID.txt", "w") as f:
        f.write(f"{os.getpid()} {now()}\n")
    for m in ("CHAIN_DONE.txt", "CHAIN_STOP.txt"):
        if os.path.exists(f"{RC}/{m}"):
            os.remove(f"{RC}/{m}")
    say(f"ZZPGC_START pid {os.getpid()} at {now()} (RAM available {avail():.2f} GB)")
    # ---------------------------------------------------------------- 0. preflight (label-free)
    if stage("preflight (seal-check + branch table + select)", ["preflight"]) != 0:
        stop("preflight failed (seal / branch table / select): no extraction started, no label opened; see run_clean/stages.log")
    say("ZZPGC_PREFLIGHT_OK")
    # ---------------------------------------------------------------- 1. extraction
    os.makedirs(SCAL, exist_ok=True)
    n0, locks0, failed0 = counts()
    if failed0:
        stop(f"stale .failed marker(s) from a previous attempt: {failed0}")
    for lk in locks0:                                                                    # locks of workers that are no longer alive (checked above: none is running)
        os.remove(f"{SCAL}/{lk}")
    say(f"ZZPGC_EXTRACT_START {N_WORKERS} workers; {n0}/{N_LOGS} logs already extracted; RAM start gate {MIN_GB} GB")
    t_ext = time.time()
    workers, respawns, claimed = {}, 0, set()
    if n0 >= N_LOGS:                                                                     # OPS1: a relaunch with every log already extracted (stale locks were removed above) spawns NO worker
        say(f"extraction already complete ({n0}/{N_LOGS}): no worker spawned")
    else:
        for k in range(1, N_WORKERS + 1):
            workers[k] = spawn(k)
            time.sleep(90)                                                               # let the first worker load its engine before the second asks for RAM
    last_state, last_change, last_report = None, time.time(), 0.0
    while True:
        n, locks, failed = counts()
        if failed:
            for p in workers.values():
                if p.poll() is None:
                    p.terminate()                                                        # our OWN children, by PID
            stop(f"extraction failed on {failed}: see run_clean/scalars/*.failed and worker logs")
        for k, p in list(workers.items()):
            rc = p.poll()
            if rc is None:
                continue
            ok, how, rec = worker_finished(rc, p.t_start, done_records(), claimed, n, locks, failed)
            if ok:
                if rec:
                    claimed.add(rec)
                say(f"worker{k} pid {p.pid} finished ({how})")
                del workers[k]
                continue
            for lk in locks:                                                             # a worker died: free the lock(s) whose owner is gone so a replacement can take that log
                try:
                    if stale_lock(lock_pid(f"{SCAL}/{lk}")):
                        os.remove(f"{SCAL}/{lk}")
                except Exception:                                                        # noqa: BLE001
                    pass
            respawns += 1
            say(f"worker{k} pid {p.pid} exited rc={rc} without finishing (respawn {respawns}/{MAX_RESPAWN})")
            if respawns > MAX_RESPAWN:
                stop(f"extraction worker crashed {respawns} times: see run_clean/worker{k}.log")
            workers[k] = spawn(k)
        if not workers:
            break
        st = (n, tuple(sorted(locks)))
        if st != last_state:
            last_state, last_change = st, time.time()
        if time.time() - last_change > STALE_MIN * 60:
            stop(f"no extraction progress for {STALE_MIN} min (logs {n}/{N_LOGS}, locks {locks})")
        if time.time() - last_report > 600:
            say(f"ZZPGC_PROGRESS logs {n}/{N_LOGS} in flight {len(locks)} workers alive {len(workers)} available RAM {avail():.2f} GB elapsed {(time.time() - t_ext) / 60:.0f} min")
            last_report = time.time()
        time.sleep(60)
    n, locks, failed = counts()
    if n < N_LOGS or locks or failed:
        stop(f"both workers finished but extraction is incomplete: logs {n}/{N_LOGS}, locks {locks}, failed {failed}")
    say(f"ZZPGC_EXTRACT_DONE {n}/{N_LOGS} logs in {(time.time() - t_ext) / 60:.0f} min")
    # ---------------------------------------------------------------- 2. join -> gate
    wait_ram("join")
    if stage("join", ["join"]) != 0:
        stop("join failed (label-free); see run_clean/stages.log")
    say("ZZPGC_JOIN_OK")
    wait_ram("gate")
    rc = stage("gate", ["gate"], OMP_NUM_THREADS="8")
    gp = f"{RC}/gate_report.json"
    if rc != 0:
        why = "see run_clean/gate_report.json"
        if os.path.exists(gp):
            g = json.load(open(gp, encoding="utf-8"))
            failed_checks = [k for k in ("C-JOIN", "C-ID-EXTRACT", "C-GRAD-REAL", "C-ADMIT", "C-REACH") if isinstance(g.get(k), dict) and g[k].get("pass") is False]
            bad_suites = [k for k, v in (g.get("suites") or {}).items() if v.get("rc") != 0]
            why = f"failed {failed_checks} suites {bad_suites} branch-table {g.get('C-RED_branch_table_failures')} labels_opened={g.get('labels_opened')}" + (" => NO-CALL(ADMIT)" if "C-ADMIT" in failed_checks else "")
        stop(f"GATE FAILED ({why}); retune NOTHING; a fresh agent reads run_clean/gate_report.json")
    say("ZZPGC_GATE_PASS")
    # ---------------------------------------------------------------- 3. fits (2 shards) -> analyze
    wait_ram("fit")
    procs = []
    for sh in (0, 1):
        lf = open(f"{RC}/fit{sh}.log", "a", encoding="utf-8")
        procs.append(subprocess.Popen([PY, "-B", "-W", "ignore", RUNPY, "fit", "--shard", str(sh), "--nshards", "2"], stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=base_env(OMP_NUM_THREADS="8"), cwd=HERE))
    rcs = [p.wait() for p in procs]
    if any(rcs):
        stop(f"fit shard failed rc {rcs}: see run_clean/fit0.log fit1.log")
    say("ZZPGC_FIT_DONE")
    if stage("analyze", ["analyze"], OMP_NUM_THREADS="8") != 0:
        stop("analyze failed: see run_clean/stages.log")
    res = json.load(open(f"{RC}/results/results.json", encoding="utf-8"))
    say(f"ZZPGC_ANALYZE_OK raw {res['raw_call']} companion {res['companion']['row_call']} void_null {res['void_null']} power {res['power']['fires_on_both_seeds']} admit_ok {res['admit_ok']} void_inputs {res['void_inputs']}")
    with open(f"{RC}/CHAIN_DONE.txt", "w", encoding="utf-8") as f:
        f.write(f"{now()} {res['final_call']}\n")
    say(f"ZZPGC_DONE {res['final_call']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as e:                                                           # noqa: BLE001
        import traceback
        say(f"ZZPGC_STOP exception {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
        with open(f"{RC}/CHAIN_STOP.txt", "w", encoding="utf-8") as f:
            f.write(f"{now()} exception {type(e).__name__}: {e}\n")
        raise SystemExit(1)
