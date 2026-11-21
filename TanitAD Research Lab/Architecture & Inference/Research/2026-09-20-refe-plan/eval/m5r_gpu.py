#!/usr/bin/env python3
"""Measure 5r (eval/PREREG_MEASURE5R.md, blob a8241136): gates -> data -> the GPU fine-tune + eval -> Stage-1 readout.

  1 wait for <m5r>/label_summary.json (eval/m5r_label.py)
  2 CPU gates: m5r_gates.py gr2, gr34 (G-R3 + G-R4), gr5. ANY failure: no GPU stage runs (G-R3/G-R4 failing means v4r
    is NOT tested -- §5); the readout records the failed gate.
  3 CPU: m5_finetune_eval.py data --m5 <m5r>  (V3 = the pure REPAIRED sets, V4 = the served v5 sets at frac 0.5)
  4 GPU START RULE (coordinator, 2026-09-28 ~07:05, superseding the 06:35 rule): ALL GPU use goes through the shared lock
    C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock, in whatever order it grants -- taken EXCLUSIVELY (create-if-absent;
    if present, wait), holding {"job": "refe-v4r-finetune", "pid", "win_pid_of_locker", "host", "acquired"}; released
    in a finally ONLY if it still holds this content -- plus nvidia-smi used + the reservation <= 7,300 MiB.
    NO clock cutoff (the 12:30 stop was withdrawn). RESUMABLE per seed: the trainer skips a seed whose trainlog exists.
    GAP SCHEME (coordinator + Master Mind, binding 2026-09-28 ~07:15): while C:/Users/Admin/qland/work/refcv7/
    devbox_gpu.chain_active exists the lock is NOT taken, not even per seed. Windows: (1) today before the refcv7 chain
    starts -- the WHOLE remaining run must end by 14:30, no start after 13:45, no overlap with 13:10-13:30 (REFe 019's
    seam); (2) after the chain removed its marker (or from 04:00 tomorrow with no marker). Preload (the fast path) iff
    >= 7 GB RAM is free at run time; otherwise per-batch reads.
  5 GPU: m5_finetune_eval.py train --m5 <m5r> --arms V3,V4 --device cuda --seeds 0,1,2 (bf16 autocast, the registered
    numerics; VRAM reserved at process start; NO --preload -- the dev box's free-RAM rule forbids a 6.3 GB preload, so
    the context is read per batch from the same memmapped arrays: identical values, only the I/O path differs)
    then m5r_eval.py eval --device cuda. Watchdog: VRAM total > 7,300 MiB for 20 s -> the stage is killed (its own
    tree, explicit PID).
  6 finally: the lock released (if ours), then <m5r>/M5R_GPU_DONE; CPU: m5r_eval.py readout, families.
Log: eval/raw/m5r/gpu_chain.log.
"""
from __future__ import annotations

import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))
M5R = Path("D:/Projects/TanitAD/data/refe_m5r")
A8LOG = PKG / "raw" / "2026-09-28-goal-clamp" / "a8_chain.log"
TOKEN = "ZZ_A8_GPU_DONE_2_ZZ"
LOCK = Path("C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock")
OUT = HERE / "raw" / "m5r"
LOG = OUT / "gpu_chain.log"
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
END = 12 * 60 + 30
VRAM_MAX, CTX = 7300, 450
RESERVE_FT, RESERVE_EV = 1800, 1200
N_STEPS = 3 * 4 * 100
CHAIN = Path("C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active")
W1_END, W1_LAST_START, SEAM_BLOCK = 14 * 60 + 30, 13 * 60 + 45, (13 * 60 + 10, 13 * 60 + 30)
START_DAY = time.localtime().tm_yday
WSTATE = {"seen_chain": False}


def free_ram_gb():
    o = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"], capture_output=True,
                       text=True).stdout.strip()
    return int(o) / 2 ** 20 if o.isdigit() else -1.0


def plan_run():
    """(preload, estimated minutes of the WHOLE remaining GPU run): seeds without a trainlog + the eval. Rates:
    preload 3.5 min/seed for 2 arms + 4 min preload (M5, MEASURED 267-299 s/seed for 3 arms, 220 s preload);
    per-batch reads ~13 min/seed (ESTIMATED: 12 passes x 6.3 GB at the MEASURED 40 MB/s of D:)."""
    left = sum(1 for sd in (0, 1, 2) if not (M5R / "ft" / f"trainlog_s{sd}.json").exists())
    pre = free_ram_gb() >= 7.0
    est = (4.0 + 3.5 * left if pre else 13.0 * left) + 1.5 + 3.0
    return pre, est


def window_ok(est):
    t = time.localtime()
    now = t.tm_hour * 60 + t.tm_min
    if CHAIN.exists():
        WSTATE["seen_chain"] = True
        return False, "chain_active marker present"
    if t.tm_yday == START_DAY and now < W1_END and not WSTATE["seen_chain"]:
        if now > W1_LAST_START:
            return False, "window 1: no start after 13:45"
        if now + est > W1_END:
            return False, f"window 1: the remaining run (est {est:.0f} min) would end after 14:30"
        if now < SEAM_BLOCK[1] and now + est > SEAM_BLOCK[0]:
            return False, "window 1: the run would overlap 13:10-13:30 (REFe 019's seam)"
        return True, "window 1"
    if WSTATE["seen_chain"] or (t.tm_yday != START_DAY and now >= 4 * 60):
        return True, "window 2"
    return False, "between windows: waiting for the refcv7 chain to start and finish (or 04:00 tomorrow)"


def log(m):
    line = f"{time.strftime('%H:%M:%S')} {m}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def now_min():
    t = time.localtime()
    return t.tm_hour * 60 + t.tm_min + t.tm_sec / 60


def hhmm(m):
    m = int(m)
    return f"{m // 60:02d}:{m % 60:02d}"


def smi():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    return tuple(int(x) for x in o[0].split(",")) if o else (-1, 10 ** 9, 8188)


def eval_procs():
    ps = ("(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match "
          "'eval_checkpoint.py|refe_navtest_seam.py' } | Measure-Object).Count")
    o = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.strip()
    return int(o) if o.isdigit() else -1


def env(cap=None, reserve=None):
    import eval_checkpoint as EC
    e = EC.env_driverl()
    e["PYTHONUNBUFFERED"] = "1"
    if cap is not None:
        e["M5_VRAM_CAP_MIB"] = str(int(cap))
    if reserve is not None:
        e["M5_VRAM_RESERVE_MIB"] = str(int(reserve))
    return e


def run(name, args, logfile, cap=None, reserve=None, watch=None, gpu=False):
    t0 = time.time()
    with open(logfile, "a", encoding="utf-8") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} cap {cap} reserve {reserve} :: {' '.join(args)}\n")
        fh.flush()
        p = subprocess.Popen([PY] + args, cwd=str(PKG), env=env(cap, reserve), stdout=fh, stderr=subprocess.STDOUT)
    log(f"{name}: PID {p.pid}")
    reason, over, peak = None, 0, 0
    while p.poll() is None:
        time.sleep(10)
        if gpu:
            _f, u, _t = smi()
            peak = max(peak, u)
            over = over + 1 if u > VRAM_MAX else 0
            if over >= 2:
                reason = f"VRAM total {u} MiB > {VRAM_MAX} for 20 s"
            elif watch is not None:
                try:
                    reason = watch(Path(logfile).read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    reason = None
        if reason:
            log(f"{name}: {reason} -- killing PID {p.pid} (tree)")
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                pass
            break
    log(f"{name}: EXIT rc {p.returncode} after {(time.time() - t0) / 60:.1f} min" +
        (f"; peak total VRAM {peak} MiB" if gpu else "") + (f" ({reason})" if reason else ""))
    return p.returncode if not reason else 99


def lock_take() -> dict | None:
    rec = {"job": "refe-v4r-finetune", "pid": os.getpid(), "win_pid_of_locker": os.getpid(),
           "host": socket.gethostname() or "FREEDOM2035",
           "acquired": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds")}
    try:
        fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    return rec


def lock_release(rec):
    try:
        cur = json.loads(LOCK.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log("LOCK: could not read it at release -- left untouched")
        return
    if cur == rec:
        LOCK.unlink()
        log("LOCK released (it held our content)")
    else:
        log(f"LOCK NOT released: it holds someone else's content {cur}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    logs = M5R / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log(f"m5r chain start (PID {os.getpid()}); GPU through the shared lock, no clock cutoff")
    while not (M5R / "label_summary.json").exists():
        time.sleep(30)
    log(f"labels complete: {(M5R / 'label_summary.json').read_text(encoding='utf-8')[:600]}")
    gate_rc = {}
    for g in ("gr2", "gr34", "gr5"):
        gate_rc[g] = run(f"C_gate_{g}", ["eval/m5r_gates.py", g], logs / f"gate_{g}.log")
    if any(v != 0 for v in gate_rc.values()):
        log(f"GATES FAILED {gate_rc} -- no GPU stage runs; the readout records the failure")
        subprocess.run([PY, "eval/m5r_eval.py", "readout"], cwd=str(PKG), env=env())
        return 1
    if run("C_data", ["eval/m5_finetune_eval.py", "data", "--m5", str(M5R)], logs / "ft_data.log") != 0:
        log("C_data FAILED")
        return 1
    rec, ran_gpu = None, False
    try:
        k = 0
        while True:
            try:
                tok = TOKEN in A8LOG.read_text(encoding="utf-8", errors="replace")
            except OSError:
                tok = False
            n_ev = eval_procs()
            f, u, t = smi()
            pre, est = plan_run()
            wok, why = window_ok(est)
            if wok and u + RESERVE_FT <= VRAM_MAX:
                rec = lock_take()
                if rec is not None:
                    f, u, t = smi()
                    if u + RESERVE_FT <= VRAM_MAX and not CHAIN.exists():
                        log(f"GPU START ({why}; preload {pre}; est {est:.0f} min): lock taken {rec}; used {u} MiB "
                            f"(A8 token {tok}, eval procs {n_ev}: informational under the lock rule)")
                        break
                    lock_release(rec)
                    rec = None
            if k % 10 == 0:
                holder = LOCK.read_text(encoding="utf-8", errors="replace")[:200] if LOCK.exists() else None
                log(f"GPU WAIT: {why}; est {est:.0f} min (preload {pre}); used {u} MiB, lock {holder}")
            k += 1
            time.sleep(30)
        ran_gpu = True

        f, u, t = smi()
        rc = run("G5_finetune", ["eval/m5_finetune_eval.py", "train", "--m5", str(M5R), "--arms", "V3,V4", "--device",
                                 "cuda", "--seeds", "0,1,2"] + (["--preload"] if pre else []), logs / "ft_train_gpu.log",
                 cap=VRAM_MAX - u - CTX, reserve=RESERVE_FT, gpu=True)
        if rc != 0:
            log("G5_finetune did not complete -- no eval")
            return 1
        f, u, t = smi()
        rc = run("G6_eval", ["eval/m5r_eval.py", "eval", "--device", "cuda"], logs / "eval_gpu.log",
                 cap=VRAM_MAX - u - CTX, reserve=RESERVE_EV, gpu=True)
        return 0 if rc == 0 else 1
    except Exception as e:                                          # noqa: BLE001
        log(f"CHAIN ERROR {type(e).__name__}: {e}")
        return 1
    finally:
        if rec is not None:
            lock_release(rec)
        if ran_gpu:
            (M5R / "M5R_GPU_DONE").touch(exist_ok=True)
            log(f"MARKER {M5R / 'M5R_GPU_DONE'} written (the GPU stages have exited)")
        if ran_gpu:
            for stg in ("readout", "families"):
                subprocess.run([PY, "eval/m5r_eval.py", stg], cwd=str(PKG), env=env(),
                               stdout=open(logs / f"{stg}.log", "a", encoding="utf-8"), stderr=subprocess.STDOUT)
                log(f"CPU {stg} done")
            subprocess.run([PY, "eval/m5r_write_result.py"], cwd=str(PKG), env=env(),
                           stdout=open(logs / "write_result.log", "a", encoding="utf-8"), stderr=subprocess.STDOUT)
            try:
                v = json.load(open(OUT / "readout_stage1.json", encoding="utf-8"))["decision"]["verdict"]
            except (OSError, KeyError, ValueError):
                v = "UNREADABLE"
            log(f"ZZM5R_STAGE1_VERDICT {v}")


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
