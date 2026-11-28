#!/usr/bin/env python3
"""Measure 5r STAGE 2: the GPU eval under the dev box's rules, then the registered readout + RESULT (CPU).

  wait    eval/raw/m5r/stage2_score_status.json with 0 failed (the 73 harness runs), then the GPU:
          * NEVER while C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active exists (coordinator, 2026-09-28);
          * the shared lock C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock taken EXCLUSIVELY (create-if-absent;
            wait if present) with {"job": "refe-v4r-stage2-eval", "pid", "win_pid_of_locker", "host", "acquired"},
            released in a finally only if it still holds our content;
          * nvidia-smi used + 1,200 MiB reservation <= 7,300 MiB; the child's allocator is capped and reserves at start.
  eval    `m5r_stage2.py eval --device cuda` (A0 + V3 / V3r / V4 / V4r x 3 seeds on the 923 contexts).
  readout `m5r_stage2.py readout` (both gating arms + Addendum 1's adoption rule), then `m5r_write_result.py --stage 2`.
Log: eval/raw/m5r/stage2_gpu.log.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))
import m5r_gpu as G  # noqa: E402  (smi, lock_take/lock_release pattern, env)

OUT = HERE / "raw" / "m5r"
LOG = OUT / "stage2_gpu.log"
CHAIN = Path("C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active")
RESERVE, VRAM_MAX, CTX = 1200, 7300, 450


def log(m):
    line = f"{time.strftime('%H:%M:%S')} {m}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def lock_take():
    rec = {"job": "refe-v4r-stage2-eval", "pid": os.getpid(), "win_pid_of_locker": os.getpid(), "host": "FREEDOM2035",
           "acquired": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds")}
    try:
        fd = os.open(str(G.LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    return rec


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    logs = Path("D:/Projects/TanitAD/data/refe_m5r/stage2")
    log(f"stage-2 GPU runner start (PID {os.getpid()})")
    st = OUT / "stage2_score_status.json"
    while not st.exists():
        time.sleep(60)
    s = json.load(open(st))
    if s.get("failed"):
        log(f"harness runs FAILED {s['failed']} -- no eval")
        return 1
    rec, k = None, 0
    try:
        while True:
            f, u, t = G.smi()
            if not CHAIN.exists() and u + RESERVE <= VRAM_MAX:
                rec = lock_take()
                if rec is not None:
                    f, u, t = G.smi()
                    if not CHAIN.exists() and u + RESERVE <= VRAM_MAX:
                        log(f"GPU START: lock taken {rec}; used {u} MiB")
                        break
                    G.lock_release(rec)
                    rec = None
            if k % 20 == 0:
                holder = G.LOCK.read_text(encoding="utf-8", errors="replace")[:160] if G.LOCK.exists() else None
                log(f"GPU WAIT: chain_active {CHAIN.exists()}, used {u} MiB, lock {holder}")
            k += 1
            time.sleep(30)
        rc = G.run("S2_eval", ["eval/m5r_stage2.py", "eval", "--device", "cuda"], logs / "eval_gpu.log",
                   cap=VRAM_MAX - u - CTX, reserve=RESERVE, gpu=True)
        log(f"S2_eval rc {rc}")
    finally:
        if rec is not None:
            G.lock_release(rec)
            log("lock released")
    subprocess.run([G.PY, "eval/m5r_stage2.py", "readout"], cwd=str(PKG), env=G.env(),
                   stdout=open(logs / "readout.log", "a", encoding="utf-8"), stderr=subprocess.STDOUT)
    subprocess.run([G.PY, "eval/m5r_write_result_stage2.py"], cwd=str(PKG), env=G.env(),
                   stdout=open(logs / "write_result.log", "a", encoding="utf-8"), stderr=subprocess.STDOUT)
    try:
        v = json.load(open(OUT / "readout_stage2.json", encoding="utf-8"))["decision"]["verdict"]
    except (OSError, KeyError, ValueError):
        v = "UNREADABLE"
    log(f"ZZM5R_STAGE2_VERDICT {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
