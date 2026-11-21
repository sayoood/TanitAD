#!/usr/bin/env python3
"""Measure 5 (eval/PREREG_MEASURE5.md, blob c668b5b6): the GPU stages as ONE supervised chain in ONE window.

PI decisions (chat, 2026-09-27): "M5 right after M6" (~19:55); "fixes first" (~20:42): M6 and then M5 get the GPU
before snapshot 017's eval; the 23:10 cap is WITHDRAWN; the whole chain runs in one window; a stage whose MEASURED
projection would run past 04:30 is refused (018 lands ~06:15, 017's eval needs ~1.5 h before it).
  WAIT     until raw/2026-09-27-m6-proxy/chain.log carries "ALL GPU STAGES DONE" (a plain read of the file, every
           30 s -- never a grep of a PTY stream)
  STAGES   G1 timing (20 samples, writes nothing to the cache) -> G2 navtrain train (1,600) -> G3 navtest (200, the
           planner's path, all-or-nothing) -> G4 navtrain val (300) -> [wait for the CPU labels, GPU idle] -> C data
           (CPU) -> G5 scorer fine-tune (3 arms x 3 seeds, bf16 autocast = the live numerics) -> G6 eval (A0 + the 9
           fine-tuned scorers on the navtest cache)
  LIMIT    04:30: a stage starts only if its estimate from MEASURED rates ends by then; the sharded forwards carry a
           midnight-safe --deadline and stop BETWEEN shards; a HARD STOP at 04:30 kills any child still alive (its own
           process tree, by explicit PID).
  VRAM     this box does NOT fail on VRAM exhaustion -- the CUDA sysmem fallback silently spills into system RAM and the
           pagefile (MEASURED 2026-09-27 by the M6 smoke). So: TOTAL memory.used (all processes) stays <= 7,300 MiB:
           (a) a stage starts only if memory.free >= its need + (total - 7,300); (b) each child's torch allocator is
           capped (M5_VRAM_CAP_MIB -> set_per_process_memory_fraction) at 7,300 - memory.used at launch - 450 MiB of
           CUDA context, so an overrun raises an OOM instead of spilling; (c) a watchdog kills the stage when
           memory.used stays > 7,300 for 20 s, or when a navtrain shard runs > 2.5x slower than the MEASURED timing
           rate (the signature of a spill). The suspended refav1 process (another session's) is never touched.
  MARKER   finally: D:/Projects/TanitAD/data/refe_m5/M5_GPU_DONE -- created the moment the LAST GPU stage has exited,
           for ANY reason (success, failure, refusal, a crash of this chain); never before, never deleted. Snapshot 017's
           eval and the refav1 resume both wait for it.
Log: eval/raw/m5_effectiveness/gpu_chain.log (append); timings gpu_chain_times.json.

    python eval/m5_gpu_chain.py [--skip-wait] [--no-finetune]
    python eval/m5_gpu_chain.py --from-stage G5 --marker D:/Projects/TanitAD/data/refe_m5/M5_GPU_DONE_2
        (the fine-tune + eval only, after the 01:08 VRAM-watchdog kill; each stage RESERVES its VRAM at process start
         -- M5_VRAM_RESERVE_MIB -- so other jobs' start gates see it through the CPU preload)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))
M6LOG = PKG / "raw" / "2026-09-27-m6-proxy" / "chain.log"
M6TOKEN = "ALL GPU STAGES DONE"
M5 = Path("D:/Projects/TanitAD/data/refe_m5")
MARKER = M5 / "M5_GPU_DONE"
OUT = HERE / "raw" / "m5_effectiveness"
LOG = OUT / "gpu_chain.log"
TIMES = OUT / "gpu_chain_times.json"
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
NT_CACHE = Path("D:/Projects/TanitAD/data/refe_trunk_cache/d7c59f4f/navtest_sub200/manifest.json")
END = 28 * 60 + 30                       # 04:30 tonight, in "minutes after midnight, after-midnight counts +24 h"
VRAM_TOTAL_MAX = 7300                    # MiB, all processes (the coordinator's measured-safe ceiling)
CTX_MIB = 450                            # CUDA context + cuBLAS workspace, outside the torch allocator
N_TRAIN, N_VAL, N_TEST = 1600, 300, 200
NT_S_PER_TOKEN_EST = 3.9                 # ESTIMATED: the STOP probe MEASURED 7.46 s/token mean with TWO forwards/token
FT_EST_MIN = float(os.environ.get("M5_FT_EST_MIN", "15"))    # ESTIMATED; the first 50 steps' projection governs
EVAL_EST_MIN = 6.0
times: dict = {}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def mark(k: str, v=None) -> None:
    times[k] = v if v is not None else time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(times, open(TIMES, "w", encoding="utf-8"), indent=1)


def now_min() -> float:
    lt = time.localtime()
    m = lt.tm_hour * 60 + lt.tm_min + lt.tm_sec / 60.0
    return m + 24 * 60 if m < 12 * 60 else m


def hhmm(m: float) -> str:
    m = int(math.floor(m)) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def smi() -> tuple:
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    if not o:
        return -1, 10 ** 9, 8188
    f, u, t = (int(x) for x in o[0].split(","))
    return f, u, t


def kill_tree(pid: int) -> None:
    """Our OWN child's process tree, by explicit PID (the venv launcher spawns the real interpreter as a child)."""
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)


FWD_BATCH = {"n": 4}                     # onpolicy_dump.py's batch; 2 only as the declared OOM fallback


def env(cap_mib: int | None = None, reserve_mib: int | None = None):
    import eval_checkpoint as EC
    e = EC.env_driverl()
    e["PYTHONUNBUFFERED"] = "1"
    e["M5_FWD_BATCH"] = str(FWD_BATCH["n"])
    if cap_mib is not None:
        e["M5_VRAM_CAP_MIB"] = str(int(cap_mib))
    if reserve_mib is not None:
        e["M5_VRAM_RESERVE_MIB"] = str(int(reserve_mib))
    return e


def wait_m6(skip: bool) -> bool:
    if skip:
        log("WAIT skipped (--skip-wait)")
        return True
    log(f"WAIT for '{M6TOKEN}' in {M6LOG}")
    while True:
        try:
            if M6TOKEN in M6LOG.read_text(encoding="utf-8", errors="replace"):
                log("M6 reports its GPU stages done -- the M5 window opens")
                mark("m6_gpu_done_seen")
                return True
        except OSError as e:
            log(f"WAIT: could not read the M6 log ({e}); retrying")
        if now_min() >= END - 5:
            log("WAIT: 04:25 reached without M6's token -- no M5 GPU stage can run tonight")
            return False
        time.sleep(30)


def gate(name: str, est_min: float, need_mib: int, disk_gb: float = 0.0) -> int | None:
    """-> the allocator cap (MiB) for the child, or None (refused). Starts only if the estimate ends by 04:30, D: keeps
    >= 20 GB free after the stage's bytes, and memory.free covers the need while TOTAL stays <= 7,300 MiB."""
    import shutil
    free_d = shutil.disk_usage("D:/").free / 2 ** 30
    if free_d - disk_gb < 20.0:
        log(f"{name}: REFUSED -- D: has {free_d:.1f} GB free and the stage writes ~{disk_gb:.1f} GB (floor 20 GB)")
        return None
    waited = 0
    while True:
        n = now_min()
        if n + est_min > END:
            log(f"{name}: REFUSED -- est {est_min:.1f} min from {hhmm(n)} would end after 04:30")
            return None
        f, u, t = smi()
        reserve = t - VRAM_TOTAL_MAX
        if f - reserve >= need_mib:
            cap = VRAM_TOTAL_MAX - u - CTX_MIB
            log(f"{name}: START (est {est_min:.1f} min -> ends ~{hhmm(n + est_min)}; GPU free {f} MiB, used {u} MiB, "
                f"need {need_mib} + reserve {reserve}; allocator cap {cap} MiB)")
            mark(f"{name}_vram_at_start", {"free": f, "used": u, "cap": cap, "need": need_mib})
            return cap
        if waited % 5 == 0:
            log(f"{name}: waiting for GPU memory (free {f} MiB, used {u} MiB; need {need_mib} + reserve {reserve})")
        waited += 1
        time.sleep(60)


def run(name: str, args: list, logfile: Path, cap: int, watch=None, slow_rate=None, reserve=None) -> tuple:
    """Run one stage under the watchdog: HARD STOP at 04:30, the VRAM ceiling, the spill-rate rule, and `watch`."""
    t0 = time.time()
    mark(f"{name}_start")
    with open(logfile, "a", encoding="utf-8") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} cap {cap} MiB :: {' '.join(args)}\n")
        fh.flush()
        p = subprocess.Popen([PY] + args, cwd=str(PKG), env=env(cap, reserve), stdout=fh, stderr=subprocess.STDOUT)
    log(f"{name}: PID {p.pid}, log {logfile}")
    reason, over, peak_used = None, 0, 0
    while p.poll() is None:
        time.sleep(10)
        _f, u, _t = smi()
        peak_used = max(peak_used, u)
        over = over + 1 if u > VRAM_TOTAL_MAX else 0
        text = ""
        try:
            text = logfile.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
        if now_min() >= END:
            reason = "HARD STOP 04:30"
        elif over >= 2:
            reason = f"VRAM total {u} MiB > {VRAM_TOTAL_MAX} for 20 s (spill risk)"
        elif slow_rate is not None:
            rates = [float(x) for x in re.findall(r"samples in \d+ s \(([\d.]+) s/sample\)", text)]
            if rates and rates[-1] > 2.5 * slow_rate:
                reason = f"shard rate {rates[-1]:.2f} s/sample > 2.5 x the timing rate {slow_rate:.2f} (spill signature)"
        if not reason and watch is not None:
            reason = watch(text)
        if reason:
            log(f"{name}: {reason} -- killing PID {p.pid} (tree)")
            kill_tree(p.pid)
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                pass
            break
    rc = p.returncode
    mins = (time.time() - t0) / 60
    mark(f"{name}_end")
    mark(f"{name}_minutes", round(mins, 2))
    mark(f"{name}_rc", rc)
    mark(f"{name}_peak_vram_used_total_mib", peak_used)
    log(f"{name}: EXIT rc {rc} after {mins:.1f} min; peak total VRAM used {peak_used} MiB"
        + (f" ({reason})" if reason else ""))
    return rc, mins, reason


def tail_text(p: Path, n: int = 20000) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")[-n:]
    except OSError:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-wait", action="store_true")
    ap.add_argument("--no-finetune", action="store_true")
    ap.add_argument("--from-stage", default="G1", choices=("G1", "G5"))
    ap.add_argument("--marker", default=str(MARKER))
    ap.add_argument("--ft-reserve-mib", type=int, default=1800)
    ap.add_argument("--eval-reserve-mib", type=int, default=1200)
    ap.add_argument("--wait-token", default="", help="G5 mode: start only once this token is in --wait-file")
    ap.add_argument("--wait-file", default=str(M6LOG))
    ap.add_argument("--wait-until", default="02:30", help="G5 mode: give up (no GPU stage, no marker) at this time")
    a = ap.parse_args()
    marker = Path(a.marker)
    OUT.mkdir(parents=True, exist_ok=True)
    logs = M5 / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if TIMES.exists():
        times.update(json.load(open(TIMES, encoding="utf-8")))
    log(f"chain start (PID {os.getpid()}); from {a.from_stage}; limit 04:30; VRAM total <= {VRAM_TOTAL_MAX} MiB; "
        f"marker {marker}")
    if marker.exists():
        log("the marker already exists -- refusing to run GPU stages after it (it signals that M5's GPU work is over)")
        return 3
    ran_gpu = False
    try:
        if a.from_stage == "G5":
            if a.wait_token and not wait_sequenced(a):
                marker_ok["write"] = False
                return 0
            ran_gpu = True
            return finetune_and_eval(a, logs)
        if not wait_m6(a.skip_wait):
            return 0
        # ------------------------------------------------------------------ G1 timing
        cap = gate("G1_timing", 4.0, 2600, 0.1)
        if cap is None:
            return 0
        ran_gpu = True
        rc, mins, _ = run("G1_timing", ["eval/m5_forward.py", "timing", "--limit", "20"], logs / "fwd_timing.log", cap)
        if rc != 0 and "out of memory" in tail_text(logs / "fwd_timing.log", 6000).lower():
            FWD_BATCH["n"] = 2
            mark("fwd_batch_fallback", 2)
            log("G1_timing: OOM under the allocator cap at batch 4 -> DECLARED fallback: batch 2 for every navtrain forward")
            cap = gate("G1_timing_b2", 4.0, 2000, 0.1)
            if cap is None:
                return 0
            rc, mins, _ = run("G1_timing_b2", ["eval/m5_forward.py", "timing", "--limit", "20"], logs / "fwd_timing.log",
                              cap)
        m = re.search(r"TIMING (\d+) samples in ([\d.]+) s = ([\d.]+) s/sample; GPU peak ([\d.]+) GB",
                      tail_text(logs / "fwd_timing.log"))
        if rc != 0 or not m:
            log("G1_timing: FAILED (no TIMING line) -- stopping the GPU chain")
            return 1
        rate, peak = float(m.group(3)), float(m.group(4))
        need = int(peak * 1024 * 1.15 + CTX_MIB)
        load_min = max(0.5, mins - 20 * rate / 60)
        mark("rate_s_per_sample", rate)
        mark("gpu_peak_gb", peak)
        mark("load_min", round(load_min, 2))
        log(f"G1_timing: MEASURED {rate:.3f} s/sample (includes warm-up), torch peak {peak:.2f} GB -> need {need} MiB; "
            f"model load ~{load_min:.1f} min")

        def sharded(name, split, n):
            est_full = load_min + n * rate / 60 + 1.0
            capx = gate(name, est_full, need, n * 4.05 / 1024)
            if capx is None:
                return None
            return run(name, ["eval/m5_forward.py", "navtrain", "--split", split, "--deadline", hhmm(END - 3)],
                       logs / f"fwd_navtrain_{split}.log", capx, slow_rate=rate)

        # ------------------------------------------------------------------ G2 navtrain train
        r2 = sharded("G2_navtrain_train", "train", N_TRAIN)
        if r2 is None or r2[0] != 0:
            log(f"G2: {'refused' if r2 is None else 'rc %s' % r2[0]} -- stopping the GPU chain")
            return 1 if r2 is not None else 0
        rates = [float(x) for x in re.findall(r"samples in \d+ s \(([\d.]+) s/sample\)",
                                              tail_text(logs / "fwd_navtrain_train.log", 40000))]
        if rates:
            rate = sorted(rates)[len(rates) // 2]
            mark("rate_s_per_sample_train_median_shard", rate)
            log(f"G2: MEASURED median shard rate {rate:.3f} s/sample over {len(rates)} shards")

        # ------------------------------------------------------------------ G3 navtest (all-or-nothing)
        est_nt = load_min + 1.5 + N_TEST * NT_S_PER_TOKEN_EST / 60
        capn = gate("G3_navtest", est_nt * 1.2, max(need, 2400), N_TEST * 8.0 / 1024)
        if capn is not None:
            def nt_watch(text):
                pr = re.findall(r"NTPROG (\d+) tokens ([\d.]+) s", text)
                if not pr:
                    return None
                k, el = int(pr[-1][0]), float(pr[-1][1])
                if k >= 12:
                    end = now_min() + (N_TEST - k) * (el / k) / 60 + 1.0
                    if end > END:
                        return f"projected end {hhmm(end)} after 04:30 ({k} tokens in {el:.0f} s)"
                return None
            run("G3_navtest", ["eval/m5_forward.py", "navtest", "--deadline", hhmm(END - 2)], logs / "fwd_navtest.log",
                capn, watch=nt_watch)
        # ------------------------------------------------------------------ G4 navtrain val
        r4 = sharded("G4_navtrain_val", "val", N_VAL)
        # ------------------------------------------------------------------ labels (CPU) -> data (CPU) -> G5 -> G6
        if a.no_finetune:
            log("G5: skipped (--no-finetune)")
            return 0
        q = M5 / "queue"
        if not all((q / f"FORWARD_DONE_{s}").exists() for s in ("train", "val")):
            log("G5: a navtrain forward is incomplete -- the labels cannot complete; the fine-tune is left for later")
            return 0
        summ = M5 / "label_summary.json"
        n_chunks = sum(1 for x in os.listdir(q) if x.startswith("props_r0_") and ".jsonl" in x)
        hist = []
        mark("label_wait_start")
        while not summ.exists():
            done = sum(1 for x in os.listdir(q) if x.startswith("props_r0_") and ".jsonl.done_" in x)
            t = time.time()
            hist.append((t, done))
            old = [h for h in hist if t - h[0] >= 360]
            if old and done > old[-1][1]:
                per_min = (done - old[-1][1]) / ((t - old[-1][0]) / 60)
                eta = now_min() + (n_chunks - done) / per_min
                if eta + 2.0 + FT_EST_MIN + EVAL_EST_MIN > END:
                    log(f"G5: labels {done}/{n_chunks} chunks at {per_min:.2f}/min -> done ~{hhmm(eta)}; fine-tune + "
                        f"eval would end after 04:30 -- releasing the GPU now")
                    return 0
                if len(hist) % 5 == 1:
                    log(f"G5: waiting for the labels (GPU idle): {done}/{n_chunks} chunks, {per_min:.2f}/min -> ~{hhmm(eta)}")
            elif len(hist) % 5 == 1:
                log(f"G5: waiting for the labels (GPU idle): {done}/{n_chunks} chunks")
            if now_min() + 2.0 + FT_EST_MIN > END:
                log("G5: the labels are not complete and the 04:30 limit is near -- the fine-tune is left for later")
                return 0
            time.sleep(60)
        mark("label_wait_end")
        log(f"G5: labels complete: {summ.read_text(encoding='utf-8').strip()}")
        rc, _, _ = run("C_data_cpu", ["eval/m5_finetune_eval.py", "data"], logs / "ft_data.log", None)
        if rc != 0:
            log("C_data_cpu: FAILED")
            return 1
        return finetune_and_eval(a, logs)
    except Exception as e:                                        # noqa: BLE001  -- logged, then the finally
        log(f"CHAIN ERROR {type(e).__name__}: {e}")
        return 1
    finally:
        # THE MARKER: every GPU stage has exited (or none could run)
        if not marker_ok["write"]:
            log(f"MARKER NOT written: the sequencing token never came -- the coordinator decides ({marker})")
            return 0
        try:
            marker.touch(exist_ok=True)
            mark(f"marker_created:{marker.name}")
            log(f"MARKER created: {marker} ({'after the last GPU stage' if ran_gpu else 'no M5 GPU stage ran'})")
        except OSError as e:
            log(f"MARKER FAILED: {e}")


marker_ok = {"write": True}


def m6b_gpu_procs() -> int:
    """python processes of M6/M6b GPU stages (trainer, decode/eval run, cache builders) -- read-only probe"""
    ps = ("(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match "
          "'proxy_train.py|proxy_eval.py run|proxy_cache.py build|proxy_eval_cache.py' } | Measure-Object).Count")
    o = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.strip()
    return int(o) if o.isdigit() else -1


def wait_sequenced(a) -> bool:
    """Coordinator 2026-09-28 01:18: SEQUENCE, do not race. Start only when --wait-file carries --wait-token (a plain
    read of the file), no M6/M6b GPU process is alive, and used + the reservation <= 7,300 MiB. Give up at
    --wait-until (no GPU stage, no marker; the coordinator is told)."""
    hh, mm = (int(x) for x in a.wait_until.split(":"))
    until = hh * 60 + mm + (24 * 60 if hh < 12 else 0)
    log(f"SEQUENCED WAIT for '{a.wait_token}' in {a.wait_file}, no M6b GPU process, used + {a.ft_reserve_mib} <= "
        f"{VRAM_TOTAL_MAX}; give up at {a.wait_until}")
    k = 0
    while True:
        try:
            tok = a.wait_token in Path(a.wait_file).read_text(encoding="utf-8", errors="replace")
        except OSError:
            tok = False
        n = m6b_gpu_procs() if tok else -2
        f, u, t = smi()
        if tok and n == 0 and u + a.ft_reserve_mib <= VRAM_TOTAL_MAX:
            log(f"SEQUENCED WAIT over: token seen, 0 M6b GPU processes, used {u} MiB")
            mark("sequenced_start")
            return True
        if now_min() >= until:
            log(f"SEQUENCED WAIT: {a.wait_until} reached (token {tok}, M6b GPU procs {n}, used {u}) -- NOT starting")
            return False
        if k % 10 == 0:
            log(f"SEQUENCED WAIT: token {tok}, M6b GPU procs {n}, used {u} MiB")
        k += 1
        time.sleep(30)


def finetune_and_eval(a, logs) -> int:
    """G5 (the scorer fine-tune, GPU, bf16) then G6 (the eval, GPU, fp32); each RESERVES its VRAM at process start."""
    if True:

        def ft_watch(text):
            pr = re.findall(r"seed (\d+) ep (\d+) step (\d+): .*?  (\d+) s  local", text)
            if not pr:
                return None
            seed, _ep, step, el = (int(x) for x in pr[-1])
            if seed == 0 and step >= 50:
                per = float(el) / step
                steps_total = 3 * 4 * math.ceil(N_TRAIN / 16)
                end = now_min() + (steps_total - step) * per * 1.15 / 60 + 3.0
                if end > END:
                    return f"projected end {hhmm(end)} after 04:30 ({step} steps in {el} s)"
            return None
        capf = gate("G5_finetune", FT_EST_MIN, a.ft_reserve_mib)
        if capf is None:
            return 0
        r5 = run("G5_finetune", ["eval/m5_finetune_eval.py", "train", "--device", "cuda", "--seeds", "0,1,2",
                                 "--preload", "--force", "--deadline", hhmm(END - 2)], logs / "ft_train_gpu.log", capf,
                 watch=ft_watch, reserve=a.ft_reserve_mib)
        if r5[0] != 0:
            log("G5_finetune: did not complete -- the eval is not run")
            return 1
        if not NT_CACHE.exists():
            log("G6_eval: the navtest cache is incomplete -- the eval is left for later")
            return 0
        cape = gate("G6_eval", EVAL_EST_MIN, a.eval_reserve_mib)
        if cape is not None:
            run("G6_eval", ["eval/m5_finetune_eval.py", "eval", "--device", "cuda"], logs / "ft_eval_gpu.log", cape,
                reserve=a.eval_reserve_mib)
        return 0


if __name__ == "__main__":
    sys.exit(main())
