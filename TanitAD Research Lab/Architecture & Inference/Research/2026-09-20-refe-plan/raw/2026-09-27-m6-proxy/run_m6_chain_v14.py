#!/usr/bin/env python3
"""The M6 proxy chain, exactly as PREREG_M6_YAWLOSS (blob de651f51) registers it -- the record of what ran and when.

GPU stages (each behind the FREE-MEMORY REFUSAL; never touching another session's process):
  A  train cache  (already launched 17:01 by hand; this waits for it, and resumes it if its 23:10 deadline stopped it)
  B  eval cache   fp32, W3's 200 + Amendment 5's 923, via the planner's own input path  (concurrent with C)
  C  G3 (5 steps at lr 0), then W0, P0, W1, P1 (50 zero-LR + 403 steps each)          (one arm at a time)
  D  decodes: base, G3, W0, W1, P0, P1 (fp32, batch 1)
PAUSE RULE (PI, via the coordinator): no GPU stage runs across 23:15 -- a stage is started only if its estimated end
is before 23:14; GPU work resumes only after snapshot 017's eval has finished its GPU phase
(points/sub200_ep017/1_seam.log carries ZZSEAM_OK, or points/sub200_ep017.json exists).
CPU stages afterwards: harness seams (gating first), families6, analyze -> the verdict.
Log: raw/2026-09-27-m6-proxy/chain.log (append); timings: chain_times.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
ROOT = PKG / "raw" / "2026-09-27-m6-proxy"
ARMS = ROOT / "arms"
D = Path("D:/Projects/TanitAD/data/refe_proxy")
TC = D / "cache_train_ep015"
# 20:30: D: reads are capped at ~40 MB/s (MEASURED: both 14.2 GB hash passes ~352 s; W0 at 37.6 MB/s, disk 102 % busy)
# and the arms' 10.8 GB visual working set does not fit in RAM -> a BIT-IDENTICAL copy on the internal NVMe for the arms
TCF = Path("C:/Users/Admin/refe_proxy_fast/cache_train_ep015")
C_FLOOR_GIB = 10.0                             # coordinator 21:24 (PI 21:18, "accelerate"): this RE-COPY may leave
#                                                C: at >= 10 GB (was 20 for the first copy)
NO_COPY = True                                 # coordinator 22:28: no new C: copies tonight
C_EMERGENCY_GIB = 8.0                          # while the copy exists: C: < 8 GB -> delete it at once, arms read D:
ARM_CACHE = {"path": TCF}                      # falls back to D: (TC) if the copy is aborted
ECD = D / "cache_eval_ep015"
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
SNAP = "D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch015.pt"
TOK_W3 = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
          "A1_sub200_tokens.json")
TOK_A5 = str(PKG / "eval" / "raw" / "a5_confirm" / "a5_confirm_tokens.json")
P017 = Path("D:/Projects/TanitAD/data/refe_navtest/points")
PAUSE_AT = 23 * 60 + 14                        # minutes after midnight: nothing may still run on the GPU at 23:15
#   (kept: the PI's withdrawal, relayed ~20:50, was refused by this session's permission layer -- reported to the
#   coordinator; it stays until the user changes it)
LOG = ROOT / "chain.log"
TIMES = ROOT / "chain_times.json"
times: dict = {}
lock = threading.Lock()


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with lock:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)


def mark(k, v=None):
    with lock:
        times[k] = v or time.strftime("%Y-%m-%dT%H:%M:%S")
        json.dump(times, open(TIMES, "w"), indent=1)


def minutes_now():
    lt = time.localtime()
    m = lt.tm_hour * 60 + lt.tm_min
    return m + 24 * 60 if m < 12 * 60 else m          # after midnight counts as "later tonight"


VRAM_CAP_MIB = 6960                            # coordinator 20:52: total VRAM <= 7.3 GB (7.3e9 B = 6,962 MiB), so no
#                                                stage pushes the card into CUDA sysmem fallback


def total_mib():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    return int(o[0]) if o else -1


def free_mib():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    return int(o[0]) if o else -1


RAM_WINDOW_END = 20 * 60 + 20                # another session's CPU suite needs >= 6 GB free RAM until 20:20
RAM_MIN_GB = 6.5


def free_ram_gb():
    try:
        o = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                           capture_output=True, text=True).stdout.strip()
        return int(o) / 1024 / 1024
    except Exception:
        return -1.0


def s017_gpu_done():
    sl = P017 / "sub200_ep017" / "1_seam.log"
    try:
        return (P017 / "sub200_ep017.json").exists() or (sl.exists() and "ZZSEAM_OK" in sl.read_text(errors="ignore"))
    except OSError:
        return False


def gpu_gate(name, est_min, need_mib):
    """block until starting `name` is allowed: it must END before 23:14, or 017's GPU phase must be over; and the
    free-memory refusal must pass"""
    announced = False
    while True:
        now = minutes_now()
        window_ok = (now + est_min <= PAUSE_AT) or s017_gpu_done()
        if window_ok:
            f = free_mib()
            used = total_mib() - f
            ram = free_ram_gb()
            ram_ok = minutes_now() >= RAM_WINDOW_END or ram >= RAM_MIN_GB
            if used + need_mib <= VRAM_CAP_MIB and ram_ok:
                log(f"{name}: START (VRAM used {used} + need {need_mib} <= cap {VRAM_CAP_MIB} MiB; free RAM "
                    f"{ram:.1f} GB; est {est_min:.0f} min)")
                return
            if not announced:
                log(f"{name}: waiting (VRAM used {used} + need {need_mib} vs cap {VRAM_CAP_MIB} MiB; free RAM "
                    f"{ram:.1f} GB)")
                announced = True
        elif not announced:
            log(f"{name}: PAUSED by the 23:15 rule (est {est_min:.0f} min would cross it); resumes after 017's GPU phase")
            announced = True
        time.sleep(30)


def run(name, cmd, cwd=None, logfile=None, env=None):
    e = dict(os.environ, OMP_NUM_THREADS="6", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    e.update(env or {})
    t0 = time.time()
    mark(f"{name}_start")
    with open(logfile or (ROOT / "logs" / f"{name}.log"), "a", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, cwd=str(cwd or PKG), stdout=fh, stderr=subprocess.STDOUT, env=e)
    mark(f"{name}_end")
    log(f"{name}: EXIT {rc} after {(time.time() - t0) / 60:.1f} min")
    return rc, (time.time() - t0) / 60


def tail_has(p, token):
    try:
        return token in Path(p).read_text(encoding="utf-8", errors="ignore")[-4000:]
    except OSError:
        return False


def pid_alive(pid):
    o = subprocess.run(["powershell", "-NoProfile", "-Command", f"if (Get-Process -Id {pid} -ErrorAction "
                        f"SilentlyContinue) {{ 'ALIVE' }} else {{ 'GONE' }}"], capture_output=True, text=True).stdout
    return "ALIVE" in o


def stage_train_cache(build_pid):
    while pid_alive(build_pid):
        time.sleep(60)
    for attempt in range(4):
        mf = json.load(open(TC / "manifest.json")) if (TC / "manifest.json").exists() else {}
        if mf.get("frames") == 10000 and tail_has(D / "cache_train_ep015.log", "ZZCACHE_OK"):
            log(f"A train cache: COMPLETE, {mf['frames']} frames, {mf.get('visual_frames')} with visual_ctx")
            mark("A_train_cache_done")
            return True
        log(f"A train cache: incomplete ({mf.get('frames')} frames) -- resuming (attempt {attempt + 1})")
        gpu_gate("A_train_cache_resume", 60, 2300)
        run("A_train_cache_resume", [PY, "refe/proxy_cache.py", "build", "--manifest",
                                     "raw/2026-09-27-training-measures/proxy_manifest_10k.json", "--images",
                                     str(D / "pixels"), "--calib", str(D / "calib_table_pod_train_grow.json"),
                                     "--snapshot", SNAP, "--out", str(TC), "--amp", "bf16", "--device", "cuda",
                                     "--shard-frames", "256", "--deadline", "23:14", "--visual-for", "sets"],
            logfile=D / "cache_train_ep015.log")
    return False


def procs_matching(pattern):
    """python processes whose command line contains `pattern`, EXCLUDING this process and its parent (the venv
    launcher) -- a probe whose own command line carries the pattern would otherwise count itself"""
    me = f"{os.getpid()},{os.getppid()}"
    o = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Process | Where-Object { "
                        f"$_.CommandLine -match '{pattern}' -and $_.Name -match 'python' -and "
                        f"@({me}) -notcontains $_.ProcessId }} | Measure-Object).Count"],
                       capture_output=True, text=True).stdout.strip()
    return int(o) if o.isdigit() else -1


def stage_eval_cache(res):
    # the eval cache may already be building (launched by hand beside the train cache at 17:40); two writers on one
    # cache directory would corrupt it, so wait for any running builder to exit, then run the (resumable) builder
    while procs_matching("proxy_eval_cache.py") != 0:
        time.sleep(60)
    mf0 = json.load(open(ECD / "manifest.json")) if (ECD / "manifest.json").exists() else {}
    if mf0.get("frames") == 1123 and all(p_["C1_decode_vs_forward_max_abs"] == 0.0 for p_ in mf0.get("passes", [])):
        log("B eval cache: already COMPLETE (1123 tokens, manifest written) -- not rebuilt")
        mark("B_eval_cache_done")
        res["B"] = True
        return
    for attempt in range(4):
        gpu_gate("B_eval_cache", 45, 2300)
        rc, _m = run("B_eval_cache", [PY, "eval/proxy_eval_cache.py", "--snapshot", "015", "--out", str(ECD),
                                      "--tokens", f"{TOK_W3},{TOK_A5}", "--device", "cuda", "--shard-frames", "64",
                                      "--deadline", "23:14"], logfile=D / "cache_eval_ep015.log")
        mf = json.load(open(ECD / "manifest.json")) if (ECD / "manifest.json").exists() else {}
        if mf.get("frames") == 1123 and rc == 0:
            log(f"B eval cache: COMPLETE, 1123 tokens; passes {[(p['C1_decode_vs_forward_max_abs'], p['C2_forward_vs_table_m']) for p in mf['passes']]}")
            mark("B_eval_cache_done")
            res["B"] = True
            return
        log(f"B eval cache: rc {rc}, {mf.get('frames')} of 1123 tokens -- {'resuming' if attempt < 3 else 'GIVING UP'}")
    res["B"] = False


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def copy_hashed(src: Path, dst: Path) -> str:
    """copy src -> dst through a .part file, hashing the SOURCE bytes as they are read; returns that sha256"""
    h = hashlib.sha256()
    tmp = dst.with_name(dst.name + ".part")
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        for b in iter(lambda: fi.read(1 << 23), b""):
            h.update(b)
            fo.write(b)
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp, dst)
    return h.hexdigest()


def c_free_gib() -> float:
    return shutil.disk_usage("C:/").free / 2 ** 30


def abort_fast_copy(why: str) -> str:
    before = c_free_gib()
    shutil.rmtree(TCF.parent, ignore_errors=True)
    log(f"FAST COPY ABORTED ({why}); partial copy deleted, C: free {before:.2f} -> {c_free_gib():.2f} GiB; the arms "
        f"read D: instead")
    ARM_CACHE["path"] = TC
    fj = ROOT / "fast_copy.json"
    old = json.load(open(fj)) if fj.exists() else {}
    old[f"aborted_{time.strftime('%H%M%S')}"] = {"why": why, "c_free_gib_before_delete": round(before, 2),
                                                 "c_free_gib_after_delete": round(c_free_gib(), 2)}
    json.dump(old, open(fj, "w"), indent=1)
    return "fallback"


def stage_fast_copy() -> str:
    """every shard verified three ways (bytes read from D: == the manifest's sha256 == bytes read back from C:);
    frames / rows / manifest (LAST) by source-vs-copy hash. A verified file already present is not copied again.
    FLOOR: before each file and after it, C: must keep >= 20 GiB free, else abort, delete the partial copy, use D:."""
    if NO_COPY:                                       # coordinator 22:28: "No new C: copies tonight."
        log("FAST COPY: disabled (coordinator: no new C: copies tonight) -- the arms read D:")
        ARM_CACHE["path"] = TC
        return "fallback"
    mf = json.load(open(TC / "manifest.json"))
    total = sum((TC / sh[k]).stat().st_size for sh in mf["shards"] for k in ("scene", "visual"))
    if c_free_gib() - total / 2 ** 30 < C_FLOOR_GIB:
        return abort_fast_copy(f"the full copy ({total / 2 ** 30:.2f} GiB) would leave C: below {C_FLOOR_GIB} GiB")
    TCF.mkdir(parents=True, exist_ok=True)
    t0, nbytes, rec = time.time(), 0, []
    free0 = c_free_gib()
    log(f"FAST COPY: C: free {free0:.2f} GiB before (floor {C_FLOOR_GIB})")
    for sh in mf["shards"]:
        for kind in ("scene", "visual"):
            name, want = sh[kind], sh[f"{kind}_sha256"]
            dst, src = TCF / name, TC / name
            if dst.exists() and sha_file(dst) == want:
                rec.append([name, want, "already-present", want])
                continue
            if c_free_gib() - src.stat().st_size / 2 ** 30 < C_FLOOR_GIB:
                return abort_fast_copy(f"{name} would take C: below {C_FLOOR_GIB} GiB")
            src_sha = copy_hashed(src, dst)
            got = sha_file(dst)
            rec.append([name, want, src_sha, got])
            if not (src_sha == want == got):
                return abort_fast_copy(f"{name} MISMATCH source {src_sha[:12]} manifest {want[:12]} copy {got[:12]}")
            nbytes += dst.stat().st_size
            if c_free_gib() < C_FLOOR_GIB:
                return abort_fast_copy(f"C: below {C_FLOOR_GIB} GiB after {name}")
    for name in ("frames.jsonl", "rows.jsonl", "manifest.json"):
        src_sha = copy_hashed(TC / name, TCF / name)
        got = sha_file(TCF / name)
        rec.append([name, None, src_sha, got])
        if got != src_sha:
            return abort_fast_copy(f"{name} MISMATCH")
    free1 = c_free_gib()
    this = {"path": str(TCF), "source": str(TC), "files_manifest_source_copy_sha256": rec, "bytes_copied": nbytes,
            "seconds": round(time.time() - t0, 1), "c_free_gib_before": round(free0, 2),
            "c_free_gib_after": round(free1, 2), "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    fj = ROOT / "fast_copy.json"
    old = json.load(open(fj)) if fj.exists() else {}
    old[("recheck_" if nbytes == 0 else "copy_") + time.strftime('%H%M%S')] = this     # APPEND, never overwrite
    json.dump(old, open(fj, "w"), indent=1)
    log(f"FAST COPY: {len(mf['shards'])} shards + frames/rows/manifest -> {TCF}, all verified "
        f"({nbytes / 1e9:.2f} GB copied in {time.time() - t0:.0f} s); C: free {free1:.2f} GiB after")
    mark("FAST_COPY_DONE")
    return "ok"


GUARD = {"stop": False}


def emergency_guard():
    """every 15 s while the copy exists: C: free < 8 GiB -> stop the arm reading the copy, delete the copy, arms
    fall back to D: (the chain's retry re-runs the stopped arm from step 0 on D:; its partial dir is set aside)"""
    while not GUARD["stop"] and TCF.parent.exists():
        f = c_free_gib()
        if f < C_EMERGENCY_GIB:
            log(f"EMERGENCY: C: free {f:.2f} GiB < {C_EMERGENCY_GIB} -- stopping the arm on the copy, deleting it")
            ARM_CACHE["path"] = TC
            subprocess.run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process | Where-Object { "
                            "$_.CommandLine -match 'proxy_train.py' -and $_.CommandLine -match 'refe_proxy_fast' } | "
                            "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"],
                           capture_output=True)
            time.sleep(5)
            abort_fast_copy(f"EMERGENCY: C: free {f:.2f} GiB < {C_EMERGENCY_GIB}")
            return
        time.sleep(15)


def cleanup_fast_copy():
    GUARD["stop"] = True
    if not TCF.parent.exists():
        return
    before = c_free_gib()
    shutil.rmtree(TCF.parent, ignore_errors=True)
    after = c_free_gib()
    log(f"FAST COPY DELETED after its last reader: C: free {before:.2f} -> {after:.2f} GiB (D: keeps the original)")
    fc = json.load(open(ROOT / "fast_copy.json")) if (ROOT / "fast_copy.json").exists() else {}
    fc.update({"deleted_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "c_free_gib_before_delete": round(before, 2),
               "c_free_gib_after_delete": round(after, 2), "still_exists": TCF.parent.exists()})
    json.dump(fc, open(ROOT / "fast_copy.json", "w"), indent=1)


def arm_cmd(name, *extra):
    return [PY, "refe/proxy_train.py", "--cache", str(ARM_CACHE["path"]), "--sets", str(D / "sets"), "--snapshot", SNAP, "--out",
            str(ARMS / name), "--sets-work", str((TCF.parent if Path(ARM_CACHE["path"]) == TCF else D) / "work" / name),
            "--device", "cuda", "--amp", "bf16",
            # micro-batch 64 x accum 4 = the registered effective batch 256 and the SAME sample stream (the sampler is
            # batch-agnostic; the loss is batch-split invariant: train.py's per-micro-batch means / accum, and the scorer
            # sum over cov_norm x micro-batch / accum). Half the kernel launches of 32 x 8 -- the proxy is
            # launch-latency bound while another session's job time-slices the GPU (MEASURED 17:20 smoke: ~5 s/step).
            "--batch", "64", "--accum", "4", *extra]


def stage_arms(res):
    est = 30.0
    todo = [("G3_zero_lr", ["--lr-override", "0", "--zero-lr-steps", "0", "--steps", "5", "--yaw-loss", "wrapped"]),
            ("W0", ["--yaw-loss", "wrapped", "--seed", "0"]), ("P0", ["--yaw-loss", "plain", "--seed", "0"]),
            ("W1", ["--yaw-loss", "wrapped", "--seed", "1"]), ("P1", ["--yaw-loss", "plain", "--seed", "1"])]
    for name, extra in todo:
        # a proxy_train.py still running (an arm started by a replaced chain version) finishes first: never two
        # writers on one arm directory, and never two arms beside the eval cache under the VRAM cap
        while procs_matching("proxy_train.py") != 0:
            time.sleep(30)
        if (ARMS / name / "final.pt").exists():
            log(f"C {name}: already complete -- kept")
            continue
        # under the 7.3 GB VRAM cap an arm (~2.3 GB) cannot sit beside the eval cache (up to ~2.0 GB) and refav1's
        # resident 3.66 GB: a point-in-time gate would admit it while the cache is between forwards, so the arms
        # wait for the eval cache to FINISH (sequential, as reported to the coordinator 21:02)
        while "B" not in res:
            time.sleep(30)
        for attempt in range(3):
            gpu_gate(f"C_{name}", 3 if name.startswith("G3") else est, 2500)
            if (ARMS / name).exists() and not (ARMS / name / "final.pt").exists():
                stale = ARMS / f"{name}_stopped_{time.strftime('%H%M%S')}"
                (ARMS / name).rename(stale)                       # an incomplete arm is kept aside, never reused
                log(f"C {name}: an incomplete run moved aside to {stale.name}")
            rc, mins = run(f"C_{name}", arm_cmd(name, *extra, "--deadline", "23:14"))
            if (ARMS / name / "final.pt").exists():
                if not name.startswith("G3"):
                    est = max(mins * 1.15, 5.0)              # the next arm's start rule uses the MEASURED duration
                break
            log(f"C {name}: rc {rc}, no final.pt (attempt {attempt + 1})")
    res["C"] = all((ARMS / n / "final.pt").exists() for n, _ in todo)
    mark("C_arms_done")


DECODES = (("base", None), ("G3", "G3_zero_lr"), ("W0", "W0"), ("P0", "P0"), ("W1", "W1"), ("P1", "P1"))


def stage_decodes(res):
    while "B" not in res:
        time.sleep(30)
    if not res["B"]:
        res["D"] = False
        return
    for name, arm in DECODES:
        while procs_matching("proxy_eval.py run") != 0:        # a decode left by a replaced chain finishes first
            time.sleep(15)
        if (ROOT / "dumps" / f"{name}.json").exists():
            continue
        if arm:
            # summary.json is written AFTER final.pt -- a decode never reads a final.pt still being written
            while not (ARMS / arm / "summary.json").exists():
                if "C" in res:
                    log(f"D decode {name}: the arms stage ended without {arm}/summary.json")
                    res["D"] = False
                    return
                time.sleep(30)
        gpu_gate(f"D_decode_{name}", 8, 800)
        rc, _m = run(f"D_decode_{name}", [PY, "eval/proxy_eval.py", "run", "--eval-caches", str(ECD), "--arm",
                                          "none" if arm is None else str(ARMS / arm), "--name", name,
                                          "--out", str(ROOT)])
        if rc != 0:
            log(f"D decode {name}: FAILED rc {rc}")
            res["D"] = False
            return
    res["D"] = True


E2 = ROOT / "epoch2"
A1_BLOB = "6c0baa506c614212e5a285419c234ca3ed074c5e"          # PREREG_M6_YAWLOSS.md with AMENDMENT 1 appended
HASH_REGISTER = PKG / "eval" / "raw" / "SPEC_PREREG_HASH.txt"
E2_ARMS = (("P0e2", "P0", "plain", "0"), ("P1e2", "P1", "plain", "1"), ("W0e2", "W0", "wrapped", "0"),
           ("W1e2", "W1", "wrapped", "1"))
# the EXACT trainer bytes per run: epoch 1 (every epoch-1 arm, incl. the G5 re-run of W1) and AMENDMENT 1's epoch 2
TRAINER = PKG / "refe" / "proxy_train.py"
TRAINER_E1 = ROOT / "code_epoch1" / "proxy_train.py"
TRAINER_E2_SHA = "576bc83dceda80de"
TRAINER_E1_SHA = "d4ecc28c75a94a68"


def install_trainer(src: Path, want_prefix: str):
    """put the given trainer bytes at refe/proxy_train.py (the path every arm's argv names) and ASSERT the sha"""
    shutil.copy2(src, TRAINER)
    got = sha_file(TRAINER)
    if not got.startswith(want_prefix):
        raise SystemExit(f"trainer install: sha {got[:16]} != {want_prefix}")
    log(f"TRAINER: refe/proxy_train.py is now sha {got[:16]}")


def a1_registered() -> bool:
    try:
        return A1_BLOB in HASH_REGISTER.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def epoch1_cpu(out):
    while procs_matching("proxy_eval.py score") != 0:
        time.sleep(15)
    rc, _m = run("E_score_gating", [PY, "eval/proxy_eval.py", "score", "--out", out, "--only",
                                    "refe_m6_W0_on,refe_m6_W1_on,refe_m6_P0_off,refe_m6_P1_off", "--workers", "2"])
    run("E_families", [PY, "eval/proxy_eval.py", "families", "--out", out])
    run("F_analyze", [PY, "eval/proxy_eval.py", "analyze", "--out", out, "--train-cache", str(TC),
                      "--eval-caches", str(ECD), "--arms-root", str(ARMS)])
    mark("VERDICT_WRITTEN")
    log("EPOCH-1 VERDICT WRITTEN")
    run("E_score_rest", [PY, "eval/proxy_eval.py", "score", "--out", out, "--workers", "2"])
    run("F_analyze_final", [PY, "eval/proxy_eval.py", "analyze", "--out", out, "--train-cache",
                            str(TC), "--eval-caches", str(ECD), "--arms-root", str(ARMS)])


def set_aside(src: Path, dst: Path):
    for k in range(20):
        try:
            src.rename(dst)
            return
        except PermissionError:
            time.sleep(3)
    raise SystemExit(f"cannot set aside {src}")


def g5_fix_w1() -> bool:
    """G5 (argv equality within a seed pair) failed on seed 1 ONLY because W1 read the NVMe copy and P1 the D: original
    after the emergency guard (bit-identical data: both configs record manifest sha256 9dc92b43...). G5 is registered
    as an argv check, so it is satisfied BY ITS LETTER: W1 re-runs with P1's exact cache path. Nothing else changes;
    the superseded W1 and all its outputs are kept aside, never reused."""
    cw = json.load(open(ARMS / "W1" / "config.json"))["argv"] if (ARMS / "W1" / "config.json").exists() else None
    cp = json.load(open(ARMS / "P1" / "config.json"))["argv"]
    done_ok = (cw and cw[cw.index("--cache") + 1] == cp[cp.index("--cache") + 1]
               and (ARMS / "W1" / "summary.json").exists())
    if done_ok:
        if not (ROOT / "dumps" / "W1.json").exists():
            gpu_gate("G5FIX_decode_W1", 8, 800)
            rc, _m = run("G5FIX_decode_W1", [PY, "eval/proxy_eval.py", "run", "--eval-caches", str(ECD), "--arm",
                                             str(ARMS / "W1"), "--name", "W1", "--out", str(ROOT)])
            return rc == 0
        return True
    stamp = time.strftime("%H%M%S")
    sup = ROOT / "superseded" / f"W1_cachepath_{stamp}"
    sup.mkdir(parents=True, exist_ok=True)
    if (ARMS / "W1").exists():
        set_aside(ARMS / "W1", ARMS / f"W1_cachepath_{stamp}")
    for f in (ROOT / "dumps" / "W1.npz", ROOT / "dumps" / "W1.json", ROOT / "seams" / "refe_m6_W1_on.npz",
              ROOT / "seams" / "refe_m6_W1_off.npz", ROOT / "score_logs" / "refe_m6_W1_on.log",
              ROOT / "score_logs" / "refe_m6_W1_off.log", ROOT / "score" / "refe_m6_W1_on",
              ROOT / "score" / "refe_m6_W1_off", ROOT / "families" / "refe_m6_W1_on.json",
              ROOT / "families" / "refe_m6_W1_on.log"):
        if f.exists():
            shutil.move(str(f), str(sup / f.name))
    for lb in ("refe_m6_W1_on", "refe_m6_W1_off"):             # the no-space scoring work dir too
        for f in (Path("D:/Projects/TanitAD/data/refe_proxy/m6_score/epoch1/seams") / f"{lb}.npz",
                  Path("D:/Projects/TanitAD/data/refe_proxy/m6_score/epoch1/score") / lb):
            if f.exists():
                shutil.move(str(f), str(sup / f"work_{f.name}"))
    log(f"G5 FIX: W1 read {cw[cw.index('--cache') + 1] if cw else '?'}, P1 read {cp[cp.index('--cache') + 1]} -- W1 and "
        f"its outputs set aside to superseded/W1_cachepath_{stamp}; W1 re-runs on P1's path")
    ARM_CACHE["path"] = TC
    install_trainer(TRAINER_E1, TRAINER_E1_SHA)        # W1 must run the SAME trainer bytes as P1 (G4 and G5 intent)
    for attempt in range(2):
        gpu_gate("G5FIX_W1", 20, 2500)
        rc, _m = run("G5FIX_W1", arm_cmd("W1", "--yaw-loss", "wrapped", "--seed", "1", "--deadline", "23:14"))
        if (ARMS / "W1" / "summary.json").exists():
            break
        if (ARMS / "W1").exists():
            set_aside(ARMS / "W1", ARMS / f"W1_stopped_{time.strftime('%H%M%S')}")
    if not (ARMS / "W1" / "summary.json").exists():
        return False
    cw = json.load(open(ARMS / "W1" / "config.json"))["argv"]
    log(f"G5 FIX: W1 re-run done; cache path now {cw[cw.index('--cache') + 1]} (P1: {cp[cp.index('--cache') + 1]})")
    gpu_gate("G5FIX_decode_W1", 8, 800)
    rc, _m = run("G5FIX_decode_W1", [PY, "eval/proxy_eval.py", "run", "--eval-caches", str(ECD), "--arm",
                                     str(ARMS / "W1"), "--name", "W1", "--out", str(ROOT)])
    return rc == 0


def e2_cache_path() -> Path:
    """the ONE extension copy (coordinator 22:34; raw/.../fast_copy_e2.py) when verified, else the D: original. Decided
    ONCE for all four epoch-2 arms, so their G5 argv comparison sees one path."""
    t0 = time.time()
    while procs_matching("fast_copy_e2.py") != 0 and not (TCF.parent / "COPY_VERIFIED").exists() \
            and time.time() - t0 < 1800:
        time.sleep(15)                                           # the copy is still being written / verified
    use = (TCF.parent / "COPY_VERIFIED").exists() and (TCF / "manifest.json").exists()
    log(f"EPOCH 2: the four extension arms read {'the verified NVMe copy ' + str(TCF) if use else 'the D: original'}")
    return TCF if use else TC


def stage_epoch2(res):
    """AMENDMENT 1, A1.2: each epoch-1 arm continued for exactly one proxy epoch; decoded as soon as it completes"""
    if not a1_registered():                          # checked once, after the G5 fix (the window the coordinator was given)
        log(f"EPOCH 2: AMENDMENT 1 (blob {A1_BLOB[:12]}) is NOT in {HASH_REGISTER.name} -- the extension does not run "
            f"in this chain")
        res["E2"] = None
        return
    install_trainer(Path("C:/Users/Admin/AppData/Local/Temp/claude/D--Projects-TanitAD/"
                         "91effc67-8c1e-4b66-9a63-341a3109dfc1/scratchpad/proxy_train_epoch2_576bc83d.py"), TRAINER_E2_SHA)
    ARM_CACHE["path"] = e2_cache_path()
    (E2 / "dumps").mkdir(parents=True, exist_ok=True)
    for f in ("base.npz", "base.json", "G3.json"):              # the same untouched-snapshot decode (G2 / G3 inputs)
        if not (E2 / "dumps" / f).exists():
            shutil.copy2(ROOT / "dumps" / f, E2 / "dumps" / f)
    for name, src, loss, seed in E2_ARMS:
        while procs_matching("proxy_train.py") != 0:
            time.sleep(30)
        if not (ARMS / name / "final.pt").exists():
            for attempt in range(3):
                gpu_gate(f"E2_{name}", 40, 2500)
                if (ARMS / name).exists() and not (ARMS / name / "final.pt").exists():
                    stale = ARMS / f"{name}_stopped_{time.strftime('%H%M%S')}"
                    (ARMS / name).rename(stale)
                    log(f"E2 {name}: an incomplete run moved aside to {stale.name}")
                cmd = arm_cmd(name, "--yaw-loss", loss, "--seed", seed, "--init-from", str(ARMS / src / "final.pt"),
                              "--start-step", "5336", "--skip-samples", "115968", "--zero-lr-steps", "50",
                              "--steps", "403", "--max-ckpt-step", "4933", "--deadline", "23:14")
                rc, _m = run(f"E2_{name}", cmd)
                if (ARMS / name / "summary.json").exists():
                    break
                log(f"E2 {name}: rc {rc}, no summary.json (attempt {attempt + 1})")
        if not (ARMS / name / "summary.json").exists():
            res["E2"] = False
            return
        if not (E2 / "dumps" / f"{name}.json").exists():
            gpu_gate(f"E2_decode_{name}", 8, 800)
            rc, _m = run(f"E2_decode_{name}", [PY, "eval/proxy_eval.py", "run", "--eval-caches", str(ECD), "--arm",
                                               str(ARMS / name), "--name", name, "--out", str(E2)])
            if rc != 0:
                res["E2"] = False
                return
    res["E2"] = True


def epoch2_cpu():
    out = str(E2)
    run("E2_score_gating", [PY, "eval/proxy_eval.py", "score", "--out", out, "--arm-suffix", "e2", "--only",
                            "refe_m6_W0e2_on,refe_m6_W1e2_on,refe_m6_P0e2_off,refe_m6_P1e2_off", "--workers", "2"])
    run("E2_families", [PY, "eval/proxy_eval.py", "families", "--out", out, "--arm-suffix", "e2"])
    run("E2_analyze", [PY, "eval/proxy_eval.py", "analyze", "--out", out, "--arm-suffix", "e2", "--epoch1-out",
                       str(ROOT), "--train-cache", str(TCF if (TCF / "manifest.json").exists() else TC),
                       "--eval-caches", str(ECD), "--arms-root", str(ARMS)])
    mark("EPOCH2_VERDICT_WRITTEN")
    log("EPOCH-2 (AMENDMENT 1) OUTCOME WRITTEN")


def main() -> int:
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    build_pid = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    log(f"chain start (train-cache build PID {build_pid})")
    if not stage_train_cache(build_pid):
        log("A train cache: FAILED -- stopping the chain")
        return 2
    if "A verify: rc 0" in LOG.read_text(encoding="utf-8", errors="ignore"):
        log("A verify: already passed on D: at 20:18 (rc 0) -- not repeated")
    else:
        v = subprocess.run([PY, "refe/proxy_cache.py", "verify", "--out", str(TC)], cwd=str(PKG),
                           capture_output=True, text=True)
        log(f"A verify: rc {v.returncode} {v.stdout.strip()[-300:]}")
        if v.returncode != 0:
            return 2
    if stage_fast_copy() == "ok":
        v = subprocess.run([PY, "refe/proxy_cache.py", "verify", "--out", str(TCF)], cwd=str(PKG),
                           capture_output=True, text=True)
        log(f"A verify (C: copy): rc {v.returncode} {v.stdout.strip()[-200:]}")
        if v.returncode != 0:
            abort_fast_copy("verify of the C: copy failed")
        else:
            threading.Thread(target=emergency_guard, daemon=True).start()
            log(f"EMERGENCY guard on: every 15 s, C: < {C_EMERGENCY_GIB} GiB deletes the copy")
    res: dict = {}
    tb = threading.Thread(target=stage_eval_cache, args=(res,))
    tc = threading.Thread(target=stage_arms, args=(res,))
    td = threading.Thread(target=stage_decodes, args=(res,))
    tb.start()
    time.sleep(90)                                   # let the eval cache take its memory first
    tc.start()
    td.start()
    tb.join()
    tc.join()
    td.join()
    if not (res.get("B") and res.get("C") and res.get("D")):
        log(f"B/C/D incomplete: {res} -- stopping before the seams")
        return 3
    out = str(ROOT)
    def pre_score():
        while procs_matching("proxy_eval.py score") != 0:     # a scorer left by a replaced chain finishes first
            time.sleep(15)
        run("E_score_pre", [PY, "eval/proxy_eval.py", "score", "--out", out, "--only",
                            "refe_m6_W0_on,refe_m6_P0_off,refe_m6_P1_off", "--workers", "2"])
    pre = threading.Thread(target=pre_score)
    pre.start()                                      # the three seams the G5 fix does not touch, scored meanwhile
    if not g5_fix_w1():
        log("G5 FIX: the W1 re-run FAILED -- the epoch-1 analysis runs on what exists")
    e2 = threading.Thread(target=stage_epoch2, args=(res,))
    e2.start()                                       # AMENDMENT 1 on the GPU at once (CPU scoring runs beside it)
    pre.join()
    te1 = threading.Thread(target=epoch1_cpu, args=(out,))
    te1.start()                                      # the epoch-1 verdict never waits for the extension
    e2.join()                                        # AMENDMENT 1, GPU, before "ALL GPU STAGES DONE"
    mark("LAST_GPU_STAGE_EXIT")
    log("ALL GPU STAGES DONE -- the GPU is free of M6 work from now on")
    if res.get("E2"):
        epoch2_cpu()
    te1.join()
    cleanup_fast_copy()                              # its last readers (both analyses' G1 verify) are done
    log("chain complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
