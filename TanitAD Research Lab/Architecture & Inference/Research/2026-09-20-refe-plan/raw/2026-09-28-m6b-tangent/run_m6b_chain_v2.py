#!/usr/bin/env python3
"""The M6b chain, exactly as PREREG_M6B_TANGENT (rev 2, registered blob 83e45c83) registers it.

Coordinator decisions: 00:14 (the order, the one C: copy, the module-overwrite assertion, the refav1 token) and
00:4x (option (a): NO dependency on M5's marker -- the VRAM gates arbitrate, M6b slots into M5's idle GPU windows; free
system RAM >= 3 GB before each arm; the token ONCE when M6b's last GPU stage exits; if the GPU work cannot finish by
~03:00, stop, write the token, report what is left).

  0  preconditions: the registered prereg blob; the selftests' tested sha256 == the package's bytes NOW
  1  CODE-IDENTITY BASELINE: sha256 of every module an arm or a decode imports -- asserted UNCHANGED before and after
     every GPU stage, and every arm's recorded code_sha256 must equal it ("nothing may overwrite a module a running
     arm has imported")
  2  the train cache: the verified epoch-2 NVMe copy, re-hashed file by file against the source manifest before any
     arm reads it (the coordinator's one authorised copy; >= 30 GiB free with it present, < 15 GiB -> emergency)
  3  GPU: G3_zero_lr (M6's gate, same argv) and Wt0 T0 Wt1 T1 Wt2 T2 (seed pairs first); decodes (base first, then
     each arm) beside the next arm. VRAM total <= 6,960 MiB (7.3 GB), free RAM >= 3 GB, the 23:15 rule (only after
     017's GPU phase). A stage starts only if its MEASURED estimate (+ its decode) ends by 03:00.
     ALL SIX arms read ONE cache path (G5 compares argv): if the copy is lost mid-run, every finished arm is retired
     and re-run on D:
  4  the refav1 token -- written ONCE, as its own line of raw/2026-09-27-m6-proxy/chain.log, the moment the LAST GPU
     stage exits, on every path, in a finally
  5  CPU: the 6 gating seams, families6, analyze-m6b -> RESULT_M6B_TANGENT.md; the copy deleted after its last reader
Log: raw/2026-09-28-m6b-tangent/chain.log (the token never appears there).

v2 (00:5x) replaces v1 (pid 41256), which admitted decode_base and G3_zero_lr in the SAME second on the SAME VRAM
reading: two point-in-time gates can each pass while their SUM crosses the cap (harmless at 00:50: 2,525 + 800 + 2,500
= 5,825 MiB; not in general). v2's gate is serialised and counts every stage it admitted in the last 120 s at its full
need until the allocation shows in nvidia-smi. v2 also adopts v1's orphans: a stage still running is waited for, and a
stage that completed meanwhile is KEPT, never set aside.
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
ROOT = PKG / "raw" / "2026-09-28-m6b-tangent"
ARMS = ROOT / "arms"
M6LOG = PKG / "raw" / "2026-09-27-m6-proxy" / "chain.log"     # the refav1 watcher reads THIS file for the token
TOKEN = "ZZ_M6B_" + "GPU_DONE_ZZ"                              # assembled: the literal sits in no source line either
LOG = ROOT / "chain.log"
TIMES = ROOT / "chain_times.json"
D = Path("D:/Projects/TanitAD/data/refe_proxy")
TC = D / "cache_train_ep015"
TCF = Path("C:/Users/Admin/refe_proxy_fast/cache_train_ep015")
ECD = D / "cache_eval_ep015"
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
SNAP = "D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch015.pt"
P017 = Path("D:/Projects/TanitAD/data/refe_navtest/points")
PREREG_BLOB = "83e45c8390f28c9e894c9e5aea0d51133939743c"
HASH_REGISTER = PKG / "eval" / "raw" / "SPEC_PREREG_HASH.txt"
STOP_AT = 27 * 60                     # 03:00, in minutes where after-midnight counts +24 h
PAUSE_AT = 23 * 60 + 14               # the 23:15 rule (kept): before 23:14, or after 017's GPU phase
VRAM_CAP_MIB = 6960
RAM_MIN_GB = 3.0
FLOOR_AFTER, EMERGENCY = 30.0, 15.0
MODULES = ("refe/proxy_train.py", "refe/measures.py", "refe/model.py", "refe/train.py", "refe/proxy_cache.py",
           "refe/planner.py", "refe/navtrain_scenarios.py", "eval/proxy_eval.py", "eval/refe_navtest_seam.py",
           "eval/eval_checkpoint.py")
G3 = ("G3_zero_lr", ["--lr-override", "0", "--zero-lr-steps", "0", "--steps", "5", "--yaw-loss", "wrapped"])
ARMS6 = (("Wt0", ["--yaw-loss", "wrapped", "--seed", "0"]),
         ("T0", ["--yaw-loss", "plain_tangent", "--tan-w", "0.1", "--seed", "0"]),
         ("Wt1", ["--yaw-loss", "wrapped", "--seed", "1"]),
         ("T1", ["--yaw-loss", "plain_tangent", "--tan-w", "0.1", "--seed", "1"]),
         ("Wt2", ["--yaw-loss", "wrapped", "--seed", "2"]),
         ("T2", ["--yaw-loss", "plain_tangent", "--tan-w", "0.1", "--seed", "2"]))
DECODE_ORDER = ["base"] + [n for n, _x in ARMS6]
_lock = threading.Lock()
GATE_LOCK = threading.Lock()
ADMITTED: list = []                   # (time admitted, need MiB) -- my own stages still ramping up their allocation
RAMP_S = 120
STATE = {"token_written": False, "cache": TC, "guard_stop": False, "arms_done": False, "stopped_by_clock": False,
         "est_arm": 8.0, "est_decode": 8.0, "gpu_started": False}
BASE: dict = {}


def log(msg):
    assert TOKEN not in msg
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with _lock:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)


def mark(k, v=None):
    with _lock:
        t = json.load(open(TIMES)) if TIMES.exists() else {}
        t[k] = v if v is not None else time.strftime("%Y-%m-%dT%H:%M:%S")
        json.dump(t, open(TIMES, "w"), indent=1)


def write_token(why):
    """the ONE line the refav1 watcher waits for -- written once, on every path"""
    with _lock:
        if STATE["token_written"]:
            return
        with open(M6LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {TOKEN}\n")
        STATE["token_written"] = True
    mark("refav1_token_written")
    log(f"refav1 token written, once, to {M6LOG.parent.name}/{M6LOG.name} ({why})")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def minutes_now():
    lt = time.localtime()
    m = lt.tm_hour * 60 + lt.tm_min
    return m + 24 * 60 if m < 12 * 60 else m


def free_mib():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    f, t = (int(x) for x in o[0].split(","))
    return f, t


def free_ram_gb():
    o = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                       capture_output=True, text=True).stdout.strip()
    return int(o) / 1024 / 1024 if o.isdigit() else -1.0


def c_free():
    return shutil.disk_usage("C:/").free / 2 ** 30


def s017_gpu_done():
    sl = P017 / "sub200_ep017" / "1_seam.log"
    try:
        return (P017 / "sub200_ep017.json").exists() or (sl.exists() and "ZZSEAM_OK" in sl.read_text(errors="ignore"))
    except OSError:
        return False


def procs(pattern):
    me = f"{os.getpid()},{os.getppid()}"
    o = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Process | Where-Object { "
                        f"$_.CommandLine -match '{pattern}' -and $_.Name -match 'python' -and "
                        f"@({me}) -notcontains $_.ProcessId }} | Measure-Object).Count"],
                       capture_output=True, text=True).stdout.strip()
    return int(o) if o.isdigit() else -1


class CodeChanged(Exception):
    pass


def assert_code(where):
    now = {m: sha(PKG / m) for m in MODULES}
    bad = [m for m in MODULES if now[m] != BASE[m]]
    if bad:
        raise CodeChanged(f"{where}: module(s) changed under the chain: {bad}")


def measures_selftest():
    """True: ROOT/selftest_measures.json passed on the chain's measures.py; None: not written yet and the suite is
    RUNNING; False: failed, tested other bytes, or neither written nor running"""
    p = ROOT / "selftest_measures.json"
    if p.exists() and procs("selftest_measures.py") == 0:
        try:
            st = json.load(open(p))
            return bool(st["tested_sha256"]["measures.py"] == BASE["refe/measures.py"] and not st["failed"])
        except Exception:
            return False
    return None if procs("selftest_measures.py") >= 1 else False


def gpu_gate(name, est_min, need_mib) -> bool:
    """True when `name` may start now; False when it can no longer END by 03:00 (the chain stops)"""
    said = False
    while True:
        now = minutes_now()
        if now + est_min > STOP_AT:
            log(f"{name}: NOT started -- est {est_min:.1f} min would end after 03:00")
            STATE["stopped_by_clock"] = True
            return False
        if now + est_min <= PAUSE_AT or s017_gpu_done():
            with GATE_LOCK:
                f, t = free_mib()
                used, ram = t - f, free_ram_gb()
                ADMITTED[:] = [(ta, nd) for ta, nd in ADMITTED if time.time() - ta < RAMP_S]
                ramp = sum(nd for _ta, nd in ADMITTED)
                if used + ramp + need_mib <= VRAM_CAP_MIB and ram >= RAM_MIN_GB:
                    ADMITTED.append((time.time(), need_mib))
                    log(f"{name}: START (VRAM used {used} + ramping {ramp} + need {need_mib} <= cap {VRAM_CAP_MIB} MiB; "
                        f"free RAM {ram:.1f} >= {RAM_MIN_GB} GB; est {est_min:.1f} min)")
                    return True
            if not said:
                log(f"{name}: waiting (VRAM used {used} + ramping {ramp} + need {need_mib} vs cap {VRAM_CAP_MIB} MiB; "
                    f"free RAM {ram:.1f} GB)")
                said = True
        elif not said:
            log(f"{name}: PAUSED by the 23:15 rule")
            said = True
        time.sleep(20)


def run(name, cmd):
    t0 = time.time()
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    mark(f"{name}_start")
    with open(ROOT / "logs" / f"{name}.log", "a", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, cwd=str(PKG), stdout=fh, stderr=subprocess.STDOUT,
                             env=dict(os.environ, OMP_NUM_THREADS="6", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"))
    mark(f"{name}_end")
    mins = (time.time() - t0) / 60
    log(f"{name}: EXIT {rc} after {mins:.1f} min")
    return rc, mins


# ------------------------------------------------------------------------------------------------ the copy
def record(key, val):
    fj = ROOT / "fast_copy.json"
    d = json.load(open(fj)) if fj.exists() else {}
    d[key] = val
    json.dump(d, open(fj, "w"), indent=1)


def delete_copy(why):
    before = c_free()
    for _ in range(20):
        shutil.rmtree(TCF.parent, ignore_errors=True)
        if not TCF.parent.exists():
            break
        time.sleep(3)
    log(f"COPY DELETED ({why}): C: free {before:.2f} -> {c_free():.2f} GiB; still exists: {TCF.parent.exists()}")
    record(f"deleted_{time.strftime('%H%M%S')}", {"why": why, "c_free_gib_before": round(before, 2),
                                                  "c_free_gib_after": round(c_free(), 2)})


def use_copy() -> bool:
    """the epoch-2 copy, KEPT as M6b's one authorised copy: every file re-hashed against the source manifest's sha256"""
    if not (TCF / "manifest.json").exists():
        log("the NVMe copy does not exist -- the arms read D: (no new copy is made)")
        return False
    f0 = c_free()
    if f0 < FLOOR_AFTER:
        log(f"C: {f0:.2f} GiB < {FLOOR_AFTER} GiB with the copy present -- deleting it, the arms read D:")
        delete_copy(f"C: {f0:.2f} < {FLOOR_AFTER} GiB floor")
        return False
    mf = json.load(open(TC / "manifest.json"))
    if sha(TCF / "manifest.json") != sha(TC / "manifest.json"):
        delete_copy("the copy's manifest differs from the source's")
        return False
    t0, rec = time.time(), []
    for sh in mf["shards"]:
        for k in ("scene", "visual"):
            name, want = sh[k], sh[f"{k}_sha256"]
            got = sha(TCF / name) if (TCF / name).exists() else None
            rec.append([name, want, got])
            if got != want:
                delete_copy(f"{name}: sha {str(got)[:12]} != manifest {want[:12]}")
                return False
    for name in ("frames.jsonl", "rows.jsonl"):
        if sha(TCF / name) != sha(TC / name):
            delete_copy(f"{name} differs from the source")
            return False
    v = subprocess.run([PY, str(PKG / "refe" / "proxy_cache.py"), "verify", "--out", str(TCF)], capture_output=True,
                       text=True)
    if v.returncode != 0:
        delete_copy(f"verify rc {v.returncode}")
        return False
    record(f"reused_{time.strftime('%H%M%S')}", {"path": str(TCF), "why": "the epoch-2 copy kept as M6b's one authorised "
                                                 "copy (coordinator 00:4x)", "files_manifest_copy_sha256": rec,
                                                 "seconds": round(time.time() - t0, 1), "c_free_gib": round(f0, 2),
                                                 "verify_rc": v.returncode})
    log(f"COPY RE-VERIFIED for M6b: {len(rec)} shard files == the manifest's sha256 + frames/rows == source + verify rc 0 "
        f"({time.time() - t0:.0f} s); C: free {f0:.2f} GiB; guard on (< {EMERGENCY} GiB deletes it)")
    threading.Thread(target=guard, daemon=True).start()
    return True


def guard():
    while not STATE["guard_stop"] and TCF.parent.exists():
        f = c_free()
        if f < EMERGENCY:
            log(f"EMERGENCY: C: {f:.2f} GiB < {EMERGENCY} -- stopping the arm on the copy, deleting it, arms read D:")
            STATE["cache"] = TC
            subprocess.run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process | Where-Object { "
                            "$_.CommandLine -match 'proxy_train.py' -and $_.CommandLine -match 'refe_proxy_fast' } | "
                            "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"],
                           capture_output=True)
            time.sleep(5)
            delete_copy(f"EMERGENCY: C: {f:.2f} GiB < {EMERGENCY}")
            return
        time.sleep(15)


# ------------------------------------------------------------------------------------------------ GPU stages
def arm_cmd(name, extra):
    c = STATE["cache"]
    return [PY, "refe/proxy_train.py", "--cache", str(c), "--sets", str(D / "sets"), "--snapshot", SNAP, "--out",
            str(ARMS / name), "--sets-work", str((TCF.parent if c == TCF else D) / "work_m6b" / name),
            "--device", "cuda", "--amp", "bf16", "--batch", "64", "--accum", "4", *extra]


def complete(name):
    return (ARMS / name / "final.pt").exists() and (ARMS / name / "summary.json").exists()


def arm_cache(name):
    try:
        ar = json.load(open(ARMS / name / "config.json"))["argv"]
        return Path(ar[ar.index("--cache") + 1])
    except Exception:
        return None


def retire(name, why):
    """move a finished arm AND its decode aside (G5 needs all six arms on ONE cache path)"""
    while procs("proxy_eval.py run") != 0:
        time.sleep(10)
    tag = f"_retired_{time.strftime('%H%M%S')}"
    moved = []
    for p in [ARMS / name, ROOT / "dumps" / f"{name}.json", ROOT / "dumps" / f"{name}.npz",
              *(ROOT / "seams").glob(f"refe_m6_{name}_*.npz")]:
        if p.exists():
            p.rename(p.with_name(p.stem + tag + p.suffix) if p.is_file() else p.with_name(p.name + tag))
            moved.append(p.name)
    log(f"{name}: RETIRED ({why}): {moved}")


def decodes(res):
    """GPU decodes beside the training: base first, then every finished arm, in order, until all seven exist"""
    while True:
        todo = [n for n in DECODE_ORDER if not (ROOT / "dumps" / f"{n}.json").exists()]
        if not todo:
            res["decodes"] = True
            return
        n = todo[0]
        if n != "base" and not complete(n):
            if STATE["arms_done"]:
                res["decodes"] = False
                return
            time.sleep(15)
            continue
        while procs("proxy_eval.py run") != 0:          # a decode left by v1 finishes first -- never two on one name
            time.sleep(15)
        if (ROOT / "dumps" / f"{n}.json").exists():
            log(f"decode_{n}: completed by the replaced chain -- kept")
            continue
        assert_code(f"before the decode of {n}")
        if not gpu_gate(f"decode_{n}", STATE["est_decode"], 800):
            res["decodes"] = False
            return
        STATE["gpu_started"] = True
        rc, mins = run(f"decode_{n}", [PY, "eval/proxy_eval.py", "run", "--eval-caches", str(ECD), "--arm",
                                       "none" if n == "base" else str(ARMS / n), "--name", n, "--out", str(ROOT)])
        assert_code(f"after the decode of {n}")
        if rc != 0:
            log(f"decode {n} FAILED rc {rc}")
            res["decodes"] = False
            return
        STATE["est_decode"] = max(mins * 1.2, 2.0)


def train_one(name, extra) -> bool:
    for attempt in range(2):
        while procs("proxy_train.py") != 0:
            time.sleep(15)
        if complete(name):                               # finished by the replaced chain while we waited: KEEP it
            rec = json.load(open(ARMS / name / "config.json"))["code_sha256"]
            if any(rec[k] != BASE[f"refe/{k}"] for k in ("proxy_train.py", "measures.py", "model.py", "train.py",
                                                          "proxy_cache.py")):
                raise CodeChanged(f"{name} (adopted) recorded code that is not the chain's")
            log(f"{name}: completed by the replaced chain -- kept")
            return True
        assert_code(f"before arm {name}")
        est = STATE["est_arm"] + (STATE["est_decode"] if name != G3[0] else 0.0)
        if not gpu_gate(name, est, 2500):
            return False
        STATE["gpu_started"] = True
        if (ARMS / name).exists():
            stale = ARMS / f"{name}_stopped_{time.strftime('%H%M%S')}"
            (ARMS / name).rename(stale)
            log(f"{name}: an incomplete run moved aside to {stale.name}")
        rc, mins = run(name, arm_cmd(name, extra))
        assert_code(f"after arm {name}")
        if complete(name):
            rec = json.load(open(ARMS / name / "config.json"))["code_sha256"]
            for k in ("proxy_train.py", "measures.py", "model.py", "train.py", "proxy_cache.py"):
                if rec[k] != BASE[f"refe/{k}"]:
                    raise CodeChanged(f"{name} recorded {k} {rec[k][:12]} != the chain's {BASE['refe/' + k][:12]}")
            if name != G3[0]:
                STATE["est_arm"] = max(mins * 1.15, 3.0)
            return True
        log(f"{name}: no final.pt (rc {rc}, attempt {attempt + 1})")
        if STATE["stopped_by_clock"]:
            return False
    return False


def gpu_stages():
    (ROOT / "dumps").mkdir(parents=True, exist_ok=True)
    res: dict = {}
    td = threading.Thread(target=decodes, args=(res,))
    td.start()
    ok = True
    try:
        if not complete(G3[0]) and not train_one(*G3):
            ok = False
        while ok:
            cur = STATE["cache"]
            for n, _x in ARMS6:                          # ONE cache path for all six (G5)
                if complete(n) and arm_cache(n) != cur:
                    retire(n, f"trained on {arm_cache(n)}, the chain now reads {cur}")
            todo = [(n, x) for n, x in ARMS6 if not complete(n)]
            if not todo:
                break
            n, x = todo[0]
            if not train_one(n, x):
                ok = False
    finally:
        STATE["arms_done"] = True
        td.join()
    done = [n for n, _x in ARMS6 if complete(n)]
    dec = [n for n in DECODE_ORDER if (ROOT / "dumps" / f"{n}.json").exists()]
    log(f"GPU stages: arms complete {done}; decodes {dec}; G3 {'complete' if complete(G3[0]) else 'MISSING'}")
    return ok and len(done) == 6 and len(dec) == 7 and complete(G3[0])


def cpu_stages():
    out = str(ROOT)
    labels = ",".join([f"refe_m6_Wt{s}_on" for s in "012"] + [f"refe_m6_T{s}_off" for s in "012"])
    for k in range(3):
        rc, _m = run(f"score_gating_{k + 1}", [PY, "eval/proxy_eval.py", "score", "--out", out, "--only", labels,
                                               "--workers", "1", "--min-free-ram-gb", "5.0"])
        if rc == 0:
            break
    run("families", [PY, "eval/proxy_eval.py", "families", "--out", out, "--only", labels])
    run("analyze_m6b", [PY, "eval/proxy_eval.py", "analyze-m6b", "--out", out,
                        "--train-cache", str(TCF if (TCF / "manifest.json").exists() else TC),
                        "--eval-caches", str(ECD), "--arms-root", str(ARMS)])
    rep = PKG / "eval" / "m6b_report.py"               # the full four-family report, a module NO arm or decode imports
    if rep.exists():
        run("m6b_report", [PY, "eval/m6b_report.py", "--out", out, "--arms-root", str(ARMS)])
    mark("M6B_RESULT_WRITTEN")
    log("M6B RESULT WRITTEN")
    STATE["guard_stop"] = True
    if TCF.parent.exists():
        delete_copy("after its last reader (analyze-m6b's G1 verify)")
    run("score_rest", [PY, "eval/proxy_eval.py", "score", "--out", out, "--workers", "1", "--min-free-ram-gb", "5.0"])


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    ARMS.mkdir(parents=True, exist_ok=True)
    log(f"M6b chain v2 start (pid {os.getpid()}; replaces v1 pid 41256, see the docstring)")
    ok = False
    try:
        if PREREG_BLOB not in HASH_REGISTER.read_text(encoding="utf-8", errors="ignore"):
            log("REFUSED: the M6b prereg blob is not in the register")
            return 2
        for m in MODULES:
            BASE[m] = sha(PKG / m)
        spt = json.load(open(ROOT / "selftest_proxy_train.json"))
        spe = json.load(open(ROOT / "selftest_proxy_eval.json"))
        stm = measures_selftest()
        pre = {"selftest_measures covers measures.py (or is RUNNING on these bytes; the analysis waits for it)":
                   stm is not False,
               "selftest_proxy_train covers proxy_train.py":
                   spt["tested_sha256"]["proxy_train.py"] == BASE["refe/proxy_train.py"] and not spt["failed"],
               "selftest_proxy_eval covers proxy_eval.py":
                   spe["tested_sha256"]["proxy_eval.py"] == BASE["eval/proxy_eval.py"] and not spe["failed"]}
        log("code baseline: " + ", ".join(f"{m.split('/')[-1]} {BASE[m][:12]}" for m in MODULES))
        log(f"selftests on the FINAL bytes: {pre}")
        if not all(pre.values()):
            log("REFUSED: the selftests do not cover the package's bytes")
            return 2
        json.dump({"base_sha256": BASE, "at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  open(ROOT / "code_baseline.json", "w"), indent=1)
        STATE["cache"] = TCF if use_copy() else TC
        log(f"ALL arms read {STATE['cache']}")
        ok = gpu_stages()
        log(f"GPU stages {'COMPLETE' if ok else 'INCOMPLETE'}")
    except CodeChanged as e:
        log(f"ABORT: {e}")
        ok = False
    finally:
        write_token("M6b's last GPU stage exited" if STATE["gpu_started"] else "no M6b GPU stage ran")
    if ok:
        while measures_selftest() is None:                    # the full suite (~25 min) may still be running
            time.sleep(30)
        if measures_selftest() is True:
            log("selftest_measures: PASS on the chain's measures.py bytes -- G4's record is in place")
            cpu_stages()
        else:
            log("REFUSED the analysis: selftest_measures did NOT pass on the chain's bytes (G4 cannot pass); the arms "
                "and decodes stay banked")
    else:
        left = [n for n, _x in ARMS6 if not complete(n)]
        dleft = [n for n in DECODE_ORDER if not (ROOT / "dumps" / f"{n}.json").exists()]
        log(f"STOPPED{' by the 03:00 limit' if STATE['stopped_by_clock'] else ''}: arms left {left}; decodes left "
            f"{dleft} -- the rest runs later on the same bytes; the NVMe copy is kept only while C: >= 30 GiB")
        STATE["guard_stop"] = True
        if TCF.parent.exists() and c_free() < FLOOR_AFTER:
            delete_copy("the chain stopped and C: is below the floor")
    log("M6b chain done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
