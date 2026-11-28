#!/usr/bin/env python3
"""SPEC_NAVTEST Amendment 8 (registered 2026-09-28 06:32, SPEC blob 8ace80cc): the confirmation run, as registered.

Coordinator 2026-09-28 06:35-06:50 (times from `date`):
  * land refe/planner.py + eval/refe_navtest_seam.py only AFTER snapshot 018's eval pipeline has fully ended
    (`ZZSNAP_DONE 018` in its log), and only the bytes the CPU checks tested (sha256 asserted)
  * GPU window: from that marker until 12:30 -- a HARD stop; the shared dev-box lock
    C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock taken EXCLUSIVELY across the GPU stages (create-if-absent, else
    wait), released in a finally only if it still holds our content; VRAM gate (refav1 runs): used + need <= 6,960 MiB
  * the GPU-done token (a FRESH one, _2_) written ONCE, as its own line of a8_chain.log, the moment the A8 GPU work
    exits, on every path, in a finally; never printed anywhere else
Stages:
  0  preconditions: the registered SPEC blob; the CPU checks (parity + gate (a) + frames; gate (b) with its mutation
     arms; the end-to-end seam smoke) all PASSED on exactly the bytes to land
  1  wait for 018's end, and for no process of its pipeline; the package's planner.py / seam still the BASE bytes
  2  land (tmp + os.replace; sha256 re-read == tested)
  3  GPU, under the lock, in priority order: OFF (the gating control, + the proposal dump for gate (c)), ON (gating),
     then the two reported-not-gating variants (clamp150, straight) -- each only if its MEASURED estimate ends by 12:30;
     the modules every stage imports are asserted unchanged before and after each stage
  4  token (finally); CPU: the unchanged harness on every seam, families6, a8_analyze.py -> RESULT_A8
Log: raw/2026-09-28-goal-clamp/a8_chain.log (the token appears there exactly once).

SCHEDULE CHANGE (coordinator ~06:58): 018's GPU stage ended 06:46, so the FORWARDS run NOW, from the TESTED SIBLING COPY
(scratchpad pkg_a8: planner.py 522a87ae / refe_navtest_seam.py 69b7dd2a, sha asserted), under the shared lock, released
as soon as the forwards are done (before any CPU scoring). The live eval/ tree stays untouched until `ZZSNAP_DONE 018`;
the landing is the chain's LAST step.
v2 (coordinator ~07:05-07:10): the lock is never acquired while refcv7/devbox_gpu.chain_active exists; the landing also waits until no REFe snapshot seam can be running or about to start -- it lands
BEFORE 12:50 or only AFTER 019's seam report exists -- and logs the live files' git blob hashes after the swap. The
12:30 stop is checked BETWEEN arms only: a running arm is never killed; OFF and ON come first, so a late stop can only
cut the reported arms (straight, then clamp150). Harness scoring: the two arms in parallel (W=2), each started only at >= 6.0 GB
free RAM, retried later on a RAM abort (never competing). Gate (b) reads the bytes that RAN (the sibling planner).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
sys.path.insert(0, str(PKG / "eval"))
import eval_checkpoint as EC  # noqa: E402

LOG = HERE / "a8_chain.log"
TIMES = HERE / "a8_chain_times.json"
TOKEN = "ZZ_A8_" + "GPU_DONE_2_ZZ"                         # assembled: the literal sits in no source line
SIB = Path("C:/Users/Admin/AppData/Local/Temp/claude/D--Projects-TanitAD/91effc67-8c1e-4b66-9a63-341a3109dfc1/"
           "scratchpad/pkg_a8")
FILES = {"refe/planner.py": ("8f15c9f62a91642568f7667f20dc32e0f0eca6d884369b473acbd5dcf64f7a9a",     # BASE
                             "522a87ae2d57acc04442effec131e5085dc8d3595e5683c9b68554af2663a94f"),    # TESTED
         "eval/refe_navtest_seam.py": ("5bf2eb7840ded30eeb17588ce5407bf7f0ac2776b9840a610e0c0b9f4d265827",
                                       "69b7dd2ae1a55c251b8e41a2128c33c431637066fe060105579d4168d21a331c")}
MODULES = ("refe/planner.py", "eval/refe_navtest_seam.py", "refe/model.py", "refe/ckpt_io.py",
           "refe/navtrain_scenarios.py", "code/augment_routes.py", "refe/load_dinov3.py")
SPEC_BLOB = "8ace80ccb7166cd7b9fc13ad4fde4272b1945320"
SNAP = "D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch015.pt"
SNAP_MD5 = "d7c59f4f2fbcbde3e2dec8f67d63a7e7"
EVAL018 = Path("C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/"
               "bd7d00af-b98e-42f1-a53c-cb4113059b0f/scratchpad/eval_snap_018.log")
MARK018 = "ZZSNAP_DONE 018"
# the scripts that IMPORT refe/planner.py (grep, 2026-09-28): no landing while one of them runs, from any session
PIPE018 = ("refe_navtest_seam.py", "eval_checkpoint.py", "proposal_table.py")
STOP_AT = 12 * 60 + 30
LAND_BEFORE = 12 * 60 + 50                                  # ... or after 019's seam report exists
REPORT019 = Path("D:/Projects/TanitAD/data/refe_navtest/seams/refe_sub200_ep019.report.json")
LOCK = Path("C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock")
CHAIN_ACTIVE = Path("C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active")
VRAM_CAP_MIB, NEED_MIB, RAM_MIN_GB = 6960, 3300, 3.0
WORK = Path("D:/Projects/TanitAD/data/refe_proxy/goal_clamp/a8")             # no spaces (the harness's Hydra grammar)
ARMS = (("off", ["--dump-proposals", "{work}/refe_a8_off_props.npz"]), ("on", ["--sanitize-goal"]),
        ("clamp150", ["--goal-variant", "clamp150"]), ("straight", ["--goal-variant", "straight"]))
GATING = ("off", "on")
_lock = threading.Lock()
ST = {"token_written": False, "gpu_started": False, "lock_content": None, "est_min": 25.0}
BASE: dict = {}


def log(msg):
    assert TOKEN not in msg
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with _lock:
        with open(LOG, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
        print(line, flush=True)


def mark(k):
    with _lock:
        t = json.load(open(TIMES)) if TIMES.exists() else {}
        t[k] = time.strftime("%Y-%m-%dT%H:%M:%S")
        json.dump(t, open(TIMES, "w"), indent=1)


def write_token(why):
    with _lock:
        if ST["token_written"]:
            return
        txt = LOG.read_text(encoding="utf-8") if LOG.exists() else ""
        if any(l.endswith(" " + TOKEN) for l in txt.splitlines()):
            ST["token_written"] = True
            return
        with open(LOG, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {TOKEN}\n")
        ST["token_written"] = True
    mark("gpu_done_token")
    log(f"A8 GPU-done token (_2_) written, once ({why})")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def minutes_now():
    lt = time.localtime()
    return lt.tm_hour * 60 + lt.tm_min + lt.tm_sec / 60.0


def procs(pattern):
    me = f"{os.getpid()},{os.getppid()}"
    o = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Process | Where-Object { "
                        f"$_.CommandLine -match '{pattern}' -and $_.Name -match 'python' -and "
                        f"@({me}) -notcontains $_.ProcessId }} | Measure-Object).Count"],
                       capture_output=True, text=True).stdout.strip()
    return int(o) if o.isdigit() else -1


def vram():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    u, t = (int(x) for x in o[0].split(","))
    return u, t


def free_ram_gb():
    o = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                       capture_output=True, text=True).stdout.strip()
    return int(o) / 1024 / 1024 if o.isdigit() else -1.0


class CodeChanged(Exception):
    pass


def assert_code(where):
    bad = [m for m in MODULES if sha(SIB / m) != BASE[m]]
    if bad:
        raise CodeChanged(f"{where}: module(s) changed under the chain: {bad}")


# ------------------------------------------------------------------------------------------------ the shared lock
def lock_acquire() -> bool:
    said = False
    while True:
        if minutes_now() + ST["est_min"] > STOP_AT:
            log(f"GPU LOCK: not acquired -- a stage of ~{ST['est_min']:.0f} min could no longer end by 12:30")
            return False
        if CHAIN_ACTIVE.exists():                              # coordinator ~07:10: never take a refcv7 chain gap
            if not said:
                log(f"GPU LOCK: {CHAIN_ACTIVE.name} exists -- the refcv7 chain owns the GPU; not acquiring")
                said = True
            time.sleep(20)
            continue
        content = json.dumps({"job": "refe-a8-confirm", "pid": os.getpid(), "win_pid_of_locker": os.getpid(),
                              "host": socket.gethostname(),
                              "acquired": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
        try:
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            ST["lock_content"] = content
            log(f"GPU LOCK acquired: {content}")
            return True
        except FileExistsError:
            if not said:
                try:
                    holder = LOCK.read_text(encoding="utf-8")[:300]
                except OSError:
                    holder = "(unreadable)"
                log(f"GPU LOCK held by another job -- waiting: {holder}")
                said = True
            time.sleep(20)


def lock_release():
    c = ST.get("lock_content")
    if not c:
        return
    try:
        if LOCK.read_text(encoding="utf-8") == c:
            LOCK.unlink()
            log("GPU LOCK released")
        else:
            log("GPU LOCK: the file no longer holds our content -- NOT removed")
    except FileNotFoundError:
        log("GPU LOCK: already gone")
    ST["lock_content"] = None


# ------------------------------------------------------------------------------------------------ stages
def preconditions() -> bool:
    ok = SPEC_BLOB in (PKG / "eval" / "raw" / "SPEC_PREREG_HASH.txt").read_text(encoding="utf-8", errors="ignore")
    log(f"SPEC blob {SPEC_BLOB[:8]} registered: {ok}")
    for f, (_b, tested) in FILES.items():
        s = sha(SIB / f)
        log(f"sibling {f}: {s[:12]} (tested {tested[:12]}) {'OK' if s == tested else 'MISMATCH'}")
        ok &= s == tested
    for name, key in (("a8_goal_checks_sibling.json", "sha256_planner"), ("a8_selftest_sibling.json", "planner_sha256"),
                      ("a8_seam_smoke.json", None)):
        p = HERE / name
        r = json.load(open(p, encoding="utf-8")) if p.exists() else {"failed": ["MISSING"]}
        good = not r.get("failed")
        if key == "sha256_planner":
            good &= r.get(key, {}).get("new") == FILES["refe/planner.py"][1]
        elif key:
            good &= r.get(key) == FILES["refe/planner.py"][1]
        log(f"CPU check {name}: {'PASS' if good else 'FAIL'} (failed: {r.get('failed')})")
        ok &= good
    m = md5(SNAP)
    log(f"snapshot 015 md5 {m} (registered {SNAP_MD5}) {'OK' if m == SNAP_MD5 else 'MISMATCH'}")
    return ok and m == SNAP_MD5


def wait_018() -> bool:
    said = False
    while True:
        try:
            done = MARK018 in EVAL018.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            done = False
        busy = {p: procs(p) for p in PIPE018}
        window = minutes_now() < LAND_BEFORE or REPORT019.exists()
        if done and not any(v != 0 for v in busy.values()) and window:
            log(f"snapshot 018's eval pipeline has ended ('{MARK018}' present; none of {list(PIPE018)} running); "
                f"landing window open ({'before 12:50' if minutes_now() < LAND_BEFORE else '019 seam report exists'})")
            return True
        if minutes_now() > 23 * 60:
            log(f"018's eval not ended by 23:00 (marker {done}; running {busy}) -- no landing")
            return False
        if not said:
            log(f"waiting to land (018 marker present: {done}; running: {  {k: v for k, v in busy.items() if v} }; "
                f"landing window open: {window})")
            said = True
        time.sleep(30)


def land() -> bool:
    for f, (base, tested) in FILES.items():
        cur = sha(PKG / f)
        if cur == tested:
            log(f"LAND {f}: already the tested bytes {tested[:12]}")
            continue
        if cur != base:
            log(f"LAND REFUSED: {f} is {cur[:12]}, neither the base {base[:12]} nor the tested {tested[:12]} -- "
                f"someone changed it; not overwriting")
            return False
        tmp = PKG / (f + ".a8tmp")
        shutil.copy2(SIB / f, tmp)
        if sha(tmp) != tested:
            tmp.unlink()
            log(f"LAND REFUSED: the copy of {f} does not hash to the tested bytes")
            return False
        os.replace(tmp, PKG / f)
        got = sha(PKG / f)
        log(f"LANDED {f}: {base[:12]} -> {got[:12]} ({'== tested' if got == tested else 'MISMATCH'})")
        if got != tested:
            return False
    for f in FILES:
        b = subprocess.run(["git", "hash-object", str(PKG / f)], cwd=str(PKG), capture_output=True,
                           text=True).stdout.strip()
        log(f"LIVE git blob {f}: {b if len(b) == 40 else 'INCONCLUSIVE (' + b + ')'}")
    return True


def run_arm(arm, extra) -> bool:
    out = WORK / f"refe_a8_{arm}.npz"
    if out.exists() and (WORK / f"refe_a8_{arm}.report.json").exists():
        log(f"{arm}: already written -- kept")
        return True
    if minutes_now() + ST["est_min"] > STOP_AT:
        log(f"{arm}: NOT started -- est {ST['est_min']:.0f} min would end after 12:30")
        return False
    said = False
    while True:
        u, _t = vram()
        ram = free_ram_gb()
        if u + NEED_MIB <= VRAM_CAP_MIB and ram >= RAM_MIN_GB:
            break
        if minutes_now() + ST["est_min"] > STOP_AT:
            log(f"{arm}: NOT started -- the VRAM/RAM gate held it until too late")
            return False
        if not said:
            log(f"{arm}: waiting (VRAM used {u} + need {NEED_MIB} vs cap {VRAM_CAP_MIB} MiB; free RAM {ram:.1f} GB)")
            said = True
        time.sleep(20)
    assert_code(f"before arm {arm}")
    cmd = [EC.DRIVERL_PY, "refe_navtest_seam.py", "--ckpt", SNAP, "--tokens", str(WORK / "tokens_a8_fresh.json"),
           "--out", str(out), "--record-inputs", str(WORK / f"refe_a8_{arm}.inputs.json"), "--arm", f"refe_a8_{arm}",
           *[x.format(work=WORK) for x in extra]]
    env = EC.env_driverl()
    env.update(OMP_NUM_THREADS="6", PYTHONIOENCODING="utf-8")
    log(f"{arm}: START from the sibling {SIB} (VRAM used {vram()[0]} MiB; est {ST['est_min']:.0f} min)")
    ST["gpu_started"] = True
    mark(f"{arm}_start")
    t0 = time.time()
    with open(HERE / "logs" / f"a8_{arm}.log", "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, cwd=str(SIB / "eval"), env=env, stdout=fh, stderr=subprocess.STDOUT)
    mins = (time.time() - t0) / 60
    mark(f"{arm}_end")
    assert_code(f"after arm {arm}")
    ok_tok = "ZZSEAM_OK" in (HERE / "logs" / f"a8_{arm}.log").read_text(encoding="utf-8", errors="ignore")
    log(f"{arm}: EXIT {rc} after {mins:.1f} min; ZZSEAM_OK {ok_tok}")
    ST["est_min"] = max(mins * 1.15, 5.0)
    return rc == 0 and out.exists()


def cpu_tail(arms_done):
    fresh = json.load(open(HERE / "amendment8_fresh_set.json", encoding="utf-8"))
    tokfile = WORK / "tokens_a8_fresh.json"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    (HERE / "a8" / "score").mkdir(parents=True, exist_ok=True)
    (HERE / "a8" / "families").mkdir(parents=True, exist_ok=True)
    def score_one(arm):
        lb = f"refe_a8_{arm}"
        seam = WORK / f"{lb}.npz"
        st = None
        for attempt in range(6):
            while free_ram_gb() < 6.0:                     # coordinator: start only at >= 6.0 GB free; never compete
                time.sleep(30)
            lf = HERE / "a8" / "score" / f"{lb}.log"
            EC.run([EC.DRIVERL_PY, "score_navtest_refe.py", "--label", lb, "--seam", str(seam), "--tokens",
                    str(tokfile), "--out", str(WORK / "score")], str(PKG / "eval"), env, str(lf))
            txt = lf.read_text(encoding="utf-8", errors="replace") if lf.exists() else ""
            st = next((json.loads(l) for l in reversed(txt.splitlines()) if l.startswith("{") and '"status"' in l), None)
            if st and st.get("status") == "PASS":
                break
            log(f"score {lb}: attempt {attempt + 1} not PASS ({(st or {}).get('failures')}) -- retrying later")
            time.sleep(120)
        csv = WORK / "score" / lb / f"{lb}.csv"
        if csv.exists():
            shutil.copy2(csv, HERE / "a8" / "score" / f"{lb}.csv")
        log(f"score {lb}: {(st or {}).get('status')} rows {(st or {}).get('csv_valid_rows')} "
            f"PDMS {((st or {}).get('summary_x100_4dp') or {}).get('PDMS')}")
    for pair in (arms_done[:2], arms_done[2:]):          # W = 2: the two gating arms together, then the two variants
        th = [threading.Thread(target=score_one, args=(arm,)) for arm in pair]
        for t_ in th:
            t_.start()
            time.sleep(60)                                 # the second starts after the first has taken its memory
        for t_ in th:
            t_.join()
    for arm in arms_done:
        lb = f"refe_a8_{arm}"
        seam = WORK / f"{lb}.npz"
        fj = HERE / "a8" / "families" / f"{lb}.json"
        EC.run([EC.TANITAD_PY, "families6.py", "--seam", str(seam), "--inputs", EC.EXPORT, "--stage", "1", "--label",
                lb, "--out", str(fj), "--n-boot", "2000"], EC.EV6, env, str(HERE / "a8" / "families" / f"{lb}.log"))
        log(f"families {lb}: {'ok' if fj.exists() else 'MISSING'}")
    for arm in arms_done:                                           # bank the seams' small artifacts beside the RESULT
        for suf in (".report.json", ".inputs.json"):
            p = WORK / f"refe_a8_{arm}{suf}"
            if p.exists():
                shutil.copy2(p, HERE / "a8" / p.name)
    rc = subprocess.call([EC.DRIVERL_PY, str(HERE / "a8_analyze.py"), "--landed-planner",
                          str(SIB / "refe" / "planner.py")], cwd=str(PKG / "eval"),
                         env=dict(EC.env_driverl(), PYTHONIOENCODING="utf-8"),
                         stdout=open(HERE / "logs" / "a8_analyze.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT)
    log(f"a8_analyze: EXIT {rc}")
    log("A8 RESULT WRITTEN" if (HERE / "RESULT_A8_GOAL_SANITISATION.md").exists() else "A8 RESULT MISSING")


def main() -> int:
    (HERE / "logs").mkdir(exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    log(f"A8 chain start (pid {os.getpid()}); GPU window: from '{MARK018}' to 12:30")
    fresh = json.load(open(HERE / "amendment8_fresh_set.json", encoding="utf-8"))
    json.dump({"tokens": fresh["fresh_tokens"], "token_log": fresh["fresh_token_log"]},
              open(WORK / "tokens_a8_fresh.json", "w", encoding="utf-8"))
    done = []
    try:
        if not preconditions():
            log("REFUSED: a precondition failed -- no GPU stage, no landing")
            return 2
        for m in MODULES:
            BASE[m] = sha(SIB / m)
        log("code baseline (the SIBLING the forwards run from): " +
            ", ".join(f"{m.split('/')[-1]} {BASE[m][:12]}" for m in MODULES))
        if not lock_acquire():
            return 5
        try:
            for arm, extra in ARMS:
                if run_arm(arm, extra):
                    done.append(arm)
                elif arm in GATING:
                    log(f"{arm} (GATING) did not complete -- the reported-not-gating arms are skipped")
                    break
        finally:
            lock_release()
    except CodeChanged as e:
        log(f"ABORT: {e}")
    finally:
        write_token("the A8 GPU work exited" if ST["gpu_started"] else "no A8 GPU stage ran")
    log(f"GPU arms complete: {done}")
    if all(a in done for a in GATING):
        cpu_tail(done)
    else:
        log("the gating arms are not both complete -- no statistic is read; what remains is reported")
    if wait_018() and land():                              # the live tree only after 018's pipeline has ended
        log("LANDING complete: the package's planner.py / refe_navtest_seam.py are the tested bytes (default OFF)")
    else:
        log("LANDING not done -- the tested bytes stay in the sibling; the coordinator decides")
    log("A8 chain done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
