#!/usr/bin/env python3
"""P2 FULL-NAVTEST 64-PROPOSAL TABLE -- the C3 pipeline (c3_run.py -> c3_run_p2.py) made CHUNKED, RESUMABLE and PIPELINED for 12,146 tokens.

This file is `../2026-10-15-p2-zero-tail/p2_own_table/c3_run_p2.py` (landed 80775e9, NOT modified) copied and parametrised; the edit list is
`c3_run_p2full_vs_p2.diff`.  What is the same: the checkpoint (sha256-preflighted), RULE navsim_v1, the dump command (refe_navtest_seam.py
--dump-proposals), the W3 harness invoked UNCHANGED per proposal column (score_navtest_refe.py), the fidelity-gate arithmetic, read_csv, the table layout.
What changed, and why:

  * CHUNKS.   12,146 tokens are cut into contiguous groups of whole navtest LOGS (<= 1,000 tokens; the map load is per log, so a log is never split).
              Each chunk is dumped in its own process (resume loses <= one chunk, ~25 min) and the GPU lock is RELEASED between chunks, so a neighbour
              that queues on the shared lock gets its turn.  W3's cmd_score takes the tokens on its command line (Windows limit 32,767 chars): a chunk
              of 1,000 tokens is ~20 KB; the whole 12,146 would not fit, which is the second reason for chunks.
  * PIPELINE. The CPU scoring of chunk c (64 harness columns x 5 workers) runs while the GPU dumps chunk c+1.  The score stage is the bottleneck, so the
              dump costs nothing extra in wall time.
  * GATE.     The fidelity gate (dump pick vs P2's banked seam, 1e-3 m, fingerprints equal) runs PER CHUNK, BEFORE that chunk's seams are written, so
              nothing is scored from a dump that does not reproduce P2.  The first failing chunk aborts the whole job (rc 2).
  * LOCK.     The brief's protocol: wait_idle.py, then an O_EXCL lock with an {pid, owner, t} body (aux_navtest_forward.acquire_gpu_lock's), memory.used
              <= 3500 MiB re-checked after taking it, the chain_active marker respected.  The wait is a poll, never a blocking subprocess.
  * IO.       Bulk dumps + the final table go to D: (BULK); per-column seams and harness outputs go to C: (WORK, NTFS: D:/E: is exFAT with 1 MiB
              clusters, which would turn 832 small seam files into ~0.8 GB).  The harness's own run dirs are on D:/Archive (hard-coded in W3's run_v1.py EXP).
  * GUARDS.   A D: free-space guard (pause, never write into a full disk); a RAM guard before each harness start (landed); a circuit breaker on repeated
              harness failures.
Nothing here fits anything on the navtest tokens.  `--tag smoke` runs the whole thing on 8 tokens in 2 chunks on the CPU (no GPU lock).
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))            # .../2026-09-20-refe-plan  (this dir is raw/<name>/)
EVAL = os.path.join(PKG, "eval")
sys.path.insert(0, EVAL)
import eval_checkpoint as EC  # noqa: E402  (one source for the venv + env, as proposal_table.py)

CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt"
CKPT_SHA256 = "ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51"   # RESULT_P2.md header; verified by preflight()
RULE = "navsim_v1"             # stage_2 p2_full seam report: rule navsim_v1, goal_fix pdm_route, repair_last_heading true
FRAMES = f"{EC.DATA}/frames"
DB_DIR = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
BANKED_SEAM = f"{EC.DATA}/seams/stage2_p2_full.npz"                      # md5 49a1f6ec... (RESULT_P2.md section 6)
BANKED_CSV = f"{EC.DATA}/score/refe_stage2_p2_full/refe_stage2_p2_full.csv"       # sha256 61073a61...
BANKED_CSV_SHA256 = "61073a61e44a3dcf41fad6dc8c5b261ce97eb8f8d8b49c1adc077dd7c04b61e6"
TOKENS_FULL = f"{EC.DATA}/navtest_full_tokens.json"
TOKENS_FULL_SHA256 = "927bbe3838dd70af5fa27eadceb09f55acc9c1cfd04cf0a6b520f1519175a955"      # == raw/p2_r2_instrument/baseline/navtest_full_tokens.json
N_FULL, N_LOGS_FULL = 12146, 136
FRAME_CONTROL_CENSUS = os.path.join(PKG, "raw", "2026-10-04-navtest-final", "frame_control_navtest_full.json")   # the 36 known over-bar tokens in 4 logs
FRAME_BAR_M = 1e-3
FRAME_KNOWN_MAX_M = 6.19         # census max 6.1806; RESULT_NAVTEST_FINAL.md section on the seam's own control

BULK = "D:/Projects/TanitAD-artifacts/p2_full_table"                     # bulk, durable-ish (external exFAT drive)
WORK = "C:/Users/Admin/p2full_work"                                      # NTFS: seams, harness outputs (must share a drive with PKG, see W3's relpath)
WAIT_IDLE = "D:/Projects/TanitAD-artifacts/_rlfan_work/wait_idle.py"
LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
MARK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"
OWNER = "p2full_table"
# Processes that must have EXITED before this job takes the lock, beyond what wait_idle.py checks (its PAT list does not name the scene-tap probe, which holds the
# lock in blocks and stays CUDA-resident in between).  "Never preempt the probe" (the Master Mind's brief): a gap in its lock holds is not a free GPU.
BLOCKER_PATTERNS = ("scene_tap_probe",)
WAITER_PATTERNS = ("gpu_lock.py acquire",)
FIDELITY_TOL_M = 1e-3
GPU_MAX_USED_MIB = 3500          # = aux_navtest_forward.GPU_MAX_USED_MIB (8188 MiB card; the desktop holds ~2.5 GB with no job)
MIN_FREE_RAM_MB = 5000           # start a harness/dump only with this much AVAILABLE RAM: W3's wrapper aborts a run (exit 3) once available RAM stays < 3,000 MB, and that
                                 # abort hits EVERY concurrent harness; one column peaks at ~0.75 GB (MEASURED 724.6 MB on a 990-token chunk), so >= 5,000 keeps >= 4,000 after it starts
STAGGER_S = 15                   # harness starts are >= 15 s apart, so the previous start's RSS has grown before the next RAM check
MIN_FREE_BULK_GB = 3.0           # pause (never write) below this on the harness's drive; the job needs ~5.3 GiB there (PLAN.md)
MAX_CHUNK_TOKENS = 1000
SUB = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
       "time_to_collision_within_bound", "comfort", "driving_direction_compliance")
M = 64            # proposals per token (the dump asserts it); --m < 64 is for the plumbing smoke only


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atomic_json(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------- chunks
def build_chunks(doc, max_tokens=MAX_CHUNK_TOKENS):
    """Contiguous groups of whole logs (sorted by log name), each <= max_tokens, tokens sorted ascending within a chunk.  Deterministic.
    A log larger than max_tokens gets a chunk of its own (never split: the map load is per log)."""
    by_log = defaultdict(list)
    for t in doc["tokens"]:
        by_log[doc["token_log"][t]].append(t)
    chunks, cur, n = [], [], 0
    for lg in sorted(by_log):
        k = len(by_log[lg])
        if cur and n + k > max_tokens:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(lg)
        n += k
    if cur:
        chunks.append(cur)
    out = []
    for i, logs in enumerate(chunks):
        toks = sorted(t for lg in logs for t in by_log[lg])
        out.append({"id": i, "logs": logs, "tokens": toks})
    return out


def write_chunks(doc, outdir, max_tokens=MAX_CHUNK_TOKENS, provenance=None):
    """Writes outdir/c{ii}_tokens.json (W3 format) and outdir/chunks.json.  Refuses to overwrite a plan that differs."""
    os.makedirs(outdir, exist_ok=True)
    ch = build_chunks(doc, max_tokens)
    plan = {"max_chunk_tokens": max_tokens, "n_tokens": len(doc["tokens"]), "n_chunks": len(ch), "provenance": provenance or {}, "chunks": []}
    for c in ch:
        p = os.path.join(outdir, f"c{c['id']:02d}_tokens.json")
        cdoc = {"rule": f"chunk {c['id']} of {len(ch)}: whole navtest logs {c['logs'][0]} .. {c['logs'][-1]}", "tokens": c["tokens"],
                "token_log": {t: doc["token_log"][t] for t in c["tokens"]}}
        new = json.dumps(cdoc, indent=0)
        if os.path.exists(p):
            assert _read_text(p) == new, f"{p} exists and differs: the chunk plan is written ONCE"
        else:
            with open(p, "w", encoding="utf-8") as f:
                f.write(new)
        plan["chunks"].append({"id": c["id"], "n": len(c["tokens"]), "n_logs": len(c["logs"]), "tokens_file": os.path.basename(p),
                               "tokens_sha256": sha256_file(p), "first_log": c["logs"][0], "last_log": c["logs"][-1]})
    pp = os.path.join(outdir, "chunks.json")
    if os.path.exists(pp):
        old = jload(pp)
        assert [(c["id"], c["n"], c["tokens_sha256"]) for c in old["chunks"]] == [(c["id"], c["n"], c["tokens_sha256"]) for c in plan["chunks"]], \
            f"{pp} exists and differs"
    else:
        atomic_json(plan, pp)
    return plan


def cmdline_chars(chunk_tokens_doc):
    """Length of the Hydra override strings W3's split_overrides puts on the command line (the Windows limit is 32,767 for the WHOLE line)."""
    toks = chunk_tokens_doc["tokens"]
    logs = sorted({chunk_tokens_doc["token_log"][t] for t in toks})
    return (sum(len(x) + 3 for x in toks) + sum(len(x) + 3 for x in logs))


# --------------------------------------------------------------------------------------------- state
class Ctx:
    def __init__(self, a, runner=None):
        self.a = a
        self.tag = a.tag
        self.main = a.tag == "main"
        self.wd = HERE if self.main else os.path.join(HERE, f"_{a.tag}")
        self.work = WORK if self.main else os.path.join(WORK, f"_{a.tag}")
        self.bulk = BULK if self.main else os.path.join(self.wd, "bulk")
        self.score_root = f"{self.work}/score"
        self.seam_root = f"{self.work}/seams"
        for d in (os.path.join(self.wd, "logs"), os.path.join(self.wd, "state"), self.score_root, self.seam_root, os.path.join(self.bulk, "dumps")):
            os.makedirs(d, exist_ok=True)
        self.M = a.m
        self.pose_tol_m = FIDELITY_TOL_M if self.main else (a.pose_tol_m or FIDELITY_TOL_M)
        if self.main:
            assert self.pose_tol_m == FIDELITY_TOL_M
        self.status_p = os.path.join(self.wd, "STATUS.json")
        self.status = jload(self.status_p) if os.path.exists(self.status_p) else {}
        self._lk = threading.RLock()
        self.abort = threading.Event()
        self.abort_reason = None
        self.rc = 0
        self.counts = {"chunks_dumped": 0, "score_pass": 0, "score_fail": 0, "score_total": 0, "score_ram_guard_retries": 0}
        self.lock_body = None
        self.start_lk = threading.Lock()
        self.last_start = 0.0
        self.t_score0 = None
        self.runner = runner or RealRunner(self)
        self._banked = None
        self._known_fc_logs = None
        cdir = os.path.join(self.wd, "chunks")
        if self.main:
            self.plan = jload(os.path.join(cdir, "chunks.json"))      # written ONCE by make_chunks_p2full.py
        else:
            doc = jload(a.tokens)
            self.plan = write_chunks(doc, cdir, max_tokens=a.chunk_tokens)
        self.chunks = [dict(c, tokens_file=os.path.join(cdir, c["tokens_file"]).replace("\\", "/")) for c in self.plan["chunks"]]
        self.counts["score_total"] = len(self.chunks) * self.M

    # -- status
    def set(self, **kw):
        with self._lk:
            self.status.update(kw, updated=now())
            atomic_json(self.status, self.status_p)
        print(f"[{now()}] STATUS {json.dumps(kw, default=str)[:300]}", flush=True)

    def progress(self, **kw):
        with self._lk:
            self.counts.update(kw)
            c = self.counts
            rate = None
            if self.t_score0 and c["score_pass"] > 0:
                rate = round(c["score_pass"] / ((time.time() - self.t_score0) / 60.0), 3)
            self.status.update(progress=dict(c, score_cols_per_min=rate), updated=now())
            atomic_json(self.status, self.status_p)

    def tokens_of(self, ch):
        return jload(ch["tokens_file"])

    def label(self, cid, k):
        return f"refe_p2full_{self.tag}_c{cid:02d}_p{k:02d}"

    def seam_k(self, cid, k):
        return f"{self.seam_root}/{self.label(cid, k)}.npz"

    def csv_k(self, cid, k):
        return f"{self.score_root}/{self.label(cid, k)}/{self.label(cid, k)}.csv"

    def dump_path(self, cid):
        return f"{self.bulk}/dumps/dump_{self.tag}_c{cid:02d}.npz"

    def rerun_seam(self, cid):
        return f"{self.seam_root}/refe_p2full_{self.tag}_c{cid:02d}_dumprun.npz"

    def state_p(self, cid):
        return os.path.join(self.wd, "state", f"chunk_{cid:02d}.json")

    def banked(self):
        if self._banked is None:
            B = np.load(BANKED_SEAM)
            self._banked = ({str(t): i for i, t in enumerate(B["token"])}, B["poses"], B["fingerprint"])
        return self._banked

    def known_fc_logs(self):
        if self._known_fc_logs is None:
            try:
                d = jload(FRAME_CONTROL_CENSUS)
                self._known_fc_logs = {v["log"] for v in d["over_bar"].values()}
            except OSError:
                self._known_fc_logs = set()
        return self._known_fc_logs


# --------------------------------------------------------------------------------------- GPU lock
def pid_alive(pid):
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True)
    return f'"{pid}"' in r.stdout.decode("utf-8", "replace")


def gpu_used_mib():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True)
    try:
        return int(r.stdout.decode("utf-8", "replace").strip().splitlines()[0])
    except (ValueError, IndexError):
        return -1


def _read_text(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def _read_json_text(p):
    return json.loads(_read_text(p))


def lock_body(job):
    return json.dumps({"pid": os.getpid(), "owner": OWNER, "job": job, "host": os.environ.get("COMPUTERNAME", "?"), "t": time.time(), "acquired": now()})


def release(body):
    try:
        if _read_json_text(LOCK).get("pid") == json.loads(body)["pid"]:
            os.remove(LOCK)
            print(f"[{now()}] GPU lock released", flush=True)
    except (OSError, ValueError):
        pass


def _python_lines(patterns):
    """Command lines (with pid) of running python processes matching any pattern (never this process).  Unreadable list -> one marker line (callers fail closed)."""
    ps = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' } | ForEach-Object { $_.ProcessId.ToString() + ' ' + $_.CommandLine }"
    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], capture_output=True, timeout=120).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError):
        return ["(process list unreadable)"]
    return [ln[:200] for ln in out.splitlines() if any(p in ln for p in patterns) and not ln.startswith(f"{os.getpid()} ")]


def blockers():
    """Running processes that must have EXITED before the lock is taken (the scene-tap probe)."""
    return _python_lines(BLOCKER_PATTERNS)


def waiters():
    """Other jobs QUEUED on the shared lock (the battery's gpu_lock.py acquire polls every 60 s for up to 12 h).  If one exists, this job yields the lock at the
    next chunk boundary; if none, it keeps the lock from chunk to chunk (no GPU idle gap between chunks)."""
    return _python_lines(WAITER_PATTERNS)


def wait_idle_once(ctx, timeout_s=3600.0):
    """wait_idle.py (the neighbours' idle waiter) as a POLLED child: exit 0 = the GPU is idle (memory < 3500 MiB, no other GPU job process, 4 polls in a row)."""
    p = subprocess.Popen([sys.executable, WAIT_IDLE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while p.poll() is None:
        if ctx.abort.is_set() or time.time() - t0 > timeout_s:
            p.kill()
            return False
        time.sleep(3)
    return p.returncode == 0


def take_lock(ctx, job):
    """Fail closed.  Returns the lock body once: no chain_active marker, no scene-tap probe process, the lock is ours by O_EXCL, wait_idle said idle, and memory.used is
    still <= 3500 MiB 4 s after taking it.  Another job's lock is NEVER removed, however old it looks (see the comment below)."""
    n, t_start = 0, time.time()
    while True:
        if ctx.abort.is_set():
            raise RuntimeError("aborted while waiting for the GPU lock")
        held = None
        try:
            holder = _read_json_text(LOCK)
            age = time.time() - os.path.getmtime(LOCK)
            held = dict(holder, _age_s=round(age))
            if "pid" in holder and not pid_alive(holder["pid"]) and age > 600:
                # NEVER removed.  gpu_lock.py (the refcv8 chain's tool) writes the JOB's MSYS pid, which tasklist cannot see, so "pid not alive" is not evidence of a stale
                # lock, and removing a live job's lock would preempt it.  A stale lock is reported here every poll; a human / the Master Mind clears it.
                held["_looks_stale"] = "holder pid not visible to tasklist and lock older than 10 min: NOT removed (an MSYS pid is invisible to Windows tools)"
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            held = {"unreadable": True}
        blk = blockers() if held is None and not os.path.exists(MARK) else []
        if held is None and not os.path.exists(MARK) and not blk:
            if wait_idle_once(ctx):
                body = lock_body(job)
                try:
                    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, body.encode("utf-8"))
                    os.close(fd)
                    time.sleep(4)                                   # the neighbour's allocation may land just after it released
                    used = gpu_used_mib()
                    if 0 <= used <= GPU_MAX_USED_MIB:
                        return body
                    release(body)                                   # memory still held by a resident process: give the lock back
                except FileExistsError:
                    pass
        n += 1
        if n % 30 == 1:                                             # every ~5 min
            ctx.set(stage="dump", state=f"waiting {round((time.time() - t_start) / 60)} min for the GPU; lock: {json.dumps(held)[:420]}; "
                                        f"chain_active={os.path.exists(MARK)}; blockers={blk}; gpu memory.used {gpu_used_mib()} MiB (need <= {GPU_MAX_USED_MIB})")
        time.sleep(20 if blk else 10)


def lock_is_ours():
    try:
        return _read_json_text(LOCK).get("pid") == os.getpid()
    except (OSError, ValueError):
        return False


def hold_gpu(ctx, job):
    """Keep the lock across chunks: take it only if it is not already ours."""
    if ctx.lock_body and lock_is_ours():
        return
    ctx.lock_body = take_lock(ctx, job)
    print(f"[{now()}] GPU lock taken ({job}); memory.used {gpu_used_mib()} MiB", flush=True)


def drop_gpu(ctx):
    if ctx.lock_body:
        release(ctx.lock_body)
        ctx.lock_body = None


# ------------------------------------------------------------------------------------------ guards
class _MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_mb():
    """Available physical memory in MiB (GlobalMemoryStatusEx; the landed powershell/CIM probe costs ~1-2 s and a process per call)."""
    try:
        s = _MS()
        s.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return int(s.ullAvailPhys // (1024 * 1024))
    except Exception:  # noqa: BLE001
        return 10 ** 9


def free_gb(path):
    try:
        return shutil.disk_usage(path).free / 2 ** 30
    except OSError:
        return 0.0


def wait_resources(ctx, what, paths):
    """Pause (never fail, never write) while a drive that the job writes to is nearly full or RAM is short.  Visible in STATUS.json."""
    t0, said = time.time(), False
    while not ctx.abort.is_set():
        low = [(p, round(free_gb(p), 2)) for p in paths if free_gb(p) < MIN_FREE_BULK_GB]
        if not low and free_mb() >= MIN_FREE_RAM_MB:
            return
        if not said or time.time() - t0 > 600:
            ctx.set(guard=f"PAUSED before {what}: free GiB {low} (need {MIN_FREE_BULK_GB}) / free RAM {free_mb()} MiB (need {MIN_FREE_RAM_MB})")
            said, t0 = True, time.time()
        time.sleep(30)


# ------------------------------------------------------------------------------------------ preflight
def preflight(ctx):
    """The checkpoint on disk IS the P2 model_final (sha256 from RESULT_P2.md), and the banked csv + token list are the originals.  A wrong file stops the run
    before any GPU time is spent."""
    got = sha256_file(CKPT)
    ok = got == CKPT_SHA256
    st = os.stat(CKPT)
    ctx.set(preflight_ckpt_sha256=got, preflight_ckpt_ok=ok, ckpt_stat=[st.st_size, st.st_mtime_ns])
    if ctx.a.device == "cuda":
        wi = os.path.exists(WAIT_IDLE)
        ctx.set(preflight_wait_idle_present=wi)
        ok = ok and wi
    if ctx.main:
        csv_ok = sha256_file(BANKED_CSV) == BANKED_CSV_SHA256
        tok_ok = sha256_file(TOKENS_FULL) == TOKENS_FULL_SHA256
        full = jload(TOKENS_FULL)
        union = [t for ch in ctx.chunks for t in ctx.tokens_of(ch)["tokens"]]
        cover = bool(len(union) == N_FULL and set(union) == set(full["tokens"]))
        ctx.set(preflight_banked_csv_ok=csv_ok, preflight_tokens_ok=tok_ok, preflight_chunks_cover_navtest=cover, n_chunks=len(ctx.chunks))
        ok = ok and csv_ok and tok_ok and cover
    return ok


# ------------------------------------------------------------------------------------------ runner
def _read_log(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read()


def status_of(txt):
    for line in reversed(txt.splitlines()):
        if line.startswith("{") and '"status"' in line:
            return json.loads(line)
    return None


class RealRunner:
    """The two external processes.  Tests replace this class with a fake that writes synthetic dumps / csvs."""

    def __init__(self, ctx):
        self.ctx = ctx

    def dump(self, ch, out_part, log):
        ctx = self.ctx
        cid = ch["id"]
        cmd = [EC.DRIVERL_PY, "refe_navtest_seam.py", "--ckpt", CKPT, "--frames", FRAMES, "--db-dir", DB_DIR,
               "--tokens", ch["tokens_file"], "--out", ctx.rerun_seam(cid), "--arm", f"REFe_p2full_{ctx.tag}_c{cid:02d}", "--rule", RULE,
               "--dump-proposals", out_part, "--device", ctx.a.device]
        with open(log, "w", encoding="utf-8") as f:
            return subprocess.call(cmd, cwd=EVAL, env=EC.env_driverl(), stdout=f, stderr=subprocess.STDOUT)

    def score(self, ch, k, lp):
        ctx = self.ctx
        EC.run([EC.DRIVERL_PY, "score_navtest_refe.py", "--label", ctx.label(ch["id"], k), "--seam", ctx.seam_k(ch["id"], k),
                "--tokens", ch["tokens_file"], "--out", ctx.score_root],
               EVAL, dict(os.environ, PYTHONIOENCODING="utf-8"), lp)
        return status_of(_read_log(lp))


# ------------------------------------------------------------------------------------- dump + gate
def gate_chunk(ctx, ch, dump_path, report_path=None):
    """FIDELITY GATE (pre-registered in C3, applied per chunk): the dump's pick reproduces P2's BANKED full-navtest plan, max |d pose| <= 1e-3 m, fingerprints equal.
    Also: every asked token present, 64 proposals, rule navsim_v1, repair_last_heading on, the checkpoint path, and the seam's own report (rows == asked, misses 0,
    frame control within the bar OR only on the 4 known logs of the 36 known over-bar tokens)."""
    D = np.load(dump_path)
    tok = [str(t) for t in D["token"]]
    P, pick = D["proposals"], D["pick"]
    N = len(tok)
    toks_doc = ctx.tokens_of(ch)
    bidx, bposes, bfp = ctx.banked()
    g = {"chunk": ch["id"], "n_tokens_asked": len(toks_doc["tokens"]), "n_rows": N, "proposals_shape": list(P.shape), "logits_shape": list(D["logits"].shape),
         "rule_in_dump": str(D["rule"]), "ckpt_in_dump": str(D["ckpt"]), "repair_last_heading_in_dump": bool(D["repair_last_heading"]), "tolerance_m": ctx.pose_tol_m}
    g["all_tokens_present"] = bool(set(tok) == set(toks_doc["tokens"]) and N == len(toks_doc["tokens"]) == len(set(tok)))
    g["tokens_in_banked_seam"] = bool(all(t in bidx for t in tok))
    ok = (g["all_tokens_present"] and g["tokens_in_banked_seam"] and P.shape[1] == M and str(D["rule"]) == RULE
          and bool(D["repair_last_heading"]) and str(D["ckpt"]) == CKPT)
    if ok:
        ban = bposes[[bidx[t] for t in tok]].astype(np.float64)
        sel = P[np.arange(N), pick].astype(np.float64)
        d = np.abs(sel - ban).reshape(N, -1).max(1)
        fp_ok = all(str(D["fingerprint"][i]) == str(bfp[bidx[t]]) for i, t in enumerate(tok))
        g.update(max_abs_pose_diff_m=float(d.max()), n_identical=int((d == 0).sum()), n_diff_gt_1e_4=int((d > 1e-4).sum()),
                 n_diff_gt_tol=int((d > ctx.pose_tol_m).sum()), fingerprints_equal=bool(fp_ok),
                 worst_tokens=[(tok[i], float(d[i])) for i in np.argsort(-d)[:5]], pick_index_distinct=int(len(set(int(x) for x in pick))))
        ok = bool(d.max() <= ctx.pose_tol_m and fp_ok and np.isfinite(P).all() and np.isfinite(D["logits"]).all())
    if report_path and os.path.exists(report_path):
        rep = jload(report_path)
        fc = rep.get("frame_control", {})
        chunk_logs = set(toks_doc["token_log"].values())
        fc_ok = bool(fc.get("max_m", 1e9) <= FRAME_BAR_M or (chunk_logs & ctx.known_fc_logs() and fc.get("max_m", 1e9) <= FRAME_KNOWN_MAX_M))
        g["seam_report"] = {"rows": rep.get("rows"), "misses": rep.get("misses"), "tokens_asked": rep.get("tokens_asked"), "frame_control_max_m": fc.get("max_m"),
                            "frame_control_ok": fc_ok, "frame_control_over_bar_only_on_the_known_4_logs": bool(fc.get("max_m", 0) > FRAME_BAR_M and fc_ok),
                            "seconds": rep.get("seconds")}
        ok = ok and rep.get("rows") == rep.get("tokens_asked") == len(toks_doc["tokens"]) and rep.get("misses") == 0 and fc_ok
    g["PASS"] = bool(ok)
    atomic_json(g, os.path.join(ctx.wd, "state", f"gate_fidelity_c{ch['id']:02d}.json"))
    return g


def write_seams(ctx, ch):
    D = np.load(ctx.dump_path(ch["id"]))
    P = D["proposals"]
    os.makedirs(ctx.seam_root, exist_ok=True)
    for k in range(ctx.M):
        p = ctx.seam_k(ch["id"], k)
        if not os.path.exists(p):
            tmp = p[:-4] + f".tmp{os.getpid()}.npz"
            np.savez(tmp, token=D["token"], fingerprint=D["fingerprint"], poses=P[:, k].astype(np.float32), sampling=D["sampling"],
                     arm=np.array(f"REFe_p2full_{ctx.tag}_c{ch['id']:02d}_p{k:02d}"))
            os.replace(tmp, p)


def chunk_dump_valid(ctx, ch):
    """A chunk is DONE iff its state file says gate PASS and the dump on disk still has the recorded sha256."""
    sp, dp = ctx.state_p(ch["id"]), ctx.dump_path(ch["id"])
    if not (os.path.exists(sp) and os.path.exists(dp)):
        return False
    try:
        st = jload(sp)
        return bool(st.get("gate_PASS") and st.get("dump_sha256") == sha256_file(dp))
    except (OSError, ValueError):
        return False


def ensure_chunk(ctx, ch, attempts=4, more_to_dump=False):
    """dump (GPU, behind the shared lock) -> per-chunk fidelity gate -> seams.  Returns 'ok' | 'dump_failed' | 'gate_failed'.
    The lock is KEPT for the next chunk unless this is the last dump, another job is queued on it (waiters()), or something failed."""
    cid, dp = ch["id"], ctx.dump_path(ch["id"])
    if chunk_dump_valid(ctx, ch):
        write_seams(ctx, ch)
        print(f"[{now()}] chunk {cid}: dump already done + gated", flush=True)
        return "ok"
    log = os.path.join(ctx.wd, "logs", f"dump_c{cid:02d}.log")
    part = dp[:-4] + ".part.npz"
    locked = ctx.a.device == "cuda"
    for attempt in range(1, attempts + 1):
        if ctx.abort.is_set():
            return "dump_failed"
        wait_resources(ctx, f"dump of chunk {cid}", [ctx.bulk, ctx.work])
        try:
            if locked:
                hold_gpu(ctx, f"p2full-{ctx.tag}-c{cid:02d}")
            if os.path.exists(part):
                os.remove(part)
            st = os.stat(CKPT)
            if [st.st_size, st.st_mtime_ns] != ctx.status.get("ckpt_stat"):
                ctx.set(stage="dump", result="FAILED: the checkpoint file changed since the preflight")
                drop_gpu(ctx)
                return "dump_failed"
            ctx.set(stage="dump", state=f"chunk {cid}/{len(ctx.chunks) - 1}: dumping {ch['n']} tokens (attempt {attempt})", chunk_started=now())
            t0 = time.time()
            rc = ctx.runner.dump(ch, part, log)
            ctx.set(stage="dump", state=f"chunk {cid}: seam rc={rc} (attempt {attempt})", last_dump_seconds=round(time.time() - t0))
        except BaseException:
            drop_gpu(ctx)
            raise
        if locked and (not os.path.exists(part) or not more_to_dump or waiters()):
            drop_gpu(ctx)                               # failed attempt / last dump / a neighbour is queued: give the GPU back
        if os.path.exists(part):
            try:
                np.load(part)["proposals"].shape
            except Exception:  # noqa: BLE001  (a truncated zip: the child died mid-write)
                os.remove(part)
        if os.path.exists(part):
            os.replace(part, dp)
            rpt = os.path.splitext(ctx.rerun_seam(cid))[0] + ".report.json"
            g = gate_chunk(ctx, ch, dp, rpt if os.path.exists(rpt) else None)
            ctx.set(stage="gate", **{f"gate_c{cid:02d}": {k: g.get(k) for k in ("PASS", "max_abs_pose_diff_m", "n_identical", "n_diff_gt_tol", "fingerprints_equal")}})
            if not g["PASS"]:
                drop_gpu(ctx)
                return "gate_failed"
            atomic_json({"chunk": cid, "n": ch["n"], "gate_PASS": True, "dump_sha256": sha256_file(dp), "dump_bytes": os.path.getsize(dp), "when": now(),
                         "gate": g}, ctx.state_p(cid))
            write_seams(ctx, ch)
            ctx.progress(chunks_dumped=ctx.counts["chunks_dumped"] + 1)
            return "ok"
        time.sleep(30 if locked else 1)                 # e.g. a CUDA OOM against a neighbour: wait for a clean window and retry
    return "dump_failed"


# -------------------------------------------------------------------------------------------- score
def score_task(ctx, ch, k):
    cid, N = ch["id"], ch["n"]
    lp = os.path.join(ctx.wd, "logs", f"score_c{cid:02d}_p{k:02d}.log")
    st, att, ram_aborts = None, 0, 0
    while att < 3:
        if ctx.abort.is_set():
            return (cid, k), {"status": "ABORTED"}
        if os.path.exists(ctx.csv_k(cid, k)) and os.path.exists(lp):
            st = status_of(_read_log(lp))
            if st and st.get("status") == "PASS" and st.get("csv_valid_rows") == N:
                return (cid, k), st
        with ctx.start_lk:                      # one start at a time, >= STAGGER_S apart, each behind the RAM / disk guard
            wait_resources(ctx, f"harness column c{cid:02d}_p{k:02d}", [ctx.work, "D:/Archive"])
            dt = time.time() - ctx.last_start
            if dt < (STAGGER_S if ctx.main else 0.05):
                time.sleep((STAGGER_S if ctx.main else 0.05) - dt)
                wait_resources(ctx, f"harness column c{cid:02d}_p{k:02d}", [ctx.work, "D:/Archive"])
            ctx.last_start = time.time()
        st = ctx.runner.score(ch, k, lp)
        if st and st.get("status") == "PASS" and st.get("csv_valid_rows") == N:
            return (cid, k), st
        if "W3_RAM_GUARD_ABORT" in _read_log(lp) and ram_aborts < 30:
            ram_aborts += 1                      # W3's wrapper aborted on low system RAM (a neighbour's spike): not this column's fault, not an attempt
            with ctx._lk:
                ctx.counts["score_ram_guard_retries"] += 1
            time.sleep(120 if ctx.main else 0.2)
            continue
        att += 1
        time.sleep(90 if ctx.main else 0.2)
    return (cid, k), st


def score_and_record(ctx, ch, k):
    key, st = score_task(ctx, ch, k)
    _record_score(ctx, key, st, ch["n"])
    return key, st


def _record_score(ctx, key, st, N):
    ok = bool(st and st.get("status") == "PASS" and st.get("csv_valid_rows") == N)
    with ctx._lk:
        if ok:
            ctx.counts["score_pass"] += 1
        elif not (st and st.get("status") == "ABORTED"):
            ctx.counts["score_fail"] += 1
            ctx.status.setdefault("score_failed_cols", []).append(list(key))
            if ctx.counts["score_fail"] >= 8:
                ctx.abort_reason = ctx.abort_reason or "circuit breaker: 8 harness columns failed"
                ctx.abort.set()
    ctx.progress()
    return ok


def run_pipeline(ctx):
    """preflight -> (resumed chunks first) -> per chunk: dump, gate, seams, 64 harness columns on the worker pool while the next chunk dumps -> rc."""
    if not preflight(ctx):
        ctx.set(stage="preflight", result="FAILED: the checkpoint / banked csv / token list / chunk plan is not what it must be")
        return 1
    ctx.t_score0 = time.time()
    ex = ThreadPoolExecutor(max_workers=max(1, ctx.a.workers))
    futs = []

    def submit(ch):
        for k in range(ctx.M):
            futs.append((ch, ex.submit(score_and_record, ctx, ch, k)))

    rc = 0
    try:
        done_first = [ch for ch in ctx.chunks if chunk_dump_valid(ctx, ch)]
        for ch in done_first:                           # resumed work: score it while the GPU queue is being waited on
            if ensure_chunk(ctx, ch) == "ok":
                ctx.progress(chunks_dumped=len([c for c in ctx.chunks if chunk_dump_valid(ctx, c)]))
                submit(ch)
        todo = [ch for ch in ctx.chunks if ch not in done_first]
        for i, ch in enumerate(todo):
            if ctx.abort.is_set():
                break
            r = ensure_chunk(ctx, ch, more_to_dump=(i < len(todo) - 1))
            if r == "gate_failed":
                ctx.abort_reason = f"FIDELITY gate failed on chunk {ch['id']}: P2's pick is not reproduced -- nothing from this chunk was scored"
                ctx.abort.set()
                rc = 2
                break
            if r != "ok":
                ctx.abort_reason = f"dump of chunk {ch['id']} failed after the retries (see logs/dump_c{ch['id']:02d}.log)"
                ctx.abort.set()
                rc = 1
                break
            submit(ch)
        if ctx.abort.is_set():
            ex.shutdown(wait=False, cancel_futures=True)
        for ch, f in futs:
            if f.cancelled():
                continue
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                ctx.abort_reason = ctx.abort_reason or f"a scoring worker raised {e!r}"
                ctx.abort.set()
                with ctx._lk:
                    ctx.counts["score_fail"] += 1
    finally:
        drop_gpu(ctx)
        ex.shutdown(wait=True, cancel_futures=True)
    if rc == 0 and ctx.abort.is_set():
        rc = 3
    if rc == 0 and ctx.counts["score_fail"]:
        rc = 3
    if rc == 0 and ctx.counts["score_pass"] != ctx.counts["score_total"]:
        rc = 3
    ctx.set(stage="score", pipeline_rc=rc, abort_reason=ctx.abort_reason, **{"score_all_pass": rc == 0})
    return rc


# -------------------------------------------------------------------------------------------- table
def read_csv(p):
    rows = {}
    with open(p, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["token"] == "average":
                continue
            rows[r["token"]] = (r["valid"] == "True", float(r["score"]), tuple(float(r[c]) for c in SUB))
    return rows


def stage_table(ctx, out_npz):
    """PDMS + the six sub-scores per (token, proposal) from the 64 x n_chunks harness csvs -> one table.npz, rows in SORTED token order (the landed analysis's
    canonical order).  G3: every cell valid + finite.  G2: table-at-pick == the banked csv (needs all 64 columns)."""
    Ds = []
    for ch in ctx.chunks:
        D = np.load(ctx.dump_path(ch["id"]))
        Ds.append(D)
    tok_all = [str(t) for D in Ds for t in D["token"]]
    N = len(tok_all)
    assert len(set(tok_all)) == N, "a token is in two chunks"
    off = np.cumsum([0] + [len(D["token"]) for D in Ds])
    Mx = ctx.M
    pdms = np.full((N, Mx), np.nan)
    sub = np.full((N, Mx, 6), np.nan)
    valid = np.zeros((N, Mx), dtype=bool)
    for ci, ch in enumerate(ctx.chunks):
        toks = [str(t) for t in Ds[ci]["token"]]
        for k in range(Mx):
            rows = read_csv(ctx.csv_k(ch["id"], k))
            for j, t in enumerate(toks):
                v, s, c = rows[t]
                i = off[ci] + j
                valid[i, k], pdms[i, k], sub[i, k] = v, s, c
    cat = lambda key: np.concatenate([D[key] for D in Ds], axis=0)  # noqa: E731
    P, LG, PK, FP = cat("proposals"), cat("logits"), cat("pick"), cat("fingerprint")
    tl = [ctx.tokens_of(ch)["token_log"] for ch in ctx.chunks]
    LOG = np.array([tl[ci][str(t)] for ci, D in enumerate(Ds) for t in D["token"]])
    CID = np.concatenate([np.full(len(D["token"]), ch["id"]) for ch, D in zip(ctx.chunks, Ds)])
    order = np.argsort(np.array(tok_all))                                    # sorted token string order
    tok = np.array(tok_all)[order]
    pdms, sub, valid, P, LG, PK, FP, LOG, CID = (x[order] for x in (pdms, sub, valid, P, LG, PK, FP, LOG, CID))
    g = {"G3_all_valid": bool(valid.all()), "G3_finite": bool(np.isfinite(pdms).all() and np.isfinite(sub).all()), "n_tokens": int(N), "n_columns": int(Mx)}
    bidx, bposes, _ = ctx.banked()
    lr = read_csv(BANKED_CSV)
    ban = bposes[[bidx[str(t)] for t in tok]].astype(np.float64)
    dpos = np.abs(P[np.arange(N), PK].astype(np.float64) - ban).reshape(N, -1).max(1)
    csv_score = np.array([lr[str(t)][1] for t in tok])
    if Mx == M:
        at_pick = pdms[np.arange(N), PK]
        dd = np.abs(at_pick - csv_score)
        same = (dpos == 0) if ctx.main else (dpos <= ctx.pose_tol_m)       # main: bit-identical poses only; smoke: CPU-vs-GPU noise allowed
        g.update(G2_banked_csv=BANKED_CSV, G2_tokens_identical_poses=int(same.sum()),
                 G2_max_abs_score_diff_identical_poses=float(dd[same].max()) if same.any() else None,
                 G2_max_abs_score_diff_all=float(dd.max()), G2_n_score_diff_gt_1e_9=int((dd > 1e-9).sum()),
                 G2_PASS=bool(same.any() and dd[same].max() <= 1e-9))
    else:
        g["G2_PASS"] = True
        g["G2_note"] = f"SMOKE with {Mx} columns: G2 not evaluated"
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    tmp = out_npz[:-4] + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, token=tok, log=LOG, chunk=CID, fingerprint=FP, pdms=pdms, sub=sub, valid=valid, logits=LG, proposals=P, pick=PK,
                        sub_names=np.array(SUB), pdms_at_pick_banked=csv_score)
    os.replace(tmp, out_npz)
    atomic_json(g, os.path.join(ctx.wd, "gate_table.json"))
    ctx.set(stage="table", table_gates=g, table_npz=out_npz)
    return bool(g["G3_all_valid"] and g["G3_finite"] and g["G2_PASS"])


# --------------------------------------------------------------------------------------------- main
def make_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("all", "table"), default="all")
    ap.add_argument("--tag", default="main", help="main = the 12,146-token job; any other tag is a plumbing smoke on --tokens")
    ap.add_argument("--tokens", default=None, help="smoke only: a W3-format token file (chunked at --chunk-tokens)")
    ap.add_argument("--chunk-tokens", type=int, default=MAX_CHUNK_TOKENS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--m", type=int, default=M, help="number of proposal columns (64; smaller ONLY for the plumbing smoke)")
    ap.add_argument("--pose-tol-m", type=float, default=None, help="smoke only (CPU-vs-GPU float noise); the main job is pinned to 1e-3 m")
    return ap


def main(argv=None):
    a = make_parser().parse_args(argv)
    ctx = Ctx(a)
    ctx.set(tag=a.tag, n_tokens=sum(c["n"] for c in ctx.chunks), n_chunks=len(ctx.chunks), pid=os.getpid(), workers=a.workers,
            started=ctx.status.get("started", now()), resumed_at=now())
    if a.stage == "all":
        rc = run_pipeline(ctx)
        if rc:
            ctx.set(result=f"FAILED rc={rc}: {ctx.abort_reason}")
            return rc
    out = f"{ctx.bulk}/table_p2_full.npz"
    if not stage_table(ctx, out):
        ctx.set(stage="table", result="INSTRUMENT FAILURE: table gates failed (gate_table.json)")
        return 4
    ctx.set(result=f"stage {a.stage} done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
