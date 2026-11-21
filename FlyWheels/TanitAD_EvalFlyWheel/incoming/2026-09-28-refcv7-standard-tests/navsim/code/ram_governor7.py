#!/usr/bin/env python3
"""RAM governor + scoring-slot lock for the refcv7 NavSim SCORER DRIVERS (ANY python >= 3.9; needs psutil).

    import ram_governor7 as RG
    with RG.governed(label, out_dir, est_rss_mb=2300) as gov:       # blocks for a slot + a RAM window
        subprocess.run(<the official scorer>)                      # its descendants are watched
    gov.summary()                                                  # n_pauses, paused_s, min_avail_mb, ...

WHY THIS EXISTS (MEASURED 2026-10-04, refcv7 step 50,400, see COMMS / the EvalFlyWheel report):
the official scorers were killed 17 times by THEIR OWN RAM GUARD (``E1/W3_RAM_GUARD_ABORT``) although
they themselves held only 0.84 GB (navhard) / 2.22 GB (navtest) -- the box (31.8 GB, shared with a
battery chain, two pytest suites and the D6 scorer) swung between 1.0 and 10.4 GB available inside
90 s. The guard's HARD floor is a SINGLE 2 s sample below 2,000 MB, so a scorer launched in a
>= 8 GB window died minutes later, and the hours it had run were lost (navhard PRIOR_ha0p: 95 % done).

WHAT IT CHANGES, AND WHAT IT DOES NOT. The guard (soft floor 3,000 MB x 60 samples, hard floor
2,000 MB x 1 sample) is NOT lowered, not bypassed and not patched -- it stays as the last resort. The
governor sits one level above it: when system-available memory falls under ``pause_mb`` (3,500 MB,
1.5 GB ABOVE the hard floor) it SUSPENDS the scorer's process tree (psutil -> NtSuspendProcess) and
trims its working set, so the scorer (a) cannot allocate, (b) gives its pages back to the box
(standby = available), and (c) is never sampled by its guard at a bad moment. It resumes only when the
box could absorb the scorer's pages again (``pause_mb + headroom_mb + <pages to fault back>``) for
``resume_ok_s`` seconds. The box is therefore protected AT LEAST as well as before (an aborted scorer
frees its RSS; a paused-and-trimmed one frees nearly all of it and keeps its progress).

It only ever touches DESCENDANTS OF THE DRIVER THAT RUNS IT (the scorer processes that driver itself
started) -- never the driver, never an ancestor, never a sibling, never another job.

SLOTS: at most ``n_slots`` (2) scorers run at once across ALL drivers (file locks, stale-safe); the
SECOND slot opens only while >= ``extra_slot_min_mb`` (8,000 MB) is available. A driver waiting for a
slot / a RAM window holds nothing but a ~40 MB python.
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import threading
import time

import psutil

#: ----- the policy (the numbers are the argument; see the docstring) -----------------------------
PAUSE_MB = 3500.0            # suspend below this (guard: soft 3,000 x 60 samples, hard 2,000 x 1)
HEADROOM_MB = 500.0          # resume only with this much spare ABOVE pause_mb after pages return
SAMPLE_S = 0.5               # running: probe twice a second (MEASURED max 1 s drop: 780 MB)
PAUSED_SAMPLE_S = 2.0
RESUME_OK_S = 30.0           # the resume condition must hold this long (+ jitter vs sibling scorers)
MAX_PAUSE_S = 3 * 3600.0     # after this, resume anyway and let the scorer's own guard decide
N_SLOTS = 2
EXTRA_SLOT_MIN_MB = 8000.0
START_GATE_SAMPLES = 3
START_GATE_EVERY_S = 5.0
COMMIT_GATE_MB = 3000.0      # free COMMIT (not RAM) needed to start a scorer
SLOT_DIR = os.environ.get("R7_SCORE_SLOT_DIR", "C:/Users/Admin/qland/work/refcv7/navsim_score_slots")


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def avail_mb() -> float:
    return psutil.virtual_memory().available / 2 ** 20


def commit_free_mb() -> float:
    """System-wide FREE COMMIT (``ullAvailPageFile``), the number a new process's private bytes are
    charged against. ``+inf`` where it cannot be read (non-Windows / ctypes failure): an unreadable
    gauge never BLOCKS a start, the RAM gate still applies."""
    if os.name != "nt":
        return float("inf")
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = _MS()
        ms.dwLength = ctypes.sizeof(_MS)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return float("inf")
        return ms.ullAvailPageFile / 2 ** 20
    except Exception:                                                     # noqa: BLE001
        return float("inf")


def trim_working_set(pid: int) -> bool:
    """``EmptyWorkingSet`` on ``pid``: its resident pages go to the standby / modified lists, which
    count as AVAILABLE. Best effort (False on any failure) -- never required for correctness."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = ctypes.c_void_p
        h = k32.OpenProcess(0x0100 | 0x0400, False, int(pid))      # SET_QUOTA | QUERY_INFORMATION
        if not h:
            return False
        try:
            return bool(ctypes.windll.psapi.EmptyWorkingSet(ctypes.c_void_p(h)))
        finally:
            k32.CloseHandle(ctypes.c_void_p(h))
    except Exception:                                                     # noqa: BLE001
        return False


def descendants_of_self() -> list:
    """The processes THIS process started (recursively). The only default target set."""
    me = psutil.Process()
    try:
        return me.children(recursive=True)
    except psutil.Error:
        return []


# --------------------------------------------------------------------------- #
# the governor                                                                 #
# --------------------------------------------------------------------------- #
class Governor(threading.Thread):
    """Pause / resume a process set on system-available memory. All inputs injectable (tests)."""

    def __init__(self, label: str, log_path: str | None = None, *, pause_mb: float = PAUSE_MB,
                 headroom_mb: float = HEADROOM_MB, sample_s: float = SAMPLE_S,
                 paused_sample_s: float = PAUSED_SAMPLE_S, resume_ok_s: float = RESUME_OK_S,
                 max_pause_s: float = MAX_PAUSE_S, probe=avail_mb, procs=descendants_of_self,
                 trim: bool = True, jitter_s: float | None = None, state_path: str | None = None,
                 lookup=psutil.Process):
        super().__init__(daemon=True, name=f"ram-governor:{label}")
        self.label, self.log_path, self.state_path = label, log_path, state_path
        self.pause_mb, self.headroom_mb = float(pause_mb), float(headroom_mb)
        self.sample_s, self.paused_sample_s = float(sample_s), float(paused_sample_s)
        self.resume_ok_s, self.max_pause_s = float(resume_ok_s), float(max_pause_s)
        self.probe, self.procs, self.trim, self.lookup = probe, procs, bool(trim), lookup
        # decorrelates two governed scorers so they do not fault their pages back in the same second
        self.jitter_s = random.uniform(0.0, 8.0) if jitter_s is None else float(jitter_s)
        self._halt = threading.Event()
        self.paused = False
        self._held: list = []                  # [(pid, create_time)] suspended by US
        self._rss_before_mb = 0.0
        self._paused_at = 0.0
        self._ok_since: float | None = None
        self.n_pauses = 0
        self.n_forced_resumes = 0
        self.paused_s = 0.0
        self.min_avail_mb = float("inf")
        self.n_samples = 0
        self.errors: list = []

    # ---- logging --------------------------------------------------------- #
    def _log(self, event: str, **kw) -> None:
        rec = {"utc": now_utc(), "label": self.label, "event": event}
        rec.update(kw)
        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, default=str) + "\n")
            except OSError:
                pass

    def _write_state(self) -> None:
        """The orphan record: if THIS process dies while its scorer is suspended, the next governed
        driver (or ``--resume-orphans``) resumes exactly these (pid, create_time) pairs."""
        if not self.state_path:
            return
        try:
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump({"owner_pid": os.getpid(), "owner_create_time": psutil.Process().create_time(),
                           "label": self.label, "held": self._held, "utc": now_utc()}, fh)
        except OSError:
            pass

    # ---- the two transitions --------------------------------------------- #
    def _targets(self) -> list:
        me = os.getpid()
        try:
            anc = {p.pid for p in psutil.Process().parents()}
        except psutil.Error:
            anc = set()
        return [p for p in self.procs() if p.pid != me and p.pid not in anc]

    def _pause(self, avail: float) -> None:
        targets = self._targets()
        if not targets:                      # the scorer is not up yet / already gone: nothing to hold
            return
        held, rss_mb = [], 0.0
        for p in targets:
            try:
                ct = p.create_time()
                rss_mb += p.memory_info().rss / 2 ** 20
                p.suspend()
                held.append((p.pid, ct))
            except psutil.Error as e:
                self.errors.append(f"suspend {p.pid}: {type(e).__name__}")
        self._held = held
        self._rss_before_mb = rss_mb
        self._paused_at = time.time()
        self._ok_since = None
        self.paused = True
        self.n_pauses += 1
        self._write_state()
        trimmed = []
        if self.trim:
            for pid, _ in held:
                trimmed.append(trim_working_set(pid))
        self._log("PAUSE", avail_mb=round(avail), pids=[h[0] for h in held],
                  rss_before_mb=round(rss_mb), trimmed=trimmed, pause_mb=self.pause_mb)

    def _resident_mb(self) -> float:
        tot = 0.0
        for pid, ct in self._held:
            try:
                p = self.lookup(pid)
                if abs(p.create_time() - ct) < 1.0:
                    tot += p.memory_info().rss / 2 ** 20
            except psutil.Error:
                pass
        return tot

    def _resume(self, avail: float, forced: bool = False) -> None:
        resumed = []
        for pid, ct in self._held:
            try:
                p = self.lookup(pid)
                if abs(p.create_time() - ct) < 1.0:       # never resume a recycled pid
                    p.resume()
                    resumed.append(pid)
            except psutil.Error as e:
                self.errors.append(f"resume {pid}: {type(e).__name__}")
        dt = time.time() - self._paused_at
        self.paused_s += dt
        self.paused = False
        self._held = []
        self._ok_since = None
        if forced:
            self.n_forced_resumes += 1
        self._write_state()
        self._log("RESUME" if not forced else "RESUME_FORCED_MAX_PAUSE", avail_mb=round(avail),
                  pids=resumed, paused_s=round(dt, 1))

    # ---- the loop -------------------------------------------------------- #
    def step(self) -> None:
        """One decision (public so the tests drive it without a thread)."""
        a = float(self.probe())
        self.n_samples += 1
        self.min_avail_mb = min(self.min_avail_mb, a)
        if not self.paused:
            if a < self.pause_mb:
                self._pause(a)
            return
        t = time.time()
        # pages that must be faulted back before the scorer is whole again
        owed = max(0.0, self._rss_before_mb - self._resident_mb())
        need = self.pause_mb + self.headroom_mb + owed
        if a >= need:
            if self._ok_since is None:
                self._ok_since = t
            if t - self._ok_since >= self.resume_ok_s + self.jitter_s:
                self._resume(a)
                return
        else:
            self._ok_since = None
        if t - self._paused_at >= self.max_pause_s:
            self._resume(a, forced=True)

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                self.step()
            except Exception as e:                                          # noqa: BLE001
                self.errors.append(f"step: {type(e).__name__}: {e}")
                if self.paused and len(self.errors) > 50:                   # never strand a pause
                    self._resume(0.0, forced=True)
            self._halt.wait(self.paused_sample_s if self.paused else self.sample_s)

    def stop(self) -> None:
        self._halt.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=10)
        if self.paused:
            self._resume(float(self.probe()), forced=True)
            self.n_forced_resumes -= 1                                      # a normal stop, not a timeout
        if self.state_path:
            try:
                os.remove(self.state_path)
            except OSError:
                pass

    def summary(self) -> dict:
        return {"label": self.label, "pause_mb": self.pause_mb, "headroom_mb": self.headroom_mb,
                "resume_ok_s": self.resume_ok_s, "n_pauses": self.n_pauses,
                "n_forced_resumes": self.n_forced_resumes, "paused_s_total": round(self.paused_s, 1),
                "min_avail_mb_seen": (None if self.min_avail_mb == float("inf")
                                      else round(self.min_avail_mb)),
                "n_samples": self.n_samples, "errors": self.errors[:10]}


def resume_orphans(state_dir: str, log=lambda m: None) -> list:
    """Resume processes a DEAD governed driver left suspended (state files whose owner is gone).
    Only pids the state file names, only while their create_time still matches."""
    done = []
    try:
        names = [n for n in os.listdir(state_dir) if n.endswith(".suspended.json")]
    except OSError:
        return done
    for n in names:
        p = os.path.join(state_dir, n)
        try:
            st = json.load(open(p, encoding="utf-8"))
            owner = psutil.Process(int(st["owner_pid"]))
            if abs(owner.create_time() - float(st["owner_create_time"])) < 1.0:
                continue                                             # the owner is alive: its business
        except psutil.NoSuchProcess:
            pass
        except Exception:                                            # noqa: BLE001
            continue
        for pid, ct in st.get("held", []):
            try:
                q = psutil.Process(int(pid))
                if abs(q.create_time() - float(ct)) < 1.0:
                    q.resume()
                    done.append(int(pid))
                    log(f"[ram_governor] resumed orphan {pid} of dead driver {st.get('owner_pid')}")
            except psutil.Error:
                pass
        try:
            os.remove(p)
        except OSError:
            pass
    return done


# --------------------------------------------------------------------------- #
# slots                                                                        #
# --------------------------------------------------------------------------- #
def _slot_alive(path: str) -> bool:
    try:
        rec = json.load(open(path, encoding="utf-8"))
        p = psutil.Process(int(rec["pid"]))
        return abs(p.create_time() - float(rec["create_time"])) < 1.0
    except psutil.NoSuchProcess:
        return False
    except (OSError, ValueError, KeyError):
        # unreadable / half-written: a very young file is a writer in flight, an old one is debris
        try:
            return time.time() - os.path.getmtime(path) < 20.0
        except OSError:
            return False


class SlotLock:
    def __init__(self, directory: str | None = None, n_slots: int = N_SLOTS,
                 extra_slot_min_mb: float = EXTRA_SLOT_MIN_MB, probe=avail_mb):
        self.dir = directory or SLOT_DIR              # resolved at call time (tests re-point SLOT_DIR)
        self.n, self.extra_min, self.probe = int(n_slots), extra_slot_min_mb, probe
        self.held: str | None = None
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, i: int) -> str:
        return os.path.join(self.dir, f"slot{i}.lock")

    def try_take(self, label: str) -> bool:
        for i in range(self.n):
            p = self._path(i)
            if os.path.exists(p) and not _slot_alive(p):
                try:
                    os.remove(p)                                         # a dead holder's debris
                except OSError:
                    pass
            if i >= 1 and self.probe() < self.extra_min:
                return False                                             # extra slots need a quiet box
            try:
                fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            except OSError:
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"pid": os.getpid(), "create_time": psutil.Process().create_time(),
                           "label": label, "utc": now_utc()}, fh)
            self.held = p
            return True
        return False

    def release(self) -> None:
        if self.held:
            try:
                rec = json.load(open(self.held, encoding="utf-8"))
                if int(rec.get("pid", -1)) == os.getpid():
                    os.remove(self.held)
            except (OSError, ValueError):
                pass
            self.held = None


def wait_for_start(label: str, est_rss_mb: float, *, pause_mb: float = PAUSE_MB,
                   headroom_mb: float = HEADROOM_MB, slots: SlotLock | None = None, poll_s: float = 10.0,
                   probe=avail_mb, commit_probe=commit_free_mb, commit_gate_mb: float = COMMIT_GATE_MB,
                   max_wait_s: float | None = None, log=print, sleep=time.sleep) -> SlotLock:
    """Block until (a) RAM >= pause_mb + est_rss_mb + headroom_mb on ``START_GATE_SAMPLES`` consecutive
    samples, (b) free commit >= commit_gate_mb, and (c) a slot is free. Returns the HELD slot."""
    slots = slots or SlotLock(probe=probe)
    need = pause_mb + est_rss_mb + headroom_mb
    t0, ok, last_msg = time.time(), 0, 0.0
    while True:
        a, c = probe(), commit_probe()
        ok = ok + 1 if (a >= need and c >= commit_gate_mb) else 0
        if ok >= START_GATE_SAMPLES and slots.try_take(label):
            log(f"[ram_governor] {label}: slot taken after {time.time() - t0:.0f} s "
                f"(avail {a:.0f} MB >= {need:.0f}; free commit {c:.0f} MB)")
            return slots
        if ok >= START_GATE_SAMPLES:
            ok = START_GATE_SAMPLES - 1                                   # window open, slot busy: keep watching
        if max_wait_s is not None and time.time() - t0 > max_wait_s:
            raise TimeoutError(f"{label}: no slot / RAM window within {max_wait_s:.0f} s")
        if time.time() - last_msg > 600:
            log(f"[ram_governor] {label}: waiting (avail {a:.0f} MB, need {need:.0f}; "
                f"free commit {c:.0f} MB, need {commit_gate_mb:.0f}; waited {time.time() - t0:.0f} s)")
            last_msg = time.time()
        sleep(START_GATE_EVERY_S if ok else poll_s)


@contextlib.contextmanager
def governed(label: str, out_dir: str, *, est_rss_mb: float, enabled: bool | None = None,
             log=print, start_kw: dict | None = None, **gov_kw):
    """Slot + RAM window + a running Governor around the caller's scorer subprocess(es).
    ``R7_GOVERNOR=0`` (or ``enabled=False``) makes it a no-op that still yields an inert object."""
    if enabled is None:
        enabled = os.environ.get("R7_GOVERNOR", "1") != "0"
    if not enabled:
        class _Off:
            @staticmethod
            def summary():
                return {"label": label, "governor": "DISABLED"}
        yield _Off()
        return
    os.makedirs(out_dir, exist_ok=True)
    resume_orphans(SLOT_DIR, log=log)
    slots = wait_for_start(label, est_rss_mb, log=log, **(start_kw or {}))
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)
    gov = Governor(label, os.path.join(out_dir, f"{safe}.governor.jsonl"),
                   state_path=os.path.join(SLOT_DIR, f"{safe}.{os.getpid()}.suspended.json"), **gov_kw)
    gov._log("START", est_rss_mb=est_rss_mb, pause_mb=gov.pause_mb, headroom_mb=gov.headroom_mb,
             avail_mb=round(avail_mb()), free_commit_mb=round(commit_free_mb()))
    gov.start()
    try:
        yield gov
    finally:
        gov.stop()
        gov._log("END", **gov.summary())
        slots.release()


class JobLock:
    """ONE live driver per scorer JOB (``job_<key>.lock`` beside the slot files; stale-safe like the slots).
    MEASURED need (2026-10-04): the step-50,400 runner re-launches a queued job without asking whether another
    driver (e.g. the governed ``score_queue_gov7.py``) is already scoring it; two scorers writing one label's log /
    CSV / call log corrupt each other. A duplicate WAITS here, and when the first holder is done the caller re-reads
    the PASS guard (``already_scored``) -- PASS: skip; FAIL: this duplicate becomes the retry."""

    def __init__(self, key: str, directory: str | None = None):
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in key)
        self.dir = directory or SLOT_DIR
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, f"job_{safe}.lock")
        self.held = False

    def try_acquire(self) -> bool:
        if os.path.exists(self.path) and not _slot_alive(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"pid": os.getpid(), "create_time": psutil.Process().create_time(), "utc": now_utc()}, fh)
        self.held = True
        return True

    def acquire(self, poll_s: float = 10.0, max_wait_s: float | None = None, log=print, sleep=time.sleep) -> None:
        t0, said = time.time(), False
        while not self.try_acquire():
            if not said:
                log(f"[ram_governor] another driver is scoring this very job ({os.path.basename(self.path)}); waiting")
                said = True
            if max_wait_s is not None and time.time() - t0 > max_wait_s:
                raise TimeoutError(f"job lock {self.path} not free within {max_wait_s:.0f} s")
            sleep(poll_s)

    def release(self) -> None:
        if self.held:
            try:
                rec = json.load(open(self.path, encoding="utf-8"))
                if int(rec.get("pid", -1)) == os.getpid():
                    os.remove(self.path)
            except (OSError, ValueError):
                pass
            self.held = False


# --------------------------------------------------------------------------- #
# idempotence: a duplicate launch must never rescore (and so never overwrite) a banked PASS                    #
# --------------------------------------------------------------------------- #
def already_scored(counts_path: str, seam_path: str, csv_path: str, *, seam_sha: str | None = None) -> str | None:
    """-> a reason string when THIS scorer run is already banked as PASS (the caller then exits 0 without touching
    anything), else None. MEASURED need (2026-10-04): the step-50,400 runner re-launches a queued job without
    re-reading its counts, and a second scorer started beside a PASS would truncate its log / CSV / call log and
    could leave a FAIL guard where a PASS stood. ``R7_FORCE_RESCORE=1`` disables the guard.
    PASS must be beside: the CSV (unchanged: sha256 when the counts record one), and the SAME seam -- by
    ``seam_sha256`` when the counts record it (score_arm7), else by the ``arm`` field naming the seam path and
    the counts being NEWER than the seam file (score_navtest7 / W3: a re-bridged seam is newer)."""
    if os.environ.get("R7_FORCE_RESCORE") == "1":
        return None
    try:
        c = json.load(open(counts_path, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if c.get("status") != "PASS" or not (csv_path and os.path.isfile(csv_path)):
        return None
    if seam_sha is not None:
        if c.get("seam_sha256") != seam_sha:
            return None
    else:
        norm = lambda x: str(x).replace("\\", "/").lower()                   # noqa: E731
        if norm(c.get("arm", "")) != norm("SEAM:" + os.path.abspath(seam_path)):
            return None
        try:
            if os.path.getmtime(counts_path) < os.path.getmtime(seam_path):
                return None
        except OSError:
            return None
    if c.get("csv_sha256"):
        import hashlib
        h = hashlib.sha256()
        with open(csv_path, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b)
        if h.hexdigest() != c["csv_sha256"]:
            return None
    return f"counts PASS beside an unchanged CSV and the same seam ({counts_path})"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ram_governor7 maintenance")
    ap.add_argument("--resume-orphans", action="store_true",
                    help="resume scorer processes left suspended by a DEAD governed driver")
    ap.add_argument("--status", action="store_true", help="print slot files + system memory")
    a = ap.parse_args()
    if a.resume_orphans:
        print(json.dumps({"resumed": resume_orphans(SLOT_DIR, log=print)}))
    if a.status or not a.resume_orphans:
        print(json.dumps({"avail_mb": round(avail_mb()), "free_commit_mb": round(commit_free_mb()),
                          "slots": {n: (open(os.path.join(SLOT_DIR, n), encoding="utf-8").read()
                                        if os.path.isfile(os.path.join(SLOT_DIR, n)) else None)
                                    for n in (sorted(os.listdir(SLOT_DIR)) if os.path.isdir(SLOT_DIR)
                                              else [])}}, indent=1))
